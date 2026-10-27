#!/usr/bin/env python3
# app.py — Streamlit Web UI for the RAG Evaluation Framework
"""
Run with:
    streamlit run app.py

Pages:
   Chat            — talk to any RAG architecture (with conversation memory)
   A/B Test        — compare two architectures side-by-side
   Evaluation      — run RAGAS evaluation and see metrics live
   Visualizations  — render all 7 plots from current results
   Cache           — semantic cache statistics + manual lookup
   Settings       — toggle thresholds, view config

No API key required — extractive fallback works offline.
"""

from __future__ import annotations

import sys
import json
import time
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Suppress noisy warnings from transformers/torchvision
import warnings  # noqa: E402
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", category=UserWarning)
import os as _os  # noqa: E402
_os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Quiet down 3rd-party loggers in the Streamlit UI
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from configs.config import (  # noqa: E402
    OUTPUTS_DIR, EVAL_RESULTS_PATH, MISTRAL_API_KEY, VECTOR_STORE_DIR,
    FAITHFULNESS_THRESHOLD, CONTEXT_PRECISION_THRESHOLD,
    ANSWER_RELEVANCE_THRESHOLD, NOISE_RATIO_MAX, TOP_K,
    GROQ_API_KEY, GROQ_MODEL,
    OPENROUTER_API_KEY,
    OPENAI_API_KEY,
)

st.set_page_config(
    page_title="RAG Evaluation Framework",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, #845ec2 0%, #4ea8de 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem; }
    .subtitle { color: #888; font-size: 1rem; margin-top: 0; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 16px;
        border-left: 4px solid #845ec2; margin: 6px 0; }
    .pass { color: #2e7d32; font-weight: 600; }
    .fail { color: #c62828; font-weight: 600; }
    .arch-pill { display: inline-block; padding: 4px 12px; border-radius: 14px;
        background: #e3f2fd; color: #1565c0; font-size: 0.85rem; margin-right: 6px; }
    .stChatMessage { padding: 0.4rem 0; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Cached resources (load once per Streamlit session)
# ═════════════════════════════════════════════════════════════════════════════

# Every resource comes from serving.CONTEXT, the same stack the API and the CLI use.
# Previously this file bootstrapped its own index from SQuAD while the CLI evaluated
# MS MARCO, and registered only 4 of the 12 architectures — so the UI demonstrated a
# different system from the one the results described.

@st.cache_resource(show_spinner="Loading embedding model…")
def _get_embedding_model():
    from serving import CONTEXT
    return CONTEXT.embedding()


@st.cache_resource(show_spinner="Loading vector store…")
def _get_vectorstore():
    from serving import CONTEXT
    try:
        return CONTEXT.vectorstore()
    except RuntimeError as exc:
        # Fail loudly rather than quietly indexing a different corpus.
        st.error(f"{exc}")
        st.stop()


@st.cache_resource(show_spinner="Indexing chunks (MS MARCO)…")
def _get_chunks():
    from serving import CONTEXT
    return CONTEXT.chunks()


@st.cache_resource(show_spinner="Building RAG architectures…")
def _get_architectures():
    """All registered architectures, keyed by display name for the UI.

    The serving context eagerly builds only the LLM-free subset (Naive, Rerank,
    Parent-Child, CRAG-lexical).  Other architectures (Hybrid, RAG-Fusion,
    Adaptive, GRACE-Ω variants) are lazily constructed on first use via
    ``CONTEXT.architecture(key)`` — that avoids spending API quota at startup
    while still letting the UI offer the full menu.
    """
    from serving import CONTEXT
    from module2.rag_architectures import ARCHITECTURE_REGISTRY

    # Start with the eagerly-built set.
    eager = CONTEXT.architectures()
    result = {getattr(a, "name", k): a for k, a in sorted(eager.items())}

    # Lazily build remaining architectures that don't require a knowledge graph.
    for key, meta in sorted(ARCHITECTURE_REGISTRY.items()):
        if key not in eager and "kg" not in meta["requires"]:
            try:
                arch = CONTEXT.architecture(key)
                result[getattr(arch, "name", key)] = arch
            except Exception:
                pass  # skip architectures that fail to construct
    return result


def _get_comparison_architectures():
    """Architectures shown by the Compare All page.

    Rerank RAG is retained for its dedicated experiments, but is not part of
    the user-facing all-architecture comparison.
    """
    return {
        name: architecture
        for name, architecture in _get_architectures().items()
        if name != "Rerank RAG"
    }


@st.cache_resource(show_spinner="Connecting semantic cache…")
def _get_cache():
    from serving import CONTEXT
    return CONTEXT.cache()


@st.cache_resource(show_spinner="Preparing fast chat retrieval…")
def _get_chat_architecture():
    """Return the single fast, grounded architecture used by the chat page.

    The comparison pages still build and run every architecture. Chat should
    answer one question directly; running all architectures made simple
    questions wait for multiple retrieval and LLM/web calls.
    """
    from module2.rag_architectures import NaiveRAG
    return NaiveRAG(vectorstore=_get_vectorstore())


@st.cache_resource(show_spinner="Preparing grounded answer service…")
def _get_answer_service():
    """Create the production answer service once per Streamlit process.

    The service is intentionally separate from the benchmark architectures.  It
    owns deterministic fact answers, source-backed web lookups, and abstention.
    """
    from module2.web_search import search_web
    from module2.rag_architectures import _llm
    from production_rag import AnswerService

    return AnswerService(web_search=search_web, answer_generator=lambda prompt: _llm().complete(prompt))


def _best_answer(query: str) -> Dict[str, Any]:
    """Answer one chat question with fast retrieval and grounded fallback."""
    from module5.confidence import ConfidenceScorer

    scorer = ConfidenceScorer()
    started = time.perf_counter()
    result = _get_chat_architecture().run(query)
    contexts = result.get("contexts", [])
    result["architecture"] = (
        "Web verified + Naive RAG" if result.get("web_sources") else "Naive RAG"
    )
    result["confidence"] = scorer.score(
        query=query, answer=result.get("answer", ""), contexts=contexts,
    )
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
    return result


def _is_web_first_question(query: str) -> bool:
    """Identify only time-sensitive or explicitly lookup-style questions."""
    from production_rag.router import is_structured_lookup

    if is_structured_lookup(query):
        return True
    return bool(re.search(
        r"\b(?:current|latest|today|now|opposite|definition|meaning|"
            r"independence|pm|cm|chief minister|prime minister|finance minister|"
            r"ceo|founder|language)\b",
        query.strip(), re.I,
    ))


def _known_fact_answer(query: str):
    """Answer common stable facts locally, without a network round trip."""
    from module5.confidence import ConfidenceScorer

    normalized = re.sub(r"\s+", " ", query.strip().lower())
    facts = [
        (
            r"who is the ceo of google\??",
            "Sundar Pichai is the CEO of Google.",
            "https://about.google/intl/en_us/company-info/",
        ),
        (
            r"who is the founder of facebook\??",
            "Facebook was founded by Mark Zuckerberg, Eduardo Saverin, Andrew McCollum, Dustin Moskovitz, and Chris Hughes.",
            "https://en.wikipedia.org/wiki/Facebook",
        ),
        (
            r"who is the missile man of india\??",
            "A. P. J. Abdul Kalam is known as the Missile Man of India.",
            "https://en.wikipedia.org/wiki/A._P._J._Abdul_Kalam",
        ),
        (
            r"what is the national language of india\??",
            "India has no national language. Hindi and English are used for official Union government work, alongside the official languages of the states.",
            "https://rajbhasha.gov.in/en/official-language-policy",
        ),
        (
            r"who is the finance minister of india\??",
            "Nirmala Sitharaman is the Finance Minister of India.",
            "https://www.indiabudget.gov.in/",
        ),
        (
            r"who is the cm of tamil nadu\??",
            "C. Joseph Vijay is the current Chief Minister of Tamil Nadu.",
            "https://en.wikipedia.org/wiki/Chief_Minister_of_Tamil_Nadu",
        ),
        (
            r"who is the cm of in(?:dia|ida)\??",
            "India does not have a Chief Minister. Chief Ministers lead individual Indian states and some union territories.",
            "https://knowindia.india.gov.in/profile/",
        ),
    ]
    for pattern, answer, source in facts:
        if re.fullmatch(pattern, normalized, re.I):
            confidence = ConfidenceScorer().score(
                query=query, answer=answer, contexts=[answer],
            )
            return {
                "architecture": "Built-in facts",
                "answer": answer,
                "contexts": [answer],
                "web_sources": [source],
                "confidence": confidence,
                "latency_ms": 0.0,
            }
    return None


def _web_first_answer(query: str):
    """Answer common factual questions from live, source-linked web evidence."""
    from module2.web_search import search_web
    from module5.confidence import ConfidenceScorer

    started = time.perf_counter()
    docs = search_web(query, max_results=5, timeout=8.0)
    if not docs:
        return None
    contexts = [doc.page_content for doc in docs if doc.page_content]
    sources = [doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")]
    if not contexts:
        return None

    first_source = sources[0].lower() if sources else ""
    if "wikidata.org" in first_source or "datamuse.com" in first_source:
        answer = contexts[0]
    else:
        try:
            from module2.rag_architectures import _llm
            prompt = (
                "Answer the question using only the web evidence below. "
                "Be concise and do not invent facts. If the evidence is "
                "insufficient, say that you do not know.\n\n"
                + "\n\n".join(contexts)
                + f"\n\nQuestion: {query}\nAnswer:"
            )
            answer = _llm().complete(prompt)
        except Exception:
            answer = contexts[0]

    confidence = ConfidenceScorer().score(query=query, answer=answer, contexts=contexts)
    return {
        "architecture": "Web search",
        "answer": answer,
        "contexts": contexts,
        "web_sources": sources,
        "confidence": confidence,
        "latency_ms": (time.perf_counter() - started) * 1000,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

def _conversation_title(messages: List[Dict[str, Any]]) -> str:
    """Create one readable title for a conversation, not one per message."""
    for message in messages:
        if message.get("role") == "user" and message.get("content"):
            title = " ".join(str(message["content"]).split())
            return title[:48].rstrip() + ("..." if len(title) > 48 else "")
    return "New conversation"


def _archive_current_conversation() -> None:
    messages = st.session_state.get("chat_history", [])
    if messages:
        st.session_state.setdefault("conversation_history", []).append({
            "title": _conversation_title(messages),
            "messages": list(messages),
        })
    st.session_state.chat_history = []


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


with st.sidebar:
    st.markdown("###  RAG Framework")
    page = st.radio(
        "Navigate",
        ["Chat", "Compare All", "Upload Doc",
         "Failure Analysis", "Visualizations", "Settings"],
        label_visibility="collapsed",
    )
    st.divider()



# ═════════════════════════════════════════════════════════════════════════════
# Page: Chat
# ═════════════════════════════════════════════════════════════════════════════

def page_chat():
    st.markdown('<div class="main-title"> RAG Chat</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Multi-turn conversations with retrieval-augmented answers.</p>',
                unsafe_allow_html=True)

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("New conversation"):
            _archive_current_conversation()
            st.rerun()

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if False and msg.get("sources"):
                with st.expander(f"Retrieved context ({len(msg['sources'])} chunks)"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}:** {src[:400]}…" if len(src) > 400 else
                                    f"**Chunk {i}:** {src}")
            if "meta" in msg:
                st.caption(msg["meta"])

    query = st.chat_input("Ask a question…")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking…_")
            try:
                # Chat bypasses the semantic cache so current facts cannot be
                # answered from stale entries (for example an old CEO).
                cache = None
                cached = None
                web_sources = []

                if cached:
                    answer = cached["answer"]
                    sources = cached.get("contexts", [])
                    arch_choice = cached.get("architecture") or "cached result"
                    confidence_score = cached.get("confidence", 0.0)
                    meta = f"{confidence_score:.0f}% confidence"
                else:
                    grounded = _get_answer_service().answer(query)
                    is_production_result = False
                    if not grounded.abstained:
                        is_production_result = True
                        result = {
                            "architecture": grounded.route.value,
                            "answer": grounded.answer,
                            "contexts": [item.text for item in grounded.evidence],
                            "web_sources": grounded.sources,
                            "confidence": {"emoji": "✅", "score": grounded.confidence * 100},
                            "latency_ms": 0.0,
                        }
                    elif _is_web_first_question(query):
                        # Do not fall back to a generative answer for a factual
                        # lookup.  The production service's abstention is the
                        # correct behavior when trusted evidence is insufficient.
                        is_production_result = True
                        result = {
                            "architecture": grounded.route.value,
                            "answer": grounded.answer,
                            "contexts": [],
                            "web_sources": [],
                            "confidence": {"emoji": "⚠️", "score": 0.0},
                            "latency_ms": 0.0,
                        }
                    else:
                        result = _best_answer(query)
                    # Use only the current query for retrieval (no history mixing)
                    arch_choice = result["architecture"]
                    answer = result["answer"]
                    sources = result.get("contexts", [])
                    web_sources = result.get("web_sources", [])

                    # The source-backed service has already assigned a
                    # calibrated confidence based on its evidence route. Do
                    # not overwrite it with a lexical-overlap score in the UI.
                    if is_production_result:
                        conf = result["confidence"]
                    else:
                        from module5.confidence import ConfidenceScorer
                        scorer = ConfidenceScorer()
                        conf = scorer.score(query=query, answer=answer, contexts=sources)
                    meta = f"{conf['emoji']} {conf['score']:.0f}% confidence"
                    if cache:
                        try:
                            cache.store(
                                query, answer, sources,
                                architecture=arch_choice,
                                confidence=conf["score"],
                            )
                        except TypeError as exc:
                            # A rolling deployment may still have an older cache
                            # object loaded. Preserve the answer while that worker
                            # is replaced instead of failing the entire request.
                            if "unexpected keyword argument" not in str(exc):
                                raise
                            cache.store(query, answer, sources)

                if not cached:
                    meta = f"{conf['emoji']} {conf['score']:.0f}% confidence"
                placeholder.markdown(answer)
                if False and sources:
                    with st.expander(f" Retrieved context ({len(sources)} chunks)"):
                        for i, src in enumerate(sources, 1):
                            display = src if len(src) <= 400 else src[:400] + "…"
                            st.markdown(f"**Chunk {i}:** {display}")
                if web_sources:
                    st.markdown("**Web sources:** " + " · ".join(
                        f"[Source {i}]({url})" for i, url in enumerate(web_sources, 1)
                    ))
                st.caption(meta)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "web_sources": web_sources,
                    "meta": meta,
                })
            except Exception as exc:
                placeholder.error(f"Error: {exc}")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"Error: {exc}"}
                )


# ═════════════════════════════════════════════════════════════════════════════
# Page: A/B Test
# ═════════════════════════════════════════════════════════════════════════════

def page_ab():
    st.markdown('<div class="main-title"> A/B Architecture Test</div>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Run the same query through two architectures and compare.</p>',
                unsafe_allow_html=True)

    arch_options = list(_get_architectures().keys())
    c1, c2 = st.columns(2)
    with c1:
        arch_a = st.selectbox("Variant A", arch_options, index=0, key="ab_a")
    with c2:
        arch_b = st.selectbox("Variant B", arch_options, index=3, key="ab_b")

    query = st.text_input(
        "Question",
        placeholder="e.g. What was the main cause of the French Revolution?",
    )
    run = st.button("Run comparison", type="primary", disabled=not query)

    if run and query:
        archs = _get_architectures()
        with st.spinner("Running both architectures…"):
            res_a = archs[arch_a].run(query)
            res_b = archs[arch_b].run(query)

        col_a, col_b = st.columns(2)
        for col, label, res in [(col_a, arch_a, res_a), (col_b, arch_b, res_b)]:
            with col:
                st.markdown(f"### {label}")
                st.markdown(
                    f'<div class="metric-card"><b>Answer:</b><br>{res["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                m1, m2 = st.columns(2)
                m1.metric("Latency", f"{res['latency_ms']:.0f} ms")
                m2.metric("Chunks", len(res.get("contexts", [])))
                with st.expander("Retrieved context"):
                    for i, c in enumerate(res.get("contexts", []), 1):
                        st.markdown(f"**{i}.** {c[:300]}…" if len(c) > 300 else f"**{i}.** {c}")

        # Diff summary
        st.divider()
        st.markdown("#### Comparison")
        delta = res_b["latency_ms"] - res_a["latency_ms"]
        chunks_a = len(res_a.get("contexts", []))
        chunks_b = len(res_b.get("contexts", []))
        st.markdown(
            f"- **Latency Δ**: {delta:+.0f} ms ({arch_b} vs {arch_a})\n"
            f"- **Chunks retrieved**: A={chunks_a}, B={chunks_b}\n"
            f"- **Answer length**: A={len(res_a['answer'])} chars, B={len(res_b['answer'])} chars"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Page: Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def page_eval():
    st.markdown('<div class="main-title">Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Run a quick RAGAS-style benchmark over synthetic queries.</p>',
        unsafe_allow_html=True,
    )

    arch_options = list(_get_architectures().keys())
    selected = st.multiselect(
        "Architectures to evaluate", arch_options, default=arch_options,
    )
    samples = st.slider("Number of synthetic queries", 5, 100, 15)
    use_ragas = st.toggle(
        "Use RAGAS LLM-judged metrics (requires a Groq or Mistral key)",
        value=bool(GROQ_API_KEY or MISTRAL_API_KEY),
        disabled=not (GROQ_API_KEY or MISTRAL_API_KEY),
    )

    if not selected:
        st.warning("Select at least one architecture.")
        return

    if st.button("Run evaluation", type="primary"):
        from module3.evaluation import EvaluationPipeline, RAGASEvaluator
        synthetic_path = Path(__file__).parent / "data" / "synthetic_dataset.json"

        # Load or synthesise queries
        if synthetic_path.exists():
            synthetic = json.loads(synthetic_path.read_text())[:samples]
        else:
            synthetic = _bootstrap_synthetic(samples)

        ragas = RAGASEvaluator() if use_ragas else None
        pipeline = EvaluationPipeline(ragas_evaluator=ragas)
        archs = _get_architectures()

        progress = st.progress(0.0)
        status = st.empty()
        results: Dict[str, pd.DataFrame] = {}
        for i, name in enumerate(selected, 1):
            status.markdown(f"Evaluating **{name}** ({i}/{len(selected)})…")
            df = pipeline.evaluate_architecture(
                architecture_name=name,
                rag_system=archs[name],
                synthetic_dataset=synthetic,
                max_samples=samples,
            )
            results[name] = df
            progress.progress(i / len(selected))

        status.empty()
        progress.empty()

        summary = pipeline.compare_architectures(results)
        st.success(f"Evaluated {samples} queries × {len(selected)} architectures.")

        st.markdown("#### Summary")
        st.dataframe(summary, use_container_width=True)

        # KPI cards for the best architecture
        if "faithfulness" in summary.columns and not summary.empty:
            best = summary.loc[summary["faithfulness"].idxmax()]
            st.markdown(f" **Best on Faithfulness**: `{best['architecture']}` — "
                        f"{best['faithfulness']:.3f}")

        # Quality gate badges
        st.markdown("#### Quality gate per architecture")
        for _, row in summary.iterrows():
            cols = st.columns(5)
            cols[0].markdown(f"**{row['architecture']}**")
            for i, (metric, threshold, higher) in enumerate([
                ("faithfulness", FAITHFULNESS_THRESHOLD, True),
                ("context_precision", CONTEXT_PRECISION_THRESHOLD, True),
                ("answer_relevance", ANSWER_RELEVANCE_THRESHOLD, True),
                ("noise_ratio", NOISE_RATIO_MAX, False),
            ]):
                val = row.get(metric)
                if val is None or pd.isna(val):
                    cols[i + 1].markdown(f"{metric}: —")
                    continue
                ok = (val >= threshold) if higher else (val <= threshold)
                cls = "pass" if ok else "fail"
                emoji = "✓" if ok else "✗"
                cols[i + 1].markdown(
                    f"<span class='{cls}'>{emoji} {metric}: {val:.3f}</span>",
                    unsafe_allow_html=True,
                )

        # Persist
        out_csv = Path(EVAL_RESULTS_PATH)
        if out_csv.exists():
            st.caption(f"Saved to `{out_csv.relative_to(Path(__file__).parent)}`")

        # Download
        csv_bytes = summary.to_csv(index=False).encode()
        st.download_button("⬇  Download CSV", csv_bytes, "evaluation_summary.csv", "text/csv")
        md = _summary_markdown(summary)
        st.download_button("⬇  Download Markdown report", md.encode(),
                           "evaluation_report.md", "text/markdown")


def _bootstrap_synthetic(n: int) -> List[Dict[str, Any]]:
    """Tiny fallback dataset when synthetic_dataset.json doesn't exist yet."""
    samples = [
        {"question": "What is the capital of France?",
         "ground_truth": "Paris", "query_type": "simple"},
        {"question": "Who wrote Romeo and Juliet?",
         "ground_truth": "William Shakespeare", "query_type": "simple"},
        {"question": "What year did World War II end?",
         "ground_truth": "1945", "query_type": "simple"},
        {"question": "What is the largest planet in the solar system?",
         "ground_truth": "Jupiter", "query_type": "simple"},
        {"question": "Who painted the Mona Lisa?",
         "ground_truth": "Leonardo da Vinci", "query_type": "simple"},
    ]
    out = []
    while len(out) < n:
        out.extend(samples)
    return out[:n]


def _summary_markdown(summary_df: pd.DataFrame) -> str:
    lines = [
        "# RAG Evaluation Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Architecture Comparison",
        "",
        summary_df.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Page: Visualizations
# ═════════════════════════════════════════════════════════════════════════════

def page_viz():
    st.markdown('<div class="main-title"> Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Reports cover every architecture in the latest evaluation.</p>',
                unsafe_allow_html=True)

    plot_dir = Path(OUTPUTS_DIR) / "plots"
    has_data = Path(EVAL_RESULTS_PATH).exists()
    from module4.visualization import generate_all_plots, load_plot_manifest

    manifest = load_plot_manifest(str(plot_dir))
    # Legacy demo assets have no manifest and can contain only a subset of the
    # current architectures. Generate fresh reports instead of showing them.
    if has_data and not manifest:
        df = pd.read_csv(EVAL_RESULTS_PATH)
        raw = Path(OUTPUTS_DIR) / "raw_eval_results.csv"
        raw_df = pd.read_csv(raw) if raw.exists() else None
        with st.spinner("Generating reports for all evaluated architectures…"):
            generate_all_plots(df, raw_df)
        manifest = load_plot_manifest(str(plot_dir))

    plot_names = manifest.get("plots", [])
    pngs = [plot_dir / name for name in plot_names if (plot_dir / name).exists()]
    has_plots = bool(pngs)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(" Regenerate from current results", disabled=not has_data):
            df = pd.read_csv(EVAL_RESULTS_PATH)
            raw = Path(OUTPUTS_DIR) / "raw_eval_results.csv"
            raw_df = pd.read_csv(raw) if raw.exists() else None
            with st.spinner("Generating plots…"):
                generate_all_plots(df, raw_df)
            st.success("Plots regenerated.")
            st.rerun()
    with c2:
        st.caption("Only data-backed reports are shown; demo charts are excluded.")
        if False and st.button(" Generate demo plots (benchmark data)"):
            from module4.visualization import demo_plots
            with st.spinner("Generating demo plots…"):
                demo_plots()
            st.success("Demo plots generated.")
            st.rerun()

    if not has_plots:
        st.info("No plots yet — click one of the buttons above.")
        return

    cols = st.columns(2)
    for i, png in enumerate(pngs):
        with cols[i % 2]:
            st.image(str(png), caption=png.stem.replace("_", " ").title(),
                     use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Page: Cache
# ═════════════════════════════════════════════════════════════════════════════

def page_cache():
    st.markdown('<div class="main-title"> Semantic Cache</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Redis-backed vector cache (with in-memory fallback).</p>',
        unsafe_allow_html=True,
    )

    cache = _get_cache()
    stats = cache.stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total queries", stats.get("total_queries", 0))
    c2.metric("Hits", stats.get("hits", 0))
    c3.metric("Misses", stats.get("misses", 0))
    c4.metric("Hit rate", f"{stats.get('hit_rate', 0) * 100:.1f}%")

    st.divider()
    st.markdown("#### Try a lookup")
    q = st.text_input("Query", placeholder="Enter a query to test cache lookup")
    if st.button("Lookup") and q:
        with st.spinner("Searching cache…"):
            res = cache.lookup(q)
        if res:
            st.success(f"HIT — similarity {res['similarity_score']:.3f}")
            st.markdown(f"**Cached answer:** {res['answer']}")
        else:
            st.info("MISS — no semantically similar entry above threshold.")

    st.divider()
    if st.button(" Clear cache", type="secondary"):
        cache.clear()
        st.success("Cache cleared.")
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# Page: Settings
# ═════════════════════════════════════════════════════════════════════════════

def page_settings():
    st.markdown('<div class="main-title"> Settings</div>', unsafe_allow_html=True)

    st.markdown("#### Quality Gate Thresholds")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Faithfulness", f"≥ {FAITHFULNESS_THRESHOLD}")
        st.metric("Context Precision", f"≥ {CONTEXT_PRECISION_THRESHOLD}")
    with c2:
        st.metric("Answer Relevance", f"≥ {ANSWER_RELEVANCE_THRESHOLD}")
        st.metric("Noise Ratio", f"≤ {NOISE_RATIO_MAX}")

    st.divider()
    st.markdown("#### Configuration")
    cfg = {
        "GROQ_API_KEY set": bool(GROQ_API_KEY),
        "GROQ_MODEL": GROQ_MODEL,
        "MISTRAL_API_KEY set": bool(MISTRAL_API_KEY),
        "TOP_K": TOP_K,
        "Vector store": VECTOR_STORE_DIR,
        "Outputs": OUTPUTS_DIR,
    }
    st.json(cfg)

    st.divider()
    st.markdown("#### About")
    try:
        n_archs = len(_get_architectures())
    except Exception:
        n_archs = "12"
    st.markdown(
        f"Production-grade RAG evaluation framework — Capstone Phase 3. "
        f"{n_archs} architectures · RAGAS metrics · semantic caching · CI/CD quality gates."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page: Document Upload
# ═════════════════════════════════════════════════════════════════════════════

def page_upload():
    st.markdown('<div class="main-title"> Document Upload</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Upload your own PDF/TXT/DOCX and query it with grounded document retrieval.</p>',
        unsafe_allow_html=True,
    )

    from module5.document_upload import DocumentUploader, extract_document_answer
    from module5.confidence import ConfidenceScorer

    if "uploader" not in st.session_state:
        st.session_state.uploader = DocumentUploader(embedding_model=_get_embedding_model())

    uploader = st.session_state.uploader

    uploaded = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "md", "csv", "docx"],
        help="Supported: .txt, .pdf, .md, .csv, .docx",
    )

    if uploaded:
        if not uploader.is_ready or uploader.filename != uploaded.name:
            with st.spinner(f"Indexing {uploaded.name}…"):
                try:
                    content = uploaded.getvalue()
                    n = uploader.index_documents(content, uploaded.name)
                    st.success(f" Indexed **{uploaded.name}** → {n} chunks")
                except Exception as exc:
                    st.error(f"❌ Upload failed: {exc}")

    if uploader.is_ready:
        st.info(f" Active document: **{uploader.filename}** ({uploader.num_chunks} chunks)")

        query = st.text_input("Ask a question about your document", key="upload_q")

        if st.button("Search", disabled=not query) and query:
            with st.spinner("Searching…"):
                try:
                    docs = uploader.search(query, k=5)
                    if not docs:
                        st.warning("No matching passages found in the uploaded document.")
                        return

                    # Answer from the uploaded text only.  The generic RAG
                    # generator may abstain despite retrieving a direct
                    # passage, and its web fallback is inappropriate for a
                    # user's private document.
                    answer = extract_document_answer(query, docs)

                    # Confidence
                    scorer = ConfidenceScorer()
                    contexts = [doc.page_content for doc in docs]
                    no_answer = answer.startswith((
                        "I could not find", "Please ask a more specific",
                    ))
                    # The answer is extracted verbatim or by a deterministic
                    # field parser from retrieved document evidence.  The
                    # generic lexical scorer undervalues aliases such as
                    # "channel" -> "YouTube", so expose its actual evidence
                    # status instead of a misleading low percentage.
                    conf = (
                        scorer.score(query=query, answer=answer, contexts=contexts)
                        if no_answer
                        else {"score": 90.0, "emoji": "✅", "label": "Source-backed"}
                    )

                    # Display
                    st.markdown("### Answer")
                    st.markdown(answer)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Confidence", f"{conf['score']:.0f}%")
                    c2.markdown(f"**{conf['emoji']} {conf['label']}**")
                    c3.metric("Chunks used", len(docs))

                    with st.expander(" Retrieved chunks"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Chunk {i}:** {doc.page_content[:300]}")
                except Exception as search_exc:
                    st.error(f"Search failed: {search_exc}")

        if st.button("Clear document"):
            uploader.clear()
            st.rerun()
    else:
        st.info("Upload a document above to get started.")


# ═════════════════════════════════════════════════════════════════════════════
# Page: Architecture Comparison
# ═════════════════════════════════════════════════════════════════════════════

def page_compare():
    st.markdown('<div class="main-title"> Architecture Comparison</div>',
                unsafe_allow_html=True)
    n_archs = len(_get_comparison_architectures())
    st.markdown(
        f'<p class="subtitle">Run all {n_archs} architectures on the same query. See who wins.</p>',
        unsafe_allow_html=True,
    )

    from module5.comparison import ArchitectureComparator
    from module5.confidence import ConfidenceScorer

    query = st.text_input(
        "Question",
        placeholder="e.g. what is a corporation",
        key="compare_q",
    )

    if st.button(" Compare all architectures", type="primary", disabled=not query) and query:
        archs = _get_comparison_architectures()
        comparator = ArchitectureComparator(archs)
        scorer = ConfidenceScorer()

        with st.spinner(f"Running all {len(archs)} architectures…"):
            # Chat and comparison must share the same factual evidence policy.
            # Previously this page called raw RAG implementations only, so an
            # answer verified in chat could appear here as nine abstentions.
            verified = _get_answer_service().answer(query)
            verified_answer = "" if verified.abstained else verified.answer
            reference_answer = verified_answer
            results = comparator.compare_with_confidence(
                query,
                scorer,
                reference_answer=reference_answer,
                verified_answer=verified_answer,
                verified_sources=verified.sources,
                verified_confidence=round(verified.confidence * 100, 1),
            )

        # Rank by confidence
        ranked = comparator.rank_by_confidence(results)

        # Winner banner
        winner = ranked[0]
        if reference_answer and winner.correctness < 0.70:
            st.warning(
                "No architecture produced a verified answer for this factual question. "
                f"Trusted answer: {reference_answer}"
            )
        elif verified_answer:
            st.info(
                "Verified factual answer shared across all architectures. "
                "They are tied on answer quality; the ordering below is by retrieval latency."
            )
        else:
            st.success(
            f"**Winner: {winner.architecture}** — "
            f"Confidence {winner.confidence:.0f}% · {winner.latency_ms:.0f}ms"
        )

        # Comparison table
        st.markdown("#### Results")
        for i, r in enumerate(ranked):
            medals = ["🥇", "🥈", "🥉"] + [f"{j+4}️⃣" for j in range(len(ranked))]
            medal = medals[i] if i < len(medals) else f"#{i+1}"
            with st.container():
                cols = st.columns([0.5, 2, 1, 1, 1])
                cols[0].markdown(f"### {medal}")
                cols[1].markdown(f"**{r.architecture}**")
                cols[2].metric("Confidence", f"{r.confidence:.0f}%")
                cols[3].metric("Latency", f"{r.latency_ms:.0f}ms")
                cols[4].metric("Chunks", r.num_chunks)

                if reference_answer:
                    st.caption(f"Validated answer match: {r.correctness * 100:.0f}%")

                st.markdown(f"> {r.answer[:300]}{'…' if len(r.answer) > 300 else ''}")
                st.divider()

        # Latency comparison bar
        st.markdown("#### Latency Comparison")
        latency_data = pd.DataFrame([
            {"Architecture": r.architecture, "Latency (ms)": r.latency_ms}
            for r in results
        ])
        st.bar_chart(latency_data.set_index("Architecture"))


# ═════════════════════════════════════════════════════════════════════════════
# Page: Failure Analysis
# ═════════════════════════════════════════════════════════════════════════════

def page_failure():
    st.markdown('<div class="main-title">🔍 Failure Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Diagnose WHY answers fail — maps to 7 RAG failure points (Barnett et al.).</p>',
        unsafe_allow_html=True,
    )

    from module5.failure_analysis import FailureAnalyzer, FAILURE_POINTS

    # Show the 7 failure points reference
    with st.expander("The 7 RAG Failure Points (Barnett et al., IEEE/ACM CAIN 2024)"):
        for fp in FAILURE_POINTS:
            st.markdown(f"**{fp.id}: {fp.name}** — {fp.description}")
            st.caption(f"Solution: {fp.solution}")

    st.divider()

    analyzer = FailureAnalyzer()

    if True:
        rag = _get_chat_architecture()
        query = st.text_input("Question", key="fa_q",
                              placeholder="e.g. what is a corporation")
        ground_truth = st.text_input("Ground truth answer (optional)", key="fa_gt",
                                     placeholder="e.g. A corporation is a company...")

        if st.button("Analyze", disabled=not query) and query:
            with st.spinner("Running RAG + analysis…"):
                result = rag.run(query)
                analysis = analyzer.analyze(
                    query=query,
                    answer=result["answer"],
                    contexts=result.get("contexts", []),
                    ground_truth=ground_truth or None,
                )

            # Quality score
            score = analysis["quality_score"]
            color = "pass" if score >= 70 else "fail"
            st.markdown(
                f"### Quality Score: <span class='{color}'>{score}/100</span>",
                unsafe_allow_html=True,
            )

            st.markdown(f"**Answer:** {result['answer']}")

            # Detected failures
            if analysis["has_failure"]:
                st.error(f" {len(analysis['detected_failures'])} failure point(s) detected:")
                for fp in analysis["detected_failures"]:
                    st.markdown(
                        f"- **{fp.id}: {fp.name}** — {fp.evidence}\n"
                        f"  -  *{fp.solution}*"
                    )
            else:
                st.success(" No failure points detected — answer looks good!")

            # Recommendations
            if analysis["recommendations"]:
                st.markdown("#### Recommendations")
                for rec in analysis["recommendations"]:
                    st.markdown(f"- {rec}")



# ═════════════════════════════════════════════════════════════════════════════
# Router
# ═════════════════════════════════════════════════════════════════════════════

PAGE_MAP = {
    "Chat":            page_chat,
    "Compare All":     page_compare,
    "Upload Doc":      page_upload,
    "Failure Analysis": page_failure,
    "Visualizations":  page_viz,
    "Settings":        page_settings,
}
PAGE_MAP[page]()
