"""Exportable reports: Markdown, self-contained HTML, JSON, CSV and a PDF of all figures."""
from __future__ import annotations

import base64
import html
import os
from typing import Any, Dict, List

import pandas as pd

from .registry import dumps

CSS = ("body{font-family:system-ui,Segoe UI,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1f2933}"
       "h1,h2{border-bottom:1px solid #d9e2ec;padding-bottom:.2rem}table{border-collapse:collapse;font-size:12px;margin:.6rem 0}"
       "td,th{border:1px solid #d9e2ec;padding:3px 7px;text-align:right}th{background:#f0f4f8}td:first-child,th:first-child{text-align:left}"
       "img{max-width:100%}.warn{background:#fff3cd;border-left:4px solid #e0a800;padding:.6rem 1rem}code,pre{background:#f0f4f8;padding:2px 4px}"
       "pre{padding:.6rem;overflow:auto}")


def limitations(R: Dict[str, Any]) -> List[str]:
    ds, sizes = R["dataset"], R["experiment"].metrics["split_sizes"]
    out = []
    if ds.name == "offline_fixture":
        out.append("**The dataset is a small synthetic offline fixture. Nothing in this report is a research result.** It verifies that the pipeline works; "
                   "re-run with `--dataset squad|msmarco|beir:<name>` on real data for findings.")
    out.append(f"Architectures are compared on the held-out **test split (n={sizes['test']})**; train (n={sizes['train']}) trained the router and "
               f"validation (n={sizes['val']}) calibrated abstention thresholds. Confidence intervals are wide when n is small — treat overlapping CIs as ties.")
    out.append("Default components are offline stand-ins: TF-IDF/LSA for dense embeddings, a lexical cross-scorer for a cross-encoder, an extractive "
               "generator for an LLM, and lexical-support proxies for faithfulness. Swap in real ones via `DenseEncoderRetriever`, `CrossEncoderReranker`, "
               "`LLMGenerator` and `LangChainAdapter`, and validate any LLM judge against human labels (Cohen's κ).")
    out.append("Latency is measured in-process on one machine (shared Python GIL under concurrency); use `load_test.py` against the HTTP API for deployment numbers. "
               "Cache latency savings use a **simulated** backend delay and are labelled as such.")
    out.append("Many comparisons are made; paired tests are Benjamini–Hochberg corrected within each metric, but exploratory findings still need replication.")
    out.append("Extractive generation is deterministic; for stochastic LLM generators repeat runs and report `run_to_run_variability`.")
    return out


def _df_md(df: pd.DataFrame, n: int = 30) -> str:
    d = df.head(n).copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: "" if x != x else f"{x:.3f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)) + "\n"


def _df_html(df: pd.DataFrame, n: int = 40) -> str:
    return df.head(n).to_html(index=False, float_format=lambda x: f"{x:.3f}", na_rep="", border=0)


def build_report(R: Dict[str, Any]) -> Dict[str, str]:
    exp, d, T, J = R["experiment"], R["dataset"], R["tables"], R["json"]
    out_dir = R["exp_dir"]; prim = R["primary"]
    sections: List[Dict[str, Any]] = []

    def add(title, text="", tables=(), images=(), pre=""):
        sections.append({"title": title, "text": text, "tables": list(tables), "images": [R["plots"][i] for i in images if i in R["plots"]], "pre": pre})

    h = R["headline"]
    add("Summary", f"Experiment **{exp.id}** on dataset `{d.name}` (fingerprint `{d.fingerprint()}`). Primary metric: **{prim}**. "
        f"Best architecture: **{h['best_arch']}** ({prim}={h['best_value']:.3f}). Pareto-optimal: {', '.join(h['pareto_front'])}. "
        f"Knee point (best quality gain per ms on the frontier): **{h['knee_point']}**. "
        f"Router significantly beats best fixed architecture: **{h['router_beats_best_fixed']}**.")
    add("1. Architecture leaderboard (held-out test split)", "Mean with 95% bootstrap CI; `n` = queries; `*_n` = queries where the metric is defined.",
        [("Leaderboard", T["leaderboard"][[c for c in T["leaderboard"].columns if not c.endswith(("_lo", "_hi", "_n"))]])], ["leaderboard"])
    add("2. Quality vs latency (Pareto frontier)", "An architecture is Pareto-optimal if no other is both better and faster.",
        [("Quality-first ranking (0.7/0.2/0.1)", T["weighted_ranking_quality_first"][["arch", "utility", prim, "lat_p50_ms"]]),
         ("Speed-first ranking (0.2/0.7/0.1)", T["weighted_ranking_speed_first"][["arch", "utility", prim, "lat_p50_ms"]])], ["pareto"])
    add("3. Performance by query type", tables=[("Primary metric by query type", T["by_query_type"])])
    if len(T.get("paired_vs_baselines", [])):
        cols = ["metric", "arch", "baseline", "n_pairs", "mean_diff", "ci_lo", "ci_hi", "p_adj_bh", "cohens_dz", "effect", "cliffs_delta", "significant"]
        add("4. Paired significance vs baselines", "Paired bootstrap CI of the per-query difference, sign-flip permutation p (BH-adjusted), Cohen's dz and Cliff's δ.",
            [("Paired comparisons", T["paired_vs_baselines"][cols])])
    add("5. Failure diagnosis", "Each failed query is assigned a stage using measured evidence (qrels ranks, gold-answer-in-context) with a stated confidence; "
        "inferred causes are labelled as such.", [("Failure rate by architecture × query type × category", T["failure_stats_by_arch_qtype"].head(25))],
        ["failure_heatmap", "failure_by_qtype"])
    if "intervention_effect" in J:
        e = J["intervention_effect"]
        add("5b. Intervention before/after", f"`{e['before']}` → `{e['after']}` on the same queries: fixed **{e['fixed']}**, regressed **{e['regressed']}**, "
            f"still failing **{e['still_failing']}** (accuracy {e['acc_before']:.3f} → {e['acc_after']:.3f}).")
    r = J["router_learned"]; rr = J["router_rule"]
    add("6. Query-aware router", f"Held-out n={r['n_test']}. Learned router {prim}={r[f'router_{prim}']:.3f} vs best fixed `{r['best_fixed_arch(train-selected)']}` "
        f"{r[f'best_fixed_{prim}']:.3f} (diff {r['router_minus_best_fixed']['mean_diff']:+.3f}, CI [{r['router_minus_best_fixed']['ci_lo']:+.3f}, "
        f"{r['router_minus_best_fixed']['ci_hi']:+.3f}]); oracle {r[f'oracle_{prim}']:.3f}. Rule router {rr[f'router_{prim}']:.3f}. "
        "A router only 'wins' if the CI excludes zero.", images=["router"], pre=J["router_tree"])
    add("7. Ablation (remove one component at a time)", tables=[("Ablated − full, paired", T["ablation"])])
    for k in ("chunk_size", "top_k", "hybrid_weight", "rerank_pool"):
        if f"sweep_{k}" in T:
            add(f"8. Sweep: {k}", tables=[(k, T[f"sweep_{k}"])], images=[f"sweep_{k}"])
    if "calibration" in J:
        c = J["calibration"]
        add("9. Confidence calibration", f"`{c['arch']}`: ECE {c['ece_raw']:.3f} → {c['ece_isotonic']:.3f} after isotonic calibration fitted on validation "
            f"(Brier {c['brier_raw']:.3f} → {c['brier_isotonic']:.3f}, n={c['n_test']}).", images=["reliability"])
    if "cache_modes" in T:
        add("10. Caching: none vs exact vs semantic", "Semantic hits require matching anchors (entities/numbers/codes), corpus fingerprint and requirements. "
            "Latency uses a simulated backend delay. `harmful_hits` = cached answer worse than a fresh one.",
            [("Cache modes", T["cache_modes"][["mode", "hit_rate", "mean_latency_ms", "p95_latency_ms", "f1", "harmful_hits", "latency_reduction_pct", "storage_bytes", "evictions"]])],
            ["cache"], pre=dumps(J.get("cache_staleness", {})))
    if "synthetic_validation" in T:
        s = J["synthetic_report"]; rv = J.get("real_vs_synthetic", {})
        add("11. Synthetic query validation", f"{s['n']} generated; valid rate {s['valid_rate']:.2f} (95% CI {s['valid_rate_ci95'][0]:.2f}–{s['valid_rate_ci95'][1]:.2f}); "
            f"duplicates {s['duplicate_rate']:.2%}; leakage {s['leak_rate']:.2%}. Real-vs-synthetic ranking agreement: Kendall τ={rv.get('kendall_tau')}, "
            f"top-1 agrees: {rv.get('top1_agrees')}, distorts ranking: {rv.get('distorts_ranking')} (degenerate ties: {rv.get('rankings_degenerate')}). "
            "A stratified sample for human review is in `human_validation_sample.csv`.", pre=dumps(s))
    if "load_test" in T:
        add("12. Load test (in-process)", tables=[("Concurrency sweep", T["load_test"])])
    if "security_prompt_injection" in T:
        add("13. Prompt-injection defence", "Passages containing an injection payload are planted next to each query's evidence.",
            [("Attack success rate", T["security_prompt_injection"])])
    add("Limitations & honesty notes", "\n".join(f"- {x}" for x in limitations(R)))
    add("Reproducibility record", pre=dumps({"id": exp.id, "seed": exp.seed, "config": exp.config, "dataset": exp.dataset, "env": exp.env}))

    # markdown
    md = [f"# RAG Benchmark Report — {exp.id}\n"]
    for s in sections:
        md.append(f"## {s['title']}\n"); md.append(s["text"] + "\n" if s["text"] else "")
        for t, df in s["tables"]:
            md.append(f"**{t}**\n\n{_df_md(df)}")
        for im in s["images"]:
            md.append(f"![{os.path.basename(im)}](plots/{os.path.basename(im)})\n")
        if s["pre"]:
            md.append(f"```\n{s['pre']}\n```\n")
    p_md = os.path.join(out_dir, "report.md")
    with open(p_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    # html
    import re
    def inline(t):
        t = html.escape(t); t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t); t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        return t.replace("\n- ", "<br>• ").replace("\n", "<br>")
    body = [f"<h1>RAG Benchmark Report — {exp.id}</h1>"]
    if d.name == "offline_fixture":
        body.append("<div class='warn'><b>Offline synthetic fixture:</b> pipeline verification only — not research results.</div>")
    for s in sections:
        body.append(f"<h2>{html.escape(s['title'])}</h2>")
        if s["text"]:
            body.append(f"<p>{inline(s['text'])}</p>")
        for t, df in s["tables"]:
            body.append(f"<h4>{html.escape(t)}</h4><div style='overflow-x:auto'>{_df_html(df)}</div>")
        for im in s["images"]:
            with open(im, "rb") as fh:
                body.append("<img src='data:image/png;base64," + base64.b64encode(fh.read()).decode() + "'>")
        if s["pre"]:
            body.append(f"<pre>{html.escape(s['pre'])}</pre>")
    p_html = os.path.join(out_dir, "report.html")
    with open(p_html, "w", encoding="utf-8") as fh:
        fh.write(f"<!doctype html><meta charset='utf-8'><title>RAG report</title><style>{CSS}</style>" + "\n".join(body))

    # pdf of figures
    p_pdf = os.path.join(out_dir, "report_figures.pdf")
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    with PdfPages(p_pdf) as pdf:
        for s in sections:
            for im in s["images"]:
                fig = plt.figure(figsize=(9, 6)); fig.suptitle(s["title"], fontsize=10)
                ax = fig.add_axes([0.02, 0.02, 0.96, 0.9]); ax.imshow(mpimg.imread(im)); ax.axis("off"); pdf.savefig(fig); plt.close(fig)
    p_json = os.path.join(out_dir, "report.json")
    with open(p_json, "w", encoding="utf-8") as fh:
        fh.write(dumps({"experiment": exp, "headline": R["headline"], "limitations": limitations(R),
                        "tables": {k: v for k, v in T.items() if isinstance(v, pd.DataFrame)}, **J}))
    return {"markdown": p_md, "html": p_html, "pdf": p_pdf, "json": p_json}
