# rag_lab — Benchmarking, Diagnosis & Adaptive Optimisation

`rag_lab/` is the research layer added on top of the existing 12-architecture framework. It turns the
project from "a set of RAG demos" into a framework that answers the five questions from the roadmap:

1. Which architecture performs best? → `leaderboard` + paired significance + Pareto frontier
2. …for which *type* of question? → `by_query_type`, failure heat-maps
3. *Why* did it fail? → `diagnosis.py` (stage, measured evidence, confidence, fix)
4. How can it improve without wasting latency? → sweeps, ablations, adaptive-k, reranker depth
5. Can the system pick an architecture for a new query? → `router.py`, evaluated on held-out queries

It depends only on numpy / pandas / scipy / scikit-learn / matplotlib, so it runs (and is tested) with no
network, GPU, LangChain or API key.

## Quick start

```bash
make lab-test        # 61 offline tests
make lab-demo        # full study on the OFFLINE fixture → outputs/lab/experiments/<id>/report.html
python main.py --mode lab                      # same, via the original entry point

# real data (needs internet + `pip install datasets`)
make lab-fetch       # caches SQuAD 2.0, MS MARCO v2.1, BEIR/scifact under data/lab/
make lab-squad       # python -m rag_lab run --dataset squad  --max-queries 2000
make lab-msmarco     # python -m rag_lab run --dataset msmarco --max-queries 2000
make lab-beir        # retrieval-only generalisation on BEIR/scifact

python -m rag_lab list
python -m rag_lab reproduce <experiment-id>    # re-runs from the stored config and diffs every metric
python -m rag_lab route "In which city was the founder of Acme born?" --exp <experiment-id>
```

> **The offline fixture is a synthetic corpus of fictional entities. It verifies that the pipeline works;
> numbers from it are not research results.** Every report produced from it says so.

## Datasets and their roles (kept distinct, as the roadmap asks)

| Dataset | Loader | Used for |
|---|---|---|
| MS MARCO v2.1 | `load_msmarco` (HF `microsoft/ms_marco`, streamed) | retrieval/ranking vs human `is_selected` labels; "No Answer Present." queries become unanswerable |
| SQuAD 2.0 | `load_squad_v2` (HF `rajpurkar/squad_v2`) | EM/F1 answer quality, grounding, **abstention** on genuinely unanswerable questions |
| BEIR (e.g. `scifact`) | `load_beir` (HF `BeIR/<name>` + `-qrels`) | multi-domain retrieval generalisation (ROADMAP #8); QA metrics are undefined and reported as such |
| Custom | `load_dataset_file(path)` | your own corpus + questions + reference answers |

## Module map

| Module | Responsibility |
|---|---|
| `datasets.py` | loaders, pure row→dataset converters (tested against the HF schemas), offline fixture |
| `architectures.py` | common `Pipeline.run()` contract; BM25, TF-IDF vector, LSA dense, hybrid (RRF/weighted), rerank, parent-child, query-fusion, **real iterative multi-hop**, adaptive-k, confidence fallback; `LangChainAdapter` for your existing `BaseRAG` classes |
| `metrics.py` | Recall/Precision/Hit@k, MRR, nDCG (graded), MAP, EM, token-F1, SQuAD-2.0 abstention scoring, p50/p95/p99, ECE/Brier |
| `stats.py` | paired bootstrap CI, sign-flip permutation, Wilcoxon, Cohen's d_z, Cliff's δ, Benjamini–Hochberg, **Cohen's κ**, Wilson CI, rank agreement |
| `evaluate.py` | one evaluation loop for every architecture; abstention-threshold calibration on a held-out split; leaderboard; latency breakdown |
| `diagnosis.py` | stage-aware failure diagnosis with measured-vs-inferred separation; failure stats/heat-maps; before/after intervention effect |
| `router.py` | rule router + learned (decision-tree) router; held-out comparison vs best fixed / oracle |
| `pareto.py` | Pareto frontier, knee point, quality/speed/cost-weighted ranking |
| `synthetic_validation.py` | duplicate / leakage / answerability / copy-bias checks, real-vs-synthetic ranking agreement, human-review sample + κ |
| `cache_eval.py` | none / exact / semantic cache with anchor guard + fingerprint invalidation; staleness experiment |
| `profiling.py`, `sweeps.py` | concurrency load tests; parameter sweeps and one-at-a-time ablations |
| `security.py` | prompt-injection scan/strip/fence; also now used by `BaseRAG.generate` in the LangChain stack |
| `registry.py`, `report.py`, `study.py`, `cli.py` | experiment records, Markdown/HTML/JSON/PDF reports, orchestration, CLI |

## Design decisions worth defending to a panel

* **No peeking.** Queries are split train/val/test by qid. Router training uses *train*; abstention thresholds are
  calibrated on *val*; leaderboard, router comparison and diagnosis report on *test*. `evaluate_router` raises if
  train and test overlap, and picks the "best fixed architecture" from train only.
* **Routers must earn their claim.** "Router wins" requires the paired-bootstrap CI of (router − best fixed) to exclude 0.
  The router deviates from the default architecture only when a training query shows a clear utility margin, so it
  cannot fit tie-break noise.
* **Measured vs inferred.** Diagnosis marks rank-of-relevant-evidence and answer-in-context as *measured* (needs
  qrels/gold answers); causes such as "extraction error" are *inferred* with lower confidence. Without qrels the
  diagnosis says it cannot separate retrieval from ranking.
* **Class-balanced abstention calibration.** A benchmark that is mostly answerable otherwise pushes the threshold to
  "never abstain".
* **Cache guards.** The audit that re-runs every cache hit uncached found the classic failure — "When was *Kar Dor*
  born?" reusing the answer for "*Kar Mir*". Semantic hits now require identical anchors (names/numbers/codes), the same
  corpus/config/prompt fingerprint and the same answer requirements. Reports include harmful-hit counts.
* **The injection test is discriminating.** It uses a deliberately vulnerable instruction-following mock (100% hijacked
  without the defence, 0% with it); an earlier version used the extractive generator, which cannot be hijacked and so
  proved nothing.

## Roadmap checklist coverage (57 items)

✅ implemented **and tested offline** · 🟡 partial (reason given) · ⬜ existing code, not audited/changed here

| # | Item | Status | Where / note |
|---|---|:-:|---|
| A1 | Standardised architectures | ✅ | `Pipeline` contract; `LangChainAdapter` bridges existing `BaseRAG` (adapter untested without LangChain) |
| A2 | Config management | ✅ | spec dict snapshot per pipeline + registry |
| A3 | Dataset management | 🟡 | converters tested against HF schemas; live HF download untested (no network here) |
| A4 | Reproducible experiments | ✅ | IDs, config/dataset fingerprints, env+git+code hash, `reproduce` verifies 37 metrics |
| A5 | Baselines | ✅ | `bm25`, `vector-tfidf`; every architecture paired against both |
| A6 | Experiment history | ✅ | `list`, `load`, `compare` |
| B1 | Retrieval metrics | ✅ | Recall/Precision/Hit@k, MRR, nDCG, MAP |
| B2 | Answer metrics | 🟡 | EM + token-F1 ✅; **semantic similarity not implemented** (use `DenseEncoderRetriever`'s encoder to add it) |
| B3 | Faithfulness | 🟡 | lexical-support **proxy** here; the existing LLM judge (`module3`) is retained and still needs κ validation vs human labels (ROADMAP #3) — `stats.cohen_kappa` is ready |
| B4 | Context relevance | ✅ | id-based context precision, answer-in-context |
| B5 | Latency | ✅ | retrieve / rerank / generate / post-process / total, p50/p95/p99 |
| B6 | Statistical confidence | ✅ | CIs, n, paired tests, effect sizes, BH; `run_to_run_variability` for stochastic LLMs |
| B7 | Cost/resources | 🟡 | tokens and configurable $/1k-token ✅; CPU/GPU/memory not tracked |
| C1–C3 | Retrieval / ranking / generation failures | ✅ | `diagnose()` |
| C4 | Question-type classification | 🟡 | dataset labels + heuristic `query_features`; no trained factual/multi-hop/ambiguous classifier |
| C5 | Evidence traceability | 🟡 | chunk id, rank, score, text in every diagnosis; upload page numbers not wired |
| C6–C7 | Failure reports / statistics | ✅ | `diagnosis.csv`, `failure_stats_by_arch_qtype.csv`, heat-maps |
| D1 | Query-aware selection | ✅ | `RuleRouter`, `LearnedRouter`, held-out evaluation |
| D2 | Adaptive retrieval depth | ✅ | `adaptive_k` + ablation |
| D3 | Reranking optimisation | 🟡 | pool-depth sweep + ablation ✅; real cross-encoder needs model download (`CrossEncoderReranker` provided, untested) |
| D4 | Chunking optimisation | ✅ | chunk-size × overlap sweep, parent-child |
| D5 | Hybrid weights | ✅ | dense-weight sweep 0→1 |
| D6 | Confidence fallback | ✅ | `FallbackPipeline` (primary → alternative → abstain), tested; not part of the default study |
| D7 | Ablations | ✅ | `ablate()` paired + BH-corrected |
| E1 | Synthetic generation | 🟡 | offline template generator + converter for the repo's LLM-made `synthetic_dataset.json`; validation is generator-agnostic |
| E2–E4 | Quality, leakage, real-vs-synthetic | ✅ | `validate_synthetic`, `compare_real_vs_synthetic` |
| E5 | Human validation | 🟡 | sample CSV + κ scoring ready; **a human must label it** |
| E6 | Impact on rankings | ✅ | Kendall τ / Spearman ρ, top-1 agreement, distortion flag |
| F1–F4 | Upload, preprocessing, citations, multi-doc | ⬜ | existing `module5/document_upload.py`; not changed |
| F5 | Unanswerable questions | ✅ | SQuAD-2.0-style scoring, abstention P/R/F1, calibrated threshold |
| F6 | Document isolation | ⬜ | not audited |
| G1–G4 | Semantic cache, correctness, metrics, comparison | ✅ | `cache_eval.py` (existing Redis cache retained) |
| G5 | Latency profiling | ✅ | per-stage breakdown |
| G6 | Concurrency/load | 🟡 | in-process sweep ✅ (shares the GIL); HTTP load test is the existing `load_test.py` |
| H1–H3, H5 | Leaderboard, quality-vs-latency, CIs, heat-maps | ✅ | static plots in the report |
| H4 | Query-level comparison | 🟡 | per-query rows for all architectures in `results.csv`; no interactive viewer |
| H6 | Exportable reports | ✅ | MD, self-contained HTML, JSON, CSV, figures PDF |
| I1 | Input validation | ✅ | `validate_question` (API validation pre-existing) |
| I2 | Secure document handling | ⬜ | existing |
| I3 | Prompt-injection defences | ✅ | `security.py`, wired into `BaseRAG.generate` |
| I4 | Monitoring/logs | ⬜ | existing `observability.py` |
| I5 | Testing & CI | ✅ | 61 new tests; new CI job |
| I6 | Docs/deployment | ✅ | this file, README, Makefile |

## Using your existing LangChain architectures

*(Not runnable in the sandbox this was built in — LangChain/Chroma weren't installed — so treat as a documented pattern.)*

```python
from module1.data_preprocessing import get_embedding_model, load_vector_store, chunk_documents, load_msmarco_documents
from module2.rag_architectures import build_all_architectures
from rag_lab.architectures import LangChainAdapter, LLMGenerator
from rag_lab.evaluate import run_benchmark
from rag_lab.datasets import load_msmarco

emb = get_embedding_model(); vs = load_vector_store(emb)
chunks = chunk_documents(load_msmarco_documents())
archs = build_all_architectures(vectorstore=vs, chunks=chunks, llm_free_only=True)
pipes = {n: LangChainAdapter(a, n) for n, a in archs.items()}          # retrieval-only; add generator=LLMGenerator(...) for answers
ds = load_msmarco(max_queries=500)
# NB: your chunks are identified by content-hash (sha1 of text[:200]); rag_lab.textutil.content_id is identical,
# so build qrels with {content_id(passage.text) for relevant passages} to evaluate them against the cached labels.
df, recs = run_benchmark(pipes, ds.queries)
```

## Known limitations (also stated in every report)

* Default components are offline stand-ins (TF-IDF/LSA for embeddings, lexical cross-scorer for a cross-encoder,
  extractive generator for an LLM). They make the *framework* verifiable; they are not claims about production models.
* Fixture results show, e.g., that the lexical reranker *lowers* nDCG and that the router finds no significant gain —
  these illustrate that the framework reports nulls honestly; they say nothing about real corpora.
* In-process latencies are single-machine and GIL-bound under concurrency.
