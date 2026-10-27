"""Common evaluation engine: one loop for every architecture, per-query rows + raw run records."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import metrics as M
from .grounding import citation_correct, lexical_faithfulness
from .types import Query, RunRecord

log = logging.getLogger(__name__)
CORRECT_F1 = 0.5


def is_correct(q: Query, rec: RunRecord) -> bool:
    qs = M.qa_scores(rec.answer.text, rec.answer.abstained, q.answers, q.answerable)
    return qs["f1"] >= CORRECT_F1 if q.answerable else bool(qs["f1"])


def row_for(q: Query, rec: RunRecord, k_metrics: int = 10, split: str = "") -> Dict[str, Any]:
    pool = [h.pid for h in rec.candidates]
    ctx = rec.context
    qs = M.qa_scores(rec.answer.text, rec.answer.abstained, q.answers, q.answerable)
    no_gold = q.answerable and not q.answers            # retrieval-only datasets (e.g. BEIR): QA metrics are undefined
    rm = M.retrieval_metrics(pool, q.relevant, k_metrics) if q.relevant else {}
    ctx_text = " ".join(h.text for h in ctx)
    row: Dict[str, Any] = {
        "arch": rec.arch, "qid": q.qid, "qtype": q.qtype, "source": q.source, "split": split,
        "answerable": q.answerable, "question": q.question, "gold": " | ".join(q.answers),
        "pred": rec.answer.text, "abstained": rec.answer.abstained, "confidence": rec.answer.confidence,
        "em": float("nan") if no_gold else qs["em"], "f1": float("nan") if no_gold else qs["f1"],
        "correct": float("nan") if no_gold else (float(qs["f1"] >= CORRECT_F1) if q.answerable else float(qs["f1"])),
        **rm,
        "context_precision": M.context_precision([h.pid for h in ctx], q.relevant) if q.relevant else None,
        "answer_in_context": float(M.answer_in_text(q.answers, ctx_text)) if q.answers else None,
        "faithfulness": lexical_faithfulness(rec.answer.text, [h.text for h in ctx]),
        "citation_correct": citation_correct(rec.answer.text, rec.answer.support, {h.chunk_id: h.text for h in ctx}),
        "first_rel_rank": M.first_relevant_rank(M.dedupe_ranking(pool), q.relevant) if q.relevant else None,
        "n_context": len(ctx), "cost_usd": rec.cost_usd, "tokens_in": rec.answer.tokens_in,
        "tokens_out": rec.answer.tokens_out, "error": rec.error or "", "cache": rec.cache,
    }
    for stage in ("retrieve_ms", "rerank_ms", "generate_ms", "postprocess_ms", "total_ms"):
        row[stage] = rec.timings_ms.get(stage, 0.0)
    return row


def run_architecture(pipe, queries: Sequence[Query], k_metrics: int = 10, split_of: Optional[Dict[str, str]] = None,
                     warmup: int = 2) -> Tuple[pd.DataFrame, List[RunRecord]]:
    """Run one architecture over `queries`. A short warm-up (not recorded) keeps first-call cost out of p95."""
    for q in list(queries)[:warmup]:
        pipe.run(q)
    rows, recs = [], []
    for q in queries:
        rec = pipe.run(q)
        recs.append(rec)
        rows.append(row_for(q, rec, k_metrics, (split_of or {}).get(q.qid, "")))
    return pd.DataFrame(rows), recs


def run_benchmark(pipes: Dict[str, Any], queries: Sequence[Query], k_metrics: int = 10,
                  split_of: Optional[Dict[str, str]] = None):
    frames, all_recs = [], {}
    for name, pipe in pipes.items():
        df, recs = run_architecture(pipe, queries, k_metrics, split_of)
        frames.append(df); all_recs[name] = recs
        log.info("%-28s n=%d  F1=%.3f  p50=%.1fms", name, len(df), df["f1"].mean(), df["total_ms"].median())
    return pd.concat(frames, ignore_index=True), all_recs


def calibrate_abstain_threshold(pipe, val_queries: Sequence[Query], grid=None) -> float:
    """Choose the abstention threshold on a HELD-OUT validation split.

    Objective = mean of (F1 on answerable) and (F1 on unanswerable), i.e. class-balanced, so a
    benchmark dominated by answerable questions cannot drive the threshold to "never abstain".
    If the split has only one class, plain mean F1 is used.
    """
    gen = pipe.generator
    if not hasattr(gen, "thr"):
        return float("nan")
    grid = grid if grid is not None else np.round(np.arange(0.25, 0.86, 0.05), 2)
    best_t, best_f = gen.thr, -1.0
    for t in grid:
        gen.thr = float(t)
        by = {True: [], False: []}
        for q in val_queries:
            r = pipe.run(q)
            by[q.answerable].append(M.qa_scores(r.answer.text, r.answer.abstained, q.answers, q.answerable)["f1"])
        groups = [float(np.mean(v)) for v in by.values() if v]
        f = float(np.mean(groups)) if groups else 0.0
        if f > best_f + 1e-9:
            best_t, best_f = float(t), f
    gen.thr = best_t
    if hasattr(pipe, "config"):
        pipe.config["abstain_threshold_calibrated"] = best_t
    return best_t


# ───────────────────────────── summaries ───────────────────────────────────
QUALITY_COLS = ["f1", "em", "correct", "ndcg@10", "mrr", "map", "recall@10", "hit@10", "precision@10",
                "context_precision", "answer_in_context", "faithfulness", "citation_correct"]


def leaderboard(df: pd.DataFrame, metrics: Sequence[str] = ("f1", "em", "ndcg@10", "mrr", "recall@10", "faithfulness"),
                seed: int = 42) -> pd.DataFrame:
    from .stats import mean_ci
    rows = []
    for arch, g in df.groupby("arch", sort=False):
        r: Dict[str, Any] = {"arch": arch, "n": len(g)}
        for m in metrics:
            if m in g and g[m].notna().any():
                mean, lo, hi = mean_ci(g[m].dropna().tolist(), seed=seed)
                r[m] = mean; r[f"{m}_lo"] = lo; r[f"{m}_hi"] = hi; r[f"{m}_n"] = int(g[m].notna().sum())
        lat = M.latency_summary(g["total_ms"])
        r.update({"lat_mean_ms": lat["mean"], "lat_p50_ms": lat["p50"], "lat_p95_ms": lat["p95"], "lat_p99_ms": lat["p99"],
                  "cost_per_query_usd": float(g["cost_usd"].mean()), "tokens_per_query": float((g["tokens_in"] + g["tokens_out"]).mean()),
                  "error_rate": float((g["error"] != "").mean())})
        ab = M.abstention_metrics(g["answerable"], g["abstained"])
        r.update({k: ab[k] for k in ("abstain_precision", "abstain_recall", "abstain_f1", "false_abstain_rate",
                                     "answered_unanswerable_rate")})
        r["ece"] = M.ece(g["confidence"].clip(0, 1), g["correct"])
        rows.append(r)
    return pd.DataFrame(rows)


def by_query_type(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    return df.pivot_table(index="arch", columns="qtype", values=metric, aggfunc="mean")


def latency_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["retrieve_ms", "rerank_ms", "generate_ms", "postprocess_ms", "total_ms"]
    g = df.groupby("arch", sort=False)[cols].mean()
    g["other_ms"] = (g["total_ms"] - g[cols[:-1]].sum(axis=1)).clip(lower=0)
    g["bottleneck"] = g[cols[:-1]].idxmax(axis=1).str.replace("_ms", "")
    return g.reset_index()
