"""Evidence-based failure diagnosis.

Separates MEASURED facts (rank of relevant evidence, whether the gold answer is in the context)
from INFERRED causes, names the failing stage, states a confidence and recommends an intervention.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import metrics as M
from .evaluate import is_correct
from .grounding import lexical_faithfulness
from .types import Query, RunRecord

CATEGORIES = ["retrieval", "ranking", "context", "generation", "grounding", "abstention"]
RECOMMEND = {
    "retrieval": "Relevant evidence never reached the candidate pool: test hybrid (BM25+dense) search, query rewriting/fusion, "
                 "a larger candidate depth, or multi-hop retrieval if several documents are required.",
    "ranking": "Evidence was retrieved but ranked below the context cut-off: add/upgrade a reranker, raise top-k, or use adaptive-k.",
    "context": "Evidence document was retrieved but the answer text is absent from the assembled context: use larger chunks or "
               "parent-child retrieval, and de-duplicate the context.",
    "generation": "Sufficient evidence was in context but the answer is wrong/missing: review the prompt, answer-extraction step and "
                  "abstention threshold; try a stronger generator.",
    "grounding": "Answer contains claims not supported by the retrieved evidence: require citation/evidence checks and abstain on unsupported claims.",
    "abstention": "Unanswerable question was answered: calibrate the abstention threshold on validation data or add an answerability classifier.",
}


@dataclass
class Diagnosis:
    arch: str
    qid: str
    qtype: str
    correct: bool
    category: str                     # "ok" or one of CATEGORIES
    stage: str
    confidence: float
    basis: str                        # "measured" | "inferred"
    measured: Dict[str, Any] = field(default_factory=dict)
    inferred_cause: str = ""
    recommendation: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    perf_flag: bool = False
    bottleneck: str = ""

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["measured"] = str(d["measured"]); d["evidence"] = str(d["evidence"])
        return d


def _ev(hits, n=5):
    return [{"rank": h.rank, "pid": h.pid, "chunk_id": h.chunk_id, "score": round(h.score, 4), "text": h.text[:160]} for h in hits[:n]]


def diagnose(q: Query, rec: RunRecord, latency_budget_ms: Optional[float] = None) -> Diagnosis:
    ok = is_correct(q, rec)
    pool = M.dedupe_ranking([h.pid for h in rec.candidates])
    ctx_pids = [h.pid for h in rec.context]
    ctx_text = " ".join(h.text for h in rec.context)
    stages = {k: v for k, v in rec.timings_ms.items() if k != "total_ms"}
    d = Diagnosis(rec.arch, q.qid, q.qtype, ok, "ok", "none", 1.0, "measured")
    if latency_budget_ms and rec.timings_ms.get("total_ms", 0) > latency_budget_ms:
        d.perf_flag = True
        d.bottleneck = max(stages, key=stages.get).replace("_ms", "") if stages else ""
    if rec.error:
        d.category, d.stage, d.basis, d.confidence = "generation", "pipeline", "measured", 1.0
        d.inferred_cause = f"pipeline raised {rec.error}"; d.recommendation = "Fix the runtime error before interpreting quality."
        return d
    if ok:
        return d

    # ── unanswerable question answered ────────────────────────────────────
    if not q.answerable:
        d.category, d.stage, d.basis, d.confidence = "abstention", "generation", "measured", 0.9
        d.measured = {"answered": True, "answer": rec.answer.text, "confidence": round(rec.answer.confidence, 3),
                      "relevant_evidence_exists": False}
        sub = q.meta.get("subtype")
        d.inferred_cause = ("entity present but the asked attribute is missing — generator matched on the entity alone"
                            if sub == "attribute_missing" else
                            "no supporting evidence exists, yet the generator produced an answer from loosely related text"
                            if sub == "entity_absent" else
                            "generator answered without sufficient evidence (answerability not checked)")
        d.recommendation = RECOMMEND["abstention"]; d.evidence = _ev(rec.context, 3)
        return d

    # ── answerable question ───────────────────────────────────────────────
    if q.relevant:
        rel = {p for p, g in q.relevant.items() if g > 0}
        in_pool = [p for p in pool if p in rel]
        in_ctx = [p for p in ctx_pids if p in rel]
        d.measured = {"n_relevant": len(rel), "relevant_in_pool": len(set(in_pool)), "relevant_in_context": len(set(in_ctx)),
                      "pool_depth": len(pool), "context_k": len(rec.context),
                      "first_relevant_rank": M.first_relevant_rank(pool, rel)}
        if not in_pool:
            d.category, d.stage, d.confidence = "retrieval", "retrieve", 0.92
            d.inferred_cause = ("multi-document question: no required document retrieved" if len(rel) > 1 else
                                "relevant document absent from the candidate pool (vocabulary mismatch or index miss)")
            d.evidence = _ev(rec.candidates, 3)
        elif len(set(in_pool)) < len(rel):
            d.category, d.stage, d.confidence = "retrieval", "retrieve", 0.85
            d.inferred_cause = f"only {len(set(in_pool))}/{len(rel)} required documents retrieved (multi-hop gap)"
            d.evidence = _ev(rec.candidates, 5)
        elif len(set(in_ctx)) < len(rel):
            d.category, d.stage, d.confidence = "ranking", "rank/rerank", 0.9
            d.inferred_cause = f"relevant evidence ranked {M.first_relevant_rank(pool, rel)} but context keeps top {len(rec.context)}"
            d.evidence = _ev(rec.candidates, 8)
        elif q.answers and not M.answer_in_text(q.answers, ctx_text):
            d.category, d.stage, d.confidence = "context", "context-assembly", 0.8
            d.measured["answer_in_context"] = False
            d.inferred_cause = "relevant document retrieved but the gold answer text is not in the context chunks (chunk boundary / truncation)"
            d.evidence = _ev(rec.context, 3)
        else:
            _generation_branch(d, q, rec)
    else:                                                    # no qrels: weaker, inferred-only diagnosis
        d.basis = "inferred"
        if q.answers and not M.answer_in_text(q.answers, ctx_text):
            d.category, d.stage, d.confidence = "retrieval", "retrieve/rank", 0.5
            d.measured = {"answer_in_context": False, "qrels": "unavailable"}
            d.inferred_cause = "gold answer not found in context; cannot separate retrieval from ranking without relevance labels"
            d.evidence = _ev(rec.context, 3)
        else:
            _generation_branch(d, q, rec)
    d.recommendation = RECOMMEND[d.category]
    return d


def _generation_branch(d: Diagnosis, q: Query, rec: RunRecord) -> None:
    d.stage = "generate"
    faith = lexical_faithfulness(rec.answer.text, [h.text for h in rec.context])
    d.measured["answer_in_context"] = True
    d.measured["faithfulness_proxy"] = None if faith is None else round(faith, 3)
    d.evidence = _ev(rec.context, 3)
    if rec.answer.abstained:
        d.category, d.confidence, d.basis = "generation", 0.85, "measured"
        d.inferred_cause = "evidence containing the answer was in context but the system abstained (over-abstention / threshold too strict)"
    elif faith is not None and faith < 0.5:
        d.category, d.confidence, d.basis = "grounding", 0.75, "inferred"
        d.inferred_cause = "answer not lexically supported by any retrieved chunk (possible hallucination; proxy check)"
    else:
        d.category, d.confidence, d.basis = "generation", 0.7, "inferred"
        d.inferred_cause = "evidence sufficient and answer supported, yet incorrect (extraction / synthesis error)"
    d.recommendation = RECOMMEND[d.category]


def diagnose_all(queries: Sequence[Query], records_by_arch: Dict[str, List[RunRecord]],
                 latency_budget_ms: Optional[float] = None) -> pd.DataFrame:
    qmap = {q.qid: q for q in queries}
    rows = [diagnose(qmap[r.qid], r, latency_budget_ms).to_row() for recs in records_by_arch.values() for r in recs if r.qid in qmap]
    return pd.DataFrame(rows)


def failure_stats(diag: pd.DataFrame, by: Sequence[str] = ("arch",)) -> pd.DataFrame:
    f = diag[~diag["correct"]]
    t = f.groupby(list(by) + ["category"]).size().rename("count").reset_index()
    tot = diag.groupby(list(by)).size().rename("n").reset_index()
    out = t.merge(tot, on=list(by))
    out["rate"] = out["count"] / out["n"]
    return out.sort_values("count", ascending=False)


def failure_heatmap(diag: pd.DataFrame, rows: str = "arch", cols: str = "category") -> pd.DataFrame:
    f = diag[~diag["correct"]]
    n = diag.groupby(rows).size()
    hm = f.pivot_table(index=rows, columns=cols, values="qid", aggfunc="count", fill_value=0)
    return hm.div(n, axis=0).fillna(0)


def intervention_effect(before: pd.DataFrame, after: pd.DataFrame, qid: str = "qid") -> Dict[str, Any]:
    """Compare two diagnosis frames over the SAME queries (e.g. baseline vs after adding a reranker)."""
    m = before[[qid, "correct", "category"]].merge(after[[qid, "correct", "category"]], on=qid, suffixes=("_before", "_after"))
    fixed = int((~m.correct_before & m.correct_after).sum()); broke = int((m.correct_before & ~m.correct_after).sum())
    still = int((~m.correct_before & ~m.correct_after).sum())
    trans = (m[~m.correct_before].groupby(["category_before", "category_after"]).size().rename("n").reset_index()
             if (~m.correct_before).any() else pd.DataFrame())
    return {"n": len(m), "fixed": fixed, "regressed": broke, "still_failing": still, "net_gain": fixed - broke,
            "acc_before": float(m.correct_before.mean()), "acc_after": float(m.correct_after.mean()), "transitions": trans}
