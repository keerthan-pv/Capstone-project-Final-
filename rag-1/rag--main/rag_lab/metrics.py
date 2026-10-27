"""Retrieval, answer, abstention, latency and calibration metrics (all LLM-free)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .textutil import normalize_answer


# ───────────────────────────── answer quality ──────────────────────────────
def exact_match(pred: str, golds: Sequence[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds)) if golds else float(p == "")


def _f1_single(pred: str, gold: str) -> float:
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(pt), n / len(gt)
    return 2 * prec * rec / (prec + rec)


def token_f1(pred: str, golds: Sequence[str]) -> float:
    if not golds:
        return float(normalize_answer(pred) == "")
    return max(_f1_single(pred, g) for g in golds)


def answer_in_text(golds: Sequence[str], text: str) -> bool:
    t = normalize_answer(text)
    return any(normalize_answer(g) and normalize_answer(g) in t for g in golds)


def qa_scores(pred: str, abstained: bool, golds: Sequence[str], answerable: bool) -> Dict[str, float]:
    """SQuAD-2.0 convention: for unanswerable questions the only correct output is no answer."""
    if not answerable:
        ok = float(abstained or normalize_answer(pred) == "")
        return {"em": ok, "f1": ok}
    if abstained:
        return {"em": 0.0, "f1": 0.0}
    return {"em": exact_match(pred, golds), "f1": token_f1(pred, golds)}


# ───────────────────────────── retrieval ───────────────────────────────────
def dedupe_ranking(ids: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); out.append(i)
    return out


def first_relevant_rank(ranked: Sequence[str], relevant) -> Optional[int]:
    for i, d in enumerate(ranked, 1):
        if relevant.get(d, 0) > 0 if isinstance(relevant, dict) else d in relevant:
            return i
    return None


def _rel(relevant, d) -> float:
    return float(relevant.get(d, 0)) if isinstance(relevant, dict) else float(d in relevant)


def recall_at_k(ranked, relevant, k) -> Optional[float]:
    total = sum(1 for v in (relevant.values() if isinstance(relevant, dict) else relevant) if v)
    if total == 0:
        return None
    return sum(1 for d in ranked[:k] if _rel(relevant, d) > 0) / total


def precision_at_k(ranked, relevant, k) -> Optional[float]:
    if not relevant:
        return None
    return sum(1 for d in ranked[:k] if _rel(relevant, d) > 0) / max(1, k)


def hit_at_k(ranked, relevant, k) -> Optional[float]:
    if not relevant:
        return None
    return float(any(_rel(relevant, d) > 0 for d in ranked[:k]))


def mrr_at_k(ranked, relevant, k) -> Optional[float]:
    if not relevant:
        return None
    r = first_relevant_rank(ranked[:k], relevant)
    return 0.0 if r is None else 1.0 / r


def ndcg_at_k(ranked, relevant, k) -> Optional[float]:
    if not relevant:
        return None
    gains = [_rel(relevant, d) for d in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((float(v) for v in (relevant.values() if isinstance(relevant, dict) else [1] * len(relevant))),
                   reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked, relevant) -> Optional[float]:
    """AP over the supplied ranking, normalised by the number of relevant docs (MAP = mean of AP)."""
    total = sum(1 for v in (relevant.values() if isinstance(relevant, dict) else relevant) if v)
    if total == 0:
        return None
    hits, s = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if _rel(relevant, d) > 0:
            hits += 1
            s += hits / i
    return s / total


def retrieval_metrics(ranked_pids: Sequence[str], relevant, k: int = 10) -> Dict[str, Optional[float]]:
    r = dedupe_ranking(ranked_pids)
    return {
        f"recall@{k}": recall_at_k(r, relevant, k), f"precision@{k}": precision_at_k(r, relevant, k),
        f"hit@{k}": hit_at_k(r, relevant, k), "mrr": mrr_at_k(r, relevant, k),
        f"ndcg@{k}": ndcg_at_k(r, relevant, k), "map": average_precision(r, relevant),
    }


def context_precision(context_pids: Sequence[str], relevant) -> Optional[float]:
    """Share of context chunks that come from a relevant document (id-based, no LLM)."""
    if not relevant or not context_pids:
        return None if not relevant else 0.0
    return sum(_rel(relevant, d) > 0 for d in context_pids) / len(context_pids)


# ───────────────────────────── abstention ──────────────────────────────────
def abstention_metrics(answerable: Sequence[bool], abstained: Sequence[bool]) -> Dict[str, float]:
    a, b = np.asarray(answerable, bool), np.asarray(abstained, bool)
    tp = int((~a & b).sum()); fp = int((a & b).sum()); fn = int((~a & ~b).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) else float("nan")
    return {
        "abstain_precision": prec, "abstain_recall": rec, "abstain_f1": f1,
        "false_abstain_rate": float((a & b).sum() / a.sum()) if a.sum() else float("nan"),
        "answered_unanswerable_rate": float((~a & ~b).sum() / (~a).sum()) if (~a).sum() else float("nan"),
        "n_unanswerable": int((~a).sum()), "n_answerable": int(a.sum()),
    }


# ───────────────────────────── latency / calibration ───────────────────────
def latency_summary(ms: Iterable[float]) -> Dict[str, float]:
    x = np.asarray([v for v in ms if v is not None and not math.isnan(v)], float)
    if x.size == 0:
        return {k: float("nan") for k in ("mean", "p50", "p95", "p99", "max")}
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def ece(confidence: Sequence[float], correct: Sequence[float], bins: int = 10) -> float:
    """Expected calibration error: gap between stated confidence and observed accuracy."""
    c, y = np.asarray(confidence, float), np.asarray(correct, float)
    if c.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if m.any():
            total += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(total)


def brier(confidence: Sequence[float], correct: Sequence[float]) -> float:
    c, y = np.asarray(confidence, float), np.asarray(correct, float)
    return float(np.mean((c - y) ** 2)) if c.size else float("nan")
