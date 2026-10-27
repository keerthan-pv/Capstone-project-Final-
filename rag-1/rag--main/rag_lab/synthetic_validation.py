"""Synthetic-query generation + validation (relevance, answerability, duplication, leakage, ranking distortion)."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import answer_in_text
from .stats import cohen_kappa, kappa_label, rank_agreement, wilson_ci, mean_ci
from .textutil import jaccard, sentences, tokens
from .types import Passage, Query


# ───────────────────────────── generation ──────────────────────────────────
_PATTERNS = [
    (re.compile(r"^(?P<s>[A-Z][\w' ]+?) was born in (?P<a>\d{4})\b"), "When did {s} enter the world?", "a"),
    (re.compile(r"^(?P<s>[A-Z][\w' ]+?) was founded in (?P<a>\d{4}) by (?P<b>[A-Z][\w ]+?)\."), "Who established {s}?", "b"),
    (re.compile(r"^The (?P<s>[A-Z][\w ]+?) carries product code (?P<a>[A-Z]{2}-\d+)"), "What code identifies the {s}?", "a"),
    (re.compile(r"^(?P<s>[A-Z][\w' ]+?) is a river town"), None, None),
]


def template_generator(passages: Sequence[Passage], n: int = 100, seed: int = 0) -> List[Query]:
    """Offline question generator for the fixture's sentence styles. For real corpora use an LLM generator
    (see `queries_from_legacy_synthetic`) — the validation below is generator-agnostic."""
    rnd = np.random.default_rng(seed)
    out: List[Query] = []
    order = rnd.permutation(len(passages))
    for i in order:
        p = passages[int(i)]
        for sent in sentences(p.text):
            for rx, tmpl, key in _PATTERNS:
                m = rx.search(sent)
                if m and tmpl:
                    out.append(Query(qid=f"syn{len(out):04d}", question=tmpl.format(s=m.group("s")), answers=[m.group(key)],
                                     relevant={p.pid: 1}, qtype="synthetic", source="synthetic",
                                     meta={"source_pid": p.pid, "source_sentence": sent}))
                    break
        if len(out) >= n:
            break
    return out[:n]


def queries_from_legacy_synthetic(path: str, passages: Sequence[Passage]) -> List[Query]:
    """Convert the repo's data/synthetic_dataset.json ({question, ground_truth, query_type}) into Query objects.
    Source passages are recovered by locating the ground truth text in the corpus."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for i, d in enumerate(data):
        gt = d.get("ground_truth", "")
        src = next((p for p in passages if gt and gt.lower()[:80] in p.text.lower()), None)
        out.append(Query(qid=f"leg{i:05d}", question=d.get("question", ""), answers=[gt] if gt else [],
                         relevant={src.pid: 1} if src else {}, qtype=d.get("query_type", "synthetic"), source="synthetic",
                         meta={"source_pid": src.pid if src else None, "source_sentence": ""}))
    return out


# ───────────────────────────── validation ──────────────────────────────────
def validate_synthetic(queries: Sequence[Query], passages: Sequence[Passage], real_queries: Sequence[Query] = (),
                       dup_threshold: float = 0.8, copy_threshold: float = 0.85) -> pd.DataFrame:
    """Per-query checks. A query is `valid` only if every check passes."""
    ptext = {p.pid: p.text for p in passages}
    real_tok = [(q.qid, set(tokens(q.question))) for q in real_queries]
    seen: List[tuple] = []
    rows = []
    for q in queries:
        qt = set(tokens(q.question))
        src_pid = q.meta.get("source_pid") or (next(iter(q.relevant)) if q.relevant else None)
        src_text = ptext.get(src_pid, "") if src_pid else ""
        sent = q.meta.get("source_sentence") or ""
        answerable_in_source = bool(q.answers) and answer_in_text(q.answers, src_text)
        answer_in_question = any(a and a.lower() in q.question.lower() for a in q.answers)
        copy_ratio = (len(qt & set(tokens(sent))) / max(1, len(qt))) if sent else float("nan")
        dup_of = next((qid for qid, s in seen if jaccard(qt, s) >= dup_threshold), None)
        contam = next((qid for qid, s in real_tok if jaccard(qt, s) >= dup_threshold), None)
        flags = {
            "source_supports_answer": answerable_in_source,
            "source_is_relevant": bool(src_pid and q.relevant.get(src_pid, 0) > 0),
            "no_answer_leak": not answer_in_question,
            "not_trivial_copy": not (copy_ratio == copy_ratio and copy_ratio >= copy_threshold),
            "not_duplicate": dup_of is None,
            "no_real_overlap": contam is None,
        }
        rows.append({"qid": q.qid, "question": q.question, "answer": " | ".join(q.answers), "source_pid": src_pid,
                     "copy_ratio": copy_ratio, "duplicate_of": dup_of, "overlaps_real": contam, **flags,
                     "valid": all(flags.values())})
        seen.append((q.qid, qt))
    return pd.DataFrame(rows)


def distinct_n(texts: Sequence[str], n: int) -> float:
    grams = [tuple(t[i:i + n]) for t in (tokens(x) for x in texts) for i in range(max(0, len(t) - n + 1))]
    return len(set(grams)) / len(grams) if grams else 0.0


def synthetic_report(val: pd.DataFrame, queries: Sequence[Query]) -> Dict[str, Any]:
    n = len(val)
    checks = ["source_supports_answer", "source_is_relevant", "no_answer_leak", "not_trivial_copy", "not_duplicate", "no_real_overlap"]
    lo, hi = wilson_ci(int(val["valid"].sum()), n)
    types = Counter(q.qtype for q in queries)
    p = np.array(list(types.values()), float) / max(1, sum(types.values()))
    return {
        "n": n, "valid_rate": float(val["valid"].mean()), "valid_rate_ci95": [lo, hi],
        "check_pass_rates": {c: float(val[c].mean()) for c in checks},
        "duplicate_rate": float(1 - val["not_duplicate"].mean()), "leak_rate": float(1 - val["no_answer_leak"].mean()),
        "real_overlap_rate": float(1 - val["no_real_overlap"].mean()),
        "distinct_1": distinct_n([q.question for q in queries], 1), "distinct_2": distinct_n([q.question for q in queries], 2),
        "type_entropy_bits": float(abs(-(p * np.log2(p)).sum())) if len(p) else 0.0,
    }


def compare_real_vs_synthetic(real_df: pd.DataFrame, syn_df: pd.DataFrame, metric: str = "f1",
                              tau_floor: float = 0.7) -> Dict[str, Any]:
    """Does the synthetic set rank architectures like the real set does? (Kendall τ / Spearman ρ, mean shift, top-1 agreement.)"""
    r = real_df.groupby("arch")[metric].mean(); s = syn_df.groupby("arch")[metric].mean()
    agree = rank_agreement(r.to_dict(), s.to_dict())
    tie_r, tie_s = r.nunique() <= 1, s.nunique() <= 1
    both = pd.DataFrame({"real": r, "synthetic": s}).dropna()
    both["delta"] = both["synthetic"] - both["real"]
    rm, rlo, rhi = mean_ci(real_df[metric].dropna().tolist()); sm, slo, shi = mean_ci(syn_df[metric].dropna().tolist())
    return {**agree, "metric": metric, "top1_real": str(r.idxmax()), "top1_synthetic": str(s.idxmax()),
            "top1_agrees": bool(r.idxmax() == s.idxmax()), "mean_shift": float(both["delta"].mean()),
            "mean_abs_shift": float(both["delta"].abs().mean()), "real_mean_ci": [rm, rlo, rhi], "synthetic_mean_ci": [sm, slo, shi],
            "distorts_ranking": bool(agree["kendall_tau"] == agree["kendall_tau"] and agree["kendall_tau"] < tau_floor),
            "rankings_degenerate": bool(tie_r or tie_s), "table": both.round(4).reset_index().to_dict("records")}


def human_validation_sample(val: pd.DataFrame, passages: Sequence[Passage], n: int = 30, seed: int = 0,
                            path: Optional[str] = None) -> pd.DataFrame:
    """Stratified (valid / invalid) sample for manual review. Fill `human_valid` with 1/0 and call score_human_labels()."""
    ptext = {p.pid: p.text for p in passages}
    rng = np.random.default_rng(seed)
    parts = []
    for flag, g in val.groupby("valid"):
        take = min(len(g), max(1, n // 2))
        parts.append(g.iloc[rng.permutation(len(g))[:take]])
    s = pd.concat(parts).copy()
    s["source_text"] = s["source_pid"].map(lambda x: ptext.get(x, "")[:600])
    s["auto_valid"] = s["valid"].astype(int); s["human_valid"] = ""; s["human_notes"] = ""
    s = s[["qid", "question", "answer", "source_text", "auto_valid", "human_valid", "human_notes"]]
    if path:
        s.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    return s


def score_human_labels(path_or_df) -> Dict[str, Any]:
    df = pd.read_csv(path_or_df) if isinstance(path_or_df, str) else path_or_df.copy()
    df = df[df["human_valid"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    if df.empty:
        return {"n_labelled": 0}
    h = df["human_valid"].astype(float).astype(int).tolist(); a = df["auto_valid"].astype(int).tolist()
    k = cohen_kappa(h, a)
    lo, hi = wilson_ci(sum(h), len(h))
    return {"n_labelled": len(df), "human_valid_rate": sum(h) / len(h), "human_valid_rate_ci95": [lo, hi],
            "cohen_kappa_auto_vs_human": k, "kappa_interpretation": kappa_label(k),
            "raw_agreement": float(np.mean([x == y for x, y in zip(h, a)]))}
