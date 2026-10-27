"""Query-aware architecture selection.

`RuleRouter`    — transparent hypotheses (keyword-heavy → BM25, multi-hop cue → iterative retrieval, …).
`LearnedRouter` — decision tree trained on per-query benchmark outcomes (train split only).
`evaluate_router` — compares routing against the best FIXED architecture and the oracle on HELD-OUT queries.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .stats import paired_bootstrap_diff
from .textutil import content_tokens, tokens
from .types import Query

_REL_NOUNS = {"founder", "author", "father", "mother", "spouse", "capital", "president", "director", "creator",
              "inventor", "founded", "born", "owner", "ceo", "leader", "teacher", "student", "successor"}
_CODE = re.compile(r"\b[A-Z0-9]{1,6}-\d{2,}\b|\b[A-Z]{2,}\d{2,}\b")
_WH = ["who", "when", "where", "what", "which", "how", "why"]
FEATURE_NAMES = ["n_tokens", "n_content", "avg_token_len", "n_digits", "has_year", "has_code", "has_quote", "cap_ratio",
                 "n_of", "has_relation_noun", "multihop_cue", "has_comparison", "n_conj", *[f"wh_{w}" for w in _WH], "wh_other"]


def query_features(question: str) -> Dict[str, float]:
    q = question.strip()
    toks, ct = tokens(q), content_tokens(q)
    words = q.split()
    caps = sum(1 for w in words[1:] if w[:1].isupper())
    n_of = len(re.findall(r"\bof\b", q.lower()))
    rel = any(t in _REL_NOUNS for t in toks)
    mh = float((n_of >= 1 and rel) or n_of >= 2 or bool(re.search(r"\b(whose|that|who)\b.+\b(of|by|from)\b", q.lower())))
    first = (toks[0] if toks else "")
    wh = {f"wh_{w}": float(first == w or (w in toks[:3])) for w in _WH}
    return {
        "n_tokens": len(toks), "n_content": len(ct), "avg_token_len": float(np.mean([len(t) for t in toks])) if toks else 0.0,
        "n_digits": sum(ch.isdigit() for ch in q), "has_year": float(bool(re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", q))),
        "has_code": float(bool(_CODE.search(q))), "has_quote": float('"' in q or "'" in q[1:-1] and False),
        "cap_ratio": caps / max(1, len(words) - 1), "n_of": n_of, "has_relation_noun": float(rel), "multihop_cue": mh,
        "has_comparison": float(bool(re.search(r"\b(compare|versus|vs\.?|difference|than|both)\b", q.lower()))),
        "n_conj": len(re.findall(r"\b(and|or|but)\b", q.lower())), **wh, "wh_other": float(not any(wh.values())),
    }


def feature_matrix(questions: Sequence[str]) -> np.ndarray:
    return np.array([[query_features(q)[n] for n in FEATURE_NAMES] for q in questions], dtype=float)


class RuleRouter:
    """Starting hypotheses, not facts — the benchmark decides whether they hold."""

    def __init__(self, available: Sequence[str], mapping: Optional[Dict[str, str]] = None):
        self.available = set(available)
        m = {"keyword": "bm25", "multihop": "multihop-hybrid", "long": "hybrid+rerank", "default": "hybrid-rrf"}
        m.update(mapping or {})
        self.mapping = {k: (v if v in self.available else next(iter(self.available))) for k, v in m.items()}

    def category(self, question: str) -> str:
        f = query_features(question)
        if f["multihop_cue"]:
            return "multihop"
        if f["has_code"] or f["has_quote"] or f["n_digits"] >= 4:
            return "keyword"
        if f["n_content"] >= 10:
            return "long"
        return "default"

    def route(self, question: str) -> str:
        return self.mapping[self.category(question)]

    def explain(self, question: str) -> Dict[str, Any]:
        c = self.category(question)
        why = {"multihop": "relational phrasing (e.g. 'the founder of X') suggests a bridge entity → iterative retrieval",
               "keyword": "rare identifiers / numbers favour exact lexical matching",
               "long": "long, information-rich query → rerank for passage ordering",
               "default": "short factual query → hybrid retrieval"}[c]
        return {"arch": self.mapping[c], "category": c, "reason": why}


def utility_table(df: pd.DataFrame, quality: str = "f1", lam_latency: float = 0.0, lam_cost: float = 0.0,
                  tie_eps: float = 1e-3) -> pd.DataFrame:
    """Per-query utility per architecture: quality − λ_lat·(latency/median) − λ_cost·(cost/median) − tiny tie-break by latency."""
    lat_ref = max(df["total_ms"].median(), 1e-6)
    cost_ref = df["cost_usd"].median() or 1.0
    u = df[quality] - lam_latency * df["total_ms"] / lat_ref - lam_cost * df["cost_usd"] / cost_ref - tie_eps * df["total_ms"] / lat_ref
    t = df.assign(_u=u).pivot_table(index="qid", columns="arch", values="_u", aggfunc="mean")
    return t


class LearnedRouter:
    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 5, seed: int = 42):
        from sklearn.tree import DecisionTreeClassifier
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)
        self.classes_: List[str] = []
        self.constant: Optional[str] = None
        self.train_leaf_stats: Dict[int, pd.DataFrame] = {}

    def fit(self, train_queries: Sequence[Query], results: pd.DataFrame, quality: str = "f1",
            lam_latency: float = 0.0, lam_cost: float = 0.0, margin: float = 0.05) -> "LearnedRouter":
        """Label each training query with its best architecture ONLY if it beats the default (best fixed
        architecture on the training split) by `margin`; otherwise label it with the default. This stops the
        tree from fitting tie-break noise when architectures perform identically."""
        res = results[results["qid"].isin({q.qid for q in train_queries})]
        U = utility_table(res, quality, lam_latency, lam_cost)
        self.default_arch = str(U.mean().idxmax())
        qs = [q for q in train_queries if q.qid in U.index]
        Usub = U.loc[[q.qid for q in qs]]
        best = Usub.idxmax(axis=1)
        gain = Usub.max(axis=1) - Usub[self.default_arch]
        y = np.where(gain > margin, best.to_numpy(), self.default_arch)
        X = feature_matrix([q.question for q in qs])
        if len(set(y)) == 1:
            self.constant = str(y[0])
        else:
            self.tree.fit(X, y)
            self.classes_ = list(self.tree.classes_)
            leaves = self.tree.apply(X)
            q2leaf = dict(zip([q.qid for q in qs], leaves))
            res = res.assign(leaf=res["qid"].map(q2leaf))
            for leaf, g in res.dropna(subset=["leaf"]).groupby("leaf"):
                self.train_leaf_stats[int(leaf)] = g.groupby("arch").agg(quality=(quality, "mean"), latency_ms=("total_ms", "mean"),
                                                                          n=("qid", "nunique"))
        self.quality = quality
        return self

    def route(self, question: str) -> str:
        if self.constant:
            return self.constant
        return str(self.tree.predict(feature_matrix([question]))[0])

    def explain(self, question: str) -> Dict[str, Any]:
        arch = self.route(question)
        out: Dict[str, Any] = {"arch": arch}
        if self.constant:
            out["reason"] = "all training queries preferred the same architecture"
            return out
        x = feature_matrix([question])
        path = self.tree.decision_path(x).indices
        t = self.tree.tree_
        why = []
        for node in path:
            if t.children_left[node] == t.children_right[node]:
                continue
            f, thr = FEATURE_NAMES[t.feature[node]], t.threshold[node]
            why.append(f"{f}={x[0, t.feature[node]]:.2f} {'≤' if x[0, t.feature[node]] <= thr else '>'} {thr:.2f}")
        out["reason"] = "; ".join(why) or "root"
        leaf = int(self.tree.apply(x)[0])
        st = self.train_leaf_stats.get(leaf)
        if st is not None:
            out["measured_tradeoffs_on_similar_training_queries"] = st.round(3).sort_values("quality", ascending=False).head(4).to_dict("index")
        return out

    def describe(self) -> str:
        if self.constant:
            return f"constant router → {self.constant}"
        from sklearn.tree import export_text
        return export_text(self.tree, feature_names=FEATURE_NAMES)


def evaluate_router(router, train_queries: Sequence[Query], test_queries: Sequence[Query], results: pd.DataFrame,
                    quality: str = "f1", seed: int = 42) -> Dict[str, Any]:
    """Held-out comparison using precomputed per-query results for every architecture."""
    tr_ids, te_ids = {q.qid for q in train_queries}, {q.qid for q in test_queries}
    if tr_ids & te_ids:
        raise ValueError("router train/test overlap — evaluation would be contaminated")
    train_res = results[results.qid.isin(tr_ids)]
    best_fixed = train_res.groupby("arch")[quality].mean().idxmax()          # chosen WITHOUT looking at test
    te = results[results.qid.isin(te_ids)]
    Q = te.pivot_table(index="qid", columns="arch", values=quality, aggfunc="mean")
    L = te.pivot_table(index="qid", columns="arch", values="total_ms", aggfunc="mean")
    qs = [q for q in test_queries if q.qid in Q.index]
    choice = pd.Series({q.qid: router.route(q.question) for q in qs})
    choice = choice.where(choice.isin(Q.columns), best_fixed)
    rq = np.array([Q.loc[qid, a] for qid, a in choice.items()])
    rl = np.array([L.loc[qid, a] for qid, a in choice.items()])
    bf = Q.loc[choice.index, best_fixed].to_numpy()
    oracle = Q.loc[choice.index].max(axis=1).to_numpy()
    diff = paired_bootstrap_diff(rq, bf, seed=seed)
    return {
        "n_test": len(rq), "best_fixed_arch(train-selected)": best_fixed,
        f"router_{quality}": float(rq.mean()), f"best_fixed_{quality}": float(bf.mean()), f"oracle_{quality}": float(oracle.mean()),
        f"uniform_random_{quality}": float(Q.loc[choice.index].mean(axis=1).mean()),
        "router_minus_best_fixed": diff, "router_beats_best_fixed_significantly": bool(diff["ci_lo"] > 0),
        "router_latency_ms": float(rl.mean()), "best_fixed_latency_ms": float(L.loc[choice.index, best_fixed].mean()),
        "oracle_gap_closed": float((rq.mean() - bf.mean()) / (oracle.mean() - bf.mean())) if oracle.mean() > bf.mean() else float("nan"),
        "choice_distribution": choice.value_counts().to_dict(),
    }
