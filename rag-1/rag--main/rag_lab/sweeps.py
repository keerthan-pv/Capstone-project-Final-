"""Parameter sweeps and component ablations (all on a shared IndexBundle for fairness and speed)."""
from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from .architectures import IndexBundle, build_pipeline
from .evaluate import run_architecture, leaderboard
from .stats import paired_comparison
from .types import Query


def sweep(bundle: IndexBundle, base_spec: Dict[str, Any], grid: Dict[str, Sequence], queries: Sequence[Query],
          abstain_threshold: Optional[float] = None) -> pd.DataFrame:
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        spec = copy.deepcopy(base_spec)
        spec.update(dict(zip(keys, combo)))
        spec["name"] = "sweep:" + ",".join(f"{k}={v}" for k, v in zip(keys, combo))
        if abstain_threshold is not None:
            spec["abstain_threshold"] = abstain_threshold
        df, _ = run_architecture(build_pipeline(spec, bundle), queries)
        lb = leaderboard(df, metrics=("f1", "ndcg@10", "recall@10", "mrr")).iloc[0].to_dict()
        rows.append({**dict(zip(keys, combo)), **{k: lb[k] for k in ("n", "f1", "ndcg@10", "recall@10", "mrr", "lat_p50_ms", "lat_p95_ms",
                                                                    "cost_per_query_usd", "tokens_per_query") if k in lb}})
    return pd.DataFrame(rows)


# (applies?, transform) — a component is only ablated if the full system actually has it.
ABLATIONS: Dict[str, Any] = {
    "rerank": (lambda s: bool(s.get("rerank")), lambda s: {**s, "rerank": False}),
    "hybrid (→bm25 only)": (lambda s: s.get("retriever") == "hybrid", lambda s: {**s, "retriever": "bm25"}),
    "adaptive_k": (lambda s: bool(s.get("adaptive_k")), lambda s: {**s, "adaptive_k": None}),
    "multihop": (lambda s: bool(s.get("multihop")), lambda s: {**s, "multihop": False}),
}


def ablate(bundle: IndexBundle, full_spec: Dict[str, Any], queries: Sequence[Query], metric: str = "f1") -> pd.DataFrame:
    """Remove ONE component at a time; paired, BH-corrected comparison of each ablated variant against the full system."""
    full_spec = {**full_spec, "name": "FULL"}
    frames = [run_architecture(build_pipeline(full_spec, bundle), queries)[0]]
    for label, (applies, fn) in ABLATIONS.items():
        if not applies(full_spec):
            continue
        abl = fn(copy.deepcopy(full_spec))
        abl["name"] = f"−{label}"
        frames.append(run_architecture(build_pipeline(abl, bundle), queries)[0])
    df = pd.concat(frames, ignore_index=True)
    out = paired_comparison(df, metric, baseline="FULL")
    out = out.rename(columns={"arch": "ablation", "mean_diff": "delta_vs_full(ablated−full)"})
    return out
