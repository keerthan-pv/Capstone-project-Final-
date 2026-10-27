"""Quality-versus-cost trade-off analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_front(df: pd.DataFrame, quality: str = "f1", cost: str = "lat_p50_ms", eps: float = 0.0) -> pd.Series:
    """True for architectures that no other architecture beats on BOTH quality (higher) and cost (lower)."""
    q, c = df[quality].to_numpy(float), df[cost].to_numpy(float)
    dominated = np.zeros(len(df), bool)
    for i in range(len(df)):
        better_eq = (q >= q[i] - eps) & (c <= c[i] + eps)
        strictly = (q > q[i] + eps) | (c < c[i] - eps)
        dominated[i] = bool(np.any(better_eq & strictly & (np.arange(len(df)) != i)))
    return pd.Series(~dominated, index=df.index, name="on_pareto_front")


def weighted_ranking(lb: pd.DataFrame, w_quality: float = 0.6, w_speed: float = 0.3, w_cost: float = 0.1,
                     quality: str = "f1", latency: str = "lat_p50_ms", cost: str = "cost_per_query_usd") -> pd.DataFrame:
    """User-selectable preference: min–max normalise each axis, then weight. Lets 'quality / speed / cost' drive the ordering."""
    def nrm(s, invert=False):
        lo, hi = s.min(), s.max()
        z = (s - lo) / (hi - lo) if hi > lo else pd.Series(1.0, index=s.index)
        return 1 - z if invert else z
    out = lb.copy()
    tot = w_quality + w_speed + w_cost
    out["utility"] = (w_quality * nrm(out[quality]) + w_speed * nrm(out[latency], True) + w_cost * nrm(out[cost], True)) / tot
    return out.sort_values("utility", ascending=False)


def knee_point(lb: pd.DataFrame, quality: str = "f1", cost: str = "lat_p50_ms") -> str:
    """Pareto architecture with the best quality gain per added millisecond relative to the cheapest frontier point."""
    f = lb[pareto_front(lb, quality, cost)].sort_values(cost)
    if len(f) == 1:
        return str(f.iloc[0]["arch"])
    base = f.iloc[0]
    gain = (f[quality] - base[quality]) / (f[cost] - base[cost]).replace(0, np.nan)
    return str(f.loc[gain.idxmax(), "arch"]) if gain.notna().any() else str(base["arch"])
