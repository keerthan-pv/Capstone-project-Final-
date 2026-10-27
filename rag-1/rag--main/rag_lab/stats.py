"""Statistical comparison tools: paired bootstrap, permutation, effect sizes, kappa, BH."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sst


def mean_ci(values: Sequence[float], iters: int = 2000, conf: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    x = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    if x.size == 0:
        return (float("nan"),) * 3
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(iters, x.size))
    means = x[idx].mean(axis=1)
    a = (1 - conf) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(x.mean()), float(lo), float(hi)


def wilson_ci(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = sst.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], iters: int = 5000,
                          conf: float = 0.95, seed: int = 42) -> Dict[str, float]:
    """Bootstrap CI for mean(a-b) over queries (resampling *queries*, keeping pairs intact)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(iters, d.size))
    means = d[idx].mean(axis=1)
    al = (1 - conf) / 2
    lo, hi = np.quantile(means, [al, 1 - al])
    return {"mean_diff": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def paired_permutation_p(a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 42) -> float:
    """Two-sided sign-flip permutation test on paired differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= obs - 1e-15) + 1) / (n_perm + 1))


def wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    if not np.any(d != 0):
        return 1.0
    try:
        return float(sst.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired effect size: mean(diff)/sd(diff)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else (0.0 if d.mean() == 0 else float("inf") * np.sign(d.mean()))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.sort(np.asarray(a, float)), np.asarray(b, float)
    gt = np.searchsorted(a, b, side="left")            # number of a < b_j
    le = np.searchsorted(a, b, side="right")           # number of a <= b_j
    greater = (a.size - le).sum(); less = gt.sum()
    return float((greater - less) / (a.size * b.size))


def effect_label(dz: float) -> str:
    x = abs(dz)
    return "negligible" if x < 0.2 else "small" if x < 0.5 else "medium" if x < 0.8 else "large"


def benjamini_hochberg(p: Sequence[float]) -> List[float]:
    p = np.asarray(p, float); n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj.tolist()


def paired_comparison(long_df: pd.DataFrame, metric: str, baseline: str, arch_col: str = "arch",
                      qid_col: str = "qid", seed: int = 42) -> pd.DataFrame:
    """Every architecture vs `baseline`, paired by query, BH-corrected within the metric."""
    wide = long_df.pivot_table(index=qid_col, columns=arch_col, values=metric, aggfunc="mean")
    if baseline not in wide.columns:
        raise KeyError(f"baseline {baseline!r} not in results")
    rows = []
    for arch in wide.columns:
        if arch == baseline:
            continue
        pair = wide[[arch, baseline]].dropna()
        if len(pair) < 2:
            continue
        a, b = pair[arch].to_numpy(), pair[baseline].to_numpy()
        bs = paired_bootstrap_diff(a, b, seed=seed)
        dz = cohens_dz(a, b)
        rows.append({"metric": metric, "arch": arch, "baseline": baseline, "n_pairs": len(pair),
                     "mean_arch": a.mean(), "mean_baseline": b.mean(), **bs,
                     "p_permutation": paired_permutation_p(a, b, seed=seed), "p_wilcoxon": wilcoxon_p(a, b),
                     "cohens_dz": dz, "effect": effect_label(dz), "cliffs_delta": cliffs_delta(a, b)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj_bh"] = benjamini_hochberg(out["p_permutation"].tolist())
        out["significant"] = (out["p_adj_bh"] < 0.05) & ((out["ci_lo"] > 0) | (out["ci_hi"] < 0))
    return out


def cohen_kappa(a: Sequence, b: Sequence, weights: Optional[str] = None) -> float:
    """Cohen's κ for two raters (nominal; weights='linear'|'quadratic' for ordinal)."""
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        raise ValueError("raters must label the same non-empty set of items")
    cats = sorted(set(a) | set(b), key=lambda x: (str(type(x)), x))
    k = len(cats); ix = {c: i for i, c in enumerate(cats)}
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[ix[x], ix[y]] += 1
    m /= m.sum()
    exp = np.outer(m.sum(1), m.sum(0))
    if weights is None:
        w = 1 - np.eye(k)
    else:
        i, j = np.indices((k, k))
        w = np.abs(i - j) / max(1, k - 1)
        if weights == "quadratic":
            w = w ** 2
    denom = (w * exp).sum()
    if denom == 0:
        return 1.0 if (w * m).sum() == 0 else 0.0
    return float(1 - (w * m).sum() / denom)


def kappa_label(k: float) -> str:
    return ("poor" if k < 0 else "slight" if k < .2 else "fair" if k < .4 else "moderate" if k < .6
            else "substantial" if k < .8 else "almost perfect")


def rank_agreement(x: Dict[str, float], y: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(set(x) & set(y))
    if len(keys) < 3:
        return {"kendall_tau": float("nan"), "spearman_rho": float("nan"), "n": len(keys)}
    xv, yv = [x[k] for k in keys], [y[k] for k in keys]
    return {"kendall_tau": float(sst.kendalltau(xv, yv).statistic),
            "spearman_rho": float(sst.spearmanr(xv, yv).statistic), "n": len(keys)}


def run_to_run_variability(runs: List[Sequence[float]]) -> Dict[str, float]:
    """For non-deterministic generation: spread of per-run means and per-query instability."""
    arr = np.asarray(runs, float)                       # (R, Q)
    run_means = arr.mean(axis=1)
    return {"n_runs": int(arr.shape[0]), "mean_of_runs": float(run_means.mean()),
            "sd_between_runs": float(run_means.std(ddof=1)) if arr.shape[0] > 1 else 0.0,
            "mean_per_query_sd": float(arr.std(axis=0, ddof=1).mean()) if arr.shape[0] > 1 else 0.0}
