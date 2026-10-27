"""Matplotlib figures (Agg backend) used by the HTML/PDF report and the CLI."""
from __future__ import annotations

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .pareto import pareto_front  # noqa: E402


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def leaderboard_ci(lb: pd.DataFrame, metric: str, path: str) -> str:
    d = lb.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 1.5))
    err = np.vstack([d[metric] - d[f"{metric}_lo"], d[f"{metric}_hi"] - d[metric]])
    ax.barh(d["arch"], d[metric], xerr=err, color="#4c78a8", capsize=3)
    for y, (v, n) in enumerate(zip(d[metric], d["n"])):
        ax.text(v + 0.005, y, f"n={n}", va="center", fontsize=7)
    ax.set_xlabel(f"{metric} (mean, 95% bootstrap CI)"); ax.set_title(f"Architecture leaderboard — {metric}")
    return _save(fig, path)


def quality_vs_latency(lb: pd.DataFrame, path: str, quality: str = "f1", cost: str = "lat_p50_ms") -> str:
    front = pareto_front(lb, quality, cost)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(lb[cost], lb[quality], c=np.where(front, "#e45756", "#9aa5b1"), s=60, zorder=3)
    offsets = [(5, 6), (5, -11), (-5, 14), (5, -24), (-5, -30), (5, 20)]
    for i, (_, r) in enumerate(lb.sort_values([quality, cost]).iterrows()):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(r["arch"], (r[cost], r[quality]), fontsize=7, xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", arrowprops=dict(arrowstyle="-", color="#c0c8d0", lw=.5))
    span = max(lb[quality].max() - lb[quality].min(), 0.05)
    ax.set_ylim(max(0, lb[quality].min() - 2 * span), min(1.0, lb[quality].max() + span))   # honest scale: don't blow up tiny gaps
    f = lb[front].sort_values(cost)
    ax.step(f[cost], f[quality], where="post", color="#e45756", alpha=.6, label="Pareto frontier")
    ax.set_xlabel("p50 latency (ms)"); ax.set_ylabel(quality); ax.set_title("Quality vs latency (red = Pareto-optimal)")
    ax.legend(); ax.grid(alpha=.3)
    return _save(fig, path)


def failure_heatmap_plot(hm: pd.DataFrame, path: str, title: str = "Failure rate by architecture and category") -> str:
    fig, ax = plt.subplots(figsize=(1.2 * max(4, hm.shape[1]) + 2, 0.45 * hm.shape[0] + 2))
    im = ax.imshow(hm.to_numpy(), cmap="Reds", vmin=0, aspect="auto")
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns, rotation=30, ha="right")
    ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index, fontsize=8)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, f"{hm.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="share of queries"); ax.set_title(title)
    return _save(fig, path)


def latency_stack(bd: pd.DataFrame, path: str) -> str:
    cols = ["retrieve_ms", "rerank_ms", "generate_ms", "postprocess_ms", "other_ms"]
    d = bd.set_index("arch")[cols]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 1.5))
    left = np.zeros(len(d))
    for c, col in zip(cols, ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#bab0ac"]):
        ax.barh(d.index, d[c], left=left, label=c.replace("_ms", ""), color=col); left += d[c].to_numpy()
    ax.set_xlabel("mean latency (ms)"); ax.set_title("Per-stage latency breakdown"); ax.legend(fontsize=7)
    return _save(fig, path)


def reliability_diagram(conf, correct, path: str, bins: int = 10) -> str:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration"); ax.plot(xs, ys, "o-", label="observed")
    ax.set_xlabel("stated confidence"); ax.set_ylabel("observed accuracy"); ax.legend(); ax.set_title("Reliability diagram")
    return _save(fig, path)


def router_comparison(res: Dict, path: str, quality: str = "f1") -> str:
    labels = ["uniform random", "best fixed", "router", "oracle"]
    vals = [res[f"uniform_random_{quality}"], res[f"best_fixed_{quality}"], res[f"router_{quality}"], res[f"oracle_{quality}"]]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(labels, vals, color=["#bab0ac", "#4c78a8", "#e45756", "#54a24b"])
    ax.set_ylim(0, 1); ax.set_ylabel(quality); ax.set_title("Router vs fixed architectures (held-out test)")
    for i, v in enumerate(vals):
        ax.text(i, v + .01, f"{v:.3f}", ha="center", fontsize=8)
    return _save(fig, path)


def sensitivity(df: pd.DataFrame, param: str, path: str, metrics=("f1", "ndcg@10"), latency: str = "lat_p50_ms") -> str:
    d = df.sort_values(param)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in metrics:
        if m in d:
            ax.plot(d[param].astype(str), d[m], "o-", label=m)
    ax.set_xlabel(param); ax.set_ylabel("quality"); ax2 = ax.twinx()
    ax2.plot(d[param].astype(str), d[latency], "s--", color="gray", label=latency); ax2.set_ylabel("latency (ms)")
    ax.legend(loc="lower left", fontsize=7); ax.set_title(f"Sensitivity to {param}")
    return _save(fig, path)


def cache_plot(cdf: pd.DataFrame, path: str) -> str:
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].bar(cdf["mode"], cdf["hit_rate"], color="#4c78a8"); axs[0].set_title("Cache hit rate"); axs[0].set_ylim(0, 1)
    axs[1].bar(cdf["mode"], cdf["mean_latency_ms"], color="#f58518"); axs[1].set_title("Mean end-to-end latency (ms)")
    return _save(fig, path)
