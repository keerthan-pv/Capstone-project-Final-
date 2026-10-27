"""Experiment registry: every run is a self-describing, reloadable, verifiable record."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, pd.DataFrame):
            return o.to_dict("records")
        return str(o)


def dumps(obj: Any, **kw) -> str:
    def clean(x):
        if isinstance(x, float) and x != x:
            return None
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    return json.dumps(clean(obj), cls=_Enc, indent=kw.pop("indent", 2), **kw)


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(_PKG_DIR, "*.py"))):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def git_revision() -> Dict[str, Any]:
    root = os.path.dirname(_PKG_DIR)
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip())
        return {"commit": rev or "unknown", "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"commit": "unknown", "dirty": None}


def env_snapshot() -> Dict[str, Any]:
    import scipy
    import sklearn
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "git": git_revision(), "rag_lab_code_hash": code_hash(),
            "cpu_count": os.cpu_count()}


def config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(dumps(config, indent=None, sort_keys=True).encode()).hexdigest()[:8]


@dataclass
class Experiment:
    id: str
    name: str
    created_utc: str
    config: Dict[str, Any]
    dataset: Dict[str, Any]
    env: Dict[str, Any]
    seed: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    status: str = "created"


def new_experiment(name: str, config: Dict[str, Any], dataset_summary: Dict[str, Any], seed: int, notes: str = "") -> Experiment:
    now = datetime.now(timezone.utc)
    return Experiment(id=f"{now:%Y%m%d-%H%M%S}-{config_hash({**config, 'ds': dataset_summary.get('fingerprint')})}", name=name,
                      created_utc=now.isoformat(timespec="seconds"), config=config, dataset=dataset_summary,
                      env=env_snapshot(), seed=seed, notes=notes)


class ExperimentStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def dir(self, exp_id: str) -> str:
        return os.path.join(self.root, exp_id)

    def save(self, exp: Experiment, frames: Optional[Dict[str, pd.DataFrame]] = None, extra_json: Optional[Dict[str, Any]] = None) -> str:
        d = self.dir(exp.id); os.makedirs(d, exist_ok=True)
        for name, df in (frames or {}).items():
            path = os.path.join(d, f"{name}.csv"); df.to_csv(path, index=False); exp.artifacts[name] = path
        for name, obj in (extra_json or {}).items():
            path = os.path.join(d, f"{name}.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(dumps(obj))
            exp.artifacts[name] = path
        with open(os.path.join(d, "experiment.json"), "w", encoding="utf-8") as fh:
            fh.write(dumps(asdict(exp)))
        return d

    def load(self, exp_id: str) -> Experiment:
        with open(os.path.join(self.dir(exp_id), "experiment.json"), encoding="utf-8") as fh:
            return Experiment(**json.load(fh))

    def frame(self, exp_id: str, name: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.dir(exp_id), f"{name}.csv"))

    def list(self) -> pd.DataFrame:
        rows = []
        for p in sorted(glob.glob(os.path.join(self.root, "*", "experiment.json"))):
            with open(p, encoding="utf-8") as fh:
                e = json.load(fh)
            rows.append({"id": e["id"], "name": e["name"], "created_utc": e["created_utc"], "dataset": e["dataset"].get("name"),
                         "n_queries": e["dataset"].get("n_queries"), "status": e.get("status"), "headline": e.get("metrics", {}).get("headline")})
        return pd.DataFrame(rows)

    def compare(self, a: str, b: str, frame: str = "leaderboard", cols=("f1", "ndcg@10", "mrr")) -> pd.DataFrame:
        fa, fb = self.frame(a, frame).set_index("arch"), self.frame(b, frame).set_index("arch")
        out = pd.DataFrame(index=fa.index.intersection(fb.index))
        for c in cols:
            if c in fa and c in fb:
                out[f"{c}_A"], out[f"{c}_B"] = fa[c], fb[c]; out[f"{c}_delta"] = fb[c] - fa[c]
        return out

    def verify_reproducibility(self, exp_id: str, rerun: Callable[[Dict[str, Any]], pd.DataFrame], tol: float = 1e-9,
                               skip_cols=("lat", "ms", "cost", "tokens")) -> Dict[str, Any]:
        """Re-run from the stored config and compare every deterministic leaderboard metric (latency columns are excluded)."""
        exp = self.load(exp_id)
        old = self.frame(exp_id, "leaderboard").set_index("arch")
        new = rerun(exp.config).set_index("arch")
        diffs = {}
        for c in old.columns:
            if any(s in c for s in skip_cols) or c not in new or not np.issubdtype(old[c].dtype, np.number):
                continue
            d = (old[c] - new.reindex(old.index)[c]).abs().max()
            if d == d and d > tol:
                diffs[c] = float(d)
        return {"experiment": exp_id, "reproduced": not diffs, "n_metrics_compared": sum(
            1 for c in old.columns if not any(s in c for s in skip_cols) and c in new and np.issubdtype(old[c].dtype, np.number)),
                "differences": diffs}
