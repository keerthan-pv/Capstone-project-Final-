"""Command line: fetch datasets, run a full study, list/verify experiments, route a question."""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys

from . import datasets as D
from .registry import ExperimentStore, dumps


def _store(a) -> ExperimentStore:
    return ExperimentStore(os.path.join(a.out_dir, "experiments"))


def cmd_fetch(a) -> int:
    ok = 0
    for spec in a.datasets:
        try:
            if spec in ("squad", "squad_v2"):
                ds = D.load_squad_v2(max_queries=a.max_queries, refresh=a.refresh)
            elif spec in ("msmarco", "ms_marco"):
                ds = D.load_msmarco(max_queries=a.max_queries, refresh=a.refresh)
            elif spec.startswith("beir:"):
                ds = D.load_beir(spec.split(":", 1)[1], max_queries=a.max_queries, refresh=a.refresh)
            else:
                print(f"unknown dataset {spec!r} (use squad, msmarco, beir:<name>)"); continue
            print(f"✓ {spec}: {dumps(ds.summary(), indent=None)}"); ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"✗ {spec}: {type(e).__name__}: {e}\n  → needs `pip install datasets` and internet access to huggingface.co")
    return 0 if ok == len(a.datasets) else 1


def cmd_run(a) -> int:
    from .study import run_study
    cfg = {"dataset": a.dataset, "max_queries": a.max_queries, "seed": a.seed, "out_dir": a.out_dir,
           "run_sweeps": not a.fast, "run_cache": not a.fast, "run_load": not a.fast, "simulate_backend_ms": a.simulate_backend_ms}
    R = run_study(cfg, name=a.name)
    print("\nHEADLINE:", dumps(R["headline"], indent=None))
    for k, v in R["report_paths"].items():
        print(f"  {k:9s} {v}")
    return 0


def cmd_list(a) -> int:
    df = _store(a).list()
    print(df.to_string(index=False) if len(df) else "no experiments yet")
    return 0


def cmd_reproduce(a) -> int:
    from .study import rerun_leaderboard
    res = _store(a).verify_reproducibility(a.id, rerun_leaderboard)
    print(dumps(res)); return 0 if res["reproduced"] else 2


def cmd_route(a) -> int:
    from .router import RuleRouter
    d = _store(a).dir(a.exp)
    with open(os.path.join(d, "router.pkl"), "rb") as f:
        blob = pickle.load(f)
    rule = RuleRouter(blob["rule_available"])
    print("rule-based :", dumps(rule.explain(a.question), indent=None))
    print("learned    :", dumps(blob["learned"].explain(a.question)))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rag_lab", description="RAG benchmarking, diagnosis and adaptive optimisation")
    p.add_argument("--out-dir", default="outputs/lab"); p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download + cache datasets from Hugging Face")
    f.add_argument("datasets", nargs="+"); f.add_argument("--max-queries", type=int, default=2000); f.add_argument("--refresh", action="store_true")
    f.set_defaults(fn=cmd_fetch)
    r = sub.add_parser("run", help="run the full benchmarking study and write the report")
    r.add_argument("--dataset", default="fixture", help="fixture | squad | msmarco | beir:<name> | path/to/cached.json.gz")
    r.add_argument("--max-queries", type=int, default=240); r.add_argument("--seed", type=int, default=42)
    r.add_argument("--name", default="study"); r.add_argument("--fast", action="store_true", help="skip sweeps, cache and load tests")
    r.add_argument("--simulate-backend-ms", type=float, default=40.0)
    r.set_defaults(fn=cmd_run)
    sub.add_parser("list", help="list stored experiments").set_defaults(fn=cmd_list)
    rp = sub.add_parser("reproduce", help="re-run an experiment from its stored config and compare metrics")
    rp.add_argument("id"); rp.set_defaults(fn=cmd_reproduce)
    ro = sub.add_parser("route", help="recommend an architecture for a question using a stored experiment's router")
    ro.add_argument("question"); ro.add_argument("--exp", required=True); ro.set_defaults(fn=cmd_route)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
