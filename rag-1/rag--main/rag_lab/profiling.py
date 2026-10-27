"""Latency profiling and concurrent load tests for in-process callables."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Sequence

import pandas as pd

from .metrics import latency_summary


def load_test(fn: Callable, queries: Sequence, concurrency: int = 4, n_requests: int = 200) -> Dict[str, float]:
    """Fire `n_requests` at `fn` from `concurrency` threads. NB: pure-Python CPU work shares the GIL, so this measures
    contention/queueing for the in-process stack; load-test the HTTP API (load_test.py) for deployment numbers."""
    lat: List[float] = []; errors = 0

    def one(i):
        t0 = time.perf_counter()
        try:
            fn(queries[i % len(queries)])
            return (time.perf_counter() - t0) * 1000, 0
        except Exception:  # noqa: BLE001
            return (time.perf_counter() - t0) * 1000, 1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for ms_, err in ex.map(one, range(n_requests)):
            lat.append(ms_); errors += err
    wall = time.perf_counter() - t0
    s = latency_summary(lat)
    return {"concurrency": concurrency, "requests": n_requests, "errors": errors, "throughput_qps": n_requests / wall,
            "mean_ms": s["mean"], "p50_ms": s["p50"], "p95_ms": s["p95"], "p99_ms": s["p99"]}


def load_sweep(fn: Callable, queries: Sequence, concurrencies=(1, 2, 4, 8), n_requests: int = 200) -> pd.DataFrame:
    return pd.DataFrame([load_test(fn, queries, c, n_requests) for c in concurrencies])
