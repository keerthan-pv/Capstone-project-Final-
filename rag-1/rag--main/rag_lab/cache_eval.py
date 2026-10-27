"""Cache modes (none / exact / semantic) with correctness guards and honest metrics."""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from . import metrics as M
from .types import Answer, Query, RunRecord

_HV = HashingVectorizer(n_features=4096, alternate_sign=False, norm="l2", ngram_range=(1, 2), analyzer="word",
                        token_pattern=r"[a-z0-9]+", lowercase=True)


def norm_question(q: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", q.lower()))


_WH_START = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how", "in", "during", "is", "are", "was", "were",
             "do", "does", "did", "the", "a", "an", "name", "give", "tell"}


def anchors(question: str) -> frozenset:
    """Tokens that pin down WHAT is asked about: numbers/codes and capitalised names (not the sentence-initial word).
    Two questions may only share a cached answer if their anchors are identical — this is what stops
    "When was Kar Dor born?" from reusing the answer to "When was Kar Mir born?"."""
    out = set()
    for i, w in enumerate(question.replace("?", " ").split()):
        w2 = re.sub(r"^\W+|\W+$", "", w)
        if not w2:
            continue
        if any(ch.isdigit() for ch in w2) or (w2[0].isupper() and not (i == 0 and w2.lower() in _WH_START)):
            out.add(w2.lower())
    return frozenset(out)


def fingerprint(*parts: Any) -> str:
    """Anything that can change an answer must be in here: corpus version, pipeline config, model, prompt."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Entry:
    key: str
    vec: Any
    answer: Answer
    fingerprint: str
    requirements: str
    size: int
    anchors: frozenset = frozenset()


class QueryCache:
    """mode: 'none' | 'exact' | 'semantic'. `check_fingerprint=False` models a naive cache that ignores corpus changes."""

    def __init__(self, mode: str = "semantic", threshold: float = 0.85, max_entries: int = 1000, check_fingerprint: bool = True):
        assert mode in ("none", "exact", "semantic")
        self.mode, self.thr, self.max, self.check_fp = mode, threshold, max_entries, check_fingerprint
        self.entries: "OrderedDict[str, Entry]" = OrderedDict()
        self.hits = self.misses = self.evictions = 0
        self.lookup_ms: List[float] = []

    def lookup(self, question: str, fp: str, requirements: str = "") -> Optional[Entry]:
        t0 = time.perf_counter()
        found = None
        if self.mode != "none":
            nq = norm_question(question)
            qa = anchors(question)
            e = self.entries.get(nq)
            if e and (not self.check_fp or e.fingerprint == fp) and e.requirements == requirements:
                found = e
            elif self.mode == "semantic" and self.entries:
                qv = _HV.transform([nq])
                keys = list(self.entries)
                from scipy.sparse import vstack
                sims = (vstack([self.entries[k].vec for k in keys]) @ qv.T).toarray().ravel()
                for i in np.argsort(-sims):
                    if sims[i] < self.thr:
                        break
                    c = self.entries[keys[int(i)]]
                    if (not self.check_fp or c.fingerprint == fp) and c.requirements == requirements and c.anchors == qa:
                        found = c; break
            if found:
                self.entries.move_to_end(found.key)
        self.lookup_ms.append((time.perf_counter() - t0) * 1000)
        if found:
            self.hits += 1
        else:
            self.misses += 1
        return found

    def store(self, question: str, answer: Answer, fp: str, requirements: str = "") -> None:
        if self.mode == "none":
            return
        nq = norm_question(question)
        size = len(json.dumps({"q": nq, "a": answer.text, "s": answer.support}, default=str).encode())
        self.entries[nq] = Entry(nq, _HV.transform([nq]), answer, fp, requirements, size, anchors(question))
        self.entries.move_to_end(nq)
        while len(self.entries) > self.max:
            self.entries.popitem(last=False); self.evictions += 1

    def stats(self) -> Dict[str, Any]:
        tot = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": self.hits / tot if tot else 0.0,
                "miss_rate": self.misses / tot if tot else 0.0, "evictions": self.evictions,
                "storage_bytes": sum(e.size for e in self.entries.values()),
                "mean_lookup_ms": float(np.mean(self.lookup_ms)) if self.lookup_ms else 0.0}


class CachedPipeline:
    def __init__(self, pipe, cache: QueryCache, fp_fn: Callable[[], str], requirements: str = ""):
        self.pipe, self.cache, self.fp_fn, self.req = pipe, cache, fp_fn, requirements
        self.name = pipe.name
        self.config = dict(getattr(pipe, "config", {}), cache=cache.mode)

    def run(self, query) -> RunRecord:
        q = query if isinstance(query, Query) else Query("adhoc", str(query))
        t0 = time.perf_counter()
        fp = self.fp_fn()
        e = self.cache.lookup(q.question, fp, self.req)
        if e is not None:
            look = (time.perf_counter() - t0) * 1000
            return RunRecord(self.name, q.qid, q.question, [], [], e.answer, {"total_ms": look, "cache_lookup_ms": look}, 0.0, cache="hit")
        rec = self.pipe.run(q)
        if not rec.error:
            self.cache.store(q.question, rec.answer, fp, self.req)
        rec.timings_ms["total_ms"] = (time.perf_counter() - t0) * 1000
        rec.cache = "miss" if self.cache.mode != "none" else "off"
        return rec


def perturb_query(q: str, rnd: np.random.Generator) -> str:
    """Meaning-preserving surface change for datasets without human paraphrases."""
    toks = q.rstrip("?").split()
    if len(toks) > 4:
        drop = [i for i, t in enumerate(toks) if t.lower() in {"the", "a", "an", "of", "in", "did", "does", "was", "is"}]
        if drop:
            toks.pop(int(rnd.choice(drop)))
    return " ".join(toks) + "?"


def build_workload(queries: Sequence[Query], repeats: int = 1, seed: int = 0) -> List[Query]:
    """original → (paraphrase | perturbed) → original again, interleaved: exercises exact AND semantic hits."""
    rnd = np.random.default_rng(seed)
    out: List[Query] = []
    for q in queries:
        out.append(q)
        for _ in range(repeats):
            para = rnd.choice(q.paraphrases) if q.paraphrases else perturb_query(q.question, rnd)
            out.append(Query(q.qid, str(para), q.answers, q.relevant, q.answerable, q.qtype, q.source, [], {"variant": "paraphrase"}))
            out.append(Query(q.qid, q.question, q.answers, q.relevant, q.answerable, q.qtype, q.source, [], {"variant": "repeat"}))
    order = rnd.permutation(len(out))
    # keep each original before its variants: stable sort by (original index) within shuffled blocks
    return [out[i] for i in sorted(order[: len(out)], key=lambda i: (i // (2 * repeats + 1), i))]


class _SlowBackend:
    """Adds a fixed delay to every uncached run — SIMULATES an LLM-backed generation step (labelled in reports)."""

    def __init__(self, pipe, ms: float):
        self.pipe, self.ms, self.name, self.config = pipe, ms, pipe.name, dict(getattr(pipe, "config", {}), simulated_backend_ms=ms)

    def run(self, q):
        time.sleep(self.ms / 1000.0)
        rec = self.pipe.run(q)
        rec.timings_ms["total_ms"] = rec.timings_ms.get("total_ms", 0.0) + self.ms
        return rec


def evaluate_cache_modes(pipe, workload: Sequence[Query], fp: str, modes=("none", "exact", "semantic"),
                         threshold: float = 0.85, audit: bool = True, simulate_backend_ms: float = 0.0) -> pd.DataFrame:
    """Same workload under each mode. Audit re-runs each hit uncached to count HARMFUL hits (cached answer worse than fresh)."""
    rows = []
    base_lat = None
    served = _SlowBackend(pipe, simulate_backend_ms) if simulate_backend_ms > 0 else pipe
    for mode in modes:
        cache = QueryCache(mode, threshold)
        cp = CachedPipeline(served, cache, lambda: fp)
        lat, f1s, harmful, n_hits = [], [], 0, 0
        for q in workload:
            rec = cp.run(q)
            lat.append(rec.timings_ms["total_ms"])
            f = M.qa_scores(rec.answer.text, rec.answer.abstained, q.answers, q.answerable)["f1"]
            f1s.append(f)
            if rec.cache == "hit":
                n_hits += 1
                if audit:
                    fresh = pipe.run(q)
                    ff = M.qa_scores(fresh.answer.text, fresh.answer.abstained, q.answers, q.answerable)["f1"]
                    harmful += int(f < ff - 1e-9)
        st = cache.stats()
        row = {"mode": mode, "n_requests": len(workload), "simulated_backend_ms": simulate_backend_ms, **st, "mean_latency_ms": float(np.mean(lat)),
               "p95_latency_ms": float(np.percentile(lat, 95)), "f1": float(np.mean(f1s)),
               "harmful_hits": harmful, "harmful_hit_rate": harmful / n_hits if n_hits else 0.0}
        if mode == "none":
            base_lat = row["mean_latency_ms"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if base_lat is not None:
        df["latency_saved_ms_total"] = (base_lat - df["mean_latency_ms"]) * df["n_requests"]
        df["latency_reduction_pct"] = 100 * (base_lat - df["mean_latency_ms"]) / base_lat
    return df


def stale_answer_experiment(dataset, spec: Dict[str, Any], n_updates: int = 15, seed: int = 0) -> Dict[str, Any]:
    """Update facts in the corpus AFTER caching. A cache keyed only on the question serves stale answers;
    a fingerprinted cache invalidates and recomputes."""
    from .architectures import IndexBundle, build_pipeline
    from .types import Passage
    rnd = np.random.default_rng(seed)
    targets = [q for q in dataset.queries if q.answerable and q.qtype == "factual" and re.fullmatch(r"\d{4}", (q.answers or [""])[0])]
    targets = [targets[i] for i in rnd.permutation(len(targets))[:n_updates]]
    old = IndexBundle(dataset.passages); pipe_old = build_pipeline(spec, old)
    new_answers: Dict[str, str] = {}
    new_passages = []
    changed = {next(iter(q.relevant)): q for q in targets}
    for p in dataset.passages:
        if p.pid in changed:
            q = changed[p.pid]; old_y = q.answers[0]; new_y = str(int(old_y) + 1)
            new_passages.append(Passage(p.pid, p.text.replace(old_y, new_y), p.title, p.meta)); new_answers[q.qid] = new_y
        else:
            new_passages.append(p)
    pipe_new = build_pipeline(spec, IndexBundle(new_passages))
    out = {}
    for label, check in (("naive_cache(no_invalidation)", False), ("fingerprinted_cache", True)):
        cache = QueryCache("exact", check_fingerprint=check)
        state = {"fp": "corpus-v1"}
        cur = {"pipe": pipe_old}
        cp = CachedPipeline(type("P", (), {"name": "p", "run": lambda self, q: cur["pipe"].run(q), "config": {}})(), cache, lambda: state["fp"])
        for q in targets:
            cp.run(q)                                           # warm cache on corpus v1
        state["fp"] = "corpus-v2"; cur["pipe"] = pipe_new       # documents change
        stale = 0
        for q in targets:
            rec = cp.run(q)
            stale += int(M.normalize_answer(rec.answer.text) != M.normalize_answer(new_answers[q.qid]))
        out[label] = {"n": len(targets), "stale_answers": stale, "stale_answer_rate": stale / max(1, len(targets)), **cache.stats()}
    return out
