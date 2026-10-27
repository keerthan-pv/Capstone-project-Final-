"""Common RAG contract: chunking → retrievers → reranker → generator → Pipeline.

Every architecture is a `Pipeline` exposing the same `.run(query) -> RunRecord`, so the evaluation
engine has no per-architecture logic. Retrievers return ranked *chunks*; `Hit.pid` is the parent
document id used by qrels.
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from . import security
from .textutil import approx_tokens, content_id, content_tokens, sentences, tokens
from .types import Answer, Hit, Passage, Query, RunRecord

ms = lambda t0: (time.perf_counter() - t0) * 1000.0  # noqa: E731


# ───────────────────────────── chunking ────────────────────────────────────
@dataclass
class Unit:
    chunk_id: str
    pid: str
    text: str


def chunk_words(text: str, size: int, overlap: int) -> List[str]:
    w = text.split()
    if size <= 0 or len(w) <= size:
        return [text]
    step = max(1, size - max(0, overlap))
    out = []
    for s in range(0, len(w), step):
        out.append(" ".join(w[s:s + size]))
        if s + size >= len(w):
            break
    return out


def build_units(passages: Sequence[Passage], chunk_size: int = 0, overlap: int = 0) -> List[Unit]:
    units = []
    for p in passages:
        for j, c in enumerate(chunk_words(p.text, chunk_size, overlap)):
            units.append(Unit(f"{p.pid}#{j}", p.pid, c))
    return units


# ───────────────────────────── retrievers ──────────────────────────────────
class Retriever:
    name = "retriever"
    units: List[Unit]

    def search(self, query: str, depth: int) -> Tuple[List[Hit], Dict[str, float]]:
        raise NotImplementedError

    def _hits(self, scores: np.ndarray, depth: int, stage: str) -> List[Hit]:
        depth = min(depth, len(scores))
        top = np.argpartition(-scores, depth - 1)[:depth] if depth < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top], kind="stable")]
        return [Hit(self.units[i].pid, self.units[i].chunk_id, self.units[i].text, float(scores[i]), r + 1, {stage: float(scores[i])})
                for r, i in enumerate(top) if scores[i] > 0]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, units: List[Unit], k1: float = 1.5, b: float = 0.75):
        self.units, self.k1, self.b = units, k1, b
        self.cv = CountVectorizer(analyzer=tokens)
        X = self.cv.fit_transform([u.text for u in units]).tocsr().astype(np.float64)
        N = X.shape[0]
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        dl = np.asarray(X.sum(axis=1)).ravel()
        avg = dl.mean() or 1.0
        X = X.tocoo()
        denom = X.data + k1 * (1 - b + b * dl[X.row] / avg)
        data = X.data * (k1 + 1) / denom * self.idf[X.col]
        from scipy.sparse import csc_matrix
        self.W = csc_matrix((data, (X.row, X.col)), shape=X.shape)
        self.vocab = self.cv.vocabulary_

    def search(self, query, depth):
        t0 = time.perf_counter()
        cols = [self.vocab[t] for t in tokens(query) if t in self.vocab]
        scores = np.asarray(self.W[:, cols].sum(axis=1)).ravel() if cols else np.zeros(len(self.units))
        return self._hits(scores, depth, "bm25"), {"retrieve_ms": ms(t0)}


class TfidfRetriever(Retriever):
    """Sparse-vector cosine search — the 'basic vector-search RAG' baseline."""
    name = "tfidf"

    def __init__(self, units: List[Unit]):
        self.units = units
        self.vec = TfidfVectorizer(analyzer=tokens, sublinear_tf=True)
        self.X = self.vec.fit_transform([u.text for u in units])

    def search(self, query, depth):
        t0 = time.perf_counter()
        scores = (self.X @ self.vec.transform([query]).T).toarray().ravel()
        return self._hits(scores, depth, "tfidf"), {"retrieve_ms": ms(t0)}


class LSARetriever(Retriever):
    """Latent-semantic dense retriever (offline stand-in for an embedding model)."""
    name = "lsa"

    def __init__(self, units: List[Unit], dims: int = 96, seed: int = 0):
        self.units = units
        self.vec = TfidfVectorizer(analyzer=tokens, sublinear_tf=True)
        X = self.vec.fit_transform([u.text for u in units])
        d = max(2, min(dims, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(d, random_state=seed)
        self.Z = normalize(self.svd.fit_transform(X))

    def search(self, query, depth):
        t0 = time.perf_counter()
        q = normalize(self.svd.transform(self.vec.transform([query])))
        scores = self.Z @ q.ravel()
        return self._hits(np.maximum(scores, 0), depth, "lsa"), {"retrieve_ms": ms(t0)}


class DenseEncoderRetriever(Retriever):
    """Plug in any real embedding model: ``encode(list[str]) -> np.ndarray`` (e.g. sentence-transformers)."""
    name = "dense"

    def __init__(self, units: List[Unit], encode: Callable[[List[str]], np.ndarray], name: str = "dense"):
        self.units, self.encode, self.name = units, encode, name
        self.Z = normalize(np.asarray(encode([u.text for u in units]), dtype=np.float32))

    def search(self, query, depth):
        t0 = time.perf_counter()
        q = normalize(np.asarray(self.encode([query]), dtype=np.float32))[0]
        return self._hits(np.maximum(self.Z @ q, 0), depth, "dense"), {"retrieve_ms": ms(t0)}


def _minmax(x: Dict[str, float]) -> Dict[str, float]:
    if not x:
        return x
    lo, hi = min(x.values()), max(x.values())
    return {k: (v - lo) / (hi - lo) if hi > lo else 1.0 for k, v in x.items()}


def fuse(rankings: List[Tuple[List[Hit], float]], method: str, depth: int, rrf_k: int = 60) -> List[Hit]:
    by_id: Dict[str, Hit] = {}
    agg: Dict[str, float] = defaultdict(float)
    if method == "rrf":
        for hits, w in rankings:
            for h in hits:
                by_id.setdefault(h.chunk_id, h); agg[h.chunk_id] += w / (rrf_k + h.rank)
    else:
        for hits, w in rankings:
            norm = _minmax({h.chunk_id: h.score for h in hits})
            for h in hits:
                by_id.setdefault(h.chunk_id, h); agg[h.chunk_id] += w * norm[h.chunk_id]
    order = sorted(agg, key=lambda k: -agg[k])[:depth]
    out = []
    for r, cid in enumerate(order, 1):
        h = by_id[cid]
        out.append(Hit(h.pid, h.chunk_id, h.text, agg[cid], r, {**h.stage_scores, f"fusion_{method}": agg[cid]}))
    return out


class HybridRetriever(Retriever):
    def __init__(self, parts: List[Tuple[Retriever, float]], method: str = "rrf", pool: int = 50):
        self.parts, self.method, self.pool = parts, method, pool
        self.units = parts[0][0].units
        self.name = "hybrid-" + method

    def search(self, query, depth):
        t0 = time.perf_counter()
        rankings = []
        for r, w in self.parts:
            h, _ = r.search(query, max(self.pool, depth)); rankings.append((h, w))
        return fuse(rankings, self.method, depth), {"retrieve_ms": ms(t0)}


class QueryVariantRetriever(Retriever):
    """Offline multi-query fusion: original + keyword-only variant, fused with RRF (no LLM)."""
    name = "fusion"

    def __init__(self, base: Retriever, pool: int = 50):
        self.base, self.pool, self.units = base, pool, base.units

    def search(self, query, depth):
        t0 = time.perf_counter()
        variants = [query, " ".join(content_tokens(query))]
        rankings = [(self.base.search(v, max(self.pool, depth))[0], 1.0) for v in dict.fromkeys(variants) if v.strip()]
        return fuse(rankings, "rrf", depth), {"retrieve_ms": ms(t0)}


class LexicalCrossScorer:
    """Offline stand-in for a cross-encoder: IDF-weighted coverage + bigram match + proximity."""
    name = "lexical-cross"

    def __init__(self, units: Sequence[Unit]):
        df: Dict[str, int] = defaultdict(int)
        for u in units:
            for t in set(tokens(u.text)):
                df[t] += 1
        n = len(units)
        self.idf = {t: math.log(1 + n / c) for t, c in df.items()}
        self.default_idf = math.log(1 + n)

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        qt = content_tokens(query)
        if not qt:
            return [0.0] * len(texts)
        qset = set(qt); qb = set(zip(qt, qt[1:]))
        wsum = sum(self.idf.get(t, self.default_idf) for t in qset)
        out = []
        for text in texts:
            tt = tokens(text); tset = set(tt)
            cov = sum(self.idf.get(t, self.default_idf) for t in qset & tset) / wsum
            tb = set(zip(tt, tt[1:]))
            big = len(qb & tb) / len(qb) if qb else 0.0
            pos = [i for i, t in enumerate(tt) if t in qset]
            prox = 0.0
            if pos:
                span = (pos[-1] - pos[0] + 1) if len(pos) > 1 else 1
                prox = min(1.0, len(set(tt[i] for i in pos)) / span)
            out.append(0.6 * cov + 0.25 * big + 0.15 * prox)
        return out


class CrossEncoderReranker:                                                    # pragma: no cover
    """Real cross-encoder (needs sentence-transformers + model download)."""

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.name, self.model = model, CrossEncoder(model)

    def score(self, query, texts):
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


class RerankRetriever(Retriever):
    def __init__(self, first: Retriever, reranker, pool: int = 30):
        self.first, self.reranker, self.pool, self.units = first, reranker, pool, first.units
        self.name = f"{first.name}+rerank"

    def search(self, query, depth):
        hits, tm = self.first.search(query, max(self.pool, depth))
        t0 = time.perf_counter()
        sc = self.reranker.score(query, [h.text for h in hits]) if hits else []
        order = sorted(range(len(hits)), key=lambda i: -sc[i])[:depth]
        out = [Hit(hits[i].pid, hits[i].chunk_id, hits[i].text, float(sc[i]), r + 1, {**hits[i].stage_scores, "rerank": float(sc[i])})
               for r, i in enumerate(order)]
        return out, {**tm, "rerank_ms": ms(t0)}


class ParentChildRetriever(Retriever):
    """Match small child chunks, return the whole parent passage."""

    def __init__(self, child: Retriever, parents: Dict[str, str]):
        self.child, self.parents, self.units = child, parents, child.units
        self.name = "parent-child"

    def search(self, query, depth):
        hits, tm = self.child.search(query, depth * 4)
        seen, out = set(), []
        for h in hits:
            if h.pid in seen:
                continue
            seen.add(h.pid)
            out.append(Hit(h.pid, f"{h.pid}#parent", self.parents[h.pid], h.score, len(out) + 1, h.stage_scores))
            if len(out) >= depth:
                break
        return out, tm


# ───────────────────────────── generators ──────────────────────────────────
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUM = re.compile(r"\b\d[\d,\.]*\b")
_CAP = re.compile(r"\b(?:[A-Z][\w'\-]+)(?:\s+[A-Z][\w'\-]+)*")
_LEAD = {"The", "A", "An", "Its", "It", "He", "She", "They", "In", "On", "At", "Later", "Several", "Local", "Visitors", "Trade", "Surviving"}


class ExtractiveGenerator:
    """Offline generator: pick the best-supported sentence, extract a span, abstain when unsupported.

    A deterministic stand-in for an LLM so the benchmark and diagnosis run offline; swap in
    :class:`LLMGenerator` for real answers.
    """
    name = "extractive"

    def __init__(self, idf: Dict[str, float], abstain_threshold: float = 0.55, default_idf: float = 5.0,
                 sanitize: bool = True):
        self.idf, self.thr, self.default_idf, self.sanitize = idf, abstain_threshold, default_idf, sanitize

    def _cov(self, qset, tset) -> float:
        if not qset:
            return 0.0
        w = sum(self.idf.get(t, self.default_idf) for t in qset)
        weighted = sum(self.idf.get(t, self.default_idf) for t in qset & tset) / w
        return 0.5 * weighted + 0.5 * len(qset & tset) / len(qset)

    def _span(self, q: str, sent: str) -> str:
        ql, qtok = q.lower(), set(tokens(q))
        if re.search(r"\b(when|what year|which year|in which year|during which year)\b", ql):
            m = [y for y in _YEAR.findall(sent) if y not in q]
            if m: return m[0]
        if re.search(r"\bhow (many|much)\b", ql):
            m = [n for n in _NUM.findall(sent) if n not in q]
            if m: return m[0]
        cands = []
        for m in _CAP.finditer(sent):
            words = m.group(0).split()
            while words and words[0] in _LEAD:
                words = words[1:]
            if words and not set(tokens(" ".join(words))) <= qtok:
                cands.append(" ".join(words))
        if re.search(r"\bwho\b", ql):
            multi = [c for c in cands if " " in c]
            if multi: return multi[0]
        if cands:
            return cands[0]
        return " ".join(sent.split()[:25])

    def generate(self, question: str, context: Sequence[Hit], support_query: Optional[str] = None) -> Answer:
        qset = set(content_tokens(support_query or question))
        best = (-1.0, "", "", 0)
        total_cov = 0.0
        for rank, h in enumerate(context):
            text = security.sanitize_context(h.text) if self.sanitize else h.text
            tset_all = set(tokens(text)); total_cov = max(total_cov, self._cov(qset, tset_all))
            for s in sentences(text):
                sc = self._cov(qset, set(tokens(s))) + 0.02 / (1 + rank)
                if sc > best[0]:
                    best = (sc, s, h.chunk_id, rank)
        tin = approx_tokens(question) + sum(approx_tokens(h.text) for h in context)
        if best[0] < 0 or best[0] < self.thr:
            return Answer("", True, confidence=max(0.0, best[0]), tokens_in=tin)
        span = self._span(question + (" " + support_query if support_query else ""), best[1])
        conf = min(1.0, 0.7 * best[0] + 0.3 * total_cov)
        return Answer(span, False, conf, [{"chunk_id": best[2], "sentence": best[1]}], tin, approx_tokens(span))


_ABSTAIN_RX = re.compile(r"(don't|do not|doesn't|does not)\s+(have|contain|provide)\s+(enough|sufficient)|not enough information|cannot (be )?answer", re.I)


class LLMGenerator:
    """Wrap any ``complete(prompt)->str`` callable (e.g. module2.rag_architectures._llm().complete)."""
    name = "llm"

    def __init__(self, complete: Callable[[str], str], sanitize_mode: str = "strip"):
        self.complete, self.mode = complete, sanitize_mode

    def generate(self, question: str, context: Sequence[Hit], support_query: Optional[str] = None) -> Answer:
        prompt = security.build_safe_prompt(question, [h.text for h in context], self.mode)
        out = (self.complete(prompt) or "").strip()
        ab = bool(_ABSTAIN_RX.search(out)) or not out
        return Answer("" if ab else out, ab, 0.0 if ab else 0.5, [], approx_tokens(prompt), approx_tokens(out))


# ───────────────────────────── pipeline ────────────────────────────────────
def adaptive_k(scores: Sequence[float], k_min: int = 2, k_max: int = 8, ratio: float = 0.6) -> int:
    """Pick context depth by score drop-off: keep hits scoring ≥ ratio × top score."""
    if not len(scores) or scores[0] <= 0:
        return k_min
    k = sum(1 for s in scores[:k_max] if s >= ratio * scores[0])
    return int(min(k_max, max(k_min, k)))


class Pipeline:
    def __init__(self, name: str, retriever: Retriever, generator, k: int = 5, depth: int = 20,
                 adaptive: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None,
                 price_in_per_1k: float = 0.0, price_out_per_1k: float = 0.0):
        self.name, self.retriever, self.generator = name, retriever, generator
        self.k, self.depth, self.adaptive = k, depth, adaptive
        self.config = dict(config or {})
        self.price_in, self.price_out = price_in_per_1k, price_out_per_1k

    def retrieve(self, question: str) -> Tuple[List[Hit], List[Hit], Dict[str, float]]:
        cands, tm = self.retriever.search(question, self.depth)
        k = adaptive_k([h.score for h in cands], **self.adaptive) if self.adaptive else self.k
        return cands, cands[:k], tm

    def run(self, query) -> RunRecord:
        q = query if isinstance(query, Query) else Query("adhoc", str(query))
        t0 = time.perf_counter()
        try:
            question = security.validate_question(q.question)
            cands, ctx, tm = self.retrieve(question)
            t1 = time.perf_counter()
            ans = self.generator.generate(question, ctx)
            gen = ms(t1)
            t2 = time.perf_counter()
            cost = ans.tokens_in / 1000 * self.price_in + ans.tokens_out / 1000 * self.price_out
            tm = {"retrieve_ms": tm.get("retrieve_ms", 0.0), "rerank_ms": tm.get("rerank_ms", 0.0),
                  "generate_ms": gen, "postprocess_ms": ms(t2)}
            tm["total_ms"] = ms(t0)
            return RunRecord(self.name, q.qid, q.question, cands, ctx, ans, tm, cost)
        except Exception as exc:  # noqa: BLE001 — one bad query must not abort a benchmark
            return RunRecord(self.name, q.qid, q.question, [], [], Answer("", True),
                             {"total_ms": ms(t0)}, 0.0, error=f"{type(exc).__name__}: {exc}")


class MultiHopPipeline(Pipeline):
    """Genuine iterative retrieval: hop-1 finds a bridge entity, hop-2 searches for facts about it.

    Unlike a single-pass retriever this can answer "In which city was the founder of X born?",
    where the answer lives in a document that never mentions X.
    """

    def _second_hop_query(self, question: str, ctx: Sequence[Hit]) -> Optional[str]:
        if not ctx:
            return None
        qtok = set(tokens(question)); qc = set(content_tokens(question))
        best, bs = "", -1.0
        for s in sentences(ctx[0].text):
            sc = len(qc & set(tokens(s))) / max(1, len(qc))
            if sc > bs:
                best, bs = s, sc
        ents = []
        for m in _CAP.finditer(best):
            words = m.group(0).split()
            while words and words[0] in _LEAD:
                words = words[1:]
            if words and not set(tokens(" ".join(words))) <= qtok:
                ents.append(" ".join(words))
        multi = [e for e in ents if " " in e]
        ent = (multi or ents or [None])[0]
        if not ent:
            return None
        sent_toks = set(tokens(best))
        rest = [t for t in content_tokens(question) if t not in sent_toks]
        return (ent + " " + " ".join(rest)).strip()

    @staticmethod
    def _bridge_entity(support: str, question: str) -> str:
        """Retrieval query for hop 2: the bridge entity only (extra words just pull in noise)."""
        ent = []
        for w in support.split():
            if w[:1].isupper():
                ent.append(w)
            else:
                break
        return " ".join(ent) or support

    def run(self, query) -> RunRecord:
        q = query if isinstance(query, Query) else Query("adhoc", str(query))
        t0 = time.perf_counter()
        try:
            question = security.validate_question(q.question)
            c1, tm1 = self.retriever.search(question, self.depth)
            ctx1 = c1[: self.k]
            support = self._second_hop_query(question, ctx1)
            tm = {"retrieve_ms": tm1.get("retrieve_ms", 0.0), "rerank_ms": tm1.get("rerank_ms", 0.0)}
            if support:
                c2, tm2 = self.retriever.search(self._bridge_entity(support, question), self.depth)
                tm["retrieve_ms"] += tm2.get("retrieve_ms", 0.0); tm["rerank_ms"] += tm2.get("rerank_ms", 0.0)
                cands = fuse([(c1, 1.0), (c2, 1.0)], "rrf", self.depth)
                half = max(1, self.k // 2)
                ctx = list(c2[:half])
                ctx += [h for h in ctx1 if h.chunk_id not in {x.chunk_id for x in ctx}][: max(1, self.k - len(ctx))]
            else:
                cands, ctx = c1, ctx1
            t1 = time.perf_counter()
            ans = self.generator.generate(question, ctx)
            if support and (ans.abstained or ans.confidence < 0.8):
                alt = self.generator.generate(question, ctx, support_query=support)
                if not alt.abstained and (ans.abstained or alt.confidence > ans.confidence):
                    ans = alt
            tm["generate_ms"] = ms(t1)
            tm["postprocess_ms"] = 0.0
            tm["total_ms"] = ms(t0)
            rec = RunRecord(self.name, q.qid, q.question, cands, ctx, ans, tm,
                            ans.tokens_in / 1000 * self.price_in + ans.tokens_out / 1000 * self.price_out)
            rec.extra["second_hop_query"] = support
            return rec
        except Exception as exc:  # noqa: BLE001
            return RunRecord(self.name, q.qid, q.question, [], [], Answer("", True), {"total_ms": ms(t0)},
                             error=f"{type(exc).__name__}: {exc}")


class FallbackPipeline:
    """Confidence-based fallback: try primary; if evidence is weak retry with an alternative; else abstain."""

    def __init__(self, name: str, primary: Pipeline, alternative: Pipeline, min_conf: float = 0.5):
        self.name, self.primary, self.alt, self.min_conf = name, primary, alternative, min_conf
        self.config = {"primary": primary.name, "alternative": alternative.name, "min_conf": min_conf}

    def run(self, query) -> RunRecord:
        r1 = self.primary.run(query)
        if not r1.answer.abstained and r1.answer.confidence >= self.min_conf:
            r1.arch = self.name; r1.extra["fallback"] = "none"; return r1
        r2 = self.alt.run(query)
        best = max((r1, r2), key=lambda r: (not r.answer.abstained, r.answer.confidence))
        total = {k: r1.timings_ms.get(k, 0) + r2.timings_ms.get(k, 0) for k in set(r1.timings_ms) | set(r2.timings_ms)}
        best.timings_ms, best.cost_usd, best.arch = total, r1.cost_usd + r2.cost_usd, self.name
        best.extra["fallback"] = "retried"
        if best.answer.abstained or best.answer.confidence < self.min_conf:
            best.answer = Answer("", True, best.answer.confidence, best.answer.support, best.answer.tokens_in)
            best.extra["fallback"] = "abstained"
        return best


# ───────────────────────────── builders ────────────────────────────────────
class IndexBundle:
    """Fits each retriever once per chunking config so architectures share indexes (fair + fast)."""

    def __init__(self, passages: Sequence[Passage]):
        self.passages = list(passages)
        self.parents = {p.pid: p.text for p in passages}
        self._units: Dict[Tuple[int, int], List[Unit]] = {}
        self._r: Dict[Tuple, Retriever] = {}
        self._cross: Dict[Tuple[int, int], LexicalCrossScorer] = {}

    def units(self, size: int, overlap: int) -> List[Unit]:
        key = (size, overlap)
        if key not in self._units:
            self._units[key] = build_units(self.passages, size, overlap)
        return self._units[key]

    def retriever(self, kind: str, size: int, overlap: int) -> Retriever:
        key = (kind, size, overlap)
        if key not in self._r:
            u = self.units(size, overlap)
            self._r[key] = {"bm25": BM25Retriever, "tfidf": TfidfRetriever, "lsa": LSARetriever}[kind](u)
        return self._r[key]

    def cross(self, size: int, overlap: int) -> LexicalCrossScorer:
        key = (size, overlap)
        if key not in self._cross:
            self._cross[key] = LexicalCrossScorer(self.units(size, overlap))
        return self._cross[key]


def build_pipeline(spec: Dict[str, Any], bundle: IndexBundle, reranker=None) -> Pipeline:
    size, ov = spec.get("chunk_size", 120), spec.get("overlap", 20)
    kind = spec["retriever"]
    if kind in ("bm25", "tfidf", "lsa"):
        r: Retriever = bundle.retriever(kind, size, ov)
    elif kind == "hybrid":
        parts = [(bundle.retriever(c, size, ov), w) for c, w in spec.get("components", [("bm25", .5), ("lsa", .5)])]
        r = HybridRetriever(parts, spec.get("fusion", "rrf"), spec.get("pool", 50))
    elif kind == "fusion":
        r = QueryVariantRetriever(bundle.retriever(spec.get("base", "bm25"), size, ov), spec.get("pool", 50))
    elif kind == "parent_child":
        child = bundle.retriever(spec.get("base", "bm25"), spec.get("child_size", 40), spec.get("child_overlap", 10))
        r = ParentChildRetriever(child, bundle.parents)
    else:
        raise ValueError(f"unknown retriever kind {kind!r}")
    if spec.get("rerank"):
        rr = reranker or bundle.cross(size, ov)
        r = RerankRetriever(r, rr, spec.get("rerank_pool", 30))
    cross = bundle.cross(size, ov)
    gen = spec.get("generator") or ExtractiveGenerator(cross.idf, spec.get("abstain_threshold", 0.55), cross.default_idf)
    cls = MultiHopPipeline if spec.get("multihop") else Pipeline
    return cls(spec["name"], r, gen, k=spec.get("k", 5), depth=spec.get("depth", 20),
                    adaptive=spec.get("adaptive_k"), config={k: v for k, v in spec.items() if k != "generator"},
                    price_in_per_1k=spec.get("price_in_per_1k", 0.0), price_out_per_1k=spec.get("price_out_per_1k", 0.0))


def default_specs(chunk_size: int = 120, overlap: int = 20, k: int = 5) -> Dict[str, Dict[str, Any]]:
    base = {"chunk_size": chunk_size, "overlap": overlap, "k": k, "depth": 20}
    S = lambda name, **kw: {**base, "name": name, **kw}  # noqa: E731
    specs = [
        S("bm25", retriever="bm25"),
        S("vector-tfidf", retriever="tfidf"),
        S("dense-lsa", retriever="lsa"),
        S("hybrid-rrf", retriever="hybrid", fusion="rrf"),
        S("hybrid-weighted", retriever="hybrid", fusion="weighted", components=[("bm25", .6), ("lsa", .4)]),
        S("hybrid+rerank", retriever="hybrid", fusion="rrf", rerank=True),
        S("bm25+rerank", retriever="bm25", rerank=True),
        S("parent-child", retriever="parent_child"),
        S("query-fusion", retriever="fusion"),
        S("multihop-hybrid", retriever="hybrid", fusion="rrf", rerank=True, multihop=True),
        S("full-stack", retriever="hybrid", fusion="rrf", rerank=True, multihop=True, adaptive_k={"k_min": 2, "k_max": 8, "ratio": 0.6}),
        S("hybrid+rerank+adaptive-k", retriever="hybrid", fusion="rrf", rerank=True, adaptive_k={"k_min": 2, "k_max": 8, "ratio": 0.6}),
    ]
    return {s["name"]: s for s in specs}


# ───────────────────────────── bridge to the existing LangChain stack ──────
class LangChainAdapter:
    """Run any existing ``module2.rag_architectures.BaseRAG`` through the rag_lab evaluation engine.

    ``generator`` is optional: with none, retrieval is evaluated and answers are left empty
    (equivalent to the repo's ``--retrieval-only`` mode); pass ``LLMGenerator(...)`` for full runs.
    """

    def __init__(self, rag: Any, name: Optional[str] = None, generator=None, k: int = 5):
        self.rag, self.generator, self.k = rag, generator, k
        self.name = name or getattr(rag, "name", type(rag).__name__)
        self.config = {"adapter": type(rag).__name__, "k": k}

    def run(self, query) -> RunRecord:
        q = query if isinstance(query, Query) else Query("adhoc", str(query))
        t0 = time.perf_counter()
        try:
            docs = self.rag.retrieve(q.question)
            t_ret = ms(t0)
            hits = [Hit(content_id(getattr(d, "page_content", str(d))), f"{content_id(getattr(d, 'page_content', str(d)))}#0",
                        getattr(d, "page_content", str(d)), 1.0 / (r + 1), r + 1) for r, d in enumerate(docs)]
            ctx = hits[: self.k]
            t1 = time.perf_counter()
            ans = self.generator.generate(q.question, ctx) if self.generator else Answer("", True)
            tm = {"retrieve_ms": t_ret, "rerank_ms": 0.0, "generate_ms": ms(t1) if self.generator else 0.0}
            tm["total_ms"] = ms(t0)
            return RunRecord(self.name, q.qid, q.question, hits, ctx, ans, tm)
        except Exception as exc:  # noqa: BLE001
            return RunRecord(self.name, q.qid, q.question, [], [], Answer("", True), {"total_ms": ms(t0)},
                             error=f"{type(exc).__name__}: {exc}")
