"""Core data model shared by every rag_lab component."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Passage:
    pid: str
    text: str
    title: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    qid: str
    question: str
    answers: List[str] = field(default_factory=list)
    relevant: Dict[str, int] = field(default_factory=dict)   # pid -> graded relevance (>0)
    answerable: bool = True
    qtype: str = "unknown"
    source: str = "real"                                      # "real" | "synthetic"
    paraphrases: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalDataset:
    name: str
    passages: List[Passage]
    queries: List[Query]
    version: str = "1"
    notes: str = ""

    def fingerprint(self) -> str:
        """Content hash: changes iff corpus text, queries, answers or labels change."""
        h = hashlib.sha256()
        for p in sorted(self.passages, key=lambda x: x.pid):
            h.update(p.pid.encode()); h.update(b"\0"); h.update(p.text.encode()); h.update(b"\1")
        for q in sorted(self.queries, key=lambda x: x.qid):
            h.update(json.dumps([q.qid, q.question, q.answers, sorted(q.relevant.items()),
                                 q.answerable, q.qtype, q.source], sort_keys=True).encode())
        return h.hexdigest()[:16]

    def summary(self) -> Dict[str, Any]:
        qt: Dict[str, int] = {}
        for q in self.queries:
            qt[q.qtype] = qt.get(q.qtype, 0) + 1
        return {
            "name": self.name, "version": self.version, "fingerprint": self.fingerprint(),
            "n_passages": len(self.passages), "n_queries": len(self.queries),
            "n_unanswerable": sum(not q.answerable for q in self.queries),
            "query_types": qt, "notes": self.notes,
        }

    def split(self, fractions=(0.5, 0.2, 0.3), seed: int = 42):
        """Deterministic qid-level train/val/test split (used by the router)."""
        import numpy as np
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.queries))
        n = len(idx)
        a = int(round(fractions[0] * n)); b = a + int(round(fractions[1] * n))
        pick = lambda ids: [self.queries[i] for i in ids]  # noqa: E731
        return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])


@dataclass
class Hit:
    pid: str                    # parent document id (what qrels refer to)
    chunk_id: str
    text: str
    score: float
    rank: int = 0
    stage_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class Answer:
    text: str
    abstained: bool = False
    confidence: float = 0.0
    support: List[Dict[str, Any]] = field(default_factory=list)   # [{chunk_id, sentence}]
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class RunRecord:
    """Everything observed while serving one query through one architecture."""
    arch: str
    qid: str
    question: str
    candidates: List[Hit]        # full retrieved pool (depth N), best first
    context: List[Hit]           # the top-k actually given to the generator
    answer: Answer
    timings_ms: Dict[str, float]
    cost_usd: float = 0.0
    error: Optional[str] = None
    cache: str = "off"           # off | miss | hit
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
