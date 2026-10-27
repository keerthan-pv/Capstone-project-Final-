"""Typed contracts used by the production answer path."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class AnswerRoute(str, Enum):
    CURATED_FACT = "curated_fact"
    STRUCTURED_WEB = "structured_web"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class Evidence:
    text: str
    source: str
    kind: str = "source"


@dataclass
class AnswerResult:
    answer: str
    route: AnswerRoute
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    abstained: bool = False
    reason: str = ""

    @property
    def sources(self) -> List[str]:
        return [item.source for item in self.evidence if item.source]
