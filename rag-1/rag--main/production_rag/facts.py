"""Data-backed stable facts.

These are not model training examples.  They are a small, auditable answer
registry for facts that should never be delegated to a generative model.  Each
entry has an exact normalized key, a canonical answer, and a public source.
Current office-holder facts intentionally do not live here; they are resolved
through a live structured source instead of becoming stale constants.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def normalize_query(value: str) -> str:
    value = value.casefold().replace("inida", "india")
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@dataclass(frozen=True)
class CuratedFact:
    key: str
    patterns: tuple[str, ...]
    answer: str
    source: str


class FactStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        path = path or Path(__file__).with_name("..") / "data" / "curated_facts.json"
        path = path.resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._facts = tuple(
            CuratedFact(
                key=item["key"],
                patterns=tuple(normalize_query(pattern) for pattern in item["patterns"]),
                answer=item["answer"],
                source=item["source"],
            )
            for item in raw["facts"]
        )
        self._by_pattern = {
            pattern: fact for fact in self._facts for pattern in fact.patterns
        }

    def lookup(self, query: str) -> Optional[CuratedFact]:
        return self._by_pattern.get(normalize_query(query))
