"""Conservative query routing for the production answer service."""

from __future__ import annotations

import re


def is_structured_lookup(query: str) -> bool:
    """Return true for short fact/definition/current-role lookups."""
    text = query.casefold().strip()
    return bool(
        re.search(
            r"\b(?:who is|who are|what is|what are|which|when did|where is|"
            r"full form|abbreviation|acronym|definition|meaning|opposite|current|latest|today|now|prime minister|"
            r"chief minister|finance minister|ceo|founder|language|national\s+(?:animal|bird|flower|tree)|"
            r"state\s+(?:animal|bird|flower|tree))\b",
            text,
        )
    )


def looks_like_document_question(query: str) -> bool:
    text = query.casefold()
    return bool(re.search(r"\b(?:according to|in this document|from the file|the paper)\b", text))
