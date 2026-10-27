"""LLM-free grounding checks (lexical-support PROXIES — validate against human labels / an LLM judge)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from .textutil import content_tokens, normalize_answer, sentences


def claim_supported(claim: str, chunks: Sequence[str], thr: float = 0.7) -> bool:
    ct = set(content_tokens(claim))
    if not ct:
        return True
    return any(len(ct & set(content_tokens(c))) / len(ct) >= thr for c in chunks)


def lexical_faithfulness(answer: str, chunks: Sequence[str], thr: float = 0.7) -> Optional[float]:
    """Share of answer sentences whose content words are covered by one retrieved chunk. None if no answer."""
    if not (answer or "").strip():
        return None
    claims = sentences(answer)
    return sum(claim_supported(c, chunks, thr) for c in claims) / len(claims)


def citation_correct(answer: str, support: Sequence[Dict], chunks_by_id: Dict[str, str]) -> Optional[float]:
    """Does each cited chunk actually contain the answer text?"""
    if not support or not (answer or "").strip():
        return None
    a = normalize_answer(answer)
    ok = [a in normalize_answer(chunks_by_id.get(s["chunk_id"], "")) for s in support]
    return sum(ok) / len(ok)
