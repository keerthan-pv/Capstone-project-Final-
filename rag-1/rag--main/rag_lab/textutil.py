"""Shared text helpers (tokenisation, SQuAD-style normalisation, sentence splitting)."""
from __future__ import annotations

import hashlib
import re
import string
from typing import List

STOPWORDS = frozenset(
    "a an the of in on at to for from by with about as into over under is are was were be been being "
    "do does did have has had will would can could should may might must shall and or but if then than "
    "that this these those it its he she they them his her their we you your i me my our us not no "
    "what which who whom whose when where why how".split()
)
_TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def tokens(text: str, drop_stop: bool = False) -> List[str]:
    toks = _TOK.findall((text or "").lower())
    return [t for t in toks if t not in STOPWORDS] if drop_stop else toks


def content_tokens(text: str) -> List[str]:
    return tokens(text, drop_stop=True)


def normalize_answer(s: str) -> str:
    """Official SQuAD normalisation: lowercase, strip punctuation/articles/extra spaces."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT.split((text or "").strip()) if p.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


def approx_tokens(text: str) -> int:
    """Cheap LLM-token estimate (≈1.3 tokens per whitespace word)."""
    return int(round(len((text or "").split()) * 1.3))


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def content_id(text: str) -> str:
    """Identical to module3.evaluation.doc_id_for — keeps legacy qrels interoperable."""
    return hashlib.sha1((text or "")[:200].strip().lower().encode("utf-8")).hexdigest()[:16]
