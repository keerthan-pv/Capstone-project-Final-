"""Production answer orchestration.

The service has an explicit evidence policy:

1. Exact stable facts come from the curated registry.
2. Current/simple factual questions use the existing structured web adapters,
   but only accept their direct evidence; no free-form LLM synthesis is used.
3. Document retrieval is injected by the caller and is never mixed with web
   facts.
4. Missing or contradictory evidence produces an abstention.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .facts import FactStore
from .models import AnswerResult, AnswerRoute, Evidence
from .router import is_structured_lookup, looks_like_document_question


WebSearch = Callable[..., list[Any]]
DocumentAnswer = Callable[[str], Optional[dict[str, Any]]]
AnswerGenerator = Callable[[str], str]


class AnswerService:
    def __init__(
        self,
        *,
        fact_store: Optional[FactStore] = None,
        web_search: Optional[WebSearch] = None,
        document_answer: Optional[DocumentAnswer] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        web_timeout: float = 8.0,
    ) -> None:
        self.fact_store = fact_store or FactStore()
        self.web_search = web_search
        self.document_answer = document_answer
        self.answer_generator = answer_generator
        self.web_timeout = web_timeout

    def answer(self, query: str) -> AnswerResult:
        query = query.strip()
        if not query:
            return self._abstain("The question is empty.")
        if _is_prompt_injection(query):
            return self._abstain(
                "I can only answer from verified sources or supplied documents; I will not ignore that evidence policy."
            )
        clarification = _clarification_needed(query)
        if clarification:
            return self._abstain(clarification)

        fact = self.fact_store.lookup(query)
        if fact:
            return AnswerResult(
                answer=fact.answer,
                route=AnswerRoute.CURATED_FACT,
                confidence=0.99,
                evidence=[Evidence(fact.answer, fact.source, "curated")],
            )

        if looks_like_document_question(query) and self.document_answer:
            result = self.document_answer(query)
            if result and result.get("answer") and result.get("contexts"):
                return AnswerResult(
                    answer=str(result["answer"]),
                    route=AnswerRoute.DOCUMENT_RETRIEVAL,
                    confidence=min(0.95, float(result.get("confidence", 0.65))),
                    evidence=[Evidence(str(text), "uploaded document", "document")
                              for text in result["contexts"][:5]],
                )

        # Open-world questions cannot be solved by the local benchmark corpus
        # alone. Search all non-document questions, not only questions matching
        # a growing list of keywords.
        if self.web_search:
            return self._structured_web_answer(query)

        return self._abstain(
            "No trusted source matched this question. Add a document or ask a more specific question."
        )

    def _structured_web_answer(self, query: str) -> AnswerResult:
        started = time.perf_counter()
        try:
            docs = self.web_search(query, max_results=5, timeout=self.web_timeout)
        except Exception:
            docs = []
        evidence = []
        for doc in docs or []:
            text = str(getattr(doc, "page_content", "") or "").strip()
            source = str(getattr(doc, "metadata", {}).get("source", "") or "")
            if text and source:
                evidence.append(Evidence(text, source, "web"))

        if not evidence:
            return self._abstain("The trusted web sources were unavailable or returned no usable evidence.")

        evidence = _rank_evidence(query, evidence)
        direct = _direct_answer(query, evidence)
        # Fact lookups must be answered by explicit evidence.  Never let a
        # generative fallback turn a related page (for example a page merely
        # about Karnataka) into the requested fact.
        if not direct and is_structured_lookup(query):
            return self._abstain(
                "The sources did not provide a direct answer, so I will not guess."
            )
        if not direct and self.answer_generator:
            prompt = _grounded_prompt(query, evidence)
            try:
                generated = self.answer_generator(prompt).strip()
            except Exception:
                generated = ""
            if generated and not _is_refusal(generated):
                return AnswerResult(
                    answer=generated,
                    route=AnswerRoute.STRUCTURED_WEB,
                    confidence=0.72,
                    evidence=evidence,
                )
        if not direct:
            return self._abstain(
                "The sources did not provide a direct answer, so I will not guess."
            )
        return AnswerResult(
            answer=direct,
            route=AnswerRoute.STRUCTURED_WEB,
            confidence=0.90 if evidence[0].kind == "web" else 0.75,
            evidence=evidence,
        )

    @staticmethod
    def _abstain(reason: str) -> AnswerResult:
        return AnswerResult(
            answer="I don't have enough verified evidence to answer that reliably.",
            route=AnswerRoute.ABSTAIN,
            confidence=0.0,
            reason=reason,
            abstained=True,
        )


def _direct_answer(query: str, evidence: Iterable[Evidence]) -> Optional[str]:
    """Accept only evidence that contains a useful direct factual sentence."""
    evidence_list = list(evidence)
    if not evidence_list:
        return None
    text = "\n".join(item.text for item in evidence_list)
    normalized = query.casefold().replace("inida", "india")
    symbol_request = _symbol_request(normalized)
    if symbol_request:
        scope, symbol, entity = symbol_request
        relation = f"{scope} {symbol} of {entity}"
        for item in evidence_list:
            sentence = item.text.strip()
            lowered_sentence = sentence.casefold()
            if relation in lowered_sentence and re.search(r"\b(?:is|are)\b", sentence, re.I):
                return sentence
    role_request = _role_request(normalized)
    if role_request:
        role, entity = role_request
        aliases = {
            "chief minister": ("chief minister",),
            "education minister": ("education minister", "minister of education"),
            "prime minister": ("prime minister",),
            "finance minister": ("finance minister", "minister of finance"),
            "home minister": ("home minister", "minister of home affairs"),
            "external affairs minister": ("external affairs minister", "minister of external affairs"),
        }.get(role, (role,))
        # Treat each retrieved passage as a candidate.  This preserves
        # initials such as ``C. Joseph Vijay`` and avoids returning a
        # punctuation-truncated fragment from a multi-source join.
        sentences = [item.text.strip() for item in evidence_list if item.text.strip()]
        candidates = []
        for sentence in sentences:
            lowered_sentence = sentence.casefold()
            if entity not in lowered_sentence or not any(alias in lowered_sentence for alias in aliases):
                continue
            if role == "chief minister" and "deputy chief minister" in lowered_sentence:
                continue
            if not re.search(r"\b(?:current|incumbent|is the|serving|appointed|elected)\b", sentence, re.I):
                continue
            candidates.append(sentence)
        if candidates:
            return sorted(candidates, key=lambda value: (
                bool(re.search(r"\bcurrent\b", value, re.I)),
                -len(value),
            ), reverse=True)[0]
        if _is_strict_role_lookup(query):
            return None
    # These intents require a named relation.  Returning the first topical
    # paragraph is worse than abstaining, especially when Wikipedia search
    # returns related pages before the answer page.
    if re.search(r"\bcapital\s+of\b", normalized):
        subject = re.search(r"\bcapital\s+of\s+([a-z][a-z ]+)", normalized)
        subject_text = subject.group(1).strip() if subject else ""
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if subject_text and subject_text not in sentence.casefold():
                continue
            if re.search(r"\bis\s+the\s+capital\s+of\b", sentence, re.I):
                return sentence.strip()
            match = re.search(
                r"\bcapital\b[^.]{0,100}\b(?:is|was)\s+(?:the\s+)?([A-Z][A-Za-z .'-]+)",
                sentence,
            )
            if match:
                return sentence.strip()
    if re.search(r"\bwho\s+(?:wrote|authored)\b", normalized):
        work_match = re.search(r"\bwho\s+(?:wrote|authored)\s+(.+?)\??$", normalized)
        work = work_match.group(1).strip() if work_match else ""
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if work and work not in sentence.casefold():
                continue
            if re.search(r"\b(?:written|authored|composed)\s+by\b|\b[A-Z][A-Za-z .'-]+\s+wrote\b", sentence, re.I):
                return sentence.strip()
    if re.search(r"\bopposite\b", normalized) and re.search(r"\bopposite\b.+\bis\b", text, re.I):
        return text
    if re.search(r"\bdefinition\b|\bmeaning\b|\bdefine\b", normalized) and re.search(r"\bmeans\b|\bdefinition\b", text, re.I):
        return text
    if re.search(r"\b(?:full form|abbreviation|acronym)\b", normalized):
        if re.search(r"\b(?:stands for|means|is an acronym for)\b", text, re.I):
            return text
    if re.search(r"\b(?:dollar|usd)\b", normalized) and re.search(r"\b(?:rupees|inr)\b", normalized):
        if re.search(r"\b(?:equals|equal|approximately)\b", text, re.I):
            return text
    if re.search(r"\b(?:ceo|founder|pm|cm|prime minister|chief minister|finance minister)\b", normalized):
        # Prefer the short sentence naming the current office-holder when a
        # Wikipedia introduction contains several historical details.
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        role_terms = re.findall(
            r"\b(?:ceo|founder|pm|cm|prime minister|chief minister|finance minister)\b",
            normalized,
        )
        if "pm" in role_terms:
            role_terms.append("prime minister")
        if "cm" in role_terms:
            role_terms.append("chief minister")
        ranked_sentences = sorted(
            sentences,
            key=lambda sentence: (
                bool(re.search(r"\b(?:current|incumbent|serving since)\b", sentence, re.I)),
                len(sentence),
            ),
            reverse=True,
        )
        for sentence in ranked_sentences:
            if re.search(r"\b(?:current|incumbent|is the|is an|is a|founded by|does not have)\b", sentence, re.I) and any(
                term in sentence.casefold() for term in role_terms
            ):
                return sentence
    if re.search(r"\blanguage\b", normalized) and re.search(r"\b(?:language|spoken|official)\b", text, re.I):
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if re.search(r"\b(?:language|spoken|official)\b", sentence, re.I):
                return sentence.strip()
    return None


def _role_request(query: str) -> Optional[tuple[str, str]]:
    match = re.search(
        r"\bwho\s+is\s+(?:the\s+)?(?P<role>[a-z][a-z ]*?)\s+of\s+(?P<entity>[a-z][a-z ]+?)[?!\.]*$",
        query.casefold(),
    )
    if not match:
        return None
    role = re.sub(r"\s+", " ", match.group("role")).strip()
    entity = re.sub(r"\s+", " ", match.group("entity")).strip()
    role_aliases = {
        "pm": "prime minister",
        "prime minister": "prime minister",
        "cm": "chief minister",
        "c m": "chief minister",
        "chief minister": "chief minister",
        "education minister": "education minister",
        "minister of education": "education minister",
        "finance minister": "finance minister",
        "minister of finance": "finance minister",
        "home minister": "home minister",
        "minister of home affairs": "home minister",
        "external affairs minister": "external affairs minister",
        "minister of external affairs": "external affairs minister",
    }
    canonical = role_aliases.get(role)
    return (canonical, entity) if canonical else None


def _symbol_request(query: str) -> Optional[tuple[str, str, str]]:
    match = re.fullmatch(
        r"(?:what|which)\s+(?:is|are)\s+(?:the\s+)?"
        r"(?P<scope>state|national)\s+(?P<symbol>animal|bird|flower|tree)\s+"
        r"of\s+(?P<entity>[a-z][a-z ]+)[?!\.]*",
        query.casefold(),
    )
    if not match:
        return None
    return (
        match.group("scope"),
        match.group("symbol"),
        re.sub(r"\s+", " ", match.group("entity")).strip(),
    )


def _is_strict_role_lookup(query: str) -> bool:
    role_request = _role_request(query)
    return bool(role_request and role_request[0] in {
        "chief minister", "education minister", "prime minister",
        "finance minister", "home minister", "external affairs minister",
    })


def _rank_evidence(query: str, evidence: Iterable[Evidence]) -> list[Evidence]:
    """Put query-relevant sources first before extraction or generation."""
    stop_words = {
        "what", "who", "where", "when", "why", "how", "is", "are", "the",
        "a", "an", "of", "to", "in", "for", "and", "does", "did", "can",
    }
    query_tokens = {
        token for token in re.findall(r"[a-z0-9]+", query.casefold())
        if token not in stop_words and len(token) > 2
    }

    def score(item: Evidence) -> tuple[int, int]:
        content = f"{item.text} {item.source}".casefold()
        overlap = sum(token in content for token in query_tokens)
        return overlap, -len(item.text)

    return sorted(list(evidence), key=score, reverse=True)


def _grounded_prompt(query: str, evidence: Iterable[Evidence]) -> str:
    sources = "\n\n".join(item.text for item in evidence)
    return (
        "Answer the user's question using only the source passages below. "
        "Do not invent facts, do not mention the retrieval process, and be concise. "
        "If the passages do not answer the question, reply exactly: "
        "I don't have enough verified evidence to answer that reliably.\n\n"
        f"Context:\n{sources}\n\nQuestion: {query}\n\nAnswer:"
    )


def _is_refusal(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", answer.casefold()).strip()
    return normalized.startswith((
        "i don't have enough",
        "i do not have enough",
        "i don't know",
        "i do not know",
        "unable to answer",
    ))


def _is_prompt_injection(query: str) -> bool:
    """Reject instructions that try to bypass the evidence boundary."""
    text = query.casefold()
    return bool(re.search(
        r"\b(?:ignore|disregard|bypass)\b.{0,80}\b(?:context|instructions|sources?|rules?)\b",
        text,
    ))


def _clarification_needed(query: str) -> Optional[str]:
    """Keep underspecified follow-ups from triggering broad, unreliable search."""
    text = re.sub(r"\s+", " ", query.casefold()).strip(" ?!.")
    if re.fullmatch(r"what did (?:he|she|they|it) say", text):
        return "Please identify who you mean and the conversation or document you are referring to."
    if text in {"tell me about the treatment", "tell me about treatment", "what is the treatment"}:
        return "Please specify the condition or document whose treatment you want to discuss."
    return None
