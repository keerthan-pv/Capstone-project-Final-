"""Regression tests for answer extraction and the online fallback."""

import io
import json

from langchain_core.documents import Document

from module2.rag_architectures import BaseRAG, _LLM
from module2.web_search import search_web


def _prompt(question, context):
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def test_first_person_question_returns_person_not_mission():
    context = (
        "Apollo 11 was the first manned mission to land on the Moon. "
        "American Neil Armstrong became the first man to walk on the Moon."
    )
    answer = _LLM._extractive_fallback(
        _prompt("who is the first person to land on moon?", context)
    )
    assert "Neil Armstrong" in answer
    assert "Apollo 11" not in answer


def test_cm_is_understood_as_chief_minister():
    answer = _LLM._extractive_fallback(_prompt(
        "who is the cm of karnataka?",
        "Siddaramaiah is the Chief Minister of Karnataka.",
    ))
    assert answer == "Siddaramaiah is the Chief Minister of Karnataka."


def test_current_ceo_beats_former_ceo_biography():
    answer = _LLM._extractive_fallback(_prompt(
        "who is ceo of microsoft?",
        "Microsoft has named Satya Nadella as its new CEO. "
        "Steve Ballmer was the former chief executive officer of Microsoft.",
    ))
    assert answer == "Satya Nadella is the CEO of Microsoft."


def test_google_ceo_is_extracted_from_biography_intro():
    answer = _LLM._extractive_fallback(_prompt(
        "who is the ceo of google?",
        "Pichai Sundararajan, better known as Sundar Pichai, is an Indian-American "
        "business executive who has been the CEO of Google since 2015.",
    ))
    assert answer == "Sundar Pichai is the CEO of Google."


def test_current_office_holder_with_initials_is_extracted():
    answer = _LLM._extractive_fallback(_prompt(
        "who is the cm of karnataka?",
        "The Chief Minister of Karnataka is the head of government. "
        "D. K. Shivakumar became the twenty-first and incumbent chief minister on 3 June 2026.",
    ))
    assert answer == "D K Shivakumar is the CM of Karnataka."


def test_ipl_winner_is_extracted_from_season_result():
    answer = _LLM._extractive_fallback(_prompt(
        "who won ipl in 2026?",
        "The 2026 Indian Premier League was the 19th edition. "
        "Defending champions Royal Challengers Bengaluru defeated Gujarat Titans "
        "by 5 wickets in the final to secure their second consecutive IPL title.",
    ))
    assert answer == "Royal Challengers Bengaluru won the 2026 IPL."


def test_pm_alias_is_grounded_to_a_named_person():
    answer = _LLM._extractive_fallback(_prompt(
        "who is the pm of india?",
        "Narendra Modi, who is the current prime minister of India, has served since 2014.",
    ))
    assert answer == "Narendra Modi is the PM of India."


def test_pm_question_always_uses_web_verification():
    assert BaseRAG._should_search_web(
        "who is the pm of india?",
        "The head of government is the Prime Minister.",
        [Document(page_content="The head of government is the Prime Minister.")],
    )


def test_year_specific_ipl_question_always_uses_web_verification():
    assert BaseRAG._should_search_web(
        "who won ipl in 2026?",
        "Mumbai is home to the IPL.",
        [Document(page_content="Mumbai is home to the IPL.")],
    )


def test_silicon_valley_of_india_resolves_to_karnataka():
    answer = _LLM._extractive_fallback(_prompt(
        "which state in india is called silicon valley?",
        "Bangalore is known as the Silicon Valley of India.",
    ))
    assert "Karnataka" in answer
    assert "California" not in answer


def test_unrelated_local_context_triggers_web_fallback():
    assert BaseRAG._should_search_web(
        "who is the cm of karnataka?",
        "Narendra Modi was the Chief Minister of Gujarat.",
        [Document(page_content="A history of Hyderabad and Andhra Pradesh.")],
    )


def test_role_question_does_not_accept_related_but_wrong_role():
    assert BaseRAG._should_search_web(
        "who is the education minister of india?",
        "Narendra Damodardas Modi is the Prime Minister of India.",
        [Document(page_content="Narendra Damodardas Modi is the Prime Minister of India.")],
    )


def test_search_web_parses_instant_answer(monkeypatch):
    payload = {
        "AbstractText": "Siddaramaiah is the Chief Minister of Karnataka.",
        "AbstractURL": "https://example.test/siddaramaiah",
        "RelatedTopics": [],
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        "module2.web_search.urlopen",
        lambda request, timeout: Response(json.dumps(payload).encode()),
    )
    docs = search_web("who is the cm of karnataka?", max_results=5)
    assert docs and "Siddaramaiah" in docs[0].page_content
    assert docs[0].metadata["web"] is True


def test_targeted_search_rewrites_capital_question():
    from module2.web_search import _targeted_search_query

    assert _targeted_search_query("What is the capital of France?") == "France capital"


def test_targeted_search_rewrites_author_question():
    from module2.web_search import _targeted_search_query

    assert _targeted_search_query("Who wrote Hamlet?") == "Hamlet author written by"


def test_definition_parser_accepts_imperative_wording(monkeypatch):
    from module2.web_search import _datamuse_definition

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "module2.web_search.urlopen",
        lambda *_args, **_kwargs: Response(b'[{"defs":["n\\ta reason against using a treatment"]}]'),
    )
    doc = _datamuse_definition("Define contraindication.", timeout=1)
    assert doc and "Contraindication means" in doc.page_content


def test_currency_rate_parser_returns_live_rate_document(monkeypatch):
    from module2.web_search import _currency_rate

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    payload = {"rates": {"INR": 83.25}, "time_last_update_utc": "Mon, 01 Jan 2026 00:00:00 +0000"}
    monkeypatch.setattr(
        "module2.web_search.urlopen",
        lambda request, timeout: Response(json.dumps(payload).encode()),
    )
    doc = _currency_rate("1 dollar is equal to how many rupees?", timeout=1)
    assert doc and "83.25" in doc.page_content


def test_currency_rate_parser_accepts_how_much_wording(monkeypatch):
    from module2.web_search import _currency_rate

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "module2.web_search.urlopen",
        lambda *_args, **_kwargs: Response(
            b'{"rates":{"INR":83.25},"time_last_update_utc":"today"}'
        ),
    )
    doc = _currency_rate("1 dollar is equal to how much rupees?", timeout=1)
    assert doc and "83.25" in doc.page_content


def test_acronym_resolver_extracts_defining_sentence(monkeypatch):
    from module2.web_search import _acronym_fact

    monkeypatch.setattr(
        "module2.web_search._wikipedia_search",
        lambda query, max_results, timeout: [Document(
            page_content="The National Aeronautics and Space Administration (NASA) is a U.S. agency.",
            metadata={"source": "https://en.wikipedia.org/wiki/NASA"},
        )],
    )
    doc = _acronym_fact("what is the full form of NASA?", max_results=5, timeout=1)
    assert doc and doc.page_content == "NASA stands for The National Aeronautics and Space Administration."


def test_state_symbol_resolver_extracts_paired_symbols(monkeypatch):
    from module2.web_search import _state_or_national_symbol_fact

    monkeypatch.setattr(
        "module2.web_search._wikipedia_search",
        lambda *_args, **_kwargs: [Document(
            page_content=(
                "The state bird and state animal of Karnataka are Indian roller "
                "and the Indian elephant."
            ),
            metadata={"source": "https://example.test/karnataka-wildlife"},
        )],
    )
    doc = _state_or_national_symbol_fact(
        "what is the state bird of karnataka?", max_results=5, timeout=1,
    )
    assert doc and doc.page_content == "Indian roller is the state bird of Karnataka."

