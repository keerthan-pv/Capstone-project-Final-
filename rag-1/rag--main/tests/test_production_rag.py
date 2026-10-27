from types import SimpleNamespace

from production_rag import AnswerService, AnswerRoute


def test_stable_facts_are_deterministic_and_source_backed():
    service = AnswerService()
    result = service.answer("Who is the founder of Facebook?")
    assert result.route is AnswerRoute.CURATED_FACT
    assert "Mark Zuckerberg" in result.answer
    assert result.sources


def test_isro_full_form_is_answered_without_retrieval_or_generation():
    result = AnswerService().answer("what is the full form of isro?")
    assert result.route is AnswerRoute.CURATED_FACT
    assert result.answer == "ISRO stands for the Indian Space Research Organisation."


def test_isro_full_form_without_of_is_normalized():
    result = AnswerService().answer("what is the full form isro?")
    assert result.answer == "ISRO stands for the Indian Space Research Organisation."


def test_india_national_animal_is_answered_from_curated_fact():
    result = AnswerService().answer("what is the national animal of india?")
    assert result.route is AnswerRoute.CURATED_FACT
    assert result.answer == "The Bengal tiger is the national animal of India."


def test_normandy_location_uses_a_disambiguated_stable_fact():
    result = AnswerService().answer("In what country is Normandy located?")
    assert result.route is AnswerRoute.CURATED_FACT
    assert result.answer == "Normandy is a region in northwestern France."


def test_currency_question_accepts_live_rate_evidence():
    def search(_query, **_kwargs):
        return [SimpleNamespace(
            page_content="1 US dollar equals approximately 83.25 Indian rupees.",
            metadata={"source": "https://open.er-api.com/v6/latest/USD"},
        )]

    result = AnswerService(web_search=search).answer(
        "1 dollar is equal to how many rupees?"
    )
    assert not result.abstained
    assert "83.25" in result.answer


def test_imperative_definition_question_accepts_dictionary_evidence():
    def search(_query, **_kwargs):
        return [SimpleNamespace(
            page_content="Contraindication means a factor that makes a treatment inadvisable.",
            metadata={"source": "https://example.test/dictionary"},
        )]

    result = AnswerService(web_search=search).answer("Define contraindication.")
    assert not result.abstained
    assert "Contraindication means" in result.answer


def test_typo_is_normalized_without_guessing_a_person():
    result = AnswerService().answer("who is the cm of inida?")
    assert "does not have a Chief Minister" in result.answer


def test_unknown_question_abstains_without_a_generator():
    result = AnswerService().answer("What is the least common cloud color on Mars?")
    assert result.abstained
    assert result.route is AnswerRoute.ABSTAIN


def test_web_answer_requires_direct_evidence():
    def search(_query, **_kwargs):
        return [SimpleNamespace(page_content="A page about unrelated history.", metadata={"source": "https://example.test"})]

    result = AnswerService(web_search=search).answer("Who is the CEO of an unknown company?")
    assert result.abstained


def test_web_answer_keeps_source_and_does_not_synthesize():
    def search(_query, **_kwargs):
        return [SimpleNamespace(page_content="Ada Example is the current CEO.", metadata={"source": "https://example.test"})]

    result = AnswerService(web_search=search).answer("Who is the CEO of Example Corp?")
    assert result.route is AnswerRoute.STRUCTURED_WEB
    assert result.answer == "Ada Example is the current CEO."
    assert result.sources == ["https://example.test"]


def test_general_question_can_use_grounded_generator():
    def search(_query, **_kwargs):
        return [SimpleNamespace(
            page_content="Photosynthesis converts light energy into chemical energy in plants.",
            metadata={"source": "https://example.test/photosynthesis"},
        )]

    def generator(prompt):
        assert "Photosynthesis" in prompt
        return "Photosynthesis converts light energy into chemical energy in plants."

    result = AnswerService(web_search=search, answer_generator=generator).answer(
        "How do plants convert light into energy?"
    )
    assert not result.abstained
    assert "chemical energy" in result.answer


def test_grounded_prompt_uses_existing_extractive_contract():
    from production_rag.service import _grounded_prompt

    prompt = _grounded_prompt(
        "What is photosynthesis?",
        [SimpleNamespace(text="Photosynthesis converts light into chemical energy.", source="https://example.test")],
    )
    assert "Context:" in prompt
    assert "Question: What is photosynthesis?" in prompt
    assert "<source" not in prompt


def test_role_extractor_ignores_related_first_paragraph():
    def search(_query, **_kwargs):
        return [
            SimpleNamespace(
                page_content="The Prime Minister's Office contains staff reporting to the Prime Minister.",
                metadata={"source": "https://example.test/pmo"},
            ),
            SimpleNamespace(
                page_content="Narendra Modi is the current Prime Minister of India.",
                metadata={"source": "https://example.test/pm"},
            ),
        ]

    result = AnswerService(web_search=search).answer("Who is the PM of India?")
    assert result.answer == "Narendra Modi is the current Prime Minister of India."


def test_education_role_does_not_accept_prime_minister_evidence():
    def search(_query, **_kwargs):
        return [
            SimpleNamespace(
                page_content="The prime minister of India is the head of government.",
                metadata={"source": "https://example.test/pm"},
            ),
            SimpleNamespace(
                page_content="Pralhad Joshi is the current Education Minister of India.",
                metadata={"source": "https://example.test/education"},
            ),
        ]

    result = AnswerService(web_search=search).answer(
        "Who is the education minister of India?"
    )
    assert result.answer == "Pralhad Joshi is the current Education Minister of India."


def test_role_lookup_abstains_on_deputy_chief_minister_evidence():
    def search(_query, **_kwargs):
        return [SimpleNamespace(
            page_content="The current government does not have a deputy chief minister, and the post is vacant.",
            metadata={"source": "https://example.test/deputy"},
        )]

    result = AnswerService(web_search=search).answer(
        "Who is the CM of Tamil Nadu?"
    )
    assert result.abstained


def test_role_lookup_preserves_initials_in_holder_name():
    def search(_query, **_kwargs):
        return [SimpleNamespace(
            page_content="C. Joseph Vijay is the current Chief Minister of Tamil Nadu.",
            metadata={"source": "https://example.test/tamil-nadu"},
        )]

    result = AnswerService(web_search=search).answer(
        "Who is the CM of Tamil Nadu?"
    )
    assert result.answer.startswith("C. Joseph Vijay")


def test_capital_question_uses_structured_entity_evidence():
    def search(_query, **_kwargs):
        return [SimpleNamespace(
            page_content="Paris is the capital of France.",
            metadata={"source": "https://www.wikidata.org/wiki/Q142"},
        )]

    result = AnswerService(web_search=search).answer("What is the capital of France?")
    assert result.answer == "Paris is the capital of France."


def test_author_question_accepts_subject_wrote_relation():
    def search(_query, **_kwargs):
        return [SimpleNamespace(
            page_content="Jane Austen wrote Pride And Prejudice.",
            metadata={"source": "https://example.test/pride-and-prejudice"},
        )]

    result = AnswerService(web_search=search).answer("Who wrote Pride and Prejudice?")
    assert result.answer == "Jane Austen wrote Pride And Prejudice."


def test_unrelated_evidence_does_not_become_a_confident_answer():
    def search(_query, **_kwargs):
        return [SimpleNamespace(
            page_content="France is a country in Western Europe.",
            metadata={"source": "https://example.test/france"},
        )]

    result = AnswerService(web_search=search).answer("What is the capital of France?")
    assert result.abstained


def test_structured_lookup_never_uses_generator_for_unrelated_evidence():
    def search(_query, **_kwargs):
        return [SimpleNamespace(
            page_content="Karnataka is a state on the southwestern coast of India.",
            metadata={"source": "https://example.test/karnataka"},
        )]

    result = AnswerService(
        web_search=search,
        answer_generator=lambda _prompt: "Karnataka is a state on the southwestern coast of India.",
    ).answer("What is the state bird of Karnataka?")
    assert result.abstained


def test_state_symbol_question_requires_and_accepts_exact_relation():
    def search(_query, **_kwargs):
        return [SimpleNamespace(
            page_content="Indian roller is the state bird of Karnataka.",
            metadata={"source": "https://example.test/karnataka-wildlife"},
        )]

    result = AnswerService(web_search=search).answer(
        "What is the state bird of Karnataka?"
    )
    assert result.answer == "Indian roller is the state bird of Karnataka."
