from module5.comparison import ArchitectureComparator


class _AbstainingArchitecture:
    def run(self, query):
        return {
            "answer": "I don't have enough information to answer this accurately.",
            "contexts": [f"unrelated benchmark context for {query}"],
        }


def test_comparison_uses_a_shared_verified_answer_before_ranking():
    comparator = ArchitectureComparator({
        "naive": _AbstainingArchitecture(),
        "adaptive": _AbstainingArchitecture(),
    })

    results = comparator.compare_with_confidence(
        "Who is the PM of India?",
        verified_answer="Narendra Modi is the Prime Minister of India.",
        verified_sources=["https://example.test/india"],
        verified_confidence=90.0,
    )

    assert [result.answer for result in results] == [
        "Narendra Modi is the Prime Minister of India.",
        "Narendra Modi is the Prime Minister of India.",
    ]
    assert all(result.correctness == 1.0 for result in results)
    assert all(result.confidence == 90.0 for result in results)
    assert all(result.num_chunks == 1 for result in results)
    assert all(result.verification_sources == ["https://example.test/india"] for result in results)


def test_comparison_leaves_architecture_answers_unchanged_without_verification():
    comparator = ArchitectureComparator({"naive": _AbstainingArchitecture()})

    [result] = comparator.compare_with_confidence("unknown question")

    assert "don't have enough information" in result.answer
    assert result.correctness == 0.0
