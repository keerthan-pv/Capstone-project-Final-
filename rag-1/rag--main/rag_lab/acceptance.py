"""Corpus-aware acceptance checks for the Adaptive RAG demo.

The suite records expected routes *and* prerequisites.  A question cannot be
declared a model failure when the active index does not contain its source
corpus; it is instead reported as ``missing_corpus`` or ``missing_graph``.
This keeps the evaluation honest and makes the next data/engineering task
explicit.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from production_rag.models import AnswerResult


SUITE_PATH = Path(__file__).resolve().parents[1] / "data" / "adaptive_rag_acceptance.json"


@dataclass(frozen=True)
class AcceptanceCase:
    id: str
    category: str
    question: str
    expected_route: str
    prerequisite: str
    expected_answer: str = ""


def load_acceptance_suite(path: Path = SUITE_PATH) -> list[AcceptanceCase]:
    rows = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return [AcceptanceCase(**row) for row in rows]


def prerequisite_status(case: AcceptanceCase, capabilities: Iterable[str]) -> str:
    """Return a stable, actionable status without pretending unsupported work passed."""
    available = set(capabilities)
    if case.prerequisite in available:
        return "runnable"
    if case.prerequisite == "knowledge_graph":
        return "missing_graph"
    if case.prerequisite in {"pride_and_prejudice", "clinical_diabetes"}:
        return "missing_corpus"
    return "out_of_scope"


def evaluate_acceptance_suite(
    cases: Iterable[AcceptanceCase],
    *,
    capabilities: Iterable[str],
    answer: Optional[Callable[[str], AnswerResult]] = None,
) -> list[dict[str, Any]]:
    """Evaluate all cases, optionally calling an answer service for runnable cases."""
    report: list[dict[str, Any]] = []
    for case in cases:
        status = prerequisite_status(case, capabilities)
        row: dict[str, Any] = {**asdict(case), "status": status}
        if status == "runnable" and answer is not None:
            result = answer(case.question)
            answer_matches = (
                not case.expected_answer
                or case.expected_answer.casefold() in result.answer.casefold()
            )
            row.update({
                "actual_answer": result.answer,
                "actual_route": result.route.value,
                "abstained": result.abstained,
                "sources": result.sources,
                "answer_matches": answer_matches,
                # The production service reports evidence routes
                # (curated_fact/structured_web/abstain), while the Adaptive
                # benchmark reports retriever routes (vector/bm25/graph).
                # Do not claim they are interchangeable.
                "route_checked": case.expected_route == "abstain",
                "passed": (
                    result.abstained
                    if case.expected_route == "abstain"
                    else (not result.abstained and answer_matches)
                ),
            })
        else:
            row["passed"] = None
        report.append(row)
    return report
