"""Run the corpus-aware Adaptive RAG acceptance suite.

Default mode is deterministic and reports which questions need an additional
corpus or a real graph.  ``--live`` also calls the production answer service
for the runnable open-web / safety cases and records its actual responses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Make direct ``python scripts/run_adaptive_acceptance.py`` execution work on
# Windows and CI without requiring callers to set PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module2.web_search import search_web
from production_rag import AnswerService
from rag_lab.acceptance import evaluate_acceptance_suite, load_acceptance_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="call the production answer service for runnable cases")
    parser.add_argument("--output", type=Path, default=Path("outputs/adaptive_acceptance_report.json"))
    args = parser.parse_args()
    cases = load_acceptance_suite()
    # The production chat can safely exercise current-fact, clarification and
    # safety behavior.  Its configured MS MARCO index is not a P&P/medical KG.
    capabilities = {"open_web", "safety"}
    service = AnswerService(web_search=search_web) if args.live else None
    report = evaluate_acceptance_suite(cases, capabilities=capabilities, answer=service.answer if service else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in report:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"output": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
