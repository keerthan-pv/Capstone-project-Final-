"""Production answer service for the RAG application.

This package deliberately separates answer selection from the research benchmark
architectures.  It is small, deterministic for supported facts, and conservative
when evidence is insufficient.
"""

from .models import AnswerResult, Evidence, AnswerRoute
from .service import AnswerService

__all__ = ["AnswerResult", "Evidence", "AnswerRoute", "AnswerService"]
