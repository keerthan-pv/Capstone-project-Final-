"""rag_lab — benchmarking, diagnosis and adaptive-optimisation layer for RAG.

Dependency-light on purpose (numpy / pandas / scipy / scikit-learn / matplotlib) so the
whole research workflow runs and is testable without LangChain, Chroma, torch or
network access. Your existing LangChain architectures plug in through
:class:`rag_lab.architectures.LangChainAdapter`.
"""

__version__ = "1.0.0"
