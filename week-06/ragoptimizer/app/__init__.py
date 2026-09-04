"""RAGOptimizer - Week 6 extension to the CitationRAG service.

Adds HyDE query transformation, cross-encoder reranking, and extractive
context compression on top of the existing CitationRAG pipeline. The
public service contract (FastAPI routes, citation format, fallback
trigger) follows CitationRAG's shape - only the retrieval path changes. Both
pipelines apply CitationRAG's threshold gate (top1 < 0.55 or spread < 0.08 →
refuse before the LLM): the baseline gates on its single dense channel,
the full pipeline on (raw_ok or hyde_ok).
"""

__version__ = "0.1.0"
