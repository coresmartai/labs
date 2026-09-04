"""Cross-encoder reranker - precision pass after the vector index.

The vector index returns top-30 (high recall, mid precision). This module
scores each (query, chunk) pair with a small cross-encoder model and
sorts by score. We keep the top-5 after sorting; the retriever does the
slicing.

Failure mode this module guards against: LATENCY BLOWOUT - calling the
cross-encoder one pair at a time. The fix is to batch the entire top-30
as one tensor and run a single forward pass. CPU thread count is pinned
so the model doesn't fight FastAPI workers for cores.

Open-source by design (bge-reranker-base). For a managed alternative
(Cohere Rerank, Voyage Rerank), swap the encoder; the function shape is
the same.
"""
from __future__ import annotations

import os

from app.config import get_settings
from app.schemas import Chunk


_RERANKER = None  # lazy global, populated on first call


def _get_reranker():
    """Lazy-load the cross-encoder; respects CPU thread pinning."""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER

    settings = get_settings()
    os.environ.setdefault("OMP_NUM_THREADS", str(settings.reranker_cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(settings.reranker_cpu_threads))

    try:
        from sentence_transformers import CrossEncoder  # lazy import - heavy
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers not installed. "
            "pip install sentence-transformers==3.0.1"
        ) from exc

    _RERANKER = CrossEncoder(settings.reranker_model, max_length=512)
    return _RERANKER


def rerank(query: str, chunks: list[Chunk]) -> list[Chunk]:
    """Batched cross-encoder rerank. Returns chunks sorted by score, descending.

    Cost: ~110 ms p95 on top-30 with bge-reranker-base on a 4-thread CPU.
    """
    if not chunks:
        return []

    settings = get_settings()
    reranker = _get_reranker()

    pairs = [(query, c.text) for c in chunks]
    scores = reranker.predict(pairs, batch_size=settings.reranker_batch_size, show_progress_bar=False)

    scored = []
    for chunk, score in zip(chunks, scores):
        scored.append(chunk.model_copy(update={"score": float(score)}))

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored
