"""Text embedder - mirrors Week 4 KnowledgeVault exactly for consistency.

Uses ``text-embedding-3-large`` (3072 dimensions, cosine space) - the same
model and collection shape used during ingestion in Week 4. Swapping this
model without re-indexing will produce dimension-mismatch errors in Qdrant.

SHA-256 cache: embeddings are deterministic for a given model + input, so we
cache in-process to avoid duplicate API calls within a request cycle (HyDE
probe and original query may embed the same text if HyDE is disabled).
"""
from __future__ import annotations

import hashlib
import logging

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

_cache: dict[str, list[float]] = {}


def embed_text(text: str) -> list[float]:
    """Return the 3072-dim embedding for *text*, using the SHA-256 cache when possible.

    Matches Week 4's ``app.embedder.embed_text`` signature exactly.
    """
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in _cache:
        logger.debug("Embedding cache hit %s…", key[:8])
        return _cache[key]

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    logger.debug("OpenAI embeddings API - model=%s", settings.openai_embed_model)
    resp = client.embeddings.create(
        model=settings.openai_embed_model,
        input=text.replace("\n", " "),
    )
    embedding: list[float] = resp.data[0].embedding
    _cache[key] = embedding
    return embedding


def clear_cache() -> None:
    """Clear the in-process cache. Useful between test runs."""
    _cache.clear()
    logger.info("Embedding cache cleared.")
