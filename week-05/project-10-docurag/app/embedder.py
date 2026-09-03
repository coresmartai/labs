"""
Text-only embedder wrapping OpenAI text-embedding-3-large.

Copied from Week 4 KnowledgeVault (app/embedder.py) - same embedding model,
same vector dimension (3072), same SHA-256 cache so re-querying the same text
never costs an extra API call.

This module has no week-5-specific logic. Swap the model string in config to
use a different embedding model; the rest of the pipeline is unchanged.
"""

import hashlib
import logging

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level SHA-256-keyed embedding cache (in-process, not persisted).
_cache: dict[str, list[float]] = {}


def embed_text(text: str) -> list[float]:
    """Return the 3072-dim embedding for *text*, using the cache when possible."""
    key = hashlib.sha256(text.encode()).hexdigest()

    if key in _cache:
        logger.debug("Cache hit for text hash %s…", key[:8])
        return _cache[key]

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.debug("Calling OpenAI embeddings API (model=%s)…", settings.embed_model)
    response = client.embeddings.create(
        model=settings.embed_model,
        input=text,
    )
    embedding: list[float] = response.data[0].embedding
    _cache[key] = embedding
    return embedding


def clear_cache() -> None:
    """Clear the in-process embedding cache (useful between tests)."""
    _cache.clear()
    logger.info("Embedding cache cleared.")
