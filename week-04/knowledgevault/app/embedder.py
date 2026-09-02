"""
Text-only embedder wrapping OpenAI ``text-embedding-3-large`` (Option 1).

Design decisions
----------------
Every chunk (prose, table row, or figure) is embedded purely as text.  
Figure chunks arrive here already processed by the vision LLM: their 
``text`` field contains the structured description produced in 
``app/vision.py``, not raw pixel data.  So this module never touches images.

Vector shape
------------
``text-embedding-3-large`` returns 3072-dimensional float vectors with a
cosine-similarity space.  That matches the Qdrant collection created in
``app/indexer.py``.

SHA-256 cache
-------------
Embeddings are expensive ($$$) and deterministic for the same model + input.
We keep a module-level dict keyed by SHA-256(text).  Re-ingesting the same
document a second time - or running tests repeatedly - hits the cache and
never calls the API again.  The cache is in-process only (not persisted to
disk); call ``clear_cache()`` between tests if you want fresh API calls.
"""

import hashlib
import logging

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level SHA-256-keyed embedding cache.
_cache: dict[str, list[float]] = {}


def embed_text(text: str) -> list[float]:
    """Return the 3072-dim embedding for *text*, using the cache when possible.

    Parameters
    ----------
    text:
        The string to embed.  Should be non-empty; the API will error on
        blank input, which is intentional - don't silently embed nothing.

    Returns
    -------
    list[float]
        A 3072-dimensional embedding vector.
    """
    key = hashlib.sha256(text.encode()).hexdigest()

    if key in _cache:
        logger.debug("Cache hit for text hash %s…", key[:8])
        return _cache[key]

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    logger.debug("Calling OpenAI embeddings API (model=%s)…", settings.openai_embed_model)
    response = client.embeddings.create(
        model=settings.openai_embed_model,
        input=text,
    )
    embedding: list[float] = response.data[0].embedding

    _cache[key] = embedding
    logger.debug("Cached embedding for text hash %s…", key[:8])

    return embedding


def clear_cache() -> None:
    """Clear the in-process embedding cache.

    Useful between test runs or when switching API keys / models so that
    stale vectors are not accidentally reused.
    """
    _cache.clear()
    logger.info("Embedding cache cleared.")
