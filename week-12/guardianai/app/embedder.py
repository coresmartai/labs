"""Text embedder.

`text-embedding-3-large`, 3072 dimensions, cosine space - the same model the
CitationRAG retriever indexes with, so ingestion and query embed identically.

Two entry points:
  embed_text(str)         -> one vector. Used by every query.
  embed_texts(list[str])  -> many vectors, ONE API call. Used by ingestion.

The batch call matters: the seed corpus is twenty chunks, and one request
beats twenty round-trips by roughly the latency of nineteen HTTP calls.

SHA-256 cache: embeddings are deterministic for a given model + input, so the
in-process cache means re-ingesting the same corpus, or asking the same
question twice, costs nothing the second time. The cache key includes the
model name - switching models must not silently return the old vectors.

Changing `OPENAI_EMBED_MODEL` changes the vector width. Qdrant collections are
fixed-width, so a model change requires a re-index:
`python -m app.ingest --reset`. `store.ensure_collection()` refuses to write
into a collection whose width disagrees rather than failing deep in the client.
"""
from __future__ import annotations

import hashlib
import logging

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

_cache: dict[str, list[float]] = {}


def _client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key, timeout=settings.request_timeout_seconds)


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many strings in one API call. Cached per (model, text).

    Only the cache misses are sent; the results are stitched back into the
    caller's original order.
    """
    if not texts:
        return []

    settings = get_settings()
    model = settings.openai_embed_model
    prepared = [t.replace("\n", " ") for t in texts]

    out: list[list[float] | None] = [None] * len(prepared)
    misses: list[int] = []
    for i, text in enumerate(prepared):
        hit = _cache.get(_key(model, text))
        if hit is None:
            misses.append(i)
        else:
            out[i] = hit

    if misses:
        logger.debug("OpenAI embeddings API - model=%s, %d input(s)", model, len(misses))
        resp = _client().embeddings.create(model=model, input=[prepared[i] for i in misses])
        # The API preserves input order, but `index` is authoritative - use it.
        for item in resp.data:
            i = misses[item.index]
            vector = list(item.embedding)
            _cache[_key(model, prepared[i])] = vector
            out[i] = vector

    return [v for v in out if v is not None]


def embed_text(text: str) -> list[float]:
    """Return the embedding for *text*."""
    return embed_texts([text])[0]


def clear_cache() -> None:
    """Clear the in-process cache. Useful between test runs."""
    _cache.clear()
    logger.info("Embedding cache cleared.")
