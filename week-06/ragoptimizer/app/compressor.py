"""Context compressor - extractive, sentence-level.

For each chunk we keep, drop sentences that don't contribute to answering
the current query. Cheap, deterministic, no extra model call. The chunk-
ID marker line is ALWAYS preserved - citation validation in app/main.py
depends on it.

Failure mode this module guards against: MARKER STRIPPING - naively
scoring sentences against the query can cause the chunk's first line
(the marker) to score low and get dropped, which breaks every downstream
citation. The fix below: anchor the marker line outside of scoring; only
the body sentences are eligible for dropping.
"""
from __future__ import annotations

import re

from app.config import get_settings
from app.schemas import Chunk


# Tokenizer-aware-ish sentence splitter. Handles version numbers like
# "v1.2.3" without treating each dot as a sentence boundary.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(body: str) -> list[str]:
    body = body.strip()
    if not body:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(body) if s.strip()]


def _score_sentence_against_query(sentence: str, query: str) -> float:
    """Simple lexical overlap score in [0, 1]. The lab swaps this for an
    embedding-similarity score using the same encoder as retrieval."""
    q_tokens = set(query.lower().split())
    s_tokens = sentence.lower().split()
    if not s_tokens:
        return 0.0
    overlap = sum(1 for t in s_tokens if t in q_tokens)
    return overlap / max(len(s_tokens), 1)


def compress_chunk(chunk: Chunk, query: str) -> Chunk:
    """Return a new Chunk whose body keeps only the top-fraction of sentences.

    The marker line (first line of chunk.text) is always preserved. Sentences
    in the body are scored against the query; top-K kept by score, in their
    ORIGINAL ORDER (not reranked locally).
    """
    settings = get_settings()
    if not settings.compressor_enabled:
        return chunk

    lines = chunk.text.split("\n", 1)
    marker = lines[0]
    body = lines[1] if len(lines) > 1 else ""

    sentences = _split_sentences(body)
    if not sentences:
        return chunk

    scored = [(i, s, _score_sentence_against_query(s, query)) for i, s in enumerate(sentences)]
    keep_n = max(1, int(round(len(sentences) * settings.compressor_keep_fraction)))
    top = sorted(scored, key=lambda t: t[2], reverse=True)[:keep_n]
    top_sorted = sorted(top, key=lambda t: t[0])

    new_body = " ".join(s for _, s, _ in top_sorted)
    return chunk.model_copy(update={"text": f"{marker}\n{new_body}"})


def compress(chunks: list[Chunk], query: str) -> list[Chunk]:
    """Apply compress_chunk to a list of chunks. Marker lines always survive."""
    return [compress_chunk(c, query) for c in chunks]
