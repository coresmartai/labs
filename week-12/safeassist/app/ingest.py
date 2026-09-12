"""Ingestion - where the PII scrub actually lives.

This is the point of H2: the chunk scrub is a WRITE-time defence, not a
read-time one. Every chunk that goes into Qdrant passes through Presidio
first, so unredacted PII never lands in the store. If the scrub only ran on
retrieval, the store would still be a PII database - and the store is the
artefact that outlives the request, gets backed up, and gets subpoenaed.

The read-time defence that DOES stay on read is the injection sanitiser
(`injection.sanitize_chunk`, applied in `main._retrieve`). Different threat,
different surface: a poisoned document is dangerous when it reaches the
prompt, not when it sits in a row.

Three entry points:
  ingest(chunks, reset=False)            - the library call.
  ingest_document(text, source, roles)   - chunk a raw document, then ingest.
  main()                                 - `python -m app.ingest [--reset]`.
"""
from __future__ import annotations

import argparse
import logging
import re
import uuid

from app import audit
from app.config import get_settings
from app.corpus import SEED_CHUNKS
from app.presidio_layer import is_redacted, scrub_chunk
from app.schemas import IngestReport, SeedChunk, StoredChunk
from app.store import upsert_chunks, reset as store_reset

logger = logging.getLogger(__name__)

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_document(text: str, source: str, visible_to: list[str], *, chunk_chars: int | None = None) -> list[SeedChunk]:
    """Split a document into chunks on paragraph, then sentence, boundaries.

    Deliberately boring. Chunking strategy is a topic of its own; here it only
    has to be deterministic so the same upload always produces the same chunks.
    """
    settings = get_settings()
    limit = chunk_chars or settings.chunk_chars
    stem = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-") or "doc"

    pieces: list[str] = []
    for para in _PARA_SPLIT.split(text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= limit:
            pieces.append(para)
            continue
        buf = ""
        for sentence in _SENT_SPLIT.split(para):
            if buf and len(buf) + len(sentence) + 1 > limit:
                pieces.append(buf.strip())
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf.strip():
            pieces.append(buf.strip())

    return [
        SeedChunk(
            chunk_id=f"{stem}-{i + 1:03d}",
            text=piece,
            source=source,
            visible_to=list(visible_to),
        )
        for i, piece in enumerate(pieces)
    ]


def ingest(
    chunks: list[SeedChunk] | None = None,
    *,
    reset: bool = False,
    request_id: str | None = None,
    user_id: str = "ingest-job",
) -> IngestReport:
    """Scrub -> embed -> upsert. In that order, always.

    Returns how many chunks were written and how many PII spans Presidio
    redacted on the way in. Every redaction writes one `pii_scrub` audit event
    with surface="retrieval" - the entity type and span length, never the
    value.
    """
    settings = get_settings()
    seeds = SEED_CHUNKS if chunks is None else chunks
    rid = request_id or f"ingest-{uuid.uuid4().hex[:8]}"

    if reset:
        store_reset()

    stored: list[StoredChunk] = []
    spans_redacted = 0

    for seed in seeds:
        text = seed.text
        if settings.chunk_scrub_enabled:
            result = scrub_chunk(seed.text)
            text = result.cleaned_text
            for det in result.detections:
                redacted = is_redacted(det)
                spans_redacted += int(redacted)
                audit.write_event(
                    request_id=rid,
                    user_id=user_id,
                    event_type="pii_scrub",
                    detail={
                        "recognizer": det.recognizer,
                        "confidence": det.confidence,
                        "span_length": det.span_end - det.span_start,
                        "surface": "retrieval",
                        "operator": settings.presidio_operator,
                    },
                    # Above the redact threshold we transform. Between log-only
                    # and redact we record and leave the text alone - the
                    # human-review queue.
                    action_taken="redacted" if redacted else "logged_for_review",
                )
        stored.append(
            StoredChunk(
                chunk_id=seed.chunk_id,
                text=text,
                source=seed.source,
                visible_to=seed.visible_to,
                scrubbed=settings.chunk_scrub_enabled,
            )
        )

    written = upsert_chunks(stored)
    return IngestReport(
        chunks=written,
        spans_redacted=spans_redacted,
        chunk_scrub=settings.chunk_scrub_enabled,
        collection=settings.qdrant_collection,
    )


def ingest_document(
    text: str,
    source: str,
    visible_to: list[str],
    *,
    request_id: str | None = None,
    user_id: str = "ingest-api",
) -> IngestReport:
    """Chunk a raw document and ingest it. Backs POST /ingest/text and
    POST /ingest/file."""
    chunks = chunk_document(text, source, visible_to)
    return ingest(chunks, request_id=request_id, user_id=user_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the seed corpus into Qdrant.")
    parser.add_argument("--reset", action="store_true", help="drop and recreate the collection first")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s - %(message)s")
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

    report = ingest(reset=args.reset)
    print(
        f"ingested {report.chunks} chunks "
        f"· chunk_scrub={'on' if report.chunk_scrub else 'OFF'} "
        f"· {report.spans_redacted} PII spans redacted "
        f"· collection={report.collection}"
    )


if __name__ == "__main__":
    main()
