"""Citation validation - deterministic guardrails on the model's output.

Two checks run in sequence after every LLM call:
  1. ID presence - every [doc#N] in the answer prose must map to a chunk that was
     actually sent to the model. Orphan payload citations (cited in JSON but not
     referenced in the prose) are stripped silently rather than failing the response.
  2. Quote substring - each citation's supporting_quote must appear verbatim
     (whitespace-normalised, case-insensitive) in the cited chunk's text.

Both checks set validation_passed=False on failure. main.py converts that to a
conservative refusal - never ships a bad citation to the user.
"""
from __future__ import annotations

import logging
import re

from app.schemas import AnswerResponse, Chunk

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[doc#\d+\]")


def parse_citation_ids(answer_text: str) -> list[str]:
    """Pull every [doc#N] marker out of the answer prose."""
    return CITATION_RE.findall(answer_text)


def validate(response: AnswerResponse, chunks: list[Chunk]) -> AnswerResponse:
    """Check that every cited chunk ID exists in the retrieved set,
    and that the supporting_quote substring appears in the cited chunk.

    Mutates response.validation_passed and returns it.
    """
    if response.refused:
        response.validation_passed = True
        return response

    chunks_by_id = {f"[{c.chunk_id}]": c for c in chunks}

    # Check 1a: every [doc#N] marker in the answer prose is a real chunk ID.
    # This is the critical check - the prose is what students read.
    cited_in_prose = set(parse_citation_ids(response.answer))
    for cid in cited_in_prose:
        if cid not in chunks_by_id:
            response.validation_passed = False
            response.validation_detail = (
                f"Fabricated chunk ID {cid} in answer prose - not in the "
                f"{len(chunks_by_id)} chunks sent to the model "
                f"({', '.join(sorted(chunks_by_id)[:5])}…)"
            )
            return response

    # Check 1b: citations payload entries that are NOT in the prose are orphans
    # (model added them without citing them).  Strip orphans silently rather than
    # failing - the prose is the contract; extra list entries are model sloppiness.
    cited_in_payload = {f"[{c.chunk_id}]" for c in response.citations}
    orphans = cited_in_payload - cited_in_prose
    if orphans:
        logger.warning(
            "Stripping %d orphan citation(s) not referenced in prose: %s",
            len(orphans), orphans,
        )
        response.citations = [
            c for c in response.citations
            if f"[{c.chunk_id}]" in cited_in_prose
        ]

    # Remaining check: every payload citation must reference a real chunk ID
    for c in response.citations:
        cid = f"[{c.chunk_id}]"
        if cid not in chunks_by_id:
            response.validation_passed = False
            response.validation_detail = (
                f"Citation payload references {cid} but it is not "
                "in the retrieved set."
            )
            return response

    # Check 2: each citation's supporting_quote actually appears in the chunk.
    # Normalise whitespace before comparing so PDF-extracted newlines / multiple
    # spaces don't cause false failures.
    _ws = re.compile(r"\s+")

    for c in response.citations:
        chunk_key = f"[{c.chunk_id}]"
        if chunk_key not in chunks_by_id:
            response.validation_passed = False
            response.validation_detail = (
                f"Citation payload references {chunk_key} but it is not "
                "in the retrieved set."
            )
            return response
        chunk_text = chunks_by_id[chunk_key].text
        quote = c.supporting_quote.strip()
        if quote:
            norm_quote = _ws.sub(" ", quote).lower()
            norm_chunk = _ws.sub(" ", chunk_text).lower()
            if norm_quote not in norm_chunk:
                response.validation_passed = False
                response.validation_detail = (
                    f"supporting_quote for {chunk_key} not found verbatim in chunk text. "
                    f"Quote: \"{quote[:120]}{'…' if len(quote) > 120 else ''}\""
                )
                return response

    response.validation_passed = True
    return response
