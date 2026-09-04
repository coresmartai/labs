"""Citation-grounded answer generation.

Sends the retrieved chunks to gpt-5.4-mini-2026-03-17 with a strict citation
prompt. Every factual claim in the answer must be backed by a (chunk_id, quote)
pair pointing to one of the retrieved chunks.

Flow:
    chunks  →  _build_messages()  →  generate_json()  →  _parse_and_validate()  →  GroundedAnswer

Citation validation (the W6 contract):
    Every chunk_id in citations must exist in the set of retrieved chunks.
    Citations that reference unknown chunk_ids are silently dropped.
    This prevents hallucinated citations from leaking into responses.
"""
from __future__ import annotations

import logging

from app.llm import generate_json
from app.schemas import Chunk, Citation, GroundedAnswer, PipelineTrace

logger = logging.getLogger("ragoptimizer.citation")

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precise document Q&A assistant.

RULES - follow all of them without exception:
1. Answer using ONLY the information present in the provided chunks.
   Never add facts from outside knowledge.
2. Every factual claim must have at least one citation.
3. A citation is: {"chunk_id": "<exact id>", "quote": "<verbatim phrase from that chunk>"}.
   Quotes must appear word-for-word in the chunk text. Keep them under 30 words.
4. Set "confidence" to:
     "high"   - chunks fully and directly answer the question
     "medium" - chunks partially answer; some inference needed
     "low"    - chunks lack sufficient information
5. Set "fallback_triggered" to true when you cannot answer from the chunks alone.
6. Output ONLY valid JSON - no markdown fences, no prose outside the JSON object.\
"""

# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_messages(query: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    # Each chunk's text already starts with "[chunk-id]\n..." from _payload_to_chunk.
    # We pass it directly so the model can read the IDs inline.
    context = "\n\n---\n".join(c.text for c in chunks)

    user_content = f"""\
Question: {query}

Retrieved chunks:
---
{context}
---

Respond with this exact JSON structure (output JSON only):
{{
  "answer": "Your answer here, written in full sentences.",
  "citations": [
    {{"chunk_id": "the-chunk-id", "quote": "verbatim phrase from that chunk"}}
  ],
  "confidence": "high",
  "fallback_triggered": false
}}\
"""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ── Citation validator ────────────────────────────────────────────────────────

def _validate_citations(
    raw: list,
    valid_ids: set[str],
) -> tuple[list[Citation], int]:
    """Keep only citations whose chunk_id is in the retrieved set.

    Returns (kept, dropped_count). The count is what makes citation validity
    measurable: a validator that silently repairs its input leaves nothing to
    score, and a metric computed downstream of it cannot fail. Measure upstream
    of the enforcement, where failure is still possible.

    Caps quote length so the response stays compact. Invalid shapes (missing
    chunk_id) are dropped but are a parse failure rather than a fabrication, so
    they are not counted against citation discipline.
    """
    result: list[Citation] = []
    seen: set[str] = set()
    dropped = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("chunk_id", "")).strip()
        quote = str(item.get("quote", "")).strip()
        if not cid or cid not in valid_ids or cid in seen:
            if cid and cid not in valid_ids:
                dropped += 1
                logger.warning("Citation dropped - chunk_id not in retrieved set: %s", cid)
            continue
        seen.add(cid)
        result.append(Citation(chunk_id=cid, quote=quote[:300]))
    return result, dropped


# ── Main entry point ──────────────────────────────────────────────────────────

def ask_with_citations(
    query: str,
    chunks: list[Chunk],
    trace: PipelineTrace | None = None,
) -> GroundedAnswer:
    """Call the LLM with the citation prompt and return a validated GroundedAnswer.

    Never raises - returns a fallback GroundedAnswer on any model/parse error
    so the /ask route always gets a well-formed response.
    """
    valid_ids = {c.chunk_id for c in chunks}
    messages = _build_messages(query, chunks)

    logger.info(
        "Citation prompt: %d chunks, query=%.60s",
        len(chunks), query,
    )

    try:
        raw = generate_json(messages, max_tokens=1024)
    except Exception as exc:
        logger.error("LLM call failed in ask_with_citations: %s", exc)
        return GroundedAnswer(
            answer="The model call failed - check OPENAI_API_KEY and retry.",
            citations=[],
            confidence="low",
            fallback_triggered=True,
            pipeline_trace=trace,
        )

    # Parse fields defensively
    answer: str = str(raw.get("answer", "")).strip()
    raw_conf = raw.get("confidence", "low")
    confidence = raw_conf if raw_conf in ("high", "medium", "low") else "low"
    fallback: bool = bool(raw.get("fallback_triggered", False))
    citations, dropped = _validate_citations(raw.get("citations", []), valid_ids)

    if not answer:
        answer = "The retrieved chunks did not contain enough information to answer this question."
        fallback = True

    logger.info(
        "Answer generated - confidence=%s, citations=%d, dropped=%d, fallback=%s",
        confidence, len(citations), dropped, fallback,
    )

    return GroundedAnswer(
        answer=answer,
        citations=citations,
        confidence=confidence,
        fallback_triggered=fallback,
        pipeline_trace=trace,
        generated=True,
        citations_dropped=dropped,
    )
