"""Thin SDK wrapper - route handlers never call the SDK directly.

Open-source pathway: replace the OpenAI client with any chat-completion
API or a local model server behind the same function signature.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from app.config import get_settings
from app.schemas import AnswerResponse, Chunk, Citation

logger = logging.getLogger(__name__)

REFUSAL_STRING = "I don't have that information in the provided sources."

SYSTEM_PROMPT = """You are a knowledge assistant. Answer only from the provided context chunks.

FORMAT: respond as JSON matching this schema exactly:
{
  "answer": "...prose answer with [doc#N] markers at the end of each sentence that uses that chunk...",
  "citations": [{"chunk_id": "doc#N", "supporting_quote": "...verbatim text copied from the chunk..."}]
}

CITATION CONTRACT - follow all four rules or your response will be rejected:
1. Every factual claim in the answer prose must end with [doc#N] where N is one of
   the chunk IDs provided below. Do NOT invent chunk IDs.
2. The citations list must contain ONLY chunk IDs that appear as [doc#N] markers
   in your answer. Do not add extra entries to citations that are not in the prose.
3. supporting_quote MUST be a short, exact verbatim substring copied character-for-
   character from that chunk's text. Do not paraphrase or rephrase.
4. One citation entry per [doc#N] marker. Do not repeat the same chunk_id twice.

FALLBACK: if the provided context does not contain enough information to answer,
respond exactly with:
{
  "answer": "I don't have that information in the provided sources.",
  "citations": []
}
Do not use your own training knowledge. Do not approximate."""


def _build_context(chunks: list[Chunk]) -> str:
    """Format chunks with short, stable [doc#N] IDs the model can reference."""
    parts = []
    for ch in chunks:
        parts.append(f"[{ch.chunk_id}]\n{ch.text}\n")
    return "\n".join(parts)


def generate_with_citations(question: str, chunks: list[Chunk]) -> AnswerResponse:
    """Call the pinned LLM with the citation prompt contract.

    Returns an AnswerResponse with the model's claims and citations.
    Caller is responsible for citation validation (see app/citations.py).
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    user_msg = (
        f"Context:\n{_build_context(chunks)}\n\nQuestion: {question}\n\n"
        "Respond as JSON only."
    )

    completion = client.chat.completions.create(
        model=settings.llm_model,
        max_completion_tokens=800,
        temperature=0,               # deterministic - required for consistent citation matching
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    text = completion.choices[0].message.content if completion.choices else "{}"
    try:
        payload: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        # Failure-first: malformed JSON triggers a refusal path,
        # not a crash. Caller sees refused=True and can retry.
        return AnswerResponse(
            answer=REFUSAL_STRING,
            citations=[],
            refused=True,
            validation_passed=False,
        )

    return AnswerResponse(
        answer=payload.get("answer", ""),
        citations=[Citation(**c) for c in payload.get("citations", [])],
        refused=payload.get("answer", "").strip() == REFUSAL_STRING,
    )
