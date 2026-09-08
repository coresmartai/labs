"""Thin HTTP client wrapping the RAG service we're evaluating.

The system under test is treated as a black box. We hit POST /answer and
don't care how it works - that's the whole point. Switching from CitationRAG
to RAGOptimizer to a fine-tuned model is a
config change, not a code change.

Two wire formats are understood transparently:

  BreakRAG /answer (this server):
    { "answer": "...", "retrieved_contexts": ["text1", "text2"],
      "refused": false, "abstained": false, "latency_ms": 150 }

  CitationRAG /answer:
    { "answer": "...", "citations": [...],
      "refused": false, "retrieved_chunks": [{"text": "...", ...}] }
    -> retrieved_chunks is mapped to retrieved_contexts automatically.
"""

from __future__ import annotations

import httpx

from app.config import get_settings
from app.schemas import AdversarialCase, SUTResponse


class SystemUnderTestError(RuntimeError):
    """Raised when the SUT can't be reached or returns malformed JSON."""


async def call_sut(case: AdversarialCase) -> SUTResponse:
    """Send one adversarial prompt to the SUT and return the typed response.

    For the CONFLICT strategy we POST the injected_context alongside the
    prompt - the SUT decides whether to absorb the lie or stay grounded.
    """
    settings = get_settings()
    payload: dict = {"question": case.prompt}
    if case.injected_context is not None:
        payload["adversarial_context"] = case.injected_context

    timeout = httpx.Timeout(settings.sut_timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{settings.sut_base_url}/answer", json=payload
            )
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as exc:
        raise SystemUnderTestError(
            f"SUT call failed for case {case.id}: {exc}"
        ) from exc

    # Support both wire formats:
    #   BreakRAG /answer:    "retrieved_contexts" is a list[str]
    #   W5 CitationRAG:      "retrieved_chunks"  is a list[{text, chunk_id, ...}]
    retrieved_contexts: list[str] = data.get("retrieved_contexts") or []
    if not retrieved_contexts:
        raw_chunks = data.get("retrieved_chunks") or []
        retrieved_contexts = [
            c.get("text", "") for c in raw_chunks if isinstance(c, dict)
        ]

    return SUTResponse(
        answer=data.get("answer", ""),
        retrieved_contexts=retrieved_contexts,
        refused=bool(data.get("refused", False)),
        abstained=bool(data.get("abstained", False)),
        latency_ms=int(data.get("latency_ms", 0)),
    )
