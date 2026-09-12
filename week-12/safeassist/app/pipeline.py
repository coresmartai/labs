"""The shared /ask pipeline - one code path for the live route and the eval harness.

`app/main.py::ask` streams these steps to the browser over SSE; `app/eval.py`
runs the same steps headlessly and scores the outcome. They MUST agree on what
the service actually does - a security demo whose eval tests a different code
path proves nothing - so the retrieval (ACL + relevance floor + injection
sanitiser) and the generation (model + streaming output scrub) live here, once.

What is deliberately NOT here: the input scrub and the injection classifier.
Those are the first two steps of the request and both routes run them directly,
because they decide whether retrieval happens at all.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterator

from app import store
from app.config import get_settings
from app.injection import sanitize_chunk
from app.llm import build_system_prompt, stream_answer
from app.presidio_layer import SentenceBufferScrubber
from app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

# Signature of the audit callback both callers pass in: (event_type, detail, action_taken).
AuditLog = Callable[[str, dict, str], None]


def retrieve(query: str, roles: list[str], log: AuditLog) -> list[RetrievedChunk]:
    """Hybrid retrieval (dense + BM25 + RRF) with the ACL filter pushed down.

    NOTE WHAT IS *NOT* HERE: there is no PII scrub. The chunks in the store
    were already scrubbed by `app/ingest.py` at write time. Scrubbing on read
    would leave the store itself full of raw PII.

    The injection sanitiser DOES run here, and that is not an inconsistency: a
    poisoned sentence is only dangerous when it reaches the prompt, so read
    time is the right place to strip it - and it keeps the audit trail of what
    the store actually contains.
    """
    settings = get_settings()

    # The ACL filter is enforced inside Qdrant, on BOTH retrieval channels, and
    # a dense-cosine relevance floor keeps this to the chunks actually about the
    # question. So an empty result is the honest "no relevant, visible context"
    # - which is also the ACL demo's punchline: an analyst asking a CEO question
    # retrieves nothing (the one relevant chunk is ACL-hidden).
    hits, _top1, _spread = store.search(query, roles=roles)

    # Report what the ACL held back: how many chunks that WOULD have been
    # relevant are hidden from these roles. One dense query (the query
    # embedding is cached), never a second gate.
    withheld = store.count_withheld(query, roles)
    if withheld > 0:
        log(
            "acl_filter",
            {"roles": roles, "chunks_withheld": withheld},
            "chunks_withheld",
        )

    cleaned: list[RetrievedChunk] = []
    for hit in hits:
        text = hit.text
        if settings.retrieval_sanitiser_enabled:
            text, stripped = sanitize_chunk(text)
            if stripped:
                log(
                    "injection_refusal",
                    {
                        "classifier": "regex_set",
                        "category": stripped[0],
                        "confidence": 0.92,
                        "surface": "retrieval",
                    },
                    # Nothing was refused - a sentence was cut out of a
                    # document. Say what happened, not what sounds tidy.
                    "instruction_text_stripped",
                )
        cleaned.append(hit.model_copy(update={"text": text}))
    return cleaned


def generate_sentences(query: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
    """Stream the model's answer, one output-scrubbed sentence at a time.

    Each yielded string has already passed through the SentenceBufferScrubber,
    so PII that the model emits never leaves this generator unredacted. The
    live route SSE-frames each sentence as it arrives; the eval harness joins
    them into the full answer. Same tokens, same scrub, same order.
    """
    system_prompt = build_system_prompt([c.text for c in chunks])
    buffer = SentenceBufferScrubber()
    for token in stream_answer(system_prompt=system_prompt, user_message=query):
        scrubbed_sentence = buffer.push(token)
        if scrubbed_sentence:
            yield scrubbed_sentence
    tail = buffer.flush()
    if tail:
        yield tail
