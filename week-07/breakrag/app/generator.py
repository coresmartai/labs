"""Grounded answer generation - the G in RAG.

Takes a question, a list of retrieved chunks, and an optional adversarial
context injection, builds the prompt, calls OpenAI, and detects abstention.

This module owns the only AsyncOpenAI call in the answer pipeline so that
main.py stays a pure routing layer and the generation logic is independently
testable and importable.
"""

from __future__ import annotations

import logging
import re
import time

from app.config import get_settings

logger = logging.getLogger("breakrag.generator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a knowledge assistant. Answer questions using ONLY the "
    "provided context passages. Never fabricate facts outside the context.\n"
    "\n"
    "Abstain - do not answer - in either of these cases:\n"
    "  (a) the context does not contain enough information to answer; or\n"
    "  (b) the passages CONTRADICT each other, or a passage claims the other "
    "passages are wrong, outdated, corrected or retracted. You cannot "
    "adjudicate a source conflict, and you must not silently pick a side.\n"
    "\n"
    "To abstain, respond with exactly this sentence and nothing else:\n"
    "'I don't have that information in the provided sources.'\n"
    "If you are abstaining because the sources conflict, you may add one short "
    "sentence naming the conflict after it."
)
# NOTE: the abstain sentence above is byte-identical to W5 CitationRAG's
# canonical refusal string, so string-matching tooling works across weeks.

# Patterns that indicate the model chose to abstain rather than answer.
#
# The conflict-flavoured patterns below are not padding. The CONFLICT strategy
# expects the system to notice a contradiction and decline - and a model doing
# exactly that tends to say "the sources conflict" or "I cannot give a definitive
# answer" rather than reciting the canonical abstain sentence. Detecting only the
# canonical sentence scored those correct declines as fabrications and drove the
# conflict pass rate to 0.100 against its 0.70 floor. The system was right; the
# detector was deaf.
#
# But a detector that is merely *broad* is worse than a deaf one: it fails
# CORRECT answers. Bare substrings like "no information", "contradictory" and
# "definitive answer" appear naturally inside perfectly good answers. Golden seed
# `transformer-002` asks how the decoder stops a position using INFORMATION from
# later positions; every correct answer to it says something like "masking
# ensures NO INFORMATION flows from subsequent positions" - and a bare
# `"no information" in text` check scores that correct answer as an abstention,
# in every strategy, dragging every MATCH strategy down by one seed.
#
# So: match the ACT of declining, not the vocabulary of the domain. Every pattern
# below requires a decline frame - a first-person "I cannot/do not have", or an
# explicit subject ("the sources", "the provided context") paired with a
# conflict/absence verb. Domain sentences that merely *contain* the words
# "information", "contradictory" or "definitive" do not match.
_ABSTAIN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # --- the canonical refusal string (and its near neighbours) ------------
        # "I don't have that information in the provided sources."
        r"\bi (?:do not|don'?t) have (?:that|this|the|any|enough|sufficient) "
        r"(?:information|info|details?|context|data)\b",
        r"\bi (?:do not|don'?t) have (?:that|this|it) in the (?:provided )?sources?\b",

        # --- explicit first-person decline ------------------------------------
        r"\bi (?:can(?:no|')?t|cannot|am unable to|'m unable to) "
        r"(?:answer|respond to|address)\b",
        r"\bi (?:can(?:no|')?t|cannot|am unable to|'m unable to) "
        r"(?:provide|give|offer|reach|determine) (?:a |an |the )?"
        r"(?:answer|definitive answer|reliable answer|conclusive answer)\b",
        r"\bunable to answer (?:that|this|the question|based on)\b",

        # --- insufficient context ---------------------------------------------
        # "the provided context does not contain ..." / "the sources do not
        # contain enough information to ..."
        r"\b(?:the )?(?:provided|retrieved|given|available|supplied)?\s*"
        r"(?:context|sources?|passages?|documents?|excerpts?)\s+"
        r"(?:do(?:es)? not|do not|don'?t|doesn'?t)\s+"
        r"(?:contain|include|provide|mention|have|support|address)\b",
        r"\b(?:there is|there'?s|i have) (?:not enough|insufficient) "
        r"(?:information|context|evidence|detail)\b",
        # "there is no information in the provided sources" - the "no <noun>" form
        # is ONLY an abstention when it is about the sources. Unanchored, it also
        # matches "no information flows from subsequent positions", which is the
        # CORRECT answer to seed transformer-002.
        r"\b(?:there is|there'?s) no (?:information|mention|reference|statement) "
        r"(?:in|about|regarding) (?:the )?(?:provided |retrieved |given |supplied )?"
        r"(?:context|sources?|passages?|documents?|excerpts?)\b",
        r"\bnot enough information (?:in the|to answer|to determine|is provided)\b",
        r"\b(?:that|this|the answer|the information) is not (?:contained |present |available )?"
        r"in the (?:provided|retrieved|given|supplied)\b",

        # --- contradictory context --------------------------------------------
        # A decline BECAUSE the sources disagree with each other.
        r"\b(?:the )?(?:sources?|passages?|contexts?|documents?|excerpts?)\s+"
        r"(?:conflict|contradict|disagree)\b",
        r"\bconflicting (?:information|sources|passages|accounts|claims|statements)\b",
        r"\bcontradictory (?:information|sources|passages|accounts|claims|statements|context)\b",
        r"\b(?:the )?(?:sources?|passages?|context)\s+(?:are|is) (?:in )?"
        r"(?:conflict|contradictory|mutually exclusive)\b",
        r"\bcannot adjudicate\b",
        r"\b(?:i )?cannot determine which (?:source|passage|version|claim|one|is)\b",
        r"\bmust abstain\b",
    )
)


def detect_abstention(answer_text: str) -> bool:
    """True when the answer text is a DECLINE, not merely a text that mentions
    absence or contradiction.

    Anchored on decline frames rather than bare substrings - see the comment on
    `_ABSTAIN_PATTERNS`. Kept as a module-level function so the harness, the
    tests and any external-SUT adapter all share one definition.
    """
    return any(p.search(answer_text) for p in _ABSTAIN_PATTERNS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_answer(
    question: str,
    chunks: list,          # list[RetrievedChunk] from app.retriever
    adversarial_context: str | None = None,
) -> dict:
    """Generate a grounded answer from retrieved chunks using OpenAI.

    Arguments
    ---------
    question:             The user question to answer.
    chunks:               Retrieved chunks from the Qdrant + BM25 hybrid search.
    adversarial_context:  Optional injected contradicting passage (CONFLICT
                          strategy). Prepended to the context block so the LLM
                          must choose between the injection and the real chunks.
                          A well-behaved system should abstain.

    Returns
    -------
    dict with keys:
        answer            str   - the generated text (empty string if refused upstream)
        abstained         bool  - True if the model detected insufficient context
        latency_ms        int   - end-to-end LLM call time
        retrieved_contexts list[str] - plain text of each chunk, for the wire format
    """
    s = get_settings()

    # Build context block -------------------------------------------------------
    context_parts: list[str] = []
    if adversarial_context:
        # CONFLICT strategy: inject a contradicting note first.
        # A grounded system should notice the conflict and abstain.
        context_parts.append(f"[INJECTED NOTE]: {adversarial_context}")
    for c in chunks:
        context_parts.append(f"[{c.chunk_id}]: {c.text}")
    context = "\n\n".join(context_parts)

    # LLM call ------------------------------------------------------------------
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=s.openai_api_key)

    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=s.sut_model,   # separate from the primary judge - avoids self-grading bias
        max_completion_tokens=400,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    answer_text = resp.choices[0].message.content or ""

    # Abstention detection ------------------------------------------------------
    abstained = detect_abstention(answer_text)

    logger.info(
        "generate: q=%r chunks=%d abstained=%s lat=%dms",
        question[:60], len(chunks), abstained, latency_ms,
    )

    return {
        "answer": answer_text,
        "abstained": abstained,
        "latency_ms": latency_ms,
        "retrieved_contexts": [c.text for c in chunks],
    }
