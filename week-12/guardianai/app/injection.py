"""Prompt-injection classifier.

Regex-based - small, fast, model-agnostic, and a deliberately swappable seam:
an LLM-based classifier can replace it behind the same return shape without
touching the route. The eval set in tests/ exercises both attack patterns and
known false positives.
"""
from __future__ import annotations

import logging
import re

from app.schemas import InjectionVerdict

logger = logging.getLogger(__name__)

# Each pattern fires a flagged=True verdict. The category names show
# up in the audit log so analysts can see what fired in production.
_PATTERNS: list[tuple[str, str, float]] = [
    # (regex, category, confidence)
    (r"(?i)\b(ignore|disregard|forget)\b\s+(all\s+|previous\s+|prior\s+)?instructions", "instruction_override", 0.92),
    (r"(?i)\breveal\s+(the\s+)?system\s+prompt", "system_prompt_leak", 0.95),
    (r"(?i)\b(your|the)\s+(initial|original)\s+(prompt|instructions)", "system_prompt_leak", 0.85),
    (r"(?i)\bact\s+as\s+(an?\s+)?(unrestricted|uncensored|jailbroken)", "role_override", 0.88),
    # Case-sensitive on purpose: "DAN" the jailbreak, not a user named Dan.
    (r"\bDAN\b", "jailbreak_named", 0.80),
    (r"(?i)\bbase64\s*[:=]\s*[A-Za-z0-9+/=]{20,}", "encoded_payload", 0.78),
]

_compiled = [(re.compile(p), c, s) for (p, c, s) in _PATTERNS]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STRIP_TOKEN = "[removed: suspected instruction]"


def sanitize_chunk(text: str) -> tuple[str, list[str]]:
    """Strip instruction-like text from a retrieved chunk.

    Same regex set as `classify`, second surface: retrieved content is
    untrusted data, so any sentence that matches an instruction pattern
    is dropped before the chunk reaches the prompt. Returns the cleaned
    text plus the categories that fired (for the audit writer).
    """
    kept: list[str] = []
    stripped: list[str] = []
    for sentence in _SENT_SPLIT.split(text):
        cats = [category for rx, category, _score in _compiled if rx.search(sentence)]
        if cats:
            stripped.extend(cats)
            kept.append(_STRIP_TOKEN)
        else:
            kept.append(sentence)
    return " ".join(kept), stripped


def classify(text: str) -> InjectionVerdict:
    """Return a verdict. False if no pattern fires."""
    for rx, category, score in _compiled:
        m = rx.search(text)
        if m:
            return InjectionVerdict(
                flagged=True,
                category=category,
                confidence=score,
                matched_pattern=rx.pattern,
            )
    return InjectionVerdict(flagged=False)
