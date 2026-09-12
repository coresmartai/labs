"""The human-review queue: what the log-only band should have been.

TASK 4. Three functions raise NotImplementedError. The state machine, the
record shape, mask_span(), load() and stats() are given; the transitions are
yours. Read the module docstring below before you start, because the three
properties it names are what the tests check.


GuardianAI's 0.35 to 0.65 confidence band writes an audit line saying a
detection was `logged_for_review` and then does nothing. There is no queue, no
reviewer, no state and nothing a person can act on. Week 12's title promises
human-in-the-loop; this module is what makes that true.

A review record is a small state machine:

    pending ---- approve ----> approved      (terminal: the text was fine)
       |
       +-------- reject -----> rejected      (terminal: redact it after all)
       |
       +-- age past the TTL -> escalated     (terminal for the reviewer,
                                              actionable for their manager)

Three properties matter more than the state names.

**A record never carries the detected value.** The same rule as the audit log,
for the same reason: a review queue holding raw personal data is the repository
the scrubber exists to avoid, with a friendlier user interface. A reviewer sees
the entity type, the confidence, the span offsets and the surrounding text with
the span itself masked. If they need the value they open the source under its
own access controls.

**A queue with no timeout is a backlog.** Records that sit in `pending` past
`review_ttl_hours` are escalated by `sweep()`, which is intended to run on a
schedule. Without this, the band's promise of review degrades into the same
silence it replaced, only now with a database behind it.

**Append-only, like the audit log.** State changes append a new record rather
than rewriting the old one, so the history of a decision survives. `load()`
folds the log into current state by taking the last record per id.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import get_settings

ReviewState = Literal["pending", "approved", "rejected", "escalated"]

MASK = "[span]"


class ReviewRecord(BaseModel):
    """One row of the queue. Never carries the detected value."""

    review_id: str
    created_at: str
    updated_at: str
    state: ReviewState
    reviewer: str
    request_id: str
    user_id_hash: str
    entity_type: str
    confidence: float
    span_start: int
    span_end: int
    # The sentence the span sat in, with the span itself masked. Enough context
    # for a human to judge whether the detection was right; not enough to
    # reconstruct the value.
    masked_context: str
    surface: str
    note: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def mask_span(text: str, start: int, end: int, window: int = 60) -> str:
    """Return the text around a span with the span replaced by a marker.

    The window is deliberately small. A reviewer needs enough to judge whether
    'Whitfield' is a surname or a ward name; they do not need the paragraph.
    """
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return text[lo:start] + MASK + text[end:hi]


def _append(record: ReviewRecord) -> None:
    settings = get_settings()
    line = record.model_dump_json() + "\n"
    fd = os.open(settings.review_queue_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def enqueue(*, request_id: str, user_id_hash: str, entity_type: str, confidence: float,
            span_start: int, span_end: int, source_text: str, surface: str) -> ReviewRecord:
    """Create a pending review record for one log-only-band detection."""
    raise NotImplementedError(
        "TASK 4: create a pending review record. It must carry the entity type, "
        "the confidence, the span offsets and a MASKED context, and it must "
        "never carry the detected value. Use mask_span()."
    )

def load() -> dict[str, ReviewRecord]:
    """Fold the append-only log into current state: last record per id wins."""
    settings = get_settings()
    out: dict[str, ReviewRecord] = {}
    if not os.path.exists(settings.review_queue_path):
        return out
    with open(settings.review_queue_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = ReviewRecord.model_validate_json(line)
            out[rec.review_id] = rec
    return out


def pending() -> list[ReviewRecord]:
    return [r for r in load().values() if r.state == "pending"]


def decide(review_id: str, *, decision: Literal["approve", "reject"], reviewer: str,
           note: str = "") -> ReviewRecord:
    """Move a pending record to a terminal state. Only pending records move."""
    raise NotImplementedError(
        "TASK 4: move a PENDING record to approved or rejected. A record that "
        "is already decided must raise ValueError rather than be overwritten: "
        "an audit trail in which a decision can be replaced is not one."
    )

def sweep(now: datetime | None = None) -> list[ReviewRecord]:
    """Escalate pending records older than the TTL. Intended to run on a schedule.

    Returns the records it escalated, so a caller can alert on the count. A
    queue that quietly grows is the failure this function exists to prevent.
    """
    raise NotImplementedError(
        "TASK 4: escalate pending records older than settings.review_ttl_hours. "
        "A queue with no timeout is a backlog nobody has noticed yet."
    )

def stats() -> dict[str, Any]:
    """Counts by state. The number a dashboard would alert on is `pending`."""
    records = load().values()
    out = {s: 0 for s in ("pending", "approved", "rejected", "escalated")}
    for r in records:
        out[r.state] += 1
    out["total"] = len(list(records))
    return out
