"""Pydantic models for structured I/O.

Four groups of types:
  - Request/response (the FastAPI surface).
  - Internal results (what scrubbers and the injection classifier return).
  - Audit events (the shared shape every log line follows).
  - Store/ingestion types (what goes into and comes out of Qdrant).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---- request / response ----

class ReviewDecision(BaseModel):
    """A reviewer's verdict on one queued detection.

    Like AskRequest, this carries no identity claim the caller controls beyond
    the reviewer name, which is recorded rather than trusted: the route is
    gated by role, and the role comes from the authenticated identity.
    """

    decision: Literal["approve", "reject"]
    reviewer: str
    note: str = ""


class AskRequest(BaseModel):
    """Roles are NOT part of the body: access decisions come from the
    authenticated identity on request.state (headers here, JWT in production).
    A client-supplied roles field would let any caller escalate - exactly the
    attack the ACL ring exists to stop."""
    query: str
    user_id: str


class Citation(BaseModel):
    """One citation card. Emitted as its own SSE frame BEFORE the tokens.

    `quote` is a fourth egress path. The sentence-buffer output filter only
    ever sees the model's token stream - it never sees this field, so a citation
    quote is only as clean as the store it renders.
    """
    marker: str
    chunk_id: str
    source: str
    quote: str
    score: float


class IngestTextRequest(BaseModel):
    """Body for POST /ingest/text."""
    text: str
    source: str = "pasted-text"
    visible_to: list[str] = Field(default_factory=lambda: ["analyst"])


# ---- internal scrub/classifier results ----

class ScrubDetection(BaseModel):
    recognizer: str
    confidence: float
    span_start: int
    span_end: int


class ScrubResult(BaseModel):
    """What `scrub_input` / `scrub_output` / `scrub_chunk` return.

    `cleaned_text` is safe to pass on. `detections` is the list of what was
    found, used by the audit writer. Raw values are never carried.
    """
    cleaned_text: str
    detections: list[ScrubDetection]
    surface: Literal["input", "output", "retrieval", "telemetry"] = "input"


class InjectionVerdict(BaseModel):
    flagged: bool
    category: str | None = None
    confidence: float = 0.0
    matched_pattern: str | None = None


# ---- audit events ----

class AuditEvent(BaseModel):
    """One JSONL line. Shape is shared across event types.

    `detail` is intentionally typed as `dict[str, object]` so each event_type
    can carry its own structured detail without subclassing. The eval set
    checks that no field of `detail` ever contains a raw PII value.
    """
    timestamp: str
    request_id: str
    user_id_hash: str
    event_type: Literal["pii_scrub", "injection_refusal", "rbac_denial", "acl_filter",
                        "review_queued", "review_decided"]
    detail: dict[str, object]
    action_taken: str


# ---- store / ingestion ----

class SeedChunk(BaseModel):
    """A chunk as authored, BEFORE Presidio has seen it."""
    chunk_id: str
    text: str
    source: str
    visible_to: list[str]


class StoredChunk(BaseModel):
    """A chunk as it is written to Qdrant - i.e. AFTER the ingestion scrub.

    `scrubbed` records whether the Presidio pass actually ran. It is the
    honest field: an index built with CHUNK_SCRUB_ENABLED=false says so.
    """
    chunk_id: str
    text: str
    source: str
    visible_to: list[str]
    scrubbed: bool


class RetrievedChunk(BaseModel):
    """A chunk as it comes back from a search, with its similarity score."""
    chunk_id: str
    text: str
    source: str
    score: float
    visible_to: list[str]


class IngestReport(BaseModel):
    chunks: int
    spans_redacted: int
    chunk_scrub: bool
    collection: str


# ---- eval harness ----

class EvalCase(BaseModel):
    """One golden row. `expected` is the behaviour the pipeline should produce.

    - "answer"             : a grounded answer with at least one citation.
    - "refuse"             : off-topic; the relevance floor returns no context.
    - "injection_refusal"  : the classifier flags the query and the route refuses.
    - "acl_block"          : the only relevant chunk is hidden from these roles,
                             so retrieval returns nothing AND the ACL logs a
                             withheld chunk.
    `must_not_leak` lists raw strings (real PII) that must never appear in the
    answer or any citation quote - the scrubber's job, checked on every row.
    """
    question: str
    roles: list[str] = Field(default_factory=lambda: ["analyst"])
    expected: Literal["answer", "refuse", "injection_refusal", "acl_block"]
    must_not_leak: list[str] = Field(default_factory=list)
    note: str = ""


class EvalCaseResult(BaseModel):
    """How one case actually behaved, and whether it matched `expected`."""
    question: str
    roles: list[str]
    expected: str
    actual: str
    passed: bool
    cited_ids: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    leaked: bool = False              # a must_not_leak string appeared in output
    answer_preview: str = ""          # first 160 chars, for the UI/notebook


class EvalReport(BaseModel):
    """Summary returned by POST /eval.

    `accuracy` is the headline: fraction of cases whose actual outcome matched
    the expected one. `by_expected` breaks that down per behaviour so a single
    weak category (e.g. injection) is visible. `pii_leaks` MUST be 0 - a single
    leak is a failed build, not a lower score.
    """
    total: int
    correct: int
    accuracy: float
    pii_leaks: int
    by_expected: dict[str, dict[str, int]]   # expected -> {"total": n, "correct": m}
    rows: list[EvalCaseResult] = Field(default_factory=list)
