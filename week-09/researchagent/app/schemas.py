"""The contract surface. Every shape that crosses a boundary lives here.

Read this file first. The output contract is what makes ResearchAgent a
different project from OpsAssist, and it is three types long.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Answerability = Literal["answerable", "partial", "unanswerable"]


# ---------- the corpus ----------

class Chunk(BaseModel):
    """One retrievable passage. `chunk_uid` is stable for the life of the index."""
    chunk_uid: str            # e.g. "rfc9989:0042" - document plus ordinal
    rfc: str                  # "rfc9989"
    section: str              # nearest preceding section heading, best effort
    page: int
    text: str


class SearchHit(BaseModel):
    """A chunk as the retriever returned it, for one query.

    `rank` is position IN THIS RESULT SET and means nothing outside it. That is
    the whole trap this project is about: rank 3 for one query and rank 3 for
    another are different passages.
    """
    rank: int
    score: float
    chunk: Chunk


# ---------- the ledger ----------

class LedgerEntry(BaseModel):
    """One piece of evidence, with an identity that outlives the query that found it."""
    ledger_id: str            # what a claim cites. Stable across the whole run
    chunk_uid: str
    rfc: str
    section: str
    page: int
    quote: str                # verbatim substring of the chunk
    question_index: int       # which of the three questions was being researched
    first_seen_iter: int      # which round of THAT question surfaced it
    queries: list[str] = Field(default_factory=list)   # every query that returned it


# ---------- the brief ----------

class Claim(BaseModel):
    """One sentence of the brief, and the evidence it rests on."""
    text: str
    ledger_ids: list[str] = Field(default_factory=list)


class BriefSection(BaseModel):
    question: str
    answerability: Answerability          # what the learner declared
    finding: Literal["answered", "partial", "declined"]   # what the run produced
    claims: list[Claim] = Field(default_factory=list)
    gap_note: str = ""        # required when finding is "partial" or "declined"


class Brief(BaseModel):
    sections: list[BriefSection]


# ---------- validation ----------

class ClaimVerdict(BaseModel):
    section: int
    claim: int
    ok: bool
    reason: str = ""


class SectionCounts(BaseModel):
    """Per-section totals, so the README table is a copy rather than a count."""
    section: int
    question: str
    answerability: Answerability
    finding: str
    claims: int
    cited: int
    valid: int


class ValidationReport(BaseModel):
    passed: bool
    claims_total: int
    claims_cited: int
    claims_valid: int
    per_section: list[SectionCounts] = Field(default_factory=list)
    verdicts: list[ClaimVerdict] = Field(default_factory=list)


# ---------- the run ----------

class TraceIteration(BaseModel):
    model_config = {"protected_namespaces": ()}

    iter: int
    question_index: int
    model_request_tokens: int
    model_response_tokens: int
    tool_calls: list[dict] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)
    new_ledger_entries: int
    elapsed_ms: int


class ResearchResponse(BaseModel):
    brief: Brief
    ledger: list[LedgerEntry]
    iter_count: int
    stop_reason: str
    trace: list[TraceIteration]
