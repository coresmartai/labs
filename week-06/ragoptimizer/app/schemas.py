"""Pydantic schemas - every cross-module boundary is typed."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One retrievable document chunk. The chunk_id is what citations point to."""

    chunk_id: str = Field(..., description="Stable ID - used in citation markers")
    text: str = Field(..., description="Chunk body; first line carries the marker")
    score: float = Field(default=0.0, description="Most recent relevance score (vector or rerank)")
    source: str = Field(default="", description="Originating document filename")

    @property
    def marker_line(self) -> str:
        """The first line carries the chunk-ID marker that citation validation depends on."""
        return self.text.split("\n", 1)[0]


class QueryRequest(BaseModel):
    """Inbound request from the FastAPI route."""

    query: str = Field(..., min_length=1)
    conversation_history: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    chunk_id: str
    quote: str = Field(default="", description="Verbatim snippet supporting the claim")


class ChunkTrace(BaseModel):
    """Lightweight snapshot of one chunk at a pipeline stage."""
    chunk_id: str
    preview: str          # body text, first 150 chars (marker line stripped)
    vector_score: float   # RRF / cosine score from the vector search stage
    rerank_score: float = 0.0   # cross-encoder score (0.0 before reranking)


class PipelineTrace(BaseModel):
    """Step-by-step record of one retrieve() call - attached to GroundedAnswer."""
    hyde_probe: str = ""                    # hypothetical answer text (empty if HyDE disabled)
    hyde_candidates_count: int = 0          # chunks returned by dense search on the probe
    original_candidates_count: int = 0      # chunks returned by hybrid search on raw query
    combined_count: int = 0                 # unique chunks after union+dedupe
    before_rerank: list[ChunkTrace] = []    # top-10 by vector/RRF score (before cross-encoder)
    after_rerank: list[ChunkTrace] = []     # top-10 after cross-encoder rerank, with both scores
    final_chunks: list[ChunkTrace] = []     # the narrow slice sent to the LLM, after compression

    # ── Threshold-gate inputs (Week 5's gate, applied to BOTH W6 channels) ────
    # Week 5 gates one channel: the dense cosines off the raw query. Week 6 has
    # two probes, so it gets two chances to clear the SAME rule at the SAME
    # thresholds:
    #     raw_ok  = top1_raw  >= similarity_threshold and spread_raw  >= spread_delta
    #     hyde_ok = top1_hyde >= similarity_threshold and spread_hyde >= spread_delta
    #     refuse unless (raw_ok or hyde_ok)
    # Surfaced on the trace so the UI can show WHY a row refused - which channel
    # cleared, which did not, and by how much. When HyDE is disabled or the probe
    # fails, the hyde pair stays (0.0, 0.0), which fails the gate rather than
    # granting a free pass.
    top1_raw: float = 0.0       # top-1 dense cosine, raw-query channel
    spread_raw: float = 0.0     # top1-top3 dense cosine spread, raw-query channel
    top1_hyde: float = 0.0      # top-1 dense cosine, HyDE-probe channel
    spread_hyde: float = 0.0    # top1-top3 dense cosine spread, HyDE-probe channel


class GroundedAnswer(BaseModel):
    """Response shape returned by the citation prompt."""

    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"] = "medium"
    fallback_triggered: bool = False
    pipeline_trace: PipelineTrace | None = None

    # Citation-discipline observability (W6 fix, see main.py::_judge_row).
    # generated=True means the model was actually called and returned an answer,
    # so its citation discipline was exercised and can be scored. Gate refusals
    # and model-call failures leave it False: nothing was generated, so there is
    # nothing to measure, and those rows leave the citation_validity denominator.
    generated: bool = False
    # How many citations the validator rejected before this response shipped.
    # >0 means the model cited a chunk it was never sent.
    citations_dropped: int = 0


class BenchmarkRow(BaseModel):
    """One row of the before/after comparison table."""

    metric: str
    baseline: float
    new_pipeline: float
    delta: float
    ci_low: float
    ci_high: float

    @property
    def is_significant(self) -> bool:
        """True when the 95% CI doesn't cross zero."""
        return (self.ci_low > 0 and self.ci_high > 0) or (self.ci_low < 0 and self.ci_high < 0)


# ── Eval schemas ──────────────────────────────────────────────────────────────

class GoldenRow(BaseModel):
    """One row from the golden evaluation dataset (golden_dataset.json)."""

    model_config = {"extra": "ignore"}   # silently drop _source_* metadata fields

    question: str
    expected_answer: str = ""
    must_cite: list[str] = []           # ["_answered"] → any citation ok; [] → should refuse


class EvalRow(BaseModel):
    """Scored result for one golden-dataset question."""

    question: str
    expected: Literal["answer", "refuse"]
    actual: Literal["answered", "refused"]
    passed: bool
    grounded: bool
    false_answer: bool
    false_refusal: bool
    cit_recall_ok: bool
    cit_valid_ok: bool | None = None    # None = model never generated
    system_answer: str                  # first 250 chars of the answer text
    cited_ids: list[str]               # chunk_ids cited by the model
    confidence: str = "low"
    fallback_triggered: bool = False
    pipeline_trace: PipelineTrace | None = None   # full trace for UI expansion


class EvalMetrics(BaseModel):
    """Aggregated metrics from a full /eval run over the golden dataset."""

    groundedness: float
    citation_validity: float
    citation_recall: float
    false_answer_rate: float
    false_refusal_rate: float
    rows_scored: int
    rows: list[EvalRow]


# ── Compare schemas ───────────────────────────────────────────────────────────

class PipelineMetrics(BaseModel):
    """Five-metric summary for one pipeline (baseline or full)."""

    groundedness: float
    citation_validity: float
    citation_recall: float
    false_answer_rate: float
    false_refusal_rate: float


class CompareRow(BaseModel):
    """Paired result for one golden question - baseline vs full W6 pipeline."""

    question: str
    expected: str                       # "answer" | "refuse"

    # Baseline (Week 5 hybrid: dense+BM25+RRF only)
    baseline_actual: str               # "answered" | "refused"
    baseline_passed: bool
    baseline_answer: str               # first 250 chars
    baseline_cited_ids: list[str]
    baseline_confidence: str = "low"
    baseline_fallback: bool = False

    # Full W6 pipeline (HyDE + hybrid + rerank + compress)
    full_actual: str                   # "answered" | "refused"
    full_passed: bool
    full_answer: str                   # first 250 chars
    full_cited_ids: list[str]
    full_confidence: str = "low"
    full_fallback: bool = False
    full_pipeline_trace: PipelineTrace | None = None   # rich trace for UI expansion


class CompareMetrics(BaseModel):
    """Before/after comparison result from /eval/compare."""

    rows_scored: int
    baseline: PipelineMetrics          # aggregated metrics for Week 5 style baseline
    full: PipelineMetrics              # aggregated metrics for full W6 pipeline
    rows: list[CompareRow]             # per-question paired results
