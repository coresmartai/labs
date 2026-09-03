"""Pydantic schemas for structured I/O."""
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrieved chunk from the KnowledgeVault index."""

    chunk_id: str
    text: str
    source_url: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    # Provenance - carried through from the KnowledgeVault payload so a
    # citation can always say WHICH document and page it came from.
    # Essential once the index holds more than one document (DocuRAG).
    document_id: str | None = None
    page_number: int | None = None


class Citation(BaseModel):
    """A citation emitted by the model - must be validated post-hoc."""

    chunk_id: str
    supporting_quote: str


class AnswerResponse(BaseModel):
    """The structured payload the /answer endpoint returns."""

    answer: str
    citations: list[Citation]
    refused: bool = False
    retrieval_top_score: float | None = None
    retrieval_spread: float | None = None
    validation_passed: bool = True

    # Diagnostic fields - always populated, shown in UI on refusals
    refusal_source: str | None = None
    # "threshold_gate" - score/spread below threshold, model never called
    # "llm"            - model was called and returned the refusal string
    # "validation"     - model answered but citation validation failed
    retrieved_chunks: list[Chunk] | None = None  # chunks sent to the model

    # Preserved only on validation failures - the raw output before the
    # conservative refusal conversion.  Lets the UI show "what the model
    # actually generated" so students can see exactly what broke.
    raw_model_answer: str | None = None
    raw_model_citations: list[Citation] | None = None
    validation_detail: str | None = None  # one-line reason why validation failed


class AnswerRequest(BaseModel):
    """Inbound /answer request."""

    question: str = Field(min_length=3)


class GoldenRow(BaseModel):
    """One row of the generated golden dataset (scripts/build_golden_dataset.py)."""

    question: str
    expected_answer: str
    must_cite: list[str] = Field(default_factory=list)


class EvalRow(BaseModel):
    """Per-row result from the eval harness - shown in the UI breakdown.

    Compact preview fields (question, system_answer, cited_ids, pass/fail chips)
    are used for the collapsed chip row in the UI.  The full `response` field
    embeds the complete /answer payload so the expanded view can reuse the same
    four-step pipeline renderer as a single-query call - no duplicated code.
    """

    question: str
    expected: str        # "answer" or "refuse"
    actual: str          # "answered" or "refused"
    passed: bool         # True = row scored correctly
    grounded: bool
    false_answer: bool
    false_refusal: bool
    cit_recall_ok: bool
    # Citation validity: True = model generated an answer and every citation
    # survived the deterministic validator; False = model generated an answer
    # but validation rejected it (refusal_source == "validation"); None = the
    # model never generated (threshold gate or correct refusal) so citation
    # discipline was not exercised on this row.
    cit_valid_ok: bool | None = None
    # Compact preview fields (used for collapsed chip row)
    system_answer: str   # truncated at 250 chars
    cited_ids: list[str] # chunk IDs cited
    refusal_source: str | None = None
    # Full /answer response - drives the expanded pipeline view in the UI.
    # None only if the eval loop failed to build the response (should not happen).
    response: "AnswerResponse | None" = None


class EvalMetrics(BaseModel):
    """Eval summary returned by POST /eval.

    Five metrics cover the full correctness surface. Each rate divides only
    by the rows that exercised the behaviour it measures (denominator
    honesty - see README, Failure #4):
      groundedness        - of the rows that shipped an answer, the fraction
                            whose answer overlaps the expected one
      citation_validity   - of the rows where the model GENERATED an answer, the
                            fraction whose citations all survived the deterministic
                            validator. Not "precision against a must-cite list" -
                            the validator already guarantees no invalid citation
                            ever ships, so what this measures is the model's raw
                            citation discipline: how often its output needed the
                            validator to step in (refusal_source == "validation").
      citation_recall     - of the rows that SHOULD cite (expected answers),
                            the fraction that emitted at least one citation
      false_answer_rate   - fraction of all rows answered when refusal (or a
                            different answer) was correct
      false_refusal_rate  - fraction of all rows refused when an answer was expected

    All five are cheap deterministic proxies (token overlap against the expected
    answer + citation-presence via the "_answered" sentinel)

    `rows` contains the per-row breakdown; each EvalRow embeds the full AnswerResponse
    so the UI can render the expanded pipeline trace for any row without a second API call.
    """

    groundedness: float
    citation_validity: float
    citation_recall: float
    false_answer_rate: float
    false_refusal_rate: float
    rows_scored: int
    rows: list[EvalRow] = Field(default_factory=list)
