"""Pydantic models for every interface in the harness.

If a value is exchanged between two layers, it has a model here. Eval data is
the kind of thing you regret being sloppy about three months in, when someone
asks "what does column thirteen mean?" and nobody remembers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Strategy(str, Enum):
    """The five adversarial mutation strategies."""

    TYPO = "typo"
    JAILBREAK = "jailbreak"
    MULTI_HOP = "multi_hop"
    CONFLICT = "conflict"
    HOSTILE = "hostile"
    SEED = "seed"  # the unmutated baseline case


class ExpectedBehaviour(str, Enum):
    """What the system *should* do for an adversarial prompt."""

    MATCH = "match"        # answer as if the original question
    REFUSE = "refuse"      # decline politely
    ABSTAIN = "abstain"    # "I don't have that information in the provided sources."


class SeedCase(BaseModel):
    """A hand-curated golden-set entry."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    expected_answer: str
    canary: str | None = Field(
        default=None,
        description="Optional nonsense string for leakage detection; a "
        "completion is a strong contamination signal worth investigating - "
        "not an automatic verdict that the eval set has been seen.",
    )
    expected_behaviour: ExpectedBehaviour = Field(
        default=ExpectedBehaviour.MATCH,
        description="What a correct system should do with THIS question, before "
        "any adversarial mutation. In-domain questions are MATCH. A golden set "
        "should also carry out-of-domain probes whose correct answer is 'I don't "
        "have that in the sources' - those are ABSTAIN. The generator propagates "
        "this expectation to the answer-the-question strategies (SEED, TYPO, "
        "MULTI_HOP, HOSTILE), so a system that correctly declines an "
        "out-of-domain probe is not punished for it.",
    )
    provenance: str = "hand-curated by reviewer"
    reviewer: str = "unknown"


class AdversarialCase(BaseModel):
    """A mutated red-team case + the expectation against which it is scored."""

    model_config = ConfigDict(frozen=True)

    id: str
    seed_id: str
    strategy: Strategy
    prompt: str
    expected_behaviour: ExpectedBehaviour
    # Optional contradicting context to inject for the CONFLICT strategy.
    injected_context: str | None = None


class SUTResponse(BaseModel):
    """What the system under test returned for one prompt."""

    answer: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    refused: bool = False
    abstained: bool = False
    latency_ms: int = 0

    @property
    def declined(self) -> bool:
        """The system did not fabricate an answer.

        `refused` (the retrieval gate fired before generation) and `abstained`
        (the generator saw contradictory or insufficient context and said so)
        are two implementation routes to the same externally-visible behaviour:
        the system declined to answer. Scoring treats them as equivalent, which
        is why the rubric tells both judges that "refusing or abstaining" is the
        correct outcome wherever a decline is expected.
        """
        return self.refused or self.abstained


class JudgeVerdict(BaseModel):
    """One judge's verdict on one (case, answer) pair."""

    judge_name: str
    score: float = Field(ge=0.0, le=5.0)
    justification: str
    refused_correctly: bool | None = None  # for binary refusal cases


class CaseResult(BaseModel):
    """All scoring outputs for a single adversarial case."""

    case: AdversarialCase
    response: SUTResponse
    judge_verdicts: list[JudgeVerdict] = Field(default_factory=list)
    ragas_scores: dict[str, float] | None = Field(
        default=None,
        description="Optional retrieval-quality rail (faithfulness, answer "
        "relevancy, context precision). We do not assert on it and the shipped "
        "harness does not compute it - the field exists so bolting RAGAS on is "
        "a pip install plus one extra scoring pass, not a schema migration.",
    )

    @property
    def judge_scores(self) -> list[float]:
        return [v.score for v in self.judge_verdicts]

    @property
    def judge_delta(self) -> float | None:
        """|Δ| between the two judges. None when fewer than two judges scored.

        More than a point apart on a 0-5 rubric is the disagreement signal the
        two-tier architecture exists to surface; `build_scorecard` counts those
        cases into `judge_disagreement_rate` and the UI flags them HUMAN REVIEW.
        """
        scores = self.judge_scores
        if len(scores) < 2:
            return None
        return abs(max(scores) - min(scores))


class ScorecardRow(BaseModel):
    """One row of the final PR-comment scorecard."""

    metric: str
    current: float
    baseline: float | None = None
    delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    verdict: str  # GREEN / AMBER / RED / INFO


class RunManifest(BaseModel):
    """The five fields that make a scorecard reproducible six months later.

    Emitted at the top of every rendered scorecard. Without them, "was this run
    before or after we changed the rubric?" is archaeology; with them it is a
    five-second diff.
    """

    model_config = ConfigDict(protected_namespaces=())

    run_id: str                 # UUID minted at the start of the run
    dataset_hash: str           # SHA-256 (first 12) of the seed set content
    rubric_hash: str            # SHA-256 (first 12) of the judge _RUBRIC string
    bootstrap_seed: int         # the fixed seed the paired bootstrap resamples on
    model_versions: dict[str, str]   # pinned ids: judge_primary/secondary + SUT


class Scorecard(BaseModel):
    """The whole scorecard the harness emits on every run."""

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    suite: str = "breakrag-v1"
    n_cases: int
    leakage_check_passed: bool
    # "passed" only when all three checks ran clean; "failed" when any check
    # found something; "skipped" when at least one check had no data to work
    # on. A skipped check is never reported as a pass.
    leakage_check_status: str = "skipped"
    leakage_check_notes: str
    judge_agreement_alpha: float
    rows: list[ScorecardRow]
    overall_verdict: str  # GREEN / AMBER / RED
    manifest: RunManifest | None = None
