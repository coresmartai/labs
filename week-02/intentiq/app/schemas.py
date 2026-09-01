"""Pydantic models - the contracts every adapter speaks."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


# The 4 intents IntentIQ classifies. Keep the list small for the Week 2 baseline.
INTENT_LABELS: tuple[str, ...] = (
    "cancel_subscription",
    "request_refund",
    "payment_question",
    "unknown",
)

Provider = Literal["openai", "nano", "ollama"]


class Result(BaseModel):
    """Unified return type from every provider. App code only ever sees this."""
    provider: Provider
    label: str = Field(description="Predicted intent. Will be normalised to one of INTENT_LABELS.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-reported model confidence. Treat as a sorting key, not a probability.")
    latency_ms: float = Field(ge=0.0)
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] = Field(default_factory=dict, description="Original provider response for debugging.")


class GoldenExample(BaseModel):
    """One row of the golden dataset."""
    id: str
    input: str
    expected: str
    note: str | None = None


class BenchmarkRow(BaseModel):
    """One row of the per-provider results table."""
    example_id: str
    provider: Provider
    predicted: str
    expected: str
    correct: bool
    confidence: float
    latency_ms: float
    input_tokens: int
    output_tokens: int


class ClassifyRequest(BaseModel):
    text: str
    provider: Provider = "openai"


class BenchmarkSummary(BaseModel):
    """The headline numbers per provider - the four-column comparison table."""
    provider: Provider
    n: int
    accuracy: float
    p50_ms: float
    p95_ms: float
    cost_per_1k_usd: float
    cold_start_ms: float | None = None
