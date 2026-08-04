"""
Pydantic schemas for structured LLM output.

Why structured output? Because free text is fine for chat, but a *product*
needs to consume model responses programmatically. Defining the shape up
front turns the LLM from a chatbot into a typed function.

The model receives this shape (as JSON Schema) via the `tools=` parameter
of the API call - not via the system prompt. The tool-call arguments come
back conforming to it, and we build the Pydantic class from them -
Pydantic validates types, required fields, lengths, counts, and enums.
"""

from typing import Literal
from pydantic import BaseModel, Field


RiskLevel = Literal["low", "med", "high"]


class ReleaseSummary(BaseModel):
    """A summary of release notes that ReleaseBot can email."""

    headline: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="One-line release headline (max 120 chars).",
    )
    bullets: list[str] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="2-6 short bullet points covering the most important changes.",
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Operational risk: 'low' = safe to ship, 'med' = monitor, 'high' = needs careful rollout.",
    )


class SummaryRequest(BaseModel):
    """Incoming request to /summarize."""

    release_notes: str = Field(..., min_length=1, description="Raw release notes to summarize.")
    recipient: str | None = Field(None, description="Recipient email address. Required for /summarize, ignored by /summarize-stream.")
