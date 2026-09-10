"""Pydantic schemas, the contract layer.

Two shapes matter here:
  * Message / ChatExample - the *generic* conversation format the app
    writes and reads. Provider-agnostic.
  * TuningJobMeta / InferenceRequest / InferenceResponse - the API
    contracts for the FastAPI serving side.

The generic → Vertex conversion lives in app/convert.py; this file only
declares shapes.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


# ── Generic conversation format (what the app produces / consumes) ──

class Message(BaseModel):
    """A single conversational turn in the app's internal representation.

    'role' is deliberately named to match Google's `role` field so mapping
    to Vertex is a rename-free copy for the value.
    """
    role: Literal["user", "model"]
    message: str = Field(..., min_length=1)


class ChatExample(BaseModel):
    """One training example - a list of alternating user/model turns."""
    messages: list[Message] = Field(..., min_length=2)


# ── Vertex AI tuning-job metadata ─────────────────────────────────────

class TuningJobMeta(BaseModel):
    """What we log locally after starting a Vertex tuning job."""
    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    state: Literal[
        "JOB_STATE_UNSPECIFIED", "JOB_STATE_QUEUED", "JOB_STATE_PENDING",
        "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLING", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED",
        "JOB_STATE_EXPIRED", "JOB_STATE_UPDATING",
    ]
    tuned_model_display_name: str
    tuned_model_endpoint: str | None = None
    source_model: str
    train_dataset_uri: str
    epochs: int
    adapter_size: int
    learning_rate_multiplier: float
    error: str | None = None


# ── FastAPI request/response contracts ────────────────────────────────

class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    temperature: float = 0.2
    max_output_tokens: int = 512


class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    output: str
    model_id: str
    latency_ms: float


class ConvertRequest(BaseModel):
    """Body shape for POST /convert - run the generic → gcloud converter."""
    generic_path: str | None = None
    gcloud_path: str | None = None


class ConvertResponse(BaseModel):
    input_rows: int
    output_rows: int
    output_path: str
    warnings: list[str] = Field(default_factory=list)


class TuneStartRequest(BaseModel):
    """Body shape for POST /v1/tune/start."""
    train_dataset_uri: str | None = None  # gs:// URI; defaults to config
    epochs: int | None = None
    adapter_size: int | None = None
    learning_rate_multiplier: float | None = None


# ── EvalResult, same shape as the local package so the harness can merge results ──

class EvalResult(BaseModel):
    """Single row of the SpecialistTuner decision memo."""
    model_config = ConfigDict(protected_namespaces=())

    approach: Literal["base", "lora", "vertex_ft"]
    model_id: str
    accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float
    cost_per_1k_calls_usd: float
    trainable_params: int
    privacy_posture: Literal["in_perimeter", "vendor_zone"]
