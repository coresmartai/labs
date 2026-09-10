"""Pydantic models for structured input/output across the training + serving stack."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class TrainingExample(BaseModel):
    """One example as it appears in the JSONL training file."""
    messages: list[ChatMessage] = Field(..., min_length=2)


class InferenceRequest(BaseModel):
    """Request to the FastAPI inference endpoint."""
    messages: list[ChatMessage]
    max_new_tokens: int = 256
    temperature: float = 0.2


class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    output: str
    model_id: str
    latency_ms: float
    finish_reason: Literal["stop", "length", "error"] = "stop"


class EvalResult(BaseModel):
    """Single row of the SpecialistTuner decision memo.

    One table, one row per approach:
      base      - prompting the base model, no training
      lora      - the local PEFT/LoRA adapter this module builds
      vertex_ft - the hosted supervised-tuning result from the Vertex track

    `vertex_ft` and `vendor_zone` are accepted here so that rows emitted by
    this package and by specialisttuner-vertex validate against one schema
    and go into one table. This package never produces those values itself.
    """
    model_config = ConfigDict(protected_namespaces=())

    approach: Literal["base", "lora", "vertex_ft"]
    model_id: str
    accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float
    cost_per_1k_calls_usd: float
    trainable_params: int
    privacy_posture: Literal["in_perimeter", "vendor_zone"]
