"""Pydantic models for every interface in the app."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolInfo(BaseModel):
    name: str
    description: str
    inputSchema: dict[str, Any]


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ToolCallResponse(BaseModel):
    success: bool
    trace_id: str
    result: dict[str, Any] | None = None
    error: ToolError | None = None


class ToolPickRequest(BaseModel):
    request: str


class ToolPickResponse(BaseModel):
    tool: str | None
    arguments: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int


class EvalResult(BaseModel):
    id: str
    request: str
    expected_tool: str
    picked_tool: str | None
    correct: bool


class EvalSummary(BaseModel):
    quality: str
    model: str
    total: int
    correct: int
    accuracy: float
    results: list[EvalResult]
