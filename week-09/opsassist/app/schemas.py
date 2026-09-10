"""Pydantic models for tool inputs and outputs.

The model never sees raw dicts; it sees JSON Schema generated from these
classes. Restrictive types (Literals) prevent the model from emitting
arbitrary values that the impl can't handle.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# ---------- get_runbook ----------

class GetRunbookInput(BaseModel):
    """Look up the canonical runbook for a known issue type."""

    topic: Literal[
        "auth-service-latency",
        "api-error-spike",
        "worker-queue-backlog",
        "db-connection-exhausted",
    ] = Field(..., description="The runbook topic to fetch. Must be one of the known topics.")


class GetRunbookOutput(BaseModel):
    topic: str
    procedure: list[str]
    severity_hint: Literal["low", "medium", "high", "critical"]


# ---------- query_metrics ----------

class QueryMetricsInput(BaseModel):
    """Fetch metrics for a service over a time window.

    Do not use this for log search or arbitrary SQL. The arguments
    are constrained to the services and windows we actually instrument.
    """

    service: Literal["api", "worker", "db", "auth", "auth-service"] = Field(
        ...,
        description="The service to query metrics for: api, worker, db, or auth. "
        "`auth` and `auth-service` are the same service.",
    )
    window: Literal["5m", "1h", "24h"] = Field(
        ..., description="Time window to aggregate over."
    )


class QueryMetricsOutput(BaseModel):
    service: str
    window: str
    p50_latency_ms: float
    p95_latency_ms: float
    error_rate_pct: float
    qps: float


# ---------- propose_remediation ----------

class ProposeRemediationInput(BaseModel):
    """Propose a remediation action for an identified issue.

    Caller must supply an `idempotency_key` so retries don't double-fire
    the side effect.
    """

    issue: Literal[
        "latency-spike",
        "error-spike",
        "queue-backlog",
        "db-exhaustion",
        "deploy-rollback-needed",
    ] = Field(..., description="The diagnosed issue this remediation addresses.")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Severity using runbook vocabulary."
    )
    idempotency_key: str = Field(
        ..., description="Unique identifier for this remediation request. "
        "Same key + same args returns the prior result instead of re-firing."
    )


class ProposeRemediationOutput(BaseModel):
    incident_id: str
    action: str
    posted_to_pagerduty: bool
    was_duplicate: bool


# ---------- request model ----------

class AgentRunRequest(BaseModel):
    user_input: str
    task_id: str | None = None
    user_id: str = "anonymous"


# ---------- agent-level shapes ----------

class TraceIteration(BaseModel):
    model_config = {"protected_namespaces": ()}

    iter: int
    model_request_tokens: int
    model_response_tokens: int
    tool_calls: list[dict]
    tool_results: list[dict]
    elapsed_ms: int


class AgentResponse(BaseModel):
    final_response: str
    iter_count: int
    stop_reason: Literal[
        "end_turn",
        "max_iters",
        "token_budget",
        "cost_budget",
        "fatal_tool_error",
        "consent_revoked",
        "timeout",
    ]
    trace: list[TraceIteration]
