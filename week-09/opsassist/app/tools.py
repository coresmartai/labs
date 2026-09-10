"""Tool registry + dispatcher.

Three tools, each defined as a {schema, impl} pair. A 10-line
`execute_tool` dispatcher does the routing. No agent framework, no
registry library - just a dict at module scope.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.schemas import (
    GetRunbookInput, GetRunbookOutput,
    QueryMetricsInput, QueryMetricsOutput,
    ProposeRemediationInput, ProposeRemediationOutput,
)

logger = logging.getLogger(__name__)

# retry canon, applied ONLY to the idempotent (read-only) tools.
# In production both hit the network (runbook store, Prometheus), so
# transient network errors are worth 3 bounded attempts with jittered
# backoff. propose_remediation is deliberately NOT wrapped: non-idempotent
# tools never blind-retry - the idempotency_key is what makes a retry safe.
_idempotent_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=8.0),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True,
)


# ---------- impls ----------

# Static runbook data. In production this would be a RAG read against
# a runbook store (KnowledgeVault).
_RUNBOOKS = {
    "auth-service-latency": {
        "procedure": [
            "Check recent deploys to auth-service",
            "Query latency metrics over 1h window",
            "Compare to baseline (p95 < 200ms)",
            "If credential rotation is overdue, run rotate-credentials",
            "If latency persists after 5 min, escalate to on-call",
        ],
        "severity_hint": "high",
    },
    "api-error-spike": {
        "procedure": [
            "Query error_rate metrics over 5m window",
            "Check dependency health (db, auth-service)",
            "If error_rate > 5%, propose rollback",
            "If error_rate < 5% but > baseline, monitor for 10m",
        ],
        "severity_hint": "high",
    },
    "worker-queue-backlog": {
        "procedure": [
            "Query queue depth over 1h",
            "Scale worker pool if backlog > 10k",
            "Investigate downstream blockers",
        ],
        "severity_hint": "medium",
    },
    "db-connection-exhausted": {
        "procedure": [
            "Query db connection metrics",
            "Identify the client exhausting the pool",
            "Recycle connections if leak suspected",
        ],
        "severity_hint": "critical",
    },
}


@_idempotent_retry
def get_runbook(args: GetRunbookInput) -> GetRunbookOutput:
    """Look up a runbook by known topic. Idempotent."""
    rb = _RUNBOOKS[args.topic]
    return GetRunbookOutput(
        topic=args.topic,
        procedure=rb["procedure"],
        severity_hint=rb["severity_hint"],
    )


# The four services we instrument: api, worker, db, auth. `auth-service` is
# the deployment's full name; `auth` is what everyone actually calls it, so
# the schema accepts both and the impl normalises to one.
_SERVICE_ALIASES = {"auth": "auth-service"}


@_idempotent_retry
def query_metrics(args: QueryMetricsInput) -> QueryMetricsOutput:
    """Query metrics for a service over a window. Idempotent.

    In production this hits Prometheus or similar.

    `auth` is an accepted alias for `auth-service`: the model may emit either,
    and both normalise to the one canonical service name before lookup.
    """
    service = _SERVICE_ALIASES.get(args.service, args.service)
    # Deterministic stub - real impl reads from Prometheus.
    base = {
        ("api", "5m"): (90, 320, 4.2, 1200),
        ("api", "1h"): (95, 280, 2.8, 1180),
        ("api", "24h"): (88, 210, 0.8, 1100),
        ("auth-service", "1h"): (140, 410, 0.4, 320),
        ("worker", "1h"): (250, 800, 0.2, 80),
        ("db", "5m"): (4, 22, 0.0, 4800),
    }
    p50, p95, err, qps = base.get(
        (service, args.window),
        (100, 300, 1.0, 500),
    )
    return QueryMetricsOutput(
        service=service,
        window=args.window,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        error_rate_pct=err,
        qps=qps,
    )


# In-memory idempotency cache. In production this is a Redis SET.
_REMEDIATION_CACHE: dict[str, ProposeRemediationOutput] = {}


def propose_remediation(args: ProposeRemediationInput) -> ProposeRemediationOutput:
    """Propose a remediation action. NON-idempotent in theory.

    We make it idempotent in practice with the idempotency_key:
    same key returns the prior result instead of re-firing.
    """
    if args.idempotency_key in _REMEDIATION_CACHE:
        prior = _REMEDIATION_CACHE[args.idempotency_key]
        # Return marker so caller knows this was a dedup hit.
        return ProposeRemediationOutput(
            incident_id=prior.incident_id,
            action=prior.action,
            posted_to_pagerduty=prior.posted_to_pagerduty,
            was_duplicate=True,
        )
    # Map issue -> canonical action.
    action_map = {
        "latency-spike": "rotate-credentials",
        "error-spike": "rollback-deploy",
        "queue-backlog": "scale-worker-pool",
        "db-exhaustion": "recycle-db-connections",
        "deploy-rollback-needed": "rollback-deploy",
    }
    incident_id = f"INC-{args.idempotency_key[:8]}"
    out = ProposeRemediationOutput(
        incident_id=incident_id,
        action=action_map[args.issue],
        posted_to_pagerduty=True,
        was_duplicate=False,
    )
    _REMEDIATION_CACHE[args.idempotency_key] = out
    return out


# ---------- registry + dispatcher ----------

TOOLS: dict[str, dict[str, Any]] = {
    "get_runbook": {
        "input_model": GetRunbookInput,
        "output_model": GetRunbookOutput,
        "impl": get_runbook,
        "description": (
            "Look up the canonical runbook for a known issue type. "
            "Use when you need the documented remediation procedure."
        ),
    },
    "query_metrics": {
        "input_model": QueryMetricsInput,
        "output_model": QueryMetricsOutput,
        "impl": query_metrics,
        "description": (
            "Fetch metrics for a service over a time window. "
            "Use after you have a specific service and window in mind. "
            "Do not query speculatively."
        ),
    },
    "propose_remediation": {
        "input_model": ProposeRemediationInput,
        "output_model": ProposeRemediationOutput,
        "impl": propose_remediation,
        "description": (
            "Propose a remediation action. "
            "Use only after you have evidence from at least one runbook and one metric query. "
            "Always supply an idempotency_key."
        ),
    },
}


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call. Always returns a dict - never raises."""
    if name not in TOOLS:
        logger.warning("Unknown tool: %s", name)
        return {"success": False, "error": "unknown_tool", "tool": name}
    spec = TOOLS[name]
    try:
        parsed = spec["input_model"](**args)
        result = spec["impl"](parsed)
        return result.model_dump()
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return {"success": False, "error": str(exc), "tool": name}


def tool_catalog_for_model() -> list[dict[str, Any]]:
    """Produce the tool catalog in the OpenAI function-calling shape.

    Each entry is {"type": "function", "function": {name, description, parameters}}.
    The parameters value is a JSON Schema object generated from the Pydantic input model.
    """
    catalog = []
    for name, spec in TOOLS.items():
        catalog.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["input_model"].model_json_schema(),
            },
        })
    return catalog
