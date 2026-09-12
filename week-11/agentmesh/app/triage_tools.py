"""Tool registry - JSON schemas + Python impls + execute_tool dispatcher.

This is the same shape we have used since Week 1. No agent framework, no
orchestration library - just a dict of {name: (schema, impl)} and a 10-line
dispatcher. The TriageFlow specialist (Week 10 import) consumes this registry.

Not to be confused with `app/tools.py`, which holds the tool bodies this repo
*publishes* over MCP. This file is what the agent calls internally; that one is
what other people call. Opposite directions, hence the two names.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _lookup_runbook(query: str) -> dict[str, Any]:
    """Fake runbook lookup for the demo. In production this hits the runbook MCP server."""
    fixtures = {
        "payments": {
            "title": "Restart payments-api",
            "steps": ["kubectl rollout restart deploy/payments-api -n prod"],
            "owner": "payments-team",
        },
        "auth degraded": {
            "title": "Auth service degraded - initial triage",
            "steps": [
                "Check identity-provider health endpoint",
                "Verify JWKS endpoint reachability",
                "Page identity-on-call if both fail",
            ],
            "owner": "identity-team",
        },
    }
    for k, v in fixtures.items():
        if k in query.lower():
            return {"runbook": v, "total_available": 1}
    return {"runbook": None, "total_available": 0}


def _propose_action(remediation: str, target: str) -> dict[str, Any]:
    """Returns a proposed action that must pass an approval gate before running."""
    return {
        "proposal": {
            "remediation": remediation,
            "target": target,
            "mutating": True,
            "requires_approval": True,
        }
    }


TOOLS: dict[str, dict[str, Any]] = {
    "lookup_runbook": {
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text search over the runbook library",
                }
            },
            "required": ["query"],
        },
        "description": (
            "Search the engineering runbook library for entries matching the query. "
            "Use when the user asks how to do something operational or where a runbook lives. "
            "Returns the matching runbook with title, steps, and owner, or null if no match."
        ),
        "impl": _lookup_runbook,
    },
    "propose_action": {
        "schema": {
            "type": "object",
            "properties": {
                "remediation": {
                    "type": "string",
                    "description": "What action to take (verb-first)",
                },
                "target": {
                    "type": "string",
                    "description": "What system/service the action targets",
                },
            },
            "required": ["remediation", "target"],
        },
        "description": (
            "Propose a remediation action against a target. "
            "Use when the incident requires a mutation. The proposal is gated by human approval "
            "before any side-effect runs. Returns the proposal payload."
        ),
        "impl": _propose_action,
    },
}


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """The 10-line dispatcher. Two failure modes, both structured:
    an unknown tool name, and an implementation that raised. Neither escapes as a
    traceback - the caller gets a dict it can branch on."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"success": False, "error": "unknown_tool", "tool": name}
    impl: Callable[..., dict[str, Any]] = tool["impl"]
    try:
        return impl(**args)
    except Exception as e:  # noqa: BLE001 - a tool bug is a result, not a crash
        logger.warning("tool.internal_error tool=%s error=%s", name, e)
        return {"success": False, "error": "internal_error", "tool": name, "message": str(e)}
