"""Action tools the Action agent can propose. Each has schema + impl.

The Action agent proposes a tool call but does NOT execute. Execution
happens after the human approval gate, inside the action_execute node.

Two things in this file exist because of the gate above it, and neither is
about the tools themselves:

  * an IDEMPOTENCY LEDGER, because resuming re-runs the executing node from the
    top. If the tool succeeds and the process dies before the state update is
    checkpointed, the replay calls it again. A key that is stable across the
    replay is what makes the second call recognisable as the same call.
  * a RESOURCE VERSION, because a human approval takes minutes and sometimes
    hours. Something the node can read before and after the pause is the
    cheapest way to notice that the world moved while the approver was deciding.

Both are in-process here, which is honest about what a demo can show and wrong
for production: the ledger belongs in the same transactional store as the
record of the call, and the version belongs to the real resource.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Callable

logger = logging.getLogger(__name__)

# resource key -> how many times it has changed. A real system reads this from
# the resource (an ETag, a row version, a deployment generation).
_VERSIONS: Counter[str] = Counter()

# idempotency key -> the result of the call that key already made.
_LEDGER: dict[str, dict[str, Any]] = {}


# --- Tool implementations (mocked for the live-code demo) ---

def restart_service(service_name: str) -> dict[str, Any]:
    logger.info("tool.restart_service service=%s", service_name)
    # Restarting the service changes it, so its version moves. This is what a
    # proposal approved before the restart will be measured against.
    _VERSIONS[f"service:{service_name}"] += 1
    return {"service": service_name, "status": "restarted", "duration_ms": 4200}


def page_team(team: str, severity: str) -> dict[str, Any]:
    logger.info("tool.page_team team=%s severity=%s", team, severity)
    _VERSIONS[f"rota:{team}"] += 1
    return {"team": team, "severity": severity, "page_id": "P-9482", "acknowledged": False}


def file_ticket(title: str, body: str, priority: str = "P3") -> dict[str, Any]:
    logger.info("tool.file_ticket title=%s priority=%s", title, priority)
    return {"ticket_id": "TF-1234", "title": title, "priority": priority, "url": "https://tickets/TF-1234"}


# --- Registry: schema + impl ---

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "restart_service": {
        "schema": {
            "name": "restart_service",
            "description": "Restart a service by name. Mutates infrastructure - requires approval.",
            "input_schema": {
                "type": "object",
                "properties": {"service_name": {"type": "string"}},
                "required": ["service_name"],
            },
        },
        "impl": restart_service,
    },
    "page_team": {
        "schema": {
            "name": "page_team",
            "description": "Page an on-call team. Notifies humans - requires approval.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "team": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                },
                "required": ["team", "severity"],
            },
        },
        "impl": page_team,
    },
    "file_ticket": {
        "schema": {
            "name": "file_ticket",
            "description": "File a ticket. Creates a tracked record - requires approval.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
                },
                "required": ["title", "body"],
            },
        },
        "impl": file_ticket,
    },
}


def resource_key(name: str, args: dict[str, Any]) -> str:
    """Which thing in the world this proposal is about."""
    if name == "restart_service":
        return f"service:{args.get('service_name', '')}"
    if name == "page_team":
        return f"rota:{args.get('team', '')}"
    if name == "file_ticket":
        return "tickets"
    return f"unknown:{name}"


def resource_version(name: str, args: dict[str, Any]) -> str:
    """A cheap, side-effect-free read of the resource the proposal targets.

    Called once when the plan is made and once more after the resume. If the two
    readings differ, someone else changed the thing while the approver was
    deciding, and the plan was made against a world that no longer exists.
    """
    return str(_VERSIONS[resource_key(name, args)])


def execute_tool(name: str, args: dict[str, Any],
                 *, idempotency_key: str | None = None) -> dict[str, Any]:
    """Dispatch a tool call by name - never raises, unknown tool returns error dict.

    The key is checked FIRST, before dispatch, because the whole point is that a
    replayed call must not reach the implementation at all. Pass a key derived
    from something stable across a replay - the thread id plus the proposal id.
    Never a freshly generated UUID: a new key every call is the same as no key.
    """
    if idempotency_key is not None and idempotency_key in _LEDGER:
        logger.warning("tool.replayed key=%s tool=%s - not executing again", idempotency_key, name)
        return {**_LEDGER[idempotency_key], "replayed": True}

    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return {"success": False, "error": "unknown_tool", "tool": name}
    impl: Callable[..., dict[str, Any]] = entry["impl"]
    try:
        result = impl(**args)
    except Exception as exc:
        # A failed call is NOT recorded. Recording it would make the retry of a
        # transient failure look like a duplicate and swallow it.
        return {"success": False, "error": str(exc), "tool": name}
    if idempotency_key is not None:
        _LEDGER[idempotency_key] = result
    return result


def tools_schema() -> list[dict[str, Any]]:
    return [t["schema"] for t in TOOL_REGISTRY.values()]


def reset_tool_state() -> None:
    """Test hook - both stores are process-global, so tests must clear them."""
    _VERSIONS.clear()
    _LEDGER.clear()
