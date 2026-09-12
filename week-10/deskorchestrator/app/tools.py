"""The three tools a specialist can propose. All mocked.

Every one of them changes the world, which is why none of them is called before
the gate. `execute_tool` never raises: unknown names and implementation errors
both come back as error dicts.
"""
from __future__ import annotations

from typing import Any


def grant_access(system: str, level: str, days: int = 14) -> dict[str, Any]:
    return {"granted": True, "system": system, "level": level, "expires_in_days": days,
            "grant_id": f"GRA-{abs(hash((system, level))) % 9000 + 1000}"}


def install_software(package: str, device: str) -> dict[str, Any]:
    return {"queued": True, "package": package, "device": device, "job_id": "JOB-2291"}


def order_hardware(item: str, quantity: int = 1) -> dict[str, Any]:
    return {"ordered": True, "item": item, "quantity": quantity, "order_id": "ORD-7742"}


TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "grant_access": {
        "impl": grant_access,
        "input_schema": {"type": "object",
                         "properties": {"system": {"type": "string"},
                                        "level": {"type": "string", "enum": ["read", "write", "admin"]},
                                        "days": {"type": "integer"}},
                         "required": ["system", "level"]},
        "description": "Grant a named access level on a named system, for a bounded number of days.",
    },
    "install_software": {
        "impl": install_software,
        "input_schema": {"type": "object",
                         "properties": {"package": {"type": "string"}, "device": {"type": "string"}},
                         "required": ["package", "device"]},
        "description": "Queue an install of a named package onto a named device.",
    },
    "order_hardware": {
        "impl": order_hardware,
        "input_schema": {"type": "object",
                         "properties": {"item": {"type": "string"}, "quantity": {"type": "integer"}},
                         "required": ["item"]},
        "description": "Order a stock hardware item from the depot.",
    },
}


def tools_schema() -> str:
    return "\n".join(
        f"- {name}: {spec['description']} arguments: "
        f"{list(spec['input_schema']['properties'])}"
        for name, spec in TOOL_REGISTRY.items())


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return {"success": False, "error": f"unknown tool: {name}", "tool": name}
    try:
        return {"success": True, "tool": name, "result": spec["impl"](**(args or {}))}
    except Exception as exc:                     # never raises out of the dispatcher
        return {"success": False, "error": str(exc), "tool": name}
