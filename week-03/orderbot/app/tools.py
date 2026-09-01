"""Tool registry, JSON-Schema, Python implementations, and 10-line dispatcher.

Pattern: TOOLS is a dict[name -> {schema, impl, description}].
The dispatcher validates args via Pydantic and calls the impl. No framework.
"""
from typing import Any, Callable
import logging
import time

from pydantic import ValidationError
from datetime import datetime
from app.schemas import OrderLookupArgs, OrderStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def lookup_order_status(order_id: str) -> OrderStatus:
    """Fake but realistic order lookup.

    Pretends to hit an orders database, sleeps 50ms to simulate real latency,
    returns a typed OrderStatus. In production this becomes an HTTP call to
    your orders service.
    """
    # Simulate I/O latency
    time.sleep(0.05)

    # Fake DB lookup; real impl would call orders-service.internal/lookup/{id}
    if order_id == "ORD-9999":
        # We use this ID in tests/test_endpoint.py to exercise the not-found
        # branch - the model should see a structured error envelope, never
        # a stack trace.
        raise OrderNotFoundError(order_id)

    return OrderStatus(
        order_id=order_id,
        state="shipped",
        eta="Friday by 8pm local",
        last_update=str(datetime.now()),
    )


class OrderNotFoundError(Exception):
    """Raised when an order ID does not exist in the orders service."""

    def __init__(self, order_id: str):
        self.order_id = order_id
        super().__init__(f"Order not found: {order_id}")


# ---------------------------------------------------------------------------
# Tool registry - name → {schema, impl, description}
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict[str, Any]] = {
    "lookup_order_status": {
        "name": "lookup_order_status",
        "description": (
            "Call this when the user asks about an order's shipping status, "
            "ETA, or delivery state. Do NOT call for billing questions or "
            "for refunds - those have separate tools (not yet wired)."
        ),
        "parameters": OrderLookupArgs.model_json_schema(),
        "impl": lookup_order_status,
        "args_model": OrderLookupArgs,
    },
}


def tool_definitions() -> list[dict[str, Any]]:
    """Return the list of tool definitions in OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in TOOLS.values()
    ]


# ---------------------------------------------------------------------------
# The 10-line dispatcher
# ---------------------------------------------------------------------------

def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate args, call impl, return a typed result.

    Returns a structured envelope so the model can incorporate failures
    into its final answer without hallucinating around a stack trace.
    """
    if name not in TOOLS:
        # The model picked a tool that is not in the registry. This is the
        # wrong_tool_choice failure mode: alert on it, because a rising rate
        # usually means the tool descriptions have drifted apart and the
        # model can no longer tell them apart.
        logger.error(
            "model called an unregistered tool: %s",
            name,
            extra={"event": "wrong_tool_choice",
                   "tool": name,
                   "known_tools": list(TOOLS)},
        )
        return {"success": False, "error": "unknown_tool", "tool": name}
    spec = TOOLS[name]
    try:
        validated = spec["args_model"].model_validate(args)
    except ValidationError as e:
        # The model called the right tool with the wrong arg shape.
        # Surface this so the model-level retry-with-correction can fix it.
        raise

    try:
        result = spec["impl"](**validated.model_dump())
        return {"success": True, "data": result.model_dump()}
    except OrderNotFoundError as e:
        return {"success": False, "error": "order_not_found",
                "hint": f"Verify the order ID format. {e.order_id} was not found."}
