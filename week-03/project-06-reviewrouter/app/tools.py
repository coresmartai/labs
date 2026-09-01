"""The tool the model may call, and the fixture behind it.

Exactly one function goes into the tool list you send the model:
lookup_order. route_to_team does NOT, and that omission is worth three
marks. See app/routing.py.
"""
from typing import Any, Optional

# A tiny fixture standing in for an orders database.
# ORD-4471 and 4471 are the same order; the model will not normalise for you.
_ORDERS: dict[str, dict[str, Any]] = {
    "4471": {"order_id": "4471", "status": "delivered", "total": "42.00",
             "placed": "2026-08-02"},
    "5120": {"order_id": "5120", "status": "in_transit", "total": "18.50",
             "placed": "2026-08-14"},
    "6033": {"order_id": "6033", "status": "refund_pending", "total": "99.99",
             "placed": "2026-07-28"},
    "7781": {"order_id": "7781", "status": "delivered", "total": "7.25",
             "placed": "2026-06-11"},
}

# This id fails on its first call in a process and succeeds afterwards, so the
# transient-failure path is reachable without waiting for a real outage.
_FLAKY_ID = "5120"
_flaky_seen: set[str] = set()


class ToolLoopExceeded(RuntimeError):
    """Raised when the model asks for more lookups than the cap allows.

    Give this its own log tag and its own status code. A runaway loop that
    surfaces as a timeout is a runaway loop nobody will ever diagnose.
    """


def lookup_order(order_id: str) -> dict[str, Any]:
    """Look one order up. Returns an envelope, never raises to the model.

    A failure is DATA, not an exception. If you let this raise into the
    dispatcher, the model receives nothing and confidently invents an answer,
    which is the silent-partial-failure mode from the reading.
    """
    key = (order_id or "").strip().upper().removeprefix("ORD-").lstrip("#")

    if key == _FLAKY_ID and key not in _flaky_seen:
        _flaky_seen.add(key)
        return {"ok": False, "order": None, "error": "upstream_timeout"}

    order = _ORDERS.get(key)
    if order is None:
        return {"ok": False, "order": None, "error": "not_found"}
    return {"ok": True, "order": order, "error": None}


# ---------------------------------------------------------------------------
# TODO: the tool DEFINITION. This is graded.
#
# Three fields, and all three are prompt content the model re-reads on every
# call:
#   name        - self-describing. lookup_order, not tool_a.
#   description - MUST say when NOT to call it. Reviews that mention no order
#                 must not trigger a lookup. If they do, `attempts` comes back
#                 greater than zero on a no-order review and you lose the mark.
#                 An anti-example in the description is the cheapest fix.
#   parameters  - OrderLookupArgs.model_json_schema()
# ---------------------------------------------------------------------------
TOOL_DEFINITION: Optional[dict[str, Any]] = None


def tool_list() -> list[dict[str, Any]]:
    """The tools the model can see. route_to_team is deliberately absent."""
    if TOOL_DEFINITION is None:
        raise NotImplementedError("Define TOOL_DEFINITION in app/tools.py")
    return [TOOL_DEFINITION]
