"""The approval policy.

=============================================================================
 TASK 3. `risky()` is yours to write. It ships returning True for everything,
 which is a gate that fires on every proposal - and a gate that wakes a human
 for a keyboard order is a gate people learn to click through.
=============================================================================

Three constraints, and they are not style preferences:

  1. It must be a PURE FUNCTION of the proposal. Resuming re-runs the whole node
     from the top, so `risky()` is evaluated again after the pause. If it can
     return a different verdict on re-entry - because it read a clock, a counter,
     or anything outside its argument - you have a gate that can be walked around.

  2. It must not perform I/O, for the same reason. Everything above the pause
     executes twice.

  3. You must be able to defend it in DESIGN.md. Which proposals pause, which do
     not, and what the worst case is for each side of that line.

`tests/test_risk.py` pins constraint 1 and the two ends of the range. The
middle is yours, and the design memo is where you argue for it.
"""
from __future__ import annotations

from typing import Any


def risky(proposal: dict[str, Any]) -> bool:
    """Return True if this proposal must pause for a human.

    TASK 3. Replace this. The shipped version pauses for everything.

    `proposal` looks like: {"tool_name": "grant_access",
                            "arguments": {"system": "billing-db", "level": "write"}}
    """
    return True
