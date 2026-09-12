"""The approval policy: which proposals wake a human.

`risky()` must be a PURE FUNCTION of the proposal, and that is a correctness
requirement rather than a style preference. Resuming re-runs the whole node from
the top, so this is evaluated again after the pause. A verdict that can change on
re-entry is a gate that can be walked around: pause, decline, resume, and the
policy now says no approval was needed.

So: no clock, no counter, no store, no module-level mutable. Read the argument.
"""
from __future__ import annotations

from typing import Any

# Tools whose blast radius reaches production. Restarting a service is
# survivable and paging a rota at 3 a.m. is not the kind of thing to do without
# a person having looked at it.
_ALWAYS_PAUSE = {"page_team", "restart_service"}


def risky(proposal: dict[str, Any] | None) -> bool:
    """True if this proposal must pause for a human."""
    if not proposal:
        return False
    name = proposal.get("tool_name")
    if name in (None, "none"):
        return False            # nothing there to approve
    if name in _ALWAYS_PAUSE:
        return True
    if name == "file_ticket":
        # Filing a ticket is a note to somebody, not a change to anything.
        return False
    return True                 # anything unrecognised pauses
