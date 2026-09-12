"""The handoff assertion. Ten lines, and the cheapest reliability spend here."""
from __future__ import annotations

from typing import Any


def assert_handoff(state: dict[str, Any], required: list[str], node: str) -> None:
    """Raise if any required field is absent or None on entry to `node`.

    Presence, NOT truthiness. A field counts as written if the key is there and
    its value is not None. `if not state.get(f)` would also reject False, 0, "",
    [] and {} - all of which are values a node may legitimately have written.
    `pending_approval` is a bool on this contract and a low-risk proposal writes
    False, so a truthiness test would reject a correctly written handoff.

    Emptiness is a separate, PER-FIELD question. Assert it where it is a bug,
    with a message that says what is empty. See knowledge-style checks in
    specialist nodes.
    """
    missing = [f for f in required if f not in state or state[f] is None]
    if missing:
        raise ValueError(f"handoff to {node} missing fields: {', '.join(missing)}")
