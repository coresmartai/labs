"""Handoff-contract enforcement for the shared TriageState.

Every node that depends on upstream writes calls assert_handoff at entry.
A missing field then fails loudly with the node name and the exact field,
instead of silently producing a wrong answer downstream.
"""
from __future__ import annotations


def assert_handoff(state: dict, required: list[str], node: str) -> None:
    """Raise if any required field is absent or None on entry to `node`.

    PRESENCE, NOT TRUTHINESS. A field counts as written if the key is there and
    its value is not None. The shorter `if not state.get(f)` also rejects False,
    0, "", [] and {} - all of which are values a node may legitimately have
    written. This contract contains one: `pending_approval` is a bool, and a
    proposal that needs no approval writes False. Under a truthiness test that
    correctly written False is indistinguishable from a field Action forgot, and
    the error blames the node that got it right.

    EMPTINESS IS A SEPARATE, PER-FIELD QUESTION. An empty `retrieved_docs` on a
    knowledge route really is a silent failure you want loud, but that is a claim
    about one field rather than about every field on the contract, so it lives in
    knowledge_node with its own message. See `assert_retrieved` below.
    """
    missing = [f for f in required if f not in state or state[f] is None]
    if missing:
        raise ValueError(f"handoff to {node} missing fields: {', '.join(missing)}")


def assert_retrieved(docs: list, node: str) -> None:
    """The per-field emptiness check, where the message can say what is empty."""
    if not docs:
        raise ValueError(f"{node}: retrieval returned no documents")
