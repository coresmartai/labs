"""The four memory layers, and ONE way in.

Every write goes through `memory_write`, which asserts the scope before it does
anything else. Nothing writes to a layer behind its back: not the endpoints, not
the graph, not the loader that seeds the corpus at startup.

  in-context  the prompt for this turn. Not stored here; it IS the prompt.
  external    global IT policies.            scope="global",      collection="policies"
  episodic    this user's prior tickets.     scope="user:{id}",   collection="tickets"
  procedural  this user's standing rules.    a dict, standing in for one SQL row
"""
from __future__ import annotations

import logging
from typing import Any

from app.corpus import POLICIES, TICKETS
from app.vectorstore import get_store

log = logging.getLogger(__name__)

GLOBAL_SCOPE = "global"
POLICIES_COLL = "policies"
TICKETS_COLL = "tickets"

_PREFS: dict[str, dict[str, Any]] = {}
_KINDS = {"external", "episodic", "procedural"}


def user_scope(user_id: str) -> str:
    return f"user:{user_id}"


def memory_write(scope: str, kind: str, payload: dict[str, Any]) -> None:
    """The single write path. scope is 'global' or 'user:{user_id}'."""
    assert scope, "scope filter required"
    if kind not in _KINDS:
        raise ValueError(f"unknown memory kind: {kind!r}")

    if kind == "external":
        if scope != GLOBAL_SCOPE:
            raise ValueError("external memory is global knowledge; it is never user-scoped")
        get_store().add(scope, POLICIES_COLL, payload["chunks"])
    elif kind == "episodic":
        if scope == GLOBAL_SCOPE:
            raise ValueError("episodic memory is personalisation; it is never global")
        get_store().add(scope, TICKETS_COLL, payload["chunks"])
    elif kind == "procedural":
        if scope == GLOBAL_SCOPE:
            raise ValueError("procedural memory is personalisation; it is never global")
        uid = scope.split(":", 1)[1]
        _PREFS.setdefault(uid, {})[payload["key"]] = payload["value"]


# ---------------------------------------------------------------- reads
def search_policies(query: str, k: int) -> list[dict[str, Any]]:
    """External memory. Global knowledge, retrievable by anyone."""
    return get_store().search(query, scope=GLOBAL_SCOPE, collection=POLICIES_COLL, k=k)


def search_tickets(query: str, user_id: str, k: int) -> list[dict[str, Any]]:
    """Episodic memory, scoped to one user. NEVER skip the scope filter."""
    assert user_id, "scope filter required"
    return get_store().search(query, scope=user_scope(user_id), collection=TICKETS_COLL, k=k)


def get_prefs(user_id: str) -> dict[str, Any]:
    return dict(_PREFS.get(user_id, {}))


# ---------------------------------------------------------------- deletion
def delete_user(user_id: str) -> dict[str, int]:
    """Right to be forgotten. One trigger, one counter per destination we have.

    `policy_rows_deleted` is structurally zero and that is the point: the policy
    corpus is scope="global", so a user-scoped delete provably cannot reach it.
    A counter is evidence about what WAS deleted and never about what was left,
    so the over-broad case needs a test rather than a counter. See tests.
    """
    store = get_store()
    return {
        "user_id": user_id,
        "policy_rows_deleted": store.delete_scope(user_scope(user_id), POLICIES_COLL),
        "ticket_rows_deleted": store.delete_scope(user_scope(user_id), TICKETS_COLL),
        "procedural_rows_deleted": 1 if _PREFS.pop(user_id, None) is not None else 0,
    }


# ---------------------------------------------------------------- seeds
def seed_memory() -> None:
    """Load the canned corpus. Even the loader goes through `memory_write`."""
    memory_write(GLOBAL_SCOPE, "external", {"chunks": POLICIES})
    for uid, chunks in TICKETS.items():
        memory_write(user_scope(uid), "episodic", {"chunks": chunks})
    log.info("memory.seed policies=%d users=%d", len(POLICIES), len(TICKETS))


def seed_demo_prefs() -> None:
    """The procedural layer has a real writer. Without one it does not exist."""
    for key, value in [("preferred_response_length", "concise"),
                       ("always_cc_manager_on_access", True)]:
        memory_write(user_scope("u_1"), "procedural", {"key": key, "value": value})
    memory_write(user_scope("u_2"), "procedural", {"key": "preferred_response_length", "value": "detailed"})
    log.info("memory.seed_demo_prefs users=2")
