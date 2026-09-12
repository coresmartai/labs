"""The four memory layers.

* session    - Redis: per-thread list (TTL 1h) + user-level list (TTL 24h). The one real backend.
* external   - vector store, scope="global", collection="runbooks"
* episodic   - vector store, scope=f"user:{uid}", collection="episodic"
* procedural - in-process dict standing in for a Postgres JSONB row per user

Session memory is the one real backend. External, episodic and procedural ship
as production-shaped stubs: the call sites, the scope discipline and the
retrieval are real; the corpus and the procedural row are canned.

EVERY write goes through `memory_write(scope, kind, payload)`. One choke point,
one place to assert the scope, one place to audit. Every read passes the same
scope back as a metadata filter.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import redis

from .config import get_settings
from .vectorstore import GLOBAL_SCOPE, get_store, user_scope

log = logging.getLogger(__name__)

RUNBOOKS = "runbooks"
EPISODIC = "episodic"


# ---------------------------- the scoped write path ----------------------------

def memory_write(scope: str, kind: str, payload: dict[str, Any], **kw: Any) -> None:
    """The single write path into memory. Every writer goes through here.

    `scope` is "global" or f"user:{user_id}" - and it is asserted, not assumed.
    `kind` is one of the four memory layers: session, external, episodic,
    procedural. The helper dispatches to the layer's own writer; the layers
    never get written to behind its back.
    """
    assert scope, "scope filter required"
    if kind not in {"session", "external", "episodic", "procedural"}:
        raise ValueError(f"unknown memory kind: {kind!r}")

    if kind == "external":
        assert scope == GLOBAL_SCOPE, "external memory is global-scoped"
        get_store().add(scope, RUNBOOKS, payload["chunks"])
        return

    # The remaining three layers are all user-scoped.
    user_id = _user_of(scope)
    if kind == "session":
        session_append(user_id, kw["thread_id"], payload)
        # User-level memory: the same turn, tagged with its thread, on one rolling
        # per-user list that survives across sessions (the session list is per-thread).
        user_mem_append(user_id, {**payload, "thread_id": kw["thread_id"]})
    elif kind == "episodic":
        # Two shapes reach this layer and both belong to it: one summary keyed on
        # a thread id, and a batch of pre-written chunks from the seed loader.
        # The choke point has to admit its own use cases or somebody routes
        # around it, which is how a write path with one documented entrance ends
        # up with two.
        if "chunks" in payload:
            get_store().add(scope, EPISODIC, payload["chunks"])
        else:
            write_episodic_summary(user_id, kw["thread_id"], payload)
    elif kind == "procedural":
        for key, value in payload.items():
            set_pref(user_id, key, value)


def _user_of(scope: str) -> str:
    """Pull the user_id back out of a "user:{uid}" scope string."""
    if not scope.startswith("user:"):
        raise ValueError(f"expected a user-scoped write, got scope={scope!r}")
    return scope.split(":", 1)[1]


# ---------------------------- Session (Redis) ----------------------------

_redis: redis.Redis | None = None


def _r() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(get_settings().redis_url, decode_responses=True)
    return _redis


def _session_key(user_id: str, thread_id: str) -> str:
    """Session keys embed the user_id so GDPR deletion can scan per user."""
    return f"session:{user_id}:{thread_id}"


def session_read_by_thread(thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Read session events by thread_id only - scans for `session:*:{thread_id}`."""
    try:
        _cursor, keys = _r().scan(0, match=f"session:*:{thread_id}", count=100)
        if not keys:
            return []
        raw = _r().lrange(keys[0], -limit, -1)
        return [json.loads(item) for item in raw]
    except Exception as exc:
        log.warning("session_read_by_thread: redis unavailable (%s), returning empty", exc)
        return []


def session_read(user_id: str, thread_id: str, limit: int = 5) -> list[dict[str, Any]]:
    try:
        key = _session_key(user_id, thread_id)
        raw = _r().lrange(key, -limit, -1)
        return [json.loads(item) for item in raw]
    except Exception as exc:
        log.warning("session_read: redis unavailable (%s), returning empty", exc)
        return []


def session_append(user_id: str, thread_id: str, payload: dict[str, Any], ttl_seconds: int = 3600) -> None:
    """Layer writer. Call `memory_write(user_scope(uid), "session", ...)` instead."""
    try:
        key = _session_key(user_id, thread_id)
        pipe = _r().pipeline()
        pipe.rpush(key, json.dumps(payload))
        pipe.expire(key, ttl_seconds)
        pipe.execute()
    except Exception as exc:
        log.warning("session_append: redis unavailable (%s), skipping", exc)


def _user_mem_key(user_id: str) -> str:
    """User-level memory: one rolling list per user, across all their threads."""
    return f"user:{user_id}:memory"


def user_mem_append(user_id: str, payload: dict[str, Any], ttl_seconds: int = 86400) -> None:
    """Append a turn to the user-level memory list. 24h TTL - it outlives a single
    session (1h) but is still demo state, not a forever store."""
    try:
        key = _user_mem_key(user_id)
        pipe = _r().pipeline()
        pipe.rpush(key, json.dumps(payload))
        pipe.expire(key, ttl_seconds)
        pipe.execute()
    except Exception as exc:
        log.warning("user_mem_append: redis unavailable (%s), skipping", exc)


def user_mem_read(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    try:
        raw = _r().lrange(_user_mem_key(user_id), -limit, -1)
        return [json.loads(item) for item in raw]
    except Exception as exc:
        log.warning("user_mem_read: redis unavailable (%s), returning empty", exc)
        return []


def read_session_bundle(thread_id: str, limit: int = 20) -> dict[str, Any]:
    """For the UI: this thread's session events PLUS the owning user's user-level memory."""
    user_id, session_events = "", []
    try:
        _cursor, keys = _r().scan(0, match=f"session:*:{thread_id}", count=100)
        if keys:
            key = keys[0]                                  # session:{user_id}:{thread_id}
            parts = key.split(":")
            if len(parts) >= 3:
                user_id = parts[1]
            raw = _r().lrange(key, -limit, -1)
            session_events = [json.loads(item) for item in raw]
    except Exception as exc:
        log.warning("read_session_bundle: redis unavailable (%s)", exc)
    user_events = user_mem_read(user_id, limit) if user_id else []
    return {"thread_id": thread_id, "user_id": user_id,
            "session_events": session_events, "user_events": user_events}


def session_count_for_user(user_id: str) -> int:
    """How many session keys this user owns - the 'before' counter for GDPR."""
    cursor, total = 0, 0
    try:
        while True:
            cursor, keys = _r().scan(cursor, match=f"session:{user_id}:*", count=100)
            total += len(keys)
            if cursor == 0:
                return total
    except Exception as exc:
        log.warning("session_count_for_user: redis unavailable (%s)", exc)
        return 0


def session_clear_for_user(user_id: str) -> int:
    """Delete a user's session keys - every per-thread list and the user-level list. Returns count deleted.

    Degrades the same way the other session helpers do: if Redis is not
    running, the deletion of the *other three* layers must still succeed and
    still report its counters. A GDPR delete that 500s because one backend is
    down is worse than one that tells you exactly what it managed to erase.
    """
    cursor, deleted = 0, 0
    try:
        while True:
            # SCAN, never KEYS: KEYS blocks Redis while it walks the keyspace.
            cursor, keys = _r().scan(cursor, match=f"session:{user_id}:*", count=100)
            if keys:
                deleted += _r().delete(*keys)
            if cursor == 0:
                break
        deleted += _r().delete(_user_mem_key(user_id))   # user-level memory list
    except Exception as exc:
        log.warning("session_clear_for_user: redis unavailable (%s)", exc)
    return deleted


# ---------------------------- External + episodic (vector store) --------------
# Real vector search - embeddings, cosine top-k, and a (scope, collection)
# metadata filter - over a canned corpus. The backend is chosen by
# MEMORY_BACKEND: an in-process cosine index by default, real Postgres +
# pgvector on request. Neither one needs Docker.

_RUNBOOK_CHUNKS = [
    {"chunk_id": "rb-1", "source": "runbooks/payments-restart.md",
     "text": "If payments-api is unresponsive, run `kubectl rollout restart deployment/payments-api`. "
             "Verify with /health within 90 seconds."},
    {"chunk_id": "rb-2", "source": "runbooks/db-failover.md",
     "text": "Database failover requires DBA approval. Steps: announce in #ops, run failover.sh, monitor lag."},
    {"chunk_id": "rb-3", "source": "runbooks/payments-rollback.md",
     "text": "To roll back a bad payments-api deploy, run `kubectl rollout undo deployment/payments-api`. "
             "Confirm the previous ReplicaSet is healthy before closing the incident."},
    {"chunk_id": "rb-4", "source": "runbooks/checkout-latency.md",
     "text": "Checkout latency spikes are usually connection-pool exhaustion. Check pgbouncer saturation, "
             "then raise the pool ceiling before restarting anything."},
    {"chunk_id": "rb-5", "source": "runbooks/oncall-escalation.md",
     "text": "Escalate to the on-call lead if an incident is unresolved after 30 minutes, or immediately "
             "for anything touching billing or customer payment data."},
]

# Episodic memory: per-user summaries of prior incidents. Seeded so the demo has
# something to retrieve; in production the background summariser writes these.
_EPISODIC_SEED: dict[str, list[dict[str, Any]]] = {
    "u_1": [
        {"chunk_id": "ep-u1-1", "source": "incident/INC-2291",
         "text": "Two weeks ago this user hit the same payments-api outage. The rollout restart fixed it, "
                 "but the root cause was a bad deploy and the rollback was the durable fix."},
        {"chunk_id": "ep-u1-2", "source": "incident/INC-2310",
         "text": "This user previously escalated a billing discrepancy to a human reviewer rather than "
                 "letting the agent act on it."},
    ],
    "u_2": [
        {"chunk_id": "ep-u2-1", "source": "incident/INC-2288",
         "text": "This user's checkout latency incident was traced to connection-pool exhaustion in pgbouncer."},
    ],
}


def seed_memory() -> None:
    """Load the canned corpus into whichever backend is configured.

    Note the seed itself goes through `memory_write` - even the corpus loader
    does not get to bypass the scope check.
    """
    memory_write(GLOBAL_SCOPE, "external", {"chunks": _RUNBOOK_CHUNKS})
    for user_id, chunks in _EPISODIC_SEED.items():
        memory_write(user_scope(user_id), "episodic", {"chunks": chunks})
    log.info("memory.seed_memory runbooks=%d episodic_users=%d",
             len(_RUNBOOK_CHUNKS), len(_EPISODIC_SEED))


def search_runbooks(query: str, k: int = 5) -> list[dict[str, Any]]:
    """External memory: vector search, scope=global, collection=runbooks."""
    hits = get_store().search(query, scope=GLOBAL_SCOPE, collection=RUNBOOKS, k=k)
    log.info("memory.search_runbooks query=%s k=%d hits=%d", query[:60], k, len(hits))
    return hits


def search_episodic(query: str, user_id: str, k: int = 3) -> list[dict[str, Any]]:
    """Episodic memory - scoped to one user_id. NEVER skip the scope filter."""
    assert user_id, "scope filter required"
    hits = get_store().search(query, scope=user_scope(user_id), collection=EPISODIC, k=k)
    log.info("memory.search_episodic query=%s user_id=%s k=%d hits=%d",
             query[:60], user_id, k, len(hits))
    return hits


def write_episodic_summary(user_id: str, thread_id: str, summary: dict[str, Any]) -> None:
    """Background summariser writes here. Scoped, idempotent on thread_id."""
    chunk = {
        "chunk_id": f"ep-{thread_id}",
        "source": f"incident/{thread_id}",
        "text": summary.get("text") or json.dumps(summary),
    }
    get_store().add(user_scope(user_id), EPISODIC, [chunk])
    log.info("memory.write_episodic_summary user_id=%s thread_id=%s", user_id, thread_id)


def count_user_long_term(user_id: str) -> dict[str, int]:
    """The 'before' counters for the deletion receipt."""
    store = get_store()
    return {
        "external_rows": store.count(user_scope(user_id), RUNBOOKS),
        "episodic_rows": store.count(user_scope(user_id), EPISODIC),
    }


def delete_user_long_term(user_id: str) -> dict[str, int]:
    """GDPR delete path - wipe both episodic and any user-scoped external memory.

    Note what is NOT deleted: the global runbook corpus. It is scoped `global`,
    not `user:{uid}`, so a user deletion cannot touch it. That is the scope
    filter earning its keep in the other direction.
    """
    store = get_store()
    scope = user_scope(user_id)
    external = store.delete_scope(scope, RUNBOOKS)
    episodic = store.delete_scope(scope, EPISODIC)
    log.warning("memory.delete_user_long_term user_id=%s external=%d episodic=%d",
                user_id, external, episodic)
    return {"external_rows_deleted": external, "episodic_rows_deleted": episodic}


# ---------------------------- Procedural (prefs) ----------------------------

# In-process dict standing in for a Postgres single-row JSONB per user.
_PREFS: dict[str, dict[str, Any]] = {}


def get_prefs(user_id: str) -> dict[str, Any]:
    return _PREFS.get(user_id, {})


def set_pref(user_id: str, key: str, value: Any) -> None:
    """Layer writer. Call `memory_write(user_scope(uid), "procedural", ...)` instead."""
    _PREFS.setdefault(user_id, {})[key] = value


def delete_prefs(user_id: str) -> int:
    """Returns the number of procedural rows deleted (0 or 1) - one row per user."""
    return 1 if _PREFS.pop(user_id, None) is not None else 0


def seed_demo_prefs() -> None:
    """The three rules `compose_system_prompt` implements - now actually written.

    Without this, prefs were always empty and no procedural rule was ever
    injected into any system prompt. The rule engine existed; nothing fed it.
    """
    memory_write(user_scope("u_1"), "procedural", {
        "always_escalate_billing": True,
        "preferred_response_length": "concise",
        "blocked_tools": ["scale_service"],
    })
    log.info("memory.seed_demo_prefs user_id=u_1 rules=3")
