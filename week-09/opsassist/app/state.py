"""Three-layer state architecture from V3, instantiated.

ShortTermState - in-process dataclass.
WorkflowState  - Redis-backed (or local dict fallback for dev).
PersistentState - SQLite-backed in this build; PostgreSQL in Production.

The interfaces are identical regardless of backend, so the lab can
run on a laptop without spinning up infrastructure.

V3's three sanctioned crossings, as implemented here:
  1. short-term reads workflow  -> the resume read, at request entry (main.py)
  2. workflow writes a checkpoint after every successful tool call (agent.py)
  3. workflow reads persistent ONCE at session start (load_session_context)
     and writes the audit log ONCE at session end (append_audit)
Every other crossing is a bug in waiting. There are zero persistent reads
inside the agent loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import time
import sqlite3
from pathlib import Path

from app.config import get_settings


# =====================================================================
# Short-term state - in-process, one iteration's working memory.
# =====================================================================

@dataclass
class ShortTermState:
    """Working memory for one task run.

    Dies when the request completes. Never persisted.
    """
    messages: list[dict] = field(default_factory=list)
    iter_count: int = 0
    token_budget: int = 16000  # decremented per iteration
    tokens_used: int = 0
    cost_usd: float = 0.0      # accumulated alongside tokens - the dollar stop
    created_at: float = field(default_factory=time.time)

    def append_message(self, msg: dict) -> None:
        self.messages.append(msg)

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> None:
        """Accumulate tokens and dollars from one model turn.

        `cached_input_tokens` is the subset of `input_tokens` the provider
        served from its prompt cache; it bills at 10% of the input rate.
        """
        self.tokens_used += input_tokens + output_tokens
        settings = get_settings()
        cached = min(max(cached_input_tokens, 0), input_tokens)
        fresh_input = input_tokens - cached
        self.cost_usd += (
            fresh_input / 1_000_000 * settings.price_per_1m_input_usd
            + cached / 1_000_000 * settings.price_per_1m_cached_input_usd
            + output_tokens / 1_000_000 * settings.price_per_1m_output_usd
        )

    def remaining_budget(self) -> int:
        return self.token_budget - self.tokens_used

    def digest(self) -> str:
        """A recovery hint, not a snapshot. Bounded, cheap, deterministic.

        No extra model call: the last thing the user asked for, plus the
        names of the tools already run. That is enough for the loop to
        pick the task back up.
        """
        last_user = next(
            (m.get("content") or "" for m in reversed(self.messages)
             if m.get("role") == "user"), "")
        tools_run = [c["function"]["name"]
                     for m in self.messages if m.get("tool_calls")
                     for c in m["tool_calls"]]
        return (f"Task so far: {last_user[:200]} | "
                f"tools already run: {', '.join(tools_run) or 'none'}")

    def snapshot(self) -> dict:
        """Dump for stdout/observability."""
        return {
            "iter_count": self.iter_count,
            "messages_len": len(self.messages),
            "tokens_used": self.tokens_used,
            "remaining_budget": self.remaining_budget(),
            "cost_usd": round(self.cost_usd, 6),
        }


# =====================================================================
# Workflow state - Redis-backed, lives 24h.
# =====================================================================

class WorkflowState:
    """One task's progress, with checkpoints.

    Uses Redis when REDIS_URL is set; otherwise falls back to a
    process-local dict (suitable for local development and tests).
    """

    _local_store: dict[str, dict] = {}

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.key = f"workflow:{task_id}"
        self._redis = self._maybe_redis()

    @staticmethod
    def _maybe_redis():
        settings = get_settings()
        if not settings.redis_url:
            return None
        try:
            import redis  # imported lazily so dev runs don't need redis-py installed
            return redis.from_url(settings.redis_url, decode_responses=True)
        except ImportError:
            return None

    def write_checkpoint(self, checkpoint: dict) -> None:
        payload = json.dumps(checkpoint)
        if self._redis is not None:
            self._redis.setex(self.key, 86400, payload)  # 24h TTL
        else:
            WorkflowState._local_store[self.key] = checkpoint

    def read_checkpoint(self) -> dict | None:
        if self._redis is not None:
            raw = self._redis.get(self.key)
            return json.loads(raw) if raw else None
        return WorkflowState._local_store.get(self.key)

    def clear(self) -> None:
        if self._redis is not None:
            self._redis.delete(self.key)
        else:
            WorkflowState._local_store.pop(self.key, None)


# =====================================================================
# Persistent state - PostgreSQL (or SQLite in dev).
# =====================================================================

class PersistentState:
    """User-level state that survives forever.

    Reads happen once at session start; writes happen once at session end.
    Never hit from inside the agent loop.
    """

    _SQLITE_PATH = Path(".opsassist.db")
    _initialised = False

    def __init__(self):
        self._ensure_local_schema()

    def _ensure_local_schema(self) -> None:
        if PersistentState._initialised:
            return
        # Always init the SQLite file in dev - production swap-in
        # would replace this with a real migration runner.
        conn = sqlite3.connect(self._SQLITE_PATH)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_prefs ("
            "user_id TEXT PRIMARY KEY, prefs_json TEXT, consent INTEGER DEFAULT 1)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS audit_log ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_id TEXT, ts REAL, event TEXT)"
        )
        conn.commit()
        conn.close()
        PersistentState._initialised = True

    def load_session_context(self, user_id: str) -> dict:
        """Crossing 3: the ONE persistent read, at session start.

        Returns {"consent": bool, "prefs": dict}. One connection, one SELECT,
        both fields together. Never called from inside the agent loop - see
        V3's three sanctioned crossings.
        """
        conn = sqlite3.connect(self._SQLITE_PATH)
        cur = conn.execute(
            "SELECT prefs_json, consent FROM user_prefs WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        conn.close()
        # Default to consent True if the row is missing - real production
        # would default to False and require explicit opt-in.
        if row is None:
            return {"consent": True, "prefs": {}}
        return {
            "consent": bool(row[1]),
            "prefs": json.loads(row[0]) if row[0] else {},
        }

    def check_consent(self, user_id: str) -> bool:
        """Consent on its own, for callers outside a session.

        The agent loop does NOT use this: it reads the flag cached on the
        Session at session start. Calling this from inside the loop would
        be a fourth crossing, which is to say a bug.
        """
        return bool(self.load_session_context(user_id)["consent"])

    def append_audit(self, user_id: str, event: dict) -> None:
        conn = sqlite3.connect(self._SQLITE_PATH)
        conn.execute(
            "INSERT INTO audit_log (user_id, ts, event) VALUES (?, ?, ?)",
            (user_id, time.time(), json.dumps(event)),
        )
        conn.commit()
        conn.close()

    def load_transcript(self, user_id: str, task_id: str) -> list[dict]:
        """Resolve a checkpoint's `transcript_ref` back to the message list.

        The checkpoint stores a hint, not a snapshot. The full conversation
        lives here, in the audit log, written once at session end. This is
        the rebuild path V3 promises - byte-exact, and off the hot path.
        """
        conn = sqlite3.connect(self._SQLITE_PATH)
        cur = conn.execute(
            "SELECT event FROM audit_log WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        )
        rows = cur.fetchall()
        conn.close()
        for (raw,) in rows:
            event = json.loads(raw)
            if event.get("task_id") == task_id and event.get("transcript"):
                return list(event["transcript"])
        return []


# =====================================================================
# Session - the cohering abstraction.
# =====================================================================

@dataclass
class Session:
    """One session per task. Created at start, destroyed at end."""
    task_id: str
    user_id: str
    short_term: ShortTermState
    workflow: WorkflowState
    persistent: PersistentState
    # Read once at session start by load_session_context, then cached here.
    # The loop's consent guard reads this flag, not the database.
    consent: bool = True
    prefs: dict = field(default_factory=dict)
