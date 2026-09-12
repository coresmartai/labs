"""Task store - one interface, two implementations.

The interface is what matters. Any backing store that satisfies the four methods
the video names on camera - `create`, `get_state`, `update_state`, `stream` -
slots in without a line of endpoint code changing. `MemoryTaskStore` is the
default and what the demo runs on. `SqliteTaskStore` is the same interface with
durable events and unbounded replay - the seam made real, not promised.

THE EVENT-ID RULE (this is the bug that ate the replay demo):
    event_id is a PER-TASK MONOTONIC COUNTER. It is never derived from the length
    of the replay buffer. `len(events) + 1` looks right until the bounded buffer
    wraps at TASK_REPLAY_BUFFER_SIZE - after that the id stalls at 101 and repeats
    forever, and every Last-Event-ID replay on a long task silently returns the
    wrong events. The counter below is the fix, and test 13 is the regression.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Protocol

from app.config import get_settings
from app.schemas import TERMINAL_STATES, TaskEvent, TaskState

logger = logging.getLogger(__name__)


def _new_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:12]}"


class TaskStore(Protocol):
    """The four methods, plus the input-required handshake."""

    def create(self) -> str: ...
    def get_state(self, task_id: str) -> TaskState | None: ...
    # The most recent event, or None for an unknown task. This is what backs the
    # `tasks/get`-style snapshot: a client that never held a stream - or lost one -
    # still needs the current state AND the payload that came with it.
    def latest(self, task_id: str) -> TaskEvent | None: ...
    async def update_state(
        self, task_id: str, new_state: TaskState, payload: dict[str, Any] | None = None
    ) -> TaskEvent: ...
    def stream(self, task_id: str, from_event_id: int = 0) -> AsyncIterator[TaskEvent]: ...

    def begin_input_required(self, task_id: str) -> None: ...
    async def submit_input(self, task_id: str, payload: dict[str, Any]) -> bool: ...
    async def wait_for_input(self, task_id: str, timeout: float = 300.0) -> dict[str, Any] | None: ...


class _InputHandshakeMixin:
    """The human-in-the-loop pause. In-process by design: an asyncio.Event cannot be
    persisted, and the approval only has to survive the lifetime of the request that
    is waiting on it."""

    def __init__(self) -> None:
        self._pending_input: dict[str, asyncio.Event] = {}
        self._pending_input_payload: dict[str, dict[str, Any]] = {}

    def begin_input_required(self, task_id: str) -> None:
        self._pending_input[task_id] = asyncio.Event()

    async def submit_input(self, task_id: str, payload: dict[str, Any]) -> bool:
        ev = self._pending_input.get(task_id)
        if ev is None:
            return False
        self._pending_input_payload[task_id] = payload
        ev.set()
        return True

    async def wait_for_input(self, task_id: str, timeout: float = 300.0) -> dict[str, Any] | None:
        ev = self._pending_input.get(task_id)
        if ev is None:
            return None
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            return self._pending_input_payload.pop(task_id, None)
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_input.pop(task_id, None)


def _guard(task_id: str, new_state: TaskState, current: TaskState | None) -> None:
    """Two invariants the state machine must never break."""
    if new_state is TaskState.TASK_STATE_UNSPECIFIED:
        # The sentinel is unassignable. It exists so an uninitialised field is
        # visibly uninitialised - it is not a state a task can be in.
        raise ValueError("TASK_STATE_UNSPECIFIED is a sentinel and can never be assigned")
    if current in TERMINAL_STATES:
        raise ValueError(f"task {task_id} is already terminal ({current.value}); no further transitions")


class MemoryTaskStore(_InputHandshakeMixin):
    """In-memory implementation. Replay buffer bounded at TASK_REPLAY_BUFFER_SIZE
    (100): the orchestrator gets a hundred events of disconnect tolerance, and
    beyond that it is told there is a gap rather than being handed the wrong tail."""

    def __init__(self) -> None:
        super().__init__()
        self._buffer_size = get_settings().task_replay_buffer_size
        self._events: dict[str, deque[TaskEvent]] = {}
        self._next_event_id: dict[str, int] = defaultdict(int)
        self._state: dict[str, TaskState] = {}
        self._subscribers: dict[str, list[asyncio.Queue[TaskEvent]]] = {}

    def create(self) -> str:
        task_id = _new_task_id()
        self._events[task_id] = deque(maxlen=self._buffer_size)
        self._subscribers[task_id] = []
        self._next_event_id[task_id] = 0
        return task_id

    def get_state(self, task_id: str) -> TaskState | None:
        return self._state.get(task_id)

    def latest(self, task_id: str) -> TaskEvent | None:
        events = self._events.get(task_id)
        return events[-1] if events else None

    async def update_state(
        self, task_id: str, new_state: TaskState, payload: dict[str, Any] | None = None
    ) -> TaskEvent:
        if task_id not in self._events:
            raise KeyError(f"unknown task: {task_id}")
        _guard(task_id, new_state, self._state.get(task_id))

        # Monotonic, per task, never derived from the buffer's length.
        self._next_event_id[task_id] += 1
        event = TaskEvent(
            event_id=self._next_event_id[task_id],
            task_id=task_id,
            state=new_state,
            timestamp=datetime.now(timezone.utc),
            payload=payload or {},
        )
        self._events[task_id].append(event)
        self._state[task_id] = new_state
        for q in self._subscribers[task_id]:
            await q.put(event)
        return event

    async def stream(self, task_id: str, from_event_id: int = 0) -> AsyncIterator[TaskEvent]:
        """Replay every buffered event past the cursor, then subscribe live until the
        task reaches a terminal state."""
        if task_id not in self._events:
            raise KeyError(f"unknown task: {task_id}")

        buffered = list(self._events[task_id])

        # Gap signalling: the caller's cursor is older than the oldest event we still
        # hold. Say so explicitly - a silent short replay is how an orchestrator ends
        # up confidently acting on a task it never saw the middle of.
        if buffered and from_event_id + 1 < buffered[0].event_id:
            yield TaskEvent(
                event_id=buffered[0].event_id - 1,
                task_id=task_id,
                state=self._state[task_id],
                timestamp=datetime.now(timezone.utc),
                payload={"replay_gap": True, "oldest_available_event_id": buffered[0].event_id},
            )

        for event in buffered:
            if event.event_id > from_event_id:
                yield event
                if event.state in TERMINAL_STATES:
                    return

        q: asyncio.Queue[TaskEvent] = asyncio.Queue()
        self._subscribers[task_id].append(q)
        try:
            while True:
                event = await q.get()
                yield event
                if event.state in TERMINAL_STATES:
                    return
        finally:
            self._subscribers[task_id].remove(q)


class SqliteTaskStore(_InputHandshakeMixin):
    """The same four methods, backed by SQLite. Events are durable and replay is
    unbounded - reconnect with any cursor and every event since is still there.
    Swapping this in is one setting: TASK_STORE_BACKEND=sqlite."""

    def __init__(self, db_path: str) -> None:
        super().__init__()
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, state TEXT NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "  task_id TEXT NOT NULL, event_id INTEGER NOT NULL, state TEXT NOT NULL,"
            "  timestamp TEXT NOT NULL, payload TEXT NOT NULL,"
            "  PRIMARY KEY (task_id, event_id))"
        )
        self._db.commit()
        self._subscribers: dict[str, list[asyncio.Queue[TaskEvent]]] = defaultdict(list)

    def create(self) -> str:
        task_id = _new_task_id()
        self._db.execute(
            "INSERT INTO tasks (task_id, state) VALUES (?, ?)",
            (task_id, TaskState.TASK_STATE_UNSPECIFIED.value),
        )
        self._db.commit()
        return task_id

    def _row_state(self, task_id: str) -> str | None:
        row = self._db.execute("SELECT state FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return row[0] if row else None

    def get_state(self, task_id: str) -> TaskState | None:
        raw = self._row_state(task_id)
        if raw is None or raw == TaskState.TASK_STATE_UNSPECIFIED.value:
            # UNSPECIFIED is the "no state yet" sentinel in the row, never a live state.
            return None
        return TaskState(raw)

    def latest(self, task_id: str) -> TaskEvent | None:
        row = self._db.execute(
            "SELECT event_id, state, timestamp, payload FROM events "
            "WHERE task_id = ? ORDER BY event_id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return TaskEvent(
            event_id=row[0], task_id=task_id, state=TaskState(row[1]),
            timestamp=row[2], payload=json.loads(row[3]),
        )

    async def update_state(
        self, task_id: str, new_state: TaskState, payload: dict[str, Any] | None = None
    ) -> TaskEvent:
        if self._row_state(task_id) is None:
            raise KeyError(f"unknown task: {task_id}")
        _guard(task_id, new_state, self.get_state(task_id))

        row = self._db.execute(
            "SELECT COALESCE(MAX(event_id), 0) FROM events WHERE task_id = ?", (task_id,)
        ).fetchone()
        event = TaskEvent(
            event_id=row[0] + 1,  # monotonic: MAX(event_id), never COUNT(*)
            task_id=task_id,
            state=new_state,
            timestamp=datetime.now(timezone.utc),
            payload=payload or {},
        )
        self._db.execute(
            "INSERT INTO events (task_id, event_id, state, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
            (
                task_id,
                event.event_id,
                new_state.value,
                event.timestamp.isoformat(),
                json.dumps(event.payload),
            ),
        )
        self._db.execute("UPDATE tasks SET state = ? WHERE task_id = ?", (new_state.value, task_id))
        self._db.commit()
        for q in self._subscribers[task_id]:
            await q.put(event)
        return event

    async def stream(self, task_id: str, from_event_id: int = 0) -> AsyncIterator[TaskEvent]:
        if self._row_state(task_id) is None:
            raise KeyError(f"unknown task: {task_id}")
        rows = self._db.execute(
            "SELECT event_id, state, timestamp, payload FROM events "
            "WHERE task_id = ? AND event_id > ? ORDER BY event_id",
            (task_id, from_event_id),
        ).fetchall()
        for event_id, state, ts, payload in rows:
            event = TaskEvent(
                event_id=event_id,
                task_id=task_id,
                state=TaskState(state),
                timestamp=ts,
                payload=json.loads(payload),
            )
            yield event
            if event.state in TERMINAL_STATES:
                return

        q: asyncio.Queue[TaskEvent] = asyncio.Queue()
        self._subscribers[task_id].append(q)
        try:
            while True:
                event = await q.get()
                yield event
                if event.state in TERMINAL_STATES:
                    return
        finally:
            self._subscribers[task_id].remove(q)


# Module-level singleton.
_store: TaskStore | None = None


def get_store() -> TaskStore:
    global _store
    if _store is None:
        settings = get_settings()
        if settings.task_store_backend == "sqlite":
            logger.info("task_store.backend=sqlite path=%s", settings.task_store_sqlite_path)
            _store = SqliteTaskStore(settings.task_store_sqlite_path)
        else:
            logger.info("task_store.backend=memory buffer=%d", settings.task_replay_buffer_size)
            _store = MemoryTaskStore()
    return _store


def reset_for_tests() -> None:
    global _store
    _store = None
