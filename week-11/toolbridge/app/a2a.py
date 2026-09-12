"""The A2A client half. THIS FILE SHIPS COMPLETE - you do not edit it.

It is the part of `scripts/a2a_client.py` from AgentMesh that a bridge needs,
with the CLI stripped off: discover a peer, pick an interface, submit a task,
follow the stream with replay, answer the gate, read a status.

Read it anyway. Three things in here are the reason the tools above it can be
short, and all three are on the exam:

1.  `select_interface` walks `supportedInterfaces` IN ORDER and stops at the
    first binding we can speak. Entry zero is the peer's preference, not an
    exclusive claim.
2.  `follow` re-attaches from the last event id it saw rather than resubmitting.
    A dropped stream on a task that pauses for a human is normal, not an error.
3.  A task in an INTERRUPTED state is alive. `follow` returns rather than
    hanging, and hands the caller a task id it can come back to.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

CARD_PATH = "/.well-known/agent-card.json"

# The protocol bindings this client can actually speak. Deliberately a SET we
# test membership against, not an exhaustive match: `protocolBinding` is an open
# string in A2A v1.0, so a binding nobody has thought of yet must not crash us.
SPEAKABLE_BINDINGS = {"HTTP+JSON"}

TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"}
INTERRUPTED = {"TASK_STATE_INPUT_REQUIRED", "TASK_STATE_AUTH_REQUIRED"}


class PeerError(Exception):
    """A peer answered, and the answer was not usable."""

    def __init__(self, code: str, message: str, retryable: bool, status: int | None = None):
        super().__init__(message)
        self.code, self.message, self.retryable, self.status = code, message, retryable, status


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def fetch_card(client: httpx.AsyncClient, base: str) -> dict[str, Any]:
    """Discovery. Unauthenticated by design: anyone may read an Agent Card."""
    url = f"{base.rstrip('/')}{CARD_PATH}"
    try:
        r = await client.get(url, timeout=15.0)
    except httpx.RequestError as exc:
        raise PeerError("peer_unreachable", f"Could not reach {url}: {exc}", retryable=True) from exc
    if r.status_code == 404:
        raise PeerError(
            "card_not_found",
            f"No Agent Card at {url}. A peer serving /.well-known/agent.json instead "
            "is running a build from before A2A v0.3.0 (July 2025).",
            retryable=False,
            status=404,
        )
    if r.status_code != 200:
        raise PeerError("card_fetch_failed", f"Card fetch returned {r.status_code}.", retryable=True, status=r.status_code)
    return r.json()


def select_interface(card: dict[str, Any]) -> dict[str, Any]:
    """Walk supportedInterfaces IN ORDER and return the first we can speak.

    Ordered by the PEER's preference. A caller that stops at entry zero because
    it cannot speak that binding is throwing away an interface it could have
    used, which is the most common misreading of this field.
    """
    interfaces = card.get("supportedInterfaces") or []
    if not interfaces:
        raise PeerError(
            "card_missing_interfaces",
            "Card has no supportedInterfaces[]. That field is REQUIRED in A2A v1.0; "
            "a card carrying only a top-level `url` predates March 2026.",
            retryable=False,
        )
    for entry in interfaces:
        if entry.get("protocolBinding") in SPEAKABLE_BINDINGS:
            return entry
    offered = [e.get("protocolBinding") for e in interfaces]
    raise PeerError(
        "no_common_binding",
        f"Peer offers {offered}; this bridge speaks {sorted(SPEAKABLE_BINDINGS)}.",
        retryable=False,
    )


async def submit(client: httpx.AsyncClient, base: str, token: str, skill: str,
                 payload: dict[str, Any]) -> str:
    """Submit a task. Returns the task id."""
    try:
        r = await client.post(
            f"{base.rstrip('/')}/tasks",
            headers={**_auth(token), "Content-Type": "application/json"},
            json={"skill": skill, "input": payload},
            timeout=30.0,
        )
    except httpx.RequestError as exc:
        raise PeerError("peer_unreachable", f"Could not reach the peer: {exc}", retryable=True) from exc
    if r.status_code == 401:
        raise PeerError("peer_unauthorized", "The peer rejected our bearer token.", retryable=False, status=401)
    if r.status_code == 403:
        raise PeerError("peer_forbidden", "Our token is valid but lacks the scope for this skill.", retryable=False, status=403)
    if r.status_code == 400:
        raise PeerError("unknown_skill", f"The peer does not offer that skill: {r.text[:160]}", retryable=False, status=400)
    if r.status_code == 422:
        raise PeerError("invalid_input", f"The peer rejected the input shape: {r.text[:200]}", retryable=False, status=422)
    if r.status_code != 202:
        raise PeerError("submit_failed", f"Submit returned {r.status_code}.", retryable=True, status=r.status_code)
    return r.json()["task_id"]


async def _stream_leg(client: httpx.AsyncClient, url: str, token: str, cursor: int,
                      timeout: float) -> tuple[str | None, int, dict[str, Any]]:
    """One leg of a stream. Returns (state, last_event_id_seen, payload).

    `state` is None when the stream ended without reporting one, which is the
    signal to stop trusting the stream and poll instead.
    """
    headers = dict(_auth(token))
    if cursor:
        headers["Last-Event-ID"] = str(cursor)
    state: str | None = None
    payload: dict[str, Any] = {}
    async with client.stream("GET", url, headers=headers, timeout=timeout) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")
            raise PeerError("stream_failed", f"Stream returned {response.status_code}: {body[:160]}",
                            retryable=True, status=response.status_code)
        async for line in response.aiter_lines():
            if line.startswith("id:"):
                try:
                    cursor = int(line[3:].strip())
                except ValueError:
                    pass
                continue
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            state = event.get("state", state)
            payload = event.get("payload", payload) or payload
            if state in TERMINAL or state in INTERRUPTED:
                return state, cursor, payload
    return state, cursor, payload


async def follow(client: httpx.AsyncClient, base: str, token: str, task_id: str,
                 timeout: float, from_event_id: int = 0) -> tuple[str, dict[str, Any], int]:
    """Follow a task to a terminal OR interrupted state.

    Returns (state, payload, cursor), where `cursor` is the last event id seen.

    `from_event_id` is THE REASON THIS SIGNATURE IS NOT SIMPLER, and it is worth
    understanding rather than copying. A stream replays every event with an id
    greater than the cursor you present. Start from zero after answering a gate
    and the peer faithfully replays the pause you just answered, you read it as
    the task's current state, and you report a pending approval for a task that
    has already completed. That is not a hypothetical: it is what this function
    did before the cursor was threaded through, and the mocked tests did not
    catch it because a fresh fake replays nothing.

    So the caller carries the cursor. Which is exactly the handle pattern from
    MCP: the protocol will not remember for you, so you hand back a name and
    take it as an argument next time.
    """
    url = f"{base.rstrip('/')}/tasks/{task_id}/stream"
    cursor = from_event_id
    for _ in range(3):
        state, cursor, payload = await _stream_leg(client, url, token, cursor, timeout)
        if state in TERMINAL or state in INTERRUPTED:
            return state, payload, cursor
        # The stream told us nothing. tasks/get is a finite response and always
        # gets through: slower, never stuck.
        snap = await get_status(client, base, token, task_id)
        state = snap.get("state", "TASK_STATE_UNSPECIFIED")
        if state in TERMINAL or state in INTERRUPTED:
            return state, snap.get("payload") or {}, cursor
    raise PeerError("follow_exhausted", "Could not reach a settled state for this task.", retryable=True)


async def reply_to_gate(client: httpx.AsyncClient, base: str, token: str, task_id: str,
                        approved: bool, note: str) -> None:
    """Answer a paused task. A SECOND request; the stream is one-directional."""
    r = await client.post(
        f"{base.rstrip('/')}/tasks/{task_id}/input",
        headers={**_auth(token), "Content-Type": "application/json"},
        json={"approved": approved, "note": note},
        timeout=30.0,
    )
    if r.status_code == 409:
        raise PeerError(
            "task_not_paused",
            "That task is not waiting for input. It is either still working or already terminal, "
            "and a terminal task accepts no further messages.",
            retryable=False, status=409,
        )
    if r.status_code == 404:
        raise PeerError("unknown_task", "The peer has no task with that id.", retryable=False, status=404)
    if r.status_code != 200:
        raise PeerError("reply_failed", f"Reply returned {r.status_code}.", retryable=True, status=r.status_code)


async def get_status(client: httpx.AsyncClient, base: str, token: str, task_id: str) -> dict[str, Any]:
    r = await client.get(f"{base.rstrip('/')}/tasks/{task_id}", headers=_auth(token), timeout=15.0)
    if r.status_code == 404:
        raise PeerError("unknown_task", "The peer has no task with that id.", retryable=False, status=404)
    if r.status_code != 200:
        raise PeerError("status_failed", f"Status returned {r.status_code}.", retryable=True, status=r.status_code)
    return r.json()
