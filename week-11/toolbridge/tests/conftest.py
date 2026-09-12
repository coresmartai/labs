"""Test fixtures. No network, no model, no live peer, sub-second.

The peer is a FAKE A2A service built on httpx.MockTransport. That is not a
shortcut: a test that needs somebody else's agent running on port 8001 is a test
nobody runs, and a bridge whose failure handling is only exercised against a
healthy peer has untested failure handling.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

os.environ.setdefault("PEER_BASE_URL", "http://peer.test")
os.environ.setdefault("PEER_TOKEN", "test-token")
os.environ.setdefault("FOLLOW_TIMEOUT_SECONDS", "5")


GOOD_CARD: dict[str, Any] = {
    "name": "agentmesh-stu_001",
    "version": "1.0.0",
    "protocolVersion": "1.0",
    "description": "AgentMesh instance operated by Alice",
    "provider": {"organization": "Alice", "url": "http://peer.test"},
    "capabilities": {"streaming": True, "pushNotifications": False,
                     "extensions": [], "extendedAgentCard": False},
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "supportedInterfaces": [
        {"url": "http://peer.test", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
    ],
    "skills": [
        {"id": "triage_incident", "name": "Triage an incident",
         "description": "Classify and act on an engineering incident.",
         "tags": ["ops"], "examples": [], "inputModes": ["application/json"],
         "outputModes": ["application/json"]},
        {"id": "whoami", "name": "Identify this agent", "description": "Who runs this agent.",
         "tags": ["identity"], "examples": [], "inputModes": ["application/json"],
         "outputModes": ["application/json"]},
    ],
    "signatures": [],
}


class FakePeer:
    """A small A2A service. Scriptable per test, and it records what it was sent."""

    def __init__(self) -> None:
        self.card: dict[str, Any] | None = dict(GOOD_CARD)
        self.card_status = 200
        self.submit_status = 202
        self.submit_body: dict[str, Any] = {"task_id": "tk_test"}
        # The task's event log: (event_id, state, payload), ids monotonic from 1.
        self.events: list[tuple[int, str, dict[str, Any]]] = []
        # When set, the stream cuts off after this event id, once.
        self.drop_after: int | None = None
        self.reply_status = 200
        self.status_snapshots: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str]] = []
        self.last_headers: dict[str, str] = {}
        self.last_submit_json: dict[str, Any] | None = None
        self.last_reply_json: dict[str, Any] | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        self.last_headers = dict(request.headers)

        if path == "/.well-known/agent-card.json":
            if self.card_status != 200 or self.card is None:
                return httpx.Response(self.card_status, text="nope")
            return httpx.Response(200, json=self.card)

        if path == "/tasks" and request.method == "POST":
            self.last_submit_json = json.loads(request.content or b"{}")
            if self.submit_status != 202:
                return httpx.Response(self.submit_status, text="refused")
            # VALIDATE, exactly as the real peer's TriageInput does. A fake that
            # is more permissive than the service it stands in for is worse than
            # no fake: it makes a broken client look tested. This block exists
            # because an earlier version of it did not, and the first run
            # against a live peer returned 422 on a missing user_id.
            payload = self.last_submit_json.get("input") or {}
            if self.last_submit_json.get("skill") == "triage_incident":
                missing = [f for f in ("description", "severity", "user_id") if not payload.get(f)]
                if missing:
                    return httpx.Response(422, json={"detail": [
                        {"type": "missing", "loc": [f], "msg": "Field required"} for f in missing]})
                if len(str(payload["description"])) < 10:
                    return httpx.Response(422, json={"detail": [
                        {"type": "string_too_short", "loc": ["description"]}]})
                if payload["severity"] not in {"low", "medium", "high"}:
                    return httpx.Response(422, json={"detail": [
                        {"type": "enum", "loc": ["severity"]}]})
            return httpx.Response(202, json=self.submit_body)

        if path.endswith("/stream"):
            # A FAITHFUL stream: replay every event with an id GREATER than the
            # Last-Event-ID the client presented, exactly as the real peer does.
            # An earlier version of this fake ignored the cursor and served a
            # fresh scripted leg each time, which made a cursor bug in the
            # bridge invisible to every test in this file. A fake that is easier
            # than the real thing on the exact dimension a bug lives on is not a
            # test, it is a reassurance.
            try:
                cursor = int(request.headers.get("last-event-id", "0"))
            except ValueError:
                cursor = 0
            lines = []
            for eid, state, payload in self.events:
                if eid <= cursor:
                    continue
                lines.append(f"id: {eid}")
                lines.append("data: " + json.dumps({"state": state, "payload": payload}))
                lines.append("")
                if self.drop_after and eid >= self.drop_after:
                    break   # the connection dies here, mid-task
            return httpx.Response(200, text="\n".join(lines),
                                  headers={"Content-Type": "text/event-stream"})

        if path.endswith("/input") and request.method == "POST":
            self.last_reply_json = json.loads(request.content or b"{}")
            if self.reply_status != 200:
                return httpx.Response(self.reply_status, text="no")
            return httpx.Response(200, json={"ok": True})

        if path.startswith("/tasks/"):
            if self.status_snapshots:
                return httpx.Response(200, json=self.status_snapshots.pop(0))
            return httpx.Response(404, text="unknown task")

        return httpx.Response(404, text="not found")


@pytest.fixture
def peer(monkeypatch):
    """Install a fake peer behind every httpx.AsyncClient the app creates."""
    fake = FakePeer()
    real_init = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)
    return fake


def script(fake, *states_and_payloads):
    """Fill a fake peer's event log. ids are assigned 1..n, monotonic."""
    fake.events = [(i, st, pl) for i, (st, pl) in enumerate(states_and_payloads, start=1)]
    return fake
