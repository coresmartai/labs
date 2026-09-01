"""The SSE contract your /intake endpoint must satisfy.

These are RED on a fresh clone and that is the point. Making them green is
the project. They run with NO API key: the three functions in app.llm are
replaced with stubs, so what is under test is your service contract, not the
model's behaviour.

If you find yourself editing this file to make it pass, stop. The contract
is fixed, and the grader runs its own copy.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import llm, main
from app.schemas import Intent, Priority, Ticket

client = TestClient(main.app)


def parse_sse(body: str):
    """Return [(event, data_dict), ...] in the order they arrived."""
    out = []
    for block in body.strip().split("\n\n"):
        ev = dat = None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                dat = json.loads(line.split(":", 1)[1].strip())
        if ev:
            out.append((ev, dat))
    return out


@pytest.fixture
def stub(monkeypatch):
    """Replace the model calls. Configure per test."""
    state = {"ticket": Ticket(type=Intent.billing, order_id="4471",
                              priority=Priority.high),
             "attempts": 1, "corrections": 0, "fail": False}

    async def resolve_lookup(review):
        return [{"role": "user", "content": review}], state["attempts"]

    async def stream_ticket(messages):
        t = state["ticket"]
        yield "type", t.type.value
        yield "order_id", t.order_id
        yield "priority", t.priority.value

    async def validate_with_correction(raw, messages):
        if state["fail"]:
            raise llm.SchemaCorrectionExhausted("priority: input not a valid enum")
        return state["ticket"], state["corrections"]

    monkeypatch.setattr(llm, "resolve_lookup", resolve_lookup)
    monkeypatch.setattr(llm, "stream_ticket", stream_ticket)
    monkeypatch.setattr(llm, "validate_with_correction", validate_with_correction)
    return state


def test_four_events_in_order(stub):
    r = client.post("/intake", json={"review": "double charged, order 4471"})
    assert r.status_code == 200
    assert [e for e, _ in parse_sse(r.text)] == [
        "intent", "priority", "routed", "done"]


def test_streaming_headers_are_set(stub):
    r = client.post("/intake", json={"review": "anything"})
    assert r.headers.get("cache-control") == "no-cache"
    assert r.headers.get("x-accel-buffering") == "no"


def test_intent_event_carries_type_and_order_id(stub):
    events = dict(parse_sse(client.post(
        "/intake", json={"review": "double charged, order 4471"}).text))
    assert events["intent"] == {"type": "billing", "order_id": "4471"}


def test_routed_uses_the_routing_table(stub):
    events = dict(parse_sse(client.post(
        "/intake", json={"review": "double charged, order 4471"}).text))
    assert events["routed"] == {"team": "billing-escalation"}


def test_done_reports_corrections_and_attempts(stub):
    stub["corrections"] = 1
    stub["attempts"] = 2
    events = dict(parse_sse(client.post("/intake", json={"review": "x"}).text))
    assert events["done"] == {"corrections": 1, "attempts": 2}


def test_praise_still_emits_routed_with_a_null_team(stub):
    stub["ticket"] = Ticket(type=Intent.praise, order_id=None,
                            priority=Priority.low)
    events = parse_sse(client.post(
        "/intake", json={"review": "brilliant app"}).text)
    assert [e for e, _ in events] == ["intent", "priority", "routed", "done"]
    assert dict(events)["routed"] == {"team": None}


def test_validation_failure_stops_before_routing(stub):
    stub["fail"] = True
    names = [e for e, _ in parse_sse(client.post(
        "/intake", json={"review": "x"}).text)]
    assert "validation_failed" in names
    assert "routed" not in names
