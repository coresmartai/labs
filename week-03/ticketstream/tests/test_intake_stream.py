"""TicketStream smoke tests - no real API calls.

Mocks at the validate_and_correct boundary so the generator can be driven
with deterministic ticket outputs.
"""
import asyncio
import json
from unittest.mock import patch, AsyncMock
import pytest
from fastapi.testclient import TestClient

from app.main import app, intake_generator
from app.schemas import (
    IntakeRequest, TicketSchema, OrderIntent, RefundIntent, RoutingResult,
)
from app.tools import team_for_intent, route_to_team

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_team_routing_is_deterministic():
    t = TicketSchema(intent=OrderIntent(order_id="ORD-1042"), priority="high")
    assert team_for_intent(t) == "fulfillment"
    t = TicketSchema(
        intent=RefundIntent(order_id="ORD-1", reason="damaged"),
        priority="medium",
    )
    assert team_for_intent(t) == "refunds"


def test_route_to_team_returns_typed_result():
    r = route_to_team("fulfillment", "high")
    assert isinstance(r, RoutingResult)
    assert r.team == "fulfillment"
    assert r.priority == "high"


def test_ticket_schema_rejects_invalid_priority():
    """The Literal enum must enforce the priority value at the boundary."""
    with pytest.raises(Exception):
        TicketSchema(
            intent=OrderIntent(order_id="ORD-1"),
            priority="urgent",  # not in the enum
        )


def test_ticket_schema_discriminator_picks_subclass():
    """Discriminated union: passing type='refund' produces RefundIntent."""
    t = TicketSchema.model_validate({
        "intent": {"type": "refund", "order_id": "ORD-1", "reason": "damaged"},
        "priority": "high",
    })
    assert isinstance(t.intent, RefundIntent)


@pytest.mark.asyncio
async def test_intake_generator_emits_frames_in_order(monkeypatch):
    """End-to-end async test of the generator with a stubbed model."""
    ticket = TicketSchema(
        intent=OrderIntent(order_id="ORD-1042"),
        priority="high",
        customer_id=8821,
    )

    async def fake_validate(msg: str, model_name: str | None = None):
        return ticket, {"correction_count": 0, "attempts": 1}

    monkeypatch.setattr("app.main.validate_and_correct", fake_validate)

    frames = []
    async for frame in intake_generator(
        IntakeRequest(message="Where is ORD-1042?", customer_id=8821)
    ):
        frames.append(frame)

    # Pull event types out of the raw SSE lines
    events = [
        line.split("event: ")[1].split("\n")[0]
        for line in frames if line.startswith("event:")
    ]
    assert events == ["intent", "priority", "customer_id", "routed", "done"]

    # The intent frame carries an order_id
    intent_data = json.loads(frames[0].split("data: ")[1].strip())
    assert intent_data["order_id"] == "ORD-1042"


@pytest.mark.asyncio
async def test_intake_generator_emits_validation_failed_when_model_fails(monkeypatch):
    """When validate_and_correct returns None, we emit validation_failed + done."""

    async def fake_validate(msg: str, model_name: str | None = None):
        return None, {"correction_count": 2, "attempts": 3}

    monkeypatch.setattr("app.main.validate_and_correct", fake_validate)

    frames = []
    async for frame in intake_generator(
        IntakeRequest(message="bla bla")
    ):
        frames.append(frame)

    events = [
        line.split("event: ")[1].split("\n")[0]
        for line in frames if line.startswith("event:")
    ]
    assert events == ["validation_failed", "done"]
