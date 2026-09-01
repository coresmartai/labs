"""Smoke tests - no real API calls. We mock at the llm.py boundary.

The model boundary is stubbed so these run under a second in CI on every
commit. Real API integration tests live separately and are gated behind
an env var (run nightly, not on every push).
"""
import json
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tools import execute_tool, OrderNotFoundError
from app.schemas import OrderLookupArgs
from pydantic import ValidationError


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model" in data           # template's checkHealth() reads data.model
    assert "models" in data          # both providers exposed


def test_lookup_validation_rejects_empty_message():
    r = client.post("/lookup", json={"message": ""})
    assert r.status_code == 422  # Pydantic min_length=1 enforced


def test_dispatcher_runs_lookup_directly():
    """The dispatcher itself, without the model in the loop."""
    out = execute_tool("lookup_order_status", {"order_id": "ORD-1042"})
    assert out["success"] is True
    assert out["data"]["state"] == "shipped"
    assert out["data"]["order_id"] == "ORD-1042"


def test_dispatcher_returns_structured_error_envelope_on_not_found():
    """Failures must come back as structured envelopes, never as exceptions."""
    out = execute_tool("lookup_order_status", {"order_id": "ORD-9999"})
    assert out["success"] is False
    assert out["error"] == "order_not_found"
    assert "hint" in out


def test_dispatcher_rejects_unknown_tool():
    out = execute_tool("nuke_database", {})
    assert out["success"] is False
    assert out["error"] == "unknown_tool"


def test_dispatcher_raises_on_malformed_args():
    """Bad args raise ValidationError so the retry-with-correction loop
    can fire. The dispatcher does NOT swallow these."""
    with pytest.raises(ValidationError):
        execute_tool("lookup_order_status", {"order_id": 42})  # int, not str


def test_args_model_description_is_prompt_content():
    """The field description is read by the model on every call;
    keep it readable and specific."""
    schema = OrderLookupArgs.model_json_schema()
    desc = schema["properties"]["order_id"]["description"]
    assert "ORD-" in desc, "Description should hint at the format the model must produce"


# ---------------------------------------------------------------------------
# OpenAI-shaped stubs for end-to-end wiring test
# ---------------------------------------------------------------------------

class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id, function):
        self.id = id
        self.function = function


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, finish_reason, message):
        self.finish_reason = finish_reason
        self.message = message


class _FakeResponse:
    def __init__(self, choices):
        self.choices = choices


def test_lookup_endpoint_with_stubbed_model(monkeypatch):
    """Stub the OpenAI client to emit tool_calls → final text round-trip.
    Verifies the wiring without spending real tokens."""
    calls = {"n": 0}

    def fake_create(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse([
                _FakeChoice(
                    finish_reason="tool_calls",
                    message=_FakeMessage(
                        content=None,
                        tool_calls=[
                            _FakeToolCall(
                                id="t1",
                                function=_FakeFunction(
                                    name="lookup_order_status",
                                    arguments=json.dumps({"order_id": "ORD-1042"}),
                                ),
                            )
                        ],
                    ),
                )
            ])
        return _FakeResponse([
            _FakeChoice(
                finish_reason="stop",
                message=_FakeMessage(content="Your order shipped on Monday."),
            )
        ])

    with patch("app.llm._client") as mock_client:
        mock_client.return_value.chat.completions.create = fake_create
        r = client.post("/lookup", json={"message": "Where is ORD-1042?"})
        assert r.status_code == 200
        body = r.json()
        assert "shipped" in body["answer"]
        assert body["tool_call_count"] == 1
        assert body["retry_count"] == 0


def test_retry_with_correction_recovers(monkeypatch):
    """The correction loop must RECOVER, not just raise.

    Turn 1: the model sends order_id=42 - an int. Pydantic rejects it, and
    the loop feeds the validator's error back as a corrective tool message.
    Turn 2: the model sends "ORD-42" - a string. The tool runs.
    Turn 3: the model writes its answer.

    We assert the request still returns 200 and that retry_count is exactly
    1, because a correction loop that fires but never converges is just a
    slower way to fail.
    """
    calls = {"n": 0}

    def fake_create(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Malformed: order_id as an integer.
            return _FakeResponse([
                _FakeChoice(
                    finish_reason="tool_calls",
                    message=_FakeMessage(
                        content=None,
                        tool_calls=[
                            _FakeToolCall(
                                id="t1",
                                function=_FakeFunction(
                                    name="lookup_order_status",
                                    arguments=json.dumps({"order_id": 42}),
                                ),
                            )
                        ],
                    ),
                )
            ])
        if calls["n"] == 2:
            # Corrected: the model read the validator error and fixed the type.
            return _FakeResponse([
                _FakeChoice(
                    finish_reason="tool_calls",
                    message=_FakeMessage(
                        content=None,
                        tool_calls=[
                            _FakeToolCall(
                                id="t2",
                                function=_FakeFunction(
                                    name="lookup_order_status",
                                    arguments=json.dumps({"order_id": "ORD-42"}),
                                ),
                            )
                        ],
                    ),
                )
            ])
        return _FakeResponse([
            _FakeChoice(
                finish_reason="stop",
                message=_FakeMessage(content="Order ORD-42 has shipped."),
            )
        ])

    with patch("app.llm._client") as mock_client:
        mock_client.return_value.chat.completions.create = fake_create
        r = client.post("/lookup", json={"message": "Just check order 42 for me"})
        assert r.status_code == 200
        body = r.json()
        assert body["retry_count"] == 1        # the correction fired exactly once
        assert body["tool_call_count"] == 2    # the bad call and the good one
        assert "ORD-42" in body["answer"]
        assert r.headers["X-Retry-Count"] == "1"
