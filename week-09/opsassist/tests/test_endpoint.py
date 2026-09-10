"""Smoke tests - no real API calls.

Twelve tests covering the six invariant families
(dispatcher contract, stop conditions, id binding, idempotency, resume,
state boundaries):
 1. App boots.
 2. Tool dispatcher routes correctly.
 3. Unknown tool returns an error envelope (never raises).
 4. Stop conditions fire (mock the LLM to spin).
 5. tool_call_id binding invariant holds.
 6. propose_remediation is idempotent via the key.
 7. Wall-clock timeout stop fires.
 8. Checkpoint resume restores the sub-step, tokens and digest.
 9. The loop makes ZERO persistent reads (V3's three sanctioned crossings).
10. The checkpoint is a recovery hint, not a full snapshot.
11. `auth` normalises to `auth-service`.
12. The dollar-cost stop fires when MAX_COST_USD is armed.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.tools import execute_tool, TOOLS, propose_remediation, _REMEDIATION_CACHE
from app.schemas import ProposeRemediationInput, QueryMetricsInput
from app.llm import ModelResponse


def _spin_then_stop(after: int):
    """A stub model: `after` tool calls, then a final text turn."""
    issued: list[str] = []

    def fake_call(**kw):
        if len(issued) >= after:
            return ModelResponse(
                content_blocks=[{"type": "text", "text": "done"}],
                stop_reason="end_turn", input_tokens=5, output_tokens=5,
            )
        new_id = f"call_{len(issued):03d}"
        issued.append(new_id)
        return ModelResponse(
            content_blocks=[{
                "type": "tool_call", "id": new_id, "name": "query_metrics",
                "input": {"service": "api", "window": "1h"},
            }],
            stop_reason="tool_calls", input_tokens=5, output_tokens=5,
        )

    return fake_call, issued


client = TestClient(app)


# ---- 1. App boots ----
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---- 2. Tool dispatcher routes correctly ----
def test_dispatcher_routes_known_tool():
    out = execute_tool("query_metrics", {"service": "api", "window": "1h"})
    assert "p95_latency_ms" in out
    assert out["service"] == "api"


# ---- 3. Unknown tool -> error envelope, not an exception ----
def test_dispatcher_returns_error_envelope_on_unknown_tool():
    result = execute_tool("does_not_exist", {})
    assert result["success"] is False
    assert result["error"] == "unknown_tool"


# ---- 4. Stop conditions fire (LLM mocked to always emit tool calls) ----
def test_max_iters_stop_fires(monkeypatch):
    """Mock the LLM to always ask for a tool call. Loop must terminate at max_iters."""
    from app import agent as agent_module

    fake_response = ModelResponse(
        content_blocks=[{
            "type": "tool_call",
            "id": "call_abc",
            "name": "query_metrics",
            "input": {"service": "api", "window": "1h"},
        }],
        stop_reason="tool_calls",
        input_tokens=10,
        output_tokens=10,
    )
    monkeypatch.setattr(agent_module, "call_with_tools", lambda **kw: fake_response)

    r = client.post("/agent/run", json={"user_input": "spin forever", "user_id": "test"})
    assert r.status_code == 200
    body = r.json()
    # Either max_iters (with default cap of 10) or token_budget should have fired.
    assert body["stop_reason"] in ("max_iters", "token_budget")
    assert body["iter_count"] >= 1


# ---- 5. tool_call_id binding invariant ----
def test_every_tool_result_has_matching_tool_call_id(monkeypatch):
    """Every tool result in the trace must reference a tool_call_id that was emitted."""
    from app import agent as agent_module

    issued_ids: list[str] = []

    def fake_call(**kw):
        if len(issued_ids) >= 2:
            return ModelResponse(
                content_blocks=[{"type": "text", "text": "done"}],
                stop_reason="end_turn", input_tokens=5, output_tokens=5,
            )
        new_id = f"call_{len(issued_ids):03d}"
        issued_ids.append(new_id)
        return ModelResponse(
            content_blocks=[{
                "type": "tool_call", "id": new_id,
                "name": "query_metrics",
                "input": {"service": "api", "window": "1h"},
            }],
            stop_reason="tool_calls", input_tokens=5, output_tokens=5,
        )

    monkeypatch.setattr(agent_module, "call_with_tools", fake_call)
    r = client.post("/agent/run", json={"user_input": "diagnose", "user_id": "test"})
    body = r.json()
    for it in body["trace"]:
        for res in it["tool_results"]:
            assert res["tool_call_id"] in issued_ids


# ---- 6. Idempotency on propose_remediation ----
def test_propose_remediation_is_idempotent():
    _REMEDIATION_CACHE.clear()
    args = ProposeRemediationInput(
        issue="latency-spike", severity="high", idempotency_key="abc-123",
    )
    first = propose_remediation(args)
    second = propose_remediation(args)
    assert first.was_duplicate is False
    assert second.was_duplicate is True
    assert first.incident_id == second.incident_id


# ---- 7. Wall-clock timeout stop ----
def test_timeout_stop_fires(monkeypatch):
    """With a zero-second wall-clock budget the loop must stop before iter 1."""
    monkeypatch.setenv("MAX_WALL_CLOCK_SECONDS", "0")
    get_settings.cache_clear()

    r = client.post("/agent/run", json={"user_input": "anything", "user_id": "test"})
    assert r.status_code == 200
    body = r.json()
    assert body["stop_reason"] == "timeout"
    assert body["iter_count"] == 0


# ---- 8. Checkpoint resume restores state ----
def test_checkpoint_resume_restores_state(monkeypatch):
    """A known task_id with a saved recovery hint resumes mid-run."""
    from app import agent as agent_module
    from app.state import WorkflowState

    task_id = "resume-test-001"
    WorkflowState(task_id=task_id).write_checkpoint({
        "task_id": task_id,
        "iter": 2,
        "awaiting_input": False,
        "last_tool_call_id": "call_a1b2",
        "summary": "Task so far: earlier turn | tools already run: query_metrics",
        "transcript_ref": f"audit:test:{task_id}",
        "tokens_used": 3000,
        "ts": 0.0,
    })

    captured: dict = {}

    def fake_call(**kw):
        captured["messages"] = list(kw["messages"])
        return ModelResponse(
            content_blocks=[{"type": "text", "text": "resumed"}],
            stop_reason="end_turn", input_tokens=5, output_tokens=5,
        )

    monkeypatch.setattr(agent_module, "call_with_tools", fake_call)
    r = client.post("/agent/run", json={
        "user_input": "continue", "task_id": task_id, "user_id": "test",
    })
    body = r.json()
    assert body["stop_reason"] == "end_turn"
    assert body["iter_count"] == 2                     # restored sub-step
    assert body["trace"][0]["iter"] == 2               # final turn at resumed index
    # The hint - not a stored message list - seeds the resumed conversation.
    assert "earlier turn" in captured["messages"][0]["content"]


# ---- 9. Zero persistent reads inside the loop ----
def test_zero_persistent_reads_inside_the_loop(monkeypatch):
    """w09v03: 'exactly three sanctioned crossings'. The loop makes none."""
    from app import agent as agent_module
    from app.state import PersistentState

    reads: list[str] = []
    real = PersistentState.load_session_context
    monkeypatch.setattr(
        PersistentState, "load_session_context",
        lambda self, uid: (reads.append(uid), real(self, uid))[1],
    )
    monkeypatch.setattr(
        PersistentState, "check_consent",
        lambda self, uid: pytest.fail("consent read from inside the agent loop"),
    )

    fake_call, _ = _spin_then_stop(3)
    monkeypatch.setattr(agent_module, "call_with_tools", fake_call)

    r = client.post("/agent/run", json={"user_input": "diagnose", "user_id": "test"})
    assert r.json()["iter_count"] == 3
    assert len(reads) == 1          # one read, at session start. That is all.


# ---- 10. The checkpoint is a recovery hint, not a snapshot ----
def test_checkpoint_is_a_recovery_hint_not_a_snapshot(monkeypatch):
    """w09v03: 'the checkpoint is a recovery hint, not a full snapshot'."""
    from app import agent as agent_module
    from app.state import WorkflowState

    # Never reach end_turn: a graceful finish clears the workflow key, and we
    # want to read the checkpoint the last successful tool call left behind.
    fake_call, issued = _spin_then_stop(99)
    monkeypatch.setattr(agent_module, "call_with_tools", fake_call)

    task_id = "hint-test-001"
    r = client.post("/agent/run", json={
        "user_input": "diagnose", "task_id": task_id, "user_id": "test",
    })
    assert r.status_code == 200
    assert r.json()["stop_reason"] in ("max_iters", "token_budget")

    cp = WorkflowState(task_id=task_id).read_checkpoint()
    assert cp is not None
    assert "messages" not in cp                      # the whole point
    assert {"task_id", "iter", "awaiting_input", "last_tool_call_id",
            "summary", "transcript_ref"} <= set(cp)
    assert cp["last_tool_call_id"] == issued[-1]     # written per tool call
    assert "query_metrics" in cp["summary"]


# ---- 11. `auth` is an alias for `auth-service` ----
def test_auth_alias_normalises_to_auth_service():
    """w09v01: the tool is 'restricted to api, worker, db, or auth'."""
    out = execute_tool("query_metrics", {"service": "auth", "window": "1h"})
    canonical = execute_tool("query_metrics", {"service": "auth-service", "window": "1h"})
    assert out["service"] == "auth-service"
    assert out == canonical


# ---- 12. The dollar-cost stop ----
def test_cost_budget_stop_fires(monkeypatch):
    """w09v01: 'a token or dollar budget hits its ceiling'. Off by default."""
    from app import agent as agent_module

    assert get_settings().max_cost_usd is None       # disabled by default

    monkeypatch.setenv("MAX_COST_USD", "0.0000001")  # arm it, absurdly low
    get_settings.cache_clear()

    fake_call, _ = _spin_then_stop(5)
    monkeypatch.setattr(agent_module, "call_with_tools", fake_call)

    r = client.post("/agent/run", json={"user_input": "spend", "user_id": "test"})
    body = r.json()
    assert body["stop_reason"] == "cost_budget"
    assert body["iter_count"] == 1                   # stopped after the first spend
