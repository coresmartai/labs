"""Smoke tests for CohortMCP.

Offline, deterministic, runs in seconds. No real OpenAI calls - `pick_tool`
is monkeypatched wherever it would otherwise fire. What is covered:
  - /health reports the pinned model + current tool-description-quality mode.
  - /demo/tools lists both tools.
  - /demo/call: cohort_lookup and incident_history return the expected shapes.
  - /demo/call: unknown tool + bad arguments return the structured error envelope.
  - Tool descriptions actually change when TOOL_DESCRIPTION_QUALITY flips.
  - /eval/run aggregates a stubbed golden-set run into an accuracy summary.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"].startswith("gpt-")
    assert body["tool_description_quality"] == "good"
    assert body["mcp_transport"] == "stdio"


def test_mcp_tools_list_returns_both_tools():
    r = client.get("/demo/tools")
    assert r.status_code == 200
    rows = r.json()
    names = {t["name"] for t in rows}
    assert names == {"cohort_lookup", "incident_history"}
    assert all({"name", "description", "inputSchema"}.issubset(t.keys()) for t in rows)


def test_mcp_call_cohort_lookup_known_and_unknown():
    r = client.post("/demo/call", json={"name": "cohort_lookup", "arguments": {"student_id": "stu_001"}})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["result"]["found"] is True
    assert body["result"]["profile"]["name"] == "Alice"

    # A miss is a normal result, not an error: HTTP 200, success true, found false.
    r_unknown = client.post("/demo/call", json={"name": "cohort_lookup", "arguments": {"student_id": "stu_999"}})
    assert r_unknown.status_code == 200
    assert r_unknown.json()["success"] is True
    assert r_unknown.json()["result"] == {"found": False}


def test_mcp_call_incident_history_returns_structured_payload():
    r = client.post("/demo/call", json={"name": "incident_history", "arguments": {"query": "payments"}})
    body = r.json()
    assert body["success"] is True
    assert "results" in body["result"]
    assert "total_available" in body["result"]
    assert all({"id", "severity", "summary"}.issubset(row.keys()) for row in body["result"]["results"])

    r_empty = client.post("/demo/call", json={"name": "incident_history", "arguments": {"query": "xyznonexistent"}})
    assert r_empty.json()["result"]["results"] == []
    assert r_empty.json()["result"]["total_available"] == 0


def test_mcp_call_unknown_tool_returns_structured_error_envelope():
    r = client.post("/demo/call", json={"name": "nonexistent_tool", "arguments": {}})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error"]["code"] == "unknown_tool"
    assert body["error"]["retryable"] is False
    assert body["trace_id"]


def test_mcp_call_bad_arguments_returns_structured_error():
    # incident_history requires `query` - omit it to trigger FastMCP's own
    # argument validation (a pydantic ValidationError chained onto ToolError).
    r = client.post("/demo/call", json={"name": "incident_history", "arguments": {}})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error"]["code"] == "bad_arguments"
    assert body["error"]["retryable"] is False


def test_tool_description_quality_is_a_pure_function_of_quality():
    from app.tools import tool_description

    for name in ("cohort_lookup", "incident_history"):
        bad = tool_description(name, "bad")
        good = tool_description(name, "good")
        assert bad != good
        assert len(good) > len(bad)


def test_eval_run_uses_stubbed_llm(monkeypatch):
    """No real network call - pick_tool is monkeypatched to a deterministic stub."""
    from app import eval as eval_module
    from app.schemas import ToolPickResponse

    async def _fake_pick_tool(request: str) -> ToolPickResponse:
        tool = "incident_history" if "incident" in request.lower() else "cohort_lookup"
        return ToolPickResponse(tool=tool, arguments={}, latency_ms=1)

    monkeypatch.setattr(eval_module, "pick_tool", _fake_pick_tool)

    r = client.get("/eval/run")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 8
    assert 0.0 <= body["accuracy"] <= 1.0
    assert len(body["results"]) == body["total"]
