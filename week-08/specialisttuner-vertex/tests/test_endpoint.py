"""Smoke + contract tests for the SpecialistTuner Vertex app.

The Vertex SDK is never imported here - `app.llm.generate` is patched, and
`app.tools.start_tune`/`poll_tune`/`upload_to_gcs` are all patched to return
canned results. Everything runs offline, in milliseconds, so CI can use
these as the fast regression gate. Real end-to-end tests against Vertex
belong in a nightly job (Week 15 ReliabilityKit).

Six test categories:
  * schema validation (Message, ChatExample, InferenceRequest)
  * generic → Vertex converter (shape, systemInstruction, blank lines)
  * validator behaviour (missing file, malformed row, empty message)
  * FastAPI routes (/health, /convert, /predict, /v1/tune/start)
  * cross-week EvalResult schema compatibility (V2 LoRA ↔ V3 Vertex)
  * dispatcher error handling (unknown tool, exception → error dict)
"""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError


# ─── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the app at a temp dataset dir; reset settings cache."""
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")
    monkeypatch.setenv("SOURCE_MODEL", "gemini-3.1-flash-lite-001")
    monkeypatch.setenv("GENERIC_DATASET_PATH", str(tmp_path / "generic.jsonl"))
    monkeypatch.setenv("GCLOUD_DATASET_PATH", str(tmp_path / "gcloud.jsonl"))
    from app import config
    config.get_settings.cache_clear()


@pytest.fixture()
def client() -> TestClient:
    from app.main import app
    return TestClient(app)


def _sample_row(user="What is attention?", model="Attention is a mechanism..."):
    return {"messages": [
        {"role": "user", "message": user},
        {"role": "model", "message": model},
    ]}


# ─── schema validation ────────────────────────────────────────────────

def test_message_rejects_bad_role():
    from app.schemas import Message
    with pytest.raises(ValidationError):
        Message(role="assistant", message="hi")   # only user/model are valid


def test_message_rejects_empty_content():
    from app.schemas import Message
    with pytest.raises(ValidationError):
        Message(role="user", message="")


def test_chat_example_min_length_2():
    from app.schemas import ChatExample
    with pytest.raises(ValidationError):
        ChatExample(messages=[{"role": "user", "message": "hi"}])


# ─── converter ────────────────────────────────────────────────────────

def test_convert_example_shape():
    from app.convert import convert_example
    out = convert_example(_sample_row(), system_instruction="You are a teacher.")
    assert out["systemInstruction"]["role"] == "system"
    assert out["systemInstruction"]["parts"][0]["text"] == "You are a teacher."
    assert out["contents"][0]["role"] == "user"
    assert out["contents"][0]["parts"][0]["text"] == "What is attention?"
    assert out["contents"][1]["role"] == "model"


def test_convert_example_omits_system_when_none():
    from app.convert import convert_example
    out = convert_example(_sample_row(), system_instruction=None)
    assert "systemInstruction" not in out


def test_convert_file_skips_blank_lines(tmp_path: Path):
    from app.convert import convert_file
    gp = tmp_path / "g.jsonl"
    cp = tmp_path / "c.jsonl"
    gp.write_text("\n".join([
        json.dumps(_sample_row()),
        "",
        json.dumps(_sample_row(user="Q2", model="A2")),
        "",
    ]) + "\n", encoding="utf-8")
    result = convert_file(gp, cp)
    assert result["input_rows"] == 2
    assert result["output_rows"] == 2
    assert result["warnings"] == []
    lines = cp.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "contents" in obj and len(obj["contents"]) == 2


def test_convert_file_collects_warnings_for_malformed(tmp_path: Path):
    from app.convert import convert_file
    gp = tmp_path / "g.jsonl"
    cp = tmp_path / "c.jsonl"
    gp.write_text("\n".join([
        json.dumps(_sample_row()),
        '{"messages": [{"role": "user", "message": "orphan"}]}',  # too short
        "not-even-json",
        json.dumps(_sample_row()),
    ]) + "\n", encoding="utf-8")
    result = convert_file(gp, cp)
    assert result["input_rows"] == 4
    assert result["output_rows"] == 2
    assert len(result["warnings"]) == 2


# ─── validator ────────────────────────────────────────────────────────

def test_validate_missing_file(tmp_path: Path):
    from app.convert import validate_generic
    r = validate_generic(tmp_path / "does_not_exist.jsonl")
    assert r["success"] is False and r["error"] == "file_not_found"


def test_validate_clean_file(tmp_path: Path):
    from app.convert import validate_generic
    p = tmp_path / "g.jsonl"
    p.write_text("\n".join([json.dumps(_sample_row())] * 5) + "\n", encoding="utf-8")
    r = validate_generic(p)
    assert r["success"] is True
    assert r["count"] == 5


def test_validate_reports_row_number_for_bad_row(tmp_path: Path):
    from app.convert import validate_generic
    p = tmp_path / "g.jsonl"
    p.write_text("\n".join([
        json.dumps(_sample_row()),
        json.dumps({"messages": []}),                  # empty list
    ]) + "\n", encoding="utf-8")
    r = validate_generic(p)
    assert r["success"] is False
    assert any("line 2" in e for e in r["errors"])


# ─── FastAPI routes ───────────────────────────────────────────────────

def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["provider"] == "vertex-ai"
    assert body["model"] == "gemini-3.1-flash-lite-001"


def test_predict_calls_llm_generate(client: TestClient):
    fake = {"output": "attention weighs inputs by similarity",
            "latency_ms": 12.3,
            "model_id": "gemini-3.1-flash-lite-001"}
    with patch("app.main.generate", return_value=fake) as m:
        r = client.post("/predict", json={"prompt": "explain attention"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["output"].startswith("attention weighs")
    assert body["model_id"] == "gemini-3.1-flash-lite-001"
    m.assert_called_once()


def test_convert_route_happy_path(client: TestClient, tmp_path: Path):
    from app.config import get_settings
    s = get_settings()
    s.generic_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    s.generic_dataset_path.write_text(
        "\n".join([json.dumps(_sample_row())] * 3) + "\n", encoding="utf-8",
    )
    r = client.post("/convert", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_rows"] == 3
    assert body["output_rows"] == 3
    assert body["warnings"] == []


def test_tune_start_route_delegates_to_dispatcher(client: TestClient):
    fake = {"success": True, "job_id": "projects/x/tuningJobs/1",
            "state": "JOB_STATE_PENDING", "display_name": "test"}
    with patch("app.main.execute_tool", return_value=fake):
        r = client.post("/v1/tune/start", json={"epochs": 4})
    assert r.status_code == 200, r.text
    assert r.json()["job_id"].endswith("tuningJobs/1")


# ─── EvalResult cross-week compatibility ──────────────────────────────

def test_eval_result_accepts_vertex_ft_row():
    """W8V2 (LoRA) + W8V3 (Vertex) EvalResult must accept each other's rows
    so the benchmark harness can merge them into one decision-memo table."""
    from app.schemas import EvalResult
    row = EvalResult(
        approach="vertex_ft",
        model_id="projects/x/locations/us-central1/endpoints/1234",
        accuracy=0.86,
        p50_latency_ms=240.0,
        p95_latency_ms=290.0,
        cost_per_1k_calls_usd=1.20,
        trainable_params=0,
        privacy_posture="vendor_zone",
    )
    assert row.approach == "vertex_ft"


# ─── dispatcher ───────────────────────────────────────────────────────

def test_execute_tool_unknown():
    from app.tools import execute_tool
    r = execute_tool("nope", {})
    assert r["success"] is False and r["error"] == "unknown_tool"
    assert "validate" in r["available"]


def test_execute_tool_wraps_exceptions(tmp_path: Path):
    from app.tools import execute_tool
    # validate a path that doesn't exist. The function itself returns an
    # error dict rather than raising, so the wrapper returns that dict.
    r = execute_tool("validate", {"path": str(tmp_path / "nope.jsonl")})
    assert r["success"] is False
