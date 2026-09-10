"""Smoke tests for the SpecialistTuner LoRA inference service.

These tests never call the real model or the real HF Hub. They verify wiring,
schema validation, and request shape so a CI run can fail fast without GPUs.
"""
from __future__ import annotations
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import LORA_TRAINABLE_PARAMS, app
from app.schemas import EvalResult, InferenceRequest, ChatMessage


client = TestClient(app)


def test_health_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_validates_empty_messages() -> None:
    r = client.post("/v1/chat/completions", json={"messages": []})
    # Pydantic catches min_length on schemas - 422 from FastAPI
    assert r.status_code in (400, 422)


def test_chat_happy_path_with_mocked_generate() -> None:
    fake = {"output": "ack", "latency_ms": 12.3}
    payload = InferenceRequest(messages=[
        ChatMessage(role="user", content="hello")
    ]).model_dump()
    with patch("app.main.generate", return_value=fake) as m:
        r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["output"] == "ack"
    assert body["latency_ms"] == 12.3
    m.assert_called_once()


def test_eval_result_schema_shape() -> None:
    row = EvalResult(
        approach="lora",
        model_id="Qwen/Qwen3-0.6B+lora",
        accuracy=0.84,
        p50_latency_ms=180.0,
        p95_latency_ms=220.0,
        cost_per_1k_calls_usd=0.30,
        trainable_params=2_293_760,
        privacy_posture="in_perimeter",
    )
    assert row.approach == "lora"
    assert row.privacy_posture == "in_perimeter"




def test_eval_run_emits_one_memo_row() -> None:
    """Every eval run produces exactly one `EvalResult` - the memo's unit."""
    fake = {"output": "self-attention", "latency_ms": 120.0}
    with patch("app.main.generate", return_value=fake):
        r = client.post("/v1/eval/run",
                        json={"adapter_path": "out/checkpoint-82"})
    assert r.status_code == 200, r.text
    row = EvalResult(**r.json()["row"])          # re-validates against the schema
    assert row.approach == "lora"
    assert row.trainable_params == LORA_TRAINABLE_PARAMS == 2_293_760
    assert row.privacy_posture == "in_perimeter"

    with patch("app.main.generate", return_value=fake):
        r = client.post("/v1/eval/run", json={"adapter_path": None})
    base = EvalResult(**r.json()["row"])
    assert base.approach == "base"
    assert base.trainable_params == 0
