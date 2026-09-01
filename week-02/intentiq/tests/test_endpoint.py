"""Smoke tests - no real API calls.

These verify the structure: routes register, schemas parse, normalisation works,
and the aggregator handles small inputs without crashing. Integration tests against
real providers belong in the guided lab, where students wire real keys.
"""
from __future__ import annotations
from fastapi.testclient import TestClient

from app.eval import summarise
from app.llm import normalise_label
from app.main import app
from app.schemas import BenchmarkRow, GoldenExample, Result


client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_normalise_label_known_intent() -> None:
    assert normalise_label("Cancel_Subscription") == "cancel_subscription"
    assert normalise_label("  CANCEL_SUBSCRIPTION  ") == "cancel_subscription"
    assert normalise_label("cancel subscription") == "cancel_subscription"


def test_normalise_label_unknown_intent_falls_back() -> None:
    assert normalise_label("buy_pizza") == "unknown"
    assert normalise_label("") == "unknown"


def test_result_schema_validation() -> None:
    r = Result(provider="openai", label="cancel_subscription", confidence=0.9, latency_ms=540.0)
    assert r.provider == "openai"
    assert 0.0 <= r.confidence <= 1.0


def test_summarise_handles_small_rows() -> None:
    rows = [
        BenchmarkRow(example_id="1", provider="openai", predicted="cancel_subscription",
                     expected="cancel_subscription", correct=True, confidence=0.9,
                     latency_ms=500.0, input_tokens=20, output_tokens=10),
        BenchmarkRow(example_id="2", provider="openai", predicted="unknown",
                     expected="cancel_subscription", correct=False, confidence=0.5,
                     latency_ms=620.0, input_tokens=22, output_tokens=8),
    ]
    summaries = summarise(rows)
    assert len(summaries) == 1
    assert summaries[0].provider == "openai"
    assert summaries[0].accuracy == 0.5
    assert summaries[0].n == 2


def test_golden_example_parses() -> None:
    ex = GoldenExample(id="001", input="can you cancel my plan", expected="cancel_subscription")
    assert ex.expected == "cancel_subscription"
