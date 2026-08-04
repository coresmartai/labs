"""
Smoke tests - do not call the real OpenAI API or SMTP server.

These tests verify wiring (FastAPI routes, schema validation, env loading)
without spending tokens. For real integration tests, see the lab repo.
"""

import os
import pytest
from fastapi.testclient import TestClient


# Make sure required env vars exist so Settings can load during import.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("SMTP_SENDER", "test@gmail.com")
os.environ.setdefault("SMTP_PASSWORD", "test-app-password")

from app.main import app  # noqa: E402
from app.schemas import ReleaseSummary  # noqa: E402

client = TestClient(app)


def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_summarize_rejects_empty_notes():
    r = client.post("/summarize", json={"release_notes": ""})
    # Pydantic enforces min_length=1 → 422
    assert r.status_code == 422


def test_release_summary_schema_round_trip():
    payload = {
        "headline": "v2.4 - auth fixes",
        "bullets": ["Fixes login retry loop", "Improves session timing"],
        "risk_level": "low",
    }
    s = ReleaseSummary.model_validate(payload)
    assert s.risk_level == "low"
    assert len(s.bullets) == 2


def test_release_summary_rejects_bad_risk_level():
    with pytest.raises(Exception):
        ReleaseSummary.model_validate(
            {
                "headline": "x",
                "bullets": ["a", "b"],
                "risk_level": "extreme",  # not in Literal
            }
        )
