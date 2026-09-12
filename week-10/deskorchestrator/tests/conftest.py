"""Fixtures. No API key, no network, no server."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm  # noqa: E402
from app.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-real-credential")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _memory():
    """A clean store and a clean prefs dict for every test."""
    from app import memory
    from app.vectorstore import reset_store
    reset_store()
    memory._PREFS.clear()
    memory.seed_memory()
    memory.seed_demo_prefs()
    yield
    reset_store()
    memory._PREFS.clear()


def _specialist_json(tool_name="grant_access", args=None):
    return json.dumps({
        "summary": "Write access to a production database is never self-service [doc:0].",
        "tool_name": tool_name,
        "arguments": args if args is not None else {"system": "billing-db", "level": "write"},
        "rationale": "The request needs a bounded grant with manager approval.",
        "evidence": ["0"],
        "alternative": "Read-only warehouse access, which is self-service.",
    })


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace the provider seam. Returns the route, then a specialist JSON body."""
    calls = {"n": 0}

    def fake_call(model, system, user, max_tokens=None):
        calls["n"] += 1
        text = "access" if max_tokens == 20 else _specialist_json()
        return llm.LLMResult(text=text, model=model, input_tokens=100, output_tokens=20)

    monkeypatch.setattr(llm, "call", fake_call)
    return calls


@pytest.fixture
def routed_llm(monkeypatch):
    """Parameterisable stub: set `.route` and `.tool` before invoking."""
    box = {"route": "access", "tool": "grant_access", "args": None}

    def fake_call(model, system, user, max_tokens=None):
        text = box["route"] if max_tokens == 20 else _specialist_json(box["tool"], box["args"])
        return llm.LLMResult(text=text, model=model, input_tokens=100, output_tokens=20)

    monkeypatch.setattr(llm, "call", fake_call)
    return box
