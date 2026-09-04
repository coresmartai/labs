"""Smoke tests - no real model calls.

These run in CI without an OPENAI_API_KEY. They verify wiring, schema
contracts, and the compressor's marker-preservation invariant - the
exact invariant that Failure #3 in the lab proves matters.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.compressor import compress_chunk
from app.main import app
from app.schemas import Chunk

client = TestClient(app)


def test_health_returns_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "ragoptimizer"
    # /health also returns model keys for the UI health chip
    assert "model" in body, "health must include 'model' key for the UI chip"
    assert "models" in body, "health must include 'models' dict for multi-model display"


def test_config_snapshot_surfaces_pinned_model() -> None:
    r = client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"].startswith("gpt-"), "model must be pinned, not 'latest'"
    assert "hyde_enabled" in body
    assert "reranker_model" in body


def test_compressor_preserves_marker_line() -> None:
    """Failure #3: the compressor must never strip the chunk-ID marker.

    The invariant: marker line (first line of chunk.text) survives compression
    regardless of how it scores against the query.
    """
    chunk = Chunk(
        chunk_id="runbook-jwt-rotation-042",
        text=(
            "[runbook-jwt-rotation-042]\n"
            "An apple a day keeps the doctor away. "
            "The JWT signing key lives in the secrets manager. "
            "Cats are good pets. "
            "Cutover happens during the overlap window."
        ),
        source="runbook.md",
    )
    compressed = compress_chunk(chunk, query="how do I rotate the JWT signing key")
    assert compressed.marker_line == "[runbook-jwt-rotation-042]", \
        "Marker line was stripped or modified - citation contract broken."
    assert "secrets manager" in compressed.text, "Relevant sentence dropped"


def test_compressor_drops_irrelevant_sentences() -> None:
    chunk = Chunk(
        chunk_id="c1",
        text=(
            "[c1]\n"
            "Pineapples are tropical. "
            "Rotate JWT keys via the secrets manager. "
            "The weather is nice today."
        ),
    )
    compressed = compress_chunk(chunk, query="rotate JWT keys secrets manager")
    # The relevant sentence must survive; at least one irrelevant one is dropped.
    assert "secrets manager" in compressed.text
    assert len(compressed.text) < len(chunk.text)


def test_config_surfaces_reranker_toggle():
    """The reranker toggle must be visible in /config.

    Lab Part C has learners switch the reranker off and re-measure. If /config
    cannot report the flag, the one instrument the lab names for confirming
    pipeline state is blind to the one thing the learner changed.
    """
    r = client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "reranker_enabled" in body
    assert body["reranker_enabled"] is True
