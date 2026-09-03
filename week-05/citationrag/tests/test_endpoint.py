"""Smoke tests for CitationRAG endpoints.

These tests do not make real API calls and do not need a running Qdrant instance.
Both the LLM wrapper and the retriever are monkeypatched so the suite is fast,
free, and deterministic.

Retriever mock: we patch app.main._get_retriever to return a FakeRetriever
that returns fixed chunks without touching Qdrant or the embedding API.
Patching the factory function keeps the test independent of how the Retriever
class is constructed internally.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class _FakeRetriever:
    """Returns fixed chunks without any Qdrant or embedding calls."""

    def search(self, query: str):
        from app.schemas import Chunk

        chunks = [
            Chunk(
                chunk_id="doc#1",
                text="The service returns 504 when upstream timeout exceeds 30 seconds.",
                score=0.88,
            ),
            Chunk(
                chunk_id="doc#2",
                text="Restart the worker pool with supervisorctl restart api.",
                score=0.82,
            ),
            Chunk(
                chunk_id="doc#3",
                text="Raise upstream_timeout in prod.yaml to 60 seconds for slow backends.",
                score=0.75,
            ),
        ]
        return chunks, 0.88, 0.13   # chunks, top1_dense, spread


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """Spin up the FastAPI app with LLM and retriever both monkeypatched."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "knowledgevault")

    # Reset cached settings + retriever between tests
    from app import config, main
    config.get_settings.cache_clear()
    main._retriever = None

    from app.schemas import AnswerResponse, Citation

    def fake_generate(question, chunks):
        return AnswerResponse(
            answer="Restart the worker pool [doc#2].",
            citations=[
                Citation(chunk_id="doc#2", supporting_quote="Restart the worker pool"),
            ],
            refused=False,
        )

    # Patch retriever factory - no Qdrant connection needed
    fake_retriever = _FakeRetriever()
    monkeypatch.setattr("app.main._get_retriever", lambda: fake_retriever)
    monkeypatch.setattr("app.main.generate_with_citations", fake_generate)

    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_answer_grounded(client):
    r = client.post("/answer", json={"question": "How do I restart the worker pool?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] in (False, True)
    if not body["refused"]:
        assert body["validation_passed"] is True


def test_answer_refusal_below_threshold(client, monkeypatch):
    """Force the threshold gate to fire by stubbing the retriever low-score path."""
    from app.schemas import Chunk

    class _LowScoreRetriever:
        def search(self, q):
            chunks = [Chunk(chunk_id="doc#1", text="x", score=0.1)]
            return chunks, 0.1, 0.0

    monkeypatch.setattr("app.main._get_retriever", lambda: _LowScoreRetriever())
    r = client.post("/answer", json={"question": "An entirely unrelated query."})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is True
    assert body["answer"] == "I don't have that information in the provided sources."
    assert body["citations"] == []


def test_answer_fabricated_citation_caught(client, monkeypatch):
    """Failure #1 - model invents a chunk ID. Validator must catch it."""
    from app.schemas import AnswerResponse, Citation

    def fab_generate(question, chunks):
        return AnswerResponse(
            answer="Some answer [doc#99].",
            citations=[Citation(chunk_id="doc#99", supporting_quote="nope")],
            refused=False,
        )

    monkeypatch.setattr("app.main.generate_with_citations", fab_generate)
    r = client.post("/answer", json={"question": "How do I restart the worker pool?"})
    body = r.json()
    # Fabricated citation should trigger the conservative refusal path
    assert body["refused"] is True


def test_spread_dedupe_collapses_sibling_chunks():
    """Overlapping sibling chunks must not fake a narrow spread (false refusal).

    Two near-identical texts (50-token chunk overlap in KnowledgeVault) score
    almost the same; spread over the RAW list would be tiny even though the
    index covers the query perfectly. _dedupe_scores collapses them so spread
    measures distance to the best DISTINCT passage.
    """
    from app.retriever import _dedupe_scores

    base = "scaled dot product attention computes a weighted sum of the values "
    hits = [
        (0.90, base + "using softmax over query key dot products"),
        (0.89, base + "using softmax over the query key dot products divided"),  # sibling
        (0.60, "the decoder stack uses masked self attention to preserve autoregression"),
        (0.55, "positional encodings inject order information into the embeddings"),
    ]
    deduped = _dedupe_scores(hits)
    # The 0.89 sibling collapses into the 0.90 hit
    assert deduped[0] == 0.90
    assert 0.89 not in deduped
    # Spread over deduped top1-top3 is wide (0.90 - 0.55), not 0.90 - 0.60 raw top3
    assert len(deduped) == 3


def test_judge_row_citation_validity_semantics():
    """cit_valid_ok: True = generated + survived, False = validator rejected,
    None = never generated. The old always-True precision tautology is gone."""
    from app.main import _judge_row
    from app.schemas import AnswerResponse, Citation, GoldenRow

    row = GoldenRow(question="q?", expected_answer="weighted values", must_cite=["_answered"])

    answered = AnswerResponse(
        answer="A weighted sum of the values [doc#1].",
        citations=[Citation(chunk_id="doc#1", supporting_quote="weighted")],
    )
    assert _judge_row(row, answered)["cit_valid_ok"] is True

    validation_refused = AnswerResponse(
        answer="I don't have that information in the provided sources.",
        citations=[], refused=True, refusal_source="validation",
    )
    assert _judge_row(row, validation_refused)["cit_valid_ok"] is False

    gated = AnswerResponse(
        answer="I don't have that information in the provided sources.",
        citations=[], refused=True, refusal_source="threshold_gate",
    )
    assert _judge_row(row, gated)["cit_valid_ok"] is None
