"""
Smoke tests for KnowledgeVault. Intentionally no real API calls - these
verify schemas, route shape, and validation. Integration tests live in
the guided lab repo.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone

import pytest


# Make import paths resolve before touching pydantic-settings
os.environ.setdefault("OPENAI_API_KEY", "test-openai")


def test_health_endpoint_returns_ok():
    """GET /health returns 200 with model names."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "vision_model" in body
    assert "embed_model" in body


def test_retrieve_request_validates_k_default():
    """RetrieveRequest defaults k=5 and accepts query."""
    from app.schemas import RetrieveRequest

    req = RetrieveRequest(query="hello")
    assert req.query == "hello"
    assert req.k == 5
    assert req.document_id is None


def test_retrieve_request_rejects_empty_query():
    """RetrieveRequest with non-string query should fail validation."""
    from pydantic import ValidationError
    from app.schemas import RetrieveRequest

    with pytest.raises(ValidationError):
        RetrieveRequest(query=None)  # type: ignore[arg-type]


def test_figure_description_validates_confidence_enum():
    """FigureDescription must have confidence in {high, medium, low}."""
    from pydantic import ValidationError
    from app.schemas import FigureDescription

    ok = FigureDescription(
        type="architecture diagram",
        summary="A three-tier system with a load balancer.",
        key_elements=["LB", "primary", "secondary"],
        confidence="high",
    )
    assert ok.confidence == "high"

    with pytest.raises(ValidationError):
        FigureDescription(
            type="x",
            summary="y",
            key_elements=[],
            confidence="banana",  # type: ignore[arg-type]
        )


def test_chunk_schema_roundtrip():
    """Chunk model accepts all required fields and round-trips JSON."""
    from app.schemas import Chunk

    c = Chunk(
        chunk_id="c_1",
        parent_id="p_1",
        document_id="runbook",
        source_url="wiki/runbook.pdf",
        section_heading="Failover",
        page_number=7,
        chunk_index=12,
        created_at=datetime.now(timezone.utc),
        chunk_type="prose",
        text="Regional failover engages when…",
        figure_ids=["f_03"],
    )
    j = c.model_dump_json()
    c2 = Chunk.model_validate_json(j)
    assert c2.chunk_id == "c_1"
    assert c2.figure_ids == ["f_03"]


def test_tool_dispatch_unknown_tool():
    """execute_tool returns an error for unknown tool names - no exceptions."""
    from app.tools import execute_tool

    result = execute_tool("not_a_tool", {})
    assert result["success"] is False
    assert result["error"] == "unknown_tool"


def test_tool_dispatch_known_tool():
    """execute_tool echoes back the recorded args for the figure tool."""
    from app.tools import execute_tool

    args = {
        "type": "chart",
        "summary": "A latency chart over time.",
        "key_elements": ["p95", "p99"],
        "confidence": "medium",
    }
    result = execute_tool("record_figure_description", args)
    assert result["success"] is True
    assert result["recorded"]["type"] == "chart"


# ---------------------------------------------------------------------------
# Hierarchy: parents span blocks, and widening reassembles them.
# These run offline. tiktoken would download its BPE table on first use, so
# the encoder is replaced with a one-token-per-word stub for the test.
# ---------------------------------------------------------------------------

class _WordEncoder:
    """One token per whitespace-separated word. Enough to test window maths."""

    def __init__(self) -> None:
        self._v: dict[str, int] = {}
        self._r: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        out = []
        for w in text.split(" "):
            if w not in self._v:
                self._v[w] = len(self._v)
                self._r[self._v[w]] = w
            out.append(self._v[w])
        return out

    def decode(self, toks: list[int]) -> str:
        return " ".join(self._r[t] for t in toks)


@pytest.fixture
def word_encoder(monkeypatch):
    import app.chunker as chunker
    enc = _WordEncoder()
    monkeypatch.setattr(chunker, "_ENC", enc)
    return enc


def test_parents_span_blocks_and_hold_six_children(word_encoder):
    """A 1,500-token parent crosses paragraph and page boundaries and holds
    six 300-token children at a 250-token step. This is the hierarchy the
    retriever widens over; if parents were per paragraph, most would hold one
    child and widening would return nothing wider than the match."""
    from app.chunker import chunk_document
    from app.schemas import ParsedBlock

    blocks = [
        ParsedBlock(
            document_id="doc", source_url="doc.pdf", page_number=page,
            section_heading=f"Section {page}", block_type="text",
            bbox=(0, 0, 1, 1), text=" ".join(f"w{page}_{i}" for i in range(120)),
        )
        for page in range(1, 8) for _ in range(4)      # 28 blocks, 3,360 tokens
    ]
    prose = [c for c in chunk_document(blocks, {}) if c.chunk_type == "prose"]
    by_parent: dict[str, list] = {}
    for c in prose:
        by_parent.setdefault(c.parent_id, []).append(c)

    sizes = sorted((len(v) for v in by_parent.values()), reverse=True)
    assert sizes[:2] == [6, 6], sizes                 # full parents hold six
    assert any(len({c.page_number for c in v}) > 1 for v in by_parent.values())
    assert prose[0].page_number == 1 and prose[-1].page_number == 7


def test_widen_stitch_removes_overlap(word_encoder):
    """Stitching a parent's children back together drops the 50-token overlap
    so the parent reads as one continuous span, not six windows with seams."""
    from app.chunker import _split_stream
    from app.retriever import _stitch

    text = " ".join(f"w{i}" for i in range(700))
    windows = _split_stream([(text, 1, None)])
    assert [len(word_encoder.encode(t)) for t, *_ in windows] == [300, 300, 200]
    assert _stitch([t for t, *_ in windows]) == text


def test_stats_counts_chunks_per_document(monkeypatch, tmp_path):
    """GET /stats inventories the index per document and per chunk type, so a
    paper whose figures were never extracted shows figure-description 0."""
    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, PointStruct, VectorParams
    import app.store as store
    from app.main import app, settings

    client = QdrantClient(path=str(tmp_path / "q"))
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    rows = [("attention", "prose"), ("attention", "figure-description"),
            ("attention", "prose"), ("lora", "prose")]
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[PointStruct(id=i + 1, vector=[0.1, 0.2, 0.3, 0.4],
                            payload={"document_id": d, "chunk_type": t, "text": "x"})
                for i, (d, t) in enumerate(rows)],
    )
    monkeypatch.setattr(store, "get_client", lambda: client)

    body = TestClient(app).get("/stats").json()
    assert body["points"] == 4
    assert body["documents"]["attention"] == {"prose": 2, "figure-description": 1, "table-row": 0, "total": 3}
    assert body["documents"]["lora"]["figure-description"] == 0
