"""
Hybrid retriever for KnowledgeVault - two retrieval channels fused via Reciprocal Rank Fusion (RRF).

Channel 1 - Dense text (Qdrant ANN):
    A query embedding is produced by ``embed_text`` and used for approximate nearest-neighbour
    search inside the Qdrant collection.  This channel captures semantic similarity.

Channel 2 - Sparse BM25 (in-memory):
    Stored points are scrolled from Qdrant (up to the 10,000-point scroll limit) and a BM25
    index is built on the fly from their ``text`` payloads.  This channel captures exact keyword
    matches that dense search may miss.

Why RRF instead of weighted score combination?
    Dense cosine scores and BM25 scores live on completely different numerical distributions.
    Scaling or normalising them introduces arbitrary hyper-parameters.  RRF avoids this by
    working purely on rank positions: each document's contribution is ``1 / (k + rank)``, which
    is independent of the absolute score magnitude.  The constant ``k = 60`` was recommended
    in the original RRF paper (Cormack et al., 2009) and remains the conventional default.

Widening (parent/child):
    Only 300-token children are embedded. Each prose child carries the
    ``parent_id`` of the 1,500-token span it was cut from. With
    ``widen=True`` the retriever groups every child that shares a hit's
    parent_id, orders them by chunk_index, and stitches them back into the
    parent span with the 50-token overlaps removed. Match small, read large.

Image URL pattern:
    Chunks whose ``chunk_type`` is ``"figure-description"`` carry an ``image_path`` payload
    field that points to the PNG on disk.  ``_image_url`` converts that to a relative URL
    served by the FastAPI ``/figures`` static-file mount so the frontend can display the
    original figure alongside the answer.
"""

import logging
import os
from typing import Optional

from rank_bm25 import BM25Okapi
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from app.chunker import _OVERLAP, _enc
from app.config import get_settings
from app.embedder import embed_text
from app.schemas import RetrievedChunk
from app.store import get_client

logger = logging.getLogger(__name__)

_RRF_K = 60


def _image_url(payload: dict) -> Optional[str]:
    """Return a frontend-accessible URL for figure chunks, or None for prose/table chunks."""
    if payload.get("chunk_type") != "figure-description":
        return None
    image_path = payload.get("image_path")
    if not image_path:
        return None
    return f"/figures/{os.path.basename(image_path)}"


def _stitch(children_texts: list[str]) -> str:
    """Reassemble consecutive overlapping child windows into one parent span.

    Children were cut with a 50-token overlap, so after the first window every
    later one starts with 50 tokens the previous window already ended with.
    Drop those from each successor and the decode is seamless.
    """
    enc = _enc()
    if not children_texts:
        return ""
    out = list(enc.encode(children_texts[0]))
    for t in children_texts[1:]:
        toks = enc.encode(t)
        out.extend(toks[_OVERLAP:] if len(toks) > _OVERLAP else toks)
    return enc.decode(out)


def retrieve(
    query: str,
    k: int = 5,
    document_id: Optional[str] = None,
    widen: bool = False,
) -> list[RetrievedChunk]:
    """Return up to *k* chunks most relevant to *query* using hybrid dense+BM25+RRF retrieval.

    Parameters
    ----------
    query:
        Natural-language question or search string.
    k:
        Number of results to return.
    document_id:
        When provided, retrieval is scoped to chunks that belong to this document only.
    widen:
        When True, prose hits also carry ``parent_text``: the full 1,500-token
        parent span, stitched from every child that shares the hit's parent_id.

    Returns
    -------
    list[RetrievedChunk]
        Ranked list (highest RRF score first) of at most *k* chunks.
    """
    settings = get_settings()
    collection = settings.qdrant_collection

    # ── Qdrant client, local or server, decided in app/store.py ────────────
    client = get_client()

    # Optional per-document filter
    qd_filter: Optional[Filter] = (
        Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )
        if document_id
        else None
    )

    # ── Channel 1: Dense text (ANN) ──────────────────────────────────────────
    query_vec = embed_text(query)
    dense_hits = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=k * 3,
        query_filter=qd_filter,
        with_payload=True,
    ).points

    # ── Channel 2: BM25 in-memory ────────────────────────────────────────────
    all_points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=qd_filter,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )

    if not all_points:
        return []

    tokenised_corpus = [
        p.payload.get("text", "").lower().split() for p in all_points
    ]
    bm25 = BM25Okapi(tokenised_corpus)
    scores = bm25.get_scores(query.lower().split())
    bm25_ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[: k * 3]

    # ── RRF fusion ───────────────────────────────────────────────────────────
    rrf_scores: dict[str, float] = {}
    id_to_payload: dict[str, dict] = {}

    # Dense channel contributions
    for rank, hit in enumerate(dense_hits, start=1):
        pid = str(hit.id)
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
        id_to_payload[pid] = hit.payload

    # BM25 channel contributions
    for rank, (idx, _score) in enumerate(bm25_ranked, start=1):
        pid = str(all_points[idx].id)
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
        id_to_payload[pid] = all_points[idx].payload

    # ── Build final result list ──────────────────────────────────────────────
    top_ids = sorted(rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True)[:k]

    # ── Optional widening: group siblings by parent_id from the scrolled set ─
    # all_points is already in memory for BM25, so widening costs no extra
    # round trip: index the children by parent, then stitch on demand.
    siblings: dict[str, list[dict]] = {}
    if widen:
        for pt in all_points:
            pl = pt.payload
            par = pl.get("parent_id")
            if par:
                siblings.setdefault(par, []).append(pl)
        for par in siblings:
            siblings[par].sort(key=lambda pl: pl.get("chunk_index", 0))

    results: list[RetrievedChunk] = []
    for pid in top_ids:
        p = id_to_payload[pid]
        score = rrf_scores[pid]
        parent_id = p.get("parent_id")
        parent_text = None
        if widen and parent_id and parent_id in siblings:
            parent_text = _stitch([pl.get("text", "") for pl in siblings[parent_id]])
        results.append(
            RetrievedChunk(
                chunk_id=p.get("chunk_id", pid),
                document_id=p.get("document_id", ""),
                chunk_type=p.get("chunk_type", "prose"),
                text=p.get("text", ""),
                page_number=p.get("page_number", 0),
                section_heading=p.get("section_heading"),
                image_url=_image_url(p),
                score=round(score, 6),
                parent_id=parent_id,
                parent_text=parent_text,
            )
        )

    logger.info(
        "retrieve: %r -> %d chunks (dense+bm25+rrf%s)",
        query[:60], len(results), ", widened" if widen else "",
    )
    return results
