"""Retrieval pipeline - HyDE + hybrid dense+BM25+RRF + cross-encoder rerank + compression.

Retrieval channels (adapted from Week 4 KnowledgeVault):
    Dense (ANN) - embed text → Qdrant ANN → semantic matches
    BM25 (sparse, in-memory) - keyword matches the dense channel misses
    RRF fusion (k=60) - combines both rank lists without score normalisation

Week 6 additions on top of Week 4 retrieval:
    HyDE - embed a hypothetical answer instead of the raw query (probe only uses dense)
    Cross-encoder rerank - bge-reranker-base precision pass after wide retrieval
    Extractive compression - sentence-level trimming, marker always preserved

Week 4 payload convention (knowledgevault collection)
------------------------------------------------------
    chunk_id        str   - stable ID used for citations
    text            str   - plain body text (NO [chunk-id] prefix stored in W4)
    chunk_type      str   - "prose" | "figure-description" | "table-row"
    page_number     int
    section_heading str | None
    image_path      str | None
    document_id     str

_point_to_chunk adds the [chunk-id] marker prefix when building Chunk objects so
that the Week 6 citation contract is satisfied downstream.
"""
from __future__ import annotations

import logging
from itertools import zip_longest

from app.compressor import compress
from app.config import get_settings
from app.embedder import embed_text
from app.hyde import hyde_probe
from app.reranker import rerank
from app.schemas import Chunk

logger = logging.getLogger("ragoptimizer.retriever")

_RRF_K = 60   # Cormack et al. (2009) conventional default


# ── Qdrant client (lazy singleton) ──────────────────────────────────────────

_qdrant_client = None


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("pip install qdrant-client==1.12.1") from exc
    s = get_settings()
    _qdrant_client = QdrantClient(
        url=s.qdrant_url,
        api_key=s.qdrant_api_key or None,
    )
    logger.info("Qdrant connected → %s  collection=%s", s.qdrant_url, s.qdrant_collection)
    return _qdrant_client


# ── Payload → Chunk mapping ──────────────────────────────────────────────────

def _payload_to_chunk(payload: dict, point_id: str, score: float) -> Chunk:
    """Convert a Week 4 Qdrant payload to a Week 6 Chunk.

    Adds the [chunk-id] marker as the first line of text if it is absent -
    the compressor and citation validator both depend on this marker.
    """
    chunk_id = str(payload.get("chunk_id") or point_id)
    text     = payload.get("text") or ""

    if not text.startswith(f"[{chunk_id}]"):
        text = f"[{chunk_id}]\n{text}"

    return Chunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        source=str(payload.get("document_id") or ""),
    )


# ── Threshold-gate statistics (Week 5's formula, one implementation) ─────────

def _gate_stats(scores: list[float]) -> tuple[float, float]:
    """Week 5's threshold-gate inputs from a list of dense cosine scores.

    W5's exact formula, kept in one place so every channel that gates is gated
    identically: sort descending, top1 = [0], top3 = [2] (0.0 when fewer than
    three hits), spread = max(0.0, top1 - top3), both clamped to [0, 1].

    An empty list yields (0.0, 0.0), which fails the gate on its own - an empty
    pool refuses through the same path rather than short-circuiting past it.
    """
    ordered = sorted(scores, reverse=True)
    top1 = ordered[0] if ordered else 0.0
    top3 = ordered[2] if len(ordered) >= 3 else 0.0
    spread = max(0.0, top1 - top3)
    return max(0.0, min(1.0, top1)), max(0.0, min(1.0, spread))


def hyde_gate_stats(hyde_candidates: list[Chunk]) -> tuple[float, float]:
    """(top1_hyde, spread_hyde) for the HyDE channel - same formula as the raw one.

    Chunk.score on a _dense_search result IS the dense cosine (it comes straight
    off `float(h.score)` from Qdrant), so the HyDE gate inputs read directly off
    `hyde_candidates` with no second query. Qdrant returns dense hits best-first,
    so these floats are pool-independent: identical at limit 15 or 90.

    An empty list (HyDE disabled, or the probe raised) returns (0.0, 0.0) →
    hyde_ok is False. A failed probe must not be a free pass through the gate.
    """
    return _gate_stats([c.score for c in hyde_candidates])


# ── Search channels ──────────────────────────────────────────────────────────

def _dense_search(probe: str, k: int) -> list[Chunk]:
    """ANN-only search. Used for the HyDE probe (semantic embedding, no keyword)."""
    s = get_settings()
    try:
        vec = embed_text(probe)
        hits = _get_qdrant().query_points(
            collection_name=s.qdrant_collection,
            query=vec,
            limit=k,
            with_payload=True,
        ).points
        chunks = [_payload_to_chunk(h.payload or {}, str(h.id), float(h.score)) for h in hits]
        logger.debug("Dense search → %d hits (probe: %.60s)", len(chunks), probe)
        return chunks
    except Exception as exc:
        logger.warning("Dense search failed: %s", exc)
        return []


def _hybrid_search(query: str, k: int) -> tuple[list[Chunk], float, float]:
    """Dense + BM25 + RRF - mirrors Week 4 KnowledgeVault retrieve() exactly.

    BM25 is built in-memory each call from a full collection scroll. For the
    'Attention is All You Need' corpus this is fast (<1s). For large corpora
    consider caching the index or switching to Qdrant sparse vectors.

    Returns (chunks, top1_dense, spread) - the same shape as Week 5's
    Retriever.search(). The dense cosine scores travel *beside* the chunks, not
    inside them: Chunk.score stays the raw RRF score that the reranker and the
    trace UI depend on, while the Week 5 threshold gate needs the dense cosines,
    which are the only channel on an interpretable [0, 1] scale.

    Both gating callers read these floats: the W5 baseline gates on them
    directly, and the full W6 pipeline uses them as its `raw` channel (see
    retrieve_with_trace). They are pool-independent - Qdrant returns dense hits
    best-first, so top1/top3 are identical whether the limit is 15 or 90.
    """
    s = get_settings()
    collection = s.qdrant_collection

    try:
        from rank_bm25 import BM25Okapi
        from qdrant_client.http.models import Filter  # noqa: F401 (may be unused)
    except ImportError as exc:
        raise RuntimeError("pip install rank-bm25 qdrant-client") from exc

    try:
        client = _get_qdrant()

        # Channel 1: dense ANN
        vec = embed_text(query)
        dense_hits = client.query_points(
            collection_name=collection,
            query=vec,
            limit=k * 3,
            with_payload=True,
        ).points

        # Dense cosine scores - the Week 5 threshold-gate input.
        # Deliberately NOT stored in Chunk.score, which stays raw RRF here.
        # _gate_stats is W5's top1/top3 formula and [0, 1] clamping, shared with
        # the HyDE channel so both are gated by identical arithmetic.
        top1_dense, spread = _gate_stats([float(h.score) for h in dense_hits])

        # Channel 2: BM25 - scroll ALL points (Week 4 approach)
        all_points, _ = client.scroll(
            collection_name=collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        if not all_points:
            logger.warning("Qdrant scroll returned 0 points - collection empty?")
            return [], 0.0, 0.0

        tokenised = [
            (p.payload or {}).get("text", "").lower().split() for p in all_points
        ]
        bm25 = BM25Okapi(tokenised)
        scores = bm25.get_scores(query.lower().split())
        bm25_ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[: k * 3]

        # RRF fusion
        rrf: dict[str, float] = {}
        id_to_payload: dict[str, dict] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            pid = str(hit.id)
            rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
            id_to_payload[pid] = hit.payload or {}

        for rank, (idx, _) in enumerate(bm25_ranked, start=1):
            pid = str(all_points[idx].id)
            rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
            id_to_payload[pid] = all_points[idx].payload or {}

        top_ids = sorted(rrf, key=lambda pid: rrf[pid], reverse=True)[:k]
        chunks = [_payload_to_chunk(id_to_payload[pid], pid, rrf[pid]) for pid in top_ids]
        logger.info(
            "Hybrid search → %d chunks (dense+BM25+RRF) top1=%.3f spread=%.3f for: %.60s",
            len(chunks), top1_dense, spread, query,
        )
        return chunks, top1_dense, spread   # already clamped by _gate_stats

    except Exception as exc:
        logger.warning("Hybrid search failed: %s", exc)
        return [], 0.0, 0.0


def _union_dedupe(a: list[Chunk], b: list[Chunk], limit: int | None = None) -> list[Chunk]:
    """Rank-interleave two candidate lists, dedupe by chunk_id, cap the union at `limit`.

    The two lists carry INCOMPARABLE scores - the HyDE list is dense cosine (0-1),
    the hybrid list is an RRF score (~0.016-0.03). Merging by score would let one
    channel swamp the other, so we merge by *rank*: a[0], b[0], a[1], b[1], ...

    `limit` (= vector_top_k_wide, 30) truncates the union to exactly the candidate
    budget the reranker is sized for. Without it the union is up to 60 chunks and
    the cross-encoder scores twice the pairs it advertises - two batches, not one.
    """
    by_id: dict[str, Chunk] = {}
    for chunk_a, chunk_b in zip_longest(a, b):
        for chunk in (chunk_a, chunk_b):
            if chunk is None or chunk.chunk_id in by_id:
                continue
            by_id[chunk.chunk_id] = chunk
            if limit is not None and len(by_id) >= limit:
                return list(by_id.values())
    return list(by_id.values())


# ── Baseline retriever (Week 5 style) ─────────────────────────────────────────

def retrieve_baseline_gated(
    query: str, k: int | None = None
) -> tuple[list[Chunk], float, float]:
    """CitationRAG-style retrieval used as the 'before' in /eval/compare.

    Mirrors Week 5's *retrieval* (same collection, same embedder, hybrid+RRF):
        Channel 1 - Dense ANN  (embed query → Qdrant approximate nearest-neighbour)
        Channel 2 - BM25       (in-memory sparse over scrolled payload texts)
        RRF fusion (k=60)      (Cormack et al. 2009, rank-position fusion)

    Nothing added by Week 6 is applied here - no HyDE probe, no cross-encoder
    reranking, no extractive compression.

    Returns (top-k chunks by RRF score, top1_dense, spread). The two floats are
    the inputs to Week 5's threshold gate; the gate itself is applied by the
    caller (see eval_compare in app/main.py), exactly as Week 5 applied it in
    its own route handler rather than inside the retriever. A "before" column
    that cannot refuse is not the before - without these stats the baseline is
    structurally incapable of the refusal Week 5 shipped.

    Returns ([], 0.0, 0.0) when retrieval finds nothing; a 0.0 top1 fails the
    gate on its own, so an empty pool refuses through the same path.
    """
    settings = get_settings()
    narrow_k = k or settings.vector_top_k_narrow
    # _hybrid_search with the NARROW k - this is the whole point of the baseline.
    # Week 5 fused its two rank lists over k*3 = 15 candidates per channel. Passing
    # the wide budget (30) here would fuse over k*3 = 90 per channel instead, and RRF
    # scores rank positions, so a deeper pool reorders the top 5: a chunk sitting at
    # dense-rank 40 scores 0 in Week 5 (outside the pool) but non-zero at 90-deep.
    # A wide pool here would silently make the "before" column a Week 6-strength
    # retriever and understate what the full pipeline adds.
    candidates, top1_dense, spread = _hybrid_search(query, narrow_k)
    logger.info(
        "Baseline retrieve → %d candidates, returning top %d "
        "(no rerank/compress) top1=%.3f spread=%.3f",
        len(candidates), narrow_k, top1_dense, spread,
    )
    return candidates[:narrow_k], top1_dense, spread


def retrieve_baseline(query: str, k: int | None = None) -> list[Chunk]:
    """Chunks-only view of retrieve_baseline_gated, for callers that do not gate.

    Week 5's retrieval, discarding the gate stats. week6_notebook.ipynb uses this
    to measure retrieval quality on its own; anything that needs to reproduce
    Week 5's *behaviour* - including the threshold gate - must call
    retrieve_baseline_gated and apply the gate. One implementation, two views.
    """
    chunks, _top1_dense, _spread = retrieve_baseline_gated(query, k)
    return chunks


# ── Full pipeline ─────────────────────────────────────────────────────────────

def retrieve(query: str, k: int | None = None) -> list[Chunk]:
    """Run the full RAGOptimizer pipeline.

    1. HyDE probe  → dense ANN search  (semantic, probe text)
    2. Original query → hybrid dense+BM25+RRF  (Week 4 style, user's words)
    3. Union + dedupe by chunk_id, truncated to top-30 (vector_top_k_wide)
    4. Cross-encoder rerank (bge-reranker-base) - exactly 30 pairs, one batch
    5. Slice top-k (vector_top_k_narrow)
    6. Extractive compression (marker always preserved)

    Chunks-only, and deliberately UNGATED: this is the raw retrieval view used by
    app/tools.py and by week6_notebook.ipynb's latency benchmark, which times
    retrieval itself and must not pay for trace construction. Callers that need
    the pipeline's *behaviour* - including the threshold gate - must use
    retrieve_with_trace, which surfaces the gate inputs for both channels, and
    apply the gate as /ask and eval_compare do. One retrieval, two views; the
    same split as retrieve_baseline / retrieve_baseline_gated.
    """
    settings = get_settings()
    k = k or settings.vector_top_k_narrow

    # 1. HyDE probe → dense candidates
    hyde_candidates: list[Chunk] = []
    if settings.hyde_enabled:
        try:
            probe = hyde_probe(query)
            hyde_candidates = _dense_search(probe, settings.vector_top_k_wide)
        except Exception as exc:
            logger.warning("HyDE probe failed, skipping: %s", exc)

    # 2. Original query → hybrid candidates (Week 4 retrieval)
    # Wide pool. The gate stats are discarded *here only* because this view does
    # not gate - not because the pipeline does not. See retrieve_with_trace.
    hybrid_candidates, _top1, _spread = _hybrid_search(query, settings.vector_top_k_wide)

    # 3. Union, truncated to the wide budget - "pull thirty candidates" (30, not 60)
    wide = settings.vector_top_k_wide
    if hyde_candidates:
        candidates = _union_dedupe(hyde_candidates, hybrid_candidates, limit=wide)
    else:
        candidates = hybrid_candidates[:wide]

    if not candidates:
        return []

    # 4. Cross-encoder rerank - exactly one batch of <= 30 pairs
    # RERANKER_ENABLED=false skips the cross-encoder and keeps the union order,
    # so the lab can measure what the reranker is actually contributing.
    reranked = rerank(query, candidates) if settings.reranker_enabled else candidates

    # 5. Narrow slice
    top_k = reranked[:k]

    # 6. Extractive compression - marker preserved inside
    return compress(top_k, query)


def retrieve_with_trace(
    query: str,
    k: int | None = None,
):
    """Run the full pipeline and return (chunks, PipelineTrace).

    Captures intermediate state at each stage so the UI can show exactly
    what was retrieved, how reranking changed the order, and what was
    finally sent to the LLM.

    Threshold-gate inputs
    ---------------------
    Also surfaces Week 5's gate inputs for BOTH W6 retrieval channels on the
    trace - top1_raw/spread_raw (raw query, dense side of the hybrid search) and
    top1_hyde/spread_hyde (the HyDE probe's dense hits). Same formula, same
    thresholds as Week 5; see hyde_gate_stats and _gate_stats.

    This function does not itself refuse - it reports. The gate is applied by the
    callers (/ask and eval_compare in app/main.py), exactly as Week 5 applied its
    gate in the route handler rather than inside the retriever:

        raw_ok  = top1_raw  >= similarity_threshold and spread_raw  >= spread_delta
        hyde_ok = top1_hyde >= similarity_threshold and spread_hyde >= spread_delta
        refuse unless (raw_ok or hyde_ok)

    Why both channels rather than the raw query alone: gating W6 on the raw query
    would reproduce the baseline's decision on every row by construction - same
    input, same thresholds - and would test nothing. Having a second probe is
    what Week 6 actually built, so HyDE gets to be a genuine second chance at
    clearing the gate. That is fair, and it carries a real downside the week
    teaches: a drifted probe can clear the gate on the wrong chunk and convert a
    safe refusal into a confident wrong answer.

    Returns:
        (list[Chunk], PipelineTrace)
    """
    from app.schemas import ChunkTrace, PipelineTrace

    settings = get_settings()
    k = k or settings.vector_top_k_narrow
    trace = PipelineTrace()

    def _preview(chunk: Chunk) -> str:
        """Body text without the marker line, first 150 chars."""
        parts = chunk.text.split("\n", 1)
        body = parts[1] if len(parts) > 1 else parts[0]
        return body.strip()[:150]

    # ── 1. HyDE probe ─────────────────────────────────────────────────────
    hyde_candidates: list[Chunk] = []
    if settings.hyde_enabled:
        try:
            probe = hyde_probe(query)
            trace.hyde_probe = probe
            hyde_candidates = _dense_search(probe, settings.vector_top_k_wide)
            trace.hyde_candidates_count = len(hyde_candidates)
        except Exception as exc:
            logger.warning("HyDE probe failed in trace mode: %s", exc)

    # HyDE's gate inputs come straight off the probe's dense hits - Chunk.score
    # IS the dense cosine there. Disabled/failed HyDE leaves this (0.0, 0.0),
    # which fails the gate rather than passing it.
    trace.top1_hyde, trace.spread_hyde = hyde_gate_stats(hyde_candidates)

    # ── 2. Hybrid search on original query ────────────────────────────────
    # Wide pool. The gate stats are the RAW channel's - kept, not discarded:
    # the full pipeline gates on (raw_ok or hyde_ok), same rule and thresholds
    # as the W5 baseline, just evaluated against both of W6's probes.
    hybrid_candidates, top1_raw, spread_raw = _hybrid_search(query, settings.vector_top_k_wide)
    trace.original_candidates_count = len(hybrid_candidates)
    trace.top1_raw, trace.spread_raw = top1_raw, spread_raw

    # ── 3. Union + dedupe, truncated to top-30 (vector_top_k_wide) ────────
    wide = settings.vector_top_k_wide
    if hyde_candidates:
        candidates = _union_dedupe(hyde_candidates, hybrid_candidates, limit=wide)
    else:
        candidates = hybrid_candidates[:wide]
    trace.combined_count = len(candidates)

    # Capture top-10 by vector/RRF score BEFORE reranking
    top_before = sorted(candidates, key=lambda c: c.score, reverse=True)[:10]
    trace.before_rerank = [
        ChunkTrace(
            chunk_id=c.chunk_id,
            preview=_preview(c),
            vector_score=round(c.score, 6),
        )
        for c in top_before
    ]

    if not candidates:
        return [], trace

    # ── 4. Cross-encoder rerank ───────────────────────────────────────────
    # Build a lookup of vector scores before we overwrite them
    vector_score_map = {c.chunk_id: c.score for c in candidates}
    # Same toggle as retrieve(); both paths must agree or /ask and /eval diverge.
    reranked = (rerank(query, candidates) if settings.reranker_enabled
                else sorted(candidates, key=lambda c: c.score, reverse=True))

    trace.after_rerank = [
        ChunkTrace(
            chunk_id=c.chunk_id,
            preview=_preview(c),
            vector_score=round(vector_score_map.get(c.chunk_id, 0.0), 6),
            rerank_score=round(c.score, 4),
        )
        for c in reranked[:10]
    ]

    # ── 5. Narrow slice ───────────────────────────────────────────────────
    top_k = reranked[:k]

    # ── 6. Compress ───────────────────────────────────────────────────────
    final = compress(top_k, query)
    trace.final_chunks = [
        ChunkTrace(
            chunk_id=c.chunk_id,
            preview=_preview(c),
            vector_score=round(vector_score_map.get(c.chunk_id, 0.0), 6),
            rerank_score=round(c.score, 4),
        )
        for c in final
    ]

    return final, trace
