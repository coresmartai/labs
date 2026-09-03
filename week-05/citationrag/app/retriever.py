"""
Hybrid retriever for CitationRAG - uses Week 4's KnowledgeVault Qdrant collection.

Two retrieval channels fused via Reciprocal Rank Fusion (RRF):

  Channel 1 - Dense (Qdrant ANN):
      Query is embedded with text-embedding-3-large and used for approximate
      nearest-neighbour search.  Captures semantic similarity.
      The raw cosine scores from this channel (in [0, 1]) are used for the
      threshold gate in main.py because they have an interpretable scale.

  Channel 2 - Sparse BM25 (in-memory):
      All stored points are scrolled from Qdrant and a BM25Okapi index is built
      over their text payloads.  Captures exact keyword matches that dense search
      may miss.

Why RRF instead of score combination?
      Dense cosine scores and BM25 scores live on different numerical distributions.
      RRF avoids this by working purely on rank positions:
          contribution = 1 / (k + rank),  k = 60  (Cormack et al., 2009)
      No arbitrary scaling or normalisation hyper-parameters.

Chunk-ID renumbering:
      Returned chunks are numbered doc#1 … doc#k in RRF rank order.
      This keeps the citation contract in app/llm.py (which expects [doc#N] markers)
      working unchanged even though the underlying Qdrant IDs are hex UUIDs.

Spread deduplication:
      KnowledgeVault chunks overlap by 50 tokens, so a well-covered query often
      returns two or three near-duplicate sibling chunks with near-identical
      dense scores. Raw top1 - top3 spread would then be tiny and the gate in
      main.py would refuse a perfectly answerable question. Before computing
      spread, near-duplicate hits (token-set Jaccard >= 0.6) are collapsed to
      their highest-scoring member, so spread measures how far ahead the best
      DISTINCT passage is - which is what the gate actually wants to know.

Collection:
      The same store Week 4 KnowledgeVault ingested into. The client comes from
      app/store.py get_client() - embedded local by default (QDRANT_MODE=local,
      reading QDRANT_LOCAL_PATH), or a server via QDRANT_MODE=server.
"""

import logging

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.embedder import embed_text
from app.schemas import Chunk
from app.store import get_client

logger = logging.getLogger(__name__)

_RRF_K = 60
_DEDUPE_JACCARD = 0.6  # token-set overlap above this = same passage for spread


def _dedupe_scores(hits: list[tuple[float, str]]) -> list[float]:
    """Collapse near-duplicate texts and return their scores, best first.

    *hits* is (score, text) sorted by score descending. Two texts whose
    token-set Jaccard similarity is >= _DEDUPE_JACCARD are treated as the same
    passage (overlapping sibling chunks); only the highest-scoring one counts.
    """
    kept_scores: list[float] = []
    kept_tokens: list[set[str]] = []
    for score, text in hits:
        tokens = set(text.lower().split())
        is_dup = False
        for seen in kept_tokens:
            union = tokens | seen
            if union and len(tokens & seen) / len(union) >= _DEDUPE_JACCARD:
                is_dup = True
                break
        if not is_dup:
            kept_scores.append(score)
            kept_tokens.append(tokens)
    return kept_scores


class Retriever:
    """Hybrid Qdrant + BM25 retriever with RRF fusion."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = get_client()
        self._collection = settings.qdrant_collection

    def search(self, query: str) -> tuple[list[Chunk], float, float]:
        """Return (top-k chunks, top-1 dense score, top-1/top-3 spread).

        Dense cosine scores are returned for the threshold gate - they live in
        [0, 1] and have a calibratable meaning.  RRF is used for ranking only.
        Returned chunks are renumbered doc#1 … doc#k for citation-contract
        compatibility.
        """
        settings = get_settings()
        k = settings.retrieval_top_k

        # ── Embed query ────────────────────────────────────────────────────────
        query_vec = embed_text(query)

        # ── Channel 1: Dense ANN ───────────────────────────────────────────────
        dense_hits = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=k * 3,
            with_payload=True,
        ).points

        # Dense cosine scores - used for threshold gate, NOT stored in Chunk.score.
        # Spread is computed AFTER near-duplicate collapse: overlapping sibling
        # chunks score almost identically and would fake a "narrow spread" for
        # queries the index covers perfectly well (a false refusal).
        scored_texts = sorted(
            ((h.score, (h.payload or {}).get("text", "")) for h in dense_hits),
            key=lambda st: st[0],
            reverse=True,
        )
        deduped = _dedupe_scores(scored_texts)
        top1_dense = deduped[0] if deduped else 0.0
        top3_dense = deduped[2] if len(deduped) >= 3 else 0.0
        spread = max(0.0, top1_dense - top3_dense)

        # ── Channel 2: BM25 in-memory ──────────────────────────────────────────
        all_points, _ = self._client.scroll(
            collection_name=self._collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )

        if not all_points:
            logger.warning("Qdrant collection %r is empty.", self._collection)
            return [], 0.0, 0.0

        tokenised = [
            p.payload.get("text", "").lower().split() for p in all_points
        ]
        bm25 = BM25Okapi(tokenised)
        bm25_raw = bm25.get_scores(query.lower().split())
        bm25_ranked = sorted(
            enumerate(bm25_raw), key=lambda x: x[1], reverse=True
        )[: k * 3]

        # ── RRF fusion ─────────────────────────────────────────────────────────
        rrf_scores: dict[str, float] = {}
        id_to_payload: dict[str, dict] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            pid = str(hit.id)
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
            id_to_payload[pid] = hit.payload

        for rank, (idx, _bm_score) in enumerate(bm25_ranked, start=1):
            pid = str(all_points[idx].id)
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
            id_to_payload[pid] = all_points[idx].payload

        top_ids = sorted(
            rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True
        )[:k]

        # ── Build Chunk list with renumbered IDs ───────────────────────────────
        # Normalise RRF scores to [0, 1] using the two-channel theoretical max.
        rrf_max = 2.0 / (_RRF_K + 1)  # max possible: top of both channels, k=60

        chunks: list[Chunk] = []
        for i, pid in enumerate(top_ids, start=1):
            p = id_to_payload[pid]
            norm_score = min(1.0, rrf_scores[pid] / rrf_max) if rrf_max > 0 else 0.0
            chunks.append(
                Chunk(
                    chunk_id=f"doc#{i}",          # renumber for citation contract
                    text=p.get("text", ""),
                    source_url=p.get("source_url"),
                    score=max(0.0, min(1.0, norm_score)),
                    # Provenance survives renumbering - a citation should be
                    # able to say which document and page it came from.
                    document_id=p.get("document_id"),
                    page_number=p.get("page_number"),
                )
            )

        logger.info(
            "retrieve: %r -> %d chunks (dense+bm25+rrf)  top1=%.3f spread=%.3f",
            query[:60], len(chunks), top1_dense, spread,
        )
        return chunks, max(0.0, min(1.0, top1_dense)), max(0.0, min(1.0, spread))
