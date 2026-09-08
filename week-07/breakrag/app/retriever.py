"""Hybrid Qdrant + BM25 retriever with RRF fusion.

Copied from CitationRAG (app/retriever.py) - same Qdrant collection,
same embedding model, same fusion formula. BreakRAG uses this to populate
retrieved_contexts in SUTResponse for the judge scoring pipeline.

Two retrieval channels fused via Reciprocal Rank Fusion (RRF):

  Channel 1 - Dense (Qdrant ANN):
      Query is embedded with text-embedding-3-large and used for approximate
      nearest-neighbour search. Captures semantic similarity.

  Channel 2 - Sparse BM25 (in-memory):
      All stored points are scrolled from Qdrant and a BM25Okapi index is
      built over their text payloads. Captures exact keyword matches.

Why RRF instead of score combination?
      Dense cosine scores and BM25 scores live on different numerical
      distributions. RRF avoids this by working purely on rank positions:
          contribution = 1 / (k + rank),  k = 60  (Cormack et al., 2009)
      No arbitrary scaling or normalisation hyper-parameters.

Collection:
      Same Qdrant collection as KnowledgeVault / CitationRAG.
      Set QDRANT_MODE and QDRANT_LOCAL_PATH (or QDRANT_URL and
      QDRANT_API_KEY) plus QDRANT_COLLECTION in .env; app/store.py opens it.
"""

import logging
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.embedder import embed_text

logger = logging.getLogger(__name__)

_RRF_K = 60


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever."""

    chunk_id: str   # renumbered doc#1 ... doc#k
    text: str
    source_url: str | None
    score: float    # normalised RRF score in [0, 1]


class Retriever:
    """Hybrid Qdrant + BM25 retriever with RRF fusion.

    Lazy-initialised - QdrantClient connects on first search() call, not at
    import time, so smoke tests don't require a live Qdrant instance.
    """

    def __init__(self) -> None:
        settings = get_settings()
        from app.store import get_client
        self._client = get_client()
        self._collection = settings.qdrant_collection

    def search(self, query: str) -> tuple[list[RetrievedChunk], float, float]:
        """Return (top-k chunks, top-1 dense score, spread).

        top-1 dense score and spread are used by the threshold gate in the
        /answer handler. RRF is used for chunk ordering only.
        """
        settings = get_settings()
        k = settings.retrieval_top_k

        # Embed the query
        query_vec = embed_text(query)

        # Channel 1: Dense ANN
        dense_hits = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=k * 3,
            with_payload=True,
        ).points

        dense_score_map: dict[str, float] = {str(h.id): h.score for h in dense_hits}
        dense_scores_sorted = sorted(dense_score_map.values(), reverse=True)
        top1_dense = dense_scores_sorted[0] if dense_scores_sorted else 0.0
        top3_dense = dense_scores_sorted[2] if len(dense_scores_sorted) >= 3 else 0.0
        spread = max(0.0, top1_dense - top3_dense)

        # Channel 2: BM25 in-memory
        all_points, _ = self._client.scroll(
            collection_name=self._collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )

        if not all_points:
            logger.warning("Qdrant collection %r is empty.", self._collection)
            return [], 0.0, 0.0

        tokenised = [p.payload.get("text", "").lower().split() for p in all_points]
        bm25 = BM25Okapi(tokenised)
        bm25_raw = bm25.get_scores(query.lower().split())
        bm25_ranked = sorted(
            enumerate(bm25_raw), key=lambda x: x[1], reverse=True
        )[: k * 3]

        # RRF fusion
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

        top_ids = sorted(rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True)[:k]

        rrf_max = 2.0 / (_RRF_K + 1)
        chunks: list[RetrievedChunk] = []
        for i, pid in enumerate(top_ids, start=1):
            p = id_to_payload[pid]
            norm_score = min(1.0, rrf_scores[pid] / rrf_max) if rrf_max > 0 else 0.0
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"doc#{i}",
                    text=p.get("text", ""),
                    source_url=p.get("source_url"),
                    score=max(0.0, min(1.0, norm_score)),
                )
            )

        logger.info(
            "retrieve: %r -> %d chunks (dense+bm25+rrf)  top1=%.3f spread=%.3f",
            query[:60], len(chunks), top1_dense, spread,
        )
        return chunks, max(0.0, min(1.0, top1_dense)), max(0.0, min(1.0, spread))
