"""The vector store - Qdrant, with the hybrid retrieval CitationRAG uses.

`QdrantClient(location=settings.qdrant_url, ...)` connects to a real Qdrant
instance: a local Docker container for development, or a free-tier cluster on
Qdrant Cloud (https://cloud.qdrant.io) - same client, same collection API,
the URL is the only difference.

`location=` rather than `url=`: it accepts a real URL exactly like `url=`
would, but it is also the one parameter that understands the special value
`":memory:"` - a fully in-process, ephemeral Qdrant instance. The test suite
points `QDRANT_URL` at `:memory:` so `pytest` never needs a running server.

Retrieval
---------
Two channels fused with Reciprocal Rank Fusion (RRF, k=60; Cormack 2009):
  - Dense ANN   - the query embedded with text-embedding-3-large. Semantic.
  - Sparse BM25 - an in-memory index over the scrolled chunk texts. Exact
                  keyword matches (ticket IDs, model names) dense search misses.
RRF fuses the two rank lists without normalising incompatible score scales.
`search()` returns `(chunks, top1_dense, spread)` - the dense cosine top-1 and
top1-top3 spread are the threshold-gate inputs; the *route* applies the gate
(see main.ask), in the handler rather than the retriever.

The ACL is pushed down into BOTH channels - `query_filter` on the dense query
and `scroll_filter` on the BM25 scroll - so a chunk the caller may not see is
never ranked, by either channel, ever. A genuine database-level filter, not a
Python `if` after the fact.

Performance note: `ensure_collection()` runs its Qdrant round-trips ONCE per
process (guarded by `_ENSURED`), not on every search. Against a cloud cluster
each round-trip is ~1s, so re-checking per search would make every `/ask` pay
for several; the collection's shape does not change under us mid-process.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.embedder import embed_text, embed_texts
from app.schemas import RetrievedChunk, StoredChunk

logger = logging.getLogger(__name__)

# text-embedding-3-large. Fixed at collection-creation time; `ensure_collection`
# refuses to write into a collection of a different width.
VECTOR_SIZE = 3072

# Points per upsert request. Keeps any single write near ~1 MB at 3072 dims.
UPSERT_BATCH = 24

# Reciprocal Rank Fusion constant (Cormack et al. 2009). Same value as CitationRAG.
_RRF_K = 60


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    """Qdrant client - local Docker by default, a Cloud cluster when
    QDRANT_URL points at one, or an in-process instance for tests (`:memory:`)."""
    settings = get_settings()
    client = QdrantClient(
        location=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        # Explicit, because the default is httpx's 5s and a batch of 3072-dim
        # vectors does not cross the public internet in 5s. See config.
        timeout=settings.qdrant_timeout_seconds,
    )
    logger.info("Qdrant connected -> %s  collection=%s", settings.qdrant_url, settings.qdrant_collection)
    return client


def _collection() -> str:
    return get_settings().qdrant_collection


# Collections this process has already verified (shape + ACL index). The
# check costs several cloud round-trips, and the collection cannot change
# shape under us mid-process, so we do it once. reset() clears the entry.
_ENSURED: set[str] = set()


def ensure_collection() -> None:
    """Verify (once per process) the collection exists, has the right vector
    width, and carries the ACL payload index. Create it if absent.

    A Qdrant collection has a fixed vector width, so the width is a contract.
    Two ways to break it, both of which a bare `collection_exists()` would wave
    straight through:

      1. QDRANT_COLLECTION still points at another project's collection.
         Sharing the Qdrant *cluster* is fine; sharing a *collection* is not -
         we would be writing these chunks into a knowledge base another service
         retrieves from.
      2. OPENAI_EMBED_MODEL changed, so the existing collection was built at a
         different width by an earlier run of this same project.

    Either way the upsert would otherwise die deep in the client with
    `could not broadcast input array from shape (N,) into shape (M,)`. Fail
    here instead, with the fix in the message.
    """
    name = _collection()
    if name in _ENSURED:
        return
    client = _client()

    if client.collection_exists(name):
        # ONE get_collection call, read for both the width contract and the
        # ACL-index check.
        info = client.get_collection(name)
        size = getattr(info.config.params.vectors, "size", None)
        if size is not None and size != VECTOR_SIZE:
            settings = get_settings()
            raise RuntimeError(
                f"Qdrant collection {name!r} already exists with {size}-dim vectors, but "
                f"{settings.openai_embed_model!r} produces {VECTOR_SIZE}-dim vectors. Either "
                f"(a) QDRANT_COLLECTION points at another project's collection sharing this "
                f"cluster; point it at this project's own collection (default 'citation_rag'), "
                f"or (b) the embedding model changed since this collection was built - re-index "
                f"with: python -m app.ingest --reset"
            )
        _ensure_acl_index(client, name, existing_schema=info.payload_schema)
    else:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        logger.info("created qdrant collection %s (size=%d, cosine)", name, VECTOR_SIZE)
        _ensure_acl_index(client, name, existing_schema=None)

    _ENSURED.add(name)


def _ensure_acl_index(client: QdrantClient, name: str, *, existing_schema) -> None:
    """Create the keyword payload index the ACL filter needs.

    NOT an optimisation - it is what makes the ACL work at all on a real
    server. Filtering on an unindexed payload field is rejected outright:

        400 Bad request: Index required but not found for "visible_to"
        of one of the following types: [keyword]

    The trap is that the in-process client (`:memory:`, used by the tests)
    filters happily WITHOUT an index, so this failure cannot reproduce under
    pytest - it only appears against Docker or Qdrant Cloud. Local Qdrant also
    ignores index creation entirely (it logs a warning saying so), hence the
    early return.

    `existing_schema` is the payload_schema from the get_collection() the
    caller already made (None for a freshly created collection), so this does
    not cost a second round-trip.
    """
    if get_settings().qdrant_url == ":memory:":
        return
    if "visible_to" in (existing_schema or {}):
        return
    client.create_payload_index(
        collection_name=name,
        field_name="visible_to",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    logger.info("created keyword payload index on visible_to for %s", name)


def _point_id(chunk_id: str) -> int:
    """Qdrant point IDs are ints or UUIDs; our chunk IDs are strings."""
    return int(hashlib.md5(chunk_id.encode("utf-8")).hexdigest()[:16], 16)


def upsert_chunks(chunks: list[StoredChunk]) -> int:
    """Write chunks to the store. Text is expected to be ALREADY scrubbed.

    Note what is embedded: `c.text`, i.e. the text AFTER the Presidio pass.
    The vectors are built from redacted text, so the raw PII is absent from
    the payload and from the embedding alike.
    """
    if not chunks:
        return 0
    ensure_collection()
    # One API call for the whole batch, not one per chunk.
    vectors = embed_texts([c.text for c in chunks])
    points = [
        models.PointStruct(
            id=_point_id(c.chunk_id),
            vector=vector,
            payload={
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                "visible_to": c.visible_to,
                "scrubbed": c.scrubbed,
            },
        )
        for c, vector in zip(chunks, vectors)
    ]
    # Upload in batches. At 3072 dims a point is ~43 KB of JSON, so a 200-chunk
    # PDF in a single request is ~9 MB - slow enough to trip even a generous
    # timeout, and it either all lands or none of it does.
    client, name = _client(), _collection()
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(collection_name=name, points=points[start:start + UPSERT_BATCH])
    return len(points)


def _to_chunk(payload: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=payload["chunk_id"],
        text=payload["text"],
        source=payload["source"],
        score=round(float(score), 4),
        visible_to=list(payload["visible_to"]),
    )


def search(
    query: str,
    roles: list[str],
    limit: int | None = None,
) -> tuple[list[RetrievedChunk], float, float]:
    """Hybrid dense + BM25 + RRF retrieval, ACL pushed down on both channels.

    Returns `(chunks, top1_dense, spread)`:
      - chunks     - the relevant, visible chunks, in fused RRF rank order.
      - top1_dense - highest dense cosine, in [0, 1].
      - spread     - top1 minus top3 dense cosine.

    Two knobs decide membership vs. order, deliberately kept separate:
      - MEMBERSHIP is a relevance floor on the DENSE cosine (>= similarity
        threshold). This is what keeps the citation panel to the chunks that
        are actually about the question - not `k` cards, most of them noise -
        and it is why an analyst asking a CEO question retrieves nothing (the
        one relevant chunk is ACL-hidden; the rest are below the floor).
      - ORDER is BM25+RRF over the visible chunks, so exact-term matches
        (ticket IDs, model names) can outrank a slightly-closer dense vector.

    The score carried on each returned chunk is the DENSE cosine - an
    interpretable [0, 1] similarity - not the RRF score, which is tiny and
    scale-free.
    """
    from app import rbac  # local import: rbac imports audit imports config.

    settings = get_settings()
    ensure_collection()
    k = limit if limit is not None else settings.retrieval_top_k
    floor = settings.similarity_threshold
    acl = rbac.acl_filter_for(roles)
    client, name = _client(), _collection()

    # ---- Channel 1: dense ANN, ACL pushed down ----
    query_vec = embed_text(query)
    dense_hits = client.query_points(
        collection_name=name,
        query=query_vec,
        query_filter=acl,
        limit=k * 3,
        with_payload=True,
    ).points

    dense_cos: dict[str, float] = {str(h.id): float(h.score) for h in dense_hits}
    payloads: dict[str, dict] = {str(h.id): (h.payload or {}) for h in dense_hits}
    dense_sorted = sorted(dense_cos.values(), reverse=True)
    top1 = dense_sorted[0] if dense_sorted else 0.0
    top3 = dense_sorted[2] if len(dense_sorted) >= 3 else 0.0
    spread = max(0.0, top1 - top3)

    # ---- Channel 2: BM25 over the ACL-visible chunks (scroll_filter = same ACL) ----
    points, _ = client.scroll(
        collection_name=name,
        scroll_filter=acl,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )
    bm25_rank: dict[str, int] = {}
    if points:
        tokenised = [(p.payload or {}).get("text", "").lower().split() for p in points]
        bm25 = BM25Okapi(tokenised)
        scored = sorted(
            enumerate(bm25.get_scores(query.lower().split())),
            key=lambda x: x[1],
            reverse=True,
        )[: k * 3]
        for rank, (i, _s) in enumerate(scored, start=1):
            bm25_rank[str(points[i].id)] = rank

    # ---- RRF fusion for ORDER ----
    rrf: dict[str, float] = {}
    for rank, h in enumerate(dense_hits, start=1):
        pid = str(h.id)
        rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (_RRF_K + rank)
    for pid, rank in bm25_rank.items():
        rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (_RRF_K + rank)

    # ---- Relevance floor for MEMBERSHIP (dense cosine), RRF order ----
    eligible = [pid for pid in rrf if dense_cos.get(pid, 0.0) >= floor]
    eligible.sort(key=lambda pid: rrf[pid], reverse=True)
    chunks = [_to_chunk(payloads[pid], dense_cos[pid]) for pid in eligible[:k]]

    logger.info(
        "retrieve %r -> %d chunk(s) (dense+bm25+rrf, floor=%.2f) top1=%.3f spread=%.3f",
        query[:60], len(chunks), floor, top1, spread,
    )
    return chunks, max(0.0, min(1.0, top1)), max(0.0, min(1.0, spread))


def count_withheld(query: str, roles: list[str]) -> int:
    """How many relevant chunks the ACL hid from these roles.

    One unrestricted dense query (the embedding is cached from `search`, so no
    extra OpenAI call): count hits that clear the gate threshold but whose
    `visible_to` does not overlap the caller's roles. This is the forensic
    number behind the `acl_filter` audit event - it is a report, never a gate.
    """
    settings = get_settings()
    ensure_collection()
    allowed = set(roles)
    hits = _client().query_points(
        collection_name=_collection(),
        query=embed_text(query),
        limit=settings.retrieval_top_k * 3,
        with_payload=True,
    ).points
    return sum(
        1
        for h in hits
        if float(h.score) >= settings.similarity_threshold
        and not (set((h.payload or {}).get("visible_to", [])) & allowed)
    )


def all_payloads() -> list[dict]:
    """Every stored payload. Used by the eval set to prove that unredacted
    PII never landed in the store."""
    ensure_collection()
    records, _ = _client().scroll(
        collection_name=_collection(), limit=1000, with_payload=True
    )
    return [dict(r.payload) for r in records]


def count() -> int:
    ensure_collection()
    return int(_client().count(collection_name=_collection()).count)


def reset() -> None:
    """Drop and recreate the collection. `python -m app.ingest --reset`."""
    client = _client()
    name = _collection()
    if client.collection_exists(name):
        client.delete_collection(collection_name=name)
    # Dropping the collection drops its payload index too - forget that we
    # ever verified it, or the rebuilt collection is left without the ACL index.
    _ENSURED.discard(name)
    ensure_collection()
