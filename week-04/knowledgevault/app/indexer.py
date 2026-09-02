"""
Qdrant indexer - Option 1: single 3072-dim cosine vector per point.

Design decisions
----------------
we store exactly one vector per Qdrant point (the text embedding from 
``app/embedder.py``).  There are no named vectors.  This keeps the 
collection schema simple: one field, ``vector``, of size 3072 
with cosine distance.

Deterministic IDs via uuid5
----------------------------
Each chunk already has a deterministic ``chunk_id`` produced by the chunker -
a 16-hex-character SHA-256 prefix derived from (document_id, kind, index),
e.g. ``"9f2a4c81d3e5b760"``.  We convert that to a UUID with
``uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)``.  uuid5 is deterministic: the
same ``chunk_id`` always maps to the same UUID.  If you re-ingest the same
document, Qdrant's upsert will overwrite the existing point (same ID) rather
than create a duplicate.  No manual deduplication needed.

Batch upsert
------------
Qdrant recommends batching large upserts.  We use a fixed batch size of 100
points.  Progress is logged at INFO level so long-running ingestion jobs are
observable without turning on DEBUG.
"""

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.config import get_settings
from app.schemas import Chunk
from app.store import get_client

logger = logging.getLogger(__name__)

_VECTOR_SIZE = 3072
_BATCH_SIZE = 100


def _client() -> QdrantClient:
    """Return the Qdrant client for the configured mode (see app/store.py)."""
    return get_client()


def ensure_collection(reset: bool = False) -> None:
    """Create the Qdrant collection if it does not already exist.

    Parameters
    ----------
    reset:
        When ``True``, delete the collection first (drops all indexed data).
        Useful for re-indexing from scratch during development.
    """
    settings = get_settings()
    name = settings.qdrant_collection
    client = _client()

    if reset:
        try:
            client.delete_collection(name)
            logger.info("Collection %r deleted (reset=True).", name)
        except Exception:  # collection may not exist yet - that is fine
            logger.debug("Delete attempted for %r but it did not exist.", name)

    existing = [c.name for c in client.get_collections().collections]

    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Collection %r created (3072-dim cosine).", name)
    else:
        logger.info("Collection %r already exists.", name)


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Upsert *chunks* with their corresponding *vectors* into Qdrant.

    Parameters
    ----------
    chunks:
        Chunk objects produced by the chunker.  Their ``model_dump`` output
        becomes the Qdrant point payload, making every metadata field
        retrievable alongside the vector at query time.
    vectors:
        Parallel list of 3072-dim embeddings, one per chunk.

    Raises
    ------
    ValueError
        If ``len(chunks) != len(vectors)``.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks and vectors must have the same length, "
            f"got {len(chunks)} chunks and {len(vectors)} vectors."
        )

    settings = get_settings()
    name = settings.qdrant_collection
    client = _client()

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
            vector=vector,
            payload=chunk.model_dump(mode="json"),
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    total = len(points)
    upserted = 0

    for batch_start in range(0, total, _BATCH_SIZE):
        batch = points[batch_start : batch_start + _BATCH_SIZE]
        client.upsert(collection_name=name, points=batch)
        upserted += len(batch)
        logger.info("Upserted %d/%d points.", upserted, total)
