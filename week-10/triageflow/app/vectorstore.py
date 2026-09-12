"""The vector store behind external and episodic memory.

Two backends, one interface. Both are REAL vector search - embeddings plus
cosine similarity plus a (scope, collection) metadata filter. What changes is
where the vectors live.

  * InProcessVectorStore  - the DEFAULT (MEMORY_BACKEND=memory).
      Embed at import, cosine top-k in process. No server, no network, no bill,
      and the tests run offline and deterministically. The corpus is canned;
      the retrieval is not.

  * PgVectorStore         - MEMORY_BACKEND=pgvector.
      A real PostgreSQL 16 with the `vector` extension, started from a
      pip-installed embedded binary (`pgserver`). NO DOCKER, no daemon, no
      compose file, no admin rights. Same interface, same call sites.

The scope argument is not decoration. Every write asserts it and every read
passes it as a metadata filter - a query scoped to `user:u_1` can never see
`user:u_2`'s rows. Multi-tenant leaks are silent; the scope filter is the only
thing that catches them.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from typing import Any, Protocol

log = logging.getLogger(__name__)

GLOBAL_SCOPE = "global"


def user_scope(user_id: str) -> str:
    """The scope string for one user's private memory."""
    assert user_id, "scope filter required"
    return f"user:{user_id}"


# ----------------------------- the embedder -----------------------------
# A deterministic, offline, dependency-free embedder: hashed bag-of-words with
# sublinear term frequency and an L2 norm. It is not sentence-transformers, but
# it is a genuine vector embedding - same text always yields the same vector,
# similar text yields nearby vectors, and cosine similarity behaves. Swap in
# real embeddings by setting EMBEDDINGS=openai; the store never notices.

_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def embed(text: str) -> list[float]:
    """Text -> a unit-length vector of _DIM floats. Deterministic across runs."""
    vec = [0.0] * _DIM
    for token, n in Counter(_tokens(text)).items():
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        # Signed hashing: two features colliding on an index cancel as often as
        # they compound, which keeps the collision noise zero-mean.
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[h % _DIM] += sign * (1.0 + math.log(n))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Both vectors are unit-length, so the dot product IS the cosine."""
    return sum(x * y for x, y in zip(a, b))


# ----------------------------- the interface -----------------------------

class VectorStore(Protocol):
    def add(self, scope: str, collection: str, chunks: list[dict[str, Any]]) -> None: ...

    def search(self, query: str, *, scope: str, collection: str, k: int) -> list[dict[str, Any]]: ...

    def delete_scope(self, scope: str, collection: str | None = None) -> int: ...

    def count(self, scope: str, collection: str | None = None) -> int: ...


# ----------------------------- in-process backend -----------------------------

class InProcessVectorStore:
    """Real cosine similarity search over an in-process corpus."""

    backend = "memory"

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def add(self, scope: str, collection: str, chunks: list[dict[str, Any]]) -> None:
        assert scope, "scope filter required"
        for c in chunks:
            row = dict(c)
            row["scope"] = scope
            row["collection"] = collection
            row["embedding"] = embed(c["text"])
            # Idempotent on chunk_id within a (scope, collection).
            self._rows = [
                r for r in self._rows
                if not (r["chunk_id"] == row["chunk_id"]
                        and r["scope"] == scope and r["collection"] == collection)
            ]
            self._rows.append(row)

    def search(self, query: str, *, scope: str, collection: str, k: int) -> list[dict[str, Any]]:
        assert scope, "scope filter required"
        q = embed(query)
        # The metadata filter runs BEFORE the similarity search. This is the
        # whole ballgame for multi-tenancy.
        candidates = [r for r in self._rows
                      if r["scope"] == scope and r["collection"] == collection]
        scored = [
            {"chunk_id": r["chunk_id"], "source": r["source"], "text": r["text"],
             "score": round(cosine(q, r["embedding"]), 4)}
            for r in candidates
        ]
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:k]

    def delete_scope(self, scope: str, collection: str | None = None) -> int:
        before = len(self._rows)
        self._rows = [
            r for r in self._rows
            if not (r["scope"] == scope and (collection is None or r["collection"] == collection))
        ]
        return before - len(self._rows)

    def count(self, scope: str, collection: str | None = None) -> int:
        return sum(1 for r in self._rows
                   if r["scope"] == scope
                   and (collection is None or r["collection"] == collection))


# ----------------------------- pgvector backend -----------------------------

class PgVectorStore:
    """Real Postgres + pgvector, from a pip wheel. No Docker.

    `pgserver` ships PostgreSQL 16.2 binaries and the pgvector 0.6.2 extension
    inside the wheel, for Linux, macOS and Windows. get_server() runs initdb on
    first use and starts a postmaster owned by this process - no root, no
    daemon, no container.
    """

    backend = "pgvector"

    def __init__(self, data_dir: str) -> None:
        import pathlib

        import pgserver  # noqa: F401  - only imported on the pgvector path
        import psycopg

        self._psycopg = psycopg
        pgdata = pathlib.Path(data_dir)
        pgdata.mkdir(parents=True, exist_ok=True)
        self._server = pgserver.get_server(pgdata)
        self._uri = self._server.get_uri()
        log.info("pgvector backend up: %s", self._uri)
        with self._conn() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute(
                f"""CREATE TABLE IF NOT EXISTS memory_chunks (
                       chunk_id   text NOT NULL,
                       scope      text NOT NULL,
                       collection text NOT NULL,
                       source     text NOT NULL,
                       text       text NOT NULL,
                       embedding  vector({_DIM}) NOT NULL,
                       PRIMARY KEY (chunk_id, scope, collection))"""
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS memory_chunks_embedding_idx "
                "ON memory_chunks USING hnsw (embedding vector_cosine_ops)"
            )

    def _conn(self):
        return self._psycopg.connect(self._uri, autocommit=True)

    @staticmethod
    def _lit(v: list[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

    def add(self, scope: str, collection: str, chunks: list[dict[str, Any]]) -> None:
        assert scope, "scope filter required"
        with self._conn() as c:
            for ch in chunks:
                c.execute(
                    """INSERT INTO memory_chunks
                          (chunk_id, scope, collection, source, text, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (chunk_id, scope, collection) DO UPDATE
                          SET text = EXCLUDED.text, embedding = EXCLUDED.embedding""",
                    (ch["chunk_id"], scope, collection, ch["source"], ch["text"],
                     self._lit(embed(ch["text"]))),
                )

    def search(self, query: str, *, scope: str, collection: str, k: int) -> list[dict[str, Any]]:
        assert scope, "scope filter required"
        q = self._lit(embed(query))
        with self._conn() as c:
            # `<=>` is pgvector's cosine-distance operator.
            #
            # The WHERE clause looks like the in-process store's filter and does
            # NOT behave like it. The in-process store filters BEFORE it ranks.
            # pgvector with an approximate index filters AFTER the index scan:
            # the index picks its candidates across the whole table, then this
            # WHERE discards the ones belonging to other scopes. Under a
            # selective per-user filter that silently returns fewer than k rows,
            # or none, with nothing erroring. Not a leak - a recall failure.
            #
            # On THIS build the two backends do agree, and the reason is not the
            # one you would guess. There IS an hnsw index on this table, created
            # in __init__ a few lines up. At eight rows the planner ignores it
            # and does a sequential scan with the filter applied, which is exact.
            # Check it yourself with EXPLAIN ANALYZE and you will see `Seq Scan
            # on memory_chunks` and `Rows Removed by Filter`.
            #
            # So "I created an index" and "my query uses the index" are two
            # different facts, and the demo corpus is too small for the second
            # one to become true. It becomes true at production scale, without
            # any code changing. Then: set `hnsw.iterative_scan` (pgvector
            # 0.8.0+), or partition by scope, or filter in a subquery you
            # over-fetch from.
            rows = c.execute(
                """SELECT chunk_id, source, text, 1 - (embedding <=> %s) AS score
                     FROM memory_chunks
                    WHERE scope = %s AND collection = %s
                 ORDER BY embedding <=> %s
                    LIMIT %s""",
                (q, scope, collection, q, k),
            ).fetchall()
        return [{"chunk_id": r[0], "source": r[1], "text": r[2], "score": round(r[3], 4)}
                for r in rows]

    def delete_scope(self, scope: str, collection: str | None = None) -> int:
        sql = "DELETE FROM memory_chunks WHERE scope = %s"
        params: list[Any] = [scope]
        if collection is not None:
            sql += " AND collection = %s"
            params.append(collection)
        with self._conn() as c:
            return c.execute(sql, params).rowcount

    def count(self, scope: str, collection: str | None = None) -> int:
        sql = "SELECT count(*) FROM memory_chunks WHERE scope = %s"
        params: list[Any] = [scope]
        if collection is not None:
            sql += " AND collection = %s"
            params.append(collection)
        with self._conn() as c:
            return int(c.execute(sql, params).fetchone()[0])


# ----------------------------- the factory -----------------------------

_store: VectorStore | None = None


def get_store() -> VectorStore:
    """One store per process, chosen by MEMORY_BACKEND. Default: in-process."""
    global _store
    if _store is None:
        from .config import get_settings
        backend = get_settings().memory_backend
        if backend == "pgvector":
            _store = PgVectorStore(get_settings().pgvector_data_dir)
        else:
            _store = InProcessVectorStore()
        log.info("vectorstore backend=%s", _store.backend)
    return _store


def reset_store() -> None:
    """Test hook - drop the cached store so the next get_store() rebuilds it."""
    global _store
    _store = None
