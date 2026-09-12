"""Retrieval behind external and episodic memory.

A hashed bag-of-words embedder and an exact cosine scan. No server, no network,
no bill, and the same vector for the same text on every machine.

=============================================================================
 THIS FILE CONTAINS THE ONE DELIBERATE DEFECT IN THIS SCAFFOLD.
 `InProcessVectorStore.search` ranks BEFORE it filters. Fixing that is task 1.
 Run `pytest -q tests/test_scope.py` to see the three properties it breaks.
=============================================================================
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Protocol

DIMS = 256


def embed(text: str) -> list[float]:
    """Hashed bag-of-words with sublinear term frequency and an L2 norm.

    Not a sentence transformer, and a genuine embedding: same text gives the same
    vector, similar text gives nearby vectors, and the whole suite stays offline
    and deterministic. Swapping in a real embedding model is a one-function change.
    """
    counts: dict[int, float] = {}
    for tok in "".join(c.lower() if c.isalnum() else " " for c in text).split():
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "big")
        counts[h % DIMS] = counts.get(h % DIMS, 0.0) + 1.0
    vec = [0.0] * DIMS
    for i, c in counts.items():
        vec[i] = 1.0 + math.log(c)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorStore(Protocol):
    def add(self, scope: str, collection: str, chunks: list[dict[str, Any]]) -> None: ...
    def search(self, query: str, *, scope: str, collection: str, k: int) -> list[dict[str, Any]]: ...
    def delete_scope(self, scope: str, collection: str | None = None) -> int: ...
    def count(self, scope: str, collection: str | None = None) -> int: ...


class InProcessVectorStore:
    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def add(self, scope: str, collection: str, chunks: list[dict[str, Any]]) -> None:
        assert scope, "scope filter required"
        for ch in chunks:
            row = dict(ch)
            row["scope"] = scope
            row["collection"] = collection
            row["vec"] = embed(ch["text"])
            self._rows = [r for r in self._rows if r["chunk_id"] != row["chunk_id"]]
            self._rows.append(row)

    # -------------------------------------------------------------- TASK 1
    def search(self, query: str, *, scope: str, collection: str, k: int) -> list[dict[str, Any]]:
        """Return up to k chunks from (scope, collection), most similar first.

        TASK 1. The implementation below is WRONG and three tests say so.

        It ranks every row in the store against the query, takes the top k across
        ALL scopes and collections, and only then discards the rows that do not
        belong to this caller. That is post-filtering, and it is what pgvector
        does with an approximate index unless you configure it not to. Read
        W10-R03 section 5, then make the filter run first.

        ONE property breaks, and being precise about which one is the project.

        What does NOT break: no row belonging to another user can reach this
        caller. Filtering after ranking still filters, so the discard at the end
        is exactly as effective as a filter at the start would have been. The
        scaffold is not leaking and never was, and there is a test that passes on
        arrival to pin that.

        What DOES break: recall. The ranking considers rows this caller is not
        entitled to, those rows consume the k slots, and the filter then throws
        them away. A caller with plenty of rows gets fewer than k back, or none,
        with nothing erroring anywhere. That failure looks like an absence of
        data rather than the presence of a bug, which is why it survives review.

        Three tests fail on arrival and all three are recall tests. The fix is
        two lines. Understanding why it is two lines, and why it is not a leak,
        is the project - and the design memo asks you which one this was.
        """
        assert scope, "scope filter required"
        q = embed(query)
        scored = [
            {"chunk_id": r["chunk_id"], "source": r["source"], "text": r["text"],
             "scope": r["scope"], "collection": r["collection"], "score": cosine(q, r["vec"])}
            for r in self._rows
        ]
        scored.sort(key=lambda d: d["score"], reverse=True)
        top = scored[:k]
        return [d for d in top if d["scope"] == scope and d["collection"] == collection]
    # ---------------------------------------------------------- END TASK 1

    def delete_scope(self, scope: str, collection: str | None = None) -> int:
        before = len(self._rows)
        self._rows = [r for r in self._rows
                      if not (r["scope"] == scope and (collection is None or r["collection"] == collection))]
        return before - len(self._rows)

    def count(self, scope: str, collection: str | None = None) -> int:
        return sum(1 for r in self._rows
                   if r["scope"] == scope and (collection is None or r["collection"] == collection))


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = InProcessVectorStore()
    return _store


def reset_store() -> None:
    global _store
    _store = None
