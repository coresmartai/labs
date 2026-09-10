"""BM25 over the chunked corpus. Deterministic, local, free.

No embeddings and no vector database, deliberately. Retrieval quality is not
what this project grades, and a deterministic retriever means your reviewer
reproduces your hits exactly. It also means the whole thing runs with no key.
"""
from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.chunker import chunk_corpus
from app.schemas import Chunk, SearchHit

_TOKEN = re.compile(r"[a-z0-9_\-]+")


def _tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class CorpusIndex:
    def __init__(self, rfc_dir: Path):
        self.chunks: list[Chunk] = chunk_corpus(rfc_dir)
        self._bm25 = BM25Okapi([_tokenise(c.text) for c in self.chunks])
        self._by_uid = {c.chunk_uid: c for c in self.chunks}

    @property
    def documents(self) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            if c.rfc not in seen:
                seen.append(c.rfc)
        return seen

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        scores = self._bm25.get_scores(_tokenise(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            SearchHit(rank=rank, score=float(scores[i]), chunk=self.chunks[i])
            for rank, i in enumerate(order, start=1)
        ]

    def get(self, chunk_uid: str) -> Chunk | None:
        return self._by_uid.get(chunk_uid)
