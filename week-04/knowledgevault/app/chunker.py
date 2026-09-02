"""
KnowledgeVault chunker - three strategies, one coherent metadata bundle.

Chunking strategy overview
---------------------------
Every chunk that comes out of this module carries the same metadata bundle
(document_id, source_url, section_heading, page_number, chunk_index,
created_at, figure_ids) so the retriever never has to treat chunk types
differently when assembling context.

Figure chunks (chunk_type="figure-description")
    One chunk per figure block. The vision LLM has already described the
    image in structured form (FigureDescription). We flatten that structure
    into a short text string - "[type] summary.  Key elements: ..." - and
    store it as the chunk's text field. The original image_path is preserved so the
    API layer can serve the actual image alongside the retrieved chunk.

Table chunks (chunk_type="table-row")
    One chunk per data row. The first row of a table block is treated as
    a header; every subsequent non-empty row is serialised as
    "header_col: value | header_col: value …" and emitted as its own chunk.
    Row-level granularity keeps individual facts retrievable without
    dragging in the whole table.

Prose chunks (chunk_type="prose")
    Fixed-size sliding windows over token-encoded text. Window size is
    _CHILD_TOKENS (300) tokens with _OVERLAP (50) token overlap between
    adjacent windows. Token counting uses tiktoken's cl100k_base encoding
    (the same vocabulary used by text-embedding-3-* models) so chunk sizes
    are exact in the embedding model's token space.

Hierarchy - children nested inside parents
    Prose is chunked hierarchically. All of a document's text blocks are
    joined, in reading order, into one token stream. Every 300-token child
    window over that stream is assigned to the _PARENT_TOKENS (1,500)
    token parent span that contains the window's first token, and carries
    that span's identity in Chunk.parent_id. Parents are laid over the
    whole document, not over individual blocks, so a parent spans
    paragraphs and page breaks and holds six children (1,500 / 250 step).
    Each child still records the page and section heading of the block its
    first token came from.

    Only the children are embedded and indexed - that is the "single
    embedder slice" the concept video describes. The parent is an
    identity, not a vector: it is what lets the retriever match on a
    small, precise child and then widen to the full parent context by
    grouping every chunk that shares the same parent_id. Small chunks
    retrieve well; large chunks read well. The parent_id is the seam
    between the two.

Figure-ID cross-linking
    Figure chunks collect their chunk_id into a per-page lookup. Table and
    prose chunks on the same page inherit that list as figure_ids, letting
    the retriever optionally surface related images when returning a prose
    or table hit.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import tiktoken

from app.schemas import Chunk, FigureDescription, ParsedBlock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Encoding / window constants
# ---------------------------------------------------------------------------

_CHILD_TOKENS: int = 300
_OVERLAP: int = 50
_PARENT_TOKENS: int = 1_500

# tiktoken.get_encoding() DOWNLOADS the BPE table on first use and caches it on
# disk. Calling it at import time made this module un-importable on an offline
# box or a network-less CI runner - `import app.chunker` would hard-fail before
# a single test ran. The encoder is therefore built lazily, on first tokenisation,
# and memoised. Importing the module is now pure-local and always succeeds; only
# code that actually chunks text needs the table (and once it is in
# TIKTOKEN_CACHE_DIR, that works offline too).
_ENC: Optional["tiktoken.Encoding"] = None


def _enc() -> "tiktoken.Encoding":
    """Return the memoised cl100k_base encoder, building it on first call."""
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_id(doc_id: str, kind: str, idx: int) -> str:
    """Return a 16-hex-char content-addressed ID for a chunk."""
    return hashlib.sha256(f"{doc_id}:{kind}:{idx}".encode()).hexdigest()[:16]


def _split_stream(
    pieces: list[tuple[str, int, Optional[str]]],
) -> list[tuple[str, int, int, Optional[str]]]:
    """Tokenise a document's prose as one stream and slide a fixed window over it.

    Parameters
    ----------
    pieces:
        (text, page_number, section_heading) for every text block, in reading
        order. They are joined with a blank line into a single token stream so
        that parent spans cross block and page boundaries.

    Returns
    -------
    list[tuple[str, int, int, Optional[str]]]
        (window_text, parent_ordinal, page_number, section_heading) per child.
        parent_ordinal is start // _PARENT_TOKENS over the whole stream, so
        consecutive children share a parent until the window crosses a
        1,500-token boundary: six children per parent at a 250-token step.
        page_number and section_heading are those of the block that owns the
        window's first token.
    """
    enc = _enc()
    tokens: list[int] = []
    # For each block, the stream offset where it starts, so a token position
    # can be mapped back to the block (and therefore page/heading) it came from.
    starts: list[tuple[int, int, Optional[str]]] = []
    sep = enc.encode("\n\n")
    for text, page, heading in pieces:
        if not text:
            continue
        if tokens:
            tokens.extend(sep)
        starts.append((len(tokens), page, heading))
        tokens.extend(enc.encode(text))
    if not tokens:
        return []

    def owner(pos: int) -> tuple[int, Optional[str]]:
        page, heading = starts[0][1], starts[0][2]
        for s0, p, h in starts:
            if s0 <= pos:
                page, heading = p, h
            else:
                break
        return page, heading

    step = _CHILD_TOKENS - _OVERLAP
    windows: list[tuple[str, int, int, Optional[str]]] = []
    start = 0
    while start < len(tokens):
        chunk_tokens = tokens[start : start + _CHILD_TOKENS]
        page, heading = owner(start)
        windows.append(
            (enc.decode(chunk_tokens), start // _PARENT_TOKENS, page, heading)
        )
        start += step
    return windows


def _split_text(text: str) -> list[tuple[str, int]]:
    """Single-block convenience wrapper kept for tests; see _split_stream."""
    return [(t, p) for t, p, _pg, _h in _split_stream([(text, 0, None)])]


def _figure_text(desc: FigureDescription) -> str:
    """Serialise a FigureDescription into a single embeddable string."""
    text = f"[{desc.type}] {desc.summary}"
    if desc.key_elements:
        text += " Key elements: " + ", ".join(desc.key_elements)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    blocks: list[ParsedBlock],
    figure_descriptions: dict[str, Optional[FigureDescription]],
) -> list[Chunk]:
    """Convert a list of ParsedBlocks into a flat list of Chunks.

    Parameters
    ----------
    blocks:
        All parsed regions for a single document, in page/reading order.
    figure_descriptions:
        Mapping from image_path → FigureDescription (or None if the vision
        LLM dropped/failed that figure).

    Returns
    -------
    list[Chunk]
        Ordered list of chunks: figures first, then tables, then prose.
        Each chunk carries a globally unique chunk_id derived from
        (document_id, kind, global_index).
    """
    chunks: list[Chunk] = []
    global_idx: int = 0

    # Per-page list of figure chunk_ids, populated in Pass 1 and consumed
    # in Passes 2 & 3 so sibling chunks can reference related figures.
    figure_ids_by_page: dict[int, list[str]] = defaultdict(list)

    now = datetime.now(tz=timezone.utc)

    # ------------------------------------------------------------------
    # Pass 1 - Figure chunks
    # ------------------------------------------------------------------
    for block in blocks:
        if block.block_type != "figure":
            continue
        if block.image_path is None:
            continue

        desc = figure_descriptions.get(block.image_path)
        if desc is None:
            # Vision LLM dropped this figure; skip entirely.
            continue

        text = _figure_text(desc)
        chunk_id = _make_id(block.document_id, "figure", global_idx)

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                parent_id=None,
                document_id=block.document_id,
                source_url=block.source_url,
                section_heading=block.section_heading,
                page_number=block.page_number,
                chunk_index=global_idx,
                created_at=now,
                chunk_type="figure-description",
                text=text,
                image_path=block.image_path,
                figure_ids=[],
            )
        )
        figure_ids_by_page[block.page_number].append(chunk_id)
        global_idx += 1

    # ------------------------------------------------------------------
    # Pass 2 - Table chunks (one chunk per data row)
    # ------------------------------------------------------------------
    for block in blocks:
        if block.block_type != "table":
            continue
        if block.table_rows is None:
            continue

        rows = block.table_rows
        if len(rows) < 2:
            # Need at least one header row + one data row.
            continue

        headers = [str(h or "").strip() for h in rows[0]]

        for row in rows[1:]:
            # Skip rows where every cell is empty.
            if all(not str(v or "").strip() for v in row):
                continue

            text = " | ".join(
                f"{h}: {str(v or '').strip()}"
                for h, v in zip(headers, row)
                if str(v or "").strip()
            )

            if not text.strip():
                continue

            chunk_id = _make_id(block.document_id, "table", global_idx)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    parent_id=None,
                    document_id=block.document_id,
                    source_url=block.source_url,
                    section_heading=block.section_heading,
                    page_number=block.page_number,
                    chunk_index=global_idx,
                    created_at=now,
                    chunk_type="table-row",
                    text=text,
                    image_path=None,
                    figure_ids=figure_ids_by_page.get(block.page_number, []),
                )
            )
            global_idx += 1

    # ------------------------------------------------------------------
    # Pass 3 - Prose chunks (300-token child windows, 50-token overlap),
    #          each nested inside a 1,500-token parent span.
    #
    # The parent is minted, not emitted: we do not embed it. Children carry
    # parent_id, so the retriever can match a precise 300-token child and
    # then widen to the whole 1,500-token parent by grouping on that id.
    # Parents are laid over the document's whole prose stream, so they
    # cross paragraph and page boundaries and hold about six children.
    # ------------------------------------------------------------------
    pieces: list[tuple[str, int, Optional[str]]] = [
        (b.text, b.page_number, b.section_heading)
        for b in blocks
        if b.block_type == "text" and b.text
    ]
    doc_id = blocks[0].document_id if blocks else ""
    source_url = next((b.source_url for b in blocks if b.block_type == "text"), "")
    parent_ids: dict[int, str] = {}

    for child_text, parent_ordinal, page, heading in _split_stream(pieces):
        parent_id = parent_ids.get(parent_ordinal)
        if parent_id is None:
            parent_id = _make_id(doc_id, "parent", parent_ordinal)
            parent_ids[parent_ordinal] = parent_id

        chunk_id = _make_id(doc_id, "prose", global_idx)

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                parent_id=parent_id,
                document_id=doc_id,
                source_url=source_url,
                section_heading=heading,
                page_number=page,
                chunk_index=global_idx,
                created_at=now,
                chunk_type="prose",
                text=child_text,
                image_path=None,
                figure_ids=figure_ids_by_page.get(page, []),
            )
        )
        global_idx += 1

    # ------------------------------------------------------------------
    # Summary log
    # ------------------------------------------------------------------
    n_fig = sum(1 for c in chunks if c.chunk_type == "figure-description")
    n_tbl = sum(1 for c in chunks if c.chunk_type == "table-row")
    n_prose = sum(1 for c in chunks if c.chunk_type == "prose")
    n_parents = len({c.parent_id for c in chunks if c.parent_id is not None})
    total = len(chunks)

    logger.info(
        "chunk_document: %d chunks (%d figure, %d table, %d prose "
        "nested in %d parents of %d tokens)",
        total,
        n_fig,
        n_tbl,
        n_prose,
        n_parents,
        _PARENT_TOKENS,
    )

    return chunks
