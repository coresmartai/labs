"""Turn an RFC's plain text into chunks, at runtime.

The corpus is never modified on disk. Chunking happens here, in memory, every
time the index is built. That is a licence requirement rather than a style
choice: see corpus/fetch.py.

RFC text is fixed-width ASCII with page furniture. Each page ends with a form
feed and carries a running footer like

    Kitterman                    Standards Track                   [Page 12]

and a header on the next line. Left in, that furniture lands inside chunks and
pollutes both retrieval and quotes. Stripping it is the whole of this file.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.schemas import Chunk

_FOOTER = re.compile(r"^.*\[Page \d+\]\s*$")
_HEADER = re.compile(r"^RFC \d+\s+.*\s+\w+ \d{4}\s*$")
_SECTION = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")

CHUNK_LINES = 28
OVERLAP_LINES = 6


def _clean(raw: str) -> list[tuple[int, str]]:
    """Return (page_number, line) with page furniture removed."""
    out: list[tuple[int, str]] = []
    page = 1
    for line in raw.replace("\r\n", "\n").split("\n"):
        stripped = line.rstrip()
        if "\f" in stripped:
            page += 1
            stripped = stripped.replace("\f", "").rstrip()
            if not stripped:
                continue
        if _FOOTER.match(stripped) or _HEADER.match(stripped):
            continue
        out.append((page, stripped))
    return out


def chunk_rfc(path: Path) -> list[Chunk]:
    """One RFC to a list of chunks with stable uids.

    `chunk_uid` is `<rfc>:<ordinal>`, and the ordinal is the chunk's position in
    this document. It does not change between runs, which is what lets a ledger
    id resolve to the same passage tomorrow.
    """
    rfc = path.stem
    lines = _clean(path.read_text(encoding="utf-8", errors="replace"))
    chunks: list[Chunk] = []
    section = "front matter"
    i = 0
    ordinal = 0
    while i < len(lines):
        window = lines[i:i + CHUNK_LINES]
        if not window:
            break
        for _, text in window:
            m = _SECTION.match(text)
            if m:
                section = f"{m.group(1)} {m.group(2)}".strip()
                break
        body = "\n".join(t for _, t in window).strip()
        if body:
            chunks.append(Chunk(
                chunk_uid=f"{rfc}:{ordinal:04d}",
                rfc=rfc,
                section=section,
                page=window[0][0],
                text=body,
            ))
            ordinal += 1
        i += CHUNK_LINES - OVERLAP_LINES
    return chunks


def chunk_corpus(rfc_dir: Path) -> list[Chunk]:
    files = sorted(rfc_dir.glob("*.txt"))
    if not files:
        raise SystemExit(
            f"No RFCs in {rfc_dir}. Run: python corpus/fetch.py"
        )
    out: list[Chunk] = []
    for f in files:
        out.extend(chunk_rfc(f))
    return out
