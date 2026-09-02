"""
Layout-aware PDF parser for the KnowledgeVault ingestion pipeline.

Two-stage pipeline:
  1. PyMuPDF (fitz) - opens the PDF for image extraction, text blocks, and
     figure cropping. Its coordinate system uses a *top-left origin*: y=0 is
     at the top of the page and increases downward - a deliberate departure
     from raw PDF/PostScript space, which is bottom-left with y increasing up.
  2. pdfplumber - opens the same PDF simultaneously for table detection. Its
     (x0, top, x1, bottom) bboxes are also *top-left origin*, so the two
     libraries agree on the y-axis direction.

The real gotcha is coordinates that arrive in raw PDF space - from PDF
annotations/metadata, or tools that report PDF-native coords - which need a
`page_height - y` flip before they can be mixed with fitz bboxes. All incoming
bboxes are routed through `_to_fitz_bbox` so there is one documented seam for
that normalisation; caller-visible output (ParsedBlock.bbox) is always in
PyMuPDF / fitz top-left space.

Processing order per page: tables → figures → text blocks.  Tables and figures
register their bounding boxes in a `taken` list; text blocks that spatially
overlap any taken region are skipped.

Public surface:
  parse_pdf(path: str) -> list[ParsedBlock]
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

from app.schemas import ParsedBlock

logger = logging.getLogger(__name__)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _to_fitz_bbox(
    plumber_bbox: tuple[float, float, float, float],
    page_height: float,
) -> tuple[float, float, float, float]:
    """
    Normalise an incoming bbox into PyMuPDF (fitz) space.

    pdfplumber gives (x0, top, x1, bottom) with a top-left origin - the same
    convention fitz uses (y=0 at the top, y increasing downward) - so for
    pdfplumber boxes this is a pass-through.

    The function exists as the single seam for coordinate normalisation: if a
    source ever supplies coordinates in raw PDF/PostScript space (bottom-left
    origin, y increasing upward), the `page_height - y` flip belongs here:

        fitz_y0 = page_height - pdf_native_y1
        fitz_y1 = page_height - pdf_native_y0

    `page_height` is accepted for that reason even though the pdfplumber
    conversion does not need it.
    """
    x0, top, x1, bottom = plumber_bbox
    return (x0, top, x1, bottom)


def _overlaps(
    bbox: tuple[float, float, float, float],
    taken: list[tuple[float, float, float, float]],
) -> bool:
    """Return True if bbox intersects any rect in taken (axis-aligned test)."""
    ax0, ay0, ax1, ay1 = bbox
    for bx0, by0, bx1, by1 in taken:
        if ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0:
            return True
    return False


# ── Figure helpers ────────────────────────────────────────────────────────────

def _save_figure(
    png_bytes: bytes,
    doc_id: str,
    page_num: int,
    xref: int,
) -> str:
    """
    Persist a PNG crop to data/figures/ and return the file path as a string.

    Filename: {doc_id}_p{page_num}_{xref}_{sha256[:10]}.png
    """
    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    sha = hashlib.sha256(png_bytes).hexdigest()[:10]
    filename = f"{doc_id}_p{page_num}_{xref}_{sha}.png"
    dest = out_dir / filename
    dest.write_bytes(png_bytes)
    return str(dest)


# ── Section heading heuristic ─────────────────────────────────────────────────

def _largest_text(page: fitz.Page) -> Optional[str]:
    """
    Return the text of the span with the largest font size on the page,
    provided its text is longer than 3 characters. Returns None if no
    qualifying span is found.
    """
    best_text: Optional[str] = None
    best_size: float = 0.0

    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size: float = span.get("size", 0.0)
                text: str = span.get("text", "").strip()
                if size > best_size and len(text) > 3:
                    best_size = size
                    best_text = text

    return best_text


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(path: str) -> list[ParsedBlock]:
    """
    Parse a PDF file into a flat list of ParsedBlock objects.

    Opens the file simultaneously with PyMuPDF (image/text extraction) and
    pdfplumber (table extraction). For each page, processes tables first, then
    figures, then text - each stage skipping regions already claimed by an
    earlier stage.

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        A list of ParsedBlock instances, one per detected region.
    """
    doc_id = Path(path).stem
    source_url = str(Path(path).resolve())
    blocks: list[ParsedBlock] = []

    fitz_doc = fitz.open(path)
    with pdfplumber.open(path) as plumber_doc:
        for page_idx in range(len(fitz_doc)):
            page_num = page_idx + 1
            fitz_page: fitz.Page = fitz_doc[page_idx]
            plumber_page = plumber_doc.pages[page_idx]

            page_height: float = fitz_page.rect.height
            taken: list[tuple[float, float, float, float]] = []
            section_heading = _largest_text(fitz_page)

            # ── 1. Tables ─────────────────────────────────────────────────────
            for tbl in plumber_page.find_tables():
                rows: list[list[str]] = [
                    [cell or "" for cell in row]
                    for row in (tbl.extract() or [])
                ]
                if not rows:
                    continue

                bbox = _to_fitz_bbox(tbl.bbox, page_height)
                taken.append(bbox)

                blocks.append(
                    ParsedBlock(
                        document_id=doc_id,
                        source_url=source_url,
                        page_number=page_num,
                        block_type="table",
                        table_rows=rows,
                        bbox=bbox,
                        section_heading=section_heading,
                    )
                )

            # ── 2. Figures ────────────────────────────────────────────────────
            for img_info in fitz_page.get_images(full=True):
                xref: int = img_info[0]
                rects = fitz_page.get_image_rects(xref)
                for rect in rects:
                    w = rect.width
                    h = rect.height
                    if w < 80 or h < 80:
                        continue

                    png_bytes: bytes = fitz_page.get_pixmap(
                        clip=rect, dpi=150
                    ).tobytes("png")

                    image_path = _save_figure(png_bytes, doc_id, page_num, xref)
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                    taken.append(bbox)

                    blocks.append(
                        ParsedBlock(
                            document_id=doc_id,
                            source_url=source_url,
                            page_number=page_num,
                            block_type="figure",
                            image_path=image_path,
                            bbox=bbox,
                            section_heading=section_heading,
                        )
                    )

            # ── 3. Text blocks ────────────────────────────────────────────────
            for raw in fitz_page.get_text("blocks"):
                # raw = (x0, y0, x1, y1, text, block_no, block_type_int)
                x0, y0, x1, y1, text, _block_no, block_type_int = raw

                if block_type_int != 0:
                    continue
                if len(text.strip()) < 20:
                    continue

                bbox = (x0, y0, x1, y1)
                if _overlaps(bbox, taken):
                    continue

                blocks.append(
                    ParsedBlock(
                        document_id=doc_id,
                        source_url=source_url,
                        page_number=page_num,
                        block_type="text",
                        text=text.strip(),
                        bbox=bbox,
                        section_heading=section_heading,
                    )
                )

    fitz_doc.close()

    logger.info("parse_pdf: %s -> %d blocks", Path(path).name, len(blocks))
    return blocks
