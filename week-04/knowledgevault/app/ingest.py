"""
KnowledgeVault ingestion pipeline - five sequential stages that turn a raw PDF into
searchable vector chunks stored in Qdrant.

Pipeline stages
---------------
1. **Parse** - ``parse_pdf`` extracts structured blocks (prose, figure, table) from the PDF.
2. **Describe** - ``describe_figure`` calls a vision model for every figure block that has an
   associated PNG on disk.  Blocks whose description cannot be produced (confidence too low,
   file missing, API error) are dropped and counted separately.
3. **Chunk** - ``chunk_document`` splits the blocks into overlapping text windows, attaching
   figure descriptions and table text as dedicated chunk types.
4. **Embed** - ``embed_text`` converts each chunk's text into a dense vector using the
   configured embedding model.
5. **Index** - ``ensure_collection`` creates (or resets) the Qdrant collection and
   ``upsert_chunks`` writes all chunk payloads and vectors in a single batch.

CLI usage
---------
    python -m app.ingest --pdf path/to/document.pdf
    python -m app.ingest --pdf path/to/document.pdf --reset

HTTP usage
----------
``run_ingest`` is also called directly by the ``POST /ingest`` FastAPI endpoint so that the
same pipeline logic is reused without spawning a subprocess.
"""

import argparse
import logging
import sys
from pathlib import Path

from app.chunker import chunk_document
from app.config import get_settings
from app.embedder import embed_text
from app.indexer import ensure_collection, upsert_chunks
from app.parser import parse_pdf
from app.schemas import FigureDescription
from app.vision import describe_figure

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


def run_ingest(pdf_path: str, reset: bool = False) -> dict:
    """Execute the full five-stage ingestion pipeline for *pdf_path*.

    Parameters
    ----------
    pdf_path:
        Absolute or relative path to the PDF file to ingest.
    reset:
        When ``True``, the Qdrant collection is dropped and recreated before
        upserting, ensuring a clean slate.

    Returns
    -------
    dict
        Summary statistics with keys ``pdf``, ``blocks``, ``chunks``,
        ``figures_described``, and ``figures_dropped``.
    """

    # ── Stage 1: Parse PDF ───────────────────────────────────────────────────
    logger.info("── Stage 1: parsing %s", pdf_path)
    blocks = parse_pdf(pdf_path)
    text_blocks = [b for b in blocks if b.block_type == "text"]
    figure_blocks = [b for b in blocks if b.block_type == "figure"]
    table_blocks = [b for b in blocks if b.block_type == "table"]
    logger.info(
        "── Stage 1 done: %d blocks total (%d text, %d figure, %d table)",
        len(blocks),
        len(text_blocks),
        len(figure_blocks),
        len(table_blocks),
    )

    # ── Stage 2: Describe figures ────────────────────────────────────────────
    logger.info("── Stage 2: describing %d figure(s)", len(figure_blocks))
    figure_descriptions: dict[str, FigureDescription | None] = {}
    for i, block in enumerate(figure_blocks, start=1):
        image_path = getattr(block, "image_path", None)
        if not image_path:
            continue
        try:
            with open(image_path, "rb") as fh:
                png_bytes = fh.read()
            description = describe_figure(png_bytes)
            figure_descriptions[image_path] = description
            confidence = description.confidence if description else "-"
            logger.info(
                "── Stage 2 [%d/%d] %s  confidence=%s",
                i,
                len(figure_blocks),
                image_path,
                confidence,
            )
        except Exception:
            logger.info(
                "── Stage 2 [%d/%d] %s  dropped",
                i,
                len(figure_blocks),
                image_path,
            )
            figure_descriptions[image_path] = None

    figures_described = sum(1 for v in figure_descriptions.values() if v is not None)
    figures_dropped = sum(1 for v in figure_descriptions.values() if v is None)

    # ── Stage 3: Chunk document ──────────────────────────────────────────────
    logger.info("── Stage 3: chunking")
    chunks = chunk_document(blocks, figure_descriptions)
    logger.info("── Stage 3 done: %d chunks", len(chunks))

    # ── Stage 4: Embed chunks ────────────────────────────────────────────────
    logger.info("── Stage 4: embedding %d chunks", len(chunks))
    vectors = []
    for i, chunk in enumerate(chunks, start=1):
        vectors.append(embed_text(chunk.text))
        if i % 10 == 0:
            logger.info("── Stage 4 [%d/%d] embedded", i, len(chunks))
    logger.info("── Stage 4 done: %d vectors produced", len(vectors))

    # ── Stage 5: Index into Qdrant ───────────────────────────────────────────
    logger.info("── Stage 5: indexing (reset=%s)", reset)
    ensure_collection(reset=reset)
    upsert_chunks(chunks, vectors)
    logger.info("── Stage 5 done: %d points upserted", len(chunks))

    return {
        "pdf": pdf_path,
        "blocks": len(blocks),
        "chunks": len(chunks),
        "figures_described": figures_described,
        "figures_dropped": figures_dropped,
    }


def _main() -> None:
    """Argparse CLI entry point for running ingestion from the command line."""
    parser = argparse.ArgumentParser(description="Ingest a PDF into KnowledgeVault.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF to ingest.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting.",
    )
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        logger.error("File not found: %s", args.pdf)
        sys.exit(1)

    run_ingest(args.pdf, reset=args.reset)


if __name__ == "__main__":
    _main()
