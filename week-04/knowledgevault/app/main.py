"""
FastAPI entrypoint for KnowledgeVault.

Routes:
  GET  /            - serves index.html (browser UI)
  GET  /health      - liveness probe (no API calls)
  GET  /readme      - renders README.md as dark-themed HTML
  GET  /figures/    - static mount for extracted figure images
  GET  /pdfs        - lists PDFs in data/sample/
  GET  /collection  - Qdrant collection stats (name, point count)
  GET  /stats       - per-document chunk counts by type (prose / figure / table)
  POST /upload      - upload a PDF to data/sample/ (multipart)
  POST /ingest      - ingest a single PDF end-to-end
  POST /ingest/all  - ingest every PDF in data/sample/
  POST /retrieve    - hybrid retrieval over the indexed corpus

Run:
    uvicorn app.main:app --reload
"""

from __future__ import annotations
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.schemas import RetrieveRequest, RetrieveResponse, RetrievedChunk


settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("knowledgevault")

# Ensure output directories exist before mounting
_FIGURES_DIR = Path("data/figures")
_DATA_DIR    = Path(get_settings().data_dir)
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="KnowledgeVault",
    version="0.1.0",
    description="Week 4 - multimodal ingestion + hybrid retrieval over engineering PDFs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted figure images at /figures/<filename>
app.mount("/figures", StaticFiles(directory=str(_FIGURES_DIR)), name="figures")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    """Serve the browser UI (index.html)."""
    ui = Path("index.html")
    if not ui.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found - run uvicorn from the knowledgevault/ folder, the one that contains app/ and index.html.",
        )
    return FileResponse(str(ui))


@app.get("/health")
def health() -> dict[str, str]:
    """
    Liveness probe. Returns 200 if .env loaded and the app started.
    """
    return {
        "status": "ok",
        "model": settings.openai_vision_model,           # used by CORE health chip
        "vision_model": settings.openai_vision_model,
        "vision_fast_model": settings.openai_vision_fast_model,
        "embed_model": settings.openai_embed_model,
        "collection": settings.qdrant_collection,
    }


@app.get("/readme", include_in_schema=False)
def serve_readme() -> Response:
    """Renders README.md as a dark-themed HTML page (opened by the UI README button)."""
    try:
        import markdown as _md
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="markdown package not installed - run: pip install markdown==3.7",
        )

    readme_path = Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        raise HTTPException(status_code=404, detail="README.md not found")

    body = _md.markdown(
        readme_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KnowledgeVault - README</title>
<style>
  body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       max-width:900px;margin:40px auto;padding:0 20px;line-height:1.7;}}
  h1,h2,h3{{color:#e6edf3;border-bottom:1px solid #30363d;padding-bottom:8px;margin:24px 0 12px;}}
  code,pre{{background:#161b22;border:1px solid #30363d;border-radius:6px;
            font-family:'Fira Code','Cascadia Code',Consolas,monospace;font-size:13px;}}
  pre{{padding:16px;overflow-x:auto;}} code{{padding:2px 6px;}}
  a{{color:#58a6ff;}} table{{border-collapse:collapse;width:100%;margin:12px 0;}}
  th,td{{border:1px solid #30363d;padding:8px 12px;text-align:left;}}
  th{{background:#161b22;color:#79c0ff;}} tr:nth-child(even){{background:#161b22;}}
  blockquote{{border-left:3px solid #30363d;margin:0;padding:0 16px;color:#7d8590;}}
</style>
</head>
<body>{body}</body>
</html>"""
    return HTMLResponse(content=page)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    """
    Upload a PDF to the data directory (data/sample/).
    The file is saved on disk and immediately available for ingestion.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only .pdf files are accepted.")
    dest = _DATA_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)
    logger.info("Uploaded %s (%d bytes) → %s", file.filename, len(content), dest)
    return {
        "filename": file.filename,
        "path": str(dest),
        "size_bytes": len(content),
    }


@app.get("/pdfs")
def list_pdfs() -> dict:
    """List all PDF files currently in the data directory."""
    pdf_files = sorted(_DATA_DIR.glob("*.pdf"))
    return {
        "directory": str(_DATA_DIR),
        "count": len(pdf_files),
        "pdfs": [
            {"filename": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
            for p in pdf_files
        ],
    }


@app.post("/ingest/all")
def ingest_all(reset: bool = False) -> dict:
    """
    Ingest every PDF in the data directory.

    reset=true drops and recreates the Qdrant collection before the first PDF.
    Subsequent PDFs in the same call append to the collection.
    """
    try:
        from app.ingest import run_ingest
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"ingest module not available: {e}")

    pdf_files = sorted(_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        return {
            "ok": True,
            "message": f"No PDFs found in {_DATA_DIR}.",
            "summaries": [],
            "errors": [],
        }

    summaries, errors = [], []
    for i, pdf_path in enumerate(pdf_files):
        do_reset = reset and i == 0  # reset collection only before first PDF
        try:
            summary = run_ingest(str(pdf_path), reset=do_reset)
            summaries.append(summary)
            logger.info("ingest_all: finished %s", pdf_path.name)
        except Exception as e:  # noqa: BLE001
            logger.exception("ingest_all: failed %s", pdf_path.name)
            errors.append({"pdf": pdf_path.name, "error": str(e)})

    return {"ok": True, "summaries": summaries, "errors": errors}


@app.get("/collection")
def collection_info() -> dict:
    """Return Qdrant collection stats: name and current point count."""
    try:
        from app.store import get_client
        client = get_client()
        info = client.get_collection(collection_name=settings.qdrant_collection)
        return {
            "collection": settings.qdrant_collection,
            "points": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "collection": settings.qdrant_collection,
            "points": None,
            "status": "unavailable",
            "error": str(e),
        }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """
    Hybrid retrieval: dense text + sparse BM25, fused via reciprocal rank fusion.
    Option 1: text-only - figure chunks are retrieved via their description text.

    The actual retriever logic lives in app/retriever.py.
    """
    try:
        from app.retriever import retrieve as do_retrieve
    except ImportError as e:
        logger.error("retriever import failed: %s", e)
        raise HTTPException(status_code=503, detail="retriever module not available")

    try:
        results: list[RetrievedChunk] = do_retrieve(
            query=req.query,
            k=req.k,
            document_id=req.document_id,
            widen=req.widen,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("retrieve failed")
        raise HTTPException(status_code=502, detail=str(e))

    return RetrieveResponse(query=req.query, chunks=results, fusion="rrf")


@app.post("/ingest")
def ingest(pdf_path: str) -> dict:
    """
    Ingest a single PDF end-to-end: parse → describe figures → chunk → embed → upsert.

    Heavy operation - prefer the CLI for large PDFs (avoids HTTP timeout):
        python -m app.ingest --pdf data/sample/attention-is-all-you-need.pdf
    """
    try:
        from app.ingest import run_ingest
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"ingest module not available: {e}")

    try:
        summary = run_ingest(pdf_path)
    except Exception as e:  # noqa: BLE001
        logger.exception("ingest failed")
        raise HTTPException(status_code=502, detail=str(e))

    return {"ok": True, "summary": summary}


@app.get("/stats")
def stats() -> dict:
    """
    Per-document inventory of what the index holds: chunk counts by type.

    This is how you find out whether a PDF's figures were seen at all. The
    parser only extracts figures that are embedded as raster images; a figure
    drawn as vector graphics inside the PDF never becomes a figure block, so
    a paper can show figures_described=0 and still have a dozen figures on
    the page. Look here before you trust a figure query.
    """
    try:
        from app.store import get_client
        client = get_client()
        points, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=10_000,
            with_payload=["document_id", "chunk_type"],
            with_vectors=False,
        )
    except Exception as e:  # noqa: BLE001
        return {"collection": settings.qdrant_collection, "documents": {}, "error": str(e)}

    docs: dict[str, dict[str, int]] = {}
    for pt in points:
        pl = pt.payload or {}
        d = docs.setdefault(pl.get("document_id", "?"), {"prose": 0, "figure-description": 0, "table-row": 0, "total": 0})
        d[pl.get("chunk_type", "prose")] = d.get(pl.get("chunk_type", "prose"), 0) + 1
        d["total"] += 1
    return {"collection": settings.qdrant_collection, "points": len(points), "documents": docs}
