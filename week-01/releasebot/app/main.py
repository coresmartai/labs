"""
FastAPI entrypoint.

Routes:
  GET  /              - serve the UI (index.html)
  GET  /health        - liveness probe
  GET  /readme        - render README.md in the browser
  POST /summarize     - structured output + one tool round-trip
  POST /summarize-stream - streaming SSE response (no tools)

Run locally:
    uvicorn app.main:app --reload
"""

from __future__ import annotations
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from app.config import get_settings
from app.llm import stream_text, summarize_with_tools
from app.schemas import SummaryRequest


settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("releasebot")

app = FastAPI(
    title="ReleaseBot",
    version="0.1.0",
    description="Week 1 starter - streaming + structured output + one tool call.",
)

# Allow the HTML UI to call the API from any origin (file:// or different port).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root of the project (one level above /app)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    """Serve the browser UI."""
    return FileResponse(_PROJECT_ROOT / "index.html")


@app.get("/readme", include_in_schema=False)
def serve_readme() -> HTMLResponse:
    """Render README.md in the browser."""
    readme = _PROJECT_ROOT / "README.md"
    content = readme.read_text(encoding="utf-8") if readme.exists() else "README not found."
    # Wrap in a page that uses marked.js for rendering
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ReleaseBot - README</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 860px; margin: 40px auto; padding: 0 24px;
           background: #0d1117; color: #e6edf3; line-height: 1.6; }}
    a {{ color: #58a6ff; }} code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; }}
    pre {{ background: #161b22; padding: 16px; border-radius: 8px; overflow-x: auto; }}
    pre code {{ background: none; padding: 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
    th {{ background: #161b22; }}
  </style>
</head>
<body>
  <div id="content"></div>
  <script>
    document.getElementById('content').innerHTML =
      marked.parse({json.dumps(content)});
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.openai_model}


@app.post("/summarize")
def summarize(req: SummaryRequest) -> dict:
    """
    Non-streaming endpoint.
    Returns parsed structured summary + the tool call(s) that were made.
    """
    if not req.recipient:
        raise HTTPException(status_code=422, detail="recipient email is required for /summarize")
    try:
        return summarize_with_tools(req.release_notes, req.recipient)
    except ValueError as e:
        logger.warning("Schema validation failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Model output failed schema: {e}")


@app.post("/summarize-stream")
async def summarize_stream(req: SummaryRequest) -> StreamingResponse:
    """
    Streaming endpoint. Returns Server-Sent Events.
    Each event: data: {"delta": "...partial text..."}\n\n
    Final event: data: [DONE]\n\n
    """

    async def event_generator():
        try:
            async for delta in stream_text(req.release_notes):
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:  # noqa: BLE001
            logger.exception("Stream failed")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
