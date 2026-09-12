"""FastAPI routes for CohortMCP.

  GET  /           -> serve browser UI (index.html)
  GET  /health     -> liveness probe with model + mcp transport
  GET  /readme     -> render README.md as dark-themed HTML
  GET  /demo/tools -> tools/list, over HTTP, for the browser UI + notebook
  POST /demo/call  -> tools/call, over HTTP
  POST /eval/pick  -> the pinned model picks a tool for a request (function calling)
  GET  /eval/run   -> run the golden set, report tool-pick accuracy

The MCP server a host (Claude Desktop, the MCP Inspector) actually talks to
is `python -m app.mcp_server` (stdio by default, Streamable HTTP if
MCP_SERVER_TRANSPORT=http). This app calls straight into `app.mcp_server`'s
real FastMCP registration for `/demo/tools` and `/demo/call` - it is the
browser teaching/demo harness and eval runner sitting on top of the same
tool surface, not a second MCP server.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from app.config import get_settings
from app.eval import run_eval
from app.llm import pick_tool
from app.mcp_server import execute_tool, list_tools
from app.schemas import (
    EvalSummary,
    ToolCallRequest,
    ToolCallResponse,
    ToolInfo,
    ToolPickRequest,
    ToolPickResponse,
)

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("cohortmcp")

app = FastAPI(title="CohortMCP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {
        "status": "ok",
        "model": s.openai_model,
        "tool_description_quality": s.tool_description_quality,
        "mcp_transport": s.mcp_server_transport,
    }


@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    idx = Path(__file__).parent.parent / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(idx, media_type="text/html")


@app.get("/readme", include_in_schema=False)
def serve_readme() -> Response:
    import markdown as _md

    readme = Path(__file__).parent.parent / "README.md"
    if not readme.exists():
        raise HTTPException(status_code=404, detail="README.md not found")
    body = _md.markdown(
        readme.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    page = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>CohortMCP - README</title>"
        "<style>body{background:#0d1117;color:#e6edf3;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "max-width:900px;margin:40px auto;padding:0 24px;line-height:1.7}"
        "h1,h2,h3{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:6px}"
        "code{background:#21262d;padding:2px 6px;border-radius:4px}"
        "pre{background:#161b22;padding:16px;border-radius:8px;overflow-x:auto}"
        "pre code{background:none;padding:0}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #30363d;padding:8px 12px;text-align:left}"
        "th{background:#161b22;color:#58a6ff}"
        "a{color:#58a6ff}</style></head><body>" + body + "</body></html>"
    )
    return Response(content=page, media_type="text/html; charset=utf-8")


@app.get("/demo/tools", response_model=list[ToolInfo])
async def demo_tools() -> list[ToolInfo]:
    """tools/list, over HTTP - what an Inspector-style client sees."""
    return await list_tools()


@app.post("/demo/call", response_model=ToolCallResponse)
async def demo_call(body: ToolCallRequest) -> ToolCallResponse:
    """tools/call, over HTTP. A tool failure is a body, never a status code."""
    return await execute_tool(body.name, body.arguments)


@app.post("/eval/pick", response_model=ToolPickResponse)
async def eval_pick(body: ToolPickRequest) -> ToolPickResponse:
    """Ask the pinned model which tool it would call for this request."""
    try:
        return await pick_tool(body.request)
    except Exception as exc:  # noqa: BLE001 - upstream provider failure, not a bug here
        logger.error("eval_pick.upstream_failed error=%s", exc)
        # The real upstream message travels in the 502 body - the browser error box
        # shows it verbatim, so a bad key reads as a bad key, not as "500".
        raise HTTPException(status_code=502, detail=f"model call failed: {exc}")


@app.get("/eval/run", response_model=EvalSummary)
async def eval_run() -> EvalSummary:
    """Run the golden tool-pick set and report accuracy for the current
    TOOL_DESCRIPTION_QUALITY setting."""
    try:
        return await run_eval()
    except Exception as exc:  # noqa: BLE001 - upstream provider failure, not a bug here
        logger.error("eval_run.upstream_failed error=%s", exc)
        raise HTTPException(status_code=502, detail=f"model call failed: {exc}")
