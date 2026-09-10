"""FastAPI routes for OpsAssist.

  GET  /            -> serve browser UI (index.html)
  GET  /health      -> liveness probe with model name
  GET  /readme      -> render README.md as dark-themed HTML
  POST /agent/run   -> run the agent loop for one user request
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from app.agent import run_agent, new_task_id
from app.config import get_settings
from app.schemas import AgentResponse, AgentRunRequest
from app.state import ShortTermState, WorkflowState, PersistentState, Session

# get_settings() is only called inside request handlers, never at import time,
# so test collection never needs OPENAI_API_KEY. The log level is hardcoded here.
logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("opsassist")

app = FastAPI(title="OpsAssist", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {"status": "ok", "model": s.openai_model}


@app.get("/", include_in_schema=False)
def serve_ui():
    idx = Path(__file__).parent.parent / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(idx, media_type="text/html")


@app.get("/readme", include_in_schema=False)
def serve_readme() -> Response:
    try:
        import markdown as _md
    except ImportError:
        raise HTTPException(status_code=503, detail="pip install markdown==3.7")
    readme = Path(__file__).parent.parent / "README.md"
    if not readme.exists():
        raise HTTPException(status_code=404, detail="README.md not found")
    body = _md.markdown(
        readme.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    page = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>OpsAssist - README</title>"
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


@app.post("/agent/run", response_model=AgentResponse)
def agent_run(req: AgentRunRequest) -> AgentResponse:
    """Run OpsAssist for one user request.

    A known task_id with a saved checkpoint resumes the prior run; a
    missing/unknown task_id starts fresh.
    """
    task_id = req.task_id or new_task_id()

    short_term = ShortTermState()
    workflow = WorkflowState(task_id=task_id)
    persistent = PersistentState()

    # Resume path (crossing 1: short-term reads workflow, at request entry).
    # The checkpoint is a recovery hint, so resume restores the sub-step and
    # the token count, and seeds the conversation with the hint's digest.
    if req.task_id:
        hint = workflow.read_checkpoint()
        if hint:
            short_term.iter_count = int(hint.get("iter", 0))
            short_term.tokens_used = int(hint.get("tokens_used", 0))
            if get_settings().rehydrate_full_transcript:
                # Optional byte-exact replay: rebuild the message list from
                # the persistent audit log, exactly as V3 says you can.
                short_term.messages = persistent.load_transcript(
                    req.user_id, task_id)
            else:
                # Default: resume from the hint alone - one synthetic context
                # message, and the loop carries on.
                short_term.messages = [{
                    "role": "assistant",
                    "content": (f"[resumed at sub-step {hint.get('iter', 0)}] "
                                f"{hint.get('summary', '')}"),
                }]
            logger.info(
                "Resuming task %s from checkpoint (iter=%s, tokens_used=%s, "
                "last_tool_call_id=%s)",
                task_id, short_term.iter_count, short_term.tokens_used,
                hint.get("last_tool_call_id"),
            )

    session = Session(
        task_id=task_id,
        user_id=req.user_id,
        short_term=short_term,
        workflow=workflow,
        persistent=persistent,
    )

    response = run_agent(session, req.user_input)

    if response.stop_reason == "end_turn":
        workflow.clear()

    return response
