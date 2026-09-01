"""TicketStream FastAPI surface - POST /intake-stream with SSE."""
import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, Response

from app.config import get_settings
from app.schemas import IntakeRequest, TicketSchema
from app.llm import validate_and_correct
from app.retry import with_tool_retry
from app.tools import route_to_team, team_for_intent

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("ticketstream")

# Map provider ID → pinned model string
_PROVIDER_MODEL: dict[str, str] = {
    "openai": settings.model_name,
    "nano":   settings.fast_model_name,
}

app = FastAPI(
    title="TicketStream",
    version="0.1.0",
    description="Week 3 - validate-first, stream-second SSE intake pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def serve_ui():
    ui = Path(__file__).parent.parent / "index.html"
    if not ui.exists():
        raise HTTPException(404, detail="index.html not found")
    return FileResponse(ui)


def _sse(event: str, data: dict | str) -> str:
    """Format a single SSE frame."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


@with_tool_retry
def _route(team: str, priority: str):
    """Wrap route_to_team in tool-level retry policy."""
    return route_to_team(team, priority)


async def intake_generator(req: IntakeRequest) -> AsyncIterator[str]:
    """Validate first, stream second.

    Frames emitted (in order):
      event: intent        - validated IntentType dict
      event: priority      - {value: str}
      event: customer_id   - {value: int|null}
      event: routed        - RoutingResult dict
      event: done          - {ok: true, ...telemetry}
    On failure:
      event: validation_failed - {error: str, correction_count: int}
      event: done              - {ok: false}
    """
    model_name = _PROVIDER_MODEL[req.provider]

    # 1. Validate the ticket (blocks until the full structured object is ready)
    ticket, telemetry = await validate_and_correct(req.message, model_name=model_name)
    if ticket is None:
        yield _sse("validation_failed", {
            "error": "Model did not produce a structured ticket.",
            "correction_count": telemetry["correction_count"],
        })
        yield _sse("done", {"ok": False})
        return

    # 2. Stream the validated fields one-by-one
    yield _sse("intent", ticket.intent.model_dump())
    await asyncio.sleep(0)  # let the event loop flush

    yield _sse("priority", {"value": ticket.priority})
    await asyncio.sleep(0)

    yield _sse("customer_id", {"value": ticket.customer_id})
    await asyncio.sleep(0)

    # 3. Side-effect: route with tenacity-backed retries
    try:
        team   = team_for_intent(ticket)
        result = _route(team=team, priority=ticket.priority)
        yield _sse("routed", result.model_dump())
    except Exception as e:
        logger.error("route_to_team failed after retries",
                     extra={"event": "tool_impl_transient_failed"})
        yield _sse("validation_failed", {
            "error": f"Routing failed: {e}",
            "stage": "post_validate",
        })

    # 4. Terminal frame - include model name in telemetry
    yield _sse("done", {"ok": True, "model": model_name, **telemetry})


@app.post("/intake-stream")
async def intake_stream(req: IntakeRequest):
    return StreamingResponse(
        intake_generator(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "model": s.model_name,
        "models": {
            "openai": s.model_name,
            "nano":   s.fast_model_name,
        },
    }


@app.get("/readme", include_in_schema=False)
def serve_readme():
    import markdown as _md
    readme = Path(__file__).parent.parent / "README.md"
    if not readme.exists():
        raise HTTPException(404, detail="README.md not found")
    body = _md.markdown(
        readme.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    page = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="UTF-8"><title>TicketStream - README</title>'
        "<style>"
        "body{background:#0d1117;color:#e6edf3;font-family:-apple-system,sans-serif;"
        "max-width:860px;margin:40px auto;padding:0 20px;line-height:1.7}"
        "h1,h2,h3{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:6px}"
        "code{background:#21262d;border-radius:4px;padding:2px 6px;font-family:monospace}"
        "pre{background:#161b22;border:1px solid #30363d;border-radius:8px;"
        "padding:16px;overflow-x:auto}"
        "a{color:#58a6ff}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #30363d;padding:8px 12px}"
        "th{background:#161b22}"
        "blockquote{border-left:3px solid #30363d;margin:0;padding-left:16px;color:#7d8590}"
        "</style></head>"
        f"<body>{body}</body></html>"
    )
    return Response(content=page, media_type="text/html; charset=utf-8")
