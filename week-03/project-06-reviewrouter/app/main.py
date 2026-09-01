"""The service.

Contract, fixed. The tests check it and so does the grader.

    POST /intake   -> SSE stream
    GET  /health   -> works with no API key set
    GET  /readme   -> your README as text

Four events, in this order, every successful request:

    event: intent      data: {"type": ..., "order_id": ...}
    event: priority    data: {"priority": ...}
    event: routed      data: {"team": ...}          <- fires even when null
    event: done        data: {"corrections": n, "attempts": n}

One more, for the unhappy path:

    event: validation_failed   data: {"error": "..."}

When validation_failed fires, `routed` must NOT. Never route on an object
that did not pass the gate.
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from app.config import get_settings, has_key

app = FastAPI(title="ReviewRouter")

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Without this, a proxy buffers the whole response and your "streaming"
    # service delivers everything at once. This is the single most common
    # reason a correct implementation looks broken.
    "X-Accel-Buffering": "no",
}


class IntakeRequest(BaseModel):
    review: str


def sse(event: str, data: dict) -> str:
    """Format one server-sent event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/health")
def health() -> dict:
    """Must answer without calling a model, and with no key set."""
    s = get_settings()
    return {"status": "ok", "model": s.model, "key_configured": has_key()}


@app.get("/readme", response_class=PlainTextResponse)
def readme() -> str:
    p = Path(__file__).resolve().parent.parent / "README.md"
    return p.read_text(encoding="utf-8") if p.exists() else "README.md not found"


@app.post("/intake")
async def intake(body: IntakeRequest) -> StreamingResponse:
    """TODO: the whole build lives here.

    Sketch, in order:

        messages, attempts = await resolve_lookup(body.review)
        async for field, value in stream_ticket(messages):
            ...accumulate, and yield `intent` once type and order_id exist,
               then `priority` when it lands...
        ticket, corrections = await validate_with_correction(raw, messages)
        team = route_to_team(ticket.type, ticket.priority)
        yield sse("routed", {"team": team})
        yield sse("done", {"corrections": corrections, "attempts": attempts})

    Wrap the gate so SchemaCorrectionExhausted emits `validation_failed`
    instead, and ToolLoopExceeded returns a distinct status code with its own
    log tag.
    """
    async def stream():
        raise NotImplementedError("TODO: build the intake stream")
        yield  # pragma: no cover

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers=SSE_HEADERS)
