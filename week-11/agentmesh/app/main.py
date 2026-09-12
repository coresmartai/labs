"""AgentMesh - an A2A v1.0 service wrapped around an in-process triage agent.

A2A protocol surface (five routes):
  GET  /.well-known/agent-card.json   - discovery, canonical path (unauthed)
  GET  /.well-known/agent.json        - legacy discovery alias (pre-v1.0 clients)
  POST /tasks                         - submit (Pydantic + JWT)
  GET  /tasks/{task_id}/stream        - SSE, with Last-Event-ID replay
  POST /tasks/{task_id}/input         - human-in-the-loop reply

Course-wide UI contract (three routes):
  GET  /                              - serves index.html
  GET  /health                        - liveness + the pinned models
  GET  /readme                        - renders README.md as a dark HTML page

Every response carries `A2A-Version`. It is emitted, never negotiated - a client
that sends a different version header is not rejected, because this build has
exactly one version to speak.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from app.cohort import load_cohort
from app.config import Settings, get_settings
from app.identity import agent_name, student_identity
from app.jwt_verify import require_scope, verify_bearer
from app.llm import get_llm
from app.logging_setup import setup_logging
from app.mcp_server import mcp as cohort_mcp
from app.schemas import (
    INTERRUPTED_STATES,
    TERMINAL_STATES,
    AgentCapabilities,
    AgentCard,
    AgentEndpoints,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    TaskAck,
    TaskStatus,
    TaskFailedPayload,
    TaskInputReply,
    TaskState,
    TaskSubmit,
    TriageInput,
    WhoAmIOutput,
)
from app.task_store import TaskStore, get_store
from app.triage_core import RejectedByCaller, run_triage

_settings = get_settings()
_log_path = setup_logging(_settings)   # console + a rotating file, all modules auto-persist
logger = logging.getLogger("agentmesh")
if _log_path:
    logger.info("logging.file path=%s level=%s", _log_path, _settings.log_level)

# ---- MCP co-hosting: the same FastMCP instance, on this app's port ----
#
# One tunnel has to expose both protocols, so the MCP Streamable HTTP surface is
# mounted here rather than run as a second process on a second port. Two ordering
# rules, both of which fail loudly if broken:
#   1. streamable_http_app() must be called before mcp.session_manager exists -
#      the manager is built lazily by that call.
#   2. The mounted app collapses its own route to "/", because it carries an
#      internal /mcp path that would otherwise land the endpoint on /mcp/mcp.
_mcp_asgi = None
if _settings.cohost_mcp:
    cohort_mcp.settings.streamable_http_path = "/"
    _mcp_asgi = cohort_mcp.streamable_http_app()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Run the MCP session manager for the lifetime of the app. Without this the
    mount raises "Task group is not initialized" on its very first request."""
    if _mcp_asgi is None:
        yield
        return
    async with cohort_mcp.session_manager.run():
        logger.info("mcp.cohosted path=%s", _settings.mcp_mount_path)
        yield


app = FastAPI(title="AgentMesh", version="0.1.0", lifespan=_lifespan)

# Open CORS for local dev - the UI calls the same origin in production but a separate
# local file:// or different port is common while iterating. `expose_headers` is what
# lets a browser MCP client read `mcp-session-id` off the initialize response: the
# Streamable-HTTP handshake hands the session id back in that header, and without it
# named-origin JS (a classmate's board hitting YOUR /mcp/) can complete the handshake
# but never see the id it needs for every subsequent call.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["mcp-session-id"],
)


@app.middleware("http")
async def a2a_version_header(request: Request, call_next):
    """Announce the protocol version on every response - including errors. No
    negotiation: we do not read the client's version header and we never 4xx on it."""
    response = await call_next(request)
    response.headers["A2A-Version"] = get_settings().a2a_protocol_version
    return response


@app.middleware("http")
async def guard_mcp_mount(request: Request, call_next):
    """Bearer-gate the co-hosted MCP surface.

    `app.mount()` bypasses FastAPI's dependency system entirely, so the
    `Depends(require_scope(...))` that protects every A2A route does NOT protect
    the mounted MCP app. Left alone, co-hosting would publish an unauthenticated
    tool surface on the same public tunnel as the gated one. This middleware runs
    the identical `verify_bearer` the A2A routes use, on the mount path only.

    Errors are returned, not raised: middleware sits outside FastAPI's exception
    handlers, so an HTTPException raised here would surface as a bare 500.
    """
    s = get_settings()
    # OPTIONS is a CORS preflight: the browser sends it with no Authorization header
    # by design, to ASK whether the real (authenticated) request is allowed. 401-ing
    # it means a cross-origin browser can never even attempt the MCP call - which is
    # exactly what a classmate's board hitting your /mcp/ is. Let the preflight through
    # to the CORS middleware; the actual POST that follows is still gated below.
    guarded = (
        s.cohost_mcp
        and request.method != "OPTIONS"
        and request.url.path.rstrip("/").startswith(s.mcp_mount_path.rstrip("/"))
    )
    if guarded:
        header = request.headers.get("authorization") or ""
        if not header.lower().startswith("bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": {"reason": "missing_bearer",
                                    "message": "Authorization: Bearer <token> required for the MCP surface"}},
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            await verify_bearer(header.split(" ", 1)[1], s, s.mcp_required_scope)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


def _caller_origin(request: Request) -> str:
    """Where the request actually came from. Behind a cloudflared tunnel the socket peer
    is Cloudflare, not the caller - the real client IP arrives in these headers, so a
    classmate reaching your agent over the tunnel shows up as their address `via tunnel`
    rather than as 127.0.0.1."""
    cf = request.headers.get("cf-connecting-ip")
    xff = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if cf or xff:
        return f"{cf or xff} via tunnel"
    return request.client.host if request.client else "-"


def _caller_identity(request: Request) -> str | None:
    """Who is calling, from the bearer - for the log only, so it is decoded WITHOUT
    verifying (the route's dependency does the real verification). Returns the caller's
    email/sub from a JWT, 'dev-token' for the fixed dev bearer, or None when unauthenticated."""
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        return None
    token = header.split(" ", 1)[1]
    parts = token.split(".")
    if len(parts) != 3:
        return "dev-token" if token else None
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims.get("email") or claims.get("sub") or claims.get("user_id")
    except Exception:  # noqa: BLE001 - a malformed token is just an unknown caller in the log
        return None


@app.middleware("http")
async def access_log(request: Request, call_next):
    """One log line per request - method, path, WHO reached out (IP via tunnel + the
    caller's identity from their token), the status, and how long it took. Auto-persisted
    to logs/agentmesh.log, so 'who called my agent, and when?' is answerable after the fact.
    Defined last so it wraps the others and sees the final status, including a 401 from the
    MCP guard above."""
    t0 = time.perf_counter()
    response = await call_next(request)
    dt_ms = (time.perf_counter() - t0) * 1000
    caller = _caller_identity(request)
    logger.info(
        "http %s %s <- %s%s -> %d %.0fms",
        request.method, request.url.path, _caller_origin(request),
        f" [{caller}]" if caller else "", response.status_code, dt_ms,
    )
    return response


# Mounted after the middleware so both guards wrap it. Only when co-hosting is on -
# otherwise the MCP surface stays a separate `python -m app.mcp_server` process.
if _mcp_asgi is not None:
    app.mount(_settings.mcp_mount_path, _mcp_asgi)


# ---- the Agent Card ----

_TRIAGE_SKILL = AgentSkill(
    id="triage_incident",
    name="Triage incident",
    description=(
        "Triage one production incident. Use when an operator describes a production "
        "issue. Returns a category (knowledge / action / escalate), a summary, and "
        "either runbook citations or an approved remediation. Mutating actions pause "
        "at a human approval gate before anything runs."
    ),
    tags=["incident", "reliability", "sre"],
    examples=[
        "payments-api is timing out after the deploy",
        "Where is the runbook for restarting payments-api?",
        "Please restart payments-api now",
    ],
    inputModes=["application/json", "text/plain"],
    outputModes=["application/json"],
)


_WHOAMI_SKILL = AgentSkill(
    id="whoami",
    name="Identify this agent",
    description=(
        "Return the identity of the student who runs this agent - student id, name, "
        "specialty, base URL and the protocols it speaks. Takes no input. Use it to "
        "confirm whose agent you have reached, and to prove an authenticated task "
        "round trip works before submitting real work."
    ),
    tags=["identity", "cohort", "discovery"],
    examples=["who are you?", "which student runs this agent?"],
    inputModes=["application/json"],
    outputModes=["application/json"],
)


def _build_card(settings: Settings) -> AgentCard:
    me = student_identity(settings)
    return AgentCard(
        name=agent_name(settings),
        version="1.0.0",
        protocolVersion=settings.a2a_protocol_version,
        description=(
            f"AgentMesh instance operated by {settings.student_name} ({settings.student_id}), "
            f"specialty {settings.student_specialty}. Triages incoming engineering incidents: "
            "classifies into knowledge, action, or escalate; retrieves runbooks for knowledge "
            "requests; proposes remediation actions behind a human approval gate; surfaces "
            "escalations directly."
        ),
        url=settings.agentmesh_base_url,
        # Who OPERATES this agent. For the cohort that is the student - which is
        # how a classmate knows whose service they are about to invoke.
        provider=AgentProvider(organization=settings.student_name, url=settings.agentmesh_base_url),
        # skills[] is what the agent DOES. capabilities is protocol flags only -
        # streaming, push notifications, extended card. Nothing else belongs in it.
        skills=[_TRIAGE_SKILL, _WHOAMI_SKILL],
        capabilities=AgentCapabilities(streaming=True, pushNotifications=False, extendedAgentCard=False),
        # The v1.0 reachability field. ORDERED: entry zero is preferred. One entry
        # here because this build speaks one binding; a service fronted by both a
        # JSON-RPC and a gRPC endpoint would list both and let the caller choose.
        supportedInterfaces=[
            AgentInterface(
                url=settings.agentmesh_base_url,
                protocolBinding="HTTP+JSON",
                protocolVersion=settings.a2a_protocol_version,
            )
        ],
        endpoints=AgentEndpoints(base=settings.agentmesh_base_url, mcp=me["endpoints"]["mcp"]),
        signatures=[],  # declared, unpopulated: this build does not sign its card
    )


@app.get("/.well-known/agent-card.json", response_model=AgentCard)
def agent_card(settings: Settings = Depends(get_settings)) -> AgentCard:
    """Discovery at the A2A v1.0 canonical well-known path. No auth - anyone can read."""
    return _build_card(settings)


@app.get("/.well-known/agent.json", response_model=AgentCard, include_in_schema=False)
def agent_card_legacy(settings: Settings = Depends(get_settings)) -> AgentCard:
    """Legacy pre-v1.0 discovery path - the same body, kept as an alias."""
    return _build_card(settings)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Liveness probe. The models block is built through the LLM seam's `model_for()`,
    so the chip on screen and the pin the code would dial cannot drift apart."""
    llm = get_llm()
    return {
        "status": "ok",
        "model": llm.model_for("knowledge"),  # primary model shown in the UI health chip
        "models": {
            "triage": llm.model_for("triage"),
            "knowledge": llm.model_for("knowledge"),
        },
        "auth_mode": settings.auth_mode,
        "a2a_version": settings.a2a_protocol_version,
    }


@app.get("/cohort")
def cohort_info() -> dict[str, Any]:
    """What the roster sweep needs to bootstrap, resolved from cohort.json + COHORT_MODE.

    Unauthenticated by design: it only exposes where classmates are, not any secret.
    In online mode `cohort_index_url` is the class's public roster (a world-readable
    Firebase URL); in solo mode `peers` are the local dummy copies. The browser board
    reads this, then fetches the index / probes each agent directly - same data the
    `cohort_roster.py` CLI works from. Returns `null`/`[]` rather than raising when a
    field is unset, so a half-configured online cohort still renders.
    """
    try:
        c = load_cohort()
    except (FileNotFoundError, ValueError) as exc:
        return {"mode": None, "cohort_index_url": None, "peers": [], "my_base_url": None,
                "error": str(exc)}
    return {
        "mode": c.mode,
        "cohort_index_url": c.cohort_index_url,
        "my_base_url": c.my_base_url,
        "peers": [
            {"student_id": p.student_id, "student_name": p.student_name, "base_url": p.base_url}
            for p in c.peers
        ],
    }


@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    idx = Path(__file__).parent.parent / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(idx, media_type="text/html")


@app.get("/readme", include_in_schema=False)
def serve_readme() -> Response:
    """Render README.md as a dark-themed HTML page."""
    import markdown as _md  # lazy import - only used in browser flows

    readme_path = Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        raise HTTPException(status_code=404, detail="README.md not found")

    body = _md.markdown(
        readme_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{background:#0d1117;color:#e6edf3;font-family:sans-serif;"
        "max-width:900px;margin:40px auto;padding:0 20px;line-height:1.6}"
        "a{color:#58a6ff}code{background:#161b22;padding:2px 6px;border-radius:4px}"
        "pre{background:#161b22;padding:16px;border-radius:8px;overflow-x:auto}"
        "table{border-collapse:collapse}td,th{border:1px solid #30363d;padding:8px 12px}"
        "h1,h2,h3{border-bottom:1px solid #30363d;padding-bottom:6px}"
        "</style></head>"
        f"<body>{body}</body></html>"
    )
    return Response(content=html, media_type="text/html; charset=utf-8")


# ---- A2A task surface ----


@app.post("/tasks", response_model=TaskAck, status_code=status.HTTP_202_ACCEPTED)
async def submit_task(
    body: TaskSubmit,
    settings: Settings = Depends(get_settings),
    claims: dict[str, Any] = Depends(require_scope("triage:invoke")),
) -> TaskAck:
    """Validate, gate on the JWT, create the task, fire the background coroutine, ack."""
    if body.skill not in ("triage_incident", "whoami"):
        raise HTTPException(status_code=400, detail={"reason": "unknown_skill", "skill": body.skill})

    store = get_store()

    # `whoami` takes no input, so it skips per-skill validation entirely and
    # completes in one hop. It exists so a classmate can prove the whole
    # authenticated task path - submit, stream, terminal state - works, before
    # trusting it with a real incident.
    if body.skill == "whoami":
        task_id = store.create()
        await store.update_state(
            task_id,
            TaskState.TASK_STATE_SUBMITTED,
            {"skill": "whoami", "caller": claims.get("sub")},
        )
        asyncio.create_task(_run_whoami(task_id, store, claims.get("sub")))
        return TaskAck(
            task_id=task_id,
            state=TaskState.TASK_STATE_SUBMITTED,
            stream_url=f"{settings.agentmesh_base_url}/tasks/{task_id}/stream",
        )

    # Per-skill schema validation. 422 with field-level detail on failure - the same
    # shape FastAPI produces for the outer body, because this validation happens one
    # level down, inside the polymorphic `input` field.
    try:
        triage_in = TriageInput.model_validate(body.input)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    task_id = store.create()
    await store.update_state(
        task_id,
        TaskState.TASK_STATE_SUBMITTED,
        {"skill": "triage_incident", "caller": claims.get("sub")},
    )

    asyncio.create_task(_run_task(task_id, triage_in, store))

    return TaskAck(
        task_id=task_id,
        state=TaskState.TASK_STATE_SUBMITTED,
        stream_url=f"{settings.agentmesh_base_url}/tasks/{task_id}/stream",
    )


@app.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task(
    task_id: str,
    _claims: dict[str, Any] = Depends(require_scope("triage:invoke")),
) -> TaskStatus:
    """A2A's `tasks/get`: where is this task right now?

    Every streaming protocol needs this. A stream is a live optimisation; it can be
    dropped, buffered by an intermediary, or never opened at all. Without a snapshot
    endpoint a caller who misses the stream has no way to learn the outcome of work
    that already happened - so this is what makes the stream *optional* rather than
    load-bearing.
    """
    store = get_store()
    event = store.latest(task_id)
    if event is None or store.get_state(task_id) is None:
        raise HTTPException(status_code=404, detail={"reason": "unknown_task"})
    return TaskStatus(
        task_id=task_id,
        state=event.state,
        timestamp=event.timestamp,
        payload=event.payload,
        last_event_id=event.event_id,
        final=event.state in TERMINAL_STATES,
        interrupted=event.state in INTERRUPTED_STATES,
    )


@app.get("/tasks/{task_id}/stream")
async def stream_task(
    task_id: str,
    request: Request,
    last_event_id_header: int = Header(default=0, alias="Last-Event-ID"),
    last_event_id: int = Query(default=0),
    _claims: dict[str, Any] = Depends(require_scope("triage:invoke")),
) -> EventSourceResponse:
    """SSE stream. Honours the `Last-Event-ID` header and - because EventSource cannot
    set headers on a reconnect it did not initiate - a `last_event_id` query param."""
    store = get_store()
    if store.get_state(task_id) is None:
        raise HTTPException(status_code=404, detail={"reason": "unknown_task"})

    cursor = max(last_event_id_header, last_event_id)

    async def event_gen():
        async for event in store.stream(task_id, from_event_id=cursor):
            if await request.is_disconnected():
                break
            # A2A marks the last event of an interaction `final` and closes the stream.
            # An INTERRUPTED state ends the interaction just as a TERMINAL one does:
            # the task is not finished, but the next move is the caller's, so there is
            # nothing more to push. The caller replies, then resubscribes with
            # Last-Event-ID to pick the stream back up exactly where it stopped.
            #
            # This also happens to be the only shape that survives a buffering proxy.
            # A CDN or tunnel releases a response body when the response ENDS; hold one
            # stream open across the gate and the caller never sees INPUT_REQUIRED, so
            # it never replies, so the stream never ends - a deadlock that reads as "the
            # network is slow". A finite response per interaction cannot deadlock.
            closing = event.state in TERMINAL_STATES or event.state in INTERRUPTED_STATES
            data = event.model_dump(mode="json")
            if closing:
                data["final"] = True
            yield {
                "id": str(event.event_id),
                "event": event.state.value,      # TASK_STATE_* - the value IS the wire string
                "data": json.dumps(data),
            }
            if closing:
                break

    # X-Accel-Buffering: no - without it nginx (and friends) buffer the stream and
    # the events arrive in one lump at the end, which looks exactly like a hang.
    #
    # There is deliberately NO `Connection: keep-alive` here. It is the obvious thing
    # to write on an SSE response, and it is wrong: `Connection` is a hop-by-hop
    # header, forbidden in HTTP/2 (RFC 9113 §8.2.2). Local uvicorn speaks HTTP/1.1 and
    # ignores it, so it looks harmless - but put an HTTP/2 proxy in front (a
    # cloudflared tunnel, or any CDN) and the response becomes malformed: headers get
    # through, the body never does. The task then runs to completion on this side
    # while the caller waits forever, which reads as "the tunnel is slow" and is not.
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return EventSourceResponse(event_gen(), headers=headers)


@app.post("/tasks/{task_id}/input")
async def submit_input(
    task_id: str,
    reply: TaskInputReply,
    _claims: dict[str, Any] = Depends(require_scope("triage:invoke")),
) -> JSONResponse:
    """Human-in-the-loop reply. 409 unless the task is paused at TASK_STATE_INPUT_REQUIRED."""
    store = get_store()
    state = store.get_state(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail={"reason": "unknown_task"})
    if state is not TaskState.TASK_STATE_INPUT_REQUIRED:
        raise HTTPException(
            status_code=409,
            detail={"reason": "task_not_paused", "current_state": state.value},
        )
    ok = await store.submit_input(task_id, reply.model_dump())
    if not ok:
        raise HTTPException(status_code=409, detail={"reason": "no_pending_input"})
    return JSONResponse({"accepted": True})


# ---- the wrapper layer: agent events -> A2A wire states ----


async def _run_whoami(task_id: str, store: TaskStore, caller: str | None) -> None:
    """The identity skill, driven through the same state machine as a real task.

    No gate, no model, no tools - submitted -> working -> completed in one hop.
    The payload goes through WhoAmIOutput first: an identity answer is a contract
    like any other, and an unvalidated dict is how a contract quietly rots.
    """
    try:
        await store.update_state(
            task_id, TaskState.TASK_STATE_WORKING, {"node": "identity", "message": "resolving"}
        )
        me = student_identity()
        payload = WhoAmIOutput(
            student_id=me["student_id"],
            student_name=me["student_name"],
            specialty=me["specialty"],
            agent_name=me["agent_name"],
            base_url=me["base_url"],
            protocols=me["protocols"],
            caller=caller,
        )
        await store.update_state(task_id, TaskState.TASK_STATE_COMPLETED, payload.model_dump())
    except Exception as e:  # noqa: BLE001
        logger.exception("whoami.failed task_id=%s error=%s", task_id, str(e))
        await store.update_state(
            task_id,
            TaskState.TASK_STATE_FAILED,
            TaskFailedPayload(reason="internal_error", evidence=str(e), retryable=True).model_dump(),
        )


class _InputTimeout(Exception):
    """The approval gate timed out. TASK_STATE_FAILED is already written when this
    is raised; it exists only to unwind the agent without executing anything."""


async def _run_task(task_id: str, triage_in: TriageInput, store: TaskStore) -> None:
    """Drive the triage agent and translate what it does into A2A task states.

    This function is the entire A2A wrapper. The agent below it emits node progress
    and asks for approval; everything protocol-shaped happens here.
    """
    approved_flag = False
    reject_note: str | None = None

    async def on_progress(node: str, payload: dict[str, Any]) -> None:
        await store.update_state(task_id, TaskState.TASK_STATE_WORKING, {"node": node, **payload})

    async def wait_for_approval(proposal: dict[str, Any]) -> bool:
        nonlocal approved_flag, reject_note
        store.begin_input_required(task_id)
        await store.update_state(task_id, TaskState.TASK_STATE_INPUT_REQUIRED, proposal)
        reply = await store.wait_for_input(task_id, timeout=300.0)
        if reply is None:
            # Input timeout. Fail loudly and structurally - without this the task sits
            # forever and the service leaks capacity one inattentive reviewer at a time.
            await store.update_state(
                task_id,
                TaskState.TASK_STATE_FAILED,
                TaskFailedPayload(
                    reason="input_timeout",
                    evidence="no approval reply in 300s",
                    retryable=False,
                ).model_dump(),
            )
            raise _InputTimeout
        approved_flag = bool(reply.get("approved"))
        reject_note = reply.get("note")
        return approved_flag

    try:
        result = await run_triage(
            description=triage_in.description,
            severity=triage_in.severity.value,
            user_id=triage_in.user_id,
            on_progress=on_progress,
            wait_for_approval=wait_for_approval,
        )
    except RejectedByCaller:
        # The gate said no. Terminal, and nothing mutating ever ran.
        await store.update_state(
            task_id,
            TaskState.TASK_STATE_REJECTED,
            {"reason": "rejected_by_caller", "note": reject_note},
        )
        return
    except _InputTimeout:
        # The gate timed out; TASK_STATE_FAILED was written above.
        return
    except Exception as e:  # noqa: BLE001
        logger.exception("task.failed task_id=%s error=%s", task_id, str(e))
        await store.update_state(
            task_id,
            TaskState.TASK_STATE_FAILED,
            TaskFailedPayload(reason="internal_error", evidence=str(e), retryable=True).model_dump(),
        )
        return

    payload: dict[str, Any] = result.model_dump()
    payload["user_id"] = triage_in.user_id
    if result.category == "action":
        payload["approved"] = approved_flag
    await store.update_state(task_id, TaskState.TASK_STATE_COMPLETED, payload)


async def cancel_task(task_id: str, store: TaskStore | None = None) -> None:
    """The internal cancel path. No script needs a public route for it, but the state
    is real and reachable - which is what keeps all four terminal states honest."""
    store = store or get_store()
    await store.update_state(
        task_id, TaskState.TASK_STATE_CANCELED, {"reason": "canceled_by_caller"}
    )
