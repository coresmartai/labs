"""FastAPI surface for TriageFlow.

POST /triage    - start a new incident, returns either a final answer or a
                  pending_approval response if the action gate fires.
POST /approve   - resume a paused thread with an approve/reject decision.

In the live demo we use an in-memory checkpointer. In production swap to
PostgresSaver - see langgraph.checkpoint.postgres.
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAIError
from langgraph.types import Command

from app.authz import (
    assert_is_approver,
    assert_owns_thread,
    caller_id,
    claim_thread,
    forget_user_threads,
)
from app.config import get_settings
from app.graph import build_graph
from app.memory import memory_write, seed_demo_prefs, seed_memory
from app.schemas import (
    ApproveRequest,
    DeletionReceipt,
    PrefRequest,
    TriageRequest,
    TriageResponse,
)
from app.vectorstore import user_scope

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("triageflow")


# ---- app + checkpointed graph singleton ----

_checkpointer = MemorySaver()
_graph: Any = None  # built on startup so settings are loaded once


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    s = get_settings()  # validate env on startup
    _graph = build_graph(checkpointer=_checkpointer)
    # Load the vector-store corpus (external + episodic) and write the demo
    # user's procedural rules. Without the second call, prefs are always empty
    # and compose_system_prompt never injects a rule.
    seed_memory()
    seed_demo_prefs()
    logger.info("triageflow.startup model_versions=triage:%s knowledge:%s action:%s memory_backend=%s",
                s.triage_model, s.knowledge_model, s.action_model, s.memory_backend)
    yield


app = FastAPI(title="TriageFlow", version="0.1.0", lifespan=lifespan)

# Not "*". This server exposes DELETE /user/{uid} and POST /approve, and a
# wildcard origin hands both to any page the user happens to have open. The
# allowed list is a setting because the deployed origin is not the dev one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-User-Id"],
)


@app.exception_handler(OpenAIError)
def provider_error(request, exc: OpenAIError) -> JSONResponse:
    """Tell the caller whose problem it is.

    Without this, a mistyped OPENAI_API_KEY surfaces in a browser as a bare 500
    "Internal Server Error" with the real cause only in the server log, and the
    first thing a learner does is go looking for a bug in the graph. A provider
    failure is a 502: this service is fine, the thing it depends on is not.
    Everything else still raises a 500, because everything else really is a bug
    in here and should stay loud.
    """
    logger.error("triageflow.provider_error %s: %s", type(exc).__name__, exc)
    return JSONResponse(
        status_code=502,
        content={"detail": f"model provider call failed ({type(exc).__name__}). "
                           f"Check OPENAI_API_KEY in .env and that the API is reachable.",
                 "error_type": type(exc).__name__},
    )


def _config(thread_id: str) -> dict[str, Any]:
    # recursion_limit caps SUPER-STEPS per run, not node executions (LangGraph's
    # default is 1000 from v1.0.6, so this must be set explicitly;
    # we set 10 - more than 10 hops through a routing graph means a loop).
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": get_settings().recursion_limit,
    }


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest, caller: str = Depends(caller_id)) -> TriageResponse:
    # You may only run the graph as yourself. Without this, user_id is just a
    # field, and a field the caller sets is not an identity.
    if caller != req.user_id:
        raise HTTPException(status_code=403, detail="user_id must be the calling user")
    thread_id = req.thread_id or f"th_{uuid.uuid4().hex[:12]}"
    # A client-supplied thread id used to be enough to resume ANOTHER user's
    # paused thread and walk their pending approval off the gate. Claiming the
    # thread is what closes that: a second user naming it gets a 403.
    claim_thread(thread_id, req.user_id)
    cfg = _config(thread_id)

    initial: dict[str, Any] = {
        "user_request": req.user_request,
        "user_id": req.user_id,
        "thread_id": thread_id,
    }

    assert _graph is not None
    _graph.invoke(initial, config=cfg)
    state = _graph.get_state(cfg).values

    memory_write(user_scope(req.user_id), "session",
                 {"request": req.user_request, "route": state.get("route")},
                 thread_id=thread_id)

    llm_calls = state.get("llm_calls") or []

    if state.get("pending_approval") and state.get("final_answer") is None:
        return TriageResponse(
            status="pending_approval",
            thread_id=thread_id,
            route=state.get("route"),
            proposed_action=state.get("proposed_action"),
            citations=state.get("citations", []),
            llm_calls=llm_calls,
        )

    if state.get("route") == "escalate_human":
        return TriageResponse(status="escalated", thread_id=thread_id,
                              route=state.get("route"),
                              final_answer=state.get("final_answer"),
                              llm_calls=llm_calls)

    return TriageResponse(
        status="completed",
        thread_id=thread_id,
        route=state.get("route"),
        final_answer=state.get("final_answer"),
        citations=state.get("citations", []),
        llm_calls=llm_calls,
    )


@app.post("/approve", response_model=TriageResponse)
def approve(req: ApproveRequest, caller: str = Depends(caller_id)) -> TriageResponse:
    # Role, not ownership: an approver is deliberately NOT the requester.
    assert_is_approver(caller, req.reviewer_id)
    cfg = _config(req.thread_id)
    assert _graph is not None

    # validate: thread must be paused at the action gate
    snapshot = _graph.get_state(cfg)
    if snapshot is None or "action_execute" not in (snapshot.next or ()):
        raise HTTPException(
            status_code=409,
            detail=f"thread {req.thread_id} is not awaiting approval at action_execute",
        )

    # The decision travels WITH the request, as the interrupt() resume payload.
    # It is not written onto the contract, so no node can read or branch on it.
    _graph.invoke(Command(resume={"approved": req.approved, "reviewer_id": req.reviewer_id}),
                  config=cfg)
    final_state = _graph.get_state(cfg).values

    memory_write(user_scope(final_state.get("user_id", "unknown")), "session",
                 {"approved": req.approved, "reviewer": req.reviewer_id},
                 thread_id=req.thread_id)

    # No llm_calls. The approver is authorised to DECIDE, not to read the
    # requester's prompts - the prompt log is a verbatim copy of what another
    # person typed, and approving an action is not a claim on it. The requester's
    # own UI already has the log from its /triage response.
    return TriageResponse(
        status="completed",
        thread_id=req.thread_id,
        route=final_state.get("route"),
        final_answer=final_state.get("final_answer"),
        citations=final_state.get("citations", []),
    )


@app.get("/session/{thread_id}")
def get_session(thread_id: str, caller: str = Depends(caller_id)) -> dict[str, Any]:
    """Return this thread's session events plus the user's user-level memory - for the UI panel.

    This used to return a user's entire cross-thread memory to anyone holding a
    thread id, and thread ids are printed in the UI. One id, every conversation.
    """
    assert_owns_thread(thread_id, caller)
    from app.memory import read_session_bundle
    b = read_session_bundle(thread_id, limit=20)
    return {"thread_id": thread_id, "user_id": b["user_id"],
            "events": b["session_events"], "user_events": b["user_events"]}


@app.post("/prefs")
def write_pref(req: PrefRequest, caller: str = Depends(caller_id)) -> dict[str, Any]:
    """Procedural memory write. Goes through the same scoped helper as every
    other write - there is no back door into a memory layer.

    Procedural memory shapes what the model is TOLD TO DO, so a preference
    somebody else can set for you is a prompt injection with a REST endpoint.
    """
    if caller != req.user_id:
        raise HTTPException(status_code=403, detail="cannot write another user's preferences")
    memory_write(user_scope(req.user_id), "procedural", {req.key: req.value})
    from app.memory import get_prefs
    return {"user_id": req.user_id, "prefs": get_prefs(req.user_id)}


def _delete_checkpoints_for(user_id: str) -> int:
    """Week 9 handed this destination to Week 10 by name.

    A checkpoint is a copy of what the user said. Framework checkpointers know
    it, which is why their interfaces carry a delete-everything-for-this-thread
    operation. Walk this user's threads and use it.
    """
    targets: set[str] = set()
    try:
        # COLLECT FIRST, THEN DELETE. list() walks the live store and
        # delete_thread mutates it, so deleting inside this loop cuts the walk
        # short: an earlier version of this function erased ONE of five threads
        # and the receipt cheerfully reported 1. A receipt that under-counts is
        # worse than no receipt, because it is evidence of a deletion that did
        # not happen.
        for t in _checkpointer.list(None):
            tid = (t.config or {}).get("configurable", {}).get("thread_id")
            if tid and (t.checkpoint.get("channel_values") or {}).get("user_id") == user_id:
                targets.add(tid)
    except Exception as exc:                      # a missing backend must not block the rest
        logger.warning("checkpoint scan skipped: %s", exc)
        return 0

    deleted = 0
    for tid in targets:
        try:
            _checkpointer.delete_thread(tid)
            deleted += 1
        except Exception as exc:
            logger.warning("checkpoint delete failed for %s: %s", tid, exc)
    return deleted


@app.delete("/user/{user_id}", response_model=DeletionReceipt)
def delete_user(user_id: str, caller: str = Depends(caller_id)) -> DeletionReceipt:
    """GDPR erasure. One request, every destination, and a receipt that proves it.

    Session keys in Redis, episodic summaries and any user-scoped external rows
    in the vector store, the procedural preferences row, AND this user's
    checkpoints. The global runbook corpus is scoped `global`, so it is
    untouched - by construction.
    """
    if caller != user_id:
        raise HTTPException(status_code=403, detail="cannot erase another user's data")
    from app.memory import delete_prefs, delete_user_long_term, session_clear_for_user

    sessions = session_clear_for_user(user_id)
    long_term = delete_user_long_term(user_id)
    procedural = delete_prefs(user_id)
    checkpoints = _delete_checkpoints_for(user_id)
    # The ownership map is a record about this person too.
    forget_user_threads(user_id)

    receipt = DeletionReceipt(
        user_id=user_id,
        session_keys_deleted=sessions,
        external_rows_deleted=long_term["external_rows_deleted"],
        episodic_rows_deleted=long_term["episodic_rows_deleted"],
        procedural_rows_deleted=procedural,
        checkpoint_threads_deleted=checkpoints,
    )
    logger.warning("triageflow.gdpr_delete %s", receipt.model_dump())
    return receipt


@app.get("/health")
def health() -> dict[str, Any]:
    """Standard liveness probe - returns model versions for the health chip."""
    s = get_settings()
    return {
        "status": "ok",
        "model": s.knowledge_model,   # primary model shown in health chip
        "models": {
            "triage": s.triage_model,
            "knowledge": s.knowledge_model,
            # The action model has its own pin, so it has to be its own line
            # here. A health chip that shows two of three pins is how a
            # re-pointed model stays invisible for a week.
            "action": s.action_model,
        },
    }


@app.get("/", include_in_schema=False)
def serve_ui():
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
