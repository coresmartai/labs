"""FastAPI surface.

  POST   /desk              start a request
  POST   /approve           resolve a pending action
  POST   /prefs             write a procedural rule
  DELETE /user/{user_id}    right to be forgotten, with a receipt
  GET    /health
  GET    /debug/retrieval   what a scoped search returned, next to what the user owns
  GET    /                  the browser UI
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph import build_graph
from app.memory import delete_user, memory_write, seed_demo_prefs, seed_memory, user_scope
from app.schemas import ApproveRequest, DeletionReceipt, DeskRequest, DeskResponse

logging.basicConfig(level=get_settings().log_level)
log = logging.getLogger(__name__)

_checkpointer = MemorySaver()
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    seed_memory()
    seed_demo_prefs()
    _graph = build_graph(checkpointer=_checkpointer)
    yield


app = FastAPI(title="DeskOrchestrator", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # Wide open because this is a local teaching demo with no auth in front of it.
    # In production this is the line that lets any page you visit call DELETE /user.
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _cfg(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": get_settings().recursion_limit}


def _respond(thread_id: str, snap_values: dict[str, Any], next_nodes: tuple) -> DeskResponse:
    pending = "execute" in next_nodes
    return DeskResponse(
        thread_id=thread_id,
        status="pending_approval" if pending else "completed",
        route=snap_values.get("route"),
        final_answer=snap_values.get("final_answer"),
        proposed_action=snap_values.get("proposed_action") if pending else None,
        citations=snap_values.get("citations", []),
        llm_calls=snap_values.get("llm_calls", []),
    )


@app.post("/desk", response_model=DeskResponse)
def desk(req: DeskRequest) -> DeskResponse:
    thread_id = req.thread_id or f"th_{uuid.uuid4().hex[:12]}"
    cfg = _cfg(thread_id)
    _graph.invoke({"user_request": req.user_request, "user_id": req.user_id,
                   "thread_id": thread_id}, config=cfg)
    snap = _graph.get_state(cfg)
    return _respond(thread_id, snap.values, snap.next)


@app.post("/approve", response_model=DeskResponse)
def approve(req: ApproveRequest) -> DeskResponse:
    cfg = _cfg(req.thread_id)
    snap = _graph.get_state(cfg)
    if "execute" not in snap.next:
        # Without this the call silently succeeds, the real pending thread sits at
        # the gate forever, and the reviewer never gets a response. Three lines.
        raise HTTPException(status_code=409,
                            detail="thread is not paused at the approval gate")
    # The reviewer's identity travels WITH the request, in the resume payload.
    # It is not written onto the contract, so no node can read or branch on it.
    _graph.invoke(Command(resume={"approved": req.approved, "reviewer_id": req.reviewer_id}),
                 config=cfg)
    snap = _graph.get_state(cfg)
    return _respond(req.thread_id, snap.values, snap.next)


class PrefRequest(BaseModel):
    user_id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    value: Any


@app.post("/prefs")
def prefs(req: PrefRequest) -> dict[str, Any]:
    memory_write(user_scope(req.user_id), "procedural", {"key": req.key, "value": req.value})
    from app.memory import get_prefs
    return {"user_id": req.user_id, "prefs": get_prefs(req.user_id)}


@app.delete("/user/{user_id}", response_model=DeletionReceipt)
def forget(user_id: str) -> DeletionReceipt:
    counts = delete_user(user_id)
    # Week 9 handed this destination over by name: a deletion path that clears the
    # memory index and leaves the checkpoint table has not deleted anything.
    # list() yields one entry per CHECKPOINT, and a thread has several. Collect
    # the distinct thread ids first, then delete: counting entries instead of
    # threads made this receipt report fifteen for three, and a receipt that
    # over-states is as useless as one that under-states.
    targets = {
        t.config["configurable"]["thread_id"]
        for t in list(_checkpointer.list(None))
        if (t.config or {}).get("configurable", {}).get("thread_id")
        and (t.checkpoint.get("channel_values") or {}).get("user_id") == user_id
    }
    for tid in targets:
        _checkpointer.delete_thread(tid)
    return DeletionReceipt(**counts, checkpoint_threads_deleted=len(targets))


@app.get("/debug/retrieval")
def debug_retrieval(q: str, user_id: str, k: int = 3) -> dict[str, Any]:
    """What a scoped ticket search RETURNED, beside how many rows that user OWNS.

    Both numbers come from the store's own public interface: one search and one
    count. Nothing here reimplements the ranking, and nothing here is part of
    the graded work - it exists so that a recall failure stops being an empty
    panel and becomes two numbers that disagree.

    On a fresh clone, a query that matches nothing lexically will return 0 of 3.
    That is not a leak, and the difference matters: post-filtering still filters,
    so no other user's rows can reach this caller whatever the order of the two
    steps. What breaks is which rows get CONSIDERED.
    """
    from app.memory import TICKETS_COLL, search_tickets, user_scope
    from app.vectorstore import get_store

    returned = search_tickets(q, user_id=user_id, k=k)
    owned = get_store().count(user_scope(user_id), TICKETS_COLL)
    return {
        "query": q,
        "user_id": user_id,
        "k": k,
        "owned_by_this_user": owned,
        "returned": returned,
        "returned_count": len(returned),
        # The verdict, so the panel does not have to infer it.
        "recall_ok": not (owned > 0 and len(returned) == 0),
    }


@app.get("/", include_in_schema=False)
def serve_ui():
    idx = Path(__file__).parent.parent / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(idx, media_type="text/html")


@app.get("/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {"status": "ok", "models": {"triage": s.triage_model, "access": s.access_model,
                                       "software": s.software_model, "hardware": s.hardware_model}}
