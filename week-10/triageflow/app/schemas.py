"""Pydantic + TypedDict schemas - the contract layer."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


# --- Domain models (Pydantic for API I/O) ---

class TriageRequest(BaseModel):
    user_request: str = Field(..., min_length=1, max_length=4000)
    user_id: str = Field(..., min_length=1)
    thread_id: str | None = None  # generated server-side if absent


class TriageResponse(BaseModel):
    status: Literal["completed", "pending_approval", "escalated"]
    thread_id: str
    # Which branch ran. The UI used to infer this from the status, which worked
    # only while every action paused: "completed" then meant "knowledge". With a
    # conditional gate a low-risk action completes too, so the route has to be
    # reported rather than guessed.
    route: str | None = None
    final_answer: str | None = None
    proposed_action: dict | None = None
    citations: list[dict] = Field(default_factory=list)
    llm_calls: list[dict] = Field(default_factory=list)  # prompt log for the UI


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool
    reviewer_id: str
    notes: str | None = None


class PrefRequest(BaseModel):
    """A procedural-memory write. One key, one value, one user."""
    user_id: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)
    value: object


class DeletionReceipt(BaseModel):
    """One counter per destination this system actually has.

    Note that is DESTINATIONS, not layers, and the two lists differ. In-context
    memory is a layer with nothing durable to erase, because the prompt is
    rebuilt every turn. The checkpoint table is not a memory layer at all and is
    absolutely a destination: a checkpoint is a verbatim copy of what a user
    said, so whatever obligations attach to remembering their words attach to
    those rows too. A deletion path that clears the memory index and leaves the
    checkpoint table has not deleted anything.

    `external_rows_deleted` is structurally zero, and that zero proves less than
    it looks like it proves: a counter is evidence about what WAS deleted and
    says nothing about what was left. The over-broad case needs a test, and
    there is one.
    """
    user_id: str
    session_keys_deleted: int
    external_rows_deleted: int
    episodic_rows_deleted: int
    procedural_rows_deleted: int
    checkpoint_threads_deleted: int


class Doc(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float


class Citation(BaseModel):
    doc_index: int
    source: str
    span: str


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict


# --- LangGraph shared state (TypedDict) ---
# RULE: a field belongs here only if at least two nodes need to read it, or if
# the trace must show it. All other working scratch stays inside the node.
#
# total=False because routes are exclusive: a knowledge run never populates the
# action keys, so no run carries every key and a type checker cannot enforce
# presence. assert_handoff does that job at run time instead.

class TriageState(TypedDict, total=False):
    # Travels WITH the request. No node owns it and no node branches on it.
    user_request: str
    thread_id: str
    user_id: str

    # written by Triage
    route: Literal["knowledge", "action", "escalate_human"]

    # written by Knowledge
    retrieved_docs: list[dict]      # serialised Doc - external memory, scope=global
    prior_incidents: list[dict]      # serialised Doc - episodic memory, scope=user:{uid}
    citations: list[dict]            # serialised Citation
    knowledge_summary: str

    # written by Action, which builds the proposal and therefore writes every
    # field that describes it, including the three the gate shows the human.
    proposed_action: dict | None     # serialised ToolCall
    # The proposal's identity and the world it was made in. All three are
    # written before the pause and read after it, which is the only reason they
    # are on the contract rather than inside the node: a value that must survive
    # a resume cannot live in a local variable.
    proposal_id: str
    proposal_created_at: float
    resource_version: str
    action_rationale: str
    action_evidence: list[str]
    action_alternative: str
    pending_approval: bool

    # written by action_execute, the only node that calls an operational tool
    execution_result: dict | None    # {"executed": bool, "tool_name": ..., "result": ...}

    # written by Compose - the ONLY node that writes final_answer
    final_answer: str | None

    # Append-only channel. Three nodes write it, so it carries a REDUCER rather
    # than an owner: the sanctioned exception to the single-writer rule. A plain
    # field written by three nodes and merged by luck is a race with a schema.
    llm_calls: Annotated[list, operator.add]

# NOTE on what is deliberately NOT here: the approver's identity. `approved` and
# `reviewer_id` used to sit on this contract, where they looked like something a
# node could read and branch on. They arrive as the interrupt() resume payload
# instead, and belong to the request's audit trail rather than to the schema
# passing between nodes - for the same reason a request id does not belong here.
