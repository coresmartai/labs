"""Pydantic API models, and the TypedDict handoff contract."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

Route = Literal["access", "software", "hardware", "escalate_human"]


# ------------------------------------------------------------- API boundary
class DeskRequest(BaseModel):
    user_request: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    thread_id: str | None = None


class DeskResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "pending_approval"]
    route: Route | None = None
    final_answer: str | None = None
    proposed_action: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = []
    llm_calls: list[dict[str, Any]] = []


class ApproveRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    reviewer_id: str = Field(min_length=1)


class DeletionReceipt(BaseModel):
    user_id: str
    policy_rows_deleted: int
    ticket_rows_deleted: int
    procedural_rows_deleted: int
    checkpoint_threads_deleted: int


# ------------------------------------------------------------- the contract
#
# RULE: a field belongs here only if at least two nodes read it, or if the trace
# must show it. All other working scratch stays inside the node that made it.
#
# The approver's identity is NOT on this contract. It arrives as the interrupt()
# resume payload and belongs to the request's audit trail, not to the schema
# passing between nodes - for the same reason a request id does not belong here.
#
# total=False because routes are exclusive: an access run never populates the
# hardware keys, so no run carries every key and a type checker cannot enforce
# presence. assert_handoff does that job at run time instead.
class DeskState(TypedDict, total=False):
    # travels with the request; no node owns it and no node branches on it
    user_request: str
    thread_id: str
    user_id: str

    route: Route                       # written by triage

    retrieved_policies: list[dict]     # written by the specialist. scope=global
    prior_tickets: list[dict]          # written by the specialist. scope=user
    citations: list[dict]
    specialist_summary: str

    proposed_action: dict | None       # written by the specialist
    action_rationale: str
    action_evidence: list[str]
    action_alternative: str
    pending_approval: bool

    execution_result: dict | None      # written by execute, the only mutating node

    # written by compose, and ONLY by compose
    final_answer: str

    # append-only channel. Four nodes write it, so it carries a reducer rather
    # than an owner - the sanctioned exception to the single-writer rule.
    llm_calls: Annotated[list, operator.add]
