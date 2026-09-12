"""TriageFlow LangGraph definition - three agents + a dynamic interrupt() gate."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from .config import get_settings
from .llm import call
from .memory import search_episodic, search_runbooks
from .risk import risky
from .schemas import TriageState
from .state import assert_handoff, assert_retrieved
from .tools import execute_tool, resource_version, tools_schema

log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[doc:(\d+)\]")


# ---- agent nodes ----

def triage_node(state: TriageState) -> dict[str, Any]:
    from prompts import TRIAGE_BASE, compose_system_prompt
    assert_handoff(state, ["user_request", "user_id"], "triage")
    sys_prompt = compose_system_prompt(state["user_id"], TRIAGE_BASE)
    user_msg = state["user_request"]
    result = call(
        model=get_settings().triage_model,
        system=sys_prompt,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=20,
    )
    route = result.text.strip().lower().strip(' "')
    if route not in {"knowledge", "action", "escalate_human"}:
        # Fails to a default, logs, and the default is the LEAST CONSEQUENTIAL
        # branch for this risk model rather than the cheapest one. Answering an
        # unroutable incident with a runbook paragraph while the disk fills is
        # worse than admitting the system does not know.
        log.warning("triage_node: unexpected route '%s', defaulting to escalate_human", route)
        route = "escalate_human"
    entry = {
        "step": "triage", "model": result.model,
        "system": sys_prompt[:600] + ("…" if len(sys_prompt) > 600 else ""),
        "user": user_msg[:600],
        "response": result.text.strip(),
        "latency_ms": result.latency_ms,
        "tokens_in": result.input_tokens, "tokens_out": result.output_tokens,
    }
    # Only this node's own entry. llm_calls carries a reducer, and the reducer
    # does the appending - adding the upstream list here appends it twice.
    return {"route": route, "llm_calls": [entry]}


def knowledge_node(state: TriageState) -> dict[str, Any]:
    from prompts import KNOWLEDGE_BASE, compose_system_prompt
    assert_handoff(state, ["user_request", "user_id", "route"], "knowledge")
    # Two retrievals, two scopes. Runbooks are scoped `global`; prior incidents
    # are scoped to this user and can never leak across tenants.
    docs = search_runbooks(state["user_request"], k=5)
    prior = search_episodic(state["user_request"], user_id=state["user_id"], k=3)
    # Per-field emptiness check, with a message that says what is empty.
    # Composing a grounded answer from zero chunks is the confident-wrong output
    # this whole layer exists to avoid.
    assert_retrieved(docs, "knowledge")
    retrieved = docs + prior
    chunks = "\n\n".join(f"[doc:{i}] (source={d['source']}) {d['text']}"
                         for i, d in enumerate(retrieved))
    sys_prompt = compose_system_prompt(state["user_id"], KNOWLEDGE_BASE)
    user_msg = f"User request:\n{state['user_request']}\n\nRetrieved chunks:\n{chunks}"

    result = call(
        model=get_settings().knowledge_model,
        system=sys_prompt,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=800,
    )
    # Parse the [doc:N] markers out of the answer - cite only what the model
    # actually used. Fall back to all retrieved docs if no markers were found.
    cited = sorted({int(m) for m in _CITATION_RE.findall(result.text) if int(m) < len(retrieved)})
    if cited:
        citations = [{"doc_index": i, "source": retrieved[i]["source"]} for i in cited]
    else:
        log.warning("knowledge_node: no [doc:N] markers in answer, citing all retrieved docs")
        citations = [{"doc_index": i, "source": d["source"]} for i, d in enumerate(retrieved)]
    entry = {
        "step": "knowledge", "model": result.model,
        "system": sys_prompt[:600] + ("…" if len(sys_prompt) > 600 else ""),
        "user": user_msg[:800] + ("…" if len(user_msg) > 800 else ""),
        "response": result.text[:600] + ("…" if len(result.text) > 600 else ""),
        "latency_ms": result.latency_ms,
        "tokens_in": result.input_tokens, "tokens_out": result.output_tokens,
    }
    return {
        "retrieved_docs": docs,
        "prior_incidents": prior,
        "citations": citations,
        "knowledge_summary": result.text,
        "llm_calls": [entry],           # the reducer appends; see triage_node
    }


def action_node(state: TriageState) -> dict[str, Any]:
    from prompts import ACTION_BASE, compose_system_prompt
    assert_handoff(state, ["user_request", "route"], "action")
    sys_prompt = compose_system_prompt(state["user_id"], ACTION_BASE)
    user_msg = (
        f"User request:\n{state['user_request']}\n\n"
        f"Available tools:\n{json.dumps(tools_schema(), indent=2)}"
    )
    result = call(
        model=get_settings().action_model,
        system=sys_prompt,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=400,
    )
    try:
        proposal = json.loads(result.text.strip())
    except json.JSONDecodeError:
        log.warning("action_node: model returned non-JSON, treating as no-op")
        proposal = {"tool_name": "none", "arguments": {}}
    entry = {
        "step": "action", "model": result.model,
        "system": sys_prompt[:600] + ("…" if len(sys_prompt) > 600 else ""),
        "user": user_msg[:800] + ("…" if len(user_msg) > 800 else ""),
        "response": result.text.strip()[:400],
        "latency_ms": result.latency_ms,
        "tokens_in": result.input_tokens, "tokens_out": result.output_tokens,
    }
    args = proposal.get("arguments", {}) or {}
    return {
        "proposed_action": proposal,
        # The proposal's identity, minted ONCE and upstream of the pause. This
        # node does not re-run on resume, so this id survives the replay - which
        # is exactly the property a fresh uuid inside the executing node lacks.
        "proposal_id": f"prop_{uuid.uuid4().hex[:12]}",
        "proposal_created_at": time.time(),
        # What the world looked like when the plan was made, for comparison
        # after the human decides.
        "resource_version": resource_version(proposal.get("tool_name", "none"), args),
        "action_rationale": str(proposal.get("rationale", "")),
        "action_evidence": [str(e) for e in (proposal.get("evidence") or [])],
        "action_alternative": str(proposal.get("alternative", "")),
        # The flag a UI or a supervisor polls. Nothing branches on it: the
        # executing node decides for itself, via risky(), for the same reason.
        "pending_approval": risky(proposal),
        "llm_calls": [entry],
    }


def idempotency_key(state: TriageState) -> str:
    """Derived from the thread id and the proposal id, both of which are already
    in state before the pause, so this returns the SAME key on a replay.

    Deriving it from time.time() or a fresh uuid4() here would produce a new key
    on every entry to the node, which is indistinguishable from having no key.
    """
    raw = f"{state.get('thread_id', '')}:{state.get('proposal_id', '')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def stale_reason(state: TriageState, proposal: dict[str, Any]) -> str | None:
    """Is this approved plan still about the world it was made in?

    Two ways it is not: the approval window lapsed, or the resource moved. Both
    are pure reads, and both must be checked BELOW the pause - checking above it
    measures the moment the proposal was made, which is not the question.
    """
    ttl = get_settings().approval_ttl_seconds
    made_at = state.get("proposal_created_at")
    if made_at is not None and (time.time() - float(made_at)) > ttl:
        return "approval_expired"
    was = state.get("resource_version")
    now = resource_version(proposal.get("tool_name", "none"), proposal.get("arguments", {}) or {})
    if was is not None and now != was:
        return "stale_proposal"
    return None


def action_execute_node(state: TriageState) -> dict[str, Any]:
    """The only node in this system that calls an operational tool.

    Read the shape, because the shape is the safety property. There is exactly
    one `execute_tool` and it is the LAST statement, below every branch that can
    pause, so no arrangement of the branches can put a side effect above the
    pause. Everything above it is a state read and a pure function, which matters
    because RESUMING RE-RUNS THIS NODE FROM THE TOP.

    The gate is a dynamic interrupt() rather than a compile-time interrupt_before,
    because approval is conditional: you want to pause for the risky action, not
    for every visit to the node. The vendor's own documentation files static
    interrupts under debugging and says they are not recommended for
    human-in-the-loop. That conditionality is the reason, and it is why risky()
    is a policy this codebase owns and tests.

    Two things guard the call at the bottom, and both exist because the human
    pause is long. The IDEMPOTENCY KEY handles the crash window: if the tool
    succeeds and the process dies before the returned state update is
    checkpointed, the replay must not fire the action twice. The STALENESS CHECK
    handles the other end: the world can move while the approver decides, and a
    plan that was right when approved can be wrong when executed. Both checks
    read state that was written before the pause, which is what makes them work.

    What is still missing, named rather than hidden: the ledger and the record of
    the call are not written in one transaction, so a crash between them can
    still lose the record. In production, one transaction or one store.
    """
    assert_handoff(state, ["proposed_action", "pending_approval", "proposal_id"], "action_execute")
    proposed = state.get("proposed_action") or {}
    name = proposed.get("tool_name", "none")
    if name == "none":
        return {"execution_result": {"executed": False, "reason": "no_tool_proposed"}}

    if risky(proposed):
        decision = interrupt({
            "about_to_happen": proposed,
            "reasoning": state.get("action_rationale", ""),
            "evidence": state.get("action_evidence", []),
            "alternative": state.get("action_alternative", ""),
        })
        if not decision.get("approved"):
            return {"execution_result": {"executed": False, "reason": "declined",
                                         "reviewer_id": decision.get("reviewer_id")}}

    # Re-read the world and refuse a plan it has moved past: approving a restart
    # is not the same as approving it forty minutes later, after someone else
    # already did it.
    #
    # This sits below the pause because that is where the intent is clearest.
    # It is NOT load-bearing, and it is worth knowing why: everything above the
    # interrupt re-executes on resume, so the same check placed above it would
    # also fire on the second entry and catch exactly the same case. Measured,
    # both ways. The statement whose POSITION genuinely matters is the mutating
    # call below, because re-executing that one changes the world twice.
    stale = stale_reason(state, proposed)
    if stale is not None:
        log.warning("action_execute: refusing %s, reason=%s", name, stale)
        return {"execution_result": {"executed": False, "reason": stale, "tool_name": name}}

    # One mutating call, and it is the node's last statement. The key is what
    # makes a replay of this line safe; see idempotency_key() for why it is
    # derived from state rather than generated here.
    result = execute_tool(name, proposed.get("arguments", {}),
                          idempotency_key=idempotency_key(state))
    return {"execution_result": {"executed": True, "tool_name": name, "result": result}}


def compose_node(state: TriageState) -> dict[str, Any]:
    """Final answer assembly - the single writer of final_answer."""
    route = state.get("route")

    if route == "knowledge":
        assert_handoff(state, ["retrieved_docs", "knowledge_summary"], "compose")
        return {"final_answer": state["knowledge_summary"]}

    if route == "action":
        assert_handoff(state, ["execution_result"], "compose")
        ar = state["execution_result"] or {}
        if not ar.get("executed"):
            reason = ar.get("reason")
            if reason == "declined":
                return {"final_answer": "Action declined by reviewer. No changes were made."}
            # A refusal has to say WHICH refusal. Reporting a stale plan as "no
            # tool was proposed" tells the user something that is not true, and
            # sends them looking for a bug in the wrong agent.
            if reason == "stale_proposal":
                return {"final_answer": "The approved action was not executed: the resource changed "
                                        "after approval. Nothing was done. Re-run to get a fresh plan."}
            if reason == "approval_expired":
                return {"final_answer": "The approved action was not executed: the approval window "
                                        "lapsed. Nothing was done. Re-run to get a fresh plan."}
            return {"final_answer": "No tool was proposed."}
        return {"final_answer": f"Action complete: {ar['tool_name']} -> {json.dumps(ar['result'])}"}

    if route == "escalate_human":
        return {"final_answer": "This case has been escalated for human review."}
    return {"final_answer": "No answer produced."}


# ---- routing functions ----

def route_after_triage(state: TriageState) -> str:
    """The whole supervisor. Returns one of the ROUTE LITERALS, never a node name.

    Returning "compose" here would look right and be dead: compose is a node and
    it is not something the router's contract permits, so the branch the router
    CAN emit would fall off the end of the graph. The edge map does the mapping.
    """
    route = state.get("route", "escalate_human")
    if route in ("knowledge", "action", "escalate_human"):
        return route
    return "escalate_human"


# ---- graph factory ----

def build_graph(checkpointer=None):
    # Node-level retries. Be honest about what this composes with: llm.py
    # already retries the network call three times via tenacity, so the worst
    # case here is 3 x 3 = NINE model calls for one node, not three. It does not
    # "catch anything that escapes" the inner layer; it multiplies it. Two is
    # deliberate: enough to survive a blip, small enough that the product stays
    # readable on a bill.
    llm_retry = RetryPolicy(max_attempts=2)

    g: StateGraph = StateGraph(TriageState)
    g.add_node("triage", triage_node, retry_policy=llm_retry)
    g.add_node("knowledge", knowledge_node, retry_policy=llm_retry)
    g.add_node("action", action_node, retry_policy=llm_retry)
    g.add_node("action_execute", action_execute_node)
    g.add_node("compose", compose_node)

    g.add_edge(START, "triage")
    g.add_conditional_edges(
        "triage",
        route_after_triage,
        # keys are what the ROUTER can emit; values are NODE NAMES. The two
        # columns coincide on two of three rows, which is what tempts people into
        # writing a "compose" key. Keep them straight and the dead-key bug cannot form.
        {"knowledge": "knowledge", "action": "action", "escalate_human": "compose"},
    )
    g.add_edge("knowledge", "compose")
    g.add_edge("action", "action_execute")
    g.add_edge("action_execute", "compose")
    g.add_edge("compose", END)

    # No interrupt_before here. The gate lives inside action_execute as a
    # dynamic interrupt(), which needs a checkpointer and nothing else.
    return g.compile(checkpointer=checkpointer)
