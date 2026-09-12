"""The DeskOrchestrator graph: one router, three specialists, one gate.

    START -> triage -> (conditional edge) -> access | software | hardware
                                                 \\-> compose   (escalate_human)
             access|software|hardware -> execute -> compose -> END

`execute` is a node name, not a route literal. The router emits exactly the four
strings in `Route` and reaches `execute` by a static edge from the specialist.
"""
from __future__ import annotations

import json
import logging
import operator
import re
import time
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app import llm
from app.config import get_settings
from app.memory import search_policies, search_tickets
from app.risk import risky
from app.schemas import DeskState
from app.state import assert_handoff
from app.tools import execute_tool, tools_schema

log = logging.getLogger(__name__)

ROUTES = ("access", "software", "hardware", "escalate_human")
_CITATION_RE = re.compile(r"\[doc:(\d+)\]")

ESCALATION_MESSAGE = (
    "This request needs a person. It touches something the desk does not decide on its own, "
    "so it has been routed to the service-desk team and nothing has been actioned.")


def _entry(node: str, r: llm.LLMResult) -> dict[str, Any]:
    return {"node": node, "model": r.model, "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens, "elapsed_ms": r.elapsed_ms}


# ------------------------------------------------------------------ triage
def triage_node(state: DeskState) -> dict[str, Any]:
    from prompts import TRIAGE_BASE, compose_system_prompt

    assert_handoff(state, ["user_request", "user_id"], "triage")
    s = get_settings()
    sys_prompt = compose_system_prompt(state["user_id"], TRIAGE_BASE)
    r = llm.call(s.triage_model, sys_prompt, state["user_request"], max_tokens=20)

    route = r.text.strip().lower().strip(' ".')
    if route not in ROUTES:
        # Fails to a default, logs, and the default is the LEAST CONSEQUENTIAL
        # branch for this risk model: the one that does nothing and asks a human.
        log.warning("unexpected route %r, defaulting to escalate_human", r.text[:40])
        route = "escalate_human"
    return {"route": route, "llm_calls": [_entry("triage", r)]}


def route_after_triage(state: DeskState) -> str:
    """The whole supervisor. Returns one of the ROUTE LITERALS, never a node name."""
    route = state.get("route")
    return route if route in ROUTES else "escalate_human"


# -------------------------------------------------------------- specialist
def _specialist(kind: str, state: DeskState) -> dict[str, Any]:
    from prompts import SPECIALIST_BASE, compose_system_prompt

    assert_handoff(state, ["user_request", "user_id", "route"], kind)
    s = get_settings()
    k = s.retrieval_k

    policies = search_policies(state["user_request"], k=k)
    tickets = search_tickets(state["user_request"], user_id=state["user_id"], k=k)

    # Emptiness is a PER-FIELD question and here it is a bug: composing a grounded
    # answer from zero policy chunks is the confident-wrong output we are avoiding.
    if not policies:
        raise ValueError(f"{kind}: policy retrieval returned no documents")

    retrieved = policies + tickets
    chunks = "\n\n".join(f"[doc:{i}] ({d['source']}) {d['text']}" for i, d in enumerate(retrieved))
    user_msg = (f"User request:\n{state['user_request']}\n\n"
                f"Retrieved chunks:\n{chunks}\n\nTools you may propose:\n{tools_schema()}")

    model = getattr(s, f"{kind}_model")
    r = llm.call(model, compose_system_prompt(state["user_id"], SPECIALIST_BASE[kind]), user_msg)

    try:
        parsed = json.loads(r.text)
    except json.JSONDecodeError:
        parsed = {"summary": r.text, "tool_name": "none", "arguments": {},
                  "rationale": "the model did not return JSON", "evidence": [], "alternative": ""}

    summary = str(parsed.get("summary", ""))
    cited = sorted({int(i) for i in _CITATION_RE.findall(summary) if int(i) < len(retrieved)})
    citations = [{"doc_index": i, "source": retrieved[i]["source"]} for i in cited]

    name = parsed.get("tool_name", "none")
    proposal = ({"tool_name": name, "arguments": parsed.get("arguments", {})}
                if name and name != "none" else {"tool_name": "none", "arguments": {}})

    return {
        "retrieved_policies": policies,
        "prior_tickets": tickets,
        "citations": citations,
        "specialist_summary": summary,
        "proposed_action": proposal,
        "action_rationale": str(parsed.get("rationale", "")),
        "action_evidence": [str(e) for e in parsed.get("evidence", [])],
        "action_alternative": str(parsed.get("alternative", "")),
        "pending_approval": risky(proposal),
        "llm_calls": [_entry(kind, r)],
    }


def access_node(state: DeskState) -> dict[str, Any]:
    return _specialist("access", state)


def software_node(state: DeskState) -> dict[str, Any]:
    return _specialist("software", state)


def hardware_node(state: DeskState) -> dict[str, Any]:
    return _specialist("hardware", state)


# ----------------------------------------------------------------- execute
def execute_node(state: DeskState) -> dict[str, Any]:
    """The only node in this system that calls an operational tool.

    Read the shape, because the shape is the safety property. There is exactly
    one `execute_tool` and it is the LAST statement, below every branch that can
    pause. Everything above it is a state read and a pure function, so it is safe
    to run twice - which matters, because resuming re-runs this node from the top.
    """
    assert_handoff(state, ["proposed_action", "pending_approval"], "execute")
    proposal = state["proposed_action"] or {"tool_name": "none", "arguments": {}}

    if proposal["tool_name"] == "none":
        return {"execution_result": {"success": True, "tool": "none", "result": "no action proposed"}}

    if risky(proposal):
        decision = interrupt({
            "about_to_happen": proposal,
            "reasoning": state.get("action_rationale", ""),
            "evidence": state.get("action_evidence", []),
            "alternative": state.get("action_alternative", ""),
        })
        if not decision.get("approved"):
            return {"execution_result": {"success": False, "tool": proposal["tool_name"],
                                         "result": "declined by approver",
                                         "reviewer_id": decision.get("reviewer_id")}}

    return {"execution_result": execute_tool(proposal["tool_name"], proposal["arguments"])}


# ----------------------------------------------------------------- compose
def compose_node(state: DeskState) -> dict[str, Any]:
    """The ONLY node that writes final_answer."""
    if state.get("route") == "escalate_human":
        return {"final_answer": ESCALATION_MESSAGE}

    assert_handoff(state, ["specialist_summary", "execution_result"], "compose")
    result = state["execution_result"] or {}
    tail = {
        True: "Done.",
        False: "Not done: " + str(result.get("result", "unknown")),
    }[bool(result.get("success"))]
    return {"final_answer": f"{state['specialist_summary']}\n\n{tail}"}


# ------------------------------------------------------------------- build
def build_graph(checkpointer=None):
    g: StateGraph = StateGraph(DeskState)
    g.add_node("triage", triage_node)
    g.add_node("access", access_node)
    g.add_node("software", software_node)
    g.add_node("hardware", hardware_node)
    g.add_node("execute", execute_node)
    g.add_node("compose", compose_node)

    g.add_edge(START, "triage")
    g.add_conditional_edges(
        "triage",
        route_after_triage,
        # keys are what the ROUTER can emit; values are NODE NAMES. Keep the two
        # columns straight and the dead-key bug cannot form.
        {"access": "access", "software": "software",
         "hardware": "hardware", "escalate_human": "compose"},
    )
    for worker in ("access", "software", "hardware"):
        g.add_edge(worker, "execute")
    g.add_edge("execute", "compose")
    g.add_edge("compose", END)

    return g.compile(checkpointer=checkpointer)
