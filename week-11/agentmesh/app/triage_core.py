"""TriageFlow core - the agent. Imported pattern from Week 10.

In a real repo this would be `from coresmart_w10_triageflow import build_graph`.
Here we ship a minimal in-file equivalent so the package is self-contained.

Three nodes (Triage -> Knowledge/Action -> Compose), one approval gate, and
**zero awareness of HTTP, JWT, SSE, A2A or MCP**. It talks to the outside world
through two callbacks it is handed - `on_progress` and `wait_for_approval` - and
that is the entire reason wrapping it in A2A changed no agent code.

The three model-shaped steps - classify, extract, synthesize - go through
`app/llm.py`, which runs a deterministic rule with the real model call shown
commented right above it. So nothing dials a provider on this path; flip those
comments in `llm.py` to make it live.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from app.llm import get_llm
from app.schemas import TriageOutput
from app.triage_tools import execute_tool

logger = logging.getLogger(__name__)


class RejectedByCaller(Exception):
    """The approval gate was answered `approved: false`.

    The agent does not know what a "task state" is - it just unwinds without
    executing anything. The wrapper is what turns this into TASK_STATE_REJECTED.
    """

    def __init__(self, proposal: dict[str, Any], note: str | None = None) -> None:
        super().__init__("action rejected by caller")
        self.proposal = proposal
        self.note = note


async def run_triage(
    description: str,
    severity: str,
    user_id: str,
    on_progress: Callable[[str, dict[str, Any]], Awaitable[None]],
    wait_for_approval: Callable[[dict[str, Any]], Awaitable[bool]],
) -> TriageOutput:
    """Run the TriageFlow specialist on one incident.

    on_progress       - coroutine called for every node transition: (node, payload).
    wait_for_approval - coroutine that pauses until the caller approves a proposed
                        action. Returns False -> we raise RejectedByCaller and nothing
                        mutating ever runs.
    """
    await on_progress("triage", {"message": "classifying", "user_id": user_id})
    category = await get_llm().classify_incident(description, severity)

    if category == "knowledge":
        await on_progress("knowledge", {"message": "retrieving", "user_id": user_id})
        runbook = execute_tool("lookup_runbook", {"query": description})
        summary = await get_llm().synthesize_answer(runbook)
        return TriageOutput(
            category="knowledge",
            summary=summary,
            citations=[runbook["runbook"]["title"]] if runbook.get("runbook") else [],
        )

    if category == "action":
        await on_progress("action", {"message": "proposing", "user_id": user_id})
        proposal = execute_tool("propose_action", await get_llm().extract_action(description))
        approved = await wait_for_approval(proposal)
        if not approved:
            # Terminal, and nothing was executed. The wrapper emits TASK_STATE_REJECTED.
            raise RejectedByCaller(proposal["proposal"])
        await on_progress("action", {"message": "executing", "user_id": user_id})
        return TriageOutput(
            category="action",
            summary=f"Executed: {proposal['proposal']['remediation']} on {proposal['proposal']['target']}",
            proposed_action=proposal["proposal"]["remediation"],
        )

    return TriageOutput(
        category="escalate",
        summary="Severity high enough to page on-call. Escalated.",
    )
