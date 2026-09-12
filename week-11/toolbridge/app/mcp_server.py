"""The MCP server. Registration, the elicitation variant of the human gate, and main().

SDK NOTE, and it is the first thing to know about this project. We pin
`mcp==2.2.0`, which speaks the 2026-07-28 revision:

  * MCPServer, not FastMCP. `mcp.server.fastmcp` does not exist in 2.x and
    importing it raises a ModuleNotFoundError that tells you so.
  * Stateless. No `initialize` handshake; every request carries its own
    protocol version and capabilities.
  * InputRequiredResult exists, so a server CAN pause and ask a human.

The two guided builds this week pin `mcp==1.28.1`, which speaks 2025-11-25.
They are different majors and they need different virtual environments.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Context, Elicit, MCPServer, Resolve
from pydantic import BaseModel, Field

from app.config import get_settings
from app.descriptions import describe
from app.errors import fail
from app.tools import delegate_triage as _delegate
from app.tools import peer_card as _peer_card
from app.tools import resume_task as _resume

logger = logging.getLogger("toolbridge")

_settings = get_settings()
_quality = _settings.tool_description_quality

mcp = MCPServer(
    name="toolbridge",
    instructions=(
        "Bridges a remote A2A triage agent onto MCP. Discover the peer with peer_card, "
        "send it work with delegate_triage, and finish a paused task with resume_task."
    ),
)


# --------------------------------------------------------------- the human gate
class ApprovalDecision(BaseModel):
    """What we ask a human when HUMAN_GATE_MODE=elicit."""

    approved: bool = Field(description="Approve the action the peer proposed?")
    reviewer_id: str = Field(description="Your identifier, recorded on the peer's audit trail.")
    note: str = Field(default="", description="Optional reason, recorded with the decision.")


async def _ask_human(ctx: Context) -> Elicit[ApprovalDecision] | None:
    """A resolver. Returns a request marker so the SDK does the round trip.

    On a 2026-07-28 client this becomes an InputRequiredResult: we return a
    result meaning "not finished", the client asks its human, and the client
    RETRIES the same call carrying inputResponses and the requestState we
    handed back. On a <= 2025-11-25 client the SDK sends a standalone
    elicitation request instead. Same code, two wire shapes, chosen by the
    version on the request.

    Returns None when the mode is `resume`, in which case nothing is asked and
    the tool takes the structured-pending path instead.
    """
    if get_settings().human_gate_mode != "elicit":
        return None
    return Elicit(
        "The peer has paused this task and proposed an action. Approve it?",
        ApprovalDecision,
    )


# --------------------------------------------------------------- registrations
async def peer_card(peer_url: str | None = None) -> dict[str, Any]:
    return await _peer_card(peer_url)


async def delegate_triage(
    description: str,
    severity: Literal["low", "medium", "high"],
    user_id: str,
    peer_url: str | None = None,
) -> dict[str, Any]:
    return await _delegate(description, severity, user_id, peer_url)


async def resume_task(
    task_id: str,
    decision: Literal["approve", "reject"],
    reviewer_id: str,
    resume_from: int = 0,
    note: str = "",
    peer_url: str | None = None,
) -> dict[str, Any]:
    return await _resume(task_id, decision, reviewer_id, resume_from, note, peer_url)


# Registered as manual calls rather than decorators so the plain functions above
# stay importable and testable without the server object in the way.
mcp.tool(name="peer_card", description=describe("peer_card", _quality))(peer_card)
mcp.tool(name="delegate_triage", description=describe("delegate_triage", _quality))(delegate_triage)
mcp.tool(name="resume_task", description=describe("resume_task", _quality))(resume_task)


async def list_tool_infos() -> list[dict[str, Any]]:
    """What tools/list would return. Used by the tests and the eval."""
    tools = await mcp.list_tools()
    # Note the attribute name. On the WIRE the field is `inputSchema`; in the
    # 2.x Python SDK the model attribute is `input_schema`, aliased. Reading the
    # wire name off a Python object is a small, reliable way to waste an hour.
    return [
        {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
        for t in tools
    ]


def main() -> None:
    """The `toolbridge` entrypoint from pyproject.toml.

    One argument decides local versus remote. Everything above this line is
    identical either way, which is the whole point of a transport-agnostic
    protocol.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    settings = get_settings()
    transport = "streamable-http" if settings.mcp_server_transport == "http" else "stdio"
    logger.info("toolbridge start transport=%s peer=%s gate=%s descriptions=%s",
                transport, settings.peer_base_url, settings.human_gate_mode, _quality)
    if transport == "streamable-http":
        # In mcp 2.x the port is an argument to run(), not a mutable settings
        # object on the server. The 1.x habit of `mcp.settings.port = ...`
        # raises a pydantic ValueError here, which is one of the small, loud
        # differences between the two majors.
        mcp.run(transport=transport, host="127.0.0.1", port=settings.mcp_server_port)
    else:
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()
