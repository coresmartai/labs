"""Cohort-shared MCP server, built on the official `mcp` Python SDK's FastMCP.

Three of MCP's primitives are exercised here on purpose, because "MCP" is not a
synonym for "tool calling":

  * **Tools** - model-invokable functions with typed arguments:
      - whoami()                  -> who runs this agent (the cohort ice-breaker)
      - cohort_lookup(student_id) -> a classmate's profile
      - incident_history(query)   -> matching incidents
      - ping(message)             -> round-trip/latency check
  * **Resources** - read-only addressable data a host can list and fetch with no
    model in the loop:
      - agentmesh://student/profile      this student's identity
      - agentmesh://incidents            the whole incident list
      - agentmesh://incident/{id}        one incident (a resource *template*)
  * **Prompts** - reusable parameterised templates the server ships, so a host can
    surface them as slash-commands:
      - triage_brief(description, severity)
      - peer_intro()

Two ways to run it:

  * **Standalone** - `python -m app.mcp_server`. stdio by default (what Claude
    Desktop talks to); MCP_SERVER_TRANSPORT=http serves Streamable HTTP on
    MCP_SERVER_PORT. No auth on that path: it is a local fixture server.
  * **Co-hosted** - `app/main.py` mounts this same instance at MCP_MOUNT_PATH on
    the A2A port, so ONE tunnel exposes both protocols. That surface IS
    bearer-gated, because it is reachable from the internet.

The tool functions stay plain, undecorated callables in `app/tools.py`
(registration is a side effect of calling `mcp.tool(...)` as a function), so tests
import and call them with no server involved at all.
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from app.config import get_settings
from app.identity import student_identity
from app.tools import (
    all_incidents,
    cohort_lookup,
    incident_history,
    ping,
    tool_description,
    whoami,
)

logger = logging.getLogger(__name__)

_settings = get_settings()
_quality = _settings.tool_description_quality

mcp = FastMCP(
    "agentmesh-cohort",
    host=_settings.agentmesh_host,
    port=_settings.mcp_server_port,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Registered as plain function calls, not `@mcp.tool()` decorators, so every tool
# stays directly callable and testable exactly as it was before this module ran a
# server. Descriptions come from the table in `app/tools.py`, not from a literal
# here, so the wording lives next to the implementation it describes.
mcp.tool(name="whoami", description=tool_description("whoami", _quality))(whoami)
mcp.tool(name="cohort_lookup", description=tool_description("cohort_lookup", _quality))(cohort_lookup)
mcp.tool(name="incident_history", description=tool_description("incident_history", _quality))(incident_history)
mcp.tool(name="ping", description=tool_description("ping", _quality))(ping)


# ---------------------------------------------------------------------------
# Resources - addressable, read-only, no model in the loop
# ---------------------------------------------------------------------------


@mcp.resource("agentmesh://student/profile", mime_type="application/json")
def resource_profile() -> str:
    """This student's identity, as a fetchable document rather than a tool call."""
    return json.dumps(student_identity(), indent=2)


@mcp.resource("agentmesh://incidents", mime_type="application/json")
def resource_incidents() -> str:
    """The full incident list. A host can read this straight into context - no
    query, no tool invocation, no model deciding anything."""
    return json.dumps(all_incidents(), indent=2)


@mcp.resource("agentmesh://incident/{incident_id}", mime_type="application/json")
def resource_incident(incident_id: str) -> str:
    """One incident by id - a resource *template*: the {incident_id} segment is
    filled in by the caller, which is how MCP addresses a family of resources
    without one registration per item."""
    for inc in all_incidents():
        if inc["id"].lower() == incident_id.lower():
            return json.dumps(inc, indent=2)
    return json.dumps({"error": "not_found", "incident_id": incident_id})


# ---------------------------------------------------------------------------
# Prompts - reusable templates the SERVER owns, surfaced by the host
# ---------------------------------------------------------------------------


@mcp.prompt(name="triage_brief", description="Draft a triage brief for one incident.")
def prompt_triage_brief(description: str, severity: str = "medium") -> str:
    """The prompt ships with the server, so every host that connects gets the same
    wording - the tool author owns the phrasing, not each client."""
    return (
        "You are an incident triage assistant.\n"
        f"Severity: {severity}\n"
        f"Incident: {description}\n\n"
        "Classify as knowledge / action / escalate, then justify in one sentence. "
        "If it is an action, state exactly what would change and why it needs approval."
    )


@mcp.prompt(name="peer_intro", description="Introduce this agent to a classmate.")
def prompt_peer_intro() -> str:
    me = student_identity()
    return (
        f"Introduce the agent '{me['agent_name']}', run by {me['student_name']} "
        f"({me['student_id']}), whose specialty is {me['specialty']}. "
        "Say in two sentences what a classmate could usefully ask it to do."
    )


def main() -> None:
    """Standalone entrypoint - `python -m app.mcp_server`."""
    settings = get_settings()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    transport = "streamable-http" if settings.mcp_server_transport == "http" else "stdio"
    logger.info("mcp.start transport=%s student=%s", transport, settings.student_id)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
