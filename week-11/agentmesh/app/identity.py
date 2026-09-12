"""Who this agent belongs to - one definition, three surfaces.

In a cohort session the same identity has to appear in three places:

  * the **Agent Card** (`/.well-known/agent-card.json`) - discovery-time identity,
    read before anyone calls you;
  * the **`whoami` MCP tool** - identity as a tool result, for an agent that found
    you through MCP;
  * the **`whoami` A2A skill** - identity as a task result, proving the
    authenticated task path works end to end.

Three copies of the same dict would drift the moment someone edits one. They all
call `student_identity()` instead, so a classmate sees the same answer however
they ask.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings


def student_identity(settings: Settings | None = None) -> dict[str, Any]:
    """The canonical identity payload for this AgentMesh instance."""
    s = settings or get_settings()
    base = s.agentmesh_base_url.rstrip("/")
    return {
        "student_id": s.student_id,
        "student_name": s.student_name,
        "specialty": s.student_specialty,
        "agent_name": agent_name(s),
        "base_url": base,
        "a2a_version": s.a2a_protocol_version,
        "endpoints": {
            "agent_card": f"{base}/.well-known/agent-card.json",
            "tasks": f"{base}/tasks",
            # Only advertised when the MCP surface actually shares this port. A
            # classmate reading this should not be told about an endpoint that is
            # really a separate process on a port they cannot reach.
            #
            # Advertised WITH the trailing slash on purpose: the mount serves its
            # route at "/", so ".../mcp" costs a 307 to ".../mcp/" on every
            # request. Well-behaved clients follow it; publishing the canonical
            # form means nobody pays for the redirect, and a client that refuses
            # to follow a redirected POST still works.
            "mcp": f"{base}{s.mcp_mount_path.rstrip('/')}/" if s.cohost_mcp else None,
        },
        "protocols": ["A2A/1.0"] + (["MCP/streamable-http"] if s.cohost_mcp else []),
    }


def agent_name(settings: Settings | None = None) -> str:
    """The Agent Card `name`. Carries the student id so a cohort of twenty cards
    is not twenty identical strings."""
    s = settings or get_settings()
    return f"agentmesh-{s.student_id}"
