"""CohortMCP - standalone MCP server, built on the
official `mcp` Python SDK's FastMCP.

Registers two tools on a real FastMCP instance:
  - cohort_lookup(student_id) -> profile
  - incident_history(query)   -> list[incident]

Defaults to **stdio** - this is what Claude Desktop or the MCP Inspector talk
to, no open port required. Set MCP_SERVER_TRANSPORT=http to serve the exact
same tools over Streamable HTTP on MCP_SERVER_PORT instead - the tool
registrations below never change; only the `transport` argument `main()`
passes to `mcp.run()` does. See `transport_choice_decision_tree.svg`.

Tool descriptions are resolved once, from TOOL_DESCRIPTION_QUALITY, when
this module is imported (i.e. at process start) - flip the setting and
restart to see the effect, the same way any other pinned config value in
this course works.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import ToolCallResponse, ToolError as ToolErrorModel, ToolInfo
from app.tools import cohort_lookup, incident_history, tool_description

logger = logging.getLogger(__name__)

_settings = get_settings()
_quality = _settings.tool_description_quality

mcp = FastMCP(
    "cohortmcp",
    host=_settings.mcp_server_host,
    port=_settings.mcp_server_port,
)

# Register the plain functions from app/tools.py directly - `mcp.tool()` used
# as a manual call rather than `@mcp.tool()` means the imported functions stay
# exactly as importable and unit-testable as they were before this file ever
# ran; registration is a side effect on `mcp`, not a rewrite of `cohort_lookup`
# or `incident_history` themselves.
mcp.tool(name="cohort_lookup", description=tool_description("cohort_lookup", _quality))(cohort_lookup)
mcp.tool(name="incident_history", description=tool_description("incident_history", _quality))(incident_history)


async def list_tools() -> list[ToolInfo]:
    """tools/list, typed - used by the browser harness (`app/main.py`) and the
    eval's OpenAI-schema builder (`app/llm.py`)."""
    tools = await mcp.list_tools()
    return [
        ToolInfo(name=t.name, description=t.description or "", inputSchema=t.inputSchema)
        for t in tools
    ]


async def execute_tool(name: str, args: dict[str, Any]) -> ToolCallResponse:
    """tools/call through the real SDK, reshaped into this course's structured
    error envelope: {code, message, retryable}. The SDK collapses every
    failure into one exception type, `ToolError`, so we classify it here:
    no chained cause means the tool name itself was not found; a chained
    pydantic ValidationError means the arguments did not match the schema
    (FastMCP validates arguments before our function ever runs); anything
    else is a genuine runtime failure. Every call - success or failure -
    gets a trace_id so a single tools/call is traceable in logs.
    """
    trace_id = uuid.uuid4().hex[:12]
    logger.info("tool.call trace_id=%s tool=%s", trace_id, name)
    try:
        _content, structured = await mcp.call_tool(name, args)
    except ToolError as exc:
        cause = exc.__cause__
        if cause is None:
            code, retryable = "unknown_tool", False
        elif isinstance(cause, ValidationError):
            code, retryable = "bad_arguments", False
        else:
            code, retryable = "internal_error", True
        logger.warning("tool.failed trace_id=%s tool=%s code=%s", trace_id, name, code)
        return ToolCallResponse(
            success=False,
            trace_id=trace_id,
            error=ToolErrorModel(code=code, message=str(exc), retryable=retryable),
        )
    logger.info("tool.call_ok trace_id=%s tool=%s", trace_id, name)
    return ToolCallResponse(success=True, trace_id=trace_id, result=structured)


def main() -> None:
    """Console-script entrypoint - see `pyproject.toml` `[project.scripts]`."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    transport = "streamable-http" if settings.mcp_server_transport == "http" else "stdio"
    logger.info("mcp.start transport=%s", transport)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
