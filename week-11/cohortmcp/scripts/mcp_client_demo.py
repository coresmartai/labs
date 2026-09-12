"""Drive CohortMCP with the *real* MCP client library - both transports.

Not curl, not a hand-rolled JSON-RPC frame: `mcp.client.stdio` and
`mcp.client.streamable_http` are the same client an MCP host uses.

    python scripts/mcp_client_demo.py stdio     # spawns `python -m app.mcp_server`
    python scripts/mcp_client_demo.py http      # expects a server already running:
                                                #   MCP_SERVER_TRANSPORT=http python -m app.mcp_server

Both paths do the same three things - initialize, list_tools, call_tool - and
both come back with the same two tools. That is the whole point of the switch.
"""
from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

_HTTP_URL = "http://localhost:8766/mcp"


async def _drive(session: ClientSession) -> None:
    await session.initialize()

    tools = await session.list_tools()
    print(f"\ntools/list -> {len(tools.tools)} tools")
    for t in tools.tools:
        first_line = (t.description or "").strip().splitlines()[0]
        print(f"  - {t.name}: {first_line[:80]}")
        print(f"    inputSchema.required = {t.inputSchema.get('required')}")

    result = await session.call_tool("cohort_lookup", {"student_id": "stu_002"})
    print(f"\ntools/call cohort_lookup(stu_002) -> {result.structuredContent}")

    result = await session.call_tool("incident_history", {"query": "payments"})
    print(f"tools/call incident_history('payments') -> {result.structuredContent}")


async def run_stdio() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            print("transport: stdio (the server is a child process on this machine)")
            await _drive(session)


async def run_http() -> None:
    async with streamable_http_client(_HTTP_URL) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            print(f"transport: streamable-http ({_HTTP_URL})")
            await _drive(session)


def main() -> None:
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport not in ("stdio", "http"):
        raise SystemExit("usage: python scripts/mcp_client_demo.py [stdio|http]")
    asyncio.run(run_stdio() if transport == "stdio" else run_http())


if __name__ == "__main__":
    main()
