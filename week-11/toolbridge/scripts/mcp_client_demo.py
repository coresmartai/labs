"""Drive ToolBridge with the REAL MCP client library, over both transports.

    python scripts/mcp_client_demo.py stdio
    python scripts/mcp_client_demo.py http      # server must already be running

Not curl. Not a hand-rolled JSON-RPC frame. This is the same client an MCP host
uses, which is the only way to know your server actually works rather than
merely responds.

Watch the protocol version it prints. On mcp 2.x this server speaks
`2026-07-28`: there is no `initialize` on the wire, every request carries its
own version, and `server/discover` is available but nobody is obliged to call
it. Run the equivalent script in either of this week's guided builds and you
will see an `initialize` go past instead. Same tools, same schemas, two eras.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp import Client, StdioServerParameters  # noqa: E402


async def drive(client: Client) -> None:
    print(f"  negotiated protocol version: {client.protocol_version}")
    info = client.server_info
    if info:
        print(f"  server: {info.name} {getattr(info, 'version', '')}")

    tools = await client.list_tools()
    print(f"\n  tools/list -> {len(tools.tools)} tools")
    for t in tools.tools:
        print(f"    - {t.name}")
        print(f"      {(t.description or '')[:96]}...")
        print(f"      args: {list((t.input_schema or {}).get('properties', {}))}")

    # NOTE the attribute names. On the wire these fields are `isError` and
    # `structuredContent`; the 2.x Python models expose them snake_cased and
    # aliased. Reading the wire spelling off a Python object gets you an
    # AttributeError with a helpful "Did you mean" and five wasted minutes.
    print("\n  tools/call peer_card")
    result = await client.call_tool("peer_card", {})
    if result.is_error:
        print("    isError -> " + "".join(getattr(c, 'text', '') for c in result.content)[:200])
    else:
        payload = result.structured_content or {}
        print(f"    agent: {payload.get('agent')}  operator: {payload.get('operator')}")
        print(f"    interface: {payload.get('interface')}")
        print(f"    skills: {[s['id'] for s in payload.get('skills', [])]}")

    print("\n  tools/call delegate_triage, with a deliberately short description")
    bad = await client.call_tool(
        "delegate_triage",
        {"description": "short", "severity": "low", "user_id": "u_alice"},
    )
    print(f"    isError: {bad.is_error}")
    print("    " + "".join(getattr(c, 'text', '') for c in bad.content).splitlines()[0][:140])
    print("    ^ a TOOL EXECUTION error: the model can read it and fix the argument.")


async def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if mode == "stdio":
        print("== stdio: the host runs us as a subprocess ==")
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "app.mcp_server"], env=dict(os.environ)
        )
        async with Client(params) as client:
            await drive(client)
    else:
        port = os.environ.get("MCP_SERVER_PORT", "8770")
        url = f"http://localhost:{port}/mcp"
        print(f"== streamable http: connecting to {url} ==")
        async with Client(url) as client:
            await drive(client)


if __name__ == "__main__":
    asyncio.run(main())
