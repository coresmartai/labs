# MCP in this build

**MCP is how an agent gets context and capability** - tools to call, data to read,
prompts to reuse. It is *vertical*: your agent reaching down to its resources.
(A2A is the horizontal one - see [`A2A.md`](A2A.md).)

---

## What this server exposes

MCP has three server-side primitives. Most tutorials only ever show the first one.
This build ships all three so the difference is visible.

### Tools - functions a model can invoke

| Tool | Argument | What it does |
|---|---|---|
| `whoami` | none | Who runs this agent: student id, name, specialty, protocols |
| `ping` | `message` | Echo + identity. Proves transport, auth and dispatch in one call |
| `cohort_lookup` | `student_id` | A classmate's profile. `{"found": false}` on a miss - not an error |
| `incident_history` | `query` | Substring search over incidents, capped at 25 |

The description a model reads comes from the table in [`app/tools.py`](app/tools.py),
which ships a deliberately good and a deliberately bad wording for every tool.
`TOOL_DESCRIPTION_QUALITY` picks one at startup - set it to `bad` and watch a
connected model start guessing.

### Resources - addressable data, no model in the loop

| URI | Contents |
|---|---|
| `agentmesh://student/profile` | This student's identity |
| `agentmesh://incidents` | The whole incident list |
| `agentmesh://incident/{incident_id}` | One incident - a **template** |

A resource is *fetched*, not *invoked*. Nothing decides anything; the host reads it
straight into context. The template form addresses a whole family of resources with
one registration instead of one per item.

### Prompts - templates the server owns

| Prompt | Arguments |
|---|---|
| `triage_brief` | `description`, `severity` |
| `peer_intro` | none |

The point is authorship: the *server* decides the wording, so every host that
connects gets the same phrasing. A host typically surfaces these as slash-commands.

### Not implemented (and why)

`sampling`, `roots` and `elicitation` are **client-side** capabilities - the host
offers them to the server, not the reverse. Nothing here needs the host's model, the
host's filesystem, or a mid-tool human prompt. A2A's approval gate is the same idea
as elicitation, one layer up.

---

## Two ways to run it, with different auth

This matters more than it looks:

| | Standalone | Co-hosted |
|---|---|---|
| Command | `python -m app.mcp_server` | part of `uvicorn app.main:app` |
| Transport | stdio (or its own HTTP port) | Streamable HTTP at `/mcp/` |
| Auth | **none** | **bearer**, scope `mcp:invoke` |
| Reachable from | your machine only | anywhere your tunnel reaches |
| Use it for | Claude Desktop, the Inspector | classmates over the network |

**Never tunnel the standalone one.** It has no auth because stdio does not need any.
The co-hosted mount is the guarded one, and it is the only MCP path that should
ever see a public URL.

Why the mount needs its own guard: `app.mount()` **bypasses FastAPI dependencies
entirely**, so the `Depends(require_scope(...))` protecting every A2A route does not
protect it. [`app/main.py`](app/main.py) guards the mount path in middleware, calling
the same `verify_bearer` the routes use - one implementation, so the two cannot drift.

---

## Try it

Start the service (`uvicorn app.main:app --reload --reload-dir app`), then run this from the repo root -
save it as `try_mcp.py`, or paste it into the notebook:

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from app.cohort import load_cohort

URL = load_cohort().url("mcp")          # no URL typed - it comes from cohort.json
AUTH = {"Authorization": "Bearer demo-token"}   # solo mode; online, use boss.token()

async def main():
    async with streamablehttp_client(URL, headers=AUTH) as (read, write, _):
        async with ClientSession(read, write) as s:
            await s.initialize()
            print("tools    :", [t.name for t in (await s.list_tools()).tools])
            print("resources:", [str(r.uri) for r in (await s.list_resources()).resources])
            print("prompts  :", [p.name for p in (await s.list_prompts()).prompts])
            print("whoami   :", (await s.call_tool("whoami", {})).structuredContent)

asyncio.run(main())
```

The shape of a session is always the same:

```python
async with streamablehttp_client(URL, headers={"Authorization": f"Bearer {TOKEN}"}) as (r, w, _):
    async with ClientSession(r, w) as s:
        await s.initialize()                       # capability negotiation
        await s.list_tools()                       # what can I call
        await s.call_tool("whoami", {})            # call one
        await s.list_resources()                   # what can I read
        await s.read_resource("agentmesh://incidents")
        await s.list_prompts()                     # what templates exist
        await s.get_prompt("peer_intro", {})
```

Drop the `Authorization` header against the co-hosted URL and you get a structured
`401 missing_bearer` - that is the guard working.

**No terminal needed.** The notebook drives this same surface in **Section 12** against
your own server (solo) and **Section 13** against a classmate's (online). The browser UI
does it two ways: the **MCP tab** (Connect & List, then Call Tool) and the **🔎 Capabilities**
button, which lists the whole surface - tools, resources, the `{id}` template, and prompts -
as **clickable chips that invoke each one**, prompting for any args from the item's own
schema. All of them run the same `initialize -> list -> call` in JavaScript, and all point
at your own `/mcp/` by default or a peer's the moment you Connect to their URL.

---

## Two things students reliably get wrong

**"Capabilities" is not "what it can do."** At `initialize` both sides declare which
*primitives* they support, plus sub-flags like `listChanged` (will I tell you when my
tool list changes?) and `subscribe` (can you watch a resource?). This build answers no
to both - its registrations are fixed at import. For "what can it do", read
`tools/list`.

**Where the tool bodies live.** [`app/tools.py`](app/tools.py) holds the plain
functions and their description tables. [`app/mcp_server.py`](app/mcp_server.py)
registers them onto FastMCP and adds the resources and prompts. The triage agent's
own internal registry is a separate file, [`app/triage_tools.py`](app/triage_tools.py) -
that one is what the agent calls, this one is what the agent offers.
