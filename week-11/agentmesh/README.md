# AgentMesh - Week 11 Starter Code

**Applied GenAI & Agentic AI Engineering Course · Week 11**

AgentMesh wraps the TriageFlow specialist behind a full **A2A v1.0** FastAPI service - an Agent Card with `skills[]`, task submission, an SSE state stream with replay, human-in-the-loop approval, and JWT/JWKS verification at the door - plus a small cohort-shared MCP server exposed from the same repo. It demonstrates the two protocols a production multi-agent system needs to interoperate: **A2A** for agent-to-agent task delegation and **MCP** for tool/context serving, both fronting the same underlying agent core.

| Pattern | Endpoint / entry-point | File |
|---|---|---|
| **A2A v1.0 task lifecycle** | `POST /tasks` → `GET /tasks/{task_id}/stream` → `POST /tasks/{task_id}/input` → resubscribe | `app/main.py`, `app/task_store.py` |
| **JWT verification (JWKS-cached)** | FastAPI dependency on every business route | `app/jwt_verify.py` |
| **MCP: tools + resources + prompts** | `python -m app.mcp_server` (stdio) **or** `/mcp` on the A2A port | `app/mcp_server.py` |
| **Cohort identity (`whoami` on both protocols)** | `POST /tasks {"skill":"whoami"}` · MCP tool `whoami` | `app/identity.py` |

### Which doc do I want?

There is a lot of surface here. Read in this order and skip what you do not need yet:

| Doc | Read it when |
|---|---|
| **this README** | you want the file-by-file tour and the endpoint list |
| [`SESSION.md`](SESSION.md) | you want to *run* it - solo first, then with the class |
| [`CLOUDFLARED.md`](CLOUDFLARED.md) | before class: install and test the tunnel |
| [`A2A.md`](A2A.md) | you want to understand agent-to-agent tasks, cards, the lifecycle |
| [`MCP.md`](MCP.md) | you want to understand tools, resources and prompts |
| [`cohort.json`](cohort.json) | you need to change a URL - every one of them lives here |

---

## Project layout

```
/
├── app/
│   ├── __init__.py
│   ├── config.py            ← typed settings + .env loader (dual model pin, JWT, MCP, task-store knobs)
│   ├── cohort.py             ← reads cohort.json - one place for URLs, solo vs online
│   ├── schemas.py            ← A2A wire models (AgentCard, TaskSubmit, TaskAck, …) + TriageInput/Output
│   ├── tools.py              ← MCP tool bodies + good/bad description tables
│   ├── triage_tools.py       ← tool schema(s) + execute_tool dispatcher (used by triage_core)
│   ├── llm.py                ← thin OpenAI wrapper - dual model (nano triage, mini knowledge)
│   ├── identity.py           ← who runs this agent - one source for card + both whoami
│   ├── jwt_verify.py         ← JWKS-cached JWT verification as a FastAPI dependency
│   ├── logging_setup.py      ← console + rotating file so a session auto-saves to logs/
│   ├── task_store.py         ← task state machine - one interface, memory + SQLite impls
│   ├── triage_core.py        ← the TriageFlow specialist, imported unchanged
│   ├── mcp_server.py         ← cohort MCP server (stdio default; HTTP transport note below)
│   ├── main.py                ← FastAPI routes + CORS + /readme
│   └── data/                 ← fixtures the MCP tools read - packaged with the wheel
│       ├── cohort.json
│       └── incidents.json
├── tests/
│   ├── __init__.py
│   ├── conftest.py           ← hard-sets AUTH_MODE=jwks so the suite tests the real path
│   └── test_endpoint.py      ← 32 smoke + contract tests (no real API calls, no real network)
├── scripts/
│   ├── a2a_client.py         ← the CLIENT half - call a peer's agent over A2A
│   ├── cohort_roster.py      ← sweep the cohort: who is up, and who is who
│   └── boss.py               ← client for the cohort site (imported, not run)
├── cohort.json                ← EVERY URL, in one file (mode is COHORT_MODE in .env)
├── SESSION.md                 ← run it solo, then with the class
├── CLOUDFLARED.md             ← install the tunnel (do this before class)
├── MCP.md                     ← what the MCP surface is and how to drive it
├── A2A.md                     ← what the A2A surface is and how to drive it
├── index.html                ← browser UI (open via http://localhost:8000)
├── week11_notebook.ipynb     ← task lifecycle + MCP, plus the whole live session (§9-13)
├── pyproject.toml            ← package metadata + dependency pins
├── requirements.txt
├── .env.example              ← copy to .env and fill in
├── .gitignore
└── README.md                 ← you are here
```

> `jwt_verify.py`, `task_store.py`, `triage_core.py`, and `mcp_server.py` are the files specific to this build. The base structure - `config`, `schemas`, `tools`, `llm`, the agent, and `main` - grows an extra auth layer and a second protocol surface because AgentMesh's job is specifically interoperability, not just one more agent loop. `cohort.json` + `app/cohort.py` exist so a URL is declared once rather than retyped into every script, doc and terminal, and `COHORT_MODE` in `.env` switches the whole build between working alone and working with the class.

---

## 1. What this app does

- Serves an **Agent Card** whose `skills[]` describes `triage_incident` and `whoami`, whose `provider` names the student operating it, and whose `capabilities` carries **protocol flags only** (streaming, pushNotifications, extensions, extendedAgentCard), and whose `supportedInterfaces[]` carries the ordered reachability list v1.0 uses in place of a single `url` → `GET /.well-known/agent-card.json` (legacy alias: `/.well-known/agent.json`)
- Emits **`A2A-Version: 1.0`** on every response - announced, never negotiated
- Accepts a task, validates it, and runs it through TriageFlow in the background → `POST /tasks`
- Streams state transitions as A2A v1.0 wire strings (`TASK_STATE_SUBMITTED → TASK_STATE_WORKING → TASK_STATE_INPUT_REQUIRED? → TASK_STATE_COMPLETED / FAILED / CANCELED / REJECTED`) with `Last-Event-ID` replay → `GET /tasks/{task_id}/stream`
- Accepts a human approval **or rejection** when a task pauses at the gate; a rejection is terminal and executes nothing → `POST /tasks/{task_id}/input`
- Exposes an MCP surface built on the official `mcp` Python SDK covering **all three primitives** - Tools (`whoami`, `cohort_lookup`, `incident_history`, `ping`), Resources (`agentmesh://student/profile`, `agentmesh://incidents`, and the `agentmesh://incident/{id}` template) and Prompts (`triage_brief`, `peer_intro`) → `python -m app.mcp_server`
- **Co-hosts MCP on the A2A port** at `/mcp` (`COHOST_MCP=true`), so a single tunnel exposes both protocols - and bearer-gates that surface, because unlike stdio it is reachable from the internet
- Answers **"whose agent is this?"** three ways from one definition: the Agent Card's `provider`, the `whoami` MCP tool, and the `whoami` A2A skill → `app/identity.py`
- **Calls another AgentMesh** - `scripts/a2a_client.py` is the client half: discover a peer's card, submit a task, follow their SSE stream, and answer their approval gate remotely. The browser UI does the same thing via its **Target service** field. See [`SESSION.md`](SESSION.md)
- Serves a **browser UI** at `GET /` - no separate server needed - which drives **both** protocols: the A2A task lifecycle and, in its MCP panel, the co-hosted MCP surface over real Streamable HTTP
- **Auto-saves an activity log** to `logs/agentmesh.log` (rotating, via Python's `logging` - `app/logging_setup.py`). One line per request records **who reached out**: the caller's IP (`… via tunnel` when it arrived through cloudflared's `CF-Connecting-IP`/`X-Forwarded-For`, not the socket's `127.0.0.1`), their identity decoded from the bearer, and the status + latency. So "which classmate called my agent, and when?" is answerable after the session → `LOG_TO_FILE`
- Renders the README as HTML at `GET /readme`

### What is mocked, and named as mocked

- **The identity provider.** `AUTH_MODE=dev` (the `.env.example` default) accepts the fixed `DEV_BEARER_TOKEN` and grants the fixed `DEV_SCOPES` - so a 403 is still reachable with nobody signed in. `AUTH_MODE=jwks` (the *code* default) switches on real JWKS verification, and in a live class that is not a mock either: the cohort site runs Firebase Auth, so every student gets a genuinely signed ID token and your agent verifies it against Google's published keys.
- **The cohort MCP backing data.** Three students, three incidents, in a dict. `whoami` and `ping` are real (they read your configured identity); the lookup fixtures are not.
- **Auth differs by MCP path, deliberately.** The standalone `python -m app.mcp_server` process has **no auth** - it is stdio on your own machine. The co-hosted `/mcp` mount **is** bearer-gated (scope `mcp:invoke`), because it shares a public origin with the A2A surface. Do not tunnel the standalone one.
- **The task store.** `memory` by default. `TASK_STORE_BACKEND=sqlite` is a real, durable second implementation behind the same four methods - the seam, not a promise.
- **The Agent Card `signatures[]` field.** Declared and unpopulated. This build does not sign its card and does not verify anyone else's. Note the plural: v1.0 carries a *list*, because key rotation means a card may legitimately carry two valid signatures at once.
- **Nothing calls a model at runtime.** `triage_core.py` is deterministic. `app/llm.py` is the SDK seam the pins are wired through - `/health` reports them via `model_for()` - and it is never dialled.

---

## 2. Setup (5 min)

> Requirements: Python 3.10+ and an OpenAI API key. `AUTH_MODE=dev` needs nothing else; `AUTH_MODE=jwks` needs a token from the issuer `JWT_ISSUER` names - in a cohort session, one you get by signing in to the cohort site.

```bash
# 1. Create and activate a venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env file and fill in real values
cp .env.example .env
# Open .env and set: OPENAI_API_KEY, JWT_ISSUER, JWT_AUDIENCE, JWT_JWKS_URL

# 4. Start the A2A service
#    --reload-dir app scopes the watcher to source; without it the rotating
#    log file the server writes would trip --reload on every request.
uvicorn app.main:app --reload --reload-dir app

# 5. In a second terminal - run the cohort MCP server (stdio by default)
python -m app.mcp_server
```

**You're live at `http://localhost:8000`.**

- Browser UI: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- README: `http://localhost:8000/readme`

> **JWT setup quirk:** `/tasks`, `/tasks/{id}/stream`, and `/tasks/{id}/input` all require a bearer token whose `aud` matches `JWT_AUDIENCE`, `iss` matches `JWT_ISSUER`, and whose `scope` contains `triage:invoke`, signed by a key published at `JWT_JWKS_URL`. There is no token-minting CLI here because there does not need to be one: in a cohort session you sign in to the cohort site and `scripts/boss.py` hands you a freshly refreshed Firebase ID token. Working alone, `AUTH_MODE=dev` skips the whole question, and `tests/test_endpoint.py` shows how to construct a token against a stubbed HS256 JWKS if you want to exercise the real path locally.

---

## 3. File-by-file walkthrough

Reading order: `config → schemas → llm → tools → jwt_verify → task_store → triage_core → main → mcp_server`.

### `app/config.py` - Settings

Typed `Settings` via `pydantic-settings`, `@lru_cache(maxsize=1)`. Beyond the standard `openai_api_key` / `log_level`, this week pins **two** models (`triage_model` = nano for cheap classification, `knowledge_model` = mini for synthesis) and carries the auth-mode (`auth_mode` = `jwks`|`dev`, `dev_bearer_token`, `dev_scopes`), JWT (`jwt_issuer`, `jwt_audience`, `jwt_jwks_url`, `jwt_jwks_cache_seconds`, `jwt_leeway_seconds`), protocol (`a2a_protocol_version`), MCP (`mcp_server_transport`, `mcp_server_port`), and task-store (`task_store_backend` = `memory`|`sqlite`, `task_store_sqlite_path`, `task_replay_buffer_size`) knobs. `openai_api_key`, `jwt_issuer` and `jwt_jwks_url` are **required** - a missing one is a startup error naming the field, not a 500 halfway through a task.

### `app/schemas.py` - A2A v1.0 wire models + skill contracts

`AgentCard` / `AgentSkill` / `AgentCapabilities` / `AgentEndpoints` are the Agent Card shape. **`skills[]` is what the agent does; `capabilities` is protocol flags only** - streaming, push notifications, extended card. Latency hints and cost hints have no home in a v1.0 card, and they are not here.

`TaskState` is the spec's **nine** `TASK_STATE_*` values, and **the value IS the string on the wire** - there is no lowercase form anywhere in this repo. This flow drives **seven** of them: `TASK_STATE_UNSPECIFIED` is a sentinel (`update_state` raises if you try to assign it) and `TASK_STATE_AUTH_REQUIRED` has no producer, because authentication happens at the door and there is no mid-task auth challenge to raise. `TERMINAL_STATES`, `INTERRUPTED_STATES` and `DRIVEN_STATES` are exported frozensets, so "is this task finished" is never a hand-written tuple that forgets `TASK_STATE_REJECTED`.

`TaskSubmit` takes **`skill`** (with `capability` accepted as a legacy alias). `TaskFailedPayload{reason, evidence, retryable}` is what a failure looks like: a state with structure, not a bare 500. `TriageInput` is the per-skill input contract, validated **inside** the handler (see the 422 note in Section 6) because it depends on `body.skill`; `TriageOutput` validates the terminal payload before it goes on the wire.

### `app/llm.py` - the SDK seam

The only file that imports the OpenAI SDK. `model_for(role)` says which pinned model a role dials (`triage` → nano, `knowledge` → mini) and **`/health` builds its `models` block from it**, so the chip on screen cannot drift from the seam. `complete()` uses `max_completion_tokens`, never `max_tokens`, and sends **no `temperature`** - nothing is being traded away, because the seam is wired and deliberately never dialled: the triage path is deterministic.

The three triage steps `triage_core.py` calls live here too - **`classify_incident`**, **`extract_action`**, and **`synthesize_answer`**. Each runs a deterministic rule (keyword / substring / runbook formatting) with the exact `await self.complete(...)` call it would make **commented out one block above it**. They are already `async`, so uncommenting one line makes that step call the model without touching any other file - which is the whole point of keeping the SDK behind a single seam.

### `app/tools.py` - MCP tool bodies

`cohort_lookup` and `incident_history` over the fixtures in `app/data/*.json`; `whoami` and `ping` over this instance's configured identity. `cohort_lookup` answers about you as well as about the fixtures, so a classmate who looks up your id is not told you do not exist. Also holds the good/bad `tool_description` tables that `TOOL_DESCRIPTION_QUALITY` selects between. No MCP import here - `mcp_server.py` does the registering, so every function stays callable in a test.

### `app/triage_tools.py` - Tool registry for the TriageFlow core

The `{schema, impl}` pairs + `execute_tool(name, args)` dispatcher that `triage_core.py` calls when the classifier proposes an action. Never raises - unknown tool and impl exceptions both return `{"success": False, ...}`. Internal to the agent, and unrelated to the MCP surface above.

### `app/jwt_verify.py` - JWT verification dependency

`require_scope(scope)` returns a FastAPI dependency: fetch (and cache) the JWKS, verify the signature against the key matching the token's `kid`, check `aud`/`iss`, then check the required scope is present. Every failure is a structured 401/403 - `{"reason": "missing_bearer" | "auth_invalid" | "scope_missing", ...}` - never a stack trace.

### `app/task_store.py` - Task state machine

One interface - **`create`, `get_state`, `latest`, `update_state`, `stream`, plus the input handshake** - and two implementations: `MemoryTaskStore` (the default; a bounded `deque(maxlen=TASK_REPLAY_BUFFER_SIZE)`) and `SqliteTaskStore` (`TASK_STORE_BACKEND=sqlite`; durable, unbounded replay). `main.py` cannot tell them apart.

**`event_id` is a per-task monotonic counter.** It is never derived from the length of the replay buffer: `len(events) + 1` stalls the moment the bounded deque wraps at 100, and after that every `Last-Event-ID` replay on a long task silently returns the wrong events. `test_sse_event_ids_are_monotonic_past_the_buffer_wrap` is the regression that would have caught it.

If a client reconnects with a cursor **older than the oldest buffered event**, `stream` emits a synthetic `{"replay_gap": true, "oldest_available_event_id": N}` event before the replay - the orchestrator is told there is a hole rather than being handed a plausible-looking tail. `update_state` refuses to assign the `TASK_STATE_UNSPECIFIED` sentinel and refuses to move a task that is already terminal.

### `app/triage_core.py` - The TriageFlow specialist

Imported as-is, and it has **zero awareness of HTTP, JWT, SSE, A2A or MCP**. It talks to the outside through two callbacks it is handed - `on_progress` and `wait_for_approval` - which is the entire reason wrapping it in A2A changed no agent code. When the gate answers `approved: false` it raises `RejectedByCaller` and unwinds without executing anything; `main.py` is what turns that into `TASK_STATE_REJECTED`. Deterministic: no model call on this path.

### `app/main.py` - FastAPI routes

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Serve browser UI (`index.html`) |
| `/health` | GET | Liveness probe - `{"status":"ok","model":"...","models":{"triage":"...","knowledge":"..."}}` |
| `/readme` | GET | Render README.md as dark-themed HTML |
| `/cohort` | GET | Unauthenticated roster bootstrap - `{mode, cohort_index_url, peers, my_base_url}` from `cohort.json`; the UI's Cohort tab and `cohort_roster.py` sweep from it |
| `/.well-known/agent-card.json` | GET | Unauthenticated Agent Card discovery (canonical A2A ≥0.3 path) |
| `/.well-known/agent.json` | GET | Legacy discovery alias for pre-0.3 clients |
| `/tasks` | POST | Validate + gate on JWT + create task + kick off background work → **202** |
| `/tasks/{task_id}` | GET | A2A `tasks/get` - point-in-time snapshot: state, payload, `last_event_id`, `final`/`interrupted` |
| `/tasks/{task_id}/stream` | GET | SSE for ONE interaction: ends with `final` at a terminal **or interrupted** state. `Last-Event-ID` header **or** `last_event_id` query param to resume |
| `/tasks/{task_id}/input` | POST | Human-in-the-loop approval **or rejection** |

An `A2A-Version` middleware stamps `A2A-Version: 1.0` on **every** response, including errors. It is emitted, never negotiated - a client that sends a different version header is not rejected, because this build speaks exactly one version.

Per-skill body validation (`TriageInput.model_validate(body.input)`) happens inside the handler and is wrapped in a `try/except ValidationError` that re-raises as `HTTPException(422, detail=e.errors())` - FastAPI only auto-converts validation errors that happen in its own request-parsing step, not ones raised manually one level down.

`_run_task` is the whole A2A wrapper: it translates what the agent does into wire states. Approve → `TASK_STATE_WORKING (executing)` → `TASK_STATE_COMPLETED`. **Reject → `TASK_STATE_REJECTED`, terminal, nothing executed.** No reply in 300 s → `TASK_STATE_FAILED` carrying `TaskFailedPayload{reason: "input_timeout", retryable: false}`.

### `app/mcp_server.py` - Cohort MCP server

Four tools - `whoami`, `ping`, `cohort_lookup(student_id)` and `incident_history(query)` - plus resources and prompts, registered onto a real `FastMCP` instance from the official `mcp` Python SDK: `mcp.tool(name=..., description=...)(cohort_lookup)`, called as a plain function rather than a `@mcp.tool()` decorator so the imported functions stay directly callable and testable (see `tests/test_endpoint.py`, which imports and calls them with no server involved at all). Defaults to stdio (`python -m app.mcp_server`); setting `MCP_SERVER_TRANSPORT=http` makes `main()` call `mcp.run(transport="streamable-http")` instead - a real Streamable HTTP MCP endpoint on `MCP_SERVER_PORT`, not a stub.

### `index.html` - Browser UI

Open at `http://localhost:8000` after starting the server.

A three-pane layout - a header, a left control panel, and a right output-and-logs panel - with a consistent set of design tokens.

**Left panel:** **Target service** (base URL + bearer token + **Connect** and **🔎 Capabilities** - the latter lists the target's *whole* feature surface in one view: A2A skills + capability flags + endpoints from the Agent Card, and MCP tools/resources/prompts + server flags from the handshake, for your own service or a classmate's. Every listed item is a **clickable chip that invokes it** - a skill runs an A2A task, a tool calls `tools/call`, a resource (or the `{id}` template) reads it, a prompt fetches it, prompting for any required args from the item's own schema), then a three-way tab - **A2A** (scenario preset, description `#notes`, severity + user id, Submit / Fetch Card / Drop stream), **MCP** (tool picker + JSON args, Connect & List / Call Tool), and **Cohort** (Sweep - lists the roster with a live `/health` probe and a **Use** button that targets any agent). Only the active tab's fields and buttons are on screen.
**Right panel:** the CORE **Output** pane (renders the live task - state pills, streamed events, the approval gate's Approve/Reject, the result - or the MCP surface listing, a tool-call result, or the Agent Card, with `{ } Format`) and the **Logs** pane - a timestamped activity trail: every network hop with its **latency** and, when you drive a classmate, the **peer host** it crossed to (`POST /tasks @ alice.trycloudflare.com → 202 · 41ms`); each SSE state transition; tool calls with their args; roster probes (`up` / `unreachable` + ms); and feature usage like tab switches and Connect. The pane header shows a live entry count and a **⤓ Save .log** button that downloads the whole trail as a timestamped `.log` file (client-side Blob, headed with the target/auth context) - useful for handing an instructor a record of a session.
**Header:** **🪪 My Card** (this service's own Agent Card, whatever the current target is), a README link, and a `/health` chip that shows the target's model **and** `auth_mode` - so a `demo-token`-vs-`jwks` mismatch is visible before you click. Errors carry an auth-aware hint that names the actual fix.

**Drop stream** closes the `EventSource` mid-task and re-opens it with the last event id it saw - the replay demo, in one click.

> The demo's `TOKEN` constant matches `DEV_BEARER_TOKEN`, which the service accepts when `AUTH_MODE=dev` (the `.env.example` default) - so the full submit → stream → approve flow works out of the box. Switch to `AUTH_MODE=jwks` and the same UI correctly 401s until you swap in a real signed JWT: the UI drives the real protected endpoints, so it fails exactly the way a real unauthenticated client would. The SSE stream passes the bearer as a `?token=` query parameter because `EventSource` cannot send headers.

### `week11_notebook.ipynb` - API notebook

Curl + Python-requests walkthrough of the task lifecycle and the MCP surface, plus the three failure modes below (422 / 401 / 409). (The two GET snapshots added for the UI - `tasks/get` and `/cohort` - have no dedicated cell; the notebook drives the lifecycle through `scripts/a2a_client.py`.)

---

## 4. Try it out

### a) Health check

```bash
curl http://localhost:8000/health
```

### b) Fetch the Agent Card

```bash
curl -s http://localhost:8000/.well-known/agent-card.json
```

### c) Knowledge request (no approval needed)

```bash
curl -s http://localhost:8000/tasks -H "Content-Type: application/json" ^
  -H "Authorization: Bearer %TOKEN%" ^
  -d "{\"skill\": \"triage_incident\", \"input\": {\"description\": \"Where is the runbook for restarting payments?\", \"severity\": \"low\", \"user_id\": \"u1\"}}"
```

With `AUTH_MODE=dev`, `%TOKEN%` is just `demo-token`. Under `AUTH_MODE=jwks` it must carry `aud=<JWT_AUDIENCE>`, `iss=<JWT_ISSUER>`, and `scope` containing `triage:invoke` - in a cohort session, the Firebase ID token you get from signing in to the cohort site. Watch the task move `TASK_STATE_SUBMITTED → TASK_STATE_WORKING → TASK_STATE_COMPLETED` on the stream.

### d) Stream, and replay what you missed

```bash
curl -N -H "Authorization: Bearer demo-token" http://localhost:8000/tasks/<id>/stream

# reconnect from a cursor - every event with a greater id comes back, in order:
curl -N -H "Last-Event-ID: 3" -H "Authorization: Bearer demo-token" \
     http://localhost:8000/tasks/<id>/stream
```

### e) The cohort MCP server

```bash
python -m app.mcp_server
```

That is the standalone stdio process a local host (Claude Desktop, the Inspector) talks to - no port, no auth. Setting `MCP_SERVER_TRANSPORT=http` gives it its own Streamable HTTP port instead.

With `COHOST_MCP=true` (the default) the **same** FastMCP instance is also mounted at `/mcp` on the A2A port, so one tunnel carries both protocols. That surface is bearer-gated with scope `mcp:invoke`:

```bash
curl -s -X POST http://localhost:8000/mcp/ -H "Authorization: Bearer demo-token" -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"c\",\"version\":\"1\"}}}"
```

Drop the `Authorization` header and you get a structured `401 missing_bearer` - `app.mount()` bypasses FastAPI dependencies, so `main.py` guards the mount in middleware using the same `verify_bearer` the A2A routes use.

### f) Who is this agent? (`whoami`, on both protocols)

```bash
python scripts/a2a_client.py http://localhost:8000 --skill whoami
```

Same answer over MCP via the `whoami` tool, and in the Agent Card's `provider` block - all three read `app/identity.py`, so they cannot disagree.

### g) Delegate a task to a classmate's agent (A2A, across machines)

```bash
python scripts/a2a_client.py https://alice.trycloudflare.com --card-only
```

```bash
python scripts/a2a_client.py https://alice.trycloudflare.com --description "Please restart payments-api now" --severity high --approve
```

Discovery → submit → stream → **approve their gate remotely** → completed. Drop `--approve` to be prompted; `--reject` lands the task in `TASK_STATE_REJECTED` with nothing executed on their side. The same flow runs from the browser: paste their base URL into the UI's **Target service** field and click Connect.

The peer must be reachable and must have set `AGENTMESH_BASE_URL` to their public URL - that one setting is stamped into their Agent Card *and* the `stream_url` they hand back, so leaving it at the default means you submit a task that runs fine and streams nowhere. Both clients here detect that and fall back with a warning. Full setup - tunnels, auth, troubleshooting - in **[`SESSION.md`](SESSION.md)**.

### h) Sweep the whole cohort

```bash
python scripts/cohort_roster.py
```

Reads `COHORT_MODE`: in **online** mode it pulls the class's published `index.json`; in **solo** mode it probes the local peers `cohort.json` lists. For each one it fetches the Agent Card and runs an authenticated `whoami`, in parallel, and prints who answered. Run it before everyone starts calling each other - it turns "it doesn't work" into a named list of whose tunnel is down and whose token is wrong.

---

## 5. Common failure modes

### a) Wrong-shape task body (422)

Submit `severity: "urgent"` (not a valid enum). `TriageInput.model_validate` raises `ValidationError`, caught in `submit_task` and re-raised as `HTTPException(422, detail=e.errors())` - the per-field detail names exactly which field and why, unlike a generic 400 from a homemade validator.

### b) Expired JWT mid-task (401)

Issue a short-lived token, submit a task that reaches the approval gate, then try to post the approval after it expires. `jwt_verify.py` raises a structured 401 with `reason: "auth_invalid"` - the orchestrator sees a clean auth failure, not a stack trace, and knows to refresh and retry.

### c) Streaming connection drops mid-task

Disconnect the client mid-stream (the UI's **Drop stream** button does exactly this), then reconnect to `/tasks/{task_id}/stream` with `Last-Event-ID` set to the last event seen. `task_store.py` replays every buffered event past that cursor before resuming live. If your cursor is older than the buffer's oldest event (default 100, `TASK_REPLAY_BUFFER_SIZE`), you get an explicit `{"replay_gap": true, "oldest_available_event_id": N}` event first - a hole you are told about beats a hole you are not.

### d) Reply to a task that is not paused (409)

`POST /tasks/{task_id}/input` on a completed task → `409 {"reason": "task_not_paused", "current_state": "TASK_STATE_COMPLETED"}`. The notebook's Section 6 triggers all three: 422, 401, 409.

---

## 6. Run the tests

```bash
pytest -q
```

**32 tests** - no real network calls, no real model calls, no real JWKS:

| # | Test | What it checks |
|---|---|---|
| 1 | `test_agent_card_is_well_formed` | Name, version, `protocolVersion`, `skills[0].id == "triage_incident"` with tags/examples/inputModes/outputModes, endpoints, `securitySchemes` |
| 2 | `test_agent_card_capabilities_are_protocol_flags_only` | `capabilities` is an object of protocol flags; no `latency_p50_ms` / `cost_hint` / schema refs anywhere in the card |
| 3 | `test_legacy_agent_json_alias_returns_the_same_card` | `/.well-known/agent.json` body == `/.well-known/agent-card.json` body |
| 4 | `test_agent_card_signatures_are_declared_and_unpopulated` | `signatures` is present and empty, and singular `signature` is absent - the omission is visible, not hidden |
| 5 | `test_a2a_version_header_on_every_response` | `A2A-Version: 1.0` on the card, `/health`, a 202 and a 401 |
| 6 | `test_task_state_enum_is_the_specs_nine` | Nine `TASK_STATE_*` members; four terminal, two interrupted, **seven driven** |
| 7 | `test_health_is_unauth_and_returns_models` | `/health` returns `model` + `models.triage` + `models.knowledge` |
| 8 | `test_tasks_returns_401_without_authorization` | Missing bearer → 401 `missing_bearer` |
| 9 | `test_tasks_returns_403_with_wrong_scope` | Wrong scope → 403 `scope_missing` |
| 10 | `test_tasks_returns_422_on_malformed_input` | Missing required fields → 422 with per-field detail |
| 11 | `test_happy_path_knowledge_request_reaches_completed` | Submit → `TASK_STATE_COMPLETED` |
| 12 | `test_reject_at_the_gate_drives_task_state_rejected` | Reject → `TASK_STATE_REJECTED`, terminal, and **no `executing` event was ever emitted** |
| 13 | `test_sse_event_ids_are_monotonic_past_the_buffer_wrap` | >100 events: ids strictly increasing across the wrap; a live cursor replays exactly the greater ids; a stale cursor gets the `replay_gap` marker |
| 14 | `test_llm_client_routes_role_to_the_pinned_model` | `model_for("triage")` / `model_for("knowledge")` hit the right pins - no network |
| 15 | `test_structured_failures_409_and_task_failed_payload` | 409 `task_not_paused` with `current_state`; `TaskFailedPayload` retryable semantics |
| 16 | `test_mcp_cohort_lookup_known_and_unknown` | Known student returns profile; unknown returns `{"found": false}` (not an error) |
| 17 | `test_mcp_incident_history_returns_structured_payload` | Structured `{results, total_available}`; empty match returns `[]`, not an error |
| 18 | `test_a2a_client_card_validation_catches_wrong_skill_and_localhost_trap` | A peer card missing the requested skill is refused; a card advertising `localhost` while you dialled a public URL is flagged |
| 19 | `test_a2a_client_prefers_the_dialled_base_over_a_bad_stream_url` | A misconfigured peer's `stream_url` is overridden by the base you dialled, with a warning |
| 20 | `test_agent_card_carries_student_identity` | `provider.organization` is the student's name; the id is in the card name and the name in the description |
| 21 | `test_whoami_skill_completes_and_reports_the_caller` | `skill: whoami` drives SUBMITTED → WORKING → COMPLETED and echoes back who called |
| 22 | `test_unknown_skill_is_refused_before_any_task_is_created` | An unknown skill is a 400 `unknown_skill`, not a task that fails later |
| 23 | `test_mcp_whoami_and_ping_report_this_student` | The identity and connectivity tools read the configured student, called directly |
| 24 | `test_mcp_surface_registers_tools_resources_and_prompts` | All three MCP primitives are registered - tools, resources (+ a template), prompts |
| 25 | `test_cohosted_mcp_mount_is_bearer_gated` | The mounted `/mcp` surface 401s without a bearer and rejects a wrong one |
| 26 | `test_cohort_index_reader_accepts_every_shape_including_firebase` | The roster reader handles all three index shapes, including Firebase's `{tag: {...}}` object |
| 27 | `test_cohort_config_exposes_the_boss_block` | `cohort.json`'s `boss` block loads with the api key and database URL the notebook needs |
| 28 | `test_stream_ends_at_the_gate_with_final_and_resumes_on_resubscribe` | The stream closes with `final` at `INPUT_REQUIRED`; resubscribing with the cursor resumes rather than replays |
| 29 | `test_tasks_get_snapshot_tracks_the_task_without_any_stream` | `tasks/get` reports state, payload and `interrupted` vs `final` with no stream open |
| 30 | `test_tasks_get_requires_the_same_scope_as_the_rest` | The snapshot is not a cheaper way past auth: 401 unauthenticated, 403 on the wrong scope |
| 31 | `test_cohort_endpoint_bootstraps_the_browser_roster` | `GET /cohort` exposes mode + peers for the UI's Cohort tab, unauthenticated, without leaking the boss block |
| 32 | `test_logging_setup_writes_a_rotating_file_and_is_idempotent` | `setup_logging()` attaches one rotating file handler and a reload does not stack a second |

`pytest -q` → **32 passed**, sub-second, no key, no network.

`conftest.py` **hard-sets `AUTH_MODE=jwks`** before importing the app. `.env.example` ships `AUTH_MODE=dev` so the browser demo runs out of the box - which means `cp .env.example .env` would otherwise leak `dev` into the suite through pydantic-settings, and dev mode would compare the tests' signed HS256 tokens against the literal string `demo-token`. The JWKS path is what these tests exist to cover.

The JWKS itself is stubbed with an in-process HS256 symmetric key (`_TEST_JWKS` in `test_endpoint.py`) - the `k` field is base64url-encoded raw key bytes, which is what `PyJWT`'s `PyJWK` expects when building a verification key from a JWK dict.

The `client` fixture is **context-managed** (`with TestClient(app) as c`). That keeps one event loop alive for the whole module, so the background task fired by `POST /tasks` is still running when the approval arrives on the next request. Without it, every cross-request flow dies at the gate.

---

## 7. Where this goes next

- **GuardianAI™** layers PII scrubbing on every input/output crossing the A2A boundary, plus a responsible-AI checklist.
- **DeployCore** puts a real deploy, a durable job store and OpenTelemetry traces behind every hop, and moves identity onto an OIDC provider you operate rather than one handed to the class.
- **CostGuard™** adds per-call cost telemetry and SLM routing to the triage classifier, building on the dual-model split already in `llm.py`.
- **AgentForge™ (capstone)** bundles the protocol surface here into a portfolio-ready deployable demo.

Don't throw this away - every week builds on it.

---

## Decisions

- **One `TaskStore` interface, two real implementations: memory (default) and SQLite.** The interface is the lesson. Shipping a second *working* backend behind the same four methods is what proves the seam is real - a comment promising some future store proves nothing. Swap `TASK_STORE_BACKEND` and `main.py` cannot tell the difference.
- **`skills[]`, not `capabilities`, for what the agent does.** An A2A v1.0 card puts its work in `skills[]` and reserves `capabilities` for protocol flags. The older shape - a `capabilities` list of `{name, latency_p50_ms, cost_hint}` - is a common anti-pattern; it is gone from the code, the card, the UI, the notebook and the tests.
- **The Agent Card is served but not signed.** `signatures` is declared and left empty. Card signing is a real A2A feature; a fake signature field would teach the wrong lesson about what verification means.
- **`A2A-Version` is emitted, never negotiated.** One version ships. A client sending a different one is not rejected - there is nothing to fall back to, and a 4xx on a header we do not act on would be theatre.
- **Dev auth grants a fixed scope set, not the scope the route asked for.** Echoing back `{"scope": <required_scope>}` would make the 403 branch unreachable in dev mode - a scope check that can never fail is not a scope check.
- **SSE with `Last-Event-ID` replay, not WebSockets.** Task state flows one direction - server to client - and SSE rides plain HTTP with a replay-on-reconnect story built into the spec (`Last-Event-ID`). WebSockets would need a hand-rolled replay protocol for the same guarantee.
- **The official `mcp` SDK (`FastMCP`), registered by calling `mcp.tool(...)` as a function, not a decorator.** Registration is a side effect on the `FastMCP` instance; keeping `cohort_lookup` and `incident_history` themselves undecorated means `tests/test_endpoint.py` calls them directly with zero server involved, and `pytest -q` stays fast and network-free without needing a stub.
- **Both transports are real, verified against the real `mcp` client library, not curl alone.** `mcp.run(transport="stdio")` and `mcp.run(transport="streamable-http")` are both genuine MCP endpoints; stubbing the HTTP path with a printed placeholder would be tempting, but the SDK makes the real thing no harder to wire up than a stub.
- **The A2A client lives in `scripts/`, not in `app/`.** The service and the thing that calls a service are different programs with different lifetimes - folding an outbound client into the FastAPI app would blur that, and nothing in `app/` should need to know a peer exists. Keeping it a standalone script also means a student can point it at *any* A2A agent, not just another copy of this one.
- **Both clients prefer the base URL you dialled over the one the peer advertises.** A peer who never set `AGENTMESH_BASE_URL` hands back a `stream_url` on `localhost`; following it faithfully means the task runs correctly and you watch an empty stream forever. Trusting your own dialled base and warning loudly turns a silent hang into a one-line diagnosis - and the two pure helpers that do it (`validate_card`, `resolve_stream_url`) are unit-tested precisely because that failure only shows up once students are in different buildings.
- **One `cohort.json` for every URL, one `COHORT_MODE` to switch.** A base URL could otherwise arrive from four places - `.env`, a script argument, a doc's copy-paste block, a hard-coded default - each a chance to point at the wrong host. Declaring them once, and flipping `COHORT_MODE=solo`/`online` in `.env` rather than editing several files, is what makes the live session survivable. Mode is in `.env` and not in `cohort.json` because which mode *you* run is a per-machine choice, while the file is shared and committed.
