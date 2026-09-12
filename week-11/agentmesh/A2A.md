# A2A in this build

**A2A is how one agent hands work to another agent.** It is *horizontal*: your agent
talking to a peer's agent, across a network, across an ownership boundary. (MCP is
the vertical one - see [`MCP.md`](MCP.md).)

The unit of work is a **task**, not a request/response. Tasks take time, stream
progress, and can stop to ask a human something.

---

## The five surfaces

| Method | Path | Auth | What it is |
|---|---|---|---|
| GET | `/.well-known/agent-card.json` | none | Discovery. Who are you, what can you do |
| GET | `/.well-known/agent.json` | none | Legacy alias for pre-0.3 clients |
| POST | `/tasks` | `triage:invoke` | Submit work → **202** with a task id |
| GET | `/tasks/{id}/stream` | `triage:invoke` | SSE state transitions, replayable |
| POST | `/tasks/{id}/input` | `triage:invoke` | Answer a paused task |

Every response carries `A2A-Version: 1.0`. It is *emitted, never negotiated* - a
client sending a different version is not rejected, because there is nothing to fall
back to.

---

## The Agent Card

Public on purpose. A card is a discovery document: it tells you how to authenticate
without containing any secret.

```
name          agentmesh-stu_007          <- carries the student id
provider      { organization: "Priya Raman", url: ... }   <- WHO operates it
skills[]      triage_incident, whoami    <- what it DOES
capabilities  { streaming, pushNotifications, extendedAgentCard }  <- protocol flags ONLY
endpoints     { base, tasks, stream, input, mcp }
securitySchemes  bearer / JWT, scope triage:invoke
supportedInterfaces[]                    <- ORDERED; entry 0 is preferred (v1.0)
signatures    []                         <- declared, unpopulated: we do not sign
```

**The mistake worth naming:** `capabilities` is *not* "what the agent can do". It is
three transport flags. Work lives in `skills[]`. The older shape - a `capabilities`
list of `{name, latency_p50_ms, cost_hint}` - is an anti-pattern, and a test fails if
anyone reintroduces it.

---

## The task lifecycle

Nine states exist in the spec; this flow drives seven. **The enum value is the wire
string** - there is no lowercase form anywhere.

```
SUBMITTED ──► WORKING ──┬─────────────────────────► COMPLETED
                        │
                        └─► INPUT_REQUIRED ──┬────► WORKING ──► COMPLETED   (approved)
                                             └────► REJECTED               (declined)
                     any point ─────────────────► FAILED / CANCELED
```

- `TASK_STATE_UNSPECIFIED` is a sentinel - `update_state` raises if you assign it.
- `TASK_STATE_AUTH_REQUIRED` has no producer here: auth happens at the door, so there
  is no mid-task auth challenge to raise.
- **Reject is terminal and executes nothing.** A test asserts no `executing` event was
  ever emitted on that path.
- No reply within 300 s → `FAILED` with `{reason: "input_timeout", retryable: false}`.

### One stream per *interaction*, not per task

This is the part people get wrong, and it is the difference between a build that works
over a tunnel and one that hangs.

A2A's streaming response covers **one interaction**, and its last event is marked
`final`. An **interrupted** state ends an interaction exactly as a terminal one does:
at `TASK_STATE_INPUT_REQUIRED` the task is not finished, but the next move is *yours*,
so the server has nothing left to push and closes the stream. You reply, then
**resubscribe** with the cursor to pick up the rest:

```
POST /tasks                      -> 202, task_id
GET  /tasks/{id}/stream          -> SUBMITTED, WORKING, INPUT_REQUIRED (final) ─┐ closes
POST /tasks/{id}/input           -> your approval                               │
GET  /tasks/{id}/stream          -> WORKING, COMPLETED (final)          resume ─┘
     Last-Event-ID: 4
```

Holding one stream open across the gate seems simpler and is a trap. Any buffering
intermediary - a CDN, a cloudflared tunnel - releases a response body when the response
**ends**. Keep it open and the caller never sees `INPUT_REQUIRED`, so it never replies,
so the stream never ends: a deadlock that presents as "the network is slow". A finite
response per interaction cannot deadlock, which is why the spec is shaped this way.

### `tasks/get` - the snapshot

`GET /tasks/{task_id}` answers "where is this task now?" in one finite response:
state, the payload that came with it, `last_event_id`, and `final` / `interrupted`.
The stream is the efficient way to watch a task; this is the reliable one. A caller
that never opened a stream, dropped one, or sits behind something that will not carry
one can still learn the outcome - which is what makes streaming an optimisation rather
than a dependency. `scripts/a2a_client.py` falls back to it automatically when a stream
yields nothing.

### Streaming and replay

Every SSE event carries a **per-task monotonic `event_id`** - never the length of the
buffer. `len(events)+1` looks right until the bounded buffer wraps at 100, after which
every `Last-Event-ID` replay silently returns the wrong events. Reconnect with a cursor
and you get everything after it; reconnect with a cursor older than the buffer and you
get an explicit `{"replay_gap": true, "oldest_available_event_id": N}` first. A hole you
are told about beats a hole you are not.

---

## The two skills

**`triage_incident`** - the real work. Input is `{description, severity, user_id}`,
validated *inside* the handler (it depends on which skill you asked for), so a bad body
is a `422` with per-field detail. Mutating actions pause at the human gate.

**`whoami`** - identity, no input, no gate, one hop. It exists so a classmate can prove
the entire authenticated path works - submit, stream, terminal state - before trusting
you with a real incident.

---

## Calling someone else

This repo ships both halves. `app/main.py` *serves* A2A;
[`scripts/a2a_client.py`](scripts/a2a_client.py) *calls* it.

```bash
python scripts/a2a_client.py --peer stu_001 --skill whoami
```

```bash
python scripts/a2a_client.py --peer stu_001 --description "Please restart payments-api now" --severity high --approve
```

`--peer` resolves the URL from [`cohort.json`](cohort.json), so nobody retypes a
tunnel address. Pass a full URL instead if you prefer. Drop `--approve` to be prompted;
`--reject` lands the task in `TASK_STATE_REJECTED` with nothing executed.

The browser UI does the same thing: paste a peer's base URL into **Target service**,
Connect, and Submit / stream / Approve (or Reject) all cross the network. Its **🔎
Capabilities** button lists a target's `skills[]` as clickable chips - clicking one runs
that skill as a task - and the **Cohort** tab sweeps the roster so you can pick a peer to
target without typing a URL. (The `tasks/get` snapshot has no button; the CLI uses it as
a fallback, and the notebook drives the lifecycle through `scripts/a2a_client.py`.)

---

## The failure that wastes an afternoon

`AGENTMESH_BASE_URL` is stamped into your Agent Card **and** into the `stream_url` of
every 202 you return. Leave it at `localhost` while running behind a tunnel and a
classmate will submit a task that runs perfectly on your machine and streams to *their*
localhost - forever, with no error anywhere.

Both clients here detect the mismatch, warn, and fall back to the URL you actually
dialled. They cannot fix what your card tells everyone else, so fix it at the source:
one URL, in [`cohort.json`](cohort.json) and `AGENTMESH_BASE_URL`, then restart.
