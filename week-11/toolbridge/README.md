# ToolBridge - Week 11, Project 22

**An MCP server that is an A2A client.** Your host speaks MCP. A classmate's
agent speaks A2A. ToolBridge sits in the middle and makes the agent look like a
tool.

That sentence is the whole of the week in one artefact, and it is why this is the
graded build rather than a third copy of something you already ran.

```
   MCP host                ToolBridge                     AgentMesh (the peer)
   (Claude Desktop,     +-----------------+            +---------------------+
    Cursor, the     --> | peer_card       | -- A2A --> | GET  /.well-known/  |
    client demo)    MCP | delegate_triage |            | POST /tasks         |
                        | resume_task     |            | GET  /tasks/{}/stream|
                        +-----------------+            | POST /tasks/{}/input|
                              mcp 2.2.0                 +---------------------+
                            protocol 2026-07-28            mcp 1.28.1 era
```

---

## Read this before you install anything

**This project pins `mcp==2.2.0`. The two guided builds in Section 3 pin
`mcp==1.28.1`. They are different majors.**

| | Section 3 builds | This project |
|---|---|---|
| SDK | `mcp==1.28.1` | `mcp==2.2.0` |
| Protocol revision | 2025-11-25 | **2026-07-28** |
| First message on the wire | `initialize` | whatever the client sends |
| Server class | `FastMCP` | `MCPServer` |
| Pause for a human | `elicitation/create`, server-sent | `InputRequiredResult`, client retries |

Use a **separate virtual environment** for this project. Mixing them produces
`ModuleNotFoundError: No module named 'mcp.server.fastmcp'`, which is the 2.x
SDK telling you exactly what happened, with a link.

That split is deliberate. The pinned older SDK is what the great majority of
deployed servers speak today, so you should be able to read it. This one is what
the specification now says, so you should be able to write it.

---

## Setup (5 min)

```bash
# 1. A venv OF ITS OWN
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Config
cp .env.example .env
# Open .env. The only value you MUST get right is PEER_BASE_URL.

# 4. Start a peer. From your agentmesh directory, in another terminal:
#    (bash / zsh / Git Bash)
STUDENT_ID=stu_001 STUDENT_NAME=Alice AUTH_MODE=dev \
  AGENTMESH_BASE_URL=http://localhost:8001 uvicorn app.main:app --port 8001
#    (PowerShell)
# $env:STUDENT_ID="stu_001"; $env:STUDENT_NAME="Alice"; $env:AUTH_MODE="dev";
# $env:AGENTMESH_BASE_URL="http://localhost:8001"; uvicorn app.main:app --port 8001

# 5. Check it, before you write a line
python scripts/peer_up.py
```

**You do not need a classmate.** The peer is a second copy of AgentMesh on your
own machine, which `SESSION.md` in that project calls "a cohort of one". Nothing
in this project is graded on somebody else being online.

---

## What ships, and what you build

| Ships complete | You write |
|---|---|
| `app/a2a.py` - the A2A client half | `app/descriptions.py` - TASK 1 |
| `app/config.py`, `app/errors.py` | `app/tools.py` - TASKS 2, 3, 4, 5 |
| `app/mcp_server.py` - registration and `main()` | |
| `app/eval.py` and the golden set | |
| `scripts/`, `tests/` - assertions already written | `pyproject.toml` - TASK 6 |

The tests are written and they fail. Making them pass, for the right reasons, is
the assignment.

---

## The six TASKs

Each one is marked `# TASK n:` in the source. Work them in order; each builds on
the last.

**TASK 1 - the descriptions.** Three tools, three descriptions, in
`app/descriptions.py`. Five moves each: verb first, scope explicit, trigger
stated, output shape declared, edge cases listed. Then run
`python -m app.eval` and again with `TOOL_DESCRIPTION_QUALITY=bad`, and write
both numbers in your memo.

**TASK 2 - `peer_card`.** Fetch a peer's card, pick an interface, report what you
found. The graded part is that you walk `supportedInterfaces` **in order** and
report the rank you used. A card with a single top-level `url` and no
`supportedInterfaces` is from before March 2026 and you should say so rather than
crash.

**TASK 3 - `delegate_triage`.** Submit a task, follow the stream to a settled
state. No busy-wait loop polling `GetTask` every hundred milliseconds: follow the
stream, and when it drops, **re-attach from the last event id** rather than
resubmitting.

**TASK 4 - the human gate.** The interesting one. The peer pauses at
`TASK_STATE_INPUT_REQUIRED`, which is **interrupted, not terminal**: the task is
alive and holding state. Decide where that pause surfaces on the MCP side. Two
paths are open to you and `HUMAN_GATE_MODE` switches between them:

- `resume` - return a structured result saying `pending_approval`, hand back the
  task id and the replay cursor, and let the model call `resume_task`.
- `elicit` - use the SDK's elicitation round trip, which on this protocol
  revision becomes an `InputRequiredResult` and a client retry.

Pick one, make it work, and defend the choice in the memo. There is no single
right answer and that is the point.

**TASK 5 - failures.** Four of them: a 401, a 422, a dropped stream, and a 409
from replying to a task that is not paused. Each returns the one failure shape
from `app/errors.py`, with a **correct `retryable` flag**. `TASK_STATE_REJECTED`
is not retryable: the peer read the request and declined it.

**TASK 6 - packaging.** Declare the entrypoint, ship the fixture data inside the
package, and prove it:

```bash
python -m venv /tmp/clean_env
/tmp/clean_env/bin/pip install .
/tmp/clean_env/bin/toolbridge          # starts, logs its config, waits on stdin
```

A server nobody can install is a server nobody will use.

---

## Try it out

```bash
pytest -q                                      # 27 tests, no network, ~1s
python scripts/peer_up.py                      # is the peer usable?
python scripts/mcp_client_demo.py stdio        # the real MCP client, over stdio
MCP_SERVER_TRANSPORT=http python -m app.mcp_server   # then, in another terminal:
python scripts/mcp_client_demo.py http
python -m app.eval                             # needs OPENAI_API_KEY
```

The client demo prints the negotiated protocol version. It says `2026-07-28`.
Run the equivalent in either Section 3 build and you will watch an `initialize`
go past instead.

---

## Common failure modes

**`ModuleNotFoundError: No module named 'mcp.server.fastmcp'`** - you are in the
right venv. That module does not exist in mcp 2.x. The error message names its
replacement.

**`AttributeError: 'Tool' object has no attribute 'inputSchema'`** - the wire
field is `inputSchema`; the Python attribute is `input_schema`. Same for
`isError` / `is_error` and `structuredContent` / `structured_content`.

**`peer_unreachable`** - the peer is not running. `python scripts/peer_up.py`.

**A 422 on submit with `user_id` in the detail** - this is worth understanding
rather than just fixing. **A2A does not carry a JSON Schema for a skill's
input.** The card declares media types and nothing about fields. The only ways
to learn the shape are to read the peer's documentation or to send something and
read the 422. That is a real gap in the protocol, and it is why this project
validates the three known fields locally before dialling the peer: failing here
with a sentence beats failing there with a validation dump.

**`pending_approval` coming back from `resume_task`** - you resumed from cursor
zero, so the peer replayed the pause you had just answered and you read it as the
current state. Carry `resume_from` through. This is the handle pattern from
W11-R01: the protocol will not remember for you.

---

## What is real and what is not

- **Real:** the MCP server, both transports, the A2A client, the card walk, the
  stream with replay, the human gate, the structured failures, the packaging.
- **Mocked:** nothing in this project. The peer is a real AgentMesh, and the
  triage core behind it is the TriageFlow you built in Week 10.
- **Not implemented:** card signature verification. We report whether a card is
  signed and proceed regardless, and `peer_card` tells you which. Deciding what
  to do about an unsigned card is a policy question, not a code one.

---

## Where this goes next

Week 12 puts guardrails in front of everything here. Week 14 deploys it and gives
it real tracing. Week 16 asks what it costs. And the capstone in Week 17 will
almost certainly want a bridge like this one somewhere in it.
