# Running AgentMesh: solo, then with the class

Two modes, one switch. Everything below is driven by
[`cohort.json`](cohort.json) - the single file where URLs live.

| | **Solo mode** | **Online mode** |
|---|---|---|
| Who you need | nobody | the class, at the same time |
| Network | localhost only | one cloudflared tunnel each |
| Peers | local dummy copies on 8001/8002 | real classmates |
| Auth | `AUTH_MODE=dev`, one shared token | `AUTH_MODE=jwks`, a real signed token each |
| Set up | nothing | tunnel + publish your card |

Start solo. Everything in the build works there. Online mode adds one thing:
the agent on the other end belongs to somebody else.

---

# Part 1 - Solo mode

`COHORT_MODE` defaults to `solo` in `.env`, so this is where you start - nothing to set.

### Run it

```bash
uvicorn app.main:app --reload --reload-dir app
```

> `--reload-dir app` scopes the file watcher to the source package. Without it,
> the rotating log file the server writes lands under the watched directory and
> trips `--reload` on every request - an endless reload loop.

That one process serves **both protocols**:

| URL | What |
|---|---|
| `http://localhost:8000/` | browser UI |
| `http://localhost:8000/.well-known/agent-card.json` | A2A discovery (no auth) |
| `http://localhost:8000/tasks` | A2A tasks (bearer, `triage:invoke`) |
| `http://localhost:8000/mcp/` | MCP (bearer, `mcp:invoke`) |
| `http://localhost:8000/health` | liveness (no auth) |

### Talk to yourself

```bash
python scripts/a2a_client.py http://localhost:8000 --skill whoami
```

Then the interesting one - it pauses at the human gate and you approve it:

```bash
python scripts/a2a_client.py http://localhost:8000 --description "Please restart payments-api now" --severity high --approve
```

### Simulate a cohort of one

Start a second copy **in another terminal** with a different identity and port. A
one-off environment variable is spelled differently in every shell - use the one that
matches yours:

**PowerShell** (the Windows default):

```powershell
$env:STUDENT_ID="stu_001"; $env:STUDENT_NAME="Alice"; $env:AGENTMESH_BASE_URL="http://localhost:8001"; uvicorn app.main:app --port 8001
```

**bash / zsh / Git Bash:**

```bash
STUDENT_ID=stu_001 STUDENT_NAME="Alice" AGENTMESH_BASE_URL=http://localhost:8001 uvicorn app.main:app --port 8001
```

**cmd** (note: no space before each `&&`, or the space lands inside the value):

```cmd
set STUDENT_ID=stu_001&& set STUDENT_NAME=Alice&& set AGENTMESH_BASE_URL=http://localhost:8001&& uvicorn app.main:app --port 8001
```

```cmd
set STUDENT_ID=stu_002&& set STUDENT_NAME=Bob&& set AGENTMESH_BASE_URL=http://localhost:8002&& uvicorn app.main:app --port 8002
```


> **PowerShell keeps `$env:` values for the rest of that terminal.** Run each instance in
> its own terminal - which you need anyway, since `uvicorn` blocks - or a later command
> in the same window will quietly inherit `stu_001`.
>
> Ports 8001/8002 are what `cohort.json` lists as solo peers. If something else on your
> machine already uses them, change the ports there and here to match.

`cohort.json`'s `solo.peers` already lists `stu_001` and `stu_002`, so now:

```bash
python scripts/a2a_client.py --peer stu_001 --skill whoami
```

```bash
python scripts/cohort_roster.py
```

The roster sweeps your solo peers and reports who answered. Peers you did not start
show up as unreachable - which is exactly what a classmate with a dead tunnel looks
like, so it is worth seeing once here.

### Dummy data

The fixtures are deliberately small and deterministic:

- **Cohort**: `stu_001` Alice, `stu_002` Bob, `stu_003` Cleo (`cohort_lookup`)
- **Incidents**: `INC-101` payments timeout, `INC-102` JWKS cache, `INC-103` db lag
- **Runbooks**: payments restart, auth-service triage (used by `triage_incident`)

`whoami` and `ping` are *not* fixtures - they read your real configured identity. So
in solo mode `cohort_lookup("stu_001")` returns dummy Alice, while
`cohort_lookup(<your own id>)` returns you.

---

# Part 2 - Online mode (the live class)

## Before class

1. Install cloudflared and test it once - [`CLOUDFLARED.md`](CLOUDFLARED.md).
2. Create your account on the cohort site and claim a callsign (step 2 below). Doing
   it in advance saves session time, and your instructor can grant your scopes early.
3. Leave `.env` alone for now. Claiming the callsign prints the block you paste -
   identity, `COHORT_MODE=online`, **and** the auth settings that switch you from `dev`
   to `jwks`. That block is the whole switch; you do not edit `cohort.json`.
4. Set `AGENTMESH_BASE_URL` to your tunnel (step 3 below). Online mode reads your URL
   from there, so `cohort.json` stays untouched.

## The three planes

This is the whole design in one table. Confusing these is what makes people think
"the static site should make it reachable" - it does not.

| Plane | Question | Solved by |
|---|---|---|
| **Discovery** | Who is in the class, and where? | a static site, public, no auth |
| **Connectivity** | Can I actually open a connection? | one cloudflared tunnel each |
| **Authorisation** | Am I allowed to invoke this? | your Firebase ID token, verified against Google's JWKS |

Cards are public on purpose - a card says *how* to authenticate without containing
any secret.

## Instructor: the coordination site, once

Deploy the cohort coordination site (Firebase Auth + Realtime Database + Hosting,
free tier); it ships with its own README and setup steps. Hand the class two things:
the **site URL**, and a running scope-granting watcher so the console's grant buttons
take effect during the session.

## Student: five steps

### 1. Open the tunnel

```bash
cloudflared tunnel --url http://localhost:8000
```

Copy the `https://<random>.trycloudflare.com` it prints. Leave this terminal alone.

### 2. Sign up and get your identity

On the cohort site: create an account, enter your name, claim a callsign. It shows you
an `.env` block - paste it in. That block sets `STUDENT_ID`, `STUDENT_NAME` and the
Firebase JWT settings, so your agent verifies classmates' tokens against Google's JWKS.

Ask the instructor to grant your scopes, then get a **fresh token** - from the notebook,
`boss.refresh()`; on the site, sign out and back in. A scope is a custom claim, and claims
are baked into a token when it is issued, so the one you are already holding will keep
reporting `scopes_present: []` and 403-ing until it is replaced.

### 3. Start the service pointed at your tunnel

Put this in `.env` (no shell syntax to get wrong):

```bash
AGENTMESH_BASE_URL=https://your-tunnel.trycloudflare.com
```

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check your own card advertises the tunnel, not localhost:

```bash
curl -s https://your-tunnel.trycloudflare.com/.well-known/agent-card.json
```

### 4. Publish your URL

Either paste it into the cohort site, or from the notebook (Section 9):

```python
boss.publish("https://your-tunnel.trycloudflare.com")
```

Re-publish every time the tunnel restarts - the URL changes each time.

### 5. See who is up

The site's live board shows everyone, probed from your browser. From the notebook,
Section 10 does the same sweep. Run it before everyone starts calling each other: it
turns "it doesn't work" into "their tunnel is down and yours has the wrong token".

## Then: actually use each other

```bash
python scripts/a2a_client.py --peer stu_012 --skill whoami
```

```bash
python scripts/a2a_client.py --peer stu_012 --description "Please restart payments-api now" --severity high --approve
```

You will watch their agent classify, pause at *their* approval gate, and resume when
*you* approve from your terminal. Two laptops, one task, and as far as the protocol
cares, two organisations.

From the browser: open your own UI, paste their base URL into **Target service (A2A)**,
set the token, Connect.

Their MCP tools work too - with a bearer, which the local stdio server never needed.
See [`MCP.md`](MCP.md).

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Their card shows `localhost` | They did not set `AGENTMESH_BASE_URL`, or did not restart after setting it |
| Task submits, stream stays empty | Same cause - their `stream_url` points at your machine. Both clients here warn and fall back |
| `401 missing_bearer` | No token sent |
| `401 auth_invalid` | Solo: not your `DEV_BEARER_TOKEN`. Online: wrong issuer/audience for the cohort project, or the token expired - they last an hour, so get a fresh one |
| `403 scope_missing` | Token lacks `triage:invoke` (A2A) or `mcp:invoke` (MCP). If `scopes_present` is `[]`, either the instructor has not granted you yet, or you are still holding the token issued *before* they did - run `boss.refresh()` |
| `409 task_not_paused` | You replied before the task reached the gate - wait for `INPUT_REQUIRED` on the stream |
| Events arrive in one lump | A proxy is buffering SSE. Harmless here: the stream ends at each interaction, so the lump is the whole leg |
| Stream opens, then nothing | An intermediary is holding the body. `scripts/a2a_client.py` falls back to polling `GET /tasks/{id}` automatically - the notebook prints when it does |
| Task fails with `input_timeout` | Nobody approved within 300 s |
| Tunnel URL stopped working | It changed on restart, or the laptop slept. Re-do steps 1-4 |

**Never tunnel the standalone MCP server** (`python -m app.mcp_server` with
`MCP_SERVER_TRANSPORT=http`). It has no auth. The co-hosted `/mcp/` surface is the
guarded one - that is the only MCP path that should see a public URL.
