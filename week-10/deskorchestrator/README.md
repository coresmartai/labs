# DeskOrchestrator

**Applied GenAI &amp; Agentic AI Engineering Program - Week 10, Project 20 (graded)**

A routed multi-agent service desk on LangGraph. One router in front of three
specialists, four memory layers behind a single scoped write path, and a human
approval gate on the only calls that change the world.

**This is a scaffold, not a finished app.** Four things are yours. Everything
else runs as shipped.

---

## 0. Before you write anything, run the tests

```
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS and Linux
pip install -r requirements.txt
pytest -q
```

You should see:

```
5 failed, 20 passed
```

Run it with no `.env` present. The settings object reads one if it is there, and
a stray value in it can change what you see. (The scope tests pass `k` explicitly
rather than reading `RETRIEVAL_K`, so tuning that one is safe either way.)

**That is the correct starting state, not a broken download.** Read the five
failure messages before you read any source file. They name properties, not
locations.

Python 3.10 or newer. No API key is needed for the tests: they never call a
provider and never open a socket.

---

## 1. What you are building

```
deskorchestrator/
├── app/
│   ├── config.py         typed settings, one model pin per agent
│   ├── schemas.py        the API models and the DeskState handoff contract
│   ├── state.py          assert_handoff: presence, not truthiness
│   ├── corpus.py         eight global policies, prior tickets per user
│   ├── vectorstore.py    <- TASK 1 lives here. Ships known-wrong
│   ├── memory.py         four layers, one scoped write path, one delete
│   ├── tools.py          three tools, all mocked, dispatcher never raises
│   ├── risk.py           <- TASK 3 lives here. Ships pausing for everything
│   ├── llm.py            the provider seam
│   ├── graph.py          the graph, the router, the gate
│   └── main.py           FastAPI
├── prompts/              four prompts, on disk, versioned like the code
├── eval/
│   ├── run_routing_eval.py    the harness. Ships working
│   └── routing_eval.json      <- TASK 2 lives here. Ships as a template
├── tests/
│   ├── test_endpoint.py       17 tests. All pass on arrival
│   ├── test_scope.py          4 tests. Three fail
│   └── test_risk.py           4 tests. Two fail
├── index.html            the browser page, with the retrieval inspector
└── DESIGN.md             <- TASK 4 lives here
```

### The graph

```
START -> triage -> access   ─┐
                   software ─┼-> execute -> compose -> END
                   hardware ─┘      ^
                                    └── the gate pauses here, conditionally
         triage -> compose                (escalate_human skips both)
```

`execute` is a node name, not a route literal. The router emits exactly four
strings and reaches `execute` by a static edge from a specialist, never by the
classifier.

---

## 2. The four tasks

### Task 1: make the scope filter run before the ranking

`app/vectorstore.py`, `InProcessVectorStore.search`. It ranks every row in the
store, takes the top k across all scopes, and only then discards the rows that
do not belong to the caller. That is post-filtering, and it is what pgvector does
with an approximate index unless you configure it otherwise.

**The fix is two lines. Understanding why it is two lines is the project.**

Three tests fail and one passes, and the one that passes matters:
`test_post_filtering_does_not_leak_and_this_passes_on_arrival`. **Filtering after
ranking still filters.** No other user's row can reach the caller, whatever order
the steps happen in. What breaks is *recall*: a user with three stored tickets is
told they have none, and nothing errors anywhere. Get that distinction straight
before you start, because the fix for a recall problem is not the fix for a leak.

### Task 2: write and run the routing eval

`eval/routing_eval.json` ships as a three-case template. Replace it with a real
labelled set: each case is a request and the route it should take.

```
python eval/run_routing_eval.py --dry-run   # no calls, checks your file parses
python eval/run_routing_eval.py             # one nano call per case
```

The harness scores the **router in isolation**, with the specialists stubbed out.
That is the step people skip and the step that makes the whole thing work: a
regression in answers then traces cleanly to either the supervisor or a worker,
and not to the soup of both.

Twenty cases is the usual floor. Cover all four routes, and include cases that
sit near a boundary, because the easy ones tell you nothing. Report the accuracy
and name the confusion that hurt most.

### Task 3: write the approval policy

`app/risk.py`. It ships returning `True` for everything, which is a gate that
fires on every proposal, and a gate that wakes somebody at 3 a.m. for a keyboard
is a gate people learn to click through.

Three constraints, and they are not style preferences:

1. **It must be a pure function of the proposal.** Resuming re-runs the node from
   the top, so `risky()` is evaluated again after the pause. A verdict that can
   change on re-entry is a gate that can be walked around.
2. **No I/O**, for the same reason. Everything above the pause executes twice.
3. **You must be able to defend it in DESIGN.md.**

`tests/test_risk.py` has four tests. One pins the purity constraint, and three pin
points on the range: a write grant on a production system must pause, a single
stock keyboard must not, and **a proposal of `none` must not**, because there is
nothing there to approve. The middle of the range is yours.

### Task 4: the design memo

`DESIGN.md`. Four questions, and the fourth is the one that cannot be copied.

---

## 3. Running it

```
cp .env.example .env      # then set OPENAI_API_KEY
uvicorn app.main:app --reload --port 8000
```

| Endpoint | What it does |
|---|---|
| `POST /desk` | Start a request. Returns `completed` or `pending_approval` |
| `POST /approve` | Resolve a pending action. 409 if the thread is not paused at the gate |
| `POST /prefs` | Write a procedural rule for a user |
| `DELETE /user/{id}` | Right to be forgotten, with a four-counter receipt |
| `GET /health` | One model pin per agent |
| `GET /debug/retrieval` | One scoped search and one count, side by side. See below |
| `GET /` | The browser page |

```
curl -s -X POST http://localhost:8000/desk -H "Content-Type: application/json" \
  -d '{"user_request":"I need write access to billing-db","user_id":"u_1"}'
```

### The retrieval inspector, which is the fastest way to see Task 1

Open `http://localhost:8000/` and use the panel at the bottom left, or call it
directly:

```
curl -s "http://localhost:8000/debug/retrieval?q=licence+seat+cost+centre&user_id=u_1&k=3"
```

On a fresh clone:

```json
{"owned_by_this_user": 3, "returned_count": 0, "recall_ok": false}
```

Three rows exist for this user and the search returned none of them. **That is
not a leak**, and the distinction is worth more than the fix: no other user's
rows reached you and none can, because filtering after ranking still filters.
What broke is which rows got *considered*. After your two-line change the same
call returns `3` and `recall_ok: true`.

The panel exists because this failure has no other symptom. It does not throw,
it does not log, and a broken request looks exactly like a working one with
nothing to say. Two numbers that disagree is the cheapest honest signal it has.

**A note on the browser page before you think it is broken.** On a fresh clone
every route pauses at the approval gate, including the one that proposes no tool
at all. That is Task 3: `risky()` ships returning `True` for everything, and two
tests in `tests/test_risk.py` fail on arrival saying so. Nothing about the UI is
wrong; the policy is not written yet.

---

## 4. What ships working, and what it is worth knowing about

**The memory layers.** External is the global policy corpus. Episodic is each
user's prior tickets. Procedural is a dict standing in for one SQL row per user,
and it has a real writer, because a layer whose writer has no callers does not
exist. Every write goes through `memory_write`, which asserts the scope before it
does anything else. Not the endpoints, not the graph, not the loader.

**The deletion path.** `DELETE /user/{id}` returns four counters, one per
destination this system actually has, **including the checkpoint table**. The
`policy_rows_deleted` counter is structurally zero, and that zero proves less
than it looks like it proves: it is evidence about what was deleted and says
nothing about what was left. The over-broad case needs a test, and there is one
in `test_endpoint.py`.

**The fallback.** An unrecognised router output falls back to `escalate_human`,
which is the least consequential branch for this risk model rather than the
cheapest one. It does nothing, costs no extra model call, and its worst case is a
person reading something they did not need to read.

**Known limits, stated rather than hidden.** There is no authentication: the
reviewer id on `/approve` is a self-asserted string, and CORS is wide open. Week
12 owns the design of human-in-the-loop; this package owns the mechanism.

**And one deliberate omission worth naming, because the second reading covers
it.** `execute_node` has the right *shape* - one mutating call, last statement,
below every branch that can pause - and that shape is necessary rather than
sufficient. If the tool succeeds and the process dies before the returned state
update is checkpointed, the resume replays the node and the action fires twice.
The remedy is an idempotency key derived from the thread id and a proposal id.
**This package does not have one.** It is not part of the graded work, and adding
it is the obvious extension if you want one. Do not deploy this.

---

## 5. Submitting

A pull request containing:

- `app/vectorstore.py` with the filter fixed
- `eval/routing_eval.json` plus the accuracy you measured
- `app/risk.py` plus any tests you added
- `DESIGN.md`

**Twenty marks, twelve to pass.** Correctness 7, Completeness 6, Design 4,
Clarity 3. Draft PR by Friday 6pm, as every week.
