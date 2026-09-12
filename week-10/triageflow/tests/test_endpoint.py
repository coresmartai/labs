"""Smoke + contract tests - exercise the FastAPI shape WITHOUT calling real models.

The graph is monkey-patched with a stub LLM call so these run offline,
deterministically, in milliseconds, and CI can use them as a regression gate.
Two categories ship this week:
  * smoke tests    - hit the endpoints end-to-end with a stubbed model
  * contract tests - assert_handoff fires loudly when an upstream write is missing
The nightly eval-regression harness arrives later.
"""
from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-stub")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    # ensure cache is fresh per test run
    from app import config
    config.get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _memory() -> Any:
    """Fresh, seeded memory per test.

    The vector store and the procedural dict are process singletons, so without
    this a deletion test would leak into the next test. Seed, run, reset.
    """
    from app import authz, memory, tools, vectorstore
    authz.reset_authz()
    vectorstore.reset_store()
    memory._PREFS.clear()
    # The idempotency ledger and the resource versions are process-global too,
    # so a replayed-call test would otherwise poison the next test.
    tools.reset_tool_state()
    memory.seed_memory()
    memory.seed_demo_prefs()
    yield
    vectorstore.reset_store()
    memory._PREFS.clear()
    tools.reset_tool_state()
    authz.reset_authz()


@pytest.fixture()
def stub_llm(monkeypatch: pytest.MonkeyPatch):
    """Replace app.llm.call with a deterministic stub."""
    from app import llm

    def _stub(model: str, system: str, messages: list[dict[str, Any]], **kw: Any) -> llm.LLMResult:
        content = messages[-1]["content"].lower()
        if "available tools" in content:
            # Action node prompt. Two proposals, because the gate is now
            # CONDITIONAL: one tool that risky() pauses and one it waves through.
            # Match the USER REQUEST half only. The tools schema below it
            # mentions every tool by name, so a naive `in content` matches always.
            if "ticket" in content.split("available tools")[0]:
                return llm.LLMResult(
                    text='{"tool_name": "file_ticket", "arguments": '
                         '{"title": "printer jam", "body": "third floor"}}',
                    model=model, latency_ms=10, input_tokens=10, output_tokens=10,
                )
            return llm.LLMResult(
                text='{"tool_name": "restart_service", "arguments": {"service_name": "payments-api"}}',
                model=model, latency_ms=10, input_tokens=10, output_tokens=10,
            )
        if "retrieved chunks" in content:
            return llm.LLMResult(
                text="Run `kubectl rollout restart deployment/payments-api` [doc:0].",
                model=model, latency_ms=10, input_tokens=10, output_tokens=10,
            )
        # Triage prompt - match only imperative action requests, not how-to questions
        if "please restart" in content or "page the" in content or "please file" in content:
            return llm.LLMResult(text="action", model=model, latency_ms=5, input_tokens=5, output_tokens=2)
        return llm.LLMResult(text="knowledge", model=model, latency_ms=5, input_tokens=5, output_tokens=2)

    monkeypatch.setattr("app.llm.call", _stub)
    monkeypatch.setattr("app.graph.call", _stub)
    # rebuild the graph using the stubbed module
    from app import main
    main._graph = main.build_graph(checkpointer=main._checkpointer)
    yield


@pytest.fixture()
def stub_redis(monkeypatch: pytest.MonkeyPatch):
    class _Pipe:
        def __init__(self, store): self.store = store
        def rpush(self, key, val): self.store.append(("rpush", key)); return self
        def expire(self, *a, **k): return self
        def execute(self): return None
    class _Stub:
        def __init__(self): self.calls: list[tuple[str, str]] = []
        def lrange(self, *a, **k): return []
        def pipeline(self): return _Pipe(self.calls)
        def scan(self, cursor, match=None, count=None):
            self.calls.append(("scan", match)); return (0, [])
        def delete(self, *a): return 0
    stub = _Stub()
    monkeypatch.setattr("app.memory._r", lambda: stub)
    return stub


def as_user(user_id: str) -> dict[str, str]:
    """Every endpoint but /health now needs a caller identity.

    In production this header is replaced by a verified bearer token. The tests
    care about the AUTHORIZATION checks behind it, which do not change when it is.
    """
    return {"X-User-Id": user_id}


# ---------------------------------------------------------------- smoke tests

def test_health(stub_llm, stub_redis):
    from app.main import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    # assert status only - /health response grows as more model keys are added
    assert r.json()["status"] == "ok"
    assert "model" in r.json()   # health chip reads this field


def test_health_reports_all_three_model_pins(stub_llm, stub_redis):
    """Three settings, three lines. A chip that shows two of three pins is how a
    re-pointed model stays invisible for a week."""
    from app.main import app
    client = TestClient(app)
    models = client.get("/health").json()["models"]
    assert set(models) == {"triage", "knowledge", "action"}


def test_a_provider_failure_is_a_502_that_names_itself(stub_llm, stub_redis, monkeypatch):
    """A mistyped key is not a bug in the graph, and a bare 500 sends a learner
    looking for one. Everything that IS a bug in here still raises a 500."""
    import httpx
    from openai import AuthenticationError

    from app import main

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    def _boom(*a, **kw):
        raise AuthenticationError("Incorrect API key provided",
                                  response=httpx.Response(401, request=req), body=None)

    monkeypatch.setattr("app.llm.call", _boom)
    monkeypatch.setattr("app.graph.call", _boom)
    main._graph = main.build_graph(checkpointer=main._checkpointer)
    with TestClient(main.app, raise_server_exceptions=False) as client:
        r = client.post("/triage", json={"user_request": "hello", "user_id": "u_1"},
                        headers=as_user("u_1"))
    assert r.status_code == 502
    assert r.json()["error_type"] == "AuthenticationError"
    assert "OPENAI_API_KEY" in r.json()["detail"]


def test_knowledge_route(stub_llm, stub_redis):
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "How do I restart the payments API?", "user_id": "u_1"}, headers=as_user("u_1"))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert "[doc:0]" in (body["final_answer"] or "")


def test_citations_parsed_from_answer(stub_llm, stub_redis):
    """Only the docs actually cited via [doc:N] markers become citations."""
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "How do I restart the payments API?", "user_id": "u_1"}, headers=as_user("u_1"))
    body = r.json()
    # the stub answer cites only [doc:0]; the stub retriever returns 3 docs
    assert body["citations"] == [{"doc_index": 0, "source": "runbooks/payments-restart.md"}]


def test_action_route_pauses_for_approval(stub_llm, stub_redis):
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "Please restart payments-api now", "user_id": "u_1"}, headers=as_user("u_1"))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending_approval"
    assert body["proposed_action"]["tool_name"] == "restart_service"


def test_a_low_risk_action_runs_without_pausing(stub_llm, stub_redis):
    """The whole point of moving the gate inside the node.

    Under interrupt_before, EVERY visit to action_execute paused, so filing a
    ticket about a printer jam woke an on-call reviewer. risky() is what makes
    the pause conditional, and this is the test that would have failed before
    the rewrite. It is the counterpart to test_action_route_pauses_for_approval:
    same graph, same node, different proposal, different behaviour.
    """
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "Please file a ticket for the printer jam",
                                         "user_id": "u_1"}, headers=as_user("u_1"))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed", "a low-risk tool call should not pause"
    assert "file_ticket" in (body["final_answer"] or "")
    # The route is REPORTED, not inferred. The UI used to read "completed" as
    # "knowledge", which was only ever true while every action paused.
    assert body["route"] == "action"


def test_a_low_risk_thread_is_not_resumable(stub_llm, stub_redis):
    """Corollary: a thread that never paused has nothing to approve, and the
    endpoint says so instead of replaying the node and firing the tool twice."""
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "Please file a ticket for the printer jam",
                                         "user_id": "u_1"}, headers=as_user("u_1"))
        thread_id = r.json()["thread_id"]
        r2 = client.post("/approve", json={"thread_id": thread_id, "approved": True,
                                           "reviewer_id": "ops_lead"}, headers=as_user("ops_lead"))
    assert r2.status_code == 409


def test_approve_resumes_thread(stub_llm, stub_redis):
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "Please restart payments-api now", "user_id": "u_1"}, headers=as_user("u_1"))
        thread_id = r.json()["thread_id"]
        r2 = client.post("/approve", json={"thread_id": thread_id, "approved": True, "reviewer_id": "ops_lead"}, headers=as_user("ops_lead"))
    assert r2.status_code == 200
    final = r2.json()["final_answer"]
    assert "Action complete" in final


def test_reject_declines_action(stub_llm, stub_redis):
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "Please restart payments-api now", "user_id": "u_1"}, headers=as_user("u_1"))
        thread_id = r.json()["thread_id"]
        r2 = client.post("/approve", json={"thread_id": thread_id, "approved": False, "reviewer_id": "ops_lead"}, headers=as_user("ops_lead"))
    assert r2.status_code == 200
    assert "declined" in r2.json()["final_answer"].lower()


def test_a_replayed_call_does_not_fire_the_tool_twice(stub_llm, stub_redis):
    """The crash window, made concrete.

    execute_tool is called twice with the SAME key, which is what a resume after
    a crash looks like from inside the node. The implementation must run once.
    """
    from app import tools

    key = "stable-key-from-thread-and-proposal"
    first = tools.execute_tool("restart_service", {"service_name": "payments-api"},
                               idempotency_key=key)
    second = tools.execute_tool("restart_service", {"service_name": "payments-api"},
                                idempotency_key=key)
    assert first["status"] == "restarted"
    assert second.get("replayed") is True, "the second call reached the implementation"
    # The resource moved exactly once, which is the property that matters.
    assert tools._VERSIONS["service:payments-api"] == 1


def test_a_fresh_key_every_call_is_the_same_as_no_key(stub_llm, stub_redis):
    """The failure mode the key is supposed to prevent, demonstrated.

    This is why idempotency_key() derives from state rather than generating a
    uuid at the call site: the guard is the STABILITY, not the key.
    """
    import uuid

    from app import tools
    for _ in range(2):
        tools.execute_tool("restart_service", {"service_name": "payments-api"},
                           idempotency_key=uuid.uuid4().hex)
    assert tools._VERSIONS["service:payments-api"] == 2


def test_the_idempotency_key_survives_a_replay(stub_llm, stub_redis):
    """Same state in, same key out - across any number of node entries."""
    from app.graph import idempotency_key

    state = {"thread_id": "th_abc", "proposal_id": "prop_1"}
    assert idempotency_key(state) == idempotency_key(dict(state))
    # A different proposal on the same thread is a different call, and must not
    # be swallowed as a duplicate of the first.
    assert idempotency_key(state) != idempotency_key({"thread_id": "th_abc",
                                                      "proposal_id": "prop_2"})


def test_an_approval_is_not_executed_after_the_world_moved(stub_llm, stub_redis):
    """The stale-proposal case. Approve a restart, have someone else restart the
    service while the approver is deciding, then resume. The plan was made
    against a world that no longer exists, so the node refuses and says why."""
    from app import tools
    from app.main import app

    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "Please restart payments-api now",
                                         "user_id": "u_1"}, headers=as_user("u_1"))
        thread_id = r.json()["thread_id"]
        # Another operator, outside this graph, during the human pause.
        tools.restart_service("payments-api")
        r2 = client.post("/approve", json={"thread_id": thread_id, "approved": True,
                                           "reviewer_id": "ops_lead"}, headers=as_user("ops_lead"))
    assert r2.status_code == 200
    assert "resource changed" in r2.json()["final_answer"]
    # The approved restart did NOT run: only the other operator's one did.
    assert tools._VERSIONS["service:payments-api"] == 1


def test_an_expired_approval_is_not_executed(stub_llm, stub_redis, monkeypatch):
    """The other end of the same window. An approval that arrives after the TTL
    is not a decision about the current world, whatever it says."""
    from app import config, tools
    from app.main import app

    monkeypatch.setenv("APPROVAL_TTL_SECONDS", "0")
    config.get_settings.cache_clear()  # type: ignore[attr-defined]

    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "Please restart payments-api now",
                                         "user_id": "u_1"}, headers=as_user("u_1"))
        thread_id = r.json()["thread_id"]
        r2 = client.post("/approve", json={"thread_id": thread_id, "approved": True,
                                           "reviewer_id": "ops_lead"}, headers=as_user("ops_lead"))
    assert r2.status_code == 200
    assert "approval window" in r2.json()["final_answer"]
    assert tools._VERSIONS["service:payments-api"] == 0


def test_the_trace_has_one_entry_per_node(stub_llm, stub_redis):
    """llm_calls carries a REDUCER, so a node returns only its own entry.

    A node that also does the read-modify-write - `state["llm_calls"] + [entry]`
    - hands the reducer a list that already contains everything upstream, and
    the reducer appends it again. The knowledge route runs two nodes and must
    therefore show exactly two calls: triage, then knowledge.
    """
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "How do I restart the payments API?",
                                         "user_id": "u_1"}, headers=as_user("u_1"))
    steps = [c["step"] for c in r.json()["llm_calls"]]
    assert steps == ["triage", "knowledge"], f"duplicated trace entries: {steps}"


def test_approve_unknown_thread_409(stub_llm, stub_redis):
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/approve", json={"thread_id": "th_nonexistent", "approved": True,
                                      "reviewer_id": "ops_lead"}, headers=as_user("ops_lead"))
    assert r.status_code == 409


# ------------------------------------------------------------- contract tests

def test_assert_handoff_raises_on_missing_field():
    from app.state import assert_handoff
    with pytest.raises(ValueError, match=r"handoff to compose missing fields: retrieved_docs"):
        assert_handoff({"route": "knowledge"}, ["retrieved_docs"], "compose")


def test_compose_fails_loudly_without_knowledge_writes():
    """Failure demo #2 from the video: drop the Knowledge writes, Compose screams."""
    from app.graph import compose_node
    state = {"route": "knowledge", "user_request": "how?", "user_id": "u_1"}
    with pytest.raises(ValueError, match="missing fields"):
        compose_node(state)


def test_compose_is_single_writer_of_final_answer(stub_llm, stub_redis):
    """knowledge_node must NOT write final_answer - Compose owns that field."""
    from app import graph
    out = graph.knowledge_node({"user_request": "How do I restart?", "user_id": "u_1", "route": "knowledge"})
    assert "final_answer" not in out
    assert out["knowledge_summary"]


# ------------------------------------------------------------- memory contract

def test_session_keys_are_user_scoped(stub_llm, stub_redis):
    """GDPR bug regression: keys must be session:{user_id}:{thread_id} so
    session_clear_for_user's SCAN pattern actually matches them."""
    from app import memory
    memory.session_append("u_9", "th_abc", {"x": 1})
    memory.session_clear_for_user("u_9")
    written = [k for op, k in stub_redis.calls if op == "rpush"]
    scanned = [k for op, k in stub_redis.calls if op == "scan"]
    assert written == ["session:u_9:th_abc"]
    assert scanned == ["session:u_9:*"]
    import fnmatch
    assert fnmatch.fnmatch(written[0], scanned[0])


def test_runbook_search_returns_five_chunks():
    """Top-5 runbook chunks, as `09_memory_write_on_handoff.svg` draws it."""
    from app.memory import search_runbooks
    assert len(search_runbooks("restart payments", k=5)) == 5


def test_vector_search_ranks_the_right_runbook_first():
    """Not a list slice: a real cosine ranking. The restart runbook wins the
    restart query, and the score ordering is monotonic."""
    from app.memory import search_runbooks
    hits = search_runbooks("payments-api is unresponsive, how do I restart it?", k=5)
    assert hits[0]["chunk_id"] == "rb-1"
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_episodic_is_scoped_to_the_user():
    """The multi-tenant pin: u_1's prior incidents are invisible to u_2."""
    from app.memory import search_episodic
    mine = search_episodic("payments outage", user_id="u_1", k=3)
    theirs = search_episodic("payments outage", user_id="u_2", k=3)
    assert {h["chunk_id"] for h in mine} == {"ep-u1-1", "ep-u1-2"}
    assert {h["chunk_id"] for h in theirs} == {"ep-u2-1"}
    assert search_episodic("anything", user_id="u_nobody", k=3) == []


def test_knowledge_node_retrieves_runbooks_and_prior_incidents(stub_llm, stub_redis):
    """w10v01: 'retrieves runbooks AND prior-incident summaries from a vector
    store'. Both retrievals must actually happen, in one node."""
    from app import graph
    out = graph.knowledge_node(
        {"user_request": "payments-api is down again", "user_id": "u_1", "route": "knowledge"}
    )
    assert len(out["retrieved_docs"]) == 5           # external, scope=global
    assert len(out["prior_incidents"]) == 2          # episodic, scope=user:u_1
    assert all(d["chunk_id"].startswith("ep-") for d in out["prior_incidents"])


def test_every_write_goes_through_the_scope_helper():
    """w10v02: 'every memory write goes through a small helper that takes a
    scope argument'. The helper asserts the scope; it does not assume it."""
    from app import memory
    with pytest.raises(AssertionError, match="scope filter required"):
        memory.memory_write("", "procedural", {"k": "v"})
    with pytest.raises(ValueError, match="unknown memory kind"):
        memory.memory_write("user:u_1", "nonsense", {})
    with pytest.raises(ValueError, match="expected a user-scoped write"):
        memory.memory_write("global", "episodic", {"text": "x"}, thread_id="t1")

    memory.memory_write("user:u_7", "procedural", {"preferred_response_length": "concise"})
    assert memory.get_prefs("u_7") == {"preferred_response_length": "concise"}


def test_procedural_rules_reach_the_system_prompt():
    """set_pref had zero callers, so prefs were always empty and no rule was
    ever injected. Seeded prefs must now show up in the composed prompt."""
    from prompts import TRIAGE_BASE, compose_system_prompt
    prompt = compose_system_prompt("u_1", TRIAGE_BASE)
    assert "User preferences (procedural memory)" in prompt
    assert "escalate billing" in prompt.lower()
    assert "concise" in prompt.lower()
    assert "scale_service" in prompt
    # A user with no prefs row gets the base prompt back, untouched.
    assert compose_system_prompt("u_nobody", TRIAGE_BASE) == TRIAGE_BASE


def test_gdpr_delete_returns_a_counter_per_destination(stub_llm, stub_redis):
    """w10v02: 'nuke their session keys in Redis, delete their episodic
    summaries, and clear their procedural preferences row.'

    Five counters, not four: the checkpoint table is a destination too, and a
    receipt that omits a destination is a receipt that proves less than it says.
    That fifth counter has its own test below."""
    from app.main import app
    with TestClient(app) as client:
        r = client.delete("/user/u_1", headers=as_user("u_1"))
    assert r.status_code == 200
    body = r.json()
    assert body["episodic_rows_deleted"] == 2      # both prior incidents, gone
    assert body["procedural_rows_deleted"] == 1    # the one prefs row
    assert body["external_rows_deleted"] == 0      # u_1 owns no external rows
    assert "session_keys_deleted" in body
    assert "checkpoint_threads_deleted" in body

    # ...and the deletion actually happened.
    from app.memory import get_prefs, search_episodic, search_runbooks
    assert search_episodic("payments", user_id="u_1", k=3) == []
    assert get_prefs("u_1") == {}
    # The global runbook corpus is scope=global, so a user delete cannot touch it.
    assert len(search_runbooks("payments", k=5)) == 5


# ======================================================================
# Authorization. Week 10's argument is that scope is the safety property, and
# these are the tests that make the package practise it. Each one names a path
# that was open in the shipped version.
# ======================================================================

def test_a_guessed_thread_id_cannot_resume_another_users_thread(stub_llm, stub_redis):
    """The worst of the four. A thread id is a client-supplied field on /triage,
    so naming somebody else's paused thread used to drop you into their run - and
    from there their pending approval is one POST away."""
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "Please restart payments-api now",
                                         "user_id": "u_victim"}, headers=as_user("u_victim"))
        victim_thread = r.json()["thread_id"]
        r2 = client.post("/triage", json={"user_request": "anything at all",
                                          "user_id": "u_attacker",
                                          "thread_id": victim_thread},
                         headers=as_user("u_attacker"))
    assert r2.status_code == 403


def test_you_cannot_run_the_graph_as_somebody_else(stub_llm, stub_redis):
    """user_id is a body field. A body field is not an identity."""
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "How do I restart the payments API?",
                                         "user_id": "u_victim"}, headers=as_user("u_attacker"))
    assert r.status_code == 403


def test_a_thread_id_does_not_unlock_a_users_memory(stub_llm, stub_redis):
    """GET /session returned a user's entire cross-thread memory to anyone holding
    a thread id, and the UI prints thread ids on screen.

    Note the 404: a 403 would confirm the thread exists, which is the one bit an
    enumeration attack wants.
    """
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "How do I restart the payments API?",
                                         "user_id": "u_victim"}, headers=as_user("u_victim"))
        tid = r.json()["thread_id"]
        mine = client.get(f"/session/{tid}", headers=as_user("u_victim"))
        theirs = client.get(f"/session/{tid}", headers=as_user("u_attacker"))
    assert mine.status_code == 200
    assert theirs.status_code == 404


def test_only_an_approver_can_approve(stub_llm, stub_redis):
    """Approval is a ROLE, not something the requester grants themselves."""
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "Please restart payments-api now",
                                         "user_id": "u_1"}, headers=as_user("u_1"))
        tid = r.json()["thread_id"]
        r2 = client.post("/approve", json={"thread_id": tid, "approved": True,
                                           "reviewer_id": "u_1"}, headers=as_user("u_1"))
    assert r2.status_code == 403


def test_an_approver_cannot_sign_somebody_elses_name(stub_llm, stub_redis):
    """A reviewer_id anybody can set to anybody is a name in a log, not a record
    of who decided."""
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "Please restart payments-api now",
                                         "user_id": "u_1"}, headers=as_user("u_1"))
        tid = r.json()["thread_id"]
        r2 = client.post("/approve", json={"thread_id": tid, "approved": True,
                                           "reviewer_id": "someone_else"},
                         headers=as_user("ops_lead"))
    assert r2.status_code == 403


def test_approving_does_not_hand_over_the_requesters_prompts(stub_llm, stub_redis):
    """The approver is authorised to DECIDE. llm_calls is a verbatim copy of what
    another person typed, and deciding is not a claim on it."""
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "Please restart payments-api now",
                                         "user_id": "u_1"}, headers=as_user("u_1"))
        tid = r.json()["thread_id"]
        r2 = client.post("/approve", json={"thread_id": tid, "approved": True,
                                           "reviewer_id": "ops_lead"}, headers=as_user("ops_lead"))
    assert r2.status_code == 200
    assert r2.json()["llm_calls"] == [], "the approver received the requester's prompt log"


def test_you_cannot_write_another_users_preferences(stub_llm, stub_redis):
    """Procedural memory shapes what the model is told to do. A preference
    somebody else can set for you is a prompt injection with a REST endpoint."""
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/prefs", json={"user_id": "u_1", "key": "blocked_tools", "value": []},
                        headers=as_user("u_attacker"))
    assert r.status_code == 403


def test_you_cannot_erase_another_users_data(stub_llm, stub_redis):
    """An unauthenticated DELETE /user/{uid} is a denial-of-service endpoint with
    a receipt."""
    from app.main import app
    with TestClient(app) as client:
        r = client.delete("/user/u_1", headers=as_user("u_attacker"))
    assert r.status_code == 403


def test_every_data_path_requires_a_caller(stub_llm, stub_redis):
    """No identity, no data. /health is the only exception, and it returns none."""
    from app.main import app
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/triage", json={"user_request": "hi", "user_id": "u_1"}).status_code == 401
        assert client.get("/session/th_anything").status_code == 401
        assert client.delete("/user/u_1").status_code == 401
        assert client.post("/prefs", json={"user_id": "u_1", "key": "k",
                                           "value": "v"}).status_code == 401


def test_cors_is_not_a_wildcard(stub_llm, stub_redis):
    """A wildcard origin on a server exposing DELETE /user/{uid} and /approve
    hands both to any page the user happens to have open."""
    from app.config import get_settings
    origins = get_settings().allowed_origins
    assert "*" not in origins
    assert origins.startswith("http")


def test_erasure_reaches_the_ownership_map(stub_llm, stub_redis):
    """Which threads a person started is a record about that person."""
    from app import authz
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/triage", json={"user_request": "How do I restart the payments API?",
                                         "user_id": "u_gone"}, headers=as_user("u_gone"))
        tid = r.json()["thread_id"]
        assert authz.owner_of(tid) == "u_gone"
        client.delete("/user/u_gone", headers=as_user("u_gone"))
    assert authz.owner_of(tid) is None


def test_prefs_endpoint_writes_procedural_memory(stub_llm, stub_redis):
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/prefs", json={"user_id": "u_3", "key": "preferred_response_length",
                                        "value": "concise"}, headers=as_user("u_3"))
    assert r.status_code == 200
    assert r.json()["prefs"]["preferred_response_length"] == "concise"


# ======================================================================
# Added with the Week 10 gate rewrite. These pin the properties that the
# earlier suite did not: a whole gate was replaced and nineteen tests
# passed without noticing, which is its own lesson.
# ======================================================================

def test_risky_is_a_pure_function_of_the_proposal():
    """Resuming re-runs the node from the top, so the verdict is computed twice.

    A risky() that can answer differently on re-entry is a gate that can be
    walked around: pause, decline, resume, and the policy says no approval was
    needed. Ten calls, one answer.
    """
    from app.risk import risky
    for proposal in ({"tool_name": "restart_service", "arguments": {"service_name": "payments"}},
                     {"tool_name": "file_ticket", "arguments": {"title": "x"}},
                     {"tool_name": "none", "arguments": {}}):
        verdicts = {risky(dict(proposal)) for _ in range(10)}
        assert len(verdicts) == 1, f"risky() gave two answers for {proposal['tool_name']!r}"


def test_a_low_risk_proposal_does_not_wake_anybody():
    """The gate is CONDITIONAL, which is the whole reason it is a dynamic
    interrupt rather than a compile-time one. Filing a ticket is a note to
    somebody, not a change to anything, and it must not pause."""
    from app.risk import risky
    assert risky({"tool_name": "file_ticket", "arguments": {"title": "printer jam"}}) is False
    assert risky({"tool_name": "none", "arguments": {}}) is False
    assert risky({"tool_name": "restart_service", "arguments": {"service_name": "payments"}}) is True


def test_the_router_emits_route_literals_and_never_a_node_name():
    """`compose` is a node name. Returning it here would look right and be dead,
    because the edge map's keys are what the ROUTER can emit."""
    from app.graph import route_after_triage
    assert route_after_triage({"route": "knowledge"}) == "knowledge"
    assert route_after_triage({"route": "action"}) == "action"
    assert route_after_triage({"route": "escalate_human"}) == "escalate_human"
    assert route_after_triage({}) == "escalate_human"
    assert route_after_triage({"route": "compose"}) == "escalate_human"


def test_the_edge_map_keys_are_exactly_the_route_literals():
    """The two vocabularies must not collapse. Keys are router outputs; values
    are node names. They coincide on two of three rows, which is the trap."""
    import inspect

    from app import graph as g
    src = inspect.getsource(g.build_graph)
    assert '"escalate_human": "compose"' in src, \
        "the escalate branch must be reachable by the literal the router emits"
    assert '"compose": "compose"' not in src, \
        "'compose' is a node name, not something the router can emit. Dead key."


def test_an_off_script_route_falls_back_to_the_safe_branch(stub_llm, stub_redis, monkeypatch):
    """Not the cheapest branch. The least consequential one for this risk model."""
    import app.graph as g
    monkeypatch.setattr(g, "call", lambda **kw: _Result("reboot-the-datacentre"))
    out = g.triage_node({"user_request": "something odd", "user_id": "u_1"})
    assert out["route"] == "escalate_human"


class _Result:
    def __init__(self, text):
        self.text, self.model, self.latency_ms = text, "stub", 1
        self.input_tokens = self.output_tokens = 1


def test_assert_handoff_accepts_a_correctly_written_false():
    """Presence, not truthiness. A proposal that needs no approval writes False,
    and a truthiness test would blame the node that got it right."""
    from app.state import assert_handoff
    assert_handoff({"proposed_action": {"tool_name": "none"}, "pending_approval": False},
                   ["proposed_action", "pending_approval"], "action_execute")


def test_the_receipt_counts_checkpoints_too(stub_llm, stub_redis):
    """Week 9 handed this destination over by name: a deletion path that clears
    the memory index and leaves the checkpoint table has not deleted anything."""
    from fastapi.testclient import TestClient

    from app.main import app as _app
    with TestClient(_app) as c:
        # Three threads, because one thread cannot tell a walk that stops early
        # from a walk that finishes.
        for _ in range(3):
            c.post("/triage", json={"user_request": "how do I restart the payments API?",
                                    "user_id": "u_del"}, headers=as_user("u_del"))
        body = c.delete("/user/u_del", headers=as_user("u_del")).json()
    assert "checkpoint_threads_deleted" in body, "the receipt is missing a destination"
    assert body["checkpoint_threads_deleted"] == 3, \
        f"three threads went in, the receipt says {body['checkpoint_threads_deleted']}"

    # The counter agreeing with itself is not the property that matters. What
    # matters is that nothing is LEFT, and an earlier version of this deletion
    # passed a >= 1 assertion while leaving four of five threads in place.
    from app.main import _checkpointer
    remaining = {
        (t.config or {}).get("configurable", {}).get("thread_id")
        for t in _checkpointer.list(None)
        if (t.checkpoint.get("channel_values") or {}).get("user_id") == "u_del"
    }
    assert not remaining, f"erasure left {len(remaining)} of this user's threads behind"
