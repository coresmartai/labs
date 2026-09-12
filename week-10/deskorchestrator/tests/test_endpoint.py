"""These pass on a fresh clone. They pin what the scaffold already does."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(stub_llm):
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------ shape
def test_health_names_one_model_per_agent(client):
    r = client.get("/health")
    assert r.status_code == 200
    models = r.json()["models"]
    assert set(models) == {"triage", "access", "software", "hardware"}
    assert models["triage"] != models["access"], "the router should not share the specialist's pin"


def test_missing_user_id_is_rejected_before_a_token_is_spent(client):
    assert client.post("/desk", json={"user_request": "hello"}).status_code == 422


def test_an_access_request_pauses_at_the_gate(client):
    r = client.post("/desk", json={"user_request": "I need write access to billing-db",
                                   "user_id": "u_1"})
    body = r.json()
    assert body["status"] == "pending_approval"
    assert body["route"] == "access"
    assert body["proposed_action"]["tool_name"] == "grant_access"
    assert body["final_answer"] is None, "nothing should be composed before the gate resolves"


def test_approving_resumes_and_composes(client):
    tid = client.post("/desk", json={"user_request": "I need write access to billing-db",
                                     "user_id": "u_1"}).json()["thread_id"]
    r = client.post("/approve", json={"thread_id": tid, "approved": True, "reviewer_id": "mgr_7"})
    body = r.json()
    assert body["status"] == "completed"
    assert "Done." in body["final_answer"]


def test_rejecting_resumes_without_executing(client):
    tid = client.post("/desk", json={"user_request": "I need write access to billing-db",
                                     "user_id": "u_1"}).json()["thread_id"]
    body = client.post("/approve", json={"thread_id": tid, "approved": False,
                                         "reviewer_id": "mgr_7"}).json()
    assert body["status"] == "completed"
    assert "declined by approver" in body["final_answer"]


def test_approving_a_thread_that_is_not_paused_is_a_409(client):
    assert client.post("/approve", json={"thread_id": "th_nope", "approved": True,
                                         "reviewer_id": "mgr_7"}).status_code == 409


def test_double_approval_is_a_409(client):
    tid = client.post("/desk", json={"user_request": "I need write access to billing-db",
                                     "user_id": "u_1"}).json()["thread_id"]
    client.post("/approve", json={"thread_id": tid, "approved": True, "reviewer_id": "mgr_7"})
    assert client.post("/approve", json={"thread_id": tid, "approved": True,
                                         "reviewer_id": "mgr_7"}).status_code == 409


# -------------------------------------------------------------- the graph
def test_escalation_skips_both_workers_and_the_gate(routed_llm):
    routed_llm["route"] = "escalate_human"
    with TestClient(app) as c:
        body = c.post("/desk", json={"user_request": "I am being investigated by HR",
                                     "user_id": "u_1"}).json()
    assert body["status"] == "completed"
    assert body["route"] == "escalate_human"
    assert body["proposed_action"] is None
    assert len(body["llm_calls"]) == 1, "escalation should cost one classifier call and nothing else"


def test_an_off_script_router_falls_back_to_escalation(routed_llm):
    routed_llm["route"] = "reboot-the-datacentre"
    with TestClient(app) as c:
        body = c.post("/desk", json={"user_request": "something odd", "user_id": "u_1"}).json()
    assert body["route"] == "escalate_human", "an unrecognised route must fail to the safe branch"


def test_compose_is_the_only_writer_of_final_answer(stub_llm):
    from app.graph import access_node
    out = access_node({"user_request": "write access to billing-db", "user_id": "u_1",
                       "route": "access"})
    assert "final_answer" not in out
    assert out["specialist_summary"]


def test_assert_handoff_names_the_node_and_the_field():
    from app.state import assert_handoff
    with pytest.raises(ValueError, match=r"handoff to execute missing fields: pending_approval"):
        assert_handoff({"proposed_action": {}}, ["proposed_action", "pending_approval"], "execute")


def test_assert_handoff_accepts_a_written_false():
    """Presence, not truthiness. A low-risk proposal writes pending_approval=False."""
    from app.state import assert_handoff
    assert_handoff({"proposed_action": {}, "pending_approval": False},
                   ["proposed_action", "pending_approval"], "execute")


def test_llm_calls_accumulates_across_nodes(client):
    body = client.post("/desk", json={"user_request": "I need write access to billing-db",
                                      "user_id": "u_1"}).json()
    nodes = [c["node"] for c in body["llm_calls"]]
    assert nodes == ["triage", "access"], f"expected one entry per node, got {nodes}"


# ------------------------------------------------------------- the tools
def test_execute_tool_never_raises():
    from app.tools import execute_tool
    for name, args in [("grant_access", None), ("nope", {}), ("order_hardware", {"item": 1, "x": 2})]:
        out = execute_tool(name, args)
        assert isinstance(out, dict) and "success" in out


# ------------------------------------------------------------- the memory
def test_every_write_goes_through_the_scoped_helper():
    import pytest as _p
    from app.memory import memory_write
    with _p.raises(AssertionError):
        memory_write("", "procedural", {"key": "k", "value": 1})
    with _p.raises(ValueError):
        memory_write("user:u_1", "nonsense", {})
    with _p.raises(ValueError):
        memory_write("global", "episodic", {"chunks": []})
    with _p.raises(ValueError):
        memory_write("user:u_1", "external", {"chunks": []})


def test_procedural_rules_reach_the_system_prompt():
    from prompts import TRIAGE_BASE, compose_system_prompt
    composed = compose_system_prompt("u_1", TRIAGE_BASE)
    assert "three sentences or fewer" in composed
    assert "manager will be copied" in composed
    assert compose_system_prompt("u_nobody", TRIAGE_BASE) == TRIAGE_BASE


def test_deleting_a_user_empties_their_layers_and_spares_the_global_corpus(client):
    from app.memory import GLOBAL_SCOPE, POLICIES_COLL, get_prefs, search_tickets
    from app.vectorstore import get_store

    from app.main import _checkpointer as _cp

    # Three threads, deliberately. A run with nothing checkpointed makes the
    # counter assertion below compare zero to zero and prove nothing.
    for _ in range(3):
        assert client.post("/desk", json={"user_request": "my licence expired",
                                          "user_id": "u_1"}).status_code == 200

    before = get_store().count(GLOBAL_SCOPE, POLICIES_COLL)
    _threads_before = len({
        t.config["configurable"]["thread_id"] for t in _cp.list(None)
        if (t.config or {}).get("configurable", {}).get("thread_id")
        and (t.checkpoint.get("channel_values") or {}).get("user_id") == "u_1"
    })
    receipt = client.delete("/user/u_1").json()

    assert _threads_before >= 3, "the three requests above did not produce three threads"
    assert receipt["ticket_rows_deleted"] == 3
    assert receipt["procedural_rows_deleted"] == 1
    assert receipt["policy_rows_deleted"] == 0
    assert search_tickets("anything", user_id="u_1", k=5) == []
    assert get_prefs("u_1") == {}
    # The over-broad failure needs a TEST, not a counter: a zero counter is
    # evidence about what was deleted and says nothing about what was left.
    assert get_store().count(GLOBAL_SCOPE, POLICIES_COLL) == before, \
        "a user-scoped delete reached the shared policy corpus"

    # And the checkpoint counter counts THREADS. list() yields one entry per
    # checkpoint and a thread has several, so counting entries over-states the
    # erasure - which is its own kind of dishonest receipt.
    from app.main import _checkpointer
    # Not a fixed number: the checkpointer is a process global and earlier tests
    # leave threads in it. The property is that the receipt matches the threads
    # that were actually there. Counting list() entries instead of distinct
    # thread ids reports several times this figure.
    assert receipt["checkpoint_threads_deleted"] == _threads_before, (
        f"{_threads_before} threads were there, the receipt says "
        f"{receipt['checkpoint_threads_deleted']}")
    assert not [
        t for t in _checkpointer.list(None)
        if (t.checkpoint.get("channel_values") or {}).get("user_id") == "u_1"
    ], "erasure left this user's checkpoints behind"
