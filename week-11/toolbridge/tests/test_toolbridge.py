"""ToolBridge test suite. No network, no model, no live peer.

Numbered by the TASK each group proves, so a failing test names the task you
have not finished yet.
"""
from __future__ import annotations

import asyncio
import copy
import json

import pytest

from app import tools
from app.config import get_settings
from app.descriptions import BAD, GOOD, describe
from app.mcp_server import list_tool_infos
from mcp.server.mcpserver.exceptions import ToolError

from tests.conftest import GOOD_CARD, script


def err(exc: ToolError) -> dict:
    """Pull the structured body back out of a ToolError message."""
    from app.errors import ENVELOPE
    text = str(exc)
    return json.loads(text.rsplit(ENVELOPE, 1)[1])


# ---------------------------------------------------------------- TASK 1
def test_every_tool_has_a_description_in_both_sets():
    infos = asyncio.run(list_tool_infos())
    names = {t["name"] for t in infos}
    assert names == {"peer_card", "delegate_triage", "resume_task"}
    assert names == set(GOOD) == set(BAD)


def test_good_descriptions_make_the_five_moves():
    """Verb first, scope, trigger, output shape, edge case. Checked crudely on
    purpose: a test cannot judge prose, but it can catch a description that
    never says when to call the tool or what comes back."""
    # An explicit list beats a clever heuristic. "Fetch" does not end in s, e or t.
    IMPERATIVES = {"Fetch", "Search", "Submit", "Create", "Cancel", "Answer",
                   "Resume", "Read", "List", "Send", "Get", "Delete", "Update"}
    for name, text in GOOD.items():
        first_word = text.split()[0]
        assert first_word in IMPERATIVES, \
            f"{name}: description should open with an imperative verb, got {first_word!r}"
        assert "Use " in text or "Use" in text.split(".")[1], f"{name}: no trigger condition stated"
        assert "Returns" in text, f"{name}: never says what comes back"
        assert text.count("Returns") >= 2, f"{name}: states no edge case (no second Returns)"
        assert len(text) > 200, f"{name}: too short to have said all five things"


def test_bad_descriptions_are_genuinely_worse_and_that_is_the_point():
    for name in BAD:
        assert len(BAD[name]) < 60
        assert "Use when" not in BAD[name]
    assert describe("peer_card", "good") != describe("peer_card", "bad")


# ---------------------------------------------------------------- TASK 2
def test_peer_card_reports_the_selected_interface_and_its_rank(peer):
    out = asyncio.run(tools.peer_card("http://peer.test"))
    assert out["agent"] == "agentmesh-stu_001"
    assert out["operator"] == "Alice"
    assert out["interface"]["protocolBinding"] == "HTTP+JSON"
    assert out["interface"]["rank"] == 0
    assert {s["id"] for s in out["skills"]} == {"triage_incident", "whoami"}


def test_peer_card_reads_past_a_binding_it_cannot_speak(peer):
    """supportedInterfaces is ORDERED BY PREFERENCE, not exclusive. A client
    that gives up at entry zero is throwing away a usable interface."""
    peer.card = copy.deepcopy(GOOD_CARD)
    peer.card["supportedInterfaces"] = [
        {"url": "grpc://peer.test:9000", "protocolBinding": "GRPC", "protocolVersion": "1.0"},
        {"url": "http://peer.test", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
    ]
    out = asyncio.run(tools.peer_card("http://peer.test"))
    assert out["interface"]["protocolBinding"] == "HTTP+JSON"
    assert out["interface"]["rank"] == 1, "must report that it used the fallback, not entry zero"


def test_peer_card_rejects_a_pre_v1_card_by_name(peer):
    """A card with a single `url` and no supportedInterfaces predates March 2026."""
    peer.card = copy.deepcopy(GOOD_CARD)
    del peer.card["supportedInterfaces"]
    peer.card["url"] = "http://peer.test"
    with pytest.raises(ToolError) as e:
        asyncio.run(tools.peer_card("http://peer.test"))
    body = err(e.value)
    assert body["code"] == "card_missing_interfaces"
    assert body["retryable"] is False


def test_peer_card_on_a_missing_card_is_not_retryable(peer):
    peer.card_status = 404
    with pytest.raises(ToolError) as e:
        asyncio.run(tools.peer_card("http://peer.test"))
    assert err(e.value)["retryable"] is False


# ---------------------------------------------------------------- TASK 3
def test_delegate_returns_completed_with_the_peer_result(peer):
    script(peer, ("TASK_STATE_SUBMITTED", {}), ("TASK_STATE_WORKING", {}),
           ("TASK_STATE_COMPLETED", {"answer": "restarted"}))
    out = asyncio.run(tools.delegate_triage("payments-api is down", "high", "u_alice", "http://peer.test"))
    assert out["status"] == "completed"
    assert out["result"]["answer"] == "restarted"
    # The exact body, including user_id. A2A carries no schema for a skill's
    # input, so the only guard against drifting away from the peer's contract is
    # a test that pins the body we actually send.
    assert peer.last_submit_json == {
        "skill": "triage_incident",
        "input": {"description": "payments-api is down", "severity": "high", "user_id": "u_alice"},
    }


def test_delegate_sends_the_bearer_on_the_stream_too(peer):
    script(peer, ("TASK_STATE_COMPLETED", {}))
    asyncio.run(tools.delegate_triage("the checkout queue is backing up", "low", "u_alice", "http://peer.test"))
    assert peer.last_headers.get("authorization") == "Bearer test-token"


def test_delegate_reattaches_rather_than_resubmitting(peer):
    """A dropped stream on a long task is normal. Two legs, ONE submit."""
    script(peer, ("TASK_STATE_SUBMITTED", {}), ("TASK_STATE_WORKING", {}),
           ("TASK_STATE_COMPLETED", {"answer": "done"}))
    peer.drop_after = 2                                   # the connection dies mid-task
    peer.status_snapshots = [{"state": "TASK_STATE_WORKING", "payload": {}}]
    out = asyncio.run(tools.delegate_triage("the checkout queue is backing up", "low", "u_alice", "http://peer.test"))
    assert out["status"] == "completed"
    submits = [c for c in peer.calls if c == ("POST", "/tasks")]
    assert len(submits) == 1, "a dropped stream must not cause a resubmit"


# ---------------------------------------------------------------- TASK 4
def test_a_pause_is_not_an_error(peer):
    """INPUT_REQUIRED is INTERRUPTED, not terminal. The task is alive on the
    peer and the caller needs a way back to it."""
    script(peer, ("TASK_STATE_SUBMITTED", {}),
           ("TASK_STATE_INPUT_REQUIRED", {"proposal": {"action": "restart payments-api"}}))
    out = asyncio.run(tools.delegate_triage("restart payments", "high", "u_alice", "http://peer.test"))
    assert out["status"] == "pending_approval"
    assert out["task_id"] == "tk_test"
    assert out["proposal"]["action"] == "restart payments-api"
    assert "resume_task" in out["next"]


def test_resume_approves_and_settles(peer):
    script(peer, ("TASK_STATE_WORKING", {}), ("TASK_STATE_COMPLETED", {"answer": "restarted"}))
    out = asyncio.run(tools.resume_task("tk_test", "approve", "u_772", peer_url="http://peer.test"))
    assert out["status"] == "completed"
    assert out["decision"] == "approve"
    assert out["reviewer_id"] == "u_772"
    assert peer.last_reply_json["approved"] is True


def test_resume_rejects_and_the_peer_executes_nothing(peer):
    script(peer, ("TASK_STATE_REJECTED", {"message": "declined by reviewer"}))
    with pytest.raises(ToolError) as e:
        asyncio.run(tools.resume_task("tk_test", "reject", "u_772", peer_url="http://peer.test"))
    assert peer.last_reply_json["approved"] is False
    assert err(e.value)["retryable"] is False


def test_resume_from_zero_replays_the_pause_it_just_answered(peer):
    """THE REGRESSION TEST. This is the bug the first live run found.

    A stream replays every event with an id greater than the cursor presented.
    Resume from zero after answering a gate and the peer faithfully replays the
    INPUT_REQUIRED event, the bridge reads it as the current state, and reports
    a pending approval for a task that has already completed. The caller then
    approves it again and gets a 409.

    The mocked tests did not catch this until the fake peer was made cursor-
    faithful, which is the real lesson: the fake has to be strict on exactly the
    dimension the bug lives on."""
    script(peer,
           ("TASK_STATE_SUBMITTED", {}),
           ("TASK_STATE_WORKING", {}),
           ("TASK_STATE_INPUT_REQUIRED", {"proposal": {"action": "restart"}}),
           ("TASK_STATE_WORKING", {}),
           ("TASK_STATE_COMPLETED", {"answer": "restarted"}))

    # Carrying the cursor: correct.
    good = asyncio.run(tools.resume_task("tk_test", "approve", "u_772",
                                         resume_from=3, peer_url="http://peer.test"))
    assert good["status"] == "completed"

    # Starting from zero: the pause comes back and we misreport a finished task.
    stale = asyncio.run(tools.resume_task("tk_test", "approve", "u_772",
                                          resume_from=0, peer_url="http://peer.test"))
    assert stale["status"] == "pending_approval", (
        "with no cursor the replayed pause is what you see. If this assertion "
        "ever fails, follow() has started ignoring from_event_id.")


def test_delegate_hands_back_the_cursor_resume_needs(peer):
    """The handle pattern: we are stateless between the two calls, so the value
    the next call needs has to travel through the caller."""
    script(peer, ("TASK_STATE_SUBMITTED", {}), ("TASK_STATE_WORKING", {}),
           ("TASK_STATE_INPUT_REQUIRED", {"proposal": {"action": "restart"}}))
    out = asyncio.run(tools.delegate_triage("restart the payments service", "high",
                                            "u_alice", "http://peer.test"))
    assert out["status"] == "pending_approval"
    assert out["resume_from"] == 3, "must be the id of the pause event, not zero"
    assert "resume_from" in out["next"]


def test_resume_requires_a_named_reviewer():
    """An approval with no named approver is not an audit trail."""
    with pytest.raises(ToolError) as e:
        asyncio.run(tools.resume_task("tk_test", "approve", "   "))
    assert err(e.value)["code"] == "missing_reviewer"


def test_resume_rejects_a_decision_that_is_not_approve_or_reject():
    with pytest.raises(ToolError) as e:
        asyncio.run(tools.resume_task("tk_test", "maybe", "u_772"))
    assert err(e.value)["code"] == "invalid_decision"


# ---------------------------------------------------------------- TASK 5
def test_rejected_is_terminal_and_never_retryable(peer):
    script(peer, ("TASK_STATE_REJECTED", {"message": "not during the freeze"}))
    with pytest.raises(ToolError) as e:
        asyncio.run(tools.delegate_triage("restart everything", "high", "u_alice", "http://peer.test"))
    body = err(e.value)
    assert body["code"] == "rejected"
    assert body["retryable"] is False, "REJECTED means the peer declined; retrying gets the same answer"


def test_failed_is_terminal_and_worth_one_retry(peer):
    script(peer, ("TASK_STATE_FAILED", {"message": "upstream timed out"}))
    with pytest.raises(ToolError) as e:
        asyncio.run(tools.delegate_triage("the checkout queue is backing up", "low", "u_alice", "http://peer.test"))
    body = err(e.value)
    assert body["code"] == "failed"
    assert body["retryable"] is True


def test_a_401_from_the_peer_is_not_retryable(peer):
    peer.submit_status = 401
    with pytest.raises(ToolError) as e:
        asyncio.run(tools.delegate_triage("the checkout queue is backing up", "low", "u_alice", "http://peer.test"))
    body = err(e.value)
    assert body["code"] == "peer_unauthorized"
    assert body["retryable"] is False
    assert body["peer_status"] == 401


def test_a_422_from_the_peer_is_not_retryable(peer):
    """A shape the peer rejects that we could not have known was wrong.

    A2A carries no schema for a skill's input, so this WILL happen to you the
    first time you meet an unfamiliar peer. The test forces it at the transport
    rather than the payload, because the payload checks below now catch the
    shapes we CAN know about locally."""
    peer.submit_status = 422
    with pytest.raises(ToolError) as e:
        asyncio.run(tools.delegate_triage("a genuine incident description", "low", "u_alice",
                                          "http://peer.test"))
    body = err(e.value)
    assert body["code"] == "invalid_input"
    assert body["retryable"] is False


def test_the_input_shape_is_checked_locally_before_the_peer_is_dialled(peer):
    """Three fields, three local checks, three sentences a model can act on.

    This exists because the peer's TriageInput requires description (min 10),
    severity (an enum) and user_id, and NONE of that is discoverable from its
    Agent Card. Failing here with a sentence beats failing there with a pydantic
    dump, and it means a bad call never creates a task somebody has to clean up."""
    cases = [
        (("short", "low", "u_alice"), "description_too_short"),
        (("a genuine incident description", "catastrophic", "u_alice"), "invalid_severity"),
        (("a genuine incident description", "low", "   "), "missing_user_id"),
    ]
    for args, code in cases:
        with pytest.raises(ToolError) as e:
            asyncio.run(tools.delegate_triage(*args, "http://peer.test"))
        assert err(e.value)["code"] == code
    assert not [c for c in peer.calls if c == ("POST", "/tasks")], \
        "a locally-invalid call must never reach the peer"


def test_replying_to_a_task_that_is_not_paused_is_a_409_and_says_so(peer):
    peer.reply_status = 409
    with pytest.raises(ToolError) as e:
        asyncio.run(tools.resume_task("tk_test", "approve", "u_772", peer_url="http://peer.test"))
    body = err(e.value)
    assert body["code"] == "task_not_paused"
    assert body["retryable"] is False
    assert "terminal" in body["message"]


def test_every_failure_carries_a_code_and_a_retryable_flag(peer):
    """The convention, asserted once over every path that can fail, because a
    caller reading five different failure shapes ends up reading none of them."""
    cases = []
    peer.card_status = 500
    cases.append(lambda: tools.peer_card("http://peer.test"))
    for make in cases:
        with pytest.raises(ToolError) as e:
            asyncio.run(make())
        body = err(e.value)
        assert set(body) >= {"code", "message", "retryable"}
        assert isinstance(body["retryable"], bool)


# ---------------------------------------------------------------- TASK 6
def test_the_entrypoint_is_declared():
    """`pip install .` has to produce a `toolbridge` command, or nobody can
    consume this server without your source tree."""
    text = open("pyproject.toml").read()
    assert "[project.scripts]" in text
    assert 'toolbridge = "app.mcp_server:main"' in text


def test_fixture_data_ships_inside_the_package():
    """A wheel that installs cleanly and then cannot find its own golden set is
    a wheel that passes the install check and fails the first run."""
    text = open("pyproject.toml").read()
    assert "package-data" in text and "data/*.json" in text


# ---------------------------------------------------------------- the SDK era
def test_this_project_is_on_the_stateless_sdk():
    """The two guided builds pin mcp 1.x and speak 2025-11-25. This one pins
    2.x and speaks 2026-07-28. If this test fails you are in the wrong venv."""
    from mcp_types.version import MODERN_PROTOCOL_VERSIONS
    assert "2026-07-28" in MODERN_PROTOCOL_VERSIONS
    with pytest.raises(ModuleNotFoundError):
        import mcp.server.fastmcp  # noqa: F401
