"""Smoke + contract tests for AgentMesh.

Offline, deterministic, sub-second. No OpenAI call, no real JWKS, no network -
the JWT fixture is a symmetric HS256 key primed straight into the JWKS cache.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time

import jwt
import pytest
from fastapi.testclient import TestClient

from app import jwt_verify, task_store
from app.config import get_settings
from app.llm import get_llm
from app.main import app
from app.mcp_server import cohort_lookup, incident_history
from app.schemas import (
    DRIVEN_STATES,
    INTERRUPTED_STATES,
    TERMINAL_STATES,
    TaskFailedPayload,
    TaskState,
)

# --- shared symmetric "test" key (HS256) used to sign + verify tokens in tests ---
_TEST_SECRET = "test-secret-key-not-for-production"
_TEST_SECRET_B64 = base64.urlsafe_b64encode(_TEST_SECRET.encode()).rstrip(b"=").decode()
_TEST_JWKS = {"keys": [{"kty": "oct", "alg": "HS256", "kid": "test-key", "k": _TEST_SECRET_B64}]}
_JWKS_URL = "https://test-idp.local/.well-known/jwks.json"

def _make_token(scope: str = "triage:invoke", aud: str = "agentmesh", ttl: int = 600) -> str:
    return jwt.encode(
        {
            "sub": "tester",
            "aud": aud,
            "iss": "https://test-idp.local",
            "scope": scope,
            "exp": int(time.time()) + ttl,
        },
        _TEST_SECRET,
        algorithm="HS256",
        headers={"kid": "test-key"},
    )


def _auth(scope: str = "triage:invoke") -> dict[str, str]:
    return {"Authorization": f"Bearer {_make_token(scope)}"}


@pytest.fixture(scope="module")
def client():
    """Context-managed on purpose: the `with` block keeps ONE event loop alive for the
    whole module, so the background task fired by POST /tasks is still running when the
    next request (the approval) arrives. Without it, every cross-request flow dies at
    the approval gate."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_state():
    task_store.reset_for_tests()
    jwt_verify.reset_jwks_cache_for_tests(_TEST_JWKS, _JWKS_URL)
    yield
    task_store.reset_for_tests()


def _submit(client: TestClient, description: str, severity: str) -> str:
    r = client.post(
        "/tasks",
        json={"skill": "triage_incident", "input": {
            "description": description, "severity": severity, "user_id": "u_42"}},
        headers=_auth(),
    )
    assert r.status_code == 202
    assert r.json()["state"] == "TASK_STATE_SUBMITTED"
    return r.json()["task_id"]


async def _wait_for(task_id: str, wanted: set[TaskState], tries: int = 60) -> TaskState | None:
    store = task_store.get_store()
    for _ in range(tries):
        state = store.get_state(task_id)
        if state in wanted:
            return state
        await asyncio.sleep(0.02)
    return store.get_state(task_id)


# ---------- 1-5 · the Agent Card and the protocol header ----------


def test_agent_card_is_well_formed(client):
    r = client.get("/.well-known/agent-card.json")
    assert r.status_code == 200
    card = r.json()
    # The card name carries the student id, so a cohort of twenty cards is twenty
    # distinguishable names rather than twenty copies of "agentmesh-triage".
    assert card["name"] == f"agentmesh-{get_settings().student_id}"
    assert card["version"] == "1.0.0"
    assert card["protocolVersion"] == "1.0"

    skills = {s["id"]: s for s in card["skills"]}
    assert {"triage_incident", "whoami"} <= set(skills)
    for skill in skills.values():
        for field in ("name", "description", "tags", "examples", "inputModes", "outputModes"):
            assert field in skill and skill[field]
    assert card["endpoints"]["tasks"] == "/tasks"
    assert card["securitySchemes"]["bearer"]["scheme"] == "bearer"


def test_agent_card_carries_student_identity(client):
    """A classmate has to learn WHOSE agent this is from the card alone - that is
    what `provider` is for in A2A."""
    s = get_settings()
    card = client.get("/.well-known/agent-card.json").json()
    assert card["provider"]["organization"] == s.student_name
    assert s.student_id in card["name"]
    assert s.student_name in card["description"]


def test_whoami_skill_completes_and_reports_the_caller(client):
    """The identity skill drives the real task machine - submitted -> working ->
    completed - so it doubles as proof that an authenticated round trip works."""
    r = client.post("/tasks", json={"skill": "whoami", "input": {}}, headers=_auth())
    assert r.status_code == 202
    task_id = r.json()["task_id"]

    final = asyncio.run(_wait_for(task_id, TERMINAL_STATES))
    assert final is TaskState.TASK_STATE_COMPLETED

    events = asyncio.run(_collect(task_id))
    payload = events[-1].payload
    s = get_settings()
    assert payload["student_id"] == s.student_id
    assert payload["student_name"] == s.student_name
    assert payload["agent_name"] == f"agentmesh-{s.student_id}"
    assert payload["caller"]  # the server echoes back who it thinks called


def test_unknown_skill_is_refused_before_any_task_is_created(client):
    r = client.post("/tasks", json={"skill": "not_a_skill", "input": {}}, headers=_auth())
    assert r.status_code == 400
    assert r.json()["detail"]["reason"] == "unknown_skill"


def test_agent_card_capabilities_are_protocol_flags_only(client):
    """`capabilities` is protocol flags. Skills live in skills[]. Latency and cost
    hints have no home in an A2A v1.0 card at all."""
    card = client.get("/.well-known/agent-card.json").json()
    caps = card["capabilities"]
    assert isinstance(caps, dict)
    # A2A v1.0 defines exactly four capability flags and no others.
    assert set(caps) <= {"streaming", "pushNotifications", "extensions", "extendedAgentCard"}
    # stateTransitionHistory was REMOVED in v1.0. A card still carrying it is stale.
    assert "stateTransitionHistory" not in caps

    blob = str(card)
    for banned in ("input_schema_ref", "output_schema_ref", "latency_p50_ms", "cost_hint"):
        assert banned not in blob


def test_legacy_agent_json_alias_returns_the_same_card(client):
    canonical = client.get("/.well-known/agent-card.json")
    legacy = client.get("/.well-known/agent.json")
    assert legacy.status_code == 200
    assert legacy.json() == canonical.json()


def test_agent_card_signatures_are_declared_and_unpopulated(client):
    """We do not sign the card and we do not verify anyone else's. The field is
    declared so the omission is visible, not hidden.

    Note the PLURAL. A2A v1.0 carries `signatures[]`, a list, because key rotation
    means a card may legitimately carry two valid signatures at once. A build that
    declares a single `signature` has copied a v0.x example."""
    card = client.get("/.well-known/agent-card.json").json()
    assert "signatures" in card
    assert card["signatures"] == []
    assert "signature" not in card


def test_agent_card_advertises_supported_interfaces(client):
    """v1.0 replaced the single `url` with `supportedInterfaces[]`. This is the
    field a conformant client reads to decide how to reach us, and it is ORDERED:
    entry zero is our preferred interface."""
    card = client.get("/.well-known/agent-card.json").json()
    ifaces = card["supportedInterfaces"]
    assert isinstance(ifaces, list) and len(ifaces) >= 1

    first = ifaces[0]
    for field in ("url", "protocolBinding", "protocolVersion"):
        assert field in first and first[field]
    # protocolBinding is an OPEN STRING, not an enum - but ours is one of the three
    # the specification names officially.
    assert first["protocolBinding"] in {"JSONRPC", "GRPC", "HTTP+JSON"}
    assert first["protocolVersion"] == get_settings().a2a_protocol_version


def test_legacy_url_field_agrees_with_the_preferred_interface(client):
    """`url` is not a v1.0 field. We keep it so this build's own UI and client
    script keep working, which means it MUST NOT be allowed to drift away from
    supportedInterfaces[0] - a card that advertises two different addresses is
    worse than one that advertises an outdated shape."""
    card = client.get("/.well-known/agent-card.json").json()
    assert card["url"] == card["supportedInterfaces"][0]["url"]


def test_a2a_version_header_on_every_response(client):
    version = get_settings().a2a_protocol_version
    assert client.get("/.well-known/agent-card.json").headers["A2A-Version"] == version
    assert client.get("/health").headers["A2A-Version"] == version

    ok = client.post(
        "/tasks",
        json={"skill": "triage_incident", "input": {
            "description": "Where is the runbook for payments-api?", "severity": "low", "user_id": "u_42"}},
        headers=_auth(),
    )
    assert ok.status_code == 202
    assert ok.headers["A2A-Version"] == version

    unauthed = client.post("/tasks", json={"skill": "triage_incident", "input": {}})
    assert unauthed.status_code == 401
    assert unauthed.headers["A2A-Version"] == version


# ---------- 6-7 · the state machine and health ----------


def test_task_state_enum_is_the_specs_nine():
    assert len(TaskState) == 9
    assert {s.value for s in TaskState} == {
        "TASK_STATE_UNSPECIFIED", "TASK_STATE_SUBMITTED", "TASK_STATE_WORKING",
        "TASK_STATE_INPUT_REQUIRED", "TASK_STATE_AUTH_REQUIRED", "TASK_STATE_COMPLETED",
        "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED",
    }
    assert TERMINAL_STATES == {
        TaskState.TASK_STATE_COMPLETED, TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED, TaskState.TASK_STATE_REJECTED,
    }
    assert INTERRUPTED_STATES == {
        TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_AUTH_REQUIRED,
    }
    # Seven driven. The sentinel and the auth-at-the-door state are not among them.
    assert len(DRIVEN_STATES) == 7
    assert TaskState.TASK_STATE_UNSPECIFIED not in DRIVEN_STATES
    assert TaskState.TASK_STATE_AUTH_REQUIRED not in DRIVEN_STATES


def test_health_is_unauth_and_returns_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"].startswith("gpt-")
    assert body["models"]["triage"].startswith("gpt-")
    assert body["models"]["knowledge"].startswith("gpt-")


# ---------- 8-11 · the door, and the happy path ----------


def test_tasks_returns_401_without_authorization(client):
    r = client.post("/tasks", json={"skill": "triage_incident", "input": {}})
    assert r.status_code == 401
    assert r.json()["detail"]["reason"] == "missing_bearer"


def test_tasks_returns_403_with_wrong_scope(client):
    r = client.post(
        "/tasks",
        json={"skill": "triage_incident", "input": {}},
        headers=_auth(scope="other:scope"),
    )
    assert r.status_code == 403
    assert r.json()["detail"]["reason"] == "scope_missing"


def test_tasks_returns_422_on_malformed_input(client):
    r = client.post(
        "/tasks",
        json={"skill": "triage_incident", "input": {"user_id": "u1"}},
        headers=_auth(),
    )
    assert r.status_code == 422
    fields = {".".join(str(p) for p in err["loc"]) for err in r.json()["detail"]}
    assert any("description" in f for f in fields)
    assert any("severity" in f for f in fields)


def test_happy_path_knowledge_request_reaches_completed(client):
    task_id = _submit(client, "Where is the runbook for restarting payments-api?", "low")
    final = asyncio.run(_wait_for(task_id, TERMINAL_STATES))
    assert final is TaskState.TASK_STATE_COMPLETED


# ---------- 12 · Reject at the gate ----------


def test_reject_at_the_gate_drives_task_state_rejected(client):
    """The Reject button is the only way TASK_STATE_REJECTED is ever reachable - and
    the whole point is that nothing executes on the way there."""
    task_id = _submit(client, "Please restart payments-api now", "high")

    async def _drive() -> TaskState | None:
        await _wait_for(task_id, {TaskState.TASK_STATE_INPUT_REQUIRED})
        store = task_store.get_store()
        assert store.get_state(task_id) is TaskState.TASK_STATE_INPUT_REQUIRED
        r = client.post(
            f"/tasks/{task_id}/input",
            json={"approved": False, "note": "not during the freeze"},
            headers=_auth(),
        )
        assert r.status_code == 200
        return await _wait_for(task_id, TERMINAL_STATES)

    final = asyncio.run(_drive())
    assert final is TaskState.TASK_STATE_REJECTED

    events = asyncio.run(_collect(task_id))
    assert [e.state for e in events][-1] is TaskState.TASK_STATE_REJECTED
    # Nothing was executed: the "executing" progress event never happened.
    assert not any(e.payload.get("message") == "executing" for e in events)
    assert events[-1].payload["reason"] == "rejected_by_caller"


async def _collect(task_id: str, from_event_id: int = 0):
    store = task_store.get_store()
    return [e async for e in store.stream(task_id, from_event_id=from_event_id)]


# ---------- 13 · the replay bug that shipped ----------


def test_sse_event_ids_are_monotonic_past_the_buffer_wrap():
    """`event_id = len(buffer) + 1` stalls the moment the bounded deque wraps, and
    every Last-Event-ID replay after that silently returns the wrong events."""

    async def _drive():
        store = task_store.get_store()
        task_id = store.create()
        await store.update_state(task_id, TaskState.TASK_STATE_SUBMITTED, {"skill": "triage_incident"})
        for i in range(1, 130):
            await store.update_state(task_id, TaskState.TASK_STATE_WORKING, {"node": "triage", "i": i})

        buffered = await _collect_no_wait(store, task_id, 0)
        ids = [e.event_id for e in buffered if not e.payload.get("replay_gap")]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)   # strictly increasing
        assert ids[-1] == 130                                     # 1 SUBMITTED + 129 WORKING
        assert len(ids) == 100                                    # the buffer is bounded at 100

        # A live cursor: only the greater-id events come back.
        tail = await _collect_no_wait(store, task_id, 125)
        assert [e.event_id for e in tail] == [126, 127, 128, 129, 130]

        # A cursor older than the buffer: the gap is signalled, not silently swallowed.
        stale = await _collect_no_wait(store, task_id, 3)
        assert stale[0].payload["replay_gap"] is True
        assert stale[0].payload["oldest_available_event_id"] == 31
        assert [e.event_id for e in stale[1:]][:1] == [31]

    asyncio.run(_drive())


async def _collect_no_wait(store, task_id: str, from_event_id: int):
    """Drain only what is already buffered - `stream` subscribes for live events after
    the replay, so pull exactly the replayed ones and stop."""
    out = []
    agen = store.stream(task_id, from_event_id=from_event_id)
    while True:
        try:
            out.append(await asyncio.wait_for(agen.__anext__(), timeout=0.05))
        except (StopAsyncIteration, asyncio.TimeoutError):
            break
    await agen.aclose()
    return out


# ---------- 14 · the SDK seam ----------


def test_llm_client_routes_role_to_the_pinned_model():
    settings = get_settings()
    llm = get_llm()
    assert llm.model_for("triage") == settings.triage_model
    assert llm.model_for("knowledge") == settings.knowledge_model


# ---------- 15 · structured failures ----------


def test_structured_failures_409_and_task_failed_payload(client):
    task_id = _submit(client, "Where is the runbook for restarting payments-api?", "low")
    asyncio.run(_wait_for(task_id, TERMINAL_STATES))

    r = client.post(f"/tasks/{task_id}/input", json={"approved": True}, headers=_auth())
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "task_not_paused"
    assert r.json()["detail"]["current_state"] == "TASK_STATE_COMPLETED"

    assert TaskFailedPayload(reason="input_timeout", evidence="no approval reply in 300s").retryable is False
    assert TaskFailedPayload(reason="internal_error", evidence="boom", retryable=True).retryable is True


# ---------- 16-17 · the second surface: the cohort MCP tools ----------


def test_mcp_cohort_lookup_known_and_unknown():
    assert cohort_lookup("stu_001")["found"] is True
    assert cohort_lookup("stu_001")["profile"]["name"] == "Alice"
    assert cohort_lookup("stu_999") == {"found": False}


def test_mcp_whoami_and_ping_report_this_student():
    """The identity tool and the connectivity tool, called directly - no server."""
    from app.mcp_server import ping, whoami

    s = get_settings()
    me = whoami()
    assert me["student_id"] == s.student_id
    assert me["student_name"] == s.student_name
    assert "A2A/1.0" in me["protocols"]

    echoed = ping("hello")
    assert echoed["echo"] == "hello"
    assert echoed["from"] == s.student_id


def test_mcp_surface_registers_tools_resources_and_prompts():
    """MCP is not a synonym for tool calling: this server ships all three
    primitives, and a regression that silently drops one should fail here."""
    from app.mcp_server import mcp

    tools = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"whoami", "cohort_lookup", "incident_history", "ping"} <= tools

    resources = {str(r.uri) for r in asyncio.run(mcp.list_resources())}
    assert {"agentmesh://student/profile", "agentmesh://incidents"} <= resources

    templates = {t.uriTemplate for t in asyncio.run(mcp.list_resource_templates())}
    assert "agentmesh://incident/{incident_id}" in templates

    prompts = {p.name for p in asyncio.run(mcp.list_prompts())}
    assert {"triage_brief", "peer_intro"} <= prompts


def test_cohosted_mcp_mount_is_bearer_gated(client):
    """`app.mount()` bypasses FastAPI dependencies, so the MCP surface is guarded in
    middleware instead. If that guard regresses, co-hosting publishes an
    unauthenticated tool surface on the same public tunnel as the gated one."""
    if not get_settings().cohost_mcp:
        pytest.skip("co-hosting disabled")
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "1"}}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    unauthed = client.post("/mcp", json=body, headers=headers)
    assert unauthed.status_code == 401
    assert unauthed.json()["detail"]["reason"] == "missing_bearer"

    wrong = client.post("/mcp", json=body, headers={**headers, "Authorization": "Bearer nope"})
    assert wrong.status_code in (401, 403)


def test_mcp_incident_history_returns_structured_payload():
    out = incident_history("payments")
    assert "results" in out and "total_available" in out
    assert all({"id", "severity", "summary"}.issubset(r.keys()) for r in out["results"])
    out_empty = incident_history("xyznonexistent")
    assert out_empty["results"] == []
    assert out_empty["total_available"] == 0


# ---------- 18-19 · the A2A client's two remote-cohort guards ----------


def _card(url: str, skill: str = "triage_incident") -> dict:
    return {
        "protocolVersion": "1.0",
        "url": url,
        "skills": [{"id": skill}],
        "endpoints": {"base": url},
    }


def test_a2a_client_card_validation_catches_wrong_skill_and_localhost_trap():
    """Both failures a peer can hand you: a card that cannot do what you want, and a
    card fetched over a tunnel that still advertises localhost."""
    from scripts.a2a_client import validate_card

    good = _card("https://alice.example.com")
    assert validate_card(good, "triage_incident", "https://alice.example.com") == []

    wrong_skill = validate_card(good, "summarise", "https://alice.example.com")
    assert any("does not offer skill" in p for p in wrong_skill)

    # The trap: dialled a public URL, card advertises localhost -> peer never set
    # AGENTMESH_BASE_URL, so every URL they publish points back at the caller.
    trap = validate_card(_card("http://localhost:8000"), "triage_incident", "https://alice.example.com")
    assert any("AGENTMESH_BASE_URL" in p for p in trap)

    # Same card is fine when you genuinely dialled localhost yourself.
    assert validate_card(_card("http://localhost:8000"), "triage_incident", "http://localhost:8000") == []


def test_cohort_index_reader_accepts_every_shape_including_firebase():
    """The roster can come from a static index.json, a bare array, or Firebase RTDB -
    which returns an object keyed by callsign with no wrapper. Getting shape 3 wrong
    means the whole cohort sweep silently reports nobody."""
    from scripts.cohort_roster import normalise_index

    rtdb = {
        "swift-falcon-42": {"tag": "swift-falcon-42", "base_url": "https://a.example", "updated_at": 1},
        "_comment": "rules and docs live in the same tree - must be skipped",
    }
    rows = normalise_index(rtdb)
    assert [r["student_id"] for r in rows] == ["swift-falcon-42"]
    assert rows[0]["student_name"] == "swift-falcon-42"

    assert [r["student_id"] for r in normalise_index({"students": [{"student_id": "stu_1"}]})] == ["stu_1"]
    assert [r["student_id"] for r in normalise_index([{"student_id": "stu_2"}])] == ["stu_2"]
    assert normalise_index(None) == []


def test_cohort_config_exposes_the_boss_block():
    """URLs for the coordination service live in cohort.json like every other URL,
    not scattered across scripts."""
    from app.cohort import load_cohort

    c = load_cohort()
    assert c.boss is not None
    for field in ("project_id", "api_key", "database_url", "site_url"):
        assert hasattr(c.boss, field)


def test_a2a_client_prefers_the_dialled_base_over_a_bad_stream_url():
    """A peer with a default AGENTMESH_BASE_URL acks a localhost stream_url. Following
    it silently streams nothing - so we fall back and say why."""
    from scripts.a2a_client import resolve_stream_url

    base = "https://alice.example.com"

    url, warning = resolve_stream_url(base, "task_1", {"stream_url": f"{base}/tasks/task_1/stream"})
    assert url == f"{base}/tasks/task_1/stream"
    assert warning is None

    url, warning = resolve_stream_url(base, "task_1", {"stream_url": "http://localhost:8000/tasks/task_1/stream"})
    assert url == f"{base}/tasks/task_1/stream"
    assert warning and "AGENTMESH_BASE_URL" in warning

    url, warning = resolve_stream_url(base, "task_1", {})
    assert url == f"{base}/tasks/task_1/stream"
    assert warning and "no stream_url" in warning


def test_stream_ends_at_the_gate_with_final_and_resumes_on_resubscribe(client):
    """A2A ends an interaction at an INTERRUPTED state, not just a terminal one.

    The stream must close at TASK_STATE_INPUT_REQUIRED carrying `final`, because the
    next move belongs to the caller and there is nothing left to push. Holding one
    stream open across the gate is what deadlocks behind any buffering proxy: the
    caller cannot see the gate until the response ends, and the response cannot end
    until the caller answers.
    """
    task_id = _submit(client, "Please restart payments-api now", "high")
    asyncio.run(_wait_for(task_id, {TaskState.TASK_STATE_INPUT_REQUIRED}))

    with client.stream("GET", f"/tasks/{task_id}/stream", headers=_auth()) as r:
        assert r.status_code == 200
        leg_one = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]

    # The leg ENDED on its own - no timeout, no cancellation - and said so.
    assert leg_one[-1]["state"] == TaskState.TASK_STATE_INPUT_REQUIRED.value
    assert leg_one[-1]["final"] is True
    assert not any(e.get("final") for e in leg_one[:-1]), "only the last event is final"
    cursor = leg_one[-1]["event_id"]

    assert client.post(f"/tasks/{task_id}/input", json={"approved": True, "note": "ok"},
                       headers=_auth()).status_code == 200

    # Resubscribe with the cursor: we RESUME, not replay - and reach a terminal final.
    with client.stream("GET", f"/tasks/{task_id}/stream",
                       headers={**_auth(), "Last-Event-ID": str(cursor)}) as r:
        leg_two = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]

    assert all(e["event_id"] > cursor for e in leg_two), "resume, not replay"
    assert leg_two[-1]["state"] == TaskState.TASK_STATE_COMPLETED.value
    assert leg_two[-1]["final"] is True


def test_tasks_get_snapshot_tracks_the_task_without_any_stream(client):
    """`tasks/get` is what makes the stream optional rather than load-bearing: a caller
    that never opened one - or sits behind a proxy that ate it - can still find out
    where the task got to, and tell 'waiting for me' apart from 'finished'."""
    assert client.get("/tasks/task_does_not_exist", headers=_auth()).status_code == 404

    task_id = _submit(client, "Please restart payments-api now", "high")
    asyncio.run(_wait_for(task_id, {TaskState.TASK_STATE_INPUT_REQUIRED}))

    paused = client.get(f"/tasks/{task_id}", headers=_auth()).json()
    assert paused["state"] == TaskState.TASK_STATE_INPUT_REQUIRED.value
    assert paused["interrupted"] is True and paused["final"] is False
    assert paused["payload"]["proposal"]["target"] == "payments-api"
    assert paused["last_event_id"] > 0

    client.post(f"/tasks/{task_id}/input", json={"approved": True, "note": "ok"}, headers=_auth())
    asyncio.run(_wait_for(task_id, TERMINAL_STATES))

    done = client.get(f"/tasks/{task_id}", headers=_auth()).json()
    assert done["state"] == TaskState.TASK_STATE_COMPLETED.value
    assert done["final"] is True and done["interrupted"] is False
    assert done["last_event_id"] > paused["last_event_id"]


def test_tasks_get_requires_the_same_scope_as_the_rest(client):
    """A snapshot leaks the task's payload, so it is not a cheaper way in."""
    task_id = _submit(client, "Please restart payments-api now", "high")
    assert client.get(f"/tasks/{task_id}").status_code == 401
    r = client.get(f"/tasks/{task_id}", headers=_auth(scope="mcp:invoke"))
    assert r.status_code == 403


def test_cohort_endpoint_bootstraps_the_browser_roster(client):
    """GET /cohort is what the web UI's Cohort tab sweeps from - it must expose the
    mode, the index URL (online) or the peers (solo), and no auth, without leaking the
    secret-ish boss block. conftest pins COHORT_MODE=solo, so we get the local peers."""
    r = client.get("/cohort")
    assert r.status_code == 200          # unauthenticated by design - it is discovery
    body = r.json()
    assert body["mode"] == "solo"
    assert "boss" not in body            # api_key etc. must not ride along
    ids = [p["student_id"] for p in body["peers"]]
    assert "stu_001" in ids and "stu_002" in ids
    assert all({"student_id", "student_name", "base_url"} <= set(p) for p in body["peers"])


def test_logging_setup_writes_a_rotating_file_and_is_idempotent(tmp_path):
    """setup_logging() attaches ONE rotating file handler to the root logger, so every
    module's logs auto-persist. Called twice (uvicorn reloads the app), it must not
    stack a second handler onto the same file."""
    import logging
    from app.config import Settings
    from app.logging_setup import setup_logging

    log_file = tmp_path / "sub" / "agentmesh.log"
    cfg = Settings(  # required fields still have to be supplied
        openai_api_key="x", jwt_issuer="https://x", jwt_jwks_url="https://x/jwks",
        log_to_file=True, log_file=str(log_file),
    )

    def _our_file_handlers():
        return [h for h in logging.getLogger().handlers if getattr(h, "_agentmesh_file", False)]

    before = len(_our_file_handlers())
    try:
        p1 = setup_logging(cfg)
        p2 = setup_logging(cfg)                     # reload - must be a no-op
        assert p1 == p2
        assert log_file.exists()                    # the parent dir was created too
        assert len(_our_file_handlers()) == before + 1, "a reload must not add a second handler"

        logging.getLogger("app.something").warning("probe-line-xyz")
        for h in _our_file_handlers():
            h.flush()
        assert "probe-line-xyz" in log_file.read_text(encoding="utf-8")
    finally:
        for h in _our_file_handlers():              # leave the root logger as we found it
            logging.getLogger().removeHandler(h)
            h.close()
