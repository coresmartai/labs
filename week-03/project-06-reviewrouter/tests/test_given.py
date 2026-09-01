"""Tests for the parts that are GIVEN to you.

These pass on a fresh clone with no API key. If one of them goes red you
have changed something you were not meant to change.
"""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.routing import route_to_team
from app.schemas import Intent, Priority

client = TestClient(app)

ROOT = Path(__file__).resolve().parent.parent


def test_health_works_without_a_key():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_label_sets_are_closed_and_unchanged():
    assert [i.value for i in Intent] == [
        "billing", "defect", "delivery", "usability", "praise", "other"]
    assert [p.value for p in Priority] == ["low", "normal", "high", "critical"]


def test_routing_table_escalates_only_where_it_should():
    assert route_to_team(Intent.billing, Priority.normal) == "billing"
    assert route_to_team(Intent.billing, Priority.high) == "billing-escalation"
    assert route_to_team(Intent.defect, Priority.critical) == "engineering-oncall"
    assert route_to_team(Intent.delivery, Priority.low) == "fulfillment"
    # The three that do NOT escalate. This is the row people get wrong.
    assert route_to_team(Intent.usability, Priority.critical) == "product"
    assert route_to_team(Intent.other, Priority.critical) == "triage"
    assert route_to_team(Intent.praise, Priority.critical) is None


def test_praise_routes_to_nobody_at_every_priority():
    for p in Priority:
        assert route_to_team(Intent.praise, p) is None


def test_review_pool_is_intact():
    lines = (ROOT / "reviews.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 40
    rows = [json.loads(x) for x in lines]
    assert len({r["id"] for r in rows}) == 40
    assert all(r["review"].strip() for r in rows)


def test_route_to_team_is_not_reachable_from_the_tool_module():
    """The safety gate, asserted.

    route_to_team must never end up in the tool list sent to the model. If
    this goes red you have handed the model the ability to choose a team,
    and an injected review can then choose one for it.
    """
    import app.tools as tools

    assert not hasattr(tools, "route_to_team")


def test_the_only_tool_is_lookup_order():
    """Skipped until you define TOOL_DEFINITION. Then it must stay green."""
    import pytest

    import app.tools as tools

    if tools.TOOL_DEFINITION is None:
        pytest.skip("TOOL_DEFINITION not written yet")
    names = {t.get("name") or t.get("function", {}).get("name")
             for t in tools.tool_list()}
    assert names == {"lookup_order"}
