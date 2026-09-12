"""TASK 1. Three of these four fail on a fresh clone, and they are the project.

Read the failure messages before you read app/vectorstore.py: they say what
property broke, not where the bug is.

The first test PASSES on arrival and it is not filler. It pins the property the
broken implementation still has, and knowing which one that is will stop you
fixing the wrong thing. W10-R03 section 5 makes the same point in prose.
"""
from __future__ import annotations

from app.memory import search_policies, search_tickets

# These tests pass k explicitly rather than reading RETRIEVAL_K, so tuning that
# setting (or leaving one in a local .env) cannot turn a correct fix red.
K = 3


def test_post_filtering_does_not_leak_and_this_passes_on_arrival():
    """The property the shipped implementation KEEPS. Do not delete this test.

    Filtering after ranking still filters. No row belonging to another user can
    reach this caller, whatever order the two steps happen in, so the scaffold is
    not leaking and never was. Be precise about that before you start, because
    the three failures below are a RECALL problem and the fix for a recall problem
    is not the fix for a leak.
    """
    for k in (1, 3, 20):
        for uid in ("u_1", "u_2"):
            hits = search_tickets("access to the billing database", user_id=uid, k=k)
            foreign = [h["chunk_id"] for h in hits if h["scope"] != f"user:{uid}"]
            assert not foreign, f"k={k}, {uid}: rows belonging to someone else came back: {foreign}"


def test_a_user_with_rows_gets_the_rows_back():
    """Recall, part one. u_1 has three prior tickets and asks for three."""
    hits = search_tickets("write access to the production billing database", user_id="u_1", k=K)
    assert len(hits) == K, (
        f"asked for {K} of u_1's tickets and got {len(hits)}. u_1 has three of them. "
        "The rows exist and the caller cannot see them, which is worse than an error "
        "because nothing errored.")


def test_a_scoped_search_can_come_back_completely_empty():
    """Recall, part two, and the failure that looks exactly like 'no data'.

    Nothing is special about this query. It happens to be one where the global
    policy corpus outranks every one of u_1's tickets, so a caller with three
    stored tickets is told they have none.
    """
    hits = search_tickets("licence seat cost centre", user_id="u_1", k=K)
    assert hits, (
        "u_1 has three tickets and this search returned none of them. Correct and empty "
        "is a real failure mode, and in production it is indistinguishable from a user "
        "who has never raised a ticket.")


def test_one_scope_does_not_crowd_out_another():
    """Recall, part three. The two scopes must not compete for the same k slots."""
    q = "monitor from the depot within the annual allowance"
    policies = search_policies(q, k=K)
    tickets = search_tickets(q, user_id="u_2", k=K)

    assert len(policies) == K, (
        f"the global policy search returned {len(policies)} of {K}. The policy corpus holds eight "
        "chunks, so a short result means rows outside that scope are competing for the slots.")
    assert all(h["collection"] == "policies" for h in policies), \
        "the global policy search returned rows from another collection."
    assert len(tickets) == 2, (
        f"u_2 has exactly two tickets and the search returned {len(tickets)}. Both should be "
        "reachable by a query about depot stock, with nothing else taking their place.")
