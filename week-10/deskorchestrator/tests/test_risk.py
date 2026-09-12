"""TASK 3. Two of these fail on a fresh clone. The middle of the range is yours."""
from __future__ import annotations

from app.risk import risky

GRANT_WRITE = {"tool_name": "grant_access", "arguments": {"system": "billing-db", "level": "write"}}
ORDER_KEYBOARD = {"tool_name": "order_hardware", "arguments": {"item": "keyboard", "quantity": 1}}
NO_ACTION = {"tool_name": "none", "arguments": {}}


def test_risky_is_a_pure_function_of_the_proposal():
    """Resuming re-runs the node from the top, so the verdict is computed twice.

    A risky() that can answer differently on re-entry is a gate that can be
    walked around. This test calls it ten times and demands one answer.
    """
    for proposal in (GRANT_WRITE, ORDER_KEYBOARD, NO_ACTION):
        verdicts = {risky(dict(proposal)) for _ in range(10)}
        assert len(verdicts) == 1, (
            f"risky() gave more than one answer for the same proposal {proposal['tool_name']!r}. "
            "It must depend on nothing but its argument.")


def test_a_write_grant_on_a_production_system_pauses():
    """The top of the range. Whatever else your policy does, it must stop this."""
    assert risky(GRANT_WRITE) is True, (
        "granting write access to a production database did not pause for a human.")


def test_a_stock_keyboard_does_not_pause():
    """The bottom of the range, and the one the shipped version fails.

    A gate that wakes somebody at 3 a.m. for a keyboard is a gate people learn to
    click through, and a rubber-stamp gate is worse than no gate at all.
    """
    assert risky(ORDER_KEYBOARD) is False, (
        "ordering one keyboard paused for a human. Everything pausing is the same as "
        "nothing pausing, once the approvers have learned the rhythm.")


def test_no_action_does_not_pause():
    """A proposal of 'none' has nothing to approve."""
    assert risky(NO_ACTION) is False, (
        "a proposal that does nothing still woke a human.")
