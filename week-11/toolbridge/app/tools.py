"""The three tool bodies. TASKS 2, 3, 4 and 5 live here.

Each tool is a plain async function over `app.a2a`. Keeping them free of any MCP
import is deliberate: `app/mcp_server.py` registers them, and these stay
testable without a server, a transport, or a client session.
"""
from __future__ import annotations

from typing import Any

import httpx

from app import a2a
from app.config import get_settings
from app.errors import INTERRUPTED_STATES, RETRYABLE_BY_STATE, TERMINAL_STATES, fail


def _peer(peer_url: str | None) -> str:
    return (peer_url or get_settings().peer_base_url).rstrip("/")


def _as_tool_error(exc: a2a.PeerError):
    """Turn a peer failure into THE failure shape, preserving retryability.

    TASK 5. Note what is NOT happening here: no bare re-raise, no stack trace
    reaching the model, and no invented retryable flag. The peer told us, or the
    status code did, and we pass that through.
    """
    return fail(exc.code, exc.message, retryable=exc.retryable, peer_status=exc.status)


# ---------------------------------------------------------------- TASK 2
async def peer_card(peer_url: str | None = None) -> dict[str, Any]:
    """Fetch a peer's Agent Card, validate it, and report what we can use."""
    # TASK 2: implement this.
    #
    # `app.a2a` gives you fetch_card() and select_interface(). Both raise
    # PeerError; _as_tool_error() above turns that into the one failure shape.
    #
    # Return a dict. What goes in it is your call, but the tests require:
    #   - the peer's name and its operator (card["provider"]["organization"])
    #   - the interface you SELECTED, including its RANK in supportedInterfaces.
    #     Rank matters: it is how a reader can tell you honoured the ordering
    #     rather than taking entry zero and hoping.
    #   - one entry per skill
    #
    # Two things to get right:
    #   - a card whose supportedInterfaces is absent is from before March 2026.
    #     Say so in the error rather than raising a KeyError.
    #   - `capabilities` is protocol flags. Do not put skills in it.
    raise NotImplementedError("TASK 2")


# ---------------------------------------------------------------- TASKS 3 and 4
async def delegate_triage(description: str, severity: str, user_id: str,
                          peer_url: str | None = None) -> dict[str, Any]:
    """Submit an incident to the peer and follow it to an outcome or a pause."""
    # TASK 3: implement this.
    #
    # a2a.submit() then a2a.follow(). The skill is "triage_incident" and the
    # input the peer wants is {description, severity, user_id}.
    #
    # You will discover that shape from a 422 the first time, because A2A does
    # NOT carry a schema for a skill's input. Once you know it, validate it
    # locally before dialling: the peer requires a description of at least 10
    # characters, a severity in {low, medium, high}, and a non-empty user_id.
    # A locally-invalid call must never reach the peer.
    #
    # NO BUSY-WAIT. Follow the stream. When it drops, a2a.follow() re-attaches
    # from the cursor rather than resubmitting, and the tests check that exactly
    # one POST /tasks happens.
    #
    # Hand what comes back to _settle(), which is TASK 4.
    raise NotImplementedError("TASK 3")


def _settle(state: str, payload: dict[str, Any], task_id: str, base: str,
            cursor: int) -> dict[str, Any]:
    """Turn an A2A state into something an MCP caller can act on."""
    # TASK 4: implement this. It is the most interesting decision in the project.
    #
    # A2A has nine states. A tool result has to say one of three things:
    #   here is your answer  /  here is why you cannot have one  /  not finished
    #
    # The middle case is easy. The third is the one to think about.
    #
    # INTERRUPTED_STATES (INPUT_REQUIRED, AUTH_REQUIRED) are NOT terminal. The
    # task is alive on the peer, holding state, waiting for an answer. Returning
    # an error here would be wrong twice: nothing failed, and the model would
    # have no way back to the task.
    #
    # So: what do you return? Whatever it is, the caller needs enough to come
    # back, which is at minimum the task id AND the replay cursor. (Read
    # a2a.follow()'s docstring on why the cursor. It is the handle pattern from
    # W11-R01 and skipping it produces a bug you will spend an hour on.)
    #
    # TERMINAL_STATES: COMPLETED returns the result. FAILED, CANCELED and
    # REJECTED raise fail(), with retryable taken from RETRYABLE_BY_STATE.
    # REJECTED is NOT retryable and the message should say why.
    #
    # HUMAN_GATE_MODE in config switches between the `resume` design above and
    # the `elicit` one in app/mcp_server.py. Implement at least one; the memo
    # asks you which you chose and what you gave up.
    raise NotImplementedError("TASK 4")


# ---------------------------------------------------------------- TASK 4 (part 2)
async def resume_task(task_id: str, decision: str, reviewer_id: str,
                      resume_from: int = 0, note: str = "",
                      peer_url: str | None = None) -> dict[str, Any]:
    """Answer a paused task and return its final outcome."""
    # TASK 4, second half, and TASK 5.
    #
    # Validate first: decision must be exactly "approve" or "reject", and
    # reviewer_id must be non-empty. An approval with no named approver is not
    # an audit trail. Both are fail(), not exceptions.
    #
    # Then a2a.reply_to_gate(), then a2a.follow() FROM resume_from, then
    # _settle(). Resuming from zero replays the pause you just answered.
    #
    # TASK 5 lives here and in the two functions above: every a2a.PeerError that
    # crosses this boundary becomes the one failure shape, with the retryable
    # flag the peer's status code implies. Four cases the tests exercise:
    #   401 from the peer          -> not retryable, our token is wrong
    #   422 from the peer          -> not retryable, our shape is wrong
    #   a dropped stream           -> handled inside follow(), not an error
    #   409 on replying            -> not retryable, the task is not paused
    raise NotImplementedError("TASK 4 and TASK 5")
