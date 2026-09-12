"""The failure shape, defined ONCE, used by every tool.

MCP gives you two error paths and the distinction decides whether a model can
recover (see W11-R01):

  * a TOOL EXECUTION error  - the tool ran, the operation did not succeed.
    Clients SHOULD pass it to the model, so the message has to be worth reading.
  * a JSON-RPC PROTOCOL error - unknown method, malformed arguments. The model
    can rarely do anything with it.

Everything in this file is the first kind. We raise ToolError, which the SDK
turns into a result with isError set, and we put a machine-readable code and an
explicit `retryable` flag in the message so the calling model can decide what to
do next instead of guessing.

`retryable` is a CONVENTION, not something either protocol mandates. That is
exactly why it is defined once here and not invented per tool.
"""
from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError


ENVELOPE = "toolbridge-error: "


def fail(code: str, message: str, *, retryable: bool, **extra: Any) -> ToolError:
    """Build the one failure shape this server returns.

    The body is JSON so that a caller can parse it, and it leads with a plain
    sentence so that a model reading it as prose still gets the point.
    """
    body = {"code": code, "message": message, "retryable": retryable, **extra}
    # The prose comes first, because the model reads it as prose. The machine-
    # readable copy follows behind a sentinel rather than brackets: a message
    # that itself mentions `supportedInterfaces[]` would otherwise make the
    # envelope unparseable, which is a bug we shipped once and only once.
    return ToolError(f"{message}\n{ENVELOPE}{json.dumps(body, separators=(',', ':'))}")


# The A2A states that are terminal: the task accepts no further messages.
TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
}

# The states that LOOK final on a stream and are not. The task is alive and
# waiting for you.
INTERRUPTED_STATES = {
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_AUTH_REQUIRED",
}

# Mapping from an A2A terminal state to whether asking again could help.
# REJECTED is the one people get wrong: the peer read the request and declined.
# Retrying will produce the same refusal and waste its capacity doing so.
RETRYABLE_BY_STATE = {
    "TASK_STATE_COMPLETED": False,
    "TASK_STATE_FAILED": True,
    "TASK_STATE_CANCELED": False,
    "TASK_STATE_REJECTED": False,
}
