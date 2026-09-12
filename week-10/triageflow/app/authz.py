"""Who is calling, and what are they allowed to touch.

This file exists because Week 10's argument is that SCOPE IS THE SAFETY
PROPERTY, and a package that argues it while serving any tenant's data to
anybody who guesses a thread id is not making the argument. Four paths in the
shipped version needed this and none of them had it.

READ THE HONEST PART FIRST. `caller_id` trusts an `X-User-Id` header. That is
not authentication: a header is supplied by the client and a client can supply
anything. It stands in for the one line you replace in production, where the
caller comes out of a validated token that your identity provider signed. The
point of the file is everything below that line, which does not change when you
swap it: the AUTHORIZATION checks, which are the part people leave out even
when they have real authentication.

Two authorities live here and they are deliberately not the same one:

  * OWNERSHIP. The user who started a thread owns it. Only they may resume it,
    read its session bundle, or see the prompt log, because the prompt log is a
    verbatim copy of what they typed.
  * APPROVAL. An approver is authorised by ROLE, not by ownership - the whole
    idea of a human gate is that somebody OTHER than the requester decides.
    An approver may approve, and that is all; approving does not make them the
    data subject and does not hand them the requester's prompts.

Collapsing those two into one check is the mistake that makes an approval queue
into a data-export endpoint.
"""
from __future__ import annotations

import logging

from fastapi import Header, HTTPException

log = logging.getLogger(__name__)

# thread id -> the user who started it. In production this is a column, not a
# dict: it must outlive the process for the same reason the checkpoint does.
_THREAD_OWNER: dict[str, str] = {}


def caller_id(x_user_id: str | None = Header(default=None)) -> str:
    """THE LINE YOU REPLACE. In production: decode and verify a bearer token and
    return the subject claim. Never read the caller's identity from a field the
    caller controls."""
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-Id header required")
    return x_user_id


def claim_thread(thread_id: str, user_id: str) -> None:
    """Record who a thread belongs to. Idempotent for the same owner."""
    owner = _THREAD_OWNER.get(thread_id)
    if owner is not None and owner != user_id:
        log.warning("authz.thread_claim_conflict thread=%s owner=%s caller=%s",
                    thread_id, owner, user_id)
        raise HTTPException(status_code=403, detail="thread belongs to another user")
    _THREAD_OWNER[thread_id] = user_id


def assert_owns_thread(thread_id: str, caller: str) -> None:
    """404, not 403, for a thread that exists but is not yours.

    Telling a caller "that thread belongs to someone else" confirms the thread
    exists, which is the one bit an enumeration attack is looking for.
    """
    owner = _THREAD_OWNER.get(thread_id)
    if owner is None or owner != caller:
        log.warning("authz.thread_denied thread=%s caller=%s", thread_id, caller)
        raise HTTPException(status_code=404, detail="thread not found")


def owner_of(thread_id: str) -> str | None:
    return _THREAD_OWNER.get(thread_id)


def assert_is_approver(caller: str, reviewer_id: str) -> None:
    """Approval is role-based, and you approve AS YOURSELF.

    The second check is the smaller one and it is the one that keeps the audit
    trail worth having: a reviewer_id that anybody can set to anybody is a name
    in a log, not a record of who decided.
    """
    from .config import get_settings

    approvers = {a.strip() for a in get_settings().approver_ids.split(",") if a.strip()}
    if caller not in approvers:
        log.warning("authz.approve_denied caller=%s", caller)
        raise HTTPException(status_code=403, detail="caller is not an approver")
    if reviewer_id != caller:
        raise HTTPException(status_code=403, detail="reviewer_id must be the calling approver")


def forget_user_threads(user_id: str) -> int:
    """Erasure reaches here too. Which threads a person started is a record ABOUT
    that person, so a deletion path that leaves this map has left a row behind."""
    gone = [t for t, owner in _THREAD_OWNER.items() if owner == user_id]
    for t in gone:
        del _THREAD_OWNER[t]
    return len(gone)


def reset_authz() -> None:
    """Test hook - the ownership map is process-global."""
    _THREAD_OWNER.clear()
