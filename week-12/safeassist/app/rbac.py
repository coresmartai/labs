"""RBAC + ACL helpers.

RBAC = endpoint-level decorator. The first gate, cheap and easy.
ACL  = data-level filter pushed into the retrieval query. The second gate.

Both ship together. The decorator stops the call from reaching the route
handler at all. The ACL filter restricts what data the retriever will return
even if the decorator is satisfied by a compromised or spoofed credential.
Defence in depth: the outer ring defends the endpoint, the inner ring defends
the data.
"""
from __future__ import annotations

import logging
from functools import wraps
from typing import Callable

from fastapi import HTTPException, Request
from qdrant_client import models

from app import audit

logger = logging.getLogger(__name__)


def requires_role(role: str) -> Callable:
    """Endpoint decorator. The caller must hold `role` in their roles list.

    The route handler must accept a `request: Request` parameter; the
    decorator pulls the user identity off `request.state.user`.
    """
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapped(*args, **kwargs):
            request: Request | None = kwargs.get("request")
            if request is None:
                # Search positional args for a Request as a safety net.
                for a in args:
                    if isinstance(a, Request):
                        request = a
                        break
            if request is None:
                raise HTTPException(status_code=500, detail="rbac: request not propagated")

            user = getattr(request.state, "user", None)
            roles = list(user.get("roles", [])) if user else []
            user_id = (user or {}).get("user_id", "anonymous")
            request_id = getattr(request.state, "request_id", None) or request.headers.get("x-request-id", "unknown")

            if role not in roles:
                audit.write_event(
                    request_id=request_id,
                    user_id=user_id,
                    event_type="rbac_denial",
                    detail={
                        "route": str(request.url.path),
                        "required_role": role,
                        "caller_roles": roles,
                    },
                    action_taken="403_forbidden",
                )
                raise HTTPException(status_code=403, detail="forbidden")

            return await fn(*args, **kwargs)
        return wrapped
    return deco


def acl_filter_for(roles: list[str]) -> models.Filter:
    """The Qdrant filter that restricts chunks to those the caller may see.

    Two lines in the search call. Every chunk carries a `visible_to` list,
    written at ingestion from wherever the source document lived; the query
    carries the caller's roles and requires an overlap. If there is no
    overlap, the chunk is never returned - the retriever, not the route, is
    what stops it.
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key="visible_to",
                match=models.MatchAny(any=list(roles)),
            )
        ]
    )
