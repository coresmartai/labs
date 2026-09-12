"""JWT verification - a FastAPI dependency factory.

Standard pattern: fetch JWKS (cached 24 h), verify the signature, check audience,
issuer, expiry (with leeway) and scope. Every failure is a structured 401/403/503
carrying a machine-readable `reason` - never a stack trace.

Auth is at the door: a request either arrives authenticated or it never reaches a
handler. That is why the A2A `TASK_STATE_AUTH_REQUIRED` state is declared but never
entered - there is no mid-task auth challenge to raise.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import httpx
import jwt
from fastapi import Depends, Header, HTTPException, Query, status
from jwt import PyJWKError, PyJWTError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Module-level cache: { jwks_url: (fetched_at_seconds, keys_payload) }
_JWKS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


async def _get_jwks(settings: Settings) -> dict[str, Any]:
    """Fetch and cache JWKS. Steady state = one fetch per cache-ttl, not per request."""
    now = time.time()
    cached = _JWKS_CACHE.get(settings.jwt_jwks_url)
    if cached and (now - cached[0]) < settings.jwt_jwks_cache_seconds:
        return cached[1]
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(settings.jwt_jwks_url)
        r.raise_for_status()
        payload = r.json()
    _JWKS_CACHE[settings.jwt_jwks_url] = (now, payload)
    return payload


def reset_jwks_cache_for_tests(payload: dict[str, Any] | None = None, url: str = "") -> None:
    """Test seam: prime or clear the cache."""
    if payload is None:
        _JWKS_CACHE.clear()
    else:
        _JWKS_CACHE[url] = (time.time(), payload)


async def verify_bearer(bearer: str, settings: Settings, required_scope: str) -> dict[str, Any]:
    """Verify one bearer token and check its scope. Raises a structured
    HTTPException on any failure, returns the claims on success.

    Factored out of the FastAPI dependency below because the co-hosted MCP mount
    cannot use dependencies at all - `app.mount()` bypasses them - so `main.py`
    guards that path in middleware. Both paths MUST share this function: two
    implementations of "is this caller allowed" is how one of them quietly stops
    matching the other.
    """
    if settings.auth_mode == "dev":
        # Local demo mode: accept the fixed dev bearer token and grant the FIXED
        # scope set from settings - NOT whatever scope the route happened to ask
        # for. Granting the asked-for scope would make the 403 branch unreachable,
        # which is the same as not having a scope check at all.
        if bearer != settings.dev_bearer_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"reason": "auth_invalid", "message": "unknown dev token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        claims: dict[str, Any] = {"sub": "dev-user", "scope": " ".join(settings.dev_scopes)}
    else:
        try:
            jwks = await _get_jwks(settings)
            header = jwt.get_unverified_header(bearer)
            alg = header.get("alg", "RS256")
            jwk = next((k for k in jwks.get("keys", []) if k.get("kid") == header.get("kid")), None)
            if jwk is None:
                raise PyJWTError(f"no matching key for kid={header.get('kid')}")
            key = jwt.PyJWK.from_dict(jwk, algorithm=alg).key
            claims = jwt.decode(
                bearer,
                key,
                algorithms=[alg],
                audience=settings.jwt_audience,   # strict: exactly "agentmesh", no wildcards
                issuer=settings.jwt_issuer,
                leeway=settings.jwt_leeway_seconds,
                options={"require": ["exp", "aud", "iss"]},
            )
        except httpx.HTTPError as e:
            # Identity provider unreachable - structured 503, never a stack trace.
            logger.warning("jwt.jwks_unreachable error=%s", str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"reason": "jwks_unavailable", "message": "identity provider JWKS endpoint unreachable"},
            )
        except (PyJWTError, PyJWKError) as e:
            logger.warning("jwt.verify_failed error=%s", str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"reason": "auth_invalid", "message": str(e)},
                headers={"WWW-Authenticate": "Bearer"},
            )

    scopes = set((claims.get("scope") or "").split())
    if required_scope not in scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "reason": "scope_missing",
                "message": f"required scope '{required_scope}' not present on token",
                "scopes_present": sorted(scopes),
            },
        )
    return claims


def require_scope(required_scope: str) -> Callable:
    """Returns a FastAPI dependency that verifies the JWT and checks for required_scope."""

    async def _dependency(
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None, include_in_schema=False),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization.split(" ", 1)[1]
        elif token:
            # EventSource cannot send an Authorization header - the browser UI
            # passes the same bearer as a query parameter on the SSE route.
            # A real orchestrator's HTTP client uses the header.
            bearer = token
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"reason": "missing_bearer", "message": "Authorization: Bearer <token> required"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await verify_bearer(bearer, settings, required_scope)

    return _dependency
