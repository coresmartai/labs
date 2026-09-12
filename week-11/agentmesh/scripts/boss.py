"""Talk to AgentMesh Boss - the cohort coordination service.

Everything a student needs during a live session, importable from the notebook:

    from scripts.boss import Boss
    boss = Boss()
    boss.login("me@example.com", "password")   # cached; call once
    boss.whoami()                              # callsign + scopes
    boss.publish("https://xxx.trycloudflare.com")
    boss.token()                               # a FRESH bearer, auto-refreshed
    boss.roster()                              # public list - needs no login

Reading the roster is unauthenticated by design; publishing and token handling are
not. Session state (the refresh token) is cached in `.boss_session.json`, which is
gitignored - treat it like a password.

Uses the Firebase REST APIs directly rather than a Firebase SDK: one dependency
(`httpx`, already required) and every HTTP call stays visible, which matters when
the point of the exercise is understanding the protocol.
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cohort import load_cohort  # noqa: E402

_SESSION_FILE = Path(__file__).resolve().parent.parent / ".boss_session.json"
_IDENTITY = "https://identitytoolkit.googleapis.com/v1/accounts"
_SECURETOKEN = "https://securetoken.googleapis.com/v1/token"
# Refresh a little early: a token that expires in-flight fails at the peer, not here.
_SKEW_S = 120


def _decode_claims(id_token: str) -> dict[str, Any]:
    """Read the payload WITHOUT verifying. Fine here - this is our own token and we
    only use it to display the callsign and scopes. The peer's server is what
    actually verifies it, against Google's JWKS."""
    payload = id_token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


class BossError(RuntimeError):
    pass


class Boss:
    def __init__(self, cohort=None) -> None:
        self.cohort = cohort or load_cohort()
        boss = self.cohort.boss
        if not boss or not boss.api_key or not boss.database_url:
            raise BossError(
                "cohort.json has no usable `boss` block. Fill in api_key and "
                "database_url from the cohort site, or ask your instructor."
            )
        self.api_key = boss.api_key
        self.db_url = boss.database_url.rstrip("/")
        self._session: dict[str, Any] = {}
        if _SESSION_FILE.exists():
            try:
                self._session = json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._session = {}

    # ---------------- auth ----------------

    def _save(self) -> None:
        _SESSION_FILE.write_text(json.dumps(self._session, indent=2), encoding="utf-8")

    def _auth_call(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = httpx.post(f"{_IDENTITY}:{path}", params={"key": self.api_key}, json=payload, timeout=20.0)
        if r.status_code != 200:
            raise BossError(_friendly(r))
        return r.json()

    def signup(self, email: str, password: str) -> str:
        """Create an account. You still have to claim a callsign on the website."""
        data = self._auth_call("signUp", {"email": email, "password": password, "returnSecureToken": True})
        self._store(data)
        return "account created - now claim your callsign on the cohort site"

    def login(self, email: str, password: str) -> str:
        data = self._auth_call(
            "signInWithPassword", {"email": email, "password": password, "returnSecureToken": True}
        )
        self._store(data)
        claims = _decode_claims(self._session["id_token"])
        return f"signed in as {claims.get('email', email)}"

    def _store(self, data: dict[str, Any]) -> None:
        self._session = {
            "id_token": data["idToken"],
            "refresh_token": data["refreshToken"],
            "uid": data["localId"],
            "expires_at": time.time() + int(data.get("expiresIn", 3600)),
        }
        self._save()

    def token(self) -> str:
        """A valid ID token, refreshing if it is close to expiry.

        Firebase ID tokens live one hour. Without this, a class longer than an hour
        starts failing with 401s that look like a code bug and are not.
        """
        if not self._session:
            raise BossError("not signed in - call boss.login(email, password) first")
        if time.time() < self._session.get("expires_at", 0) - _SKEW_S:
            return self._session["id_token"]
        return self._exchange_refresh_token()

    def refresh(self) -> dict[str, Any]:
        """Force a new ID token now, even though the cached one is still valid.

        This is what you run straight after the instructor grants your scopes. A
        scope is a *custom claim*, and claims are baked into a token when it is
        issued - the one in your pocket will keep saying `scopes_present: []` until
        it is replaced, which is why granting alone changes nothing. Exchanging the
        refresh token mints a fresh ID token carrying whatever claims you have now,
        and saves you signing in again.
        """
        if not self._session:
            raise BossError("not signed in - call boss.login(email, password) first")
        self._exchange_refresh_token()
        return self.whoami()

    def _exchange_refresh_token(self) -> str:
        r = httpx.post(
            _SECURETOKEN,
            params={"key": self.api_key},
            data={"grant_type": "refresh_token", "refresh_token": self._session["refresh_token"]},
            timeout=20.0,
        )
        if r.status_code != 200:
            raise BossError(f"session expired, sign in again ({_friendly(r)})")
        d = r.json()
        self._session.update(
            id_token=d["id_token"],
            refresh_token=d["refresh_token"],
            expires_at=time.time() + int(d.get("expires_in", 3600)),
        )
        self._save()
        return self._session["id_token"]

    def logout(self) -> str:
        self._session = {}
        _SESSION_FILE.unlink(missing_ok=True)
        return "signed out"

    # ---------------- identity ----------------

    def whoami(self) -> dict[str, Any]:
        claims = _decode_claims(self.token())
        uid = claims.get("user_id") or claims.get("sub")
        record = self._get(f"students/{uid}", auth=True) or {}
        tag = record.get("tag")
        scope = claims.get("scope", "")
        return {
            "callsign": tag,
            "name": record.get("display_name") or tag,
            "email": claims.get("email"),
            "uid": claims.get("sub"),
            "scope": scope,
            "scope_ok": bool(scope),
            "note": "" if scope else (
                "No scope claim yet - ask the instructor to grant it, then log out and "
                "back in. Claims only appear on a freshly issued token."
            ),
        }

    def env_block(self) -> str:
        """The exact lines to paste into .env."""
        me = self.whoami()
        b = self.cohort.boss
        return (
            f"STUDENT_ID={me['callsign']}\n"
            f"STUDENT_NAME={me.get('name') or me['callsign']}\n"
            f"STUDENT_SPECIALTY=incident-triage\n\n"
            # Signing in to the cohort site means you are going online - flip both switches.
            f"COHORT_MODE=online\n"
            f"AUTH_MODE=jwks\n"
            f"JWT_ISSUER=https://securetoken.google.com/{b.project_id}\n"
            f"JWT_AUDIENCE={b.project_id}\n"
            f"JWT_JWKS_URL=https://www.googleapis.com/service_accounts/v1/jwk/"
            f"securetoken@system.gserviceaccount.com\n"
        )

    # ---------------- roster ----------------

    def _get(self, path: str, auth: bool = False) -> Any:
        params = {"auth": self.token()} if auth else {}
        r = httpx.get(f"{self.db_url}/{path}.json", params=params, timeout=20.0)
        if r.status_code != 200:
            raise BossError(f"database read failed ({r.status_code}): {r.text[:200]}")
        return r.json()

    def publish(self, base_url: str) -> dict[str, Any]:
        """Announce your current tunnel URL to the cohort. Re-run after every restart."""
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith("https://"):
            raise BossError(
                f"{base_url!r} is not an https:// URL. Publish your tunnel URL, not localhost - "
                "the rules reject anything else, and a localhost URL would point classmates "
                "back at their own machine."
            )
        me = self.whoami()
        tag = me["callsign"]
        if not tag:
            raise BossError("you have not claimed a callsign yet - do that on the cohort site")

        # student_name is required by the database rules - a published agent has to
        # say who is behind it, so a classmate is never calling an anonymous callsign.
        body = {"tag": tag, "base_url": base_url, "mcp": f"{base_url}/mcp/",
                "student_name": me.get("name") or tag,
                "updated_at": int(time.time() * 1000)}
        r = httpx.put(
            f"{self.db_url}/public/agents/{tag}.json",
            params={"auth": self.token()}, json=body, timeout=20.0,
        )
        if r.status_code != 200:
            raise BossError(f"publish rejected ({r.status_code}): {r.text[:300]}")
        return body

    def roster(self, live_only: bool = False, stale_minutes: int = 10) -> list[dict[str, Any]]:
        """The public cohort list. No auth - anyone can read this."""
        data = self._get("public/agents") or {}
        now = time.time() * 1000
        rows = []
        for tag, a in data.items():
            age_min = (now - a.get("updated_at", 0)) / 60000
            rows.append({**a, "tag": tag, "age_minutes": round(age_min, 1),
                         "stale": age_min > stale_minutes})
        rows.sort(key=lambda r: r["tag"])
        return [r for r in rows if not r["stale"]] if live_only else rows

    def peers(self) -> list[dict[str, Any]]:
        """Everyone except you - the list you actually want to call."""
        me = self.whoami()["callsign"]
        return [r for r in self.roster() if r["tag"] != me]


def _friendly(r: httpx.Response) -> str:
    try:
        msg = r.json().get("error", {}).get("message", r.text[:200])
    except Exception:  # noqa: BLE001 - a non-JSON error body is itself the signal
        return f"HTTP {r.status_code}: {r.text[:200]}"
    return {
        "EMAIL_EXISTS": "That email already has an account - use login() instead.",
        "EMAIL_NOT_FOUND": "No account with that email.",
        "INVALID_PASSWORD": "Wrong password.",
        "INVALID_LOGIN_CREDENTIALS": "Wrong email or password.",
        "WEAK_PASSWORD : Password should be at least 6 characters": "Password must be 6+ characters.",
        "TOKEN_EXPIRED": "Session expired - log in again.",
    }.get(msg, f"{msg} (HTTP {r.status_code})")
