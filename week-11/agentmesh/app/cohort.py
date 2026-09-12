"""Loader for `cohort.json` - the one place URLs live.

Before this file, a base URL could come from four places: `.env`, a script's
positional argument, a doc's copy-paste block, and a hard-coded default. Every
one of them was a chance to point at the wrong host. Now there is one file, two
modes, and everything else reads from here.

    from app.cohort import load_cohort
    c = load_cohort()
    c.my_base_url           # your service, per the active mode
    c.cohort_index_url      # the class's published index.json (online mode)
    c.peers                 # who to sweep when there is no index
    c.url("mcp")            # your MCP endpoint, path joined for you
    c.peer_url(base, "tasks")

Mode is set in `.env` as `COHORT_MODE` (solo | online), alongside every other
switch a student flips. It is NOT in `cohort.json`: that file is committed and
shared, and which mode YOU are running is a per-machine choice, not a cohort-wide
one. A real shell variable overrides `.env` for a one-off command.

Secrets are NOT here either - the bearer token stays in `.env` (DEV_BEARER_TOKEN)
so `cohort.json` can be committed and shared.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class _CohortEnv(BaseSettings):
    """Just the mode, read from `.env` (or a real env var, which wins).

    Deliberately its own tiny settings object rather than a field on the app's main
    `Settings`: `cohort_roster.py` and the Boss client both call `load_cohort()`
    without a full `.env`, and must not be forced to supply OPENAI_API_KEY and the
    JWT block just to learn whether they are solo or online. Nothing here is
    required, and every other key in `.env` is ignored.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    cohort_mode: Literal["solo", "online"] = "solo"
    # In online mode this fills in online.my_base_url when it is blank - a student sets
    # their tunnel URL here anyway (it is what the Agent Card advertises), so they need
    # not repeat it in cohort.json. Same name/case as the app's AGENTMESH_BASE_URL.
    agentmesh_base_url: str = ""

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "cohort.json"


class Peer(BaseModel):
    student_id: str
    student_name: str = ""
    base_url: str


class Boss(BaseModel):
    """The cohort coordination service, when one is running.

    None of these are secrets: a Firebase web api_key is a public project identifier
    that ships in every visitor's browser. The security is in the database rules.
    """

    project_id: str = ""
    api_key: str = ""
    database_url: str = ""
    site_url: str = ""


class _ModeBlock(BaseModel):
    my_base_url: str = ""
    cohort_index_url: str | None = None
    peers: list[Peer] = Field(default_factory=list)


class CohortConfig(BaseModel):
    """The resolved view for whichever mode is active."""

    mode: Literal["solo", "online"]
    paths: dict[str, str]
    my_base_url: str
    cohort_index_url: str | None
    peers: list[Peer]
    boss: Boss | None = None

    # -- helpers so no caller ever concatenates a URL by hand --

    def url(self, name: str) -> str:
        """A path on YOUR service, e.g. url('mcp')."""
        return self.peer_url(self.my_base_url, name)

    def peer_url(self, base: str, name: str) -> str:
        """The same path on someone else's service."""
        if name not in self.paths:
            raise KeyError(f"unknown path {name!r} - cohort.json declares {sorted(self.paths)}")
        return f"{base.rstrip('/')}{self.paths[name]}"

    def peer(self, student_id: str) -> Peer:
        for p in self.peers:
            if p.student_id == student_id:
                return p
        known = ", ".join(p.student_id for p in self.peers) or "(none)"
        raise KeyError(f"no peer {student_id!r} in cohort.json - known: {known}")


def load_cohort(path: str | Path | None = None, mode: str | None = None) -> CohortConfig:
    """Read cohort.json and flatten the active mode into one object.

    `mode` defaults to `COHORT_MODE` from `.env` (or a shell override); pass it
    explicitly only to force a mode, e.g. in a test. Fails loudly and specifically:
    an empty URL in online mode is the single most common cause of "my classmate
    cannot reach me", so it is caught here rather than three layers down as a
    connection error.
    """
    p = Path(path or os.environ.get("COHORT_CONFIG") or _DEFAULT_PATH)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - copy cohort.json from the repo root")

    raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    env = _CohortEnv()
    mode = mode or env.cohort_mode
    if mode not in ("solo", "online"):
        raise ValueError(f"COHORT_MODE must be 'solo' or 'online', got {mode!r}")

    block = _ModeBlock.model_validate(
        {k: v for k, v in (raw.get(mode) or {}).items() if not k.startswith("_")}
    )
    paths = {k: v for k, v in (raw.get("paths") or {}).items() if not k.startswith("_")}

    # Online: your URL comes from AGENTMESH_BASE_URL if cohort.json leaves it blank, so the
    # .env block the cohort site hands you is enough - no second place to edit. Only when
    # NEITHER is set is it the classic "my classmate cannot reach me" mistake, caught here.
    my_base_url = block.my_base_url or (env.agentmesh_base_url if mode == "online" else "")
    if mode == "online" and not my_base_url:
        raise ValueError(
            "COHORT_MODE=online but no base URL is set. Put your cloudflared URL in "
            "AGENTMESH_BASE_URL (or cohort.json's online.my_base_url) - see SESSION.md."
        )

    boss_raw = {k: v for k, v in (raw.get("boss") or {}).items() if not k.startswith("_")}

    return CohortConfig(
        mode=mode,
        paths=paths,
        my_base_url=my_base_url,
        cohort_index_url=block.cohort_index_url,
        peers=block.peers,
        boss=Boss.model_validate(boss_raw) if boss_raw else None,
    )


def active_mode() -> str:
    """Which mode COHORT_MODE selects - 'solo' or 'online' - without reading
    cohort.json. So it still answers "which mode am I in?" when the online config is
    incomplete and load_cohort() would raise."""
    return _CohortEnv().cohort_mode


@lru_cache(maxsize=1)
def get_cohort() -> CohortConfig:
    """Cached accessor, mirroring get_settings()."""
    return load_cohort()
