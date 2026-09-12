"""Drive a *peer's* AgentMesh service over A2A - the client half of the cohort demo.

`app/main.py` only ever **serves** A2A. This script is the other side of the
conversation: it discovers a classmate's agent from its Agent Card, submits a real
triage task, follows the SSE stream, answers the human-approval gate remotely, and
reports the terminal state.

    # discovery only - fetch and validate the card, submit nothing
    python scripts/a2a_client.py https://alice.trycloudflare.com --card-only

    # a knowledge request: submitted -> working -> completed, no gate
    python scripts/a2a_client.py https://alice.trycloudflare.com \
        --description "Where is the runbook for restarting payments-api?" --severity low

    # an action request: pauses at input-required, we approve it remotely
    python scripts/a2a_client.py https://alice.trycloudflare.com \
        --description "Please restart payments-api now" --severity high --approve

Auth: `--token`, or the `A2A_PEER_TOKEN` environment variable. When the peer runs
`AUTH_MODE=dev` this is whatever the cohort agreed as `DEV_BEARER_TOKEN` (the shipped
default is `demo-token`).

Unlike the browser UI - which has to smuggle the bearer through a `?token=` query
parameter because `EventSource` cannot set headers - this client sends a normal
`Authorization` header on every call, including the stream. That is what a real
orchestrator does.

See `SESSION.md` for the tunnel + `AGENTMESH_BASE_URL` setup this assumes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

# Python puts THIS file's directory on sys.path, not the repo root, so `app.cohort`
# would not import however you invoked us. Add the root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CARD_PATH = "/.well-known/agent-card.json"
DEFAULT_SKILL = "triage_incident"

TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
}

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


# ---------------------------------------------------------------------------
# Pure helpers - no network, no I/O. These are the two things that actually go
# wrong when a cohort first tries this across the internet, so they are unit
# tested in tests/test_endpoint.py rather than only discovered on the day.
# ---------------------------------------------------------------------------


def _is_local(url: str) -> bool:
    try:
        return (urlsplit(url).hostname or "").lower() in _LOCAL_HOSTS
    except ValueError:
        return False


def validate_card(card: dict[str, Any], skill: str, dialled_base: str) -> list[str]:
    """Check a peer's Agent Card before trusting it. Returns a list of problems,
    empty when the card is usable. Order is stable so output reads the same twice.

    The last check is the one that bites remote cohorts: a card fetched from a
    public URL that advertises `localhost` means the peer never set
    AGENTMESH_BASE_URL, so every URL they hand out points at *your* machine.
    """
    problems: list[str] = []

    version = card.get("protocolVersion")
    if not version:
        problems.append("card has no protocolVersion")
    elif version != "1.0":
        problems.append(f"card advertises protocolVersion={version!r}; this client speaks 1.0")

    skills = card.get("skills") or []
    if not skills:
        problems.append("card advertises no skills[]")
    elif not any(s.get("id") == skill for s in skills):
        offered = ", ".join(sorted(str(s.get("id")) for s in skills)) or "(none)"
        problems.append(f"card does not offer skill {skill!r} - it offers: {offered}")

    endpoints = card.get("endpoints") or {}
    if not endpoints.get("base"):
        problems.append("card has no endpoints.base")

    if not _is_local(dialled_base):
        advertised = [u for u in (card.get("url"), endpoints.get("base")) if u]
        if any(_is_local(u) for u in advertised):
            problems.append(
                f"card advertises a localhost URL ({', '.join(advertised)}) but you dialled "
                f"{dialled_base} - the peer has not set AGENTMESH_BASE_URL to their public URL"
            )

    return problems


def resolve_stream_url(base: str, task_id: str, ack: dict[str, Any] | None) -> tuple[str, str | None]:
    """Pick the SSE URL to follow, and say so when the peer's ack is wrong.

    `POST /tasks` returns a `stream_url` built from the peer's own
    AGENTMESH_BASE_URL. If they left it at the default, that ack tells you to
    stream from *your own* localhost: the task runs correctly on their machine
    and you simply never see it. We prefer the base we actually dialled and
    return a warning rather than failing silently.
    """
    base = base.rstrip("/")
    fallback = f"{base}/tasks/{task_id}/stream"
    advertised = (ack or {}).get("stream_url") or ""

    if not advertised:
        return fallback, "the 202 ack carried no stream_url; using the base URL you dialled"
    if advertised.startswith(base):
        return advertised, None
    return fallback, (
        f"the peer advertised stream_url={advertised}, which is not on {base} - their "
        "AGENTMESH_BASE_URL is not set to their public URL. Using the dialled base instead."
    )


# ---------------------------------------------------------------------------
# The A2A round trip
# ---------------------------------------------------------------------------


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def fetch_card(client: httpx.AsyncClient, base: str) -> dict[str, Any]:
    """Discovery. Unauthenticated by design - anyone may read an Agent Card."""
    r = await client.get(f"{base.rstrip('/')}{CARD_PATH}", timeout=15.0)
    r.raise_for_status()
    return r.json()


async def submit_task(
    client: httpx.AsyncClient, base: str, token: str, skill: str, payload: dict[str, Any]
) -> dict[str, Any]:
    r = await client.post(
        f"{base.rstrip('/')}/tasks",
        headers={**_auth(token), "Content-Type": "application/json"},
        json={"skill": skill, "input": payload},
        timeout=30.0,
    )
    if r.status_code != 202:
        raise SystemExit(f"submit failed: {r.status_code} {r.text}")
    print(f"  A2A-Version: {r.headers.get('A2A-Version', '(absent)')}")
    return r.json()


async def reply_to_gate(
    client: httpx.AsyncClient, base: str, token: str, task_id: str, approved: bool, note: str
) -> None:
    r = await client.post(
        f"{base.rstrip('/')}/tasks/{task_id}/input",
        headers={**_auth(token), "Content-Type": "application/json"},
        json={"approved": approved, "note": note},
        timeout=30.0,
    )
    verdict = "approved" if approved else "rejected"
    if r.status_code == 200:
        print(f"  -> {verdict} remotely")
    else:
        print(f"  -> reply rejected: {r.status_code} {r.text}", file=sys.stderr)


async def _decide(proposal: dict[str, Any], mode: str) -> tuple[bool, str]:
    if mode == "approve":
        return True, "auto-approved by a2a_client"
    if mode == "reject":
        return False, "auto-rejected by a2a_client"
    answer = await asyncio.to_thread(input, "  approve this action? [y/N] ")
    ok = answer.strip().lower().startswith("y")
    return ok, "answered interactively"


async def follow_stream(
    client: httpx.AsyncClient, base: str, token: str, task_id: str, stream_url: str, mode: str
) -> str | None:
    """Consume the SSE stream to a terminal state, answering the gate on the way.

    The approval POST happens *while* this stream is still open - a second
    connection, exactly as an orchestrator would do it. The peer's background
    task is parked on an asyncio.Event; our reply releases it and the remaining
    events arrive on the stream we are already holding.
    """
    cursor = 0
    while True:
        state, cursor, payload = await _stream_leg(client, stream_url, token, cursor)

        if state is None:
            # The stream told us nothing - dropped, or an intermediary is holding the
            # body. Fall back to tasks/get, which is a finite response and therefore
            # always gets through. Slower, never stuck.
            print("      (no events on the stream - falling back to polling tasks/get)")
            state, cursor, payload = await _poll_until_settled(client, base, token, task_id)
            if state is None:
                return None

        if state in TERMINAL_STATES:
            if state == "TASK_STATE_COMPLETED":
                print(f"      result: {json.dumps(payload)}")
            return state

        # INPUT_REQUIRED: the interaction ended because it is our move now. Answer,
        # then resubscribe from where the stream stopped - Last-Event-ID means we
        # resume rather than replay, and never miss what happened in between.
        proposal = payload.get("proposal") or {}
        print(f"      proposal: {json.dumps(proposal)}")
        approved, why = await _decide(proposal, mode)
        await reply_to_gate(client, base, token, task_id, approved, why)
        print(f"  ...resubscribing from event #{cursor}")


async def _stream_leg(
    client: httpx.AsyncClient, stream_url: str, token: str, cursor: int
) -> tuple[str | None, int, dict[str, Any]]:
    """One interaction's worth of stream: consume until the `final` event, return it.

    Returns (state, last_event_id, payload). state is None if the stream ended
    without ever reporting one - the signal to stop trusting the stream.
    """
    state: str | None = None
    payload: dict[str, Any] = {}
    headers = {**_auth(token), "Last-Event-ID": str(cursor)} if cursor else _auth(token)

    async with client.stream("GET", stream_url, headers=headers, timeout=None) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")
            raise SystemExit(f"stream failed: {response.status_code} {body}")

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:].strip())
            state, payload = event["state"], event.get("payload") or {}
            cursor = max(cursor, event["event_id"])

            if payload.get("replay_gap"):
                print(f"  #{event['event_id']:<3} REPLAY GAP - oldest available "
                      f"#{payload['oldest_available_event_id']}")
                state = None
                continue

            note = payload.get("message") or payload.get("reason") or ""
            print(f"  #{event['event_id']:<3} {state:26s} {note}")

            if event.get("final"):
                break
    return state, cursor, payload


async def get_task_status(
    client: httpx.AsyncClient, base: str, token: str, task_id: str
) -> dict[str, Any]:
    """A2A `tasks/get` - one snapshot, no streaming involved."""
    r = await client.get(
        f"{base.rstrip('/')}/tasks/{task_id}", headers=_auth(token), timeout=30.0
    )
    if r.status_code != 200:
        raise SystemExit(f"tasks/get failed: {r.status_code} {r.text}")
    return r.json()


async def _poll_until_settled(
    client: httpx.AsyncClient, base: str, token: str, task_id: str, every: float = 1.0,
    limit: int = 300,
) -> tuple[str | None, int, dict[str, Any]]:
    """Poll tasks/get until the task is final or wants us. Same two exits as a stream."""
    for _ in range(limit):
        s = await get_task_status(client, base, token, task_id)
        if s["final"] or s["interrupted"]:
            print(f"  #{s['last_event_id']:<3} {s['state']:26s} (via tasks/get)")
            return s["state"], s["last_event_id"], s.get("payload") or {}
        await asyncio.sleep(every)
    return None, 0, {}


async def run(args: argparse.Namespace) -> int:
    base = args.base_url.rstrip("/")
    async with httpx.AsyncClient(follow_redirects=True) as client:
        print(f"\n[1] discovery - GET {base}{CARD_PATH}")
        try:
            card = await fetch_card(client, base)
        except httpx.HTTPError as exc:
            print(f"    cannot reach the peer: {exc}", file=sys.stderr)
            return 1

        print(f"    {card.get('name')} v{card.get('version')} "
              f"(protocol {card.get('protocolVersion')})")
        for skill in card.get("skills") or []:
            print(f"    skill: {skill.get('id')} - {str(skill.get('description', ''))[:60]}...")

        problems = validate_card(card, args.skill, base)
        for p in problems:
            print(f"    WARNING: {p}", file=sys.stderr)
        if any("does not offer skill" in p for p in problems):
            return 1
        if args.card_only:
            return 0

        # `whoami` takes no input; sending triage fields would be noise on the wire.
        payload: dict[str, Any] = {} if args.skill == "whoami" else {
            "description": args.description, "severity": args.severity, "user_id": args.user_id
        }
        print(f"\n[2] submit - POST {base}/tasks (skill={args.skill})")
        ack = await submit_task(client, base, args.token, args.skill, payload)
        task_id = ack["task_id"]
        print(f"    task_id={task_id} state={ack['state']}")

        stream_url, warning = resolve_stream_url(base, task_id, ack)
        if warning:
            print(f"    WARNING: {warning}", file=sys.stderr)

        print(f"\n[3] stream - GET {stream_url}")
        final = await follow_stream(client, base, args.token, task_id, stream_url, args.mode)

    print(f"\nterminal state: {final}")
    # A deliberate --reject that lands in REJECTED is a successful run, not a failure:
    # exit non-zero only when the task ended somewhere we did not ask it to.
    expected = "TASK_STATE_REJECTED" if args.mode == "reject" else "TASK_STATE_COMPLETED"
    return 0 if final == expected else 2


def _resolve_token() -> str:
    """Find a bearer, preferring the freshest source.

    A Firebase ID token lives one hour, so pasting one into `.env` guarantees it goes
    stale mid-session and produces 401s that look like a code bug. If the cohort site
    is configured and you have signed in with `boss.login(...)`, take a freshly
    refreshed token from there instead; `A2A_PEER_TOKEN` still wins if you set it.
    """
    env = os.environ.get("A2A_PEER_TOKEN")
    if env:
        return env
    try:
        from scripts.boss import Boss  # noqa: PLC0415 - optional, only when configured

        return Boss().token()
    except Exception:  # noqa: BLE001 - no cohort site, or not signed in: fall through
        return "demo-token"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("base_url", nargs="?",
                   help="the peer's base URL. Omit it and pass --peer to resolve one from cohort.json")
    p.add_argument("--peer", metavar="STUDENT_ID",
                   help="look the peer's base URL up in cohort.json instead of typing it")
    p.add_argument("--token", default=None,
                   help="bearer token. Default: $A2A_PEER_TOKEN, else a fresh token from "
                        "the cohort site if you are signed in, else demo-token")
    p.add_argument("--skill", default=DEFAULT_SKILL, help=f"skill id to invoke (default: {DEFAULT_SKILL})")
    p.add_argument("--description", default="Where is the runbook for restarting payments-api?",
                   help="incident description (min 10 chars)")
    p.add_argument("--severity", default="low", choices=["low", "medium", "high", "critical"])
    p.add_argument("--user-id", default="peer_client", dest="user_id")
    p.add_argument("--card-only", action="store_true", help="fetch and validate the card, submit nothing")

    gate = p.add_mutually_exclusive_group()
    gate.add_argument("--approve", dest="mode", action="store_const", const="approve",
                      help="auto-approve at the human gate")
    gate.add_argument("--reject", dest="mode", action="store_const", const="reject",
                      help="auto-reject at the human gate")
    p.set_defaults(mode="prompt")

    args = p.parse_args()

    if not args.base_url:
        if not args.peer:
            p.error("give a base URL, or --peer <student_id> to resolve one from cohort.json")
        try:
            from app.cohort import load_cohort
            args.base_url = load_cohort().peer(args.peer).base_url
        except ImportError:
            p.error("run me from the repo root so `app.cohort` is importable")
        except (FileNotFoundError, ValueError, KeyError) as exc:
            p.error(f"cohort.json: {exc}")
        print(f"resolved --peer {args.peer} -> {args.base_url}")

    if not args.base_url.startswith(("http://", "https://")):
        p.error("base_url must start with http:// or https://")

    args.token = args.token or _resolve_token()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
