"""Sweep the whole cohort: who is up, and who is behind each agent.

Reads the cohort's published index (a static JSON file - GitHub Pages, a gist,
any world-readable URL), then asks every listed agent the same two questions:

    GET  /.well-known/agent-card.json   - unauthenticated discovery
    POST /tasks {"skill": "whoami"}     - authenticated identity, full task round trip

    python scripts/cohort_roster.py https://<org>.github.io/<repo>/index.json
    python scripts/cohort_roster.py <index-url> --cards-only     # skip the auth check

This is the tool for the live session: run it once and you know who is reachable,
whose tunnel has gone stale, and whose token does not match the cohort's - before
twenty people start debugging each other simultaneously.

Index format (see PUBLISHING.md):

    {"cohort": "week11",
     "students": [{"student_id": "stu_007", "student_name": "Priya Raman",
                   "base_url": "https://xxx.trycloudflare.com"}]}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

# Python puts THIS file's directory on sys.path, not the repo root, so `app.cohort`
# would not import however you invoked us. Add the root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TIMEOUT = 12.0


def normalise_index(data: Any) -> list[dict[str, Any]]:
    """Accept any of the three shapes a cohort index turns up in.

    1. A static site's index.json    -> {"students": [ {...}, ... ]}
    2. A bare array                  -> [ {...}, ... ]
    3. Firebase RTDB REST            -> {"swift-falcon-42": {...}, ...}

    Shape 3 is why this exists: the Realtime Database returns an object keyed by
    callsign, with no "students" wrapper, and the key IS the identity. Rather than
    force everyone to run a converter, the reader adapts.
    """
    if data is None:
        return []
    if isinstance(data, list):
        return [s for s in data if isinstance(s, dict)]
    if not isinstance(data, dict):
        raise SystemExit(f"cohort index is a {type(data).__name__}, not an object or array")

    if isinstance(data.get("students"), list):
        return [s for s in data["students"] if isinstance(s, dict)]

    rows: list[dict[str, Any]] = []
    for key, value in data.items():
        if key.startswith("_") or not isinstance(value, dict):
            continue
        row = dict(value)
        # RTDB keys by callsign; the rest of this script speaks student_id.
        row.setdefault("student_id", row.get("tag", key))
        row.setdefault("student_name", row.get("student_name", row.get("tag", key)))
        rows.append(row)
    if not rows:
        raise SystemExit("cohort index is empty - nobody has published a URL yet")
    return rows


async def _load_index(client: httpx.AsyncClient, url: str) -> list[dict[str, Any]]:
    r = await client.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return normalise_index(r.json())


async def _probe(client: httpx.AsyncClient, entry: dict[str, Any], token: str, cards_only: bool) -> dict[str, Any]:
    """One student: fetch the card, then (unless --cards-only) run whoami.

    Every failure is captured as a row rather than raised - one classmate with a
    dead tunnel must not abort the sweep for the other nineteen.
    """
    listed_id = entry.get("student_id", "?")
    base = (entry.get("base_url") or "").rstrip("/")
    row: dict[str, Any] = {"listed_id": listed_id, "listed_name": entry.get("student_name", "?"),
                           "base_url": base, "card": "-", "whoami": "-", "note": ""}
    if not base:
        row["note"] = "no base_url in the index"
        return row

    try:
        r = await client.get(f"{base}/.well-known/agent-card.json", timeout=TIMEOUT)
        r.raise_for_status()
        card = r.json()
        row["card"] = "ok"
        row["agent_name"] = card.get("name")
        skills = [s.get("id") for s in card.get("skills") or []]
        if "whoami" not in skills:
            row["note"] = "card offers no whoami skill"
    except httpx.HTTPError as exc:
        row["card"] = "FAIL"
        row["note"] = f"unreachable: {type(exc).__name__}"
        return row
    except json.JSONDecodeError:
        row["card"] = "FAIL"
        row["note"] = "card is not JSON (tunnel interstitial page?)"
        return row

    if cards_only or row["note"]:
        return row

    try:
        ack = await client.post(
            f"{base}/tasks",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"skill": "whoami", "input": {}},
            timeout=TIMEOUT,
        )
        if ack.status_code != 202:
            reason = ""
            try:
                reason = ack.json().get("detail", {}).get("reason", "")
            except Exception:  # noqa: BLE001 - a non-JSON error body is itself the signal
                pass
            row["whoami"] = f"HTTP {ack.status_code}"
            row["note"] = reason or "task refused"
            return row

        task_id = ack.json()["task_id"]
        async with client.stream(
            "GET", f"{base}/tasks/{task_id}/stream",
            headers={"Authorization": f"Bearer {token}"}, timeout=TIMEOUT,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                ev = json.loads(line[5:].strip())
                if ev["state"] == "TASK_STATE_COMPLETED":
                    p = ev.get("payload") or {}
                    row["whoami"] = "ok"
                    row["real_id"] = p.get("student_id")
                    row["real_name"] = p.get("student_name")
                    if p.get("student_id") and p["student_id"] != listed_id:
                        row["note"] = f"index says {listed_id}, agent says {p['student_id']}"
                    break
                if ev["state"] in ("TASK_STATE_FAILED", "TASK_STATE_REJECTED"):
                    row["whoami"] = ev["state"].replace("TASK_STATE_", "")
                    break
    except httpx.HTTPError as exc:
        row["whoami"] = "FAIL"
        row["note"] = f"{type(exc).__name__}"
    return row


async def run(index_url: str, token: str, cards_only: bool) -> int:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        students = await _load_index(client, index_url)
        print(f"cohort index: {len(students)} student(s) listed at {index_url}\n")
        # Everyone in parallel - a twenty-student sweep should take as long as the
        # slowest peer, not the sum of all of them.
        rows = await asyncio.gather(*[_probe(client, e, token, cards_only) for e in students])

    _render(rows, cards_only)
    return 0 if any(r["card"] == "ok" for r in rows) else 1


async def run_peers(peers: list[dict[str, Any]], token: str, cards_only: bool) -> int:
    """Solo mode: sweep the peers listed in cohort.json instead of a published index."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        print(f"solo mode: probing {len(peers)} peer(s) from cohort.json\n")
        rows = await asyncio.gather(*[_probe(client, e, token, cards_only) for e in peers])
    _render(rows, cards_only)
    return 0 if any(r["card"] == "ok" for r in rows) else 1


def _render(rows: list[dict[str, Any]], cards_only: bool) -> None:
    print(f"{'id':<10} {'name':<20} {'card':<6} {'whoami':<8} note")
    print("-" * 78)
    for r in sorted(rows, key=lambda x: str(x["listed_id"])):
        name = r.get("real_name") or r["listed_name"]
        print(f"{str(r['listed_id']):<10} {str(name)[:19]:<20} {r['card']:<6} {str(r['whoami']):<8} {r['note']}")
    reachable = sum(1 for r in rows if r["card"] == "ok")
    answered = sum(1 for r in rows if r["whoami"] == "ok")
    print(f"\n{reachable}/{len(rows)} card(s) reachable" + ("" if cards_only else f", {answered} answered whoami"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("index_url", nargs="?",
                   help="URL of the cohort's published index.json (default: from cohort.json)")
    p.add_argument("--token", default=os.environ.get("A2A_PEER_TOKEN", "demo-token"),
                   help="bearer for the whoami probe (default: $A2A_PEER_TOKEN, else demo-token)")
    p.add_argument("--cards-only", action="store_true",
                   help="only fetch Agent Cards - no auth, no task submitted")
    args = p.parse_args()

    index_url = args.index_url
    peers: list[dict[str, Any]] = []
    if not index_url:
        # No argument: fall back to cohort.json. Online mode has an index URL;
        # solo mode has a peers list and no index at all.
        try:
            from app.cohort import load_cohort
        except ImportError:
            print("run me from the repo root so `app.cohort` is importable, or pass an index URL",
                  file=sys.stderr)
            return 1
        try:
            c = load_cohort()
        except (FileNotFoundError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
        index_url = c.cohort_index_url
        peers = [pr.model_dump() for pr in c.peers]
        if not index_url and not peers:
            print(f"{c.mode} mode has neither cohort_index_url nor peers - "
                  f"nothing to sweep. See SESSION.md.", file=sys.stderr)
            return 1

    try:
        if index_url:
            return asyncio.run(run(index_url, args.token, args.cards_only))
        return asyncio.run(run_peers(peers, args.token, args.cards_only))
    except KeyboardInterrupt:
        return 130
    except httpx.HTTPError as exc:
        print(f"cannot read the index: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
