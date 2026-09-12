"""Is the peer up, and is its card the shape this bridge needs?

    python scripts/peer_up.py                       # uses PEER_BASE_URL
    python scripts/peer_up.py http://localhost:8001

Run this FIRST, every time. Ninety percent of "ToolBridge is broken" is a peer
that is not running, and the other ten percent is a peer serving a card shape
from before A2A v1.0.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app import a2a  # noqa: E402
from app.config import get_settings  # noqa: E402


async def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else get_settings().peer_base_url
    print(f"checking {base}")
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            card = await a2a.fetch_card(client, base)
        except a2a.PeerError as exc:
            print(f"  FAIL  {exc.code}: {exc.message}")
            print("\n  Start a peer with, from the agentmesh directory:")
            print("    STUDENT_ID=stu_001 STUDENT_NAME=Alice AUTH_MODE=dev \\")
            print("      AGENTMESH_BASE_URL=http://localhost:8001 \\")
            print("      uvicorn app.main:app --port 8001")
            return 1
        print(f"  ok    card served: {card.get('name')} v{card.get('version')}")
        try:
            iface = a2a.select_interface(card)
        except a2a.PeerError as exc:
            print(f"  FAIL  {exc.code}: {exc.message}")
            return 1
        rank = card["supportedInterfaces"].index(iface)
        print(f"  ok    interface[{rank}]: {iface['protocolBinding']} {iface['url']}")
        skills = [s.get("id") for s in card.get("skills") or []]
        print(f"  ok    skills: {skills}")
        if "triage_incident" not in skills:
            print("  FAIL  this peer does not offer triage_incident")
            return 1
        print(f"  note  signatures: {'present' if card.get('signatures') else 'none, proceeding unsigned'}")
    print("\n  peer looks usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
