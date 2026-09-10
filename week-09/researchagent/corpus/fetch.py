#!/usr/bin/env python3
"""Fetch the seven RFCs, hash them, and write the manifest.

The corpus is NOT committed. The IETF Trust Legal Provisions grant the right to
distribute IETF documents "in full and without modification", with no
derivative-works right outside the standards process, so a chunked or cleaned
copy in a public repository would breach it.

So the documents are fetched, and what you commit is `manifest.json`: seven
SHA-256 digests. That does three jobs at once. Nothing is redistributed. Your
reviewer runs this same script and gets byte-identical files. And the manifest
proves which bytes your run was built on, which is the same discipline as
pinning a model to a dated snapshot.

Usage:
    python corpus/fetch.py            # fetch, hash, write manifest
    python corpus/fetch.py --verify   # re-hash what is on disk, compare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://www.rfc-editor.org/rfc"
HERE = Path(__file__).resolve().parent
# RESEARCHAGENT_RFC_DIR lets a reviewer fetch once and verify many clones
# against the same files. Unset, the corpus lives beside this script.
RFC_DIR = Path(os.environ.get("RESEARCHAGENT_RFC_DIR", HERE / "rfc"))
MANIFEST = HERE / "manifest.json"

# Seven documents, one topic. They interlock: DMARC alignment is meaningless
# without SPF and DKIM, and ARC exists only because forwarding breaks both.
#
# Note 9989/9990/9991. RFC 7489 was the DMARC specification until 19 May 2026
# and is now obsolete. Your model was trained on 7489. The corpus is the truth.
RFCS = [
    ("rfc7208", "Sender Policy Framework (SPF) for Authorizing Use of Domains in Email, Version 1"),
    ("rfc6376", "DomainKeys Identified Mail (DKIM) Signatures"),
    ("rfc8601", "Message Header Field for Indicating Message Authentication Status"),
    ("rfc8617", "The Authenticated Received Chain (ARC) Protocol"),
    ("rfc9989", "Domain-based Message Authentication, Reporting, and Conformance (DMARC)"),
    ("rfc9990", "DMARC Aggregate Reporting"),
    ("rfc9991", "DMARC Failure Reporting"),
]

USER_AGENT = "CoreSmart-ResearchAgent-coursework/1.0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def fetch_one(name: str) -> Path:
    dest = RFC_DIR / f"{name}.txt"
    if dest.exists():
        print(f"  have  {name}.txt")
        return dest
    url = f"{BASE}/{name}.txt"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Could not fetch {url}\n  {exc}\n"
            "  RFC URLs are immutable, so this is a network problem rather than a\n"
            "  missing document. Check your connection or any proxy, and retry."
        )
    if len(body) < 10_000:
        raise SystemExit(f"{url} returned {len(body)} bytes, which is too small to be an RFC.")
    dest.write_bytes(body)
    print(f"  got   {name}.txt  ({len(body):,} bytes)")
    return dest


def write_manifest() -> dict:
    RFC_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    print(f"Fetching {len(RFCS)} RFCs into {RFC_DIR.relative_to(Path.cwd()) if RFC_DIR.is_relative_to(Path.cwd()) else RFC_DIR}")
    for name, title in RFCS:
        path = fetch_one(name)
        entries.append({
            "rfc": name,
            "title": title,
            "url": f"{BASE}/{name}.txt",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "corpus": "ietf-email-authentication",
        "count": len(entries),
        "documents": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {MANIFEST.name}. Commit it. Do NOT commit corpus/rfc/.")
    return manifest


def verify() -> int:
    if not MANIFEST.exists():
        print("No manifest.json. Run without --verify first.", file=sys.stderr)
        return 1
    manifest = json.loads(MANIFEST.read_text())
    bad = 0
    for entry in manifest["documents"]:
        path = RFC_DIR / f"{entry['rfc']}.txt"
        if not path.exists():
            print(f"  MISSING  {entry['rfc']}.txt")
            bad += 1
            continue
        actual = sha256(path)
        if actual == entry["sha256"]:
            print(f"  ok       {entry['rfc']}.txt")
        else:
            print(f"  CHANGED  {entry['rfc']}.txt\n    manifest {entry['sha256']}\n    on disk  {actual}")
            bad += 1
    print("\nAll seven match the manifest." if not bad else f"\n{bad} document(s) do not match the manifest.")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="re-hash what is on disk and compare")
    args = ap.parse_args()
    sys.exit(verify() if args.verify else (write_manifest() and 0))
