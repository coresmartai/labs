"""Write a corpus snapshot for the leakage checks, so the harness never has to
open the store while another process holds it.

Why this exists
---------------
The three leakage checks need the indexed passages (the n-gram haystack) and
their stored vectors (the embedding baseline). In Project 13 the harness and
the service under test are one process, so the harness reads the store it is
already holding. In Project 14 the service under test is your own Week 5 or
Week 6 service, running in its own process, and if that service uses the
embedded store the folder is locked to it: a second process opening the same
`qdrant_local/` folder fails with "already accessed by another instance".

So: run this script once, while nothing else has the store open. It reads the
collection through the same client the harness uses, in local or server mode,
and writes the passages and vectors to one JSON file. Then set
`QDRANT_MODE=snapshot` in the harness's `.env`, start your service, start the
harness on another port, and the leakage checks read the file.

A snapshot is only as fresh as its last run. Re-ingest the corpus and run
this again; the scorecard note names the snapshot and when it was written so
a stale one is visible.

Usage
-----
    python scripts/snapshot_corpus.py                     # mode from .env
    python scripts/snapshot_corpus.py --mode local        # force the embedded folder
    python scripts/snapshot_corpus.py --mode server       # force QDRANT_URL
    python scripts/snapshot_corpus.py --out runs/my.json  # somewhere else

`--mode` overrides QDRANT_MODE for this run only. The output path defaults to
CORPUS_SNAPSHOT_PATH (runs/corpus_snapshot.json), which is gitignored: the
file carries your corpus text and must not be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["local", "server"], default=None,
                    help="override QDRANT_MODE for this run")
    ap.add_argument("--out", default=None,
                    help="output path (default: CORPUS_SNAPSHOT_PATH from .env)")
    args = ap.parse_args()

    if args.mode:
        os.environ["QDRANT_MODE"] = args.mode
    elif os.environ.get("QDRANT_MODE", "").lower() == "snapshot":
        # .env says snapshot because the harness will run that way; the
        # snapshot itself has to come from a live store.
        os.environ["QDRANT_MODE"] = "local"

    from app.config import get_settings
    from app.leakage import _scroll_collection

    s = get_settings()
    out = Path(args.out or s.corpus_snapshot_path)

    try:
        texts, vectors = _scroll_collection()
    except Exception as exc:  # noqa: BLE001 - one message for every store failure
        msg = str(exc)
        if "already accessed by another instance" in msg:
            hint = "another process holds the embedded store; stop your service and run this again"
        elif "not found" in msg.lower() and "collection" in msg.lower():
            hint = f"QDRANT_COLLECTION={s.qdrant_collection} is not in this store; check the name"
        elif "is not a folder" in msg:
            hint = "set QDRANT_LOCAL_PATH to your service's qdrant_local folder"
        else:
            hint = "check QDRANT_MODE, QDRANT_LOCAL_PATH or QDRANT_URL in .env"
        print(f"snapshot failed: {exc.__class__.__name__}: {msg[:200]}", file=sys.stderr)
        print(f"hint: {hint}", file=sys.stderr)
        return 1
    # Release the embedded store before the interpreter exits, so the service
    # under test can open it next, and so the client does not complain at
    # shutdown.
    from app import store
    if store._client is not None:
        store._client.close()
        store._client = None
    if not texts:
        print(f"collection {s.qdrant_collection!r} is empty; nothing written", file=sys.stderr)
        return 1

    cap = s.leakage_corpus_cap
    kept = vectors[:cap] if cap and len(vectors) > cap else vectors
    rounded = [[round(float(x), 5) for x in v] for v in kept]
    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:12]

    snapshot = {
        "collection": s.qdrant_collection,
        "mode": s.qdrant_mode,
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "passages": len(texts),
        "vectors_kept": len(rounded),
        "vector_dim": len(rounded[0]) if rounded else 0,
        "text_hash": digest,
        "texts": texts,
        "vectors": rounded,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot), encoding="utf-8")
    size_mb = out.stat().st_size / 1_000_000
    print(
        f"wrote {out}: {len(texts)} passages, {len(rounded)} vectors of "
        f"{snapshot['vector_dim']} dims, text hash {digest}, {size_mb:.1f} MB"
    )
    print("now set QDRANT_MODE=snapshot in .env before starting the harness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
