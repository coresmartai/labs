#!/usr/bin/env python3
"""
PaperFinder benchmark: recall@k and latency for KnowledgeVault over the
five-paper corpus.

    python bench/eval.py                          # given golden set, k=5
    python bench/eval.py --set mine               # your own queries
    python bench/eval.py --set all --k 10 --widen # everything, widened, k=10
    python bench/eval.py --set all --label baseline   # writes bench/results_baseline.json
    python bench/eval.py --quick                  # three queries, for the reviewer

Run it against a server that is already up (uvicorn app.main:app) with all
five PDFs ingested. It only ever calls GET /stats and POST /retrieve, so it
works against your KnowledgeVault whether or not you changed the internals.

What counts as a hit
--------------------
A query is a hit if ANY chunk in the top-k satisfies all of:
  1. chunk.document_id == expected_document_id
  2. chunk.chunk_type  == expected_chunk_type   (only if the query sets one)
  3. any term in must_contain appears in chunk.text, case-insensitive
     (only if must_contain is non-empty; for widened prose hits the stitched
     parent_text is searched as well)

"Unreachable" figure queries
----------------------------
KnowledgeVault's parser extracts figures that are embedded as raster images.
A figure drawn as vector graphics inside the PDF never becomes a figure
chunk. If GET /stats shows zero figure-description chunks for the paper a
figure query targets, the query cannot be hit with the stock parser through
no fault of your retrieval. It is reported as `unreachable`, excluded from
the recall denominator, and listed separately, so the number you commit is
about retrieval and the gap is visible rather than hidden inside a miss.
If you change the parser so those figures are seen, they stop being
unreachable and start counting. That is a legitimate "one change" for the
design memo.

Outputs
-------
Prints a table and writes bench/results_<label>.json with the config, the
per-query rows, and the summary. Commit the JSON; the memo cites it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIVEN = HERE / "golden_queries.jsonl"
MINE = HERE / "my_queries.jsonl"
QUICK_IDS = {"g01", "g03", "g06"}


def _http(method: str, url: str, body: dict | None = None, timeout: float = 60.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def load_queries(which: str) -> list[dict]:
    paths = {"given": [GIVEN], "mine": [MINE], "all": [GIVEN, MINE]}[which]
    out: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"missing query file: {p}", file=sys.stderr)
            sys.exit(2)
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{p.name}:{n}: bad JSON ({e})", file=sys.stderr)
                sys.exit(2)
            for key in ("id", "query", "expected_document_id", "modality"):
                if key not in q:
                    print(f"{p.name}:{n}: missing field {key!r}", file=sys.stderr)
                    sys.exit(2)
            q["_source"] = "given" if p == GIVEN else "mine"
            out.append(q)
    return out


def is_hit(q: dict, chunk: dict) -> bool:
    if chunk.get("document_id") != q["expected_document_id"]:
        return False
    want_type = q.get("expected_chunk_type")
    if want_type and chunk.get("chunk_type") != want_type:
        return False
    terms = [t.lower() for t in q.get("must_contain", []) if t]
    if not terms:
        return True
    hay = (chunk.get("text") or "").lower()
    if chunk.get("parent_text"):
        hay += " " + chunk["parent_text"].lower()
    return any(t in hay for t in terms)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round((pct / 100.0) * (len(s) - 1))))
    return s[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000", help="KnowledgeVault base URL")
    ap.add_argument("--set", choices=["given", "mine", "all"], default="given")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--widen", action="store_true", help="send widen=true on every request")
    ap.add_argument("--label", default=None, help="results file suffix; default is <set>-k<k>[-widen]")
    ap.add_argument("--quick", action="store_true", help="three given queries only; for the reviewer's smoke run")
    ap.add_argument("--no-write", action="store_true", help="print only, do not write results JSON")
    args = ap.parse_args()

    queries = load_queries("given" if args.quick else args.set)
    if args.quick:
        queries = [q for q in queries if q["id"] in QUICK_IDS]

    # Inventory first: which papers have any figure chunks at all.
    try:
        stats = _http("GET", f"{args.base}/stats")
    except (urllib.error.URLError, OSError) as e:
        print(f"cannot reach {args.base}/stats: {e}\nIs the server running? (uvicorn app.main:app)", file=sys.stderr)
        return 2
    docs = stats.get("documents", {})
    if not docs:
        print("the index is empty. Ingest the five PDFs first (POST /ingest/all?reset=true).", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for q in queries:
        expected_doc = q["expected_document_id"]
        want_type = q.get("expected_chunk_type")
        unreachable = False
        if want_type and docs.get(expected_doc, {}).get(want_type, 0) == 0:
            unreachable = True

        payload = {"query": q["query"], "k": args.k, "widen": args.widen}
        t0 = time.perf_counter()
        try:
            resp = _http("POST", f"{args.base}/retrieve", payload)
            err = None
        except urllib.error.HTTPError as e:
            resp, err = {"chunks": []}, f"HTTP {e.code}"
        except (urllib.error.URLError, OSError) as e:
            resp, err = {"chunks": []}, str(e)
        ms = (time.perf_counter() - t0) * 1000.0

        chunks = resp.get("chunks", [])
        hit_rank = None
        for i, c in enumerate(chunks, start=1):
            if is_hit(q, c):
                hit_rank = i
                break
        top_doc = chunks[0].get("document_id") if chunks else None
        rows.append({
            "id": q["id"], "source": q["_source"], "modality": q["modality"],
            "expected_document_id": expected_doc, "expected_chunk_type": want_type,
            "hit": hit_rank is not None, "hit_rank": hit_rank, "unreachable": unreachable,
            "top1_document_id": top_doc, "returned": len(chunks),
            "latency_ms": round(ms, 1), "error": err,
        })

    scored = [r for r in rows if not r["unreachable"]]

    def recall(subset: list[dict]) -> float | None:
        return round(sum(r["hit"] for r in subset) / len(subset), 3) if subset else None

    summary = {
        "n_queries": len(rows),
        "n_scored": len(scored),
        "n_unreachable": sum(r["unreachable"] for r in rows),
        f"recall_at_{args.k}": recall(scored),
        "recall_prose": recall([r for r in scored if r["modality"] == "prose"]),
        "recall_figure": recall([r for r in scored if r["modality"] == "figure"]),
        "recall_table": recall([r for r in scored if r["modality"] == "table"]),
        "recall_given": recall([r for r in scored if r["source"] == "given"]),
        "recall_mine": recall([r for r in scored if r["source"] == "mine"]),
        "mrr": round(statistics.mean((1.0 / r["hit_rank"]) if r["hit_rank"] else 0.0 for r in scored), 3) if scored else None,
        "latency_p50_ms": round(percentile([r["latency_ms"] for r in rows], 50), 1),
        "latency_p95_ms": round(percentile([r["latency_ms"] for r in rows], 95), 1),
        "errors": sum(1 for r in rows if r["error"]),
    }

    # -- print --
    print(f"\nPaperFinder benchmark  set={args.set if not args.quick else 'quick'}  k={args.k}  widen={args.widen}  base={args.base}")
    print(f"index: {stats.get('points')} points over {len(docs)} documents")
    for d, c in sorted(docs.items()):
        print(f"  {d:<28} prose {c.get('prose',0):>4}  figure {c.get('figure-description',0):>3}  table {c.get('table-row',0):>4}")
    print()
    print(f"{'id':<5} {'src':<5} {'mod':<7} {'result':<12} {'rank':<5} {'top-1 doc':<28} {'ms':>7}")
    for r in rows:
        result = "unreachable" if r["unreachable"] else ("HIT" if r["hit"] else "miss")
        if r["error"]:
            result = f"error"
        print(f"{r['id']:<5} {r['source']:<5} {r['modality']:<7} {result:<12} {str(r['hit_rank'] or '-'):<5} {str(r['top1_document_id']):<28} {r['latency_ms']:>7.1f}")
    print()
    for key, val in summary.items():
        if val is not None:
            print(f"  {key:<16} {val}")
    if summary["n_unreachable"]:
        print("\n  unreachable = the target paper has no figure chunks in the index (vector-drawn figures). "
              "Excluded from recall. Explain it in the memo, or fix the parser and they start to count.")

    # -- write --
    if not args.no_write:
        label = args.label or (("quick" if args.quick else args.set) + f"-k{args.k}" + ("-widen" if args.widen else ""))
        out = HERE / f"results_{label}.json"
        out.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {"base": args.base, "set": args.set, "k": args.k, "widen": args.widen, "quick": args.quick},
            "index": {"points": stats.get("points"), "documents": docs},
            "summary": summary,
            "rows": rows,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
