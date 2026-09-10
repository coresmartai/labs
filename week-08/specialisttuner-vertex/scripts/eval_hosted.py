"""Score a tuned Vertex endpoint on the 42-question AIAYN benchmark.

This is the fourth row of your decision memo. It exists as a script rather
than a route because this track has no browser page: the job runs in
somebody else's data centre and their console is the surface for it.

The scoring rule here is character-for-character the one the local package
uses in `POST /v1/eval/run`. That is the point. Two rows scored two
different ways are not comparable, and the whole week is built on the
comparison being fair.

Usage
-----
    # the tuned endpoint, printed by the tuning job when it succeeded
    python scripts/eval_hosted.py --endpoint projects/.../endpoints/1234

    # the untuned base model, for the row you compare against
    python scripts/eval_hosted.py --base

    # write the row where the memo can pick it up
    python scripts/eval_hosted.py --endpoint ... --out results/vertex_ft.json

Every call is a real, billed call to the service. Forty-two of them per run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings          # noqa: E402
from app.schemas import EvalResult           # noqa: E402


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Same implementation as the local package."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered) + 0.5) - 1))
    return round(ordered[idx], 1)


def load_eval(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score(rows: list[dict], endpoint: str | None) -> tuple[list[dict], float]:
    """Run every question and mark it. Identical rule to the local harness:
    a question passes if any accepted answer appears in the output."""
    from app.llm import generate

    results: list[dict] = []
    for i, row in enumerate(rows, 1):
        question = row["question"]
        accept = row.get("accept", [row["expected"]])
        try:
            out = generate(question, tuned_endpoint=endpoint,
                           temperature=0.0, max_output_tokens=64)
            got = out["output"].strip()
            got_lower = got.lower()
            passed = any(a.lower() in got_lower for a in accept)
            latency = out["latency_ms"]
        except Exception as exc:                      # noqa: BLE001
            got, passed, latency = f"ERROR: {exc}", False, 0.0
        results.append({
            "id": row.get("id"), "question": question,
            "expected": row["expected"], "got": got, "passed": passed,
            "category": row.get("category", "general"), "latency_ms": latency,
        })
        mark = "pass" if passed else "FAIL"
        print(f"  {i:2d}/{len(rows)}  {mark}  {question[:64]}")

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    accuracy = round(passed_count / total * 100, 1) if total else 0.0
    return results, accuracy


def memo_row(endpoint: str | None, accuracy: float,
             results: list[dict], cost_per_1k: float) -> EvalResult:
    """The one row this run contributes to the memo."""
    s = get_settings()
    lats = [float(r["latency_ms"]) for r in results if r["latency_ms"]]
    tuned = bool(endpoint)
    return EvalResult(
        approach="vertex_ft" if tuned else "base",
        model_id=endpoint if tuned else s.source_model,
        accuracy=accuracy,
        p50_latency_ms=percentile(lats, 50),
        p95_latency_ms=percentile(lats, 95),
        # You have to put this in yourself. It is the provider's published
        # price for this model, and it is the number the local row sets to
        # zero because you already own the hardware.
        cost_per_1k_calls_usd=cost_per_1k,
        # Vertex supervised tuning does not report a trainable-parameter
        # count. `adapter_size` is the rank knob, and it is not the same
        # number. Reporting zero here would be a lie, so the memo carries
        # the adapter size in prose instead.
        trainable_params=0,
        # The data and the weights are in the vendor's systems. That is the
        # honest posture, and it is half the argument of the memo.
        privacy_posture="vendor_zone",
    )


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--endpoint", help="tuned endpoint resource name")
    g.add_argument("--base", action="store_true",
                   help="score the untuned source model instead")
    ap.add_argument("--eval-path", default=str(s.eval_dataset_path))
    ap.add_argument("--cost-per-1k", type=float, default=0.0,
                    help="the provider's price per 1000 calls, for the memo row")
    ap.add_argument("--out", help="write the memo row to this JSON file")
    args = ap.parse_args()

    eval_path = Path(args.eval_path)
    if not eval_path.exists():
        print(f"Eval file not found: {eval_path}", file=sys.stderr)
        return 1

    endpoint = None if args.base else args.endpoint
    rows = load_eval(eval_path)
    target = endpoint if endpoint else f"{s.source_model} (untuned)"
    print(f"Scoring {len(rows)} questions against {target}")
    print("Every one of these is a billed call.\n")

    results, accuracy = score(rows, endpoint)
    row = memo_row(endpoint, accuracy, results, args.cost_per_1k)

    passed = sum(1 for r in results if r["passed"])
    print(f"\nAccuracy: {accuracy}%  ({passed}/{len(results)})")
    print(f"p50 {row.p50_latency_ms} ms, p95 {row.p95_latency_ms} ms")
    if args.cost_per_1k == 0.0:
        print("\ncost_per_1k_calls_usd is 0.0 because you did not pass "
              "--cost-per-1k. Look the price up and pass it, or the memo "
              "row is wrong in the one column the hosted path exists to fill.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row.model_dump(), indent=2) + "\n",
                       encoding="utf-8")
        print(f"\nMemo row written to {out}")
    else:
        print("\n" + json.dumps(row.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
