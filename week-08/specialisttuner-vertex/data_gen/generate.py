"""Rebuild data/aiayn_generic.jsonl from a Python source of Q&A tuples.

Usage:
    python data_gen/generate.py               # emits data/aiayn_generic.jsonl
    python data_gen/generate.py --convert     # also emits data/aiayn_gcloud.jsonl

The source lives in data_gen/aiayn_pairs.py as a list of (user, model) tuples.
Keeping the pairs in code (not YAML/CSV) means you can grep, refactor, and
review them like any other Python source. Every pair is a two-turn
conversation in the app's generic format:

    [{"role": "user", "message": "..."},
     {"role": "model", "message": "..."}]

The result is 210+ examples covering: motivation, encoder/decoder
architecture, scaled dot-product attention, multi-head attention,
positional encoding, feed-forward blocks, layer-norm + residuals,
regularisation, training details, results, and ablations from
"Attention Is All You Need" (Vaswani et al., 2017).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Make the app importable when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_gen.aiayn_pairs import PAIRS  # noqa: E402


def build_generic_example(user: str, model: str) -> dict:
    return {"messages": [
        {"role": "user",  "message": user},
        {"role": "model", "message": model},
    ]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild the AIAYN training set.")
    ap.add_argument("--out", default="data/aiayn_generic.jsonl",
                    help="Output path for the generic JSONL")
    ap.add_argument("--convert", action="store_true",
                    help="Also emit the Vertex-shaped JSONL alongside")
    ap.add_argument("--gcloud-out", default="data/aiayn_gcloud.jsonl",
                    help="Output path for the Vertex JSONL when --convert is set")
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for user, model in PAIRS:
            example = build_generic_example(user, model)
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"wrote {len(PAIRS)} examples → {out_path}")

    if args.convert:
        from app.convert import convert_file
        from prompts import load_system_instruction
        gcloud_out = ROOT / args.gcloud_out
        result = convert_file(out_path, gcloud_out,
                              system_instruction=load_system_instruction())
        print(f"converted {result['output_rows']} rows → {gcloud_out}")
        if result["warnings"]:
            print(f"warnings ({len(result['warnings'])}):")
            for w in result["warnings"][:10]:
                print(f"  - {w}")


if __name__ == "__main__":
    main()
