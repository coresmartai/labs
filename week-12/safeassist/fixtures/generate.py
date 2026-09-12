"""Synthetic fixtures, generated at test time and never committed.

Why this file exists rather than a fixtures/*.json you could read:

The cohort repository is shared, and the Week 7 project scans the whole of it.
Committing realistic patient records into it would be a genuine harm rather
than a hypothetical one, and "they are fake" is not a defence anyone can verify
from the outside: a reviewer looking at a committed file full of names,
addresses and medical record numbers cannot tell synthetic from real, and
neither can a scanner.

So three rules, and your graded project is marked against them:

  1. **Fixtures are generated, not stored.** Call `generate()` from a test or a
     script. The output directory is gitignored.
  2. **The generator is seeded.** The same seed produces the same records, so a
     failing test is reproducible without a committed file.
  3. **Every generated value carries a canary.** Each record embeds CANARY in a
     field, and every synthetic MRN uses the reserved 9xxx block. If a value
     with the canary ever appears in a commit, a log, or an answer, you know
     exactly where it came from. If a value WITHOUT the canary appears in your
     fixtures, something real got in.

The canary is also what the grader greps for. A submission whose fixtures are
committed, or whose fixtures lack the canary, fails this criterion regardless of
how good the rest of the code is.
"""
from __future__ import annotations

import json
import pathlib
import random
from typing import Any

# The marker every synthetic record carries. Deliberately ugly and searchable.
CANARY = "SYNTHETIC-W12-DO-NOT-SHIP"

# Synthetic MRNs use a reserved block that the seed corpus never uses, so a
# fixture identifier can never collide with one in the indexed documents.
MRN_BLOCK = 9000

_FIRST = ["Alex", "Robin", "Sam", "Jordan", "Casey", "Morgan", "Riley", "Quinn",
          "Avery", "Rowan", "Emery", "Finley"]
_LAST = ["Whitfield", "Okafor", "Lindqvist", "Marchetti", "Delacroix", "Aliyev",
         "Nakamura", "Oyelaran", "Petrov", "Halloran", "Sandoval", "Bergstrom"]
_NOTE = [
    "was referred to cardiology after two abnormal readings",
    "was discharged with a fourteen day course and a follow-up booked",
    "did not attend the first appointment and was sent a reminder",
    "had consent recorded before the procedure and scanned the same day",
]


def _record(rng: random.Random, i: int) -> dict[str, Any]:
    first = rng.choice(_FIRST)
    last = rng.choice(_LAST)
    mrn = f"MRN-{MRN_BLOCK + i:04d}-{rng.randint(1000, 9999)}"
    email = f"{first.lower()}.{last.lower()}@example.com"
    phone = f"+1-415-555-{rng.randint(100, 199):04d}"
    return {
        "canary": CANARY,
        "mrn": mrn,
        "name": f"{first} {last}",
        "email": email,
        "phone": phone,
        "text": (
            f"Referral note {mrn}: {first} {last} ({email}, {phone}) "
            f"{rng.choice(_NOTE)}. [{CANARY}]"
        ),
    }


def generate(n: int = 24, seed: int = 20261116) -> list[dict[str, Any]]:
    """Return n synthetic records. Deterministic for a given seed."""
    rng = random.Random(seed)
    return [_record(rng, i) for i in range(n)]


def write(path: str | pathlib.Path = "fixtures/generated/records.json", **kwargs: Any) -> pathlib.Path:
    """Write the records to a gitignored directory. For local inspection only."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(generate(**kwargs), indent=2), encoding="utf-8")
    return p


def assert_synthetic(records: list[dict[str, Any]]) -> None:
    """Fail loudly if anything in this set is not demonstrably synthetic.

    Call this before any fixture is used. It is cheap, and the failure it
    prevents is the one you cannot undo.
    """
    for r in records:
        if r.get("canary") != CANARY:
            raise AssertionError(f"record {r.get('mrn')!r} has no canary: it may not be synthetic")
        if CANARY not in r.get("text", ""):
            raise AssertionError(f"record {r.get('mrn')!r} text has no canary marker")
        if not r.get("mrn", "").startswith(f"MRN-{MRN_BLOCK // 1000}"):
            raise AssertionError(f"record {r.get('mrn')!r} is outside the reserved synthetic block")


if __name__ == "__main__":
    recs = generate()
    assert_synthetic(recs)
    out = write()
    print(f"wrote {len(recs)} synthetic records to {out} (gitignored)")
    print(f"canary: {CANARY}")
