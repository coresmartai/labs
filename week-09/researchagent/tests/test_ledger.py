"""The three properties your ledger needs.

These fail against NaiveLedger. Run them before you write anything, so you have
seen the failure rather than taken my word for it:

    pytest -q tests/test_ledger.py

Then implement EvidenceLedger, change the one line in app/main.py, and run them
again. When these pass, line 3 of the rubric is yours and line 1 becomes
possible.
"""
from __future__ import annotations

import pytest

from app.main import build_ledger
from tests.fixtures import CHUNKS, hits


def test_ids_are_stable_across_rounds():
    """Property 1. An id names the same passage for the whole run.

    Two queries in two rounds. The naive ledger issues doc#1 for the first hit
    of each, so the second overwrites the first and a claim citing doc#1 from
    round 0 silently resolves to round 1's passage.
    """
    ledger = build_ledger()
    round0 = ledger.record(hits(CHUNKS[0], CHUNKS[1]), "dmarc alignment", 0, 0)
    first_id = round0[0].ledger_id
    resolved_before = ledger.get(first_id).chunk_uid

    ledger.record(hits(CHUNKS[2], CHUNKS[3]), "spf check_host", 1, 0)
    resolved_after = ledger.get(first_id).chunk_uid

    assert resolved_before == resolved_after, (
        f"{first_id} resolved to {resolved_before} in round 0 and {resolved_after} "
        "in round 1. A citation that changes meaning between rounds is not a citation."
    )


def test_same_passage_recorded_once():
    """Property 2. Two queries returning the same passage give one entry, one id."""
    ledger = build_ledger()
    a = ledger.record(hits(CHUNKS[0], CHUNKS[1]), "identifier alignment", 0, 0)
    b = ledger.record(hits(CHUNKS[1], CHUNKS[0]), "from domain match", 1, 0)

    ids_for_chunk0 = {e.ledger_id for e in a + b if e.chunk_uid == CHUNKS[0].chunk_uid}
    assert len(ids_for_chunk0) == 1, (
        f"{CHUNKS[0].chunk_uid} was given {len(ids_for_chunk0)} ids: {sorted(ids_for_chunk0)}. "
        "One passage, one identity."
    )
    assert len({e.chunk_uid for e in ledger.entries()}) == len(ledger.entries()), (
        "The ledger holds duplicate entries for the same chunk."
    )


def test_entries_record_their_provenance():
    """Property 3. Every entry knows the round that found it and every query that did."""
    ledger = build_ledger()
    ledger.record(hits(CHUNKS[0]), "identifier alignment", 0, 0)
    ledger.record(hits(CHUNKS[0]), "from domain match", 3, 0)

    entry = next(e for e in ledger.entries() if e.chunk_uid == CHUNKS[0].chunk_uid)
    assert entry.first_seen_iter == 0, (
        f"first_seen_iter is {entry.first_seen_iter}; the passage was first surfaced in round 0."
    )
    assert set(entry.queries) == {"identifier alignment", "from domain match"}, (
        f"queries is {entry.queries}; both queries returned this passage and both should be here."
    )


def test_unknown_id_does_not_resolve():
    """A citation the ledger never issued must come back as None, not as something."""
    ledger = build_ledger()
    ledger.record(hits(CHUNKS[0]), "identifier alignment", 0, 0)
    assert ledger.get("ev-does-not-exist") is None
    assert ledger.get("") is None
