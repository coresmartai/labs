"""What the validator catches, and what it lets through."""
from __future__ import annotations

from app.schemas import Brief, BriefSection, Claim, Chunk, LedgerEntry
from app.validator import validate_brief
from tests.fixtures import CHUNKS


class _Ledger:
    def __init__(self, entries): self._e = {e.ledger_id: e for e in entries}
    def get(self, lid): return self._e.get(lid)
    def entries(self): return list(self._e.values())
    def record(self, hits, query, iteration, question_index): raise NotImplementedError


class _Index:
    def __init__(self, chunks): self._c = {c.chunk_uid: c for c in chunks}
    def get(self, uid): return self._c.get(uid)


def _entry(lid, chunk: Chunk, quote=None):
    return LedgerEntry(ledger_id=lid, chunk_uid=chunk.chunk_uid, rfc=chunk.rfc,
                       section=chunk.section, page=chunk.page,
                       quote=quote if quote is not None else chunk.text[:60],
                       question_index=0, first_seen_iter=0, queries=["q"])


def _brief(*claims, finding="answered", gap=""):
    return Brief(sections=[BriefSection(question="q", answerability="answerable",
                                        finding=finding, claims=list(claims), gap_note=gap)])


def test_a_supported_claim_passes():
    led = _Ledger([_entry("ev-001", CHUNKS[0])])
    r = validate_brief(_brief(Claim(text="Alignment matters.", ledger_ids=["ev-001"])),
                       led, _Index(CHUNKS))
    assert r.passed and r.claims_valid == 1


def test_an_uncited_claim_fails():
    led = _Ledger([_entry("ev-001", CHUNKS[0])])
    r = validate_brief(_brief(Claim(text="Alignment matters.", ledger_ids=[])),
                       led, _Index(CHUNKS))
    assert not r.passed
    assert r.claims_cited == 0
    assert "no citation" in r.verdicts[0].reason


def test_a_fabricated_id_fails():
    led = _Ledger([_entry("ev-001", CHUNKS[0])])
    r = validate_brief(_brief(Claim(text="Alignment matters.", ledger_ids=["ev-999"])),
                       led, _Index(CHUNKS))
    assert not r.passed
    assert "never issued" in r.verdicts[0].reason


def test_a_quote_that_is_not_in_the_chunk_fails():
    led = _Ledger([_entry("ev-001", CHUNKS[0], quote="DMARC requires TLS on port 25.")])
    r = validate_brief(_brief(Claim(text="Alignment matters.", ledger_ids=["ev-001"])),
                       led, _Index(CHUNKS))
    assert not r.passed
    assert "verbatim substring" in r.verdicts[0].reason


def test_whitespace_differences_do_not_fail_a_real_quote():
    quote = "  Identifier   alignment REQUIRES\n the RFC5322.From domain  "
    led = _Ledger([_entry("ev-001", CHUNKS[0], quote=quote)])
    r = validate_brief(_brief(Claim(text="Alignment matters.", ledger_ids=["ev-001"])),
                       led, _Index(CHUNKS))
    assert r.passed, [v.reason for v in r.verdicts if not v.ok]


def test_a_declined_section_needs_a_gap_note():
    led = _Ledger([_entry("ev-001", CHUNKS[0])])
    ok = _brief(Claim(text="x", ledger_ids=["ev-001"]), finding="declined",
                gap="The corpus does not cover deployment statistics.")
    bad = _brief(Claim(text="x", ledger_ids=["ev-001"]), finding="declined", gap="")
    assert validate_brief(ok, led, _Index(CHUNKS)).passed
    r = validate_brief(bad, led, _Index(CHUNKS))
    assert not r.passed
    assert "gap_note" in r.verdicts[-1].reason
