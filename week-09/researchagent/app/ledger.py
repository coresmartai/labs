"""The evidence ledger. THIS FILE IS THE PROJECT.

--------------------------------------------------------------------------
WHAT IS WRONG WITH THIS FILE, STATED PLAINLY
--------------------------------------------------------------------------

`NaiveLedger` below is Week 5's citation model, ported unchanged. Week 5's
validator rested on one sentence:

    "the retrieved chunks define the set of legal IDs"

That sentence is TRUE of a single response to a single query. It is FALSE of a
brief assembled from six research rounds, and this file is where it breaks.

`NaiveLedger.record()` numbers evidence per query: the first hit of any query
is `doc#1`. So `doc#1` from round one and `doc#1` from round four are different
passages wearing the same name. A claim in the brief that cites `doc#1` cannot
be resolved, because "which round?" is not part of the citation. The validator
will happily pass a claim whose evidence came from a completely different RFC.

Nothing crashes. Nothing logs. The brief looks correct and cites confidently.

--------------------------------------------------------------------------
WHAT YOU HAVE TO BUILD
--------------------------------------------------------------------------

Replace `NaiveLedger` with a ledger whose identities are stable for the whole
run. The interface below is what the rest of the package calls; keep it, or
change the callers too and say so in your README.

Three properties your implementation needs. `tests/test_ledger.py` holds four
tests: one per property, plus one that already passes and is there to stop a
ledger that resolves everything to something.

  1. STABLE. A `ledger_id` names the same passage from the moment it is issued
     until the run ends, whatever any later query returns.

  2. DEDUPLICATED. The same passage surfaced by two different queries is ONE
     ledger entry with ONE id, not two. `queries` accumulates.

  3. TRACEABLE. Every entry records the round that first surfaced it and every
     query that returned it. That is what lets a reviewer ask how a claim was
     arrived at, not just whether it is supported.

`tests/test_ledger.py` fails against the naive implementation and passes against
a correct one. Run it before you write anything, so you have seen it fail.
"""
from __future__ import annotations

from app.schemas import LedgerEntry, SearchHit


class LedgerProtocol:
    """The interface `agent.py` and `validator.py` call. Do not change these
    signatures without changing both callers and saying so in your README."""

    def record(self, hits: list[SearchHit], query: str, iteration: int,
               question_index: int) -> list[LedgerEntry]:
        """Take one query's hits into the ledger. Return the entries for those hits,
        whether newly created or already present."""
        raise NotImplementedError

    def get(self, ledger_id: str) -> LedgerEntry | None:
        """Resolve a citation. Return None if the id was never issued."""
        raise NotImplementedError

    def entries(self) -> list[LedgerEntry]:
        """Every entry, in the order the ids were issued."""
        raise NotImplementedError


class NaiveLedger(LedgerProtocol):
    """Week 5's model. Correct for one response, wrong for a brief.

    Left here on purpose so you can run it, watch it produce a brief that
    validates, and then check one of its citations by hand.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, LedgerEntry] = {}

    def record(self, hits: list[SearchHit], query: str, iteration: int,
               question_index: int) -> list[LedgerEntry]:
        out: list[LedgerEntry] = []
        for hit in hits:
            # The bug, in one line: the id is the rank within THIS query.
            ledger_id = f"doc#{hit.rank}"
            entry = LedgerEntry(
                ledger_id=ledger_id,
                chunk_uid=hit.chunk.chunk_uid,
                rfc=hit.chunk.rfc,
                section=hit.chunk.section,
                page=hit.chunk.page,
                quote=hit.chunk.text[:400],
                question_index=question_index,
                first_seen_iter=iteration,
                queries=[query],
            )
            self._by_id[ledger_id] = entry     # silently overwrites the earlier round
            out.append(entry)
        return out

    def get(self, ledger_id: str) -> LedgerEntry | None:
        return self._by_id.get(ledger_id)

    def entries(self) -> list[LedgerEntry]:
        return list(self._by_id.values())


# ---------------------------------------------------------------------------
# YOUR IMPLEMENTATION GOES HERE.
#
# class EvidenceLedger(LedgerProtocol):
#     ...
#
# Then change the one line in app/main.py that constructs the ledger.
# ---------------------------------------------------------------------------
