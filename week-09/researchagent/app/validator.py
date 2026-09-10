"""Validate a brief against the ledger. Deterministic. No model call.

Two checks, in order, and both are Week 5's, extended to a brief:

  1. RESOLUTION. Every ledger id a claim cites must be an id the ledger issued.
     Week 5 checked markers against one query's chunks. Here the legal set is
     the ledger, which is the union across every round of the run.

  2. QUOTE SUBSTRING. Each cited entry's quote must be a verbatim substring of
     the chunk it names, whitespace-normalised. A quote the model invented,
     or an entry pointing at the wrong chunk, fails here.

Why deterministic rather than a judge model: it is free, instant, and immune to
a judge's own hallucinations. Exact lookup always works for citation identity.
"""
from __future__ import annotations

import re

from app.ledger import LedgerProtocol
from app.retrieval import CorpusIndex
from app.schemas import Brief, ClaimVerdict, SectionCounts, ValidationReport

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def validate_brief(brief: Brief, ledger: LedgerProtocol, index: CorpusIndex) -> ValidationReport:
    verdicts: list[ClaimVerdict] = []
    per_section: list[SectionCounts] = []
    total = cited = valid = 0

    for si, section in enumerate(brief.sections):
        s_claims = s_cited = s_valid = 0
        for ci, claim in enumerate(section.claims):
            total += 1
            s_claims += 1
            if not claim.ledger_ids:
                verdicts.append(ClaimVerdict(
                    section=si, claim=ci, ok=False,
                    reason="no citation. Every claim in the brief must cite the ledger",
                ))
                continue
            cited += 1
            s_cited += 1

            problem = ""
            for lid in claim.ledger_ids:
                entry = ledger.get(lid)
                if entry is None:
                    problem = f"cites {lid}, which the ledger never issued"
                    break
                chunk = index.get(entry.chunk_uid)
                if chunk is None:
                    problem = f"{lid} points at chunk {entry.chunk_uid}, which is not in the index"
                    break
                if _norm(entry.quote) not in _norm(chunk.text):
                    problem = (f"{lid}'s quote is not a verbatim substring of "
                               f"{entry.chunk_uid}")
                    break

            if problem:
                verdicts.append(ClaimVerdict(section=si, claim=ci, ok=False, reason=problem))
            else:
                valid += 1
                s_valid += 1
                verdicts.append(ClaimVerdict(section=si, claim=ci, ok=True))

        per_section.append(SectionCounts(
            section=si, question=section.question, answerability=section.answerability,
            finding=section.finding, claims=s_claims, cited=s_cited, valid=s_valid))

    # A declined section carries no claims by design. It must carry a gap note.
    for si, section in enumerate(brief.sections):
        if section.finding in ("declined", "partial") and not section.gap_note.strip():
            verdicts.append(ClaimVerdict(
                section=si, claim=-1, ok=False,
                reason=f"finding is '{section.finding}' but no gap_note says what is missing",
            ))

    passed = all(v.ok for v in verdicts) and total > 0
    return ValidationReport(
        passed=passed,
        claims_total=total,
        claims_cited=cited,
        claims_valid=valid,
        per_section=per_section,
        verdicts=verdicts,
    )
