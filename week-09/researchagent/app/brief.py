"""Assemble the brief from what the research rounds put in the ledger.

One model call per question, after the loop for that question has stopped. It
sees the question, the loop's own closing text, and the ledger entries that
question surfaced, and it returns claims with citations.

The prompt does two jobs and the second is the one that carries marks: it makes
declining an available, named outcome rather than something the model has to
invent under pressure.
"""
from __future__ import annotations

import json

from app.llm import call_with_tools
from app.schemas import Brief, BriefSection, Claim, LedgerEntry

ASSEMBLE_PROMPT = """You are writing one section of a research brief.

You are given a question, the notes from the research pass, and the evidence ledger for that \
question. Every entry has a ledger_id, a quote and the document it came from.

Return JSON with exactly these keys:
  "finding":  one of "answered", "partial", "declined"
  "claims":   a list of {"text": "...", "ledger_ids": ["..."]}
  "gap_note": a sentence, REQUIRED when finding is "partial" or "declined"

Rules:
- Every claim cites at least one ledger_id. A sentence you cannot cite is a sentence you \
delete, not one you leave uncited.
- Cite only ledger_ids that appear in the evidence below. Do not invent one.
- "declined" is correct when the evidence does not answer the question. Say what the corpus \
does not cover in gap_note. Do not answer from your own knowledge of these protocols.
- "partial" is correct when you can answer some of the question. Name the missing part in \
gap_note.
- Claims are sentences, not paragraphs. Six to ten for an answered section is plenty."""


def assemble_section(question: str, answerability: str, notes: str,
                     evidence: list[LedgerEntry]) -> BriefSection:
    ledger_view = [
        {"ledger_id": e.ledger_id, "rfc": e.rfc, "section": e.section,
         "page": e.page, "quote": e.quote}
        for e in evidence
    ]
    user = json.dumps({
        "question": question,
        "research_notes": notes,
        "evidence": ledger_view,
    }, indent=2)

    resp = call_with_tools(system_prompt=ASSEMBLE_PROMPT,
                           messages=[{"role": "user", "content": user}],
                           tools=[])
    text = "\n".join(b["text"] for b in resp.content_blocks if b["type"] == "text")
    try:
        parsed = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return BriefSection(question=question, answerability=answerability,
                            finding="declined",
                            gap_note="The assembler did not return parseable JSON.")

    return BriefSection(
        question=question,
        answerability=answerability,
        finding=parsed.get("finding", "declined"),
        claims=[Claim(text=c.get("text", ""), ledger_ids=c.get("ledger_ids", []))
                for c in parsed.get("claims", [])],
        gap_note=parsed.get("gap_note", ""),
    )


def render_markdown(brief: Brief) -> str:
    out = ["# Research brief", ""]
    for section in brief.sections:
        out.append(f"## {section.question}")
        out.append("")
        out.append(f"*Declared: {section.answerability}. Finding: {section.finding}.*")
        out.append("")
        for claim in section.claims:
            cites = " ".join(f"[{lid}]" for lid in claim.ledger_ids)
            out.append(f"- {claim.text} {cites}".rstrip())
        if section.gap_note:
            out.append("")
            out.append(f"**What the corpus does not settle:** {section.gap_note}")
        out.append("")
    return "\n".join(out)
