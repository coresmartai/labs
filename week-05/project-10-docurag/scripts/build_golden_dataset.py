"""
Build the golden evaluation dataset from real Qdrant chunks.

Run once after Week 4 ingestion is complete:
    python scripts/build_golden_dataset.py

Writes:  data/golden_dataset.json
Costs:   TARGET_ANSWERABLE small LLM calls (default 8) using gpt-5.4-mini

Strategy
--------
1.  Scroll all points from the knowledgevault Qdrant collection.
2.  Filter:  prose / table-row chunks with >= MIN_CHARS characters
             (skips figure captions, page headers, single-word noise).
3.  Sample:  pick TARGET_ANSWERABLE chunks evenly spaced across the document
             so the dataset covers the full text, not just the beginning.
4.  Generate: ask the LLM to produce one answerable question + expected_answer
              per chunk.  Retries once on bad JSON.
5.  Append:  TARGET_UNANSWERABLE hard-coded off-topic rows.
6.  Write:   data/golden_dataset.json  (overwrites any previous file).

must_cite convention
--------------------
  []            -- expected behaviour is REFUSAL (not in document)
  ["_answered"] -- expected behaviour is ANSWER with at least one citation
                   chunk IDs are renumbered doc#1..k per query, so we use a
                   sentinel instead of fragile specific IDs.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

TARGET_ANSWERABLE   = 8    # how many LLM-generated Q&A rows to produce
TARGET_UNANSWERABLE = 2    # hard-coded off-topic refusal rows
MIN_CHARS           = 150  # minimum chunk text length to consider
MAX_PAGE            = 10   # skip chunks beyond this page (refs / appendix)
MIN_AVG_TOKEN_LEN   = 3.0  # avg chars per whitespace-token - filters garbled/reversed text
MAX_RETRIES         = 2    # retries per chunk on bad LLM JSON

# Licence / copyright boilerplate produces "valid-looking" chunks that generate
# useless eval questions ("Under what condition does Google grant permission to
# reproduce the tables?"). Any chunk matching this pattern is skipped.
BOILERPLATE_RE = re.compile(
    r"copyright|all rights reserved|permission to (make digital|reproduce)"
    r"|licen[cs]e|attribution[- ]noncommercial|creative commons",
    re.IGNORECASE,
)

# Questions clearly outside the document scope - always refused correctly.
UNANSWERABLE_ROWS = [
    {
        "question": "What is the current stock price of NVIDIA?",
        "expected_answer": "I don't have that information in the provided sources.",
        "must_cite": [],
    },
    {
        "question": "What is the lunchroom Wi-Fi password?",
        "expected_answer": "I don't have that information in the provided sources.",
        "must_cite": [],
    },
]

QA_PROMPT = """\
You are building an evaluation dataset for a RAG system.

Given the text chunk below, write ONE clear question that this chunk directly
and specifically answers.  Then write the expected answer (1-3 sentences,
using your own words but staying faithful to the chunk content).

Rules:
- The question must be answerable from THIS chunk alone.
- The answer must be factual - do not invent details absent from the chunk.
- Prefer specific, concrete questions over vague ones.
  Good: "What distance metric does scaled dot-product attention use?"
  Bad:  "What does this section discuss?"

Chunk text:
{chunk_text}

Respond as JSON only:
{{"question": "...", "expected_answer": "..."}}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _scroll_all_chunks(client, collection: str) -> list[dict]:
    """Return all Qdrant points as plain dicts, sorted by chunk_index."""
    all_points = []
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset

    logger.info("Scrolled %d total points from '%s'.", len(all_points), collection)

    # Convert to plain dicts; sort by chunk_index so sampling is document-ordered.
    records = []
    for p in all_points:
        pay = p.payload or {}
        records.append(pay)

    records.sort(key=lambda r: r.get("chunk_index", 0))
    return records


def _is_clean(text: str) -> bool:
    """Return True if the text looks like readable prose, not garbled PDF output."""
    tokens = text.split()
    if not tokens:
        return False
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    if avg_len < MIN_AVG_TOKEN_LEN:
        return False   # reversed / symbol-heavy text
    # Skip reference-section patterns: lines that look like "[N] Author, Title, year"
    # Heuristic: if more than 30% of tokens are 1-2 chars, it's noisy
    short_ratio = sum(1 for t in tokens if len(t) <= 2) / len(tokens)
    if short_ratio > 0.30:
        return False
    return True


def _filter_chunks(records: list[dict]) -> list[dict]:
    """Keep only high-quality prose / table-row chunks within the main document body."""
    good = []
    for r in records:
        if r.get("chunk_type") not in ("prose", "table-row"):
            continue
        text = r.get("text", "")
        if len(text) < MIN_CHARS:
            continue
        page = r.get("page_number", 0)
        if page > MAX_PAGE:
            continue          # skip references / appendix pages
        if not _is_clean(text):
            continue          # garbled or noisy text
        if BOILERPLATE_RE.search(text):
            continue          # licence / copyright boilerplate - useless eval rows
        good.append(r)

    logger.info(
        "After filtering: %d / %d chunks usable (type + length + page + cleanliness).",
        len(good), len(records),
    )
    return good


def _sample_evenly(records: list[dict], n: int) -> list[dict]:
    """Pick n chunks evenly spaced across the filtered list."""
    if len(records) <= n:
        return records
    step = len(records) / n
    return [records[int(i * step)] for i in range(n)]


def _generate_qa(openai_client, model: str, chunk_text: str) -> dict | None:
    """Ask the LLM to produce a question + expected_answer for this chunk."""
    import json as _json

    prompt = QA_PROMPT.format(chunk_text=chunk_text[:1500])  # cap context length

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                max_completion_tokens=300,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or "{}"
            payload = _json.loads(text)
            q = payload.get("question", "").strip()
            a = payload.get("expected_answer", "").strip()
            if q and a:
                return {"question": q, "expected_answer": a}
            logger.warning("Attempt %d: empty question or answer, retrying.", attempt)
        except (_json.JSONDecodeError, Exception) as exc:
            logger.warning("Attempt %d failed: %s", attempt, exc)

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load settings from .env
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.config import get_settings

    settings = get_settings()

    from openai import OpenAI

    from app.store import get_client

    # Same client factory as the app - embedded local by default.
    # One-process rule: run this script with the CitationRAG (and Week 4)
    # servers stopped, or the embedded store will refuse to open.
    qdrant = get_client()
    oai = OpenAI(api_key=settings.openai_api_key)

    # ── 1. Fetch & filter ────────────────────────────────────────────────────
    all_records = _scroll_all_chunks(qdrant, settings.qdrant_collection)
    usable      = _filter_chunks(all_records)

    if not usable:
        logger.error(
            "No usable chunks found in '%s'. "
            "Run Week 4 ingestion first.",
            settings.qdrant_collection,
        )
        sys.exit(1)

    # ── 2. Sample evenly across document ────────────────────────────────────
    sampled = _sample_evenly(usable, TARGET_ANSWERABLE)
    logger.info(
        "Sampled %d chunks (every ~%d of %d) for Q&A generation.",
        len(sampled),
        max(1, len(usable) // TARGET_ANSWERABLE),
        len(usable),
    )

    # ── 3. Generate Q&A rows ─────────────────────────────────────────────────
    rows: list[dict] = []
    for i, chunk in enumerate(sampled, start=1):
        text = chunk.get("text", "")
        logger.info(
            "[%d/%d] Generating Q&A for chunk %s (%d chars) …",
            i, len(sampled),
            chunk.get("chunk_id", "?"),
            len(text),
        )
        qa = _generate_qa(oai, settings.llm_model, text)
        if qa is None:
            logger.warning("Skipping chunk %s - could not generate Q&A.", chunk.get("chunk_id"))
            continue

        rows.append({
            "question":        qa["question"],
            "expected_answer": qa["expected_answer"],
            "must_cite":       ["_answered"],
            # metadata - not used by eval, useful for debugging
            "_source_chunk_id":    chunk.get("chunk_id"),
            "_source_chunk_type":  chunk.get("chunk_type"),
            "_source_page":        chunk.get("page_number"),
        })
        logger.info("  Q: %s", qa["question"])

    # ── 4. Append unanswerable rows ──────────────────────────────────────────
    rows.extend(UNANSWERABLE_ROWS)

    # ── 5. Write ─────────────────────────────────────────────────────────────
    out = Path("data")
    out.mkdir(exist_ok=True)
    out_path = out / "golden_dataset.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    n_answerable   = sum(1 for r in rows if r.get("must_cite") == ["_answered"])
    n_unanswerable = sum(1 for r in rows if r.get("must_cite") == [])

    print()
    print(f"Written {len(rows)} rows -> {out_path}")
    print(f"  answerable   : {n_answerable}")
    print(f"  unanswerable : {n_unanswerable}")
    print()
    print("Preview (first 3 questions):")
    for r in rows[:3]:
        print(f"  [{r['must_cite']}] {r['question']}")


if __name__ == "__main__":
    main()
