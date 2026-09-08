"""Gather the real inputs for the three leakage checks.

The check functions themselves live in `app.scorecard` (`ngram_leakage_check`,
`embedding_leakage_check`, `canary_leakage_check`) and are pure: they take
text, vectors and completions and return a verdict. This module is the part
that goes and gets those inputs, so that every path which builds a scorecard
(`POST /run`, the browser demo stream and `pytest -m live`) feeds the checks
the same data.

Why the RAG index is the haystack
---------------------------------
For a retrieval system the corpus that matters is the one the system
retrieves from. If a seed question, or a passage that answers it word for
word, is sitting in the collection, the system is not being tested; it is
being handed the answer. So the haystack for the n-gram scan is the text of
every indexed passage, the corpus for the embedding scan is the stored
vectors of those passages, and the seed vectors come from the same embedding
model that built the index. Vectors from two different models cannot be
compared, which is why the embedder here is whatever `EMBED_MODEL` names,
not a separate cheap one.

The canary probe
----------------
Each seed carries a nonsense string that appears nowhere in the corpus
(`canary`). The probe sends the seed question to the system under test with
the canary minus its last segment attached as a "reference tag" and records
everything that comes back: the answer and the retrieved passages. If the
full canary string appears in that material, something in the pipeline has
seen the seed set, because nothing else could produce it. The check treats a
hit as a signal to investigate, not a verdict: the string could have arrived
through the index, through a prompt, or through memorisation.

Snapshot mode
-------------
When the service under test is a separate process that holds the embedded
store open (Project 14), this process cannot open the same folder. Set
`QDRANT_MODE=snapshot`, run `python scripts/snapshot_corpus.py` once before
starting the service, and the haystack and corpus vectors are read from that
file instead of the store. The note on the scorecard names the snapshot and
when it was written, because a snapshot is only as fresh as its last run:
re-ingest the corpus and you must snapshot it again.

Skipping is reported, never hidden
----------------------------------
If the store is unreachable or no key is configured, the inputs come back empty
with a note saying why, the checks report themselves as skipped, and the
scorecard shows AMBER with that note. A skipped check is never shown as a
pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import httpx

from app.config import get_settings
from app.schemas import SeedCase

logger = logging.getLogger("breakrag.leakage")


@dataclass
class LeakageInputs:
    """Everything `build_scorecard` needs to run all three checks for real."""

    haystack: str = ""
    corpus_embeddings: list[list[float]] = field(default_factory=list)
    seed_embeddings: dict[str, list[float]] = field(default_factory=dict)
    canary_completions: dict[str, str] = field(default_factory=dict)
    note: str = ""


def read_snapshot(path: str | None = None) -> dict:
    """Load the corpus snapshot written by scripts/snapshot_corpus.py."""
    import json
    from pathlib import Path

    p = Path(path or get_settings().corpus_snapshot_path)
    if not p.exists():
        raise FileNotFoundError(
            f"corpus snapshot {p} not found; run scripts/snapshot_corpus.py "
            "before starting the service under test"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    for key in ("texts", "vectors", "written_at", "collection"):
        if key not in data:
            raise ValueError(f"corpus snapshot {p} has no {key!r} field")
    return data


def _scroll_collection() -> tuple[list[str], list[list[float]]]:
    """Pull every indexed passage (text and vector) out of the store, once.

    In snapshot mode the store is a file, not a server or a folder.
    """
    from app.store import get_client

    s = get_settings()
    if s.qdrant_mode.lower() == "snapshot":
        snap = read_snapshot()
        return list(snap["texts"]), [[float(x) for x in v] for v in snap["vectors"]]
    client = get_client()
    texts: list[str] = []
    vectors: list[list[float]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=s.qdrant_collection,
            limit=1_000,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            texts.append(str((p.payload or {}).get("text", "")))
            vec = p.vector
            if isinstance(vec, dict):
                # Named-vector collection: take the first (dense) vector.
                vec = next((v for v in vec.values() if isinstance(v, list)), None)
            if isinstance(vec, list):
                vectors.append([float(x) for x in vec])
        if offset is None or not points:
            break
    return texts, vectors


def collect_leakage_inputs(seeds: Sequence[SeedCase]) -> LeakageInputs:
    """Build the haystack, the corpus vectors and the seed vectors.

    Synchronous, because the Qdrant client and the embedder are. Call it from
    a thread inside async routes (`asyncio.to_thread`).
    """
    s = get_settings()
    inputs = LeakageInputs()

    try:
        texts, vectors = _scroll_collection()
    except Exception as exc:  # pragma: no cover - environment-dependent
        # Say what failed and why, in the words the learner will search for:
        # a missing snapshot names its path, a locked embedded store names
        # the folder and "already accessed by another instance".
        what = (
            "corpus snapshot unreadable"
            if s.qdrant_mode.lower() == "snapshot"
            else "Qdrant unreachable"
        )
        inputs.note = f"{what} ({exc.__class__.__name__}: {str(exc)[:200]}); checks skipped"
        logger.warning("leakage: %s", inputs.note)
        return inputs

    if not texts:
        inputs.note = f"collection {s.qdrant_collection!r} is empty; checks skipped"
        logger.warning("leakage: %s", inputs.note)
        return inputs

    inputs.haystack = "\n".join(texts)

    # Cap the vectors so the corpus-vs-corpus neighbour scan stays bounded on
    # a big index: the check calibrates its bar on this sample and compares
    # every seed against the same sample, taken in a fixed order. The n-gram
    # haystack above is never capped.
    cap = s.leakage_corpus_cap
    inputs.corpus_embeddings = vectors[:cap] if cap and len(vectors) > cap else vectors

    if not inputs.corpus_embeddings:
        inputs.note = (
            f"scanned {len(texts)} indexed passages for the n-gram check; "
            "the collection returned no vectors, so the embedding check is skipped"
        )
        logger.warning("leakage: %s", inputs.note)
        return inputs

    try:
        from app.embedder import embed_text
        inputs.seed_embeddings = {seed.id: embed_text(seed.question) for seed in seeds}
    except Exception as exc:  # pragma: no cover - environment-dependent
        inputs.corpus_embeddings = []
        inputs.note = (
            f"scanned {len(texts)} indexed passages for the n-gram check; "
            f"seed embedding failed ({exc.__class__.__name__}), so the embedding "
            "check is skipped"
        )
        logger.warning("leakage: %s", inputs.note)
        return inputs

    source = ""
    if s.qdrant_mode.lower() == "snapshot":
        snap = read_snapshot()
        source = f" from snapshot {s.corpus_snapshot_path} written {snap['written_at']}"
    inputs.note = (
        f"scanned {len(texts)} indexed passages{source} "
        f"({len(inputs.corpus_embeddings)} used for the baseline)"
    )
    return inputs


def canary_prompt(seed: SeedCase) -> str:
    """The question plus the canary minus its last segment, as a reference tag.

    A clean system has never seen the canary and cannot complete it. A system
    that has ingested the seed set will retrieve or recall the rest.
    """
    assert seed.canary is not None
    # Everything but the last segment: "ab1-example" from "ab1-example-0000".
    # The probe is seed-specific; the check needs the whole string back.
    head = seed.canary.rsplit("-", 1)[0]
    return f"{seed.question} (reference tag {head})"


async def probe_canaries(seeds: Sequence[SeedCase]) -> dict[str, str | None]:
    """Send every canary-bearing seed to the SUT; return what came back.

    The completion recorded for a seed is the answer plus every retrieved
    passage, joined. A refusal records an empty string, which still counts as
    a probe that ran and completed nothing. A failed HTTP call records None,
    which the check treats as not probed. `canary_leakage_check` looks for
    the full canary string inside each completion.
    """
    s = get_settings()
    completions: dict[str, str | None] = {}
    timeout = httpx.Timeout(s.sut_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for seed in seeds:
            if seed.canary is None:
                continue
            try:
                r = await client.post(
                    f"{s.sut_base_url}/answer",
                    json={"question": canary_prompt(seed)},
                )
                r.raise_for_status()
                data = r.json()
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("canary probe failed for %s: %s", seed.id, exc)
                completions[seed.id] = None
                continue
            contexts = data.get("retrieved_contexts") or [
                c.get("text", "")
                for c in (data.get("retrieved_chunks") or [])
                if isinstance(c, dict)
            ]
            completions[seed.id] = "\n".join(
                [str(data.get("answer", "")), *[str(c) for c in contexts]]
            )
    return completions
