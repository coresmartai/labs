"""Things that must be true before you start, and still true when you finish."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.retrieval import CorpusIndex
from app.schemas import Chunk
from app.tools import build_tools, catalog_for_model, execute_tool
from tests.fixtures import CHUNKS

ROOT = Path(__file__).resolve().parent.parent


class _Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._by_uid = {c.chunk_uid: c for c in chunks}

    @property
    def documents(self):
        seen = []
        for c in self.chunks:
            if c.rfc not in seen:
                seen.append(c.rfc)
        return seen

    def search(self, query, k=5):
        from tests.fixtures import hits
        return hits(*self.chunks[:k])

    def get(self, uid):
        return self._by_uid.get(uid)


def test_dispatcher_returns_an_envelope_for_an_unknown_tool():
    tools = build_tools(_Index(CHUNKS))
    out = execute_tool(tools, "summarise_the_internet", {})
    assert out["success"] is False and out["error"] == "unknown_tool"


def test_read_chunk_on_a_bad_uid_is_terminal_not_an_exception():
    tools = build_tools(_Index(CHUNKS))
    out = execute_tool(tools, "read_chunk", {"chunk_uid": "rfc0000:9999"})
    assert out["success"] is False and out["terminal"] is True


def test_the_catalogue_is_three_tools_with_schemas():
    catalog = catalog_for_model(build_tools(_Index(CHUNKS)))
    names = {c["function"]["name"] for c in catalog}
    assert names == {"search_corpus", "read_chunk", "list_documents"}
    for c in catalog:
        assert c["function"]["description"].strip()
        assert "properties" in c["function"]["parameters"]


def test_questions_example_declares_all_three_answerabilities():
    qs = json.loads((ROOT / "questions.example.json").read_text())
    assert len(qs) == 3
    assert {q["answerability"] for q in qs} == {"answerable", "partial", "unanswerable"}


def test_the_corpus_is_not_tracked_by_git():
    """The IETF licence permits distribution in full and without modification only,
    so the RFCs are fetched rather than vendored. A committed corpus is a licence
    breach and gets the pull request returned unmarked.

    This asks git, not the filesystem. Having the RFCs on disk is normal and
    required; having them in the index is the thing that must never happen.
    """
    import subprocess
    assert "corpus/rfc/" in (ROOT / ".gitignore").read_text()
    assert "rfc/" in (ROOT / "corpus" / ".gitignore").read_text()
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "corpus/rfc"], cwd=ROOT,
            capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.SubprocessError):
        pytest.skip("git not available, so tracking cannot be checked here")
    if tracked.returncode != 0:
        pytest.skip("not inside a git working tree")
    assert tracked.stdout.strip() == "", (
        "git is tracking the corpus:\n  " + tracked.stdout.strip().replace("\n", "\n  ") +
        "\nRemove it before you push. See corpus/fetch.py for why."
    )
