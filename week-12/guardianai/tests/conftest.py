"""Test environment.

The application has no offline mode - it talks to a real embedding model, a
real chat model and a real Qdrant. The suite still must not: per CODE_GUIDE,
tests run with no key and no network. So the fakes live HERE, at the two
network seams, rather than as a branch inside production code:

  * `app.embedder`  - a deterministic hashed bag-of-words vector stands in for
    the OpenAI embedding API. Lexical, not semantic, but it is a real vector
    that a real (in-process) Qdrant really indexes and searches, which is all
    the store-level assertions need.
  * `app.llm.stream_answer` - a scripted generator stands in for the chat
    completion. It deliberately behaves like a COMPLIANT model, including the
    compliance we do not want: with the classifier off it reads its system
    prompt out loud, and with the retrieval sanitiser off it follows an
    instruction planted in a document. That is what makes the attack tests
    meaningful without a network call.

Everything under test - Presidio, the classifier, the sanitiser, RBAC, the
ACL filter, the audit log, the sentence buffer - is the real code.

The vector store is Qdrant's own `:memory:` mode: a real, fully in-process
instance (see `app/store.py::_client` - `location=":memory:"` is what triggers
it, which is why the client is built with `location=`, not `url=`).

These values are set BEFORE `app.config` is first imported, because
`get_settings()` is cached for the process lifetime.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="guardianai-tests-")

os.environ["AUDIT_LOG_PATH"] = os.path.join(_TMP, "audit.jsonl")
os.environ["QDRANT_URL"] = ":memory:"
os.environ["USER_ID_HASH_SALT"] = "test-salt-only"
os.environ["OPENAI_API_KEY"] = "sk-test-not-real"
# SIMILARITY_THRESHOLD ships at 0.55, calibrated for text-embedding-3-large.
# The stub embedder below is lexical, and lexical cosines run far lower, so the
# suite uses a threshold calibrated for the stub. The threshold's *job* - keep
# Qdrant from back-filling weak matches, so "returns zero chunks" stays true -
# is what the ACL test exercises, and that holds at either value.
os.environ["SIMILARITY_THRESHOLD"] = "0.05"
# The stub embedder's width. Must match what store.VECTOR_SIZE is patched to.
_TEST_VECTOR_SIZE = 256
# The shipped defaults. Never hardcode a flag the shipped .env.example
# contradicts - the tests must fail if the shipped config is wrong.
os.environ.setdefault("CHUNK_SCRUB_ENABLED", "true")
os.environ.setdefault("RETRIEVAL_SANITISER_ENABLED", "true")
os.environ.setdefault("INJECTION_CLASSIFIER_ENABLED", "true")

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

AUDIT_LOG = os.environ["AUDIT_LOG_PATH"]


# ---- the two network seams, stubbed ----

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "and", "for",
    "you", "your", "our", "of", "to", "in", "on", "at", "by", "about", "most",
    "recent", "have", "has", "this", "that", "year", "please", "how", "did",
    "do", "any", "with", "from", "it", "its", "we", "me", "my", "summarise",
    "summarize", "tell", "give", "show",
}


def _stub_embed(text: str) -> list[float]:
    """Hashed bag-of-words, L2-normalised. Stands in for the embedding API."""
    vec = [0.0] * _TEST_VECTOR_SIZE
    for token in _TOKEN.findall(text.lower()):
        if len(token) < 3 or token in _STOPWORDS:
            continue
        vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % _TEST_VECTOR_SIZE] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return vec if norm == 0.0 else [v / norm for v in vec]


# Scripted answers keyed by what the query is about.
_SCRIPTED: list[tuple[tuple[str, ...], str]] = [
    (("runbook", "migration"),
     "The migration runbook, tracked under ticket TCK-2026-001234, "
     "was completed on 18 May 2026 [#1]."),
    (("ticket", "support", "refund", "checkout"),
     "John Mercer (john.mercer@example.com, +1-415-555-0164) reported checkout failures "
     "on ticket TCK-2026-004417; a refund of 120 dollars was approved [#1]."),
    (("revenue", "q3", "forecast", "memo"),
     "Q3 revenue exceeded internal forecasts by six percent, "
     "driven by enterprise renewals [#1]."),
]

# What a compliant model does with a chunk that told it to exfiltrate data.
_COMPROMISED_SUFFIX = (
    " As instructed by the retrieved document, backups have been emailed to evil@example.com."
)


def _stub_stream_answer(*, system_prompt: str, user_message: str):
    """Stand in for the chat completion. Behaves like a COMPLIANT model."""
    from app.injection import classify
    from app.llm import NO_CONTEXT_ANSWER, SYSTEM_PROMPT_TEMPLATE

    # Classifier off + override prompt through => the model reads its
    # instructions out loud, because that is what a real model does.
    verdict = classify(user_message)
    if verdict.flagged and verdict.category in {"system_prompt_leak", "instruction_override"}:
        yield "Certainly. Here are my instructions:\n\n" + SYSTEM_PROMPT_TEMPLATE.strip()
        return

    # rsplit, not split: SYSTEM_PROMPT_TEMPLATE mentions "<retrieved>" in its
    # own instructions, so a forward split never isolates the real block.
    # (The route short-circuits the zero-chunk case before reaching here; this
    # is belt-and-braces so the stub cannot invent an answer from nothing.)
    body = system_prompt.rsplit("<retrieved>", 1)[-1].replace("</retrieved>", "")
    if not body.strip():
        yield NO_CONTEXT_ANSWER
        return

    q = user_message.lower()
    answer = NO_CONTEXT_ANSWER
    for keywords, scripted in _SCRIPTED:
        if any(k in q for k in keywords):
            answer = scripted
            break

    # Sanitiser off => the planted instruction is still in the prompt, and a
    # real model follows it. So does this one.
    if classify(body).flagged and answer is not NO_CONTEXT_ANSWER:
        answer += _COMPROMISED_SUFFIX

    for word in answer.split(" "):
        yield word + " "


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Patch the embedding and chat seams. No test may reach the network."""
    from app import embedder, llm, store

    monkeypatch.setattr(embedder, "embed_text", _stub_embed)
    monkeypatch.setattr(embedder, "embed_texts", lambda ts: [_stub_embed(t) for t in ts])
    # store imported the names directly, so patch them there too.
    monkeypatch.setattr(store, "embed_text", _stub_embed)
    monkeypatch.setattr(store, "embed_texts", lambda ts: [_stub_embed(t) for t in ts])
    # The stub's vectors are narrower than the real model's.
    monkeypatch.setattr(store, "VECTOR_SIZE", _TEST_VECTOR_SIZE)
    monkeypatch.setattr(llm, "stream_answer", _stub_stream_answer)
    # main.py imported the llm MODULE, so patching the attribute is enough.
    yield


@pytest.fixture(autouse=True)
def pristine_store(no_network):
    """Every test starts against a freshly ingested twenty-chunk corpus."""
    from app.ingest import ingest

    ingest(reset=True)
    yield
    # Leave the flags as shipped, whatever a test did to them.
    for key, value in {
        "CHUNK_SCRUB_ENABLED": "true",
        "RETRIEVAL_SANITISER_ENABLED": "true",
        "INJECTION_CLASSIFIER_ENABLED": "true",
    }.items():
        os.environ[key] = value
    get_settings.cache_clear()


@pytest.fixture()
def audit_lines():
    """Read the audit log written during this test."""
    def _read() -> list[dict]:
        import json

        if not os.path.exists(AUDIT_LOG):
            return []
        with open(AUDIT_LOG, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    return _read


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


def sse_frames(body: str) -> list[dict]:
    """Parse an SSE body into its JSON frames (ignoring the [DONE] terminal)."""
    import json

    frames = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        frames.append(json.loads(payload))
    return frames
