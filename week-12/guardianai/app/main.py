"""FastAPI routes and middleware.

This is the only file that wires everything together. Every module elsewhere is
callable in isolation; main.py is where the security overlay becomes a single
request-handling pipeline.

Surface:
  GET  /              -> serves index.html (browser UI)
  GET  /health        -> liveness probe + pinned model + index size
  GET  /readme        -> renders README.md as dark-themed HTML
  POST /ask           -> the hardened streaming endpoint (the business route)
  POST /eval          -> run the golden set through the pipeline; return the scorecard
  POST /ingest/text   -> ingest pasted text  (scrub -> embed -> upsert)
  POST /ingest/file   -> ingest an uploaded .txt / .md / .pdf
  POST /ingest/seed   -> re-ingest the twenty seed chunks (optionally --reset)

NOTE: no `from __future__ import annotations` here, deliberately. The RBAC
decorator wraps the route handlers, and FastAPI resolves a wrapped handler's
string annotations against the DECORATOR's module globals - where names like
UploadFile do not exist. Real annotation objects sidestep the whole problem.
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from app import audit, eval as eval_mod, ingest as ingest_mod, llm, rbac, store
from app.config import get_settings
from app.injection import classify as classify_injection
from app.pipeline import generate_sentences, retrieve
from app.presidio_layer import is_redacted, scrub_input, startup_check
# NOTE: `scrub_output` is deliberately NOT imported here. The only output filter
# on the token stream is the SentenceBufferScrubber inside `pipeline`; the
# citation quote is never scrubbed on egress. See `_citation_frames`.
from app.schemas import AskRequest, Citation, EvalReport, IngestTextRequest, InjectionVerdict, RetrievedChunk

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("guardianai")

# Presidio's registry loader warns once per non-English recogniser it skips.
# Twenty lines of noise before the first request is not a boot log anyone reads.
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

_ALLOWED_UPLOAD_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot the service, narrating each slow step.

    Two things make startup take a beat: the spaCy model load (Presidio) and
    the first Qdrant round-trip (a free-tier Cloud cluster may be waking from
    suspend). Without narration a 15-second boot looks like a hang, so each step
    is announced before it starts and timed when it finishes.
    """
    settings = get_settings()
    boot_started = perf_counter()

    # Step 1: Presidio + spaCy. This is the big one - `en_core_web_lg` is ~560MB
    # and the first load reads it all off disk.
    if settings.startup_check_enabled:
        logger.info(
            "startup 1/2: loading Presidio + spaCy model %r (first load reads ~560MB, please wait)...",
            settings.spacy_model,
        )
        t = perf_counter()
        # Crash boot if Presidio is not actually operational. A guardrail that
        # fails silently is worse than no guardrail. STARTUP_CHECK_ENABLED=false
        # disables it, which reproduces the silent-scrubber failure.
        startup_check()
        logger.info("startup 1/2: Presidio ready (%.1fs)", perf_counter() - t)
    else:
        logger.warning(
            "startup 1/2: startup_check disabled - Presidio not loaded at boot, will load "
            "lazily on the first request (Failure 1 is staged this way)"
        )

    # Step 2: reach Qdrant. The raw client exception on failure is a wall of
    # httpx/httpcore frames that buries the one line worth reading, so re-raise
    # with the actual URL and the things that are actually wrong.
    logger.info(
        "startup 2/2: pinging Qdrant at %s (collection %r)...",
        settings.qdrant_url, settings.qdrant_collection,
    )
    t = perf_counter()
    try:
        indexed = store.count()
    except Exception as exc:
        raise RuntimeError(
            f"cannot reach Qdrant at {settings.qdrant_url!r} ({type(exc).__name__}: {exc}). "
            f"Check, in order: the cluster is running (free-tier Qdrant Cloud clusters are "
            f"suspended after a period of inactivity - resume it in the console), QDRANT_URL "
            f"is right, and QDRANT_API_KEY is set for a Cloud cluster. For a local instance: "
            f"docker run -d -p 6333:6333 qdrant/qdrant"
        ) from exc
    logger.info("startup 2/2: Qdrant reachable, %d chunks indexed (%.1fs)", indexed, perf_counter() - t)

    if indexed == 0:
        logger.warning("vector store is empty - run: python -m app.ingest --reset")
    logger.info("startup complete in %.1fs - ready on the configured port", perf_counter() - boot_started)
    yield


app = FastAPI(title="GuardianAI", version="0.2.0", lifespan=lifespan)

# Same-origin UI; no wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_user_middleware(request: Request, call_next):
    """Stash request_id + a stub user identity on request.state.

    In production the user comes from the JWT/session. Here we read it from
    headers so the service can be exercised from curl without standing up an
    auth provider.
    """
    request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.user = {
        "user_id": request.headers.get("x-user-id", "anonymous"),
        "roles": [r for r in request.headers.get("x-user-roles", "").split(",") if r],
    }
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    return response


# ---- probes and static ----

@app.get("/health")
def health() -> dict:
    """Liveness probe with pinned model name.

    The browser UI calls this to populate its top-right health chip, so a
    misconfigured deployment shows the wrong model visibly.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.openai_model,
        "embed_model": settings.openai_embed_model,
        "redact_threshold": settings.presidio_redact_threshold,
        "log_only_threshold": settings.presidio_log_only_threshold,
        "output_buffer_chars": settings.output_scrubber_max_buffer,
        "similarity_threshold": settings.similarity_threshold,
        "chunks_indexed": store.count(),
    }


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the browser UI."""
    return FileResponse(_PROJECT_ROOT / "index.html")


@app.get("/readme", include_in_schema=False)
def readme() -> Response:
    """Render README.md as a dark-themed HTML page."""
    import markdown as _md

    readme_path = _PROJECT_ROOT / "README.md"
    body = _md.markdown(
        readme_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>GuardianAI - README</title>"
        "<style>"
        "body{background:#0d1117;color:#e6edf3;font-family:-apple-system,Segoe UI,sans-serif;"
        "max-width:900px;margin:40px auto;padding:0 24px;line-height:1.7}"
        "h1,h2,h3{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:6px}"
        "a{color:#58a6ff}"
        "code{background:#161b22;padding:2px 6px;border-radius:4px}"
        "pre{background:#161b22;border:1px solid #30363d;padding:16px;"
        "border-radius:8px;overflow-x:auto;font-family:'Fira Code',Consolas,monospace;font-size:13px}"
        "pre code{background:none;padding:0}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #30363d;padding:8px 12px;text-align:left}"
        "th{background:#161b22;color:#58a6ff}"
        "</style></head>"
        f"<body>{body}</body></html>"
    )
    return Response(content=html, media_type="text/html; charset=utf-8")


# ---- retrieval ----

def _citation_frames(chunks: list[RetrievedChunk]) -> list[Citation]:
    """One citation card per retrieved chunk.

    The `quote` is a FOURTH EGRESS PATH. The sentence-buffer output filter only
    ever sees the model's token stream - it never sees this string. If the
    store holds an unredacted chunk (CHUNK_SCRUB_ENABLED=false at ingest), the
    answer streams clean and the citation card leaks the customer's email.

    So: NO SCRUB HERE. Deliberately. The quote is whatever the store holds,
    verbatim. Scrubbing on the way out would make the citation card safe no
    matter how dirty the store is - which is exactly the read-time scrub this
    design argues against, and it would make the check self-confirming: one flag
    would gate both the ingest scrub and this one, so flipping it would prove
    nothing about what is actually sitting in Qdrant. The store is the only
    thing standing between the corpus and this card. That is the point.
    """
    frames: list[Citation] = []
    for i, chunk in enumerate(chunks):
        frames.append(
            Citation(
                marker=f"[#{i + 1}]",
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                quote=chunk.text,
                score=chunk.score,
            )
        )
    return frames


# ---- the business route ----

@app.post("/ask")
@rbac.requires_role("analyst")
async def ask(payload: AskRequest, request: Request) -> StreamingResponse:
    """Hardened streaming endpoint.

    Pipeline:
      1. Input scrubber over the query.
      2. Injection classifier - refuse outright if flagged.
      3. Retrieval with the ACL filter; chunks were scrubbed at ingest.
      4. Citation frames (emitted before the tokens).
      5. Stream model output through SentenceBufferScrubber.
      6. Audit every scrub, refusal, denial and filter decision.
    """
    settings = get_settings()
    rid = request.state.request_id
    user_id = payload.user_id

    # Audit events raised while handling this request, captured so the
    # streaming response can surface them to the UI trust-events panel.
    trust_events: list[dict] = []

    def _log(event_type: str, detail: dict, action_taken: str) -> None:
        audit.write_event(
            request_id=rid,
            user_id=user_id,
            event_type=event_type,
            detail=detail,
            action_taken=action_taken,
        )
        trust_events.append({
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "detail": detail,
        })

    # 1. Input scrub. The audit detail carries the recogniser and span length,
    # NEVER the raw value - putting the value here would rebuild the leak.
    scrubbed = scrub_input(payload.query)
    for det in scrubbed.detections:
        _log(
            event_type="pii_scrub",
            detail={
                "recognizer": det.recognizer,
                "confidence": det.confidence,
                "span_length": det.span_end - det.span_start,
                "surface": "input",
                "operator": settings.presidio_operator,
            },
            action_taken="redacted" if is_redacted(det) else "logged_for_review",
        )

    # 2. Injection classifier (toggleable so the live demo can show the attack
    # landing before the defence goes on).
    if settings.injection_classifier_enabled:
        verdict = classify_injection(scrubbed.cleaned_text)
    else:
        verdict = InjectionVerdict(flagged=False)
    if verdict.flagged:
        _log(
            event_type="injection_refusal",
            detail={
                "classifier": "regex_set",
                "category": verdict.category,
                "confidence": verdict.confidence,
            },
            action_taken="refused_with_polite_message",
        )

        async def refusal() -> AsyncIterator[bytes]:
            for ev in trust_events:
                yield ("data: " + json.dumps({"event": ev}) + "\n\n").encode()
            yield ("data: " + json.dumps({"text": llm.REFUSAL_MESSAGE}) + "\n\n").encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(refusal(), media_type="text/event-stream")

    # 3. Retrieval with the ACL filter. Roles come from the AUTHENTICATED
    # identity only (request.state) - never from the request body, which any
    # caller could forge.
    roles = request.state.user["roles"]
    chunks = retrieve(scrubbed.cleaned_text, roles, _log)
    citations = _citation_frames(chunks)

    async def streamed() -> AsyncIterator[bytes]:
        for ev in trust_events:
            yield ("data: " + json.dumps({"event": ev}) + "\n\n").encode()
        for c in citations:
            yield ("data: " + json.dumps({"citation": c.model_dump()}) + "\n\n").encode()

        # Retrieval came back empty - say so in code, and never call the model.
        # An empty <retrieved> block invites the model to answer from its
        # training data, which is the one thing a cite-only assistant must not
        # do. This is the ACL demo's punchline: filtered to zero chunks, the
        # service declines because the ROUTE decided to, not because the model
        # felt like it. Also saves a pointless API call.
        if not chunks:
            logger.info("no chunks retrieved - declining without a model call")
            yield ("data: " + json.dumps({"text": llm.NO_CONTEXT_ANSWER}) + "\n\n").encode()
            yield b"data: [DONE]\n\n"
            return

        # Model output, one output-scrubbed sentence per SSE frame. Same
        # generator the eval harness joins - one pipeline, two consumers.
        for sentence in generate_sentences(scrubbed.cleaned_text, chunks):
            yield ("data: " + json.dumps({"text": sentence}) + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(streamed(), media_type="text/event-stream")


# ---- eval harness ----

@app.post("/eval", response_model=EvalReport)
def eval_endpoint() -> EvalReport:
    """Run the golden set through the real pipeline and return the scorecard.

    This calls the model once per answerable row, so it costs a handful of
    completions - a deliberate action, not something the UI fires on load.
    `pii_leaks` must be 0; `accuracy` is the headline. The harness runs
    `app.pipeline`, the same code `/ask` streams, so a green result is a
    statement about the live route, not a parallel copy of it.
    """
    return eval_mod.evaluate()


# ---- ingestion API ----

def _extract_text(filename: str, raw: bytes) -> str:
    """Pull plain text out of an upload. .txt / .md natively; .pdf via pypdf
    (pure Python, no system libraries, no OCR - a text-layer PDF only)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix!r}; allowed: .txt, .md, .pdf",
        )
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:  # pragma: no cover
            raise HTTPException(status_code=503, detail="pypdf not installed; pip install pypdf")
        import io

        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return raw.decode("utf-8", errors="replace")


@app.post("/ingest/text")
@rbac.requires_role("analyst")
async def ingest_text(payload: IngestTextRequest, request: Request) -> dict:
    """Ingest pasted text. Chunk -> Presidio scrub -> embed -> upsert.

    The scrub is not optional here and it is not on the read path: whatever
    PII is in this body is redacted BEFORE it is written to Qdrant.
    """
    report = ingest_mod.ingest_document(
        payload.text,
        source=payload.source,
        visible_to=payload.visible_to,
        request_id=request.state.request_id,
        user_id=request.state.user["user_id"],
    )
    return {**report.model_dump(), "chunks_indexed": store.count()}


@app.post("/ingest/file")
@rbac.requires_role("analyst")
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    visible_to: str = Form("analyst"),
) -> dict:
    """Ingest an uploaded .txt / .md / .pdf. Same path, same scrub."""
    settings = get_settings()
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file too large ({len(raw)} bytes; max {settings.max_upload_bytes})",
        )
    text = _extract_text(file.filename or "upload.txt", raw)
    if not text.strip():
        raise HTTPException(status_code=422, detail="no extractable text in file")

    roles = [r.strip() for r in visible_to.split(",") if r.strip()] or ["analyst"]
    report = ingest_mod.ingest_document(
        text,
        source=file.filename or "upload.txt",
        visible_to=roles,
        request_id=request.state.request_id,
        user_id=request.state.user["user_id"],
    )
    return {**report.model_dump(), "chunks_indexed": store.count()}


@app.post("/ingest/seed")
@rbac.requires_role("analyst")
async def ingest_seed(request: Request, reset: bool = False) -> dict:
    """Re-ingest the twenty seed chunks. `?reset=true` rebuilds the collection -
    the same thing `python -m app.ingest --reset` does, from the UI."""
    report = ingest_mod.ingest(
        reset=reset,
        request_id=request.state.request_id,
        user_id=request.state.user["user_id"],
    )
    return {**report.model_dump(), "chunks_indexed": store.count()}
