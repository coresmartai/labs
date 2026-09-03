# CitationRAG - Week 5

**Applied GenAI & Agentic AI Engineering Course · Week 5**

CitationRAG is a FastAPI service that answers questions from a Qdrant knowledge index
with inline source citations, validates every citation deterministically,
and refuses gracefully when retrieved context is too weak to support an answer.
It demonstrates the three patterns at the core of every production RAG system:
threshold-gated retrieval, citation-contract generation, and grounded eval against a golden dataset.

| Pattern | Endpoint / entry-point | File |
|---|---|---|
| **Retrieval + threshold gate** | `POST /answer` | `app/retriever.py`, `app/main.py` |
| **Citation-contract generation + validation** | `POST /answer` | `app/llm.py`, `app/citations.py` |
| **Groundedness eval harness** | `POST /eval` | `app/main.py` |

---

## 1. Project layout

```
/
├── app/
│   ├── __init__.py
│   ├── config.py            <- typed Settings + .env loader; pinned model versions + thresholds
│   ├── schemas.py           <- Pydantic models (Chunk, Citation, AnswerResponse, EvalMetrics)
│   ├── store.py             <- ONE place that reaches Qdrant: embedded local (default) or server
│   ├── retriever.py         <- KnowledgeVault wrapper; returns chunks + scores + spread
│   ├── embedder.py          <- text-embedding-3-large wrapper for the dense channel
│   ├── citations.py         <- deterministic citation validator (ID presence + quote substring)
│   ├── llm.py               <- thin OpenAI SDK wrapper; citation prompt contract
│   └── main.py              <- FastAPI routes: /answer, /eval, /, /readme, /health
├── tests/
│   └── test_endpoint.py     <- smoke tests (no real API calls - LLM monkeypatched)
├── scripts/
│   └── build_golden_dataset.py  <- fetch real chunks from Qdrant, generate golden_dataset.json
├── data/
│   └── golden_dataset.json  <- checked-in reference eval set; regenerate for your own index
├── index.html               <- browser UI (open via http://localhost:8000)
├── week5_notebook.ipynb     <- curl + Python requests for every endpoint
├── requirements.txt
├── .env.example             <- copy to .env and fill in OPENAI_API_KEY
├── citationrag_architecture.svg
├── groundedness_eval_flow.svg
├── 3_three_refusal_sources.svg
├── .gitignore
└── README.md                <- you are here
```

`retriever.py`, `citations.py`, and the golden-dataset builder are new in Week 5; `store.py`
and `embedder.py` follow the exact Week 4 KnowledgeVault pattern. The knowledge index is the
one Week 4 ingested - the `knowledgevault` collection in the embedded Qdrant store your
KnowledgeVault project created (its `qdrant_local/` folder). There is no local JSON index in
this package, and this project never ingests anything itself.
`scripts/build_golden_dataset.py` fetches real chunks from that store and uses the LLM to
generate grounded Q&A pairs, so the eval dataset always reflects what's actually in the
collection.

> **Which index?** Everything in this guided build - the thresholds, the UI's demo pills, the
> shipped golden dataset - targets the **Week 4 lab index: the single Attention Is All You Need
> paper**. Your 5-paper PaperFinder index is not used here; running CitationRAG over your own
> corpus (with your own thresholds and golden set) is exactly what Project 10 DocuRAG is for.

---

## 2. What this app does

- **Retrieves** top-k chunks from a knowledge index with similarity scores and spread -> `POST /answer`
- **Generates** answers with inline `[doc#N]` citations using `gpt-5.4-mini-2026-03-17` -> `POST /answer`
- **Validates** every citation ID and supporting quote deterministically -> `POST /answer`
- **Refuses** with an exact string when retrieval is weak -> `POST /answer`
- **Evaluates** groundedness against the golden dataset (`data/golden_dataset.json`); reports five metrics -> `POST /eval`
- Serves a **browser UI** at `GET /` - no separate server needed
- Renders the README as HTML at `GET /readme`


---

## 3. Setup (5 min)

> Requirements: Python 3.11, an OpenAI API key with access to `gpt-5.4-mini-2026-03-17`,
> and a completed Week 4 KnowledgeVault ingestion (the `qdrant_local/` folder with the
> Attention paper in it).

```bash
# 1. Create and activate a venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env file and fill in your key
cp .env.example .env
# Open .env - set OPENAI_API_KEY

# 4. Point at your Week 4 index (embedded local Qdrant - the default)
# In .env, set QDRANT_LOCAL_PATH to the qdrant_local/ folder inside your
# Week 4 knowledgevault project, e.g.
#   QDRANT_LOCAL_PATH=../../week-04/knowledgevault/qdrant_local
# (or copy that folder into this project and leave the default ./qdrant_local)
#
# ONE-PROCESS RULE: the embedded store allows one process at a time.
# Stop the Week 4 uvicorn server before starting CitationRAG.
#
# Running Qdrant as a server or in the cloud instead? Set QDRANT_MODE=server
# plus QDRANT_URL (and QDRANT_API_KEY for Qdrant Cloud).

# 5. (Optional) Regenerate the golden evaluation dataset from your live chunks
# A reference golden_dataset.json is checked in; regenerate to match your index.
# Run with both servers stopped - the script opens the embedded store itself.
python scripts/build_golden_dataset.py

# 6. Start the server
uvicorn app.main:app --reload
```

**You're live at `http://localhost:8000`.**

- Browser UI: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- README: `http://localhost:8000/readme`

> **Knowledge index:** Lives in the Qdrant `knowledgevault` collection ingested in Week 4,
> reached through `app/store.py` (embedded local by default, server mode via `QDRANT_MODE`).
> The retriever uses `text-embedding-3-large` + BM25 + RRF - thresholds `0.55 / 0.08` are
> calibrated for real semantic embeddings against the Attention-paper lab index.

---

## 4. File-by-file walkthrough

---

### `app/config.py` - Settings

All environment variables live in one typed `Settings` class (powered by `pydantic-settings`).

```python
settings = get_settings()
print(settings.llm_model)             # gpt-5.4-mini-2026-03-17 (override with LLM_MODEL in .env)
print(settings.similarity_threshold)  # 0.55  - calibrated for text-embedding-3-large cosine scores
print(settings.spread_delta)          # 0.08  - minimum top-1 minus top-3 spread
```

Two reasons to centralise config here:
1. Reading `os.environ["KEY"]` from random places is how secrets end up in logs.
2. `pydantic-settings` validates types on startup - a missing key fails loudly before the first request, not mid-flight.

The `@lru_cache` on `get_settings()` means `.env` is read exactly once.

**Model-pinning:** `llm_model = "gpt-5.4-mini-2026-03-17"` - the exact version string, not `"gpt-5.4-mini"`.
An alias like `"gpt-5.4-mini"` resolves to whatever OpenAI's current default is - silent upstream
changes are how eval numbers drift without warning.

**Thresholds:** `similarity_threshold` and `spread_delta` are calibrated against real
`text-embedding-3-large` cosine scores from the Qdrant collection. The gate checks the dense
cosine score (interpretable scale in [0, 1]), not the RRF score.

---

### `app/store.py` - One place that reaches Qdrant

Identical pattern to Week 4 KnowledgeVault. Every module that needs a client calls
`get_client()`; nobody constructs `QdrantClient` anywhere else, so switching modes is one
line in `.env`:

- `QDRANT_MODE=local` (default) - Qdrant runs **embedded in this Python process**, reading
  the store at `QDRANT_LOCAL_PATH`. Point it at the `qdrant_local/` folder your Week 4
  ingestion created. Nothing to install, nothing to sign up for. One rule: only one process
  may hold the local store at a time - stop the Week 4 server before starting CitationRAG,
  and stop CitationRAG before running `build_golden_dataset.py`.
- `QDRANT_MODE=server` - a running Qdrant at `QDRANT_URL` (Docker on your laptop, or Qdrant
  Cloud with `QDRANT_API_KEY`). Many processes may connect.

---

### `app/schemas.py` - Data contracts

```python
class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_url: str | None = None
    score: float = Field(ge=0.0, le=1.0)   # validated at schema level

class Citation(BaseModel):
    chunk_id: str
    supporting_quote: str

class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool = False
    retrieval_top_score: float | None = None
    retrieval_spread: float | None = None
    validation_passed: bool = True
    refusal_source: str | None = None      # "threshold_gate" | "llm" | "validation"
    retrieved_chunks: list[Chunk] | None = None
    raw_model_answer: str | None = None    # preserved on validation failures for UI
    validation_detail: str | None = None   # one-line reason why validation failed
```

`Chunk.score` is bounded `[0, 1]` - a retriever bug that returns 1.8 is caught here, not downstream.
`AnswerResponse` defaults to `refused=False` and `validation_passed=True` - the happy-path state.
Failures explicitly flip these. `refusal_source` tells the UI exactly which pipeline stage refused
(`"threshold_gate"` before the LLM is called, `"llm"` if the model itself refused, `"validation"`
if citation checking failed). `raw_model_answer` is only populated on validation failures so the UI
can show the wrong answer alongside the reason it was rejected.

`EvalRow` is a standalone Pydantic model that carries per-row scoring fields (`expected`,
`actual`, `passed`, `grounded`, `false_answer`, `false_refusal`, `cit_recall_ok`,
`cit_valid_ok` - a tri-state: True = the model generated and every citation survived the
validator, False = the validator rejected it, None = the model never generated) plus a
compact preview (`system_answer`, `cited_ids`, `refusal_source`) and embeds the full
`AnswerResponse` for that row in a `response` field. That embedded response lets the eval
UI render the same four-step pipeline trace for any row without an extra API call.
`EvalMetrics` aggregates the five rates across all rows and carries the `rows` list.

---

### `app/retriever.py` - Hybrid retrieval with confidence scores

The retriever exposes one public method: `search(query)`. It always returns three things:

1. `chunks` - top-k `Chunk` objects, renumbered `doc#1…doc#k` in RRF rank order, each
   carrying `document_id` and `page_number` so a citation can always say where it came from
2. `top1` - highest dense cosine score from Qdrant (used for the threshold gate; lives in [0, 1])
3. `spread` - top-1 minus top-3 dense score **after near-duplicate collapse**; narrow spread
   means no single distinct passage is clearly most relevant

**Why spread is deduplicated first:** KnowledgeVault chunks overlap by 50 tokens, so a
well-covered query often returns two or three near-duplicate sibling chunks with almost
identical scores. Raw top1 minus top3 would then be tiny and the gate would refuse a
perfectly answerable question - a false refusal caused by our own chunking, not by weak
retrieval. The retriever collapses hits whose token-set Jaccard similarity is >= 0.6 to
their highest-scoring member before computing spread, so spread measures how far ahead the
best *distinct* passage is.

Two retrieval channels are fused with **Reciprocal Rank Fusion (RRF)**:

- **Dense (Qdrant ANN):** the query is embedded with `text-embedding-3-large` via `app/embedder.py`
  and used for approximate nearest-neighbour search against the `knowledgevault` collection.
  Captures semantic similarity. Raw cosine scores from this channel are returned for the threshold
  gate because they have an interpretable scale.

- **Sparse (BM25Okapi in-memory):** all Qdrant points are scrolled and a BM25 index is built over
  their text payloads. Captures exact keyword matches that dense search misses.

RRF fuses the two rank lists without score normalisation:
`contribution = 1 / (k + rank)`, k = 60. The chunk ranked first by both channels scores highest.
No arbitrary scaling between incompatible score distributions.

Chunks are renumbered `doc#1…doc#k` on return so the citation contract in `llm.py` (which expects
`[doc#N]` markers) works regardless of the underlying Qdrant UUID-based point IDs.

---

### `app/citations.py` - Deterministic validator

Two checks, in order:

**Check 1 - ID presence:** every `[doc#N]` in the answer prose and every `chunk_id` in the
citations list must correspond to a chunk that was actually sent in context. If the model
invents `[doc#99]` when only `[doc#1]`–`[doc#5]` were provided, this check fails.

**Check 2 - Quote substring:** each citation's `supporting_quote` must appear verbatim
(case-insensitive) in the cited chunk's text. If the model fabricates a quote, this fails.

Both checks set `validation_passed=False` on the response. `main.py` converts any validation
failure to a conservative refusal - never ships a citation the user can't inspect.

Why deterministic instead of LLM-as-judge? Because deterministic is free, instant, and immune
to the judge's own hallucinations. LLM-as-judge is reserved for paraphrase-detection in the
eval harness (Week 7), where substring matching is too strict.

---

### `app/llm.py` - Citation prompt contract

One exported function: `generate_with_citations(question, chunks)`.

The system prompt has four sections in this order:
1. **Task** - name the assistant and audience
2. **Format** - require JSON with `answer` and `citations` fields
3. **Citation contract** - every claim must end with `[doc#N]` matching a context chunk ID
4. **Fallback** - if context doesn't cover the question, return the exact refusal string

`response_format={"type": "json_object"}` enforces JSON at the API layer, not just via prompt.
If the model still returns malformed JSON (it can), the `JSONDecodeError` handler returns a
refused `AnswerResponse` instead of raising - fail-first, not crash-first.

---

### `app/main.py` - FastAPI routes

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Serve browser UI (`index.html`) |
| `/health` | GET | Liveness probe - returns `{status, service, model}` |
| `/readme` | GET | Render `README.md` as dark-themed HTML |
| `/answer` | POST | Retrieve -> threshold gate -> generate -> validate |
| `/eval` | POST | Loop golden dataset through `/answer`; return five metrics |

CORS is open (`allow_origins=["*"]`) so the UI works from `file://` or any local port.
**Narrow the allow-list before production.**

The `/answer` four-step pipeline in plain English:
1. Retrieve top-k chunks with scores
2. If `top1 < similarity_threshold` or `spread < spread_delta` - refuse immediately (no LLM call)
3. Call `generate_with_citations` - the only place in the codebase that touches the OpenAI SDK
4. Validate citations - convert validation failure to refusal

---

### `index.html` - Browser UI

Open at `http://localhost:8000` after starting the server.

**Left panel:**
- Question textarea with six pills - five Attention-paper queries + one off-topic (Wi-Fi password)
- Active thresholds display (similarity 0.55, spread 0.08, top-k 5)
- "Ask CitationRAG" button (primary) and "Run Full Eval" button (secondary)

**Right panel - single query view:**
Full four-step pipeline trace after every `/answer` call:
1. **Query** - the question as typed
2. **Retrieved Chunks** - all chunks sent to the model, each expandable to show full text
3. **Pipeline · Threshold Gate & Citation Validation** - scores bar (top1 / spread / citation valid / outcome)
4. **Model Output** - answer with highlighted `[doc#N]` markers and citation cards; or a refused
   amber box with the source label (`threshold_gate` / `llm` / `validation`); on validation
   failures the raw wrong answer and failure reason are shown inside the refused box.

**Right panel - eval view:**
- Five-metric summary grid (groundedness / citation validity / recall / false-answer / false-refusal)
- Per-row breakdown: each row is a `<details>` element showing PASS/FAIL chip, expected vs actual,
  refusal source, and cited IDs - click **▸ expand** to open the full four-step pipeline trace
  for that row (same renderer as the single-query view, reused via `_answerHTML()`).

**Logs pane:** timestamped request trace with retrieval scores and validation outcomes.
**Header:** health chip (auto-checks on load, shows model name), **📖 README** - opens `http://localhost:8000/readme` in a new tab

---

### `week5_notebook.ipynb` - API notebook

A Jupyter notebook covering every endpoint two ways (cURL + Python):

| Section | curl (`%%cmd`, Windows) | Python (`requests`) |
|---|---|---|
| Health check | ✓ | ✓ |
| POST /answer (covered) | ✓ | ✓ |
| POST /answer (off-topic) | ✓ | ✓ |
| Two questions side by side | - | ✓ |
| POST /eval | ✓ | ✓ |
| Failure: question too short (422) | ✓ | ✓ |
| Failure: missing field (422) | ✓ | ✓ |
| Failure: index missing (500) | - | note |
| Full raw response dump | - | ✓ |
| Swagger UI link | - | ✓ |

All `%%cmd` cells use Windows double-quote syntax - single quotes cause errors in `cmd.exe`.

---

## 5. Try it out

### a) Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"citationrag","model":"gpt-5.4-mini-2026-03-17"}
```

### b) Ask a covered question

```bash
curl -X POST http://localhost:8000/answer ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What is self-attention and how does it work?\"}"
```

Expected: JSON response with an `answer` containing `[doc#N]` markers, a non-empty `citations`
list, `refused=false`, and `validation_passed=true`.

### c) Ask an off-topic question

```bash
curl -X POST http://localhost:8000/answer ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What is the lunchroom Wi-Fi password?\"}"
```

Expected: `refused=true`, empty `citations`, `answer` equals the exact refusal string,
`refusal_source` is either `"threshold_gate"` (score too low) or `"llm"` (model refused from context).

### d) Run the full eval

```bash
curl -X POST http://localhost:8000/eval
```

Expected: five metrics JSON. `rows_scored=10` (8 answerable + 2 unanswerable).
`false_answer_rate` should be 0.0 if both off-topic golden rows were correctly
refused - that is the metric those two rows drive. `false_refusal_rate` is the
other direction: answerable rows the gate turned away, so it is the number to
read when a question you know the paper covers comes back refused.

---

## 6. Diagrams

| File | What it shows |
|---|---|
| `citationrag_architecture.svg` | Request -> retrieve -> threshold gate -> generate -> validate -> answer / fallback. Each named code file sits next to its pipeline stage. |
| `groundedness_eval_flow.svg` | Golden dataset row -> `/answer` -> judge -> tally -> five-metric summary card. |
| `3_three_refusal_sources.svg` | The three refusal sources (`threshold_gate` / `llm` / `validation`), what each means, and who owns the fix (retrieval / chunking-or-prompt / instruction-following). |


---

## 7. Common failure modes

### a) Fabricated citation - model invents a chunk ID (Failure #1)

**What:** The model emits `[doc#99]` when only `[doc#1]`–`[doc#5]` were in context.
**What you see:** `validation_passed=false`, the route returns the conservative refusal.
**Diagnostic:** Log line `validation failed - <detail>`, e.g. `validation failed - Fabricated chunk ID [doc#99] in answer prose - not in the 5 chunks sent to the model ([doc#1], [doc#2], …)`.
**Fix:** The validator in `citations.py` catches this. If you're seeing it frequently, make
chunk IDs more visually prominent in the user message in `llm.py:_build_context`.

### b) Generation drift past the threshold (Failure #2)

**What:** Retrieval scores are marginal - top-1 just above threshold, spread narrow.
The model answers confidently from parametric knowledge rather than the chunks.
**What you see:** An answer that sounds plausible but cites chunks that don't support the claim.
**Fix:** Tighten `SIMILARITY_THRESHOLD` and `SPREAD_DELTA` in `.env`. Recalibrate by running
the retriever against the golden dataset and plotting top-1 scores for `must_cite != []` rows
versus `must_cite == []` rows. The ideal threshold separates the two distributions.

### c) Eval rubric misclassifies a correct refusal (Failure #3)

**What:** A golden-dataset row with `must_cite=[]` (expected refusal) is marked wrong because
the judge compared the refusal string to `expected_answer` literally.
**What you see:** `false_answer_rate` is higher than it should be; inspecting tallies shows
refusal rows counting as false answers.
**Fix:** Already applied in `_judge_row` - if `must_cite=[]` and `refused=True`, the row
scores as a correct refusal. The pattern to internalise: test the judge before trusting
any number it produces.

### d) A metric that cannot fail (Failure #4)

**What:** An earlier draft of this eval reported "citation precision" computed as
`cited_ids.issubset(must_cite_set | cited_ids)` - a set is always a subset of a union that
includes itself, so the metric was 1.0 by construction. A judge that cannot say no is not
a judge; the dashboard was presenting an enforced invariant as a measurement.
**Why it was invisible:** the validator upstream already guarantees no invalid citation
ever ships (failures become refusals), so the number LOOKED plausible.
**Fix:** the metric is now **citation validity**: of the rows where the model actually
generated an answer, the fraction whose citations survived the deterministic validator.
Rows the threshold gate stopped are excluded from the denominator - the model never
exercised citation discipline on them. This number CAN drop, which is what makes it a metric.

---

## 8. Run the tests

```bash
pytest -v
```

4 smoke tests - no real API calls, no OpenAI key needed:

| Test | What it checks |
|---|---|
| `test_health` | `/health` returns 200 with `status=ok` |
| `test_answer_grounded` | `/answer` returns 200; if not refused, `validation_passed=True` |
| `test_answer_refusal_below_threshold` | Low-score retrieval triggers `refused=True` and exact refusal string |
| `test_answer_fabricated_citation_caught` | Model inventing `[doc#99]` is caught by the validator; route returns refusal |

Smoke tests are not a replacement for integration tests - those come in Week 14 DeployCore.
They are a `pytest -v` you can run after every edit to confirm you haven't broken the contract.

---

## 9. Where this goes next

The same service is extended each week:

- Query transformation, reranking, and chunk-size sweeps lift retrieval relevance measurably - the five metrics from `/eval` are the baseline to beat.
- Adversarial query generator + LLM-as-judge harness; uses this `/eval` endpoint and the golden dataset as the target to attack.
- PII scrubbing + RBAC layered on top of `/answer`.
- Streaming UI + user feedback piped back into the eval store.
- Prompt versioning + automated threshold recalibration against the golden dataset on every prompt change.

Don't throw this away - every week builds on it.

---

## 10. Notes & errata

### Note - the judge in this week is deterministic, on purpose

Groundedness, citation validity/recall, false-answer rate and false-refusal rate are all computed with
cheap deterministic proxies - string containment, span checks, set arithmetic. **There is no
LLM-as-judge in current week.** That arrives in BreakRAG™, where it comes with a rubric, two judges
and a Krippendorff's-α agreement floor - because an LLM judge you haven't measured is just a second
model you haven't evaluated.

---

## Decisions *(week project - CitationRAG)*

- **Deterministic validator over LLM-as-judge for citation IDs.** The ID-presence check
  and quote-substring check are O(1) lookups - free, instant, and unable to hallucinate.
  LLM-as-judge is reserved for paraphrase detection in the eval harness (Week 7), where
  substring matching is too strict; it is not appropriate where an exact lookup is possible.

- **Exact refusal string, not approximate detection.** The refusal string is byte-identical
  everywhere: in the system prompt, in `REFUSAL_STRING`, and in the test assertion.
  Three consumers depend on it - downstream code, eval code, UI code. The day the string
  drifts is the day the eval starts marking correct refusals as wrong answers.

- **Conservative refusal on validation failure, not "return anyway."** A response with a
  fabricated citation is worse than a refusal because it ships a lie with a false source
  attached. The conservative path loses some recall but eliminates a class of misinformation.

- **Hybrid dense + sparse retrieval (Qdrant ANN + BM25Okapi) fused with RRF.** Dense search
  captures semantic similarity; sparse BM25 captures exact keyword matches that dense search
  misses (acronyms, model names, numeric values). RRF fuses rank lists without any score
  normalisation - dense cosine scores and BM25 scores live on incompatible distributions and
  cannot be linearly combined without calibration data. RRF sidesteps that entirely.

- **Thresholds 0.55 / 0.08 calibrated against real `text-embedding-3-large` cosine scores.**
  The gate checks the raw dense cosine score (interpretable in [0, 1]), not the RRF score.
  0.55 separates on-topic queries (typically ≥ 0.65) from off-topic queries (typically < 0.45)
  against the Attention paper `knowledgevault` collection. Recalibrate by plotting top-1 dense
  scores for must-answer vs must-refuse rows from your golden dataset.

- **Spread is computed after near-duplicate collapse, not on raw scores.** KnowledgeVault's
  50-token chunk overlap means sibling chunks of the SAME passage score nearly identically;
  raw top1 minus top3 spread would read "ambiguous" precisely when the index covers the query
  best, and the gate would false-refuse. Collapsing hits at token-set Jaccard >= 0.6 before
  measuring spread makes the gate ask the right question: is the best distinct passage clearly
  ahead of the next distinct one?

- **Citation validity over citation "precision."** The deterministic validator makes
  invalid-citation shipping impossible, so precision-against-the-shipped-answer is 1.0 by
  definition - an enforced invariant, not a measurement. The eval instead reports the model's
  raw citation discipline: of the answers the model generated, how many needed the validator
  to step in. See Failure #4.
