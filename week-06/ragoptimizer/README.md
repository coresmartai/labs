# RAGOptimizer - Week 6

**Applied GenAI & Agentic AI Engineering Course · Week 6**

RAGOptimizer extends Week 5's CitationRAG service with three retrieval improvements - HyDE query transformation, cross-encoder reranking, and extractive context compression - and measures the before/after gain against the Week 5 golden dataset with a deterministic five-metric proxy dashboard (`/eval/compare`). It demonstrates the retrieval-quality patterns used in every production RAG system: retrieve wide, rerank for precision, compress for token efficiency.

| Pattern | Endpoint / entry-point | File |
|---|---|---|
| **HyDE query transformation** | `retrieve()` internal → `POST /ask` | `app/hyde.py` |
| **Cross-encoder reranking** | `retrieve()` internal → `POST /ask` | `app/reranker.py` |
| **Extractive context compression** | `retrieve()` internal → `POST /ask` | `app/compressor.py` |
| **Full pipeline eval** | `POST /eval` | `app/main.py` |
| **Before/after comparison** | `POST /eval/compare` | `app/main.py` |

---

## Project layout

```
/
├── app/
│   ├── __init__.py              ← version = "0.1.0"
│   ├── config.py                ← typed Settings + .env loader (pydantic-settings)
│   ├── schemas.py               ← Chunk, QueryRequest, Citation, GroundedAnswer, BenchmarkRow
│   ├── tools.py                 ← {schema, impl} pairs + execute_tool dispatcher  ← W1 pattern carried forward
│   ├── llm.py                   ← OpenAI SDK wrapper; gpt-5.4-mini-2026-03-17 + gpt-5.4-nano-2026-03-17
│   ├── hyde.py                  ← hyde_probe(): generates hypothetical answer as retrieval probe
│   ├── reranker.py              ← rerank(): batched cross-encoder, CPU-thread-pinned
│   ├── compressor.py            ← compress(): sentence-level trimming, marker-preservation invariant
│   ├── retriever.py             ← retrieve(): composes HyDE → vector → union → rerank → compress
│   └── main.py                  ← FastAPI routes: /, /readme, /health, /config, /ask, /eval, /eval/compare
├── tests/
│   ├── test_endpoint.py         ← 5 smoke tests (no API calls needed)
│   ├── test_gate.py             ← 10 gate + channel tests
│   └── test_citation_validity.py ← 6 tests: the metric must be able to say no
├── index.html                   ← browser demo UI
├── week6_notebook.ipynb         ← curl + Python walkthrough for every endpoint and module
├── data/
│   └── golden_dataset.json      ← the ten questions: 8 answerable, 2 refusal
├── requirements.txt
├── .env.example                 ← copy to .env and fill OPENAI_API_KEY
├── .gitignore
├── before_after_results_dashboard.svg
├── ragoptimizer_architecture.svg
└── README.md                    ← you are here
```

---

## 1. What this app does

- **Transforms queries with HyDE** - asks `gpt-5.4-nano-2026-03-17` to write a hypothetical answer, embeds that instead of the raw query, closes the vocabulary gap → `POST /ask`
- **Reranks with a cross-encoder** - `BAAI/bge-reranker-base` scores all top-30 `(query, chunk)` pairs in one batched tensor pass, surfaces highest-precision chunks → `POST /ask`
- **Compresses each chunk** - drops low-relevance sentences while always preserving the `[chunk-id]` marker line that citation validation depends on → `POST /ask`
- **Surfaces active config** - confirms which modules are wired and what model is pinned → `GET /config`
- **Evaluates the full pipeline** - runs all 10 golden questions and returns groundedness, citation validity/recall, false-answer and false-refusal rates → `POST /eval`
- **Before/after comparison** - runs the same 10 questions through the W5 baseline (hybrid only) and the full W6 pipeline simultaneously, returns paired metrics with per-row pipeline traces → `POST /eval/compare`
- **Browser demo UI** - live query interface with pipeline status chips and "Compare Pipelines" button → `GET /`
- **Renders README as HTML** - dark-themed documentation page → `GET /readme`
- Renders a liveness probe → `GET /health`

---

## 2. Setup (5 min)

> ### ⚠ First run pulls ~2 GB
>
> `pip install -r requirements.txt` resolves cleanly, but `sentence-transformers` drags in **torch (a ~527 MB wheel)**. Then the **first `rerank()` call** downloads the `BAAI/bge-reranker-base` weights, another ~1 GB, into your HuggingFace cache.
>
> **Budget ~2 GB of disk and a coffee for the first run.** After that it is instant, and the reranker runs on CPU at ~110 ms p95 for a 30-pair batch - no GPU required.
>
> The tests pass in a light venv without any of this, because the `CrossEncoder` import is lazy (`reranker.py:39`). That is deliberate, and it is worth copying: **your CI should not download a gigabyte of model weights to run twenty-one smoke tests.**

> Requirements: Python 3.10+, OpenAI API key.

```bash
# 1. Create and activate a venv
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env file and fill in real values
copy .env.example .env
# Open .env and set: OPENAI_API_KEY

# 4. Run the smoke tests (no API key needed for these)
pytest -q

# 5. Start the server
uvicorn app.main:app --reload --port 8000
```

**You're live at `http://localhost:8000`.**

- Browser UI: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- README: `http://localhost:8000/readme`

> **HuggingFace model download:** `BAAI/bge-reranker-base` is lazy-loaded on the first call to `rerank()`. The first request that hits the reranker will block for 20-40 s while the model downloads (~1 GB). Subsequent calls use the local HuggingFace cache. Set `HF_HOME` in your environment to control where the cache lands.
>
> **Vector index:** retrieval runs against the Week 4 KnowledgeVault Qdrant collection. `_dense_search` in `app/retriever.py` serves the HyDE probe (ANN only) and `_hybrid_search` runs dense+BM25+RRF for the original query. Set `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` in `.env` before querying.

---

## 3. File-by-file walkthrough

### `app/config.py` - Settings

All environment variables live in one typed `Settings` class (powered by `pydantic-settings`), cached with `@lru_cache(maxsize=1)`. `.env` is read exactly once at startup.

```python
settings = get_settings()
print(settings.model_id)              # "gpt-5.4-mini-2026-03-17"  (override with MODEL_ID)
print(settings.hyde_model)            # "gpt-5.4-nano-2026-03-17"              (override with HYDE_MODEL)
print(settings.hyde_enabled)          # True
print(settings.reranker_enabled)      # True   (RERANKER_ENABLED=false skips the cross-encoder)
print(settings.reranker_model)        # "BAAI/bge-reranker-base"
print(settings.compressor_keep_fraction)  # 0.80
print(settings.vector_top_k_wide)     # 30
print(settings.vector_top_k_narrow)   # 5
```

Two reasons to centralise config here:
1. Reading `os.environ["KEY"]` from arbitrary places is how secrets end up in logs.
2. `pydantic-settings` validates types on startup - a missing or wrong-typed key fails loudly before the first request, not mid-flight.

**Model-pinning:** `gpt-5.4-mini-2026-03-17` and `gpt-5.4-nano-2026-03-17` are explicit version strings, never `"latest"`. A model upgrade is a deliberate action with its own before/after benchmark run.

---

### `app/schemas.py` - Data contracts

```python
class Chunk(BaseModel):
    chunk_id: str          # stable ID - citations point here; first line carries the marker
    text: str              # "[chunk-id]\nbody text..."
    score: float = 0.0     # most recent relevance score (vector or rerank)
    source: str = ""       # originating document filename

class QueryRequest(BaseModel):
    query: str                          # min_length=1 - empty strings rejected at Pydantic layer
    conversation_history: list[str] = []

class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"] = "medium"
    fallback_triggered: bool = False
    pipeline_trace: PipelineTrace | None = None  # full trace attached to every /ask response
```

`ChunkTrace` and `PipelineTrace` capture intermediate pipeline state - HyDE probe text, candidate counts before/after union, the full reranking table (vector score → cross-encoder score with move arrows), and the final compressed chunks sent to the LLM. Attached to every `/ask` response and embedded in `/eval/compare` per-row detail.

`GoldenRow`, `EvalRow`, `EvalMetrics` power `/eval`. `CompareRow`, `CompareMetrics`, `PipelineMetrics` power `/eval/compare`.

**Schema-first habit:** define the shape first, build the pipeline around it. Pydantic validates every response - invalid output becomes a clean 422/502, never silent garbage downstream.

---

### `app/tools.py` - Tool definition + dispatcher

The `retrieve_chunks` tool follows the `{schema, impl}` pair pattern from Week 1. The JSON schema and Python implementation live side by side as a dict - tools are data, not classes.

```python
TOOLS = {
    "retrieve_chunks": {
        "schema": { "name": "retrieve_chunks", "input_schema": { ... } },
        "impl":   _retrieve_chunks_impl,
    }
}
```

`execute_tool(name, args)` is a 10-line dispatcher - no agent framework. RAGOptimizer is a retrieval improvement, not an agent loop.

---

### `app/hyde.py` - HyDE query transformation

`hyde_probe(query)` asks `gpt-5.4-nano` to write a four-sentence technical answer in the voice of the corpus (configurable via `HYDE_PROMPT_VOICE`, default: `"academic_researcher"`). Only the embedding of that answer is used as the retrieval probe - the text is immediately discarded.

**Why `gpt-5.4-nano` here?** The probe content is discarded after embedding. Generation quality is less important than speed and cost - nano is the right tool for a step where content doesn't matter.

**Failure guarded:** HyDE neighbourhood drift - the model writes a plausible answer about the wrong subsystem. `retriever.py` guards against this by unioning HyDE candidates with original-query candidates and letting the reranker handle precision.

---

### `app/reranker.py` - Cross-encoder rerank

`rerank(query, chunks)` scores every `(query, chunk)` pair with `BAAI/bge-reranker-base` and returns chunks sorted by score, descending. The model is lazy-loaded on first call; CPU thread count is pinned via `OMP_NUM_THREADS` and `MKL_NUM_THREADS`.

**Why batch?** Calling the cross-encoder one pair at a time adds ~3-4 ms per pair. With top-30 candidates, serial scoring takes 90-120 ms and, combined with HyDE's ~380 ms call, pushes p95 past the 2000 ms SLA. Batching all 30 pairs as one tensor (`batch_size=30`) runs a single forward pass in ~110 ms total.

**Toggle:** `RERANKER_ENABLED=false` skips the cross-encoder entirely and keeps the union order. Both `retrieve()` and `retrieve_with_trace()` honour it, so `/ask`, `/eval` and `/eval/compare` never disagree about whether the reranker ran, and `/config` reports the flag so you can confirm it before a measurement.

**Failure guarded:** Reranker latency blowout from one-pair-at-a-time scoring.

---

### `app/compressor.py` - Extractive context compression

`compress(chunks, query)` scores body sentences by lexical overlap with the query and keeps the top fraction (`COMPRESSOR_KEEP_FRACTION`, default: 0.80). The `[chunk-id]` marker on the first line of every chunk is always preserved - it is split off before scoring and written back regardless.

**The invariant:** only body sentences are eligible for removal. This is enforced by `test_compressor_preserves_marker_line` in CI - the test passes a chunk whose marker has zero query overlap and asserts it survives.

**Failure guarded:** Compressor strips the citation marker - marker scores low, naive compressor drops it, downstream citation lookup breaks entirely.

---

### `app/retriever.py` - Pipeline composer

`retrieve(query)` is the only module that knows about all four pieces:

```
query
  → (if HyDE enabled)  hyde_probe() → _dense_search()  → HyDE candidates (top-30, ANN only)
  →                    _hybrid_search()                → Original candidates (top-30, dense+BM25+RRF)
  →                    _union_dedupe()  → Rank-interleaved, deduped, TRUNCATED to top-30 (vector_top_k_wide)
  →                    rerank()         → 30 (query, chunk) pairs, one batch, sorted by cross-encoder score
  →                    [:top_k]         → Top-5 (vector_top_k_narrow)
  →                    compress()       → Sentence-trimmed chunks, markers preserved
  →                    list[Chunk]       ← same output type as W5 CitationRAG
```

**Why the union is truncated.** HyDE returns 30 candidates and the hybrid channel returns 30, so the raw union can be up to 60 - twice the candidate budget the reranker advertises, and two cross-encoder batches instead of one. `_union_dedupe(..., limit=vector_top_k_wide)` caps it at 30. It merges by **rank**, not by score: the two lists carry incomparable scores (HyDE = dense cosine 0-1; hybrid = RRF ~0.016-0.03), so score-sorting would let the dense channel swamp the hybrid one and quietly undo the whole point of the union guard. Rank-interleaving keeps both channels represented inside the 30.

Everything downstream - the `/ask` route, the citation prompt, the benchmark harness - sees the same `list[Chunk]` it always did. Only the retrieval path changed.

---

### `app/llm.py` - OpenAI SDK wrapper

`generate_text(prompt, *, model, max_tokens, system)` is the sync path used by HyDE. The `model` parameter is optional: callers that omit it get `settings.model_id` (`gpt-5.4-mini-2026-03-17`). HyDE passes `settings.hyde_model` (`gpt-5.4-nano-2026-03-17`) explicitly.

`generate_json(messages, *, max_tokens, model)` uses `response_format={"type": "json_object"}` - the citation prompt always returns parseable JSON without markdown fences. Business code never imports the OpenAI SDK directly; it always imports from this module.

---

### `app/main.py` - FastAPI routes

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Serves `index.html` browser demo UI |
| `/readme` | GET | Renders `README.md` as a dark-themed HTML page (`markdown` library) |
| `/health` | GET | Liveness probe - returns `status`, `service`, `model`, and `models` dict for the UI health chip |
| `/config` | GET | Surfaces active settings: model IDs, HyDE toggle + voice, reranker model + batch size, compressor toggle + keep fraction, top-k values |
| `/ask` | POST | Full pipeline: HyDE → hybrid → rerank → compress → citation LLM → `GroundedAnswer` with `pipeline_trace` |
| `/eval` | POST | Runs `golden_dataset.json` through the full pipeline; returns five-metric dashboard + per-row detail |
| `/eval/compare` | POST | Runs every golden question through **both** W5 baseline and W6 full pipeline, sequentially row by row; returns paired metrics + per-row side-by-side answers with W6 pipeline trace |

**Honest read on the five metrics.** They are deterministic proxies, not statistics. Four of the five use all-10-rows denominators, so a single row moves them by exactly 0.10. **`citation_validity` does not**, and that is deliberate: it divides only by the rows where the model actually generated an answer, because a row the threshold gate refused never exercised citation discipline and counting it as a pass is how a denominator lies. `groundedness` is a token-overlap proxy that counts correct refusals as grounded.

**On `citation_validity`, and why it is not called precision.** The first draft of `_judge_row()` scored `cited_ids.issubset(must_cite_set | cited_ids)`, which is `True` for every possible input, because a set is always a subset of a union containing itself. That is the judge-that-cannot-say-no this course caught in Week 5, and it came back in the very next build. It is fixed here by measuring **upstream of the enforcement that hides it**: the validator counts the citations it rejects (`GroundedAnswer.citations_dropped`), so the metric asks whether the model cited chunks it was actually sent, and can answer no. `tests/test_citation_validity.py` is the guard that stops it returning a third time.

---

### `week6_notebook.ipynb` - Code walkthrough notebook

A Jupyter notebook covering every endpoint and every module directly:

| Section | curl (`%%cmd`, Windows) | Python (`requests` / direct import) |
|---|---|---|
| Health check | ✓ | ✓ |
| Active config | ✓ | ✓ |
| HyDE module demo | - | ✓ (direct `hyde_probe()` call) |
| Reranker demo | - | ✓ (direct `rerank()` call with sample chunks) |
| Compressor demo | - | ✓ (direct `compress_chunk()` + marker invariant) |
| Full pipeline trace | - | ✓ (direct `retrieve()` call) |
| `/ask` endpoint | ✓ | ✓ |
| `/eval` endpoint | ✓ | ✓ |
| `/eval/compare` endpoint | - | ✓ |
| Failure: 503 no candidates | - | ✓ |
| Failure: 422 empty query | ✓ | ✓ |
| Failure: marker stripping | - | ✓ |
| Swagger UI link | - | ✓ |

---

### `index.html` - Browser UI

Open at `http://localhost:8000` after starting the server.

**Left panel:**
- **Query** textarea (`id="notes"`) - type any retrieval question
- **Demo pills** - one-click fill for four sample queries from the Attention paper (scaled dot-product attention, multi-head attention, positional encoding, encoder-decoder)
- **Pipeline status chips** - auto-loaded from `/config` on startup; green = enabled, red = disabled. Chips show HyDE voice, reranker model short name, and compressor keep fraction. Model labels (HyDE probe / main) shown below.
- **Ask RAGOptimizer** (primary) - `POST /ask`, renders the `GroundedAnswer` card with citations, confidence badge, and full pipeline trace (HyDE probe text, retrieval counts, reranking table, final selection).
- **Compare Pipelines** (secondary) - `POST /eval/compare`, runs all 10 golden questions through both the W5 baseline and the full W6 pipeline, renders a delta metric table and per-row side-by-side answers with expandable W6 pipeline trace.

**Right panel:**
- **Output** pane - structured response with `{ } Format` / `≡ Raw` toggle
- **Logs** pane - timestamped request log

**Header:**
- Health chip - auto-checks on load, shows both model names in the log (`main: gpt-5.4-mini-2026-03-17 | hyde: gpt-5.4-nano-2026-03-17`)
- **📖 README** - opens `http://localhost:8000/readme` in a new tab

---

## 4. Try it out

### a) Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"ragoptimizer","model":"gpt-5.4-mini-2026-03-17","models":{"main":"gpt-5.4-mini-2026-03-17","hyde":"gpt-5.4-nano-2026-03-17"}}
```

### b) Active config - confirm what is wired before benchmarking

```bash
curl http://localhost:8000/config
```

Expected:
```json
{
  "model_id": "gpt-5.4-mini-2026-03-17",
  "hyde_enabled": true,
  "reranker_enabled": true,
  "reranker_model": "BAAI/bge-reranker-base",
  "compressor_enabled": true,
  "vector_top_k_wide": 30,
  "vector_top_k_narrow": 5
}
```

Run this before every benchmark run. If `hyde_enabled` is accidentally `false`, the before/after delta is meaningless.

### c) Ask endpoint

```bash
curl -X POST http://localhost:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"what is scaled dot-product attention\"}"
```

Returns a `GroundedAnswer` with citations, confidence level, and a full `pipeline_trace` showing the HyDE probe text, retrieval counts, reranking table, and final compressed chunks sent to the LLM. Set `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` in `.env` to connect to the real vector index.

### d) Before/after comparison

```bash
curl -X POST http://localhost:8000/eval/compare
```

Runs all 10 golden questions through both pipelines (~60–120 s). Returns:
- `baseline` - CitationRAG hybrid (dense+BM25+RRF, no HyDE, no reranker, no compressor) **plus CitationRAG's threshold gate**, re-implemented verbatim: refuse before the LLM when `top1_dense < similarity_threshold` (0.55) or `spread < spread_delta` (0.08). Without the gate the baseline cannot refuse, so the false-refusal metric measures nothing.
- **The gate is symmetric.** The full W6 pipeline applies the *same rule at the same values*, evaluated against **both** of its retrieval channels:

  ```
  raw_ok  = top1_raw  >= similarity_threshold and spread_raw  >= spread_delta
  hyde_ok = top1_hyde >= similarity_threshold and spread_hyde >= spread_delta
  refuse unless (raw_ok or hyde_ok)
  ```

  Both columns gate, so the delta measures **retrieval quality** rather than gate removal. The rule is identical; only the *inputs* differ, because a second probe is what Week 6 actually built. Gating W6 on the raw query alone would reproduce the baseline's decision on every row by construction and test nothing. HyDE earns its place on rows where the raw query fails the gate and the probe clears it. It pays for that on rows where a drifted probe clears the gate on the wrong chunk, turning a safe refusal into a confident wrong answer (`false_answer`). That trade is the week's teaching point. The same gate applies to `/ask` and `/eval`, so the endpoint and the eval agree on what the pipeline does. All four floats (`top1_raw`, `spread_raw`, `top1_hyde`, `spread_hyde`) are on the `PipelineTrace`. Note the shipped `index.html` does **not** render them; they are visible in the `/ask` and `/eval/compare` JSON and in the server log line from `_log_gate()`. Surfacing them in the trace panel is the first extension in §9.
- `full` - RAGOptimizer (HyDE + hybrid + rerank + compress)
- `rows` - per-question paired results with RAGOptimizer pipeline trace embedded

**What it measured on this golden set** (10 questions, both pipelines gated on the same rule at the same values):

| metric | baseline (CitationRAG hybrid + gate) | full (RAGOptimizer) | Δ |
|---|---|---|---|
| `groundedness` | 0.8 | 1.0 | +0.2 |
| `citation_validity` | 1.0 | 1.0 | 0.0 |
| `citation_recall` | 0.8 | 1.0 | +0.2 |
| `false_answer_rate` | 0.0 | 0.0 | 0.0 |
| `false_refusal_rate` | 0.2 | 0.0 | −0.2 |

**Every one of those deltas traces to the same two rows.** Eight of ten rows land identically in both
columns (including rows #9 and #10, which correctly refuse the off-topic questions). Rows **#3** and
**#5** both improved - the baseline refused each, W6 recovered both. Nothing else moved. **Two
questions out of ten is directional, not statistical** - report the sample size in the same breath as
the delta, or the delta is a lie of omission. §6(d) below walks the #3 recovery from the trace floats.

**Read `citation_validity` as a guardrail, not a score.** It reads 1.0 in both columns because on
this run the model cited only chunks it was actually sent, in every row where it generated at all.
That is worth having: it is a live assertion that the citation contract held. It is **not** evidence
that the answers got better, and it is not a tautology any more - fabricate one citation and it
drops below 1.0, which `tests/test_citation_validity.py` demonstrates. Do not report it as "flat - invariant
held" and let it pass for a quality signal.

**The `baseline` column is a real Week 5**, threshold gate included. A "before" column that cannot
refuse is not the before: it scores 0.0 on `false_refusal_rate` for free, and no retrieval change
could ever move the metric.

Or click **Compare Pipelines** in the browser UI for the visual delta table.

---

## 5. Diagrams

| File | What it shows |
|---|---|
| `ragoptimizer_architecture.svg` | CitationRAG service with the three new modules (HyDE, reranker, compressor) wired in as a retrieval layer |
| `before_after_results_dashboard.svg` | Before/after comparison table: the five product metrics (groundedness, citation validity/recall, false answer/refusal rates) per pipeline, with the per-question strip showing that the whole delta is rows #3 and #5, both recovered by W6 |

---

## 6. Common failure modes

### a) HyDE neighbourhood drift

The model writes a confident hypothetical answer about a different subsystem - "rotation" triggers a response about log rotation, not JWT key rotation. The probe vector lands in the wrong neighbourhood and top-30 candidates are confidently wrong.

**Fix in this codebase:** `retriever.py` runs the HyDE probe through `_dense_search` and the original raw query through `_hybrid_search`, unions the candidate sets (deduped by `chunk_id`, highest score wins), and lets the reranker resolve precision from the combined set. HyDE drift degrades precision; it no longer destroys recall. The symmetric threshold gate is the second line of defence: a drifted probe usually leaves its candidates bunched, and `spread_hyde` catches that before the LLM ever sees them.

**How much did drift cost on this run? Nothing measurable.** `false_answer_rate` held at 0.0 in both columns - nothing drifted into a confident wrong answer (§6(d)). Drift is a real failure mode and the union guard plus the gate are why you build for it; this particular 10-row run is evidence that the guards *worked*, not evidence that drift is harmless. Do not read either way past ten rows.

### b) Reranker latency blowout

Scoring each `(query, chunk)` pair in a separate forward pass adds ~3-4 ms per pair - 90-120 ms in serial for top-30. Combined with HyDE's ~380 ms generation call, p95 exceeds the 2000 ms SLA.

**Fix in this codebase:** `reranker.py` passes all 30 pairs to `predict(pairs, batch_size=30)` - one tensor, one forward pass, ~110 ms total. `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are pinned to `reranker_cpu_threads` (default: 4) so the model does not contend with FastAPI worker threads for CPU cores.

### c) Compressor strips the citation marker

The `[chunk-id]` marker on the first line of every chunk has zero lexical overlap with most queries. A naive compressor scores it low and drops it. Every downstream citation lookup then fails - the validator cannot find the chunk by ID.

**Fix in this codebase:** `compressor.py` splits chunk text into `marker + body` before scoring. Only `body` sentences are scored and eligible for removal. The marker is always written back. `test_compressor_preserves_marker_line` enforces this invariant in CI.

### d) HyDE precision drift on specific fact lookups

HyDE generates a **hypothetical answer** and retrieves by embedding that answer rather than the raw query. For broad conceptual queries ("how does multi-head attention work?") this closes the vocabulary gap and improves recall. For narrow factual lookups, it can hurt precision - the model writes a plausible-sounding hypothetical that is about the right *topic* but the wrong *specific fact*, pulling the retrieval probe into an adjacent neighbourhood.

**How the gate guards against it:** the symmetric threshold gate evaluates *both* the raw query and
the HyDE probe, and a drifted probe typically leaves its candidates bunched - so `spread_hyde` falls
short of the 0.08 floor and the row refuses rather than shipping a confident wrong answer. On the
current 10-row run nothing drifted into a failure: `false_answer_rate` held at 0.0 in both columns
and the two previously-refused rows (#3 and #5) were both recovered. Drift stays the failure mode to
watch as the set grows - the union guard and the spread check are the two lines of defence built for it.

**Mitigations (trade-offs, not bugs to fix):**

| Option | Effect |
|---|---|
| Increase `VECTOR_TOP_K_NARROW` (5 → 8) | More chunks reach the LLM; right chunk more likely included. Increases token cost and LLM latency. |
| Tune `HYDE_PROMPT_VOICE` | A more factual/specific voice produces narrower probes. May degrade recall on broader queries. |
| Disable HyDE (`HYDE_ENABLED=false`) | Removes the drift risk entirely - and costs you row #3, where the probe is the only reason the pipeline answered at all. Measure both rows before you decide. |
| Swap lexical compressor for embedding-based | Stops sentences being dropped on vocabulary mismatch; does not fix retrieval order. |

**The row the probe recovered - row #3:**

> *Q: "How does the decoder prevent a position from using information from later positions when making predictions?"*

| channel | `top1` (floor 0.55) | `spread` (floor 0.08) | verdict |
|---|---|---|---|
| raw query | 0.579 ✓ | 0.0676 ✗ | `raw_ok = False` → **baseline refused** |
| HyDE probe | 0.757 ✓ | 0.1796 ✓ | `hyde_ok = True` → **W6 answered** |

Read the raw row carefully, because the obvious reading is the wrong one. The raw query's top-1
**cleared** the floor. This was **not** a recall failure - the right chunk was already there. It died
on **spread**: its candidates were bunched, i.e. ambiguous. The HyDE probe's contribution here is
**disambiguation, not recall** - it produced a clear winner where the raw query produced a tie.

The reranker then earned its own keep on the same row: it promoted `chunk1` (the
decoder-masking chunk) to rank 1 at **0.9916**, over `chunk2` at **0.758**. The final answer
cites both. Two components, two distinct jobs, one row - HyDE got past the gate, the reranker picked
the chunk.

**Teaching point:** HyDE is not universally better, and its benefit is not the one people assume. It trades keyword-match precision for semantic recall - but on row #3 it bought neither: it bought *disambiguation*, on a query whose right chunk was already retrieved. The before/after dashboard (`/eval/compare`) makes the trade visible in both directions - `B: FAIL / W6: PASS` rows are where the probe cleared a gate the raw query could not (rows #3 and #5). A row that refuses in *both* columns is where a probe drifted and the spread check caught it - none did on this run, but that is the shape to watch for. Read the four trace floats (`top1_raw`, `spread_raw`, `top1_hyde`, `spread_hyde`), not the verdict: they tell you *which* check failed, and "top-1 fine, spread short" is a completely different diagnosis from "top-1 short".

---

## 7. Run the tests

```bash
pytest -q             # all 21 tests, no API key needed
pytest -q -k marker   # just the marker-preservation invariant
```

21 tests - no real API calls, no vector index needed:

| File | Tests | What it checks |
|---|---|---|
| `tests/test_endpoint.py` | 5 | `GET /health` returns 200 with `status == "ok"`, `service == "ragoptimizer"` and both model keys; `GET /config` returns 200 with a pinned `model_id`, `hyde_enabled` and `reranker_model`; `compress_chunk()` never strips the `[chunk-id]` marker regardless of query overlap (the Failure #3 invariant); `compress_chunk()` drops low-relevance body sentences while keeping high-overlap ones; and `GET /config` reports `reranker_enabled`, the toggle the guided lab switches |
| `tests/test_gate.py` | 10 | The threshold gate and the two-channel rule: refuse when `top1 < 0.55` or `spread < 0.08`; the full pipeline answers when either channel passes; the baseline gates on the raw channel only; both columns use the same values |
| `tests/test_citation_validity.py` | 6 | The metric must be able to say no: the old `issubset(must_cite | cited)` expression is always true (asserted, as a permanent record); a fabricated citation drops the metric below 1.0; gate-refused rows leave the denominator; each pipeline gets its own denominator in the comparison |

> The suite covers the wiring contract, the gate rule, and the two invariants this week depends on: the marker line survives compression, and the citation metric can fail.

---

## 8. Guided lab - the four retrieval-layer metrics

The concept video closes on the eval-first lens and hands you this lab:

> *"In this week's guided lab, these four retrieval-layer numbers - recall at five, p fifty, p ninety five, tokens per query - are yours to compute against the golden set as the lab extension."*

The five metrics on `/eval` and `/eval/compare` are **product-layer**: they grade the *answer*.
These four are **retrieval-layer**: they grade the *pipeline that fed the answer*. When
groundedness drops, these are the numbers that tell you whether retrieval or generation broke.

| Metric | Definition in this lab | Why it matters |
|---|---|---|
| **recall@5** | fraction of answerable golden rows whose `_source_chunk_id` appears in the final top-5 | if the right chunk never arrives, no prompt can save you |
| **p50 latency** | median wall-clock time of `retrieve()` | what a typical user feels |
| **p95 latency** | 95th-percentile wall-clock time of `retrieve()` | what your angriest user feels - always report the tail |
| **tokens/query** | context tokens injected into the prompt, per query | the line item nobody tracks until finance asks |

**The ground truth is already in the dataset.** Week 5's `build_golden_dataset.py` generated each
answerable question *from a specific indexed chunk* and recorded it as `_source_chunk_id`. That is
the chunk the retriever must find. The two refusal rows (`must_cite == []`) carry no source chunk
and are excluded from the recall denominator - a refusal row has nothing to recall.

**Run it:** `week6_notebook.ipynb` § 15 (*Guided Lab*) has the whole thing in two cells - it calls
`retrieve()` and `retrieve_baseline()` in-process over `data/golden_dataset.json`, times each call,
estimates injected tokens at ~4 chars/token, and prints the W5-baseline-vs-full-W6 table:

```
pipeline        recall@5   p50 ms   p95 ms   tokens/query
--------------------------------------------------------
W5 baseline        ...       ...      ...            ...
full W6            ...       ...      ...            ...

delta: recall@5 ... - p95 ... ms - tokens/query ...
```

Read it like an engineer, not a fan:

- **recall@5 up, p95 inside your budget** → the pipeline earned its keep.
- **recall@5 flat, p95 up** → you bought latency and got nothing. Turn HyDE off (`HYDE_ENABLED=false`) and prove it.
- **tokens/query down at equal recall** → compression is doing real work.

Three caveats worth internalising, because they generalise well past this repo:

1. **Ten rows resolve to 0.10 on recall@5** - one row. Do not report a "10-point improvement" off a
   10-row golden set. Grow the set before you grow the claim. Week 7 (BreakRAG) makes that argument
   properly, with a power calculation.
2. **The first `retrieve()` call pays the reranker model load.** Discard the first row, or make one
   warm-up call, before you trust a p50.
3. **`tokens/query` is estimated at 4 chars/token.** Swap in `tiktoken` for an exact count - but the
   metric's job is the trend, not the third decimal place.

**Extensions, in order of value:** turn HyDE off and re-measure (on an academic corpus like the
*Attention* paper it usually helps; on a corpus already written in your users' voice it can make
retrieval *worse* - measure, don't assume) · sweep `COMPRESSOR_KEEP_FRACTION` (1.00 / 0.80 / 0.50)
and plot tokens/query against groundedness to find the knee · swap extractive compression for
generative and see whether the extra model call buys anything at all.

---

## 9. Where this goes next

The same pipeline is extended and stress-tested across upcoming weeks:

- Wrap this exact retrieval pipeline in an adversarial eval harness. The three failure modes above (HyDE drift, latency blowout, marker stripping) are the first three attack vectors BreakRAG™ probes. Don't delete this folder - BreakRAG™ imports from it.
- Extend the eval harness with prompt-version pinning so quality regressions can be attributed to specific prompt changes, not just model drift.
- Layer SLM routing on top - easy queries routed to `gpt-5.4-nano` without the full reranker pass, hard ones to the full pipeline. The latency cost accepted in this week gets partially reclaimed through cache hits and selective routing.

Don't throw this away - every week builds on it.

---

## Decisions

- **`gpt-5.4-nano-2026-03-17` for HyDE probe, `gpt-5.4-mini-2026-03-17` for the main pipeline.** The probe text is discarded immediately after embedding, so generation quality has no effect on retrieval quality - the fastest, cheapest model wins. The main pipeline uses the higher-capability model only where output quality actually reaches the user.

- **Open-source cross-encoder (`BAAI/bge-reranker-base`), not a managed reranker API.** Running on CPU keeps the latency budget predictable and removes an external network hop per inference, keeping p95 inside the 2000 ms SLA. Cohere Rerank and Voyage Rerank are documented swap-ins for teams that prefer managed infrastructure.

- **Extractive compression, not generative.** Deterministic sentence selection requires no additional model call, has no hallucination risk, and makes the marker-preservation invariant straightforward to enforce. Generative compression is the guided lab extension - students swap it in, re-run the harness, and measure whether the quality gain justifies the extra latency and cost.

- **Union HyDE + original query candidates, not HyDE-only.** HyDE neighbourhood drift is a real failure mode, not a theoretical one. Running both probes and unioning candidates means HyDE drift degrades precision without destroying recall - the original-query candidates carry it, and the reranker resolves which chunks are actually relevant.

- **`{schema, impl}` tool pairs + 10-line dispatcher, no agent framework.** RAGOptimizer improves retrieval quality; it is not an agent loop. Adding LangChain or LlamaIndex would increase dependency weight, obscure the call graph, and make the Week 7 adversarial tests harder to wire. Tools are data, not classes.
