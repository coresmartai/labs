# SpecialistTuner (Vertex AI Gemini), Week 8

**Applied GenAI & Agentic AI Engineering Course · Week 8**

SpecialistTuner takes a curated domain-specific JSONL dataset (**211 Q&A
examples on the paper "Attention Is All You Need"**), converts it from the
app's provider-agnostic generic chat format into Google's Vertex AI Gemini
supervised-tuning JSONL format, uploads it to Cloud Storage, kicks off a
supervised fine-tuning job against **Gemini 3.1 Flash Lite**, polls the
job to completion, and serves the tuned model behind a small FastAPI
surface. The whole pipeline is six composable functions plus a
ten-line dispatcher; no orchestration framework, no agent wrapper.

| Pattern | Endpoint / entry-point | File |
|---|---|---|
| **Generic → Vertex converter** | `POST /convert` | `app/convert.py` |
| **Supervised fine-tuning** | `POST /v1/tune/start` | `app/tools.py:start_tune` |
| **Tuning progress** | `GET /v1/tune/status?job_id=...` | `app/tools.py:poll_tune` |
| **Tuned-model inference** | `POST /predict?tuned_endpoint=...` | `app/llm.py:generate` |

---

## Why two formats?

The app authors its data once in the **generic format**, a plain list of
`{role, message}` turns that any human can read and any Pydantic model can
validate. `app/convert.py` re-emits it in whichever provider's shape you
happen to be targeting this quarter. Today Vertex; tomorrow whoever ships
the next cheap tuning API. The generic file is the asset. The vendor
JSONL is a build artefact.

**Generic (what you write):**

```json
{"messages": [
  {"role": "user",  "message": "What problem does the Transformer solve?"},
  {"role": "model", "message": "The Transformer replaces the sequential..."}
]}
```

**Vertex AI Gemini SFT JSONL (what gets uploaded):**

```json
{"systemInstruction": {"role": "system", "parts": [{"text": "You are..."}]},
 "contents": [
   {"role": "user",  "parts": [{"text": "What problem does the Transformer solve?"}]},
   {"role": "model", "parts": [{"text": "The Transformer replaces the sequential..."}]}
]}
```

---

## Project layout

```
specialisttuner-vertex/
├── app/
│   ├── config.py          ← typed settings + .env loader (GCP project, model pin)
│   ├── schemas.py         ← Message, ChatExample, TuningJobMeta, EvalResult
│   ├── convert.py         ← generic → Vertex JSONL converter (no vendor imports)
│   ├── llm.py             ← Vertex AI SDK wrapper (lazy imports)
│   ├── tools.py           ← 6-function pipeline + execute_tool dispatcher
│   └── main.py            ← FastAPI routes
├── prompts/
│   └── system_instruction.txt   ← teacher persona used in training + inference
├── data/
│   ├── aiayn_generic.jsonl      ← 211 examples (checked in, the source of truth)
│   ├── aiayn_gcloud.jsonl       ← build artefact, not checked in. POST /convert writes it
│   └── eval_aiayn.jsonl         ← 42-question eval set (the same 42 the local track scores)
├── data_gen/
│   └── generate.py              ← rebuilds aiayn_generic.jsonl from a Python source
├── scripts/
│   └── eval_hosted.py           ← scores a tuned endpoint on the 42 questions
├── results/                      ← your memo rows land here, not checked in
├── tests/
│   └── test_endpoint.py         ← 17 smoke + contract tests, no network
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## What this app does

- **Reads** the 211-example dataset in the app's generic chat format (`data/aiayn_generic.jsonl`)
- **Converts** it to Vertex AI Gemini SFT JSONL, injecting the teacher-persona system instruction on every row (`app/convert.py`)
- **Uploads** the converted file to a GCS bucket in the same region as the tuning job
- **Kicks off** a supervised tuning job against `gemini-3.1-flash-lite-001` (pinned; never `*-latest`)
- **Polls** the job through Vertex's tuning API until it reaches a terminal state (`SUCCEEDED` / `FAILED` / `CANCELLED` / `EXPIRED`)
- **Serves** the resulting tuned endpoint behind `POST /predict?tuned_endpoint=...` for the eval harness
- **Scores** the tuned endpoint on the same 42 questions the local track uses, with the same matching rule, via `scripts/eval_hosted.py`. That is the row your memo needs
- **Renders** this README as a dark-themed HTML page at `GET /readme`, which is also where `GET /` sends you

---

## Setup (≈ 5 min)

> Requirements: Python 3.10+, a GCP project with the Vertex AI API enabled, a service account with the `roles/aiplatform.user` + `roles/storage.objectAdmin` roles, and a GCS bucket in the same region as your tuning job (typically `us-central1`).

```bash
# 1. venv
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. deps
pip install -r requirements.txt

# 3. auth (once per machine)
gcloud auth application-default login
gcloud config set project <YOUR_PROJECT_ID>

# 4. env
cp .env.example .env
# edit GCP_PROJECT, GCS_BUCKET, TUNED_MODEL_DISPLAY_NAME as needed

# 5. rebuild the training data (produces data/aiayn_generic.jsonl)
python data_gen/generate.py

# 6. run the tests (no GCP calls, no network)
pytest -q

# 7. start the server
uvicorn app.main:app --reload
```

`http://localhost:8000` redirects to the README. **This track has no browser page of
its own, and that is deliberate.** The local package ships a UI because the training
runs on your machine and there is something to watch. Here the job runs in somebody
else's data centre, and their console is the surface for it: watch the tuning job in
the Vertex AI console under Vertex AI > Tuning, and drive everything else from the API
or from `scripts/eval_hosted.py`.

That asymmetry is not an oversight in the package. It is one of the things you gave up.

---

## Try it out

### a) Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","provider":"vertex-ai","model":"gemini-3.1-flash-lite-001",...}
```

### b) Convert generic → Vertex JSONL

```bash
curl -X POST http://localhost:8000/convert -H "Content-Type: application/json" -d "{}"
# {"input_rows":211,"output_rows":211,"output_path":"./data/aiayn_gcloud.jsonl","warnings":[]}
```

### c) Kick off a tuning job

```bash
# First upload the converted file to GCS:
python -c "from app.tools import execute_tool; \
  print(execute_tool('upload', {'local_path': './data/aiayn_gcloud.jsonl'}))"
# {"success": True, "uri": "gs://your-tuning-bucket/aiayn/train/aiayn_gcloud.jsonl", ...}

# Then start the tuning job:
curl -X POST http://localhost:8000/v1/tune/start -H "Content-Type: application/json" \
  -d "{\"train_dataset_uri\":\"gs://your-tuning-bucket/aiayn/train/aiayn_gcloud.jsonl\",\"epochs\":4}"
```

### d) Poll for status

```bash
curl "http://localhost:8000/v1/tune/status?job_id=projects/.../tuningJobs/1234"
# When state == JOB_STATE_SUCCEEDED, the response includes `tuned_model_endpoint`.
```

### e) Predict against the tuned model

```bash
curl -X POST "http://localhost:8000/predict?tuned_endpoint=projects/.../endpoints/5678" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Why does the Transformer use multi-head attention?\"}"
```

### f) Score it, and get the row your memo needs

```bash
# the tuned endpoint
python scripts/eval_hosted.py --endpoint projects/.../endpoints/5678 \
  --cost-per-1k 0.00 --out results/vertex_ft.json

# the same 42 questions against the untuned model, for the row you compare to
python scripts/eval_hosted.py --base --out results/vertex_base.json
```

The matching rule is character-for-character the one the local package uses in
`POST /v1/eval/run`: a question passes if any accepted answer appears in the output.
Two rows scored two different ways are not comparable, and the whole point of running
both tracks is that the comparison is fair.

Look the per-call price up and pass it as `--cost-per-1k`. It defaults to zero, and a
zero there makes the memo wrong in the one column the hosted path exists to fill.

**Every question is a billed call. Forty-two per run.**

---

## Diagrams

The Week 8 SVGs (produced in the V1 concept video and reused here):
`07_peft_lora_pipeline_architecture.svg`, `13_openai_finetune_lifecycle.svg`,
`14_cost_per_finetune_comparison.svg`, `16_three_row_benchmark_table.svg`,
`17_decision_memo_logic_tree.svg`, `18_privacy_posture_comparison.svg`.

The lifecycle diagram (`13`) is intentionally provider-agnostic. The five
stages (**validate → upload → create job → monitor → retrieve endpoint**)
are the same on Vertex as they were on OpenAI. Only the SDK changes.

---

## Common failure modes

### a) Wrong region for the bucket

Vertex tuning jobs require the training data GCS bucket to be in the **same
region** as the tuning job (typically `us-central1`). If the bucket is in
`us-east1` you'll get a `400 INVALID_ARGUMENT` at `start_tune` with a message
about location mismatch. Fix: create a new bucket in `us-central1` and
re-upload.

### b) `Permission denied` on tuning start

The service account driving the API needs both `roles/aiplatform.user` and
`roles/storage.objectAdmin` on the bucket. Missing either yields a 403 at
either `upload` or `start_tune`.

### c) Adapter size vs dataset size

`adapter_size=1` works with ~50 examples; `adapter_size=4` (our default)
matches the 211 examples in the shipped dataset. Pushing `adapter_size=8`
or higher with only 211 rows overfits, and you'll see eval loss stall or
climb after the first epoch.

### d) Model deprecation

`gemini-3.1-flash-lite-001` is a **dated** identifier. If Google
retires the version your fine-tune was trained against, your tuned
endpoint stops responding. That's why the tuning job records both
`source_model` and the exact date in `TuningJobMeta`. Never use the
undated alias (`gemini-3.1-flash-lite`) in production.

---

## Run the tests

```bash
pytest -q
```

17 smoke + contract tests. **No network. No GCP calls.** `app.llm.generate`
is patched; `execute_tool` targets are patched at the FastAPI seam;
converter tests write to `tmp_path`. Runs in under a second on any laptop.

---

## Where this goes next

- **Week 11 (AgentMesh)** registers the tuned endpoint as a specialist worker behind an A2A interface, pulling the endpoint resource name from the tuning job's metadata.
- **Week 14 (DeployCore)** containerises this service with observability + async jobs.
- **Week 16 (CostGuard)** routes per-request between base Gemini prompting, the tuned Vertex model, and the local LoRA, based on `$/quality` for the specific request. SpecialistTuner is the input that makes that routing decision data-driven.

---

## Decisions

- **Vertex AI Gemini 3.1 Flash Lite over the discontinued OpenAI fine-tuning path.** OpenAI closed new fine-tuning submissions; Vertex is the equivalent hosted API and is available in the region we already provision for the rest of the course. The workflow shape (validate → upload → create job → monitor → retrieve endpoint) is identical, so the V1 mental model transfers directly.
- **Provider-agnostic generic format + a converter.** The team writes data once in a plain `{role, message}` shape; `app/convert.py` re-emits it for whichever provider owns the tuning job this quarter. Making the vendor shape a build artefact rather than the source of truth is what makes moving off a discontinued provider a one-file change.
- **Adapter tuning over full fine-tune.** Vertex's supervised tuning uses an adapter approach internally (`adapter_size` picks the rank family). We surface `adapter_size` as a first-class hyperparameter so the V1 rank intuition applies directly: small dataset, small adapter; large dataset, bigger adapter.
- **`@lru_cache` on `get_settings` + lazy vendor imports.** The FastAPI + tests + converter path never touches the Vertex SDK; that's how `pytest -q` runs in milliseconds on a machine without `google-cloud-aiplatform` installed.
