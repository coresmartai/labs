# SpecialistTuner (LoRA) - Week 8

**Applied GenAI & Agentic AI Engineering Course · Week 8**

SpecialistTuner takes a curated domain-specific JSONL dataset (46 Q&A examples on *Attention Is All You Need*), fine-tunes a LoRA adapter on **Qwen/Qwen3-0.6B on a plain laptop CPU** (no GPU required - the same code takes a QLoRA 4-bit path automatically when CUDA is present), and serves the specialist model behind a FastAPI inference endpoint. It demonstrates the two open-source fine-tuning patterns that appear in nearly every production self-hosted LLM pipeline: parameter-efficient adapter training with PEFT/LoRA and lazy model loading with `@lru_cache`. The recipe is identical at 8B scale - only the hardware bill changes.

| Pattern | Endpoint / entry-point | File |
|---|---|---|
| **LoRA fine-tuning (CPU; QLoRA on GPU)** | `python -c "from app.tools import ..."` or `POST /v1/fine-tune/start` | `app/tools.py` |
| **LoRA adapter inference** | `POST /v1/chat/completions` | `app/llm.py` |
| **Completion-only loss masking** | `make_trainer(model, dataset, ...)` | `app/tools.py` |

---

## Project layout

```
/
├── app/
│   ├── __init__.py
│   ├── config.py             ← typed settings + .env loader (HF_TOKEN, BASE_MODEL, ports)
│   ├── schemas.py            ← ChatMessage, TrainingExample, InferenceRequest, InferenceResponse, EvalResult
│   ├── tools.py              ← 5-function training pipeline + execute_tool dispatcher
│   ├── llm.py                ← HuggingFace inference with lazy imports (no GPU stack at import time)
│   └── main.py               ← FastAPI routes + CORS + /readme
├── tests/
│   └── test_endpoint.py      ← smoke tests (no GPU, no real Hub calls)
├── data/
│   ├── specialisttuner.jsonl ← training data (46 examples, ships in the repo)
│   └── eval_aiayn.jsonl      ← 42-question eval benchmark (ships in the repo)
├── out/                      ← out/checkpoint-82 = config + loss curve of the reference run (no weights)
├── index.html                ← browser UI (open via http://localhost:8000)
├── week8_notebook.ipynb      ← curl + Python requests for every endpoint
├── requirements.txt
├── WebUI_finetuning_eval.png ← screenshot used in this README
├── WebUI_finetuning.png      ← screenshot used in this README
├── .env.example              ← copy to .env and fill in
├── .gitignore
└── README.md                 ← you are here
```

---

## 1. What this app does

- Loads Qwen3-0.6B - full fp32 on CPU (no GPU needed), or 4-bit QLoRA automatically when CUDA is present - and wraps it with a LoRA adapter targeting the four attention projections (r=8, α=16 → **~2.3M trainable params, 0.38% of the model**) → `python -c "from app.tools import ..."`
- Fine-tunes with completion-only loss masking - gradients flow only through assistant turns, not instruction tokens → `make_trainer` in `app/tools.py` (custom collator; trl is not a dependency)
- Runs training in a background thread with live progress via `POST /v1/fine-tune/start` / `GET /v1/fine-tune/status`
- Pushes the trained adapter (**~9.2MB**) to the Hugging Face Hub → `push_adapter` in `app/tools.py`
- Serves the adapted model via `POST /v1/chat/completions` with a cached model load (no per-request init overhead)
- Scores the model against the 42-question AIAYN eval set via `POST /v1/eval/run`
- Serves a **browser UI** at `GET /` with three tabs (Inference / Fine-Tune / Eval) - no separate server needed
- Renders this README as dark-themed HTML at `GET /readme`

---

## 2. Setup (5 min)

> Requirements: Python 3.10+. **No GPU required** - Qwen3-0.6B trains and serves on a modern laptop CPU (training: under an hour for the shipped 46-example / 4-epoch run). Qwen/Qwen3-0.6B is not gated; an HF token is only needed to push your adapter to your own namespace. A CUDA GPU is optional and is used automatically (QLoRA 4-bit) when present.

```bash
# 1. Create and activate a venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env file and adjust if needed (defaults work out of the box)
cp .env.example .env
# Optionally set HF_TOKEN + HF_HUB_REPO if you plan to push the adapter to the Hub

# 4. The training dataset ships in the repo at ./data/specialisttuner.jsonl
#    Each line: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

# 5. Start the server (adapter optional on first boot)
uvicorn app.main:app --reload
```

**You're live at `http://localhost:8000`.**

- Browser UI: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- README: `http://localhost:8000/readme`

> **No gated access needed:** `Qwen/Qwen3-0.6B` downloads without approval. An `HF_TOKEN` is only required for `push_adapter` (WRITE access to your namespace). The server imports the model lazily - startup never blocks on the download; the first inference call triggers it.
>
> **Adapter location:** train first. `out/checkpoint-82` ships as a **reference record only** - `adapter_config.json` and `trainer_state.json`, so you can read the configuration and the loss curve of the run this README describes. It carries no weights and cannot be loaded. Once your own training run finishes, your best checkpoint lands under `./out/`; copy it to `out/adapter`, or pass the checkpoint path directly via the `?adapter=` query parameter or the UI dropdown.
>
> **CPU-only machine?** That's the intended setup. Training takes under an hour on a modern laptop CPU; inference takes a few seconds per request. Run `pytest -q` to verify the wiring without downloading the model at all.

---

## 3. File-by-file walkthrough

The order here matches the reading order for tool-based weeks: `config → schemas → tools → llm → main`.

### `app/config.py` - Settings

All environment variables in one typed `Settings` class (`pydantic-settings`). The structure is fixed across every week - only the field names and defaults change.

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        extra="ignore", protected_namespaces=(),
    )
    hf_token: str = ""
    hf_hub_repo: str = "your-org/specialisttuner-qwen3-lora"
    base_model: str = "Qwen/Qwen3-0.6B"
    dataset_path: Path = Path("./data/specialisttuner.jsonl")
    output_dir: Path = Path("./out")
    log_dir: Path = Path("./logs")
    model_host: str = "0.0.0.0"
    model_port: int = 8000
    log_level: str = "INFO"
```

`@lru_cache(maxsize=1)` means `.env` is read exactly once per process.

> **Model-pinning:** `base_model` stores the HF repo slug - treat it like a dated model pin. Changing it here is the single place that propagates through all five training functions and the inference path.

### `app/schemas.py` - Pydantic models

```python
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]   # Literal guards bad role values at the boundary
    content: str

class TrainingExample(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=2)  # at least user + assistant

class InferenceRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = 256
    temperature: float = 0.2

class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # suppresses pydantic "model_id" warning
    output: str
    model_id: str
    latency_ms: float
    finish_reason: Literal["stop", "length", "error"] = "stop"
```


### `app/tools.py` - Training pipeline

Five composable functions; each can be called independently so a failed step can be re-run without restarting from scratch.

| Function | What it does |
|---|---|
| `load_base_model()` | Loads Qwen3-0.6B - fp32 on CPU; `BitsAndBytesConfig` (nf4, bfloat16 compute) when CUDA is present |
| `wrap_with_lora(model, r, alpha)` | Applies `LoraConfig` to q/k/v/o projections, calls `get_peft_model` (~2.3M trainable, 0.38%) |
| `load_and_tokenize_dataset(path, max_len)` | Reads JSONL, applies the model's chat template (ChatML for Qwen3), tokenises, 90/10 split |
| `make_trainer(model, dataset, output_dir, epochs, lr)` | Builds `TrainingArguments` + a custom completion-only collator + `Trainer` (batch 1 on CPU; batch 4 × grad-accum 4 on GPU) |
| `push_adapter(model, repo)` | Saves adapter locally then calls `push_to_hub` |

A `TOOLS` dict and a 10-line `execute_tool(name, args: dict)` dispatcher give the pipeline a callable interface without requiring an agent framework - e.g. `execute_tool("load_dataset", {"jsonl_path": Path("data/specialisttuner.jsonl")})`. Unknown tools and tool exceptions return an error dict (`{"success": False, "error": ...}`) instead of raising.

**Completion-only collator:** trl removed `DataCollatorForCompletionOnlyLM` in 1.x, so `tools.py` ships its own ~30-line version (`_make_completion_collator`). It cuts on the **assistant response template** (`<|im_start|>assistant\n` for Qwen3 - `_get_response_template` picks the right header per model family) and sets labels to `-100` for everything before it **and for all pad positions**, so only assistant tokens contribute to the loss.

**Key design choice:** all HuggingFace imports (`torch`, `transformers`, `peft`, `datasets`) are deferred inside each function - lazy imports. This means `from app.tools import execute_tool` works on a dev machine without the ML stack installed; tests can patch `generate` without requiring it.

### `app/llm.py` - HuggingFace inference

Three public functions (the first two `@lru_cache`d):
- `get_tokenizer()` - loads the tokenizer once per process and adds a dedicated `[PAD]` token if none exists (never eos-as-pad - see §6b)
- `get_model(adapter_path)` - loads the base model (fp32 on CPU / 4-bit on GPU) and applies the PEFT adapter once per process; `adapter_path=None` = base model only
- `generate(messages, adapter_path, max_new_tokens, temperature)` → `dict` with `output`, `latency_ms`

All `torch`, `transformers`, and `peft` imports are lazy (inside the functions), for the same reason as `tools.py` - tests import this module without the ML stack.

### `app/main.py` - FastAPI routes

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Serve browser UI (`index.html`) |
| `/health` | GET | Liveness probe - returns `{"status":"ok","model":"..."}` |
| `/readme` | GET | Render README.md as dark-themed HTML |
| `/v1/chat/completions` | POST | Run inference; optional `?adapter=` query param (default: base model, no adapter) |
| `/v1/fine-tune/start` | POST | Begin background LoRA training (daemon thread) |
| `/v1/fine-tune/status` | GET | Live training progress - step, epoch, train/eval loss series |
| `/v1/fine-tune/stop` | POST | Signal training to stop at the next step boundary |
| `/v1/checkpoints` | GET | List adapter/checkpoint dirs under `out/` (feeds the UI dropdowns) |
| `/v1/eval/run` | POST | Score `data/eval_aiayn.jsonl` against base or adapter |
| `/v1/eval/results` | GET | Last eval run results (overall + per-category accuracy) |

CORS is open (`allow_origins=["*"]`) - fine for local dev, narrow before production.

### `index.html` - Browser UI

Open at `http://localhost:8000` after starting the server. Three tabs (see `WebUI_finetuning.png` / `WebUI_finetuning_eval.png`):

**🧠 Inference** - message textarea, adapter/checkpoint dropdown (populated from `/v1/checkpoints`), Run Inference button, result card (response text + latency chip), logs pane.

**⚙️ Fine-Tune** - dataset path, r / alpha / epochs / lr inputs, Start/Stop buttons, live step progress and loss readout polled from `/v1/fine-tune/status`.

**📊 Eval** - eval file path, adapter/checkpoint dropdown, Run Eval button, overall accuracy hero number, per-category breakdown, question-by-question results table.

### `week8_notebook.ipynb` - API notebook

Covers every endpoint two ways (cURL `%%cmd` and Python `requests`):

| Section | curl (`%%cmd`) | Python |
|---|---|---|
| Health check | ✓ | ✓ |
| LoRA inference | ✓ | ✓ |
| Full raw response dump | - | ✓ |
| Failure: empty messages (422) | ✓ | ✓ |
| Failure: bad role literal (422) | ✓ | ✓ |
| Swagger UI link | - | ✓ |

All `%%cmd` cells use Windows double-quote syntax (`\"`).


---

## 4. Try it out

### a) Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"Qwen/Qwen3-0.6B+lora"}
```

### b) Train the adapter (any machine - CPU is fine)

```bash
python -c "from pathlib import Path; from app.tools import load_base_model, wrap_with_lora, load_and_tokenize_dataset, make_trainer, push_adapter; m = wrap_with_lora(load_base_model()); ds = load_and_tokenize_dataset(jsonl_path=Path('data/specialisttuner.jsonl')); t = make_trainer(m, ds, output_dir='out', epochs=4); t.train(); push_adapter(m, 'your-org/specialisttuner-qwen3-lora')"
```

The shipped run (46 examples, 4 epochs, batch 1) finishes in under an hour on a modern laptop CPU. Epoch checkpoints land under `./out/`; the trainer keeps the best-eval one, which in the reference run was epoch 2 (`out/checkpoint-82`, whose config and loss curve ship here). `push_adapter` also writes `./out/adapter`. You can also drive training from the browser UI's Fine-Tune tab.

### c) Run inference

Windows cmd (single line):

```bash
curl -X POST "http://localhost:8000/v1/chat/completions?adapter=out/adapter" -H "Content-Type: application/json" -d "{\"messages\": [{\"role\": \"user\", \"content\": \"Why does the Transformer use multi-head attention instead of a single attention function?\"}], \"temperature\": 0.2}"
```

Expected response shape:

```json
{
  "output": "...",
  "model_id": "Qwen/Qwen3-0.6B+lora",
  "latency_ms": 1820,
  "finish_reason": "stop"
}
```

### d) Trigger a 422 (no GPU needed)

```bash
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\": []}"
# 422 Unprocessable Entity - empty list blocked by Pydantic before model load
```

---

## 5. Diagrams

The Week 8 SVGs : `peft_lora_pipeline_architecture.svg`, `qlora_memory_breakdown.svg`, `trainer_loop_checkpoint_cadence.svg`, `three_failure_modes_annotated.svg`, `adapter_vs_merged_flow.svg`, `loss_curve_anatomy.svg`.

---

## 6. Common failure modes

### a) Rank too low (`r=2`)

Training loss plateaus high; eval accuracy barely moves. The LoRA rank is the capacity dial - `r=2` gives the adapter only 2 free dimensions to learn the domain shift. Fix: bump rank to `r=8` or `r=16`, not the learning rate. Capacity is the bottleneck, not speed.

### b) Pad-token confusion

Many open-model tokenizers ship without a dedicated pad token (Llama 3 famously; this code guards for the same case on Qwen). The common fix `tokenizer.pad_token = tokenizer.eos_token` silently breaks training - but not for the reason most tutorials give. The completion-only collator cuts on the **assistant response template** (not on EOS), then masks pad positions out of the loss with label `-100`. Once EOS is also the pad token, the real end-of-sequence token at the end of each answer is masked out along with the padding - so the model **never learns to stop** and rambles past the end of its answers at inference. Fix: `tokenizer.add_special_tokens({"pad_token": "[PAD]"})` then `model.resize_token_embeddings(len(tokenizer))` - already handled in `app/llm.py:get_tokenizer()`. (Note: the shipped collator masks pads via label `-100` regardless of which token plays the pad role - belt and braces - but don't rely on every collator doing that.)

### c) Overfitting (eval loss climbs while train loss dives)

Training loss keeps falling while eval loss bottoms out and climbs - the adapter memorises the training examples rather than generalising. The shipped 4-epoch run shows it directly: train loss 2.86 → ~1.60, eval loss 2.268 → **2.205 (epoch 2, best)** → 2.240 → 2.299. `TrainingArguments` here sets `load_best_model_at_end=True` and `metric_for_best_model="eval_loss"` so the trainer rolls back to the best checkpoint automatically - that's why `out/checkpoint-82` is the artifact to keep. If eval loss peaks early, reduce `num_train_epochs` or increase the held-out split size. Note these are two distinct eval sets: the `eval_loss` series above comes from the 5-example held-out split carved from the 46 training examples (a 41/5 train/val split, scored each epoch during training), whereas the headline accuracy number comes from the separate 42-question `eval_aiayn.jsonl` benchmark run by `POST /v1/eval/run` after training.

---

## 7. Run the tests

```bash
pytest -q
```

5 smoke tests - no GPU, no real Hub calls, no network:

| Test | What it checks |
|---|---|
| `test_health_ok` | `GET /health` returns 200 and `{"status": "ok"}` |
| `test_chat_validates_empty_messages` | Empty `messages` list returns 400 or 422 before any model call |
| `test_chat_happy_path_with_mocked_generate` | `POST /v1/chat/completions` returns 200 and correct `InferenceResponse` shape (generate mocked) |
| `test_eval_result_schema_shape` | `EvalResult` schema accepts the decision-memo row shape |
| `test_eval_run_emits_one_memo_row` | `POST /v1/eval/run` emits exactly one `EvalResult`, with the tuned row carrying 2,293,760 trainable params and the base row carrying zero |

> `generate` is patched in all tests - the HuggingFace stack is never imported. Run `pytest -q` on any machine, GPU or not.

---

## 8. Where this goes next

- Register this LoRA adapter as a specialist worker behind an A2A interface, pulling weights from the Hub model card.
- Containerise this service, adds observability, deploys to cloud.
- Routes per-request between base prompting and this LoRA based on $/quality; this adapter is one of the backends.

---

## Decisions

- **Qwen3-0.6B on CPU over an 8B on rented GPUs.** The lab model is small enough to fine-tune on the laptop you already have - no rental, no gated access, no CUDA setup - and the recipe (LoRA r=8 on q/k/v/o) is identical at 8B; only the hardware bill changes. The code keeps the QLoRA 4-bit GPU path for when you scale up: at Qwen3-8B, QLoRA fits a 12–16GB consumer card and LoRA fp16 fits a 24GB card, while full fine-tuning (~128GB) is multi-GPU territory.
- **Completion-only loss masking over standard causal LM.** Training on instruction tokens too inflates validation loss (the model "learns" to complete prompts it already knows) and wastes gradient steps on the non-specialist parts of the conversation. The assistant header token (`<|im_start|>assistant` in Qwen3's ChatML) is the clean cutpoint.
- **Custom collator over a trl dependency.** trl removed `DataCollatorForCompletionOnlyLM` in 1.x; the ~30-line replacement in `tools.py` does the same job and removes a fast-moving dependency.
- **`@lru_cache` model load over per-request loading.** Model load takes tens of seconds. Caching it means the server starts slow but every subsequent request is fast. The trade-off is that changing the adapter path requires a process restart - acceptable for a teaching server, not for a multi-tenant production service.
- **Lazy imports for `torch` / `transformers` / `peft`.** Deferring these imports inside each function means `from app.tools import execute_tool` works on a dev machine without the ML stack installed. Tests patch `generate` at the function level and never need the full HuggingFace stack; `pytest -q` runs in seconds on a laptop.
