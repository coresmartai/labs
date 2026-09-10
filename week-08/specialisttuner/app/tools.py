"""Training pipeline as a small set of composable functions.

Pattern: tools are pairs of (config callable, execute callable). A 10-line
dispatcher lets us trigger pieces from a CLI or from tests without an
orchestration framework.

Heavy imports (torch, transformers, peft, datasets) are lazy - deferred
to function call time so the module can be imported without the GPU stack.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Callable

from app.config import get_settings
from app.llm import get_tokenizer

logger = logging.getLogger(__name__)


# ── 1. Load base + wrap with LoRA ────────────────────────────────────────────
def load_base_model() -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    s = get_settings()
    cuda = torch.cuda.is_available()

    if cuda:
        # GPU path: 4-bit QLoRA (requires bitsandbytes + CUDA)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            s.base_model, quantization_config=bnb, device_map="auto",
            token=s.hf_token or None,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # CPU path: full fp32, no quantization (slow but runnable)
        model = AutoModelForCausalLM.from_pretrained(
            s.base_model, torch_dtype=torch.float32,
            token=s.hf_token or None,
        )
    return model


def wrap_with_lora(model, r: int = 8, alpha: int = 16) -> Any:
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    # Sanity check - should be ~0.38% for Qwen3-0.6B (~2.3M params). Run this every time.
    model.print_trainable_parameters()
    return model


# ── 2. Dataset ───────────────────────────────────────────────────────────────
def load_and_tokenize_dataset(jsonl_path: Path, max_len: int = 2048):
    from datasets import load_dataset

    tok = get_tokenizer()
    ds = load_dataset("json", data_files=str(jsonl_path))["train"]

    def to_text(example):
        return {"text": tok.apply_chat_template(example["messages"], tokenize=False)}

    ds = ds.map(to_text, remove_columns=ds.column_names)
    ds = ds.map(lambda e: tok(e["text"], truncation=True, max_length=max_len), batched=True)
    return ds.train_test_split(test_size=0.1, seed=42)


# ── 3. Trainer ──────────────────────────────────────────────────────────────
def _make_completion_collator(tokenizer, response_template: str):
    """Minimal completion-only collator - no trl dependency.

    Finds the last occurrence of `response_template` token IDs in each
    sequence and sets labels to -100 (ignored by cross-entropy) for every
    token before it.  This is exactly what trl.DataCollatorForCompletionOnlyLM
    used to do before it was removed in trl 1.x.
    """
    import torch
    template_ids: list[int] = tokenizer.encode(
        response_template, add_special_tokens=False
    )
    tlen = len(template_ids)
    pad_id: int = tokenizer.pad_token_id or 0

    def collate(features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_ids, batch_attn, batch_labels = [], [], []

        for f in features:
            ids    = list(f["input_ids"])
            attn   = list(f["attention_mask"])
            labels = list(ids)

            # Find the last assistant-header position; mask everything before it.
            cut = 0  # fallback: train on full sequence
            for i in range(len(ids) - tlen, -1, -1):
                if ids[i: i + tlen] == template_ids:
                    cut = i + tlen  # first token the model should predict
                    break
            for j in range(cut):
                labels[j] = -100

            # Right-pad to max_len
            pad = max_len - len(ids)
            ids    += [pad_id] * pad
            attn   += [0]      * pad
            labels += [-100]   * pad

            batch_ids.append(ids)
            batch_attn.append(attn)
            batch_labels.append(labels)

        return {
            "input_ids":      torch.tensor(batch_ids,    dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn,   dtype=torch.long),
            "labels":         torch.tensor(batch_labels, dtype=torch.long),
        }

    return collate


def _get_response_template(base_model: str) -> str:
    """Return the assistant-turn header used by this model family's chat template.

    The custom collator scans for this token sequence and masks everything
    before it so the model only trains on its own responses.

    This lab is Qwen-only. Qwen3 speaks ChatML, and so do most other chat
    models (Phi, Falcon, and friends), so ChatML is both the answer and the
    fallback. Swap in a family with a different header and this is the one
    function you extend.
    """
    bm = base_model.lower()
    if "qwen" not in bm:
        logger.warning(
            "No response template registered for %s - falling back to ChatML. "
            "Check the assistant header in the tokenizer's chat template.",
            base_model,
        )
    return "<|im_start|>assistant\n"              # Qwen2 / Qwen3 ChatML


def make_trainer(model, dataset, output_dir: Path, epochs: int = 4,
                 lr: float = 2e-4) -> Any:
    from transformers import TrainingArguments, Trainer

    tok = get_tokenizer()
    s = get_settings()
    # Pick the right response template for this model family, then build collator.
    response_template = _get_response_template(s.base_model)
    collator = _make_completion_collator(tok, response_template)

    import torch
    cuda = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=lr,
        # On CPU: batch size 1 to avoid OOM; on GPU: 4 with grad accum
        per_device_train_batch_size=4 if cuda else 1,
        gradient_accumulation_steps=4 if cuda else 1,
        num_train_epochs=epochs,
        warmup_ratio=0.03,
        bf16=cuda,              # bf16 requires GPU
        fp16=False,             # never use fp16 (unstable on most setups)
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        no_cuda=not cuda,       # explicit: don't try to use CUDA when it's absent
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
    )


# ── 4. Push to Hub ──────────────────────────────────────────────────────────
def push_adapter(model, repo: str) -> None:
    model.save_pretrained("out/adapter")
    model.push_to_hub(repo)


# ── 5. Tiny dispatcher ──────────────────────────────────────────────────────
TOOLS: dict[str, Callable[..., Any]] = {
    "load_base": load_base_model,
    "wrap_lora": wrap_with_lora,
    "load_dataset": load_and_tokenize_dataset,
    "make_trainer": make_trainer,
    "push_adapter": push_adapter,
}


def execute_tool(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a named training step - never raises, unknown tool returns error dict."""
    if name not in TOOLS:
        return {"success": False, "error": "unknown_tool", "tool": name}
    try:
        return TOOLS[name](**args)
    except Exception as exc:
        return {"success": False, "error": str(exc), "tool": name}
