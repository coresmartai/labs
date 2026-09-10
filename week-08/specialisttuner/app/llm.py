"""Thin wrapper around the model load + generate path.

Route handlers should never touch transformers / peft directly. Everything
goes through here so the model identifier + tokenizer setup live in one place.

Heavy imports (torch, transformers, peft) are lazy - deferred to function call
time so the module can be imported on machines without the GPU stack. Tests that
mock `generate` work without these packages present.
"""
from __future__ import annotations
import logging
from functools import lru_cache
from typing import Any
import time

from app.config import get_settings
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    s = get_settings()
    tok = AutoTokenizer.from_pretrained(s.base_model, token=s.hf_token or None)
    if tok.pad_token is None:
        # IMPORTANT: do NOT use eos as pad - see V2 failure mode #2.
        tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


@lru_cache(maxsize=1)
def get_model(adapter_path: str | None = None):
    """Load base model and optionally apply a LoRA adapter.

    GPU path : 4-bit QLoRA (BitsAndBytesConfig + device_map="auto") - fast.
    CPU path : full fp32, no quantization - slower but always works.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    s = get_settings()
    cuda = torch.cuda.is_available()

    if cuda:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            s.base_model,
            quantization_config=bnb_cfg,
            device_map="auto",
            token=s.hf_token or None,
        )
    else:
        # CPU path: full fp32 (Qwen 0.6B is ~2.4 GB in fp32 - fits comfortably in RAM)
        model = AutoModelForCausalLM.from_pretrained(
            s.base_model,
            torch_dtype=torch.float32,
            token=s.hf_token or None,
        )

    tok = get_tokenizer()
    if len(tok) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def generate(messages: list[ChatMessage], adapter_path: str | None = None,
             max_new_tokens: int = 256, temperature: float = 0.2) -> dict[str, Any]:
    """Inference path used by the FastAPI route. Returns text + timing."""
    import torch

    tok = get_tokenizer()
    model = get_model(adapter_path)

    prompt = tok.apply_chat_template(
        [m.model_dump() for m in messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tok.pad_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return {"output": text, "latency_ms": latency_ms}
