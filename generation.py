"""
generation.py - generate normal or KGW-watermarked text with the shared engine.

Same chat template and sampling settings as main.py / kgw_detect.py.
"""

from __future__ import annotations

import torch

from detection import Engine

DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


def generate_text(
    prompt: str,
    engine: Engine,
    watermark: bool = True,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int | None = None,
) -> str:
    """Generate a reply to `prompt`. watermark=True applies the KGW config."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt is empty")
    if seed is not None:
        torch.manual_seed(seed)  # makes sampling repeatable

    messages = [{"role": "user", "content": prompt}]
    inputs = engine.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(engine.model.device)
    input_length = inputs["input_ids"].shape[-1]

    extra = {"watermarking_config": engine.watermark_config} if watermark else {}
    output = engine.model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p, **extra,
    )
    # Keep only the newly generated tokens (drop the prompt).
    return engine.tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
