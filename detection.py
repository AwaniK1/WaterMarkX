"""
detection.py - shared KGW settings, model loading and watermark detection.

Why this file exists:
    kgw_detect.py runs everything at import time (it even calls input()),
    so other code cannot reuse it. main.py has a clean detection function
    but is tied to FastAPI. This module holds the same logic in a form that
    attacks.py, run_attack.py and the tests can import.

The KGW values below are copied exactly from main.py / kgw_detect.py.
Do NOT change them here without changing them everywhere - the detector
must use the same settings that were used when the text was generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WatermarkDetector,
    WatermarkingConfig,
)

# ------------------------------------------------------------------
# KGW configuration (same as main.py and kgw_detect.py)
# ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
GREENLIST_RATIO = 0.5
BIAS = 2.0
SEEDING_SCHEME = "lefthash"
Z_THRESHOLD = 3.0

# "lefthash" seeds each token's green list from the 1 token before it,
# so the first token can never be scored. We need at least 2 tokens.
MIN_TOKENS_FOR_DETECTION = 2

# Below this many scored tokens a z-score is statistically weak.
# We do not change the threshold - we only attach a warning.
SHORT_TEXT_WARNING_TOKENS = 50


@dataclass
class Engine:
    """Everything that is loaded once and shared: model, tokenizer, detector."""

    model: Any
    tokenizer: Any
    watermark_config: Any
    detector: Any


def build_watermark_config() -> WatermarkingConfig:
    """The one place the KGW config object is built."""
    return WatermarkingConfig(
        greenlist_ratio=GREENLIST_RATIO,
        bias=BIAS,
        seeding_scheme=SEEDING_SCHEME,
    )


def load_engine(model_name: str = MODEL_NAME) -> Engine:
    """Load Qwen + tokenizer + KGW detector once (same way main.py does)."""
    print(f"Loading {model_name} ... (first run downloads it)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
    model.eval()

    watermark_config = build_watermark_config()
    detector = WatermarkDetector(
        model_config=model.config,
        device=model.device,
        watermarking_config=watermark_config,
    )
    print("Model ready on device:", model.device)
    return Engine(model=model, tokenizer=tokenizer,
                  watermark_config=watermark_config, detector=detector)


def _empty_detection(error: str) -> dict:
    """A detection result where nothing could be scored (keeps the same keys)."""
    return {
        "num_tokens_scored": 0,
        "num_green_tokens": None,
        "green_fraction": None,
        "z_score": None,
        "p_value": None,
        "is_watermarked": None,  # None = "unknown", NOT "False"
        "z_threshold": Z_THRESHOLD,
        "warning": None,
        "error": error,
    }


def detect_text(text: str, engine: Engine) -> dict:
    """
    Run KGW detection on a piece of text.

    Same logic as main.py's _run_detection(), plus:
      - empty / too-short text returns an error instead of crashing,
      - short texts get a warning (threshold is NOT changed).
    """
    if text is None or not text.strip():
        return _empty_detection("empty text - nothing to score")

    tokens = engine.tokenizer(text, return_tensors="pt")["input_ids"].to(engine.model.device)
    if tokens.shape[-1] < MIN_TOKENS_FOR_DETECTION:
        return _empty_detection(
            f"only {tokens.shape[-1]} token(s); need at least {MIN_TOKENS_FOR_DETECTION}"
        )

    result = engine.detector(tokens, z_threshold=Z_THRESHOLD, return_dict=True)

    scored = int(result.num_tokens_scored[0])
    warning = None
    if scored < SHORT_TEXT_WARNING_TOKENS:
        warning = (f"only {scored} tokens scored (<{SHORT_TEXT_WARNING_TOKENS}); "
                   "z-score is statistically weak")

    return {
        "num_tokens_scored": scored,
        "num_green_tokens": int(result.num_green_tokens[0]),
        "green_fraction": float(result.green_fraction[0]),
        "z_score": float(result.z_score[0]),
        "p_value": float(result.p_value[0]),
        "is_watermarked": bool(result.prediction[0]),
        "z_threshold": Z_THRESHOLD,
        "warning": warning,
        "error": None,
    }
