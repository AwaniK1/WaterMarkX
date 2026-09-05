from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from threading import Thread
import json

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    WatermarkingConfig,
    WatermarkDetector,
    TextIteratorStreamer
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
GREENLIST_RATIO = 0.5
BIAS = 2.0
SEEDING_SCHEME = "lefthash"
Z_THRESHOLD = 3.0

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading tokenizer + model... this can take a while on first run.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto"
    ).to(device)
    print("Using device:", device)
    watermark_config = WatermarkingConfig(
        greenlist_ratio=GREENLIST_RATIO,
        bias=BIAS,
        seeding_scheme=SEEDING_SCHEME
    )
    detector = WatermarkDetector(
        model_config=model.config,
        device=model.device,
        watermarking_config=watermark_config
    )
    state["tokenizer"] = tokenizer
    state["model"] = model
    state["watermark_config"] = watermark_config
    state["detector"] = detector
    print("Model ready on device:", model.device)
    yield
    state.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9


class DetectRequest(BaseModel):
    text: str


def _build_inputs(prompt: str):
    tokenizer = state["tokenizer"]
    model = state["model"]
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    return inputs


@app.post("/generate")
def generate(req: GenerateRequest):
    tokenizer = state["tokenizer"]
    model = state["model"]
    watermark_config = state["watermark_config"]

    inputs = _build_inputs(req.prompt)
    input_length = inputs["input_ids"].shape[-1]

    normal_output = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p
    )
    original_text = tokenizer.decode(
        normal_output[0][input_length:], skip_special_tokens=True
    )

    watermarked_output = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
        watermarking_config=watermark_config
    )
    watermarked_text = tokenizer.decode(
        watermarked_output[0][input_length:], skip_special_tokens=True
    )

    return {
        "baseline_text": original_text,
        "watermarked_text": watermarked_text
    }


@app.post("/generate-stream")
def generate_stream(req: GenerateRequest):
    tokenizer = state["tokenizer"]
    model = state["model"]
    watermark_config = state["watermark_config"]

    inputs = _build_inputs(req.prompt)

    def run_generation():
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            watermarking_config=watermark_config,
            streamer=streamer,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        tokens_so_far = 0
        full_text = ""

        for new_text in streamer:
            full_text += new_text
            tokens_so_far += 1
            percent = min(100, int((tokens_so_far / req.max_new_tokens) * 100))

            message = {
                "type": "progress",
                "percent": percent,
                "text_so_far": full_text
            }
            yield json.dumps(message) + "\n"

        thread.join()

        final_message = {
            "type": "done",
            "percent": 100,
            "text_so_far": full_text
        }
        yield json.dumps(final_message) + "\n"

    return StreamingResponse(run_generation(), media_type="application/x-ndjson")


def _run_detection(text: str):
    tokenizer = state["tokenizer"]
    model = state["model"]
    detector = state["detector"]

    tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)
    result = detector(tokens, z_threshold=Z_THRESHOLD, return_dict=True)

    return {
        "num_tokens_scored": int(result.num_tokens_scored[0]),
        "num_green_tokens": int(result.num_green_tokens[0]),
        "green_fraction": float(result.green_fraction[0]),
        "z_score": float(result.z_score[0]),
        "p_value": float(result.p_value[0]),
        "is_watermarked": bool(result.prediction[0]),
        "z_threshold": Z_THRESHOLD
    }


@app.post("/detect")
def detect(req: DetectRequest):
    return _run_detection(req.text)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}