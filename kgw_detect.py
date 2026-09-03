import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    WatermarkingConfig,
    WatermarkDetector
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# -----------------------------
# LOAD MODEL
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# -----------------------------
# INPUT
# -----------------------------

messages = [
    {
        "role": "user",
        "content": "Explain artificial intelligence in simple words."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

input_length = inputs["input_ids"].shape[-1]

# -----------------------------
# ORIGINAL GENERATION
# -----------------------------

normal_output = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7
)

original_text = tokenizer.decode(
    normal_output[0][input_length:],
    skip_special_tokens=True
)

# -----------------------------
# KGW CONFIGURATION
# -----------------------------

watermark_config = WatermarkingConfig(
    greenlist_ratio=0.5,
    bias=2.0,
    seeding_scheme="lefthash"
)

# -----------------------------
# WATERMARKED GENERATION
# -----------------------------

watermarked_output = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    watermarking_config=watermark_config
)

watermarked_text = tokenizer.decode(
    watermarked_output[0][input_length:],
    skip_special_tokens=True
)

# -----------------------------
# DISPLAY TEXT
# -----------------------------

print("\n" + "=" * 60)
print("ORIGINAL TEXT")
print("=" * 60)
print(original_text)

print("\n" + "=" * 60)
print("WATERMARKED TEXT")
print("=" * 60)
print(watermarked_text)

# -----------------------------
# CREATE DETECTOR
# -----------------------------

# -----------------------------
# CREATE KGW DETECTOR
# -----------------------------

detector = WatermarkDetector(
    model_config=model.config,
    device=model.device,
    watermarking_config=watermark_config
)

# -----------------------------
# TOKENIZE ORIGINAL TEXT
# -----------------------------

original_tokens = tokenizer(
    original_text,
    return_tensors="pt"
)["input_ids"].to(model.device)

# -----------------------------
# TOKENIZE WATERMARKED TEXT
# -----------------------------

watermarked_tokens = tokenizer(
    watermarked_text,
    return_tensors="pt"
)["input_ids"].to(model.device)

# -----------------------------
# RUN DETECTOR
# -----------------------------

original_result = detector(original_tokens, return_dict=True)
watermarked_result = detector(watermarked_tokens, return_dict=True)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("\n" + "=" * 60)
print("KGW DETECTION RESULTS")
print("=" * 60)

print("\nORIGINAL TEXT")
print("-" * 60)
print("Tokens scored     :", original_result.num_tokens_scored)
print("Green tokens      :", original_result.num_green_tokens)
print("Green fraction    :", original_result.green_fraction)
print("Z-score           :", original_result.z_score)
print("P-value           :", original_result.p_value)
print("Confidence        :", original_result.confidence)
print("Watermark detected:", original_result.prediction)

print("\nWATERMARKED TEXT")
print("-" * 60)
print("Tokens scored     :", watermarked_result.num_tokens_scored)
print("Green tokens      :", watermarked_result.num_green_tokens)
print("Green fraction    :", watermarked_result.green_fraction)
print("Z-score           :", watermarked_result.z_score)
print("P-value           :", watermarked_result.p_value)
print("Confidence        :", watermarked_result.confidence)
print("Watermark detected:", watermarked_result.prediction)