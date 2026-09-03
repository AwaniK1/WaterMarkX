import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    WatermarkingConfig,
    WatermarkDetector
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Our prompt
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


# ------------------------------------------------
# 1. NORMAL GENERATION
# ------------------------------------------------

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


# ------------------------------------------------
# 2. KGW-STYLE WATERMARKED GENERATION
# ------------------------------------------------

watermark_config = WatermarkingConfig(
    greenlist_ratio=0.5,
    bias=2.0,
    seeding_scheme="lefthash"
)

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


# ------------------------------------------------
# DISPLAY BOTH
# ------------------------------------------------

print("\n" + "=" * 60)
print("ORIGINAL TEXT")
print("=" * 60)
print(original_text)

print("\n" + "=" * 60)
print("WATERMARKED TEXT")
print("=" * 60)
print(watermarked_text)