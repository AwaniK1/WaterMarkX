import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    WatermarkingConfig,
    WatermarkDetector
)


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# KGW parameters
GREENLIST_RATIO = 0.5
BIAS = 2.0
SEEDING_SCHEME = "lefthash"

# Detection threshold
Z_THRESHOLD = 3.0


# ============================================================
# 2. LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# 3. LOAD QWEN MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING QWEN MODEL")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded successfully.")
print("Device:", model.device)


# ============================================================
# 4. USER INPUT
# ============================================================

prompt = input("\nEnter your prompt: ")

messages = [
    {
        "role": "user",
        "content": prompt
    }
]


# ============================================================
# 5. TOKENIZE INPUT USING QWEN CHAT TEMPLATE
# ============================================================

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# We need to know where the prompt ends.
# Everything after this position is generated text.

input_length = inputs["input_ids"].shape[-1]


# ============================================================
# 6. GENERATE NORMAL / ORIGINAL TEXT
# ============================================================

print("\n" + "=" * 70)
print("GENERATING ORIGINAL TEXT")
print("=" * 70)

normal_output = model.generate(
    **inputs,

    # Number of new tokens generated
    max_new_tokens=200,

    # Sampling enabled
    do_sample=True,

    # Controls randomness
    temperature=0.7,

    # Prevents repeated phrases
    top_p=0.9
)


# Remove the original prompt tokens.
# We only want the newly generated portion.

original_text = tokenizer.decode(
    normal_output[0][input_length:],
    skip_special_tokens=True
)


# ============================================================
# 7. CREATE KGW WATERMARK CONFIGURATION
# ============================================================

watermark_config = WatermarkingConfig(
    greenlist_ratio=GREENLIST_RATIO,
    bias=BIAS,
    seeding_scheme=SEEDING_SCHEME
)


# ============================================================
# 8. GENERATE WATERMARKED TEXT
# ============================================================

print("\n" + "=" * 70)
print("GENERATING WATERMARKED TEXT")
print("=" * 70)

watermarked_output = model.generate(
    **inputs,

    max_new_tokens=200,

    do_sample=True,

    temperature=0.7,

    top_p=0.9,

    # THIS activates KGW watermarking
    watermarking_config=watermark_config
)


# Again, remove the prompt portion.

watermarked_text = tokenizer.decode(
    watermarked_output[0][input_length:],
    skip_special_tokens=True
)


# ============================================================
# 9. DISPLAY GENERATED TEXT
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL TEXT")
print("=" * 70)

print(original_text)


print("\n" + "=" * 70)
print("WATERMARKED TEXT")
print("=" * 70)

print(watermarked_text)


# ============================================================
# 10. CREATE KGW DETECTOR
# ============================================================

print("\n" + "=" * 70)
print("CREATING KGW DETECTOR")
print("=" * 70)

detector = WatermarkDetector(
    model_config=model.config,

    # Must be the same device used by watermark generation
    device=model.device,

    # IMPORTANT:
    # Detector needs the SAME KGW configuration.
    watermarking_config=watermark_config
)

print("Detector created successfully.")


# ============================================================
# 11. TOKENIZE ORIGINAL TEXT
# ============================================================

original_tokens = tokenizer(
    original_text,
    return_tensors="pt"
)["input_ids"].to(model.device)


# ============================================================
# 12. TOKENIZE WATERMARKED TEXT
# ============================================================

watermarked_tokens = tokenizer(
    watermarked_text,
    return_tensors="pt"
)["input_ids"].to(model.device)


print("\nOriginal token count:",
      original_tokens.shape[-1])

print("Watermarked token count:",
      watermarked_tokens.shape[-1])


# ============================================================
# 13. DETECT WATERMARK IN ORIGINAL TEXT
# ============================================================

original_result = detector(
    original_tokens,
    z_threshold=Z_THRESHOLD,
    return_dict=True
)


# ============================================================
# 14. DETECT WATERMARK IN WATERMARKED TEXT
# ============================================================

watermarked_result = detector(
    watermarked_tokens,
    z_threshold=Z_THRESHOLD,
    return_dict=True
)


# ============================================================
# 15. FUNCTION TO DISPLAY DETECTION RESULTS
# ============================================================

def display_result(name, result):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Tokens scored     :",
        result.num_tokens_scored
    )

    print(
        "Green tokens      :",
        result.num_green_tokens
    )

    print(
        "Green fraction    :",
        result.green_fraction
    )

    print(
        "Z-score           :",
        result.z_score
    )

    print(
        "P-value           :",
        result.p_value
    )

    print(
        "Confidence        :",
        result.confidence
    )

    print(
        "Watermark detected:",
        result.prediction
    )


# ============================================================
# 16. DISPLAY RESULTS
# ============================================================

display_result(
    "ORIGINAL TEXT - KGW DETECTION",
    original_result
)

display_result(
    "WATERMARKED TEXT - KGW DETECTION",
    watermarked_result
)


# ============================================================
# 17. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

print(
    "Detection threshold:",
    Z_THRESHOLD
)

print(
    "\nOriginal detected as watermark:",
    bool(original_result.prediction[0])
)

print(
    "Watermarked detected as watermark:",
    bool(watermarked_result.prediction[0])
)

