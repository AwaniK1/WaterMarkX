"""
build_dataset.py - Phases 1, 2 and 4: build the experiment dataset,
measure baseline detection, then run every attack on it.

Steps (run from the WaterMarkX folder, venv active):

    python build_dataset.py generate          # normal + watermarked text per prompt
    python build_dataset.py attack            # every attack on every watermarked text
    python build_dataset.py summary           # print results + draw charts

    python build_dataset.py generate --limit 2   # quick trial on 2 prompts

Output files (in experiments/):
    dataset.csv                 one row per generated text (normal / watermarked)
    dataset_attack_results.csv  one row per (prompt, attack)
    dataset_run_info.json       model + settings used, for reproducibility
    charts/dataset/             charts and summary tables

Both steps are RESUMABLE: rows already saved without an error are skipped,
so you can stop with Ctrl+C and run the same command again later.
Failed rows are saved too (with the error) and retried on the next run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime, timezone

DATASET_CSV = "experiments/dataset.csv"
ATTACK_CSV = "experiments/dataset_attack_results.csv"
RUN_INFO = "experiments/dataset_run_info.json"
CHART_DIR = "experiments/charts/dataset"

BASE_SEED = 1000  # prompt i uses seed BASE_SEED + i (same seed for both conditions)

# Open-ended prompts that produce ~150-200 tokens, across varied topics.
DEFAULT_PROMPTS = [
    "Explain how vaccines work in simple words.",
    "Describe the water cycle for a school student.",
    "What are the main causes of climate change?",
    "Explain how a computer stores information.",
    "Write a short story about a lost dog finding its way home.",
    "Describe the benefits and risks of social media for teenagers.",
    "Explain how photosynthesis works.",
    "What should a beginner know before learning to cook?",
    "Describe the history of the printing press and why it mattered.",
    "Explain what inflation is and how it affects ordinary people.",
    "Give advice to a student preparing for their final exams.",
    "Describe a typical day in the life of a farmer.",
    "Explain how the internet sends a message from one computer to another.",
    "What are the advantages of renewable energy?",
    "Describe how the human heart pumps blood.",
    "Write a short essay about the importance of friendship.",
    "Explain what machine learning is to someone with no technical background.",
    "Describe the main features of a healthy diet.",
    "Explain why the sky appears blue.",
    "Describe how a city could reduce traffic congestion.",
]

CONDITIONS = ("normal", "watermarked")
ATTACKS_TO_RUN = ("paraphrase", "synonym", "translation", "combined")


# ------------------------------------------------------------------
# Small CSV helpers
# ------------------------------------------------------------------
def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(row: dict, path: str) -> None:
    """Append one row immediately, so progress survives a crash or Ctrl+C."""
    from attacks import append_rows_csv
    append_rows_csv([row], path)


def latest_rows(rows: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    """If a sample was retried, keep only its most recent row."""
    latest: dict[tuple, dict] = {}
    for r in rows:
        latest[tuple(r[k] for k in key_fields)] = r
    return list(latest.values())


def done_keys(rows: list[dict], key_fields: tuple[str, ...]) -> set[tuple]:
    """Keys of rows that finished without an error (these are skipped)."""
    return {tuple(r[k] for k in key_fields)
            for r in latest_rows(rows, key_fields) if not r.get("error")}


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


# ------------------------------------------------------------------
# Step 1: generate the dataset (Phases 1 + 2)
# ------------------------------------------------------------------
def save_run_info(engine) -> None:
    import torch
    import transformers
    from detection import BIAS, GREENLIST_RATIO, MODEL_NAME, SEEDING_SCHEME, Z_THRESHOLD
    from generation import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P

    gc = engine.model.generation_config
    info = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL_NAME, "device": str(engine.model.device),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
        "kgw": {"greenlist_ratio": GREENLIST_RATIO, "bias": BIAS,
                "seeding_scheme": SEEDING_SCHEME, "z_threshold": Z_THRESHOLD},
        "generation": {"max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                       "temperature": DEFAULT_TEMPERATURE, "top_p": DEFAULT_TOP_P,
                       "base_seed": BASE_SEED},
        # Qwen's own defaults also apply during sampling. Recorded because they
        # limit token choice and can weaken the watermark (Phase 1 question).
        "model_default_generation_config": {
            k: getattr(gc, k, None)
            for k in ("top_k", "repetition_penalty", "temperature", "top_p", "do_sample")
        },
    }
    os.makedirs(os.path.dirname(RUN_INFO), exist_ok=True)
    with open(RUN_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def generate_dataset(prompts: list[str], limit: int | None) -> None:
    from detection import (BIAS, GREENLIST_RATIO, SEEDING_SCHEME, Z_THRESHOLD,
                           detect_text, load_engine)
    from generation import (DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE,
                            DEFAULT_TOP_P, generate_text)

    prompts = prompts[:limit] if limit else prompts
    done = done_keys(read_csv(DATASET_CSV), ("prompt_id", "condition"))
    todo = [(i, c) for i in range(len(prompts)) for c in CONDITIONS
            if (str(i), c) not in done]
    print(f"{len(prompts)} prompts x 2 conditions; {len(todo)} texts to generate "
          f"({len(prompts) * 2 - len(todo)} already done).")
    if not todo:
        return

    engine = load_engine()
    save_run_info(engine)

    for n, (i, condition) in enumerate(todo, 1):
        prompt, seed = prompts[i], BASE_SEED + i
        print(f"[{n}/{len(todo)}] prompt {i} ({condition}) ...", end=" ", flush=True)
        start = time.time()
        text, error, det = "", None, None
        try:
            # Same seed for both conditions: the only difference is the watermark.
            text = generate_text(prompt, engine, watermark=(condition == "watermarked"), seed=seed)
            det = detect_text(text, engine)
            error = det["error"]
        except Exception as exc:  # keep going; record the failure
            error = f"{type(exc).__name__}: {exc}"
        seconds = round(time.time() - start, 1)

        pred = det["is_watermarked"] if det else None
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prompt_id": i, "prompt": prompt, "condition": condition, "seed": seed,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS, "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P, "greenlist_ratio": GREENLIST_RATIO, "bias": BIAS,
            "seeding_scheme": SEEDING_SCHEME, "z_threshold": Z_THRESHOLD,
            "text": text,
            "num_tokens_scored": det["num_tokens_scored"] if det else 0,
            "num_green_tokens": det["num_green_tokens"] if det else None,
            "green_fraction": det["green_fraction"] if det else None,
            "z_score": det["z_score"] if det else None,
            "p_value": det["p_value"] if det else None,
            "predicted_watermarked": pred,
            # Correct = watermarked text detected, or normal text NOT detected.
            "correct": None if pred is None else (pred == (condition == "watermarked")),
            "generation_seconds": seconds,
            "warning": det["warning"] if det else None,
            "error": error,
        }
        append_row(row, DATASET_CSV)
        z = "n/a" if row["z_score"] is None else f"{row['z_score']:.2f}"
        print(f"z={z} detected={pred} ({seconds}s)" + (f" ERROR: {error}" if error else ""))


# ------------------------------------------------------------------
# Step 2: attack every watermarked text (Phase 4)
# ------------------------------------------------------------------
def attack_dataset(attacks: tuple[str, ...], limit: int | None) -> None:
    from attacks import evaluate_attack
    from detection import load_engine

    samples = [r for r in latest_rows(read_csv(DATASET_CSV), ("prompt_id", "condition"))
               if r["condition"] == "watermarked" and not r["error"]]
    samples.sort(key=lambda r: int(r["prompt_id"]))
    if limit:
        samples = samples[:limit]
    if not samples:
        raise SystemExit(f"No watermarked samples in {DATASET_CSV}. Run 'generate' first.")

    done = done_keys(read_csv(ATTACK_CSV), ("prompt_id", "attack_type"))
    todo = [(s, a) for s in samples for a in attacks if (s["prompt_id"], a) not in done]
    print(f"{len(samples)} watermarked samples x {len(attacks)} attacks; "
          f"{len(todo)} to run ({len(samples) * len(attacks) - len(todo)} already done).")
    if not todo:
        return

    engine = load_engine()
    for n, (sample, attack) in enumerate(todo, 1):
        seed = int(sample["seed"])
        params = {"pivot": "de"} if attack == "translation" else {"seed": seed}
        if attack in ("synonym", "combined"):
            params["proportion"] = 0.3
        print(f"[{n}/{len(todo)}] prompt {sample['prompt_id']} / {attack} ...", end=" ", flush=True)
        start = time.time()
        rec = evaluate_attack(sample["text"], attack, engine, **params)
        row = {"prompt_id": sample["prompt_id"], "prompt": sample["prompt"],
               "attack_seconds": round(time.time() - start, 1), **rec.to_row()}
        append_row(row, ATTACK_CSV)
        za = "n/a" if rec.attacked_z_score is None else f"{rec.attacked_z_score:.2f}"
        print(f"z {rec.original_z_score:.2f} -> {za}  {rec.outcome}"
              + (f" ERROR: {rec.error}" if rec.error else ""))


# ------------------------------------------------------------------
# Step 3: summary + charts
# ------------------------------------------------------------------
def summary() -> None:
    from charts import chart_baseline_z, make_all_charts
    from metrics import baseline_summary

    rows = latest_rows(read_csv(DATASET_CSV), ("prompt_id", "condition"))
    if not rows:
        raise SystemExit(f"No data in {DATASET_CSV}. Run 'generate' first.")
    os.makedirs(CHART_DIR, exist_ok=True)

    print("\nBASELINE DETECTION (no attack)")
    print(f"{'condition':<13}{'n':>4}{'detected':>10}{'rate':>8}{'mean z':>9}{'min z':>8}{'max z':>8}{'failed':>8}")
    for s in baseline_summary(rows):
        rate = "n/a" if s["detection_rate"] is None else f"{s['detection_rate']:.0%}"
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        print(f"{s['condition']:<13}{s['n']:>4}{s['n_detected']:>10}{rate:>8}"
              f"{fmt(s['mean_z']):>9}{fmt(s['min_z']):>8}{fmt(s['max_z']):>8}{s['n_failed']:>8}")
    print("  watermarked rate = true positive rate (want high)")
    print("  normal rate      = false positive rate (want ~0)")
    chart_baseline_z(rows, os.path.join(CHART_DIR, "baseline_z_scores.png"))

    attack_rows = latest_rows(read_csv(ATTACK_CSV), ("prompt_id", "attack_type"))
    if attack_rows:
        tmp = os.path.join(CHART_DIR, "_latest_attack_rows.csv")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(attack_rows[0].keys()))
            w.writeheader()
            w.writerows(attack_rows)
        print("\nATTACKS (only samples detected before the attack are comparable)")
        print(f"{'attack':<12}{'n':>4}{'comparable':>12}{'z before':>10}{'z after':>9}"
              f"{'still detected':>16}{'meaning':>9}{'failed':>8}")
        for s in make_all_charts(tmp, CHART_DIR):
            fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
            rate = "n/a" if s["detection_rate_after"] is None else f"{s['detection_rate_after']:.0%}"
            print(f"{s['attack_type']:<12}{s['n_samples']:>4}{s['n_comparable']:>12}"
                  f"{fmt(s['mean_z_before']):>10}{fmt(s['mean_z_after']):>9}{rate:>16}"
                  f"{fmt(s['mean_similarity']):>9}{s['n_failed']:>8}")
        print("  meaning = mean similarity to the original (1.0 = same meaning)")
        os.remove(tmp)
    else:
        print("\nNo attack results yet. Run 'attack' next.")
    print(f"\nCharts saved to {CHART_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the WatermarkX experiment dataset.")
    parser.add_argument("step", choices=["generate", "attack", "summary"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts/samples")
    parser.add_argument("--prompts-file", default=None, help="one prompt per line (default: built-in 20)")
    parser.add_argument("--attacks", default=",".join(ATTACKS_TO_RUN),
                        help="comma-separated, e.g. paraphrase,synonym")
    args = parser.parse_args()

    if args.step == "generate":
        generate_dataset(load_prompts(args.prompts_file), args.limit)
    elif args.step == "attack":
        attack_dataset(tuple(a.strip() for a in args.attacks.split(",") if a.strip()), args.limit)
    else:
        summary()


if __name__ == "__main__":
    main()
