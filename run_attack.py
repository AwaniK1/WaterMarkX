"""
run_attack.py - try an attack on one watermarked text and show before/after.

Examples (from the WaterMarkX folder, venv active):
    python run_attack.py --prompt "Explain photosynthesis in simple words."
    python run_attack.py --text-file sample.txt --attack paraphrase --seed 1
    python run_attack.py --prompt "..." --attack synonym --proportion 0.3 --seed 1
    python run_attack.py --prompt "..." --attack translation --pivot de
    python run_attack.py --prompt "..." --attack combined --proportion 0.3 --seed 1

Results are appended to experiments/attack_results.csv.
"""

from __future__ import annotations

import argparse

from attacks import AVAILABLE_ATTACKS, append_records_to_csv, evaluate_attack
from detection import load_engine
from generation import generate_text

LINE = "=" * 70


def show(label: str, z, pred, gf, n) -> None:
    z_s = "n/a" if z is None else f"{z:.3f}"
    gf_s = "n/a" if gf is None else f"{gf:.3f}"
    print(f"{label:<10} z-score={z_s:>8}  green_fraction={gf_s:>6}  "
          f"tokens_scored={n:<4} detected={pred}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a text attack and re-detect the watermark.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="generate a watermarked text from this prompt first")
    src.add_argument("--text-file", help="file containing watermarked text to attack")
    parser.add_argument("--attack", default="paraphrase", choices=AVAILABLE_ATTACKS)
    parser.add_argument("--seed", type=int, default=None, help="random seed (for repeatability)")
    parser.add_argument("--proportion", type=float, default=0.3,
                        help="synonym/combined: share of eligible words to replace (0-1)")
    parser.add_argument("--pivot", default="de", help="translation: pivot language (de, fr, es)")
    parser.add_argument("--out", default="experiments/attack_results.csv")
    args = parser.parse_args()

    engine = load_engine()  # loaded ONCE, reused for generation, attack and detection

    if args.prompt:
        print(f"\n{LINE}\nGENERATING WATERMARKED TEXT\n{LINE}")
        text = generate_text(args.prompt, engine, watermark=True, seed=args.seed)
    else:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()

    print(f"\n{LINE}\nRUNNING ATTACK: {args.attack}\n{LINE}")
    # Each attack takes slightly different settings.
    if args.attack == "translation":
        params = {"pivot": args.pivot}  # beam search: deterministic, no seed
    elif args.attack in ("synonym", "combined"):
        params = {"proportion": args.proportion, "seed": args.seed}
    else:
        params = {"seed": args.seed}
    rec = evaluate_attack(text, args.attack, engine, **params)

    print(f"\n{LINE}\nORIGINAL (watermarked) TEXT\n{LINE}\n{rec.original_text}")
    print(f"\n{LINE}\nATTACKED TEXT\n{LINE}\n{rec.attacked_text}")
    print(f"\n{LINE}\nDETECTION (threshold z >= {rec.z_threshold})\n{LINE}")
    show("before", rec.original_z_score, rec.original_prediction,
         rec.original_green_fraction, rec.original_tokens_scored)
    show("after", rec.attacked_z_score, rec.attacked_prediction,
         rec.attacked_green_fraction, rec.attacked_tokens_scored)
    print(f"\nOutcome : {rec.outcome}")
    print(f"Params  : {rec.attack_params}")
    if rec.warnings:
        print("Warnings:\n  - " + rec.warnings.replace("; ", "\n  - "))
    if rec.error:
        print(f"ERROR   : {rec.error}")

    append_records_to_csv([rec], args.out)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
