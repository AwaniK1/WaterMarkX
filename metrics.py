"""
metrics.py - summarise attack results per attack type.

Reads the rows written by attacks.append_records_to_csv() and computes,
for each attack:
    - n_samples, n_failed (attack or detection failed)
    - n_comparable: samples whose ORIGINAL was detected (only these can
      show a detected -> undetected change)
    - removed / survived counts and the detection rate after the attack
    - mean z-score and green fraction before and after, and the mean z drop

Only the standard library is used, so this is easy to test.
"""

from __future__ import annotations

import csv
from statistics import mean

OUTCOME_REMOVED = "watermark_removed"
OUTCOME_SURVIVED = "watermark_survived"
FAILED_OUTCOMES = {"attack_failed", "detection_failed"}

# Fixed order so charts and tables always list attacks the same way.
ATTACK_ORDER = ["paraphrase", "synonym", "translation", "combined"]


def _float(value: str) -> float | None:
    """CSV stores numbers as text and missing values as ''."""
    return float(value) if value not in ("", None, "None") else None


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def summarize(rows: list[dict]) -> list[dict]:
    """One summary dict per attack type, in ATTACK_ORDER (unknown types last)."""
    by_attack: dict[str, list[dict]] = {}
    for row in rows:
        by_attack.setdefault(row["attack_type"], []).append(row)

    order = [a for a in ATTACK_ORDER if a in by_attack] + \
            sorted(a for a in by_attack if a not in ATTACK_ORDER)

    summary = []
    for attack in order:
        group = by_attack[attack]
        ok = [r for r in group if r["outcome"] not in FAILED_OUTCOMES]
        # Only samples where the original was detected can be "removed".
        comparable = [r for r in ok if r["outcome"] in (OUTCOME_REMOVED, OUTCOME_SURVIVED)]
        removed = sum(r["outcome"] == OUTCOME_REMOVED for r in comparable)
        survived = len(comparable) - removed

        z_before = _mean([_float(r["original_z_score"]) for r in comparable])
        z_after = _mean([_float(r["attacked_z_score"]) for r in comparable])
        summary.append({
            "attack_type": attack,
            "n_samples": len(group),
            "n_failed": len(group) - len(ok),
            "n_not_comparable": len(ok) - len(comparable),
            "n_comparable": len(comparable),
            "n_removed": removed,
            "n_survived": survived,
            # Share of comparable samples still detected after the attack.
            "detection_rate_after": survived / len(comparable) if comparable else None,
            "mean_z_before": z_before,
            "mean_z_after": z_after,
            "mean_z_drop": (z_before - z_after) if comparable else None,
            "mean_green_before": _mean([_float(r["original_green_fraction"]) for r in comparable]),
            "mean_green_after": _mean([_float(r["attacked_green_fraction"]) for r in comparable]),
            # Meaning kept, over every sample the attack ran on (not only comparable).
            "mean_similarity": _mean([_float(r.get("semantic_similarity", "")) for r in ok]),
            "n_similarity": sum(_float(r.get("semantic_similarity", "")) is not None for r in ok),
        })
    return summary


def baseline_summary(rows: list[dict]) -> list[dict]:
    """
    Detection WITHOUT any attack, per condition (rows from dataset.csv).

    For "watermarked" the detection rate is the TRUE POSITIVE rate;
    for "normal" it is the FALSE POSITIVE rate.
    """
    out = []
    for condition in ("normal", "watermarked"):
        group = [r for r in rows if r["condition"] == condition]
        scored = [r for r in group if not r["error"] and r["predicted_watermarked"] in ("True", "False")]
        zs = [_float(r["z_score"]) for r in scored]
        zs = [z for z in zs if z is not None]
        detected = sum(r["predicted_watermarked"] == "True" for r in scored)
        out.append({
            "condition": condition,
            "n": len(scored),
            "n_failed": len(group) - len(scored),
            "n_detected": detected,
            "detection_rate": detected / len(scored) if scored else None,
            "mean_z": mean(zs) if zs else None,
            "min_z": min(zs) if zs else None,
            "max_z": max(zs) if zs else None,
            "mean_green_fraction": _mean([_float(r["green_fraction"]) for r in scored]),
            "mean_tokens_scored": _mean([_float(r["num_tokens_scored"]) for r in scored]),
        })
    return out


def write_summary_csv(summary: list[dict], path: str) -> None:
    if not summary:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
