"""
charts.py - comparison charts for the attack results (Phase 4).

Usage (from the WaterMarkX folder, venv active):
    python charts.py
    python charts.py --csv experiments/attack_results.csv --out experiments/charts

Creates in the output folder:
    z_score_before_after.png      mean z-score before vs after each attack
    green_fraction_before_after.png
    detection_rate_after.png      share of samples still detected after each attack
    attack_summary.csv            the numbers behind the charts

Only samples whose ORIGINAL text was detected are used ("comparable"
samples). Samples that were not detected to begin with, or where the attack
failed, are counted in attack_summary.csv but not plotted.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # draw to files; no window needed
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from detection import Z_THRESHOLD
from metrics import load_rows, summarize, write_summary_csv

# ------------------------------------------------------------------
# Colours (validated palette: slot 1 blue, slot 2 orange)
# ------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BEFORE_COLOR = "#2a78d6"  # blue
AFTER_COLOR = "#eb6834"   # orange

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
})


def _style_axes(ax) -> None:
    """Recessive chrome: no box, hairline vertical gridlines only."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.96, title, fontsize=13, fontweight="bold",
             color=INK_PRIMARY, va="top")
    fig.text(0.02, 0.90, subtitle, fontsize=9.5, color=INK_SECONDARY, va="top")


def _row_labels(summary: list[dict]) -> list[str]:
    return [f"{s['attack_type'].capitalize()}  (n={s['n_comparable']})" for s in summary]


def _dumbbell(summary, before_key, after_key, ref_value, ref_label,
              title, subtitle, xlabel, path, value_fmt, start_at_zero=True) -> None:
    """One row per attack: a 'before' dot and an 'after' dot joined by a line."""
    rows = [s for s in summary if s[before_key] is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.6 * len(rows)))
    fig.subplots_adjust(left=0.25, right=0.95, top=0.68, bottom=0.18)
    _style_axes(ax)

    y = list(range(len(rows)))[::-1]  # first attack at the top
    for yi, s in zip(y, rows):
        ax.plot([s[before_key], s[after_key]], [yi, yi], color=BASELINE,
                linewidth=2, solid_capstyle="round", zorder=1)
    ax.scatter([s[before_key] for s in rows], y, s=90, color=BEFORE_COLOR,
               edgecolors=SURFACE, linewidths=2, zorder=3, label="Before attack")
    ax.scatter([s[after_key] for s in rows], y, s=90, color=AFTER_COLOR,
               edgecolors=SURFACE, linewidths=2, zorder=3, label="After attack")

    # Direct label on the "after" value only (the one the story is about),
    # placed above the dot so it never collides with the reference line.
    for yi, s in zip(y, rows):
        ax.annotate(value_fmt(s[after_key]), (s[after_key], yi),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)

    # Reference line (detection threshold / expected value).
    ax.axvline(ref_value, color=INK_SECONDARY, linewidth=1, zorder=2)
    ax.annotate(ref_label, (ref_value, 1), xycoords=("data", "axes fraction"),
                xytext=(4, 4), textcoords="offset points", fontsize=8.5,
                color=INK_SECONDARY, va="bottom")

    values = [s[before_key] for s in rows] + [s[after_key] for s in rows] + [ref_value]
    low = min(min(values), 0) if start_at_zero else min(values)
    span = (max(values) - low) or 1.0
    ax.set_xlim(low - (0 if start_at_zero else 0.1 * span), max(values) + 0.1 * span)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(y, _row_labels(rows))
    ax.set_xlabel(xlabel)

    _titles(fig, title, subtitle)
    fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.84), ncol=2, frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY, handletextpad=0.3, columnspacing=1.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_z_scores(summary, path, n_note) -> None:
    _dumbbell(summary, "mean_z_before", "mean_z_after", Z_THRESHOLD,
              f"Detection threshold (z = {Z_THRESHOLD:g})",
              "Watermark z-score before and after each attack",
              f"Mean z-score of samples detected before the attack. {n_note}",
              "z-score (higher = stronger watermark evidence)", path,
              lambda v: f"{v:.2f}")


def chart_green_fraction(summary, path, n_note) -> None:
    _dumbbell(summary, "mean_green_before", "mean_green_after", 0.5,
              "Expected for unwatermarked text (0.5)",
              "Green-token fraction before and after each attack",
              f"Share of tokens on the KGW green list. {n_note}",
              "Green-token fraction", path, lambda v: f"{v:.2f}",
              start_at_zero=False)  # all values sit near 0.5-0.8; zoom in


def chart_detection_rate(summary, path, n_note) -> None:
    """Horizontal bars: % of comparable samples still detected after the attack."""
    rows = [s for s in summary if s["n_comparable"] > 0]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.55 * len(rows)))
    fig.subplots_adjust(left=0.25, right=0.95, top=0.72, bottom=0.18)
    _style_axes(ax)

    y = list(range(len(rows)))[::-1]
    rates = [s["detection_rate_after"] for s in rows]
    ax.barh(y, rates, height=0.45, color=BEFORE_COLOR, zorder=2)
    for yi, s, r in zip(y, rows, rates):
        ax.annotate(f"{r:.0%}  ({s['n_survived']} of {s['n_comparable']} still detected)",
                    (r, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_PRIMARY)

    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_xlim(0, 1.45)  # room for labels on the right
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(y, _row_labels(rows))
    ax.set_xlabel("Still detected as watermarked after the attack (lower = stronger attack)")

    _titles(fig, "Detection rate after each attack", n_note)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chart_baseline_z(dataset_rows: list[dict], path: str) -> None:
    """Phase 1 chart: every text's z-score, normal vs watermarked, no attack.
    One dot per text, so the spread (not just the mean) is visible."""
    from metrics import baseline_summary

    groups = []
    for condition, color in (("watermarked", BEFORE_COLOR), ("normal", INK_MUTED)):
        zs = [float(r["z_score"]) for r in dataset_rows
              if r["condition"] == condition and not r["error"] and r["z_score"] not in ("", "None")]
        groups.append((condition, color, zs))
    if not any(zs for _, _, zs in groups):
        return
    stats = {s["condition"]: s for s in baseline_summary(dataset_rows)}

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.7, bottom=0.2)
    _style_axes(ax)

    labels = []
    for yi, (condition, color, zs) in enumerate(groups[::-1]):
        # Small fixed vertical offsets so overlapping dots stay visible.
        offsets = [((k % 5) - 2) * 0.06 for k in range(len(zs))]
        ax.scatter(zs, [yi + o for o in offsets], s=60, color=color,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
        s = stats[condition]
        labels.append(f"{condition.capitalize()} text  (n={s['n']})")
        if s["n"]:
            ax.annotate(f"{s['n_detected']} of {s['n']} detected ({s['detection_rate']:.0%})",
                        (1.0, yi), xycoords=("axes fraction", "data"), xytext=(0, 14),
                        textcoords="offset points", ha="right", fontsize=9, color=INK_PRIMARY)

    ax.axvline(Z_THRESHOLD, color=INK_SECONDARY, linewidth=1, zorder=2)
    ax.annotate(f"Detection threshold (z = {Z_THRESHOLD:g})", (Z_THRESHOLD, 1),
                xycoords=("data", "axes fraction"), xytext=(4, 4), textcoords="offset points",
                fontsize=8.5, color=INK_SECONDARY, va="bottom")
    all_z = [z for _, _, zs in groups for z in zs] + [Z_THRESHOLD, 0]
    span = max(all_z) - min(all_z)
    ax.set_xlim(min(all_z) - 0.08 * span, max(all_z) + 0.08 * span)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("z-score (each dot is one generated text)")

    _titles(fig, "Baseline detection: normal vs watermarked text",
            "No attack applied. Right of the line = detected as watermarked.")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_all_charts(csv_path: str, out_dir: str) -> list[dict]:
    rows = load_rows(csv_path)
    summary = summarize(rows)
    os.makedirs(out_dir, exist_ok=True)
    write_summary_csv(summary, os.path.join(out_dir, "attack_summary.csv"))

    n_min = min((s["n_comparable"] for s in summary if s["n_comparable"]), default=0)
    n_note = (f"Small sample (n={n_min} per attack) - shows the effect, not a reliable rate."
              if n_min < 10 else f"At least {n_min} samples per attack.")

    chart_z_scores(summary, os.path.join(out_dir, "z_score_before_after.png"), n_note)
    chart_green_fraction(summary, os.path.join(out_dir, "green_fraction_before_after.png"), n_note)
    chart_detection_rate(summary, os.path.join(out_dir, "detection_rate_after.png"), n_note)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw attack comparison charts.")
    parser.add_argument("--csv", default="experiments/attack_results.csv")
    parser.add_argument("--out", default="experiments/charts")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"No results file at {args.csv}. Run run_attack.py first.")

    summary = make_all_charts(args.csv, args.out)
    print(f"{'attack':<12}{'n':>3}{'comparable':>12}{'z before':>10}{'z after':>9}{'still detected':>16}")
    for s in summary:
        zb = "n/a" if s["mean_z_before"] is None else f"{s['mean_z_before']:.2f}"
        za = "n/a" if s["mean_z_after"] is None else f"{s['mean_z_after']:.2f}"
        rate = "n/a" if s["detection_rate_after"] is None else f"{s['detection_rate_after']:.0%}"
        print(f"{s['attack_type']:<12}{s['n_samples']:>3}{s['n_comparable']:>12}{zb:>10}{za:>9}{rate:>16}")
    print(f"\nCharts and attack_summary.csv saved to {args.out}")


if __name__ == "__main__":
    main()
