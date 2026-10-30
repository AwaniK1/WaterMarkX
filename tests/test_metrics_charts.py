"""Tests for metrics.py and charts.py (no model needed)."""

import csv
import os

from attacks import AttackRecord
from metrics import summarize


def row(attack, outcome, z_before=5.0, z_after=1.0, g_before=0.7, g_after=0.5):
    r = {f: "" for f in AttackRecord.__dataclass_fields__}
    r.update(attack_type=attack, outcome=outcome,
             original_z_score=str(z_before), attacked_z_score=str(z_after),
             original_green_fraction=str(g_before), attacked_green_fraction=str(g_after))
    return r


def test_summary_counts_and_rates():
    rows = [
        row("synonym", "watermark_survived", 5.0, 4.0),
        row("synonym", "watermark_removed", 6.0, 2.0),
        row("synonym", "not_comparable", 1.0, 0.5),   # excluded from means
        row("synonym", "attack_failed"),
        row("paraphrase", "watermark_removed", 5.0, 0.5),
    ]
    s = {x["attack_type"]: x for x in summarize(rows)}
    syn = s["synonym"]
    assert (syn["n_samples"], syn["n_failed"], syn["n_not_comparable"], syn["n_comparable"]) == (4, 1, 1, 2)
    assert syn["detection_rate_after"] == 0.5
    assert syn["mean_z_before"] == 5.5 and syn["mean_z_after"] == 3.0 and syn["mean_z_drop"] == 2.5
    assert [x["attack_type"] for x in summarize(rows)] == ["paraphrase", "synonym"]  # fixed order


def test_no_comparable_samples_gives_none_not_zero():
    s = summarize([row("translation", "not_comparable")])[0]
    assert s["n_comparable"] == 0
    assert s["detection_rate_after"] is None and s["mean_z_before"] is None


def test_charts_are_written(tmp_path):
    from charts import make_all_charts
    path = tmp_path / "results.csv"
    rows = [row("paraphrase", "watermark_removed", 5.1, 0.6),
            row("synonym", "watermark_survived", 5.1, 4.0)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "charts"
    make_all_charts(str(path), str(out))
    for name in ("z_score_before_after.png", "green_fraction_before_after.png",
                 "detection_rate_after.png", "attack_summary.csv"):
        assert os.path.getsize(out / name) > 0
