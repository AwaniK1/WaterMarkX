"""Tests for build_dataset.py and the baseline metrics (no model needed)."""

import build_dataset as bd
import detection
import generation
from metrics import baseline_summary


def fake_detect(text, engine):
    z = 5.0 if text.startswith("WM") else 0.5
    return {"num_tokens_scored": 120, "num_green_tokens": 70, "green_fraction": 0.58,
            "z_score": z, "p_value": 0.01, "is_watermarked": z >= 3.0,
            "z_threshold": 3.0, "warning": None, "error": None}


def setup_fakes(monkeypatch, tmp_path, fail_prompt=None):
    calls = []

    def fake_generate(prompt, engine, watermark=True, seed=None, **kw):
        calls.append((prompt, watermark, seed))
        if prompt == fail_prompt:
            raise RuntimeError("CUDA out of memory")
        return ("WM " if watermark else "") + prompt

    monkeypatch.setattr(detection, "load_engine", lambda: object())
    monkeypatch.setattr(detection, "detect_text", fake_detect)
    monkeypatch.setattr(generation, "generate_text", fake_generate)
    monkeypatch.setattr(bd, "save_run_info", lambda engine: None)
    monkeypatch.setattr(bd, "DATASET_CSV", str(tmp_path / "dataset.csv"))
    return calls


def test_generate_writes_both_conditions_with_same_seed(monkeypatch, tmp_path):
    calls = setup_fakes(monkeypatch, tmp_path)
    bd.generate_dataset(["p0", "p1"], limit=None)
    rows = bd.read_csv(bd.DATASET_CSV)
    assert len(rows) == 4
    assert {(r["prompt_id"], r["condition"]) for r in rows} == {
        ("0", "normal"), ("0", "watermarked"), ("1", "normal"), ("1", "watermarked")}
    # Same seed for normal and watermarked versions of a prompt.
    seeds = {(p, wm): s for p, wm, s in calls}
    assert seeds[("p0", True)] == seeds[("p0", False)] == bd.BASE_SEED
    # 'correct' = watermarked detected / normal not detected.
    assert all(r["correct"] == "True" for r in rows)
    # Settings stored in every row.
    assert all(r["z_threshold"] == "3.0" and r["bias"] == "2.0" for r in rows)


def test_generate_is_resumable_and_retries_failures(monkeypatch, tmp_path):
    calls = setup_fakes(monkeypatch, tmp_path, fail_prompt="p1")
    bd.generate_dataset(["p0", "p1"], limit=None)
    rows = bd.read_csv(bd.DATASET_CSV)
    failed = [r for r in rows if r["error"]]
    assert len(failed) == 2 and "out of memory" in failed[0]["error"]  # failure recorded

    calls.clear()
    setup_fakes(monkeypatch, tmp_path)  # now nothing fails
    bd.generate_dataset(["p0", "p1"], limit=None)
    latest = bd.latest_rows(bd.read_csv(bd.DATASET_CSV), ("prompt_id", "condition"))
    assert len(latest) == 4 and not any(r["error"] for r in latest)


def test_resume_skips_finished_rows(monkeypatch, tmp_path):
    calls = setup_fakes(monkeypatch, tmp_path)
    bd.generate_dataset(["p0"], limit=None)
    calls.clear()
    bd.generate_dataset(["p0"], limit=None)
    assert calls == []  # nothing regenerated


def test_baseline_summary_rates():
    rows = [
        {"condition": "watermarked", "error": "", "predicted_watermarked": "True",
         "z_score": "5.0", "green_fraction": "0.7", "num_tokens_scored": "150"},
        {"condition": "watermarked", "error": "", "predicted_watermarked": "False",
         "z_score": "2.0", "green_fraction": "0.6", "num_tokens_scored": "150"},
        {"condition": "watermarked", "error": "boom", "predicted_watermarked": "",
         "z_score": "", "green_fraction": "", "num_tokens_scored": "0"},
        {"condition": "normal", "error": "", "predicted_watermarked": "False",
         "z_score": "0.1", "green_fraction": "0.5", "num_tokens_scored": "150"},
    ]
    s = {x["condition"]: x for x in baseline_summary(rows)}
    wm = s["watermarked"]
    assert (wm["n"], wm["n_failed"], wm["n_detected"], wm["detection_rate"]) == (2, 1, 1, 0.5)
    assert wm["mean_z"] == 3.5 and wm["min_z"] == 2.0
    assert s["normal"]["detection_rate"] == 0.0  # false positive rate
