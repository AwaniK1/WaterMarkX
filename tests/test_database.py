"""Tests for database.py (temporary database file, no model)."""

import csv

import database
from attacks import AttackRecord

DETECTION = {"num_tokens_scored": 150, "num_green_tokens": 100, "green_fraction": 0.66,
             "z_score": 4.1, "p_value": 0.0001, "is_watermarked": True,
             "z_threshold": 3.0, "warning": None, "error": None}


def make_record(outcome="watermark_removed"):
    values = {f: None for f in AttackRecord.__dataclass_fields__}
    values.update(timestamp="t", attack_type="synonym", attack_params="{}",
                  original_text="a", attacked_text="b", outcome=outcome, warnings="",
                  original_tokens_scored=10, attacked_tokens_scored=10)
    return AttackRecord(**values)


def test_tables_created_and_rows_saved(tmp_path):
    conn = database.connect(str(tmp_path / "h.db"))
    database.save_detection(conn, "some text", DETECTION, source="verify")
    database.save_attack(conn, make_record())
    dets = database.list_rows(conn, "detections")
    atts = database.list_rows(conn, "attacks")
    assert len(dets) == 1 and dets[0]["z_score"] == 4.1 and dets[0]["is_watermarked"] == 1
    assert dets[0]["z_threshold"] == 3.0 and '"bias": 2.0' in dets[0]["kgw_settings"]
    assert len(atts) == 1 and atts[0]["outcome"] == "watermark_removed"


def test_history_survives_reconnect(tmp_path):
    path = str(tmp_path / "h.db")
    conn = database.connect(path)
    database.save_detection(conn, "x", DETECTION, source="verify")
    conn.close()
    assert len(database.list_rows(database.connect(path), "detections")) == 1


def test_most_recent_first_and_limit(tmp_path):
    conn = database.connect(str(tmp_path / "h.db"))
    for outcome in ("a", "b", "c"):
        database.save_attack(conn, make_record(outcome))
    rows = database.list_rows(conn, "attacks", limit=2)
    assert [r["outcome"] for r in rows] == ["c", "b"]


def test_export_csv(tmp_path):
    conn = database.connect(str(tmp_path / "h.db"))
    database.save_attack(conn, make_record("first"))
    database.save_attack(conn, make_record("second"))
    out = tmp_path / "attacks.csv"
    assert database.export_csv(conn, "attacks", str(out)) == 2
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["outcome"] for r in rows] == ["first", "second"]  # oldest first
