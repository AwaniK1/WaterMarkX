"""
database.py - SQLite history of detections and attacks (Phase 6).

Everything the app does is saved here, so results survive a restart:
    detections : each text that was generated or verified, with its KGW result
    attacks    : each attack run, with before/after results (same fields as
                 attacks.AttackRecord)

SQLite is a single file (experiments/watermarkx.db) - no server needed.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from dataclasses import fields
from datetime import datetime, timezone

from attacks import AttackRecord
from detection import BIAS, GREENLIST_RATIO, SEEDING_SCHEME, Z_THRESHOLD

DB_PATH = "experiments/watermarkx.db"

DETECTION_COLUMNS = [
    "created_at", "source", "prompt", "text", "num_tokens_scored", "num_green_tokens",
    "green_fraction", "z_score", "p_value", "is_watermarked", "z_threshold",
    "kgw_settings", "warning", "error",
]
ATTACK_COLUMNS = [f.name for f in fields(AttackRecord)]

TABLES = {"detections": DETECTION_COLUMNS, "attacks": ATTACK_COLUMNS}


def connect(path: str | None = None) -> sqlite3.Connection:
    """Open the database (default DB_PATH), creating the file and tables if needed."""
    path = path or DB_PATH
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    for table, cols in TABLES.items():
        col_sql = ", ".join(f'"{c}"' for c in cols)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} "
                     f"(id INTEGER PRIMARY KEY AUTOINCREMENT, {col_sql})")
    conn.commit()
    return conn


def _insert(conn: sqlite3.Connection, table: str, row: dict) -> int:
    cols = TABLES[table]
    values = [row.get(c) for c in cols]
    placeholders = ", ".join("?" for _ in cols)
    col_sql = ", ".join(f'"{c}"' for c in cols)
    cur = conn.execute(f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})", values)
    conn.commit()
    return cur.lastrowid


def save_detection(conn: sqlite3.Connection, text: str, result: dict,
                   source: str, prompt: str | None = None) -> int:
    """Save one detection result (from detection.detect_text).
    source: e.g. 'generate-normal', 'generate-watermarked', 'verify'."""
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source, "prompt": prompt, "text": text,
        "kgw_settings": json.dumps({"greenlist_ratio": GREENLIST_RATIO, "bias": BIAS,
                                    "seeding_scheme": SEEDING_SCHEME}),
        **{k: result.get(k) for k in ("num_tokens_scored", "num_green_tokens", "green_fraction",
                                      "z_score", "p_value", "is_watermarked", "warning", "error")},
        "z_threshold": result.get("z_threshold", Z_THRESHOLD),
    }
    return _insert(conn, "detections", row)


def save_attack(conn: sqlite3.Connection, record: AttackRecord) -> int:
    return _insert(conn, "attacks", record.to_row())


def list_rows(conn: sqlite3.Connection, table: str, limit: int | None = None) -> list[dict]:
    """Most recent first."""
    if table not in TABLES:
        raise ValueError(f"unknown table {table}")
    sql = f"SELECT * FROM {table} ORDER BY id DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [dict(r) for r in conn.execute(sql).fetchall()]


def export_csv(conn: sqlite3.Connection, table: str, path: str) -> int:
    """Write a whole table to CSV (oldest first). Returns the row count."""
    rows = list_rows(conn, table)[::-1]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id"] + TABLES[table])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
