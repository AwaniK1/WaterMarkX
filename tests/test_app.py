"""
Smoke tests for the Streamlit app using Streamlit's AppTest.
The model is replaced with fakes, so nothing is downloaded or loaded.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import attacks
import database
import detection
import generation

FAKE_RESULT = {"num_tokens_scored": 150, "num_green_tokens": 100, "green_fraction": 0.66,
               "z_score": 4.2, "p_value": 1e-5, "is_watermarked": True,
               "z_threshold": 3.0, "warning": None, "error": None}


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(detection, "load_engine", lambda: "fake-engine")
    monkeypatch.setattr(detection, "detect_text", lambda text, engine: dict(FAKE_RESULT))
    monkeypatch.setattr(generation, "generate_text",
                        lambda prompt, engine, watermark=True, **kw: f"generated wm={watermark}")
    app_path = str(Path(__file__).parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=60)
    at.run()
    return at


def go_to(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", ["Generate", "Verify", "Robustness", "Analytics", "History"])
def test_every_page_renders(app, page):
    go_to(app, page)
    assert app.header[0].value


def test_generate_saves_to_history(app):
    go_to(app, "Generate")
    app.radio(key="gen_mode").set_value("Both").run()
    app.button[0].click().run()
    assert not app.exception
    texts = [t.value for t in app.text_area]
    assert "generated wm=False" in texts and "generated wm=True" in texts
    rows = database.list_rows(database.connect(), "detections")
    assert {r["source"] for r in rows} == {"generate-normal", "generate-watermarked"}


def test_verify_shows_detection(app):
    go_to(app, "Verify")
    app.text_area[0].set_value("some text to check").run()
    app.button[0].click().run()
    assert any("Watermark detected" in s.value for s in app.success)
    assert database.list_rows(database.connect(), "detections")[0]["source"] == "verify"


def test_robustness_runs_attack_and_saves(app, monkeypatch):
    def fake_attack(text, attack_type, engine, **params):
        return attacks.AttackOutput(text="rewritten text", params=params)
    monkeypatch.setattr(attacks, "apply_attack", fake_attack)
    monkeypatch.setattr(attacks, "detect_text", lambda text, engine: dict(FAKE_RESULT))

    go_to(app, "Robustness")
    app.text_area[0].set_value("watermarked text").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("survived" in s.value for s in app.info)
    assert database.list_rows(database.connect(), "attacks")[0]["attacked_text"] == "rewritten text"
