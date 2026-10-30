"""
Real-model integration test: loads Qwen, generates watermarked text,
paraphrases it and runs the real KGW detector on both.

Skipped by default (slow, needs the model). Run it with:
    $env:RUN_INTEGRATION = "1"; python -m pytest tests/test_integration_paraphrase.py -v -s

It checks that the pipeline WORKS, not that the attack "wins": the
outcome depends on the sample, so we only check it is a valid label.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1",
                                reason="set RUN_INTEGRATION=1 to run the real-model test")


def test_paraphrase_end_to_end():
    import attacks
    from detection import Z_THRESHOLD, load_engine
    from generation import generate_text

    engine = load_engine()
    text = generate_text("Explain how vaccines work in simple words.", engine,
                         watermark=True, seed=42)
    rec = attacks.evaluate_attack(text, "paraphrase", engine, seed=42)

    print("\nORIGINAL:", rec.original_text[:300])
    print("\nATTACKED:", (rec.attacked_text or "")[:300])
    print(f"\nz before={rec.original_z_score}  after={rec.attacked_z_score}  "
          f"outcome={rec.outcome}\nwarnings={rec.warnings}")

    assert rec.error is None
    assert isinstance(rec.attacked_text, str) and rec.attacked_text.strip()
    assert rec.attacked_text != rec.original_text
    assert rec.z_threshold == Z_THRESHOLD == 3.0
    assert rec.original_tokens_scored > 0 and rec.attacked_tokens_scored > 0
    assert rec.outcome in {attacks.OUTCOME_REMOVED, attacks.OUTCOME_SURVIVED,
                           attacks.OUTCOME_NOT_COMPARABLE}
