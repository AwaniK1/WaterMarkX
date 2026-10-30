"""
Unit tests for attacks.py and detection.py.

These use small FAKE model/tokenizer/detector objects, so they run in
seconds and never download or load Qwen.

Run:  python -m pytest tests -v
"""

import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import attacks
import detection
from attacks import (AttackError, AttackRecord, append_records_to_csv,
                     apply_attack, clean_paraphrase, evaluate_attack,
                     paraphrase_text)
from detection import Engine, detect_text

WATERMARKED = ("Artificial intelligence is a branch of computer science that builds "
               "systems able to learn from data, recognise patterns and make decisions.")
PARAPHRASED = ("AI is an area of computing focused on creating systems that can learn "
               "from information, spot patterns and take decisions.")


# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------
class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """Splits on spaces. decode() returns a canned reply."""

    def __init__(self, reply=PARAPHRASED):
        self.reply = reply
        self.last_messages = None

    def __call__(self, text, return_tensors=None):
        ids = list(range(1, len(text.split()) + 1))
        if return_tensors == "pt":
            ids = torch.tensor([ids])
        return {"input_ids": ids}

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        return FakeBatch(input_ids=torch.zeros((1, 7), dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return self.reply


class FakeModel:
    device = "cpu"

    def __init__(self, n_new_tokens=10, fail=False):
        self.generation_config = SimpleNamespace(watermarking_config=None)
        self.calls = []
        self.n_new_tokens = n_new_tokens
        self.fail = fail

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return torch.zeros((1, 7 + self.n_new_tokens), dtype=torch.long)


class FakeDetector:
    """Returns pre-set results in order and remembers what it was called with."""

    def __init__(self, z_scores):
        self.z_scores = list(z_scores)
        self.calls = []

    def __call__(self, tokens, z_threshold, return_dict):
        self.calls.append({"n_tokens": tokens.shape[-1], "z_threshold": z_threshold})
        z = self.z_scores.pop(0)
        n = tokens.shape[-1] - 1
        return SimpleNamespace(
            num_tokens_scored=np.array([n]), num_green_tokens=np.array([n // 2]),
            green_fraction=np.array([0.5]), z_score=np.array([z]),
            p_value=np.array([0.01]), prediction=np.array([z >= z_threshold]),
        )


def make_engine(z_scores=(5.0, 1.0), reply=PARAPHRASED, **model_kw):
    return Engine(model=FakeModel(**model_kw), tokenizer=FakeTokenizer(reply),
                  watermark_config=detection.build_watermark_config(),
                  detector=FakeDetector(z_scores))


# ------------------------------------------------------------------
# Empty input
# ------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", None])
def test_paraphrase_rejects_empty_input(bad):
    with pytest.raises(AttackError):
        paraphrase_text(bad, FakeModel(), FakeTokenizer())


def test_apply_attack_rejects_empty_and_unknown():
    with pytest.raises(AttackError):
        apply_attack("", "paraphrase", make_engine())
    with pytest.raises(AttackError, match="unknown attack"):
        apply_attack(WATERMARKED, "nonsense", make_engine())


def test_detect_text_handles_empty_without_crashing():
    engine = make_engine()
    result = detect_text("", engine)
    assert result["error"] and result["is_watermarked"] is None
    assert engine.detector.calls == []  # detector never called on empty text


# ------------------------------------------------------------------
# Paraphrase behaviour
# ------------------------------------------------------------------
def test_paraphrase_returns_string_and_params():
    out = paraphrase_text(WATERMARKED, FakeModel(), FakeTokenizer(), seed=1)
    assert isinstance(out.text, str) and out.text == PARAPHRASED
    assert out.params["seed"] == 1 and out.warnings == []


def test_paraphrase_does_not_watermark_its_output():
    model = FakeModel()
    paraphrase_text(WATERMARKED, model, FakeTokenizer())
    assert "watermarking_config" not in model.calls[0]


def test_paraphrase_refuses_if_model_watermarks_by_default():
    model = FakeModel()
    model.generation_config.watermarking_config = object()
    with pytest.raises(AttackError):
        paraphrase_text(WATERMARKED, model, FakeTokenizer())


def test_prompt_wraps_text_as_data():
    tok = FakeTokenizer()
    paraphrase_text(WATERMARKED, FakeModel(), tok)
    assert tok.last_messages[0]["role"] == "system"
    assert f"<text>\n{WATERMARKED}\n</text>" in tok.last_messages[1]["content"]


def test_empty_model_output_is_an_error():
    with pytest.raises(AttackError, match="empty"):
        paraphrase_text(WATERMARKED, FakeModel(), FakeTokenizer(reply="   "))


def test_generation_exception_becomes_attack_error():
    with pytest.raises(AttackError, match="generation failed"):
        paraphrase_text(WATERMARKED, FakeModel(fail=True), FakeTokenizer())


def test_identical_and_answered_outputs_are_flagged():
    same = paraphrase_text(WATERMARKED, FakeModel(), FakeTokenizer(reply=WATERMARKED))
    assert any("identical" in w for w in same.warnings)
    short = paraphrase_text(WATERMARKED, FakeModel(), FakeTokenizer(reply="Yes."))
    assert any("ratio" in w for w in short.warnings)


def test_truncation_is_flagged():
    out = paraphrase_text(WATERMARKED, FakeModel(n_new_tokens=5), FakeTokenizer(),
                          max_new_tokens=5)
    assert any("truncated" in w for w in out.warnings)


@pytest.mark.parametrize("raw", [
    "Here is the paraphrased text: AI is great.",
    "Sure! Here's a rewritten version:\nAI is great.",
    "<text>\nAI is great.\n</text>",
    '"AI is great."',
])
def test_clean_paraphrase_strips_wrappers(raw):
    assert clean_paraphrase(raw) == "AI is great."


# ------------------------------------------------------------------
# Detection integration + KGW config preserved
# ------------------------------------------------------------------
def test_kgw_config_unchanged():
    assert (detection.GREENLIST_RATIO, detection.BIAS, detection.SEEDING_SCHEME,
            detection.Z_THRESHOLD) == (0.5, 2.0, "lefthash", 3.0)
    cfg = detection.build_watermark_config()
    assert cfg.greenlist_ratio == 0.5 and cfg.bias == 2.0 and cfg.seeding_scheme == "lefthash"


def test_main_py_uses_same_constants():
    """Guards against main.py and detection.py drifting apart."""
    import ast, pathlib
    src = pathlib.Path(__file__).parent.parent.joinpath("main.py").read_text(encoding="utf-8")
    consts = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in {"GREENLIST_RATIO", "BIAS", "SEEDING_SCHEME", "Z_THRESHOLD"}}
    assert consts == {"GREENLIST_RATIO": 0.5, "BIAS": 2.0,
                      "SEEDING_SCHEME": "lefthash", "Z_THRESHOLD": 3.0}


def test_both_texts_go_through_detector_with_same_threshold():
    engine = make_engine(z_scores=(5.0, 1.0))
    rec = evaluate_attack(WATERMARKED, "paraphrase", engine)
    assert len(engine.detector.calls) == 2
    assert all(c["z_threshold"] == 3.0 for c in engine.detector.calls)
    assert rec.original_z_score == 5.0 and rec.attacked_z_score == 1.0
    assert rec.outcome == attacks.OUTCOME_REMOVED


def test_outcome_survived():
    rec = evaluate_attack(WATERMARKED, "paraphrase", make_engine(z_scores=(5.0, 4.0)))
    assert rec.outcome == attacks.OUTCOME_SURVIVED


def test_undetected_original_is_flagged_not_comparable():
    rec = evaluate_attack(WATERMARKED, "paraphrase", make_engine(z_scores=(1.2, 0.3)))
    assert rec.outcome == attacks.OUTCOME_NOT_COMPARABLE
    assert "NOT detected" in rec.warnings


def test_failed_attack_is_recorded_not_hidden():
    engine = make_engine(z_scores=(5.0,), fail=True)
    rec = evaluate_attack(WATERMARKED, "paraphrase", engine)
    assert rec.outcome == attacks.OUTCOME_ATTACK_FAILED
    assert rec.error and rec.attacked_text is None
    assert rec.attacked_z_score is None and rec.attacked_prediction is None
    assert len(engine.detector.calls) == 1  # only the original was scored


# ------------------------------------------------------------------
# CSV-ready structure
# ------------------------------------------------------------------
def test_records_export_to_csv_consistently(tmp_path):
    ok = evaluate_attack(WATERMARKED, "paraphrase", make_engine(z_scores=(5.0, 1.0)))
    bad = evaluate_attack(WATERMARKED, "paraphrase", make_engine(z_scores=(5.0,), fail=True))
    assert list(ok.to_row()) == list(bad.to_row())  # same columns, same order

    path = tmp_path / "results.csv"
    append_records_to_csv([ok], str(path))
    append_records_to_csv([bad], str(path))  # second call must not repeat the header
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert set(rows[0]) == set(AttackRecord.__dataclass_fields__)
    assert rows[1]["outcome"] == "attack_failed"


# ------------------------------------------------------------------
# Attack B: synonym replacement (uses the real, local WordNet data)
# ------------------------------------------------------------------
from attacks import combined_attack, replace_synonyms, translation_round_trip

LONG_TEXT = (
    "The quick student made a big decision about the difficult project. "
    "She wanted a simple plan, a clear goal and a happy team. "
    "Microsoft released 3 new tools in London last year, and the large "
    "company said the fast change was important for every small business."
)


@pytest.mark.parametrize("bad", ["", "  ", None])
def test_synonym_rejects_empty_input(bad):
    with pytest.raises(AttackError):
        replace_synonyms(bad)


def test_synonym_rejects_bad_proportion():
    with pytest.raises(AttackError):
        replace_synonyms(LONG_TEXT, proportion=1.5)


@pytest.mark.parametrize("proportion", [0.0, 0.2, 0.5, 1.0])
def test_synonym_respects_proportion(proportion):
    out = replace_synonyms(LONG_TEXT, proportion=proportion, seed=0)
    p = out.params
    assert isinstance(out.text, str)
    assert p["words_targeted"] == round(proportion * p["words_eligible"])
    assert p["words_replaced"] <= p["words_targeted"] <= p["words_eligible"]
    assert p["words_attempted"] >= p["words_replaced"]
    if p["words_replaced"] < p["words_targeted"]:
        assert any("usable synonym" in w for w in out.warnings)  # shortfall is reported
    else:
        assert p["words_replaced"] == p["words_targeted"]


def test_synonym_counts_match_changed_words():
    out = replace_synonyms(LONG_TEXT, proportion=0.5, seed=1)
    before, after = LONG_TEXT.split(), out.text.split()
    # Single-word synonyms only, so word count and positions are unchanged.
    assert len(before) == len(after)
    changed = sum(1 for a, b in zip(before, after) if a != b)
    assert changed == out.params["words_replaced"]


def test_synonym_keeps_names_numbers_stopwords_and_punctuation():
    out = replace_synonyms(LONG_TEXT, proportion=1.0, seed=2)
    for kept in ("Microsoft", "3", "London", " the ", " and ", "."):
        assert kept in out.text
    strip = lambda s: [c for c in s if not c.isalpha()]
    assert strip(out.text) == strip(LONG_TEXT)  # punctuation/spacing identical


def test_synonym_is_repeatable_with_seed():
    a = replace_synonyms(LONG_TEXT, proportion=0.4, seed=7)
    b = replace_synonyms(LONG_TEXT, proportion=0.4, seed=7)
    assert a.text == b.text


def test_synonym_no_eligible_words_warns():
    out = replace_synonyms("It is in the of and to.", proportion=0.5)
    assert out.text == "It is in the of and to."
    assert any("no words were replaced" in w for w in out.warnings)


def test_synonym_missing_wordnet_gives_setup_hint(monkeypatch):
    from nltk.corpus import wordnet
    def boom():
        raise LookupError("Resource wordnet not found")
    monkeypatch.setattr(wordnet, "ensure_loaded", boom)
    with pytest.raises(AttackError, match="nltk.downloader"):
        replace_synonyms(LONG_TEXT)


# ------------------------------------------------------------------
# Attack C: translation round-trip (translator is faked - no download)
# ------------------------------------------------------------------
def fake_translate(sentences, model_name, device, num_beams, batch_size=8):
    tag = "DE" if model_name.endswith("en-de") else "EN"
    return [f"[{tag}] {s.replace('[DE] ', '')}" for s in sentences]


def test_translation_round_trip_with_fake_models(monkeypatch):
    monkeypatch.setattr(attacks, "_translate", fake_translate)
    text = "First sentence. Second one!\n\nNew paragraph?"
    out = translation_round_trip(text, pivot="de", device="cpu")
    assert out.text == "[EN] First sentence. [EN] Second one!\n\n[EN] New paragraph?"
    assert out.params["forward_model"] == "Helsinki-NLP/opus-mt-en-de"
    assert out.params["backward_model"] == "Helsinki-NLP/opus-mt-de-en"
    assert out.params["pivot_text"].startswith("[DE] First sentence.")


def test_translation_rejects_empty_and_bad_pivot():
    with pytest.raises(AttackError):
        translation_round_trip("")
    with pytest.raises(AttackError, match="pivot"):
        translation_round_trip(WATERMARKED, pivot="xx")


def test_translation_failure_becomes_attack_error(monkeypatch):
    def broken(*a, **k):
        raise OSError("no internet")
    monkeypatch.setattr(attacks, "_translate", broken)
    with pytest.raises(AttackError, match="could not load model"):
        translation_round_trip(WATERMARKED, device="cpu")


def test_translation_failure_recorded_by_evaluate(monkeypatch):
    monkeypatch.setattr(attacks, "_translate", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    rec = evaluate_attack(WATERMARKED, "translation", make_engine(z_scores=(5.0,)), device="cpu")
    assert rec.outcome == attacks.OUTCOME_ATTACK_FAILED and rec.attacked_text is None


# ------------------------------------------------------------------
# Attack D: combined
# ------------------------------------------------------------------
def test_combined_records_order_and_both_steps():
    engine = make_engine(reply=LONG_TEXT)  # fake paraphrase returns LONG_TEXT
    out = combined_attack(WATERMARKED, engine, proportion=0.3, seed=3)
    assert out.params["order"] == ["paraphrase", "synonym"]
    assert out.params["synonym"]["words_replaced"] > 0
    assert out.text != LONG_TEXT  # synonyms applied on top of the paraphrase
    assert "watermarking_config" not in engine.model.calls[0]


def test_every_attack_returns_string_or_clear_error(monkeypatch):
    monkeypatch.setattr(attacks, "_translate", fake_translate)
    params = {"paraphrase": {}, "synonym": {"seed": 0}, "translation": {"device": "cpu"},
              "combined": {"seed": 0}}
    assert set(params) == set(attacks.AVAILABLE_ATTACKS)
    for name, p in params.items():
        out = apply_attack(LONG_TEXT, name, make_engine(reply=LONG_TEXT), **p)
        assert isinstance(out.text, str) and out.text.strip()
        with pytest.raises(AttackError):
            apply_attack("", name, make_engine(), **p)


# ------------------------------------------------------------------
# Semantic similarity + CSV schema changes
# ------------------------------------------------------------------
import similarity


def test_similarity_is_recorded():
    rec = evaluate_attack(WATERMARKED, "paraphrase", make_engine())
    assert rec.semantic_similarity == 0.9  # faked in conftest.py


def test_similarity_failure_is_a_warning_not_a_crash(monkeypatch):
    def broken(a, b):
        raise ImportError("sentence-transformers not installed")
    monkeypatch.setattr(similarity, "semantic_similarity", broken)
    rec = evaluate_attack(WATERMARKED, "paraphrase", make_engine())
    assert rec.semantic_similarity is None
    assert "similarity not computed" in rec.warnings
    assert rec.outcome == attacks.OUTCOME_REMOVED  # detection result unaffected


def test_similarity_skipped_when_attack_failed():
    rec = evaluate_attack(WATERMARKED, "paraphrase", make_engine(z_scores=(5.0,), fail=True))
    assert rec.semantic_similarity is None


def test_similarity_rejects_empty_text():
    import importlib
    real = importlib.reload(similarity)  # undo the fake for this check
    with pytest.raises(ValueError):
        real.semantic_similarity("", "text")


def test_csv_gains_new_columns_without_losing_rows(tmp_path):
    from attacks import append_rows_csv
    path = str(tmp_path / "r.csv")
    append_rows_csv([{"a": 1, "b": 2}], path)            # old format
    append_rows_csv([{"a": 3, "b": 4, "c": 5}], path)    # new column added
    append_rows_csv([{"a": 6, "b": 7}], path)            # old format again
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["a"] for r in rows] == ["1", "3", "6"]
    assert [r["c"] for r in rows] == ["", "5", ""]
