"""
app.py - WatermarkX Streamlit interface (Phase 5).

Run (from the WaterMarkX folder, venv active):
    streamlit run app.py

Pages:
    Generate    - generate normal and/or KGW-watermarked text
    Verify      - paste any text and check it for the watermark
    Robustness  - attack a watermarked text and re-check it
    Analytics   - dataset results and comparison charts
    History     - everything saved in SQLite, with CSV download

The model is loaded once (on first use) and shared between pages.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

import database
from attacks import AVAILABLE_ATTACKS, SUPPORTED_PIVOTS, evaluate_attack
from detection import (BIAS, GREENLIST_RATIO, MODEL_NAME, SEEDING_SCHEME,
                       Z_THRESHOLD, detect_text, load_engine)
from generation import generate_text

st.set_page_config(page_title="WatermarkX", layout="wide")

DATASET_CSV = "experiments/dataset.csv"
DATASET_ATTACK_CSV = "experiments/dataset_attack_results.csv"
DATASET_CHART_DIR = "experiments/charts/dataset"

ATTACK_HELP = {
    "paraphrase": "Qwen rewrites the text in new words (watermark switched off).",
    "synonym": "Replaces a share of words with WordNet synonyms.",
    "translation": "Translates to another language and back (MarianMT).",
    "combined": "Paraphrase first, then synonym replacement.",
}


# ------------------------------------------------------------------
# Shared resources
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Qwen and the KGW detector (first time only)...")
def get_engine():
    return load_engine()


def get_db():
    # A new connection per run is safest with Streamlit's threads.
    return database.connect()


def show_detection(result: dict, title: str | None = None) -> None:
    """KGW result as metrics plus a plain-language explanation."""
    if title:
        st.markdown(f"**{title}**")
    if result.get("error"):
        st.error(f"Could not score this text: {result['error']}")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("z-score", f"{result['z_score']:.2f}", help=f"Detected if z ≥ {Z_THRESHOLD}")
    c2.metric("Green-token fraction", f"{result['green_fraction']:.2f}",
              help="About 0.5 is expected for unwatermarked text")
    c3.metric("Tokens scored", result["num_tokens_scored"])
    c4.metric("p-value", f"{result['p_value']:.1e}")
    if result["is_watermarked"]:
        st.success(f"Watermark detected (z = {result['z_score']:.2f} ≥ {Z_THRESHOLD}).")
    else:
        st.info(f"Watermark not detected (z = {result['z_score']:.2f} < {Z_THRESHOLD}).")
    if result.get("warning"):
        st.warning(result["warning"])


def uncertainty_note() -> None:
    st.caption(
        "This is statistical evidence, not proof of authorship. A detection means the text "
        "contains far more 'green' tokens than chance would give. **No detection does not "
        "mean a human wrote it**: the text may be from another model, or edited, or too short."
    )


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------
def page_generate() -> None:
    st.header("Generate text")
    prompt = st.text_area("Prompt", "Explain how vaccines work in simple words.", height=100)
    c1, c2, c3 = st.columns(3)
    mode = c1.radio("Generate", ["Watermarked", "Normal", "Both"], horizontal=True, key="gen_mode")
    max_tokens = c2.slider("Max new tokens", 50, 400, 200, step=25)
    seed = c3.number_input("Seed (for repeatable output)", value=42, step=1)

    if st.button("Generate", type="primary", disabled=not prompt.strip()):
        engine = get_engine()
        conn = get_db()
        conditions = {"Watermarked": [True], "Normal": [False], "Both": [False, True]}[mode]
        cols = st.columns(len(conditions))
        for col, wm in zip(cols, conditions):
            label = "Watermarked" if wm else "Normal"
            with col:
                with st.spinner(f"Generating {label.lower()} text (slow on CPU)..."):
                    text = generate_text(prompt, engine, watermark=wm,
                                         max_new_tokens=max_tokens, seed=int(seed))
                    result = detect_text(text, engine)
                database.save_detection(conn, text, result,
                                        source=f"generate-{label.lower()}", prompt=prompt)
                st.subheader(label)
                st.text_area(f"{label} text", text, height=260, key=f"out_{label}")
                show_detection(result)
                if wm:
                    st.session_state["last_watermarked"] = text
        uncertainty_note()
        st.caption("Saved to History. The watermarked text is ready on the Robustness page.")


def page_verify() -> None:
    st.header("Verify text")
    st.write("Paste any text to check it for the KGW watermark "
             f"(greenlist {GREENLIST_RATIO}, bias {BIAS}, {SEEDING_SCHEME}, threshold z ≥ {Z_THRESHOLD}).")
    text = st.text_area("Text to check", st.session_state.get("last_watermarked", ""), height=260)
    if st.button("Check for watermark", type="primary", disabled=not text.strip()):
        engine = get_engine()
        result = detect_text(text, engine)
        database.save_detection(get_db(), text, result, source="verify")
        show_detection(result)
        uncertainty_note()


def page_robustness() -> None:
    st.header("Robustness testing")
    st.write("Apply an attack to watermarked text, then check both versions with the same detector.")
    text = st.text_area("Watermarked text", st.session_state.get("last_watermarked", ""), height=220,
                        help="Generate one on the Generate page, or paste your own.")

    c1, c2 = st.columns([1, 2])
    attack = c1.selectbox("Attack", AVAILABLE_ATTACKS, format_func=str.capitalize)
    c2.caption(ATTACK_HELP[attack])
    params: dict = {}
    p1, p2 = st.columns(2)
    if attack in ("synonym", "combined"):
        params["proportion"] = p1.slider("Share of eligible words to replace", 0.0, 1.0, 0.3, 0.05)
    if attack == "translation":
        params["pivot"] = p1.selectbox("Pivot language", SUPPORTED_PIVOTS,
                                       format_func={"de": "German", "fr": "French", "es": "Spanish"}.get)
    else:
        params["seed"] = int(p2.number_input("Seed", value=42, step=1))

    if st.button("Run attack", type="primary", disabled=not text.strip()):
        engine = get_engine()
        with st.spinner(f"Running {attack} attack..."):
            rec = evaluate_attack(text, attack, engine, **params)
        database.save_attack(get_db(), rec)

        if rec.error:
            st.error(f"Attack failed: {rec.error}")
        verdict = {
            "watermark_removed": ("success", "Watermark removed: detected before, not detected after."),
            "watermark_survived": ("info", "Watermark survived: still detected after the attack."),
            "not_comparable": ("warning", "The original text was not detected as watermarked, "
                                          "so this sample cannot show the watermark being removed."),
            "attack_failed": ("error", "The attack failed; see the error above."),
            "detection_failed": ("error", "The attacked text could not be scored."),
        }[rec.outcome]
        getattr(st, verdict[0])(verdict[1])

        left, right = st.columns(2)
        fmt = lambda v, f="{:.2f}": "n/a" if v is None else f.format(v)
        with left:
            st.subheader("Before attack")
            st.metric("z-score", fmt(rec.original_z_score))
            st.metric("Green-token fraction", fmt(rec.original_green_fraction))
            st.metric("Detected", str(rec.original_prediction))
            st.text_area("Original", rec.original_text, height=260, key="orig_view")
        with right:
            st.subheader("After attack")
            delta = (None if rec.attacked_z_score is None or rec.original_z_score is None
                     else f"{rec.attacked_z_score - rec.original_z_score:+.2f}")
            st.metric("z-score", fmt(rec.attacked_z_score), delta=delta, delta_color="off")
            st.metric("Green-token fraction", fmt(rec.attacked_green_fraction))
            st.metric("Detected", str(rec.attacked_prediction))
            st.text_area("Attacked", rec.attacked_text or "", height=260, key="att_view")

        st.metric("Meaning kept (similarity, 1.0 = same)", fmt(rec.semantic_similarity))
        if rec.warnings:
            with st.expander("Warnings"):
                for w in rec.warnings.split("; "):
                    st.write("- " + w)
        with st.expander("Attack parameters"):
            st.code(rec.attack_params, language="json")


def page_analytics() -> None:
    st.header("Analytics")
    from metrics import baseline_summary, summarize

    if not os.path.exists(DATASET_CSV):
        st.info("No dataset yet. Build it from a terminal:\n\n"
                "`python build_dataset.py generate`, then `attack`, then `summary`.")
        return

    from build_dataset import latest_rows, read_csv
    rows = latest_rows(read_csv(DATASET_CSV), ("prompt_id", "condition"))
    st.subheader("Baseline detection (no attack)")
    base = pd.DataFrame(baseline_summary(rows))
    st.dataframe(base.style.format(precision=2, na_rep="n/a"), hide_index=True)
    st.caption("Watermarked detection rate = true positive rate. "
               "Normal detection rate = false positive rate.")

    attack_rows = latest_rows(read_csv(DATASET_ATTACK_CSV), ("prompt_id", "attack_type"))
    if attack_rows:
        st.subheader("Attacks")
        st.dataframe(pd.DataFrame(summarize(attack_rows)).style.format(precision=2, na_rep="n/a"),
                     hide_index=True)
        st.caption("Only samples detected before the attack are 'comparable'. "
                   "detection_rate_after = share still detected (lower = stronger attack).")
    else:
        st.info("No attack results yet: run `python build_dataset.py attack`.")

    if st.button("Redraw charts"):
        with st.spinner("Drawing charts..."):
            from build_dataset import summary
            buf = io.StringIO()
            from contextlib import redirect_stdout
            with redirect_stdout(buf):
                summary()
    charts = ["baseline_z_scores.png", "z_score_before_after.png", "detection_rate_after.png",
              "green_fraction_before_after.png", "meaning_similarity.png"]
    found = [c for c in charts if os.path.exists(os.path.join(DATASET_CHART_DIR, c))]
    if not found:
        st.caption("No charts yet: click 'Redraw charts'.")
    for c in found:
        st.image(os.path.join(DATASET_CHART_DIR, c), use_container_width=True)


def page_history() -> None:
    st.header("History")
    conn = get_db()
    for table, title in (("detections", "Generated and verified texts"), ("attacks", "Attack runs")):
        rows = database.list_rows(conn, table)
        st.subheader(f"{title} ({len(rows)})")
        if not rows:
            st.caption("Nothing saved yet.")
            continue
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, height=300)
        st.download_button(f"Download {table}.csv", df.iloc[::-1].to_csv(index=False),
                           file_name=f"watermarkx_{table}.csv", mime="text/csv")


# ------------------------------------------------------------------
PAGES = {"Generate": page_generate, "Verify": page_verify, "Robustness": page_robustness,
         "Analytics": page_analytics, "History": page_history}

st.sidebar.title("WatermarkX")
st.sidebar.caption("AI text provenance verification and robustness analysis")
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.divider()
st.sidebar.caption(f"Model: {MODEL_NAME}\n\nKGW: greenlist {GREENLIST_RATIO}, bias {BIAS}, "
                   f"{SEEDING_SCHEME}, z ≥ {Z_THRESHOLD}")
PAGES[choice]()
