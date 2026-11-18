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


# ------------------------------------------------------------------
# WatermarkX visual theme (colors sampled from the reference screenshot)
# ------------------------------------------------------------------
WX_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --wx-bg:          #111717;   /* main background            */
    --wx-sidebar:     #141d1c;   /* sidebar background         */
    --wx-border:      #252e2d;   /* hairlines / top bar border */
    --wx-border-soft: #1d2625;   /* sidebar / main edge        */
    --wx-editor:      #0d1313;   /* textarea + inputs          */
    --wx-nav-active:  #182c2a;   /* active nav pill fill       */
    --wx-nav-border:  #243735;   /* active nav pill border     */
    --wx-accent:      #57c8b6;   /* teal accent (dot, eyebrow) */
    --wx-accent-text: #7abbb5;   /* active nav label           */
    --wx-btn:         #27625a;   /* primary button             */
    --wx-btn-hover:   #2f7a70;
    --wx-btn-text:    #0d302c;
    --wx-title:       #f2fbfa;
    --wx-text:        #e6f4f4;
    --wx-secondary:   #adb8b4;
    --wx-nav:         #a0acaa;
    --wx-muted:       #6e7c7c;
    --wx-faint:       #697372;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--wx-bg) !important;
    color: var(--wx-secondary);
    font-family: "Plus Jakarta Sans", Inter, "Segoe UI", sans-serif !important;
}

/* ---------- top bar ---------- */
[data-testid="stHeader"] {
    background: var(--wx-bg);
    border-bottom: 1px solid var(--wx-border);
    height: 3.25rem;
}
.wx-topbar {
    position: fixed; top: 0; left: 245px; right: 0; height: 52px; z-index: 999991;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 6rem 0 5.6rem; pointer-events: none;
}
.wx-crumb { font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: #78837f; }
.wx-crumb b { color: var(--wx-text); font-weight: 500; letter-spacing: .02em;
              text-transform: none; font-size: .78rem; }
.wx-crumb i { font-style: normal; margin: 0 .6rem; color: #3b4644; }
.wx-env { font-size: .68rem; color: #6d7d7a; display: flex; align-items: center; gap: .45rem; }
.wx-env:before { content: ""; width: 5px; height: 5px; border-radius: 50%; background: var(--wx-accent); }

/* ---------- main column ---------- */
[data-testid="stMainBlockContainer"] {
    max-width: 800px;
    padding: 5.2rem 2rem 3rem;
}
[data-testid="stVerticalBlock"] { gap: .9rem; }

.wx-eyebrow { color: var(--wx-accent); font-size: .66rem; font-weight: 700;
              letter-spacing: .16em; text-transform: uppercase; margin-bottom: .2rem; }
h1, h2, h3, h4 { color: var(--wx-title) !important; letter-spacing: -.03em; }
h1 { font-size: 2.1rem !important; font-weight: 700 !important; }
h2 { font-size: 2.1rem !important; font-weight: 700 !important; padding-top: 0 !important; }
h3 { font-size: 1.05rem !important; font-weight: 650 !important; color: var(--wx-text) !important; }
p, label, [data-testid="stMarkdownContainer"] { color: var(--wx-secondary); }
[data-testid="stCaptionContainer"] p { color: var(--wx-muted); font-size: .78rem; }
.wx-subtitle { color: var(--wx-secondary); font-size: .9rem; margin: -.4rem 0 .6rem; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--wx-sidebar);
    border-right: 1px solid var(--wx-border-soft);
    min-width: 245px; max-width: 245px;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.1rem; }
.wx-brand { display: flex; align-items: center; gap: .7rem; }
.wx-logo { font-weight: 800; font-size: 1.05rem; letter-spacing: -.08em; color: #30745d; }
.wx-brand-name { color: #e6f8fa; font-size: 1.1rem; font-weight: 700; letter-spacing: -.03em; line-height: 1.1; }
.wx-brand-sub { color: #6b7672; font-size: .56rem; letter-spacing: .14em;
                text-transform: uppercase; margin-top: .3rem; line-height: 1.5; }
.wx-side-label { color: #7b8785; font-size: .64rem; font-weight: 650;
                 letter-spacing: .14em; text-transform: uppercase; margin: 2.4rem 0 .6rem .2rem; }

[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .3rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    border: 1px solid transparent; border-radius: 6px;
    padding: .62rem .75rem; margin: 0; width: 100%;
    color: var(--wx-nav); font-size: .88rem; transition: background .12s, color .12s;
}
[data-testid="stSidebar"] [role="radiogroup"] label p { color: inherit !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #182120; color: var(--wx-text); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: var(--wx-nav-active); border-color: var(--wx-nav-border);
    color: var(--wx-accent-text);
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }

.wx-project {
    position: fixed; left: 0; bottom: 0; width: 245px; padding: 0 1.75rem 1.4rem;
    background: var(--wx-sidebar);
}
.wx-project hr { border: 0; border-top: 1px solid #222b2a; margin: 0 0 1.3rem; }
.wx-project .k { color: #7b8785; font-size: .6rem; font-weight: 650; letter-spacing: .14em; text-transform: uppercase; }
.wx-project .v { color: #e3efed; font-size: .92rem; font-weight: 700; margin: .7rem 0 .55rem; }
.wx-project .s { color: #6b7672; font-size: .66rem; }
.wx-project .ver { display: flex; justify-content: space-between; margin-top: 1.4rem;
                   color: #6b7672; font-size: .6rem; letter-spacing: .12em; text-transform: uppercase; }

/* ---------- inputs ---------- */
[data-testid="stTextArea"] label p, [data-testid="stTextInput"] label p,
[data-testid="stSelectbox"] label p, [data-testid="stSlider"] label p,
[data-testid="stNumberInput"] label p, [data-testid="stRadio"] > label p {
    color: var(--wx-text) !important; font-size: .92rem; font-weight: 650;
}
[data-testid="stTextArea"] textarea, textarea, input, [data-baseweb="select"] > div {
    background: var(--wx-editor) !important;
    color: var(--wx-text) !important;
    border: 1px solid #262f2c !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
[data-baseweb="textarea"], [data-baseweb="base-input"] { background: var(--wx-editor) !important; border-radius: 6px !important; }
textarea { line-height: 1.55 !important; font-size: .9rem !important; }
textarea::placeholder, input::placeholder { color: var(--wx-faint) !important; }
textarea:focus, input:focus {
    border-color: var(--wx-btn) !important; box-shadow: 0 0 0 1px var(--wx-btn) !important;
}

/* ---------- buttons ---------- */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
    min-height: 2.6rem; border-radius: 6px; font-weight: 650;
    background: var(--wx-sidebar); border: 1px solid var(--wx-border); color: var(--wx-text);
    transition: background .12s, border-color .12s;
}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
    background: #182120; border-color: #335a55; color: var(--wx-text);
}
[data-testid="stButton"] button[kind="primary"] {
    background: var(--wx-btn); border: 1px solid #335a55; color: #e6f8f5;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: var(--wx-btn-hover); border-color: #3d6f68; color: #fff;
}
[data-testid="stButton"] button[kind="primary"]:disabled {
    background: var(--wx-btn); color: var(--wx-btn-text); opacity: 1; border-color: #2a4a47;
}

/* ---------- cards ---------- */
[data-testid="stMetric"] {
    background: var(--wx-sidebar); border: 1px solid var(--wx-border);
    border-radius: 7px; padding: .8rem .9rem;
}
[data-testid="stMetricLabel"] p { color: var(--wx-muted) !important; font-size: .76rem; }
[data-testid="stMetricValue"] { color: var(--wx-title) !important; font-size: 1.45rem; font-weight: 650; }
[data-testid="stAlert"] {
    background: var(--wx-sidebar); border: 1px solid var(--wx-border);
    border-left: 2px solid var(--wx-accent); border-radius: 7px; color: var(--wx-secondary);
}
[data-testid="stExpander"] { background: var(--wx-sidebar); border: 1px solid var(--wx-border); border-radius: 7px; }
[data-testid="stDataFrame"] { border: 1px solid var(--wx-border); border-radius: 7px; overflow: hidden; }
hr { border-color: var(--wx-border) !important; }
[data-testid="stProgress"] > div > div { background: var(--wx-btn); }

/* thin dark scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #252e2d; border-radius: 4px; }
::-webkit-scrollbar-track { background: var(--wx-bg); }

@media (max-width: 900px) {
    [data-testid="stSidebar"] { min-width: 205px; max-width: 205px; }
    .wx-topbar { left: 205px; padding: 0 1.5rem; }
    .wx-project { width: 205px; }
}
"""
st.markdown(f"<style>{WX_CSS}</style>", unsafe_allow_html=True)


DATASET_CSV = "experiments/dataset.csv"
DATASET_ATTACK_CSV = "experiments/dataset_attack_results.csv"
DATASET_CHART_DIR = "experiments/charts/dataset"

ATTACK_HELP = {
    "paraphrase": "Qwen rewrites the text in new words (watermark switched off).",
    "synonym": "Replaces a share of words with WordNet synonyms.",
    "translation": "Translates to another language and back (MarianMT).",
    "combined": "Paraphrase first, then synonym replacement.",
}

PAGE_DETAILS = {
    "Generate":   ("TEXT GENERATION",     "Enter a prompt to generate text and begin the watermarking process."),
    "Verify":     ("WATERMARK DETECTION", "Check whether text carries a detectable KGW watermark."),
    "Robustness": ("ROBUSTNESS LAB",      "Test how editing attacks affect watermark detection."),
    "Analytics":  ("EXPERIMENT RESULTS",  "Review detection performance and robustness metrics."),
    "History":    ("ACTIVITY LOG",        "Browse saved generations, verification checks and attack runs."),
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


def page_title(title: str, subtitle: str) -> None:
    st.header(title)
    st.markdown(f'<div class="wx-subtitle">{subtitle}</div>', unsafe_allow_html=True)


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
    page_title("Generate text", PAGE_DETAILS["Generate"][1])
    prompt = st.text_area("Input prompt", "", height=220,
                          placeholder="Start typing your prompt...",
                          help="What would you like the language model to generate?")
    c1, c2, c3 = st.columns(3)
    mode = c1.radio("Generate", ["Watermarked", "Normal", "Both"], horizontal=True, key="gen_mode")
    max_tokens = c2.slider("Max new tokens", 50, 400, 200, step=25)
    seed = c3.number_input("Seed (for repeatable output)", value=42, step=1)

    st.caption("Make your prompt clear and descriptive.")
    if st.button("Generate text  →", type="primary", disabled=not prompt.strip()):
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
    page_title("Verify text", PAGE_DETAILS["Verify"][1])
    st.caption(f"KGW: greenlist {GREENLIST_RATIO}, bias {BIAS}, {SEEDING_SCHEME}, "
               f"threshold z ≥ {Z_THRESHOLD}")
    text = st.text_area("Text to check", st.session_state.get("last_watermarked", ""), height=260,
                        placeholder="Paste text here...")
    if st.button("Check for watermark", type="primary", disabled=not text.strip()):
        engine = get_engine()
        result = detect_text(text, engine)
        database.save_detection(get_db(), text, result, source="verify")
        show_detection(result)
        uncertainty_note()


def page_robustness() -> None:
    page_title("Robustness testing", PAGE_DETAILS["Robustness"][1])
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
    page_title("Analytics", PAGE_DETAILS["Analytics"][1])
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
    page_title("History", PAGE_DETAILS["History"][1])
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
# Chrome: sidebar, top bar, page routing
# ------------------------------------------------------------------
PAGES = {"Generate": page_generate, "Verify": page_verify, "Robustness": page_robustness,
         "Analytics": page_analytics, "History": page_history}

st.sidebar.markdown(
    '<div class="wx-brand"><span class="wx-logo">WX</span><div>'
    '<div class="wx-brand-name">WatermarkX</div>'
    '<div class="wx-brand-sub">LLM watermark<br>robustness</div></div></div>'
    '<div class="wx-side-label">Workspace</div>',
    unsafe_allow_html=True,
)
choice = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
st.sidebar.markdown(
    '<div class="wx-project"><hr><div class="k">Current project</div>'
    '<div class="v">WatermarkX</div><div class="s">LLM Watermark Analysis</div>'
    '<div class="ver"><span>Version</span><span>1.0.0</span></div></div>',
    unsafe_allow_html=True,
)

eyebrow, _ = PAGE_DETAILS[choice]
st.markdown(
    f'<div class="wx-topbar"><div class="wx-crumb">Workspace<i>/</i><b>{choice}</b></div>'
    f'<div class="wx-env">Local environment</div></div>'
    f'<div class="wx-eyebrow">{eyebrow}</div>',
    unsafe_allow_html=True,
)
PAGES[choice]()