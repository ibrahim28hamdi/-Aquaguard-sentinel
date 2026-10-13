"""AquaGuard Sentinel AI — Science-First Research Dashboard.

Seven fully functional sections built around the project's real capabilities.
Run:  streamlit run app.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from aquaguard.loaders import (
    synthetic_demo, load_nyc, load_uci, load_usgs, load_custom,
    NYC_URL, UCI_URL, USGS_URL,
)
from aquaguard.experiment import test_benchmark
from aquaguard.fetch_uci import ensure_uci_file
from aquaguard.model import Sentinel, split_three, select_features, numeric

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AquaGuard Sentinel AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# COLOUR SYSTEM  — deep navy + cyan/teal scientific palette
# ═══════════════════════════════════════════════════════════════════════
N0  = "#040C1A"   # deepest navy
N1  = "#071424"
N2  = "#0A1B30"
N3  = "#0D2240"
C0  = "#00C8F0"   # primary cyan
C1  = "#00A8CC"
T0  = "#00B898"   # primary teal
T1  = "#008F74"
W0  = "#E8F4FB"   # primary text
W1  = "#9BB5C8"   # dimmed text
AMB = "#F0A020"   # amber warning
RED = "#E84040"
GRN = "#38C878"
PRP = "#7B6CF0"
BLU = "#1E6FD4"
BOR = "rgba(0,200,240,.14)"   # border
BOR2= "rgba(0,200,240,.25)"

# ═══════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════
_DEFAULTS: dict = {
    "page":        "Home",
    "kind":        "Synthetic demo (offline)",
    "run_done":    False,
    "data":        None,
    "dataset_key": None,
    "summary":     None,
    "trials":      None,
    "manifest":    None,
    "mdl":         None,
    "test_df":     None,
    "alerts_df":   None,
    "scores_arr":  None,
    "pvals_arr":   None,
    "feat_list":   [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
/* ── fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"] {{
    background: {N0} !important;
    color: {W0};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
[data-testid="stAppViewContainer"] > .main {{ background: {N0} !important; }}
.block-container {{ padding-top: 0 !important; max-width: 1360px; }}
* {{ box-sizing: border-box; }}
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {N1}; }}
::-webkit-scrollbar-thumb {{ background: rgba(0,200,240,.22); border-radius: 3px; }}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {N1} 0%, {N2} 60%, #091E38 100%) !important;
    border-right: 1px solid {BOR};
    min-width: 224px !important;
    max-width: 238px !important;
}}
[data-testid="stSidebar"] * {{ color: {W0} !important; }}
[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important;
    border: none !important;
    color: {W1} !important;
    text-align: left !important;
    padding: .46rem .85rem !important;
    width: 100% !important;
    border-radius: 7px !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
    justify-content: flex-start !important;
    letter-spacing: .01em !important;
    transition: background .12s, color .12s !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(0,200,240,.08) !important;
    color: {W0} !important;
}}
.sb-brand {{
    display: flex; align-items: center; gap: .55rem;
    padding: 1.2rem .9rem .95rem;
    border-bottom: 1px solid {BOR};
    margin-bottom: .3rem;
}}
.sb-brand-dot {{
    width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0;
    background: linear-gradient(135deg, {C0}, {T0});
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}}
.sb-brand-txt b  {{ font-size: .86rem; font-weight: 700; display: block; line-height: 1.2; }}
.sb-brand-txt em {{ font-size: .64rem; font-style: normal; color: {W1} !important; }}
.nav-active > button {{
    background: linear-gradient(90deg,rgba(0,200,240,.15),rgba(0,184,152,.07)) !important;
    border-left: 2.5px solid {C0} !important;
    color: {W0} !important;
    padding-left: calc(.85rem - 2.5px) !important;
}}
.sb-sep {{ height: 1px; background: {BOR}; margin: .42rem .65rem; }}
.sb-section-lbl {{
    font-size: .59rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: {C0} !important;
    padding: .35rem .85rem .18rem;
}}
.sb-footer {{
    padding: .8rem .9rem; border-top: 1px solid {BOR};
    font-size: .68rem; color: rgba(155,181,200,.42) !important;
    line-height: 1.55;
}}

/* ── page header ── */
.pg-header {{
    padding: 1.6rem 0 .5rem;
    border-bottom: 1px solid {BOR};
    margin-bottom: 1.4rem;
}}
.pg-header h1 {{
    font-size: 1.45rem; font-weight: 800; letter-spacing: -.03em;
    color: {W0}; margin: 0 0 .2rem;
}}
.pg-header p {{ font-size: .86rem; color: {W1}; margin: 0; line-height: 1.5; }}

/* ── section divider ── */
.sec-hdr {{
    display: flex; align-items: center; gap: .6rem;
    margin: 1.7rem 0 .85rem;
}}
.sec-hdr-line {{
    flex: 1; height: 1px;
    background: linear-gradient(90deg, {BOR2}, transparent);
}}
.sec-hdr h3 {{
    font-size: .78rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: {C0}; margin: 0; white-space: nowrap;
}}

/* ── cards ── */
.card {{
    background: linear-gradient(145deg, {N2}, {N1});
    border: 1px solid {BOR};
    border-radius: 12px;
    padding: 1rem 1.1rem .95rem;
    position: relative; overflow: hidden;
    transition: border-color .18s, box-shadow .18s;
}}
.card::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, {C0}, {T0}); opacity: .45;
}}
.card:hover {{ border-color: {BOR2}; box-shadow: 0 3px 18px rgba(0,200,240,.07); }}
.card-lbl {{
    font-size: .61rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: {W1}; margin-bottom: .2rem;
}}
.card-val {{ font-size: 1.7rem; font-weight: 800; color: {W0}; line-height: 1; }}
.card-sub {{ font-size: .68rem; margin-top: .25rem; }}
.card-sub .g {{ color: {GRN}; }}
.card-sub .c {{ color: {C0}; }}

/* ── metric grid ── */
.metric-grid {{
    display: grid; grid-template-columns: repeat(5, 1fr); gap: .75rem;
    margin-bottom: 1.2rem;
}}
@media (max-width: 1100px) {{ .metric-grid {{ grid-template-columns: repeat(3,1fr); }} }}
@media (max-width: 700px)  {{ .metric-grid {{ grid-template-columns: repeat(2,1fr); }} }}

/* ── chart panel ── */
.chart-panel {{
    background: linear-gradient(145deg, {N2}, {N1});
    border: 1px solid {BOR};
    border-radius: 12px;
    padding: 1.1rem 1.2rem .9rem;
    margin-bottom: .85rem;
}}
.chart-panel-title {{ font-size: .84rem; font-weight: 700; color: {W0}; margin-bottom: .15rem; }}
.chart-panel-sub   {{ font-size: .7rem;  color: {W1}; margin-bottom: .75rem; }}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {{ background: {N2}; border-radius: 8px; gap: 3px; padding: 3px; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 6px; color: {W1}; font-size: .79rem; font-weight: 600; padding: .36rem .82rem; }}
.stTabs [aria-selected="true"] {{ background: linear-gradient(90deg,{BLU},{T1}) !important; color: {W0} !important; }}

/* ── dataframe ── */
[data-testid="stDataFrame"] {{ border: 1px solid {BOR} !important; border-radius: 10px !important; overflow: hidden; }}

/* ── expander ── */
[data-testid="stExpander"] {{ background: {N2}; border: 1px solid {BOR} !important; border-radius: 9px !important; }}

/* ── disclaimer ── */
.disc {{
    display: flex; align-items: flex-start; gap: .6rem;
    background: rgba(240,160,32,.07); border-left: 3px solid {AMB};
    border-radius: 0 8px 8px 0; padding: .7rem .95rem;
    font-size: .81rem; color: #F2C040; margin: 0 0 .9rem; line-height: 1.55;
}}

/* ── data badge ── */
.badge-synth {{
    display: flex; align-items: flex-start; gap: .55rem;
    background: rgba(240,160,32,.07); border: 1px solid rgba(240,160,32,.25);
    border-radius: 8px; padding: .6rem .9rem; font-size: .8rem; color: #F2C040;
    margin-bottom: 1rem;
}}
.badge-real {{
    display: flex; align-items: flex-start; gap: .55rem;
    background: rgba(0,200,240,.05); border: 1px solid {BOR};
    border-radius: 8px; padding: .6rem .9rem; font-size: .8rem; color: #7DD8F0;
    margin-bottom: 1rem;
}}

/* ── stat chip ── */
.stat-row {{ display: flex; gap: .7rem; flex-wrap: wrap; margin-bottom: .9rem; }}
.stat {{
    background: {N2}; border: 1px solid {BOR}; border-radius: 9px;
    padding: .65rem .9rem; flex: 1 1 110px; min-width: 100px;
}}
.stat-lbl {{ font-size: .61rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: {W1}; margin-bottom: .18rem; }}
.stat-val {{ font-size: 1.05rem; font-weight: 700; color: {W0}; }}

/* ── run-needed banner ── */
.run-needed {{
    background: rgba(0,200,240,.05); border: 1px solid {BOR};
    border-radius: 10px; padding: 2rem; text-align: center; margin: 1.5rem 0;
}}
.run-needed h3 {{ color: {C0}; font-size: 1rem; font-weight: 700; margin: 0 0 .4rem; }}
.run-needed p  {{ color: {W1}; font-size: .84rem; margin: 0; }}

/* ── method pill ── */
.method-pill {{
    display: inline-flex; align-items: center; gap: .3rem;
    background: rgba(0,200,240,.1); border: 1px solid {BOR2};
    color: {C0}; font-size: .72rem; font-weight: 600;
    padding: .2rem .6rem; border-radius: 10px; margin: .1rem;
}}

/* ── workflow steps ── */
.workflow {{ display: flex; gap: 0; align-items: stretch; flex-wrap: wrap; margin: .8rem 0 1.2rem; }}
.wf-step {{
    flex: 1 1 120px; background: {N2}; border: 1px solid {BOR};
    border-radius: 0; padding: .8rem .9rem; position: relative;
}}
.wf-step:first-child {{ border-radius: 9px 0 0 9px; }}
.wf-step:last-child  {{ border-radius: 0 9px 9px 0; }}
.wf-step-num {{
    font-size: .62rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: {C0}; margin-bottom: .22rem;
}}
.wf-step-title {{ font-size: .82rem; font-weight: 700; color: {W0}; margin-bottom: .15rem; }}
.wf-step-desc  {{ font-size: .71rem; color: {W1}; line-height: 1.5; }}
.wf-arrow {{
    display: flex; align-items: center; padding: 0 .1rem;
    color: {C0}; font-size: 1rem; flex-shrink: 0;
}}

/* ── quality donut ── */
.dq-tile {{
    background: {N2}; border: 1px solid {BOR}; border-radius: 10px;
    padding: .85rem .95rem; text-align: center;
}}
.dq-val {{ font-size: 1.55rem; font-weight: 800; color: {W0}; line-height: 1; }}
.dq-lbl {{ font-size: .65rem; color: {W1}; margin-top: .22rem; }}

/* ── limitation item ── */
.lim {{
    display: flex; align-items: flex-start; gap: .45rem;
    margin-bottom: .5rem; font-size: .81rem; color: {W1};
}}

/* ── footer ── */
.page-footer {{
    position: relative; margin-top: 2.5rem; padding: 1.1rem 0 .6rem;
    text-align: center; font-size: .71rem;
    color: rgba(155,181,200,.35); letter-spacing: .025em;
}}
.page-footer::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, {BOR2}, transparent);
}}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════════
def _plo(h=220, margin=None):
    m = margin or dict(l=0, r=0, t=4, b=0)
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=W1, family="Inter, sans-serif", size=10),
        height=h, margin=m,
        xaxis=dict(gridcolor="rgba(0,200,240,.08)", zerolinecolor="rgba(0,200,240,.15)",
                   tickfont=dict(size=9)),
        yaxis=dict(gridcolor="rgba(0,200,240,.08)", zerolinecolor="rgba(0,200,240,.15)",
                   tickfont=dict(size=9)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=W1)),
    )

METHOD_COLORS = {
    "isolation_forest":  C0,
    "robust_global":     T0,
    "sentinel_station":  PRP,
}
METHOD_LABELS = {
    "isolation_forest":  "Isolation Forest",
    "robust_global":     "Robust Global",
    "sentinel_station":  "Sentinel Station",
}
STRESS_LABELS = {
    "untouched_heldout":          "Untouched",
    "artificial_spike":           "Spike",
    "artificial_gradual_drift":   "Drift",
    "artificial_sensor_dropout":  "Dropout",
}
STRESS_COLORS = [T0, C0, PRP, AMB]

# ═══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════
def _sec(label: str, icon: str = ""):
    ic = f'<span style="opacity:.65;font-size:.95rem">{icon}</span> ' if icon else ""
    st.markdown(f"""
<div class="sec-hdr">
  <h3>{ic}{label}</h3>
  <div class="sec-hdr-line"></div>
</div>""", unsafe_allow_html=True)


def _disc():
    st.markdown(f"""
<div class="disc">
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
       stroke="{AMB}" stroke-width="2.3" stroke-linecap="round"
       stroke-linejoin="round" style="flex-shrink:0;margin-top:.15rem">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0
             1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/>
    <line x1="12" y1="9"  x2="12"    y2="13"/>
    <line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
  <div><strong>Research &amp; education only.</strong>
  Anomaly alerts flag unusual sensor readings &#8212; they do <em>not</em> certify water safety,
  prove contamination, or serve as public-health guidance.
  All stress tests inject <em>artificial</em> faults; no benchmark here measures real
  contamination detection.</div>
</div>""", unsafe_allow_html=True)


def _data_badge():
    dk = st.session_state.dataset_key
    if not dk:
        return
    if dk == "synthetic":
        st.markdown("""
<div class="badge-synth">&#9888;&#65039;
  <div><strong>Synthetic demo data.</strong>
  Computer-generated sensor signals &#8212; not real environmental measurements.</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="badge-real">&#128202;
  <div><strong>Real historical measurements</strong> with
  <em>artificially injected</em> sensor faults.
  No verified contamination labels. Dataset: <code>{dk.upper()}</code>.</div>
</div>""", unsafe_allow_html=True)


def _no_run(section="this section"):
    st.markdown(f"""
<div class="run-needed">
  <h3>&#9654; Run an experiment first</h3>
  <p>Select a dataset in the sidebar and click
  <strong>Run Transparent Comparison</strong> to populate {section}.</p>
</div>""", unsafe_allow_html=True)


def _metric_cards():
    m = st.session_state.manifest
    total, tr, ca, ho = (m["total_records"], m["train"],
                         m["calibration"], m["holdout"])
    fn = len(m["features"])
    tp = round(100 * tr / total) if total else 0
    cp = round(100 * ca / total) if total else 0
    hp = round(100 * ho / total) if total else 0
    cards = [
        ("&#128230;", "Total Records",      f"{total:,}", f'<span class="c">source rows</span>'),
        ("&#128200;", f"Training ({tp}%)",  f"{tr:,}",    f'<span class="g">fit anomaly models</span>'),
        ("&#9201;",   f"Calibration ({cp}%)",f"{ca:,}",   f'<span class="c">set alert thresholds</span>'),
        ("&#128274;", f"Held-out ({hp}%)",  f"{ho:,}",    f'<span class="c">evaluate on unseen data</span>'),
        ("&#128225;", "Sensor Features",    str(fn),      f'<span class="c">numeric columns used</span>'),
    ]
    html = '<div class="metric-grid">'
    for ico, lbl, val, sub in cards:
        html += f"""<div class="card">
  <div style="font-size:1.2rem;margin-bottom:.45rem">{ico}</div>
  <div class="card-lbl">{lbl}</div>
  <div class="card-val">{val}</div>
  <div class="card-sub">{sub}</div>
</div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _lims():
    lims = st.session_state.manifest.get("claim_limitations", []) \
           if st.session_state.manifest else []
    for l in lims:
        st.markdown(f"""
<div class="lim">
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
       stroke="{AMB}" stroke-width="2.5"
       style="flex-shrink:0;margin-top:.18rem">
    <polyline points="9 18 15 12 9 6"/>
  </svg>{l}</div>""", unsafe_allow_html=True)


def _footer():
    st.markdown(f"""
<div class="page-footer">
  AquaGuard Sentinel AI &#183; Educational ML Portfolio &#183;
  Code developed with AI assistance &#183; MIT licence &#183;
  Datasets separately licensed by their publishers
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# CACHED COMPUTATION
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_dataset(kind: str, max_rows):
    if kind.startswith("Synthetic"):  return synthetic_demo()
    if kind.startswith("UCI"):        ensure_uci_file(); return load_uci()
    if kind.startswith("NYC"):        return load_nyc(max_rows=max_rows)
    if kind.startswith("USGS"):       return load_usgs()
    raise ValueError("Use CSV uploader.")


@st.cache_data(show_spinner=False)
def _fit_sentinel_model(cache_key: str, _data: pd.DataFrame):
    """Fit Sentinel on train/cal; score test set with all 3 methods."""
    try:
        tr, cal, te, _ = split_three(_data)
        mdl = Sentinel().fit(tr, cal)
        out = {}
        for method in ("robust_global", "isolation_forest", "sentinel_station"):
            alerts = mdl.alerts(te, method, alpha=0.05)
            scores = mdl.score(te, method)
            pvals  = mdl.pvalues(te, method)
            out[method] = dict(alerts=alerts, scores=scores, pvals=pvals)
        return mdl, te, out, mdl.features
    except Exception:
        return None, None, {}, []


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
NAV = [
    ("Home",               "&#127968;"),
    ("Overview",           "&#128204;"),
    ("Datasets & Quality", "&#128196;"),
    ("Anomaly Explorer",   "&#128268;"),
    ("Model Comparison",   "&#128200;"),
    ("Stress-Test Lab",    "&#9889;"),
    ("Methodology",        "&#128218;"),
    ("Results & Reports",  "&#11015;"),
]

with st.sidebar:
    st.markdown(f"""
<div class="sb-brand">
  <div class="sb-brand-dot">&#128167;</div>
  <div class="sb-brand-txt">
    <b>AquaGuard Sentinel</b>
    <em>Water-Quality Anomaly Research</em>
  </div>
</div>""", unsafe_allow_html=True)

    for label, icon in NAV:
        active = st.session_state.page == label
        wrap   = "nav-active" if active else ""
        if label == "Datasets & Quality":
            st.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)
        if label == "Overview":
            st.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)
        with st.container():
            if wrap:
                st.markdown(f'<div class="{wrap}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"nav_{label}",
                         use_container_width=True):
                st.session_state.page = label
                st.rerun()
            if wrap:
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-lbl">Dataset</div>', unsafe_allow_html=True)

    KIND_OPTIONS = [
        "Synthetic demo (offline)",
        "UCI wastewater (local / internet)",
        "NYC drinking water (internet)",
        "USGS river/estuary (internet)",
        "Upload CSV (your own or public)",
    ]
    try:
        kind_idx = KIND_OPTIONS.index(st.session_state.kind)
    except ValueError:
        kind_idx = 0

    kind = st.selectbox("Dataset", KIND_OPTIONS, index=kind_idx,
                        label_visibility="collapsed", key="kind_sel",
                        help="Each dataset evaluated separately, never combined.")

    # Invalidate stale results if dataset changed
    if kind != st.session_state.kind:
        for k in list(_DEFAULTS.keys()):
            if k not in ("page",):
                st.session_state[k] = _DEFAULTS[k]
        st.session_state.kind = kind

    max_rows   = None
    csv_file   = None
    water_type = None
    if "NYC" in kind:
        max_rows = st.slider("Max NYC rows", 1000, 50_000, 12_000, step=1000)
    if "Upload" in kind:
        csv_file   = st.file_uploader("Upload CSV", type=["csv"],
                                      help="≥ 150 rows, ≥ 2 numeric columns.")
        water_type = st.selectbox("Water type",
                                  ["river", "drinking", "wastewater", "other"])

    st.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)

    run_disabled = ("Upload" in kind and csv_file is None)
    if st.button("&#9654;  Run Transparent Comparison",
                 type="primary", use_container_width=True,
                 disabled=run_disabled, key="run_btn"):

        with st.spinner("Loading dataset…"):
            try:
                data = (load_custom(csv_file, water_type=water_type)
                        if "Upload" in kind
                        else _load_dataset(kind, max_rows))
                st.session_state.data       = data
                st.session_state.dataset_key = kind.split()[0].lower()
            except Exception as exc:
                st.error(f"Could not load dataset: {exc}")
                st.stop()

        with st.spinner("Running benchmark (10–30 s)…"):
            try:
                summary, trials, manifest = test_benchmark(
                    st.session_state.data,
                    dataset=st.session_state.dataset_key)
                st.session_state.summary  = summary
                st.session_state.trials   = trials
                st.session_state.manifest = manifest
            except Exception as exc:
                st.error(f"Benchmark failed: {exc}")
                st.stop()

        with st.spinner("Fitting Sentinel model for visualisations…"):
            mdl, te, method_out, feats = _fit_sentinel_model(
                st.session_state.dataset_key, st.session_state.data)
            st.session_state.mdl        = mdl
            st.session_state.test_df    = te
            st.session_state.method_out = method_out   # dict keyed by method
            st.session_state.feat_list  = feats
            st.session_state.run_done   = True

        st.session_state.page = "Overview"
        st.rerun()

    if run_disabled:
        st.caption("Upload a CSV to enable.")

    st.markdown(f"""
<div class="sb-footer">
  <strong style="color:{C0}">Synthetic demo</strong> &#8212; instant, no internet.<br>
  <strong style="color:{C0}">UCI dataset</strong> &#8212; auto-downloaded on first use.<br><br>
  Not a water-safety tool. Research use only.
</div>""", unsafe_allow_html=True)

# ── convenience aliases ──────────────────────────────────────────────
page        = st.session_state.page
run_done    = st.session_state.run_done
data        = st.session_state.data
dataset_key = st.session_state.dataset_key
summary     = st.session_state.summary
trials      = st.session_state.trials
manifest    = st.session_state.manifest
mdl         = st.session_state.mdl
test_df     = st.session_state.test_df
feat_list   = st.session_state.feat_list
method_out  = st.session_state.get("method_out", {})


# ╔═══════════════════════════════════════════════════════════════════════
# ║  PAGE 0 — HOME
# ╚═══════════════════════════════════════════════════════════════════════
if page == "Home":

    # ── Hero CSS (animations only — no SVG content in markdown) ──────
    st.markdown(f"""
<style>
@keyframes ripple {{
  0%   {{ transform: scale(1);   opacity: .6; }}
  100% {{ transform: scale(2.8); opacity: 0;  }}
}}
@keyframes wave-scroll {{
  0%   {{ background-position-x: 0; }}
  100% {{ background-position-x: 1440px; }}
}}
@keyframes glow-pulse {{
  0%,100% {{ opacity: .55; }}
  50%     {{ opacity: .90; }}
}}
@keyframes float-up {{
  0%   {{ transform: translateY(0);   opacity: .7; }}
  60%  {{ opacity: .85; }}
  100% {{ transform: translateY(-55px); opacity: 0; }}
}}

/* dark navy page background */
.hero-shell {{
    background: radial-gradient(ellipse 90% 70% at 60% 40%,
                  #082240 0%, #050E1F 55%, #030A14 100%);
    border-radius: 14px;
    border: 1px solid rgba(0,200,240,.12);
    padding: 0;
    margin-bottom: 1.4rem;
    overflow: hidden;
    position: relative;
}}
/* cyan glow top-right */
.hero-shell::before {{
    content: "";
    position: absolute; top: -60px; right: -60px;
    width: 340px; height: 340px; border-radius: 50%;
    background: radial-gradient(circle,
        rgba(0,200,240,.18) 0%, transparent 70%);
    pointer-events: none;
}}
/* teal glow bottom-left */
.hero-shell::after {{
    content: "";
    position: absolute; bottom: -40px; left: -40px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle,
        rgba(0,184,152,.12) 0%, transparent 68%);
    pointer-events: none;
}}
/* wave bar at bottom of hero */
.hero-wave-bar {{
    height: 5px;
    background: linear-gradient(90deg, transparent 0%, {C0} 30%,
                {T0} 65%, transparent 100%);
    opacity: .45;
}}
/* ripple circles behind image */
.ripple-ring {{
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(0,200,240,.35);
    animation: ripple 3s ease-out infinite;
    pointer-events: none;
}}
.ripple-ring:nth-child(2) {{ animation-delay: 1s;  opacity: .55; }}
.ripple-ring:nth-child(3) {{ animation-delay: 2s;  opacity: .35; }}
/* floating bubble dots */
.bubble {{
    position: absolute; border-radius: 50%;
    background: radial-gradient(circle at 35% 30%,
        rgba(0,200,240,.30), rgba(0,200,240,.04));
    border: 1px solid rgba(0,200,240,.40);
    animation: float-up 5s ease-in-out infinite;
    pointer-events: none;
}}
.bubble:nth-child(2) {{ animation-delay: 1.4s; }}
.bubble:nth-child(3) {{ animation-delay: 2.8s; }}
.bubble:nth-child(4) {{ animation-delay: 0.7s; }}
/* eyebrow badge */
.eyebrow {{
    display: inline-flex; align-items: center; gap: .4rem;
    background: rgba(0,200,240,.10);
    border: 1px solid rgba(0,200,240,.28);
    color: {C0}; font-size: .7rem; font-weight: 700;
    letter-spacing: .13em; text-transform: uppercase;
    padding: .24rem .78rem; border-radius: 20px;
    margin-bottom: .85rem;
}}
.eyebrow-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: {C0}; flex-shrink: 0;
    animation: glow-pulse 2s ease-in-out infinite;
}}
/* hero title */
.hero-title {{
    font-size: clamp(1.9rem, 3.6vw, 2.9rem);
    font-weight: 800; letter-spacing: -.035em; line-height: 1.08;
    color: {W0}; margin: 0 0 .1rem;
}}
.hero-title-accent {{
    background: linear-gradient(95deg, {C0} 0%, {T0} 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero-tagline {{
    font-size: 1rem; font-weight: 600; color: {W0};
    margin: .12rem 0 .6rem; line-height: 1.35;
}}
.hero-desc {{
    font-size: .86rem; color: {W1}; line-height: 1.68;
    max-width: 440px; margin-bottom: 1.15rem;
}}
.badge-row {{
    display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1.2rem;
}}
.hero-badge {{
    display: inline-flex; align-items: center; gap: .28rem;
    background: rgba(0,184,152,.10);
    border: 1px solid rgba(0,184,152,.28);
    color: #6ECEC7; font-size: .71rem; font-weight: 500;
    padding: .19rem .58rem; border-radius: 11px;
}}
.cta-row {{ display: flex; gap: .7rem; flex-wrap: wrap; margin-bottom: 1rem; }}
.cta-primary {{
    display: inline-flex; align-items: center; gap: .4rem;
    background: linear-gradient(90deg, {C0}, {T0});
    color: {N0}; font-size: .84rem; font-weight: 700;
    padding: .56rem 1.2rem; border-radius: 8px;
    box-shadow: 0 4px 16px rgba(0,200,240,.28);
    cursor: default; white-space: nowrap;
}}
.cta-secondary {{
    display: inline-flex; align-items: center; gap: .4rem;
    background: transparent; color: {C0};
    font-size: .84rem; font-weight: 600;
    padding: .56rem 1.2rem; border-radius: 8px;
    border: 1px solid rgba(0,200,240,.30);
    cursor: default; white-space: nowrap;
}}
.feat-row {{ display: flex; gap: 1.3rem; flex-wrap: wrap; margin-top: .85rem; }}
.feat-item {{
    display: flex; align-items: center; gap: .38rem;
    font-size: .76rem; color: {W1};
}}
.feat-dot {{
    width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient({C0}, {T0});
}}
/* image wrapper keeps the ripples positioned relative to it */
.img-wrap {{
    position: relative;
    display: flex; align-items: center; justify-content: center;
    padding: 1.4rem .5rem 1rem;
}}
</style>
""", unsafe_allow_html=True)

    # ── generate hero illustration (matplotlib → base64 PNG) ─────────
    import io as _io, base64 as _b64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import matplotlib.patches as _patches
    from matplotlib.patches import FancyArrowPatch
    import numpy as _np

    @st.cache_data(show_spinner=False)
    def _make_hero_png() -> str:
        """Return base64-encoded PNG of the water-tech hero illustration."""
        fig, ax = _plt.subplots(figsize=(5.6, 5.0), dpi=130)
        fig.patch.set_facecolor("#050E1F")
        ax.set_facecolor("#050E1F")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.axis("off")

        # ── background radial glow ──
        for r, alpha in [(4.5, .08), (3.2, .11), (2.0, .14)]:
            circ = _plt.Circle((5.5, 5.5), r, color="#00C8F0",
                               alpha=alpha, zorder=1)
            ax.add_patch(circ)
        teal_glow = _plt.Circle((3.0, 2.5), 3.0, color="#00B898",
                                alpha=.07, zorder=1)
        ax.add_patch(teal_glow)

        # ── data-network lines (background) ──
        nodes = [(1.2,8.2),(2.8,7.0),(1.5,5.5),(3.8,8.5),
                 (4.8,6.8),(3.2,5.0),(2.0,3.5),(4.2,4.0)]
        edges = [(0,1),(1,2),(0,3),(3,4),(4,5),(1,5),(2,6),(5,6),(5,7),(6,7)]
        for i,j in edges:
            x = [nodes[i][0], nodes[j][0]]
            y = [nodes[i][1], nodes[j][1]]
            ax.plot(x, y, color="#00C8F0", lw=.55, alpha=.22, zorder=2)
        for nx_, ny_ in nodes:
            ax.plot(nx_, ny_, 'o', color="#00C8F0", ms=2.8,
                    alpha=.55, zorder=3)

        # ── wave surface ──
        x_wave = _np.linspace(0, 10, 400)
        for i, (amp, freq, phase, col, alpha) in enumerate([
            (.25, 1.8, 0.0,  "#00C8F0", .22),
            (.18, 2.2, .8,   "#00B898", .18),
            (.12, 1.4, 1.6,  "#00A8CC", .14),
        ]):
            y_w = 2.5 + amp * _np.sin(freq * x_wave + phase)
            ax.fill_between(x_wave, 0, y_w,
                            color=col, alpha=alpha, zorder=4)
            ax.plot(x_wave, y_w, color=col,
                    lw=.8, alpha=alpha + .1, zorder=5)

        # ── ripple ellipses on water surface ──
        for rw, rh, al in [(1.4,.4,.45),(1.0,.28,.55),(.55,.16,.65)]:
            ell = _patches.Ellipse((5.5, 2.5), rw*2, rh*2,
                                   fill=False, edgecolor="#00C8F0",
                                   lw=.7, alpha=al, zorder=6)
            ax.add_patch(ell)

        # ── glass droplet ──
        drop_x, drop_y = 5.5, 6.2
        t = _np.linspace(0, _np.pi * 2, 300)
        # parametric teardrop shape
        scale_x, scale_y = 1.35, 1.9
        drop_px = drop_x + scale_x * _np.sin(t)
        drop_py = drop_y - scale_y + scale_y * (1 - _np.cos(t)) / 2 \
                  + 1.5 * (1 - _np.cos(t/2))
        # body gradient via scatter fill
        ax.fill(drop_px, drop_py,
                color="#0099BB", alpha=.72, zorder=7)
        # inner glow
        ax.fill(drop_px * .65 + drop_x * .35,
                drop_py * .65 + drop_y * .35,
                color="#00DFFF", alpha=.28, zorder=8)
        # outline
        ax.plot(drop_px, drop_py, color="#00E5FF",
                lw=1.1, alpha=.75, zorder=9)
        # highlight (top-left)
        hl_x = drop_x - .35
        hl_y = drop_y + .55
        hl = _patches.Ellipse((hl_x, hl_y), .28, .52,
                               angle=-28, color="white",
                               alpha=.28, zorder=10)
        ax.add_patch(hl)
        # inner data-wave lines
        for dy_off, col_w, lw_w, alp in [
            (.10, "#80EEFF", .6, .45),
            (-.15, "#00C8F0", .5, .35),
        ]:
            xw = _np.linspace(drop_x-1.1, drop_x+1.1, 100)
            yw = drop_y + dy_off + .15 * _np.sin(
                3.5 * (xw - drop_x))
            # only draw inside drop (rough clip)
            mask = _np.abs(xw - drop_x) < 1.05
            ax.plot(xw[mask], yw[mask], color=col_w,
                    lw=lw_w, alpha=alp, zorder=11)
        # impact point
        ax.plot(drop_x, 2.55, 'o', color="#00C8F0",
                ms=4, alpha=.78, zorder=12)

        # ── sensor station concentric rings (right side) ──
        for r, al, lw_, dash in [
            (.85, .18, .7, (4,3)), (.62, .26, .9, None),
            (.40, .35, 1.1, None)]:
            kw = dict(fill=False, edgecolor="#00C8F0", lw=lw_, alpha=al, zorder=5)
            if dash:
                kw["linestyle"] = (0, dash)
            c_ = _plt.Circle((8.2, 7.8), r, **kw)
            ax.add_patch(c_)
        # tick marks
        for ang in [0, 90, 180, 270]:
            rad = _np.radians(ang)
            r0, r1 = .9, 1.05
            x0 = 8.2 + r0 * _np.cos(rad)
            y0 = 7.8 + r0 * _np.sin(rad)
            x1 = 8.2 + r1 * _np.cos(rad)
            y1 = 7.8 + r1 * _np.sin(rad)
            ax.plot([x0,x1],[y0,y1], color="#00C8F0",
                    lw=1.0, alpha=.45, zorder=5)
        ax.plot(8.2, 7.8, 'o', color="#00C8F0",
                ms=3.5, alpha=.75, zorder=6)

        # ── floating bubbles ──
        for bx, by, br, al in [
            (3.2, 7.8, .16, .40), (7.5, 5.8, .12, .35),
            (2.5, 4.8, .10, .32), (8.5, 4.5, .14, .38),
            (6.5, 8.5, .10, .30), (4.5, 9.0, .08, .28),
        ]:
            bc = _plt.Circle((bx, by), br, fill=False,
                             edgecolor="#00C8F0", lw=.6,
                             alpha=al, zorder=7)
            ax.add_patch(bc)
            # inner highlight
            ib = _plt.Circle((bx - br*.3, by + br*.3), br*.35,
                             color="white", alpha=.18, zorder=8)
            ax.add_patch(ib)

        # ── "AquaGuard Sentinel AI" text watermark inside illustration ──
        ax.text(5.5, .65, "AquaGuard Sentinel AI",
                ha="center", va="center",
                fontsize=7.5, color="#00C8F0", alpha=.38,
                fontweight="bold", zorder=13)

        # ── particle dots ──
        _plt.scatter([2.2,7.1,4.0,8.8,6.2],
                     [9.2,9.4,7.5,6.5,4.2],
                     s=[8,6,5,7,5], c="#00C8F0",
                     alpha=.55, zorder=12, linewidths=0)

        buf = _io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor(), dpi=130)
        _plt.close(fig)
        buf.seek(0)
        return _b64.b64encode(buf.read()).decode()

    hero_b64 = _make_hero_png()
    hero_src  = f"data:image/png;base64,{hero_b64}"

    # ── hero layout: dark panel, two columns inside ───────────────────
    st.markdown('<div class="hero-shell">', unsafe_allow_html=True)

    col_text, col_img = st.columns([10, 9], gap="small")

    with col_text:
        st.markdown(f"""
<div style="padding:2.2rem 1.2rem 2rem 2rem">
  <div class="eyebrow">
    <div class="eyebrow-dot"></div>
    Environmental AI Research
  </div>
  <div class="hero-title">AquaGuard</div>
  <div class="hero-title">
    <span class="hero-title-accent">Sentinel AI</span>
  </div>
  <div class="hero-tagline">
    Anomaly detection for water-quality sensor measurements.
  </div>
  <p class="hero-desc">
    A reproducible, open-data research platform for comparing
    anomaly-detection baselines on environmental sensor data &#8212;
    built for transparency, scientific honesty, and educational
    exploration.
  </p>
  <div class="badge-row">
    <span class="hero-badge">&#9672; 3 Detection Methods</span>
    <span class="hero-badge">&#9672; 4 Public Datasets</span>
    <span class="hero-badge">&#9672; Reproducible Experiments</span>
    <span class="hero-badge">&#9672; Conformal Calibration</span>
    <span class="hero-badge">&#9672; 12 Automated Tests</span>
  </div>
  <div class="cta-row">
    <span class="cta-primary">&#9654; Use the sidebar to run</span>
    <span class="cta-secondary">&#8594; Overview</span>
  </div>
  <div class="feat-row">
    <div class="feat-item">
      <div class="feat-dot"></div><span>Real public sensor data</span>
    </div>
    <div class="feat-item">
      <div class="feat-dot"></div><span>Synthetic fault injection</span>
    </div>
    <div class="feat-item">
      <div class="feat-dot"></div><span>No fabricated metrics</span>
    </div>
    <div class="feat-item">
      <div class="feat-dot"></div><span>Full scientific disclaimers</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    with col_img:
        st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
        # Ripple animation rings (CSS only, no SVG in markdown)
        st.markdown(f"""
<div style="position:relative;display:inline-block">
  <div class="ripple-ring"
       style="width:260px;height:260px;top:50%;left:50%;
              transform:translate(-50%,-50%)">
  </div>
  <div class="ripple-ring"
       style="width:260px;height:260px;top:50%;left:50%;
              transform:translate(-50%,-50%)">
  </div>
  <div class="ripple-ring"
       style="width:260px;height:260px;top:50%;left:50%;
              transform:translate(-50%,-50%)">
  </div>
  <div class="bubble"
       style="width:14px;height:14px;top:18px;right:24px"></div>
  <div class="bubble"
       style="width:9px;height:9px;top:55px;right:8px"></div>
  <div class="bubble"
       style="width:11px;height:11px;top:82px;left:12px"></div>
  <div class="bubble"
       style="width:7px;height:7px;top:140px;right:6px"></div>
</div>
""", unsafe_allow_html=True)
        # The actual illustration — rendered via st.image(), guaranteed to show
        st.image(hero_src, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # wave bar at bottom of hero panel
    st.markdown('<div class="hero-wave-bar"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # close hero-shell

    # ── disclaimer ────────────────────────────────────────────────────
    _disc()

    # ── get started callout ───────────────────────────────────────────
    st.markdown(f"""
<div style="background:rgba(0,200,240,.05);border:1px solid {BOR};
            border-radius:10px;padding:1rem 1.2rem;margin-bottom:1.2rem">
  <div style="font-size:.85rem;color:{W0};font-weight:600;margin-bottom:.28rem">
    &#9654; Get started
  </div>
  <div style="font-size:.81rem;color:{W1};line-height:1.65">
    Select a dataset in the left sidebar and click
    <strong style="color:{C0}">Run Transparent Comparison</strong>
    to run the full anomaly-detection benchmark.
    Results populate <strong>Overview</strong>,
    <strong>Anomaly Explorer</strong>, <strong>Model Comparison</strong>,
    and <strong>Results &amp; Reports</strong>.
    The <strong>Synthetic demo</strong> works offline and completes in seconds.
  </div>
</div>""", unsafe_allow_html=True)

    # ── three method cards ────────────────────────────────────────────
    _sec("Three Detection Methods", "&#128200;")
    mc1, mc2, mc3 = st.columns(3)
    for col_, border_c, title_, desc_, metric_ in [
        (mc1, C0, "Robust Global",
         "Scores each row by its maximum median/IQR-scaled deviation across all "
         "sensor features. Simple, interpretable, and robust to outliers in "
         "individual channels. One highly variable feature can dominate the score.",
         f'<span style="color:{C0}">&#8635; Max IQR deviation score</span>'),
        (mc2, T0, "Isolation Forest",
         "Isolation-based anomaly detection (scikit-learn, 180 trees, seed=42). "
         "Effective for multi-dimensional outliers. Less sensitive to isolated "
         "single-feature spikes on high-dimensional datasets.",
         f'<span style="color:{T0}">&#8635; Isolation score</span>'),
        (mc3, PRP, "Sentinel Station-Aware",
         "Per-station robust scoring shrunk toward a global reference for stability. "
         "Conformal p-values from site-specific calibration sets. "
         "Matches Robust Global when only one station is available.",
         f'<span style="color:{PRP}">&#8635; Conformal p-value (&#945;=0.05)</span>'),
    ]:
        with col_:
            st.markdown(f"""
<div style="background:linear-gradient(145deg,{N2},{N1});
            border:1px solid {BOR};border-top:2px solid {border_c};
            border-radius:12px;padding:1rem 1.1rem">
  <div style="font-size:.83rem;font-weight:700;color:{W0};
              margin-bottom:.3rem">{title_}</div>
  <div style="font-size:.77rem;color:{W1};line-height:1.55;
              margin-bottom:.5rem">{desc_}</div>
  <div style="font-size:.72rem">{metric_}</div>
</div>""", unsafe_allow_html=True)

    # ── dataset quick-reference ───────────────────────────────────────
    _sec("Available Datasets", "&#128196;")
    st.markdown(f"""
| Dataset | Records | Sensor Features | Domain |
|---|---|---|---|
| **Synthetic demo** | 900 | 4 (sensor_a, sensor_b, sensor_c, sensor_d) | Artificial |
| **UCI Wastewater** | 527 | 38 (Q-E, PH-E, DBO-E&#8230;) | Wastewater treatment |
| **NYC Drinking Water** | Variable | 3 chemistry columns | Drinking water |
| **USGS River/Estuary** | Variable | Up to 5 columns | River monitoring |
| **Custom CSV** | &#8805;150 | Any &#8805;2 numeric | User-supplied |

&#9888; NYC and USGS require internet. UCI auto-downloads on first use.
""")

    # ── scientific scope ──────────────────────────────────────────────
    _sec("Scientific Scope", "&#128218;")
    st.markdown(f"""
<div style="background:rgba(0,200,240,.04);border:1px solid {BOR};
            border-radius:10px;padding:1rem 1.2rem;
            font-size:.82rem;color:{W1};line-height:1.7">
  <strong style="color:{W0}">What this project does:</strong>
  Flags sensor readings that are statistically unusual compared to the
  training distribution using three unsupervised baselines.<br><br>
  <strong style="color:{W0}">What it does NOT do:</strong>
  It does not detect real contamination, certify water safety,
  provide real-time monitoring, or compute precision/recall/F1/ROC-AUC
  (no verified ground-truth contamination labels exist).<br><br>
  <strong style="color:{W0}">All stress tests</strong> inject artificial sensor faults
  in software &#8212; not real pollution incidents.
</div>""", unsafe_allow_html=True)

    _footer()

# ╔═══════════════════════════════════════════════════════════════════════
# ║  PAGE 1 — OVERVIEW
# ╚═══════════════════════════════════════════════════════════════════════
elif page == "Overview":
    st.markdown(f"""
<div class="pg-header">
  <h1>&#128167; AquaGuard Sentinel AI</h1>
  <p>Open-data anomaly detection for water-quality sensor measurements &#183;
     Python &#183; scikit-learn &#183; Streamlit &#183;
     Reproducible experiments</p>
</div>""", unsafe_allow_html=True)
    _disc()

    # Project at a glance
    _sec("What This Project Does", "&#128270;")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"""
AquaGuard Sentinel is a **research and education platform** for exploring
unusual patterns in water-quality sensor data. It compares three anomaly-detection
baselines on public datasets using a fully reproducible 60/20/20 evaluation protocol.

<div class="stat-row" style="margin-top:.8rem">
  <div class="stat">
    <div class="stat-lbl">Detection Methods</div>
    <div class="stat-val">3</div>
  </div>
  <div class="stat">
    <div class="stat-lbl">Public Datasets</div>
    <div class="stat-val">4</div>
  </div>
  <div class="stat">
    <div class="stat-lbl">Stress-Test Types</div>
    <div class="stat-val">4</div>
  </div>
  <div class="stat">
    <div class="stat-lbl">Automated Tests</div>
    <div class="stat-val">12</div>
  </div>
</div>
""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
<div class="chart-panel" style="padding:.9rem 1rem">
  <div class="chart-panel-title">Detection Methods</div>
  <div style="margin-top:.5rem">
    <div class="method-pill">&#128200; Robust Global</div>
    <div class="method-pill">&#127807; Isolation Forest</div>
    <div class="method-pill">&#128268; Sentinel Station-Aware</div>
  </div>
  <div class="chart-panel-title" style="margin-top:.8rem">Available Datasets</div>
  <div style="margin-top:.5rem;font-size:.78rem;color:{W1};line-height:1.7">
    &#8226; Synthetic demo (offline, instant)<br>
    &#8226; UCI Wastewater &#8212; 527 records, 38 features<br>
    &#8226; NYC Drinking Water (Open Data API)<br>
    &#8226; USGS River/Estuary (NWIS API)
  </div>
</div>""", unsafe_allow_html=True)

    # Evaluation workflow
    _sec("Evaluation Workflow", "&#9654;")
    st.markdown(f"""
<div class="workflow">
  <div class="wf-step">
    <div class="wf-step-num">01</div>
    <div class="wf-step-title">Select Dataset</div>
    <div class="wf-step-desc">Public UCI wastewater, NYC drinking water,
    USGS river data, synthetic demo, or your own CSV</div>
  </div>
  <div class="wf-arrow">&#8250;</div>
  <div class="wf-step">
    <div class="wf-step-num">02</div>
    <div class="wf-step-title">Train (60%)</div>
    <div class="wf-step-desc">Fit anomaly models on training split only.
    No test data seen during training.</div>
  </div>
  <div class="wf-arrow">&#8250;</div>
  <div class="wf-step">
    <div class="wf-step-num">03</div>
    <div class="wf-step-title">Calibrate (20%)</div>
    <div class="wf-step-desc">Set conformal alert thresholds from
    held-out calibration scores (&#945;=0.05).</div>
  </div>
  <div class="wf-arrow">&#8250;</div>
  <div class="wf-step">
    <div class="wf-step-num">04</div>
    <div class="wf-step-title">Evaluate (20%)</div>
    <div class="wf-step-desc">Score unseen held-out rows.
    Inject artificial faults. Measure alert rates.</div>
  </div>
  <div class="wf-arrow">&#8250;</div>
  <div class="wf-step">
    <div class="wf-step-num">05</div>
    <div class="wf-step-title">Compare</div>
    <div class="wf-step-desc">Report alert rates per method and
    stress type. Export CSV/JSON.</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Results summary (if run)
    if run_done and manifest and summary is not None:
        _sec("Experiment Results Summary", "&#128202;")
        _data_badge()
        _metric_cards()

        # Quick 3-method alert rate chart
        base = (summary[summary.stress == "untouched_heldout"]
                .set_index("method")["mean_alert_rate"].reindex(
                    ["isolation_forest", "robust_global", "sentinel_station"]))
        spike3 = (summary[(summary.stress == "artificial_spike") &
                           (summary.strength_iqr == 3.0)]
                  .set_index("method")["mean_new_alert_rate"].reindex(
                      ["isolation_forest", "robust_global", "sentinel_station"]))

        if HAS_PLOTLY and not base.dropna().empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
<div class="chart-panel">
  <div class="chart-panel-title">Untouched Holdout Alert Rates</div>
  <div class="chart-panel-sub">Fraction of real test rows flagged &#8212;
  not a false-positive rate (no ground-truth labels)</div>""",
                    unsafe_allow_html=True)
                fig = go.Figure(go.Bar(
                    x=[METHOD_LABELS.get(m, m) for m in base.index],
                    y=base.fillna(0).tolist(),
                    marker_color=[METHOD_COLORS.get(m, C0) for m in base.index],
                    marker_line_width=0, text=[f"{v:.1%}" for v in base.fillna(0)],
                    textposition="outside", textfont=dict(size=10, color=W0),
                ))
                fig.update_layout(**_plo(180))
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
<div class="chart-panel">
  <div class="chart-panel-title">New-Alert Rate at 3&#215;IQR Spike</div>
  <div class="chart-panel-sub">Rows newly flagged after injecting artificial
  single-feature spikes &#8212; stress-test sensitivity, not contamination accuracy</div>""",
                    unsafe_allow_html=True)
                fig2 = go.Figure(go.Bar(
                    x=[METHOD_LABELS.get(m, m) for m in spike3.index],
                    y=spike3.fillna(0).tolist(),
                    marker_color=[METHOD_COLORS.get(m, C0) for m in spike3.index],
                    marker_line_width=0,
                    text=[f"{v:.1%}" for v in spike3.fillna(0)],
                    textposition="outside", textfont=dict(size=10, color=W0),
                ))
                fig2.update_layout(**_plo(180))
                fig2.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig2, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

        # Key limitations
        _sec("Scientific Limitations", "&#9888;&#65039;")
        _lims()

    else:
        # Before run: show pre-computed UCI results as context
        _sec("Pre-Computed UCI Wastewater Results (for context)", "&#128203;")
        st.markdown(f"""
<div class="chart-panel">
  <div class="chart-panel-sub">
  Results from a reproducible run on the UCI Water Treatment Plant dataset
  (527 historical records, 38 sensor features, one wastewater facility, 106 held-out rows).
  These are <strong>fixed benchmark values</strong> &#8212; run the experiment yourself to
  reproduce or explore other datasets.
  </div>

| Method | Holdout alert rate | New-alert rate at 5&#215;IQR | Note |
|---|---|---|---|
| Isolation Forest | 2.83 % | **0.00 %** | Not sensitive to isolated single-feature spikes |
| Robust Global | 1.89 % | **0.95 %** | Detects some injected spikes |
| Sentinel Station | 1.89 % | **0.95 %** | Same as Robust Global with only one station |

&#9888;&#65039; **These are weak results &#8212; shown transparently.**
Alert rate &#8800; false-positive rate. Spike rate &#8800; contamination sensitivity.
Real fault labels do not exist.
</div>""", unsafe_allow_html=True)

    _footer()


# ╔═══════════════════════════════════════════════════════════════════════
# ║  PAGE 2 — DATASETS & QUALITY
# ╚═══════════════════════════════════════════════════════════════════════
elif page == "Datasets & Quality":
    st.markdown(f"""
<div class="pg-header">
  <h1>Datasets &amp; Quality</h1>
  <p>Available public datasets, record counts, feature schemas, split sizes,
     data completeness, and interactive exploration</p>
</div>""", unsafe_allow_html=True)
    _disc()

    # Dataset catalogue (always shown)
    _sec("Available Datasets", "&#128196;")
    st.markdown(f"""
| Dataset | Domain | Records | Features | Source |
|---|---|---|---|---|
| **Synthetic demo** | Artificial | 900 | 4 sensors (sensor_a, sensor_b, sensor_c, sensor_d) | Local, fixed seed |
| **UCI Wastewater** | Wastewater treatment | 527 | 38 process measurements (Q-E, PH-E, DBO-E&#8230;) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/106/water+treatment+plant) CC BY 4.0 |
| **NYC Drinking Water** | Drinking water | Variable | residual_free_chlorine_mg_l, turbidity_ntu, fluoride_mg_l | [NYC Open Data API](https://data.cityofnewyork.us) |
| **USGS River/Estuary** | River monitoring | Variable | temperature_c, conductivity_us_cm, dissolved_oxygen_mg_l, ph, turbidity_fnu | [USGS NWIS API](https://waterservices.usgs.gov) |
| **Custom CSV** | User-supplied | &#8805;150 | Any &#8805;2 numeric columns | User upload |

**Split protocol:** 60% training &#183; 20% calibration &#183; 20% held-out test,
ordered by timestamp when available, otherwise source record order.
""")

    if not run_done or data is None:
        _no_run("Datasets & Quality")
    else:
        _data_badge()
        _sec("This Experiment's Dataset", "&#128200;")
        _metric_cards()

        feat_cols = [c for c in manifest["features"] if c in data.columns]
        non_feat  = [c for c in data.columns if c not in manifest["features"]]

        # Feature schema
        _sec("Sensor Features Used", "&#128225;")
        st.markdown(f"**{len(feat_cols)} numeric measurement columns** selected "
                    f"by `select_features()` (≥40% observed, ≥6 distinct values):")
        # Show as code block — real names, no invention
        st.code(", ".join(feat_cols) if feat_cols else "None", language=None)
        if non_feat:
            st.caption(f"Non-feature columns (metadata): {', '.join(non_feat[:8])}"
                       + (" …" if len(non_feat) > 8 else ""))

        # Data quality
        _sec("Data Completeness", "&#9989;")
        if feat_cols:
            num = data[feat_cols].apply(pd.to_numeric, errors="coerce")
            tc  = num.size
            obs = int(num.notna().sum().sum())
            miss = tc - obs
            comp = obs / tc if tc else 1.0
            miss_f = miss / tc if tc else 0.0
            finite_frac = (float(np.isfinite(num.values[num.notna().values]).mean())
                           if obs > 0 else 1.0)
            overall = (comp * finite_frac) ** 0.5

            q1, q2, q3, q4 = st.columns(4)
            for col_, v_, l_ in [
                (q1, f"{comp:.1%}",    "Completeness"),
                (q2, f"{finite_frac:.1%}", "Valid Range"),
                (q3, f"{miss_f:.1%}", "Missing Values"),
                (q4, f"{overall:.1%}", "Overall Quality"),
            ]:
                col_.markdown(f"""
<div class="dq-tile">
  <div class="dq-val">{v_}</div>
  <div class="dq-lbl">{l_}</div>
</div>""", unsafe_allow_html=True)

            st.caption(f"Computed on all {len(data):,} records and "
                       f"{len(feat_cols)} feature columns. "
                       "Not feature importance — these are raw measurement quality indicators.")

            # Per-feature completeness
            feat_comp = num.notna().mean().sort_values()
            if HAS_PLOTLY:
                colors = [T0 if v >= .9 else (AMB if v >= .7 else RED)
                          for v in feat_comp.values]
                fig = go.Figure(go.Bar(
                    x=feat_comp.values.tolist(),
                    y=feat_comp.index.tolist(),
                    orientation="h",
                    marker_color=colors, marker_line_width=0,
                ))
                layout = _plo(max(180, len(feat_comp) * 14))
                layout.update(margin=dict(l=0, r=5, t=4, b=0))
                fig.update_layout(**layout)
                fig.update_xaxes(tickformat=".0%", range=[0, 1.05],
                                 title_text="Completeness", title_font_size=9)
                with st.expander("Per-feature completeness chart", expanded=True):
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})
                    st.markdown(f"""
<div style="display:flex;gap:1rem;font-size:.71rem;color:{W1};margin-top:.3rem">
  <span>&#9632; <span style="color:{T0}">≥90%</span></span>
  <span>&#9632; <span style="color:{AMB}">70–90%</span></span>
  <span>&#9632; <span style="color:{RED}">&lt;70%</span></span>
</div>""", unsafe_allow_html=True)

        # Interactive data explorer
        _sec("Interactive Data Explorer", "&#128270;")
        if feat_cols:
            sel = st.selectbox("Select a sensor feature to explore",
                               feat_cols, key="dq_feat")
            fv = pd.to_numeric(data[sel], errors="coerce")
            ec1, ec2, ec3, ec4, ec5 = st.columns(5)
            ec1.metric("Observed",  f"{fv.notna().sum():,}")
            ec2.metric("Missing",   f"{fv.isna().sum():,}")
            ec3.metric("Mean",      f"{fv.mean():.4g}" if fv.notna().any() else "—")
            ec4.metric("Std",       f"{fv.std():.4g}"  if fv.notna().any() else "—")
            ec5.metric("Range",     f"{fv.min():.3g}–{fv.max():.3g}"
                                     if fv.notna().any() else "—")
            step = max(1, len(fv) // 600)
            fvs  = fv.iloc[::step].reset_index(drop=True)
            if HAS_PLOTLY:
                fig = go.Figure(go.Scatter(
                    x=list(range(len(fvs))), y=fvs.tolist(),
                    mode="lines", name=sel,
                    line=dict(color=C0, width=1.3), opacity=.88))
                fig.update_layout(**_plo(220, dict(l=0, r=0, t=4, b=0)))
                fig.update_yaxes(title_text=sel, title_font_size=9)
                fig.update_xaxes(title_text="Record index", title_font_size=9)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.line_chart(pd.DataFrame({sel: fvs}),
                              use_container_width=True, height=220, color=C0)

        # Raw data preview
        _sec("Data Table (first 200 rows)", "&#128196;")
        preview_cols = feat_cols[:8] if feat_cols else list(data.columns)[:8]
        preview = data[preview_cols].head(200).copy()
        for c in preview_cols:
            preview[c] = pd.to_numeric(preview[c], errors="coerce")
        st.dataframe(preview.round(4), use_container_width=True, hide_index=True)
        st.caption(f"Showing {min(200,len(data))} of {len(data):,} rows · "
                   f"{len(preview_cols)} of {len(data.columns)} columns displayed")

    _footer()


# ╔═══════════════════════════════════════════════════════════════════════
# ║  PAGE 3 — ANOMALY EXPLORER
# ╚═══════════════════════════════════════════════════════════════════════
elif page == "Anomaly Explorer":
    st.markdown(f"""
<div class="pg-header">
  <h1>Anomaly Explorer</h1>
  <p>Anomaly scores, conformal p-values, alert thresholds, and per-feature visualisations
     for the held-out test set</p>
</div>""", unsafe_allow_html=True)
    _disc()

    if not run_done or not method_out:
        _no_run("the Anomaly Explorer")
    else:
        _data_badge()

        st.markdown(f"""
<div class="chart-panel" style="margin-bottom:.8rem">
  <div class="chart-panel-sub">
  An <strong>anomaly alert</strong> means a sensor reading is statistically
  unusual compared to the training distribution at &#945; = 0.05.
  It does <em>not</em> mean the water is contaminated or unsafe.
  No ground-truth contamination labels exist in these datasets.
  </div>
</div>""", unsafe_allow_html=True)

        method_sel = st.selectbox(
            "Detection method",
            list(method_out.keys()),
            format_func=lambda m: METHOD_LABELS.get(m, m),
            key="ae_method")

        mo = method_out[method_sel]
        alerts_df = mo["alerts"]
        scores    = mo["scores"]
        pvals     = mo["pvals"]

        al    = alerts_df["anomaly_alert"].values
        insuf = alerts_df["sensor_data_insufficient"].values
        pv    = pvals if pvals is not None else np.full(len(al), np.nan)

        n_total = len(al)
        n_insuf = int(insuf.sum())
        n_obs   = n_total - n_insuf
        n_alert = int(al.sum())

        st.markdown(f"""
<div class="stat-row">
  <div class="stat"><div class="stat-lbl">Method</div>
    <div class="stat-val" style="font-size:.85rem">
    {METHOD_LABELS.get(method_sel, method_sel)}</div></div>
  <div class="stat"><div class="stat-lbl">Test Rows</div>
    <div class="stat-val">{n_total:,}</div></div>
  <div class="stat"><div class="stat-lbl">Rows Evaluated</div>
    <div class="stat-val">{n_obs:,}</div></div>
  <div class="stat"><div class="stat-lbl">Anomaly Alerts</div>
    <div class="stat-val" style="color:{RED}">{n_alert:,}</div></div>
  <div class="stat"><div class="stat-lbl">Alert Rate</div>
    <div class="stat-val">{(al.mean()*100):.2f}%</div></div>
  <div class="stat"><div class="stat-lbl">Insufficient Rows</div>
    <div class="stat-val">{n_insuf:,}</div></div>
</div>""", unsafe_allow_html=True)
        st.caption(f"Alert threshold: &#945; = {manifest['alpha']} &#183; "
                   "Abstains when < 50% of sensor channels are observed")

        # Anomaly score scatter
        _sec("Anomaly Score Timeline", "&#128200;")
        if scores is not None:
            valid  = np.isfinite(scores)
            sv     = scores[valid]
            al_v   = al[valid]
            step   = max(1, len(sv) // 600)
            sv_s   = sv[::step]
            al_s   = al_v[::step]
            cal_sc = (mdl.calibration.get(method_sel, np.array([]))
                      if mdl else np.array([]))
            thresh = float(np.percentile(cal_sc, 95)) if len(cal_sc) > 10 else None

            if HAS_PLOTLY:
                fig = go.Figure()
                norm_i  = [i for i,f in enumerate(al_s) if not f]
                alert_i = [i for i,f in enumerate(al_s) if f]
                fig.add_trace(go.Scatter(
                    x=norm_i, y=[sv_s[i] for i in norm_i],
                    mode="markers", name="Normal reading",
                    marker=dict(color=T0, size=3, opacity=.65)))
                if alert_i:
                    fig.add_trace(go.Scatter(
                        x=alert_i, y=[sv_s[i] for i in alert_i],
                        mode="markers", name="Anomaly alert",
                        marker=dict(color=RED, size=5, opacity=.9)))
                if thresh:
                    fig.add_hline(y=thresh, line_dash="dash",
                                  line_color=AMB, line_width=1.5,
                                  annotation_text=f"Threshold ({thresh:.3f})",
                                  annotation_font_size=9, annotation_font_color=AMB)
                layout = _plo(260, dict(l=0, r=0, t=4, b=0))
                layout["legend"] = dict(orientation="h", y=-0.2, x=0,
                                        font=dict(size=9, color=W1),
                                        bgcolor="rgba(0,0,0,0)")
                fig.update_layout(**layout)
                fig.update_yaxes(title_text="Anomaly score", title_font_size=9)
                fig.update_xaxes(title_text="Test record index", title_font_size=9)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.line_chart(pd.DataFrame({"score": sv_s}),
                              use_container_width=True, height=260, color=C0)

            st.caption("Higher score = more unusual relative to training distribution. "
                       "Threshold = 95th percentile of calibration-set scores. "
                       "Unusual ≠ contaminated.")

        # P-value distribution
        _sec("Conformal P-value Distribution", "&#128200;")
        finite_pv = pv[np.isfinite(pv)]
        if len(finite_pv) > 0 and HAS_PLOTLY:
            fig_pv = go.Figure(go.Histogram(
                x=finite_pv.tolist(), nbinsx=40,
                marker_color=C0, marker_line_width=0,
                opacity=.85, name="p-values"))
            fig_pv.add_vline(x=0.05, line_dash="dash", line_color=AMB,
                             line_width=1.5,
                             annotation_text="α = 0.05",
                             annotation_font_size=9,
                             annotation_font_color=AMB)
            layout_pv = _plo(200, dict(l=0, r=0, t=4, b=0))
            fig_pv.update_layout(**layout_pv)
            fig_pv.update_xaxes(title_text="Conformal p-value (0–1)", title_font_size=9)
            fig_pv.update_yaxes(title_text="Count", title_font_size=9)
            st.plotly_chart(fig_pv, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption("Rows with p-value ≤ α receive an anomaly alert. "
                       "Under exchangeability this controls the false-alert rate at 5%, "
                       "but exchangeability is not guaranteed for time-series data.")

        # Per-feature score explorer
        _sec("Sensor Feature Explorer", "&#128225;")
        if feat_list and test_df is not None:
            sel_f = st.selectbox("Select feature to overlay with alerts",
                                 feat_list, key="ae_feat")
            fv = pd.to_numeric(
                test_df[sel_f] if sel_f in test_df.columns
                else pd.Series(dtype=float), errors="coerce")
            step = max(1, len(fv) // 600)
            fvs  = fv.iloc[::step].reset_index(drop=True)
            al_s2 = al[::step]

            if HAS_PLOTLY:
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(
                    x=list(range(len(fvs))), y=fvs.tolist(),
                    mode="lines", name=sel_f,
                    line=dict(color=C0, width=1.3), opacity=.85))
                al_i = [i for i in range(len(fvs))
                        if i < len(al_s2) and al_s2[i]
                        and np.isfinite(fvs.iloc[i])]
                if al_i:
                    fig5.add_trace(go.Scatter(
                        x=al_i, y=[fvs.iloc[i] for i in al_i],
                        mode="markers", name="Alert",
                        marker=dict(color=RED, size=6)))
                layout5 = _plo(240, dict(l=0, r=0, t=4, b=0))
                layout5["legend"] = dict(orientation="h", y=-0.22, x=0,
                                         font=dict(size=9, color=W1),
                                         bgcolor="rgba(0,0,0,0)")
                fig5.update_layout(**layout5)
                fig5.update_yaxes(title_text=sel_f, title_font_size=9)
                fig5.update_xaxes(title_text="Test record index", title_font_size=9)
                st.plotly_chart(fig5, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.line_chart(pd.DataFrame({sel_f: fvs.values}),
                              use_container_width=True, height=240, color=C0)

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Observed readings", f"{fv.notna().sum():,}")
            sc2.metric("Missing readings",  f"{fv.isna().sum():,}")
            sc3.metric("Alerts on this feature's rows", f"{int(al.sum()):,}")
            sc4.metric("Alert rate", f"{al.mean():.1%}")

    _footer()


# ╔═══════════════════════════════════════════════════════════════════════
# ║  PAGE 4 — MODEL COMPARISON
# ╚═══════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown(f"""
<div class="pg-header">
  <h1>Model Comparison</h1>
  <p>Side-by-side evaluation of the three detection approaches on identical
     held-out data &#8212; using only metrics computable without ground-truth labels</p>
</div>""", unsafe_allow_html=True)
    _disc()

    # Always show what metrics ARE and are NOT valid
    st.markdown(f"""
<div class="chart-panel" style="margin-bottom:.9rem">
  <div class="chart-panel-title">About the Available Metrics</div>
  <div class="chart-panel-sub">
  Because no verified contamination-event labels exist, this project reports only
  metrics that are computable without ground truth.
  </div>
  <div style="margin-top:.6rem">
  <div style="margin-bottom:.35rem;font-size:.82rem">
    <span style="color:{T0}">&#9745;</span>
    <strong style="color:{W0}"> Computed from real data:</strong>
    <span style="color:{W1}"> Mean alert rate on untouched holdout data &#183;
    New-alert rate on artificially injected spikes/drift/dropout &#183;
    Insufficient-readings rate &#183; Split sizes</span>
  </div>
  <div style="font-size:.82rem">
    <span style="color:{RED}">&#9746;</span>
    <strong style="color:{W0}"> Not available (no ground-truth labels):</strong>
    <span style="color:{W1}"> Precision &#183; Recall &#183; F1 &#183;
    ROC-AUC &#183; Feature importance &#183; True positive / false positive rates</span>
  </div>
  </div>
</div>""", unsafe_allow_html=True)

    if not run_done or summary is None:
        _no_run("Model Comparison")
    else:
        _data_badge()

        METHODS_ORD = ["isolation_forest", "robust_global", "sentinel_station"]

        # Untouched holdout + spike at 3x IQR
        _sec("Alert Rates on Held-out Data", "&#128200;")
        base = (summary[summary.stress == "untouched_heldout"]
                .set_index("method")["mean_alert_rate"]
                .reindex(METHODS_ORD))
        sp3  = (summary[(summary.stress == "artificial_spike") &
                        (summary.strength_iqr == 3.0)]
                .set_index("method")["mean_new_alert_rate"]
                .reindex(METHODS_ORD))

        if HAS_PLOTLY:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
<div class="chart-panel">
  <div class="chart-panel-title">Untouched Holdout Alert Rate</div>
  <div class="chart-panel-sub">Fraction of real test rows flagged per method
  (baseline — not a false-positive rate)</div>""", unsafe_allow_html=True)
                fig = go.Figure(go.Bar(
                    x=[METHOD_LABELS.get(m, m) for m in METHODS_ORD],
                    y=base.fillna(0).tolist(),
                    marker_color=[METHOD_COLORS.get(m, C0) for m in METHODS_ORD],
                    marker_line_width=0,
                    text=[f"{v:.2%}" for v in base.fillna(0)],
                    textposition="outside", textfont=dict(size=10, color=W0)))
                fig.update_layout(**_plo(190))
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
<div class="chart-panel">
  <div class="chart-panel-title">New-Alert Rate at 3&#215;IQR Artificial Spike</div>
  <div class="chart-panel-sub">Rows newly flagged after injecting
  one-feature spikes at 3&#215; IQR (5 seeds averaged)</div>""", unsafe_allow_html=True)
                fig2 = go.Figure(go.Bar(
                    x=[METHOD_LABELS.get(m, m) for m in METHODS_ORD],
                    y=sp3.fillna(0).tolist(),
                    marker_color=[METHOD_COLORS.get(m, C0) for m in METHODS_ORD],
                    marker_line_width=0,
                    text=[f"{v:.2%}" for v in sp3.fillna(0)],
                    textposition="outside", textfont=dict(size=10, color=W0)))
                fig2.update_layout(**_plo(190))
                fig2.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig2, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

        # All stress types grouped
        _sec("All Stress Types", "&#9889;")
        stress_order = ["untouched_heldout", "artificial_spike",
                        "artificial_gradual_drift", "artificial_sensor_dropout"]
        ap = (summary[summary.stress.isin(stress_order)]
              .groupby(["method", "stress"])["mean_alert_rate"]
              .mean().unstack("stress")
              .reindex(index=METHODS_ORD, columns=stress_order, fill_value=0))
        ap.columns = [STRESS_LABELS.get(c, c) for c in ap.columns]

        if HAS_PLOTLY:
            fig3 = go.Figure()
            for col, clr in zip(ap.columns, STRESS_COLORS):
                fig3.add_trace(go.Bar(
                    name=col,
                    x=[METHOD_LABELS.get(m, m) for m in ap.index],
                    y=ap[col].tolist(),
                    marker_color=clr, marker_line_width=0))
            layout3 = _plo(240, dict(l=0, r=0, t=4, b=0))
            layout3["barmode"] = "group"
            layout3["legend"]["orientation"] = "h"
            layout3["legend"]["y"] = -0.22
            fig3.update_layout(**layout3)
            fig3.update_yaxes(tickformat=".0%",
                              title_text="Mean alert rate", title_font_size=9)
            st.plotly_chart(fig3, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.bar_chart(ap, use_container_width=True)
        st.caption("Alert rate ≠ contamination detection accuracy. "
                   "All spikes, drift, and dropout are artificially injected.")

        # Full table
        _sec("Complete Benchmark Table", "&#128203;")
        disp_cols = ["method", "stress", "strength_iqr", "trials",
                     "evaluated_rows_each_trial", "mean_alert_rate",
                     "mean_new_alert_rate", "mean_insufficient_readings_rate"]

        def _sr(row):
            s = row.get("stress", ""); q = row.get("strength_iqr", 0)
            if s == "untouched_heldout":
                return [f"background-color:rgba(0,184,152,.08)"] * len(row)
            if s == "artificial_spike" and q == 3.0:
                return [f"background-color:rgba(0,200,240,.07)"] * len(row)
            return [""] * len(row)

        styled = (summary[disp_cols].style.apply(_sr, axis=1)
                  .format({"mean_alert_rate": "{:.2%}",
                           "mean_new_alert_rate": "{:.2%}",
                           "mean_insufficient_readings_rate": "{:.2%}",
                           "strength_iqr": "{:.1f}"}, na_rep="—"))
        st.dataframe(styled, use_container_width=True, hide_index=True)

    _footer()


# ╔═══════════════════════════════════════════════════════════════════════
# ║  PAGE 5 — STRESS-TEST LAB
# ╚═══════════════════════════════════════════════════════════════════════
elif page == "Stress-Test Lab":
    st.markdown(f"""
<div class="pg-header">
  <h1>Stress-Test Lab</h1>
  <p>Explore synthetic sensor-fault experiments &#8212; what disturbances are injected,
     how each method responds, and what the results actually mean</p>
</div>""", unsafe_allow_html=True)
    _disc()

    # Always explain the experiment design
    _sec("How the Stress Tests Work", "&#128270;")
    st.markdown(f"""
<div class="chart-panel">
  <div class="chart-panel-sub">
  All disturbances are <strong>injected in software</strong> into the held-out test rows.
  The original measurements are not altered. No real contamination events are modelled.
  </div>
</div>""", unsafe_allow_html=True)

    cols = st.columns(4)
    STRESS_INFO = [
        ("Artificial Spike", AMB, "artificial_spike",
         "A single randomly chosen sensor value is shifted by 1.5, 3.0, or 5.0 × its IQR. "
         "Simulates a sudden sensor malfunction. "
         "5 independent random seeds per strength level. "
         "<strong>New-alert rate</strong> measures how often the model flags rows "
         "it wouldn't have flagged on clean data."),
        ("Gradual Drift", PRP, "artificial_gradual_drift",
         "One sensor channel ramps linearly by 2 × IQR over the entire holdout period. "
         "Simulates slow sensor degradation or seasonal shift. "
         "The <strong>alert rate</strong> shows whether the model catches the trend."),
        ("Sensor Dropout", C0, "artificial_sensor_dropout",
         "65% of feature values are set to NaN for 20% of holdout rows. "
         "Simulates communication failure or instrument loss. "
         "Rows with fewer than 50% observed channels receive an "
         "<strong>abstention</strong> — no alert is issued."),
        ("Untouched Baseline", T0, "untouched_heldout",
         "No disturbance is injected. The model scores the real held-out records. "
         "The alert rate is the baseline fraction of historical readings flagged. "
         "This is <strong>not</strong> a false-positive rate — no labels exist."),
    ]
    for col, (name, clr, _, desc) in zip(cols, STRESS_INFO):
        with col:
            st.markdown(f"""
<div class="chart-panel" style="border-top:2px solid {clr};padding:.85rem .9rem">
  <div class="chart-panel-title" style="color:{clr}">{name}</div>
  <div style="font-size:.78rem;color:{W1};line-height:1.55;margin-top:.4rem">{desc}</div>
</div>""", unsafe_allow_html=True)

    if not run_done or summary is None:
        _no_run("the Stress-Test Lab")
    else:
        _data_badge()
        METHODS_ORD = ["isolation_forest", "robust_global", "sentinel_station"]

        # Spike strength sweep
        _sec("Spike Strength Sensitivity", "&#128200;")
        all_sp = (summary[summary.stress == "artificial_spike"]
                  .groupby(["method", "strength_iqr"])["mean_new_alert_rate"]
                  .mean().reset_index())

        if HAS_PLOTLY and not all_sp.empty:
            fig = go.Figure()
            for m in METHODS_ORD:
                sub = all_sp[all_sp.method == m].sort_values("strength_iqr")
                fig.add_trace(go.Scatter(
                    x=sub["strength_iqr"].tolist(),
                    y=sub["mean_new_alert_rate"].tolist(),
                    mode="lines+markers",
                    name=METHOD_LABELS.get(m, m),
                    line=dict(color=METHOD_COLORS.get(m, C0), width=1.8),
                    marker=dict(size=6)))
            layout_sp = _plo(240, dict(l=0, r=0, t=4, b=0))
            layout_sp["legend"]["orientation"] = "h"
            layout_sp["legend"]["y"] = -0.22
            fig.update_layout(**layout_sp)
            fig.update_xaxes(title_text="Spike strength (× IQR)", title_font_size=9,
                             tickvals=[1.5, 3.0, 5.0])
            fig.update_yaxes(tickformat=".0%",
                             title_text="New-alert rate", title_font_size=9)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption("Higher = method detects more injected spikes. "
                       "Low rates are not a failure — calibrated at α=0.05 on clean data.")

        # Per-type tabs
        _sec("Results by Disturbance Type", "&#9889;")
        tab1, tab2, tab3, tab4 = st.tabs([
            "&#9889; Spike", "&#128200; Drift",
            "&#128277; Dropout", "&#127777; Untouched"])

        with tab1:
            sp_df = (summary[(summary.stress == "artificial_spike") &
                              (summary.strength_iqr == 3.0)]
                     [["method", "mean_new_alert_rate", "trials",
                       "evaluated_rows_each_trial"]].copy())
            if not sp_df.empty:
                st.markdown("**New-alert rate at 3×IQR spike (5 seeds averaged)**")
                if HAS_PLOTLY:
                    fig_t = go.Figure(go.Bar(
                        x=[METHOD_LABELS.get(m, m) for m in sp_df["method"]],
                        y=sp_df["mean_new_alert_rate"].fillna(0).tolist(),
                        marker_color=[METHOD_COLORS.get(m, C0) for m in sp_df["method"]],
                        marker_line_width=0,
                        text=[f"{v:.2%}" for v in sp_df["mean_new_alert_rate"].fillna(0)],
                        textposition="outside", textfont=dict(size=10, color=W0)))
                    fig_t.update_layout(**_plo(200))
                    fig_t.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig_t, use_container_width=True,
                                    config={"displayModeBar": False})
                st.dataframe(sp_df.rename(columns={
                    "method": "Method",
                    "mean_new_alert_rate": "New-alert rate (3×IQR)",
                    "trials": "Seeds",
                    "evaluated_rows_each_trial": "Rows per trial",
                }).style.format({"New-alert rate (3×IQR)": "{:.2%}"}),
                use_container_width=True, hide_index=True)

        with tab2:
            dr_df = summary[summary.stress == "artificial_gradual_drift"][
                ["method", "mean_alert_rate", "evaluated_rows_each_trial"]]
            if not dr_df.empty:
                st.markdown("**Alert rate after gradual 2×IQR drift on one sensor**")
                st.dataframe(dr_df.rename(columns={
                    "method": "Method",
                    "mean_alert_rate": "Alert rate",
                    "evaluated_rows_each_trial": "Rows evaluated",
                }).style.format({"Alert rate": "{:.2%}"}),
                use_container_width=True, hide_index=True)
                st.caption("Alert rate — not a false-positive rate. No labels exist.")

        with tab3:
            do_df = summary[summary.stress == "artificial_sensor_dropout"][
                ["method", "mean_alert_rate", "mean_insufficient_readings_rate",
                 "evaluated_rows_each_trial"]]
            if not do_df.empty:
                st.markdown("**Dropout: 65% of features set to NaN for 20% of rows**")
                st.dataframe(do_df.rename(columns={
                    "method": "Method",
                    "mean_alert_rate": "Alert rate",
                    "mean_insufficient_readings_rate": "Abstention rate",
                    "evaluated_rows_each_trial": "Rows affected",
                }).style.format({"Alert rate": "{:.2%}",
                                 "Abstention rate": "{:.2%}"}),
                use_container_width=True, hide_index=True)
                st.caption("Abstention rate = fraction where < 50% channels observed "
                           "— no alert is issued for those rows.")

        with tab4:
            uh_df = summary[summary.stress == "untouched_heldout"][
                ["method", "mean_alert_rate", "evaluated_rows_each_trial"]]
            if not uh_df.empty:
                st.markdown("**Baseline: no disturbance, real holdout measurements**")
                st.dataframe(uh_df.rename(columns={
                    "method": "Method",
                    "mean_alert_rate": "Alert rate",
                    "evaluated_rows_each_trial": "Rows evaluated",
                }).style.format({"Alert rate": "{:.2%}"}),
                use_container_width=True, hide_index=True)
                st.caption("This is not a false-positive rate without contamination labels.")

    _footer()


# ╔═══════════════════════════════════════════════════════════════════════
# ║  PAGE 6 — METHODOLOGY
# ╚═══════════════════════════════════════════════════════════════════════
elif page == "Methodology":
    st.markdown(f"""
<div class="pg-header">
  <h1>Methodology</h1>
  <p>Evaluation protocol, detection approaches, calibration design,
     reproducibility, and scientific assumptions</p>
</div>""", unsafe_allow_html=True)
    _disc()

    _sec("Evaluation Pipeline", "&#9654;")
    st.markdown(f"""
<div class="workflow">
  <div class="wf-step">
    <div class="wf-step-num">01</div>
    <div class="wf-step-title">Dataset Selection</div>
    <div class="wf-step-desc">UCI wastewater, NYC drinking water,
    USGS river, synthetic demo, or CSV upload.
    Each dataset evaluated independently.</div>
  </div>
  <div class="wf-arrow">&#8250;</div>
  <div class="wf-step">
    <div class="wf-step-num">02</div>
    <div class="wf-step-title">Preprocessing (train only)</div>
    <div class="wf-step-desc">Impute medians, scale with RobustScaler.
    Fit on training set only — no test leakage.</div>
  </div>
  <div class="wf-arrow">&#8250;</div>
  <div class="wf-step">
    <div class="wf-step-num">03</div>
    <div class="wf-step-title">Model Training</div>
    <div class="wf-step-desc">Fit global median/IQR profiles, Isolation Forest
    (n=180, random_state=42), and per-station local profiles.</div>
  </div>
  <div class="wf-arrow">&#8250;</div>
  <div class="wf-step">
    <div class="wf-step-num">04</div>
    <div class="wf-step-title">Conformal Calibration</div>
    <div class="wf-step-desc">Score calibration set. Compute empirical
    rank p-values with +1 finite-sample correction.
    Alert threshold: &#945; = 0.05.</div>
  </div>
  <div class="wf-arrow">&#8250;</div>
  <div class="wf-step">
    <div class="wf-step-num">05</div>
    <div class="wf-step-title">Held-out Evaluation</div>
    <div class="wf-step-desc">Score unseen test rows. Inject 4 types
    of artificial faults. Record alert rates per method.</div>
  </div>
</div>""", unsafe_allow_html=True)

    _sec("Detection Methods", "&#128200;")
    c1, c2, c3 = st.columns(3)
    for col_, m_, desc_ in [
        (c1, "Robust Global",
         "Computes the **maximum median/IQR-scaled deviation** across all sensor "
         "features. The anomaly score for a row is its most unusual single-feature "
         "reading. Simple and interpretable, but one highly variable feature can "
         "dominate the score."),
        (c2, "Isolation Forest",
         "**Isolation-based outlier detection** (scikit-learn, 180 trees). "
         "Isolates anomalies by random feature/value splits. Effective for "
         "multi-dimensional outliers but less sensitive to single-feature spikes "
         "on datasets with many features."),
        (c3, "Sentinel Station-Aware",
         "**Station-specific robust scoring** with per-site local profiles "
         "shrunk toward the global profile for stability. Uses conformal p-values "
         "from site-specific calibration sets when enough calibration data exist. "
         "With only one station, it matches the Robust Global method."),
    ]:
        with col_:
            st.markdown(f"""
<div class="chart-panel" style="height:100%">
  <div class="chart-panel-title">{m_}</div>
  <div style="font-size:.81rem;color:{W1};line-height:1.6;margin-top:.4rem">{desc_}</div>
</div>""", unsafe_allow_html=True)

    _sec("Conformal Calibration Details", "&#9201;")
    st.markdown(f"""
<div class="chart-panel">
  <div class="chart-panel-sub">
  The calibration split produces empirical conformal p-values using the
  upper-tail rank formula with +1 finite-sample correction:
  </div>

```
p-value(x) = (1 + |{{ s in CAL : s >= score(x) }}|) / (|CAL| + 1)
```

A row receives an **anomaly alert** when its p-value ≤ α = 0.05.

**Important caveats:**
- This provides approximate 5% false-alert rate control **only if the test scores are exchangeable with calibration scores** — not guaranteed for time-ordered measurements.
- With a single station the site-specific and global calibrators are identical.
- No ground-truth contamination labels exist; false-alert rate cannot be verified empirically.
</div>""", unsafe_allow_html=True)

    _sec("Reproducibility", "&#128203;")
    st.markdown(f"""
<div class="chart-panel">

| Property | Value |
|---|---|
| Spike injection seed | Configurable (11, 22, 33, 44, 55 by default) |
| Missingness injection seed | 22 |
| Isolation Forest random_state | 42 |
| Split ratios | 60 / 20 / 20 (train / calibration / holdout) |
| Minimum dataset size | 150 records |
| Minimum features required | 2 numeric columns |
| Feature selection | ≥ 40% observed, ≥ 6 distinct values, capped at 38 |
| Abstention rule | Row has < 50% valid sensor channels → no alert |
| CI | GitHub Actions on Python 3.10 and 3.12 |

</div>""", unsafe_allow_html=True)

    _sec("Scientific Limitations", "&#9888;&#65039;")
    with st.expander("Read all stated limitations", expanded=True):
        if run_done and manifest:
            _lims()
        else:
            # Show fixed UCI limitations as canonical reference
            FIXED_LIMS = [
                "No measurement of real contamination detection accuracy",
                "Untouched historical alert rate is NOT a false positive rate without labels",
                "Conformal calibration under temporal shift lacks finite-sample coverage guarantee",
                "Station-specific inference requires enough training and calibration records",
                "Ablation study and expert validation remain necessary before claiming novelty",
            ]
            for l in FIXED_LIMS:
                st.markdown(f"""
<div class="lim">
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
       stroke="{AMB}" stroke-width="2.5"
       style="flex-shrink:0;margin-top:.18rem">
    <polyline points="9 18 15 12 9 6"/>
  </svg>{l}</div>""", unsafe_allow_html=True)

    _footer()


# ╔═══════════════════════════════════════════════════════════════════════
# ║  PAGE 7 — RESULTS & REPORTS
# ╚═══════════════════════════════════════════════════════════════════════
elif page == "Results & Reports":
    st.markdown(f"""
<div class="pg-header">
  <h1>Results &amp; Reports</h1>
  <p>Experiment summaries, benchmark tables, methodology notes,
     reproducibility information, and CSV/JSON exports</p>
</div>""", unsafe_allow_html=True)
    _disc()

    if not run_done or summary is None:
        _no_run("Results & Reports")
        # Show pre-computed UCI results even without running
        _sec("Pre-Computed UCI Benchmark (canonical results)", "&#128203;")
        st.markdown(f"""
<div class="chart-panel">
  <div class="chart-panel-sub">
  These are the verified results from the UCI Water Treatment Plant dataset
  (527 records, 38 features, 1 facility). Independently re-run and confirmed
  on 2026-10-10.
  </div>
</div>""", unsafe_allow_html=True)
        import json as _json
        from pathlib import Path as _Path
        _csv_path = _Path("benchmarks/uci_v3_exploratory_summary.csv")
        _man_path = _Path("benchmarks/uci_v3_exploratory_manifest.json")
        if _csv_path.exists():
            _precomp = pd.read_csv(_csv_path)
            st.dataframe(_precomp.style.format({
                "mean_alert_rate": "{:.2%}",
                "mean_new_alert_rate": "{:.2%}",
                "mean_insufficient_readings_rate": "{:.2%}",
                "strength_iqr": "{:.1f}",
            }, na_rep="—"), use_container_width=True, hide_index=True)
            dl_c1, dl_c2 = st.columns(2)
            dl_c1.download_button(
                "⬇ UCI Summary CSV (pre-computed)",
                _precomp.to_csv(index=False),
                file_name="uci_summary_precomputed.csv",
                mime="text/csv", use_container_width=True)
        if _man_path.exists():
            _man_text = _man_path.read_text(encoding="utf-8")
            dl_c2.download_button(
                "⬇ UCI Manifest JSON (pre-computed)",
                _man_text,
                file_name="uci_manifest_precomputed.json",
                mime="application/json", use_container_width=True)
    else:
        _data_badge()

        # Run info
        m = manifest
        _sec("Experiment Summary", "&#128203;")
        st.markdown(f"""
<div class="stat-row">
  <div class="stat"><div class="stat-lbl">Dataset</div>
    <div class="stat-val">{m.get('dataset','N/A').upper()}</div></div>
  <div class="stat"><div class="stat-lbl">Total Records</div>
    <div class="stat-val">{m.get('total_records',0):,}</div></div>
  <div class="stat"><div class="stat-lbl">Training</div>
    <div class="stat-val">{m.get('train',0):,}</div></div>
  <div class="stat"><div class="stat-lbl">Calibration</div>
    <div class="stat-val">{m.get('calibration',0):,}</div></div>
  <div class="stat"><div class="stat-lbl">Holdout</div>
    <div class="stat-val">{m.get('holdout',0):,}</div></div>
  <div class="stat"><div class="stat-lbl">Features</div>
    <div class="stat-val">{len(m.get('features',[]))}</div></div>
</div>
<div class="stat-row">
  <div class="stat"><div class="stat-lbl">Order Basis</div>
    <div class="stat-val" style="font-size:.78rem">{m.get('order_basis','N/A')}</div></div>
  <div class="stat"><div class="stat-lbl">Alert Threshold</div>
    <div class="stat-val">&#945; = {m.get('alpha',0.05)}</div></div>
  <div class="stat"><div class="stat-lbl">Train-only Preproc</div>
    <div class="stat-val">{str(m.get('train_only_preprocessing',''))}</div></div>
  <div class="stat"><div class="stat-lbl">Contamination Labels</div>
    <div class="stat-val">{str(m.get('verified_contamination_labels',''))}</div></div>
  <div class="stat"><div class="stat-lbl">All Faults Artificial</div>
    <div class="stat-val">{str(m.get('all_perturbations_are_artificial',''))}</div></div>
</div>""", unsafe_allow_html=True)

        with st.expander("Feature list", expanded=False):
            st.code(", ".join(m.get("features", [])), language=None)

        # Full results table
        _sec("Benchmark Results Table", "&#128200;")
        disp_cols = ["method", "stress", "strength_iqr", "trials",
                     "evaluated_rows_each_trial", "mean_alert_rate",
                     "mean_new_alert_rate", "mean_insufficient_readings_rate"]

        def _sr2(row):
            s = row.get("stress", ""); q = row.get("strength_iqr", 0)
            if s == "untouched_heldout":
                return ["background-color:rgba(0,184,152,.08)"] * len(row)
            if s == "artificial_spike" and q == 3.0:
                return ["background-color:rgba(0,200,240,.07)"] * len(row)
            return [""] * len(row)

        styled2 = (summary[disp_cols].style.apply(_sr2, axis=1)
                   .format({"mean_alert_rate": "{:.2%}",
                            "mean_new_alert_rate": "{:.2%}",
                            "mean_insufficient_readings_rate": "{:.2%}",
                            "strength_iqr": "{:.1f}"}, na_rep="—"))
        st.dataframe(styled2, use_container_width=True, hide_index=True)
        st.markdown(f"""
<div style="display:flex;gap:1rem;flex-wrap:wrap;font-size:.7rem;color:{W1};margin-top:.3rem">
  <span>&#9632; <span style="color:{T0}">Teal</span> = untouched holdout</span>
  <span>&#9632; <span style="color:{C0}">Cyan</span> = 3&#215;IQR spike</span>
  <span>&#945; = {m.get('alpha',0.05)} &#183; Alert rate &#8800; contamination probability</span>
</div>""", unsafe_allow_html=True)

        # Limitations
        _sec("Scientific Limitations", "&#9888;&#65039;")
        _lims()

        # Downloads
        _sec("Download Exports", "&#11015;")
        st.markdown(
            "All downloads contain **genuine experiment results** from the run above.")
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(
            "⬇  Summary CSV",
            summary.to_csv(index=False),
            file_name=f"aquaguard_{dataset_key}_summary.csv",
            mime="text/csv", use_container_width=True,
            help="Aggregated alert rates by method, stress type, spike strength.")
        dl2.download_button(
            "⬇  Full Trials CSV",
            trials.to_csv(index=False),
            file_name=f"aquaguard_{dataset_key}_trials.csv",
            mime="text/csv", use_container_width=True,
            help="Individual trial rows — one per method × stress × seed.")
        dl3.download_button(
            "⬇  Run Manifest JSON",
            json.dumps(manifest, indent=2),
            file_name=f"aquaguard_{dataset_key}_manifest.json",
            mime="application/json", use_container_width=True,
            help="Exact split sizes, features, alpha, and all stated limitations.")
        st.caption("The manifest JSON fully documents the experimental conditions "
                   "for reproducibility.")

    _footer()
