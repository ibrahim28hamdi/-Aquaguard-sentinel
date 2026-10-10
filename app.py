"""AquaGuard Sentinel — Streamlit dashboard.

Run: streamlit run app.py
"""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from aquaguard.loaders import (
    synthetic_demo, load_nyc, load_uci, load_usgs, load_custom,
    NYC_URL, UCI_URL, USGS_URL,
)
from aquaguard.experiment import test_benchmark
from aquaguard.fetch_uci import ensure_uci_file  # thin data-fetch helper, not research code

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaGuard Sentinel",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── minimal style tweaks ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; max-width: 1280px; }
    h1 { letter-spacing: -.03em; }
    h2, h3 { letter-spacing: -.02em; margin-top: 1.4rem; }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.2);
        border-radius: 12px;
        padding: .9rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── header ────────────────────────────────────────────────────────────────────
st.title("💧 AquaGuard Sentinel")
st.caption("Open-data analytics · Anomaly monitoring · Reproducible experiments")

st.warning(
    "⚠️ **Research and education only.** "
    "Anomaly alerts flag unusual sensor readings — they do not certify water safety "
    "or prove contamination.",
    icon=None,
)

with st.expander("ℹ️ About this project", expanded=False):
    st.markdown(
        """
        AquaGuard Sentinel is an **educational ML portfolio project** that compares three
        anomaly-detection baselines on public water-quality datasets:

        | Method | What it scores |
        |---|---|
        | **Robust global** | Max median/IQR deviation across all features |
        | **Isolation Forest** | Isolation-based outlier score |
        | **Sentinel station** | Station-aware robust deviation with conformal calibration |

        **Evaluation:** first 60 % of records for training, next 20 % for calibration,
        last 20 % for testing. Stress tests inject **artificial** spikes, drift, and
        missing-data — not verified pollution events.

        Data sources: [NYC Drinking Water]({nyc}) · [UCI Wastewater]({uci}) ·
        [USGS River/Estuary]({usgs})
        """.format(nyc=NYC_URL, uci=UCI_URL, usgs=USGS_URL)
    )

st.divider()

# ── sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Dataset")
    kind = st.selectbox(
        "Choose a dataset",
        [
            "Synthetic demo (offline)",
            "UCI wastewater (local / internet)",
            "NYC drinking water (internet)",
            "USGS river/estuary (internet)",
            "Upload CSV (your own or public)",
        ],
        help="Each dataset type is evaluated separately — they are never combined.",
    )

    max_rows = None
    csv_file = None
    water_type = None

    if "NYC" in kind:
        max_rows = st.slider(
            "Maximum NYC rows", 1000, 50_000, 12_000, step=1000,
            help="NYC Open Data API — large requests take longer.",
        )

    if "Upload" in kind:
        csv_file = st.file_uploader(
            "Upload CSV", type=["csv"],
            help="Needs ≥ 150 rows and ≥ 2 numeric measurement columns.",
        )
        water_type = st.selectbox(
            "Water type", ["river", "drinking", "wastewater", "other"]
        )

    st.divider()
    st.header("Run")
    go = st.button(
        "▶ Run transparent comparison",
        type="primary",
        use_container_width=True,
        disabled=("Upload" in kind and csv_file is None),
    )
    if "Upload" in kind and csv_file is None:
        st.caption("Upload a CSV file above to enable the run button.")

    st.divider()
    st.caption(
        "**Tip:** The offline demo runs instantly without any download. "
        "UCI wastewater loads from a local file if present, or fetches it automatically."
    )

# ── main panel placeholder text (before run) ─────────────────────────────────
if not go:
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("How it works")
        st.markdown(
            """
            1. **Select** a dataset in the sidebar.
            2. **Click** *Run transparent comparison*.
            3. Explore the results table and charts below.
            4. **Download** the summary CSV, full trial CSV, or JSON manifest.

            The offline Synthetic demo requires no internet and completes in seconds.
            """
        )
    with col_r:
        st.subheader("What the numbers mean")
        st.markdown(
            """
            - **`mean_alert_rate` on untouched data** — fraction of holdout rows flagged.
              *Not* a verified false-positive rate (no ground-truth labels exist).
            - **`mean_new_alert_rate` on artificial spikes** — fraction of newly triggered
              alerts on rows that were not already flagged.
              *Not* pollution-detection accuracy.
            - **Sensor data insufficient** — row abstained because fewer than 50 % of
              sensor channels had a valid reading.
            """
        )
    st.stop()

# ── data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load(kind: str, max_rows):
    if kind.startswith("Synthetic"):
        return synthetic_demo()
    if kind.startswith("UCI"):
        ensure_uci_file()      # download if missing — no-op if already present
        return load_uci()
    if kind.startswith("NYC"):
        return load_nyc(max_rows=max_rows)
    if kind.startswith("USGS"):
        return load_usgs()
    raise ValueError("Use the CSV uploader for custom data.")


dataset_key = kind.split()[0].lower()

with st.spinner("Loading dataset…"):
    try:
        if "Upload" in kind:
            data = load_custom(csv_file, water_type=water_type)
        else:
            data = _load(kind, max_rows)
    except Exception as exc:
        st.error(
            f"**Could not load dataset.**\n\n{exc}\n\n"
            "**Suggestions:**\n"
            "- For UCI: place `water-treatment.data` in the `data/` folder.\n"
            "- For NYC / USGS: check your internet connection and try again.\n"
            "- For a CSV: make sure it has ≥ 150 rows and ≥ 2 numeric columns.",
        )
        st.stop()

# ── benchmark ─────────────────────────────────────────────────────────────────
with st.spinner("Running deterministic stress tests… (may take 10–30 s)"):
    try:
        summary, trials, manifest = test_benchmark(data, dataset=dataset_key)
    except Exception as exc:
        st.error(
            f"**Benchmark failed.**\n\n{exc}\n\n"
            "This usually means the dataset has too few rows (< 150) "
            "or fewer than 2 variable numeric columns.",
        )
        st.stop()

# ── data-type banner ──────────────────────────────────────────────────────────
if dataset_key == "synthetic":
    st.warning(
        "🔬 **Synthetic data only.** "
        "These measurements are computer-generated. "
        "No scientific conclusion about real water quality can be drawn.",
    )
else:
    st.info(
        "📊 **Real historical measurements with artificially injected sensor faults.** "
        "No verified contamination labels exist. Results reflect sensor-disturbance "
        "stress tests, not pollution-detection performance.",
    )

# ── record count metrics ──────────────────────────────────────────────────────
st.subheader("Dataset overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total records", f"{manifest['total_records']:,}")
c2.metric("Training rows", f"{manifest['train']:,}")
c3.metric("Calibration rows", f"{manifest['calibration']:,}")
c4.metric("Holdout rows", f"{manifest['holdout']:,}")
c5.metric("Input features", len(manifest["features"]))

with st.expander("Feature list and run details", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Order basis:** `{manifest['order_basis']}`")
        st.markdown(f"**Alert significance threshold (α):** `{manifest['alpha']}`")
        st.markdown(f"**Stations — train / cal / holdout:** "
                    f"`{manifest['sites_train']}` / `{manifest['sites_calibration']}` / `{manifest['sites_heldout']}`")
        st.markdown(f"**Local profiles fit:** `{manifest['local_profiles_fit']}`")
        st.markdown(f"**Verified contamination labels:** `{manifest['verified_contamination_labels']}`")
    with col_b:
        st.markdown("**Features used:**")
        st.code(", ".join(manifest["features"]), language=None)

# ── model comparison table ────────────────────────────────────────────────────
st.subheader("Model comparison")

# Highlight untouched holdout and 3-IQR spike rows for quick reading
def _style_row(row):
    if row.get("stress") == "untouched_heldout":
        return ["background-color: rgba(0,120,212,.07)"] * len(row)
    if row.get("stress") == "artificial_spike" and row.get("strength_iqr") == 3.0:
        return ["background-color: rgba(255,165,0,.07)"] * len(row)
    return [""] * len(row)

display_cols = [
    "method", "stress", "strength_iqr", "trials",
    "evaluated_rows_each_trial", "mean_alert_rate",
    "mean_new_alert_rate", "mean_insufficient_readings_rate",
]
styled = (
    summary[display_cols]
    .style
    .apply(_style_row, axis=1)
    .format({
        "mean_alert_rate": "{:.1%}",
        "mean_new_alert_rate": "{:.1%}",
        "mean_insufficient_readings_rate": "{:.1%}",
        "strength_iqr": "{:.1f}",
    }, na_rep="—")
)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.caption(
    "Blue rows = untouched holdout (baseline flag rate). "
    "Orange rows = 3-IQR artificial spike (stress test). "
    "α = 0.05 is the alert significance threshold, not a validated contamination probability."
)

# ── charts ────────────────────────────────────────────────────────────────────
st.subheader("Charts")

tab1, tab2, tab3 = st.tabs([
    "🔴 Artificial spike detection",
    "📉 Untouched holdout alert rate",
    "❓ Sensor data insufficient",
])

with tab1:
    spike_df = summary[
        (summary.stress == "artificial_spike") &
        (summary.strength_iqr == 3.0)
    ][["method", "mean_new_alert_rate"]].dropna()

    if not spike_df.empty:
        st.markdown(
            "**New-alert rate on rows with 3-IQR artificial single-feature spikes** "
            "(rows not already flagged on untouched data). "
            "Higher = more stress-test anomalies caught."
        )
        st.bar_chart(
            spike_df.set_index("method")["mean_new_alert_rate"],
            use_container_width=True,
        )
        st.caption(
            "This measures response to simulated sensor errors, "
            "**not** real-contamination detection accuracy."
        )
    else:
        st.info("No 3-IQR spike data available for this dataset.")

with tab2:
    base_df = summary[summary.stress == "untouched_heldout"][
        ["method", "mean_alert_rate"]
    ]
    if not base_df.empty:
        st.markdown(
            "**Fraction of held-out historical rows flagged without any injected faults.** "
            "This is *not* a false-positive rate — no ground-truth labels exist."
        )
        st.bar_chart(
            base_df.set_index("method")["mean_alert_rate"],
            use_container_width=True,
        )
    else:
        st.info("No untouched holdout data available.")

with tab3:
    insuff_df = summary[summary.stress == "untouched_heldout"][
        ["method", "mean_insufficient_readings_rate"]
    ]
    if not insuff_df.empty:
        st.markdown(
            "**Fraction of held-out rows where fewer than 50 % of sensor channels "
            "had a valid reading.** These rows are abstained from — no alert is issued."
        )
        st.bar_chart(
            insuff_df.set_index("method")["mean_insufficient_readings_rate"],
            use_container_width=True,
        )
    else:
        st.info("No missingness data available.")

# ── interpretation note ───────────────────────────────────────────────────────
with st.expander("📖 How to interpret these results", expanded=False):
    st.markdown(
        """
        **What these results show:**
        - Anomaly detectors respond differently to different types of artificial sensor faults.
        - Isolation Forest is less sensitive to isolated single-feature spikes on this dataset.
        - Robust and station-aware methods respond similarly when there is only one station.
        - Low new-alert rates are **not** a failure — they mean few injected errors exceeded the
          calibration threshold, which is expected when spikes are modest relative to natural variation.

        **What these results do NOT show:**
        - Whether the detectors would catch real contamination events (no labels exist).
        - Whether station-aware scoring is genuinely better than the global baseline.
        - Whether 5 % is the right alert threshold for any real monitoring application.

        See [`docs/EVALUATION.md`](https://github.com) for full benchmark design notes.
        """
    )

# ── downloads ─────────────────────────────────────────────────────────────────
st.subheader("Download results")
dl1, dl2, dl3 = st.columns(3)
dl1.download_button(
    "⬇ Summary CSV",
    summary.to_csv(index=False),
    file_name="aquaguard_summary.csv",
    mime="text/csv",
    use_container_width=True,
)
dl2.download_button(
    "⬇ Full trials CSV",
    trials.to_csv(index=False),
    file_name="aquaguard_trials.csv",
    mime="text/csv",
    use_container_width=True,
)
dl3.download_button(
    "⬇ Run manifest JSON",
    json.dumps(manifest, indent=2),
    file_name="aquaguard_manifest.json",
    mime="application/json",
    use_container_width=True,
)

st.caption(
    "The manifest JSON records the exact split sizes, feature list, alpha, "
    "and all stated limitations for reproducibility."
)

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AquaGuard Sentinel · Educational portfolio project · "
    "Code developed with AI assistance · "
    "MIT licence for project code · Datasets separately licensed by their publishers"
)
