# Getting Started — AquaGuard Sentinel

This is the stable **GitHub portfolio edition** of AquaGuard Sentinel. It is based on the tested V3 application. The experimental V4/V5 prototypes are **not** required for this release.

## 1. Requirements

- Windows 10 or 11 (instructions below), or macOS/Linux (see manual setup)
- Python 3.10 or newer
- Internet access on the first launch to install Python dependencies
- A web browser

## 2. Run the app on Windows

1. Extract `AquaGuard_Sentinel_GitHub_English.zip`.
2. Open the `AquaGuard_GitHub_Release` folder.
3. Double-click **`START_WINDOWS.cmd`**. The script creates a local `.venv` environment if needed, installs packages from `requirements.txt`, and starts Streamlit. Review scripts before running them.
4. Open the URL shown in the terminal, typically **http://localhost:8501**.
5. Select **`Synthetic demo (offline)`** and click **`Run transparent comparison`** to check that the dashboard works. This demo uses synthetic measurements, not verified pollution events.
6. Use the dashboard download options to export benchmark summaries and experiment metadata.

**Run the automated tests:** double-click `RUN_TESTS_WINDOWS.cmd` after creating the virtual environment.

## 3. Run the app from a terminal (all platforms)

Open a terminal in `AquaGuard_GitHub_Release` and run:

```bash
python -m venv .venv
```

Activate the environment:

- **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`
- **Windows Command Prompt:** `.venv\Scripts\activate.bat`
- **macOS/Linux:** `source .venv/bin/activate`

Then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

## 4. Work with actual water-quality measurements

For the public **UCI Water Treatment Plant** dataset:

1. Select **`UCI wastewater (local / internet)`** in the dashboard and click *Run transparent comparison*.
2. The app **automatically downloads** `water-treatment.data` from the UCI legacy mirror on first use and caches it in `data/`. No manual action is needed.
3. If the automatic download fails (no internet access), download the file manually:
   - Visit https://archive.ics.uci.edu/dataset/106/water+treatment+plant
   - Direct file URL: `https://archive.ics.uci.edu/ml/machine-learning-databases/water-treatment/water-treatment.data`
   - Place it at `data/water-treatment.data`

> **Note:** The `ucimlrepo` Python API no longer serves dataset 106 (confirmed 2026-10). The app uses the stable UCI legacy file URL instead; the `ucimlrepo` package is still listed in `requirements-research-only.txt` for the CLI experiment runner but is not used by the dashboard.

Alternatively, upload a CSV through the dashboard. See [docs/DATASETS.md](docs/DATASETS.md) for the supported data sources and input guidance.

**Important:** Although the underlying UCI data consists of actual historical measurements, stress-test anomalies are *artificially injected*. The system does not verify contamination, pathogen presence, or whether water is safe to drink.

## 5. Publish the project on GitHub

1. Sign in to https://github.com and create a **public** repository named `aquaguard-sentinel`.
2. Upload the **contents of** `AquaGuard_GitHub_Release` to the repository root, not the outer ZIP file and not the older V3/V4/V5 development folders.
3. Include `.github/workflows/tests.yml`, `.gitignore`, and `.streamlit/config.toml`. When uploading in the browser, verify that hidden files and folders are included; use Git on your computer if the browser upload omits them.
4. Do **not** upload `.venv`, `data/water-treatment.data`, private datasets, secrets, or generated run folders. All of these are covered by `.gitignore`.
5. Set the repository description to: **Open-data water-quality anomaly explorer built with Python, scikit-learn, and Streamlit.**
6. Check the **Actions** tab to confirm that automated tests pass on GitHub.
7. Optionally include a screenshot of your dashboard, then link the repository in your scholarship application portfolio.

## 5b. Deploy to Streamlit Community Cloud (optional, free)

1. Push the repository to GitHub (step 5 above).
2. Go to https://share.streamlit.io and sign in with your GitHub account.
3. Click **New app**, select your repository, branch `main`, and main file `app.py`.
4. Click **Deploy** — no secrets or API keys are required.
5. On first load the app will automatically download `water-treatment.data` from the UCI legacy mirror.
6. Share the public URL in your portfolio.

## 6. What to say about the project

Present AquaGuard as a Python and machine-learning portfolio project demonstrating data processing, model comparisons, interactive dashboards, experimental reproducibility, and responsible reporting of limitations. Accurately describe the parts you built, reviewed, tested, and understood yourself, including any AI coding assistance. Do not claim that this software detects real pollution or that its test results are peer-reviewed research findings.

More details are in [README.md](README.md), [docs/EVALUATION.md](docs/EVALUATION.md), and [docs/PORTFOLIO_DESCRIPTION.md](docs/PORTFOLIO_DESCRIPTION.md).

## Troubleshooting

- **`Python launcher is missing`**: Install Python 3.10+ from https://www.python.org/downloads/ and reopen the terminal.
- **`No module named streamlit` or `numpy`**: Use `START_WINDOWS.cmd` or install dependencies *inside the project's environment* with `python -m pip install -r requirements.txt`.
- **The browser does not open automatically**: Copy the Streamlit URL from the terminal into your browser.
- **UCI download fails**: Download `water-treatment.data` manually and place it in `data/` as described above.
- **Port 8501 is already in use**: Close the previous Streamlit session, or run `python -m streamlit run app.py --server.port 8502`.
