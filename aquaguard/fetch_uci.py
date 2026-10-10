"""Utility: ensure the UCI Water Treatment Plant raw data file is present.

This is a data-acquisition helper, not part of the anomaly-detection research code.
It is called only from the Streamlit dashboard (app.py) before load_uci() runs.

The dataset is licensed under CC BY 4.0:
  Poch, M. (1993). Water Treatment Plant [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5FS4C
Attribution is preserved in docs/DATASETS.md and README.md.

The legacy direct-file URL is used because the ucimlrepo Python API no longer
serves dataset 106 programmatically (DatasetNotFoundError, confirmed 2026-10).
The file is excluded from version control via .gitignore (/data/*.data) in
accordance with UCI's data terms. Users who clone the repo will have the file
auto-fetched on first run.
"""
from __future__ import annotations

from pathlib import Path
import urllib.request

# Official UCI legacy machine-learning-databases mirror (stable for 30+ years)
_UCI_DIRECT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "water-treatment/water-treatment.data"
)

# Resolved relative to the project root (parent of this file's package dir)
_LOCAL_PATH = Path(__file__).resolve().parent.parent / "data" / "water-treatment.data"

_EXPECTED_ROWS = 527
_EXPECTED_COLS = 39


def _file_looks_valid(path: Path) -> bool:
    """Quick structural check: 527 non-empty lines, 39 comma-separated fields."""
    try:
        lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        return len(lines) == _EXPECTED_ROWS and len(lines[0].split(",")) == _EXPECTED_COLS
    except Exception:
        return False


def ensure_uci_file(force: bool = False) -> Path:
    """Download water-treatment.data if it is absent or corrupt.

    Returns the local path. Raises RuntimeError with a clear message if the
    download fails so the dashboard can surface a helpful error.
    """
    if not force and _LOCAL_PATH.exists() and _file_looks_valid(_LOCAL_PATH):
        return _LOCAL_PATH

    _LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(_UCI_DIRECT_URL, _LOCAL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the UCI Water Treatment Plant dataset.\n\n"
            f"URL tried: {_UCI_DIRECT_URL}\n"
            f"Error: {exc}\n\n"
            "To fix this manually:\n"
            "1. Visit https://archive.ics.uci.edu/dataset/106/water+treatment+plant\n"
            "2. Download the file named water-treatment.data\n"
            "3. Place it in the project's data/ folder\n\n"
            "The dataset is licensed CC BY 4.0 (Poch, 1993 — doi:10.24432/C5FS4C)."
        ) from exc

    if not _file_looks_valid(_LOCAL_PATH):
        _LOCAL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded file did not pass validation "
            f"(expected {_EXPECTED_ROWS} rows × {_EXPECTED_COLS} columns). "
            "Please download water-treatment.data manually from "
            "https://archive.ics.uci.edu/dataset/106/water+treatment+plant "
            "and place it in data/."
        )

    return _LOCAL_PATH
