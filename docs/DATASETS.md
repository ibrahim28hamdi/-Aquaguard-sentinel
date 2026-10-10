# Public datasets and provenance

| Source | Domain | Acquisition | Limitations |
|---|---|---|---|
| [UCI Water Treatment Plant](https://archive.ics.uci.edu/dataset/106/water+treatment+plant) | Wastewater treatment | Auto-downloaded from UCI legacy mirror on first dashboard use; cached in `data/` | 527 historical records · single site · no verified pollution-event labels |
| [NYC Drinking Water Quality](https://data.cityofnewyork.us/Environment/Drinking-Water-Quality-Distribution-Monitoring-Dat/bkwf-xfky) | Drinking water monitoring | NYC Open Data API | Selected chemistry fields only; reporting frequency and missingness vary |
| [USGS Water Data](https://api.waterdata.usgs.gov/download) | River / estuary monitoring | USGS NWIS legacy IV API or manual CSV import | Station availability varies; older endpoint may be retired |
| Synthetic demo | Artificial examples | Generated locally, fixed seed | Not environmental evidence |

Water classes are evaluated **separately**, never combined into one classifier.

---

## UCI Water Treatment Plant — detailed notes

- **License:** CC BY 4.0 — Poch, M. (1993). doi:[10.24432/C5FS4C](https://doi.org/10.24432/C5FS4C)
- **Expected format:** 39 columns — one sample-date field followed by 38 numeric measurements.
- **Record ordering:** source file order is preserved (no verified timestamps); no chronological assumption is made.
- **Labels:** no verified contamination-event labels; anomaly alerts are not safety verdicts.
- **Auto-fetch:** `aquaguard/fetch_uci.py` downloads `water-treatment.data` from the UCI legacy mirror (`archive.ics.uci.edu/ml/machine-learning-databases/water-treatment/`) if the local cache is absent. The file is excluded from version control (`.gitignore: /data/*.data`) in accordance with UCI's data terms.
- **Note (2026-10):** The `ucimlrepo` Python API returns `DatasetNotFoundError` for dataset 106. The legacy direct-file URL is used instead.

---

## CSV upload — requirements

A valid uploaded CSV requires:

- ≥ 150 rows
- At least 2 numeric columns with ≥ 40 % non-missing values and ≥ 6 distinct values
- Optional: `station` column (text), `observed_at` column (ISO 8601 timestamps)

If `observed_at` is present and parseable for ≥ 90 % of rows, the split is chronological; otherwise source order is preserved.

---

## Ethics

Do not use this application to decide that water is potable, safe, or contaminated. Anomaly scores flag unusual sensor readings only.
