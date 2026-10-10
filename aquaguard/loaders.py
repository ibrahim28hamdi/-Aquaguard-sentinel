"""Open-data adapters. No private data, and no fabricated real-world records."""
from __future__ import annotations
import io
from pathlib import Path
import requests
import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/dataset/106/water+treatment+plant"
NYC_URL = "https://data.cityofnewyork.us/Environment/Drinking-Water-Quality-Distribution-Monitoring-Dat/bkwf-xfky"
USGS_URL = "https://api.waterdata.usgs.gov/download"
NYC_API = "https://data.cityofnewyork.us/resource/bkwf-xfky.csv"
USGS_LEGACY_IV = "https://waterservices.usgs.gov/nwis/iv/"  # Service may be retired/changed.


UCI_FEATURES = ['Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SSV-E', 'SED-E', 'COND-E', 'PH-P', 'DBO-P', 'SS-P', 'SSV-P', 'SED-P', 'COND-P', 'PH-D', 'DBO-D', 'DQO-D', 'SS-D', 'SSV-D', 'SED-D', 'COND-D', 'PH-S', 'DBO-S', 'DQO-S', 'SS-S', 'SSV-S', 'SED-S', 'COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P', 'RD-DBO-S', 'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G', 'RD-SS-G', 'RD-SED-G']


def load_uci_local(path='data/water-treatment.data'):
    """Read UCI's original 39-column raw file (date + 38 measurement fields).

    Use the official UCI file, not an arbitrary CSV. We preserve source order.
    No contamination labels are implied.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f'Expected UCI raw file at {file}')
    frame = pd.read_csv(file, header=None, na_values=['?', ' ?'], skipinitialspace=True)
    if frame.shape[1] != 39:
        raise ValueError(f'Expected 39 columns in UCI raw data, got {frame.shape[1]}')
    frame.columns = ['sample_date'] + UCI_FEATURES
    for col in UCI_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame.insert(0, 'station', 'wastewater_plant_1')
    frame.insert(1, 'water_type', 'wastewater')
    return frame


def load_uci():
    """Use locally cached UCI file first; fall back to public ucimlrepo API."""
    local_path=Path(__file__).resolve().parent.parent/'data'/'water-treatment.data'
    if local_path.exists():
        return load_uci_local(local_path)
    try:
        from ucimlrepo import fetch_ucirepo
        source = fetch_ucirepo(id=106)
    except Exception as e:
        raise RuntimeError(
            'UCI download unavailable. Place the official water-treatment.data file '
            'in the project data/ folder and try again. Dataset: ' + UCI_URL
        ) from e
    frame = source.data.features.copy()
    if frame.empty:
        raise ValueError('UCI dataset returned no measurements.')
    frame.columns = [str(c).strip().replace(' ', '_') for c in frame.columns]
    frame.insert(0, 'station', 'wastewater_plant_1')
    frame.insert(1, 'water_type', 'wastewater')
    return frame

def load_nyc(max_rows: int = 30000):
    """Open NYC drinking-water samples; text fields stay NA if not unambiguously numeric."""
    if not 200 <= max_rows <= 50000:
        raise ValueError("max_rows must be 200 to 50000")
    params = {'$limit': max_rows, '$order': 'sample_date DESC'}
    try:
        r = requests.get(NYC_API, params=params, timeout=75)
        r.raise_for_status()
        frame = pd.read_csv(io.BytesIO(r.content), low_memory=False)
    except Exception as e:
        raise RuntimeError("NYC OpenData download unsuccessful. Try again or use a CSV upload.") from e
    chosen = [
        'residual_free_chlorine_mg_l', 'turbidity_ntu', 'fluoride_mg_l',
    ]
    needed = ['sample_site', 'sample_date']
    if not all(c in frame.columns for c in needed):
        raise ValueError(f"NYC returned unexpected columns: {frame.columns.tolist()}")
    columns = [c for c in chosen if c in frame.columns]
    if len(columns) < 2:
        raise ValueError("Insufficient matching NYC measurement columns.")
    out = frame[needed + columns].copy()
    out = out.rename(columns={'sample_site': 'station', 'sample_date': 'observed_at'})
    out['station'] = out['station'].astype('string').fillna('unknown').astype(str)
    out['water_type'] = 'drinking'
    return out


def parse_usgs_timeseries(payload: dict) -> pd.DataFrame:
    """Parse USGS NWIS iv JSON, preserving the actual agency variable codes."""
    series = payload.get('value', {}).get('timeSeries', [])
    code_to_name = {'00010': 'temperature_c', '00095': 'conductivity_us_cm',
                    '00300': 'dissolved_oxygen_mg_l', '00400': 'ph',
                    '63680': 'turbidity_fnu'}
    parts = []
    for ts in series:
        variable = ts.get('variable', {})
        codes = variable.get('variableCode', [])
        code = str(codes[0].get('value')) if codes else None
        if code not in code_to_name:
            continue
        site_codes = ts.get('sourceInfo', {}).get('siteCode', [])
        site = str(site_codes[0].get('value')) if site_codes else 'unknown'
        samples = []
        for block in ts.get('values', []):
            samples.extend(block.get('value', []))
        if not samples:
            continue
        part = pd.DataFrame(samples)
        if 'dateTime' not in part or 'value' not in part:
            continue
        part = part[['dateTime', 'value']].copy()
        part['value'] = pd.to_numeric(part['value'], errors='coerce')
        part = part.rename(columns={'dateTime': 'observed_at', 'value': code_to_name[code]})
        part['station'] = site
        # Combine codes with outer join after building all independent series.
        parts.append(part.groupby(['observed_at', 'station'], as_index=False).mean(numeric_only=True))
    if not parts:
        raise ValueError('USGS returned no matching numeric measurements at this site/time.')
    out = parts[0]
    for next_part in parts[1:]:
        out = out.merge(next_part, on=['observed_at', 'station'], how='outer')
    out['water_type'] = 'river_or_estuary'
    return out


def load_usgs(site: str = '01482800', start: str = '2025-06-01', end: str = '2025-07-01'):
    """Best-effort USGS legacy JSON endpoint; user may instead upload USGS CSV."""
    params = {'format': 'json', 'sites': site,
              'parameterCd': '00010,00095,00300,00400,63680',
              'startDT': start, 'endDT': end, 'siteStatus': 'all'}
    try:
        resp = requests.get(USGS_LEGACY_IV, params=params, timeout=90)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        raise RuntimeError('USGS endpoint unavailable. Download an official CSV at '
                           + USGS_URL + ' and use the custom CSV route.') from e
    return parse_usgs_timeseries(payload)


def load_custom(file_or_path, *, water_type: str, station_col: str = 'station',
                time_col: str = 'observed_at'):
    if isinstance(file_or_path, pd.DataFrame):
        frame = file_or_path.copy()
    else:
        frame = pd.read_csv(file_or_path, low_memory=False)
    if station_col not in frame:
        frame['station'] = 'unknown'
    else:
        frame['station'] = frame[station_col].astype('string').fillna('unknown').astype(str)
    if time_col in frame:
        frame['observed_at'] = frame[time_col]
    frame['water_type'] = water_type
    frame = frame.drop(columns=[c for c in [station_col, time_col] if c in frame and c not in ['station','observed_at']], errors='ignore')
    return frame


def synthetic_demo(n=900, seed=42):
    """Simulated measurements for testing USER INTERFACE ONLY. NEVER field data."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    sites = np.array(['demo_site_a','demo_site_b','demo_site_c'])[t % 3]
    shift = (t % 3) * 1.8
    frame = pd.DataFrame({
        'station': sites, 'water_type': 'synthetic_demo',
        'observed_at': pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC'),
        'sensor_a': 7.2 + 0.05 * shift + .18 * rng.normal(size=n),
        'sensor_b': 15 + 2.5 * shift + 1.8 * np.sin(t/13) + rng.normal(size=n),
        'sensor_c': 120 + 25 * shift + 8 * rng.normal(size=n),
        'sensor_d': 2 + 0.2 * shift + 0.18 * rng.normal(size=n),
    })
    frame.loc[rng.choice(n,size=n//20,replace=False),'sensor_c'] = np.nan
    return frame
