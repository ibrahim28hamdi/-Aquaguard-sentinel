"""Transparent research baselines and a *proposed* station-aware method.

Scores express atypical measurements, NOT pathogens, safety, or pollution.
Empirical conformal p values require exchangeability for coverage statements;
time drift/new stations can invalidate that assumption.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

METHODS = ('robust_global', 'isolation_forest', 'sentinel_station')
META = {'station', 'water_type', 'observed_at', 'sample_id', 'sample_number', 'sample_date', 'date', 'id', 'target', 'label'}


def split_three(frame: pd.DataFrame):
    """Chronological if valid timestamps exist; otherwise preserve source row order."""
    x = frame.copy()
    ordered_by = 'source_order_only'
    if 'observed_at' in x:
        times = pd.to_datetime(x['observed_at'], errors='coerce', utc=True)
        if times.notna().mean() > .9:
            x = x.assign(observed_at=times).sort_values('observed_at', kind='stable')
            ordered_by = 'observed_at'
    x = x.reset_index(drop=True)
    if len(x) < 150:
        raise ValueError('Need at least 150 ordered records for three-way evaluation.')
    a, b = int(len(x) * .6), int(len(x) * .8)
    return x.iloc[:a].copy(), x.iloc[a:b].copy(), x.iloc[b:].copy(), ordered_by


def select_features(train: pd.DataFrame, candidate=None):
    candidates = [str(c) for c in train.columns if str(c) not in META]
    if candidate:
        candidates = [c for c in candidates if c in candidate]
    cols = []
    for col in candidates:
        arr = pd.to_numeric(train[col], errors='coerce')
        if arr.notna().mean() >= 0.4 and arr.nunique(dropna=True) >= 6:
            cols.append(col)
    if len(cols) < 2:
        raise ValueError('Need >= 2 variable numeric columns present in >= 40% of TRAIN records.')
    return cols[:38]


def numeric(frame, features):
    return frame.reindex(columns=features).apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)


def robust_stats(matrix):
    with np.errstate(all='ignore'):
        median = np.nanmedian(matrix,axis=0)
        q25 = np.nanpercentile(matrix,25,axis=0)
        q75 = np.nanpercentile(matrix,75,axis=0)
    iqr = q75-q25
    iqr = np.where((iqr > 1e-8)&np.isfinite(iqr),iqr,1.)
    median = np.where(np.isfinite(median),median,0.)
    return median, iqr


def conformal_pvalues(query: np.ndarray, cal: np.ndarray):
    """Upper-tail split-conformal ranks, including +1 finite sample correction."""
    query = np.asarray(query,float)
    cal = np.asarray(cal,float)
    cal = cal[np.isfinite(cal)]
    if len(cal) < 10:
        raise ValueError('Insufficient calibration scores')
    sorted_cal=np.sort(cal)
    return (1 + len(cal) - np.searchsorted(sorted_cal,query,side='left'))/(len(cal)+1)


@dataclass
class Sentinel:
    features: list[str] = field(default_factory=list)
    global_median: np.ndarray | None = None
    global_scale: np.ndarray | None = None
    local: dict = field(default_factory=dict)
    imputer: object | None = None
    scaler: object | None = None
    forest: object | None = None
    calibration: dict = field(default_factory=dict)
    site_cal: dict = field(default_factory=dict)
    min_local: int = 25
    min_site_cal: int = 25

    def fit(self, train, calibration):
        self.features=select_features(train)
        x=numeric(train,self.features)
        self.global_median,self.global_scale=robust_stats(x)
        # Local reference profiles fit on training data ONLY.
        self.local={}
        for site in train['station'].astype(str).unique():
            sub = x[(train['station'].astype(str).values == site)]
            if len(sub) < self.min_local:
                continue
            center,scale=robust_stats(sub)
            # Shrink site profile toward global for stability on small sites.
            weight=len(sub)/(len(sub)+60.)
            center=weight*center+(1-weight)*self.global_median
            scale=weight*scale+(1-weight)*self.global_scale
            self.local[site]=(center,scale)
        self.imputer=SimpleImputer(strategy='median')
        train_x=self.imputer.fit_transform(x)
        self.scaler=RobustScaler().fit(train_x)
        self.forest=IsolationForest(n_estimators=180,max_samples='auto',
                                    contamination='auto',random_state=42,n_jobs=-1)
        self.forest.fit(self.scaler.transform(train_x))
        self.calibration={}
        self.site_cal={}
        for method in METHODS:
            scores=self.score(calibration, method)
            self.calibration[method]=scores[np.isfinite(scores)]
            if len(self.calibration[method])<20:
                raise ValueError('Too few calibration rows with usable observations')
            if method=='sentinel_station':
                stations=calibration['station'].astype(str).to_numpy()
                for site in np.unique(stations):
                    site_scores=scores[stations==site]
                    site_scores=site_scores[np.isfinite(site_scores)]
                    if site in self.local and len(site_scores)>=self.min_site_cal:
                        self.site_cal[site]=site_scores
        return self

    def scores_and_coverage(self, frame, method):
        if method not in METHODS:
            raise ValueError(f'Unknown method {method}')
        x=numeric(frame,self.features)
        observed=np.isfinite(x)
        coverage=observed.mean(axis=1)
        insufficient=coverage < .5
        if method=='isolation_forest':
            xx=self.imputer.transform(x)
            scores=-self.forest.score_samples(self.scaler.transform(xx))
        elif method=='robust_global':
            z=np.abs((x-self.global_median)/self.global_scale)
            z=np.where(observed,z,-np.inf)
            scores=np.max(z,axis=1)
        else:
            scores=np.empty(len(x),float)
            sites=frame['station'].astype(str).to_numpy()
            for idx,(row,mask,site) in enumerate(zip(x,observed,sites)):
                center,scale=self.local.get(site,(self.global_median,self.global_scale))
                deviations=np.abs((row-center)/scale)
                z=np.where(mask,deviations,-np.inf)
                # Missingness penalty is fixed beforehand, NOT tuned on held-out cases.
                scores[idx]=np.max(z)+0.65*(1-mask.mean())
        # Scores on insufficiently observed rows are deliberately withheld.
        scores=np.asarray(scores,float)
        scores[insufficient]=np.nan
        return scores,coverage,insufficient

    def score(self, frame,method):
        return self.scores_and_coverage(frame,method)[0]

    def pvalues(self,frame,method):
        scores=self.score(frame,method)
        p=np.full(len(scores),np.nan,float)
        sites=frame['station'].astype(str).to_numpy()
        valid=np.isfinite(scores)
        for site in np.unique(sites[valid]):
            mask=(sites==site)&valid
            reference=(self.site_cal.get(site,self.calibration[method])
                       if method=='sentinel_station' else self.calibration[method])
            sorted_ref=np.sort(reference)
            p[mask]=(1+len(reference)-np.searchsorted(sorted_ref,scores[mask],side='left'))/(len(reference)+1)
        return p

    def alerts(self,frame,method,alpha=0.05):
        if not 0 < alpha < 1:
            raise ValueError('alpha must be between zero and one')
        p=self.pvalues(frame,method)
        _, coverage, insufficient=self.scores_and_coverage(frame,method)
        alert=np.isfinite(p)&(p <= alpha)
        return pd.DataFrame({'station':frame['station'].astype(str).to_numpy(),
                             'p_value':p,'anomaly_alert':alert,
                             'sensor_data_insufficient':insufficient,
                             'observation_fraction':coverage})
