"""Predeclared stress tests. Not accuracy on verified contamination events."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .loaders import load_nyc, load_uci, load_usgs, load_custom, synthetic_demo
from .model import Sentinel, METHODS, numeric, split_three


def inject_spikes(test,model,seed=41,fraction=.2,strength=3.):
    """Add artificial errors to randomly selected EXISTING measurements."""
    rng=np.random.default_rng(seed)
    corrupted=test.copy()
    # Explicit float conversion avoids dtype-incompatible writes on integer columns.
    for column in model.features:
        corrupted[column] = pd.to_numeric(corrupted[column], errors='coerce').astype(float)
    x=numeric(corrupted,model.features)
    possible=np.flatnonzero(np.isfinite(x).sum(axis=1)>0)
    if not len(possible):
        raise ValueError('No observations eligible for perturbation')
    rows=rng.choice(possible,max(1,int(len(possible)*fraction)),replace=False)
    for i in rows:
        valid=np.flatnonzero(np.isfinite(x[i]))
        j=int(rng.choice(valid))
        col=model.features[j]
        corrupted.at[corrupted.index[i],col] = x[i,j] + float(rng.choice((-1.,1.)))*strength*model.global_scale[j]
    return corrupted,rows


def inject_missingness(test,model,seed=22,fraction=.2):
    """Simulate instrument loss (distinct from toxic contaminant events)."""
    rng=np.random.default_rng(seed)
    result=test.copy()
    # Missing values require a nullable numeric dtype, even for integer inputs.
    for column in model.features:
        result[column] = pd.to_numeric(result[column], errors='coerce').astype(float)
    xs=numeric(test,model.features)
    possible=np.flatnonzero(np.isfinite(xs).sum(axis=1)>=2)
    if len(possible)==0:
        raise ValueError('Not enough observed values for missingness experiment')
    rows=rng.choice(possible,max(1,int(len(possible)*fraction)),replace=False)
    for i in rows:
        observed=np.flatnonzero(np.isfinite(xs[i]))
        drop_n=max(1,int(np.ceil(len(model.features)*.65)))
        to_drop=rng.choice(observed,min(drop_n,len(observed)),replace=False)
        for j in to_drop:
            result.at[result.index[i],model.features[j]]=np.nan
    return result,rows


def inject_drift(test,model,strength=2.):
    """Gradually shift one sensor over the full held-out period; artificial scenario."""
    result=test.copy()
    x=numeric(test,model.features)
    viable=np.flatnonzero(np.isfinite(x).mean(axis=0)>.8)
    if not len(viable):
        raise ValueError('No sufficiently observed feature for drift stress test')
    j=int(viable[0])
    delta=np.linspace(0,strength*model.global_scale[j],len(test))
    result[model.features[j]]=pd.to_numeric(result[model.features[j]],errors='coerce')+delta
    return result


def test_benchmark(raw:pd.DataFrame, *, dataset:str, alpha=.05,
                   seeds=(11,22,33,44,55), strengths=(1.5,3.,5.)):
    if 'station' not in raw:
        raw=raw.assign(station='unknown')
    if 'water_type' not in raw:
        raw=raw.assign(water_type='unspecified')
    train,cal,test,ordering=split_three(raw)
    model=Sentinel().fit(train,cal)
    baseline={m:model.alerts(test,m,alpha) for m in METHODS}
    trials=[]
    for method in METHODS:
        base=baseline[method]['anomaly_alert'].to_numpy(dtype=bool)
        valid=~baseline[method]['sensor_data_insufficient'].to_numpy(dtype=bool)
        trials.append({'dataset':dataset,'method':method,'stress':'untouched_heldout',
                       'strength_iqr':0.,'seed':-1,'count':int(valid.sum()),
                       'alert_rate':float(base[valid].mean()) if valid.any() else np.nan,
                       'new_alert_rate':np.nan,'insufficient_readings_rate':float((~valid).mean())})
        for strength in strengths:
            for seed in seeds:
                altered,rows=inject_spikes(test,model,seed=seed,strength=strength)
                now=model.alerts(altered,method,alpha)
                alerted=now['anomaly_alert'].to_numpy(dtype=bool)
                eligible=~base[rows]
                trials.append({'dataset':dataset,'method':method,'stress':'artificial_spike',
                               'strength_iqr':strength,'seed':seed,'count':len(rows),
                               'alert_rate':float(alerted[rows].mean()),
                               'new_alert_rate':float(alerted[rows][eligible].mean()) if eligible.any() else np.nan,
                               'insufficient_readings_rate':float(now['sensor_data_insufficient'].iloc[rows].mean())})
        drift=inject_drift(test,model)
        drift_flags=model.alerts(drift,method,alpha)
        trials.append({'dataset':dataset,'method':method,'stress':'artificial_gradual_drift',
                       'strength_iqr':2.,'seed':-1,'count':len(test),
                       'alert_rate':float(drift_flags['anomaly_alert'].mean()),
                       'new_alert_rate':float(drift_flags.loc[~base,'anomaly_alert'].mean()) if (~base).any() else np.nan,
                       'insufficient_readings_rate':float(drift_flags['sensor_data_insufficient'].mean())})
        missing,rows=inject_missingness(test,model)
        missing_flags=model.alerts(missing,method,alpha)
        trials.append({'dataset':dataset,'method':method,'stress':'artificial_sensor_dropout',
                       'strength_iqr':0.,'seed':22,'count':len(rows),
                       'alert_rate':float(missing_flags['anomaly_alert'].iloc[rows].mean()),
                       'new_alert_rate':np.nan,
                       'insufficient_readings_rate':float(missing_flags['sensor_data_insufficient'].iloc[rows].mean())})
    trial_df=pd.DataFrame(trials)
    summary=trial_df.groupby(['dataset','method','stress','strength_iqr'],dropna=False,as_index=False).agg(
        trials=('seed','count'), evaluated_rows_each_trial=('count','min'),
        mean_alert_rate=('alert_rate','mean'),
        mean_new_alert_rate=('new_alert_rate','mean'),
        mean_insufficient_readings_rate=('insufficient_readings_rate','mean'))
    metadata={
        'dataset':dataset,'total_records':len(raw),'train':len(train),'calibration':len(cal),'holdout':len(test),
        'order_basis':ordering,'features':model.features,'alpha':alpha,
        'sites_train':int(train['station'].nunique()),'sites_calibration':int(cal['station'].nunique()),
        'sites_heldout':int(test['station'].nunique()),
        'local_profiles_fit':len(model.local), 'site_calibrators_fit':len(model.site_cal),
        'train_only_preprocessing':True, 'all_perturbations_are_artificial':True,
        'verified_contamination_labels':False,
        'claim_limitations':[ 'No measurement of real contamination detection accuracy',
                              'Untouched historical alert rate is NOT a false positive rate without labels',
                              'Conformal calibration under temporal shift lacks finite-sample coverage guarantee',
                              'Station-specific inference requires enough training and calibration records',
                              'Ablation study and expert validation remain necessary before claiming novelty']}
    return summary,trial_df,metadata


def save_report(summary,trials,metadata,directory):
    out=Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out/'summary.csv',index=False)
    trials.to_csv(out/'trials.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        subset=summary[(summary.stress=='artificial_spike') & (summary.strength_iqr==3.)]
        if not subset.empty:
            ax=subset.plot(kind='bar',x='method',y='mean_new_alert_rate',legend=False,
                           title='Artificial spike detection on previously unflagged rows (3 IQR)')
            ax.set_ylabel('New-alert rate (simulated errors only)')
            ax.set_ylim(0,1)
            ax.figure.tight_layout()
            ax.figure.savefig(out/'benchmark.png',dpi=160)
            plt.close(ax.figure)
    except ImportError:
        pass
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description='AquaGuard Sentinel: honest sensor-disturbance benchmarks')
    parser.add_argument('--dataset',choices=['demo','uci','nyc','usgs','csv'],default='demo')
    parser.add_argument('--csv',default=None)
    parser.add_argument('--water-type',default='custom')
    parser.add_argument('--nyc-rows',type=int,default=30000)
    parser.add_argument('--usgs-site',default='01482800')
    parser.add_argument('--usgs-start',default='2025-06-01')
    parser.add_argument('--usgs-end',default='2025-07-01')
    parser.add_argument('--out',default=None)
    args=parser.parse_args(argv)
    if args.dataset=='demo': frame=synthetic_demo()
    elif args.dataset=='uci': frame=load_uci()
    elif args.dataset=='nyc': frame=load_nyc(max_rows=args.nyc_rows)
    elif args.dataset=='usgs': frame=load_usgs(args.usgs_site,args.usgs_start,args.usgs_end)
    else:
        if args.csv is None: parser.error('--csv PATH required for dataset csv')
        frame=load_custom(args.csv,water_type=args.water_type)
    summary,trials,metadata=test_benchmark(frame,dataset=args.dataset)
    out=save_report(summary,trials,metadata,args.out or f'runs/{args.dataset}')
    print('Exported',out.resolve())
    print('Records / features',len(frame),len(metadata['features']))
    print(summary[['method','stress','strength_iqr','mean_alert_rate','mean_new_alert_rate']].to_string(index=False))
    if args.dataset=='demo':
        print('ALL MEASUREMENTS ARE SYNTHETIC. These results cannot support environmental conclusions.')
    else:
        print('REAL MEASUREMENTS + ARTIFICIALLY INJECTED FAULTS. Not pollution-detection accuracy.')


if __name__=='__main__':
    main()
