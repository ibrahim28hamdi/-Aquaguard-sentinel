# Benchmark design and interpretation

1. Records are separated into **60% training, 20% calibration, and 20% holdout**, sorted by timestamp when available. Otherwise, original source record order is preserved, and no chronological assumption is made.
2. The stable project compares three methods: robust median/IQR deviation (`robust_global`), `IsolationForest`, and station-aware robust scores (`sentinel_station`). With a single station, station-aware scoring may coincide with the global reference method.
3. Methods obtain empirical rank p-values from the calibration partition at a prespecified alpha of 0.05. Exchangeability is not guaranteed for time-series data, so this is not proof of a 5% real-world false-alarm rate.
4. Simulated disturbances include one-feature artificial spikes at 1.5 / 3 / 5 IQR, gradual single-sensor drift, and missing sensor values. These simulate sensor faults, **not verified pollution events**.
5. `mean_alert_rate` on untouched holdout data means *fraction flagged*, NOT ground-truth false-positive rate. `mean_new_alert_rate` on artificial spikes means new anomalies raised relative to the unmodified row, NOT real-contamination sensitivity.
6. Report negative results; do not post-hoc optimize on the inspected holdout data or assert that the station-aware method is better without controlled evidence.

## Previously run UCI experiment (exploratory, not independent validation)

Reported from a local run on the public UCI Water Treatment Plant dataset, with 527 records, 38 numeric measurement columns, and one site. The split was 316 training / 105 calibration / 106 holdout. At 5-IQR artificial one-feature spikes (five seeds, 21 modified records per trial), the reported mean **new-alert rate** was 0.0% for Isolation Forest and about 0.95% for both robust global and station-aware methods. Untouched holdout alert rate: about 2.83%, 1.89%, and 1.89%, respectively. These are exploratory outcomes, not positive claims of effectiveness.

The source summary and run manifest are included in `benchmarks/`; real measurements are **not** included. V5 EFT feature-tail experiments were repaired separately in Kiro but were not included in this stable release because the reported improvement was small and inconclusive on the repeatedly inspected holdout. No new method is being claimed as research novelty.
