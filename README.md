# ctr-calibration

**Calibration Decay under Temporal Drift in CTR Prediction: An Empirical Study**

> In click-through-rate (CTR) prediction, the predicted probability *is* the bid: pCTR is
> multiplied directly by advertiser value to price an impression. This project measures how
> the calibration of CTR models degrades as test data moves further in time from the training
> window, and how much of that degradation standard post-hoc calibration methods recover —
> i.e., what is the **shelf life of a fitted calibrator**?

## Why calibration, not (just) AUC

1. **pCTR is a price, not a score.** In performance advertising the bid for an impression is
   computed as `bid = pCTR × advertiser_value`. The predicted probability is multiplied
   directly into money, so its *scale* matters, not only its ordering.

2. **AUC grades ranking; calibration grades the scale.** Multiply every prediction by 0.1 and
   AUC does not move — yet every bid is now 10× wrong. A model can be an excellent ranker and
   a terrible probability estimator at the same time. These are separate abilities, and only
   the second one prices bids.

3. **Calibration is verified on groups, never on single predictions.** A single impression
   only reveals a 0/1 outcome, so "was 0.03 correct?" is unanswerable per-impression. But
   among 100K impressions predicted at 0.03, roughly 3K should be clicked. Expected
   Calibration Error (ECE) formalizes this by binning predictions and comparing predicted
   probability against empirical click frequency per bin (visualized as a reliability
   diagram).

4. **Post-hoc calibration is a feedback loop that assumes stationarity.** In production one
   scores yesterday's predictions against yesterday's clicks, fits a corrector (e.g., a
   temperature), and applies it to today's traffic. This silently assumes that yesterday's
   miscalibration is still today's miscalibration.

5. **Ad traffic drifts, so that assumption decays.** New campaigns launch, user mixes shift,
   weekday becomes weekend. A frozen model keeps emitting probabilities on yesterday's scale
   while the world's empirical frequencies move — a miscalibrated probability is a mispriced
   bid, continuously.

**Positioning.** "Calibration drift" is a named phenomenon in clinical ML (Davis et al.,
2020), while the CTR-under-drift literature measures accuracy (AUC/LogLoss), not calibration.
We bring the calibration-drift lens to CTR prediction — the domain where miscalibration has a
direct monetary interpretation — and provide a systematic public-benchmark measurement of
calibration decay under *real* (not synthetic) temporal drift. See `references.bib`.

## Pre-registered hypotheses

Registered **before** running any experiment (started 2026-08-21). Actual outcomes will be
recorded against each hypothesis; a falsified hypothesis is a finding, not a failure.

- **H1.** DeepFM beats logistic regression on AUC, but is *worse* calibrated (higher ECE)
  before post-hoc correction.
- **H2.** ECE increases monotonically with temporal distance from the training window, for
  all models.
- **H3.** Post-hoc calibration fitted on the validation day decays in effectiveness on later
  test days — calibration itself drifts.

## Data

[Kaggle Avazu CTR Prediction](https://www.kaggle.com/c/avazu-ctr-prediction) — `train.csv`,
~40M rows covering 10 consecutive days of mobile ad impressions.

- Label: `click`. Temporal axis: `hour` (format `YYMMDDHH`).
- All predictive features are categorical (`site_*`, `app_*`, `device_*`, `banner_pos`,
  anonymized `C1, C14–C21`) → feature hashing + embeddings downstream.
- On first load the raw CSV is converted to **parquet partitioned by day**
  (`data/parquet/day=DD/`); the raw CSV is never re-read afterwards.

```bash
pip install -r requirements.txt
# requires a configured Kaggle API token (~/.kaggle/kaggle.json)
# and accepted competition rules for avazu-ctr-prediction
bash data/download.sh
```

## Split design: temporal only, never random

In production, a model always predicts a future it has never seen — it is trained up to day
T and serves on day T+1 onward. A random split leaks the future into training: the model
learns from day-10 logs and is then tested on day-10 logs, so train and test distributions
are effectively identical (temporal leakage). Calibration measured this way answers "how good
is the scale when the world hasn't changed?" — an optimistically contaminated number that
production never gets to enjoy. Worse, shuffling destroys the time axis itself: our target
figure is **ECE as a function of days-since-training**, and that x-axis is undefined if test
days are mixed. Random splitting doesn't just bias the measurement — it erases the phenomenon
we are measuring.

| Split | Days | Role |
|---|---|---|
| Train | 1–8 | model fitting |
| Validation | 9 | model selection **and** fitting post-hoc calibrators |
| Test | 10 | evaluated per-day and per-hour-block slices → ECE-vs-time curves |

Stress variant (longer drift horizon): train on days 1–5, test on days 6–10.

## Experiment matrix (Phase 1)

| Axis | Values |
|---|---|
| Model | Logistic Regression (hashed features) · DeepFM (PyTorch) |
| Calibration | None · Temperature scaling · Isotonic regression |

- Metrics are always reported in pairs: **AUC + LogLoss** (ranking/fit) and
  **ECE + reliability diagram** (calibration), plus the predicted-CTR / empirical-CTR ratio
  per time slice as a bid-bias proxy.
- 3 seeds per configuration; mean ± std. One config file per experiment; all runs append to
  a single `results/results.csv`.

## Repository layout

```
ctr-calibration/
├── README.md
├── CITATION.cff             # citation metadata (GitHub "Cite this repository")
├── references.bib           # related work, grows with the paper
├── configs/                 # one yaml per experiment        (Week 1+)
├── data/                    # gitignored; script only
│   └── download.sh          # kaggle download + day-partitioned parquet
├── notebooks/
│   └── 01_eda_temporal.ipynb  # per-day volume/CTR, per-day feature cardinality
├── src/                     # data, models, calibrate, metrics, train   (Week 1+)
├── paper/                   # LaTeX draft                    (later)
└── results/                 # results.csv + figures
```

## Status

Phase 1 — empirical study on Avazu. Phase 2 (planned): online recalibration under drift /
constant-memory streaming calibration monitoring.

## Citation

If you use this repository, please cite it via the metadata in [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
