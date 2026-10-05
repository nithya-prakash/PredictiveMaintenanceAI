# Evaluation report: NASA C-MAPSS FD001

Test set: NASA's official FD001 test set, 100 engines. All model choices (algorithm, hyperparameters, decision threshold) were made with engine-wise cross-validation on the training engines; the test engines were used only for this report.

## Remaining useful life (official protocol: last cycle of each test engine, RUL capped at 125)

| Model | RMSE | MAE | NASA score | CV RMSE (train) |
|---|---|---|---|---|
| LinearRegression | 19.31 | 14.91 | 718 | 19.22 |
| RandomForestRegressor (deployed) | 17.19 | 12.22 | 573 | 16.93 |
| HistGradientBoostingRegressor | 17.35 | 12.44 | 580 | 17.05 |
| Mean baseline | 41.21 | 34.85 | 25451 | – |

Published FD001 results under the same capped-RUL protocol, for context (Ragab et al. 2020, Table III, arXiv:2007.09868):

| Published method | RMSE | NASA score |
|---|---|---|
| Random Forest | 17.91 | 480 |
| Gradient Boosting | 15.67 | 474 |
| 1D CNN (Li et al.) | 12.61 | 274 |
| Deep LSTM | 16.14 | 338 |
| ATS2S (Ragab et al.) | 12.63 | 243 |

RMSE of the deployed model over all test cycles, by true RUL: true RUL <= 30: 18.4, 31-60: 25.2, 61-125: 17.3, > 125: 14.3

Conformal 90% interval (+/-31.2 cycles, from training-engine out-of-fold residuals). Empirical test coverage: 91% at each engine's last cycle, 91% over all test cycles.

## Imminent failure (true RUL <= 30 cycles), all test cycles with 15+ cycles of history

332 positive and 11364 negative cycles, from 25 engines that come within 30 cycles of failure. Consecutive cycles of one engine are correlated, so treat these as 100 engines' worth of evidence, not 11696 independent samples.

| Model | PR-AUC | ROC-AUC | Threshold | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|---|
| LogisticRegression (deployed) | 0.842 | 0.994 | 0.80 | 0.770 | 0.717 | 0.743 | 0.013 |
| HistGradientBoostingClassifier | 0.820 | 0.991 | 0.50 | 0.665 | 0.777 | 0.717 | 0.013 |
| Prior baseline | 0.028 | 0.500 | 0.50 | 0.000 | 0.000 | 0.000 | 0.045 |

The deployed threshold (0.80) was chosen on out-of-fold training predictions and is exactly the one the API uses.

## Anomaly flag rate by true RUL (should rise towards failure)

| True RUL | Cycles | Flagged |
|---|---|---|
| true RUL <= 30 | 332 | 99.1% |
| 31-60 | 908 | 55.5% |
| 61-125 | 3904 | 8.4% |
| > 125 | 6552 | 1.2% |

## SHAP: most influential features for the failure model (mean |SHAP| on 300 test cycles)

- `s11_roll_mean_5`: 2.759
- `s14_roll_mean_15`: 2.206
- `s4_roll_mean_15`: 1.856
- `s17_roll_mean_15`: 1.824
- `s12_roll_mean_5`: 1.745
- `s8_roll_mean_5`: 1.696
- `s4_roll_mean_5`: 1.616
- `s20_roll_mean_15`: 1.596

## Latency

200 sequential requests, 30 readings each, FastAPI TestClient (full HTTP stack in-process, no network), after 10 warm-up calls; laptop CPU. Not a production load test.

- `/api/v1/predict-failure`: p50 9.1 ms, p95 9.3 ms, p99 10.1 ms
- `/api/v1/explain`: p50 9.3 ms, p95 12.8 ms, p99 13.8 ms

Model bundle trained 2026-09-27T11:32:43 UTC on NASA C-MAPSS FD001 (training engines only).
