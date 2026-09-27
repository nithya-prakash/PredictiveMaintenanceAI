# Models & evaluation

## Data

NASA C-MAPSS FD001 (Saxena et al. 2008, NASA Prognostics Center of Excellence): simulated turbofan engines, one operating condition, one fault mode (HPC degradation). 100 training engines run until failure; 100 test engines stop before failure, and NASA provides their true RUL at the last cycle. The data is downloaded and checksum-verified by `ml/data/cmapss.py`. The 7 sensors that are constant in FD001 are dropped; the RUL target is capped at 125 cycles, the standard piecewise-linear convention.

## Features

Raw values of the 14 sensors, the cycle count, and rolling means and standard deviations over the last 5 and 15 cycles, per engine: 71 features. Only cycles with a full 15-cycle history are used; the API requires 15+ readings, so served and training features are identical (tested).

## Training protocol

- 5-fold `GroupKFold` by engine on the training engines; Optuna tunes each candidate (8 trials).
- RUL candidates: linear regression, Random Forest, HistGradientBoosting. Selected by cross-validated RMSE → **Random Forest** (CV RMSE 16.93).
- Failure candidates: logistic regression, HistGradientBoosting (both class-balanced). Selected by cross-validated PR-AUC → **logistic regression** (CV PR-AUC 0.966). The decision threshold (0.80) maximises F1 on out-of-fold predictions.
- Anomaly: IsolationForest fitted on cycles with true RUL > 125; flags scores below the 1st percentile of those healthy cycles.
- The official test set is never loaded during training (a test enforces this).

## Results on the official test set

See the full report in the repository: `evaluation/results/FINAL_REPORT.md`.

| RUL model | RMSE | NASA score |
|---|---|---|
| Random Forest (deployed) | 17.19 | 573 |
| HistGradientBoosting | 17.35 | 580 |
| Linear regression | 19.31 | 718 |
| Mean baseline | 41.21 | 25,451 |
| *Published (Ragab et al. 2020): Random Forest / 1D CNN* | *17.91 / 12.61* | *480 / 274* |

| Failure model | PR-AUC | Precision | Recall |
|---|---|---|---|
| Logistic regression (deployed, 0.80) | 0.842 | 0.770 | 0.717 |
| HistGradientBoosting (0.50) | 0.820 | 0.665 | 0.777 |

Anomaly flag rate by true RUL: 1.2% (> 125), 8.4% (61–125), 55.5% (31–60), 99.1% (≤ 30).

## Explainability

`backend/services/xai.py` uses the SHAP explainer that matches the deployed failure model (`LinearExplainer` for logistic regression, `TreeExplainer` for tree models). Contributions are in log-odds of failure within 30 cycles. On the test set the most influential features are rolling means of `s11` (HPC outlet static pressure), `s14` (corrected core speed), `s4` (LPT outlet temperature) and `s17` (bleed enthalpy).

## Limitations

FD001 only; simulated engines; classical models (deep sequence models reach about 12–13 RMSE); failure metrics rest on 25 test engines that approach failure; the recommendation is a transparent rule, not a learned policy.
