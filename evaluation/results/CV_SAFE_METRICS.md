# CV Safe Metrics

| Metric | Value | Model | Dataset | Split | Script | Evidence | CV Safe? | Caveat |
|---|---|---|---|---|---|---|---|---|
| PR-AUC | 0.997 | LogisticRegression (deployed) | Synthetic Sensors | Machine-wise Holdout | `eval_classification.py` | `classification.json` | YES | RandomForest scored 0.991 PR-AUC in the same benchmark; the pipeline deploys whichever wins, not RandomForest by default. |
| RUL RMSE | 9.72 | Random Forest | Synthetic Sensors | Machine-wise Holdout | `eval_rul.py` | `rul.json` | YES | Tested specifically in low-RUL regions. |
| GroupKFold ROC-AUC | 0.998 | Random Forest | Synthetic Sensors | 5-Fold GroupKFold | `eval_robustness.py` | `cross_validation.json` | YES | Shows zero entity leakage, but std dev is 0 due to synthetic uniformity. |
| API p99 Latency | 21.26 ms | Full Pipeline | 100 requests | Sequential Local | `eval_latency.py` | `latency.json` | YES | Local benchmark only, not production cloud environment. |
