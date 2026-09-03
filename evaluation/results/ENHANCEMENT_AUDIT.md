# Enhancement Audit
- **Current State**: Evaluates RF, LR, and Majority baselines on classification and RUL. GroupKFold implemented. Latency implemented. SHAP implemented. Pytest integration for API and preprocessing.
- **Weak Areas**: Missing Brier score, calibration analysis. Missing threshold tuning on validation set. Missing temporal split comparison. Error analysis (false negatives) is not dumped to a file. Missing testing on anomaly detection.
- **Actions Required**: Rewrite eval_classification, eval_rul, eval_robustness (to include temporal split). Add calibration and threshold logic. Add Brier score.
