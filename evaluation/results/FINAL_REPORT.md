# ML Evaluation & Benchmarking Final Report

## 1. Executive Summary
This report evaluates the Predictive Maintenance AI platform following strict integrity rules. 

## 2. Dataset Audit & Leakage
- **Split Strategy**: Machine-wise holdout (Machines 1-120 Train, 121-150 Test).
- **Preprocessing Leakage Check**: StandardScaler and Threshold Tuning strictly fitted on training/validation subsets.

## 3. Classification Results (30-Cycle Imminent Failure)
Threshold 0.8000000000000003 selected via validation set tuning.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Brier Score |
|---|---|---|---|---|---|---|
| **Random Forest** | 0.999 | 0.992 | 0.950 | 0.957 | 0.942 | 0.014 |
| **Logistic Regression** | 1.000 | 0.997 | 0.967 | 0.973 | 0.961 | 0.008 |
| **Majority Baseline** | 0.500 | 0.132 | 0.000 | 0.000 | 0.000 | 0.132 |

### Error Analysis (False Negatives)
- Missed Failures: 54
- Mean Probability assigned to missed failures: 0.629

## 4. Remaining Useful Life (RUL) Results
| Model | RMSE | MAE | Median AE | R² |
|---|---|---|---|---|
| **Random Forest** | 9.72 | 6.89 | 4.81 | 0.981 |
| **Linear Regression** | 15.77 | 12.74 | 11.72 | 0.951 |
| **Mean Baseline** | 71.30 | 60.84 | 58.75 | -0.004 |

### RUL Error by Range
- Low RUL (<=30 cycles) RMSE: 3.06
- Medium RUL (31-100 cycles) RMSE: 5.44
- High RUL (>100 cycles) RMSE: 12.18

## 5. Cross-Validation & Robustness (GroupKFold)
- **ROC-AUC**: 0.998 ± 0.000
- **PR-AUC**: 0.990 ± 0.002
- **Temporal Split**: The synthetic dataset generates machines independently without a global timestamp column tracking chronological factory deployment. Hence, chronological TimeSeriesSplit across machines is not physically meaningful for this specific synthetic generation logic.

## 6. Inference Latency (Full Pipeline)
Local execution, 100 sequential requests, 10 warmup calls. Simulates preprocessing + scaling + inference overhead.
- Mean: 48.52 ms
- p99: 68.21 ms

## 7. SHAP Explainability
- **Top 5 Features**: vibration_roll_mean_15, temperature_roll_mean_15, current_roll_mean_15, vibration_roll_mean_5, temperature_roll_mean_5
- **Note**: Quantitative SHAP evaluation (e.g. fidelity/stability metrics) requires specialized perturbation testing which is beyond the current scope. These represent global feature attribution magnitudes on a test sample.
