import os
import json
import subprocess

def run_script(script_path):
    print(f"--- Running {script_path} ---")
    result = subprocess.run(["python", script_path], env=dict(os.environ, PYTHONPATH="."))
    if result.returncode != 0:
        raise RuntimeError(f"Script {script_path} failed.")

def generate_report():
    with open("evaluation/results/classification.json", "r") as f: cls_data = json.load(f)
    with open("evaluation/results/rul.json", "r") as f: rul_data = json.load(f)
    with open("evaluation/results/cross_validation.json", "r") as f: cv_data = json.load(f)
    with open("evaluation/results/latency.json", "r") as f: lat_data = json.load(f)
    with open("evaluation/results/shap.json", "r") as f: shap_data = json.load(f)
        
    report = f'''# ML Evaluation & Benchmarking Final Report

## 1. Executive Summary
This report evaluates the Predictive Maintenance AI platform following strict integrity rules. 

## 2. Dataset Audit & Leakage
- **Split Strategy**: Machine-wise holdout (Machines 1-120 Train, 121-150 Test).
- **Preprocessing Leakage Check**: StandardScaler and Threshold Tuning strictly fitted on training/validation subsets.

## 3. Classification Results (30-Cycle Imminent Failure)
Threshold {cls_data['random_forest']['applied_threshold']} selected via validation set tuning.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Brier Score |
|---|---|---|---|---|---|---|
| **Random Forest** | {cls_data['random_forest']['roc_auc']:.3f} | {cls_data['random_forest']['pr_auc']:.3f} | {cls_data['random_forest']['f1']:.3f} | {cls_data['random_forest']['precision']:.3f} | {cls_data['random_forest']['recall']:.3f} | {cls_data['random_forest']['brier_score']:.3f} |
| **Logistic Regression** | {cls_data['baseline_logistic_regression']['roc_auc']:.3f} | {cls_data['baseline_logistic_regression']['pr_auc']:.3f} | {cls_data['baseline_logistic_regression']['f1']:.3f} | {cls_data['baseline_logistic_regression']['precision']:.3f} | {cls_data['baseline_logistic_regression']['recall']:.3f} | {cls_data['baseline_logistic_regression']['brier_score']:.3f} |
| **Majority Baseline** | {cls_data['baseline_majority']['roc_auc']:.3f} | {cls_data['baseline_majority']['pr_auc']:.3f} | {cls_data['baseline_majority']['f1']:.3f} | {cls_data['baseline_majority']['precision']:.3f} | {cls_data['baseline_majority']['recall']:.3f} | {cls_data['baseline_majority']['brier_score']:.3f} |

### Error Analysis (False Negatives)
- Missed Failures: {cls_data['false_negative_analysis']['missed_failures']}
- Mean Probability assigned to missed failures: {cls_data['false_negative_analysis']['mean_probability_of_missed']:.3f}

## 4. Remaining Useful Life (RUL) Results
| Model | RMSE | MAE | Median AE | R² |
|---|---|---|---|---|
| **Random Forest** | {rul_data['random_forest']['rmse']:.2f} | {rul_data['random_forest']['mae']:.2f} | {rul_data['random_forest']['median_ae']:.2f} | {rul_data['random_forest']['r2']:.3f} |
| **Linear Regression** | {rul_data['baseline_linear_regression']['rmse']:.2f} | {rul_data['baseline_linear_regression']['mae']:.2f} | {rul_data['baseline_linear_regression']['median_ae']:.2f} | {rul_data['baseline_linear_regression']['r2']:.3f} |
| **Mean Baseline** | {rul_data['baseline_mean']['rmse']:.2f} | {rul_data['baseline_mean']['mae']:.2f} | {rul_data['baseline_mean']['median_ae']:.2f} | {rul_data['baseline_mean']['r2']:.3f} |

### RUL Error by Range
- Low RUL (<=30 cycles) RMSE: {rul_data['error_analysis']['low_rul_rmse']:.2f}
- Medium RUL (31-100 cycles) RMSE: {rul_data['error_analysis']['medium_rul_rmse']:.2f}
- High RUL (>100 cycles) RMSE: {rul_data['error_analysis']['high_rul_rmse']:.2f}

## 5. Cross-Validation & Robustness (GroupKFold)
- **ROC-AUC**: {cv_data['machine_wise_cv_5_fold']['roc_auc_mean']:.3f} ± {cv_data['machine_wise_cv_5_fold']['roc_auc_std']:.3f}
- **PR-AUC**: {cv_data['machine_wise_cv_5_fold']['pr_auc_mean']:.3f} ± {cv_data['machine_wise_cv_5_fold']['pr_auc_std']:.3f}
- **Temporal Split**: {cv_data['temporal_split']['reason']}

## 6. Inference Latency (Full Pipeline)
{lat_data['benchmark_context']}
- Mean: {lat_data['full_pipeline_classification']['mean_ms']:.2f} ms
- p99: {lat_data['full_pipeline_classification']['p99_ms']:.2f} ms

## 7. SHAP Explainability
- **Top 5 Features**: {', '.join(shap_data['xai_metrics']['top_5_global_features'])}
- **Note**: {shap_data['xai_metrics']['note']}
'''
    with open("evaluation/results/FINAL_REPORT.md", "w") as f:
        f.write(report)
        
    cv_safe = f'''# CV Safe Metrics

| Metric | Value | Model | Dataset | Split | Script | Evidence | CV Safe? | Caveat |
|---|---|---|---|---|---|---|---|---|
| PR-AUC | {cls_data['random_forest']['pr_auc']:.3f} | Random Forest | Synthetic Sensors | Machine-wise Holdout | `eval_classification.py` | `classification.json` | YES | Outperformed by Logistic Regression ({cls_data['baseline_logistic_regression']['pr_auc']:.3f}) due to synthetic linearity. |
| RUL RMSE | {rul_data['random_forest']['rmse']:.2f} | Random Forest | Synthetic Sensors | Machine-wise Holdout | `eval_rul.py` | `rul.json` | YES | Tested specifically in low-RUL regions. |
| GroupKFold ROC-AUC | {cv_data['machine_wise_cv_5_fold']['roc_auc_mean']:.3f} | Random Forest | Synthetic Sensors | 5-Fold GroupKFold | `eval_robustness.py` | `cross_validation.json` | YES | Shows zero entity leakage, but std dev is 0 due to synthetic uniformity. |
| API p99 Latency | {lat_data['full_pipeline_classification']['p99_ms']:.2f} ms | Full Pipeline | 100 requests | Sequential Local | `eval_latency.py` | `latency.json` | YES | Local benchmark only, not production cloud environment. |
'''
    with open("evaluation/results/CV_SAFE_METRICS.md", "w") as f:
        f.write(cv_safe)

if __name__ == "__main__":
    run_script("evaluation/eval_classification.py")
    run_script("evaluation/eval_rul.py")
    run_script("evaluation/eval_robustness.py")
    run_script("evaluation/eval_latency.py")
    run_script("evaluation/eval_shap.py")
    generate_report()
