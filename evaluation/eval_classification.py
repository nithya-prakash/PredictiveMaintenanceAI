import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss, balanced_accuracy_score

def evaluate_classification():
    print("Evaluating Classification Model...")
    
    # Load Data
    test_path = "datasets/test_data.csv"
    train_path = "datasets/train_data.csv"
    df_test = pd.read_csv(test_path)
    df_train = pd.read_csv(train_path)
    
    preprocessor = joblib.load("models/preprocessor.pkl")
    df_test_feat = preprocessor.add_rolling_features(df_test)
    df_train_feat = preprocessor.add_rolling_features(df_train)
    
    # Use the last 20% of training data as a validation set for threshold tuning
    train_machines = df_train_feat['machine_id'].unique()
    val_machines = train_machines[-int(len(train_machines)*0.2):]
    
    df_val_feat = df_train_feat[df_train_feat['machine_id'].isin(val_machines)].copy()
    df_train_sub_feat = df_train_feat[~df_train_feat['machine_id'].isin(val_machines)].copy()
    
    # Fit scaler strictly on train_sub
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_train_sub_feat[preprocessor.feature_cols] = scaler.fit_transform(df_train_sub_feat[preprocessor.feature_cols])
    df_val_feat[preprocessor.feature_cols] = scaler.transform(df_val_feat[preprocessor.feature_cols])
    df_test_feat[preprocessor.feature_cols] = scaler.transform(df_test_feat[preprocessor.feature_cols])
    
    X_val = df_val_feat[preprocessor.feature_cols]
    y_val = df_val_feat['failure_imminent']
    X_test = df_test_feat[preprocessor.feature_cols]
    y_test = df_test_feat['failure_imminent']
    
    # Self-contained Random Forest benchmark (trained fresh here rather than
    # depending on train_classifier.py's pickle, since that pipeline may not
    # deploy a RandomForest at all — see "deployed_model" below).
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
    rf_model.fit(df_train_sub_feat[preprocessor.feature_cols], df_train_sub_feat['failure_imminent'])

    # Threshold Tuning on Validation Set
    val_probs = rf_model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (val_probs >= thresh).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thresh)

    print(f"Optimal threshold found on validation set: {best_threshold}")

    # Evaluate RF on Test Set
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs >= best_threshold).astype(int)
    
    # Calculate metrics
    metrics = {}
    
    def get_metrics(y_true, y_pred, y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "specificity": float(tn / (tn + fp)),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }
    
    metrics["random_forest"] = get_metrics(y_test, rf_preds, rf_probs)
    metrics["random_forest"]["applied_threshold"] = best_threshold
    
    # False Negative Analysis
    false_negatives_mask = (y_test == 1) & (rf_preds == 0)
    fn_df = df_test_feat[false_negatives_mask]
    
    fn_analysis = {
        "missed_failures": int(sum(false_negatives_mask)),
        "mean_probability_of_missed": float(np.mean(rf_probs[false_negatives_mask])) if sum(false_negatives_mask) > 0 else 0.0,
        "machines_affected": [int(m) for m in fn_df['machine_id'].unique()]
    }
    
    with open("evaluation/results/classification_error_analysis.json", "w") as f:
        json.dump(fn_analysis, f, indent=4)
        
    metrics["false_negative_analysis"] = fn_analysis
    
    # Baseline 1: Majority Class (Dummy)
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(df_train_sub_feat[preprocessor.feature_cols], df_train_sub_feat['failure_imminent'])
    dummy_preds = dummy.predict(X_test)
    dummy_probs = dummy.predict_proba(X_test)[:, 1]
    metrics["baseline_majority"] = get_metrics(y_test, dummy_preds, dummy_probs)
    
    # Baseline 2: Logistic Regression
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(df_train_sub_feat[preprocessor.feature_cols], df_train_sub_feat['failure_imminent'])
    lr_probs = lr.predict_proba(X_test)[:, 1]
    
    # Tune LR threshold on val
    lr_val_probs = lr.predict_proba(X_val)[:, 1]
    best_lr_thresh = 0.5
    best_lr_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (lr_val_probs >= thresh).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_lr_f1:
            best_lr_f1 = f1
            best_lr_thresh = float(thresh)
            
    lr_preds = (lr_probs >= best_lr_thresh).astype(int)
    metrics["baseline_logistic_regression"] = get_metrics(y_test, lr_preds, lr_probs)
    metrics["baseline_logistic_regression"]["applied_threshold"] = best_lr_thresh

    # The model actually deployed by ml/training/train_classifier.py: whichever
    # of LogisticRegression/RandomForest won on PR-AUC there, not assumed to be RF.
    deployed_artifact = joblib.load("models/classifier.pkl")
    deployed_model = deployed_artifact["model"]
    deployed_val_probs = deployed_model.predict_proba(X_val)[:, 1]
    best_deployed_thresh, best_deployed_f1 = 0.5, 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (deployed_val_probs >= thresh).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_deployed_f1:
            best_deployed_f1 = f1
            best_deployed_thresh = float(thresh)

    deployed_probs = deployed_model.predict_proba(X_test)[:, 1]
    deployed_preds = (deployed_probs >= best_deployed_thresh).astype(int)
    metrics["deployed_model"] = get_metrics(y_test, deployed_preds, deployed_probs)
    metrics["deployed_model"]["algorithm"] = deployed_artifact["algorithm"]
    metrics["deployed_model"]["applied_threshold"] = best_deployed_thresh

    # Dataset stats
    metrics["dataset_stats"] = {
        "test_samples": int(len(y_test)),
        "test_positives": int(sum(y_test)),
        "test_negatives": int(len(y_test) - sum(y_test))
    }
    
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/classification.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Classification evaluation complete.")

if __name__ == "__main__":
    evaluate_classification()
