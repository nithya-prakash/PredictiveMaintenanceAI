import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def evaluate_robustness():
    print("Evaluating Robustness (Machine & Temporal)...")
    
    train_path = "datasets/train_data.csv"
    df_train = pd.read_csv(train_path)
    preprocessor = joblib.load("models/preprocessor.pkl")
    df_train_feat = preprocessor.add_rolling_features(df_train)
    
    groups = df_train_feat['machine_id'].values
    X = df_train_feat[preprocessor.feature_cols].values
    y = df_train_feat['failure_imminent'].values
    
    # 1. GroupKFold (Machine-wise)
    gkf = GroupKFold(n_splits=5)
    
    roc_aucs, pr_aucs, f1s, fold_details = [], [], [], []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        clf.fit(X_tr_scaled, y_tr)
        
        preds = clf.predict(X_va_scaled)
        probs = clf.predict_proba(X_va_scaled)[:, 1]
        
        roc, pr, f1 = roc_auc_score(y_va, probs), average_precision_score(y_va, probs), f1_score(y_va, preds)
        roc_aucs.append(roc); pr_aucs.append(pr); f1s.append(f1)
        
        fold_details.append({"fold": fold + 1, "roc_auc": float(roc), "pr_auc": float(pr), "f1": float(f1)})
    
    metrics = {
        "machine_wise_cv_5_fold": {
            "roc_auc_mean": float(np.mean(roc_aucs)),
            "roc_auc_std": float(np.std(roc_aucs)),
            "pr_auc_mean": float(np.mean(pr_aucs)),
            "pr_auc_std": float(np.std(pr_aucs)),
            "fold_details": fold_details
        },
        "temporal_split": {
            "status": "NOT APPLICABLE",
            "reason": "The synthetic dataset generates machines independently without a global timestamp column tracking chronological factory deployment. Hence, chronological TimeSeriesSplit across machines is not physically meaningful for this specific synthetic generation logic."
        }
    }
    
    with open("evaluation/results/cross_validation.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Robustness evaluation complete.")

if __name__ == "__main__":
    evaluate_robustness()
