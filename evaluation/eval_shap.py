import os
import json
import joblib
import pandas as pd
import numpy as np
import shap

def evaluate_shap():
    print("Evaluating SHAP Explainability...")
    
    # Load Models & Sample Data
    try:
        rf_classifier = joblib.load("models/rf_classifier.pkl")
        preprocessor = joblib.load("models/preprocessor.pkl")
    except Exception as e:
        print(f"Error loading models for SHAP eval: {e}")
        return
        
    df_test = pd.read_csv("datasets/test_data.csv")
    df_test_feat = preprocessor.add_rolling_features(df_test)
    df_test_feat[preprocessor.feature_cols] = preprocessor.scaler.transform(df_test_feat[preprocessor.feature_cols])
    
    # For SHAP stability, we will take a sample of positive and negative cases
    pos_idx = df_test_feat[df_test_feat['failure_imminent'] == 1].index[:50]
    neg_idx = df_test_feat[df_test_feat['failure_imminent'] == 0].index[:50]
    
    sample_idx = list(pos_idx) + list(neg_idx)
    X_sample = df_test_feat.loc[sample_idx, preprocessor.feature_cols].values
    
    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(X_sample)
    
    # For RandomForest in SHAP, shap_values is a list of arrays (one for each class)
    # We want the positive class [1]
    if isinstance(shap_values, list):
        sv_pos = shap_values[1]
    else:
        if len(shap_values.shape) == 3:
            sv_pos = shap_values[:, :, 1]
        else:
            sv_pos = shap_values
            
    # Calculate mean absolute SHAP value for each feature across the sample
    mean_abs_shap = np.mean(np.abs(sv_pos), axis=0)
    feature_importance = {feat: float(val) for feat, val in zip(preprocessor.feature_cols, mean_abs_shap)}
    
    # Sort features by importance
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
    top_5_features = list(sorted_importance.keys())[:5]
    
    metrics = {
        "xai_metrics": {
            "top_5_global_features": top_5_features,
            "feature_importance_scores": sorted_importance,
            "note": "Quantitative SHAP evaluation (e.g. fidelity/stability metrics) requires specialized perturbation testing which is beyond the current scope. These represent global feature attribution magnitudes on a test sample."
        }
    }
    
    with open("evaluation/results/shap.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("SHAP evaluation complete.")

if __name__ == "__main__":
    evaluate_shap()
