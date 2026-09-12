import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

def evaluate_rul():
    print("Evaluating RUL Regression Model...")
    
    test_path = "datasets/test_data.csv"
    train_path = "datasets/train_data.csv"
    df_test = pd.read_csv(test_path)
    df_train = pd.read_csv(train_path)
    
    preprocessor = joblib.load("models/preprocessor.pkl")
    df_test_feat = preprocessor.add_rolling_features(df_test)
    df_train_feat = preprocessor.add_rolling_features(df_train)
    
    df_test_feat[preprocessor.feature_cols] = preprocessor.scaler.transform(df_test_feat[preprocessor.feature_cols])
    df_train_feat[preprocessor.feature_cols] = preprocessor.scaler.transform(df_train_feat[preprocessor.feature_cols])
    
    X_test = df_test_feat[preprocessor.feature_cols]
    y_test = df_test_feat['RUL']
    
    X_train = df_train_feat[preprocessor.feature_cols]
    y_train = df_train_feat['RUL']
    
    rf_model = joblib.load("models/rf_rul.pkl")
    rf_preds = rf_model.predict(X_test)
    
    metrics = {}
    
    def get_metrics(y_true, y_pred):
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "median_ae": float(median_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))
        }
    
    metrics["random_forest"] = get_metrics(y_test, rf_preds)
    
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)
    metrics["baseline_mean"] = get_metrics(y_test, dummy_preds)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    metrics["baseline_linear_regression"] = get_metrics(y_test, lr_preds)
    
    # Error Analysis by Range
    low_rul_mask = y_test <= 30
    med_rul_mask = (y_test > 30) & (y_test <= 100)
    high_rul_mask = y_test > 100
    
    metrics["error_analysis"] = {
        "low_rul_rmse": float(np.sqrt(mean_squared_error(y_test[low_rul_mask], rf_preds[low_rul_mask]))) if sum(low_rul_mask)>0 else 0.0,
        "medium_rul_rmse": float(np.sqrt(mean_squared_error(y_test[med_rul_mask], rf_preds[med_rul_mask]))) if sum(med_rul_mask)>0 else 0.0,
        "high_rul_rmse": float(np.sqrt(mean_squared_error(y_test[high_rul_mask], rf_preds[high_rul_mask]))) if sum(high_rul_mask)>0 else 0.0
    }
    
    with open("evaluation/results/rul.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("RUL evaluation complete.")

if __name__ == "__main__":
    evaluate_rul()
