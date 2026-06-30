import pandas as pd
import numpy as np
import mlflow
from sklearn.ensemble import IsolationForest
import joblib
import os
from ml.features.preprocessing import PredictiveMaintenancePreprocessor

def train_anomaly_detector(data_dir: str = "datasets"):
    print("Training Isolation Forest Anomaly Detector...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Anomaly_Detection")
    
    train_path = os.path.join(data_dir, "train_data.csv")
    df_train = pd.read_csv(train_path)
    
    preprocessor = PredictiveMaintenancePreprocessor.load("models/preprocessor.pkl")
    
    df_train_feat = preprocessor.add_rolling_features(df_train)
    df_train_feat[preprocessor.feature_cols] = preprocessor.scaler.transform(df_train_feat[preprocessor.feature_cols])
    
    # Train only on 'healthy' data (where failure is not imminent) to learn normal behavior
    X_train_healthy = df_train_feat[df_train_feat['failure_imminent'] == 0][preprocessor.feature_cols]
    
    with mlflow.start_run(run_name="IsolationForest"):
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(X_train_healthy)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("contamination", 0.05)
        
        model_path = "models/isolation_forest.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        print(f"Saved anomaly detector model to {model_path}")

if __name__ == "__main__":
    train_anomaly_detector()
