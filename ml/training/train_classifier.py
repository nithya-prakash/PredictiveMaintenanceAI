import pandas as pd
import numpy as np
import mlflow
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import argparse
from ml.features.preprocessing import PredictiveMaintenancePreprocessor

def load_data(train_path: str, test_path: str):
    print("Loading data for classification...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    preprocessor = PredictiveMaintenancePreprocessor.load("models/preprocessor.pkl")
    
    df_train_feat = preprocessor.add_rolling_features(df_train)
    df_test_feat = preprocessor.add_rolling_features(df_test)
    
    df_train_feat[preprocessor.feature_cols] = preprocessor.scaler.transform(df_train_feat[preprocessor.feature_cols])
    df_test_feat[preprocessor.feature_cols] = preprocessor.scaler.transform(df_test_feat[preprocessor.feature_cols])
    
    X_train = df_train_feat[preprocessor.feature_cols]
    y_train = df_train_feat['failure_imminent']
    
    X_test = df_test_feat[preprocessor.feature_cols]
    y_test = df_test_feat['failure_imminent']
    
    return X_train, X_test, y_train, y_test

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    
    return f1

def train_classifier_model(data_dir: str = "datasets", n_trials: int = 10):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Failure_Classification")
    
    train_path = os.path.join(data_dir, "train_data.csv")
    test_path = os.path.join(data_dir, "test_data.csv")
    
    X_train, X_test, y_train, y_test = load_data(train_path, test_path)
    
    print(f"Running Optuna optimization with {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)
    
    best_params = study.best_params
    print("Best Params:", best_params)
    
    with mlflow.start_run(run_name="RandomForest_Classifier_Best"):
        mlflow.log_params(best_params)
        
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = 42
        final_model = RandomForestClassifier(**best_params)
        final_model.fit(X_train, y_train)
        
        preds = final_model.predict(X_test)
        probs = final_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)
        
        print(f"Final Metrics - Acc: {acc:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")
        
        os.makedirs("models", exist_ok=True)
        model_path = "models/rf_classifier.pkl"
        joblib.dump(final_model, model_path)
        mlflow.sklearn.log_model(final_model, "model")
        print(f"Saved classifier model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    
    train_classifier_model(n_trials=args.trials)
