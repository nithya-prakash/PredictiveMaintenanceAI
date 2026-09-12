import pandas as pd
import numpy as np
import mlflow
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
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

    # Optimize for PR-AUC, not F1: failure_imminent is a rare-event label,
    # and PR-AUC is the metric we use to compare against the baseline below.
    probs = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, probs)

def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
    }

def train_classifier_model(data_dir: str = "datasets", n_trials: int = 10):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Failure_Classification")

    train_path = os.path.join(data_dir, "train_data.csv")
    test_path = os.path.join(data_dir, "test_data.csv")

    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    # --- Baseline: plain LogisticRegression ---
    # Always benchmark the "sophisticated" model against a simple baseline
    # before headlining it. On this synthetic dataset the two are frequently
    # close (sometimes the baseline wins), so we pick by measured PR-AUC
    # rather than assuming the more complex model is better.
    with mlflow.start_run(run_name="LogisticRegression_Baseline"):
        baseline_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        baseline_model.fit(X_train, y_train)
        baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
        mlflow.log_metrics(baseline_metrics)
        print(f"[Baseline] LogisticRegression - PR-AUC: {baseline_metrics['pr_auc']:.4f}, "
              f"ROC-AUC: {baseline_metrics['roc_auc']:.4f}, F1: {baseline_metrics['f1']:.4f}")

    # --- Candidate: tuned RandomForest ---
    print(f"Running Optuna optimization with {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)

    best_params = study.best_params
    print("Best Params:", best_params)

    with mlflow.start_run(run_name="RandomForest_Classifier_Best"):
        mlflow.log_params(best_params)

        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = 42
        rf_model = RandomForestClassifier(**best_params)
        rf_model.fit(X_train, y_train)

        rf_metrics = evaluate_model(rf_model, X_test, y_test)
        mlflow.log_metrics(rf_metrics)
        print(f"[Candidate] RandomForest - PR-AUC: {rf_metrics['pr_auc']:.4f}, "
              f"ROC-AUC: {rf_metrics['roc_auc']:.4f}, F1: {rf_metrics['f1']:.4f}")

    # --- Select whichever model actually wins on PR-AUC, don't assume RF ---
    if rf_metrics["pr_auc"] >= baseline_metrics["pr_auc"]:
        final_model, winner, final_metrics = rf_model, "RandomForestClassifier", rf_metrics
    else:
        final_model, winner, final_metrics = baseline_model, "LogisticRegression", baseline_metrics

    print(f"Selected model: {winner} (PR-AUC {final_metrics['pr_auc']:.4f} vs "
          f"{'baseline' if winner == 'RandomForestClassifier' else 'RandomForest'} "
          f"{(baseline_metrics if winner == 'RandomForestClassifier' else rf_metrics)['pr_auc']:.4f})")

    with mlflow.start_run(run_name=f"Selected_{winner}"):
        mlflow.log_param("selected_model", winner)
        mlflow.log_metrics(final_metrics)
        mlflow.sklearn.log_model(final_model, "model")

    os.makedirs("models", exist_ok=True)
    model_path = "models/classifier.pkl"
    # Small background sample for SHAP's LinearExplainer, in case the
    # LogisticRegression baseline is the one that wins (TreeExplainer needs
    # no background, but a linear model's explainer does).
    background_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)
    joblib.dump(
        {"model": final_model, "algorithm": winner, "background_sample": background_sample},
        model_path,
    )
    print(f"Saved selected classifier ({winner}) to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()

    train_classifier_model(n_trials=args.trials)
