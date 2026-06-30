import pandas as pd
import numpy as np
import mlflow
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import argparse
from ml.features.preprocessing import PredictiveMaintenancePreprocessor

def load_data(train_path: str, test_path: str):
    print("Loading data...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    preprocessor = PredictiveMaintenancePreprocessor(sequence_length=1)
    
    df_train_feat = preprocessor.fit_transform(df_train, is_training=True)
    df_test_feat = preprocessor.fit_transform(df_test, is_training=False)
    
    X_train = df_train_feat[preprocessor.feature_cols]
    y_train = df_train_feat['RUL']
    
    X_test = df_test_feat[preprocessor.feature_cols]
    y_test = df_test_feat['RUL']
    
    preprocessor.save("models/preprocessor.pkl")
    return X_train, X_test, y_train, y_test, preprocessor.feature_cols

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'random_state': 42
    }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

def train_rul_model(data_dir: str = "datasets", n_trials: int = 10):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("RUL_Prediction")
    
    train_path = os.path.join(data_dir, "train_data.csv")
    test_path = os.path.join(data_dir, "test_data.csv")
    
    X_train, X_test, y_train, y_test, feature_cols = load_data(train_path, test_path)
    
    print(f"Running Optuna optimization with {n_trials} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)
    
    best_params = study.best_params
    print("Best Params:", best_params)
    
    with mlflow.start_run(run_name="RandomForest_RUL_Best"):
        mlflow.log_params(best_params)
        
        best_params['random_state'] = 42
        final_model = RandomForestRegressor(**best_params)
        final_model.fit(X_train, y_train)
        
        preds = final_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        print(f"Final Metrics - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        
        os.makedirs("models", exist_ok=True)
        model_path = "models/rf_rul.pkl"
        joblib.dump(final_model, model_path)
        mlflow.sklearn.log_model(final_model, "model")
        
        joblib.dump(feature_cols, "models/feature_cols.pkl")
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    
    train_rul_model(n_trials=args.trials)
