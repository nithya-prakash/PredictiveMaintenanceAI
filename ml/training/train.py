"""
Trains the RUL regressor, the imminent-failure classifier and the anomaly
detector on NASA C-MAPSS FD001 training engines, and saves them as one bundle.

Every choice (hyperparameters, which algorithm, the decision threshold) is made
with engine-wise cross-validation on the TRAINING engines only. NASA's official
test engines are never touched here; they are used once, by evaluation/, to
report the final numbers. (An earlier version tuned and selected models on its
test split, which made its reported test scores optimistic.)
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.cmapss import DATA_SHA256, DEFAULT_DIR, RUL_CAP, FAILURE_WINDOW, download, load_train
from ml.features.preprocessing import FEATURE_COLS, MIN_HISTORY, build_features
from ml.training.intervals import DEFAULT_ALPHA, conformal_halfwidth

BUNDLE_PATH = Path("models/model_bundle.joblib")
N_FOLDS = 5
SEED = 42


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def cv_predict(model, X, y, groups, method="predict"):
    """Out-of-fold predictions with engine-wise folds (no engine in both train and validation)."""
    return cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=N_FOLDS), method=method)


def tune(make_model, space, X, y, groups, score, direction, n_trials):
    def objective(trial):
        return score(y, cv_predict(make_model(space(trial)), X, y, groups, *(
            ["predict_proba"] if direction == "maximize" else [])))
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def pr_auc(y, proba):
    return float(average_precision_score(y, proba[:, 1] if proba.ndim == 2 else proba))


def train(n_trials: int = 8, data_dir: Path = DEFAULT_DIR):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data_dir = download(data_dir)
    df = build_features(load_train(data_dir))
    X, groups = df[FEATURE_COLS], df["engine_id"].to_numpy()
    y_rul, y_fail = df["rul_capped"].to_numpy(), df["failure_imminent"].to_numpy()
    print(f"Training rows (cycles with >= {MIN_HISTORY} history): {len(df)} from {len(np.unique(groups))} engines")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("CMAPSS_FD001")
    report = {}

    # ---------------- RUL regression ----------------
    rul_candidates = {
        "LinearRegression": (lambda p: make_pipeline(StandardScaler(), LinearRegression()), None),
        "RandomForestRegressor": (
            lambda p: RandomForestRegressor(random_state=SEED, n_jobs=-1, **p),
            lambda t: {"n_estimators": t.suggest_int("n_estimators", 100, 150, step=50),
                       "max_depth": t.suggest_int("max_depth", 6, 12),
                       "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 30),
                       "max_features": t.suggest_categorical("max_features", [0.3, 0.5])}),
        "HistGradientBoostingRegressor": (
            lambda p: HistGradientBoostingRegressor(random_state=SEED, **p),
            lambda t: {"learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                       "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 8, 48),
                       "min_samples_leaf": t.suggest_int("min_samples_leaf", 10, 80),
                       "l2_regularization": t.suggest_float("l2_regularization", 1e-3, 1.0, log=True)}),
    }
    rul_results = {}
    for name, (make, space) in rul_candidates.items():
        params, cv_rmse = {}, None
        if space is not None:
            params, cv_rmse = tune(make, space, X, y_rul, groups, rmse, "minimize", n_trials)
        else:
            cv_rmse = rmse(y_rul, cv_predict(make({}), X, y_rul, groups))
        rul_results[name] = {"cv_rmse": cv_rmse, "params": params}
        print(f"[RUL] {name}: engine-wise CV RMSE {cv_rmse:.2f}")
    rul_best = min(rul_results, key=lambda k: rul_results[k]["cv_rmse"])
    rul_model = rul_candidates[rul_best][0](rul_results[rul_best]["params"]).fit(X, y_rul)
    # Split-conformal interval from engine-wise out-of-fold residuals of the selected model.
    rul_make, rul_params = rul_candidates[rul_best][0], rul_results[rul_best]["params"]
    oof_rul = cv_predict(rul_make(rul_params), X, y_rul, groups)
    interval = {"alpha": DEFAULT_ALPHA, "halfwidth": conformal_halfwidth(y_rul - oof_rul, DEFAULT_ALPHA),
                "method": "split conformal, engine-wise out-of-fold |residual|"}
    print(f"[RUL] {1 - DEFAULT_ALPHA:.0%} conformal half-width: +/-{interval['halfwidth']:.1f} cycles")
    report["rul"] = {"candidates": rul_results, "selected": rul_best, "interval": interval}

    # ---------------- Imminent-failure classification ----------------
    clf_candidates = {
        "LogisticRegression": (
            lambda p: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", **p)),
            lambda t: {"C": t.suggest_float("C", 1e-3, 10.0, log=True)}),
        "HistGradientBoostingClassifier": (
            lambda p: HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced", **p),
            lambda t: {"learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                       "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 8, 48),
                       "min_samples_leaf": t.suggest_int("min_samples_leaf", 10, 80)}),
    }
    clf_results = {}
    for name, (make, space) in clf_candidates.items():
        params, cv_pr = tune(make, space, X, y_fail, groups, pr_auc, "maximize", n_trials)
        clf_results[name] = {"cv_pr_auc": cv_pr, "params": params}
        print(f"[Failure] {name}: engine-wise CV PR-AUC {cv_pr:.3f}")
    clf_best = max(clf_results, key=lambda k: clf_results[k]["cv_pr_auc"])
    clf_make = clf_candidates[clf_best][0]
    clf_params = clf_results[clf_best]["params"]

    # Decision threshold from out-of-fold probabilities on training engines: the
    # threshold that maximises F1. The API uses exactly this threshold, and the
    # evaluation reports test metrics at exactly this threshold.
    oof = cv_predict(clf_make(clf_params), X, y_fail, groups, "predict_proba")[:, 1]
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [f1_score(y_fail, oof >= t) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(f1s))])
    clf_model = clf_make(clf_params).fit(X, y_fail)
    report["failure"] = {"candidates": clf_results, "selected": clf_best,
                         "threshold": threshold, "threshold_cv_f1": float(max(f1s))}
    print(f"[Failure] selected {clf_best}, threshold {threshold} (CV F1 {max(f1s):.3f})")

    # ---------------- Anomaly detection ----------------
    # Learns "healthy" behaviour from early life (true RUL above the cap, i.e.
    # before degradation is expected to be visible) and flags readings less
    # normal than 99% of healthy training cycles.
    healthy = X[df["rul"].to_numpy() > RUL_CAP]
    iso = IsolationForest(n_estimators=200, random_state=SEED).fit(healthy)
    anomaly_threshold = float(np.percentile(iso.score_samples(healthy), 1))
    report["anomaly"] = {"trained_on_rows": int(len(healthy)), "score_threshold": anomaly_threshold,
                         "definition": f"score below the 1st percentile of healthy (RUL > {RUL_CAP}) training cycles"}

    bundle = {
        "feature_cols": FEATURE_COLS,
        "min_history": MIN_HISTORY,
        "rul": {"model": rul_model, "algorithm": rul_best, "cap": RUL_CAP, "interval": interval},
        "failure": {"model": clf_model, "algorithm": clf_best, "threshold": threshold, "window": FAILURE_WINDOW},
        "anomaly": {"model": iso, "score_threshold": anomaly_threshold},
        # Plain float array (not a DataFrame), so loading the bundle needs no
        # optional pandas backends (e.g. pyarrow) in the slim API image.
        "shap_background": X.sample(n=200, random_state=SEED).to_numpy(dtype=float),
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "NASA C-MAPSS FD001 (training engines only)",
            "dataset_sha256": DATA_SHA256,
            "training_rows": int(len(df)),
            "training_engines": int(len(np.unique(groups))),
            "sklearn_version": sklearn.__version__,
            "selection": "engine-wise GroupKFold on training engines; official test set untouched",
        },
    }
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BUNDLE_PATH.with_name(BUNDLE_PATH.name + ".tmp")
    joblib.dump(bundle, tmp, compress=3)
    tmp.replace(BUNDLE_PATH)
    Path("models/training_report.json").write_text(json.dumps({**report, "metadata": bundle["metadata"]}, indent=2))

    with mlflow.start_run(run_name="train_all"):
        mlflow.log_params({"rul_model": rul_best, "failure_model": clf_best, "threshold": threshold,
                           "n_trials": n_trials, "min_history": MIN_HISTORY})
        mlflow.log_metrics({"rul_cv_rmse": rul_results[rul_best]["cv_rmse"],
                            "failure_cv_pr_auc": clf_results[clf_best]["cv_pr_auc"],
                            "failure_cv_f1_at_threshold": float(max(f1s))})
        mlflow.log_artifact(str(BUNDLE_PATH))
    print(f"Saved {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / 1e6:.1f} MB)")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all models on C-MAPSS FD001 training engines.")
    parser.add_argument("--trials", type=int, default=8, help="Optuna trials per tuned candidate")
    args = parser.parse_args()
    train(n_trials=args.trials)
