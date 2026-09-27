"""
Final evaluation on NASA's official C-MAPSS FD001 test engines.

The test engines are used only here, after every modelling choice was made with
cross-validation on the training engines (ml/training/train.py). Writes
evaluation/results/*.json and evaluation/results/FINAL_REPORT.md.

Run:  python -m evaluation.run_all
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.cmapss import DEFAULT_DIR, RUL_CAP, SENSORS, download, load_test, load_train
from ml.features.preprocessing import FEATURE_COLS, build_features
from ml.training.train import BUNDLE_PATH, SEED

RESULTS = Path("evaluation/results")

# Published FD001 results under the same protocol (RUL capped at 125), from
# Ragab et al., "Attention Sequence to Sequence Model for Machine Remaining Useful
# Life Prediction", 2020, Table III (arXiv:2007.09868). Included for context
# only: those models use sliding windows of 30 cycles and deep networks.
PUBLISHED_FD001 = [
    ("Random Forest", 17.91, 480), ("Gradient Boosting", 15.67, 474), ("1D CNN (Li et al.)", 12.61, 274),
    ("Deep LSTM", 16.14, 338), ("ATS2S (Ragab et al.)", 12.63, 243),
]


def nasa_score(y_true, y_pred):
    """PHM08 scoring function: late predictions (predicted RUL too high) are
    penalised more than early ones. Lower is better."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def rul_metrics(y_capped, y_true, pred):
    return {"rmse": rmse(y_capped, pred), "mae": float(mean_absolute_error(y_capped, pred)),
            "nasa_score": nasa_score(y_capped, pred), "rmse_vs_uncapped_truth": rmse(y_true, pred)}


def clf_metrics(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"pr_auc": float(average_precision_score(y, proba)), "roc_auc": float(roc_auc_score(y, proba)),
            "threshold": float(threshold), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
            "brier": float(brier_score_loss(y, proba)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def rebuild(algorithm, params, kind):
    """Refits a non-selected candidate with its CV-tuned params, for comparison only."""
    makers = {
        "LinearRegression": lambda p: make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForestRegressor": lambda p: RandomForestRegressor(random_state=SEED, n_jobs=-1, **p),
        "HistGradientBoostingRegressor": lambda p: HistGradientBoostingRegressor(random_state=SEED, **p),
        "LogisticRegression": lambda p: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", **p)),
        "HistGradientBoostingClassifier": lambda p: HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced", **p),
    }
    return makers[algorithm](params)


def evaluate():
    data_dir = download(DEFAULT_DIR)
    bundle = joblib.load(BUNDLE_PATH)
    report = json.loads(Path("models/training_report.json").read_text())
    train = build_features(load_train(data_dir))
    test = build_features(load_test(data_dir))
    X_tr, X_te = train[FEATURE_COLS], test[FEATURE_COLS]
    results = {"dataset": {"test_engines": int(test.engine_id.nunique()), "test_rows_with_full_history": int(len(test)),
                           "note": "Each test engine's trajectory stops before failure; NASA gives the true RUL at its last cycle."}}

    # ---- RUL: official protocol = prediction at each test engine's last cycle ----
    last = test.groupby("engine_id").tail(1)
    y_last, y_last_capped = last["rul"].to_numpy(), last["rul_capped"].to_numpy()
    rul = {"selected": bundle["rul"]["algorithm"], "models": {}}
    for name, info in report["rul"]["candidates"].items():
        model = bundle["rul"]["model"] if name == bundle["rul"]["algorithm"] else rebuild(name, info["params"], "rul").fit(X_tr, train["rul_capped"])
        rul["models"][name] = {**rul_metrics(y_last_capped, y_last, model.predict(X_te.loc[last.index])), "cv_rmse": info["cv_rmse"]}
    mean_pred = np.full(len(last), train["rul_capped"].mean())
    rul["models"]["Mean baseline"] = rul_metrics(y_last_capped, y_last, mean_pred)
    all_pred = bundle["rul"]["model"].predict(X_te)
    buckets = {"true RUL <= 30": test["rul"] <= 30, "31-60": (test["rul"] > 30) & (test["rul"] <= 60),
               "61-125": (test["rul"] > 60) & (test["rul"] <= RUL_CAP), "> 125": test["rul"] > RUL_CAP}
    rul["rmse_by_true_rul_all_cycles"] = {k: rmse(test["rul_capped"][m], all_pred[m.to_numpy()]) for k, m in buckets.items()}
    results["rul"] = rul

    # ---- Imminent failure: every test cycle with full history ----
    y = test["failure_imminent"].to_numpy()
    fail = {"selected": bundle["failure"]["algorithm"], "served_threshold": bundle["failure"]["threshold"], "models": {},
            "positives": int(y.sum()), "negatives": int(len(y) - y.sum()),
            "engines_with_positive_cycles": int(test.loc[test.failure_imminent == 1, "engine_id"].nunique())}
    for name, info in report["failure"]["candidates"].items():
        model = bundle["failure"]["model"] if name == bundle["failure"]["algorithm"] else rebuild(name, info["params"], "clf").fit(X_tr, train["failure_imminent"])
        thr = bundle["failure"]["threshold"] if name == bundle["failure"]["algorithm"] else 0.5
        fail["models"][name] = {**clf_metrics(y, model.predict_proba(X_te)[:, 1], thr), "cv_pr_auc": info["cv_pr_auc"]}
    dummy = DummyClassifier(strategy="prior").fit(X_tr, train["failure_imminent"])
    fail["models"]["Prior baseline"] = clf_metrics(y, dummy.predict_proba(X_te)[:, 1], 0.5)
    results["failure"] = fail

    # ---- Anomaly flag rate by true RUL: should rise as failure approaches ----
    iso = bundle["anomaly"]
    flagged = iso["model"].score_samples(X_te) < iso["score_threshold"]
    results["anomaly"] = {k: {"flag_rate": float(flagged[m.to_numpy()].mean()), "cycles": int(m.sum())} for k, m in buckets.items()}

    # ---- SHAP: global importance of the served failure model on test cycles ----
    from backend.services.xai import explain_rows
    sample = X_te.sample(n=min(300, len(X_te)), random_state=SEED)
    contrib = explain_rows(bundle, sample)
    importance = contrib.abs().mean().sort_values(ascending=False)
    results["shap_top_features"] = {k: float(v) for k, v in importance.head(8).items()}

    # ---- Latency through the real API stack (in-process HTTP, no network) ----
    results["latency"] = measure_latency(test)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "FINAL_REPORT.md").write_text(render(results, bundle))
    print((RESULTS / "FINAL_REPORT.md").read_text())
    return results


def measure_latency(test: pd.DataFrame, n: int = 200):
    from fastapi.testclient import TestClient
    from backend.core.config import settings
    from backend.main import app
    settings.RATE_LIMIT_PER_MINUTE = 10**6  # measure the model path, not the rate limiter
    engine = test[test.engine_id == test.engine_id.iloc[0]].tail(30)
    payload = {"machine_id": 1, "readings": engine[["cycle"] + SENSORS].to_dict(orient="records")}
    out = {"context": f"{n} sequential requests, 30 readings each, FastAPI TestClient (full HTTP stack in-process, "
                      "no network), after 10 warm-up calls; laptop CPU. Not a production load test."}
    with TestClient(app) as client:
        for path in ("/api/v1/predict-failure", "/api/v1/explain"):
            for _ in range(10):
                client.post(path, json=payload)
            times = []
            for _ in range(n):
                t0 = time.perf_counter()
                r = client.post(path, json=payload)
                times.append((time.perf_counter() - t0) * 1000)
                assert r.status_code == 200, r.text
            out[path] = {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                         "p99_ms": float(np.percentile(times, 99))}
    return out


def render(r, bundle):
    rul, fail = r["rul"], r["failure"]
    lines = ["# Evaluation report: NASA C-MAPSS FD001", "",
             f"Test set: NASA's official FD001 test set, {r['dataset']['test_engines']} engines. "
             "All model choices (algorithm, hyperparameters, decision threshold) were made with engine-wise "
             "cross-validation on the training engines; the test engines were used only for this report.", "",
             "## Remaining useful life (official protocol: last cycle of each test engine, RUL capped at 125)", "",
             "| Model | RMSE | MAE | NASA score | CV RMSE (train) |", "|---|---|---|---|---|"]
    for name, m in rul["models"].items():
        tag = " (deployed)" if name == rul["selected"] else ""
        cv = f"{m['cv_rmse']:.2f}" if "cv_rmse" in m else "–"
        lines.append(f"| {name}{tag} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['nasa_score']:.0f} | {cv} |")
    lines += ["", "Published FD001 results under the same capped-RUL protocol, for context "
              "(Ragab et al. 2020, Table III, arXiv:2007.09868):", "", "| Published method | RMSE | NASA score |", "|---|---|---|"]
    lines += [f"| {n} | {a:.2f} | {b} |" for n, a, b in PUBLISHED_FD001]
    lines += ["", "RMSE of the deployed model over all test cycles, by true RUL: " +
              ", ".join(f"{k}: {v:.1f}" for k, v in rul["rmse_by_true_rul_all_cycles"].items()), "",
              "## Imminent failure (true RUL <= 30 cycles), all test cycles with 15+ cycles of history", "",
              f"{fail['positives']} positive and {fail['negatives']} negative cycles, from "
              f"{fail['engines_with_positive_cycles']} engines that come within 30 cycles of failure. Consecutive "
              "cycles of one engine are correlated, so treat these as 100 engines' worth of evidence, not "
              f"{fail['positives'] + fail['negatives']} independent samples.", "",
              "| Model | PR-AUC | ROC-AUC | Threshold | Precision | Recall | F1 | Brier |", "|---|---|---|---|---|---|---|---|"]
    for name, m in fail["models"].items():
        tag = " (deployed)" if name == fail["selected"] else ""
        lines.append(f"| {name}{tag} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['threshold']:.2f} | {m['precision']:.3f} | "
                     f"{m['recall']:.3f} | {m['f1']:.3f} | {m['brier']:.3f} |")
    lines += ["", f"The deployed threshold ({fail['served_threshold']:.2f}) was chosen on out-of-fold training predictions "
              "and is exactly the one the API uses.", "",
              "## Anomaly flag rate by true RUL (should rise towards failure)", "",
              "| True RUL | Cycles | Flagged |", "|---|---|---|"]
    lines += [f"| {k} | {v['cycles']} | {v['flag_rate']:.1%} |" for k, v in r["anomaly"].items()]
    lines += ["", "## SHAP: most influential features for the failure model (mean |SHAP| on 300 test cycles)", ""]
    lines += [f"- `{k}`: {v:.3f}" for k, v in r["shap_top_features"].items()]
    lat = r["latency"]
    lines += ["", "## Latency", "", lat["context"], ""]
    lines += [f"- `{p}`: p50 {v['p50_ms']:.1f} ms, p95 {v['p95_ms']:.1f} ms, p99 {v['p99_ms']:.1f} ms"
              for p, v in lat.items() if p.startswith("/")]
    lines += ["", f"Model bundle trained {bundle['metadata']['trained_at'][:19]} UTC on {bundle['metadata']['dataset']}."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    evaluate()
