"""Export model outputs as star-schema CSVs for Power BI.

    python -m scripts.export_bi          # writes bi/*.csv

Tables (join on engine_id [+ cycle]):
  dim_engine.csv             one row per official FD001 test engine (final-cycle view)
  fact_cycle_predictions.csv one row per test engine-cycle with full history
  fact_sensor_readings.csv   raw sensor values for the same cycles
  model_metrics.csv          long-format metrics from evaluation/results/results.json
All numbers come from the trained bundle and the official test set; nothing is
simulated. Requires evaluation/results/results.json (run `python -m evaluation.run_all`).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.data.cmapss import DEFAULT_DIR, SENSORS, download, load_test
from ml.features.preprocessing import FEATURE_COLS, build_features

OUT = Path("bi")
BUNDLE_PATH = Path("models/model_bundle.joblib")  # same path as ml.training.train (not imported: needs mlflow)


def risk_band(p: pd.Series, threshold: float) -> pd.Series:
    """High = at/above the served failure threshold; Elevated = above half of it."""
    return pd.Series(np.where(p >= threshold, "High", np.where(p >= 0.5 * threshold, "Elevated", "Low")), index=p.index)


def metrics_long(results: dict) -> pd.DataFrame:
    rows = []
    for name, m in results["rul"]["models"].items():
        rows += [("RUL", name, k, v) for k, v in m.items() if isinstance(v, (int, float))]
    for name, m in results["failure"]["models"].items():
        rows += [("Failure", name, k, v) for k, v in m.items() if isinstance(v, (int, float))]
    for bucket, m in results["anomaly"].items():
        rows.append(("Anomaly", bucket, "flag_rate", m["flag_rate"]))
    return pd.DataFrame(rows, columns=["task", "model", "metric", "value"])


def export() -> None:
    bundle = joblib.load(BUNDLE_PATH)
    test = build_features(load_test(download(DEFAULT_DIR)))
    X = test[FEATURE_COLS]
    threshold = bundle["failure"]["threshold"]
    iso = bundle["anomaly"]

    facts = test[["engine_id", "cycle", "rul", "rul_capped", "failure_imminent"]].rename(
        columns={"rul": "rul_true", "rul_capped": "rul_true_capped"}).copy()
    facts["rul_pred"] = bundle["rul"]["model"].predict(X)
    facts["failure_probability"] = bundle["failure"]["model"].predict_proba(X)[:, 1]
    facts["failure_flag"] = (facts["failure_probability"] >= threshold).astype(int)
    facts["anomaly_flag"] = (iso["model"].score_samples(X) < iso["score_threshold"]).astype(int)
    facts["risk_band"] = risk_band(facts["failure_probability"], threshold)

    last = facts.groupby("engine_id").tail(1).set_index("engine_id")
    dim = pd.DataFrame({
        "cycles_observed": test.groupby("engine_id")["cycle"].max(),
        "rul_true_at_last_cycle": last["rul_true"], "rul_pred_at_last_cycle": last["rul_pred"],
        "abs_error_at_last_cycle": (last["rul_true_capped"] - last["rul_pred"]).abs(),
        "failure_probability_at_last_cycle": last["failure_probability"],
        "risk_band_at_last_cycle": last["risk_band"],
    }).reset_index()

    OUT.mkdir(exist_ok=True)
    dim.round(4).to_csv(OUT / "dim_engine.csv", index=False)
    facts.round(4).to_csv(OUT / "fact_cycle_predictions.csv", index=False)
    test[["engine_id", "cycle", *SENSORS]].to_csv(OUT / "fact_sensor_readings.csv", index=False)
    results = json.loads(Path("evaluation/results/results.json").read_text())
    metrics_long(results).round(4).to_csv(OUT / "model_metrics.csv", index=False)
    print({p.name: len(pd.read_csv(p)) for p in sorted(OUT.glob("*.csv"))})


if __name__ == "__main__":
    export()
