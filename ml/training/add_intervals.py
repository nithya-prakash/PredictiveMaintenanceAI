"""Adds the conformal RUL interval to an EXISTING model bundle without retraining
or re-tuning: refits the already-selected model (params from training_report.json)
engine-wise to get out-of-fold residuals, then stores the half-width in the bundle.

Run:  python -m ml.training.add_intervals
"""
import json
from pathlib import Path

import joblib

from ml.data.cmapss import DEFAULT_DIR, download, load_train
from ml.features.preprocessing import FEATURE_COLS, build_features
from ml.training.intervals import DEFAULT_ALPHA, conformal_halfwidth
from ml.training.train import BUNDLE_PATH, cv_predict


def main():
    from evaluation.run_all import rebuild

    report_path = Path("models/training_report.json")
    report = json.loads(report_path.read_text())
    bundle = joblib.load(BUNDLE_PATH)
    algo = bundle["rul"]["algorithm"]
    params = report["rul"]["candidates"][algo]["params"]

    df = build_features(load_train(download(DEFAULT_DIR)))
    X, y, groups = df[FEATURE_COLS], df["rul_capped"].to_numpy(), df["engine_id"].to_numpy()
    oof = cv_predict(rebuild(algo, params, "rul"), X, y, groups)
    interval = {"alpha": DEFAULT_ALPHA, "halfwidth": conformal_halfwidth(y - oof, DEFAULT_ALPHA),
                "method": "split conformal, engine-wise out-of-fold |residual|"}
    bundle["rul"]["interval"] = interval
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    report["rul"]["interval"] = interval
    report_path.write_text(json.dumps(report, indent=2))
    print(f"{1 - DEFAULT_ALPHA:.0%} interval half-width: +/-{interval['halfwidth']:.1f} cycles")


if __name__ == "__main__":
    main()
