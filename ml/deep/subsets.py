"""Runs the 1D-CNN benchmark on all four C-MAPSS subsets (FD001-FD004).

FD002 and FD004 have six operating conditions, which shift every sensor, so each
sensor is standardised per operating condition (conditions found by k-means on
the three operating settings, fit on training engines only). FD001/FD003 have one
condition, which reduces this to plain standardisation. Protocol is unchanged:
early stopping on held-out training engines, official test engines scored once at
each engine's last cycle, capped RUL.

Run:  python -m ml.deep.subsets          (needs requirements-deep.txt)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ml.data.cmapss import COLUMNS, DEFAULT_DIR, RUL_CAP, SENSORS, add_labels, download
from ml.deep.cnn_rul import SEED, _predict, make_windows, nasa_score, train

SUBSETS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
RESULTS_PATH = Path("evaluation/results/deep_rul_subsets.json")


def read_raw(data_dir: Path, subset: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / f"{split}_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    return df.astype({c: float for c in SENSORS + OP_COLS})


def load_labelled(data_dir: Path, subset: str):
    """Train and test frames (with op settings and RUL labels) for one subset."""
    tr = read_raw(data_dir, subset, "train")
    tr = add_labels(tr, tr.groupby("engine_id")["cycle"].transform("max") - tr["cycle"])
    te = read_raw(data_dir, subset, "test")
    final = pd.read_csv(Path(data_dir) / f"RUL_{subset}.txt", header=None).iloc[:, 0]
    final.index = range(1, len(final) + 1)
    last = te.groupby("engine_id")["cycle"].transform("max")
    return tr, add_labels(te, te["engine_id"].map(final) + (last - te["cycle"]))


def normalise_per_condition(train: pd.DataFrame, test: pd.DataFrame, n_conditions: int):
    """Per-condition standardisation of SENSORS, with conditions and statistics fit on train only."""
    train, test = train.copy(), test.copy()
    if n_conditions == 1:
        train["cond"], test["cond"] = 0, 0
    else:
        km = KMeans(n_clusters=n_conditions, n_init=10, random_state=SEED).fit(train[OP_COLS])
        train["cond"], test["cond"] = km.labels_, km.predict(test[OP_COLS])
    stats = train.groupby("cond")[SENSORS].agg(["mean", "std"])
    for df in (train, test):
        for c in sorted(df["cond"].unique()):
            m = df["cond"] == c
            mean = stats.loc[c].xs("mean", level=1)[SENSORS].to_numpy()
            std = stats.loc[c].xs("std", level=1)[SENSORS].to_numpy()
            df.loc[m, SENSORS] = (df.loc[m, SENSORS].to_numpy() - mean) / (std + 1e-6)
    return train, test


def run_subset(data_dir: Path, subset: str, n_conditions: int, **train_kwargs) -> dict:
    tr, te = load_labelled(data_dir, subset)
    tr, te = normalise_per_condition(tr, te, n_conditions)
    model, mean, std, info = train(tr, **train_kwargs)
    X, y, _ = make_windows(te, last_only=True)
    pred = _predict(model, X, mean, std)
    return {"subset": subset, "operating_conditions": n_conditions, "train_engines": int(tr.engine_id.nunique()),
            "test_engines": int(te.engine_id.nunique()), "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "mae": float(np.mean(np.abs(pred - y))), "nasa_score": nasa_score(y, pred), **info}


def main():
    data_dir = download(DEFAULT_DIR)
    results = [run_subset(data_dir, s, n) for s, n in SUBSETS.items()]
    out = {"model": "1D-CNN (30-cycle windows), per-condition sensor standardisation",
           "protocol": f"official test engines, last cycle, RUL capped at {RUL_CAP}", "results": results}
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    for r in results:
        print(f"{r['subset']}: RMSE {r['rmse']:.2f}  NASA {r['nasa_score']:.0f}  (val {r['val_rmse']:.2f}, {r['epochs_run']} epochs)")


if __name__ == "__main__":
    main()
