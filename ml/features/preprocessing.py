"""
Feature engineering shared by training, evaluation and the API.

Each row describes one engine at one cycle, using only that engine's readings up
to and including that cycle: the raw sensors, the cycle count (engine age), and
rolling means/standard deviations over the last 5 and 15 cycles.

Training only uses rows with at least MIN_HISTORY cycles of history, and the API
requires at least MIN_HISTORY readings. That way a served prediction is computed
from exactly the same kind of feature vector the models were trained on. (An
earlier version fed the API a single reading, which silently turned rolling
means into the raw value and rolling stds into 0 — inputs the models had never
seen.)
"""
from typing import List

import pandas as pd

from ml.data.cmapss import SENSORS

WINDOWS = [5, 15]
MIN_HISTORY = max(WINDOWS)
FEATURE_COLS: List[str] = (
    ["cycle"] + SENSORS
    + [f"{s}_roll_{stat}_{w}" for w in WINDOWS for s in SENSORS for stat in ("mean", "std")]
)


def build_features(df: pd.DataFrame, full_history_only: bool = True) -> pd.DataFrame:
    """Adds rolling features per engine. Rows must be ordered by cycle within each
    engine. With full_history_only, rows with fewer than MIN_HISTORY cycles of
    history are dropped (their windows would be incomplete)."""
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("engine_id", sort=False)
    new = {}
    for w in WINDOWS:
        rolling = grouped[SENSORS].rolling(window=w, min_periods=w)
        means = rolling.mean().reset_index(level=0, drop=True)
        stds = rolling.std().reset_index(level=0, drop=True)
        for s in SENSORS:
            new[f"{s}_roll_mean_{w}"] = means[s]
            new[f"{s}_roll_std_{w}"] = stds[s]
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    if full_history_only:
        out = out[grouped.cumcount() + 1 >= MIN_HISTORY]
    return out


def latest_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Feature row for the most recent reading of one engine's history (what the
    API predicts from). Requires at least MIN_HISTORY readings."""
    if len(readings) < MIN_HISTORY:
        raise ValueError(f"At least {MIN_HISTORY} consecutive readings are required, got {len(readings)}.")
    featured = build_features(readings.assign(engine_id=0), full_history_only=True)
    return featured[FEATURE_COLS].tail(1)
