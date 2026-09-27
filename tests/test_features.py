import numpy as np
import pandas as pd
import pytest

from ml.data.cmapss import SENSORS, load_train
from ml.features.preprocessing import FEATURE_COLS, MIN_HISTORY, build_features, latest_features


def test_rolling_features_do_not_mix_engines():
    rows = []
    for engine, level in ((1, 0.0), (2, 1000.0)):
        for c in range(1, 21):
            rows.append({"engine_id": engine, "cycle": c, **{s: level + c for s in SENSORS}})
    feats = build_features(pd.DataFrame(rows))
    first_of_2 = feats[(feats.engine_id == 2)].iloc[0]  # engine 2 at cycle 15
    assert first_of_2["s2_roll_mean_15"] == pytest.approx(1000 + np.mean(range(1, 16)))


def test_rows_without_full_history_are_dropped(fixture_dir):
    feats = build_features(load_train(fixture_dir))
    assert feats.groupby("engine_id")["cycle"].min().eq(MIN_HISTORY).all()
    assert not feats[FEATURE_COLS].isna().any().any()


def test_serving_features_equal_training_features(fixture_dir):
    """Train/serve parity: the API builds features from an engine's most recent
    readings; they must equal the training feature row for that cycle."""
    df = load_train(fixture_dir)
    engine = df[df.engine_id == 2]
    training_row = build_features(df).query("engine_id == 2 and cycle == 120")[FEATURE_COLS]
    served = latest_features(engine[engine.cycle <= 120].tail(40)[["cycle"] + SENSORS])
    pd.testing.assert_frame_equal(served.reset_index(drop=True), training_row.reset_index(drop=True))


def test_single_reading_is_rejected():
    # The old API silently predicted from one reading (rolling std became 0).
    one = pd.DataFrame([{"cycle": 1, **{s: 1.0 for s in SENSORS}}])
    with pytest.raises(ValueError):
        latest_features(one)
