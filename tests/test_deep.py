import numpy as np
import pytest

torch = pytest.importorskip("torch")  # optional benchmark: skipped when torch isn't installed

from ml.data.cmapss import SENSORS, load_test, load_train  # noqa: E402
from ml.deep.cnn_rul import CNN, WINDOW, make_windows, nasa_score, train  # noqa: E402


def test_windows_are_left_padded_for_short_histories(fixture_dir):
    df = load_train(fixture_dir).query("engine_id == 1").head(5)
    X, y, ids = make_windows(df)
    assert X.shape == (5, WINDOW, len(SENSORS)) and len(y) == 5 and set(ids) == {1}
    first_reading = df.sort_values("cycle")[SENSORS].to_numpy(dtype=np.float32)[0]
    assert np.allclose(X[0][:-1], first_reading)  # padding repeats the first reading
    assert np.allclose(X[0][-1], first_reading)


def test_last_only_gives_one_window_per_engine(fixture_dir):
    df = load_test(fixture_dir)
    X, y, ids = make_windows(df, last_only=True)
    assert len(X) == df["engine_id"].nunique() and sorted(ids) == sorted(df["engine_id"].unique())
    # The window ends at the engine's final cycle, whose label is NASA's capped final RUL.
    last = df.sort_values("cycle").groupby("engine_id").tail(1).sort_values("engine_id")
    assert np.allclose(y, last["rul_capped"].to_numpy())


def test_model_output_shape():
    out = CNN()(torch.zeros(4, WINDOW, len(SENSORS)))
    assert out.shape == (4,)


def test_nasa_score_penalises_late_predictions_more():
    assert nasa_score([50], [60]) > nasa_score([50], [40])
    assert nasa_score([50], [50]) == 0


def test_training_runs_and_stays_within_the_rul_range(fixture_dir):
    from ml.deep.cnn_rul import _predict

    model, mean, std, info = train(load_train(fixture_dir), epochs=2, patience=2)
    X, _, _ = make_windows(load_test(fixture_dir), last_only=True)
    pred = _predict(model, X, mean, std)
    assert info["epochs_run"] >= 1 and np.isfinite(info["val_rmse"])
    assert pred.min() >= 0 and pred.max() <= 125
