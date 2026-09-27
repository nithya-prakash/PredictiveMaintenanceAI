from ml.data.cmapss import CONSTANT_SENSORS, FAILURE_WINDOW, RUL_CAP, SENSORS, load_test, load_train


def test_training_labels(fixture_dir):
    df = load_train(fixture_dir)
    last = df.groupby("engine_id").tail(1)
    assert (last["rul"] == 0).all()  # training engines run until failure
    assert df["rul_capped"].max() == RUL_CAP
    assert ((df["rul"] <= FAILURE_WINDOW) == df["failure_imminent"].astype(bool)).all()


def test_test_rul_is_reconstructed_from_nasa_final_values(fixture_dir):
    df = load_test(fixture_dir)
    final = [int(x) for x in (fixture_dir / "RUL_FD001.txt").read_text().split()]
    last = df.groupby("engine_id").tail(1)
    assert list(last["rul"]) == final
    # Earlier cycles have more life left, one cycle per row.
    e1 = df[df.engine_id == 1]
    assert (e1["rul"].diff().dropna() == -1).all()


def test_constant_sensors_are_dropped(fixture_dir):
    df = load_train(fixture_dir)
    assert len(SENSORS) == 14
    assert not set(CONSTANT_SENSORS) & set(df.columns)
