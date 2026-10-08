from backend.services import drift
from backend.services.drift import DriftMonitor, MIN_OBSERVATIONS

REF = {"s2": {"mean": 100.0, "std": 10.0}, "s3": {"mean": 50.0, "std": 5.0}}


def _feed(monitor, n, s2, s3):
    for _ in range(n):
        monitor.observe({"s2": s2, "s3": s3})


def test_shift_is_in_training_std_units_and_counts_drifted_sensors():
    m = DriftMonitor(REF)
    _feed(m, MIN_OBSERVATIONS, 120.0, 50.0)  # s2 +2 std, s3 unchanged
    assert round(m.shifts()["s2"], 6) == 2.0 and round(m.shifts()["s3"], 6) == 0.0
    assert drift.input_shift.labels(sensor="s2")._value.get() == 2.0
    assert drift.drifted_sensors._value.get() == 1


def test_window_forgets_old_observations():
    m = DriftMonitor(REF, window=MIN_OBSERVATIONS)
    _feed(m, MIN_OBSERVATIONS, 150.0, 50.0)
    _feed(m, MIN_OBSERVATIONS, 100.0, 50.0)
    assert m.shifts()["s2"] == 0.0


def test_missing_reference_disables_monitor_without_error(tmp_path):
    m = DriftMonitor.from_file(str(tmp_path / "nope.json"))
    m.observe({"s2": 1.0})
    assert m.shifts() == {}


def test_real_reference_file_loads():
    m = DriftMonitor.from_file("models/drift_reference.json")
    assert len(m.reference) == 14 and all(v["std"] > 0 for v in m.reference.values())
