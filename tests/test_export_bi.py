import pandas as pd

from scripts.export_bi import metrics_long, risk_band


def test_risk_band_uses_served_threshold():
    p = pd.Series([0.1, 0.5, 0.8, 0.95])
    assert risk_band(p, 0.8).tolist() == ["Low", "Elevated", "High", "High"]


def test_metrics_long_flattens_models_and_skips_nested():
    results = {
        "rul": {"models": {"RF": {"rmse": 17.2, "cv_rmse": 20.0}}},
        "failure": {"models": {"LR": {"pr_auc": 0.8, "confusion": {"tp": 1}}}},
        "anomaly": {"true RUL <= 30": {"flag_rate": 0.9, "cycles": 10}},
    }
    df = metrics_long(results)
    assert len(df) == 4 and set(df.columns) == {"task", "model", "metric", "value"}
    assert "confusion" not in set(df.metric)
