import pandas as pd
import shap

_explainers = {}


def _explainer(bundle):
    """SHAP explainer matching the deployed failure model, cached per bundle.
    Contributions are in log-odds of 'failure within the window'."""
    key = id(bundle)
    if key not in _explainers:
        model = bundle["failure"]["model"]
        if bundle["failure"]["algorithm"] == "LogisticRegression":
            scaler, estimator = model[0], model[-1]
            background = scaler.transform(pd.DataFrame(bundle["shap_background"], columns=bundle["feature_cols"]))
            _explainers[key] = ("linear", shap.LinearExplainer(estimator, background), scaler)
        else:
            _explainers[key] = ("tree", shap.TreeExplainer(model), None)
    return _explainers[key]


def explain_rows(bundle, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature SHAP contributions for each row of X (same columns as X)."""
    kind, explainer, scaler = _explainer(bundle)
    values = explainer.shap_values(scaler.transform(X) if scaler is not None else X)
    if isinstance(values, list):  # older SHAP returns one array per class
        values = values[1]
    if getattr(values, "ndim", 2) == 3:
        values = values[:, :, 1]
    return pd.DataFrame(values, columns=X.columns, index=X.index)
