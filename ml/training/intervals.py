"""Split-conformal prediction intervals for the RUL regressor.

The half-width is a quantile of engine-wise OUT-OF-FOLD absolute residuals on
the training engines, so no engine is scored by a model that saw it and the
official test engines are never used. Under exchangeability the interval
[pred - q, pred + q] covers the true (capped) RUL with probability >= 1 - alpha.
Engines are not independent cycles, so test coverage is reported by
evaluation/run_all.py rather than assumed.
"""
import numpy as np

DEFAULT_ALPHA = 0.10


def conformal_halfwidth(residuals, alpha: float = DEFAULT_ALPHA) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest |residual| (finite-sample-corrected quantile)."""
    abs_res = np.abs(np.asarray(residuals, dtype=float))
    n = len(abs_res)
    if n == 0:
        raise ValueError("need at least one residual")
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(abs_res)[k - 1])


def interval_bounds(pred: float, halfwidth: float, cap: float) -> tuple[float, float]:
    """Interval around a prediction, clipped to the feasible range [0, cap]."""
    return max(0.0, pred - halfwidth), min(float(cap), pred + halfwidth)


def coverage(y_true, pred, halfwidth: float, cap: float) -> float:
    y, p = np.asarray(y_true, dtype=float), np.asarray(pred, dtype=float)
    lo, hi = np.maximum(0.0, p - halfwidth), np.minimum(cap, p + halfwidth)
    return float(np.mean((y >= lo) & (y <= hi)))
