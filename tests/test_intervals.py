import numpy as np
import pytest

from ml.training.intervals import conformal_halfwidth, coverage, interval_bounds


def test_halfwidth_is_a_finite_sample_corrected_quantile():
    residuals = np.arange(1, 100)  # |r| = 1..99
    # ceil(100 * 0.9) = 90 -> the 90th smallest of 99 residuals.
    assert conformal_halfwidth(residuals, alpha=0.1) == 90.0


def test_halfwidth_ignores_residual_sign():
    assert conformal_halfwidth([-5, 5, -5, 5], alpha=0.1) == 5.0


def test_halfwidth_needs_data():
    with pytest.raises(ValueError):
        conformal_halfwidth([])


def test_bounds_are_clipped_to_the_feasible_range():
    assert interval_bounds(10, 31, cap=125) == (0.0, 41.0)
    assert interval_bounds(120, 31, cap=125) == (89.0, 125.0)


def test_empirical_coverage_matches_nominal_on_exchangeable_data():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(0, 10, 5000), rng.normal(0, 10, 5000)
    hw = conformal_halfwidth(cal, alpha=0.1)
    # Shift to positive RUL-like values so clipping at 0 does not interfere.
    assert coverage(test + 100, np.full(5000, 100.0), hw, cap=1e9) == pytest.approx(0.9, abs=0.02)
