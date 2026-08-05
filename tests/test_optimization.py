"""Tests for DCF model calibration/optimization module."""

import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/project')
from src.optimization import calibrate_model1, calibrate_model2, calibrate_model3
from src.models import model3_multiplier


@pytest.fixture
def tiny_dataset():
    """Small synthetic dataset generated with known Model 3 params for fast testing."""
    np.random.seed(99)
    n = 50
    ages = np.random.uniform(2, 20, n)
    terms = np.random.choice(['10Y', 'LOR'], n)
    ltm = np.random.uniform(5000, 30000, n)
    lty = ltm * (1 + np.random.normal(0, 0.15, n))
    lty = np.abs(lty) + 1.0  # ensure positive

    # Generate prices using Model 3 true params with noise
    r, a, k, b = 0.083, 0.61, 0.058, 0.0098
    prices = []
    for i in range(n):
        s = abs(ltm[i] / lty[i] - 1)
        m = model3_multiplier(s, ages[i], terms[i], r, a, k, b)
        prices.append(ltm[i] * (m + np.random.normal(0, 1.5)))
    prices = np.maximum(prices, np.array([l * 0.1 for l in ltm]))

    return pd.DataFrame({'ltm': ltm, 'lty': lty, 'age': ages, 'term': terms, 'price': prices})


def test_calibrate_model1_returns_valid_params(tiny_dataset):
    """Model 1 calibration must return a dict with valid positive r."""
    result = calibrate_model1(tiny_dataset)
    assert 'params' in result
    assert 'mse' in result
    assert 'fitted_multipliers' in result
    assert 0 < result['params']['r'] < 1.0
    assert result['mse'] > 0


def test_calibrate_model2_returns_valid_params(tiny_dataset):
    """Model 2 calibration must return r, a, k all economically valid."""
    result = calibrate_model2(tiny_dataset)
    p = result['params']
    assert all(k in p for k in ['r', 'a', 'k'])
    assert p['r'] > 0
    assert p['a'] > 0
    assert p['k'] >= 0
    assert result['mse'] > 0


def test_calibrate_model3_returns_valid_params(tiny_dataset):
    """Model 3 calibration must return r, a, k, b all economically valid."""
    result = calibrate_model3(tiny_dataset)
    p = result['params']
    assert all(key in p for key in ['r', 'a', 'k', 'b'])
    assert p['r'] > 0
    assert p['a'] > 0
    assert p['k'] >= 0
    assert p['b'] >= 0
    assert result['mse'] > 0


def test_model3_better_than_model1(tiny_dataset):
    """Model 3 should achieve strictly lower MSE than Model 1."""
    r1 = calibrate_model1(tiny_dataset)
    r3 = calibrate_model3(tiny_dataset)
    assert r3['mse'] < r1['mse'], (
        f"Expected MSE3 < MSE1, got MSE3={r3['mse']:.4f}, MSE1={r1['mse']:.4f}"
    )


def test_mse_ranking_all_models(tiny_dataset):
    """MSE ranking should be Model 3 <= Model 2 <= Model 1."""
    r1 = calibrate_model1(tiny_dataset)
    r2 = calibrate_model2(tiny_dataset)
    r3 = calibrate_model3(tiny_dataset)
    assert r3['mse'] <= r2['mse'] <= r1['mse'], (
        f"MSE ordering violated: M3={r3['mse']:.4f}, M2={r2['mse']:.4f}, M1={r1['mse']:.4f}"
    )


def test_fitted_multipliers_length(tiny_dataset):
    """fitted_multipliers array must have same length as input DataFrame."""
    result = calibrate_model3(tiny_dataset)
    assert len(result['fitted_multipliers']) == len(tiny_dataset)


def test_fitted_multipliers_positive(tiny_dataset):
    """All fitted multipliers should be positive for valid inputs."""
    result = calibrate_model3(tiny_dataset)
    assert (result['fitted_multipliers'] > 0).all()


def test_calibration_lor_only(tiny_dataset):
    """Calibration should work when only LOR term is present."""
    lor_only = tiny_dataset[tiny_dataset['term'] == 'LOR'].copy()
    if len(lor_only) >= 5:
        result = calibrate_model3(lor_only)
        assert result['mse'] > 0
        assert result['params']['r'] > 0


def test_calibration_10y_only(tiny_dataset):
    """Calibration should work when only 10Y term is present."""
    ten_y_only = tiny_dataset[tiny_dataset['term'] == '10Y'].copy()
    if len(ten_y_only) >= 5:
        result = calibrate_model3(ten_y_only)
        assert result['mse'] > 0


def test_mse_is_positive(tiny_dataset):
    """MSE must always be strictly positive for non-trivial data."""
    for calibrate_fn in [calibrate_model1, calibrate_model2, calibrate_model3]:
        result = calibrate_fn(tiny_dataset)
        assert result['mse'] > 0, f"{calibrate_fn.__name__} returned non-positive MSE"


def test_model3_params_near_paper_values():
    """On full synthetic dataset, Model 3 params should be near paper targets."""
    sys.path.insert(0, '/workspace/project/data')
    from generate_royalty_data import generate_royalty_dataset
    df = generate_royalty_dataset(n=1295, seed=42)
    result = calibrate_model3(df)
    p = result['params']
    # Paper targets: r=8.3%, a=0.61, k=0.058, b=0.0098 — allow generous tolerance
    assert 0.04 < p['r'] < 0.20, f"r={p['r']:.4f} out of range"
    assert 0.3 < p['a'] < 1.5,   f"a={p['a']:.4f} out of range"
    assert 0.0 <= p['k'] < 0.3,  f"k={p['k']:.4f} out of range"
    assert 0.0 <= p['b'] < 0.05, f"b={p['b']:.6f} out of range"


def test_model3_mse_near_paper_value():
    """Model 3 MSE on full synthetic dataset should be near paper target of ~5.7."""
    sys.path.insert(0, '/workspace/project/data')
    from generate_royalty_data import generate_royalty_dataset
    df = generate_royalty_dataset(n=1295, seed=42)
    result = calibrate_model3(df)
    # Allow broad tolerance: synthetic data may not perfectly match paper noise level
    assert result['mse'] < 15.0, f"MSE={result['mse']:.4f} much larger than expected"
