import pytest
import numpy as np
import sys
sys.path.insert(0, '/workspace/project')
from src.models import (
    finite_horizon_multiplier, perpetuity_multiplier, get_horizon,
    model1_multiplier, model2_multiplier, model3_multiplier,
    model3_multiplier_vec
)


def test_finite_horizon_multiplier_basic():
    """Test annuity formula with known values."""
    # For x=1, R=0.1, N=1: should be 1/(1.1) ≈ 0.9091
    result = finite_horizon_multiplier(1.0, 0.1, 1)
    assert abs(result - 1/1.1) < 1e-10


def test_finite_horizon_multiplier_n10():
    """Test 10-year annuity with R=0.1, x=1."""
    expected = (1 - (1.1)**(-10)) / 0.1
    result = finite_horizon_multiplier(1.0, 0.1, 10)
    assert abs(result - expected) < 1e-10


def test_perpetuity_multiplier():
    """Test perpetuity: x/R."""
    result = perpetuity_multiplier(1.0, 0.1)
    assert abs(result - 10.0) < 1e-10


def test_perpetuity_multiplier_scaling():
    """Multiplier should scale linearly with x."""
    r1 = perpetuity_multiplier(2.0, 0.1)
    r2 = perpetuity_multiplier(1.0, 0.1)
    assert abs(r1 - 2*r2) < 1e-10


def test_model1_lor():
    """Model 1 LOR = 1/r."""
    m = model1_multiplier('LOR', 0.10)
    assert abs(m - 10.0) < 1e-10


def test_model1_10y():
    """Model 1 10Y is a finite annuity."""
    m = model1_multiplier('10Y', 0.10)
    expected = (1 - 1.1**(-10)) / 0.1
    assert abs(m - expected) < 1e-10


def test_model2_higher_instability_lower_multiplier():
    """Higher instability should lower multiplier in Model 2."""
    m_stable = model2_multiplier(0.0, 'LOR', 0.08, 0.69, 0.071)
    m_unstable = model2_multiplier(1.0, 'LOR', 0.08, 0.69, 0.071)
    assert m_stable > m_unstable


def test_model3_higher_age_higher_multiplier():
    """Older assets should have higher Model 3 multiplier when b > 0."""
    m_young = model3_multiplier(0.0, 5.0, 'LOR', 0.083, 0.61, 0.058, 0.0098)
    m_old = model3_multiplier(0.0, 20.0, 'LOR', 0.083, 0.61, 0.058, 0.0098)
    assert m_old > m_young


def test_model3_near_ltm_lty_higher_multiplier():
    """Assets with LTM/LTY near 1 (s≈0) should have higher multiplier."""
    m_stable = model3_multiplier(0.0, 10.0, '10Y', 0.083, 0.61, 0.058, 0.0098)
    m_unstable = model3_multiplier(0.5, 10.0, '10Y', 0.083, 0.61, 0.058, 0.0098)
    assert m_stable > m_unstable


def test_model3_multiplier_vec_matches_scalar():
    """Vectorized version should match scalar."""
    s = np.array([0.0, 0.1, 0.5])
    age = np.array([5.0, 10.0, 15.0])
    term = np.array(['10Y', 'LOR', '10Y'])
    r, a, k, b = 0.083, 0.61, 0.058, 0.0098
    vec_result = model3_multiplier_vec(s, age, term, r, a, k, b)
    for i in range(3):
        scalar_result = model3_multiplier(s[i], age[i], term[i], r, a, k, b)
        assert abs(vec_result[i] - scalar_result) < 1e-10


def test_multiplier_positive():
    """All multipliers should be positive for valid inputs."""
    assert model1_multiplier('10Y', 0.05) > 0
    assert model1_multiplier('LOR', 0.05) > 0
    assert model2_multiplier(0.1, '10Y', 0.05, 0.5, 0.05) > 0
    assert model3_multiplier(0.1, 10.0, 'LOR', 0.083, 0.61, 0.058, 0.0098) > 0


def test_30y_multiplier_between_10y_and_lor():
    """For same parameters, 30Y multiplier should be between 10Y and LOR."""
    m10 = model1_multiplier('10Y', 0.10)
    m30 = model1_multiplier('30Y', 0.10)
    mlor = model1_multiplier('LOR', 0.10)
    assert m10 < m30 < mlor
