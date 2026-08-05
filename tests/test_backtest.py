"""Tests for the music royalty backtest engine."""

import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/project')
from src.backtest_engine import (
    price_model3,
    compute_transaction_costs,
    compute_return_components,
    run_one_year_backtest,
    run_five_year_backtest,
    DEFAULT_PARAMS,
    compute_state_variables,
)


# ---------------------------------------------------------------------------
# Transaction cost tests
# ---------------------------------------------------------------------------

def test_transaction_costs_zero_exit():
    """Fixed fee still applies even when exit price is zero."""
    assert compute_transaction_costs(0.0) == 500.0


def test_transaction_costs_formula():
    """Verify t = 500 + 0.08 * p_exit."""
    assert compute_transaction_costs(10_000) == pytest.approx(500 + 800)
    assert compute_transaction_costs(50_000) == pytest.approx(500 + 4_000)
    assert compute_transaction_costs(100_000) == pytest.approx(500 + 8_000)


def test_transaction_costs_proportional():
    """Doubling p_exit should nearly double total costs (fixed fee breaks linearity)."""
    t1 = compute_transaction_costs(50_000)
    t2 = compute_transaction_costs(100_000)
    # Difference should equal 8% of the extra 50k = 4000
    assert abs((t2 - t1) - 4_000) < 1e-6


# ---------------------------------------------------------------------------
# Pricing tests
# ---------------------------------------------------------------------------

def test_price_model3_lor_is_perpetuity():
    """LOR price = LTM * (a + b*age) / R for perfectly stable revenue."""
    params = DEFAULT_PARAMS
    ltm, lty, age = 10_000.0, 10_000.0, 10.0
    p = price_model3(ltm, lty, age, 'LOR', 0, params)
    R = params['r']  # s=0 for ltm==lty
    x = params['a'] + params['b'] * age
    expected = ltm * x / R
    assert abs(p - expected) < 1.0


def test_price_model3_10y_lower_than_lor():
    """10Y contract should be priced below LOR for the same asset."""
    p10 = price_model3(10_000, 10_000, 10.0, '10Y', 10, DEFAULT_PARAMS)
    plor = price_model3(10_000, 10_000, 10.0, 'LOR', 0, DEFAULT_PARAMS)
    assert p10 < plor


def test_price_model3_30y_between_10y_and_lor():
    """30Y price should lie between 10Y and LOR prices."""
    ltm, lty, age = 10_000, 10_000, 10.0
    p10 = price_model3(ltm, lty, age, '10Y', 10, DEFAULT_PARAMS)
    p30 = price_model3(ltm, lty, age, '30Y', 30, DEFAULT_PARAMS)
    plor = price_model3(ltm, lty, age, 'LOR', 0, DEFAULT_PARAMS)
    assert p10 < p30 < plor


def test_price_model3_positive():
    """All contract types should yield positive prices for valid inputs."""
    for term, years in [('10Y', 10), ('30Y', 30), ('LOR', 0)]:
        p = price_model3(10_000, 10_000, 5.0, term, years, DEFAULT_PARAMS)
        assert p > 0


def test_price_model3_higher_ltm_higher_price():
    """Higher LTM should yield a proportionally higher price (linear in LTM)."""
    p1 = price_model3(10_000, 10_000, 10.0, 'LOR', 0, DEFAULT_PARAMS)
    p2 = price_model3(20_000, 20_000, 10.0, 'LOR', 0, DEFAULT_PARAMS)
    assert abs(p2 / p1 - 2.0) < 1e-6


def test_price_model3_older_asset_higher_lor_price():
    """For LOR, older asset (larger age) should yield higher price when b > 0."""
    p_young = price_model3(10_000, 10_000, 5.0, 'LOR', 0, DEFAULT_PARAMS)
    p_old = price_model3(10_000, 10_000, 20.0, 'LOR', 0, DEFAULT_PARAMS)
    assert p_old > p_young


# ---------------------------------------------------------------------------
# Return component tests
# ---------------------------------------------------------------------------

def test_return_components_known_values():
    """Verify return decomposition with hand-calculated values."""
    result = compute_return_components(
        p_purchase=100_000, p_exit=105_000, cashflows=8_000, costs=9_000
    )
    # total = (105000 + 8000 - 100000 - 9000) / 100000 = 4000 / 100000 = 0.04
    assert result['r_i'] == pytest.approx(0.04, abs=1e-10)
    # d_i = 8000 / 100000 = 0.08
    assert result['d_i'] == pytest.approx(0.08, abs=1e-10)
    # e_i = (105000 - 100000) / 100000 = 0.05
    assert result['e_i'] == pytest.approx(0.05, abs=1e-10)
    # f_i = 9000 / 100000 = 0.09
    assert result['f_i'] == pytest.approx(0.09, abs=1e-10)


def test_return_decomposition_identity():
    """Total return must equal d_i + e_i - f_i (accounting identity)."""
    result = compute_return_components(50_000, 55_000, 4_000, 4_500)
    assert result['r_i'] == pytest.approx(
        result['d_i'] + result['e_i'] - result['f_i'], abs=1e-10
    )


def test_return_components_negative_total_possible():
    """If costs + price decline exceed income, total return should be negative."""
    result = compute_return_components(100_000, 80_000, 5_000, 10_000)
    assert result['r_i'] < 0


def test_return_components_raises_on_zero_purchase_price():
    """Should raise ValueError when p_purchase = 0."""
    with pytest.raises((ValueError, ZeroDivisionError)):
        compute_return_components(0, 10_000, 1_000, 500)


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------

def test_annualised_return_formula_lor():
    """Five-year LOR target: ~82.64% total ≈ 12.8% annualised."""
    r_total = 0.8264
    ann = (1 + r_total) ** (1 / 5) - 1
    assert abs(ann - 0.128) < 0.005


def test_annualised_return_formula_10y():
    """Five-year 10Y target: ~42.5% total ≈ 7.3% annualised."""
    r_total = 0.425
    ann = (1 + r_total) ** (1 / 5) - 1
    assert abs(ann - 0.073) < 0.005


# ---------------------------------------------------------------------------
# State variable tests
# ---------------------------------------------------------------------------

def test_compute_state_variables_stable():
    """For ltm == lty, instability s should be 0."""
    sv = compute_state_variables(10_000.0, 10_000.0, 10.0)
    assert sv['s'] == pytest.approx(0.0, abs=1e-10)
    assert sv['R'] == pytest.approx(DEFAULT_PARAMS['r'], abs=1e-10)


def test_compute_state_variables_unstable():
    """Instability should increase effective discount rate."""
    sv_stable = compute_state_variables(10_000.0, 10_000.0, 10.0)
    sv_unstable = compute_state_variables(10_000.0, 8_000.0, 10.0)
    assert sv_unstable['R'] > sv_stable['R']


# ---------------------------------------------------------------------------
# Backtest integration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_backtest_data():
    """Minimal 20-asset dataset for backtest engine integration testing."""
    n = 20
    np.random.seed(7)
    ltm_vals = np.random.uniform(5_000, 20_000, n)
    df = pd.DataFrame({
        'transaction_id': range(n),
        'asset_id': range(n),
        'transaction_date': pd.date_range('2017-01-01', periods=n, freq='ME'),
        'ltm': ltm_vals,
        'lty': ltm_vals * np.random.uniform(0.8, 1.2, n),
        'age': np.random.uniform(3, 15, n),
        'term': np.random.choice(['10Y', 'LOR'], n),
        'price': ltm_vals * np.random.uniform(3, 12, n),
        'is_primary': True,
    })
    rows = []
    for i in range(n):
        base_rev = ltm_vals[i] / 4
        for q in range(24):
            rows.append({
                'asset_id': i,
                'quarter_date': pd.Timestamp('2015-01-01') + pd.DateOffset(months=q * 3),
                'revenue': base_rev * (0.97 ** q) + np.random.normal(0, 50),
            })
    quarters = pd.DataFrame(rows)
    return df, quarters


def test_one_year_backtest_returns_dataframe(minimal_backtest_data):
    """One-year backtest must return a non-empty DataFrame."""
    df, quarters = minimal_backtest_data
    result = run_one_year_backtest(df, quarters)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_one_year_backtest_required_columns(minimal_backtest_data):
    """One-year backtest result must have required return columns."""
    df, quarters = minimal_backtest_data
    result = run_one_year_backtest(df, quarters)
    required = ['asset_id', 'cohort_year', 'term', 'p_purchase', 'p_exit',
                'cashflows_1y', 'costs', 'd_i', 'e_i', 'f_i', 'r_i']
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_five_year_backtest_returns_dataframe(minimal_backtest_data):
    """Five-year backtest must return a non-empty DataFrame."""
    df, quarters = minimal_backtest_data
    result = run_five_year_backtest(df, quarters)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_five_year_backtest_has_annualized_return(minimal_backtest_data):
    """Five-year result must include ann_r_i column."""
    df, quarters = minimal_backtest_data
    result = run_five_year_backtest(df, quarters)
    assert 'ann_r_i' in result.columns


def test_return_decomposition_holds_in_backtest(minimal_backtest_data):
    """r_i ≈ d_i + e_i - f_i must hold for every row in one-year backtest."""
    df, quarters = minimal_backtest_data
    result = run_one_year_backtest(df, quarters)
    for _, row in result.iterrows():
        computed = row['d_i'] + row['e_i'] - row['f_i']
        assert abs(row['r_i'] - computed) < 1e-8, (
            f"Decomposition mismatch: r_i={row['r_i']:.6f}, "
            f"d_i+e_i-f_i={computed:.6f}"
        )


def test_five_year_backtest_only_2017_cohort(minimal_backtest_data):
    """Five-year backtest must only return assets whose cohort year is 2017."""
    df, quarters = minimal_backtest_data
    result = run_five_year_backtest(df, quarters)
    # Fixture spans 2017-01 to 2018-08 (20 months); only the 12 that fall in
    # 2017 should be included in the five-year cohort.
    n_2017 = (df['transaction_date'].dt.year == 2017).sum()
    assert len(result) == n_2017
    if 'cohort_year' in result.columns:
        assert (result['cohort_year'] == 2017).all()


def test_backtest_prices_positive(minimal_backtest_data):
    """Purchase and exit prices must be positive."""
    df, quarters = minimal_backtest_data
    result = run_one_year_backtest(df, quarters)
    assert (result['p_purchase'] > 0).all()
    assert (result['p_exit'] > 0).all()
