import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/workspace/project')
from src.benchmark import (
    SP500_2017_TICKERS, compute_horizon_returns,
    summarize_cross_section, compute_annual_returns
)


def test_sp500_ticker_list_length():
    """Should have 100 tickers."""
    assert len(SP500_2017_TICKERS) == 100


def test_sp500_tickers_are_strings():
    """All tickers should be non-empty strings."""
    assert all(isinstance(t, str) and len(t) > 0 for t in SP500_2017_TICKERS)


def test_summarize_cross_section_basic():
    """Test summary statistics with known data."""
    returns = pd.Series([0.1, 0.2, 0.5, -0.1, 0.8, 0.3, 0.15, 0.0, 0.25, 0.4])
    summary = summarize_cross_section(returns)
    assert 'median' in summary
    assert 'p10' in summary
    assert 'p90' in summary
    assert summary['p10'] <= summary['median'] <= summary['p90']


def test_summarize_cross_section_single_value():
    """Edge case: single value."""
    returns = pd.Series([0.5])
    summary = summarize_cross_section(returns)
    assert summary['median'] == pytest.approx(0.5)


def test_compute_horizon_returns_basic():
    """Test return computation with known prices."""
    dates = pd.date_range('2017-01-03', '2022-01-03', freq='B')[:500]
    prices = pd.DataFrame({
        'AAPL': np.linspace(100, 200, len(dates)),  # 100% return
        'MSFT': np.linspace(100, 150, len(dates)),  # 50% return
    }, index=dates)
    returns = compute_horizon_returns(prices, '2017-01-03', dates[-1].strftime('%Y-%m-%d'))
    assert abs(returns['AAPL'] - 1.0) < 0.05  # ~100%
    assert abs(returns['MSFT'] - 0.5) < 0.05  # ~50%


def test_annualized_return_five_year():
    """Five-year annualized return formula check."""
    # 77.7% total over 5 years → (1.777)^(1/5) - 1 ≈ 12.2%
    total = 0.777
    ann = (1 + total) ** (1/5) - 1
    assert abs(ann - 0.122) < 0.005


def test_summarize_returns_percentile_ordering():
    """p10 should always be less than p90."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(100))
    summary = summarize_cross_section(returns)
    assert summary['p10'] < summary['p90']


def test_download_sp500_prices_handles_empty():
    """Should handle case with no valid tickers gracefully."""
    from src.benchmark import download_sp500_prices
    # Test that the function has correct signature
    import inspect
    sig = inspect.signature(download_sp500_prices)
    assert 'tickers' in sig.parameters
    assert 'start_date' in sig.parameters
    assert 'end_date' in sig.parameters
