"""S&P 500 benchmark backtest utilities.

Since historical S&P 500 constituent data requires CRSP/WRDS access,
we use a static list of large-cap stocks that were in the S&P 500 in 2017.
This is documented as an assumption: full survivorship-bias-free historical
constituents are not available; we use a representative 100-stock subset.

For prices we use yfinance adjusted close (dividend-adjusted).
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional, Tuple

# Representative S&P 500 constituents as of 2017 (100 large-cap stocks)
# Assumption documented: full historical constituent list not available
SP500_2017_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'JNJ', 'JPM', 'XOM', 'BAC', 'WFC', 'CVX', 'PG',
    'UNH', 'HD', 'INTC', 'DIS', 'MRK', 'PEP', 'KO', 'ABT', 'WMT', 'CSCO',
    'VZ', 'T', 'MA', 'BMY', 'ABBV', 'MO', 'MMM', 'UPS', 'MDT', 'HON',
    'UTX', 'GE', 'AMGN', 'CMCSA', 'GILD', 'COST', 'AVGO', 'QCOM', 'TXN', 'ORCL',
    'SLB', 'USB', 'BK', 'AXP', 'GS', 'MS', 'C', 'BRK-B', 'V', 'IBM',
    'NKE', 'LLY', 'ACN', 'SBUX', 'MCD', 'PM', 'CAT', 'BA', 'F', 'GM',
    'DHR', 'EMR', 'ETN', 'ITW', 'DOW', 'DD', 'PPG', 'APD', 'SHW', 'ECL',
    'AIG', 'MET', 'TRV', 'PRU', 'AFL', 'LNC', 'HIG', 'UNM', 'MTB', 'KEY',
    'PNC', 'RF', 'CFG', 'FITB', 'HBAN', 'CMA', 'ZION', 'STI', 'BBT', 'STT',
    'SPG', 'O', 'PSA', 'EQR', 'AVB', 'PLD', 'ARE', 'VTR', 'WY', 'AMT'
]


def download_sp500_prices(tickers: List[str], start_date: str, end_date: str,
                           adjusted: bool = True) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers using yfinance.

    Args:
        tickers: List of ticker symbols
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        adjusted: If True, use adjusted close (default, includes dividends and splits)

    Returns:
        DataFrame with tickers as columns and dates as index. Columns with no
        data are dropped; failed tickers are silently excluded.
    """
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=adjusted,
        progress=False,
        threads=True,
    )
    prices = raw['Close'].copy()
    if isinstance(prices, pd.Series):
        # Guard for unexpected single-column result
        name = tickers[0] if isinstance(tickers, list) and tickers else 'UNKNOWN'
        prices = prices.to_frame(name)
    # Remove tickers for which no data was returned
    prices = prices.dropna(axis=1, how='all')
    return prices


def compute_horizon_returns(prices_df: pd.DataFrame, start_date: str,
                             end_date: str) -> pd.Series:
    """Compute total returns for each ticker over the specified horizon.

    Uses the nearest available trading day when the exact date is not present.

    Args:
        prices_df: Adjusted prices DataFrame (dates as index, tickers as columns)
        start_date: Start date for return computation ('YYYY-MM-DD')
        end_date: End date for return computation ('YYYY-MM-DD')

    Returns:
        Series of total returns indexed by ticker (NaN tickers dropped).
    """
    idx = prices_df.index
    ts_start = pd.Timestamp(start_date)
    ts_end = pd.Timestamp(end_date)

    # First available date >= start_date
    pos_start = idx.searchsorted(ts_start, side='left')
    # Last available date <= end_date
    pos_end = idx.searchsorted(ts_end, side='right') - 1

    pos_start = min(pos_start, len(idx) - 1)
    pos_end = min(max(pos_end, 0), len(idx) - 1)

    if pos_start > pos_end:
        return pd.Series(dtype=float)

    start_prices = prices_df.iloc[pos_start]
    end_prices = prices_df.iloc[pos_end]

    returns = (end_prices - start_prices) / start_prices
    return returns.dropna()


def compute_annual_returns(prices_df: pd.DataFrame, year: int) -> pd.Series:
    """Compute one-year returns for given calendar year start to end.

    Uses first and last available trading days of the year.
    """
    return compute_horizon_returns(prices_df, f'{year}-01-01', f'{year}-12-31')


def summarize_cross_section(returns: pd.Series) -> Dict:
    """Compute cross-sectional summary statistics.

    Returns dict with: median, p10, p90, mean, std, n_stocks
    """
    clean = returns.dropna()
    arr = clean.values.astype(float)
    return {
        'median': float(np.median(arr)),
        'p10': float(np.percentile(arr, 10)),
        'p90': float(np.percentile(arr, 90)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'n_stocks': int(len(arr)),
    }


def compute_benchmark_summary(tickers: Optional[List[str]] = None) -> Dict:
    """Compute full benchmark summary for one-year and five-year periods.

    Downloads price data for 2017-01-01 through 2022-12-31 and returns a
    structured summary dict with keys:
      'one_year'            – dict keyed by year (2017-2021)
      'five_year_total'     – cross-section summary of total returns
      'five_year_annualized'– cross-section summary of annualized returns
    """
    if tickers is None:
        tickers = SP500_2017_TICKERS

    prices = download_sp500_prices(tickers, '2017-01-01', '2022-12-31')

    one_year: Dict[int, Dict] = {}
    for year in range(2017, 2022):
        rets = compute_annual_returns(prices, year)
        one_year[year] = summarize_cross_section(rets)

    five_yr = compute_horizon_returns(prices, '2017-01-01', '2022-12-31')
    five_yr_ann = (1 + five_yr) ** (1 / 5) - 1

    return {
        'one_year': one_year,
        'five_year_total': summarize_cross_section(five_yr),
        'five_year_annualized': summarize_cross_section(five_yr_ann),
    }
