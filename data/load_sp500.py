"""S&P 500 data loading module.

Assumption: Using yfinance for adjusted price data.
Historical constituent membership: using a static 100-stock representative list
from the S&P 500 as of 2017. Full CRSP survivorship-safe constituent data
requires licensed access (WRDS/CRSP) which is not available.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import List, Optional

DATA_DIR = Path(__file__).parent


def get_sp500_constituents_2017() -> List[str]:
    """Return representative list of S&P 500 constituents active in 2017.

    Documented assumption: This is a static 100-stock list, not full CRSP
    historical constituent data. Does not fully eliminate survivorship bias
    but provides a reasonable benchmark for replication purposes.
    """
    from src.benchmark import SP500_2017_TICKERS
    return SP500_2017_TICKERS


def load_or_download_prices(tickers: List[str], start: str, end: str,
                             cache_path: Optional[str] = None) -> pd.DataFrame:
    """Load prices from cache if available, otherwise download via yfinance.

    Args:
        tickers: List of ticker symbols to fetch.
        start: Start date string 'YYYY-MM-DD'.
        end: End date string 'YYYY-MM-DD'.
        cache_path: Optional path to a CSV file used as a local price cache.
                    If the file exists it is loaded directly; otherwise prices
                    are downloaded and, if cache_path is provided, written there.

    Returns:
        DataFrame with tickers as columns and trading dates as index.
    """
    if cache_path is not None:
        cache = Path(cache_path)
        if cache.exists():
            prices = pd.read_csv(cache, index_col=0, parse_dates=True)
            prices.index = pd.DatetimeIndex(prices.index)
            return prices

    from src.benchmark import download_sp500_prices
    prices = download_sp500_prices(tickers, start, end)

    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache)

    return prices
