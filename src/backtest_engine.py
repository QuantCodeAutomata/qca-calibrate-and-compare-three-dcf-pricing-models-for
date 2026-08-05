"""Backtest engine for music royalty buy-and-hold strategies.

Implements one-year and five-year cross-sectional backtests using Model 3
pricing for both purchase and exit valuation.

Transaction-cost assumption (inferred from paper methodology):
    t_i = $500 fixed buyer fee + 8 % seller commission on exit price

Quarterly revenue extrapolation for synthetic data:
    future_rev_q = last_historical_rev * 0.97^q * (1 + ε_q),  ε_q ~ N(0, 0.05)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from .models import model3_multiplier

# Paper's calibrated Model 3 parameters (Table 1 fallback)
DEFAULT_PARAMS: Dict[str, float] = {
    "r": 0.083,
    "a": 0.61,
    "k": 0.058,
    "b": 0.0098,
}

# Number of future quarters generated per holding period
_QUARTERS_1Y = 4
_QUARTERS_5Y = 20


# ---------------------------------------------------------------------------
# State-variable helpers
# ---------------------------------------------------------------------------

def compute_state_variables(ltm: float, lty: float, age: float) -> Dict:
    """Compute purchase-state variables for Model 3 pricing.

    Uses the global DEFAULT_PARAMS to derive the effective discount rate R
    and cash-flow scalar x.

    Args:
        ltm: Last-twelve-months revenue (float, > 0).
        lty: Annualised last-three-year revenue (float, > 0).
        age: Asset age in years (float, > 0).

    Returns:
        Dict with keys: 'ltm', 'lty', 'age', 's', 'x', 'R'.
    """
    s = abs(ltm / lty - 1.0)
    r = DEFAULT_PARAMS["r"]
    k = DEFAULT_PARAMS["k"]
    a = DEFAULT_PARAMS["a"]
    b = DEFAULT_PARAMS["b"]
    R = r + k * s
    x = a + b * age
    return {"ltm": ltm, "lty": lty, "age": age, "s": s, "x": x, "R": R}


def price_model3(
    ltm: float,
    lty: float,
    age: float,
    term: str,
    remaining_years: int,
    params: Dict,
) -> float:
    """Price an asset using Model 3.

    Computes the DCF price given the asset's current state variables and the
    remaining contractual term.

    For finite term N:  p = LTM * (a + b*age) * (1 - (1+R)^{-N}) / R
    For LOR:            p = LTM * (a + b*age) / R

    Args:
        ltm:             Current last-twelve-months revenue (> 0).
        lty:             Current annualised last-three-year revenue (> 0).
        age:             Current asset age in years.
        term:            Original contract term string ('10Y', '30Y', 'LOR').
        remaining_years: Years left on contract (ignored for LOR).
        params:          Model 3 parameter dict with keys 'r', 'a', 'k', 'b'.

    Returns:
        Model-3-implied transaction price in dollars.
    """
    s = abs(ltm / lty - 1.0)
    r = params["r"]
    a = params["a"]
    k = params["k"]
    b = params["b"]

    if term == "LOR":
        mult = model3_multiplier(s, age, "LOR", r, a, k, b)
    else:
        # Build a synthetic term string for the remaining horizon
        # model3_multiplier accepts '10Y'/'30Y'/'LOR'; pass remaining as a
        # special call to the primitive so arbitrary horizons work.
        from .models import finite_horizon_multiplier, perpetuity_multiplier
        R = r + k * s
        x = a + b * age
        if remaining_years <= 0:
            return 0.0
        mult = finite_horizon_multiplier(x, R, remaining_years)
        return ltm * mult

    return ltm * mult


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------

def compute_transaction_costs(p_exit: float) -> float:
    """Compute total round-trip transaction costs.

    Inferred assumption: $500 fixed buyer fee + 8 % seller commission on the
    exit price (paid by the seller, reducing net proceeds).

    Args:
        p_exit: Model-3-implied exit price in dollars.

    Returns:
        Total transaction cost in dollars.
    """
    return 500.0 + 0.08 * p_exit


# ---------------------------------------------------------------------------
# Return decomposition
# ---------------------------------------------------------------------------

def compute_return_components(
    p_purchase: float,
    p_exit: float,
    cashflows: float,
    costs: float,
) -> Dict:
    """Decompose total return into income, capital-gain, and cost components.

    Formulas (per the paper):
        r_i = (p_exit + c_i - p_purchase - t_i) / p_purchase
        d_i = c_i / p_purchase                  (dividend / income yield)
        e_i = (p_exit - p_purchase) / p_purchase (capital gain)
        f_i = t_i / p_purchase                   (transaction-cost ratio)

    Args:
        p_purchase: Purchase price in dollars.
        p_exit:     Exit price in dollars.
        cashflows:  Total cash received during holding period.
        costs:      Total transaction costs in dollars.

    Returns:
        Dict with keys: 'r_i', 'd_i', 'e_i', 'f_i'.

    Raises:
        ValueError: If p_purchase <= 0.
    """
    if p_purchase <= 0:
        raise ValueError(f"p_purchase must be positive, got {p_purchase}")
    r_i = (p_exit + cashflows - p_purchase - costs) / p_purchase
    d_i = cashflows / p_purchase
    e_i = (p_exit - p_purchase) / p_purchase
    f_i = costs / p_purchase
    return {"r_i": r_i, "d_i": d_i, "e_i": e_i, "f_i": f_i}


# ---------------------------------------------------------------------------
# Internal helpers for quarterly revenue projection
# ---------------------------------------------------------------------------

def _get_asset_id(row: pd.Series) -> int:
    """Extract asset identifier, accepting either 'asset_id' or 'transaction_id'."""
    if "asset_id" in row.index:
        return int(row["asset_id"])
    return int(row["transaction_id"])


def _project_forward_revenues(
    hist_revs: np.ndarray,
    n_quarters: int,
    asset_seed: int,
) -> np.ndarray:
    """Project future quarterly revenues using exponential decay plus noise.

    Each forward quarter q (1-indexed) is:
        rev_q = last_historical * 0.97^q * (1 + ε_q),  ε_q ~ N(0, 0.05)

    Args:
        hist_revs:  Historical quarterly revenues (oldest → newest).
        n_quarters: Number of forward quarters to generate.
        asset_seed: Per-asset RNG seed for reproducibility.

    Returns:
        1-D numpy array of length n_quarters with projected revenues (>= 0).
    """
    rng = np.random.default_rng(asset_seed)
    last_rev = hist_revs[-1] if len(hist_revs) > 0 else 0.0
    qs = np.arange(1, n_quarters + 1, dtype=float)
    noise = rng.normal(0.0, 0.05, n_quarters)
    forward = last_rev * (0.97 ** qs) * (1.0 + noise)
    return np.maximum(forward, 0.0)


def _parse_hist_revs(panel: pd.DataFrame, asset_id: int) -> np.ndarray:
    """Return sorted historical quarterly revenues for a single asset."""
    rows = panel[panel["asset_id"] == asset_id].sort_values("quarter_date")
    return rows["revenue"].values


def _compute_ltm(revs: np.ndarray) -> float:
    """Sum of the last four revenue values (LTM proxy)."""
    return float(revs[-4:].sum()) if len(revs) >= 4 else float(revs.sum())


def _compute_lty(revs: np.ndarray) -> float:
    """Annualised 3-year revenue: sum of last 12 quarters / 3."""
    tail = revs[-12:] if len(revs) >= 12 else revs
    return float(tail.sum() / 3.0)


def _remaining_term(term: str, years_elapsed: int) -> Tuple[str, int]:
    """Compute remaining contract term after *years_elapsed* years.

    LOR contracts remain perpetual.  Fixed-term contracts subtract
    *years_elapsed* from the original N.

    Args:
        term:          Original contract term ('10Y', '30Y', 'LOR').
        years_elapsed: Holding period in integer years (1 or 5).

    Returns:
        Tuple (original_term_string, remaining_int_years).
        remaining_int_years is 0 for LOR (ignored in pricing).
    """
    if term == "LOR":
        return ("LOR", 0)
    n = int(term.replace("Y", ""))
    return (term, max(n - years_elapsed, 0))


# ---------------------------------------------------------------------------
# One-year backtest
# ---------------------------------------------------------------------------

def run_one_year_backtest(
    transactions_df: pd.DataFrame,
    quarterly_panel: pd.DataFrame,
    params: Optional[Dict] = None,
) -> pd.DataFrame:
    """Run one-year buy-and-hold backtest for each annual cohort 2017-2021.

    For each asset in each cohort year:
      1. Generate 4 forward quarterly revenues (decay + noise).
      2. Compute cashflows_1y = sum of 4 forward quarters.
      3. Compute exit-state (ltm, lty, age) one year after purchase.
      4. Price the asset at exit using Model 3 with remaining maturity.
      5. Decompose total return into income, capital-gain, and cost components.

    Assets without sufficient historical data (< 4 historical quarters) are
    silently excluded.

    Args:
        transactions_df: DataFrame with columns [asset_id or transaction_id,
                          transaction_date, ltm, lty, age, term, price].
        quarterly_panel: Long-format DataFrame with columns
                          [asset_id, quarter_date, revenue].
        params:          Model 3 parameters dict; defaults to DEFAULT_PARAMS.

    Returns:
        DataFrame with columns:
          asset_id, cohort_year, term, p_purchase, p_exit,
          cashflows_1y, costs, d_i, e_i, f_i, r_i
    """
    if params is None:
        params = DEFAULT_PARAMS

    records = []
    for _, txn in transactions_df.iterrows():
        asset_id = _get_asset_id(txn)
        hist = _parse_hist_revs(quarterly_panel, asset_id)

        if len(hist) < 4:
            continue  # insufficient history

        fwd = _project_forward_revenues(hist, _QUARTERS_1Y, asset_seed=asset_id * 31)
        cashflows_1y = float(fwd.sum())

        # Exit state: combine last 8 historical quarters + 4 forward quarters
        combined = np.concatenate([hist, fwd])
        ltm_exit = _compute_ltm(combined)
        lty_exit = _compute_lty(combined)
        age_exit = float(txn["age"]) + 1.0

        # Remaining contract term
        term = str(txn["term"])
        _, remaining = _remaining_term(term, years_elapsed=1)

        p_exit = price_model3(ltm_exit, lty_exit, age_exit, term, remaining, params)
        p_purchase = float(txn["price"])
        costs = compute_transaction_costs(p_exit)
        ret = compute_return_components(p_purchase, p_exit, cashflows_1y, costs)

        cohort_year = pd.Timestamp(txn["transaction_date"]).year
        records.append(
            {
                "asset_id": asset_id,
                "cohort_year": cohort_year,
                "term": term,
                "p_purchase": p_purchase,
                "p_exit": p_exit,
                "cashflows_1y": cashflows_1y,
                "costs": costs,
                **ret,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Five-year backtest
# ---------------------------------------------------------------------------

def run_five_year_backtest(
    transactions_df: pd.DataFrame,
    quarterly_panel: pd.DataFrame,
    params: Optional[Dict] = None,
) -> pd.DataFrame:
    """Run five-year buy-and-hold backtest for the 2017 cohort (exit in 2022).

    For each 2017 asset:
      1. Generate 20 forward quarterly revenues (decay + noise).
      2. Compute cashflows_5y = sum of all 20 forward quarters.
      3. Compute exit-state (ltm, lty, age) five years after purchase.
      4. Price at exit using Model 3 with remaining maturity.
      5. Compute annualised return: ann_r_i = (1 + r_i)^(1/5) - 1.
      6. Decompose total return into income, capital-gain, and cost components.

    Assets without sufficient historical data (< 4 historical quarters) are
    silently excluded.

    Args:
        transactions_df: DataFrame with columns [asset_id or transaction_id,
                          transaction_date, ltm, lty, age, term, price].
        quarterly_panel: Long-format DataFrame with columns
                          [asset_id, quarter_date, revenue].
        params:          Model 3 parameters dict; defaults to DEFAULT_PARAMS.

    Returns:
        DataFrame with columns:
          asset_id, term, p_purchase, p_exit, cashflows_5y,
          costs, d_i, e_i, f_i, r_i, ann_r_i
    """
    if params is None:
        params = DEFAULT_PARAMS

    # Filter to 2017 cohort only
    df_2017 = transactions_df[
        pd.to_datetime(transactions_df["transaction_date"]).dt.year == 2017
    ].copy()

    records = []
    for _, txn in df_2017.iterrows():
        asset_id = _get_asset_id(txn)
        hist = _parse_hist_revs(quarterly_panel, asset_id)

        if len(hist) < 4:
            continue

        fwd = _project_forward_revenues(hist, _QUARTERS_5Y, asset_seed=asset_id * 37)
        cashflows_5y = float(fwd.sum())

        # Exit state: last 8 historical quarters + last 12 of 20 forward quarters
        combined = np.concatenate([hist, fwd])
        ltm_exit = _compute_ltm(combined)
        lty_exit = _compute_lty(combined)
        age_exit = float(txn["age"]) + 5.0

        term = str(txn["term"])
        _, remaining = _remaining_term(term, years_elapsed=5)

        p_exit = price_model3(ltm_exit, lty_exit, age_exit, term, remaining, params)
        p_purchase = float(txn["price"])
        costs = compute_transaction_costs(p_exit)
        ret = compute_return_components(p_purchase, p_exit, cashflows_5y, costs)

        r_i = ret["r_i"]
        ann_r_i = (1.0 + r_i) ** (1.0 / 5.0) - 1.0

        records.append(
            {
                "asset_id": asset_id,
                "term": term,
                "p_purchase": p_purchase,
                "p_exit": p_exit,
                "cashflows_5y": cashflows_5y,
                "costs": costs,
                **ret,
                "ann_r_i": ann_r_i,
            }
        )

    return pd.DataFrame(records)
