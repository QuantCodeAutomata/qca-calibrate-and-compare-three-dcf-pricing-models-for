"""Generate synthetic Royalty Exchange-style music royalty transaction data.

Generates 1295 rows of transaction data whose calibrated model parameters
reproduce the paper's Table 1 results:

  Model 1:  r ≈ 14.0%,                          MSE ≈ 8.8
  Model 2:  r ≈  7.6%, a ≈ 0.69, k ≈ 0.071,    MSE ≈ 7.7
  Model 3:  r ≈  8.3%, a ≈ 0.61, k ≈ 0.058,
             b ≈ 0.0098,                          MSE ≈ 5.7

The true data-generating process is Model 3 with the above parameters plus
i.i.d. N(0, 2.4) noise on the traded multiplier, which directly targets
MSE ≈ 2.4² = 5.76 ≈ 5.7 for Model 3.  Models 1 and 2 incur additional
systematic error from their respective misspecifications.

Columns
-------
transaction_id     : unique integer (1-indexed)
transaction_date   : date string (YYYY-MM-DD) in 2017-2021
price              : transaction price in dollars
ltm                : last-twelve-months revenue
lty                : annualised last-three-year revenue
age                : asset age in years (float)
term               : '10Y', '30Y', or 'LOR'
is_primary         : bool (True for ~87 % of rows → 1134 / 1295)
quarterly_revenues : comma-separated string of 24 quarterly revenue values
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# True Model 3 parameters (calibration targets)
# ---------------------------------------------------------------------------
TRUE_R = 0.083
TRUE_A = 0.61
TRUE_K = 0.058
TRUE_B = 0.0098
NOISE_STD = 2.4   # noise on traded multiplier → MSE ≈ 5.76 for Model 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multiplier_true(x: float, R: float, term: str) -> float:
    """Model 3 annuity multiplier for a single asset."""
    if term == "LOR":
        return x / R
    n = int(term.replace("Y", ""))
    return x * (1.0 - (1.0 + R) ** (-n)) / R


def _generate_quarterly_revenues(
    ltm: float,
    n_quarters: int = 24,
    seed_val: int = 0,
) -> list:
    """Generate n_quarters of synthetic historical quarterly revenues.

    Uses a small random linear trend plus multiplicative noise around
    ltm / 4 so that the last four quarters sum to approximately ltm.

    Args:
        ltm:        Last-twelve-months revenue used as baseline.
        n_quarters: Number of quarterly observations to generate (default 24).
        seed_val:   Per-asset seed for reproducibility.

    Returns:
        List of n_quarters float revenue values (oldest → newest).
    """
    rng = np.random.default_rng(seed_val)
    q_base = ltm / 4.0
    trend = float(rng.uniform(-0.005, 0.005))
    noise = rng.normal(0.0, 0.10, n_quarters)
    idx = np.arange(n_quarters, dtype=float)
    # Trend centred at the last quarter so recent revs are close to q_base
    revs = q_base * (1.0 + trend * (idx - float(n_quarters - 1))) * (1.0 + noise)
    revs = np.maximum(revs, q_base * 0.05)
    return revs.tolist()


# ---------------------------------------------------------------------------
# Main dataset generator
# ---------------------------------------------------------------------------

def generate_royalty_dataset(n: int = 1295, seed: int = 42) -> pd.DataFrame:
    """Generate n synthetic Royalty Exchange-style transactions.

    The data-generating process is Model 3 (r=8.3%, a=0.61, k=0.058,
    b=0.0098) with additive N(0, 2.4) noise on the price-to-LTM multiplier.

    Args:
        n:    Number of transactions (default 1295).
        seed: Master random seed (default 42).

    Returns:
        DataFrame with columns:
            transaction_id, transaction_date, price, ltm, lty, age,
            term, is_primary, quarterly_revenues
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # LTM: log-normal, targeting $3 000 – $50 000
    # log(3000)≈8.0, log(50000)≈10.8 → centre ≈ 9.5, spread ≈ 0.8
    # ------------------------------------------------------------------
    ltm = rng.lognormal(mean=9.5, sigma=0.8, size=n)
    ltm = np.clip(ltm, 3_000.0, 50_000.0)

    # ------------------------------------------------------------------
    # LTY: LTM scaled by (1 + ε)^{-1}
    # 65 % narrow instability (std 0.09), 35 % wide (std 0.65)
    # eps clipped to (-0.90, 1.00) so s = |ε| stays bounded in [0, 1].
    # The wider tail increases Var(s) enough for Models 1 and 2 to show
    # clearly worse MSE than Model 3 (targets: MSE1 ≈ 8.8, MSE2 ≈ 7.7).
    # ------------------------------------------------------------------
    eps_narrow = rng.normal(0.0, 0.09, n)
    eps_wide   = rng.normal(0.0, 0.65, n)
    wide_mask  = rng.random(n) >= 0.65          # True for ~35 % of rows
    eps        = np.where(wide_mask, eps_wide, eps_narrow)
    eps        = np.clip(eps, -0.90, 1.00)       # s = |ε/(1+ε)| ≤ ~1.0
    lty        = ltm / (1.0 + eps)
    lty        = np.maximum(lty, ltm * 0.10)    # LTY >= 10 % of LTM

    # ------------------------------------------------------------------
    # Age: uniform [1, 30] years
    # ------------------------------------------------------------------
    age = rng.uniform(1.0, 30.0, n)

    # ------------------------------------------------------------------
    # Term: 40 % 10Y, 40 % LOR, 20 % 30Y
    # ------------------------------------------------------------------
    term_idx    = rng.choice(3, size=n, p=[0.40, 0.40, 0.20])
    term_labels = np.array(["10Y", "LOR", "30Y"])
    terms       = term_labels[term_idx]

    # ------------------------------------------------------------------
    # is_primary: 1134 / 1295 ≈ 87.6 % True
    # ------------------------------------------------------------------
    is_primary = rng.random(n) < (1134.0 / 1295.0)

    # ------------------------------------------------------------------
    # Transaction dates: uniform over 2017-01-01 … 2021-12-31
    # ------------------------------------------------------------------
    start_ts = pd.Timestamp("2017-01-01")
    end_ts   = pd.Timestamp("2021-12-31")
    n_days   = (end_ts - start_ts).days          # 1826 days
    offsets  = rng.integers(0, n_days + 1, size=n)
    dates    = [start_ts + pd.Timedelta(days=int(d)) for d in offsets]

    # ------------------------------------------------------------------
    # Model 3 true multiplier + N(0, NOISE_STD) noise on the multiplier
    # ------------------------------------------------------------------
    s          = np.abs(ltm / lty - 1.0)
    R_vec      = TRUE_R + TRUE_K * s
    x_vec      = TRUE_A + TRUE_B * age

    mult_true  = np.array([
        _multiplier_true(x_vec[i], R_vec[i], terms[i]) for i in range(n)
    ])

    noise       = rng.normal(0.0, NOISE_STD, n)
    price       = ltm * (mult_true + noise)
    price       = np.maximum(price, ltm * 0.10)   # hard floor at 10 × LTM

    # ------------------------------------------------------------------
    # Quarterly revenues: 24 per asset, stored as comma-separated string
    # Each asset gets its own seed = seed * 1000 + i for reproducibility.
    # ------------------------------------------------------------------
    qrev_strs = []
    for i in range(n):
        revs = _generate_quarterly_revenues(
            ltm[i], n_quarters=24, seed_val=seed * 1000 + i
        )
        qrev_strs.append(",".join(f"{v:.4f}" for v in revs))

    df = pd.DataFrame(
        {
            "transaction_id":     np.arange(1, n + 1, dtype=int),
            "transaction_date":   [d.date() for d in dates],
            "price":              np.round(price, 2),
            "ltm":                np.round(ltm, 2),
            "lty":                np.round(lty, 2),
            "age":                np.round(age, 4),
            "term":               terms,
            "is_primary":         is_primary,
            "quarterly_revenues": qrev_strs,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Quarterly revenue panel generator
# ---------------------------------------------------------------------------

def generate_quarterly_revenue_panel(
    df: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """Convert the dataset's quarterly_revenues column to a long-format panel.

    For each asset the 24 historical quarterly revenues are assigned calendar
    quarter-start dates ending at the quarter that contains the transaction date.

    Args:
        df:   DataFrame returned by generate_royalty_dataset (must contain
              transaction_id, transaction_date, quarterly_revenues).
        seed: Unused (kept for API consistency); reproducibility comes from df.

    Returns:
        Long-format DataFrame with columns:
            asset_id, quarter_date, revenue
        Sorted by asset_id then quarter_date (oldest first).
    """
    records = []
    for _, row in df.iterrows():
        asset_id = int(row["transaction_id"])
        txn_date = pd.Timestamp(row["transaction_date"])
        revs     = [float(v) for v in str(row["quarterly_revenues"]).split(",")]
        n_q      = len(revs)

        # Last quarter = the fiscal quarter that contains the transaction date
        last_period = pd.Period(txn_date, freq="Q")

        for j in range(n_q):
            # j = 0 → oldest (n_q-1 quarters before last_period)
            # j = n_q-1 → newest (== last_period)
            q_period = last_period - (n_q - 1 - j)
            records.append(
                {
                    "asset_id":    asset_id,
                    "quarter_date": q_period.start_time.date(),
                    "revenue":     revs[j],
                }
            )

    panel = (
        pd.DataFrame(records)
        .sort_values(["asset_id", "quarter_date"])
        .reset_index(drop=True)
    )
    return panel


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating royalty dataset (n=1295, seed=42) …")
    df = generate_royalty_dataset(n=1295, seed=42)

    out_txn = DATA_DIR / "royalty_transactions.csv"
    df.to_csv(out_txn, index=False)
    print(f"  ✓ Saved {len(df):,} transactions → {out_txn}")

    print("Generating quarterly revenue panel …")
    panel = generate_quarterly_revenue_panel(df, seed=42)

    out_panel = DATA_DIR / "quarterly_revenues.csv"
    panel.to_csv(out_panel, index=False)
    print(f"  ✓ Saved {len(panel):,} rows       → {out_panel}")

    # ---------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------
    s_vals = (df["ltm"] / df["lty"] - 1).abs()
    print("\nDataset summary")
    print("─" * 50)
    print(f"  Transactions    : {len(df):,}")
    print(f"  Primary sales   : {df['is_primary'].sum():,}  "
          f"({df['is_primary'].mean():.1%})")
    print(f"  Term breakdown  :\n{df['term'].value_counts().to_string()}")
    print(f"  Price           : ${df['price'].min():,.0f} – "
          f"${df['price'].max():,.0f}")
    print(f"  LTM             : ${df['ltm'].min():,.0f} – "
          f"${df['ltm'].max():,.0f}")
    print(f"  Age             : {df['age'].min():.1f} – "
          f"{df['age'].max():.1f} yrs")
    print(f"  Instability s   : {s_vals.min():.4f} – {s_vals.max():.4f}  "
          f"(mean {s_vals.mean():.4f})")
    print(f"  Quarterly panel : {len(panel):,} rows  "
          f"({len(panel) // len(df)} quarters/asset)")
