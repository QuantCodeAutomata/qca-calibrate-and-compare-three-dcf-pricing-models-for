#!/usr/bin/env python3
"""Exp 2: One-Year Cross-Sectional Backtest of Music Royalty Assets.

Uses Model 3 pricing (paper params: r=8.3%, a=0.61, k=0.058, b=0.0098) to
price assets at purchase and exit, collects 4 quarters of realized cashflows,
and decomposes total return into dividend, capital-gain, and cost components.

Results are reported separately for LOR and 10Y cohorts for years 2017-2021.

Paper targets for LOR median annual returns:
  2017: ~12.5%, 2018: ~7.1%, 2019: ~9.0%, 2020: ~10.6%, 2021: ~12.9%
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/project')
from src.backtest_engine import run_one_year_backtest, DEFAULT_PARAMS

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

DATA_DIR = '/workspace/project/data'
RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

TERMS_OF_INTEREST = ['LOR', '10Y']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load royalty transaction data and quarterly revenue panel.

    Returns:
        Tuple of (transactions_df, quarterly_panel).
    """
    txn_path = os.path.join(DATA_DIR, 'royalty_transactions.csv')
    qrev_path = os.path.join(DATA_DIR, 'quarterly_revenues.csv')

    if not os.path.exists(txn_path) or not os.path.exists(qrev_path):
        log.info("Data files not found; generating synthetic data...")
        sys.path.insert(0, DATA_DIR)
        from generate_royalty_data import generate_royalty_dataset, generate_quarterly_revenue_panel
        df = generate_royalty_dataset(n=1295, seed=42)
        df.to_csv(txn_path, index=False)
        qpanel = generate_quarterly_revenue_panel(df)
        qpanel.to_csv(qrev_path, index=False)
        log.info("Generated and saved synthetic data.")
    else:
        df = pd.read_csv(txn_path)
        qpanel = pd.read_csv(qrev_path)

    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    qpanel['quarter_date'] = pd.to_datetime(qpanel['quarter_date'])
    log.info("Loaded %d transactions and %d quarterly revenue rows.", len(df), len(qpanel))
    return df, qpanel


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_cohort(result: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional summary for each (cohort_year, term) group.

    Returns:
        DataFrame with median, p10, p90 of d_i, e_i, f_i, r_i per group.
    """
    rows = []
    for (year, term), grp in result.groupby(['cohort_year', 'term']):
        if term not in TERMS_OF_INTEREST:
            continue
        n = len(grp)
        row = {
            'cohort_year': year,
            'term': term,
            'n_assets': n,
            'median_div': grp['d_i'].median() * 100,
            'median_capgain': grp['e_i'].median() * 100,
            'median_cost': grp['f_i'].median() * 100,
            'median_total_return': grp['r_i'].median() * 100,
            'p10_total_return': grp['r_i'].quantile(0.10) * 100,
            'p90_total_return': grp['r_i'].quantile(0.90) * 100,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table2(summary: pd.DataFrame) -> None:
    """Print formatted Table 2 equivalent."""
    print("\n=== TABLE 2: ONE-YEAR CROSS-SECTIONAL BACKTEST RESULTS ===")
    print(f"{'Year':<6} {'Term':<6} {'N':>5} {'Div%':>8} {'CapGn%':>8} "
          f"{'Cost%':>8} {'Med Ret%':>10} {'P10%':>8} {'P90%':>8}")
    print("-" * 70)
    for _, r in summary.sort_values(['cohort_year', 'term']).iterrows():
        print(f"{int(r['cohort_year']):<6} {r['term']:<6} {int(r['n_assets']):>5} "
              f"{r['median_div']:>8.1f} {r['median_capgain']:>8.1f} "
              f"{r['median_cost']:>8.1f} {r['median_total_return']:>10.1f} "
              f"{r['p10_total_return']:>8.1f} {r['p90_total_return']:>8.1f}")
    print()
    print("Paper targets for LOR median returns:")
    paper_lor = {2017: 12.5, 2018: 7.1, 2019: 9.0, 2020: 10.6, 2021: 12.9}
    for yr, target in paper_lor.items():
        row = summary[(summary['cohort_year'] == yr) & (summary['term'] == 'LOR')]
        if len(row) > 0:
            actual = row['median_total_return'].iloc[0]
            print(f"  {yr} LOR: paper={target:.1f}%, model={actual:.1f}%")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_one_year_returns(summary: pd.DataFrame) -> None:
    """Figure 5: Bar chart of median one-year returns by cohort year and term."""
    years = sorted(summary['cohort_year'].unique())
    x = np.arange(len(years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    lor_medians, lor_p10, lor_p90 = [], [], []
    y10_medians, y10_p10, y10_p90 = [], [], []

    for yr in years:
        lor_row = summary[(summary['cohort_year'] == yr) & (summary['term'] == 'LOR')]
        y10_row = summary[(summary['cohort_year'] == yr) & (summary['term'] == '10Y')]
        lor_medians.append(lor_row['median_total_return'].iloc[0] if len(lor_row) > 0 else 0)
        lor_p10.append(lor_row['p10_total_return'].iloc[0] if len(lor_row) > 0 else 0)
        lor_p90.append(lor_row['p90_total_return'].iloc[0] if len(lor_row) > 0 else 0)
        y10_medians.append(y10_row['median_total_return'].iloc[0] if len(y10_row) > 0 else 0)
        y10_p10.append(y10_row['p10_total_return'].iloc[0] if len(y10_row) > 0 else 0)
        y10_p90.append(y10_row['p90_total_return'].iloc[0] if len(y10_row) > 0 else 0)

    lor_medians = np.array(lor_medians)
    y10_medians = np.array(y10_medians)
    lor_err_low = lor_medians - np.array(lor_p10)
    lor_err_high = np.array(lor_p90) - lor_medians
    y10_err_low = y10_medians - np.array(y10_p10)
    y10_err_high = np.array(y10_p90) - y10_medians

    bars1 = ax.bar(x - width / 2, lor_medians, width, label='LOR',
                   color='steelblue', alpha=0.8)
    ax.errorbar(x - width / 2, lor_medians,
                yerr=[lor_err_low, lor_err_high],
                fmt='none', color='navy', capsize=4, linewidth=1.5)

    bars2 = ax.bar(x + width / 2, y10_medians, width, label='10Y',
                   color='darkorange', alpha=0.8)
    ax.errorbar(x + width / 2, y10_medians,
                yerr=[y10_err_low, y10_err_high],
                fmt='none', color='saddlebrown', capsize=4, linewidth=1.5)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Cohort Year', fontsize=12)
    ax.set_ylabel('Median Annual Return (%)', fontsize=12)
    ax.set_title('Figure 5: One-Year Median Returns by Cohort Year\n'
                 '(error bars: p10/p90 range)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'figure5_one_year_returns.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved figure: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    transactions_df, quarterly_panel = load_data()

    log.info("Running one-year backtest for cohorts 2017-2021...")
    result = run_one_year_backtest(transactions_df, quarterly_panel, params=DEFAULT_PARAMS)
    log.info("Backtest complete: %d asset-cohort observations.", len(result))

    # Save asset-level results
    asset_path = os.path.join(RESULTS_DIR, 'exp2_asset_level.csv')
    result.to_csv(asset_path, index=False)
    log.info("Saved asset-level results to %s", asset_path)

    # Compute and print summary
    summary = summarize_cohort(result)
    print_table2(summary)

    # Save summary table
    table_path = os.path.join(RESULTS_DIR, 'table2_one_year_backtest.csv')
    summary.to_csv(table_path, index=False)
    log.info("Saved Table 2 to %s", table_path)

    # Plot
    plot_one_year_returns(summary)

    log.info("Exp 2 complete. Outputs saved to %s", RESULTS_DIR)


if __name__ == '__main__':
    main()
