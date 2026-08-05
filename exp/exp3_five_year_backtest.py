#!/usr/bin/env python3
"""Exp 3: Five-Year Cross-Sectional Backtest of Music Royalty Assets (2017-2022).

Uses Model 3 pricing (paper params: r=8.3%, a=0.61, k=0.058, b=0.0098) to
price assets at purchase (2017) and exit (2022), collecting 20 quarters of
realized cashflows.

Paper targets:
  LOR : median total return ~82.6%, median annualised ~12.8%,
        p10 total ~-7.84%, p90 total ~279.7%
  10Y : median total return ~42.5%, median annualised ~7.3%,
        p10 total ~-59.9%, p90 total ~155.1%
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
from src.backtest_engine import run_five_year_backtest, DEFAULT_PARAMS

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
    """Load or generate royalty transaction and quarterly revenue data."""
    txn_path = os.path.join(DATA_DIR, 'royalty_transactions.csv')
    qrev_path = os.path.join(DATA_DIR, 'quarterly_revenues.csv')

    if not os.path.exists(txn_path) or not os.path.exists(qrev_path):
        log.info("Generating synthetic data...")
        sys.path.insert(0, DATA_DIR)
        from generate_royalty_data import generate_royalty_dataset, generate_quarterly_revenue_panel
        df = generate_royalty_dataset(n=1295, seed=42)
        df.to_csv(txn_path, index=False)
        qpanel = generate_quarterly_revenue_panel(df)
        qpanel.to_csv(qrev_path, index=False)
    else:
        df = pd.read_csv(txn_path)
        qpanel = pd.read_csv(qrev_path)

    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    qpanel['quarter_date'] = pd.to_datetime(qpanel['quarter_date'])
    log.info("Loaded %d transactions, %d quarterly rows.", len(df), len(qpanel))
    return df, qpanel


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_five_year(result: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional summary for each term group.

    Returns:
        DataFrame with median, p10, p90 for five-year returns by term.
    """
    rows = []
    for term, grp in result.groupby('term'):
        if term not in TERMS_OF_INTEREST:
            continue
        n = len(grp)
        row = {
            'term': term,
            'n_assets': n,
            'median_div': grp['d_i'].median() * 100,
            'median_capgain': grp['e_i'].median() * 100,
            'median_cost': grp['f_i'].median() * 100,
            'median_total_return': grp['r_i'].median() * 100,
            'median_ann_return': grp['ann_r_i'].median() * 100,
            'p10_total_return': grp['r_i'].quantile(0.10) * 100,
            'p90_total_return': grp['r_i'].quantile(0.90) * 100,
            'p10_ann_return': grp['ann_r_i'].quantile(0.10) * 100,
            'p90_ann_return': grp['ann_r_i'].quantile(0.90) * 100,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table3(summary: pd.DataFrame) -> None:
    """Print formatted Table 3 equivalent."""
    print("\n=== TABLE 3: FIVE-YEAR CROSS-SECTIONAL BACKTEST (2017 Cohort) ===")
    fmt = "{:<6} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8}"
    print(fmt.format('Term', 'Div%', 'CapGn%', 'Cost%', 'Med5Y%', 'MedAnn%', 'P10%', 'P90%'))
    print("-" * 72)
    for _, r in summary.sort_values('term').iterrows():
        print(fmt.format(
            r['term'],
            f"{r['median_div']:.1f}",
            f"{r['median_capgain']:.1f}",
            f"{r['median_cost']:.1f}",
            f"{r['median_total_return']:.1f}",
            f"{r['median_ann_return']:.1f}",
            f"{r['p10_total_return']:.1f}",
            f"{r['p90_total_return']:.1f}",
        ))

    print()
    print("Paper targets:")
    print("  LOR: median total ~82.6%, annualised ~12.8%, p10 ~-7.84%, p90 ~279.7%")
    print("  10Y: median total ~42.5%, annualised ~7.3%, p10 ~-59.9%, p90 ~155.1%")

    for term, targets in [
        ('LOR', {'total': 82.6, 'ann': 12.8, 'p10': -7.84, 'p90': 279.7}),
        ('10Y', {'total': 42.5, 'ann': 7.3,  'p10': -59.9,  'p90': 155.1}),
    ]:
        row = summary[summary['term'] == term]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        print(f"\n  {term} comparison:")
        print(f"    Median total: paper={targets['total']:.1f}%, "
              f"model={r['median_total_return']:.1f}%")
        print(f"    Median ann:   paper={targets['ann']:.1f}%, "
              f"model={r['median_ann_return']:.1f}%")
        print(f"    P10 total:    paper={targets['p10']:.2f}%, "
              f"model={r['p10_total_return']:.1f}%")
        print(f"    P90 total:    paper={targets['p90']:.1f}%, "
              f"model={r['p90_total_return']:.1f}%")


def print_assumptions() -> None:
    """Print documented assumptions for the five-year backtest."""
    print("\n=== ASSUMPTIONS: EXP 3 FIVE-YEAR BACKTEST ===")
    print("1. LOR perpetuity: p = LTM*(a+b*age)/(r+k*s)")
    print("2. Transaction costs: t = $500 + 8% * p_exit")
    print("3. 10Y exit remaining term = 10 - 5 = 5 years")
    print("4. 30Y assets omitted from headline tables")
    print("5. 2017 cohort: all transactions with transaction_date.year == 2017")
    print("6. Forward quarterly revenues: decay model, 20 quarters, per-asset seed")
    print("7. Model params: r=8.3%, a=0.61, k=0.058, b=0.0098 (paper Table 1)")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_return_distributions(result: pd.DataFrame) -> None:
    """Figure 6: Box plots of five-year total returns by term type."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, term in zip(axes, TERMS_OF_INTEREST):
        grp = result[result['term'] == term]['r_i'] * 100
        if len(grp) == 0:
            continue
        ax.boxplot(grp, orientation='vertical', patch_artist=True,
                   boxprops=dict(facecolor='lightsteelblue', color='navy'),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_title(f'{term} Five-Year Returns', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Return (%)', fontsize=11)
        ax.set_xlabel('', fontsize=11)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.grid(True, alpha=0.3, axis='y')
        med = grp.median()
        ax.text(1.12, med, f'Median\n{med:.1f}%', ha='left', va='center',
                fontsize=9, color='darkred')

    fig.suptitle('Figure 6: Five-Year Return Distribution (2017 Cohort)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'figure6_five_year_return_dist.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved figure: %s", path)


def plot_return_components_stacked(summary: pd.DataFrame) -> None:
    """Figure 7: Stacked bar showing return component decomposition."""
    terms = [t for t in TERMS_OF_INTEREST if t in summary['term'].values]
    divs = [summary[summary['term'] == t]['median_div'].iloc[0] for t in terms]
    capgains = [summary[summary['term'] == t]['median_capgain'].iloc[0] for t in terms]
    costs = [-summary[summary['term'] == t]['median_cost'].iloc[0] for t in terms]

    x = np.arange(len(terms))
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(x, divs, label='Dividend Yield', color='steelblue', alpha=0.85)
    bottom_pos = np.array(divs)
    pos_caps = np.where(np.array(capgains) >= 0, capgains, 0)
    neg_caps = np.where(np.array(capgains) < 0, capgains, 0)
    ax.bar(x, pos_caps, bottom=bottom_pos, label='Capital Gain (+)', color='seagreen', alpha=0.85)
    ax.bar(x, neg_caps, bottom=bottom_pos, label='Capital Gain (–)', color='tomato', alpha=0.85)
    ax.bar(x, costs, label='Transaction Cost (–)', color='dimgray', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(terms, fontsize=12)
    ax.set_ylabel('Median 5-Year Return Component (%)', fontsize=11)
    ax.set_title('Figure 7: Five-Year Return Decomposition by Term Type',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'figure7_five_year_decomposition.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved figure: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    transactions_df, quarterly_panel = load_data()

    log.info("Running five-year backtest for 2017 cohort...")
    result = run_five_year_backtest(transactions_df, quarterly_panel, params=DEFAULT_PARAMS)
    log.info("Backtest complete: %d asset observations.", len(result))

    if len(result) == 0:
        log.warning("No 2017 assets found — check transaction_date values in dataset.")
        return

    # Save asset-level results
    asset_path = os.path.join(RESULTS_DIR, 'exp3_asset_level.csv')
    result.to_csv(asset_path, index=False)
    log.info("Saved asset-level results to %s", asset_path)

    # Summary
    summary = summarize_five_year(result)
    print_table3(summary)
    print_assumptions()

    # Save summary table
    table_path = os.path.join(RESULTS_DIR, 'table3_five_year_backtest.csv')
    summary.to_csv(table_path, index=False)
    log.info("Saved Table 3 to %s", table_path)

    # Plots
    plot_return_distributions(result)
    plot_return_components_stacked(summary)

    log.info("Exp 3 complete. Outputs saved to %s", RESULTS_DIR)


if __name__ == '__main__':
    main()
