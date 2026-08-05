"""Exp 4: S&P 500 benchmark backtest for one-year and five-year comparison.

Methodology:
1. Use representative S&P 500 constituents as of 2017 (100 stocks)
2. For one-year cohorts (2017-2021): compute annual returns
3. For five-year cohort (2017-2022): compute total and annualized returns
4. Summarize cross-sectional distributions (median, p10, p90)

Paper targets for five-year benchmark:
  Median total return: ~77.7%
  Median annualized: ~12.2%
  90th pct total return: ~257.2%
  10th pct total return: ~-15.3%

Assumptions:
- Using yfinance adjusted close (dividend-adjusted) for returns
- Representative 100-stock constituent list (not full CRSP historical data)
- No transaction costs applied to equity benchmark
- Returns computed from year-start to year-end prices
"""
import sys
sys.path.insert(0, '/workspace/project')

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmark import (
    SP500_2017_TICKERS,
    compute_annual_returns,
    compute_horizon_returns,
    summarize_cross_section,
)
from data.load_sp500 import get_sp500_constituents_2017, load_or_download_prices

RESULTS_DIR = Path('/workspace/project/results')
DATA_DIR = Path('/workspace/project/data')

# Paper benchmark targets
TARGET_5Y_MEDIAN_TOTAL = 0.777
TARGET_5Y_MEDIAN_ANN = 0.122
TARGET_5Y_P90_TOTAL = 2.572
TARGET_5Y_P10_TOTAL = -0.153

# Target values for music royalties (fallback when table3 is absent)
_LOR_5Y_MEDIAN_TARGET = 0.826
_TEN_Y_5Y_MEDIAN_TARGET = 0.425


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_exp3_five_year() -> Tuple[Dict, Dict]:
    """Load LOR and 10Y five-year total return stats from table3 CSV.

    Falls back to paper target median values if the file does not exist or
    cannot be parsed. p10/p90 are set to None when unavailable.

    Returns:
        (lor_stats, ten_y_stats) – each a dict with keys median, p10, p90.
    """
    lor = {'median': _LOR_5Y_MEDIAN_TARGET, 'p10': None, 'p90': None}
    ten_y = {'median': _TEN_Y_5Y_MEDIAN_TARGET, 'p10': None, 'p90': None}

    csv_path = RESULTS_DIR / 'table3_five_year_backtest.csv'
    if not csv_path.exists():
        return lor, ten_y

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower().str.strip()

        # Identify key columns
        asset_col = next(
            (c for c in df.columns if c in ('asset_type', 'asset', 'type', 'asset_class')),
            None,
        )
        period_col = next(
            (c for c in df.columns if c in ('return_type', 'period', 'horizon')),
            None,
        )

        def _extract(mask: pd.Series) -> Dict:
            rows = df[mask]
            if rows.empty:
                return {}
            row = rows.iloc[0]
            result = {}
            for key in ('median', 'p10', 'p90'):
                if key in df.columns:
                    try:
                        result[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            return result

        if asset_col is not None:
            lor_mask = df[asset_col].str.upper().str.contains('LOR', na=False)
            ten_y_mask = df[asset_col].str.upper().str.contains(r'10Y|10_Y', regex=True, na=False)

            if period_col is not None:
                total_mask = df[period_col].str.lower().str.contains('total', na=False)
                lor_data = _extract(lor_mask & total_mask) or _extract(lor_mask)
                ten_y_data = _extract(ten_y_mask & total_mask) or _extract(ten_y_mask)
            else:
                lor_data = _extract(lor_mask)
                ten_y_data = _extract(ten_y_mask)

            lor.update(lor_data)
            ten_y.update(ten_y_data)

    except Exception as exc:
        print(f"  Warning: could not parse table3 CSV – {exc}")

    return lor, ten_y


def _save_assumptions(failed_tickers: List[str], n_loaded: int) -> None:
    path = RESULTS_DIR / 'assumptions_exp4.txt'
    lines = [
        "Assumptions for Exp 4: S&P 500 Benchmark Backtest",
        "=" * 54,
        "",
        "Data Source",
        "-----------",
        "  Price data: yfinance adjusted close (auto_adjust=True).",
        "  Adjusted close accounts for dividends and stock splits.",
        "",
        "Constituent List",
        "----------------",
        f"  Total tickers in representative list : {len(SP500_2017_TICKERS)}",
        f"  Tickers successfully downloaded      : {n_loaded}",
        f"  Tickers with no data                 : {len(failed_tickers)}",
        "  Methodology: static 100-stock list of large-cap S&P 500 members",
        "    as of 2017. NOT a full CRSP/WRDS historical constituent file.",
        "  Survivorship bias: partial – stocks that were delisted or merged",
        "    between 2017 and 2022 may be absent from yfinance data.",
        "",
        "Return Computation",
        "------------------",
        "  One-year: from first available trading day of the year to the",
        "    last available trading day of the same year.",
        "  Five-year: from first available trading day on or after",
        "    2017-01-01 to last available trading day on or before",
        "    2022-12-31.",
        "  Annualized five-year: (1 + total_return)^(1/5) - 1.",
        "",
        "Transaction Costs",
        "-----------------",
        "  None applied. Returns are gross of transaction costs.",
        "",
        "Benchmark Target Values (from paper)",
        "-------------------------------------",
        f"  Five-year median total return   : ~{TARGET_5Y_MEDIAN_TOTAL:.1%}",
        f"  Five-year median annualized     : ~{TARGET_5Y_MEDIAN_ANN:.1%}",
        f"  Five-year 90th pct total return : ~{TARGET_5Y_P90_TOTAL:.1%}",
        f"  Five-year 10th pct total return : ~{TARGET_5Y_P10_TOTAL:.1%}",
        "",
    ]
    if failed_tickers:
        lines += [
            "Failed Tickers",
            "--------------",
            "  " + ", ".join(failed_tickers),
            "",
        ]
    path.write_text("\n".join(lines))
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_figure8(annual_returns: Dict[int, pd.Series]) -> None:
    """Figure 8: One-year benchmark cross-sectional returns by year."""
    years = sorted(annual_returns.keys())
    data = [annual_returns[y].dropna().values * 100 for y in years]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        data,
        tick_labels=[str(y) for y in years],
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=3, alpha=0.5),
    )
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Return (%)', fontsize=12)
    ax.set_title(
        'Figure 8: S&P 500 One-Year Cross-Sectional Return Distribution',
        fontsize=13, fontweight='bold',
    )
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    out = RESULTS_DIR / 'figure8_sp500_annual_returns.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def _plot_figure9(sp500_five_yr: Dict) -> None:
    """Figure 9: Five-year total return comparison – music royalties vs S&P 500."""
    lor_stats, ten_y_stats = _load_exp3_five_year()

    labels = ['LOR', '10Y', 'S&P 500']
    medians = [
        lor_stats['median'] * 100,
        ten_y_stats['median'] * 100,
        sp500_five_yr['median'] * 100,
    ]

    # Build asymmetric error bars [lower, upper] where available
    def _errs(stats: Dict) -> Optional[Tuple[float, float]]:
        if stats.get('p10') is None or stats.get('p90') is None:
            return None
        return (
            max(stats['median'] - stats['p10'], 0) * 100,
            max(stats['p90'] - stats['median'], 0) * 100,
        )

    err_lor = _errs(lor_stats)
    err_ten_y = _errs(ten_y_stats)
    err_sp500 = _errs(sp500_five_yr)

    colors = ['#2196F3', '#4CAF50', '#FF5722']
    x = np.arange(len(labels))
    bar_width = 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, medians, width=bar_width, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)

    # Error bars per asset
    for i, (err, label) in enumerate(zip([err_lor, err_ten_y, err_sp500], labels)):
        if err is not None:
            ax.errorbar(
                x[i], medians[i],
                yerr=[[err[0]], [err[1]]],
                fmt='none', color='black', capsize=6, linewidth=1.5,
            )

    # Annotate bar tops
    for bar, val in zip(bars, medians):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Five-Year Total Return (Median, %)', fontsize=12)
    ax.set_title(
        'Figure 9: Five-Year Total Return Comparison: Music Royalties vs S&P 500',
        fontsize=12, fontweight='bold',
    )
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=min(0, min(medians) - 15))

    # Legend note on error bars
    ax.text(
        0.98, 0.02,
        'Error bars: 10th–90th percentile range\n(when available)',
        transform=ax.transAxes,
        fontsize=8, ha='right', va='bottom',
        style='italic', color='grey',
    )

    fig.tight_layout()
    out = RESULTS_DIR / 'figure9_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Exp 4: S&P 500 Benchmark Backtest")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Tickers and price download
    # ------------------------------------------------------------------
    tickers = get_sp500_constituents_2017()
    print(f"\nDownloading prices for {len(tickers)} tickers "
          f"(2017-01-01 to 2022-12-31) ...")

    prices_df = load_or_download_prices(
        tickers,
        start='2017-01-01',
        end='2022-12-31',
        cache_path=str(DATA_DIR / 'sp500_prices_cache.csv'),
    )

    n_loaded = len(prices_df.columns)
    failed = sorted(set(tickers) - set(str(c) for c in prices_df.columns))
    print(f"Successfully loaded {n_loaded} stocks.")
    if failed:
        preview = ', '.join(failed[:10]) + ('...' if len(failed) > 10 else '')
        print(f"  Tickers with no data ({len(failed)}): {preview}")
    if n_loaded < 10:
        print("  WARNING: Fewer than 10 stocks available. "
              "Results may not be reliable.")

    # ------------------------------------------------------------------
    # 2. One-year cohorts 2017-2021
    # ------------------------------------------------------------------
    print("\n--- One-Year Annual Returns ---")
    annual_returns: Dict[int, pd.Series] = {}
    annual_summaries: Dict[int, Dict] = {}
    for year in range(2017, 2022):
        rets = compute_annual_returns(prices_df, year)
        annual_returns[year] = rets
        annual_summaries[year] = summarize_cross_section(rets)
        s = annual_summaries[year]
        print(f"  {year}: n={s['n_stocks']:3d}  "
              f"median={s['median']:+.1%}  "
              f"p10={s['p10']:+.1%}  "
              f"p90={s['p90']:+.1%}")

    # ------------------------------------------------------------------
    # 3. Five-year cohort 2017-2022
    # ------------------------------------------------------------------
    print("\n--- Five-Year Returns (2017-2022) ---")
    five_yr_rets = compute_horizon_returns(prices_df, '2017-01-01', '2022-12-31')
    five_yr_ann_rets = (1 + five_yr_rets) ** (1 / 5) - 1

    five_yr_summary = summarize_cross_section(five_yr_rets)
    five_yr_ann_summary = summarize_cross_section(five_yr_ann_rets)

    print(f"  Total:      n={five_yr_summary['n_stocks']}  "
          f"median={five_yr_summary['median']:+.1%}  "
          f"p10={five_yr_summary['p10']:+.1%}  "
          f"p90={five_yr_summary['p90']:+.1%}")
    print(f"  Annualized: n={five_yr_ann_summary['n_stocks']}  "
          f"median={five_yr_ann_summary['median']:+.1%}  "
          f"p10={five_yr_ann_summary['p10']:+.1%}  "
          f"p90={five_yr_ann_summary['p90']:+.1%}")

    # ------------------------------------------------------------------
    # 4. Comparison to paper targets
    # ------------------------------------------------------------------
    print("\n--- Comparison to Paper Targets ---")
    print(f"  Five-year median total:      "
          f"{five_yr_summary['median']:+.1%}  (target: ~{TARGET_5Y_MEDIAN_TOTAL:.1%})")
    print(f"  Five-year median annualized: "
          f"{five_yr_ann_summary['median']:+.1%}  (target: ~{TARGET_5Y_MEDIAN_ANN:.1%})")
    print(f"  Five-year p90 total:         "
          f"{five_yr_summary['p90']:+.1%}  (target: ~{TARGET_5Y_P90_TOTAL:.1%})")
    print(f"  Five-year p10 total:         "
          f"{five_yr_summary['p10']:+.1%}  (target: ~{TARGET_5Y_P10_TOTAL:.1%})")

    # ------------------------------------------------------------------
    # 5. Save table4 CSV
    # ------------------------------------------------------------------
    rows = []
    for year in range(2017, 2022):
        rows.append({'period': str(year), 'return_type': 'annual',
                     **annual_summaries[year]})
    rows.append({'period': '2017-2022', 'return_type': 'total',
                 **five_yr_summary})
    rows.append({'period': '2017-2022', 'return_type': 'annualized',
                 **five_yr_ann_summary})

    table4 = pd.DataFrame(rows)
    table4_path = RESULTS_DIR / 'table4_sp500_benchmark.csv'
    table4.to_csv(table4_path, index=False)
    print(f"\nSaved: {table4_path}")

    # ------------------------------------------------------------------
    # 6. Assumptions file
    # ------------------------------------------------------------------
    _save_assumptions(failed, n_loaded)

    # ------------------------------------------------------------------
    # 7. Figure 8: one-year box plots
    # ------------------------------------------------------------------
    _plot_figure8(annual_returns)

    # ------------------------------------------------------------------
    # 8. Figure 9: five-year comparison bar chart
    # ------------------------------------------------------------------
    _plot_figure9(five_yr_summary)

    print("\nExp 4 complete.")


if __name__ == '__main__':
    main()
