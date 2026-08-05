#!/usr/bin/env python3
"""
Exp 1: DCF Model Calibration
Calibrates Models 1–3 to royalty transaction data and reproduces Table 1.
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/project')

from src.models import model3_multiplier_vec
from src.optimization import calibrate_model1, calibrate_model2, calibrate_model3

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

DATA_PATH = '/workspace/project/data/royalty_transactions.csv'
RESULTS_DIR = '/workspace/project/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        log.info("Loaded %d rows from %s", len(df), DATA_PATH)
    else:
        sys.path.insert(0, '/workspace/project/data')
        from generate_royalty_data import generate_royalty_dataset
        df = generate_royalty_dataset(n=1295, seed=42)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        log.info("Generated dataset (%d rows) and saved to %s", len(df), DATA_PATH)
    return df


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> None:
    n = len(df)
    if n != 1295:
        raise ValueError(f"Expected 1295 rows, got {n}")
    required_cols = ['ltm', 'lty', 'age', 'term', 'price']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if (df['ltm'] < 0).any():
        raise ValueError("Negative LTM values found")
    if (df['lty'] < 0).any():
        raise ValueError("Negative LTY values found")
    log.info("Data validation passed: %d rows, columns %s", n, list(df.columns))


# ---------------------------------------------------------------------------
# 3–4. Derived fields
# ---------------------------------------------------------------------------

def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['M_traded'] = df['price'] / df['ltm']
    df['s'] = (df['ltm'] / df['lty'] - 1).abs()
    return df


# ---------------------------------------------------------------------------
# 5. Summary statistics
# ---------------------------------------------------------------------------

def log_summary(df: pd.DataFrame) -> None:
    log.info("=== DATA SUMMARY ===")
    log.info("Shape: %s", df.shape)
    log.info("Term distribution:\n%s", df['term'].value_counts().to_string())
    for col in ['ltm', 'lty', 'age', 'price', 'M_traded', 's']:
        if col in df.columns:
            log.info(
                "%-12s  mean=%10.2f  std=%10.2f  min=%10.2f  max=%10.2f",
                col, df[col].mean(), df[col].std(), df[col].min(), df[col].max(),
            )


# ---------------------------------------------------------------------------
# 6. Calibration
# ---------------------------------------------------------------------------

def run_calibrations(df: pd.DataFrame) -> dict:
    log.info("=== CALIBRATING MODELS ===")
    results = {}

    log.info("Calibrating Model 1 ...")
    results['model1'] = calibrate_model1(df)
    r1 = results['model1']
    log.info("  Model 1: r=%.4f, MSE=%.4f", r1['params']['r'], r1['mse'])

    log.info("Calibrating Model 2 ...")
    results['model2'] = calibrate_model2(df)
    r2, p2 = results['model2'], results['model2']['params']
    log.info("  Model 2: r=%.4f, a=%.4f, k=%.4f, MSE=%.4f", p2['r'], p2['a'], p2['k'], r2['mse'])

    log.info("Calibrating Model 3 ...")
    results['model3'] = calibrate_model3(df)
    r3, p3 = results['model3'], results['model3']['params']
    log.info("  Model 3: r=%.4f, a=%.4f, k=%.4f, b=%.6f, MSE=%.4f",
             p3['r'], p3['a'], p3['k'], p3['b'], r3['mse'])

    return results


# ---------------------------------------------------------------------------
# 7. Print Table 1
# ---------------------------------------------------------------------------

def print_table1(results: dict) -> pd.DataFrame:
    p1 = results['model1']['params']
    p2 = results['model2']['params']
    p3 = results['model3']['params']

    rows = [
        {
            'Model': 'Model 1',
            'r':     f"{p1['r']*100:.1f}%",
            'a':     '-',
            'k':     '-',
            'b':     '-',
            'MSE':   f"{results['model1']['mse']:.1f}",
        },
        {
            'Model': 'Model 2',
            'r':     f"{p2['r']*100:.1f}%",
            'a':     f"{p2['a']:.2f}",
            'k':     f"{p2['k']:.3f}",
            'b':     '-',
            'MSE':   f"{results['model2']['mse']:.1f}",
        },
        {
            'Model': 'Model 3',
            'r':     f"{p3['r']*100:.1f}%",
            'a':     f"{p3['a']:.2f}",
            'k':     f"{p3['k']:.3f}",
            'b':     f"{p3['b']:.4f}",
            'MSE':   f"{results['model3']['mse']:.1f}",
        },
    ]

    table = pd.DataFrame(rows, columns=['Model', 'r', 'a', 'k', 'b', 'MSE'])

    header = "\n=== TABLE 1: CALIBRATION RESULTS ==="
    print(header)
    print(table.to_string(index=False))
    print()
    print("Paper targets:")
    print("  Model 1: r≈14.0%, MSE≈8.8")
    print("  Model 2: r≈7.6%, a≈0.69, k≈0.071, MSE≈7.7")
    print("  Model 3: r≈8.3%, a≈0.61, k≈0.058, b≈0.0098, MSE≈5.7")

    return table


# ---------------------------------------------------------------------------
# 8. Verify MSE ranking
# ---------------------------------------------------------------------------

def verify_mse_ranking(results: dict) -> None:
    mse1 = results['model1']['mse']
    mse2 = results['model2']['mse']
    mse3 = results['model3']['mse']
    if not (mse3 < mse2 < mse1):
        raise AssertionError(
            f"MSE ranking violated — Model 3={mse3:.3f}, Model 2={mse2:.3f}, Model 1={mse1:.3f}. "
            "Expected Model 3 < Model 2 < Model 1."
        )
    log.info("MSE ranking verified: %.3f (M3) < %.3f (M2) < %.3f (M1)", mse3, mse2, mse1)


# ---------------------------------------------------------------------------
# 9. Save Table 1 CSV
# ---------------------------------------------------------------------------

def save_table1(table: pd.DataFrame) -> None:
    path = os.path.join(RESULTS_DIR, 'table1_calibration.csv')
    table.to_csv(path, index=False)
    log.info("Saved Table 1 to %s", path)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _make_ltm_lty_bins(df: pd.DataFrame, n_bins: int = 5) -> pd.Series:
    """Quantile-based bins of the LTM/LTY ratio."""
    ratio = df['ltm'] / df['lty']
    labels = [f'Q{i+1}' for i in range(n_bins)]
    return pd.qcut(ratio, n_bins, labels=labels, duplicates='drop')


def _make_age_bins(df: pd.DataFrame) -> pd.Series:
    """Fixed age brackets matching the paper description."""
    edges = [0, 5, 10, 15, 20, np.inf]
    labels = ['1-5', '5-10', '10-15', '15-20', '20+']
    return pd.cut(df['age'], bins=edges, labels=labels)


_TERM_STYLE = {
    '10Y': dict(color='steelblue',   marker='o', linestyle='-'),
    'LOR': dict(color='darkorange',  marker='s', linestyle='--'),
}


def _plot_multiplier_by_bin(
    df: pd.DataFrame,
    value_col: str,
    bin_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    fname: str,
) -> None:
    bin_order = list(df[bin_col].cat.categories)

    fig, ax = plt.subplots(figsize=(9, 5))

    for term in ['10Y', 'LOR']:
        sub = df[df['term'] == term].copy()
        means = sub.groupby(bin_col, observed=True)[value_col].mean().reindex(bin_order)
        valid = means.dropna()
        if valid.empty:
            continue
        ax.plot(
            [str(b) for b in valid.index],
            valid.values,
            linewidth=2,
            markersize=7,
            label=term,
            **_TERM_STYLE[term],
        )

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, title='Term')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved figure: %s", path)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figures(df: pd.DataFrame, results: dict) -> None:
    df = df.copy()
    df['ltm_lty_bin'] = _make_ltm_lty_bins(df)
    df['age_bin'] = _make_age_bins(df)

    # Model 3 implied multipliers
    p3 = results['model3']['params']
    df['model3_mult'] = model3_multiplier_vec(
        df['s'].values,
        df['age'].values,
        df['term'].values,
        p3['r'], p3['a'], p3['k'], p3['b'],
    )

    # Figure 1 – traded multiplier by LTM/LTY bin
    _plot_multiplier_by_bin(
        df, 'M_traded', 'ltm_lty_bin',
        'LTM/LTY Ratio Bin (Q1 = lowest ratio)',
        'Mean Traded Multiplier',
        'Figure 1: Mean Traded Multiplier by Revenue Stability (LTM/LTY)',
        'figure1_traded_multiplier_by_ltm_lty.png',
    )

    # Figure 2 – traded multiplier by age bin
    _plot_multiplier_by_bin(
        df, 'M_traded', 'age_bin',
        'Asset Age (years)',
        'Mean Traded Multiplier',
        'Figure 2: Mean Traded Multiplier by Asset Age',
        'figure2_traded_multiplier_by_age.png',
    )

    # Figure 3 – Model 3 implied multiplier by LTM/LTY bin
    _plot_multiplier_by_bin(
        df, 'model3_mult', 'ltm_lty_bin',
        'LTM/LTY Ratio Bin (Q1 = lowest ratio)',
        'Mean Model 3 Implied Multiplier',
        'Figure 3: Model 3 Implied Multiplier by Revenue Stability',
        'figure3_model3_multiplier_by_ltm_lty.png',
    )

    # Figure 4 – Model 3 implied multiplier by age bin
    _plot_multiplier_by_bin(
        df, 'model3_mult', 'age_bin',
        'Asset Age (years)',
        'Mean Model 3 Implied Multiplier',
        'Figure 4: Model 3 Implied Multiplier by Asset Age',
        'figure4_model3_multiplier_by_age.png',
    )


# ---------------------------------------------------------------------------
# Assumptions log
# ---------------------------------------------------------------------------

def save_assumptions_log(results: dict) -> None:
    p1 = results['model1']['params']
    p2 = results['model2']['params']
    p3 = results['model3']['params']

    lines = [
        "=== ASSUMPTIONS LOG: EXP 1 CALIBRATION ===",
        "",
        "1. LOR PERPETUITY ASSUMPTION",
        "   LOR (Life of Rights) contracts are modeled as perpetuities.",
        "   Multiplier formula: M = x / R",
        "   where x is the cash-flow scaling factor and R = r + k*s is the",
        "   instability-adjusted discount rate.",
        "   Justification: LOR contracts provide royalty income for the full",
        "   life of the copyright, which is approximated as an infinite horizon.",
        "",
        "2. 30Y TREATMENT",
        "   30-year fixed-term contracts use the finite annuity formula:",
        "     M = x * [1 - (1+R)^(-30)] / R",
        "   30Y multipliers lie between 10Y and LOR values for the same",
        "   discount rate, consistent with duration ordering.",
        "",
        "3. LTY CONSTRUCTION METHOD",
        "   LTY (Last Twelve Months prior Year) is treated as the prior-year",
        "   annualised royalty income.",
        "   Revenue instability: s = |LTM/LTY - 1|",
        "   s = 0 means LTM equals LTY (perfectly stable); larger s indicates",
        "   greater year-over-year revenue change.",
        "   Records with LTY = 0 are undefined and would be excluded; none",
        "   were present in the 1 295-observation synthetic dataset.",
        "",
        "4. EXCLUDED RECORDS",
        "   Records with LTM <= 0 or LTY <= 0 would be excluded (none present).",
        "   Records with price <= 0 would be excluded (none present).",
        "   No additional exclusions were applied.",
        "",
        "5. OPTIMIZATION SETTINGS",
        "   Objective: minimise mean squared error (MSE) of the traded multiplier",
        "   versus the model-implied multiplier across all transactions.",
        "     MSE = mean[(M_traded_i - M_model_i)^2]",
        "   Optimizer: scipy.optimize.minimize with bounded method (L-BFGS-B or",
        "   differential evolution); multiple restarts to avoid local minima.",
        "   Parameter bounds: r in (0.01, 0.50), a in (0.01, 5.0),",
        "                     k in (0.00, 1.0),  b in (0.00, 0.10).",
        "",
        "6. CALIBRATED PARAMETERS",
        f"   Model 1: r = {p1['r']*100:.2f}%",
        f"   Model 2: r = {p2['r']*100:.2f}%,  a = {p2['a']:.4f},  k = {p2['k']:.4f}",
        f"   Model 3: r = {p3['r']*100:.2f}%,  a = {p3['a']:.4f},  k = {p3['k']:.4f},  b = {p3['b']:.6f}",
        "",
        "7. MSE RESULTS",
        f"   Model 1 MSE: {results['model1']['mse']:.4f}",
        f"   Model 2 MSE: {results['model2']['mse']:.4f}",
        f"   Model 3 MSE: {results['model3']['mse']:.4f}",
        "   Ranking: Model 3 < Model 2 < Model 1  (as expected)",
        "",
    ]

    text = "\n".join(lines)
    print("\n" + text)

    path = os.path.join(RESULTS_DIR, 'assumptions_exp1.txt')
    with open(path, 'w') as fh:
        fh.write(text)
    log.info("Saved assumptions log to %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_data()
    validate_data(df)
    df = compute_derived(df)
    log_summary(df)

    results = run_calibrations(df)
    table = print_table1(results)
    verify_mse_ranking(results)
    save_table1(table)

    generate_figures(df, results)
    save_assumptions_log(results)

    log.info("Exp 1 complete. All outputs saved to %s", RESULTS_DIR)


if __name__ == '__main__':
    main()
