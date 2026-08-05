# DCF Pricing Models for Music Royalty Transactions

Calibration and comparison of three discounted cash flow pricing models for music royalty assets using Royalty Exchange-style transaction data.

## Overview

This repository implements and compares three DCF pricing models:
- **Model 1**: Constant cash flow, constant discount rate (r)
- **Model 2**: Scaled cash flow with instability-adjusted discount rate (r, a, k)
- **Model 3**: Age-adjusted cash flow with instability-adjusted discount rate (r, a, k, b)

## Repository Structure

```
repo/
├── data/                   # Data loading and synthetic data generation
│   ├── generate_royalty_data.py
│   └── load_sp500.py
├── src/                    # Core implementations
│   ├── models.py           # DCF Model 1, 2, 3 pricing functions
│   ├── optimization.py     # Parameter calibration engine
│   ├── backtest_engine.py  # One-year and five-year backtest logic
│   └── benchmark.py        # S&P 500 benchmark
├── exp/                    # Experiment scripts
│   ├── exp1_calibration.py
│   ├── exp2_one_year_backtest.py
│   ├── exp3_five_year_backtest.py
│   └── exp4_sp500_benchmark.py
├── tests/                  # pytest test suite
├── results/                # Outputs: metrics, charts, RESULTS.md
└── AGENTS.md               # Repository knowledge base
```

## Experiments

### Exp 1: DCF Model Calibration
Calibrate Models 1–3 to Royalty Exchange-style transaction data. Reproduces Table 1 from the paper with target MSE values: Model 1 ≈ 8.8, Model 2 ≈ 7.7, Model 3 ≈ 5.7.

### Exp 2: One-Year Cross-Sectional Backtest
Annual buy-and-hold backtest for 2017–2021 cohorts using Model 3 pricing. Separately reports LOR and 10Y asset returns.

### Exp 3: Five-Year Cross-Sectional Backtest
Five-year (2017–2022) buy-and-hold backtest for LOR and 10Y music royalty assets. Target: LOR median ≈ 82.6%, 10Y median ≈ 42.5%.

### Exp 4: S&P 500 Benchmark
Equity cross-sectional benchmark for one-year and five-year comparison. Target: five-year median total return ≈ 77.7%.

## Setup

```bash
pip install numpy pandas scipy matplotlib seaborn quantstats yfinance pytest
```

## Running Experiments

```bash
python exp/exp1_calibration.py
python exp/exp2_one_year_backtest.py
python exp/exp3_five_year_backtest.py
python exp/exp4_sp500_benchmark.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Assumptions

- LOR contracts use perpetuity pricing: multiplier = x / R
- Transaction costs: fixed $500 buyer fee + 8% seller commission on exit price
- Synthetic data generated to reproduce paper-consistent parameter calibration
- S&P 500 benchmark uses 2017 constituent list with yfinance adjusted prices
