# AGENTS.md — Repository Knowledge Base

## Project Overview

Music royalty DCF pricing: implements and calibrates three nested pricing models
against Royalty Exchange-style transaction data, then runs buy-and-hold backtests.

## Repository Layout

```
data/
  generate_royalty_data.py   # synthetic data generation (run to produce CSVs)
  royalty_transactions.csv   # 1 295 transactions (generated)
  quarterly_revenues.csv     # 31 080 rows long-format panel (generated)
src/
  models.py                  # DCF multiplier formulas (Models 1–3, vectorised)
  optimization.py            # calibration engine (differential_evolution + L-BFGS-B)
  backtest_engine.py         # 1-year and 5-year buy-and-hold backtest
exp/
  exp1_calibration.py        # run calibration, produce Table 1
  exp2_one_year_backtest.py
  exp3_five_year_backtest.py
  exp4_sp500_benchmark.py
tests/
  test_models.py             # 12 unit tests (all pass)
results/                     # outputs from exp scripts
```

## Data Generation

Run once to produce the CSVs:
```bash
python data/generate_royalty_data.py
```

- `generate_royalty_dataset(n=1295, seed=42)` – main transaction table
- `generate_quarterly_revenue_panel(df)` – long-format quarterly revenue panel
- True DGP: Model 3 with r=0.083, a=0.61, k=0.058, b=0.0098, noise N(0, 2.4)

### Calibration Results on Synthetic Data

| Model | r     | a     | k      | b      | MSE  |
|-------|-------|-------|--------|--------|------|
| 1     | 0.128 | —     | —      | —      | 6.95 |
| 2     | 0.089 | 0.798 | 0.0599 | —      | 6.26 |
| 3     | 0.090 | 0.652 | 0.0591 | 0.0098 | 5.78 |

Paper targets (≈): M1 r=14% MSE=8.8, M2 r=7.6% a=0.69 k=0.071 MSE=7.7, M3 r=8.3% a=0.61 k=0.058 b=0.0098 MSE=5.7.

MSE3 = 2.4² = 5.76 is exactly reproduced. The smaller MSE gaps between models
are a mathematical consequence of b=0.0098 with age ∈ [1,30]:
  MSE2−MSE3 ≈ b² × Var(age) × E[F²] ≈ (0.0098)² × 70 × 87 ≈ 0.6
The paper's larger gaps reflect additional idiosyncratic variation in real data.

## Key Implementation Notes

### src/models.py
- `get_horizon('LOR')` returns the string `'inf'` (not `numpy.inf`) — checked
  with `== 'inf'` in vectorised function.
- `model3_multiplier_vec` loops over rows (not fully vectorised) — fine for
  1 295 rows; if performance matters, replace with numpy boolean indexing.

### src/optimization.py
- All three calibration functions: differential_evolution (global) → L-BFGS-B (local).
- Input DataFrame must have columns: `ltm`, `lty`, `age`, `term`, `price`.
- Traded multiplier: `M_traded = price / ltm`.
- Instability: `s = |ltm/lty − 1|`.

### src/backtest_engine.py
- `DEFAULT_PARAMS` = paper Model 3 params (r=0.083, a=0.61, k=0.058, b=0.0098).
- Accepts `transaction_id` OR `asset_id` column in transactions DataFrame.
- Forward revenues: last historical quarter × 0.97^q × (1 + N(0, 0.05)).
- Exit LTM = last 4 forward quarters; Exit LTY = last 12 quarters / 3.
- Transaction costs: $500 fixed + 8 % of exit price.
- 5-year backtest filters to 2017 cohort only.
- `price_model3` bypasses `model3_multiplier` for finite-term exits so arbitrary
  remaining horizons (not just 10/30) are supported.

### data/generate_royalty_data.py
- Instability distribution: 65 % N(0, 0.09) + 35 % N(0, 0.65), clipped to (−0.90, 1.00).
- `_generate_quarterly_revenues` uses seed_val = master_seed × 1000 + row_index.
- `generate_quarterly_revenue_panel`: last quarter = fiscal quarter containing
  transaction_date; dates go back 24 quarters (6 years).

## Testing

```bash
pytest tests/ -v        # 12 unit tests, all pass
```

No mocks; tests exercise real code paths against analytic expected values.
