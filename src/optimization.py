"""Calibration engine for DCF pricing models.

Each model is calibrated by minimising the mean squared error (MSE) between
the traded price-to-LTM multiplier and the model-implied multiplier.

Global search uses ``scipy.optimize.differential_evolution``; the best
solution is then polished with ``scipy.optimize.minimize`` (L-BFGS-B).

Input DataFrame requirements
-----------------------------
Columns: ``ltm``, ``lty``, ``age``, ``term``, ``price``
  - ltm   : last-twelve-months revenue (float, > 0)
  - lty   : last-three-year annualised revenue (float, > 0)
  - age   : asset age in years (float)
  - term  : contract term string: '10Y', '30Y', or 'LOR'
  - price : transaction price in dollars (float, > 0)
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from typing import Dict, Tuple

from .models import (
    model1_multiplier,
    model2_multiplier,
    model3_multiplier_vec,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _traded_multipliers(df: pd.DataFrame) -> np.ndarray:
    """Return price / ltm for every row."""
    return (df["price"].values / df["ltm"].values)


def _instability(df: pd.DataFrame) -> np.ndarray:
    """Return |ltm/lty - 1| for every row."""
    return np.abs(df["ltm"].values / df["lty"].values - 1.0)


# ---------------------------------------------------------------------------
# SSE objectives
# ---------------------------------------------------------------------------

def sse_model1(params: np.ndarray, df: pd.DataFrame) -> float:
    """Compute SSE for Model 1.

    Objective: sum_i (M_traded_i - M1_i(r))^2

    Args:
        params: 1-element array [r].
        df:     Calibration DataFrame (must contain ltm, price, term).

    Returns:
        Scalar SSE value.
    """
    (r,) = params
    M_traded = _traded_multipliers(df)
    terms = df["term"].values
    M_model = np.array([model1_multiplier(t, r) for t in terms])
    return float(np.sum((M_traded - M_model) ** 2))


def sse_model2(params: np.ndarray, df: pd.DataFrame) -> float:
    """Compute SSE for Model 2.

    Objective: sum_i (M_traded_i - M2_i(r, a, k))^2

    Args:
        params: 3-element array [r, a, k].
        df:     Calibration DataFrame (must contain ltm, lty, price, term).

    Returns:
        Scalar SSE value.
    """
    r, a, k = params
    s = _instability(df)
    M_traded = _traded_multipliers(df)
    terms = df["term"].values
    M_model = np.array([model2_multiplier(s[i], terms[i], r, a, k) for i in range(len(df))])
    return float(np.sum((M_traded - M_model) ** 2))


def sse_model3(params: np.ndarray, df: pd.DataFrame) -> float:
    """Compute SSE for Model 3.

    Objective: sum_i (M_traded_i - M3_i(r, a, k, b))^2

    Args:
        params: 4-element array [r, a, k, b].
        df:     Calibration DataFrame (must contain ltm, lty, age, price, term).

    Returns:
        Scalar SSE value.
    """
    r, a, k, b = params
    s = _instability(df)
    age = df["age"].values
    terms = df["term"].values
    M_traded = _traded_multipliers(df)
    M_model = model3_multiplier_vec(s, age, terms, r, a, k, b)
    return float(np.sum((M_traded - M_model) ** 2))


# ---------------------------------------------------------------------------
# Calibration routines
# ---------------------------------------------------------------------------

def _polish(objective, x0: np.ndarray, bounds: list, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Run L-BFGS-B from x0 to refine a DE solution.

    Args:
        objective: Callable(params, df) → float.
        x0:        Starting point from differential evolution.
        bounds:    List of (lo, hi) tuples, one per parameter.
        df:        Calibration DataFrame.

    Returns:
        (best_params, best_sse) after local refinement.
    """
    result = minimize(
        objective,
        x0,
        args=(df,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 10_000},
    )
    return result.x, float(result.fun)


def calibrate_model1(df: pd.DataFrame) -> Dict:
    """Calibrate Model 1 by minimising MSE over r.

    Bounds: r ∈ (0.01, 0.50).
    Strategy: differential_evolution (global) → L-BFGS-B (local).

    Args:
        df: Calibration DataFrame with columns ltm, price, term.

    Returns:
        Dict with keys:
          - ``params``: dict {'r': float}
          - ``mse``:    mean squared error (SSE / n)
          - ``fitted_multipliers``: np.ndarray of model-implied multipliers
    """
    bounds = [(0.01, 0.50)]

    de = differential_evolution(
        sse_model1,
        bounds,
        args=(df,),
        seed=42,
        maxiter=2000,
        tol=1e-10,
        polish=True,
        workers=1,
    )
    best_x, best_sse = _polish(sse_model1, de.x, bounds, df)

    r = float(best_x[0])
    fitted = np.array([model1_multiplier(t, r) for t in df["term"].values])
    mse = best_sse / len(df)

    return {"params": {"r": r}, "mse": mse, "fitted_multipliers": fitted}


def calibrate_model2(df: pd.DataFrame) -> Dict:
    """Calibrate Model 2 by minimising MSE over (r, a, k).

    Bounds: r ∈ (0.01, 0.50), a ∈ (0.01, 5.0), k ∈ (0.0, 5.0).
    Strategy: differential_evolution (global) → L-BFGS-B (local).

    Args:
        df: Calibration DataFrame with columns ltm, lty, price, term.

    Returns:
        Dict with keys:
          - ``params``: dict {'r', 'a', 'k'}
          - ``mse``:    mean squared error (SSE / n)
          - ``fitted_multipliers``: np.ndarray of model-implied multipliers
    """
    bounds = [(0.01, 0.50), (0.01, 5.0), (0.0, 5.0)]

    de = differential_evolution(
        sse_model2,
        bounds,
        args=(df,),
        seed=42,
        maxiter=2000,
        tol=1e-10,
        polish=True,
        workers=1,
    )
    best_x, best_sse = _polish(sse_model2, de.x, bounds, df)

    r, a, k = best_x.tolist()
    s = _instability(df)
    terms = df["term"].values
    fitted = np.array([model2_multiplier(s[i], terms[i], r, a, k) for i in range(len(df))])
    mse = best_sse / len(df)

    return {
        "params": {"r": r, "a": a, "k": k},
        "mse": mse,
        "fitted_multipliers": fitted,
    }


def calibrate_model3(df: pd.DataFrame) -> Dict:
    """Calibrate Model 3 by minimising MSE over (r, a, k, b).

    Bounds: r ∈ (0.01, 0.50), a ∈ (0.01, 5.0), k ∈ (0.0, 5.0), b ∈ (0.0, 0.50).
    Strategy: differential_evolution (global) → L-BFGS-B (local).

    Args:
        df: Calibration DataFrame with columns ltm, lty, age, price, term.

    Returns:
        Dict with keys:
          - ``params``: dict {'r', 'a', 'k', 'b'}
          - ``mse``:    mean squared error (SSE / n)
          - ``fitted_multipliers``: np.ndarray of model-implied multipliers
    """
    bounds = [(0.01, 0.50), (0.01, 5.0), (0.0, 5.0), (0.0, 0.50)]

    de = differential_evolution(
        sse_model3,
        bounds,
        args=(df,),
        seed=42,
        maxiter=2000,
        tol=1e-10,
        polish=True,
        workers=1,
    )
    best_x, best_sse = _polish(sse_model3, de.x, bounds, df)

    r, a, k, b = best_x.tolist()
    s = _instability(df)
    age = df["age"].values
    terms = df["term"].values
    fitted = model3_multiplier_vec(s, age, terms, r, a, k, b)
    mse = best_sse / len(df)

    return {
        "params": {"r": r, "a": a, "k": k, "b": b},
        "mse": mse,
        "fitted_multipliers": fitted,
    }
