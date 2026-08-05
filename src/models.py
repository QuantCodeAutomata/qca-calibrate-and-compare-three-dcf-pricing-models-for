"""DCF pricing models for music royalty transactions.

Three nested models that price royalty contracts as discounted cash flow streams:
  Model 1 – constant cash flow, constant discount rate
  Model 2 – scaled cash flow with instability-adjusted discount rate
  Model 3 – age-adjusted cash flow with instability-adjusted discount rate
"""

import numpy as np
from typing import Union


# ---------------------------------------------------------------------------
# Primitive annuity calculators
# ---------------------------------------------------------------------------

def finite_horizon_multiplier(x: float, R: float, N: int) -> float:
    """Compute finite-horizon annuity multiplier.

    For a contract paying x * LTM annually for N years at discount rate R:
        multiplier = x * sum_{j=1}^N (1+R)^{-j} = x * (1 - (1+R)^{-N}) / R

    Args:
        x: Cash flow scalar (dimensionless scaling of LTM).
        R: Annual discount rate (decimal, must be > 0).
        N: Horizon in years (integer >= 1).

    Returns:
        Model-implied price-to-LTM multiplier.

    Raises:
        ValueError: If R <= 0 or N < 1.
    """
    if R <= 0:
        raise ValueError(f"Discount rate R must be positive, got {R}")
    if N < 1:
        raise ValueError(f"Horizon N must be >= 1, got {N}")
    return x * (1.0 - (1.0 + R) ** (-N)) / R


def perpetuity_multiplier(x: float, R: float) -> float:
    """Compute perpetuity multiplier for Life of Rights (LOR) contracts.

    Under the perpetuity assumption (paper's inferred assumption for LOR):
        multiplier = x / R

    Args:
        x: Cash flow scalar.
        R: Annual discount rate (decimal, must be > 0).

    Returns:
        Model-implied price-to-LTM multiplier.

    Raises:
        ValueError: If R <= 0.
    """
    if R <= 0:
        raise ValueError(f"Discount rate R must be positive, got {R}")
    return x / R


# ---------------------------------------------------------------------------
# Term → horizon mapping
# ---------------------------------------------------------------------------

def get_horizon(term: str) -> Union[int, str]:
    """Map a term label to its numeric horizon.

    Args:
        term: One of '10Y', '30Y', or 'LOR'.

    Returns:
        Integer 10 or 30 for fixed-term contracts; the string 'inf' for LOR.

    Raises:
        ValueError: If term is not recognised.
    """
    if term == "10Y":
        return 10
    if term == "30Y":
        return 30
    if term == "LOR":
        return "inf"
    raise ValueError(f"Unknown term {term!r}. Expected '10Y', '30Y', or 'LOR'.")


# ---------------------------------------------------------------------------
# Model 1 – C = LTM, R = r (no asset-level heterogeneity)
# ---------------------------------------------------------------------------

def model1_multiplier(term: str, r: float) -> float:
    """Model 1: C = LTM, R = r.

    The multiplier depends only on the contractual term and the single
    calibrated discount rate r; there are no asset-level inputs.

    For finite horizon N:  M = sum_{j=1}^N (1+r)^{-j} = (1 - (1+r)^{-N}) / r
    For LOR:               M = 1 / r

    Args:
        term: '10Y', '30Y', or 'LOR'.
        r:    Annual discount rate (decimal, > 0).

    Returns:
        Price-to-LTM multiplier implied by Model 1.
    """
    N = get_horizon(term)
    if N == "inf":
        return perpetuity_multiplier(1.0, r)
    return finite_horizon_multiplier(1.0, r, N)


# ---------------------------------------------------------------------------
# Model 2 – C = LTM * a, R = r + k * s
# ---------------------------------------------------------------------------

def model2_multiplier(s: float, term: str, r: float, a: float, k: float) -> float:
    """Model 2: C = LTM * a, R = r + k * |LTM/LTY - 1|.

    Introduces a cash-flow scalar *a* and an instability premium *k* on top
    of the base discount rate *r*.

    Args:
        s:    Revenue instability s = abs(LTM/LTY - 1).
        term: '10Y', '30Y', or 'LOR'.
        r:    Base annual discount rate (decimal, > 0).
        a:    Cash-flow scalar (dimensionless).
        k:    Instability loading on discount rate (>= 0).

    Returns:
        Price-to-LTM multiplier implied by Model 2.
    """
    R = r + k * s
    N = get_horizon(term)
    if N == "inf":
        return perpetuity_multiplier(a, R)
    return finite_horizon_multiplier(a, R, N)


# ---------------------------------------------------------------------------
# Model 3 – C = LTM * (a + b * age), R = r + k * s
# ---------------------------------------------------------------------------

def model3_multiplier(
    s: float, age: float, term: str, r: float, a: float, k: float, b: float
) -> float:
    """Model 3: C = LTM * (a + b * age), R = r + k * |LTM/LTY - 1|.

    Extends Model 2 by making the cash-flow scalar depend linearly on the
    asset's age, capturing the empirical finding that older catalogues
    generate relatively higher royalty income per dollar of LTM.

    Args:
        s:    Revenue instability s = abs(LTM/LTY - 1).
        age:  Asset age in years.
        term: '10Y', '30Y', or 'LOR'.
        r:    Base annual discount rate (decimal, > 0).
        a:    Intercept of cash-flow scalar.
        k:    Instability loading on discount rate (>= 0).
        b:    Age slope of cash-flow scalar (>= 0).

    Returns:
        Price-to-LTM multiplier implied by Model 3.
    """
    R = r + k * s
    x = a + b * age
    N = get_horizon(term)
    if N == "inf":
        return perpetuity_multiplier(x, R)
    return finite_horizon_multiplier(x, R, N)


# ---------------------------------------------------------------------------
# Vectorised Model 3 for calibration
# ---------------------------------------------------------------------------

def model3_multiplier_vec(
    s: np.ndarray,
    age: np.ndarray,
    term: np.ndarray,
    r: float,
    a: float,
    k: float,
    b: float,
) -> np.ndarray:
    """Vectorised Model 3 for efficient calibration over all transactions.

    Applies Model 3 element-wise across arrays of instability, age, and term,
    using a single set of scalar parameters.

    Args:
        s:    1-D array of revenue instability values.
        age:  1-D array of asset ages (years).
        term: 1-D array of term strings ('10Y', '30Y', 'LOR').
        r:    Base annual discount rate.
        a:    Cash-flow scalar intercept.
        k:    Instability loading.
        b:    Age slope.

    Returns:
        1-D numpy array of model-implied price-to-LTM multipliers.
    """
    R = r + k * s          # shape (n,)
    x = a + b * age        # shape (n,)

    result = np.empty(len(s), dtype=float)
    for i in range(len(s)):
        N = get_horizon(term[i])
        if N == "inf":
            result[i] = x[i] / R[i]
        else:
            result[i] = x[i] * (1.0 - (1.0 + R[i]) ** (-N)) / R[i]
    return result
