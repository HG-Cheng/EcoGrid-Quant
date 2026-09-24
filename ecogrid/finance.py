"""Capital-budgeting and risk valuation for grid storage investment.

Replaces the V3 "nominal total cost" U-curve with the valuation primitives a
project-finance or infrastructure-fund analyst would actually use: discounted
cash flow / net present value, the levelised cost of energy, and Conditional
Value at Risk for the cost distribution under weather uncertainty.
"""

from __future__ import annotations

from itertools import chain
from math import fsum

import numpy as np


def discount_factors(discount_rate: float, n_years: int) -> np.ndarray:
    """Return per-year discount factors ``1 / (1 + r) ** y`` for ``y = 1..n``.

    Parameters
    ----------
    discount_rate
        Annual discount rate as a fraction (e.g. ``0.08`` for 8%).
    n_years
        Number of future years.

    Returns
    -------
    numpy.ndarray
        Discount factors of shape ``(n_years,)``.
    """
    years = np.arange(1, n_years + 1)
    return 1.0 / (1.0 + discount_rate) ** years


def npv(
    annual_savings: float | np.ndarray,
    capex: float,
    discount_rate: float,
    lifespan_years: int,
) -> float:
    """Net present value of a storage investment.

    The battery is paid for today (``capex``) and returns a stream of annual
    operating savings (e.g. avoided carbon tax + fuel) over its economic life.
    Future savings are discounted to present value before subtracting capex.

    Parameters
    ----------
    annual_savings
        Either a scalar constant annual saving, or an array of per-year savings
        of length ``lifespan_years``, in EUR.
    capex
        Overnight capital expenditure incurred at year 0, in EUR.
    discount_rate
        Annual discount rate as a fraction.
    lifespan_years
        Economic life of the asset, in years.

    Returns
    -------
    float
        Net present value in EUR. Positive means value-accretive.
    """
    factors = discount_factors(discount_rate, lifespan_years)
    savings = np.asarray(annual_savings, dtype=float)
    if savings.ndim == 0:
        savings = np.full(lifespan_years, float(savings))
    if savings.shape[0] != lifespan_years:
        raise ValueError("annual_savings length must equal lifespan_years")
    return float(np.sum(savings * factors) - capex)


def lcoe(
    capex: float,
    annual_cost: float | np.ndarray,
    annual_energy_mwh: float | np.ndarray,
    discount_rate: float,
    lifespan_years: int,
) -> float:
    """Levelised cost of energy (LCOE), in EUR/MWh.

    LCOE is the constant price per MWh that makes the discounted revenue equal
    the discounted lifecycle cost (capex plus discounted operating costs),
    using a consistently discounted energy denominator.

    Parameters
    ----------
    capex
        Overnight capital expenditure at year 0, in EUR.
    annual_cost
        Scalar or per-year operating cost over the life, in EUR.
    annual_energy_mwh
        Scalar or per-year energy served, in MWh.
    discount_rate
        Annual discount rate as a fraction.
    lifespan_years
        Economic life of the asset, in years.

    Returns
    -------
    float
        Levelised cost in EUR/MWh.
    """
    factors = discount_factors(discount_rate, lifespan_years)
    cost = np.asarray(annual_cost, dtype=float)
    energy = np.asarray(annual_energy_mwh, dtype=float)
    if cost.ndim == 0:
        cost = np.full(lifespan_years, float(cost))
    if energy.ndim == 0:
        energy = np.full(lifespan_years, float(energy))

    discounted_cost = capex + float(np.sum(cost * factors))
    discounted_energy = float(np.sum(energy * factors))
    if discounted_energy <= 0:
        raise ValueError("discounted energy must be positive to compute LCOE")
    return discounted_cost / discounted_energy


def _risk_distribution(
    costs: np.ndarray,
    alpha: float,
    probabilities: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a discrete loss distribution and return positive-mass support."""
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or costs.size == 0 or not np.all(np.isfinite(costs)):
        raise ValueError("costs must be a nonempty 1D array of finite losses")
    if (
        not isinstance(alpha, (int, float, np.integer, np.floating))
        or isinstance(alpha, (bool, np.bool_))
        or not np.isfinite(alpha)
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("alpha must lie strictly between 0 and 1")

    probs: np.ndarray
    if probabilities is None:
        probs = np.full(costs.size, 1.0 / costs.size)
    else:
        probs = np.asarray(probabilities, dtype=float)
        if probs.shape != costs.shape or not np.all(np.isfinite(probs)):
            raise ValueError("probabilities must match costs and be finite")
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError("probabilities must lie between 0 and 1")
    total = fsum(probs)
    if probabilities is not None and not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("probabilities must sum to 1")
    probs = probs / total

    order = np.argsort(costs)
    sorted_costs = costs[order]
    sorted_probs = probs[order]
    positive = sorted_probs > 0.0
    return sorted_costs[positive], sorted_probs[positive]


def value_at_risk(
    costs: np.ndarray,
    alpha: float = 0.95,
    probabilities: np.ndarray | None = None,
) -> float:
    """Value at Risk (VaR) of a cost distribution at confidence ``alpha``.

    For costs (losses), VaR is the lower empirical ``alpha`` quantile:
    ``inf{x : P(cost <= x) >= alpha}``.

    Parameters
    ----------
    costs
        Realised cost per scenario, shape ``(n,)``.
    alpha
        Confidence level in ``(0, 1)``.
    probabilities
        Optional scenario probabilities; defaults to equiprobable.

    Returns
    -------
    float
        The VaR threshold in the same units as ``costs``.
    """
    sorted_costs, sorted_probs = _risk_distribution(costs, alpha, probabilities)
    left, right = 0, sorted_costs.size - 1
    while left < right:
        mid = (left + right) // 2
        # Subtract inside fsum so a tiny positive mass is not rounded away.
        if fsum(chain(sorted_probs[: mid + 1], (-alpha,))) >= 0.0:
            right = mid
        else:
            left = mid + 1
    return float(sorted_costs[left])


def cvar(
    costs: np.ndarray,
    alpha: float = 0.95,
    probabilities: np.ndarray | None = None,
) -> float:
    """Conditional Value at Risk (expected shortfall) of a cost distribution.

    CVaR at level ``alpha`` is the probability-weighted mean cost of the worst
    ``(1 - alpha)`` tail, including just the needed mass at the VaR threshold.
    Equal losses and scenarios with zero probability are handled naturally.

    Parameters
    ----------
    costs
        Realised cost per scenario, shape ``(n,)``.
    alpha
        Confidence level in ``(0, 1)``.
    probabilities
        Optional scenario probabilities; defaults to equiprobable.

    Returns
    -------
    float
        The CVaR in the same units as ``costs``. Always ``>= VaR``.
    """
    sorted_costs, sorted_probs = _risk_distribution(costs, alpha, probabilities)
    largest_first_probs = sorted_probs[::-1]
    prior_tail_mass = np.concatenate(([0.0], np.cumsum(largest_first_probs)[:-1]))
    tail_mass = 1.0 - alpha
    tail_weights = np.clip(tail_mass - prior_tail_mass, 0.0, largest_first_probs)
    return float(np.sum(tail_weights * sorted_costs[::-1]) / tail_mass)
