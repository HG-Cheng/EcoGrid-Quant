"""Capital-budgeting and risk valuation for grid storage investment.

Replaces the V3 "nominal total cost" U-curve with the valuation primitives a
project-finance or infrastructure-fund analyst would actually use: discounted
cash flow / net present value, the levelised cost of energy, and Conditional
Value at Risk for the cost distribution under weather uncertainty.
"""

from __future__ import annotations

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


def value_at_risk(
    costs: np.ndarray,
    alpha: float = 0.95,
    probabilities: np.ndarray | None = None,
) -> float:
    """Value at Risk (VaR) of a cost distribution at confidence ``alpha``.

    For costs (losses), VaR is the ``alpha`` quantile: the threshold that cost
    does not exceed with probability ``alpha``.

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
    costs = np.asarray(costs, dtype=float)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    order = np.argsort(costs)
    sorted_costs = costs[order]
    if probabilities is None:
        return float(np.quantile(sorted_costs, alpha, method="higher"))
    probs = np.asarray(probabilities, dtype=float)[order]
    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, alpha))
    idx = min(idx, sorted_costs.shape[0] - 1)
    return float(sorted_costs[idx])


def cvar(
    costs: np.ndarray,
    alpha: float = 0.95,
    probabilities: np.ndarray | None = None,
) -> float:
    """Conditional Value at Risk (expected shortfall) of a cost distribution.

    CVaR at level ``alpha`` is the probability-weighted mean cost of the worst
    ``(1 - alpha)`` tail of scenarios, i.e. the expected cost *given* that cost
    exceeds the VaR. It is the coherent tail-risk metric used to answer "in the
    worst weather outcomes, how bad does the bill get?".

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
    costs = np.asarray(costs, dtype=float)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    n = costs.shape[0]
    probs = (
        np.full(n, 1.0 / n)
        if probabilities is None
        else np.asarray(probabilities, dtype=float)
    )

    order = np.argsort(costs)
    sorted_costs = costs[order]
    sorted_probs = probs[order]
    cdf = np.cumsum(sorted_probs)

    # Tail mass beyond the alpha quantile.
    tail_mass = 1.0 - alpha
    # Weight of each scenario falling in the worst (1 - alpha) tail.
    in_tail = np.clip(cdf - alpha, 0.0, None)
    # Convert cumulative excess into per-scenario tail weights.
    tail_weights = np.diff(np.concatenate([[0.0], in_tail]))
    if tail_weights.sum() <= 0:
        return float(sorted_costs[-1])
    return float(np.sum(tail_weights * sorted_costs) / tail_mass)
