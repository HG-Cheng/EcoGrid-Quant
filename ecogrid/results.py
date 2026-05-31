"""Unified dispatch result container.

Replaces the bare :class:`scipy.optimize.OptimizeResult` returned by the legacy
SLSQP engine with a typed, engine-agnostic structure so that downstream code
(finance, back-testing, plotting) does not depend on a specific solver's API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DispatchResult:
    """Outcome of a (multi-period) dispatch optimisation.

    All per-period arrays share the same length ``horizon`` and are aligned by
    index, so ``coal[t]`` and ``soc[t]`` describe the same hour ``t``.

    Parameters
    ----------
    coal, wind, solar, charge, discharge, soc
        Per-period decision trajectories, each a 1-D array of length
        ``horizon`` in MW (MWh for ``soc``).
    commitment
        Binary on/off status of the thermal unit per period, or ``None`` for
        engines without commitment variables. Length ``horizon``.
    total_cost
        Optimal objective value (operating cost incl. carbon tax and start-up
        cost), in EUR over the optimisation horizon.
    success
        Whether the solver reported an optimal / feasible solution.
    solver
        Identifier of the engine that produced the result (e.g. ``"milp"``).
    message
        Human-readable solver status message.
    """

    coal: np.ndarray
    wind: np.ndarray
    solar: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    total_cost: float
    success: bool
    solver: str
    commitment: np.ndarray | None = None
    message: str = ""

    @property
    def horizon(self) -> int:
        """Number of time periods in the dispatch trajectory."""
        return int(self.coal.shape[0])

    @property
    def total_emissions(self) -> float:
        """Total CO2 emitted over the horizon, in tonnes.

        Uses a 0.8 t/MWh default intensity only as a fallback; callers that
        need an exact figure should compute it from their own
        :class:`~ecogrid.config.GridConfig`.
        """
        return float(np.sum(self.coal) * 0.8)

    def to_frame(self) -> pd.DataFrame:
        """Return the dispatch trajectory as a tidy :class:`pandas.DataFrame`."""
        data = {
            "coal": self.coal,
            "wind": self.wind,
            "solar": self.solar,
            "charge": self.charge,
            "discharge": self.discharge,
            "soc": self.soc,
        }
        if self.commitment is not None:
            data["commitment"] = self.commitment
        return pd.DataFrame(data)


@dataclass
class StochasticResult:
    """Outcome of a two-stage stochastic unit-commitment optimisation.

    The first-stage commitment ``commitment`` is decided here-and-now and is
    shared across every scenario; the recourse dispatch differs per scenario.
    The probability-weighted *mean* recourse trajectories are returned for
    plotting, while ``scenario_costs`` carries the full cost distribution used
    for tail-risk analysis.

    Parameters
    ----------
    commitment
        First-stage thermal on/off schedule, shape ``(horizon,)``.
    coal_mean, charge_mean, discharge_mean, soc_mean
        Probability-weighted mean recourse trajectories, shape ``(horizon,)``.
    scenario_costs
        Total system cost realised in each scenario, shape ``(n_scenarios,)``.
    expected_cost
        Probability-weighted mean of ``scenario_costs``, in EUR.
    cvar, var
        Conditional Value at Risk and Value at Risk of the cost distribution at
        confidence ``alpha`` (from the Rockafellar-Uryasev formulation).
    alpha
        CVaR confidence level used by the optimiser.
    success
        Whether the solver reported an optimal solution.
    message
        Solver status message.
    """

    commitment: np.ndarray
    coal_mean: np.ndarray
    charge_mean: np.ndarray
    discharge_mean: np.ndarray
    soc_mean: np.ndarray
    scenario_costs: np.ndarray
    expected_cost: float
    cvar: float
    var: float
    alpha: float
    success: bool
    message: str = ""

    @property
    def n_scenarios(self) -> int:
        """Number of scenarios in the cost distribution."""
        return int(self.scenario_costs.shape[0])
