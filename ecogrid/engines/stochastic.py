"""Two-stage stochastic / robust unit-commitment engine.

Extends the deterministic MILP to optimise over an *ensemble* of weather
scenarios instead of a single forecast:

* **First stage (here-and-now):** the binary commitment schedule ``u(t)`` is
  decided before the weather is known and is therefore shared across every
  scenario.
* **Second stage (recourse):** the continuous dispatch (coal output, battery
  charge/discharge, state of charge) adapts to each realised scenario.

The risk-neutral objective minimises the *expected* cost. Setting
``risk_lambda > 0`` blends in the Conditional Value at Risk (CVaR) via the
Rockafellar-Uryasev linear formulation, yielding a robust schedule that hedges
against the worst-case tail of weather outcomes (e.g. a multi-day overcast,
wind-still spell).
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo

from ecogrid.config import GridConfig
from ecogrid.results import StochasticResult
from ecogrid.scenarios import ScenarioSet


class StochasticDispatchEngine:
    """Scenario-based two-stage stochastic unit-commitment MILP.

    Parameters
    ----------
    config
        Micro-grid parameters.
    risk_lambda
        Weight on CVaR in the objective. ``0.0`` is risk-neutral (expected
        cost); ``1.0`` minimises CVaR alone. Values in between trade expected
        cost against tail risk.
    cvar_alpha
        Confidence level for the CVaR term (e.g. ``0.95`` hedges the worst 5%).
    initial_commitment
        Whether the thermal unit is on at ``t = -1``.
    solver_name
        Pyomo solver factory key. Defaults to ``"appsi_highs"``.

    Notes
    -----
    Charge/discharge exclusivity binaries are intentionally omitted in the
    second stage to keep the extensive form tractable: with a sub-unity
    round-trip efficiency and no negative prices, simultaneous charge and
    discharge is always strictly sub-optimal and never appears in the solution.
    """

    def __init__(
        self,
        config: GridConfig | None = None,
        risk_lambda: float = 0.0,
        cvar_alpha: float = 0.95,
        initial_commitment: bool = False,
        solver_name: str = "appsi_highs",
    ) -> None:
        self.config = config if config is not None else GridConfig()
        self.risk_lambda = float(risk_lambda)
        self.cvar_alpha = float(cvar_alpha)
        self.initial_commitment = initial_commitment
        self.solver_name = solver_name

    def solve(self, demand: np.ndarray, scenarios: ScenarioSet) -> StochasticResult:
        """Solve the two-stage stochastic unit-commitment problem.

        Parameters
        ----------
        demand
            Demand per period, shape ``(horizon,)`` in MW. Assumed identical
            across scenarios (only weather is uncertain here).
        scenarios
            Monte-Carlo ensemble of wind / solar availability.

        Returns
        -------
        StochasticResult
            First-stage commitment, mean recourse dispatch and the full cost
            distribution with its CVaR / VaR.
        """
        demand = np.asarray(demand, dtype=float)
        horizon = demand.shape[0]
        if scenarios.horizon != horizon:
            raise ValueError("scenarios horizon must match demand length")
        cfg = self.config
        n_s = scenarios.n_scenarios
        prob = scenarios.probabilities
        alpha = self.cvar_alpha

        m = pyo.ConcreteModel(name="EcoGrid-StochasticUC")
        m.T = pyo.RangeSet(0, horizon - 1)
        m.S = pyo.RangeSet(0, n_s - 1)
        u0 = 1 if self.initial_commitment else 0

        # --- First-stage (scenario-independent) commitment variables ---
        m.u = pyo.Var(m.T, domain=pyo.Binary)
        m.su = pyo.Var(m.T, domain=pyo.Binary)
        m.sd = pyo.Var(m.T, domain=pyo.Binary)

        # --- Second-stage (per-scenario) recourse variables ---
        m.p_coal = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
        m.p_wind = pyo.Var(
            m.S, m.T, domain=pyo.NonNegativeReals,
            bounds=lambda _m, s, t: (0.0, float(scenarios.wind[s, t])),
        )
        m.p_solar = pyo.Var(
            m.S, m.T, domain=pyo.NonNegativeReals,
            bounds=lambda _m, s, t: (0.0, float(scenarios.solar[s, t])),
        )
        m.charge = pyo.Var(
            m.S, m.T, domain=pyo.NonNegativeReals, bounds=(0.0, cfg.bess_max_power)
        )
        m.discharge = pyo.Var(
            m.S, m.T, domain=pyo.NonNegativeReals, bounds=(0.0, cfg.bess_max_power)
        )
        m.soc = pyo.Var(
            m.S, m.T, domain=pyo.NonNegativeReals, bounds=(0.0, cfg.bess_capacity)
        )

        # --- Commitment logic (first stage) ---
        m.coal_max = pyo.Constraint(
            m.S, m.T, rule=lambda _m, s, t: _m.p_coal[s, t] <= cfg.coal_pmax * _m.u[t]
        )
        m.coal_min = pyo.Constraint(
            m.S, m.T, rule=lambda _m, s, t: _m.p_coal[s, t] >= cfg.coal_pmin * _m.u[t]
        )

        def transition_rule(_m, t):
            prev = u0 if t == 0 else _m.u[t - 1]
            return _m.u[t] - prev == _m.su[t] - _m.sd[t]

        m.transition = pyo.Constraint(m.T, rule=transition_rule)

        def min_up_rule(_m, t):
            window = list(range(t, min(t + cfg.min_up_time, horizon)))
            return sum(_m.u[k] for k in window) >= len(window) * _m.su[t]

        m.min_up = pyo.Constraint(m.T, rule=min_up_rule)

        def min_down_rule(_m, t):
            window = list(range(t, min(t + cfg.min_down_time, horizon)))
            return sum((1 - _m.u[k]) for k in window) >= len(window) * _m.sd[t]

        m.min_down = pyo.Constraint(m.T, rule=min_down_rule)

        # --- Per-scenario recourse constraints ---
        def balance_rule(_m, s, t):
            supply = _m.p_coal[s, t] + _m.p_wind[s, t] + _m.p_solar[s, t] + _m.discharge[s, t]
            return supply == demand[t] + _m.charge[s, t]

        m.balance = pyo.Constraint(m.S, m.T, rule=balance_rule)

        def soc_rule(_m, s, t):
            soc_prev = cfg.initial_soc if t == 0 else _m.soc[s, t - 1]
            inflow = cfg.eff_charge * _m.charge[s, t]
            outflow = _m.discharge[s, t] / cfg.eff_discharge
            return _m.soc[s, t] == soc_prev + inflow - outflow

        m.soc_dynamics = pyo.Constraint(m.S, m.T, rule=soc_rule)

        # --- Cost expressions ---
        startup_total = cfg.startup_cost * sum(m.su[t] for t in m.T)

        def scenario_cost(s):
            op = sum(m.p_coal[s, t] * cfg.coal_marginal_cost for t in m.T)
            return op + startup_total

        expected_cost = sum(prob[s] * scenario_cost(s) for s in m.S)

        # --- CVaR (Rockafellar-Uryasev): eta = VaR, excess_s = (cost_s - eta)+ ---
        m.eta = pyo.Var(domain=pyo.Reals)
        m.excess = pyo.Var(m.S, domain=pyo.NonNegativeReals)
        m.cvar_link = pyo.Constraint(
            m.S, rule=lambda _m, s: _m.excess[s] >= scenario_cost(s) - _m.eta
        )
        cvar_expr = m.eta + (1.0 / (1.0 - alpha)) * sum(prob[s] * m.excess[s] for s in m.S)

        m.obj = pyo.Objective(
            expr=(1.0 - self.risk_lambda) * expected_cost + self.risk_lambda * cvar_expr,
            sense=pyo.minimize,
        )

        solver = pyo.SolverFactory(self.solver_name)
        results = solver.solve(m, load_solutions=False)
        tc = str(results.solver.termination_condition).lower()
        success = "optimal" in tc

        if not success:
            z = np.zeros(horizon)
            return StochasticResult(
                commitment=z.copy(), coal_mean=z.copy(), charge_mean=z.copy(),
                discharge_mean=z.copy(), soc_mean=z.copy(),
                scenario_costs=np.zeros(n_s), expected_cost=float("inf"),
                cvar=float("inf"), var=float("inf"), alpha=alpha,
                success=False, message=f"solver status: {tc}",
            )

        m.solutions.load_from(results)

        commitment = np.round(np.array([pyo.value(m.u[t]) for t in m.T]))
        startup_val = cfg.startup_cost * float(sum(pyo.value(m.su[t]) for t in m.T))

        coal = np.array([[pyo.value(m.p_coal[s, t]) for t in m.T] for s in m.S])
        charge = np.array([[pyo.value(m.charge[s, t]) for t in m.T] for s in m.S])
        discharge = np.array([[pyo.value(m.discharge[s, t]) for t in m.T] for s in m.S])
        soc = np.array([[pyo.value(m.soc[s, t]) for t in m.T] for s in m.S])

        scenario_costs = coal.sum(axis=1) * cfg.coal_marginal_cost + startup_val
        w = prob[:, None]
        return StochasticResult(
            commitment=commitment,
            coal_mean=(coal * w).sum(axis=0),
            charge_mean=(charge * w).sum(axis=0),
            discharge_mean=(discharge * w).sum(axis=0),
            soc_mean=(soc * w).sum(axis=0),
            scenario_costs=scenario_costs,
            expected_cost=float(np.dot(prob, scenario_costs)),
            cvar=float(pyo.value(cvar_expr)),
            var=float(pyo.value(m.eta)),
            alpha=alpha,
            success=True,
            message=tc,
        )
