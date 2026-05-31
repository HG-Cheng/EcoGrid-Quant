"""Industrial MILP dispatch engine (Pyomo + HiGHS).

Upgrades the continuous SLSQP baseline to a Mixed-Integer Linear Program that
captures the *discrete* physics of a thermal unit: it cannot run below a stable
minimum output, it pays a fixed start-up cost each time it is switched on, and
it must respect minimum up / down times. These realities require binary
commitment variables and are exactly what separates a toy from an industrial
unit-commitment engine.
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo

from ecogrid.config import GridConfig
from ecogrid.engines.base import DispatchEngine
from ecogrid.results import DispatchResult


class MilpDispatchEngine(DispatchEngine):
    """Unit-commitment economic dispatch as a MILP solved with HiGHS.

    Parameters
    ----------
    config
        Micro-grid parameters. The thermal floor (``coal_pmin``), start-up cost
        and minimum up/down times drive the integer structure of the model.
    initial_commitment
        Whether the thermal unit is already running at ``t = -1`` (affects the
        first start-up and minimum-time constraints). Defaults to ``False``.
    solver_name
        Pyomo solver factory key. Defaults to the in-process HiGHS interface
        ``"appsi_highs"`` which requires no external binary.
    """

    def __init__(
        self,
        config: GridConfig | None = None,
        initial_commitment: bool = False,
        solver_name: str = "appsi_highs",
    ) -> None:
        super().__init__(config)
        self.initial_commitment = initial_commitment
        self.solver_name = solver_name

    def build_model(
        self,
        demand: np.ndarray,
        wind_avail: np.ndarray,
        solar_avail: np.ndarray,
    ) -> pyo.ConcreteModel:
        """Construct the Pyomo unit-commitment model without solving it.

        Exposed separately so tests and the stochastic extension can introspect
        or compose the model.
        """
        horizon = self._validate_inputs(demand, wind_avail, solar_avail)
        demand = np.asarray(demand, dtype=float)
        wind_avail = np.asarray(wind_avail, dtype=float)
        solar_avail = np.asarray(solar_avail, dtype=float)
        cfg = self.config

        m = pyo.ConcreteModel(name="EcoGrid-UnitCommitment")
        m.T = pyo.RangeSet(0, horizon - 1)

        # Big-M for charge/discharge exclusivity: zero capacity collapses it.
        big_m = max(cfg.bess_max_power, 1e-9)
        u0 = 1 if self.initial_commitment else 0

        # --- Continuous decision variables (all in MW, soc in MWh) ---
        m.p_coal = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.p_wind = pyo.Var(
            m.T, domain=pyo.NonNegativeReals,
            bounds=lambda _m, t: (0.0, float(wind_avail[t])),
        )
        m.p_solar = pyo.Var(
            m.T, domain=pyo.NonNegativeReals,
            bounds=lambda _m, t: (0.0, float(solar_avail[t])),
        )
        m.charge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, cfg.bess_max_power))
        m.discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, cfg.bess_max_power))
        m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, cfg.bess_capacity))

        # --- Binary variables: commitment, start-up, shutdown, charge flag ---
        m.u = pyo.Var(m.T, domain=pyo.Binary)
        m.su = pyo.Var(m.T, domain=pyo.Binary)
        m.sd = pyo.Var(m.T, domain=pyo.Binary)
        m.z = pyo.Var(m.T, domain=pyo.Binary)

        # --- Power balance: supply == demand + charging load ---
        def balance_rule(_m: pyo.ConcreteModel, t: int):
            supply = _m.p_coal[t] + _m.p_wind[t] + _m.p_solar[t] + _m.discharge[t]
            return supply == demand[t] + _m.charge[t]

        m.balance = pyo.Constraint(m.T, rule=balance_rule)

        # --- Thermal output linked to commitment (creates the MILP coupling) ---
        m.coal_max = pyo.Constraint(
            m.T, rule=lambda _m, t: _m.p_coal[t] <= cfg.coal_pmax * _m.u[t]
        )
        m.coal_min = pyo.Constraint(
            m.T, rule=lambda _m, t: _m.p_coal[t] >= cfg.coal_pmin * _m.u[t]
        )

        # --- State-of-charge dynamics (linear, fixed efficiency) ---
        def soc_rule(_m: pyo.ConcreteModel, t: int):
            soc_prev = cfg.initial_soc if t == 0 else _m.soc[t - 1]
            inflow = cfg.eff_charge * _m.charge[t]
            outflow = _m.discharge[t] / cfg.eff_discharge
            return _m.soc[t] == soc_prev + inflow - outflow

        m.soc_dynamics = pyo.Constraint(m.T, rule=soc_rule)

        # --- Commitment state transition: u_t - u_{t-1} = su_t - sd_t ---
        def transition_rule(_m: pyo.ConcreteModel, t: int):
            prev = u0 if t == 0 else _m.u[t - 1]
            return _m.u[t] - prev == _m.su[t] - _m.sd[t]

        m.transition = pyo.Constraint(m.T, rule=transition_rule)

        # --- Minimum up time: once started, stay on for min_up_time hours ---
        def min_up_rule(_m: pyo.ConcreteModel, t: int):
            window = range(t, min(t + cfg.min_up_time, horizon))
            return sum(_m.u[k] for k in window) >= len(list(window)) * _m.su[t]

        m.min_up = pyo.Constraint(m.T, rule=min_up_rule)

        # --- Minimum down time: once stopped, stay off for min_down_time hours ---
        def min_down_rule(_m: pyo.ConcreteModel, t: int):
            window = range(t, min(t + cfg.min_down_time, horizon))
            return sum((1 - _m.u[k]) for k in window) >= len(list(window)) * _m.sd[t]

        m.min_down = pyo.Constraint(m.T, rule=min_down_rule)

        # --- Charge / discharge mutual exclusivity ---
        m.charge_excl = pyo.Constraint(
            m.T, rule=lambda _m, t: _m.charge[t] <= big_m * _m.z[t]
        )
        m.discharge_excl = pyo.Constraint(
            m.T, rule=lambda _m, t: _m.discharge[t] <= big_m * (1 - _m.z[t])
        )

        # --- Objective: fuel + carbon + renewable margin + start-up cost ---
        def objective_rule(_m: pyo.ConcreteModel):
            return sum(
                _m.p_coal[t] * cfg.coal_marginal_cost
                + _m.p_wind[t] * cfg.wind_cost
                + _m.p_solar[t] * cfg.solar_cost
                + _m.su[t] * cfg.startup_cost
                for t in _m.T
            )

        m.cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        return m

    def solve(
        self,
        demand: np.ndarray,
        wind_avail: np.ndarray,
        solar_avail: np.ndarray,
    ) -> DispatchResult:
        """Solve the unit-commitment MILP. See :meth:`DispatchEngine.solve`."""
        horizon = self._validate_inputs(demand, wind_avail, solar_avail)
        model = self.build_model(demand, wind_avail, solar_avail)

        solver = pyo.SolverFactory(self.solver_name)
        # Defer solution loading so an infeasible solve does not raise.
        results = solver.solve(model, load_solutions=False)
        # The legacy SolverFactory wrapper nests status under results.solver.
        tc = str(results.solver.termination_condition).lower()
        success = "optimal" in tc

        if not success:
            zeros = np.zeros(horizon)
            return DispatchResult(
                coal=zeros.copy(), wind=zeros.copy(), solar=zeros.copy(),
                charge=zeros.copy(), discharge=zeros.copy(), soc=zeros.copy(),
                total_cost=float("inf"), success=False, solver="milp",
                commitment=zeros.copy(), message=f"solver status: {tc}",
            )

        model.solutions.load_from(results)

        def vec(var: pyo.Var) -> np.ndarray:
            return np.array([pyo.value(var[t]) for t in model.T], dtype=float)

        return DispatchResult(
            coal=vec(model.p_coal),
            wind=vec(model.p_wind),
            solar=vec(model.p_solar),
            charge=vec(model.charge),
            discharge=vec(model.discharge),
            soc=vec(model.soc),
            total_cost=float(pyo.value(model.cost)),
            success=True,
            solver="milp",
            commitment=np.round(vec(model.u)),
            message=tc,
        )
