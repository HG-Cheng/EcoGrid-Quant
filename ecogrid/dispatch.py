"""Active hourly PV/battery/import-grid model (kW, kWh, EUR).

Only the first battery action is shared across scenarios. Later scenario paths
are full-information recourse: a two-stage planning approximation, not a
multistage-optimal policy. See docs/MODEL_SPECIFICATION.md.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pyomo.environ as pyo

from ecogrid.finance import cvar, value_at_risk


@dataclass(frozen=True)
class SystemConfig:
    """One-hour intervals; fixed system and cost assumptions across policies."""

    battery_kwh: float = 8.0
    battery_kw: float = 3.0
    grid_kw: float = 3.0
    initial_soc_kwh: float = 4.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    throughput_eur_per_kwh: float = 0.01
    lost_load_eur_per_kwh: float = 10.0
    terminal_eur_per_kwh: float = 0.18
    solver_time_limit: float = 30.0
    feasibility_tolerance: float = 1e-6

    def __post_init__(self):
        vals = np.array(list(vars(self).values()), dtype=float)
        if not np.isfinite(vals).all() or (vals < 0).any():
            raise ValueError("System parameters must be finite and nonnegative")
        if not 0 <= self.initial_soc_kwh <= self.battery_kwh:
            raise ValueError("Initial SOC must be within battery capacity")
        if not (0 < self.charge_efficiency <= 1 and 0 < self.discharge_efficiency <= 1):
            raise ValueError("Battery efficiencies must lie in (0, 1]")
        if self.solver_time_limit <= 0 or self.feasibility_tolerance <= 0:
            raise ValueError("Solver time limit and feasibility tolerance must be positive")


@dataclass
class DispatchPlan:
    """Scenario schedules are diagnostics; only row-zero first action is issued."""

    flows: dict[str, np.ndarray]
    probabilities: np.ndarray
    scenario_losses: np.ndarray
    expected_loss_eur: float
    var_eur: float
    cvar_eur: float
    objective_eur: float
    solve_seconds: float
    max_residual: float
    termination: str
    primary_expected_loss_eur: float | None = None
    primary_cvar_eur: float | None = None
    tail_tiebreak_tolerance_eur: float = 0.0
    tail_tiebreak_used: bool = False

    @property
    def first_action(self) -> tuple[float, float]:
        return float(self.flows["charge"][0, 0]), float(self.flows["discharge"][0, 0])


def solve_dispatch(
    load: np.ndarray,
    price: np.ndarray,
    pv_scenarios: np.ndarray,
    config: SystemConfig | None = None,
    *,
    probabilities: np.ndarray | None = None,
    initial_soc: float | None = None,
    risk_weight: float = 0.0,
    alpha: float = 0.9,
    tail_tiebreak: bool = True,
) -> DispatchPlan:
    """Solve one shared physical model for point, expected-loss or CVaR planning.

    Loss = imports + battery throughput + unserved load - terminal inventory
    change. First charge/discharge/mode are non-anticipative. Supply balancing
    may respond to PV. Forecast errors outside the ensemble are handled by the
    common load-first execution guard, not by these future scenario schedules.
    At risk_weight=1, the optional second solve minimizes expected loss with
    CVaR bounded by the first solve's CVaR plus 1e-7 EUR. This is an explicit
    lexicographic tolerance, not an expected-loss term in the primary target.
    """
    started = perf_counter()
    cfg = config or SystemConfig()
    demand = np.asarray(load, dtype=float)
    tariff = np.asarray(price, dtype=float)
    pv = np.asarray(pv_scenarios, dtype=float)
    if demand.ndim != 1 or demand.size == 0 or tariff.shape != demand.shape:
        raise ValueError("Load and price must be nonempty matching vectors")
    if pv.ndim != 2 or pv.shape[0] == 0 or pv.shape[1] != demand.size:
        raise ValueError("PV scenarios must have shape (scenarios, horizon)")
    if any(not np.isfinite(x).all() or (x < 0).any() for x in (demand, tariff, pv)):
        raise ValueError("Load, prices and PV must be finite and nonnegative")
    if not 0 <= risk_weight <= 1 or not 0 < alpha < 1:
        raise ValueError("Risk weight must be in [0,1] and alpha in (0,1)")
    probs = (
        np.full(len(pv), 1 / len(pv))
        if probabilities is None
        else np.asarray(probabilities, dtype=float)
    )
    # The common statistics function also validates the probability contract.
    value_at_risk(np.zeros(len(pv)), alpha, probs)
    positive = probs > 0
    pv, probs = pv[positive], probs[positive]
    probs = probs / probs.sum()  # only floating-point sum drift after validation
    soc0 = cfg.initial_soc_kwh if initial_soc is None else float(initial_soc)
    if not np.isfinite(soc0) or not 0 <= soc0 <= cfg.battery_kwh:
        raise ValueError("Initial SOC must be finite and within capacity")
    n, horizon = pv.shape
    m = pyo.ConcreteModel("PV-battery-grid")
    m.S = pyo.RangeSet(0, n - 1)
    m.T = pyo.RangeSet(0, horizon - 1)
    m.grid = pyo.Var(m.S, m.T, bounds=(0, cfg.grid_kw))
    m.pv_used = pyo.Var(m.S, m.T, bounds=lambda _, s, t: (0, pv[s, t]))
    m.shed = pyo.Var(m.S, m.T, bounds=lambda _, s, t: (0, demand[t]))
    m.charge = pyo.Var(m.S, m.T, bounds=(0, cfg.battery_kw))
    m.discharge = pyo.Var(m.S, m.T, bounds=(0, cfg.battery_kw))
    m.soc = pyo.Var(m.S, m.T, bounds=(0, cfg.battery_kwh))
    m.mode = pyo.Var(m.S, m.T, domain=pyo.Binary)
    m.balance = pyo.Constraint(
        m.S,
        m.T,
        rule=lambda x, s, t: (
            x.grid[s, t] + x.pv_used[s, t] + x.discharge[s, t] + x.shed[s, t]
            == demand[t] + x.charge[s, t]
        ),
    )
    m.energy = pyo.Constraint(
        m.S,
        m.T,
        rule=lambda x, s, t: (
            x.soc[s, t]
            == (soc0 if t == 0 else x.soc[s, t - 1])
            + cfg.charge_efficiency * x.charge[s, t]
            - x.discharge[s, t] / cfg.discharge_efficiency
        ),
    )
    m.charge_mode = pyo.Constraint(
        m.S, m.T, rule=lambda x, s, t: x.charge[s, t] <= cfg.battery_kw * x.mode[s, t]
    )
    m.discharge_mode = pyo.Constraint(
        m.S, m.T, rule=lambda x, s, t: x.discharge[s, t] <= cfg.battery_kw * (1 - x.mode[s, t])
    )
    m.shared_action = pyo.ConstraintList()
    for s in range(1, n):
        for name in ("charge", "discharge", "mode"):
            v = getattr(m, name)
            m.shared_action.add(v[s, 0] == v[0, 0])

    def loss_rule(x, s):
        return sum(
            tariff[t] * x.grid[s, t]
            + cfg.throughput_eur_per_kwh * (x.charge[s, t] + x.discharge[s, t])
            + cfg.lost_load_eur_per_kwh * x.shed[s, t]
            for t in x.T
        ) - cfg.terminal_eur_per_kwh * (x.soc[s, horizon - 1] - soc0)

    m.loss = pyo.Expression(m.S, rule=loss_rule)
    expected = sum(probs[s] * m.loss[s] for s in m.S)
    if risk_weight > 0:
        m.eta = pyo.Var(domain=pyo.Reals)
        m.excess = pyo.Var(m.S, domain=pyo.NonNegativeReals)
        m.tail = pyo.Constraint(m.S, rule=lambda x, s: x.excess[s] >= x.loss[s] - x.eta)
        tail = m.eta + sum(probs[s] * m.excess[s] for s in m.S) / (1 - alpha)
        objective = (1 - risk_weight) * expected + risk_weight * tail
    else:
        objective = expected
    m.objective = pyo.Objective(expr=objective, sense=pyo.minimize)
    solver = pyo.SolverFactory("appsi_highs")
    # Leave thread count at HiGHS' default: its scheduler is process-global and
    # may already have been initialized by another Pyomo model in this process.
    solver.options.update(
        {"time_limit": cfg.solver_time_limit, "mip_rel_gap": 1e-8, "random_seed": 0}
    )

    def solve_and_validate(stage: str) -> tuple[str, float]:
        result = solver.solve(m, load_solutions=False)
        termination = str(result.solver.termination_condition).lower()
        if termination != "optimal":
            raise RuntimeError(f"Dispatch rejected ({stage}): solver termination={termination}")
        m.solutions.load_from(result)
        residual = 0.0
        for constraint in m.component_data_objects(pyo.Constraint, active=True):
            body = pyo.value(constraint.body)
            if not np.isfinite(body):
                raise RuntimeError(f"Dispatch rejected ({stage}): nonfinite constraint")
            if constraint.has_lb():
                residual = max(residual, pyo.value(constraint.lower) - body)
            if constraint.has_ub():
                residual = max(residual, body - pyo.value(constraint.upper))
        for variable in m.component_data_objects(pyo.Var):
            value = pyo.value(variable)
            if not np.isfinite(value):
                raise RuntimeError(f"Dispatch rejected ({stage}): nonfinite variable")
            if variable.has_lb():
                residual = max(residual, variable.lb - value)
            if variable.has_ub():
                residual = max(residual, value - variable.ub)
            if variable.is_binary():
                residual = max(residual, abs(value - round(value)))
        if residual > cfg.feasibility_tolerance:
            raise RuntimeError(f"Dispatch rejected ({stage}): constraint residual={residual:g}")
        return termination, residual

    termination, max_residual = solve_and_validate("primary")
    primary_losses = np.array([pyo.value(m.loss[s]) for s in m.S])
    primary_expected = float(probs @ primary_losses)
    primary_cvar = cvar(primary_losses, alpha, probs)
    tie_used = tail_tiebreak and risk_weight == 1.0
    tie_tolerance = 1e-7 if tie_used else 0.0
    if tie_used:
        m.tail_bound = pyo.Constraint(expr=tail <= primary_cvar + tie_tolerance)
        m.objective.set_value(expected)
        termination, secondary_residual = solve_and_validate("tail tiebreak")
        max_residual = max(max_residual, secondary_residual)
    names = ("grid", "pv_used", "shed", "charge", "discharge", "soc")
    flows = {
        name: np.array([[pyo.value(getattr(m, name)[s, t]) for t in m.T] for s in m.S])
        for name in names
    }
    flows["curtailment"] = pv - flows["pv_used"]
    losses = np.array([pyo.value(m.loss[s]) for s in m.S])
    expected_loss = float(probs @ losses)
    actual_cvar = cvar(losses, alpha, probs)
    if tie_used:
        if actual_cvar > primary_cvar + tie_tolerance + cfg.feasibility_tolerance:
            raise RuntimeError("Dispatch rejected (tail tiebreak): recomputed CVaR exceeds bound")
        if expected_loss > primary_expected + cfg.feasibility_tolerance:
            raise RuntimeError("Dispatch rejected (tail tiebreak): expected loss increased")
    return DispatchPlan(
        flows=flows,
        probabilities=probs,
        scenario_losses=losses,
        expected_loss_eur=expected_loss,
        var_eur=value_at_risk(losses, alpha, probs),
        cvar_eur=actual_cvar,
        objective_eur=(1 - risk_weight) * expected_loss + risk_weight * actual_cvar,
        solve_seconds=perf_counter() - started,
        max_residual=max_residual,
        termination=termination,
        primary_expected_loss_eur=primary_expected,
        primary_cvar_eur=primary_cvar,
        tail_tiebreak_tolerance_eur=tie_tolerance,
        tail_tiebreak_used=tie_used,
    )
