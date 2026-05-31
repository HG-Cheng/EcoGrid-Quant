"""Unit tests for the two-stage stochastic / robust engine."""

from __future__ import annotations

import numpy as np

from ecogrid.config import GridConfig, ScenarioConfig
from ecogrid.engines import StochasticDispatchEngine
from ecogrid.scenarios import generate_scenarios


def _setup(h: int = 8, n: int = 30):
    hours = np.arange(h)
    demand = np.full(h, 110.0)
    wind = np.clip(30 + 12 * np.cos(hours / h * 2 * np.pi), 0, 50)
    solar = np.clip(45 * np.exp(-((hours - h / 2) ** 2) / 6), 0, 50)
    scen = generate_scenarios(wind, solar, ScenarioConfig(n_scenarios=n, seed=11))
    cfg = GridConfig(
        carbon_tax_rate=100.0, bess_capacity=60.0, coal_pmin=20.0,
        coal_pmax=500.0, startup_cost=1500.0, min_up_time=2, min_down_time=2,
    )
    return demand, scen, cfg


def test_stochastic_solves_and_is_consistent():
    demand, scen, cfg = _setup()
    res = StochasticDispatchEngine(cfg).solve(demand, scen)
    assert res.success
    assert np.isfinite(res.expected_cost)
    assert res.scenario_costs.shape == (scen.n_scenarios,)
    # Expected cost equals the probability-weighted mean of scenario costs.
    assert res.expected_cost == np.float64(
        np.dot(scen.probabilities, res.scenario_costs)
    ) or abs(res.expected_cost - scen.probabilities @ res.scenario_costs) < 1e-3


def test_cvar_not_below_var():
    demand, scen, cfg = _setup()
    res = StochasticDispatchEngine(cfg, risk_lambda=0.5).solve(demand, scen)
    assert res.cvar >= res.var - 1e-3


def test_risk_aversion_does_not_lower_tail_cost():
    """A CVaR-aware schedule should not have a worse tail than risk-neutral."""
    demand, scen, cfg = _setup()
    neutral = StochasticDispatchEngine(cfg, risk_lambda=0.0).solve(demand, scen)
    averse = StochasticDispatchEngine(cfg, risk_lambda=0.9).solve(demand, scen)
    assert neutral.success and averse.success
    # Risk-averse worst-case cost is no greater than the risk-neutral one.
    assert averse.scenario_costs.max() <= neutral.scenario_costs.max() + 1e-3
