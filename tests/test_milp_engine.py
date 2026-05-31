"""Unit tests for the MILP unit-commitment engine.

These tests pin down the *physical correctness* of the engine on extreme and
boundary conditions: power balance, state-of-charge conservation, the thermal
minimum-output floor, charge/discharge exclusivity, infeasibility detection and
the economic effect of start-up cost.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecogrid.config import GridConfig
from ecogrid.engines import MilpDispatchEngine

TOL = 1e-4


def _balance_residual(result, demand: np.ndarray) -> np.ndarray:
    supply = result.coal + result.wind + result.solar + result.discharge
    return supply - (demand + result.charge)


def test_power_balance_holds(base_config, simple_profile):
    demand, wind, solar = simple_profile
    res = MilpDispatchEngine(base_config).solve(demand, wind, solar)
    assert res.success
    assert np.max(np.abs(_balance_residual(res, demand))) < TOL


def test_soc_conservation(base_config, simple_profile):
    demand, wind, solar = simple_profile
    cfg = base_config
    res = MilpDispatchEngine(cfg).solve(demand, wind, solar)
    for t in range(res.horizon):
        prev = cfg.initial_soc if t == 0 else res.soc[t - 1]
        expected = prev + cfg.eff_charge * res.charge[t] - res.discharge[t] / cfg.eff_discharge
        assert res.soc[t] == pytest.approx(expected, abs=TOL)


def test_thermal_minimum_respected(base_config, simple_profile):
    demand, wind, solar = simple_profile
    res = MilpDispatchEngine(base_config).solve(demand, wind, solar)
    on = res.commitment > 0.5
    if on.any():
        assert res.coal[on].min() >= base_config.coal_pmin - TOL
    # When the unit is off, its output must be exactly zero.
    off = ~on
    if off.any():
        assert np.max(res.coal[off]) < TOL


def test_no_simultaneous_charge_and_discharge(base_config, simple_profile):
    demand, wind, solar = simple_profile
    res = MilpDispatchEngine(base_config).solve(demand, wind, solar)
    simultaneous = (res.charge > TOL) & (res.discharge > TOL)
    assert not simultaneous.any()


def test_zero_capacity_battery_is_idle(base_config, simple_profile):
    """With cap=0 the battery cannot store energy: SoC and flows are zero."""
    demand, wind, solar = simple_profile
    cfg = GridConfig(carbon_tax_rate=80.0, bess_capacity=0.0, coal_pmin=0.0)
    res = MilpDispatchEngine(cfg).solve(demand, wind, solar)
    assert res.success
    assert np.max(res.soc) < TOL
    assert np.max(res.charge) < TOL
    assert np.max(res.discharge) < TOL
    assert np.max(np.abs(_balance_residual(res, demand))) < TOL


def test_zero_demand_is_zero_cost(small_horizon):
    """No demand and no renewables -> nothing runs, zero cost."""
    h = small_horizon
    zeros = np.zeros(h)
    cfg = GridConfig(carbon_tax_rate=80.0, bess_capacity=50.0, coal_pmin=20.0)
    res = MilpDispatchEngine(cfg).solve(zeros, zeros, zeros)
    assert res.success
    assert res.total_cost == pytest.approx(0.0, abs=TOL)
    assert np.max(res.commitment) < TOL


def test_infeasible_when_capacity_too_small(small_horizon):
    """Demand exceeds the only dispatchable resource -> infeasible."""
    h = small_horizon
    demand = np.full(h, 500.0)
    zeros = np.zeros(h)
    cfg = GridConfig(carbon_tax_rate=80.0, bess_capacity=0.0, coal_pmax=100.0)
    res = MilpDispatchEngine(cfg).solve(demand, zeros, zeros)
    assert not res.success
    assert res.total_cost == float("inf")


def test_startup_cost_increases_total_cost(simple_profile):
    """Adding a start-up cost can only raise (never lower) the optimum."""
    demand, wind, solar = simple_profile
    cheap = GridConfig(carbon_tax_rate=80.0, bess_capacity=0.0, startup_cost=0.0)
    pricey = GridConfig(carbon_tax_rate=80.0, bess_capacity=0.0, startup_cost=5000.0)
    c0 = MilpDispatchEngine(cheap).solve(demand, wind, solar)
    c1 = MilpDispatchEngine(pricey).solve(demand, wind, solar)
    assert c0.success and c1.success
    assert c1.total_cost >= c0.total_cost - TOL


def test_mismatched_input_lengths_raise():
    cfg = GridConfig()
    with pytest.raises(ValueError):
        MilpDispatchEngine(cfg).solve(
            np.zeros(4), np.zeros(3), np.zeros(4)
        )
