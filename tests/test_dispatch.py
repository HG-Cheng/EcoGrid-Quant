"""Hand-checkable regressions for the active PV/battery/grid benchmark."""

import numpy as np
import pytest

from ecogrid.dispatch import SystemConfig, solve_dispatch
from ecogrid.simulation import execute_hour


def test_one_hour_accounting_and_no_unpowered_charging():
    cfg = SystemConfig(
        battery_kwh=3,
        battery_kw=3,
        grid_kw=1,
        initial_soc_kwh=0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
    )
    # Audit regression: 1 kW grid must first serve 1 kW load, not charge 3 kW.
    r = execute_hour(cfg, soc=0, load=1, pv=0, price=0.2, requested_charge=3, requested_discharge=0)
    assert r["charge_kw"] == 0
    assert r["soc_kwh"] == 0
    assert r["grid_kw"] == 1
    assert r["ens_kwh"] == 0
    assert r["charge_override_kw"] == 3
    assert r["grid_cost_eur"] == pytest.approx(0.2)
    r = execute_hour(cfg, soc=0, load=2, pv=0, price=0.2, requested_charge=0, requested_discharge=0)
    assert r["ens_kwh"] == 1
    assert r["loss_eur"] == pytest.approx(0.2 + cfg.lost_load_eur_per_kwh)


def test_execution_conserves_energy_and_rejects_invalid_state():
    cfg = SystemConfig(
        battery_kwh=4,
        battery_kw=2,
        grid_kw=1,
        initial_soc_kwh=1,
        charge_efficiency=0.8,
        discharge_efficiency=0.8,
    )
    r = execute_hour(
        cfg, soc=1, load=2, pv=0.5, price=0.2, requested_charge=0, requested_discharge=0.8
    )
    assert r["soc_kwh"] == pytest.approx(0)
    assert r["grid_kw"] == pytest.approx(0.7)
    assert r["balance_residual_kw"] == pytest.approx(0, abs=1e-9)
    with pytest.raises(ValueError):
        execute_hour(cfg, soc=5, load=2, pv=0, price=0.2, requested_charge=0, requested_discharge=0)
    with pytest.raises(ValueError):
        execute_hour(cfg, soc=1, load=2, pv=0, price=0.2, requested_charge=1, requested_discharge=1)


def test_deficit_discharge_respects_power_energy_and_depleted_next_hour():
    cfg = SystemConfig(
        battery_kwh=4,
        battery_kw=1,
        grid_kw=1,
        initial_soc_kwh=2,
        charge_efficiency=0.8,
        discharge_efficiency=0.8,
        throughput_eur_per_kwh=0.1,
        lost_load_eur_per_kwh=10,
        terminal_eur_per_kwh=0.2,
    )
    soc = 2.0
    # Power-limited first; cancel unpowered charging and empty the battery next.
    # The final shortage must reflect the energy spent in the preceding hours.
    cases = [
        (3, 0, 0.25, 1, 0.75, 0.75, 0.75, 8.05),
        (2, 1, 0, 0.6, 0.6, 0, 0.15, 1.91),
        (2, 0, 0, 0, 0, 0, 0.75, 7.7),
    ]
    for load, charge, discharge, actual, emergency, next_soc, ens, loss in cases:
        r = execute_hour(
            cfg,
            soc=soc,
            load=load,
            pv=0.25,
            price=0.2,
            requested_charge=charge,
            requested_discharge=discharge,
            execution_rule="deficit_discharge",
        )
        assert r["execution_rule"] == "deficit_discharge"
        assert r["requested_charge_kw"] == charge
        assert r["requested_discharge_kw"] == discharge
        assert r["charge_kw"] == 0
        assert r["charge_override_kw"] == charge
        assert r["discharge_kw"] == pytest.approx(actual)
        assert r["emergency_discharge_kw"] == pytest.approx(emergency)
        assert r["discharge_override_kw"] == 0
        assert r["soc_kwh"] == pytest.approx(next_soc)
        assert r["soc_kwh"] == soc - r["discharge_kw"] / 0.8
        assert r["soc_kwh"] >= 0
        assert r["grid_kw"] == pytest.approx(1)
        assert r["pv_used_kw"] == pytest.approx(0.25)
        assert r["ens_kwh"] == pytest.approx(ens)
        assert r["throughput_kwh"] == pytest.approx(actual)
        assert r["loss_eur"] == pytest.approx(loss)
        assert r["balance_residual_kw"] == pytest.approx(0, abs=1e-9)
        soc = r["soc_kwh"]


def test_planned_execution_remains_default_and_reports_only_original_curtailment():
    cfg = SystemConfig(battery_kwh=4, battery_kw=1, grid_kw=1, initial_soc_kwh=2)
    args = dict(soc=2, load=3, pv=0.25, price=0.2, requested_charge=1, requested_discharge=0)
    default = execute_hour(cfg, **args)
    explicit = execute_hour(cfg, **args, execution_rule="planned")
    assert default == explicit
    assert default["execution_rule"] == "planned"
    assert default["charge_kw"] == 0
    assert default["discharge_kw"] == 0
    assert default["emergency_discharge_kw"] == 0
    assert default["soc_kwh"] == 2
    assert default["ens_kwh"] == 1.75
    assert default["loss_eur"] == pytest.approx(17.7)
    for rule in ("planned", "deficit_discharge"):
        curtailed = execute_hour(
            cfg, **{**args, "requested_charge": 0, "requested_discharge": 2}, execution_rule=rule
        )
        assert curtailed["discharge_kw"] == 1
        assert curtailed["discharge_override_kw"] == 1
        assert curtailed["emergency_discharge_kw"] == 0
    with pytest.raises(ValueError, match="execution_rule"):
        execute_hour(cfg, **args, execution_rule="unknown")


def test_representable_soc_survives_full_discharge_without_state_clipping():
    cfg = SystemConfig()
    r = execute_hour(
        cfg,
        soc=1.3962225291522277,
        load=4,
        pv=0,
        price=0.2,
        requested_charge=0,
        requested_discharge=3,
    )
    assert r["soc_kwh"] >= 0
    assert r["soc_kwh"] == r["soc_before_kwh"] - r["discharge_kw"] / 0.95
    execute_hour(
        cfg, soc=r["soc_kwh"], load=4, pv=0, price=0.2, requested_charge=0, requested_discharge=0
    )
    cfg = SystemConfig(
        battery_kwh=62.96261726083157,
        battery_kw=1000,
        grid_kw=1000,
        charge_efficiency=0.8513315455629952,
    )
    r = execute_hour(
        cfg,
        soc=24.50091888467629,
        load=1000,
        pv=1000,
        price=0.2,
        requested_charge=1000,
        requested_discharge=0,
    )
    assert r["soc_kwh"] <= cfg.battery_kwh
    assert r["soc_kwh"] == r["soc_before_kwh"] + cfg.charge_efficiency * r["charge_kw"]


def test_shared_first_action_physics_and_independent_risk():
    cfg = SystemConfig(battery_kwh=2, battery_kw=1, grid_kw=1, initial_soc_kwh=1)
    plan = solve_dispatch(
        np.array([2.0, 1.0]),
        np.array([0.2, 0.3]),
        np.array([[0.0, 0.0], [3.0, 3.0]]),
        cfg,
        probabilities=np.array([0.75, 0.25]),
        risk_weight=0,
        alpha=0.5,
    )
    for name in ("charge", "discharge", "soc"):
        assert np.ptp(plan.flows[name][:, 0]) < 1e-7
    f = plan.flows
    assert np.max(f["charge"] * f["discharge"]) < 1e-7
    assert (
        np.max(
            np.abs(
                f["grid"]
                + f["pv_used"]
                + f["discharge"]
                + f["shed"]
                - np.array([2.0, 1.0])
                - f["charge"]
            )
        )
        < 1e-7
    )
    previous = np.column_stack([np.full(2, 1.0), f["soc"][:, :-1]])
    assert np.max(np.abs(f["soc"] - previous - 0.95 * f["charge"] + f["discharge"] / 0.95)) < 1e-7
    # The .75 mass at the larger loss covers the entire .5 upper tail.
    assert plan.cvar_eur == pytest.approx(max(plan.scenario_losses))
    assert plan.max_residual < 1e-6


def test_hand_solved_shortage_and_terminal_inventory_value():
    cfg = SystemConfig(
        battery_kwh=1,
        battery_kw=1,
        grid_kw=1,
        initial_soc_kwh=0,
        throughput_eur_per_kwh=0.01,
        terminal_eur_per_kwh=0.2,
        lost_load_eur_per_kwh=10,
    )
    # No battery energy or PV: 1 kWh imported, 1 kWh not served.
    p = solve_dispatch(np.array([2.0]), np.array([0.3]), np.array([[0.0]]), cfg)
    assert p.expected_loss_eur == pytest.approx(10.3)
    assert p.flows["shed"][0, 0] == pytest.approx(1)
    assert p.flows["charge"][0, 0] == pytest.approx(0)
    # With free PV, storage retains value; no forced end-of-window emptying.
    p = solve_dispatch(np.array([0.0]), np.array([0.3]), np.array([[2.0]]), cfg)
    assert p.flows["soc"][0, 0] == pytest.approx(0.95)
    assert p.expected_loss_eur == pytest.approx(0.01 - 0.2 * 0.95)


def test_perfect_information_bounds_real_feasible_execution():
    cfg = SystemConfig(battery_kwh=2, battery_kw=1, grid_kw=1, initial_soc_kwh=1)
    load = np.array([2.0, 1.0, 2.0])
    pv = np.array([0.0, 3.0, 0.0])
    price = np.array([0.2, 0.2, 0.3])
    soc = cfg.initial_soc_kwh
    loss = 0.0
    for t in range(3):
        r = execute_hour(
            cfg,
            soc=soc,
            load=load[t],
            pv=pv[t],
            price=price[t],
            requested_charge=1 if t == 1 else 0,
            requested_discharge=0.9 if t != 1 else 0,
        )
        soc = r["soc_kwh"]
        loss += r["loss_eur"]
    perfect = solve_dispatch(load, price, pv[None, :], cfg)
    assert perfect.expected_loss_eur <= loss + 1e-6


def test_pure_cvar_tiebreak_minimizes_non_tail_cost_with_bounded_tail():
    cfg = SystemConfig(
        battery_kwh=2,
        battery_kw=1,
        grid_kw=1,
        initial_soc_kwh=1,
        charge_efficiency=1,
        discharge_efficiency=1,
        throughput_eur_per_kwh=0.01,
        terminal_eur_per_kwh=0,
    )
    load = np.array([1.0, 2.0, 1.0])
    price = np.array([0.2, 0.4, 0.1])
    pv = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    raw = solve_dispatch(load, price, pv, cfg, risk_weight=1, alpha=0.5, tail_tiebreak=False)
    tied = solve_dispatch(load, price, pv, cfg, risk_weight=1, alpha=0.5)
    # The dark path requires 3 grid kWh (.70 EUR) and 1 discharge (.01 EUR).
    # The sunny path can cost zero or waste .01 on discharge; either leaves the
    # .5 upper tail at .71. Do not assume which raw optimum a solver returns.
    assert raw.cvar_eur == pytest.approx(0.71)
    assert tied.scenario_losses == pytest.approx([0.71, 0.0], abs=2e-7)
    assert tied.expected_loss_eur == pytest.approx(0.355, abs=2e-7)
    assert tied.primary_expected_loss_eur == pytest.approx(raw.expected_loss_eur)
    assert tied.primary_cvar_eur == pytest.approx(raw.cvar_eur)
    assert tied.expected_loss_eur <= raw.expected_loss_eur + cfg.feasibility_tolerance
    assert tied.tail_tiebreak_used
    assert 0 < tied.tail_tiebreak_tolerance_eur <= 1e-6
    assert tied.cvar_eur <= (
        tied.primary_cvar_eur + tied.tail_tiebreak_tolerance_eur + cfg.feasibility_tolerance
    )
    assert tied.objective_eur == pytest.approx(tied.cvar_eur)
    assert not raw.tail_tiebreak_used
    assert raw.tail_tiebreak_tolerance_eur == 0


@pytest.mark.parametrize("risk_weight", [0.0, 0.5])
def test_tail_tiebreak_leaves_expected_and_mixed_objectives_unchanged(risk_weight):
    cfg = SystemConfig(battery_kwh=2, battery_kw=1, grid_kw=1, initial_soc_kwh=1)
    args = (np.array([2.0, 1.0]), np.array([0.2, 0.3]), np.array([[0.0, 0.0], [3.0, 3.0]]))
    raw = solve_dispatch(*args, cfg, risk_weight=risk_weight, alpha=0.5, tail_tiebreak=False)
    default = solve_dispatch(*args, cfg, risk_weight=risk_weight, alpha=0.5)
    assert not default.tail_tiebreak_used
    assert default.tail_tiebreak_tolerance_eur == 0
    assert default.scenario_losses == pytest.approx(raw.scenario_losses)
    assert default.first_action == pytest.approx(raw.first_action)
    assert default.objective_eur == pytest.approx(
        (1 - risk_weight) * default.expected_loss_eur + risk_weight * default.cvar_eur
    )
