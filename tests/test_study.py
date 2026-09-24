"""Small interpretation checks, separate from model optimization tests."""

import numpy as np
import pandas as pd
import pytest

from ecogrid import diagnostics
from ecogrid.diagnostics import compare_actions, tail_contributions


def test_protection_collapse_and_tail_cost_attribution():
    index = pd.date_range("2017-01-01", periods=2, freq="h", tz="UTC")
    rows = []
    for policy, request, discharge in (
        ("stochastic", [3, 0], [0, 1]),
        ("cvar_0.5", [1, 0], [0, 0.5]),
    ):
        for i, time in enumerate(index):
            rows.append(
                dict(
                    experiment="main",
                    window="winter",
                    grid_limit_kw=2.5,
                    seed=42,
                    policy=policy,
                    time=time,
                    requested_charge_kw=request[i],
                    requested_discharge_kw=discharge[i],
                    charge_kw=0,
                    discharge_kw=discharge[i],
                    loss_eur=1,
                    ens_kwh=0,
                )
            )
    out = compare_actions(pd.DataFrame(rows)).iloc[0]
    assert out.request_different_hours == 2
    assert out.execution_different_hours == 1
    assert out.protection_collapsed_hours == 1
    ledger = pd.DataFrame(
        dict(
            grid_cost_eur=[1, 2],
            battery_cost_eur=[0, 1],
            shortage_cost_eur=[0, 10],
            inventory_adjustment_eur=[0, -1],
            loss_eur=[1, 12],
        )
    )
    parts = tail_contributions(ledger, alpha=0.25)
    assert parts["tail_shortage_cost_eur"] == pytest.approx(10 / 1.5)
    assert sum(parts.values()) == pytest.approx((12 + 0.5) / 1.5)
    assert np.isfinite(list(parts.values())).all()


def test_finite_member_references_and_summary_keep_distinct_conditions():
    members = np.arange(12, dtype=float)
    outcomes = [diagnostics.ensemble_diagnostics(members, actual) for actual in members - 0.5]
    outcomes.append(diagnostics.ensemble_diagnostics(members, 11.5))
    first = outcomes[0]
    assert first["scenario_member_count"] == first["scenario_unique_member_count"] == 12
    assert first["scenario_p10_pv_kw"] == 1
    assert first["scenario_p90_pv_kw"] == 10
    assert first["ideal_below_ensemble_min_fraction"] == pytest.approx(1 / 13)
    assert first["ideal_p10_p90_coverage"] == pytest.approx(9 / 13)
    assert first["ideal_ensemble_range_coverage"] == pytest.approx(11 / 13)
    assert sum(row["below_ensemble_min"] for row in outcomes) == 1
    assert sum(row["above_ensemble_max"] for row in outcomes) == 1
    assert sum(row["inside_p10_p90_strict"] for row in outcomes) == 9
    middle = outcomes[6]
    assert middle["observation_rank_lower"] == middle["observation_rank_upper"] == 7
    assert middle["observation_pit_lower"] == pytest.approx(6 / 13)
    assert middle["observation_pit_upper"] == pytest.approx(7 / 13)

    rows = pd.DataFrame(
        [
            {
                **first,
                "experiment": "main",
                "window": "winter",
                "grid_limit_kw": 2.5,
                "seed": 42,
                "forecast_lead_hours": 0,
                "forecast_daylight_proxy": True,
                "forecast_error_pv_kw": 1,
                "actual_pv_kw": -0.5,
                "scenario_mode": mode,
                "execution_rule": rule,
            }
            for mode in ("raw", "centered")
            for rule in ("protective", "requested")
        ]
    )
    summary = diagnostics.forecast_summary(rows)
    assert len(summary) == 8
    assert summary.sample_count.eq(1).all()
    assert summary.ideal_p10_p90_coverage.eq(9 / 13).all()
    assert summary.observation_tie_fraction.eq(0).all()
    legacy = rows.drop(columns=["scenario_mode", "execution_rule", *first.keys()]).assign(
        inside_p10_p90=False,
        inside_min_max=False,
        scenario_min_pv_kw=0,
        scenario_p10_pv_kw=1,
        scenario_p90_pv_kw=10,
    )
    assert diagnostics.forecast_summary(legacy).sample_count.eq(4).all()


def test_all_tied_members_keep_full_rank_interval_and_unwidened_coverage():
    result = diagnostics.ensemble_diagnostics(np.zeros(12), 0)
    assert result["scenario_member_count"] == result["observation_tied_member_count"] == 12
    assert result["scenario_unique_member_count"] == 1
    assert result["tie_tolerance_kw"] == 1e-9
    assert result["observation_rank_lower"] == 1
    assert result["observation_rank_upper"] == 13
    assert result["observation_rank_midpoint"] == 7
    assert result["observation_pit_lower"] == 0
    assert result["observation_pit_upper"] == 1
    assert result["observation_pit_midpoint"] == 0.5
    for interval in ("min_max", "p10_p90"):
        assert result[f"inside_{interval}"]
        assert not result[f"inside_{interval}_strict"]
    for bound in ("ensemble_min", "ensemble_max", "p10", "p90"):
        assert result[f"equal_{bound}"]
    assert not result["below_ensemble_min"]
    assert not result["above_ensemble_max"]
    assert result["scenario_p90_pv_kw"] - result["scenario_p10_pv_kw"] == 0
    near = diagnostics.ensemble_diagnostics(np.zeros(12), 0.5e-9)
    assert near["observation_tied_member_count"] == 12
    assert not near["inside_p10_p90"]
    assert not near["above_p90"]
    outside = diagnostics.ensemble_diagnostics(np.zeros(12), 2e-9)
    assert outside["above_p90"]
    assert outside["observation_tied_member_count"] == 0
