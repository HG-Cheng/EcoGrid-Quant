"""Evaluation contracts: causal forecasts, feasible replay, and cost reporting."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecogrid import evaluation
from ecogrid.dispatch import SystemConfig
from ecogrid.evaluation import apply_stress, daily_metrics, perfect_foresight, run_policy, summarize


class LastObservedForecaster:
    pv_capacity_kw = 5.0

    def predict(self, history, valid_index):
        assert history.index.max() < valid_index[0]
        return np.full(len(valid_index), history.pv_kw.iloc[-1])


class ZeroResidualLibrary:
    def sample_scenarios(self, point_forecast, issue_time, n_scenarios, seed):
        return np.tile(point_forecast, (n_scenarios, 1))


def test_emergency_state_reaches_next_solve_and_all_matching_keeps_equal_mass(monkeypatch):
    class ThreeSourceLibrary:
        def matching_candidate_count(self, issue):
            return 3

        def sample_scenarios(self, point, issue, n, seed, *, mode="sampled"):
            assert mode == "all_matching"
            return np.tile(np.array([0.0, 0.0, 1.0])[:, None], (1, 24))

    states = []

    def no_requested_action(load, price, scenarios, system, **kwargs):
        states.append(kwargs["initial_soc"])
        np.testing.assert_allclose(kwargs["probabilities"], [1 / 3] * 3)
        assert len(scenarios) == 3  # Duplicate sources are not silently deduplicated.
        return SimpleNamespace(
            first_action=(0, 0),
            expected_loss_eur=0,
            var_eur=0,
            cvar_eur=0,
            objective_eur=0,
            solve_seconds=0,
            max_residual=0,
        )

    monkeypatch.setattr(evaluation, "solve_dispatch", no_requested_action)
    frame = frame_with_tail()
    frame["pv_kw"] = 0.0
    frame["load_kw"] = 2.0
    cfg = SystemConfig(
        battery_kwh=2, battery_kw=2, grid_kw=1, initial_soc_kwh=1, discharge_efficiency=0.8
    )
    result = run_policy(
        frame,
        LastObservedForecaster(),
        ThreeSourceLibrary(),
        frame.index[1],
        2,
        cfg,
        policy="stochastic",
        scenario_mode="all_matching",
        execution_rule="deficit_discharge",
    )
    assert states == pytest.approx([1, 0])
    assert result.emergency_discharge_kw.tolist() == pytest.approx([0.8, 0])
    assert result.ens_kwh.tolist() == pytest.approx([0.2, 1])
    assert result.scenario_candidate_count.tolist() == [3, 3]
    assert result.scenario_tail_equivalent_count.tolist() == pytest.approx([0.3, 0.3])


def frame_with_tail():
    index = pd.date_range("2024-01-01", periods=27, freq="h", tz="UTC", name="time")
    pv = np.zeros(len(index))
    pv[0] = 1.0
    pv[1] = 4.0
    pv[2] = 3.0
    return pd.DataFrame(
        {
            "pv_kw": pv,
            "load_kw": np.ones(len(index)),
            "price_eur_per_kwh": np.full(len(index), 0.2),
        },
        index=index,
    )


def test_run_policy_forecast_uses_only_prior_pv_and_replays_physical_state():
    frame = frame_with_tail()
    cfg = SystemConfig(battery_kwh=2, battery_kw=1, grid_kw=1, initial_soc_kwh=1)
    hourly = run_policy(
        frame,
        LastObservedForecaster(),
        ZeroResidualLibrary(),
        frame.index[1],
        2,
        cfg,
        policy="deterministic",
    )
    assert hourly.index.equals(frame.index[1:3])
    assert hourly.point_forecast_pv_kw.tolist() == [1.0, 4.0]
    assert hourly.pv_kw.tolist() == [4.0, 3.0]
    assert hourly.forecast_issue_time.tolist() == list(hourly.index)
    assert hourly.forecast_valid_time.tolist() == list(hourly.index)
    assert hourly.forecast_lead_hours.tolist() == [0, 0]
    assert hourly.soc_before_kwh.iloc[1] == pytest.approx(hourly.soc_kwh.iloc[0])
    assert np.max(np.abs(hourly.balance_residual_kw)) < 1e-7
    assert hourly.loss_eur.sum() == pytest.approx(
        hourly.grid_cost_eur.sum()
        + hourly.battery_cost_eur.sum()
        + hourly.shortage_cost_eur.sum()
        + hourly.inventory_adjustment_eur.sum()
    )


def test_stress_scales_future_actuals_and_adds_fixed_forecast_optimism():
    frame = frame_with_tail()
    original = frame.copy(deep=True)
    stressed = apply_stress(frame, frame.index[1])
    assert stressed.pv_kw.iloc[:3].tolist() == pytest.approx([1.0, 0.8, 0.6])
    pd.testing.assert_frame_equal(frame, original)
    cfg = SystemConfig(battery_kwh=2, battery_kw=1, grid_kw=1, initial_soc_kwh=1)
    hourly = run_policy(
        frame, LastObservedForecaster(), ZeroResidualLibrary(), frame.index[1], 2, cfg, stress=True
    )
    assert hourly.pv_kw.tolist() == pytest.approx([0.8, 0.6])
    assert hourly.point_forecast_pv_kw.tolist() == pytest.approx([1.75, 1.55])
    perfect = perfect_foresight(stressed, frame.index[1], 2, cfg)
    assert perfect.stress.tolist() == [True, True]


def test_perfect_foresight_full_period_is_lower_bound_on_policy():
    frame = frame_with_tail()
    cfg = SystemConfig(battery_kwh=2, battery_kw=1, grid_kw=1, initial_soc_kwh=1)
    start = frame.index[1]
    policy = run_policy(frame, LastObservedForecaster(), ZeroResidualLibrary(), start, 2, cfg)
    perfect = perfect_foresight(frame, start, 2, cfg)
    assert perfect.index.equals(policy.index)
    assert perfect.loss_eur.sum() <= policy.loss_eur.sum() + 1e-6
    assert perfect.soc_before_kwh.iloc[1] == pytest.approx(perfect.soc_kwh.iloc[0])
    assert perfect.loss_eur.sum() == pytest.approx(perfect.plan_expected_loss_eur.iloc[0], abs=1e-6)
    assert np.max(np.abs(perfect.balance_residual_kw)) < 1e-7


def test_summary_uses_empirical_hourly_tail_and_includes_inventory():
    index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    hourly = pd.DataFrame(
        {
            "loss_eur": [1.0, 2.0, 3.0, 4.0],
            "grid_cost_eur": [1.0, 1.0, 1.0, 1.0],
            "battery_cost_eur": [0.0, 0.0, 0.0, 0.0],
            "shortage_cost_eur": [0.0, 0.0, 0.0, 0.0],
            "inventory_adjustment_eur": [0.0, 1.0, 2.0, 3.0],
            "ens_kwh": [0.0, 1.0, 0.0, 2.0],
            "grid_kw": [1.0, 1.0, 1.0, 1.0],
            "curtailment_kwh": [0.0, 0.0, 0.0, 0.0],
            "throughput_kwh": [0.0, 0.0, 0.0, 0.0],
            "solve_seconds": [0.1, 0.1, 0.1, 0.1],
            "max_residual": [0.0, 0.0, 0.0, 0.0],
            "charge_override_kw": [0.0, 0.0, 0.0, 0.0],
            "discharge_override_kw": [0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    out = summarize(hourly, alpha=0.5, perfect_total=8.0)
    assert out["total_loss_eur"] == pytest.approx(10.0)
    assert out["total_operating_cost_eur"] == pytest.approx(4.0)
    assert out["total_actual_expense_eur"] == pytest.approx(4.0)
    assert out["total_inventory_adjustment_eur"] == pytest.approx(6.0)
    assert out["var_hourly_loss_eur"] == pytest.approx(2.0)
    assert out["cvar_hourly_loss_eur"] == pytest.approx(3.5)
    assert out["upper_tail_equivalent_hours"] == pytest.approx(2.0)
    assert out["eens_kwh_per_hour"] == pytest.approx(0.75)
    assert out["lolp"] == pytest.approx(0.5)
    assert out["regret_eur"] == pytest.approx(2.0)
    daily = daily_metrics(hourly)
    assert daily.iloc[0].total_loss_eur == pytest.approx(10.0)


def risk_ledger():
    """Two complete UTC days and one deliberately expensive partial hour."""
    index = pd.date_range("2024-01-01", periods=49, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "grid_cost_eur": [1.0] * 24 + [3.0] * 24 + [100.0],
            "battery_cost_eur": [0.5] * 49,
            "shortage_cost_eur": [2.0] * 49,
            "inventory_adjustment_eur": [-0.25] * 49,
            "loss_eur": [3.25] * 24 + [5.25] * 24 + [102.25],
            "ens_kwh": [0.0] * 24 + [1.0] * 24 + [10.0],
            "grid_kw": [1.0] * 49,
        },
        index=index,
    )


def test_risk_summary_keeps_expense_penalty_and_complete_block_populations_separate():
    hourly = risk_ledger()
    blocks = evaluation.nonoverlapping_24h_metrics(hourly)
    assert blocks.hours.tolist() == [24, 24, 1]
    assert blocks.complete_block.tolist() == [True, True, False]
    assert blocks.total_actual_expense_eur.tolist() == pytest.approx([36, 84, 100.5])
    risk = evaluation.risk_summary(hourly, alpha=0.5).set_index(["scale", "metric"])
    expense = risk.loc[("utc_day", "actual_expense")]
    assert expense["unit"] == "EUR/block"
    assert expense["sample_count"] == 2
    assert expense["tail_equivalent_count"] == pytest.approx(1)
    assert expense["excluded_partial_blocks"] == 1
    assert expense["mean"] == pytest.approx(60)
    assert expense["var"] == pytest.approx(36)
    assert expense["cvar"] == pytest.approx(84)
    assert risk.loc[("utc_day", "adjusted_loss"), "cvar"] == pytest.approx(126)
    assert risk.loc[("utc_day", "shortage_penalty"), "mean"] == pytest.approx(48)
    assert risk.loc[("utc_day", "inventory_adjustment"), "mean"] == pytest.approx(-6)
    assert risk.loc[("nonoverlapping_24h", "ens"), "cvar"] == pytest.approx(24)
    assert risk.loc[("hour", "actual_expense"), "sample_count"] == 49
    assert risk.loc[("hour", "actual_expense"), "unit"] == "EUR/hour"
    shifted = evaluation.risk_summary(hourly.iloc[1:], alpha=0.5).set_index(["scale", "metric"])
    assert shifted.loc[("utc_day", "actual_expense"), "sample_count"] == 1
    assert shifted.loc[("nonoverlapping_24h", "actual_expense"), "sample_count"] == 2


def test_block_risk_rejects_compressed_gaps_and_excludes_partial_utc_days():
    hourly = risk_ledger()
    gapped = hourly.drop(hourly.index[10])
    daily = daily_metrics(gapped)
    assert daily.complete_block.tolist() == [False, True, False]
    pooled = evaluation.risk_summary_blocks(daily, alpha=0.9, scale="utc_day")
    expense = pooled.loc[pooled.metric == "actual_expense"].iloc[0]
    assert expense.sample_count == 1
    assert expense.excluded_partial_blocks == 2
    assert expense["mean"] == pytest.approx(84)
    with pytest.raises(ValueError, match="contiguous"):
        evaluation.nonoverlapping_24h_metrics(gapped)
    with pytest.raises(ValueError, match="contiguous"):
        evaluation.risk_summary(gapped)


def test_all_lead_forecast_diagnostics_do_not_change_preissued_actions():
    class FourLevelLibrary:
        def sample_scenarios(self, point_forecast, issue_time, n_scenarios, seed):
            return np.tile(np.arange(4)[:, None], (1, len(point_forecast)))

    frame = frame_with_tail()
    changed = frame.copy()
    changed.loc[changed.index >= frame.index[1], "pv_kw"] = 0.0
    cfg = SystemConfig(battery_kwh=2, battery_kw=1, grid_kw=1, initial_soc_kwh=1)
    diagnostic_rows = []
    original = run_policy(
        frame,
        LastObservedForecaster(),
        FourLevelLibrary(),
        frame.index[1],
        1,
        cfg,
        policy="stochastic",
        n_scenarios=4,
        collect_diagnostics=diagnostic_rows,
    )
    other = run_policy(
        changed,
        LastObservedForecaster(),
        FourLevelLibrary(),
        frame.index[1],
        1,
        cfg,
        policy="stochastic",
        n_scenarios=4,
    )
    assert original.requested_charge_kw.iloc[0] == pytest.approx(other.requested_charge_kw.iloc[0])
    assert original.requested_discharge_kw.iloc[0] == pytest.approx(
        other.requested_discharge_kw.iloc[0]
    )
    assert len(diagnostic_rows) == 24
    assert [row["forecast_lead_hours"] for row in diagnostic_rows] == list(range(24))
    assert diagnostic_rows[0]["actual_pv_kw"] == pytest.approx(4.0)
    assert diagnostic_rows[1]["actual_pv_kw"] == pytest.approx(3.0)
    assert diagnostic_rows[0]["point_forecast_pv_kw"] == pytest.approx(1.0)
    assert diagnostic_rows[0]["forecast_error_pv_kw"] == pytest.approx(-3.0)
    assert diagnostic_rows[0]["scenario_p10_pv_kw"] == pytest.approx(0.0)
    assert diagnostic_rows[0]["scenario_p90_pv_kw"] == pytest.approx(3.0)
    assert original.actual_expense_eur.iloc[0] == pytest.approx(
        original.grid_cost_eur.iloc[0] + original.battery_cost_eur.iloc[0]
    )
