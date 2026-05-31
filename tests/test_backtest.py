"""Unit tests for the parallel grid scan and rolling-horizon back-test."""

from __future__ import annotations

import numpy as np

from ecogrid.backtest import rolling_backtest, scan_capacities
from ecogrid.config import GridConfig


def _profile(h: int = 12):
    hours = np.arange(h)
    demand = 90 + 30 * np.exp(-((hours - 9) ** 2) / 6)
    wind = np.clip(30 + 12 * np.cos(hours / h * 2 * np.pi), 0, 50)
    solar = np.clip(45 * np.exp(-((hours - 6) ** 2) / 6), 0, 50)
    return demand, wind, solar


def test_scan_capacities_returns_sorted_frame():
    demand, wind, solar = _profile()
    cfg = GridConfig(carbon_tax_rate=80.0, coal_pmin=0.0, coal_pmax=500.0)
    caps = np.array([0.0, 40.0, 80.0])
    df = scan_capacities(demand, wind, solar, caps, cfg, n_jobs=2)
    assert list(df["capacity"]) == [0.0, 40.0, 80.0]
    assert df["success"].all()
    # More storage cannot increase the optimal operating cost.
    assert df["total_cost"].iloc[-1] <= df["total_cost"].iloc[0] + 1e-6


def test_rolling_backtest_regret_non_negative_under_forecast_error():
    demand, wind, solar = _profile(18)
    # Forecast systematically over-predicts renewables -> real coal higher.
    wind_fc = wind * 1.15
    solar_fc = solar * 1.15
    cfg = GridConfig(carbon_tax_rate=90.0, bess_capacity=40.0, coal_pmin=0.0,
                     coal_pmax=500.0, startup_cost=500.0)
    res = rolling_backtest(demand, wind, solar, wind_fc, solar_fc, cfg,
                           lookahead=6, step=1)
    assert np.isfinite(res.realized_cost)
    assert np.isfinite(res.perfect_cost)
    # Imperfect foresight can never meaningfully beat perfect foresight
    # (allow a tiny relative tolerance for solver/float noise).
    assert res.regret >= -1e-3 * max(abs(res.perfect_cost), 1.0)


def test_rolling_backtest_matches_perfect_when_forecast_is_perfect():
    demand, wind, solar = _profile(12)
    cfg = GridConfig(carbon_tax_rate=90.0, bess_capacity=0.0, coal_pmin=0.0,
                     coal_pmax=500.0, startup_cost=0.0)
    # Forecast == actual, full lookahead -> realised should equal perfect.
    res = rolling_backtest(demand, wind, solar, wind, solar, cfg,
                           lookahead=12, step=1)
    assert abs(res.regret) <= 1e-3 * max(abs(res.perfect_cost), 1.0)
