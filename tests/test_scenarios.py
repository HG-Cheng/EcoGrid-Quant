"""Unit tests for Monte-Carlo scenario generation."""

from __future__ import annotations

import numpy as np

from ecogrid.config import PlantConfig, ScenarioConfig
from ecogrid.scenarios import generate_scenarios


def _forecasts(h: int = 24) -> tuple[np.ndarray, np.ndarray]:
    hours = np.arange(h)
    wind = np.clip(30 + 10 * np.cos(hours / 24 * 2 * np.pi), 0, 50)
    solar = np.clip(50 * np.exp(-((hours - 12) ** 2) / 10), 0, 50)
    return wind, solar


def test_scenario_shapes_and_probabilities():
    wind, solar = _forecasts()
    cfg = ScenarioConfig(n_scenarios=200, seed=1)
    s = generate_scenarios(wind, solar, cfg)
    assert s.wind.shape == (200, 24)
    assert s.solar.shape == (200, 24)
    assert s.probabilities.shape == (200,)
    assert np.isclose(s.probabilities.sum(), 1.0)


def test_scenarios_are_non_negative():
    wind, solar = _forecasts()
    s = generate_scenarios(wind, solar, ScenarioConfig(n_scenarios=100, seed=2))
    assert (s.wind >= 0).all()
    assert (s.solar >= 0).all()


def test_seed_reproducibility():
    wind, solar = _forecasts()
    cfg = ScenarioConfig(n_scenarios=50, seed=42)
    a = generate_scenarios(wind, solar, cfg)
    b = generate_scenarios(wind, solar, cfg)
    assert np.allclose(a.wind, b.wind)
    assert np.allclose(a.solar, b.solar)


def test_ensemble_mean_is_close_to_forecast():
    wind, solar = _forecasts()
    s = generate_scenarios(wind, solar, ScenarioConfig(n_scenarios=4000, seed=7))
    # Multiplicative mean-1 errors -> ensemble mean tracks the forecast.
    assert np.allclose(s.wind.mean(axis=0), wind, rtol=0.1, atol=2.0)


def test_capacity_cap_is_enforced():
    wind, solar = _forecasts()
    plant = PlantConfig(wind_capacity=50.0, solar_capacity=50.0)
    s = generate_scenarios(wind, solar, ScenarioConfig(n_scenarios=500, seed=3), plant=plant)
    assert s.wind.max() <= 50.0 + 1e-9
    assert s.solar.max() <= 50.0 + 1e-9
