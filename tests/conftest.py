"""Shared pytest fixtures for the EcoGrid-Quant test suite."""

from __future__ import annotations

import numpy as np
import pytest

from ecogrid.config import GridConfig


@pytest.fixture
def small_horizon() -> int:
    """A short horizon keeps MILP solves fast in unit tests."""
    return 6


@pytest.fixture
def simple_profile(small_horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A deterministic (demand, wind, solar) profile over the horizon."""
    h = small_horizon
    demand = np.full(h, 100.0)
    wind = np.array([40.0, 20.0, 50.0, 10.0, 30.0, 45.0])[:h]
    solar = np.array([0.0, 10.0, 40.0, 50.0, 20.0, 0.0])[:h]
    return demand, wind, solar


@pytest.fixture
def base_config() -> GridConfig:
    """A canonical grid configuration used across engine tests."""
    return GridConfig(
        carbon_tax_rate=80.0,
        bess_capacity=80.0,
        coal_pmin=20.0,
        coal_pmax=500.0,
        startup_cost=1000.0,
        min_up_time=2,
        min_down_time=2,
    )
