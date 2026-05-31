"""Unit tests for the financial valuation primitives."""

from __future__ import annotations

import numpy as np
import pytest

from ecogrid.finance import cvar, discount_factors, lcoe, npv, value_at_risk


def test_discount_factors_match_formula():
    f = discount_factors(0.10, 3)
    assert f == pytest.approx([1 / 1.1, 1 / 1.1**2, 1 / 1.1**3])


def test_npv_against_hand_calculation():
    # 50k/yr for 10 years at 8%, capex 300k.
    factors_sum = sum(1 / 1.08**y for y in range(1, 11))
    expected = 50_000 * factors_sum - 300_000
    assert npv(50_000, 300_000, 0.08, 10) == pytest.approx(expected)


def test_npv_negative_when_capex_dominates():
    assert npv(1_000, 1_000_000, 0.08, 10) < 0


def test_npv_array_savings_length_validation():
    with pytest.raises(ValueError):
        npv(np.array([1.0, 2.0]), 100.0, 0.08, 10)


def test_lcoe_is_positive_and_reasonable():
    value = lcoe(1_000_000, 50_000, 10_000, 0.08, 10)
    assert value > 0
    # Discounting cost and energy by the same factors -> independent of rate
    # for constant streams; only the capex term scales it up.
    assert value == pytest.approx(
        (1_000_000 / sum(1 / 1.08**y for y in range(1, 11)) + 50_000) / 10_000
    )


def test_var_and_cvar_on_uniform_distribution():
    costs = np.arange(100, dtype=float)  # 0..99 equiprobable
    assert value_at_risk(costs, 0.95) == pytest.approx(95.0)
    # Worst 5% = {95,96,97,98,99}, mean = 97.
    assert cvar(costs, 0.95) == pytest.approx(97.0)


def test_cvar_never_below_var():
    rng = np.random.default_rng(0)
    costs = rng.lognormal(mean=10, sigma=1.0, size=2000)
    assert cvar(costs, 0.9) >= value_at_risk(costs, 0.9) - 1e-6


def test_cvar_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        cvar(np.arange(10.0), alpha=1.5)
