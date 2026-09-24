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
    # The lower empirical quantile is 94: F(94) = 0.95.
    assert value_at_risk(costs, 0.95) == pytest.approx(94.0)
    assert value_at_risk(costs, 0.95, np.full(100, 0.01)) == pytest.approx(94.0)
    # Worst 5% = {95,96,97,98,99}, mean = 97.
    assert cvar(costs, 0.95) == pytest.approx(97.0)
    assert cvar(costs, 0.95, np.full(100, 0.01)) == pytest.approx(97.0)
    assert value_at_risk([0, 100], np.nextafter(0.5, 1.0)) == 100


@pytest.mark.parametrize(
    ("costs", "alpha", "probabilities", "expected_var", "expected_cvar"),
    [
        ([0, 100], 0.25, None, 0, 100 / 1.5),
        ([10, 20, 30], 0.8, [0.7, 0.2, 0.1], 20, 25),
        ([10, 20, 20, 30], 0.4, [0.2, 0.3, 0.1, 0.4], 20, 80 / 3),
        ([0, 50, 100], 0.5, [0.5, 0.0, 0.5], 0, 100),
        ([42], 0.95, None, 42, 42),
    ],
)
def test_var_and_cvar_use_empirical_tail_mass(
    costs, alpha, probabilities, expected_var, expected_cvar
):
    assert value_at_risk(costs, alpha, probabilities) == pytest.approx(expected_var)
    assert cvar(costs, alpha, probabilities) == pytest.approx(expected_cvar)


@pytest.mark.parametrize("risk_measure", [value_at_risk, cvar])
def test_risk_measures_reject_invalid_costs(risk_measure):
    for costs in ([], [[1, 2]], [1, np.nan]):
        with pytest.raises(ValueError):
            risk_measure(costs)


@pytest.mark.parametrize("risk_measure", [value_at_risk, cvar])
def test_risk_measures_reject_invalid_probabilities(risk_measure):
    for probabilities in ([1.0], [-0.1, 1.1], [np.inf, 0.5], [0.3, 0.3], [1e308, 1e308]):
        with pytest.raises(ValueError):
            risk_measure([10, 20], probabilities=probabilities)


@pytest.mark.parametrize("risk_measure", [value_at_risk, cvar])
def test_risk_measures_reject_invalid_alpha(risk_measure):
    for alpha in (0, 1, 1.5, np.nan, "0.5"):
        with pytest.raises(ValueError):
            risk_measure([10, 20], alpha=alpha)


def test_cvar_never_below_var():
    rng = np.random.default_rng(0)
    costs = rng.lognormal(mean=10, sigma=1.0, size=2000)
    assert cvar(costs, 0.9) >= value_at_risk(costs, 0.9) - 1e-6


def test_cvar_tiny_tail_does_not_exceed_largest_loss():
    costs = np.arange(100, dtype=float)
    alpha = np.nextafter(1.0, 0.0)
    assert cvar(costs, alpha) == pytest.approx(99.0)


def test_var_preserves_tiny_probability_mass_at_boundary():
    costs = np.arange(202, dtype=float)
    alpha = np.nextafter(0.5, 1.0)
    probabilities = np.array([0.5] + [1e-18] * 200 + [0.5 - 200e-18])
    assert value_at_risk(costs, alpha, probabilities) == 112.0

    # The near-one total needs accurate normalization before the quantile search.
    probabilities[-1] = 0.5
    assert value_at_risk(costs, alpha, probabilities) == 201.0
