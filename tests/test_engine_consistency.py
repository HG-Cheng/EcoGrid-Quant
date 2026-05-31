"""Cross-engine consistency: MILP must agree with SLSQP on the LP relaxation.

When the integer structure is switched off (no minimum output floor, no
start-up cost, unit minimum up/down times), the unit-commitment MILP reduces to
the same continuous economic-dispatch problem the SLSQP baseline solves, so the
two engines must reach the same optimal cost. This guards against modelling
regressions in either engine.
"""

from __future__ import annotations

import numpy as np

from ecogrid.config import GridConfig
from ecogrid.engines import MilpDispatchEngine, SlsqpDispatchEngine


def test_milp_matches_slsqp_on_relaxed_problem():
    h = 8
    rng = np.random.default_rng(123)
    demand = np.full(h, 110.0)
    wind = np.clip(rng.normal(40, 12, h), 0, 50)
    solar = np.clip(rng.normal(25, 15, h), 0, 50)

    # Relaxed config: no integer-driven economics -> identical optimum expected.
    cfg = GridConfig(
        carbon_tax_rate=80.0,
        bess_capacity=60.0,
        coal_pmin=0.0,
        coal_pmax=1000.0,
        startup_cost=0.0,
        min_up_time=1,
        min_down_time=1,
    )

    milp = MilpDispatchEngine(cfg).solve(demand, wind, solar)
    slsqp = SlsqpDispatchEngine(cfg).solve(demand, wind, solar)

    assert milp.success and slsqp.success
    # SLSQP is a numerical NLP; allow a small relative tolerance.
    assert milp.total_cost == np.float64(slsqp.total_cost) or abs(
        milp.total_cost - slsqp.total_cost
    ) <= 1e-2 * max(abs(slsqp.total_cost), 1.0)
