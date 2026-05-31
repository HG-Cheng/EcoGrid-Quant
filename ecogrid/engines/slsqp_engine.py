"""Legacy SLSQP dispatch engine (baseline).

This is the V2/V3 engine, refactored to satisfy the :class:`DispatchEngine`
interface. It solves a *continuous* multi-period non-linear program with
``scipy.optimize.minimize(method="SLSQP")`` and has **no** integer commitment
variables. It is retained purely as a performance / realism baseline against
which the industrial MILP engine is benchmarked.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ecogrid.config import GridConfig
from ecogrid.engines.base import DispatchEngine
from ecogrid.results import DispatchResult

_VARS_PER_STEP = 6  # [coal, wind, solar, charge, discharge, soc]


class SlsqpDispatchEngine(DispatchEngine):
    """Continuous multi-period dispatch via sequential least-squares (SLSQP)."""

    def __init__(self, config: GridConfig | None = None) -> None:
        super().__init__(config)

    def solve(
        self,
        demand: np.ndarray,
        wind_avail: np.ndarray,
        solar_avail: np.ndarray,
    ) -> DispatchResult:
        """Solve the continuous dispatch NLP. See :meth:`DispatchEngine.solve`."""
        horizon = self._validate_inputs(demand, wind_avail, solar_avail)
        demand = np.asarray(demand, dtype=float)
        wind_avail = np.asarray(wind_avail, dtype=float)
        solar_avail = np.asarray(solar_avail, dtype=float)
        cfg = self.config

        n_vars = _VARS_PER_STEP * horizon

        bounds: list[tuple[float, float | None]] = []
        for t in range(horizon):
            bounds.extend(
                [
                    (0, cfg.coal_pmax),
                    (0, float(wind_avail[t])),
                    (0, float(solar_avail[t])),
                    (0, cfg.bess_max_power),
                    (0, cfg.bess_max_power),
                    (0, cfg.bess_capacity),
                ]
            )

        def objective(x: np.ndarray) -> float:
            total = 0.0
            for t in range(horizon):
                p_coal = x[t * _VARS_PER_STEP]
                total += p_coal * cfg.coal_marginal_cost
            return total

        constraints: list[dict] = []
        for t in range(horizon):

            def make_balance(step: int):
                def balance_rule(x: np.ndarray) -> float:
                    idx = step * _VARS_PER_STEP
                    supply = x[idx] + x[idx + 1] + x[idx + 2] + x[idx + 4]
                    consume = demand[step] + x[idx + 3]
                    return supply - consume

                return balance_rule

            constraints.append({"type": "eq", "fun": make_balance(t)})

            def make_soc(step: int):
                def soc_rule(x: np.ndarray) -> float:
                    idx = step * _VARS_PER_STEP
                    actual_charge = x[idx + 3] * cfg.eff_charge
                    actual_drain = x[idx + 4] / cfg.eff_discharge
                    soc_prev = (
                        cfg.initial_soc
                        if step == 0
                        else x[(step - 1) * _VARS_PER_STEP + 5]
                    )
                    return x[idx + 5] - (soc_prev + actual_charge - actual_drain)

                return soc_rule

            constraints.append({"type": "eq", "fun": make_soc(t)})

        x0 = np.zeros(n_vars)
        res = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        x = res.x.reshape(horizon, _VARS_PER_STEP)
        return DispatchResult(
            coal=x[:, 0],
            wind=x[:, 1],
            solar=x[:, 2],
            charge=x[:, 3],
            discharge=x[:, 4],
            soc=x[:, 5],
            total_cost=float(res.fun),
            success=bool(res.success),
            solver="slsqp",
            commitment=None,
            message=str(res.message),
        )
