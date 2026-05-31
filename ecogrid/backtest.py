"""Performance and out-of-sample back-testing utilities.

Two industrial-grade capabilities:

* :func:`scan_capacities` parallelises an embarrassingly-parallel capacity (or
  carbon-price) grid sweep across all CPU cores with :mod:`joblib`, replacing
  the V3 serial double ``for`` loop.
* :func:`rolling_backtest` runs a realistic *rolling-horizon* simulation: at
  each decision time the engine only sees a forecast window, commits, executes
  one step against the *realised* weather, then advances and re-optimises. This
  is the only honest way to measure performance when the forecast is imperfect.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ecogrid.config import GridConfig
from ecogrid.engines.milp_engine import MilpDispatchEngine

# ----------------------------------------------------------------------------
# Parallel grid scan
# ----------------------------------------------------------------------------


def _solve_capacity(
    capacity: float,
    base_config: GridConfig,
    demand: np.ndarray,
    wind: np.ndarray,
    solar: np.ndarray,
) -> dict[str, float | bool]:
    """Worker: solve one MILP for a single battery capacity.

    Defined at module level so it is picklable by the joblib ``loky`` backend
    (required on Windows, which spawns rather than forks worker processes).
    """
    cfg = replace(base_config, bess_capacity=float(capacity))
    res = MilpDispatchEngine(cfg).solve(demand, wind, solar)
    return {
        "capacity": float(capacity),
        "total_cost": res.total_cost,
        "coal_energy": float(np.sum(res.coal)),
        "success": res.success,
    }


def scan_capacities(
    demand: np.ndarray,
    wind: np.ndarray,
    solar: np.ndarray,
    capacities: np.ndarray,
    base_config: GridConfig | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Parallel battery-capacity grid sweep.

    Parameters
    ----------
    demand, wind, solar
        Dispatch inputs over the horizon, in MW.
    capacities
        Battery capacities (MWh) to evaluate.
    base_config
        Template configuration; ``bess_capacity`` is overridden per grid point.
    n_jobs
        Number of parallel workers (``-1`` uses all cores).

    Returns
    -------
    pandas.DataFrame
        One row per capacity with ``total_cost``, ``coal_energy`` and
        ``success``, sorted by capacity.
    """
    base = base_config if base_config is not None else GridConfig()
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_capacity)(cap, base, demand, wind, solar) for cap in capacities
    )
    return pd.DataFrame(rows).sort_values("capacity").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Rolling-horizon out-of-sample back-test
# ----------------------------------------------------------------------------


@dataclass
class RollingBacktestResult:
    """Outcome of a rolling-horizon out-of-sample back-test.

    Parameters
    ----------
    realized_coal
        Coal output actually dispatched each executed hour, shape ``(H,)``.
    soc
        Battery state-of-charge trajectory across executed hours, shape ``(H,)``.
    commitment
        Realised thermal on/off status across executed hours, shape ``(H,)``.
    realized_cost
        Total cost incurred under the *realised* weather, in EUR.
    perfect_cost
        Cost of a single perfect-foresight optimisation over the true weather.
    regret
        ``realized_cost - perfect_cost``: the price of forecast imperfection.
    unmet_energy
        Total demand that could not be served (MWh); non-zero only if the
        thermal cap was exceeded by a forecast bust.
    """

    realized_coal: np.ndarray
    soc: np.ndarray
    commitment: np.ndarray
    realized_cost: float
    perfect_cost: float
    regret: float
    unmet_energy: float


def rolling_backtest(
    demand: np.ndarray,
    wind_actual: np.ndarray,
    solar_actual: np.ndarray,
    wind_forecast: np.ndarray,
    solar_forecast: np.ndarray,
    config: GridConfig | None = None,
    lookahead: int = 24,
    step: int = 1,
) -> RollingBacktestResult:
    """Run a rolling-horizon out-of-sample back-test.

    At each decision time ``t0`` the engine optimises over the forecast window
    ``[t0, t0 + lookahead)`` starting from the *carried* battery state, then the
    first ``step`` hours of the plan are executed against the **actual** weather
    (the battery follows the plan; coal fills the residual). Time advances by
    ``step`` and the problem is re-solved with updated information.

    Parameters
    ----------
    demand
        Realised demand over the full horizon ``H``, in MW.
    wind_actual, solar_actual
        Realised renewable availability, shape ``(H,)``.
    wind_forecast, solar_forecast
        Forecast renewable availability the engine sees, shape ``(H,)``.
    config
        Grid configuration.
    lookahead
        Forecast window length (hours) visible at each decision.
    step
        Number of hours executed before re-optimising.

    Returns
    -------
    RollingBacktestResult
        Realised dispatch, cost and the regret versus perfect foresight.
    """
    cfg = config if config is not None else GridConfig()
    horizon = int(np.asarray(demand).shape[0])
    cmc = cfg.coal_marginal_cost

    realized_coal = np.zeros(horizon)
    soc_traj = np.zeros(horizon)
    commitment = np.zeros(horizon)
    realized_cost = 0.0
    unmet = 0.0

    soc = cfg.initial_soc
    prev_commit = 0

    for t0 in range(0, horizon, step):
        window = slice(t0, min(t0 + lookahead, horizon))
        cfg_window = replace(cfg, initial_soc=float(soc))
        engine = MilpDispatchEngine(cfg_window, initial_commitment=bool(prev_commit))
        plan = engine.solve(
            demand[window], wind_forecast[window], solar_forecast[window]
        )

        for k in range(step):
            t = t0 + k
            if t >= horizon:
                break
            charge = plan.charge[k] if plan.success else 0.0
            discharge = plan.discharge[k] if plan.success else 0.0

            # The battery executes its plan; coal fills the residual against
            # the *actual* renewable realisation.
            net_load = demand[t] + charge - discharge
            renewable = wind_actual[t] + solar_actual[t]
            coal = max(net_load - renewable, 0.0)
            if coal > cfg.coal_pmax:
                unmet += coal - cfg.coal_pmax
                coal = cfg.coal_pmax

            u = 1 if coal > 1e-6 else 0
            startup = 1 if (u == 1 and prev_commit == 0) else 0
            realized_cost += coal * cmc + startup * cfg.startup_cost

            soc = float(
                np.clip(
                    soc + cfg.eff_charge * charge - discharge / cfg.eff_discharge,
                    0.0,
                    cfg.bess_capacity,
                )
            )
            realized_coal[t] = coal
            soc_traj[t] = soc
            commitment[t] = u
            prev_commit = u

    perfect = MilpDispatchEngine(cfg).solve(demand, wind_actual, solar_actual)
    perfect_cost = perfect.total_cost if perfect.success else float("nan")

    return RollingBacktestResult(
        realized_coal=realized_coal,
        soc=soc_traj,
        commitment=commitment,
        realized_cost=realized_cost,
        perfect_cost=perfect_cost,
        regret=realized_cost - perfect_cost,
        unmet_energy=unmet,
    )
