"""PV forecast uncertainty, battery control and finite grid imports.

The active research entry point is ``python -m ecogrid``. SystemConfig and
solve_dispatch implement the new benchmark. The coal/wind engines and old
rolling_backtest remain importable for archival compatibility, not as validated
research baselines (see archive/notebooks/README.md).
"""

from ecogrid.backtest import (
    RollingBacktestResult,
    rolling_backtest,
    scan_capacities,
)
from ecogrid.config import (
    FinanceConfig,
    GridConfig,
    PlantConfig,
    ScenarioConfig,
)
from ecogrid.dispatch import SystemConfig, solve_dispatch
from ecogrid.engines import (
    DispatchEngine,
    MilpDispatchEngine,
    SlsqpDispatchEngine,
    StochasticDispatchEngine,
)
from ecogrid.finance import cvar, lcoe, npv, value_at_risk
from ecogrid.results import DispatchResult, StochasticResult
from ecogrid.scenarios import ScenarioSet, generate_scenarios

__version__ = "0.8.0"

__all__ = [
    "SystemConfig",
    "solve_dispatch",
    "FinanceConfig",
    "GridConfig",
    "PlantConfig",
    "ScenarioConfig",
    "DispatchEngine",
    "MilpDispatchEngine",
    "SlsqpDispatchEngine",
    "StochasticDispatchEngine",
    "DispatchResult",
    "StochasticResult",
    "ScenarioSet",
    "generate_scenarios",
    "npv",
    "lcoe",
    "cvar",
    "value_at_risk",
    "scan_capacities",
    "rolling_backtest",
    "RollingBacktestResult",
    "__version__",
]
