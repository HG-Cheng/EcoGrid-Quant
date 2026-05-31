"""EcoGrid-Quant: micro-grid dispatch and climate-risk engine.

Public API surface for the industrial-grade engine. Import the configuration
dataclasses and dispatch engines directly from the top-level package::

    from ecogrid import GridConfig, MilpDispatchEngine

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
