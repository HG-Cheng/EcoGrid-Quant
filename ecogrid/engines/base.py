"""Abstract dispatch-engine interface.

Both the legacy SLSQP engine and the industrial MILP engine implement the same
:class:`DispatchEngine` contract so they can be benchmarked head-to-head and
swapped transparently inside the back-testing framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ecogrid.config import GridConfig
from ecogrid.results import DispatchResult


class DispatchEngine(ABC):
    """Common interface for multi-period economic-dispatch optimisers.

    Parameters
    ----------
    config
        Physical and economic parameters of the micro-grid. Defaults to a
        fresh :class:`~ecogrid.config.GridConfig`.
    """

    def __init__(self, config: GridConfig | None = None) -> None:
        self.config: GridConfig = config if config is not None else GridConfig()

    @abstractmethod
    def solve(
        self,
        demand: np.ndarray,
        wind_avail: np.ndarray,
        solar_avail: np.ndarray,
    ) -> DispatchResult:
        """Optimise dispatch over a horizon.

        Parameters
        ----------
        demand
            Electricity demand per period, in MW. Length defines the horizon.
        wind_avail, solar_avail
            Maximum available wind / solar power per period, in MW. Must match
            the length of ``demand``.

        Returns
        -------
        DispatchResult
            The optimal dispatch trajectory and objective value.
        """
        raise NotImplementedError

    @staticmethod
    def _validate_inputs(
        demand: np.ndarray,
        wind_avail: np.ndarray,
        solar_avail: np.ndarray,
    ) -> int:
        """Validate input arrays and return the common horizon length.

        Raises
        ------
        ValueError
            If the three arrays do not share a single common length.
        """
        demand = np.asarray(demand, dtype=float)
        wind_avail = np.asarray(wind_avail, dtype=float)
        solar_avail = np.asarray(solar_avail, dtype=float)
        lengths = {demand.shape[0], wind_avail.shape[0], solar_avail.shape[0]}
        if len(lengths) != 1:
            raise ValueError(
                f"demand, wind_avail and solar_avail must share one length; "
                f"got {sorted(lengths)}"
            )
        return demand.shape[0]
