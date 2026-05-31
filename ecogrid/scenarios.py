"""Monte-Carlo weather scenario generation for stochastic optimisation.

The deterministic engines assume the weather forecast is perfect. Reality is
uncertain, so this module draws an ensemble of plausible wind / solar
realisations around a point forecast. Forecast errors are modelled as an AR(1)
process so that errors are *temporally correlated* (a cloudy spell persists for
several hours) rather than independent hour-to-hour noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecogrid.config import PlantConfig, ScenarioConfig


@dataclass
class ScenarioSet:
    """An ensemble of weather scenarios with associated probabilities.

    Parameters
    ----------
    wind, solar
        Available power per scenario and period, each of shape
        ``(n_scenarios, horizon)`` in MW.
    probabilities
        Probability weight of each scenario, shape ``(n_scenarios,)``, summing
        to one.
    """

    wind: np.ndarray
    solar: np.ndarray
    probabilities: np.ndarray

    @property
    def n_scenarios(self) -> int:
        """Number of scenarios in the ensemble."""
        return int(self.wind.shape[0])

    @property
    def horizon(self) -> int:
        """Number of time periods per scenario."""
        return int(self.wind.shape[1])


def _ar1_factors(
    horizon: int,
    n_scenarios: int,
    ar: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw multiplicative AR(1) error factors of shape ``(n_scenarios, horizon)``.

    The factor applied to the forecast is ``1 + e_t`` where ``e_t`` follows a
    stationary AR(1) process with marginal standard deviation ``sigma`` and
    persistence ``ar``. Factors are clipped at zero so power stays non-negative.
    """
    factors = np.empty((n_scenarios, horizon))
    # Innovation variance chosen so the marginal variance equals sigma**2.
    innov_sigma = sigma * np.sqrt(max(1.0 - ar**2, 1e-9))
    e = rng.normal(0.0, sigma, size=n_scenarios)
    factors[:, 0] = 1.0 + e
    for t in range(1, horizon):
        e = ar * e + rng.normal(0.0, innov_sigma, size=n_scenarios)
        factors[:, t] = 1.0 + e
    return np.clip(factors, 0.0, None)


def generate_scenarios(
    wind_forecast: np.ndarray,
    solar_forecast: np.ndarray,
    config: ScenarioConfig | None = None,
    plant: PlantConfig | None = None,
) -> ScenarioSet:
    """Generate a Monte-Carlo ensemble around a wind / solar point forecast.

    Parameters
    ----------
    wind_forecast, solar_forecast
        Point-forecast available power per period, each shape ``(horizon,)``.
    config
        Scenario generation parameters (count, AR coefficient, sigmas, seed).
    plant
        Plant capacities used to cap each scenario at nameplate power. If
        ``None`` no capacity cap is applied.

    Returns
    -------
    ScenarioSet
        Equiprobable ensemble of wind / solar realisations.
    """
    cfg = config if config is not None else ScenarioConfig()
    wind_forecast = np.asarray(wind_forecast, dtype=float)
    solar_forecast = np.asarray(solar_forecast, dtype=float)
    horizon = wind_forecast.shape[0]
    if solar_forecast.shape[0] != horizon:
        raise ValueError("wind_forecast and solar_forecast must share a length")

    rng = np.random.default_rng(cfg.seed)

    wind = wind_forecast[None, :] * _ar1_factors(
        horizon, cfg.n_scenarios, cfg.ar_coefficient, cfg.wind_sigma, rng
    )
    solar = solar_forecast[None, :] * _ar1_factors(
        horizon, cfg.n_scenarios, cfg.ar_coefficient, cfg.solar_sigma, rng
    )

    if plant is not None:
        wind = np.clip(wind, 0.0, plant.wind_capacity)
        solar = np.clip(solar, 0.0, plant.solar_capacity)

    probabilities = np.full(cfg.n_scenarios, 1.0 / cfg.n_scenarios)
    return ScenarioSet(wind=wind, solar=solar, probabilities=probabilities)
