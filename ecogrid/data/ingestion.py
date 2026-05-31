"""Weather data ingestion with on-disk caching.

Fetches hourly short-wave irradiance and 100 m wind speed from the Open-Meteo
historical archive API and converts them into available renewable power using
the simplified linear model in :class:`~ecogrid.config.PlantConfig`. Responses
are cached to a local CSV so repeated back-tests do not hammer the public API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ecogrid.config import PlantConfig

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_RADIATION_COL = "光伏辐射_W/m2"
_WIND_COL = "百米风速_m/s"


def fetch_weather(
    latitude: float = 48.137,
    longitude: float = 11.575,
    start_date: str = "2024-05-01",
    end_date: str = "2024-05-15",
    cache_path: str | Path | None = "data/munich_weather.csv",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch hourly weather, caching the result to ``cache_path``.

    Parameters
    ----------
    latitude, longitude
        Site coordinates. Defaults to Munich, Germany.
    start_date, end_date
        Inclusive ISO date range for the historical archive query.
    cache_path
        Where to read/write the cached CSV. ``None`` disables caching.
    force_refresh
        If ``True``, ignore any existing cache and re-query the API.

    Returns
    -------
    pandas.DataFrame
        Indexed by timestamp with columns for irradiance and wind speed.
    """
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists() and not force_refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    params: dict[str, str | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "shortwave_radiation,wind_speed_100m",
    }
    response = requests.get(_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    df = pd.DataFrame(
        {
            _RADIATION_COL: hourly["shortwave_radiation"],
            _WIND_COL: hourly["wind_speed_100m"],
        },
        index=pd.to_datetime(hourly["time"]),
    )
    df.index.name = "time"

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
    return df


def weather_to_power(
    df: pd.DataFrame, plant: PlantConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw weather columns into available wind / solar power (MW).

    Parameters
    ----------
    df
        Frame produced by :func:`fetch_weather`.
    plant
        Nameplate capacities and conversion coefficients.

    Returns
    -------
    tuple of numpy.ndarray
        ``(wind_avail, solar_avail)`` arrays in MW, clipped to capacity.
    """
    plant = plant if plant is not None else PlantConfig()
    wind = np.clip(
        df[_WIND_COL].to_numpy() * plant.wind_speed_to_mw, 0, plant.wind_capacity
    )
    solar = np.clip(
        df[_RADIATION_COL].to_numpy() * plant.irradiance_to_mw, 0, plant.solar_capacity
    )
    return wind, solar
