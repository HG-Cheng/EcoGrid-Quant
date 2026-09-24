"""Offline hourly PV benchmark data in kW, kWh intervals, and EUR/kWh.

Each UTC index label is the *start* of the one-hour energy interval [t, t+1h).
The public series converts historical weather reanalysis to a PV proxy; it is
neither measured PV production nor a forecast as issued in the past.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_COLUMNS = ["pv_kw", "load_kw", "price_eur_per_kwh"]
_DEFAULT_CACHE = Path("data/raw/open_meteo_munich_2024.json")


def _fixed_load_and_price(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Known deterministic benchmark load and tariff, independent of PV."""
    hour = index.hour.to_numpy()
    weekend = np.asarray(index.dayofweek >= 5, dtype=float)
    load = 2.1 + 0.45 * ((hour >= 6) & (hour < 9)) + 0.85 * ((hour >= 17) & (hour < 23))
    load = np.asarray(load, dtype=float) + 0.25 * weekend
    price = 0.16 + 0.04 * ((hour >= 17) & (hour < 22)) + 0.015 * ((hour >= 8) & (hour < 17))
    return load, np.asarray(price, dtype=float)


def _validate_frame(frame: pd.DataFrame, pv_capacity_kw: float) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC":
        raise ValueError("benchmark timestamps must be UTC")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("benchmark timestamps must be strictly increasing and unique")
    if len(frame) == 0 or not (frame.index[1:] - frame.index[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("benchmark must have contiguous hourly intervals")
    if list(frame.columns) != _COLUMNS:
        raise ValueError(f"benchmark columns must be {_COLUMNS}")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("benchmark values must be finite and nonnegative")
    if (frame.pv_kw > pv_capacity_kw + 1e-9).any() or (frame.load_kw <= 0).any():
        raise ValueError("benchmark PV exceeds capacity or load is zero")


def synthetic_data(days: int = 100, seed: int = 42, pv_capacity_kw: float = 5.0) -> pd.DataFrame:
    """Create reproducible hourly weather-like PV plus known load and tariff.

    The site clock is represented in UTC; all power columns are kW and each
    hourly power value also gives kWh over its one-hour interval.
    """
    if days < 1 or pv_capacity_kw <= 0 or not np.isfinite(pv_capacity_kw):
        raise ValueError("days and pv_capacity_kw must be positive")
    index = pd.date_range("2024-05-01", periods=days * 24, freq="h", tz="UTC", name="time")
    rng = np.random.default_rng(seed)
    hour = index.hour.to_numpy()
    clear_sky = np.maximum(np.sin(np.pi * (hour - 4.5) / 15.0), 0.0) ** 1.5
    day_cloud = np.empty(days)
    cloud_state = 0.7
    for day in range(days):
        cloud_state = 0.65 * cloud_state + 0.35 * rng.uniform(0.35, 1.05)
        day_cloud[day] = cloud_state
    hourly_cloud = np.clip(np.repeat(day_cloud, 24) + rng.normal(0, 0.08, days * 24), 0.05, 1)
    pv = np.clip(pv_capacity_kw * clear_sky * hourly_cloud, 0, pv_capacity_kw)
    load, price = _fixed_load_and_price(index)
    frame = pd.DataFrame({"pv_kw": pv, "load_kw": load, "price_eur_per_kwh": price}, index=index)
    frame.attrs = {
        "source": "synthetic",
        "pv_capacity_kw": float(pv_capacity_kw),
        "index_semantics": "UTC interval start, [t,t+1h)",
        "pv_observation_type": "synthetic_pv",
        "load_price_type": "deterministic_benchmark_assumptions",
    }
    _validate_frame(frame, pv_capacity_kw)
    return frame


def load_public_data(
    cache_path: str | Path = _DEFAULT_CACHE,
    start_date: str = "2024-05-01",
    end_date: str = "2024-08-08",
    pv_capacity_kw: float = 5.0,
) -> pd.DataFrame:
    """Load Open-Meteo historical shortwave radiation through an immutable raw cache.

    Open-Meteo's ``shortwave_radiation`` is a preceding-hour mean in W/m².
    Its response timestamp is shifted back one hour to label the interval
    start. PV is ``capacity * radiation / 1000 * 0.8``, clipped to capacity;
    0.8 is an explicit aggregate loss assumption, not a calibrated PV model.
    """
    if pv_capacity_kw <= 0 or not np.isfinite(pv_capacity_kw):
        raise ValueError("pv_capacity_kw must be positive")
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("start_date must be no later than end_date")
    request: dict[str, str | float] = {
        "latitude": 48.137,
        "longitude": 11.575,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "shortwave_radiation",
        "timezone": "GMT",
    }
    cache = Path(cache_path)
    if cache.exists():
        envelope = json.loads(cache.read_text(encoding="utf-8"))
        if envelope.get("request") != request:
            raise ValueError("cached raw weather request differs from requested dates or site")
        response_data = envelope["response"]
    else:
        last_error: requests.RequestException | None = None
        for _ in range(2):
            try:
                response = requests.get(_ARCHIVE_URL, params=request, timeout=20)
                response.raise_for_status()
                response_data = response.json()
                break
            except requests.RequestException as error:
                last_error = error
        else:
            raise RuntimeError(
                "Open-Meteo historical weather fetch failed after 2 attempts"
            ) from last_error
        envelope = {"request": request, "response": response_data}

    if response_data.get("utc_offset_seconds") != 0:
        raise ValueError("public weather response must use UTC/GMT")
    if response_data.get("hourly_units", {}).get("shortwave_radiation") != "W/m²":
        raise ValueError("expected shortwave_radiation units W/m²")
    hourly = response_data["hourly"]
    response_times = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True))
    index = response_times - pd.Timedelta(hours=1)
    index.name = "time"
    radiation = np.asarray(hourly["shortwave_radiation"], dtype=float)
    if len(radiation) != len(index) or not np.isfinite(radiation).all() or (radiation < 0).any():
        raise ValueError("public radiation must be complete, finite and nonnegative")
    pv = np.clip(pv_capacity_kw * radiation / 1000.0 * 0.8, 0, pv_capacity_kw)
    load, price = _fixed_load_and_price(index)
    frame = pd.DataFrame({"pv_kw": pv, "load_kw": load, "price_eur_per_kwh": price}, index=index)
    frame.attrs = {
        "source": "open_meteo_historical_weather_proxy",
        "source_url": _ARCHIVE_URL,
        "site_requested": "Munich, Germany (48.137 N, 11.575 E)",
        "weather_start_date": start_date,
        "weather_end_date": end_date,
        "weather_unit": "shortwave_radiation W/m² preceding-hour mean",
        "pv_conversion": "clip(capacity_kw * shortwave_radiation / 1000 * 0.8, 0, capacity_kw)",
        "pv_observation_type": "irradiance_derived_proxy_not_measured_pv",
        "weather_observation_type": "historical_archive_not_issued_forecast",
        "load_price_type": "deterministic_benchmark_assumptions",
        "pv_capacity_kw": float(pv_capacity_kw),
        "index_semantics": "UTC interval start, [t,t+1h); API end labels shifted -1h",
        "raw_cache_path": str(cache),
    }
    _validate_frame(frame, pv_capacity_kw)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("x", encoding="utf-8") as handle:
            json.dump(envelope, handle, ensure_ascii=False)
    return frame
