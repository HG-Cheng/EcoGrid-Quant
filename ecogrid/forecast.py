"""Causal PV forecasts and frozen, hour-matched residual block scenarios.

All timestamps are UTC starts of one-hour intervals. A forecast issued at t
has first valid interval t (lead 0); observed history must end before t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _utc_hourly(index: pd.DatetimeIndex, name: str) -> None:
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC":
        raise ValueError(f"{name} must have UTC timestamps")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError(f"{name} must be sorted and unique")
    if len(index) == 0 or not (index[1:] - index[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{name} must be contiguous hourly timestamps")


@dataclass(frozen=True)
class PVForecaster:
    """Training-hour climatology blended with recent observed same-hour PV."""

    hourly_mean_kw: np.ndarray
    pv_capacity_kw: float

    def predict(self, history: pd.DataFrame, valid_index: pd.DatetimeIndex) -> np.ndarray:
        """Predict PV for each valid interval using observations strictly before issue.

        ``issue_time = valid_index[0]``. Any rows in history at or after issue
        are ignored, so accidental future inclusion cannot change the result.
        """
        _utc_hourly(valid_index, "valid_index")
        _utc_hourly(history.index, "history.index")
        if "pv_kw" not in history:
            raise ValueError("history needs pv_kw")
        issue = valid_index[0]
        observed = history.index < issue
        if "observation_available_at" in history:
            available = pd.to_datetime(history.observation_available_at, utc=True)
            if available.isna().any():
                raise ValueError("Observation availability timestamps must be complete")
            observed = observed & (available <= issue)
        past = history.loc[observed, "pv_kw"]
        if len(past) == 0 or not np.isfinite(past.to_numpy(dtype=float)).all():
            raise ValueError("history needs finite PV observations before issue")
        recent = past.loc[past.index >= issue - pd.Timedelta(days=7)]
        result = np.empty(len(valid_index), dtype=float)
        for i, valid in enumerate(valid_index):
            base = self.hourly_mean_kw[valid.hour]
            same_hour = recent.loc[recent.index.hour == valid.hour]
            result[i] = 0.5 * base + 0.5 * same_hour.mean() if len(same_hour) else base
        return np.clip(result, 0.0, self.pv_capacity_kw)


def fit_forecaster(train_frame: pd.DataFrame) -> PVForecaster:
    """Fit a simple 24-hour PV climatology using only the supplied training frame."""
    _utc_hourly(train_frame.index, "train_frame.index")
    if "pv_kw" not in train_frame:
        raise ValueError("train_frame needs pv_kw")
    pv = train_frame.pv_kw.to_numpy(dtype=float)
    capacity = float(train_frame.attrs.get("pv_capacity_kw", 5.0))
    if len(pv) < 24 or not np.isfinite(pv).all() or (pv < 0).any() or (pv > capacity).any():
        raise ValueError("training PV needs at least 24 valid hourly values within capacity")
    means = train_frame.groupby(train_frame.index.hour).pv_kw.mean().reindex(range(24))
    if means.isna().any():
        raise ValueError("training frame must include every UTC hour")
    return PVForecaster(means.to_numpy(dtype=float), capacity)


@dataclass(frozen=True)
class ResidualLibrary:
    """Frozen 24-hour (or selected horizon) PV error trajectories.

    Each row holds actual minus predicted PV from one historical issue time,
    preserving correlation across leads. Sampling selects a whole row with
    the same UTC issue hour. Scenarios are physically projected into
    ``[0, pv_capacity_kw]``; clipping may change their residual distribution.
    """

    residuals: np.ndarray
    origins: pd.DatetimeIndex
    horizon: int
    pv_capacity_kw: float
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp

    def sample_scenarios(
        self,
        point_forecast: np.ndarray,
        issue_time: pd.Timestamp,
        n_scenarios: int,
        seed: int,
        *,
        mode: str = "sampled",
    ) -> np.ndarray:
        """Build equal-weight whole residual blocks for this UTC issue hour.

        ``sampled`` draws ``n_scenarios`` blocks with replacement using ``seed``.
        ``all_matching`` uses each matching source once in stored origin order,
        independently of the requested count and seed, after calibration ends.
        Duplicate paths remain separate source rows, including after clipping.
        """
        if mode not in ("sampled", "all_matching"):
            raise ValueError("mode must be 'sampled' or 'all_matching'")
        issue = pd.Timestamp(issue_time)
        if str(issue.tz) != "UTC":
            raise ValueError("issue_time must be UTC")
        if mode == "all_matching" and issue < self.calibration_end:
            raise ValueError("all_matching issue_time must be at or after calibration_end")
        if n_scenarios < 1:
            raise ValueError("n_scenarios must be positive")
        point = np.asarray(point_forecast, dtype=float)
        if point.shape != (self.horizon,) or not np.isfinite(point).all():
            raise ValueError("point_forecast must be finite and match calibration horizon")
        if self.residuals.ndim != 2 or self.residuals.shape != (len(self.origins), self.horizon):
            raise ValueError("residual library has invalid shape")
        candidates = np.flatnonzero(self.origins.hour == issue.hour)
        if len(candidates) == 0:
            raise ValueError("no calibration residuals for issue UTC hour")
        if mode == "all_matching":
            selected = candidates
        else:
            rng = np.random.default_rng(seed)
            selected = rng.choice(candidates, size=n_scenarios, replace=True)
        return np.clip(point[None, :] + self.residuals[selected], 0.0, self.pv_capacity_kw)


def calibrate_residuals(
    forecaster: PVForecaster,
    full_prefix_frame: pd.DataFrame,
    calibration_start: pd.Timestamp,
    calibration_end: pd.Timestamp,
    horizon: int = 24,
) -> ResidualLibrary:
    """Calibrate historical rolling forecast errors in [start, end).

    Every valid interval of each error trajectory lies strictly before the
    exclusive calibration end. Each prediction sees observations strictly
    before its issue time, so even overlapping windows remain causal.
    """
    _utc_hourly(full_prefix_frame.index, "full_prefix_frame.index")
    start, end = pd.Timestamp(calibration_start), pd.Timestamp(calibration_end)
    if str(start.tz) != "UTC" or str(end.tz) != "UTC":
        raise ValueError("calibration bounds must be UTC")
    if horizon < 1 or end <= start:
        raise ValueError("horizon must be positive and calibration end after start")
    if "pv_kw" not in full_prefix_frame:
        raise ValueError("full_prefix_frame needs pv_kw")
    last_origin = end - pd.Timedelta(hours=horizon)
    origins = full_prefix_frame.index[
        (full_prefix_frame.index >= start) & (full_prefix_frame.index <= last_origin)
    ]
    if len(origins) == 0:
        raise ValueError("no complete calibration windows before calibration_end")
    blocks = []
    usable_origins = []
    for issue in origins:
        valid = pd.date_range(issue, periods=horizon, freq="h", tz="UTC")
        if not valid.isin(full_prefix_frame.index).all():
            raise ValueError("calibration frame is missing valid intervals")
        if "observation_available_at" in full_prefix_frame:
            available = pd.to_datetime(
                full_prefix_frame.loc[valid, "observation_available_at"], utc=True
            )
            if available.isna().any():
                raise ValueError("Observation availability timestamps must be complete")
            if (available > end).any():
                continue
        history = full_prefix_frame.loc[full_prefix_frame.index < issue]
        point = forecaster.predict(history, valid)
        actual = full_prefix_frame.loc[valid, "pv_kw"].to_numpy(dtype=float)
        if not np.isfinite(actual).all() or (actual < 0).any():
            raise ValueError("calibration PV must be finite and nonnegative")
        blocks.append(actual - point)
        usable_origins.append(issue)
    if not blocks:
        raise ValueError("no calibration windows fully available by calibration_end")
    return ResidualLibrary(
        residuals=np.asarray(blocks, dtype=float),
        origins=pd.DatetimeIndex(usable_origins),
        horizon=horizon,
        pv_capacity_kw=forecaster.pv_capacity_kw,
        calibration_start=start,
        calibration_end=end,
    )
