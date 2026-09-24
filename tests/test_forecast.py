"""Contract tests for the offline PV benchmark and causal forecast scenarios."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ecogrid.data.benchmark import load_public_data, synthetic_data
from ecogrid.forecast import ResidualLibrary, calibrate_residuals, fit_forecaster


def test_synthetic_data_is_reproducible_hourly_utc_and_physical():
    a = synthetic_data(days=100, seed=42, pv_capacity_kw=5.0)
    b = synthetic_data(days=100, seed=42, pv_capacity_kw=5.0)

    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 2400
    assert str(a.index.tz) == "UTC"
    assert a.index.is_monotonic_increasing and a.index.is_unique
    assert (a.index[1:] - a.index[:-1] == pd.Timedelta(hours=1)).all()
    assert list(a.columns) == ["pv_kw", "load_kw", "price_eur_per_kwh"]
    assert a.pv_kw.between(0, 5).all()
    assert (a.load_kw > 0).all()
    assert (a.price_eur_per_kwh >= 0).all()
    assert a.attrs["source"] == "synthetic"
    assert a.attrs["pv_capacity_kw"] == 5.0


def test_public_cache_converts_hourly_radiation_proxy_and_preserves_provenance(tmp_path):
    cache = tmp_path / "raw.json"
    request = {
        "latitude": 48.137,
        "longitude": 11.575,
        "start_date": "2024-05-01",
        "end_date": "2024-05-01",
        "hourly": "shortwave_radiation",
        "timezone": "GMT",
    }
    hours = pd.date_range("2024-05-01", periods=24, freq="h", tz="UTC")
    radiation = [0.0] * 24
    radiation[11:14] = [0.0, 500.0, 1000.0]
    response = {
        "latitude": 48.125,
        "longitude": 11.625,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "hourly_units": {"time": "iso8601", "shortwave_radiation": "W/m²"},
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in hours],
            "shortwave_radiation": radiation,
        },
    }
    cache.write_text(json.dumps({"request": request, "response": response}), encoding="utf-8")

    frame = load_public_data(cache_path=cache, start_date="2024-05-01", end_date="2024-05-01")

    np.testing.assert_allclose(frame.pv_kw.iloc[11:14], [0.0, 2.0, 4.0])
    assert len(frame) == 24 and str(frame.index.tz) == "UTC"
    assert frame.index[0] == pd.Timestamp("2024-04-30T23:00:00Z")
    assert frame.attrs["source"] == "open_meteo_historical_weather_proxy"
    assert frame.attrs["pv_observation_type"] == "irradiance_derived_proxy_not_measured_pv"
    assert frame.attrs["weather_observation_type"] == "historical_archive_not_issued_forecast"


def test_forecast_ignores_mutations_at_and_after_first_valid_time():
    frame = synthetic_data(days=35)
    train = frame.iloc[: 20 * 24]
    valid = frame.index[25 * 24 : 26 * 24]
    forecaster = fit_forecaster(train)

    baseline = forecaster.predict(frame, valid)
    changed = frame.copy()
    changed.loc[valid[0] :, "pv_kw"] = 0.0
    changed.loc[valid[0] :, "load_kw"] = 999.0

    np.testing.assert_array_equal(forecaster.predict(changed, valid), baseline)
    assert baseline.shape == (24,)
    assert np.isfinite(baseline).all()
    assert ((0 <= baseline) & (baseline <= 5)).all()


def test_calibration_is_frozen_before_end_and_scenario_seed_is_repeatable():
    frame = synthetic_data(days=60)
    forecaster = fit_forecaster(frame.iloc[: 20 * 24])
    start, end = frame.index[25 * 24], frame.index[40 * 24]

    original = calibrate_residuals(forecaster, frame, start, end, horizon=24)
    changed = frame.copy()
    changed.loc[end:, "pv_kw"] = 5.0
    altered = calibrate_residuals(forecaster, changed, start, end, horizon=24)

    np.testing.assert_array_equal(original.residuals, altered.residuals)
    assert original.origins.equals(altered.origins)
    assert (original.origins + pd.Timedelta(hours=23) < end).all()
    issue = frame.index[45 * 24]
    valid = pd.date_range(issue, periods=24, freq="h", tz="UTC")
    point = forecaster.predict(frame.loc[frame.index < issue], valid)
    a = original.sample_scenarios(point, issue, n_scenarios=10, seed=7)
    b = original.sample_scenarios(point, issue, n_scenarios=10, seed=7)
    c = altered.sample_scenarios(point, issue, n_scenarios=10, seed=7)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    assert a.shape == (10, 24)
    assert ((0 <= a) & (a <= 5)).all()


def test_residual_sampling_keeps_whole_blocks_and_matches_issue_hour():
    origins = pd.DatetimeIndex(
        ["2024-05-01T00:00:00Z", "2024-05-02T00:00:00Z", "2024-05-01T01:00:00Z"]
    )
    library = ResidualLibrary(
        residuals=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [9.0, 9.0, 9.0]]),
        origins=origins,
        horizon=3,
        pv_capacity_kw=20.0,
        calibration_start=pd.Timestamp("2024-05-01T00:00:00Z"),
        calibration_end=pd.Timestamp("2024-05-03T00:00:00Z"),
    )
    sampled = library.sample_scenarios(np.array([5.0, 5.0, 5.0]), origins[0], 20, seed=4)

    expected_choices = [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1]
    expected = np.array([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])[expected_choices]
    np.testing.assert_array_equal(sampled, expected)
    np.testing.assert_array_equal(
        library.sample_scenarios(np.full(3, 5.0), origins[0], 20, 4, mode="sampled"),
        expected,
    )
    allowed = {(6.0, 7.0, 8.0), (9.0, 10.0, 11.0)}
    assert {tuple(row) for row in sampled}.issubset(allowed)
    assert {tuple(row) for row in sampled} == allowed
    with pytest.raises(ValueError, match="UTC"):
        library.sample_scenarios(np.ones(3), pd.Timestamp("2024-05-04 00:00"), 2, seed=4)


@pytest.mark.parametrize("candidate_count", [3, 5])
def test_all_matching_keeps_each_source_once_in_origin_order(candidate_count):
    matching_origins = pd.date_range("2024-05-01", periods=candidate_count, freq="D", tz="UTC")
    origins = matching_origins.insert(1, matching_origins[0] + pd.Timedelta(hours=1))
    residuals = np.array(
        [[-2, 1, 3], [-2, 1, 3], [1, -1, -2], [3, 0, 0], [-3, 1, 4]], dtype=float
    )[:candidate_count]
    library = ResidualLibrary(
        residuals=np.insert(residuals, 1, [9.0, 9.0, 9.0], axis=0),
        origins=origins,
        horizon=3,
        pv_capacity_kw=5.0,
        calibration_start=origins[0],
        calibration_end=pd.Timestamp("2024-05-06T00:00:00Z"),
    )
    point = np.array([1.0, 2.0, 4.0])
    expected = np.array([[0, 3, 5], [0, 3, 5], [2, 1, 2], [4, 2, 4], [0, 3, 5]])[
        :candidate_count
    ]

    for requested_count, seed in [(12, 7), (1, 99)]:
        scenarios = library.sample_scenarios(
            point, library.calibration_end, requested_count, seed, mode="all_matching"
        )
        np.testing.assert_array_equal(scenarios, expected)
        assert scenarios.shape == (candidate_count, 3)


def test_all_matching_rejects_unfrozen_library_and_invalid_mode():
    origins = pd.date_range("2024-05-01", periods=2, freq="D", tz="UTC")
    library = ResidualLibrary(
        residuals=np.zeros((2, 3)),
        origins=origins,
        horizon=3,
        pv_capacity_kw=5.0,
        calibration_start=origins[0],
        calibration_end=pd.Timestamp("2024-05-03T00:00:00Z"),
    )
    with pytest.raises(ValueError, match="calibration_end"):
        library.sample_scenarios(np.ones(3), origins[1], 12, 7, mode="all_matching")
    with pytest.raises(ValueError, match="mode"):
        library.sample_scenarios(np.ones(3), library.calibration_end, 12, 7, mode="unknown")
    with pytest.raises(ValueError, match="UTC"):
        library.sample_scenarios(
            np.ones(3), library.calibration_end.tz_localize(None), 12, 7, mode="all_matching"
        )
    with pytest.raises(ValueError, match="no calibration residuals"):
        library.sample_scenarios(
            np.ones(3), library.calibration_end + pd.Timedelta(hours=1), 12, 7, mode="all_matching"
        )


def test_delayed_meter_values_and_unreleased_calibration_truth_are_not_available():
    frame = synthetic_data(days=40)
    frame["observation_available_at"] = frame.index + pd.Timedelta(hours=2)
    model = fit_forecaster(frame.iloc[: 20 * 24])
    issue = frame.index[30 * 24]
    valid = pd.date_range(issue, periods=24, freq="h", tz="UTC")
    before = model.predict(frame, valid)
    changed = frame.copy()
    # The most recent completed nominal hour is still unavailable to this policy.
    changed.loc[issue - pd.Timedelta(hours=1), "pv_kw"] = 5.0
    np.testing.assert_array_equal(before, model.predict(changed, valid))
    library = calibrate_residuals(model, frame, frame.index[22 * 24], issue)
    assert (library.origins + pd.Timedelta(hours=25) <= issue).all()
