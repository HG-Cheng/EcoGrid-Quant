"""Meter reconstruction, quality-window and immutable-cache contracts."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ecogrid.data import measured

COLUMN = "DE_KN_residential6_pv"


def _raw():
    index = pd.date_range("2016-10-02", "2017-10-09", freq="h", tz="UTC", name="time")
    return pd.DataFrame(
        {COLUMN: np.arange(len(index), dtype=float), "interpolated": ""}, index=index
    )


def test_backward_meter_difference_preserves_bins_and_delays_observation():
    raw = _raw().iloc[:4].copy()
    raw[COLUMN] = [100.0, 102.0, 103.0, 103.0]
    raw["interpolated"] = ["", COLUMN + "_other", "DE_KN_residential4_pv", ""]

    frame = measured.build_measured_frame(raw, pv_scale=0.5, pv_capacity_kw=5.0)

    assert frame.index.equals(raw.index)
    np.testing.assert_allclose(frame.pv_kw, [np.nan, 1.0, 0.5, 0.0], equal_nan=True)
    assert frame.pv_quality_good.tolist() == [False, True, True, True]
    assert frame.observation_available_at.iloc[1] == pd.Timestamp("2016-10-02T03:00:00Z")
    assert frame.attrs["pv_scale"] == 0.5
    assert frame.attrs["pv_observation_type"] == "measured_meter_energy_processed_by_publisher"


@pytest.mark.parametrize(
    "corruption", ["previous_endpoint", "calibration_flag", "missing", "negative", "gap"]
)
def test_selected_windows_reject_corrupt_training_calibration_and_endpoint(corruption):
    raw = _raw()
    if corruption == "previous_endpoint":
        raw.loc["2016-10-02T23:00:00Z", "interpolated"] = "other | " + COLUMN
    elif corruption == "calibration_flag":
        raw.loc["2016-12-15T12:00:00Z", "interpolated"] = COLUMN + ";"
    elif corruption == "missing":
        raw.loc["2017-01-02T12:00:00Z", COLUMN] = np.nan
    elif corruption == "negative":
        raw.loc["2017-01-02T12:00:00Z", COLUMN] = 0.0
    else:
        raw = raw.drop(pd.Timestamp("2017-01-02T12:00:00Z"))

    with pytest.raises(ValueError, match="quality|hourly"):
        measured.select_measured_windows(raw)


def test_calendar_windows_include_full_history_and_four_distinct_seasons():
    windows = measured.select_measured_windows(_raw())

    assert [window.test_start.strftime("%Y-%m-%d") for window in windows] == [
        "2017-01-01",
        "2017-04-01",
        "2017-07-01",
        "2017-10-01",
    ]
    for window in windows:
        assert len(window.frame) == 2352
        assert window.calibration_start - window.train_start == pd.Timedelta(days=60)
        assert window.test_start - window.calibration_start == pd.Timedelta(days=30)
        assert window.test_end - window.test_start == pd.Timedelta(days=7)
        assert window.frame.pv_quality_good.all()
        assert window.metadata["relevant_interpolation_count"] == 0
        assert (
            window.metadata["selection_basis"]
            == "calendar_and_source_quality_before_strategy_results"
        )


def test_loader_verifies_pinned_content_and_never_overwrites_cache(tmp_path, monkeypatch):
    csv_name = "household_data_60min_singleindex.csv"
    metadata = {
        "version": "2020-04-15",
        "schemas": {
            "60min": {
                "fields": [
                    {"name": "utc_timestamp"},
                    {"name": "interpolated"},
                    {"name": COLUMN, "unit": "kWh", "opsd-properties": {"Feed": "pv"}},
                ]
            }
        },
    }
    contents = {
        "README.md": b"OPSD household data fixture\n",
        "datapackage.json": json.dumps(metadata).encode(),
        csv_name: (
            f"utc_timestamp,interpolated,{COLUMN}\n"
            "2017-01-01T00:00:00Z,,100\n2017-01-01T01:00:00Z,,102\n"
        ).encode(),
    }
    for name, content in contents.items():
        (tmp_path / name).write_bytes(content)
    monkeypatch.setattr(
        measured,
        "_RAW_HASHES",
        {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()},
    )
    before = {name: (tmp_path / name).stat().st_mtime_ns for name in contents}

    raw = measured.load_opsd(tmp_path)

    assert raw[COLUMN].tolist() == [100.0, 102.0]
    assert str(raw.index.tz) == "UTC"
    assert raw.attrs["raw_sha256"][csv_name] == hashlib.sha256(contents[csv_name]).hexdigest()
    assert before == {name: (tmp_path / name).stat().st_mtime_ns for name in contents}
    (tmp_path / csv_name).write_bytes(contents[csv_name] + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        measured.load_opsd(tmp_path)
    assert (tmp_path / csv_name).read_bytes() == contents[csv_name] + b"\n"


def test_original_meter_uses_only_actual_prior_boundary_samples_without_interpolation():
    index = pd.to_datetime(
        [
            "2016-10-01T23:59:30Z",
            "2016-10-02T00:59:30Z",
            "2016-10-02T01:00:30Z",
            "2016-10-02T01:59:30Z",
        ],
        utc=True,
    )
    raw = pd.DataFrame({COLUMN: [100.0, 102.0, 104.0, 105.0]}, index=index)
    raw.attrs["original_meter_readings"] = True

    frame = measured.build_measured_frame(raw)

    assert frame.index[0] == pd.Timestamp("2016-10-02T00:00:00Z")
    assert frame.pv_kw.iloc[:2].tolist() == [1.0, 1.5]
    assert frame.meter_end_time.iloc[0] == pd.Timestamp("2016-10-02T00:59:30Z")
    assert frame.meter_end_age_seconds.iloc[0] == 30.0
    assert frame.pv_quality_good.iloc[:2].all()
    assert frame.attrs["publisher_preprocessing_applied"] is False
    assert frame.attrs["interpolation_applied"] is False


def test_original_meter_rejects_stale_boundary_even_if_later_reading_looks_usable():
    index = pd.to_datetime(
        [
            "2016-10-01T23:59:30Z",
            "2016-10-02T00:56:00Z",
            "2016-10-02T01:00:30Z",
            "2016-10-02T01:59:30Z",
        ],
        utc=True,
    )
    raw = pd.DataFrame({COLUMN: [100.0, 102.0, 104.0, 105.0]}, index=index)
    raw.attrs["original_meter_readings"] = True

    frame = measured.build_measured_frame(raw)

    assert not frame.pv_quality_good.iloc[0]
    assert frame.meter_end_age_seconds.iloc[0] == 240.0
    assert frame.meter_end_time.iloc[0] < frame.index[0] + pd.Timedelta(hours=1)
