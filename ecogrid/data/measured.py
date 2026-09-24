"""Quality-selected OPSD original meter data and auditable processed-source QA.

The benchmark uses the original residential6 PV feed, decoded with the
publisher's little-endian uint32 UTC seconds / float32 kWh record format.
Actual readings at or before hourly boundaries give interval energy without
interpolation, filtering, normalization, or clock shifts. Each endpoint must
be <=180 seconds old. The measured delta is assigned to the nominal UTC hour;
actual endpoint times and elapsed seconds remain available for audit.

OPSD's 2020-04-15 hourly CSV stores cumulative kWh, using the last minute
reading of each left-labelled hourly bin. Backward differences are nominal
hourly energy: C[t] - C[t-1h], approximately spanning t-1min to t+59min.
We retain the published UTC bin labels and divide by one hour for mean kW.

The publisher regularizes gaps <=15 minutes before attaching gap markers,
and also applies whole-series outlier filtering. Consequently an unmarked
row is NOT proof of an uninterpolated or historically available observation.
The simulated two-hour availability delay covers bin closure and short
regularization; it does not undo retrospective publisher outlier filtering.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ecogrid.data.benchmark import _fixed_load_and_price

SOURCE_URL = "https://data.open-power-system-data.org/household_data/2020-04-15/"
PROCESSING_URL = "https://github.com/isc-konstanz/household_data/blob/2020-04-15/"
DEFAULT_PV_COLUMN = "DE_KN_residential6_pv"
_CSV_NAME = "household_data_60min_singleindex.csv"
_ORIGINAL_URL = SOURCE_URL + "original_data/original_data.zip"
_ORIGINAL_NAME = "residential6_pv_feed_43_current.MYD"
_ORIGINAL_MEMBER = "original_data/DE_KN_residential_006/phptimeseries/feed_43.MYD"
_ORIGINAL_HASH = "e4607b7971c98026cabaea253f181e3b9b90efbcbe91cb784c41653d0b87e73c"
_ORIGINAL_SIZE = 9413424
_ORIGINAL_CRC32 = 3286998230
_ARCHIVE_SIZE = 516671538
# ZIP central directory: local header offset 487123806, header 30, filename 61.
_COMPRESSED_START = 487123897
_COMPRESSED_SIZE = 2870309
_RAW_HASHES = {
    "README.md": "b3c7b45c40f5e0a3e9a6505b61a40132a0b5b6dc63fd45432f0cd1fa8ec7a01f",
    "datapackage.json": "93500241f4bb19f84f2668b100f53ec531e0c7dcbfa5d8759425c1375d588670",
    _CSV_NAME: "15956440b27465686eae2abf5895cbc3dab80f934a891bcb2f4f944428e97d60",
}
# Calendar anchors fixed after source-quality inspection and before strategy runs.
_TEST_WEEKS = (
    ("winter", "2017-01-01"),
    ("spring", "2017-04-01"),
    ("summer", "2017-07-01"),
    ("autumn", "2017-10-01"),
)


def _validate_hourly(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC":
        raise ValueError("measured data requires UTC hourly timestamps")
    if len(index) == 0 or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("measured data requires increasing, unique hourly timestamps")
    if not (index[1:] - index[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("measured data must retain contiguous hourly timestamps")
    if not (index == index.floor("h")).all():
        raise ValueError("measured data must retain whole-hour bin labels")


def _read_snapshot(cache: Path, name: str, expected_hash: str) -> bytes:
    path = cache / name
    if path.exists():
        content = path.read_bytes()
    else:
        last_error: requests.RequestException | None = None
        for _ in range(2):
            try:
                response = requests.get(SOURCE_URL + name, timeout=30)
                response.raise_for_status()
                content = response.content
                break
            except requests.RequestException as error:
                last_error = error
        else:
            raise RuntimeError(f"OPSD download failed for {name} after 2 attempts") from last_error
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"OPSD SHA256 mismatch for {path}; existing caches are never overwritten")
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
    return content


def load_opsd(cache_dir: str | Path = "data/raw/opsd") -> pd.DataFrame:
    """Load the pinned hourly source and metadata, preserving cumulative values.

    Only README.md, datapackage.json and the 15 MB hourly CSV are downloaded.
    Every cached byte stream is verified; a mismatch fails without replacement.
    The hourly CSV hash also matches the publisher's release checksums.txt.
    """
    cache = Path(cache_dir)
    snapshots = {name: _read_snapshot(cache, name, digest) for name, digest in _RAW_HASHES.items()}
    metadata = json.loads(snapshots["datapackage.json"])
    if metadata.get("version") != "2020-04-15":
        raise ValueError("unexpected OPSD release version")
    fields = metadata["schemas"]["60min"]["fields"]
    pv_fields = [field for field in fields if "_pv" in field["name"]]
    if not pv_fields or any(field.get("unit") != "kWh" for field in pv_fields):
        raise ValueError("OPSD hourly PV metadata must specify cumulative kWh")
    columns = ["utc_timestamp", "interpolated", *[field["name"] for field in pv_fields]]
    raw = pd.read_csv(cache / _CSV_NAME, usecols=columns, dtype={"interpolated": "string"})
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.pop("utc_timestamp"), utc=True), name="time")
    raw["interpolated"] = raw["interpolated"].fillna("")
    _validate_hourly(raw.index)
    raw.attrs = {
        "source": "opsd_household_data",
        "source_url": SOURCE_URL,
        "source_version": "2020-04-15",
        "license": "CC-BY-4.0",
        "attribution": (
            "Open Power System Data. 2020. Data Package Household Data. Version 2020-04-15."
        ),
        "source_unit": "cumulative kWh, last minute reading in each hourly bin",
        "raw_cache_dir": str(cache.resolve()),
        "raw_sha256": dict(_RAW_HASHES),
        "csv_publisher_checksum_url": PROCESSING_URL + "checksums.txt",
        "processing_source_url": PROCESSING_URL + "processing.ipynb",
        "pv_columns": [field["name"] for field in pv_fields],
    }
    return raw


def _original_range(bounds: tuple[int, int]) -> bytes:
    start, end = bounds
    for attempt in range(3):
        try:
            with requests.get(
                _ORIGINAL_URL,
                headers={"Range": f"bytes={start}-{end - 1}"},
                timeout=30,
                stream=True,
            ) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{end - 1}/{_ARCHIVE_SIZE}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("OPSD archive did not honor the exact bounded byte range")
                content = bytearray()
                for chunk in response.iter_content(16384):
                    content.extend(chunk)
                    if len(content) > end - start:
                        raise ValueError("OPSD original byte range exceeded its declared size")
                if len(content) != end - start:
                    raise ValueError("OPSD original byte range was truncated")
                return bytes(content)
        except requests.RequestException:
            if attempt == 2:
                raise
    raise RuntimeError("unreachable original data download state")


def load_opsd_original_pv(cache_dir: str | Path = "data/raw/opsd") -> pd.DataFrame:
    """Load untouched cumulative PV readings using ~2.9 MB of archive ranges.

    The exact ZIP member, compressed offsets, CRC32 and SHA256 are pinned to
    the official release. Existing cached bytes are never replaced. The SHA256
    is computed from the original member, not a publisher-signed checksum.
    """
    cache = Path(cache_dir)
    for name in ("README.md", "datapackage.json"):
        _read_snapshot(cache, name, _RAW_HASHES[name])
    path = cache / _ORIGINAL_NAME
    if path.exists():
        content = path.read_bytes()
    else:
        end = _COMPRESSED_START + _COMPRESSED_SIZE
        ranges = [
            (start, min(start + 131072, end)) for start in range(_COMPRESSED_START, end, 131072)
        ]
        with ThreadPoolExecutor(max_workers=4) as executor:
            compressed = b"".join(executor.map(_original_range, ranges))
        decoder = zlib.decompressobj(-15)
        content = decoder.decompress(compressed, _ORIGINAL_SIZE + 1)
        if not decoder.eof or decoder.unconsumed_tail:
            raise ValueError("OPSD original ZIP member exceeds its declared size or is incomplete")
    if (
        len(content) != _ORIGINAL_SIZE
        or zlib.crc32(content) != _ORIGINAL_CRC32
        or hashlib.sha256(content).hexdigest() != _ORIGINAL_HASH
    ):
        raise ValueError("OPSD original member SHA256/CRC32 mismatch; cache is not overwritten")
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
    # Official household/read.py read_feed uses struct.unpack('<xIf', record).
    records = np.frombuffer(content, dtype=[("padding", "u1"), ("time", "<u4"), ("kwh", "<f4")])
    index = pd.DatetimeIndex(pd.to_datetime(records["time"], unit="s", utc=True), name="time")
    raw = pd.DataFrame({DEFAULT_PV_COLUMN: records["kwh"].astype(float)}, index=index)
    raw = raw.loc[raw.index.year > 1970].copy()
    if not raw.index.is_monotonic_increasing or not raw.index.is_unique:
        raise ValueError("original PV timestamps must be increasing and unique; no deduplication")
    raw.attrs = {
        "original_meter_readings": True,
        "source": "opsd_original_cossmic_meter",
        "source_url": _ORIGINAL_URL,
        "source_version": "2020-04-15",
        "source_member": _ORIGINAL_MEMBER,
        "source_unit": "cumulative kWh",
        "source_encoding": (
            "9-byte <xIf records: padding, uint32 Unix UTC seconds, float32 cumulative kWh"
        ),
        "raw_cache_path": str(path.resolve()),
        "raw_sha256": {_ORIGINAL_NAME: _ORIGINAL_HASH},
        "raw_member_crc32": _ORIGINAL_CRC32,
        "source_decoder_url": PROCESSING_URL + "household/read.py#L144-L171",
        "source_feed_mapping_url": PROCESSING_URL + "conf/households.yml",
        "license": "CC-BY-4.0",
        "attribution": (
            "Open Power System Data. 2020. Data Package Household Data. Version 2020-04-15."
        ),
    }
    return raw


def _original_hourly_boundaries(raw: pd.DataFrame, column: str) -> pd.DataFrame:
    index = raw.index
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC":
        raise ValueError("original meter timestamps must be UTC")
    if len(index) < 2 or not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("original meter timestamps must be increasing and unique")
    if column not in raw:
        raise ValueError("original source requires the selected PV column")
    boundaries = pd.date_range(index[0].ceil("h"), index[-1].ceil("h"), freq="h", name="time")
    positions = index.searchsorted(boundaries, side="right") - 1
    times = index.take(positions)
    result = pd.DataFrame(
        {
            column: raw[column].iloc[positions].to_numpy(dtype=float),
            "interpolated": "",
            "meter_sample_time": times,
            "meter_age_seconds": (boundaries - times).total_seconds(),
        },
        index=boundaries,
    )
    result.attrs = dict(raw.attrs)
    return result


def build_measured_frame(
    raw: pd.DataFrame,
    column: str = DEFAULT_PV_COLUMN,
    pv_scale: float = 0.5,
    pv_capacity_kw: float = 5.0,
) -> pd.DataFrame:
    """Reconstruct power without filling, clipping or deleting invalid intervals.

    ``pv_scale`` is a fixed declared scenario scaling, never a test-maximum
    normalization. Capacity is a simulation bound, not a reported site rating.
    Consumers must reject rows where ``pv_quality_good`` is false and may
    access observations only when ``observation_available_at <= issue_time``.
    Use ``select_measured_windows`` to obtain dispatchable contiguous segments.
    """
    original = bool(raw.attrs.get("original_meter_readings", False))
    if original:
        raw = _original_hourly_boundaries(raw, column)
    _validate_hourly(raw.index)
    if column not in raw or "interpolated" not in raw or "_pv" not in column:
        raise ValueError("measured source requires a PV column and interpolation markers")
    if not np.isfinite(pv_scale) or pv_scale <= 0:
        raise ValueError("pv_scale must be finite and positive")
    if not np.isfinite(pv_capacity_kw) or pv_capacity_kw <= 0:
        raise ValueError("pv_capacity_kw must be finite and positive")
    energy = pd.to_numeric(raw[column], errors="raise").astype(float)
    pv = (energy.shift(-1) - energy if original else energy.diff()) * pv_scale
    markers = raw["interpolated"].fillna("").astype(str)
    relevant = markers.map(
        lambda value: column in {part.strip() for part in re.split(r"[|;]", value)}
    )
    endpoint_shift = -1 if original else 1
    endpoint_marked = relevant | relevant.shift(endpoint_shift, fill_value=False)
    finite_endpoints = np.isfinite(energy) & np.isfinite(energy.shift(endpoint_shift))
    good = finite_endpoints & ~endpoint_marked & np.isfinite(pv) & pv.between(0, pv_capacity_kw)
    load, price = _fixed_load_and_price(raw.index)
    frame = pd.DataFrame(
        {
            "pv_kw": pv,
            "load_kw": load,
            "price_eur_per_kwh": price,
            "source_interpolated": relevant,
            "endpoint_interpolated": endpoint_marked,
            "pv_quality_good": good,
            "observation_available_at": raw.index + pd.Timedelta(hours=2),
        },
        index=raw.index.copy(),
    )
    frame.attrs = {
        **raw.attrs,
        "source": "opsd_household_data_processed_measured_pv",
        "source_url": SOURCE_URL,
        "pv_source_column": column,
        "pv_observation_type": "measured_meter_energy_processed_by_publisher",
        "load_price_type": "deterministic_benchmark_assumptions",
        "pv_scale": float(pv_scale),
        "pv_capacity_kw": float(pv_capacity_kw),
        "capacity_basis": "declared_simulation_bound_not_verified_site_nameplate",
        "scaling_basis": "fixed_declared_scale_not_normalized_by_test_data",
        "pv_conversion": "(cumulative_kwh[t] - cumulative_kwh[t-1h]) / 1h * pv_scale",
        "index_semantics": "published UTC hour-start bins; nominal [t,t+1h)",
        "meter_interval_semantics": (
            "last-minute cumulative endpoints; approximately (t-1min,t+59min]"
        ),
        "observation_delay_hours": 2,
        "observation_availability_basis": (
            "simulated conservative release, not historical telemetry"
        ),
        "as_issued_measurements": False,
        "publisher_preprocessing_is_causal": False,
        "quality_policy": "reject relevant interpolation at both cumulative endpoints; retain gaps",
        "quality_caveat": (
            "Zero provided markers does not establish zero interpolation: publisher regularizes "
            "gaps <=15 minutes before marking, and applies retrospective whole-series outlier "
            "filters. The two-hour simulated release addresses only bin closure and short "
            "regularization; this is a processed measured-profile simulation, "
            "not an as-issued backtest."
        ),
        "regularization_source_url": PROCESSING_URL + "household/imputation.py#L32-L66",
        "retrospective_filter_source_url": PROCESSING_URL + "household/validation.py#L58-L110",
    }
    if original:
        frame["meter_start_time"] = raw.meter_sample_time
        frame["meter_end_time"] = raw.meter_sample_time.shift(-1)
        frame["meter_start_age_seconds"] = raw.meter_age_seconds
        frame["meter_end_age_seconds"] = raw.meter_age_seconds.shift(-1)
        frame["meter_elapsed_seconds"] = (
            frame.meter_end_time - frame.meter_start_time
        ).dt.total_seconds()
        fresh = frame.meter_start_age_seconds.le(180) & frame.meter_end_age_seconds.le(180)
        frame["pv_quality_good"] &= fresh & frame.meter_elapsed_seconds.gt(0)
        frame.attrs.update(
            {
                "source": "opsd_original_cossmic_meter",
                "source_url": _ORIGINAL_URL,
                "pv_observation_type": "measured_original_cumulative_meter",
                "pv_conversion": (
                    "(last_actual_kwh_at_or_before_t+1h - "
                    "last_actual_kwh_at_or_before_t) / 1h * pv_scale"
                ),
                "index_semantics": "nominal UTC hour-start bins [t,t+1h)",
                "meter_interval_semantics": (
                    "actual cumulative endpoints <= nominal boundaries, each <=180s old"
                ),
                "maximum_endpoint_age_seconds": 180,
                "publisher_preprocessing_applied": False,
                "publisher_preprocessing_is_causal": None,
                "interpolation_applied": False,
                "whole_series_statistics_applied": False,
                "quality_policy": (
                    "reject missing/stale endpoints and nonphysical interval deltas; no filling"
                ),
                "quality_caveat": (
                    "Original meter energy assigned to hourly bins with <=180s endpoint lag; "
                    "actual timestamps and elapsed seconds retained. No interpolation or publisher "
                    "validation is applied. Two-hour release is a simulation assumption because "
                    "historical transmission arrival times are not available."
                ),
            }
        )
    return frame


@dataclass(frozen=True)
class MeasuredWindow:
    """One contiguous train/calibration/test block; bounds are UTC and exclusive."""

    name: str
    frame: pd.DataFrame
    train_start: pd.Timestamp
    calibration_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metadata: dict


def select_measured_windows(
    raw: pd.DataFrame,
    train_days: int = 60,
    calibration_days: int = 30,
    test_days: int = 7,
    column: str = DEFAULT_PV_COLUMN,
    pv_scale: float = 0.5,
    pv_capacity_kw: float = 5.0,
    lookahead_hours: int = 24,
) -> list[MeasuredWindow]:
    """Validate four fixed 2017 calendar weeks, chosen before strategy results.

    Validation covers every training, calibration and evaluation row, including
    the raw predecessor needed for the first training difference. Failure does
    not search for replacement dates or silently compress the calendar.
    """
    for days in (train_days, calibration_days, test_days):
        if isinstance(days, bool) or not isinstance(days, int) or days < 1:
            raise ValueError("window lengths must be positive integer days")
    if (
        isinstance(lookahead_hours, bool)
        or not isinstance(lookahead_hours, int)
        or lookahead_hours < 0
    ):
        raise ValueError("lookahead_hours must be a nonnegative integer")
    full = build_measured_frame(raw, column, pv_scale, pv_capacity_kw)
    windows = []
    for name, test_date in _TEST_WEEKS:
        test_start = pd.Timestamp(test_date, tz="UTC")
        calibration_start = test_start - pd.Timedelta(days=calibration_days)
        train_start = calibration_start - pd.Timedelta(days=train_days)
        test_end = test_start + pd.Timedelta(days=test_days)
        frame_end = test_end + pd.Timedelta(hours=lookahead_hours)
        expected = pd.date_range(train_start, frame_end, inclusive="left", freq="h")
        frame = full.loc[(full.index >= train_start) & (full.index < frame_end)].copy()
        if not frame.index.equals(expected):
            raise ValueError(f"{name} quality failure: missing requested hourly intervals")
        bad = ~frame.pv_quality_good
        if bad.any():
            first_bad = frame.index[np.flatnonzero(bad.to_numpy())[0]]
            raise ValueError(
                f"{name} quality failure: {int(bad.sum())} invalid intervals; first {first_bad}"
            )
        metadata = {
            "name": name,
            "train_start": train_start.isoformat(),
            "calibration_start": calibration_start.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "frame_end": frame_end.isoformat(),
            "differentiation_predecessor": (
                train_start
                if raw.attrs.get("original_meter_readings")
                else train_start - pd.Timedelta(hours=1)
            ).isoformat(),
            "training_hours": train_days * 24,
            "calibration_hours": calibration_days * 24,
            "evaluation_hours": test_days * 24,
            "lookahead_hours": lookahead_hours,
            "total_hours": len(frame),
            "relevant_interpolation_count": int(frame.endpoint_interpolated.sum()),
            "invalid_interval_count": int(bad.sum()),
            "selection_basis": "calendar_and_source_quality_before_strategy_results",
            "pv_source_column": column,
            "pv_scale": float(pv_scale),
            "pv_capacity_kw": float(pv_capacity_kw),
        }
        if raw.attrs.get("original_meter_readings"):
            metadata.update(
                {
                    "maximum_endpoint_age_seconds": float(
                        max(frame.meter_start_age_seconds.max(), frame.meter_end_age_seconds.max())
                    ),
                    "minimum_actual_interval_seconds": float(frame.meter_elapsed_seconds.min()),
                    "maximum_actual_interval_seconds": float(frame.meter_elapsed_seconds.max()),
                    "publisher_preprocessing_applied": False,
                    "interpolation_applied": False,
                }
            )
        frame.attrs["selection"] = dict(metadata)
        windows.append(
            MeasuredWindow(
                name,
                frame,
                train_start,
                calibration_start,
                test_start,
                test_end,
                metadata,
            )
        )
    return windows
