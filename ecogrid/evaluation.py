"""Causal hourly policy replay and full-period hindsight evaluation.

Every timestamp labels the beginning of a one-hour UTC interval. Decisions
are made before that interval's PV is observed. The one-hour execution ledger
is the source of realized costs; plan losses are retained only as diagnostics.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ecogrid.diagnostics import ensemble_diagnostics
from ecogrid.dispatch import SystemConfig, solve_dispatch
from ecogrid.finance import cvar, value_at_risk
from ecogrid.simulation import execute_hour

_HORIZON = 24
_FRAME_COLUMNS = ("pv_kw", "load_kw", "price_eur_per_kwh")


def _validate_window(
    frame: pd.DataFrame, start: pd.Timestamp, hours: int, lookahead: int = 0
) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("frame must be a UTC hourly DataFrame")
    index = frame.index
    if str(index.tz) != "UTC" or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("frame index must be sorted, unique UTC timestamps")
    if len(index) == 0 or not (index[1:] - index[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("frame must contain contiguous hourly timestamps")
    if any(name not in frame.columns for name in _FRAME_COLUMNS):
        raise ValueError(f"frame needs columns {_FRAME_COLUMNS}")
    values = frame.loc[:, _FRAME_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("frame values must be finite and nonnegative")
    origin = pd.Timestamp(start)
    if str(origin.tz) != "UTC" or origin not in index:
        raise ValueError("start must be a UTC timestamp in frame")
    if isinstance(hours, bool) or not isinstance(hours, (int, np.integer)) or hours < 1:
        raise ValueError("hours must be a positive integer")
    first = int(index.get_loc(origin))
    if first + hours + lookahead > len(index):
        raise ValueError("frame lacks evaluation hours or required lookahead")
    return origin, index[first : first + hours]


def apply_stress(frame: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Scale PV from evaluation start onward by 0.2, preserving past history.

    The caller retains the original frame and the trained forecast and residual
    library. Later forecasts can see only already realized stressed PV.
    """
    origin, _ = _validate_window(frame, start, 1)
    stressed = frame.copy(deep=True)
    stressed.attrs = frame.attrs.copy()
    stressed.attrs["stress"] = True
    stressed.loc[stressed.index >= origin, "pv_kw"] *= 0.2
    return stressed


def _execute(
    config: SystemConfig,
    soc: float,
    timestamp: pd.Timestamp,
    observed: pd.Series,
    charge: float,
    discharge: float,
    execution_rule: str = "planned",
) -> dict:
    execution = execute_hour(
        config,
        soc=soc,
        load=float(observed.load_kw),
        pv=float(observed.pv_kw),
        price=float(observed.price_eur_per_kwh),
        requested_charge=charge,
        requested_discharge=discharge,
        execution_rule=execution_rule,
    )
    # Cash expense excludes the artificial ENS penalty and inventory valuation.
    execution["actual_expense_eur"] = float(execution["grid_cost_eur"]) + float(
        execution["battery_cost_eur"]
    )
    return execution


def run_policy(
    frame: pd.DataFrame,
    forecaster,
    residual_library,
    start: pd.Timestamp,
    hours: int,
    system: SystemConfig,
    *,
    policy: str = "deterministic",
    risk_weight: float = 0.5,
    alpha: float = 0.9,
    n_scenarios: int = 12,
    seed: int = 42,
    stress: bool = False,
    collect_diagnostics: list[dict] | None = None,
    tail_tiebreak: bool = True,
    scenario_mode: str = "sampled",
    execution_rule: str = "planned",
) -> pd.DataFrame:
    """Replan 24 hours at every issue, then physically execute only hour zero.

    Load and tariff are known over the planning horizon. Only PV observations
    strictly before each issue reach the forecaster; the current observation is
    passed to ``execute_hour`` after the dispatch solve. Optional all-lead
    diagnostics observe future actual PV only after that issue's action has
    been executed; those values never feed a decision. Forecast error is
    forecast minus actual; the daylight proxy is positive forecast or actual
    PV, not a solar-geometry daylight determination.
    """
    if policy not in ("deterministic", "stochastic", "cvar"):
        raise ValueError("policy must be deterministic, stochastic, or cvar")
    if scenario_mode not in ("sampled", "all_matching"):
        raise ValueError("scenario_mode must be sampled or all_matching")
    if execution_rule not in ("planned", "deficit_discharge"):
        raise ValueError("execution_rule must be planned or deficit_discharge")
    if not 0 <= risk_weight <= 1 or not 0 < alpha < 1:
        raise ValueError("risk_weight must be in [0,1] and alpha in (0,1)")
    if (
        isinstance(n_scenarios, bool)
        or not isinstance(n_scenarios, (int, np.integer))
        or n_scenarios < 1
    ):
        raise ValueError("n_scenarios must be a positive integer")
    origin, eval_index = _validate_window(frame, start, hours, lookahead=_HORIZON - 1)
    if origin == frame.index[0]:
        raise ValueError("evaluation needs PV history before start")
    working = apply_stress(frame, origin) if stress else frame
    capacity = float(getattr(forecaster, "pv_capacity_kw", frame.attrs.get("pv_capacity_kw", 5.0)))
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("PV capacity must be finite and positive")
    soc = float(system.initial_soc_kwh)
    records = []
    for issue in eval_index:
        valid = pd.date_range(issue, periods=_HORIZON, freq="h", tz="UTC")
        future = working.loc[valid, ["load_kw", "price_eur_per_kwh"]]
        # Both the forecaster and its input enforce the strict as-of boundary.
        history = working.loc[working.index < issue]
        point = np.asarray(forecaster.predict(history, valid), dtype=float)
        if point.shape != (_HORIZON,) or not np.isfinite(point).all() or (point < 0).any():
            raise ValueError("forecaster must return 24 finite nonnegative PV values")
        if stress:
            point = np.clip(point + 0.75 * (point > 0), 0.0, capacity)
        if policy == "deterministic":
            scenarios = point[None, :]
        else:
            # Timestamp-derived seed fixes each hour's ensemble across risk weights.
            hour_seed = int(
                np.random.SeedSequence([int(seed), int(issue.timestamp() // 3600)]).generate_state(
                    1
                )[0]
            )
            options = {"mode": scenario_mode} if scenario_mode != "sampled" else {}
            scenarios = np.asarray(
                residual_library.sample_scenarios(point, issue, n_scenarios, hour_seed, **options),
                dtype=float,
            )
        if scenarios.ndim != 2 or scenarios.shape[1] != _HORIZON or len(scenarios) == 0:
            raise ValueError("PV scenarios must have shape (scenarios, 24)")
        probabilities = np.full(len(scenarios), 1.0 / len(scenarios))
        candidate_count = (
            residual_library.matching_candidate_count(issue)
            if hasattr(residual_library, "matching_candidate_count")
            else int(np.count_nonzero(residual_library.origins.hour == issue.hour))
            if hasattr(residual_library, "origins")
            else np.nan
        )
        scenario_metadata = {
            "scenario_mode": scenario_mode,
            "scenario_candidate_count": candidate_count,
            "scenario_tail_equivalent_count": len(scenarios) * (1 - alpha),
            "scenario_unique_path_count": len(np.unique(scenarios, axis=0)),
            "scenario_sha256": hashlib.sha256(scenarios.tobytes()).hexdigest(),
            "execution_rule": execution_rule,
        }
        plan = solve_dispatch(
            future.load_kw.to_numpy(dtype=float),
            future.price_eur_per_kwh.to_numpy(dtype=float),
            scenarios,
            system,
            probabilities=probabilities,
            initial_soc=soc,
            risk_weight=risk_weight if policy == "cvar" else 0.0,
            alpha=alpha,
            tail_tiebreak=tail_tiebreak,
        )
        charge, discharge = plan.first_action
        record = _execute(system, soc, issue, working.loc[issue], charge, discharge, execution_rule)
        scenario_p10 = value_at_risk(scenarios[:, 0], 0.1)
        scenario_p90 = value_at_risk(scenarios[:, 0], 0.9)
        record.update(
            {
                **scenario_metadata,
                "policy": policy,
                "stress": bool(stress),
                "forecast_issue_time": issue,
                "forecast_valid_time": issue,
                "forecast_lead_hours": 0,
                "point_forecast_pv_kw": float(point[0]),
                "scenario_mean_pv_kw": float(np.mean(scenarios[:, 0])),
                "scenario_min_pv_kw": float(np.min(scenarios[:, 0])),
                "scenario_max_pv_kw": float(np.max(scenarios[:, 0])),
                "scenario_p10_pv_kw": scenario_p10,
                "scenario_p90_pv_kw": scenario_p90,
                "scenario_std_pv_kw": float(np.std(scenarios[:, 0])),
                "forecast_error_pv_kw": float(point[0] - record["pv_kw"]),
                "forecast_daylight_proxy": bool(point[0] > 1e-7 or record["pv_kw"] > 1e-7),
                "scenario_count": len(scenarios),
                "plan_expected_loss_eur": plan.expected_loss_eur,
                "plan_var_eur": plan.var_eur,
                "plan_cvar_eur": plan.cvar_eur,
                "plan_objective_eur": plan.objective_eur,
                "plan_primary_expected_loss_eur": getattr(plan, "primary_expected_loss_eur", None),
                "plan_primary_cvar_eur": getattr(plan, "primary_cvar_eur", None),
                "plan_tail_tiebreak_tolerance_eur": getattr(
                    plan, "tail_tiebreak_tolerance_eur", 0.0
                ),
                "plan_tail_tiebreak_used": getattr(plan, "tail_tiebreak_used", False),
                "solve_seconds": plan.solve_seconds,
                "max_residual": plan.max_residual,
            }
        )
        record.update(ensemble_diagnostics(scenarios[:, 0], record["pv_kw"]))
        records.append(record)
        soc = record["soc_kwh"]
        if collect_diagnostics is not None:
            # Keep observation of held-out PV downstream of solving/execution.
            actual_pv = working.loc[valid, "pv_kw"].to_numpy(dtype=float)
            for lead, valid_time in enumerate(valid):
                values = scenarios[:, lead]
                p10 = value_at_risk(values, 0.1)
                p90 = value_at_risk(values, 0.9)
                collect_diagnostics.append(
                    {
                        **scenario_metadata,
                        **ensemble_diagnostics(values, float(actual_pv[lead])),
                        "policy": policy,
                        "stress": bool(stress),
                        "forecast_issue_time": issue,
                        "forecast_valid_time": valid_time,
                        "forecast_lead_hours": lead,
                        "point_forecast_pv_kw": float(point[lead]),
                        "scenario_mean_pv_kw": float(np.mean(values)),
                        "scenario_min_pv_kw": float(np.min(values)),
                        "scenario_max_pv_kw": float(np.max(values)),
                        "scenario_p10_pv_kw": p10,
                        "scenario_p90_pv_kw": p90,
                        "scenario_count": len(scenarios),
                        "actual_pv_kw": float(actual_pv[lead]),
                        "forecast_error_pv_kw": float(point[lead] - actual_pv[lead]),
                        "forecast_daylight_proxy": bool(
                            point[lead] > 1e-7 or actual_pv[lead] > 1e-7
                        ),
                        "inside_p10_p90": bool(p10 <= actual_pv[lead] <= p90),
                        "inside_min_max": bool(np.min(values) <= actual_pv[lead] <= np.max(values)),
                    }
                )
    return pd.DataFrame(records, index=eval_index)


def perfect_foresight(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    hours: int,
    system: SystemConfig,
    *,
    execution_rule: str = "planned",
) -> pd.DataFrame:
    """Solve the entire evaluation period once using the actual PV path.

    Actions are re-executed through the same physical guard as policy runs and
    checked against the optimization trajectories and objective accounting.
    """
    _, eval_index = _validate_window(frame, start, hours)
    actual = frame.loc[eval_index]
    plan = solve_dispatch(
        actual.load_kw.to_numpy(dtype=float),
        actual.price_eur_per_kwh.to_numpy(dtype=float),
        actual.pv_kw.to_numpy(dtype=float)[None, :],
        system,
    )
    records = []
    soc = float(system.initial_soc_kwh)
    tolerance = max(system.feasibility_tolerance * 10, 1e-7)
    for i, timestamp in enumerate(eval_index):
        charge = float(plan.flows["charge"][0, i])
        discharge = float(plan.flows["discharge"][0, i])
        record = _execute(
            system, soc, timestamp, actual.loc[timestamp], charge, discharge, execution_rule
        )
        for recorded, planned in (
            ("charge_kw", "charge"),
            ("discharge_kw", "discharge"),
            ("soc_kwh", "soc"),
            ("grid_kw", "grid"),
            ("pv_used_kw", "pv_used"),
            ("ens_kwh", "shed"),
            ("curtailment_kwh", "curtailment"),
        ):
            if abs(record[recorded] - plan.flows[planned][0, i]) > tolerance:
                raise RuntimeError(f"Perfect foresight execution differs from plan: {recorded}")
        record.update(
            {
                "policy": "perfect_foresight",
                "stress": bool(frame.attrs.get("stress", False)),
                "plan_expected_loss_eur": plan.expected_loss_eur,
                "plan_var_eur": plan.var_eur,
                "plan_cvar_eur": plan.cvar_eur,
                "plan_objective_eur": plan.objective_eur,
                "solve_seconds": plan.solve_seconds if i == 0 else 0.0,
                "max_residual": plan.max_residual,
            }
        )
        records.append(record)
        soc = record["soc_kwh"]
    hourly = pd.DataFrame(records, index=eval_index)
    if abs(hourly.loss_eur.sum() - plan.expected_loss_eur) > tolerance:
        raise RuntimeError("Perfect foresight objective differs from executed losses")
    return hourly


def summarize(
    hourly: pd.DataFrame, alpha: float = 0.9, perfect_total: float | None = None
) -> dict[str, float | int]:
    """Report realized totals and empirical risk across hourly losses.

    EENS is mean unserved energy per sampled hour; LOLP is the fraction of
    sampled hours with nonzero unserved energy. The CVaR tail has fractional
    weight when the sample count times (1-alpha) is not an integer.
    """
    if hourly.empty:
        raise ValueError("hourly ledger cannot be empty")
    losses = hourly.loss_eur.to_numpy(dtype=float)
    ens = hourly.ens_kwh.to_numpy(dtype=float)
    emergency = hourly.get("emergency_discharge_kw", pd.Series(0.0, index=hourly.index))
    total = float(losses.sum())
    result = {
        "hours": len(hourly),
        "hourly_sample_count": len(hourly),
        "total_loss_eur": total,
        "total_emergency_discharge_kwh": float(emergency.sum()),
        "hours_with_emergency_discharge": int((emergency > 1e-7).sum()),
        "total_accounted_loss_eur": total,
        "total_actual_expense_eur": float(
            hourly.grid_cost_eur.sum() + hourly.battery_cost_eur.sum()
        ),
        "total_operating_cost_eur": float(hourly.operating_cost_eur.sum())
        if "operating_cost_eur" in hourly
        else float(
            hourly.grid_cost_eur.sum()
            + hourly.battery_cost_eur.sum()
            + hourly.shortage_cost_eur.sum()
        ),
        "total_grid_cost_eur": float(hourly.grid_cost_eur.sum()),
        "total_battery_cost_eur": float(hourly.battery_cost_eur.sum()),
        "total_shortage_cost_eur": float(hourly.shortage_cost_eur.sum()),
        "total_inventory_adjustment_eur": float(hourly.inventory_adjustment_eur.sum()),
        "mean_hourly_loss_eur": float(np.mean(losses)),
        "var_hourly_loss_eur": value_at_risk(losses, alpha),
        "cvar_hourly_loss_eur": cvar(losses, alpha),
        "upper_tail_equivalent_hours": len(hourly) * (1.0 - alpha),
        "total_ens_kwh": float(ens.sum()),
        "eens_kwh_per_hour": float(np.mean(ens)),
        "lolp": float(np.mean(ens > 1e-7)),
        "total_grid_import_kwh": float(hourly.grid_kw.sum()),
        "peak_grid_import_kw": float(hourly.grid_kw.max()),
        "total_curtailment_kwh": float(hourly.curtailment_kwh.sum()),
        "total_throughput_kwh": float(hourly.throughput_kwh.sum()),
        "total_solve_seconds": float(hourly.solve_seconds.sum()),
        "max_residual": float(hourly.max_residual.max()),
        "hours_with_loss": int(np.count_nonzero(losses > 1e-7)),
        "hours_with_physical_override": int(
            np.count_nonzero(
                (hourly.charge_override_kw.to_numpy(dtype=float) > 1e-7)
                | (hourly.discharge_override_kw.to_numpy(dtype=float) > 1e-7)
            )
        ),
        "total_physical_override_kwh": float(
            hourly.charge_override_kw.sum() + hourly.discharge_override_kw.sum()
        ),
    }
    if perfect_total is not None:
        result["perfect_foresight_loss_eur"] = float(perfect_total)
        result["regret_eur"] = total - float(perfect_total)
    return result


def _validate_ledger_index(hourly: pd.DataFrame, *, contiguous: bool = False) -> None:
    if (
        hourly.empty
        or not isinstance(hourly.index, pd.DatetimeIndex)
        or str(hourly.index.tz) != "UTC"
    ):
        raise ValueError("hourly must be a nonempty UTC-indexed ledger")
    index = hourly.index
    if (
        not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.floor("h"))
    ):
        raise ValueError("hourly ledger must have sorted, unique timestamps on UTC hour boundaries")
    if contiguous and not (index[1:] - index[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("24-hour block populations require a contiguous hourly ledger")


def _aggregate_blocks(hourly: pd.DataFrame, labels: pd.DatetimeIndex) -> pd.DataFrame:
    sums = {
        "loss_eur": "total_loss_eur",
        "operating_cost_eur": "total_operating_cost_eur",
        "grid_cost_eur": "total_grid_cost_eur",
        "battery_cost_eur": "total_battery_cost_eur",
        "shortage_cost_eur": "total_shortage_cost_eur",
        "inventory_adjustment_eur": "total_inventory_adjustment_eur",
        "ens_kwh": "total_ens_kwh",
        "grid_kw": "total_grid_import_kwh",
        "curtailment_kwh": "total_curtailment_kwh",
        "throughput_kwh": "total_throughput_kwh",
        "solve_seconds": "total_solve_seconds",
    }
    present = {column: label for column, label in sums.items() if column in hourly}
    blocks = hourly.groupby(labels)[list(present)].sum().rename(columns=present)
    if "grid_cost_eur" in hourly and "battery_cost_eur" in hourly:
        blocks["total_actual_expense_eur"] = (
            (hourly.grid_cost_eur + hourly.battery_cost_eur).groupby(labels).sum()
        )
    blocks["hours"] = hourly.groupby(labels).size()
    blocks["expected_hours"] = 24
    blocks["complete_block"] = blocks.hours == 24
    blocks["partial_block"] = ~blocks.complete_block
    blocks["peak_grid_import_kw"] = hourly.grid_kw.groupby(labels).max()
    blocks["hours_with_ens"] = (hourly.ens_kwh > 1e-7).groupby(labels).sum()
    return blocks


def daily_metrics(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by UTC date and retain partial days with explicit flags.

    Complete days contain all 24 distinct UTC hour intervals. Missing dates
    are not fabricated, and missing hours cannot make a complete day. The
    legacy operating-cost column includes the ENS penalty; actual expense
    includes only grid purchases and battery throughput expense.
    """
    _validate_ledger_index(hourly)
    daily = _aggregate_blocks(hourly, hourly.index.normalize())
    daily.index.name = "date_utc"
    return daily


def nonoverlapping_24h_metrics(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consecutive 24-hour blocks anchored at this run's first hour.

    A final partial block is retained and flagged. Gaps are rejected rather
    than compressed. Aggregate separate evaluation windows separately before
    pooling their complete block tables through ``risk_summary_blocks``.
    """
    _validate_ledger_index(hourly, contiguous=True)
    labels = hourly.index[0] + pd.to_timedelta(np.arange(len(hourly)) // 24 * 24, unit="h")
    blocks = _aggregate_blocks(hourly, labels)
    blocks.index.name = "block_start_utc"
    blocks["block_end_utc"] = blocks.index + pd.to_timedelta(blocks.hours.to_numpy(), unit="h")
    return blocks


_RISK_COLUMNS = {
    "actual_expense": "actual_expense_eur",
    "shortage_penalty": "shortage_cost_eur",
    "inventory_adjustment": "inventory_adjustment_eur",
    "adjusted_loss": "loss_eur",
    "ens": "ens_kwh",
}


def risk_summary_samples(
    samples: pd.DataFrame,
    alpha: float = 0.9,
    scale: str = "hour",
    excluded_partial_blocks: int = 0,
) -> pd.DataFrame:
    """Summarize already observed samples, including pooled disjoint windows.

    Rows are the supplied hourly or preaggregated block population. This
    helper performs no time compression, constructs no additional blocks,
    and makes no claim that observations are independent. Use
    ``risk_summary_blocks`` to exclude partial blocks before pooling them.
    """
    # Use the same alpha validation and lower discrete quantile as financial
    # risk even when a run is too short to yield any complete blocks.
    value_at_risk(np.array([0.0]), alpha)
    rows = []
    for metric, column in _RISK_COLUMNS.items():
        values = samples[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"risk samples must be finite: {column}")
        count = len(values)
        currency = "kWh" if metric == "ens" else "EUR"
        rows.append(
            {
                "scale": scale,
                "metric": metric,
                "unit": f"{currency}/{'hour' if scale == 'hour' else 'block'}",
                "alpha": alpha,
                "mean": float(np.mean(values)) if count else np.nan,
                "var": value_at_risk(values, alpha) if count else np.nan,
                "cvar": cvar(values, alpha) if count else np.nan,
                "sample_count": count,
                "tail_equivalent_count": count * (1.0 - alpha),
                "excluded_partial_blocks": excluded_partial_blocks,
            }
        )
    return pd.DataFrame(rows)


def risk_summary_blocks(
    blocks: pd.DataFrame, alpha: float = 0.9, *, scale: str = "utc_day"
) -> pd.DataFrame:
    """Compute empirical risk from already aggregated distinct run tables.

    ``blocks`` may concatenate tables from disjoint evaluation windows. Rows
    remain observed blocks, not independent draws: no independence or sample
    size inflation is assumed. Retained partial blocks are excluded and their
    count is disclosed. Empty populations have NaN estimates and zero counts.
    """
    if scale not in ("utc_day", "nonoverlapping_24h"):
        raise ValueError("block scale must be utc_day or nonoverlapping_24h")
    required = ["complete_block", "hours", *(f"total_{name}" for name in _RISK_COLUMNS.values())]
    missing = [name for name in required if name not in blocks]
    if missing:
        raise ValueError(f"block table lacks required columns: {missing}")
    flags = blocks.complete_block
    if flags.isna().any() or not flags.isin([True, False]).all():
        raise ValueError("complete_block must contain boolean flags")
    if (flags.astype(bool) & (blocks.hours != 24)).any():
        raise ValueError("complete blocks must contain exactly 24 hours")
    complete = flags.astype(bool)
    samples = blocks.loc[complete].rename(
        columns={f"total_{name}": name for name in _RISK_COLUMNS.values()}
    )
    return risk_summary_samples(samples, alpha, scale, int((~complete).sum()))


def risk_summary(hourly: pd.DataFrame, alpha: float = 0.9) -> pd.DataFrame:
    """Tidy realized risk at hourly, complete UTC-day and 24-hour-block scales.

    Each row reports a distinct empirical population, its units and effective
    tail size. These are realized outcomes, not the overlapping 24-hour plan
    scenario-loss distributions optimized at each issue. Full-period perfect
    foresight bounds adjusted total loss only, not every row in this table.
    """
    _validate_ledger_index(hourly, contiguous=True)
    samples = hourly.assign(actual_expense_eur=hourly.grid_cost_eur + hourly.battery_cost_eur)
    return pd.concat(
        [
            risk_summary_samples(samples, alpha, "hour"),
            risk_summary_blocks(daily_metrics(hourly), alpha, scale="utc_day"),
            risk_summary_blocks(
                nonoverlapping_24h_metrics(hourly), alpha, scale="nonoverlapping_24h"
            ),
        ],
        ignore_index=True,
    )
