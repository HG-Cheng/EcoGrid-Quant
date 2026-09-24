"""Fixed calendar measured-PV experiment; small one-factor sensitivities."""

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ecogrid.cli import _sha, _versions, _write_json
from ecogrid.data.measured import load_opsd_original_pv, select_measured_windows
from ecogrid.diagnostics import (
    GROUP_KEYS,
    compare_actions,
    forecast_summary,
    tail_contributions,
    write_figures,
)
from ecogrid.dispatch import SystemConfig
from ecogrid.evaluation import (
    apply_stress,
    daily_metrics,
    nonoverlapping_24h_metrics,
    perfect_foresight,
    risk_summary_blocks,
    risk_summary_samples,
    run_policy,
    summarize,
)
from ecogrid.forecast import calibrate_residuals, fit_forecaster
from ecogrid.simulation import execute_hour


def _experiments(settings: dict) -> list[dict]:
    """No outcome input: dates, windows and one-factor variations are predeclared."""
    groups = []
    common = {"grid_limit_kw": settings["system"]["grid_kw"], "seed": settings["seed"]}
    for window in settings["windows"]:
        groups.append(
            {
                **common,
                "window": window["name"],
                "experiment": "main",
                "hours": settings["evaluation_days"] * 24,
            }
        )
    for window in settings["representative_windows"]:
        for grid in settings["grid_limits_kw"]:
            groups.append(
                {
                    **common,
                    "window": window,
                    "experiment": "grid_capacity",
                    "grid_limit_kw": grid,
                    "hours": settings["sensitivity_hours"],
                }
            )
        for seed in settings["additional_seeds"]:
            groups.append(
                {
                    **common,
                    "window": window,
                    "experiment": "seed_repeat",
                    "seed": seed,
                    "hours": settings["sensitivity_hours"],
                }
            )
        groups.append(
            {**common, "window": window, "experiment": "stress", "hours": settings["stress_hours"]}
        )
        groups.append(
            {
                **common,
                "window": window,
                "experiment": "pure_cvar_ties",
                "hours": settings["sensitivity_hours"],
            }
        )
    return groups


def _policies(experiment: str, weight: float) -> list[tuple[str, str, float, bool]]:
    base = [
        ("perfect_foresight", "perfect_foresight", 0.0, True),
        ("deterministic", "deterministic", 0.0, True),
        ("stochastic", "stochastic", 0.0, True),
        (f"cvar_{weight:g}", "cvar", weight, True),
    ]
    if experiment == "pure_cvar_ties":
        return [p for p in base if p[0] != "deterministic"] + [
            ("cvar_1_raw", "cvar", 1.0, False),
            ("cvar_1", "cvar", 1.0, True),
        ]
    return base


def _run_id(group: dict, policy: str) -> str:
    return (
        f"{group['experiment']}_{group['window']}_g{group['grid_limit_kw']:g}"
        f"_s{group['seed']}_{policy}"
    ).replace(".", "p")


def _assemble(output: Path, groups: list[dict], settings: dict, *, render: bool = True) -> None:
    """Aggregate distinct windows only after all requested policies complete."""
    ledgers, days, blocks, summary_rows, diagnostics = [], [], [], [], []
    for group in groups:
        perfect_id = _run_id(group, "perfect_foresight")
        perfect = pd.read_csv(
            output / "runs" / f"{perfect_id}.csv", index_col="time", parse_dates=["time"]
        )
        perfect_total = float(perfect.loss_eur.sum())
        for label, _, weight, tiebreak in _policies(group["experiment"], settings["risk_weight"]):
            run_id = _run_id(group, label)
            hourly = pd.read_csv(
                output / "runs" / f"{run_id}.csv", index_col="time", parse_dates=["time"]
            )
            meta = {**group, "policy": label, "risk_weight": weight, "tail_tiebreak": tiebreak}
            stats = summarize(hourly, settings["alpha"], perfect_total)
            if stats["regret_eur"] < -1e-5:
                raise RuntimeError(f"Incomparable or invalid full-period regret: {run_id}")
            summary_rows.append({**meta, **stats})
            daily = daily_metrics(hourly).reset_index()
            block = nonoverlapping_24h_metrics(hourly).reset_index()
            ledger = hourly.reset_index()
            for table in (ledger, daily, block):
                for name, value in meta.items():
                    if name != "hours":
                        table[name] = value
            ledgers.append(ledger)
            days.append(daily)
            blocks.append(block)
            path = output / "runs" / f"{run_id}_forecast.csv"
            if label == "stochastic" and not path.exists():
                raise ValueError(f"Required forecast diagnostics missing: {run_id}")
            if path.exists():
                diagnostic = pd.read_csv(path)
                for name, value in group.items():
                    diagnostic[name] = value
                diagnostics.append(diagnostic)
    hourly = pd.concat(ledgers, ignore_index=True)
    daily = pd.concat(days, ignore_index=True)
    block24 = pd.concat(blocks, ignore_index=True)
    diag = pd.concat(diagnostics, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    hourly.to_csv(output / "hourly.csv", index=False)
    daily.to_csv(output / "daily.csv", index=False)
    block24.to_csv(output / "blocks24.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)

    # Pool separate calendar windows only within exactly the same experiment,
    # capacity, seed and policy; seed repeats never inflate the 28-day population.
    pool_keys = ["experiment", "grid_limit_kw", "seed", "policy"]
    risk_rows, tails = [], []
    for key, population in hourly.groupby(pool_keys, sort=False):
        info = dict(zip(pool_keys, key, strict=True))
        mask_daily = np.ones(len(daily), dtype=bool)
        mask_block = np.ones(len(block24), dtype=bool)
        for name, value in info.items():
            mask_daily &= (daily[name] == value).to_numpy()
            mask_block &= (block24[name] == value).to_numpy()
        samples = population.assign(
            actual_expense_eur=population.grid_cost_eur + population.battery_cost_eur
        )
        metrics = pd.concat(
            [
                risk_summary_samples(samples, settings["alpha"]),
                risk_summary_blocks(daily.loc[mask_daily], settings["alpha"], scale="utc_day"),
                risk_summary_blocks(
                    block24.loc[mask_block], settings["alpha"], scale="nonoverlapping_24h"
                ),
            ],
            ignore_index=True,
        )
        for name, value in info.items():
            metrics[name] = value
        risk_rows.append(metrics)
        tails.append({**info, "scale": "hour", **tail_contributions(population, settings["alpha"])})
        day_population = daily.loc[mask_daily].rename(
            columns={
                f"total_{name}": name
                for name in (
                    "loss_eur",
                    "grid_cost_eur",
                    "battery_cost_eur",
                    "shortage_cost_eur",
                    "inventory_adjustment_eur",
                )
            }
        )
        tails.append(
            {**info, "scale": "utc_day", **tail_contributions(day_population, settings["alpha"])}
        )
    risks = pd.concat(risk_rows, ignore_index=True)
    risks.to_csv(output / "risk_summary.csv", index=False)
    pd.DataFrame(tails).to_csv(output / "tail_attribution.csv", index=False)
    actions = compare_actions(hourly)
    actions.to_csv(output / "action_comparison.csv", index=False)
    forecast_summary(diag).to_csv(output / "forecast_diagnostics.csv", index=False)

    # Verify that each matched stochastic policy received the same information.
    for _, group in hourly.groupby(GROUP_KEYS, sort=False):
        ref = group.loc[group.policy == "stochastic"].set_index("time")
        for policy in group.policy.unique():
            if not policy.startswith("cvar"):
                continue
            other = group.loc[group.policy == policy].set_index("time")
            for column in (
                "point_forecast_pv_kw",
                "scenario_mean_pv_kw",
                "scenario_min_pv_kw",
                "scenario_max_pv_kw",
                "scenario_p10_pv_kw",
                "scenario_p90_pv_kw",
            ):
                np.testing.assert_allclose(ref[column], other[column], atol=1e-12, rtol=0)

    representative = hourly.loc[
        (hourly.experiment == "main") & hourly.window.isin(settings["representative_windows"])
    ].copy()
    times = pd.to_datetime(representative.time, utc=True)
    start = times.groupby(representative.window).transform("min")
    representative = representative.loc[times < start + pd.Timedelta(hours=48)]
    representative.to_csv(output / "representative_hourly.csv", index=False)
    if render:
        write_figures(hourly, daily, risks, diag, output, settings["representative_windows"])
    print(
        risks.loc[
            (risks.experiment == "main") & (risks.metric == "adjusted_loss"),
            ["policy", "scale", "mean", "cvar", "sample_count", "tail_equivalent_count"],
        ].to_string(index=False),
        flush=True,
    )


def run_study(settings: dict, output: Path) -> None:
    """Validate and freeze every window before any optimization, then checkpoint."""
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir(exist_ok=True)
    if settings["stress"] != {"pv_multiplier": 0.2, "positive_forecast_bias_kw": 0.75}:
        raise ValueError("Study stress must match the common predeclared execution protocol")
    if settings["risk_weight"] != 0.5:
        raise ValueError("This fixed study declares lambda=.5; do not tune it on these results")
    raw = load_opsd_original_pv(root / settings["data"]["cache_dir"])
    windows = select_measured_windows(
        raw,
        train_days=settings["train_days"],
        calibration_days=settings["calibration_days"],
        test_days=settings["evaluation_days"],
        column=settings["data"]["column"],
        pv_scale=settings["data"]["pv_scale"],
        pv_capacity_kw=settings["data"]["pv_capacity_kw"],
    )
    configured = {w["name"]: pd.Timestamp(w["start"]) for w in settings["windows"]}
    if {w.name: w.test_start for w in windows} != configured:
        raise ValueError("Calendar windows differ from predeclared configuration")
    groups = _experiments(settings)
    code = {str(p.relative_to(root)): _sha(p) for p in sorted((root / "ecogrid").rglob("*.py"))}
    # Reporting changes do not invalidate completed optimizations; their hash is
    # still recorded separately in the final manifest for figure provenance.
    model_code = {
        name: digest for name, digest in code.items() if not name.endswith("diagnostics.py")
    }
    payload = {
        "settings": settings,
        "windows": [w.metadata for w in windows],
        "source": raw.attrs,
        "derived_data": windows[0].frame.attrs,
        "run_code_sha256": model_code,
        "groups": groups,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run_fingerprint"] != fingerprint:
            raise ValueError(
                "Existing study has different code/data/protocol; use a new output directory"
            )
    else:
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_fingerprint": fingerprint,
            "completed_runs": {},
            "versions": _versions(),
            "protocol": payload,
            "system": asdict(SystemConfig(**settings["system"])),
            "risk_population": "observed hours and distinct complete UTC days; no IID claim",
            "tail_tiebreak": "lambda=1 only: minimize E subject to primary CVaR + EUR1e-7",
            "solver": {
                "name": "appsi_highs",
                "mip_rel_gap": 1e-8,
                "random_seed": 0,
                "threads": "default",
                "acceptance": "optimal + residual check",
            },
        }
        _write_json(output / "protocol.json", payload)  # written before the first strategy solve
    manifest["status"] = "running"
    manifest["code_sha256"] = code
    _write_json(manifest_path, manifest)
    try:
        prepared = {}
        for window in windows:
            frame = window.frame
            train = frame.loc[
                (frame.index < window.calibration_start)
                & (frame.observation_available_at <= window.calibration_start)
            ]
            model = fit_forecaster(train)
            residuals = calibrate_residuals(
                model,
                frame.loc[frame.index < window.test_start],
                window.calibration_start,
                window.test_start,
            )
            prepared[window.name] = (window, model, residuals)
            _write_json(
                output / f"{window.name}_calibration.json",
                {
                    "training_usable_hours": len(train),
                    "calibration_blocks": len(residuals.origins),
                    "last_residual_origin": residuals.origins.max(),
                    "freeze_time": window.test_start,
                    "trained_hourly_mean_kw": model.hourly_mean_kw.tolist(),
                    "dataframe_sha256": hashlib.sha256(frame.to_csv().encode()).hexdigest(),
                },
            )
        for group in groups:
            window, forecast, residuals = prepared[group["window"]]
            config = replace(SystemConfig(**settings["system"]), grid_kw=group["grid_limit_kw"])
            stress = group["experiment"] == "stress"
            for label, policy, weight, tiebreak in _policies(
                group["experiment"], settings["risk_weight"]
            ):
                run_id = _run_id(group, label)
                path = output / "runs" / f"{run_id}.csv"
                if run_id in manifest["completed_runs"]:
                    if (
                        not path.exists()
                        or _sha(path) != manifest["completed_runs"][run_id]["ledger_sha256"]
                    ):
                        raise ValueError(f"Checkpoint ledger changed: {run_id}")
                    record = manifest["completed_runs"][run_id]
                    if "forecast_sha256" in record:
                        forecast_path = output / "runs" / f"{run_id}_forecast.csv"
                        if (
                            not forecast_path.exists()
                            or _sha(forecast_path) != record["forecast_sha256"]
                        ):
                            raise ValueError(f"Checkpoint forecast diagnostics changed: {run_id}")
                    continue
                print(f"[{run_id}] {group['hours']}h", flush=True)
                diagnostics: list[dict] = []
                if policy == "perfect_foresight":
                    actual = (
                        apply_stress(window.frame, window.test_start) if stress else window.frame
                    )
                    hourly = perfect_foresight(actual, window.test_start, group["hours"], config)
                else:
                    hourly = run_policy(
                        window.frame,
                        forecast,
                        residuals,
                        window.test_start,
                        group["hours"],
                        config,
                        policy=policy,
                        risk_weight=weight,
                        alpha=settings["alpha"],
                        n_scenarios=settings["n_scenarios"],
                        seed=group["seed"],
                        stress=stress,
                        tail_tiebreak=tiebreak,
                        collect_diagnostics=diagnostics if policy == "stochastic" else None,
                    )
                hourly.to_csv(path, index_label="time")
                record = {
                    "ledger_sha256": _sha(path),
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
                if diagnostics:
                    forecast_path = output / "runs" / f"{run_id}_forecast.csv"
                    pd.DataFrame(diagnostics).to_csv(forecast_path, index=False)
                    record["forecast_sha256"] = _sha(forecast_path)
                manifest["completed_runs"][run_id] = record
                _write_json(manifest_path, manifest)
        _assemble(output, groups, settings)
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        _write_json(manifest_path, manifest)
        raise


def _verify_original_ledger(
    path: Path, digest: str, frame: pd.DataFrame, system: SystemConfig
) -> pd.DataFrame:
    """Check immutable baseline bytes, actual inputs, physics and full cost accounting."""
    if _sha(path) != digest:
        raise ValueError(f"Original ledger hash differs: {path}")
    ledger = pd.read_csv(path, index_col="time", parse_dates=["time"])
    np.testing.assert_allclose(
        ledger[["pv_kw", "load_kw", "price_eur_per_kwh"]],
        frame.loc[ledger.index, ["pv_kw", "load_kw", "price_eur_per_kwh"]],
        atol=1e-12,
        rtol=0,
    )
    soc = system.initial_soc_kwh
    for row in ledger.itertuples():
        executed = execute_hour(
            system,
            soc=soc,
            load=row.load_kw,
            pv=row.pv_kw,
            price=row.price_eur_per_kwh,
            requested_charge=row.requested_charge_kw,
            requested_discharge=row.requested_discharge_kw,
        )
        for name in (
            "charge_kw",
            "discharge_kw",
            "soc_kwh",
            "grid_kw",
            "ens_kwh",
            "pv_used_kw",
            "curtailment_kwh",
            "grid_cost_eur",
            "battery_cost_eur",
            "shortage_cost_eur",
            "inventory_adjustment_eur",
            "loss_eur",
        ):
            if abs(float(executed[name]) - getattr(row, name)) > 1e-8:
                raise ValueError(
                    f"Original execution no longer reproduced: {path}, {row.Index}, {name}"
                )
        soc = float(executed["soc_kwh"])
    return ledger


def run_mechanism_study(settings: dict, output: Path) -> None:
    """One-factor follow-up, explicitly post hoc; original study stays immutable."""
    started = perf_counter()
    root = Path(__file__).resolve().parents[1]
    baseline = (root / settings["baseline_results"]).resolve()
    if output.resolve() == baseline or baseline in output.resolve().parents:
        raise ValueError("Post-hoc output must be outside the original result directory")
    base = json.loads((root / settings["base_config"]).read_text(encoding="utf-8"))
    original = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    if original["status"] != "complete" or original["protocol"]["settings"] != base:
        raise ValueError("Need a complete, matching original measured study")
    original_code = {
        name.replace("\\", "/"): digest for name, digest in original["code_sha256"].items()
    }
    for name in (
        "ecogrid/dispatch.py",
        "ecogrid/finance.py",
        "ecogrid/data/measured.py",
        "ecogrid/data/benchmark.py",
    ):
        if _sha(root / name) != original_code[name]:
            raise ValueError(f"Cannot reuse baseline after physical/data/risk change: {name}")
    if settings["scenario_modes"] != ["sampled", "all_matching"] or (
        settings["execution_rules"] != ["planned", "deficit_discharge"]
        or settings["control_windows"] != base["representative_windows"]
        or settings["control_hours"] != 48
    ):
        raise ValueError("This follow-up fixes two separate, one-factor comparisons")
    system = SystemConfig(**base["system"])
    raw = load_opsd_original_pv(root / base["data"]["cache_dir"])
    windows = select_measured_windows(
        raw,
        train_days=base["train_days"],
        calibration_days=base["calibration_days"],
        test_days=base["evaluation_days"],
        column=base["data"]["column"],
        pv_scale=base["data"]["pv_scale"],
        pv_capacity_kw=base["data"]["pv_capacity_kw"],
    )
    if raw.attrs["raw_sha256"] != original["protocol"]["source"]["raw_sha256"]:
        raise ValueError("Original data snapshot differs")
    groups = []
    common = {"grid_limit_kw": system.grid_kw, "seed": base["seed"]}
    for mode in settings["scenario_modes"]:
        for window in windows:
            groups.append(
                {
                    **common,
                    "experiment": f"{mode}_28d",
                    "window": window.name,
                    "hours": 168,
                    "scenario_mode": mode,
                    "execution_rule": "planned",
                }
            )
    for rule in settings["execution_rules"]:
        for name in settings["control_windows"]:
            groups.append(
                {
                    **common,
                    "experiment": f"{rule}_48h",
                    "window": name,
                    "hours": 48,
                    "scenario_mode": "sampled",
                    "execution_rule": rule,
                }
            )
    code = {str(p.relative_to(root)): _sha(p) for p in sorted((root / "ecogrid").rglob("*.py"))}
    protocol = {
        "interpretation": settings["interpretation"],
        "settings": settings,
        "base_settings": base,
        "groups": groups,
        "original_manifest_sha256": _sha(baseline / "manifest.json"),
        "run_code_sha256": {k: v for k, v in code.items() if not k.endswith("diagnostics.py")},
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run_fingerprint"] != fingerprint:
            raise ValueError("Changed follow-up code or protocol: use a new output directory")
    else:
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_fingerprint": fingerprint,
            "completed_runs": {},
            "versions": _versions(),
            "protocol": protocol,
        }
        _write_json(output / "protocol.json", protocol)
    manifest.update(status="running", code_sha256=code)
    _write_json(manifest_path, manifest)
    try:
        prepared, old_ledgers, candidates = {}, {}, []
        for window in windows:
            frame = window.frame
            train = frame.loc[
                (frame.index < window.calibration_start)
                & (frame.observation_available_at <= window.calibration_start)
            ]
            forecast = fit_forecaster(train)
            residuals = calibrate_residuals(
                forecast,
                frame.loc[frame.index < window.test_start],
                window.calibration_start,
                window.test_start,
            )
            calibration = json.loads((baseline / f"{window.name}_calibration.json").read_text())
            assert (
                hashlib.sha256(frame.to_csv().encode()).hexdigest()
                == calibration["dataframe_sha256"]
            )
            np.testing.assert_array_equal(
                forecast.hourly_mean_kw, calibration["trained_hourly_mean_kw"]
            )
            assert len(residuals.origins) == calibration["calibration_blocks"]
            prepared[window.name] = (window, forecast, residuals)
            for hour in range(24):
                origins = residuals.origins[residuals.origins.hour == hour]
                candidates.append(
                    {
                        "window": window.name,
                        "issue_utc_hour": hour,
                        "candidate_count": len(origins),
                        "first_origin": origins.min(),
                        "last_origin": origins.max(),
                        "freeze_time": window.test_start,
                    }
                )
            for label, _, _, _ in _policies("main", base["risk_weight"]):
                run_id = _run_id({**common, "experiment": "main", "window": window.name}, label)
                old_ledgers[window.name, label] = _verify_original_ledger(
                    baseline / "runs" / f"{run_id}.csv",
                    original["completed_runs"][run_id]["ledger_sha256"],
                    frame,
                    system,
                )
        pd.DataFrame(candidates).to_csv(output / "candidates.csv", index=False)
        for group in groups:
            window, forecast, residuals = prepared[group["window"]]
            for label, policy, weight, _ in _policies("main", base["risk_weight"]):
                run_id = _run_id(group, label)
                path = output / "runs" / f"{run_id}.csv"
                if run_id in manifest["completed_runs"]:
                    record = manifest["completed_runs"][run_id]
                    if _sha(path) != record["ledger_sha256"]:
                        raise ValueError(f"Changed checkpoint: {run_id}")
                    if (
                        "forecast_sha256" in record
                        and _sha(output / "runs" / f"{run_id}_forecast.csv")
                        != record["forecast_sha256"]
                    ):
                        raise ValueError(f"Changed diagnostics: {run_id}")
                    continue
                reuse = (
                    group["hours"] == 168 and policy in ("deterministic", "perfect_foresight")
                ) or (group["experiment"] == "planned_48h" and policy != "perfect_foresight")
                print(
                    f"[{run_id}] {group['hours']}h {'verified reuse' if reuse else 'solve'}",
                    flush=True,
                )
                run_start = perf_counter()
                diagnostic_rows: list[dict] = []
                if reuse:
                    hourly = old_ledgers[window.name, label].iloc[: group["hours"]].copy()
                    hourly["original_solve_seconds"] = hourly.solve_seconds
                    hourly["solve_seconds"] = 0.0
                    hourly["emergency_discharge_kw"] = 0.0
                    if policy == "stochastic":
                        source_group = {**group, "experiment": "sampled_28d"}
                        diagnostics = pd.read_csv(
                            output / "runs" / f"{_run_id(source_group, label)}_forecast.csv"
                        )
                        diagnostic_rows = diagnostics.loc[
                            pd.to_datetime(diagnostics.forecast_issue_time, utc=True)
                            < window.test_start + pd.Timedelta(hours=group["hours"])
                        ].to_dict("records")
                elif policy == "perfect_foresight":
                    hourly = perfect_foresight(
                        window.frame,
                        window.test_start,
                        group["hours"],
                        system,
                        execution_rule=group["execution_rule"],
                    )
                else:
                    hourly = run_policy(
                        window.frame,
                        forecast,
                        residuals,
                        window.test_start,
                        group["hours"],
                        system,
                        policy=policy,
                        risk_weight=weight,
                        alpha=base["alpha"],
                        n_scenarios=base["n_scenarios"],
                        seed=base["seed"],
                        scenario_mode=group["scenario_mode"],
                        execution_rule=group["execution_rule"],
                        collect_diagnostics=diagnostic_rows if policy == "stochastic" else None,
                    )
                    if group["experiment"] == "sampled_28d":
                        old = old_ledgers[window.name, label]
                        for column in (
                            "point_forecast_pv_kw",
                            "scenario_mean_pv_kw",
                            "scenario_p10_pv_kw",
                            "scenario_p90_pv_kw",
                            "requested_charge_kw",
                            "requested_discharge_kw",
                            "soc_kwh",
                            "loss_eur",
                            "ens_kwh",
                        ):
                            np.testing.assert_allclose(
                                hourly[column], old[column], atol=1e-5, rtol=0
                            )
                hourly["reused_original"] = reuse
                hourly["execution_rule"] = group["execution_rule"]
                hourly["scenario_mode"] = group["scenario_mode"]
                hourly.to_csv(path, index_label="time")
                record = {
                    "ledger_sha256": _sha(path),
                    "reused_original": reuse,
                    "wall_seconds": perf_counter() - run_start,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
                if diagnostic_rows:
                    forecast_path = output / "runs" / f"{run_id}_forecast.csv"
                    pd.DataFrame(diagnostic_rows).to_csv(forecast_path, index=False)
                    record["forecast_sha256"] = _sha(forecast_path)
                manifest["completed_runs"][run_id] = record
                _write_json(manifest_path, manifest)
        _assemble(output, groups, base, render=False)
        hourly = pd.read_csv(output / "hourly.csv")
        representative = hourly.loc[hourly.experiment.str.endswith("48h")]
        representative.to_csv(output / "representative_hourly.csv", index=False)
        manifest.update(
            status="complete",
            completed_utc=datetime.now(timezone.utc).isoformat(),
            invocation_wall_seconds=perf_counter() - started,
        )
        manifest.pop("failure", None)
        _write_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        _write_json(manifest_path, manifest)
        raise
