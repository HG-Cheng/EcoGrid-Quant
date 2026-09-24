"""Small reproducible experiment runner; no notebook or parameter grid required."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecogrid.data.benchmark import load_public_data, synthetic_data
from ecogrid.dispatch import SystemConfig
from ecogrid.evaluation import apply_stress, daily_metrics, perfect_foresight, run_policy, summarize
from ecogrid.forecast import calibrate_residuals, fit_forecaster


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions() -> dict:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "pyomo", "highspy", "scipy")
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _plots(summary: pd.DataFrame, hourly: pd.DataFrame, output: Path, alpha: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = [
        ("mean_hourly_loss_eur", "Mean accounted loss (EUR/hour)"),
        ("cvar_hourly_loss_eur", f"Hourly CVaR {alpha:.0%} (EUR)"),
        ("total_ens_kwh", "Unserved energy (kWh / evaluation)"),
    ]
    colors = {
        "deterministic": "#506784",
        "stochastic": "#2e927d",
        "cvar_0.5": "#d68936",
        "cvar_1": "#ac5d8b",
        "perfect_foresight": "#777777",
    }
    for row, case in enumerate(("normal", "stress")):
        part = summary.loc[summary["case"] == case]
        labels = [f"{d}\n{p}" for d, p in zip(part.dataset, part.policy, strict=True)]
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]
            ax.bar(
                np.arange(len(part)),
                part[metric],
                color=[colors.get(p, "#7b61a1") for p in part.policy],
            )
            ax.set_xticks(np.arange(len(part)), labels, rotation=65, ha="right", fontsize=7)
            ax.set_title(f"{case.capitalize()}: {title}")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Same conditions, common physical execution; hindsight is an ex post reference", fontsize=12
    )
    fig.savefig(output / "comparison.png")
    plt.close(fig)

    datasets = list(hourly.dataset.unique())
    fig, axes = plt.subplots(
        len(datasets), 2, figsize=(13, 3.5 * len(datasets)), squeeze=False, constrained_layout=True
    )
    for row, dataset in enumerate(datasets):
        data = hourly.loc[(hourly.dataset == dataset) & (hourly["case"] == "normal")]
        reference = data.loc[data.policy == "deterministic"].iloc[:48]
        ax = axes[row, 0]
        ax.plot(
            reference.time, reference.pv_kw, label="Realized PV (proxy if public)", color="#d69b24"
        )
        ax.plot(
            reference.time,
            reference.point_forecast_pv_kw,
            label="Issued point forecast",
            ls="--",
            color="#d65f38",
        )
        ax.plot(reference.time, reference.load_kw, label="Known load", color="#334455")
        ax.set_title(f"{dataset}: first 48 evaluation hours")
        ax.set_ylabel("kW")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        for policy, group in data.groupby("policy", sort=False):
            group = group.iloc[:48]
            axes[row, 1].plot(
                group.time,
                group.soc_kwh,
                label=policy,
                color=colors.get(policy, "#7b61a1"),
                alpha=0.85,
            )
        axes[row, 1].set_title("SOC from executed energy")
        axes[row, 1].set_ylabel("kWh")
        axes[row, 1].legend(fontsize=7)
        axes[row, 1].grid(alpha=0.2)
        for ax in axes[row]:
            ax.tick_params(axis="x", rotation=25, labelsize=7)
    fig.savefig(output / "dispatch.png")
    plt.close(fig)


def run_experiment(settings: dict, sources: list[str], output: Path) -> pd.DataFrame:
    """Fixed chronological split; no selection or tuning based on evaluation."""
    system = SystemConfig(**settings["system"])
    for key in ("train_days", "calibration_days", "evaluation_days", "stress_days", "n_scenarios"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    alpha = float(settings["alpha"])
    weights = list(dict.fromkeys(float(w) for w in settings["risk_weights"]))
    if not 0 < alpha < 1 or not weights or any(not 0 <= w <= 1 for w in weights):
        raise ValueError("Invalid confidence level or risk weights")
    positive_weights = [w for w in weights if w > 0]
    if not positive_weights:
        raise ValueError("Include at least one positive CVaR weight")
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    code_paths = sorted((root / "ecogrid").rglob("*.py"))
    manifest: dict[str, Any] = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "effective_system": asdict(system),
        "versions": _versions(),
        "python": sys.version,
        "interpreter": sys.executable,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "code_sha256": {str(p.relative_to(root)): _sha(p) for p in code_paths},
        "solver": {
            "name": "appsi_highs",
            "mip_rel_gap": 1e-8,
            "random_seed": 0,
            "time_limit": system.solver_time_limit,
            "threads": "HiGHS default",
            "acceptance": "optimal and residual <= feasibility_tolerance",
        },
        "planning_horizon_hours": 24,
        "execution_step_hours": 1,
        "risk_population": "equally weighted realized hours, not forecast scenario weights",
        "solve_timing": "wall seconds including model construction, solver and validation",
        "stress": {
            "pv_multiplier": 0.2,
            "positive_forecast_bias_kw": 0.75,
            "retune_on_stress": False,
        },
        "sources": {},
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    summaries, ledgers, daily_rows, forecast_metrics = [], [], [], []
    try:
        for source in sources:
            print(f"[{source}] preparing train/calibration/test split", flush=True)
            frame = (
                synthetic_data(days=settings["synthetic_days"], seed=settings["seed"])
                if source == "synthetic"
                else load_public_data(root / "data/raw/open_meteo_munich_2024.json")
            )
            anchor = frame.index[0].ceil("D")
            training_end = anchor + pd.Timedelta(days=settings["train_days"])
            evaluation_start = training_end + pd.Timedelta(days=settings["calibration_days"])
            max_hours = 24 * max(settings["evaluation_days"], settings["stress_days"])
            if evaluation_start + pd.Timedelta(hours=max_hours + 22) > frame.index[-1]:
                raise ValueError(
                    f"{source}: insufficient data for evaluation plus full 24h lookahead"
                )
            train = frame.loc[(frame.index >= anchor) & (frame.index < training_end)]
            forecast = fit_forecaster(train)
            library = calibrate_residuals(
                forecast, frame.loc[frame.index < evaluation_start], training_end, evaluation_start
            )
            provenance = dict(frame.attrs)
            provenance.update(
                {
                    "training_start": anchor,
                    "training_end_exclusive": training_end,
                    "calibration_start": training_end,
                    "calibration_end_exclusive": evaluation_start,
                    "evaluation_start": evaluation_start,
                    "calibration_blocks": len(library.origins),
                    "forecast": "50% training hourly mean + 50% prior-seven-day same-hour mean",
                    "scenario_method": (
                        "hour-matched whole 24h residual block bootstrap, frozen library"
                    ),
                    "derived_frame_sha256": hashlib.sha256(frame.to_csv().encode()).hexdigest(),
                }
            )
            if source == "public":
                raw = root / "data/raw/open_meteo_munich_2024.json"
                provenance["raw_sha256"] = _sha(raw)
            manifest["sources"][source] = provenance
            derived = root / "data/derived" / source
            derived.mkdir(parents=True, exist_ok=True)
            frame.to_csv(derived / "hourly.csv", index_label="time")
            np.savez_compressed(
                output / f"{source}_calibration.npz",
                residuals=library.residuals,
                origin_unix_ns=library.origins.asi8,
                forecast_hourly_mean_kw=forecast.hourly_mean_kw,
            )
            for case, days in (
                ("normal", settings["evaluation_days"]),
                ("stress", settings["stress_days"]),
            ):
                hours = int(days) * 24
                actual = apply_stress(frame, evaluation_start) if case == "stress" else frame
                print(f"[{source}/{case}] perfect foresight, {hours}h", flush=True)
                perfect = perfect_foresight(actual, evaluation_start, hours, system)
                perfect_total = float(perfect.loss_eur.sum())
                policies = [
                    ("deterministic", "deterministic", 0.0),
                    ("stochastic", "stochastic", 0.0),
                ]
                case_weights = positive_weights if case == "normal" else positive_weights[:1]
                policies += [(f"cvar_{w:g}", "cvar", w) for w in case_weights]
                results = [("perfect_foresight", 0.0, perfect)]
                for label, policy, weight in policies:
                    print(f"[{source}/{case}] {label}: {hours} hourly decisions", flush=True)
                    hourly = run_policy(
                        frame,
                        forecast,
                        library,
                        evaluation_start,
                        hours,
                        system,
                        policy=policy,
                        risk_weight=weight,
                        alpha=alpha,
                        n_scenarios=settings["n_scenarios"],
                        seed=settings["seed"],
                        stress=case == "stress",
                    )
                    results.append((label, weight, hourly))
                for label, weight, hourly in results:
                    stats = summarize(hourly, alpha, perfect_total)
                    if stats["regret_eur"] < -1e-5:
                        raise RuntimeError(
                            f"Negative comparable-period regret: {source}/{case}/{label}"
                        )
                    summaries.append(
                        {
                            "dataset": source,
                            "case": case,
                            "policy": label,
                            "risk_weight": weight,
                            "alpha": alpha,
                            **stats,
                        }
                    )
                    daily = daily_metrics(hourly).reset_index()
                    daily["dataset"], daily["case"], daily["policy"] = source, case, label
                    daily_rows.append(daily)
                    ledger = hourly.copy()
                    ledger["dataset"], ledger["case"], ledger["policy"] = source, case, label
                    ledger.index.name = "time"
                    ledgers.append(ledger.reset_index())
                    if label == "deterministic":
                        error = hourly.point_forecast_pv_kw - hourly.pv_kw
                        forecast_metrics.append(
                            {
                                "dataset": source,
                                "case": case,
                                "hours": hours,
                                "lead_hours": 0,
                                "mae_kw": float(error.abs().mean()),
                                "rmse_kw": float(np.sqrt(np.mean(error**2))),
                                "forecast_minus_actual_bias_kw": float(error.mean()),
                            }
                        )
                # Persist completed case ledgers before starting another expensive case.
                pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
                pd.concat(ledgers, ignore_index=True).to_csv(output / "hourly.csv", index=False)
                pd.concat(daily_rows, ignore_index=True).to_csv(output / "daily.csv", index=False)
                _write_json(manifest_path, manifest)
        summary = pd.DataFrame(summaries)
        pd.DataFrame(forecast_metrics).to_csv(output / "forecast_metrics.csv", index=False)
        summary.loc[
            (summary["case"] == "normal") & summary.policy.str.match("stochastic|cvar_")
        ].to_csv(output / "risk_sweep.csv", index=False)
        _plots(summary, pd.concat(ledgers, ignore_index=True), output, alpha)
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
        print(
            summary[
                [
                    "dataset",
                    "case",
                    "policy",
                    "mean_hourly_loss_eur",
                    "cvar_hourly_loss_eur",
                    "total_ens_kwh",
                    "regret_eur",
                ]
            ].to_string(index=False),
            flush=True,
        )
        return summary
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        _write_json(manifest_path, manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PV uncertainty / battery / finite-grid research demo"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/quick.json"))
    parser.add_argument(
        "--source", choices=("synthetic", "public", "both", "measured"), default="synthetic"
    )
    parser.add_argument("--output", type=Path, default=Path("results/quick"))
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if args.source == "measured":
        from ecogrid.study import run_mechanism_study, run_study

        if settings.get("analysis") == "posthoc_mechanisms":
            run_mechanism_study(settings, args.output)
        else:
            run_study(settings, args.output)
    else:
        sources = ["synthetic", "public"] if args.source == "both" else [args.source]
        run_experiment(settings, sources, args.output)
    print(f"Results: {args.output.resolve()}", flush=True)
    return 0
