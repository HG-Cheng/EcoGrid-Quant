"""Transparent cost attribution and fixed-window figures for the measured study."""

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from ecogrid.finance import value_at_risk

GROUP_KEYS = ["experiment", "window", "grid_limit_kw", "seed"]
COLORS = {
    "deterministic": "#536b88",
    "stochastic": "#238574",
    "cvar_0.5": "#d48732",
    "perfect_foresight": "#888888",
    "cvar_1": "#9a578b",
    "cvar_1_raw": "#bca0ba",
}


def ensemble_diagnostics(values: np.ndarray, actual: float) -> dict[str, float | int | bool]:
    """Describe one equally weighted ensemble without dropping duplicate members.

    Inclusive coverage uses the original exact endpoints. Equality and strict
    exclusions use absolute tolerance 1e-9 kW (zero relative tolerance); no
    forecast interval is widened. Unique values are counted by comparing sorted
    members with the first value in each tolerance group.

    Ranks span all placements of the observation among tied members. The PIT
    interval spans their rank cells, [(rank_lower-1)/(n+1), rank_upper/(n+1)];
    its midpoint is deterministic, not a randomized calibration diagnostic.
    Ideal references assume n+1 continuous exchangeable draws. They are not
    expected coverage for clipped, tied, or nonexchangeable forecasts and do
    not establish empirical calibration.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("ensemble values must be a nonempty finite 1D array")
    if not np.isfinite(actual):
        raise ValueError("actual must be finite")
    tolerance = 1e-9
    n = values.size
    ordered = np.sort(values)
    unique_count = 1
    anchor = ordered[0]
    for value in ordered[1:]:
        if value - anchor > tolerance:
            unique_count += 1
            anchor = value
    tied_count = int(np.count_nonzero(np.abs(values - actual) <= tolerance))
    rank_lower = 1 + int(np.count_nonzero(values < actual - tolerance))
    rank_upper = rank_lower + tied_count
    pit_lower = (rank_lower - 1) / (n + 1)
    pit_upper = rank_upper / (n + 1)
    minimum, maximum = float(ordered[0]), float(ordered[-1])
    p10, p90 = value_at_risk(values, 0.1), value_at_risk(values, 0.9)
    return {
        "scenario_member_count": int(n),
        "scenario_unique_member_count": unique_count,
        "observation_tied_member_count": tied_count,
        "tie_tolerance_kw": tolerance,
        "scenario_min_pv_kw": minimum,
        "scenario_max_pv_kw": maximum,
        "scenario_p10_pv_kw": p10,
        "scenario_p90_pv_kw": p90,
        "inside_min_max": bool(minimum <= actual <= maximum),
        "inside_p10_p90": bool(p10 <= actual <= p90),
        "inside_min_max_strict": bool(minimum + tolerance < actual < maximum - tolerance),
        "inside_p10_p90_strict": bool(p10 + tolerance < actual < p90 - tolerance),
        "below_ensemble_min": bool(actual < minimum - tolerance),
        "above_ensemble_max": bool(actual > maximum + tolerance),
        "below_p10": bool(actual < p10 - tolerance),
        "above_p90": bool(actual > p90 + tolerance),
        "equal_ensemble_min": bool(abs(actual - minimum) <= tolerance),
        "equal_ensemble_max": bool(abs(actual - maximum) <= tolerance),
        "equal_p10": bool(abs(actual - p10) <= tolerance),
        "equal_p90": bool(abs(actual - p90) <= tolerance),
        "observation_rank_lower": rank_lower,
        "observation_rank_upper": rank_upper,
        "observation_rank_midpoint": (rank_lower + rank_upper) / 2,
        "observation_pit_lower": pit_lower,
        "observation_pit_upper": pit_upper,
        "observation_pit_midpoint": (pit_lower + pit_upper) / 2,
        "ideal_below_ensemble_min_fraction": 1 / (n + 1),
        "ideal_p10_p90_coverage": (ceil(0.9 * n) - ceil(0.1 * n)) / (n + 1),
        "ideal_ensemble_range_coverage": (n - 1) / (n + 1),
    }


def tail_contributions(hourly: pd.DataFrame, alpha: float = 0.9) -> dict[str, float]:
    """Attribute the adjusted-loss tail to costs, not a sum of component CVaRs.

    Equal loss samples carry equal mass; a fractional boundary observation is
    included. Stable order handles exact ties deterministically. This is an
    accounting attribution of one tail, not a causal effect estimate.
    """
    if hourly.empty or not 0 < alpha < 1:
        raise ValueError("tail attribution needs samples and alpha in (0,1)")
    order = np.argsort(-hourly.loss_eur.to_numpy(), kind="stable")
    mass = len(hourly) * (1 - alpha)
    weights = np.minimum(np.maximum(mass - np.arange(len(hourly)), 0), 1) / mass
    return {
        f"tail_{column}": float(weights @ hourly[column].to_numpy()[order])
        for column in (
            "grid_cost_eur",
            "battery_cost_eur",
            "shortage_cost_eur",
            "inventory_adjustment_eur",
        )
    }


def compare_actions(hourly: pd.DataFrame) -> pd.DataFrame:
    """Compare matched requests and executions; never match across conditions."""
    rows = []
    pairs = [("stochastic", "cvar_0.5"), ("deterministic", "stochastic"), ("cvar_1_raw", "cvar_1")]
    for key, group in hourly.groupby(GROUP_KEYS, sort=False):
        for reference, policy in pairs:
            left = group.loc[group.policy == reference].set_index("time")
            right = group.loc[group.policy == policy].set_index("time")
            if left.empty or right.empty:
                continue
            if not left.index.equals(right.index):
                raise ValueError("action comparison requires identical ordered evaluation hours")
            request = (
                np.maximum(
                    (right.requested_charge_kw - left.requested_charge_kw).abs(),
                    (right.requested_discharge_kw - left.requested_discharge_kw).abs(),
                )
                > 1e-6
            )
            executed = (
                np.maximum(
                    (right.charge_kw - left.charge_kw).abs(),
                    (right.discharge_kw - left.discharge_kw).abs(),
                )
                > 1e-6
            )
            rows.append(
                {
                    **dict(zip(GROUP_KEYS, key, strict=True)),
                    "reference": reference,
                    "policy": policy,
                    "hours": len(left),
                    "request_different_hours": int(request.sum()),
                    "execution_different_hours": int(executed.sum()),
                    "protection_collapsed_hours": int((request & ~executed).sum()),
                    "adjusted_loss_difference_eur": float(
                        right.loss_eur.sum() - left.loss_eur.sum()
                    ),
                    "ens_difference_kwh": float(right.ens_kwh.sum() - left.ens_kwh.sum()),
                }
            )
    return pd.DataFrame(rows)


def forecast_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize legacy or tie-aware rows without pooling scenario/execution modes.

    Daylight is predeclared as positive point forecast OR positive realization.
    Ideal references are means of continuous exchangeable finite-member ideals,
    not empirical calibration targets for the actual forecast procedure.
    """
    rows = []
    keys = [
        *GROUP_KEYS,
        *(key for key in ("scenario_mode", "execution_rule") if key in diagnostics),
        "forecast_lead_hours",
    ]
    optional_means = {
        "inside_p10_p90_strict": "p10_p90_strict_coverage",
        "inside_min_max_strict": "ensemble_range_strict_coverage",
        **{
            name: f"{name}_fraction"
            for name in (
                "below_ensemble_min",
                "above_ensemble_max",
                "below_p10",
                "above_p90",
                "equal_ensemble_min",
                "equal_ensemble_max",
                "equal_p10",
                "equal_p90",
            )
        },
        **{
            name: f"mean_{name}"
            for name in (
                "scenario_member_count",
                "scenario_unique_member_count",
                "observation_tied_member_count",
                "observation_rank_lower",
                "observation_rank_upper",
                "observation_rank_midpoint",
                "observation_pit_lower",
                "observation_pit_upper",
                "observation_pit_midpoint",
            )
        },
        **{
            name: name
            for name in (
                "ideal_below_ensemble_min_fraction",
                "ideal_p10_p90_coverage",
                "ideal_ensemble_range_coverage",
                "tie_tolerance_kw",
            )
        },
    }
    for key, group in diagnostics.groupby(keys, sort=False):
        for population, selected in (
            ("all_hours", group),
            ("daylight_proxy", group.loc[group.forecast_daylight_proxy]),
        ):
            error = selected.forecast_error_pv_kw
            row = {
                **dict(zip(keys, key, strict=True)),
                "population": population,
                "sample_count": len(selected),
                "mae_kw": error.abs().mean(),
                "bias_forecast_minus_actual_kw": error.mean(),
                "rmse_kw": np.sqrt((error**2).mean()),
                "p10_p90_coverage": selected.inside_p10_p90.mean(),
                "ensemble_range_coverage": selected.inside_min_max.mean(),
                "below_ensemble_min_fraction": (
                    selected.actual_pv_kw < selected.scenario_min_pv_kw - 1e-9
                ).mean(),
                "mean_p10_p90_width_kw": (
                    selected.scenario_p90_pv_kw - selected.scenario_p10_pv_kw
                ).mean(),
            }
            row.update(
                {
                    destination: selected[source].mean()
                    for source, destination in optional_means.items()
                    if source in selected
                }
            )
            if "observation_tied_member_count" in selected:
                row["observation_tie_fraction"] = (
                    selected.observation_tied_member_count > 0
                ).mean()
            if {"scenario_unique_member_count", "scenario_member_count"} <= set(selected):
                row["any_member_ties_fraction"] = (
                    selected.scenario_unique_member_count < selected.scenario_member_count
                ).mean()
            for name in ("rank", "pit"):
                lower, upper = f"observation_{name}_lower", f"observation_{name}_upper"
                if {lower, upper} <= set(selected):
                    row[f"mean_{name}_interval_width"] = (selected[upper] - selected[lower]).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def write_figures(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    risks: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output: Path,
    representatives: list[str],
) -> None:
    """Three figures; calendar-selected examples, with no best-result selection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 140})
    main = daily.loc[daily.experiment == "main"]
    policies = ["deterministic", "stochastic", "cvar_0.5", "perfect_foresight"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    means = main.groupby("policy").mean(numeric_only=True).reindex(policies)
    bottom = np.zeros(len(policies))
    for column, label, color in (
        ("total_actual_expense_eur", "Grid + battery expense", "#617b92"),
        ("total_shortage_cost_eur", "ENS penalty", "#cf7752"),
        ("total_inventory_adjustment_eur", "Inventory adjustment", "#b5bd81"),
    ):
        values = means[column].to_numpy()
        axes[0].bar(policies, values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[0].set_title("Mean daily cost decomposition")
    axes[0].set_ylabel("EUR / UTC day")
    axes[0].legend(fontsize=7)
    daily_risk = (
        risks.loc[
            (risks.experiment == "main")
            & (risks.scale == "utc_day")
            & (risks.metric == "adjusted_loss")
        ]
        .set_index("policy")
        .reindex(policies)
    )
    axes[1].bar(policies, daily_risk.cvar, color=[COLORS[p] for p in policies])
    axes[1].set_title(f"Daily loss CVaR 90%; n={int(daily_risk.sample_count.iloc[0])} days")
    axes[1].set_ylabel("EUR / day-loss sample")
    for i, policy in enumerate(policies):
        vals = main.loc[main.policy == policy, "total_ens_kwh"].to_numpy()
        axes[2].scatter(
            i + np.linspace(-0.1, 0.1, len(vals)), vals, color=COLORS[policy], alpha=0.7, s=14
        )
    axes[2].set_xticks(range(len(policies)), policies)
    axes[2].set_title("Daily ENS: each point is one evaluated day")
    axes[2].set_ylabel("kWh / day")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Measured-profile simulation; hindsight bounds total adjusted loss only")
    fig.savefig(output / "comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    diag = diagnostics.loc[diagnostics.experiment == "main"]
    for col, window in enumerate(representatives[:2]):
        selected = diag.loc[diag.window == window]
        first = selected.loc[selected.forecast_issue_time == selected.forecast_issue_time.min()]
        x = pd.to_datetime(first.forecast_valid_time, utc=True)
        ax = axes[0, col]
        ax.fill_between(
            x,
            first.scenario_min_pv_kw,
            first.scenario_max_pv_kw,
            color="#c8dace",
            label="Scenario min-max",
        )
        ax.fill_between(
            x,
            first.scenario_p10_pv_kw,
            first.scenario_p90_pv_kw,
            color="#8cbaa3",
            alpha=0.65,
            label="Scenario p10-p90",
        )
        ax.plot(x, first.point_forecast_pv_kw, ls="--", color="#d48732", label="Point")
        ax.plot(x, first.actual_pv_kw, color="#33485c", label="Realized")
        ax.set_title(f"{window}: preselected first 24h forecast")
        ax.set_ylabel("kW")
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.legend(fontsize=7)
    daylight = diag.loc[diag.forecast_daylight_proxy]
    grouped = daylight.groupby("forecast_lead_hours")
    axes[1, 0].plot(grouped.inside_p10_p90.mean(), label="p10-p90")
    axes[1, 0].plot(grouped.inside_min_max.mean(), label="Min-max")
    if "ideal_p10_p90_coverage" in daylight:
        reference = grouped.ideal_p10_p90_coverage.mean()
    elif "scenario_member_count" in daylight:
        counts = daylight.scenario_member_count
        reference = (
            ((np.ceil(0.9 * counts) - np.ceil(0.1 * counts)) / (counts + 1))
            .groupby(daylight.forecast_lead_hours)
            .mean()
        )
    else:
        reference = None
    if reference is not None:
        axes[1, 0].plot(
            reference, color="grey", ls=":", label="Ideal continuous/exchangeable p10-p90"
        )
    axes[1, 0].set(
        xlabel="Lead hours",
        ylabel="Empirical coverage",
        ylim=(0, 1.05),
        title="All normal windows: daylight proxy only",
    )
    axes[1, 0].legend(fontsize=7)
    lead0 = daylight.loc[daylight.forecast_lead_hours == 0].copy()
    lead0["utc_hour"] = pd.to_datetime(lead0.forecast_valid_time, utc=True).dt.hour
    error = lead0.groupby("utc_hour").forecast_error_pv_kw
    axes[1, 1].plot(error.mean(), label="Point minus realized")
    axes[1, 1].plot(error.apply(lambda x: x.abs().mean()), label="MAE")
    axes[1, 1].axhline(0, color="grey", ls=":")
    axes[1, 1].set(xlabel="UTC valid hour", ylabel="kW", title="Lead 0 daytime forecast errors")
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Ideal reference assumes continuous exchangeable draws; no calibration claim")
    fig.savefig(output / "forecast.png")
    plt.close(fig)

    fig, axes = plt.subplots(
        len(representatives),
        3,
        figsize=(15, 3.3 * len(representatives)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, window in enumerate(representatives):
        group = hourly.loc[(hourly.experiment == "main") & (hourly.window == window)]
        base = group.loc[group.policy == "stochastic"].iloc[:48]
        time = pd.to_datetime(base.time, utc=True)
        axes[row, 0].plot(time, base.pv_kw, label="Realized PV", color="#33485c")
        axes[row, 0].plot(
            time, base.point_forecast_pv_kw, label="Point forecast", ls="--", color="#d48732"
        )
        axes[row, 0].fill_between(
            time,
            base.scenario_min_pv_kw,
            base.scenario_max_pv_kw,
            alpha=0.2,
            color=COLORS["stochastic"],
            label="First-hour scenario range",
        )
        axes[row, 0].set(title=f"{window}: first 48h, selected by calendar", ylabel="kW")
        for policy in policies[:3]:
            part = group.loc[group.policy == policy].iloc[:48]
            executed = part.discharge_kw - part.charge_kw
            requested = part.requested_discharge_kw - part.requested_charge_kw
            axes[row, 1].step(time, executed, where="post", label=policy, color=COLORS[policy])
            axes[row, 1].step(
                time, requested, where="post", ls=":", alpha=0.7, color=COLORS[policy]
            )
            axes[row, 2].plot(time, part.soc_kwh, label=policy, color=COLORS[policy])
        axes[row, 1].set(
            title="Battery: solid executed, dotted requested", ylabel="kW (+discharge)"
        )
        axes[row, 2].set(title="SOC from actual energy", ylabel="kWh")
        for ax in axes[row]:
            ax.legend(fontsize=6)
            ax.grid(alpha=0.2)
            ax.tick_params(axis="x", rotation=25, labelsize=7)
    fig.savefig(output / "dispatch.png")
    plt.close(fig)


def write_mechanism_figures(output: Path) -> None:
    """Render saved mechanism results without fitting forecasts or optimizing.

    Reads hourly.csv, daily.csv, risk_summary.csv and the stochastic per-run
    forecast CSVs for sampled_28d/all_matching_28d. Writes mechanisms.png and
    finite_ensemble.png, and regenerates forecast.png (plus comparison.png and
    dispatch.png) for sampled_28d using the original calendar-selected examples.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    hourly = pd.read_csv(output / "hourly.csv")
    daily = pd.read_csv(output / "daily.csv")
    risks = pd.read_csv(output / "risk_summary.csv")
    policies = ["deterministic", "stochastic", "cvar_0.5", "perfect_foresight"]
    policy_labels = ["Deterministic", "Stochastic", "CVaR 0.5", "Perfect foresight"]
    scenario_experiments = ("sampled_28d", "all_matching_28d")
    control_experiments = ("planned_48h", "deficit_discharge_48h")
    expected = {*scenario_experiments, *control_experiments}
    if not expected <= set(hourly.experiment):
        raise ValueError("mechanism figures require all four completed experiment populations")
    if len(hourly[["grid_limit_kw", "seed"]].drop_duplicates()) != 1:
        raise ValueError("mechanism figures require one grid limit and seed")

    summary_rows = []
    for (experiment, policy), population in hourly.groupby(["experiment", "policy"], sort=False):
        selected_risk = risks.loc[(risks.experiment == experiment) & (risks.policy == policy)]
        stats = {"experiment": experiment, "policy": policy, "hours": len(population)}
        for column in (
            "actual_expense_eur",
            "grid_cost_eur",
            "battery_cost_eur",
            "shortage_cost_eur",
            "inventory_adjustment_eur",
            "loss_eur",
            "ens_kwh",
            "curtailment_kwh",
            "throughput_kwh",
            "emergency_discharge_kw",
            "solve_seconds",
            "original_solve_seconds",
        ):
            unit_column = (
                "emergency_discharge_kwh" if column == "emergency_discharge_kw" else column
            )
            stats[f"total_{unit_column}"] = (
                population[column].sum() if column in population else 0.0
            )
        stats["total_grid_import_kwh"] = population.grid_kw.sum()
        stats["hours_with_emergency_discharge"] = int(
            (population.emergency_discharge_kw > 1e-7).sum()
        )
        for column in (
            "scenario_candidate_count",
            "scenario_count",
            "scenario_unique_path_count",
            "scenario_tail_equivalent_count",
        ):
            values = population[column].dropna()
            stats[f"min_{column}"] = values.min()
            stats[f"max_{column}"] = values.max()
            stats[f"mean_{column}"] = values.mean()
        for scale in ("hour", "utc_day"):
            for metric in ("adjusted_loss", "ens"):
                risk = selected_risk.loc[
                    (selected_risk.scale == scale) & (selected_risk.metric == metric)
                ].iloc[0]
                for field in ("mean", "var", "cvar", "sample_count", "tail_equivalent_count"):
                    stats[f"{scale}_{metric}_{field}"] = risk[field]
        summary_rows.append(stats)
    pd.DataFrame(summary_rows).to_csv(output / "mechanism_summary.csv", index=False)

    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    x = np.arange(len(policies))
    colors = ("#397d72", "#c18a41")
    for row, (experiments, labels, scope) in enumerate(
        (
            (scenario_experiments, ("Sampled members", "All matching members"), "Scenario mode"),
            (control_experiments, ("Planned execution", "Deficit discharge"), "Execution rule"),
        )
    ):
        day_counts = []
        for position, (experiment, label, color) in enumerate(
            zip(experiments, labels, colors, strict=True)
        ):
            totals = (
                hourly.loc[hourly.experiment == experiment]
                .groupby("policy")
                .sum(numeric_only=True)
                .reindex(policies)
            )
            risk = (
                risks.loc[
                    (risks.experiment == experiment)
                    & (risks.scale == "utc_day")
                    & (risks.metric == "adjusted_loss")
                ]
                .set_index("policy")
                .reindex(policies)
            )
            day_counts.append(int(risk.sample_count.iloc[0]))
            if not np.allclose(risk.alpha, 0.9):
                raise ValueError("mechanism figure labels require alpha=0.9")
            for col, values in enumerate((totals.loss_eur, risk.cvar, totals.ens_kwh)):
                bars = axes[row, col].bar(
                    x + (position - 0.5) * 0.36, values, 0.36, label=label, color=color
                )
                # Format numerical zeros in labels only; CSV values and bars remain intact.
                labels_text = ["0" if abs(value) < 1e-9 else f"{value:.3g}" for value in values]
                axes[row, col].bar_label(bars, labels=labels_text, padding=3, fontsize=7)
        if len(set(day_counts)) != 1:
            raise ValueError("paired mechanism populations must have the same UTC-day count")
        population = (
            f"{day_counts[0]} days across four windows"
            if row == 0
            else f"winter + summer, 48 h each; {day_counts[0]} days total"
        )
        for col, (title, unit) in enumerate(
            (
                ("Total adjusted loss", "EUR / evaluated population"),
                ("Daily adjusted-loss CVaR 90%", "EUR / UTC-day loss sample"),
                ("Total energy not served", "kWh / evaluated population"),
            )
        ):
            ax = axes[row, col]
            ax.set_title(f"{scope}: {title}\n{population}", fontsize=10)
            ax.set_ylabel(unit)
            ax.set_xticks(x, policy_labels, rotation=18, ha="right")
            ax.margins(y=0.2)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        axes[row, 0].legend(fontsize=8)
    fig.suptitle(
        "Post-hoc sensitivity; perfect foresight bounds total adjusted loss only", fontsize=13
    )
    fig.savefig(output / "mechanisms.png")
    plt.close(fig)

    forecast_rows = []
    groups = hourly.loc[hourly.experiment.isin(scenario_experiments), GROUP_KEYS].drop_duplicates()
    for group in groups.itertuples(index=False):
        capacity = f"{group.grid_limit_kw:g}".replace(".", "p")
        filename = (
            f"{group.experiment}_{group.window}_g{capacity}_s{group.seed}_stochastic_forecast.csv"
        )
        forecast_rows.append(pd.read_csv(output / "runs" / filename).assign(**group._asdict()))
    forecasts = pd.concat(forecast_rows, ignore_index=True)
    lead0 = forecasts.loc[(forecasts.forecast_lead_hours == 0) & forecasts.forecast_daylight_proxy]
    samples = [lead0.loc[lead0.experiment == experiment] for experiment in scenario_experiments]
    if any(frame.empty for frame in samples):
        raise ValueError("finite-ensemble figures require lead-zero daylight observations")
    mode_labels = []
    for label, frame in zip(("Sampled", "All matching"), samples, strict=True):
        low, high = int(frame.scenario_member_count.min()), int(frame.scenario_member_count.max())
        members = str(low) if low == high else f"{low}-{high}"
        mode_labels.append(f"{label}\nn={members} members; {len(frame)} observations")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = np.arange(2)
    for ax, interval, ideal, title in (
        (axes[0, 0], "p10_p90", "ideal_p10_p90_coverage", "p10-p90 interval coverage"),
        (axes[0, 1], "min_max", "ideal_ensemble_range_coverage", "Ensemble range coverage"),
    ):
        for position, (column, label, color) in enumerate(
            (
                (f"inside_{interval}", "Inclusive", "#397d72"),
                (f"inside_{interval}_strict", "Strict", "#80ada2"),
                (ideal, "Ideal continuous/exchangeable", "#b4b8bc"),
            )
        ):
            bars = ax.bar(
                x + (position - 1) * 0.24,
                [frame[column].mean() for frame in samples],
                0.24,
                color=color,
                label=label,
            )
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 0]
    for position, (column, label, color) in enumerate(
        (
            ("below_ensemble_min", "Observed strict below minimum", "#397d72"),
            ("ideal_below_ensemble_min_fraction", "Ideal continuous/exchangeable", "#b4b8bc"),
        )
    ):
        bars = ax.bar(
            x + (position - 0.5) * 0.3,
            [frame[column].mean() for frame in samples],
            0.3,
            color=color,
            label=label,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title("Observation below ensemble minimum")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 1]
    tie_statistics = (
        [
            (frame.scenario_unique_member_count < frame.scenario_member_count).mean()
            for frame in samples
        ],
        [(frame.observation_tied_member_count > 0).mean() for frame in samples],
        [
            (
                (frame.observation_rank_upper - frame.observation_rank_lower)
                / frame.scenario_member_count
            ).mean()
            for frame in samples
        ],
    )
    for position, (values, label, color) in enumerate(
        zip(
            tie_statistics,
            ("Any tied members", "Observation tied to member", "Mean tie rank width / n"),
            ("#397d72", "#80ada2", "#c18a41"),
            strict=True,
        )
    ):
        bars = ax.bar(x + (position - 1) * 0.24, values, 0.24, color=color, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title("Ties retained with equal source-member weights")
    ax.legend(fontsize=8, loc="upper left")
    for ax in axes.flat:
        ax.set_xticks(x, mode_labels)
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1.28)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Lead 0, daylight proxy: point forecast or observation > 1e-7 kW\n"
        "Ideal references require continuous exchangeable draws; no calibration verdict",
        fontsize=12,
    )
    fig.supxlabel(
        "Ties and strict exclusions use absolute tolerance 1e-9 kW; inclusive endpoints unchanged.",
        fontsize=8,
    )
    fig.savefig(output / "finite_ensemble.png")
    plt.close(fig)

    baseline_tables = [
        table.loc[table.experiment == "sampled_28d"].assign(experiment="main")
        for table in (hourly, daily, risks, forecasts)
    ]
    write_figures(
        baseline_tables[0],
        baseline_tables[1],
        baseline_tables[2],
        baseline_tables[3],
        output,
        representatives=["winter", "summer"],
    )
