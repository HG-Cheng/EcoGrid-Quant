"""Causal battery requests and common, physically feasible hourly execution.

PV for [t,t+1h) is not known when its battery request is selected. Execution
summarises ideal within-hour balancing; it does not establish subhour feasibility.
"""

import numpy as np

from ecogrid.dispatch import SystemConfig


def execute_hour(
    config: SystemConfig,
    *,
    soc: float,
    load: float,
    pv: float,
    price: float,
    requested_charge: float,
    requested_discharge: float,
    execution_rule: str = "planned",
) -> dict[str, float | str]:
    """Apply a pre-issued request, serving load before charging.

    Online safeguards may reduce the charge request when real supply is scarce,
    or discharge when state/load limits require it. Reductions are reported.
    With ``deficit_discharge``, extra feasible discharge can cover load beyond
    realized PV plus grid capacity, after any unpowered charging is canceled.
    The returned state is computed from executed energy, never clipped.
    """
    cfg = config
    if execution_rule not in ("planned", "deficit_discharge"):
        raise ValueError("execution_rule must be 'planned' or 'deficit_discharge'")
    values = np.array([soc, load, pv, price, requested_charge, requested_discharge])
    if not np.isfinite(values).all() or np.min(values) < -cfg.feasibility_tolerance:
        raise ValueError("Execution inputs must be finite and nonnegative")
    if min(load, pv, price) < 0:
        raise ValueError("Realized load, PV and price must be nonnegative")
    if not 0 <= soc <= cfg.battery_kwh:
        raise ValueError("Invalid physical SOC; no clipping or repair is allowed")
    if (
        requested_charge > cfg.feasibility_tolerance
        and requested_discharge > cfg.feasibility_tolerance
    ):
        raise ValueError("Simultaneous charging/discharging request is invalid")
    # Only remove solver-scale signed zeros from actions, never alter the state.
    requested_charge = max(0.0, requested_charge)
    requested_discharge = max(0.0, requested_discharge)
    # Move energy-derived POWER bounds inward by one representable float. This
    # prevents cancellation producing SOC=-2e-16; energy remains exactly the
    # result of the executed action, rather than repairing/clipping SOC later.
    energy_charge_cap = float(np.nextafter((cfg.battery_kwh - soc) / cfg.charge_efficiency, 0.0))
    energy_discharge_cap = float(np.nextafter(soc * cfg.discharge_efficiency, 0.0))
    charge = min(
        requested_charge, cfg.battery_kw, energy_charge_cap, max(pv + cfg.grid_kw - load, 0.0)
    )
    discharge = min(requested_discharge, cfg.battery_kw, energy_discharge_cap, load)
    if execution_rule == "deficit_discharge" and charge == 0:
        deficit = max(load - pv - cfg.grid_kw, 0.0)
        discharge = max(discharge, min(deficit, cfg.battery_kw, energy_discharge_cap))
    for _ in range(8):
        soc_next = soc + cfg.charge_efficiency * charge - discharge / cfg.discharge_efficiency
        if 0 <= soc_next <= cfg.battery_kwh:
            break
        if soc_next > cfg.battery_kwh:
            charge = float(np.nextafter(charge, 0.0))
        else:
            discharge = float(np.nextafter(discharge, 0.0))
    else:
        raise RuntimeError("Cannot represent a feasible executed battery action")
    net = load + charge - discharge
    pv_used = min(pv, net)
    grid = min(cfg.grid_kw, max(net - pv_used, 0.0))
    shed = max(net - pv_used - grid, 0.0)
    balance = grid + pv_used + discharge + shed - load - charge
    if (
        soc_next < -cfg.feasibility_tolerance
        or soc_next > cfg.battery_kwh + cfg.feasibility_tolerance
        or shed > load + cfg.feasibility_tolerance
        or abs(balance) > cfg.feasibility_tolerance
    ):
        raise RuntimeError("Executed energy balance or physical bounds violated")
    grid_cost = grid * price
    battery_cost = cfg.throughput_eur_per_kwh * (charge + discharge)
    shortage_cost = cfg.lost_load_eur_per_kwh * shed
    inventory = -cfg.terminal_eur_per_kwh * (soc_next - soc)
    return {
        "load_kw": float(load),
        "pv_kw": float(pv),
        "price_eur_per_kwh": float(price),
        "requested_charge_kw": requested_charge,
        "requested_discharge_kw": requested_discharge,
        "charge_kw": charge,
        "discharge_kw": discharge,
        "soc_before_kwh": soc,
        "soc_kwh": soc_next,
        "pv_used_kw": pv_used,
        "curtailment_kwh": pv - pv_used,
        "grid_kw": grid,
        "ens_kwh": shed,
        "throughput_kwh": charge + discharge,
        "grid_cost_eur": grid_cost,
        "battery_cost_eur": battery_cost,
        "shortage_cost_eur": shortage_cost,
        "inventory_adjustment_eur": inventory,
        "operating_cost_eur": grid_cost + battery_cost + shortage_cost,
        "loss_eur": grid_cost + battery_cost + shortage_cost + inventory,
        "charge_override_kw": requested_charge - charge,
        "discharge_override_kw": max(requested_discharge - discharge, 0.0),
        "emergency_discharge_kw": max(discharge - requested_discharge, 0.0),
        "execution_rule": execution_rule,
        "balance_residual_kw": balance,
    }
