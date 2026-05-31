"""EcoGrid-Quant central configuration.

This module centralises every physical, thermodynamic and financial parameter
of the micro-grid into immutable :class:`~dataclasses.dataclass` objects so that
the optimisation engines, scenario generator and financial valuation layer all
read from a single, type-checked source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridConfig:
    """Physical, thermodynamic and economic parameters of the micro-grid.

    All monetary values use euros (EUR) and all power values use mega-watts
    (MW). Energy quantities use mega-watt-hours (MWh) on an hourly grid, so a
    power of ``P`` MW sustained for one hour stores ``P`` MWh.

    Parameters
    ----------
    carbon_tax_rate
        Carbon price applied to thermal emissions, in EUR per tonne of CO2.
    coal_cost
        Marginal fuel cost of the thermal (coal) unit, in EUR/MWh.
    coal_emission
        Emission intensity of the thermal unit, in tonnes CO2/MWh.
    bess_capacity
        Usable energy capacity of the battery energy storage system, in MWh.
    c_rate
        Charge/discharge rate of the battery as a fraction of capacity per
        hour. A 4-hour storage system corresponds to ``c_rate = 0.25``.
    eff_charge, eff_discharge
        One-way charging / discharging round-trip efficiencies (0-1).
    initial_soc
        Battery state of charge at ``t = 0``, in MWh.
    coal_pmax
        Maximum power of the thermal unit when committed, in MW.
    coal_pmin
        Minimum stable power of the thermal unit when committed, in MW. A
        committed coal unit cannot operate below this floor, which is what
        creates the need for binary commitment variables.
    startup_cost
        Fixed cost incurred whenever the thermal unit transitions from off to
        on, in EUR.
    min_up_time, min_down_time
        Minimum number of consecutive hours the thermal unit must remain on
        (respectively off) once it has been started (respectively stopped).
    wind_cost, solar_cost
        Marginal generation cost of renewables, in EUR/MWh. Kept near zero but
        non-negative so the solver still prefers the cheapest feasible mix.
    """

    carbon_tax_rate: float = 50.0
    coal_cost: float = 30.0
    coal_emission: float = 0.8

    bess_capacity: float = 100.0
    c_rate: float = 0.25
    eff_charge: float = 0.95
    eff_discharge: float = 0.95
    initial_soc: float = 0.0

    coal_pmax: float = 1000.0
    coal_pmin: float = 0.0
    startup_cost: float = 0.0
    min_up_time: int = 1
    min_down_time: int = 1

    wind_cost: float = 0.0
    solar_cost: float = 0.0

    @property
    def bess_max_power(self) -> float:
        """Maximum charge/discharge power of the battery, in MW."""
        return self.bess_capacity * self.c_rate

    @property
    def coal_marginal_cost(self) -> float:
        """Effective marginal cost of coal incl. carbon tax, in EUR/MWh."""
        return self.coal_cost + self.coal_emission * self.carbon_tax_rate


@dataclass(frozen=True)
class PlantConfig:
    """Renewable plant nameplate capacities and the weather-to-power model.

    The mapping from raw weather observations to available power follows the
    simplified physical model used since V1: available wind power scales
    linearly with wind speed and available solar power scales linearly with
    short-wave irradiance, each capped by the installed nameplate capacity.

    Parameters
    ----------
    wind_capacity, solar_capacity
        Installed nameplate capacity of the wind / solar farm, in MW.
    wind_speed_to_mw
        Linear coefficient converting wind speed (m/s) to power (MW).
    irradiance_to_mw
        Linear coefficient converting irradiance (W/m^2) to power (MW).
    """

    wind_capacity: float = 50.0
    solar_capacity: float = 50.0
    wind_speed_to_mw: float = 3.0
    irradiance_to_mw: float = 0.05


@dataclass(frozen=True)
class FinanceConfig:
    """Capital-budgeting parameters for the financial valuation layer.

    Parameters
    ----------
    unit_capex
        Overnight capital cost of the battery, in EUR per MWh of capacity.
    discount_rate
        Annual discount rate used for NPV / LCOE, as a fraction (e.g. 0.08).
    lifespan_years
        Economic life of the battery asset, in years.
    cvar_alpha
        Confidence level for Conditional Value at Risk (e.g. 0.95 means the
        worst 5% of outcomes are summarised).
    """

    unit_capex: float = 300_000.0
    discount_rate: float = 0.08
    lifespan_years: int = 10
    cvar_alpha: float = 0.95


@dataclass(frozen=True)
class ScenarioConfig:
    """Monte-Carlo scenario generation parameters for stochastic optimisation.

    Parameters
    ----------
    n_scenarios
        Number of Monte-Carlo weather scenarios to draw.
    ar_coefficient
        AR(1) persistence coefficient of the forecast error process (0-1).
        Higher values produce more temporally correlated (clustered) errors.
    wind_sigma, solar_sigma
        Standard deviation of the multiplicative forecast error innovations
        for wind and solar, as a fraction of the forecast value.
    seed
        Optional random seed for reproducible scenario draws.
    """

    n_scenarios: int = 500
    ar_coefficient: float = 0.7
    wind_sigma: float = 0.20
    solar_sigma: float = 0.25
    seed: int | None = 42
