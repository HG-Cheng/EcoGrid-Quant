# 🌍 EcoGrid-Quant: Microgrid Dispatch & Climate Risk Engine

> **An energy-informatics backtesting engine for optimizing wind-solar economic dispatch and quantifying ESG carbon risk.**

## 🎯 Project Vision
EcoGrid-Quant is an object-oriented, Python-based mathematical optimization engine. It bridges the gap between **Physical Energy Systems** (weather data, generation bounds) and **Quantitative Financial Risk** (ESG carbon tax penalties & CAPEX/OPEX valuation). 

It is designed to compute the most cost-effective power dispatch strategy under strict environmental constraints and perform sensitivity analysis for infrastructure sizing.

## 🚀 Core Architecture & Features
- **Data Ingestion Pipeline**: Automated fetching of hourly shortwave radiation and wind speed data from the Open-Meteo API (Munich, Germany).
- **Inter-temporal Dynamic Dispatch (MPC)**: Built with `scipy.optimize.minimize` (SLSQP solver) to execute multi-period non-linear programming (NLP). It respects strict thermodynamic state-of-charge (SoC) transitions for Battery Energy Storage Systems (BESS).
- **Dual-Layer Attribution**: Advanced backtesting visualizations separating physical asset dispatch (Wind/Solar/Coal) from financial risk (Carbon Penalties).
- **Climate Risk vs. Asset Sizing (MACC)**: Deep grid-search sensitivity analysis computing the optimal capital expenditure (CAPEX) for BESS against varying carbon tax scenarios (OPEX).

## 🗺️ Roadmap Milestones
- [x] **V1**: Static Single-Period Optimization (Economic Dispatch + Carbon Tax)
- [x] **V2**: Inter-temporal Dynamic Dispatch (Model Predictive Control for BESS Arbitrage)
- [x] **V3**: Sensitivity Analysis & Asset Pricing (Marginal Abatement Cost Curve & CAPEX/OPEX Heatmaps)

## 🛠️ Quick Start & Reproducibility
The project strictly follows zero-allocation and defensive programming principles.

1. Install dependencies via `conda` or `pip`: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `tqdm`, `requests`.
2. Core solver logic is encapsulated in `core/dispatch_engine.py`.
3. Run `01_energy_data.ipynb` for real-world data ingestion and 72-hour MPC dynamic dispatch backtesting.
4. Run `02_sensitivity_macc.ipynb` for multi-universe grid search on Carbon Tax vs. Battery Capacity.

---
*Built for exploring the intersection of Energy Informatics, Systems Engineering, and Quantitative Finance.*
