# 🌍 EcoGrid-Quant: Microgrid Dispatch & Climate Risk Engine

> **An energy-informatics backtesting engine for optimizing wind-solar economic dispatch and quantifying grid carbon emission risks.**

## 🎯 Project Vision
EcoGrid-Quant is an object-oriented, Python-based mathematical optimization engine. It bridges the gap between **Physical Energy Systems** (weather data, generation bounds) and **Quantitative Financial Risk** (ESG carbon tax penalties). 

It is designed to automatically compute the most cost-effective power dispatch strategy under strict environmental constraints.

## 🚀 Core Features (V1.0)
- **Real-World Data Ingestion**: Automated pipeline fetching hourly shortwave radiation and wind speed data from the Open-Meteo Satellite API (Target Location: Munich, Germany).
- **NLP Optimization Engine**: Built with `scipy.optimize.minimize` (SLSQP solver) to execute non-linear programming for supply-demand balancing.
- **Dual-Layer Attribution**: Advanced backtesting visualizations separating physical asset dispatch (Wind/Solar/Coal) from financial risk (Carbon Penalties).

## 🗺️ Roadmap (Next Steps)
- [x] **V1**: Static Single-Period Optimization (Economic Dispatch + Carbon Tax)
- [ ] **V2**: Inter-temporal Dynamic Dispatch (Introducing Battery Energy Storage Systems - BESS)
- [ ] **V3**: Sensitivity Analysis (Marginal Abatement Cost Curve for Wind/Solar capacity sizing)

## 🛠️ Quick Start
1. Ensure you have the required dependencies: `pandas`, `requests`, `scipy`, `matplotlib`, `tqdm`.
2. Run the backtesting pipeline in `01_energy_data.ipynb`.
3. The core algorithmic solver is located in `core/dispatch_engine.py`.

---
*Built for exploring the intersection of Energy Informatics, Systems Engineering, and Climate Tech.*