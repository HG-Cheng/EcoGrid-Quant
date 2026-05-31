# EcoGrid-Quant: Micro-grid Unit-Commitment & Climate-Risk Quant Engine

> An industrial-grade, object-oriented engine that fuses **physical energy
> systems** (weather-driven generation), **mixed-integer optimisation**
> (unit commitment under uncertainty) and **quantitative finance** (DCF / NPV,
> LCOE, CVaR) to price grid-storage assets and quantify climate risk.

## Why this exists

Real power grids are not continuous. Thermal units have a minimum stable
output, pay a fixed start-up cost, and must respect minimum up/down times. The
weather forecast is never perfect, and capital is never free. EcoGrid-Quant
models all of this with the tools an infrastructure fund or a transmission
operator would actually trust.

## Architecture

```
ecogrid/
├── config.py            # 不可变 dataclass：物理/财务/情景参数的单一真相源
├── data/ingestion.py    # Open-Meteo 拉取 + 本地缓存 + 天气->功率转换
├── engines/
│   ├── base.py          # DispatchEngine 抽象基类（统一接口）
│   ├── slsqp_engine.py  # 连续 NLP baseline（保留用于对标）
│   ├── milp_engine.py    # V4: Pyomo + HiGHS 机组启停 MILP
│   └── stochastic.py    # V5: 两阶段随机规划 + CVaR 鲁棒优化
├── scenarios.py         # V5: 蒙特卡洛 + AR(1) 风光情景生成
├── finance.py           # V6: NPV / LCOE / VaR / CVaR
├── backtest.py          # V8: joblib 并行网格扫描 + 滚动窗口回测
└── results.py           # 统一结果 dataclass（DispatchResult / StochasticResult）
```

**Legacy (V1–V3):** teaching notebooks and the original SLSQP MPC engine live
outside the package — see `core/dispatch_engine.py` and the root-level
`01_energy_data.ipynb` / `02_sensitivity_macc.ipynb`.

## Capability ladder (V4 → V8)

- **V4 — MILP unit commitment.** Binary on/off, start-up cost, minimum up/down
  times and charge/discharge exclusivity, solved with the open-source HiGHS
  solver via Pyomo. Benchmarked head-to-head against the legacy SLSQP engine.
- **V5 — Stochastic & robust optimisation.** Monte-Carlo weather ensembles with
  temporally-correlated AR(1) forecast errors; a two-stage program where the
  commitment is here-and-now and dispatch is recourse; an optional CVaR
  objective for tail-risk-averse (robust) schedules.
- **V6 — Project-finance valuation.** Discounted cash flow / NPV, levelised
  cost of energy (LCOE) and Conditional Value at Risk. The optimal battery size
  is chosen by `argmax(NPV)`, not by minimising nominal cost.
- **V7 — Engineering rigour.** Full type hints, NumPy-style docstrings, a
  `pytest` suite covering extreme boundaries, `ruff` + `mypy` clean, and a
  GitHub Actions CI matrix.
- **V8 — Performance & back-testing.** Multi-core grid sweeps with `joblib`, and
  a rolling-horizon out-of-sample back-test that quantifies the *regret* of
  imperfect forecasts versus perfect foresight.

## Quick start

```bash
# 1. Create the environment and install the package (editable) with dev tools
conda create -n ecogrid python=3.10 -y
conda activate ecogrid
pip install -e ".[dev]"      # installs pyomo, highspy, joblib, ruff, mypy, pytest

# 2. Verify the HiGHS solver is reachable through Pyomo
python -c "from pyomo.environ import SolverFactory; print(SolverFactory('appsi_highs').available())"

# 3. Run the test suite, linter and type checker
ruff check .
mypy ecogrid
pytest tests -q
```

For V1–V3 exploration only, open the legacy notebooks at the repo root; the
industrial API lives under `ecogrid/` (see Minimal usage below).

### Minimal usage

```python
import numpy as np
from ecogrid import GridConfig, MilpDispatchEngine

cfg = GridConfig(carbon_tax_rate=80.0, bess_capacity=120.0,
                 coal_pmin=25.0, startup_cost=2000.0, min_up_time=3)
engine = MilpDispatchEngine(cfg)
result = engine.solve(demand, wind_avail, solar_avail)   # numpy arrays, MW
print(result.total_cost, result.commitment)
```

## Notebooks

Run with the registered `ecogrid` Jupyter kernel:

- `notebooks/03_milp_vs_slsqp.ipynb` — V4 MILP vs SLSQP head-to-head.
- `notebooks/05_capital_valuation.ipynb` — V6 NPV / CVaR capital-allocation decision.
- `notebooks/06_rolling_backtest.ipynb` — V8 rolling-horizon back-test and parallel sweep.
- `01_energy_data.ipynb`, `02_sensitivity_macc.ipynb` — legacy V1-V3 exploration.

---
*Energy Informatics × Systems Engineering × Quantitative Finance.*
