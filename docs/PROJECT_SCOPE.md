# Risk-Aware Energy System Operation under Weather Uncertainty

*A Reproducible Framework for Stochastic Dispatch, Tail-Risk Control, and Rolling-Horizon Decision-Making*

The repository name remains **EcoGrid-Quant**.

## Project Purpose

This project studies how a single-node, grid-connected energy system should coordinate battery storage and an external grid connection when future available photovoltaic generation is uncertain and represented through forecast errors. The research will examine how operational decisions can control average operating cost while reducing the high costs, load shedding, and reliability risks caused by adverse and extreme photovoltaic forecast deviations.

The project is intended to establish a focused and reproducible research benchmark. This document defines that intended scope; it does not assert that every capability described below has been implemented or validated.

## Core Research Question

For a fixed 24-hour single-node benchmark with uncertain photovoltaic generation, how do forecast-based deterministic, risk-neutral stochastic, and CVaR-based risk-averse operating policies compare in terms of out-of-sample operating cost, tail-cost exposure, and reliability risk when evaluated under the same data, physical assumptions, information constraints, and realized conditions?

The deterministic perfect-information solution serves as an ex post reference for measuring operational regret and the value of improved information. It is not treated as a directly attainable decision policy under forecast uncertainty.

The comparison should additionally examine whether reductions in tail risk and reliability risk require an increase in expected operating cost, and how that trade-off changes with scenario assumptions and the chosen level of risk aversion.

## Frozen First Benchmark

The first formal benchmark is limited to:

- one electrical node;
- photovoltaic generation as the only modeled source of operational uncertainty;
- battery energy storage;
- an external grid connection for electricity imports;
- an exogenous, non-flexible electrical load profile, with load uncertainty excluded;
- a 24-hour operating horizon; and
- hourly time resolution.

These elements define the fixed common physical and temporal boundary for model development, comparison, and review. Electricity export and any other addition that changes this boundary are follow-on extensions rather than prerequisites for the first benchmark. All decision methods in the formal comparison must use the same frozen system boundary.

## In-Scope Capabilities

The intended scope includes:

- a reproducible data foundation with explicit schemas, units, provenance, validation outcomes, and repair records;
- simple deterministic forecasting baselines and residual evaluation without future-information leakage;
- a deterministic operational baseline for the frozen benchmark;
- forecast-error scenario generation based on documented assumptions and separated training and evaluation periods;
- risk-neutral stochastic operation that minimizes expected scenario-weighted operating loss using explicitly specified costs and penalties;
- CVaR-based risk-averse operation that controls the tail of a clearly defined operational-loss distribution;
- rolling-horizon control that separates information available at decision time from later realizations;
- out-of-sample evaluation on observations or scenarios not used to make the corresponding decisions;
- reliability, cost, and risk metrics suitable for comparing decision layers; and
- reproducible tests and documentation, including small cases that can be checked manually.

The exact operating-loss definition, cost components, penalties, risk parameters, and CVaR target belong in `MODEL_SPECIFICATION.md`. Their wording here must not be interpreted as a completed mathematical formulation.

The benchmark is designed to support controlled comparisons among these capabilities. Their inclusion here describes the research target, not implementation status.

## Explicitly Out of Scope for the First Benchmark

The frozen first benchmark excludes:

- electricity export to the external grid;
- any source of operational uncertainty other than PV forecast error, including uncertainty in electrical load, electricity prices, or external-grid availability and outages;
- flexible or price-responsive demand;
- energy assets beyond the frozen PV, battery, and import-only grid configuration, including wind generation, on-site conventional generation, and electric vehicles;
- unit-commitment, investment, asset-sizing, and capacity-expansion decisions;
- multi-node electricity networks, associated transmission or distribution network constraints, and AC or DC power-flow modeling;
- carbon-intensity optimization;
- multi-energy coupling;
- reinforcement-learning control and deep-learning forecasting; and
- a web frontend.

The first-benchmark data foundation is not required to reserve fields or interfaces for wind or any other excluded capability. Any later inclusion of an excluded capability must be identified as an explicit extension and must not become a required input to, or silently alter, the frozen first-benchmark comparison case.

## Decision Policies and Evaluation Structure

The project will distinguish three forecast-based decision-policy classes from an ex post reference case and a cross-cutting execution protocol. These roles are not five parallel alternatives.

**Operational decision policies.** Under the information available at the relevant decision time, the first benchmark will compare:

1. a forecast-deterministic policy, which uses the available point-forecast trajectory to represent PV generation and does not represent PV forecast uncertainty within that optimization;
2. a risk-neutral stochastic policy, which represents PV forecast uncertainty through a documented probabilistic representation and assesses decisions by expected operational performance; and
3. a CVaR-based risk-averse policy, which incorporates tail-risk control so that trade-offs between expected operational performance and tail-risk exposure can be evaluated.

The probabilistic semantics of stochastic inputs, including any scenario probabilities or weights, must be documented. Their construction and numerical values are not defined in this scope document.

**Perfect-foresight reference.** A deterministic optimization using realized benchmark inputs over the relevant evaluation horizon will serve as an ex post ideal-information reference under the same frozen system boundary. It is not a deployable policy and must not be interpreted as a competitor operating under the same information set as the forecast-based policies. It supports regret and value-of-information assessment.

**Rolling-horizon execution and evaluation.** Rolling horizon is a cross-cutting protocol applied to each of the three forecast-based policies, not a separate policy class. In out-of-sample operation, at each hourly decision time, the applicable policy will be re-solved over the 24-hour look-ahead horizon using the current system state and only the information then available. Only the first-hour decisions will be implemented; the realized outcome will then be observed, the system state, including battery state of charge, will be updated, and the process will repeat. The resulting comparison will include the perfect-foresight reference and the deterministic, risk-neutral stochastic, and CVaR-based risk-averse rolling-horizon policies under the same frozen system boundary.

This scope section fixes these roles and information restrictions only. It does not define the scenario structure, construction, or numerical probabilities; decision variables or timing; first- and second-stage variables; recourse or non-anticipativity equations; objective functions, constraints, or operating loss; battery-degradation, load-shedding, or terminal-SOC treatment; or the CVaR target, confidence level, risk weight, or role within the optimization. Those choices belong in [MODEL_SPECIFICATION.md](MODEL_SPECIFICATION.md) and must be applied consistently across formal comparisons.

## Evaluation Scope

Planned evaluation measures include:

- expected realized operating loss;
- Value at Risk (VaR) and Conditional Value at Risk (CVaR) of the same realized-operating-loss random variable;
- realized energy not served (ENS) and expected energy not served (EENS);
- loss-of-load probability (LOLP);
- total grid energy imported and peak grid import;
- PV curtailment;
- battery throughput, reported as a physical-use measure rather than as a definition of battery degradation or degradation cost;
- computational time; and
- out-of-sample regret relative to the same-period deterministic perfect-foresight reference.

Policy-to-policy differences may also be reported as performance gaps, with the compared policies and baseline identified explicitly.

These metric families define the intended evaluation scope. Their inclusion does not imply that they are implemented, computed, or validated. `MODEL_SPECIFICATION.md` will define the realized operating-loss random variable and the exact conventions for VaR and CVaR; the accounting and aggregation of ENS and EENS; the loss-of-load event and aggregation convention for LOLP; the accounting of grid imports, PV curtailment, and battery throughput; and the mathematical definition of regret. Reporting ex post VaR or CVaR does not determine the random variable, confidence level, or risk weight used by a risk-averse optimization policy.

ENS is a pathwise energy quantity over a stated evaluation period; EENS is its expectation over a stated evaluation population or sample. Expected, tail, and probability-based measures must identify that population or sample and any probability or empirical weights, which must not be conflated with internal forecast-scenario weights from an individual optimization solve. Every reported metric must state its units, evaluation period, aggregation, weighting, and comparison baseline where applicable. The experiment protocol will define the computational-time measurement convention and the reporting of solve, feasibility, and validation failures.

## Reproducibility Principles

Work within this scope will follow these principles:

- assumptions and experiment settings are configuration-driven, and the effective configuration is recorded with each result;
- physical and monetary units are explicit;
- each result is traceable to its data source, retrieval time or snapshot/version identifier, transformation lineage, and applicable model, code, and solver versions;
- source-data snapshots are treated as immutable inputs, and cleaning, repair, and other transformations produce traceable derived data;
- data checks are separated from data repairs;
- records are not silently dropped, filled, clipped, converted, or otherwise altered;
- each permitted repair is traceable to the affected records, the rule applied, and the resulting derived data;
- stochastic procedures use configurable random seeds where applicable, and the effective seeds and relevant stochastic settings are recorded; unavoidable nondeterminism is disclosed;
- all time-dependent preprocessing, forecasting, scenario construction, policy execution, and evaluation respect temporal ordering and use only information available at the relevant decision time;
- training, calibration, validation, and test roles and data splits are separated and documented where those stages apply;
- outputs support both machine-readable analysis and human-readable review;
- small, manually verifiable cases accompany larger experiments and exercise key physical and accounting invariants;
- automated tests cover important data, transformation, and model invariants; and
- where a numerical solver is used, its identity, version, and material settings, including applicable tolerances, are recorded, and its termination status, feasibility, and relevant residuals are checked against stated acceptance criteria before results are accepted.

Validation failures and solver outcomes that do not meet the stated acceptance criteria must remain visible. No undocumented repair, fallback, filtering, or relabeling may convert a failed or unverified run into an apparently successful result.

## Software and Solver Boundary

Pyomo with HiGHS is the intended mathematical reference implementation for the first benchmark. The project will not immediately migrate its full implementation to PyPSA or Linopy.

After the reference model is stable and verified, a small benchmark may be reproduced with PyPSA and Linopy to support independent comparison, documentation, testing, and open-source collaboration. Such a bridge must preserve the frozen system boundary and clearly document any difference in model semantics.

Operational two-stage stochastic formulations must handle non-anticipativity explicitly: decisions made before uncertainty is revealed must be shared where required, while recourse decisions may respond only to information available at their decision stage. The project must not assume that a default investment-oriented scenario structure automatically represents this operational requirement. No unverified PyPSA API is specified here.

## Evidence and Learning Objectives

The project is intended to produce reviewable evidence of capability in:

- probabilistic modeling;
- uncertainty quantification;
- reliability and risk assessment;
- decision-making under uncertainty;
- energy and infrastructure risk;
- climate and weather uncertainty;
- reproducible scientific software; and
- open-source collaboration.

The resulting benchmark, documentation, tests, and reproducible experiments may support future postgraduate research applications and open-source contributions. This purpose does not change the document into a personal statement and does not imply affiliation with any unpublished program or organization.

## Scope Control

New work must first be compared with the frozen first benchmark. A feature belongs to the current scope only when it strengthens the defined data foundation, decision layers, evaluation, reproducibility, or verification without silently expanding the benchmark system boundary.

Capabilities that add assets, network structure, energy carriers, forecasting methods, interfaces, or other excluded elements are follow-on extensions. Their existence elsewhere in the repository does not make them part of the first benchmark.

File names, README descriptions, and historical labels are not evidence that a research stage is complete. Code, automated tests, and reproducible experimental results take precedence over historical README claims when assessing capability. Detailed mathematical definitions belong in `MODEL_SPECIFICATION.md`; executable experiment settings belong in `../configs/`; usage and verified results belong in `../README.md`; and current progress or status must not be recorded in this scope document.
