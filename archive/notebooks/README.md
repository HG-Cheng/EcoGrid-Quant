# Archived teaching prototypes

These five notebooks preserve the earlier coal/wind/PV unit-commitment and
investment experiments. Their bytes were retained when moved here. They are
not executed by the active CLI or tests; stored outputs are historical artifacts,
not evidence for the new PV/battery/import-grid benchmark.

The retained `core/dispatch_engine.py`, `ecogrid/engines/`, `ecogrid/backtest.py`,
and `ecogrid/scenarios.py` support old experiments only. In particular, the old
rolling replay can violate commitment constraints and energy accounting, and
old stochastic risk reports/exclusivity have known limitations. Do not use their
regret or risk outputs as validated research results. Financial risk functions
in `ecogrid/finance.py` have been corrected and are shared by the active model.

For the preserved original paths/import assumptions use the historical baseline
`b37931e` or the existing backup branch in a separate checkout. This upgrade did
not alter the backup branch. The active workflow is documented in ../../README.md.
