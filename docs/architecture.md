# Architecture

Stocker is split into a heavy desktop research half and a lightweight server execution
half. The shared packages sit between them, but they do not collapse research,
backtesting, risk, and execution into one unit.

## Desktop And Research

The desktop side is allowed to be heavy. It can use notebooks, plotting libraries,
large local datasets, statistical tooling, and slow experiments. Its job is to reject
weak ideas before they get near execution.

Desktop responsibilities:

- Audit raw and processed data.
- Import local CSV data into a canonical OHLCV schema.
- Store audited datasets as partitioned Parquet.
- Query local datasets through DuckDB.
- Build baseline summaries and null comparisons.
- Load written hypothesis definitions before running experiments.
- Generate chronological walk-forward splits with embargo gaps.
- Check parameter stability rather than selecting one lucky setting.
- Label simple historical regimes and compare performance across them.
- Generate features and labels from audited data.
- Run vectorized and future event-driven backtests.
- Produce reports that document why an idea failed or deserves more testing.

## Server And Execution

The server side should be boring. It should run a small dependency set, load typed
configuration, evaluate risk checks, reconcile state, expose dry-run or paper behavior,
and eventually call a broker adapter through a narrow interface.

Server responsibilities:

- Load safe server configuration.
- Run only in `paper` mode until live execution is explicitly added.
- Block orders when risk limits fail.
- Keep future broker integrations behind `stocker_execution.broker.Broker`.
- Prefer observability and predictability over research flexibility.

## Shared Packages

- `stocker_core`: shared config, logging, time, CLI, and type helpers.
- `stocker_data`: CSV ingestion, schema, Parquet I/O, catalog, DuckDB queries,
  validation, audit reports, vendor adapters, vendor QA, and calendars.
- `stocker_research`: written hypotheses, features, labels, baseline reports,
  walk-forward splits, parameter grids, stability checks, leakage checks, regime
  labels, experiment runner, and research report indexes.
- `stocker_backtest`: cost models, transparent vectorized evaluation, and future
  event-driven interfaces.
- `stocker_execution`: broker abstraction, orders, risk, state, and paper broker.

## Separation Rules

Signal code should not place orders. Backtests should not know about live broker
credentials. Risk checks should be pure and testable. Execution should consume approved
orders and current state, not research notebooks.

Data trust is a separate boundary too. CSV ingestion, validation, audit reporting, and
baseline reporting happen before edge discovery. A dataset that fails audit should not
be used for backtests without a written reason.

Vendor APIs are data-pipeline concerns only. EODHD lives under
`stocker_data.vendors.eodhd`, normalizes responses to the Stocker OHLCV schema, writes
Parquet, refreshes the catalog, and produces audit/QA reports. Strategy templates,
backtests, research experiments, server runtime code, and future execution code should
not call EODHD directly.

Research discipline is another boundary. A strategy test should be attached to a
written hypothesis, chronological walk-forward split, explicit cost model, and
conservative classification. Random train/test splits are not valid for trading
research because they let future market regimes influence past decisions.

This separation makes it easier to prove that a weak signal is weak, identify whether
a result came from costs or market behavior, and keep future live execution from
depending on exploratory research code.
