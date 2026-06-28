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
  validation, audit reports, vendor placeholders, and calendars.
- `stocker_research`: features, labels, baseline reports, splits, and metrics.
- `stocker_backtest`: cost models and backtest interfaces.
- `stocker_execution`: broker abstraction, orders, risk, state, and paper broker.

## Separation Rules

Signal code should not place orders. Backtests should not know about live broker
credentials. Risk checks should be pure and testable. Execution should consume approved
orders and current state, not research notebooks.

Data trust is a separate boundary too. CSV ingestion, validation, audit reporting, and
baseline reporting happen before edge discovery. A dataset that fails audit should not
be used for backtests without a written reason.

This separation makes it easier to prove that a weak signal is weak, identify whether
a result came from costs or market behavior, and keep future live execution from
depending on exploratory research code.
