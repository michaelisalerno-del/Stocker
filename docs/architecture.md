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
- Run written hypotheses across qualified universe exports without selecting trades.
- Generate features and labels from audited data.
- Run vectorized and future event-driven backtests.
- Produce reports that document why an idea failed or deserves more testing.

## Server And Prospective Evaluation

The server side should be boring. Its currently implemented operational behavior is
market-data-only prospective recording and shadow evaluation. V2 contracts represent
observations, signals, and unapproved proposals without broker authority.

Server responsibilities:

- Accept only `prospective_record` or `shadow` at the V2 contract boundary.
- Keep idea plugins independent of broker, account, risk, and execution capabilities.
- Reject paper and live modes because neither is implemented.
- Prefer observability and predictability over research flexibility.

## Shared Packages

- `stocker_core`: shared config, logging, time, CLI, and type helpers.
- `stocker_data`: CSV ingestion, schema, Parquet I/O, catalog, DuckDB queries,
  validation, audit reports, vendor adapters, vendor QA, and calendars.
- `stocker_research`: written hypotheses, features, labels, baseline reports,
  walk-forward splits, parameter grids, stability checks, leakage checks, regime
  labels, single-symbol and universe experiment runners, and research report indexes.
- `stocker_backtest`: cost models, transparent vectorized evaluation, and future
  event-driven interfaces.
- `stocker_runtime`: V2 authority-free domain DTOs and first-party plugin contracts.

## Separation Rules

Signal code must not place orders. Backtests should not know about broker credentials.
No current V2 contract represents approval, an executable order, an account, or broker
state. Those boundaries require separate owner-approved phases.

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
