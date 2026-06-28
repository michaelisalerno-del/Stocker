# Stocker

Stocker is a from-scratch trading research and execution foundation. The first goal is
not to find an edge or place trades. The goal is to make bad ideas cheap to disprove on
a Mac, while keeping any future server execution small, boring, and protected by hard
risk boundaries.

## Repo Split

- `apps/desktop/`: macOS research workspace for notebooks, data audits, baseline
  research, feature experiments, and backtest reports.
- `apps/server/`: Linux execution workspace for dry runs, paper execution, future
  broker adapters, state reconciliation, and monitoring hooks.
- `packages/stocker_core/`: config, logging, time, shared types, and CLI entry points.
- `packages/stocker_data/`: local dataset paths, Parquet storage, validators, and
  exchange-calendar helpers.
- `packages/stocker_research/`: features, labels, baselines, walk-forward splits, and
  research metrics.
- `packages/stocker_backtest/`: cost models and future vectorized/event-driven
  backtest interfaces.
- `packages/stocker_execution/`: broker interface, orders, paper broker placeholder,
  risk checks, and execution state.

## Python And Dependency Management

Stocker targets Python 3.12 because it is the stable choice for the current quant
Python stack. Python 3.13 is intentionally avoided for now until all research and
backtesting dependencies are boring there.

The repo uses `uv` with dependency groups:

- Default project dependencies: core config, logging, CLI, and settings libraries.
- `research`: heavy Mac research stack.
- `server`: lightweight server execution stack.
- `dev`: tests, linting, typing, and pre-commit.

## Bootstrap On Mac

```bash
bash scripts/bootstrap_mac.sh
```

The script checks for Homebrew, `uv`, and Python. It does not silently install global
software. If `uv` is missing, it prints the install command and exits.

After bootstrap:

```bash
uv run stocker check
uv run pytest
uv run jupyter lab apps/desktop/notebooks
```

## Bootstrap On Server

On a Linux server:

```bash
bash scripts/bootstrap_server.sh
```

The server bootstrap installs only core and `server` dependency groups:

```bash
uv sync --no-dev --group server
uv run --no-dev --group server stocker server dry-run --config configs/server.example.yaml
```

## Tests And Checks

```bash
bash scripts/test.sh
bash scripts/check.sh
```

`check.sh` runs Ruff format checks, Ruff linting, mypy, and pytest through `uv`.

## Intentionally Not Implemented Yet

- No broker integration.
- No live trading.
- No API keys or secrets.
- No data vendor ingestion.
- No strategy logic.
- No Docker, systemd, or deployment automation.
- No event-driven accounting engine beyond an explicit placeholder.

## Next Development Stages

1. Add a reproducible market-data ingest path with raw-data immutability.
2. Build strict data audit reports before researching signals.
3. Add null models and baseline comparisons.
4. Add cost-adjusted vectorized backtests only after a hypothesis is written down.
5. Add walk-forward and regime evaluation.
6. Add event-driven backtests for candidates that survive.
7. Add paper trading with stale-data checks and broker-state reconciliation.
8. Add tiny live tests only after paper results and operational safety are proven.
