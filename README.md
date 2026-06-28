# Stocker

![CI](https://github.com/michaelisalerno-del/Stocker/actions/workflows/ci.yml/badge.svg)

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
- `packages/stocker_research/`: written hypotheses, simple strategy templates,
  baselines, walk-forward splits, leakage checks, regime labels, stability analysis,
  and research reports.
- `packages/stocker_backtest/`: cost models and transparent vectorized/event-driven
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
uv run stocker data import-csv \
  --file tests/fixtures/market_data/clean_ohlcv.csv \
  --symbol AAPL \
  --source manual \
  --timeframe 1d \
  --instrument-type stock \
  --timezone America/New_York \
  --currency USD
uv run stocker data audit --symbol AAPL --timeframe 1d
uv run stocker research baseline --symbol AAPL --timeframe 1d
uv run stocker research run \
  --hypothesis research/hypotheses/examples/moving_average_momentum.yaml \
  --symbol AAPL \
  --timeframe 1d
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
uv sync --no-default-groups --group server
uv run --no-default-groups --group server stocker server dry-run --config configs/server.example.yaml
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
- No vendor credentials in the repo.
- No strategy optimization.
- No Docker, systemd, or deployment automation.
- No event-driven accounting engine beyond an explicit placeholder.

## Data Pipeline

The data pipeline supports local CSV import and EODHD vendor ingestion. Both paths
normalize to the same Stocker OHLCV schema, validate the result, write partitioned
Parquet under
`data/processed/source=.../instrument_type=.../symbol=.../timeframe=.../data.parquet`,
update `data/catalog.json`, and can generate audit and baseline reports.

EODHD fetch commands load `configs/research.example.yaml` by default. The config
controls the data directory, default currency, vendor base URL, token environment
variable, retry count, timeout, and default raw-response saving behavior. Dry-runs do
not require a token. Live fetches require `data_vendors.eodhd.enabled: true` unless
you explicitly pass `--enable-disabled-vendor`.

EODHD credentials are read only from the configured environment variable, normally
`EODHD_API_TOKEN`:

```bash
export EODHD_API_TOKEN="your_token_here"
uv run stocker data fetch-eodhd-eod \
  --config configs/research.example.yaml \
  --symbol AAPL.US \
  --from 2015-01-01 \
  --to 2026-06-28 \
  --period d \
  --instrument-type stock \
  --merge \
  --save-raw \
  --audit
```

See [docs/data_pipeline.md](docs/data_pipeline.md) before researching any edge.
See [docs/vendors/eodhd.md](docs/vendors/eodhd.md) for EODHD-specific commands.
See [docs/universes.md](docs/universes.md) for universe generation, batch fetch, and
research-ready exports.

Local EODHD smoke test:

```bash
bash scripts/smoke_eodhd_local.sh
```

Without `EODHD_API_TOKEN`, the script runs safe dry-runs and skips the live fetch.
With a token, it fetches a tiny EOD sample into `data_smoke/`, then runs catalog,
audit, vendor QA, and baseline checks.

Local Stage 3 research smoke:

```bash
bash scripts/research_smoke_local.sh
```

Without `EODHD_API_TOKEN`, the script skips live fetch and explains how to provide
existing local data. With a token, it fetches a bounded `us_test_5` sample, qualifies
a tiny research-ready universe, runs one moving-average momentum universe report, and
prints the qualified universe path plus Markdown/JSON report paths. Rejections are
expected and are not smoke failures.

## Continuous Integration

GitHub Actions runs on push and pull request with Python 3.12:

```bash
uv sync --all-groups
uv run ruff format --check .
uv run ruff check .
uv run mypy packages apps
uv run pytest
```

CI does not require an EODHD token. Vendor tests use mocked HTTP only.

## Stage 3 Research Readiness

- `bash scripts/check.sh` passes locally.
- GitHub Actions passes.
- EODHD dry-runs work without a token.
- One small real EODHD EOD fetch works with `EODHD_API_TOKEN`.
- `uv run stocker data catalog` sees the fetched dataset.
- Audit and EODHD QA reports are generated.
- Baseline reports can consume the EODHD dataset.
- Re-running the fetch with `--merge` does not duplicate rows.
- A research-ready universe export exists for the symbols under test.

## Research Harness

Stage 3 adds a hypothesis-first research harness. Each experiment starts from a YAML
hypothesis, loads an audited local Parquet dataset, builds chronological walk-forward
splits, evaluates a small guarded parameter grid, warms rolling indicators with
pre-window historical context while scoring only actual train/test rows, applies
explicit costs, checks train-side parameter selection, keeps the best test-return row
as diagnostic only, runs leakage checks, compares against cash and same-window long
buy-and-hold, applies a same-window deterministic null timing test, separates
preferred intraday/session-flat evidence from stricter swing evidence, reports
overnight/weekend/gap contribution where measurable, summarizes performance by simple
historical regimes, and writes Markdown/JSON reports under `data/reports/research/`.

The initial templates are deliberately basic: moving-average momentum, pullback in
uptrend, mean reversion after a large down day, and volatility breakout. They are test
vehicles for the harness, not claims of edge.

Most research results should still be rejected. This stage is not paper trading, live
trading, broker integration, dashboard work, ML, or automatic strategy mining.
Daily-bar results are useful for context and universe research, but they do not prove
session-flat tradability; swing candidates need exceptional evidence.

Example:

```bash
uv run stocker research run \
  --hypothesis research/hypotheses/examples/moving_average_momentum.yaml \
  --symbol AAPL.US \
  --timeframe 1d \
  --source eodhd \
  --config configs/research.example.yaml
```

Universe example:

```bash
uv run stocker research run-universe \
  --hypothesis research/hypotheses/examples/moving_average_momentum.yaml \
  --qualified-universe data/universes/research_ready/us_test_5_1d.json \
  --config configs/research.example.yaml \
  --max-symbols 5
```

See [docs/research_harness.md](docs/research_harness.md) for the full workflow and
classification rules.

## Universe Workflow

Stage 2.7 adds a stock-universe data manager so research starts from a reproducible
symbol set instead of ad hoc tickers:

```bash
uv run stocker universe validate --universe universes/manual/us_test_5.yaml
uv run stocker universe fetch \
  --universe universes/manual/us_test_5.yaml \
  --from 2024-01-01 \
  --to 2024-02-01 \
  --timeframe 1d \
  --source eodhd \
  --dry-run \
  --max-symbols 2
```

The intended flow is EODHD screener or manual YAML, batch EODHD fetch, audit/QA for
each dataset, local liquidity and history qualification, then a research-ready JSON
export consumed by `stocker research run-universe`. FMP and dashboard work are
intentionally deferred.

## Next Development Stages

1. Add stock-suitability scoring across qualified universes.
2. Expand event-driven accounting for candidates that survive the harness.
3. Add paper trading with stale-data checks and broker-state reconciliation.
4. Add tiny live tests only after paper results and operational safety are proven.
5. Add production monitoring, deployment, and operational runbooks only after the
   execution boundary is proven in paper mode.
