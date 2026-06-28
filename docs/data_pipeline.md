# Data Pipeline

The data pipeline makes local market data trustworthy enough for basic research. It
supports local CSV import and EODHD vendor downloads. It does not add broker
execution, live trading, or strategy optimization.

## Accepted CSV Input

CSV import accepts common OHLCV column names:

- Timestamp: `datetime`, `date`, `timestamp`, `time`
- Open: `open`, `o`
- High: `high`, `h`
- Low: `low`, `l`
- Close: `close`, `c`
- Volume: `volume`, `vol`, `v`

Optional columns include `bid`, `ask`, `spread`, `spread_bps`, `adjusted_close`,
`corporate_action_flag`, and `session`.

If automatic mapping is not enough, pass explicit mapping:

```bash
uv run stocker data import-csv \
  --file path/to/file.csv \
  --symbol AAPL \
  --source manual \
  --timeframe 1d \
  --instrument-type stock \
  --timezone America/New_York \
  --column-map "timestamp=Date,open=O,high=H,low=L,close=C,volume=Vol"
```

## Canonical Schema

Required fields:

`source`, `symbol`, `instrument_type`, `timeframe`, `timestamp`, `open`, `high`,
`low`, `close`, `volume`, `currency`, `timezone`.

Volume can be missing for instruments that do not provide useful volume, but the audit
will warn clearly. Timestamps are timezone-aware internally and data is sorted before
storage.

## Storage Layout

Imported data is written to partitioned Parquet:

```text
data/processed/
  source=manual/
    instrument_type=stock/
      symbol=AAPL/
        timeframe=1d/
          data.parquet
```

`data/catalog.json` is regenerated after import and can be refreshed with:

```bash
uv run stocker data catalog
```

EODHD data uses the same processed layout with `source=eodhd`:

```text
data/processed/
  source=eodhd/
    instrument_type=stock/
      symbol=AAPL.US/
        timeframe=1m/
          data.parquet
```

Raw EODHD JSON can optionally be saved under `data/raw/source=eodhd/...`.

## Validation Checks

Validation returns structured issues with severity, code, message, count, and first/last
seen values where available.

Checks include:

- duplicate timestamps
- missing or unparseable timestamps
- non-monotonic timestamp order
- timezone-naive timestamps
- OHLC high/low containment
- non-positive prices
- negative volume
- suspicious zero-volume runs
- calendar-aware gaps based on timeframe cadence
- large close-to-close jumps
- missing sessions when a market calendar is supplied

For stock data, pass a market calendar such as `XNYS` when you want strict session
gap checks:

```bash
uv run stocker data audit \
  --symbol AAPL.US \
  --timeframe 1d \
  --source eodhd \
  --market-calendar XNYS
```

Daily data with `XNYS` checks exchange sessions, so weekends and holidays are not
flagged. Intraday data with `XNYS` checks expected bars only inside regular sessions,
so overnight closures are not flagged. Without a market calendar, Stocker records an
informational `calendar_gap_check_skipped` issue instead of creating noisy weekend or
overnight false positives.

## Audit Reports

Run:

```bash
uv run stocker data audit --symbol AAPL --timeframe 1d
```

Outputs:

```text
data/reports/audits/AAPL_1d_audit.md
data/reports/audits/AAPL_1d_audit.json
```

The audit report summarizes row count, date range, validation findings, gap/duplicate
status, OHLC sanity, volume quality, return distribution, volatility, and largest
up/down bars. A dataset with warnings or errors should not be treated as backtest-safe
without a written reason.

## Baseline Reports

Run:

```bash
uv run stocker research baseline --symbol AAPL --timeframe 1d
```

Outputs:

```text
data/reports/baselines/AAPL_1d_baseline.md
data/reports/baselines/AAPL_1d_baseline.json
```

Baselines are intentionally simple:

- buy and hold
- always flat
- random entry/exit with fixed seed
- simple moving-average momentum
- simple mean reversion

Metrics include gross and net return, annualized return when timeframe allows,
volatility, Sharpe-like ratio, max drawdown, win rate, number of trades, exposure, and
estimated costs.

## Why Audit Comes First

Bad timestamps, duplicate bars, impossible OHLC values, negative volume, and unhandled
gaps can create fake edges. Stocker’s research process starts by proving the data is
usable before running baselines, hypotheses, or backtests.

## EODHD Vendor Ingestion

EODHD lives entirely inside `stocker_data.vendors.eodhd`. Strategy templates,
backtests, research experiments, and execution code consume normalized Stocker
datasets and do not call vendor APIs.

Fetch commands load research config by default:

```bash
uv run stocker data fetch-eodhd-eod --config configs/research.example.yaml --help
```

The config supplies:

- `data.data_dir`
- `data.default_currency`
- `data_vendors.eodhd.enabled`
- `data_vendors.eodhd.base_url`
- `data_vendors.eodhd.api_token_env`
- `data_vendors.eodhd.request_timeout_seconds`
- `data_vendors.eodhd.max_retries`
- `data_vendors.eodhd.save_raw_by_default`

Dry-runs do not require the token and may run while the vendor is disabled, but they
print the disabled state. Live fetches require the vendor to be enabled unless
`--enable-disabled-vendor` is passed deliberately.

Dry-run a download without a token:

```bash
uv run stocker data fetch-eodhd-eod \
  --config configs/research.example.yaml \
  --symbol AAPL.US \
  --from 2024-01-01 \
  --to 2024-02-01 \
  --period d \
  --instrument-type stock \
  --dry-run
```

Run a real EOD fetch after setting the token:

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

Intraday ranges are chunked to stay inside EODHD-safe spans:

```bash
uv run stocker data fetch-eodhd-intraday \
  --config configs/research.example.yaml \
  --symbol AAPL.US \
  --interval 1m \
  --from 2024-01-01 \
  --to 2024-06-01 \
  --instrument-type stock \
  --merge \
  --save-raw \
  --audit
```

After fetching:

```bash
uv run stocker data catalog
uv run stocker data audit \
  --symbol AAPL.US \
  --timeframe 1m \
  --source eodhd \
  --market-calendar XNYS
uv run stocker data qa-eodhd \
  --symbol AAPL.US \
  --timeframe 1m \
  --source eodhd \
  --market-calendar XNYS \
  --require-raw
uv run stocker research baseline --symbol AAPL.US --timeframe 1m --source eodhd
```

Vendor QA writes Markdown/JSON under `data/reports/vendor_qa/` and summarizes raw
response coverage, adjusted-close policy, calendar validation, and refresh guidance.
Raw coverage is dataset-specific: daily QA looks for `endpoint=eod/period=d`, while
minute QA looks for `endpoint=intraday/interval=1m`. A daily raw file cannot satisfy
intraday QA, and an intraday raw file cannot satisfy daily QA.

## Local EODHD Smoke

Run:

```bash
bash scripts/smoke_eodhd_local.sh
```

The script always runs EOD and intraday dry-runs. If `EODHD_API_TOKEN` is set, it
fetches a tiny EOD sample into `data_smoke/`, then runs catalog, audit, EODHD QA, and
baseline. If the token is missing, it prints that the live smoke was skipped and exits
successfully.

## CI Expectations

GitHub Actions runs `uv sync --all-groups`, Ruff formatting, Ruff linting, mypy, and
pytest on push and pull request. CI does not need `EODHD_API_TOKEN`; all vendor HTTP
tests use mocks.

Before starting Stage 3 research harness work, the local check script, CI, EODHD
dry-runs, one tiny real EOD fetch, catalog, audit, vendor QA, baseline, and a merge
re-run without duplicate rows should all pass.

## Universe Data Manager

Stage 2.7 adds a universe workflow on top of the single-symbol data pipeline:

```text
EODHD screener or manual universe file
  -> universe YAML/JSON
  -> batch EODHD fetch
  -> audit every dataset
  -> EODHD QA every dataset
  -> local liquidity/history filters
  -> research-ready universe export
```

Universe files live under `universes/manual/` or `universes/generated/`. Generated
research-ready exports live under `data/universes/research_ready/`, and universe
reports live under `data/reports/universes/`.

Example dry-run fetch:

```bash
uv run stocker universe fetch \
  --universe universes/manual/us_test_5.yaml \
  --from 2024-01-01 \
  --to 2024-02-01 \
  --timeframe 1d \
  --source eodhd \
  --dry-run \
  --max-symbols 2 \
  --config configs/research.example.yaml
```

See [universes.md](universes.md) for the complete build, fetch, qualify, and health
workflow.
