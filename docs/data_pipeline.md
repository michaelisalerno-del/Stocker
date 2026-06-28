# Data Pipeline

Stage 2 makes local market data trustworthy enough for basic research. It does not add
vendor downloads, broker execution, live trading, or strategy optimization.

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
- gaps based on timeframe cadence
- large close-to-close jumps
- missing sessions when a market calendar is supplied

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
