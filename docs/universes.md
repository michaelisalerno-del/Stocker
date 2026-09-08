# Universes

Stocker uses universes so runtime and research start from reproducible symbol sets instead of
random manual ticker tests.

## Runtime US Named Universes

Production runtime configuration supports:

- `US_ALL`: all eligible issues in the cached Nasdaq Trader Nasdaq-listed and other-listed files;
- `NASDAQ`: Nasdaq-listed eligible issues;
- `NYSE`: issues whose primary listing code is NYSE;
- `CUSTOM`: inline research/testing members; not the live method builder's universe.

The committed `universes/us-listed.csv` snapshot is sourced from the official Nasdaq Trader Symbol
Directory. It records source URLs, retrieval time, both source-file creation times, and supported
universe names. Refresh it explicitly with:

```bash
uv run stocker universe refresh-us-listings --output universes/us-listed.csv
```

The refresh excludes rows marked as ETFs or test issues (and Nasdaq NextShares). It does not guess
security type from issuer names. Stage 2 requests an IBKR stock contract for each member; warrants,
units, stale symbols, ambiguous symbols, and other non-stock issues therefore fail qualification,
are reported for that symbol, and do not stop the batch. Loading the snapshot itself makes no IBKR,
historical-bar, option, PRE, or order request.

The method owns universe construction. The dashboard selects authoritative membership for the
market and embeds it with the new run; refreshing listings does not mutate existing runs.
The minimal configuration has no active runs until a method run is created:

```yaml
named_universe_snapshot: ../universes/us-listed.csv
universes: []
runs: []
```

Legacy HOT_BY_VOLUME and Activity Shortlist sources remain for historical/research compatibility.
They cannot be selected as current Session HARD filters. Cap remains metadata/research input,
not a global live strategy control. See [ARCHITECTURE.md](ARCHITECTURE.md).

The older research universe data manager below remains separate. Its current provider is EODHD,
and its history is not a production or PAPER PRE input.

## Files

Committed examples:

```text
universes/manual/us_test_5.yaml
universes/manual/uk_test_5.yaml
```

Generated files:

```text
universes/generated/
data/universes/research_ready/
data/reports/universes/
```

Generated universe files, research-ready exports, and reports are ignored by git.

## Universe Definition

A universe YAML contains an id, name, description, source, creation timestamp, filters,
and symbols:

```yaml
id: us_large_liquid
name: US Large Liquid Stocks
description: US stocks generated from EODHD screener
source: eodhd_screener
created_at: "2026-06-28T00:00:00Z"
filters:
  exchange: US
  min_price: 5
  min_market_cap: 1000000000
  min_avgvol_200d: 500000
  sectors: []
  industries: []
symbols:
  - symbol: AAPL.US
    name: Apple Inc
    exchange: US
    currency: USD
    instrument_type: stock
```

Symbols are normalized to uppercase while preserving EODHD exchange suffixes such as
`.US`.

## Build From EODHD Screener

```bash
uv run stocker universe build-eodhd \
  --id us_large_liquid \
  --name "US Large Liquid Stocks" \
  --exchange US \
  --min-price 5 \
  --min-market-cap 1000000000 \
  --min-avgvol-200d 500000 \
  --limit 500 \
  --output universes/generated/us_large_liquid.yaml \
  --config configs/research.example.yaml
```

Use `--dry-run` first. Dry-run prints the planned pages, filters, sort, output path,
and vendor enabled state without requiring a token.

Supported first-pass filters:

- `--exchange`
- `--min-price`
- `--min-market-cap`
- `--min-avgvol-200d`
- repeated `--sector`
- repeated `--industry`

Default sort is `market_capitalization.desc`.

## Validate

```bash
uv run stocker universe validate \
  --universe universes/generated/us_large_liquid.yaml
```

Validation checks that required fields exist, symbols exist, duplicate symbols are
rejected, symbols are uppercase, and numeric metadata is sensible.

## Batch Fetch

```bash
uv run stocker universe fetch \
  --universe universes/generated/us_large_liquid.yaml \
  --from 2018-01-01 \
  --to 2026-06-28 \
  --timeframe 1d \
  --source eodhd \
  --merge \
  --audit \
  --qa \
  --market-calendar XNYS \
  --max-symbols 20 \
  --config configs/research.example.yaml
```

The universe layer orchestrates. It does not duplicate vendor fetch logic. Daily,
weekly, and monthly requests use the existing EODHD EOD path. Intraday requests use the
existing chunked EODHD intraday path.

Batch safety options:

- `--max-symbols`
- `--fail-fast`
- `--sleep-seconds-between-symbols`
- `--resume`
- `--skip-existing`
- `--overwrite`
- `--merge`

`--overwrite` and `--merge` are mutually exclusive. The default is conservative and
does not delete existing datasets.

Fetch reports are written to:

```text
data/reports/universes/<universe_id>_<timeframe>_fetch.md
data/reports/universes/<universe_id>_<timeframe>_fetch.json
```

Each symbol result includes status, rows fetched/saved, date range, output path,
audit/QA paths, errors, and duration.

## Qualify

Screener metadata is not enough. Stocker qualifies a universe using actual local
historical data:

```bash
uv run stocker universe qualify \
  --universe universes/generated/us_large_liquid.yaml \
  --timeframe 1d \
  --source eodhd \
  --min-history-days 750 \
  --min-last-close 5 \
  --min-median-dollar-volume-60d 10000000 \
  --output data/universes/research_ready/us_large_liquid_1d.json
```

Qualification calculates row count, date range, last close, median 60-day volume,
median 60-day dollar volume, optional 200-day dollar volume, validation errors, missing
session warnings, zero-volume warnings, and large-jump warnings. Symbols are rejected
with explicit reasons such as `missing_dataset`, `insufficient_history`,
`low_last_close`, `low_dollar_volume`, `validation_errors`, or `missing_sessions`.

The JSON output is the Stage 3 handoff:

```bash
uv run stocker research run-universe \
  --hypothesis research/hypotheses/examples/moving_average_momentum.yaml \
  --qualified-universe data/universes/research_ready/us_large_liquid_1d.json \
  --config configs/research.example.yaml \
  --max-symbols 20
```

The research runner does not select trades or paper trade. It only applies a written
hypothesis across qualified symbols, writes per-symbol reports, aggregates conservative
classifications, and updates the research index.

## Health

```bash
uv run stocker universe health \
  --universe universes/generated/us_large_liquid.yaml \
  --timeframe 1d \
  --source eodhd
```

Health reports summarize total symbols, present/missing datasets, fetch failures from
the latest fetch report, audit pass/warn/fail counts, QA pass/warn/fail counts, row
count coverage, date coverage, research-ready count, rejected count, top rejection
reasons, and the next recommended command.

## Bias Notes

Universe management reduces ad hoc selection, but it does not eliminate survivorship
or selection bias. A current screener universe can miss delisted names and historical
index membership changes. Treat EODHD screener universes as a practical starting point,
then document filters, export universe files, and avoid repeatedly changing the
universe to rescue a weak hypothesis.

## Intentionally Not Implemented

- No FMP integration.
- No broker execution.
- No live trading.
- No dashboard.
- No strategy discovery or parameter optimization.
- No automatic universe strategy mining.
