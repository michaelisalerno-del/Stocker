# EODHD Vendor Adapter

EODHD is implemented as a data-vendor adapter inside `stocker_data`. The rest of
Stocker consumes normalized Parquet datasets and does not know about EODHD response
formats, endpoints, or API tokens.

## Configuration

Research config supports:

```yaml
data_vendors:
  eodhd:
    enabled: true
    base_url: "https://eodhd.com/api"
    api_token_env: "EODHD_API_TOKEN"
    default_fmt: "json"
    request_timeout_seconds: 30
    max_retries: 3
    save_raw_by_default: true
```

Set the token in your shell or local `.env` file. Never commit real tokens.

```bash
export EODHD_API_TOKEN="your_token_here"
```

EODHD symbols include an exchange suffix. For US Apple shares, use `AAPL.US`.

## EOD Historical Data

Dry run:

```bash
uv run stocker data fetch-eodhd-eod \
  --symbol AAPL.US \
  --from 2024-01-01 \
  --to 2024-02-01 \
  --period d \
  --instrument-type stock \
  --dry-run
```

Fetch and audit:

```bash
uv run stocker data fetch-eodhd-eod \
  --symbol AAPL.US \
  --from 2015-01-01 \
  --to 2026-06-28 \
  --period d \
  --instrument-type stock \
  --merge \
  --save-raw \
  --audit \
  --qa \
  --market-calendar XNYS
```

Supported EOD periods:

- `d` -> `1d`
- `w` -> `1w`
- `m` -> `1mo`

## Intraday Data

Dry run:

```bash
uv run stocker data fetch-eodhd-intraday \
  --symbol AAPL.US \
  --interval 1m \
  --from 2024-01-01 \
  --to 2024-06-01 \
  --instrument-type stock \
  --dry-run
```

Fetch and audit:

```bash
uv run stocker data fetch-eodhd-intraday \
  --symbol AAPL.US \
  --interval 1m \
  --from 2024-01-01 \
  --to 2024-06-01 \
  --instrument-type stock \
  --merge \
  --save-raw \
  --audit \
  --qa
```

Supported intervals:

- `1m`, chunked at 120 days
- `5m`, chunked at 600 days
- `1h`, chunked at 7200 days

Chunking keeps each request inside safe vendor spans. Chunk boundaries may overlap at
the edge; normalized data is sorted and deduped by timestamp before storage.

## Vendor QA

After fetching, run a vendor-specific QA report:

```bash
uv run stocker data qa-eodhd \
  --symbol AAPL.US \
  --timeframe 1d \
  --instrument-type stock \
  --market-calendar XNYS \
  --adjusted-price-policy adjusted_available \
  --require-raw
```

Outputs:

```text
data/reports/vendor_qa/AAPL.US_1d_eodhd_qa.md
data/reports/vendor_qa/AAPL.US_1d_eodhd_qa.json
```

The report checks:

- normalized validation counts
- raw JSON presence
- adjusted-close availability
- count of bars where `adjusted_close` differs from `close`
- missing exchange sessions when a calendar is supplied
- repeatable refresh guidance

Adjusted-close policy values:

- `raw_close`: research should use the raw `close` column unless explicitly changed.
- `adjusted_available`: warn when adjusted close differs from raw close.
- `require_adjusted_close`: fail QA if `adjusted_close` is missing.

Stocker does not silently replace `close` with `adjusted_close`. Corporate actions and
adjustment policy must be visible in reports before research uses the dataset.

## Raw And Processed Storage

Raw JSON, when enabled:

```text
data/raw/
  source=eodhd/
    endpoint=eod/
      symbol=AAPL.US/
        period=d/
          2015-01-01_2026-06-28.json
```

```text
data/raw/
  source=eodhd/
    endpoint=intraday/
      symbol=AAPL.US/
        interval=1m/
          2024-01-01_2024-04-30.json
```

Processed Parquet:

```text
data/processed/
  source=eodhd/
    instrument_type=stock/
      symbol=AAPL.US/
        timeframe=1m/
          data.parquet
```

## After Fetching

List the catalog:

```bash
uv run stocker data catalog
```

Audit:

```bash
uv run stocker data audit \
  --symbol AAPL.US \
  --timeframe 1m \
  --source eodhd
```

Run a basic baseline:

```bash
uv run stocker research baseline \
  --symbol AAPL.US \
  --timeframe 1m \
  --source eodhd
```

The research harness can use the dataset the same way it uses CSV-imported data.

## Refresh Rule

Refresh existing datasets with `--merge`, keep raw JSON, run audit, and run QA before
research:

```bash
uv run stocker data fetch-eodhd-eod \
  --symbol AAPL.US \
  --from <last_dataset_date> \
  --to <new_end_date> \
  --period d \
  --instrument-type stock \
  --merge \
  --save-raw \
  --audit \
  --qa \
  --market-calendar XNYS
```
