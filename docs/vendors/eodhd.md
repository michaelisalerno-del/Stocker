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

The EODHD CLI commands load `configs/research.example.yaml` by default. Use
`--config` to point at another research config. The config supplies the default data
directory, default currency, vendor base URL, token environment variable, timeout,
retry count, and default raw-response behavior.

Dry-runs do not require the API token and may run even if `enabled: false`; the output
prints the disabled state. Live fetches require `enabled: true` unless
`--enable-disabled-vendor` is passed explicitly.

Set the token in your shell or local `.env` file. Never commit real tokens.

```bash
export EODHD_API_TOKEN="your_token_here"
```

EODHD symbols include an exchange suffix. For US Apple shares, use `AAPL.US`.

## EOD Historical Data

Dry run:

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

Fetch and audit:

```bash
uv run stocker data fetch-eodhd-eod \
  --config configs/research.example.yaml \
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
  --config configs/research.example.yaml \
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
  --config configs/research.example.yaml \
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

## Retry And Rate Limits

The adapter retries temporary transport failures and HTTP `429`, `500`, `502`, `503`,
and `504` responses up to the configured attempt count. It does not retry permanent
client/auth errors such as `400`, `401`, `403`, or `404`, nor schema or empty-response
errors. If EODHD returns `Retry-After` on a rate limit response, the client respects
that delay. Error messages include status code, endpoint path, a short redacted body
preview, and whether retries were exhausted. API tokens are not printed.

## Vendor QA

After fetching, run a vendor-specific QA report:

```bash
uv run stocker data qa-eodhd \
  --config configs/research.example.yaml \
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
- dataset-specific raw JSON presence
- raw endpoint and selector, such as `endpoint=eod` plus `period=d`
- whether matching raw files appear to cover the processed dataset date range
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

Raw coverage is scoped to the dataset under review:

- EOD daily QA looks under `raw/source=eodhd/endpoint=eod/symbol=AAPL.US/period=d/`.
- EOD weekly/monthly QA uses `period=w` or `period=m`.
- Intraday QA looks under `raw/source=eodhd/endpoint=intraday/symbol=AAPL.US/interval=1m/`.

A daily raw response cannot satisfy intraday QA, and an intraday raw response cannot
satisfy daily QA.

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
  --source eodhd \
  --market-calendar XNYS
```

Run a basic baseline:

```bash
uv run stocker research baseline \
  --symbol AAPL.US \
  --timeframe 1m \
  --source eodhd
```

The research harness can use the dataset the same way it uses CSV-imported data.

## Screener Universes

EODHD is also the first universe source. Stocker uses `/api/screener` through
`stocker_data.vendors.eodhd`; FMP is intentionally not used in this stage.

Dry-run a screener build without calling EODHD:

```bash
uv run stocker universe build-eodhd \
  --id us_large_liquid \
  --name "US Large Liquid Stocks" \
  --exchange US \
  --min-price 5 \
  --min-market-cap 1000000000 \
  --min-avgvol-200d 500000 \
  --limit 100 \
  --output universes/generated/us_large_liquid.yaml \
  --dry-run \
  --config configs/research.example.yaml
```

The screener adapter supports `filters`, `signals`, `sort`, `limit`, and `offset`.
Page requests are limited to 1-100 rows, offset starts at 0, and the adapter refuses
requests that would exceed the documented safe offset guardrail. Tokens are never
printed in dry-runs or error messages.

The universe layer saves deterministic YAML, then batch fetches historical data
through the same EODHD EOD/intraday functions documented above.

## Calendar-Aware Validation

For US stocks, use `--market-calendar XNYS` when auditing or QA-ing daily and intraday
datasets. Daily validation checks exchange sessions so weekends and holidays are not
false gaps. Intraday validation checks only regular session bars, so overnight
closures are not false gaps. Without a calendar, Stocker records an informational
`calendar_gap_check_skipped` issue instead of producing noisy false warnings.

## Local Smoke

Run:

```bash
bash scripts/smoke_eodhd_local.sh
```

The script always runs EOD and intraday dry-runs. If `EODHD_API_TOKEN` is set, it
fetches a tiny EOD sample into `data_smoke/`, then runs catalog, audit, EODHD QA, and
baseline. `data_smoke/` is ignored by git.

## CI

GitHub Actions runs on push and pull request using Python 3.12 and `uv`. It installs
all groups, checks formatting, lints, type checks, and runs pytest. CI does not require
an EODHD token because all HTTP tests are mocked.

## Refresh Rule

Refresh existing datasets with `--merge`, keep raw JSON, run audit, and run QA before
research:

```bash
uv run stocker data fetch-eodhd-eod \
  --config configs/research.example.yaml \
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
