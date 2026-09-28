#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-data_smoke}"
CONFIG_PATH="${STOCKER_RESEARCH_CONFIG:-configs/research.example.yaml}"

if [[ -z "${EODHD_API_TOKEN:-}" && -f .env ]]; then
  EODHD_API_TOKEN="$(
    grep -E '^EODHD_API_TOKEN=' .env | tail -n 1 | cut -d '=' -f 2- || true
  )"
  export EODHD_API_TOKEN
fi

echo "Running safe EODHD dry-runs..."
uv run stocker data fetch-eodhd-eod \
  --config "$CONFIG_PATH" \
  --symbol AAPL.US \
  --from 2024-01-01 \
  --to 2024-02-01 \
  --period d \
  --instrument-type stock \
  --dry-run

uv run stocker data fetch-eodhd-intraday \
  --config "$CONFIG_PATH" \
  --symbol AAPL.US \
  --interval 1m \
  --from 2024-01-01 \
  --to 2024-01-05 \
  --instrument-type stock \
  --dry-run

if [[ -z "${EODHD_API_TOKEN:-}" ]]; then
  echo "EODHD_API_TOKEN is not set; live smoke fetch skipped."
  exit 0
fi

echo "EODHD_API_TOKEN is set; running tiny live EOD smoke into ${SMOKE_DATA_DIR}."
uv run stocker data fetch-eodhd-eod \
  --config "$CONFIG_PATH" \
  --symbol AAPL.US \
  --from 2024-01-02 \
  --to 2024-01-05 \
  --period d \
  --instrument-type stock \
  --data-dir "$SMOKE_DATA_DIR" \
  --merge \
  --save-raw \
  --audit \
  --qa \
  --market-calendar XNYS

uv run stocker data catalog --data-dir "$SMOKE_DATA_DIR"
uv run stocker data audit \
  --symbol AAPL.US \
  --timeframe 1d \
  --source eodhd \
  --data-dir "$SMOKE_DATA_DIR" \
  --market-calendar XNYS
uv run stocker data qa-eodhd \
  --config "$CONFIG_PATH" \
  --symbol AAPL.US \
  --timeframe 1d \
  --source eodhd \
  --data-dir "$SMOKE_DATA_DIR" \
  --require-raw \
  --market-calendar XNYS
uv run stocker research baseline \
  --symbol AAPL.US \
  --timeframe 1d \
  --source eodhd \
  --data-dir "$SMOKE_DATA_DIR"

echo "EODHD local smoke completed."
