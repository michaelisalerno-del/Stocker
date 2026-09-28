#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-data_smoke_research}"
CONFIG_PATH="${STOCKER_RESEARCH_CONFIG:-configs/research.example.yaml}"
UNIVERSE_PATH="${RESEARCH_SMOKE_UNIVERSE:-universes/manual/us_test_5.yaml}"
HYPOTHESIS_PATH="${RESEARCH_SMOKE_HYPOTHESIS:-research/hypotheses/examples/moving_average_momentum.yaml}"
TIMEFRAME="${RESEARCH_SMOKE_TIMEFRAME:-1d}"
SOURCE="${RESEARCH_SMOKE_SOURCE:-eodhd}"
FROM_DATE="${RESEARCH_SMOKE_FROM:-2021-01-01}"
TO_DATE="${RESEARCH_SMOKE_TO:-2026-06-28}"
MAX_SYMBOLS="${RESEARCH_SMOKE_MAX_SYMBOLS:-5}"
MARKET_CALENDAR="${RESEARCH_SMOKE_MARKET_CALENDAR:-XNYS}"
QUALIFIED_UNIVERSE_PATH="${SMOKE_DATA_DIR}/universes/research_ready/us_test_5_${TIMEFRAME}.json"

if [[ -z "${EODHD_API_TOKEN:-}" && -f .env ]]; then
  EODHD_API_TOKEN="$(
    grep -E '^EODHD_API_TOKEN=' .env | tail -n 1 | cut -d '=' -f 2- || true
  )"
  export EODHD_API_TOKEN
fi

has_local_data() {
  [[ -d "${SMOKE_DATA_DIR}/processed" ]] \
    && find "${SMOKE_DATA_DIR}/processed" -path "*source=${SOURCE}*" -name data.parquet \
      -print -quit | grep -q .
}

if [[ -n "${EODHD_API_TOKEN:-}" ]]; then
  echo "EODHD_API_TOKEN is set; fetching bounded ${SOURCE} ${TIMEFRAME} data into ${SMOKE_DATA_DIR}."
  uv run stocker universe fetch \
    --config "$CONFIG_PATH" \
    --universe "$UNIVERSE_PATH" \
    --from "$FROM_DATE" \
    --to "$TO_DATE" \
    --timeframe "$TIMEFRAME" \
    --source "$SOURCE" \
    --data-dir "$SMOKE_DATA_DIR" \
    --merge \
    --audit \
    --market-calendar "$MARKET_CALENDAR" \
    --max-symbols "$MAX_SYMBOLS"
elif ! has_local_data; then
  cat <<EOF
EODHD_API_TOKEN is not set and no local ${SOURCE} data was found under ${SMOKE_DATA_DIR}.
Live fetch is skipped. To run the full real-data research smoke, either:
- export EODHD_API_TOKEN and rerun this script, or
- place existing ${SOURCE} ${TIMEFRAME} Parquet data under ${SMOKE_DATA_DIR}/processed.
EOF
  echo "qualified_universe_path=${QUALIFIED_UNIVERSE_PATH}"
  echo "universe_report_markdown_path="
  echo "universe_report_json_path="
  exit 0
else
  echo "EODHD_API_TOKEN is not set; live fetch skipped. Reusing local data in ${SMOKE_DATA_DIR}."
fi

mkdir -p "$(dirname "$QUALIFIED_UNIVERSE_PATH")"

uv run stocker universe qualify \
  --universe "$UNIVERSE_PATH" \
  --output "$QUALIFIED_UNIVERSE_PATH" \
  --timeframe "$TIMEFRAME" \
  --source "$SOURCE" \
  --data-dir "$SMOKE_DATA_DIR" \
  --min-history-days 626 \
  --min-last-close 1 \
  --min-median-dollar-volume-60d 0 \
  --market-calendar "$MARKET_CALENDAR"

QUALIFIED_COUNT="$(
  uv run python -c '
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(len(payload.get("qualified_symbols", [])))
' "$QUALIFIED_UNIVERSE_PATH"
)"

if [[ "$QUALIFIED_COUNT" == "0" ]]; then
  echo "No symbols qualified for the research smoke; this is a data-readiness result, not an edge result."
  echo "qualified_universe_path=${QUALIFIED_UNIVERSE_PATH}"
  echo "universe_report_markdown_path="
  echo "universe_report_json_path="
  exit 0
fi

uv run stocker research run-universe \
  --hypothesis "$HYPOTHESIS_PATH" \
  --qualified-universe "$QUALIFIED_UNIVERSE_PATH" \
  --config "$CONFIG_PATH" \
  --data-dir "$SMOKE_DATA_DIR" \
  --source "$SOURCE" \
  --timeframe "$TIMEFRAME" \
  --max-symbols "$MAX_SYMBOLS"

REPORT_PATHS="$(
  uv run python -c '
from pathlib import Path
import sys
root = Path(sys.argv[1]) / "reports" / "research" / "universe"
json_reports = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime)
md_reports = sorted(root.glob("*.md"), key=lambda path: path.stat().st_mtime)
print(md_reports[-1] if md_reports else "")
print(json_reports[-1] if json_reports else "")
' "$SMOKE_DATA_DIR"
)"
UNIVERSE_REPORT_MARKDOWN_PATH="$(printf "%s\n" "$REPORT_PATHS" | sed -n '1p')"
UNIVERSE_REPORT_JSON_PATH="$(printf "%s\n" "$REPORT_PATHS" | sed -n '2p')"

echo "qualified_universe_path=${QUALIFIED_UNIVERSE_PATH}"
echo "universe_report_markdown_path=${UNIVERSE_REPORT_MARKDOWN_PATH}"
echo "universe_report_json_path=${UNIVERSE_REPORT_JSON_PATH}"
echo "classification_counts are in the universe JSON report."
echo "Rejected classifications are expected and are not a smoke failure."
