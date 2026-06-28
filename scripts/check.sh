#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for full checks."
  echo "Install uv, then run: uv sync --all-groups && bash scripts/check.sh"
  exit 1
fi

uv run ruff format --check .
uv run ruff check .
uv run mypy packages apps
uv run pytest
