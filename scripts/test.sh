#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if command -v uv >/dev/null 2>&1; then
  uv run pytest
else
  echo "uv not found; running tests with the current Python environment."
  python3 -m pytest tests
fi
