#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Stocker Linux server bootstrap"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Warning: this script is intended for Linux servers."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found."
  echo "Install uv manually, then rerun this script:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "Syncing core and server dependencies with uv..."
uv sync --no-dev --group server

echo
echo "Bootstrap complete."
echo "Next steps:"
echo "  uv run --no-dev --group server stocker server dry-run --config configs/server.example.yaml"
echo "  uv run --no-dev --group server python apps/server/scripts/run_executor.py --config configs/server.example.yaml"
