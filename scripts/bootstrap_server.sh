#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "SLRNO Linux server bootstrap"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Warning: this script is intended for Linux servers."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found."
  echo "Install uv manually, then rerun this script:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "Syncing runtime dependencies with uv (no development or research groups)..."
uv sync --locked --no-default-groups

echo
echo "Bootstrap complete."
echo "Next steps:"
echo "  uv run --no-sync stocker paper-run --config configs/paper.example.yaml --database .stocker/paper-record.sqlite3"
