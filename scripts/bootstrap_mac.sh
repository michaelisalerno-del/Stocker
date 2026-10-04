#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Stocker macOS bootstrap"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew was not found."
  echo "Install it manually if you want Homebrew-managed tools: https://brew.sh/"
else
  echo "Homebrew found: $(brew --version | head -n 1)"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found."
  echo "Install uv manually, then rerun this script:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found on PATH."
  echo "Install Python 3.12 manually or let uv manage it with:"
  echo "  uv python install 3.12"
  exit 1
fi

echo "Syncing runtime and dev dependencies with uv..."
uv sync --locked

echo
echo "Bootstrap complete."
echo "Next steps:"
echo "  bash scripts/check.sh"
echo "  uv run --no-sync stocker futures-run --config configs/futures.paper.yaml --database .stocker/saxo-sim-disabled.sqlite3"
