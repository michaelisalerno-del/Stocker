set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

run_check() {
  case "$1" in
    format) uv run --no-sync ruff format --check . ;;
    lint) uv run --no-sync ruff check . ;;
    typing) uv run --no-sync mypy packages apps ;;
    python) uv run --no-sync pytest ;;
    frontend) npm test ;;
    server) uv run --no-sync python scripts/server_smoke.py ;;
    *) echo "Unknown check: $1"; return 2 ;;
  esac
}
if [[ $# -gt 0 ]]; then
  run_check "$1"
  exit $?
fi
failed=0
for check in format lint typing python frontend server; do
  echo "Running $check"
  if run_check "$check"; then
    echo "PASS $check"
  else
    echo "FAIL $check"
    failed=1
  fi
done
exit "$failed"
