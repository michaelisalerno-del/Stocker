# Stocker Server

The server workspace is for boring execution processes: dry runs, paper trading,
future broker adapters, state reconciliation, risk checks, and monitoring hooks.

This bootstrap does not include Docker, systemd, broker credentials, or live trading.

Run later on Linux:

```bash
uv sync --no-dev --group server
uv run --no-dev --group server python apps/server/scripts/run_executor.py --config configs/server.example.yaml
```
