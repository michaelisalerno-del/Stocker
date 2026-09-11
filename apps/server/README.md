# Stocker server

Use a separate release directory with Python 3.12 and uv:

```bash
uv sync --locked --no-default-groups --group server
.venv/bin/stocker stage10-dashboard --runs-config configs/runs.example.yaml --ibkr-config configs/ibkr.example.yaml --database .stocker/standalone.sqlite3
```

The standalone command does not connect IBKR. Integrated execution uses
`stage10-run` with operator-owned configuration and database paths. Launch the
prepared `.venv/bin/stocker` directly; do not let service startup sync default
development/research groups. Broker login remains in IB Gateway/TWS.

Read the [root README](../../README.md), [security guide](../../docs/STAGE10_DASHBOARD.md)
and [release/recovery runbook](../../docs/UNATTENDED_RECOVERY.md).
