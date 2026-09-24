# SLRNO

SLRNO runs one US-only frozen FIRST4 method against IBKR PAPER account `DUP655399`.
LIVE execution is unavailable. Research and raw data remain separate from trading.

See [current deployment and effective configuration](docs/CURRENT-DEPLOYMENT.md).
Older cutover reports describe their recorded deployment, not current settings.

Run `uv sync --locked --no-default-groups --group server`, then:

```sh
stocker first4-run --config configs/first4.example.yaml --database .stocker/first4.sqlite3
```

The existing dashboard is served on loopback port 8765. Its authenticated reverse-proxy boundary is unchanged.
The example config contains the user's PAPER execution settings and remains unarmed until the non-transmitting broker checks pass, including fresh real-time option quotes.
See [FIRST4 protocol and execution](docs/FIRST4.md) for exact sources, differences and recovery.

Data and research CLI commands remain available via `stocker data --help` and `stocker research --help`.
Historical operational reports are retained under `research/operational-history`; they are not instructions for starting this application.

## Dashboard

Overview shows four permanent slots, frozen first-appearance proximity snapshots and
session economics. Opportunities provides bounded decision history. Execution groups
actual fills and remaining exposure by allocation. System contains opening verification,
read-only configuration and diagnostics. Unrealised P&L is unavailable without a reliable
valuation; quote comparisons are never realised broker results.

Research workstations keep the default dev/research groups; production must use
`--no-default-groups --group server`. The internal `stocker` CLI, environment variables,
database names and service paths are intentionally unchanged.

See [SLRNO implementation and verification](docs/SLRNO-implementation.md) for local
changes, synthetic screenshots and measured performance. See the current deployment record
for the active release and operational state.
