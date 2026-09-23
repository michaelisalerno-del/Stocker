# Stocker

Stocker runs one US-only frozen FIRST4 method against IBKR PAPER account `DUP655399`.
LIVE execution is unavailable. Research and raw data remain separate from trading.

Run `uv sync --locked --group server`, then:

```sh
stocker first4-run --config configs/first4.example.yaml --database .stocker/first4.sqlite3
```

The existing dashboard is served on loopback port 8765. Its authenticated reverse-proxy boundary is unchanged.
The example config is deliberately unarmed: fill every required listed-option execution setting before setting `armed: true`.
See [FIRST4 protocol and execution](docs/FIRST4.md) for exact sources, differences and recovery.

Data and research CLI commands remain available via `stocker data --help` and `stocker research --help`.
Historical operational reports are retained under `research/operational-history`; they are not instructions for starting this application.
