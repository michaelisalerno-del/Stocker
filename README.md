# Stocker

Stocker runs one US-only frozen FIRST4 method against IBKR PAPER account `DUP655399`.
LIVE execution is unavailable. Research and raw data remain separate from trading.

The server runs release `0080a786d78660e04d9a17d310b8322c4eecb3c8`.
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
