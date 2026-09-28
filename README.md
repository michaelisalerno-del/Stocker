# SLRNO

Saxo-only futures data and disarmed options paper trading for **CL, GC, NG, NQ, SI**.
The existing Python runtime and authenticated dashboard are retained. No LIVE execution mode exists.

```sh
uv sync --locked --no-default-groups --group server
uv run --no-sync stocker futures-run --config configs/futures.paper.yaml --database .stocker/saxo-sim-disabled.sqlite3
```

The example starts without broker credentials, with execution disabled. Complete server-side OAuth,
verify each actual futures UIC, and obtain recording permission before enabling event archives.
SIM and LIVE data use distinct credentials, identities, token stores, caches and ledgers.
LIVE data can feed INTERNAL_PAPER; the LIVE client blocks broker mutations.

The server keeps a bounded 15-minute L1/L2 window independently of browsers. Frozen clock events
preserve available prehistory and at least 60 minutes afterward, including skipped trades. There
is no permanent full-session raw archive. Missing depth remains L1_ONLY/L2_UNAVAILABLE.

Paper admission means one whole long option, at most £10 including costs, and four concurrent
reservations/open trades. Ambiguous orders retain capacity. The frozen research does not approve
listed option products, actual 0DTE expiry clocks or delta tolerances; missing evidence blocks entries.
GC remains monitored. No orders were sent or paper execution armed during this migration.

[Setup and OAuth](docs/SAXO-SETUP.md) · [Cutover and rollback](docs/DEPLOYMENT.md) ·
[Rules and provenance](docs/FUTURES-RULEBOOK.md) · [Storage and stream policy](docs/MARKET-DATA.md) ·
[Delivery status and checks](docs/IMPLEMENTATION-REPORT.md) · [Server evidence status](docs/CURRENT-DEPLOYMENT.md)

IBKR's recoverable adapter/configuration is in `research/operational-history/parked-ibkr-runtime`.
Original ledgers, research, dated deployment evidence and historical provider labels remain intact.
EODHD external transport is disabled; retained offline research parsers/caches still work.
Server jobs must be stopped and masked through the controlled cutover; this checkout is not evidence
that the previously deployed server has changed.

Run `bash scripts/check.sh` for format, lint, types, Python tests, browser checks and server-only smoke.
