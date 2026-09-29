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

Paper admission means one whole long option, at most £50 including entry and reserved exit costs,
and four concurrent reservations/open trades against a £200 concurrent allocation ceiling.
Historical £10 reservations retain their original policy amounts. Ambiguous orders retain capacity. The frozen research does not approve
listed option products, actual 0DTE expiry clocks or delta tolerances; missing evidence blocks entries.
GC remains monitored. No orders were sent or paper execution armed during this migration.

[Focused cleanup, verification and migration](docs/SLRNO-CLEANUP.md) ·
[Setup and OAuth](docs/SAXO-SETUP.md) · [Cutover and rollback](docs/DEPLOYMENT.md) ·
[Rules and provenance](docs/FUTURES-RULEBOOK.md) · [Storage and stream policy](docs/MARKET-DATA.md) ·
[Option context and validation](docs/saxo-option-context-addendum.md) ·
[Current server evidence](docs/CURRENT-DEPLOYMENT.md) · [Initial migration report](docs/IMPLEMENTATION-REPORT.md)

IBKR's recoverable adapter/configuration is in `research/operational-history/parked-ibkr-runtime`.
Original ledgers, research, dated deployment evidence and historical provider labels remain intact.
EODHD external transport is disabled; retained offline research parsers/caches still work.
The option-context release `5aa3a7b` is deployed. A read-only check on **2026-09-29 at 07:51 UTC**
confirmed authenticated Saxo SIM, a connected stream, disabled/disarmed execution and no active
reservations. Actual futures selection, market-data entitlements and recording rights remain
unverified; no option field availability is claimed. See the [dated runtime snapshot](docs/github-sync-runtime-20260929.json).
The 13 legacy provider/scanner units were verified parked at deployment; historical evidence is retained.

Run `bash scripts/check.sh` for format, lint, types, Python tests, browser checks and server-only smoke.
