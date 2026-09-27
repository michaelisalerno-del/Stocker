# SLRNO

A continuous six-market futures-options PAPER application: **BTC, CL, GC, NG, NQ, SI**.
One shared IBKR connection, a fresh durable ledger and a compact authenticated dashboard.
LIVE is unavailable. Historical synthetic results never enter trading P&L.

```sh
uv sync --locked --no-default-groups --group server
stocker futures-run --config configs/futures.paper.yaml --database .stocker/futures.sqlite3
```

The example is **unarmed**. Its listed-product mappings are deliberately empty: the frozen
research defined continuous strikes and a hypothetical expiry, without approving real products
or delta tolerance. All six markets can be monitored; none may submit an order until its
specific mapping is established. See [execution rulebook](docs/FUTURES-RULEBOOK.md).

The executable path submits actual IBKR PAPER limit orders to allowlisted account DUP655399.
Connection is verified by broker-returned account identity, never by a port or UI label.
One option contract, at most £10 including premium and conservative fee reserve, four concurrent
reserved/open trades and £40 concurrent allocation. Full £10 is reserved per admitted trade.
Uncertain submissions and exits retain reservations until reconciled closure.

Overview contains six permanent market cards. Trades & signals contains orders, executions
and skipped opportunities. System contains connectivity, readiness and configuration.
Quotes, observations and broker-simulated fills are separate. P&L with missing fees or FX is
provisional. Refreshes preserve cards, focus, filters, expansion and scroll.

[Deployment and non-transmitting preflight](docs/DEPLOYMENT.md) ·
[Implementation status](docs/CURRENT-DEPLOYMENT.md)
· [Implementation report and fixture screenshots](docs/IMPLEMENTATION-REPORT.md)

Existing authentication, loopback proxy boundary, locked server installation, research/data
CLI, backups and raw research records remain. The independent scanner research schedule is
not part of this application and is not changed. Retired operational evidence is retained in
research/operational-history, outside the active runtime and its economics.

Run `bash scripts/check.sh` for format, lint, typing, Python, browser and server-only checks.
