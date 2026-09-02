# Execution Safety

The execution server must be boring. It should do fewer things than the research
machine, with fewer dependencies and more hard stops.

## Environment Gate

Stage 7 broker transmission is enabled only for an explicitly writable PAPER connection whose
verified connected account matches the run's expected `D`-prefixed IBKR account. LIVE exists in the
run model but returns `LIVE_EXECUTION_DISABLED` before broker transmission.

## Risk Checks

No Stage 7 order reaches IBKR without an explicit per-run `risk_per_trade`, authoritative IBKR
account equity, valid Stage 6 protection geometry, conservative whole-share sizing, no existing
same-instrument position, and optional actual-position capacity. The earlier generic placeholder
risk limits remain separate from this Stage 7 runtime contract.

## Stale Data

No trading should occur if market data is stale, timestamps are ambiguous, or a data
feed has gaps during an expected session. Stale data should fail closed.

The server should only consume datasets or signals that have passed the research-side
audit process. CSV import, DuckDB cataloging, audit reports, and baseline reports are
desktop responsibilities, not live execution responsibilities.

## State Reconciliation

The executor compares normalized IBKR open/completed orders, executions, and positions with its
SQLite execution ledger at startup and after every reconnect. Unknown or unresolved exposure blocks
new orders with `EXECUTION_RECONCILIATION_REQUIRED`; it is never automatically flattened.

## Sessions

No trading should occur outside allowed sessions. Future session checks should use
exchange calendars, instrument-specific trading hours, and broker availability.

## Broker Boundaries

Stage 7 reuses the concrete Stage 2 `IbkrConnection`; IBKR is the only broker. Strategy, risk, and
research code receive normalized models and never call `ib_async` or submit orders directly.
