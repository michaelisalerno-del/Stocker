# Execution Safety

The execution server must be boring. It should do fewer things than the research
machine, with fewer dependencies and more hard stops.

## Environment Gate

Stage 7 broker transmission is enabled only for an explicitly writable PAPER or LIVE connection
whose environment and verified connected account exactly match the run and configured expected
account. The per-run execution router has no cross-environment fallback. A missing session returns
`EXECUTION_ENVIRONMENT_UNAVAILABLE`; a mismatch returns `ACCOUNT_OR_ENVIRONMENT_MISMATCH`.

## Risk Checks

No Stage 7 order reaches IBKR without an explicit per-run `risk_per_trade`, authoritative selected
IBKR account equity, valid Stage 6 protection geometry, conservative whole-share sizing, no existing
same-instrument position, and optional actual-position capacity. The earlier generic placeholder
risk limits remain separate from this Stage 7 runtime contract.

## Stale Data

No trading should occur if market data is stale, timestamps are ambiguous, or a data
feed has gaps during an expected session. Stale data should fail closed.

The server should only consume datasets or signals that have passed the research-side
audit process. CSV import, DuckDB cataloging, audit reports, and baseline reports are
desktop responsibilities, not live execution responsibilities.

## State Reconciliation

Each environment/account executor compares normalized IBKR open/completed orders, executions, and
positions with its SQLite execution ledger at startup and after every reconnect. Unknown or
unresolved exposure blocks new orders for that environment with
`EXECUTION_RECONCILIATION_REQUIRED`; it is never automatically flattened. Open
orders can be recovered after a crash only when their deterministic IBKR `orderRef` matches a
reserved local plan. A filled position is not reconciled unless both protective children remain
open. Every candidate attempt, including pre-plan rejection, preserves its run, environment, and
expected/actual account identity in SQLite.

## Sessions

No trading occurs outside allowed sessions. The scheduler uses exchange calendars,
method-owned checkpoints, run session windows and broker availability.

## Broker Boundaries

Stage 7 reuses the concrete Stage 2 `IbkrConnection`; IBKR is the only broker. Strategy, risk, and
research code receive normalized models and never call `ib_async` or submit orders directly.
