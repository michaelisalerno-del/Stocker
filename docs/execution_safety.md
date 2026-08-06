# Execution Safety

The execution server must be boring. It should do fewer things than the research
machine, with fewer dependencies and more hard stops.

## Kill Switches

Trading must be disabled by default. A future live executor needs explicit kill
switches at config, process, and broker-adapter levels. If any switch is off, orders are
blocked.

## Risk Checks

No order should reach a broker adapter without passing risk checks. Required checks
include max order size, max position size, max daily loss, max orders per day, and
trading-enabled state.

## Stale Data

No trading should occur if market data is stale, timestamps are ambiguous, or a data
feed has gaps during an expected session. Stale data should fail closed.

The server should only consume datasets or signals that have passed the research-side
audit process. CSV import, DuckDB cataloging, audit reports, and baseline reports are
desktop responsibilities, not live execution responsibilities.

## State Reconciliation

The server must compare broker positions and cash with internal state. If they disagree
outside an allowed tolerance, trading stops until the discrepancy is resolved.

## Sessions

No trading should occur outside allowed sessions. Future session checks should use
exchange calendars, instrument-specific trading hours, and broker availability.

## Broker Boundaries

No order-capable broker interface is currently implemented. A future dedicated execution
subsystem may accept only durable approved order intents under a separately accepted
plan. Research code, plugins, notebooks, and backtests must not call broker order APIs
directly.
