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

## Shared entry admission

The unchanged starting quantity is floor(equity × risk fraction / stop distance).
The admitted quantity is whole shares, at least one, and no larger than that risk
quantity. Shared admission may reduce it against `risk.max_gross_notional`.
The ledger records risk-derived quantity, admitted quantity and limiting reason;
risk budget and method/broker price geometry are preserved.

`max_gross_notional` is a positive amount in the verified account currency, not a
leverage multiple. It is an account/environment ceiling shared across active runs;
the stricter applicable active commitment limit wins. Set the same intended ceiling
on runs sharing an account. Existing YAML without this field remains readable,
but an entry receives `EXPOSURE_POLICY_REQUIRED` until configured. There is no
default cap. This is an execution admission change; historical trade counts and
performance have not been revalidated under this policy.

Admission obtains a fresh request-specific IBKR NetLiquidation/GrossPositionValue
snapshot and positions. Both monetary fields must identify the same concrete
currency, and that currency must equal the qualified stock currency. BASE-only,
mixed, missing, nonfinite and IBKR unset values cannot establish usable capacity.
No FX rate is invented. BuyingPower is reported but is not treated as cash or
universally usable margin. The final quantity requires a bounded IBKR what-if
credit preview with valid initial-margin-after and equity-with-loan-after values;
a warning or unavailable preview rejects the entry. No preview becomes a real order.

An immediate SQLite transaction reserves account/environment exposure by conId,
across run identities. It combines actual broker positions with unresolved local
entry commitments. Stop/target/timeout children add no entry slot. A partial fill
and its entry remainder occupy one instrument slot; the unfilled remainder still
has notional exposure. Correlated broker orders are reconciled to the same plan
and are not added a second time. Unknown broker exposure blocks reconciliation.

The current IBKR preview does not attribute credit to individual working orders.
Therefore admission permits **one unresolved entry commitment per account/environment**:
another entry receives `PENDING_ENTRY_CAPACITY_UNVERIFIED` until the earlier parent
is fully filled or conclusively cancelled/rejected. Filled, reconciled positions may
coexist subject to configured limits. This explicit conservative policy avoids both
double-debiting broker-held credit and pretending concurrent previews reserve funds.
A ledger revision captured before account preparation is rechecked in the reservation
transaction; changed fills/reservations or a broker/local position mismatch require
reconciliation rather than using a stale funding snapshot.

Confirmed parent cancellation/rejection releases only its unfilled commitment;
filled exposure persists until broker-confirmed exits. Submission timeout/lost
acknowledgement retains the plan, signal idempotency and capacity across restart.
There is no arbitrary expiry or blind resubmission. Additive nullable ledger columns
preserve earlier rows and audit identities; unavailable historical sizing remains unknown.

Broker status and execution-detail callbacks may arrive in either order. The ledger
retains the highest reported cumulative parent fill separately from actual execution
records. A terminal parent releases its cancelled remainder, but reported shares
whose executions have not arrived still reserve exposure and block reconciliation.
The ledger never creates a position from a status report. Later lower/duplicate
reports cannot erase known fills. Older rows without this field require fresh
broker status when their terminal quantity is unresolved; conclusively rejected
local pre-submission plans remain settled.

Run configuration is captured before admission's first await and checked again after
quote preparation and credit preview. A newer risk setting or pause invalidates that
preparation. After the last credit await, final checks, marking SUBMITTING and the
IBKR adapter's synchronous bracket transmission contain no intervening event-loop
yield. Once transmission starts, its outcome must be reconciled; a later
pause does not revoke or delete it.

Supported operation is one integrated runtime/control process using one authoritative
SQLite state database for an account. The reservation transaction also serializes
independent connections to that database. Independent databases or multiple independently
configured trading services for the same broker account are outside this boundary.
Broker fills and market value can change independently of local transactions; a
notional admission ceiling is not a promise that future marked exposure never rises.
