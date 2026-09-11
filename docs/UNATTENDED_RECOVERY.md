# Unattended broker reconciliation

Stocker always verifies account identity, current open orders, executions, and
positions before marking an execution environment ready. Unknown exposure,
missing protective orders, or unresolved local plans continue to block new orders.

Completed-order history is required when an unreconciled connection has unfinished
local plans for the account, including plans belonging to another run. A settled
ledger does not need historical order statuses to reconcile current broker state.
Orders placed on an already reconciled connection use the broker's live order
callbacks. A subsequent disconnect requires historical recovery again if any plan
remains unfinished. Skipping the history request does not mark it as loaded.

## Incident: 2026-09-05

IBKR upstream connectivity was lost around 04:51 UTC and reported restored at
04:51:41. Starting at 04:52:41, all six PAPER runs repeatedly failed reconciliation
with `IBKR order-status request timed out`, cycling the socket about once a minute.
The previous scheduled Gateway restart had passed its port check at 23:46 UTC.

A separate read-only API probe confirmed that current positions, open orders, and
executions completed in milliseconds. `reqCompletedOrdersAsync(apiOnly=False)`
did not complete within eight seconds; the application had independently timed out
the same request at its configured sixty seconds. The precise reason the Gateway
stopped answering that history request was not established.

The sole recorded trade was CLOSED with equal entry and exit quantities, and
broker exposure was zero. The application defect was making completed history
an unconditional dependency of recovery even in that settled state.

`tests/test_unattended_reconciliation.py` reproduces the unavailable endpoint through
the actual IBKR adapter and execution service. It covers closed-ledger reconnects,
new live order callbacks, unfinished plans in sibling runs, unknown positions, and
disconnects with unfinished orders. Existing reconciliation and duplicate-order
checks remain in force.

For runtime health, inspect `/api/overview`: `system`, environment `ready` and
`reconciled`, and run reasons. A listening Gateway port alone is not evidence that
Stocker has completed reconciliation. Market-closed READY runs are expected.
