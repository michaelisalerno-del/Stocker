# FIRST4 reliability addendum — September 24, 2026

Inspected clean checkout: `0c55648d70d97804a741f0aa006632131a2e7b84`.
GitHub main was rechecked before editing and remained
`505d2429cf109a733f63d296a9943b974b222dcc`. PR #10 reviewed `0764f95680`;
its findings were checked against the newer checkout, not applied blindly.
This patch is local and uncommitted. No push, merge or deployment was performed.

| Finding | Result | Implementation |
| --- | --- | --- |
| Unowned open orders | Fixed | `first4_broker.py`: `_reconcile`, `owns_order`, `cancel_due_entries`. Foreign PAPER orders block entries but are untouched; verified owned exits remain permitted. |
| Identity mismatch | Fixed | Order status, cancellation and leg executions require account/client/API order ID and agreement with an existing permanent ID. Completed orders lacking a previously verified permanent ID remain ambiguous. |
| Historical allocation hot loop | Already fixed | `Store.unresolved()` and the existing partial index exclude only verified resolved allocations. Late fills/status changes reopen obligations; no history was deleted. |
| Calendar cache | Already fixed | Requested coverage is cached; holiday, Saturday, early-close and year-boundary regressions pass. |
| Obsolete research fetch | Fixed/retired | Deleted only the broken fetch implementation, fetch-only helpers/imports and parser branch. Offline `compare`, calculations and research artifacts remain. |
| IBKR 10197 | Fixed | Durable active data block, timestamp/request/contract context, separate diagnostic, data generation checks and positive recovery through the existing `option_access` verification. |
| Dashboard duplicate work | Fixed | Single in-flight refresh, coalesced refresh requests, bounded view-specific overview queries and a single request for an older history page. Existing endpoints/default overview remain available. |
| Unchanged position callbacks | Fixed | Conditional SQLite upsert avoids writing an identical snapshot. |
| Startup/legacy runtime | No removal warranted | Startup imports no research/backtest/legacy execution modules or sklearn. Retained research commands are lazy imports, not active trading tasks. |

10197 means market data is unavailable during a competing session. IBKR documents
live-account priority when live and PAPER sessions request real-time data together:
[official error codes](https://www.interactivebrokers.com/docs/tws-api/doc/error-handling/error-codes).

The active blocker survives process restart and is independent of socket connectivity,
reconciliation and deadline blockers. Repeated errors invalidate older preparation,
quote, metadata and scanner results. The existing complete option-access check must
verify real-time stock data, qualified option legs, fresh two-sided real-time option
quotes and combo metadata before clearing it. Silence, connection restoration and
reconciliation cannot clear it. The last 10197 diagnostic remains after recovery.
Recovery neither clears scanner continuity nor restores revoked opening authority.
No recovery worker, alternative quote mode, client-ID workaround or new arming path
was added. Owned exits retain their existing quote and exposure checks; inability to
obtain valid exit quotes remains an explicit unresolved obligation.

## Validation

- Initial targeted regression run: five failures (foreign open order ignored,
  three durable-identity cases, and 10197 not blocking entries).
- New isolated addendum: 27 passing cases. The decoder-shaped broker fakes now carry
  real account/client/order/permanent identifiers; no safety assertion was removed.
- Full Python suite: **514 passed**, five existing warnings, **62.81 seconds**.
- Focused FIRST4/scanner/reliability/capacity/addendum suite: **125 passed**.
- Ruff format: 186 files; Ruff lint passed; mypy: 107 source files passed.
- Mocked Playwright dashboard tests passed, including pagination, mobile menu,
  escaping, active data-block visibility and overlapping refresh prevention.
- Locked server-only installation/smoke passed with `UV_OFFLINE=true`; the smoke
  forbids socket connections. No broker was connected by these checks.
- Frozen calculation/artifact reproducibility tests ran in the full suite. Configs,
  dependency locks, frozen FIRST4 calculations and fixtures are unchanged.
- Offline REALIZED_M_20 tests/help passed. Full historical report reproduction was
  not run: raw cached `*_ibkr_1m.csv` inputs are not present in this checkout (only
  retrieval metrics are retained there). No replacement data was downloaded.
- GitHub CI was not triggered because pushing is prohibited for this pass.

## Local measurements

10,000 verified closed two-leg allocations (20,000 historical orders / 40,000 fills),
plus 0, 1 or 4 unresolved allocations. SQLite WAL and `synchronous=FULL` unchanged.
Repeated manager cycle includes deadline processing, close dispatch and owned quantities.

| Active | SQL statements | SQLite VM steps | Writes | Local cycle |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4 | 56 | 0 | 0.494 ms |
| 1 | 5 | 203 | 0 | 0.514 ms |
| 4 | 8 | 638 | 0 | 0.655 ms |

These structural results are unchanged from the inspected checkout's existing fix.
The earlier full-history baseline and its different measurement scope remain in
`FIRST4-reliability-audit-20260924.md`; no new speedup is claimed for the manager.

For an older-orders page, using the same 10,000 + 4 fixture and five local in-process
ASGI samples against the inspected dashboard source versus this patch:

| | Requests | SQL statements | Response bytes | Writes | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 2 | 12 | 344,178 | 0 | 2.280 ms |
| After | 1 | 5 | 146,642 | 0 | 0.994 ms |

The mocked browser burst produced three simultaneous requests before the change and
one after. An identical position callback produced one redundant row write before
and zero after (verified by SQLite total_changes). No remote/server/network latency
estimate is implied. Timing assertions are not used in CI.

Reproduce the dashboard comparison locally with the repository package source paths
on PYTHONPATH: `python tests/test_first4_capacity.py --dashboard`. The baseline commit
must be present in Git. Structural hot-loop checks run under normal pytest.

The unchanged 25-symbol/50-history-request virtual-time test completes at 13 seconds
with 1-second responses, concurrency four and 12 seconds maximum queueing. At 4-second
responses it fails closed at the fixed 45-second observation deadline: 48 requests
started, 44 finished, four cancelled and two never sent; no slots or tasks leak.
No timeout, concurrency, lateness or selection rule was relaxed.

Startup import inspection (sockets forbidden) took 0.313 seconds locally and loaded
zero research, backtest or deleted legacy execution modules; sklearn was absent.
This is an observation, not a claimed startup speedup. The task graph is unchanged:
server plus Runtime.run; scanner loop plus broker manager and dated opening check;
only existing per-allocation execution tasks and bounded history requests.

## Operator requirements and remaining limits

Resolve the competing IBKR login/market-data entitlement session before requesting
an authorized real-time verification. API clients 81 and 181 on one Gateway are not
separate IBKR logins; do not change their IDs or stop one merely because both exist.
Do not clear the active flag manually, switch to delayed/frozen data or restart
Gateway repeatedly. This patch does not schedule extra diagnostic data requests.

If the dated opening check already failed/completed or scanner continuity was lost,
restored data does not permit resuming entries. The existing explicit date-specific
permission and restart rules still apply; no extra date was authorized here.

Earlier exit-policy limits remain: no invented retry after a terminal failed combo
exit and no removal of fresh two-sided quote protection. Automatic retry bounds,
pricing protection and any additional residual-leg liquidation convention still need
an explicit approved policy. Ambiguous acknowledgements never trigger another sell.

Actual competing-session recovery, all required contract quotes, broker fills and
session-close flattening require separate authorized PAPER-session evidence. Offline
passes do not establish that end-to-end evidence.

## Files changed

- `packages/stocker_execution/src/stocker_execution/first4_broker.py`
- `packages/stocker_execution/src/stocker_execution/first4_runtime.py`
- `packages/stocker_execution/src/stocker_execution/first4_readiness.py`
- `packages/stocker_dashboard/src/stocker_dashboard/app.py`
- `packages/stocker_dashboard/src/stocker_dashboard/static/dashboard.js`
- `research/realized_m_20_ibkr_fast_v0/run_experiment.py`
- `docs/REALIZED_M_20_IBKR_FAST_TEST_V0.md`
- `docs/FIRST4-reliability-addendum-20260924.md`
- `tests/test_first4_addendum.py`
- `tests/test_first4.py`
- `tests/test_first4_reliability.py`
- `tests/test_first4_capacity.py`
- `tests/test_realized_m_20_ibkr_fast_v0.py`
- `tests/dashboard_first4.cjs`
