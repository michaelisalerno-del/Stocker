# FIRST4 reliability audit — 2026-09-24

Baseline checkout: `0764f95680951c069c02f3c752466105a19f3d70`, initially clean, at
`2026-09-23-replace-stocker-s-old-trading-method`. The task's supplied directory was
empty. A read-only GitHub comparison showed HEAD `505d2429cf109a733f63d296a9943b974b222dcc`
adds a merge commit with no file differences. No reset, rebase, push or merge was used.

## Findings and disposition

| Finding | Disposition | Implementation |
|---|---|---|
| Scanner API error can return empty/partial success; subscription leak | Confirmed and fixed | `first4_runtime.py:scanner_rows`, `scan`: request completion/error events and unconditional cleanup; UNKNOWN blocks continuity before ledger admission |
| Upstream 1100 leaves local socket and stale readiness | Confirmed and fixed | `first4_broker.py:error`, `invalidate_connection`, `_reconcile`: generation checks, readiness revocation, durable gap; restoration reconciles, never re-arms |
| Zero recorded exposure can hide an ambiguous exit reservation | Confirmed and fixed | `close_one`: terminal-order checks, attributable fills, fresh zero-position reconciliation and working-order recheck |
| Working exit labelled operator failure; terminal residual exposure indistinct | Confirmed and fixed | Explicit working, ambiguous acknowledgement, terminal residual and overdue outcomes; no duplicate submission |
| Terminal failed exits require an operator | Policy decision required, intentional current convention | No retry policy is specified; no retry was invented |
| Zero/stale bid on one leg blocks complete-package market exit | Policy decision required | Current approved convention explicitly requires fresh two-sided executable leg quotes; preserved |
| Invalid expiry candidate fails whole mapping | Confirmed and fixed | `contract_details`, `contracts`: completed empty or explicit nonexistent contract can be excluded; ambiguous metadata/errors/timeouts block mapping |
| Historical-ledger scans and repeated outcome writes | Confirmed and fixed | Indexed unresolved entries, identifier lookups and allocation-scoped execution sums; changed-only writes |
| Unbounded dashboard history | Confirmed and fixed | Bounded newest-first pages, limit/offset endpoints and UI navigation; labelled session P&L |
| 25-symbol history burst can exceed observation deadline | Confirmed capacity limit; cancellation fixed, rules preserved | Four concurrent requests, 15-second per-request timeout, 45-second whole observation; native-rank batch admission unchanged |
| Non-session calendar refresh loop | Confirmed and fixed | Requested end-date coverage, using the existing NYSE calendar |
| Restart/date-specific arming allegedly repeats automatically | Already guarded | Dated authority remains process-local; persisted check results do not authorise restart or later dates |
| EXIT_SUBMITTED itself treated as CLOSED | Not reproduced in original broker close path | Original code already required zero leg quantities; the separate ambiguous-reservation case above was reproduced and repaired |

Installed and locked `ib_async` is 2.1.0. Its default `RaiseRequestErrors=False`,
`Wrapper.error`, scanner completion and cleanup were inspected and exercised with the
real wrapper and fake client methods. The global request-error setting is unchanged.
IB's [official system codes](https://www.interactivebrokers.com/docs/tws-api/doc/error-handling/system-message-codes)
define 1100 as upstream loss, 1101 as restoration with market-data requests lost, and
1102 as restoration with data maintained. Farm informational messages are not treated
as full outages. All application market-data subscriptions are request-scoped.

## Measured capacity

Fixture: 10,000 verified closed two-leg allocations, 20,000 orders, 40,000 fills,
plus 0/1/4 active allocations. Local temporary SQLite databases; fake broker only.

| Active | Original dispatch + ownership/deadline reads | Original one settled close | Repaired complete repeated management pass |
|---:|---|---|---|
| 0 | 80,000 rows, 10,000 dispatched, 221.159 ms | 150,000 rows, 1 unnecessary write, 273.832 ms | 4 queries, 56 SQLite instructions, 0 writes, 0.457 ms |
| 1 | 80,004 rows, 10,001 dispatched, 210.641 ms | 150,008 rows, 1 unnecessary write, 278.657 ms | 5 queries, 203 SQLite instructions, 0 writes, 0.529 ms |
| 4 | 80,016 rows, 10,004 dispatched, 210.176 ms | 150,032 rows, 1 unnecessary write, 275.167 ms | 8 queries, 638 SQLite instructions, 0 writes, 0.681 ms |

These are different measurement scopes, not a claimed whole-loop speedup ratio.
Original dispatch replaces `close_one` with a counter; the separate original-close
column invokes one real `close_one`. Both original classes are loaded read-only from
the recorded commit. Repaired timing covers cancellation management, close management
and ownership aggregation. Active positions are held before their exit deadline.
No full original 10,000-close loop timing, broker latency estimate, initial migration
or restart reconciliation benchmark is claimed. Initial legacy records remain unresolved
until verified; restart reconciliation still inspects history. CI asserts SQL work,
query counts and zero redundant writes, not timing thresholds.

Virtual-time burst: 25 new symbols, two distinct histories each. At 1 second/request,
all 50 finish in 13 seconds, peak concurrency 4, final queue wait 12 seconds. At
4 seconds/request, the fixed 45-second observation deadline interrupts: 48 started,
44 completed, four cancelled, two never sent, final queue wait 44 seconds. No slots or
candidates commit, and no request futures/containers remain. A separate test verifies
the 15-second individual timeout. There is no validated reusable history among these
50 distinct requests; concurrency, lateness allowance and timeouts were not increased.

Reproduce with `rtk proxy .venv/bin/pytest tests/test_first4_capacity.py -s`.
Original-only measurements: `rtk proxy .venv/bin/python tests/test_first4_capacity.py`
(requires the recorded baseline in local Git history; not part of shallow-clone CI).

## Verification and remaining operating requirements

Executed checks: full Python suite **467 passed** (60.56 seconds, five existing
deprecation/empty-slice warnings); final focused FIRST4 suite **81 passed** (4.93
seconds), including two additional ordering/restoration tests added after the full
run. Repository formatting passed for 185 files, lint passed, and strict typing
passed for 107 source files. The locked server-only offline installation/startup
smoke passed. Browser dashboard smoke, including pagination, mobile navigation and
escaping, passed using bundled Node/Playwright: `npm` itself was unavailable on
PATH, so the exact `npm test` script was invoked directly. No test was disabled.
`git diff --check` and the unchanged frozen/configuration/artifact/lockfile diff
check passed. Broker-session tests were not run because they are outside scope.

The regression suite covers successful empty/nonempty scanners, API error plus empty
and partial results, timeout/cancellation/cleanup and durable continuity loss; upstream
loss/restoration, stale reconciliation, harmless notifications; rejected/cancelled,
working and ambiguous exits, restart with a partial fill, duplicate/out-of-order
callbacks, late executions, quote rejection and a conflicting order during quote
preparation; contract invalidity/ambiguity/cancellation, calendar holidays/weekends,
early closes/year boundaries, and history pagination.

The frozen 1,293 appearances and 80 allocations and SHA-256 fixture checks pass.
This checkout's FIRST4 path is the pure PRIOR15/Q5 rule, not a learned classifier/scaler;
there is no separate classifier/scaler reproducibility suite on this path. Strategy
code, configuration, dependency locks, research artifacts and fixtures are unchanged.

Operator decisions remain precise: whether to permit market exits with missing, stale
or zero leg quotes and under what freshness/liquidity protection; and whether to permit
a further order after a definitively terminal exit, with explicit retry bound, timing,
price protection and treatment of unmatched residual legs. Both changes remain disabled.
No post-close submission or new trading date is authorised by this patch.

Offline tests do not establish actual broker quotes, fills, restored subscriptions or
session-close flattening. Separate authorised PAPER-session evidence remains required.
No broker connection, market-data request, actual PAPER/LIVE order, deployment, restart,
server setting change or arming change occurred during this task.
