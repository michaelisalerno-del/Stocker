# SLRNO operational reliability repair — local only

Baseline: `12e3993642ae7d75095bcd9211257084aea1cf7e`. The new task directory was
empty, so the committed September 24 dashboard checkout was cloned locally into
it. No remote fetch, broker connection, production configuration or arming change,
order transmission, service restart, or deployment was performed. Changes remain
local and uncommitted.

## Reproduced weaknesses and repairs

1. **Snapshot completion can outlive freshness.** The controlled-clock regression
   delivers a valid stock price immediately, advances eleven seconds, and completes
   the snapshot. The old sequence then sees a price older than the configured
   five seconds. This establishes a possible mechanism, not the cause of the two
   historical Ford failures: their underlying quote evidence was not retained.

   `option_access()` now resolves the chain before acquiring the diagnostic
   reference, then consumes a fresh update from one temporary stream. It uses the
   same diagnostic stock, `marketPrice()` last/midpoint choice, real-time type 1,
   positive finite price requirement, configured age threshold and contract rules.
   Relevant price-tick timestamps are checked synchronously where the reference
   is consumed for strike selection. General ticker updates and size ticks cannot
   renew a price. No previous close, historical, delayed or frozen substitute is
   allowed. Request-local ticker state excludes cached and unrelated updates.

2. **Attempt three was terminal despite remaining time.** Opening verification
   now retries only the explicit transient quote predicates and timeouts from
   permitted option-access stages. Every attempt has at most 60 seconds, including
   reconciliation, clipped to the original session-open-plus-14-minute cutoff.
   Failed attempts retain the five-second pause. Cleanup finishes before the next
   attempt; the deadline is never reset. There is no special long final attempt.

   **This intentionally changes opening-verification retry policy.** It is not
   merely a diagnostic change, and retry behaviour is not unchanged. A fifth
   attempt can succeed; persistent failure leaves entries unarmed. Insufficient
   time for the full pause means no further attempt starts. Reconciliation
   timeouts, ownership/account problems, missing permission, operator pause,
   continuity interruption and fatal persistence errors remain terminal.
   Guards run before attempts, while waiting/backing off and before readiness.
   Cancellation drains the in-flight task. A fatal exception raised during
   timeout cleanup takes precedence over the timeout.

3. **Option/BAG stream errors were lost to their awaiting tasks.** The pinned
   `ib_async` 2.1.0 wrapper reuses tickers by contract hash; ordinary streaming
   requests do not register a request future. `First4IB.market_data()` now uses
   that wrapper's existing request/error lifecycle with one request-local ticker,
   a registered future, and unconditional callback/request-map cleanup. It cancels
   the exact request ID. The diagnostic stock, option quote and BAG increment
   waits share this helper. A broker subscription rejection now reaches the
   caller as its original `RequestError`, rather than a generic unavailable
   quote/increment at the deadline. It does not replace or alter the strategy's
   tick-by-tick anchor subscription.

4. **An anchor subscription rejection could wait behind pending chain work.**
   Execution now waits for either chain completion or the anchor future. An
   anchor error interrupts pending metadata and drains it. A valid early anchor
   remains retained while chain work completes. Existing wire-trade timestamp,
   first-valid-anchor and chain-cache repairs were verified and preserved.

5. **Early option quotes could expire behind slow BAG metadata.** The entry path
   already requested these concurrently, but closed the quote streams as soon as
   prices initially passed. A six-second metadata delay reproduced a subsequent
   `QUOTES_EXPIRED_DURING_PREPARATION` despite available fresh updates. Those same
   two streams now remain open until BAG metadata is ready; validation uses their
   latest price observations. The regression reaches submission with exactly
   three streams (BAG plus two legs), without sequential reacquisition or extra
   requests. Final preparation still rechecks quote age and the entry deadline.
   Bid and ask ages are each checked, so a future timestamp cannot be hidden by
   taking only the maximum age.

## Diagnostics and display

- Existing `opening_check:<date>` metadata retains each attempt's number, stage,
  start/end time, attempt deadline, opening-deadline margins, outcome and bounded,
  whitespace-normalized exception information with account IDs redacted. The
  diagnostic stock record contains actual consumed reference, returned data type,
  ticker/price timestamps, individual ages and failed predicate. A stage that did
  not reach the stock check records no stock evidence. Earlier failed attempts
  remain after success; their error is removed from the current blocker field.
- Execution errors retain their actual contract/quote/BAG/submission stage,
  anchor receipt evidence, durable order reservation, observed order status and
  effective fill evidence. RESERVED is explicitly an unresolved submission, not
  proof that the broker accepted an order. No diagnostic broker requests are added.
- Current process readiness is separate from the historical opening report.
  Active verification displays attempt count and remaining time without a
  three-attempt denominator. ARMED/CHECKING history cannot replay after restart or
  become a new process's authorization token.
- Early missing PRIOR15 displays “PRIOR15 unavailable at first appearance — fewer
  than 15 session minutes”. Rejected decisions display “Rejected for this session
  — not reconsidered”. Later missing values remain generic, and the original
  higher-priority decision stays primary. Rejected snapshots have no activation
  progress bar. Failed/unfilled allocations without fills display “No position
  opened”; actual positions still derive from effective fills.
- Refresh continues updating existing DOM nodes. Polling periods remain 5/15/5/30
  seconds for overview/opportunities/execution/system. No additional dashboard was
  introduced.

## Offline verification

New reproductions were run red before their associated repairs: the fixed
three-attempt stop, quote/BAG error propagation, pending-chain anchor errors, and
slow-BAG expiry of early option quotes. The snapshot-age test reproduces the old
request sequence directly using the pinned wrapper and a controlled clock.

Final relevant Python suite: **197 passed**, one existing Starlette/httpx
deprecation warning, 6.37 seconds. Executed via the baseline checkout's existing
Python 3.12 virtualenv, with pytest resolving package sources from this checkout:

```text
rtk proxy <baseline>/.venv/bin/python -m pytest \
  tests/test_first4.py tests/test_first4_operational.py \
  tests/test_first4_opening_retry.py tests/test_first4_anchor.py \
  tests/test_first4_chain_diagnostics.py tests/test_first4_repairs.py \
  tests/test_first4_deployment.py tests/test_first4_option_volume_filter.py \
  tests/test_dashboard_views.py tests/test_dashboard_security.py \
  -p no:cacheprovider --tb=short
```

This includes the frozen **1,293 appearances / 80 allocations** equivalence test,
PRIOR15/completed-bar boundaries, permanent first-appearance/slot decisions,
persistence, reconciliation, order identity, execution and safety tests. New
operational tests block socket connections and use controlled time/event-loop
yields rather than real sleeps. They cover cached-only, unrelated, missing,
invalid, stale, future, delayed/frozen data; reference revalidation after wakeup;
more than three failures; clipped budgets; fatal cleanup; pause, cancellation and
continuity interruption; all temporary-stream cancellation paths; and restart
non-replay. The complete valid mocked candidate reaches the PAPER submission
boundary; invalid candidates do not. An ambiguous mock write is reserved once,
and a repeated invocation cannot transmit another order.

The existing Playwright browser regression passed using intercepted local mock
responses. Horizontal scroll remained **260 → 260**, expanded evidence and
selection survived refresh, maximum concurrent refreshes remained one, unchanged
content caused zero text mutations, and hidden pages issued zero requests over
60 simulated seconds. Added assertions cover historical versus active opening
status and removal of rejected-candidate progress meters. Screenshots are in a
temporary directory, not substituted into deployment records.

Ruff lint/format, mypy on the six changed Python source files and
`git diff --check` were also run. Strategy calculation code, frozen fixtures,
configuration and lockfiles are unchanged.

## Changed files

- `packages/stocker_execution/src/stocker_execution/first4_requests.py`
- `packages/stocker_execution/src/stocker_execution/first4_readiness.py`
- `packages/stocker_execution/src/stocker_execution/first4_broker.py`
- `packages/stocker_execution/src/stocker_execution/first4_runtime.py`
- `packages/stocker_execution/src/stocker_execution/first4_store.py`
- `packages/stocker_dashboard/src/stocker_dashboard/views.py`
- `packages/stocker_dashboard/src/stocker_dashboard/static/dashboard.js`
- `tests/test_first4_operational.py`
- `tests/test_first4_opening_retry.py`
- `tests/test_first4_anchor.py`
- `tests/test_first4.py`
- `tests/test_dashboard_views.py`
- `tests/dashboard_first4.cjs`
- This report.

## Limits and external blockers

The original anchor definition, entry deadline, expiry/strike/trading-class and
multiplier conventions, one-package sizing, premium cap, order safeguards,
PRIOR15 window, scanner filters, rejection precedence and four consumed slots are
unchanged. No early rejected stock is recovered or reconsidered.

These tests do not establish the cause of the September 24 historical stock or
candidate failures. Missing entitlements, absent eligible trades, unavailable or
ambiguous permitted contracts, and genuine broker errors remain explicit blockers.
The historical chain responses and failed stock snapshots are unavailable here.
Actual broker data delivery, request cancellation acknowledgements, opening-time
performance and fills remain unverified in production. Mocked submission proves
the software boundary and duplicate protection, not real broker acceptance or
fills. Nothing has been deployed.
