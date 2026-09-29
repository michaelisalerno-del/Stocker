# SLRNO focused cleanup — 29 September 2026

This change is local to `codex/slrno-cleanup`, based on `c947421800e6f643a10a6c074706754af646b9e1`.
It has **not been deployed**, production has not been restarted, and execution has not been enabled.
No authenticated account values or live orders were requested during this work.

## Inspection and changes

The inspected baseline still enforced £10/1,000p in configuration, contract costing, broker validation
and reservations. Its reservation schema fixed each allocation at 1,000p. `decisions()` consumed an
hourly clock before the final completed minute necessarily arrived. Every page fetched the complete
five-market overview; the recorder repeatedly decompressed derived flow history and serialized
unchanged state for size accounting. Retained overlapping capture membership also protected old
option subscriptions after their observation obligations ended. These were confirmed in current
code, with focused regressions, before changing them.

Previously completed Saxo routing, compression, price-to-contract-factor handling, option analytics,
OAuth, stream freshness and safety work were retained. The inspected deployment evidence was the
existing [29 September runtime snapshot](github-sync-runtime-20260929.json) and
[current deployment record](CURRENT-DEPLOYMENT.md), not a new production measurement.

- New opportunities have a £50 all-in ceiling: premium, entry costs and reserved exit costs under
  the existing policy. One contract stays one contract. Configuration is authoritative; pence and
  the £200 concurrent ceiling are derived. Contract costing, broker validation and transactional
  admission independently enforce the limits. Pending/reserved/open opportunities count once
  toward four slots. Historical £10 records retain their amounts and entry policy.
- A missing final completed minute keeps the clock pending only inside its original 20 seconds.
  A priority boundary history refresh uses the existing worker and cadence; earlier gaps remain
  final skips. The original data cutoff and exit anchor are unchanged. Awaited precheck cannot
  extend the admission deadline, bypass a pause, or duplicate a durable order intent.
- Overview has compact state-first market cards, four slots and an Account strip. Other pages
  request their own state: one selected market, filtered/sorted history, execution evidence, or
  system health. Diagnostic evidence is expanded on demand. Execution uses tables with raw data
  behind an expansion. Chart geometry is reused until bars change. Keyed rows preserve focus,
  selections, scroll and open panels.
- One request helper checks HTTP errors and bounds request duration. Refresh guards release after
  failure, timeout and cancellation; obsolete responses cannot replace newer selections. Controls
  show server-confirmed results. Last successful refresh and stale states remain visible.
- Derived book-flow history is bounded and counted inside the unchanged rolling quota. State sizes
  update at mutation points. Compressed raw evidence and checkpoint reconstruction remain intact.
  The existing writer wakes early when 64 items are queued, preserving its normal batching delay
  at lower traffic. This prevents faster ingestion outrunning the delayed writer during bursts.
- Each capture event records its own instrument obligations. Retained evidence stays in the shared
  archive while obsolete subscriptions can retire. Positions, reservations and unfinished/post-close
  observation obligations remain protected. No quotas or subscription budgets were increased.
- Closed-economics display summaries have a process-local five-second cache invalidated by local
  writes or SQLite external-connection data-version changes. Admission and reconciliation never
  read this cache. History sorts in SQL before pagination. Unused `tenacity` and a redundant `httpx`
  dependency declaration were removed; no framework, service or runtime dependency was added.

The frozen rules module, five markets, GC monitor-only policy, long-options-only policy, one-contract
restriction, live-order prohibition, account verification, freshness, ownership, reconciliation,
expiry protections and durable order intent remain in place. Default configuration remains
`DISABLED`, unarmed and persistent recording off.

## Saxo balance contract

Official Saxo documentation checked for this implementation:

- [Get account balances](https://www.developer.saxo/openapi/referencedocs/port/v1/balances/get__port)
- [Create a balance subscription](https://www.developer.saxo/openapi/referencedocs/port/v1/balances/post__port__subscriptions)
- [Balance request schema](https://www.developer.saxo/openapi/referencedocs/port/v1/balances/post__port__subscriptions/schema-balancerequest)
- [Account details](https://www.developer.saxo/openapi/learn/account-details)
- [Streaming snapshots, deltas and heartbeats](https://www.developer.saxo/openapi/learn/streaming)

The existing authenticated server connection creates one account-scoped balance subscription with
`AccountKey`, `FieldGroups: [CalculateCashForTrading]` and a requested 10-second refresh. It uses
`TotalValue`, `CashBalance`, `CashAvailableForTrading` and `Currency` directly from Saxo. The returned
inactivity timeout and stream contact determine freshness. Heartbeats extend stream health without
inventing a new successful balance-update time. Subscription recovery is bounded to one attempt
per minute; browsers only read the in-process snapshot and never initiate Saxo requests.

SIM is prominently labelled simulated funds, with “Real-money balances are not connected.” LIVE
data is labelled a real-money account while ordering stays disabled. Only the selected authenticated
account is displayed, with a masked identifier and native currency. Missing fields are Unavailable;
a disconnected/unhealthy subscription or unreliable calculation retains the previous valid snapshot
as Stale, with its last successful update. Account changes cannot relabel the prior account's values.
No currency conversion is performed. Strategy realised P&L and estimated open-trade P&L remain
separate; internal allocation is neither added to account value nor subtracted from broker cash.

## Checks actually executed here

- Final full Python suite: **501 passed**, 7 third-party/deprecation/numeric warnings, 68.83s.
  The pre-review full run also passed (499 tests, 80.55s); affected review regressions passed
  separately (63 tests, 8.53s).
- Focused new regression suite initially **15 passed**, then **17 passed** with review regressions, including boundary arrival at 10/20/21 seconds,
  duplicate suppression, £50/four-slot/£200 limits, historical £10 migration and broker enforcement,
  account scope/SIM/LIVE/stale/missing values, SQL pagination ordering, cached display isolation,
  derived-flow replay/accounting, writer wake-up and accelerated four-hour candidate rotation.
- Ruff format: **203 files already formatted**. Ruff lint: passed. Mypy: **117 source files passed**.
- All three Playwright scripts passed: dashboard desktop/mobile interactions, OAuth flow,
  request timeout/error/cancellation recovery, confirmed controls, obsolete filters/selections,
  balance labels and four simultaneous browser consumers. Hidden-tab refresh suppression passed.
- Locked server-only install/startup smoke passed in a fresh temporary environment, with network
  connections forbidden during startup. No research/development-only import was required.
- Desktop/mobile fixture screenshots were regenerated and visually inspected. They contain
  illustrative values and a visible offline-fixture banner; they are not account evidence.

The host had no `npm` executable on the selected PATH, so the three scripts declared in `npm test`
were executed directly using the available Node runtime and the existing Playwright installation.
The checks above are newly executed, not copied from earlier deployment reports.

## Measurements

Recorder: five futures plus five options over two virtual hours; burst: five futures plus
16 options, seven triggers including five simultaneous, over 1,200 virtual seconds.

| Measurement | Normal before → after | Burst before → after |
|---|---:|---:|
| Ingest p99 | 11.53 → 2.11 ms | 7.17 → 2.47 ms |
| Maximum event-loop delay | 336.90 → 75.90 ms | 470.20 → 383.00 ms |
| CPU time | 274.88 → 34.88 s | 70.47 → 8.24 s |
| Elapsed time | 277.80 → 41.90 s | 70.53 → 8.37 s |
| Peak process RSS | 86.92 → 101.00 MiB | 138.11 → 143.98 MiB |
| Accounted rolling peak | 13.00 → 15.12 MiB | 24.66 → 26.78 MiB |
| Peak queued bytes | 0.54 → 0.57 MiB | 3.51 → 3.51 MiB |

Both versions retained **24,755** normal and **34,821** burst archived message rows,
with identical SHA-256 hashes per workload and no duplicate rows or recording gaps. Normal
captures completed; burst captures report INCOMPLETE solely because the test stops before their
required end. All futures/options retained 900 seconds of rolling coverage. Queues drained to zero.
The memory tradeoff is explicit: bounded derived history costs about 2.2 MB of accounted rolling
space. The 32 MiB rolling, 8 MiB/512-item queue and 2 GiB disk caps stayed unchanged.

Five concurrent API clients each refreshed 40 times against 400 closed fixture trades. These
include the full request sequence per page used by its respective frontend. Broker requests: **0**.

| Page | Response bytes before → after | p95 before → after |
|---|---:|---:|
| Overview | 85,154 → 3,736 | 4.17 → 0.45 ms |
| Markets | 85,154 → 12,919 | 3.45 → 1.01 ms |
| Opportunities | 145,871 → 62,314 | 4.08 → 0.68 ms |
| Execution | 145,874 → 28,923 | 4.18 → 1.03 ms |
| System | 110,694 → 9,490 | 4.47 → 0.37 ms |

API workload CPU: 0.739 → 0.130s; maximum event-loop delay: 60.29 → 12.44ms; peak RSS: 67.9 → 64.9 MiB.

Evidence: [normal before](cleanup-benchmarks/normal-before.json), [normal after](cleanup-benchmarks/normal-after.json),
[burst before](cleanup-benchmarks/burst-before.json), [burst after](cleanup-benchmarks/burst-after.json),
[API before](cleanup-benchmarks/dashboard-before.json), [API after](cleanup-benchmarks/dashboard-after.json).

The existing `scripts/saxo_recorder_benchmark.py` workload was reused, adding CPU and raw-row
fidelity measurements. Run with `--option-count 5 --virtual-seconds 7200`, or
`--option-count 16 --virtual-seconds 1200 --burst-events`, plus `--output <file>`.
`scripts/slrno_dashboard_benchmark.py --output <file>` measures page requests; `--legacy`
reproduces the former frontend request pattern. Baselines imported execution/dashboard/core
modules from the clean original checkout using PYTHONPATH; after runs imported this checkout.
RSS is sampled before evidence-hash materialization. These are individual runs, not a statistical
production benchmark; OS scheduling and batching influence maxima and compressed file sizes.
An intermediate faster recorder hit its queue cap; that result was rejected and the bounded writer
wake-up fix was verified with the complete-evidence final reruns.

These are offline, accelerated, generated workloads on this development machine. Recorder runs
exercise compression, the existing writer and fsync. Five API clients use an in-process ASGI transport;
network, TLS, reverse proxy and production contention are excluded. The four real browser clients
use a local fixture server. These measurements do not establish production latency or Saxo capacity.

## Migration, configuration and rollback

No operator action has been performed. For a separately authorized future rollout:

1. Keep execution disabled/unarmed. Verify the selected environment and ledger; use the established
   deployment procedure. Take a consistent SQLite backup including WAL state (SQLite backup API,
   or a cleanly stopped ledger), plus the configuration. Do not copy only a live main database file.
2. If a custom configuration explicitly pins `max_premium_risk_gbp: 10`, remove that override or set
   it to `50`. The shipped example inherits 50. Leave `entry_deadline_seconds: 20`, four slots,
   one contract, market selections, mappings, costs, credentials and recording gates unchanged.
3. On opening the ledger, `Store` automatically performs a single transactional table rebuild for
   the old reservation schema. Every prior row keeps its allocation, plan and timestamps; new
   `policy_pennies` equals its original allocation. Orders/fills and foreign-key references survive.
   New reservations use 5,000p; aggregate committed allocation includes any old 1,000p amounts.
   The migration checks foreign keys before commit and restores enforcement on failure. SQLite
   WAL and FULL synchronous durability are unchanged. Reopening is idempotent.
4. Verify integrity/foreign keys, old row counts and amounts, account environment/identifier,
   unavailable/stale rendering and disabled execution before any separately authorized enablement.
   A connected account alone does not mean market data, products or entry gates are ready.

Rollback is a coordinated code/configuration/ledger decision. **Do not run the old binary against
an upgraded ledger**: it assumes a six-column £10 reservation table. Before any new ledger activity,
restore the consistent pre-upgrade backup and matching old configuration with the old code. After
new reservations/orders/fills exist, preserve that evidence and reconcile exposure; use a forward
fix or an explicitly reviewed reverse migration. Never drop new activity or rewrite £50 trades as
£10 to force rollback. Keep execution disabled throughout.

## Remaining limitations

- Actual balance field availability, CalculationReliability and subscription behavior on the selected
  Saxo SIM/LIVE account remain unverified by an authenticated run. Unsupported/absent fields show
  Unavailable. Account data is a small in-process snapshot; after restart it remains Unavailable
  until a valid new snapshot arrives.
- Production performance, entitlements, actual futures/option selections and recording rights were
  not reverified. Persistent recording was enabled only inside generated offline test fixtures.
- The burst fixture deliberately ends at virtual minute 20, before its hour-long captures finish;
  those archives correctly report INCOMPLETE on shutdown. Their raw-row hashes are compared for
  fidelity; the two-hour workload and four-hour rotation regression exercise completed captures.
- Bounded quotas can still fail closed under workloads beyond those tested. No quota was expanded
  and no required evidence was discarded to obtain the reported performance.

## Independent review

### Standards

The initial review found one issue: rejected balance deltas could restore freshness using an older
accepted snapshot. A separate bounded, sanitized stream state now reconstructs all deltas, publishing
only verified values. Numeric-only updates cannot clear invalid reliability/currency. The reviewer
reproduced the correction and reported **no residual findings**.

### Spec

The initial review found two issues: the balance-delta issue above, and missing replacement evidence
when an old option was reattached to an overlapping archive. Reattachment now bridges unwritten
rows, supplies a checkpoint when needed, records event prehistory and preserves sequence uniqueness
when a retired rolling window returns. Both retained-window and retired-window scenarios are tested.
The reviewer reran the original reproductions and reported **no residual findings**.

Standards: 1 finding resolved, 0 outstanding. Spec: 2 findings resolved, 0 outstanding.
