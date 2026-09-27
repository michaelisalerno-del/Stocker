# FIRST4 source repair verification — September 24, 2026

Starting commit: `505d2429cf109a733f63d296a9943b974b222dcc`.
The task directory was empty, so the repository's current default branch was
cloned there. It happened to equal the review commit; no historical checkout or
revert was performed. The resulting starting tree was clean. Existing local
installations and production state were not used.

This is source verification, not deployment verification. No brokerage
connection, account check, market-data download, order transmission, service
restart, deployment, push, merge, activation change or observer change occurred.

## Findings and evidence

Source links below are relative to this repository. Regression names refer to
[test_first4_repairs.py](../tests/test_first4_repairs.py) unless stated otherwise.

| Finding | Result | Implementation and evidence |
|---|---|---|
| Silent manager/worker termination; failed reporting | FIXED | [Runtime](../packages/stocker_execution/src/stocker_execution/first4_runtime.py) supervises critical tasks, inhibits entries inside the failing task, retrieves background exceptions and bounds cancellation waits. [CLI](../packages/stocker_core/src/stocker_core/cli.py) propagates worker failure to process exit and retains management after listener failure. `test_critical_failure_and_reporting_failure_are_visible`, `test_background_execution_exception_is_retrieved`, `test_cli_surfaces_worker_failure_and_web_failure_keeps_worker`. |
| False pre-market continuity block | FIXED | Required observation clock advances only after the scanner observation commits. Reconnect itself no longer marks a populated session invalid. `test_reconnect_continuity_depends_on_due_observations` covers repeated reconnects, a genuine missed minute and rollover. |
| Dated opening authority could return after an interrupted check | FIXED | Opening verification rejects a changed data generation; its durable failed/interrupted result prevents replay. `test_opening_check_cannot_reauthorize_after_interruption`; existing opening/expiry tests remain passing. |
| Local API connectivity mistaken for upstream/data readiness; empty vs failed requests | FIXED | [Broker](../packages/stocker_execution/src/stocker_execution/first4_broker.py) tracks connectivity/farm notices independently, revokes dated authority on loss, and reconnects once after 1101 to restore account subscriptions. [Pinned request adapter](../packages/stocker_execution/src/stocker_execution/first4_requests.py) raises request errors and always clears owned request state. `test_pinned_library_request_semantics_and_cleanup`, `test_upstream_restoration_and_notices_do_not_clear_independent_blocks`. Historical-farm outages inhibit entries but do not block otherwise safe fresh exit quotes. |
| Full-history management reads and unchanged terminal writes | FIXED | [Store](../packages/stocker_execution/src/stocker_execution/first4_store.py) supplies indexed active/deadline/allocation queries. Completion markers are reactivated by late execution changes; the quantity join is explicitly driven by active entries. `test_management_and_dashboard_cost_independent_of_closed_history` measures statement/write counts and SQLite instruction counts, not just elapsed time. `test_terminal_outcome_does_not_repeat_writes`. |
| Unknown pending exposure and configured-arming bypass | FIXED | Unknown PAPER open orders block new entries. Session validity applies to both authorization paths and is rechecked at submission. Existing exits use independent account/ownership/quote checks. `test_unowned_pending_order_blocks_entries_only`, `test_configured_arming_cannot_bypass_session_block`, `test_session_block_during_entry_preparation_cannot_submit`; existing pause-during-preparation tests. |
| Duplicate candidate history inputs | NOT REPRODUCED | The 960-second current-session request and 60-second previous-close request have different identities. Each symbol is evaluated only at first appearance. Neither request was removed or cached under a different identity. |
| Candidate cancellation leaks and batch failure consequences | FIXED | Failed siblings drain together; a shared 43-second candidate budget remains inside the existing 45-second scan timeout. Concurrency remains four. Known native appearances commit together by rank; failed history remains explicit and cannot be retried as a later appearance. `test_scanner_burst_rank_partial_failure_and_cancellation`, pinned request cleanup test, benchmark below. |
| Full-ledger dashboard refresh; overlapping requests; unchecked pause | FIXED | [Dashboard API](../packages/stocker_dashboard/src/stocker_dashboard/app.py) returns ordered, capped session pages and aggregate session accounting. Health survives ledger-view failure. [Browser](../packages/stocker_dashboard/src/stocker_dashboard/static/dashboard.js) permits one refresh, uses an eight-second timeout, rejects stale control-era responses and checks pause responses. [Browser regression](../tests/dashboard_first4.cjs); `test_pause_write_failure_is_not_reported_as_success`. Existing dashboard-security tests pass. |
| Weekend/holiday calendar rebuilds | FIXED | Cache validity uses the NY local date, separately from the last trading date. `test_calendar_refresh_is_once_per_local_date` covers weekend, holiday, early close and both sides of DST; rollover is covered separately. |
| A neighbouring expiry failure aborts a provably valid selection | FIXED, narrowly | Successful empty searches or explicit no-security-definition errors can be skipped only when another broker timestamp exactly matches the frozen target, proving it cannot be outranked. Otherwise missing metadata remains an explicit ambiguity. Error 200's ambiguous-contract variant is not treated as absence. `test_bounded_expiry_alternatives`; original nearest/tie/standard-contract tests. No search expansion. |
| Lost/stuck exit obligations, repeated exceptions and late reports | FIXED | Active obligations remain visible through terminal rejection, partial legs, ambiguous reservations and session close. Changed errors are recorded once. Late fills/corrections reactivate obligations; older correction revisions remain as audit records and are excluded from effective accounting. `test_incomplete_exit_never_resubmits_or_finishes`, `test_late_fill_reopens_obligation_and_corrections_do_not_double_count`, `test_pre_change_schema_migration_preserves_ledger`. |
| Entry budget/pause/arming prevent owned exits; blanket stale quote acceptance | ALREADY FIXED in baseline | Existing `test_exit_ignores_entry_budget_pause_and_arming_but_never_invents_fills`, `test_exit_rechecks_positions_after_async_quotes`, and parameterized quote-freshness tests passed before changes. These protections were retained. |
| All realised results hidden by another open allocation | FIXED | Session P&L groups actual leg executions by allocation; completed known-fee net results, completed gross, partial-close gross, exposure and pending fees are separate. `test_completed_pnl_survives_open_allocation_and_late_fees` includes restart reads and idempotent late fees. |
| README server install pulls default research/dev groups | FIXED | README now matches the smoke-tested `uv sync --locked --no-default-groups --group server`. No dependency upgrade or speculative dependency deletion. Server smoke installed 60 locked packages and forbade network connections during runtime startup checks. |
| Frozen strategy, account restrictions, durable reservation and security | ALREADY FIXED / PRESERVED | Original 1,293-appearance/80-allocation equivalence and ownership/submission/security tests remain unchanged and passing. Frozen fixture files, manifests, strategy calculations, example configuration and lockfiles have no diff. |

The compact integration regression
`test_fault_injection_scanner_management_dashboard_and_persistence` combines a
real scanner commit, owned exit management, concurrent ASGI reads, upstream
loss/restoration and an injected persistence/reporting failure. It asserts
truthful health, retained obligations, cancellation of critical siblings and
exactly one fake exit submission.

## Verification commands

All shell invocations used the required `rtk` prefix. Dependency installation
was isolated to this checkout or temporary test environments.

| Command | Actual result |
|---|---|
| `rtk uv sync --all-groups --locked` | Passed; pinned `ib-async==2.1.0`, Python 3.12.13. |
| `rtk uv run --no-sync pytest tests/test_first4.py tests/test_dashboard_security.py tests/test_ci_smoke.py` before edits | 50 passed. |
| `rtk uv run --no-sync pytest tests/test_first4_repairs.py` before fixes | The initial seven regressions all failed on their intended symptoms. |
| `rtk uv run --no-sync pytest tests/test_first4.py tests/test_first4_repairs.py tests/test_dashboard_security.py` after fixes | 91 passed. |
| `rtk bash scripts/check.sh format` | 186 files already formatted. Only touched files were formatted. |
| `rtk bash scripts/check.sh lint` | All checks passed. |
| `rtk bash scripts/check.sh typing` | No issues in 108 source files. |
| `rtk bash scripts/check.sh python` | 474 passed in 59.39 seconds; five warnings (Starlette/httpx deprecation and four existing empty-slice warnings in research tests). |
| `rtk bash scripts/check.sh server` | Passed fresh locked server-only installation, FIRST4 imports, offline dashboard startup and assets. |
| `rtk git diff --check` | Passed. |
| `rtk git diff --exit-code -- tests/fixtures/first4 configs/first4.example.yaml uv.lock package-lock.json packages/stocker_execution/src/stocker_execution/first4.py packages/stocker_execution/src/stocker_execution/first4_config.py` | Passed; no frozen/configuration/lockfile drift. |

The host had no `npm` on PATH. Frontend CI's exact check script was run with
bundled Node 24.19.0 and a temporary npm 10.9.2 runner (CI specifies Node 22):

```sh
rtk proxy env PATH=/Users/michaelsalerno/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:/usr/bin:/bin /Users/michaelsalerno/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/pnpm dlx npm@10.9.2 ci
rtk proxy env PATH=/Users/michaelsalerno/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:/usr/bin:/bin /Users/michaelsalerno/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/pnpm dlx npm@10.9.2 exec -- playwright install chromium
rtk proxy env PATH=/Users/michaelsalerno/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:/usr/bin:/bin /Users/michaelsalerno/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/pnpm --package=npm@10.9.2 dlx -c 'bash scripts/check.sh frontend'
```

Result: passed escaping, removed controls, mobile navigation, one refresh in
flight, stale response rejection, timeout recovery and failed/successful pause.
Node 22/Linux CI was not run locally; no result is claimed for that environment.

## Performance evidence

Command: `rtk uv run --no-sync python scripts/first4_offline_benchmark.py`.
The benchmark reads baseline source from the local starting commit and forbids
every socket connection. Both versions receive identical synthetic ledger
content: 1,000 closed allocations and one active allocation, with fake brokers.

| Measurement | Baseline | Repaired |
|---|---:|---:|
| SQL statements per ordinary management cycle, including transaction statements | 9,004 | 3 |
| Actual SQLite writes in that cycle | 1,000 | 0 |
| Overview response bytes | 2,358,066 | 4,115 |
| Candidates / orders / fills returned | 1,001 / 2,001 / 4,002 | 1 / 1 / 2 |
| Calendar builds over 20 Saturday iterations | 20 | 1 |
| History requests for 25 new candidates | 50 | 50 |
| Peak concurrent history requests | 4 | 4 |
| Requests still running when a partially failed batch returned | 1 | 0 |

At controlled two-millisecond fake history delays, successful batch times were
34.0 ms before and 34.3 ms after; the partial-failure batch was 44.1 ms before
and 34.1 ms after. The repaired successful batch recorded 30.7 ms maximum queue
time and 2.65 ms maximum request time. These illustrative wall times are not
acceptance thresholds or claims about IBKR latency. The meaningful invariants
are fixed request count/concurrency, stable native-rank commitment and zero
orphaned work. SQL instruction-count assertions additionally protect against
an optimizer choosing full historical fills as the aggregate's driving table.

## Operational prerequisites and limits

This patch is commit-ready source, not a deployed or armed release. Existing
dated deployment evidence was left intact. A separately authorized deployment
must back up the ledger, allow its additive migration and retain the existing
service supervisor. Startup/reconnect still requires account-wide broker
reconciliation; the ordinary-loop performance figures exclude that necessary
work. Migrated historical obligations start active until completion is proven.

Dated authority is never restored from audit records. A restart or interrupted
opening check does not re-run a completed/interrupted dated check, reconstruct
missed scanner appearances or authorize a new date. Rejected, ambiguous and
overdue exits can still require operator action; no new unattended retry or
after-close policy was added. All broker behavior here was exercised with
fakes or disconnected library objects, not an IBKR session.

Request handling was checked against the installed 2.1.0 implementation and
[its API documentation](https://ib-api-reloaded.github.io/ib_async/api.html).
Connectivity handling follows IB's documented
[1100/1101/1102 and farm notices](https://interactivebrokers.github.io/tws-api/message_codes.html).
Correction accounting uses IB's documented
[execution-ID revision convention](https://interactivebrokers.github.io/tws-api/classIBApi_1_1Execution.html).
The small request adapter depends on that pinned library's bookkeeping; a
future library upgrade must re-run its cancellation/error regressions.
