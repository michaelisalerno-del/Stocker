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


## Historical exit audit and deployment record

Moved from ARCHITECTURE.md on 2026-09-11; the following evidence describes the
140e50b release, not the current deployment.

### Exit audit verification and deployment (140e50b)

Before the audit, the current method already selected the first causal break, froze threshold-based
brackets independently of fills, and submitted the original T0+15 timed close. The mismatches were:
no-print expiry strictly after T0+5 → expiry at T0+5; TIMEOUT/closed-order callbacks treated as
unexpected → resolved through persisted broker identities; absent timeout identity tolerated during
recovery → explicit reconciliation failure; ignored late commissions and blank trade R/exit reason
→ idempotent commission enrichment and actual execution reporting. Existing broker tick rounding,
Q1 admission, account capacity, permissions, exposure and connectivity controls were retained.

Changed implementation files, relative to their package's `src` directory:

| File / function | Purpose |
|---|---|
| `stocker_execution/session_hard_method.py::expire_waiting_before` | Exact half-open window expiry without a new print. |
| `stocker_execution/execution_models.py::OrderPlan` | Preserve exact method prices alongside broker prices. |
| `stocker_execution/stage7.py::build_order_plan`, `reconcile` | Carry method prices; detect missing deadline identity on recovery. |
| `stocker_execution/execution_ledger.py::ExecutionRecord`, `record_fill`, `_refresh_aggregate`, `record_for_order` | Additive persistence, reference/fill/R diagnostics, exit reasons, late commissions and all-leg lookup. |
| `stocker_execution/runtime.py::record_fill` | Recognize deadline and settled-order callbacks without duplicate fill counts. |
| `stocker_execution/ibkr.py::read_fills` | Distinguish a missing commission report from a reported zero commission. |
| `stocker_dashboard/read_service.py::_order`, `_trade`, `_position_row` | Expose method provenance and execution diagnostics; populate trade R/exit reason. |
| `stocker_dashboard/static/dashboard.js::executionDetails` | Expandable method/fill diagnostics in order and position details. |
| `tests/test_session_hard_exit_contract.py` | 24 geometry, fill independence, deadline, causality, migration, recovery and reporting cases. |
| `tests/test_stage7_ibkr.py`, `tests/test_stage8_runtime.py` | Commission-report availability and deadline/late-commission callback regressions. |
| `docs/ARCHITECTURE.md` | Verified contract, recovery behavior and this audit record. |

Validation: `rtk .venv/bin/pytest -q` passed 852 tests with five existing warnings. The subsequently
added missing-deadline recovery case passed in the final 24-case focused suite. On the staged Linux
release, `pytest -o addopts= -q tests/test_session_hard_exit_contract.py tests/test_method_package.py
tests/test_stage7_ibkr.py tests/test_stage8_runtime.py` passed 118 tests. Ruff passed for changed
Python files; mypy passed all 42 core/execution/dashboard source files. JavaScript syntax and a
mocked-browser check of the expandable diagnostics passed. All order tests terminated at fakes.

Release `140e50b551033feae9d8dc29e2310ef0419c2d96` reached READY at 13:28 UTC on 2026-09-08,
before the US open, with 6,570 qualified US instruments. Deployment verified the saved US/LSE/ASX
runs and configuration unchanged, all frozen artifact bytes unchanged, the additive migration,
one existing execution plan, 28 fills and one historical trade retained, and no open orders or
positions. Backup: `/var/lib/stocker/backups/exit-contract-140e50b` on the server.
No broker orders were placed by implementation/testing. LIVE remains disabled for this method.
Venue execution of the timed order was not tested with a real order; only its emitted broker
contract and fake-broker lifecycle were verified. The existing LSE/ASX runs had missed their
local capture windows and remain unvalidated cross-market PAPER tests, as described above.

Migration is additive: new method tables and nullable execution provenance/deadline/timeout
columns; old rows and research artifacts are retained. Old signals deserialize with absent new
fields. Legacy configuration enums and old calculation/payoff/scanner sources remain solely to
read history and reproduce research. Saved V7 runs remain runnable with their original specifications;
new run creation selects V9. Earlier archived method versions remain read-only.
There is no destructive reset or conversion of old decisions into the new method.

Retired runs can be marked `archived: true` with `enabled: false`. They disappear from operational
run lists while remaining available to historical trades, orders and candidate details. Archived
runs cannot be enabled; archiving never deletes ledger rows or changes broker orders.

The FastAPI/vanilla-JS dashboard is a consumer/controller of these boundaries. Market, Method
and Start PAPER run are primary. Account risk/capacity and detailed method provenance are
expandable. Standalone dashboard mode edits saved configuration but does not connect or trade.
Dashboard failures do not stop execution.

## Release procedure

1. Record the active release, service command, configuration hashes, run identities,
   PAPER/LIVE readiness and broker-authoritative positions/open orders. Verify the
   proposed revision includes required concurrent changes. Do not infer broker
   readiness from a listening socket.
2. Complete `bash scripts/check.sh` and review the diff for frozen artifacts, market
   times, account routing and unintended configuration changes. Record local results
   separately from GitHub CI.
3. Prepare an immutable release directory and run
   `uv sync --locked --no-default-groups --group server` there. Use its prepared
   `.venv/bin/stocker` directly. Set `STOCKER_BUILD_REVISION` to the verified commit.
   Verify imports/model/static assets offline; do not reuse the live environment for testing.
4. Prepare the explicit dashboard security mode and exposure ceiling migration.
   Missing ceilings block new entries. Never invent a ceiling, FX rate or account mapping.
   Keep original config and service/proxy settings for rollback.
5. Pause new entries before a controlled cutover; read back both saved and applied state.
   Check unresolved submissions and broker-held protection. A pause does not flatten,
   cancel protection or authorize discarding history.
6. Quiesce config changes and make a consistent backup below. Switch the release symlink
   atomically, then restart only the application under explicit deployment authorization.
   Do not restart Gateway or change broker accounts as part of a code release.
7. Allow the measured startup/reconciliation interval. Verify exact code revision,
   PAPER/LIVE/account identity, ledger recovery, run readiness/reasons, dashboard authentication,
   assets and database/config compatibility. “Saved” or an HTTP 200 alone is insufficient.
   Historical missed-window/acquisition failures must remain visible; do not erase them
   to make the release appear healthy.

These steps describe the supported procedure. A release report must say which steps
were actually performed, with the revision, timestamp, backup location and results.

## Consistent backup and disconnected restore

Use SQLite's online backup API, never a bare copy of an active database. The helper
bundles the consistent database snapshot with runs/IBKR configuration, any referenced
named-universe snapshot, frozen artifacts and SHA-256 manifest. Keep control edits
quiescent during capture; config changes detected during copying fail the operation.

```bash
.venv/bin/python scripts/backup_state.py \
  --database /var/lib/stocker/v1/runtime.sqlite3 \
  --runs-config /etc/stocker/v1/runs.yaml \
  --ibkr-config /etc/stocker/v1/ibkr.yaml \
  --artifacts packages/stocker_core/src/stocker_core/method_artifacts/session_hard \
  --output /var/lib/stocker/backups/RELEASE-UNIQUE
```

Resolve the artifact path from `stocker_core.methods.ARTIFACTS` in the prepared release
before running; package layouts may differ. The output directory must not already exist.
Treat backups as private because they include account configuration.

Restore into an isolated directory and verify the manifest and SQLite integrity.
Load the matching configuration, method artifacts and release there. Use a standalone
dashboard or fake broker only, with network blocked; never point a restore exercise
at the real Gateway. Confirm run IDs, fills, unresolved plans and idempotency survive.
The regression test keeps a WAL writer open and confirms committed pages are captured
while an uncommitted row is absent.

A database restore in production is a separate reconciliation operation: broker orders
and fills may have occurred since the snapshot. Retain the current database and logs.
Recover against authoritative broker open/completed orders, executions and positions
before any new entry. Never delete unknown exposure or reuse old signals to bypass recovery.

## Rollback and diagnosis

For a move to different CPU hardware, verify frozen candidate fixtures in the prepared
environment before activating it. The installed console launcher configures NumPy's
x86 V2/V3 numerical path before imports, as do pytest and the isolated server smoke.
Do not bypass this initialization when embedding numerical modules. Some AVX-512
NumPy kernels differ by one ULP from the frozen evidence even with the same lockfile.
Inspect `numpy.show_runtime()` and `numpy.lib.introspect.opt_func_info(func_name="^log$")`
after calling `stocker_launcher.configure_numeric_runtime()` when diagnosing this.
Do not loosen score assertions, regenerate fixtures or change selection to mask it.

The account-currency sizing migration adds valuation/currency evidence and price-unit
and normalized minimum-quantity/increment columns to the ledger. It does not infer units
for old records. Before enabling any market, verify broker contract quotation/quantity
units, fresh broker FX where required, account-currency risk/notional settings and the
non-executing credit preview. Missing broker size metadata blocks entry; it is not
permission to substitute one-share lots. Fractional execution remains unsupported.
An active legacy record with unknown currency blocks new entries until settled.
Rollback to a release without these unit/quantity controls must keep new entries paused:
it cannot apply the complete admission policy, even though additive columns are readable.
Keep the current database, valuation evidence and broker protection through recovery.

Code rollback switches to a retained release; it is **not** permission to restore an older
database or configuration over newer history. This release adds nullable ledger columns
and an optional YAML exposure setting; earlier binaries may reject newer configuration
or omit the new admission policy. Check compatibility in isolation first. Keep new entries
paused if rolling back to code without the shared commitment protection.

For a failed activation, inspect its reference ID, System revision/config hash, run status,
environment reconciliation, data capacity and checkpoint coverage. For an uncertain control
response, read back the existing run/start operation before retrying. For unknown submission,
reconcile broker evidence; never release the reservation based on elapsed time.

The continuation adds nullable `broker_reported_entry_filled` metadata to existing
execution plans. Upgrade retains all identities, fills and protection. A migrated
terminal order with an unknown cumulative fill remains unresolved until fresh broker
status/execution reconciliation; it is not assumed flat. A terminal status received
before its fills also blocks new entry until those details arrive. Investigate the
broker's executions and protective orders if that state persists; do not remove the row.
Rolling back to a binary that ignores this metadata restores the callback-ordering
gap, so keep entries paused until using a verified compatible release.

The latest isolated restore regression and opening-to-dashboard rehearsal are listed
in [continuation acceptance](robustness-implementation.md#continuation-acceptance).
They use temporary state and test inputs, and do not verify production scheduling,
offsite retention or current server security.

Recommended server checks include systemd state and ExecStart, socket bind addresses,
proxy authentication on controls/streams, spoofed-header rejection, TLS validity and an
external backend-port probe. An isolated restore does not establish that the production
backup schedule, retention, firewall or network boundary works. Record those separately.
