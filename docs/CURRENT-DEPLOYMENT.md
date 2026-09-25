# SLRNO — current server deployment

Verified September 25, 2026 at **11:19:09 UTC (12:19 UK)**.

Active release: **`42a266d02a8dc83dfa363445f0e2f1a7c3d363c1`**, pushed to GitHub
`main` and activated at **11:15:47 UTC** following explicit user authorization to
push and deploy. It adds FIRST4 order-flow observation V1. **Observation remains
DISABLED**; deployment did not authorize or enable additional data subscriptions.
See the [implementation and offline verification report](first4-order-flow-v1.md).

All six [CI checks](https://github.com/michaelisalerno-del/Stocker/actions/runs/36126802550)
passed for the exact deployed commit. Local validation passed 629 Python tests.
All 383 archived files matched their SHA-256 manifest. The locked 56-package
server installation and network-blocked smoke test passed as the service user.
All fourteen checked authenticated health/data/page/asset endpoints returned HTTP
200; served JavaScript, CSS and the new order-flow asset match the release.
Historical allocation cards and detail API explicitly report order-flow DISABLED.

Only `stocker-v1.service` was restarted: PID **1344003**, active with zero automatic
restarts. Worker, manager and web health are RUNNING; the normal PAPER client is
connected and reconciled. Gateway PID **1335744** stayed unchanged. The separate
frozen three-scanner collector, service and timer hashes are unchanged; its service
remains inactive pending its existing schedule. No additional broker probe was run.

Execution configuration SHA-256 remains
`5dfd04751b08394d8e7f47a0f84442c2b1e5861cfe6e8e1a343d3bbd79a53818`.
Configured and effective arming are false; LIVE remains disabled. The existing
September 25 opening permission and WAITING_FOR_OPEN state remain intact. All 51
events, permanent slots, session rows, dated checks and pause state were preserved.
Orders, fills, positions and outstanding obligations are zero. The only changed
metadata key was the normal startup `reconciled_at` timestamp. SQLite quick_check
passed.

Root-only consistent ledger/configuration backups and activation/verification
records are at `/var/lib/stocker/backups/first4-42a266d02a8d/`. The previous release
`2a31cde8a31a09f350d197d03b4dee779485819a` is retained for a coordinated code rollback.
Never restore the ledger snapshot over subsequent trading activity. No operational
configuration, arming or observation enablement was changed. Reload an existing
browser tab to load the new assets.

Status: **DEPLOYED_WITH_OBSERVATION_DISABLED**. The order-flow feature remains
**IMPLEMENTED_AND_OFFLINE_TESTED**, not **CONNECTED_AND_RECORDING_VERIFIED**.
Actual order-flow entitlement, pacing, units and recording still require separately
authorized verification and explicit observation configuration.

[Sanitized deployment evidence](first4-order-flow-deployment-20260925.json).

## Previous September 24 20:51 UTC deployment record (superseded)

Verified September 24, 2026 at **20:56:22 UTC (21:56 UK)**.

Active release: **`2a31cde8a31a09f350d197d03b4dee779485819a`**, published to
GitHub `main` and activated at **20:51:27 UTC** after explicit user authorization
to deploy. This release repairs fresh diagnostic-stock acquisition, replaces the
three-attempt opening limit with deadline-driven retries, propagates streaming
request errors, interrupts failed anchor requests during metadata work, and keeps
option streams fresh while BAG metadata is pending. See the
[repair report](SLRNO-operational-repair-20260924.md).

All six [CI checks](https://github.com/michaelisalerno-del/Stocker/actions/runs/36054172363)
passed for the exact release. Local full Python validation passed **581 tests**
with five existing warnings. All 371 release files matched the SHA-256 manifest;
the locked 56-package server installation and service-user offline smoke passed
before activation. Served JavaScript/CSS match the release. All twelve checked
authenticated health, data, page and asset endpoints returned HTTP 200.

Only `stocker-v1.service` was stopped and started: **PID 1314073**, active with
**zero automatic restarts**. Worker, manager and web health are RUNNING; PAPER is
connected and reconciled. Gateway PID 1274679 remained unchanged. The observer
was already inactive before deployment and remains inactive; its service/timer
definitions were unchanged.

Execution configuration SHA-256 remains
`5dfd04751b08394d8e7f47a0f84442c2b1e5861cfe6e8e1a343d3bbd79a53818`.
Configured and effective arming remain false; LIVE remains disabled. The existing
September 25 permission and **WAITING_FOR_OPEN** record are preserved. No opening
check was replayed. All 51 event records, four consumed slots, session blocks,
dated checks and pause state matched the pre-activation snapshot. Orders, fills,
positions and exit obligations remain zero. The ledger passed `quick_check`.

Root-only consistent ledger/configuration backups and verification records are in
`/var/lib/stocker/backups/first4-2a31cde8a31a/`. The previous release `92393ff` is
retained. Never restore the ledger backup over subsequent trading activity.
Only the normal service startup reconnected to IBKR; no additional broker probe,
manual order, configuration change or arming change was performed. Live opening
data delivery and real fills remain unverified by this deployment.

See the [sanitised verification record](slrno-operational-deployment-20260924.json).

## Previous 17:16 UTC deployment record (superseded)

Verified on September 24, 2026 at 17:16:44 UTC (18:16 UK).

The active release is **`92393ff7c62ca7c0d7c612252797d684aa792732`**, published to
GitHub `main` and activated at **17:14:23 UTC** after the user explicitly requested
push, deployment and restart. It adds the SLRNO interface, compact page-specific
reads, permanent allocation cards and stable browser refresh. Frozen FIRST4 strategy,
execution and safety semantics are unchanged; LIVE remains disabled.

All 368 tracked release files matched their SHA-256 manifest before activation.
All six [CI checks](https://github.com/michaelisalerno-del/Stocker/actions/runs/36031473141)
passed for this exact commit. The locked server-only install contains 56 distributions
including Stocker; the service-user offline smoke passed with network connections
forbidden. The initial staged install could not write the service user's default uv
cache; using a cache inside the new release resolved this before touching the running
service. No existing cache permissions were changed.

`stocker-v1.service` stopped cleanly and started once, **PID 1307977**, with **zero
automatic restarts**. Worker, manager and web health are RUNNING. PAPER is connected
and reconciled; the ledger is available. Authenticated health, Overview, Opportunities,
Execution, System, allocation detail and static assets returned HTTP 200. Served
JavaScript/CSS hashes match the release. The real current-session Overview response
was 6,409 bytes. Existing browser tabs should reload to load the new UI/API contract.

**Execution configuration was not changed.** Its SHA-256 remains
`5dfd04751b08394d8e7f47a0f84442c2b1e5861cfe6e8e1a343d3bbd79a53818`.
`armed` remains false; the existing `arm_after_quote_check_on: 2026-09-25` permission
and **WAITING_FOR_OPEN** state remain intact. September 24 retains the existing
`SESSION_START_OR_SCANNER_HISTORY_MISSED` entry block. No date change, rearming,
opening-check replay, pause mutation or slot reset was performed.

All 51 candidate events remain. Today's four permanent slots are still GCDT, MMTIF,
SOS and PMAX, with their existing EXECUTION_FAILED outcomes, correctly displayed as
BLOCKED / FAILED. Orders, fills, positions and exit obligations remain zero.
Both the consistent backup and runtime ledger passed SQLite `quick_check`.

Root-only pre-activation ledger/configuration backups and verification records are in
`/var/lib/stocker/backups/first4-92393ff7c62c/`. The former release remains available
at `/opt/stocker/releases/b5d6962cbd2d13eb39129ed0a116d9f4c87a9cac`.
Do not restore the ledger snapshot over subsequent trading activity.

Gateway **PID 1274679** and independent scanner observer **PID 1297975** stayed active
and unchanged. Observer collector/service/timer hashes were unchanged. Only the existing
FIRST4 service reconnected through its normal startup; no extra broker client or test
order was used. Credentials are excluded from the
[verification record](slrno-deployment-20260924.json).

See [implementation and local measurements](SLRNO-implementation.md).

## Previous 15:32 UTC deployment record (superseded)

The following is the prior release and permission record, retained as historical evidence.

Verified on September 24, 2026 at 15:32 UTC (16:32 UK).

The active release is `b5d6962cbd2d13eb39129ed0a116d9f4c87a9cac`, published to
GitHub `main` and activated at 15:31:51 UTC after explicit user authorization.
It adds the approved native `averageOptionVolumeAbove=1` scanner restriction.
All 353 tracked files matched the release manifest. The locked server-only
installation and offline smoke passed. All six
[CI checks](https://github.com/michaelisalerno-del/Stocker/actions/runs/36020107705)
passed before activation; local full Python validation passed 530 tests with five
existing warnings. See [the approved universe change](FIRST4.md#approved-scanner-universe-change--september-24-2026)
for the read-only broker filter verification and historical-fixture limitations.

The user subsequently authorized preparing the September 25 PAPER session.
Only `arm_after_quote_check_on` changed, from `2026-09-24` to `2026-09-25`;
`armed` remains false and every other parsed configuration value was verified
unchanged. The new configuration SHA-256 is
`5dfd04751b08394d8e7f47a0f84442c2b1e5861cfe6e8e1a343d3bbd79a53818`.
The authenticated dashboard now shows **2026-09-25 — WAITING_FOR_OPEN**.
The service is prepared to run the existing dated opening checks and may enable
PAPER entries only after they pass. This is not a recurring authorization or a
claim that tomorrow's quotes or trading will succeed. LIVE remains disabled.

FIRST4 PID 1305023 is active with zero automatic restarts. Worker, manager and
web health are RUNNING; the broker is connected and reconciled. Current entries
are unarmed, and September 24 retains `SESSION_START_OR_SCANNER_HISTORY_MISSED`
and its last 14:59 UTC scanner observation. All four consumed slots and the
September 24 opening audit remain intact. Orders, fills, positions and exit
obligations are zero. No replay, slot reset or further broker probe was performed
during this activation.

Consistent root-only ledger and prior-configuration backups are in
`/var/lib/stocker/backups/first4-b5d6962cbd2d/`, as `first4-before.sqlite3` and
`first4-before.yaml`. The ledger snapshot passed SQLite `quick_check`.
Gateway PID 1274679 and observer PID 1297975 remained running; observer
collector/service/timer hashes were unchanged.

## Previous 15:00 UTC deployment record (superseded)

The following describes the prior release and September 24 permission, before
the explicitly authorized deployment and September 25 date change above.

The active release is `1fef5daa2c375466e82accaf787353e1b2820b1f`, published to
GitHub `main` and activated with explicit user authorization at 14:59:57 UTC.
All 352 tracked release files matched the commit. The locked server-only
installation installed 60 packages; `scripts/server_smoke.py --installed` passed
with broker/network connections prohibited by the smoke test. All six
[CI checks](https://github.com/michaelisalerno-del/Stocker/actions/runs/36015646462)
passed for this exact commit, including 528 Python tests (five warnings).

The release preserves wire trade timestamps for baseline anchors, reports
subscription errors without waiting for a generic timeout, and distinguishes
empty, incompatible and ambiguous option chains with bounded diagnostics.
See the [entry failure investigation](FIRST4-entry-failures-20260924.md).
It does not establish the missing broker evidence behind today's four failures.

FIRST4 stopped cleanly and started once, PID 1302657, with zero automatic
restarts. The authenticated dashboard confirmed worker, manager and web health
RUNNING, broker connected/reconciled, and zero positions or exit obligations.
New entries are **unarmed**, with `SESSION_START_OR_SCANNER_HISTORY_MISSED`
following the mid-session restart. The last recorded scanner observation is
14:59 UTC. The opening check's persisted ARMED record is historical evidence;
the new process has no effective dated opening authority and did not replay it.
Option quote readiness is UNOBSERVED in the restarted process.

Today's four permanent slots remain GCDT, MMTIF, SOS and PMAX, each with
EXECUTION_FAILED outcome. Before activation the consistent ledger snapshot
contained zero orders, fills and positions. No slots, blocks, permissions or
opening-check state were reset. A fifth opportunity cannot replace those slots.
Another session requires its own explicit dated authorization under the existing
protocol; this deployment does not grant it.

The root-only consistent backup passed SQLite `quick_check` and is stored at
`/var/lib/stocker/backups/first4-1fef5daa2c37/first4-before.sqlite3`.
Do not restore it after subsequent ledger activity without coordinated recovery.
The execution configuration hash remains
`e9e5c4200332b4cd22ac808ab2217d75581f154272497f3b4384106053a31c12`.
Gateway PID 1274679 and independent observer PID 1297975 remained running.
Observer collector/service/timer hashes were unchanged. No additional diagnostic
broker client, manual order, configuration change or manual arming was used.

## Previous release and investigation record (superseded)

The following records the earlier release and the investigation before the
14:59:57 UTC activation above. Statements about a pending patch or premarket
readiness below describe that earlier state, not the current deployment.

Earlier service verification: September 24, 2026 at 12:31 UTC.

The active release is `b655d4b27cd4b86662f0c25f8111eb0a0b68988c`.
The subsequent source fix preserving broker trade timestamps and adding
stage-specific timeout diagnostics is **not deployed**. The running process has
no supported code-reload mechanism and was left uninterrupted at the user's
request. Loading the fix requires a planned restart; restarting during the dated
session would revoke its in-memory entry authority, without permission to replay
the opening check or recover missed scanner observations.

Offline validation for that pending patch: eight focused wire-decoder/entry tests
passed (six regressions failed against the prior implementation); the complete
Python suite passed 513 tests with five existing warnings. Repository format,
lint, typing and locked server-only smoke checks passed. The frontend test entry
point passed using bundled Node directly because local `npm` is unavailable.
Frozen fixtures, calculation code, execution configuration and lockfiles were
unchanged. These checks do not establish why MMTIF's live stream supplied no
anchor, nor do they make the patch active in the running process.

The pending follow-up also distinguishes chain-response failures and propagates
tick-subscription request errors to the anchor wait. See the dated
[entry failure investigation](FIRST4-entry-failures-20260924.md) for evidence,
remaining uncertainty and activation constraints. None of these source changes
has replaced the active release identified above.

All 349 tracked release files matched the published commit. The six GitHub CI
checks passed before activation, and the server's locked server-only installation
and offline smoke passed. FIRST4 was restarted at 12:28:48 UTC after explicit
user authorization. This includes the supervised-runtime repair and preservation
of the separately deployed `a8fe377` safeguards; see
[the integration record](FIRST4-deployment-integration-20260924.md).

The latest change permits up to three attempts for narrowly classified temporary
read-only opening-check failures, with five seconds between attempts. The first
two attempts are capped at 60 seconds; the final attempt retains the original
13:44 UTC deadline. Safety failures and connection-generation changes still stop
the check. Durable CHECKING state prevents restart replay. The dashboard now
shows attempt counts, next attempt time and the last attempt error. The exact
release passed all six [CI checks](https://github.com/michaelisalerno-del/Stocker/actions/runs/35998962800),
including 505 Python tests (five warnings) and frontend tests.

During the earlier 11:54 deployment, the first start failed before application
launch because that release lacked
the systemd-required `release.env`. It contains only `STOCKER_BUILD_REVISION`;
creating it with the correct public commit identifier and resetting systemd's
failed-start counter resolved that failure. No ledger block was cleared. The
latest release included this metadata before its successful first start. Its
process has PID 1295813 and zero automatic restarts at verification time.
Future installations must create this metadata before start.

`stocker-v1.service` is active and runs `stocker first4-run`, using
`/etc/stocker/v1/first4.yaml` and `/var/lib/stocker/v1/first4.sqlite3`.
The authenticated dashboard showed worker, broker manager and
web health RUNNING, ledger available, and PAPER connected/reconciled on client 81.
Reconciliation completed at 12:28:51 UTC. LIVE remained disabled. Orders, fills,
positions and outstanding obligations were all zero; today's session had no
continuity block and no scanner observations yet (premarket).

The earlier effective configuration is recorded in
[the historical deployment snapshot](first4-opening-deployment-evidence.json).
The configuration file's SHA-256 was unchanged before/after this restart:
`e9e5c4200332b4cd22ac808ab2217d75581f154272497f3b4384106053a31c12`.
`armed` is false and `arm_after_quote_check_on` is `2026-09-24`.
The opening verification status was `WAITING_FOR_OPEN`. The September 24
US session opens at 13:30 UTC (14:30 UK). Entries are permitted only if the
existing dated opening gate succeeds; this snapshot does not assert that the
future check has passed or that any PAPER trade has occurred.

The new process reports option quote state UNOBSERVED, no active data-farm or
10197 block, and no missing execution settings. No additional diagnostic broker
connection or test order was made for this restart. Earlier explicitly requested
read-only LSE diagnostics confirmed live VOD quotes at 12:00 UTC and historical
bars at 12:01 UTC; diagnostic client 181 disconnected afterward. Those results
do not establish US option readiness. Fresh real-time two-sided option quotes and
combo metadata must still pass the existing opening gate; this is premarket
service readiness, not proof of executable quotes or a successful trading session.
The earlier non-transmitting check remains
[historical evidence](first4-openai-disconnect-verification.json).
Restoration of Gateway's normal nightly restart time remains unverified;
see [the recovery record](FIRST4-opening-deployment.md).

The independent three-scanner observer's collector, service and timer hashes
were unchanged. Its service was inactive with the existing 13:29 UTC timer.
Gateway PID 1274679 was unchanged. Socket/process inspection found one Gateway
and only FIRST4's API connection; client 181 was already disconnected. No local
TWS/Gateway process was running on the operator's Mac.

A consistent root-only ledger backup, verified with SQLite `quick_check`, is at
`/var/lib/stocker/backups/first4-b655d4b27cd4/first4-before.sqlite3`.
The live ledger passed `quick_check` at the earlier migration. Keep the backup for a
coordinated rollback: older application code must not blindly reuse the expanded
schema, and restoring this premarket snapshot after new executions would lose
ledger history. No rollback, manual arming, permission change or state reset occurred.

Broker credentials,
proxy credentials, local runtime databases and broker transaction records are
not part of this deployment snapshot.

The exact research rules, sources and PAPER execution conventions are in
[FIRST4.md](FIRST4.md). Earlier dated cutover reports are historical evidence.
