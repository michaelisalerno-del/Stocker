# Current server deployment

Verified on September 24, 2026 at 12:31 UTC.

The active release is `b655d4b27cd4b86662f0c25f8111eb0a0b68988c`.
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
