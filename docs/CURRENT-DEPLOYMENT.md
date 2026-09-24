# Current server deployment

Verified on September 24, 2026 at 07:13 UTC.

The active release is `0080a786d78660e04d9a17d310b8322c4eecb3c8`.
All 332 tracked files in that release matched their Git blob hashes on the
server. Application code in the publication branch is identical to that
release; subsequent changes are documentation and packaging of the existing
frozen test fixtures. No new application deployment or restart was performed
to synchronize GitHub.

`stocker-v1.service` is active and runs `stocker first4-run`, using
`/etc/stocker/v1/first4.yaml` and `/var/lib/stocker/v1/first4.sqlite3`.
The observed runtime was connected and reconciled to its explicitly allowed
PAPER account, with no current runtime problem. LIVE remained disabled.

The effective configuration and deployment verification are recorded in
[the deployment snapshot](first4-opening-deployment-evidence.json).
`armed` is false and `arm_after_quote_check_on` is `2026-09-24`.
The opening verification status was `WAITING_FOR_OPEN`. The September 24
US session opens at 13:30 UTC (14:30 UK). Entries are permitted only if the
existing dated opening gate succeeds; this snapshot does not assert that the
future check has passed or that any PAPER trade has occurred.

The latest non-transmitting option checks after manual Gateway login did not
reproduce error 10197 and verified a combo increment of 0.01. Fresh real-time
two-sided option quotes remain a prerequisite for arming. See
[the check result](first4-openai-disconnect-verification.json).
Restoration of Gateway's normal nightly restart time remains unverified;
see [the recovery record](FIRST4-opening-deployment.md).

The independent three-scanner observer is unchanged. Broker credentials,
proxy credentials, local runtime databases and broker transaction records are
not part of this deployment snapshot.

The exact research rules, sources and PAPER execution conventions are in
[FIRST4.md](FIRST4.md). Earlier dated cutover reports are historical evidence.
