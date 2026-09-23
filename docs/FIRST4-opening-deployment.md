# FIRST4 opening verification deployment — September 23, 2026

Code `0080a786d78660e04d9a17d310b8322c4eecb3c8` was installed at
`/opt/stocker/releases/0080a786d78660e04d9a17d310b8322c4eecb3c8`, selected by
`/opt/stocker/current`, and restarted through the existing `stocker-v1.service`
at 22:48:21 UTC. No new service or trading pipeline was added.

The only configuration addition in `/etc/stocker/v1/first4.yaml` was
`arm_after_quote_check_on: '2026-09-24'`. `armed: false` and all accepted PAPER
conventions were preserved. The full effective configuration and service result
are in [the deployment evidence](first4-opening-deployment-evidence.json).
The root-only rollback backup is
`/var/lib/stocker/backups/first4-opening-0080a786d786/`.

The actual service was active, connected and reconciled to PAPER DUP655399,
with `WAITING_FOR_OPEN` for September 24, effective arming false, and LIVE false.
There were zero orders, fills and positions. Authenticated dashboard assets
matched the release. The independent observer's file hashes remained unchanged.
The September 23 missed-session status is retained; it is not authorization to
replay that day's scanner history.

On September 24 the calendar opens at 13:30 UTC (14:30 UK). The single opening
check must pass before 13:44 UTC, while the normal scanner records first
appearances. Success enables the same entry path without restarting. Failure
leaves entries disabled, and disconnect/restart cannot restore the memory-only
authorization. The earliest possible frozen admission is 13:45 UTC and baseline
entry is 13:46 UTC. This does not guarantee any candidate or fill.

Verification completed:

- All 430 tests passed, including saved frozen fixtures (1,293 appearances and
  80 allocations), PAPER isolation, restart, order and opening-check regression tests.
- Mypy passed across 105 source files; changed Python files passed Ruff lint and
  formatting; `git diff --check` passed.
- Dashboard browser checks passed, including menu navigation and escaped data.
- Locked server-only installation, CLI import/help, and installed offline
  dashboard/startup smoke passed.
- Independent standards and specification reviews found no remaining blocker
  after serializing reconciliation requests.

Broker readiness is still conditional. New non-transmitting checks at 22:48 UTC
verified PAPER identity and no open orders, but failed with
`COMBO_PRICE_INCREMENT_UNAVAILABLE`. A comparison using the previous FIRST4
read-only probe at 22:49:30 UTC also failed; IBKR returned error 10197 at
22:49:37 UTC: `No market data during competing live session`. That probe qualified
the standard Ford contracts, but neither fresh two-sided option quotes nor the
combo increment passed. The earlier successful combo check after Gateway restart
does not establish continuing access. No order was sent by any diagnostic.

Do not call the deployment fully trade-ready. The opening check requires valid
real-time option quotes and a broker-supplied combo increment with all PAPER,
reconciliation and scanner-continuity checks passing. If error 10197 persists,
the conflicting/stale IBKR market-data session must be cleared; do not bypass
the check or substitute another data source or price increment.

## Update after disconnecting OpenAI and signing into PAPER again

The user disconnected the direct IBKR/OpenAI integration. Immediate checks still
returned 10197. Gateway's authenticated restart at 22:59 UTC stopped at login;
the user then requested a full close/reopen and manually signed into PAPER.
The full service restart occurred at 23:06:17 UTC. Stocker subsequently confirmed
PAPER DUP655399 connected and reconciled, with no orders, fills or positions,
LIVE disabled and the September 24 opening check still waiting.

Non-transmitting checks at 23:07:55 and 23:09:16 UTC did not reproduce 10197.
The latter explicitly returned the BAG price increment 0.01 and verified standard
contracts. Fresh real-time two-sided option quotes remained unavailable; entry
authority stays conditional on the opening check. This improves the observed
status but does not prove the original cause or guarantee continuing access.
The latest result is [recorded here](first4-openai-disconnect-verification.json)
and in the runtime's `paper_execution_verification` audit record.

Restoring the usual 23:45 UTC Gateway restart time is pending. The temporary
authenticated-restart setting was 22:59 UTC. Reopening the stale Screen Sharing
view requested VNC authentication; no further Gateway setting was changed while
that authentication was unavailable.
