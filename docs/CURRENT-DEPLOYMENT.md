# Deployed futures PAPER application

SLRNO is deployed at [the authenticated dashboard](https://139.59.178.164).
Execution remains **unarmed**, optional L2 **disabled**, and LIVE unavailable.

Runtime release: `3ac8cf32afcc78d602bf1041d79b72fa97e99212`.
Branch: `codex/futures-paper-replacement`.
The first futures cutover was 2026-09-27 15:22:14 UTC. The latest broker chain-ID correction
started at **2026-09-27 17:28:32 UTC (18:28:32 Europe/London)**. The old runtime is no longer active.

The latest recorded postflight is **2026-09-27 17:31:23 UTC**; see the
[actual server evidence](btc-chain-fix-20260927/postflight.json). This is a point-in-time observation,
not a guarantee of later positions, data or configuration.

## Verified runtime state

- Broker-returned account exactly `DUP655399`; connected and reconciled.
- Fresh read-only preflight immediately before each cutover. Final preflight:
  17:28:32.677419 UTC, zero open orders, nonzero positions or returned executions.
- Fresh futures namespace: `/var/lib/stocker/v1/futures.sqlite3`.
  Zero signals, reservations, orders, fills or positions at postflight.
- All six permanent cards and actual standard futures qualified. BTC symbol is IB `BRR`,
  trading class BTC, multiplier 5. Silver is class SI, multiplier 5000; micro SIL is excluded.
- 13 app-owned lines: six quotes, six one-minute bar streams and one FX quote; app ceiling 60.
  Zero temporary option quotes, L2 books or tick-by-tick feeds.
- Account allowance 100 remains **ASSUMED**, external consumption unknown, headroom 40.
  Wire cap 40 requests/second, ten reserved for urgent work; observed queue high water one.
- App active/running, PID 1441785, zero automatic restarts, about 62.4 MiB memory at postflight.
- Authenticated overview/history/system/health/pages/assets returned 200; public HTTPS without
  authentication returned 401. Direct requests without the proxy token and foreign-origin
  reads/writes returned 403. Retired write route returned 405.
- Searches of active application source and server startup/proxy wiring found no retired
  strategy references. Immutable historical releases and audit records remain outside the active runtime.
- IB Gateway and the independent scanner collector/timer retained their original process/status
  and configuration hashes; they were not restarted or modified.

## Market readiness at postflight

| Market | Monitored contract | Market/data | Additional readiness block |
|---|---|---|---|
| BTC | BTCV6, conId 876880607 | OPEN, current Sunday quotes/bars; five reference sessions loaded | Listed option mapping unapproved |
| CL | CLX6, conId 304037511 | CLOSED; five reference sessions loaded | Listed option mapping unapproved |
| GC | GCZ6, conId 462941472 | CLOSED; five reference sessions loaded | Listed option mapping unapproved |
| NG | NGX26, conId 269460170 | CLOSED; five reference sessions loaded | Listed option mapping unapproved |
| NQ | NQZ6, conId 563947726 | CLOSED; underlying quotes/bars subscribed | Required prior volume for historical reference selection unavailable |
| SI | SIZ6, conId 535526329 | CLOSED; five reference sessions loaded | Listed option mapping unapproved |

All six also remain subject to the shared unarmed configuration, empty approved option mappings
and unverified allowance. No trade is authorised by the above monitoring state. Closed-market
prices are labelled stale rather than described as live.

GC experimental management remains disabled. BTC uses its actual broker calendar, including
weekend intervals and maintenance. All intervals of one exchange trade date must finish before
that day's volume can be used for rollover; the completed Friday/Saturday fragment cannot turn
still-forming Monday volume into a prior completed day.

## Deployment work and validation

The latest [BTC chain-ID correction](btc-chain-fix-20260927/README.md) normalizes validated
text IDs at the broker callback boundary before strict underlying matching. Seven regressions
failed before the fix and passed afterward; 76 focused tests, Ruff, Mypy and installed startup
checks passed. A non-transmitting probe verified the real chain now matches. Raw broker metadata
still contains conflicting BTC future/option expiry clocks, and the bounded weekly search did
not establish a real same-day mapping. Neither issue was overridden. The protected pre-update
database is `chain-id-fix-20260927T172832Z.sqlite3` in the backup directory below.
The [prior history-repair snapshot](btc-history-fix-20260927.json) remains available.


Installed the locked server-only environment, preserving authenticated loopback access and
systemd hardening. Changed the app command to `futures-run` and the Caddy write allowlist to
`/api/entries/pause` and `/api/entries/resume`. Only the app was restarted; Caddy was reloaded.

Deployment checks found and corrected three integration issues before final verification:
IB's Bitcoin symbol, ambiguous standard/micro silver metadata, and BTC's split weekend trade
date. Reference-cache identity was versioned; prior observations were retained and are not reused.
Missing reference data now blocks strategy readiness while the verified current underlying
continues monitoring.

The subsequent BTC history repair replaced a ten-day minute-bar request ending at the current
time with five exact 08:00–17:00 America/New_York reference windows. The original expired BTCU6
request timed out at 15 seconds even with includeExpired enabled. The corrected non-transmitting
probe returned all 540 bars per window in 0.205–0.439 seconds, selecting BTCU6 for September 21–24
and BTCV6 for September 25 using the unchanged frozen rollover calculation. The deployed app
then reported five references and current BTC data. No threshold, entry window, exit anchor,
data requirement or request timeout was relaxed. Existing same-day reference summaries remain cached.
NQ's separate prior-volume history block remains unresolved.

This repair passed **67 focused tests**, including the regression demonstrated failing before
the fix and exact reference-summary/contract-selection/cache parity. Three CI smoke tests, Ruff,
Mypy and installed server-only startup/assets checks passed. Fresh broker preflight was flat;
the app-only restart preserved the ledger and left Gateway/scanner processes unchanged.
The protected pre-repair database is
`btc-history-fix-20260927T163804Z.sqlite3` in the backup directory below.
The earlier [initial deployment snapshot](futures-deployment-20260927.json) is retained as history.


The final corrections passed **61 focused tests**, Ruff lint/format and Mypy over the execution
package; three CI smoke tests also passed. Each installed release passed server-only offline
startup/assets checks as the service user. The preceding migration/data extension had 441
passing tests and browser fixture checks; those are historical validation, not a claim that
broker fills have been tested.

Protected consistent SQLite/config/unit/proxy backups:
`/var/lib/stocker/backups/futures-cutover-20260927T152213Z`.
SQLite integrity checks passed. Original and intermediate databases were retained, not overwritten.
No test trade, cancellation, subscription purchase or account-permission change was performed.

## Remaining arming requirements

Approve concrete listed option product/expiry/delta-tolerance mappings, verify account-wide
data allowance/entitlements and external headroom, resolve required history readiness, then
perform a separate reviewed arming operation. One contract, £10 including fees, four concurrent
reserved/open trades and £40 concurrent allocation remain enforced. L2 permission/routing must
be verified before enabling optional collection.

Actual FOP acknowledgements, fills, commissions, cancellation races and exercise handling remain
offline-tested, not verified through transmitting broker orders. PAPER fills, when authorised,
will be broker simulations. See [deployment procedure](DEPLOYMENT.md) and
[market-data operating behaviour](MARKET-DATA.md).
