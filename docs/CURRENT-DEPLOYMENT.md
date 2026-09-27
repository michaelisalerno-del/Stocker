# Deployed futures PAPER application

SLRNO is deployed at [the authenticated dashboard](https://139.59.178.164).
Execution remains **unarmed**, optional L2 **disabled**, and LIVE unavailable.

Runtime release: `a1cc1dac652b94eafeb8b0aab7ac1c43635ee8db`.
Branch: `codex/futures-paper-replacement`.
The first futures cutover was 2026-09-27 15:22:14 UTC. The final corrected release started at
**2026-09-27 16:17:37 UTC (17:17:37 Europe/London)**. The old runtime is no longer active.

The latest recorded postflight is **2026-09-27 16:18:24 UTC**; see the
[actual server evidence](futures-deployment-20260927.json). This is a point-in-time observation,
not a guarantee of later positions, data or configuration.

## Verified runtime state

- Broker-returned account exactly `DUP655399`; connected and reconciled.
- Fresh read-only preflight immediately before each cutover. Final preflight:
  16:17:36.765279 UTC, zero open orders, nonzero positions or returned executions.
- Fresh futures namespace: `/var/lib/stocker/v1/futures.sqlite3`.
  Zero signals, reservations, orders, fills or positions at postflight.
- All six permanent cards and actual standard futures qualified. BTC symbol is IB `BRR`,
  trading class BTC, multiplier 5. Silver is class SI, multiplier 5000; micro SIL is excluded.
- 13 app-owned lines: six quotes, six one-minute bar streams and one FX quote; app ceiling 60.
  Zero temporary option quotes, L2 books or tick-by-tick feeds.
- Account allowance 100 remains **ASSUMED**, external consumption unknown, headroom 40.
  Wire cap 40 requests/second, ten reserved for urgent work; observed queue high water one.
- App active/running, PID 1435627, zero automatic restarts, about 67 MiB memory at postflight.
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
| BTC | BTCV6, conId 876880607 | OPEN, current Sunday quotes and bars | Reference history request timed out; bounded retry remains active |
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

Installed the locked server-only environment, preserving authenticated loopback access and
systemd hardening. Changed the app command to `futures-run` and the Caddy write allowlist to
`/api/entries/pause` and `/api/entries/resume`. Only the app was restarted; Caddy was reloaded.

Deployment checks found and corrected three integration issues before final verification:
IB's Bitcoin symbol, ambiguous standard/micro silver metadata, and BTC's split weekend trade
date. Reference-cache identity was versioned; prior observations were retained and are not reused.
Missing reference data now blocks strategy readiness while the verified current underlying
continues monitoring.

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
