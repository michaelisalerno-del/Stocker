# PAPER arming and BTC RV15 check — 2026-09-27

The user explicitly requested arming. The existing server's PAPER switch changed from
`armed: false` to `armed: true` at 18:08:21 UTC (19:08:21 Europe/London).
This is an arming-state change, **not evidence of trading readiness or a submitted trade**.

Only that configuration field changed. The immutable code release remains
`3ac8cf32afcc78d602bf1041d79b72fa97e99212`. Repository installation defaults remain unarmed.
No product approval, strategy mathematics, £10/one-contract/four-trade/£40 limit, data allowance,
LIVE availability or L2 setting changed.

## Broker and runtime verification

Fresh preflight immediately before restart verified account `DUP655399`, no open orders or
nonzero positions, and zero returned executions. The durable ledger also had no obligations.
The old configuration and consistent SQLite backup are retained in
`/var/lib/stocker/backups/futures-arm-20260927T180821Z`; database integrity passed.
Only the application restarted. Gateway and the independent scanner/timer were unchanged.

[Postflight at 18:20:27 UTC](postflight.json) verified:

- PAPER armed, entries not paused, broker connected and reconciled.
- Shared entry block: `MARKET_DATA_ALLOCATION_UNVERIFIED`.
- No approved option mappings; all six cards still show entry disabled.
- BTC has five reference sessions; its option mapping/expiry/tolerance remains unresolved.
- NQ retains its independent prior-volume reference-history block.
- Zero ledger orders, fills, reservations and positions; no test orders were submitted.
- 13 of 60 app-owned data lines; zero temporary option quotes or L2 books.
- One historical-request cancellation (IBKR 162, request 59) remains recorded honestly;
  no entry guard was relaxed.
- Authenticated routes and unauthorized-request rejection passed. App PID 1444449,
  zero automatic restarts, approximately 63.7 MiB memory.

Arming does not approve any unresolved mapping or claim that the assumed account allowance is
verified. Changing those controls still requires concrete evidence and review.

## BTC RV15 report

The user clarified that the value appeared missing, rather than requesting removal of RV15.
No strategy change was made.

The API returned `rv15: 0.0`. A separate bounded, non-transmitting broker request at 17:39:59 UTC
returned 30 one-minute TRADES bars. The 18 overlapping stored/broker bars from 17:21–17:38 UTC
all matched: close 84,850, reported volume zero. Recomputing RV15 from the latest 16 completed
closes gave exactly zero. See [actual broker comparison](btc-rv15.json).

These are IBKR-reported values, not an inference of zero exchange-wide activity. No missing
return was filled, volatility floor added, quote substituted for a trade bar, or gate bypassed.
A positive RV remains required by the frozen method when an opportunity is evaluated.

The existing browser test was replayed with BTC RV15 set to zero. Desktop and mobile both
displayed `RV15 0.0000% · completed bars`; focus/scroll/stable-card tests also passed.
The [screenshot](rv15-zero-desktop-fixture.png) is an **offline fixture**, not a broker dashboard
screenshot. The user's exact browser session was unavailable, so a transient or cached display
issue in that session was not reproduced. No application-code fix was justified by this check.

The initial diagnostic attempt used the wrong local ledger column and stopped before its
historical request. The corrected comparison above completed. No runtime instrumentation or
background probe was installed.
