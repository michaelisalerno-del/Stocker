# Futures PAPER replacement — implementation report

Implemented on `codex/futures-paper-replacement` from GitHub main
`86d4790eb142f85a75792546dd19da4512e05a9e`. **Local, unarmed, not deployed and not trading.**
The existing server, Gateway and independent scanner schedule were not modified.

The subsequent market-data/API/L2 extension is documented in
[MARKET-DATA-VALIDATION.md](MARKET-DATA-VALIDATION.md), with current budgets, measurements,
review outcomes and additional fixture screenshots. It is integrated into this same migration,
also local, unarmed and awaiting deployment.

## Replacement

Removed the retired stock strategy, broker/runtime/slot logic, stock discovery dependencies
from execution, opening checks, order-flow workers, routes, UI, startup command, configuration,
strategy-only tests and scripts. No selector, compatibility layer or importable fallback remains.
The final active code/startup/configuration search has no retired strategy references. Git history
is unchanged; 51 retired documents, screenshots and data fixtures are preserved byte-for-byte
under `research/operational-history/retired-stock-runtime`, with a preservation manifest.

Retained the existing Python/FastAPI/vanilla-JS stack, IB connectivity/request cancellation
infrastructure, authentication and proxy boundary, logging, locked installation and backups.
Neutral execution modules own a fresh SQLite WAL/FULL ledger, durable order intent, reconciliation,
recovery and broker callbacks. There is one shared broker manager; position management remains
independent of entry pause, capacity and per-market data readiness.

## Frozen authority and readiness

All rows use `CURRENT_RESEARCH_NG13_ONLY`, the copied `fixed_spec.json` and corresponding
`MARKET_TAIL_FROZEN.json`; exact source paths and SHA-256 hashes are in
`research/futures-integration-sources/manifest.json`. Version: `CLOCK60_NG13_20260927`.

| Market | Frozen option method | Entry clock | Status in delivered configuration |
|---|---|---|---|
| BTC | call_10d_0DTE, absolute delta 0.10 | 09–16 NY hourly weekdays | Visible; orders blocked |
| CL | call_10d_0DTE, absolute delta 0.10 | 09–16 NY hourly weekdays | Visible; orders blocked |
| GC | put_10d_0DTE, absolute delta 0.10 | 09–16 NY hourly weekdays | Visible; orders blocked |
| NG | put_10d_0DTE, absolute delta 0.10 | Same, excluding exact 13:00 NY | Visible; orders blocked |
| NQ | put_10d_0DTE, absolute delta 0.10 | 09–16 NY hourly weekdays | Visible; orders blocked |
| SI | put_20d_0DTE, absolute delta 0.20 | 09–16 NY hourly weekdays | Visible; orders blocked |

All six are observation-capable, subject to actual data permissions/history. None is order-ready:
the accepted sources do not approve a listed product, real-expiry adaptation and strike/delta
tolerance. The concrete default reason is `LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED`.
The implementation does not invent these missing rules. [The rulebook](FUTURES-RULEBOOK.md)
records each source, completed-bar/rollover rule, finite eligibility gate, entry/exit timing,
pricing, retries and explicit execution adaptations. No strategy optimisation was performed.

Every exit remains anchored to the **original opportunity +60 minutes**, not actual fill time.
GC has no G2/GC16/volume veto or experimental exit. Relevant completed OHLCV/range/volume
observations are retained separately; the card says “Experimental management disabled”.

BTC monitors contract calendars through weekends without extending weekday entry authority.
It uses broker trading hours and explicit exchange trade dates when available, with maintenance
and unknown-calendar states. BTC/MBT 16:00 London and BFF 16:00 New York cutoffs are separate;
UK/US DST differences are tested. No listed same-day match means `NO_REAL_0DTE_MATCH`; an exit
that cannot precede cutoff plus the safety buffer is unsupported. No synthetic 17:00 or product
substitution is allowed. Actual BTC contract/calendar/quote readiness remains a deployed preflight
requirement, not an inference from historical research.

## Broker execution, limits and evidence

The actual order path builds IBKR transmitting LMT/GTD FOP orders for exact allowlisted account
`DUP655399`, after broker-returned account verification and reconciliation. LIVE has no route,
endpoint fallback or configuration value. Offline fake-broker tests exercise this same path.

Admission requires exactly one contract and at most £10 for full premium debit plus conservative
fees, using verified product units/multiplier/ticks and fresh GBP FX. It rechecks FX at admission.
Unaffordable selection is `SKIP_BUDGET_TOO_SMALL`; no cheaper strike/expiry/product is substituted.
An atomic transaction reserves £10 and one slot, capped at four trades and £40 concurrently.
Pending, uncertain and closing exposure retains reservations. Confirmed closures free slots,
including one closed trade while another still owns the same contract. £40 is not a daily-loss cap.

Signals, eligibility, durable submission, fills, open exposure, closures and skips are distinct.
Decision evidence includes original clock, inputs, capacity, actual contracts, quotes, FX, fees,
order references, fills and commissions. Historical synthetic returns are excluded. Missing
commissions/FX leave realised P&L provisional; stale quotes leave unrealised estimates unavailable.

## Dashboard and verification

Six permanent cards, compact account/allocation/P&L strip, Trades & signals filters and System
diagnostics replace the stock homepage. Updates preserve DOM identity, keyboard focus, expanded
details, filters, sort and both scroll axes. Signal and fill markers are separate.

These screenshots are actual browser renders using **labelled offline fixtures**, not broker trades:

- [Desktop overview](futures-screenshots/overview-desktop-fixture.png)
- [Mobile overview](futures-screenshots/overview-mobile-fixture.png)
- [Trades and signal history](futures-screenshots/history-desktop-fixture.png)

Validation: **422 Python tests passed**, full Ruff formatting/lint, mypy on 108 source files,
Playwright desktop/mobile and refresh-state checks, and an isolated locked server-only installation
and offline startup smoke. Five existing warnings concern Starlette/httpx deprecation and empty
research aggregates. Independent specification and standards reviews identified safety corrections
that were implemented with regression fixtures.

## Remaining cutover and arming

A direct read-only broker snapshot at **2026-09-27 09:20:13.033138 UTC** returned only account
`DUP655399`, no open orders, no nonzero positions and no returned executions. This is a point-in-time
check, not permission to cut over later. See [current status](CURRENT-DEPLOYMENT.md).

No test orders were placed. Actual FOP acknowledgements, simulated fills, commissions, cancellations
and exercise behavior remain unverified against the live broker connection; their failure paths
were tested offline. No claim is made about live-market fill quality.

The [deployment workflow](DEPLOYMENT.md) and non-transmitting preflight script are prepared.
Remaining requirements are review/approval of coordinated cutover, a fresh legacy-exposure check,
approved real listed mappings and operational preflight, then separate explicit arming approval.
Remaining account market-data allocation must also be verified and recorded; the default does
not invent an available-line allowance, and missing authority blocks new entries.
Existing exposure must retain its manager until an explicit safe resolution. The delivered example
stays `armed: false` with empty mappings. No unavailable manual login was encountered during the
read-only snapshot; future Gateway login and market-data permissions must be checked at cutover.
