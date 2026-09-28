# Saxo migration delivery — 2026-09-28

The existing SLRNO launcher, authenticated dashboard, frozen mathematics and durable execution
store now use a Saxo-only active runtime. Five active families: CL, GC, NG, NQ, SI. IBKR adapters,
configuration and obsolete provider launchers are parked as hashed historical text outside imports;
IB dependencies are removed. EODHD external transport is blocked. Original caches, provenance and
immutable trading/research evidence remain. The original source checkout was not changed.

| Stage | Evidence |
|---|---|
| IMPLEMENTED | OAuth, allowlisted clients, subscriptions, references, rolling/event recorder, paper risk, dashboard and cutover guides in this branch |
| OFFLINE_TESTED | Sanitised fixtures, provider-independent frozen/ledger regressions, browser checks and server-only installation smoke |
| AUTHENTICATED | **No** — Saxo credentials/grant unavailable |
| DATA_VERIFIED | **No** — actual account/UIC/quotes/history/session unverified |
| L2_VERIFIED | **No** — no received broker depth or entitlement confirmation |
| RECORDER_VERIFIED | **Offline fixtures/workload only**; persistent broker-data permission unverified |
| DEPLOYED | **No** — authorised SSH identity/current server inventory unavailable |

No live or SIM order was sent, no paper execution armed, no broker subscription purchased, and
no account/legal agreement changed. Tests simulate the broker boundary without transmitting.

Validation details and resource measurements: [MARKET-DATA-VALIDATION.md](MARKET-DATA-VALIDATION.md).
[Capability matrix](saxo-capabilities.json) explicitly marks every unverified field for each market.
[Source hashes](saxo-frozen-sources.json) preserve original strategy provenance.
[Setup/OAuth](SAXO-SETUP.md), [cutover/rollback](DEPLOYMENT.md), [current server evidence limits](CURRENT-DEPLOYMENT.md).

Remaining external/strategy blockers: authenticate separate SIM/LIVE apps/accounts; discover/pin
actual futures; verify actual L1/L2/option/FX/chart/session permissions; verify recording rights;
confirm prior-session volume selection audits and five reference sessions; approve listed option
product/expiry/cutoffs/delta tolerance/fees. GC remains monitor-only without a listed execution rule.
Date-only option expiry or ambiguous broker obligations block the affected decision. The code never
substitutes a synthetic expiry, guessed volume, alternative vendor or unapproved cheaper instrument.

Production cutover also needs a fresh obligation check on the current IBKR/FIRST4 managers and a
verified complete inventory of app-owned services, timers, cron and supervisor restart paths. Those
services have not been stopped based on dated evidence. The supplied guide preserves position
management and recoverable configurations; the old adapter/config remain available for controlled rollback.
