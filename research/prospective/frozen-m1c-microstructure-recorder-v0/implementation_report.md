# Implementation report

Implemented as an extension of the existing Python/FastAPI/SQLite/vanilla-web stack.

## Frozen inference

- Exact M1C manifest, preprocessing, coefficients, stock controls, checkpoint controls, threshold, and fresh crossing/30-minute spacing are loaded without fitting.
- Exact A1/C1/R1 stock-local normalisation, fallbacks, beta parameters, coefficients, and OOF confidence boundaries are loaded without fitting.
- Direction inputs are built through T-1 and reject trigger-bar leakage.
- D-1 Group O packages use an explicit signal-session path and reject missing, same-day, stale, future, or unauthorised contexts.

## Market data and persistence

- The official market-data-only IBKR facade exposes Level I, BidAsk/Last tick-by-tick, SMART depth, five-minute historical updates, option computation, exact contract qualification, clock, and depth-exchange callbacks.
- One subscription manager tracks capacity, request rate, priority, ownership, errors, denials, starts, and cancellations. Universe Level I and active episodes are protected.
- Every callback is assigned a deterministic request owner and converted to an immutable activation-bounded raw event.
- Raw event fragments are atomically finalised as Zstandard Parquet with content hashes and claims metadata. SQLite stores bounded projections and audit metadata.
- Lost-data reconnects mark gaps and rebuild underlying and active option streams. Maintained-data reconnects do not invent a gap. Depth reset is retained as a raw event, invalidates and empties the book, and replaces the exact depth subscription.

## Signals, microstructure, and options

- Bars are scored only after explicit completion by the next `keepUpToDate` bar.
- Continuous and episode-relative quote/trade summaries include quote flow, probable trade flow, impact, 1/3/5-second replenishment proxies, periodic depth snapshots, depth primitives, MC/MD/MA/MB descriptive scores, and separate frozen-archetype relationship fields.
- Option discovery is limited to nearest valid 0DTE, 1DTE, and 3–5 DTE expiries, common call/put strikes, ATM plus configured symmetric wings, and the remaining option capacity.
- Raw option top-of-book and computation updates are retained. The frozen ledger uses first valid ask after entry and last valid bid at/before 5/10/15/30-minute horizons, with first-bid-after sensitivity. ATM straddles and the hidden-from-live-panel retrospective oracle are persisted separately.
- A chronological immutable phase ledger assigns complete episodes to 30 engineering-shakedown, 100 development, and 100 unopened confirmation observations. End-of-session quality reports are atomically written and mirrored into SQLite.

## Web and safety

- Read-only status, capabilities, universe, episode, microstructure, options, shadow ledger, audit, session-report, and broker-isolated replay endpoints are exposed.
- The dashboard permanently displays `RECORD ONLY — ORDER ROUTING DISABLED`.
- No order, account, position, buy, sell, or trade route exists. The recorder imports no order-placement class and the audit finds no forbidden broker callable.
- Replay controls read persisted SQLite/Parquet evidence, verify partition hashes, preserve recorded event order, and never construct an IBKR adapter.

## Verification

- M1C parity passed on 250 rows: maximum feature difference `0`, maximum probability difference `2.220446049250313e-16`, and no threshold-membership mismatch.
- A1, C1, and R1 parity passed on all 417 stored assessment episodes per model. Action mismatches were zero; the largest probability difference was `9.992007221626409e-16`.
- The independent reconstruction audit passed: 100 M1C probabilities, 100 rows per archetype, 100 microstructure windows, 50 ask-to-bid shadow outcomes, and two identical 200-event replays.
- The 72 focused recorder/web tests and all 165 non-skipped prospective tests passed. The full repository collected 1,262 tests; 1,229 passed and one skipped. Thirteen failures and nineteen setup errors are confined to pre-existing SLRNO integration tests whose frozen `trade_decisions.parquet` input is absent.
- Scoped Ruff formatting/lint and strict mypy passed. The source distribution and production wheel were built successfully.
- The credential-free fake adapter completed a full application startup/poll/shutdown smoke test with 20 stocks plus VTI and 42 protected always-on streams.
- This repository has no TypeScript package or `package.json`; the existing frontend is static HTML/CSS/JavaScript, so no TypeScript check or component-runner command exists. FastAPI component/API tests and the production package build cover the current stack.

## Activation blocker

This repository does not contain live IBKR entitlements, Gateway/TWS, credentials, a production bar-compatibility report, a production prior-session activity baseline, or a session-specific signed Group O package. Consequently no real prospective session was activated during implementation. The runtime records diagnostic delayed/frozen data but marks it scientifically invalid.
