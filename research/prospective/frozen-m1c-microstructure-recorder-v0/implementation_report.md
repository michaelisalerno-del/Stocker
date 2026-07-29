# Implementation report

Implemented as an extension of the existing Python/FastAPI/SQLite/vanilla-web stack.

## Frozen inference

- Exact M1C manifest, preprocessing, coefficients, stock controls, checkpoint controls, threshold, and fresh crossing/30-minute spacing are loaded without fitting. Startup now fails closed unless the manifest, threshold, and causal scaling files match the three preregistered SHA-256 identities.
- The quiet-state extension uses that same causal M1C artifact, whose frozen manifest hash is `6f59177a58973d33a24741e3c265e1831bfb6dc07afac17ae371501019bdc5cc`. It records inclusive bottom-5 (`0.115697407847643`), bottom-10 (`0.135896965695626`), and bottom-20 (`0.167095528962669`) flags without fitting, calibration, or threshold changes.
- A quiet episode begins only on a bottom-10 downward crossing (or the first eligible checkpoint) and is spaced at least 30 minutes from the prior same-stock, same-session quiet episode. Its prospective entry is the next completed five-minute bar boundary and remains a research timestamp.
- Neutral controls come only from the frozen interval above bottom-20 and below high-tail. A SHA-256 mapping of the frozen salt, session, symbol, checkpoint, and model hash selects a fixed 10% sample. Every existing fresh high-tail episode remains a control.
- Exact A1/C1/R1 stock-local normalisation, fallbacks, beta parameters, coefficients, and OOF confidence boundaries are loaded without fitting.
- Direction inputs are built through T-1 and reject trigger-bar leakage.
- D-1 Group O packages use an explicit signal-session path and reject missing, same-day, stale, future, or unauthorised contexts.

## Market data and persistence

- The official market-data-only IBKR facade exposes Level I, BidAsk/Last tick-by-tick, SMART depth, five-minute historical updates, option computation, exact contract qualification, clock, and depth-exchange callbacks.
- One subscription manager tracks capacity, request rate, priority, ownership, errors, denials, starts, and cancellations. Universe Level I and active episodes are protected. Scientific capability preflight requires live Level-I evidence for all 20 stocks, VTI, and every frozen sector proxy.
- Every callback is assigned a deterministic request owner and converted to an immutable activation-bounded raw event.
- Raw event fragments are atomically finalised as Zstandard Parquet with content hashes and claims metadata. SQLite stores bounded projections and audit metadata.
- Lost-data reconnects mark gaps and rebuild underlying and active option streams. Maintained-data reconnects do not invent a gap. Depth reset is retained as a raw event, invalidates and empties the book, and replaces the exact depth subscription.

## Signals, microstructure, and options

- Bars are scored only after explicit completion by the next `keepUpToDate` bar.
- Continuous and episode-relative evidence retains completed bars, underlying Level I, sizes, last trades, spread, microprice, imbalance, raw tick-by-tick events, bounded depth, clocks, subscriptions, and data gaps. VTI plus the frozen seven-ETF sector-proxy panel are protected Level-I/five-minute streams; quiet, neutral, and high-tail path records embed both the market and stock-specific sector path at 5/10/15/30/60 minutes.
- Option discovery is limited to valid 0DTE, 1DTE, and 3–5 DTE expiries, common call/put strikes, ATM plus four frozen symmetric target offsets (1%, 3%, 6%, and 10%), and the remaining option capacity. The first qualified iron-fly pair is the nearest symmetric pair at least 1% away; sub-1% wings are excluded. It never substitutes an expiry outside its bucket or streams a complete chain.
- Raw option top-of-book and computation updates are retained. The frozen ledger uses first valid ask after entry and last valid bid at/before 5/10/15/30-minute horizons, with first-bid-after sensitivity. ATM straddles and the hidden-from-live-panel retrospective oracle are persisted separately.
- Quiet observations add conservative long call, long put, and long straddle outcomes plus four bounded structures: ATM iron butterfly, 25/10-delta iron condor, fixed-width call credit spread, and fixed-width put credit spread. No naked structure is representable.
- Credit structures open short legs at observed bid and protective legs at observed ask; they close short legs at observed ask and protective legs at observed bid. A zero exit bid is retained as an observed full-loss quote with a quality flag. Missing contracts, legs, horizons, and wholly omitted buckets remain explicit incomplete attempts. Touch/cross evidence comes only from retained underlying Level-I/trade events; option callback reference prices never substitute for a missing path, and retained halt evidence disqualifies strict quality. The ledger retains net credit/debit, P&L, commissions sensitivity, maximum risk, returns, breakevens, strike/wing touches, mark-path adverse/favourable P&L, and explicit quote-quality failures.
- The predecessor high-movement ledger remains 30/100/100. Quiet, neutral, and high-tail comparison observations now share the bounded 60-minute option panel while high-tail directional shadow records remain intact. A separate immutable quiet-state phase ledger assigns only episodes with an unreduced, complete requested-contract-by-horizon matrix to 30 engineering-shakedown, 150 development, and 150 unopened confirmation observations; capacity-truncated plans cannot advance the ordinal. Neutral and high-tail controls inherit the contemporaneous quiet phase. The cohorts are never merged.

## Web and safety

- The existing read-only application adds `/api/quiet-state/status`, `/universe`, `/episodes`, `/episodes/{episode_id}`, `/episodes/{episode_id}/options`, `/shadow-structures`, `/concentration-audit`, and `/session-quality`.
- Four separate screens show the quiet-state universe, observation detail, defined-risk shadow ledger, and retrospective concentration audit. The audit screen permanently shows the original failed gate rather than recomputing or relaxing it.
- The dashboard permanently displays `RESEARCH ONLY — RECORD ONLY — NO ORDERS`.
- No order, account, position, buy, sell, or trade route exists. The recorder imports no order-placement class and the audit finds no forbidden broker callable.
- Replay controls read persisted SQLite/Parquet evidence, verify partition hashes, preserve recorded event order, and never construct an IBKR adapter.

## Verification

- M1C parity passed on 250 rows: maximum feature difference `0`, maximum probability difference `2.220446049250313e-16`, and no threshold-membership mismatch.
- A1, C1, and R1 parity passed on all 417 stored assessment episodes per model. Action mismatches were zero; the largest probability difference was `9.992007221626409e-16`.
- The independent reconstruction audit passed: 100 M1C probabilities, 100 rows per archetype, 100 microstructure windows, 50 ask-to-bid shadow outcomes, and two identical 200-event replays.
- The quiet-state independent audit passed 100 independently reconstructed predictions, every threshold and episode identity, neutral sampling, 50 iron-fly outcomes, 50 iron-condor outcomes, phase boundaries, and route/broker safety. The committed fixture now materialises all 100 prediction inputs and all 50 contract/quote/path cases; those exact stored inputs were replayed twice with zero probability, membership, episode, control, contract, leg, P&L, or floating-point mismatch.
- The retrospective concentration audit passed its independent reconstruction and determinism checks, including 100 bottom-tail rows and every clustered surprise event required by its sampling rule.
- The final combined focused audit/recorder/options/web suite passed 91 tests, including the fake-adapter application and retained-event smoke coverage.
- The repository-wide suite passed 1,262 tests with one skip. Thirteen failures and nineteen setup errors remain confined to pre-existing SLRNO integration tests whose frozen `trade_decisions.parquet` input is absent.
- Scoped Ruff formatting/lint and strict mypy passed for all 59 task source files. Repository-wide mypy still reports 138 pre-existing errors in 27 unrelated research/MCP files. The source distribution and production wheel were built successfully.
- The credential-free fake adapter completed a full application startup/poll/shutdown smoke test with 20 stocks, VTI, seven sector ETFs, and 56 protected always-on streams.
- This repository has no TypeScript package or `package.json`; the existing frontend is static HTML/CSS/JavaScript, so no TypeScript check or component-runner command exists. FastAPI component/API tests and the production package build cover the current stack.
- In-app browser QA exercised all four new views and all quiet-state API calls at desktop and a 390-pixel breakpoint. The permanent claims boundary remained visible and the narrow page had no document-level horizontal overflow.

## Activation blocker

This repository does not contain live IBKR entitlements, Gateway/TWS, credentials, a production bar-compatibility report, a production prior-session activity baseline, or a session-specific signed Group O package. Consequently no real prospective session was activated during implementation. No live market data or option outcomes were committed. Actual future bid/ask observations are still required to determine whether quiet state is useful as a long-premium veto or a defined-risk short-premium opportunity.
