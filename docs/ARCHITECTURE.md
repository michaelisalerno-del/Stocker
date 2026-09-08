# Stocker architecture

## Market → Method → Run

A Method owns how it finds, qualifies, vetoes, enters, manages and exits trades.
The user selects the market and method; the method determines what stocks are appropriate.

The current catalogue exposes one method, **Session HARD**, across Stocker's market catalogue.
Its stable internal identity is `SESSION_HARD_HV_HIGH_PRE_MOVE_DOWN_STRUCTURE_D`;
its current version is `SESSION_HARD_CAUSAL_Q1_FIT_V1`. The old name remains an internal
identity for history continuity, not a second selectable strategy. The method remains PAPER-only.
Non-US markets are labelled **unvalidated cross-market PAPER tests**. Operational calendar/scanner
support is not validation of Session HARD, MODEL_T0 or the FIT-derived cutoff on those markets.

```text
Market → Method → saved Run
                    ↓
 method-owned listing universe / required-data screening
                    ↓
 suitability diagnostics → T0 qualification → MODEL_T0 Q1 veto
                    ↓
 arm P0 ± 0.20M → first actual break chooses LONG or SHORT
                    ↓
 shared account checks → protected entry → method stop / target / deadline
                    ↓
 broker-authoritative fills and positions → persisted lifecycle
```

There is no user-selected cap bucket, activity screen, strategy ranking, or generic exit override
in the current run builder. Market cap remains metadata and a research diagnostic. No exploratory
midcap, volatility or liquidity finding has become an admission filter.

## Small method boundary

- `stocker_core.methods.MethodDefinition` supplies identity/version, supported markets,
  execution environments, a specification builder and an explicit universe builder.
- The serializable specification describes search, data/history/warmup, suitability,
  qualification, vetoes, direction, capacity, entry, exits, economics, shortability,
  sessions, runtime state and artifact provenance. Its canonical JSON SHA-256 is saved per run.
- `stocker_execution.strategy_factory` is the explicit composition seam. It associates the
  method identity/version with its decision engine and `MethodServices`: feature producer,
  context producer, event source, checkpoint schedule and universe qualification function.
- The engine evaluates candidates, observes entry events, restores/saves its additional state
  and advances background state. Session HARD reuses frozen qualification calculations from
  `session_hard_structure_d.py`; that historical class is not an installed engine.
- Shared execution consumes intents with side, entry reference, absolute stop/target and deadline.
  Historical M-distance geometry remains readable. Future methods can supply their own prices.
  Broker objects stay inside the adapter.

To add a future method, provide these catalogue and composition entries, its specification,
universe/data services and engine, plus focused behavioral tests. The dashboard reads the catalogue;
the scheduler uses the supplied checkpoints and services. Do not add method-name branches to the UI
or shared runtime. Different methods need not share Session HARD's features, vetoes or entry logic.
There is no plugin loader, factory hierarchy or dependency-injection container.

## Session HARD package

### Universe, data and suitability

For US markets the universe builder resolves authoritative NASDAQ/NYSE/US_ALL listing
membership. The existing Nasdaq Trader snapshot records source URLs, retrieval/file times and
non-ETF/non-test issues. IBKR qualification proves tradable STK identity and reports failures
per symbol. A run embeds the actual membership snapshot, so refreshing the catalogue does not
silently change that run. Refresh membership explicitly with
`stocker universe refresh-us-listings` before creating a new run.

For other markets, `session_hard_universe.py` reuses the existing Activity Shortlist V1 profile
as an explicitly unvalidated PAPER test input: the existing scanner components, deterministic
ranking, 50-stock limit, 15 active-minute capture and one-minute capture window. No cap restriction
is applied. This bounded activity population is not a complete exchange listing or a validated
Session HARD suitability filter. The existing per-market/session snapshot preserves the discovery
population across reloads; a missed capture stays `SCREEN_MISSED` until the next session.
The frozen qualification, MODEL_T0 preprocessing/score, inclusive FIT cutoff and entry/exit geometry
are identical across markets. No model or trading thresholds are re-fit for transfer.

The dashboard starts runs with `POST /api/universe-runs/paper?background=true`, receiving `202`
while broker connection and qualification continue. `/api/universe-runs/start-status` exposes
starting/completed/failed status across page reloads. The start button disables immediately,
duplicate submissions reuse the pending operation and failures remain visible. The runtime saves
the run configuration after application; an unfinished start interrupted by a server restart must
be submitted again. Existing synchronous API callers remain supported.
The first new run reconciles the account and updates shared readiness before it can trade.

An explicit/manual basket remains a research or test input, not the live builder's dependency.
There is no newly validated stock-suitability rule beyond universe eligibility and required data.
Cap, volatility and liquidity relationships are **research-only diagnostics**.

IBKR is the exclusive production/PAPER history source; the cache stores IBKR bars with exact
conId, timestamp, bar size, TRADES and RTH semantics. No vendor substitution or interpolation occurs.
Session HARD uses 21 consecutive prior exchange-session final RTH one-minute closes:
20 log returns, sample standard deviation (`ddof=1`), annualized by `sqrt(252)`.
Generic tick 104 is retained for historical diagnostics but cannot replace this frozen input.

The existing expected-move arithmetic uses market active regular minutes:
`M = P0 × HV × sqrt(15 / (252 × regular_minutes)) × 0.67448975`.
T0 qualification aggregates the exact completed one-minute prefix into five-minute bars.
MODEL_T0's four directional predictors use only the prefix ending T0−1 minute.
Missing exact history rejects the affected candidate; it is not estimated.

### Qualification, veto and causal entry

The reused frozen Model B score qualifies at `score >= 0.999361477`;
PRE_MOVE must be strictly greater than `0.475764059845861`.
Checkpoints remain completed five-minute counts 6, 8, …, 34, relative to the market session.
Existing exchange holidays, timezone/DST and break handling remain in the calendar layer.

The original prior-session HIGH_PRE_MOVE_DOWN_ONLY cohort predictor is maintained independently
of prospective Q1. It uses the previous 20 qualifying session dates, at least 30 observations,
the original percentile formula and NON_MID veto. Missing percentile retains the frozen missing
semantics. Historical completed-bar labels update only this later-session predictor, never entry.
Persisting a cohort update removes its in-memory duplicate.

The exact saved MODEL_T0 pipeline is loaded only after verifying artifact bytes. Its 19 input
columns, nonfinite-to-missing preprocessing, FIT imputer, scaler and logistic model are reused.
There is no live fit, recalibration, assessment ranking or silent veto bypass.

Only qualified, non-vetoed candidates arm UP=`P0+0.20M`, DOWN=`P0−0.20M`.
Ordered actual TRADES events choose LONG on the first upper break or SHORT on the first lower break.
Qualification and Q1 must already be available. If the first break preceded arming, the candidate
expires. A missing event prefix after disconnect/restart cannot be reconstructed from OHLC.
Later opposite movement never changes an established side or retrospectively cancels an entry.

The entry window is `[T0, T0+5 minutes)`, including when no new print arrives.
An untriggered candidate expires as `ENTRY_WINDOW_EXPIRED` at the boundary.
Entry references the frozen threshold, independently of the actual broker fill:

| Direction | Trigger / entry reference | Method stop | Method target |
|---|---|---|---|
| LONG | P0 + 0.20M | reference − 0.50M | reference + 1.00M |
| SHORT | P0 − 0.20M | reference + 0.50M | reference − 1.00M |

Nominal 1R is 0.50M and the deadline is always original T0+15. A first break at
T0+3 leaves twelve minutes until that deadline, not fifteen.
The package emits explicit prices/deadline; account settings cannot replace those exits.
The old TOP5 filter and pooled historical payoff hurdle are historical-only.

### Prospective Q1 freeze and development reconciliation

The old research definition ranks the complete assessment by ascending MODEL_T0 score, then
signal_id, and splits equal-count quintiles: Q1 was 225/1,121. It is not a prospective rule.

The separate frozen prospective specification uses only the original FIT score distribution:
931 candidates, 2025-05-19 through 2025-06-30. NumPy 2.4.6 `numpy.quantile`, float64,
`q=0.20`, `method="linear"` produces **0.22995371253992852**.
Admission is `whipsaw_risk_score <= q1_risk_cutoff`; equality is admitted.
There is no live ranking or signal_id tie-break.

Files under `stocker_core/method_artifacts/session_hard/` preserve the exact model,
parameters/preprocessing, source research specification and FIT score distribution. The full
prospective specification SHA-256 is
`de2c4b3b90e9ecfff44cc7da971b2d9700eb277ece423bab6a4ed8e1fb3b6965`.
MODEL_T0 SHA-256 is
`68c0f5ebf4a23744a171e32a32a8b336e8d832013692913ca034f744a88e8bc2`.
The specification records all other hashes, quantile implementation and inclusive tie policy.

`scripts/freeze_session_hard_q1.py` separates freeze and reconciliation; freeze refuses overwrite.
No protected stocks or FINAL_TIME_HOLDOUT were inspected. No threshold search or refit occurred.

The development-only reconciliation in `session_hard_q1_reconciliation.json` reports:

| Population / measure | Result |
|---|---:|
| FIT admitted | 187/931 (20.0859%) |
| Assessment admitted | 157/1,121 (14.0054%) |
| Overlap with old assessment Q1 | 157/225; 68 removed, 0 added |
| Assessment whipsaw | 17/157 (10.8280%) |
| Fixed first-break total net R bounds | +10.0743 to +28.0501 |
| Fixed first-break mean net R bounds | +0.06417 to +0.17866 |

The count change is material: the prospective population is about 30.2% smaller than old Q1.
The cutoff was not adjusted. Economic bounds use the already frozen pre-tick-overlay outcomes,
10bps round-trip cost, entry ±0.20M, stop 0.50M, target 1.00M, original T0+15 deadline.
These are operationalisation evidence, **not untouched validation** or a LIVE promotion.

### Execution and account controls

Shared Stage 7 retains explicit environment/account routing, equity/exposure/position limits,
permissions, quote/signal freshness, duplicate reservation and reconciliation.
SHORT additionally requires current sufficient IBKR shortable quantity.
The protected limit entry and stop/target children mirror LONG and SHORT.
A broker time-conditioned market child implements the deadline; stop, target and timeout share
OCA reduction for remaining quantity. Stopping a run stops future entries and does not flatten
exposure or remove protective orders.

The existing close path is installed with the bracket: the parent-linked TIMEOUT market order
has an IBKR `TimeCondition(isMore=True)` at the original deadline. It shares OCA type 2 with
the GTC stop/target, reducing/cancelling remaining siblings as exits fill. It does not depend on
the application's poll loop or a timer restarted at entry. Its fill records `METHOD_DEADLINE`;
stop and target fills record `STOP` and `TARGET`. There is no second runtime flatten mechanism.
Broker acceptance/fills remain authoritative; reconciliation reports missing protection explicitly.

Stage 7 already rounds broker prices to the instrument tick grid (LONG stop up / target down;
SHORT stop down / target up). This execution rule is unchanged. Exact `method_stop_price` and
`method_target_price` are now stored alongside these submitted `stop_price` and `target_price`
values. Neither pair is recentered after a fill. For P0=100, M=1, LONG reference=100.20,
a fill at 100.23 leaves method stop=99.70 and target=101.20.

The ledger retains `entry_reference` separately from `average_fill_price` (exposed as
`actual_fill_price`), exit fills and commissions. Diagnostics derive signed entry slippage,
fill-relative stop distance and reward remaining from those persisted prices. Method-reference R
is the directional exit-price move from the frozen reference divided by nominal risk; execution R
is actual realised P/L divided by filled quantity times the same nominal risk. Neither uses a
fill-redefined 1R. The research 10bps assumption never changes broker prices or actual execution P/L.
Late commission reports enrich an existing execution without adding quantity or another fill count;
`commissions_complete` distinguishes final reports from P/L that includes only costs received so far.
Order/position details expose these diagnostics; trade rows expose execution R and exit reason.

TRADES subscriptions start before T0 and share the existing market-data budget. Unavailable
capacity is observable per-symbol failure; the method does not invent a different stock filter.
Consumed/expired streams are released, including before quote/borrow admission where possible.
Broker disconnect protection and exact account verification remain shared boundaries.

## Runs, persistence and recovery

A saved run contains run_id, market, method identity/version/spec/hash, method-generated search
configuration and universe snapshot, environment, account risk configuration and session window.
SQLite `method_runs` records start/update times, status and stop reason; configuration revisions
retain risk/capacity edits. The audit view includes source/mode, universe count, candidate,
screened/qualified/vetoed/armed/triggered counts, sessions, errors and active/completed plan IDs.
Candidate count counts recorded opportunities; universe count counts saved listing members.

Signals retain inputs, risk score, Q1 result, armed time, first-break side, event cursor,
absolute exits, deadline and run/method/artifact provenance. Checkpoints are reserved durably.
Reload restores candidate and cohort state; it never replays a missed live first break.
The execution ledger reserves a unique signal before transmission, stores fills once and
reconciles broker-authoritative orders/positions on every connection epoch.

ARMED recovery restores P0/M, both triggers, arming time, event cursor and window. A missing
TRADES prefix expires explicitly rather than inferring which threshold broke during downtime.
A saved first break retains its side, time, threshold reference, absolute geometry and deadline;
later opposite prints cannot modify it. Pending/open execution recovers persisted broker leg IDs,
including TIMEOUT, actual fills and the original deadline. Reservation prevents resubmission;
reconciliation ingests a deadline exit that occurred while disconnected without extending the clock.

The September 2026 exit audit found existing causal first-break and fill-independent geometry
correct. It corrected no-print expiry at exactly T0+5, recognition of TIMEOUT/closed-order callbacks,
missing timeout identity during recovery, and missing exit/commission/R diagnostics.
The execution diagnostics migration adds nullable
method-price, exit-reason and commission fields plus a completeness flag; existing rows are kept,
and unknown historical method prices remain unknown. No model, Q1 cutoff, capacity rule or shared
account protection was changed. Focused tests use in-memory broker clients and temporary SQLite
files; they verify the emitted timed-close contract without placing venue orders.

Migration is additive: new method tables and nullable execution provenance/deadline/timeout
columns; old rows and research artifacts are retained. Old signals deserialize with absent new
fields. Legacy configuration enums and old calculation/payoff/scanner sources remain solely to
read history and reproduce research. Old methods cannot start through runtime, API or controls.
There is no destructive reset or conversion of old decisions into the new method.

Retired runs can be marked `archived: true` with `enabled: false`. They disappear from operational
run lists while remaining available to historical trades, orders and candidate details. Archived
runs cannot be enabled; archiving never deletes ledger rows or changes broker orders.

The FastAPI/vanilla-JS dashboard is a consumer/controller of these boundaries. Market, Method
and Start PAPER run are primary. Account risk/capacity and detailed method provenance are
expandable. Standalone dashboard mode edits saved configuration but does not connect or trade.
Dashboard failures do not stop execution.
