# Stocker V1 Architecture

## Purpose

Stocker V1 is a deliberately simple modular trading application. It will navigate between stock
universes, run multiple or overlapping universes, screen instruments, obtain IBKR history,
calculate PRE levels and features, qualify and rank candidates, apply risk rules, paper trade, and
eventually run LIVE alongside PAPER. A later dashboard may expose status and control.

Modularity supports real new strategies and universes. It must not become speculative framework
building.

## Primary pipeline

```text
Run Manager / Scheduler
        ↓
Universe
        ↓
Basic Screening
        ↓
IBKR Historical Data
        ↓
Local IBKR History Cache
        ↓
Prior-Session IV Context / Expected-Move Fraction
        ↓
Current-Session M / PRE_MOVE / Generic Feature Snapshot
        ↓
Strategy Cohort / Percentile / Band / Qualification
        ↓
Candidate Ranking
        ↓
Risk
        ↓
Execution
      ↙     ↘
   PAPER    LIVE
        ↓
Trade/Event Ledger
```

The dashboard is a later consumer/controller. Trading must not depend on it.

## Core concepts

### Universe

A universe defines instruments that may be considered, such as `NASDAQ`, `FTSE350`, `US_MIDCAP`,
or a custom list. It contains neither strategy nor execution logic.

Stage 3 loads immutable named definitions into `UniverseCatalog`. Membership uses normalized,
value-based `InstrumentReference` keys containing symbol, exchange context, currency, and security
type; the universe does not manufacture a universe-specific security identity. Thus overlapping
universes can share the same conceptual instrument while IBKR qualification and `conId` remain at
the Stage 2 boundary. Exact duplicate members within one universe are removed in first-seen order.

### Run

A run is one active combination of universe, strategy, `PAPER` or `LIVE` environment, trading or
session window, and relevant configuration. Runs may execute sequentially or concurrently.

```text
FTSE Morning          NASDAQ Main       US Midcap Experiment
FTSE350               NASDAQ            US_MIDCAP
SESSION_HARD          SESSION_HARD      NEW_STRATEGY
PAPER                 LIVE              PAPER
```

`RunManager` holds any number of independent in-memory `RunInstance` values in `CONFIGURED`,
`ACTIVE`, or `STOPPED` state. Multiple runs may share one immutable universe definition while
retaining separate strategy, environment, optional session window, and lifecycle state. Starting
or stopping a run only changes that run. Session windows are descriptive in Stage 3: no scheduler,
calendar processing, broker connection, bulk data request, screening, or trading starts with them.

### Historical data

IBKR is the exclusive source for historical bars used by production and paper PRE calculations.
The local cache stores only IBKR-originated history. Stocker must not silently fall back to EODHD,
FMP, TwelveData, or another provider for those calculations.

The contract-independent Stage 4 substrate is implemented in
`stocker_execution.history`. `IbkrHistoryService` accepts only the concrete Stage 2
`IbkrConnection`; the cache write operation is private to that service so arbitrary provider
bars cannot be relabelled as IBKR history. It writes validated bars to one SQLite
`IbkrHistoryCache`. Cache identity is
`conId + bar size + whatToShow + RTH mode + UTC timestamp`; it contains neither universe, run,
strategy, nor PAPER/LIVE identity. Instrument metadata and `fetched_at` are retained with each bar.
Reads name their exact required timezone-aware timestamps and an `as_of` cutoff. Missing bars
or cached rows that fail the shared historical-bar validation produce `NOT_READY` with the exact
gaps; bars are never interpolated, substituted, or silently dropped from the requirement.

#### Canonical PRE contract status

The detailed recovery evidence, backward lineage, confirmed `PRE_MOVE_M` behavior, reference rows,
and exact unblock requirements are recorded once in
[`PRE_LINEAGE_RECOVERY.md`](PRE_LINEAGE_RECOVERY.md).
The follow-up row audit and corrected causal ownership are recorded in
[`M_PRE_MOVE_AUDIT.md`](M_PRE_MOVE_AUDIT.md).

The accepted research lineage is:

- research worktree `2026-09-01-session-hard-structure-d-price-volume/`
  `rvol_efficiency_context_v0/contract.json` and `run_experiment.py` (research runner SHA-256
  `3e0c2884ff3f0277265b86d4feca447954e75a349d3d81388390007e50c47f25`), which define
  `PRE_MOVE_M` as the split-aligned absolute open-price change from exactly `T0-3 minutes` to
  `T0`, divided by upstream canonical `M`;
- Stocker research worktree `2026-08-15-you-are-working-in-my-existing/`, file
  `research/directional-readiness/20260830-session-hard-broad-universe-expansion-v0/`
  `run_broad_experiment.py`, which produces that upstream `M_price` from prior-session ATM option
  IV (`P0 * atm_iv * sqrt(15 / (252 * 390)) * sqrt(2 / pi)`), rather than from a frozen
  stock-bar-only history calculation.

The follow-up audit confirms that the research normalisation is row-specific and correct:

```text
expected_absolute_return_15m = ATM_IV * sqrt(15 / (252 * 390)) * sqrt(2 / pi)
M_price = current-session P0 at T0 * expected_absolute_return_15m
PRE_MOVE_M = abs(P0 - raw_open[T0-3m] * (P0 / raw_open[T0])) / M_price
```

The unchanged strict threshold `PRE_MOVE_M > 0.475764059845861` is downstream of this
normalisation and is never `M`.

Stage 4 is implemented in `stocker_execution.pre_context` as `PRE_CONTEXT_V1`. Its pure selector
preserves the frozen 7--45 calendar-DTE, 75%--125% strike bounds, nearest common-strike expiry,
`abs(log(strike / previous_close))`, descending minimum open interest, combined relative spread,
IV-gap, strike, and contract-ID ordering. It validates the selected pair without trying a fallback:
model IV `[0.005, 5]`, nonnegative bid, ask at least bid, positive midpoint, open interest at least
10 per leg, relative spread at most 1, and the recovered optional delta/gamma checks.

The PAPER/LIVE IV source is frozen by user decision as the contract-specific IBKR Model Option
Computation `impliedVol` delivered by `tickOptionComputation` tick type 13 (the `ib_async`
`ticker.modelGreeks.impliedVol` field). Call and put model IV are averaged. Tick types 10, 11, and
12 are not fallbacks; generic tick 106 `OPTION_IMPLIED_VOLATILITY` is excluded. Generic tick 101 is
requested only for per-leg open interest. The adapter explicitly requests live market data type 1
and accepts only returned type 1 or frozen type 2, whose model computation uses tick 13; delayed
types 3/4 (model tick 83) are rejected. A missing model computation makes the result
`PRE_CONTEXT_NOT_READY`.

The research specified five-minute regular-session traded-price context functionally. The explicit
**IBKR implementation mapping** is `5 mins`, `TRADES`, and `useRTH=True`, ending at the previous
XNYS session close. A completely absent session uses `duration=1 D`; partial gaps request only
contiguous missing five-minute ranges. The existing exchange calendar supplies normal and half-day
bounds; every scheduled five-minute bar is required and the final bar close is the selection
reference. No missing bar is filled. Acquisition qualifies every bounded strike for each expiry in
order until the first expiry with an actual common call/put strike, then snapshots only the
primary-distance strike set required for the remaining frozen tie-breakers. The canonical option
observation is always 16:00 America/New_York on the previous session date, including XNYS
half-days. An uncached request must start during that timestamp's one-minute resolution and stores
both the 16:00 observation point and actual receipt time. This is necessary because IBKR does not
provide this contract-specific model IV as a historical option-bar series.

One SQLite table persists the context by `underlying conId + target session + PRE_CONTEXT_V1` with
underlying/call/put `conId`, selected expiry/strike, both model IVs, ATM IV, expected-return fraction,
source, market-data types, and timestamps. The first valid context stored for that key is immutable;
the key has no universe, run, strategy, or PAPER/LIVE dimension. The
user-accepted EODHD/IBKR parity conclusion comes from prior empirical testing; its old numerical
artefact is unavailable and is not a blocker. Runtime inputs remain IBKR-only.

### IBKR boundary

`stocker_execution.ibkr.IbkrConnection` is the Stage 2 broker boundary for connection/session
identity, stock and option qualification, option definitions/model snapshots, historical bars,
current stock snapshots, and Stage 7 PAPER execution. Each instance owns its own
client state, so PAPER and LIVE can later use separate Gateway sessions without a global singleton.
The adapter is read-only unless PAPER execution is explicitly enabled. It exposes small Stocker
models; strategies and calculations do not import IBKR objects.

PAPER and LIVE host, port, client ID, and environment are separate explicit configuration blocks.
The Stage 1 run environment selects one block. Connection verification uses the managed account ID,
not the port: IBKR's `D`-prefixed simulated accounts are PAPER and non-`D` accounts are LIVE. A
multi-account session must configure `expected_account` so Stocker does not select arbitrarily.

### M and PRE calculator boundary

The Stage 4 pure calculator receives two normalized model IV values and exposes no IBKR objects.
`PriorSessionContextService` composes the Stage 2 adapter, SQLite stores, calendar validation,
selector, and pure calculation. Stage 5 combines the result with current-session `P0` and `T0`:

```python
result = await pre_context_service.get_or_create(instrument, session=session)
# Stage 5 consumes result.context.expected_absolute_return_15m with P0/T0.
```

The pure arithmetic and selector run without a Gateway. Stage 4 does not define or calculate
`P0`, `T0`, `M_price`, or `PRE_MOVE_M`; those current-session measurements are Stage 5 concerns.
The `0.475764059845861` threshold, cohort bands, qualification, and ranking are strategy concerns
owned by Stage 6.

### Current-session feature layer

This layer produces reusable market measurements for later strategies. Calculate a shared snapshot
once where appropriate instead of duplicating it for PAPER, LIVE, or multiple strategies.

Stage 5 implements this as a non-trading pipeline in `stocker_execution.stage5`:

```text
active run universe membership
        ↓
Stage 2 stock qualification and conId deduplication
        ↓
Stage 4 expected_absolute_return_15m (demand-loaded)
        ↓
exact IBKR native 5-minute T0 open plus raw 1-minute T0-3m/T0 opens
        ↓
M_price → split-aligned raw PRE move → PRE_MOVE_M
        ↓
deterministic reusable feature snapshot
```

There is no additional generic cheap price, volume, market-cap, or option-availability screen. The
accepted broad-universe runner used a pre-frozen 511-security universe and data completeness; the
older `$5`, first-six-bar dollar-volume, and 100-session rules belonged to an unseen-cohort builder
and were not the frozen broad-universe Stage 5 contract. Stage 5 therefore narrows through active
membership and Stage 2 qualification, deduplicates overlapping members by `conId`, and requests
Stage 4 context only for the survivors. A failed symbol is recorded and does not stop the batch.

The caller supplies timezone-aware `T0`. The accepted Session HARD research generated checkpoints
at completed native five-minute RTH prefix counts `6, 8, ..., 34`: with a 09:30 America/New_York
bar zero, their next-bar `T0` opens were 10:00, 10:10, ..., 12:20 local time. Score inputs stopped
at `checkpoint - 1`; `P0` was the native five-minute open at the next bar. That cadence belongs to
the Session HARD strategy and is not a global Stage 5 scheduler. Stage 5 accepts any caller-supplied
causally valid `T0` and does not expose a result before its opening print.

Current-session history uses the existing IBKR-only cache with `TRADES`, `useRTH=True`, and native
`5 mins`/`1 min` semantics. Only exact timestamps are read as of `T0`; no close, interpolation,
nearest bar, fill, stale value, or other provider is accepted. The pure calculation is:

```text
M_price = P0 * expected_absolute_return_15m
alignment_factor = P0 / raw_1m_open[T0]
aligned_pre_open = raw_1m_open[T0 - 3 minutes] * alignment_factor
raw_PRE_move_price = abs(P0 - aligned_pre_open)
PRE_MOVE_M = raw_PRE_move_price / M_price
```

`PRE_MOVE_M` is a generic dimensionless measurement. The frozen strict comparison
`PRE_MOVE_M > 0.475764059845861` came from accepted Session HARD research and is not a universal
market-feature rule. Stage 5 neither applies it nor emits a qualification flag.

Feature rows retain run IDs, universe ID, qualified `conId`, exact input values, status/reason,
and calculation version in the existing SQLite storage approach. Core feature work is done once per
`conId + session + T0`; overlapping universe projections reuse it. Rows are emitted deterministically
by universe, qualified rows before unqualified rows, `conId`, and symbol. Once a READY snapshot is
stored, a later transient NOT_READY diagnostic cannot replace it.

Stage 5 has no strategy entry rules, risk sizing, orders, fills, positions, trade management, or
dashboard behavior. `stocker stage5-diagnostic` exercises the read-only PAPER data path for a small
custom universe and displays only generic feature fields.

### Strategy

A strategy evaluates prepared candidates and market state. It does not download history, resolve
universes, communicate with IBKR, or submit orders. Implement the first strategy concretely; let a
second real strategy reveal the generalisation actually needed.

Stage 6 implements the one concrete
`SESSION_HARD_HIGH_PRE_MOVE_DOWN_STRUCTURE_D` strategy in
`stocker_execution.session_hard_structure_d`. It consumes immutable Stage 5 snapshots plus causal
Session HARD score inputs. It neither mutates nor recalculates Stage 5 `P0`, `M_price`, or
`PRE_MOVE_M`. The frozen flow is:

```text
Stage 5 snapshot + run-specific strategy context
        ↓
prior HIGH_PRE_MOVE_DOWN_ONLY baseline-opportunity cohort
        ↓
COHORT_PRE_MOVE_PERCENTILE → LOW / MID / HIGH
        ↓
strict PRE_MOVE_M > 0.475764059845861 + Session HARD
        ↓
global known-MID veto; LOW, HIGH, and unavailable percentile remain eligible
        ↓
Structure D DOWN-first-touch within the five one-minute bars from T0
        ↓
simultaneous-candidate TOP_SESSION_HARD_SCORE_5 ranking
        ↓
deterministic SHORT order intention
```

The run-specific percentile cohort is the accepted `HIGH_PRE_MOVE_DOWN_ONLY` baseline opportunity
ledger: all symbols in the same named cohort that passed the strict absolute PRE gate, Session
HARD, and Structure D DOWN-first-touch. Its history is the previous 20 distinct qualifying session
dates, excludes the current session, and requires at least 30 prior qualifying observations.
Percentile is exactly
`100 * count(prior PRE_MOVE_M <= current PRE_MOVE_M) / prior_count`; ties therefore count below the
current value and no rounding is applied. Bands are LOW `<= 33.33`, MID `> 33.33 and <= 66.67`,
and HIGH `> 66.67`. A missing percentile is retained as unavailable rather than treated as MID.
The later global MID veto suppresses an order intention but does not rewrite the earlier baseline
cohort definition.

Session HARD is the frozen Model B movement score (`1 - quiet_probability`) over its 15 named
causal price/session-volume features and checkpoint one-hot, using completed native five-minute
bars only through checkpoint minus one. Checkpoints are `6, 8, ..., 34`; `P0` is the next bar open.
The exact frozen standardisation, coefficients, intercept, and inclusive
`score >= 0.999361477` cutoff live beside the strategy. Prepared score/checkpoint assessments are
keyed by `conId + session + T0` in the strategy context; the strategy does not fetch history itself.

Structure D sets `upper = P0 + 0.20M` and `lower = P0 - 0.20M`, then inspects the first five
one-minute bars from T0 in time order. A touch is interpreted only when the complete causal minute
prefix through that bar is present. An opening gap through a level enters at that open. Otherwise a
single lower-level touch establishes DOWN and enters at the level; an upper first touch rejects the
candidate, and a bar touching both levels is ambiguous and rejected. A non-gap touch becomes
actionable after that one-minute bar completes. DOWN is therefore a recovered first-touch state,
not the sign of PRE or a generic `price < P0` test.

Once every eligible candidate has a causal observation for the simultaneous timestamp, candidates
within the same run are ordered by Session HARD score descending, stock ascending, then the frozen
`stock|session|checkpoint` row identity, with at most five Stage 6 intentions. This is strategy
candidate capacity only. It does not model active account positions or available broker slots.

The intention records run, `conId`, symbol, session/T0, feature version, qualification and cohort
details, score/checkpoint, direction/side, candidate rank, `P0`, `M_price`, entry level/reference,
timestamps, and the frozen `0.50M` stop and `1.00M` target distances as metadata. Its ID is derived
from the strategy/version/run/`conId`/session/T0/feature version so repeated observations do not
emit a second intention. PAPER and LIVE produce the same deterministic strategy result.

The 10 bps cost, T0+15 outcome horizon, and conservative same-bar stop/target resolution are
historical research accounting rules. They are not runtime signal rules, live forced exits, or
broker-order behavior. The offline
`python -m stocker_execution.stage6_diagnostic --fixture <path>` command evaluates a frozen fixture
and prints cohort, qualification, ranking, and entry state without broker or account access.

### Ranking

Ranking remains strategy-specific in Stage 6. There is no generic ranking framework.

### Risk

Stage 7 consumes the selected, `ENTRY_TRIGGERED` Stage 6 `StrategySignal` as the authoritative
`OrderIntent`. It does not recalculate `PRE_MOVE_M`, cohort percentile/band, Session HARD,
Structure D, ranking, direction, or entry qualification.

Each run may carry an explicit `RunRiskConfig`. `risk_per_trade` has no default and is required by
Stage 7 execution; `max_concurrent_positions` is the only optional capacity setting. Actual PAPER
`NetLiquidation` from the connected IBKR account is authoritative. The pure calculation is:

```text
risk_budget = account_equity * risk_per_trade
per_share_risk = abs(entry_reference - stop_price)
quantity = floor(risk_budget / per_share_risk)
```

Stock quantities are whole shares and never round upward. Missing/invalid equity, nonpositive risk,
invalid protection, zero quantity, an existing same-`conId` position, or reached configured capacity
rejects that candidate without broker transmission.

### Execution

Stage 7 is implemented as:

```text
Stage 6 OrderIntent
        ↓
Stage7RiskDecision
        ↓
OrderPlan
        ↓
IBKR PAPER parent + protective children
        ↓
broker statuses / fills / positions / execution ledger
        ↓
startup and reconnect reconciliation
```

The first-touch signal becomes actionable only at its Stage 6 `signal_timestamp`, so its executable
entry mapping is a market SELL parent. The Stage 6 `entry_reference` remains the sizing and strategy
geometry reference. For the frozen SHORT strategy, Stage 7 calculates `stop = entry + 0.50M` and
`target = entry - 1.00M`, then attaches a BUY stop and BUY limit target. The parent and target use
`transmit=False`; the final attached stop uses `transmit=True`, following IBKR bracket transmission.
The market parent is `DAY`; both protective children are `GTC`, so a filled entry cannot outlive its
protection at the session boundary. For SHORT protection, the stop rounds down and target rounds up
to IBKR's qualified-contract minimum tick. Both move toward entry, so tick normalization cannot
increase requested per-share risk.

`Stage7PaperRuntime.observe_and_execute` is the direct production seam: it advances the concrete
Stage 6 strategy with entry bars, takes the resulting selected `ENTRY_TRIGGERED` intentions, and
passes them to the execution service as a batch. Candidate failures remain isolated. Stage 7 does
not impose another entry-expiry rule; the Stage 6 status and timestamps remain authoritative
strategy output.

`RunConfig.environment` remains independent of strategy and universe. The Stage 7 router state is:

```text
PAPER execution: ENABLED
LIVE execution: DISABLED (LIVE_EXECUTION_DISABLED)
```

The execution environment, expected account, and actual connected account are checked immediately
before planning/transmission. This retains the future shape in which individual
`strategy + universe + run` combinations can be promoted independently; there is no global
PAPER-to-LIVE switch and no LIVE order routing in Stage 7.

The Stage 2 `IbkrConnection` remains read-only by default for data consumers. Stage 7 explicitly
constructs a writable instance only for PAPER. It normalizes account state, minimum tick, open and
completed order status, executions, and positions instead of leaking `ib_async` callbacks upward.

### Storage and audit

`ExecutionLedger` uses SQLite. An atomic unique `signal_id` reservation occurs before broker
transmission, so repeated evaluation, replay, restart, or concurrent handling cannot submit the same
Stage 6 opportunity twice. It stores every execution attempt—including pre-plan safety rejections—
with run/environment/expected/actual account identity. Plans retain run/strategy/signal lineage,
intended quantity and geometry, all three IBKR order IDs, meaningful lifecycle state, deduplicated
IBKR execution IDs, aggregate entry/exit fills, timestamps, positions, and realized P&L.

Only fills create local exposure. Partial executions aggregate by quantity-weighted price and a
repeated IBKR execution callback is ignored. IBKR positions remain authoritative. At connect or
reconnect, new execution stays blocked until broker statuses, fills, open orders, and positions agree
with local records. A reserved plan can recover its broker IDs after a crash from the deterministic
IBKR `orderRef`, including a parent that filled before its ID was persisted. Unknown orders, fills,
positions, missing broker exposure, a filled position without both protective children, or
unresolved local plans return `EXECUTION_RECONCILIATION_REQUIRED`; Stage 7 never auto-flattens or
cancels everything.

The explicit `stocker stage7-paper-diagnostic` command requires caller-specified signal, instrument,
entry reference, M price, risk fraction, an expected PAPER account, and
`--confirm-paper-order`. The normal test suite never invokes broker transmission.

### Production runtime and recovery

Stage 8 is the orchestration layer over the existing public boundaries; it adds no second source of
trading decisions. `StockerRuntime` composes `RunManager`/`UniverseCatalog`, Stage 2 qualification,
`PriorSessionContextService`, `Stage5Analyzer`, the concrete Session HARD strategy,
`Stage7ExecutionService`, and `ExecutionLedger`. `build_paper_runtime` is the production composition
entry point and `runtime.start()` is the single lifecycle entry point.

Startup is deterministic:

```text
load run and PAPER IBKR config
        ↓
open the shared SQLite stores
        ↓
connect and verify the configured PAPER account
        ↓
Stage 7 reads broker orders, statuses/fills, and positions and reconciles the ledger
        ↓
load/start enabled PAPER runs and qualify shared instruments by conId
        ↓
resolve each timezone-aware exchange session
        ↓
READY and normal checkpoint processing
```

The invariant is absolute: no new order is submitted before account verification and successful
reconciliation for the current IBKR connection epoch. Disconnection changes global readiness to
`DEGRADED`; one bounded reconnect is followed by account verification and complete reconciliation
before `READY` is restored. Unknown broker exposure remains untouched and returns
`EXECUTION_RECONCILIATION_REQUIRED`. Stage 8 never guesses ownership, cancels all orders, or flattens
positions.

Run configuration remains per-run: `run_id`, `enabled`, universe, strategy, environment, risk, and
an explicit timezone/calendar session for every enabled executable run. No market or timezone is
silently substituted. PAPER and future LIVE runs can coexist in configuration.
Stage 8 activates only supported PAPER runs. An enabled LIVE run reports `LIVE_EXECUTION_DISABLED`
and is never silently sent to PAPER. A run-local data or strategy failure degrades that run while
unrelated reconciled runs continue.

Session HARD uses its frozen `6, 8, ..., 34` completed-five-minute checkpoints relative to the
configured exchange-local session open. Stage 5 work is shared for overlapping runs at the same
session/T0 and retains the exact `conId + session + T0` identity. A checkpoint is atomically reserved
and durably completed once. A process that starts after T0, misses the five-minute processing
window, or restarts after an interrupted checkpoint records the opportunity as skipped; it does not
manufacture a historical live signal. Stage 4 continues to receive the exact target session, so its
previous-session context cache cannot drift across trade sessions.

Stage 6 signal state is durably upserted as it is produced and after entry observation. On restart,
waiting signals inside their original five-minute causal entry window are restored without
reevaluating the checkpoint. Waiting signals whose live window elapsed while Stocker was offline
are marked expired and never reconstructed from historical bars for broker submission.

Shutdown stops evaluation and transmission first, records stopped run state, and disconnects. It
does not cancel protective children or flatten positions. Restart reconnects and lets Stage 7
reconcile pending, partially filled, open, closed-offline, and rejected-offline orders against the
same durable ledger before resuming. Runtime checkpoint/counter/cohort rows share the existing
SQLite approach and expose a small text/JSON status snapshot for the later dashboard.

`stocker stage8-paper-smoke` deliberately validates connection, account identity, reconciliation,
qualification, readiness, and one market-data snapshot. It does not transmit an order. The separate
Stage 7 diagnostic remains the only explicit manual diagnostic order path.

Extended PAPER burn-in can continue while Stages 9 and 10 are developed; Stage 8 code completion is
based on runtime, recovery, deterministic integration tests, and the manual diagnostic rather than
elapsed observation time.

## Failure philosophy

- Trading-critical invalid state: do not initiate a new trade until valid.
- Single-symbol problem: skip that symbol and continue.
- Non-critical subsystem problem: log it and continue where safe.

Avoid elaborate retry and fallback state machines.

## Planned stages

0. Freeze architecture and development rules.
1. Add the minimal application skeleton and run configuration.
2. Add the IBKR foundation: connection, accounts, contract resolution, and historical/current data.
3. Add universes and multiple runs.
4. Add the IBKR history cache and PRE-level calculation.
5. Add reusable current-session feature production.
6. Add the first real strategy, including its cohort, bands, qualification, and ranking.
7. Add risk and PAPER execution.
8. Add production PAPER orchestration, restart/reconnect recovery, deterministic scheduling,
   operational status, and burn-in support over Stages 1--7.
9. Enable controlled LIVE order transmission per run while preserving explicit PAPER/LIVE routing.
10. Add the dashboard as a non-critical consumer of the Stage 8 status surface.

Future stages must not be implemented prematurely.
