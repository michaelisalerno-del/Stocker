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

`stocker_execution.ibkr.IbkrConnection` is the Stage 2 read-only boundary for connection/session
identity, stock and option qualification, option definitions/model snapshots, historical bars,
and current stock snapshots. Each instance owns its own
client state, so PAPER and LIVE can later use separate Gateway sessions without a global singleton.
The adapter exposes small Stocker models and no order methods; strategies and calculations must not
import IBKR objects.

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

Stage 6 will define the first strategy-specific eligible cohort, calculate the recovered causal
`COHORT_PRE_MOVE_PERCENTILE`, derive LOW/MID/HIGH, apply the frozen strategy threshold, perform
Session HARD qualification, and rank qualified candidates before producing any signal or order
intention. None of those operations is Stage 5 runtime infrastructure.

### Ranking

Ranking orders already-qualified candidates when research or capacity selection requires it.

### Risk

Risk converts a qualified trade intention into permitted size and risk parameters.

### Execution

Execution owns broker-facing order activity. PAPER and LIVE routing is explicit. Later they may run
simultaneously through separate IBKR Gateway sessions or accounts.

### Storage and audit

Store enough to explain the run, universe, strategy, environment, instrument/conId, relevant
feature and level snapshot, signal, order, fill, position, exit, and P&L. Do not build elaborate
event sourcing unless it becomes necessary.

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
8. Burn in paper trading and fix primarily real observed failures.
9. Add LIVE execution alongside PAPER.
10. Add the dashboard.

Future stages must not be implemented prematurely.
