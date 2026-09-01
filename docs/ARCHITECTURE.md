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
Current-Session M / PRE_MOVE / Feature / Cohort / Band Calculation
        ↓
Strategy Qualification
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

The current Stage 1--3 branch does not contain an executable or frozen bars-only PRE-level
definition. The latest accepted research lineage inspected was:

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

Those sources prove exact-minute missing-bar failure and the downstream PRE_MOVE formula, but they
do not establish the requested production PRE history contract: canonical IBKR request semantics,
lookback/session completeness, option-IV acquisition equivalence, price-adjustment policy, or a
bars-only PRE-level formula and golden outputs. Therefore no `PreHistorySpec`, PRE calculator,
calculation version, automatic gap fetch, or live diagnostic is implemented yet. Choosing those
values would create new trading mathematics. The cache exposes exact gaps so the eventual frozen
contract can request only missing history once that contract is supplied.

The follow-up audit confirms that the research normalisation itself is row-specific and correct:

```text
expected_absolute_return_15m = ATM_IV * sqrt(15 / (252 * 390)) * sqrt(2 / pi)
M_price = current-session P0 at T0 * expected_absolute_return_15m
PRE_MOVE_M = abs(P0 - raw_open[T0-3m] * (P0 / raw_open[T0])) / M_price
```

The unchanged strict threshold `PRE_MOVE_M > 0.475764059845861` is downstream of this
normalisation and is never `M`.

This timing corrects the Stage boundary. Stage 4 may own the exact-prior-session IBKR option/IV
context and its dimensionless expected-move fraction after source parity is proven. Stage 5 owns
`T0`, current-session `P0`, final dollar `M_price`, raw PRE movement, `PRE_MOVE_M`, thresholds,
bands, and ranking. Final dollar `M_price` is not known on the prior session because its scale uses
current-session `P0`.

No IBKR option-IV field has yet been proven equivalent to the frozen research provider's selected
call/put `implied_volatility`. Production implementation remains gated on a prospective,
source-labelled parity capture; research/vendor data may be used only as the comparison reference
and never as a PAPER/LIVE fallback.

### IBKR boundary

`stocker_execution.ibkr.IbkrConnection` is the Stage 2 read-only boundary for connection/session
identity, stock qualification, historical bars, and current snapshots. Each instance owns its own
client state, so PAPER and LIVE can later use separate Gateway sessions without a global singleton.
The adapter exposes small Stocker models and no order methods; strategies and calculations must not
import IBKR objects.

PAPER and LIVE host, port, client ID, and environment are separate explicit configuration blocks.
The Stage 1 run environment selects one block. Connection verification uses the managed account ID,
not the port: IBKR's `D`-prefixed simulated accounts are PAPER and non-`D` accounts are LIVE. A
multi-account session must configure `expected_account` so Stocker does not select arbitrarily.

### M and PRE calculator boundary

The intended Stage 4 context service receives normalized prior-session inputs and exposes no IBKR
objects. Stage 5 will later combine that context with current-session `P0` and `T0`:

```python
prior_context = stage4.get_prior_session_context(instrument, session)
# Stage 5 later consumes prior_context.expected_absolute_return_15m with P0/T0.
```

This boundary will make calculation tests deterministic without a broker connection. The recovered
arithmetic is frozen by research-reference tests, but the current-session calculator belongs to
Stage 5 and is not implemented here. Stage 4's remaining blocker is an authoritative IBKR
option-IV acquisition/parity contract.

### Feature and band layer

This layer derives values required by strategies, including PRE_MOVE, percentiles, and LOW/MID/HIGH
cohorts. Calculate a shared snapshot once where appropriate instead of duplicating it for PAPER,
LIVE, or multiple strategies.

### Strategy

A strategy evaluates prepared candidates and market state. It does not download history, resolve
universes, communicate with IBKR, or submit orders. Implement the first strategy concretely; let a
second real strategy reveal the generalisation actually needed.

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
5. Add screening, features/bands, and candidate ranking.
6. Add the first real strategy.
7. Add risk and PAPER execution.
8. Burn in paper trading and fix primarily real observed failures.
9. Add LIVE execution alongside PAPER.
10. Add the dashboard.

Future stages must not be implemented prematurely.
