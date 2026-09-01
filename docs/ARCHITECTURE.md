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
PRE Level Calculation
        ↓
Feature / Cohort / Band Calculation
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

### Run

A run is one active combination of universe, strategy, `PAPER` or `LIVE` environment, trading or
session window, and relevant configuration. Runs may execute sequentially or concurrently.

```text
FTSE Morning          NASDAQ Main       US Midcap Experiment
FTSE350               NASDAQ            US_MIDCAP
SESSION_HARD          SESSION_HARD      NEW_STRATEGY
PAPER                 LIVE              PAPER
```

### Historical data

IBKR is the exclusive source for historical bars used by production and paper PRE calculations.
The local cache stores only IBKR-originated history. Stocker must not silently fall back to EODHD,
FMP, TwelveData, or another provider for those calculations.

### PRE level calculator

The calculator receives bars and knows nothing about IBKR:

```python
bars = history_service.get(...)
levels = pre_levels.calculate(bars)
```

This makes calculation tests deterministic without a broker connection.

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
