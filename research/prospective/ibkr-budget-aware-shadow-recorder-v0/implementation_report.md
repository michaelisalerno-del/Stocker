# Implementation Report

## Outcome

The existing prospective application was extended in place. It now runs the
exact frozen causal M1C runtime from minimal IBKR completed-bar subscriptions,
monitors later EODHD provider transfer, begins bounded option shadow recording
immediately, and preserves a configurable 12-line future-trading market-data
reserve. No order, account, position, or portfolio surface was added.

The historical conclusion is unchanged:
`blocked_insufficient_low_tail_support`.

## Runtime

- Capacity discovery and provenance:
  `stocker_prospective.capacity`
- Priority, ownership, deduplication, shedding, and reconciliation:
  `stocker_prospective.subscriptions`
- Minimal bars and promoted underlying data:
  `stocker_prospective.live_subscriptions`
- Bounded option queue and state machine:
  `stocker_prospective.option_budget`
- Metadata-only discovery and bounded snapshots:
  `stocker_prospective.option_discovery`
- Conservative shadow outcomes:
  `stocker_prospective.option_recorder`
- EODHD/IBKR transfer metrics and V1 guard:
  `stocker_prospective.transfer`
- Later-provider reconstruction and aggregate decision:
  `stocker_prospective.source_transfer`
- Daily report packages:
  `stocker_prospective.budget_reports`

## Persistence and web

Migration `0010_ibkr_budget_transfer_v0.sql` adds runtime capacity,
subscription transitions, explicit option allocation/degradation, skipped
recordings, provider observations, transfer-session decisions, and prospective
session phases. The read-only web app adds budget, source-transfer, and report
download endpoints plus a permanent no-orders banner.

## Scientific controls

M1C artifact hashes and all four thresholds are fail-closed. No contaminated
peer-slate feature or descendant is introduced. Transfer evaluation uses
provider alignment and rank/threshold/episode behavior rather than exact bar
equality. Option outcomes from the first 20 valid sessions remain engineering
only. A V1 calibration candidate can use only the frozen 20-session IBKR
probability distribution.

## Verification status

The feature-scoped verification suite passes:

- 77 focused backend, web, fake-adapter, budget-degradation, reconnect, and
  deterministic tests.
- Scoped Ruff formatting and lint checks.
- Strict mypy for all 63 prospective source files.
- Independent reconstruction of 100 IBKR M1C probabilities, 100 EODHD/IBKR
  probability comparisons, 50 subscription decisions, 25 constrained option
  episodes, and 50 conservative shadow outcomes.
- Two identical audit replays with zero M1C, tail, episode, subscription, DTE,
  contract, or outcome mismatches and maximum floating difference `0.0`.

The repository-wide test run reached 1,291 passes and one skip. Thirteen
failures and 19 setup errors are confined to unrelated historical SLRNO
integration tests whose frozen parquet inputs are absent from this checkout;
those prior artifacts were not modified or reconstructed. This repository has
no Node package manifest or TypeScript build surface.

Commit SHAs and push status are recorded in the delivery response.
