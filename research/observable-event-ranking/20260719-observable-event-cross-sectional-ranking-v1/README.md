# Observable Event Cross-Sectional Ranking V1

This is a clean-slate, research-only lineage for one frozen hypothesis: rank
simultaneously eligible US stocks after a newly confirmed positive market-and-sector-
relative acceleration event. It does not import or depend on the archived SLRNO regime,
loop, excursion, posterior, route, personality, or payoff-selection line.

The historical source boundary is audited regular-session five-minute OHLCV, initially
EODHD, with a hard development cutoff of `2025-12-31`. Provider volume is treated only as
a provider-reported activity proxy. Historical bar references are not fill claims.

## Current scientific status

The local source inventory cannot support a scientific event run. It has no trusted
effective-dated point-in-time sector membership, the safe pre-cutoff raw population has
only 43 symbols, the source bar-label convention is not proven, and corporate-action
handling is unresolved. The run therefore stops before event calibration or target
construction with `blocked_missing_point_in_time_sector_membership`. It does not read
protected 2026-containing market data, construct targets, fit M1, select a baseline, or
claim structural, directional, economic, or executable evidence.

## Frozen experiment

- Event: `E1_POSITIVE_RELATIVE_ACCELERATION` only.
- Candidate: `M1_POOLED_LINEAR_RANKER`, ridge alpha `1.0`, exact 12-feature surface.
- Side: structural long-only ranking; no short side and no order simulation.
- Decision clocks: 10:00 through 14:30 New York time at 30-minute intervals.
- Entry reference: open of `t+2`, after one complete five-minute dispatch delay.
- Primary target: within-slate percentile rank of the 60-minute future raw return.
- Primary uncertainty: paired 2,000-draw session-block bootstrap.
- Primary support gate: 1,000 supported slates, 5,000 unique events, and every other
  frozen population/concentration requirement in the contract.

The canonical full contract is implemented in
`packages/stocker_research/src/stocker_research/observable_event_ranking_v1/contract.py`
and materialized as `frozen_experiment_contract.json` in each artifact directory.

## Commands

From the repository root, use the standalone runner:

```bash
uv run python research/observable-event-ranking/20260719-observable-event-cross-sectional-ranking-v1/work/run_observable_event_ranking_v1.py preflight --data-dir <local-data-root>
uv run python research/observable-event-ranking/20260719-observable-event-cross-sectional-ranking-v1/work/run_exact_rerun_v1.py --data-dir <local-data-root>
uv run python research/observable-event-ranking/20260719-observable-event-cross-sectional-ranking-v1/work/audit_observable_event_ranking_v1.py --primary research/observable-event-ranking/20260719-observable-event-cross-sectional-ranking-v1/work/artifacts/primary --exact-rerun research/observable-event-ranking/20260719-observable-event-cross-sectional-ranking-v1/work/artifacts/exact_rerun
```

The main runner also exposes `build-events`, `audit-events`, `build-targets`,
`run-development`, `audit-development`, `freeze-prospective`, `score-prospective`,
`settle-prospective`, `ibkr-resolve-contracts`, `ibkr-capture-quotes`, and
`ibkr-observability-dry-run`. Each command enforces its frozen prerequisite. Bounded
`--max-symbols` and `--max-sessions` outputs are explicitly non-scientific; there is no
protected-data bypass flag.

## IBKR boundary

IBKR is a prospective quote-observability source only. The separate observability package
does not subclass the order-capable broker, import order classes, or expose account,
position, execution, or order methods. Automated tests use a fake client and never open a
network connection. Live commands are disabled by default, localhost-only, and remain
blocked until the official locally installed TWS Python API transport is independently
validated with TWS/Gateway read-only API mode enabled.

An observed bid/ask bounds a reference quote; it never proves a fill. Frozen, delayed,
partial, stale, or late data cannot be classified as a complete live top-of-book
observation.
