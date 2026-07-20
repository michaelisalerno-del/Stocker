# Immediate Regime-Pair Closure History Diagnostic V1 — Run Note

Safety boundary: `research_only=true`, `execution_enabled=false`, `order_placement=disabled`, `live_ordering_enabled=false`, `strategy_promotion=false`.

## Start

- Starting branch: `agent/observable-event-ranking-v1` at `35cd9d45` with a clean worktree.
- Research branch: `agent/slrno-regime-loop-history-test-v1` from archived `origin/agent/slrno-research-handoff` at `04c6d45589e0c114dc0b03f6f98b4858bde7dffe`.
- The observable-event lineage is not imported or modified by this test.
- Git LFS is unavailable locally. The 20260719 LFS Parquet outputs remain pointers; the test reconstructs bounded panels from audited local EODHD sources and uses the small hash-bound repaired-model files tracked in Git.

## Available evidence

- Audited 2024 panel identity: 424,583 rows, 22 stocks, snapshot `48d2141...`.
- Unchanged 2025 assessment identity: 424,827 rows, 22 stocks, snapshot `29e82d65...`.
- Source: provider five-minute OHLCV, primarily EODHD. Volume is provider-reported historical activity, not order flow or executable liquidity.
- Full right-censored refit model: `4fc1a02d...`.
- Repaired Part A still failed numeric semantic stability: minimum aligned K=8 NMI `0.434553`; minimum sample event agreement `0.002017`; dictionary work remained paused.
- The validated cluster-invariant excursion lineage is read-only evidence and is not converted back into numeric loop identities.

## Frozen question

At the first completed bar of a causal state run B, after a completed state run A, does deeper causal state history improve prediction that the next completed state returns to A, forming the corrected primitive closure A→B→A?

Primary comparison: fixed `M5_LAST_FIVE_STATES` versus `M2_IMMEDIATE_PAIR` on unchanged 2025 structural outcomes after 2024-only fitting. Numeric pair identities are explicitly non-promotable and cannot establish direction, payoff, or execution edge.

## Planned outputs

- Reconstructed source and state identities.
- Outcome-free-at-decision transition population with censored-run accounting.
- 2024 expanding-fold and 2025 frozen-fit predictions.
- Model ladder, pair-orientation, quarter, concentration, leave-one-stock-out, and paired session-bootstrap results.
- Exact rerun comparison and independent audit.

## Completed run

- Run ID: `20407e328d4930182be5a6d6`.
- Development population: 177,096 decisions across both required representations; 2024 primary evaluable rows: 76,383 (57,599 expanding-fold OOF rows after the initial training months).
- Unchanged assessment population: 180,313 decisions across both required representations; 2025 primary evaluable rows: 78,297 across 250 sessions.
- Primary closure rate: 0.386452.
- `M2_IMMEDIATE_PAIR` assessment log loss: 0.617531; `M5_LAST_FIVE_STATES`: 0.611971.
- Frozen M5-minus-M2 log-loss improvement: 0.005560 with paired session-block 95% interval [0.004603, 0.006575].
- Frozen Brier improvement: 0.002481 with paired session-block 95% interval [0.002069, 0.002914].
- All four 2025 quarters and all 22 leave-one-stock-out estimates had the same positive direction.
- Causal hard-label sensitivity improvement: 0.004820 with 95% interval [0.003965, 0.005660].
- `M4_LAST_FOUR_STATES` was marginally better than M5 on 2025 log loss (0.611892 versus 0.611971), indicating that the fourth preceding run added no measurable benefit beyond the shorter context.
- Of 36 supported, BH-significant primary pair orientations selected in 2024, 32 replicated with the same direction and BH significance in 2025.
- Final decision: `fixed_model_history_increment_observed_nonpromotable`.

## Reproducibility and audit

- Primary and exact-rerun directories were regenerated after the final reporting correction.
- Exact rerun: 18 scientific files compared, all byte-identical.
- Independent audit: 63 of 63 checks passed; the auditor did not import the candidate module or runner.
- Focused new tests: 7 passed.
- Inherited right-censored refit tests: 28 passed.
- Inherited loop-event semantics tests: 16 passed and 5 failed because the archived `20260718-loop-event-semantics-v2` Parquet artifacts are absent locally while Git LFS is unavailable. These failures do not exercise the new diagnostic.
- Ruff format, Ruff lint, strict mypy, and `git diff --check`: passed.
- Static production-module scan found no forbidden broker or order-capable calls.
- No broker connection, account/position request, order action, price target, payoff target, or protected 2026 data was used.
