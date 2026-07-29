# Dynamic loop temporary payoff edge state V2

## Baseline

- Branch: `agent/slrno-research-handoff`
- Frozen pre-experiment commit: `8baf974f2d13751064dbc4d2c7cf65d02e3a8912`
- Final scored implementation commit: `ca3537a0f337097a9a75abf87ae4bf419fae6a5d`
- V1 exact rerun passed before V2 scoring; summary SHA-256 `3d1238a567479b48c87dcf20b76c89b9204c42909b894cc98cfde5e2a38605e1` matched the archived result.
- V1 h24 mean net payoff was -0.01 bps/trade in 2025 and +1.25 bps/trade in backward-2023; its original hypothesis was rejected.

## Registered hypotheses

1. Payoff-history BOCPD can reduce useful activation/termination lag versus V1's 60-session selector.
2. Shared online payoff state can improve sparse cells when independent-session, independent-stock, and ESS evidence is relevant.
3. Compact lagged breadth/coherence features can improve next-payoff calibration over the identical hierarchy without those features.
4. Equal-stock robust session aggregation can prevent correlated fill clusters and single outliers from creating false support.

The 24-bar horizon, 0.05 primary hazard, two hazard sensitivities, probability thresholds, feature weights, and stress set were frozen in the versioned contract. No final-P&L parameter search was run.

## Implementation and review corrections

- Added one capped stock contribution per session × loop × orientation × 24 bars, with equal-stock 10%-winsorised mean primary aggregation and median sensitivity.
- Added bounded Student-t BOCPD, conditional onset/termination probabilities, next-observation predictive probability, hierarchical shared environment, evidence-aware shrinkage, four operational states, abstention, and frozen admission joins.
- Separated the structural loop forecast from the economic payoff-state filter and left all existing-position exits unchanged.
- Added `hierarchical_payoff_history_change_point` so feature value is compared against the same hierarchy rather than confounded with pooling.
- Rebuilt breadth, market context, payoff panels, shared state, cell states, features, forecasts, and admissions for every leave-one-stock-out stress.
- Corrected calibration to score `p_next_payoff_positive` against the next settled robust payoff event; latent-mean probability remains a separate output.
- Counted BOCPD change points separately from operational state transitions and measured pre-decay features at the hindsight decay boundary.
- Added a separate exact-rerun/causality auditor.
- The hash-verified recovery adapter reconstructed all 250 causal sessions per year from sealed derived artifacts after ephemeral inputs expired, proving exact top-loop/probability/state/history-token equality on all 10,382 V1-scored h24 rows.

## Validation

- Focused V2 suite: `37 passed`.
- Full repository suite: `406 passed`; four pre-existing NumPy `Mean of empty slice` warnings in `test_behavioral_state_similarity.py`.
- Scoped Ruff format and lint: passed for all V2 source, runner, auditor, and tests.
- Strict mypy: passed for the four reusable V2 modules plus the independent auditor.
- Independent primary/exact audit: `48/48 passed`.
- Audit coverage: exact equality of 13 machine-readable tables, identical episode-plot hashes, both manifests, safety flags, metadata identity, V1 recovery, unique frozen forecasts, strict training/feature availability, freeze timestamps, decision joins, statistical units, support counts, run metadata, all 20 fully retrained stock deletions, and period boundaries.
- `git diff --check`: passed.
- Repository-wide Ruff remains red on 1,153 pre-existing errors outside this experiment; no unrelated cleanup was attempted.

## Primary results

| model | Brier | ECE | lag ratio | accepted fills | net P&L (bps) | net bps/fill | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 60-session | 0.2579 | 0.0633 | 0.1458 | 4,230 | 2,903.70 | 0.69 | 42.52% |
| EWMA | 0.2585 | 0.0741 | 0.5897 | 165 | -3,360.83 | -20.37 | 1.62% |
| Payoff-only BOCPD | 0.2961 | 0.1485 | 0.4983 | 324 | 8,380.76 | 25.87 | 3.27% |
| Hierarchy, no leading features | 0.2757 | 0.1123 | 0.4284 | 375 | -4,466.42 | -11.91 | 3.75% |
| Full hierarchy + features | 0.3250 | 0.2385 | 0.4932 | 275 | -9,416.38 | -34.24 | 2.75% |
| No filter | NA | NA | NA | 9,926 | -12,317.08 | -1.24 | 100.00% |

The full model detected 47/215 hindsight-positive episodes, with a mean 10.45-session activation delay, 4-session median delay, and 0.4932 lag ratio. V1 detected 127/215 with a 0.1458 lag ratio. Hindsight labels were evaluation-only.

## Stress and concentration

- Twice costs: full model -12,166.38 bps; payoff-only +5,140.76 bps; V1 -39,396.30 bps.
- One-session policy delay: full model +9,014.54 bps; payoff-only +3,373.28 bps; V1 +1,407.89 bps. The sign reversal reinforces instability rather than rescuing the failed primary model.
- Fully retrained leave-one-stock-out full-model range: -10,516.26 to +7,306.34 bps; only 9/20 deletions were positive.
- Full-model top-five stocks supplied 88.36% of positive contribution; the best episode supplied 4,529.95 bps and the top five episodes 82.96%.
- Breadth rose before 37.21% of hindsight episodes and coherence before 40.47%; dispersion rose before 35.35% of decay boundaries and structural surprise before 43.26%. These are descriptive, not predictions.

## Decision

- **Rejected:** the registered full breadth/coherence hierarchical gate. It was worse calibrated than the same hierarchy without features, slower than V1, lost 9,416.38 bps after costs, failed twice-cost robustness, and was concentrated/unstable under fully retrained stock deletions.
- **Not promoted:** payoff-only BOCPD. Its retrospective P&L and cost/delay stresses are interesting, but calibration and detection lag were materially worse than V1, coverage was only 3.27%, and stock contribution remained concentrated.
- **Kept:** the causal statistical unit, explicit settlement clock, reusable online model, immutable forecast ledger, abstention/state machinery, audit, and negative artifacts.
- Safety held: research only; no broker, paper/demo, deployment, position-management, order-placement, or frozen-exit code changed.

## Exact commands

```bash
rtk uv run python research/slrno-v2/20260714-regime-loop-handoff/work/run_dynamic_loop_context_edge_v1.py --output /private/tmp/stocker_dynamic_loop_context_edge_v1_20260714_v2_baseline
rtk .venv/bin/pytest -q tests/test_dynamic_loop_edge_state_session_payoff.py tests/test_dynamic_loop_edge_state_model.py tests/test_dynamic_loop_edge_state_walkforward.py tests/test_run_dynamic_loop_edge_state_v2.py tests/test_audit_dynamic_loop_edge_state_v2.py
rtk .venv/bin/pytest -q
rtk .venv/bin/mypy packages/stocker_research/src/stocker_research/dynamic_loop_edge_state/session_payoff.py packages/stocker_research/src/stocker_research/dynamic_loop_edge_state/online_state.py packages/stocker_research/src/stocker_research/dynamic_loop_edge_state/decision.py packages/stocker_research/src/stocker_research/dynamic_loop_edge_state/walkforward.py research/slrno-v2/20260714-regime-loop-handoff/work/audit_dynamic_loop_edge_state_v2.py
rtk sh -c 'MPLCONFIGDIR=/private/tmp/stocker-mpl-cache .venv/bin/python research/slrno-v2/20260714-regime-loop-handoff/work/run_dynamic_loop_edge_state_v2.py'
rtk sh -c 'MPLCONFIGDIR=/private/tmp/stocker-mpl-cache .venv/bin/python research/slrno-v2/20260714-regime-loop-handoff/work/run_dynamic_loop_edge_state_v2.py --output research/slrno-v2/20260714-regime-loop-handoff/work/artifacts/20260714-dynamic-loop-edge-state-v2/exact_rerun --report research/slrno-v2/20260714-regime-loop-handoff/work/reports/20260714-dynamic-loop-edge-state-v2-exact-rerun.md'
rtk .venv/bin/python research/slrno-v2/20260714-regime-loop-handoff/work/audit_dynamic_loop_edge_state_v2.py --primary research/slrno-v2/20260714-regime-loop-handoff/work/artifacts/20260714-dynamic-loop-edge-state-v2/primary --exact-rerun research/slrno-v2/20260714-regime-loop-handoff/work/artifacts/20260714-dynamic-loop-edge-state-v2/exact_rerun --output research/slrno-v2/20260714-regime-loop-handoff/work/artifacts/20260714-dynamic-loop-edge-state-v2/exact_rerun/independent_audit.json
```

## Next experiment

Seal this implementation and prospectively log, without execution, the full model versus the identical hierarchical model with leading features disabled. Do not retune thresholds or feature weights on the opened 2023/2025 surfaces.
