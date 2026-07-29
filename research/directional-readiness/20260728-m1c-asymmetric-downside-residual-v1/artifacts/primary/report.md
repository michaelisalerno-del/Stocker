# M1C Asymmetric Downside Residual V1

## Decision

- Probability-decomposition decision: `target_mismatch_prevents_exact_probability_decomposition`.
- Endpoint-direction decision: `blocked_insufficient_support`.
- Descriptive endpoint finding: `low_downside_does_not_imply_upside`.
- This is retrospective out-of-development evidence from already-opened 2025 periods, not untouched confirmation.

## Target audit

M1C enters at the next five-minute bar open and measures the absolute log return to the close of the third post-entry bar (15 minutes). Its material-move label is strictly `absolute return > prior-close ATM IV × sqrt(15/(252×390)) × sqrt(2/π)`.
The required directional endpoint partition assigns exact positive and negative threshold equality to UP and DOWN. The events therefore do not form the literal complement of the frozen strict M1C event. No joint up/down/no-move probabilities were constructed.

## Direct answers

1. **Clean three-state decomposition?** No—not exactly, because of the strict-versus-inclusive equality mismatch. The endpoint diagnostic itself is exhaustive.
2. **High-M1C fresh material movers?** Development 253, assessment 205, stress 270.
3. **Rank downside among movers?** Assessment AUC 0.5045; stress 0.4977. Assessment session-bootstrap 95% CI [0.4038, 0.6086].
4. **Proper-score improvement?** Assessment log-loss/Brier improvements -0.0138/-0.0065; stress -0.0072/-0.0038.
5. **PUT more downside than CALL?** Assessment spread 0.0414 (95% CI -0.1754 to 0.2649); stress -0.0049. The assessment PUT cell has 29 actions, so the formal action-support result is `blocked_insufficient_support`.
6. **CALL more upside than downside?** Assessment 0.2333 up versus 0.2000 down; stress 0.1944 versus 0.4167.
7. **CALL merely no-move?** In assessment, yes: 0.5667 were no-moves, a majority. In stress the no-move rate was 0.3889, but downside outnumbered upside, so low downside score still did not imply upside.
8. **Selective policy versus baselines?** No consistent outperformance. Assessment mean aligned return was -0.002574 versus recent 5m/15m, market, and A1 -0.000641/-0.000579/0.000029/0.001212. Stress was 0.000377, beating the momentum/market means but not frozen A1 consistently across both periods. Frozen D2 is blocked by contaminated or unreproducible lineage. Acted-timestamp comparisons are reported separately in `baseline_comparisons_v1.csv`.
9. **Stable assessment/stress?** No. AUC moved from 0.5045 to 0.4977, down-rate spread changed from 0.0414 to -0.0049, and mean aligned return changed sign. No refit or recalibration occurred.
10. **Broad support?** See leave-one-out and concentration tables. The summary robustness result is `{'all_leave_one_stock_month_checkpoint_down_spreads_positive_and_returns_positive': False, 'all_stock_month_session_checkpoint_time_concentrations_below_50_percent': False, 'maximum_acted_concentration': 0.6857142857142857, 'assessment_one_percent_winsorised_mean_aligned_return': -0.0026095908575780734, 'stress_one_percent_winsorised_mean_aligned_return': 0.00044370607580315947}`.
11. **Tail Phase?** It did not reveal a consistent modifier. FIRST_ENTRY AUC was 0.5203 assessment and 0.4928 stress; PERSISTENT checkpoint rows were secondary and changed sign; assessment RE_ENTRY status was `blocked_insufficient_support`. No phase gated, fit, or changed thresholds. Full action/A1/checkpoint/time strata are in `tail_phase_diagnostics_v1.csv`.
12. **Movement remaining?** Mean canonical post-entry local-range share among acted episodes was assessment 0.4099 and stress 0.4293.
13. **Ranking/policy conclusion?** Formal endpoint decision `blocked_insufficient_support`; descriptive finding `low_downside_does_not_imply_upside`.
14. **Still unknowable?** Option profitability, executable bid/ask prices, slippage, market impact, fill probability, and prospective behavioural stability remain unknown. Underlying aligned returns are not option P&L.

## Action accounting

| period | CALL | PUT | ABSTAIN | mean aligned return |
|---|---:|---:|---:|---:|
| assessment | 30 | 29 | 358 | -0.002574 |
| stress | 36 | 34 | 455 | 0.000377 |

## Null tests

Assessment permutation p-values: AUC 0.2398, log-loss improvement 0.2907, Brier improvement 0.2827, down-rate spread 0.2937.
Stress permutation p-values: AUC 0.4006, log-loss improvement 0.2607, Brier improvement 0.2707, down-rate spread 0.3946.

The fixed temporal placebo reassigns each stock's prior episode predictors to the next episode and reruns the same fixed procedure; results are in `temporal_placebo_v1.csv`.

The inherited 10-minute endpoint diagnostic is retained only as a secondary table in `secondary_10m_directional_v1.csv`; it is not substituted for the M1C-compatible 15-minute primary horizon.

## Safety and scope

- M1C, A1, Tail Phase V1, the frozen cohort, checkpoint grid, and fresh-episode identifiers were unchanged.
- No archived pressure, tension, peer-slate normalisation, future-dependent membership, option outcomes, or execution fields entered the model.
- No protected 2026 outcome was read, calculated, displayed, or inspected.
- No broker was accessed, no order routing was enabled, and no order was placed.
