# Pre-Trigger Quiet Accumulation / Distribution Direction Screen V0

**Overall decision:** `blocked_insufficient_pretrigger_history`

This is retrospective directional candidate evidence based on underlying-stock returns and bar-derived behaviour. It does not observe institutional accumulation, direct order flow, exchange-verified volume, option P&L, or prospective execution.

## Chronology and frozen movement gate

- Development: 2024-01-01 through 2024-12-31.
- Retrospective assessment: 2025-01-01 through 2025-08-22.
- Excluded opened holdout: 2025-09-01 through 2025-12-31.
- Protected: 2026-01-01 onward; no protected outcomes were read or materialised.
- Frozen M1 threshold: `0.49588519865576763`.
- Frozen reconstruction: 1,266 raw above-threshold rows and 538 fresh episodes (285 development, 253 assessment).

## Pre-trigger construction

- Binding window: five completed five-minute bars (25 minutes), T-5 through T-1.
- Direction marker: close of T-1.
- The complete M1 trigger bar T was excluded from every direction feature.
- Entry remained the first post-trigger completed bar open.
- Signed pressure: exact audited even-checkpoint snapshots only; intervening bars remain missing and are never interpolated, carried, or redefined.
- Activity: repository causal activity proxy `historical_relative_activity`; it is not asserted to be exchange volume.
- Complete five-bar pressure windows: 0/253 assessment episodes.

The composite is `quietness_25 × mean(13 clipped signed component z-scores)`, with development-only median/IQR preprocessing, development-median missing imputation, clipping to [-3,+3], and equal weights.

## Assessment model metrics

| model_id | episodes | log_loss | brier_score | auc | average_precision | accuracy | balanced_accuracy | calibration_intercept | calibration_slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q0 | 250 | 0.73281266 | 0.26747924 | 0.49729976 | 0.5527029 | 0.544 | 0.53952762 | 0.26507933 | -0.16635356 |
| QS | 250 | 0.7148901 | 0.26047646 | 0.47205413 | 0.54534612 | 0.492 | 0.49551044 | 0.25649811 | -0.36956922 |
| Q1 | 250 | 0.73765152 | 0.26943738 | 0.49417659 | 0.54893966 | 0.524 | 0.51971501 | 0.26335923 | -0.1787372 |

## Frozen selective policy

Q1 confidence boundary: `0.131965383428`.

| horizon_minutes | actions | abstentions | action_coverage | call_count | put_count | directional_accuracy | balanced_accuracy | mean_aligned_return | median_aligned_return | positive_aligned_return_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 77 | 176 | 0.30434783 | 45 | 32 | 0.49350649 | 0.48571429 | -0.0012302881 | -0.00022092981 | 0.49350649 |
| 10 | 77 | 176 | 0.30434783 | 45 | 32 | 0.45454545 | 0.43854167 | -0.0013580758 | -0.00078950414 | 0.45454545 |
| 15 | 77 | 176 | 0.30434783 | 45 | 32 | 0.51948052 | 0.50315568 | -0.0012285701 | 0.001136697 | 0.51948052 |
| 30 | 77 | 176 | 0.30434783 | 45 | 32 | 0.51948052 | 0.53472222 | -0.00010478007 | 0.0018753874 | 0.51948052 |

Aligned return is the underlying-stock log return multiplied by +1 for CALL and -1 for PUT. It is not option P&L.

## Baselines

| baseline_id | directional_accuracy | balanced_accuracy_direction | mean_aligned_return | median_aligned_return |
| --- | --- | --- | --- | --- |
| B0_development_prior | 0.436 | 0.5 | -0.0026405858 | -0.0009680785 |
| B1_always_up | 0.564 | 0.5 | 0.0026405858 | 0.0009680785 |
| B2_five_minute_momentum | 0.496 | 0.48448175 | 0.00069821822 | 0 |
| B3_ten_minute_momentum | 0.54 | 0.52348884 | 0.001351596 | 0.00057170625 |
| B4_market_direction | 0.472 | 0.45904093 | -0.0010593091 | -0.00084546198 |
| B5_relative_strength_direction | 0.556 | 0.5387143 | 0.0018669943 | 0.00076834426 |

## Timing and remaining movement

| subgroup | mean_pre_entry_signed_return | median_pre_entry_signed_return | mean_absolute_pre_entry_displacement | mean_remaining_fraction_10m | median_remaining_fraction_10m | episodes_at_least_50pct_remaining_10m | mean_remaining_fraction_30m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_actions | 0.00016079323 | 0.00082678798 | 0.0083537914 | 0.54536559 | 0.56535116 | 0.5974026 | 0.64576674 |

`late_direction_problem = false`.

## Material-movement diagnostics

| subgroup | episodes | accuracy | balanced_accuracy | mean_aligned_return | median_aligned_return | positive_aligned_return_rate | mean_remaining_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ten_minute_iv_excess | 39 | 0.43589744 | 0.43552632 | -0.0025384753 | -0.0079050966 | 0.43589744 | 0.65136065 |
| ten_minute_non_iv_excess | 38 | 0.47368421 | 0.41538462 | -0.00014661316 | -0.00035890186 | 0.47368421 | 0.43658118 |
| largest_absolute_movement_quartile | 16 | 0.4375 | 0.42063492 | -0.0031652103 | -0.016485254 | 0.4375 | 0.70664302 |

## Frozen score bins

| score_bin | episodes | mean_future_signed_return_10m | median_future_signed_return_10m | up_rate | mean_absolute_movement_10m | iv_excess_rate | mean_remaining_fraction_10m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strong_distribution | 58 | 0.0029667194 | 0.002843022 | 0.63157895 | 0.009355624 | 0.53448276 | 0.51315367 |
| moderate_distribution | 48 | 0.0016902183 | 0.00054947268 | 0.55319149 | 0.010905911 | 0.47916667 | 0.50542274 |
| neutral | 46 | 0.0025427912 | -0.00071528637 | 0.47826087 | 0.011591049 | 0.56521739 | 0.59800676 |
| moderate_accumulation | 58 | 0.004376007 | 0.0050956807 | 0.63793103 | 0.011088337 | 0.55172414 | 0.55737095 |
| strong_accumulation | 43 | 0.001025377 | -0.00078950414 | 0.47619048 | 0.0070666194 | 0.34883721 | 0.45647296 |

Correct signed monotonic ordering: `false`; slope `-5.9925746e-05`.

## Grouped permutation attribution

| group_id | log_loss_deterioration | brier_deterioration | auc_deterioration | selective_accuracy_deterioration | mean_aligned_return_deterioration | median_aligned_return_deterioration |
| --- | --- | --- | --- | --- | --- | --- |
| Group_P_persistent_pressure | 0 | 0 | 0 | 0 | 0 | 0 |
| Group_A_absorption_response | 0.0041951088 | 0.0019913908 | 0.0066074566 | -0.021123177 | -0.00076588892 | -0.00020253965 |
| Group_C_compression_context | 9.0833802e-05 | -0.00043056096 | -0.012739931 | -0.013795907 | -0.00041839315 | -0.00023628849 |
| quiet_absorption_score_25 | -0.0015071225 | -0.00096184416 | -0.010833496 | 0.0027100111 | -0.00018105165 | 0.00023615996 |

## Temporal placebo and nulls

| model | log_loss | brier_score | auc | selective_directional_accuracy | selective_mean_aligned_return | confidence_boundary |
| --- | --- | --- | --- | --- | --- | --- |
| real_Q1 | 0.73765152 | 0.26943738 | 0.49417659 | 0.45454545 | -0.0013580758 | 0.13196538 |
| temporally_misaligned_placebo_Q1 | 0.74692939 | 0.27353044 | 0.48675906 | 0.41666667 | -0.0020651332 | 0.1495222 |

Real Q1 outperformed the temporally misaligned placebo under the frozen comparison: `true`.

Five-null comparison: `{"null_gate_passed": false, "real_exceeds_auc": 0, "real_exceeds_brier": 0, "real_exceeds_log_loss": 0, "real_exceeds_log_loss_or_auc": 0, "real_exceeds_mean_aligned_return": 2, "real_exceeds_quiet_score_monotonicity": 5, "real_exceeds_selective_accuracy": 1}`.

## Whole-session bootstrap intervals

| interval_level | metric | bound | value |
| --- | --- | --- | --- |
| 0.8 | q1_minus_q0_log_loss_improvement | lower | -0.012850461 |
| 0.8 | q1_minus_q0_log_loss_improvement | upper | 0.0048613101 |
| 0.8 | q1_minus_q0_brier_improvement | lower | -0.0060000413 |
| 0.8 | q1_minus_q0_brier_improvement | upper | 0.0024832083 |
| 0.8 | q1_minus_q0_auc_improvement | lower | -0.026290553 |
| 0.8 | q1_minus_q0_auc_improvement | upper | 0.017646139 |
| 0.8 | q1_selective_action_coverage | lower | 0.26497224 |
| 0.8 | q1_selective_action_coverage | upper | 0.35007463 |
| 0.8 | q1_selective_accuracy | lower | 0.38264669 |
| 0.8 | q1_selective_accuracy | upper | 0.5312989 |
| 0.8 | q1_selective_balanced_accuracy | lower | 0.3735708 |
| 0.8 | q1_selective_balanced_accuracy | upper | 0.51748515 |
| 0.8 | mean_aligned_ten_minute_return | lower | -0.0030387128 |
| 0.8 | mean_aligned_ten_minute_return | upper | 0.00053691129 |
| 0.8 | median_aligned_ten_minute_return | lower | -0.003354668 |
| 0.8 | median_aligned_ten_minute_return | upper | 0.00058418061 |
| 0.8 | positive_aligned_return_rate | lower | 0.38264669 |
| 0.8 | positive_aligned_return_rate | upper | 0.5312989 |
| 0.8 | mean_remaining_fraction | lower | 0.5074131 |
| 0.8 | mean_remaining_fraction | upper | 0.58884068 |
| 0.8 | quiet_absorption_score_monotonic_slope | lower | -0.0010102122 |
| 0.8 | quiet_absorption_score_monotonic_slope | upper | 0.00063390156 |
| 0.8 | iv_excess_subgroup_accuracy | lower | 0.34602122 |
| 0.8 | iv_excess_subgroup_accuracy | upper | 0.54103194 |
| 0.8 | largest_movement_quartile_accuracy | lower | 0.28380952 |
| 0.8 | largest_movement_quartile_accuracy | upper | 0.58333333 |
| 0.9 | q1_minus_q0_log_loss_improvement | lower | -0.015612849 |
| 0.9 | q1_minus_q0_log_loss_improvement | upper | 0.0072304024 |
| 0.9 | q1_minus_q0_brier_improvement | lower | -0.0072290144 |
| 0.9 | q1_minus_q0_brier_improvement | upper | 0.0036825628 |
| 0.9 | q1_minus_q0_auc_improvement | lower | -0.031380941 |
| 0.9 | q1_minus_q0_auc_improvement | upper | 0.022549786 |
| 0.9 | q1_selective_action_coverage | lower | 0.2569064 |
| 0.9 | q1_selective_action_coverage | upper | 0.36052222 |
| 0.9 | q1_selective_accuracy | lower | 0.35541499 |
| 0.9 | q1_selective_accuracy | upper | 0.55298507 |
| 0.9 | q1_selective_balanced_accuracy | lower | 0.34643668 |
| 0.9 | q1_selective_balanced_accuracy | upper | 0.54179503 |
| 0.9 | mean_aligned_ten_minute_return | lower | -0.0036008173 |
| 0.9 | mean_aligned_ten_minute_return | upper | 0.0014264872 |
| 0.9 | median_aligned_ten_minute_return | lower | -0.0036920983 |
| 0.9 | median_aligned_ten_minute_return | upper | 0.0018521459 |
| 0.9 | positive_aligned_return_rate | lower | 0.35541499 |
| 0.9 | positive_aligned_return_rate | upper | 0.55298507 |
| 0.9 | mean_remaining_fraction | lower | 0.49620535 |
| 0.9 | mean_remaining_fraction | upper | 0.59763317 |
| 0.9 | quiet_absorption_score_monotonic_slope | lower | -0.0013221819 |
| 0.9 | quiet_absorption_score_monotonic_slope | upper | 0.00078177801 |
| 0.9 | iv_excess_subgroup_accuracy | lower | 0.32297297 |
| 0.9 | iv_excess_subgroup_accuracy | upper | 0.56835586 |
| 0.9 | largest_movement_quartile_accuracy | lower | 0.25 |
| 0.9 | largest_movement_quartile_accuracy | upper | 0.60125 |
| 0.95 | q1_minus_q0_log_loss_improvement | lower | -0.017273128 |
| 0.95 | q1_minus_q0_log_loss_improvement | upper | 0.0077938851 |
| 0.95 | q1_minus_q0_brier_improvement | lower | -0.0074288336 |
| 0.95 | q1_minus_q0_brier_improvement | upper | 0.0038939767 |
| 0.95 | q1_minus_q0_auc_improvement | lower | -0.032737697 |
| 0.95 | q1_minus_q0_auc_improvement | upper | 0.03024825 |
| 0.95 | q1_selective_action_coverage | lower | 0.2514645 |
| 0.95 | q1_selective_action_coverage | upper | 0.36312866 |
| 0.95 | q1_selective_accuracy | lower | 0.32507042 |
| 0.95 | q1_selective_accuracy | upper | 0.57537594 |
| 0.95 | q1_selective_balanced_accuracy | lower | 0.31166611 |
| 0.95 | q1_selective_balanced_accuracy | upper | 0.56885121 |
| 0.95 | mean_aligned_ten_minute_return | lower | -0.0043499394 |
| 0.95 | mean_aligned_ten_minute_return | upper | 0.0019416072 |
| 0.95 | median_aligned_ten_minute_return | lower | -0.0046792242 |
| 0.95 | median_aligned_ten_minute_return | upper | 0.0029155202 |
| 0.95 | positive_aligned_return_rate | lower | 0.32507042 |
| 0.95 | positive_aligned_return_rate | upper | 0.57537594 |
| 0.95 | mean_remaining_fraction | lower | 0.48533673 |
| 0.95 | mean_remaining_fraction | upper | 0.60665039 |
| 0.95 | quiet_absorption_score_monotonic_slope | lower | -0.0013471296 |
| 0.95 | quiet_absorption_score_monotonic_slope | upper | 0.00097006213 |
| 0.95 | iv_excess_subgroup_accuracy | lower | 0.27400821 |
| 0.95 | iv_excess_subgroup_accuracy | upper | 0.60930736 |
| 0.95 | largest_movement_quartile_accuracy | lower | 0.18845029 |
| 0.95 | largest_movement_quartile_accuracy | upper | 0.65 |

## Stability, concentration, and support

Positive Q1 selective mean aligned return occurred in 3 of eight month groups.

| metric | value |
| --- | --- |
| maximum_stock_share_of_episodes | 0.11067194 |
| maximum_stock_share_of_actions | 0.1038961 |
| maximum_month_share_of_actions | 0.20779221 |
| maximum_session_share_of_actions | 0.051948052 |

Support gates: `{"assessment": {"down": 109, "episodes": 253, "items": {"all_eight_month_groups": true, "down_at_least_75": true, "episodes_at_least_180": true, "no_month_above_25pct": true, "no_stock_above_15pct": true, "sessions_at_least_45": true, "stocks_at_least_15": true, "up_at_least_75": true}, "maximum_month_share": 0.15019762845849802, "maximum_stock_share": 0.11067193675889328, "months": 8, "passed": true, "sessions": 105, "stocks": 20, "up": 141}, "development": {"down": 142, "episodes": 285, "items": {"down_at_least_90": true, "episodes_at_least_220": true, "months_at_least_10": true, "sessions_at_least_60": true, "stocks_at_least_15": true, "up_at_least_90": true}, "months": 12, "passed": true, "sessions": 145, "stocks": 19, "up": 140}, "selective": {"actions": 77, "calls": 45, "items": {"actions_at_least_80": false, "calls_at_least_25": true, "month_groups_at_least_6": true, "no_month_above_30pct": true, "no_session_above_8pct": true, "no_stock_above_20pct": true, "puts_at_least_25": true, "sessions_at_least_30": true, "stocks_at_least_12": true}, "maximum_month_share": 0.2077922077922078, "maximum_session_share": 0.05194805194805195, "maximum_stock_share": 0.1038961038961039, "months": 8, "passed": false, "puts": 32, "sessions": 57, "stocks": 20}}`.

## Component statuses

- `movement_gate_status`: `supported`
- `episode_reconstruction_status`: `supported`
- `pretrigger_history_status`: `insufficient_support`
- `quietness_status`: `not_supported`
- `persistent_pressure_status`: `insufficient_support`
- `absorption_response_status`: `not_supported`
- `relative_resilience_status`: `not_supported`
- `vwap_defence_status`: `promising`
- `composite_score_status`: `insufficient_support`
- `selective_direction_status`: `insufficient_support`
- `remaining_movement_status`: `supported`
- `prospective_recorder_priority`: `not_supported`

## Primary pass gates

- `1_q1_improves_log_loss_vs_q0`: `false`
- `2_q1_improves_brier_vs_q0`: `false`
- `3_q1_auc_at_least_0_55`: `false`
- `4_q1_balanced_accuracy_above_0_52`: `false`
- `5_selective_coverage_between_20_and_50pct`: `true`
- `6_selective_accuracy_at_least_57pct`: `false`
- `7_selective_accuracy_beats_required_baselines`: `false`
- `8_mean_aligned_return_positive`: `false`
- `9_median_aligned_return_positive`: `false`
- `10_bootstrap_80_accuracy_lower_above_50pct`: `false`
- `11_bootstrap_80_mean_return_lower_nonnegative`: `false`
- `12_positive_mean_return_in_six_of_eight_months`: `false`
- `13_real_exceeds_four_of_five_nulls`: `false`
- `14_real_q1_outperforms_temporal_placebo`: `true`
- `15_score_bins_correct_signed_monotonic_direction`: `false`
- `16_assessment_and_selective_support`: `false`
- `17_concentration_gates`: `true`
- `18_late_direction_problem_false`: `true`

## Claims boundary

This experiment is not institutional-accumulation observation, direct order-flow research, an option profitability study, realistic bid/ask execution, prospective validation, paper readiness, live readiness, or a deployable strategy.
