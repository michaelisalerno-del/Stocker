# M1C Opening Market Transition V1

## Scope and interpretation

This is a fixed retrospective underlying-direction experiment on previously
opened 2025 periods. It is not untouched confirmation, an option-edge test, or
a tradeability claim. M1C, its threshold, its 15-minute endpoint, the frozen
20-stock cohort, freshness, Tail Phase V1, and A1 were not changed.

## Population-accounting audit

Status: `fully_reconciled`.

The prior apparent 9-versus-15 assessment and 29-versus-34 stress differences
were different population definitions, not missing outcomes or a construction
bug. Tail diagnostics admitted every high-M1C `FIRST_ENTRY`/`RE_ENTRY`
checkpoint row. The primary signed-shock study admitted only canonical fresh
episodes. All 11 extra `RE_ENTRY` rows occurred 20 minutes after the preceding
fresh episode and failed the frozen 30-minute spacing rule (6 assessment, 5
stress). The prior scientific conclusion does not change. The prior
`fresh_tail_entries` label was terminologically ambiguous; the exact
episode-ID reconciliation is now explicit.

## Exact checkpoint-6 timing

Checkpoint 6 means six complete five-minute bars. The fixed opening window is
09:30-10:00 New York time: ordinals 0 through 5, with the final included bar
starting 09:55 and completing at the 10:00 signal. Frozen M1C entry is the
10:00 next-bar open. Bar 6 (10:00-10:05), partial bars, future bars, and prior
sessions are excluded. Expected bar count: `6`.

The prior two-window shock experiment excluded checkpoint 6 because W1 needed
a close reference before the regular-session open. This separately versioned
experiment uses only the fixed same-session opening window for its primary
state; the previous regular-session close is used solely to audit the gap and
total-transition identity.

## Canonical VTI and frozen thresholds

The canonical proxy was already available: raw/unadjusted EODHD VTI
five-minute OHLC, UTC bar-start timestamps, final after each five-minute
interval, aligned to the NYSE calendar. No alternative proxy or new dataset was
tested.

- Opening return q10: `-0.00288963733897`
- Opening return q90: `0.00225522676046`
- Opening range q75: `0.00384818171835`
- Overnight gap q10/q90 (descriptive): `-0.00382056890751 / 0.0063796856309`
- Total transition q10/q90 (descriptive): `-0.00536060944383 / 0.00643755517767`
- Complete 2024 predictor support: `247`

The severe state uses only opening return and opening range. Negative is
`return <= q10 and range >= q75`; positive is `return >= q90 and range >=
q75`. Elevated range without either signed tail is nondirectional; other
complete rows are normal. Equality is inclusive.

## Population and structural opening-regime evidence

| period | stage | row_count |
| --- | --- | --- |
| development | canonical_fresh_first_entry_rows | 413 |
| development | complete_market_transition_state | 407 |
| development | complete_stock_opening_response | 407 |
| development | complete_15m_outcome | 413 |
| development | final_eligible_rows | 407 |
| assessment | canonical_fresh_first_entry_rows | 360 |
| assessment | complete_market_transition_state | 356 |
| assessment | complete_stock_opening_response | 356 |
| assessment | complete_15m_outcome | 360 |
| assessment | final_eligible_rows | 356 |
| stress | canonical_fresh_first_entry_rows | 442 |
| stress | complete_market_transition_state | 437 |
| stress | complete_stock_opening_response | 437 |
| stress | complete_15m_outcome | 442 |
| stress | final_eligible_rows | 437 |

| period | severe_stock_episode_count | unique_session_count | unique_opening_transition_event_count | negative_transition_event_count | positive_transition_event_count | complete_normal_opening_event_count | incomplete_event_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development | 65 | 26 | 26 | 23 | 15 | 185 | 5 |
| assessment | 130 | 43 | 43 | 28 | 30 | 72 | 1 |
| stress | 101 | 13 | 13 | 7 | 11 | 56 | 2 |

All primary episode rows are checkpoint-6 `FIRST_ENTRY`. Later `PERSISTENT`
rows and `RE_ENTRY` rows are not independent support.

## Absolute-movement evidence

| period | population | episode_count | unique_transition_event_count | material_move_rate | no_material_move_rate | exceed_iv_rate | mean_absolute_15m_movement | mean_iv_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessment | all_severe_opening_transitions | 130 | 43 | 0.507692 | 0.492308 | 0.507692 | 0.011164 | 0.002264 |
| assessment | normal_opening | 168 | 0 | 0.494048 | 0.505952 | 0.494048 | 0.011237 | 0.002388 |
| assessment | amplifying | 83 | 35 | 0.566265 | 0.433735 | 0.566265 | 0.011885 | 0.003038 |
| assessment | resisting | 47 | 28 | 0.404255 | 0.595745 | 0.404255 | 0.009892 | 0.000898 |
| stress | all_severe_opening_transitions | 101 | 13 | 0.564356 | 0.435644 | 0.564356 | 0.016723 | 0.006993 |
| stress | normal_opening | 267 | 0 | 0.494382 | 0.505618 | 0.494382 | 0.011230 | 0.001980 |
| stress | amplifying | 69 | 13 | 0.565217 | 0.434783 | 0.565217 | 0.018805 | 0.008775 |
| stress | resisting | 32 | 11 | 0.562500 | 0.437500 | 0.562500 | 0.012233 | 0.003150 |

## Directional evidence

| period | mechanism | acted_episode_count | unique_transition_event_count | mean_aligned_return | session_cluster_lower_95 | event_cluster_lower_95 | material_direction_accuracy | accuracy_counting_no_move_as_failure | support_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessment | market_following | 130 | 43 | -0.001365 | -0.005000 | -0.004950 | 0.424242 | 0.215385 | pass |
| assessment | amplification_continuation | 83 | 35 | -0.001541 | -0.006581 | -0.006318 | 0.404255 | 0.228916 | pass |
| assessment | resistance_reversal | 47 | 28 | 0.001053 | -0.003054 | -0.003108 | 0.526316 | 0.212766 | pass |
| stress | market_following | 101 | 13 | -0.010308 | -0.019420 | -0.019104 | 0.175439 | 0.099010 | blocked_insufficient_support |
| stress | amplification_continuation | 69 | 13 | -0.011239 | -0.023166 | -0.022801 | 0.179487 | 0.101449 | blocked_insufficient_support |
| stress | resistance_reversal | 32 | 11 | 0.008300 | 0.003504 | 0.003582 | 0.833333 | 0.468750 | blocked_insufficient_support |

Continuous response ranking:

| period | material_episode_count | roc_auc_followed_opening_transition_v1 | session_cluster_auc_lower_95 | event_cluster_auc_lower_95 | spearman_stock_relative_response_vs_market_follow_return |
| --- | --- | --- | --- | --- | --- |
| assessment | 66 | 0.359962 | 0.203458 | 0.207138 | -0.164047 |
| stress | 57 | 0.351064 | 0.139915 | 0.150522 | -0.327688 |

Severe versus normal opening:

| period | regime | acted_episode_count | material_direction_accuracy | accuracy_counting_no_move_as_failure | mean_market_aligned_return | no_move_rate | iv_excess_rate | mean_absolute_movement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessment | severe_opening | 130 | 0.424242 | 0.215385 | -0.001365 | 0.492308 | 0.507692 | 0.011164 |
| assessment | normal_opening | 168 | 0.457831 | 0.226190 | 0.000187 | 0.505952 | 0.494048 | 0.011237 |
| assessment | severe_minus_normal | 298 | -0.033589 | -0.010806 | -0.001552 | -0.013645 | 0.013645 | -0.000073 |
| stress | severe_opening | 101 | 0.175439 | 0.099010 | -0.010308 | 0.435644 | 0.564356 | 0.016723 |
| stress | normal_opening | 267 | 0.477273 | 0.235955 | -0.000623 | 0.505618 | 0.494382 | 0.011230 |
| stress | severe_minus_normal | 368 | -0.301834 | -0.136945 | -0.009685 | -0.069974 | 0.069974 | 0.005493 |

Selected same-timestamp baselines:

| period | evaluation_scope | policy | acted_episode_count | mean_aligned_return | material_direction_accuracy | accuracy_counting_no_move_as_failure |
| --- | --- | --- | --- | --- | --- | --- |
| assessment | amplification_acted_episodes | follow_vti_opening_transition | 83 | -0.001541 | 0.404255 | 0.228916 |
| assessment | amplification_acted_episodes | recent_stock_5m_momentum | 81 | 0.000921 | 0.456522 | 0.259259 |
| assessment | amplification_acted_episodes | stock_opening_window_momentum | 83 | -0.001541 | 0.404255 | 0.228916 |
| assessment | amplification_acted_episodes | frozen_A1 | 33 | 0.003639 | 0.681818 | 0.454545 |
| assessment | amplification_acted_episodes | existing_clean_market_direction_baseline | 83 | -0.002889 | 0.340426 | 0.192771 |
| assessment | resistance_acted_episodes | follow_vti_opening_transition | 47 | -0.001053 | 0.473684 | 0.191489 |
| assessment | resistance_acted_episodes | recent_stock_5m_momentum | 47 | 0.000300 | 0.368421 | 0.148936 |
| assessment | resistance_acted_episodes | stock_opening_window_momentum | 46 | 0.001898 | 0.578947 | 0.239130 |
| assessment | resistance_acted_episodes | frozen_A1 | 16 | -0.000303 | 0.500000 | 0.250000 |
| assessment | resistance_acted_episodes | existing_clean_market_direction_baseline | 47 | 0.000695 | 0.578947 | 0.234043 |
| stress | amplification_acted_episodes | follow_vti_opening_transition | 69 | -0.011239 | 0.179487 | 0.101449 |
| stress | amplification_acted_episodes | recent_stock_5m_momentum | 68 | -0.010277 | 0.210526 | 0.117647 |
| stress | amplification_acted_episodes | stock_opening_window_momentum | 69 | -0.011239 | 0.179487 | 0.101449 |
| stress | amplification_acted_episodes | frozen_A1 | 27 | 0.000305 | 0.538462 | 0.259259 |
| stress | amplification_acted_episodes | existing_clean_market_direction_baseline | 69 | -0.012286 | 0.179487 | 0.101449 |
| stress | resistance_acted_episodes | follow_vti_opening_transition | 32 | -0.008300 | 0.166667 | 0.093750 |
| stress | resistance_acted_episodes | recent_stock_5m_momentum | 32 | -0.000716 | 0.388889 | 0.218750 |
| stress | resistance_acted_episodes | stock_opening_window_momentum | 32 | 0.005464 | 0.666667 | 0.375000 |
| stress | resistance_acted_episodes | frozen_A1 | 22 | 0.001753 | 0.625000 | 0.227273 |
| stress | resistance_acted_episodes | existing_clean_market_direction_baseline | 32 | -0.004747 | 0.277778 | 0.156250 |

Frozen D2 is `blocked_contaminated_or_unreproducible_lineage`; it was not
approximately reconstructed. No option outcome was used.

## Nulls, placebo, and shared-event uncertainty

The primary null used `1000` fixed-seed reassignments to a different
session while preserving stock, checkpoint 6, period, and outcome
completeness. The temporal placebo used the next eligible checkpoint-6 fresh
`FIRST_ENTRY` outcome for the same stock and period. Assessment null p-values:
market following `0.841159`, amplification `0.803197`,
resistance `0.315684`.
Holm-adjusted amplification/resistance p-values are
`{"amplification_continuation": 0.8031968031968032, "resistance_reversal": 0.6313686313686314}`.

Session and whole opening-transition-event bootstraps each used
`5000` replications. At checkpoint 6 each severe event is one
session/sign, so the two cluster units coincide for severe-only mechanisms;
both are nevertheless persisted and the more conservative conclusion governs.
Null/placebo metadata: `{"assessment": {"chronology_crossed": false, "pair_count": 336, "period": "assessment", "protected_boundary_crossed": false, "same_checkpoint": true, "same_period": true, "same_stock": true, "source_episode_count": 356, "statistics": {"amplification_accuracy_including_no_move": 0.24675324675324675, "amplification_material_direction_accuracy": 0.4418604651162791, "amplification_mean_aligned_return": -0.0008158144064303169, "amplifying_minus_resisting_follow_rate": 0.06686046511627908, "continuous_ranking_auc": 0.5787545787545787, "market_follow_accuracy_including_no_move": 0.22764227642276422, "market_follow_material_direction_accuracy": 0.417910447761194, "market_follow_mean_aligned_return": -0.0015155926561428279, "resistance_accuracy_including_no_move": 0.32608695652173914, "resistance_material_direction_accuracy": 0.625, "resistance_mean_aligned_return": 0.0026869605958789876, "severe_minus_normal_accuracy_including_no_move": -0.05440900562851783, "severe_minus_normal_mean_market_aligned_return": -0.0035816777322874227}}, "stress": {"chronology_crossed": false, "pair_count": 417, "period": "stress", "protected_boundary_crossed": false, "same_checkpoint": true, "same_period": true, "same_stock": true, "source_episode_count": 437, "statistics": {"amplification_accuracy_including_no_move": 0.19117647058823528, "amplification_material_direction_accuracy": 0.38235294117647056, "amplification_mean_aligned_return": -0.0027089736341101797, "amplifying_minus_resisting_follow_rate": -0.1470588235294118, "continuous_ranking_auc": 0.42946708463949845, "market_follow_accuracy_including_no_move": 0.2222222222222222, "market_follow_material_direction_accuracy": 0.43137254901960786, "market_follow_mean_aligned_return": -0.0014515998001699741, "resistance_accuracy_including_no_move": 0.25806451612903225, "resistance_material_direction_accuracy": 0.47058823529411764, "resistance_mean_aligned_return": -0.001306510545247252, "severe_minus_normal_accuracy_including_no_move": -0.08977777777777779, "severe_minus_normal_mean_market_aligned_return": -0.003399837377401884}}}`.

## Decisions

- Broad market following: `blocked_insufficient_support`
- Amplification continuation: `blocked_insufficient_support`
- Resistance reversal: `blocked_insufficient_support`
- Overall: `blocked_insufficient_support`

These are separate frozen mechanisms. They were not merged and the better arm
was not selected as a policy.

## Tail Phase

The experiment addresses the dominant checkpoint-6 `FIRST_ENTRY` population
excluded by the prior two-window design. Tail Phase V1 is attached unchanged
for provenance. No phase gate, phase interaction, later persistence, or
`RE_ENTRY` support is used.

## Prospective recorder integration

The existing IBKR recorder stores the frozen VTI opening fields after core
checkpoint and episode processing. The integration is logging-only and
failure-contained; it does not change M1C scoring, episode inclusion,
promotion priority, subscriptions, recorder capacity, option selection,
direction decisions, or routing. The first 20 transfer sessions remain
`engineering_transfer` and cannot recalibrate these definitions.

## Option profitability

**Not tested**

Prior-close ATM IV is used only for the unchanged strict M1C movement threshold
and the stock-local response scale.

## Execution realism and remaining unknowns

Five-minute historical bars cannot observe bid withdrawal, ask withdrawal,
replenishment, trade impact, spread changes, queue behaviour, executable
option outcomes, slippage, fill probability, market impact, or prospective
behavioural stability. No broker was accessed, no routing path was enabled,
and no order was placed.

## Operational blockers

Pipeline/data blockers are separated from scientific negative results. The
canonical VTI source, checkpoint-6 bars, stock bars, IV scales, and outcomes
were available for the reported eligible rows. Protected 2026 outcomes were
not opened, calculated, inspected, or displayed.
