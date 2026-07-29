# M1C Signed Market Shock Transition V1

Research-only retrospective report. The 2025 assessment and stress periods were opened by earlier research; this is out-of-development evidence, not untouched confirmation.

## Decisions

- Continuation: `blocked_insufficient_support`
- Resistance: `blocked_insufficient_support`
- Overall: `blocked_insufficient_support`
- Option profitability: **Not tested**
- Tradeability: not claimed

## Structural regime evidence

A canonical causal market proxy was available: **VTI**, from the existing EODHD five-minute market-direction baseline. Bars are raw/unadjusted OHLC, timestamped at bar start in UTC, final only five minutes later, and aligned to the NYSE regular-session calendar. No alternative proxy was tested.

The bounded proxy audit compared 19,860 archived causal bar observations; maximum return and range differences were 0 and 0.

Checkpoint 6 is `UNKNOWN_INCOMPLETE`: W1 would require a previous-session reference. It was not pooled, imputed, or given a fallback.

| period | sessions_with_complete_market_data | scheduled_sessions | unique_negative_shock_onsets | unique_positive_shock_onsets | ongoing_shocks | elevated_range_nondirectional_states | unique_market_shock_event_ids | mean_stocks_per_shock_event | incomplete_fresh_episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | 248 | 252 | 260 | 210 | 89 | 323 | 470 | 0.0170213 | 413 |
| assessment | 159 | 160 | 232 | 244 | 154 | 362 | 476 | 0.0189076 | 360 |
| stress | 85 | 85 | 97 | 90 | 44 | 153 | 187 | 0.15508 | 442 |

### Exact fixed market windows

- W0 uses market bars `checkpoint-3` through `checkpoint-1`, with return `log(close[checkpoint-1] / close[checkpoint-4])`.
- W1 uses market bars `checkpoint-6` through `checkpoint-4`, with return `log(close[checkpoint-4] / close[checkpoint-7])`.
- Ranges are `log(max(high) / min(low))` within the respective three bars.
- Both windows end at or before the M1C signal; neither contains the next-bar-open entry bar.

### Frozen 2024 predictor-only thresholds

| checkpoint | market_return_w0_q10_v1 | market_return_w0_q90_v1 | market_range_w0_q75_v1 | market_return_w1_q10_v1 | market_return_w1_q90_v1 | market_range_w1_q75_v1 | market_return_w0_support_v1 | market_range_w0_support_v1 | market_return_w1_support_v1 | market_range_w1_support_v1 | calibration_complete_v1 | calibration_missing_reason_v1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | NA | NA | NA | NA | NA | NA | 0 | 0 | 0 | 0 | False | insufficient_predictor_support:market_return_w0_v1=0,market_range_w0_v1=0,market_return_w1_v1=0,market_range_w1_v1=0 |
| 8 | -0.00190193 | 0.0020241 | 0.00295751 | -0.00213438 | 0.00153341 | 0.0026537 | 251 | 251 | 251 | 251 | True | NA |
| 10 | -0.00183828 | 0.00137763 | 0.00268315 | -0.00229991 | 0.00227197 | 0.00310779 | 250 | 250 | 250 | 250 | True | NA |
| 12 | -0.00153017 | 0.00157542 | 0.00237366 | -0.00228875 | 0.00194901 | 0.0030122 | 250 | 250 | 250 | 250 | True | NA |
| 14 | -0.00169867 | 0.00155297 | 0.0024114 | -0.00133559 | 0.00157576 | 0.00240289 | 250 | 250 | 250 | 250 | True | NA |
| 16 | -0.00145824 | 0.00144104 | 0.00215025 | -0.00167549 | 0.00155711 | 0.00237366 | 250 | 250 | 250 | 250 | True | NA |
| 18 | -0.00145938 | 0.00114791 | 0.00204763 | -0.00152609 | 0.00161914 | 0.00225702 | 252 | 252 | 252 | 252 | True | NA |
| 20 | -0.00130535 | 0.00131969 | 0.00205366 | -0.00176245 | 0.00125606 | 0.00205936 | 252 | 252 | 252 | 252 | True | NA |
| 22 | -0.00125398 | 0.0013369 | 0.00191556 | -0.00164859 | 0.00140261 | 0.00209289 | 251 | 251 | 251 | 251 | True | NA |
| 24 | -0.00114952 | 0.00115699 | 0.00170207 | -0.00147203 | 0.00143878 | 0.00200458 | 251 | 251 | 251 | 251 | True | NA |
| 26 | -0.00112316 | 0.00119781 | 0.00174443 | -0.0011497 | 0.00116816 | 0.00171856 | 251 | 251 | 251 | 251 | True | NA |
| 28 | -0.0012971 | 0.00110547 | 0.00169576 | -0.00123748 | 0.00122549 | 0.00174896 | 251 | 251 | 251 | 251 | True | NA |
| 30 | -0.0010989 | 0.00117863 | 0.00168588 | -0.00115371 | 0.00102777 | 0.00170579 | 251 | 251 | 251 | 251 | True | NA |
| 32 | -0.00122264 | 0.00114972 | 0.0016847 | -0.00104763 | 0.00114533 | 0.0016972 | 251 | 251 | 251 | 251 | True | NA |
| 34 | -0.00099864 | 0.000897949 | 0.00153296 | -0.00115021 | 0.00118969 | 0.00168612 | 251 | 251 | 251 | 251 | True | NA |

The definitions use inclusive 10th/90th signed-return tails and the inclusive 75th-percentile range boundary. Current shocks absent the same prior shock are onsets; repeated same-signed shocks are ongoing. Elevated range without a signed tail is nondirectional; all other complete rows are normal.

## Absolute-movement evidence

- Assessment: checkpoint-standardised shock no-move 0.888889 versus normal 0.805556; IV-excess 0.111111 versus 0.194444; mean absolute movement 0.009135 versus 0.004960.
- Stress: checkpoint-standardised shock no-move 0.680952 versus normal 0.651361; IV-excess 0.319048 versus 0.348639; mean absolute movement 0.008538 versus 0.008959.

No-move outcomes remain in every action-policy denominator where specified.

## Directional evidence

The stock response is fixed as `shock_sign × (stock_return_w0 - market_return_w0) / threshold_15m`. Positive is AMPLIFYING; negative is RESISTING; exact zero is NEUTRAL_EXACT. The resistance and continuation policies remain separate.
The assessment/stress outcome artifacts include both descriptive RESISTING subtypes and every response class within each shock sign; `checkpoint_stratified_mechanism_results_v1.csv` reports every frozen checkpoint without selection.

### Continuation arm

| period | acted_episode_count | unique_shock_event_count | session_count | stock_count | call_count | put_count | no_material_move_count | material_direction_accuracy | accuracy_counting_no_move_as_failure | mean_aligned_return | session_cluster_lower_95 | shock_event_cluster_lower_95 | one_percent_winsorised_mean_aligned_return | all_shock_follow_mean_aligned_return | amplifying_selection_mean_return_delta_vs_all_shock_follow | positive_session_rate | positive_month_rate | support_status | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessment | 7 | 7 | 7 | 5 | 6 | 1 | 3 | 0.5 | 0.285714 | -0.000305169 | -0.0182111 | -0.018665 | -0.000234325 | 0.000197731 | -0.0005029 | 0.571429 | 0.5 | blocked_insufficient_support | blocked_insufficient_support |
| stress | 25 | 19 | 16 | 11 | 8 | 17 | 15 | 0.4 | 0.16 | -0.0023131 | -0.00755712 | -0.0076165 | -0.00234092 | -0.00159043 | -0.000722675 | 0.625 | 0.5 | blocked_insufficient_support | blocked_insufficient_support |

### Resistance arm

| period | acted_episode_count | unique_shock_event_count | session_count | stock_count | call_count | put_count | no_material_move_count | material_direction_accuracy | accuracy_counting_no_move_as_failure | mean_aligned_return | session_cluster_lower_95 | shock_event_cluster_lower_95 | one_percent_winsorised_mean_aligned_return | follow_market_same_acted_mean_aligned_return | resistance_minus_follow_market_same_acted_mean_return | positive_session_rate | positive_month_rate | support_status | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessment | 2 | 2 | 2 | 2 | 2 | 0 | 2 | NA | 0 | -0.00195788 | -0.00925964 | -0.00925964 | -0.00195788 | 0.00195788 | -0.00391576 | 0.5 | 0.5 | blocked_insufficient_support | blocked_insufficient_support |
| stress | 4 | 3 | 3 | 3 | 4 | 0 | 2 | 0.5 | 0.25 | -0.00292629 | -0.0295427 | -0.0295427 | -0.00281909 | 0.00292629 | -0.00585258 | 0.666667 | 0.5 | blocked_insufficient_support | blocked_insufficient_support |

### Continuous ranking

| period | eligible_shock_onset_episode_count | material_mover_count | unique_shock_event_count | session_count | roc_auc_followed_shock_among_material_movers | spearman_response_vs_continuation_aligned_return | spearman_two_sided_p_value | session_cluster_auc_lower_95 | session_cluster_auc_upper_95 | shock_event_cluster_auc_lower_95 | shock_event_cluster_auc_upper_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessment | 9 | 4 | 9 | 9 | 0 | -0.4 | 0.286105 | 0 | 0 | 0 | 0 |
| stress | 29 | 12 | 21 | 16 | 0.314286 | -0.196552 | 0.30682 | 0 | 1 | 0.0137338 | 0.946806 |

### Sign consistency

| period | shock_state | arm | episode_count | session_count | unique_shock_event_count | stock_count | mean_aligned_return | material_direction_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessment | NEGATIVE_SHOCK_ONSET | continuation | 1 | 1 | 1 | 1 | 0.00684147 | NA |
| assessment | NEGATIVE_SHOCK_ONSET | resistance | 2 | 2 | 2 | 2 | -0.00195788 | NA |
| assessment | POSITIVE_SHOCK_ONSET | continuation | 6 | 6 | 6 | 4 | -0.00149628 | 0.5 |
| assessment | POSITIVE_SHOCK_ONSET | resistance | 0 | 0 | 0 | 0 | NA | NA |
| stress | NEGATIVE_SHOCK_ONSET | continuation | 17 | 11 | 13 | 7 | -0.00332415 | 0.444444 |
| stress | NEGATIVE_SHOCK_ONSET | resistance | 4 | 3 | 3 | 3 | -0.00292629 | 0.5 |
| stress | POSITIVE_SHOCK_ONSET | continuation | 8 | 6 | 6 | 6 | -0.000164625 | 0 |
| stress | POSITIVE_SHOCK_ONSET | resistance | 0 | 0 | 0 | 0 | NA | NA |

### Fixed baselines

| period | evaluation_scope | policy | acted_episode_count | unique_shock_event_count | material_direction_accuracy | accuracy_counting_no_move_as_failure | mean_aligned_return | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessment | all_signed_shock_onsets | follow_market_shock | 9 | 9 | 0.5 | 0.222222 | 0.000197731 | NA |
| assessment | all_signed_shock_onsets | oppose_market_shock | 9 | 9 | 0.5 | 0.222222 | -0.000197731 | NA |
| assessment | all_signed_shock_onsets | recent_stock_momentum_5m | 9 | 9 | 0.25 | 0.111111 | -0.00806675 | NA |
| assessment | all_signed_shock_onsets | trailing_stock_momentum_15m | 8 | 8 | 0.5 | 0.25 | 0.000400961 | NA |
| assessment | all_signed_shock_onsets | frozen_A1 | 7 | 7 | 0.5 | 0.285714 | 0.00134651 | NA |
| assessment | all_signed_shock_onsets | always_CALL | 9 | 9 | 0.5 | 0.222222 | -0.00219276 | NA |
| assessment | all_signed_shock_onsets | always_PUT | 9 | 9 | 0.5 | 0.222222 | 0.00219276 | NA |
| assessment | continuation_amplifying_acted | follow_market_shock | 7 | 7 | 0.5 | 0.285714 | -0.000305169 | NA |
| assessment | continuation_amplifying_acted | oppose_market_shock | 7 | 7 | 0.5 | 0.285714 | 0.000305169 | NA |
| assessment | continuation_amplifying_acted | recent_stock_momentum_5m | 7 | 7 | 0.25 | 0.142857 | -0.0124578 | NA |
| assessment | continuation_amplifying_acted | trailing_stock_momentum_15m | 7 | 7 | 0.5 | 0.285714 | -0.000305169 | NA |
| assessment | continuation_amplifying_acted | frozen_A1 | 5 | 5 | 0.5 | 0.4 | 0.00110196 | NA |
| assessment | continuation_amplifying_acted | always_CALL | 7 | 7 | 0.5 | 0.285714 | -0.00225987 | NA |
| assessment | continuation_amplifying_acted | always_PUT | 7 | 7 | 0.5 | 0.285714 | 0.00225987 | NA |
| assessment | continuation_amplifying_acted | continuation_v1 | 7 | 7 | 0.5 | 0.285714 | -0.000305169 | NA |
| assessment | resistance_resisting_acted | follow_market_shock | 2 | 2 | NA | 0 | 0.00195788 | NA |
| assessment | resistance_resisting_acted | oppose_market_shock | 2 | 2 | NA | 0 | -0.00195788 | NA |
| assessment | resistance_resisting_acted | recent_stock_momentum_5m | 2 | 2 | NA | 0 | 0.00730175 | NA |
| assessment | resistance_resisting_acted | trailing_stock_momentum_15m | 1 | 1 | NA | 0 | 0.00534387 | NA |
| assessment | resistance_resisting_acted | frozen_A1 | 2 | 2 | NA | 0 | 0.00195788 | NA |
| assessment | resistance_resisting_acted | always_CALL | 2 | 2 | NA | 0 | -0.00195788 | NA |
| assessment | resistance_resisting_acted | always_PUT | 2 | 2 | NA | 0 | 0.00195788 | NA |
| assessment | resistance_resisting_acted | resistance_v1 | 2 | 2 | NA | 0 | -0.00195788 | NA |
| assessment | all_signed_shock_onsets | existing_frozen_D2 | NA | NA | NA | NA | NA | blocked_contaminated_or_unreproducible_lineage |
| assessment | continuation_amplifying_acted | existing_frozen_D2 | NA | NA | NA | NA | NA | blocked_contaminated_or_unreproducible_lineage |
| assessment | resistance_resisting_acted | existing_frozen_D2 | NA | NA | NA | NA | NA | blocked_contaminated_or_unreproducible_lineage |
| stress | all_signed_shock_onsets | follow_market_shock | 29 | 21 | 0.416667 | 0.172414 | -0.00159043 | NA |
| stress | all_signed_shock_onsets | oppose_market_shock | 29 | 21 | 0.583333 | 0.241379 | 0.00159043 | NA |
| stress | all_signed_shock_onsets | recent_stock_momentum_5m | 29 | 21 | 0.416667 | 0.172414 | -0.00280303 | NA |
| stress | all_signed_shock_onsets | trailing_stock_momentum_15m | 28 | 21 | 0.363636 | 0.142857 | -0.00314751 | NA |
| stress | all_signed_shock_onsets | frozen_A1 | 15 | 14 | 0.833333 | 0.333333 | 0.00433211 | NA |
| stress | all_signed_shock_onsets | always_CALL | 29 | 21 | 0.5 | 0.206897 | 0.0014996 | NA |
| stress | all_signed_shock_onsets | always_PUT | 29 | 21 | 0.5 | 0.206897 | -0.0014996 | NA |
| stress | continuation_amplifying_acted | follow_market_shock | 25 | 19 | 0.4 | 0.16 | -0.0023131 | NA |
| stress | continuation_amplifying_acted | oppose_market_shock | 25 | 19 | 0.6 | 0.24 | 0.0023131 | NA |
| stress | continuation_amplifying_acted | recent_stock_momentum_5m | 25 | 19 | 0.4 | 0.16 | -0.00276194 | NA |
| stress | continuation_amplifying_acted | trailing_stock_momentum_15m | 25 | 19 | 0.4 | 0.16 | -0.0023131 | NA |
| stress | continuation_amplifying_acted | frozen_A1 | 13 | 12 | 0.75 | 0.230769 | 0.00145351 | NA |
| stress | continuation_amplifying_acted | always_CALL | 25 | 19 | 0.5 | 0.2 | 0.00220774 | NA |
| stress | continuation_amplifying_acted | always_PUT | 25 | 19 | 0.5 | 0.2 | -0.00220774 | NA |
| stress | continuation_amplifying_acted | continuation_v1 | 25 | 19 | 0.4 | 0.16 | -0.0023131 | NA |
| stress | resistance_resisting_acted | follow_market_shock | 4 | 3 | 0.5 | 0.25 | 0.00292629 | NA |
| stress | resistance_resisting_acted | oppose_market_shock | 4 | 3 | 0.5 | 0.25 | -0.00292629 | NA |
| stress | resistance_resisting_acted | recent_stock_momentum_5m | 4 | 3 | 0.5 | 0.25 | -0.00305985 | NA |
| stress | resistance_resisting_acted | trailing_stock_momentum_15m | 3 | 3 | 0 | 0 | -0.0101009 | NA |
| stress | resistance_resisting_acted | frozen_A1 | 2 | 2 | 1 | 1 | 0.023043 | NA |
| stress | resistance_resisting_acted | always_CALL | 4 | 3 | 0.5 | 0.25 | -0.00292629 | NA |
| stress | resistance_resisting_acted | always_PUT | 4 | 3 | 0.5 | 0.25 | 0.00292629 | NA |
| stress | resistance_resisting_acted | resistance_v1 | 4 | 3 | 0.5 | 0.25 | -0.00292629 | NA |
| stress | all_signed_shock_onsets | existing_frozen_D2 | NA | NA | NA | NA | NA | blocked_contaminated_or_unreproducible_lineage |
| stress | continuation_amplifying_acted | existing_frozen_D2 | NA | NA | NA | NA | NA | blocked_contaminated_or_unreproducible_lineage |
| stress | resistance_resisting_acted | existing_frozen_D2 | NA | NA | NA | NA | NA | blocked_contaminated_or_unreproducible_lineage |

Frozen D2 is blocked because its clean reproducible lineage is not available; it was not approximately reconstructed.

### Null and placebo

The primary null reassigns each outcome within stock, checkpoint, and period to a different session. It used 1,000 fixed-seed replications. The temporal placebo uses the next eligible fresh high-M1C outcome for the same stock/checkpoint/period without crossing chronology.

```json
{
  "assessment_holm_adjusted_p_values": {
    "continuation": 0.9630369630369631,
    "resistance": 0.9630369630369631
  },
  "assessment_observed": {
    "continuation": -0.0003051693023393129,
    "resistance": -0.001957882315141551
  },
  "assessment_one_sided_raw_p_values": {
    "continuation": 0.48151848151848153,
    "resistance": 0.6403596403596403
  },
  "primary_null_audit": [
    {
      "donor_rows": 13914,
      "draws": 1000,
      "eligible_predictor_rows": 9,
      "grouping": [
        "stock",
        "checkpoint",
        "period"
      ],
      "outcome_completeness_preserved": true,
      "period": "assessment",
      "same_session_allowed": false,
      "seed": 2026072805
    },
    {
      "donor_rows": 18136,
      "draws": 1000,
      "eligible_predictor_rows": 29,
      "grouping": [
        "stock",
        "checkpoint",
        "period"
      ],
      "outcome_completeness_preserved": true,
      "period": "stress",
      "same_session_allowed": false,
      "seed": 2026072805
    }
  ],
  "temporal_placebo": [
    {
      "amplifying_minus_resisting_follow_shock_rate": null,
      "construction": "next_eligible_fresh_high_m1c_outcome_same_stock_checkpoint_period",
      "continuation_material_direction_accuracy": null,
      "continuation_mean_aligned_return": 0.0054371397892856235,
      "continuous_ranking_auc": null,
      "cross_chronology_allowed": false,
      "paired_episode_count": 4,
      "period": "assessment",
      "resistance_material_direction_accuracy": null,
      "resistance_mean_aligned_return": -0.0029375110070023257
    },
    {
      "amplifying_minus_resisting_follow_shock_rate": -0.5,
      "construction": "next_eligible_fresh_high_m1c_outcome_same_stock_checkpoint_period",
      "continuation_material_direction_accuracy": 0.5,
      "continuation_mean_aligned_return": -0.001439351041278703,
      "continuous_ranking_auc": 0.5,
      "cross_chronology_allowed": false,
      "paired_episode_count": 13,
      "period": "stress",
      "resistance_material_direction_accuracy": 0.0,
      "resistance_mean_aligned_return": -0.01071946302010941
    }
  ]
}
```

Both session-cluster and shared-shock-event-cluster uncertainty govern the decisions; the more conservative conclusion is used.

## Normal-regime comparison

| period | regime | aggregation | episode_count | session_count | checkpoint_count | conditional_direction_accuracy_among_material_movers | accuracy_counting_no_move_as_failure | mean_market_aligned_15m_return | no_move_rate | iv_excess_rate | mean_absolute_15m_movement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessment | SIGNED_SHOCK_ONSET | raw | 9 | 9 | 6 | 0.5 | 0.222222 | 0.000197731 | 0.555556 | 0.444444 | 0.0143653 |
| assessment | NORMAL_OTHER | raw | 25 | 20 | 6 | 0.333333 | 0.12 | -0.00126079 | 0.64 | 0.36 | 0.0075268 |
| assessment | SIGNED_SHOCK_ONSET | equal_weight_common_checkpoint_standardised | 6 | NA | 3 | 1 | 0.111111 | 0.00749541 | 0.888889 | 0.111111 | 0.00913461 |
| assessment | NORMAL_OTHER | equal_weight_common_checkpoint_standardised | 20 | NA | 3 | 0.1 | 0.0222222 | -0.00167451 | 0.805556 | 0.194444 | 0.00495972 |
| stress | SIGNED_SHOCK_ONSET | raw | 29 | 16 | 9 | 0.416667 | 0.172414 | -0.00159043 | 0.586207 | 0.413793 | 0.00943601 |
| stress | NORMAL_OTHER | raw | 34 | 22 | 8 | 0.384615 | 0.147059 | -0.000750318 | 0.617647 | 0.382353 | 0.00893442 |
| stress | SIGNED_SHOCK_ONSET | equal_weight_common_checkpoint_standardised | 27 | NA | 7 | 0.288889 | 0.0619048 | -0.00384651 | 0.680952 | 0.319048 | 0.00853846 |
| stress | NORMAL_OTHER | equal_weight_common_checkpoint_standardised | 32 | NA | 7 | 0.333333 | 0.0884354 | -0.00264857 | 0.651361 | 0.348639 | 0.00895881 |

This is an exact/equal-weight checkpoint adjustment, not a fitted model and not favourable-checkpoint selection.

## Tail Phase evidence

Tail Phase V1 is attached unchanged and is descriptive only. FIRST_ENTRY and RE_ENTRY retain fresh-episode interpretation. PERSISTENT rows are labelled as dependent checkpoint observations and never counted as independent primary episode support. No phase gate or interaction was fitted.

| period | market_shock_state_v1 | tail_phase_v1 | policy | eligible_episode_count | acted_episode_count | unique_shock_event_count | material_direction_accuracy | mean_aligned_return | support_status | phase_population | checkpoint_distribution_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessment | NEGATIVE_SHOCK_ONSET | FIRST_ENTRY | continuation_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| assessment | NEGATIVE_SHOCK_ONSET | FIRST_ENTRY | resistance_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| assessment | NEGATIVE_SHOCK_ONSET | FIRST_ENTRY | frozen_A1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| assessment | NEGATIVE_SHOCK_ONSET | PERSISTENT | continuation_v1 | 47 | 35 | 20 | 0.684211 | 0.00182691 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":11,"rate":0.23404255319148937},"12":{"count":6,"rate":0.1276595744680851},"14":{"count":3,"rate":0.06382978723404255},"16":{"count":1,"rate":0.02127659574468085},"18":{"count":2,"rate":0.0425531914893617},"20":{"count":3,"rate":0.06382978723404255},"22":{"count":2,"rate":0.0425531914893617},"24":{"count":4,"rate":0.0851063829787234},"26":{"count":1,"rate":0.02127659574468085},"8":{"count":14,"rate":0.2978723404255319}} |
| assessment | NEGATIVE_SHOCK_ONSET | PERSISTENT | resistance_v1 | 47 | 12 | 11 | 0.857143 | 0.00777156 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":11,"rate":0.23404255319148937},"12":{"count":6,"rate":0.1276595744680851},"14":{"count":3,"rate":0.06382978723404255},"16":{"count":1,"rate":0.02127659574468085},"18":{"count":2,"rate":0.0425531914893617},"20":{"count":3,"rate":0.06382978723404255},"22":{"count":2,"rate":0.0425531914893617},"24":{"count":4,"rate":0.0851063829787234},"26":{"count":1,"rate":0.02127659574468085},"8":{"count":14,"rate":0.2978723404255319}} |
| assessment | NEGATIVE_SHOCK_ONSET | PERSISTENT | frozen_A1 | 47 | 20 | 15 | 0.454545 | -0.00332599 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":11,"rate":0.23404255319148937},"12":{"count":6,"rate":0.1276595744680851},"14":{"count":3,"rate":0.06382978723404255},"16":{"count":1,"rate":0.02127659574468085},"18":{"count":2,"rate":0.0425531914893617},"20":{"count":3,"rate":0.06382978723404255},"22":{"count":2,"rate":0.0425531914893617},"24":{"count":4,"rate":0.0851063829787234},"26":{"count":1,"rate":0.02127659574468085},"8":{"count":14,"rate":0.2978723404255319}} |
| assessment | NEGATIVE_SHOCK_ONSET | RE_ENTRY | continuation_v1 | 6 | 4 | 3 | 0 | -0.00902321 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":3,"rate":0.5},"14":{"count":1,"rate":0.16666666666666666},"20":{"count":1,"rate":0.16666666666666666},"22":{"count":1,"rate":0.16666666666666666}} |
| assessment | NEGATIVE_SHOCK_ONSET | RE_ENTRY | resistance_v1 | 6 | 2 | 2 | NA | -0.00195788 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":3,"rate":0.5},"14":{"count":1,"rate":0.16666666666666666},"20":{"count":1,"rate":0.16666666666666666},"22":{"count":1,"rate":0.16666666666666666}} |
| assessment | NEGATIVE_SHOCK_ONSET | RE_ENTRY | frozen_A1 | 6 | 4 | 4 | NA | -5.54198e-05 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":3,"rate":0.5},"14":{"count":1,"rate":0.16666666666666666},"20":{"count":1,"rate":0.16666666666666666},"22":{"count":1,"rate":0.16666666666666666}} |
| assessment | POSITIVE_SHOCK_ONSET | FIRST_ENTRY | continuation_v1 | 1 | 1 | 1 | 0 | -0.0460008 | blocked_insufficient_support | fresh_tail_entries | {"8":{"count":1,"rate":1.0}} |
| assessment | POSITIVE_SHOCK_ONSET | FIRST_ENTRY | resistance_v1 | 1 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"8":{"count":1,"rate":1.0}} |
| assessment | POSITIVE_SHOCK_ONSET | FIRST_ENTRY | frozen_A1 | 1 | 1 | 1 | 1 | 0.0460008 | blocked_insufficient_support | fresh_tail_entries | {"8":{"count":1,"rate":1.0}} |
| assessment | POSITIVE_SHOCK_ONSET | PERSISTENT | continuation_v1 | 44 | 31 | 24 | 0.722222 | 0.00559359 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":9,"rate":0.20454545454545456},"12":{"count":4,"rate":0.09090909090909091},"14":{"count":4,"rate":0.09090909090909091},"16":{"count":4,"rate":0.09090909090909091},"18":{"count":1,"rate":0.022727272727272728},"20":{"count":6,"rate":0.13636363636363635},"22":{"count":2,"rate":0.045454545454545456},"26":{"count":2,"rate":0.045454545454545456},"28":{"count":2,"rate":0.045454545454545456},"30":{"count":1,"rate":0.022727272727272728},"8":{"count":9,"rate":0.20454545454545456}} |
| assessment | POSITIVE_SHOCK_ONSET | PERSISTENT | resistance_v1 | 44 | 13 | 11 | 0.888889 | 0.00979047 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":9,"rate":0.20454545454545456},"12":{"count":4,"rate":0.09090909090909091},"14":{"count":4,"rate":0.09090909090909091},"16":{"count":4,"rate":0.09090909090909091},"18":{"count":1,"rate":0.022727272727272728},"20":{"count":6,"rate":0.13636363636363635},"22":{"count":2,"rate":0.045454545454545456},"26":{"count":2,"rate":0.045454545454545456},"28":{"count":2,"rate":0.045454545454545456},"30":{"count":1,"rate":0.022727272727272728},"8":{"count":9,"rate":0.20454545454545456}} |
| assessment | POSITIVE_SHOCK_ONSET | PERSISTENT | frozen_A1 | 44 | 24 | 22 | 0.428571 | -0.00352835 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":9,"rate":0.20454545454545456},"12":{"count":4,"rate":0.09090909090909091},"14":{"count":4,"rate":0.09090909090909091},"16":{"count":4,"rate":0.09090909090909091},"18":{"count":1,"rate":0.022727272727272728},"20":{"count":6,"rate":0.13636363636363635},"22":{"count":2,"rate":0.045454545454545456},"26":{"count":2,"rate":0.045454545454545456},"28":{"count":2,"rate":0.045454545454545456},"30":{"count":1,"rate":0.022727272727272728},"8":{"count":9,"rate":0.20454545454545456}} |
| assessment | POSITIVE_SHOCK_ONSET | RE_ENTRY | continuation_v1 | 8 | 8 | 8 | 0.6 | 0.005493 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":3,"rate":0.375},"14":{"count":1,"rate":0.125},"20":{"count":2,"rate":0.25},"24":{"count":1,"rate":0.125},"30":{"count":1,"rate":0.125}} |
| assessment | POSITIVE_SHOCK_ONSET | RE_ENTRY | resistance_v1 | 8 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":3,"rate":0.375},"14":{"count":1,"rate":0.125},"20":{"count":2,"rate":0.25},"24":{"count":1,"rate":0.125},"30":{"count":1,"rate":0.125}} |
| assessment | POSITIVE_SHOCK_ONSET | RE_ENTRY | frozen_A1 | 8 | 5 | 5 | 0.25 | -0.0108652 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":3,"rate":0.375},"14":{"count":1,"rate":0.125},"20":{"count":2,"rate":0.25},"24":{"count":1,"rate":0.125},"30":{"count":1,"rate":0.125}} |
| assessment | ONGOING_NEGATIVE_SHOCK | FIRST_ENTRY | continuation_v1 | 2 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.5},"8":{"count":1,"rate":0.5}} |
| assessment | ONGOING_NEGATIVE_SHOCK | FIRST_ENTRY | resistance_v1 | 2 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.5},"8":{"count":1,"rate":0.5}} |
| assessment | ONGOING_NEGATIVE_SHOCK | FIRST_ENTRY | frozen_A1 | 2 | 1 | 0 | 1 | 0.0149019 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.5},"8":{"count":1,"rate":0.5}} |
| assessment | ONGOING_NEGATIVE_SHOCK | PERSISTENT | continuation_v1 | 21 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":8,"rate":0.38095238095238093},"14":{"count":1,"rate":0.047619047619047616},"16":{"count":5,"rate":0.23809523809523808},"8":{"count":7,"rate":0.3333333333333333}} |
| assessment | ONGOING_NEGATIVE_SHOCK | PERSISTENT | resistance_v1 | 21 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":8,"rate":0.38095238095238093},"14":{"count":1,"rate":0.047619047619047616},"16":{"count":5,"rate":0.23809523809523808},"8":{"count":7,"rate":0.3333333333333333}} |
| assessment | ONGOING_NEGATIVE_SHOCK | PERSISTENT | frozen_A1 | 21 | 6 | 0 | 0.25 | -0.00591051 | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":8,"rate":0.38095238095238093},"14":{"count":1,"rate":0.047619047619047616},"16":{"count":5,"rate":0.23809523809523808},"8":{"count":7,"rate":0.3333333333333333}} |
| assessment | ONGOING_NEGATIVE_SHOCK | RE_ENTRY | continuation_v1 | 6 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"12":{"count":1,"rate":0.16666666666666666},"14":{"count":5,"rate":0.8333333333333334}} |
| assessment | ONGOING_NEGATIVE_SHOCK | RE_ENTRY | resistance_v1 | 6 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"12":{"count":1,"rate":0.16666666666666666},"14":{"count":5,"rate":0.8333333333333334}} |
| assessment | ONGOING_NEGATIVE_SHOCK | RE_ENTRY | frozen_A1 | 6 | 2 | 0 | 1 | 0.00548322 | blocked_insufficient_support | fresh_tail_entries | {"12":{"count":1,"rate":0.16666666666666666},"14":{"count":5,"rate":0.8333333333333334}} |
| assessment | ONGOING_POSITIVE_SHOCK | FIRST_ENTRY | continuation_v1 | 4 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"8":{"count":4,"rate":1.0}} |
| assessment | ONGOING_POSITIVE_SHOCK | FIRST_ENTRY | resistance_v1 | 4 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"8":{"count":4,"rate":1.0}} |
| assessment | ONGOING_POSITIVE_SHOCK | FIRST_ENTRY | frozen_A1 | 4 | 2 | 0 | 1 | 0.00201765 | blocked_insufficient_support | fresh_tail_entries | {"8":{"count":4,"rate":1.0}} |
| assessment | ONGOING_POSITIVE_SHOCK | PERSISTENT | continuation_v1 | 15 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":6,"rate":0.4},"14":{"count":1,"rate":0.06666666666666667},"32":{"count":1,"rate":0.06666666666666667},"8":{"count":7,"rate":0.4666666666666667}} |
| assessment | ONGOING_POSITIVE_SHOCK | PERSISTENT | resistance_v1 | 15 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":6,"rate":0.4},"14":{"count":1,"rate":0.06666666666666667},"32":{"count":1,"rate":0.06666666666666667},"8":{"count":7,"rate":0.4666666666666667}} |
| assessment | ONGOING_POSITIVE_SHOCK | PERSISTENT | frozen_A1 | 15 | 7 | 0 | 0.8 | 0.00588973 | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":6,"rate":0.4},"14":{"count":1,"rate":0.06666666666666667},"32":{"count":1,"rate":0.06666666666666667},"8":{"count":7,"rate":0.4666666666666667}} |
| assessment | ONGOING_POSITIVE_SHOCK | RE_ENTRY | continuation_v1 | 1 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"14":{"count":1,"rate":1.0}} |
| assessment | ONGOING_POSITIVE_SHOCK | RE_ENTRY | resistance_v1 | 1 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"14":{"count":1,"rate":1.0}} |
| assessment | ONGOING_POSITIVE_SHOCK | RE_ENTRY | frozen_A1 | 1 | 1 | 0 | NA | -0.000476758 | blocked_insufficient_support | fresh_tail_entries | {"14":{"count":1,"rate":1.0}} |
| assessment | ELEVATED_RANGE_NONDIRECTIONAL | FIRST_ENTRY | continuation_v1 | 1 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":1.0}} |
| assessment | ELEVATED_RANGE_NONDIRECTIONAL | FIRST_ENTRY | resistance_v1 | 1 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":1.0}} |
| assessment | ELEVATED_RANGE_NONDIRECTIONAL | FIRST_ENTRY | frozen_A1 | 1 | 1 | 0 | NA | -0.0072126 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":1.0}} |
| assessment | ELEVATED_RANGE_NONDIRECTIONAL | PERSISTENT | continuation_v1 | 80 | 0 | 0 | NA | NA | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":16,"rate":0.2},"12":{"count":9,"rate":0.1125},"14":{"count":3,"rate":0.0375},"16":{"count":10,"rate":0.125},"18":{"count":9,"rate":0.1125},"20":{"count":1,"rate":0.0125},"22":{"count":5,"rate":0.0625},"24":{"count":3,"rate":0.0375},"26":{"count":1,"rate":0.0125},"28":{"count":1,"rate":0.0125},"34":{"count":1,"rate":0.0125},"8":{"count":21,"rate":0.2625}} |
| assessment | ELEVATED_RANGE_NONDIRECTIONAL | PERSISTENT | resistance_v1 | 80 | 0 | 0 | NA | NA | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":16,"rate":0.2},"12":{"count":9,"rate":0.1125},"14":{"count":3,"rate":0.0375},"16":{"count":10,"rate":0.125},"18":{"count":9,"rate":0.1125},"20":{"count":1,"rate":0.0125},"22":{"count":5,"rate":0.0625},"24":{"count":3,"rate":0.0375},"26":{"count":1,"rate":0.0125},"28":{"count":1,"rate":0.0125},"34":{"count":1,"rate":0.0125},"8":{"count":21,"rate":0.2625}} |
| assessment | ELEVATED_RANGE_NONDIRECTIONAL | PERSISTENT | frozen_A1 | 80 | 32 | 0 | 0.444444 | -0.00256084 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":16,"rate":0.2},"12":{"count":9,"rate":0.1125},"14":{"count":3,"rate":0.0375},"16":{"count":10,"rate":0.125},"18":{"count":9,"rate":0.1125},"20":{"count":1,"rate":0.0125},"22":{"count":5,"rate":0.0625},"24":{"count":3,"rate":0.0375},"26":{"count":1,"rate":0.0125},"28":{"count":1,"rate":0.0125},"34":{"count":1,"rate":0.0125},"8":{"count":21,"rate":0.2625}} |
| assessment | ELEVATED_RANGE_NONDIRECTIONAL | RE_ENTRY | continuation_v1 | 13 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":4,"rate":0.3076923076923077},"14":{"count":7,"rate":0.5384615384615384},"20":{"count":1,"rate":0.07692307692307693},"22":{"count":1,"rate":0.07692307692307693}} |
| assessment | ELEVATED_RANGE_NONDIRECTIONAL | RE_ENTRY | resistance_v1 | 13 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":4,"rate":0.3076923076923077},"14":{"count":7,"rate":0.5384615384615384},"20":{"count":1,"rate":0.07692307692307693},"22":{"count":1,"rate":0.07692307692307693}} |
| assessment | ELEVATED_RANGE_NONDIRECTIONAL | RE_ENTRY | frozen_A1 | 13 | 5 | 0 | 1 | 0.00316626 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":4,"rate":0.3076923076923077},"14":{"count":7,"rate":0.5384615384615384},"20":{"count":1,"rate":0.07692307692307693},"22":{"count":1,"rate":0.07692307692307693}} |
| assessment | NORMAL_OTHER | FIRST_ENTRY | continuation_v1 | 3 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":0.6666666666666666},"18":{"count":1,"rate":0.3333333333333333}} |
| assessment | NORMAL_OTHER | FIRST_ENTRY | resistance_v1 | 3 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":0.6666666666666666},"18":{"count":1,"rate":0.3333333333333333}} |
| assessment | NORMAL_OTHER | FIRST_ENTRY | frozen_A1 | 3 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":0.6666666666666666},"18":{"count":1,"rate":0.3333333333333333}} |
| assessment | NORMAL_OTHER | PERSISTENT | continuation_v1 | 255 | 0 | 0 | NA | NA | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":68,"rate":0.26666666666666666},"12":{"count":21,"rate":0.08235294117647059},"14":{"count":25,"rate":0.09803921568627451},"16":{"count":17,"rate":0.06666666666666667},"18":{"count":9,"rate":0.03529411764705882},"20":{"count":12,"rate":0.047058823529411764},"22":{"count":13,"rate":0.050980392156862744},"24":{"count":4,"rate":0.01568627450980392},"26":{"count":2,"rate":0.00784313725490196},"28":{"count":3,"rate":0.011764705882352941},"30":{"count":4,"rate":0.01568627450980392},"32":{"count":4,"rate":0.01568627450980392},"34":{"count":3,"rate":0.011764705882352941},"8":{"count":70,"rate":0.27450980392156865}} |
| assessment | NORMAL_OTHER | PERSISTENT | resistance_v1 | 255 | 0 | 0 | NA | NA | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":68,"rate":0.26666666666666666},"12":{"count":21,"rate":0.08235294117647059},"14":{"count":25,"rate":0.09803921568627451},"16":{"count":17,"rate":0.06666666666666667},"18":{"count":9,"rate":0.03529411764705882},"20":{"count":12,"rate":0.047058823529411764},"22":{"count":13,"rate":0.050980392156862744},"24":{"count":4,"rate":0.01568627450980392},"26":{"count":2,"rate":0.00784313725490196},"28":{"count":3,"rate":0.011764705882352941},"30":{"count":4,"rate":0.01568627450980392},"32":{"count":4,"rate":0.01568627450980392},"34":{"count":3,"rate":0.011764705882352941},"8":{"count":70,"rate":0.27450980392156865}} |
| assessment | NORMAL_OTHER | PERSISTENT | frozen_A1 | 255 | 92 | 0 | 0.35 | -0.00432844 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":68,"rate":0.26666666666666666},"12":{"count":21,"rate":0.08235294117647059},"14":{"count":25,"rate":0.09803921568627451},"16":{"count":17,"rate":0.06666666666666667},"18":{"count":9,"rate":0.03529411764705882},"20":{"count":12,"rate":0.047058823529411764},"22":{"count":13,"rate":0.050980392156862744},"24":{"count":4,"rate":0.01568627450980392},"26":{"count":2,"rate":0.00784313725490196},"28":{"count":3,"rate":0.011764705882352941},"30":{"count":4,"rate":0.01568627450980392},"32":{"count":4,"rate":0.01568627450980392},"34":{"count":3,"rate":0.011764705882352941},"8":{"count":70,"rate":0.27450980392156865}} |
| assessment | NORMAL_OTHER | RE_ENTRY | continuation_v1 | 31 | 0 | 0 | NA | NA | reported_descriptive | fresh_tail_entries | {"10":{"count":9,"rate":0.2903225806451613},"14":{"count":15,"rate":0.4838709677419355},"16":{"count":1,"rate":0.03225806451612903},"18":{"count":1,"rate":0.03225806451612903},"20":{"count":4,"rate":0.12903225806451613},"22":{"count":1,"rate":0.03225806451612903}} |
| assessment | NORMAL_OTHER | RE_ENTRY | resistance_v1 | 31 | 0 | 0 | NA | NA | reported_descriptive | fresh_tail_entries | {"10":{"count":9,"rate":0.2903225806451613},"14":{"count":15,"rate":0.4838709677419355},"16":{"count":1,"rate":0.03225806451612903},"18":{"count":1,"rate":0.03225806451612903},"20":{"count":4,"rate":0.12903225806451613},"22":{"count":1,"rate":0.03225806451612903}} |
| assessment | NORMAL_OTHER | RE_ENTRY | frozen_A1 | 31 | 10 | 0 | 1 | 0.00283411 | reported_descriptive | fresh_tail_entries | {"10":{"count":9,"rate":0.2903225806451613},"14":{"count":15,"rate":0.4838709677419355},"16":{"count":1,"rate":0.03225806451612903},"18":{"count":1,"rate":0.03225806451612903},"20":{"count":4,"rate":0.12903225806451613},"22":{"count":1,"rate":0.03225806451612903}} |
| assessment | UNKNOWN_INCOMPLETE | FIRST_ENTRY | continuation_v1 | 360 | 0 | 0 | NA | NA | reported_descriptive | fresh_tail_entries | {"6":{"count":360,"rate":1.0}} |
| assessment | UNKNOWN_INCOMPLETE | FIRST_ENTRY | resistance_v1 | 360 | 0 | 0 | NA | NA | reported_descriptive | fresh_tail_entries | {"6":{"count":360,"rate":1.0}} |
| assessment | UNKNOWN_INCOMPLETE | FIRST_ENTRY | frozen_A1 | 360 | 114 | 0 | 0.527273 | 0.00090231 | reported_descriptive | fresh_tail_entries | {"6":{"count":360,"rate":1.0}} |
| assessment | UNKNOWN_INCOMPLETE | PERSISTENT | continuation_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {} |
| assessment | UNKNOWN_INCOMPLETE | PERSISTENT | resistance_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {} |
| assessment | UNKNOWN_INCOMPLETE | PERSISTENT | frozen_A1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {} |
| assessment | UNKNOWN_INCOMPLETE | RE_ENTRY | continuation_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| assessment | UNKNOWN_INCOMPLETE | RE_ENTRY | resistance_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| assessment | UNKNOWN_INCOMPLETE | RE_ENTRY | frozen_A1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| stress | NEGATIVE_SHOCK_ONSET | FIRST_ENTRY | continuation_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| stress | NEGATIVE_SHOCK_ONSET | FIRST_ENTRY | resistance_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| stress | NEGATIVE_SHOCK_ONSET | FIRST_ENTRY | frozen_A1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| stress | NEGATIVE_SHOCK_ONSET | PERSISTENT | continuation_v1 | 59 | 38 | 22 | 0.5 | 5.18037e-05 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":4,"rate":0.06779661016949153},"12":{"count":2,"rate":0.03389830508474576},"14":{"count":10,"rate":0.1694915254237288},"16":{"count":2,"rate":0.03389830508474576},"18":{"count":9,"rate":0.15254237288135594},"20":{"count":8,"rate":0.13559322033898305},"22":{"count":5,"rate":0.0847457627118644},"24":{"count":3,"rate":0.05084745762711865},"28":{"count":1,"rate":0.01694915254237288},"30":{"count":2,"rate":0.03389830508474576},"32":{"count":2,"rate":0.03389830508474576},"8":{"count":11,"rate":0.1864406779661017}} |
| stress | NEGATIVE_SHOCK_ONSET | PERSISTENT | resistance_v1 | 59 | 21 | 16 | 0.333333 | -0.000919831 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":4,"rate":0.06779661016949153},"12":{"count":2,"rate":0.03389830508474576},"14":{"count":10,"rate":0.1694915254237288},"16":{"count":2,"rate":0.03389830508474576},"18":{"count":9,"rate":0.15254237288135594},"20":{"count":8,"rate":0.13559322033898305},"22":{"count":5,"rate":0.0847457627118644},"24":{"count":3,"rate":0.05084745762711865},"28":{"count":1,"rate":0.01694915254237288},"30":{"count":2,"rate":0.03389830508474576},"32":{"count":2,"rate":0.03389830508474576},"8":{"count":11,"rate":0.1864406779661017}} |
| stress | NEGATIVE_SHOCK_ONSET | PERSISTENT | frozen_A1 | 59 | 24 | 23 | 0.538462 | 0.00107602 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":4,"rate":0.06779661016949153},"12":{"count":2,"rate":0.03389830508474576},"14":{"count":10,"rate":0.1694915254237288},"16":{"count":2,"rate":0.03389830508474576},"18":{"count":9,"rate":0.15254237288135594},"20":{"count":8,"rate":0.13559322033898305},"22":{"count":5,"rate":0.0847457627118644},"24":{"count":3,"rate":0.05084745762711865},"28":{"count":1,"rate":0.01694915254237288},"30":{"count":2,"rate":0.03389830508474576},"32":{"count":2,"rate":0.03389830508474576},"8":{"count":11,"rate":0.1864406779661017}} |
| stress | NEGATIVE_SHOCK_ONSET | RE_ENTRY | continuation_v1 | 22 | 18 | 14 | 0.444444 | -0.00327756 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.045454545454545456},"14":{"count":10,"rate":0.45454545454545453},"16":{"count":2,"rate":0.09090909090909091},"18":{"count":1,"rate":0.045454545454545456},"20":{"count":6,"rate":0.2727272727272727},"28":{"count":1,"rate":0.045454545454545456},"30":{"count":1,"rate":0.045454545454545456}} |
| stress | NEGATIVE_SHOCK_ONSET | RE_ENTRY | resistance_v1 | 22 | 4 | 3 | 0.5 | -0.00292629 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.045454545454545456},"14":{"count":10,"rate":0.45454545454545453},"16":{"count":2,"rate":0.09090909090909091},"18":{"count":1,"rate":0.045454545454545456},"20":{"count":6,"rate":0.2727272727272727},"28":{"count":1,"rate":0.045454545454545456},"30":{"count":1,"rate":0.045454545454545456}} |
| stress | NEGATIVE_SHOCK_ONSET | RE_ENTRY | frozen_A1 | 22 | 10 | 10 | 0.8 | 0.00493817 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.045454545454545456},"14":{"count":10,"rate":0.45454545454545453},"16":{"count":2,"rate":0.09090909090909091},"18":{"count":1,"rate":0.045454545454545456},"20":{"count":6,"rate":0.2727272727272727},"28":{"count":1,"rate":0.045454545454545456},"30":{"count":1,"rate":0.045454545454545456}} |
| stress | POSITIVE_SHOCK_ONSET | FIRST_ENTRY | continuation_v1 | 2 | 2 | 2 | 0 | -0.0076642 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.5},"8":{"count":1,"rate":0.5}} |
| stress | POSITIVE_SHOCK_ONSET | FIRST_ENTRY | resistance_v1 | 2 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.5},"8":{"count":1,"rate":0.5}} |
| stress | POSITIVE_SHOCK_ONSET | FIRST_ENTRY | frozen_A1 | 2 | 1 | 1 | 1 | 0.021843 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.5},"8":{"count":1,"rate":0.5}} |
| stress | POSITIVE_SHOCK_ONSET | PERSISTENT | continuation_v1 | 67 | 56 | 28 | 0.472222 | -0.002827 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":18,"rate":0.26865671641791045},"12":{"count":12,"rate":0.1791044776119403},"14":{"count":7,"rate":0.1044776119402985},"16":{"count":5,"rate":0.07462686567164178},"18":{"count":2,"rate":0.029850746268656716},"20":{"count":1,"rate":0.014925373134328358},"24":{"count":3,"rate":0.04477611940298507},"26":{"count":1,"rate":0.014925373134328358},"28":{"count":6,"rate":0.08955223880597014},"30":{"count":1,"rate":0.014925373134328358},"8":{"count":11,"rate":0.16417910447761194}} |
| stress | POSITIVE_SHOCK_ONSET | PERSISTENT | resistance_v1 | 67 | 11 | 10 | 0.714286 | 0.00162812 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":18,"rate":0.26865671641791045},"12":{"count":12,"rate":0.1791044776119403},"14":{"count":7,"rate":0.1044776119402985},"16":{"count":5,"rate":0.07462686567164178},"18":{"count":2,"rate":0.029850746268656716},"20":{"count":1,"rate":0.014925373134328358},"24":{"count":3,"rate":0.04477611940298507},"26":{"count":1,"rate":0.014925373134328358},"28":{"count":6,"rate":0.08955223880597014},"30":{"count":1,"rate":0.014925373134328358},"8":{"count":11,"rate":0.16417910447761194}} |
| stress | POSITIVE_SHOCK_ONSET | PERSISTENT | frozen_A1 | 67 | 35 | 22 | 0.64 | 0.00525717 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":18,"rate":0.26865671641791045},"12":{"count":12,"rate":0.1791044776119403},"14":{"count":7,"rate":0.1044776119402985},"16":{"count":5,"rate":0.07462686567164178},"18":{"count":2,"rate":0.029850746268656716},"20":{"count":1,"rate":0.014925373134328358},"24":{"count":3,"rate":0.04477611940298507},"26":{"count":1,"rate":0.014925373134328358},"28":{"count":6,"rate":0.08955223880597014},"30":{"count":1,"rate":0.014925373134328358},"8":{"count":11,"rate":0.16417910447761194}} |
| stress | POSITIVE_SHOCK_ONSET | RE_ENTRY | continuation_v1 | 10 | 8 | 6 | 0.5 | -0.000252591 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":4,"rate":0.4},"14":{"count":5,"rate":0.5},"32":{"count":1,"rate":0.1}} |
| stress | POSITIVE_SHOCK_ONSET | RE_ENTRY | resistance_v1 | 10 | 2 | 1 | NA | 0.00518424 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":4,"rate":0.4},"14":{"count":5,"rate":0.5},"32":{"count":1,"rate":0.1}} |
| stress | POSITIVE_SHOCK_ONSET | RE_ENTRY | frozen_A1 | 10 | 4 | 3 | NA | -0.00156076 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":4,"rate":0.4},"14":{"count":5,"rate":0.5},"32":{"count":1,"rate":0.1}} |
| stress | ONGOING_NEGATIVE_SHOCK | FIRST_ENTRY | continuation_v1 | 2 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":1.0}} |
| stress | ONGOING_NEGATIVE_SHOCK | FIRST_ENTRY | resistance_v1 | 2 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":1.0}} |
| stress | ONGOING_NEGATIVE_SHOCK | FIRST_ENTRY | frozen_A1 | 2 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":1.0}} |
| stress | ONGOING_NEGATIVE_SHOCK | PERSISTENT | continuation_v1 | 27 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"20":{"count":4,"rate":0.14814814814814814},"22":{"count":2,"rate":0.07407407407407407},"24":{"count":5,"rate":0.18518518518518517},"26":{"count":5,"rate":0.18518518518518517},"28":{"count":2,"rate":0.07407407407407407},"30":{"count":3,"rate":0.1111111111111111},"32":{"count":1,"rate":0.037037037037037035},"34":{"count":2,"rate":0.07407407407407407},"8":{"count":3,"rate":0.1111111111111111}} |
| stress | ONGOING_NEGATIVE_SHOCK | PERSISTENT | resistance_v1 | 27 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"20":{"count":4,"rate":0.14814814814814814},"22":{"count":2,"rate":0.07407407407407407},"24":{"count":5,"rate":0.18518518518518517},"26":{"count":5,"rate":0.18518518518518517},"28":{"count":2,"rate":0.07407407407407407},"30":{"count":3,"rate":0.1111111111111111},"32":{"count":1,"rate":0.037037037037037035},"34":{"count":2,"rate":0.07407407407407407},"8":{"count":3,"rate":0.1111111111111111}} |
| stress | ONGOING_NEGATIVE_SHOCK | PERSISTENT | frozen_A1 | 27 | 17 | 0 | 0.666667 | -0.00446072 | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"20":{"count":4,"rate":0.14814814814814814},"22":{"count":2,"rate":0.07407407407407407},"24":{"count":5,"rate":0.18518518518518517},"26":{"count":5,"rate":0.18518518518518517},"28":{"count":2,"rate":0.07407407407407407},"30":{"count":3,"rate":0.1111111111111111},"32":{"count":1,"rate":0.037037037037037035},"34":{"count":2,"rate":0.07407407407407407},"8":{"count":3,"rate":0.1111111111111111}} |
| stress | ONGOING_NEGATIVE_SHOCK | RE_ENTRY | continuation_v1 | 8 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":0.25},"22":{"count":1,"rate":0.125},"24":{"count":1,"rate":0.125},"28":{"count":2,"rate":0.25},"32":{"count":1,"rate":0.125},"34":{"count":1,"rate":0.125}} |
| stress | ONGOING_NEGATIVE_SHOCK | RE_ENTRY | resistance_v1 | 8 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":0.25},"22":{"count":1,"rate":0.125},"24":{"count":1,"rate":0.125},"28":{"count":2,"rate":0.25},"32":{"count":1,"rate":0.125},"34":{"count":1,"rate":0.125}} |
| stress | ONGOING_NEGATIVE_SHOCK | RE_ENTRY | frozen_A1 | 8 | 3 | 0 | 1 | 0.0113768 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":0.25},"22":{"count":1,"rate":0.125},"24":{"count":1,"rate":0.125},"28":{"count":2,"rate":0.25},"32":{"count":1,"rate":0.125},"34":{"count":1,"rate":0.125}} |
| stress | ONGOING_POSITIVE_SHOCK | FIRST_ENTRY | continuation_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| stress | ONGOING_POSITIVE_SHOCK | FIRST_ENTRY | resistance_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| stress | ONGOING_POSITIVE_SHOCK | FIRST_ENTRY | frozen_A1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| stress | ONGOING_POSITIVE_SHOCK | PERSISTENT | continuation_v1 | 15 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":3,"rate":0.2},"12":{"count":1,"rate":0.06666666666666667},"16":{"count":5,"rate":0.3333333333333333},"18":{"count":2,"rate":0.13333333333333333},"26":{"count":1,"rate":0.06666666666666667},"30":{"count":2,"rate":0.13333333333333333},"8":{"count":1,"rate":0.06666666666666667}} |
| stress | ONGOING_POSITIVE_SHOCK | PERSISTENT | resistance_v1 | 15 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":3,"rate":0.2},"12":{"count":1,"rate":0.06666666666666667},"16":{"count":5,"rate":0.3333333333333333},"18":{"count":2,"rate":0.13333333333333333},"26":{"count":1,"rate":0.06666666666666667},"30":{"count":2,"rate":0.13333333333333333},"8":{"count":1,"rate":0.06666666666666667}} |
| stress | ONGOING_POSITIVE_SHOCK | PERSISTENT | frozen_A1 | 15 | 9 | 0 | 0 | -0.00505672 | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":3,"rate":0.2},"12":{"count":1,"rate":0.06666666666666667},"16":{"count":5,"rate":0.3333333333333333},"18":{"count":2,"rate":0.13333333333333333},"26":{"count":1,"rate":0.06666666666666667},"30":{"count":2,"rate":0.13333333333333333},"8":{"count":1,"rate":0.06666666666666667}} |
| stress | ONGOING_POSITIVE_SHOCK | RE_ENTRY | continuation_v1 | 4 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":0.5},"30":{"count":2,"rate":0.5}} |
| stress | ONGOING_POSITIVE_SHOCK | RE_ENTRY | resistance_v1 | 4 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":0.5},"30":{"count":2,"rate":0.5}} |
| stress | ONGOING_POSITIVE_SHOCK | RE_ENTRY | frozen_A1 | 4 | 3 | 0 | 1 | 0.00671723 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":2,"rate":0.5},"30":{"count":2,"rate":0.5}} |
| stress | ELEVATED_RANGE_NONDIRECTIONAL | FIRST_ENTRY | continuation_v1 | 3 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"14":{"count":2,"rate":0.6666666666666666},"8":{"count":1,"rate":0.3333333333333333}} |
| stress | ELEVATED_RANGE_NONDIRECTIONAL | FIRST_ENTRY | resistance_v1 | 3 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"14":{"count":2,"rate":0.6666666666666666},"8":{"count":1,"rate":0.3333333333333333}} |
| stress | ELEVATED_RANGE_NONDIRECTIONAL | FIRST_ENTRY | frozen_A1 | 3 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"14":{"count":2,"rate":0.6666666666666666},"8":{"count":1,"rate":0.3333333333333333}} |
| stress | ELEVATED_RANGE_NONDIRECTIONAL | PERSISTENT | continuation_v1 | 97 | 0 | 0 | NA | NA | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":21,"rate":0.21649484536082475},"12":{"count":13,"rate":0.13402061855670103},"14":{"count":4,"rate":0.041237113402061855},"16":{"count":10,"rate":0.10309278350515463},"18":{"count":5,"rate":0.05154639175257732},"20":{"count":3,"rate":0.030927835051546393},"22":{"count":9,"rate":0.09278350515463918},"24":{"count":3,"rate":0.030927835051546393},"26":{"count":5,"rate":0.05154639175257732},"28":{"count":1,"rate":0.010309278350515464},"30":{"count":1,"rate":0.010309278350515464},"34":{"count":2,"rate":0.020618556701030927},"8":{"count":20,"rate":0.20618556701030927}} |
| stress | ELEVATED_RANGE_NONDIRECTIONAL | PERSISTENT | resistance_v1 | 97 | 0 | 0 | NA | NA | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":21,"rate":0.21649484536082475},"12":{"count":13,"rate":0.13402061855670103},"14":{"count":4,"rate":0.041237113402061855},"16":{"count":10,"rate":0.10309278350515463},"18":{"count":5,"rate":0.05154639175257732},"20":{"count":3,"rate":0.030927835051546393},"22":{"count":9,"rate":0.09278350515463918},"24":{"count":3,"rate":0.030927835051546393},"26":{"count":5,"rate":0.05154639175257732},"28":{"count":1,"rate":0.010309278350515464},"30":{"count":1,"rate":0.010309278350515464},"34":{"count":2,"rate":0.020618556701030927},"8":{"count":20,"rate":0.20618556701030927}} |
| stress | ELEVATED_RANGE_NONDIRECTIONAL | PERSISTENT | frozen_A1 | 97 | 49 | 0 | 0.521739 | 0.000464799 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":21,"rate":0.21649484536082475},"12":{"count":13,"rate":0.13402061855670103},"14":{"count":4,"rate":0.041237113402061855},"16":{"count":10,"rate":0.10309278350515463},"18":{"count":5,"rate":0.05154639175257732},"20":{"count":3,"rate":0.030927835051546393},"22":{"count":9,"rate":0.09278350515463918},"24":{"count":3,"rate":0.030927835051546393},"26":{"count":5,"rate":0.05154639175257732},"28":{"count":1,"rate":0.010309278350515464},"30":{"count":1,"rate":0.010309278350515464},"34":{"count":2,"rate":0.020618556701030927},"8":{"count":20,"rate":0.20618556701030927}} |
| stress | ELEVATED_RANGE_NONDIRECTIONAL | RE_ENTRY | continuation_v1 | 12 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":3,"rate":0.25},"12":{"count":1,"rate":0.08333333333333333},"14":{"count":3,"rate":0.25},"18":{"count":2,"rate":0.16666666666666666},"20":{"count":1,"rate":0.08333333333333333},"22":{"count":2,"rate":0.16666666666666666}} |
| stress | ELEVATED_RANGE_NONDIRECTIONAL | RE_ENTRY | resistance_v1 | 12 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":3,"rate":0.25},"12":{"count":1,"rate":0.08333333333333333},"14":{"count":3,"rate":0.25},"18":{"count":2,"rate":0.16666666666666666},"20":{"count":1,"rate":0.08333333333333333},"22":{"count":2,"rate":0.16666666666666666}} |
| stress | ELEVATED_RANGE_NONDIRECTIONAL | RE_ENTRY | frozen_A1 | 12 | 6 | 0 | 0.333333 | 0.00106625 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":3,"rate":0.25},"12":{"count":1,"rate":0.08333333333333333},"14":{"count":3,"rate":0.25},"18":{"count":2,"rate":0.16666666666666666},"20":{"count":1,"rate":0.08333333333333333},"22":{"count":2,"rate":0.16666666666666666}} |
| stress | NORMAL_OTHER | FIRST_ENTRY | continuation_v1 | 6 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.16666666666666666},"16":{"count":2,"rate":0.3333333333333333},"20":{"count":1,"rate":0.16666666666666666},"8":{"count":2,"rate":0.3333333333333333}} |
| stress | NORMAL_OTHER | FIRST_ENTRY | resistance_v1 | 6 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.16666666666666666},"16":{"count":2,"rate":0.3333333333333333},"20":{"count":1,"rate":0.16666666666666666},"8":{"count":2,"rate":0.3333333333333333}} |
| stress | NORMAL_OTHER | FIRST_ENTRY | frozen_A1 | 6 | 2 | 0 | NA | -0.00307702 | blocked_insufficient_support | fresh_tail_entries | {"10":{"count":1,"rate":0.16666666666666666},"16":{"count":2,"rate":0.3333333333333333},"20":{"count":1,"rate":0.16666666666666666},"8":{"count":2,"rate":0.3333333333333333}} |
| stress | NORMAL_OTHER | PERSISTENT | continuation_v1 | 329 | 0 | 0 | NA | NA | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":80,"rate":0.24316109422492402},"12":{"count":27,"rate":0.08206686930091185},"14":{"count":33,"rate":0.10030395136778116},"16":{"count":25,"rate":0.07598784194528875},"18":{"count":17,"rate":0.05167173252279635},"20":{"count":15,"rate":0.04559270516717325},"22":{"count":13,"rate":0.03951367781155015},"24":{"count":5,"rate":0.015197568389057751},"26":{"count":2,"rate":0.0060790273556231},"28":{"count":1,"rate":0.00303951367781155},"30":{"count":3,"rate":0.00911854103343465},"8":{"count":108,"rate":0.3282674772036474}} |
| stress | NORMAL_OTHER | PERSISTENT | resistance_v1 | 329 | 0 | 0 | NA | NA | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":80,"rate":0.24316109422492402},"12":{"count":27,"rate":0.08206686930091185},"14":{"count":33,"rate":0.10030395136778116},"16":{"count":25,"rate":0.07598784194528875},"18":{"count":17,"rate":0.05167173252279635},"20":{"count":15,"rate":0.04559270516717325},"22":{"count":13,"rate":0.03951367781155015},"24":{"count":5,"rate":0.015197568389057751},"26":{"count":2,"rate":0.0060790273556231},"28":{"count":1,"rate":0.00303951367781155},"30":{"count":3,"rate":0.00911854103343465},"8":{"count":108,"rate":0.3282674772036474}} |
| stress | NORMAL_OTHER | PERSISTENT | frozen_A1 | 329 | 137 | 0 | 0.462687 | -0.00221945 | reported_descriptive | persistent_checkpoint_rows_descriptive_not_independent | {"10":{"count":80,"rate":0.24316109422492402},"12":{"count":27,"rate":0.08206686930091185},"14":{"count":33,"rate":0.10030395136778116},"16":{"count":25,"rate":0.07598784194528875},"18":{"count":17,"rate":0.05167173252279635},"20":{"count":15,"rate":0.04559270516717325},"22":{"count":13,"rate":0.03951367781155015},"24":{"count":5,"rate":0.015197568389057751},"26":{"count":2,"rate":0.0060790273556231},"28":{"count":1,"rate":0.00303951367781155},"30":{"count":3,"rate":0.00911854103343465},"8":{"count":108,"rate":0.3282674772036474}} |
| stress | NORMAL_OTHER | RE_ENTRY | continuation_v1 | 41 | 0 | 0 | NA | NA | reported_descriptive | fresh_tail_entries | {"10":{"count":13,"rate":0.3170731707317073},"14":{"count":14,"rate":0.34146341463414637},"16":{"count":2,"rate":0.04878048780487805},"18":{"count":3,"rate":0.07317073170731707},"20":{"count":6,"rate":0.14634146341463414},"22":{"count":2,"rate":0.04878048780487805},"30":{"count":1,"rate":0.024390243902439025}} |
| stress | NORMAL_OTHER | RE_ENTRY | resistance_v1 | 41 | 0 | 0 | NA | NA | reported_descriptive | fresh_tail_entries | {"10":{"count":13,"rate":0.3170731707317073},"14":{"count":14,"rate":0.34146341463414637},"16":{"count":2,"rate":0.04878048780487805},"18":{"count":3,"rate":0.07317073170731707},"20":{"count":6,"rate":0.14634146341463414},"22":{"count":2,"rate":0.04878048780487805},"30":{"count":1,"rate":0.024390243902439025}} |
| stress | NORMAL_OTHER | RE_ENTRY | frozen_A1 | 41 | 21 | 0 | 0 | -0.00184785 | reported_descriptive | fresh_tail_entries | {"10":{"count":13,"rate":0.3170731707317073},"14":{"count":14,"rate":0.34146341463414637},"16":{"count":2,"rate":0.04878048780487805},"18":{"count":3,"rate":0.07317073170731707},"20":{"count":6,"rate":0.14634146341463414},"22":{"count":2,"rate":0.04878048780487805},"30":{"count":1,"rate":0.024390243902439025}} |
| stress | UNKNOWN_INCOMPLETE | FIRST_ENTRY | continuation_v1 | 442 | 0 | 0 | NA | NA | reported_descriptive | fresh_tail_entries | {"6":{"count":442,"rate":1.0}} |
| stress | UNKNOWN_INCOMPLETE | FIRST_ENTRY | resistance_v1 | 442 | 0 | 0 | NA | NA | reported_descriptive | fresh_tail_entries | {"6":{"count":442,"rate":1.0}} |
| stress | UNKNOWN_INCOMPLETE | FIRST_ENTRY | frozen_A1 | 442 | 181 | 0 | 0.534091 | 0.000195133 | reported_descriptive | fresh_tail_entries | {"6":{"count":442,"rate":1.0}} |
| stress | UNKNOWN_INCOMPLETE | PERSISTENT | continuation_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {} |
| stress | UNKNOWN_INCOMPLETE | PERSISTENT | resistance_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {} |
| stress | UNKNOWN_INCOMPLETE | PERSISTENT | frozen_A1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | persistent_checkpoint_rows_descriptive_not_independent | {} |
| stress | UNKNOWN_INCOMPLETE | RE_ENTRY | continuation_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| stress | UNKNOWN_INCOMPLETE | RE_ENTRY | resistance_v1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |
| stress | UNKNOWN_INCOMPLETE | RE_ENTRY | frozen_A1 | 0 | 0 | 0 | NA | NA | blocked_insufficient_support | fresh_tail_entries | {} |

## Data completeness and operational blockers

| period | category | reason | episode_count |
| --- | --- | --- | --- |
| development | market_window | invalid_market_bar:1 | 1 |
| development | market_window | non_contiguous_market_timestamps:0-2 | 1 |
| development | market_window | w1_reference_would_cross_session | 413 |
| development | market_shock | insufficient_predictor_support:market_return_w0_v1=0,market_range_w0_v1=0,market_return_w1_v1=0,market_range_w1_v1=0 | 413 |
| development | market_shock | invalid_market_bar:1 | 1 |
| development | market_shock | market_window_measurement_incomplete | 413 |
| development | market_shock | non_contiguous_market_timestamps:0-2 | 1 |
| development | market_shock | w1_reference_would_cross_session | 413 |
| development | stock_response | insufficient_predictor_support:market_return_w0_v1=0,market_range_w0_v1=0,market_return_w1_v1=0,market_range_w1_v1=0 | 413 |
| development | stock_response | invalid_market_bar:1 | 1 |
| development | stock_response | market_window_measurement_incomplete | 413 |
| development | stock_response | non_contiguous_market_timestamps:0-2 | 1 |
| development | stock_response | w1_reference_would_cross_session | 413 |
| assessment | market_window | w1_reference_would_cross_session | 360 |
| assessment | market_shock | insufficient_predictor_support:market_return_w0_v1=0,market_range_w0_v1=0,market_return_w1_v1=0,market_range_w1_v1=0 | 360 |
| assessment | market_shock | market_window_measurement_incomplete | 360 |
| assessment | market_shock | w1_reference_would_cross_session | 360 |
| assessment | stock_response | insufficient_predictor_support:market_return_w0_v1=0,market_range_w0_v1=0,market_return_w1_v1=0,market_range_w1_v1=0 | 360 |
| assessment | stock_response | market_window_measurement_incomplete | 360 |
| assessment | stock_response | w1_reference_would_cross_session | 360 |
| stress | market_window | w1_reference_would_cross_session | 442 |
| stress | market_shock | insufficient_predictor_support:market_return_w0_v1=0,market_range_w0_v1=0,market_return_w1_v1=0,market_range_w1_v1=0 | 442 |
| stress | market_shock | market_window_measurement_incomplete | 442 |
| stress | market_shock | w1_reference_would_cross_session | 442 |
| stress | stock_response | insufficient_predictor_support:market_return_w0_v1=0,market_range_w0_v1=0,market_return_w1_v1=0,market_range_w1_v1=0 | 442 |
| stress | stock_response | market_window_measurement_incomplete | 442 |
| stress | stock_response | w1_reference_would_cross_session | 442 |

Missing windows, calibration, timestamps, stock bars, or IV scales fail closed as `UNKNOWN_INCOMPLETE`. Operational blockers are not interpreted as negative scientific evidence.

## Execution realism

Five-minute historical bars cannot observe bid withdrawal, ask withdrawal, replenishment, trade impact, spread changes, queue behaviour, or executable option outcomes. Prospective bid/ask and trade-impact recording is required to learn those quantities.

No option P&L, hypothetical option return, midpoint fill, broker access, order routing, or order placement occurred.

## Direct answers

1. **Canonical proxy?** Yes—VTI, already used by the causal baseline.
2. **Onset frequency?** Assessment: 232 negative and 244 positive; stress: 97 negative and 90 positive.
3. **Spread?** Session, checkpoint, stock, and shock-event support and concentration are reported explicitly; decision caps were enforced.
4. **Movement difference?** The checkpoint-standardised no-move, IV-excess, and absolute-movement comparisons above answer this without directional selection.
5. **Amplifying continuation?** `blocked_insufficient_support`; assessment/stress mean aligned returns were -0.000305/-0.002313.
6. **Resisting opposition?** `blocked_insufficient_support`; assessment/stress mean aligned returns were -0.001958/-0.002926.
7. **Continuous rank?** Assessment/stress AUCs were 0.000000/0.314286.
8. **Assessment and stress unchanged?** Frozen 2024 thresholds and response definitions were applied unchanged to both.
9. **Baselines?** The fixed same-timestamp comparisons above include market direction, opposition, recent 5m and trailing 15m momentum, frozen A1, always CALL/PUT, and blocked D2.
10. **Clustering?** Both session and shared-shock-event bootstraps were required by the decision contract.
11. **Both shock signs?** Separate positive/negative results are shown above and are contract inputs.
12. **Dominance?** Leave-one-out and concentration artifacts cover stock, month, checkpoint, session, and shock event.
13. **Tail Phase?** Descriptive only; it did not alter any action.
14. **Easier than normal?** See the checkpoint-standardised normal-regime table; no model or checkpoint selection was used.
15. **Classification?** `blocked_insufficient_support`.
16. **Still unknowable?** Executable option prices, liquidity withdrawal/replenishment, impact, spreads, queueing, fills, and prospective behavioural stability.

## Frozen-system confirmations

- M1C probabilities, threshold, horizon, high-tail membership, and fresh episode identifiers were unchanged.
- Tail Phase V1 and frozen A1 were unchanged.
- Archived signed pressure, archived tension, contaminated descendants, future peer slates, and cross-sectional normalisation were not used.
- No protected 2026 historical outcome was opened, calculated, displayed, or inspected.
- No broker was accessed; no order-routing path was enabled; no order was placed.
