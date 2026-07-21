# Behavioural-Trajectory Funnel V0.1 — Corrected Anchors and Later Loops

Decision: `trajectory_main_effects_only`.

This was a retrospective, observable-only, structural quick feasibility screen. It did not
open economic outcomes, test price direction or trading rules, enable execution, or provide
prospective validation.

## Population and causal anchors

- Development: 2024-01-01 through 2024-12-31.
- Assessment: 2025-01-01 through 2025-08-22.
- Protected rows materialised: 0.
- Corrected anchors: 6→2/4/6, 12→4/8/12, 24→8/16/24, 36→12/24/36.
- Final-anchor rows compared: 15549.
- Maximum final-level difference: 0.
- Maximum final-scaling difference: 0.
- Trajectory retention: 0.999067.
- Assessment support: `{"late_after_open": {"class_support": {"NO_REGISTERED_COMPLETION": 135, "REGISTERED_COMPLETION": 31, "UNREGISTERED_LOOP": 131}, "maximum_stock_share": 0.08754208754208755, "maximum_target_class_share": 0.45454545454545453, "months": 8, "rows": 297, "sessions": 93, "stocks": 20}, "late_no_open": {"class_support": {"NO_REGISTERED_COMPLETION": 2783, "REGISTERED_COMPLETION": 460, "UNREGISTERED_LOOP": 2672}, "maximum_stock_share": 0.05156382079459002, "maximum_target_class_share": 0.47049873203719356, "months": 8, "rows": 5915, "sessions": 157, "stocks": 20}, "later": {"class_support": {"NO_REGISTERED_COMPLETION": 2918, "REGISTERED_COMPLETION": 491, "UNREGISTERED_LOOP": 2803}, "maximum_stock_share": 0.05054732775273664, "maximum_target_class_share": 0.46973599484867995, "months": 8, "rows": 6212, "sessions": 157, "stocks": 20}, "opening": {"class_support": {"NO_REGISTERED_COMPLETION": 4308, "REGISTERED_COMPLETION": 312, "UNREGISTERED_LOOP": 1641}, "maximum_stock_share": 0.050151732950007986, "maximum_target_class_share": 0.6880689985625299, "months": 8, "rows": 6261, "sessions": 157, "stocks": 20}, "pooled": {"class_support": {"NO_REGISTERED_COMPLETION": 7226, "REGISTERED_COMPLETION": 803, "UNREGISTERED_LOOP": 4444}, "maximum_stock_share": 0.05034875330714343, "maximum_target_class_share": 0.5793313557283732, "months": 8, "rows": 12473, "sessions": 157, "stocks": 20}}`.
- Assessment structural targets: `{"NO_REGISTERED_COMPLETION": 7226, "REGISTERED_COMPLETION": 803, "UNREGISTERED_LOOP": 4444}`.

## Pooled assessment metrics

| model | multiclass_log_loss | multiclass_brier | top_one_accuracy | top_two_accuracy | mean_probability_realised_class | prediction_entropy | effective_candidate_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | 0.702177746 | 0.422797986 | 0.681806380 | 0.939347506 | 0.572021153 | 0.723612011 | 2.061867263 |
| T1 | 0.701017660 | 0.421677728 | 0.684271739 | 0.937600570 | 0.573599740 | 0.720026617 | 2.054487893 |
| T2 | 0.700948884 | 0.421578225 | 0.684533407 | 0.937919041 | 0.573749162 | 0.719694657 | 2.053806001 |

## Preregistered increments

| population | comparison | log_loss_improvement | brier_improvement | top_two_change | prediction_entropy_reduction |
| --- | --- | --- | --- | --- | --- |
| pooled | T1_minus_T0 | 0.001160087 | 0.001120258 | -0.001746936 | 0.003585394 |
| pooled | T2_minus_T1 | 0.000068775 | 0.000099504 | 0.000318471 | 0.000331959 |
| opening | T1_minus_T0 | 0.004960595 | 0.003520285 | -0.000636943 | 0.001965010 |
| opening | T2_minus_T1 | 0.000087896 | 0.000125077 | 0.000000000 | 0.000447532 |
| later | T1_minus_T0 | -0.002640421 | -0.001279770 | -0.002856930 | 0.005205778 |
| later | T2_minus_T1 | 0.000049654 | 0.000073930 | 0.000636943 | 0.000216386 |
| late_no_open | T1_minus_T0 | -0.002342016 | -0.001174515 | -0.002823851 | 0.005263848 |
| late_no_open | T2_minus_T1 | -0.000033286 | 0.000021667 | 0.000501605 | 0.000208796 |

The bootstrap used exactly 25 fixed-prediction whole-session draws. The null used exactly five
fixed-seed trajectory-bundle permutations with T1/T2 refits. Full 80%, 90%, and 95% intervals
and the five-draw comparisons are in `bootstrap_metrics.csv` and `null_metrics.csv`.

Bootstrap summary keys: `late_no_open|T1_minus_T0, late_no_open|T2_minus_T1, later|T1_minus_T0, later|T2_minus_T1, opening|T1_minus_T0, opening|T2_minus_T1, pooled|T1_minus_T0, pooled|T2_minus_T1`.

Null summary keys: `late_no_open|T1_minus_T0, late_no_open|T2_minus_T1, later|T1_minus_T0, later|T2_minus_T1, opening|T1_minus_T0, opening|T2_minus_T1, pooled|T1_minus_T0, pooled|T2_minus_T1`.

## Verification

- Fast determinism check: `True`; maximum probability difference
  `0`.
- Independent lightweight audit: `True`.

The result is descriptive feasibility evidence only. It is not economic-edge evidence, a
trading strategy, a deployable model, or a claim of achieved P&L.


## Stability and subgroup appendix

Positive log-loss months by preregistered population and comparison:
`{"late_no_open": {"T1_minus_T0": 0, "T2_minus_T1": 4}, "later": {"T1_minus_T0": 0, "T2_minus_T1": 4}, "opening": {"T1_minus_T0": 7, "T2_minus_T1": 5}, "pooled": {"T1_minus_T0": 5, "T2_minus_T1": 6}}`.

### Checkpoint increments

| group_type | group_value | comparison | log_loss_improvement | brier_improvement | top_one_change | top_two_change |
| --- | --- | --- | --- | --- | --- | --- |
| decision_ordinal | 6.000000000 | T1_minus_T0 | 0.007469436 | 0.005034370 | 0.004458599 | -0.000955414 |
| decision_ordinal | 6.000000000 | T2_minus_T1 | 0.000113323 | 0.000166385 | 0.001592357 | 0.000318471 |
| decision_ordinal | 12.000000000 | T1_minus_T0 | 0.002451754 | 0.002006201 | 0.001469438 | -0.000318471 |
| decision_ordinal | 12.000000000 | T2_minus_T1 | 0.000062470 | 0.000083769 | 0.000620181 | -0.000318471 |
| decision_ordinal | 24.000000000 | T1_minus_T0 | -0.001166187 | -0.000322366 | 0.007131151 | -0.002210675 |
| decision_ordinal | 24.000000000 | T2_minus_T1 | -0.000055327 | 0.000022084 | -0.000335233 | 0.000603419 |
| decision_ordinal | 36.000000000 | T1_minus_T0 | -0.004114656 | -0.002237175 | -0.003197750 | -0.003503185 |
| decision_ordinal | 36.000000000 | T2_minus_T1 | 0.000154636 | 0.000125777 | -0.000830633 | 0.000670466 |

### Phase, posterior-entropy, and transition-probability increments

| group_type | group_value | comparison | log_loss_improvement | brier_improvement | top_one_change | top_two_change |
| --- | --- | --- | --- | --- | --- | --- |
| phase | LATER_PHASE | T1_minus_T0 | -0.002640421 | -0.001279770 | 0.001966700 | -0.002856930 |
| phase | LATER_PHASE | T2_minus_T1 | 0.000049654 | 0.000073930 | -0.000582933 | 0.000636943 |
| phase | OPENING_PHASE | T1_minus_T0 | 0.004960595 | 0.003520285 | 0.002964018 | -0.000636943 |
| phase | OPENING_PHASE | T2_minus_T1 | 0.000087896 | 0.000125077 | 0.001106269 | 0.000000000 |
| posterior_entropy_split | HIGH | T1_minus_T0 | 0.000196902 | 0.001013238 | 0.003684986 | -0.003371412 |
| posterior_entropy_split | HIGH | T2_minus_T1 | 0.000047239 | 0.000091729 | 0.001358820 | 0.000674480 |
| posterior_entropy_split | LOW | T1_minus_T0 | 0.002021715 | 0.001215993 | 0.001374328 | -0.000293743 |
| posterior_entropy_split | LOW | T2_minus_T1 | 0.000088040 | 0.000106459 | -0.000719802 | 0.000000000 |
| transition_probability_split | HIGH | T1_minus_T0 | -0.001995672 | -0.001256330 | -0.001536809 | -0.002663524 |
| transition_probability_split | HIGH | T2_minus_T1 | 0.000082926 | 0.000121442 | 0.000994898 | -0.000009415 |
| transition_probability_split | LOW | T1_minus_T0 | 0.003690968 | 0.003026253 | 0.005675052 | -0.001011844 |
| transition_probability_split | LOW | T2_minus_T1 | 0.000057426 | 0.000081910 | -0.000326374 | 0.000581433 |

### Later opening-history subgroup increments

| group_type | group_value | comparison | log_loss_improvement | brier_improvement | top_one_change | top_two_change |
| --- | --- | --- | --- | --- | --- | --- |
| late_loop_subgroup | LATE_AFTER_OPEN_REGISTERED_LOOP | T1_minus_T0 | -0.008605338 | -0.003383756 | -0.006684495 | -0.003518155 |
| late_loop_subgroup | LATE_AFTER_OPEN_REGISTERED_LOOP | T2_minus_T1 | 0.001707580 | 0.001118647 | 0.000000000 | 0.003342248 |
| late_loop_subgroup | LATE_NO_OPEN_REGISTERED_LOOP | T1_minus_T0 | -0.002342016 | -0.001174515 | 0.002399491 | -0.002823851 |
| late_loop_subgroup | LATE_NO_OPEN_REGISTERED_LOOP | T2_minus_T1 | -0.000033286 | 0.000021667 | -0.000612095 | 0.000501605 |

### Realised-target-class increments

| group_type | group_value | comparison | log_loss_improvement | brier_improvement | top_one_change | top_two_change |
| --- | --- | --- | --- | --- | --- | --- |
| realised_target_class | NO_REGISTERED_COMPLETION | T1_minus_T0 | -0.002096496 | -0.001994170 | 0.002395244 | -0.001085383 |
| realised_target_class | NO_REGISTERED_COMPLETION | T2_minus_T1 | 0.000260277 | 0.000142885 | 0.000551139 | 0.000680677 |
| realised_target_class | REGISTERED_COMPLETION | T1_minus_T0 | -0.018776119 | -0.005782161 | -0.007487467 | -0.010026694 |
| realised_target_class | REGISTERED_COMPLETION | T2_minus_T1 | -0.001896857 | -0.000088439 | 0.003711179 | -0.001171951 |
| realised_target_class | UNREGISTERED_LOOP | T1_minus_T0 | 0.010034210 | 0.007413859 | 0.004373577 | -0.001326552 |
| realised_target_class | UNREGISTERED_LOOP | T2_minus_T1 | 0.000112729 | 0.000063062 | -0.000829584 | 0.000000000 |

### Session-bootstrap proper-score intervals

| population | comparison | metric | 80% | 90% | 95% |
| --- | --- | --- | --- | --- | --- |
| late_no_open | T1_minus_T0 | brier_improvement | [-0.002128109, -0.000235960] | [-0.002332042, 0.000005150] | [-0.002407190, 0.000241618] |
| late_no_open | T1_minus_T0 | log_loss_improvement | [-0.003824046, -0.000981239] | [-0.003960511, -0.000936391] | [-0.004005400, -0.000899700] |
| late_no_open | T2_minus_T1 | brier_improvement | [-0.000208430, 0.000131571] | [-0.000284196, 0.000168373] | [-0.000328152, 0.000228851] |
| late_no_open | T2_minus_T1 | log_loss_improvement | [-0.000335610, 0.000176994] | [-0.000387574, 0.000262559] | [-0.000502177, 0.000315642] |
| later | T1_minus_T0 | brier_improvement | [-0.002228948, -0.000325175] | [-0.002416342, -0.000195646] | [-0.002457921, 0.000034131] |
| later | T1_minus_T0 | log_loss_improvement | [-0.004191722, -0.001313655] | [-0.004373454, -0.001258536] | [-0.004455614, -0.001155634] |
| later | T2_minus_T1 | brier_improvement | [-0.000128980, 0.000186761] | [-0.000187252, 0.000214304] | [-0.000251199, 0.000270619] |
| later | T2_minus_T1 | log_loss_improvement | [-0.000241008, 0.000279693] | [-0.000261143, 0.000347690] | [-0.000390994, 0.000383409] |
| opening | T1_minus_T0 | brier_improvement | [0.002350869, 0.004868038] | [0.002086857, 0.005054104] | [0.001889699, 0.005109041] |
| opening | T1_minus_T0 | log_loss_improvement | [0.003285847, 0.006718767] | [0.002805560, 0.007058716] | [0.002617337, 0.007137099] |
| opening | T2_minus_T1 | brier_improvement | [-0.000110220, 0.000328600] | [-0.000145716, 0.000356412] | [-0.000177597, 0.000410660] |
| opening | T2_minus_T1 | log_loss_improvement | [-0.000126503, 0.000409050] | [-0.000158413, 0.000462057] | [-0.000230436, 0.000496813] |
| pooled | T1_minus_T0 | brier_improvement | [0.000336034, 0.001980007] | [0.000285764, 0.002034000] | [0.000081709, 0.002222588] |
| pooled | T1_minus_T0 | log_loss_improvement | [0.000106179, 0.002141557] | [0.000026766, 0.002653758] | [-0.000287018, 0.002801678] |
| pooled | T2_minus_T1 | brier_improvement | [-0.000070064, 0.000205361] | [-0.000095114, 0.000230184] | [-0.000129173, 0.000256132] |
| pooled | T2_minus_T1 | log_loss_improvement | [-0.000132115, 0.000217299] | [-0.000140248, 0.000241754] | [-0.000217843, 0.000301942] |

### Five-draw trajectory-null comparisons

| population | comparison | metric | real_increment | null_draws_exceeded |
| --- | --- | --- | --- | --- |
| late_no_open | T1_minus_T0 | log_loss_improvement | -0.002342016 | 0.000000000 |
| late_no_open | T1_minus_T0 | brier_improvement | -0.001174515 | 0.000000000 |
| late_no_open | T2_minus_T1 | log_loss_improvement | -0.000033286 | 3.000000000 |
| late_no_open | T2_minus_T1 | brier_improvement | 0.000021667 | 4.000000000 |
| later | T1_minus_T0 | log_loss_improvement | -0.002640421 | 0.000000000 |
| later | T1_minus_T0 | brier_improvement | -0.001279770 | 0.000000000 |
| later | T2_minus_T1 | log_loss_improvement | 0.000049654 | 5.000000000 |
| later | T2_minus_T1 | brier_improvement | 0.000073930 | 5.000000000 |
| opening | T1_minus_T0 | log_loss_improvement | 0.004960595 | 5.000000000 |
| opening | T1_minus_T0 | brier_improvement | 0.003520285 | 5.000000000 |
| opening | T2_minus_T1 | log_loss_improvement | 0.000087896 | 4.000000000 |
| opening | T2_minus_T1 | brier_improvement | 0.000125077 | 5.000000000 |
| pooled | T1_minus_T0 | log_loss_improvement | 0.001160087 | 5.000000000 |
| pooled | T1_minus_T0 | brier_improvement | 0.001120258 | 5.000000000 |
| pooled | T2_minus_T1 | log_loss_improvement | 0.000068775 | 5.000000000 |
| pooled | T2_minus_T1 | brier_improvement | 0.000099504 | 5.000000000 |

### Concentration

| population | gate | value | threshold | passed |
| --- | --- | --- | --- | --- |
| pooled | maximum_stock_share | 0.050348753 | 0.100000000 | True |
| pooled | maximum_target_class_share | 0.579331356 | 0.750000000 | True |
| opening | maximum_stock_share | 0.050151733 | 0.100000000 | True |
| opening | maximum_target_class_share | 0.688068999 | 0.750000000 | True |
| later | maximum_stock_share | 0.050547328 | 0.100000000 | True |
| later | maximum_target_class_share | 0.469735995 | 0.750000000 | True |
| late_no_open | maximum_stock_share | 0.051563821 | 0.100000000 | True |
| late_no_open | maximum_target_class_share | 0.470498732 | 0.750000000 | True |
| late_after_open | maximum_stock_share | 0.087542088 | 0.100000000 | True |
| late_after_open | maximum_target_class_share | 0.454545455 | 0.750000000 | True |

### Assessment trajectory diagnostics

| emotion | statistic | value |
| --- | --- | --- |
| arousal | change_mean | 0.012239823 |
| arousal | acceleration_mean | -0.000636992 |
| arousal | reversal_frequency | 0.434618777 |
| arousal | persistence_frequency | 0.565381223 |
| arousal | level_change_opposite_sign_frequency | 0.461717309 |
| arousal | at_local_peak_at_decision_frequency | 0.325503087 |
| conviction | change_mean | 0.041947066 |
| conviction | acceleration_mean | 0.012391125 |
| conviction | reversal_frequency | 0.585825383 |
| conviction | persistence_frequency | 0.414174617 |
| conviction | level_change_opposite_sign_frequency | 0.311232262 |
| conviction | at_local_peak_at_decision_frequency | 0.356530105 |
| frustration | change_mean | -0.048306247 |
| frustration | acceleration_mean | 0.051524947 |
| frustration | reversal_frequency | 0.545418103 |
| frustration | persistence_frequency | 0.454581897 |
| frustration | level_change_opposite_sign_frequency | 0.333841097 |
| frustration | at_local_peak_at_decision_frequency | 0.305940832 |
| tension | change_mean | 0.027890091 |
| tension | acceleration_mean | -0.023878106 |
| tension | reversal_frequency | 0.553435421 |
| tension | persistence_frequency | 0.446564579 |
| tension | level_change_opposite_sign_frequency | 0.376894091 |
| tension | at_local_peak_at_decision_frequency | 0.360538764 |
| signed_pressure | change_mean | -0.012587662 |
| signed_pressure | acceleration_mean | -0.005142370 |
| signed_pressure | reversal_frequency | 0.475827788 |
| signed_pressure | persistence_frequency | 0.524172212 |
| signed_pressure | level_change_opposite_sign_frequency | 0.362222400 |
| signed_pressure | at_local_peak_at_decision_frequency | 0.347630883 |
| signed_exhaustion | change_mean | 0.025921469 |
| signed_exhaustion | acceleration_mean | -0.015694463 |
| signed_exhaustion | reversal_frequency | 0.601058286 |
| signed_exhaustion | persistence_frequency | 0.398941714 |
| signed_exhaustion | level_change_opposite_sign_frequency | 0.299206286 |
| signed_exhaustion | at_local_peak_at_decision_frequency | 0.328068628 |

Missing anchor records: 43. The complete-case missingness ledger is
`trajectory_missingness.csv`; no alternative anchor was substituted.
