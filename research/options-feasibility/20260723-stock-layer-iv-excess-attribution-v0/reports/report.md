# Stock-Layer Attribution and IV-Excess Tail Quick Screen V0

## Result

Overall decision: `stock_layers_improve_ranking_but_not_positive_iv_tail`.

Component statuses:

- Daily stock: `not_supported`
- Intraday H0: `supported`
- Route competition: `not_supported`
- Cross-market mismatch: `not_supported`
- Final G4 top decile: `not_supported`
- G0 versus G4 tail: `supported`

## Frozen reconstruction

Branch C panel passed: `True`; rows `24130`; assessment rows `10265`; row mismatches `0`; selected-contract mismatches `0`; maximum feature difference `0.0`; maximum outcome difference `0.0`.

G0/C0 and G4/C1 reconstruction passed: `True`. Maximum probability differences: G0 `0.0`, G4 `0.0`.

## G0-G4 assessment metrics

| model | log_loss | brier_score | auc | average_precision | expected_calibration_error | calibration_intercept | calibration_slope | base_rate | mean_probability_realised_class | top_decile_precision | top_decile_lift | top_quintile_precision | top_quintile_lift | rows | sessions | stocks | positive_outcomes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | 0.58655429 | 0.19921836 | 0.61141718 | 0.37755012 | 0.03902956 | -0.00819676 | 1.25613884 | 0.28546003 | 0.58633157 | 0.43818880 | 1.53502681 | 0.39230448 | 1.37428865 | 10265 | 154 | 20 | 2921 |
| G1 | 0.58827742 | 0.19995997 | 0.60470527 | 0.37499626 | 0.04087865 | -0.11936024 | 1.11237214 | 0.28546003 | 0.58551330 | 0.42806693 | 1.49956872 | 0.38704477 | 1.35586327 | 10265 | 154 | 20 | 2921 |
| G2 | 0.57331842 | 0.19356186 | 0.65841594 | 0.43079451 | 0.05211934 | -0.10987242 | 1.23574431 | 0.28546003 | 0.59333322 | 0.49153961 | 1.72192098 | 0.44737185 | 1.56719613 | 10265 | 154 | 20 | 2921 |
| G3 | 0.57192941 | 0.19307884 | 0.65523669 | 0.42736602 | 0.04302908 | -0.10254793 | 1.16789967 | 0.28546003 | 0.59734672 | 0.47613132 | 1.66794393 | 0.44202822 | 1.54847675 | 10265 | 154 | 20 | 2921 |
| G4 | 0.57207160 | 0.19314455 | 0.65462995 | 0.42687388 | 0.04309972 | -0.11489600 | 1.15013634 | 0.28546003 | 0.59749100 | 0.47713095 | 1.67144572 | 0.43883609 | 1.53729434 | 10265 | 154 | 20 | 2921 |

## Every adjacent increment

| comparison | earlier_model | later_model | log_loss_improvement | brier_improvement | auc_improvement | average_precision_improvement | expected_calibration_error_improvement | top_decile_precision_improvement | top_quintile_precision_improvement | mean_realised_class_probability_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1-G0 | G0 | G1 | -0.00172313 | -0.00074161 | -0.00671191 | -0.00255386 | -0.00184909 | -0.01012187 | -0.00525971 | -0.00081827 |
| G2-G1 | G1 | G2 | 0.01495900 | 0.00639811 | 0.05371067 | 0.05579825 | -0.01124068 | 0.06347268 | 0.06032708 | 0.00781992 |
| G3-G2 | G2 | G3 | 0.00138901 | 0.00048302 | -0.00317925 | -0.00342849 | 0.00909026 | -0.01540829 | -0.00534364 | 0.00401350 |
| G4-G3 | G3 | G4 | -0.00014219 | -0.00006571 | -0.00060674 | -0.00049213 | -0.00007065 | 0.00099962 | -0.00319213 | 0.00014428 |

## Monthly stability

| comparison | positive_log_loss_months | positive_brier_months | worst_log_loss_increment | best_log_loss_increment |
| --- | --- | --- | --- | --- |
| G1-G0 | 3 | 3 | -0.00737910 | 0.00288707 |
| G2-G1 | 8 | 8 | 0.00391650 | 0.02560957 |
| G3-G2 | 6 | 6 | -0.00253534 | 0.00433302 |
| G4-G3 | 5 | 5 | -0.00158498 | 0.00045275 |

## Checkpoint stability

| comparison | group | log_loss_improvement | brier_improvement | auc_improvement | average_precision_improvement |
| --- | --- | --- | --- | --- | --- |
| G1-G0 | early_6_14 | -0.00311934 | -0.00140668 | -0.01347359 | -0.00365466 |
| G2-G1 | early_6_14 | 0.01742173 | 0.00799391 | 0.08681734 | 0.05773656 |
| G3-G2 | early_6_14 | 0.00057921 | 0.00017946 | -0.00264693 | -0.00350958 |
| G4-G3 | early_6_14 | -0.00003815 | -0.00002415 | -0.00049493 | 0.00024640 |
| G1-G0 | middle_16_24 | -0.00027063 | -0.00007337 | 0.00544944 | 0.01658618 |
| G2-G1 | middle_16_24 | 0.01480837 | 0.00619869 | 0.07687801 | 0.07907672 |
| G3-G2 | middle_16_24 | 0.00256410 | 0.00099034 | -0.00475870 | -0.00331221 |
| G4-G3 | middle_16_24 | -0.00016732 | -0.00009999 | -0.00052775 | -0.00160372 |
| G1-G0 | late_26_34 | -0.00115166 | -0.00044421 | -0.00347320 | -0.00027164 |
| G2-G1 | late_26_34 | 0.01138340 | 0.00418960 | 0.06869542 | 0.06820999 |
| G3-G2 | late_26_34 | 0.00136625 | 0.00040334 | -0.00725927 | -0.00667793 |
| G4-G3 | late_26_34 | -0.00027327 | -0.00009226 | -0.00326728 | -0.00065822 |

## Route-state stability

| comparison | group | rows | log_loss_improvement | brier_improvement | auc_improvement | average_precision_improvement |
| --- | --- | --- | --- | --- | --- | --- |
| G1-G0 | BROAD_CONFLICT | 2132 | -0.00220679 | -0.00090547 | -0.01307193 | -0.00124583 |
| G2-G1 | BROAD_CONFLICT | 2132 | 0.01402096 | 0.00555714 | 0.00887399 | 0.01361277 |
| G3-G2 | BROAD_CONFLICT | 2132 | -0.00285405 | -0.00118001 | -0.00734451 | -0.00430951 |
| G4-G3 | BROAD_CONFLICT | 2132 | -0.00010852 | 0.00002793 | -0.00227707 | -0.00135532 |
| G1-G0 | LOW_ROUTE_SUPPORT | 2883 | 0.00109998 | 0.00046351 | 0.00992844 | 0.01802003 |
| G2-G1 | LOW_ROUTE_SUPPORT | 2883 | 0.02672911 | 0.01148795 | 0.06115688 | 0.09663787 |
| G3-G2 | LOW_ROUTE_SUPPORT | 2883 | 0.00770534 | 0.00297040 | -0.00066571 | 0.00063949 |
| G4-G3 | LOW_ROUTE_SUPPORT | 2883 | 0.00021732 | 0.00006588 | -0.00012646 | 0.00005565 |
| G1-G0 | OTHER | 4999 | -0.00340014 | -0.00151030 | -0.01107891 | -0.01436014 |
| G2-G1 | OTHER | 4999 | 0.00846787 | 0.00372229 | 0.04492133 | 0.03252043 |
| G3-G2 | OTHER | 4999 | 0.00010041 | 0.00000987 | -0.00360087 | -0.00810270 |
| G4-G3 | OTHER | 4999 | -0.00023582 | -0.00012103 | -0.00086267 | 0.00023563 |
| G1-G0 | NARROWING | 251 | 0.00406386 | 0.00237613 | 0.00210882 | 0.05633796 |
| G2-G1 | NARROWING | 251 | 0.02146872 | 0.01015664 | 0.06042648 | 0.04759915 |
| G3-G2 | NARROWING | 251 | -0.00677385 | -0.00342759 | -0.00501769 | -0.00383284 |
| G4-G3 | NARROWING | 251 | -0.00236242 | -0.00111857 | -0.00565235 | -0.00427092 |

## Frozen-G4 grouped permutation attribution

| group | draws | mean_log_loss_deterioration | mean_brier_deterioration | mean_auc_deterioration | mean_average_precision_deterioration | mean_top_decile_precision_deterioration |
| --- | --- | --- | --- | --- | --- | --- |
| D | 5 | 0.00007624 | 0.00000973 | 0.00022862 | -0.00009246 | -0.00060124 |
| I | 5 | 0.01933573 | 0.00861728 | 0.04606100 | 0.06444921 | 0.08098877 |
| M | 5 | 0.00080036 | 0.00033967 | 0.00191875 | 0.00314620 | 0.00927139 |
| R | 5 | -0.00066993 | -0.00032499 | -0.00222782 | -0.00333921 | -0.00600307 |

## Group-specific null refits

| group | refits | real_beats_log_loss_nulls | real_beats_brier_nulls | real_beats_auc_nulls | real_beats_average_precision_nulls |
| --- | --- | --- | --- | --- | --- |
| D | 3 | 0 | 0 | 0 | 3 |
| I | 3 | 3 | 3 | 3 | 3 |
| M | 3 | 1 | 1 | 0 | 1 |
| R | 3 | 0 | 0 | 0 | 1 |

## G4 tails and G0 comparison

| model | tail | rows | sessions | stocks | months | mean_absolute_movement | median_absolute_movement | mean_iv_expectation | mean_iv_residual | median_iv_residual | exceed_iv_rate | iv_sigma_ratio | mean_maximum_absolute_excursion | maximum_absolute_excursion_available | trimmed_10pct_mean_iv_residual | positive_residual_rate | top_5pct_positive_residual_contribution | maximum_stock_share | maximum_month_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G4 | top_decile | 1145 | 135 | 20 | 8 | 0.01060800 | 0.00797869 | 0.00887518 | 0.00173282 | -0.00066379 | 0.47713095 | 0.95366604 |  | False | 0.00047797 | 0.47713095 | 0.33485260 | 0.12576419 | 0.15021834 |
| G4 | top_quintile | 2200 | 150 | 20 | 8 | 0.00965616 | 0.00721893 | 0.00885771 | 0.00079845 | -0.00117399 | 0.43883609 | 0.86980703 |  | False | -0.00034250 | 0.43883609 | 0.38340043 | 0.10727273 | 0.14863636 |
| G4 | top_5pct | 612 | 108 | 20 | 8 | 0.01178196 | 0.00992413 | 0.00893121 | 0.00285076 | 0.00102333 | 0.54247368 | 1.05256164 |  | False | 0.00156070 | 0.54247368 | 0.29689543 | 0.15196078 | 0.17810458 |
| G4 | top_2pct | 287 | 64 | 20 | 8 | 0.01217737 | 0.01017308 | 0.00907153 | 0.00310584 | 0.00099591 | 0.56328639 | 1.07105827 |  | False | 0.00205799 | 0.56328639 | 0.27343662 | 0.19860627 | 0.20905923 |
| G0 | top_decile | 951 | 146 | 20 | 8 | 0.00933182 | 0.00655624 | 0.00819236 | 0.00113946 | -0.00089405 | 0.43818880 | 0.90886089 |  | False | 0.00002855 | 0.43818880 | 0.41075760 | 0.09568875 | 0.18611987 |

| comparison | mean_iv_residual_difference | median_iv_residual_difference | exceed_iv_rate_difference | absolute_movement_difference | iv_sigma_ratio_difference | positive_residual_rate_difference | top_5pct_contribution_difference | G4_maximum_stock_share | G0_maximum_stock_share | G4_maximum_month_share | G0_maximum_month_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G4_top_decile-G0_top_decile | 0.00059336 | 0.00023026 | 0.03894215 | 0.00127618 | 0.04480516 | 0.03894215 | -0.07590500 | 0.12576419 | 0.09568875 | 0.15021834 | 0.18611987 |

Tail overlap:

| intersection_rows | union_rows | jaccard_overlap | G4_only_rows | G0_only_rows |
| --- | --- | --- | --- | --- |
| 449 | 1647 | 0.27261688 | 696 | 502 |

Incremental top-decile capture:

| comparison | entering_rows | leaving_rows | new_positive_targets_entering_top_decile | positive_targets_leaving_top_decile | net_change_captured_positive_targets | mean_iv_residual_entering_top_decile | mean_iv_residual_leaving_top_decile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1-G0 | 196 | 209 | 76 | 88 | -12 | -0.00049510 | 0.00066277 |
| G2-G1 | 678 | 442 | 343 | 162 | 181 | 0.00173289 | -0.00023647 |
| G3-G2 | 181 | 213 | 61 | 95 | -34 | 0.00015237 | 0.00086363 |
| G4-G3 | 30 | 27 | 15 | 15 | 0 | 0.00173209 | 0.00211914 |

## Coarse fixed-prediction bootstrap (80% intervals)

| statistic | point_estimate | lower | upper | draws |
| --- | --- | --- | --- | --- |
| G1_minus_G0_log_loss_improvement | -0.00172313 | -0.00194324 | -0.00066387 | 10 |
| G1_minus_G0_brier_improvement | -0.00074161 | -0.00081182 | -0.00032245 | 10 |
| G1_minus_G0_auc_improvement | -0.00671191 | -0.00808516 | -0.00072245 | 10 |
| G1_minus_G0_average_precision_improvement | -0.00255386 | -0.00683293 | 0.00226437 | 10 |
| G1_minus_G0_top_decile_precision_improvement | -0.01012187 | -0.02215980 | 0.00056022 | 10 |
| G2_minus_G1_log_loss_improvement | 0.01495900 | 0.01183826 | 0.01643275 | 10 |
| G2_minus_G1_brier_improvement | 0.00639811 | 0.00506305 | 0.00700410 | 10 |
| G2_minus_G1_auc_improvement | 0.05371067 | 0.04145046 | 0.05566125 | 10 |
| G2_minus_G1_average_precision_improvement | 0.05579825 | 0.04544086 | 0.06094547 | 10 |
| G2_minus_G1_top_decile_precision_improvement | 0.06347268 | 0.03321138 | 0.07303837 | 10 |
| G3_minus_G2_log_loss_improvement | 0.00138901 | 0.00078016 | 0.00172189 | 10 |
| G3_minus_G2_brier_improvement | 0.00048302 | 0.00024277 | 0.00067130 | 10 |
| G3_minus_G2_auc_improvement | -0.00317925 | -0.00417808 | -0.00173755 | 10 |
| G3_minus_G2_average_precision_improvement | -0.00342849 | -0.00819824 | -0.00088066 | 10 |
| G3_minus_G2_top_decile_precision_improvement | -0.01540829 | -0.02375333 | -0.00363266 | 10 |
| G4_minus_G3_log_loss_improvement | -0.00014219 | -0.00029672 | 0.00013578 | 10 |
| G4_minus_G3_brier_improvement | -0.00006571 | -0.00012385 | 0.00006977 | 10 |
| G4_minus_G3_auc_improvement | -0.00060674 | -0.00101543 | 0.00033019 | 10 |
| G4_minus_G3_average_precision_improvement | -0.00049213 | -0.00082844 | 0.00053357 | 10 |
| G4_minus_G3_top_decile_precision_improvement | 0.00099962 | -0.00460826 | 0.00502059 | 10 |
| G4_top_decile_mean_iv_residual | 0.00173282 | 0.00114047 | 0.00205148 | 10 |
| G4_top_decile_median_iv_residual | -0.00066379 | -0.00130665 | -0.00013560 | 10 |
| G4_top_decile_exceed_iv_rate | 0.47713095 | 0.44971792 | 0.49376780 | 10 |
| G4_minus_G0_top_decile_mean_iv_residual_difference | 0.00059336 | 0.00000392 | 0.00094519 | 10 |
| G4_minus_G0_top_decile_median_iv_residual_difference | 0.00023026 | -0.00008405 | 0.00031915 | 10 |
| G4_minus_G0_top_decile_exceed_iv_rate_difference | 0.03894215 | 0.01241450 | 0.05657852 | 10 |
| G4_minus_G0_top_decile_iv_sigma_ratio_difference | 0.04480516 | -0.01222209 | 0.08299225 | 10 |

## Reproducibility

Determinism passed: `True`; joined-row mismatches `0`; maximum model probability difference `0.0`; tail membership mismatches `0`. Bootstrap, grouped permutations, and null refits were not repeated.
Independent audit status: `passed`. The auditor rebuilt panel layers, chronology, outcomes, probabilities, metrics, tail membership, grouped permutations, null metrics, bootstrap intervals, and decision logic without refitting null models.

## Plots

- `/Users/michaelsalerno/Documents/Codex/2026-07-23-you-are-working-in-the-github-4/research/options-feasibility/20260723-stock-layer-iv-excess-attribution-v0/reports/g0_g4_metric_ladder.png`
- `/Users/michaelsalerno/Documents/Codex/2026-07-23-you-are-working-in-the-github-4/research/options-feasibility/20260723-stock-layer-iv-excess-attribution-v0/reports/g0_g4_tail_comparison.png`

## Scientific boundary

This is a retrospective, research-only, previous-close-options-conditioned underlying-movement attribution and tail-feasibility screen. It does not test option P&L, contracts, fills, DTE strategies, direction, entries, exits, execution, economic edge, prospective validity, trading utility, or a deployable strategy.
