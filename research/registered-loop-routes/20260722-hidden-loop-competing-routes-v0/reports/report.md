# Hidden-Loop Competing Routes and Registered-Loop Recurrence Quick Screen V0

Decision: `blocked_sequential_model_support_failure`.

- Target A precursor: `insufficient_support`
- Target B precursor: `insufficient_support`
- Target C recurrence: `insufficient_support`
- Hidden 2→3→2 diversion: `insufficient_support`
- Registered-history increment: `insufficient_support`
- Hidden-history increment: `insufficient_support`
- Protected rows materialised: `0`
- Determinism: `True`
- Independent audit: `True`

This is retrospective, research-only, observable structural feasibility evidence. Economic and
directional outcomes stayed closed. It is not prospective validation, trading utility, or a
deployable strategy.

## Development-frozen route classes

Retained exact targets: `["loop_p_2-5-6-2", "loop_p_2-6-2", "loop_p_4-6-4"]`.

Final classes: `["NO_REGISTERED_COMPLETION", "OTHER_REGISTERED_COMPLETION", "loop_p_2-5-6-2", "loop_p_2-6-2", "loop_p_4-6-4"]`.

## Corrected transition census

| period | hypothesis_id | lookback_bars | eligible_events | ineligible_events | precursor_events | observed_prevalence | sessions | stocks | months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | H3 | 6 | 203 | 7 | 61 | 0.300492611 | 42 | 16 | 12 |
| development | H4 | 6 | 2468 | 87 | 104 | 0.0421393841 | 74 | 16 | 12 |
| assessment | H3 | 6 | 239 | 3 | 70 | 0.292887029 | 40 | 16 | 8 |
| assessment | H4 | 6 | 1327 | 29 | 50 | 0.0376789751 | 33 | 15 | 8 |
| development | H1 | 12 | 108 | 27 | 53 | 0.490740741 | 37 | 16 | 11 |
| development | H2 | 12 | 125 | 95 | 38 | 0.304 | 26 | 15 | 11 |
| development | H3 | 12 | 191 | 19 | 67 | 0.35078534 | 42 | 17 | 11 |
| assessment | H1 | 12 | 52 | 21 | 22 | 0.423076923 | 18 | 13 | 7 |
| assessment | H2 | 12 | 85 | 44 | 30 | 0.352941176 | 17 | 14 | 6 |
| assessment | H3 | 12 | 210 | 32 | 82 | 0.39047619 | 41 | 16 | 8 |

## Matched transition null

| period | hypothesis_id | lookback_bars | observed_prevalence | mean_null_prevalence | enrichment | null_percentile | null_10th_percentile | null_90th_percentile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessment | H1 | 12 | 0.423076923 | 0.246923077 | 0.176153846 | 100 | 0.153846154 | 0.307692308 |
| assessment | H2 | 12 | 0.352941176 | 0.187294118 | 0.165647059 | 100 | 0.134117647 | 0.242352941 |
| assessment | H3 | 6 | 0.292887029 | 0.0374895397 | 0.25539749 | 100 | 0.0225941423 | 0.0543933054 |
| assessment | H3 | 12 | 0.39047619 | 0.055047619 | 0.335428571 | 100 | 0.0428571429 | 0.0666666667 |
| assessment | H4 | 6 | 0.0376789751 | 0.0772871138 | -0.0396081387 | 0 | 0.0679728711 | 0.0845516202 |
| development | H1 | 12 | 0.490740741 | 0.237037037 | 0.253703704 | 100 | 0.188888889 | 0.277777778 |
| development | H2 | 12 | 0.304 | 0.23168 | 0.07232 | 100 | 0.1872 | 0.28 |
| development | H3 | 6 | 0.300492611 | 0.035270936 | 0.265221675 | 100 | 0.0197044335 | 0.0522167488 |
| development | H3 | 12 | 0.35078534 | 0.052565445 | 0.298219895 | 100 | 0.0356020942 | 0.0732984293 |
| development | H4 | 6 | 0.0421393841 | 0.101410049 | -0.0592706645 | 0 | 0.0953808752 | 0.107941653 |

Multiplicity across the four fixed hypotheses:

| hypothesis_id | p_value | q_value | q_le_0_10 |
| --- | --- | --- | --- |
| H1 | 0.0384615385 | 0.0384615385 | True |
| H2 | 0.0384615385 | 0.0384615385 | True |
| H3 | 0.0384615385 | 0.0384615385 | True |
| H4 | 0.0384615385 | 0.0384615385 | True |

## C0/C1/C2 pooled assessment metrics

| model | multiclass_log_loss | multiclass_brier | top_one_accuracy | top_two_accuracy | mean_reciprocal_rank | mean_probability_realised_class | expected_calibration_error | prediction_entropy | effective_candidate_count | rows | unique_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 0.527250995 | 0.302243611 | 0.795292104 | 0.970932633 | 0.892224452 | 0.678344861 | 0.0360680957 | 0.603738198 | 1.82894299 | 6189 | 960 |
| C1 | 0.521810738 | 0.299163946 | 0.801981321 | 0.97126433 | 0.89557538 | 0.680804263 | 0.0333094894 | 0.597942774 | 1.81837414 | 6189 | 960 |
| C2 | 0.521859577 | 0.299466737 | 0.801535758 | 0.971412851 | 0.895364975 | 0.681134089 | 0.0323557164 | 0.595093886 | 1.81320117 | 6189 | 960 |

## Target-specific model contrasts

| contrast_id | probability_effect_original_minus_counterfactual | observed_outcome_rate | rows | unique_candidates | sessions | stocks |
| --- | --- | --- | --- | --- | --- | --- |
| A_hidden_5_6_5_to_target_a | -0.00123068623 | 0.018960972 | 204 | 43 | 31 | 19 |
| B_hidden_5_6_5_to_target_b | -0.000893535201 | 0 | 204 | 43 | 31 | 19 |
| C_recent_4_6_4_to_target_c | 0.174733809 | 0.570089154 | 125 | 35 | 25 | 15 |
| D_hidden_2_3_2_to_any_registered | -0.00948736408 | 0.0489678349 | 340 | 96 | 68 | 17 |

## Same-stage matched route comparisons

| precursor | outcome | treated_rows | control_relation_rows | treated_completion_rate | matched_control_completion_rate | treated_minus_control_rate |
| --- | --- | --- | --- | --- | --- | --- |
| hidden_5_6_5 | loop_p_2-5-6-2 | 1 | 5 | 0 | 0 | 0 |
| hidden_5_6_5 | loop_p_2-6-2 | 1 | 5 | 0 | 0 | 0 |
| hidden_5_6_5 | ANY_REGISTERED_COMPLETION | 1 | 5 | 0 | 0 | 0 |
| hidden_2_3_2 | ANY_REGISTERED_COMPLETION | 6 | 43 | 0 | 0.121296296 | -0.121296296 |
| registered_4_6_4 | loop_p_4-6-4 | 0 | 0 |  |  |  |

## 80% whole-session bootstrap intervals

| metric | value | lower | upper |
| --- | --- | --- | --- |
| A_hidden_5_6_5_to_target_a_probability_effect | -0.00115154782 | -0.0014179698 | -0.000789372656 |
| B_hidden_5_6_5_to_target_b_probability_effect | -0.000855811512 | -0.00106586379 | -0.000642011682 |
| C1_minus_C0_brier_improvement | 0.00312396956 | 0.0015058759 | 0.00488517911 |
| C1_minus_C0_log_loss_improvement | 0.00548326828 | 0.00365269408 | 0.00707050052 |
| C1_minus_C0_top_two_change | 0.000595822079 | -0.00131220482 | 0.00264410875 |
| C2_minus_C1_brier_improvement | -0.000378457313 | -0.00092569307 | 0.000168495958 |
| C2_minus_C1_log_loss_improvement | -0.000258755847 | -0.00206589717 | 0.00117722466 |
| C2_minus_C1_top_two_change | -5.38699888e-05 | -0.0010938403 | 0.000771656852 |
| C_recent_4_6_4_to_target_c_probability_effect | 0.174001199 | 0.169018736 | 0.179091783 |
| D_hidden_2_3_2_to_any_registered_probability_effect | -0.00950631254 | -0.0100848953 | -0.00890494278 |
| candidate_level_any_registered_rate | 0.240303877 | 0.22446706 | 0.250793188 |

## Five-draw hidden-history null

| metric | real_increment | null_increment | null_draws_exceeded |
| --- | --- | --- | --- |
| multiclass_brier_improvement | -0.000302790083 | -9.15538888e-05 | 1 |
| multiclass_log_loss_improvement | -4.88390215e-05 | 7.95215209e-05 | 2 |
| top_two_accuracy_change | 0.000148520914 | 0.000640620209 | 1 |

## Boundary

No account, position, order, broker, P&L, MFE, MAE, direction, entry, exit, stop, target,
portfolio-sizing, deployment, or production runtime surface was accessed or modified.
