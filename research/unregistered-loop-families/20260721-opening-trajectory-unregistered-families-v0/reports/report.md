# Opening Behavioural Trajectory → Unregistered Loop Families Quick Screen V0

Decision: `opening_trajectories_predict_unregistered_events_only`.

Stage B status: `hidden_family_prediction_not_supported`.

This is retrospective, observable-only structural feasibility evidence. Economic and
directional outcomes remained closed; no trading, execution, broker, or deployment surface was
opened.

## Support

- Opening assessment rows: 6261.
- Sessions/stocks/months: 157 / 20 / 8.
- Assessment outcomes: `{"NO_REGISTERED_COMPLETION": 4308, "REGISTERED_COMPLETION": 312, "UNREGISTERED_LOOP": 1641}`.
- Trajectory retention: 1.000000000.
- Stage B assessment family rows: 1641.
- Stage B family counts: `{"OTHER_UNREGISTERED_FAMILY": 684, "unregistered_primitive_like__2-3-2": 211, "unregistered_primitive_like__2-5-2": 75, "unregistered_primitive_like__4-7-4": 74, "unregistered_primitive_like__5-6-5": 597}`.

## Stage A occurrence metrics

| model | log_loss | brier_score | auc | average_precision | top_decile_precision | top_decile_lift | top_quintile_precision | top_quintile_lift | mean_probability_realised_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U0 | 0.519242576 | 0.172538110 | 0.715535121 | 0.458900696 | 0.557116691 | 2.126237559 | 0.498487343 | 1.902478475 | 0.649299492 |
| U1 | 0.515027173 | 0.171076405 | 0.724741676 | 0.465798261 | 0.563203854 | 2.149469235 | 0.497468511 | 1.898590098 | 0.651505156 |

## Registered-completion diagnostic

| model | log_loss | brier_score | auc | average_precision | mean_probability_realised_class |
| --- | --- | --- | --- | --- | --- |
| R0 | 0.215255129 | 0.056462056 | 0.810691535 | 0.283368690 | 0.859338093 |
| R1 | 0.214912357 | 0.056460824 | 0.812176775 | 0.280995073 | 0.859123195 |

## Hidden-family support

| period | hidden_family_class | outcomes | sessions | stocks | months | maximum_stock_share |
| --- | --- | --- | --- | --- | --- | --- |
| development | unregistered_primitive_like__5-6-5 | 967 | 215 | 20 | 12 | 0.124095140 |
| assessment | unregistered_primitive_like__5-6-5 | 597 | 125 | 20 | 8 | 0.082077052 |
| development | unregistered_primitive_like__2-3-2 | 447 | 157 | 20 | 12 | 0.125279642 |
| assessment | unregistered_primitive_like__2-3-2 | 211 | 94 | 20 | 8 | 0.151658768 |
| development | unregistered_primitive_like__2-5-2 | 125 | 92 | 20 | 12 | 0.104000000 |
| assessment | unregistered_primitive_like__2-5-2 | 75 | 52 | 20 | 8 | 0.213333333 |
| development | unregistered_primitive_like__4-7-4 | 99 | 66 | 17 | 11 | 0.272727273 |
| assessment | unregistered_primitive_like__4-7-4 | 74 | 53 | 17 | 8 | 0.135135135 |
| development | OTHER_UNREGISTERED_FAMILY | 1038 | 227 | 20 | 12 | 0.081888247 |
| assessment | OTHER_UNREGISTERED_FAMILY | 684 | 147 | 20 | 8 | 0.077485380 |

## Stage B pooled metrics

| model | multiclass_log_loss | multiclass_brier | top_one_accuracy | top_two_accuracy | mean_probability_realised_family | prediction_entropy | effective_candidate_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | 0.734354810 | 0.410206951 | 0.699194678 | 0.918874255 | 0.567770598 | 0.832051796 | 2.298028993 |
| F1 | 0.730324042 | 0.410174601 | 0.698789529 | 0.918842269 | 0.569821403 | 0.822578425 | 2.276361706 |

## Fixed trajectory-group attribution

| trajectory_group | sum_absolute_standardised_coefficients | signed_standardised_coefficient_sum | log_loss_deterioration | brier_deterioration | auc_deterioration |
| --- | --- | --- | --- | --- | --- |
| AROUSAL_TRAJECTORY | 0.117464261 | 0.064356207 | 0.002444394 | 0.000975719 | 0.004997345 |
| CONVICTION_TRAJECTORY | 0.162130346 | 0.162130346 | 0.003240829 | 0.001338184 | 0.005781929 |
| FRUSTRATION_TRAJECTORY | 0.130190621 | -0.030586864 | 0.000625420 | 0.000250493 | 0.000935624 |
| TENSION_TRAJECTORY | 0.023433143 | 0.023433143 | 0.000061924 | 0.000026248 | 0.000227399 |
| SIGNED_PRESSURE_TRAJECTORY | 0.231950546 | -0.231950546 | 0.000689310 | 0.000140725 | 0.002970482 |
| SIGNED_EXHAUSTION_TRAJECTORY | 0.063966725 | 0.024858558 | 0.000026034 | 0.000005714 | 0.000093458 |

## Pooled bootstrap intervals

| stage | comparison | metric | interval_level | lower | upper |
| --- | --- | --- | --- | --- | --- |
| A_diagnostic | R1_minus_R0 | brier_improvement | 0.800000000 | -0.000272249 | 0.000219386 |
| A_diagnostic | R1_minus_R0 | brier_improvement | 0.900000000 | -0.000297159 | 0.000277525 |
| A_diagnostic | R1_minus_R0 | brier_improvement | 0.950000000 | -0.000322241 | 0.000307158 |
| A_diagnostic | R1_minus_R0 | log_loss_improvement | 0.800000000 | -0.000594674 | 0.001007386 |
| A_diagnostic | R1_minus_R0 | log_loss_improvement | 0.900000000 | -0.000696213 | 0.001048417 |
| A_diagnostic | R1_minus_R0 | log_loss_improvement | 0.950000000 | -0.000817123 | 0.001312744 |
| A_primary | U1_minus_U0 | auc_improvement | 0.800000000 | 0.007011949 | 0.010624731 |
| A_primary | U1_minus_U0 | auc_improvement | 0.900000000 | 0.004414530 | 0.011615086 |
| A_primary | U1_minus_U0 | auc_improvement | 0.950000000 | 0.003279026 | 0.012380264 |
| A_primary | U1_minus_U0 | brier_improvement | 0.800000000 | 0.000871968 | 0.001946689 |
| A_primary | U1_minus_U0 | brier_improvement | 0.900000000 | 0.000494186 | 0.001992234 |
| A_primary | U1_minus_U0 | brier_improvement | 0.950000000 | 0.000238843 | 0.002076912 |
| A_primary | U1_minus_U0 | log_loss_improvement | 0.800000000 | 0.002717289 | 0.005046147 |
| A_primary | U1_minus_U0 | log_loss_improvement | 0.900000000 | 0.001748302 | 0.005384974 |
| A_primary | U1_minus_U0 | log_loss_improvement | 0.950000000 | 0.001156407 | 0.005774200 |
| A_primary | U1_minus_U0 | top_decile_precision_improvement | 0.800000000 | -0.006984091 | 0.019628067 |
| A_primary | U1_minus_U0 | top_decile_precision_improvement | 0.900000000 | -0.007877895 | 0.021272298 |
| A_primary | U1_minus_U0 | top_decile_precision_improvement | 0.950000000 | -0.009268361 | 0.024103449 |
| B | F1_minus_F0 | multiclass_brier_improvement | 0.800000000 | -0.002383555 | 0.001705539 |
| B | F1_minus_F0 | multiclass_brier_improvement | 0.900000000 | -0.003017734 | 0.002665875 |
| B | F1_minus_F0 | multiclass_brier_improvement | 0.950000000 | -0.003411803 | 0.003396976 |
| B | F1_minus_F0 | multiclass_log_loss_improvement | 0.800000000 | 0.000266512 | 0.007164585 |
| B | F1_minus_F0 | multiclass_log_loss_improvement | 0.900000000 | -0.000054348 | 0.007619916 |
| B | F1_minus_F0 | multiclass_log_loss_improvement | 0.950000000 | -0.000653643 | 0.008403995 |
| B | F1_minus_F0 | top_two_accuracy_improvement | 0.800000000 | -0.001916267 | 0.002013424 |
| B | F1_minus_F0 | top_two_accuracy_improvement | 0.900000000 | -0.002831898 | 0.002338970 |
| B | F1_minus_F0 | top_two_accuracy_improvement | 0.950000000 | -0.003611822 | 0.002939618 |

## Five-draw trajectory null

| comparison | metric | real_increment | null_draws_exceeded |
| --- | --- | --- | --- |
| U1_minus_U0 | log_loss_improvement | 0.004215404 | 5.000000000 |
| U1_minus_U0 | brier_improvement | 0.001461705 | 5.000000000 |
| R1_minus_R0 | log_loss_improvement | 0.000342772 | 5.000000000 |
| R1_minus_R0 | brier_improvement | 0.000001232 | 4.000000000 |
| F1_minus_F0 | multiclass_log_loss_improvement | 0.004030768 | 5.000000000 |
| F1_minus_F0 | multiclass_brier_improvement | 0.000032350 | 5.000000000 |

## Verification

- Determinism check: `True`; maximum probability difference
  `0.0`.
- Independent lightweight audit: `True`.

The findings are not prospective validation, economic-edge evidence, trading utility, or P&L.
