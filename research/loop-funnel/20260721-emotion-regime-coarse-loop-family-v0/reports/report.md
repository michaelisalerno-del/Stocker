# Emotion × Regime-Mix Coarse Loop-Family Funnel V0

Retrospective, research-only, structural quick feasibility screen. Economic outcomes were not opened; execution and strategy promotion remained disabled.

Decision: `descriptive_coarse_funnel_only`

## Support

- Development: 9284 rows, 234 sessions, 20 stocks.
- Assessment: 6261 rows, 157 sessions, 20 stocks, 8 months.
- Ties excluded: 4; unavailable excluded: 0.
- Frozen target variant: three_class_fallback; classes: REGISTERED_COMPLETION, UNREGISTERED_LOOP, NO_REGISTERED_COMPLETION.

## Pooled proper scores and ranking

```text
model  multiclass_log_loss  multiclass_brier  top_one_accuracy  top_two_accuracy  top_three_accuracy  mean_reciprocal_rank  mean_probability_realised_class  expected_calibration_error  prediction_entropy  effective_candidate_count
   M0             0.678060          0.404750          0.702754          0.949896                 1.0              0.843026                         0.587983                    0.031782            0.703725                   2.050738
   M1             0.674652          0.403273          0.704345          0.950214                 1.0              0.843875                         0.592118                    0.014561            0.693609                   2.035655
   M2             0.672908          0.402718          0.703683          0.950851                 1.0              0.843650                         0.593470                    0.012247            0.688954                   2.028801
```

## Funnel diagnostics

```text
model  mean_prediction_entropy  median_prediction_entropy  mean_effective_candidate_count  mean_probability_realised_class  realised_class_top_one_percent  realised_class_top_two_percent  realised_class_top_three_percent  rows
   M0                 0.703725                   0.708282                        2.050738                         0.587983                       70.275357                       94.989571                             100.0  6261
   M1                 0.693609                   0.700085                        2.035655                         0.592118                       70.434499                       95.021418                             100.0  6261
   M2                 0.688954                   0.703363                        2.028801                         0.593470                       70.368291                       95.085112                             100.0  6261
```

## Binding comparisons

- M1 versus M0: log-loss improvement 0.00340802; Brier improvement 0.00147667; top-two change 0.00031847; passes=False.
- M2 versus M1: log-loss improvement 0.00174322; Brier improvement 0.00055504; top-two change 0.00063694; passes=False.

## Resampling

The paired session bootstrap used 50 fixed draws. The within-slate behavioural null used 10 fixed draws. Full 90% and 95% bootstrap intervals and real-result null percentiles are in the CSV artifacts.

Lower prediction entropy is treated as descriptive unless proper scores also improve. This screen is not prospective validation and supplies no evidence about economic or trading utility.
