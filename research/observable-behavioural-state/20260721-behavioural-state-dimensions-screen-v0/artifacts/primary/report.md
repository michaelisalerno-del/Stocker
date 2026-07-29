# Observable Behavioural-State Dimensions Screen V0 report

**Decision:** `behavioural_descriptions_only_no_predictive_increment`

This is a retrospective, research-only, observable-only feasibility screen. It is not prospective validation, achieved P&L, a strategy, or executable-edge evidence. The behavioural vocabulary describes continuous participant behaviour; the stock is not assigned literal emotions.

## Support and boundary

- Development rows / sessions / stocks / large moves: 9287 / 234 / 20 / 2310.
- Assessment rows / sessions / stocks / large moves: 6262 / 157 / 20 / 1309.
- Exact predecessor rows: 15617; eligibility exclusions: 68.
- Assessment dates: 2025-01-01 through 2025-08-22; protected rows materialised: 0.
- Decision checkpoints: completed five-minute bars 6 (10:00) and 12 (10:30), America/New_York.

## Movement models

| Model | Brier | Log loss | AUC |
|---|---:|---:|---:|
| P0 | 0.167080435 | 0.517497266 | 0.511348617 |
| P1 | 0.157758121 | 0.494048518 | 0.668069155 |
| P2 | 0.156512976 | 0.488210086 | 0.669225325 |
| P3 | 0.156898142 | 0.488873548 | 0.667873735 |

## Direction models among actual large moves

| Model | Brier | Log loss | AUC |
|---|---:|---:|---:|
| D0 | 0.247652103 | 0.688434583 | 0.540945204 |
| D1 | 0.248670949 | 0.690457080 | 0.528091764 |
| D2 | 0.252038205 | 0.698662434 | 0.527696635 |
| D3 | 0.253431033 | 0.701930378 | 0.523140896 |

## Gate results

- P2 versus P1 passes: `False`; Brier improvement 0.00124514421555; log-loss improvement 0.00583843215698.
- D2 versus D1 passes: `False`; Brier improvement -0.00336725591769; log-loss improvement -0.00820535374813.
- P3 versus P2 passes: `False`.
- D3 versus D2 passes: `False`.

## Bootstrap intervals

- `D2_minus_D1_brier_improvement`: 90% [-0.0068718386716, 0.000129442452498]; 95% [-0.00704222085385, 0.000431011192224].
- `D2_minus_D1_log_loss_improvement`: 90% [-0.0160598413014, -0.000861510663266]; 95% [-0.0164097979396, 0.000477154327921].
- `D3_minus_D2_brier_improvement`: 90% [-0.00262153366152, -0.000327605256276]; 95% [-0.00295551324527, -0.000193991403628].
- `P2_minus_P1_brier_improvement`: 90% [-3.66508331348e-05, 0.00241216393908]; 95% [-0.000244806844533, 0.00272859871953].
- `P2_minus_P1_log_loss_improvement`: 90% [0.00213104162377, 0.00944034806132]; 95% [0.00121580935927, 0.00994572489208].
- `P3_minus_P2_brier_improvement`: 90% [-0.00080603208427, -9.214238271e-06]; 95% [-0.000902101190681, 8.21457778209e-05].
- `behavioural_dimensions_minus_predecessor_return_after_20bps`: 90% [-66.7684124905, 20.3766068275]; 95% [-86.5102065899, 25.7776921492].
- `conjunction_minus_behavioural_dimensions_return_after_20bps`: 90% [-57.2726078556, 1.45881553998]; 95% [-69.3209789116, 6.20691104567].

## Bundled within-slate null

- `D2_minus_D1_brier_improvement`: real -0.00336725591769; null q90 4.98878703572e-05; percentile 0.100.
- `D2_minus_D1_log_loss_improvement`: real -0.00820535374813; null q90 -6.27023057233e-05; percentile 0.060.
- `D3_minus_D2_brier_improvement`: real -0.00139282766626; null q90 0.000226582453636; percentile 0.200.
- `P2_minus_P1_brier_improvement`: real 0.00124514421555; null q90 0.00127797640298; percentile 0.860.
- `P2_minus_P1_log_loss_improvement`: real 0.00583843215698; null q90 0.00380532024382; percentile 1.000.
- `P3_minus_P2_brier_improvement`: real -0.000385165174707; null q90 6.88769484084e-05; percentile 0.000.
- `behavioural_system_minus_predecessor_delayed_return_after_20bps`: real -25.380983515; null q90 11.8125330928; percentile 0.100.

## Delayed economic-reference diagnostic

- `behavioural_conjunctions` at 0 bps: signed gross -13.152445 bps; signed cohort-relative -7.604015 bps.
- `behavioural_conjunctions` at 10 bps: signed gross -23.152445 bps; signed cohort-relative -17.604015 bps.
- `behavioural_conjunctions` at 20 bps: signed gross -33.152445 bps; signed cohort-relative -27.604015 bps.
- `behavioural_dimensions` at 0 bps: signed gross 7.113808 bps; signed cohort-relative 20.794853 bps.
- `behavioural_dimensions` at 10 bps: signed gross -2.886192 bps; signed cohort-relative 10.794853 bps.
- `behavioural_dimensions` at 20 bps: signed gross -12.886192 bps; signed cohort-relative 0.794853 bps.
- `highest_frozen_movement_probability` at 0 bps: signed gross 51.780191 bps; signed cohort-relative 45.562634 bps.
- `highest_frozen_movement_probability` at 10 bps: signed gross 41.780191 bps; signed cohort-relative 35.562634 bps.
- `highest_frozen_movement_probability` at 20 bps: signed gross 31.780191 bps; signed cohort-relative 25.562634 bps.
- `highest_open_to_decision_relative_momentum` at 0 bps: signed gross 94.500544 bps; signed cohort-relative 93.848286 bps.
- `highest_open_to_decision_relative_momentum` at 10 bps: signed gross 84.500544 bps; signed cohort-relative 83.848286 bps.
- `highest_open_to_decision_relative_momentum` at 20 bps: signed gross 74.500544 bps; signed cohort-relative 73.848286 bps.
- `predecessor` at 0 bps: signed gross 30.826384 bps; signed cohort-relative 46.175837 bps.
- `predecessor` at 10 bps: signed gross 20.826384 bps; signed cohort-relative 36.175837 bps.
- `predecessor` at 20 bps: signed gross 10.826384 bps; signed cohort-relative 26.175837 bps.
- `random_within_slate` at 0 bps: signed gross 10.773607 bps; signed cohort-relative 25.229438 bps.
- `random_within_slate` at 10 bps: signed gross 0.773607 bps; signed cohort-relative 15.229438 bps.
- `random_within_slate` at 20 bps: signed gross -9.226393 bps; signed cohort-relative 5.229438 bps.
- `strongest_reversal` at 0 bps: signed gross -48.147173 bps; signed cohort-relative -79.403810 bps.
- `strongest_reversal` at 10 bps: signed gross -58.147173 bps; signed cohort-relative -89.403810 bps.
- `strongest_reversal` at 20 bps: signed gross -68.147173 bps; signed cohort-relative -99.403810 bps.

The economic diagnostic is delayed and gross apart from synthetic friction. It cannot rescue a failed proper-score gate and is not achieved P&L.

Exact rerun passed: `True`. Independent audit passed: `True`.
