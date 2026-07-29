# Dynamic Loop Edge State Lead-Lag V1

## Decision

**leading_features_no_incremental_value**. This is an opened-data attribution experiment, not a strategy search or
prospective validation. The original one-session sign reversal is **population-confounded**;
the frozen structural feature overlay does not improve next-session state calibration.

## Hypothesis and frozen registration

V2's same-session full hierarchy lost
**-9,416.38 bps**, while its shifted-policy diagnostic made
**9,014.54 bps**. The registered post-V2 question was whether
the unchanged feature overlay at session *t* predicts the same loop/orientation's settled robust
payoff at *t+1* better than the otherwise identical hierarchy without the feature overlay. Lead 1
was primary before scoring; leads 0, 2, 3, and 5 were shape diagnostics. All V2 model, feature,
24-bar horizon, hazard, threshold, cost, settlement, and exit settings remained frozen.

The state-lead test and executable trade-delay test are separate. State targets use the explicit
within-period trading-session calendar and never turn a missing payoff into zero. The matched trade
test requires a persistent same-setup identifier and fails closed.

## Data, timestamps, and boundaries

- Opened V2 periods: 2023 and 2025; period joins reset and cannot bridge the gap.
- Forecast freeze: V2 `prediction_frozen_at`, equal to its decision timestamp.
- Feature and settled-training availability must be strictly earlier than the freeze.
- Target: robust equal-stock winsorised session net payoff at the registered 24-bar horizon,
  strictly positive for the binary event.
- V2 stores unique `opportunity_id`/`anchor_id` within a session, but no persistent cross-session
  setup or event-lineage identifier. Consequently exact delayed-trade identity is unavailable
  rather than inferred.

## Reconstruction of the V2 delay

The V2 implementation grouped by period × loop × orientation × horizon, shifted `accepted` by one
**opportunity session**, then applied that flag to current opportunities and their unchanged current
entries, exits, costs, and 24-bar outcomes. It did not shift a forecast onto the same trade. It
retained 55 accepted signals, dropped 231, and introduced 213; introduced payoff minus dropped
payoff is **18,430.92 bps**, exactly the reported sign change. Policy gaps were not always one
calendar step. There is no overlap resolver or portfolio-capacity allocator in this ledger, so the
effect is changed admission population/composition, not freed capacity.

## Primary paired state result

At lead 1, there were 2,787 paired observable cells.
Control-minus-full Brier improvement was
**-0.054737** (negative means the full model is worse; 95%
session-block interval -0.063929 to
-0.045803). Log-loss improvement was
**-0.260320** and the frozen active-state economic
increment was **-403.68 bps**. Posterior expected
payoff is identical by construction; only the frozen feature overlay changes predictive and
operational probabilities.

| target_lead_sessions | paired_observable_targets | paired_brier_improvement | paired_log_loss_improvement | paired_economic_increment_bps | brier_ci_lower | brier_ci_upper |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2801 | -0.049267 | -0.214038 | 942.152506 | -0.058516 | -0.040081 |
| 1 | 2787 | -0.054737 | -0.260320 | -403.678587 | -0.063929 | -0.045803 |
| 2 | 2769 | -0.052289 | -0.254886 | -249.341061 | -0.061078 | -0.043727 |
| 3 | 2752 | -0.047269 | -0.212269 | -683.434956 | -0.055926 | -0.038950 |
| 5 | 2722 | -0.043659 | -0.247633 | -1222.623634 | -0.053979 | -0.034655 |

No lead has positive paired Brier improvement. Lead 1 is worse than same-session Brier, not better
calibrated for t+1. Holm adjustment does not rescue the result; the signed effect is adverse.

### Calibration at leads 0 and 1

| model_name | target_lead_sessions | observable_targets | brier_score | log_loss | ece | calibration_slope | calibration_intercept | auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hierarchical_change_point | 0 | 2801 | 0.324974 | 0.971992 | 0.238520 | -0.000865 | -0.125246 | 0.487873 |
| hierarchical_change_point | 1 | 2787 | 0.328632 | 1.014223 | 0.239685 | -0.036139 | -0.132902 | 0.477357 |
| hierarchical_payoff_history_change_point | 0 | 2801 | 0.275706 | 0.757954 | 0.112271 | -0.035931 | -0.127310 | 0.500482 |
| hierarchical_payoff_history_change_point | 1 | 2787 | 0.273894 | 0.753903 | 0.111327 | -0.000316 | -0.127354 | 0.506258 |

The contextual lead-1 comparator table is:

| model_name | observable_targets | brier_score | log_loss | ece | calibration_slope | auc | active_count | mean_target_payoff_when_active_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hierarchical_change_point | 2787 | 0.328632 | 1.014223 | 0.239685 | -0.036139 | 0.477357 | 88 | -21.061682 |
| hierarchical_payoff_history_change_point | 2787 | 0.273894 | 0.753903 | 0.111327 | -0.000316 | 0.506258 | 122 | -11.883192 |
| payoff_only_change_point | 2787 | 0.296938 | 0.854785 | 0.155151 | -0.050561 | 0.484063 | 101 | -16.356629 |
| v1_60_session_selector | 2787 | 0.258234 | 0.711107 | 0.064472 | 0.050451 | 0.506466 | 1293 | -4.268479 |

Operational onset precision uses only frozen non-active-to-active state transitions, while onset
probability and survival calibration retain their separate frozen probabilities:

| model_name | onset_operational_predictions | onset_precision | onset_recall | false_onset_rate | onset_probability_brier_score | survival_brier_score | survival_log_loss | survival_ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hierarchical_change_point | 63 | 0.063492 | 0.019324 | 0.936508 | 0.070547 | 0.626026 | 3.002732 | 0.631277 |
| hierarchical_payoff_history_change_point | 88 | 0.056818 | 0.024155 | 0.943182 | 0.071185 | 0.642957 | 3.175651 | 0.645272 |

### Feature attribution and period stability

The feature increment is not monotonic with next-session payoff. Its lead-1 Spearman association
with realised payoff is **-0.0448**
(p=0.0179); its association with the
positive-payoff event is **-0.0472**
(p=0.0127). The strongest
positive-contribution bin has mean t+1 payoff
**-14.22 bps**, versus
**-5.72 bps** in the most
negative-contribution bin. No target-informed cutoff was searched.

| contribution_bin | forecasts | mean_feature_contribution | mean_future_payoff_bps | positive_payoff_rate | independent_stock_support |
| --- | --- | --- | --- | --- | --- |
| bin_1 | 355 | -0.383425 | -5.723048 | 0.478873 | 1142.000000 |
| bin_2 | 384 | -0.194992 | 21.285976 | 0.528646 | 1288.000000 |
| bin_3 | 568 | -0.061105 | -9.006209 | 0.438380 | 1747.000000 |
| bin_4 | 381 | 0.042880 | 11.269748 | 0.519685 | 1157.000000 |
| bin_5 | 1099 | 0.225353 | -14.224494 | 0.441310 | 4166.000000 |

Lead-1 Brier improvement is negative in both opened periods; economic translation is approximately
flat in 2023 and negative in 2025.

| period | paired_observable_targets | paired_brier_improvement | paired_log_loss_improvement | paired_economic_increment_bps | brier_ci_lower | brier_ci_upper |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | 1415 | -0.060328 | -0.270911 | 9.128068 | -0.071991 | -0.048411 |
| 2025 | 1372 | -0.048971 | -0.249398 | -412.806655 | -0.064367 | -0.034557 |

## Matched trade-delay result

Exact same-setup matches: **0 /
286 (0.0%)**. Therefore
restarted-horizon, constant-terminal, and twice-cost exact paired effects are unavailable, not zero.
A separately labelled structural-lineage diagnostic found
**29 /
286** different later setups: their source trades made
**-729.60 bps** and the later distinct setups
made **3,276.07 bps**. This is
composition context, not evidence for delayed execution of the original setup. Original intraday
terminal times generally precede next-session entries, so constant-terminal exposure is impossible
for those rows. MFE, MAE, exposure, and drawdown comparisons are consequently unavailable for the
zero-row exact population rather than imputed from different setups. Existing-position exits remain
unchanged.

## Stress, concentration, and episodes

At twice costs, the paired lead-1 **state-level active-set translation** is
**-63.68 bps**; no exact matched trade population
exists for an executable cost stress. Median aggregation gives
**1,100.27 bps** but still worsens Brier by
**-0.057224**, so it cannot support the hypothesis.
Both frozen hazard sensitivities also worsen Brier:

| stress_test | paired_brier_improvement | paired_log_loss_improvement | paired_economic_increment_bps |
| --- | --- | --- | --- |
| hazard_0.033333 | -0.055794 | -0.259457 | -968.456448 |
| hazard_0.071429 | -0.053277 | -0.260741 | 263.875267 |

Fully rebuilt leave-one-stock-out lead-1 calibration improves in
**0/20** exclusions and the economic increment is positive in only
**6/20**; every excluded-stock run rebuilds the payoff
panel, breadth/context, shared hierarchy, cell states, and targets.

The separately predeclared best-stock and top-five-stock removals also rebuild every stock-dependent
input. Their lead-1 Brier/economic results are
**-0.056358 /
1,263.69 bps** and
**-0.053176 /
284.35 bps**, respectively.

The rejected lead-1 economic difference is materially concentrated: top stock
**RGTI** contributes
18.2% of absolute stock allocation and the top five
contribute 58.1%; top episode **episode_0103** contributes
15.2% and the top five contribute
58.0%. The largest loop (**cycle_04**) and orientation
(**state_5**) absolute shares are
29.3% and
47.2%, respectively.

Of 215 hindsight-labelled episode rows, 61 meet the predeclared
descriptive structurally-led rule (51 have positive mean episode payoff). The
other classes are 77 simultaneous,
42 payoff-history-led, and
35 unpredicted. Rising breadth precedes
80, rising coherence precedes 87, and neither precursor appears
for 90; these overlapping descriptive counts do not establish incremental
prediction because the paired probability and rank tests are adverse. Episode labels were attached
after forecast freezing and never entered features.

## Scientific interpretation and failures

The feature overlay makes probabilities substantially more extreme without improving ranking or
calibration against future settled payoff. It neither establishes a general one-session precursor
nor turns the V2 policy shift into an executable same-setup delay. The one-session P&L reversal is
explained exactly by dropped versus introduced opportunities; changed stock/loop/time composition
is consequential, while overlap, capacity, retained-row entry clocks, holding periods, and costs
are unchanged.

The experiment cannot estimate a physical one-session delayed trade effect because V2 lacks
persistent setup lineage and because the original 24-bar intraday exit is over before the next
session. That is a data-identity limitation, not permission to substitute another setup.

## Reproducibility and safety

Run `20260715-edge-lead-lag-ba0b41bd-66be016d` used git `2341be2e22a01eec4e290667f8bde2dd08ddced6`, contract
`ba0b41bd9dab96617358f6029bca13d6acfd205842a0706cee709cada5a65e9a`, and V2 data snapshot `66be016db2be4b55e0309dfe7a1ec4ee6b99a0b490af8d9abfa11b5832dc1a6a`. Primary and
exact-rerun tables/plots are audited byte-for-byte. The runner is research-only and touches no
broker, order, deployment, position, exit, or application-runtime path.

Prospective mode rejects the opened V2 source and periods, requires a newly generated external
forecast surface and data-snapshot hash, accepts only a contemporaneous freeze, and appends later
outcomes through a separate create-only command. It does not score or execute trades.

## Exact recommendation

Do not promote or retune the V2 structural feature gate. The single most valuable next experiment
is a **prospective, execution-free holdout log** of the frozen full/control pair on genuinely
unopened sessions, with a persistent cross-session setup/event-lineage identifier added at
research-data creation time; settle outcomes append-only and revisit only after the predeclared
sample is complete.
