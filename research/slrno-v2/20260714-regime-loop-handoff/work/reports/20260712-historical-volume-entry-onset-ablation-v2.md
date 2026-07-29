# Historical-volume entry-onset ablation V2

Date: 2026-07-12

`research_only: true`  
`live_ordering_enabled: false`  
`order_placement: disabled`

Volume label: `historical_volume` — provider volume attached to historical five-minute OHLCV bars. It is not exchange-wide consolidated volume, buyer/seller initiated volume, order flow, quote count, tick count, or order-book depth.

## Decision

Provider historical volume adds a small, stable amount of probability information beyond price-only OHLC features, but it does not produce a retained directional entry-onset alert.

The probability hypothesis passed at 6, 12, and 24 bars: both multiclass log loss and Brier improved in the pooled result, at least four of six months improved both, and all 22 unchanged-prediction stock deletions improved both.

The broad entry hypothesis failed. Volume-selected long and short alert timestamps did not improve conservative correct-first precision, rapid confirmation, directional dominance, and pre-confirmation adversity together versus matched price-only HGB controls.

The correct interpretation is:

> Historical volume is useful mainly as an activity/movement-timing feature. It does not reliably tell us which direction should be entered.

One narrower component is worth a future test: 24-bar long alerts selected with historical volume had materially faster correct confirmation, despite no reliable improvement in overall long-first precision.

## Status and chronology

This is a separately versioned experiment. The frozen bar-only V1 runner and artifacts were not changed.

The 2024 price-path outcomes were already known before this volume question was frozen. These results are therefore post-outcome internal development, not untouched validation. Each monthly probability remained causal: no score month used its own outcome, while completed earlier months could train later expanding folds.

No 2023, 2025, or 2026 rows were read.

## Data coverage

Among 424,583 accepted regular-session price bars:

- 424,472 had finite positive `historical_volume`;
- 111 were missing or nonfinite;
- none were zero or negative;
- median provider volume was 72,435.5;
- the observed range was 1 to 104,903,877.

Missing volume was represented explicitly and derived continuous features used fold-training median imputation. No forward fill, backward fill, or future volume was used.

## Experimental comparison

Both models used the same shallow `HistGradientBoostingClassifier` parameters, monthly folds, training weights, path target, and 95th-percentile onset rule.

- `price_hgb`: the frozen 40 causal price-and-clock features;
- `price_historical_volume_hgb`: those same 40 features plus 11 causal volume features.

No raw volume level was included. The volume features described relative activity:

- current log volume versus the previous 1, 3, 6, and 12 exact bars;
- current volume versus the earlier exact segment mean;
- recent-three versus disjoint older-twelve activity;
- prior volume variability and availability;
- relative-volume interactions with bar range and signed body.

The signed-body interaction is a price–volume interaction, not signed or aggressor-side volume.

## Probability evidence

Positive loss improvements mean the price-plus-volume model was better. Values are absolute loss reductions.

| Horizon | Price log loss | Price+volume | Improvement | Price Brier | Price+volume | Improvement | Both-better months | Both-better stock deletions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.994442 | 0.993252 | +0.001189 | 0.609942 | 0.609451 | +0.000491 | 6/6 | 22/22 |
| 12 | 0.846946 | 0.845725 | +0.001221 | 0.545931 | 0.545643 | +0.000288 | 4/6 | 22/22 |
| 24 | 0.769059 | 0.768209 | +0.000850 | 0.519234 | 0.519095 | +0.000139 | 4/6 | 22/22 |

Relative log-loss improvement was only about 0.11% to 0.14%; Brier improvement was about 0.03% to 0.08%. This is real but small.

Accuracy rose by 0.09, 0.16, and 0.16 percentage points at 6, 12, and 24 bars. Macro AUC rose by 0.0021, 0.0037, and 0.0063.

## What volume predicted

Most of the loss improvement came from class 0: distinguishing no-hit/ambiguous paths from paths that reached a barrier.

| Horizon | No-entry binary log-loss improvement | No-entry AUC lift | Long binary log-loss improvement | Short binary log-loss improvement |
|---:|---:|---:|---:|---:|
| 6 | +0.001180 | +0.0040 | +0.000277 | +0.000073 |
| 12 | +0.001117 | +0.0065 | +0.000185 | +0.000151 |
| 24 | +0.000757 | +0.0125 | +0.000190 | +0.000007 |

After removing no-entry paths and renormalizing long versus short probabilities, directional AUC was only 0.5053, 0.5014, and 0.5035 at 6, 12, and 24 bars. Directional log-loss improvement was just 0.000010, 0.000098, and 0.000076.

Historical volume therefore helped answer “will meaningful movement occur?” more than “which side will move first?”

## Entry-onset evidence

Candidate onsets used price plus historical volume. Each received one outcome-blind matched control selected from the price-only HGB surface in the same symbol, month, horizon, and nearby clock/history bucket. All 18,167 controls matched.

| Horizon | Side | Onsets | Candidate precision | Matched price precision | Precision lift | Rapid-correct lift | Dominance lift | Pre-adverse improvement |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | Long | 4,548 | 45.83% | 47.63% | -1.80 pp | -2.18 pp | +0.018 | -0.102 |
| 6 | Short | 2,431 | 47.52% | 48.53% | -1.01 pp | -1.42 pp | +0.004 | +0.082 |
| 12 | Long | 3,214 | 48.21% | 47.89% | +0.32 pp | +0.37 pp | -0.534 | -0.046 |
| 12 | Short | 2,509 | 50.84% | 50.65% | +0.19 pp | -1.34 pp | +0.111 | +0.036 |
| 24 | Long | 2,227 | 49.06% | 48.06% | +1.01 pp | +4.06 pp | +0.165 | +0.007 |
| 24 | Short | 3,238 | 51.49% | 53.47% | -1.98 pp | -0.80 pp | -0.304 | -0.006 |

None passed the frozen all-required entry gates.

At six bars, volume-selected alerts were generally worse than price-only controls. At twelve bars, differences were negligible and path dominance was unstable. At 24 bars, long precision improved only 1.01 percentage points with a bootstrap interval spanning approximately -3.12 to +5.14 points.

High or unusual historical volume should therefore not be treated as directional confirmation.

## Narrow timing clue

The predeclared rapid-correct metric produced one coherent follow-up question.

For 24-bar long alerts:

- rapid correct confirmation within three bars was 33.56% with volume versus 29.50% for matched price controls;
- lift was +4.06 percentage points;
- the 95% moving-block interval was approximately +0.63 to +7.45 points;
- rapid lift was positive in 6/6 months, all 22 leave-one-stock-out deletions, and all three available clock quartiles.

Overall long-first precision did not improve reliably, so this is not a retained entry signal. It suggests that historical volume may help distinguish faster long confirmation after a price model has already supplied directional context.

Post-outcome inspection found that rapidly correct 24-bar long onsets had median current log volume 0.102 above their prior-six-bar geometric activity level, versus -0.020 for other onsets—roughly 10.8% above versus 2.0% below after exponentiation. This feature-level contrast was not frozen beforehand and is only a candidate for future testing.

## Why a volume filter is insufficient

The HGB volume group increased the emitted-side probability margin for most alerts, but this fitted confidence did not separate successes cleanly from failures. Correct and failed onsets had very similar volume sensitivities and relative-volume medians in most horizon/side cells.

Examples of median current volume versus the previous six-bar geometric activity level:

- 6-bar long onsets: approximately +46%;
- 6-bar short onsets: approximately +34%;
- 12-bar long onsets: approximately +18%;
- 12-bar short onsets: approximately -25%;
- 24-bar long onsets: approximately +2%;
- 24-bar short onsets: approximately -20%.

The inconsistent signs across horizons show why a simple “high volume confirms” rule would be misleading.

## Independent audit

Independent audit passed 13/13 checks.

The auditor did not import the V2 runner. It rebuilt volume coverage and all 11 features, refitted all 42 HGB folds using mathematically equivalent nested weights evaluated in a different floating-point order, and reproduced:

- all probabilities within `6.42e-15`;
- all prior-month thresholds within `1.67e-16`;
- all 18,167 hysteresis onset IDs exactly;
- all 18,167 matched price-control IDs exactly;
- all 489,991 validation paths exactly;
- pooled probability metrics within `1.11e-16`.

This model family did not reproduce the numerical fragility of the capped logistic model in V1.

Focused validation: 10/10 tests passed; syntax and Ruff checks passed. The full workspace suite produced 333 passes and one failure. That failure is the deliberately visible, previously documented bar-only V1 test requiring an `independent_audit.json`; V1 did not receive one because its capped logistic model failed numerical-stability audit. It is unrelated to the audited historical-volume V2.

## Recommended next experiment

Keep historical volume, but use it as a separate activation/timing component rather than a directional confirmation rule.

A clean V3 should freeze two outputs before genuinely new outcomes:

1. price-only directional context: long versus short;
2. price-plus-`historical_volume` activation timing: probability of any barrier hit or correct confirmation within three bars.

The narrow 24-bar long timing hypothesis can be declared separately, using the current-versus-prior-six activity feature without tuning its threshold on these same outcomes. It must be recorded prospectively before outcome attachment.

## Files and artifacts

- Contract: `work/contracts/20260712-historical-volume-entry-onset-ablation-v2.json`
- Pre-score manifest: `work/contracts/20260712-historical-volume-entry-onset-ablation-v2-pre-score.json`
- Runner: `work/run_historical_volume_entry_onset_ablation_v2.py`
- Auditor: `work/audit_historical_volume_entry_onset_ablation_v2.py`
- Tests: `work/tests/test_historical_volume_entry_onset_ablation_v2.py`
- Artifacts: `/private/tmp/stocker_historical_volume_entry_onset_ablation_v2_20260712`

The `/private/tmp` artifacts are ephemeral and should be archived separately before a reboot if exact ledgers are needed without recomputation.
