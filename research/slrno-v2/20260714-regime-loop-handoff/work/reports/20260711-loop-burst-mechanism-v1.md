# Why loops persist for limited time: loop-burst mechanism V1

Date: 2026-07-11

Decision: `burst_continuation_model_rejected_or_unconfirmed`

Scientific status: post-inspection 2024 causal-OOF development test. This is not prospective validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no direction, signed return, price-consequence target, P&L, trading rule, broker, order, position, deployment, or strategy-promotion path was used
- direct volume label: `historical_volume_not_used`
- no loop was promoted to good/high movement quality

## Question

Are two-state loops temporary recurrent bursts whose continuation and termination depend on:

- whether the same orientation just completed;
- how many times it has already repeated;
- the completed dwell times of the prior loop;
- scheduled session time remaining;
- the already-frozen causal history/context probability?

This tests the idea that a regime label describes the market's location, while loop phase describes its current motion inside that regime.

## Frozen scope

The test used all 13 fixed two-transition cycles and both rotations, giving 26 loop-current-state orientations. The source was the independently audited July-December 2024 causal OOF occurrence ledger.

The new causal continuation features were reconstructed at each run entry:

- consecutive prior same-orientation repeat count;
- duration of the completed prior current-state run;
- duration of the completed intervening-state run;
- their completed total duration;
- scheduled five-minute bars remaining in the regular session.

Current-run final duration, future dwell times, future destinations, session-end realization, and validation outcomes were forbidden as predictors. Outcome dwell times were read only for the separate detector-chatter diagnostic.

Expanding-month fits were:

- October trained on July-September;
- November trained on July-October;
- December trained on July-November.

The primary continuation evaluation contained 8,540 rows and 2,062 positive repeat events across 64 sessions. The full feature reconstruction contained 201,376 compatible two-state rows, including 18,406 continuation rows and 4,045 continuation positives across all six OOF months.

## Main mechanism result

The burst mechanism is strong descriptively:

| Cohort | Weighted probability of same loop next |
|---|---:|
| Initiation: no immediately completed same-orientation loop | 9.22% |
| Continuation: at least one immediately completed same-orientation loop | 30.09% |

The continuation/initiation rate ratio was `3.2633`, an absolute difference of `20.87` percentage points. The five-session bootstrap interval for the daily risk difference was `[16.64, 21.86]` percentage points.

All 20 orientations meeting the frozen recurrence-support rule had a rate ratio above one. The other six orientations also had raw ratios above one, ranging from 1.50 to 5.05, but each had fewer than the required 20 recurrent positives.

The contract required at least 24 fully supported orientations. Only 20 were support-qualified, so the formal H1 orientation-coverage gate failed. This was a support failure, not an observed sign reversal.

Among supported orientations, recurrence ratios ranged from `1.0720` to `5.5810`. Cycle 13 in state 5 remained the strongest supported unit:

- continuation rate: 45.99%;
- initiation rate: 8.24%;
- rate ratio: 5.581.

## Why the burst ends

Continuation probability rose and then flattened as a burst repeated:

| Prior same-orientation repeats | Rows | Next-repeat probability | Mean scheduled bars remaining |
|---|---:|---:|---:|
| 1 | 6,478 | 27.48% | 39.69 |
| 2 | 1,450 | 36.81% | 32.30 |
| 3 | 431 | 35.96% | 25.78 |
| 4+ | 181 | 33.11% | 17.21 |

This supports an activation-and-decay interpretation:

1. the first completed loop makes another recurrence much more likely;
2. recurrence strengthens into the second completion;
3. it then plateaus or declines as scheduled time is consumed and the burst eventually escapes into another state.

The causal global phase coefficients agreed across all three folds. Standardized repeat count and scheduled bars remaining were positive every time. Scheduled time remaining was the largest and most stable global phase coefficient. Completed-dwell effects were smaller and mixed.

## Session boundary is important but not sufficient

For cycle 13, `5→7→5`, entered from state 5 during October-December:

| Clock quartile | Rows | Raw occurrence rate | Session-boundary fraction | Rate with two destinations available |
|---:|---:|---:|---:|---:|
| 0 | 912 | 12.61% | 0.44% | 12.69% |
| 1 | 756 | **20.63%** | 5.16% | **21.88%** |
| 2 | 631 | 12.84% | 6.97% | 14.06% |
| 3 | 624 | 5.61% | **44.07%** | 12.50% |

Late-session truncation therefore explains a large part of the raw decline. It does not explain all of it: after restricting to rows where two future destinations existed, the mid-session rate still exceeded the late rate by `9.38` percentage points.

The correct interpretation is that session end is both:

- a mechanical boundary in the loop label; and
- one component of a real phase change that remains after boundary-eligible restriction.

## Detector-chatter falsification

The test did not eliminate detector boundary noise, but cycle 13 was not merely a collection of one-bar flips.

Among 387 realized October-December cycle-13/state-5 loops:

- 44.19% had all three state runs lasting at least two bars;
- 55.81% contained at least one one-bar leg;
- median full `5→7→5` duration was 22 bars;
- interquartile duration was 13-31 bars.

Both frozen chatter checks passed. Short state legs contribute materially, so posterior-hysteresis or minimum-dwell sensitivity remains necessary. They cannot explain the complete recurrence effect because nearly half the loops survived the two-bar-per-leg criterion and typical total duration was much longer.

## Continuation models

The fixed reference was `qfull9`, the earlier causal history/context probability. A fold-local intercept-only correction formed `qoffset_calibration`. Two new models added the causal burst-phase variables:

- `qburst_global`: one shared phase relationship across all loops;
- `qburst_orientation`: shared phase plus loop-orientation intercepts and interactions.

| Model | Log loss | Brier | ECE | Maximum supported-bin error |
|---|---:|---:|---:|---:|
| History only | 0.578766 | 0.197481 | 0.053059 | 0.115488 |
| Full causal context | 0.545918 | 0.182955 | 0.008276 | 0.015765 |
| Offset calibration | 0.545953 | 0.182983 | 0.008258 | 0.014896 |
| Global burst phase | **0.543538** | **0.182010** | **0.007138** | **0.015684** |
| Orientation burst phase, primary | 0.544083 | 0.182399 | 0.008241 | 0.037203 |

The predeclared primary orientation model improved log loss by `0.3425%` versus offset calibration, below the required `0.5%`, while improving Brier by `0.000584`.

Its session intervals crossed zero:

- log loss versus offset: `[-0.003569, +0.002456]`;
- Brier versus offset: `[-0.001211, +0.001325]`;
- every Holm-adjusted p-value was `0.912`.

It improved in October and December but became worse in November. It improved log loss in 14 of 20 supported orientations rather than the required 20, and its worst orientation harm was `0.005413`, just above the `0.005` ceiling. One leave-MSTR-out Brier comparison also reversed. Maximum supported-bin error was 3.72%, above the 3% ceiling.

The durable-prior slice was better: on 3,502 rows whose completed prior loop had both legs lasting at least two bars, the primary model improved log loss from `0.548730` to `0.545898` and Brier from `0.184500` to `0.183678`. This supports phase information beyond pure one-bar chatter, but it cannot override the global failures.

## Simpler global phase diagnostic

The simpler global model was better than the orientation-interaction model:

- 0.4424% pooled log-loss improvement versus offset calibration;
- lower log loss and Brier in October, November, and December;
- pooled maximum-bin error 1.57%;
- no adverse leave-one-stock-out deletion in a post-score diagnostic.

Its session bootstrap intervals still crossed zero, it improved only 13 of 20 supported orientations, and it had no predeclared retention gate separate from the primary orientation model. It is therefore an unqualified diagnostic, not a retained forecaster. The evidence nevertheless suggests that burst phase is shared more consistently across loops than orientation-specific phase coefficients; the latter overfit this sample.

## Decision

The formal mechanism gate failed because only 20, not 24, orientations met the required recurrence support. The primary continuation forecaster also failed its magnitude, bootstrap, multiplicity, month, stock, orientation, and calibration gates.

No new forecaster is retained. No loop is promoted to good/high.

The scientifically defensible conclusion is narrower:

- two-state loops behave like temporary recurrent bursts in every observed orientation;
- the effect is large and session-level stable in pooled opened-2024 data;
- scheduled time remaining explains much of loop termination, but a mid-session phase effect remains;
- completed repeat count and dwell information contain incremental causal information;
- a shared phase head is more promising than loop-specific phase interactions;
- dictionary-wide and prospective reliability remain unproven.

The next clean test is to freeze the simple global phase head—without tuning it on these results—and score genuinely unseen post-freeze sessions. A separate detector-falsification test should reconstruct states with posterior hysteresis or a predeclared minimum-dwell rule and ask whether recurrence ratios survive.

## Integrity and reproducibility

- contract SHA-256: `29a6e219d3886f7617bc417797c7cc8b7f66f02347d93e866276f40db3c90360`
- runner SHA-256: `ce0899590698b16281e9f4cb4f732940ea6d82097594b6143c7e5531ea8b9c46`
- independent auditor SHA-256: `f702c637bd7e677cedb5476f3d8d73afcaf54fcd693ae2adea27045186ac5769`
- independent audit-result SHA-256: `46cc20a358c6e8db4fe0e26fe90229c37b057fb44a1b1c0a3073cc25b2905610`
- independent audit: 22/22 checks passed;
- all 201,376 sequence-feature rows replayed exactly;
- all nine parameter fits and all 8,540 validation predictions replayed with error `0.0`;
- metrics, calibration, bootstrap, Holm, temporal, stock, orientation, durable, recurrence, boundary, repeat-count, gates, and decision replayed within `2.28e-13`;
- focused runner/auditor suite: 13 tests passed;
- full workspace research suite: 279 tests passed;
- `git diff --check`: passed.

Artifact root: `/private/tmp/stocker_loop_burst_mechanism_v1_20260711`

The artifact root is ephemeral and should be archived before reboot if exact replay without recomputation is required.
