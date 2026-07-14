# Individual algorithms by regime and loop V1

Date: 2026-07-11

Decision: `individual_expert_selectors_rejected_or_unconfirmed`

Scientific status: post-inspection 2024 causal-OOF development test. This is not prospective validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no direction, signed return, P&L, trading rule, cost/exit model, broker, order, position, deployment, or strategy-promotion path was used
- direct volume label: `historical_volume_not_used`
- “good” and “high” mean frozen absolute-movement or future-range classes only, not trading performance

## Direct answer

Yes. This experiment tested whether different regimes, loops, loop-regime orientations, and supported loop-regime-time cells should use different probability algorithms.

The answer is narrower than hoped:

- loops and regimes **did select different algorithms** from earlier-month evidence;
- a guarded loop × regime selector slightly improved pooled log loss and removed one log-loss and one Brier orientation reversal;
- those selections did **not** improve both proper losses, did not remain better in both validation months, did not fix calibration, and did not pass multiplicity;
- therefore no individual algorithm assignment is reliable enough to retain or promote yet.

## Frozen expert set

Every expert was already present in the independently audited parent prediction ledger. No expert was refitted in this experiment.

1. `baseline`: history occurrence × context movement;
2. `minimal_time_topology`: limited regime/time occurrence × route topology movement;
3. `raw_full_link`: full factor occurrence × hierarchical loop movement quality;
4. `partial_full_link`: fixed half-shrinkage toward retained baselines;
5. `calibrated_raw_product`: causal expanding-month calibration;
6. `dependency_stack`: causal learned interaction stack.

The diagnostic occurrence-only and topology-only variants were deliberately excluded from selection because they were not selectable candidates in their parent experiment.

## Selectors tested

| Selector | Assignment unit | Guard/backoff |
|---|---|---|
| `global_best` | One expert for all rows | Diagnostic control |
| `regime_best` | Current state | Support fallback |
| `loop_best` | Fixed cycle | Support fallback |
| `loop_regime_best` | Cycle × current state | Support fallback |
| `guarded_loop_regime_best` | Cycle × current state | Must beat global expert by at least 0.0005 earlier-month log loss |
| `hierarchical_clock_best` | Loop → loop-state → loop-state-clock | Fixed support and 0.0005 child-improvement margin at each level |

Each unit chose one expert using the equal mean of all 12 inverse-compatible-weighted joint-event log losses. The chosen expert was then shared across both movement targets, all three horizons, and both P75/P90 tiers. There was no cell-by-cell algorithm shopping.

## Causal selection schedule

- November assignments used September-October outcomes and causal OOF predictions only.
- December assignments used September-November only.
- Validation contained 51,235 compatible rows, 9,229 anchors, 41 sessions, 22 stocks, all 20 cycles, and all 8 states.
- Validation outcomes never affected their own expert assignment.
- 2025, backward-2023, partial 2026, and prospective-shadow outcomes were not read.

The chronology is causal, but all 2024 months are already-opened development evidence.

## Pooled result

The earlier-month global winner was `raw_full_link` for both validation folds, so `global_best` exactly equals the raw reference.

| Selector | Log loss | Improvement vs raw | Brier difference vs raw | Individualized rows | Assignment stability | Formal pass |
|---|---:|---:|---:|---:|---:|---|
| Global raw reference | 0.122110 | 0.0000% | 0.000000 | 0.00% | 100.00% | Reference |
| Regime | 0.122077 | +0.0272% | +0.000093 | 13.10% | 100.00% | No |
| Loop | 0.122211 | -0.0825% | +0.000001 | 53.99% | 50.00% | No |
| Loop × regime | 0.122106 | +0.0034% | +0.000020 | 51.79% | 75.00% | No |
| Guarded loop × regime | **0.122026** | **+0.0687%** | **+0.000009** | 14.35% | 86.36% | **No** |
| Hierarchical loop × regime × clock | 0.122257 | -0.1203% | +0.000053 | 29.87% | 80.30% | No |

The guarded orientation selector was the best specialization method on log loss, but its improvement was tiny and its Brier score was worse. Its raw-comparison results were not statistically secure:

- log-loss 95% session-block interval: `[-0.001619, +0.000239]`;
- log-loss p-value: `0.2598`, Holm-adjusted `1.0`;
- Brier 95% session-block interval: `[-0.000030, +0.000047]`;
- Brier p-value: `0.5876`, Holm-adjusted `1.0`.

Every individualized selector was better than raw on at least one proper-loss or temporal view and worse on another. None beat raw on both log loss and Brier.

## Temporal failure

All three main loop selectors improved in November and became worse in December.

| Selector | November LL difference vs raw | December LL difference vs raw |
|---|---:|---:|
| Loop | -0.000063 | +0.000244 |
| Loop × regime | -0.000421 | +0.000361 |
| Guarded loop × regime | **-0.000469** | **+0.000254** |

This is the main reason earlier-month “best expert” selection cannot yet be trusted. The expert ranking moved faster than the hard assignment rule could follow.

## Which regimes chose different algorithms?

The regime map was perfectly stable between the two folds:

- state 0 selected `calibrated_raw_product`;
- state 5 selected `dependency_stack`;
- states 1, 2, 3, 4, 6, and 7 retained `raw_full_link`.

Stable assignment did not mean stable validation benefit:

| State | Selected expert | November LL gain vs raw | December LL gain vs raw |
|---:|---|---:|---:|
| 0 | Calibrated raw product | -0.71% | -5.27% |
| 5 | Dependency stack | +0.61% | -0.55% |

The state-0 earlier-month preference failed in both validation months. State 5 showed a small benefit in November and a small reversal in December. Consequently the regime selector slightly improved pooled log loss but worsened Brier and failed nine of twelve cell comparisons.

## Which loops chose different algorithms?

The loop-only selector used five experts, but only 10 of 20 loop assignments were unchanged between November and December. Examples of stable assignments were:

- cycles 03, 07, 10, 12, 13, 17, and 20: `raw_full_link`;
- cycles 06 and 18: `partial_full_link`;
- cycle 15: `minimal_time_topology`.

The remaining ten loops changed expert after one additional month of evidence. This 50% stability, combined with worse pooled loss, rejects fixed algorithm-per-loop selection in its present form.

## Loop × regime result

The unguarded selector used all six experts and changed the global assignment on 51.79% of validation rows. The guarded selector concentrated 85.65% of rows back into the raw expert and used five experts overall.

Among 26 supported validation orientations:

| Model | Log-loss reversals vs baseline | Brier reversals vs baseline |
|---|---:|---:|
| Raw/global expert | 5 | 4 |
| Unguarded loop × regime | **4** | **3** |
| Guarded loop × regime | **4** | **3** |
| Hierarchical clock | 7 | 5 |

Individual selection therefore removed one reversal on each proper loss, but it did not reach the required zero. The clock hierarchy added noise rather than resolving heterogeneity.

## Cycle 13 in state 5

Cycle 13, `5→7→5`, entered from state 5 selected `dependency_stack` in both folds. This is a real repeat of the previously observed cycle-13/state-5 attraction, but its validation effect changed sign:

| Validation month | Compatible rows | Selected expert | LL gain vs raw |
|---|---:|---|---:|
| November | 774 | Dependency stack | +1.95% |
| December | 675 | Dependency stack | -1.21% |
| Combined | 1,449 | Dependency stack | +0.71% |

Combined log loss improved by `0.001625`, but Brier worsened by `0.000318`. This orientation is not certified good/high: the probability benefit was not temporally stable and did not improve both proper losses.

Cycle 11 in state 5 was a useful secondary diagnostic. It selected the half-shrunk expert for November and dependency stack for December, improving log loss versus raw in both months. Because the assignment itself changed and the result was inspected after the global rejection, it remains a hypothesis rather than a retained rule.

## Post-score selection diagnostic

After the frozen decision, each earlier-month assignment was compared with the hindsight-best expert on its validation unit. This is explanatory only and was not a pass gate.

| Selector | November unit hit rate | December unit hit rate |
|---|---:|---:|
| Regime | 25.0% | 37.5% |
| Loop | 30.0% | 35.0% |
| Loop × regime | 22.7% | 36.4% |
| Guarded loop × regime | **34.1%** | **45.5%** |
| Hierarchical clock | 29.5% | 36.4% |

The guarded selector was best, but it still chose a non-hindsight-best expert for most validation units. Because the six experts are strongly correlated, these percentages should not be interpreted against a simple one-in-six random baseline. They show that earlier-month expert rank was not sufficiently persistent.

## Calibration and other gates

All five candidate selectors failed calibration in all 12 cells. Their worst supported-bin errors ranged from 8.86% to 9.30%, versus 7.94% for raw on the same November-December surface; the contract ceiling was 2%.

The guarded selector also had:

- 6/12 cells worse than raw on at least one proper loss;
- 3/12 top-three-recall degradations;
- at least one worse leave-one-stock-out deletion;
- no raw-comparison bootstrap or Holm endpoint pass.

No selector passed the complete frozen gate. No algorithm assignment, named loop, state, or time cell was promoted.

## Interpretation and next boundary

The experiment supports three limited conclusions:

1. algorithm preference is heterogeneous across regimes and loops;
2. a conservative loop-regime guard can reduce local reversals slightly;
3. hard selection from recent historical winner tables is too unstable and does not repair probability calibration.

The next algorithm should not be another hard winner lookup or another margin tuned on these same months. A more defensible research candidate is a **soft causal gating model** that predicts expert weights from pre-entry regime, loop identity, clock, and the existing frozen context, with hierarchical shrinkage and an explicit calibration head. Its weights must be trained on earlier months, frozen before each scored month, and evaluated against the raw expert on proper loss and calibration.

For scientific confidence, the cleanest next action is to register the guarded selector and the cycle-13/state-5 dependency preference as frozen post-inspection hypotheses, then judge them only on genuinely unseen post-freeze sessions. September-December should not be used to tune another support threshold, margin, expert list, or hierarchy.

## Integrity and reproducibility

- contract SHA-256: `d34dade298518eb37a1e710c838d59d7e53875c9dc33c0054934b53902966267`
- runner SHA-256: `7f95e30126cc9c207a589dd00ad64c1ca02cc8d52be5ce2ffff2d73e87a44ad5`
- independent auditor SHA-256: `ba40c4aa379482ddadff66a0e6d31a90f5e9f8970d957cddb7074ea400dd1507`
- independent audit-result SHA-256: `8b59e89cfddf7bc560607218d6eeb77282f070d64bae47a01f119fa277cc94eb`
- independent audit: 19/19 checks passed;
- all causal assignments and all selector probabilities replayed with error `0.0`;
- objectives, losses, calibration, bootstrap, Holm, temporal, stock, orientation, ranking, assignment summaries, gates, and decision replayed within `1.12e-16`;
- focused runner/auditor suite: 13 tests passed;
- full workspace research suite: 266 tests passed;
- `git diff --check`: passed.

Artifact root: `/private/tmp/stocker_regime_loop_individual_expert_selection_v1_20260711`

The artifact root is ephemeral and should be archived before reboot if exact replay without recomputation is required.
