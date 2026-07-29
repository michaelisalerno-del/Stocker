# Regime-loop orientation calibration algorithms V1

Date: 2026-07-11

Decision: `orientation_calibration_algorithms_rejected_or_unconfirmed`

Scientific status: post-inspection 2024 causal-OOF algorithm-development test. This is not prospective validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no direction, signed return, P&L, trading rule, cost/exit model, broker, order, position, deployment, or strategy-promotion path was used
- direct volume label: `historical_volume_not_used`
- “good” and “high” refer only to frozen absolute-movement or future-range classes, not trading performance

## Question

Can fixed causal calibration or partially pooled loop-regime algorithms retain the information in the raw regime-to-loop movement link while removing its two reliability failures:

1. supported probability-bin miscalibration; and
2. reversals in particular loop × current-regime orientations?

The target in each cell was the joint event that a compatible loop occurs and its subsequent absolute return or future range exceeds a frozen 2024 P75 or P90 threshold at 6, 12, or 24 five-minute bars.

## What was and was not individualized

This experiment did **not** select a separate algorithm family for each loop or regime.

Four common algorithm families were fitted separately for every target, horizon, tier, and causal month fold. The two orientation models included loop × current-regime coefficients, so their predictions could adapt across 44 orientations; the clock model also adapted across 132 loop × regime × existing-clock cells. Those coefficients were partially pooled through one fixed ridge model and one fixed set of feature scales.

Therefore:

- individual loop/regime performance was measured;
- loop/regime-specific adjustments were learned;
- no “best algorithm for this particular loop/regime” selection was performed.

A causally nested per-orientation mixture-of-experts experiment would be a separate next test.

## Frozen algorithms

1. `weighted_isotonic`: monotone weighted isotonic recalibration of the raw full-link probability;
2. `beta_global`: fixed ridge logistic calibration using `log(p)` and `log(1-p)`;
3. `orientation_residual`: global baseline and raw-link residual plus partially pooled loop × current-state intercepts and residual slopes;
4. `orientation_clock_residual`: the orientation model plus partially pooled existing clock-quartile intercepts and residual slopes.

There was no hyperparameter search. All model definitions, feature scales, support rules, tests, and pass gates were frozen before scoring.

## Causal development split

The frozen source was the independently audited September-December 2024 linkage ledger.

- November validation used September-October training only.
- December validation used September-November training only.
- validation surface: 51,235 compatible rows, 9,229 anchors, 41 sessions, 22 stocks, all 20 cycles, and all 8 states;
- 12 joint-event cells: two movement targets × three horizons × two tiers;
- 96 models: four algorithms × twelve cells × two validation folds.

The source period had already been inspected. These results are development evidence, even though the monthly fits themselves are causal.

## Pooled result

The state/context independence product was the baseline. The previously retained raw full link was the stronger reference.

| Model | Log loss | Improvement vs baseline | LL difference vs raw | Brier | Brier difference vs raw | Formal pass |
|---|---:|---:|---:|---:|---:|---|
| State/context baseline | 0.129698 | 0.0000% | +0.007588 | 0.036844 | +0.002034 | Reference |
| Raw full link | 0.122110 | 5.8505% | 0.000000 | 0.034810 | 0.000000 | Diagnostic reference only |
| Weighted isotonic | 0.122809 | 5.3117% | +0.000699 | 0.034852 | +0.000042 | No |
| Global beta calibration | 0.122645 | 5.4381% | +0.000535 | 0.035008 | +0.000198 | No |
| Orientation residual | 0.121831 | 6.0655% | -0.000279 | 0.034785 | -0.000025 | No |
| Orientation + clock residual | **0.121760** | **6.1202%** | **-0.000350** | **0.034764** | **-0.000046** | **No** |

All four algorithms improved log loss and Brier versus the baseline in every one of the 12 cells and under every leave-one-stock-out deletion. Every algorithm also passed the session-block bootstrap and familywise Holm tests against the baseline. This confirms that the joint regime-loop-movement link contains pooled information.

The orientation + clock model was the best average model and was slightly better than the raw link on both proper losses. That improvement was not reliable against the raw reference:

- log-loss 95% session-block interval: `[-0.001563, +0.000145]`;
- raw-comparison log-loss p-value: `0.0392`, Holm-adjusted `0.1568`;
- Brier 95% session-block interval: `[-0.000222, +0.000134]`;
- raw-comparison Brier p-value: `0.1957`, Holm-adjusted `0.7828`.

It beat the raw reference in November but was slightly worse in December on both log loss and Brier. The contract required both months to pass.

## Why the best algorithm failed

### Absolute calibration

The orientation + clock model reduced ECE in 11 of 12 cells, but only one cell met the absolute 2% maximum-supported-bin-error ceiling. The worst cell was 24-bar future-range P90:

- candidate maximum supported-bin error: `0.104239`;
- raw-reference maximum supported-bin error: `0.044273`.

Across the 12 cells, candidate maximum-bin errors ranged from `0.019860` to `0.104239`. Eleven cells failed the frozen absolute calibration gate.

### Loop × current-regime reversals

Twenty-six orientations met the support rule. The best model still had:

- 5/26 log-loss reversals versus baseline;
- 7/26 Brier reversals versus baseline;
- 5/12 top-three-recall degradations versus the raw link.

The worst log-loss orientations included cycle 01 in state 1, cycle 20 in state 2, and cycle 02 in state 1. This is exactly the heterogeneity the model was intended to remove, so zero reversals was a necessary gate.

## Strong regime × loop × time diagnostic

No time slice formally qualified because the global orientation + clock algorithm failed. One post-inspection diagnostic nevertheless became sharper.

Cycle 13, `5→7→5`, entered from state 5 during existing clock quartile 1, corresponding to approximately 105-195 minutes after the open:

- 510 compatible validation rows across 22 stocks;
- 430 joint P75 positives summed across six target/horizon cells;
- 17.4905% pooled P75 log-loss improvement versus baseline;
- all six P75 cells improved;
- both November and December improved;
- maximum supported-bin error `0.010737`, below the slice ceiling of 3%;
- Holm-adjusted p-value `0.0039`.

This repeats the earlier cycle-13/state-5/time attraction and repairs its local calibration in this model. It still cannot be promoted to “good” or “high” because:

- the global algorithm failed;
- the time slice was already visible in prior development results;
- the rows and six outcomes are not independent trades;
- no genuinely unseen post-freeze sessions were read.

Cycle 09, `3→6→3`, entered from state 3 in clock quartile 2 was a secondary diagnostic: 15.2260% P75 log-loss improvement, maximum bin error `0.021191`, and Holm-adjusted p-value `0.0408`. It failed the same global prerequisite.

## Decision and next algorithm idea

None of the four algorithms passed the complete frozen gate. No algorithm, time slice, or named loop was selected or promoted. The raw full link remains a diagnostic research signal only.

The next genuinely distinct algorithm test is a causally nested mixture of experts:

1. define a small frozen expert set, such as raw link, fixed shrinkage, global calibration, and orientation residual;
2. for each supported loop × current-regime orientation, choose an expert using strictly earlier-month performance only;
3. use hierarchical backoff to the global expert when an orientation lacks training support;
4. lock the assignment before predicting the next month or session;
5. assess proper loss, absolute calibration, orientation reversals, stock deletion, and temporal stability under the same fail-closed rules.

That test must not select experts by looking at the month being scored. Reusing the opened September-December results can provide only exploratory development evidence. The credible final test is a frozen assignment evaluated on genuinely unseen post-freeze sessions.

## Integrity and reproducibility

- contract SHA-256: `900f14c8c43456a28e3532be1cc499fe61d9b4b26b0f0904d0afafa1c7ad525d`
- runner SHA-256: `ecc4cc1d3e2fb574bb0ea0d792bbc5bae5083ec9264b9857a46b9d6ac4f5919a`
- independent auditor SHA-256: `808212be19341ca8dd6dddd8bfd6ef13a19c1c2663e8240f014845a2bcfd03d1`
- independent audit-result SHA-256: `dac303825079fc3fc8a77f188f2cb9851296019013a9dd29e6c5365fc6ba8de9`
- independent audit: 19/19 checks passed;
- all 96 model fits replayed with parameter and prediction error `0.0`;
- losses, calibration, bootstrap, Holm, temporal, stock, orientation, ranking, gates, time slices, and final decision replayed within `1.12e-16`;
- focused runner/auditor suite: 13 tests passed;
- full workspace research suite: 253 tests passed;
- `git diff --check`: passed.

Artifact root: `/private/tmp/stocker_regime_loop_orientation_calibration_algorithms_v1_20260711`

The artifact root is ephemeral and should be archived before reboot if exact replay without recomputation is required.
