# Regime usefulness ablation V1

Date: 2026-07-11

Decision: `state_and_history_retained_as_incremental_movement_features`

Later additions: `departure_loop_and_burst_layers_not_retained`

Scientific status: 2024 internal causal-forward development test. This is not prospective validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- direct provider-volume label: `historical_volume_not_used`
- no direction, signed return, P&L, cost, spread, slippage, position, broker, order, strategy, deployment, or economic-edge target was used
- no loop was promoted to good/high movement quality

## Question

Does the detected eight-state regime have practical predictive value after ordinary causal price/context controls, and do state history, departure risk, loop probabilities, or burst phase add further value?

The test deliberately treated a regime as a conditioning coordinate rather than a buy/sell signal. The only outcomes were subsequent absolute return and future high-low range at 6, 12, and 24 five-minute bars.

## Frozen test

The outcome models used six expanding 2024 folds:

- July trained through June;
- August trained through July;
- September trained through August;
- October trained through September;
- November trained through October;
- December trained through November.

The validation cohort contained 34,169 exact run-entry anchors, 22 stocks, and 128 sessions. Every outcome began after the completed anchor bar and had exact regular-session support through 24 bars.

The nested feature layers were:

1. nine causal price, B0, stress, and entry-clock controls without the eight-state regime;
2. current eight-state regime;
3. the last-three-state history token;
4. a fold-local probability that the current run would leave within three bars;
5. twenty fold-local fixed-loop path probabilities;
6. past-only two-state burst phase: prior repeats, completed prior-pair duration, durable-prior flags, and loop-by-repeat interactions.

For every validation month, the departure and destination-history models were refitted only on earlier state runs. Loop probabilities were then rebuilt by multiplying those earlier-month destination probabilities through each compatible frozen path. The outcome model was the frozen-form Ridge model with `alpha=10`, identical training-fold scaling, and no stock identity.

The eight-state representation itself remains the detector fitted on all of 2024. Therefore the outcome and structural auxiliary fits are forward by month, but this is not a fully untouched test of the state-discovery parameters.

## Result

Positive values below mean lower out-of-fold MSE. Each row compares a layer with the immediately preceding layer.

| Increment | Absolute-return MSE | Future-range MSE | Frozen robust gate |
| --- | ---: | ---: | --- |
| Current state over context | +0.4813% | +2.0849% | **Pass** |
| Last-three-state history over current state | +0.8106% | +3.9424% | **Pass** |
| Three-bar departure proxy over history | +0.0049% | +0.0504% | Fail |
| Loop probabilities over departure/history | -0.0069% | -0.0504% | Fail |
| Burst phase over loops | +0.0472% | +0.1875% | Fail |

The current state passed every predeclared gate:

- MSE improved at all six target/horizon cells;
- improvement remained positive under every leave-one-stock-out deletion at every cell;
- absolute-return months improved in five of six cases and range improved in six of six;
- pooled MAE improved for both target families;
- five-session normalized-MSE intervals were entirely beneficial for absolute return `[-0.8505%, -0.1661%]` and future range `[-3.3011%, -1.4253%]`.

State-only MSE improvements by horizon were:

| Target | 6 bars | 12 bars | 24 bars |
| --- | ---: | ---: | ---: |
| Absolute return | +0.3523% | +0.4762% | +0.5282% |
| Future range | +2.0345% | +2.0891% | +2.0980% |

State history also passed every gate. It improved absolute-return MSE in five of six months and range in all six. Its normalized-MSE intervals were `[-1.5494%, -0.3228%]` for absolute return and `[-5.7179%, -2.9536%]` for range.

History improvements over current state were:

| Target | 6 bars | 12 bars | 24 bars |
| --- | ---: | ---: | ---: |
| Absolute return | +0.6443% | +0.5923% | +0.9976% |
| Future range | +3.8742% | +3.9036% | +3.9847% |

The combined state-history layer improved MSE over context by 1.2880% for absolute return and 5.9451% for future range.

## What did not add reliable information

The departure proxy had a tiny favorable pooled mean, but absolute-return MSE reversed at 6 and 24 bars, only two of six absolute-return months improved, stock-deletion failures occurred, and both bootstrap intervals crossed zero. This does not invalidate the separately retained departure-timing model. It says run-entry exit risk did not add reliable movement magnitude after raw state history.

Fold-local loop probabilities became slightly worse after state history and departure were already present. This is an important distinction from the earlier frozen price-consequence result:

- the earlier test showed that loop probabilities beat state/context and usefully compressed raw history across 2025 and backward-2023;
- this test imposed the stronger requirement that loop probabilities add value on top of the complete raw history token;
- because loop probabilities are largely a structured compression of that history, little genuinely new information remained.

The two findings are compatible. Loop scores remain a useful compact representation, but stacking them on top of their raw history source is not justified by this test.

Burst phase improved all six MSE cells and every stock deletion, but the gains were only 0.0472% for absolute-return MSE and 0.1875% for future-range MSE. Both session bootstrap intervals crossed zero. Burst phase remains a descriptive research lead, not a retained movement layer.

## Overall magnitude diagnostic

The full descriptive stack improved MSE over context by:

- 1.3327% for absolute return;
- 6.1215% for future range.

It passed the predeclared 1%/3% magnitude thresholds, every horizon, every stock deletion, and both session-bootstrap gates. This confirms that the regime system as a whole contains meaningful movement information.

It does not override the sequential ablation. Only current state and state history earned retention. Departure, loop, and burst additions cannot be kept merely because the final stack remained better than the weak context baseline.

## Scientific conclusion

The detected regime is useful for a specific purpose: it improves forecasts of how much the market is likely to move and how wide the subsequent range may be. Its recent path through regimes is even more useful than the current label alone.

It is not supported as a directional signal. It also does not, by itself, identify a high/good loop. Loop identity and conditional movement quality must remain separate probabilities.

The best current interpretation is:

- current regime describes the market's local movement environment;
- recent regime history describes how it arrived there and carries substantial extra range information;
- loop probabilities provide a compact route summary when raw history is not used;
- departure timing answers when a state may end, but adds little to run-entry movement magnitude after history;
- burst phase is plausible but presently too small and uncertain to retain.

## Reliability boundary and next test

This result is reliable under the frozen internal-2024 gates, not prospectively established. The detector's state definition was fitted on 2024, while 2025, backward-2023, and partial 2026 are already development or portability periods. None is a clean future validation set.

The existing prospective movement shadow must remain unchanged. A later, separately frozen future comparison should evaluate mutually exclusive representations:

1. context only;
2. context plus current state;
3. context plus current state history;
4. context plus loop-score compression instead of raw history.

That comparison is more informative than adding loop scores on top of raw history. Departure timing should stay a separate timing output. Burst phase should remain exploratory until more genuinely unseen support accumulates.

## Integrity and reproducibility

- contract SHA-256: `f5869446b675626d324d5d43a14cc104a6997cd1ef69b2690e4b1cb07a001644`
- pre-score runner SHA-256: `9e88ff700c663d0985f02082a8a9a13a1e5e4d5825048c11c9cd256f58ae7cf5`
- independent auditor SHA-256: `c8784fbf1eae42e719e8fb386795e334bed2a47d33605bb4e78f72448911e257`
- independent audit-result SHA-256: `ff123248b3f00830c7d8d00ab4a2ffb18deed6b0627ed177e2aad8ca62fb3ade`
- independent audit: 23/23 checks passed
- all 20 fold-local loop probabilities, departure probabilities, past-only burst features, 36 outcome predictions, metric tables, bootstraps, and decisions were reconstructed
- exact outcome and loop prediction reconstruction error: `0.0`
- full workspace research suite: 284 tests passed
- scoped `git diff --check`: passed

Artifact root: `/private/tmp/stocker_regime_utility_ablation_v1_20260711`

The artifact root is ephemeral and should be archived before reboot if exact replay without recomputation is required.
