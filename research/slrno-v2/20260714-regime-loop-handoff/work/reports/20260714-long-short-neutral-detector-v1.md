# Long / short / neutral detector V1

Date: 2026-07-14

Decision: **`all_three_state_detectors_rejected_or_descriptive_only`**

Scientific status: causal retrospective three-state development on already-opened 2024, 2025, and partial-2026 data. This is not validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no trading app, broker, paper/demo account, deployment, position, or order functionality was changed or used
- provider volume is labelled `historical_volume_activity_proxy_not_quote_flow_or_order_book_volume`

## Direct answer

**Long, short, and neutral are the right first-class output states, but the tested one-shot equation could not detect them accurately enough.**

The target itself was meaningful rather than dominated by one class:

| Period | Long | Neutral | Short |
|---|---:|---:|---:|
| 2025 | 24.09% | 50.33% | 25.58% |
| partial 2026 | 26.97% | 49.50% | 23.53% |

However, a causal price-context model mostly learned to say neutral. Its rare argmax long/short predictions had low precision and extremely low recall. A separate cost-aware probability-margin equation emitted more long and short states, but those directions were only correct about 23-25% of the time and lost after 5 bps per side in both periods.

Adding historical-volume activity did not make direction portable.

So the decomposition is useful, but the next detector should be **dynamic**. It should update two competing first-touch hazards after each completed bar and remain neutral until one side has supported positive expectancy. Loops are not required at this stage.

## Frozen experiment

The population deliberately removed the earlier setup-selection problem:

- two fixed decision bars per complete stock-session: ordinals 12 and 36;
- two non-overlapping 24-bar outcome windows;
- no setup, loop, regime, symbol-identity, or direction filter;
- exact regular-session five-minute timestamps only;
- 20 stocks in 2024/2025 and the frozen 19-stock 2026 cohort;
- 2024 supplied past-only training history;
- 2025 and partial 2026 were scored prequentially;
- every scored session used only the previous 120 completed sessions, with a minimum of 60 sessions and 1,500 rows.

The outcome-free seal contained 24,410 anchors:

| Period | Frozen anchors |
|---|---:|
| 2024 warm-up | 9,931 |
| 2025 score period | 9,926 |
| partial 2026 score period | 4,553 |

There were 24,096 exact outcome paths. Primary model metrics used 9,815 rows in 2025 and 4,501 in partial 2026. Missing exact paths remained in the sealed population and were counted rather than silently removed before the seal.

## What long, short, and neutral meant

At each decision:

`prior scale = median true-range bps of the previous 12 completed bars`

`barrier = clip(4 × prior scale, 40 bps, 250 bps)`

Entry for economic translation was the exact next-bar open. Over the next 24 bars:

- **long:** the upper barrier was uniquely touched first;
- **short:** the lower barrier was uniquely touched first;
- **neutral:** neither barrier was touched, or both first appeared inside the same five-minute bar and their order was unknowable.

The barrier was often clipped at 250 bps at the first clock, yet neutral still represented about half the observations. The score periods had no first-touch dual-bar ambiguity, so their neutral observations were genuine no-touch paths rather than an OHLC-order artefact.

## Equations

All three models were fixed L2 multinomial logistic equations.

### M0: clock prior

Only the predeclared decision clock. This measured the rolling causal class base rate.

### M1: price context

Compact causal price information only:

- current range, body, close location, and upper/lower wicks;
- one-, three-, six-, and twelve-bar returns;
- recent absolute movement and compression;
- session return and distance from the causal session typical-price mean;
- opening-range position and width;
- causal scale and barrier width.

Every continuous feature was scale-normalised where appropriate. Symbol identity was excluded so the equation could not memorise individual stocks.

### M2: price context plus activity

M2 added only two predeclared fields:

- current historical-volume activity relative to the prior 12 bars;
- recent three-bar activity relative to the prior 12 bars.

These are provider historical-volume activity proxies, not quote flow, order-book volume, or exchange participation evidence.

## Predictive result

| Period | Model | Accuracy | Balanced accuracy | Macro F1 | Log loss |
|---|---|---:|---:|---:|---:|
| 2025 | M0 clock prior | 0.503 | 0.333 | 0.223 | **1.0415** |
| 2025 | M1 price | 0.498 | 0.343 | 0.267 | 1.0423 |
| 2025 | M2 price + activity | 0.496 | 0.342 | 0.269 | 1.0434 |
| 2026 | M0 clock prior | 0.495 | 0.333 | 0.221 | **1.0461** |
| 2026 | M1 price | 0.489 | 0.332 | 0.227 | 1.0589 |
| 2026 | M2 price + activity | 0.490 | 0.333 | 0.231 | 1.0595 |

The clock comparator always classified neutral. M1 and M2 improved macro F1 by emitting a few directional argmax classes, but both made probabilistic log loss worse in both periods.

M1's 2025 macro-F1 improvement had a positive block interval, but its log-loss improvement was negative and insecure. In 2026, its log-loss deterioration was clear: -0.01274 improvement, with a 95% block interval from -0.02036 to -0.00522.

Adding activity to price context also worsened log loss:

- 2025 M2 minus M1 improvement: -0.00112;
- 2026 M2 minus M1 improvement: -0.00061.

The activity proxy therefore did not supply the missing directional information.

## What the model actually predicted

The M1 argmax classifier emitted:

| Period | Predicted long | Predicted neutral | Predicted short |
|---|---:|---:|---:|
| 2025 | 2.93% | 93.60% | 3.46% |
| 2026 | 1.00% | 98.33% | 0.67% |

Its class-specific results were weak:

| Period | Class | Precision | Recall |
|---|---|---:|---:|
| 2025 | Long | 29.51% | 3.60% |
| 2025 | Short | 30.00% | 4.06% |
| 2026 | Long | 22.22% | 0.82% |
| 2026 | Short | 16.67% | 0.47% |

This is not a high-accuracy selective detector. It is a neutral-base-rate model with occasional, poorly resolved directional guesses.

## Cost-aware state result

The predeclared economic translation did not force argmax classification. It estimated:

`long proxy EV = (p_long − p_short) × barrier − 10 bps`

`short proxy EV = (p_short − p_long) × barrier − 10 bps`

It emitted the higher positive direction and otherwise remained neutral. Actual path payoff—not proxy EV—governed the result.

| Period | Model | Directional coverage | Directional precision | Net/directional output | Net/full opportunity |
|---|---|---:|---:|---:|---:|
| 2025 | M0 clock prior | 14.98% | 22.11% | -23.82 bps | -3.57 bps |
| 2025 | M1 price | 45.24% | 24.41% | -14.86 bps | -6.72 bps |
| 2025 | M2 price + activity | 46.31% | 24.82% | -12.33 bps | -5.71 bps |
| 2026 | M0 clock prior | 1.89% | 11.76% | -128.37 bps | -2.42 bps |
| 2026 | M1 price | 49.50% | 23.16% | -19.38 bps | -9.59 bps |
| 2026 | M2 price + activity | 51.06% | 23.46% | -17.63 bps | -9.00 bps |

Every M1/M2 net interval was wholly negative. For M2, the 95% session-block intervals for net per directional output were:

- 2025: -21.85 to -2.98 bps;
- partial 2026: -31.79 to -2.63 bps.

The failure was not caused by the 5 bps-per-side assumption. At only 2.5 bps per side, M2 still averaged:

- -7.33 bps per directional output in 2025;
- -12.63 bps in partial 2026.

No accuracy, economic, time-stability, stock-deletion, cost-survival, or multiplicity qualification was possible.

## Interpretation

### The three-state framing is still useful

The experiment successfully separated three questions that a binary entry signal hides:

1. Is there a usable upward path?
2. Is there a usable downward path?
3. Is neither side currently supported or observable?

Neutral was a real and common market state. Treating it as a first-class output is better than forcing every observation into long or short.

### Static context does not know which side wins

The class distribution was stable enough and both directions had ample support. The failure was not a missing-class problem. The local completed-bar snapshot simply did not contain portable information about which substantial barrier would touch first.

### Macro F1 alone is misleading here

M1 improved macro F1 in 2025 because it occasionally guessed a directional minority class. Those guesses were not calibrated, portable, or profitable. Log loss and realised economic paths both worsened. A first-class detector must satisfy all three:

- direction discrimination;
- probability calibration;
- positive realised post-cost payoff with coverage.

### Activity did not supply timing

Provider historical-volume activity slightly reduced the magnitude of economic losses relative to M1, but it worsened probabilistic log loss and remained securely negative. It should not be described as volume confirmation.

### Loops are not the immediate missing variable

No loop information was used. The detector failed before loop identity became relevant. A loop can remain a later structural/context comparator, but it should not be added until a directional state model works and the loop demonstrates incremental payoff information.

## The next equation

The next clean model is a **dynamic competing-risk state detector**, not another static feature matrix.

After each completed bar, estimate:

`h_up(k) = P(upper barrier touches next | neither touched yet, causal path through k)`

`h_down(k) = P(lower barrier touches next | neither touched yet, causal path through k)`

Survival supplies the neutral probability:

`S(k) = product(1 − h_up(j) − h_down(j))`

The resulting state is:

- long only when the lower uncertainty bound of realised long expected net is above zero;
- short only under the symmetric condition;
- neutral/unknown otherwise.

This is a better match for the combined evidence:

- the static three-state snapshot failed here;
- earlier one-to-three-bar path research produced the strongest occurrence-level ranking information;
- that earlier information was better at vetoing bad paths than confirming good entries;
- a hazard model explicitly represents when direction emerges and preserves abstention while it has not.

`work/contracts/20260714-dynamic-long-short-neutral-prospective-log-v1.json` records this prospective-only hypothesis. It is not activated and does not authorise an application logger or trading implementation.

## Retired opened-data paths

Do not tune on these opened results:

- the fixed decision ordinals 12 and 36;
- the 4× scale barrier or its 40/250 bps clamps;
- the 24-bar horizon;
- M0, M1, or M2 regularisation, features, or training window;
- post-score price-feature subsets or interactions;
- historical-volume thresholds or interactions;
- forced directional thresholds selected from the probability tails.

Any dynamic-hazard implementation should be frozen before genuinely new observations or used only to specify a future logging contract.

## Integrity and reproducibility

- Frozen contract SHA-256: `11d9a61bc72fe95cd5fa3c6809f971b1eeb053392c8175334f3d090210d751d1`.
- Frozen runner SHA-256: `71581388a32d028f91c40feafbba2df1e0ca572a17227a9ae7c992138e78f45f`.
- Frozen auditor SHA-256: `4482e2abd87d881891e5f43e291841187699680e086887444d88d00c8fe65623`.
- Focused tests SHA-256: `6dfdc61bfaa472ca032bf242bc9360c454b4ebe9f13d12905a3c39942c797371`.
- Pre-score manifest SHA-256: `39d9b9a33d4660709e758bd7504440b22cc6f610796ef7bcf9356d062b50c963`.
- Artifact manifest SHA-256: `83960bb61d91088af7b9dace64bd1b455d2e7cf6457c59132e0f8da03a1bb6f8`.
- Focused tests: 13/13 passed.
- Independent audit: 12/12 checks passed.
- Maximum independently reconstructed causal-feature, target-path, payoff, and sampled model-refit error: `0.0`.
- Exact rerun: every artifact file was byte-identical.
- The optional Arbor evaluator was not present.
- The pre-existing dirty `StockerLocal` worktree was not modified.

Primary artifacts:

`work/artifacts/20260714-long-short-neutral-detector-v1/primary`

Exact rerun:

`work/artifacts/20260714-long-short-neutral-detector-v1/exact_rerun`
