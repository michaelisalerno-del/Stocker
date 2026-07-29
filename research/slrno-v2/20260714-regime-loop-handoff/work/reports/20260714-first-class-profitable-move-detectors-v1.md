# First-class profitable-move detectors V1

Date: 2026-07-14

Decision: **`all_detectors_rejected_or_descriptive_only`**

Scientific status: causal retrospective portability development on already-opened 2025 and partial-2026 data. This is not validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no trading app, broker, paper/demo account, deployment, position, or order path was changed or used
- provider historical volume is labelled `historical_volume_activity_proxy_not_quote_flow_or_order_book_volume`

## Direct answer

The five tested entry-condition detectors do **not** detect profitable moves with great accuracy. None reached 60% target-before-invalidation precision, none had positive mean payoff after 5 bps per side, none securely beat its outcome-blind matched control, and none survived the frozen stability and robustness gates.

The strongest-looking cell was `failed_breakdown_reclaim_long` in partial 2026:

- 1,381 scored paired events;
- 50.47% target-first precision versus 46.49% for controls;
- +3.98 percentage-point paired lift, with a 95% session-block interval of -0.37 to +8.76 points;
- -9.07 bps mean net per event, with a 95% interval of -13.22 to -4.75 bps;
- Holm-adjusted precision-lift p-value 0.392.

It was worse in 2025: 45.85% precision, -1.53-point lift, and -14.52 bps mean net. It is therefore not a detector to retain.

The main conclusion is sharper than “these patterns failed.” A single completed-bar dictionary condition can identify an alert or movement context, but it did not identify whether favorable excursion would occur **before** adverse invalidation. That ordering is the first-class detection problem.

## What was tested

Five long-only hypotheses were frozen before outcomes were opened:

1. `H1_downside_expansion_exhaustion_long`: downside expansion with a large negative body, low close location, and negative three-bar displacement.
2. `H2_failed_breakdown_reclaim_long`: breach of the prior 12-bar low followed by a close back above it with lower-wick rejection.
3. `H3_two_bar_reversal_confirmation_long`: downside expansion followed by a positive bar closing above the expansion midpoint.
4. `H4_opening_range_failed_breakdown_long`: opening-range low breach and reclaim during the frozen session window.
5. `H5_activity_activated_downside_expansion_long`: H1 plus elevated provider historical-volume activity relative to the prior six completed bars.

Loops, regimes, realised topology, future volume, future extrema, and future outcome fields were excluded from the detector inputs. The purpose was to test whether a first-class entry detector could stand on causal price, bar, time, and activity information alone.

Every event used:

- a decision only after the signal bar completed;
- hypothetical entry at the next provider open;
- +1R target and structural invalidation;
- a 24-bar maximum path;
- conservative stop-first treatment when a five-minute bar touched target and stop;
- a 24-bar causal cooldown by detector family;
- 5 bps per side as the primary cost.

Primary accuracy was target-before-invalidation precision over **all emitted events**. Stops, ambiguous dual touches, and no-touch paths were not hidden. One outcome-blind control was matched by symbol, period, month, clock bucket, and causal scale.

## Frozen population

The admission-only seal contained 8,656 events and 8,656 controls. Outcomes produced 8,544 scored pairs.

| Detector | 2025 scored | 2026 scored |
|---|---:|---:|
| H1 downside expansion exhaustion | 655 | 261 |
| H2 failed breakdown reclaim | 2,949 | 1,381 |
| H3 two-bar reversal confirmation | 1,283 | 573 |
| H4 opening-range failed breakdown | 473 | 239 |
| H5 activity-activated downside expansion | 519 | 211 |

The original 20-stock universe was used in 2025. AAL was excluded from every 2026 result under the frozen source contract, leaving 19 stocks. One off-grid CIFR source row was excluded without timestamp rounding before the event ledger was frozen.

## Primary results

| Detector | Period | Precision | Control | Lift | Mean net/event | Fixed h24 net |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 2025 | 32.37% | 40.61% | -8.24 pp | -20.38 bps | +3.15 bps |
| H1 | 2026 | 40.61% | 38.70% | +1.92 pp | -14.73 bps | -12.49 bps |
| H2 | 2025 | 45.85% | 47.37% | -1.53 pp | -14.52 bps | -1.32 bps |
| H2 | 2026 | 50.47% | 46.49% | +3.98 pp | -9.07 bps | -5.21 bps |
| H3 | 2025 | 45.91% | 47.47% | -1.56 pp | -15.07 bps | +5.70 bps |
| H3 | 2026 | 45.03% | 50.44% | -5.41 pp | -15.47 bps | -7.11 bps |
| H4 | 2025 | 39.75% | 48.63% | -8.88 pp | -22.79 bps | +4.37 bps |
| H4 | 2026 | 47.70% | 51.05% | -3.35 pp | -11.89 bps | -41.78 bps |
| H5 | 2025 | 30.64% | 37.76% | -7.13 pp | -21.89 bps | +7.83 bps |
| H5 | 2026 | 42.18% | 39.81% | +2.37 pp | -13.72 bps | +5.14 bps |

Every dynamic-net 95% session-block interval was wholly negative. Even at only 2.5 bps per side, all ten detector-period means remained negative. Transaction costs matter, but they are not the reason these detectors failed.

No 2025 detector had positive paired precision lift. In partial 2026, the positive lifts for H1, H2, and H5 all had intervals crossing zero and failed Holm control.

## Stability and breadth

The result is not caused by one symbol:

- no leave-one-stock-out deletion made any detector-period dynamic mean net positive;
- no deletion reached 60% target-first precision;
- H2 had zero positive-net months in all 12 months of 2025 and all six opened months of 2026;
- H1, H4, and H5 also had zero positive-net months in both periods;
- H3 had only 2/12 positive-net months in 2025 and 1/6 in 2026, while its pooled period means remained negative.

The failure is therefore broad. It is not a promising average spoiled by a single stock or month.

## What the paths say

### Tight invalidation creates ambiguity, but ambiguity is not the whole problem

H1 and H5 used median risks near 25 bps. Target and stop appeared in the same five-minute bar in 25%–34% of scored rows, so these exact paths cannot be ordered from five-minute OHLC. Conservative stop-first scoring was appropriate.

H2 reduced dual-touch ambiguity to about 3% by using a wider structural risk, yet remained near coin-flip precision and negative after costs. H3 made dual touches almost disappear and still failed in both periods. Finer path data would improve label fidelity for tight setups, but it cannot by itself rescue the causal signal.

### Movement potential is not path-order accuracy

H1/H5 often had large full-window MFE and MAE relative to their tight risk. That means the price moved substantially in both directions; it does not mean the entry was good. A profitable detector must forecast which excursion comes first and whether the reward is large enough after gaps and costs.

For H5, every target-first event did so within three bars, but only 30.64% and 42.18% of all admissions followed that path. Conditioning on those future winners would be oracle filtering, not a detector.

### The positive H5 24-bar mean is descriptive, not a hidden rule

H5 was the only detector with a positive fixed-24-bar net mean in both periods, so the frozen secondary horizon was examined without changing the entry condition:

| H5 fixed h24 diagnostic | 2025 | 2026 |
|---|---:|---:|
| Rows | 519 | 211 |
| Mean net | +7.83 bps | +5.14 bps |
| Median net | -2.61 bps | -12.89 bps |
| 5% trimmed mean | +1.09 bps | -10.23 bps |
| 95% session-block interval | -16.88 to +31.42 | -46.94 to +53.92 bps |
| Positive months | 7/12 | 3/6 |
| Net at 7.5 bps/side | +2.83 bps | +0.14 bps |
| Net at 10 bps/side | -2.17 bps | -4.86 bps |

The paired fixed-h24 advantage over controls was +25.65 bps in 2025 and +6.65 bps in 2026, but the respective block intervals were -11.76 to +62.43 and -59.06 to +68.50 bps. Both cross zero widely.

More importantly, H5 events that failed the structural target-first test still averaged -34.03 bps at 24 bars in 2025 and -29.29 bps in 2026. Removing the stop would not have converted the losing path into a profitable one on average. The positive headline mean is unstable and positively skewed, not evidence of a reliable delayed bounce.

## How to define a first-class detector

“Great accuracy” must not mean a cosmetically high win rate. A closer target or wider stop can raise hit rate while reducing expectancy. With symmetric reward and loss `R` and round-trip cost `c`, the approximate break-even probability is:

`p > (1 + c / R) / 2`

At H2's roughly 43 bps median risk and 10 bps round-trip cost, break-even is about 61.6%, before gap asymmetry. At a 25 bps risk it is about 70%. A fixed 60% accuracy target is therefore not sufficient for every admission.

The detector should instead be a selective state machine:

1. **Alert:** identify a possible onset state, with no admission assumed.
2. **Confirm:** update after each completed bar and ask whether the move has actually begun without invalidation.
3. **Admit or abstain:** emit positive only when the lower uncertainty bound of cost-adjusted expectancy is above zero.
4. **Manage separately:** after hypothetical admission, predict remaining payoff from the completed path; do not reuse the entry label as an exit rule.

Its primary outputs should be calibrated target-before-invalidation probability, expected win and loss sizes, expected net bps, uncertainty, support, and an explicit `unknown_abstain` state. Performance must be reported jointly as precision **and coverage**. A 90% hit rate on a tiny hindsight-selected subset is not a detector.

## Next research path

Do not add another large bar-condition matrix to the opened periods. The clean next hypothesis is `alert → causal confirmation → continuation`, but V1 has already exposed these outcomes. Any reclaim level, bar count, stop width, target width, or continuation horizon selected now would be post-score tuning.

`work/contracts/20260714-first-class-profitable-move-prospective-log-v1.json` specifies a future non-executing research ledger:

- log every eligible completed bar before detector selection;
- keep alert, confirmation, admission, active path, and outcome clocks separate;
- allow positive, negative, or unknown/abstain;
- use risk-specific break-even expectancy rather than a fixed hit-rate threshold;
- preserve outcome seals, source timestamps, support, costs, controls, calibration, time stability, leave-one-stock-out checks, and multiplicity control;
- optionally use a separately frozen finer-bar source to resolve five-minute dual touches;
- require separate authorization for any logger or application implementation.

No retrospective V1 detector is retained for promotion or prospective activation.

## Integrity and reproducibility

- Pre-score manifest SHA-256: `f3c0d4bae26ce371fc96b90bee9d60af340692ef8180bf5382069357e56610c0`.
- Frozen contract SHA-256: `00a80375accbc0f29f36900c5d5d78e7a37e0ca495c097f38365d52d873cd8a9`.
- Frozen runner SHA-256: `4a8eacf8b57abf0971c2e9d5e43cef4bc1685d88a7cfce028c1e4ab06a64b5ed`.
- Frozen auditor SHA-256: `62b9d9013317abba0bc936e2fc7952076b722add79c59b343ff1d083b550bd57`.
- Audit addendum SHA-256: `8ff113904d0892611d3beb2d876fca1de98467b27c99c7511579048aebcd9caf`.
- Focused tests SHA-256: `21968371caaa6284d7232dc2c700623f5fff3c4fc0ba08b214717e9ff030c730`.
- Artifact manifest SHA-256: `26cf5de2e603bdbd3229299b8b7048ddfe8e121e6ac769497a86fbcbce85ce14`.
- Focused tests: 10/10 passed; lint passed.
- Independent audit: 12/12 checks passed with maximum event/control/path error 0, maximum aggregate error `1.78e-15`, and maximum bootstrap/Holm error `3.55e-15`.
- The frozen auditor was preserved after its comparison loop encountered an unscored missing-path row. A transparent addendum supplied placeholders only for absent fields on unscored rows; every scored path and aggregate continued through the original independent implementation.
- Exact rerun: all 19 files were byte-identical.
- The pre-existing dirty `StockerLocal` worktree was not modified by this experiment.

Primary artifacts:

`work/artifacts/20260714-first-class-profitable-move-detectors-v1/primary`

Exact rerun:

`work/artifacts/20260714-first-class-profitable-move-detectors-v1/exact_rerun`
