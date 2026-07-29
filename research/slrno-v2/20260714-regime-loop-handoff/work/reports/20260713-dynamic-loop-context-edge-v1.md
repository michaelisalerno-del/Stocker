# Dynamic loop × regime profitability and drift V1

Date: 2026-07-13

Decision: **`dynamic_loop_context_profitability_hypothesis_not_supported`**

Scientific status: post-inspection retrospective development and backward-portability test. This is not prospective validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no broker, paper/demo account, app runtime, deployment, or order path was used
- provider volume is labelled `historical_volume_activity_proxy`; no quotes, tick counts, spread, or order-book fields were available

## Direct answer

The observation contains one true part and one unsupported part:

1. **True descriptively:** loop × current-regime cells move in and out of profitability. Most supported profitable streaks lasted one 20-session block, and adjacent-block profitability was only weakly persistent.
2. **Not supported actionably:** a causal rolling dictionary could not identify the profitable cells early enough to produce portable net returns. It lost after 5 bps per side in five of six period × horizon cells and did not beat the loop-only selector reliably.

So the issue is no longer whether drift exists. It does. The issue is that the drift is faster and noisier than the historical winner table can track.

## Frozen test

The test reused the already-frozen causal breakout ledger and did not change entries, exits, overlap handling, or costs.

- Data: EODHD/provider 5-minute regular-session OHLCV.
- Universe: the frozen common 20-stock cohort.
- Periods: forward-ordered 2025 and backward-portability 2023, 250 sessions each.
- Warm-up: first 60 completed sessions of each period.
- Score surface: 190 sessions per period and 40,512 accepted signal rows across 6-, 12-, and 24-bar horizons.
- Economic translation: causal breakout, fixed exit at 6/12/24 bars, 5 bps per side.
- Loop identity: highest causal `loop_score` at the anchor. The later realized loop label was never loaded.
- Primary cell: top predicted parent loop × current state. Parent loops are rotation-invariant, so current state identifies their orientation.
- Rolling estimate: previous 60 completed sessions only; minimum 20 filled trades; 50-trade shrinkage first toward the loop mean and then the rolling global mean.
- Active rule: shrunk historical net mean greater than zero.
- Overlay boundary: the selector could reject a frozen accepted signal but could not refill a later overlapping opportunity.

Secondary context families were tested diagnostically and could not replace the frozen primary: previous regime, current-plus-previous regime, direction, volatility, bar range, session phase, historical relative volume, and their joint matrix.

## Primary result

The horizons correspond to approximately 30, 60, and 120 minutes.

| Period | Horizon | Filled trades | Mean net/trade | Portfolio return | Loop-only mean net/trade |
|---|---:|---:|---:|---:|---:|
| 2025 | 6 bars | 2,609 | -8.22 bps | -10.31% | -7.68 bps |
| 2025 | 12 bars | 2,351 | -9.20 bps | -10.50% | -0.14 bps |
| 2025 | 24 bars | 1,888 | -0.01 bps | -0.72% | -4.32 bps |
| 2023 | 6 bars | 3,421 | -1.19 bps | -2.38% | -2.24 bps |
| 2023 | 12 bars | 2,762 | -7.50 bps | -10.33% | -5.33 bps |
| 2023 | 24 bars | 2,342 | +1.25 bps | +0.49% | -7.33 bps |

Only backward-2023 at 24 bars was slightly positive. It was not robust:

- no paired daily-return 95% lower bound versus unfiltered or loop-only was above zero;
- all 18 familywise bootstrap endpoints failed Holm adjustment;
- only 2/4 quarters were positive;
- only 13/20 leave-one-stock-out deletions stayed positive.

The most damaging direct comparison occurred in 2025 at 12 bars: regime conditioning was worse than loop-only, with the paired daily interval entirely below zero before multiplicity adjustment.

## How fast did the map drift?

Cells needed at least five filled trades inside a 20-session block to enter this descriptive table.

| Period | Horizon | Adjacent sign agreement | Positive-cell retention | Adjacent rank correlation | Median profitable streak |
|---|---:|---:|---:|---:|---:|
| 2025 | 6 | 47.4% | 42.7% | -0.081 | 20 sessions |
| 2025 | 12 | 50.7% | 52.0% | -0.011 | 20 sessions |
| 2025 | 24 | 46.7% | 47.4% | +0.102 | 20 sessions |
| 2023 | 6 | 60.7% | 54.8% | +0.086 | 20 sessions |
| 2023 | 12 | 60.1% | 52.7% | +0.192 | 20 sessions |
| 2023 | 24 | 52.5% | 47.1% | -0.002 | 40 sessions |

The mean profitable streak was roughly 31–40 sessions because a minority of cells lasted much longer, but the median was only about one trading month in five of six surfaces. The rank correlation of cell returns from one block to the next was close to zero.

This is the practical timeframe answer: **the typical hindsight label lasts around 20 sessions, but it cannot be forecast from the prior 20-session label with useful reliability.** A faster refresh alone is therefore unlikely to solve the problem; it would also reduce support and increase noise.

## Did “trade only early in the active life” work?

No common expiry rule emerged.

| Period | Horizon | First 1–10 active sessions | Active age 21+ | Late minus early |
|---|---:|---:|---:|---:|
| 2025 | 6 | +1.17 bps | -17.64 bps | -18.82 bps |
| 2025 | 12 | -1.74 bps | -20.52 bps | -18.78 bps |
| 2025 | 24 | -12.61 bps | -2.20 bps | +10.41 bps |
| 2023 | 6 | +1.53 bps | -2.88 bps | -4.41 bps |
| 2023 | 12 | -4.57 bps | -4.62 bps | -0.05 bps |
| 2023 | 24 | +11.61 bps | +3.48 bps | -8.13 bps |

The first ten sessions were positive in only three of six cells. “Expire after ten sessions” would be a post-inspection rule and is contradicted by both 12-bar surfaces and 2025 at 24 bars.

## Other conditions

No secondary family was positive in all six period × horizon cells.

- Direction, historical relative volume, and the full joint matrix were each positive in 3/6 cells, but in different periods and horizons.
- Session conditioning was positive in 1/6.
- Current regime and current-plus-previous regime were each positive in 1/6.
- Previous regime, volatility, and bar-range selectors were positive in 0/6.

This matters because the negative result is not simply caused by choosing the wrong single conditioner. The broad matrix also failed to transfer.

## Two post-inspection leads—not retained rules

After the primary rejection, two 24-bar loop orientations were positive in both opened score periods:

| Cell | 2023 trades / mean net | 2025 trades / mean net | Important failure |
|---|---:|---:|---|
| `cycle_04` while in state 4 | 132 / +26.54 bps | 96 / +30.75 bps | 2023 Q4 averaged -33.54 bps |
| `cycle_07` while in state 5 | 722 / +7.02 bps | 713 / +17.04 bps | 2023 Q3 averaged -13.72 bps |

These were found after inspecting many loop × regime × horizon cells. They are multiplicity-exposed hypotheses, not qualified strategies or evidence that the general selector works. They are the strongest candidates to freeze before a genuinely unseen test because they have support across all 20 stocks for `cycle_07` and 13–16 stocks for `cycle_04`, rather than being single-stock accidents.

## What this means for the app idea

The app should not maintain a simple “currently profitable loop-regime” whitelist and assume the latest winner remains valid. The evidence supports maintaining a research log, but the log needs to retain uncertainty and lifecycle state:

- loop forecast and current-state orientation;
- observation count and shrunk expected return;
- first activation date and age;
- 20-session block history and sign flips;
- cost-adjusted performance;
- explicit `unknown`, `active`, `decaying`, and `retired` research labels;
- no action when support or portability is absent.

The next clean test is not a larger hindsight matrix. It is to freeze the two post-inspection 24-bar orientations above, plus a null/control set, and judge them once on unseen sessions. If they fail, the parent-loop dictionary is probably too coarse for profitability selection and should be split by child/morph or by a directly modelled payoff target rather than by more admission conditions.

## Integrity and reproducibility

- Contract SHA-256: `3c1f9d07de31491650740d9bf3ef7c568d3c679b916de5196f61bab8dd9ddc0b`.
- Runner SHA-256: `695cf6cc676128ac0941df93751428d761d2229f6fc08e6d706e5968d3d27c3d`.
- Independent auditor SHA-256: `111f6562dfbbd1db577d41b58f12d818fc594ab9ba0cf5bf48eea676e33f212b`.
- Independent audit: 18/18 checks passed.
- Maximum loop-probability, selector-estimate, sampled-volume, and daily-return replay errors: `0.0`.
- Complete artifact manifest SHA-256: `9f65b75c95e657c70e09ac54b28593d2431b5be043b408686c8f5dde2a1735d9`.
- Clean exact rerun: all 20 artifact hashes and sizes matched byte-for-byte.
- Leakage scan: no future loop label, future state, target return, next-state, broker, network, or order path in the scored research surface.
- The optional Arbor evaluator was unavailable at the expected repository path; the independent local replay was used instead.
- The pre-existing dirty `StockerLocal` worktree was not modified.

Primary artifact root:

`/private/tmp/stocker_dynamic_loop_context_edge_v1_20260713`

Exact rerun root:

`/private/tmp/stocker_dynamic_loop_context_edge_v1_20260713_exact_rerun`
