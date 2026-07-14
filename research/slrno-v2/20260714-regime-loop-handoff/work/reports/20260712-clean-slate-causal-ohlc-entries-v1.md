# Clean-slate causal OHLC entries V1

Date: 2026-07-12

Decision: `both_candidates_rejected_without_rescue_tuning`

Scientific status: 2024 internal monthly expanding out-of-fold entry research. This is retrospective development evidence, not prospective validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no broker, live, demo, paper, position, deployment, or strategy-promotion path exists
- provider-volume label: `historical_volume_not_used`
- no regime, state, loop, cycle, B0, template, named pattern, prior detector output, volume, quote, tick, news, or fundamental input was used

## Question

Can fixed algorithms identify causal long, short, or abstain entries directly from continuous measurements of completed five-minute OHLC bars, without using the earlier regime/loop research?

The experiment is a clean slate with respect to features and models. It reuses the previously assembled twenty-two-stock cohort, so it is not independent with respect to universe selection and cannot support broad-universe claims.

## Frozen design

The source was 2024 regular-session provider OHLC for twenty-two stocks. Only `timestamp`, `open`, `high`, `low`, and `close` were read. Parquet timestamp predicates prevented 2023, 2025, or 2026 rows from being materialized.

Every completed bar with an exact contiguous same-session horizon was a candidate decision. Gaps started a new causal segment. Entry was the next five-minute bar open and exit was the close exactly 6, 12, or 24 bars after the decision bar. The regression target was:

`10,000 × (exit close / next-bar open - 1)`

July through December 2024 were scored one month at a time. Each model was refitted using only anchors whose target exit occurred strictly before the score month. Preprocessing medians and scalers were fitted on training rows only, and each represented stock received equal total training weight.

The fixed feature surface contained forty continuous causal measurements:

- six session-clock terms;
- current bar return, range, body, wick, and close-location geometry;
- contiguous 1/3/6/12-bar returns;
- rolling absolute return, volatility, and range measurements;
- session return, running high/low distance, and running range location;
- rolling 6/12-bar high/low distances;
- rolling-history availability fractions.

The algorithms were:

- `clock_ridge`: time-only Ridge null baseline;
- `full_ridge`: the complete feature surface with fixed Ridge regularization;
- `full_hgb`: a fixed shallow histogram gradient-boosting regressor.

There was no cross-validation, hyperparameter search, early stopping, feature selection, ensembling, calibration, target clipping, or result-driven algorithm choice.

The primary action rule was symmetric and fixed before scoring:

- prediction at least +10 gross bps: long;
- prediction at most -10 gross bps: short;
- otherwise: abstain.

Non-overlapping actions were selected greedily within stock/session/model/horizon. Primary costs were five bps per side. The predeclared ±20 and ±40 bps thresholds were descriptive sensitivities only and were forbidden from rescuing a failed primary result.

## Data and support

- 424,583 accepted regular-session OHLC rows;
- 252 union session dates and 5,539 stock/session pairs;
- 2,612 within-session non-five-minute gaps;
- 195,292 July-December candidate bars at six bars;
- 177,276 at twelve bars;
- 143,472 at twenty-four bars;
- 128 scored session dates;
- 1,548,120 frozen algorithm/horizon predictions and actions.

The provider contained 5,539 invalid OHLC placeholder rows that were excluded and explicitly counted. AXTI and OKLO accounted for 5,167 of them and 2,481 of the 2,612 gaps. Exact-continuity checks prevented these missing intervals from being treated as elapsed five-minute bars.

Predictions and fixed actions were written and hashed before any July-December validation price or target was joined.

## Prediction result

Relative improvements are versus the clock-only Ridge baseline.

| Candidate | Horizon | MSE improvement | MAE improvement | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full Ridge | 6 | -0.0585% | -0.0618% | 0.0123 | 0.0110 |
| Full Ridge | 12 | +0.0590% | -0.0270% | 0.0264 | 0.0112 |
| Full Ridge | 24 | +0.0651% | -0.0776% | 0.0311 | 0.0074 |
| Shallow HGB | 6 | +0.0125% | +0.0229% | 0.0208 | 0.0138 |
| Shallow HGB | 12 | +0.0942% | +0.0165% | 0.0328 | 0.0131 |
| Shallow HGB | 24 | +0.2029% | +0.0263% | 0.0487 | 0.0192 |

The nonlinear model extracted a weak signed-return association, strongest at twenty-four bars, but every MSE gain was below the frozen 0.25% minimum. None of the daily paired MSE-improvement block-bootstrap intervals had a lower bound above zero.

Temporal stability was also insufficient. Positive MSE-improvement months were 3/6, 3/6, and 3/6 for HGB at 6/12/24 bars. Ridge achieved 3/6, 4/6, and 2/6. Stock-deletion breadth was generally better than month persistence, but it did not overcome the small and statistically unresolved pooled gains.

## Primary entry result

All cells below use the frozen ±10 predicted-bps action threshold and five bps per side.

| Candidate | Horizon | Trades | Long / short | Mean net trade | Long mean | Short mean | Cumulative return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Ridge | 6 | 7,552 | 3,136 / 4,416 | -9.95 bps | -2.16 bps | -15.48 bps | -29.63% |
| Full Ridge | 12 | 8,485 | 3,153 / 5,332 | -3.14 bps | +6.62 bps | -8.91 bps | -12.34% |
| Full Ridge | 24 | 6,155 | 2,192 / 3,963 | -6.94 bps | +2.00 bps | -11.89 bps | -19.94% |
| Shallow HGB | 6 | 4,905 | 2,352 / 2,553 | -6.71 bps | +1.41 bps | -14.20 bps | -14.65% |
| Shallow HGB | 12 | 5,116 | 1,799 / 3,317 | -0.22 bps | +12.67 bps | -7.21 bps | -1.46% |
| Shallow HGB | 24 | 4,703 | 1,387 / 3,316 | -12.48 bps | -4.30 bps | -15.90 bps | -25.00% |

Every primary short cell was negative. Several long cells were positive, especially HGB at twelve bars, but long-only selection was not a frozen candidate and cannot be used as a post-score rescue.

At zero assumed cost, four of the six candidate cells had positive mean trade returns. At five bps per side every primary cell was negative. The clean-slate entry information was therefore weak and friction-sensitive.

Monthly and stock-deletion persistence failed:

- Ridge positive months: 0/6, 2/6, 2/6; positive stock deletions: 0/22 at every horizon;
- HGB positive months: 1/6, 3/6, 1/6; positive stock deletions: 0/22, 6/22, 0/22.

No candidate/horizon had an absolute daily-return bootstrap lower bound above zero or a paired daily-return-advantage lower bound above zero versus the clock baseline.

## Descriptive confidence sensitivities

The predeclared higher action thresholds cannot alter the rejection. They are reported because they may define a genuinely new future hypothesis.

At ±40 predicted bps and five bps per side, HGB produced:

| Horizon | Trades | Mean net trade | Cumulative return | Positive months | Positive stock deletions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 438 | +14.50 bps | +2.89% | 4/6 | 22/22 |
| 12 | 644 | +27.32 bps | +8.71% | 4/6 | 22/22 |
| 24 | 700 | +25.24 bps | +8.57% | 5/6 | 22/22 |

This is not a retained result:

- threshold sensitivities were explicitly barred from rescuing the primary model;
- six-bar support was below the frozen 500-trade minimum;
- these cells did not receive the primary bootstrap retention test;
- they were observed inside the same retrospective 2024 development experiment;
- selecting ±40 now would be outcome-informed threshold choice.

The correct interpretation is a follow-up question: does the shallow nonlinear score rank a small high-confidence tail more effectively than its absolute calibration suggests? Answering that requires a separately frozen V2 rule and genuinely new post-freeze sessions, not another pass over these outcomes.

## Decision

Both `full_ridge` and `full_hgb` failed the all-horizon forecast, support, temporal, stock-deletion, long/short, bootstrap, and five-bps-per-side economic gates.

Retained candidates: none.

The defensible findings are narrower:

- raw causal OHLC measurements contain a very small signed-return association;
- a shallow nonlinear model captures more of it than a linear model;
- the association is too weak for the frozen general entry rule;
- short predictions were the main primary drag, but changing to long-only now would be post-score adaptation;
- stronger predicted-score tails are interesting but remain unvalidated descriptive evidence;
- no directional reliability, economic edge, tradability, or prospective success has been established.

## Skill-workflow impact

The bar-pattern research workflow kept this to two fixed candidate algorithms, required exact data and volume labels, enforced next-open execution and explicit friction, and prevented the attractive higher-threshold cells from being promoted after the primary rule failed.

## Integrity and reproducibility

- contract SHA-256: `129d9765545efc0d07ce5752fc8cc22aaf118856caafdb48dcfdfd18eaab763d`
- frozen runner SHA-256: `c6199de8dbd6456d6008fa297ea825894da64d779f6d25a054c8b7127da59114`
- pre-score manifest SHA-256: `c82b70b477d8ff0900052a96ead03b5dd9d66fca6afa05fa69f897b4b552fb88`
- independent auditor SHA-256: `85c19de35f5728c7e654e777c88d215e6242af90a74c5b6b170b46e9f2d76ce9`
- independent audit-result SHA-256: `9c4a5b4ec119eb0b2840c071fc3f450fe2cb03da0060e57f8068a63019ef050e`
- independent audit: 18/18 checks passed
- exact prediction replay: 1,548,120 rows, maximum absolute error 0
- exact outcome replay: 516,040 rows, maximum absolute error 0
- maximum downstream table difference: `1.46e-11`
- no later-period or forbidden detector input was read

Artifact root: `/private/tmp/stocker_clean_slate_causal_ohlc_entries_v1_20260712`

The artifact root is ephemeral and should be archived before reboot if exact replay without recomputation is required.
