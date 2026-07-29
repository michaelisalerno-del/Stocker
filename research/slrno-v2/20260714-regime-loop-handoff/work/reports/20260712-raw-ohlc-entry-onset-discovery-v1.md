# Raw-OHLC entry-onset discovery V1

Date: 2026-07-12

`research_only: true`  
`live_ordering_enabled: false`  
`order_placement: disabled`

Provider data: regular-session five-minute provider OHLC. `historical_volume_not_used`. No regime, state, loop, cycle, B0, template, named pattern, earlier prediction, volume, order-flow, news, or fundamental input was used.

## Decision

No broad entry-sign hypothesis passed every frozen gate. The independent raw-data replay also failed exact probability reconstruction in a capped full-logit fold, so no broad or narrow result is reproducibly retained.

The test nevertheless found one useful descriptive lead: high-tail `full_logit` long onsets reached the upper volatility-scaled barrier before the lower barrier more often than near-time-matched clock controls at all three horizons. The conservative precision lifts were 5.70, 6.82, and 7.97 percentage points at 6, 12, and 24 bars.

That is not yet a retained entry algorithm. The alerts did not establish robustly cleaner whole paths: pre-confirmation adverse movement and horizon-wide directional dominance failed the multiplicity-adjusted moving-block bootstrap gates. Eight of twenty-one full-logit folds also reached the frozen 500-iteration ceiling, including five of the six scored 24-bar folds. An exact independent replay of the December 24-bar fold then produced probabilities differing by as much as 0.001648 despite identical anchors, targets, and bitwise-identical features. The only input discrepancy was floating-point operation order in a mathematically equivalent weight calculation, with a maximum weight difference of `1.776e-15`.

The correct interpretation is:

> The raw OHLC model detects a recurring long-first impulse, especially after downside expansion, but it does not yet distinguish reliably between a clean reversal entry and a brief favorable first touch followed by reversal.

No economic, execution, or tradability claim is permitted.

## Question tested

After a completed bar `t`, can continuous causal OHLC conditions identify the onset of a directional path beginning at the exact next-bar open?

For each 6-, 12-, and 24-bar horizon:

- the reference was the exact open of `t+1`;
- the scale was the median true range over the last 3 to 12 contiguous completed bars ending at `t-1`, floored at one basis point;
- symmetric barriers were one scale above and below the reference open;
- `long_first` meant the upper barrier was reached first;
- `short_first` meant the lower barrier was reached first;
- no hit and same-bar dual-touch ambiguity both counted as failures;
- an open beyond a barrier established that side first;
- no ordering was inferred inside a bar that touched both barriers.

This is an operational first-confirmation definition, not a claim to know the true start of a move.

## Causal design

The test used the inherited 22-stock convenience cohort and 2024 only. It is internal development, not an untouched validation period.

Monthly expanding out-of-fold models produced June through December probabilities. June calibrated July alert thresholds; every later month used the immediately prior out-of-fold month for thresholds. Each score-month probability was generated before that same month's outcomes. Completed earlier months could train later expanding folds. The global alert bundle was written and hashed before the final all-fold evaluation join.

The algorithms were:

- `clock_logit`: time-of-session-only baseline;
- `full_logit`: interpretable multinomial logistic model using 40 continuous causal OHLC and clock features;
- `full_hgb`: shallow nonlinear histogram gradient boosting using the same 40 features.

Alert onset used the prior-month weighted 95th percentile of the side probability. A side could not fire again until its probability fell below the prior-month 75th percentile. This hysteresis turned a persistent high score into one onset rather than repeated bar-by-bar alerts.

Controls were chosen before outcome attachment. Each candidate onset received one clock-model control from the same symbol, month, horizon, and imposed side, normally in the same 15-minute and history-availability bin but on a different session. All 32,662 onsets were matched; more than 99% used the strictest tier.

Target-blind support was:

| Horizon | Annual eligible anchors | June threshold anchors | July-December scored anchors |
|---:|---:|---:|---:|
| 6 | 365,075 | 27,733 | 186,112 |
| 12 | 330,577 | 25,143 | 168,639 |
| 24 | 264,817 | 20,214 | 135,240 |

## Probability layer

Positive values below mean lower loss than the clock-only model. These are absolute loss reductions, not percentages.

| Candidate | Horizon | Log-loss improvement | Brier improvement | Months both better | Leave-one-stock-out both better | Horizon gate |
|---|---:|---:|---:|---:|---:|---|
| HGB | 6 | +0.007338 | +0.002945 | 6/6 | 22/22 | Pass |
| HGB | 12 | +0.003156 | +0.000098 | 3/6 | 22/22 | Fail |
| HGB | 24 | +0.000725 | -0.000880 | 1/6 | 0/22 | Fail |
| Logit | 6 | +0.005544 | +0.002041 | 4/6 | 22/22 | Pass |
| Logit | 12 | +0.001623 | -0.000247 | 3/6 | 0/22 | Fail |
| Logit | 24 | -0.000318 | -0.000838 | 2/6 | 0/22 | Fail |

Only the six-bar probability layers passed. Neither candidate passed across all horizons.

## Alert-onset evidence

Precision is the equal-symbol/equal-session weighted probability that the emitted side reached its barrier first. Wrong-first, no-hit, and ambiguity all remain failures.

| Candidate | Horizon | Side | Onsets | Candidate precision | Matched clock | Lift | Positive months | Positive stock deletions |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| HGB | 6 | Long | 4,350 | 44.86% | 40.63% | +4.22 pp | 6/6 | 22/22 |
| HGB | 6 | Short | 2,374 | 48.06% | 44.59% | +3.46 pp | 4/6 | 22/22 |
| HGB | 12 | Long | 3,270 | 46.13% | 42.50% | +3.63 pp | 4/6 | 22/22 |
| HGB | 12 | Short | 2,134 | 49.94% | 50.71% | -0.78 pp | 3/6 | 0/22 |
| HGB | 24 | Long | 2,286 | 47.27% | 43.17% | +4.10 pp | 5/6 | 22/22 |
| HGB | 24 | Short | 3,105 | 52.12% | 52.07% | +0.05 pp | 4/6 | 14/22 |
| Logit | 6 | Long | 3,696 | 45.21% | 39.51% | +5.70 pp | 6/6 | 22/22 |
| Logit | 6 | Short | 2,357 | 50.35% | 46.26% | +4.09 pp | 5/6 | 22/22 |
| Logit | 12 | Long | 2,753 | 47.62% | 40.81% | +6.82 pp | 6/6 | 22/22 |
| Logit | 12 | Short | 1,476 | 48.21% | 46.98% | +1.22 pp | 2/6 | 22/22 |
| Logit | 24 | Long | 1,888 | 50.55% | 42.59% | +7.97 pp | 5/6 | 22/22 |
| Logit | 24 | Short | 2,973 | 54.43% | 54.04% | +0.39 pp | 4/6 | 22/22 |

The logit-long precision and rapid-confirmation lifts had positive 98.75% one-sided moving-block lower bounds at every horizon. That favorable first-touch evidence was not enough for retention:

- at 6 bars, directional dominance was slightly worse and pre-confirmation adverse movement was worse;
- at 12 and 24 bars, the point estimates for dominance and pre-confirmation adversity improved, but their block-bootstrap lower bounds crossed zero;
- every algorithm/side/horizon cell failed at least one dominance or adverse-path bootstrap requirement;
- most failures were the opposite barrier being reached first, not lack of a move or ambiguous intrabar ordering.

This distinction matters. A correct first touch can still be followed by a larger move in the other direction.

## What the broad alerts looked like

The reason ledger was written before the final evaluation join. A reason-group percentage below is the share of alerts where that group appeared anywhere in the top three absolute model contributions; groups can therefore overlap.

For logit-long onsets:

| Horizon | Most common top-three reason groups |
|---:|---|
| 6 | Bar geometry 71%; range change 70%; location versus extremes 62% |
| 12 | Location versus extremes 68%; recent motion 59%; bar geometry 50% |
| 24 | Bar geometry 69%; recent motion 59%; location versus extremes 58% |

The most common directional contributions were more revealing:

- at 6 bars, range change supported long in 67% of alerts and location supported long in 50%, while current-bar geometry opposed long in 54%;
- at 12 bars, location supported long in 60%, recent motion in 55%, and range change in 35%, while geometry opposed long in 38%;
- at 24 bars, recent motion supported long in 51%, location in 50%, and volatility/range level in 38%, while geometry opposed long in 44%.

This was not simply a bullish-candle detector. Many long alerts fired because broader range, motion, and location context outweighed a completed bar whose geometry still leaned bearish.

Short alerts were much more time-of-day driven. Clock was a top-three group in 75%, 91%, and 84% of logit short onsets at 6, 12, and 24 bars. That makes the short-side result less convincing as a raw price-structure discovery.

The broad groups recurred across all six months and generally all 22 stocks, but they explained model firing more than success: correct-first and failed alerts had nearly identical mean chosen probabilities, and common reason-family frequencies typically differed by only about three percentage points or less.

## Exploratory narrower sign

Outcome inspection exposed a narrower reversal signature. This was not predeclared, is multiplicity-exposed, and cannot rescue V1.

At 6 bars, the ordered logit-long reason sequence was:

1. range change supports long;
2. current-bar geometry opposes long;
3. recent directional motion supports long.

There were 299 such alerts across 21 stocks. Its median completed bar had approximately:

- a -117 bp open-to-close body;
- a 144 bp high-low range;
- a close at 7% of the bar range;
- -115 bp motion over three bars;
- -100 bp motion over six bars.

At 24 bars, the analogous ordered sequence was recent motion support, geometry opposition, then range-change support. Its median bar was even more downside-expanded: roughly a -143 bp body, 166 bp range, 5% close location, and -160 bp over three bars.

In plain language, the candidate is a downside-expansion reversal onset: a sharp bearish bar near its low causes a high-tail long-first alert because the fitted context treats the selloff as exhaustion or mean reversion. It is not a momentum-long pattern.

Raw, unweighted post-hoc rates were 53.5% correct-first for the 299-case six-bar signature and 65.1% for the 106-case 24-bar signature. Their one-to-one matched controls were 36.5% and 40.6%. These figures are descriptive only: they do not use the contract's nested weighting, the subsets were selected after outcome inspection, and the 24-bar subset had only one July case.

## Representative examples

Times are provider UTC. Barrier prices use the exact next open and the frozen lagged scale. Excursions are scale units, not returns.

Correct 24-bar long example:

- IREN decision: `2024-10-21 15:25`; reference open: `15:30` at 9.30000;
- scale: 114.58 bp; lower/upper barriers: 9.19405 / 9.40717;
- probabilities: no-entry 0.0151, long 0.5108, short 0.4741;
- reasons: recent motion `+0.1012`, geometry `-0.0793`, range change `+0.0370`;
- long confirmed at step 3; favorable/adverse/pre-confirmation adverse: 2.589 / 0.282 / 0.282; dominance: +2.307.

Failed 24-bar long example with the same reason order:

- CIFR decision: `2024-11-05 18:10`; reference open: `18:15` at 5.26000;
- scale: 68.79 bp; lower/upper barriers: 5.22394 / 5.29631;
- probabilities: no-entry 0.0054, long 0.5125, short 0.4822;
- reasons: recent motion `+0.0767`, geometry `-0.0732`, range change `+0.0610`;
- short confirmed first at step 4; favorable/adverse/pre-confirmation adverse: 1.922 / 2.931 / 1.668; dominance: -1.009.

Correct 6-bar long example:

- IREN decision: `2024-11-05 18:10`; reference open: `18:15` at 8.92000;
- scale: 57.85 bp; lower/upper barriers: 8.86854 / 8.97176;
- probabilities: no-entry 0.0743, long 0.4752, short 0.4505;
- reasons: range change `+0.0942`, geometry `-0.0522`, motion `+0.0291`;
- long confirmed at step 6; favorable/adverse/pre-confirmation adverse: 1.447 / 0.972 / 0.972; dominance: +0.476.

Failed 6-bar long example with the same reason order:

- ASTS decision: `2024-11-06 17:05`; reference open: `17:10` at 23.58010;
- scale: 63.23 bp; lower/upper barriers: 23.43148 / 23.72967;
- probabilities: no-entry 0.1062, long 0.4726, short 0.4211;
- short confirmed first at step 1; favorable/adverse/pre-confirmation adverse: 0.735 / 3.871 / 1.415; dominance: -3.135.

The paired examples show the remaining problem: very similar causal probabilities and reason orderings can precede opposite path outcomes.

## Time-of-session clue

The predeclared clock-quartile slices suggest that the logit-long lift was strongest early in the session. Versus matched clock controls, first-quartile precision lift was +28.6 pp at 6 bars, +18.5 pp at 12 bars, and +18.5 pp at 24 bars. Support was only 158, 256, and 176 alerts, respectively. Later-quartile lifts were smaller, and the sparse final 12-bar quartile was negative.

This is a development clue, not permission to add an early-session filter to V1 after seeing outcomes.

## Numerical warning

The clock model converged below its 500-iteration ceiling in every fold. HGB ran the frozen 100 rounds with early stopping disabled. Full logit reached 500 iterations in 8 of 21 folds:

- 6 bars: August and December;
- 12 bars: July;
- 24 bars: July, September, October, November, and December.

The apparently strongest 24-bar long result therefore depends heavily on capped fits. The independent auditor reconstructed the December 24-bar training anchor order and target vector exactly and rebuilt all 40 features bitwise identically across 424,583 rows. The runner evaluated the frozen raw weight as `1 / (sessions * rows)`, while the auditor independently evaluated the mathematically equivalent `(1 / sessions) / rows`. Their maximum weight difference was `1.776e-15`; the capped optimizer amplified it to a maximum coefficient difference of 0.0351 and maximum probability difference of 0.001648.

Runner refits were bit-for-bit identical to one another and to the stored predictions. Auditor refits were likewise bit-for-bit identical to one another. This is therefore not random run-to-run nondeterminism, but it is severe sensitivity to negligible floating-point perturbation. The independent December replay produced 568 onsets versus 568 stored onsets, with only 567 overlapping: one stored onset disappeared and a different onset appeared. Because probabilities determine thresholds and timestamps, relaxing the audit tolerance would be statistically dishonest.

Any continuation should first establish numerical stability in a separately versioned diagnostic. The V1 cap must not be changed in place, and the descriptive full-logit findings above must not be treated as reproducible model evidence.

## Recommended next test

Do not turn the exploratory reversal subset into an entry rule on these same outcomes.

The scientifically useful next target is a clean onset rather than first barrier only. Freeze a V2 before new outcomes with, for example:

- emitted side reaches its barrier within three bars;
- pre-confirmation adverse excursion stays below a fixed fraction of the lagged scale;
- horizon-wide directional dominance remains positive;
- the exact six-bar ordered reversal rationale is a separately declared candidate, not a rescue filter;
- a numerically converged logistic implementation and the existing HGB model are compared without tuning on the new sessions.

Record the alerts and reasons on genuinely new post-freeze sessions before attaching outcomes. Existing 2023, 2025, and partial 2026 data can only be labelled development/backward portability because they have already participated elsewhere in the research lineage.

## Integrity and artifacts

Frozen sources:

- contract: `work/contracts/20260712-raw-ohlc-entry-onset-discovery-v1.json`;
- pre-score manifest: `work/contracts/20260712-raw-ohlc-entry-onset-discovery-v1-pre-score.json`;
- runner: `work/run_raw_ohlc_entry_onset_discovery_v1.py`;
- independent auditor: `work/audit_raw_ohlc_entry_onset_discovery_v1.py`;
- tests: `work/tests/test_raw_ohlc_entry_onset_discovery_v1.py`.

Scored artifacts: `/private/tmp/stocker_raw_ohlc_entry_onset_discovery_v1_20260712`.

The `/private/tmp` artifact root is ephemeral and should be archived separately before a reboot if exact ledgers are needed without recomputation.

Independent reconstruction status: **failed**.

Six preliminary audit checks passed: contract and safety, static future-boundary checks, frozen hashes and environment, exact tape/segment/scale reconstruction, sampled feature causality, and exact support. The first exact probability-table comparison failed in `full_logit`, horizon 24, December 2024. The capped fit amplified a `1.776e-15` maximum weight-rounding difference into a 0.001648 probability difference and one changed onset timestamp. No `independent_audit.json` was written and the artifact manifest correctly remains at `pre_independent_audit_complete_artifact_manifest` with 26 bound files.

The full workspace test run produced 323 passes and one failure. The failing test is the artifact-completion test requiring a passing `independent_audit.json`; its failure is the intended visible consequence of the substantive audit failure, not a test-infrastructure defect.

Auditor SHA-256 after its independent join-cardinality corrections: `a01be373ee57cab27b48da10c3fccfe3cff87279b089f8831c3f9749b7694da3`.
