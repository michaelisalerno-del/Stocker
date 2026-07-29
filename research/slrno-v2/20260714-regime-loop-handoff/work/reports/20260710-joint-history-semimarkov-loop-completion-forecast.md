# Joint history-conditioned semi-Markov loop-completion forecast

## Decision

The full order-3 joint history/destination dwell kernel (`q3`) is rejected.
It improved average proper scores over frozen state-only timing, but it did not
clear the frozen ranking, incremental-improvement, long-horizon calibration,
or path-only sanity gates in both scoring periods.

The simpler destination-conditioned dwell kernel (`q1`) is also not retained.
Its gains over frozen state-only timing fell below the predeclared log-loss and
top-three-recall thresholds in both periods. The order-2 kernel (`q2`) was
declared diagnostic before scoring and has no post-hoc promotion path.

Keep the previously retained history-only fixed-loop identity forecaster. This
experiment does not establish a reliable model of which loop completes within
6, 12, or 24 bars.

`research_only: true`

`live_ordering_enabled: false`

`order_placement: disabled`

## Frozen scope

The contract was frozen at `2026-07-10T19:46:51Z` before the new completion
scores were calculated. The eight-state causal detector, the 2024-fitted
last-three-state destination model, and the twenty fixed cycles were not
refitted. The joint kernel was

`P_history(next | prev2, prev1, current)`

`x P_backoff(dwell | prev2, prev1, current, next)`.

Dwell categories were exact durations 1--23 plus an overflow category for
duration at least 24. The hierarchical prior strengths were 256 for
`q1(dwell | current, next)`, 256 for
`q2(dwell | prev1, current, next)`, and 1024 for
`q3(dwell | prev2, prev1, current, next)`. The predeclared expanding-month
2024 screen reproduced `(256, 256, 1024)` before either scoring period was
opened.

Only complete non-terminal 2024 causal runs fitted dwell distributions:
105,410 of 110,949 runs across 252 dates and 22 stocks. The 5,539 terminal
session runs were excluded because their duration is boundary-truncated. The
fit contained 923 duration-overflow observations, 56 supported state/destination
cells, 418 supported order-2 cells, and 2,035 supported order-3 cells.

Forecasts were issued at every causal state-run entry whose zero-based New
York session-bar ordinal was at most 53. Compatible cycle outputs remained
overlapping binary events. No mutually exclusive cycle softmax or `no loop`
class was created.

## Support and causality

Every frozen support gate passed:

| Period | Horizon | Rows | Positive completions | Minimum cycle positives | Stocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025 | 6 | 464,179 | 15,052 | 50 | 22 |
| 2025 | 12 | 464,179 | 24,864 | 184 | 22 |
| 2025 | 24 | 464,179 | 30,459 | 344 | 22 |
| 2023 | 6 | 460,965 | 15,300 | 77 | 20 |
| 2023 | 12 | 460,965 | 24,689 | 327 | 20 |
| 2023 | 24 | 460,965 | 29,594 | 586 | 20 |

Both periods contained all twenty cycles, four quarters, and all eight current
states. The 2024-only pre-score audit passed all 17 checks: source hashes,
count and probability reconstruction, terminal exclusion, smoothing
selection, joint normalization, exact destination marginals, dynamic-program
brute force, cycle count, separation from the shadow harness, and absence of
an execution surface. Maximum joint-normalization error was
`4.44e-16`; maximum destination-marginal error was `1.11e-16`.

## Primary q3 results

### Against frozen state-only timing

Average proper scores improved consistently, but near-term loop ranking did
not improve by the required amount.

| Period | Pooled log-loss improvement | Required | Log-loss daily 95% interval | Brier daily 95% interval | Better cycles | Top-three recall gain | Required |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025 | 0.5874% | 0.5000% | [-0.001090, -0.000838] | [-0.000116, -0.000082] | 17/20 | 0.1037 pp | 0.5000 pp |
| 2023 | 0.6095% | 0.5000% | [-0.001171, -0.000860] | [-0.000123, -0.000084] | 18/20 | 0.2199 pp | 0.5000 pp |

Both loss differences were negative at every horizon, in every quarter, and
under every leave-one-stock-out deletion; the calibration gates against this
baseline also passed. The frozen comparison nevertheless failed in each
period because top-three recall missed the 0.005 absolute requirement.

### Against destination-conditioned timing

The incremental order-3 contribution was too small and failed the strongest
long-horizon calibration comparison.

| Period | Pooled log-loss improvement | Required | Log-loss daily 95% interval | Brier daily 95% interval | Better cycles | Top-three recall gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025 | 0.1749% | 0.2500% | [-0.000382, -0.000255] | [-0.000056, -0.000034] | 16/20 | 0.0213 pp |
| 2023 | 0.1574% | 0.2500% | [-0.000343, -0.000213] | [-0.000049, -0.000028] | 13/20 | 0.0805 pp |

The relative-improvement gate failed in both periods. The per-cycle gate also
failed in 2023 because only 13 of 20 cycles improved. At 24 bars, q3 maximum
supported-bin error was `0.107432` versus q1's `0.015615` in 2025, and
`0.077251` versus `0.015146` in 2023. Both exceed the frozen baseline-plus-0.01
tolerance, even though aggregate ECE was slightly lower. Thus the
maximum-supported-bin calibration gate failed in both periods.

The q3 top-three recall itself was `0.818004` in 2025 and `0.812296` in 2023.
Those values were non-lower than q1 as required, but the increments were too
small to rescue the failed proper-score and calibration gates.

## q1 partial decision

Destination conditioning produced a real but sub-threshold average increment
over frozen state-only timing:

| Period | Pooled log-loss improvement | Required | Better cycles | Top-three recall gain | Required |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025 | 0.4133% | 0.5000% | 16/20 | 0.0824 pp | 0.5000 pp |
| 2023 | 0.4527% | 0.5000% | 18/20 | 0.1394 pp | 0.5000 pp |

Its daily intervals, all horizons, all quarters, every stock deletion,
per-cycle count, and calibration gates passed in both periods. However, the
predeclared minimum log-loss and ranking improvements failed twice. The
contract's simpler-model retention clause therefore does not activate; q1 is
not retained.

## Path-only sanity failure

At 24 bars, q3 was worse than the retained path-only probability on log loss
in both periods:

| Period | Model | 24-bar log loss | 24-bar Brier |
| --- | --- | ---: | ---: |
| 2025 | path only | 0.213510 | 0.056983 |
| 2025 | q3 joint timing | 0.213790 | 0.057005 |
| 2023 | path only | 0.210338 | 0.055983 |
| 2023 | q3 joint timing | 0.210520 | 0.055963 |

In 2025 q3 was also worse on 24-bar Brier; in 2023 Brier was slightly better,
but the contract required both losses to be lower at every horizon. Top-three
recall improved over path-only by only 0.4348 percentage points in 2025 and
0.2946 percentage points in 2023, below the required 0.5000 percentage points.
The path-only sanity gate therefore failed in both periods. As in the earlier
factorised experiment, long-horizon timing remains the decisive weakness.

## q2 diagnostic boundary

The order-2 kernel was frozen as a diagnostic, not a selectable model. Its
pooled 2025 log loss was `0.173501030273` versus q3's `0.173501520252`, and its
pooled Brier was `0.044814456528` versus q3's `0.044816318272`. In 2023 its
pooled log loss was `0.172910431871` versus q3's `0.172901357892`, while pooled
Brier was `0.044644461626` versus `0.044644402711`.

These differences are tiny and mixed across periods. More importantly, q2
had no predeclared primary comparisons or retention gates. It cannot be
promoted after observing q3's rejection. Any q2 investigation would require a
new, separately frozen development contract and later prospective evidence.

## Validation and interpretation

The implementation self-tests passed for synthetic convolution, destination/
duration coupling, and dynamic-program/brute-force agreement. The destination
marginal remained the retained history model exactly. No 2026 row or provider
price, `historical_volume`, direction, return, range, P&L, spread, cost, order,
broker, position, strategy-promotion, or deployment field entered this
experiment.

The independent post-score audit passed all 46 checks. It reconstructed 74,398
2025 anchors and 1,392,537 scored rows, plus 73,071 backward-2023 anchors and
1,382,895 scored rows. Metadata, completion labels, eventual-loop labels,
completion times, and all five stored model probabilities had zero mismatches;
kernel counts and PMFs matched exactly. Aggregate metrics, calibration,
top-three ranking, per-cycle results, comparisons, and support reproduced to
less than `1e-16`, and every frozen gate reproduced exactly. The protected
pre-score, post-score, and current shadow snapshots plus all 18 protected
bundle hashes matched; `outcomes_opened` remained false. The audit independently
reproduced both rejection decisions: q3 rejected and q1 not retained.

The frozen prospective movement shadow tree was unchanged before and after
the experiment. Its prediction ledger remained empty, no runtime outcome was
opened, and this completion experiment neither adds to nor validates the
prospective movement hypothesis.

2025 is still a development period. Backward-2023 still uses future-fitted
2024 structural parameters and is only a portability/falsification period.
Neither is an untouched prospective validation set. The rejection concerns
the tested q3 and q1 completion specifications; it does not erase the retained
last-three-state loop-identity result, and it does not support price direction,
economic edge, tradability, or live use.

## Final disposition

- Keep frozen: the eight-state causal regime detector, calibrated three-bar
  state-departure component, and retained last-three-state fixed-loop identity
  probabilities.
- Reject: q3 full-history/destination-conditioned completion timing.
- Do not retain: q1 destination-conditioned completion timing.
- Do not promote: q2, which remains diagnostic only.
- Preserve unchanged: the offline prospective movement shadow contract,
  ledger, and frozen bundle.

Safety result: research-only; live ordering disabled; order placement
disabled; no deployment or strategy promotion.
