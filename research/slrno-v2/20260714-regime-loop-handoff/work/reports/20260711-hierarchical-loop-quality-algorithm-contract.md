# Hierarchical loop-quality algorithm V1 — frozen contract

## Decision

The research-only algorithm is now specified and frozen before fitting. It is
a single multinomial movement-quality model that combines causal state/context,
the existing semantic route-topology representation, a partially pooled cycle
effect, and a more strongly pooled compatible cycle/current-state effect.

The corrected contract SHA-256 is
`f6956b6ab0495a49669f714df834d1fd0fdaa13b0ecf4b123d6c54c0fc9b5936`.

An independent pre-implementation review rejected the first draft before any
fit because several audit formulas were underspecified. This corrected freeze
defines within-cycle route centering, exact resampling and calibration
semantics, strict orientation support, named-hypothesis intersection tests,
and falsification multiplicity. No model or outcome evaluation occurred under
the rejected draft.

No fit, prediction, result, grade change, later-period outcome load, or shadow
operation is reported by this document. The user's instructions to use an
algorithm and continue authorize the contract's narrowly bounded research run;
they do not authorize any live or ordering operation.

`research_only: true`, `live_ordering_enabled: false`, and
`order_placement: disabled`.

This experiment predicts only the probabilities that absolute return or future
range exceeds frozen 2024 P75/P90 thresholds conditional on a frozen loop
occurring. It does not predict direction, signed return, P&L, economic edge,
tradability, or trading performance.

## Why this algorithm

V3 showed that semantic route topology retained most of the full loop model's
average proper-loss gain, while explicit cycle identity, cycle/current-state,
and the large history-token block added little on average. The remaining
problem was heterogeneous calibration: pooled gains repeated, but some
supported orientations reversed.

V1 therefore keeps the compact topology block and adds only two centered,
regularized categorical levels:

- a twenty-cycle deviation from shared context and topology; and
- a forty-four-unit compatible cycle/current-state deviation from its pooled
  representation.

Both blocks use the same ridge penalty supplied by multinomial logistic
regression. Their feature scales determine shrinkage. The scale pair is chosen
causally and jointly for all six target/horizon tasks, so a thin route cannot
receive its own post-hoc model or hyperparameter.

## Exact representation

Every nonzero model has 144 columns:

| Block | Width | Construction |
| --- | ---: | --- |
| Causal context | 17 | Frozen eight-state one-hot and nine causal numeric controls, using the training-fold imputer/scaler |
| V3 route topology | 63 | Exact frozen compatible rotations, uniform aggregation, transition length, centroid expectations/deltas, and ambiguity features |
| Centered cycle | 20 | Cycle one-hot minus its realized-training-fold inverse-overlap-weighted mean, multiplied by `a_cycle` |
| Centered oriented route | 44 | Within the row's cycle, compatible cycle/current-state one-hot minus that cycle's realized-training-fold inverse-overlap-weighted route proportions, multiplied by `a_route`; every coordinate belonging to another cycle is zero |

Cycle indicators are centered globally across the twenty cycles. Route
indicators are centered within their parent cycle, not globally across all 44
units. Validation and scoring rows reuse the training fold's centering vectors;
they are never recentered. A training fold with zero total weight or zero
weight for any frozen cycle stops. The 44-column order is pinned by the V3
route map. A row outside that map stops the run.

The model is one three-class multinomial logistic regression per target and
horizon: `C=0.2`, `lbfgs`, `max_iter=2000`, `tol=1e-10`, seed `20260711`, and
temperature `1.0`. The ordered outputs remain
`q75=P(class_1)+P(class_2)` and `q90=P(class_2)`, with exact nesting required.

The algorithm does not add a history token, realized rotation, future state,
duration, stock identity, or new row-level volume feature. Two coordinates in
the frozen semantic centroids originated from provider `historical_volume` in
the original detector. That is neither exchange-wide volume nor order flow.

## Literal fifteen-pair grid

The following ordered list is normative; it is not to be regenerated from an
informal set expression:

```text
(0, 0)
(.125, .0625)  (.125, .125)
(.25, .0625)   (.25, .125)   (.25, .25)
(.5, .0625)    (.5, .125)    (.5, .25)    (.5, .5)
(1, .0625)     (1, .125)     (1, .25)     (1, .5)     (1, 1)
```

The pair is `(a_cycle, a_route)`. The zero endpoint is not refitted with V1's
tighter tolerance. For July-December it is the exact sealed V3 causal
`qroute_topology` probability. April-June zero-endpoint predictions are
regenerated causally with the exact V3 settings. If zero is selected for the
full fit, V1 falls back to the exact V3 full topology model and stores no new
hierarchical coefficients.

## Causal selection

For each July-December outer month, pair selection uses only three earlier
inner validation months:

| Outer month | Inner validation months |
| --- | --- |
| July | April, May, June |
| August | May, June, July |
| September | June, July, August |
| October | July, August, September |
| November | August, September, October |
| December | September, October, November |

Each inner prediction is itself generated by a fit using only months strictly
before that inner validation month. The objective is the equal mean of twelve
conditional binary log-loss cells: two outcomes, three horizons, and P75/P90.
One pair is selected for all six ordered models. Every pair within `1e-6` of
the minimum enters the tie set; smaller `a_route`, then smaller `a_cycle`, wins.

The final full-2024 pair is selected from causal October, November, and
December candidate predictions using the same criterion and tie rule. A
nonzero selected pair is then fitted on all eligible 2024 realized-loop rows.

## Frozen evaluation gates

The primary family compares the hierarchical model with both frozen context
and V3 topology on conditional and joint surfaces, using log loss and Brier
score. The eight endpoints use common 20,000-draw, five-session moving-block
resamples and Bonferroni one-sided 99.375% upper bounds.

Minimum pooled relative log-loss improvements are:

| Comparison | Conditional | Joint |
| --- | ---: | ---: |
| Hierarchical versus context | 0.50% | 0.25% |
| Hierarchical versus topology | 0.10% | 0.05% |

Every corresponding Brier difference and block upper bound must be below zero.
Every required quarter and leave-one-stock-out deletion must be no worse; both
targets and all horizons must be no worse; and no individual one of the twelve
cells may degrade relative log loss by more than 0.25%.

The common bootstrap orders sessions by date and forms each daily endpoint as
the arithmetic mean of its available finite within-session cell-weighted
differences; unavailable cells are NaN, not zero. Every draw samples exactly
`ceil(n/L)` starts from all overlapping blocks of length `L=min(5,n)` using one
common PCG64 block-start matrix, concatenates, trims to `n`, and uses linear
quantiles. An endpoint or resample without a finite value stops. Secondary and
named tests reuse that exact calendar and block-start matrix. Context and
topology probabilities must replay sealed V3 values within `1e-12`.

Calibration is deliberately stricter than in V3. Fixed-bin index is
`min(floor(10p),9)`. ECE uses every nonempty bin with conditional overlap
weights or joint unit weights; maximum bin error uses only bins meeting the
raw-row threshold, and no supported bin is a failure. In every cell, hierarchical
ECE must be no greater than both context ECE and topology ECE. Supported-bin
absolute error may not exceed 0.02 conditionally or 0.01 jointly. Every
supported cycle/current-state, rotation-count, and frozen entropy-quartile
slice must have nonpositive hierarchical-minus-context and
hierarchical-minus-topology log-loss and Brier differences on both surfaces.
One supported reversal fails the algorithm gate.

The secondary comparison with sealed `qfull` uses a noninferiority margin equal
to ten percent of each positive `qfull`-versus-context gain. Its four endpoints
use one-sided 98.75% bounds. It cannot rescue a failed primary result.

The fixed falsification test evaluates 999 outcome-label circular shifts or
permutations against the frozen OOF predictions, without 999 model refits. The
complete twelve-label vector is moved at the unique-anchor level within stock
and quarter under draw-major, then lexicographic-stratum-major traversal; every stratum must contain at least two
unique anchors. The four relative log-loss improvements are context/topology
by conditional/joint. Each empirical one-sided p-value must be no greater than
0.01, stricter than the four-test Bonferroni threshold of 0.0125.

## Named loops remain harder to qualify

Named testing occurs only if the global primary algorithm gate passes. All
twenty cycles and all three horizons are then assessed under the exact V1
support, structural-reliability, P75/P90 rate, mean-probability, lift,
proper-loss, quarter, stock-deletion, two-target, calibration, horizon, and
global-grade semantics. V1 adds:

- Holm familywise alpha 0.025 across sixty cycle/horizon good hypotheses;
- a separate Holm familywise alpha 0.025 across sixty high hypotheses;
- high gatekeeping by good; and
- mandatory support and non-reversal for every compatible orientation of a
  named cycle.

Each cycle/horizon/tier unit is the intersection of exactly ten component
tests: conditional log loss, conditional Brier, joint log loss, joint Brier,
and conditional context-residual lift for each of two targets. Component
p-values use centered-null, common-block, 20,000-draw tests; the unit p-value is
their maximum. Holm ordering is stable by p-value, cycle index, then horizon,
and stops at the first failure. A failed good unit receives high-family
p-value one before the separate high Holm procedure. Every one of the 44
pinned cycle/state units belonging to a cycle is a required orientation; no
frequency-selected subset is permitted.

In these named gates, causal `qhier75/qhier90` replace only the V1 candidate
`qcycle75/qcycle90`, and joint candidate probabilities become frozen
`s*qhier`. Context probabilities, structural probabilities, first-order
structural baseline, targets, thresholds, weights, and support cohorts remain
sealed. Conditional lift remains the weighted daily mean of `y-qcontext`.

Opened-period outputs may therefore be called only
`development_good_candidate`, `development_high_candidate`, or
`development_unqualified`. Existing frozen parent grades remain unchanged.
2025 and backward-2023 can only demote a 2024 development candidate; they
cannot create, substitute, or promote one. None of these labels is prospective.

## Audit and execution lock

The fit-only bundle must contain the frozen grid, fold schedule, feature map,
training-fold centers/scalers, all inner selections, outer predictions, full
parameters, support, calibration, rotation, falsification, algorithm, and
candidate diagnostics. Those artifacts and all source hashes must be frozen
before any later-period row-level outcome is loaded.

An independent implementation must reconstruct every feature, fit, selection,
probability, bootstrap, calibration value, slice decision, falsification,
multiplicity adjustment, and label without importing production runner, model,
metric, gate, or grade code. Only a complete passing pre-score audit may set
`scoring_authorized: true`. A second independent audit must reconstruct the
2025 and backward-2023 scoring artifacts.

The contract fails closed on a hash mismatch, causal-fold violation, missing
class, support failure, nonconvergence, nonfinite or nonnested probability,
unsupported required orientation, calibration failure, or sign reversal. No
post-result feature, scale, fold, threshold, gate, or label revision is
permitted under this contract.

No 2026, prospective-shadow, live, demo, paper, broker, order, position, P&L,
deployment, runtime, or strategy path may be read or changed.

## Files and ephemeral artifacts

- Contract: `work/contracts/20260711-hierarchical-loop-quality-algorithm-v1.json`
- Planned runner: `work/run_hierarchical_loop_quality_algorithm_v1.py`
- Planned independent audit: `work/audit_hierarchical_loop_quality_algorithm_v1.py`
- Planned artifacts: `/private/tmp/stocker_hierarchical_loop_quality_algorithm_v1_20260711`

The `/private/tmp` bundle will be ephemeral and must be archived separately
before reboot if exact replay without recomputation is required.
