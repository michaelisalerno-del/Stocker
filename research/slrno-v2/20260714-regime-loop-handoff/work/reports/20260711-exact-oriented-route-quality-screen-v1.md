# Exact oriented-route quality screen — 2024 development result

Status: **complete; zero qualifying route-horizon candidates; independent audit passed**

`research_only: true`  
`live_ordering_enabled: false`  
`order_placement: disabled`

## Decision

The frozen 2024 causal OOF screen did not narrow the current cycle dictionary to any route-horizon that can be advanced under the predeclared requirements.

All 135 exact route × horizon units received:

> `development_unqualified`

The frozen decision is:

> `no_exact_route_quality_screen_candidate`

No gate was weakened, no closest route was selected, and no second-stage refinement was performed. No 2025, backward-2023, partial-2026, or prospective-shadow path was resolved or read.

This experiment concerns absolute movement magnitude and future range only. It does not test or support direction, signed return, profitability, P&L, economic edge, tradability, strategy, order, position, or deployment claims.

## Test performed

The twenty frozen rotation-invariant cycles were expanded into 45 exact directed routes:

- 43 parent cycle/current-state units already represented one exact route;
- the ambiguous `cycle_15@state_1` union was split into `1->2->1->3->1` and `1->3->1->2->1`;
- 221,894 compatible exact-route rows and 15,584 realised exact-route rows were evaluated in causal July–December 2024 OOF predictions;
- the full expanding-prefix training cohort contained 32,677 realised exact-route rows.

The split was exhaustive, exclusive, and weight-preserving. It retained all 15,584 parent realised rows, assigned every positive cycle-15 union row to exactly one child, and reproduced every anchor's parent inverse-overlap weight with maximum error `0.0`.

The new `qexact` ordered movement model used:

- the frozen eight-state one-hot entry regime;
- the same nine causal entry controls as the parent quality model;
- one 45-column exact-route block with scale `0.5`;
- expanding-month fits using only strictly earlier 2024 months;
- raw probabilities without temperature or post-hoc recalibration.

Provider volume was not used: `historical_volume_not_used`.

## What survived in aggregate

Exact route identity retained meaningful pooled movement information versus current-state/context alone. Across the twelve target × horizon × tier cells, conditional qexact log-loss improvement over `qcontext` ranged from `1.8614%` to `7.6176%`, and Brier score improved in every cell.

It did not improve the richer frozen cycle/history quality model. Relative qexact log-loss improvement versus the parent `qcycle` ranged from `-0.4066%` to `+0.1738%`; qexact was slightly worse in nine of twelve cells. The exact-route block is therefore a useful compression of broad movement information, but it does not add a robust new source beyond the existing cycle/current-state/history representation.

## Qualification funnel

| Gate stage | Passing units |
| --- | ---: |
| Exact routes with base support | 21 / 45 |
| Routes with measurable exact structural probability | 43 / 45 |
| Structurally reliable routes | 32 / 45 |
| Evaluated target/horizon/tier cells passing strict calibration plus qcycle non-inferiority | 14 / 252 |
| Route-horizons passing both P75 targets | 0 / 135 |
| Final development good/high screen candidates | 0 / 135 |

The Holm families contained 126 supported target-route-horizon tests at each tier. Several individual loss improvements survived multiplicity, but no route-horizon passed both targets together with strict calibration and non-inferiority to the stronger parent model.

## The `3->6->3` result

The previously interesting `cycle_09@state_3` route again showed large six-bar movement uplift over context:

| Six-bar target | Observed P75 rate | Mean qexact P75 | Mean qcontext P75 | Conditional LL improvement | Joint LL improvement | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Absolute return | 46.42% | 42.18% | 16.80% | 25.29% | 11.37% | 0.0126 |
| Future range | 33.21% | 32.29% | 12.89% | 18.50% | 10.56% | 0.0456 |

It still failed the frozen exact-route screen:

- conditional maximum supported-bin error was `4.47%` for absolute return and `2.68%` for future range, above the `2%` limit;
- qexact ECE was worse than the richer parent qcycle probability for both targets;
- absolute-return qexact log loss was `0.375%` worse than qcycle, outside the `0.25%` non-inferiority margin;
- joint absolute-return calibration error was `1.36%`, above the `1%` limit;
- six-bar future-range P90 had only 21 positives, below the frozen support minimum of 25.

Thus `3->6->3@H6` remains an explanatory research observation, not a good/high candidate.

## Cycle-15 exact split

The two formerly combined paths contained 141 and 125 realised OOF rows, respectively, below the 250-row route support minimum. Their movement rates were generally low, and neither has an exact frozen structural occurrence probability because the retained loop model outputs only their union probability.

Both split paths therefore failed closed and cannot advance. The split did remove an ambiguity, but it did not reveal a hidden movement-rich candidate.

## Interpretation

The test answers the proposed narrowing question negatively under the frozen evidence standard:

- exact directed routes explain movement better than state/context on average;
- most of that information was already represented by the richer frozen cycle/history quality model;
- support, route-level calibration, and two-target consistency do not align for the same route and horizon;
- there is no defensible subset to narrow further in this experiment.

Choosing the largest observed uplift or relaxing calibration now would be post-result selection. A genuinely different next experiment would need a new candidate-generating principle, not another filter applied to these same 135 opened grades.

## Integrity and audit

- Contract SHA-256: `858b02722ba1a4f6fe487977971510209edbcb3e9fc2f8eaf93034e1ef50bed2`.
- Runner SHA-256: `062ceec9557b11893b8ded9d73277977a71897cd86e67d32f78cbb5eacdd7b6e`.
- Auditor SHA-256: `1f5619453ab7da8da9284077a85e32910c1ff3bbcc30ded54a5f7c257dfbe581`.
- Independent audit-result SHA-256: `a9ba59883c8c0b9d311b5f2ee3909eb4b65cf2af726deb384d6b8dadf1d6154b`.
- Independent audit: 17/17 checks passed.
- All 36 causal qexact model predictions replayed with maximum error `0.0`.
- Focused screen and audit tests: 13/13 passed.
- Full workspace research suite: 204/204 passed.
- Later-period paths resolved/read: false/false.
- Shadow tree read/written: false/false.
- Parent grades changed: no.
- Certified good/high routes: zero.

Files:

- Contract: `work/contracts/20260711-exact-oriented-route-quality-screen-v1.json`
- Runner: `work/run_exact_oriented_route_quality_screen_v1.py`
- Auditor: `work/audit_exact_oriented_route_quality_screen_v1.py`
- Tests: `work/tests/test_exact_oriented_route_quality_screen_v1.py` and `work/tests/test_exact_oriented_route_quality_screen_v1_audit.py`

The row-level artifacts are under:

`/private/tmp/stocker_exact_oriented_route_quality_screen_v1_20260711`

That directory is ephemeral and should be archived before reboot if exact replay without recomputation is required.
