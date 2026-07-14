# Oriented-route movement diagnostic

## Decision

Splitting each frozen cycle by its causal current-state rotation explains much
of the apparent contradiction between useful pooled `qcycle` predictions and
the failure of named loops to qualify. The key distinction is between:

- high movement already implied by the current regime state; and
- additional movement information supplied by the route that leaves that
  state and returns.

The split is meaningful, but it does not rescue a certified loop. Of 44
`(cycle_id, current_state)` units, 17 had base support in all three tested
periods. Only `cycle_09@state_3`, the exact route `3->6->3`, received a
non-failed horizon grade: `diagnostic_good_candidate` at six bars in causal
2024 OOF. It did not repeat in 2025 or backward-2023, and it failed at 12 and
24 bars. Every cross-period route/horizon and global route grade was therefore
either `diagnostic_unqualified` or `diagnostic_not_supported`.

These are post-outcome exploratory labels. They cannot promote a cycle, enter
either shadow, establish prospective validity, or support a direction,
profitability, P&L, economic-edge, tradability, or trading-performance claim.

`research_only: true`, `live_ordering_enabled: false`, and
`order_placement: disabled`.

## What was evaluated

The diagnostic reused the exact frozen 2024 OOF, 2025 development, and
backward-2023 rows and probabilities. It did not refit the eight-state
detector, structural loop model, conditional quality model, temperature,
threshold, cycle dictionary, or any probability.

For each supported oriented route, it reapplied the parent experiment's:

- base and endpoint support counts;
- absolute-return and future-range P75/P90 rate thresholds;
- conditional `qcycle` versus `qcontext` log-loss and Brier gates;
- exact joint `s*qcycle` versus `s*qcontext` gates;
- five-session, 5,000-draw daily block intervals;
- every required quarter and every leave-one-stock-out deletion;
- fixed-bin calibration gates; and
- route-specific structural history-versus-first-order gates.

The base support gate passed 22 routes in 2024 OOF, 21 in 2025, and 21 in
backward-2023. Seventeen routes passed it in all three periods. The diagnostic
evaluated 768 supported route/target/horizon/tier cells; seventeen individual
cells passed every applicable gate, but only two P75 target cells aligned at
the same route, period, and horizon to produce one good horizon grade.

## Frozen topology map

The cycles are rotation invariant, but their oriented diagnostic units are
not. All routes below are exact except the two-path union at
`cycle_15@state_1`.

| Cycle | Length | Current-state route(s) |
| --- | ---: | --- |
| 01 | 2 | `1: 1->3->1`; `3: 3->1->3` |
| 02 | 2 | `1: 1->2->1`; `2: 2->1->2` |
| 03 | 2 | `0: 0->1->0`; `1: 1->0->1` |
| 04 | 2 | `2: 2->4->2`; `4: 4->2->4` |
| 05 | 2 | `0: 0->3->0`; `3: 3->0->3` |
| 06 | 2 | `4: 4->6->4`; `6: 6->4->6` |
| 07 | 2 | `5: 5->6->5`; `6: 6->5->6` |
| 08 | 2 | `3: 3->4->3`; `4: 4->3->4` |
| 09 | 2 | `3: 3->6->3`; `6: 6->3->6` |
| 10 | 2 | `1: 1->4->1`; `4: 4->1->4` |
| 11 | 2 | `2: 2->5->2`; `5: 5->2->5` |
| 12 | 4 | `0: 0->1->0->1->0`; `1: 1->0->1->0->1` |
| 13 | 2 | `5: 5->7->5`; `7: 7->5->7` |
| 14 | 4 | `1: 1->3->1->3->1`; `3: 3->1->3->1->3` |
| 15 | 4 | `1: 1->2->1->3->1 OR 1->3->1->2->1`; `2: 2->1->3->1->2`; `3: 3->1->2->1->3` |
| 16 | 3 | `1: 1->2->3->1`; `2: 2->3->1->2`; `3: 3->1->2->3` |
| 17 | 4 | `1: 1->2->1->2->1`; `2: 2->1->2->1->2` |
| 18 | 3 | `0: 0->3->1->0`; `1: 1->0->3->1`; `3: 3->1->0->3` |
| 19 | 3 | `0: 0->1->3->0`; `1: 1->3->0->1`; `3: 3->0->1->3` |
| 20 | 2 | `2: 2->3->2`; `3: 3->2->3` |

## The state ladder explains most raw movement

The frozen standardized state centroids form a clear activity/range ladder:

| State | 12-bar activity centroid | Bar-range centroid |
| ---: | ---: | ---: |
| 0 | -0.856 | -1.223 |
| 1 | -0.486 | -0.505 |
| 2 | -0.238 | -0.336 |
| 3 | -0.214 | -0.164 |
| 4 | 0.281 | 0.272 |
| 5 | 0.587 | 0.550 |
| 6 | 0.709 | 0.618 |
| 7 | 1.057 | 0.981 |

Across the twenty cycles, mean P75 movement rate had Spearman correlation
0.925 with the largest member-state bar-range centroid. Across oriented
routes, raw P75 rate correlated 0.752 with the current-state bar-range
centroid. By contrast, the `qcycle/qcontext` probability ratio correlated
0.455 with the upward range-centroid contrast between the current state and
the other state(s) visited by the route.

This supports a bounded interpretation:

- current-state intensity largely determines how much movement occurs; and
- route identity is most useful when it says that a lower-intensity current
  state is about to visit a higher-intensity state and return.

The quality diagnostic itself used no volume feature. The frozen regime
detector's original emissions include provider `historical_volume`; that is
not exchange-wide volume or order flow.

## Focus-route evidence

P75 rates below are means across both targets and all three horizons within
each period. `q ratio` is mean calibrated `qcycle/qcontext` on realized-loop
rows. Structural pass is the strict route-specific loss/calibration result.

| Exact route | 2024 OOF rows / P75 / q ratio / structural | 2025 rows / P75 / q ratio / structural | 2023 rows / P75 / q ratio / structural |
| --- | --- | --- | --- |
| `4->6->4` | 594 / 0.589 / 1.222 / pass | 934 / 0.548 / 1.217 / pass | 859 / 0.522 / 1.255 / fail |
| `6->4->6` | 544 / 0.580 / 0.951 / pass | 822 / 0.561 / 0.945 / fail | 727 / 0.524 / 0.950 / fail |
| `5->6->5` | 756 / 0.600 / 1.061 / pass | 1,807 / 0.589 / 1.046 / pass | 1,347 / 0.610 / 1.040 / pass |
| `6->5->6` | 637 / 0.603 / 1.092 / fail | 1,505 / 0.599 / 1.088 / pass | 1,119 / 0.620 / 1.084 / pass |
| `3->6->3` | 545 / 0.372 / 2.379 / pass | 1,065 / 0.328 / 2.426 / pass | 850 / 0.305 / 2.448 / pass |
| `6->3->6` | 339 / 0.419 / 0.981 / pass | 666 / 0.398 / 0.968 / pass | 587 / 0.371 / 0.970 / pass |
| `5->7->5` | 460 / 0.613 / 1.170 / fail | 995 / 0.610 / 1.174 / fail | 708 / 0.706 / 1.152 / fail |
| `7->5->7` | 513 / 0.700 / 0.998 / pass | 919 / 0.660 / 1.013 / pass, unsupported | 694 / 0.770 / 0.999 / pass, unsupported |

Three different phenomena are visible:

1. `3->6->3` is the clearest incremental route. Its P75 `qcycle/qcontext`
   ratio was 2.38-2.45 across periods, while the reverse `6->3->6` ratio was
   about 0.97. The lower-state start carries path information; the high-state
   start already carries the movement information in `qcontext`.
2. `5->6->5` and `6->5->6` are consistently movement rich, but both states
   are high-intensity and P75 probabilities rise little beyond context. This
   explains how cycle 07 can have robust raw absolute P75/P90 rates without
   becoming a loop-specific quality effect.
3. `5->7->5` combines high raw rates and moderate incremental probability,
   but its structural probability is not calibrated reliably and its
   conditional loss advantage is not stable across daily, quarter, and stock
   slices. The reverse `7->5->7` is extremely movement rich but is almost
   entirely state-7 context and lacks the frozen compatible-row support
   required in 2025 and 2023.

`4->6->4` is another plausible oriented hypothesis: it combines high raw
movement and an incremental probability ratio above 1.21. It nevertheless
failed the 2023 structural gate and its joint proper-loss advantage did not
survive the later periods.

## Cycle 09 and cycle 13 under the exact gates

For `3->6->3`, four P75 cells passed in 2024 OOF: absolute return at 6, 12,
and 24 bars and future range at 6 bars. Only the two six-bar targets aligned,
producing the sole `diagnostic_good_candidate` horizon grade. In 2025 only
12-bar absolute-return P75 passed. No cell passed in backward-2023. P90 rates
and mean P90 probabilities were too low, and range stability weakened outside
the short 2024 OOF window.

For `5->7->5`, three P75 and four P90 cells passed, all in backward-2023. Its
high raw rate is real descriptively, but the route failed the strict
structural gate in every period. At P75, conditional daily or quarter/stock
robustness failed in eleven of eighteen evaluated cells. At P90 those
conditional robustness failures appeared in twelve of eighteen cells. It is
therefore a period-sensitive hypothesis, not a qualified route.

## Why pooled `qcycle` still improves

The pooled representation is solving a broader probability problem than
"find a high-movement named loop." Conditional log-loss improvement remained
positive for every target/horizon/tier cell in all periods. It was strongest
at short horizons and decayed with horizon:

| Period | Absolute P75: H6 / H12 / H24 | Range P75: H6 / H12 / H24 |
| --- | --- | --- |
| 2024 OOF | 5.795% / 3.626% / 2.472% | 7.842% / 6.405% / 3.970% |
| 2025 | 4.420% / 3.707% / 1.827% | 7.782% / 5.637% / 3.541% |
| Backward-2023 | 5.131% / 3.444% / 2.104% | 8.632% / 6.071% / 3.811% |

Cycle 09 supplied 31.9%, 23.2%, and 16.2% of summed conditional loss-gain
mass in 2024 OOF, 2025, and backward-2023 respectively. It is a genuine driver
of the pooled improvement. But pooled loss also rewards correctly lowering
movement probabilities for quiet cycles. For example, cycles 11, 16, 15, 17,
and 12 often improve loss by downshifting `qcycle` far below `qcontext`; that
does not make them high-movement loops. Cycle 13 supplied only 1.7% and 2.6%
of conditional gain mass in 2024 OOF and 2025, rising to 11.4% in
backward-2023, another sign of period dependence.

## Length and repeated-state motifs

Mean per-cycle P75 rates were 0.267/0.250/0.254 for length-two cycles in 2024
OOF/2025/2023, versus 0.039/0.034/0.046 for length-three cycles and
0.028/0.027/0.026 for length-four cycles. This is partly confounded: all
cycles reaching states 5-7 are length two.

The exact repeated-pair comparisons point in the same direction without that
particular topology change:

- one `0<->1` revolution exceeded the two-revolution motif's P75 rate by
  0.022, 0.018, and 0.020;
- one `1<->3` revolution exceeded two revolutions by 0.028, 0.021, and 0.022;
- one `1<->2` revolution exceeded two revolutions by 0.040, 0.036, and 0.028.

The longer repeated motifs are structurally predictable but quiet. All four
length-four cycles passed the structural gate in every tested period, although
cycle 12 later lacked support. The three length-three triangles are the
opposite: they have weak structural improvement and none passed the structural
gate consistently. Structural recurrence and movement quality are therefore
separate properties.

## Structural reliability remains separate

At the original cycle level, `cycle_09` was structurally reliable in every
period: occurrence rates were 8.3%, 7.9%, and 7.4%, and history-path log-loss
improvements over first order were 14.5%, 12.2%, and 12.6%. Both oriented
cycle-09 routes also passed the route-specific structural gate throughout.

`cycle_13` had larger structural discrimination and log-loss improvement, but
its calibrated probability was less reliable. At `5->7->5`, frozen history
AUC was 0.879/0.816/0.834 in 2024 OOF/2025/2023, while strict ECE/maximum-bin
conditions failed each period. High discrimination is not the same as a
calibrated occurrence probability.

Across all cycles, the Spearman correlation between structural log-loss
improvement and P75 movement rate was only 0.374. A loop can be recurrent and
forecastable without being movement rich, or movement rich without adding
information beyond the current regime.

## Candidate hypotheses for a separately frozen V2

These candidates were selected after inspecting outcomes and must be tested
under a new nested-2024 development contract before any later data is read:

1. Treat the oriented route, not a rotation-invariant cycle, as the movement
   evaluation unit. Predeclare `4->6->4`, `3->6->3`, and `5->7->5`, with their
   reverse rotations retained as mandatory matched controls.
2. Keep two descriptive axes separate: `movement-rich given current state`
   and `incremental route uplift versus qcontext`. A final good/high label may
   still require both, but failure on one axis should explain why rather than
   obscure the surviving information.
3. Test a hierarchical topology representation based on current-state
   intensity, maximum visited-state intensity, and upward intensity contrast.
   It should compete against—not supplement after inspection—the frozen
   cycle/current-state interaction.
4. Predeclare horizon-specific hypotheses. The pooled gain and the only good
   oriented grade were strongest at six bars. Do not retroactively weaken the
   existing all-horizon global grade.
5. Require nested causal 2024 selection/calibration, explicit quarter gates,
   all-stock-deletion gates, and a stop if the oriented effect is merely a
   high-state context effect. Do not reuse 2025 or 2023 as fresh validation.

The rejected state-only duration product and rejected joint semi-Markov
completion kernel should remain closed. This diagnostic concerns route
identity and movement magnitude, not full-loop completion timing.

## Integrity and artifacts

- Contract:
  `work/contracts/20260710-oriented-route-movement-diagnostic-v1.json`.
- Runner:
  `work/run_oriented_route_movement_diagnostic.py`.
- Tests:
  `work/tests/test_oriented_route_movement_diagnostic.py`.
- Ephemeral artifacts:
  `/private/tmp/stocker_oriented_route_movement_diagnostic_20260710`.
- Contract SHA-256:
  `d8288969b26b1d314c4d7762fba3bd8fed4dc01c935100d63780e88b8f2c8f12`.
- Frozen input/source manifest SHA-256:
  `828984e02c5b767ffd0a24123b44869739559ae83df01e2c4e1d5d3ac822b973`.
- Tests: 11/11 passed.
- No model was refitted and no threshold was changed.
- Both protected shadow trees were byte-for-byte identical before and after;
  both ledgers remained empty and both outcome flags remained unopened.

The large diagnostic artifacts are under `/private/tmp` and may not survive a
reboot.
