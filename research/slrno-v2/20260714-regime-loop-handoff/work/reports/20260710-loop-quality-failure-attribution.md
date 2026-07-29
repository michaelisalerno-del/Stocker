# Frozen loop-quality failure attribution

## Decision

The earlier all-unqualified result remains correct, but the read-only failure
attribution sharpens its interpretation. Absolute movement levels associated
with the cycles are highly stable across periods. What fails is the stronger
claim that the same named loop supplies stable, incremental information beyond
the causal state/context controls across both targets, all three horizons, and
every robustness slice.

No model was fitted, recalibrated, or retuned. No frozen grade changed, and no
prospective shadow was read or written. All twenty final grades remain
`unqualified`.

`research_only: true`, `live_ordering_enabled: false`, and
`order_placement: disabled`.

These diagnostics concern absolute movement magnitude and future range only.
They do not concern direction, signed return, P&L, economic edge, tradability,
orders, or deployment.

## Read-only diagnostic package

`work/run_loop_quality_failure_diagnostics.py` reads only the sealed artifact
tree at:

`/private/tmp/stocker_per_loop_movement_quality_20260710`

It writes a separate diagnostic bundle to:

`/private/tmp/stocker_loop_quality_failure_diagnostics_20260710`

The runner hashes every required frozen input before and after analysis. All
fifteen hashes matched exactly. It has no fitting-library dependency, does not
call a model fit, and restricts output to a dedicated top-level `/private/tmp`
diagnostic directory.

The package contains:

- frozen gate-family pass counts;
- a twenty-cycle decomposition;
- cross-period Pearson and Spearman correlations on matched cells;
- a two-axis movement-level versus incremental-evidence table;
- raw-rate quarter and leave-one-stock-out reconstructions;
- structural-reliability attribution;
- focused diagnostics for cycles 06, 07, 09, and 13;
- a machine-readable summary with input and output hashes.

## Gate-family attribution

Each row below contains 120 cycle-target-horizon cells. `Rate level` means that
both the observed rate and mean frozen `qcycle` cleared the applicable absolute
threshold. `Rate/context` is the required ratio over `qcontext`; `residual CI`
requires its moving-block lower bound to be positive.

| Period/tier | Full | Event support | Rate level | Rate/context | Residual CI | Conditional core | Conditional daily CI | Conditional robustness | Conditional calibration | Joint core | Joint daily CI | Joint robustness | Joint calibration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 OOF P75 | 6 | 63 | 24 | 19 | 15 | 104 | 45 | 91 | 108 | 97 | 61 | 85 | 90 |
| 2025 P75 | 1 | 60 | 24 | 7 | 7 | 109 | 50 | 74 | 113 | 104 | 75 | 73 | 95 |
| 2023 P75 | 6 | 58 | 23 | 15 | 17 | 110 | 60 | 80 | 112 | 94 | 70 | 58 | 85 |
| 2024 OOF P90 | 0 | 43 | 19 | 7 | 5 | 96 | 45 | 85 | 103 | 110 | 65 | 95 | 108 |
| 2025 P90 | 0 | 42 | 19 | 3 | 1 | 101 | 51 | 63 | 98 | 107 | 63 | 59 | 107 |
| 2023 P90 | 5 | 38 | 18 | 13 | 10 | 107 | 72 | 74 | 103 | 106 | 71 | 68 | 96 |

The conditional and joint proper-loss cores pass most cells, whereas absolute
rate levels pass about one fifth and statistically positive incremental lift
passes far fewer. Daily intervals and quarter/stock robustness are also much
weaker than pooled proper loss. A model can therefore improve probability
estimates without making a given cycle a context-adjusted high-movement class.

## This is not primarily a support failure

Base support passed 20/20 cycles in 2024 OOF, 17/20 in 2025, and 19/20 in
backward-2023. The 2025 failures were cycles 12, 18, and 19 because they had
fewer than 500 realised-loop rows. Cycle 12 failed backward-2023 because it
occurred in only fifteen stocks versus eighteen required.

Those sparse cycles are not the candidate set. Cycles 06, 07, 09, and 13 each
passed base support in every period, with 884 to 3,312 realised-loop rows.
Removing the per-cell event-support gate alone adds no P75 or P90 passing cell.
Support can amplify uncertainty, especially at P90, but it is not the decisive
reason no cycle receives a global grade.

## Continuous stability is stronger than threshold stability

Across all 120 matched cells per tier, continuous absolute movement measures
are remarkably portable. The Pearson correlation ranges over the three period
pairs are:

| Metric | P75 Pearson range | P90 Pearson range |
| --- | ---: | ---: |
| Observed exceedance rate | 0.9861 to 0.9955 | 0.9865 to 0.9962 |
| Mean frozen `qcycle` | 0.9983 to 0.9993 | 0.9957 to 0.9993 |
| Observed rate / `qcontext` | 0.8571 to 0.9141 | 0.6251 to 0.6607 |
| Residual-CI lower bound | 0.7460 to 0.7855 | 0.5938 to 0.7487 |
| Conditional log-loss improvement | 0.8164 to 0.8612 | 0.6583 to 0.7206 |
| Joint log-loss improvement | 0.7186 to 0.8688 | 0.5829 to 0.6632 |

The absolute movement ordering is stable. Incremental lift, uncertainty, and
loss allocation to individual cycles are materially less stable, particularly
at P90. The full correlation file also reports Spearman coefficients. This
distinction explains why apparent high-movement loops can recur while strict
all-gate classifications change at the boundary.

No P75 target-horizon cell passes all three periods. No cycle qualifies at 24
bars in any period, and 2025 has no qualifying horizon. Thus the frozen global
failure reflects incremental and slice robustness, not an absence of any
recurrent movement association.

## Two independent diagnostic axes

The diagnostics deliberately separate:

1. **Absolute movement level:** whether observed P75 movement and mean
   `qcycle` are at least 0.35 for both targets and all horizons.
2. **Incremental evidence:** whether the cycle clears the ratio over causal
   `qcontext`, has a positive residual lower bound, and survives the remaining
   proper-loss, calibration, time, and stock gates.

Three cycles—06, 07, and 13—meet the six-cell absolute-high level within every
whole period. Requiring every represented quarter and every leave-one-stock-out
rate also to remain at least 0.35 narrows the exploratory absolute-high set to
cycles 07 and 13.

| Cycle | Diagnostic interpretation | Minimum P75 quarter rate | Minimum P75 stock-deletion rate | Structural reliability in all periods | Frozen grade |
| --- | --- | ---: | ---: | --- | --- |
| `cycle_06` (`4→6→4`) | Whole-period absolute-high, but quarter instability | 0.3305 | 0.3578 | No | Unqualified |
| `cycle_07` (`5→6→5`) | Exploratory absolute-high candidate | 0.3989 | 0.4420 | **Yes** | Unqualified |
| `cycle_09` (`3→6→3`) | Strongest incremental candidate, not robust | 0.2184 | 0.2656 | Yes | Unqualified |
| `cycle_13` (`5→7→5`) | Exploratory absolute-high candidate | 0.4427 | 0.5139 | No | Unqualified |

Cycles 07 and 13 are therefore exploratory absolute-high movement candidates,
not qualified loops. Of those two, cycle 07 alone also passes structural
reliability in all three periods. Cycle 13's movement level is strong, but much
of it is already identified by context, and its structural calibration fails
in 2024 OOF and 2025.

Cycle 09 remains the strongest incremental candidate. Across the three periods
it passes seven of eighteen complete P75 cells and twelve of eighteen
ratio-plus-residual cells, more than any other cycle. Its 2024 OOF 24-bar range
cell missed only the conditional Brier daily upper bound, and several later
cells were close to a rate or residual boundary. Nevertheless, the quarter,
stock, and daily results do not support promotion.

## Structural reliability

Per-cycle structural reliability passes 16/20 cycles in 2024 OOF, 16/20 in
2025, and 14/20 in backward-2023. The 2024 OOF structural result is diagnostic
only under the frozen contract.

Failures are overwhelmingly calibration failures. Only cycle 16 fails the
structural log-loss comparison, and only in 2025; none of the structural
failures is caused by Brier ranking. This supports retaining the aggregate
structural forecaster while keeping per-cycle reliability visible.

## Scientific interpretation and next use

The frozen experiment supports three narrower statements:

- recurrent loop identity contains pooled movement information;
- cycles 07 and 13 repeatedly inhabit absolute-high movement conditions;
- cycle 09 contains the strongest context-adjusted evidence, but its effect is
  not robust enough for the frozen qualification rule.

It does not support calling any cycle high or good under the existing frozen
grade. The absolute-high diagnostic is deliberately post-result and
exploratory; it must not be substituted for the frozen qualification rule.

A separate V2 experiment can now predeclare these two axes before reading new
outcomes: absolute-high persistence for cycles 07 and 13, and context-adjusted
robustness for cycle 09. That experiment must remain separate from both frozen
shadows and cannot retroactively change this result.

## Validation

- Focused diagnostic tests: 7/7 passed.
- Frozen input hashes before/after: exact match for all fifteen inputs.
- Reconstructed raw rates: exact to an absolute tolerance of `1e-12` against
  every frozen quality cell.
- Diagnostic outputs: nine CSV files plus `summary.json`.
- Final frozen grades: 20 `unqualified`, zero changed.
- Model refit or recalibration: none.
- Prospective-shadow access: none.

The `/private/tmp` diagnostic bundle is ephemeral and should be archived before
a reboot if exact reproduction without rerunning the read-only package is
required.
