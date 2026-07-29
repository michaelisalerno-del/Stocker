# Loop-quality V3 feature-ablation result

## Decision

The five-representation V3 experiment completed under its frozen contract. It
does not identify a robust source for the loop-conditioned movement-quality
signal. The frozen final development/portability label is
`no_reference_signal`.

That label has a narrow technical meaning: the sealed full loop model did not
clear every predeclared reference gate. It does **not** mean that the pooled
movement signal disappeared. The full model improved average conditional log
loss over causal state/context by 4.0893% in 2024 OOF, 3.7893% in 2025, and
4.3155% in backward-2023. It failed the complete reference precondition because
calibration was not uniformly non-inferior and supported cycle/current-state
rotation slices contained sign reversals.

Candidate-route topology carries most of the average signal, but it also fails
the frozen calibration, rotation, and full-retention requirements. Explicit
cycle identity, cycle by current state, and the last-three-state history token
add only small pooled increments and none passes its complete superiority
gate. This is useful diagnostic evidence about where to improve the loop
model, but it is not a validated feature-source claim and it does not qualify
any named cycle as good or high.

All twenty frozen loop-quality grades remain `unqualified`.

`research_only: true`, `live_ordering_enabled: false`, and
`order_placement: disabled`.

The experiment concerns probabilities of large absolute return and future
range only. It provides no direction, signed-return, P&L, economic-edge,
tradability, strategy, order, position, or deployment conclusion.

## Frozen execution sequence

V3 changed only the unique-cohort support rule that deterministically stopped
V2. The contract was frozen before fitting at SHA-256
`221a016e78c353a70261fe724cdfc4d312e355febfc353449844b31b8862702d`.
Every model, feature, scale, fold, loss, multiplicity correction, comparison,
rotation, transfer, safety, and no-shadow rule remained unchanged from V2.

The causal July-December 2024 OOF support gate passed:

| Support measure | Observed | Frozen minimum |
| --- | ---: | ---: |
| Unique inverse-overlap effective weight | 14,167 | 10,000 |
| 2024 Q3 effective weight | 7,635 | 5,000 |
| 2024 Q4 effective weight | 6,532 | 5,000 |
| Sessions | 128 | 100 |
| Stocks | 22 | 18 |
| Minimum per-stock effective weight | 93 | 50 |
| Realised anchor-cycle rows | 15,584 | 15,000 integrity only |

The realised-row count was used only to verify reconstruction; overlapping
cycle rows were not treated as independent support. All 108 causal OOF fits
and all 18 full-2024 fits converged. The fit-only bundle was frozen with runner
SHA-256
`c3aa481dd880e35cc0cc07baa41b6d6c2ed1c380d935e31ce8c1a9d4ff7f05c8`
and fit-marker SHA-256
`99c4e97d9779a4fa190640628ea8482aa258eebb820921b3bdbdde1d89b67730`.

An independent pre-score replay then passed 55/55 checks, reproducing all 108
OOF model predictions and all 18 full-model parameter/scaler bundles with
maximum error `0.0`. Only after that audit set `scoring_authorized: true` did
the locked runner load 2025 and backward-2023. Both later support gates passed:

| Period | Effective weight | Minimum | Realised rows | Stocks |
| --- | ---: | ---: | ---: | ---: |
| 2025 development | 28,239 | 25,000 | 30,778 | 22 |
| Backward-2023 portability | 26,182 | 25,000 | 29,358 | 20 |

The later periods were already opened development/portability evidence. They
are not prospective validation and were permitted only to demote, never to
promote, the frozen 2024 conclusion.

## Average signal and topology retention

The table reports equal-cell pooled log-loss improvements. Retention is the
route-topology gain over context divided by the sealed full-model gain over
context on the same surface.

| Period | Topology vs context, conditional | Full vs context, conditional | Conditional retention | Topology vs context, joint | Full vs context, joint | Joint retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 OOF | 3.6880% | 4.0893% | 90.19% | 1.6413% | 1.8145% | 90.46% |
| 2025 | 3.4435% | 3.7893% | 90.87% | 1.5251% | 1.6545% | 92.18% |
| Backward-2023 | 3.9759% | 4.3155% | 92.13% | 1.3218% | 1.5432% | 85.65% |

Conditional Brier-gain retention was 86.92%, 87.77%, and 91.24% in 2024,
2025, and backward-2023. Joint Brier-gain retention was 100.28%, 90.00%, and
85.57%. Thus topology retained most pooled signal, but backward-2023 joint
log-loss retention fell below the frozen 90% minimum, and the stricter
cell/interval/quarter/stock/calibration/rotation requirements also did not all
pass.

The result suggests that broad semantic route geometry is more important than
memorising a named loop. It does not prove that topology is sufficient:
topology versus context failed uniform calibration and the no-sign-reversal
rotation rule in every period.

## What the deeper loop blocks added

The pooled log-loss increments below are candidate improvements over the
preceding representation. Positive values are better; a negative value is
worse. None passed both conditional and joint surfaces plus all frozen
robustness gates.

| Period | Cycle identity over topology, conditional / joint | Cycle × state over identity, conditional / joint | History over cycle × state, conditional / joint |
| --- | ---: | ---: | ---: |
| 2024 OOF | 0.3511% / 0.1484% | 0.0288% / 0.0311% | 0.0370% / -0.0033% |
| 2025 | 0.2656% / 0.0920% | -0.0134% / 0.0132% | 0.1062% / 0.0262% |
| Backward-2023 | 0.2163% / 0.1860% | 0.0574% / 0.0227% | 0.0803% / 0.0157% |

The explicit cycle identity block gives the largest of these residual average
increments, but it is small, fails calibration/rotation robustness, and is a
non-nested comparison with topology. Current-state rotation and the 648-token
history interaction add almost no stable pooled improvement. Adding more
history complexity is therefore not supported as the immediate remedy.

## Why the frozen source label is negative

The sealed full model versus context passed the broad pooled-loss direction,
block-interval, quarter, stock-deletion, target, horizon, and cell-degradation
checks. Its complete reference comparison nevertheless failed on both
surfaces in all three periods because the frozen calibration-noninferiority
gate failed; supported cycle/current-state slices also reversed the pooled
gain. Since a valid full reference signal was a precondition for all source
labels, the contract requires `no_reference_signal` and forbids later-period
promotion or substitution.

Likewise, topology's pooled improvements were repeatable, including every
quarter and stock deletion in the main comparisons, but its calibration and
supported-rotation behaviour were not uniform. The scientific issue is
heterogeneity and probability reliability, not a lack of average association.

## High-movement loops remain a separate axis

V3 kept raw absolute movement level separate from incremental proper-loss
evidence. Cycles 06 (`4→6→4`), 07 (`5→6→5`), and 13 (`5→7→5`) exceeded the
whole-period six-cell P75 absolute-high level in 2024 OOF, 2025, and
backward-2023. The earlier frozen read-only quarter and stock-deletion audit
narrows the persistent absolute-high hypotheses to cycles 07 and 13:

- cycle 06 falls below the exploratory threshold in at least one quarter;
- cycle 07 is the only absolute-high candidate that also retained structural
  reliability in all three periods;
- cycle 13 has the strongest raw movement level, but its structural
  calibration failed in 2024 OOF and 2025;
- cycle 09 (`3→6→3`) remains the strongest context-adjusted incremental
  candidate, but its time and stock robustness still fails.

These are exploratory diagnostics, not frozen grades. An absolute-high raw
rate cannot replace incremental calibration and robustness, and an average
proper-loss gain cannot replace the absolute-high requirement. No cycle may be
described as qualified good/high from this experiment.

## Research implication

The next loop-model work should target calibrated heterogeneity: identify why
the same semantic route helps on average but reverses in particular
cycle/current-state rotations. A separately frozen experiment could use
hierarchical or partial-pooling calibration across route topology and rotation
slices. V3 does not support simply expanding the history token, changing a
threshold after seeing these outcomes, or retroactively grading cycles 07,
09, or 13.

This ablation did not retest the eight-state regime detector itself. The
regime detector retains its previously documented causal portability result;
V3 addresses only which loop representation explains large-movement and range
probabilities conditional on those states.

## Integrity and safety

- Final development/portability attribution: `no_reference_signal`.
- Portable comparison families passed: 0/6.
- Parent grade changes: none; all twenty remain `unqualified`.
- Prospective validation performed: no.
- Partial-2026 rows read: no.
- Live shadow tree read or written: no/no.
- Later-period promotion performed: no.
- Direction, signed return, P&L, edge, or tradability tested: no.
- Focused V3 runner and audit tests: 19/19 passed.
- Full workspace research test suite: 94/94 passed.
- Independent pre-score replay: 55/55 checks passed.
- Independent post-score replay: 44/44 checks passed.
- Maximum independently replayed scoring-probability error: `4.44e-16`.
- Independent artifact-audit SHA-256:
  `b31c2d5c1b367f1ef93d4b24f00e6ee0c94c7cf64a63a8cc3ad975d9bd40ab5f`.

Artifacts are under:

`/private/tmp/stocker_loop_quality_feature_ablation_v3_20260710`

The bundle is ephemeral and should be archived before reboot if exact replay
without recomputation is required.
