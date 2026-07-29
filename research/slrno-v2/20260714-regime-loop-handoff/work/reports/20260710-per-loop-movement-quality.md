# Per-loop movement-quality qualification

## Decision

Retain the frozen eight-state regime detector and retained last-three-state
loop-identity probabilities. Retain the new conditional quality model as an
aggregate research signal, but do not certify or surface any individual fixed
cycle as good or high movement quality.

All twenty cycles have a final global grade of `unqualified`. The final grade
is the minimum of causal July-December 2024 out-of-fold development,
full-2025 development, and backward-2023 portability. A loop had to pass both
absolute-return and future-range gates at 6, 12, and 24 five-minute bars, plus
support, calibration, proper-loss, moving-block, quarter, stock-deletion, and
structural-reliability gates. No threshold or gate was relaxed after results
were opened.

This result concerns movement magnitude and range only. It is not a claim
about direction, signed return, profitability, P&L, economic edge,
tradability, or trading performance.

`research_only: true`, `live_ordering_enabled: false`, and
`order_placement: disabled`.

## Frozen experiment

The quality model was fitted only on 2024 anchors where each candidate loop
actually occurred. Overlapping positive loop labels received reciprocal
overlap weights. It estimated the ordered probability that absolute return or
future range would exceed fixed 2024 P75 and P90 thresholds at 6, 12, and 24
bars.

The retained structural probability remained separate:

- `s(i,c)`: frozen probability that loop `c` occurs;
- `q75(i,c)` and `q90(i,c)`: movement-quality probabilities conditional on
  loop `c` occurring;
- `j75(i,c)=s(i,c)q75(i,c)` and
  `j90(i,c)=s(i,c)q90(i,c)`.

The model did not refit the regime detector, change the twenty fixed cycles,
alter the retained loop-identity forecaster, use future-state information as
an input, or use provider `historical_volume`. It used no volume feature.

The 2024 reconstruction contained 70,374 anchors, 460,276 compatible
anchor-cycle rows, 32,677 realized-loop rows, and 29,296 effective
inverse-overlap weight. The causal OOF cohort contained 216,438 rows. The
sealed scoring panels contained 441,983 compatible rows in 2025 and 423,083
in backward-2023.

## Qualification results

| Period | Globally high | Globally good | Unqualified |
| --- | ---: | ---: | ---: |
| 2024 causal OOF | 0 | 0 | 20 |
| 2025 development | 0 | 0 | 20 |
| Backward-2023 portability | 0 | 0 | 20 |
| Final minimum grade | 0 | 0 | 20 |

The closest partial results did not satisfy the global contract:

- `cycle_09` was `good_movement_quality` at 6 and 12 bars in 2024 OOF,
  but failed 24 bars. It qualified at no horizon in 2025 and remained globally
  unqualified.
- `cycle_13` was `high_movement_quality` at 6 bars and
  `good_movement_quality` at 12 bars in backward-2023, but failed 24 bars and
  was not stable across periods.

Support passed for 20/20 cycles in 2024 OOF, 17/20 in 2025, and 19/20 in
backward-2023. The stricter per-cycle structural-reliability condition passed
16/20 cycles in 2025 and 14/20 in backward-2023. This is materially weaker
than the retained forecaster's aggregate result and shows that reliability is
not uniform cycle by cycle.

## What survived

The pooled conditional quality representation contains information even
though no individual cycle earned a stable tier. Against the context-only
conditional model, cycle-aware log loss improved in all twelve
target/horizon/tier cells in each scoring period, with every paired daily
interval below zero:

| Period | Conditional log-loss improvement range | Joint `s*q` log-loss improvement range |
| --- | ---: | ---: |
| 2025 development | 1.5948% to 7.7823% | 0.9314% to 2.5292% |
| Backward-2023 portability | 2.1040% to 8.6320% | 0.8197% to 2.4905% |

The corresponding Brier improvements were positive in all cells as well.
However, calibration, support, rate/lift, and per-cycle robustness did not
align for the same named cycle across both targets, all horizons, and all
periods. The correct interpretation is therefore: useful pooled movement
information, but no certified high/good member of the current twenty-cycle
dictionary.

## Reliability of the regime and loop algorithms

The tested frozen regime detector remains a strong research candidate, not a
prospectively validated truth label. Its backward-2023 semi-Markov state NLL
was 5.9041% better than the IID-emission mixture, all eight states transferred,
and the frozen state-centroid drift remained bounded. The causal probability
that the current state leaves within three bars had ECE 0.006632 in 2025 and
0.011108 in backward-2023, with all quarters, stock deletions, and states
improving over the frozen state-only baseline. Destination history also
improved backward-2023 conditional destination log loss by 10.3476%.

The retained structural loop forecaster is likewise useful in aggregate. Its
last-three-state history improved loop-label log loss by 9.7818% in 2025 and
9.5371% in backward-2023 versus first order. Top-three recall was 0.794116 and
0.786156, with ECE 0.004595 and 0.003386 respectively. Because loop labels
overlap, top-three recall is not conventional exact-loop accuracy, and its
roughly 10.5% top-three precision must also be kept visible.

These results establish stable latent-state and state-path predictability
under the tested lineage. They do not establish that the latent states are
externally true market regimes, and neither 2025 nor backward-2023 is a clean
prospective holdout. A frozen post-freeze shadow remains necessary before the
detector can be called prospectively reliable.

If "the user's regime-discovery algorithm" means a separate, still-uncoded V2
method rather than this frozen eight-state detector, no reliability conclusion
about that V2 follows from these results.

## Consequence for further loop work

More work is needed on the loop-quality layer, not on replacing the frozen
state detector merely because this quality contract failed. The current
fixed-cycle dictionary identifies recurring state paths, but no member has
stable all-horizon movement quality.

Both tested full-loop timing approaches are now closed: the independent
state-only duration product failed, and the later history/destination-
conditioned joint semi-Markov kernel also failed its ranking, incremental,
long-horizon calibration, and path-only gates. Neither should be revived or
quietly retuned under the same evidence.

The next separate development experiment should instead target the cycle
dictionary and quality stability. A defensible V2 would use a nested 2024
discovery/calibration split, hierarchical multi-horizon shrinkage, and
predeclared horizon-specific quality classes before any later period is read.
The user's alternative regime-discovery method may also be implemented as a
separate V2 and compared with the frozen detector; it must not overwrite it.
`cycle_09` and `cycle_13` remain exploratory observations, and the present
contract cannot be retroactively weakened to promote them.

The existing aggregate movement prospective shadow remains unchanged. A
separate per-loop quality shadow has been initialized in a dormant state with
zero eligible cycles, an empty ledger, unopened outcomes, and no outcome
evaluator. It cannot issue an unqualified loop.

## Audit and reproducibility

- Frozen contract SHA-256:
  `67d64c463df52f01f360561ef0a69d5772b7eec0409468c93d6eb5a630dee02e`.
- Frozen runner SHA-256:
  `7da5e88e603583d3dba7422569bc8e27837171c7165e69bcaafade472738e2ea`.
- Independent pre-score audit: 35/35 checks passed, including exact OOF
  predictions and all provisional gates.
- Independent post-score audit: 48/48 checks passed. It reproduced every
  label and probability with zero mismatch, all aggregate and per-cycle
  metrics, 480 quality cells, 120 horizon grades, 40 period grades, and every
  final minimum-tier decision. Maximum numerical drift was
  `2.22e-16`.
- Existing aggregate shadow protected-tree SHA-256 remained
  `38e90e9db3ae2974db2f6726bb69dfadde0410c2b200e7ce9e080fcbd22bc267`;
  its ledger remained empty.
- The compact frozen model, final decisions, manifests, and independent audit
  are archived inside
  `work/shadow_validation/frozen_loop_quality_shadow_v1/frozen_bundle`.

The large reconstruction/scoring panels remain under
`/private/tmp/stocker_per_loop_movement_quality_20260710` and are ephemeral.
They are not needed for quality-shadow issuance, which is disabled, but must
be archived separately before reboot if exact row-level audit replay is
required without recomputation.
