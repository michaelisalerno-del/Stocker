# Loop-quality V2 feature-ablation contract

## Frozen decision

A minimal V2 source-attribution experiment is now specified and frozen, but it
has not been run. Its purpose is to determine where the pooled conditional
movement-quality information comes from:

- semantic candidate-route topology;
- explicit cycle identity;
- cycle by current-state rotation;
- or the last-three-state history token.

The contract does not relax the prior movement-quality gates, change any of the
twenty frozen cycles, or reopen the all-unqualified decision. It creates no
shadow and authorizes no fit by itself.

`research_only: true`, `live_ordering_enabled: false`, and
`order_placement: disabled`.

The experiment concerns absolute movement magnitude and future range only. It
cannot be interpreted as direction, signed return, P&L, economic edge,
tradability, strategy performance, or an execution path.

## Why this experiment

The read-only attribution found a real separation:

- raw cycle-associated movement rates are extremely stable across periods;
- the full cycle-aware quality model improves pooled proper loss;
- but strict incremental lift and named-cycle robustness do not attach to the
  same loop across every period, horizon, target, quarter, and stock deletion.

That leaves an unresolved representation question. The existing `qfull` model
may be exploiting broad facts such as "the candidate next state is a high-
activity state" or "this route contains states 5 and 7," rather than unique
cycle identity or the complete last-three-state history. V2 isolates those
possibilities without touching the detector or outcomes.

## The five representations

All models use the same ordered multinomial specification, 2024 folds,
realised-loop rows, inverse-overlap weights, causal controls, `C=0.2`, solver,
seed, and fixed temperature `1.0`. The temperature is fixed because every
frozen `qcontext` and `qfull` OOF selection previously chose `1.0`; calibration
tuning must not become another varying component.

| Model | Added information | Width | Primary comparison |
| --- | --- | ---: | --- |
| `qcontext` | Exact frozen state/context controls | 17 | Baseline; reuse sealed predictions |
| `qroute_topology` | Candidate next-state distribution, route-state composition, transition length, frozen centroid summaries, rotation ambiguity | 80 | Topology versus context; topology retention versus full |
| `qcycle_main` | Frozen twenty-cycle one-hot | 37 | Cycle representation versus route topology |
| `qcycle_state` | Cycle main plus cycle × current state | 197 | Current-state rotation increment versus cycle main |
| `qfull` | Existing cycle main, cycle × state, and cycle × 648-history token | 13,157 | History increment versus cycle-state; sealed full reference |

`qcontext` and `qfull` must be read from the sealed experiment and replay every
probability and loss to `1e-12`; they are not refitted. Only the three interior
ablation models would be fitted if execution is separately authorized.

`qcycle_main` and `qroute_topology` are deliberately competing, non-nested
representations. If cycle main outperforms route topology, that is evidence
that explicit cycle identity retains information omitted by this topology
summary. It is not described as a formal conditional-variable proof.

## Causal route topology

At an anchor, some cycles have more than one compatible rotation because the
current state appears multiple times in the cycle core. Choosing the rotation
that later occurs would leak the future. The frozen rule is therefore:

1. Enumerate and deduplicate every rotation beginning at the current filtered
   state.
2. Close each rotation by returning to that state.
3. Give every compatible rotation equal weight.
4. Build topology only from that uniform candidate set.

The topology block contains:

- an eight-state distribution for the candidate next state;
- an eight-state composition of all required future destinations;
- a one-hot transition length of two, three, or four;
- candidate-rotation count and normalized next-state entropy;
- the expected frozen centroid of the next state;
- the expected frozen centroid of the route composition;
- the expected next-state centroid minus the current-state centroid.

The centroids are the frozen `8 × 14` standardized-emission means estimated
after semantic state remapping in 2024. They are normalized across the eight
frozen centroids only. No current bar emission, future state, realized route,
duration, session end, future price, or stock identity can enter this block.

Two original centroid dimensions were derived from provider
`historical_volume`. They remain frozen semantic coordinates. They are not a
new volume input, exchange-wide volume, or order flow.

## Exact evaluation unit

The cohort, labels, P75/P90 thresholds, targets, horizons, controls, and overlap
weights are inherited without change from `per_loop_movement_quality_v1`.
Each ordered model produces P75 and P90 probabilities for:

- absolute return at 6, 12, and 24 five-minute bars;
- future range at 6, 12, and 24 bars.

This gives twelve binary evaluation cells. Within each cell, conditional losses
use the frozen inverse-overlap weight; joint `s × q` losses use every compatible
anchor-cycle row and the unchanged structural probability. The primary pooled
loss is the arithmetic mean of the twelve within-cell losses, so common P75
events cannot dominate rarer P90 events.

## Predeclared comparisons and gates

Five primary comparisons form one family:

1. `qroute_topology` versus `qcontext`;
2. `qroute_topology` non-inferiority versus `qfull`;
3. `qcycle_main` versus `qroute_topology`;
4. `qcycle_state` versus `qcycle_main`;
5. `qfull` versus `qcycle_state`.

Every primary comparison uses a one-sided 99% five-session moving-block
interval with 10,000 draws. Five 1% tests provide a Bonferroni familywise level
of 5%. Paired models use identical resampled block indices.

Except for topology non-inferiority, a superiority comparison must satisfy all
of the following on both conditional and joint surfaces:

- pooled relative log-loss improvement of at least 0.25%;
- negative pooled Brier difference;
- 99% block upper bounds below zero for log loss and Brier;
- negative pooled log-loss and Brier differences in every required quarter;
- negative differences under every leave-one-stock-out deletion;
- no twelve-cell relative log-loss degradation worse than 0.25%;
- neither target aggregate nor any horizon aggregate may be worse;
- predeclared ECE and supported-bin non-inferiority tolerances.

Topology explains most of the full signal only if it retains at least:

- 90% of the conditional and joint `qfull` log-loss gain over context;
- 80% of the corresponding Brier gain;
- 75% of every positive cell-level full gain;

and its pooled, quarter, stock-deletion, block-interval, and calibration results
remain within the frozen non-inferiority margins.

These gates are intentionally about source attribution, not loop
qualification. An ablation result cannot replace the older good/high movement
contract.

## Source labels

The only permitted V2 source conclusions are:

- `no_reference_signal` — sealed `qfull` does not reproduce or lacks the
  required robust reference gain;
- `topology_sufficient` — topology retains the full gain and no deeper block
  adds robust information;
- `topology_dominant_with_residual_detail` — topology retains most of the gain,
  but identity, rotation, or history also adds robust information;
- `cycle_identity_representation_needed` — cycle main robustly beats the
  non-nested topology representation;
- `current_state_rotation_needed` — cycle × current state beats cycle main;
- `history_token_needed` — full history beats cycle × state;
- `unresolved` — a full reference signal exists but the frozen comparisons do
  not identify its source.

A pooled label cannot be applied to a target, horizon, tier, cycle, or supported
rotation slice whose own result reverses sign.

## Rotation diagnostics

Confirmatory causal slices are predeclared for:

- cycle by current state;
- compatible-rotation count;
- next-state-entropy quartile using 2024-frozen cut points.

A source claim fails its rotation robustness condition if any supported causal
slice has positive candidate-minus-baseline pooled log-loss or Brier
differences. OOF slices require at least 100 realised rows, ten stocks, and both
quarters; later scoring slices require at least 200 rows, ten stocks, and all
four quarters.

The exact realized rotation may be reconstructed after outcomes solely for a
descriptive table. It cannot enter fitting, calibration, selection, or a
confirmatory gate.

## Absolute-high and incremental axes remain separate

The already observed hypotheses are explicitly recorded before V2 execution:

- cycles 07 and 13 are exploratory absolute-high persistence candidates;
- cycle 09 is the strongest prior incremental candidate but failed robustness.

Raw absolute-high rates cannot satisfy an incremental proper-loss gate, and an
incremental loss improvement cannot substitute for the fixed absolute-high
rate requirement. V2 must report both axes side by side for every cycle and
must not combine them into a new score or tier.

## Period handling

The causal July-December 2024 OOF result is the only source of a provisional
V2 attribution. Even that result is exploratory because this lineage has
already been studied extensively.

Before any scoring process may load later row-level outcomes, it must freeze:

- all 2024 predictions and model parameters;
- topology and rotation mappings;
- every loss, gate, and provisional source label;
- contract, source, and artifact hashes;
- an independent passing pre-score audit.

The full 2025 and backward-2023 panels are already opened development and
portability evidence, not validation. They may only demote a 2024 source
conclusion. They cannot promote a failed comparison or substitute another
source label. Partial 2026 is prohibited.

## Stop rules

Execution must stop without source attribution if:

- a pinned source, cohort, threshold, label, weight, cycle, state index, or
  centroid fails reconstruction;
- any topology input depends on the future or on stock identity;
- `qcontext` or `qfull` fails exact replay;
- a 2024 fold is not causal, a model fails convergence, probabilities are
  invalid, support fails, or the independent pre-score audit fails;
- the full reference signal fails its precondition;
- a result suggests changing a feature, scale, penalty, threshold,
  calibration, rotation aggregation, or gate after outcomes are read.

Any redesign requires a separately frozen V3 contract. Neither prospective
shadow may be read or modified at any point.

## Frozen files and hashes

- Contract:
  `work/contracts/20260710-loop-quality-feature-ablation-v2.json`
- Contract SHA-256:
  `33d109a1bcc7ee58fb5ee65a5a5c1075a233baa07d50b1219db8358af22f4728`
- Planned artifact root if a later request authorizes execution:
  `/private/tmp/stocker_loop_quality_feature_ablation_v2_20260710`

Pinned predecessor hashes are embedded in the JSON contract. The contract is
valid JSON and has been frozen before any V2 model fit, prediction, or outcome
comparison.

No V2 runner, model, prediction, or score was created in this step.
