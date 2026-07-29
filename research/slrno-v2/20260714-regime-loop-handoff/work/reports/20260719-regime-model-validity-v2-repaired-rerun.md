# Regime Model Validity V2 — Repaired Unchanged-Gate Rerun

Safety boundary: `research_only=True`, `execution_enabled=False`, `order_placement=disabled`, `broker_connected=False`, `economic_outcomes_used=False`, `payoff_selection_used=False`, `production_runtime_modified=False`, `strategy_promotion=False`, `part_b_interaction_scoring_enabled=False`, `semantic_dictionary_promotion_enabled=False`.

## Exact scope

An unchanged-gate Part A rerun over the repaired primary state model; Part B remained closed.

## Source identity

Model `4fc1a02dce9ac2311dabaeb4623a559d37286dfe58baffef53828cc7415a3425`, panel `801c0bf9d69ecdd58b21fb2ba4392137048b466668344ebfc4c8faf6a0d3e2f1`, contract `7c23ba6c613d79731d3e3f1f37122ff5f97e6e6845274121313ccb1ba1459e88`.

## Current repaired state implementation

K=8 combined fourteen-feature causal semi-Markov model with right-censored 1–78 duration support and segment resets.

## Mathematical audit

Posterior normalization, transition normalization, hazards, survival, and probability conservation passed.

## Causality audit

Completed-bar features and segment-local recursion passed; no protected 2026 data or future outcome was opened.

## Duration and censoring status

Terminal sessions are right-censored; gap and unavailable endings are excluded from the primary duration fit.

## Offline-cleaning findings

Historical CLEANING_1 remains the primary declared noncausal training cleanup; CLEANING_0 and CLEANING_CAUSAL remain separate sensitivities.

## Raw versus cleaned labels

Detailed deterministic raw and cleaned assignments and cleaning metrics are archived.

## Hard-state churn

Low-margin and reversal diagnostics are archived in the unchanged-gate artifacts.

## Posterior-confidence results

Entropy, top-two margin, and hysteretic agreement were evaluated without creating trading thresholds.

## K sensitivity

K={6,8,10,12} was rerun unchanged.

## Seed sensitivity

Minimum K=8 NMI: `0.43455277851926727`.

## Training-sample sensitivity

Minimum coverage ratio: `0.21438617641149288`; minimum selected-event agreement: `0.002016868353502017`.

## State alignment

All alternatives use deterministic Hungarian centroid/transition/duration alignment.

## Semantic drift

Gate pass: `False`.

## Stock heterogeneity

Maximum single-stock share: `0.21212691836926842`.

## Clock heterogeneity

Clock-phase profiles are archived without threshold search.

## Combined versus stock-only representation

The comparison retains unchanged structural likelihood, drift, concentration, and loop diagnostics.

## Hierarchical market × stock representation

The preregistered hierarchy rules were rerun and did not receive outcome-based selection.

## Hard, hysteretic, and soft loop robustness

Hysteretic selected same-primitive fraction: `0.9028073495781089`; soft mass did not create hard events.

## Primitive-loop stability

K=8 positive-excess counts: `{'loop_p_4-6-4': 5, 'loop_p_5-6-5': 5}`.

## Dictionary stability

The existing dictionary remained diagnostic and promotion-disabled.

## Failure cases

- `semantic_drift`: 0.0 against 1.0.
- `minimum_k8_seed_nmi`: 0.434552778519 against 0.5.
- `training_sample_dictionary_coverage`: 0.214386176411 against 0.75.

## Missing evidence

Historical panel byte-equivalence remains unavailable. Five frozen detailed Loop Event Semantics V2 Parquet ledgers and the historical dynamic-loop `trade_decisions.parquet` used by unrelated repository integration tests are also absent. The repaired lineage itself is reproducible and independently audited.

## Part A scientific decision

`regime_representation_unstable_loop_dictionary_must_pause`.

## Whether dictionary work may proceed

`False`; promotion remained disabled in this task.

## Exact next step

replace exact numeric loop identities with cluster-invariant closure topology or continuous posterior trajectories
