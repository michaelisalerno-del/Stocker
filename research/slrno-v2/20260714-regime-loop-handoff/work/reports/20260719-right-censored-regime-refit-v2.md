# Right-Censored Regime Refit and Stability Rerun V2

Safety boundary: `research_only=True`, `execution_enabled=False`, `order_placement=disabled`, `broker_connected=False`, `economic_outcomes_used=False`, `payoff_selection_used=False`, `production_runtime_modified=False`, `strategy_promotion=False`, `part_b_interaction_scoring_enabled=False`, `semantic_dictionary_promotion_enabled=False`.

## Exact scope

Terminal-duration censoring, causal gap resets, deterministic K=8 refit, and unchanged Part A rerun only. No predictor, interaction scoring, dictionary promotion, or economic testing was performed.

## Source identity

Implementation target `91996a9cf747a614ff6d9e08eaafc3583a58b91c`; contract `7c23ba6c613d79731d3e3f1f37122ff5f97e6e6845274121313ccb1ba1459e88`; development snapshot `48d2141ef993928d4e8a01d6b3c24dff665280c67f4167115b453613460cc661`.

## Frozen lineage protection

All pre-existing tracked files were hash-compared to the frozen pre-repair manifest; the independent audit records the result.

## Missing historical panel dependency

`run_sealed_2025_sec_raw_activity_validation.py` remains unavailable. Historical KMeans byte-equivalence is therefore not claimed.

## New archived panel builder

The archived builder produced 424,583 development rows with all fourteen declared emissions.

## Deterministic row ordering

Natural order is symbol/session/timestamp/bar ordinal; row-key hash `2248079ba30c6fb7aa780f12ee50006391ffdac6a882a1aa9a04e1bebee12080`.

## Emission reconstruction

Every emission has an explicit stock, market, or stock-relative partition and completed-bar provenance. The deterministic sample was independently checked.

## Source-gap segmentation

Sessions split into independent contiguous causal segments at missing ordinals or timestamps.

## Run-ending classification

- `INCOMPLETE_OR_UNAVAILABLE_SESSION`: 5,037 runs, 17,577 observed bars.
- `INVALIDATED_BY_SOURCE_GAP`: 3,757 runs, 10,766 observed bars.
- `OBSERVED_STATE_EXIT`: 77,481 runs, 352,508 observed bars.
- `RIGHT_CENSORED_SESSION_END`: 5,080 runs, 43,732 observed bars.

## Exact exits

Observed exits: 77,481.

## Right-censored terminal runs

Session-terminal censored runs: 5,080.

## Gap-invalidated runs

Gap-invalidated runs: 3,757.

## Incomplete sessions

Incomplete/unavailable runs: 5,037.

## Corrected at-risk counts

Exact exits and censored terminal runs contribute exposure through observed age; excluded endings contribute none.

## Corrected exit counts

Only OBSERVED_STATE_EXIT contributes an exit at its exact age.

## Hazard estimation

Hazards use frozen Beta(0.5, 0.5) smoothing with deterministic state-to-pooled-to-tail backoff.

## Survival curves

Nonnegative, non-increasing survival and conserved mass passed: `True`.

## Duration 24

Age 24 is exact and is not a forced exit.

## Durations greater than 24

Ages 25–78 remain separate support points.

## Duration 78

Age 78 is representable and retains survival mass where the hazard is below one.

## No forced terminal hazard

Forced age-24 or age-78 exits observed: `False`.

## Tail backoff

Sparse cells blend deterministically toward pooled-age evidence and a preregistered 0.05 tail prior.

## Duration-only repair

Parameter hash `40d5b2c149856e2e2cdbf3df15adfe0c8108c1bb45ada73235a87bf67f87ce44`; all frozen non-duration arrays remain byte-identical.

## Complete deterministic refit

Model hash `4fc1a02dce9ac2311dabaeb4623a559d37286dfe58baffef53828cc7415a3425`; training-row hash `6224fe722280312a7f3d11a953ea13ecdbc8edafac31206d776dc78e8b2e6b3a`.

## Determinism results

Clean second fit and directory rerun identity passed: `True` across 83 artifacts.

## Frozen versus repaired comparison

- `MODEL_FROZEN`: NLL 13.543580, hard transitions 103,506, hysteretic agreement 0.9533.
- `MODEL_DURATION_REPAIR`: NLL 13.542797, hard transitions 103,960, hysteretic agreement 0.9531.
- `MODEL_FULL_REFIT`: NLL 13.257229, hard transitions 94,184, hysteretic agreement 0.9656.

## Posterior impact

See `duration_defect_impact.csv` and `repair_component_attribution.csv`; differences are separated into duration-only and complete-refit consequences.

## State-boundary impact

Aligned run-boundary changes are archived without comparing arbitrary numeric labels.

## Loop-event impact

- `MODEL_FROZEN`: coverage 0.093242, bounded event agreement 1.000000.
- `MODEL_DURATION_REPAIR`: coverage 0.092884, bounded event agreement 0.909742.
- `MODEL_FULL_REFIT`: coverage 0.059734, bounded event agreement 0.121407.

## Dictionary-coverage impact

Coverage is diagnostic only; it was not used to select the repair, K, sample, cleanup, or smoothing.

## Tests

- New package and research tests: 62 passed.
- Existing top-level posterior, duration, loop-event, Regime Validity, and Semantic Dictionary tests: 157 passed.
- Archived Regime Model Validity V2 tests: 14 passed.
- Archived Semantic Loop Dictionary V2 tests: 50 passed with its declared research-work import path.
- Archived Loop Event Semantics V2 tests: 16 passed and 5 failed because five frozen detailed Parquet ledgers are absent from this checkout.
- Scoped Ruff format and lint: passed. Strict mypy: passed for all five reusable modules. `git diff --check`: passed.
- Full top-level repository suite: 13 failures and 19 setup errors in unrelated historical execution/integration tests, led by the absent frozen `20260714-dynamic-loop-edge-state-v2/primary/trade_decisions.parquet`. No repair-focused test failed.

## Independent audit

Independent audit passed: `True`.

## Exact rerun

Byte-identical: `True`.

## Repair scientific decision

`right_censored_regime_repair_complete_with_known_limitations`.

## Remaining limitations

missing historical ephemeral panel builder prevents byte-equivalence with the original KMeans fit

## Exact next step

replace exact numeric loop identities with cluster-invariant closure topology or continuous posterior trajectories
