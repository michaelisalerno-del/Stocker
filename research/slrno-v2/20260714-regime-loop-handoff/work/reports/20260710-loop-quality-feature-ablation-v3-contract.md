# Loop-quality V3 feature-ablation contract

## Frozen before fitting

V3 is frozen as a support-corrected successor to the deterministically stopped
V2 experiment. No V3 model had been fitted and no V3 prediction or later-period
row had been read when this contract was hashed.

Contract:

`work/contracts/20260710-loop-quality-feature-ablation-v3.json`

SHA-256:

`221a016e78c353a70261fe724cdfc4d312e355febfc353449844b31b8862702d`

`research_only: true`, `live_ordering_enabled: false`, and
`order_placement: disabled`.

## Authorized scope

The user explicitly authorized this turn to freeze V3 and then execute the
research-only five-representation experiment. That authorization covers:

- causal July-December 2024 OOF fitting for the three interior ablations;
- sealed reuse of existing `qcontext` and `qfull` predictions;
- full-2024 fitting of the three interior models;
- 2025 development and backward-2023 portability scoring only after a frozen
  fit bundle and passing independent pre-score audit.

It does not authorize live, demo, paper, broker, order, position, deployment,
strategy, or prospective-shadow activity.

## V2 stop lineage

V2 independently verified that its 20,000 effective-weight threshold could not
be met by the unique OOF cohort:

- realised rows: 15,584;
- unique realised anchors/effective weight: 14,167;
- independent V2 audit: 57/57 checks passed;
- no V2 model fit, prediction, later-period read, or attribution occurred.

V3 pins the V2 contract, runner, audit source, fit marker, stop decision,
support audit, pre-score audit, and final artifact-audit hashes.

## Only semantic change: unique-cohort support

The V3 OOF support gate is:

- total unique inverse-overlap effective weight at least 10,000;
- effective weight in each of 2024 Q3 and Q4 at least 5,000;
- at least 100 sessions;
- at least 18 stocks;
- at least 50 effective weight for every represented stock;
- at least 15,000 realised anchor-cycle rows as reconstruction integrity only.

The realised-row count is not an independent information or support gate,
because overlapping loop rows share anchors. Repeating the cohort across
targets, horizons, or tiers is forbidden for support. These thresholds were
chosen from cohort structure only, not movement outcomes or model performance.

The already verified support-only values are 14,167 total weight, 7,635 and
6,532 by quarter, 128 sessions, 22 stocks, minimum per-stock weight 93, and
15,584 realised rows.

## Everything else preserved

V3 preserves V2's complete scientific design:

- the five representations `qcontext`, `qroute_topology`, `qcycle_main`,
  `qcycle_state`, and `qfull`;
- every feature, width, feature scale, centroid transformation, and uniform
  compatible-rotation rule;
- the same causal 2024 folds, labels, controls, thresholds, overlap weights,
  solver, `C`, seed, and fixed temperature;
- the same conditional and joint losses, equal-cell pooling, calibration,
  99% Bonferroni block intervals, quarter, stock-deletion, cell, target,
  horizon, and rotation gates;
- topology retention and identity/state/history comparisons;
- separate absolute-high and incremental axes;
- demotion-only 2025/2023 transfer logic;
- every future-data, grade, safety, no-shadow, and no-live stop rule.

Any later change requires a separately frozen V4.

## Required execution sequence

1. Reconstruct pins, rotations, features, OOF rows, labels, and unique support.
2. Fit and evaluate 2024 OOF only.
3. Fit the three interior models on full 2024.
4. Freeze code, parameters, predictions, gates, provisional attribution, and
   artifact hashes in `fit_complete.json`.
5. Run an independent pre-score audit that imports no production runner code.
6. Permit later-period loading only if the audit passes and explicitly sets
   `scoring_authorized: true`.
7. Score 2025 and backward-2023, which may only demote the 2024 result.
8. Run a final independent artifact audit.

Planned artifacts are isolated under:

`/private/tmp/stocker_loop_quality_feature_ablation_v3_20260710`

The `/private/tmp` bundle is ephemeral and should be archived before reboot if
exact replay without recomputation is required.
