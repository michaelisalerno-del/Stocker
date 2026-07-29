# Loop-quality V2 deterministic support stop

## Decision

The frozen V2 feature-ablation experiment stopped before any model fit or
prediction. Its causal July-December 2024 OOF cohort contains `14,167.0` units
of unique inverse-overlap conditional weight, below the frozen minimum of
`20,000.0`.

No topology-versus-cycle source conclusion is permitted. No 2025,
backward-2023, or partial-2026 scoring panel was read by the V2 runner. The
frozen V2 contract remains byte-identical.

`research_only: true`, `live_ordering_enabled: false`, and
`order_placement: disabled`.

## Why the support gate fails

The exact frozen OOF reconstruction is:

| Measure | Value |
| --- | ---: |
| Compatible anchor-cycle rows | 216,438 |
| Realised anchor-cycle rows | 15,584 |
| Unique realised anchors | 14,167 |
| Unique inverse-overlap weight | 14,167.0 |
| Frozen minimum effective weight | 20,000.0 |
| Unique symbols | 22 |
| Sessions | 128 |
| Quarters | 2024 Q3 and Q4 |

Overlapping realised loops receive weight
`1 / number_of_realised_cycles_at_the_anchor`. Consequently, the weight sums
to one per unique realised anchor and exactly equals the 14,167 unique anchors.
That is the relevant information unit.

Summing the same conditional weight separately over twelve target-horizon-tier
cells would produce `170,004.0`. V2 explicitly rejects that number for support:
the twelve outcomes do not create twelve independent observations from one
anchor. Under the scientifically conservative unique-cohort interpretation,
the frozen gate deterministically fails.

## What the runner did

`work/run_loop_quality_feature_ablation_v2.py` performed only pre-fit work:

- verified the frozen V2 contract SHA-256;
- verified all pinned parent, cycle, threshold, feature, state-centroid, OOF,
  training, and 2024-anchor inputs;
- reconstructed all compatible cycle rotations without future-route selection;
- independently generated the declared 63-column topology design;
- merged the nine raw causal controls onto all 216,438 frozen OOF rows;
- reconstructed the unique OOF row count and inverse-overlap weight;
- sealed the failed support decision and disabled fitting, prediction, later
  scoring, and source attribution;
- compared pre/post hashes of parent decisions and previously saved shadow
  snapshots.

The topology design columns are stored as raw normalized expectations.
Centroid columns have not been multiplied by their planned `0.5` model feature
scale because no model matrix was built.

## What the runner did not do

- No logistic-regression or other model was fitted.
- No `qroute_topology`, `qcycle_main`, or `qcycle_state` prediction was made.
- Sealed `qcontext` and `qfull` predictions were not reinterpreted as a V2
  result.
- No proper-loss, calibration, bootstrap, rotation-performance, or source-
  attribution gate was opened.
- No later-period panel path exists in the runner.
- No prospective-shadow tree was traversed, read, or written.
- No parent grade changed; all twenty remain `unqualified`.

This is a support stop, not evidence that topology, cycle identity, state
rotation, or history does or does not explain the pooled signal.

## Static topology reconstruction

The pre-fit design follows the frozen representation exactly:

- next-state distribution: width 8;
- route-state composition: width 8;
- transition-length one-hot: width 3;
- next-state centroid expectation: width 14;
- route-composition centroid expectation: width 14;
- next-minus-current centroid: width 14;
- rotation ambiguity and normalized entropy: width 2.

Total topology width is 63. Compatible rotations are deduplicated and uniformly
averaged using only the frozen cycle and current filtered state. No realised
future path chooses a rotation.

## Integrity result

The output bundle records:

- `model_fit_performed: false`;
- `prediction_generated: false`;
- `support_pass: false`;
- `later_scoring_authorized: false`;
- `source_attribution_permitted: false`;
- `parent_grade_changed: false`;
- `live_shadow_tree_read: false`;
- `live_shadow_tree_written: false`.

The predecessor's saved aggregate-shadow snapshots still agree at tree SHA-256
`38e90e9db3ae2974db2f6726bb69dfadde0410c2b200e7ce9e080fcbd22bc267`,
with zero ledger rows and `outcomes_opened: false`. These are previously saved
snapshots; V2 did not inspect the live shadow.

## Artifacts

Runner:

`work/run_loop_quality_feature_ablation_v2.py`

Focused tests:

`work/tests/test_loop_quality_feature_ablation_v2_stop.py`

Independent audit and its tests:

- `work/audit_loop_quality_feature_ablation_v2.py`;
- `work/tests/test_loop_quality_feature_ablation_v2_audit.py`.

Ephemeral artifact root:

`/private/tmp/stocker_loop_quality_feature_ablation_v2_20260710`

Key files include:

- `fit_complete.json`;
- `stop_reason.json`;
- `support_audit.csv` and `support_audit.json`;
- `oof_design_rows_2024.parquet`;
- `rotation_mapping.csv`;
- `topology_feature_manifest.json`;
- `planned_gate_manifest.json`;
- `fit_source_hashes.json`;
- `parent_integrity_snapshot.json`;
- `pre_score_audit.json` and `independent_artifact_audit.json`;
- `summary.json`.

No model-parameter or prediction artifact exists.

## Validation

- Frozen contract SHA-256:
  `33d109a1bcc7ee58fb5ee65a5a5c1075a233baa07d50b1219db8358af22f4728`.
- Runner self-tests: 7/7 passed.
- Focused tests: 7/7 passed.
- Independent audit tests: 10/10 passed.
- Independent pre-score audit: 57/57 checks passed.
- Independent final artifact audit: 57/57 checks passed.
- Full workspace test suite: 75/75 passed.
- OOF design rows: 216,438 with exact parent row order.
- Realised rows: 15,584.
- Unique effective weight: 14,167.0.
- Independently reconstructed topology values: maximum absolute error `0.0`.
- Rotation reconstruction: 44 cycle-state units and 45 deduplicated routes.
- Frozen parent final grades: 20 `unqualified` before and after.

## Required next step

The V2 support threshold cannot be amended after this stop. Continuing requires
a separately frozen V3 contract before any model fit.

A principled V3 correction should define support in unique-cohort units rather
than repeated outcome cells. The recommended proposal is:

- minimum unique inverse-overlap OOF weight: 10,000;
- minimum weight in each represented OOF quarter: 5,000;
- minimum OOF sessions: 100;
- minimum stocks: 18;
- minimum per-stock unique effective weight: 50.

The observed support-only values are 14,167 total weight, 7,635 and 6,532 by
quarter, 128 sessions, 22 stocks, and minimum per-stock weight 93. These
criteria use only cohort structure, not outcome performance.

The tentative 15,000 realised-row gate may be retained as a reconstruction
integrity check—the observed count is 15,584—but it should not be treated as an
independent sample-size gate because overlapping cycle rows share anchors. All
other V2 fields and gates should remain semantically identical in V3.

The `/private/tmp` bundle is ephemeral and should be archived before reboot if
exact replay without reconstruction is required.
