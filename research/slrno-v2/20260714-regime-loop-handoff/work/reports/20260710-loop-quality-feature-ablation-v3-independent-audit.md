# Loop-quality feature ablation V3 — independent audit

Date: 2026-07-10

`research_only: true`  
`live_ordering_enabled: false`  
`order_placement: disabled`

## Outcome

The V3 fit and the two permitted scoring periods were independently reproduced without importing the production V3 runner.

- Pre-score audit: **55/55 checks passed**.
- Post-score audit: **44/44 checks passed**.
- Independently refitted causal 2024 OOF models: **108/108**, maximum probability error **0.0**.
- Independently refitted full-2024 models: **18/18**, maximum scaler/coefficient/intercept error **0.0**.
- Independently reconstructed 2025 predictions: maximum error **4.440892098500626e-16**.
- Independently reconstructed backward-2023 predictions: maximum error **3.885780586188048e-16**.
- Saved cell metrics, calibration, moving-block intervals, quarter checks, stock-deletion checks, rotation diagnostics, two-axis cycle diagnostics, gates, and demotion-only transfer attribution all reproduced within the frozen numerical tolerance.

The saved final attribution is **`no_reference_signal`**. This means the sealed `qfull` model did not pass the frozen robust reference gate against `qcontext`; therefore V3 cannot make a reliable source-attribution claim about topology, cycle identity, current-state rotation, or the history token. It does not overturn the earlier movement/range hypothesis, and it does not establish high-performance loops.

## Integrity checks

- V3 differs from stopped V2 only in the declared V3 lineage/execution fields and unique-cohort support semantics.
- Unique OOF support independently reconstructed as 14,167 effective inverse-overlap weight, 15,584 realized rows, 128 sessions, and 22 stocks.
- The 44 cycle/current-state topology units, compatible-rotation deduplication, uniform mixtures, 63 topology values, entropy groups, and centroid normalization were reconstructed independently.
- Sealed `qcontext` and `qfull` probabilities replayed exactly.
- The fit call graph did not reach later-period outcome paths.
- The scoring call graph enforced the successful independent pre-score audit before loading 2025 or 2023.
- No 2026 outcome, live prospective shadow, ledger, broker, order, position, P&L, or deployment path was read or changed.
- Frozen parent grades remained unchanged.
- 2025 remains development data and 2023 remains backward portability data; neither is prospective validation.

## Verification artifacts

- Pre-score audit: `/private/tmp/stocker_loop_quality_feature_ablation_v3_20260710/pre_score_audit.json`
  - SHA-256: `f4c79f0935c8e65eeb5235879d007b5213b882dc592efdde19f38aa172eca53b`
- Post-score audit: `/private/tmp/stocker_loop_quality_feature_ablation_v3_20260710/independent_artifact_audit.json`
  - SHA-256: `b31c2d5c1b367f1ef93d4b24f00e6ee0c94c7cf64a63a8cc3ad975d9bd40ab5f`
- Scoring freeze: `/private/tmp/stocker_loop_quality_feature_ablation_v3_20260710/scoring_complete.json`
  - SHA-256: `c4a168100f122f7421dd1e4ad0f55cca9dd9c98dde86401c513351f0880f932a`

The `/private/tmp` artifacts are ephemeral and should be archived separately if they must survive a reboot.

## Tests

The V3 production and independent-audit test files pass together: **19 passed**.

```text
rtk python3 -m pytest \
  work/tests/test_loop_quality_feature_ablation_v3_audit.py \
  work/tests/test_loop_quality_feature_ablation_v3.py -q
```
