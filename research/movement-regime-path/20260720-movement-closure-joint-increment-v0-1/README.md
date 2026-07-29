# Movement × Closure-History Joint Increment V0.1

This directory contains a bounded retrospective relationship screen combining two
already-frozen probability surfaces. It tests movement predictability, immediate
pair-closure predictability, mutual incremental information, and the joint event only.

It is research-only, a feasibility screen, and specific to the frozen repaired K=8
representation. Execution and order placement are disabled. It is not prospective
validation, a strategy, directional evidence, payoff evidence, or executable-edge
evidence.

## Frozen scope

- Movement predictions: the committed OOF 2024 and frozen pre-2025-08-23 ledgers from
  `Movement-Conditioned Regime-Path Probability Chain V0`.
- Closure predictions: the OOF 2024 and frozen 2025 M2/M5 ledgers from
  `Immediate Regime-Pair Closure History Diagnostic V1`.
- Representation: `CAUSAL_HARD_SEMANTIC`, using the hash-bound raw-to-semantic mapping
  for the same fitted regime model used by the movement experiment.
- Join anchor: the movement feature-availability timestamp, not nearest time.
- Models: fixed C=1 L2 liblinear stackers fitted on 2024 only.
- Uncertainty: 500 paired session-block bootstraps. The preregistered 100-draw
  whole-session null requires exact stock-cross-section preservation within each
  decision ordinal. The joined panel contains singleton session-membership blocks,
  so some sessions cannot be shifted without becoming identity mappings. The runner
  and auditor therefore fail closed with `blocked_join_semantics_failure`; no invalid
  null percentiles are emitted.

The contract's six-model hard cap conflicts with fitting all three optional interaction
sensitivities in addition to the five required baseline/candidate stackers. The primary
models therefore take precedence; A4, B2, and C2 are recorded as not fitted and cannot
rescue a primary result.

Point estimates, calibration metrics, monthly breakdowns, and paired bootstrap
intervals remain descriptive feasibility results. They do not override the blocked
primary decision because the required null construction is not feasible under the
exact joined-population preservation contract.

## Commands

From the repository root, using the research environment:

```bash
PYTHONPATH=packages/stocker_research/src \
python research/movement-regime-path/20260720-movement-closure-joint-increment-v0-1/run_joint_screen_v0_1.py

PYTHONPATH=packages/stocker_research/src \
python research/movement-regime-path/20260720-movement-closure-joint-increment-v0-1/audit_joint_screen_v0_1.py \
  --artifacts research/movement-regime-path/20260720-movement-closure-joint-increment-v0-1/artifacts/primary
```

Run the same runner with `--output .../artifacts/exact_rerun`, audit that directory,
then use `--verify-rerun --reference .../artifacts/primary` to write the deterministic
rerun manifest to both artifact directories.
