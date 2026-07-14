# Frozen per-loop movement-quality shadow harness

## Decision

The separate per-loop quality shadow is finalized in a deliberately dormant state. No frozen cycle is eligible to surface.

This is the direct consequence of the predeclared grade logic, not a discretionary filter added after seeing results. All 20 cycles have an `unqualified` global grade in causal July–December 2024 OOF scoring. Cycle 09 was `good_movement_quality` at 6 and 12 bars but `unqualified` at 24 bars, so its global grade is still `unqualified`. The contract forbids promotion after that provisional freeze: 2025 development and backward-2023 portability may only preserve or demote a provisional grade. They cannot create a globally good/high cycle.

Sealed scoring then confirmed that all 20 cycles are `unqualified` in 2025 development, backward-2023 portability, and the final minimum grade. The frozen final artifacts are `final_cycle_tiers.csv` (SHA-256 `2d4e4bd2ef26db396244fe7cd20a8485aba1814eaeacf5326916823225d7c598`) and `gates.json` (SHA-256 `0e64c9a9dee02b1860117078a811387f64ec6324e7edb2ec2b4b2104ee3b7637`).

The independent post-score audit passed 48 of 48 checks. Its exact archived artifact has SHA-256 `2f969f2bb751da5d781227feb8edb0af5d1166cfb3a09758806cdb6f94c713a7`. It independently reconstructed the 2025 and 2023 labels, features, structural and conditional probabilities, chain-rule outputs, aggregate metrics, calibration, support, structural gates, every period/cycle grade, and final minimum-tier decision. It also confirmed `research_only: true`, `live_ordering_enabled: false`, `order_placement: disabled`, no 2026 rows, no execution surface, zero qualified cycles, an empty aggregate ledger, and unopened outcomes.

Therefore:

- eligible cycles: 0;
- surfaced predictions: 0;
- prediction ledger: empty and append-only;
- outcomes opened: false;
- prospective performance claim: none.

`high_movement_quality` and `good_movement_quality` refer only to conditional absolute-movement and future-range evidence. They do not mean trading performance, direction, signed return, P&L, economic edge, or tradability.

## Separation from the existing aggregate shadow

This runtime lives at `work/shadow_validation/frozen_loop_quality_shadow_v1`. It does not write into `work/shadow_validation/frozen_loop_movement_shadow_v1`.

Before and after initialization and final certification, the existing aggregate shadow is checked against a content-only snapshot of every file. Its existing prediction ledger remains empty with SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Two protected hashes appear in the artifacts and intentionally differ because their domains differ:

- `ffc5f1ccb572e120b8001cb0d9a93cfb4d37946e53c2ac39c3d6970fb8e8d766` is the canonical hash of 22 files inside the aggregate runtime tree only, recording relative path, byte size, and file SHA-256 while excluding directories and workspace source files.
- `38e90e9db3ae2974db2f6726bb69dfadde0410c2b200e7ce9e080fcbd22bc267` is the quality scorer's broader 40-entry protected-path hash. It also covers the workspace contract, manifest, core, runner, tests, reports, and runtime directory entries, including kind, mode, and size.

Both snapshots match their own frozen baseline. They are not hashes of the same serialization and therefore should not match each other.

## Fail-closed interface

The harness has `init`, `status`, `issue`, and `self-test` commands. It intentionally has no outcome-evaluation command.

The `issue` path verifies the runtime and checks the frozen eligible-cycle set before reading a candidate batch. Because the set is empty, it exits without opening the candidate. A non-empty batch cannot enter the ledger.

The frozen prediction schema keeps these quantities distinct for every target and horizon:

- structural loop probability `s(i,c)`;
- conditional movement-quality probabilities `q75(i,c)` and `q90(i,c)`;
- joint chain-rule probabilities `j75(i,c)=s(i,c)q75(i,c)` and `j90(i,c)=s(i,c)q90(i,c)`.

The validator enforces unit-interval bounds, `q90 <= q75`, `j90 <= j75`, and the two rowwise chain-rule identities. It forbids outcome, direction, signed-return, P&L, broker, order, position, cost, spread, and slippage fields. Probabilities may not be summed across overlapping cycles.

## What would be required to activate a quality shadow

This contract cannot be mutated to activate a cycle. Activation would require a separate, newly frozen development candidate and prospective contract. The current frozen 20-cycle quality hypothesis has no eligible member under its own rules.

Safety labels:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
