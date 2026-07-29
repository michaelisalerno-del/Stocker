# M1C Tail Phase V1

This is a narrow, preregistered structural assessment of the frozen M1C
top-five-percent movement tail. It does not fit a direction model, alter M1C,
alter A1, redefine fresh episodes, calculate option P&L, or permit order
routing.

## Canonical code paths

- Frozen M1C scoring and the exact `0.488333710794033` threshold:
  `packages/stocker_prospective/src/stocker_prospective/frozen_m1c.py`.
- Frozen checkpoint grid and stock-local causal M1C inputs:
  `packages/stocker_prospective/src/stocker_prospective/m1c_features.py`.
- Existing fresh-episode runtime:
  `FreshEpisodeTracker` in `frozen_m1c.py`.
- Existing retrospective fresh-episode definition:
  `construct_fresh_episodes` in
  `packages/stocker_research/src/stocker_research/stock_local_directional_archetypes_v0.py`.
- Previous-close IV scaling and canonical 10/15-minute outcomes:
  `packages/stocker_research/src/stocker_research/m1c_low_movement_v0.py`.
- Frozen A1 runtime and T-1 feature builder:
  `packages/stocker_prospective/src/stocker_prospective/direction.py` and
  `direction_features.py`.
- Prospective M1C records:
  `recorder_v0.py`, `recorder_repository.py`, and migration
  `0011_m1c_tail_phase_v1.sql`.
- Chronology and protected-data guards:
  this runner, `tail_phase_v1.py`, and the hash-verified loader in the
  20260726 stock-local archetype study.

The 20260726 implementation is authoritative because its artifacts are
committed, its no-fit prospective runtimes reproduce its known rows in tests,
and the later recorder commits bind directly to those artifact schemas.

## Run

```bash
uv run python research/directional-readiness/20260728-m1c-tail-phase-v1/run_tail_phase_v1.py
```

The runner calls only the prior study's hash-verified input loader. It never
calls that study's fitting phase. All outcomes are rejected before
`2026-01-01` can enter memory.
