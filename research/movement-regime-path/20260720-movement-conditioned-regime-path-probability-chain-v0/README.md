# Movement-Conditioned Regime-Path Probability Chain V0

This lineage implements a tightly bounded retrospective feasibility screen. It asks
whether observable movement forecasts increment a frozen repaired regime-path model,
and whether those structural probabilities increment conditional direction. It does
not test or promote a trading strategy.

Scientific status: `representation_specific_feasibility_evidence`. A positive structural
result is not directional evidence, gross association is not executable net edge, and
none of the outputs are achieved-P&L estimates.

Safety is fixed: `research_only=true`, `feasibility_screen=true`,
`execution_enabled=false`, `order_placement=disabled`,
`broker_integration_required=false`, `strategy_promotion=false`, and
`production_runtime_modified=false`. The runner has no broker, account, position,
order, sizing, portfolio-risk, server, or deployment dependency.

## Frozen scope

- The 20-stock Directional Signature Atlas V1 cohort.
- Decision ordinals 12 and 36 only (10:30 and 12:30 New York, zero-based bars).
- One 24-bar horizon.
- The repaired deterministic K=8 full-refit representation from Right-Censored
  Regime Refit V2, scored from its frozen parameters without refitting.
- 2024 development and chronology-safe stacking; pre-2025-08-23 retrospective
  portability scoring only.
- Fixed L2 logistic and Ridge models; 500 paired session bootstraps and 100
  whole-session circular nulls in one process.

## Commands

Run the complete primary screen from the repository root:

```bash
PYTHONPATH=packages/stocker_research/src \
python research/movement-regime-path/20260720-movement-conditioned-regime-path-probability-chain-v0/run_screen_v0.py
```

The default provider root is discovered as `StockerLocal/data/processed` below the
current user's home directory. It can be overridden without recording an absolute path:

```bash
PYTHONPATH=packages/stocker_research/src \
python research/movement-regime-path/20260720-movement-conditioned-regime-path-probability-chain-v0/run_screen_v0.py \
  --provider-root /path/to/source=eodhd/instrument_type=stock
```

Exact rerun and audit:

```bash
PYTHONPATH=packages/stocker_research/src \
python research/movement-regime-path/20260720-movement-conditioned-regime-path-probability-chain-v0/run_screen_v0.py \
  --exact-rerun --audit
```

`--output` selects another artifact directory. `--max-rows` is available only for
bounded smoke checks. It retains complete slates round-robin across every available
calendar month, so the limit must fit at least one complete slate per month. Any such
output is marked `scientific_run=false` and cannot pass the scientific support gate.

Run the independent auditor directly:

```bash
PYTHONPATH=packages/stocker_research/src \
python research/movement-regime-path/20260720-movement-conditioned-regime-path-probability-chain-v0/audit_screen_v0.py \
  --artifacts research/movement-regime-path/20260720-movement-conditioned-regime-path-probability-chain-v0/artifacts/primary
```

The auditor does not import the runner or refit the historical regime model.
