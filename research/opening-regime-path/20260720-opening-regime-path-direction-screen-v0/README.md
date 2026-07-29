# Opening Regime-Path Direction Screen V0

This directory contains a bounded, retrospective feasibility screen asking one
question: do causal opening regimes, transitions, and short closure topology at
the completed 30- and 60-minute checkpoints improve fixed-terminal remaining
movement and direction probabilities beyond observable opening prices?

The experiment is research-only, representation-specific, and not prospective
validation, a strategy, achieved P&L, or executable-edge evidence. Execution,
orders, broker integration, position sizing, and production runtime changes are
disabled. The secondary top-one calculation is a delayed, gross economic-reference
ranking diagnostic only.

The fixed development interval is 2024-01-01 through 2024-12-31. The unchanged
retrospective assessment interval is 2025-01-01 through 2025-08-22. Provider
reads apply date and symbol predicates before materialisation; 2025-08-23 and
later rows are prohibited.

Run the complete primary screen, exact rerun, and independent audit from the
repository root:

```bash
uv run python research/opening-regime-path/20260720-opening-regime-path-direction-screen-v0/run_screen_v0.py
```

Run the independent auditor alone against an existing artifact directory:

```bash
uv run python research/opening-regime-path/20260720-opening-regime-path-direction-screen-v0/audit_screen_v0.py \
  --artifacts research/opening-regime-path/20260720-opening-regime-path-direction-screen-v0/artifacts/primary
```

Primary artifacts are written under `artifacts/primary`, the deterministic
second run under `artifacts/exact_rerun`, and the concise narrative report under
both artifact directories plus `reports/report.md`.
