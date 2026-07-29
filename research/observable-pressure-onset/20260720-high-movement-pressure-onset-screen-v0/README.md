# High-Movement Pressure-Onset Screen V0

This directory contains a bounded retrospective feasibility screen asking whether
causal changes in price efficiency, cohort-relative strength, range, and historical
activity reveal directional pressure inside an observably high remaining-movement
population.

It is research-only and observable-only. It is not prospective validation, a
strategy, achieved P&L, or evidence of executable net edge. Execution, orders,
broker integration, position sizing, and production runtime changes are disabled.
The delayed 30-minute and remaining-session calculations are gross economic-reference
diagnostics only and cannot rescue failed probability gates.

The fixed development interval is 2024-01-01 through 2024-12-31. The fixed
assessment interval is 2025-01-01 through 2025-08-22. Source reads apply symbol and
date predicates before materialisation; market rows dated 2025-08-23 or later are
forbidden.

Run the complete primary screen, exact rerun, and independent audit from the
repository root:

```bash
uv run python research/observable-pressure-onset/20260720-high-movement-pressure-onset-screen-v0/run_screen_v0.py
```

Run the independent auditor alone:

```bash
uv run python research/observable-pressure-onset/20260720-high-movement-pressure-onset-screen-v0/audit_screen_v0.py \
  --artifacts research/observable-pressure-onset/20260720-high-movement-pressure-onset-screen-v0/artifacts/primary
```

Primary artifacts are written under `artifacts/primary`, the deterministic rerun
under `artifacts/exact_rerun`, and the narrative result under `reports/report.md`.

The frozen run stopped with `blocked_insufficient_pressure_onset_support` before
fitting A0–A3 or D0–D3. The 2025 high-movement pocket had 1,560 rows across 153
sessions and 20 stocks, but its thinnest slate had only one admitted candidate
versus the fixed minimum of ten, and QBTS contributed 10.961538% of rows versus
the fixed 10% ceiling. The population and thresholds were not changed. The
predecessor reconstruction, exact rerun, and standalone audit all passed.
