# Emotion × Regime-Mix Loop Funnel Quick Screen V0

This is a bounded retrospective structural feasibility experiment. It asks whether the frozen
behavioural dimensions improve prediction of the first semantic loop completed during the next
six five-minute bars, and whether preregistered behavioural × soft-regime interactions improve the
distribution beyond behavioural main effects.

The screen uses only the audited 20-stock cohort, 10:00 and 10:30 America/New_York checkpoints,
2024 development, and 2025-01-01 through 2025-08-22 assessment. It does not require an active loop
prefix. Exact oriented-loop classes are frozen from 2024 before any 2025 metric is calculated.

The experiment is research-only, quick-feasibility-only, structural-only, pre-loop, and
non-deployable. Execution, order placement, broker integration, strategy promotion, and production
runtime changes are disabled. Economic outcomes are not opened.

Run from the repository root with the research environment:

```bash
python research/loop-funnel/20260721-emotion-regime-loop-funnel-quick-v0/run_screen_v0.py
```

The primary artifacts and report are written beneath `artifacts/primary/`; the rendered handoff
report is mirrored to `reports/report.md`. The runner performs the bounded determinism refit and
the lightweight independent audit without rerunning bootstrap or null draws.
