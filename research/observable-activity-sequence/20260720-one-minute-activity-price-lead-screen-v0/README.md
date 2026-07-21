# One-Minute Activity–Price Lead Screen V0

This bounded experiment asks whether one-minute historical activity patterns add to
price sequencing among stocks already nominated by the frozen five-minute
high-movement model.

The local-data availability gate failed closed. No one-minute EODHD source files are
present for the frozen 20-stock cohort, so timestamp semantics, feature construction,
outcomes, models, bootstrap, nulls, and economic-reference diagnostics were not
opened. The scientific decision is:

`blocked_one_minute_history_unavailable`

The runner still reconstructs the frozen nomination population, emits exact
symbol/month/session/minute-of-session missingness, proves that no protected row was
opened, performs a deterministic exact rerun, and invokes an independent auditor.

This is retrospective, research-only, observable-only feasibility work. It is not
prospective validation, achieved P&L, a strategy, or evidence of executable edge.
Execution, order placement, broker integration, strategy promotion, and production
runtime changes are disabled.

Run from the repository root:

```bash
uv run python research/observable-activity-sequence/20260720-one-minute-activity-price-lead-screen-v0/run_screen_v0.py
```

Primary artifacts are under `artifacts/primary`, the deterministic rerun is under
`artifacts/exact_rerun`, and the narrative report is under `reports/report.md`.
