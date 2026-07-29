# High-Movement Pressure-Onset Screen V0.1

This is the support-semantics repair of High-Movement Pressure-Onset Screen V0.
It reopens only aggregate support counts already observed in V0. No V0 downstream
model, bootstrap, null, or economic-reference result had been fitted or opened.

The repair validates the fixed-clock parent slate before movement admission and
does not require ten stocks to exceed the frozen movement threshold. Singleton
admitted slates remain valid. Every admitted slate receives total model weight
one, using `1 / admitted_stock_count` for each admitted row.

All dates, sources, rows, movement probabilities, thresholds, onset barriers,
labels, causal features, model specifications, evaluation gates, horizons,
frictions, and random seeds remain frozen from V0. The null preserves admission
and permutes the complete pressure bundle among admitted stocks within the same
valid parent slate.

This is retrospective, observable-only feasibility evidence. It is not
prospective validation, a strategy, achieved P&L, or evidence of executable edge.
Execution, orders, broker integration, and production runtime changes are disabled.

Run from the repository root:

```bash
uv run python research/observable-pressure-onset/20260720-high-movement-pressure-onset-screen-v0-1/run_screen_v0_1.py
```

Primary artifacts are written to `artifacts/primary`, the deterministic rerun to
`artifacts/exact_rerun`, and the narrative result to `reports/report.md`.
