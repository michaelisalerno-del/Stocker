# One-Minute Activity–Price Lead Screen V0

This bounded experiment asks whether one-minute historical activity patterns add to
price sequencing among stocks already nominated by the frozen five-minute
high-movement model.

The frozen 20-stock EODHD one-minute inputs were acquired locally in a separate,
user-authorized preparation step. The runner itself has no network or credential
path. It reads only bounded local inputs, reconstructs the unchanged frozen
nomination population, proves the one-minute bar-start convention by independent
cross-timeframe OHLC alignment, emits exact symbol/month/session/minute coverage,
and applies the fixed probability ladders, session bootstrap, bundled-activity null,
and delayed economic-reference diagnostic.

The scientific result is written to `artifacts/primary/decision.json`. The complete
run is repeated byte-for-byte under `artifacts/exact_rerun` and checked by a
standalone auditor that does not import the runner.

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
Downloaded raw and processed market data remain outside Git.
