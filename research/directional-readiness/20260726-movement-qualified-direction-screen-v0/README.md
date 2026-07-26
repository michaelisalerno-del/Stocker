# Movement-Qualified Directional Readiness Quick Screen V0

This research-only retrospective screen keeps the validated M1 movement model
and its `0.49588519865576763` threshold frozen. It tests a separate
second-stage `CALL` / `PUT` / `ABSTAIN` direction layer on fresh movement
episodes.

- Direction development: 2024-01-01 through 2024-12-31.
- Retrospective assessment: 2025-01-01 through 2025-08-22.
- Excluded opened movement holdout: 2025-09-01 through 2025-12-31.
- Protected and never materialised: 2026-01-01 onward.
- Binding direction horizon: ten minutes from the next-bar open.

The experiment studies underlying-stock direction only. It does not calculate
option P&L, use intraday option quotes, access a broker, place orders, modify
production runtime, or promote a strategy.

The frozen overall decision is `no_incremental_directional_signal`. Both the
independent audit and exact deterministic rebuild passed. See
[`reports/report.md`](reports/report.md) for the concise research result and
[`artifacts/primary/`](artifacts/primary/) for the complete evidence bundle.

Run only the scoped workflow:

```bash
uv run python research/directional-readiness/20260726-movement-qualified-direction-screen-v0/run_screen_v0.py
uv run python research/directional-readiness/20260726-movement-qualified-direction-screen-v0/audit_screen_v0.py
```
