# Hidden-Loop Competing Routes and Registered-Loop Recurrence Quick Screen V0

This is a fast, bounded, retrospective structural experiment. It tests four fixed
hidden/registered precursor relationships and a three-model target-specific competing-route
ladder using causal updates through at most six completed bars after the frozen opening
decision.

The screen uses only the audited 20-stock cohort, development dates from 2024-01-01 through
2024-12-31, and assessment dates from 2025-01-01 through 2025-08-22. It fails closed if any
row on or after 2025-08-23 is materialised.

Economic outcomes, return direction, entries, exits, trading rules, execution, accounts,
positions, orders, broker integration, sizing, and deployment remain outside the experiment.

Run the bounded screen:

```bash
rtk uv run python research/registered-loop-routes/20260722-hidden-loop-competing-routes-v0/run_screen_v0.py
```

Run the independent artifact audit:

```bash
rtk uv run python research/registered-loop-routes/20260722-hidden-loop-competing-routes-v0/audit_screen_v0.py
```

Run focused tests only:

```bash
rtk uv run pytest research/registered-loop-routes/20260722-hidden-loop-competing-routes-v0/tests/test_screen_v0.py
```
