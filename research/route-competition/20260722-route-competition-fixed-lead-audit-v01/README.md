# Route-Competition Fixed-Lead Audit Quick Screen V0.1

This retrospective structural audit separates next-bar registered-loop completion from clean
two-to-three-bar advance warning. It reuses the frozen V0 decision panel and exact H0/H1 feature
surfaces, then excludes lead-one completions and every checkpoint with a registered prefix one
canonical transition from completion.

The screen is research-only. It does not access returns, direction, options outcomes, P&L,
entries, exits, accounts, orders, brokers, deployment, or strategy-promotion surfaces.

Run from the repository root:

```bash
rtk uv run python research/route-competition/20260722-route-competition-fixed-lead-audit-v01/run_screen_v01.py
rtk uv run python research/route-competition/20260722-route-competition-fixed-lead-audit-v01/audit_screen_v01.py
```

Focused validation:

```bash
rtk uv run pytest research/route-competition/20260722-route-competition-fixed-lead-audit-v01/tests/test_screen_v01.py
```
