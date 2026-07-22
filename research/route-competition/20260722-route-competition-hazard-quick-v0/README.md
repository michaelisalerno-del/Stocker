# Route-Competition Completion-Hazard Quick Screen V0

This directory contains a bounded retrospective screen of whether causal registered-prefix
competition adds information about any registered-loop completion in the next three completed
five-minute bars. It uses exactly eight checkpoints, two primary models, fifteen fixed-prediction
session-bootstrap draws, and three route-bundle null refits.

The screen is research-only and structural. It does not open returns, direction, P&L, entries,
exits, accounts, orders, broker integration, deployment, or strategy promotion.

Run from the repository root:

```bash
rtk uv run python research/route-competition/20260722-route-competition-hazard-quick-v0/run_screen_v0.py
rtk uv run python research/route-competition/20260722-route-competition-hazard-quick-v0/audit_screen_v0.py
```

Focused validation:

```bash
rtk uv run pytest research/route-competition/20260722-route-competition-hazard-quick-v0/tests/test_screen_v0.py
```
