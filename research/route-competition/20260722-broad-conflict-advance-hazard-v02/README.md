# Broad-Conflict Advance-Hazard Dense-Checkpoint Quick Screen V0.2

This retrospective, observable, structural screen tests whether the predecessor's frozen route
competition bundle—and specifically its frozen `BROAD_CONFLICT` state—adds information about a
registered-loop completion two or three completed five-minute bars ahead.

It reuses the audited V0 causal state trace and structural ledger, adds only the seven missing even
checkpoints, and excludes both next-bar completions and every row with a prefix already one
transition from completion. Historical activity remains the EODHD historical activity proxy; it
is not described as confirmed exchange volume.

The experiment is research-only. It does not access or modify returns, direction, options,
accounts, positions, orders, entries, exits, brokers, portfolio sizing, deployment, or production
runtime.

Run from the repository root:

```bash
rtk .venv/bin/python research/route-competition/20260722-broad-conflict-advance-hazard-v02/run_screen_v02.py
rtk .venv/bin/python research/route-competition/20260722-broad-conflict-advance-hazard-v02/audit_screen_v02.py
```

The audit command independently reads every timestamp surface, reconstructs chronology labels,
support, and every decision gate from frozen predictions and result tables, fail-closes
`decision.json` on a discrepancy or audit exception, and synchronizes both report copies to the
audited decision.

Focused validation:

```bash
rtk .venv/bin/pytest -q research/route-competition/20260722-broad-conflict-advance-hazard-v02/tests
```
