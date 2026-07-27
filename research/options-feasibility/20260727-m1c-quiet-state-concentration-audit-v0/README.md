# M1C Quiet-State Concentration Audit V0

This experiment explains the stress-month and surprise-mover concentration
that blocked `Frozen Causal M1C Low-Movement Veto and Short-Premium Readiness
Screen V0`.

It reconstructs the frozen bottom-10% checkpoint tail exactly, then compares
raw checkpoint rows with quiet-state runs, frozen fresh quiet episodes, and one
observation per stock-session.  These representations are explanatory only:
the original 35% checkpoint month-share gate and the original decision remain
unchanged.

Run the retrospective audit with the repository research environment:

```bash
uv run python research/options-feasibility/20260727-m1c-quiet-state-concentration-audit-v0/run_audit_v0.py --run
```

The runner reads only opened historical evidence ending on `2025-12-31`.  It
makes no network or broker call and exposes no order, account, or position
surface.

Primary machine-readable outputs are under `artifacts/primary/`; at most three
descriptive plots are under `reports/`.

