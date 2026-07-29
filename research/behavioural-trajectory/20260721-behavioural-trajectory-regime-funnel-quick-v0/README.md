# Behavioural-Trajectory × Regime-Mix Funnel Quick Screen V0

This is a retrospective, observable-only, structural feasibility screen. It asks whether causal
behavioural trajectories improve the frozen three-class coarse structural forecast beyond the
completed Emotion × Regime-Mix Coarse Loop-Family Funnel V0 M2 model.

The screen reads only frozen predecessor artifacts and bounded dates through 2025-08-22. It does
not open economic outcomes, implement trading rules, or modify execution or production runtime.

Run the focused screen with a materialized frozen behavioural component ledger:

```bash
PYTHONPATH=packages/stocker_research/src python \
  research/behavioural-trajectory/20260721-behavioural-trajectory-regime-funnel-quick-v0/run_screen_v0.py \
  --behavioural-component-ledger /path/to/behavioural_component_ledger.parquet
```

The frozen behavioural formula accepts only even completed-bar windows. The preregistered bar-9
anchor is therefore evaluated fail-closed by the support preflight; no alternative split is
substituted.
