# Regime × Loop Prefix × Behavioural Context Quick Screen V0

This is a retrospective, research-only structural feasibility screen. It asks whether the ten
already-frozen continuous behavioural dimensions add pre-completion information about which
active registered oriented loop prefix completes first within six completed five-minute bars.

The screen reuses, without redesign:

- the completed Observable Behavioural-State Dimensions Screen V0 ledger and scaling;
- the repaired 2024-fitted eight-state V2 causal semi-Markov regime model;
- the frozen 20-entry V2 semantic loop dictionary and active-prefix/first-event semantics.

It uses only decision ordinals 6 and 12, fits only on 2024, assesses only from 2025-01-01 through
2025-08-22, and rejects every row on or after 2025-08-23. It does not open economic outcomes and
has no execution, order-placement, broker, deployment, or strategy-promotion surface.

Run from the repository root with a materialised Git LFS predecessor checkout:

```bash
PYTHONPATH=packages/stocker_research/src python \
  research/regime-loop-behaviour/20260721-regime-loop-behaviour-quick-screen-v0/run_screen_v0.py \
  --materialized-predecessor-repo /path/to/materialized/Stocker
```

The runner creates `artifacts/primary`, independently audits it, repeats the screen into
`artifacts/exact_rerun`, and checks deterministic identity. A failed preregistered support gate
stops before model fitting, bootstrap, and null refits.
