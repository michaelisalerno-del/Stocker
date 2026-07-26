# Dense Five-Minute Signed-Pressure Reconstruction and Frozen Directional Rerun V0

This retrospective, research-only experiment asks whether the repository's exact audited
`signed_pressure` primitive can be evaluated at every completed five-minute bar while
remaining causal and reproducing every sparse checkpoint within `1e-12`.

Phase 1 traces the existing formula and upstream dependencies, builds the completed-bar grid,
tests sparse compatibility, performs truncated-history and future-mutation audits, and checks
fresh-episode pressure-window coverage. Phase 2 may run only when Phase 1 returns
`dense_signed_pressure_reconstruction_supported`.

Run the scoped work:

```bash
uv run python research/directional-readiness/20260726-dense-signed-pressure-reconstruction-v0/reconstruct_dense_pressure.py
uv run python research/directional-readiness/20260726-dense-signed-pressure-reconstruction-v0/audit_dense_pressure.py
uv run python research/directional-readiness/20260726-dense-signed-pressure-reconstruction-v0/rerun_frozen_direction_screen.py
```

The existing activity field remains labelled `historical_relative_activity` or activity
proxy. This work does not measure direct order flow, observe institutional accumulation,
calculate option P&L, access a broker, or modify production execution.
