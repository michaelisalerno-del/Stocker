# Frozen Causal M1C Low-Movement Veto and Short-Premium Readiness Screen V0

This retrospective experiment asks whether the frozen causal `M1C` bottom
probability tail identifies stock/checkpoint observations whose subsequent
underlying movement remains below the exact previous-close ATM-IV expectation.
The binding population is the 2024-frozen bottom 10% tail and the binding
horizon is 15 minutes. Bottom 5% and 20% tails and 5-, 10-, 30-, and 60-minute
horizons are fixed secondary diagnostics.

The second question is range containment. It measures underlying-stock path
excursions against one-, 1.5-, and two-sigma IV boundaries. It does not model
option strikes, premiums, Greeks, fills, or option P&L. A passing readiness gate
can recommend only prospective shadow recording of defined-risk structures.
It cannot authorise naked options, paper orders, live orders, or deployment.

`M1C` is reconstructed from the Stock-Local Directional Archetype Screen V0:
unchanged previous-close Group O plus the frozen causally valid Group I fields.
Signed pressure, tension, peer-normalised Group I fields, and every numerical
descendant of the future-filtered peer slate remain excluded.

Run the retrospective screen and its independent auditor with the repository
environment:

```bash
uv run python research/options-feasibility/20260727-m1c-low-movement-short-premium-v0/run_screen_v0.py --run
uv run python research/options-feasibility/20260727-m1c-low-movement-short-premium-v0/audit_screen_v0.py --audit
```

The runner reuses byte-identical local causal-panel, exact-date EODHD options,
and completed five-minute-bar artifacts. It makes no network or broker calls
and rejects any session at or after `2026-01-01` before outcome construction.

Primary artifacts and the report are in `artifacts/primary/`. Large derived
parquets, resampling identities, and the full probability comparison remain
local research evidence and are intentionally not committed.
