# Movement-Conditioned Regime-Path Probability Chain V0

Retrospective research-only feasibility screen. Scientific status: `representation_specific_feasibility_evidence`. This is not prospective validation, a strategy, achieved P&L, or executable net-edge evidence.

## Safety

`research_only=true`, `feasibility_screen=true`, `execution_enabled=false`, `order_placement=disabled`, `broker_integration_required=false`, `strategy_promotion=false`, `production_runtime_modified=false`.

## Population and support

- Rows: 6294
- Sessions: 158
- Stocks: 20
- Actual large moves: 1364
- Transition bursts: 5323
- Short closures: 3081 (secondary_sufficient_support)
- Clock-12 q75: 221.446493 bps
- Clock-36 q75: 168.087466 bps

## Layer comparisons

- P1 versus P0: Brier improvement 0.00202044; log-loss improvement 0.00472939; AUC 0.662383 -> 0.688834.
- B1 versus B0: Brier improvement 0.00081408; log-loss improvement 0.00285313; AUC 0.874756 -> 0.878609.
- C1 versus C0 (secondary): Brier improvement 0.00364271; log-loss improvement 0.00337218; AUC 0.669032 -> 0.682388.
- D1 versus D0: Brier improvement -0.00151047; log-loss improvement -0.00312359; AUC 0.510415 -> 0.495521.

## Chain-ranking diagnostic

- Mean Spearman: observable -0.008854; path -0.011906.
- Mean top-one minus slate median: observable 14.504060 bps; path 7.568981 bps.
- These are ranking diagnostics, not expected or achieved P&L.

## Nulls, concentration, and delayed reference

- B1 real improvement percentile under 100 session shifts: 1.000.
- D1 real improvement percentile under 100 session shifts: 0.930.
- Maximum stock decision-row fraction: 0.0500.
- Maximum stock top-one fraction: 0.1492.
- observable_chain delayed top-one reference at 0 bps synthetic friction: 26.816640 bps.
- observable_chain delayed top-one reference at 20 bps synthetic friction: 6.816640 bps.
- path_chain delayed top-one reference at 0 bps synthetic friction: 19.524292 bps.
- path_chain delayed top-one reference at 20 bps synthetic friction: -0.475708 bps.

## Decision

`structural_increment_without_directional_value`

Movement predictability, structural-path predictability, directional predictability, gross economic association, and executable net edge remain separate conclusions. This V0 cannot establish the last conclusion.

## Reproducibility and audit
- Decision status: `final`.
- Exact rerun: passed.
- Independent audit: passed.
