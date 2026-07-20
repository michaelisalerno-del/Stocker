# Movement × Closure-History Joint Increment V0.1

Research-only, retrospective, representation-specific feasibility screen. Execution and order placement are disabled. No direction, payoff, or executable edge was tested.

## Population and join

- Common materialised period: 2024-07-01 through 2025-08-22.
- Exact joined rows across both years: 8614.
- 2025 assessment joined rows: 4449 (157 sessions, 20 stocks).
- Immediate closures: 1401.
- Large movements: 795.
- Joint positives: 247.
- Duplicate later clocks removed: 68.

## Results

- Direct M5 vs M2: Brier improvement -0.00026176; log-loss improvement -0.00017341.
- A3 vs A2 closure: Brier 0.00022398; log loss 0.00067843; 90% Brier interval [-0.00002052, 0.00050407]; null not run: exact membership blocks cannot all be shifted.
- B1 vs B0 movement: Brier 0.00000278; log loss 0.00000097; 90% Brier interval [-0.00003921, 0.00004197]; null not run: exact membership blocks cannot all be shifted.
- C1 vs product: Brier -0.00025022; log loss -0.00199769; 90% Brier interval [-0.00069673, 0.00016886].

## Monthly stability

- A3 vs A2 positive-Brier months: 5/8.
- B1 vs B0 positive-Brier months: 3/8.
- C1 vs C0 positive-Brier months: 4/8.

## Decision

`blocked_join_semantics_failure`

The preregistered session-shift null is not identified on this irregular joined panel: singleton exact-stock membership blocks cannot receive a non-identity whole-session shift. The screen therefore fails closed.

Arm A pass: `False`. Arm B pass: `False`. Arm C pass: `False`.

The optional A4, B2, and C2 interaction sensitivities were not fitted because the five required baseline/candidate stackers take precedence under the explicit six-model cap.

This result does not claim a directional signal, strategy return, economic payoff, or executable system.
