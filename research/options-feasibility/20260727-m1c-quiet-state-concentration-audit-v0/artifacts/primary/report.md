# M1C Quiet-State Concentration Audit V0

**RESEARCH ONLY — RECORD ONLY — NO ORDERS**

The original frozen decision remains
`blocked_insufficient_low_tail_support`. No historical threshold, support gate, or decision was
changed, and no retrospective gate relaxation is allowed.

## Binding answer

The failed stress month was **2025-10**:
529/1426 frozen
bottom-10 checkpoint rows, or **37.096774%**.
Its source exposure was 27.81%, while its
within-month bottom-tail incidence was 10.49%.
The exact explanation is
`month_concentration_has_multiple_causes`: October had the largest eligible
source exposure, the highest low-tail incidence, and repeated checkpoint
persistence supplied the final increment above the frozen 35% limit.

| Month | XNYS sessions | Eligible rows | Source exposure | Tail incidence | Tail composition | Fresh share |
|---|---:|---:|---:|---:|---:|---:|
| 2025-09 | 21 | 4313 | 23.781% | 7.953% | 24.053% | 28.466% |
| 2025-10 | 23 | 5043 | 27.807% | 10.490% | 37.097% | 33.087% |
| 2025-11 | 19 | 4104 | 22.629% | 4.751% | 13.675% | 13.678% |
| 2025-12 | 22 | 4676 | 25.783% | 7.678% | 25.175% | 24.769% |

For the failed month, composition falls from
37.10% at raw
checkpoints to 33.56% for
quiet runs, 33.09%
for frozen fresh episodes, and
33.05% with one
observation per stock-session. These are explanatory views and do not replace
the checkpoint support gate.

## Surprise movers

At 1.5σ, the binding stress fresh-episode population contains
11 original rows and
11 clustered events.
At 2.0σ it contains 2 original
rows and 2 events.
The binding 1.5σ maximum month share is
63.64%; maximum stock share is
27.27%; maximum stock-month share is
27.27%. Event clustering removed
0 rows from the binding
fresh population. One event changes the share by
9.09%; one event is the difference between
passing and failing. The exact explanation is
`surprise_concentration_is_small_count_fragile`.

## Descriptive sensitivities

Equal-month weighting reports a remains-below-IV rate of
91.36%, NPV lift of
+18.01%, and maximum month
concentration of
34.03%.
These weighted results are descriptive only.

| Omitted month | Rows | Remains below IV | NPV lift | Mean residual | 1.5σ breach | 2.0σ breach |
|---|---:|---:|---:|---:|---:|---:|
| 2025-09 | 1083 | 93.07% | +20.19% | -0.005361 | 0.74% | 0.28% |
| 2025-10 | 897 | 90.08% | +17.48% | -0.004176 | 1.34% | 0.22% |
| 2025-11 | 1231 | 91.47% | +16.56% | -0.004801 | 1.38% | 0.32% |
| 2025-12 | 1067 | 91.19% | +17.65% | -0.005322 | 1.31% | 0.28% |

## Reproducibility and claims boundary

- Independent audit passed: `True`.
- Retrospective deterministic replay passed: `True`.
- Maximum floating difference: `0`.
- Protected historical start: `2026-01-01`; protected rows read: `0`.
- No option P&L, order, account, position, paper-trading, or live-trading path
  was used.

Plots: `reports/01_stress_month_exposure_vs_tail_composition.png`, `reports/02_raw_vs_clustered_surprise_concentration.png`, `reports/03_leave_one_month_out.png`.

This audit does not claim that either historical gate passed, does not claim
option profitability or realistic fill expectancy, and does not create a
replacement validation decision.
