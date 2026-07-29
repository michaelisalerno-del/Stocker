# Opening Regime-Path Direction Screen V0 report

## Boundary and interpretation

This is a retrospective, research-only, representation-specific feasibility
screen. It is not prospective validation, a strategy, achieved P&L, or evidence
of executable net edge. The economic-reference calculation is delayed, gross,
and secondary; it cannot rescue failed probability gates.

## Population

- Development: 9329 rows, 239 sessions,
  20 stocks, 2024-01-01 through 2024-12-31.
- Assessment: 6288 rows, 158 sessions,
  20 stocks, 2025-01-01 through 2025-08-22.
- Development q75 movement thresholds: 30-minute 360.846284 bps;
  60-minute 317.133692 bps.
- Assessment large moves: 1316; opening short
  closure rows: 1238.
- Protected market rows materialised: 0.

## Pooled movement models

| Model | Brier | Log loss | AUC |
|---|---:|---:|---:|
| M0 | 0.168002 | 0.519866 | 0.488948 |
| M1 | 0.157911 | 0.494397 | 0.667605 |
| M2 | 0.158891 | 0.496017 | 0.660344 |
| M3 | 0.159673 | 0.498451 | 0.654775 |

## Pooled direction models among actual large moves

| Model | Brier | Log loss | AUC |
|---|---:|---:|---:|
| M0 | 0.248166 | 0.689475 | 0.509193 |
| M1 | 0.248775 | 0.690685 | 0.527443 |
| M2 | 0.248681 | 0.691799 | 0.561626 |
| M3 | 0.251049 | 0.697390 | 0.556222 |

## Additive and interaction increments

- Movement M2-minus-M1: Brier -0.000981; log loss
  -0.001620; bootstrap 90% lower bounds
  -0.001697 and
  -0.004290; structural-null percentile
  0.180.
- Direction M2-minus-M1: Brier 0.000093; log loss
  -0.001114; AUC change 0.034183;
  structural-null percentile 0.990.
- Movement M3-minus-M2: Brier
  -0.000781; log loss
  -0.002435.
- Direction M3-minus-M2: Brier
  -0.002368; log loss
  -0.005591.

## Checkpoints

### Movement

- 30-minute checkpoint: M2-minus-M1 Brier improvement -0.000835; M3-minus-M2 -0.000147.
- 60-minute checkpoint: M2-minus-M1 Brier improvement -0.001127; M3-minus-M2 -0.001415.
### Direction among large moves

- 30-minute checkpoint: M2-minus-M1 Brier improvement -0.000307; M3-minus-M2 -0.001112.
- 60-minute checkpoint: M2-minus-M1 Brier improvement 0.000466; M3-minus-M2 -0.003538.

## Monthly stability

- large_remaining_move: M2 beat M1 on Brier in 1 represented months; M3 beat M2 in 1.
- up_given_large_move: M2 beat M1 on Brier in 5 represented months; M3 beat M2 in 3.

## Delayed economic-reference diagnostic

- M3 top-one selected 315 slates; mean gross
  cohort-relative remaining return at zero synthetic friction was
  42.292658 bps.
- Paired M3-minus-M1 top-one cohort-relative result was
  6.058131 bps per slate.

## Concentration and decision

- Maximum stock row share: 0.050254.
- Maximum current-state row share: 0.370706.
- Interaction support: `sufficient`.
- Final category: `opening_structure_no_increment_over_price`.
- Exact rerun: passed.
- Independent audit: passed.
