# Frozen Hidden-Loop Economics and Registered-Loop Bridge Quick Screen V0

Decision: `blocked_support_failure`

- Economic status: `insufficient_support`
- Registered-lead status: `insufficient_support`
- Predictive-bridge status: `insufficient_support`
- Scope: retrospective research-only quick feasibility screen.
- Synthetic friction is not realised P&L.
- Protected rows materialised: `0`
- Frozen-family support: 1638 development / 957 assessment events.

## Binding question A — post-completion economics

| period | events | mean_bps | cohort_relative_return_after_20bps | excess_vs_matched_control_bps | matched_control_coverage |
| --- | --- | --- | --- | --- | --- |
| development | 1638 | -22.518172 | -18.477940 | -6.411119 | 0.996337 |
| assessment | 957 | -13.693017 | -18.289453 | 9.176075 | 0.989551 |

Family assessment results (12-bar opening-pressure direction, after 20 bps):

| scope | events | mean_bps | cohort_relative_return_after_20bps | excess_vs_matched_control_bps | maximum_stock_share | maximum_month_share |
| --- | --- | --- | --- | --- | --- | --- |
| unregistered_primitive_like__5-6-5 | 597 | -7.971429 | -17.702742 | 12.223355 | 0.082077 | 0.244880 |
| unregistered_primitive_like__2-3-2 | 211 | -32.190041 | -32.797019 | -7.288623 | 0.151659 | 0.303913 |
| unregistered_primitive_like__2-5-2 | 75 | -32.744493 | -24.656593 | -3.704558 | 0.213333 | 0.315421 |
| unregistered_primitive_like__4-7-4 | 74 | 12.198127 | 24.796574 | 44.968647 | 0.135135 | 0.285734 |
| OTHER_UNREGISTERED_FAMILY | 684 | -18.697177 | -13.922993 | 5.207821 | 0.077485 | 0.219994 |

Monthly stability:

| period | events | mean_bps | positive_rate |
| --- | --- | --- | --- |
| 2025-01 | 166 | 16.005799 | 0.506024 |
| 2025-02 | 118 | -20.873303 | 0.500000 |
| 2025-03 | 144 | 0.560747 | 0.513889 |
| 2025-04 | 139 | 29.226783 | 0.503597 |
| 2025-05 | 88 | -43.518088 | 0.375000 |
| 2025-06 | 81 | -31.728748 | 0.382716 |
| 2025-07 | 114 | -32.101828 | 0.359649 |
| 2025-08 | 107 | -68.992262 | 0.345794 |

Checkpoint stability:

| period | source_checkpoint | events | mean_bps |
| --- | --- | --- | --- |
| development | 6 | 787 | -35.093148 |
| development | 12 | 851 | -10.888905 |
| assessment | 6 | 493 | -3.387631 |
| assessment | 12 | 464 | -24.642491 |

Bootstrap intervals:

| metric | value | interval_80_lower | interval_80_upper | interval_90_lower | interval_90_upper | interval_95_lower | interval_95_upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| primary_net_return_20bps | -14.249554 | -31.100601 | -0.590961 | -33.895127 | 1.744539 | -34.782182 | 2.954248 |
| cohort_relative_net_return_20bps | -18.751404 | -25.347162 | -10.323025 | -27.767475 | -9.314746 | -28.304771 | -7.802188 |
| matched_control_excess | 7.405033 | -3.013532 | 16.492993 | -5.548408 | 20.496237 | -9.528487 | 21.846998 |

Family multiplicity:

| hidden_family_class | p_value | q_value | q_le_0_10 |
| --- | --- | --- | --- |
| unregistered_primitive_like__5-6-5 | 0.764706 | 1 | False |
| unregistered_primitive_like__2-3-2 | 1.000000 | 1 | False |
| unregistered_primitive_like__2-5-2 | 0.941176 | 1 | False |
| unregistered_primitive_like__4-7-4 | 0.392157 | 1 | False |

## Binding question B1 — realised hidden-to-registered lead

- Assessment six-bar completion rate: 0.049112
- Assessment twelve-bar completion rate: 0.128527
- Six-bar structural-null percentile: 0.00
- Observed rate exceeds null 90th percentile: False

Supported exact transitions:

No rows.

## Binding question B2 — predictive bridge

| model | rows | base_rate | log_loss | brier_score | auc | average_precision | expected_calibration_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 6262 | 0.127444 | 0.364340 | 0.107323 | 0.665779 | 0.230687 | 0.024758 |
| B1 | 6262 | 0.127444 | 0.364226 | 0.107289 | 0.665562 | 0.230787 | 0.024126 |

Monthly metrics:

| year_month | model | log_loss | brier_score | auc |
| --- | --- | --- | --- | --- |
| 2025-01 | B0 | 0.377471 | 0.113575 | 0.680784 |
| 2025-01 | B1 | 0.377329 | 0.113524 | 0.681076 |
| 2025-02 | B0 | 0.340049 | 0.098936 | 0.705506 |
| 2025-02 | B1 | 0.339876 | 0.098894 | 0.705423 |
| 2025-03 | B0 | 0.389691 | 0.116542 | 0.647982 |
| 2025-03 | B1 | 0.389762 | 0.116569 | 0.647010 |
| 2025-04 | B0 | 0.326372 | 0.092406 | 0.666622 |
| 2025-04 | B1 | 0.326215 | 0.092346 | 0.665627 |
| 2025-05 | B0 | 0.353285 | 0.102665 | 0.640892 |
| 2025-05 | B1 | 0.353383 | 0.102700 | 0.640177 |
| 2025-06 | B0 | 0.359368 | 0.104582 | 0.631577 |
| 2025-06 | B1 | 0.358789 | 0.104420 | 0.633069 |
| 2025-07 | B0 | 0.371480 | 0.110107 | 0.695132 |
| 2025-07 | B1 | 0.371409 | 0.110076 | 0.694608 |
| 2025-08 | B0 | 0.401623 | 0.121609 | 0.655934 |
| 2025-08 | B1 | 0.401679 | 0.121631 | 0.655509 |

Checkpoint metrics:

| decision_ordinal | model | log_loss | brier_score | auc |
| --- | --- | --- | --- | --- |
| 6 | B0 | 0.320793 | 0.089915 | 0.635919 |
| 6 | B1 | 0.320570 | 0.089850 | 0.636148 |
| 12 | B0 | 0.407888 | 0.124731 | 0.667815 |
| 12 | B1 | 0.407881 | 0.124729 | 0.667338 |

Fixed-prediction session-bootstrap intervals:

| metric | value | interval_80_lower | interval_80_upper | interval_90_lower | interval_90_upper | interval_95_lower | interval_95_upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| log_loss_improvement | 0.000109 | 0.000029 | 0.000202 | -0.000018 | 0.000227 | -0.000061 | 0.000244 |
| brier_improvement | 0.000031 | 0.000001 | 0.000063 | -0.000008 | 0.000070 | -0.000017 | 0.000072 |
| auc_improvement | -0.000256 | -0.000739 | 0.000148 | -0.000825 | 0.000376 | -0.001075 | 0.000537 |
| average_precision_improvement | 0.000063 | -0.000565 | 0.000736 | -0.000653 | 0.000840 | -0.000748 | 0.000879 |

Null draws exceeded: `{"auc_improvement": 4, "average_precision_improvement": 8, "brier_improvement": 9, "log_loss_improvement": 9}`.

## Integrity

- Population reconstruction: passed; maximum difference 0.
- Determinism: `True`.
- Maximum probability difference: 0.
- Maximum return difference: 0 bps.
- This is not prospective validation, achieved P&L, a deployable model,
  strategy promotion, or permission to trade.
