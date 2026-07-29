# OR Stack Robustness And Wider Candidate Test

Safety labels:
- research_only: true
- live_ordering_enabled: false
- order_placement: disabled
- edge_claimed: false

Accepted-book stack:
`stable activation + no precursor conflict + (same_phase_win_rate >= 0.65 OR range == high_range)`.

Candidate-book analogue:
`stable activation + no precursor conflict + (prior candidate pair win-rate >= 0.65 OR range == high_range)`.
Candidate prior pair is `future_loop || current_regime`, computed expanding in timestamp order.

## Accepted Summary

| view | filter | split | events | symbols | months | total_r | mean_r | win_rate | max_symbol_share | negative_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accepted | and_stack | forward_2026 | 13 | 9 | 5 | 8.430 | 0.648 | 0.846 | 0.154 | 1 |
| accepted | and_stack | train_pre2026 | 22 | 13 | 15 | 11.820 | 0.537 | 0.818 | 0.136 | 2 |
| accepted | captured_120h | forward_2026 | 56 | 15 | 7 | 17.850 | 0.319 | 0.696 | 0.250 | 1 |
| accepted | captured_120h | train_pre2026 | 137 | 19 | 28 | 32.108 | 0.234 | 0.642 | 0.153 | 5 |
| accepted | conflict_blocker | forward_2026 | 4 | 2 | 3 | -3.156 | -0.789 | 0.250 | 0.750 | 2 |
| accepted | conflict_blocker | train_pre2026 | 12 | 9 | 8 | -1.389 | -0.116 | 0.500 | 0.250 | 4 |
| accepted | high_range | forward_2026 | 19 | 11 | 6 | 13.404 | 0.705 | 0.895 | 0.158 | 1 |
| accepted | high_range | train_pre2026 | 39 | 16 | 20 | 15.156 | 0.389 | 0.718 | 0.154 | 5 |
| accepted | high_range_only_stack | forward_2026 | 6 | 5 | 3 | 4.974 | 0.829 | 1.000 | 0.333 | 0 |
| accepted | high_range_only_stack | train_pre2026 | 17 | 9 | 14 | 3.336 | 0.196 | 0.588 | 0.235 | 6 |
| accepted | or_stack | forward_2026 | 29 | 13 | 6 | 21.775 | 0.751 | 0.931 | 0.138 | 0 |
| accepted | or_stack | train_pre2026 | 72 | 17 | 22 | 24.614 | 0.342 | 0.694 | 0.139 | 3 |
| accepted | same_phase_wr65 | forward_2026 | 23 | 11 | 6 | 16.801 | 0.730 | 0.913 | 0.174 | 0 |
| accepted | same_phase_wr65 | train_pre2026 | 55 | 15 | 20 | 21.278 | 0.387 | 0.727 | 0.145 | 3 |
| accepted | stable_any | forward_2026 | 44 | 14 | 7 | 16.555 | 0.376 | 0.750 | 0.205 | 1 |
| accepted | stable_any | train_pre2026 | 110 | 19 | 26 | 26.495 | 0.241 | 0.655 | 0.155 | 6 |
| accepted | stable_no_conflict | forward_2026 | 40 | 14 | 7 | 19.712 | 0.493 | 0.800 | 0.150 | 1 |
| accepted | stable_no_conflict | train_pre2026 | 98 | 17 | 26 | 27.884 | 0.285 | 0.673 | 0.143 | 5 |
| accepted | wr65_only_stack | forward_2026 | 10 | 6 | 5 | 8.370 | 0.837 | 1.000 | 0.300 | 0 |
| accepted | wr65_only_stack | train_pre2026 | 33 | 12 | 16 | 9.458 | 0.287 | 0.667 | 0.182 | 4 |

## Accepted By Year

| filter | year | events | symbols | months | total_r | mean_r | win_rate | max_symbol_share | negative_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| captured_120h | 2023 | 6 | 4 | 4 | 2.905 | 0.484 | 0.667 | 0.333 | 0 |
| captured_120h | 2024 | 50 | 12 | 12 | 12.719 | 0.254 | 0.680 | 0.180 | 3 |
| captured_120h | 2025 | 81 | 18 | 12 | 16.483 | 0.203 | 0.617 | 0.148 | 2 |
| captured_120h | 2026 | 56 | 15 | 7 | 17.850 | 0.319 | 0.696 | 0.250 | 1 |
| conflict_blocker | 2024 | 2 | 2 | 2 | -0.200 | -0.100 | 0.500 | 0.500 | 1 |
| conflict_blocker | 2025 | 10 | 8 | 6 | -1.189 | -0.119 | 0.500 | 0.300 | 3 |
| conflict_blocker | 2026 | 4 | 2 | 3 | -3.156 | -0.789 | 0.250 | 0.750 | 2 |
| or_stack | 2023 | 1 | 1 | 1 | -0.433 | -0.433 | 0.000 | 1.000 | 1 |
| or_stack | 2024 | 24 | 10 | 9 | 9.941 | 0.414 | 0.750 | 0.208 | 0 |
| or_stack | 2025 | 47 | 16 | 12 | 15.106 | 0.321 | 0.681 | 0.170 | 2 |
| or_stack | 2026 | 29 | 13 | 6 | 21.775 | 0.751 | 0.931 | 0.138 | 0 |
| stable_no_conflict | 2023 | 4 | 2 | 2 | 1.105 | 0.276 | 0.500 | 0.500 | 0 |
| stable_no_conflict | 2024 | 38 | 11 | 12 | 8.693 | 0.229 | 0.658 | 0.184 | 3 |
| stable_no_conflict | 2025 | 56 | 16 | 12 | 18.086 | 0.323 | 0.696 | 0.161 | 2 |
| stable_no_conflict | 2026 | 40 | 14 | 7 | 19.712 | 0.493 | 0.800 | 0.150 | 1 |

## Accepted 2026 By Month

| filter | month | events | symbols | months | total_r | mean_r | win_rate | max_symbol_share | negative_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conflict_blocker | 2026-01 | 1 | 1 | 1 | 0.144 | 0.144 | 1.000 | 1.000 | 0 |
| conflict_blocker | 2026-03 | 2 | 1 | 1 | -2.200 | -1.100 | 0.000 | 1.000 | 1 |
| conflict_blocker | 2026-05 | 1 | 1 | 1 | -1.100 | -1.100 | 0.000 | 1.000 | 1 |
| or_stack | 2026-01 | 5 | 4 | 1 | 0.601 | 0.120 | 0.600 | 0.400 | 0 |
| or_stack | 2026-02 | 6 | 5 | 1 | 5.400 | 0.900 | 1.000 | 0.333 | 0 |
| or_stack | 2026-03 | 5 | 4 | 1 | 4.500 | 0.900 | 1.000 | 0.400 | 0 |
| or_stack | 2026-04 | 5 | 5 | 1 | 4.500 | 0.900 | 1.000 | 0.200 | 0 |
| or_stack | 2026-05 | 1 | 1 | 1 | 0.900 | 0.900 | 1.000 | 1.000 | 0 |
| or_stack | 2026-06 | 7 | 6 | 1 | 5.874 | 0.839 | 1.000 | 0.286 | 0 |

## Candidate Confirmation-R Summary

| view | filter | split | events | symbols | months | total_r | mean_r | win_rate | max_symbol_share | negative_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_confirmation_r | baseline_all | forward_2026 | 80 | 20 | 6 | 18.798 | 0.235 | 0.675 | 0.188 | 1 |
| candidate_confirmation_r | baseline_all | train_pre2026 | 383 | 23 | 24 | 76.884 | 0.201 | 0.645 | 0.157 | 7 |
| candidate_confirmation_r | candidate_or_stack_2 | forward_2026 | 23 | 13 | 5 | 5.882 | 0.256 | 0.652 | 0.130 | 1 |
| candidate_confirmation_r | candidate_or_stack_2 | train_pre2026 | 121 | 20 | 24 | 27.834 | 0.230 | 0.686 | 0.107 | 9 |
| candidate_confirmation_r | candidate_or_stack_5 | forward_2026 | 19 | 12 | 5 | 5.040 | 0.265 | 0.632 | 0.158 | 1 |
| candidate_confirmation_r | candidate_or_stack_5 | train_pre2026 | 87 | 18 | 22 | 20.322 | 0.234 | 0.678 | 0.115 | 9 |
| candidate_confirmation_r | candidate_prior_wr65_2 | forward_2026 | 17 | 11 | 5 | 4.135 | 0.243 | 0.647 | 0.176 | 2 |
| candidate_confirmation_r | candidate_prior_wr65_2 | train_pre2026 | 106 | 20 | 23 | 23.090 | 0.218 | 0.679 | 0.123 | 9 |
| candidate_confirmation_r | candidate_prior_wr65_5 | forward_2026 | 12 | 8 | 5 | 2.392 | 0.199 | 0.583 | 0.250 | 1 |
| candidate_confirmation_r | candidate_prior_wr65_5 | train_pre2026 | 67 | 18 | 17 | 15.239 | 0.227 | 0.687 | 0.149 | 9 |
| candidate_confirmation_r | captured_120h | forward_2026 | 68 | 15 | 6 | 16.373 | 0.241 | 0.662 | 0.206 | 1 |
| candidate_confirmation_r | captured_120h | train_pre2026 | 360 | 20 | 24 | 72.839 | 0.202 | 0.644 | 0.153 | 7 |
| candidate_confirmation_r | conflict_blocker | forward_2026 | 13 | 5 | 5 | 1.669 | 0.128 | 0.615 | 0.385 | 2 |
| candidate_confirmation_r | conflict_blocker | train_pre2026 | 26 | 10 | 11 | -1.312 | -0.050 | 0.423 | 0.154 | 7 |
| candidate_confirmation_r | high_range | forward_2026 | 11 | 8 | 5 | 0.871 | 0.079 | 0.545 | 0.182 | 2 |
| candidate_confirmation_r | high_range | train_pre2026 | 31 | 11 | 19 | 11.031 | 0.356 | 0.710 | 0.129 | 5 |
| candidate_confirmation_r | stable_no_conflict | forward_2026 | 41 | 14 | 6 | 14.055 | 0.343 | 0.707 | 0.146 | 1 |
| candidate_confirmation_r | stable_no_conflict | train_pre2026 | 267 | 20 | 24 | 52.923 | 0.198 | 0.652 | 0.135 | 7 |

## Candidate Source-R Summary

| view | filter | split | events | symbols | months | total_r | mean_r | win_rate | max_symbol_share | negative_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_source_r | baseline_all | forward_2026 | 80 | 20 | 6 | -4.094 | -0.051 | 0.500 | 0.188 | 4 |
| candidate_source_r | baseline_all | train_pre2026 | 383 | 23 | 24 | -38.995 | -0.102 | 0.488 | 0.157 | 16 |
| candidate_source_r | candidate_or_stack_2 | forward_2026 | 23 | 13 | 5 | -2.502 | -0.109 | 0.478 | 0.130 | 3 |
| candidate_source_r | candidate_or_stack_2 | train_pre2026 | 121 | 20 | 24 | -25.851 | -0.214 | 0.438 | 0.107 | 18 |
| candidate_source_r | candidate_or_stack_5 | forward_2026 | 19 | 12 | 5 | -0.102 | -0.005 | 0.526 | 0.158 | 3 |
| candidate_source_r | candidate_or_stack_5 | train_pre2026 | 87 | 18 | 22 | -15.070 | -0.173 | 0.448 | 0.115 | 13 |
| candidate_source_r | candidate_prior_wr65_2 | forward_2026 | 17 | 11 | 5 | -3.902 | -0.230 | 0.412 | 0.176 | 4 |
| candidate_source_r | candidate_prior_wr65_2 | train_pre2026 | 106 | 20 | 23 | -23.351 | -0.220 | 0.434 | 0.123 | 17 |
| candidate_source_r | candidate_prior_wr65_5 | forward_2026 | 12 | 8 | 5 | -0.402 | -0.034 | 0.500 | 0.250 | 3 |
| candidate_source_r | candidate_prior_wr65_5 | train_pre2026 | 67 | 18 | 17 | -10.070 | -0.150 | 0.463 | 0.149 | 11 |
| candidate_source_r | captured_120h | forward_2026 | 68 | 15 | 6 | -1.577 | -0.023 | 0.529 | 0.206 | 4 |
| candidate_source_r | captured_120h | train_pre2026 | 360 | 20 | 24 | -44.544 | -0.124 | 0.478 | 0.153 | 16 |
| candidate_source_r | conflict_blocker | forward_2026 | 13 | 5 | 5 | -0.170 | -0.013 | 0.538 | 0.385 | 2 |
| candidate_source_r | conflict_blocker | train_pre2026 | 26 | 10 | 11 | -2.128 | -0.082 | 0.462 | 0.154 | 7 |
| candidate_source_r | high_range | forward_2026 | 11 | 8 | 5 | -0.100 | -0.009 | 0.545 | 0.182 | 3 |
| candidate_source_r | high_range | train_pre2026 | 31 | 11 | 19 | -2.379 | -0.077 | 0.484 | 0.129 | 10 |
| candidate_source_r | stable_no_conflict | forward_2026 | 41 | 14 | 6 | 1.317 | 0.032 | 0.585 | 0.146 | 4 |
| candidate_source_r | stable_no_conflict | train_pre2026 | 267 | 20 | 24 | -40.251 | -0.151 | 0.468 | 0.135 | 16 |

## Candidate By Year

| filter | year | events | symbols | months | total_r | mean_r | win_rate | max_symbol_share | negative_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_all | 2024 | 194 | 21 | 12 | 41.866 | 0.216 | 0.644 | 0.134 | 2 |
| baseline_all | 2025 | 189 | 23 | 12 | 35.018 | 0.185 | 0.646 | 0.180 | 5 |
| baseline_all | 2026 | 80 | 20 | 6 | 18.798 | 0.235 | 0.675 | 0.188 | 1 |
| candidate_or_stack_2 | 2024 | 55 | 16 | 12 | 15.102 | 0.275 | 0.691 | 0.127 | 3 |
| candidate_or_stack_2 | 2025 | 66 | 20 | 12 | 12.732 | 0.193 | 0.682 | 0.121 | 6 |
| candidate_or_stack_2 | 2026 | 23 | 13 | 5 | 5.882 | 0.256 | 0.652 | 0.130 | 1 |
| conflict_blocker | 2024 | 21 | 9 | 7 | -1.812 | -0.086 | 0.381 | 0.190 | 5 |
| conflict_blocker | 2025 | 5 | 4 | 4 | 0.500 | 0.100 | 0.600 | 0.400 | 2 |
| conflict_blocker | 2026 | 13 | 5 | 5 | 1.669 | 0.128 | 0.615 | 0.385 | 2 |
| stable_no_conflict | 2024 | 137 | 18 | 12 | 33.157 | 0.242 | 0.657 | 0.124 | 2 |
| stable_no_conflict | 2025 | 130 | 20 | 12 | 19.766 | 0.152 | 0.646 | 0.146 | 5 |
| stable_no_conflict | 2026 | 41 | 14 | 6 | 14.055 | 0.343 | 0.707 | 0.146 | 1 |

## Randomization Checks

```json
{
  "accepted": {
    "or_vs_captured_forward": {
      "events": 56,
      "selected_n": 29,
      "observed_total_r": 21.774751207706334,
      "unselected_total_r": -3.924823549415361,
      "null_mean_total_r": 9.210498256506296,
      "null_p95_total_r": 14.364529222841062,
      "null_p99_total_r": 16.352060700887296,
      "p_ge_observed": 0.0
    }
  },
  "candidate": {
    "candidate_or_stack_2_vs_forward_baseline": {
      "events": 80,
      "selected_n": 23,
      "observed_total_r": 5.88214088295404,
      "unselected_total_r": 12.916089240083217,
      "null_mean_total_r": 5.427171767778831,
      "null_p95_total_r": 10.903790760869523,
      "null_p99_total_r": 13.078289914569414,
      "p_ge_observed": 0.45005
    }
  }
}
```

## Files

- `accepted_or_stack_summary.csv`
- `accepted_or_stack_by_year.csv`
- `accepted_or_stack_2026_by_month.csv`
- `candidate_book_with_or_stack.csv`
- `candidate_or_stack_summary.csv`
- `candidate_or_stack_source_r_summary.csv`
- `candidate_or_stack_by_year.csv`
- `candidate_or_stack_2026_by_month.csv`
- `summary.json`

## Interpretation Guardrail

This is research-only analysis over existing local historical data. It does not alter exits, live/paper trading, broker execution, order placement, deployment, API keys, runtime paths, or YAML rules.
