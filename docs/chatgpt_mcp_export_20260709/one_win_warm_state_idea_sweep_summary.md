# One-Win Warm-State Idea Sweep

Safety labels:
- research_only: true
- live_ordering_enabled: false
- order_placement: disabled
- edge_claimed: false

Key is strict same-stock `symbol_norm + current_loop + current_regime`.
The OR-stack layer is stable activation + no precursor conflict + (`same_phase_win_rate >= 0.65` OR `range == high_range`).

## Forward 2026 Ranked Ideas

| idea | events | symbols | months | total_r | mean_r | win_rate | max_symbol_share | negative_months | forward_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tier1_or_stack | 29 | 13 | 6 | 21.775 | 0.751 | 0.931 | 0.138 | 0 | 26.249 |
| baseline_captured_120h | 56 | 15 | 7 | 17.850 | 0.319 | 0.696 | 0.250 | 1 | 20.332 |
| baseline_all_accepted | 57 | 16 | 7 | 16.750 | 0.294 | 0.684 | 0.246 | 1 | 19.053 |
| one_win_plus_or_stack | 10 | 7 | 5 | 8.574 | 0.857 | 1.000 | 0.300 | 0 | 11.785 |
| first_win_next_5_events_plus_or_stack | 9 | 6 | 5 | 8.100 | 0.900 | 1.000 | 0.333 | 0 | 11.300 |
| last_event_winner_plus_or_stack | 9 | 7 | 5 | 7.674 | 0.853 | 1.000 | 0.333 | 0 | 10.732 |
| first_win_next_3_events_plus_or_stack | 8 | 6 | 5 | 7.200 | 0.900 | 1.000 | 0.375 | 0 | 10.046 |
| first_winner_no_conflict_then_warm_plus_or_stack | 8 | 7 | 5 | 6.774 | 0.847 | 1.000 | 0.250 | 0 | 9.669 |
| one_win_plus_stable_no_conflict | 14 | 7 | 6 | 7.398 | 0.528 | 0.857 | 0.286 | 1 | 9.633 |
| one_win_anytime | 23 | 8 | 7 | 7.371 | 0.320 | 0.696 | 0.391 | 2 | 8.574 |
| one_win_captured | 23 | 8 | 7 | 7.371 | 0.320 | 0.696 | 0.391 | 2 | 8.574 |
| last_win_next_3_events | 23 | 8 | 7 | 7.371 | 0.320 | 0.696 | 0.391 | 2 | 8.574 |
| last_win_next_5_events | 23 | 8 | 7 | 7.371 | 0.320 | 0.696 | 0.391 | 2 | 8.574 |
| first_win_next_2_events_plus_or_stack | 7 | 5 | 5 | 6.300 | 0.900 | 1.000 | 0.429 | 0 | 8.553 |
| first_win_next_1_events_plus_or_stack | 6 | 5 | 5 | 5.400 | 0.900 | 1.000 | 0.333 | 0 | 8.105 |
| last_win_next_2_events | 22 | 8 | 7 | 6.471 | 0.294 | 0.682 | 0.364 | 2 | 7.724 |
| first_winner_same_phase_wr65_then_warm_plus_or_stack | 5 | 5 | 5 | 4.500 | 0.900 | 1.000 | 0.200 | 0 | 7.012 |
| first_win_next_5_events | 18 | 8 | 6 | 5.297 | 0.294 | 0.667 | 0.222 | 1 | 6.613 |
| first_win_next_2_events | 15 | 8 | 6 | 4.597 | 0.306 | 0.667 | 0.200 | 0 | 5.951 |
| first_winner_or_stack_then_warm_plus_or_stack | 4 | 4 | 4 | 3.600 | 0.900 | 1.000 | 0.250 | 0 | 5.900 |
| first_win_next_3_events | 17 | 8 | 6 | 4.397 | 0.259 | 0.647 | 0.235 | 1 | 5.511 |
| first_winner_no_conflict_then_warm | 18 | 7 | 6 | 4.647 | 0.258 | 0.667 | 0.389 | 2 | 5.398 |
| last_win_within_45d | 3 | 3 | 2 | 2.700 | 0.900 | 1.000 | 0.333 | 0 | 4.759 |
| last_win_within_45d_plus_or_stack | 3 | 3 | 2 | 2.700 | 0.900 | 1.000 | 0.333 | 0 | 4.759 |
| last_event_winner | 18 | 8 | 6 | 3.647 | 0.203 | 0.611 | 0.333 | 2 | 4.418 |
| last_win_next_1_events | 18 | 8 | 6 | 3.647 | 0.203 | 0.611 | 0.333 | 2 | 4.418 |
| first_winner_same_phase_wr65_then_warm | 12 | 8 | 5 | 2.800 | 0.233 | 0.667 | 0.250 | 3 | 3.475 |
| first_win_next_1_events | 12 | 8 | 5 | 2.673 | 0.223 | 0.583 | 0.167 | 2 | 3.328 |
| first_win_within_20d | 2 | 2 | 2 | 1.800 | 0.900 | 1.000 | 0.500 | 0 | 2.373 |
| last_win_within_20d | 2 | 2 | 2 | 1.800 | 0.900 | 1.000 | 0.500 | 0 | 2.373 |
| first_win_within_20d_plus_or_stack | 2 | 2 | 2 | 1.800 | 0.900 | 1.000 | 0.500 | 0 | 2.373 |
| last_win_within_20d_plus_or_stack | 2 | 2 | 2 | 1.800 | 0.900 | 1.000 | 0.500 | 0 | 2.373 |
| first_win_within_45d | 2 | 2 | 2 | 1.800 | 0.900 | 1.000 | 0.500 | 0 | 2.373 |
| first_win_within_45d_plus_or_stack | 2 | 2 | 2 | 1.800 | 0.900 | 1.000 | 0.500 | 0 | 2.373 |
| first_winner_high_range_then_warm_plus_or_stack | 2 | 2 | 2 | 1.800 | 0.900 | 1.000 | 0.500 | 0 | 2.373 |

## Important Baselines

| idea | events | symbols | months | total_r | mean_r | win_rate | max_symbol_share | negative_months | forward_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_all_accepted | 57 | 16 | 7 | 16.750 | 0.294 | 0.684 | 0.246 | 1 | 19.053 |
| baseline_captured_120h | 56 | 15 | 7 | 17.850 | 0.319 | 0.696 | 0.250 | 1 | 20.332 |
| last_event_winner | 18 | 8 | 6 | 3.647 | 0.203 | 0.611 | 0.333 | 2 | 4.418 |
| last_event_winner_plus_or_stack | 9 | 7 | 5 | 7.674 | 0.853 | 1.000 | 0.333 | 0 | 10.732 |
| one_win_anytime | 23 | 8 | 7 | 7.371 | 0.320 | 0.696 | 0.391 | 2 | 8.574 |
| one_win_plus_or_stack | 10 | 7 | 5 | 8.574 | 0.857 | 1.000 | 0.300 | 0 | 11.785 |
| tier1_or_stack | 29 | 13 | 6 | 21.775 | 0.751 | 0.931 | 0.138 | 0 | 26.249 |

## Files

- `one_win_warm_state_idea_summary_all_splits.csv`
- `one_win_warm_state_forward_ranked.csv`
- `accepted_events_with_one_win_warm_state.csv`
- `captured_events_with_one_win_warm_state_and_or_stack.csv`
- `summary.json`

Interpretation guardrail: this is chronological research evidence only, not a trading rule or edge claim.
