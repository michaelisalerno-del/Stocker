# One-Win Tier-2 Union Check

Safety labels:
- research_only: true
- live_ordering_enabled: false
- order_placement: disabled
- edge_claimed: false

This checks whether one-win variants add useful events outside the existing OR-stack layer.

## Forward 2026 Union Results

| idea | events | symbols | months | total_r | mean_r | win_rate | max_symbol_share | negative_months | forward_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tier1_or_stack | 29 | 13 | 6 | 21.775 | 0.751 | 0.931 | 0.138 | 0 | 26.249 |
| tier1_or_one_win_stable_no_conflict | 33 | 13 | 7 | 20.599 | 0.624 | 0.879 | 0.152 | 0 | 24.563 |
| tier1_or_one_win_anytime | 42 | 13 | 7 | 20.572 | 0.490 | 0.786 | 0.238 | 1 | 23.932 |
| tier1_or_first_winner_same_phase_warm | 36 | 13 | 6 | 20.075 | 0.558 | 0.833 | 0.139 | 1 | 23.654 |
| tier1_or_first_win_next_2_events | 37 | 13 | 7 | 20.072 | 0.542 | 0.811 | 0.135 | 0 | 23.683 |
| tier1_or_first_winner_no_conflict_warm | 39 | 13 | 6 | 19.648 | 0.504 | 0.795 | 0.205 | 1 | 22.989 |
| tier1_or_first_win_next_1_event | 35 | 13 | 6 | 19.048 | 0.544 | 0.800 | 0.114 | 1 | 22.468 |
| tier1_or_first_win_next_3_events | 38 | 13 | 7 | 18.972 | 0.499 | 0.789 | 0.158 | 1 | 22.239 |
| tier1_or_first_win_next_5_events | 38 | 13 | 7 | 18.972 | 0.499 | 0.789 | 0.158 | 1 | 22.239 |
| tier1_or_last_event_winner | 38 | 13 | 6 | 17.748 | 0.467 | 0.763 | 0.184 | 1 | 20.790 |

## Forward 2026 Add-On Only

| idea | events | symbols | months | total_r | mean_r | win_rate | max_symbol_share | negative_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| addon_only_one_win_stable_no_conflict | 4 | 2 | 3 | -1.176 | -0.294 | 0.500 | 0.750 | 1 |
| addon_only_one_win_anytime | 13 | 6 | 6 | -1.203 | -0.093 | 0.462 | 0.615 | 3 |
| addon_only_first_winner_same_phase_warm | 7 | 5 | 4 | -1.700 | -0.243 | 0.429 | 0.429 | 3 |
| addon_only_first_win_next_2_events | 8 | 6 | 5 | -1.703 | -0.213 | 0.375 | 0.375 | 3 |
| addon_only_first_winner_no_conflict_warm | 10 | 5 | 5 | -2.127 | -0.213 | 0.400 | 0.600 | 3 |
| addon_only_first_win_next_1_event | 6 | 5 | 4 | -2.727 | -0.454 | 0.167 | 0.333 | 3 |
| addon_only_first_win_next_3_events | 9 | 6 | 5 | -2.803 | -0.311 | 0.333 | 0.444 | 3 |
| addon_only_first_win_next_5_events | 9 | 6 | 5 | -2.803 | -0.311 | 0.333 | 0.444 | 3 |
| addon_only_last_event_winner | 9 | 5 | 5 | -4.027 | -0.447 | 0.222 | 0.556 | 3 |
