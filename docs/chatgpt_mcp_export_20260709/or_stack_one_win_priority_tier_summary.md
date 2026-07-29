# OR-stack one-win priority tier test

research_only: true
live_ordering_enabled: false
order_placement: disabled

## Inputs
- captured events: `/private/tmp/stocker_one_win_warm_state_idea_sweep_20260709/captured_events_with_one_win_warm_state_and_or_stack.csv`
- source-visible pairs: `/private/tmp/stocker_smid24_source_visible_candidate_book_fresh_20260708/loop_source_visible_candidate_book_v0_20260707T230822Z/source_visible_paired_entry_timing_detail.csv`
- source dedupe: earliest source-visible row per confirmed `future_loop_key` (max `delay_min`, tie lower `book_rank`).

## Forward 2026 priority tiers
- or_stack_all: 29 events, +21.775R, mean +0.751R, win 93.1%, losses 2, worst month +0.601R
- tier_a_or_stack_one_win_warm: 10 events, +8.574R, mean +0.857R, win 100.0%, losses 0, worst month +0.900R
- tier_b_or_stack_not_one_win_warm: 19 events, +13.201R, mean +0.695R, win 89.5%, losses 2, worst month +0.601R
- or_stack_first_win_next5: 9 events, +8.100R, mean +0.900R, win 100.0%, losses 0, worst month +0.900R
- or_stack_last_prior_winner: 9 events, +7.674R, mean +0.853R, win 100.0%, losses 0, worst month +0.900R

## Train pre-2026 check
- or_stack_all: 72 events, +24.614R, mean +0.342R, win 69.4%, negative months 3
- tier_a_or_stack_one_win_warm: 22 events, +4.684R, mean +0.213R, win 63.6%, negative months 4
- tier_b_or_stack_not_one_win_warm: 50 events, +19.930R, mean +0.399R, win 72.0%, negative months 3

## Source-visible early-entry diagnostic
- tier_a_warm: matched 0/10 forward events, source +0.000R vs confirmed +0.000R, delta +0.000R, source win 0.0%, median lead 0 min
- tier_b_cold: matched 0/19 forward events, source +0.000R vs confirmed +0.000R, delta +0.000R, source win 0.0%, median lead 0 min
- direct OR-stack early-entry bridge: 0 matched forward events. The source-visible paired table is a separate event universe here, so it cannot validate entering the OR-stack events earlier.

## Hybrid scenario, forward 2026
- confirmed_only: 29 events, source entries 0, +21.775R, mean +0.751R, win 93.1%
- tier_a_source_if_available_b_confirmed: 29 events, source entries 0, +21.775R, mean +0.751R, win 93.1%
- all_source_if_available: 29 events, source entries 0, +21.775R, mean +0.751R, win 93.1%

## Additive source-visible harvest layer, forward 2026
These rows use the train-selected activation-harvest candidates as a separate add-on layer, not as an early entry into the OR-stack events.
- 2025_to_2026|5d|thr=1.1|refresh|all|top3: add-on 8 events, source +5.200R / confirmed -0.827R; combined source +26.975R over 37 events, win 91.9%, median lead 12 min
- pre2026_to_2026|5d|thr=1.0|refresh|all|top5: add-on 11 events, source +4.069R / confirmed +0.873R; combined source +25.844R over 40 events, win 87.5%, median lead 15 min
- pre2026_to_2026|10d|thr=1.0|refresh|first2_per_window|top1: add-on 10 events, source +1.869R / confirmed +1.453R; combined source +23.644R over 39 events, win 84.6%, median lead 12 min
- 2025_to_2026|10d|thr=1.0|refresh|loop_first|top2: add-on 12 events, source +1.669R / confirmed +1.253R; combined source +23.444R over 41 events, win 82.9%, median lead 15 min

## Output files
- `/private/tmp/stocker_or_stack_one_win_priority_tier_test_20260709/tier_summary_all_splits.csv`
- `/private/tmp/stocker_or_stack_one_win_priority_tier_test_20260709/source_visible_match_summary.csv`
- `/private/tmp/stocker_or_stack_one_win_priority_tier_test_20260709/or_stack_source_visible_match_detail.csv`
- `/private/tmp/stocker_or_stack_one_win_priority_tier_test_20260709/hybrid_source_confirmed_scenarios.csv`
- `/private/tmp/stocker_or_stack_one_win_priority_tier_test_20260709/or_stack_plus_source_visible_harvest_summary.csv`
- `/private/tmp/stocker_or_stack_one_win_priority_tier_test_20260709/summary.json`

## Interpretation
Tier A is cleaner than Tier B on win rate, but it is not an expansion gate. It is a priority/sizing label inside the existing OR stack.
The direct source-visible timing bridge does not overlap the OR-stack accepted events, so source-visible has to be evaluated as a separate candidate layer unless a compatible source-to-accepted-event map is built.
