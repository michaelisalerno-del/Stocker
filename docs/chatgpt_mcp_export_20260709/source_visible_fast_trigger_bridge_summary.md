# Source-visible fast-trigger bridge test

research_only: true
live_ordering_enabled: false
order_placement: disabled
edge_claimed: false

## Baseline
- OR-stack forward 2026: 29 events, +21.775R, mean +0.751R, win 93.1%

## Source-to-OR-stack bridge
- Exact future-event bridge, forward 2026: 0 covered events.
- Best same-future-loop loose bridges, forward 2026:
  - same_future_loop <= 1h: 0/29 covered, source +0.000R, accepted-covered +0.000R, exact matches 0, median lead 0 min
  - same_future_loop <= 4h: 0/29 covered, source +0.000R, accepted-covered +0.000R, exact matches 0, median lead 0 min
  - same_future_loop <= 24h: 0/29 covered, source +0.000R, accepted-covered +0.000R, exact matches 0, median lead 0 min
  - same_future_loop <= 120h: 0/29 covered, source +0.000R, accepted-covered +0.000R, exact matches 0, median lead 0 min
  - same_future_loop_regime <= 1h: 0/29 covered, source +0.000R, accepted-covered +0.000R, exact matches 0, median lead 0 min
  - same_future_loop_regime <= 4h: 0/29 covered, source +0.000R, accepted-covered +0.000R, exact matches 0, median lead 0 min
  - same_future_loop_regime <= 24h: 0/29 covered, source +0.000R, accepted-covered +0.000R, exact matches 0, median lead 0 min
  - same_future_loop_regime <= 120h: 0/29 covered, source +0.000R, accepted-covered +0.000R, exact matches 0, median lead 0 min

## Source-scored fast-trigger grid
Rows below are selected by training-side source-entry R, not by forward result.
- 2025_to_2026 5d thr=1.0 all top1: train source +1.755R over 9; forward source +0.000R over 0, mean +0.000R, win 0.0%; forward confirmed +0.000R
- 2025_to_2026 10d thr=1.07 all top2: train source +3.716R over 11; forward source +0.000R over 0, mean +0.000R, win 0.0%; forward confirmed +0.000R
- pre2026_to_2026 5d thr=1.1 all top1: train source +3.252R over 51; forward source +0.000R over 0, mean +0.000R, win 0.0%; forward confirmed +0.000R
- pre2026_to_2026 10d thr=1.05 all top1: train source +5.872R over 24; forward source +0.000R over 0, mean +0.000R, win 0.0%; forward confirmed +0.000R

## OR-stack plus source-scored add-on
- No forward add-on candidates were selected by the source-scored train objective.

## Contrast: confirmation-scored harness
This is not a new selection rule; it explains the prior source-entry +5.2R observation.
- 2025_to_2026 5d thr=1.07 all top2: train source -9.499R over 64; forward source +5.200R over 8; forward confirmed -0.827R
- 2025_to_2026 5d thr=1.07 all top3: train source -8.844R over 65; forward source +5.200R over 8; forward confirmed -0.827R
- 2025_to_2026 5d thr=1.07 all top5: train source -8.844R over 65; forward source +5.200R over 8; forward confirmed -0.827R
- 2025_to_2026 5d thr=1.07 all top10: train source -8.844R over 65; forward source +5.200R over 8; forward confirmed -0.827R
- 2025_to_2026 5d thr=1.07 first2_per_window top2: train source -9.499R over 64; forward source +5.200R over 8; forward confirmed -0.827R

## Files
- `/private/tmp/stocker_source_visible_fast_trigger_bridge_test_20260709/source_to_or_stack_loose_bridge_summary.csv`
- `/private/tmp/stocker_source_visible_fast_trigger_bridge_test_20260709/source_to_or_stack_loose_bridge_detail.csv`
- `/private/tmp/stocker_source_visible_fast_trigger_bridge_test_20260709/source_scored_harvest_grid_all.csv`
- `/private/tmp/stocker_source_visible_fast_trigger_bridge_test_20260709/source_scored_selected_by_split_horizon.csv`
- `/private/tmp/stocker_source_visible_fast_trigger_bridge_test_20260709/source_scored_selected_forward_candidates.csv`
- `/private/tmp/stocker_source_visible_fast_trigger_bridge_test_20260709/or_stack_plus_source_scored_harvest_summary.csv`
- `/private/tmp/stocker_source_visible_fast_trigger_bridge_test_20260709/confirmation_scored_forward_source_contrast.csv`
- `/private/tmp/stocker_source_visible_fast_trigger_bridge_test_20260709/summary.json`

## Interpretation
A loose source-to-OR-stack bridge can show nearby source-visible activity, but without exact future-event matches it is not evidence that the OR-stack event was enterable earlier.
The source-scored fast-trigger layer should be treated as a separate add-on candidate layer. Its confirmation R and source R can diverge materially, so it should stay separate from the parent-loop OR-stack admission work.
