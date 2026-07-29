# Stocker ChatGPT MCP Research Export - 2026-07-09

Search terms:
- stocker_latest_loop_regime_research_20260709
- OR-stack one-win priority tier
- source-visible fast-trigger bridge
- one-win warm-state idea sweep
- child morph template discovery handover

research_only: true
live_ordering_enabled: false
order_placement: disabled
edge_claimed: false

## Why This Export Exists

The running MCP server is currently rooted at the older Stocker checkout:

`/Users/michaelsalerno/Documents/Codex/2026-06-29-we-are-working-in-my-stocker`

The active research repo for this work is:

`/Users/michaelsalerno/Documents/Codex/2026-06-30-we-are-working-in-my-stocker-2`

The newest research outputs were written under `/private/tmp`, which the MCP server
cannot read by design. This bundle copies the latest summaries into
`~/StockerLocal/data/reports/research`, which is already visible to the running
MCP server. No server restart is required.

## Current Practical State

- Best stable layer remains the OR stack:
  stable activation + no precursor conflict + (`same_phase_win_rate >= 0.65` OR `range == high_range`).
- Forward 2026 OR stack: 29 events, +21.775R, mean +0.751R, 93.1% win.
- One-win warmth is useful as a priority label inside the OR stack, not as a standalone expansion rule.
- Direct source-visible early entry into OR-stack events failed: 0/29 forward OR-stack events matched by exact or same-future-loop bridge.
- Source-scored fast-trigger selection failed forward: training-side source-R winners selected 0 forward events.
- The prior +5.2R source-entry observation came from a confirmation-scored harness, not source-R-selectable evidence.
- Do not chase source-visible early entry as the main route right now.
- Continue with parent-loop OR-stack admission and staged parent_loop -> child/morph -> prediction gate -> admission gate work.

## Files In This MCP Export

- `child_morph_template_discovery_handover_20260708.md`: copied, 14070 bytes
- `one_win_warm_state_idea_sweep_summary.md`: copied, 4848 bytes
- `one_win_warm_state_idea_sweep_summary.json`: copied, 465 bytes
- `one_win_tier2_union_summary.md`: copied, 2398 bytes
- `or_stack_robustness_candidate_summary.md`: copied, 11908 bytes
- `or_stack_one_win_priority_tier_summary.md`: copied, 4147 bytes
- `or_stack_one_win_priority_tier_summary.json`: copied, 6133 bytes
- `or_stack_one_win_tier_summary_all_splits.csv`: copied, 2307 bytes
- `or_stack_plus_source_visible_harvest_summary.csv`: copied, 1588 bytes
- `source_visible_fast_trigger_bridge_summary.md`: copied, 4295 bytes
- `source_visible_fast_trigger_bridge_summary.json`: copied, 27835 bytes
- `source_to_or_stack_loose_bridge_summary.csv`: copied, 5084 bytes
- `source_scored_selected_by_split_horizon.csv`: copied, 5321 bytes
- `confirmation_scored_forward_source_contrast.csv`: copied, 2287 bytes

## Recommended Prompt For ChatGPT

Use the Stocker Research connector. Search for `stocker_latest_loop_regime_research_20260709`,
fetch this export, then read:

1. `summary.md`
2. `or_stack_one_win_priority_tier_summary.md`
3. `source_visible_fast_trigger_bridge_summary.md`
4. `one_win_warm_state_idea_sweep_summary.md`
5. `child_morph_template_discovery_handover_20260708.md`

Then continue research-only. Do not claim an edge, do not touch broker/order/live/paper/deployment paths,
and keep blocker/no-trade screening separate from candidate/admission work.

## Safety Footer

This is a research handover only. It does not approve live trading, paper trading, order placement,
broker execution, deployment, runtime trading changes, or YAML rule promotion.
