# Stocker ChatGPT Research Summary Export - 2026-07-09

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

## Purpose

This file is committed to GitHub so ChatGPT can read the latest Stocker research state when the local MCP server or `/private/tmp` summaries are not accessible.

Active local repo for the research work:

`/Users/michaelsalerno/Documents/Codex/2026-06-30-we-are-working-in-my-stocker-2`

Branch:

`codex-state-event-detector-v0`

Start future local work with:

```bash
rtk git -C /Users/michaelsalerno/Documents/Codex/2026-06-30-we-are-working-in-my-stocker-2 status --short --branch
```

## Hard Constraints

- Research-only.
- Do not touch broker execution, IG, live trading, paper trading, order placement, API keys, deployment, or runtime trading paths.
- Do not claim an edge.
- Preserve uncommitted work.
- Use existing local 5m OHLCV/event-report data unless explicitly asked to download or generate more.
- Keep candidate/admission work separate from blocker/no-trade screening.
- Keep exits separate from directional discovery until a stable directional split exists.
- Do not save YAML rules unless explicitly asked.
- Focus on liquid midcap / broader SMID.
- Large caps are only negative-control / transfer diagnostic for now.

## Current Practical State

The best current layer is the OR stack:

```text
stable activation
+ no precursor conflict
+ (same_phase_win_rate >= 0.65 OR range == high_range)
```

Forward 2026 OR stack:

- 29 events
- +21.775R
- mean +0.751R
- 93.1% win
- 2 losses
- 0 negative months

Train pre-2026 OR stack:

- 72 events
- +24.614R
- mean +0.342R
- 69.4% win
- 3 negative months

Interpretation: the OR stack remains the strongest current admission layer, but this is research evidence only, not a trading rule or edge claim.

## One-Win Warm-State Test

Strict key used:

```text
symbol_norm + current_loop + current_regime
```

Forward 2026 priority tiers:

- OR stack all: 29 events, +21.775R, mean +0.751R, 93.1% win.
- Tier A, OR stack + one-win warm: 10 events, +8.574R, mean +0.857R, 100.0% win.
- Tier B, OR stack without one-win warm: 19 events, +13.201R, mean +0.695R, 89.5% win.
- OR stack + first-win next 5 events: 9 events, +8.100R, mean +0.900R, 100.0% win.
- OR stack + last-prior-winner: 9 events, +7.674R, mean +0.853R, 100.0% win.

Train pre-2026 check:

- OR stack all: 72 events, +24.614R, mean +0.342R, 69.4% win.
- Tier A one-win warm: 22 events, +4.684R, mean +0.213R, 63.6% win.
- Tier B not one-win warm: 50 events, +19.930R, mean +0.399R, 72.0% win.

Conclusion: one-win warmth is useful as a priority label inside the OR stack. It is not robust enough as a standalone expansion gate.

A union test confirmed this. Adding one-win outside OR stack weakened forward results:

- tier1 OR stack: 29 events, +21.775R.
- OR stack OR one-win stable/no-conflict: 33 events, +20.599R. Add-on only was -1.176R.
- OR stack OR one-win anytime: 42 events, +20.572R. Add-on only was -1.203R.
- OR stack OR first-winner same-phase warm: 36 events, +20.075R. Add-on only was -1.700R.
- OR stack OR last-event-winner: 38 events, +17.748R. Add-on only was -4.027R.

Conclusion: do not use one-win as a Tier 2 expansion layer. Use it only to rank or prioritize OR-stack entries.

## Source-Visible Fast-Trigger And Early-Entry Tests

Question tested: can source-visible signals let us enter the same OR-stack parent-loop event earlier?

Direct source-to-OR-stack bridge, forward 2026:

- Exact future-event bridge: 0/29 covered.
- Same future parent loop <= 1h, 4h, 24h, 120h: 0/29 covered.
- Same future loop + same regime <= 1h, 4h, 24h, 120h: 0/29 covered.
- Same source loop <= 1h, 4h, 24h, 120h: 0/29 covered.
- Loosest same-symbol-any-loop <= 24h: 2/29 covered, but both were different parent loops and not usable as early entry into the OR-stack event.

Conclusion: source-visible timing does not currently bridge into OR-stack accepted events. It cannot validate entering those OR-stack events earlier.

Source-scored fast-trigger grid:

Rows were selected by training-side source-entry R, not by forward result.

- 2025_to_2026, 5d, threshold 1.0, all top1: train source +1.755R over 9; forward source 0 events.
- 2025_to_2026, 10d, threshold 1.07, all top2: train source +3.716R over 11; forward source 0 events.
- pre2026_to_2026, 5d, threshold 1.1, all top1: train source +3.252R over 51; forward source 0 events.
- pre2026_to_2026, 10d, threshold 1.05, all top1: train source +5.872R over 24; forward source 0 events.

Conclusion: the source-entry layer is not currently train-source-selectable.

Important contrast: an earlier confirmation-scored harness showed +5.200R source-entry forward, but the training-side source R for that row was negative:

- 2025_to_2026, 5d, threshold 1.07, all top2: train source -9.499R over 64; forward source +5.200R over 8; forward confirmed -0.827R.

Conclusion: the +5.2R source-entry observation is not evidence of a stable source-entry predictor. Do not chase source-visible early entry as the main route right now.

## Parent Loop / Child / Morph Discovery Handover

Current thesis:

```text
template = parent loop + child/morph identity + prediction gate + admission gate
```

Parent-loop templates alone are too coarse. The useful ladder is:

```text
parent_loop
-> child/morph bucket
-> prediction gate
-> admission gate
-> blocker audit
-> forward survival ranking
```

Blockers must remain separate from candidate/admission. Previous child/morph blocker tests blocked too many winners.

### Branch 1: pullback_resolution

Prior participation gate works.

Feature-rich Branch 1 base:

- 74 events
- +21.57R
- mean +0.292R
- 64.9% win
- all years positive

Prior-only participation gate:

- 31 events
- +15.25R
- mean +0.492R
- 80.6% win
- negative symbol-month share 20.0%
- years positive: 2023 +1.80R, 2024 +5.35R, 2025 +4.93R, 2026 +3.17R

Child/morph hard gates and blockers did not improve forward. Keep child/morph diagnostic for Branch 1 until the staged matrix is applied.

### Branch 2: chop_breakout_or_decay

Best simple prior-only gate so far:

```text
same_phase_win_rate >= 0.65
```

Fixed diagnostic:

- 20 events
- +14.40R
- mean +0.720R
- 90.0% win

Honest expanding-year test:

- 2026 learned same_phase_win_rate >= 0.65 from prior rows
- 9 accepted events
- +6.27R
- mean +0.697R
- 88.9% win

Child/morph hard gates became too narrow and child/morph blockers blocked winners. Keep child/morph diagnostic for Branch 2 for now.

### Branch 3: liquidation_reclaim_or_recoil

Whole branch is weak as a single unit and needs parent-loop treatment.

Base Branch 3:

- 83 events
- +15.43R
- mean +0.186R
- 62.7% win
- all years positive, but 2025 nearly flat

Prior participation gate was worse:

- 35 events
- +4.47R
- mean +0.128R
- 2024 went negative

Best controlled parent loop tested:

```text
failed_bounce_active_liquidation__to__liquidation_failed_low_reclaim
```

Baseline for that parent loop:

- 19 events
- +5.34R
- mean +0.281R
- 73.7% win
- all tested years positive

Best staged child/morph matrix result:

```text
morph == morph_4
-> broad_cycle_share >= ~0.369
-> no extra admission gate
```

Full-sample diagnostic:

- 6 events
- +5.40R
- mean +0.900R
- 100% win
- appeared in 2024, 2025, and 2026

Corrected prior-only forward loose result:

- 6 events
- +3.18R
- mean +0.529R
- 83.3% win
- 2025 +2.28R
- 2026 +0.90R

Balanced/strict version:

- 1 event
- +0.90R
- too narrow

Conclusion: Branch 3 should not be promoted as a whole branch. Continue parent-loop-specific staged matrix testing.

## Latest Temp Outputs Behind This Summary

These are local paths from the Codex machine. They are listed for reproducibility but are not assumed accessible to ChatGPT:

```text
/private/tmp/stocker_one_win_warm_state_idea_sweep_20260709/summary.md
/private/tmp/stocker_one_win_warm_state_idea_sweep_20260709/tier2_union_summary.md
/private/tmp/stocker_or_stack_robustness_candidate_test_20260709/summary.md
/private/tmp/stocker_or_stack_one_win_priority_tier_test_20260709/summary.md
/private/tmp/stocker_source_visible_fast_trigger_bridge_test_20260709/summary.md
/private/tmp/stocker_child_morph_gate_matrix_fast_test_20260708/summary.json
```

## Recommended Next Step

Do not go further down the source-visible early-entry path as the main route.

The best next research step is:

```text
Run a reusable staged child/morph gate matrix across parent loops.
```

Ladder:

```text
parent_loop
-> child/morph bucket
-> prediction gate
-> admission gate
-> blocker audit
-> forward survival ranking
```

Start with Branch 3 parent loops, then apply to Branch 1 and Branch 2.

Minimum output columns:

```text
parent_loop
branch_type
child_morph_bucket
prediction_gate
admission_gate
blocker_audit
train_events
forward_events
forward_symbols
forward_total_r
forward_mean_r
forward_win_rate
forward_negative_symbol_month_share
forward_years_positive
forward_years_negative
selected_years
rejected_reason
edge_claimed
research_only
```

Fail-closed gates:

- minimum prior events before selecting a rule
- minimum selected train events
- minimum selected train symbols
- max single-symbol share
- positive train total R
- selected train mean R better than parent baseline
- forward result reported even if it collapses

Do not force a result. If a child/morph bucket has too little history, mark it as insufficient support and do not select it.

## Prompt For Next ChatGPT Session

```text
Use the Stocker Research connector or this GitHub file.
Read docs/chatgpt_mcp_export_20260709/summary.md first.
Continue research-only.
Do not claim an edge.
Do not touch broker execution, IG, live trading, paper trading, order placement, API keys, deployment, or runtime trading paths.
Keep candidate/admission separate from blocker/no-trade screening.
Focus on liquid midcap / broader SMID.
Next task: run a reusable staged child/morph gate matrix across parent loops, starting with Branch 3 parent loops, then Branch 1 and Branch 2.
```

## Safety Footer

This summary is research-only. It does not claim an edge. It does not approve live trading, paper trading, order placement, broker execution, deployment, runtime trading changes, or YAML rule promotion.
