# Stocker Research Handoff - 2026-07-02

## Workspace

- Codex repo: `/Users/michaelsalerno/Documents/Codex/2026-06-30-we-are-working-in-my-stocker-2`
- MCP-visible local copy: `/Users/michaelsalerno/StockerLocal`
- Branch: `codex-state-event-detector-v0`
- Research-only constraints remain in force: no broker execution, IG, live trading, paper trading, order placement, vendor fetching, or edge claims.
- Pipeline under investigation: `personality -> mixed regime -> filter -> caveat/admission -> exit`.

## What Was Built

- Reusable bad-trade sequence caveat CLI/reporting.
- Conditional context caveat lab.
- Personality expression lab.
- Pre-registered edge proof report scaffold.
- Shadow candidate trigger audit.
- State lifecycle context lab.
- Staged mixed-regime caveat/exit walk-forward runner.
- Personality-specific context admission lab:
  - `stocker research personality-context-admission`
  - train/test month split, candidate-only diff, same-count random baselines, and strict/OOS support labels.
- Blank-slate personality context rule discovery lab:
  - `stocker research personality-context-rule-discovery`
  - generates simple single/AND/OR context rules from report-pair features without hardcoding the hand-designed `slow_repair` rule.
- Staged runner fixes:
  - unscored warmup months;
  - less aggressive prior-replay default gate;
  - personality-floor exit-sweep cap so valid personalities are not crowded out before selection.

## Key Reports

- Main earlier selected-filter result:
  - `data/reports/research/walk_forward_selected_filter_exit_v0/walk_forward_selected_filter_exit_v0_20260701T181229Z`
  - 130 trades, `+43.3414R`, 64.62% win rate.
  - Broader and less defensive than the staged pipeline.

- Old staged run:
  - `data/reports/research/walk_forward_staged_mixed_regime_caveat_exit_v0/walk_forward_staged_mixed_regime_caveat_exit_v0_20260701T205047Z`
  - 37 trades, `+9.4287R`, mostly `open_down_pressure`.

- Current full staged hard-prior-gate run after personality floor:
  - `data/reports/research/full_staged_mixed_regime_pipeline_v0_personality_floor/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T110135Z`
  - Jan-Jun 2026 replay, Jul-Dec 2025 warmup.
  - 27 trades, `+17.4204R`, 77.78% win rate, 5/6 positive months.
  - Decision: `continue_research_sparse_high_quality`.
  - `slow_repair` is blocked by prior replay.

- No-prior-gate comparison:
  - `data/reports/research/full_staged_mixed_regime_pipeline_v0_no_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T111703Z`
  - 49 trades, `+15.6245R`, 63.27% win rate.
  - Adds 22 `slow_repair` trades, but those add `-1.7959R`.

- Random six-month staged audit:
  - `data/reports/research/random_6m_staged_audit_v0_prior_replay_fallback_warmup_fixed_defaults/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T104545Z`
  - Aug 2025-Jan 2026, 39 trades, `+5.3001R`.

## Current Research Finding

Blanket removal of the prior-replay gate is worse in Jan-Jun 2026 because it admits losing `slow_repair` trades. But `slow_repair` is context-dependent, so the right next layer is personality-specific context admission, not a global blocker.

Manual/post-hoc `slow_repair` findings:

- Admit all `slow_repair` in Jan-Jun 2026: 22 trades, `-1.7959R`.
- `volume_x_vwap_regime == low_relative_volume|below` helped in Jan-Jun 2026 and Aug 2025-Jan 2026, but later windows showed it was not robust enough by itself.
- Stronger manual candidate:
  - `prior_3_bar_return >= 0.00819399 AND prev_event_personality != dead_chop_noise`
  - Good precision, but missed many good trades.
- Broader manual candidate `slow_repair_admit_v1_1`:
  - `(prior_3_bar_return >= 0.00819399 AND prev_event_personality != dead_chop_noise) OR (prev_event_personality == reclaim_reversal AND distance_from_recent_high_pct >= -0.0199422 AND volume_x_vwap_regime != high_relative_volume|above)`
  - Across eight paired windows, candidate-only re-entry admitted 15 trades, `+9.4653R`, with no negative admitted windows.
  - This remains research-only and is not promoted as live logic.

Blank-slate coded discovery then tested whether this class of rule can be found without starting from the manual expression.

Report:

- `data/reports/research/personality_context_rule_discovery_v0_slow_repair/personality_context_rule_discovery_v0_20260702T140930Z`
- Decision: `continue_research_blank_slate_context_rule_supported`
- Report pairs: 8
- Target personality: `slow_repair`
- Selected rules: 255
- Safety: `research_only: true`, `live_ordering_enabled: false`, `order_placement: disabled`, `edge_claimed: false`

Top blank-slate discovered rule:

```text
(prev_event_personality != dead_chop_noise AND relative_cumulative_volume <= 0.5)
OR
(prev_event_personality != reclaim_reversal AND distance_from_recent_high_pct <= -0.0209511)
```

Observed result:

- Candidate-only re-entry: 11 trades, `+7.5676R`
- Candidate-only positive windows: 2
- No-prior `slow_repair` slice: 26 trades, `+17.4871R`
- No-prior positive windows: 7
- No-prior max single-window share: `0.2681`
- Support status: `supported_candidate_only_reentry`

Interpretation: the blank-slate search can discover personality-specific context rules from report-pair features. It did not simply rediscover the hand-written `v1_1` as the top rule, which is useful evidence that the code is genuinely searching. These are still post-hoc research candidates and need frozen validation on different windows before any staged-runner integration.

## Saved Discovery Template Settings

Use this template for the next personality. Change only `--target-personalities` first; keep the report pairs and discovery thresholds stable for comparability.

```bash
rtk uv run stocker research personality-context-rule-discovery \
  --output-dir data/reports/research/personality_context_rule_discovery_v0_<personality> \
  --report-pair 2024H2=data/reports/research/random_6m_staged_audit_v0_2024h2_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T122143Z,data/reports/research/random_6m_staged_audit_v0_2024h2_no_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T122226Z \
  --report-pair 2024_11_2025_04=data/reports/research/random_6m_staged_audit_v0_second_sample_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T121442Z,data/reports/research/random_6m_staged_audit_v0_second_sample_no_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T121629Z \
  --report-pair 2025_02_2025_07=data/reports/research/random_6m_staged_audit_v0_2025_feb_jul_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T122518Z,data/reports/research/random_6m_staged_audit_v0_2025_feb_jul_no_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T122811Z \
  --report-pair 2025_05_2025_10=data/reports/research/random_6m_staged_audit_v0_2025_may_oct_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T125945Z,data/reports/research/random_6m_staged_audit_v0_2025_may_oct_no_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T124359Z \
  --report-pair 2025_07_2025_12=data/reports/research/random_6m_staged_audit_v0_2025_jul_dec_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T135512Z,data/reports/research/random_6m_staged_audit_v0_2025_jul_dec_no_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T140005Z \
  --report-pair 2025_08_2026_01=data/reports/research/random_6m_staged_audit_v0_prior_replay_fallback_warmup_fixed_defaults/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T104545Z,data/reports/research/random_6m_staged_audit_v0_no_prior_gate_for_admission/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T115141Z \
  --report-pair 2025_11_2026_04=data/reports/research/random_6m_staged_audit_v0_2025_nov_2026_apr_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T133811Z,data/reports/research/random_6m_staged_audit_v0_2025_nov_2026_apr_no_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T134421Z \
  --report-pair 2026H1=data/reports/research/full_staged_mixed_regime_pipeline_v0_personality_floor/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T110135Z,data/reports/research/full_staged_mixed_regime_pipeline_v0_no_prior_gate/walk_forward_staged_mixed_regime_caveat_exit_v0_20260702T111703Z \
  --target-personalities <personality> \
  --categorical-features prev_event_personality,volume_x_vwap_regime,time_x_vwap_regime \
  --numeric-features prior_3_bar_return,distance_from_recent_high_pct,relative_cumulative_volume,same_direction_other_symbol_count_15m,same_personality_other_symbol_count_15m,close_location_value \
  --min-rule-trades 3 \
  --min-rule-windows 2 \
  --min-positive-windows 2 \
  --max-negative-windows 0 \
  --max-single-window-share 0.65 \
  --random-iterations 1000
```

Report pairs used for the saved template, with abbreviated names:

- `2024H2`: prior `random_6m_staged_audit_v0_2024h2_prior_gate/...T122143Z`, no-prior `random_6m_staged_audit_v0_2024h2_no_prior_gate/...T122226Z`
- `2024_11_2025_04`: prior `random_6m_staged_audit_v0_second_sample_prior_gate/...T121442Z`, no-prior `random_6m_staged_audit_v0_second_sample_no_prior_gate/...T121629Z`
- `2025_02_2025_07`: prior `random_6m_staged_audit_v0_2025_feb_jul_prior_gate/...T122518Z`, no-prior `random_6m_staged_audit_v0_2025_feb_jul_no_prior_gate/...T122811Z`
- `2025_05_2025_10`: prior `random_6m_staged_audit_v0_2025_may_oct_prior_gate/...T125945Z`, no-prior `random_6m_staged_audit_v0_2025_may_oct_no_prior_gate/...T124359Z`
- `2025_07_2025_12`: prior `random_6m_staged_audit_v0_2025_jul_dec_prior_gate/...T135512Z`, no-prior `random_6m_staged_audit_v0_2025_jul_dec_no_prior_gate/...T140005Z`
- `2025_08_2026_01`: prior `random_6m_staged_audit_v0_prior_replay_fallback_warmup_fixed_defaults/...T104545Z`, no-prior `random_6m_staged_audit_v0_no_prior_gate_for_admission/...T115141Z`
- `2025_11_2026_04`: prior `random_6m_staged_audit_v0_2025_nov_2026_apr_prior_gate/...T133811Z`, no-prior `random_6m_staged_audit_v0_2025_nov_2026_apr_no_prior_gate/...T134421Z`
- `2026H1`: prior `full_staged_mixed_regime_pipeline_v0_personality_floor/...T110135Z`, no-prior `full_staged_mixed_regime_pipeline_v0_no_prior_gate/...T111703Z`

## Recommended Next Task

Move to the next personality using the saved blank-slate discovery template. If the user has not named the personality, first rank personalities by candidate-only count across the same eight report pairs and start with the largest unresolved blocker.

For each next personality:

- Run the discovery template with that personality target.
- Inspect top `supported_candidate_only_reentry` rules.
- Compare candidate-only result, no-prior result, same-count random median, and concentration.
- Freeze one or two candidate rules only if they survive the support filters.
- Validate frozen rules on at least one different/random six-month window.
- Keep everything research-only. Do not wire to live, paper, broker, vendor fetch, or order placement.

## Paste-Ready Next Chat Prompt

```text
We are working in my Stocker repo.

Codex repo path:
/Users/michaelsalerno/Documents/Codex/2026-06-30-we-are-working-in-my-stocker-2

MCP-visible StockerLocal path:
/Users/michaelsalerno/StockerLocal

Branch:
codex-state-event-detector-v0

Start with:
rtk git status

Research-only constraints:
- Do not touch broker execution, IG, live trading, paper trading, order placement, or vendor fetching.
- Use existing local 5m OHLCV data and existing research report outputs only.
- Do not claim an edge.
- Preserve uncommitted work.

Read handoff first:
docs/stocker_research_handoff_20260702.md

Where we are:
- The staged pipeline is investigating:
  personality -> mixed regime -> filter -> caveat/admission -> exit
- Blanket no-prior admission is worse for slow_repair in Jan-Jun 2026.
- slow_repair is context-dependent.
- A research-only blank-slate personality context rule discovery lab has been added:
  stocker research personality-context-rule-discovery
- The slow_repair discovery run is:
  data/reports/research/personality_context_rule_discovery_v0_slow_repair/personality_context_rule_discovery_v0_20260702T140930Z
- Top discovered slow_repair rule:
  (prev_event_personality != dead_chop_noise AND relative_cumulative_volume <= 0.5)
  OR
  (prev_event_personality != reclaim_reversal AND distance_from_recent_high_pct <= -0.0209511)
- It admitted 11 candidate-only re-entry trades for +7.5676R, with 2 candidate-only positive windows and no negative admitted candidate-only windows.
- This is still post-hoc research only, not live logic.

Next task:
Move on to the next personality. Use the saved discovery template settings in the handoff. If I do not specify the personality, rank personalities by candidate-only count across the same eight report pairs and start with the largest unresolved blocker. Compare top rules against same-count random baselines and concentration, then suggest which rule, if any, deserves frozen OOS validation.
```

## Verification Completed

- Focused personality-context tests passed:
  - `python3 -m pytest tests/test_personality_context_admission_v0.py -q -p no:cacheprovider`
- Ruff passed with `--no-cache` for the changed research, CLI, and test files.
- `git diff --check` passed.
