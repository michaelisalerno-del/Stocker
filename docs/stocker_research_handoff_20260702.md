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

Blanket removal of the prior-replay gate is worse in Jan-Jun 2026 because it admits losing `slow_repair` trades. But the user is probably right that personalities are preferable only in certain contexts.

For `slow_repair`, post-hoc tests found:

- Admit all `slow_repair` in Jan-Jun 2026: 22 trades, `-1.7959R`.
- Admit `slow_repair` only when `volume_x_vwap_regime == low_relative_volume|below`:
  - Jan-Jun 2026 added 7 trades, `+2.3163R`.
  - Portfolio becomes 34 trades, `+19.7368R`.
  - Prior Aug 2025-Jan 2026 window: 10 filtered `slow_repair` trades, `+5.2408R`; blocked remainder was `-3.8195R`.
- A broader candidate, `low_relative_volume|below OR return_zscore >= 0.497`, added 9 Jan-Jun trades and `+3.1476R`, portfolio `+20.5680R`, but needs stricter train/OOS validation before coding as logic.

Interpretation: the next research layer should be personality-specific caveat/admission, not a global blocker.

## Recommended Next Task

Build/test a `personality_context_caveat_lab_v0` or extend staged runner with train-selected personality-specific admission rules:

- For each personality, learn blocker and re-entry/admission rules on train only.
- Evaluate OOS and random same-count baselines.
- Classify rules:
  - `strict_train_and_oos_supported`
  - `oos_only_not_train_supported`
  - `train_only_not_oos_supported`
  - `not_supported`
- Treat admission as research-only. Do not promote as live logic.
- First candidate to validate: `slow_repair` re-entry when `volume_x_vwap_regime == low_relative_volume|below`.

## Verification Completed

- Full pytest passed after latest code changes.
- Staged runner focused tests passed.
- Ruff passed with `--no-cache`.
- `git diff --check` passed.
