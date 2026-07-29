# Child/Morph Template Discovery Handover - 2026-07-08

Research-only handover for Stocker template discovery work on branch
`codex-state-event-detector-v0`.

No edge is claimed. Nothing here authorizes live trading, broker execution,
paper trading, order placement, deployment, or runtime trading changes.

## Current Thesis

The useful unit is no longer just:

```text
template = parent loop
```

It is closer to:

```text
template = parent loop + child/morph identity + prediction gate + admission gate
```

The research process has converged toward this ladder:

```text
loop discovery
-> branch / parent loop
-> child/morph identity
-> prediction environment
-> admission quality
-> blocker audit
-> forward replay
```

The central finding remains:

```text
useful containers are mostly source-visible location + participation + sometimes tempo/structure
not broad semantic market labels
```

But the latest refinement is that child/morph variants need their own
prediction/admission structure. Treating child/morphs as raw hard gates or raw
blockers was too crude.

## Hard Research Constraints To Preserve

- Research-only.
- Do not touch broker execution, IG, live trading, paper trading, order placement, API keys, deployment, or runtime trading paths.
- Do not claim an edge.
- Preserve uncommitted work.
- Use existing local 5m OHLCV/event-report data unless explicitly asked to download/generate more.
- Keep candidate/admission work separate from blockers/no-trade screening.
- Keep exits separate from directional discovery until a stable directional split exists.
- Do not save YAML rules unless explicitly asked.
- Focus on liquid midcap / broader SMID.
- Large caps are only a negative-control / transfer diagnostic for now.

## Repo State At Handover

Repo:

```text
/Users/michaelsalerno/Documents/Codex/2026-06-30-we-are-working-in-my-stocker-2
```

Branch:

```text
codex-state-event-detector-v0
```

Start next chat with:

```bash
rtk git -C /Users/michaelsalerno/Documents/Codex/2026-06-30-we-are-working-in-my-stocker-2 status --short --branch
```

There is existing uncommitted work. Do not revert it. Recent status included
modified research/core files and many untracked research modules/tests/docs.

Important repo files to inspect when needed:

```text
packages/stocker_research/src/stocker_research/template_discovery_system_v0.py
packages/stocker_research/src/stocker_research/frozen_template_technique_v0.py
packages/stocker_research/src/stocker_research/loop_variant_optimizer_v0.py
packages/stocker_research/src/stocker_research/loop_morph_transition_map_v0.py
packages/stocker_research/src/stocker_research/pre_loop_discovery_v0.py
packages/stocker_core/src/stocker_core/cli.py
tests/test_template_discovery_system_v0.py
tests/test_frozen_template_technique_v0.py
tests/test_loop_variant_optimizer_v0.py
tests/test_loop_morph_transition_map_v0.py
tests/test_pre_loop_discovery_v0.py
```

Earlier docs worth reading:

```text
docs/stocker_template_discovery_system_handover_20260704.md
docs/stocker_personality_container_handover_20260704.md
docs/personality_acceptance_compartments_20260704.md
docs/template_branch_candidate_notes_20260705.md
```

## Important Prior Fix Context

The packaged frozen-template transfer replay was fixed to avoid lookahead:

- Source-context / next-event rules must not score the earlier source event after using future confirmation.
- Confirmation-dependent rules should enter at confirmation-event timestamp.
- Unsafe future-dependent templates should be excluded from headline R.
- Headline R should default to conservative target-capped R, with final-close R only diagnostic.
- State detector rejection should block packaged continuation by default.
- Manual audit should not be disabled by default.

Do not tune thresholds to preserve any old large R result. It is acceptable if
R collapses after honest replay fixes.

## Key Temporary Inputs

Most recent research used these local outputs:

```text
/private/tmp/stocker_same_phase_symbol_admission_sweep_20260708/deduped_strict_same_phase_events.csv
/private/tmp/stocker_deduped_branch_split_participation_test_20260708/deduped_annotated_branch_events.csv
/private/tmp/stocker_branch3_condition_forward_test_20260708/branch3_full_feature_rows.csv
```

The full same-phase file has the rich feature columns. The branch label file
has branch/transition/participation decisions but fewer feature columns. For
feature-level branch tests, merge them by:

```text
current_loop, symbol_norm, entry_timestamp, _r
```

Always dedupe on that event key before scoring. Earlier branch results were
inflated until duplicate event keys were removed.

## Branch Summary

### Branch 1: Pullback Resolution

Branch type:

```text
pullback_resolution
```

Base branch rows from the feature-rich Branch 1 test:

- 74 events
- +21.57R
- mean +0.292R
- win rate 64.9%
- all years positive

Prior-only participation gate:

- 31 events
- +15.25R
- mean +0.492R
- win rate 80.6%
- negative symbol-month share 20.0%
- years positive:
  - 2023: +1.80R
  - 2024: +5.35R
  - 2025: +4.93R
  - 2026: +3.17R

Interpretation:

- Branch 1 likes broad/local participation gating.
- Child/morph as primary gate did not improve it forward.
- Child/morph as blocker did not improve it forward.
- Child/morph should stay diagnostic for Branch 1 until a better child-level
  prediction/admission matrix is tested.

Branch 1 child/morph report:

```text
/private/tmp/stocker_branch1_child_morph_forward_test_20260708/summary.json
```

### Branch 2: Chop Breakout Or Decay

Branch type:

```text
chop_breakout_or_decay
```

Base Branch 2 after key dedupe:

- 40 events
- +11.06R
- mean +0.276R
- win rate 70.0%
- 2026 was almost flat without filtering

Best simple Branch 2 gate:

```text
same_phase_win_rate >= 0.65
```

Fixed diagnostic:

- 20 events
- +14.40R
- mean +0.720R
- win rate 90.0%

Honest expanding-year test:

- 2024: no trade, no prior data
- 2025: no trade, only 2 prior rows
- 2026: learned `same_phase_win_rate >= 0.65` from prior rows
- 9 accepted events
- +6.27R
- mean +0.697R
- win rate 88.9%

Child/morph tests:

- Child/morph as primary hard gate became too narrow.
- Child/morph blockers learned from prior rows blocked winners.
- Keep child/morph as diagnostics for Branch 2 for now.

Reports:

```text
/private/tmp/stocker_chop_branch_condition_search_20260708/summary.json
/private/tmp/stocker_chop_branch_forward_threshold_test_20260708/summary.json
/private/tmp/stocker_chop_child_morph_forward_test_20260708/summary.json
/private/tmp/stocker_chop_branch_blocker_forward_test_20260708/summary.json
```

### Branch 3: Liquidation Reclaim Or Recoil

Branch type:

```text
liquidation_reclaim_or_recoil
```

Base Branch 3:

- 83 events
- +15.43R
- mean +0.186R
- win rate 62.7%
- all years positive, but 2025 nearly flat

Prior participation gate was worse:

- 35 events
- +4.47R
- mean +0.128R
- 2024 went negative

Forward selectors were weak:

- source-visible forward: 5 events, +1.84R
- same-phase/prior forward: 24 events, +3.28R, but 2025 was -1.51R
- child/morph forward: 8 events, about flat

Conclusion:

- Do not promote Branch 3 as one branch.
- Branch 3 needs individual parent-loop treatment.

Branch 3 report:

```text
/private/tmp/stocker_branch3_condition_forward_test_20260708/summary.json
```

## Controlled Child/Morph Matrix Test

The first proper child/morph prediction/admission matrix was tested on the best
Branch 3 loop:

```text
failed_bounce_active_liquidation__to__liquidation_failed_low_reclaim
```

Baseline for that parent loop:

- 19 events
- +5.34R
- mean +0.281R
- win rate 73.7%
- all tested years positive

Staged matrix structure:

```text
child/morph bucket -> prediction gate -> admission gate
```

Best full-sample diagnostic:

```text
morph == morph_4
-> broad_cycle_share >= ~0.369
-> no extra admission gate
```

Full-sample diagnostic:

- 6 events
- +5.40R
- mean +0.90R
- win rate 100%
- appeared in 2024, 2025, and 2026

Corrected prior-only forward result:

Loose:

- 6 events
- +3.18R
- mean +0.529R
- win rate 83.3%
- years:
  - 2025: +2.28R
  - 2026: +0.90R

Balanced/strict:

- 1 event
- +0.90R
- too narrow

Important bug caught and fixed in the temporary script:

- First version accidentally overwrote the child/morph bucket with `none`.
- Corrected version preserves `morph == morph_4` as the bucket.

Report:

```text
/private/tmp/stocker_child_morph_gate_matrix_fast_test_20260708/summary.json
```

## Current Interpretation

1. Parent branches are not enough.
2. Child/morph identity matters, but only with a prediction environment.
3. Admission gates are not always needed; the first controlled matrix found:

```text
morph_4 + broad_cycle_share
```

with no extra admission gate.

4. Full-sample child/morph pockets can look excellent, but must be checked with
prior-only forward selection.
5. The right next system should automate the staged ladder:

```text
parent loop
-> child/morph bucket candidates
-> prediction gate candidates
-> admission gate candidates
-> blocker/no-trade audit
-> forward survival ranking
```

6. Blockers must remain separate. Previous child/morph blocker tests blocked
too many winners.

## Where To Go Next

The next best step is not another one-off manual branch test. It is to build or
run a reusable staged matrix across parent loops:

```text
for each parent_loop:
  find child/morph buckets with enough support
  for each child/morph bucket:
    find prediction gates using prior/source-visible environment
    find admission gates using same-phase/prior/source-visible quality
    evaluate forward-year survival
    record rejected blockers separately
```

Recommended starting scope:

- Run the staged matrix across all parent loops inside Branch 3 first.
- Then run the same across Branch 1 and Branch 2.
- Rank results by forward survival, not full-sample score.

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

Suggested fail-closed gates:

- minimum prior events before selecting a rule
- minimum selected train events
- minimum selected train symbols
- max single-symbol share
- positive train total R
- selected train mean R better than parent baseline
- forward result reported even if it collapses

Do not force a result. If a child/morph bucket has too little history, mark it
as insufficient support and do not trade/select it.

## Copy/Paste Prompt For Next Chat

```text
We are working in my Stocker repo.

Repo path:
/Users/michaelsalerno/Documents/Codex/2026-06-30-we-are-working-in-my-stocker-2

Branch:
codex-state-event-detector-v0

Start with:
rtk git -C /Users/michaelsalerno/Documents/Codex/2026-06-30-we-are-working-in-my-stocker-2 status --short --branch

Use rtk for shell commands.

Research-only constraints:
- Do not touch broker execution, IG, live trading, paper trading, order placement, API keys, deployment, or runtime trading paths.
- Do not claim an edge.
- Preserve uncommitted work.
- Use existing local 5m OHLCV/event-report data unless I explicitly ask to download/generate more.
- Keep candidate/admission work separate from blocker/no-trade screening.
- Keep exits separate from directional discovery until a stable directional split exists.
- Do not save YAML rules unless I explicitly ask.
- Focus on liquid midcap / broader SMID.
- Large caps are only negative-control / transfer diagnostic for now.

Read this handover first:
docs/child_morph_template_discovery_handover_20260708.md

Useful temporary reports:
/private/tmp/stocker_deduped_branch_split_participation_test_20260708/summary.json
/private/tmp/stocker_branch1_child_morph_forward_test_20260708/summary.json
/private/tmp/stocker_chop_branch_condition_search_20260708/summary.json
/private/tmp/stocker_chop_branch_forward_threshold_test_20260708/summary.json
/private/tmp/stocker_chop_child_morph_forward_test_20260708/summary.json
/private/tmp/stocker_chop_branch_blocker_forward_test_20260708/summary.json
/private/tmp/stocker_branch3_condition_forward_test_20260708/summary.json
/private/tmp/stocker_child_morph_gate_matrix_fast_test_20260708/summary.json

Where we are:
We found that parent-loop templates are too coarse. The likely unit is:
parent loop + child/morph identity + prediction gate + admission gate.

Branch 1 pullback_resolution:
- Prior participation gate works.
- Child/morph hard gates/blockers did not improve forward.

Branch 2 chop_breakout_or_decay:
- same_phase_win_rate >= 0.65 is the best simple prior-only gate so far.
- Child/morph hard gates/blockers did not improve forward.

Branch 3 liquidation_reclaim_or_recoil:
- Whole branch is weak as a single unit.
- Needs parent-loop treatment.
- Best controlled parent loop tested:
  failed_bounce_active_liquidation__to__liquidation_failed_low_reclaim

Latest important result:
For that Branch 3 parent loop, a staged child/morph matrix found:
morph == morph_4 -> broad_cycle_share >= ~0.369 -> no extra admission gate

Full-sample diagnostic:
- 6 events
- +5.40R
- 100% win

Prior-only forward loose result:
- 6 events
- +3.18R
- mean +0.529R
- 83.3% win

Next task:
Build or run a reusable staged child/morph gate matrix across parent loops.
The ladder should be:
parent_loop -> child/morph bucket -> prediction gate -> admission gate -> blocker audit -> forward survival ranking.

Start with Branch 3 parent loops, then apply to Branch 1 and Branch 2.
Do not code into the repo until you inspect existing modules and propose the smallest integration point. If using temporary scripts first, write outputs under /private/tmp and clearly report that no repo files changed.
```

## Safety Footer

This handover is research-only. It does not claim an edge. It does not approve
live trading. It does not define broker, paper, live, deployment, or runtime
trading behavior.
