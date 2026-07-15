# Dynamic loop edge-state lead-lag V1

## Baseline

- Branch: `agent/slrno-research-handoff`
- Frozen pre-V2 baseline: `8baf974f2d13751064dbc4d2c7cf65d02e3a8912`
- Frozen scored V2 implementation: `ca3537a0f337097a9a75abf87ae4bf419fae6a5d`
- V2 focused suite before edits: `37 passed`.
- Arbor evaluator: unavailable in this repository; root tests and frozen V2 artifacts are the applicable evaluators.
- Safety boundary: research only; live ordering, brokers, paper/demo execution, deployment, orders, positions, and frozen exits are out of scope.

## Exact V2 delay reconstruction before new implementation

The V2 `_delayed_policy` did not move an original opportunity or entry. It shifted the prior *opportunity-session* accepted flag within each period × loop × orientation × horizon and applied it to current opportunities, current entry clocks, and current 24-bar outcomes. Missing cell-opportunity sessions were skipped, so the source policy lag ranged from 1 to 26 calendar sessions.

- Immediate: 286 accepted signals, 275 fills, -9,416.3830 bps.
- Shifted policy: 268 accepted signals, 259 fills, +9,014.5353 bps.
- Retained: 55 signals, 53 fills, -5,167.6341 bps.
- Dropped: 231 signals, 222 fills, -4,248.7489 bps.
- Introduced: 213 signals, 206 fills, +14,182.1694 bps.
- Exact delta: introduced minus dropped = +18,430.9183 bps.
- There was no overlap resolver, capacity allocator, changed entry clock, changed holding period, or changed cost on a retained row.

## Registered hypotheses

1. **Frozen feature overlay leads state by one session.** Target files: new lead-target/metrics module and runner. Benefit: distinguish prediction timing from same-session failure. Safety risk: target-session leakage. Validation: synthetic lead shift, explicit-calendar joins, appended-future invariance, paired lead-1 bootstrap. Stop: reject if full does not improve paired lead-1 calibration and economic state value versus the identical no-feature hierarchy.
2. **V2 sign reversal is opportunity-population attribution, not delayed execution.** Target files: matching/decomposition module and auditor. Benefit: explain the exact +18,430.92 bps delta. Safety risk: calling a replacement setup a delayed trade. Validation: exact reconstruction and immutable opportunity IDs. Stop: classify as population-confounded if introduced-minus-dropped rows explain the delta.
3. **A persistent exact setup survives to the next session.** Target files: exact matcher. Benefit: executable timing counterfactual. Safety risk: treating same-loop rows as the same setup. Validation: require persistent opportunity or event-lineage identity; structural lineage is separate. Stop: tradeability unknown if no exact matches exist.
4. **Only a structurally led episode subtype responds.** Target files: episode attribution. Benefit: narrower descriptive mechanism. Safety risk: hindsight labels entering forecasts. Validation: episode labels joined only after immutable forecasts. Stop: descriptive/unknown if non-monotonic, concentrated, or post-onset.
5. **Prospective logging remains immutable and execution-free.** Target files: immutable-ledger module and research CLI mode. Benefit: enable genuinely new-session follow-up. Safety risk: runtime coupling or forecast revision. Validation: create-only IDs, separate outcome appends, safety tests. Stop: fail closed on duplicate or revised IDs.

## Pre-agreed public test seams

- Explicit-session forecast-to-target lead join.
- Paired full/no-feature population and metrics.
- Exact-setup and structural-lineage matching.
- Original delay population decomposition.
- Immutable forecast/outcome ledger append API.
- Independent auditor output.

## Commands and results

- The first primary command at implementation SHA `07be73486fe8cd658261a65ea287be4318fb8b47`
  stopped before creating the output directory. A reporting-only conversion tried to cast the
  nullable mean of an empty active slice (`pd.NA`) to `float`. No model, threshold, feature,
  target, or metric changed and no artifact was written.
- The correction maps only that empty descriptive rate to missing (`NaN`). Every model and
  comparator is regenerated from scratch under the subsequent committed SHA.
- The next primary command at SHA `00090ce69f300e137598563bfeb3496622614bf4` also stopped
  before output creation. The post-freeze episode identifier had been attached to the lead ledger
  but omitted from the paired table's evaluation-only target whitelist, so concentration reporting
  could not find it. The correction carries only that hindsight diagnostic identifier into the
  paired output; it remains absent from all forecast features. All outputs are regenerated again.
- The following command at SHA `891c7e7ce873fa8002242e807994519903685676` reached the
  registered sensitivity rebuild and stopped before output creation. The frozen decision ledger
  calls its session `score_session`, while V2's rebuilt raw research surface retains
  `session_date`; the opportunity-status adapter required an explicit causal alias. The correction
  renames that already-known session field only for missing/unfilled target classification.
- The run at SHA `0776ed0969be5105b46e8aef9485e7b7a0375ad7` completed every model and
  stress rebuild, then failed while rendering the Markdown report because pandas' optional
  `tabulate` package is not installed. Its incomplete, unmanifested output was quarantined under
  `/tmp` and is not an experiment result. A local deterministic Markdown renderer removes the
  optional dependency; every scored comparator is rerun rather than copying the partial tables.
- Primary and exact scoring completed at SHA `858517e84a8cecf3c46ca5557821887ef804c4d0`.
  The auditor was then corrected to exclude the exact report's in-directory copy from the generic
  artifact set; reports remain subject to their separate required hash comparison. This audit-only
  normalization does not alter scored artifacts, and the audit records its own commit and file hash.
- Visual QA found that the first chronological structurally-led classification was also a negative
  false lead, causing both representative plot selectors to choose it. The positive example is now
  constrained to positive mean episode payoff and the false example to non-positive payoff. Only
  plots, manifests, and their audit hashes are regenerated; scored tables are unchanged.

## Scientific decision

Pending frozen scoring.
