# Scanner-assisted acquisition implementation report

Implemented locally against completed V8 baseline `9181eff`. Current selectable version:
`SESSION_HARD_CAUSAL_Q1_ACQUISITION_V9`. Experimental acquisition recipe:
`SESSION_HARD_IBKR_ACQUISITION_EXPERIMENT_V1`. Stable Session HARD identity and frozen
candidate recipe `SESSION_HARD_RANGE5_250_RV10_50_RV15_30_V1` remain intact.

## Acquisition and operational evidence

| Item | Implemented / observed status |
|---|---|
| Actual connected Gateway capability codes | Not observed. No Gateway connection was made. Cached/persisted reqScannerParameters response is authoritative when the opt-in command runs. |
| Components | TOP_TRADE_RATE, TOP_VOLUME_RATE, HOT_BY_VOLUME, plus opening percentage gain/loss families resolved only to unambiguous exact advertised codes. Unsupported components fail explicitly. |
| Coverage | Each family has UNCAPPED, BELOW_MICRO (<$50m), MICRO, SMALL, MID, LARGE and MEGA; canonical cap bounds, existing FX conversion, no price/volume floor. |
| Schedule | OPEN+60/+180/+240 active seconds through canonical market-session slots; never machine-local time. |
| Scanner concurrency | Two per sweep; existing shared broker maximum ten and message throttle preserved. |
| Requests | Fully supported default matrix: 35 components × 3 sweeps = 105. Shared matching sweeps reuse results. |
| Union | Append-only qualified conIds matching saved broad membership; no default cap and no scanner-rank trade admission. Explicit experimental cap records its policy/status. |
| Expected population | No numerical coverage guarantee; the experiment aims for a manageable union of hundreds. Actual union size is unmeasured. |
| OPEN+5 request count | One exact opening-prefix request per acquired conId, minus cache/in-flight hits. Never automatically all 6,570 references. Actual Gateway count is unmeasured. |
| Throughput / completion | No Gateway p50/p90/p95/p99 or Range250 completion timestamp observed. Instrumentation persists those measurements. |
| Range250 before RV10 | Required by the unchanged inherited transport deadline, but operational feasibility remains unproven. |
| Prospective recalls / unique component contributions | No real-session results yet. Metrics and paginated missed-identity/contribution analysis are implemented. |

Scanner-assisted acquisition is not a trading filter. It is a high-recall data-acquisition
mechanism upstream of the frozen Range250 → RV50 → RV30 chain.

Broad US membership still uses authoritative saved listing snapshots. Other markets require
their own saved broad population. Pre-deadline contract qualification/history is restricted
to scanner-acquired identities. Method-specific prior-close/HV/PRE preparation remains gated
until final TOP30, preserving stock-local historical cache regardless of prior selections.
No tick-by-tick feeds are allocated to the acquisition population.

The first-five-minute feature is still `(max(H0..H4)-min(L0..L4))/O0`. RV10 and RV15 remain
`sqrt(log(C0/O0)^2 + sum(log(Cj/Cj-1)^2))` over exact 10/15 active-minute prefixes.
HIGH orientation, capacities and SHA256(symbol) ties are unchanged. No cap/scanner/liquidity
score enters these formulas. There is no replenishment.

Feature cutoffs remain OPEN+5/+10/+15. The existing V8 transport window allows the exact +5
prefix to be retrieved once final and requires calculation before +10; +10 before +15; +15
before the first HARD checkpoint. Actual arrival/calculation timestamps are kept separately
from feature cutoffs. The experiment does not claim zero-latency bars exactly at +5.

## Delayed oracle

After session close, one saved broad reference per background step is qualified and its exact
opening 15-minute prefix obtained through the existing IBKR history/cache and bounded semaphore.
The audit pauses for active/upcoming enabled market windows and foreground preparation. SQL
work stays off the polling loop; optional audit errors cannot abort trading polling. Incomplete
requests are recorded and may be explicitly resumed. Historical timeout/cancellation is handled
by the existing broker adapter with request-scoped errors and cancellation.

Full-market Range250 → RV50 → RV30 is reconstructed with the same frozen candidate functions.
Oracle persistence is separate from candidate and strategy tables. No orders, cohort updates
or same-day candidate writes occur. The audit retains scores, ranks, missingness and denominators.
The optional five-minute transport experiment compares scores, ordering and TOP250 with the
exact one-minute calculation; production continues to use one-minute bars. No real transport
parity sample has been measured yet.

The full-market oracle is delayed audit-only information and can never affect the same day's
strategy decisions.

Metrics include available-target recall at all three stages, mean/median/worst day/exact days,
missed identities with oracle rank, six Range250 rank buckets and component unique contribution.
Late or failed-component raw observations remain audited but cannot inflate shadow recall.
Five predefined shadow masks reuse the same scanner results; only the explicit active recipe
feeds PAPER selection. No trade-P&L optimization or oracle stateful trade replay was performed.

## Persistence, API, dashboard and failure behavior

Additive acquisition tables save recipe/session metadata, Gateway capability XML, broad source
references, every scanner component/request/response/warning, raw hits, normalized union, history
request telemetry and oracle ranks/metrics. Existing candidate-stage tables remain authoritative
for PAPER decisions. Saved unions/stages are reused; missed/interrupted causal observations
never restart as hindsight selection. Recipe changes require a new version.

The normal dashboard uses SQL aggregate counts and compact JSON projections, without broad
identity or oracle-target loads. It shows broad membership, raw hits, acquired stocks, ready
prefixes, Range/RV stages and oracle progress/recall. On-demand paginated diagnostics:
`/api/runs/{run_id}/acquisition?session=YYYY-MM-DD&kind=components&limit=50&offset=0`.
Supported detail views include hits, components, pool, broad, requests, oracle, targets, misses
and contributions. Benchmark diagnostics expose request latency distributions, cache/request
counts, errors, availability delay, component/sweep rows and actual Range/RV calculation times.

Unsupported scans, late sweeps, failed qualification and incomplete acquisition are explicit.
The default recipe does not admit partially completed components. Missing exact live prefixes
or history transport/deadline failures degrade PAPER selection; no legacy shortlist, fabricated
score, same-day retrospective repair or stock resurrection is used. The delayed oracle can
still evaluate that day's infrastructure failure.

US candidate evidence remains VALIDATED_EXISTING_RESEARCH for its historical development
population. US acquisition is PROSPECTIVE_IBKR_TEST / UNVALIDATED_UPSTREAM_ACQUISITION.
Non-US acquisition is UNVALIDATED_CROSS_MARKET_ACQUISITION_TRANSFER; candidate transfer stays
UNVALIDATED_CROSS_MARKET_PAPER_TRANSFER. Running code is not cross-market profitability evidence.

## Validation

- Full suite: **1,025 passed, 14 opt-in Gateway tests skipped**, 10 existing warnings.
- New acquisition suite: **18 tests**, all pass. Final acquisition/broker regression run passes
  after removing redundant broker cancellation messages for already-completed requests.
- Real runtime test: **533 saved references → 350 scanner-acquired identities → 250 → 50 → 30**;
  only final30 receive history, entry preparation, real method signals and cohort state. No orders.
- Existing frozen US fixtures all pass: 2025-06-02, 2025-07-03 half-day, 2025-07-17 BURU/ACHR.
  Exact Range250, RV50 and RV30 list parity remains 250/250, 50/50, 30/30 on each session,
  including original feature/missingness/ranking checks.
- US/LSE/ASX acquisition timing/isolation tests pass; existing 14-profile calendar/DST/holiday/
  shortened-session/active-break tests pass.
- V7 and V8 historical specification hashes are unchanged across all 14 market profiles.
  V9 changes only acquisition/operational provenance; trading specification values match V8.
- Changed-file Ruff passes. Repository Ruff reports only **47 pre-existing findings** in the
  untouched behavioral_state_similarity research module and its test.
- Mypy passes for **52 production/benchmark source files**.
- JavaScript syntax and both mocked browser suites pass: compact run summaries/acquisition
  display and PAPER-only run creation. All browser API traffic is mocked.
- Read-only benchmark CLI help loads successfully; no Gateway integration run was performed.
- Final diff inspection finds no changes to frozen candidate mathematics, Session HARD engine,
  qualification/Q1 model artifacts or order/execution rules.

Standards review: two findings (blocking audit SQL and optional-error isolation) fixed and
verified; no remaining findings. Spec review: late-hit shadow attribution fixed and verified;
no remaining findings.

RANGE5 HIGH TOP250 → RV10 HIGH TOP50 → RV15 HIGH TOP30 remains unchanged.
Session HARD qualification, MODEL_T0, Q1, MID/cohort, entry, exits and execution rules were not retuned.
No broker orders were placed. Session HARD remains PAPER-only.

## Migration and remaining prospective work

No deployment, activation or active-configuration migration was performed. The existing migration
script creates disabled current V9 PAPER replacements in a separate output and retains legacy
specifications, risk settings and historical records. V8 direct-broad runs and V7 legacy scanner
runs keep their original specifications; they are not silently changed.

`scripts/benchmark_scanner_acquisition.py` is opt-in, requires a dedicated PAPER Gateway and
an explicit unused client ID, uses a separate benchmark run/database and disables execution.
It supports capability inspection, predeclared recipe JSON, opening sweeps/Range/RV and after-close
oracle/resume. Usage is in [candidate-discovery.md](candidate-discovery.md).

Several real opening sessions are still required to establish actual capability compatibility,
union size, final-bar availability, history pacing/entitlements, completion before subsequent
stage deadlines, oracle coverage and Range250/RV50/RV30 recall. Scanner-component redundancy
and pool caps must be judged from prospective evidence, with a new recipe version/evaluation
period after any change. The scanner acquisition recipe has not been declared validated or frozen.

## Files changed

- `docs/ARCHITECTURE.md`
- `docs/candidate-discovery.md`
- `docs/scanner-acquisition-implementation.md`
- `docs/session-hard-candidates-implementation.md`
- `packages/stocker_core/src/stocker_core/acquisition.py`
- `packages/stocker_core/src/stocker_core/discovery.py`
- `packages/stocker_core/src/stocker_core/methods.py`
- `packages/stocker_dashboard/src/stocker_dashboard/app.py`
- `packages/stocker_dashboard/src/stocker_dashboard/read_service.py`
- `packages/stocker_dashboard/src/stocker_dashboard/static/dashboard.js`
- `packages/stocker_dashboard/src/stocker_dashboard/universe_runs.py`
- `packages/stocker_execution/src/stocker_execution/acquired_candidates.py`
- `packages/stocker_execution/src/stocker_execution/acquisition_store.py`
- `packages/stocker_execution/src/stocker_execution/activity_shortlist.py`
- `packages/stocker_execution/src/stocker_execution/candidate_oracle.py`
- `packages/stocker_execution/src/stocker_execution/candidate_pipeline.py`
- `packages/stocker_execution/src/stocker_execution/ibkr.py`
- `packages/stocker_execution/src/stocker_execution/runtime.py`
- `packages/stocker_execution/src/stocker_execution/scanner_acquisition.py`
- `packages/stocker_execution/src/stocker_execution/strategy_factory.py`
- `scripts/benchmark_scanner_acquisition.py`
- `scripts/migrate_candidate_selection.py`
- `tests/dashboard_run_summary.cjs`
- `tests/fixtures/session_hard_candidates/v8_spec_hashes.json`
- `tests/test_candidate_migration_dashboard.py`
- `tests/test_candidate_pipeline.py`
- `tests/test_scanner_acquisition.py`
- `tests/test_stage2_ibkr.py`
