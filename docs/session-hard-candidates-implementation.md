# Session HARD candidate pipeline implementation

Implemented locally on branch `implement/session-hard-candidates-20260910`, based on
`88fcaf2670e934687fbc96da1cc23e0c558da2fc`. No deployment, migration of active files, run
activation, broker connection, or order submission was performed.

## Current method and recipe

Stable method identity remains `SESSION_HARD_HV_HIGH_PRE_MOVE_DOWN_STRUCTURE_D`.
The new selectable PAPER version is `SESSION_HARD_CAUSAL_Q1_CANDIDATES_V8`.
Its immutable specification contains
`SESSION_HARD_RANGE5_250_RV10_50_RV15_30_V1`, all three stages, offsets, capacities,
orientations, formulas, missing/tie policies, research hashes and market evidence.

New Session HARD candidate selection is RANGE5 HIGH TOP250 → RV10 HIGH TOP50 → RV15 HIGH TOP30.

For exact completed active-minute prefixes, with first open O0 and minute closes Cj:

- Stage 1: `(max(H0..H4) - min(L0..L4)) / O0`, HIGH, TOP250.
- Stage 2: `sqrt(log(C0/O0)^2 + sum(log(Cj/Cj-1)^2, j=1..9))`, HIGH, TOP50.
- Stage 3: `sqrt(log(C0/O0)^2 + sum(log(Cj/Cj-1)^2, j=1..14))`, HIGH, TOP30.

Missing/unusable prefixes rank last. SHA256(symbol) breaks ties. There is no scanner,
cap, volume, liquidity or HARD-score blend, quota, replacement or later replenishment.
NumPy float64 log and pandas compensated summation reproduce the frozen numerical results.

## Architecture and timing

Reused MethodDefinition, immutable method specifications, explicit universe builders,
MethodServices, the shared runtime scheduler, canonical exchange calendars, the existing
IBKR history ingress/cache, Stage 5 requests and the existing Session HARD strategy engine.
Added a small shared candidate-math module, provider interface, durable candidate service,
and generic method lifecycle/readiness hooks. No second engine, broker adapter or plugin framework.

MarketSession expands existing active five-minute slots into exact one-minute prefixes.
The boundary is the final expected minute start plus one minute. Timezone-aware UTC
timestamps come from the existing market calendar, including local DST, holidays,
shortened sessions and breaks. No machine/browser timezone or literal US stage clock is used.
Stages run on the first runtime poll after their boundary with explicit fixed end timestamps;
transport completion must precede the next stage, and the final stage must finish before
the first existing HARD checkpoint. Actual persistence time and decision boundary are audited.

The existing registry has 14 market profiles: US_ALL, US_NASDAQ, US_NYSE, CANADA_TSX,
UK_LSE, GERMANY_XETRA, FRANCE_PARIS, NETHERLANDS_AMSTERDAM, SWITZERLAND_SIX,
AUSTRALIA_ASX, HONG_KONG_HKEX, JAPAN_TSE, SOUTH_KOREA_KRX and SOUTH_AFRICA_JSE.
These are existing calendar/profile capabilities, not a claim that broad live coverage is available.

## Broad acquisition and operational limitations

US uses existing authoritative Nasdaq Trader named membership. Other markets use an
explicit matching cached ALL-cap market population. The entire saved input is qualified;
legacy activity universes and old scanner TOP50 are not inserted ahead of Range250.
Source references, eligible broker identities and rejection outcomes are persisted separately.
Identity retains conId, symbol, primary/listing exchange, routing exchange, currency,
market and security type; existing broker stock classification excludes unsupported instruments.

A missing population gives `BROAD_UNIVERSE_UNAVAILABLE`. Infrastructure failures or stage
capacity/deadline failures give `BROAD_OPENING_DATA_CAPACITY_UNRESOLVED`; they cannot
silently produce a smaller actionable population. Successful empty/unusable individual
history responses remain missing-last. No validated production scanner recipe was invented.
Scanner/universe acquisition remains separate from the validated candidate-selection chain.

Broad first-five-minute coverage is **not established as feasible** for thousands of stocks
on the current Gateway/account. Opening data uses bounded existing one-minute RTH TRADES
historical requests with exact end times, without broad tick-by-tick feeds. Calls are shared
for equivalent concurrent requests. Only survivors receive the next stage's requests;
no candidate subscriptions require cancellation. Existing broker concurrency, history cache,
tick budget, account entitlements and subscription ownership remain unchanged.

Only identity/classification and opening-prefix bars remain broad. Expensive prior-close/HV/PRE
preparation is gated until final TOP30, then uses the unchanged IBKR stock-local cache and
completeness requirements, including earlier days when that stock was not selected.
Entry-feed preparation and real strategy state receive only TOP30. No completed-trade filter
implements the restriction.

Gateway testing remains necessary for listing qualification throughput, opening-history
pacing/entitlements/latency, final-bar availability, concurrent markets and shared resource
behavior. Upstream scanner recall and sufficiently broad acquisition remain separate work.

## Persistence, recovery, API and evidence

Additive SQLite tables store run/session metadata, exact source population, normalized
identities, eligibility failures, all ranked stage rows, explicit score names, selection flags,
missing reasons, boundary/selection timestamps and original input bars. Population and stage
reliability are committed atomically with their rows. Failed snapshots cannot activate after
a crash. Persisted successful lists are reused; missing past decision windows give
`CANDIDATE_SELECTION_WINDOW_MISSED`. Interrupted acquisition is explicit. No historical
watchlist is fabricated from later data and no old database records are deleted.

Run summary/detail APIs expose SQL aggregate stage counts. The normal refresh does not
load the large population. A paginated
`/api/runs/{run_id}/candidate-selection?session=YYYY-MM-DD&limit=...&offset=...`
endpoint exposes detailed lineage. The generic dashboard shows progress, HARD-qualified
count, evidence and readiness limitations; capacities are not user controls.

US: `VALIDATED_EXISTING_RESEARCH`, explicitly scoped to the US development population.
Other markets: `UNVALIDATED_TRANSFER` and
`UNVALIDATED_CROSS_MARKET_PAPER_TRANSFER`.
Other markets use the same mechanical recipe only as an UNVALIDATED_TRANSFER until
independently tested. No cross-market profitability or Q1 portfolio validation is claimed.

## Legacy behavior and migration

Legacy scanner/activity profiles are retained only for historical compatibility and are not
the candidate-selection rule for new Session HARD runs.

Saved V7 runs remain runnable with original specifications and original discovery behavior.
Earlier archived versions remain readable. Every V7 specification/hash was checked for exact
equality against the untouched baseline across all 14 profiles. New V8 hashes include the recipe.

`scripts/migrate_candidate_selection.py --runs-config OLD --output NEW` creates a separate,
exclusive output file, archives prior PAPER configurations and creates disabled V8 replacements
with preserved risk settings. Existing V8 collisions, including archived configurations,
are rejected. Historical signals/orders/trades and active configuration files are untouched.
The prior activity migration remains pinned to V7 for reproduction. No migration was applied.
Deployment requires the existing backup and run-activation procedure; LIVE remains unavailable.

## Validation and frozen-rule audit

Final full suite: **1,005 passed, 14 opt-in integration tests skipped**, 10 existing dependency/
research warnings. Both mocked browser suites pass, including candidate progress, transfer
evidence, capacity failures, compact refreshes and PAPER-only run creation. JavaScript syntax passes.
Changed-file Ruff passes. Strict mypy passes for 47 production/migration source files.
Repository-wide Ruff was run: 47 pre-existing findings remain in two untouched files
(`behavioral_state_similarity.py` and its test); no unrelated cleanup was made.

The three historical fixtures copy original reference scores and candidate lists; the exporter
verifies raw-source hashes. Production never generates its own expected values.

| US session | Broad scored population | Range identities | RV10 identities | RV15 identities |
|---|---:|---:|---:|---:|
| 2025-06-02 | 532 | 250/250 exact | 50/50 exact | 30/30 exact |
| 2025-07-03, half-day | 531 | 250/250 exact | 50/50 exact | 30/30 exact |
| 2025-07-17 | 531 | 250/250 exact | 50/50 exact | 30/30 exact |

All **4,782 feature-value/missingness comparisons**, **2,494 ranking positions** and
**990 selected list positions** match exactly. Fixtures include low-priced stocks, missing
prefixes and six cap groups. July 17 excludes BURU and includes ACHR throughout the chain.
Synthetic tests additionally cover deterministic ties, flat/invalid prices and nonfinite inputs.
This proves mathematical/list parity on representative sessions, not a new all-83-day
economic replay. The older research engine differs from the preserved production Q1 engine.

The real-runtime state test starts with 533 identities, narrows to 250/50/30 and verifies only
the final30 receive history preparation, entry preparation, real method signals and cohort labels.
Recovery, no re-entry/reselection, cross-market isolation, transport failures and migration
collisions pass. Calendar tests cover all 14 profiles, explicit US/LSE/ASX DST, US/UK transition
mismatch weeks, holidays, shortened days and active-minute break semantics.

Frozen artifact files are unchanged. MODEL_T0 SHA256 remains
`68c0f5ebf4a23744a171e32a32a8b336e8d832013692913ca034f744a88e8bc2`;
prospective Q1 spec SHA256 remains
`de2c4b3b90e9ecfff44cc7da971b2d9700eb277ece423bab6a4ed8e1fb3b6965`.
The trading specification differs only in operational version, candidate acquisition and capacity
lineage. Model B 0.999361477, PRE_MOVE >0.475764059845861, inclusive Q1 <=0.22995371253992852,
checkpoints 6,8,...,34, cohort/MID, ±0.20M arm, first-break direction, entry window, 0.50M stop,
1.00M target, T0+15 deadline, tick rounding, risk, shortability and broker fill handling are unchanged.

Session HARD trading logic itself was not retuned or changed.
Session HARD qualification, Q1, MID/cohort, entry, exits and execution rules were not retuned.
US candidate recipe remains exactly RANGE5 HIGH TOP250 → RV10 HIGH TOP50 → RV15 HIGH TOP30.

Standards review: all reported findings resolved; no outstanding documented-standard violation.
Spec review: all reported failure/recovery/migration findings resolved.

## Files changed

- `AGENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/candidate-discovery.md`
- `docs/session-hard-candidates-implementation.md`
- `docs/universes.md`
- `packages/stocker_core/src/stocker_core/candidate_selection.py`
- `packages/stocker_core/src/stocker_core/discovery.py`
- `packages/stocker_core/src/stocker_core/methods.py`
- `packages/stocker_core/src/stocker_core/runs.py`
- `packages/stocker_dashboard/src/stocker_dashboard/app.py`
- `packages/stocker_dashboard/src/stocker_dashboard/controls.py`
- `packages/stocker_dashboard/src/stocker_dashboard/read_service.py`
- `packages/stocker_dashboard/src/stocker_dashboard/static/dashboard.js`
- `packages/stocker_dashboard/src/stocker_dashboard/universe_runs.py`
- `packages/stocker_execution/src/stocker_execution/candidate_pipeline.py`
- `packages/stocker_execution/src/stocker_execution/ibkr.py`
- `packages/stocker_execution/src/stocker_execution/runtime.py`
- `packages/stocker_execution/src/stocker_execution/session_hard_method.py`
- `packages/stocker_execution/src/stocker_execution/strategy_factory.py`
- `scripts/export_candidate_parity_fixture.py`
- `scripts/migrate_activity_filter.py`
- `scripts/migrate_candidate_selection.py`
- `tests/dashboard_run_summary.cjs`
- `tests/fixtures/session_hard_candidates/expected_scores.parquet`
- `tests/fixtures/session_hard_candidates/expected_watchlists.json`
- `tests/fixtures/session_hard_candidates/legacy_spec_hashes.json`
- `tests/fixtures/session_hard_candidates/manifest.json`
- `tests/fixtures/session_hard_candidates/opening_bars.parquet`
- `tests/legacy_discovery_support.py`
- `tests/test_activity_filter_migration.py`
- `tests/test_candidate_discovery.py`
- `tests/test_candidate_discovery_gateway.py`
- `tests/test_candidate_migration_dashboard.py`
- `tests/test_candidate_pipeline.py`
- `tests/test_discovery_global.py`
- `tests/test_frozen_candidate_selection.py`
- `tests/test_session_hard_universe.py`
- `tests/test_stage10_dashboard.py`
- `tests/test_stage10_extension_builder.py`
