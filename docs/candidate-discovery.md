# Session HARD universe acquisition and candidate selection

## Current: new PAPER runs

Current method version: `SESSION_HARD_CAUSAL_Q1_CANDIDATES_V8`.
Stable method identity: `SESSION_HARD_HV_HIGH_PRE_MOVE_DOWN_STRUCTURE_D`.
Candidate recipe: `SESSION_HARD_RANGE5_250_RV10_50_RV15_30_V1`.

```
Saved broad eligible market universe
  -> first 5 completed active regular-session minutes: RANGE_PCT HIGH TOP250
  -> first 10 completed active regular-session minutes: RV HIGH TOP50
  -> first 15 completed active regular-session minutes: RV HIGH TOP30
  -> existing IBKR history/PRE -> Session HARD qualification, cohort/MID, MODEL_T0/Q1
  -> existing causal entry, exits, account/risk/execution
```

Acquisition determines the available population; candidate selection reduces that population.
No scanner rank, activity rank, liquidity score, cap balance or cap quota enters these features.
There is no activity TOP50 or cap-scanner monitoring limit ahead of Range250 on new runs.
A missing broad source or failed opening-data acquisition never invokes legacy selection.

US runs use the existing named authoritative Nasdaq Trader membership (`US_ALL`, `NASDAQ`,
`NYSE`) from configured listing snapshots. The run embeds the exact source references;
refreshing the catalogue does not mutate it. Contract resolution and the existing IBKR
COMMON/CORP/ADR/REIT classification check reject unsupported securities, ETFs, warrants,
unknown classifications and wrong currencies before opening calculations. Rejections are audited.

Other markets use an explicitly configured broad market universe, with the selected market's
market specification. Without that source, the run reports `BROAD_UNIVERSE_UNAVAILABLE`.
An arbitrary CUSTOM basket is not silently promoted to a market universe. FIXED/RESEARCH
snapshots remain explicit testing inputs. `UniverseProvider` supplies normalized identities;
a future validated discovery provider can replace acquisition without changing the selector.
Scanners remain available for legacy runs and diagnostics. No new scanner recipe was created.
A future scanner-sourced population must carry `UNVALIDATED_UPSTREAM_ACQUISITION` until its
recall is independently established. This implementation does not enable that source for V8.

## Frozen recipe and evidence

The method specification includes recipe/version, source policy, exchange calendar/timezone,
formulas, offsets, capacities, HIGH orientation, missing-last handling, ascending SHA256(symbol)
ties and all three frozen research recipe hashes. The canonical run-specification hash changes;
trading specification fields and model artifacts do not.

Range5 = `(max(H0..H4) - min(L0..L4)) / O0`.
RV10/RV15 = `sqrt(log(C0/O0)^2 + sum(log(Cj/Cj-1)^2))` over the exact prefix.
There is no annualisation, volume/cap weighting, relative normalization or blended score.
NumPy float64 log and pandas compensated group sum preserve frozen research numerical parity.
Separate score names are `initial_range_5m_score`, `preselector_rv_10m_score` and
`watchlist_rv_15m_score`; Session HARD and whipsaw-risk scores remain downstream.

US candidate evidence is `VALIDATED_EXISTING_RESEARCH`, specifically the already exposed
533-stock US development/validation population. This is not proof of current entire-market
coverage, profitability, causal Q1 portfolio outcomes, or untouched validation.
Other markets are `UNVALIDATED_TRANSFER` with display status
`UNVALIDATED_CROSS_MARKET_PAPER_TRANSFER`. The identical mechanical recipe is used without refits.

The evidence comes from `session_hard_initial_rv50_recall_v0`, with recipe SHA256
`76b6202ce8ad5c6c57b44fe0fefe8611fda39d0bba48c8810690210878cc9bfa`, upstream RV10 SHA256
`39f730eac22d5bc9a380a9dbf028c9c374716b1c2c51d1f1e1f80a482fc91aab`, and RV15 SHA256
`aca8441a0190ca8d66d7668ad152e06c05a15d1859b696d089a67747d8eb0364`.
The historical economic replay used the older pooled/TOP5 engine; production V8 retains
causal Q1. Calculation/list parity must not be described as economic parity between engines.

## Timing, finality and lifecycle

`ExchangeSessionResolver` is the calendar authority. `MarketSession.minute_prefix` expands
its active five-minute slots into exact one-minute timestamps, excluding breaks and respecting
holidays, special opens, shortened sessions and DST. The stage completion time is the last
required active-minute start plus one minute. No machine/browser timezone participates.
The same implementation serves all 14 existing market selections, including US, LSE and ASX.

MethodServices supplies the candidate lifecycle, readiness and summary callbacks. The shared
runtime invokes these hooks and gates method history preparation/checkpoints on readiness.
No strategy-name branch, second strategy engine or second broker adapter is introduced.
Acquisition and opening-bar jobs run as bounded background work, outside the scheduler lock.
They use the existing IBKR history ingress (`1 min`, `TRADES`, RTH, exact end time), never
broad tick-by-tick entry streams. Same-service concurrent opening requests share completed data.
Actual processing timestamps are saved separately from intended stage timestamps.

A run must initialize before its first stage. Each due stage runs on the first available live
scheduler cycle, using only that stage's completed prefix. Work must finish before the next
stage boundary (final reduction before the first method checkpoint). A late start, restart
without a persisted stage whose deadline has passed, or missed cycle spanning the next stage
reports `CANDIDATE_SELECTION_WINDOW_MISSED`; it never rebuilds a hypothetical earlier list.

State is DISCOVERY -> BROAD_ELIGIBLE -> RANGE5_SELECTED -> RV10_SELECTED -> RV15_SELECTED
-> SESSION_HARD_ACTIVE. Every stage is an atomic immutable snapshot. Only survivors feed the
next stage, including missing-last candidates when the population falls below capacity.
TOP capacities are maxima, never promises to fabricate identities. No discarded stock returns,
and no Q1/qualification/no-trigger/no-entry failure replenishes the final watchlist.

A successful incomplete prefix retains the identity with a missing score/reason. No bar is
interpolated, carried forward or substituted. Data-request failures are distinct from zero
movement and degrade the run. A disconnected or unentitled broker, insufficient throughput,
or broad acquisition that does not finish by the stage deadline reports
`BROAD_OPENING_DATA_CAPACITY_UNRESOLVED`. No subscription budget or entitlement is increased.

## Data cost and state boundaries

Broad work is limited to identity/classification and the opening minute prefixes. All current
method-specific prior-close history, HV/M and PRE preparation starts after final TOP30.
A selected stock still has its complete legitimate stock-local IBKR cache across previous
sessions, including days when it was not selected. Nothing truncates or deletes that history.
Only TOP30 enter Stage 5 requests and the existing run-scoped strategy/cohort/Q1/event state.
There is no posthoc filtering of trades. Prior strategy events remain attached to their runs.

No opening-selector subscriptions are allocated, so discarded names require no tick-feed
unsubscribe. Further minute requests cover only 250 then 50 survivors. Existing ordered TRADES
entry feeds begin at the normal method prefetch window for the final population, with existing
shared line budgets and release logic unchanged. Broad first-five-minute coverage of thousands
of equities is **not proven feasible**; Gateway throughput, history permissions, latency and
normalization still require read-only opening-session tests.

## Persistence and API

The existing runtime SQLite database adds:

- `opening_candidate_sessions`: small run/session/market/spec summary and recipe/source metadata.
- `opening_candidate_sources`: exact unresolved source references.
- `opening_candidate_population`: normalized conId identities.
- `opening_candidate_rejections`: acquisition failures/rejections.
- `opening_candidate_stages`: contextual score name, value, rank, selected flag, reason,
  intended/actual timestamps and exact input bars per stock and stage.

Population/stage writes are batched; stages commit in one transaction. Restart reads the saved
lists without reranking, including the final TOP30. Summary reads aggregate counts in SQL and
never load the large source population or bar payloads. The run-summary/detail APIs expose
`candidate_selection`; `/api/runs/{run_id}/candidate-selection?session=YYYY-MM-DD&limit=50&offset=0`
serves on-demand paginated details. Dashboard progress shows broad, 250, 50, 30, qualification,
evidence and explicit capacity/readiness failures. Recipe values are not user controls.

## Legacy compatibility and migration

Existing V7 `SESSION_HARD_CAUSAL_Q1_DISCOVERY_V7` runs retain their exact specifications,
hashes and five-cap-scan acquisition/selection behavior. They remain runnable if already saved.
V6 and the older activity versions remain archived/readable according to existing policy.
`ACTIVITY_LIQUIDITY_V2`, `ACTIVITY_SHORTLIST_V1`, `ACTIVITY_CAPACITY_V3_SCREEN50`,
`ACTIVITY_CAPACITY_V2_WARNINGS`, scanner enums, audits and snapshots are retained for compatibility.
They are not the candidate-selection rule for new Session HARD runs.

V7's five TOP_TRADE_RATE cap scans, native FX cap conversion, metadata limits and 150-name
monitoring budget describe only its original behavior. Historical scanner ranks are never
relabelled as Range/RV evidence. Its explicit rebuild action remains confined to that old path.

`scripts/migrate_candidate_selection.py --runs-config OLD --output NEW` writes a separate
configuration, archives previous PAPER runs, creates disabled V8 replacements and preserves
risk settings and historical configurations. It never activates, deploys, submits an order,
changes broker routing or deletes database records. Review missing broad populations before
activation. Back up production state through the established procedure before any deployment.

## Verification

Saved fixtures cover 2025-06-02, 2025-07-03 (half-day) and 2025-07-17 (BURU/ACHR substitution).
The exporter authenticates original bar files by SHA256 and copies original reference scores
and watchlists; it never generates expected values using production code. Tests assert exact
float values, missingness, complete ranking and 250/50/30 identities. Fixtures are research-only;
they are never a production/PAPER history source.

Lifecycle tests cover stage narrowing, restart, missed windows, failures and separate markets.
The real runtime integration test checks that only final30 enter history preparation, feed
preparation, signals and cohort labels. Existing Q1, entry, risk and PAPER/LIVE tests still apply.
