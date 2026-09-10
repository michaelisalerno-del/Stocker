# Session HARD universe acquisition and candidate selection

## Current: new PAPER runs

Current method version: `SESSION_HARD_CAUSAL_Q1_ACQUISITION_V9`.
Stable method identity: `SESSION_HARD_HV_HIGH_PRE_MOVE_DOWN_STRUCTURE_D`.
Candidate recipe: `SESSION_HARD_RANGE5_250_RV10_50_RV15_30_V1`.

```
Saved broad eligible market membership
  -> experimental IBKR scanner acquisition union (conId, append-only)
  -> exact opening one-minute bars for that union
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
V9 supplies `ScannerAcquisition` through that existing interface. Broad membership is saved
unchanged; only acquired contracts are qualified before Range5. Full-population qualification
and opening-history work move to the delayed oracle. Acquisition always carries
`UNVALIDATED_UPSTREAM_ACQUISITION`. V8 retains its original direct broad-source behavior.

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
The historical economic replay used the older pooled/TOP5 engine; production V8/V9 retain
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

A successful incomplete prefix retains the identity with a missing score/reason. V9 marks
the PAPER stage incomplete if an acquired stock lacks its exact prefix; the delayed oracle
and historical V8 retain missing-last ranking. No bar is interpolated or substituted. Data-request failures are distinct from zero
movement and degrade the run. A disconnected or unentitled broker, insufficient throughput,
or broad acquisition that does not finish by the stage deadline reports
`BROAD_OPENING_DATA_CAPACITY_UNRESOLVED`. No subscription budget or entitlement is increased.

## Data cost and state boundaries

Pre-deadline work is limited to scanner acquisition, acquired-identity classification and
the acquired population’s opening minute prefixes. All current
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

Existing V8 `SESSION_HARD_CAUSAL_Q1_CANDIDATES_V8` and V7 `SESSION_HARD_CAUSAL_Q1_DISCOVERY_V7` runs retain their exact specifications,
hashes and original acquisition/selection behavior. They remain runnable if already saved.
V6 and the older activity versions remain archived/readable according to existing policy.
`ACTIVITY_LIQUIDITY_V2`, `ACTIVITY_SHORTLIST_V1`, `ACTIVITY_CAPACITY_V3_SCREEN50`,
`ACTIVITY_CAPACITY_V2_WARNINGS`, scanner enums, audits and snapshots are retained for compatibility.
They are not the candidate-selection rule for new Session HARD runs.

V7's five TOP_TRADE_RATE cap scans, native FX cap conversion, metadata limits and 150-name
monitoring budget describe only its original behavior. Historical scanner ranks are never
relabelled as Range/RV evidence. Its explicit rebuild action remains confined to that old path.

`scripts/migrate_candidate_selection.py --runs-config OLD --output NEW` writes a separate
configuration, archives previous PAPER runs, creates disabled V9 replacements and preserves
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

## Prospective scanner-assisted acquisition (V9)

`SESSION_HARD_IBKR_ACQUISITION_EXPERIMENT_V1` is an experiment, not a validated scanner
filter. It declares three sweeps at OPEN+60/+180/+240 active seconds, two concurrent scanner
requests per sweep, and up to 50 rows per component. The broker's shared ten-scanner ceiling
and message throttle still apply. No acquisition-pool cap is enabled; there is no expected
measured union size yet. A 35-component matrix would make 105 requests over three sweeps,
with at most 5,250 raw observations before duplicates, eligibility and actual Gateway limits.

The matrix requests TOP_TRADE_RATE, TOP_VOLUME_RATE and HOT_BY_VOLUME only when advertised.
Opening percentage gain/loss families resolve an unambiguous exact advertised code from
Gateway code/description text. These family identifiers are not invented IBKR scan codes.
Capabilities include raw XML, retrieval time, available server version, locations, instruments,
codes and filter fields, cached per connection and persisted by content hash. Unsupported or
ambiguous components are individually failed; no scan is substituted. Actual connected
capabilities have not been measured by this implementation task.

Each family has UNCAPPED, BELOW_MICRO (<$50m), MICRO, SMALL, MID, LARGE and MEGA requests,
using existing canonical cap definitions. UNCAPPED has no cap floor or ceiling and retains
access to unknown-cap stocks. Cap slices allocate scanner coverage; they are not stock
admission rules. Non-USD cap conversion uses the existing broker FX path, otherwise affected
components fail. There are no acquisition price/volume thresholds. Exact filters and every
raw hit/rank are saved before qualification, including duplicate hits.

Contracts must match saved broad membership and resolve to the same broker conId. Scanner
rank, first-seen sweep, component, filters and observation time remain distinct from Range/RV
and strategy scores. The union accumulates across all sweeps and is sealed before OPEN+5.
Unsupported/failed/late components produce PARTIAL/FAILED; the default recipe does not admit
partial components. A changed matrix or optional experimental pool cap requires a new recipe
ID and evaluation period. The cap policy, if explicitly configured, is best scanner rank,
first observation, then conId, with ACQUISITION_POOL_CAP_APPLIED recorded.

The selector requests exact one-minute TRADES/RTH prefixes only for acquired conIds, then
only Range250 survivors and RV50 survivors. Cache/in-flight sharing uses the existing opening
source and IBKR-only history cache. It does not allocate broad tick-by-tick streams. The
inherited V8 transport window is explicit: the +5 prefix becomes final at +5 and processing
must finish before +10; +10 finishes before +15; +15 before the first HARD checkpoint.
No bar after the feature cutoff enters the score. Wall-clock completion times and delays
are recorded separately; this is not a claim that historical responses arrive instantly at +5.
Transport failures/deadline expiry degrade the session with no fallback or retrospective repair.
The oracle remains available even when the PAPER candidate chain fails.

## Delayed full-market oracle and recall

The method-supplied background hook runs after regular-session close, one saved broad
reference per step on the existing bounded history ingress. It pauses while any enabled market
is active or approaching its open, or foreground history/checkpoints are in progress. It does
not hold scheduler locks while requesting data. Cancellation propagates to the exact IBKR
historical request; pacing/entitlement/timeout failures cannot masquerade as flat stocks.
Failed audit requests mark the audit INCOMPLETE; the opt-in benchmark can explicitly resume
only those requests. Successful absent/incomplete prefixes retain missing-last semantics.

The audit obtains the full first-15-minute prefix per eligible identity, reuses frozen
candidate_value/rank_candidates, and reconstructs the full Range250 -> RV50 -> RV30 chain.
It stores only acquisition/oracle audit tables, never candidate, cohort, order or trade tables.
Full-market oracle results are hindsight AUDIT_ONLY and cannot repair the same day's lists.

Recall denominators contain available (nonmissing) oracle target identities; missing selected
targets are reported separately. Metrics include each stage's captured/available ratio, exact
days, mean, median, worst day, missed identities with target rank, Range250 rank buckets
1–25/26–50/51–100/101–150/151–200/201–250, and component captured/unique contribution counts.
Five predeclared shadow masks reuse raw hits: activity-only, opening-only, uncapped-only,
partitioned-only and hybrid. They do not request extra scans or feed strategy state.
No scanner recipe is optimized using P&L. Oracle stateful opportunity/trade replay (Target D)
is not run by the acquisition audit and needs a separately requested study.

US acquisition evidence is PROSPECTIVE_IBKR_TEST, separately from the candidate chain's
VALIDATED_EXISTING_RESEARCH. Other markets use
UNVALIDATED_CROSS_MARKET_ACQUISITION_TRANSFER and retain unvalidated candidate transfer labels.
Several prospective sessions and adequate recall/latency evidence are required before freezing
any acquisition recipe. Changes informed by observed misses begin a new version/evaluation.

Optional recipe field `transport_parity` requests the first five-minute IBKR TRADES/RTH bar
during the delayed audit, compares it with the exact five one-minute bars, and records score
differences, ordering and TOP250 parity. It never changes the production one-minute transport.
A representative cached/Gateway sample has not yet been observed.

## Acquisition diagnostics and opt-in benchmark

The additive acquisition_* tables retain sessions/recipes, capability XML, broad references,
scanner components/timings, raw hits, normalized union, history request diagnostics and oracle
ranks. Normal run summaries use SQL counts and SQL JSON projections, without loading broad
identities or detailed oracle targets. The dashboard shows broad membership, raw/unique acquired
stocks, five-minute prefixes, stage counts, separate evidence and oracle progress/recall.

`/api/runs/{run_id}/acquisition?session=YYYY-MM-DD&kind=components&limit=50&offset=0`
supports hits, components, pool, broad, requests, oracle, targets, misses and contributions.
On-demand benchmark diagnostics report scanner/history latency p50/p90/p95/p99, actual/shared
requests, rows per sweep, prefix/cache counts, failure text, pacing/entitlement counts,
availability delay and Range/RV calculation timestamps. The broad request-count baseline
makes the avoided full-membership opening workload visible; throughput is measured, not assumed.

Run `scripts/migrate_candidate_selection.py --runs-config OLD --output NEW` to prepare
disabled V9 PAPER configurations; old rows/specifications/history remain intact. No migration,
activation or deployment occurs automatically.

On a dedicated PAPER Gateway (not alongside an external trading process), opt in with:

```sh
rtk .venv/bin/python scripts/benchmark_scanner_acquisition.py   --ibkr-config PAPER_CONNECTION.yaml --runs-config NEW.yaml --run-id SAVED_V9_RUN   --client-id UNUSED_CLIENT_ID --state benchmark.sqlite --history-cache ibkr-history.sqlite   --confirm-dedicated-paper-gateway --oracle
```

Start before the first sweep. Use `--capabilities-only` to persist/inspect actual scanner
parameters without requesting opening history. Use `--recipe PREDECLARED.json` for an explicitly
versioned experiment; changing V1 under its original ID is rejected. The benchmark uses a
separate run ID/database, never constructs execution services, and keeps broker execution
disabled. `--audit-only --oracle --session YYYY-MM-DD --resume-audit` resumes an incomplete
saved audit. The audit waits until that session's close. Normal pytest uses fakes and never
needs a Gateway. No actual scanner count, recall or completion-time claim is made until this
command or the PAPER runtime observes real sessions.
