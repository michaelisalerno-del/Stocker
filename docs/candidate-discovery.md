# Session HARD candidate discovery

US / All + Session HARD defaults to `DYNAMIC_IBKR`. The normal UI remains Market → Method.
This is an operational watchlist, not a predictor of trading outcomes. No historical outcomes
were used to choose the profile. The trading method remains PAPER-only.

## Boundary and flow

The method catalogue owns a `DiscoveryProfile` for each supported discovery market. Future
methods can supply different profiles without changing Session HARD. Other existing market
selections retain their current activity discovery; this release does not claim global validation.

US major listed equities → five independent IBKR TOP_TRADE_RATE scans with server-side price
and liquidity checks → retain raw rows → interleave raw ranks across cap bands → merge by conId
while retaining every observation → cheap security/currency/identity checks → merged resource
limit → cached conId contract details and COMMON/CORP/ADR/REIT eligibility → monitoring resource
limit → qualified identities into `qualify_active_runs(candidate_identities=...)` → existing
IBKR history cache, exact-session PRE/MID preparation, frozen qualification and existing
ranking/capacity/execution.

No discovery code requests historical bars, quotes, tick streams or orders. Server-side price
and liquidity checks are recorded as `PASSED_BY_SCANNER`, not fabricated local measurements.
IBKR only returns surfaced contracts/ranks, so the audit cannot enumerate securities which the
server never returned, nor attribute their absence to a particular server filter. It does
reconstruct every returned observation and every subsequent local decision.

Scanner rank is original zero-based IBKR rank; discovery order is only a resource-admission
order. It is not passed to the strategy. Stage 5 retains its existing conId ordering and
memberships. Fixed/research members enter the same normalized Stage 5 requests after the
existing contract resolver. Frozen calculations, complete-session/history requirements,
entry timing, exits, risk, trade ranking and execution are unchanged.

## Profile and limits

Defaults live in `stocker_core.discovery.SESSION_HARD_DISCOVERY`, version 1:

| Setting | Default |
| --- | --- |
| Scanner location / instrument | Existing US market definition: STK.US.MAJOR / STK |
| Scanner | TOP_TRADE_RATE (activity, no direction rule) |
| Native stock filter | CORP, followed by verified contract classification |
| Price above | USD 1 |
| Today's volume above | 1,000 shares |
| Average volume above | 100,000 shares (IBKR avgVolumeAbove semantics) |
| Maximum results requested per band | 50 |
| Maximum merged identities receiving contract work | 250 |
| Maximum identities promoted into history/monitoring | 150 |
| Concurrent scanner requests | 2, within the existing shared broker semaphore |
| Contract metadata concurrency | 4, within the existing broker metadata semaphore |

Thresholds are broad tradability/resource defaults, not validated edge thresholds. Average
volume is broker-defined; no local replacement series is built. Spread filtering is omitted:
this stage does not spend market-data lines on transient quotes. Existing entry/data availability
checks still apply. The current shared 100-line budget permits five tick-by-tick feeds; a
150-stock preparation pool does not promise 150 complete entry streams.

Cap boundaries reuse **CAP_BUCKETS_V1** in `markets.py`, expressed in USD:

| Band | Canonical range |
| --- | --- |
| Micro | $50m–<$300m |
| Small | $300m–<$2bn |
| Mid | $2bn–<$10bn |
| Large | $10bn–<$200bn |
| Mega | >=$200bn |

The existing definition excludes below-$50m nano caps. STK.US.MAJOR is the existing major-listing
scope, not OTC or a guaranteed enumeration of every US equity. Each band gets its own scan
to prevent one cap group consuming all scanner slots. Raw rank is interleaved by canonical
band order for watch-capacity selection. Boundaries use native IBKR marketCapAbove/Below in
millions; broker boundary overlap is deduplicated by conId. Returned rows do not supply
independently verifiable market caps.

Operational settings are saved on each run as `discovery_profile`. Change resource limits or
the three tradability minima in the run YAML while disabled, then load/apply the configuration.
The run saves the complete effective profile and each discovery saves its configuration hash.
Profile identity, version, cap bands, scanner and stock eligibility remain method-owned.
For example, `monitoring_limit: 200` changes the resource budget without strategy-code changes;
it must not exceed `merged_candidate_limit`.

## Broker capability validation

Initialization validates location, instrument, scanner code, price/volume/average-volume and
native million-unit cap filters against cached `reqScannerParameters` metadata. Unsupported
requests fail with `SCANNER_NOT_SUPPORTED` and a diagnostic; there is no scanner substitution.
The existing IbkrConnection owns subscriptions, cancellation, request errors, timeout,
semaphores and connection caches. Every cap segment must complete; a partial failure produces
no watch pool. Empty successful bands are retained, and an entirely empty pool reports EMPTY.
No automatic retries or fallback to fixed listings occur.

See IBKR's [scanner documentation](https://www.interactivebrokers.com/docs/tws-api/doc/market-scanner/introduction)
and [subscription fields](https://www.interactivebrokers.com/docs/tws-api/protobuf/scanner-subscription).
The API caps responses at 50 and concurrent scans at 10. Capability metadata is evidence of
advertised support; an actual broker rejection still fails the attempt and is retained.

## Lifecycle and audit

Initialization creates an audited SCHEDULED attempt and validates capabilities. The first
available attempt at/after 15 active minutes scans once. Later calls/restarts reuse the same
run/session/configuration/generation record. The existing daily run lifecycle schedules the
next session automatically. No retrospective entry window or missing trade prefix is replayed.

To rebuild today: **Disable run → Rebuild on next enable → Enable run**. The control is serialized
with existing run controls. Disabling retains existing exposure management; refresh changes
only the next discovery generation. Failed and empty attempts are retained until an explicit
rebuild or the next session. Interrupted RUNNING attempts become FAILED when recovered.

SQLite in the existing runtime database stores:

- `candidate_discovery_runs`: UUID, parent run, session, configuration hash, generation and
  complete audit document.
- `candidate_discovery_refresh`: requested generation per parent run.

Documents retain method/version/source, effective settings, cap version, planned/native scanner
fields, start/capture/completion timestamps, each segment's status/error/count, all raw ranks
and returned metadata, canonical identities, provenance indices, stage decisions, rejection
reasons, discovery order and watch-pool admission. Writes occur at stage boundaries, never
per market-data tick. Records from previous generations are never replaced by a rebuild.

Local rejection reasons are INVALID_CONTRACT, WRONG_SECURITY_TYPE, MARKET_DATA_UNAVAILABLE,
RESOURCE_LIMIT and DUPLICATE_CONID (duplicate raw observations retain their canonical link).
Failures also report SCANNER_NOT_SUPPORTED, SCANNER_FAILED or DISCOVERY_INTERRUPTED.
Rejected counts describe canonical candidates; duplicate/raw-observation counts are separate.
Server-filter exclusions are unavailable, rather than guessed as PRICE_FILTER/LIQUIDITY_FILTER.

GET `/api/runs/{run_id}` and run cards expose status/counts; the detail page shows last success,
per-band/raw/unique/rejected/watch counts, preparation and latest qualification counts.
GET `/api/runs/{run_id}/discovery?limit=20&offset=0` exposes paginated complete attempt audits.
POST `/api/runs/{run_id}/discovery/refresh` requests a generation only when disabled.
Existing `/api/candidates` and Stage 5/runtime storage link by run_id, conId, session and
checkpoint after ready_at and before the next successful ready_at. Qualification counts are
labelled by checkpoint; they are not scanner ranks or fabricated discovery-time qualifications.

## Upgrade and verification

Use the existing `scripts/migrate_activity_filter.py --runs-config OLD --output NEW` to archive
V4/V5 configurations and construct current V6 runs. It writes a separate file, preserves old
run specifications/history and per-run risk/enabled state, and does not activate anything.
Back up the runtime database and configuration before the existing deployment procedure.
Old enabled versions must be migrated before starting this package; archived versions remain
readable. US / All no longer needs a listing snapshot; other market policies are preserved.

Explicit `universe_source: FIXED` or `RESEARCH` requires a populated `universe_snapshot` and
no discovery profile. No implicit fallback is permitted.

Normal tests mock broker calls. `tests/test_candidate_discovery_gateway.py` is skipped by default.
To run the read-only PAPER probe, set `STOCKER_DISCOVERY_IBKR_CONFIG` to a PAPER configuration
with an unused client ID and run that test while Gateway/TWS is available. It exercises all
five cap requests, metadata qualification and persistence without constructing an execution
engine. Real entitlement coverage, cap-filter behavior and non-empty active-session results
still need confirmation on the target Gateway before rollout.

## Changed files

- Core: `discovery.py` (profile/source), `methods.py` (method ownership/version),
  `runs.py` (effective run profile), `config.py` (empty catalogue can start dynamic discovery).
- Execution: `discovery.py` (service/audit store), `ibkr.py` (raw scans/conId metadata),
  `session_hard_universe.py` (composition), `stage5.py` (qualified identity input only),
  `runtime.py` (profile readiness and existing daily/refresh lifecycle).
- Dashboard: `universe_runs.py`, `read_service.py`, `controls.py`, `app.py`,
  `static/dashboard.js`.
- Migration: `scripts/migrate_activity_filter.py`.
- Tests: `test_candidate_discovery.py`, `test_candidate_discovery_gateway.py`,
  `test_discovery_dashboard.py`, `test_activity_filter_migration.py`,
  `test_session_hard_universe.py`, `test_stage10_extension_builder.py`,
  `dashboard_run_summary.cjs`.
- Documentation: this file, `docs/ARCHITECTURE.md`, `docs/universes.md`, `AGENTS.md`.
