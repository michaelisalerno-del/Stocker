# Stocker V2 platform redesign

Status: **Proposed for owner review. Not accepted for implementation.**

Architect: project `architect` role, read-only, high reasoning effort.

This plan replaces the active prospective recorder and web application with a small,
bounded, generic platform for IBKR prospective recording, idea evaluation, and shadow
outcomes. It preserves clean extension points for later portfolio, risk, paper, and
live work, but it does not authorize or implement any order routing.

## Decision summary

Immediate V2 implements only:

- prospective, market-data-only IBKR recording;
- first-party generic idea plugins;
- observations, signals, unapproved proposed positions, and unapproved proposed trades;
- generic shadow positions, marks, and outcomes;
- one compact SQLite WAL operational database;
- a transient durable callback inbox, compact receipts, and automatic compaction;
- bounded operational retention;
- checked, compressed, rotating daily and weekly backups with a total byte cap;
- a bounded generic read API;
- exactly three primary web views: **Live**, **Ideas**, and **Results**;
- a diagnostics drawer;
- removal of EODHD, cross-vendor, source-transfer, Parquet, and report-package paths
  from the active prospective runtime;
- one-way migration from the legacy operational database into a new database; and
- proposal-domain seams that do not make later risk and execution impossible.

Later phases require separate owner authorization and plans for:

- portfolio construction;
- server-side risk approval;
- durable approved order intents;
- isolated IBKR paper execution;
- broker-order reconciliation;
- broker-position and cash reconciliation; and
- controlled live execution.

Later phases are not implemented merely because this plan names their boundaries.
Paper and live remain unavailable. Live remains disabled by default.

## 1. Current-state findings

### Repository state inspected

- Planning branch: `codex/stocker-v2-platform-redesign`.
- Remote base: `origin/main` at `3da8648`.
- Governance commit carried onto the branch: `1074f0a`.
- The worktree was clean before this plan was written.
- `docs/plans/README.md` was the only prior plan document and is not an active plan.
- The Architect inspected the database, migrations, recorder, IBKR bridge, callback
  inbox, partition store, read store, web/API/static application, backups, reports,
  deployment units, idea implementations, shadows, execution placeholders, and tests.

### Useful invariants to preserve

- The recorder is explicitly no-order and tests its public adapter surface for the
  absence of order, account, and position methods.
- `OfficialMarketDataOnlyClient` is a narrow facade around the official IBKR client.
  Loopback restrictions, request budgets, request generations, tombstones, reconnect
  handling, callback containment, and API provenance are valuable.
- SQLite already uses WAL, `synchronous=FULL`, foreign keys, a busy timeout, and a
  recorder lease.
- The durable callback inbox has strong crash invariants: admission before processing,
  generation-fenced leases, acknowledgement after durable projection, poison
  quarantine, and a bounded unacknowledged backlog.
- The web store opens SQLite using URI `mode=ro` and `PRAGMA query_only=ON`; the web
  process has no writer authority.
- Existing tests cover duplicate callbacks, stale leases, crash points, raw-store
  recovery, reconnect gaps, read-only adapter shape, and no-order imports. Preserve
  these behaviours through focused V2 tests, not compatibility wrappers.
- The IB Gateway loopback/firewall/manual-authentication boundary and official API
  provenance/update jobs remain useful as market-data-only infrastructure.

### Size and coupling

The current active prospective runtime is too large and too idea-specific:

- about 67,380 lines across Python and prospective migrations;
- 28 migration files through `0026`;
- 95 tables, 13 views, 23 triggers, and 68 indexes in the migration history;
- approximately 6,100 lines in `recorder_repository.py`;
- approximately 3,200 lines in `live_recorder.py`;
- approximately 2,600 lines in `read_store.py`;
- approximately 2,000 lines in `durable_inbox.py`;
- 39 FastAPI operations: 37 GET routes and two replay POST routes;
- 11 primary web screens and idea-aware polling; and
- 4,571 lines across the web Python, HTML, CSS, and JavaScript.

M1C, quiet state, opening reversal, and opening leader logic crosses core migrations,
repositories, orchestration, routes, reports, polling, navigation, and renderers.
Opening leader demonstrates the cost: one idea required another core table, repository
path, route, polling path, and primary screen.

### Storage and backup problems

- `callback_inbox_v1` can replace acknowledged payloads with hashes, but it retains
  every row forever. Width becomes smaller while row count stays unbounded.
- Evidence is duplicated across SQLite and immutable Parquet/raw partitions. The split
  requires manifests, sidecars, staging, quarantine, recovery reconciliation, hashes,
  and Parquet-aware web reads because the two stores cannot commit atomically.
- Online backups are checked and hashed, but they are full uncompressed timestamped
  database copies whose declared retention is `immutable_until_explicit_operator_removal`.
- The backup timer runs daily and has no daily/weekly rotation or total-directory cap.
- Daily ChatGPT report packages create append-only directories and ZIP generations.
  They are source-transfer-specific and have no automatic retention.

### EODHD and source-transfer coupling

The active prospective runtime still contains and sometimes defaults to:

- EODHD after-session capture;
- `EODHD_API_TOKEN` deployment fields;
- provider-parity observations and migrations;
- transfer coordinators and transfer validity;
- cross-vendor comparison reports and ZIPs;
- source-transfer routes, web panels, and blockers; and
- EODHD Group-O preparation inside the prospective package.

Historical EODHD research under `stocker_data` and `stocker_research` is legitimate
reproducibility material and remains outside the active runtime. V2 removes EODHD only
from prospective, shadow, future paper, and future live paths.

### Execution placeholders are not a paper foundation

- `Broker.place_order(ProposedOrder)` relies on unspecified external risk checks.
- `PaperBroker` is in-memory and immediately labels submissions filled.
- Current risk checks have no durable decision identity, state freshness, account
  binding, emergency state, or reconciliation.
- Execution state is only positions, cash, and a sync timestamp.
- `configs/server.example.yaml` says `mode: paper`, and the executor script logs a dry
  run, which overstates current capability.

There is no order-capable IBKR adapter, durable approved intent, safe paper-account
validation, idempotent submission, uncertain-submission handling, or order/position
reconciliation. Do not preserve these placeholders through wrappers.

## 2. Target architecture

### Runtime flow

```text
IB Gateway/TWS with Read-Only API enabled
        ↓
narrow IBKR market-data-only bridge
        ↓
transient durable callback inbox
        ↓
normalised SQLite market events + latest projection
        ↓
independently checkpointed generic idea plugins
        ↓
observations / signals / unapproved proposals
        ↓
generic shadow engine
        ↓
compact indexed SQLite projections
        ↓
read-only FastAPI
        ↓
Live | Ideas | Results
        ↘ diagnostics drawer
```

Future controlled trading extends only after `proposed trade`:

```text
proposed trade
        ↓
portfolio construction
        ↓
server-side risk approval
        ↓
durable approved order intent
        ↓
paper or live execution adapter
        ↓
broker acknowledgement and fills
        ↓
order, position, and cash reconciliation
```

No immediate component below the proposal boundary exists in V2.

### Deployment shape

Keep four simple OS boundaries:

1. `stocker-recorder`: sole authoritative V2 database writer; owns market-data
   subscriptions, callback durability, plugin scheduling, and shadow projection.
2. `stocker-web`: query-only SQLite reader serving FastAPI and static files; never
   receives an IBKR, plugin runner, risk, or execution object.
3. `stocker-backup`: SQLite online-backup reader writing only the backup directory.
4. Existing IB Gateway/display/loopback-proxy units, relabelled market-data-only.

Use FastAPI, plain HTML/CSS/JavaScript, SQLite WAL, systemd, direct Python, and a small
number of focused modules. Do not introduce Redis, Celery, Kafka, RabbitMQ,
Kubernetes, a message bus, a workflow engine, a third-party plugin framework, or a
network microservice mesh.

### Proposed packages

Build a clean package rather than a `v2` compatibility layer over the old runtime:

```text
packages/stocker_runtime/src/stocker_runtime/
  config.py
  domain.py
  cli.py
  storage/
    connection.py
    repository.py
    retention.py
    backup.py
    legacy_import.py
    migrations/0001_v2.sql
  ingestion/
    official_bridge.py
    ibkr_market_data.py
    inbox.py
    recorder.py
  ideas/
    contract.py
    discovery.py
    runner.py
  shadow/
    engine.py
  web/
    app.py
    queries.py
    static/index.html
    static/app.css
    static/app.js

packages/stocker_ideas/src/stocker_ideas/
  plugins/
    opening_leader_continuation_v0.py
```

`stocker_ideas` depends only on immutable domain/plugin DTOs. It must not depend on
ingestion, IBKR, web, operational repositories, environment readers, risk, or
execution.

## 3. Operating modes, authority, and protected data

### Immediate modes

| Mode | Inputs | Permitted outputs | Forbidden |
| --- | --- | --- | --- |
| `prospective_record` | Live IBKR market data | Market events, observations, signals, unapproved proposed positions/trades | Shadow positions, approvals, intents, orders, account reads |
| `shadow` | Live IBKR market data | Record outputs plus virtual positions, marks, and outcomes | Approvals, intents, orders, account reads |
| Research/replay | Separate historical tooling | Research artifacts outside operational V2 | Silent writes into protected V2 data |
| Paper | Not implemented | None | Any broker transmission |
| Live | Not implemented | None | Any broker transmission |

Immediate recorder configuration accepts only `prospective_record` or `shadow`.
`paper`, `live`, unknown, missing, or conflicting values fail before connecting.

### Trust and data ownership

- Ingestion owns the IBKR market-data connection, subscription identity, callback
  durability, timestamps, ordering, duplicates, gaps, and staleness.
- Plugins declare requirements but never own subscriptions or IBKR clients.
- Plugins interpret admitted data and produce evidence with no authority.
- Portfolio construction, risk, intents, execution, and reconciliation do not exist in
  immediate V2.
- The shadow engine owns virtual evidence only. It never claims broker fills or positions.
- The web owns presentation only and exposes GET routes only.
- The backup job owns backup files only.
- Historical research may use archived EODHD data but cannot determine active runtime
  provider behaviour.
- One recorder process is the only operational database writer. There is no V1/V2
  dual write.

The official IBKR `EClient` inherently contains order methods, so only a private bridge
may hold it. The public facade, callback object, runtime configuration, and tests expose
market-data methods only. Startup requires `read_only=true`, loopback-only socket
evidence, and documented operator verification of IB Gateway Read-Only API mode.

### Protected data classes

Every run and output records exactly one class:

- `development`
- `retrospective`
- `stress`
- `prospective_protected`
- `shadow_protected`
- future-only `paper_protected`
- future-only `live_protected`

Production V2 permits only `prospective_protected` or `shadow_protected`.

- Freeze plugin version, code hash, parameters, and universe before activation.
- A parameter change creates a new idea instance; it never rewrites prior results.
- Research code does not open the operational V2 database by default.
- Migration preserves original classification; uncertain data becomes protected.
- Reuse for fitting, feature selection, threshold selection, or hypothesis repair
  requires an explicit owner-approved protocol naming the period, contamination, use,
  and new holdout.
- Retention expiry is not permission to inspect protected results for tuning.

## 4. Generic idea-plugin contract

Use a small first-party Python `Protocol` and stdlib module discovery. This is not a
third-party framework or a hostile-code sandbox.

Each configured module exposes one `plugin()` factory implementing:

```python
class IdeaPlugin(Protocol):
    @property
    def manifest(self) -> IdeaManifest: ...

    def requirements(
        self,
        activation: IdeaActivation,
    ) -> tuple[MarketDataRequirement, ...]: ...

    def evaluate(
        self,
        batch: IdeaBatch,
        state: JsonValue,
    ) -> IdeaEvaluation: ...
```

### Contract objects and limits

- `IdeaManifest` contains API version 1, stable `idea_id`, immutable
  `idea_version`, display metadata, permitted record/shadow modes, output kinds,
  parameter schema/version, maximum state bytes, and maximum outputs per batch.
- `IdeaActivation` contains a core-generated instance ID, canonical parameters/hash,
  plugin code hash, activation time, run ID, protected-data class, and configured
  universe.
- `MarketDataRequirement` declares feed kind, instrument identity, cadence, and whether
  gaps or staleness block that plugin. Core aggregates requirements and owns requests.
- `IdeaBatch` contains immutable bounded market-event DTOs, mode, input watermark, and
  causal timestamps. It has no database connection, filesystem path, environment
  accessor, IBKR client, account, risk, or web object.
- `IdeaEvaluation` contains bounded state and typed outputs. State is at most 64 KiB,
  each output payload is at most 16 KiB, and one batch emits at most 256 outputs.

Permitted outputs are:

- `Observation`
- `Signal`
- `ProposedPosition`
- `ProposedTrade`

Proposals are explicitly `unapproved`. They cannot contain approval identity, broker
account, broker order ID, transmit flags, or mode mutation.

Core derives deterministic output IDs from instance ID, input range, kind, ordinal,
as-of timestamp, and payload hash. Retry produces the same ID. A collision with
different content is a plugin-fatal invariant violation.

Each plugin owns an independent checkpoint and transaction. A failure rolls back only
that plugin's state and outputs, marks the instance degraded or disabled, and lets
ingestion and unaffected plugins continue.

Discovery rejects duplicate `(idea_id, idea_version)`, unsupported API versions,
invalid manifests, forbidden outputs, and out-of-bound requirements before IBKR
subscriptions start. Static import tests and review enforce the trust boundary; an
in-process Python plugin is not claimed to be safe against malicious code.

### Adding an idea

After the contract is accepted, adding a reviewed idea normally requires only:

1. its plugin module;
2. its tests and frozen/golden fixtures; and
3. one activation entry in configuration.

It must not require a primary tab, FastAPI route, core migration, central renderer,
repository projection, broker change, risk change, or execution change.

### Current ideas

Opening Leader Continuation V0 is the recommended first reference plugin because it is
the newest active prospective idea and provides a direct test of the generic seam.
Port its causal ranking and outputs; do not wrap its current table, route, repository,
or screen.

M1C, quiet state, and opening-reversal families remain readable in the archived legacy
database until the owner chooses which exact frozen contracts deserve independent
plugin ports. Do not migrate every historical experiment automatically.

### Generic shadow engine

Immediate V2 shadows only `ProposedTrade`. A `ProposedPosition` remains recorded
evidence until future portfolio construction exists.

- Open using the first valid causal quote after the proposal commits durably.
- Use ask for buy and bid for sell at entry, with the inverse at exit.
- Record exact market-event IDs and a named versioned cost model.
- Allow at most eight configured horizons and a maximum 30-day horizon.
- Record incomplete/invalid evidence instead of inventing a price.
- Resume deterministically after restart.
- Never call IBKR, read account state, or describe a virtual event as a broker fill.

## 5. Proposed V2 schema

Create a new database. Never apply this schema to the legacy operational file.

Writer pragmas:

- `journal_mode=WAL`
- `synchronous=FULL`
- `foreign_keys=ON`
- `busy_timeout=5000`
- `wal_autocheckpoint=1000`
- `journal_size_limit=67108864`
- `auto_vacuum=INCREMENTAL`, set before schema creation

Use integer UTC microseconds for ordering, typed columns for indexed fields, and
canonical compact JSON only for bounded extension data. API queries never filter by
arbitrary JSON.

### Core and runtime

1. `schema_migrations`
   - `version INTEGER PRIMARY KEY`, `name TEXT UNIQUE`, `sha256`, `applied_at_us`.
2. `runs`
   - `run_id PRIMARY KEY`, constrained `mode` and `source='ibkr'`, start/end times,
     config hash, Git commit, data class, status, and optional prior run.
3. `recorder_generations`
   - `(run_id,generation) PRIMARY KEY`, owner, start/end, clean-stop flag, termination
     code.
4. `runtime_state`
   - one row per run containing generation, lifecycle/reason, separate process,
     callback, admission, and projection heartbeats, connection generation, inbox
     measures, DB/WAL bytes, and `order_capability_observed=0`.
5. `incidents`
   - stable ID, run, scope, severity, code, optional plugin/subscription scope,
     open/resolved times, bounded details; partial index on unresolved rows.
6. `gaps`
   - stable ID, run/subscription, interval, reason, data-loss-possible, required,
     resolution; partial index on unresolved rows.

### Market-data ingestion

7. `instruments`
   - stable identity hash, optional IBKR conId, kind, symbol, exchange, currency, and
     nullable option identity fields; unique conId when present.
8. `subscriptions`
   - stable ID, run and generations, instrument/feed/request identity, lifecycle,
     requirements hash, latest event; unique `(run_id,connection_generation,request_id)`.
9. `callback_inbox`
   - autoincrement source sequence, unique event UID, run/generation/request/callback
     identity, received/provider times, bounded original payload, lifecycle/lease/
     attempts/failure; indexed source-order leasing.
   - hard maximum 50,000 nonterminal rows and 64 KiB per payload.
10. `callback_receipts`
    - stable batch ID, run, contiguous first/last source sequence, counts/time bounds,
      kind/status counts, chained payload hash, first/last normalized event IDs.
11. `callback_compaction_watermarks`
    - one row per run with compacted-through sequence, cumulative counts/time bounds,
      and a rolled receipt-chain hash. This is the permanent proof after granular
      receipts rotate.
12. `market_events`
    - stable event UID, run/source sequence, instrument/feed/kind, event/receive time,
      connection generation, quality bitset, typed OHLCV/bid/ask/last/size fields, and
      bounded extension payload.
    - indexes by run/time, instrument/kind/time, and run/kind/time.
13. `market_latest`
    - one latest row per `(instrument_id,feed_kind)`, updated transactionally with its
      source market event.

### Ideas and shadows

14. `idea_plugins`
    - `(idea_id,idea_version) PRIMARY KEY`, API version, display metadata, manifest/code
      hashes, bounded manifest JSON, discovery time.
15. `idea_instances`
    - instance ID, plugin identity, run/mode, parameter JSON/hash, activation/
      deactivation, health/error, data class; unique activation identity per run.
16. `idea_checkpoints`
    - one row per instance with last committed market-event ID, bounded state/hash,
      success/update time, and consecutive failures.
17. `idea_outputs`
    - stable output ID, run/instance, constrained output kind, subject instrument,
      emitted/as-of/valid-until times, generic direction/strength/confidence/horizon,
      input event range, bounded canonical payload/hash, data class, and constrained
      authority status `recorded` or `unapproved`.
    - indexes by run/kind/time, instance/kind/time, and instrument/time.
18. `idea_output_legs`
    - `(output_id,leg_number) PRIMARY KEY`, instrument, action/target, quantity/value/
      currency, and optional non-authoritative price hint. No account or order columns.
19. `shadow_positions`
    - stable ID, source proposed-trade output, run/idea, open/close times, lifecycle,
      cost/fill model IDs, currency, invalid reason, data class.
20. `shadow_legs`
    - position/leg identity, instrument, side, quantity, exact entry/exit market-event
      IDs and prices.
21. `shadow_marks`
    - position/time, gross value/P&L/return, quality bitset, bounded payload; unique
      position/time.
22. `shadow_outcomes`
    - one row per position with outcome time/reason, gross/net P&L, return, MFE, MAE,
      completeness, and bounded payload.
23. `migration_manifests`
    - source DB hash/schema digest, importer version, times, source/imported/omitted
      counts, target digest, and verification status.

Do not add portfolio, risk-decision, order-intent, broker-order, fill, account, cash, or
reconciled-position tables in immediate V2.

## 6. Compact API and three-view web application

### API

Immediate OpenAPI contains these GET routes only:

- `GET /api/v2/meta`
- `GET /api/v2/live`
- `GET /api/v2/ideas`
- `GET /api/v2/ideas/{instance_id}`
- `GET /api/v2/results`
- `GET /api/v2/results/{position_id}`
- `GET /api/v2/diagnostics`

The static root and asset paths are outside the JSON API.

Rules:

- default page 50, maximum 200;
- cursor pagination ordered by `(timestamp,id)`, never offsets;
- cursor binds route and normalized filter hash; malformed/mismatched cursors return 422;
- `/live` reads only current state/latest/active rows and returns at most 250 instruments;
- `/ideas` returns at most 100 instances and bounded generic summaries;
- idea detail returns at most 200 outputs and has no idea-specific response schema;
- `/results` clearly marks every item `shadow=true`, `broker_position=false`, `fill=false`;
- diagnostics caps incidents, gaps, subscriptions, and backup entries at 200 each;
- maximum requested output window is seven days and results window is 30 days;
- maximum serialized response is 512 KiB;
- SQLite query budget is 100 ms using a progress handler; timeouts return a stable error;
- no callback-payload, arbitrary SQL, bulk export, report ZIP, transfer, replay mutation,
  broker, account, order, fill, or broker-position endpoint.

### Primary view 1: Live

- Recorder lifecycle and health.
- IBKR connection/freshness/gaps.
- Active feed counts and latest generic instrument values.
- Storage, WAL, callback-inbox, and backup summaries.
- No recommendation, broker state, or order control.

### Primary view 2: Ideas

- Discovered and activated plugin cards.
- Healthy, degraded, or disabled state per instance.
- One generic stream of observations, signals, proposed positions, and proposed trades.
- Common fields plus a bounded generic detail panel.
- No idea-specific tab, renderer, or route.

### Primary view 3: Results

- Open, closed, incomplete, and invalid shadow positions.
- Virtual marks, outcomes, and bounded generic aggregates.
- Exact source proposal and market-event references.
- Conspicuous `virtual only; no broker orders, fills, or positions` language.

A global diagnostics drawer contains incidents, gaps, subscriptions, backup state,
database/WAL size, build/config/plugin hashes, and retention status. It is not a fourth
primary view.

The fixed application banner is:

`PROSPECTIVE / SHADOW ONLY — NO APPROVAL OR EXECUTION`

Remove primary screens for System, Universe, Episode, Options, Ledgers, Audit, Quiet
Tape, Quiet Episode, Risk Ledger, Concentration, Opening Leader, Source Transfer, and
Report Packages. Relevant evidence appears through generic idea/result detail or the
diagnostics drawer.

## 7. Retention, backups, and report policy

### Proposed operational retention defaults

All values are frozen before a run and require owner approval:

- terminal callback payload rows: 24 hours after normalized persistence and ack;
- granular callback receipt batches: 90 days and at most 2,048 per run, then rolled
  transactionally into the permanent chained watermark before deletion;
- request tombstones/dedup identities: seven days;
- raw tick/quote/depth market events: 30 days;
- completed bars: 400 days;
- closed subscription history and routine diagnostics: 90 days;
- resolved incidents and gaps: 400 days; unresolved rows never prune;
- idea outputs, shadow positions, marks, and outcomes: seven years;
- small run, activation, migration, and compaction metadata: retained;
- operational DB soft cap: 8 GiB;
- WAL cap: 64 MiB;
- maintenance batches: at most 10,000 rows or 100 ms per transaction outside callback
  pressure.

At 85% of the DB cap, prune only already-expired classes and run bounded checkpoints/
incremental vacuum after session. At 95%, stop optional high-volume feeds and report
degraded. At the hard cap, cancel subscriptions, disconnect, and enter
`STORAGE_CAP_FATAL`. Never delete unexpired protected evidence silently.

### Callback lifecycle

1. Durably admit the original bounded callback payload to `callback_inbox`.
2. Normalize deterministically into `market_events` and `market_latest` in SQLite.
3. Commit plugin/shadow work through their independent checkpoints.
4. Acknowledge only after required durable projections commit.
5. Create compact chained receipts for contiguous source-sequence batches.
6. Delete terminal payload rows only after their receipt is durable and the configured
   24-hour diagnostic window expires.
7. Roll old granular receipts into the permanent compaction watermark before rotating
   them.

V2 does not write duplicate routine callback evidence to Parquet or raw partitions.
Malformed payloads stay in the bounded inbox/quarantine until resolved or the run is
closed; they do not create an unbounded second store.

### Backup policy

- Use SQLite's online-backup API into a temporary database.
- Run `quick_check` before compression.
- Compress deterministically with gzip, fsync, hash both compressed and uncompressed
  content, write an atomic JSON manifest, and remove the temporary uncompressed file.
- Retain 14 daily archives and 12 weekly archives.
- Enforce a recommended total backup-directory cap of 8 GiB.
- Remove the oldest excess daily archives first, then excess weekly archives, while
  preserving at least two valid archives in each tier.
- If a new backup cannot fit without breaking the floors, preserve existing valid
  backups, fail the new backup, and record a degraded incident.
- Restore tests always decompress into a new file, verify both hashes, and run
  `quick_check`; they never overwrite the active DB in place.

The legacy database receives one checked, hashed, compressed, read-only archival
recovery set at cutover. It is not copied into every V2 daily backup.

### Reports

Remove automatic ChatGPT report ZIP creation and persistent report-download routes.
Immediate V2 has no bulk report export. Any future export requires a separately
approved, authenticated, bounded design with explicit protected-data rules.

## 8. Failure and recovery behaviour

### Global fatal conditions

Stop subscription admission, disconnect IBKR, persist the condition when possible,
and require operator action for:

- SQLite corruption, failed integrity/foreign-key checks, unsupported schema, or
  migration checksum mismatch;
- inability to durably admit a callback or a hard inbox overflow;
- critical disk exhaustion or the hard DB cap with no expired data eligible to prune;
- loss of the sole writer lease or duplicate authoritative writers;
- inability to preserve an accepted callback in either inbox or normalized storage;
- unsafe IBKR capability: order/account/position surface or callback observed;
- non-read-only configuration, non-loopback socket, or unsupported operating mode;
- broken receipt chain or deletion-before-projection invariant;
- configuration/plugin identity mutation within a run; or
- corruption of required plugin activation or code hashes.

Restart never clears a persisted fatal condition automatically.

### Recoverable degraded conditions

Continue unaffected ingestion/plugins for:

- a temporary IBKR disconnect, farm reset, pacing error, stale stream, or reconnect;
- an unclean restart where every received callback is durable;
- one plugin exception, timeout, malformed output, excessive output/state, or invariant
  collision;
- denial of an optional subscription or line-budget pressure;
- one symbol/plugin gap;
- missing/stale/crossed quotes for a shadow result;
- a failed on-demand diagnostic or web request;
- a non-critical backup or retention-maintenance failure; or
- temporary compaction delay below the hard inbox limit.

Record scoped gaps and block only plugins whose declared requirements need the affected
continuity. Market closed is expected, not degraded. Web failure never stops recorder
operation.

### Restart sequence

1. Verify database/schema/config/plugin hashes and acquire the writer lease.
2. Load prior generation and active fatal state.
3. Reclaim only expired callback leases.
4. Normalize pending callbacks to deterministic IDs; unique conflicts must match exact
   content.
5. Acknowledge callbacks only after market-event/latest commit.
6. Resume each plugin from its independent committed event checkpoint.
7. Open scoped uncertainty gaps for subscription intervals.
8. Reconnect with a new connection/request generation and rebuild aggregated requirements.
9. Resume shadow positions from durable proposal/leg/mark state.
10. Run bounded compaction/retention only after recovery backlog is healthy.

Immediate V2 has no submitted-order state. Later execution must define uncertain
submission explicitly and must never reuse callback-recovery semantics as an order
retry policy.

## 9. Migration, cutover, rollback, and deletions

### One-way legacy import

Do not alter `/var/lib/stocker/prospective/prospective.sqlite3` in place.

Create:

- a read-only legacy recovery set;
- `/var/lib/stocker/v2/stocker-v2.sqlite3`; and
- `/var/lib/stocker/backups-v2`.

The importer opens legacy SQLite in `mode=ro`, verifies `quick_check`, records its
SHA-256 and schema digest, and writes only a new empty V2 database.

Mapping policy:

- `prospective_run` to `runs`;
- durable underlying/option identities to `instruments`;
- retained normalized bars/quotes to `market_events`;
- selected current scores/checkpoints/evidence to generic `idea_outputs`, preserving
  source table/key and semantic provenance in bounded migration payloads;
- legacy signal episodes to `signal` outputs;
- selected shadow structures/valuations/outcomes to generic shadow tables;
- closed incidents/gaps to V2 incidents/gaps;
- active legacy incidents/gaps become a migration incident, not active V2 truth;
- raw Parquet, callback payload duplicates, and unmapped experiment tables remain in
  the immutable legacy archive; and
- `migration_manifests` reconciles every source row as imported or intentionally omitted.

Do not expose an active legacy-table wrapper from V2.

Rehearse against copied databases at every schema state through `0026`. Require source
byte identity, deterministic target digest, imported-plus-omitted counts, foreign-key
checks, `quick_check`, golden generic projections, protected-class preservation, and
absence of EODHD credentials/config in the target.

### Cutover

1. Pre-create V2 users, directories, configuration, release, and service units.
2. Stop at a planned market-closed window.
3. Take a checked compressed V1 database backup and preserve SQLite/WAL, raw partitions,
   sidecars/staging/quarantine, bundles, and required reports as one read-only recovery set.
4. Stop V1 recorder and web; prove no writer/lease remains.
5. Import from stopped V1 into a temporary V2 database.
6. Verify hashes, counts, foreign keys, `quick_check`, ownership, and byte cap.
7. Atomically rename V2 into place.
8. Install V2 units/config with no EODHD, transfer, report-ZIP, Parquet, paper, or live fields.
9. Start V2 web and prove query-only behaviour.
10. Start V2 recorder with a new run and connection generation.
11. Verify loopback/read-only IBKR, callback durability, plugin discovery, and zero
    order capability/broker mutation.
12. Declare completion only after a bounded observation window and successful checked
    V2 backup.

There is no dual write.

### Rollback

- Before V2 admits its first callback: stop V2 and restore the prior release/service
  pointer and untouched V1 database.
- After V2 admits data: stop and back up V2, restore the prior release against untouched
  V1, create a new V1 run/generation, and record the V2 interval as an explicit gap.
- Never reverse-import V2 rows into V1 or claim uninterrupted V1 scientific continuity.
- Prefer roll-forward repair after V2 admission.
- Preserve the old release and recovery set until the owner closes the rollback window.

### Files and runtime surfaces to remove after verified cutover

Remove rather than wrap:

- active `source_transfer.py`, `transfer.py`, `parallel.py`, EODHD prospective config/env,
  transfer reports/routes/tests/UI, and active EODHD preparation;
- `partition_store.py`, `parquet_read_projection.py`, `storage_recovery.py`, raw partition
  manifests, sidecars, staging/quarantine runtime, and Parquet-aware web reads;
- `budget_reports.py`, ChatGPT comparison ZIPs, report routes, and report panels;
- idea-specific migrations, repository methods, orchestration branches, routes, screens,
  polling paths, and renderers for M1C, quiet state, opening reversal, and opening leader;
- old prospective `database.py`, `recorder_repository.py`, `live_recorder.py`,
  `read_store.py`, `web.py`, and static dashboard after V2 proves their required invariants;
- the model-specific active bundle graph if no accepted V2 plugin needs it, retaining
  only small immutable plugin activation/code hashes;
- `stocker_execution.paper`, the direct `Broker.place_order(ProposedOrder)` shortcut,
  stateless risk/execution placeholders, executor dry-run entry point, and misleading
  paper-mode example; and
- legacy migration files from the active installed release after the rollback window.

Keep historical EODHD research/data code, Git history, archived V1 database/runtime
evidence, official IBKR provenance/update tooling, and loopback/firewall/manual Gateway
infrastructure.

## 10. Sequential implementation phases

No Implementer is authorized by this proposed plan. After owner acceptance, assign one
bounded phase at a time. Central schema, recorder ownership, broker access, risk,
execution, reconciliation, and cutover stay sequential.

### Phase 1 — V2 contracts and false-capability removal

- **Objective:** establish mode, domain, and plugin authority boundaries without a
  running service.
- **Files/packages:** add `stocker_runtime/domain.py`, `ideas/contract.py`, exports, and
  import-boundary tests; update packaging.
- **Schema/API:** none.
- **Modes:** types only for `prospective_record` and `shadow`.
- **Paper/live:** absent; no account, approval, or executable order type reaches plugins.
- **Deletions:** remove in-memory `PaperBroker`, direct broker submission interface,
  executor dry-run, stateless execution placeholders, and misleading paper example
  after dependency inspection.
- **Tests:** DTO validation, serialization/size bounds, forbidden fields/imports,
  unsupported mode rejection.
- **Completion:** all four plugin outputs are representable, but approval/execution is not.
- **Dependencies:** accepted plan and owner decisions in Section 14.

### Phase 2 — Compact SQLite schema and bounded retention

- **Objective:** create the authoritative operational store.
- **Files/packages:** storage connection/repository/retention, `0001_v2.sql`, DB CLI.
- **Schema:** all immediate tables/indexes in Section 5; no future trading tables.
- **API:** none.
- **Modes:** offline fixtures only.
- **Paper/live:** no implications; no order state.
- **Deletions:** none yet.
- **Tests:** checksum/order, pragmas, foreign keys, deterministic identity, collision
  equality, payload bounds, query plans, retention/cap behaviour, incremental vacuum,
  stop-at-cap.
- **Completion:** new DB passes integrity checks and cannot grow beyond configured policy
  without a visible fail-stop.
- **Dependencies:** Phase 1.

### Phase 3 — Read-only IBKR ingestion and transient inbox

- **Objective:** record normalized prospective market data with no ideas.
- **Files/packages:** official bridge, IBKR facade, inbox, recorder, config/CLI.
- **Schema:** instruments, subscriptions, inbox/receipts/watermark, market events/latest,
  generations/state/incidents/gaps.
- **API:** none.
- **Modes:** `prospective_record` only.
- **Paper/live:** no account/order methods; fakes/replay only.
- **Deletions:** deliberately introduce no Parquet/raw-partition store.
- **Tests:** callback-before-return, duplicate/order/staleness/gaps, locked DB, overflow,
  poison, reconnect/late generations, all admit-normalize-ack crash points, lease
  recovery, receipt chain, compaction, disk/cap failure, forbidden IBKR surfaces.
- **Completion:** all admitted events survive restart deterministically within bounds.
- **Dependencies:** Phase 2.

### Phase 4 — Plugin discovery, isolation, and reference idea

- **Objective:** activate generic ideas through configuration.
- **Files/packages:** discovery, runner, Opening Leader reference plugin and tests.
- **Schema:** plugin/instance/checkpoint/output/leg tables.
- **API:** none.
- **Modes:** record outputs in `prospective_record`; `shadow` accepted but not yet valued.
- **Paper/live:** outputs remain unapproved.
- **Deletions:** no call into current idea repositories/tables/routes.
- **Tests:** manifests/API versions, aggregated requirements, deterministic retry,
  independent failure, checkpoint restart, causality, all bounds, forbidden imports,
  frozen Opening Leader parity, no historical backfill.
- **Completion:** a synthetic new idea needs only module, tests, and config.
- **Dependencies:** Phase 3.

### Phase 5 — Generic shadow positions and outcomes

- **Objective:** value proposed trades as explainable virtual evidence.
- **Files/packages:** shadow engine and versioned fill/cost policy DTOs.
- **Schema:** shadow position/leg/mark/outcome tables.
- **API:** none.
- **Modes:** behaviour only in `shadow`.
- **Paper/live:** no broker identity, position, order, or fill claims.
- **Deletions:** no idea-specific virtual-ledger view reuse.
- **Tests:** causal entries, bid/ask conventions, missing/stale/crossed quotes, multi-leg
  atomicity, costs/horizons, restart/duplicates/open recovery, MFE/MAE determinism,
  IBKR/import isolation.
- **Completion:** every result traces to proposal and market-event IDs.
- **Dependencies:** Phase 4.

### Phase 6 — Compact API and three-view UI

- **Objective:** expose only the generic read model.
- **Files/packages:** web app/queries and replacement HTML/CSS/JS.
- **Schema:** no domain tables; measured indexes only.
- **API:** exactly the seven GET routes in Section 6.
- **Modes:** display record/shadow only.
- **Paper/live:** fixed no-approval/no-execution banner; no broker-aware controls.
- **Deletions:** replay POSTs, full snapshot, idea-specific/transfer/report routes,
  idea-specific screens, pollers, and renderers.
- **Tests:** GET-only OpenAPI, route count, cursors/filters/limits, query-only DB,
  response/query budgets, auth/rate limits, polling, exactly three nav labels,
  diagnostics drawer, unknown-plugin generic rendering, accessibility/no-order language.
- **Completion:** Live/Ideas/Results are the only primary views.
- **Dependencies:** Phases 3–5.

### Phase 7 — Compressed backup tiers and deployment rehearsal

- **Objective:** make V2 operable/recoverable without activation.
- **Files/packages:** V2 backup module; recorder/web/daily/weekly units and env examples;
  SQLite boundary preparation.
- **Schema/API:** diagnostics reads only bounded backup manifests.
- **Modes:** inactive deployment rehearsal.
- **Paper/live:** IB Gateway labelled market-data-only; no order service.
- **Deletions:** prospective EODHD/token/parallel/transfer fields from V2 deployment.
- **Tests:** online backup under writes, hashes/compression/restore, rotation/byte cap,
  atomic failure, users/permissions/query-only/loopback, SIGTERM/restart.
- **Completion:** backups are checked, compressed, bounded, and restored successfully.
- **Dependencies:** Phase 6.

### Phase 8 — Import, cutover, and legacy active-runtime deletion

- **Objective:** migrate once, cut over safely, and delete superseded active code.
- **Files/packages:** legacy importer, migration fixtures/scripts, deployment/runbooks.
- **Schema/API:** migration manifest populated; V2 API only.
- **Modes:** new protected V2 run after cutover.
- **Paper/live:** still absent.
- **Deletions:** all post-cutover surfaces listed in Section 9 after the rollback window.
- **Tests:** every schema through `0026`, deterministic import, source identity,
  count/hash reconciliation, omissions, cutover/rollback rehearsal, one writer, full
  repository checks, no forbidden active references.
- **Completion:** only V2 services are active and no prospective runtime reference remains
  to EODHD, transfer, Parquet, report ZIPs, legacy idea wiring, paper, or live execution.
- **Dependencies:** Phases 1–7 and explicit cutover approval.

### Later authorized Phase 9 — Portfolio construction

- Consume eligible proposals and objectives into deterministic desired exposure.
- Add portfolio requests, not approvals or orders.
- Test conflicts, aggregation, stale proposals, and deterministic construction.

### Later authorized Phase 10 — Risk approval

- Add server-side limits, emergency state, reconciled-state freshness, and immutable
  decisions bound to exact portfolio requests.
- No broker submission.
- Test every reject boundary, stale state, loss/exposure limits, mode/account absence,
  and emergency stop.

### Later authorized Phase 11 — Durable order intents

- Create immutable unique intents only from valid risk approval.
- Model pending/ready/blocked/uncertain without an adapter.
- Test duplicates, binding/expiry, restart, and immutability.

### Later authorized Phase 12 — Isolated IBKR paper execution

- Add the sole order-capable boundary for an explicitly allowlisted paper account.
- Validate broker-reported identity; unknown or live fails closed.
- Persist client order identity before submission.
- No live configuration or fallback.

### Later authorized Phase 13 — Broker-order reconciliation

- Persist acknowledgements, open/completed orders, executions, fills, rejection,
  cancellation, and uncertain submission.
- Reconcile before retry; never infer a fill from submission.

### Later authorized Phase 14 — Broker position/cash reconciliation

- Persist broker snapshots and mismatch incidents.
- Risk reads fresh reconciled broker truth, never desired/shadow state.
- Block action on stale or mismatched state.

### Later authorized Phase 15 — Controlled live execution

- Separate live allowlist and explicit operator authorization, disabled by default.
- Require conspicuous mode, emergency stop, proven paper/live separation, production
  runbook, and a tiny controlled rollout.
- Requires separate owner approval of accounts, limits, emergency behaviour, and paper/
  reconciliation evidence.

## 11. Acceptance and performance tests

### Immediate V2 release gates

- Formatting, lint, strict typing, focused tests, and full repository checks pass.
- OpenAPI exposes only seven V2 GET operations; no mutating API operation exists.
- Runtime/plugins/web do not import execution. Plugins do not import IBKR, storage,
  environment/config, filesystem, or web modules.
- Public IBKR facade/callback surfaces contain no order, account, position, execution,
  P&L, or completed-order method/callback.
- Unsafe host, non-read-only config, unsupported mode, missing loopback, unknown plugin
  hash, or schema mismatch fails before subscribing.
- All callback admission/normalization/acknowledgement crash points pass.
- Duplicate delivery produces one normalized event and one deterministic output set.
- One failed plugin cannot block ingestion or another plugin.
- A gap blocks only plugins whose declared requirements need continuity.
- Proposals remain unapproved; no table/API/model can promote them.
- Shadows use recorded causal quotes and contain no broker identity.
- Inbox backlog/payload, receipts/watermark, DB, WAL, plugin state/output, query, response,
  page, and history bounds are enforced.
- Retention never deletes unexpired protected evidence; hard cap stops recording.
- Daily/weekly backups restore, verify both hashes, and pass `quick_check` within count/
  byte caps.
- Import leaves source byte-identical and reconciles every row imported or omitted.
- Active V2 source/config/env/API contains no prospective EODHD, transfer, cross-vendor,
  Parquet, report ZIP, paper broker, or live route.
- Primary navigation text is exactly `Live`, `Ideas`, `Results`; diagnostics is a drawer.
- The fixed no-approval/no-execution banner is visible on all views.
- An unknown synthetic plugin renders without core route/schema/frontend changes.
- Cutover proves one writer/no dual write; rollback preserves both recovery sets.

### Performance fixture and targets

Measure on the intended server class with:

- 1,000,000 market events;
- 100 active instruments;
- 100 idea instances;
- 100,000 idea outputs;
- 10,000 shadow positions/outcomes; and
- 50,000 pending inbox rows for recovery.

Targets:

- durable callback admission p95 <= 15 ms and p99 <= 50 ms during 200 callbacks/s burst;
- normalization sustained >= 500 events/s, separate from plugin transactions;
- plugin batch <= 256 outputs and target <= 50 ms; overruns degrade that plugin;
- `/api/v2/live` warmed p95 <= 50 ms;
- other bounded API pages warmed p95 <= 100 ms;
- no API response above 512 KiB;
- no-backlog recorder startup <= 5 seconds before connection attempt;
- maximum 50,000-row inbox recovery <= 60 seconds;
- 10,000 terminal-row receipt compaction <= 2 seconds outside callback pressure;
- online backup increases callback p95 by no more than 20%; and
- every API query uses declared indexes; `/live` never scans `market_events`.

Report host/storage details. Laptop `TestClient` timing is not production proof.

## 12. Work suitable for parallel agents

Read-only work may run in parallel after plan acceptance:

- legacy table-to-generic mapping audit;
- test-gap and adversarial-failure inventory;
- benchmark design and SQLite query-plan review;
- candidate plugin semantic/golden-fixture audit;
- systemd/filesystem security review;
- protected-data classification audit;
- UI accessibility/payload-size review; and
- backup capacity measurements on copied databases.

After the plugin contract is frozen, independent plugin golden-fixture analysis may be
parallel, but central config/schema integration is sequential.

Do not parallelize write-capable work on the central schema, recorder/inbox, risk,
execution, broker adapter, reconciliation state machine, or migration cutover.

## 13. Explicit non-goals

- No broker paper execution or live execution.
- No order-capable IBKR adapter.
- No broker account, cash, order, fill, execution, P&L, or position reads.
- No portfolio construction, risk approval, or durable order intent in immediate V2.
- No claim that the current `PaperBroker` is production paper trading.
- No proposal-to-order shortcut.
- No replay controls in the production web API.
- No broker-aware frontend.
- No idea-specific route, table, tab, schema, or renderer.
- No dual write or in-place legacy DB mutation.
- No active-runtime EODHD, cross-vendor, transfer, or provider-equality requirement.
- No deletion of historical EODHD research capability.
- No Parquet/raw-partition side store in V2.
- No automatic large report ZIPs or unbounded exports.
- No malicious-plugin sandbox claim or third-party plugin framework.
- No message bus, task queue, generalized workflow engine, microservices, or Kubernetes.
- No protected-result tuning without a separate approved protocol.
- No silent retention shortening to keep recording.
- No live-order automated test, ever.

## 14. Unresolved owner decisions and assumptions

Owner approval is required before implementation for:

1. **Retention and caps:** 24-hour terminal payloads; 90-day/2,048 granular receipts;
   seven-day tombstones; 30-day raw high-volume events; 400-day bars/incidents; seven-
   year idea/shadow results; 8 GiB DB; 64 MiB WAL; 14 daily/12 weekly backups; 8 GiB
   backup cap.
2. **First plugin:** Opening Leader Continuation V0 is recommended. Decide whether M1C,
   quiet state, and opening reversal are later ports, archive-only, or retired.
3. **Migration depth:** accept importing selected durable SQLite bars/quotes/outputs/
   shadows while raw Parquet and unmapped detail stay in the immutable archive.
4. **Plugin trust:** accept reviewed in-process plugins with import tests and operational
   isolation, acknowledging there is no malicious-code sandbox.
5. **Shadow convention:** approve first-causal-quote entry, conservative bid/ask,
   versioned costs, at most eight horizons, and 30-day maximum.
6. **IBKR market-data environment:** choose account/environment and verify entitlements
   plus Read-Only API externally. This never authorizes paper execution.
7. **Cutover:** approve date, downtime, rollback window, and new-run/explicit-gap rule
   after any V2 callback admission.
8. **Legacy deletion timing:** delete superseded active source only after rollback window,
   retaining Git history and the recovery set.
9. **Web exposure:** retain loopback/SSH as recommended or approve authenticated access
   behind a same-host TLS proxy.
10. **Hard-cap response:** approve fail-stop instead of deleting unexpired evidence.
11. **Report exports:** accept none in immediate V2; any future export needs separate design.
12. **Later execution ownership:** every portfolio, risk, intent, paper, reconciliation,
    and live phase requires separate acceptance. No account allowlist, risk limit,
    emergency action, order transmission, or live enablement is authorized here.

## Owner review gate

Stop here. Do not spawn an Implementer and do not modify production Python, SQL,
JavaScript, CSS, deployment services, or runtime configuration until the owner accepts
this plan and resolves the decisions required for the first assigned phase.
