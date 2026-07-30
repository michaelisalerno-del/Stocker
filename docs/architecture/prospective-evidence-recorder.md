# Prospective Stocker evidence recorder

## Supported scientific statement

The prospective runtime preserves exactly this classification:

> Previous-close front-options context + current intraday H0 stock condition →
> improved prediction that near-term underlying movement exceeds previous-close
> option-implied movement.

This is an underlying-movement selection claim, not an options-profitability
claim.

- Structural loop-completion evidence: supported.
- Short-horizon underlying movement-selection model: validated on the
  previously untouched September–December 2025 holdout.
- Frozen positive IV-excess tail: promising, but not fully validated.
- Actual options edge after observed quotes and costs: unknown.

The holdout has been opened. Dates from `2026-01-01` onward are protected from
retrospective research. Runtime evidence may be appended only at or after the
explicit `prospective_start_utc`.

No part of this slice refits a model, chooses a threshold, tunes a quote filter,
ranks a stock, changes the universe, or lowers a gate after observing an
outcome.

## Hard machine boundary

```text
RESEARCH / DEVELOPMENT MACHINE
  frozen artifacts + registered contracts + deterministic fixtures
                     │
                     │ immutable verified bundle / signed daily context
                     ▼
DEDICATED SERVER
  stocker-recorder ──► SQLite WAL evidence database ◄── stocker-web
          │                                                 │
          ▼                                                 ▼
  local IB Gateway / TWS                         secured browser session
  market data only                              read-only API and screens
```

The server never reads a path on the research machine. Bundle installation
copies every required artifact into `/var/lib/stocker/bundles/installed/<id>`.
Activation replaces one small pointer atomically. Daily options context has its
own exact-session pointer and is reverified when loaded.

The browser never receives an IBKR client, socket configuration, signing
secret, database path, model file, or mutating control. `stocker-web` opens
SQLite in URI `mode=ro` with `PRAGMA query_only=ON`. Its OpenAPI surface contains
GET routes only.

## Process responsibilities

### `stocker-recorder`

- owns the only prospective database writer and singleton recorder lease;
- owns the optional official IBKR callback loop;
- durably admits every scientific stream callback to the WAL-backed callback
  inbox before updating a bounded in-memory cache or making the callback
  available to the poller;
- creates completed-bar, underlying-quote, connection, budget, rejection, and
  health evidence in the admitted live record-only path;
- maintains a heartbeat and can reclaim expired callback leases after the
  configured interval; a stale process-owner takeover is latched before a
  socket opens because process-local episode and subscription continuity
  cannot be proved;
- uses monotonic request IDs, a bounded durable inbox, bounded state caches,
  market-data line headroom, and a local request-rate limit;
- qualifies the exact `STK` contract for every registered anchor symbol; the
  frozen M1C mode then maintains one audited completed-five-minute update
  stream per stock plus only its required market proxies, without duplicate
  quote, trade, tick-by-tick, or depth streams;
- aggregates exactly 60 distinct five-second callbacks into a completed
  five-minute diagnostic bar without filling gaps, and persists partial bars
  as rejections;
- records executable-side underlying quotes at completed-bar checkpoints,
  preserving per-field freshness and live/frozen/delayed identity;
- rebuilds subscriptions only after an official lost-data reconnect and
  records any discarded buffered callbacks explicitly;
- exposes bounded option-chain metadata, exact-contract qualification, and
  temporary-snapshot primitives without any whole-chain streaming;
- exercises diagnostic 5/10-minute and required 15/30/60-minute option
  captures, plus session-end capture where it falls inside the bounded
  recording window, and reconstructs all shadow accounting deterministically
  in replay; and
- cancels temporary market-data requests on completion, timeout, shutdown, or
  failure.

The frozen M1C mode loads and hash-verifies the causal feature manifest,
preprocessing, coefficients, intercept, stock/checkpoint levels, and frozen
thresholds. Runtime parity, causal inputs, and completed-bar compatibility
fail closed for scientific evidence. A pending completed-bar compatibility
receipt does not suppress the engineering shadow projection: when the other
causal inputs are present, the recorder may persist the frozen score, arm
bounded Level I acquisition, and mimic option observations, while recording
`scientific_recording_not_authorized` and forcing all descendant option
evidence to non-scientific. The first 20 valid sessions remain
`engineering_transfer` evidence only; they are a source-transfer gate, not a
ban on shadow capture. The later EODHD reconstruction monitors ranking,
threshold meaning, signal frequency, and episode identity without requiring
exact vendor OHLC equality. Optional capacity exhaustion degrades, queues,
reduces, or records a skip and does not stop Class 0–1 M1C streams. Only
`critical_budget_unavailable` blocks signal capture. The historical decision
remains `blocked_insufficient_low_tail_support`.

M1C Tail Phase V1 is an additional logging-only projection at each frozen
checkpoint. It records strict stock-session `FIRST_ENTRY`, `PERSISTENT`,
`RE_ENTRY`, `OUTSIDE_TAIL`, or `UNKNOWN_INCOMPLETE` state and a stock-local
pre-trigger 15-minute range divided by the explicit previous-close
option-implied 15-minute movement. Missing checkpoints are never bridged and
incomplete denominators remain visible. The frozen 2024 consumed median is
loaded from the versioned V1 artifact. These fields do not alter M1C scoring,
fresh-episode identity, promotion, subscription priority, A1/C1/R1 actions,
option selection, capacity allocation, or order safety.

The external Group-O session-package producer must populate
`previous_close_implied_movement_15m` from the exact prior-session ATM IV using
`atm_iv * sqrt(15 / (252 * 390)) * sqrt(2 / pi)`, the same convention as
`m1c_low_movement_v0.iv_expected_absolute`. The package must retain the option
observation session and receipt hashes already carried by Group O. This field
is optional for recorder continuity: absent or invalid values produce an
auditable `UNKNOWN_INCOMPLETE` consumed bucket and never make Group O or the
M1C universe ineligible. The first transfer sessions must verify this external
producer handoff before treating prospective consumed buckets as complete.

During the first 20 `engineering_transfer` sessions, Tail Phase logging may
verify session reset, checkpoint chronology, missing-checkpoint handling,
threshold equality, prior-close denominator identity, timestamps, episode
linkage, and feed gaps. Those sessions may not optimise phase, consumption,
direction, microstructure, or option-selection rules. Optional Tail Phase
input exhaustion records `UNKNOWN_INCOMPLETE` and never stops the universe
recorder.

The official `ibapi` dependency remains absent from the repository and
immutable model bundle. A server release may install it only from an
operator-accepted official IBKR archive; startup hashes the installed Python
tree against an immutable provenance record. Because official `EClient`
inseparably contains order methods, Stocker retains it behind an explicit
market-data-only facade. Both the static contract and runtime attachment gate
reject any recorder-visible order/account surface. A weekly read-only job
checks official release metadata and can raise an update-review blocker, but
it never downloads or installs broker code.

### `stocker-web`

- reads only persisted state;
- serves the Live Monitor, Signal Detail, Shadow Blotter, and Safety + Audit
  screens;
- exposes only:
  - `GET /api/health`
  - `GET /api/runtime`
  - `GET /api/universe`
  - `GET /api/signals`
  - `GET /api/signals/{signal_id}`
  - `GET /api/shadow`
  - `GET /api/shadow/{structure_id}`
  - `GET /api/audit`
  - `GET /api/config/public`
  - `GET /api/recorder/status`
  - `GET /api/recorder/capabilities`
  - `GET /api/dashboard-snapshot`
  - `GET /api/recorder/session-reports`
  - `GET /api/market-data-budget`
  - `GET /api/source-transfer`
  - `GET /api/reports/daily`
  - `GET /api/reports/daily/{session_date}/{archive_name}`
  - `GET /api/universe/live`
  - `GET /api/episodes`
  - `GET /api/episodes/{episode_id}`
  - `GET /api/episodes/{episode_id}/microstructure`
  - `GET /api/episodes/{episode_id}/options`
  - `GET /api/shadow-outcomes`
- applies host validation, rate limiting, no-store and browser security
  headers, optional environment-backed authentication, and production-safe
  error responses; and
- never imports an execution broker or runs an IBKR callback loop.

### IB Gateway or TWS

IB Gateway/TWS is installed and authenticated independently on the dedicated
server. It remains loopback-only and in Read-Only API mode. Stocker does not
store or automate usernames, passwords, 2FA, sessions, or GUI login.

The official API review and source links checked on 2026-07-25 are in
[ibkr-official-api-review.md](../operations/ibkr-official-api-review.md).

## Frozen bundle contract

Manifest version 1 binds:

- M0 artifact;
- M1 artifact;
- preprocessor;
- ordered feature names;
- expected dtypes and missing-value policy;
- M1 threshold and provenance;
- registered 20-stock universe and identity;
- canonical ordered-symbol universe hash and source-artifact SHA-256;
- previous-session context schema and ordered-feature hashes;
- training, reference, holdout, and protected intervals;
- code/feature contract version;
- audit and determinism references; and
- a SHA-256 for every file.

The builder refuses database/data-set suffixes, missing artifacts, an altered
20-stock membership, a different threshold provenance, and an existing output
directory. Verification requires the manifest identity set and file map to
match exactly, rejects unlisted files, non-canonical paths, traversal and
symlinks, and checks both size and SHA-256. Installed bundles are made
read-only and never overwritten. Activation is a compare-and-swap operation
with operator identity and an append-only action log. Every score attempt
reloads and reverifies the active bundle.

The audited V0.1 research run did not write serialized estimator objects. The
deployment reconstruction seam therefore verifies the pre-outcome freeze
hashes and safety flags, materializes no-fit scorers from the frozen medians,
means, scales, category levels, coefficients, and intercepts, and emits a
machine-readable record with zero fit invocations and zero protected
observations read. The reconstructed M0 and M1 probabilities are pinned by
independent deterministic fixtures before bundle construction. This removes an
artifact-format blocker only; feature-source parity and exact previous-session
context remain separate fail-closed scoring gates.

The registered anchor cohort is mechanically extracted from the actual local
M1 artifact at
`research/options-feasibility/20260724-minimal-intraday-iv-excess-holdout-v01/artifacts/primary/model_coefficients.json`
and stored in
`configs/prospective/anchor-frozen-20.json`. It contains:

`AAL, AAOI, APLD, ASTS, CIFR, HIMS, IONQ, IREN, MARA, MP, MRNA, MSTR,
NVTS, QBTS, RGTI, RIOT, RIVN, SMCI, SOFI, WULF`.

The optional `prospective_external_universe_exploratory` cohort is empty,
disabled, separately projected by the API, and never pooled with anchor
records.

## Current frozen-artifact and parity state

The audited numerical handoff is deterministically reconstructable into M0,
M1, and the ordered preprocessor under the operator's explicit no-refit
authorization. Reconstruction verifies the frozen JSON hashes and audit flags,
invokes no fit method, reads no 2026+ observation, and emits a self-contained
deployment bundle.

`configs/prospective/frozen-feature-runtime-v1.json` separately registers the
exact frozen H0 parameters/preprocessing, semantic loop dictionary,
front-options dimension parameters, and serialized four-state regime mapping.
A version-2 bundle copies and hashes all five assets. The server loader rechecks
their embedded identities and the exact implementation-source hashes. The
bundle never points back to a research-machine path.

This makes the frozen transform machinery deployable; it does **not** declare
the live inputs equivalent or authorize scoring.

The exact frozen threshold is registered from
`frozen_tail_thresholds.json` as `0.49588519865576763`; it was not reselected.

The machine-readable report at
`configs/prospective/feature-parity-m1.json` covers all 57 M1 numeric features
in exact model order:

| Status | Count | Runtime consequence |
| --- | ---: | --- |
| `incompatible` | 3 | IBKR volume cannot replace the EODHD historical activity proxy |
| `missing` | 32 | Live completed-bar H0/loop feature construction is not yet authorized |
| `requires_parallel_validation` | 22 | IBKR versus EODHD bar construction is not yet verified |
| `exact` / `verified_equivalent` | 0 | No real scoring permission exists yet |

Overall real-scoring blocker:
`blocked_feature_source_semantics_mismatch`.

The server may record source-labelled IBKR bars and quotes once a verified
bundle/universe and official client are installed, but it may not label an
altered reconstruction as frozen M1.

The append-only EODHD parallel path retrieves the latest due XNYS session after
a fixed two-hour delay, records every returned five-minute bar as
`parallel_validation_only`, and never exposes those rows to the scorer. The
pre-observation gate is
`configs/prospective/parallel-feature-validation-v1.json`: 20 complete
sessions, all 20 anchor symbols, no outcome fields, no automatic promotion, and
an independent signed parity audit before any future run can use a revised
parity report. EODHD is an API request source, not a daemon that must remain
running.

## Dynamic previous-session context

This slice implements signed daily context import (Mode B):

1. A complete package declares the observation session, schema and feature
   hashes, provider/source-record identities, creation time, symbol features,
   and key ID.
2. HMAC-SHA256 authenticates it; an independent SHA-256 identifies its exact
   content.
3. Import validates the observation against the exact previous XNYS session
   using the US exchange calendar.
4. The package is copied into the server context store.
5. An atomic pointer maps one exact current session to one package.
6. Loading never scans for the newest or nearest file.

Same-day, future, old/stale, incomplete, mismatched-schema, mismatched-feature,
or incorrectly signed context fails with a visible blocker. The HMAC secret
exists only in the server environment file.

## Signal and quote semantics

Only eligible completed five-minute scores participate in eventisation.

- First score above the threshold after process/session startup:
  `startup_above_threshold`; no new episode.
- Previous eligible score below and current eligible score at/above:
  one crossing episode.
- Continued eligible scores above: checkpoints on the existing episode.
- Eligible score below: resets the next crossing.
- Restart/replay: uniqueness keys prevent duplicate scores, episodes,
  checkpoints, captures, contracts, quotes, structures, and horizons.

An eventisation row persists every decision, including
`startup_above_threshold`.

Expiry buckets use calendar-day DTE and never substitute:

- `0DTE`
- `1DTE`
- `3_TO_5_DTE`

ATM uses a fresh underlying midpoint when valid. The selector resolves an
exact strike-distance tie to the lower strike, chooses only the configured
number of adjacent strike steps, requests calls and puts, and never expands
after failure.

Only actual live market-data type may enter the primary quoted research
ledger. Frozen, delayed, and delayed-frozen observations remain diagnostic.
Missing values remain null. Missed target-time captures remain missed.

## Shadow accounting boundary

The quoted research ledger contains:

1. long ATM call;
2. long ATM put;
3. long ATM straddle;
4. one-step call debit spread; and
5. one-step put debit spread.

Long legs enter at ask and exit at bid. Short spread legs enter at bid and exit
at ask. Returns and gross P&L use the exact contract multiplier; estimated fees
remain separate. Missing legs, stale or non-live quotes, crossed markets,
nonpositive debits, and excessive capture lag reject the valuation.

There are no midpoint fills, last-price fills, model-price fills, paper fills,
or best-later-interval fills.

## Virtual position evidence ledgers

The read-only dashboard projects two deliberately separate virtual ledgers
from immutable recorder evidence. Neither projection creates a fill, order,
account balance, buying power, margin, assignment, or broker position.

The opening-reversal ledger admits only an exact capture-eligible M1C V1.1
prediction receipt and its causal-barrier, activation, promotion, and strict
primary-pair evidence. It projects quantity one of the predicted 1DTE leg.
Engineering-shadow rows are labelled `scientific_eligible=false`; the
separate scientific-eligible view additionally requires a scientifically
valid parent episode. Thus the recorder can mimic the frozen protocol during
the transfer gate without admitting those outcomes to scientific analysis.
The opposite leg remains control evidence. A row stays `SCHEDULED` before
contract discovery and `CAPTURING` while the exact two-line call/put evidence
is incomplete. It becomes `CLOSED` only when both same-strike primary outcomes
are complete; its virtual entry is the first valid live ask and its frozen
15-minute exit is the first valid live bid at or after the horizon. Missing
or invalid evidence produces `INVALID`, never a synthetic zero or fill.

The quiet-state ledger is a different projection with a different identity.
It contains only `quiet_bottom_10` observations and the frozen short-premium
structures: ATM iron butterfly, delta iron condor, call credit spread, and put
credit spread. Long-option candidates, straddles, neutral controls, and
high-tail controls do not enter it. Opening short bids/long asks and closing
short asks/long bids remain the conservative convention. Each structure and
horizon remains a separate research outcome. Before finalization, a separate
quiet capture projection shows the bounded option plan and each contract's
latest durably persisted bid and ask. Those latest quotes are diagnostics, not
fills; a finalized outcome freezes the exact per-leg entry and exit bid/ask,
timestamps, conservative quote side, expiry, strike, DTE, and multiplier used
to calculate its virtual P&L.

The API returns these as separate, bounded collections and publishes no
combined total. The main dashboard snapshot uses smaller limits than the
on-demand ledger route so polling cannot repeatedly render the full history.
The web process obtains both with its query-only SQLite connection and cannot
receive the recorder, adapter, or any mutable database object.

## Persistence

SQLite runs in WAL mode behind repository/read-model interfaces. Migrations
live under `stocker_prospective/migrations` and create:

- prospective run and evidence envelope;
- bundle and universe state;
- runtime session and recorder lease;
- underlying contract, bar, and quote;
- previous-session options context;
- feature snapshot and model score;
- signal eventisation, episode, and checkpoint;
- option contract, surface capture, and quote;
- source-separated bid, ask, last, and model option computations;
- shadow structure, leg, and horizon valuation;
- data-health, IBKR connection, and ordered audit events.
- append-only market-data budget telemetry.

Evidence tables reference an envelope containing run ID, prospective start,
application version, Git commit, artifact identity, universe identity, cohort,
source timestamps, and recording timestamp. Inserts are append-oriented and
uniqueness-constrained.

Migration `0003` makes `option_quote_computation` the authoritative,
source-separated Greeks/IV record. The older nullable collapsed computation
columns remain only for migration compatibility and are not written by this
runtime. Migration `0004` adds symbol/permanent-contract identity to underlying
quotes. Migration application and its registry insert are one SQLite
transaction.
Migration `0005` records current active/pending/cancelling lines, request rate,
waiting/rejected signals, and reserved capacity for the separate web process.
Migration `0011` adds nullable M1C Tail Phase V1 checkpoint and episode fields
plus the explicit previous-close implied 15-minute movement on Group O
context. Existing rows remain readable; new recorder rows preserve the exact
phase-at-trigger values without changing the episode definition.
Migration `0016` adds the durable callback inbox, retry-stable lease batches,
raw-materialization and generation-fenced processing commits, request
tombstones, recorder generations and heartbeats, fatal
latches, first-class gap and operational incidents, runtime artifact
verification, and the strict V1.1 eligible-episode projection and guards. The
migration is forward-only. An application that encounters a migration version
newer than it supports fails closed without deleting or rewriting persistent
data. Migration `0017` preserves the already-published `0016` boundary while
adding typed callback-owner receipts and the explicit normal-versus-raw-only
processing disposition for crash recovery.
Migration `0018` adds the two read-only virtual-ledger views over existing
immutable V1.1 and quiet-state evidence. It adds no mutable trading or account
state and preserves the experiment boundary in the SQL predicates. Migration
`0019` forward-replaces those views for databases already on `0018`, adding
role-aware partial-pair status, exact frozen quote-timing labels, terminal
schedule handling, and fail-closed immutable quiet-leg evidence checks without
changing any source evidence row. Migration `0020` separates V1.1 capture
eligibility from scientific eligibility. It permits an engineering-shadow
episode to record only the same primary 1DTE call/put pair while the strict
scientific projection additionally requires a scientifically valid parent.

Online backups use SQLite's backup API, run `quick_check`, hash the resulting
file, and write an adjacent manifest. Prospective observations and backups have
no automatic deletion policy.

## Crash-consistent callback and raw-evidence protocol

Scientific stream callbacks do not use a destructive in-memory drain. The
official callback boundary allocates a source sequence and commits the
following original envelope to `callback_inbox_v1` under SQLite WAL with full
synchronous durability:

1. stable inbox event identity and source sequence;
2. callback kind, request ID, connection generation, textual owner, symbol,
   and the full typed stream-owner receipt when registration is available;
3. UTC and monotonic receipt times plus the original provider time when one
   exists;
4. the original JSON-safe payload, preserving null values and explicitly
   tagging non-finite malformed provider values;
5. callback classification, lease owner/generation/time, attempts, status,
   acknowledgement, failure class, and associated raw partition hashes.

The official boundary first stores a lossless provider envelope in
`provider_pending`. Every official invocation gets a new delivery identity and
source sequence; equal market values are not assumed to be duplicate
deliveries. A scientific stream row is inserted and the provider envelope is
made diagnostic in the same SQLite transaction. Control, bounded
option-parameter, contract-detail, and snapshot callbacks explicitly complete
their provider envelope after the bounded registry update. A process death
between provider admission and canonical materialisation therefore leaves
durable original evidence. The next generation quarantines that envelope as
`CALLBACK_PROVIDER_MATERIALIZATION_INTERRUPTED`, latches ingestion fatal, and
never silently resumes. Retry idempotency begins with the already admitted
event identity; it is not inferred from callback content.

IBKR may invoke a callback synchronously before its request method returns.
The original callback is admitted first; stream registration then backfills
the typed owner only on unprocessed rows matching the exact run, recorder
generation, connection generation and request ID, without rewriting an
existing receipt. A positive-request callback in the current generation is
not leaseable while that receipt remains null, so mutable ownership can never
stand in for an unfinished durable bind. A replacement generation cannot
backfill the old row even if its request and connection counters restart at
the same values. A crash before the original backfill still retains the
complete provider callback and is recovered only as scientifically blocked
raw-envelope evidence.

Cancellation cannot invalidate a callback that was already admitted as
active. Normalisation treats that callback's typed receipt as authoritative,
even if the mutable request ID has since been removed or reused. Incremental
Level I state is keyed by the full receipt rather than request ID and is
retained across bounded inbox leases only while SQLite still has
unacknowledged rows for that owner. It is rolled back on a failed processing
attempt and released after the last acknowledgement. The retired-state cache
is bounded and fails explicitly on exhaustion. Callbacks classified as
expected-late at admission remain diagnostic and never enter this path.

The poller leases a source-ordered batch without deleting it. Its stable batch
identity and membership survive lease expiry, so a callback arriving during
recovery cannot change the immutable retry batch. Owner and recorder
generation fence every transition. After normalisation, the recorder writes
immutable raw Parquet, atomically installs data and metadata, registers the
manifest and hashes, and records `callback_raw_materialization_v1`, including
the exact sorted raw event-ID set.

Once ingestion or storage uncertainty is latched, a replacement generation
never re-runs the stateful quote/bar normalizer or scientific projection for
pending callbacks. If a prior raw materialization exists, every referenced
SQLite manifest, metadata sidecar, file, and SHA-256 is verified and its stored
event identity is reused exactly. Otherwise the recorder persists one
`raw_callback_envelope_event` per inbox row with the original nullable payload,
ordering identity, classification, and ownership receipt. The processing
commit is explicitly marked `scientifically_blocked_raw_only` with
`scientific_projection_complete = 0`; it cannot be mistaken for a completed
scientific projection.

Only after the outer application has durably completed
promotion, option scheduling, episode work, reports, capability evidence and
checkpoint markers does it record `callback_processing_commit_v1` and
acknowledge the batch. A crash after the final commit but before acknowledgement
is recovered by acknowledgement alone and is reported as degraded, not
evidence loss. Retention compaction is a separate explicit operation and
affects acknowledged and completed-diagnostic rows only.

Raw Parquet and SQLite cannot share one transaction. The partition store
therefore writes a staged file and metadata sidecar, fsyncs data, performs
atomic renames, and fsyncs parent directories where supported. Startup
reconciliation:

- completes or deterministically quarantines staged files;
- registers a valid immutable partition whose manifest transaction was
  interrupted;
- verifies every registered path and SHA-256 hash;
- treats a missing or corrupt registered partition as `STORAGE_FATAL`; and
- recovers manifest-before-materialization and processing-commit-before-ack
  crashes without emitting a duplicate scientific row.

Derived scoring and episode logic run only after raw evidence is recoverable.
A normalisation poison event remains quarantined with its reason and latches
the affected run invalid. Neither reconnect nor restart clears an ingestion or
storage latch. A replacement generation may reclaim expired callback leases
and continue raw-only materialisation and acknowledgement, but a fatal latch
survives readiness refreshes and suppresses every score, promotion, episode,
and outcome projection. Interrupted streaming option episodes create
first-class unresolved scientific gaps. The original run remains blocked until
an operator completes an evidence audit; when loss cannot be disproved, the
operator starts a new run ID and retains the old run as invalid evidence.
Inbox capacity, backlog, leasing, and health are scoped to run ID, so retained
poison evidence from the invalid run cannot contaminate the new run.
The immutable activation receipt is not regenerated for that rollover. The
replacement ID is accepted only when substituting a historical
`prospective_run.run_id` reconstructs the receipt's exact configuration hash;
all scientific configuration and artifact checks remain unchanged. This
database-backed proof prevents an arbitrary run-ID change from bypassing the
activation boundary.

## Callback containment and request generations

Every official IBKR callback entry point is a true exception boundary. Queue
pressure, malformed values, unknown IDs, late callbacks, cache failures, and
durable-store failures are converted to stable operational incidents; none is
re-raised into IBKR's callback loop. Best-effort incident evidence includes
the callback/request identity, source sequence when allocated, connection
generation, owner/symbol when known, exception class, stable code, and whether
loss is possible.

Recently cancelled or replaced request IDs remain in a bounded, expiring
tombstone table. A callback is classified as active, expected-late,
previous-generation, duplicate, unknown, or after a data-loss latch. Expected
late and previous-generation callbacks are diagnostic and cannot mutate the
active subscription cache. Unknown callbacks, durable inbox overflow, and any
possibly lost callback immediately latch ingestion fatal and disable scoring.

## One authoritative operational projection

`recorder_operational_state_v1` plus its generation, lease, incidents, gaps,
and artifact evidence is the only source for `/api/health`,
`/api/recorder/status`, the dashboard, session reports, and operational
alerts. The explicit states are:

| State | Meaning |
| --- | --- |
| `INACTIVE` | No current recorder generation exists. |
| `STARTING` | A generation owns startup but has not satisfied health gates. |
| `WAITING_FOR_PROSPECTIVE_START` | The lease is fresh but the preregistered start has not arrived. |
| `MARKET_CLOSED` | The calendar says callbacks are not expected; quiet is not a failure. |
| `RECORDING_HEALTHY` | Every lease, heartbeat, storage, inbox, broker-mode, artifact, prerequisite, and required-gap condition passes. |
| `RECORDING_DEGRADED` | Recording continues but a nonfatal live condition is stale or unexpected. |
| `RECONNECTING` | The expected IBKR connection is being rebuilt. |
| `STALE_HEARTBEAT` | The current lease or process heartbeat is stale. |
| `INGESTION_FATAL` | Callback evidence may have been lost or poisoned. |
| `STORAGE_FATAL` | Immutable raw evidence or its manifest is inconsistent. |
| `SCIENTIFICALLY_BLOCKED` | Prerequisites, artifacts, market-data mode, broker mutation, or a required gap forbids scoring. |
| `STOPPING` | The owning generation is shutting down. |
| `STOPPED_CLEANLY` | That generation completed a clean bounded shutdown. |

The projection stores process, callback-received, durable-admission,
raw-partition, inbox-acknowledgement, completed-five-minute-bar, and checkpoint
timestamps separately. `RECORDING_HEALTHY` requires a fresh current-generation
lease, fresh expected heartbeats, a bounded/young inbox backlog, the expected
connection and observed market-data mode, exact runtime artifact verification,
valid scientific prerequisites, no fatal latch, no broker-state mutation, and
no unresolved required-stream gap. A historical run row without a live lease
can never produce a recording label.

## Gap, replay, dashboard, and verification projections

A discontinuity is one `gap_incident_v1` row with a stable identity, affected
stream/sequence range, cause, severity, recoverability, optional backfill
result, affected episodes, and resolution evidence. Dashboard counts come
from those incident identities: active gaps, resolved recoverable gaps,
unresolved scientific gaps, connection interruptions, and optional-feed
degradations. Legacy per-partition `gap_count` remains readable but is never
summed into the new operational total.

Every replay start owns a new operation generation and its own stop event.
Only the active generation may publish counters, digest, completion, or error.
Stop moves to `STOPPING`, signals that generation, and joins for a bound. A
worker that ignores the bound produces an explicit failed-stop result and
blocks a new start while it can still mutate state. Replays remain
broker-isolated.

The main browser poll is one read-only `GET /api/dashboard-snapshot` request
inside one SQLite read transaction. Optional section failures carry stable
section error codes while successful sections remain visible. The browser
prevents overlap, aborts obsolete requests, pauses while hidden, and fetches
episode detail only on selection or explicit refresh. It renders values with
text nodes, preserves CSP/authentication/no-store controls, and distinguishes
unavailable data from a stale last-successful snapshot.

Artifact verification is persisted by the recorder generation that actually
found, loaded, schema-validated, hash-verified, contract-checked, and used each
artifact. A configured path is not evidence. The web process displays the
persisted bundle/artifact IDs, expected and observed hashes, feature-contract
version, activation receipt, load time, generation, result, and blocker.
No-order reporting is likewise derived from named checks for risk disablement,
web isolation/read-only access, absent HTTP/adapter/runtime order surfaces,
loopback/read-only configuration, and a zero broker mutation count. It clearly
separates local enforcement from what cannot be externally verified about the
IBKR environment.

## M1C V1.1 experiment segregation

The V1.1 eligible view and insert/update guards bind the opening-reversal
experiment ID, activation receipt, causal-barrier audit, eligible episode,
primary 1DTE expiry, one call and one put, the same nearest valid common
strike, exactly two subscription lines, and their bid/ask outcomes. Quiet-state
short-premium, condor roles, secondary expiries, generic/high-tail option
episodes, and mismatched experiment identities cannot enter that projection.
These guards add no new rule and change no frozen coefficient, threshold,
checkpoint, cohort member, timestamp, or causal barrier.

## Explicitly absent

This phase has no:

- order endpoint, command, method, submission, cancellation, or global cancel;
- paper/live position, account, exercise, assignment, or routing logic;
- trading-enable UI or threshold editor;
- model upload;
- broker credential field;
- daily stock regime/context;
- route competition or route-state feature;
- hand-built mismatch feature;
- new hidden-loop feature; or
- direction inferred from option returns.
