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
SQLite in URI `mode=ro` with `PRAGMA query_only=ON`. Its evidence surface is
read-only. The only POST routes control broker-isolated replay of already
persisted evidence; they cannot reach a recorder, IBKR adapter, order API, or
broker state.

## Process responsibilities

### `stocker-recorder`

- owns the only prospective database writer and singleton recorder lease;
- owns the optional official IBKR callback loop;
- creates completed-bar, underlying-quote, connection, budget, rejection, and
  health evidence in the admitted live record-only path;
- maintains a heartbeat and recovers a stale owner only after the configured
  lease interval;
- uses monotonic request IDs, bounded callback queues, market-data line
  headroom, and a local request-rate limit;
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
thresholds. Runtime parity and completed-bar compatibility still fail closed,
but a passing installation may score live IBKR bars and begin bounded option
shadow recording immediately. The first 20 valid sessions remain
`engineering_transfer` evidence only; the later EODHD reconstruction monitors
ranking, threshold meaning, signal frequency, and episode identity without
requiring exact vendor OHLC equality. Optional capacity exhaustion degrades,
queues, reduces, or records a skip and does not stop Class 0–1 M1C streams.
Only `critical_budget_unavailable` blocks signal capture. The historical
decision remains `blocked_insufficient_low_tail_support`.

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
- exposes read-only status, current-universe, episode, quiet-state, bounded
  audit, and report GET routes, including the compact
  `GET /api/dashboard/summary`;
- exposes only two mutations:
  `POST /api/replay/start` and `POST /api/replay/stop`; these operate on a
  read-only replay worker and are mechanically excluded from broker access;
- applies host validation, rate limiting, no-store and browser security
  headers, optional environment-backed authentication, and production-safe
  error responses; and
- never imports an execution broker or runs an IBKR callback loop.

## Web operational reliability contract

### Authoritative recorder state

`/api/health`, `/api/recorder/status`, and `/api/dashboard/summary` use the
same evidence-derived recorder-state projection. Historical run, episode, or
bar rows never imply that a recorder process is alive.

| State | Meaning |
| --- | --- |
| `inactive` | No run/lease exists, or the latest runtime session is explicitly stopped, closed, or complete. |
| `waiting_for_prospective_start` | The lease heartbeat is valid and fresh, but the configured prospective start is still in the future. |
| `recording` | The run and lease agree, the timezone-aware heartbeat is within the configured stale interval, the prospective start has passed, and no runtime blocker is active. This is the only healthy/green state. |
| `stale` | A valid lease exists, but its heartbeat exceeds `recorder_lease_stale_seconds`. |
| `blocked` | A current runtime blocker is active. |
| `unknown` | Lease evidence is malformed, timezone-naive, mismatched to the run, or otherwise not safely interpretable. |

An absent, malformed, or stale lease cannot produce `recording`. The
projection also carries the evaluation time, stale threshold, heartbeat age,
latest completed-bar/capture timestamps, and blocker codes.

### Fresh-episode meaning

`fresh_episode` is true only when the latest valid episode for a symbol:

1. belongs to the configured current run, current runtime session, and symbol;
2. points to that symbol's latest completed checkpoint in that session;
3. matches the checkpoint number and triggering bar timestamp;
4. has `scientific_recording_valid = 1`; and
5. has lifecycle status `active`, `scheduled`, `streaming`, or `complete`.

An older-session, older-checkpoint, rejected, or expired episode is historical,
not fresh. The projection separately exposes `has_historical_episode`,
`latest_episode_id`, `latest_episode_session_date`, and
`latest_episode_status`.

### Polling tiers and request budget

The browser has one request coordinator and one in-flight refresh generation.
Manual refresh or screen activation aborts and supersedes the previous
generation. Duplicate automatic refreshes share the existing promise.
Automatic polling pauses while the document is hidden.

| Tier | Interval | Requests |
| --- | ---: | --- |
| Fast | 15 seconds | Only `GET /api/dashboard/summary`; no Parquet or historical tables. |
| Slow | 90 seconds | Only the visible screen's episode index, quiet-state summary, budget/capability details, or session-quality summary. |
| Manual/very slow | 5 minutes, screen activation, or explicit refresh | Only the visible screen's audit, reports, transfer, concentration, shadow-outcome, or completed quiet-state table. |

The busiest steady-state screen is
`4 + (2 × 60/90) + (2 × 60/300) = 5.733` requests/minute. Two tabs produce
at most `11.467` steady-state requests/minute, below both the 40-request
acceptance ceiling and the example 240-request rate limit. Initial navigation
can create a small bounded screen-activation burst.

The fast summary and full health route call the same compact health projection.
Bundle, parity, first-party IBKR-package, and live-parity checks use one
thread-safe 60-second TTL cache (`operational_projection_cache_seconds`).
That bounds filesystem/hash work without letting IBKR's time-sensitive
freshness check or replaced parity/blocker artifacts remain stale for the
process lifetime. Runtime, credential-presence, lease, and blocker evidence
remains live. A non-runtime safety blocker therefore cannot disappear from the
15-second view.

### Bounded evidence reads

- Quote charts project only event identity, timestamps, bid/ask, and sizes.
  Manifest predicates, Parquet timestamp filters, and row-group metadata limit
  reads to 15 minutes before the trigger through 30 minutes after entry.
  Before a scanner is constructed, the sum of physical rows in every
  overlapping row group must fit
  `parquet_projection_maximum_input_rows`; an oversized overlapping group is
  rejected even when its timestamp predicate would return only one row.
  Output is deterministically sampled to `quote_series_maximum_points`,
  preserving the first and last valid observations.
- Depth reads use the same episode window and a minimal column projection.
- The fast summary reads constant-size SQLite latest-event/active-blocker
  projections and indexed latest rows. Historical cancelled subscriptions do
  not enter the partial active-subscription index.
- Audit pages use the indexed `web_audit_projection_v0` identity table,
  opaque cursors, and a configured maximum page size. They never scan raw
  Parquet. Subscription history comes from immutable
  `subscription_lifecycle_event_v0` transitions, not a mutable lifecycle-row
  snapshot.
- Raw market-event detail is an explicit hash-addressed request. It reads only
  bounded trailing row groups and safe projected columns; paths are never
  returned.
- Daily-report metadata and archives are rejected if the date directory,
  metadata, or archive is a symlink, or if resolution escapes the configured
  report root.

### Replay lifecycle and memory bound

Replay remains in-process and read-only. Each invocation receives a UUID
execution ID and monotonic generation. States are `stopped`, `running`,
`stopping`, `completed`, and `failed`.

`stop()` first sets cancellation and publishes `stopping`, then joins for at
most `replay_stop_timeout_seconds` (2 seconds by default). A still-live worker
produces `blocked_replay_stop_timeout_worker_alive`; no restart is admitted
until that thread has actually exited. Completion from an older generation
cannot overwrite a newer generation. Application shutdown follows the same
cancel-and-join path.

Canonical replay ordering still requires materialising globally sortable
records. Replay stores bounded canonical JSON bytes rather than nested payload
objects, preflights manifest/SQLite row counts, run-scoped SQLite scalar source
bytes with a 4× allowance, and Parquet uncompressed row-group bytes, streams
all SQLite cursors and Parquet batches, and hashes the stored bytes
incrementally. Parquet batches are converted to Python one row at a time only
after charging 1,024 bytes per Arrow source byte/column unit. Before
`json.loads`, the aggregate JSON documents in that row are scanned without
allocation, capped at 64 container levels, and charged 1,024 bytes per source
character. A separate conservative JSON-output upper bound is checked before
allocating each canonical string. Replay fails before decode when any of those
envelopes would exceed
`replay_maximum_materialized_bytes`, and fails before retaining a record or
auxiliary episode identity when either that byte budget or
`replay_maximum_records` would be exceeded. Defaults are 64 MiB and 250,000
records. At the 64 MiB default this admits at most 64 Ki source units to any
one Python/JSON expansion, while the largest transient source input and
retained encoded-record collection remain independently bounded by 64 MiB.
Parquet batch size is at most 1,024 rows, but conversion is row-wise. Safety
fields remain `ibkr_connections_attempted = 0` and
`broker_state_mutated = false`.

### Operational logging and request IDs

Every response carries `X-Request-ID`. Request-completion JSON logs contain
the request ID, method, route template, status, elapsed milliseconds, run ID,
aggregate SQLite operation count/duration, aggregate Parquet file/row-group
and input/output row counts, and the current replay execution ID. Unexpected
exceptions retain the safe `{"detail":"internal_error"}` response and log the
exception class with a server-side stack trace.

Logs deliberately exclude request headers, cookies, authentication material,
credentials, configuration bodies, SQL text, filesystem evidence paths, and
raw market payloads.

### Measured synthetic acceptance evidence

On 2026-07-30, CPython 3.12/TestClient against temporary synthetic SQLite data
measured a median compact-summary latency of 7.846 ms over 12 warmed requests.
After adding 5,000 synthetic raw-manifest identities, 5,000 cancelled
subscription rows, 5,000 resolved data-health events, 1,000 checkpoints, 1,000
episodes, and 1,000 informational IBKR events, the same measurement was
7.896 ms. Actual query-plan assertions prove that global latest checkpoint,
episode, and connection queries use matching indexes; the indexed latest-event
lookup remained below 50 SQLite VM progress steps. This is a regression
measurement, not a production capacity claim.

The quote-window test used 1,000 synthetic rows in 10 row groups. It examined
metadata for 10 groups, read the one overlapping group (100 physical input
rows), and returned 10 deterministic points with both endpoints preserved. A
second test rejects a 1,000-row overlapping group before constructing a
scanner when the physical-row budget is 100. Replay likewise rejects a
Parquet file whose uncompressed metadata exceeds its byte limit before calling
`iter_batches`, rejects a row before `to_pylist` when its Python expansion
allowance is too large, and rejects oversized, structurally dense, or
over-64-level SQLite JSON before M1C verification or JSON decode. The polling
contract test measures 5.733 steady-state requests/minute. These results are
reproducible with:

```bash
uv run pytest tests/test_prospective_web.py -q -s \
  -k dashboard_summary_latency_is_stable
uv run pytest tests/test_fresh_episode_projection.py \
  tests/test_parquet_read_projection.py -q
uv run pytest tests/test_web_polling_contract.py -q
```

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
or best-later-interval fills. The future paper ledger is an intentionally empty
read-only schema view and has no submission interface.

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
The already-deployed migration ledger uses complete filenames as identities.
It contains two historical `0011_*` files and two historical `0012_*` files.
Those four filenames and their explicit historical within-prefix order remain
recognized and must not be renamed. Every new migration uses one unique,
monotonically increasing four-digit prefix. `migration_order.py` validates the
plan before application, and both CI and `scripts/check.sh` reject any new
duplicate prefix.

`0011_m1c_tail_phase_v1.sql` adds nullable M1C Tail Phase V1 checkpoint and
episode fields plus the explicit previous-close implied 15-minute movement on
Group O context. Existing rows remain readable; new recorder rows preserve the
exact phase-at-trigger values without changing the episode definition.
Migration `0016` adds the bounded audit identity projection and supporting
indexes. Migration `0017` adds the constant-size latest raw-event/gap summary
and active-runtime-blocker projections, partial active-subscription index, and
exact latest-state indexes. Fresh-from-zero and upgrade-from-`0010` schema
tests compare tables, columns, indexes, foreign keys, normalized table
constraints, foreign-key integrity, and the complete filename ledger. The
upgrade fixture hash-pins every deployed migration through `0015`, starts with
both `0011` and both `0012` filenames already in its ledger, and applies only
the new web migrations. Editing a historical migration therefore cannot make
both sides of the comparison silently agree.

Online backups use SQLite's backup API, run `quick_check`, hash the resulting
file, and write an adjacent manifest. Prospective observations and backups have
no automatic deletion policy.

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
