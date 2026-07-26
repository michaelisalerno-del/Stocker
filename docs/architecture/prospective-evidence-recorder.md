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
- creates completed-bar, underlying-quote, connection, budget, rejection, and
  health evidence in the admitted live record-only path;
- maintains a heartbeat and recovers a stale owner only after the configured
  lease interval;
- uses monotonic request IDs, bounded callback queues, market-data line
  headroom, and a local request-rate limit;
- qualifies the exact `STK` contract for every registered anchor symbol, then
  maintains at most one bounded quote and one five-second real-time-bar
  subscription per qualified symbol;
- aggregates exactly 60 distinct five-second callbacks into a completed
  five-minute diagnostic bar without filling gaps, and persists partial bars
  as rejections;
- records executable-side underlying quotes at completed-bar checkpoints,
  preserving per-field freshness and live/frozen/delayed identity;
- rebuilds subscriptions only after an official lost-data reconnect and
  records any discarded buffered callbacks explicitly;
- exposes bounded option-chain metadata, exact-contract qualification, and
  temporary-snapshot primitives without any whole-chain streaming;
- exercises entry and 5/10/15/30-minute option captures and all shadow
  accounting deterministically in replay; and
- cancels temporary market-data requests on completion, timeout, shutdown, or
  failure.

The current checkout has no approved frozen model bundle and its feature
parity gate is blocked. Consequently, deterministic replay works and the
record-only IBKR service may use the independently hash-verified registered
universe while persisting `blocked_missing_verified_frozen_bundle` and
`blocked_feature_source_semantics_mismatch`. Shadow/frozen-M1 scoring refuses
to start. Since no eligible real score can cross the gate, live
signal-triggered option scheduling is not admitted in this deployment; the
option capture scheduler and shadow path are replay evidence, not a claimed
live option recorder. The optional official `ibapi` dependency is absent from
the repository and immutable model bundle by design. A server release may
install it only from an operator-accepted official IBKR archive; startup hashes
the installed Python tree against an immutable provenance record. A weekly
read-only job checks official release metadata and can raise an update-review
blocker, but it never downloads or installs broker code.

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
