# Stocker V2 market-open reliability remediation

Status: **Accepted for implementation on 2026-08-16.**

Architect: independent project Architect role, read-only, against branch
`codex/stocker-v2-platform-redesign` at
`37715c4978a8000a052b325c4e1eb65acc5ffd12`.

This plan is limited to the prospective/shadow market-data recorder and the read-only
web application. It changes market-data ingestion, operational lifecycle/storage
metadata, derived/shadow failure containment, and read-only health reporting. It does
not add or alter risk, order intents, execution, reconciliation, credentials, account
selection, account allowlists, positions, paper trading, or live trading.

## 1. Current-state findings

The audit was performed against actual current HEAD rather than an older reviewed
commit.

1. **Same-run restart — confirmed defect.** A clean stop closes the recorder
   generation and also sets `runs.status='stopped'`; startup rejects the same run.
   Generation sequencing, callback identity, durable-before-return admission, and
   callback provenance already provide the right foundations.
2. **Fatal-state scope — confirmed defect.** Recorder startup, writer verification,
   and callback admission contain database-wide `runs.status='fatal'` predicates.
3. **Crash recovery — partly implemented.** Generation fencing and auditable stale
   takeover exist, but the default 60-second database heartbeat lease is longer than
   systemd's ordinary 15-second retry.
4. **Database preservation — sound foundation, change required.** The schema-15
   migration system verifies checksums, structure, foreign keys, and `quick_check` and
   applies migrations atomically. A forward schema-16 migration is required. V1 must
   remain rejected and untouched.
5. **Per-feed recovery — partly implemented.** Dynamic subscriptions have bounded
   independent retries. Static/base subscriptions marked stale or rejected do not
   recover while the socket remains connected.
6. **Continue after one failure — confirmed defect for required feeds.** A required
   startup failure aborts later attempts, disconnects successful feeds, and marks the
   socket disconnected. Optional/dynamic failure isolation already exists.
7. **Production inputs — partly implemented.** Duplicate identities and missing
   instrument references are checked, but empty instruments/subscriptions and a set
   with no required subscription are accepted. Deployment preflight validates only
   recorder JSON, and the tracked market-data example is empty.
8. **Downstream isolation — partly implemented.** Individual plugin and shadow
   position errors are internally isolated, and web is a separate read-only process.
   Canonical projection, option snapshot projection, the whole idea runner, dynamic
   reconciliation, the whole shadow pass, backup status, and most retention failures
   still share recorder-wide fatal handling.
9. **Readiness — missing.** `/` is process/static liveness and the seven JSON routes
   have no machine readiness endpoint.
10. **Run selection — confirmed defect.** Unpinned web selection uses only the newest
    `runs.started_at_us`.
11. **Query budget — partly correct.** Reads are query-only and timeout through a
    progress handler with a stable 503, but both default and maximum are 100 ms.
12. **Opening replay — partly present.** A 7,200-bar correctness fixture exists, but it
    is bar-only and records no admission latency, backlog, heartbeat, busy errors, or
    memory baseline.
13. **Optimization — not justified yet.** No ingestion optimization is authorized
    unless the fixed replay fails.
14. **Failure concentration — confirmed.** The recorder module combines lifecycle,
    subscription supervision, and downstream exception policy.
15. **Adapter reflection — confirmed.** Recorder safety scans public method-name
    substrings despite an existing narrow adapter protocol/facade.

The XNYS/NYSE regular-session calendar is already correct. It is the only period in
which tick absence triggers staleness. Holidays, daylight-saving changes, and early
closes are calendar-driven. This plan preserves that behavior exactly and adds no
pre-market, after-hours, futures, forex, international, hard-coded UTC, or
per-instrument calendar behavior.

## 2. Target design

Use small focused modules, not a framework:

- `ingestion/lifecycle.py` owns the local writer lock, generation acquisition/closure,
  and fatal-generation recovery validation.
- `ingestion/subscription_supervisor.py` owns bounded retry policy and desired-versus-
  actual reconciliation for the one validated subscription set.
- `market_session.py` exposes the existing XNYS regular-session calculation to both
  recorder health and web readiness.
- `web/readiness.py` contains pure run-selection and readiness calculations used by
  the read-only query layer.

### Writer ownership

Acquire a nonblocking kernel `flock` on a hardened regular file beside the local
SQLite database and hold its descriptor for the recorder lifetime. Persist
`ownership_protocol='local_flock_v1'` on every new generation.

After the replacement acquires the lock, it may immediately take over a prior active
generation only when that generation advertises `local_flock_v1`; the kernel then
proves the prior local owner is gone. Legacy/null-protocol generations retain the
existing heartbeat timeout during rollout. Do not shorten the lease, probe PIDs, hold
a process-long SQLite transaction, or add a distributed lease.

A clean stop closes subscriptions and the current generation only. It leaves the run
lineage resumable (`runs.status='running'`, no lineage end time, runtime lifecycle
`stopped`). Because schema 15 had no distinct finalisation state, its cleanly stopped
lineages are also resumable with the exact run/mode/config identity; the first schema-16
generation binds the separately validated market-data input hash. No finalisation
workflow is added.

### Fatal recovery

Fatal checks are scoped to the affected run and generation. A narrow offline command,
`stocker-runtime recorder recover-fatal-generation`, acquires the same writer lock and
requires the exact run, generation, mode/config/input identity, and fatal code. It
verifies schema, foreign keys, `quick_check`, database writability, capacity, and the
absence of an active owner. It records an authorization incident/metadata without
deleting or resolving the original fatal evidence.

Eligibility is a small code allowlist for legacy operational failures made recoverable
by this release. Corruption, incompatible identity, ownership loss/competition,
callback ordering/provenance loss, unsafe broker capability, unknown codes, inbox
overflow, hard storage cap, and unresolved durable-admission failure remain blocked.

### Subscription supervision

Use the `subscriptions` rows created from the validated configuration as the sole
desired/actual representation. Base and dynamic requests receive fresh fenced request
incarnations for retry. Cancel/tombstone before replacement, persist exponential
backoff, never disturb healthy subscriptions, and suppress request retries while a
matching farm outage remains open. Streaming incidents resolve only after a valid
callback is durably projected; a snapshot resolves only on valid completion evidence.

## 3. Mode and trust boundaries

- Affected modes: `prospective_record`, `shadow`, and read-only web presentation.
- Affected systems: market data, recorder lifecycle/operational storage metadata,
  derived/shadow projection availability, retention/backup visibility, and readiness.
- Not affected: risk, execution, reconciliation, broker orders, fills, broker
  positions, account data, credentials, paper trading, or live trading.
- Automated validation uses fake, replay, or simulation adapters only and never sends
  a broker order.
- IBKR remains the sole active prospective source. EODHD, cross-vendor parity,
  provider-equality gates, and source-transfer logic do not return.
- The adapter advertises one explicit capability set. Recorder accepts exactly
  market-data-only capability and rejects an order-capable declaration before connect.
  The brittle name-substring scan is removed rather than duplicated. This is an
  explicit architecture boundary, not a malicious-Python-object sandbox claim.

## 4. Schema and API impacts

Add `0016_market_open_reliability.sql` while preserving all existing evidence.

Minimal generation fields:

- `ownership_protocol` (nullable for legacy rows);
- generation `git_commit`;
- canonical caller-owned market-data `input_hash` (nullable for legacy rows and bound
  on their first schema-16 restart);
- fatal-recovery authorization timestamp, operator/reason, and recovered fatal code.

Minimal subscription fields:

- validated `stale_after_us`;
- `retry_count`, `next_retry_at_us`, and `last_attempt_at_us`;
- `last_error_code` and `permanent_failure`.

New rows must populate these fields from the validated configuration. Legacy rows are
preserved and may have unknown staleness/retry metadata until a new generation is
created. No second expected-subscription table or manual expected count is added.

Add `GET /api/v2/ready`. It returns 200 only when the selected current operational run
is ready and otherwise 503. Its bounded response includes:

- selected run and explicit selection reason;
- recorder generation/lifecycle and process heartbeat;
- `XNYS` session state, clearly inside or outside expected regular hours;
- socket, admission/storage, and durable-inbox state;
- explicit readiness reason codes;
- each desired feed's identity, required/optional status, active/stale/retrying/
  permanently-rejected state, latest callback, unresolved incident, and retry state.

Outside regular hours, callback age alone cannot fail readiness. Database readability,
writer/admission health, configuration, socket, and subscription lifecycle remain
visible. No secret, credential, client account, or broker account identifier is
returned.

Set the web query default/example to 250 ms, the smallest requested value, and the hard
maximum to 500 ms. Preserve progress-handler interruption, SQLite busy bounds,
query-only mode, and the stable `query_timeout` 503. Increase the default within the
250–500 ms range only if the unchanged opening replay/query tests prove 250 ms
unreliable.

## 5. Migration and rollback

1. Stop recorder and web.
2. Create an integrity-checked compressed schema-15 backup and restore-check it to a
   new path.
3. Apply schema 16 atomically and verify migration hashes, schema structure, foreign
   keys, and `quick_check`.
4. Legacy generations retain null ownership protocol and therefore lease-timeout
   takeover behavior.
5. Run combined recorder and market-data-input preflight.
6. Restart with the same run ID and compatible frozen identity.

Before schema-16 admission, rollback requires the matching old release and checked
schema-15 backup. After a schema-16 generation admits callbacks, do not run old code;
preserve a checked backup and roll forward. Never modify, adopt, or migrate a V1
operational database in place.

## 6. Failure modes

Fail closed for duplicate writer, ownership loss, schema/integrity failure,
incompatible run/input identity, unsafe adapter capability, inability to durably admit
raw callbacks, callback ordering/provenance corruption, and unresolved hard storage
limits.

Degrade, persist an incident, and retry with bounded component-specific backoff for an
individual subscription, plugin, option projection/discovery, shadow pass, transient
retention/backup problem, and web query timeout. A canonical projection failure keeps
the durable raw inbox row and retries; a confirmed identity collision/corruption or
database unwritability still fails closed. No new global latch is introduced.

## 7. Serial implementation phases

### Phase 0 — Plan and baseline

Save this accepted plan. Define the replay below before measurement. When practical,
run it against an untouched `37715c4` worktree and retain a small textual baseline;
do not commit generated databases or raw evidence.

### Phase 1 — Lifecycle, schema, and writer recovery

Add schema 16, the hardened lifetime lock, resumable clean generations, scoped fatal
authority, explicit fatal recovery, and exact input hash binding. Extract lifecycle
code and delete superseded recorder logic. Run focused lifecycle, provenance,
ownership, migration, and fatal tests, then commit the bounded phase.

### Phase 2 — Input and subscription resilience

Add combined production preflight, a nonempty unmistakably illustrative market-data
example, explicit adapter capability, independent startup attempts, and connected-
socket subscription supervision. Preserve farm ordering and dynamic reconciliation.
Run focused input, retry, XNYS, reconnect, duplicate-request, and no-order tests, then
commit.

### Phase 3 — Downstream isolation and truthful web health

Add component-specific retry/incident boundaries, shared XNYS session calculation,
current-run selection, readiness, per-feed diagnostics, and the 250 ms query budget.
Keep `/` as liveness. Run failure-injection and web tests, then commit.

### Phase 4 — Opening replay, justified optimization, docs/deployment

Add the deterministic CI replay and a bounded larger benchmark command. Measure the
untouched baseline when practical, then measure final code with unchanged thresholds.
Optimize only a demonstrated bottleneck, using the smallest local change and
preserving durable-before-return admission, bounded memory, single-writer semantics,
and deterministic ordering. Update deployment/runbook documentation and remove
superseded logic. Run failure-oriented tests, full tests/checks, final review, and
resolve blocking findings before the final commit.

## 8. Acceptance tests and fixed opening replay

The larger replay begins at the XNYS open on 2026-08-10 and uses the current 100-line
hard cap:

- 21 five-second bar feeds at one callback per five seconds;
- 40 quote feeds at three callbacks per second;
- 39 trade feeds at two callbacks per second;
- 60 seconds and exactly 12,132 callbacks;
- deterministic ordering by `(received_at_us, desired identity, ordinal)`;
- drain batches of at most 256.

The normal CI correctness replay uses the first ten seconds: exactly 2,022 callbacks.
Dedicated tests separately cover required-versus-optional readiness and failure
semantics; the load case keeps all 100 feeds required to make per-feed freshness the
conservative case.

Thresholds are frozen before baseline:

- presented = durably admitted; zero missing callbacks;
- zero duplicate durable callbacks caused by retry;
- zero ordering or provenance violations;
- zero SQLite busy/locked failures escaping normal handling;
- admission p50 <= 5 ms, p95 <= 15 ms, p99 <= 50 ms;
- admission throughput >= 200 callbacks/second;
- projection/drain throughput >= 500 callbacks/second;
- maximum nonterminal inbox backlog <= 5,000;
- backlog <= 256 within 15 seconds after burst and zero within 30 seconds;
- process heartbeat delay <= 5 seconds;
- every required feed stays within its configured 15-second freshness bound;
- all healthy required feeds remain subscribed;
- RSS growth <= 64 MiB where reliably measurable, otherwise report unavailable.

The 5,000-row replay/readiness threshold is 10% of the 50,000-row hard inbox cap and
represents 25 seconds at the accepted 200/s ingress. It is an early operational
degradation threshold, not a new admission latch.

### Unmodified baseline

The fixed larger replay was run before any production-code change on the local macOS
development host against `37715c4` plus this plan-only commit. The harness used a
temporary SQLite database and fake market-data adapter; neither was committed.

| Measurement | Baseline |
| --- | ---: |
| Presented / admitted / projected | 12,132 / 12,132 / 12,132 |
| Missing / duplicate / provenance violations | 0 / 0 / 0 |
| Escaped SQLite busy/locked errors | 0 |
| Admission p50 / p95 / p99 | 0.111 / 0.154 / 0.286 ms |
| Measured admission throughput | 1,327 callbacks/s |
| Maximum / final nonterminal backlog | 219 / 0 |
| Final drain time | 0.052 s |
| Effective process-heartbeat delay | 0 s |
| Required feeds fresh | 100 / 100 |
| Peak RSS growth | 16,842,752 bytes |

The temporary harness advanced the final heartbeat one simulated second beyond the
end-of-burst measurement timestamp; the reported effective delay is therefore clamped
to zero. These local fake-adapter results prove no real IBKR or production-host
performance.

Focused and failure-oriented tests cover every item in the user request, including
clean/crash restart, duplicate writer, generation provenance, historical fatal
isolation, migration from schema 15, callback ordering/duplicates/gaps, XNYS-only
staleness, socket and per-feed recovery, startup continuation, strict input validation,
component isolation/recovery, hard admission failure, per-feed readiness/run
selection/query timeout, opening replay, and absence of order capability.

After focused tests:

```text
bash scripts/test.sh
bash scripts/check.sh
```

The independent Reviewer must explicitly challenge all fifteen mandate items:
restart terminality; historical fatal poisoning; connected-socket dead feeds;
one-feed cancellation; downstream interruption of raw admission; busy-feed false
green; out-of-hours false stale; split brain; retry duplicates; migration evidence;
unnecessary state/latches; paper/live expansion; unnecessary infrastructure; empty
production universe; and post-hoc replay thresholds.

## 9. Explicit non-goals

No order transmission, paper/live activation, accounts, credentials, allowlists, risk,
positions, execution, reconciliation, EODHD, cross-vendor gates, additional database,
service, daemon, Redis, Kafka, message bus, distributed lease, workflow engine, broad
IBKR taxonomy, in-memory pre-durability queue, calendar expansion, or unrelated
strategy/research rewrite.

## 10. Resolved decisions, assumptions, and evidence limits

The user authorized the bounded scope; no additional owner design decision blocks
implementation.

- The operational database and lock file are on one local single-server filesystem.
  Shared-filesystem/cross-host writer ownership is unsupported.
- Fatal recovery uses a narrow allowlist plus proven remediation and never overrides
  corruption, ordering/provenance, ownership, incompatible identity, unsafe capability,
  or unknown fatal evidence.
- Simulation can prove deterministic callback durability, ordering, isolation,
  boundedness, and readiness behavior under the defined load. It cannot prove real
  IBKR entitlements, exchange-farm behavior, pacing behavior, or actual market-open
  host/storage performance. Those remain attended operational evidence after rollout.
