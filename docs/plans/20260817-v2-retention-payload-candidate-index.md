# V2 retention payload-candidate index

Status: accepted deployment-remediation plan, 2026-08-17

## Scope and safety boundaries

This bounded follow-up addresses a deployment-blocking retention failure discovered
after the market-open reliability release was installed against the current 3.3 GiB
V2 operational database. It affects prospective-record and shadow market-data
retention. Durable callback admission continues while the component is degraded, but
cleanup currently cannot make progress and therefore creates eventual storage-cap
risk.

The change does not affect risk, execution, reconciliation, accounts, credentials,
broker capabilities, order transmission, paper trading, or live trading. It does not
change XNYS calendar or regular-session behavior. The read-only web service is affected
only during an attended migration stop and by the readiness incident resolving after
retention recovers.

## Reproduction and measured cause

A checked deployment backup was restored and migrated in a disposable database. The
following command is the exact red-capable reproduction:

```bash
stocker-runtime retain /tmp/stocker-retention-repro.sqlite3
```

It deterministically raises `MaintenanceDeadlineExceeded` in about 1.2 seconds.
Boundary timing showed receipt verification completing in 12.8 ms and
`_compact_payloads` being interrupted at the 100 ms transaction deadline.

Both payload-candidate statements force
`callback_inbox_run_sequence_idx(run_id, source_sequence)`. On a mature run they scan
the large prefix whose payloads have already been compacted to `NULL`. On the restored
production data, the acknowledged query took 1,022.7 ms and the empty failed query
took 2,962.9 ms.

## Accepted design

Add schema migration 17 containing one partial index:

```sql
CREATE INDEX callback_inbox_payload_run_sequence_idx
    ON callback_inbox(run_id, lifecycle, source_sequence)
    WHERE payload_json IS NOT NULL;
```

Point the acknowledged and failed payload-candidate queries at that index. Preserve
all existing lifecycle, cutoff, watermark, ordering, receipt-proof, rollback,
durability, storage-cap, 2,000-row batch, and 100 ms per-transaction semantics.

Alternatives rejected:

- The existing global timestamp indexes require global range scans and sequence
  sorting as retained history grows.
- A persisted compaction cursor adds mutable state and could skip callbacks that become
  terminal later.
- Two lifecycle-specific indexes duplicate schema objects without adding safety.
- Increasing the deadline or reducing safety checks conceals the query defect.

A disposable probe of the accepted shape reduced the acknowledged lookup to 4.618 ms
and the failed lookup to 0.266 ms. A complete retention pass then succeeded, compacted
exactly 2,000 payloads, and took 111 ms total across its two separately bounded
transactions. Index construction took approximately 14 seconds on the 3.3 GiB copy.

## Serial implementation and verification

1. Add schema migration 17 and switch only the two payload-candidate index hints.
2. Add deterministic regression coverage using production-shaped tombstoned prefixes,
   query-plan assertions, and a SQLite progress/opcode budget rather than wall-clock
   timing.
3. Test schema-16 to schema-17 migration with preserved rows, exact index SQL,
   migration verification, `quick_check`, and foreign-key validation. V1 remains
   rejected and must not be modified.
4. Run focused retention, migration, callback ordering/duplicate/gap, and durable
   admission tests. Run the unchanged 2,022-callback CI replay and 12,132-callback
   opening replay to measure index-write overhead.
5. Use a separate read-only Reviewer and resolve every blocking finding. Then run
   `bash scripts/test.sh` and `bash scripts/check.sh`.

No acceptance threshold may be weakened. The production-copy reproduction must become
green under the unchanged 100 ms transaction policy, compact no more than 2,000 rows
per pass, and preserve evidence counts and receipt/watermark invariants.

## Migration, rollback, and rollout

Before migration, stop recorder and web cleanly, verify disk headroom, create and
restore-check a compressed schema-16 backup, and record integrity/hash evidence. Apply
schema 17 offline and verify migration checksums, schema structure, `quick_check`, and
foreign keys. Run one offline retention pass before restarting the same run ID.

If index creation fails, the atomic migration must leave schema 16 intact. Restore only
the checked schema-16 backup while no schema-17 callbacks have been admitted. After
schema-17 admission, do not restore an older database and lose evidence; roll forward.
The old schema-16 release rejects schema 17, so release rollback after admission also
requires a schema-17-compatible forward fix.

After restart, require a new recorder generation, fresh heartbeat, connected IBKR
socket, every exact required subscription identity reported by validated preflight
active in readiness, zero duplicate active requests, healthy durable admission,
bounded inbox/WAL, and resolution of only the retention component incident. Outside
XNYS regular hours, absent ticks remain non-stale. No broker-order or live-trading test
is authorised.

## Known failure modes and non-goals

Index creation can fail for insufficient disk or I/O; that remains fail-closed. A
different later maintenance query may still exceed the deadline and must be measured
separately. This change adds no queue, cursor, table, daemon, loop, global latch,
storage layer, feed source, calendar scope, or trading authority.

## Accepted deployment addendum: generation-scoped component incidents

The schema-17 rollout exposed a separate current-HEAD lifecycle defect. A retention
failure from recorder generation 2 remained unresolved, while generation 3 retried
retention successfully. Component incident identifiers already contain the generation,
but `_component_recovered()` checks every unresolved component incident for the run.
The historical generation-2 row therefore kept healthy generation 3 degraded and made
readiness false even though raw admission, ownership, the IBKR socket, and the complete
required subscription set remained healthy.

The accepted smallest evidence-preserving correction is schema migration 18 with one
nullable column:

```sql
ALTER TABLE incidents ADD COLUMN recorder_generation INTEGER
    CHECK(recorder_generation IS NULL OR recorder_generation >= 0);
```

New component incidents record the authoritative recorder generation. Component
recovery checks only unresolved component incidents for the current run and generation.
Schema-17 and older incident rows retain every existing value and receive `NULL`; they
remain visible historical evidence but cannot poison a schema-18 generation. Do not
resolve, backfill, rewrite, or delete historical incidents at generation start. Keep the
generation-bearing incident-ID formula. Do not duplicate generation in `details_json`,
add an index without measured need, or generalise non-component incident producers.

Rejected alternatives were resolving old incidents at startup, resolving prior episodes
after a later generation succeeds, encoding the lifecycle relationship only in JSON,
and relying on process-local failure dictionaries. Each would either falsify evidence or
lose durable lifecycle scope across a restart.

Acceptance requires schema-17-to-18 evidence preservation and integrity checks; exact
generation provenance on new component incidents; repeated-failure deduplication; a
generation-2 failure remaining visible while generation 3 recovers to `running`; legacy
`NULL` incidents not blocking current recovery; another unresolved incident in the same
generation continuing to block recovery; contention deferring recovery; raw admission
continuing under degradation; and hard durable-admission/storage failures remaining
fail-closed.

Deploy this addendum serially. Stop recorder and web, create and restore-check a fresh
schema-17 backup, migrate offline, verify migration ledger/schema/`quick_check`/foreign
keys, and restart the same run as a new recorder generation. Require a fresh heartbeat,
a connected market-data socket, every exact required subscription identity reported by
validated preflight active, healthy durable admission, and readiness for the new
generation. Before schema-18 admission the checked schema-17 backup and matching release
may be restored; after schema-18 callbacks are admitted, roll forward to avoid evidence
loss. Do not manually edit or resolve the historical incident.

This addendum affects prospective-record and shadow component health, incident
provenance, and read-only readiness. It does not affect risk, execution, reconciliation,
orders, fills, positions, accounts, credentials, paper/live trading, broker capability,
subscription behavior, or XNYS regular-session semantics. No live order test is
authorised.

## Accepted deployment addendum: bounded per-feed readiness lookup

After schema 18 restored truthful generation health, production readiness still timed
out after approximately 3.88 seconds on about 1.5 million callbacks. The current query
performs one correlated `max(received_at_us)` lookup for every desired subscription, but
the best available callback index begins with only `(run_id, source_sequence)`. Each of
the 41 lookups can therefore scan a large run prefix; a missing feed is the worst case.
The 224 ms opening replay used a small temporary database and did not prove
mature-history lookup behavior.

The accepted schema-19 correction is one partial covering index:

```sql
CREATE INDEX callback_inbox_readiness_latest_idx
    ON callback_inbox(
        run_id,
        recorder_generation,
        connection_generation,
        request_id,
        received_at_us DESC
    )
    WHERE lifecycle = 'acknowledged';
```

Replace the aggregate subquery with an explicitly indexed `ORDER BY received_at_us DESC
LIMIT 1` seek for the exact run, recorder generation, connection generation, and request
identity. Keep the correlated shape: 41–100 bounded B-tree seeks are preferable to a
grouped scan of the active generation. Do not increase the 300 ms web query budget.

Rejected alternatives are a grouped/window scan, scanning the run/sequence index
backwards, using a derived market projection, persisting another latest-time projection
on subscriptions, using a global maximum, increasing the timeout, or adding a cache or
background service. These are unbounded, can conceal a dead feed, can diverge from raw
callback evidence, or add unnecessary state/infrastructure.

Migration and query acceptance require schema-18-to-19 evidence preservation, exact
index SQL/predicate, atomic failure, `quick_check`, zero foreign-key violations, and an
`EXPLAIN` plan that uses the new index without a temporary B-tree, including an index
miss. Fixtures must distinguish old recorder/connection generations, other requests,
and newer pending/failed callbacks. Only the latest acknowledged callback for the exact
identity may establish freshness. Work must scale with subscription count and B-tree
depth rather than callback history. Existing per-feed, busy-versus-dead-feed, timeout,
ordering, duplicate, gap, admission, generation-provenance, retention, and capability
tests remain unchanged.

The mature-copy performance threshold is frozen before the fix: ten sequential
production-shaped 41-feed readiness calculations must all finish within 300 ms and p95
must be at most 275 ms, returning the full diagnostics. Record index build time,
database/index size, before/after timings, and rerun the unchanged 2,022- and
12,132-callback opening replays without relaxing any acceptance threshold.

Roll out serially: stop recorder and web; verify space; create and restore-check a fresh
schema-18 backup; apply schema 19 offline; verify ledger, exact index SQL, integrity,
foreign keys, and query plan; run the ten-query mature acceptance measurement; then
restart the same run ID as a new generation. Require fresh heartbeat, connected socket,
every exact required identity reported by validated preflight active, healthy admission,
raw sequence growth, no duplicate active request, and `/api/v2/ready` inside 300 ms.
Outside XNYS regular hours the response must explicitly report quiet-session state and
must contain no stale-feed reason. Before schema-19 admission the checked schema-18
backup and matching release may be restored; afterward roll forward to avoid evidence
loss.

This addendum affects prospective-record and shadow callback projection index
maintenance and read-only readiness. Pending durable admission does not enter the
partial index. No callback evidence or lifecycle is rewritten. Risk, execution,
reconciliation, paper/live enablement, orders, accounts, credentials, IBKR subscription
behavior, and XNYS calendar semantics are unaffected.

## Final unchanged-threshold schema-18 replay

The post-schema-18 60-second replay passed all frozen thresholds with 12,132 presented,
admitted, durable, and projected callbacks; zero missing, duplicate, ordering,
provenance, or escaped SQLite busy/locked failures; admission p50/p95/p99 of
0.159/0.289/9.802 ms; admission/projection throughput of 414.48/1,456.55 callbacks per
second; maximum/final backlog of 219/0; zero seconds to the 256-row safe range and
0.169 seconds to drain; heartbeat delay 0 seconds; 100/100 feeds active and fresh;
readiness true in 224.24 ms; and RSS growth of 27,639,808 bytes.

## Unchanged-threshold schema-19 replay

The post-schema-19 60-second replay passed all frozen thresholds with 12,132 presented,
admitted, durable, and projected callbacks; zero missing, duplicate, ordering,
provenance, or escaped SQLite busy/locked failures; admission p50/p95/p99 of
0.166/0.372/9.866 ms; admission/projection throughput of 388.01/1,341.14 callbacks per
second; maximum/final backlog of 219/0; zero seconds to the 256-row safe range and
0.158 seconds to drain; heartbeat delay 0 seconds; 100/100 feeds active and fresh;
readiness true in 2.19 ms; and RSS growth of 20,234,240 bytes. The deterministic
2,022-callback CI replay also passed unchanged.

## Accepted deployment addendum: exclude XNYS initialization from the SQLite budget

The mature schema-19 query uses bounded covering-index seeks, but the first readiness
calculation in a new web process still failed after 5.67 seconds. Direct measurement
showed that lazy initialization of the existing XNYS calendar took 3.802 seconds; the
subsequent call took 0.027 ms. The 300 ms deadline was being started before that
calendar work, so the first SQLite progress check reported a false web-query timeout.

The accepted correction computes `market_data_expected_since_us` from one captured
timestamp before opening the bounded read-only connection and passes that result into
the pure readiness calculation. Web application construction also calls the same
calendar function once so that the existing date-keyed LRU cache is warm before the
server accepts requests. This adds no cache, state, loop, service, or hard-coded market
hours. The SQLite query budget remains 300 ms and continues to cover only connection
acquisition and SQLite work; a genuinely slow statement must still time out.

Acceptance requires tests proving calendar evaluation precedes the connection deadline,
application startup prewarms the shared calendar function, the supplied session value
is used consistently, and the existing SQLite timeout remains enforced. XNYS holiday,
DST, early-close, regular-session freshness, and outside-session quietness tests remain
unchanged. After a clean web restart, record startup duration separately and require
the first request accepted by the server plus ten mature 41-feed requests to finish
within 300 ms with the correct run, generation, feed diagnostics, and session state.

This addendum affects only read-only web startup and readiness calculation. Recorder
ownership, durable admission, retention, IBKR subscriptions, market-data evidence,
risk, execution, reconciliation, paper/live trading, orders, accounts, credentials,
and broker capabilities are unchanged.

## Mature schema-19 deployment evidence

The restored schema-18 production backup contained about 1.5 million callbacks and a
41-feed active generation. Before schema 19, the first readiness calculation failed at
2,583.129 ms and the following nine each failed at approximately 301.2 ms. After the
schema-19 migration but before the calendar correction, nine warm calculations passed
at 9.576–35.436 ms, while the cold first calculation still failed after 5,674.064 ms.
Direct isolation measured the cold XNYS initialization at 3,801.698 ms and its cached
call at 0.027 ms.

With final commit `5e82955b9a7ed420188a3a4dc1ec005759cdfe74`, web startup prewarming
took 2,795.926 ms before the application accepted requests. The first accepted
readiness calculation took 14.081 ms; ten consecutive calculations all returned the
correct run, generation, and 41 green feed diagnostics in 8.202–14.081 ms, with p95
11.903 ms and no errors. Thus every accepted request passed the frozen 300 ms maximum
and 275 ms p95 thresholds.

The real schema-19 migration framework completed against the disposable copy with no
evidence rewrite. Database size increased from 3,617,161,216 to 3,706,494,976 bytes;
the new index occupies 89,227,264 bytes. An isolated exact index rebuild took 8.76
seconds. The final database reported schema 19, `quick_check=ok`, and zero foreign-key
violations. The hit and miss plans both use the covering index without a callback-sort
temporary B-tree.

## Accepted deployment addendum: set-based payload compaction under live admission

Post-deployment monitoring reproduced a bounded-maintenance failure under actual IBKR
callback traffic. The recorder remained connected, all 41 required subscriptions
continued raw durable admission, and the web truthfully degraded, but retention
repeatedly raised `MaintenanceDeadlineExceeded`. A controlled clean stop followed by
the exact production `stocker-runtime retain` command succeeded and compacted 2,000
payloads. Twelve consecutive mature-copy passes also succeeded at 2,000 rows each.
Production still contains approximately 775,000 old acknowledged payload rows across
historical runs, so repeated live rollback would prevent the backlog from draining and
eventually threaten the 8 GiB cap.

The measured cause is the payload candidate update shape. Schema 17 made discovery a
bounded index seek, but `_compact_payloads` still materializes up to 2,000 sequences and
uses `executemany` to issue 2,000 individual updates. On the mature disposable copy,
the per-row update took 22.151–33.854 ms without live load, while one equivalent
set-based update took 12.914–13.662 ms. Under production callback I/O and CPU load, the
per-row loop intermittently crosses the unchanged 100 ms second-transaction deadline
and all 2,000 updates roll back.

The accepted correction keeps schema 19 and replaces the materialized per-row update
loop with at most two set-based statements: acknowledged candidates first, then failed
candidates only for the remaining shared budget. Each statement selects bounded
source sequences through `callback_inbox_payload_run_sequence_idx`, retains the exact
cutoff, watermark, lifecycle, proof, and source-sequence ordering predicates, and
reports its affected count. Keep the 2,000-row default, 100 ms transaction deadline,
`BEGIN IMMEDIATE`, writer precondition, progress interruption, full rollback, receipt
proof, cap behavior, checkpoint, and incremental vacuum unchanged. Delete the
superseded candidate materialization and `executemany` loop. Add no migration, cursor,
state, adaptive batch, calendar deferral, queue, service, or deadline increase.

The deterministic regression uses 2,000 eligible watermarked acknowledged payloads and
a traced connection that advances a fake clock for every payload-nulling update
statement. The old 2,000-statement loop must exceed 100 ms and roll back; the set-based
implementation must execute one acknowledged update (and at most one failed update),
compact exactly the shared bound, and preserve recent, pending, unwatermarked, and
non-payload evidence. Also prove acknowledged-first failed-fill behavior, zero-
remaining failed skip, interruption rollback, exact index plans, writer loss, receipt
corruption, callback ordering/duplicates/gaps, and both fixed opening replays.

Before rollout, cleanly stop recorder and web, verify lock release, take and restore-
check a fresh schema-19 backup, and run the prior release's retention command offline
in an explicitly capped loop until one zero-work pass, repeated once. Every pass must
exit zero and compact no more than 2,000 rows. Preserve callback/receipt/watermark
counts and hashes, require `quick_check=ok` and zero foreign-key violations, then run
one offline pass with the set-based release. Binary rollback keeps the current schema-
19 database; after new callback admission, never restore the pre-drain backup.

Live acceptance requires at least three scheduled maintenance opportunities under
actual callback traffic with no new current-generation retention incident, a fresh
heartbeat, growing raw source sequence, healthy writer admission, connected socket,
all exact configured required identities active without duplicates, bounded inbox/WAL,
and truthful HTTP 200 readiness when all conditions hold. Outside XNYS regular hours,
tick quietness remains non-stale. This does not prove real market-open host I/O; retain
the attended regular-session observation requirement.

This addendum affects only prospective-record/shadow market-data retention and
readiness availability. It does not affect risk, execution, reconciliation, accounts,
credentials, broker/order capability, paper/live trading, subscription semantics, or
XNYS calendar behavior.

### Accepted operational amendment after prior-release drain failure

The attended production procedure proved that the planned prior-release drain was not
dependable enough to complete. With recorder and web stopped, the schema-19 writer lock
available, and the checked pre-drain backup already restored successfully, the
per-row release completed 23 batches and a second bounded segment completed 21 batches,
all at exactly 2,000 payloads. One intervening manual pass also completed 2,000. The
first remote session then closed and the detached segment exited before producing its
next valid pass record. A manual retry succeeded, confirming rollback/evidence safety,
but the requirement that every prior-release pass succeed was not met. The 45 valid
batches remain legitimate policy compaction and reduced non-null payloads from 651,498
to 561,498 without callback admission.

The Architect therefore superseded only the prior-release drain ordering: do not keep
retrying the measured per-row implementation. Install the reviewed set-based candidate
offline while both services remain stopped and runtime-masked, then verify release
artifacts, configuration, and schema 19. Use the purpose-built command:

```bash
stocker-runtime recorder drain-payloads \
  --database /var/lib/stocker/v2/stocker-v2.sqlite3 \
  --max-passes 1000 \
  --max-wall-seconds 2700
```

The command acquires the canonical `LocalWriterLock` once for its full loop, fixes one
cutoff timestamp, verifies the lock in every transaction, inherits the 2,000-row and
100 ms limits, and succeeds only after two consecutive zero-compaction passes. It is
payload-only: it cannot roll/delete receipts, advance watermarks, terminalize shadow
positions, prune any table, vacuum, or mutate runtime/incident/cap state. Runtime
masking keeps the recorder start-prevented during the surrounding backup, verification,
and restart preparation. Any internally reported lock conflict, deadline, pass/time
cap, invalid result, or other error exits nonzero with its completed-pass count and
committed payload total, without claiming that earlier committed batches were undone.
An outer process timeout may prevent JSON emission, so operators must reconcile its
committed total from the recorded pre/post payload counts.

This amendment is supported by two independent candidate measurements. The original
mature copy completed 12 consecutive 2,000-row passes in 0.783–1.198 seconds. A fresh
restore of the production pre-drain backup completed another 12 in 0.745–1.295 seconds.
Across the fresh-copy run, callback row count and the non-payload callback hash,
receipt count/hash, and watermark count/hash were identical; only 24,000 eligible
payloads became null. Both before and after reported schema 19, `quick_check=ok`, and
zero foreign-key violations.

There is no migration rollback. An ordinary candidate failure with intact evidence
keeps the current database and permits binary rollback to the matching prior schema-19
release, but the flaky drain must not then be resumed as a dependable recovery path.
Actual corruption detected before services resume permits restoring the already checked
pre-drain schema-19 backup because no callback admission occurred after it. Once a new
callback is admitted, never restore that backup; preserve evidence and roll forward.

## Accepted deployment addendum: isolate the remaining normal-maintenance deadline

Deployment of `aee5693ae3010c191546a8e984963b9e927aa001` proved the payload-only
recovery path: 515,606 payloads were compacted in 260 passes over 14.561 seconds,
ending with two zero-work passes. Together with the 90,000 earlier policy-equivalent
compactions, non-null payloads fell exactly from 651,498 to 45,892. Callback count and
immutable hash, receipt count/hash, watermark count/hash, schema 19, `quick_check=ok`,
and zero foreign-key violations all remained exact.

Recorder generation 7 then connected with all 41 identities active and continued raw
sequence growth, but normal `RetentionManager.run` entered repeated
`MaintenanceDeadlineExceeded` episodes. One episode recovered after five failures;
the next reached seven failures and kept truthful readiness at HTTP 503. Database size
was 3.731 GB against the existing 8 GiB cap, WAL about 5 MB against 64 MiB, and the
nonterminal inbox remained below 70 against 50,000. The Architect accepted continued,
time-boxed degraded operation while durable admission, ownership, integrity, heartbeat,
socket, subscriptions, and hard limits remain healthy. Stop cleanly if any existing
hard boundary is threatened; do not conceal the incident or call the service ready.

The next phase must measure before changing behavior. On a fresh disposable copy of
the post-drain schema-19 database, identify and time checkpoint selection,
`_roll_receipts`, transaction-A commit, pending terminalization, payload compaction,
each named prune operation, transaction-B commit, passive checkpoint, and incremental
vacuum across at least 12 normal-retention passes. Add controlled fake callback-write
or CPU pressure. If the copy cannot reproduce, add only bounded phase labels to
`MaintenanceDeadlineExceeded` and its current-generation incident; add no schema,
state, loop, or payload detail.

Make exactly one correction justified by that evidence. A bad query may receive only
its required query/index correction. Per-row overhead may receive the same local
set-based treatment. Split a transaction only if every component is independently
within 100 ms but their measured cumulative work crosses the bound; retain the shared
2,000-row budget, authority checks before commit, atomic rollback within each phase,
and exact partial-commit accounting. Do not raise the 100 ms deadline, lower capacity
below measured ingress, add adaptive policy speculatively, calendar-gate an unresolved
failure, suppress incidents, or rerun the payload-only drain.

The deterministic regression must advance fake monotonic time at the measured
statement/work boundary, be red on the current implementation, and pass without
relaxing deadline or capacity. Reprove rollback, authority, receipt/watermark chains,
corrupt-proof fatal behavior, recoverable DB-lock isolation, raw admission continuity,
payload-drain safety, callback ordering/duplicates/gaps, incident recovery, and the
fixed 2,022/12,132 opening replays. Validate at least 12 consecutive full maintenance
passes under simulated writer pressure on the mature copy. After review and attended
rollout, require at least 12 consecutive scheduled live successes, no unresolved
current-generation retention incident, fresh heartbeat, growing raw sequence, every
configured identity active without duplicates, bounded DB/WAL/inbox, and HTTP 200
readiness. Real market-open host-I/O observation remains separate.

This addendum affects prospective/shadow market-data retention and readiness only.
Risk, execution, reconciliation, accounts, credentials, paper/live/order capability,
subscription semantics, and XNYS regular-session behavior remain unchanged.

## Accepted implementation addendum: bound receipt proof without reducing capacity

The required phase measurement on the post-drain schema-19 disposable copy identified
transaction A, not payload compaction or pruning, as the remaining deadline source.
Across 12 baseline full-retention passes, 8 exceeded the unchanged 100 ms writer
deadline and 4 succeeded. Nineteen no-work `_receipt_actions` scans consumed 55–70 ms
before the one useful `_verified_receipt_prefix` consumed another 48–70 ms. Direct
child timings kept `_prune_expired` at 15–32 ms, derivation pruning at 13–26 ms, raw
market-event pruning at no more than 1.4 ms, and completed-event pruning at no more
than 0.5 ms. The expired-payload candidate work was already exhausted by the bounded
payload-only drain.

The first measured query correction adds one read-only aggregate
`RECEIPT_WORK_RUN_SQL`. It selects exactly one lexicographically first run with real
work: no watermark, an unverified receipt suffix, a receipt straddling the watermark,
an age-expired receipt, or receipt count above the configured cap. A fully verified,
non-straddling, recent run at or below the cap does not match. An expired-payload run
has priority only when that run also has actual receipt work. The selector uses the
existing receipt sequence and watermark indexes and supplies only a run-ID hint;
`_receipt_actions` and `_verified_receipt_prefix` remain authoritative inside the
writer transaction. Direct `_roll_receipts(target_run_id=None)` fallback behavior is
unchanged.

The exact 1,200-callback proof still failed 12 of 12 mature-copy passes under the
measured production-host load when kept in one transaction. A provisional 600-callback
transaction succeeded 10 of 12 because the 34 ms aggregate selector work still ran
inside its deadline; successful proof/commit work was otherwise within the bound. A
400-callback transaction succeeded 12 of 12, establishing a safe fallback but requiring
three writer transactions. The accepted smaller design therefore performs read-only
run selection immediately before `BEGIN IMMEDIATE` under its own 100 ms progress
deadline, clears that handler, and then runs at most two authoritative 600-callback
writer transactions. Each writer deadline begins only after `BEGIN IMMEDIATE` acquires
the transaction. The recorder's process-lifetime `LocalWriterLock` remains held,
writer authority is checked after each begin and immediately before each commit, and
SQLite excludes concurrent writers while proof is validated.

The hint may become stale between selection and begin, but it cannot authorize a
mutation. The transaction rereads current receipt and watermark state and recomputes
callback hashes/counts. New same-run work may be included within 600 or deferred; new
work on another run waits only until the second selection or next pass. A stale no-work
hint commits no mutation. A newly introduced straddle or corrupt receipt still fails
closed. Reselect before transaction two. Re-select the payload-compaction run inside
transaction B rather than carrying a stale receipt hint.

Keep the existing aggregate ceilings unchanged: no more than 1,200 proof callbacks and
2,000 receipt changes per normal pass. Carry the one 2,000 change budget across both
receipt transactions; do not add an unmeasured per-transaction deletion slice. If
transaction one commits and transaction two fails, the first watermark/rollup and any
authorized receipt deletions remain valid. The second transaction rolls back, the
generation-scoped retention incident remains degraded, and the exception reports the
bounded phase plus committed transaction/rolled-row counts. Retry must resume strictly
after the authoritative committed watermark without duplicate rollup, deletion, skip,
or chain divergence. Transaction B keeps its independent existing 2,000-row budget.

Required deterministic tests cover the 19-run red seam; all exact selector classes and
priority; no-work skip; 1,200 callbacks proven across two no-larger-than-600
transactions at receipt boundaries no larger than the runtime's 256-callback creation
limit; the shared 2,000 change budget; reselection of the same and another run; writer
authority before every commit; transaction-two failure with transaction-one evidence
preserved and exact retry; selector timeout before mutation; work added to the selected
or a higher-priority run between selection and begin; a selector becoming no-work;
straddle/corruption introduced between phases; current-transaction deadline rollback;
receipt chain/corruption/skip invariants; unchanged transaction-B behavior; payload
drain safety; and component failure isolation.

The deployment gate is fixed before final measurement. On the same mature schema-19
copy under the measured production-host pressure, the exact outside-selection/two-by-
600 implementation must complete at least 12 consecutive full normal-retention passes
with both writer transactions below 100 ms, no `MaintenanceDeadlineExceeded`, exact
receipt/watermark mutations, `quick_check=ok`, and zero foreign-key violations. It must
also pass the unchanged 2,022 and 12,132 opening replays, including their admission
latency thresholds. If even one mature pass or opening threshold fails, do not deploy
this design; use the already measured three-by-400 fallback and rerun the identical
gate. Do not alter deadline, cadence, capacity, schema, or market-hours behavior to
make the result pass.

Rollout remains serial on schema 19: keep the current generation recording only while
durable admission, ownership, integrity, socket/subscriptions, WAL, database cap, and
inbox bounds remain healthy; then stop recorder/web, take and restore-check a fresh
post-drain backup, deploy the reviewed release, run one offline full pass and integrity
checks, and restart the same run as a new generation. Require 12 consecutive scheduled
normal-maintenance successes under actual callbacks, no unresolved current-generation
retention incident, fresh heartbeat, growing raw sequence, every exact configured
identity active without duplicates, bounded DB/WAL/inbox, and truthful HTTP 200
readiness. Binary rollback remains schema-19-compatible and must preserve committed
evidence. A real regular-session observation remains required for host/IBKR market-open
proof.

This change affects prospective-record/shadow market-data receipt retention and the
readiness it degrades. It does not affect risk, execution, reconciliation, accounts,
credentials, paper/live/order capability, subscription semantics, or XNYS regular-
session behavior.

### Fixed-gate evidence for the exact two-by-600 implementation

The exact implementation passed the frozen mature-host gate while the production
recorder continued at approximately 98% CPU. Twelve consecutive full normal-retention
passes completed without an error. The two per-pass receipt proofs produced 24
successful `_roll_receipts` measurements of 28.583–52.438 ms; their authoritative
`_receipt_actions` work took 8.756–16.307 ms, proof verification took 17.926–34.171 ms,
and the corresponding commits took 1.531–5.013 ms. The independently bounded read-only
selectors took 10.532–23.618 ms. Pruning remained 13.169–29.525 ms. Overall manager
wall time was 119.562–324.924 ms, which is intentionally reported separately from the
unchanged per-writer-transaction 100 ms contract.

A second 12-pass exact sample also completed 12/12. Callback count (1,542,306), maximum
source sequence (1,542,586), non-null payload count (50,981), receipt count (9,846),
receipt callback total (1,321,955), and the complete ordered receipt hash
`f88aec07e06ae5abd38a8f35322526efbb8e9dc9d3261679692fa0c4d98a9c68` were unchanged.
The authoritative watermark callback total advanced from 1,516,706 to 1,530,398
callbacks; the non-round 13,692 delta is expected because receipt boundaries are
atomic and each transaction stops before exceeding 600. The watermark hash changed as
expected from proof advancement. Schema remained 19. Final integrity/FK verification
is retained as a stopped-service cutover gate because a concurrent read-only
`quick_check` over the 3.5 GB disposable copy exceeded the attended 90-second
diagnostic window under live recorder CPU pressure and was interrupted without a
result or mutation.

The unchanged deterministic 10-second replay admitted, durably stored, and projected
all 2,022 callbacks with no acceptance failure. The unchanged 60-second replay passed
with 12,132 presented/admitted/durable/projected callbacks, zero missing or duplicate
callbacks, zero ordering/provenance violations, zero escaped busy/locked errors,
maximum/final backlog 219/0, backlog drain 0.180 seconds, p50/p95/p99 admission latency
0.156/0.306/9.660 ms, admission/projection throughput 407.49/1,433.68 callbacks per
second, zero heartbeat delay, 100/100 required feeds fresh and active, readiness true
in 2.344 ms, and RSS growth 34,029,568 bytes. No acceptance threshold was changed.

The independent Reviewer found one deployment-blocking audit gap: the storage
exception carried its bounded phase and committed receipt progress, but
`Recorder.maintain` persisted only the exception class. The closure binds the exact
`MaintenanceDeadlineExceeded`, whitelists only the four maintenance phase names and
integer ranges 0–2 committed receipt transactions / 0–2,000 rolled receipt rows, and
persists those fields with the generation-scoped component incident. Arbitrary
exception attributes and callback payload material cannot enter incident details. A
recorder integration regression proves transaction-two visibility, continued
degradation, resolution on retry, and retention of the diagnostic evidence.

After that closure, the focused retention/component set passed 8/8, the 2,022 replay
passed, and the unchanged 12,132 replay again passed with zero loss, duplicates,
ordering/provenance violations, or escaped SQLite errors; maximum/final backlog was
219/0, drain time 0.161 seconds, p50/p95/p99 admission latency
0.154/0.309/9.712 ms, admission/projection throughput 402.60/1,410.53 callbacks per
second, heartbeat delay zero, 100/100 required feeds fresh/active, readiness true in
2.259 ms, and RSS growth 24,444,928 bytes.

## Accepted cutover addendum: split the cumulative evidence transaction

The stopped-service cutover on the exact reviewed receipt release exposed one further
measured boundary. Its required offline full-retention pass committed both receipt
transactions and exactly 233 receipt-row changes, then raised
`MaintenanceDeadlineExceeded` in
`terminalization_payload_compaction_and_pruning`. The current transaction rolled back
atomically: callback count, maximum sequence, payload count, and market-event count
were unchanged. Receipt count fell by 233 and the verified watermark advanced by the
same 1,187 callbacks removed from receipts. Preserve those valid commits; do not
restore the pre-pass backup or retry the same combined transaction.

The Architect accepted the smallest measured correction on schema 19. Split the
existing second writer transaction into two transactions with independent, unchanged
100 ms deadlines and one carried 2,000-row budget:

1. `terminalization_and_payload_compaction` performs the existing bounded pending
   terminalization and set-based proof-authorized payload compaction, subtracts both
   counts from 2,000, verifies writer authority after `BEGIN IMMEDIATE` and immediately
   before commit, and commits.
2. `expired_evidence_pruning` runs only when budget remains, begins a new transaction,
   passes exactly that remainder to the unchanged `_prune_expired`, repeats both
   authority checks, and commits.

If the first transaction commits and pruning fails, retain and report its valid
terminalization/payload changes while rolling back every prune change. Extend bounded
maintenance diagnostics with `terminalizations_committed`,
`payloads_compacted_committed`, and `expired_rows_deleted_committed`; keep the existing
exception type and recorder degradation boundary. Do not reset the budget, change a
cutoff or predicate, relax the deadline, add a schema/index/state/loop, or modify the
receipt transactions, cadence, cap behavior, subscription handling, or XNYS calendar.

TDD must reproduce cumulative work crossing 100 ms before the split and prove the two
new transactions pass independently, the carried budget never exceeds 2,000, zero
remainder skips pruning, first-phase deadline/authority loss rolls it back, and
second-phase failure preserves/accountably reports only the first commit. Existing
proof corruption, payload authorization, protected-row anti-join, ownership, component
incident, callback ordering/duplicate/gap, payload-drain, and opening-replay tests
remain required. Validate at least 12 consecutive full passes on the disposable exact
schema-19 production copy, then obtain independent read-only review.

For rollout, keep both services stopped and runtime-masked, retain the fresh checked
backup, and preserve the already committed 233 receipt changes. Deploy the exact
reviewed split release, require one offline full-retention pass to exit successfully,
verify schema 19, `quick_check=ok`, and zero foreign-key violations, then restart the
same run as a new recorder generation. Require 12 consecutive scheduled maintenance
opportunities under callbacks, fresh heartbeat, raw sequence growth, all exact 41
validated identities active without duplicates, bounded DB/WAL/inbox, no unresolved
current-generation retention incident, and truthful readiness before declaring the
cutover complete. If implementation/review cannot complete before the operational
cutoff, temporarily restart the reviewed receipt release in explicitly degraded mode
to preserve raw market evidence; readiness must remain 503 and that is not release
acceptance.

This addendum affects only prospective/shadow market-data retention and its readiness
signal. It does not change risk, execution, reconciliation, accounts, credentials,
paper/live/order capability, subscription semantics, or NYSE/XNYS regular-session
behavior.

## Accepted final addendum: keep heavy retention outside the regular session

The exact B1/B2 split was rejected by its fixed production-copy deployment gate:
under the single-vCPU host workload only one of 12 passes completed, with bounded
failures observed in receipt selection, both receipt transactions, and pruning. This
proves further transaction slicing would not solve CPU/I/O starvation. Generation 9
therefore remains on the reviewed receipt release through the current session; raw
admission, the connected socket, all 41 identities, and fail-closed caps remain active,
while readiness truthfully reflects any degradation.

The Architect accepted session-aware scheduling as the smallest measured correction.
At timestamps where the existing `market_data_expected_since_us(now_us)` says XNYS
regular-session data is expected, `Recorder.maintain` performs only storage-cap
measurement and runtime DB/WAL publication. It must retain the existing hard-cap fatal
actions and 95% optional-feed pause, must not call full `RetentionManager.run`, and must
neither open nor resolve a retention incident merely because work was intentionally
deferred. Outside that exact existing calendar session—including holidays and after an
early close—the recorder runs the full bounded maintenance path every 10 seconds using
the two receipt transactions and B1/B2 evidence split. No new calendar, setting,
timer, daemon, state, or hard-coded UTC window is allowed.

Capacity remains based on the frozen market-open workload, not a relaxed post-result
threshold. The ordinary 17.5-hour off-session window provides 6,300 maintenance
opportunities: 7.56 million callbacks of receipt-proof capacity at 1,200 per pass and
12.6 million physical evidence rows at 2,000 per pass. The fixed 12,132-callback per
minute scenario extrapolates to approximately 4.73 million callbacks over a 6.5-hour
regular session, leaving material proof/drain headroom.

Tests must cover before open, exact open, in-session, exact close, early close,
holiday, and DST through the existing XNYS calendar; normal/soft/degraded/fatal cap
behavior during scheduled deferral; no false incident opening/resolution; actual
off-session recovery; the capacity arithmetic from frozen constants; and unchanged
ordering, duplicates, gaps, admission, per-feed staleness, and 2,022/12,132 replays.
The exact candidate must pass at least 12 consecutive full passes on the mature copy
without callback pressure and a session-pressure test proving zero heavy-retention
calls while raw admission continues.

Deploy only in an attended off-session window: stop/mask, take and restore-check a
fresh schema-19 backup, require one successful offline full pass and integrity/FK
proof, restart the same run as a new generation, prove 12 off-session maintenance
successes, then observe the next real regular session. No paper/live/order, risk,
execution, reconciliation, account, subscription, or XNYS-session semantics change.

## Accepted incident addendum: preserve passive WAL control during deferral

Generation 9 failed closed with `WAL_CAP_FATAL` 26 seconds after the 2026-08-17 XNYS
open. The WAL grew from approximately 19.8 MiB to the 64 MiB hard boundary while the
single-vCPU recorder handled the callback burst. Process exit checkpointed it to zero,
but the session scheduler above had accidentally removed the existing every-pass
`PRAGMA wal_checkpoint(PASSIVE)` behavior. Deploying that scheduler unchanged would
therefore make a repeat fatal likely. The recorder remains stopped while this bounded
correction is implemented and reviewed.

During XNYS regular hours the recorder will still skip all heavy receipt,
terminalization, payload-compaction, pruning, incremental-vacuum, and backup work, but
it will attempt the existing passive checkpoint at the existing 10-second cadence and
measure DB/WAL size afterward. The post-checkpoint sizes drive the unchanged soft,
degraded, and hard-cap actions. A passive-checkpoint error or incomplete result
(including fewer checkpointed frames than logged frames even when SQLite's busy column
is zero) is visible as recoverable retention degradation unless it is an existing hard
storage/integrity error; a WAL that remains at or above 64 MiB still fails closed.
Outside the existing XNYS session,
the full bounded path remains unchanged. No new loop, setting, calendar, state,
service, deadline, or cap is introduced.

`WAL_CAP_FATAL` becomes eligible for the existing explicit, audited exact-generation
recovery command only after the process-local writer lock is acquired, database schema
and integrity/FK verification pass, a passive checkpoint leaves WAL below its hard
cap, the database is below its hard cap and writable, and run mode/config/input/current
generation/fatal termination evidence match exactly. The failed generation and fatal
incident remain immutable evidence. `STORAGE_CAP_FATAL`, corruption, ownership,
durable-admission, provenance, identity, and unknown fatal codes remain
non-recoverable.

Blocking tests cover post-checkpoint measurement, a busy/partial checkpoint below the
cap, a WAL still at the cap, DB hard-cap preservation, callback admission and fixed
opening-burst ordering/latency, and exact `WAL_CAP_FATAL` recovery acceptance/denials
for capped WAL, capped DB, corruption, live ownership, incompatible identity and wrong
fatal evidence. Deployment requires a new checked post-fatal schema-19 backup, one
successful offline full-retention pass with integrity/FK proof, the exact audited
generation-9 recovery, and same-run generation-10 startup. Poll WAL more frequently
than the observed 26-second failure window for the first 60–120 seconds and require a
demonstrated passive checkpoint/reduction, fresh heartbeat, 41 exact active identities,
source-sequence growth, backlog drain, no current-generation incident, and truthful
readiness. Because the feed is already stopped, this reviewed incident correction may
be deployed during the current session; waiting for close would only extend the gap.

## Accepted urgent addendum: admission-based per-feed freshness

Generation 10 reproduced a distinct market-data supervision failure under the real
single-vCPU session load. Durable callback admission and source sequencing continued,
but the nonterminal inbox reached 19,457 rows while projection lagged. Feed staleness
was calculated from `subscriptions.latest_event_id`, which advances only after
projection, so all 41 healthy broker requests repeatedly cycled through stale,
disconnected, and connecting states even though fresh callbacks had already been
durably admitted. The recorder was clean-stopped before the 50,000-row hard inbox
boundary; this preserves the same run lineage and avoids converting a recoverable
condition into `INBOX_FULL` fatal evidence.

The Architect accepted one schema-20 observation field on the existing authoritative
subscription row: nullable `last_admitted_callback_at_us`, constrained to be no earlier
than `opened_at_us`. Exact current-fence callback insertion and this timestamp update
must occur in the same durable transaction. A rejected stale-generation/request
callback and an idempotent duplicate must not advance it. `mark_stale` and per-feed
readiness use this admitted timestamp rather than projected `market_events`; readiness
continues to report the independent durable-inbox backlog, so projection delay cannot
be hidden. Migration 20 removes the superseded schema-19 acknowledged-callback
freshness index.

Admission proves transport activity only. It must not close a stale, rejection,
disconnect, or retry incident. Recovery still requires an acknowledged callback with
a non-null normalized event identity from the exact current run, recorder generation,
connection generation, subscription, and logical request, received no earlier than
the subscription's `last_attempt_at_us`. Permanent rejection and unresolved farm
outage rules remain unchanged. Thus malformed evidence can defer a stale cancellation
without falsely restoring readiness, and valid evidence queued before a retry cannot
restore the replacement request.

TDD must cover fresh admitted-but-unprojected callbacks, per-feed isolation, malformed
evidence, pre-retry queued evidence, stale-generation/request fencing, insertion and
timestamp atomicity, duplicate replay, existing XNYS outside-session behavior, and an
evidence-preserving schema-19-to-20 migration. Rerun the unchanged 2,022- and
12,132-callback opening replays plus callback ordering, duplicate, gap, ownership,
database-writability, and inbox-full failure tests. Obtain a separate read-only review
before deployment.

Rollout is serial and attended: keep recorder/web stopped and runtime-masked, take and
restore-check a fresh schema-19 quiescent backup, apply schema 20 offline, verify the
migration ledger, `quick_check=ok`, and zero foreign-key violations, then restart the
same run as a new recorder generation. Require all exact validated subscriptions to
remain active without stale/retry cycling, per-feed admission timestamps to advance,
WAL to remain bounded, and backlog to trend below the 5,000 readiness limit. After any
schema-20 callback admission, roll forward only. Online backup timers remain disabled
until the separately observed `_online_copy` failure is diagnosed and reviewed.

This addendum affects prospective-record and shadow raw market-data freshness,
subscription supervision, and the read-only readiness projection. It does not alter
risk, execution, reconciliation, order capability, fills, positions, accounts,
credentials, paper/live trading, or the existing NYSE/XNYS regular-session calendar.

The unchanged fixed 60-second replay passed after schema 20 with 12,132 callbacks
presented, admitted, durable, and projected; zero missing, duplicate, ordering,
provenance, or escaped SQLite busy/locked failures; maximum/final backlog 219/0;
backlog drain 0.135 seconds; admission p50/p95/p99 0.165/0.317/8.826 ms;
admission/projection throughput 413.50/1,508.70 callbacks per second; heartbeat delay
zero; all 100 required feeds fresh and active; readiness true in 2.150 ms; RSS growth
21,544,960 bytes; and post-session-checkpoint WAL 5,162,392 bytes. Acceptance failures
were empty; no threshold was changed after observing the result.

## Accepted deployment addendum: index-align derivation expiry selection

The attended schema-20 cutover reproduced a deterministic offline maintenance failure.
The required full-retention pass twice exited nonzero in `expired_evidence_pruning`;
the second pass committed no receipt, terminalization, payload-compaction, or prune
change. A disposable copy of the exact schema-20 production database reproduced the
failure. Direct rollback-only timing isolated 321.963 ms of the 329.207 ms prune pass
to a zero-result `market_event_derivations` deletion. The table contained 86,109 rows,
all newer than the cutoff, but `EXPLAIN QUERY PLAN` showed a full table scan because
the generic helper ordered this table by `rowid` rather than the declared
`market_event_derivations_retention_idx(created_at_us, derived_event_id,
input_ordinal)`.

The Architect accepted one code-only correction: map `market_event_derivations` to
`ORDER BY created_at_us, derived_event_id, input_ordinal` in the existing bounded
delete helper. This preserves the retention predicate, oldest-first semantics, the
shared 2,000-row pass budget, the 100 ms transaction deadline, writer-authority
checks, rollback behavior, and all protected-evidence rules. It adds no schema,
index, state, setting, service, or loop.

TDD uses the public `RetentionManager.run` seam with a production-shaped large
all-newer derivation fixture under the unchanged deadline, plus a plan assertion that
the exact candidate seek uses the existing retention index without a full table scan.
Coverage also proves eligible rows are selected oldest-first, protected recent rows
remain, the carried physical-row budget is respected, and deadline/authority loss
still rolls back. Before production, the exact disposable schema-20 copy must complete
the direct prune harness materially below 100 ms and at least 12 consecutive full
retention passes without a deadline failure. The fixed opening replays and focused,
failure-oriented, full repository, and independent review gates remain unchanged.

Production stays stopped and runtime-masked until the exact reviewed code-only release
passes those gates. The verified pre-migration snapshot and schema-20 database remain
authoritative; the 64 receipt rows committed before the rolled-back prune attempt are
valid evidence and are not reversed. Deployment then requires one successful full
production retention pass, schema-20 ledger/checksum verification, `quick_check=ok`,
zero foreign-key violations, DB/WAL below their hard caps, and retained generation-9
fatal evidence before same-run startup. This affects prospective/shadow retention
only and does not change subscriptions, XNYS session behavior, risk, execution,
reconciliation, paper/live trading, accounts, credentials, or order capability.

The exact code-only candidate was measured on the disposable schema-20 production
copy before cutover. The rollback-only direct prune harness completed with zero
eligible/deleted rows in 1.743 ms across 15 statements; the formerly dominant
`market_event_derivations` statement completed in 0.460 ms, versus the fixed baseline
of 321.963 ms and 329.207 ms total. Twelve consecutive full `stocker-runtime retain`
passes then exited zero with `status:"ok"`, `cap_state:"normal"`, no deadline failure,
and a reported WAL of 20,632 bytes on every pass. No pass compacted payloads, rolled
receipts, or deleted expired rows; incremental vacuum reduced the copy by one 4 KiB
page per pass as designed.

The unchanged 2,022-callback automated replay passed. The unchanged fixed 60-second
replay also passed with 12,132 callbacks presented, admitted, durable, and projected;
zero missing, duplicate, ordering, provenance, or escaped SQLite busy/locked failures;
maximum/final backlog 219/0; drain time 0.153 seconds; admission p50/p95/p99
0.194/0.440/10.039 ms; admission/projection throughput 372.93/1,363.86 callbacks per
second; heartbeat delay zero; all 100 required feeds fresh and active; readiness true
in 3.102 ms; RSS growth 22,134,784 bytes; and post-session-checkpoint WAL 5,162,392
bytes. Acceptance failures were empty and no threshold was changed after measurement.
The full storage suite passed 130/130 tests. The candidate's migration-ledger and
integrity verifier then exited zero on that same disposable copy with
`applied_versions:[]`, `current_version:20`, and `status:"ok"`; this also verified the
schema-20 migration checksums, `quick_check=ok`, and zero foreign-key violations.

## Accepted live addendum: actual-mix projection capacity

Generation 11 proved the schema-20 admission-freshness correction against the real
IBKR session: the socket connected, every exact 41/41 subscription remained active,
every feed acquired a durable admission timestamp, source sequence advanced from
1,639,710 to 1,647,967, no subscription stale/retry cycle occurred, and observed order
capability remained zero. It also exposed a separate single-vCPU capacity boundary.
The inherited nonterminal backlog grew from 22,038 to 26,711 in 146 seconds while the
process consumed approximately one full CPU. The 8,257 admitted callbacks imply about
3,393/minute ingress; only 3,584 older callbacks were projected, about 1,473/minute,
so continuation would have reached the existing 50,000-row fail-closed boundary.
Generation 11 was therefore clean-stopped and runtime-masked with `CLEAN_STOP`; its
source sequence and backlog remained stable, the writer lock was free, and a passive
checkpoint completed all 196 frames and left WAL at zero.

Diagnosis and correction remain off the production database. A disposable exact
generation-11 copy must exercise the real pending payload mix through
`lease_pending -> project_batch -> acknowledge/fail -> receipt` under the server's
single-vCPU constraint. Measure lease, each 32-row projection transaction, each 32-row
terminalization/count publication transaction, receipt creation, writer wait, and any
downstream work that runs despite a full backlog. The fixed capacity target is at
least 1.25 times the observed peak ingress used in the Architect review: 6,412
projected callbacks/minute, with monotonically draining backlog, no loss, duplicate,
ordering or provenance violation, bounded heartbeat/WAL/memory, and no weakened
50,000-row cap.

The leading hypothesis is measured, not assumed: each callback admission and each
terminalization chunk currently executes an exact nonterminal `count(*)`. If that
scan dominates under concurrent admission, transactionally maintain the existing
authoritative `runtime_state.inbox_nonterminal_count`, reconciling it exactly during
startup/recovery and proving duplicate admission, stale-fence failure, rollback,
crash/restart, acknowledgement and hard-cap behavior. If projection SQL or downstream
work instead dominates, change only that measured boundary. Do not add a queue,
thread, daemon, table, service, global latch, larger batch chosen blindly, relaxed cap,
or callback acknowledgement before durable evidence. A fixed actual-mix red/green
replay, focused failure tests, opening replays, full checks, and independent read-only
review are required before same-run restart.

This addendum affects prospective/shadow durable callback projection and readiness
only. It does not change XNYS hours, broker subscriptions, risk, execution,
reconciliation, accounts, credentials, paper/live trading, or order capability.

The generation-11 disposable copy disproved the initial count-scan hypothesis.
Projection without production plugins sustained 26,414 callbacks/minute; production
plugins sustained 16,949/minute; repeated nonterminal counts completed at 1.018 ms
p50 and 1.202 ms p95; sequential durable admission sustained 402.48 callbacks/second;
and a 30-second concurrent 56-callback/second simulation with production plugins
projected about 13,103 callbacks/minute while the backlog fell by 4,871 rows. All
measurements exceed the frozen 6,412/minute target.

The actual starvation cause is the full-batch downstream deferral predicate. In the
stopped production database, 5,893 terminal callbacks are not yet receipted and the
receipt frontier at source sequence 1,615,363 trails the first pending callback at
1,617,759. A drain projects 256 callbacks but `create_pending_receipts` creates one
bounded receipt of at most 256 callbacks for the run. Its returned receipt therefore
covers an older terminal prefix, not the newly leased callbacks. The receipt-coverage
comprehension evaluates false on every saturated batch, so option discovery, plugins,
dynamic reconciliation and shadow work run before the next raw batch. The receipt
frontier and projection frontier then advance at the same rate and the condition
cannot recover while the backlog remains saturated.

The Architect accepted removal of only that superseded receipt-coverage condition.
After `project_batch`, the existing receipt attempt and heartbeat all succeed, a full
lease (`len(leased_callbacks) == limit`) is sufficient to defer optional downstream
work and immediately re-service the already rearmed callback wake-up. The leased
callbacks are already durably terminalized; receipt creation was attempted and any
failure still escapes before deferral. Receipt proof is an independent retained
frontier and may trail by a fixed amount while saturated; optional downstream work
cannot advance it. Underfull and idle drains continue to run downstream and allow the
existing receipt path to converge. Replay explicitly passing
`defer_downstream_when_full=False` remains unchanged. Do not increase receipt limits,
add transactions, or weaken receipt/hash verification.

TDD must reproduce at least 2,395 older terminal unreceipted callbacks followed by a
full 256 pending batch. The full batch must project and acknowledge all callbacks,
rearm the wake-up, attempt receipt creation before returning, and invoke zero option,
plugin, dynamic or shadow work even though its newly returned receipt covers only the
older prefix. Repeated full batches must keep the receipt lag non-growing and defer
downstream; an underfull batch and explicit offline override must run downstream; an
injected receipt failure must still prevent downstream and remain visible. Callback
ordering, duplicates, gaps, provenance and plugin isolation remain covered.

Before rollout, rerun the actual generation-11 mix on a fresh disposable copy under a
single-vCPU-equivalent constraint. Require at least 6,412 projections/minute, a
monotonically falling 26,711-row backlog, no loss/duplicate/order/provenance error,
non-growing then converging receipt lag, and bounded heartbeat, WAL and memory. There
is no migration. Generation 11 ended cleanly, so restart the same run as generation
12 without fatal recovery. Poll backlog, WAL and heartbeat every few seconds; require
backlog reduction within 10 seconds and sustained negative slope for 60--120 seconds,
41 exact active subscriptions without stale cycling, zero order capability, and a
bounded first passive checkpoint. Clean-stop if the backlog has not fallen by 30
seconds or reaches the conservative 35,000-row operational cutoff. Start the web only
after backlog is below the existing 5,000 readiness threshold and downstream plus
receipt catch-up have resumed. Real market-open host-I/O observation remains a
separate required proof.

## Accepted live addendum: bounded observation during saturated recovery

The exact reviewed projection candidate passed the generation-11 disposable-copy
gate. It drained 26,455 measured pending callbacks to zero in 95.745 seconds, or
16,578.36 callbacks/minute, with individual saturated batches normally completing in
about 0.36--1.13 seconds. The initial unreceipted terminal count fell from 5,893 to
5,724 after the underfull batch, callback row/event/source-sequence identities remained
one-to-one, WAL returned to zero, and the latest diagnostic generation clean-stopped.
A second single-CPU 30-second simulation admitted 1,680 callbacks at 56/second without
producer error while backlog fell from 26,455 to 22,684; derived projection throughput
was 10,840/minute and maximum RSS was 219,868 KiB with zero swap. Both exceed the fixed
6,412/minute acceptance threshold.

Generation 12 then confirmed the correction against IBKR: all 41 subscriptions became
active and produced callbacks, order capability remained zero, and backlog fell from
26,711 to 16,113 while source sequence advanced from 1,647,967 to 1,649,145. During
rollout observation, however, an operator `sqlite3 -readonly` process issued several
unbounded aggregate queries against the active 1.65-million-row database while the
single CPU was saturated. That reader overlapped the scheduled passive checkpoint.
The checkpoint correctly reported incomplete, physical WAL reached 120,118,632 bytes,
and the existing 64 MiB rule fail-closed generation 12 as `WAL_CAP_FATAL`. Process exit
then checkpointed WAL to zero. The fatal evidence and valid projection progress are
retained.

The Architect rejected an unmeasured checkpoint-code change. The platform is not
required to tolerate arbitrary local direct readers, and `RESTART`, `TRUNCATE`, or a
checkpoint after every saturated batch could block durable admission. One explicit
audited same-run generation-12 `WAL_CAP_FATAL` recovery on unchanged reviewed commit
`1a4eec2fdeae1108a328ee45875ce2645f2b9023` is accepted only after a new verified
quiescent snapshot, exact schema/integrity/FK/writability/cap checks, WAL zero or a
complete passive checkpoint, and proof that no process or writer lock remains. The
authorization reason must name the unbounded rollout reader and its replacement with
hard-bounded monitoring.

Generation 13 monitoring must not run `count(*)`, joins, integrity checks, or large
scans on the active database. Filesystem `stat`, process state and logs are safe.
Any SQLite observation must be a single-row/indexed lookup protected by a hard process
deadline below 300 ms; SQLite `busy_timeout` alone is not a query deadline. Require
backlog decline from 16,113 within 10 seconds, the first scheduled passive checkpoint
to complete without a retention incident, WAL to remain below a conservative 48 MiB
abort threshold and visibly reset, fresh heartbeat, connected socket, all 41 exact
subscriptions, and zero order capability. Observe for 60--120 seconds, require backlog
below 5,000 and receipt/downstream catch-up, then start the bounded web application.
Clean-stop at no backlog decline by 30 seconds, WAL at 48 MiB, checkpoint incomplete
without a known bounded reader, or any integrity/ownership/admission fault. A second
autonomous WAL fatal must not be recovered without a measured code correction.

Add a disposable regression proving that a deliberately held read transaction makes
the passive checkpoint incomplete, while no-reader and bounded web-budget reads keep
WAL below the existing cap. Update the linked runbook to forbid unbounded direct SQLite
reads on an active operational database and name the supported bounded observation
path. No schema, XNYS, subscription, risk, execution, reconciliation, paper/live,
account, credential, or order-capability behavior changes.

## Accepted live addendum: one optional downstream component per drain

Generation 13 recovered the exact generation-12 `WAL_CAP_FATAL` through the audited
same-run path after a restore-verified schema-20 snapshot. Raw projection recovered
from 16,113 nonterminal callbacks to 109, all exact 41 subscriptions were active and
fresh, the socket stayed connected, order capability remained zero, and physical WAL
stabilised at 40,936,352 bytes. With the bounded web reader enabled, readiness then
oscillated between 200 and 503. The only reason was `RECORDER_HEARTBEAT_STALE`: the
heartbeat advanced again after roughly seven-second gaps while backlog remained
119--430, feeds remained fresh, and no feed/component incident appeared. This proves
that an underfull drain resumes several individually optional components in one
orchestration turn and can cumulatively exceed the fixed five-second main-loop
heartbeat contract.

The accepted correction preserves the five-second threshold and the existing single
orchestration thread. A full callback lease continues to defer all optional downstream
work. Each underfull or idle `Recorder.drain` invocation may run at most one existing
component, chosen by one process-local round-robin cursor in this order: option
projection, idea runner, option discovery, shadow evaluation. The cursor advances
after success, recoverable failure, or component backoff so no component can starve
the others. The next recorder iteration therefore services durable raw callbacks,
publishes the existing heartbeat, and runs health supervision before another optional
component. No heartbeat thread, additional heartbeat transaction, database state,
schema, setting, queue, service, threshold increase, or new failure latch is added. A
single component that genuinely exceeds five seconds remains truthfully visible as a
stale heartbeat.

TDD uses the public `Recorder.drain` seam. Four bounded fake components whose aggregate
runtime exceeds five seconds reproduce the old single-turn exposure; the corrected
behavior runs exactly one component per eligible drain in deterministic order. Tests
also prove full batches run none, underfull and idle calls rotate without starvation,
recoverable failure/backoff advances rotation and persists its incident, callbacks
arriving during one component are drained before the next, offline
`defer_downstream_when_full=False` retains deliberate downstream execution one
component per invocation, and existing receipt failure/order/duplicate/gap/plugin
isolation behavior remains unchanged. Before rollout, measure each component on a
disposable stopped-generation snapshot and require each individual call at or below
four seconds; if any component exceeds that bound, split or bound only that measured
component rather than masking it with scheduling.

This is a code-only schema-20 rollout. Generation 13 may continue collecting while raw
admission remains healthy, backlog stays below 5,000 without a rising trend, WAL stays
below 48 MiB, every feed stays active/fresh, and heartbeat advances within 30 seconds.
For deployment, clean-stop generation 13, mask the web reader, install the exact
reviewed candidate, and restart the same run as generation 14 without fatal recovery.
The fixed acceptance gate is 60--120 seconds of stable readiness with maximum heartbeat
gap below five seconds, backlog below 5,000, WAL below 48 MiB, all downstream/receipt
frontiers progressing, 41/41 fresh active feeds, and zero order capability. This
affects prospective/shadow downstream scheduling and readiness only; XNYS hours,
subscriptions, risk, execution, reconciliation, accounts, credentials, paper/live
trading, and broker-order capability remain unchanged.

## Superseding decision: component-completion heartbeats preserve causal ordering

The round-robin candidate above is rejected and must not be deployed. Although its
focused tests and opening replays passed, the full repository suite produced 33 new
failures across dynamic option interest activation, discovery/fence publication,
snapshot completion/reconnect races, and shadow invocation. Those failures prove that
the existing same-`drain` downstream order is a causal domain contract. Rewriting
those contracts to fit scheduling would broaden this remediation and risk delayed or
missing derived evidence.

The Architect accepted the smaller compatibility-preserving seam. Add an optional
caller-supplied `component_completion_clock_us` to `Recorder.drain` and the component
boundary. The production recorder loop supplies `time.time_ns() // 1_000`;
deterministic direct callers and replays may omit it or inject a fake. Preserve the
existing order and run every applicable component in the same drain: option
projection, idea runner, connected option discovery, then shadow evaluation. After a
component returns successfully, or after its recoverable failure has been isolated
and persisted, publish the existing process heartbeat at
`max(causal_now_us, component_completion_clock_us())`. Do not publish a completion
heartbeat for a backed-off component that did not run, while a component is still
running, or after ownership loss, `RecorderFatalError`, hard storage/I/O failure, or
another exception that escapes the component boundary.

This preserves truthful readiness without a thread or threshold increase. Several
healthy bounded components may cumulatively take more than five seconds without
making the recorder appear wedged because each completed boundary proves main-loop
progress. One component that alone blocks beyond five seconds remains visibly stale
until it actually returns. Full production batches with normal deferral still skip all
optional downstream work; the explicit offline/replay override continues to run all
stages in their original order. No schema, persisted cursor, state, queue, service,
daemon, additional writer, or trading capability is added.

Blocking tests use a fake completion clock to prove same-call order and a heartbeat
after every completed boundary, recoverable failure incident persistence plus later
component execution, an event-blocked component remaining stale until release, no
false completion heartbeat after fatal/ownership failure, deterministic behavior when
the clock is omitted, production CLI clock injection, and unchanged full-batch
deferral/override behavior. All dynamic option/shadow tests that rejected round-robin
must pass unchanged. Before deployment, a disposable exact generation-13 workload
must show every individual component at or below four seconds, maximum heartbeat gap
below five seconds, stable readiness, and bounded callback backlog/WAL. If one
component exceeds five seconds, this candidate is insufficient and only that measured
component may be bounded.

Generation 13 may continue recording under the existing attended stop gates while the
candidate is tested. Rollout remains code-only on schema 20: clean-stop generation 13,
mask the web reader, install the exact reviewed candidate, and restart the same run as
generation 14 without fatal recovery. Require 60--120 seconds of stable readiness,
backlog below 5,000, WAL below 48 MiB, all 41 feeds active/fresh, downstream evidence
progress, and zero order capability before handoff. XNYS hours, subscriptions, risk,
execution, reconciliation, accounts, credentials, and paper/live behavior remain
unchanged.

The component boundary must also suppress every completion heartbeat while any
component incident is still pending durable publication. A contended incident write
keeps its process-local pending marker and permits later components to continue, but
neither the failed component nor those later successful components may advance the
heartbeat until the evidence has been persisted. Once publication succeeds, the
completion heartbeat may advance while the durable component incident keeps readiness
degraded. Direct authority loss and durable-inbox authority loss converted to
`AuthoritativeLeaseLost` remain heartbeat-silent. This closes the false-green window
without adding another latch: the existing pending-incident collection is the exact
publication boundary.

The final local candidate passed the focused ingestion, deployment, idea and shadow
suites together, including every same-drain causal contract that rejected the
round-robin design. The independent read-only Reviewer reported no remaining blocking
finding after five focused closure tests and `git diff --check`. The unchanged opening
replays then produced:

| replay | presented/admitted/projected | loss/duplicates/order/provenance | p50 / p95 / p99 admission | max/final backlog | heartbeat delay | readiness | WAL after checkpoint |
|---|---:|---:|---:|---:|---:|---|---:|
| 10 seconds | 2,022 / 2,022 / 2,022 | 0 / 0 / 0 / 0 | 0.164 / 0.299 / 8.688 ms | 219 / 0 | 0 s | true | 4,622,672 B |
| 60 seconds | 12,132 / 12,132 / 12,132 | 0 / 0 / 0 / 0 | 0.164 / 0.324 / 9.257 ms | 219 / 0 | 0 s | true | 5,162,392 B |

Both replays reported zero escaped SQLite busy/locked errors, all 100 simulated
required feeds active and fresh, and no acceptance failure. These simulations do not
replace the required stopped-generation component timing or the attended generation-14
IBKR observation.

The final repository test command completed with 2,082 passing and one skipped test.
It retained 13 failures and 19 setup errors, all caused by the already-absent protected
`trade_decisions.parquet` historical research input; the same inherited failures were
present before this candidate. The repository check stopped at its first formatting
gate because 151 unrelated, pre-existing research/core files would be reformatted.
Every changed Python file passes focused Ruff formatting/checking and MyPy; the focused
runtime suites and replays above are green. Neither inherited condition is concealed
or modified by this operational rollout.

## Stopped-generation correction: typed idea candidate lookup

The exact generation-13 disposable-copy gate rejected the first
component-completion-heartbeat candidate before deployment. Three idle drains measured
option projection at no more than 31.583 ms, option discovery at 66.763 ms, and shadow
evaluation at 22.402 ms, but the idea runner took 6,852.991--7,423.190 ms. Per-instance
instrumentation isolated one `frozen_m1c_signal` load at 6,977.196 ms; the other three
instances took 391.231, 643.290 and 408.590 ms.

Statement timing proved the final candidate SELECT was the bottleneck. Five-minute bar
projection took 262.697 ms and session projection 93.244 ms, while the candidate query
took 7,931.587 ms to execute and 177.657 ms to fetch 256 rows. Its plan searched
`market_events_causal_sequence_idx` by run and executed a correlated `json_each` scan
for each causal event. The production-shaped run held 344,707 events, including 288,919
irrelevant quotes, and the instance's checkpoint began at sequence zero.

The accepted smallest correction adds only an all-typed fast path. When the validated,
nonempty in-memory batch requirements all have an exact `event_kind`, `json_each`
drives `market_events_instrument_kind_time_idx` by exact instrument and event kind;
the existing run/feed/availability/causal/tie-break/bar-fence predicates, distinctness,
ordering and 256 limit remain unchanged. Empty, mixed or nullable/dynamic requirement
sets execute the former general SQL and parameter order byte-for-byte. No query is
moved across a projector or gap boundary, and no schema or index is added.

On the exact copy, the replacement returned the identical ordered 256 event identities
and used the intended `(instrument_id,event_kind)` index search. Its first cold run took
474.973 ms and subsequent runs 16.346--30.272 ms; a separate equality run measured the
old query at 7,369.783 ms and the replacement at 16.346--25.294 ms. The accepted tests
pin exact row/order/filter equivalence, duplicate requirements, source/derived causal
ties, wrong run/feed/kind, availability boundaries, the bar activation fence, the
index plan, a deterministic SQLite opcode budget, and exact fallback selection.

The Architect rejected a per-instance heartbeat callback because this local query fix
should bring the complete runner below four seconds. The existing outer component
completion heartbeat remains the only new heartbeat boundary. Deployment remains
blocked until three or more cold/warm drains on the exact generation-13 copy show the
whole idea runner and every component at or below four seconds, the maximum heartbeat
gap below five seconds, and unchanged idea/checkpoint/interest/fence/shadow evidence.
Production remains stopped and runtime-masked until that gate and independent review
pass. This remains a schema-20 code-only same-run generation-14 restart with no fatal
recovery and no trading/session scope change.

## Superseding correction: sparse-derived candidate selection

The exact generation-13 copy rejected commit `e0f9c00`; it must not be deployed. The
all-typed predicate fixed the `frozen_m1c_signal` instance (301.874 ms load) but sent
two other typed instances from their approximately 0.38--0.40 second baseline to
6,613.639 ms and 6,449.350 ms. Complete idea-runner drains regressed from
7.542--9.345 seconds to 16.220--25.262 seconds. The returned evidence remained equal;
the defect was the unbounded intermediate match/sort cost for dense raw event kinds.

The accepted correction retains the same SQL but selects it only for a nonempty set in
which every requirement is a bars feed and every event kind is one of the two
projector-owned sparse derived kinds: `bar_5m_session_prefix` or
`session_volume_baseline`. Raw bar, quote, trade, option, `bar_5m`, nullable, non-bars,
and mixed sparse/dense sets use the original causal query byte-for-byte. This adds no
runtime statistics, plugin identity, duplicated subscription count, schema, index,
state, setting or background work.

The fixed acceptance thresholds remain unchanged: deterministic tests must prove both
directions under SQLite opcode budgets and exact row/order equality; the exact copy
must show at least three drains with every component and the complete idea runner at
or below four seconds, process-heartbeat gaps below five seconds, the formerly slow
derived instance remaining subsecond, the raw instances returning to their baseline
range, and identical idea/checkpoint/interest/fence/shadow evidence. The 2,022- and
12,132-callback opening replays retain their existing fixed thresholds. Any failure
stops deployment rather than broadening the selector or heartbeat threshold.

This remains prospective/shadow market-data projection and readiness work only. It
does not alter XNYS session rules, risk, execution, reconciliation, accounts,
credentials, broker capabilities, paper trading or live trading.
