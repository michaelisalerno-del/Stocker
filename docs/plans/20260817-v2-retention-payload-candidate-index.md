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
