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

## Final unchanged-threshold replay

The post-change 60-second replay passed all frozen thresholds with 12,132 presented,
admitted, durable, and projected callbacks; zero missing, duplicate, ordering,
provenance, or escaped SQLite busy/locked failures; admission p50/p95/p99 of
0.154/0.287/9.617 ms; admission/projection throughput of 421.59/1,496.65 callbacks per
second; maximum/final backlog of 219/0; zero seconds to the 256-row safe range and
0.150 seconds to drain; heartbeat delay 0 seconds; 100/100 feeds active and fresh;
readiness true in 214.01 ms; and RSS growth of 31,326,208 bytes.
