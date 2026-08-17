# Stocker V2 market-open reliability operations

This runbook applies only to `prospective_record`, `shadow`, and the read-only web
application. It changes market-data collection and operational visibility. It does not
authorize broker orders, account access, paper/live trading, risk, positions, execution,
or reconciliation. IBKR remains the active market-data source.

## Run and recorder-generation lifecycle

A run is the compatible prospective/shadow evidence lineage. A recorder generation is
one process lifetime within that run. `SIGTERM`, `systemctl restart`, a deployment
restart, and orderly shutdown close only the current generation. Starting the same
configuration and market-data input with the same `run_id` creates the next generation;
callback source sequences remain append-only and existing callback evidence is neither
rewritten nor duplicated.

An incompatible mode, frozen recorder configuration, or market-data input hash is
rejected with the mismatched identity. Do not rotate `run_id` for an ordinary restart.
Do not edit SQLite lifecycle rows manually.

The recorder holds a kernel file lock beside the canonical database path for its whole
lifetime. A second local process fails closed. After `kill -9` or host restart, the
kernel releases that lock and the first normal systemd retry can close the abandoned
generation and create an auditable replacement. Schema-15 generations have no lock
protocol marker and retain the old heartbeat-expiry takeover rule during rollout.

## Schema 16–20 rollout and rollback

Schema 16 adds generation ownership, commit/input identity, fatal-recovery audit fields,
and per-subscription staleness/retry state. Existing rows and fatal evidence are
preserved. Schema 17 adds the partial payload-compaction candidate index. Schema 18
adds nullable recorder-generation provenance to component incidents; legacy `NULL`
incidents remain visible but cannot poison a new generation. Schema 19 added an
acknowledged-callback covering index for exact per-feed readiness. Schema 20 adds a
nullable durable last-admitted-callback timestamp to each existing subscription row
and removes that superseded history index. None of these migrations rewrites callback
evidence. V1 databases remain rejected and must never be migrated in place. The
accepted migration design and measurements are recorded in
[`20260817-v2-retention-payload-candidate-index.md`](../plans/20260817-v2-retention-payload-candidate-index.md).

The historical schema-18-to-19 production migration built an approximately 89.2 MB index.
On the restored 3.62 GB production backup, the exact index build took 8.76 seconds and
the complete migration framework, including verification, took approximately 137
seconds. Verify enough free space for the database, WAL/temp work, the new index, and a
checked backup before beginning. Do not increase the 300 ms query budget to compensate
for a missing or incomplete index.

In a market-closed attended window:

1. Stop recorder and web.
2. For the current schema-19-to-20 route, create a fresh checked compressed schema-19
   backup and retain the matching schema-19 release. Restore-check the backup to a
   disposable path using the existing backup commands. If the managed online-copy path
   is unavailable while services are fully quiescent, use the reviewed canonical-lock,
   complete-WAL-truncate, byte-identical emergency snapshot procedure; do not mark that
   emergency artifact as a healthy managed backup.
3. Run `stocker-runtime migrate /var/lib/stocker/v2/stocker-v2.sqlite3`.
4. Require the ledger to report schema 20, exact schema checksum verification,
   `foreign_key_check` with zero rows, and `quick_check=ok`.
5. Confirm every current subscription exposes its durable
   `last_admitted_callback_at_us`, existing schema-19 rows begin `NULL`, and the
   superseded `callback_inbox_readiness_latest_idx` is absent. On a restored mature
   copy, require ten exact-feed readiness calculations to return all diagnostics, each
   within 300 ms and p95 within 275 ms.
6. Run the combined preflight shown below.
7. Restart with the same V2 `run_id`, mode, frozen configuration, and validated input.
   Require a new recorder generation, a fresh heartbeat, connected IBKR market-data
   socket, every exact required configured identity active, raw sequence growth, no
   duplicate active request, and zero unresolved current-generation component
   incidents.
8. Start web only after the recorder is healthy. Require `/` to return liveness and
   `/api/v2/ready` to select that exact generation and complete within 300 ms.

Before the new schema admits callbacks, rollback requires the matching old release and
the checked pre-migration backup. After any callback is admitted under the new schema,
preserve the database and roll forward; restoring the older backup would lose evidence,
and old code must not open the newer schema. Never drop an index or edit migration or
incident rows manually on production.

### Quiescent emergency snapshot when managed backup is unavailable

The tracked fallback is
[`scripts/quiescent_v2_snapshot.py`](../../scripts/quiescent_v2_snapshot.py). Use it
only after both services are stopped and runtime-masked. It acquires the canonical
`<database>.writer.lock` non-blockingly and holds it across WAL truncation, source
stability checks, byte-for-byte copy, matching-runtime schema/checksum verification,
`quick_check`, foreign keys, callback/generation evidence checks, deterministic gzip,
and decompression verification. It fails before producing a manifest if any service or
database descriptor remains, the checkpoint is incomplete, the source changes, hashes
differ, or restored evidence differs. It does not publish managed-backup health or
forge a `BackupManifest`.

Record the stopped generation's exact maximum source sequence and pending/leased count,
then run from the exact candidate source tree while naming the still-installed matching
schema release:

```bash
uv run python scripts/quiescent_v2_snapshot.py \
  --database /var/lib/stocker/v2/stocker-v2.sqlite3 \
  --destination-directory /var/lib/stocker/recovery-snapshots \
  --runtime-executable \
    /opt/stocker/releases/<matching-schema-commit>/.venv/bin/stocker-runtime \
  --expected-schema 20 \
  --run-id stocker-v2-shadow-20260816t191816z \
  --generation <stopped-generation> \
  --expected-termination-code CLEAN_STOP \
  --expected-max-source-sequence <recorded-maximum> \
  --expected-nonterminal <recorded-pending-plus-leased-count> \
  --release-commit <matching-schema-commit> \
  --operator <operator-identity>
```

For a fail-closed generation snapshot, pass its exact recorded fatal termination code
instead of `CLEAN_STOP`. The verifier then requires `clean_stop=0`; for `CLEAN_STOP`
it requires `clean_stop=1`. It never accepts a mismatched generation state or code.

Require exit zero and one JSON object with `status=ok`, exact expected evidence,
identical snapshot/decompressed hashes, `restore_verified=true`, and
`uncompressed_retained=false`. The command requires free space for three database-size
working files plus 512 MiB, limits each compressed artifact to 9 GiB, retains exactly
the two newest compressed recovery snapshots with their manifests, and deletes the
verified uncompressed working copy. These artifacts remain outside managed backup
health/rotation, but the command's own two-copy/18-GiB maximum is mandatory. Never
manually retain its uncompressed working copy. This fallback is not permission to
snapshot a live database or a nonzero WAL separately.

### Unattended managed backups

The production daily and weekly timers use the tracked quiescent helper
[`run-v2-quiescent-managed-backup.py`](../../deploy/scripts/run-v2-quiescent-managed-backup.py),
not the live online-copy command. Daily work is scheduled for 22:00 UTC with a bounded
random delay; weekly work is scheduled for Sunday 06:00 UTC. The helper independently
uses the existing XNYS calendar and requires at least one hour before the next regular
session. A persistent timer caught up during regular hours or too near the next open
fails before stopping either service.

For an accepted window the helper clean-stops web and recorder, holds the canonical
writer lock, proves no database descriptor remains, completely truncates the WAL, and
creates a byte-identical source copy. The ordinary managed-backup verifier, deterministic
compression, manifest, rotation floors, and 8 GiB directory cap then apply. The newly
published archive is decompressed to a temporary path and schema, hashes, `quick_check`,
and foreign keys are verified before success. Recorder is restarted first with the same
`run_id` and a new generation. After systemd confirms recorder is active, the helper
runs the tracked root-owned SQLite-boundary preparer, then requires a fresh heartbeat
from that exact owned generation. This avoids a root read-only heartbeat query racing
recorder-owned WAL/SHM creation while still proving the recorder is operational before
web starts. The preparer tolerates only bounded SQLite create/unlink races; identity,
ownership, file-type, permission, and storage failures remain fail-closed. Recorder,
heartbeat, or boundary failure keeps web stopped and the backup degraded rather than
presenting stale data as recovered. Both
the helper and systemd
`ExecStopPost` provide this ordered restart boundary on success, ordinary failure, or
service timeout. A failed snapshot remains nonzero/degraded; it is never reported as a
healthy backup merely because the services restarted.

The backup unit intentionally cannot write the `/var/lib/stocker` parent directory.
The boundary preparer validates that parent's exact type, owner, group and mode, but
does not issue a redundant metadata write when the mode is already correct. A real
mode correction inside the read-only backup namespace fails with the named
`persistent_root_mode_update_failed` boundary error; do not widen the unit's
`ReadWritePaths` or retry that safety failure. WAL/SHM corrections remain restricted
to the exact writable V2 database directory.

The first invocation after each release is attended and off-session:

```bash
sudo systemctl daemon-reload
sudo systemctl start stocker-v2-backup-daily.service
sudo systemctl status stocker-v2-backup-daily.service --no-pager
sudo cat /var/lib/stocker/backups-v2/backup-status.json
```

Require service exit zero, backup status `healthy`, a current daily manifest and
restore-checked archive, a cleanly ended prior generation, the same run in a new
generation, fresh recorder heartbeat, connected socket, every exact configured
required identity active, increasing raw sequence, bounded inbox/WAL, zero order
capability, and truthful `/api/v2/ready`. Only after that proof enable the timers:

```bash
sudo systemctl enable --now \
  stocker-v2-backup-daily.timer stocker-v2-backup-weekly.timer
systemctl list-timers 'stocker-v2-backup-*' --no-pager
```

Do not run `stocker-runtime backup create` against the active production database as a
substitute. The command remains available for disposable/offline administration, but
the installed timer units accept only the fixed quiescent helper and named service
pair. Never run an unbounded direct SQLite read while the recorder is active.

### IB Gateway authentication is not unattended

The installed Gateway unit intentionally uses manual paper-account authentication.
Its daily readiness probe correctly failed on 14, 15, and 16 August 2026: systemd
restarted the Gateway at 23:45 UTC, but API port 4002 did not reopen during the bounded
120-second probe. This is an operator alarm, not a recorder defect.

Do not disable the probe, store credentials, automate login or 2FA, change account
selection, or lengthen the probe merely to make systemd green. After a broker restart,
an operator must authenticate through the existing VNC procedure unless an already
authorized IBKR session is separately proven to resume without credential or account
changes. `/api/v2/ready` remains 503 while the socket or required subscriptions are
unavailable. Consequently Stocker process recovery and backups are unattended, but
end-to-end IBKR availability still has this explicit broker-authentication boundary.

## Production preflight

The service unit validates both files before creating an IBKR connection:

```bash
/opt/stocker/v2-current/.venv/bin/stocker-runtime validate-recorder \
  /etc/stocker/recorder.json --inputs /etc/stocker/market-data.json
```

Preflight rejects an empty universe, no required feeds, duplicate or contradictory
identities, missing instruments, malformed requests, configured base snapshots, and a
set above `market_data_line_limit`. The tracked market-data file is conspicuously
example-only and must be replaced with an operator-reviewed universe. Errors identify
the invalid file and reason without printing credentials.

Required subscription failure makes readiness false, but healthy feeds remain active
and continue durable capture. Optional failure is visible degradation and does not make
the required set incomplete. Request retries are independent, fenced, exponentially
backed off, and use a fresh transport incarnation. Permanent contract/entitlement
rejection remains visible and does not spin.

## Fatal-generation recovery

A fatal row from another run never blocks a new run. Fatal evidence is never cleared.
Only the narrow recoverable code printed by the failed generation may be authorized,
offline and exactly once, after the cause is repaired:

```bash
stocker-runtime recorder recover-fatal-generation \
  --config /etc/stocker/recorder.json \
  --inputs /etc/stocker/market-data.json \
  --generation REPLACE_WITH_GENERATION \
  --fatal-code REPLACE_WITH_ELIGIBLE_FATAL_CODE \
  --operator REPLACE_WITH_OPERATOR \
  --reason 'REPLACE_WITH_RECORDED_REMEDIATION'
```

The command acquires the same writer lock and checks exact run/mode/config/input
identity, integrity, writability, hard capacity, fatal code, and absence of an owner.
`WAL_CAP_FATAL` is eligible only after its passive-checkpoint remediation is installed;
the recovery command also checkpoints and verifies that WAL is below the hard cap.
Corruption, ownership loss, callback ordering/provenance loss, hard capacity, unsafe
adapter capability, unknown fatal codes, and incompatible identity remain blocked.

## Liveness, readiness, and regular-session scope

`GET /` is process/web liveness only. It deliberately says nothing about recorder or
feed readiness. `GET /api/v2/ready` returns HTTP 200 only for the selected operational
run and HTTP 503 otherwise. Its JSON names the selection reason, run, recorder
generation, lifecycle/heartbeat, socket, inbox threshold, database admission state,
and every current expected feed with active, stale, retrying, permanent-rejection,
callback, incident, and retry diagnostics.

When web configuration pins `run_id`, that exact run is used or reported unavailable.
Without a pin, fresh current recorder-generation evidence selects the operational run;
a newer failed/empty attempt is reported and cannot silently obscure it. Historical
inspection remains separate from current readiness.

Tick freshness is evaluated only inside the existing XNYS regular session. Per-feed
freshness uses the callback timestamp committed atomically with raw durable admission;
it does not wait for canonical projection. A malformed callback may prove that a
transport is active, but only a normalized callback from the current retry attempt may
close a subscription incident. The independent inbox-backlog reason prevents delayed
projection from being presented as ready. The shared
exchange calendar retains holidays, daylight-saving changes, and early closes. There
is no pre-market, after-hours, futures, forex, international, hard-coded UTC, or
per-instrument calendar expansion. Outside regular hours, quiet feeds are not stale,
while process, socket, configuration, and subscription lifecycle remain reported.

The web query budget defaults to 300 ms and is capped at 500 ms. Schema 19 changed the
mature lookup from repeated history scans to exact acknowledged-callback covering-index
seeks; schema 20 replaces that lookup with the durable value on the selected current
subscription row. On the restored 41-feed production copy, the pre-schema-19 query
failed between about 301 and 2,583 ms. The accepted schema-19 measurement completed ten
calculations in 8.202–14.081 ms, with p95 11.903 ms. Revalidate schema-20 timing during
rollout. Timeout remains bounded and returns a web-query timeout/503 without implying
recorder ingestion failure.

Web startup prewarms the existing XNYS calendar before the listening socket is made
available. The accepted deployment measurement was 2,795.926 ms; treat it as startup
time, not a failed health query. Session calculation also occurs before the
SQLite deadline, so a calendar cache miss cannot consume the database query budget. A
long-lived process can still pay calendar-library initialization latency on the first
request for a newly uncached New York date; that may delay or false-red that request,
but cannot make readiness falsely green or weaken the 300 ms SQLite bound. Do not add
hard-coded hours or a second calendar cache to avoid this behavior.

Do not run unbounded direct `sqlite3` aggregates, joins, integrity checks, or table
scans against the active operational database. A long-lived direct read snapshot can
prevent a passive WAL checkpoint from advancing and correctly drive the recorder into
`WAL_CAP_FATAL`. SQLite `busy_timeout` does not bound query execution. During an
attended saturated-backlog recovery, observe WAL with filesystem `stat`, process state
with systemd/journal, and recorder state only through the bounded web readiness path or
a single-row/indexed query protected by an OS process deadline below 300 ms. Stop the
recorder before running `quick_check`, `foreign_key_check`, callback counts, or other
diagnostic scans.

## Downstream degradation

Canonical derived projection, individual plugins, the idea runner, option projection
and discovery, shadow evaluation, retention, and backup maintenance have narrow
incident/retry boundaries. Their ordinary failures do not disconnect healthy feeds or
erase durable raw callbacks. Repeated failures back off to 60 seconds. Recovery closes
the matching component incident. A web query cannot own or mutate recorder state.

Database corruption/unwritability, duplicate ownership, callback identity/order loss,
inbox exhaustion, and hard database/WAL pressure remain fail-closed. At the 95% storage
degradation boundary optional feeds are paused idempotently. If SQLite is temporarily
unable to persist a component incident, the process retains and republishes it before
normal recovery; a process crash in that narrow interval can lose the in-memory health
marker, but never the already durable raw callback or writer evidence.

## Schema-19 retention backlog recovery

Repeated `COMPONENT_RETENTION_MAINTENANCE_FAILED` incidents may leave durable raw
admission healthy while an old payload backlog cannot drain. Do not suppress the
incident, increase the 100 ms writer-transaction deadline, or manually edit SQLite.
The current schema-19 release uses at most two bounded set-based payload updates per
pass while retaining receipt/watermark proof, acknowledged-first ordering, the shared
2,000-row cap, and atomic rollback.

Normal schema-19 maintenance first performs a separately bounded read-only selector,
then at most two receipt-proof writer transactions. Each writer transaction verifies
no more than 600 callbacks and has its own unchanged 100 ms deadline; together they
retain the existing 1,200-callback proof ceiling and carry the existing shared 2,000
receipt-change budget. Receipt IDs, hashes, watermark bounds, and deletions are always
re-read and validated after `BEGIN IMMEDIATE`; the outside selector supplies only a
run-ID hint. Writer authority is checked after begin and immediately before every
commit. If transaction two fails, transaction one's watermark/rollup remains valid,
the current incident stays degraded, and retry resumes from durable evidence. Do not
manually alter a watermark or retry by suppressing the incident.

The remaining evidence work also uses two bounded writer transactions with one carried
2,000-row budget. The first terminalizes expired pending shadow positions and compacts
proof-authorized callback payloads; the second prunes expired evidence only with the
remaining budget. Both keep the same 100 ms deadline and repeat writer-authority checks
after `BEGIN IMMEDIATE` and before commit. A pruning failure therefore retains and
reports any valid first-transaction progress while rolling back the prune transaction.
Generation-scoped incident details report only bounded phase and committed-count
fields. Do not interpret an overall maintenance wall time above 100 ms as a breach:
the deadline applies independently to each short writer transaction.

Heavy retention is scheduled only outside the existing NYSE/XNYS regular session. At
regular-session timestamps the recorder preserves the existing passive WAL checkpoint
and measures and publishes DB/WAL cap state and heartbeat after that attempt. Hard-cap
failure remains fail-closed and the 95% boundary still pauses optional feeds. Receipt
proof, payload compaction, evidence pruning, incremental vacuum, and backup work resume
at the unchanged 10-second cadence outside the session, including holidays and after
an early close. This uses the same exchange calendar as feed
staleness; it does not add pre-market/after-hours data collection or hard-coded UTC
hours. An intentional in-session skip neither opens nor resolves a retention incident;
only a real off-session maintenance success resolves one.

Use an attended offline recovery only after the incident has been diagnosed:

1. Stop recorder and web, runtime-mask both units, and verify they are inactive with no
   recorder process. The mask is the fail-closed start barrier around backup,
   verification, and restart preparation; do not depend on operator convention.

   ```bash
   systemctl stop stocker-v2-recorder.service stocker-v2-web.service
   systemctl mask --runtime stocker-v2-recorder.service stocker-v2-web.service
   test "$(systemctl is-active stocker-v2-recorder.service)" = inactive
   test "$(systemctl is-active stocker-v2-web.service)" = inactive
   ! pgrep -af '[s]tocker-runtime recorder run'
   ```
2. Create a fresh checked compressed backup of the exact deployed schema and
   restore-check it to a disposable path. Record callback, receipt, and watermark counts/hashes plus
   `quick_check` and `foreign_key_check`.
3. Verify the exact release artifacts, recorder/input preflight, and deployed schema. Run the
   tracked bounded command below. It acquires the canonical `.writer.lock` once for the
   entire loop and verifies it in every transaction; an active recorder or another
   drain fails before mutation.

   ```bash
   timeout --foreground 2705 \
     /opt/stocker/v2-current/.venv/bin/stocker-runtime recorder drain-payloads \
       --database /var/lib/stocker/v2/stocker-v2.sqlite3 \
       --max-passes 1000 \
       --max-wall-seconds 2700
   ```

   Require exit zero, `status=ok`, at most 1,000 passes, at most 2,000 payloads per
   committed pass, and `consecutive_zero_passes=2`. The command uses one fixed cutoff
   and emits total compacted rows. Pass/time/deadline/lock loss exits nonzero and names
   the incomplete committed total; it never claims earlier successful passes rolled
   back. The outer `timeout` is only a last-resort process bound: if it fires before
   the command emits JSON, reconcile the committed total from the recorded pre/post
   payload counts instead of assuming the current pass or earlier passes rolled back.
4. Recompute the evidence hashes. Because this command is purpose-built payload-only,
   callback row count and every immutable callback field (including `payload_sha256`),
   receipt row/count/hash, watermark row/count/hash, and every other table must match.
   The non-null payload reduction must exactly equal the reported compacted total.
   Require the exact deployed schema, `quick_check=ok`, zero foreign-key violations,
   and bounded DB/WAL.
5. Keep the recorder runtime-masked while installing/verifying the final release and
   updating the frozen commit identity. Unmask only at the intentional handoff; start
   recorder first and web only after recorder health is proven.

   ```bash
   systemctl unmask --runtime stocker-v2-recorder.service
   systemctl start stocker-v2-recorder.service
   # Verify generation, heartbeat, socket, exact subscriptions, and raw sequence here.
   systemctl unmask --runtime stocker-v2-web.service
   systemctl start stocker-v2-web.service
   ```
6. Restart the same `run_id`, creating a new recorder generation. Require a fresh
   heartbeat, connected market-data socket, every exact identity from validated input
   active without duplicates, growing raw sequence, and no new generation-scoped
   retention incident across at least 12 consecutive scheduled maintenance
   opportunities under callback traffic. Then start web and require truthful HTTP 200
   readiness. If one of those passes fails, keep the incident visible and diagnose the
   reported phase; do not raise the deadline or rotate the run ID merely to clear it.

A successful full command may exceed 100 ms because it contains many separately
bounded payload transactions plus passive WAL checkpoints; that is not a transaction-
deadline violation. This recovery nulls only proven eligible `payload_json`; callback
identity/provenance, payload hashes, receipts, and watermarks remain. If release
rollback is needed while evidence is valid, preserve the current database
and roll back only the binary. Restore the checked pre-drain backup only for actual
corruption found before callback admission resumes. After any new callback is admitted,
never restore the older backup.

## Opening replay

Run the fixed larger fake-adapter simulation from the release root:

```bash
uv run python scripts/replay_market_open.py --seconds 60
```

It uses the XNYS open on 2026-08-10, 21 five-second bar feeds, 40 quote feeds at three
callbacks/second, and 39 trade feeds at two callbacks/second: 100 required feeds and
exactly 12,132 callbacks. The 10-second, 2,022-callback form is the normal deterministic
correctness test. The command uses a temporary database, never connects to IBKR, and
fails nonzero if the frozen acceptance thresholds are missed.

Simulation proves callback durability, uniqueness, ordering/provenance, bounded
backlog, heartbeat/freshness, feed continuity, and local SQLite performance for this
host. It does not prove IBKR entitlements, farm/pacing behavior, network behavior, or
production-host market-open performance. Keep rollout attended and observe a real
read-only IBKR regular session before making a stronger operational claim.
