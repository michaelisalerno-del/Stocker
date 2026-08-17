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

## Schema 16–19 rollout and rollback

Schema 16 adds generation ownership, commit/input identity, fatal-recovery audit fields,
and per-subscription staleness/retry state. Existing rows and fatal evidence are
preserved. Schema 17 adds the partial payload-compaction candidate index. Schema 18
adds nullable recorder-generation provenance to component incidents; legacy `NULL`
incidents remain visible but cannot poison a new generation. Schema 19 adds the
acknowledged-callback covering index used by exact per-feed readiness. None of these
migrations rewrites callback evidence. V1 databases remain rejected and must never be
migrated in place. The accepted migration design and measurements are recorded in
[`20260817-v2-retention-payload-candidate-index.md`](../plans/20260817-v2-retention-payload-candidate-index.md).

The current schema-18-to-19 production migration builds an approximately 89.2 MB index.
On the restored 3.62 GB production backup, the exact index build took 8.76 seconds and
the complete migration framework, including verification, took approximately 137
seconds. Verify enough free space for the database, WAL/temp work, the new index, and a
checked backup before beginning. Do not increase the 300 ms query budget to compensate
for a missing or incomplete index.

In a market-closed attended window:

1. Stop recorder and web.
2. For the current schema-18-to-19 route, create a fresh checked compressed schema-18
   backup and retain the matching schema-18 release. Restore-check the backup to a
   disposable path using the existing backup commands. For a later route, record and
   back up its exact deployed schema and retain its matching release.
3. Run `stocker-runtime migrate /var/lib/stocker/v2/stocker-v2.sqlite3`.
4. Require the ledger to report schema 19, exact schema/index checksum verification,
   `foreign_key_check` with zero rows, and `quick_check=ok`.
5. Confirm both a hit and a miss use
   `callback_inbox_readiness_latest_idx` as a covering seek without a callback-sort
   temporary B-tree. On a restored mature copy, require ten exact-feed readiness
   calculations to return all diagnostics, each within 300 ms and p95 within 275 ms.
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
  --fatal-code POST_ADMISSION_PRESERVATION_FAILED \
  --operator REPLACE_WITH_OPERATOR \
  --reason 'REPLACE_WITH_RECORDED_REMEDIATION'
```

The command acquires the same writer lock and checks exact run/mode/config/input
identity, integrity, writability, hard capacity, fatal code, and absence of an owner.
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

Tick freshness is evaluated only inside the existing XNYS regular session. The shared
exchange calendar retains holidays, daylight-saving changes, and early closes. There
is no pre-market, after-hours, futures, forex, international, hard-coded UTC, or
per-instrument calendar expansion. Outside regular hours, quiet feeds are not stale,
while process, socket, configuration, and subscription lifecycle remain reported.

The web query budget defaults to 300 ms and is capped at 500 ms. Schema 19 changes the
mature lookup from repeated history scans to exact acknowledged-callback covering-index
seeks. On the restored 41-feed production copy, the old query failed between about 301
and 2,583 ms. The accepted final measurement completed ten calculations in
8.202–14.081 ms, with p95 11.903 ms. Timeout remains bounded and returns a web-query
timeout/503 without implying recorder ingestion failure.

Web startup prewarms the existing XNYS calendar before the listening socket is made
available. The accepted deployment measurement was 2,795.926 ms; treat it as startup
time, not a failed health query. Session calculation also occurs before the
SQLite deadline, so a calendar cache miss cannot consume the database query budget. A
long-lived process can still pay calendar-library initialization latency on the first
request for a newly uncached New York date; that may delay or false-red that request,
but cannot make readiness falsely green or weaken the 300 ms SQLite bound. Do not add
hard-coded hours or a second calendar cache to avoid this behavior.

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
