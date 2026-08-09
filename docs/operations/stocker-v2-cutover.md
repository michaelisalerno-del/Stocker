# Stocker V2 cutover and recovery runbook

This is the frozen internal deployment contract for the one-way Stocker V1 to V2
cutover. It is an operator procedure, not deployment automation. Run it only in a
planned market-closed window with recorded owner approval. Paper and live remain
absent and disabled; this procedure starts only prospective-record or shadow data
collection and the query-only web application.

There is no dual write.

## Fixed paths and identities

- stopped V1 database: `/var/lib/stocker/prospective/prospective.sqlite3`
- immutable V1 recovery sets: `/var/lib/stocker/recovery-v1/`
- new V2 database: `/var/lib/stocker/v2/stocker-v2.sqlite3`
- V2 backups: `/var/lib/stocker/backups-v2/`
- recorder identity: `stocker-recorder`
- web identity: `stocker-web`
- backup identity: `stocker-backup`
- read group: `stocker-readers`
- Stocker IBKR proxy: `127.0.0.1:4003`

The release contains only `stocker-v2-recorder.service`, `stocker-v2-web.service`,
and the V2 daily/weekly backup units for the Stocker application. Gateway units remain
market-data-only infrastructure. Never install a V1 recorder, web, or backup unit from
an earlier release over these files.

## 1. Approval and preflight

Record the owner approval, release commit, operator, UTC window, V1 database path,
intended V2 mode, and rollback-window owner in the change record. Confirm the market is
closed and no unattended start is pending. The only allowed V2 modes are
`prospective_record` and `shadow`.

Prepare users and paths from the reviewed release without starting services:

```bash
getent group stocker-readers >/dev/null || sudo groupadd --system stocker-readers
id -u stocker-recorder >/dev/null 2>&1 || sudo useradd --system --gid stocker-readers \
  --home-dir /var/lib/stocker --shell /usr/sbin/nologin stocker-recorder
id -u stocker-web >/dev/null 2>&1 || sudo useradd --system --gid stocker-readers \
  --home-dir /var/lib/stocker --shell /usr/sbin/nologin stocker-web
id -u stocker-backup >/dev/null 2>&1 || sudo useradd --system --gid stocker-readers \
  --home-dir /var/lib/stocker --shell /usr/sbin/nologin stocker-backup
sudo usermod --append --groups stocker-readers stocker-recorder
sudo usermod --append --groups stocker-readers stocker-web
sudo usermod --append --groups stocker-readers stocker-backup
sudo install -d -o root -g stocker-readers -m 0750 /var/lib/stocker
sudo install -d -o stocker-recorder -g stocker-readers -m 2750 /var/lib/stocker/v2
sudo install -d -o stocker-backup -g stocker-readers -m 2750 /var/lib/stocker/backups-v2
sudo systemctl daemon-reload
sudo systemctl is-enabled stocker-v2-recorder.service stocker-v2-web.service || true
```

Both services and both backup timers must still be disabled. Validate the reviewed
configuration files, their hashes, the release commit, the 8 GiB database/backup caps,
and that there are no broker account, credential, order, paper, or live fields.

## 2. Preserve the complete V1 recovery set

Stop all V1 scheduling and application processes before copying anything:

```bash
sudo systemctl disable --now stocker-backup.timer
sudo systemctl stop stocker-recorder.service stocker-web.service stocker-backup.service
sudo systemctl reset-failed stocker-recorder.service stocker-web.service
```

Prove that no V1 process has the database open. Checkpoint it, then require both SQLite
checks to pass and no rollback journal, `-wal`, or `-shm` sidecar to remain:

```bash
sudo -u stocker-recorder sqlite3 /var/lib/stocker/prospective/prospective.sqlite3 \
  'PRAGMA wal_checkpoint(TRUNCATE); PRAGMA quick_check; PRAGMA foreign_key_check;'
sudo lsof -- /var/lib/stocker/prospective/prospective.sqlite3
sudo test ! -e /var/lib/stocker/prospective/prospective.sqlite3-journal
sudo test ! -e /var/lib/stocker/prospective/prospective.sqlite3-wal
sudo test ! -e /var/lib/stocker/prospective/prospective.sqlite3-shm
```

The `lsof` command must print no open handle (its normal no-match exit status is
acceptable). Do not proceed merely because the service manager reports the unit as
stopped.

Query `recorder_lease` and `recorder_generation_v1`; there must be no lease and no
generation with a null stop timestamp. If either exists, abort rather than deleting it.

Create one checked, read-only recovery set containing the database, a compressed
database copy, SQLite/WAL state, raw partitions, sidecars, staging/quarantine, bundles,
and required reports. Record file sizes and SHA-256 values in its manifest, verify the
compressed database with `quick_check` and `foreign_key_check`, then remove write bits
from the set. Preserve it until the owner closes the rollback window. Do not copy V1
credentials or vendor tokens into V2 configuration.

## 3. One-way import into a new V2 target

The target and its reconciliation file must not exist. The importer opens V1 with
SQLite `mode=ro&immutable=1`, accepts only frozen schema prefixes through `0026`, and
writes a new temporary V2 database in the target directory:

```bash
sudo -u stocker-recorder /opt/stocker/current/.venv/bin/stocker-runtime legacy import \
  --source /var/lib/stocker/prospective/prospective.sqlite3 \
  --target /var/lib/stocker/v2/stocker-v2.sqlite3
sudo /opt/stocker/current/deploy/scripts/prepare-v2-sqlite-boundary.py
```

The setgid V2 directory makes the imported `0640` target inherit
`stocker-recorder:stocker-readers`; the boundary verifier then confirms that ownership
and creates only the coordinated WAL/SHM files. The target hard link is the atomic
commit marker. On any failure, keep V1 stopped,
remove no recovery evidence, and investigate before retrying with a new target path.
Never import into an existing V2 database and never reverse-import V2 rows into V1.

Review `<target>.migration-reconciliation.json` and the matching
`migration_manifests` row. Require all of the following:

- the current source SHA-256 equals `source_database_hash`;
- the frozen source schema digest matches the recorded prefix;
- `source_row_count = imported_row_count + omitted_row_count` globally and per table;
- every omission has a fixed reason and every table has a row-classification hash;
- the manifest reconciliation digest and target digest match the sidecar;
- `PRAGMA quick_check` returns `ok` and `PRAGMA foreign_key_check` returns no rows;
- the target is owned by `stocker-recorder:stocker-readers`, mode `0640`, and no larger
  than 8 GiB; and
- imported runs preserve protected data classes and are stopped.

Do not proceed if the source hash changed, a WAL appeared, any row is unreconciled, or
the target contains an account, order, fill, broker-position, transfer, report-package,
or legacy vendor runtime surface.

## 4. Install the V2-only service surface

Install the reviewed V2 units under their exact names and remove installed V1 Stocker
application unit files. Do not alias old names to new services. Install configuration
with root ownership and least-privilege read access. Confirm:

```bash
systemctl list-unit-files 'stocker*' --no-pager
systemctl is-active stocker-recorder.service stocker-web.service stocker-backup.timer
systemctl is-enabled stocker-v2-recorder.service stocker-v2-web.service \
  stocker-v2-backup-daily.timer stocker-v2-backup-weekly.timer
```

The three V1 checks must report absent/inactive and the four V2 checks must remain
disabled until the next two sections. Only the recorder may write the database or WAL.
Web and backup hold read-only database/WAL paths plus the narrow SQLite SHM permission
defined in their reviewed units.

## 5. Prove the web boundary before recording

Start the V2 web first:

```bash
sudo systemctl start stocker-v2-web.service
```

Authenticate over the loopback/reverse-proxy boundary. Verify the exact seven GET API
routes, three primary views, diagnostics drawer, 512 KiB response cap, rate limiting,
and the fixed prospective/shadow banner. Attempt a write through the web service
identity and require SQLite query-only rejection. The database hash and row counts must
not change while exercising the web UI.

## 6. Admit new V2 data

Create a new V2 run identifier and recorder generation; never resume an imported V1
run. Verify the official IBKR API provenance record and loopback/read-only boundary,
then start the V2 recorder:

```bash
sudo -u stocker-recorder /opt/stocker/current/.venv/bin/stocker-runtime ibkr-api verify \
  --provenance /var/lib/stocker/ibkr-api/active-provenance.json
sudo /usr/local/libexec/stocker-verify-ibgateway-loopback-boundary
sudo systemctl start stocker-v2-recorder.service
```

Before declaring admission complete, prove one callback was durably acknowledged into
a generic market event, the expected plugin set was discovered, and the runtime has
zero order capability and zero broker mutation. Record the first callback sequence and
UTC timestamp in the change record. No broker account or position read is part of this
proof.

## 7. Observation and completion

Keep the deployment attended for the bounded observation window recorded in the change
approval. Review recorder lifecycle, freshness, gaps, incidents, plugin isolation,
callback backlog, database/WAL size, and web query-only behaviour. Run one checked V2
backup and a disposable restore verification. Only then enable the reviewed V2 daily
and weekly timers and declare cutover complete.

## 8. Rollback

### Before first callback

If V2 has not admitted its first callback, stop V2, preserve its logs and failed import
artifacts, restore the prior release/service pointer, and restart the untouched V1
database. Do not copy any V2 row into V1.

### After first callback

If V2 has admitted data, stop V2 and take a checked V2 backup first. Restore the prior
release against the untouched V1 database, create a new V1 run and recorder generation,
and record the entire V2 interval as an explicit gap. Never reverse-import V2 into V1
or claim uninterrupted scientific continuity. Prefer roll-forward repair.

In both paths, preserve both recovery sets and releases until the owner closes the
rollback window.

## 9. Market-data-only IB Gateway boundary

Gateway login is manual. Use an SSH tunnel:

```bash
ssh -N -L 5901:127.0.0.1:5901 stocker-host
```

Enter the manual IBKR username, password, and 2FA only in Gateway; never enter the Stocker website.
Enable Gateway's Read-Only API and bind its upstream socket to
loopback. The Stocker recorder connects only to `127.0.0.1:4003` through the verified
proxy.

The host firewall starts with `sudo ufw default deny incoming`. Validate the upstream
port before installing the exact loopback guard:

```bash
case "$IBKR_GATEWAY_PORT" in
  ''|*[!0-9]*) exit 78 ;;
esac
sudo ufw insert 1 deny in proto tcp to any port "$IBKR_GATEWAY_PORT"
sudo ufw insert 1 allow in on lo proto tcp to any port "$IBKR_GATEWAY_PORT"
sudo /usr/local/libexec/stocker-install-ibgateway-loopback-boundary
sudo /usr/local/libexec/stocker-verify-ibgateway-loopback-boundary
sudo /usr/local/libexec/stocker-verify-ibgateway-nft-boundary-json
sudo nft -j list table inet stocker_ibgateway
```

The nftables guard uses priority -300. Store only
`IBGATEWAY_UPSTREAM_PORT=` in the root-readable proxy configuration. The Gateway units
contain no credentials.

Install the reviewed Gateway tree and verifier atomically, with recorded hashes:

```bash
sudo useradd --system --home-dir /var/lib/ibgateway --shell /usr/sbin/nologin ibgateway
sudo mkdir --mode=0750 "$TARGET"
sha256sum --check ibgateway-release.sha256
sudo test ! -e "$PROVENANCE"
sudo ln "$PROVENANCE_TMP" "$PROVENANCE"
sudo ln "$INSTALLER_TMP" "$INSTALLER"
sudo /opt/stocker/current/deploy/scripts/verify-ibgateway-installation.sh
sudo install -d -o root -g ibgateway -m 0710 /etc/ibgateway
```

Use the provided X11/VNC units for attended login; VNC is loopback-only and uses the
pre-provisioned `/var/lib/ibgateway/vnc.pass`. Stocker never creates or reads it.
