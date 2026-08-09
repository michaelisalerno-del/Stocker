# Stocker V2 cutover and recovery runbook

This is the frozen internal deployment contract for the one-way Stocker V1 to V2
cutover. It is an operator procedure, not deployment automation. Run it only in a
planned market-closed window with recorded owner approval. Paper and live remain
absent and disabled; this procedure starts only prospective-record or shadow data
collection and the query-only web application.

There is no dual write.

## Fixed paths and identities

- immutable V1 import snapshot:
  `/var/lib/stocker/backups/prospective-20260805T141203Z.sqlite3`
- preserved V1 rollback database:
  `/var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3`
- separately authorised retirement candidate:
  `/var/lib/stocker/prospective/prospective.sqlite3` and its exact `-wal` and `-shm`
  sidecars
- immutable V1 recovery sets: `/var/lib/stocker/recovery-v1/`
- new V2 database: `/var/lib/stocker/v2/stocker-v2.sqlite3`
- V2 backups: `/var/lib/stocker/backups-v2/`
- preserved V1 release pointer: `/opt/stocker/current`
- immutable V2 release: `/opt/stocker/releases/<release-commit>`
- V2 service pointer: `/opt/stocker/v2-current`
- preserved V1 writer identity: `stocker`
- recorder identity: `stocker-recorder`
- web identity: `stocker-web`
- backup identity: `stocker-backup`
- read group: `stocker-readers`
- Stocker IBKR proxy: `127.0.0.1:4003`

The reviewed V2 release contains only `stocker-v2-recorder.service`,
`stocker-v2-web.service`, and the V2 daily/weekly backup units for the Stocker
application. Gateway units remain market-data-only infrastructure. Stage V2 under its
immutable versioned path and point `/opt/stocker/v2-current` at that exact release.
The V2 units execute through this separate pointer. `/opt/stocker/current` must continue
to identify the complete, runnable V1 release throughout the rollback window. Preserve
the installed V1 units, V1 configuration, V1 database, and recovery set for the same
period; do not copy them into the V2 release tree.

## 1. Approval and preflight

Record the owner approval, release commit, operator, UTC window, all three V1 paths,
intended V2 mode, and rollback-window owner in the change record. For this attended
cutover the approved mode is `prospective_record`; Michael owns a seven-day rollback
window. The owner separately authorised irreversible deletion of only the three exact
retirement-candidate paths above after the no-handle/dependency and preservation gates
in Section 2 pass, accepting loss of evidence unique to that aggregate. That approval
does not authorise deletion of the import snapshot or rollback database. Confirm the
market is closed and no unattended start is pending. The only allowed V2 modes remain
`prospective_record` and `shadow`.

Record the exact V1 path returned by `readlink -f /opt/stocker/current`. Replace the
placeholder below with the reviewed V2 commit. The selected official
`ibapi==10.49.1` source declares exactly `protobuf==5.29.5`, but the official archive
contains no wheel. Before cutover, use a separate reviewed packaging environment to
build exactly `ibapi-10.49.1-py3-none-any.whl` from the already hash-verified official
source tree. Review its unpacked `ibapi` Python tree against the immutable official
provenance. The derived wheel must not come from a package registry; it is only an
offline transport artifact and does not replace the official archive as provenance.
Do not vendor the derived wheel in this repository.

The release builder must receive that derived IBAPI wheel and the exact protobuf wheel
as separately reviewed offline artifacts; the uv cache is not evidence that either
artifact exists. Record an exact regular path and a checked one-entry SHA-256 manifest
for each wheel, with the manifest entry naming that exact path. Install both wheels
while the versioned release is still unpublished. The stdlib-only release-artifact
verifier must run under trusted system Python both before and after IBAPI installation;
do not execute Python or a CLI from the new environment first. It validates the
complete wheel and installed RECORD/file set as well as the exact root-owned active
provenance symlink, whose literal target must be
`provenance/10.49.1.json`. Both phases revalidate the exact one-entry manifests and
hashes for both wheels plus root ownership and ACL-free, non-writable trust boundaries for the
artifacts, complete official source tree, reviewed release/verifier, virtual
environment, site-packages, and installed distribution. Group-write is admitted only
for group ID 0; world-write is never admitted. The only admitted directory symlink is
the host-shaped internal `.venv/lib64 -> lib`; its real `.venv/lib` target tree is
scanned in full. Only then verify metadata, concrete
imports, and the installed
`ibapi` Python tree through the new environment. Never modify the release environment
after publishing the V2 pointer.

The reviewed installer is exactly `/usr/local/bin/uv`, SHA-256
`da15297d6879b2cfbe5ea3cb03725c1613d51ba72892cc996468d871f0a532fb`, reporting
`uv 0.11.32 (x86_64-unknown-linux-gnu)`. Do not select it through `PATH`. Before
the release verifier is executed as root, isolated system Python checks its exact
reviewed hash and its root-owned, ACL-free, non-writable ancestry; the verifier then
rechecks the complete release and all installer inputs in both phases.

Security review note: `protobuf==5.29.5` is named in
GHSA-7gcm-g887-7qv7 / CVE-2026-0994. The reviewed official IBKR client source does not
call the affected `json_format.ParseDict` path; it uses generated binary message
modules and `text_format.MessageToString`. Keep the exact compatibility pin until a
separately reviewed official IBKR client/dependency update supplies refreshed
artifacts, hashes, provenance, and regression evidence. Do not silently override the
pin with an unrelated protobuf release.

Prepare the separate V2 pointer, users, and paths without starting services:

```bash
bash -euo pipefail <<'STOCKER_V2_RELEASE_PREP' || exit 78
export STOCKER_V2_RELEASE=/opt/stocker/releases/REPLACE_WITH_REVIEWED_COMMIT
export STOCKER_IBAPI_SOURCE=/var/lib/stocker/ibkr-api/install/IBJts/source/pythonclient
export STOCKER_IBAPI_WHEEL=/var/lib/stocker/ibkr-api/install/ibapi-10.49.1-py3-none-any.whl
export STOCKER_IBAPI_WHEEL_SHA256=/var/lib/stocker/ibkr-api/install/ibapi-10.49.1-wheel.sha256
export STOCKER_IBAPI_PROVENANCE=/var/lib/stocker/ibkr-api/active-provenance.json
export STOCKER_PROTOBUF_WHEEL=/var/lib/stocker/ibkr-api/install/REPLACE_WITH_REVIEWED_PROTOBUF_5_29_5_WHEEL.whl
export STOCKER_PROTOBUF_WHEEL_SHA256=/var/lib/stocker/ibkr-api/install/protobuf-5.29.5-wheel.sha256
export STOCKER_TRUSTED_PYTHON=/usr/bin/python3
export STOCKER_RELEASE_ARTIFACT_VERIFIER="$STOCKER_V2_RELEASE/deploy/scripts/verify_v2_release_artifacts.py"
export STOCKER_RELEASE_ARTIFACT_VERIFIER_SHA256=ebd97c106d28edbbdb54f6e367f336d591880344d90b3c2e40f9b9effd0bafa6
export STOCKER_UV_BIN=/usr/local/bin/uv
export STOCKER_UV_SHA256=/var/lib/stocker/ibkr-api/install/uv-0.11.32.sha256
test "$STOCKER_V2_RELEASE" != \
  /opt/stocker/releases/REPLACE_WITH_REVIEWED_COMMIT || exit 78
command -v setfacl >/dev/null || exit 78
command -v getfacl >/dev/null || exit 78
test -x "$STOCKER_UV_BIN" || exit 78
test "${STOCKER_PROTOBUF_WHEEL##*/}" != REPLACE_WITH_REVIEWED_PROTOBUF_5_29_5_WHEEL.whl || exit 78
sudo test -d "$STOCKER_V2_RELEASE"
sudo test -x "$STOCKER_V2_RELEASE/.venv/bin/python"
sudo test -x "$STOCKER_TRUSTED_PYTHON"
sudo test -f "$STOCKER_RELEASE_ARTIFACT_VERIFIER"
sudo test ! -L "$STOCKER_RELEASE_ARTIFACT_VERIFIER"
sudo test "$(readlink -f /opt/stocker/current)" != "$STOCKER_V2_RELEASE"
sudo test ! -e /opt/stocker/v2-current
sudo test -d "$STOCKER_IBAPI_SOURCE"
sudo test ! -L "$STOCKER_IBAPI_SOURCE"
sudo test -f "$STOCKER_PROTOBUF_WHEEL"
sudo test ! -L "$STOCKER_PROTOBUF_WHEEL"
sudo test -f "$STOCKER_PROTOBUF_WHEEL_SHA256"
sudo test ! -L "$STOCKER_PROTOBUF_WHEEL_SHA256"
case "${STOCKER_PROTOBUF_WHEEL##*/}" in
  protobuf-5.29.5-*.whl) ;;
  *) exit 78 ;;
esac
sudo awk -v expected="$STOCKER_PROTOBUF_WHEEL" \
  'NR == 1 && length($1) == 64 && $1 !~ /[^0-9a-f]/ && $2 == expected { ok = 1 } END { exit !(NR == 1 && ok) }' \
  "$STOCKER_PROTOBUF_WHEEL_SHA256"
sudo sha256sum --check "$STOCKER_PROTOBUF_WHEEL_SHA256"
sudo test -f "$STOCKER_IBAPI_WHEEL"
sudo test ! -L "$STOCKER_IBAPI_WHEEL"
sudo test -f "$STOCKER_IBAPI_WHEEL_SHA256"
sudo test ! -L "$STOCKER_IBAPI_WHEEL_SHA256"
test "${STOCKER_IBAPI_WHEEL##*/}" = ibapi-10.49.1-py3-none-any.whl || exit 78
sudo awk -v expected="$STOCKER_IBAPI_WHEEL" \
  'NR == 1 && length($1) == 64 && $1 !~ /[^0-9a-f]/ && $2 == expected { ok = 1 } END { exit !(NR == 1 && ok) }' \
  "$STOCKER_IBAPI_WHEEL_SHA256"
sudo sha256sum --check "$STOCKER_IBAPI_WHEEL_SHA256"
sudo test -f "$STOCKER_UV_SHA256"
sudo test ! -L "$STOCKER_UV_SHA256"
sudo awk -v expected="$STOCKER_UV_BIN" \
  'NR == 1 && length($1) == 64 && $1 == "da15297d6879b2cfbe5ea3cb03725c1613d51ba72892cc996468d871f0a532fb" && $2 == expected { ok = 1 } END { exit !(NR == 1 && ok) }' \
  "$STOCKER_UV_SHA256"
sudo /usr/bin/sha256sum --check "$STOCKER_UV_SHA256"
sudo "$STOCKER_TRUSTED_PYTHON" -I - \
  "$STOCKER_RELEASE_ARTIFACT_VERIFIER" "$STOCKER_V2_RELEASE" \
  "$STOCKER_RELEASE_ARTIFACT_VERIFIER_SHA256" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path

verifier = Path(sys.argv[1])
release = Path(sys.argv[2])
expected_hash = sys.argv[3]
acl_names = {"system.posix_acl_access", "system.posix_acl_default"}
listxattr = getattr(os, "listxattr", None)
if listxattr is None:
    raise SystemExit("verifier bootstrap ACL inspection unavailable")


def require_controlled(path: Path, *, directory: bool) -> None:
    metadata = path.lstat()
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    mode = stat.S_IMODE(metadata.st_mode)
    if not expected_type(metadata.st_mode) or metadata.st_uid != 0:
        raise SystemExit("verifier bootstrap identity failure")
    if mode & stat.S_IWOTH or (mode & stat.S_IWGRP and metadata.st_gid != 0):
        raise SystemExit("verifier bootstrap mode failure")
    names = set(listxattr(path, follow_symlinks=False))
    if names & acl_names:
        raise SystemExit("verifier bootstrap ACL failure")


if (
    not verifier.is_absolute()
    or Path(os.path.normpath(verifier)) != verifier
    or Path(os.path.normpath(release)) != release
    or verifier != release / "deploy/scripts/verify_v2_release_artifacts.py"
):
    raise SystemExit("verifier bootstrap path failure")
current = Path("/")
require_controlled(current, directory=True)
for part in verifier.parent.relative_to(current).parts:
    current /= part
    require_controlled(current, directory=True)
require_controlled(verifier, directory=False)
if hashlib.sha256(verifier.read_bytes()).hexdigest() != expected_hash:
    raise SystemExit("verifier bootstrap hash failure")
PY
sudo "$STOCKER_TRUSTED_PYTHON" -I "$STOCKER_RELEASE_ARTIFACT_VERIFIER" preinstall \
  --ibapi-wheel "$STOCKER_IBAPI_WHEEL" \
  --ibapi-manifest "$STOCKER_IBAPI_WHEEL_SHA256" \
  --protobuf-wheel "$STOCKER_PROTOBUF_WHEEL" \
  --protobuf-manifest "$STOCKER_PROTOBUF_WHEEL_SHA256" \
  --uv-bin "$STOCKER_UV_BIN" \
  --uv-manifest "$STOCKER_UV_SHA256" \
  --official-source-root "$STOCKER_IBAPI_SOURCE" \
  --release-root "$STOCKER_V2_RELEASE" \
  --verifier-path "$STOCKER_RELEASE_ARTIFACT_VERIFIER" \
  --verifier-sha256 "$STOCKER_RELEASE_ARTIFACT_VERIFIER_SHA256" \
  --venv "$STOCKER_V2_RELEASE/.venv"
test "$(sudo "$STOCKER_UV_BIN" --version)" = "uv 0.11.32 (x86_64-unknown-linux-gnu)"
sudo "$STOCKER_UV_BIN" pip install --python "$STOCKER_V2_RELEASE/.venv/bin/python" --offline --no-deps --reinstall "$STOCKER_PROTOBUF_WHEEL"
sudo "$STOCKER_UV_BIN" pip install --python "$STOCKER_V2_RELEASE/.venv/bin/python" --offline --no-deps --reinstall "$STOCKER_IBAPI_WHEEL"
sudo "$STOCKER_TRUSTED_PYTHON" -I "$STOCKER_RELEASE_ARTIFACT_VERIFIER" postinstall \
  --ibapi-wheel "$STOCKER_IBAPI_WHEEL" \
  --ibapi-manifest "$STOCKER_IBAPI_WHEEL_SHA256" \
  --protobuf-wheel "$STOCKER_PROTOBUF_WHEEL" \
  --protobuf-manifest "$STOCKER_PROTOBUF_WHEEL_SHA256" \
  --uv-bin "$STOCKER_UV_BIN" \
  --uv-manifest "$STOCKER_UV_SHA256" \
  --official-source-root "$STOCKER_IBAPI_SOURCE" \
  --release-root "$STOCKER_V2_RELEASE" \
  --verifier-path "$STOCKER_RELEASE_ARTIFACT_VERIFIER" \
  --verifier-sha256 "$STOCKER_RELEASE_ARTIFACT_VERIFIER_SHA256" \
  --venv "$STOCKER_V2_RELEASE/.venv"
sudo "$STOCKER_V2_RELEASE/.venv/bin/python" - <<'PY'
from importlib.metadata import requires, version

if "protobuf==5.29.5" not in (requires("ibapi") or ()):
    raise SystemExit("ibapi declared dependency mismatch")
if version("ibapi") != "10.49.1":
    raise SystemExit("ibapi installed version mismatch")
if version("protobuf") != "5.29.5":
    raise SystemExit("protobuf installed version mismatch")
import google.protobuf
from ibapi.client import EClient

if not isinstance(EClient, type):
    raise SystemExit("ibapi concrete client is invalid")
PY
sudo test -x "$STOCKER_V2_RELEASE/.venv/bin/stocker-runtime"
sudo "$STOCKER_V2_RELEASE/.venv/bin/stocker-runtime" ibkr-api verify --provenance "$STOCKER_IBAPI_PROVENANCE"
sudo ln -s "$STOCKER_V2_RELEASE" /opt/stocker/v2-current
sudo test "$(readlink -f /opt/stocker/v2-current)" = "$STOCKER_V2_RELEASE"
STOCKER_V2_RELEASE_PREP

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
sudo install -d -o root -g root -m 0755 /usr/local/libexec
sudo install -o root -g root -m 0755 \
  /opt/stocker/v2-current/deploy/scripts/prepare-v2-sqlite-boundary.py \
  /usr/local/libexec/stocker-prepare-v2-sqlite-boundary
sudo systemctl daemon-reload
sudo systemctl is-enabled stocker-v2-recorder.service stocker-v2-web.service || true
```

Both services and both backup timers must still be disabled. Validate the reviewed
configuration files, their hashes, the release commit, the 8 GiB database/backup caps,
and that there are no broker account, credential, order, paper, or live fields. Install
the V2 web environment only as `/etc/stocker/stocker-v2-web.env`; never overwrite the
preserved V1 `/etc/stocker/stocker-web.env`, which remains part of rollback evidence.

## 2. Quiesce and preserve the distinct V1 sources

Stop and disable all V1 scheduling and application processes before checking any
source. The session-readiness timer has a persistent start and requires the recorder,
so leaving it enabled can restart V1 during import:

```bash
sudo systemctl disable --now stocker-backup.timer \
  stocker-recorder-session-readiness.timer \
  stocker-recorder.service stocker-web.service
sudo systemctl stop stocker-backup.service stocker-recorder-session-readiness.service
sudo systemctl reset-failed stocker-recorder.service stocker-web.service
```

The rollback database is the database named by the stopped recorder and web
configuration; it is not the import source and is never a retirement target. Prove no
process has it open. Checkpoint it as the preserved V1 writer, require both SQLite
checks to pass, hash it into the change record, and require no sidecar:

```bash
id -u stocker >/dev/null 2>&1
sudo test "$(stat -c '%U' \
  /var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3)" = stocker
sudo -u stocker sqlite3 \
  /var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3 \
  'PRAGMA wal_checkpoint(TRUNCATE); PRAGMA quick_check; PRAGMA foreign_key_check;'
sudo lsof -- /var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3
sudo test ! -e \
  /var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3-journal
sudo test ! -e \
  /var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3-wal
sudo test ! -e \
  /var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3-shm
sudo sha256sum /var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3
```

The `lsof` command must print no open handle (its normal no-match exit status is
acceptable). The preserved V1 writer performs the write-requiring checkpoint; the new
V2 recorder identity is only a read-group member at this point. Do not change V1
ownership and do not proceed merely because the service manager reports the unit as
stopped. Preserve its unclosed generation rows unchanged as rollback provenance; do not
close or rewrite them.

Independently verify the selected import snapshot against its manifest. Its SHA-256 is
`9672d119395e9f2946dcec1e1d8bd7a332a0cb031d3be7a56601117db4217d90` and its
frozen schema-0030 digest is
`afb0e2d62ddd671daaaa9fbab7b2100be7efa7649443b47f358acb2f36319d59`.
Require `quick_check=ok`, no foreign-key rows, no `recorder_lease` row, no SQLite
sidecar, and byte identity before and after import. Do not update the snapshot's 254
unclosed generation rows. They are accepted only by the explicit attended assertion in
Section 3 and must be archived by count, canonical full-row digest, bounded per-run
summary, and a resolved migration incident.

Record the pre-change ownership, mode, and ACL, then remove all write bits from the
immutable snapshot. Do not grant the shared `stocker-readers` group access: that group
also contains the web and backup identities, which must never be able to read legacy
callback payloads or protected evidence. Retain `stocker` as owner and group, keep the
backup directory non-world-accessible, and confirm that none of the V2 identities can
read it yet. Section 3 adds and removes a recorder-only ACL immediately around the
import:

```bash
set -o pipefail
export STOCKER_V1_SNAPSHOT=/var/lib/stocker/backups/prospective-20260805T141203Z.sqlite3
export STOCKER_V1_ROLLBACK_DB=/var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3
export STOCKER_V1_RELEASE="$(readlink -f /opt/stocker/current)"
export STOCKER_V1_PRESERVATION=/var/lib/stocker/recovery-v1/REPLACE_WITH_CHANGE_ID-preservation
test "$STOCKER_V1_PRESERVATION" != \
  /var/lib/stocker/recovery-v1/REPLACE_WITH_CHANGE_ID-preservation || exit 78
sudo install -d -o root -g root -m 0700 "$STOCKER_V1_PRESERVATION"
snapshot_acl_before="$(sudo getfacl -p /var/lib/stocker/backups \
  "$STOCKER_V1_SNAPSHOT")" || exit 78
rollback_acl_before="$(sudo getfacl -p "$STOCKER_V1_ROLLBACK_DB")" || exit 78
test -n "$snapshot_acl_before" || exit 78
test -n "$rollback_acl_before" || exit 78
sudo tee "$STOCKER_V1_PRESERVATION/snapshot-access-control-before.txt" \
  >/dev/null <<<"$snapshot_acl_before" || exit 78
sudo tee "$STOCKER_V1_PRESERVATION/rollback-access-control-before.txt" \
  >/dev/null <<<"$rollback_acl_before" || exit 78
sudo test -s "$STOCKER_V1_PRESERVATION/snapshot-access-control-before.txt" || exit 78
sudo test -s "$STOCKER_V1_PRESERVATION/rollback-access-control-before.txt" || exit 78
sudo chmod 0400 "$STOCKER_V1_SNAPSHOT"
sudo -u stocker-recorder test ! -r \
  /var/lib/stocker/backups/prospective-20260805T141203Z.sqlite3
sudo -u stocker-web test ! -r \
  /var/lib/stocker/backups/prospective-20260805T141203Z.sqlite3
sudo -u stocker-backup test ! -r \
  /var/lib/stocker/backups/prospective-20260805T141203Z.sqlite3
echo '9672d119395e9f2946dcec1e1d8bd7a332a0cb031d3be7a56601117db4217d90  /var/lib/stocker/backups/prospective-20260805T141203Z.sqlite3' | \
  sudo sha256sum --check --strict
```

Before deleting the retirement candidate, create and lock a **pre-deletion
preservation manifest**. This first, read-only recovery set binds the existing immutable
snapshot and untouched rollback database by exact path, device, inode, allocated size,
mode, owner, SHA-256, `quick_check`, and `foreign_key_check`. It also contains a
deterministic full-tree hash inventory of the intact current V1 release plus a root-only
archive of its pointer, configuration, and installed unit definitions. Do not duplicate
the multi-gigabyte release before capacity is reclaimed. The two checked database files
and the existing immutable V1 release are themselves preserved members of this set:
all are outside the three retirement paths, and none may be deleted, replaced, or
modified. This evidence-bound set must be complete and synced before deletion; it does
not depend on a copy that can only fit after space is reclaimed.

Use a new, recorded change identifier, preserve the recorded pre-change metadata, and
do not place credentials in the manifest output:

```bash
set -o pipefail
sudo chmod 0400 "$STOCKER_V1_ROLLBACK_DB"
sudo sha256sum "$STOCKER_V1_SNAPSHOT" "$STOCKER_V1_ROLLBACK_DB" | \
  sudo tee "$STOCKER_V1_PRESERVATION/database-sha256.txt" >/dev/null
sudo stat -c '%n|%d|%i|%b|%B|%s|%U|%G|%a' \
  "$STOCKER_V1_SNAPSHOT" "$STOCKER_V1_ROLLBACK_DB" | \
  sudo tee "$STOCKER_V1_PRESERVATION/database-stat.txt" >/dev/null
sudo find "$STOCKER_V1_RELEASE" -xdev -type f -print0 | sudo sort -z | \
  sudo xargs -0 sha256sum | \
  sudo tee "$STOCKER_V1_PRESERVATION/v1-release-sha256.txt" >/dev/null
sudo test -s "$STOCKER_V1_PRESERVATION/v1-release-sha256.txt" || exit 78
sudo sha256sum "$STOCKER_V1_PRESERVATION/v1-release-sha256.txt" | \
  sudo tee "$STOCKER_V1_PRESERVATION/v1-release-manifest-sha256.txt" >/dev/null
sudo tar --create --gzip \
  --file "$STOCKER_V1_PRESERVATION/v1-control-plane.tar.gz" -- \
  /opt/stocker/current /etc/stocker \
  /etc/systemd/system/stocker-recorder.service \
  /etc/systemd/system/stocker-web.service \
  /etc/systemd/system/stocker-backup.service \
  /etc/systemd/system/stocker-backup.timer \
  /etc/systemd/system/stocker-recorder-session-readiness.service \
  /etc/systemd/system/stocker-recorder-session-readiness.timer
sudo sha256sum "$STOCKER_V1_PRESERVATION/v1-control-plane.tar.gz" | \
  sudo tee "$STOCKER_V1_PRESERVATION/control-plane-sha256.txt" >/dev/null
{
  echo snapshot
  sudo -u stocker sqlite3 -readonly \
    "file:$STOCKER_V1_SNAPSHOT?mode=ro&immutable=1" \
    'PRAGMA quick_check; PRAGMA foreign_key_check;'
  echo rollback
  sudo -u stocker sqlite3 -readonly \
    "file:$STOCKER_V1_ROLLBACK_DB?mode=ro&immutable=1" \
    'PRAGMA quick_check; PRAGMA foreign_key_check;'
} | sudo tee "$STOCKER_V1_PRESERVATION/sqlite-integrity.txt" >/dev/null
for checked_db in "$STOCKER_V1_SNAPSHOT" "$STOCKER_V1_ROLLBACK_DB"; do
  sudo test ! -e "${checked_db}-journal" || exit 78
  sudo test ! -e "${checked_db}-wal" || exit 78
  sudo test ! -e "${checked_db}-shm" || exit 78
done
sudo sha256sum --check --strict \
  "$STOCKER_V1_PRESERVATION/database-sha256.txt"
sudo chmod 0440 "$STOCKER_V1_PRESERVATION/database-sha256.txt" \
  "$STOCKER_V1_PRESERVATION/database-stat.txt" \
  "$STOCKER_V1_PRESERVATION/control-plane-sha256.txt" \
  "$STOCKER_V1_PRESERVATION/v1-release-sha256.txt" \
  "$STOCKER_V1_PRESERVATION/v1-release-manifest-sha256.txt" \
  "$STOCKER_V1_PRESERVATION/snapshot-access-control-before.txt" \
  "$STOCKER_V1_PRESERVATION/rollback-access-control-before.txt" \
  "$STOCKER_V1_PRESERVATION/sqlite-integrity.txt" \
  "$STOCKER_V1_PRESERVATION/v1-control-plane.tar.gz"
sudo chmod 0550 "$STOCKER_V1_PRESERVATION"
sudo sync -f "$STOCKER_V1_PRESERVATION"
```

Require `sqlite-integrity.txt` to contain exactly the two labels and one `ok` line for
each database, with no foreign-key rows. `-readonly` alone is insufficient for a frozen
WAL-mode database because SQLite may still materialise WAL/SHM beside it; every
observational check therefore uses the explicit `mode=ro&immutable=1` URI and then
reasserts that no journal, WAL, or SHM exists. Compare the hashes and stat identity with
the locked manifest. Prove that no process has any of the three retirement paths open, that
the V1 app, backup, and session-readiness units are inactive and disabled, and that
every remaining reference to the candidate is confined to preserved disabled V1
material. Record that dependency inventory. Then, and only after the preservation,
no-handle, dependency, and separately recorded owner-approval gates all pass, remove
exactly these paths without a glob. A plain `lsof` invocation is not a gate: it exits
zero when it finds the unsafe state. Fence the stopped V1 surface with exact runtime
drop-ins; do not move, replace, or overwrite any preserved `/etc/systemd/system` unit.
`RefuseManualStart=yes` rejects operator starts, while an absent
`ConditionPathExists` sentinel blocks dependency activation. Verify the properties
parsed by systemd, the exact regular drop-in content, and the false condition result.
Do not use `systemctl show Conditions` for this gate: systemd 255 can serialize that
property as `[unprintable]` while returning success, and a standalone
`systemd-analyze condition` does not load the merged unit. For each unit, require
`systemd-analyze verify` to return status 0, then require
`systemd-analyze condition --unit` to return exactly status 1 with the C-locale output
line naming the failed sentinel condition. A true condition, command error, or another
false condition cannot satisfy the gate. Prove an attempted start fails and every unit
remains inactive, then invoke `rm` only from the guarded function. On any failure,
leave the drop-ins in place:

```bash
STOCKER_V1_FENCE_SENTINEL=/run/stocker-v1-cutover-start-authorised
STOCKER_V1_FENCE_NAME=99-stocker-v1-cutover-start-fence.conf
STOCKER_V1_UNITS="stocker-recorder.service stocker-web.service
stocker-backup.service stocker-backup.timer
stocker-recorder-session-readiness.service stocker-recorder-session-readiness.timer"

install_v1_runtime_start_fence() {
  local unit dropin_directory dropin active_state fragment_path refuse_manual
  local dropin_paths actual_fence expected_fence condition_output condition_status
  sudo test ! -e "$STOCKER_V1_FENCE_SENTINEL" || return 78
  expected_fence="$(printf '%s\n' \
    '[Unit]' \
    'RefuseManualStart=yes' \
    "ConditionPathExists=$STOCKER_V1_FENCE_SENTINEL")"
  for unit in $STOCKER_V1_UNITS; do
    sudo test -f "/etc/systemd/system/$unit" || return 78
    sudo test ! -L "/etc/systemd/system/$unit" || return 78
    dropin_directory="/run/systemd/system/${unit}.d"
    dropin="$dropin_directory/$STOCKER_V1_FENCE_NAME"
    sudo test ! -e "$dropin" || return 78
    sudo install -d -o root -g root -m 0755 "$dropin_directory" || return 78
    printf '%s\n' \
      '[Unit]' \
      'RefuseManualStart=yes' \
      "ConditionPathExists=$STOCKER_V1_FENCE_SENTINEL" | \
      sudo tee "$dropin" >/dev/null || return 78
    sudo chmod 0644 "$dropin" || return 78
  done
  sudo systemctl daemon-reload || return 78
  for unit in $STOCKER_V1_UNITS; do
    dropin="/run/systemd/system/${unit}.d/$STOCKER_V1_FENCE_NAME"
    sudo test -f "$dropin" || return 78
    sudo test ! -L "$dropin" || return 78
    actual_fence="$(sudo cat "$dropin")" || return 78
    test "$actual_fence" = "$expected_fence" || return 78
    LC_ALL=C systemd-analyze verify "$unit" >/dev/null 2>&1 || return 78
    if condition_output="$(
      LC_ALL=C systemd-analyze condition --unit="$unit" 2>&1
    )"; then
      echo "refusing retirement: merged V1 fence condition is true: $unit" >&2
      return 78
    else
      condition_status=$?
    fi
    test "$condition_status" -eq 1 || {
      echo "refusing retirement: cannot evaluate merged V1 condition for $unit" >&2
      printf '%s\n' "$condition_output" >&2
      return 78
    }
    case "$condition_output" in
      *"ConditionPathExists=$STOCKER_V1_FENCE_SENTINEL failed."*) ;;
      *)
        echo "refusing retirement: sentinel condition did not fail for $unit" >&2
        printf '%s\n' "$condition_output" >&2
        return 78
        ;;
    esac
    active_state="$(sudo systemctl show --property=ActiveState --value "$unit")" || return 78
    fragment_path="$(sudo systemctl show --property=FragmentPath --value "$unit")" || return 78
    refuse_manual="$(sudo systemctl show --property=RefuseManualStart --value "$unit")" || \
      return 78
    dropin_paths="$(sudo systemctl show --property=DropInPaths --value "$unit")" || return 78
    test "$active_state" = inactive || return 78
    test "$fragment_path" = "/etc/systemd/system/$unit" || return 78
    test "$refuse_manual" = yes || return 78
    case "$dropin_paths" in
      *"/run/systemd/system/${unit}.d/$STOCKER_V1_FENCE_NAME"*) ;;
      *) return 78 ;;
    esac
    if sudo systemctl start "$unit"; then
      echo "refusing retirement: fenced V1 unit started: $unit" >&2
      return 78
    fi
    active_state="$(sudo systemctl show --property=ActiveState --value "$unit")" || return 78
    test "$active_state" = inactive || return 78
  done
}

retire_authorised_v1_candidate() {
  local handle_report lsof_status
  install_v1_runtime_start_fence || return 78
  sudo test -f /var/lib/stocker/prospective/prospective.sqlite3 || return 78
  sudo test -f /var/lib/stocker/prospective/prospective.sqlite3-wal || return 78
  sudo test -f /var/lib/stocker/prospective/prospective.sqlite3-shm || return 78
  if handle_report="$(sudo lsof -Fn -- \
    /var/lib/stocker/prospective/prospective.sqlite3 \
    /var/lib/stocker/prospective/prospective.sqlite3-wal \
    /var/lib/stocker/prospective/prospective.sqlite3-shm 2>&1)"; then
    echo "refusing retirement: open V1 handle detected" >&2
    printf '%s\n' "$handle_report" >&2
    return 78
  else
    lsof_status=$?
  fi
  if test "$lsof_status" -ne 1 || test -n "$handle_report"; then
    echo "refusing retirement: lsof could not prove an empty handle set" >&2
    printf '%s\n' "$handle_report" >&2
    return 78
  fi
  sudo rm -- /var/lib/stocker/prospective/prospective.sqlite3 \
    /var/lib/stocker/prospective/prospective.sqlite3-wal \
    /var/lib/stocker/prospective/prospective.sqlite3-shm
}
retire_authorised_v1_candidate
```

The removal is irreversible and may lose evidence unique to that aggregate. It does
not remove either preserved database member. Recheck allocated free space immediately.
Using only the reclaimed capacity, augment the pre-deletion set with **checked
compressed recovery copies** of both databases and a second manifest. Restore each copy
to a disposable path, compare its SHA-256 with the corresponding original, and require
`quick_check=ok` plus no `foreign_key_check` rows. Remove only the named disposable
restores, remove write bits from the compressed-copy set, and sync it. All of this must
finish before the importer in Section 3 runs. Preserve the original snapshot, original
rollback database, preservation manifest, and compressed recovery copies until Michael
closes the seven-day rollback window. Do not copy V1 credentials or vendor tokens into
V2 configuration.

Run every command below fail-closed and stop on any nonzero status:

```bash
set -o pipefail
export STOCKER_V1_RECOVERY_COPIES=/var/lib/stocker/recovery-v1/REPLACE_WITH_CHANGE_ID-copies
test "$STOCKER_V1_RECOVERY_COPIES" != \
  /var/lib/stocker/recovery-v1/REPLACE_WITH_CHANGE_ID-copies || exit 78
sudo install -d -o root -g root -m 0700 "$STOCKER_V1_RECOVERY_COPIES"
sudo gzip --stdout "$STOCKER_V1_SNAPSHOT" | sudo tee \
  "$STOCKER_V1_RECOVERY_COPIES/import-snapshot.sqlite3.gz" >/dev/null
sudo gzip --stdout "$STOCKER_V1_ROLLBACK_DB" | sudo tee \
  "$STOCKER_V1_RECOVERY_COPIES/rollback.sqlite3.gz" >/dev/null
sudo gzip --test "$STOCKER_V1_RECOVERY_COPIES/import-snapshot.sqlite3.gz"
sudo gzip --test "$STOCKER_V1_RECOVERY_COPIES/rollback.sqlite3.gz"
sudo install -d -o root -g root -m 0700 \
  "$STOCKER_V1_RECOVERY_COPIES/restore-check"
sudo gzip --decompress --stdout \
  "$STOCKER_V1_RECOVERY_COPIES/import-snapshot.sqlite3.gz" | sudo tee \
  "$STOCKER_V1_RECOVERY_COPIES/restore-check/import-snapshot.sqlite3" >/dev/null
sudo gzip --decompress --stdout \
  "$STOCKER_V1_RECOVERY_COPIES/rollback.sqlite3.gz" | sudo tee \
  "$STOCKER_V1_RECOVERY_COPIES/restore-check/rollback.sqlite3" >/dev/null
sudo cmp --silent "$STOCKER_V1_SNAPSHOT" \
  "$STOCKER_V1_RECOVERY_COPIES/restore-check/import-snapshot.sqlite3"
sudo cmp --silent "$STOCKER_V1_ROLLBACK_DB" \
  "$STOCKER_V1_RECOVERY_COPIES/restore-check/rollback.sqlite3"
{
  echo snapshot_restore
  sudo sqlite3 -readonly \
    "file:$STOCKER_V1_RECOVERY_COPIES/restore-check/import-snapshot.sqlite3?mode=ro&immutable=1" \
    'PRAGMA quick_check; PRAGMA foreign_key_check;'
  echo rollback_restore
  sudo sqlite3 -readonly \
    "file:$STOCKER_V1_RECOVERY_COPIES/restore-check/rollback.sqlite3?mode=ro&immutable=1" \
    'PRAGMA quick_check; PRAGMA foreign_key_check;'
} | sudo tee "$STOCKER_V1_RECOVERY_COPIES/restore-integrity.txt" >/dev/null
for checked_restore in \
  "$STOCKER_V1_RECOVERY_COPIES/restore-check/import-snapshot.sqlite3" \
  "$STOCKER_V1_RECOVERY_COPIES/restore-check/rollback.sqlite3"; do
  sudo test ! -e "${checked_restore}-journal" || exit 78
  sudo test ! -e "${checked_restore}-wal" || exit 78
  sudo test ! -e "${checked_restore}-shm" || exit 78
done
sudo sha256sum "$STOCKER_V1_RECOVERY_COPIES/import-snapshot.sqlite3.gz" \
  "$STOCKER_V1_RECOVERY_COPIES/rollback.sqlite3.gz" | sudo tee \
  "$STOCKER_V1_RECOVERY_COPIES/archive-sha256.txt" >/dev/null
sudo rm -- "$STOCKER_V1_RECOVERY_COPIES/restore-check/import-snapshot.sqlite3" \
  "$STOCKER_V1_RECOVERY_COPIES/restore-check/rollback.sqlite3"
sudo rmdir "$STOCKER_V1_RECOVERY_COPIES/restore-check"
sudo chmod 0440 "$STOCKER_V1_RECOVERY_COPIES/import-snapshot.sqlite3.gz" \
  "$STOCKER_V1_RECOVERY_COPIES/rollback.sqlite3.gz" \
  "$STOCKER_V1_RECOVERY_COPIES/restore-integrity.txt" \
  "$STOCKER_V1_RECOVERY_COPIES/archive-sha256.txt"
sudo chmod 0550 "$STOCKER_V1_RECOVERY_COPIES"
sudo sync -f "$STOCKER_V1_RECOVERY_COPIES"
```

## 3. One-way import into a new V2 target

The target and its reconciliation file must not exist. The importer opens the selected
snapshot with SQLite `mode=ro&immutable=1`, accepts only exact frozen schema prefixes
through `0030`, and writes a new temporary V2 database in the target directory. The
following flag is a narrow attended assertion about preserved unclosed generation
evidence; it never permits a lease, journal, WAL, or SHM:

```bash
revoke_v1_snapshot_access() {
  local snapshot_acl backup_acl
  snapshot_acl="$(sudo getfacl -cp "$STOCKER_V1_SNAPSHOT")" || return 78
  case "$snapshot_acl" in
    *"user:stocker-recorder:"*)
      sudo setfacl -x u:stocker-recorder "$STOCKER_V1_SNAPSHOT" || return 78
      ;;
  esac
  backup_acl="$(sudo getfacl -cp /var/lib/stocker/backups)" || return 78
  case "$backup_acl" in
    *"user:stocker-recorder:"*)
      sudo setfacl -x u:stocker-recorder /var/lib/stocker/backups || return 78
      ;;
  esac
  sudo chmod 0400 "$STOCKER_V1_SNAPSHOT" || return 78
  snapshot_acl="$(sudo getfacl -cp "$STOCKER_V1_SNAPSHOT")" || return 78
  backup_acl="$(sudo getfacl -cp /var/lib/stocker/backups)" || return 78
  case "$snapshot_acl$backup_acl" in
    *"user:stocker-recorder:"*) return 78 ;;
  esac
  sudo -u stocker-recorder test ! -r "$STOCKER_V1_SNAPSHOT" || return 78
  sudo -u stocker-web test ! -r "$STOCKER_V1_SNAPSHOT" || return 78
  sudo -u stocker-backup test ! -r "$STOCKER_V1_SNAPSHOT" || return 78
}
fail_after_snapshot_revoke() {
  if ! revoke_v1_snapshot_access; then
    echo "cutover stopped: legacy snapshot ACL cleanup failed" >&2
  fi
  exit 78
}
sudo setfacl -m u:stocker-recorder:--x /var/lib/stocker/backups || \
  fail_after_snapshot_revoke
sudo setfacl -m u:stocker-recorder:r-- "$STOCKER_V1_SNAPSHOT" || \
  fail_after_snapshot_revoke
sudo -u stocker-recorder test -r "$STOCKER_V1_SNAPSHOT" || \
  fail_after_snapshot_revoke
sudo -u stocker-web test ! -r "$STOCKER_V1_SNAPSHOT" || \
  fail_after_snapshot_revoke
sudo -u stocker-backup test ! -r "$STOCKER_V1_SNAPSHOT" || \
  fail_after_snapshot_revoke
if ! sudo -u stocker-recorder \
  /opt/stocker/v2-current/.venv/bin/stocker-runtime legacy import \
    --source /var/lib/stocker/backups/prospective-20260805T141203Z.sqlite3 \
    --target /var/lib/stocker/v2/stocker-v2.sqlite3 \
    --accept-quiescent-unclean-generations; then
  fail_after_snapshot_revoke
fi
revoke_v1_snapshot_access || exit 78
sudo /usr/local/libexec/stocker-prepare-v2-sqlite-boundary
```

The temporary source ACL is revoked on both success and failure. Do not retry, run the
boundary verifier, or start web/backup while `stocker-recorder` can still read the
legacy snapshot. The setgid V2 directory makes the imported `0640` target inherit
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
- the quiescent-unclean archive records 254 rows, its canonical full-row digest, its
  bounded per-run summary, no source mutation, and the matching resolved migration
  incident;
- the manifest reconciliation digest and target digest match the sidecar;
- `PRAGMA quick_check` returns `ok` and `PRAGMA foreign_key_check` returns no rows;
- the target is owned by `stocker-recorder:stocker-readers`, mode `0640`, and no larger
  than 8 GiB; and
- imported runs preserve protected data classes and are stopped.

Do not proceed if the source hash changed, a lease or SQLite sidecar appeared, any row
is unreconciled, or
the target contains an account, order, fill, broker-position, transfer, report-package,
or legacy vendor runtime surface.

## 4. Install the V2 service surface beside the stopped V1 surface

Install the reviewed V2 units under their exact names without overwriting or removing
the installed V1 Stocker application unit files. Do not alias old names to new
services. Keep every V1 application unit stopped and disabled, and keep its release and
configuration intact for rollback. Install V2 configuration with root ownership and
least-privilege read access. Confirm:

```bash
systemctl list-unit-files 'stocker*' --no-pager
systemctl is-active stocker-recorder.service stocker-web.service stocker-backup.timer
systemctl is-enabled stocker-recorder.service stocker-web.service stocker-backup.timer
systemctl is-enabled stocker-v2-recorder.service stocker-v2-web.service \
  stocker-v2-backup-daily.timer stocker-v2-backup-weekly.timer
```

The three V1 checks must report inactive and disabled; their unit definitions remain
installed until the owner closes the rollback window. The four V2 checks must remain
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
sudo -u stocker-recorder /opt/stocker/v2-current/.venv/bin/stocker-runtime ibkr-api verify \
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
and weekly timers and declare V2 admission operational. This does not close the
rollback window: `/opt/stocker/current`, the installed V1 units, the complete V1
release, its untouched database, and both recovery sets remain preserved.

## 8. Rollback

For either rollback path, first verify and remove only the exact V1 runtime-fence
drop-ins, reload systemd, and prove the preserved regular unit files are again parsed
from `/etc/systemd/system` without the fence. Do not remove another drop-in or unit
file. Any unexpected fence type/content, failed removal, stale parsed fence, or
state-query failure stops rollback before any V1 start attempt. Removing some files
cannot weaken the live fence before the one final successful `daemon-reload`; a failed
cleanup therefore leaves systemd's already parsed fence in force. After the reload,
syntax-check each merged unit and inspect `systemctl cat` output for the absence of
the sentinel; do not require unrelated normal unit conditions to be true:

```bash
export STOCKER_V1_ROLLBACK_DB=/var/lib/stocker/prospective/prospective-20260806t163100z.sqlite3
export STOCKER_V1_PRESERVATION=/var/lib/stocker/recovery-v1/REPLACE_WITH_CHANGE_ID-preservation
test "$STOCKER_V1_PRESERVATION" != \
  /var/lib/stocker/recovery-v1/REPLACE_WITH_CHANGE_ID-preservation || exit 78
STOCKER_V1_FENCE_SENTINEL=/run/stocker-v1-cutover-start-authorised
STOCKER_V1_FENCE_NAME=99-stocker-v1-cutover-start-fence.conf
STOCKER_V1_UNITS="stocker-recorder.service stocker-web.service
stocker-backup.service stocker-backup.timer
stocker-recorder-session-readiness.service stocker-recorder-session-readiness.timer"
sudo test ! -e "$STOCKER_V1_FENCE_SENTINEL" || exit 78
expected_fence="$(printf '%s\n' \
  '[Unit]' \
  'RefuseManualStart=yes' \
  "ConditionPathExists=$STOCKER_V1_FENCE_SENTINEL")"
for unit in $STOCKER_V1_UNITS; do
  dropin="/run/systemd/system/${unit}.d/$STOCKER_V1_FENCE_NAME"
  if sudo test -e "$dropin"; then
    sudo test -f "$dropin" || exit 78
    sudo test ! -L "$dropin" || exit 78
    actual_fence="$(sudo cat "$dropin")" || exit 78
    test "$actual_fence" = "$expected_fence" || exit 78
  fi
  sudo test -f "/etc/systemd/system/$unit" || exit 78
  sudo test ! -L "/etc/systemd/system/$unit" || exit 78
  active_state="$(sudo systemctl show --property=ActiveState --value "$unit")" || exit 78
  test "$active_state" = inactive || exit 78
done
for unit in $STOCKER_V1_UNITS; do
  dropin_directory="/run/systemd/system/${unit}.d"
  dropin="$dropin_directory/$STOCKER_V1_FENCE_NAME"
  if sudo test -e "$dropin"; then
    sudo rm -- "$dropin" || exit 78
  fi
done
sudo systemctl daemon-reload || exit 78
for unit in $STOCKER_V1_UNITS; do
  LC_ALL=C systemd-analyze verify "$unit" >/dev/null 2>&1 || exit 78
  merged_unit="$(sudo systemctl cat "$unit")" || exit 78
  case "$merged_unit" in
    *"$STOCKER_V1_FENCE_SENTINEL"*) exit 78 ;;
  esac
  active_state="$(sudo systemctl show --property=ActiveState --value "$unit")" || exit 78
  load_state="$(sudo systemctl show --property=LoadState --value "$unit")" || exit 78
  fragment_path="$(sudo systemctl show --property=FragmentPath --value "$unit")" || exit 78
  refuse_manual="$(sudo systemctl show --property=RefuseManualStart --value "$unit")" || exit 78
  dropin_paths="$(sudo systemctl show --property=DropInPaths --value "$unit")" || exit 78
  test "$active_state" = inactive || exit 78
  test "$load_state" = loaded || exit 78
  test "$fragment_path" = "/etc/systemd/system/$unit" || exit 78
  test "$refuse_manual" = no || exit 78
  case "$dropin_paths" in
    *"/run/systemd/system/${unit}.d/$STOCKER_V1_FENCE_NAME"*) exit 78 ;;
  esac
done
sudo setfacl --restore="$STOCKER_V1_PRESERVATION/rollback-access-control-before.txt" || exit 78
sudo -u stocker test -w "$STOCKER_V1_ROLLBACK_DB" || exit 78
```

### Before first callback

If V2 has not admitted its first callback, stop V2, preserve its logs and failed import
artifacts, restore the prior release/service pointer, and restart the untouched V1
database. Restore the rollback database's recorded V1 owner, group, mode, and ACL from
the common rollback gate before starting its sole V1 writer. Do not copy any V2 row into
V1.

### After first callback

If V2 has admitted data, stop V2 and take a checked V2 backup first. Restore the prior
release against the untouched V1 database, create a new V1 run and recorder generation,
and record the entire V2 interval as an explicit gap. Never reverse-import V2 into V1
or claim uninterrupted scientific continuity. Prefer roll-forward repair.

In both paths, preserve both recovery sets and releases until the owner closes the
rollback window.

## 9. Owner-only rollback-window closure

Closing the rollback window is a separate recorded owner decision made only after the
observation window and a checked V2 backup and restore have succeeded. It is not an
automatic consequence of starting V2. Only after that decision may the operator:

1. atomically repoint `/opt/stocker/current` to the exact release already selected by
   `/opt/stocker/v2-current` and verify that both resolve to the same immutable path;
2. remove the stopped V1 application unit definitions and V1-only configuration, then
   run `systemctl daemon-reload`; and
3. retire the versioned V1 release under the approved recovery-retention policy.

Never remove the V1 units, release, database, or recovery set, and never repoint
`/opt/stocker/current`, before the owner records this decision. The V2 release contains
no compatibility copy of V1.

## 10. Market-data-only IB Gateway boundary

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
sudo /opt/stocker/v2-current/deploy/scripts/verify-ibgateway-installation.sh
sudo install -d -o root -g ibgateway -m 0710 /etc/ibgateway
```

Use the provided X11/VNC units for attended login; VNC is loopback-only and uses the
pre-provisioned `/var/lib/ibgateway/vnc.pass`. Stocker never creates or reads it.
