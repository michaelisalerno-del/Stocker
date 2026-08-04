# Dedicated-server runbook: prospective evidence recorder

This runbook prepares a separate Linux server. It does not authorize deployment
to any existing host. Run the server-side sections only after explicitly
choosing the target server and prospective run.

The application records prospective evidence. It does not place paper or live
orders and does not establish an options edge.

## Fixed layout

```text
/opt/stocker/releases/<git-commit>/    immutable application releases
/opt/stocker/current                  atomic symlink to one release
/etc/stocker/prospective.yaml         runtime config (root:stocker-readers 0640)
/etc/stocker/stocker.env              recorder identity + secrets (root:stocker 0640)
/etc/stocker/stocker-web.env          web identity (root:stocker-readers 0640)
/var/lib/stocker/prospective/         SQLite database and WAL
/var/lib/stocker/bundles/             immutable bundle store + active pointer
/var/lib/stocker/daily-context/       signed package store + session pointers
/var/lib/stocker/ibkr-api/provenance/ immutable official-archive records
/var/lib/stocker/ibkr-api/status/     read-only release-check result for the app
/var/lib/stocker/secure-transfer/     restricted operator transfer staging
/var/lib/stocker/backups/             checked immutable SQLite backups
/var/lib/ibgateway/                   isolated Gateway home, GUI settings, auth
/var/lib/ibgateway/installers/        reviewed installer + immutable provenance
/etc/ibgateway/vnc-password           root-only maintenance credential
/opt/ibgateway/releases/<identity>/   versioned official Gateway installations
/opt/ibgateway/current                atomic pointer to one Gateway installation
```

Application rollback changes only `/opt/stocker/current`. It never replaces,
restores, or rolls back `/var/lib/stocker`.

## 1. Prepare the service user and directories

On the dedicated server:

```bash
sudo groupadd --system stocker
sudo groupadd --system stocker-readers
sudo useradd --system --gid stocker --home-dir /var/lib/stocker \
  --shell /usr/sbin/nologin stocker
sudo usermod --append --groups stocker-readers stocker
sudo useradd --system --home-dir /nonexistent --no-create-home \
  --gid stocker-readers --shell /usr/sbin/nologin stocker-web
sudo install -d -o root -g root -m 0755 /opt/stocker/releases
sudo install -d -o root -g stocker-readers -m 0750 /etc/stocker
sudo install -d -o root -g stocker-readers -m 0750 /var/lib/stocker
sudo install -d -o stocker -g stocker-readers -m 2750 \
  /var/lib/stocker/prospective
sudo install -d -o stocker -g stocker-readers -m 2750 \
  /var/lib/stocker/bundles
sudo install -d -o stocker -g stocker -m 0750 /var/lib/stocker/daily-context/incoming
sudo install -d -o root -g stocker-readers -m 0750 /var/lib/stocker/ibkr-api
sudo install -d -o root -g stocker-readers -m 0750 \
  /var/lib/stocker/ibkr-api/provenance
sudo install -d -o stocker -g stocker-readers -m 2750 \
  /var/lib/stocker/ibkr-api/status
sudo install -d -o root -g stocker -m 0750 /var/lib/stocker/secure-transfer
sudo install -d -o stocker -g stocker -m 0700 /var/lib/stocker/backups
```

Do not add `stocker` to privileged groups. Keep the IBKR GUI/session under an
independently managed desktop login if required by the host. Do not make
`stocker-web` the database owner and do not add it to `stocker` or any
privileged group. Its only group is the non-secret `stocker-readers` group.

### 1a. Migrate an existing installation to the reader boundary

For an existing single-`stocker` installation, first stage the new versioned
release through Sections 2–4, but stop before changing `/opt/stocker/current`.
Then stop both application services and install the migration helper from that
staged release. These commands are idempotent and do not replace the database,
bundles, or evidence records:

```bash
STAGED_RELEASE=/opt/stocker/releases/REPLACE_WITH_GIT_COMMIT
sudo test -x "$STAGED_RELEASE/deploy/scripts/prepare-web-sqlite-boundary.py"
sudo systemctl stop stocker-web.service stocker-recorder.service
getent group stocker-readers >/dev/null ||
  sudo groupadd --system stocker-readers
sudo usermod --append --groups stocker-readers stocker
if id -u stocker-web >/dev/null 2>&1; then
  sudo usermod --gid stocker-readers stocker-web
else
  sudo useradd --system --home-dir /nonexistent --no-create-home \
    --gid stocker-readers --shell /usr/sbin/nologin stocker-web
fi

sudo install -o root -g root -m 0755 \
  "$STAGED_RELEASE/deploy/scripts/prepare-web-sqlite-boundary.py" \
  /usr/local/libexec/stocker-prepare-web-sqlite-boundary

sudo chown root:stocker-readers /etc/stocker
sudo chmod 0750 /etc/stocker
sudo chown root:stocker-readers /etc/stocker/prospective.yaml
sudo chown root:stocker /etc/stocker/stocker.env
sudo chown root:stocker-readers /etc/stocker/stocker-web.env
sudo chmod 0640 \
  /etc/stocker/prospective.yaml \
  /etc/stocker/stocker.env \
  /etc/stocker/stocker-web.env

sudo /usr/local/libexec/stocker-prepare-web-sqlite-boundary \
  --migrate-existing

sudo chown root:stocker-readers \
  /var/lib/stocker/ibkr-api \
  /var/lib/stocker/ibkr-api/provenance
sudo chmod 0750 \
  /var/lib/stocker/ibkr-api \
  /var/lib/stocker/ibkr-api/provenance
sudo find /var/lib/stocker/ibkr-api/provenance -xdev -type f \
  -exec chown root:stocker-readers {} + \
  -exec chmod 0640 {} +
sudo chown stocker:stocker-readers /var/lib/stocker/ibkr-api/status
sudo chmod 2750 /var/lib/stocker/ibkr-api/status
sudo find /var/lib/stocker/ibkr-api/status -xdev -type f \
  -exec chown stocker:stocker-readers {} + \
  -exec chmod 0640 {} +

sudo -u stocker-web -g stocker-readers test ! -r /etc/stocker/stocker.env
sudo -u stocker-web -g stocker-readers test ! -w \
  /var/lib/stocker/prospective/prospective.sqlite3
sudo -u stocker-web -g stocker-readers test -x /var/lib/stocker/bundles
if sudo test -d /var/lib/stocker/bundles/installed; then
  sudo -u stocker-web -g stocker-readers test -x \
    /var/lib/stocker/bundles/installed
fi
if sudo test -f /var/lib/stocker/bundles/active.json; then
  sudo -u stocker-web -g stocker-readers test -r \
    /var/lib/stocker/bundles/active.json
fi
```

Do not recursively change `/var/lib/stocker/secure-transfer`,
`/var/lib/stocker/daily-context`, or `/var/lib/stocker/backups`; those remain
recorder/operator-private. The migration helper opens the recorder-writable
persistent root, bundle store, installed bundle tree, control files, database
directory, database, WAL, and SHM by descriptor with `O_NOFOLLOW`; it refuses
unexpected owners, entries, file types, links, or groups before applying
ownership and mode changes. Installed-bundle files become reader-group
read-only, while `active.json` and `operator-actions.jsonl` remain
recorder-owned and reader-readable.

## 2. Install Python and `uv`

Install the repository's required CPython 3.12 and `uv` using the server's
approved package/software process. Verify, rather than silently accepting a
different Python:

```bash
python3.12 --version
uv --version
```

Inside each copied release:

```bash
cd /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT
sudo -u stocker uv sync --locked --no-editable --no-default-groups --group server
```

Keep the staging release owned by `stocker` through official-API installation
and verification. Section 4 transfers it to root and removes write access only
after every release-local dependency is complete. Production data does not
enter the virtual environment.

## 3. Install the optional official IBKR Python API

Recheck the official download page at install time:
<https://interactivebrokers.github.io/>.

Do not run `pip install ibapi` from a package registry. On 2026-08-03, Python
was provided in the official **Latest** Mac/Unix archive. The current verified
server archive is `twsapi_macunix.1049.01.zip`, which contains
`API_Version=10.49.01` and installs `ibapi==10.49.1`. Its expected SHA-256 is
`f5d31e05f63be0d0fddc13ea8267c3a1625b0783baa17a44832e9151f8402b27`.

The operator must accept IBKR's licence and download the archive manually.
Copy it directly into restricted server staging:

```bash
scp /path/to/twsapi_macunix.1049.01.zip \
  root@SERVER:/var/lib/stocker/secure-transfer/twsapi_macunix.1049.01.zip
```

Then verify it and install it into the **staged release** before that release is
made immutable or promoted:

```bash
sudo chown root:stocker \
  /var/lib/stocker/secure-transfer/twsapi_macunix.1049.01.zip
sudo chmod 0640 \
  /var/lib/stocker/secure-transfer/twsapi_macunix.1049.01.zip
sha256sum /var/lib/stocker/secure-transfer/twsapi_macunix.1049.01.zip
IBKR_EXTRACT_DIR="$(
  sudo -u stocker mktemp -d \
    /var/lib/stocker/secure-transfer/ibkr-api-extract.XXXXXX
)"
sudo -u stocker python3.12 -m zipfile -e \
  /var/lib/stocker/secure-transfer/twsapi_macunix.1049.01.zip \
  "$IBKR_EXTRACT_DIR"
cat "$IBKR_EXTRACT_DIR/IBJts/API_VersionNum.txt"
sudo -u stocker uv --no-config pip install \
  --python /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT/.venv/bin/python \
  "$IBKR_EXTRACT_DIR/IBJts/source/pythonclient"
sudo -u stocker rm -rf -- "$IBKR_EXTRACT_DIR"
/opt/stocker/releases/REPLACE_WITH_GIT_COMMIT/.venv/bin/python \
  -c 'import ibapi; print(ibapi.__version__, ibapi.__file__)'
```

Register provenance with the same staged release. Registration fetches only
IBKR's current official release metadata; it refuses an archive that is no
longer the advertised Latest Mac/Unix release or whose installed source tree
differs:

```bash
IBKR_PACKAGE_ROOT="$(
  /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT/.venv/bin/python \
  -c 'import pathlib, ibapi; print(pathlib.Path(ibapi.__file__).parent)'
)"
sudo /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT/.venv/bin/stocker-prospective \
  ibkr-api register \
  --archive /var/lib/stocker/secure-transfer/twsapi_macunix.1049.01.zip \
  --installed-package-root "$IBKR_PACKAGE_ROOT" \
  --provenance /var/lib/stocker/ibkr-api/provenance/10.49.1.json \
  --operator REPLACE_WITH_OPERATOR_ID
sudo chown root:stocker-readers \
  /var/lib/stocker/ibkr-api/provenance/10.49.1.json
sudo chmod 0640 /var/lib/stocker/ibkr-api/provenance/10.49.1.json
sudo ln -s provenance/10.49.1.json \
  /var/lib/stocker/ibkr-api/active-provenance.json.next
sudo mv -Tf /var/lib/stocker/ibkr-api/active-provenance.json.next \
  /var/lib/stocker/ibkr-api/active-provenance.json
```

The active pointer is an operator-owned atomic selection. Installed provenance
records are never mutated. Every application release must contain the exact
registered `ibapi` tree before promotion; rollback verification will fail
closed if a release contains a different or absent tree.

Install the read-only weekly release checker:

```bash
sudo install -o root -g root -m 0644 \
  /opt/stocker/current/deploy/systemd/stocker-ibkr-api-update.service \
  /opt/stocker/current/deploy/systemd/stocker-ibkr-api-update.timer \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stocker-ibkr-api-update.timer
sudo systemctl start stocker-ibkr-api-update.service
sudo -u stocker /opt/stocker/current/.venv/bin/stocker-prospective \
  ibkr-api verify \
  --provenance /var/lib/stocker/ibkr-api/active-provenance.json
```

The timer fetches metadata and writes update status only. It has no package
installer, download command, or environment-file access. When an update is
reported, repeat the manual licence, hash, staging, tests, and atomic promotion
procedure during planned maintenance. Never replace broker code automatically
while the recorder is running. A missing check or a check older than 14 days is
an explicit health blocker; a future-dated check is rejected.

If the official client is absent, replay remains available and IBKR mode
returns `blocked_official_ibkr_api_not_installed`.
If it is present without matching immutable provenance, IBKR mode returns
`blocked_unverified_official_ibkr_api`.

### 3a. Install the official IB Gateway and private login display

The Python API is only the socket client. A running, authenticated TWS or IB
Gateway is also required. Use the current official Linux X86_64 installer from
<https://www.interactivebrokers.com/en/trading/ibgateway-latest.php>. The
official Latest Gateway does not update itself; install each reviewed download
into a new versioned directory and atomically change `/opt/ibgateway/current`
during planned maintenance.

The first server installation on 2026-07-25 used the official Latest URL below.
Its observed SHA-256 was
`791abe12594c0d9c8736769fd8ee6368861b0d1a6b70e11275b32dedfec16692`.
Recheck the official page and record a new hash for any later release rather
than treating this observed hash as permanent.

```bash
sudo useradd --system --home-dir /var/lib/ibgateway \
  --create-home --shell /usr/sbin/nologin ibgateway
sudo install -d -o ibgateway -g ibgateway -m 0700 /var/lib/ibgateway
sudo install -d -o root -g ibgateway -m 0750 \
  /var/lib/ibgateway/installers
sudo install -d -o root -g ibgateway -m 0710 /etc/ibgateway
sudo apt-get update
sudo apt-get install --no-install-recommends \
  xvfb x11vnc openbox xauth dbus-x11 nftables
set -o errexit -o nounset -o pipefail
IDENTITY=REPLACE_WITH_VERSIONED_ID
INSTALLER=/var/lib/ibgateway/installers/ibgateway-20260725-latest-linux-x64.sh
EXPECTED_SHA=791abe12594c0d9c8736769fd8ee6368861b0d1a6b70e11275b32dedfec16692
TARGET="/opt/ibgateway/releases/$IDENTITY"
MANIFEST="/var/lib/ibgateway/installers/$IDENTITY.manifest.sha256"
SYMLINKS="/var/lib/ibgateway/installers/$IDENTITY.symlinks"
PROVENANCE="/var/lib/ibgateway/installers/$IDENTITY.runtime-provenance"
sudo test ! -e "$MANIFEST"
sudo test ! -e "$SYMLINKS"
sudo test ! -e "$PROVENANCE"
INSTALLER_TMP="$(
  sudo mktemp /var/lib/ibgateway/installers/.installer.XXXXXX
)"
sudo curl --fail --location --proto '=https' --tlsv1.2 \
  --output "$INSTALLER_TMP" \
  https://download2.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh
printf '%s  %s\n' "$EXPECTED_SHA" "$INSTALLER_TMP" |
  sudo sha256sum --check
sudo chown root:ibgateway "$INSTALLER_TMP"
sudo chmod 0750 "$INSTALLER_TMP"
sudo ln "$INSTALLER_TMP" "$INSTALLER"
sudo rm -- "$INSTALLER_TMP"
sudo install -d -o root -g ibgateway -m 0750 /opt/ibgateway/releases
sudo mkdir --mode=0750 "$TARGET"
sudo chown ibgateway:ibgateway "$TARGET"
cd /var/lib/ibgateway
sudo -H -u ibgateway "$INSTALLER" -q -dir "$TARGET"
sudo chown -R root:ibgateway "$TARGET"
sudo chmod -R go-w "$TARGET"

MANIFEST_TMP="$(
  sudo mktemp /var/lib/ibgateway/installers/.manifest.XXXXXX
)"
sudo sh -c \
  'cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum' \
  sh "$TARGET" |
  sudo tee "$MANIFEST_TMP" >/dev/null
sudo chown root:ibgateway "$MANIFEST_TMP"
sudo chmod 0640 "$MANIFEST_TMP"
sudo ln "$MANIFEST_TMP" "$MANIFEST"
sudo rm -- "$MANIFEST_TMP"
MANIFEST_SHA="$(sudo sha256sum "$MANIFEST" | awk '{print $1}')"

SYMLINKS_TMP="$(
  sudo mktemp /var/lib/ibgateway/installers/.symlinks.XXXXXX
)"
sudo sh -c \
  'cd "$1" && find . -type l -printf "%P\t%l\n" | LC_ALL=C sort' \
  sh "$TARGET" |
  sudo tee "$SYMLINKS_TMP" >/dev/null
sudo chown root:ibgateway "$SYMLINKS_TMP"
sudo chmod 0640 "$SYMLINKS_TMP"
sudo ln "$SYMLINKS_TMP" "$SYMLINKS"
sudo rm -- "$SYMLINKS_TMP"
SYMLINKS_SHA="$(sudo sha256sum "$SYMLINKS" | awk '{print $1}')"

PROVENANCE_TMP="$(
  sudo mktemp /var/lib/ibgateway/installers/.provenance.XXXXXX
)"
{
  printf 'manifest_version=2\n'
  printf 'source_url=%s\n' \
    'https://download2.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh'
  printf 'installer_path=%s\n' "$INSTALLER"
  printf 'installer_sha256=%s\n' "$EXPECTED_SHA"
  printf 'installed_path=%s\n' "$TARGET"
  printf 'file_manifest_path=%s\n' "$MANIFEST"
  printf 'file_manifest_sha256=%s\n' "$MANIFEST_SHA"
  printf 'symlink_manifest_path=%s\n' "$SYMLINKS"
  printf 'symlink_manifest_sha256=%s\n' "$SYMLINKS_SHA"
  printf 'recorded_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | sudo tee "$PROVENANCE_TMP" >/dev/null
sudo chown root:ibgateway "$PROVENANCE_TMP"
sudo chmod 0640 "$PROVENANCE_TMP"
sudo ln "$PROVENANCE_TMP" "$PROVENANCE"
sudo rm -- "$PROVENANCE_TMP"

sudo install -d -o root -g root -m 0755 /usr/local/libexec
sudo install -o root -g root -m 0755 \
  /opt/stocker/current/deploy/scripts/verify-ibgateway-installation.sh \
  /usr/local/libexec/stocker-verify-ibgateway
sudo test ! -e /opt/ibgateway/current.next
sudo test ! -L /opt/ibgateway/current.next
sudo ln -s "$TARGET" /opt/ibgateway/current.next
sudo env \
  IBGATEWAY_ACTIVE_LINK=/opt/ibgateway/current.next \
  /usr/local/libexec/stocker-verify-ibgateway
sudo mv -Tf /opt/ibgateway/current.next /opt/ibgateway/current
```

The archive, regular-file manifest, symlink-target manifest, and provenance
names are create-only. Reusing an identity must stop at a `test` or hard-link
failure; never overwrite one. Every Gateway start rechecks the archived
installer, both manifests, ownership, write permissions, every installed file,
and that every symlink resolves inside the immutable release. An integrity
failure makes the systemd `ExecCondition` skip startup instead of entering a
restart loop.

The verifier is a read-only root `ExecCondition` because the official release
contains a root-only uninstaller that the `ibgateway` account cannot hash. The
Gateway main process and every GUI helper still run as the unprivileged,
isolated `ibgateway` user; the verifier performs no writes or network access.

Legacy version-1 Gateway provenance does not satisfy this contract. Generate
new version-2 regular-file and symlink manifests and create a new immutable
runtime-provenance record before re-promoting an existing verified release, or
install it under a new identity. Never relabel or overwrite the version-1
record.

Create X11 and VNC authentication material. This VNC credential is only for the
loopback tunnel; it is not an IBKR credential and is never exposed by Stocker:

```bash
sudo install -d -o ibgateway -g ibgateway -m 0700 /var/lib/ibgateway
X_COOKIE="$(mcookie)"
sudo -H -u ibgateway touch /var/lib/ibgateway/.Xauthority
printf 'add :71 . %s\n' "$X_COOKIE" |
  sudo -H -u ibgateway xauth -f /var/lib/ibgateway/.Xauthority source -
unset X_COOKIE
VNC_PASSWORD="$(openssl rand -base64 6)"
printf '%s\n%s\ny\n' "$VNC_PASSWORD" "$VNC_PASSWORD" |
  sudo -H -u ibgateway env SHELL=/bin/sh script -q -c \
    'x11vnc -storepasswd /var/lib/ibgateway/vnc.pass' /dev/null \
    >/dev/null 2>&1
sudo install -o root -g root -m 0600 /dev/null \
  /etc/ibgateway/vnc-password
printf '%s\n' "$VNC_PASSWORD" |
  sudo tee /etc/ibgateway/vnc-password >/dev/null
unset VNC_PASSWORD
sudo chown ibgateway:ibgateway /var/lib/ibgateway/vnc.pass
sudo chmod 0600 \
  /var/lib/ibgateway/.Xauthority \
  /var/lib/ibgateway/vnc.pass \
  /etc/ibgateway/vnc-password
```

Configure a non-secret, exact upstream port for the loopback proxy. The
Gateway may bind that upstream port to a wildcard address even when its own
localhost-only policy is enabled. The host firewall therefore blocks the
upstream port on every non-loopback interface, while Stocker connects only to
the separate `127.0.0.1:4003` proxy:

```bash
IBKR_GATEWAY_PORT=REPLACE_WITH_EXACT_CONFIGURED_PORT
case "$IBKR_GATEWAY_PORT" in
  ''|*[!0-9]*|22|80|443|4003)
    echo 'Refusing missing, non-numeric, public-service, or proxy port' >&2
    exit 1
    ;;
esac
test "$IBKR_GATEWAY_PORT" -ge 1
test "$IBKR_GATEWAY_PORT" -le 65535
printf 'IBGATEWAY_UPSTREAM_PORT=%s\n' "$IBKR_GATEWAY_PORT" |
  sudo tee /etc/ibgateway/loopback-proxy.env >/dev/null
sudo chown root:ibgateway /etc/ibgateway/loopback-proxy.env
sudo chmod 0640 /etc/ibgateway/loopback-proxy.env

sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80/tcp comment 'HTTP redirect and ACME'
sudo ufw allow 443/tcp comment 'Stocker HTTPS'
sudo ufw insert 1 deny in to any port "$IBKR_GATEWAY_PORT" proto tcp \
  comment 'Block public IBKR API'
sudo ufw insert 1 allow in on lo to any port "$IBKR_GATEWAY_PORT" proto tcp \
  comment 'IB Gateway loopback only'
sudo ufw --force enable
```

The two inserted UFW rules must remain the first two rules in both the
`ufw-user-input` and `ufw6-user-input` chains: loopback allow first, then
non-loopback deny. This prevents an older broad allow from shadowing the broker
deny inside those chains. A separate Stocker-owned `inet stocker_ibgateway`
nftables base chain is the primary effective-path guard. It runs at input
priority -300, accepts the exact broker port only on loopback, and drops that
port without a source restriction everywhere else before UFW or another normal
filter chain can accept it. The installer replaces that two-rule table in one
atomic nftables transaction.

Port 4003 is the frozen internal deployment contract for this unit set; changing
it requires a coordinated unit, runtime-template, documentation, and
contract-version change.

Install the private graphical-session, firewall verifier, and loopback-proxy
units:

```bash
sudo install -o root -g root -m 0755 \
  /opt/stocker/current/deploy/scripts/install-ibgateway-loopback-boundary.sh \
  /usr/local/libexec/stocker-install-ibgateway-loopback-boundary
sudo install -o root -g root -m 0755 \
  /opt/stocker/current/deploy/scripts/verify-ibgateway-nft-boundary-json.py \
  /usr/local/libexec/stocker-verify-ibgateway-nft-boundary-json
sudo install -o root -g root -m 0755 \
  /opt/stocker/current/deploy/scripts/verify-ibgateway-loopback-boundary.sh \
  /usr/local/libexec/stocker-verify-ibgateway-loopback-boundary
sudo install -o root -g root -m 0755 \
  /opt/stocker/current/deploy/scripts/verify-ibgateway-daily-readiness.sh \
  /usr/local/libexec/stocker-verify-ibgateway-daily-readiness
sudo install -o root -g root -m 0755 \
  /opt/stocker/current/deploy/scripts/run-ibgateway-loopback-proxy.sh \
  /usr/local/libexec/stocker-ibgateway-loopback-proxy
sudo install -o root -g root -m 0644 \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-display.service \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-window-manager.service \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-vnc.service \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-loopback-boundary.service \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-loopback-proxy.socket \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-loopback-proxy.service \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-daily-readiness.service \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-daily-readiness.timer \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway.service \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo /usr/local/libexec/stocker-install-ibgateway-loopback-boundary
sudo /usr/local/libexec/stocker-verify-ibgateway-loopback-boundary
sudo systemctl enable stocker-ibgateway-loopback-proxy.socket
sudo systemctl enable --now stocker-ibgateway-daily-readiness.timer
sudo systemctl enable --now stocker-ibgateway.service
```

The installer and read-only verifier run before both Gateway and the proxy
socket. The verifier parses nftables JSON and requires exactly the unrestricted
drop, loopback allow, input hook, and priority above; a source-restricted rule
does not pass. Missing or malformed proxy configuration, inactive UFW, a
shadowing UFW rule, a missing IPv6 rule, or a missing/altered effective-path
guard blocks startup with `blocked_unsafe_runtime_configuration`. The proxy
socket therefore cannot look healthy before its configuration and firewall
boundary have passed. Proxy recovery is bounded to three activations in five
minutes; exit 78 is never automatically restarted.

The display has no TCP listener. VNC listens only on server loopback and
requires its separate random password. Do not add port 5901 to the public
firewall.

From the operator workstation, create an SSH tunnel:

```bash
ssh -N -L 5901:127.0.0.1:5901 operator@SERVER
```

Open a native VNC client at `127.0.0.1:5901`. Retrieve the VNC password through
an authenticated SSH session:

```bash
sudo cat /etc/ibgateway/vnc-password
```

Enter the manual IBKR username, password, and 2FA only in the official Gateway
window. They must never enter the Stocker website, configuration, environment,
database, commands, or logs. Choose the paper session. In Gateway API settings:

1. Keep **Read-Only API** enabled.
2. Record the exact socket port; paper Gateway commonly defaults to 4002, but
   the runtime must use the value actually displayed.
3. Permit localhost only.
4. Keep API message logging free of unnecessary market-data payloads.
5. Configure **Auto restart** for 23:45 UTC. Plan for manual authentication
   again after the weekly reset.

The systemd unit uses `ExitType=cgroup` because Gateway's scheduled restart
passes its short-lived broker session to an authenticated handoff child before
the original Java process exits. Tracking the whole cgroup keeps the unit active
and prevents systemd's control-group cleanup from killing that child. The unit's
`Restart=always` and `RestartSec=1` remain a fallback only when the entire cgroup
exits without a surviving handoff process. An operator-issued `systemctl stop`
does not trigger a restart.
The read-only readiness timer checks the authenticated upstream port at 23:46
UTC, retrying for up to two minutes without starting, stopping, or restarting
Gateway. A weekly broker reset may still require manual authentication.

Verify that the upstream port is firewall-restricted, the Stocker endpoint is
loopback-only, and VNC remains private:

```bash
IBKR_GATEWAY_PORT=REPLACE_WITH_EXACT_CONFIGURED_PORT
STOCKER_IBKR_PROXY_PORT=4003
sudo ss -H -ltnp "( sport = :$IBKR_GATEWAY_PORT )"
sudo ss -H -ltnp "( sport = :$STOCKER_IBKR_PROXY_PORT )"
sudo ss -H -ltnp "( sport = :5901 )"
sudo ufw status verbose
sudo nft -j list table inet stocker_ibgateway | jq .
sudo systemctl status \
  stocker-ibgateway.service \
  stocker-ibgateway-daily-readiness.service \
  stocker-ibgateway-daily-readiness.timer \
  stocker-ibgateway-loopback-boundary.service \
  stocker-ibgateway-loopback-proxy.socket \
  stocker-ibgateway-vnc.service
```

The Gateway upstream may appear as `*:<configured Gateway port>`, but UFW must
deny that port on every non-loopback interface. The Stocker-facing listener
must be exactly `127.0.0.1:4003`; configure the recorder with that proxy port,
never the wildcard upstream. The recorder independently parses the Linux
listener table and refuses any wildcard or non-loopback configured endpoint
with `blocked_unsafe_runtime_configuration`. VNC must remain on loopback port
5901. Stop the VNC helper between maintenance sessions if desired; Gateway and
the Stocker recorder do not depend on VNC after login.

## 4. Copy a versioned application release

On the research/development machine, from a clean intended commit:

```bash
git rev-parse HEAD
git status --short
git archive --format=tar.gz --output=/tmp/stocker-REPLACE_WITH_GIT_COMMIT.tar.gz HEAD
sha256sum /tmp/stocker-REPLACE_WITH_GIT_COMMIT.tar.gz
scp /tmp/stocker-REPLACE_WITH_GIT_COMMIT.tar.gz \
  root@SERVER:/var/lib/stocker/secure-transfer/
```

On the server:

```bash
sudo chown root:stocker \
  /var/lib/stocker/secure-transfer/stocker-REPLACE_WITH_GIT_COMMIT.tar.gz
sudo chmod 0640 \
  /var/lib/stocker/secure-transfer/stocker-REPLACE_WITH_GIT_COMMIT.tar.gz
sudo install -d -o stocker -g stocker -m 0755 \
  /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT
sudo -u stocker tar -xzf \
  /var/lib/stocker/secure-transfer/stocker-REPLACE_WITH_GIT_COMMIT.tar.gz \
  -C /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT
cd /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT
sudo -u stocker uv sync --locked --no-editable --no-default-groups --group server
# Repeat section 3's extraction/install commands; do not rewrite provenance.
sudo -u stocker .venv/bin/stocker-prospective ibkr-api verify \
  --provenance /var/lib/stocker/ibkr-api/active-provenance.json
sudo chown -R root:stocker \
  /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT
sudo chmod -R go-w \
  /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT
sudo ln -s /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT /opt/stocker/current.next
sudo mv -T /opt/stocker/current.next /opt/stocker/current
```

Use `ln -sfn` instead of `ln -s` only when replacing an existing staging
symlink. Do not point a release at the database or bundle directory.

## 5. Build, transfer, install, and activate a frozen bundle

The original research run wrote a complete audited numerical handoff, but did
not serialize deployable estimator objects. The repository therefore provides
an explicit, no-fit reconstruction command. It accepts only the hash-bound
frozen JSON files, verifies their pre-outcome freeze identities and safety
flags, and emits deterministic M0, M1, preprocessing, feature-schema,
previous-session-context-schema, threshold-provenance, and reconstruction
artifacts. It never invokes model fitting and never reads observations.

This command is authorized only for the audited
`20260724-minimal-intraday-iv-excess-holdout-v01` handoff. It does not refit,
change the frozen threshold or universe, use 2026+ observations, establish an
options edge, or enable orders.

On the research machine:

```bash
FROZEN_ROOT=research/options-feasibility/20260724-minimal-intraday-iv-excess-holdout-v01/artifacts/primary
BUNDLE_ID=m1-frozen-20260724-feature-runtime-v1
CREATED_AT_UTC=REPLACE_WITH_CURRENT_UTC_ISO_TIMESTAMP
OPERATOR_ID=REPLACE_WITH_OPERATOR_ID

uv run --no-editable stocker-prospective bundle reconstruct \
  --frozen-root "$FROZEN_ROOT" \
  --universe configs/prospective/anchor-frozen-20.json \
  --feature-runtime-registry \
    configs/prospective/frozen-feature-runtime-v1.json \
  --repository-root "$PWD" \
  --output "/tmp/$BUNDLE_ID-reconstructed" \
  --bundle-id "$BUNDLE_ID" \
  --created-at-utc "$CREATED_AT_UTC" \
  --operator "$OPERATOR_ID"
uv run --no-editable stocker-prospective bundle build \
  --spec "/tmp/$BUNDLE_ID-reconstructed/bundle-spec.yaml" \
  --output "/tmp/$BUNDLE_ID"
uv run --no-editable stocker-prospective bundle inspect "/tmp/$BUNDLE_ID"
uv run --no-editable stocker-prospective bundle verify "/tmp/$BUNDLE_ID"
tar -C /tmp -czf "/tmp/$BUNDLE_ID.tar.gz" "$BUNDLE_ID"
scp "/tmp/$BUNDLE_ID.tar.gz" \
  root@SERVER:/var/lib/stocker/secure-transfer/
```

On the server:

```bash
sudo tar -xzf /var/lib/stocker/secure-transfer/REPLACE_WITH_BUNDLE_ID.tar.gz \
  -C /var/lib/stocker/secure-transfer
sudo chown -R stocker:stocker \
  /var/lib/stocker/secure-transfer/REPLACE_WITH_BUNDLE_ID
sudo -u stocker /opt/stocker/current/.venv/bin/stocker-prospective bundle verify \
  /var/lib/stocker/secure-transfer/REPLACE_WITH_BUNDLE_ID
sudo -u stocker /opt/stocker/current/.venv/bin/stocker-prospective bundle install \
  /var/lib/stocker/secure-transfer/REPLACE_WITH_BUNDLE_ID \
  --bundle-root /var/lib/stocker/bundles \
  --operator REPLACE_WITH_OPERATOR_ID
sudo -u stocker /opt/stocker/current/.venv/bin/stocker-prospective bundle list \
  --bundle-root /var/lib/stocker/bundles
sudo -u stocker /opt/stocker/current/.venv/bin/stocker-prospective bundle activate \
  REPLACE_WITH_BUNDLE_ID \
  --bundle-root /var/lib/stocker/bundles \
  --operator REPLACE_WITH_OPERATOR_ID
```

For later activation, supply
`--expected-current CURRENT_BUNDLE_ID`. Activation will refuse a stale operator
view or a hash mismatch. An installed bundle is never edited or overwritten.

## 6. Configure environment and runtime

Copy the templates and restrict them:

```bash
sudo install -o root -g stocker-readers -m 0640 \
  /opt/stocker/current/configs/prospective/server.example.yaml \
  /etc/stocker/prospective.yaml
sudo install -o root -g stocker -m 0640 \
  /opt/stocker/current/deploy/stocker.env.example \
  /etc/stocker/stocker.env
sudo install -o root -g stocker-readers -m 0640 \
  /opt/stocker/current/deploy/stocker-web.env.example \
  /etc/stocker/stocker-web.env
sudoedit /etc/stocker/prospective.yaml
sudoedit /etc/stocker/stocker.env
sudoedit /etc/stocker/stocker-web.env
```

Before first prospective start, set:

- a new immutable `runtime.run_id`;
- the exact future `prospective_start_utc`;
- server instance identity, release version, and Git commit;
- `record_only` or `shadow` only;
- the Stocker endpoint exactly as `127.0.0.1:4003`, a dedicated non-zero client
  ID, and the paper environment (the Gateway upstream port belongs only in
  `/etc/ibgateway/loopback-proxy.env`);
- measured or API-observed market-data capacity, externally consumed lines,
  a protected future-trading reserve, and a request rate no greater than half
  the configured line budget;
- context-signing secret in the environment file;
- leave `parallel_validation.enabled: false` unless the optional EODHD
  cross-vendor diagnostic is intentionally operated; and
- optional web auth token only when `authentication_enabled: true`.

Keep these server paths in both `/etc/stocker/stocker.env` and the
secret-minimal `/etc/stocker/stocker-web.env`:

```dotenv
STOCKER_IBKR_API_PROVENANCE=/var/lib/stocker/ibkr-api/active-provenance.json
STOCKER_IBKR_API_UPDATE_STATUS=/var/lib/stocker/ibkr-api/status/update-status.json
```

Set recorder-only market-data limits in `/etc/stocker/stocker.env`. IBKR limits
that the API cannot expose must be explicit; the startup manifest records
whether each value was discovered, configured by environment, or taken from
the reviewed configuration fallback:

```dotenv
IBKR_TOTAL_MARKET_DATA_LINES=100
IBKR_EXTERNALLY_RESERVED_LINES=0
IBKR_RESERVED_FUTURE_TRADING_LINES=12
IBKR_MAX_TICK_BY_TICK=2
IBKR_MAX_DEPTH=0
IBKR_MAX_CONCURRENT_SNAPSHOTS=2
IBKR_MAX_ACTIVE_OPTION_EPISODES=1
IBKR_MAX_OPTION_LINES_PER_EPISODE=8
IBKR_HISTORICAL_REQUESTS_PER_WINDOW=60
IBKR_HISTORICAL_REQUEST_WINDOW_SECONDS=600
```

Replace the example values with the account/TWS allowances actually observed
at deployment. Never reduce the 12-line future-trading reserve to admit
neutral controls, alternate DTEs, outer strikes, tick-by-tick, or depth. On
startup inspect `ibkr_runtime_capacity_manifest.json`; the always-on target is
20 stock bar streams plus only required proxies. Level II remains optional and
disabled by default; enabling it does not change scientific eligibility. The
recorder enforces the resolved historical-bar request allowance as a rolling
window during startup and reconnect restoration.

IBKR is the prospective live-data source. An otherwise valid, complete IBKR
session is usable from the first session as **prospective evaluation of the
frozen implementation using IBKR market data**. This does not claim that IBKR
bars are identical to historical EODHD bars. Source identity remains explicit
on evidence and reports. The earlier mandatory 20-session
`engineering_transfer` authorization gate is superseded; its existing rows and
receipts remain immutable historical records. Runtime parity, completeness,
freshness, gap, artifact, and no-order checks still fail closed on their own
merits.

Put the token only in `/etc/stocker/stocker.env`:

```dotenv
# Optional: set only when parallel_validation.enabled is true for the
# diagnostic. Never put a real token in this runbook or a shell command.
EODHD_API_TOKEN=
```

Put only its non-secret status projection in
`/etc/stocker/stocker-web.env`:

```dotenv
STOCKER_EODHD_TOKEN_CONFIGURED=0
```

Put the context-signing secret only in `stocker.env`; never expose it to the web
process. Put an optional built-in web-auth token only in `stocker-web.env`.
If the optional diagnostic is enabled, put the EODHD token only in
`stocker.env`; the web process receives only a boolean
`credential_configured` projection. Set
`STOCKER_EODHD_TOKEN_CONFIGURED=1` in `stocker-web.env` only after the token is
present in `stocker.env`; otherwise leave it `0`. A value of `0` projects the
neutral `cross_vendor_validation_not_configured` diagnostic and is not a
recorder blocker. EODHD does not run as a separate service. When enabled, the
recorder makes bounded requests after the session and stores source-labelled
diagnostic evidence that is permanently ineligible for scoring.
Never put IBKR username, password, or 2FA material in any Stocker file. Stocker
has no fields for them.

### Group O exact-chain publication and pre-signal recovery

Group O is a separately frozen previous-session context input, not the
IBKR-versus-EODHD transfer diagnostic. Prepare its signed exact-chain package
before recorder startup and let the pre-adapter verifier fail closed if it is
absent or invalid. The dedicated
`scientific-inputs recover-group-o-exact-chain-v2` workflow may use EODHD only
for that protected context recovery. A missing EODHD credential during later
best-effort preparation is recorded as a preparation error and does not turn
cross-vendor comparison into a scientific authorization gate.

An EODHD HTTP 200 response is not sufficient to finalize Group O context. Every
frozen cohort symbol must have at least one canonical exact-session option row.
If any symbol has zero canonical rows, the separate preparation workflow
writes an immutable
`pending_exact_chain` attempt receipt under:

```text
/var/lib/stocker/daily-context/source-cache/eodhd-group-o/
  <observation-session>/attempts/<attempt-id>/attempt_receipt.json
```

The signal-session package is not published in that state. Each retry uses the
next four-digit attempt directory so an earlier empty provider response is
preserved and cannot be reused as a completed cache entry.

If an older recorder already finalized a `missing_exact_chain` base package,
the base file remains byte-for-byte unchanged. A successful later acquisition
may append a self-binding revision under:

```text
/var/lib/stocker/daily-context/group-o/revisions/
  <signal-session>/<four-digit-revision-number>.json
```

Revision numbers must form a contiguous hash-linked chain from the exact base
file. The frozen contract version, feature and regime hashes, cohort ordering,
signal session, and exact D-1 observation identity cannot change. The loader
rejects a revision created at or after the exact XNYS signal-session open. This
permits delayed D-1 publication before a future signal while prohibiting
same-session or retrospective eligibility changes.

For the audited Friday 2026-07-31 source-publication recovery targeting Monday
2026-08-03, stop the recorder and run the dedicated pre-adapter command from
the staged release before promotion or recorder restart:

```bash
sudo -u stocker sh -c '
  set -a
  . /etc/stocker/stocker.env
  exec /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT/.venv/bin/stocker-prospective \
    scientific-inputs recover-group-o-exact-chain-v2 \
    --config /etc/stocker/prospective.yaml \
    --release-directory /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT
'
```

The command verifies the signed recovery freeze, exact 20-symbol cohort, failed
base hash, exact V1 deployment/start/failed-attempt chain, and pre-open cutoff;
then it writes
`<attempt>/recovery_start_receipt.json` before making any EODHD request. The
same release blocks before constructing the IBKR adapter unless that start
receipt, acquisition receipt, hash-linked revision, and the self-binding
`<attempt>/recovery_completion_receipt.json` agree. The completion receipt
hash-links the deployment, start, failed base, staged candidate, revision, and
acquisition-attempt identities and file bytes. If EODHD still has no exact
Friday rows, the command preserves the attempt, waits until its signed
`retry_after_utc`, and retries automatically at the frozen 15-minute interval
until the pre-open cutoff. Leave the recorder stopped and do not run a second
recovery process. Never bypass the gate or edit the failed package.

This pre-adapter gate is permanent for the recovery version; it does not
expire after the target session closes. Removing it requires a new signed
recorder version and deployment receipt, not a clock-based bypass.

Recovery V1 failed closed in append-only attempt `0001` after EODHD began
publishing the Friday rows: the provider's after-midnight-UTC `dte` values did
not equal the calendar interval from the EOD resource identity. V2 does not
rewrite or reuse that attempt. Its signed policy treats every provider `dte`
value as a non-admission diagnostic, deterministically recomputes DTE from the
exact EOD observation date and expiration, and writes an immutable
`provider_dte_diagnostics.json` linked from the signed attempt receipt. It
continues to reject inconsistent bid/ask observation dates and downloads a
fresh attempt. V2 also enforces V1 freeze, V1 start, V1 completion, V2 freeze,
then V2 start chronology. Reconciliation verifies and skips V1 attempt `0001`
before examining subsequent V2 attempts. Never run the retired V1 command
again.

Do not delete, rename, edit, or replace the failed base, an acquisition-attempt
receipt, or a revision. A missing or invalid chain remains fail-closed for M1C.
Opening Leader Continuation V0 treats M1C as context only and continues to obey
its independently signed rank-selection contract. None of these files enables
orders or changes the record-only runtime mode.

### Opening Leader runtime V10 supersession

Runtime V10 supersedes two operational assumptions recorded in the original
Opening Leader package documentation. The 420-second signal-capture deadline is
now a processing-latency diagnostic, not a scientific failure boundary, and the
selected rank-1 symbol receives exactly one replaceable post-selection Level I
subscription. The original package and every earlier receipt remain immutable;
V10 records this semantic change in a new append-only deployment receipt.

Selection still uses only completed causal bars admitted by the nominal
deadline. A delayed callback worker or restart may project those timely inputs
and complete the receipt later, but genuinely late causal rank inputs fail.
The receipt records processing delay and E0 must be a strictly later quote than
the immutable receipt time. There is no reconstructed or backfilled entry.
Inputs or a quote that remain unresolved at end-of-session reconciliation fail
explicitly. A failure already recorded by an earlier runtime, including a
prior C6 deadline failure, is never rewritten or promoted after the fact.

Opening Leader option evidence freezes the exact P20, P30, and BPS20 leg
identities at E0, requests those same conIds at later observations, and never
substitutes a new spot-band contract. Its
segregated ledger uses executable bid/ask P&L after configured costs as the
primary result. Missing Greeks remain diagnostic, and absent reliable observed
margin leaves margin ROI unavailable while cash-secured or defined-risk ROI
remains available. This path is research-only: it has no account, position,
execution, order, buy, or sell surface and cannot route an order.

### Quiet Options runtime V11 supersession

Runtime V11 supersedes V10's post-selection Level-I assumption for Quiet
Options. Before bars are started, recorder preflight must be able to protect
one live Level-I stream for every frozen stock plus VTI and one completed
five-minute-bar stream for every stock and proxy. Sector proxies remain
bar-only. If either the per-kind limits or the total research-line budget is
insufficient, startup exits with `critical_budget_unavailable` instead of
running a permanently ineligible recorder. Do not work around that failure by
raising `maximum_quote_age_seconds`, reducing the frozen universe, or removing
reserved/safety capacity.

At a quiet checkpoint, the recorder selects the latest valid quote at or before
`trigger_end` from its bounded per-symbol history. Freshness remains two
seconds and is calculated against `trigger_end`; processing time is irrelevant.
A T+9 quote therefore cannot overwrite a valid T-1 boundary quote when the
worker completes at T+10. Missing, invalid, stale, or non-primary market data
still rejects the checkpoint. Each new quiet checkpoint persists the selected
event ID, timestamp, age, and
`latest_valid_at_or_before_checkpoint_boundary_v0` policy.

Migration `0030` never updates a pre-existing prospective checkpoint. It
snapshots affected pre-fix stale classifications into the separate
`quiet_quote_instrumentation_defect_*_v0` audit dataset with recomputation and
observation creation disabled. The originally reported 320 rows, plus any
additional pre-migration classifications exposed to the same defect, remain
exactly as first recorded; do not relabel them or use the audit dataset as
prospective evidence.
Verify the snapshot after migration with read-only SQL:

```sql
SELECT affected_checkpoint_count, dataset_scope,
       original_evidence_modified, recomputation_authorized,
       may_create_quiet_observation
FROM quiet_quote_instrumentation_defect_v0;

SELECT COUNT(*)
FROM quiet_quote_instrumentation_defect_checkpoint_v0;
```

For a new checkpoint, inspect the causal selection without consulting the
mutable current quote:

```sql
SELECT symbol, session_date, checkpoint, eligible,
       selected_underlying_quote_timestamp_utc,
       selected_underlying_quote_age_seconds,
       underlying_quote_selection_policy,
       data_quality_flags_json
FROM quiet_state_checkpoint_v0
ORDER BY id DESC
LIMIT 20;
```

The bottom-10 threshold remains `0.135896965695626`. A probability above it,
including IREN at `0.157624`, is not a trigger and must not be forced. A valid
crossing creates the immutable quiet observation and schedules bounded shadow
option capture. The system remains research/shadow-only, read-only at IBKR,
and has no order-capable API surface.

Before the first recorder start, create the immutable frozen activity baseline:

```bash
sudo install -d -o stocker -g stocker -m 0750 /var/lib/stocker/preprocessing
sudo -u stocker sh -c '
  set -a
  . /etc/stocker/stocker.env
  exec /opt/stocker/current/.venv/bin/stocker-prospective \
    scientific-inputs build-activity-baseline \
    --config /etc/stocker/prospective.yaml \
    --from-session 2024-01-02 \
    --latest-authorised-session 2026-06-29
'
```

The command validates the exact registered 20-stock cohort, rejects missing
bar-ordinal support, and refuses to replace a different existing baseline. The
subshell loads the recorder-only environment without copying its secrets onto
the command line or exposing them to the web process.

## 7. Migrate the database

Never migrate the active recorder database while the recorder, web process, or
an evidence-replay worker is running. Before changing runtime state, require a
proven terminal dashboard replay-controller state: `stopped`, `completed`, or
`failed`. Abort on `running`, `stopping`, or `stop_failed`; `stop_failed` means
the worker remained alive after the bounded join, and an OS process search
cannot detect that in-process thread. Once the terminal state is proven, run:

```bash
sudo systemctl is-active stocker-recorder.service
pgrep -af '[s]tocker-prospective.*replay|[e]vidence_replay'
```

The recorder result must already be `inactive`, and `pgrep` must return no
external replay worker. If either check differs, stop the deployment without
changing runtime state. Once those checks pass, stop the web service and verify
the complete application boundary is inactive:

```bash
sudo systemctl stop stocker-web.service
sudo systemctl is-active stocker-recorder.service stocker-web.service
pgrep -af '[s]tocker-prospective.*replay|[e]vidence_replay'
```

Both `systemctl` results must now be `inactive`, and `pgrep` must still return
no worker. The web shutdown hook cancels and joins an in-process replay before
the process exits. With both services stopped, create a checked backup using
the currently installed release:

```bash
sudo -u stocker /opt/stocker/current/.venv/bin/stocker-prospective db backup \
  --database /var/lib/stocker/prospective/prospective.sqlite3 \
  --destination /var/lib/stocker/backups
```

Record the emitted backup and manifest paths. Do not continue unless the
command succeeds and reports `quick_check: ok`; the backup command uses
SQLite's backup API and writes a SHA-256 manifest. The recorder stays stopped
after a web-only deployment unless an operator separately authorizes a new
recording session.

Only after that preflight may the staged release apply its migrations:

```bash
sudo -u stocker /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT/.venv/bin/stocker-prospective \
  db migrate \
  --database /var/lib/stocker/prospective/prospective.sqlite3
sudo -u stocker sqlite3 /var/lib/stocker/prospective/prospective.sqlite3 \
  'PRAGMA journal_mode; PRAGMA quick_check; PRAGMA foreign_key_check;'
```

Expected results include `wal` and `ok`, with no foreign-key rows. Abort the
release promotion if any validation fails. Do not restore a pre-migration
backup automatically after a forward migration; retain it for an explicit
operator-led recovery decision.

Migration `0016_prospective_recorder_hardening_v1` is forward-only and retains
all pre-hardening rows. It places the callback inbox, leases,
acknowledgements, fatal latches, gap incidents, runtime artifact receipts,
recorder generations, and operational incidents in the same checked SQLite
backup boundary. Startup fails closed if the database contains a migration
newer than the installed application supports.

Migration `0020_opening_reversal_shadow_capture_v1` historically encoded the
then-current 20-session authorization gate. That gate is superseded for new
IBKR prospective observations; existing engineering-shadow rows remain
unchanged. Runtime verification and per-observation quality checks continue to
control eligibility.

Migration `0021_opening_reversal_activation_run_binding_v1` separates
operational run lineage from immutable V1/V1.1 activation identity. It
preserves existing receipt rows as originals and permits only append-only,
byte-identical audited bindings in a replacement run.

Migrations `0022_web_read_projections_v0` and
`0023_web_latest_state_v0` add only derived web indexes, bounded audit
identities, latest raw-event counters, and current runtime-blocker projections.
They do not rewrite immutable raw evidence or alter scientific rows.

The migration ledger key is the complete filename. Already deployed
filenames are never renamed. Historical duplicate prefixes `0011` and `0012`
have an explicit frozen order; every new migration must use a unique,
monotonically increasing four-digit prefix. Before packaging a release run:

```bash
uv run python scripts/check_prospective_migrations.py
uv run pytest tests/test_migration_policy.py
```

The equivalence test builds one database from zero and upgrades another from
the frozen through-`0021` migration-hash fixture, then compares schema
objects, foreign keys, indexes, constraints visible in SQLite schema SQL, and
the complete migration ledger.

## 8. Start deterministic replay mode

Replay needs no bundle, EODHD credential, or IBKR client:

```bash
cd /opt/stocker/current
sudo -u stocker env STOCKER_GIT_COMMIT=REPLACE_WITH_GIT_COMMIT \
  .venv/bin/stocker-prospective replay run \
  --config configs/prospective/replay.example.yaml
sudo -u stocker env STOCKER_GIT_COMMIT=REPLACE_WITH_GIT_COMMIT \
  .venv/bin/stocker-prospective replay run \
  --config configs/prospective/replay.example.yaml
```

The second invocation must report the same counts. Scores must read
`synthetic_replay_not_frozen_m1`. The replay includes two crossing episodes,
entry plus 5/10/15/30 captures, a missing expiry, stale and delayed data,
connection loss, both reconnect variants, a budget rejection, ten structures,
and forty horizon valuations.

To serve that replay:

```bash
cd /opt/stocker/current
sudo -u stocker env STOCKER_GIT_COMMIT=REPLACE_WITH_GIT_COMMIT \
  .venv/bin/stocker-prospective web run \
  --config configs/prospective/replay.example.yaml
```

For a persistent systemd replay deployment, copy
`configs/prospective/server-replay.example.yaml` to
`/etc/stocker/prospective.yaml`. It keeps the database and fixtures outside and
inside the correct server boundaries respectively; the workstation-oriented
replay example intentionally uses `/tmp` and must not be installed as the
service configuration.

## 9. Verify health and API safety

From the server:

```bash
curl --fail --silent http://127.0.0.1:8765/api/health | jq .
curl --fail --silent http://127.0.0.1:8765/api/dashboard/summary | jq .
curl --fail --silent http://127.0.0.1:8765/api/recorder/status | jq .
curl --fail --silent http://127.0.0.1:8765/api/virtual-ledgers | jq .
curl --fail --silent http://127.0.0.1:8765/api/config/public | jq .
curl --fail --silent http://127.0.0.1:8765/openapi.json | jq '.paths | keys'
```

The replay health response should be correctly `blocked`, with synthetic data
visible and real-scoring/IBKR blockers present. It should state `LIVE TRADING
DISABLED`, expose the named no-order checks and their aggregate verdict, and
expose no secret or database path. The `ibkr_api` section independently reports
whether the first-party package is installed, source-tree verified, and
current. A YAML `read_only` value is only configured evidence: local adapter
surface enforcement is a separate check, and the Gateway/TWS environment is
reported as not externally verifiable.

The fast dashboard summary deliberately includes only recorder state and
heartbeats, callback admission/acknowledgement and inbox counts, the latest
completed bar/checkpoint/episode, IBKR connection and subscription counts, one
compact alert list, dynamic no-order state, and replay state. Use
`/api/recorder/status` or explicit screen refreshes for gap detail, persisted
artifact receipts, history, audits, reports, and ledgers. An old historical run
without a current fresh lease must show `INACTIVE`, `STOPPED_CLEANLY`, or
`STALE_HEARTBEAT`; it must never show recording merely because rows exist.

The virtual-ledger response contains separate `opening_reversal` and
`quiet_state` collections. It must state that the ledgers are not combined and
that no execution or broker positions are claimed. An opening-reversal row is
`CLOSED` only after both strict primary-pair outcomes are complete. Quiet-state
rows include only bottom-10 short-premium structures; controls and long-option
candidates remain outside that ledger. An empty collection means no qualifying
immutable outcome exists for the selected run, not that a hidden fill occurred.

## 10. Start record-only IBKR mode

First configure TWS or IB Gateway manually:

1. Use the paper account.
2. Enable socket clients.
3. Keep Read-Only API enabled.
4. Keep localhost-only connections enabled.
5. Set and record the exact Gateway upstream socket port only in
   `/etc/ibgateway/loopback-proxy.env`.
6. Use a dedicated non-zero client ID that is not the Master client.
7. Authenticate manually, including 2FA.

Confirm that the fail-closed boundary passes and that Stocker uses only its
loopback proxy. The Gateway listener may be wildcard-bound because UFW blocks
it before Gateway can start:

```bash
sudo /usr/local/libexec/stocker-verify-ibgateway-loopback-boundary
sudo nft -j list table inet stocker_ibgateway | jq .
sudo ss -H -ltnp '( sport = :4003 )'
sudo grep -A3 '^ibkr:' /etc/stocker/prospective.yaml
```

The runtime configuration must contain `host: 127.0.0.1` and `port: 4003`,
never the Gateway upstream port.

Record-only IBKR mode requires the durable raw recorder, hash-verified
registered universe, exact frozen artifact root, and official dependency.
Legacy memory-drain diagnostic mode is not allowed to open an IBKR socket.
Missing, partial, schema-invalid, contract-incompatible, or hash-mismatched
artifacts fail closed. A configured path does not remove the blocker: only
persisted receipts from the active recorder generation can do so. Shadow
scoring still requires passing feature parity and exact signed
previous-session context. Do not lower either gate after observing outcomes.
Install the units only after the gates for the selected mode are satisfied:

When `parallel_validation.enabled` is true, the recorder also makes one bounded
five-minute-history request per anchor symbol after the configured
`capture_delay_seconds`. It does not backfill those rows into a score. The
comparison may report timestamp agreement, missing bars, return/probability
correlation and bias, tail membership, and episode timing. Its states are
diagnostic (`pending`, `insufficient_sessions`, `available`, `warning`, or
`failed_diagnostic`); none authorizes trading or invalidates otherwise sound
IBKR evidence. Exact OHLCV equality is not required.

Executable bid/ask P&L after commissions and configured fees is the primary
option-performance measure. Each strategy exposes one primary capital basis:
premium paid for long options, cash-secured capital for a naked short put, or
maximum defined risk for a defined-risk spread. Reliable observed IBKR margin
may be shown as a secondary diagnostic; absent reliable margin leaves margin
ROI unavailable and does not suppress the applicable primary ROI. Raw IBKR
Greeks remain source-separated. Taylor/Greek attribution is diagnostic only,
may be generated after finalization, and missing Greeks do not block P&L.

```bash
sudo install -o root -g root -m 0644 \
  /opt/stocker/current/deploy/systemd/stocker-recorder.service \
  /opt/stocker/current/deploy/systemd/stocker-web.service \
  /etc/systemd/system/
sudo install -o root -g root -m 0755 \
  /opt/stocker/current/deploy/scripts/prepare-web-sqlite-boundary.py \
  /usr/local/libexec/stocker-prepare-web-sqlite-boundary
sudo systemctl daemon-reload
sudo systemctl enable --now stocker-recorder.service
sudo systemctl enable --now stocker-web.service
```

Configuration/bundle-integrity exit code 78 is in
`RestartPreventExitStatus`; systemd will not loop on those failures.
Transient runtime failures use bounded systemd restart limits.

The web service runs as the distinct `stocker-web:stocker-readers` OS identity;
the database and its directory remain owned by `stocker`. The recorder secret
environment stays `root:stocker 0640`, so the reader group cannot open it.
Before web startup, a root helper uses descriptor-relative `O_NOFOLLOW` opens
and exclusive creation; it never performs a check-then-path mutation. It
requires the root-owned persistent parent, restores directory mode 2750,
database/WAL mode 0640, and creates or verifies only the SQLite WAL/SHM
coordination files (SHM mode 0660). The web identity
cannot write, truncate, replace, rename, or unlink the database or WAL through
filesystem syscalls. It can update only the group-writable SHM file. The
systemd mount keeps every other `/var/lib/stocker` path read-only. The helper
runs as an `ExecCondition`; its exit-78 safety rejection skips web startup
without entering the service's automatic restart policy. The recorder holds a
process-lifetime SQLite connection while it owns the lease, so a live recorder
cannot remove WAL coordination files between the condition and the web
process's read-only anchor.

The web repository independently opens SQLite with `mode=ro` and
`PRAGMA query_only=ON`, and a process-lifetime read-only anchor keeps WAL/SHM
coordination files present while the recorder opens and closes writer
connections. It also refuses any migration version newer than the installed
application understands. Tests require destructive SQL to fail. On the deployed host,
verify the independent filesystem boundary too:

```bash
sudo -u stocker-web -g stocker-readers test ! -w \
  /var/lib/stocker/prospective/prospective.sqlite3
sudo -u stocker-web -g stocker-readers test ! -w \
  /var/lib/stocker/prospective/prospective.sqlite3-wal
sudo -u stocker-web -g stocker-readers test -w \
  /var/lib/stocker/prospective/prospective.sqlite3-shm
sudo -u stocker-web -g stocker-readers test ! -w /var/lib/stocker/prospective
sudo -u stocker-web -g stocker-readers test ! -r /etc/stocker/stocker.env
```

IBKR mode may run the hash-verified frozen causal M1C model only when the
bundle, runtime-parity, completed-bar, signed Group O context, live-data, and
loopback/read-only capability gates all pass. It does not require exact EODHD
and IBKR bar equality. Optional market-data exhaustion must appear as a queued,
partial, degraded, or skipped recording while universe scoring continues;
only `critical_budget_unavailable` may block M1C signal recording. The service
remains record-only and exposes no order, account, position, or portfolio
method.

### Durable admission and restart rule

Migration `0017_callback_raw_only_recovery_v1.sql` is intentionally separate
from `0016`; databases that already applied the hardening schema receive the
new ownership and raw-only disposition columns without replaying or editing a
tracked migration.

For each official callback, expect a short-lived `provider_pending` row,
followed atomically by a canonical `pending` scientific row and a diagnostic
provider row. The recorder then leases a stable ordered batch, writes and
registers immutable raw partitions, records the exact partition hashes and raw
event IDs, commits application processing, and finally acknowledges. Never
manually mark a row acknowledged. A processing commit without an
acknowledgement is safe for acknowledgement-only recovery; a provider envelope
without canonical materialisation is not safe and becomes ingestion-fatal.
A current-generation positive request with a null `stream_owner_json` is
intentionally head-of-line blocked until the same recorder generation binds
its typed receipt. A later process must not attach a newly reused request ID
to that row; it recovers the old callback as raw-only evidence instead.

If cancellation occurs after active admission but before polling, the recorder
uses the immutable owner receipt and retains that owner's incremental quote
state across lease boundaries until SQLite shows no unacknowledged callbacks
for it. This is distinct from a callback classified after cancellation:
`expected_late_callback_after_cancellation` is diagnostic, already
acknowledged, and cannot update the live quote or option projection.

The default lease batch is bounded at 256 callbacks. This amortizes immutable
partition and checkpoint overhead across the 28 required bar streams while
leaving excess callbacks pending and recoverable. Processing refreshes
generation ownership every eight projected callbacks and between each grouped
immutable partition write, before compression, hashing, fsync, and atomic
rename work begins. The batch is acknowledged only after all partition
manifests and recorder-side processing state commit. Do not raise the bound
merely to hide a growing backlog; compare callback admission and processing
rates, oldest-unacknowledged age, lease freshness, and host capacity first.

An expired callback batch lease can be reclaimed by a newer generation. A
stale recorder-process owner is different: the executable cannot prove that
its in-memory active episodes and option subscriptions were continuous.
Startup therefore records `RECORDER_UNCLEAN_RESTART_STATE_UNCERTAIN` and keeps
the run ingestion-fatal. The replacement generation is allowed to reclaim
expired callback leases and persist/acknowledge raw evidence, but it does not
reconstruct or run the old generation's stateful normalizer. Existing
materialization is reused only after the manifest, sidecar, file, and content
hash verify. A not-yet-materialized callback becomes immutable
`raw_callback_envelope_event` evidence. Confirm its processing row says
`scientifically_blocked_raw_only` and
`scientific_projection_complete = 0`; this is recovery evidence, not a score.
The restored fatal latch continues to quarantine ordinary stream callbacks,
but an already-pending bounded bootstrap request (such as exact contract
qualification) may complete so the replacement generation can open its
raw-only recorder. Its original provider envelope retains the
`callback_after_data_loss_latch` classification and becomes diagnostic; it
cannot enter a scientific projection. A bootstrap provider envelope
quarantined by an older executable may be retired only through the explicit
bootstrap-envelope resolution operation, with exact failure classification,
non-empty operator evidence, and an atomic operational incident. Live quote,
bar, option, and depth envelopes are not eligible for that resolution.
The fatal latch cannot be cleared by artifact, context, capability, or
connection readiness. Every score, checkpoint, promotion, episode, and outcome
projection remains disabled. An interrupted streaming option episode also creates one
unresolved `PROCESS_RESTART_OPTION_CONTINUITY_LOST` scientific gap. Preserve
the raw recovery set; start a new preregistered run ID for scientifically valid
recording unless an explicit evidence audit resolves the latch and gaps. Old
pending or quarantined rows remain queryable but are excluded from a different
run's leasing, capacity, backlog, and health.

Do not edit or replace the activation receipt during this rollover. Startup
authorizes the replacement run ID only by reconstructing the receipt's exact
configuration hash with a prior run ID already persisted in
`prospective_run`. If no historical identity reproduces the immutable hash, or
if any non-operational configuration field differs, startup exits with
`blocked_existing_activation_configuration_mismatch`. The replacement run
therefore creates a new operational lineage without changing activation time,
artifacts, experiment receipts, thresholds, cohort, or causal rules.
Migration `0021_opening_reversal_activation_run_binding_v1` records the V1 and
V1.1 receipts for the replacement run as byte-identical, hash-identical
bindings to each original activation row. Database triggers reject a binding
whose timestamps, hashes, frozen rules, receipt JSON, reserved-line count, or
no-order flag differs, and reject every later update or delete.

Official IBKR dependency maintenance does not rewrite that activation. The
activation's API and Gateway values remain the immutable first-collection
baseline, while the active official-archive provenance and Gateway release
manifest verify the maintained runtime versions. Startup reconstructs the old
configuration hash with only the baseline Gateway identity; every signal,
capacity, cohort, threshold, artifact, and safety field must still match. The
current API and Gateway identities are recorded separately in capability
evidence. For activations created before mandatory option accounting, startup
also reconstructs the legacy shape without the three output-only commission,
regulatory-fee, and exchange-fee fields; their current configured values are
still recorded with each observation and cannot affect leader admission.
Missing or unverified current identities remain fail-closed.

A graceful process stop is not allowed to hide an incomplete streaming option
episode behind `STOPPED_CLEANLY`: shutdown records the same scientific gap and
an ingestion-fatal continuity incident before releasing the recorder lease.

## 11. Interpret recorder health

The web process reads `recorder_operational_state_v1`; it does not infer
activity from configured paths or historical data. Treat only
`RECORDING_HEALTHY` as green. That state requires all of the following:

- the current generation owns a fresh recorder lease;
- process, callback-received, durable-admission, raw-storage, and inbox-ack
  heartbeats are fresh when the market calendar expects callbacks;
- inbox count and oldest-unacknowledged age remain within configured bounds;
- the expected IBKR connection and market-data mode are observed;
- every expected frozen artifact has a verified active-generation receipt;
- scientific prerequisites pass, the broker mutation count is zero, and no
  required stream gap or fatal latch is active.

When only the always-on five-minute bar surface is active, a normalised
immutable raw partition is expected when the next bar proves the prior bar
complete. The default raw-storage freshness bound is therefore six minutes.
This does not relax the separate 30-second callback-receipt, durable-admission,
or inbox-acknowledgement bounds. Once Level I is promoted, partitions normally
arrive more often; a raw-storage age beyond six minutes during an open session
is degraded evidence and must be investigated.

`MARKET_CLOSED` is expected outside the configured session and does not demand
a callback heartbeat. `RECORDING_DEGRADED` means evidence is still being
recorded but a nonfatal live condition is unhealthy. `SCIENTIFICALLY_BLOCKED`,
`INGESTION_FATAL`, and `STORAGE_FATAL` prohibit scoring. `STALE_HEARTBEAT`
means the process or ownership evidence is stale even if historical rows are
present. `RECONNECTING`, `STARTING`, `STOPPING`, `STOPPED_CLEANLY`,
`WAITING_FOR_PROSPECTIVE_START`, and `INACTIVE` are literal lifecycle states.

Inspect the independent times rather than one combined heartbeat:

```bash
curl --fail --silent http://127.0.0.1:8765/api/recorder/status \
  | jq '.operational | {state, reason_code, timestamps, inbox, conditions}'
```

## 12. Respond to ingestion or storage fatal state

Do not restart repeatedly, clear a row, or assume a socket reconnect repaired
the evidence boundary.

1. Stop the recorder while leaving the read-only web service available.
2. Take a checked SQLite backup and preserve the immutable raw directory,
   staged directory, and quarantine directory together.
3. Inspect the active `recorder_fatal_latch_v1`,
   `operational_incident_v1`, `callback_inbox_v1`,
   `callback_raw_materialization_v1`, `callback_processing_commit_v1`,
   `raw_partition_manifest_v0`, and
   `gap_incident_v1` rows. Do not include secrets in the incident record.
4. For storage failures, independently hash every named partition and sidecar.
   A controlled recorder startup runs deterministic staged-file and manifest
   reconciliation, but the persisted fatal latch still blocks scoring.
5. For poison or callback-loss failures, identify the first possibly lost
   source sequence. Never fabricate a callback or coerce a missing value to
   zero.
6. Record the audit and recovery evidence in the operator incident system. If
   loss cannot be disproved, keep the original run invalid and start a new
   preregistered run ID. No reconnect or process restart automatically clears
   the old latch.

Optional-feed gaps may resolve when a complete valid book/capture returns and
are counted separately. Required-stream gaps and any possibly lost callback
invalidate scientific scoring until the evidence audit is complete.

## 13. View logs and health

```bash
sudo systemctl status stocker-recorder.service stocker-web.service
sudo journalctl -u stocker-recorder.service -n 200 --no-pager
sudo journalctl -u stocker-web.service -n 200 --no-pager
sudo journalctl -u stocker-recorder.service -f
```

Do not paste the environment file into logs or support tickets.

Every HTTP response includes the same value in `X-Request-ID` and
`X-Correlation-ID`. Production bodies remain generic; unexpected failures
return exactly `{"detail":"internal_error"}`. Structured server logs contain the request ID,
method, route template, response status, elapsed milliseconds, safe run ID,
exception class and stack trace, SQLite operation count/duration, Parquet
files and row groups examined/read, input/output row counts, and replay
execution ID. They never include authorization headers, cookies, tokens,
credentials, full configuration, or raw market-data payloads.

To correlate a browser response:

```bash
REQUEST_ID=REPLACE_WITH_X_REQUEST_ID
sudo journalctl -u stocker-web.service --since "15 minutes ago" --no-pager \
  | grep --fixed-strings "$REQUEST_ID"
```

An evidence-validity error must also appear as a persisted operational
incident, not only a log line.

## 14. Graceful shutdown

```bash
sudo systemctl stop stocker-recorder.service
sudo systemctl stop stocker-web.service
sudo systemctl status stocker-recorder.service stocker-web.service
```

The recorder handles `SIGTERM`, moves its generation to `STOPPING`, cancels
temporary subscriptions, disconnects the callback source, and records
`STOPPED_CLEANLY` only after successful cleanup. An already leased or pending
callback is not declared processed during shutdown: it remains in the durable
inbox for generation-fenced restart recovery. A fatal latch remains fatal
across shutdown and restart.

## 15. Roll back the application release

The database and bundle store remain untouched:

```bash
sudo systemctl stop stocker-recorder.service stocker-web.service
sudo ln -s /opt/stocker/releases/REPLACE_WITH_PRIOR_COMMIT /opt/stocker/current.next
sudo mv -Tf /opt/stocker/current.next /opt/stocker/current
sudo systemctl start stocker-web.service
sudo systemctl start stocker-recorder.service
curl --fail --silent http://127.0.0.1:8765/api/health | jq .
```

Confirm the older release supports the already-applied database schema before
rollback. Confirm it also contains the exact `ibapi` tree named by the active
provenance record. Never restore an older database as part of application
rollback.

## 16. Roll back the active bundle

Bundle rollback changes only the atomic active pointer:

```bash
sudo -u stocker /opt/stocker/current/.venv/bin/stocker-prospective bundle activate \
  REPLACE_WITH_PRIOR_BUNDLE_ID \
  --bundle-root /var/lib/stocker/bundles \
  --operator REPLACE_WITH_OPERATOR_ID \
  --expected-current REPLACE_WITH_CURRENT_BUNDLE_ID
```

Use a new prospective run for a changed active bundle. Do not rewrite scores
already recorded under the prior bundle.

## 17. Back up and restore the database

Install and enable the daily checked backup timer:

```bash
sudo install -o root -g root -m 0644 \
  /opt/stocker/current/deploy/systemd/stocker-backup.service \
  /opt/stocker/current/deploy/systemd/stocker-backup.timer \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stocker-backup.timer
sudo systemctl start stocker-backup.service
sudo systemctl status stocker-backup.service stocker-backup.timer
sudo ls -l /var/lib/stocker/backups
```

The timer starts at 00:05 UTC with up to five minutes of random delay, keeping
backup work outside the Gateway's short-lived 23:45 restart-resume window. The
SQLite backup includes the durable callback inbox and every hardening state
table. It does not copy immutable Parquet files; replicate the entire raw
partition tree, metadata sidecars, staged/quarantine tree, SQLite backup, and
its hash manifest as one recovery set. The application never automatically
deletes evidence or backups. Configure encrypted off-host replication and
retention in the server's approved backup system; deletion remains an explicit
operator action.

Restore is deliberately manual and requires both services stopped:

```bash
sudo systemctl stop stocker-recorder.service stocker-web.service
sudo -u stocker sqlite3 /var/lib/stocker/backups/REPLACE_WITH_BACKUP.sqlite3 \
  'PRAGMA quick_check;'
sudo -u stocker sqlite3 /var/lib/stocker/prospective/restored.sqlite3 \
  ".restore /var/lib/stocker/backups/REPLACE_WITH_BACKUP.sqlite3"
sudo -u stocker sqlite3 /var/lib/stocker/prospective/restored.sqlite3 \
  'PRAGMA quick_check;'
sudo -u stocker mv /var/lib/stocker/prospective/prospective.sqlite3 \
  /var/lib/stocker/prospective/pre-restore.sqlite3
sudo -u stocker mv /var/lib/stocker/prospective/restored.sqlite3 \
  /var/lib/stocker/prospective/prospective.sqlite3
sudo systemctl start stocker-web.service stocker-recorder.service
```

Retain `pre-restore.sqlite3` until the restored run is independently audited.

## Replay ownership and dashboard polling

Each replay start has a UUID execution ID and monotonic generation. `stop`
changes only that execution to `STOPPING`, signals its private event, and joins
for the configured `replay_stop_timeout_seconds`. Do not start another replay
while an earlier worker can still mutate state. If a worker ignores stop, the
API exposes `stop_failed`, clean stop is false, and restart remains blocked. A
stale worker cannot publish completion, digest, counters, or errors into a
newer execution. Application shutdown calls the same bounded stop path.
Replay has no broker object; both `ibkr_connections_attempted` and
`broker_state_mutated` remain fixed at zero.

Replay global ordering is deliberately bounded rather than silently
unbounded. The defaults are 250,000 records and 64 MiB of materialised
canonical data. The preflight includes manifest row counts, SQLite scalar
bytes, Parquet uncompressed row-group bytes, JSON expansion/depth, and encoded
record bytes. Exceeding a limit produces a `blocked_replay_*_limit_exceeded`
failure; raise a limit only after measuring a copied or synthetic fixture,
never by trying the active database.

The browser polling budget is:

| Tier | Interval | Requests |
| --- | --- | --- |
| Fast | 15 seconds | One compact `/api/dashboard/summary`; no Parquet, report enumeration, transfer history, universe history, or static audit reconstruction. |
| Slow | 90 seconds | Only lightweight endpoints associated with the visible screen. |
| Manual/very slow | 300 seconds, screen activation, or explicit refresh | Audit, reports, concentration/source-transfer history, virtual ledgers, and completed shadow outcomes for that screen only. |

The busiest scheduled screen is approximately 6.34 requests/minute for one
visible tab and 12.67 for two tabs. The scheduler never overlaps refresh
generations, explicit refresh cancels/supersedes prior work, and hidden tabs
pause. Opening the application paints the fast summary before detailed screen
requests; changing screens does not re-fetch the fast summary. The legacy full
snapshot route is not automatically polled. Static route/adapter, bundle,
parity, provenance, and read-only checks are warmed at application creation and
held in the explicit five-minute cache. Recorder heartbeat, inbox, bar,
checkpoint, episode, connection, and subscription state is never held in that
cache.

Audit pages are capped by `audit_page_maximum_items` and use the opaque
`next_cursor` from `/api/audit/events`; do not construct cursors manually.
The list comes from the indexed SQLite audit-identity projection and does not
open raw Parquet. Raw rows are read only through an explicit
`/api/audit/raw-events/{content_hash}` request, from one partition, with
projected columns and the configured input-row bound.

Episode quote/depth views use only manifest partitions and row groups
overlapping the episode window. They project required columns, fail closed if
the input-row ceiling would be exceeded, deterministically sample chart
points, and retain the first and last valid observations. The quiet capture
table shows the latest persisted bid/ask for each selected quiet contract,
while the finalized quiet table shows immutable per-leg quote evidence and
conservative virtual P&L. Neither table represents an IBKR account or order.

### Reliability measurement evidence

The committed measurement is temporary and synthetic:

```bash
uv run python scripts/measure_prospective_web_reliability.py
uv run pytest \
  tests/test_fresh_episode_projection.py \
  tests/test_parquet_read_projection.py \
  tests/test_web_polling_contract.py
```

On 2026-07-31, 30 warmed sequential FastAPI `TestClient` summary requests had
a 3.626 ms median and 3.782 ms p95 on the baseline synthetic database. After
adding 10,000 synthetic manifest/audit-projection rows, the median was
3.613 ms and p95 was 3.817 ms; response sizes were 29,343 and 29,366 bytes.
This measures application projection cost on the test host, not network or
production-host latency.

On 2026-08-03, the target-branch baseline measured before this refactor was
6.021 ms median / 6.804 ms p95 / 7.154 ms maximum end-to-end in `TestClient`,
with a 29,343-byte response. A representative warmed server request performed
33 SQLite operations (2.536 ms SQLite time) and took 4.333 ms server time. The
same script after the refactor measured 3.373 ms median / 3.863 ms p95 / 4.058
ms maximum end-to-end and a 1,981-byte response. Structured request metrics
measured 2.548 ms median / 2.926 ms p95 / 3.085 ms maximum server time, exactly
14 SQLite operations per request, 1.776 ms median SQLite time, zero Parquet
files, and zero Parquet input rows. Adding 10,000 historical manifest rows left
the revised route at 3.299 ms median / 3.673 ms p95 / 3.857 ms maximum
end-to-end and the same 1,981-byte response. These synthetic local results
enforce route shape and show headroom; they are not a production-host latency
guarantee.

The enlarged quote fixture contains 1,000 rows in ten row groups, including a
100-row episode window and an unprojected 2 KiB payload column. The test reads
one row group and seven required columns, examines 100 input rows, returns 100
valid rows, and deterministically emits ten chart points with the first and
last retained. The raw-tail fixture contains 2,000 rows in 40 row groups; a
five-row request reads one 50-row group. Oversized row groups fail before the
scanner runs. Static polling tests calculate 7.000 scheduled requests per
minute on the busiest visible screen and verify that the fast function
contains no audit, report, transfer, shadow-history, concentration, or episode
detail path.

## Gap incident lifecycle

One real discontinuity creates one stable `gap_incident_v1` row even when it
affects several Parquet partitions. The operational projection separately
counts active gaps, resolved recoverable gaps, unresolved scientific gaps,
connection interruptions, and optional-feed degradation. Do not sum legacy
partition `gap_count` values. Resolution requires a timestamp and evidence;
optional recovery cannot convert a required scientific gap into valid
evidence without its own audit.

## Troubleshooting

| Symptom | Meaning and operator action |
| --- | --- |
| Stale lease/heartbeat | Confirm the configured owner and process are alive. Stop duplicate processes and preserve the DB/WAL. A replacement generation latches `RECORDER_UNCLEAN_RESTART_STATE_UNCERTAIN`, reclaims expired callback batches for raw-only persistence/acknowledgement, and keeps all scientific projection blocked. Interrupted option streams create unresolved scientific gaps. Use a new run ID for valid recording unless an explicit audit resolves every latch/gap. Historical rows are not proof of life. |
| `CALLBACK_OVERFLOW` | The bounded durable inbox could not admit another callback. Stop recording, preserve inbox/raw state, identify the first possibly lost sequence, and treat the run as ingestion-fatal. Do not raise the bound as a substitute for the audit. |
| Growing unacknowledged backlog | Compare callback, raw-commit, and ack heartbeats. Inspect leased attempts and storage incidents. A processing-committed batch may be acknowledged after lease recovery; a poison or interrupted `provider_pending` row is quarantined and fatal. |
| Valid Parquet but missing manifest | Preserve the sidecar and restart under controlled conditions. Reconciliation verifies the hash and registers the manifest idempotently before inbox acknowledgement. |
| Manifest points to missing file | Treat as `STORAGE_FATAL`; restore the exact hash-matching immutable file from the paired recovery set or retain the run as invalid. |
| IBKR reconnect | `RECONNECTING` is expected while ownership is rebuilt. Lost-data reconnect creates one connection gap and rebuilt request generation. The socket reconnect never clears a prior fatal latch. |
| Late callback | Expected post-cancel callbacks remain diagnostic through the expiring tombstone and cannot mutate the active stream. Unknown or previous-generation behavior is visible in incidents. |
| Invalid artifact hash | Compare expected/observed hashes and activation receipt in runtime verification. Replace neither in place; activate the correct immutable bundle and begin the appropriate generation/run. |
| Replay worker will not stop | Keep the controller in the explicit failed-stop state, do not start a replacement worker, collect its termination reason, and repair the isolated fixture/worker first. |
| Gateway process restarted but API port is absent | The Java process and loopback proxy are not proof of an authenticated API session. Inspect `stocker-ibgateway-daily-readiness.service` and confirm the configured upstream port is listening. Confirm the unit still has `ExitType=cgroup`, `Restart=always`, and `RestartSec=1`; default main-process exit semantics kill the authenticated handoff child. Authenticate only in the official Gateway window, keep Read-Only API and localhost-only enabled, and confirm `AutoRestart=1`. A broker weekly reset can still require manual credentials and 2FA; never store them in Stocker. |
| Recorder exits at after-session capture with `prospective run identity mismatch` | Preserve the immutable first-activation `prospective_run` row. After-session source capture must obtain metadata from the frozen application's activation metadata factory; the current release SHA belongs in generation artifact receipts, not a replacement run identity. Do not edit the existing run row to match a deployment. |
| Universe Tape has symbols and bars but no probabilities | Confirm a frozen checkpoint (6, 8, …, 34) completed with the prior-session activity baseline and Group-O package available. A pending cross-vendor bar diagnostic does not suppress or de-authorize an otherwise valid IBKR score. Bid/ask remains intentionally blank until the bounded promotion scheduler arms Level I for a low/high candidate. |
| Virtual ledger is empty | Confirm the selected run has an eligible receipt/observation and bounded contract plan. The quiet capture table may show current persisted bid/ask before a structure closes; a finalized row additionally requires complete immutable per-leg entry/exit quotes. Inspect the wait/invalid reason and never manufacture a position from configuration, a latest quote, or a partial leg. |

## Secure browser access

The default bind is `127.0.0.1`; `0.0.0.0` and `::` are rejected.

### SSH tunnel

From the operator workstation:

```bash
ssh -N -L 8765:127.0.0.1:8765 operator@SERVER
```

Then open `http://127.0.0.1:8765`. The IBKR socket is not tunneled or exposed.

### Private VPN

Bind `web.host` only to the exact private VPN interface address and include
that hostname/address in `allowed_hosts`. Keep the host firewall restricted to
the VPN subnet.

### Authenticated TLS reverse proxy

Prefer leaving Stocker on loopback and proxying from a same-host authenticated
TLS service. Keep `trust_proxy_headers: false` unless the proxy source IP is
explicitly listed. When application authentication is enabled, the proxy may
send the token as a bearer credential or a `Secure`, `HttpOnly`,
`SameSite=Strict`, `__Host-stocker_session` cookie.

There are no state-changing web routes in this slice, so there is no CSRF
mutation surface. Any future recorder start/stop control must add authenticated
operator authorization and CSRF protection and still cannot enable orders.
