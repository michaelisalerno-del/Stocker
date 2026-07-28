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

Do not run `pip install ibapi` from a package registry. On 2026-07-25, Python
was provided in the official **Latest** Mac/Unix archive. The first verified
server archive is `twsapi_macunix.1048.01.zip`, which contains
`API_Version=10.48.01` and installs `ibapi==10.48.1`. Its expected SHA-256 is
`0446c403cdfd3a059685c5e11814b32e0b811fdf5e1f68564f8e08b655e49547`.

The operator must accept IBKR's licence and download the archive manually.
Copy it directly into restricted server staging:

```bash
scp /path/to/twsapi_macunix.1048.01.zip \
  root@SERVER:/var/lib/stocker/secure-transfer/twsapi_macunix.1048.01.zip
```

Then verify it and install it into the **staged release** before that release is
made immutable or promoted:

```bash
sudo chown root:stocker \
  /var/lib/stocker/secure-transfer/twsapi_macunix.1048.01.zip
sudo chmod 0640 \
  /var/lib/stocker/secure-transfer/twsapi_macunix.1048.01.zip
sha256sum /var/lib/stocker/secure-transfer/twsapi_macunix.1048.01.zip
IBKR_EXTRACT_DIR="$(
  sudo -u stocker mktemp -d /var/lib/stocker/ibkr-api-extract.XXXXXX
)"
sudo -u stocker python3.12 -m zipfile -e \
  /var/lib/stocker/secure-transfer/twsapi_macunix.1048.01.zip \
  "$IBKR_EXTRACT_DIR"
cat "$IBKR_EXTRACT_DIR/IBJts/API_VersionNum.txt"
sudo -u stocker uv pip install \
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
  --archive /var/lib/stocker/secure-transfer/twsapi_macunix.1048.01.zip \
  --installed-package-root "$IBKR_PACKAGE_ROOT" \
  --provenance /var/lib/stocker/ibkr-api/provenance/10.48.1.json \
  --operator REPLACE_WITH_OPERATOR_ID
sudo chown root:stocker-readers \
  /var/lib/stocker/ibkr-api/provenance/10.48.1.json
sudo chmod 0640 /var/lib/stocker/ibkr-api/provenance/10.48.1.json
sudo ln -s provenance/10.48.1.json \
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
  /opt/stocker/current/deploy/scripts/run-ibgateway-loopback-proxy.sh \
  /usr/local/libexec/stocker-ibgateway-loopback-proxy
sudo install -o root -g root -m 0644 \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-display.service \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-window-manager.service \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-vnc.service \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-loopback-boundary.service \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-loopback-proxy.socket \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway-loopback-proxy.service \
  /opt/stocker/current/deploy/systemd/stocker-ibgateway.service \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo /usr/local/libexec/stocker-install-ibgateway-loopback-boundary
sudo /usr/local/libexec/stocker-verify-ibgateway-loopback-boundary
sudo systemctl enable stocker-ibgateway-loopback-proxy.socket
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
5. Configure the supported daily auto-restart window if desired. Plan for
   manual authentication again after the weekly reset.

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
- context-signing secret in the environment file; and
- `parallel_validation.enabled: true` for the required first-20-session EODHD
  source-transfer comparison; and
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
20 stock bar streams plus only required proxies. Level II is disabled during
the 20-session engineering-transfer phase and remains optional afterward. The
recorder enforces the resolved historical-bar request allowance as a rolling
window during startup and reconnect restoration.

Put the token only in `/etc/stocker/stocker.env`:

```dotenv
# Required for the frozen activity baseline, D-1 Group O preparation, and
# source-transfer capture. Set the value with
# sudoedit; never put a real token in this runbook or a shell command.
EODHD_API_TOKEN=REPLACE_IN_EDITOR
```

Put only its non-secret status projection in
`/etc/stocker/stocker-web.env`:

```dotenv
STOCKER_EODHD_TOKEN_CONFIGURED=0
```

Put the context-signing secret only in `stocker.env`; never expose it to the web
process. Put an optional built-in web-auth token only in `stocker-web.env`.
Put the EODHD token only in `stocker.env`; the web process receives only a
boolean `credential_configured` projection. Set
`STOCKER_EODHD_TOKEN_CONFIGURED=1` in `stocker-web.env` only after the token is
present in `stocker.env`; otherwise leave it `0`. EODHD does not run as a
separate service. The recorder makes bounded requests after the session and
stores source-labelled evidence that is permanently ineligible for scoring.
Never put IBKR username, password, or 2FA material in any Stocker file. Stocker
has no fields for them.

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

```bash
sudo -u stocker /opt/stocker/current/.venv/bin/stocker-prospective db migrate \
  --database /var/lib/stocker/prospective/prospective.sqlite3
sudo -u stocker sqlite3 /var/lib/stocker/prospective/prospective.sqlite3 \
  'PRAGMA journal_mode; PRAGMA quick_check;'
```

Expected results include `wal` and `ok`.

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
curl --fail --silent http://127.0.0.1:8765/api/config/public | jq .
curl --fail --silent http://127.0.0.1:8765/openapi.json | jq '.paths | keys'
```

The replay health response should be correctly `blocked`, with synthetic data
visible and real-scoring/IBKR blockers present. It should state
`LIVE TRADING DISABLED`, `no_order_path_verified: true`, and expose no secret or
database path. The `ibkr_api` section independently reports whether the
first-party package is installed, source-tree verified, and current. That does
not imply that IB Gateway is installed, authenticated, connected, or permitted
for live market data.

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

Record-only IBKR diagnostics require the hash-verified registered universe and
the official dependency. A missing active frozen bundle remains an explicit
health blocker but does not prevent underlying evidence recording; a bundle
hash mismatch still fails closed. Installing the reconstructed bundle removes
only the missing-artifact blocker. Record-only deliberately persists
source-semantic blockers instead of scoring. Shadow scoring still requires
passing feature parity and exact signed previous-session context. Do not lower
either gate after observing outcomes. Install the units only after the gates
for the selected mode are satisfied:

When `parallel_validation.enabled` is true, the recorder also makes one bounded
five-minute-history request per anchor symbol after the configured
`capture_delay_seconds`. It does not backfill those rows into a score. The
fixed acceptance contract requires at least 20 complete prospective sessions
and an independent audit; collecting data does not automatically change
`feature-parity-m1.json`.

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
connections. Tests require destructive SQL to fail. On the deployed host,
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

## 11. View logs and health

```bash
sudo systemctl status stocker-recorder.service stocker-web.service
sudo journalctl -u stocker-recorder.service -n 200 --no-pager
sudo journalctl -u stocker-web.service -n 200 --no-pager
sudo journalctl -u stocker-recorder.service -f
```

Do not paste the environment file into logs or support tickets.

## 12. Graceful shutdown

```bash
sudo systemctl stop stocker-recorder.service
sudo systemctl stop stocker-web.service
sudo systemctl status stocker-recorder.service stocker-web.service
```

The recorder handles `SIGTERM`, stops admissions, cancels pending temporary
subscriptions, disconnects, and leaves missed captures missed.

## 13. Roll back the application release

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

## 14. Roll back the active bundle

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

## 15. Back up and restore the database

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

The application never automatically deletes evidence or backups. Configure
encrypted off-host replication and retention in the server's approved backup
system; deletion remains an explicit operator action.

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
