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
/etc/stocker/prospective.yaml         runtime configuration (root:stocker 0640)
/etc/stocker/stocker.env              release identity and secrets (root:stocker 0640)
/var/lib/stocker/prospective/         SQLite database and WAL
/var/lib/stocker/bundles/             immutable bundle store + active pointer
/var/lib/stocker/daily-context/       signed package store + session pointers
/var/lib/stocker/backups/             checked immutable SQLite backups
```

Application rollback changes only `/opt/stocker/current`. It never replaces,
restores, or rolls back `/var/lib/stocker`.

## 1. Prepare the service user and directories

On the dedicated server:

```bash
sudo useradd --system --home-dir /var/lib/stocker --shell /usr/sbin/nologin stocker
sudo install -d -o root -g root -m 0755 /opt/stocker/releases
sudo install -d -o root -g root -m 0755 /etc/stocker
sudo install -d -o stocker -g stocker -m 0750 /var/lib/stocker
sudo install -d -o stocker -g stocker -m 0750 /var/lib/stocker/prospective
sudo install -d -o stocker -g stocker -m 0750 /var/lib/stocker/bundles
sudo install -d -o stocker -g stocker -m 0750 /var/lib/stocker/daily-context/incoming
sudo install -d -o stocker -g stocker -m 0700 /var/lib/stocker/backups
```

Do not add `stocker` to privileged groups. Keep the IBKR GUI/session under an
independently managed desktop login if required by the host.

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

This installs the server dependency group into the release-local `.venv`.
Production data does not enter the virtual environment.

## 3. Install the optional official IBKR Python API

Recheck the official download page at install time:
<https://interactivebrokers.github.io/>.

Do not run `pip install ibapi` from a package registry. On 2026-07-24, Python
was provided in the official **Latest** Mac/Unix archive, under
`source/pythonclient`.

After an operator manually downloads the official archive:

```bash
sha256sum /secure-transfer/TWS_API_REPLACE_WITH_VERSION.zip
unzip -q /secure-transfer/TWS_API_REPLACE_WITH_VERSION.zip -d /secure-transfer/tws-api
cat /secure-transfer/tws-api/IBJts/source/Api_VersionNum.txt
sudo -u stocker uv pip install \
  --python /opt/stocker/current/.venv/bin/python \
  /secure-transfer/tws-api/IBJts/source/pythonclient
/opt/stocker/current/.venv/bin/python -c 'import ibapi; print(ibapi.__file__)'
```

Record the archive hash, API version, TWS/IB Gateway version, operator, and
install time in the server change record. Do not copy the extracted source into
Git or a Stocker bundle.

If the official client is absent, replay remains available and IBKR mode
returns `blocked_official_ibkr_api_not_installed`.

## 4. Copy a versioned application release

On the research/development machine, from a clean intended commit:

```bash
git rev-parse HEAD
git status --short
git archive --format=tar.gz --output=/tmp/stocker-REPLACE_WITH_GIT_COMMIT.tar.gz HEAD
sha256sum /tmp/stocker-REPLACE_WITH_GIT_COMMIT.tar.gz
scp /tmp/stocker-REPLACE_WITH_GIT_COMMIT.tar.gz operator@SERVER:/secure-transfer/
```

On the server:

```bash
sudo install -d -o stocker -g stocker -m 0755 \
  /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT
sudo -u stocker tar -xzf \
  /secure-transfer/stocker-REPLACE_WITH_GIT_COMMIT.tar.gz \
  -C /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT
cd /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT
sudo -u stocker uv sync --locked --no-editable --no-default-groups --group server
sudo ln -s /opt/stocker/releases/REPLACE_WITH_GIT_COMMIT /opt/stocker/current.next
sudo mv -T /opt/stocker/current.next /opt/stocker/current
```

Use `ln -sfn` instead of `ln -s` only when replacing an existing staging
symlink. Do not point a release at the database or bundle directory.

## 5. Build, transfer, install, and activate a frozen bundle

The inspected repository is currently missing the serialized M0, M1,
preprocessor, and approved dtype/missing-policy schema. The build command below
must fail with `blocked_missing_verified_frozen_bundle` until those approved
files are supplied. Never substitute `model_coefficients.json`.

On the research machine:

```bash
uv run stocker-prospective bundle build \
  --spec configs/prospective/bundle-spec.example.yaml \
  --output /tmp/REPLACE_WITH_BUNDLE_ID
uv run stocker-prospective bundle inspect /tmp/REPLACE_WITH_BUNDLE_ID
uv run stocker-prospective bundle verify /tmp/REPLACE_WITH_BUNDLE_ID
tar -C /tmp -czf /tmp/REPLACE_WITH_BUNDLE_ID.tar.gz REPLACE_WITH_BUNDLE_ID
scp /tmp/REPLACE_WITH_BUNDLE_ID.tar.gz operator@SERVER:/secure-transfer/
```

On the server:

```bash
tar -xzf /secure-transfer/REPLACE_WITH_BUNDLE_ID.tar.gz -C /secure-transfer
sudo -u stocker /opt/stocker/current/.venv/bin/stocker-prospective bundle verify \
  /secure-transfer/REPLACE_WITH_BUNDLE_ID
sudo -u stocker /opt/stocker/current/.venv/bin/stocker-prospective bundle install \
  /secure-transfer/REPLACE_WITH_BUNDLE_ID \
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
sudo install -o root -g stocker -m 0640 \
  /opt/stocker/current/configs/prospective/server.example.yaml \
  /etc/stocker/prospective.yaml
sudo install -o root -g stocker -m 0640 \
  /opt/stocker/current/deploy/stocker.env.example \
  /etc/stocker/stocker.env
sudoedit /etc/stocker/prospective.yaml
sudoedit /etc/stocker/stocker.env
```

Before first prospective start, set:

- a new immutable `runtime.run_id`;
- the exact future `prospective_start_utc`;
- server instance identity, release version, and Git commit;
- `record_only` or `shadow` only;
- the exact TWS/Gateway host, socket port, dedicated non-zero client ID, and
  paper environment;
- measured line budget, reserved headroom, and a request rate no greater than
  half that line budget;
- context-signing secret in the environment file; and
- optional web auth token only when `authentication_enabled: true`.

Never put IBKR username, password, or 2FA material in either file. Stocker has
no fields for them.

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
database path.

## 10. Start record-only IBKR mode

First configure TWS or IB Gateway manually:

1. Use the paper account.
2. Enable socket clients.
3. Keep Read-Only API enabled.
4. Keep localhost-only connections enabled.
5. Set and record the exact socket port.
6. Use a dedicated non-zero client ID that is not the Master client.
7. Authenticate manually, including 2FA.

Confirm the socket is not publicly bound:

```bash
sudo ss -ltnp
sudo ufw deny REPLACE_WITH_IBKR_SOCKET_PORT/tcp
```

Record-only IBKR diagnostics require the hash-verified registered universe and
the official dependency. A missing active frozen bundle remains an explicit
health blocker but does not prevent underlying evidence recording; a bundle
hash mismatch still fails closed. Record-only deliberately persists
source-semantic blockers instead of scoring. Shadow scoring requires a
verified active bundle, passing feature parity, and exact signed
previous-session context. Install the units only after the gates for the
selected mode are satisfied:

```bash
sudo install -o root -g root -m 0644 \
  /opt/stocker/current/deploy/systemd/stocker-recorder.service \
  /opt/stocker/current/deploy/systemd/stocker-web.service \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stocker-recorder.service
sudo systemctl enable --now stocker-web.service
```

Configuration/bundle-integrity exit code 78 is in
`RestartPreventExitStatus`; systemd will not loop on those failures.
Transient runtime failures use bounded systemd restart limits.

At the current repository state, IBKR mode is expected to block rather than
claim frozen M1 scoring.

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
rollback. Never restore an older database as part of application rollback.

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
