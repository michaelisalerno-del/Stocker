# Deployed SLRNO — Saxo migration and sampled book flow

Deployed to [the authenticated dashboard](https://139.59.178.164) on **2026-09-28 at
11:20:47 UTC / 12:20:47 Europe/London** following the user's deployment instruction.
Current runtime release: `b1e9f5c248efc2eaf2267a46b8b47119fc3109cc`, branch `codex/saxo-only`.
This includes migration `c74e68a`, safety/recovery fixes `b066386`, book flow `c809f9f`, and
OAuth bootstrap `b1e9f5c` deployed at **12:54:49 UTC**. The first supplied SIM AppSecret is held
in a stocker-owned 0600 file; no credential values appear in this repository. Account selection may
follow OAuth, but remains mandatory before data subscriptions or broker order permission.
[OAuth setup postflight](saxo-oauth-setup-20260928.json): healthy, disarmed, SIM login redirect verified.
The original sibling checkout and historical releases remain unchanged.

The application is deployed and healthy, but **Saxo is not authenticated**. CL, GC, NG, NQ and SI
show authentication required, unavailable quotes/L2/book flow, and zero prehistory. GC remains
monitor-only. There is no Bitcoin card or active stock scanner.

| Stage | Current evidence |
|---|---|
| IMPLEMENTED | Saxo migration and observation-only sampled book flow deployed |
| OFFLINE_TESTED | 437 Python tests, lint/format/types and installed smoke; prior browser/workload checks unchanged |
| AUTHENTICATED | **No** — SIM application credentials configured; browser login/grant pending |
| DATA_VERIFIED | **No** — actual Saxo contracts, quotes, history and permissions unverified |
| L2_VERIFIED | **No** — levels, counts, delay and cadence unverified for all five markets |
| RECORDER_VERIFIED | **Offline only** — no market data received; persistent capture permission-gated |
| DEPLOYED | **Yes** — symlink, service, authenticated HTTPS routes and served asset hash verified |

## Cutover evidence

[Sanitised server postflight](saxo-deployment-20260928.json), verified at 11:22:40 UTC.
Entries were paused while the old manager remained connected. The installed old release's
non-transmitting preflight verified the sole allowlisted managed account at **11:20:46 UTC**,
immediately before cutover: zero open orders, nonzero positions or returned executions. Futures
reservations/orders/fills/positions and FIRST4 orders/fills/positions were empty. No obligation was
abandoned, cancelled or liquidated. The old futures config was also disarmed for recovery safety.

The existing immutable release/locked uv/systemd/Caddy process was used. The server-only environment
installed 54 locked packages and passed installed startup/assets smoke as `stocker`. The app retains
its service user, authentication, filesystem/process hardening and loopback listener. The old
loopback-only **outbound** IP restriction was removed because Saxo requires external HTTPS/WebSocket
connections. No public application port was opened.

Thirteen app-owned definitions were stopped, disabled where applicable, and masked:

- Frozen scanner observation service and timer (`stocker-frozen-scanner-observation-20260923`).
- Gateway service, display, window manager and VNC services.
- Gateway loopback boundary service, proxy service and proxy socket.
- Gateway daily-readiness service and timer.
- IBKR API-update service and timer.

No stocker timers remain scheduled; reverse-dependency checks found no Gateway/scanner restart
path. Process/socket checks found no remaining Gateway, IBKR proxy, VNC or scanner process/listener.
Root/stocker crontabs were empty. Inspected cron/systemd definitions and processes revealed no
additional FMP/EODHD jobs or supervisor/container runtime. FIRST4 remains stopped. Historical data,
broker credentials/accounts/subscriptions and old releases are preserved. No financial account,
subscription or legal-agreement action was performed.

## Active configuration and checks

- `stocker-v1.service`: active/running, zero automatic restarts at postflight.
- `/opt/stocker/current` → `/opt/stocker/releases/b1e9f5c248efc2eaf2267a46b8b47119fc3109cc`.
- Config: `/etc/stocker/v1/saxo.sim.yaml`, mode 0640, root:stocker.
- Data: **SAXO_SIM**; execution: **DISABLED**; armed: **false**; **LIVE ORDERS DISABLED**.
- New ledger: `/var/lib/stocker/v1/saxo-sim-disabled.sqlite3`; old ledgers retained separately.
- Recorder: 32 MiB RAM, 8 MiB/512-item queue, 2 GiB event limit, 2 GiB reserve; persistent capture off.
- Callback: `https://139.59.178.164/oauth/saxo/callback`; access logging disabled, authentication retained.
- Proxy permits OAuth start, non-transmitting preflight, disarm and explicit archive prune;
  paper arming remains excluded from its write allowlist.

Authenticated HTTPS and trusted loopback checks returned 200 for all five pages, health/system/
overview/history/recordings and JS/CSS. Served JS matches the installed book-flow release hash.
Anonymous HTTPS returned 401; untrusted loopback and foreign-origin requests returned 403.
An invalid cross-site OAuth callback returned 400. Bitcoin history selection returned 422.
At initial cutover, non-transmitting preflight returned **409 NOT_CONFIGURED**. The subsequent
OAuth setup now reports login required; no authentication or market-data verification is claimed.
External direct-port access timed out; the application listens on `127.0.0.1:8765` only.
All three ledgers passed integrity checks and contain no orders or positions.
The service used approximately 54 MiB at postflight; this is idle evidence, not a feed benchmark.

## Recovery and remaining setup

Protected backups: `/var/lib/stocker/backups/saxo-cutover-20260928` (root-only). They contain
consistent SQLite backups, configuration/proxy/unit originals, SHA-256 manifests, old release pointer,
fresh broker snapshots, staged definitions and deployment/postflight reports. Parked unit originals
are in `parked-units/`; `/etc/systemd/system` contains masks.
See [controlled rollback](DEPLOYMENT.md). Never restore an old ledger over newer activity, unmask
old providers or rearm automatically. Default recovery leaves execution stopped/disarmed.

The subsequent OAuth upgrade backup is `/var/lib/stocker/backups/saxo-oauth-20260928`, including
the prior config/ledger and protected credential backup. Both backups have SHA-256 manifests.

Remaining user actions: complete OAuth in System, select/verify the returned SIM account,
discover/pin actual futures UICs, verify
per-market quote/options/history/L2 permissions, and document recording rights before enabling
persistent capture. Follow [Saxo setup](SAXO-SETUP.md); do not put secrets in chat or Git.
Listed-option product/expiry/cutoff/delta/fee approvals remain separate execution blockers.

## OAuth bootstrap Standards review

No concrete findings. Private file permissions and SIM/LIVE separation remain enforced; missing
account selection cannot grant subscription or order permissions.

## OAuth bootstrap Spec review

No concrete findings. App-only OAuth setup is enabled; explicit verified account selection still
precedes streaming and execution. The offline regression covers authentication, account reads and
order rejection. Standards: 0 open findings; Spec: 0 open findings.
