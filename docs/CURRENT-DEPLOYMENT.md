# Deployed SLRNO — Saxo migration and sampled book flow

Deployed to [the authenticated dashboard](https://139.59.178.164) on **2026-09-28 at
11:20:47 UTC / 12:20:47 Europe/London** following the user's deployment instruction.
Current runtime release: `94339bb27dec08996d9054500243993eadcf26ba`, branch `codex/saxo-only`.
This includes migration `c74e68a`, safety/recovery fixes `b066386`, book flow `c809f9f`, and
OAuth bootstrap `b1e9f5c` deployed at **12:54:49 UTC**. The first supplied SIM AppSecret is held
in a stocker-owned 0600 file; no credential values appear in this repository. Account selection may
follow OAuth, but remains mandatory before data subscriptions or broker order permission.
[OAuth setup postflight](saxo-oauth-setup-20260928.json): healthy, disarmed, SIM login redirect verified.
The OAuth Origin fix `d4df009` was deployed at **13:19:58 UTC**, with
[HTTPS postflight](saxo-oauth-origin-20260928.json) verified at **13:20:02 UTC**. The production
`no-referrer` policy caused the native Connect form to send `Origin: null`. Connect now uses
a same-origin fetch followed by navigation. The GET callback and top-level HTML landing permit
redirected navigation while retaining authentication, host checks, OAuth state and browser binding.
API, write and WebSocket origin protections remain enforced. Configuration and credentials are unchanged.
Safe OAuth failure diagnostics `7ec4c50` were deployed at **13:33:26 UTC**, with
[HTTPS postflight](saxo-oauth-diagnostics-20260928.json) verified at **13:33:29 UTC**. A subsequent
user login had reached the token exchange but failed with the old generic error. The next fresh
attempt now exposes only a fixed failure category in the callback and System; no provider payloads
or exception text are exposed. At that postflight the underlying authentication failure was unresolved.
The next browser attempt identified `OAUTH_TOKEN_HTTP_201`, confirmed in running service state.
The handler incorrectly rejected this response before validating its body. Release `94339bb`,
deployed at **13:43:07 UTC**, accepts HTTP 200/201 for issuance and renewal while retaining all
payload, lifetime, state and storage checks. [Postflight](saxo-oauth-created-20260928.json) passed
at **13:43:10 UTC**. A fresh browser grant is still required to verify actual token acceptance.
The original sibling checkout and historical releases remain unchanged.

The application is deployed and healthy, but **Saxo is not authenticated**. CL, GC, NG, NQ and SI
show authentication required, unavailable quotes/L2/book flow, and zero prehistory. GC remains
monitor-only. There is no Bitcoin card or active stock scanner.

| Stage | Current evidence |
|---|---|
| IMPLEMENTED | Saxo migration and observation-only sampled book flow deployed |
| OFFLINE_TESTED | 451 Python tests, lint/format/types and locked server-only smoke; unchanged browser suites and workload previously passed |
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
- `/opt/stocker/current` → `/opt/stocker/releases/94339bb27dec08996d9054500243993eadcf26ba`.
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
The Origin fix backup is `/var/lib/stocker/backups/saxo-oauth-origin-20260928`, including the previous
release pointer, consistent ledger backup, unchanged config/unit and postflight/hash manifests.
Rollback this fix by restoring the previous release symlink after checking current obligations;
retain the current ledger, credentials and tokens, and keep execution disabled. The previous UI has
the reported OAuth defect. No provider units need changing for this rollback.
The diagnostic upgrade backup is `/var/lib/stocker/backups/saxo-oauth-diagnostics-20260928`;
its previous release is `d4df009`. Restore that code pointer only after an obligation check,
retaining newer ledger/token state. This restores the generic token error without affecting the
previously fixed Connect origin handling. All upgrades preserve the disabled execution config.
The HTTP 201 upgrade backup is `/var/lib/stocker/backups/saxo-oauth-created-20260928`;
its previous release is `7ec4c50`. Code rollback would restore the HTTP 201 rejection. Preserve
any newly issued tokens and ledger activity when reverting code; never automatically rearm.

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

## OAuth Origin fix verification

- Full Python suite: **438 passed**, 7 existing dependency/numerical warnings, 65 seconds.
- Ruff lint/format passed; mypy passed for 116 source files. Locked server-only smoke passed both
  locally in an isolated environment and on the staged release as `stocker`.
- Chromium regression reproduced the native form's null Origin and now passes the Connect → fixture
  identity provider → callback → System flow under `Referrer-Policy: no-referrer`. Existing dashboard
  focus/scroll/filter/DOM identity and mobile checks also passed. These are offline fixture checks.
- Deployed authenticated HTTPS: JSON OAuth start 200 with SIM authorize URL and secure binding cookie;
  redirected HTML navigation 200; invalid callback state 400 before token exchange; null-origin POST
  and cross-origin API 403; anonymous access 401. Served JS hash matches the release.
- All 13 legacy units remain masked. Service healthy, zero restarts, approximately 54 MiB idle memory;
  no orders, fills, positions or active reservations. No new feed workload measurement was needed.
- Standards review: **0 findings**. Spec review: **0 findings**.

Authentication is still pending at this postflight; no account-specific feed/L2 claim is made.
Reload System and start a fresh Connect attempt rather than reusing an earlier callback URL.

## Token-exchange investigation and diagnostics

The reported `AUTHENTICATION_EXPIRED_RECONNECT_REQUIRED` was confirmed in the running service;
no token file was created. Service-user token-directory write access and outbound network access
were verified. Requests with an intentionally invalid diagnostic grant returned non-JSON HTTP 401
for both supplied AppSecrets; the alternate was held in memory only. The configured secret was
also checked with Saxo's documented body-credential format. No credentials or configuration changed.
These probes are **not** evidence that either secret is invalid or that a real grant would fail
for the same reason. The old exception handler discarded the actual callback failure detail.

The regression test reproduced that masking with a sanitized HTML 401 fixture. The new handler
retains only a fixed stage category, numeric HTTP status or allowlisted OAuth error, and clears it
after successful authentication. Tests cover untrusted/malformed payloads, incomplete tokens,
network/storage failures and successful reconnect. **446 tests passed** in 63.48 seconds, with
7 existing warnings; lint, format, mypy (116 files), both browser suites and isolated/staged smoke
passed. Standards review: **0 findings**. Spec review: **0 findings**.

Production HTTPS and served asset checks passed; service healthy with zero restarts, all 13 legacy
units masked, execution disabled/disarmed and no orders sent. Authentication/data/L2 remain
unverified pending a fresh browser grant. Only the operator needs to complete that login;
do not send the callback URL or any secrets to obtain the safe server-side failure category.

## HTTP 201 token issuance fix

The actual user callback produced HTTP 201 from the SIM token endpoint; no response body or
tokens were retained by the old error path. A sanitized regression reproduced
`OAUTH_TOKEN_HTTP_201` using a complete token payload. The one-condition fix now routes 200 and
201 through the same validation and atomic private-file write. It does not accept all 2xx responses.

Tests cover both statuses through the browser-bound callback, secure token storage, concurrent
refresh and SIM/LIVE separation. Malformed/incomplete 201 payloads and empty 204 responses remain
rejected. **451 tests passed** in 62.89 seconds, with 7 existing warnings. Ruff lint/format, mypy
(116 files), isolated server-only smoke and installed server smoke passed. The browser code and its
previous fixture test results are unchanged. Standards review: **0 findings**; Spec: **0 findings**.

Authenticated dashboard HTTPS/security checks passed after deployment. All 13 legacy units remain
masked, the service has zero restarts, credentials/configuration are unchanged, execution remains
disabled/disarmed, and no orders were sent. Saxo authentication, account data and L2 are still
unverified at this postflight; start a fresh Connect flow because the discarded grant cannot be reused.
