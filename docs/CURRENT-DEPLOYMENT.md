# Current deployed SLRNO — Saxo option context

Current application release: `b5b739e922e7b776cb0a97d6a2dff84229fdf85e`, deployed 2026-10-04 22:17–22:21 UTC (Sunday
18:17 New York, 17 minutes after the open, with `FORCE=1` on the user's decision: no open trades, and the only capture cut was
the 18:00 clock's, which this release removes; no configuration change; backup `review-20261004-review`): the fixes from the
2026-10-04 code review of main. Clocks are now 19:00–23:00 and 00:00–15:00 New York (CLOCK-SET-AMENDMENT-20261004.md; the
18:00 and 16:00 clocks could never enter; rule version unchanged); a clock decides on every trailing minute the chart stream
holds and may wait for up to five quiet minutes before the boundary (a wait that runs out is
`INCOMPLETE_COMPLETED_HISTORY_DEADLINE`); a market's or the FX feed's failed connect-time step is retried after a minute;
the recorder's sixteenth reference version evicts the oldest; an approved option root Saxo does not relate blocks the
family (`APPROVED_OPTION_ROOT_NOT_RELATED`); subscription replacement runs beside the socket reader; reconnect cleanup
deletes per context; SIM orders refused before sending are skips; paper exits retry instead of locking after three; a
reservation without an order closes; the reservations CHECK lists every ceiling; a 5xx from the token endpoint keeps the
refresh token; recorder write batches of one second; setup checklist and benchmarks follow the four markets. Verified after
the restart: served assets match, 27 of 48 lines, chart streams CURRENT on all four markets, option families loaded at the
same counts as before (CL 22 / ES 24 / GC 21 / NQ 17), five reference sessions per market, preflight clean, re-armed,
FullTradingAndChat with DataLevel Premium, 0 recording gaps, next clock 23:00 UTC. Rollback: `1cce2bb…`.

Previous release `1cce2bb3ed581e7f447872b0a4424cdb62e40d67`, deployed 2026-10-04 17:57–18:00 UTC (Sunday, inside the
restart window; no configuration change; backup `refresh-20261004-refresh`): the chart stream at Saxo's measured 300 ms floor
and the chain windows at 1,000 ms instead of 2,000 (MARKET-DATA.md; the probe also found the ~10-per-session cap on chain
subscriptions, of which the service holds eight). Verified after the restart: served assets match, 20 of 48 lines before
candidates warm, chart streams CURRENT on all four markets, preflight clean, re-armed, FullTradingAndChat with DataLevel
Premium, 0 recording gaps. Rollback: `93d188b…`.

Previous release `93d188bb7f9054946209d0e712d6642644bc1b5e`, deployed 2026-10-04 17:41–17:46 UTC (Sunday,
inside the restart window) with the configuration addition `next_contracts: CL → CLZ6` (NYMEX, Dec-26, Saxo LIVE UIC
31051110; previous file `saxo.live.paper.yaml.bak-20261004-nextstreams`, backup `nextstreams-20261004-nextstreams`):
the subscription guard 32 → 48 (Saxo accepted 55 in one session, `scripts/saxo_subscription_capacity_probe.py`), the
following expiry's chain window per market recorded as `OptionsChainNext`, and the CLZ6 pre-roll stream recorded as
`ContractFutures` with `role: NEXT_CONTRACT` (MARKET-DATA.md). Verified after the restart: served assets match,
20 of 48 lines before option candidates warm, chart streams CURRENT on all four markets, next-chain recording clean on
all four, CLZ6 SUBSCRIBED, preflight clean, re-armed, FullTradingAndChat with DataLevel Premium, 0 recording gaps.
Rollback: `ba082b8…` with the `.bak` configuration.

Previous release `ba082b829dd9ee4c0fd4a6df5323c5d0b9ababcf`, deployed 2026-10-04 17:12–17:16 UTC (Sunday,
inside the restart window; no configuration change; backup `chartstream-20261004-chartstream`): the clock's boundary
bar from Saxo's chart stream with REST as the check (MARKET-DATA.md, "Chart stream"). Verified after the restart:
served assets match, chart streams CURRENT on CL, ES, GC and NQ (real-time, ten samples each), 15 of 32 subscription
lines before option candidates warm, preflight clean, re-armed, FullTradingAndChat with DataLevel Premium, 0 recording
gaps. First live boundary decisions: the 19:00 New York clocks of 4 October; the tripwire (`chart_stream_problem`)
had not fired at deployment. Rollback: `86f463c…`.

Previous release `86f463ca1ff1bf21978ef992a1d8bada7638e3bf`, deployed 2026-10-04 15:47–16:05 UTC (Sunday,
inside the restart window) with the configuration change `option_chain_strikes` 25 → 61 and
`recorder.archive_max_bytes` 50 → 70 GiB (previous file `saxo.live.paper.yaml.bak-20261004-chain61`, backup
`chain61-20261004-chain61`): the review of what Saxo delivers against what is kept (DASHBOARD-AND-DATA.md,
"Archive runway"). Verified after the restart: served assets match the release, effective 61 strikes / 70 GiB, chain
window and chain recording clean on CL, ES, GC and NQ, option spaces 22/24/21/17 roots (ES needed its 15-minute retry
after Saxo's rate limit), preflight clean, re-armed, session FullTradingAndChat with DataLevel Premium, recorder
354 MB with 0 gaps. Rollback: `b7e21f4…` with the `.bak` configuration. Earlier the same day: `aa0cb76…` (dashboard
visual refresh, 14:25 UTC) and `b7e21f4…` (setup checklist folded, 14:56 UTC), both restart deploys, both re-armed;
every capture of 1–2 October had ended `INTERRUPTED_RESTART`, hence the restart window in DEPLOYMENT.md.

Previous release `f0cbd06d95689f308c3e370a98b1aa9c2524f8ab`, deployed 2026-10-01 21:08 UTC and re-armed
(session stayed FullTradingAndChat): the market page's L1 label follows the server's standing-quote time
(`l1.standing_receipt`) instead of the last price change, so a quiet market no longer flickers between
STALE OR MISSING and CURRENT on every refresh (user's report). Backup `l1-label-20261001`. Rollback: `8dbc6df…`.

Previous release `8dbc6df3267aa985a6a55280a213513bf342c6d2`, deployed 2026-10-01 20:51 UTC and re-armed
(session stayed FullTradingAndChat): the phone-first dashboard (user's priorities: overview, trades view, Market and
System tidy; see DASHBOARD-AND-DATA.md), per-trade results from the ledger, and the LIVE balance fix (the balance
subscription now sends the ClientKey; Saxo LIVE answered 400 without it). Verified after the restart: balance Current
with all three figures, today's 9 trades and net -£673.99 on the overview, trades-only history, labelled fills.
Backup `dashboard-20261001`. Rollback: `0732d95…` (same config).

Previous release `0732d9584fcc9a22a555f99d365f4d6557d580ef`, deployed 2026-10-01 20:23 UTC and re-armed
(session stayed FullTradingAndChat): the depth table drops the order-count imbalance column, which Saxo LIVE never
fills (`UsingOrders: false`, order fields repeat the sizes), and says why. Backup `depth-table-20261001`. Rollback:
`792e3cf…` (same config).

Previous release `792e3cf9c732152153cc98ed470f8cbee5db9d05`, deployed 2026-10-01 20:12 UTC by the
one-shot timer after the session (no open trades) and re-armed; the session stayed FullTradingAndChat. It carries
`ea1695e` (two-tap real-time confirm), `c902219` (an unsellable paper option is written off at zero after 60 s with
no bid), `587812f` (no-trade runs of up to 5 minutes filled at the last close), `462605f` (smile near the money,
traded-volume change), `942d2bd`/`ababf6d` (ladder bids on the right) and `792e3cf` (the look14 forecast and the
option's implied move against it, recorded at every clock and shown on the ticket). Config change at the same restart:
`event_calendar_file: /etc/stocker/v1/event-calendar.yaml`, the frozen `docs/release-calendar-2026-10-11.yaml`
(sha256 `43ae308b…`); previous config `saxo.live.paper.yaml.bak-20261001-calendar`. Verified after the restart: all
five option roots loaded, quotes DelayedByMinutes 0, calendar loaded, forecast OBSERVED in all five markets. Backup
`evening-fixes-20261001`. Rollback: `c15d937…` with the previous config.

Previous release `c15d937f8f19a1ad6d7bf36bdaf71e34ad8c781c`, deployed 2026-10-01 15:46 UTC and re-armed.
It carries `e33b6d4` (a stream reconnect keeps today's option spaces instead of reloading ~70 roots before the price
feeds) and fixes the start-up recording repair: `recover_members` looped forever when a gzip member ended inside an
output-limited tail, so the 15:17 UTC restart of `e33b6d4` hung at 100% CPU before the stream started (service stopped
at about 15:31 UTC; no open trades). Every capture segment was dry-run on copies first (all complete, nothing cut).
After the restart the session came back OrdersOnly (the real-time slot must be taken again) and entries remain paused
as they were. Backup `recorder-fix-20261001`. Rollback: `2555819…` / `acd4779…` (both have the old repair loop).

Previous release `acd47794caa03de5df0efbd6e332c936cc3d7ef1`, deployed 2026-10-01 10:23 UTC, re-armed.
At the user's request an unchanged quote now stands while Saxo confirms its feed (see the rulebook). Measured
before (06:16 New York): options current in 0-8% of samples except NQ (61-68%). After (06:29 New York, 90 s):
CL 88-100%, GC 99%, NQ 99%, SI 96-99%, while their last price changes were typically 3-65 s old. Backup
`standing-quotes-20261001`. Rollback: `2555819…`.

Previous release `2555819e914d9ce4a3b98cd0d546e881fbcea02d`, deployed 2026-10-01 07:28 UTC with
a config change, re-armed. At the user's request NG and SI mappings use `expiry_rule:
SAME_DAY_OR_NEXT_LISTED` (same-day option when one trades past the exit, otherwise the next listed expiry;
see the rulebook); CL, GC and NQ stay same-day. `option_candidate_window: 2` (07:33 UTC) so all five
markets' warm candidates fit the option subscription budget (with 3, GC was crowded out). Verified: SI now
warms Friday 2 Oct puts with real-time quotes; NG's Friday strikes are 0.05 apart and none was within the
approved 0.03 of the frozen 0.10 delta at 03:30 New York (2.85 at 0.04, 2.90 at 0.15). Backup
`expiry-rule-20261001` (with both previous configs). Rollback: `ebb4719…` with `saxo.live.paper.yaml.before`.

Previous release `ebb47197e80f652579c65977ffd42d7eca4c18dd`, deployed 2026-10-01 06:51 UTC
and re-armed. Real-time CME data arrived overnight (quotes, charts and 10x10 depth, DelayedByMinutes 0).
LIVE option reference data then exposed three SIM-era assumptions, fixed in `1cbe226` and `ebb4719`:
lot fields (IncrementSize/MinimumLotSize, not MinimumTradeSize/LotSize), NQ's price-tiered
TickSizeScheme, and cost conditions (CarryingCost, short-overnight only, is not charged; the GBP
account's 0.6% conversion markup is priced into costs and fills). Offline on the real LIVE CL/GC/NQ data
every check passes; on the server NQ waits only for the reference audit, CL/GC also for a fresh
option quote. Backups `live-option-schema-20261001`, `live-costs-20261001`. Rollback: `a65b52e…`.

Previous release `a65b52e22fa6bdd8436c48a7be3c96dfa408eb60`, deployed 2026-09-30 23:19 UTC and
re-armed: a header chip and a line under the button show Real-time ON / ON but Saxo still delayed / OFF.
The session was made primary at 23:15 UTC (FullTradingAndChat, DataLevel Premium; it survived the restart),
yet fresh subscriptions and charts still report DelayedByMinutes 10: Saxo is not yet granting real-time CME
data to the API (Open API Access market data in SaxoTraderGO, or the subscription itself). Backup
`/var/lib/stocker/backups/realtime-chip-20260930`. Rollback: symlink to `a70f244…`, restart.

Previous release `a70f2442a3d6391796bbe5e2492224bc2f57b964`, deployed 2026-09-30 23:06 UTC
and re-armed after preflight. A repeat real-time click now renews the price streams too (a click made
before the data subscription starts would otherwise leave them delayed), and the market card rounds the
futures spread and change. Verified backup `/var/lib/stocker/backups/repeat-click-20260930`; CI green;
installed smoke passed; all five option families reloaded. Rollback: symlink to `49c9390…`, restart.

Previous release `49c9390144af3570cf1e08a2e8d90b6afc6594e9` (branch
`codex/live-paper-l2-observation`), deployed 2026-09-30 22:40 UTC, **armed** (INTERNAL_PAPER, re-armed with
`ENABLE PAPER ONLY` after preflight). It adds the System page's **Use real-time in SLRNO** button: Saxo
sends real-time prices only to the user's one FullTradingAndChat session, the LIVE session is OrdersOnly,
and entries require it. The user chose an explicit click over automatic re-taking (SaxoTraderGO shares the
slot). Verified backup `/var/lib/stocker/backups/primary-session-20260930`; installed smoke passed; all
five option families loaded after restart. Rollback: symlink to `/opt/stocker/releases/ebb2665…`, restart.

Previous release `ebb2665e9d2c32addf599ab8e7d62d634d2bbe58` (branch
`codex/live-paper-l2-observation`), **LIVE market data with internal paper fills**, disarmed, since
2026-09-30 21:37 UTC. Approved by the user: pins CLX6/GCZ6/NGX6/NQZ6/SIZ6, option-root families with
per-day expiries from Saxo timestamps at the exchange clock, delta tolerance 0.03, persistent recording,
and a daily reference-session audit (timers, 07:30 and 17:10 New York). All four option families load; entries wait for
real-time data (subscriptions start 2026-10-01). [Setup record](live-paper-setup-20260930.json). Return to SIM:
delete `/etc/systemd/system/stocker-v1.service.d/50-live-paper.conf`, daemon-reload, restart.

Previous release `2c2a6e96c78b3dd8425556f55be8d8d425563591` (branch
`codex/live-paper-l2-observation`), deployed on **2026-09-30 at about 19:30 UTC**, replacing `82577e0`.
It unblocks GC: gold now warms and records option candidates and is labelled like the other four
markets, entering only once its listed mapping is approved like any market. [Postflight](unblock-gc-deployment-20260930.json):
flat ledger, verified backup `/var/lib/stocker/backups/unblock-gc-20260930`, installed smoke, all loopback
routes 200, anonymous HTTPS 401, GC now `BLOCKED` by `REFERENCE_CONTRACT_SELECTION_REQUIRED` like the others,
runtime unchanged (SIM authenticated, connected, DISABLED/disarmed, no orders). Rollback: symlink to
`/opt/stocker/releases/82577e0…` and restart.

Previous release `82577e0471331aaf1e73b81a9ce961f334957951` (branch
`codex/live-paper-l2-observation`), deployed on **2026-09-30 at 19:08:30 UTC** through the existing
immutable-release process, replacing `bd2d7a0`. Changes: every observed clock records `book_flow` and
`observation` (see [the experiment](LIVE-PAPER-EXPERIMENT.md)); paper admission buys one contract up to a
£1,000 guard (was £50). The [sanitised postflight](live-paper-l2-deployment-20260930.json) records a flat
ledger before the switch, the verified backup `/var/lib/stocker/backups/live-paper-l2-20260930`, the
reservation-check migration rehearsed on a copy and then applied at start (`IN (1000,5000,100000)`),
installed smoke, all loopback routes 200, anonymous HTTPS 401 and an unchanged runtime state (SIM
authenticated, stream connected, DISABLED/disarmed, no orders). Configuration, credentials and tokens
are unchanged. Rollback: restore the symlink to `/opt/stocker/releases/bd2d7a0…` and restart; the
migrated reservation check also accepts every earlier amount.

LIVE data and paper arming remain user actions: see [LIVE-PAPER-EXPERIMENT.md](LIVE-PAPER-EXPERIMENT.md).

Previous release `bd2d7a0d7def351d583e8b85d57b8c236babb427` (branch
`codex/audit-fixes`), deployed on **2026-09-29 at 18:19:57 UTC** through the existing
immutable-release process, replacing `98a772c`. [Audit fixes](SLRNO-AUDIT-20260929.md) describe the
changes; the [sanitised postflight](saxo-audit-fixes-deployment-20260929.json) records a flat ledger
before the switch, the verified backup `/var/lib/stocker/backups/audit-fixes-20260929`, the
`recorder.pre_event_minutes` configuration migration, installed smoke, matching served assets and an
unchanged runtime state (SIM authenticated, stream connected, DISABLED/disarmed, no orders). The first
token renewal after deployment (18:25:49 UTC) re-authorised the open stream without a reconnect.
Rollback: restore the symlink to `/opt/stocker/releases/98a772c…` and restart; the migrated
configuration loads under both releases.

Earlier: release `5aa3a7b` was deployed on 2026-09-28 at 18:05 UTC
([postflight](saxo-option-deployment.json)). A fresh read-only check on **2026-09-29 at 07:51:16 UTC** confirmed the same release,
an active service with zero automatic restarts, authenticated Saxo SIM and a connected stream:
[sanitised runtime snapshot](github-sync-runtime-20260929.json).
Execution remained **DISABLED**, armed **false**, **LIVE ORDERS DISABLED**, with zero reserved/open
trades. The session was `Authenticated / Standard / OrdersOnly`; no session upgrade or order was sent.

| Stage | Current evidence |
|---|---|
| IMPLEMENTED | Saxo-only five-market runtime, bounded option context, costs/reference safety, existing event recorder and compact Markets display |
| OFFLINE_TESTED | 483 Python tests; Ruff lint/format; mypy (117 source files); both browser suites; locked local and installed server-only smoke |
| AUTHENTICATED | **Yes** at the dated runtime check — real SIM grant, verified account and connected stream |
| FEED_VERIFIED | **Partial** — account/reference access and session transport; actual selected futures/options, quotes, Greeks, IV, volume, OI, costs and L2 remain unverified |
| RECORDER_VERIFIED | **Offline only** — measured bounded workloads; no market prehistory; persistent capture permission-gated |
| DEPLOYED | **Yes** — option-context release; service, authenticated routes and served asset hash verified at deployment, release/service rechecked on 2026-09-29 |

| Market | Authenticated discovery candidates at the latest check | Actual field availability |
|---|---:|---|
| CL | 16 | No selected future/option; quotes, Greeks, IV, volume, OI, costs and L2 unverified |
| GC | 6 | Same; frozen strategy remains MONITOR_ONLY |
| NG | 7 | No selected future/option; quotes, Greeks, IV, volume, OI, costs and L2 unverified |
| NQ | 0 for the configured keyword query | Reference discovery needs refinement; this does not establish lack of entitlement |
| SI | 6 | No selected future/option; quotes, Greeks, IV, volume, OI, costs and L2 unverified |

The current subscription count is two (session and FX), with zero option subscriptions.
The option budget is 16 within the existing application budget of 32, with a candidate window of
three. All five markets report `REFERENCE_CONTRACT_SELECTION_REQUIRED` and zero prehistory.
Persistent capture remains off with `RECORDING_PERMISSION_NOT_VERIFIED`.
Provider states are SAXO ACTIVE, IBKR PARKED, FMP/EODHD INACTIVE. Bitcoin is absent from active paths.
Optional analytics are provider-supplied context, not strategy probabilities or new signals.

The option-context upgrade preserved the existing £10 policy, four-trade reservation limit, frozen
entries/exits and monitor-only decisions. Its backup is
`/var/lib/stocker/backups/saxo-option-context-20260928`; the previous application release was
`2ff57756e272b53b900a7c6e1291a97372c62af8`. Roll back code only after checking current obligations,
preserving current credentials, tokens, configuration, ledger and event data. Never rearm or
reactivate parked providers automatically.

The sections below retain the dated migration and authentication history. Earlier test counts,
release pointers and availability observations are historical, not the latest runtime status.

## Migration and authentication history

Deployed to [the authenticated dashboard](https://139.59.178.164) on **2026-09-28 at
11:20:47 UTC / 12:20:47 Europe/London** following the user's deployment instruction.
At the 14:03 UTC milestone, the runtime release was: `2ff57756e272b53b900a7c6e1291a97372c62af8`, branch `codex/saxo-only`.
This includes migration `c74e68a`, safety/recovery fixes `b066386`, book flow `c809f9f`, and
OAuth bootstrap `b1e9f5c` deployed at **12:54:49 UTC**. The first supplied SIM AppSecret is held
in a stocker-owned 0600 file; no credential values appear in this repository. The sole active EUR SIM account was selected from the authenticated account response and verified
on 2026-09-28; selection and verification remain mandatory before subscriptions or broker-paper orders.
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
at **13:43:10 UTC**. A subsequent user grant succeeded and created the private token file;
the runtime became authenticated. Renewal health determines whether another login is needed.
The original sibling checkout and historical releases remain unchanged.

At that milestone the application was deployed and healthy, **Saxo SIM was authenticated**, and the configured SIM
account was verified. Authenticated account/reference reads and the streaming connection work.
The REST decompression fix `4b17a3d` was deployed at **13:49:22 UTC**; account selection was verified
at **13:51:03 UTC**. The `/connect` endpoint fix `9c1fc30` was deployed at **13:55:36 UTC**.
The subsequent heartbeat fix `2ff5775` was deployed at **14:03:23 UTC**, verified at **14:03:36 UTC**.
See the [heartbeat deployment postflight](saxo-stream-heartbeat-20260928.json) and
[authenticated status/stability check](saxo-authenticated-status-20260928.json).

Actual contract selection was still required for CL, GC, NG, NQ and SI; quotes/L2/book flow remain
unavailable and raw prehistory is zero. GC remains monitor-only. There is no Bitcoin card or active
stock scanner. Session state is `Authenticated / Standard / OrdersOnly`; no session upgrade was sent.

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
- `/opt/stocker/current` → `/opt/stocker/releases/5aa3a7be4c1f37245ba171dde08968eb43b54375`.
- Config: `/etc/stocker/v1/saxo.sim.yaml`, mode 0640, root:stocker.
- Data: **SAXO_SIM**; execution: **DISABLED**; armed: **false**; **LIVE ORDERS DISABLED**.
- New ledger: `/var/lib/stocker/v1/saxo-sim-disabled.sqlite3`; old ledgers retained separately.
- Recorder: 32 MiB RAM, 8 MiB/512-item queue, 2 GiB event limit, 2 GiB reserve; persistent capture off.
- Callback: `https://139.59.178.164/oauth/saxo/callback`; access logging disabled, authentication retained.
- Proxy permits OAuth start, non-transmitting preflight, disarm and explicit archive prune;
  paper arming remains excluded from its write allowlist.

Authenticated HTTPS and trusted loopback checks returned 200 for all five pages, health/system/
overview/history/recordings and JS/CSS. Served JS matched the installed release hash at each deployment, including the option-context release.
Anonymous HTTPS returned 401; untrusted loopback and foreign-origin requests returned 403.
An invalid cross-site OAuth callback returned 400. Bitcoin history selection returned 422.
At initial cutover, non-transmitting preflight returned **409 NOT_CONFIGURED**. The subsequent
OAuth setup originally reported login required. The current release is authenticated, but market
quotes/options/history/L2 remain unverified.
External direct-port access timed out; the application listens on `127.0.0.1:8765` only.
All three ledgers passed integrity checks and contain no orders or positions.
The service used approximately 54 MiB at the initial cutover postflight; this is idle evidence, not a feed benchmark.

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

OAuth and account selection are complete. Remaining setup: discover/pin actual futures contracts
and deterministic rollover identities, verify per-market quote/options/history/L2 permissions,
and document recording rights before enabling persistent capture. The current `NQ` keyword search
returned no candidates; this is not evidence that the account cannot access Nasdaq futures.
Refine discovery using verified Saxo reference identifiers before selecting that market. Follow [Saxo setup](SAXO-SETUP.md); do not put secrets in chat or Git.
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

Authentication was pending at the Origin-fix postflight; later authenticated evidence is above.
The earlier browser fixture is not account-specific market-data/L2 evidence.

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
units masked, execution disabled/disarmed and no orders sent. Authentication/data/L2 were
unverified at that diagnostic postflight. Later authentication succeeded; market data/L2 remain
unverified. Callback URLs and secrets must not be shared for diagnostics.

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
disabled/disarmed, and no orders were sent. Authentication, account data and L2 were unverified
at this historical postflight. A later fresh grant succeeded, as recorded above.


## Authenticated REST and stream follow-up

The successful SIM grant exposed two transport defects: an already-decompressed HTTP body retained
its `Content-Encoding` header and was decoded twice, and the stream used the base URL without the
required `/connect` suffix. Authenticated non-ordering probes reproduced the response-decoding
failure and bare-path HTTP 404; the corrected endpoint completed the WebSocket handshake.
[REST postflight](saxo-rest-decode-20260928.json),
[account-selection report](saxo-account-selection-20260928.json), and
[connect postflight](saxo-stream-connect-20260928.json) preserve the separate evidence.

The first connected release then repeatedly reconnected. A regression reproduced its rejection
of the array-shaped heartbeat in Saxo's [current streaming guide](https://www.developer.saxo/openapi/learn/streaming).
Release `2ff5775` accepts both envelope forms. Heartbeats update stream contact without changing
quote receipt or field-change timestamps. Three authenticated checks from **14:04:14 to 14:05:14 UTC**
received new stream messages, reported no error and retained connection count 1 (the initial
connection; zero reconnects). This verifies session-stream health over that interval, not market
quote or L2 availability. The final full suite passed **457 tests in 63.58 seconds**
with seven existing warnings; Ruff lint/format, mypy (116 source files), isolated locked server
smoke and staged installed smoke all passed. Browser assets are unchanged; earlier browser fixtures
remain applicable. This is not a five-market feed or recorder workload benchmark.

Standards review: **0 findings**. Spec review: **0 new findings**. The review also identified an
inherited follow-up: a `SubscriptionPermanentlyDisabled` price heartbeat caused recreation;
Saxo specifies removing that subscription without resetting it. The option-context release `5aa3a7b`
corrected this handling with a focused regression. It had not been observed on the configured
session-only stream; the fix is offline-tested, not a verified entitlement response. No automatic session upgrade or entitlement changes were performed.

| Market | Discovery candidates at the 14:05 UTC milestone | Historical market-data evidence |
|---|---:|---|
| CL | 16 | Contract selection required; quotes, levels, counts, delay and cadence unverified |
| GC | 6 | Same; strategy remains MONITOR_ONLY |
| NG | 7 | Contract selection required; quotes, levels, counts, delay and cadence unverified |
| NQ | 0 for current keyword query | Refine reference discovery; no conclusion about account entitlement |
| SI | 10 | Contract selection required; quotes, levels, counts, delay and cadence unverified |

All five have zero raw prehistory, no active capture and unavailable book-flow features. No fixture
proves account L2 access. Before account-selection restart, authenticated broker reads returned no
orders or positions. Subsequent releases preserve the credentials, token store, disabled/disarmed
configuration and flat ledger. All 13 legacy units remain masked. No orders were transmitted.

Additional root-only backups are under `/var/lib/stocker/backups/`:
`saxo-rest-decode-20260928`, `saxo-account-selection-20260928`, `saxo-stream-connect-20260928`,
and `saxo-stream-heartbeat-20260928`. The heartbeat release's previous code pointer was `9c1fc30`.
Rollback only after checking current obligations, preserving the current account selection,
tokens and ledger. It would reintroduce the heartbeat defect; retain disabled execution and
provider masks. Never restore old credentials or ledger snapshots over newer state.
