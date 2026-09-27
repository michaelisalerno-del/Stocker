# Futures PAPER deployment — approval required for cutover and arming

The futures replacement is deployed, unarmed. See [current deployment evidence](CURRENT-DEPLOYMENT.md).
The procedure below applies to future approved cutovers; installation never authorises arming.
The existing release-directory, locked uv, systemd and authenticated reverse-proxy workflow remains.
Do not alter the independent scanner collector service, timer or observations.

1. Review code/tests and product mapping gaps in FUTURES-RULEBOOK.md. Install a new immutable
   release with `uv sync --locked --no-default-groups --group server`. Run `scripts/server_smoke.py
   --installed`. Do not move `/opt/stocker/current` or restart any service yet.
2. Immediately before an approved cutover, request current broker orders/positions/executions
   read-only, and inspect the old ledger's obligations. Earlier flat reports are insufficient.
   If exposure exists, keep its current manager running and defer cutover until an explicit safe
   resolution. Do not cancel/flatten/adopt unrelated exposure. Never use global cancel.
3. Make a consistent SQLite backup and protected config/unit/proxy backup using the existing
   backup workflow. Preserve historical audit records outside the new runtime. Never overwrite
   a live ledger with an old backup after new activity.
4. On the initial replacement only, stage `/etc/stocker/v1/futures.paper.yaml` from the unarmed
   example and a new empty
   `/var/lib/stocker/v1/futures.sqlite3`, owned by the existing service user. Do not reuse old
   slot/economic tables. Subsequent releases preserve the existing futures ledger and configuration;
   never reset them during an upgrade. Preserve existing authentication environment. The new owner is IB client 83;
   only one futures process may own that ledger. The service must remain loopback-only.
5. After cutover approval, change the service ExecStart to:
   `/opt/stocker/current/.venv/bin/stocker futures-run --config /etc/stocker/v1/futures.paper.yaml
   --database /var/lib/stocker/v1/futures.sqlite3 --host 127.0.0.1 --port 8765`.
   Update the proxy POST allowlist to `/api/entries/pause` and `/api/entries/resume`, remove old
   trading write routes, validate the proxy configuration, switch the release and restart only
   the app through the normal coordinated process. Never restart Gateway or the scanner collector.
6. Verify authenticated `/api/health`, `/api/system`, `/api/overview`, history and assets; six
   permanent cards, actual account identity, reconciliation, fresh calendars and per-market history.
   “Installed”, “connected”, “monitoring” and “armed” are distinct. Do not place test orders.
7. Verify the market-data section described in [MARKET-DATA.md](MARKET-DATA.md): actual account
   lines and data entitlements, source/timestamp, external-client headroom, L1 and one-minute bars.
   The 100 account lines / 60 app lines / 3 depth books are planning ceilings, not verified rights.
   External consumption is unknown unless separately observed. Depth routing and permission can
   be checked non-transmitting; leave optional recording disabled until verified. Never purchase
   data, stop an independent collector or use a second client to evade shared limits.
8. Arming is a separate user-reviewed operation. Resolve each product/expiry/tolerance approval,
   verify remaining account market-data capacity and record its source in configuration,
   verify actual contract metadata, fresh quotes, budget and cutoff support through read-only
   preflight. Unresolved markets stay blocked. Configure only approved mappings. `armed: true`
   must never be introduced by install scripts. Resume in the UI merely removes pause and does
   not arm, approve mappings or override reconciliation. LIVE has no setting or route.

The earlier read-only broker snapshot is saved in CURRENT-DEPLOYMENT.md. Repeat it at cutover.
Run `uv run --no-sync python scripts/futures_preflight.py --config configs/futures.paper.yaml`
only on the server hosting the verified loopback PAPER endpoint. It never calls order submission.

Resource use: 13 conservatively counted monitoring lines (six L1, six bar streams, one FX),
16 with three optional books; up to 39 with four options, four retained underlyings and fifteen
temporary quotes (40 during a serial admission quote handoff). These fit the default 60 ceiling,
but reduced available capacity can block entries.
The one wire scheduler allows at most 40 outbound messages/second, with ten reserved for urgent
operations. Core state is durable; optional capture windows are bounded separately in
`futures.l2` beside `futures.sqlite3`. Include that directory in backups without replacing or deleting
the trading ledger. Storage saturation pauses optional recording instead of pruning audit history.
Completed OHLCV remains retained. No tick-by-tick, stock discovery or full-chain quote streaming
is enabled. Read [MARKET-DATA.md](MARKET-DATA.md) before changing collection settings.
