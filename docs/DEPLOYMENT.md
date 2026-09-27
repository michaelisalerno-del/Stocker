# Futures PAPER deployment — approval required for cutover and arming

This change is prepared locally. It has not restarted the server or enabled transmission.
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
4. Stage `/etc/stocker/v1/futures.paper.yaml` from the unarmed example and a new empty
   `/var/lib/stocker/v1/futures.sqlite3`, owned by the existing service user. Do not reuse old
   slot/economic tables. Preserve existing authentication environment. The new owner is IB client 83;
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
7. Arming is a separate user-reviewed operation. Resolve each product/expiry/tolerance approval,
   verify remaining account market-data capacity and record its source in configuration,
   verify actual contract metadata, fresh quotes, budget and cutoff support through read-only
   preflight. Unresolved markets stay blocked. Configure only approved mappings. `armed: true`
   must never be introduced by install scripts. Resume in the UI merely removes pause and does
   not arm, approve mappings or override reconciliation. LIVE has no setting or route.

The current read-only broker snapshot is saved in CURRENT-DEPLOYMENT.md. Repeat it at cutover.
Run `uv run --no-sync python scripts/futures_preflight.py --config configs/futures.paper.yaml`
only on the server hosting the verified loopback PAPER endpoint. It never calls order submission.

Bounded resource use: six futures history streams and L1 + one FX line; one temporary option selection/exit
quote per serialized broker operation and four owned positions. No tick-by-tick observation worker,
stock discovery, daily allocation reset or full-chain quote download exists in the futures runtime.
Completed OHLCV is retained; tick history is not collected. Contract chain metadata is cached by
underlying/date, reference histories by bootstrap contract, and dashboard history is indexed/paged.
