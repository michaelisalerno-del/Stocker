# Controlled data-only cutover and rollback

The initial Saxo cutover took place on 2026-09-28. The current option-context application release is
`5aa3a7b`, authenticated to SIM and still disarmed; see the dated
[current verified deployment](CURRENT-DEPLOYMENT.md). For subsequent releases, use the existing
immutable `/opt/stocker/releases/<commit>`, `/opt/stocker/current`, locked uv, systemd and authenticated
Caddy process. Do not infer today's running state from old reports. No live or SIM test order or
unattended arming is authorised. This guide authorises no financial account/subscription action.

The numbered procedure below describes the initial migration from the parked IBKR runtime.
A routine Saxo code upgrade preserves the existing Saxo configuration, credentials, token store and
ledger; do not recreate them or reconnect retired providers. Inspect current obligations, back up
state, stage the committed release, run installed smoke, switch the code symlink and verify the
runtime using the same existing release process.

1. With the user's existing authorised SSH identity, inspect the current release symlink, process
   arguments, service users, unit paths and restart relationships. Inventory `systemctl list-units
   --all`, `systemctl list-unit-files`, `systemctl list-timers --all`, `systemctl list-dependencies
   --reverse <app-unit>`, application-user/root crontabs and applicable supervisor/container/launch
   definitions. Examine relevant ExecStart/Requires/Wants/Restart/trigger definitions locally without
   copying Environment/secret values to reports. Inspect actual RAM/disk and lower quotas if needed.
   Candidate names in old evidence include `stocker-ibgateway.service` and
   `stocker-frozen-scanner-observation-20260923.{service,timer}`; verify them, don't assume completeness.
2. Pause new entries on the current application while leaving its manager connected. From the
   **currently installed old release**, run its existing `scripts/futures_preflight.py` read-only
   using the current protected config and authorised Gateway connection. Inspect all accounts,
   outstanding orders/partial fills, positions, exercise/delivery obligations and active ledger
   reservations, including removed markets and old stock execution. If the old helper covers only
   one account, inspect any additional authorised account state before stopping its manager.
   The helper blocks order placement/cancellation; use its existing old environment, not the new
   Saxo installation. A historical flat snapshot or an empty dashboard is insufficient.
   If obligations are present, ambiguous or inaccessible, **defer stopping that manager and Gateway**;
   retain their position management and report the cutover block. Never liquidate or globally cancel.
3. Back up current unit files, timer/cron/supervisor definitions, proxy config, protected app configs,
   OAuth/legacy credentials and databases using the existing SQLite backup API/process. Preserve
   ownership/mode and record SHA-256 manifests. Keep all original ledgers/data with their provenance.
   Do not replace a live ledger with an old backup after new activity.
4. Stage this committed release using the established archive/copy process, then
   `uv sync --locked --no-default-groups --group server`. Run `scripts/server_smoke.py --installed`
   in its locked environment. Keep the existing service user, hardening and loopback port. Stage
   `/etc/stocker/v1/saxo.sim.yaml` from the sanitised example, DISABLED/disarmed, and a new
   `/var/lib/stocker/v1/saxo-sim-disabled.sqlite3`. Never repurpose the old futures/FIRST4 database.
   Configure OAuth and confirm secure callback logging as in [SAXO-SETUP.md](SAXO-SETUP.md).
5. Once fresh obligations are verified flat/accounted for, stop and **disable plus mask** every
   verified app-owned IBKR scanner, observer, collector/download service **and its timers/activators**.
   Remove/disable their exact cron/supervisor restart entries using the captured inventory, leaving
   a recoverable backup. Include FMP/EODHD jobs and FIRST4 stock runtime paths. Check reverse triggers
   again; a stopped service with an active timer is not parked. Do not broadly stop unrelated apps.
   Gateway can be parked only after confirming it is not managing an outstanding or unrelated
   authorised obligation. Retain credentials/accounts/subscriptions; no cancellation or financial action.
6. Change only the verified app unit's ExecStart to:

   ```text
   /opt/stocker/current/.venv/bin/stocker futures-run --config /etc/stocker/v1/saxo.sim.yaml --database /var/lib/stocker/v1/saxo-sim-disabled.sqlite3 --host 127.0.0.1 --port 8765
   ```

   Remove its Gateway/TWS Requires/Wants/readiness/reconnect/startup hooks. Keep unrelated security
   settings. Validate the unit and Caddy config; permit authenticated `/oauth/saxo/start` POST,
   `/oauth/saxo/callback` GET and the required read routes. Existing pause/resume POSTs can remain.
   Paper arm need not be enabled at the proxy for this data-only cutover. Switch the symlink,
   `systemctl daemon-reload`, start only the new app and reload the validated proxy.
7. Authenticate `/api/health`, `/api/system`, `/api/overview`, `/api/history`, all five pages and
   assets. Confirm five cards, LIVE ORDERS DISABLED, DISABLED/disarmed, no startup requirement for
   other credentials, no old provider processes/requests or timers able to reactivate. Open/close
   multiple browser tabs and confirm unchanged server subscription count/continuous buffers.
   Run the non-transmitting API preflight, collect per-market actual capabilities and 15-minute
   coverage. Enable persistent capture only after recording permission is documented. Verify a
   fixture trigger offline; do not invent a live strategy event or place any broker order to test.
8. Save a timestamped sanitised release/capability/resource report. Authentication, L1, L2, recorder
   verification and deployment are distinct states. Keep unavailable markets visibly blocked.

Rollback: pause/disarm new entries, reconcile all new internal/SIM obligations and retain their
manager if any remain. Back up the new ledger and event files. Restore the previous symlink and
unit/proxy definitions only through the same controlled process. Do not copy a Saxo ledger over
IBKR history, rearm either runtime automatically, or unmask old schedules wholesale. Reactivating
IBKR/FIRST4/other providers is a separate explicit operator decision; default rollback is a stopped,
recoverable deployment while required obligations remain managed. No automatic provider fallback.

A reviewed current inventory is required to make stop/mask commands concrete. The 2026-09-28
deployment report records the 13 parked units and protected original definitions. Reinspect before
future changes; repository changes alone cannot guarantee a server supervisor remains parked.
