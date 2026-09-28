# Saxo setup

Start with [the sanitised config](../configs/futures.paper.yaml). No one-day token, IB Gateway,
FMP token or EODHD token is required. Missing credentials leave an honest authentication-required UI.
Do not paste secrets into chat, Git, browser payloads, screenshots or the recording archive.

1. Register/approve a server-side Saxo OpenAPI app in the intended environment. Register exactly
   the configured callback URL: local `http://127.0.0.1:8765/oauth/saxo/callback` or your existing
   authenticated HTTPS host plus that path. Remote HTTP, userinfo, alternate callback paths,
   query strings and fragments are rejected. Any Saxo legal/permission/subscription changes are
   user actions; the application never accepts or purchases them.
2. On the runtime host, create a private directory owned by the existing service user and a
   **0600**, non-symlink JSON credentials file. Use a secure editor/secret provisioning process:

   ```json
   {"environment":"SAXO_SIM","client_id":"REPLACE_LOCALLY","client_secret":"REPLACE_LOCALLY","account_key":"REPLACE_WITH_SIM_ACCOUNT_KEY"}
   ```

   `account_key` may be omitted for the initial OAuth login. After authentication, obtain the
   account list using the same environment's read-only `GET /port/v1/accounts/me` and explicitly
   select the intended AccountKey in the protected file. Until selected and verified, the service
   reports `ACCOUNT_SELECTION_REQUIRED`, starts no data subscriptions, and permits no broker orders.
   Never use an AppSecret as an AccountKey. When Saxo lists two active AppSecrets, configure one;
   the second is not an account identifier. Restart disarmed after updating the selected account.

   For LIVE data use an entirely separate file with `environment: SAXO_LIVE`, LIVE app credentials
   and LIVE account key. Set only its path in `saxo.credentials_file`. The application verifies the
   returned account in that environment; UICs/account identifiers are never copied between them.
3. Start the existing `stocker futures-run` launcher. Open System → Connect Saxo securely. This
   starts authorization-code OAuth with expiring single-use state bound to an HttpOnly browser
   cookie. The callback validates state before exchanging the code server-side. The refresh token
   rotates atomically in `<ledger-directory>/<ENV>/oauth-tokens.json` with owner-only permissions.
   An expired/revoked refresh grant produces a reconnect-required state. Never manually paste a
   daily token. Do not run multiple OAuth owners for one environment/root.
4. Read System's actual discovery candidates and capabilities. Pin one verified standard
   ContractFutures UIC per family in `contracts`, including environment, exact symbol, exchange,
   contract month and selection approval. Unknown/ambiguous IDs remain blocked. This initially
   requires an OAuth connection without pinned contracts; then edit local config and restart
   **while disarmed and after accounting for any existing obligations**. No root-only recording keys.
5. Request/verify account market-data permission in Saxo yourself if reported unavailable. An
   OrdersOnly session may provide delayed prices. Upgrading to FullTradingAndChat can downgrade
   another Saxo application. SLRNO never automatically upgrades or fights another session.
   Verify L1 delay, actual granted cadence, real received L2 fields and option quotes per market.
   A MarketDepth schema is not proof of your entitlement. Missing depth keeps L1 monitoring alive.
6. Obtain and retain the applicable Saxo/exchange permission for your intended raw recording,
   research retention and exports. Only then set `persistent_capture: true` and put the actual
   permission record/citation in `recording_permission_evidence`. Merely entering text does not
   establish legal rights. Until verified, the RAM window runs but persistent event/bar storage is off.
7. Execution remains DISABLED/disarmed for this task. Future explicitly authorised paper use may
   select INTERNAL_PAPER with SIM or LIVE data, or SAXO_SIM with verified SIM data/account only.
   Use a separate ledger for every environment/mode; never reopen an old IBKR/FIRST4 ledger as Saxo.
   Resolve the exact [strategy/data blockers](FUTURES-RULEBOOK.md), then run non-transmitting
   preflight. The API `/api/paper/preflight` performs reads/reconciliation only; it sends no order
   precheck, test order or arm request. A later explicit arm requires the exact acknowledgement
   `ENABLE PAPER ONLY` and successful preflight less than 60 seconds old. Every restart disarms.
   Connecting a feed or resuming paused entries never arms execution.

Paper fills also require a currently open option session, verified instrument trading permission,
fresh tradable bid/ask prices and independently fresh available size on the side being filled.
An indicative quote may be displayed but cannot generate an internally simulated fill.

For a stopped service/isolated preflight owner, use:

```sh
uv run --no-sync python scripts/saxo_preflight.py --config /etc/stocker/v1/saxo.sim.yaml --database /var/lib/stocker/v1/saxo-sim-disabled.sqlite3 --seconds 60 --output /var/lib/stocker/v1/saxo-preflight.json
```

Do not stop a manager with obligations just to run the script. Use the running authenticated
`POST /api/paper/preflight` instead. The script runs the data service only, never the manager or
strategy decision loop, and refuses another owner via the same environment lock.

The LIVE REST client is default-deny by method and path. Permitted POST/DELETE operations are
only known market-data/session-event/chart subscription lifecycle operations. Required reference,
portfolio and session reads are enumerated. Order placement/precheck/change/cancel, exercise,
cash transfers, account mutations and automatic capability upgrades are blocked in LIVE.
OAuth uses only the fixed environment auth endpoints. No external fallback provider exists.

Dashboard reverse-proxy authentication must be retained. Allow the OAuth start POST and callback
GET through the authenticated proxy, preserving Host/Origin. Disable/redact access-log query strings
for `/oauth/saxo/callback`; never log callback codes, Authorization headers or cookies. Uvicorn access
logging is disabled. The runtime log rotates at 2 MiB × 4 files and records exception classes only.
HTTP/WebSocket debug logging must remain off.

Token issuance/renewal accepts HTTP 200 or 201. HTTP 201 was observed from the configured Saxo SIM
token endpoint on 2026-09-28. Both statuses still require complete tokens, valid lifetimes and atomic
private-file storage before authentication succeeds; a status alone never establishes authentication.

OAuth exchange failures expose only a safe category in the callback and System's `oauth_problem`:

- `OAUTH_INVALID_CLIENT`: Saxo explicitly rejected the application credentials.
- `OAUTH_INVALID_GRANT`: start a fresh browser login; the code/refresh grant was rejected.
- `OAUTH_TOKEN_HTTP_401` (or another status): the token endpoint rejected the request without
  a recognised OAuth error. This alone does not identify which credential or grant was wrong.
- `OAUTH_TOKEN_NETWORK_ERROR`: check server DNS/TLS/connectivity to the configured auth host.
- `OAUTH_TOKEN_RESPONSE_INVALID`, `OAUTH_TOKEN_RESPONSE_INCOMPLETE`, `OAUTH_LIFETIME_INVALID`:
  the response cannot safely supply usable tokens.
- `OAUTH_TOKEN_STORAGE_FAILED`: restore service-user write access/disk space for the private token
  directory before another login. Successful reconnect clears the prior diagnostic.

Do not paste callback URLs/codes, token responses or AppSecrets into reports. The application
never renders arbitrary provider descriptions or exception strings. Diagnostic requests using an
invalid grant are not proof that a real browser login or account authentication works.

Verified official documentation (2026-09-28):
[environments](https://www.developer.saxo/openapi/learn/environments),
[authorization code](https://www.developer.saxo/openapi/learn/oauth-authorization-code-grant),
[planned API changes](https://www.developer.saxo/openapi/releasenotes/planned-changes),
[session capabilities](https://www.developer.saxo/openapi/learn/session-capabilities).
`/root/v2/user` replaces the retired `/root/v1/user`.
SIM uses `gateway.saxobank.com/sim/openapi`, `sim.logonvalidation.net`, and
`wss://sim-streaming.saxobank.com/sim/oapi/streaming/ws`; LIVE uses
`gateway.saxobank.com/openapi`, `live.logonvalidation.net`, and
`wss://live-streaming.saxobank.com/oapi/streaming/ws`.
Old streamingws sample hosts are not used. A token renewal reconnects/authorizes a fresh socket
and obtains new snapshots, explicitly recording a continuity gap instead of claiming seamless replay.
