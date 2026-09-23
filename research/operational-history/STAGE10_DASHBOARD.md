# Operational dashboard

Start standalone mode to inspect state and edit configuration without connecting or trading:

```bash
uv run stocker stage10-dashboard \
  --runs-config configs/runs.example.yaml \
  --ibkr-config configs/ibkr.example.yaml \
  --database .stocker/stage8-runtime.sqlite3
```

Open http://127.0.0.1:8000. The main workflow is **Market → Method → Run**.
Select US All, NASDAQ or NYSE, then **Session HARD**. There is no cap selector or
independent activity-screen/exit setting. Account risk and capacity are expandable.

**Start PAPER run** saves the method specification, version/hash and authoritative listing
snapshot. In standalone mode this persists configuration; it does not start execution.
The integrated `stage10-run` command attaches these controls to the runtime and enabled runs
retain their normal algorithmic PAPER authority. The current method does not support LIVE.

Run detail exposes universe/data/search status, qualification/veto/arming/entry counts,
active/completed positions and performance. Expand method specification and provenance for the
full saved configuration. Candidate details include P0, M, score, PRE_MOVE, whipsaw risk,
Q1 eligibility, both triggers, armed time, first-break direction, entry, exits and deadline.

Disabling stops future entries and preserves fills, history and broker protection.
Re-enabling an unchanged current run preserves its identity. Historical methods are readable
but cannot be enabled. Settings retain broker/account controls; they do not define method logic.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the method contract and prospective Q1 reconciliation.

## Security

Every HTTP route (including controls, static assets and event streams) and websocket
passes the same boundary. CORS and account confirmations are not authentication.

Local mode has no dashboard credentials and permits loopback peers with a local
Host only. Bind to 127.0.0.1. Do not expose this through a remote proxy with rewritten
localhost Host: remote operation must select a protected mode. Uvicorn forwarded
identity/client headers are disabled; X-Forwarded-For cannot confer local access.

Protected mode requires `STOCKER_DASHBOARD_ORIGIN` with the exact public HTTPS
origin (no path), plus **one** of:

- `STOCKER_DASHBOARD_PASSWORD`: at least 24 characters; HTTP Basic user `stocker`.
  TLS must terminate at a private reverse proxy and the backend must stay loopback.
- `STOCKER_DASHBOARD_PROXY_TOKEN`: at least 24 unpredictable characters shared only
  by Stocker and an authenticated loopback proxy. Preserve the proxy's existing user
  authentication, and overwrite `X-Stocker-Proxy-Token` upstream on **every** request.
  The backend requires both a loopback peer and the exact token, plus the public Host.
  A user identity header or forged forwarded address is insufficient.

For Caddy, keep its existing `basic_auth` block and configure the upstream:

```caddyfile
reverse_proxy 127.0.0.1:8765 {
    header_up X-Stocker-Proxy-Token {$STOCKER_DASHBOARD_PROXY_TOKEN}
}
```

Load the token into Caddy and Stocker from a root-owned mode-0600 systemd EnvironmentFile.
Generate it on the server, never in a browser URL, static JavaScript or normal logs.
Alternatively, place the upstream directive with a literal generated token in a
root-owned, caddy-group-readable mode-0640 Caddy include; Stocker reads the same token
from its mode-0600 EnvironmentFile. This permits a graceful Caddy configuration reload.
Preserve existing proxy authentication credentials. Validate Caddy config without
printing its expanded secret. Reject cross-site fetch metadata and mismatched Origin
on reads and controls; unauthenticated websocket access is rejected as well.

For migration, prepare credentials and proxy configuration before switching releases.
Test rejection directly at the backend and through the public proxy. Repository
inspection alone does not prove firewall, proxy, TLS or external access configuration.
An invalid dashboard configuration is isolated by the integrated dashboard supervisor;
it must not terminate trading/recovery.

## Control state and freshness

New identities are atomically saved before activation. A write failure cannot leave an
unsaved new run active. A saved change whose activation fails is reported as **saved,
activation failed**; restart reads and retries that saved identity. Repeat start selects
the existing market/method/environment lineage rather than creating a duplicate.
Saved and applied are distinct from preparing, ready, paused and degraded.

“Pause new entries” takes the entry gate immediately, independently of slow scheduler
or command preparation, and saves it. Existing positions, unfinished submissions and
protective broker orders remain managed. Late preparation cannot clear the pause.
If saving a pause fails, the response explicitly says the runtime is paused but
restart will use the prior saved configuration; repair persistence before restarting.

Risk inputs/display use percent, converting once to the existing internal fraction.
Candidate/trade tables, run cards and runtime status refresh independently of editable
forms. Each refreshed panel shows its own receipt time; failed reads retain data marked
STALE. Eight-second read deadlines allow polling to recover. Older page responses
cannot overwrite a newer selection. A 30-second control deadline means **outcome
unknown**; read run/start status before retrying. Mutations are never automatically retried.

Client failures show an appropriate message and reference identifier. Server logs record
the corresponding exception type/location without echoing arbitrary secret-bearing messages.
System includes observed code revision, method version and a non-secret configuration hash.
Unavailable revision metadata is reported as unavailable; release installation should set
`STOCKER_BUILD_REVISION` to the exact verified 40-character commit.

The supported application currently defines no SSE or websocket data route. The shared
ASGI security middleware also rejects unauthorized websocket scopes should a route be
added. Route-inventory regression tests cover every current HTTP handler and static assets.
Source guarantees do not establish today's service bind, proxy configuration or TLS;
verify these on the actual host for each release.
