# SLRNO FIRST4 PAPER server

Use `stocker first4-run --config /etc/stocker/v1/first4.yaml --database /var/lib/stocker/v1/first4.sqlite3 --host 127.0.0.1 --port 8765`.

Only US FIRST4 and verified IBKR PAPER account DUP655399 are supported. See [FIRST4](../../docs/FIRST4.md).
# Authenticated proxy boundary

FIRST4 starts Uvicorn with `proxy_headers=False`: dashboard authentication must see the
actual loopback socket peer, not the browser address in Caddy's `X-Forwarded-For`.
The private proxy token, expected host and origin checks remain mandatory.
The deployed Caddy write allowlist must allow `/api/first4/pause`; legacy run/settings
write routes are no longer part of the application.
