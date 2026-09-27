# SLRNO futures PAPER server

Use `stocker futures-run --config /etc/stocker/v1/futures.paper.yaml --database /var/lib/stocker/v1/futures.sqlite3 --host 127.0.0.1 --port 8765`.

See [deployment](../../docs/DEPLOYMENT.md) and [rulebook](../../docs/FUTURES-RULEBOOK.md).
The authenticated reverse proxy remains required. Uvicorn uses `proxy_headers=False` so
security receives the actual loopback peer. The write allowlist contains only
`/api/entries/pause` and `/api/entries/resume`; neither route can arm configuration.
