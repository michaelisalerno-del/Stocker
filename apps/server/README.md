# SLRNO futures PAPER server

The server runs `stocker futures-run`; the exact command, paths, proxy routes and rollout steps
are maintained in [deployment](../../docs/DEPLOYMENT.md), with rules in the
[rulebook](../../docs/FUTURES-RULEBOOK.md).

The authenticated reverse proxy remains required. Uvicorn uses `proxy_headers=False` so the
dashboard's security layer sees the actual loopback peer. Execution starts disabled and
disarmed on every launch; arming is a separate, explicit operator action.
