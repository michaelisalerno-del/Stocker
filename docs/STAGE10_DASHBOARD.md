# Stage 10 operational dashboard

The safe integrated diagnostic runs Stocker and the dashboard in the same modular-monolith
process, while keeping the HTTP boundary outside the trading loop:

```bash
uv run stocker stage10-run \
  --runs-config configs/runs.example.yaml \
  --ibkr-config configs/ibkr.example.yaml \
  --database .stocker/stage10-diagnostic.sqlite3
```

Open `http://127.0.0.1:8000`. Use a PAPER/TWS or IB Gateway test session for control
verification. The dashboard submits no manual order, but enabled runs retain their normal
algorithmic execution authority; use a PAPER-only configuration for this diagnostic.

## Safe control check

1. Start with a disabled PAPER run in the runs configuration and launch `stage10-run`.
2. Open Runs, select it, and choose **Enable run**. Confirm the returned outcome says it was
   applied and that the authoritative runtime state becomes READY or ACTIVE without restarting
   Stocker.
3. Choose **Disable run**. Future evaluations stop; existing orders and positions are not
   cancelled or flattened.
4. Edit risk or maximum positions. The result is `HOT_APPLY`; only later Stage 7 decisions use
   the new values.
5. In Settings, replace the symbols in a `CUSTOM_...` universe. Active runs using that universe
   are re-prepared from the current causal point; added symbols remain demand-driven and removed
   symbols are excluded only from future entries.
6. In Settings, edit the PAPER host, port, client ID, or expected account and save. PAPER alone
   disconnects, reconnects, verifies its account, and reconciles before becoming ready again.
7. LIVE configuration can be verified without enabling a LIVE run or placing a trade. Moving a
   run to LIVE still requires the single account/risk confirmation shown by the dashboard.

The separate `stage10-dashboard` command remains available as a standalone read/persist mode.
It does not start the trading runtime, and therefore reports that saved controls require an active
runtime rather than claiming they were applied.
