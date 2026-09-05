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

## Safe Universes diagnostic

1. Use a PAPER-only broker configuration and launch `stage10-run` before the selected market's
   Activity Shortlist screen time. No manual or LIVE order is part of this check.
2. Open **Universes** and select **US / NASDAQ**, **Mid Cap**, and **Session HARD**. Confirm the
   read-only metadata shows USD, XNYS, America/New_York, regular session 09:30–16:00, Activity
   Shortlist V1, a 15-active-minute screen time, and a watch limit of 50.
3. Choose **Add to PAPER**. Confirm the run appears only under PAPER as
   `NASDAQ · HARD-HV · MID`, with distinct market/cap/strategy/screen lineage. The LIVE action is
   permitted only because the exact PAPER counterpart now exists; do not use it in this diagnostic.
4. At 09:45 America/New_York, inspect the persisted screen endpoint and Run Detail. Confirm the
   snapshot reports the actual supported subset of `TOP_TRADE_RATE`, `TOP_VOLUME_RATE`, and
   `HOT_BY_VOLUME`, has no more than 50 selected candidates, and exposes component ranks, hit count,
   aggregate screen score, and final shortlist rank. Disconnect/reconnect and confirm the same
   snapshot is loaded without another scan.
5. Confirm Run Detail shows the extended funnel, screen timestamp/state, watch size, checkpoints,
   native-currency realised/unrealised fields, and Today/5 Sessions/20 Sessions/All performance.
6. Choose **Remove from PAPER**. Confirm future evaluation is disabled while the run ID, fills,
   P&L history, broker orders, and existing exposure remain. Re-add the exact combination and
   confirm the same lineage is enabled.
7. If the connected account permits it, repeat scanner-parameter discovery and contract
   qualification for one non-US catalogue market in read-only mode. Missing entitlements,
   unsupported scanner location/components, or unavailable cap filters must be reported explicitly;
   do not purchase subscriptions and do not submit an order.
8. In Settings, edit the PAPER host, port, client ID, or expected account and save. PAPER alone
   disconnects, reconnects, verifies its account, and reconciles before becoming ready again.

Settings no longer exposes normal CUSTOM-universe creation/editing. The backend Stage 3 CUSTOM
contract remains available for configuration, CLI, and tests.

The separate `stage10-dashboard` command remains available as a standalone read/persist mode.
It does not start the trading runtime, and therefore reports that saved controls require an active
runtime rather than claiming they were applied.
