# IBKR resource boundaries

The runtime shares stock qualification and historical bars by physical contract identity.
Activity scanners and historical requests retain their existing concurrency limits.
Current Session HARD computes prior-20-return HV from IBKR final RTH minute closes.
Its ordered TRADES streams capture causal threshold breaks and share the existing market-data
line budget with quotes and shortability. Streams are released after rejection/expiry/entry
when no other run or imminent checkpoint needs them. Capacity failures are per-symbol diagnostics.
The original generic tick 104 path remains a historical diagnostic, not a current model input.

`stocker ibkr-resources` reports connection-local resource counters and saved watchlists.
Its deterministic capacity exercise uses temporary stock HV snapshots, shared history and
scanner requests. No diagnostic submits an order.
