# IBKR resource boundaries

The runtime shares stock qualification and historical bars by physical contract identity.
Activity scanners and historical requests retain their existing concurrency limits.
Session HARD · HV captures generic tick 104 on a temporary stock market-data line before
each checkpoint. The line registry releases the final consumer on success, timeout,
exception and disconnect. It never requests derivative contracts to produce HV.

`stocker ibkr-resources` reports connection-local resource counters and saved watchlists.
Its deterministic capacity exercise uses temporary stock HV snapshots, shared history and
scanner requests. No diagnostic submits an order.
