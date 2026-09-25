# FIRST4 order-flow observation V1

Status: **IMPLEMENTED_AND_OFFLINE_TESTED**. **CONNECTED_AND_RECORDING_VERIFIED: no**.
No broker connection, subscription purchase, operational config change, arming, deployment or
service restart was performed.
Source: `main`, `95685c0bdc22a2cbd7f4f1efd8de25cf0b6f9d50`.
Branch: `codex/first4-order-flow-v1`.

## Boundary and ownership

`order_authoritative=false`, `may_submit_orders=false` are validated literals. The checked-in
configuration is disabled. Observation health is absent from all entry and management guards.
The scanner, chronological permanent slots, anchor time, option selection, order requests,
account/reconciliation safeguards and scheduled exit are unchanged.

Runtime schedules entry work first. The separate observer reads only four allocated rows and
retains failed/unfilled entries. Its optional task never reconnects or resets the Gateway.
An existing Last subscription has two explicit owners: entry and observer. The final owner
cancels the wire request; entry still owns its callback/error future. The original broker-time
adapter stays in place. BidAsk/L1 use independent request tickers and wrapper-level taps; they
never enter the entry ticker's trade list. Raw events are copied in receive order, before packet
aggregation. An observer error listener does not consume or clear the existing broker listener.
In particular competing-session 10197 and shared disconnect/farm errors keep existing safety effects.

The actual calendar-derived `close_at` ends collection, including early closes; option exits do
not. Restart uses persisted allocated stocks, retains historical summaries, and starts a new UUID
capture segment. Cumulative values restart per segment. No pre-selection prints or backfill.
The dashboard links by session/conId/slot to the existing anchor, selected option contracts,
quotes, actual PAPER fills, P&L evidence and scheduled exit. No synthetic option values are added.

## Sources and client verification

Verified on 2026-09-25 against installed/locked `ib_async==2.1.0`; no upgrade:

- [IBKR tick-by-tick introduction](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-live/tick-by-tick-data/introduction): user entitlement is 5% of market-data lines; same-instrument requests require 15 seconds. Last excludes additional trade types present in AllLast; live option tick-by-tick is unavailable.
- [Request contract](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-live/tick-by-tick-data/request-tick-by-tick-data): `numberOfTicks=0` requests no historical ticks; BidAsk uses `ignoreSize=False` so size changes are retained.
- [Receive contract](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-live/tick-by-tick-data/receive-tick-by-tick-data): retain request, integer broker time, trade/quote attributes, exchange and opaque conditions. Last callback type 1 is the sole eligible trade source.
- [Live-data limitations](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-live/live-data-limitations): verify L1 subscriptions and shares-mode settings. Receipt precision is not broker timestamp precision.

Pinned client inspection covered `IB.reqTickByTickData`, cancellation, `Wrapper.startTicker`,
`endTicker`, trade/BidAsk/price/size callbacks and packet processing. Its generic ticker reuse
and receipt-time trade substitution are why ownership and capture occur at these explicit seams.

## Capacity and modes

The existing execution path uses up to four Last streams, plus temporary option/combo L1 requests.
The observer conservatively charges each stock one Last stream even when shared, plus one BidAsk
in preferred mode. It reserves at least four TBT slots and sixteen L1 lines for execution/safety.
`available_*` means capacity remaining for this app after other API clients and TWS usage; another
client is not additional entitlement. Defaults are zero available capacity.

- `TBT_TRADES_TBT_QUOTES`: capacity `min(max_stocks, floor((available_tbt-reserved_tbt)/2))`.
- `TBT_TRADES_L1_QUOTES`: one TBT and one spare L1 line per stock; both budgets constrain capacity.
- `UNAVAILABLE`: no active/retained usable capture. This is never substituted with ordinary last updates.

Slots 1..capacity are chosen deterministically. No outcome-based replacement or rotation. A shared
Last can attach immediately; a new observer request observes 15-second per-conId spacing across
stream types and reconnects, plus a 15-second initial-process quarantine when no Last exists.
Quote pacing waits are partial coverage. Request rejection/entitlement failures are terminal for
that allocation in the process; no automatic mode fallback, subscription purchase or permanent-error retry.
A later separately authorised restart must not be mistaken for continuous coverage.

## Classification and clocks

Every eligible print is BUY_EST only at the preceding valid ask, SELL_EST only at the preceding
valid bid. Absolute equality tolerance is **1e-9 currency units**, with zero relative tolerance.
This is numerical precision, not a spread/trading threshold. The engineering quote-age default
is **1,000 ms**; it is configurable, not validated as optimal. Both L1 price-side ages must pass;
size-only updates never refresh a price. L1 must report live data type 1. It remains a lower-quality
source without exchange quote time or tick-by-tick matching equivalence.

Missing/stale quotes, locked/crossed/nonpositive/nonfinite markets, future quotes, unsupported quote
attributes, inside/outside-spread and ambiguous equality are UNKNOWN. Later receive-sequence quotes
never classify earlier trades, even within one socket packet. Equal broker seconds are flagged as
coarse timestamp ambiguity, without inventing sub-second exchange times.

Raw trade attributes/conditions are preserved. Nonempty opaque conditions or `pastLimit`/`unreported`
are conservatively excluded; no condition-code meaning is guessed. Unexpected non-Last callback types,
invalid prices/sizes and prints demonstrably older than quote-age plus one second of timestamp
quantisation are also excluded. The age test assumes the local UTC clock is reasonably synchronised;
it does not infer an exchange condition. Excluded volume/reasons remain separate from eligible UNKNOWN.
Invalid-size raw records have unknowable volume and contribute zero known volume, not an invented size.
Identical prints survive. Only capture-ID/sequence prevents duplicate processing. Reconnect duplication
cannot be resolved from IBKR's fields and is explicitly uncertain.

Minute bars use local UTC **receipt time**, never broker time, so late prints do not revise earlier
information availability. Rolling 1/5-minute views sum the last 1/5 completed receipt minutes and
advance at minute boundaries. Missing minutes are absent, not manufactured zero volume. A minute
containing quote observations but no prints has zero *observed* trades; stale/unavailable intervals
and partial windows are flagged. Cumulative delta is estimated share-volume delta, not an option Greek.

`eligible = buy_est + sell_est + unknown`; `volume_delta = buy_est - sell_est`.
Classified coverage is `(buy_est+sell_est)/eligible`; buy share is `buy_est/(buy_est+sell_est)`;
delta fraction is `volume_delta/(buy_est+sell_est)`. Zero denominators yield null.
Displayed quote size is liquidity, never executed volume. PAPER fill side does not establish market
aggressor direction. None of these observations establishes absorption, exhaustion or predictive value.

## Persistence, bounds, and failures

One observer-owned thread handles a queue of **8,192 events**, batches up to **512**, flushes within
**250 ms** when idle enough to keep up, and fsyncs JSONL before committing the separate SQLite index.
No callback writes files, calculates charts or does network work. Each event contains capture/session,
conId, request, generation, monotonic sequence/clock, UTC receipt, actual broker time/precision,
source/mode and available fields. L1 field events retain individual receipt/update times.
Each tape header records classification version, configuration, source commit, allocation and anchor IDs.

Defaults: **2 GB** observer-directory budget; **1 GB** free-disk reserve; **16 KB** maximum encoded raw
record; **4** active reducers; **400** minute bins per segment; **16** segments per stock/session (max 32);
**63** nonterminal gap markers plus a terminal end marker per segment; **10,000** files at startup. Limits stop capture explicitly and never
prune evidence. A per-stock gap/segment limit stops that stock, not the shared writer or other stocks.
Open stale gaps continue marking affected minutes until an explicit resume; coverage duration excludes
recorded gaps and stops at the last evidence on an interrupted restart. Summary queries use indexed session/conId lookup, at most 32 captures and 400 minute
rows, off the event loop. Raw directory inspection occurs only at writer startup/offline replay.

Queue overflow stops collection, preserving queued evidence where writable and reporting dropped
counts. Writer failure exposes uncommitted-event uncertainty; saved totals never advance to uncommitted
raw data. A crash can leave fsynced raw ahead of its index: replay discovers raw headers independently
and reports missing/stale summaries. Corrupt/partial JSON tails fail loudly. Storage health is isolated
from the execution DB; shared broker failures remain visible to the existing safety logic.
No recorder can protect free space against unrelated processes consuming the same filesystem.

Shutdown cancels observer-owned subscriptions, drains queued evidence on its worker thread and uses a
bounded join. A stalled writer cannot block order management; its health is unavailable, never current.
The UI keeps existing polling/selection/scroll and updates stable slot nodes. Detail charts update on
the existing explicit detail refresh; their scroll/zoom shell persists. No dashboard route requests data
from the broker. Gaps break chart paths; each segment's cumulative delta starts at its own actual capture.

## Offline use and later PAPER activation

Replay/export (no broker access):

```sh
uv run --no-sync python -m stocker_execution.first4_flow_store \
  --root /path/to/capture-directory --session 2026-09-25 --con-id 12345
```

This verifies each saved classification and committed totals/minute rows, exports raw-derived summaries,
and reports any raw tail after the summary checkpoint. It never edits research evidence.

For a **separately authorised** PAPER activation, copy the existing PAPER configuration and merge only
`order_flow` from `configs/first4-order-flow.paper.example.yaml`. Set the actual available/reserved
budgets, desired mode/max stocks and approved capture path. Do not change `armed`, dated opening checks,
account settings, ports, order rules or LIVE restrictions. The example's eight TBT slots are illustrative.
Validate the copied config offline before use:

```sh
uv run --no-sync python -c 'import sys; from pathlib import Path; from stocker_execution.first4_config import load; c=load(Path(sys.argv[1])); print(c.order_flow.model_dump(mode="json"))' /path/to/reviewed-paper.yaml
```

Use the existing launcher with that reviewed config when separately authorised:

```sh
stocker first4-run --config /path/to/reviewed-paper.yaml \
  --database /path/to/existing-first4.sqlite3 --host 127.0.0.1 --port 8765
```

That command connects to the broker; **it was not run in this task**. Use the normal operational
maintenance procedure rather than restarting while FIRST4 owns open obligations. Rollback sets only
`order_flow.enabled: false` in the reviewed config and uses the same separately authorised maintenance
procedure. Preserve the raw directory/index. The dashboard has no enable/arm control for observation.

Real-broker prerequisites still requiring evidence: current account-wide TBT/L1 entitlement and unused
capacity, current Gateway/API compatibility and shares units, request/pacing acceptance, quote/trade
attributes and arrival behavior, actual disconnect/reconnect behavior, disk throughput/free-space reserve,
and sustained recording through a real regular/early close. No CONNECTED_AND_RECORDING_VERIFIED claim.

## Verification

Fixtures use temporary stores, fake wire callbacks and intercepted browser responses. No test connects
to IBKR. Results on macOS 26.6.2 arm64, Python 3.12.13, pinned ib_async 2.1.0 and Playwright 1.62.1:

| Check | Actual result |
| --- | --- |
| `rtk .venv/bin/pytest` | **629 passed in 59.30 s**, including FIRST4 regression tests and 43 dedicated order-flow tests |
| `rtk .venv/bin/ruff check .` | Passed |
| `rtk .venv/bin/ruff format --check .` | 201 files already formatted |
| `rtk .venv/bin/mypy packages apps` | Passed, 113 source files |
| `rtk git diff --check` | Passed |
| `node tests/dashboard_first4.cjs` | SLRNO browser acceptance passed with mocked endpoints and bundled pinned Playwright |

The Python suite emitted five existing warnings: one Starlette/httpx deprecation and four NumPy empty-slice
warnings in behavioral-state tests. Browser checks retained selection, expanded card, horizontal scroll
(260 to 260), detail chart zoom and scroll. Maximum concurrent refreshes: 1; requests while hidden for
60 seconds: 0; unchanged main-content mutations: 0. Existing polling rates remain 12/4/12/2 requests per
minute for overview/opportunities/execution/system. Flow rendering has no broker calls.

For a normal installed Node/Playwright environment, reproduce browser validation without overwriting
existing screenshots using:

```sh
STOCKER_SCREENSHOT_DIR=/tmp/first4-flow-ui rtk npm test
```

This run used the Codex bundled Node executable and `NODE_PATH` for the same pinned Playwright version.
[Fixture screenshot](first4-order-flow-fixture.png) shows synthetic evidence, three separately labelled
axes, unknown volume and a gap. It is not broker recording evidence.

### Measured performance

Reproduce with `rtk .venv/bin/python scripts/first4_order_flow_benchmark.py`.
[All six trial results](first4-order-flow-benchmark.json) retain measurements without rounding.
Each enabled trial sends 4,000 trades and 4,000 quotes across four stocks in 1,000 packets, concurrently
with four real FIRST4 entry coroutines using a fake broker, plus a cooperative event-loop yield probe.
The disabled comparison sends the same trades without optional quote callbacks or recording.

| Measurement (three trials) | Observer disabled | Observer enabled |
| --- | --- | --- |
| Median callback duration | 1.625–1.708 μs | 9.166–9.375 μs |
| p99 callback duration | 2.000–2.583 μs | 20.334–29.416 μs |
| Worst callback | 249.542 μs | 5,873.667 μs |
| Median loop yield interval | 0.111–0.116 ms | 0.817–0.846 ms |
| Maximum loop yield interval | 0.742–14.063 ms | 7.800–8.088 ms |
| Complete burst duration | 12.38–25.36 ms | 111.37–127.45 ms |
| Maximum queue depth | 0 | 5,960 of 8,192 |
| Maximum queue latency | N/A | 369.617 ms |
| Dropped events | 0 | 0 |

All trials completed four entry preparations with identical anchor prices. All enabled trials replayed
saved classifications, totals and minute bars exactly; there were no writer errors. The observer has
measurable CPU/queue overhead. Host scheduling produced outliers in both configurations; this cooperative
probe does not establish production position-management latency or real-broker sustained throughput.

### Standards review

Resolved both findings: isolate a stock's request exception from other captures; stop only the affected
stock at its gap limit. A follow-up terminal-reason overwrite was also fixed and regression tested.

### Spec review

Resolved all three findings: give reconnect segments deterministic ordering and never relabel older totals;
flag mid-minute starts as partial; keep ongoing stale gaps visible and avoid growing interrupted coverage
on restart. Both reviews used the source commit above as their fixed baseline.

## Changed files

- Execution: `first4_config.py`, `first4_requests.py`, `first4_runtime.py`; new
  `first4_flow.py`, `first4_flow_wire.py`, `first4_flow_observer.py`, `first4_flow_store.py`
  under `packages/stocker_execution/src/stocker_execution/`.
- Dashboard: `app.py`, `views.py`, `static/dashboard.js`, `static/dashboard.css`, `static/index.html`;
  new `static/order-flow.js` under `packages/stocker_dashboard/src/stocker_dashboard/`.
- Configuration examples: `configs/first4.example.yaml`, `configs/first4-order-flow.paper.example.yaml`.
- Verification: `tests/test_first4_order_flow.py`, `tests/test_first4_operational.py`,
  `tests/dashboard_first4.cjs`, `scripts/first4_order_flow_benchmark.py`.
- Handover: this document, `docs/first4-order-flow-benchmark.json`, `docs/first4-order-flow-fixture.png`.

`first4_broker.py`, the execution store/ledger schema, pinned dependencies and operational configurations
are unchanged. The separate frozen three-scanner experiment was not modified.
