# Session HARD dual-feed equivalence diagnostic

Research only; PAPER order placement remains disabled. No deployment or real-market
observation is performed by adding this facility. Production never imports these modules.

## Revision and existing recording audit

Inspected current origin/main **4282ee339a6f0c92c56e157edb655c0c50e10b2d** on
2026-09-12. Read-only SSH `readlink -f /opt/stocker/current` identified the identical
release directory. No service, run configuration, account subscription or broker setting
was changed. The generic `stocker` systemd unit reported inactive; that alone is not an
inventory of active deployment units or proof that every run is paused.

`IbkrConnection.prepare_trade_events` requests `reqTickByTickData(contract, "Last", 0,
False)`, deduplicated by conId and connection epoch. It appends `TradeEvent(tick.time,
tick.price, len(events)+1)` from each `ticker.tickByTicks` batch. Sequence is monotonically
increasing per subscription in callback receipt order; there is no price deduplication.
The retained production fields are timestamp, price and sequence only. Size, exchange,
attributes and broker request identity are discarded by that production conversion.

**Important SDK finding:** locked `ib_async==2.1.0` `Wrapper.tickByTickAllLast` uses
`self.lastTime`, the local packet receipt timestamp, when constructing
`TickByTickAllLast.time`. It ignores the callback's broker `time` argument. Thus the
current production `TradeEvent.timestamp` is local packet receipt time, not broker
event time. This task does not fix/change that behavior. The diagnostic captures the
raw broker seconds argument separately and also copies the actual production timestamp.

`Runtime._prepare_upcoming_expected_moves` calls `prepare_trades` during the saved
method's prefetch lead before an upcoming T0, for ready/active owning runs. Late or
failed preparation does not retrospectively reconstruct a prefix. `trade_events`
requires connection, same epoch and subscription-created time <= T0; it returns
received events with source timestamp >= T0 in original order. It has no upper bound;
the method enforces `[T0,T0+5 minutes)`. `trade_stream_status` distinguishes
NOT_SUBSCRIBED, LATE_SUBSCRIPTION, WAITING_FOR_VALID_PRINT, VALID_CAUSAL_STREAM and
broker rejection. A quiet stream is not proved ready from a request alone.

The raw production tape is held in memory, discarded on release/disconnect. RuntimeStore
persists signal/arming/first-break/cursor state, cohort labels, checkpoints and coverage,
not a reconstructable raw tape. `IbkrSessionDataSource.trades_for` calls only
`broker.trade_events`; runtime passes this map to `SessionHardMethod.observe_trades`.
It does not read this diagnostic's alternative tape. Qualification/Q1/arming precede
first-break logic; earlier breaks expire, and wall-clock expiry handles no-print windows.

Ordinary `_acquire_market_data_stream` keys by conId/security type/exchange/exact generic
tick string/market-data type. Equal requests share a ticker with consumer_count; only
the last release cancels. Every physical ordinary request and TBT stream consumes one
local configured line; TBT also has `max(1, budget//20)` capacity. Capacity is connection
local, not an account-wide entitlement measurement. SDK tick lists are packet batches
cleared on the next packet; `updateEvent` can include quotes, sizes and multiple trades.

## Ordinary source and fixed conversion

Use `reqMktData(..., genericTickList="375", snapshot=False)` and raw
`tickString(request_id, 77, payload)` exclusively. No events are fabricated from
`ticker.last`, quote changes, `updateEvent`, cumulative volume deltas or generic 233.

[IBKR's trade-volume documentation](https://interactivebrokers.github.io/tws-api/tick_types.html#rt_trade_volume)
describes 375/77 as RT Trade Volume, excluding unreportable trades. Generic 233/48
includes those trades and is a weaker semantic match to Last. This is the basis for
the choice, not a claim of equal completeness. IBKR also documents that tick-by-tick
Time & Sales has greater granularity than RTVolume.

Verified against the locally installed locked SDK and its
[wrapper source](https://ib-api-reloaded.github.io/ib_async/_modules/ib_async/wrapper.html):
payload fields are `price;size;Unix milliseconds;total volume;VWAP;single trade flag`.
The SDK updates `last`, `lastSize`, `rtTime`, `rtTradeVolume`, `vwap` and appends
`TickData(local_packet_time,77,price,size)`. Scalar fields retain only the last update
of a batch; the SDK discards the single-trade flag. Temporary wrapper observers retain
every raw payload, including repeated price/size/time combinations, before that loss.
The real SDK decoder looks up wrapper methods at dispatch time; tests exercise it.

The alternative conversion is frozen in `dual_feed_comparison.CRITERIA`: each valid
77 payload becomes one `TradeEvent(broker_millisecond_timestamp,raw_price,local_sequence)`.
Do not round its timestamp to match TBT, substitute receipt time, rescale size, remove
repeats, expand aggregates or tune conversion after outcomes. Missing timestamp/price
prevents conversion; missing size stays null. Source timestamps have **different
semantics** in the two tapes and are explicitly reported. Broker timestamps on both
sides are separately retained for descriptive alignment. There is no exchange trade ID.

Each TapeEvent records conId/symbol, feed, source timestamp, broker timestamp, local
packet receipt time, monotonic callback clock and sequence, price/size, raw provenance,
market-data type, epoch and physical subscription-created time. Market-data type is
the SDK ticker value (the SDK initially defaults it to 1); it is not an independent
entitlement acknowledgment. Successful observation also requires actual valid prints.
Global sequence gives callback observation order; TBT's per-feed order is unchanged.

## Isolation and ownership

`DualFeedRecorder` only accepts a connected, non-executable PAPER adapter. It attaches
to caller-prepared reference streams; closing it removes observers but **never cancels
caller-owned TBT**. The operator owns an otherwise unused connection, prepares at most
five references, then disconnects that connection to release its physical TBT requests.
No changes to production `ibkr.py`, runtime, method, structure D, execution or risk code
are required. Wrapper callbacks are restored on close and reject epoch discontinuity.

Ordinary acquisition uses the existing accounting/release seam and reuses exact 375
subscriptions. Duplicate recorder requests are idempotent. An incompatible existing
ordinary subscription on the same conId is rejected explicitly, not upgraded or
cancelled: ib_async shares a ticker by contract and one cancellation mapping per
subscription type. Starting another generic-tick request could steal another owner's
cancellation mapping. There is no generic framework or subscription rotation.

The default global evidence bound is 250,000 observations. Overflow terminates the
operator and invalidates comparison. Raw callbacks never append to the production
TradeEvent list. Evidence and temporary IBKR history are written only beneath the new
timestamped diagnostic output directory; no general tick database is introduced.

With budget 100, five TBT plus thirty ordinary lines requires **35 local lines** and
fits the separate five-TBT cap. Fake-broker tests establish this arithmetic and clean
release. Other consumers/account-wide IBKR limits/entitlements can still reject it.
This task has **not observed thirty ordinary streams in a real open session**.

## Acceptance criteria fixed before observation

The operator writes acceptance-criteria.json and its SHA-256 before connecting.
Code/specification changes require a new preregistration; no outcome-based retuning.

- Sufficient evidence: same full method interval, recording on both sides by T0,
  retained through T0+5m, valid prints on both, unchanged epoch, live ticker data,
  no dropped/malformed/error evidence, and prospectively produced method context.
- Strict: every sufficient pair EXACT_METHOD_MATCH, no opposite/missed reference
  signal (including late arrival), and identical ordered price prefix through the
  decision, including repeats. Aggregation cannot silently pass ordering.
- Descriptive: always retain counts, price changes/levels, repeats, ties, timestamp
  inversions, broker-second occurrence alignment, unmatched sequence IDs, price
  differences and equal-price aligned receive lag (median/p95/max), plus events
  received after the reference decision. Alignment does not establish trade identity;
  unequal bucket counts are not proof of which individual trades were lost.

Both offline replays restore the identical unconsumed result of the real method's
prospective qualification/Q1 evaluation. `restore_signals`, `observe_trades` and
`expire_waiting_before` are unchanged. Each tape is delivered in its receipt sequence;
events buffered before arming are delivered at the original arming instant. Wall-clock
expiry precedes delivery, so a print received after T0+5 cannot rescue a missed signal.
Initial state already containing a consumed cursor/entry or wrong spec is rejected.

Reports include qualifying break, first crossing, direction, triggering source event,
signal timestamp/minute, local decision receipt time, entry side/reference, no-signal
reason and classification. All seven requested classifications are supported. Incomplete
pairs keep descriptive results but are labelled INSUFFICIENT_DATA. A verdict never hides
those cases or treats non-paired TOP30 observations as equivalence evidence.

## Operator procedure (separate open-session operation)

Use an authenticated dedicated PAPER/data Gateway, distinct client ID, and all trading
runs paused. The operator refuses enabled saved runs, LIVE runs and standard LIVE ports;
account verification and readonly connection remain in the existing adapter. It also
blocks the SDK order and what-if methods and counts attempted invocations. It never
constructs Stage7 execution, starts the trading runtime, scans, changes run state or
places an order. A dedicated-Gateway assertion is required because local counters and
one config file cannot prove account-wide usage or another process's paused state.

The existing current-session RV15-selected population must already be frozen in the
runtime database. Selection/cohort reads use SQLite mode=ro and one transaction. There
is no fallback population, prospective scanner run or enable-disabled override. If a
paused deployment has no current selection, stop; arrange a separately authorized safe
selection preparation first. The first up-to-five conIds form a diagnostic sample only,
never a production ranking, with no replacement after rejection.

```bash
rtk uv run --no-sync python scripts/dual_feed_diagnostic.py \
  --runs-config /path/to/paused-runs.yaml \
  --ibkr-config /path/to/paper-config.yaml \
  --database /path/to/runtime.sqlite3 \
  --run-id EXISTING_RUN_ID \
  --t0 '2026-09-14T14:00:00+00:00' \
  --client-id 293 --ordinary-top30 \
  --confirm-dedicated-paper-gateway \
  --output .stocker/dual-feed
```

T0 must be an actual saved-method checkpoint within the next 30 minutes on that market's
open calendar session. Replace the example accordingly. Current-session selection must
predate collection. Features/history/cohort/model use the existing Session HARD producers;
only the paired sample needs context preparation. Exact context and availability/arming
times are frozen before tape outcomes are inspected. Missing/late preparation remains
insufficient; no model inputs or T0 are invented. A fresh history cache can be slow.

The command records through original T0+5 plus a fixed two-second descriptive grace,
then cancels ordinary leases, detaches observers and disconnects owned TBT. It does not
reconnect. It exports comparison.json, events.csv, comparison.md, criteria and prospective
context files with exact release/dirty state, masked account, session/instruments,
subscription intervals/counts, errors, resource snapshots, per-stock replay and explicit
TOP30 simultaneous-observation status. Actual print receipt is stronger evidence than
`reqMktData` merely returning a ticker, but quiet periods cannot prove continuous delivery.

## Interpretation and handover

No real broker observation was run during implementation. No empirical equivalence or
thirty-symbol entitlement claim is available. The timestamp-semantic difference and
documented feed granularity make this a useful diagnostic, not a presumed replacement.

If strict comparison fails, keep true TBT capacity for the current frozen method. If
the sample agrees, keep production unchanged and design a larger preregistered validation
across sessions/stocks before any separate data-source decision. Never emit
PRODUCTION_REPLACEMENT_VALIDATED. No buying capacity, subscription changes or deployment
is authorized by this facility.

Validation commands/results are recorded in the task handover after checks complete.
