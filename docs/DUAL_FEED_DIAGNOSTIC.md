# Session HARD dual-feed equivalence diagnostic

Research only; PAPER order placement remains disabled. No deployment or real-market
observation is performed by adding this facility. Production never imports these modules.

## Revision and existing recording audit

Inspected current origin/main **4282ee339a6f0c92c56e157edb655c0c50e10b2d** on
2026-09-12. Read-only SSH `readlink -f /opt/stocker/current` identified the identical
release directory. Later read-only SHA-256 checks matched all four deployed causal files
(ibkr, runtime, session_hard_method and session_hard_structure_d) against this checkout.
No service, run configuration, account subscription or broker setting
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

The corrected conversion is frozen in `dual_feed_comparison.CRITERIA`, version
`DUAL_FEED_V2_LOCAL_RECEIPT`, before any real-market observation:

- REFERENCE_TBT REPLAY: existing production TradeEvents exactly as received.
- ORDINARY REPLAY: one TradeEvent per valid RT Trade Volume tick-77 payload,
  `TradeEvent(local_packet_receipt_timestamp, raw_price, callback_receipt_sequence)`.
  Broker milliseconds remain descriptive metadata only.

No rounding, shifting, bucketing, resampling, deduplication, expansion, interpolation
or later outcome-based adjustment of replay inputs is permitted. Payload validation
is unchanged; missing broker timestamp/price still marks raw evidence invalid and
missing size stays null. There is no exchange trade ID.

The code and reports explicitly distinguish three domains:

| Domain | Meaning |
|---|---|
| METHOD_TIME | Local packet receipt time used by both replays; existing TBT `event_at`, ordinary `received_at` |
| BROKER_EVENT_TIME | Retained `broker_at`; ordinary raw `event_at` also remains broker milliseconds; descriptive only |
| MONOTONIC_RECEIPT_TIME | `received_monotonic_ns`, for descriptive callback latency/order, never method time |

Both causal-window filtering and alternative TradeEvent conversion use METHOD_TIME.
Raw evidence is not rewritten. Broker-second descriptive alignment still uses broker
time and never forces a common trade identity. Receive lag and broker-time lag are
reported separately. Monotonic lag includes the existing difference between ordinary
tickString dispatch and the later TBT packet-update observer; it is not a pure network
latency measurement. Timestamp ties and counts of observations sharing receipt timestamps
describe packet grouping without expanding payloads into inferred trades.

The production TBT timestamp behavior was discovered during the diagnostic audit.
Changing production from local packet receipt time to broker event time would alter
the effective method/data semantics and requires separate validation. This correction
reproduces current production behavior as-is for equivalence testing; it does not endorse
or change it. No production-fix branch or production-file modification is part of this task.

Each TapeEvent records conId/symbol, feed, source timestamp, broker timestamp, local
packet receipt time, monotonic callback clock and sequence, price/size, raw provenance,
market-data type, epoch and physical subscription-created time. Market-data type is
the SDK ticker value (the SDK initially defaults it to 1); it is not an independent
entitlement acknowledgment. Successful observation also requires actual valid prints.
Global sequence gives callback observation order; TBT's per-feed order is unchanged.

## Isolation and ownership

`DualFeedRecorder` only accepts a connected, non-executable PAPER `DualFeedConnection`.
This diagnostic-only subclass inherits every production feed/order method unchanged.
Its one override routes known ordinary-request errors to normal resource counters without
letting the base adapter's conId-wide entitlement rejection cancel a separate TBT owner.
Reference request errors retain the existing invalidation behavior. Ordinary request IDs
remain known for the connection epoch after cancellation to handle late broker errors.
There is no change to the production adapter's error handler.
It attaches
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
V1 used mismatched replay clocks and is superseded before any observation by V2.
The hash changes from
`8a35d0fa4dd1628db0a6b75580aa5a7c4d62860084f212ccd0844cf7a55f5dff` to
`003143a335d883f3855c9a46c84e1a41a3437a7e2f1089707402d850d8d2d041`.
Future operator runs serialize the corrected CRITERIA directly and hash those exact bytes;
the deterministic fake-session test verifies the written file and reported hash.
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
pairs keep descriptive results but are labelled INSUFFICIENT_DATA. With no sufficient
paired replay, the verdict is DUAL_FEED_DIAGNOSTIC_NOT_RUN (collection may have been
attempted; the equivalence comparison could not be completed). A verdict never hides
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
thirty-symbol entitlement claim is available. Broker and receipt clocks remain separately
available for analysis, but replay now compares matching local receipt semantics.
Documented feed granularity still requires observed validation.

If strict comparison fails, keep true TBT capacity for the current frozen method. If
the sample agrees, keep production unchanged and design a larger preregistered validation
across sessions/stocks before any separate data-source decision. Never emit
PRODUCTION_REPLACEMENT_VALIDATED. No buying capacity, subscription changes or deployment
is authorized by this facility.

## Initial implementation handover and validation (before V2 correction)

Implemented on `research/dual-feed-diagnostic-20260912`, based on main/deployed
`4282ee339a6f0c92c56e157edb655c0c50e10b2d`. Implementation commits:
`d8105261ef1292c1c7dd0ed88357cfd1f33861e9` and reviewed isolation fixes
`3843f38581b1fed03536870de5043a80a1cfc17c`. The later handover commit changes documentation only.

Changed files/functions:

- `dual_feed.py`: `DualFeedConnection`, `DualFeedRecorder`, `TapeEvent`,
  `StreamEvidence`, `ordinary_fields`; raw callbacks and bounded ownership-aware recording.
- `dual_feed_comparison.py`: frozen `CRITERIA`, `coverage`, `replay_method`,
  `compare_pair`, `verdict`; offline metrics and unchanged-method replay.
- `dual_feed_operator.py`: `frozen_selection`, `prepare_contexts`, `observe`,
  `validate_operator`, `block_order_methods`, `export_report`; finite PAPER operator.
- `scripts/dual_feed_diagnostic.py`: entry point applies the existing numerical profile.
- `tests/test_dual_feed.py`: 33 deterministic tests using real ib_async decoder/wrapper
  with fake socket requests. README links this operating contract.

Focused command (exit 0, **218 passed**, one pre-existing dependency deprecation warning):

```bash
rtk uv run --no-sync pytest tests/test_dual_feed.py tests/test_method_package.py tests/test_ibkr_resources.py tests/test_session_hard_exit_contract.py tests/test_stage8_session_data.py tests/test_stage8_runtime.py tests/test_stage7_separation.py tests/test_stage7_ibkr.py
```

Final canonical command (exit 0; bundled Node and isolated local npm supplied on PATH):

```bash
rtk proxy env PATH="/Users/michaelsalerno/Documents/Codex/2026-09-12-task-add-a-minimal-research-diagnostic/.stocker/test-tools/node_modules/.bin:/Users/michaelsalerno/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH" bash scripts/check.sh
```

| Canonical check | Final result |
|---|---|
| Ruff format | PASS, 280 files already formatted |
| Ruff lint | PASS |
| mypy packages/apps | PASS, 146 source files |
| Python | PASS, 1,193 passed, 14 skipped, 10 warnings, 121.52 seconds |
| Frontend | PASS, all three existing Playwright suites |
| Isolated server install | PASS, server-only imports, frozen model, offline dashboard/assets |

The first full attempt recorded a timing-sensitive failure in the unchanged
`test_pending_contract_checks_cannot_write_after_acquisition_seals[cancel]` and a local
npm launcher-path failure. `rtk uv run --no-sync pytest tests/test_scanner_acquisition.py`
then passed all 26 tests. No scanner code/test was changed. Correct npm was installed
under ignored `.stocker/test-tools`; the final full run above passed all six checks.
Dependency declarations and lockfiles remain unchanged. The existing Python warnings
concern Starlette test-client deprecation, research empty-slice calculations and KRX
calendar discontinued break metadata; they are not suppressed by this task.

`rtk uv run --no-sync python scripts/dual_feed_diagnostic.py --help` passed without
connecting to a broker. Standards and spec reviews found no remaining issues after
fixes for ordinary entitlement isolation, empty-evidence verdict and missing market context.

Boundary verification (exit 0, no diff):

```bash
rtk git diff --exit-code 4282ee339a6f0c92c56e157edb655c0c50e10b2d -- packages/stocker_execution/src/stocker_execution/ibkr.py packages/stocker_execution/src/stocker_execution/runtime.py packages/stocker_execution/src/stocker_execution/session_hard_method.py packages/stocker_execution/src/stocker_execution/session_hard_structure_d.py packages/stocker_core/src/stocker_core/methods.py packages/stocker_core/src/stocker_core/method_artifacts packages/stocker_core/src/stocker_core/runs.py configs
```

Only the seven listed additive diagnostic/documentation/test files differ from baseline.
Frozen method artifacts, PRE, T0/checkpoints, candidate selection/scanner recipe/TOP30,
thresholds, entry/exit geometry, market/session rules, production feed, execution and risk
code/configuration remain unchanged. No order, what-if, LIVE connection, run activation,
broker subscription purchase, server change, push or deployment occurred.

The complete fake-session test observed 5 TBT + 30 ordinary lines, released all 35,
left its run paused, and recorded zero order-method invocations. This is implementation
evidence only. There are **no paired real-market results**, no observed real TOP30
capacity result, and no empirical Session HARD equivalence classification to report.

Exact next recommendation: run one supervised open-session comparison with this frozen
conversion/criteria, a current-session frozen selection, a dedicated read-only PAPER
Gateway, and every trading run paused. Any server deployment remains a separate action.
Keep the production Last TBT source. A successful small sample would justify designing
larger preregistered multi-session validation, not switching feeds.

DUAL_FEED_DIAGNOSTIC_NOT_RUN

## V2 timestamp correction handover

This narrow correction starts from `ec1c926e3aaffcaf42c1bf1000fbe56c81ce1d31` on the
existing diagnostic branch. Only `dual_feed_comparison.py`, `dual_feed_operator.py`,
`tests/test_dual_feed.py` and this document change. The raw recorder `dual_feed.py`
is unchanged, as are all production, configuration, risk and execution files.

`method_time` selects the unchanged reference timestamp or ordinary local packet receipt
time. `replay_method` and `compare_pair` use it for method-window filtering and replay;
operator coverage counts use the same domain. `coverage` and comparison results retain
broker-time relationships and add explicit method/receipt/monotonic timing, packet
grouping and receipt-order diagnostics. Late observations in the collection grace period
remain visible even though they cannot enter the method's expired window.

Existing timing-disagreement coverage now delays the actual ordinary receipt instead of
only changing its broker clock. New fixtures cover both T0 boundaries, broker times a day
apart, after-expiry receipt, repeated packet order, unchanged raw broker provenance,
independent timing metrics and criteria hashing. These are fake-broker fixtures only.
No real-market data was inspected before this correction, and no broker connection,
market experiment, deployment, configuration or broker-setting change was performed.

Focused command:

```bash
rtk uv run --no-sync pytest tests/test_dual_feed.py tests/test_method_package.py tests/test_session_hard_exit_contract.py
```

Result: **93 passed**, one existing Starlette deprecation warning. Changed-file Ruff and
mypy checks also passed. Final canonical and production-boundary results follow below.
