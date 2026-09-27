# Market data and optional L2 observations

This extends the futures migration at `4eff5fc3b55f703bd78cf1794a25561847413f9b`,
on `codex/futures-paper-replacement`. It is implemented locally, **unarmed and not deployed**.
No broker orders, live entitlement probes, service restarts or subscription purchases were made
for this addition. The independent research collector is untouched.

## Allowances and resource accounting

`configs/futures.paper.yaml` is authoritative. Defaults are planning assumptions, not purchased
entitlements. `allowance_status` is `ASSUMED`, `CONFIGURED` (operator-supplied source), or
`BROKER_VERIFIED` (source plus timezone-aware verification timestamp). No account-wide usage
endpoint is claimed. External usage remains unknown unless explicitly supplied; System displays
that distinction. Assumed allowance prevents new entries; monitoring and management continue.

| Setting | Default and enforcement |
|---|---|
| `total_lines` | 100 assumed account allowance |
| `app_line_cap` | 60 maximum, never a consumption target |
| `external_headroom` | 40 lines reserved outside SLRNO |
| `known_external_lines` | Unknown; effective budget is min(60, total minus max(headroom, known usage)) |
| `depth_slots` | At most 3 futures books; independent of 4 trading slots |
| `temporary_option_quotes` | At most 15 across all consumers |
| `option_batch_size` | At most 5 per batch, sequential expansion before the entry deadline |
| `tick_by_tick_slots` | 5 planning ceiling; tick-by-tick requests are disabled/rejected |
| `outbound_limit` / `outbound_headroom` | 50 assumed ceiling minus 10 = 40 messages/second |
| `urgent_reserve` | 10 of those 40 reserved for reconciliation, order cancellations and exits |
| `queue_size` | 128; optional traffic can occupy at most half, core 3/4, exposure 7/8 |
| `rejection_backoff_seconds` | 300; entitlement/rejection requests are not hammered |
| `cancel_drain_seconds` | 1 after local cancellation wire completion; **not** a broker acknowledgement |

The installed ib_async 2.1.0 FIFO throttle is replaced by one priority wire scheduler, not stacked
with another. Nonurgent traffic is limited to 30 messages/second and optional traffic to five.
Every subscription, cancellation and metadata request uses that socket. Streaming callbacks do
not consume outbound request tokens. Order intents still persist before immediate, finally guarded
transmission. A pacing timeout cannot enqueue an order to be sent after its frozen deadline.

Subscriptions share an actual conId, security type, exchange, currency, feed and parameter key.
Distinct consumers own references; closing UI details or adding browser tabs cannot change broker
subscriptions. REQUESTED, ACTIVE, FAILED, CANCELLING and DISCONNECTED states are explicit.
Cancellation debt counts until the local cancellation has been sent and drained. An uncertain
cancellation retains its resource reservation until disconnect. Cleanup checks generation ownership.

Conservative line accounting charges each independent quote, streamed bar or depth feed once,
even where IBKR might internally share capacity for the same instrument:

| Scenario | L1 futures | Bar streams | FX | Owned options | Retained old underlyings | Temporary FOP quotes | L2 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Six-market monitoring | 6 | 6 | 1 | 0 | 0 | 0 | 0 | 13 |
| Monitoring + optional books | 6 | 6 | 1 | 0 | 0 | 0 | 3 | 16 |
| Four positions, all underlyings rolled, full option pool + L2 | 6 | 6 | 1 | 4 | 4 | 15 | 3 | 39 |

The last row is a tested full-pool count. A serial selected-quote handoff before atomic trade admission
can briefly add one line: **40**, still within 60. Shared identities reduce these counts.
Rollovers cancel old monitoring bars before replacement and retain old futures quotes when owned
positions require them; a roll does not erase an actual FOP identity. Historical snapshots are
separate requests, not additional enduring streams. Connection/account subscriptions are not price
lines; their requests still use the scheduler. An option on a different future requires explicit
approved linkage; current mappings reject mismatched linkage instead of guessing another underlying.

Eight lines are reserved against optional selection/depth use for four options and four retained
underlyings. Existing position data is never evicted for candidates or L2. On reduced capacity,
optional subscriptions go first; if necessary unshared entry-monitoring feeds yield to exposure
data. Affected markets show a data block and retry with backoff. Reusing an existing exposure feed
does not wait for optional teardown. IBKR 309 backs off only its depth request; it cannot block
option preparation. Shared line/message-limit errors reduce optional use and surface the failure.
SLRNO does not stop another client or buy capacity to fix an allowance problem.

## Bars, quotes and recovery

The frozen feed remains IBKR `reqHistoricalData`, **TRADES / 1 min / useRTH=False /
formatDate=2 / keepUpToDate=True**. Each timestamp identifies the interval's beginning. IBKR revises
the mutable current bar; only a later bar plus elapsed minute makes the previous bar eligible.
OHLCV and reported volume are retained unchanged. L1 last-price sampling is never used to build
strategy bars. Repeated unchanged prices, last receipt, last price change, bar completion and
connection health have separate fields.

History has two response slots, at most 16 queued/deduplicated requests, 350ms between starts,
15-second request-local deadlines and a 20-second queue/response envelope. Only 1-minute and
daily TRADES plus exchange SCHEDULE requests are supported. The small-bar/5-second-bar and
tick-by-tick endpoints are unavailable, so their special limits cannot be accidentally bypassed.
Identical completed history requests share a five-minute bounded cache; five-session reference
summaries are persisted. Reconnect reuses recent same-contract bars and requests 30 minutes where
possible. Gap repair is bounded to 30 minutes per market per five minutes, uses only broker bars,
never fills a session gap, and cannot replay a repaired opportunity into an order.

Reconnection reconciles account exposure first, restores its data and FX, then restores the six
markets, then optional L2. Old books and request generations are discarded. Durable reservations,
fills, exit anchors and duplicate protection stay intact. One failed history request cannot consume
both response slots forever; dependency cancellation is reported as a data error, not cancellation
of the position manager. Pending and uncertain orders retain option/future ownership until confirmed
closure. Order/fill evidence is never routed through the optional writer queue.

Selection caches chain metadata by qualified underlying and date. The current frozen method ranks
all listed strikes using its approved frozen RV model; it needs **one selected-contract quote**, not
five unnecessary quote streams. It does not select from only the first five strikes. The shared
batch interface permits at most five actual FOP quotes at once for a bounded selection operation,
and at most fifteen globally. No whole-chain streaming, cheaper-strike substitution or Greek
fallback exists. `selected_delta` explicitly records the frozen model basis; missing quote/model
Greeks are not treated as that delta. Quotes must be real-time, fresh and uncrossed before entry.
Unused candidates are released on completion/failure/deadline; owned feeds survive that release.

## Observation policy `L2_CLOCK_CONTEXT_V1`

L2 defaults **disabled**. Enabling it does not arm execution or change any frozen rule. Direct,
qualified underlying FUT contracts only; no options or SMART depth. The configured exchange must
match contract metadata and a FUT/Deep entry from `reqMktDepthExchanges`. Missing routing or
permissions is an L2-only UNAVAILABLE state.

Priority is: active 120-second signal capture, then a frozen opportunity within 120 seconds,
then fair rotation. Active windows already allocated retain priority; scheduled ties favour the
market least recently captured, ordinary ties the least recently allocated. Final ties use
BTC, CL, GC, NG, NQ, SI. Normal dwell is 60 seconds; higher-priority captures may preempt it.
Open positions do not pin books. No new prediction, approaching-trigger threshold, veto or ranking
model is introduced. NG's existing approved clock veto and GC's base policy remain unchanged.

At most three books, five levels per side. Replacement awaits cancellation wire completion and
the configured drain before requesting another book. IBKR has no depth cancellation acknowledgement;
the drain is local accounting, not evidence of acknowledgement. Late callbacks are ignored by
request generation and ownership. Broker depth limits remain authoritative and backed off.

Both `updateMktDepth` and `updateMktDepthL2` enter the same ordered row reconstructor: insert shifts
rows, update replaces a row, delete shifts remaining rows. Side 0 is ask, side 1 bid. Error 317
clears the book before further rows. Invalid rows, resets, disconnects and preemptions produce gap
markers. Empty/one-sided/partial/reset books are INCOMPLETE; missing levels are never padded.
Displayed books expire after 30 seconds without receipt. Missing browser responses hide the ladder.

Rolling buffers retain only actually received context. Every opportunity gets a small durable
coverage row, including disabled, budget-skipped and capacity-skipped opportunities. It joins the
core signal by identity, so the denominator includes uncaptured opportunities. Each retained window
contains original signal time, contract checkpoints, conId, request/generation, local sequence,
UTC callback receipt, monotonic time, row operation/side/position/price/size, resets/gaps and policy.
There is **no exchange-event timestamp** where the API supplies none. These are IBKR-delivered
displayed rows, not exchange-native order-by-order events or evidence of trader intent.

Coverage is the union of received callback spans, split by resets/gaps/generation or silence over
30 seconds and clipped to the requested window. It does not extrapolate a silent tail. Separate
`complete_book_pre_seconds` and `complete_book_post_seconds` identify spans with all requested levels
valid. Pre-trigger and post-trigger event arrays are separate; post-trigger data never enters
entry features. Missing pre-context stays missing. Optional live summaries are spread, displayed
size per side and `(bid size − ask size)/(bid size + ask size)` only when the book is complete/fresh.

Default ceilings: 10,000 events per book, 32 MiB conservative memory accounting (8 KiB per event
including checkpoint/writer overhead), 32 queued batches, 256 MiB compressed files. Windows and
allocation events are gzip-batched on a worker thread, not inserted into SQLite per callback.
Only opportunity summaries are updated in the core ledger. On memory/disk/queue failure, mark the
gap and pause optional recording; position management continues. Restart interrupts unfinished
windows honestly. Existing files, research and trade records are never deleted to make space;
an operator must archive recording files deliberately before resuming after a storage limit.

## UI and verification

Stable cards show L1, bars, strategy, L2 and positions separately. Expand a card for the depth ladder
and its research-only label. System shows app/account allowances, unknown external use, depth
assignments, option pool, rate/queue pressure, errors, gaps and storage. Browser rendering is five
seconds apart, independent of ingestion. Nodes, expanded details, focus, filters and scroll persist.

Offline tests cover quotas, simultaneous batches, reference ownership, pending/unknown submissions,
urgent pacing, depth-only rejection, history cleanup, generation reuse, row operations/317,
partial coverage, rotation/preemption, memory/disk bounds and identical orders with L2 off/on/failed.
Existing frozen signal/exit, calendar/DST, budgets, reconciliation and removed-runtime checks remain.
Screenshots are actual browser renders with clearly labelled offline fixtures:
[overview](futures-screenshots/overview-desktop-fixture.png),
[expanded depth](futures-screenshots/depth-expanded-fixture.png),
[System/API](futures-screenshots/system-api-fixture.png),
[mobile](futures-screenshots/overview-mobile-fixture.png).

## Official sources checked 2026-09-27

- [Account market-data lines](https://www.interactivebrokers.com/docs/general/market-data-subscriptions/market-data-lines/introduction): TWS and API clients share allowance.
- [Specialized lines](https://www.interactivebrokers.com/docs/general/market-data-subscriptions/market-data-lines/specialized-market-data-lines): baseline three depth/five tick-by-tick; no assumed booster formula.
- [API pacing](https://www.interactivebrokers.com/docs/tws-api/doc/pacing-limitations/introduction): line allocation informs message allowance; default 100 lines / 50 requests per second.
- [Depth overview](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-live/market-depth-l-2/introduction), [callbacks](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-live/market-depth-l-2/receive-market-depth), [cancellation](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-live/market-depth-l-2/cancel-market-depth): row semantics, resets and cancellation interface.
- [Depth exchange metadata](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-live/market-depth-exchanges/receive-market-depth-exchanges).
- [Five-second bars](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-live/5-second-bars/introduction): separate endpoint with special pacing; not introduced here.
- [Historical live updates](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-bars/keep-up-to-date): mutable bars revise at approximately 4–6 second intervals.
- [Small-bar historical pacing](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-data-limitations/pacing-violations-for-small-bars-30-secs-or-less): separate restrictions; unsupported small-bar requests are rejected.

These sources verify API behaviour and planning limits, **not** this account's current entitlements,
available capacity, exchange routing or live callback behaviour. Check those non-transmitting at
approved preflight. Product mappings remain unapproved and all six default entry blocks remain.
