# IBKR API Resource Audit

This audit covers Stocker V1 through Stage 10. It distinguishes the TWS / IB Gateway socket API used by Stocker from the Client Portal Web API; no Stocker runtime code uses the Web API.

## Request map

| Request type | Caller | Streaming or finite | Uses market-data line | Current cancellation | Shared across runs | Potential burst |
|---|---|---|---|---|---|---|
| SCANNER — scanner parameters | `IbkrConnection.scanner_capabilities` | Finite RPC | No | Completes naturally | Per broker process/session cache | Low; one large XML response on cold cache |
| SCANNER — Activity Shortlist components | `ActivityShortlistService` → `IbkrConnection.activity_scan` (`TOP_TRADE_RATE`, `TOP_VOLUME_RATE`, `HOT_BY_VOLUME`) | Finite scanner collection | No quote line; returned rows are contracts | Stocker uses ib_async's low-level request/future and cancels in `finally`, including timeout | Frozen screen keyed by physical market/cap/session | Three sequential cold-screen requests; broker-local concurrency cannot exceed 10 |
| SCANNER — legacy hot-volume screen | `IbkrConnection.hot_us_stocks_by_volume` | Finite scanner collection | No | Same timeout-safe low-level request and `finally` cancellation as Activity Shortlist | Used during universe materialisation | Low; broker-local bound applies |
| FINITE RPC — stock qualification | Stage 2 `IbkrConnection.qualify_stock` | Finite `qualifyContractsAsync` | No | Not applicable | Session cache by symbol/exchange/primary exchange/currency, then stable `conId` | One cold request per unique physical stock |
| FINITE RPC — option qualification | Stage 4 `PreContextService` | Finite `qualifyContractsAsync` | No | Not applicable | Stage 4 context persistence prevents repeat work across equivalent runs | Can qualify several strike candidates for the selected expiry, but no quote lines are opened by qualification |
| FINITE RPC — option-chain definition | Stage 4 `IbkrConnection.option_chains` | Finite `reqSecDefOptParamsAsync` | No | Not applicable | Session cache by underlying `conId` | One cold request per underlying context |
| FINITE RPC — minimum tick | Stage 7 execution planning via `minimum_tick` | Finite `reqContractDetailsAsync` | No | Not applicable | Existing order-plan flow | Only due entry candidates |
| HISTORICAL — prior-session PRE bars | Stage 4 context | Finite `reqHistoricalDataAsync`, `keepUpToDate=False` | No streaming line | Completes naturally | Stage 4 context key is `conId` + target session + calculation version | One cold request per unique context |
| HISTORICAL — current-session 1m/5m bars | Stage 5 checkpoints | Finite `reqHistoricalDataAsync`, `keepUpToDate=False` | No streaming line | Completes naturally | Persistent cache keys physical `conId` and exact bar semantics, independent of run/environment | Cold upper shape is two requests per unique stock/checkpoint; broker-local concurrency is 4 |
| HISTORICAL — entry observation | Stage 8 waiting-entry observation | Finite 1m history | No streaming line | Completes naturally | Only active waiting signals; cache remains physical-data keyed | Bounded by active immediate candidates, not the watchlist |
| SNAPSHOT — underlying quote | data diagnostic and open-position marking through `current_quote` | Finite `reqTickersAsync` snapshot | Transient server-side snapshot use, not a long-lived Stocker stream | Snapshot completes server-side | One preferred generic market-data broker; callers remain sequential | Only requested for the specific diagnostic/position being marked |
| TEMPORARY STREAM — option quote/model | Stage 4 `option_snapshots` | Streaming capture for bid/ask, generic tick 101 open interest, and tick-13 model computation | Yes, one per selected option contract | Registry release in `finally`, including timeout, invalid result, exception, and `NOT_READY`; final consumer calls `cancelMktData` | Exact physical stream key (`conId`, security type, exchange, ticks, market-data type) with reference counting | Normally one call/put pair (2 lines); an exact nearest-strike tie may select two pairs (4 lines) |
| PERMANENT / LONG-LIVED | None in the Stage 1–10 watchlist or PRE path | None | None | Not applicable | Not applicable | A 50-stock watchlist opens zero streams by membership alone |
| SNAPSHOT / TICK-BY-TICK | No `reqTickByTickData` caller exists | Not used | Not used | Not applicable | Not applicable | Tick-by-tick was not introduced |
| FINITE RPC — account state | Stage 7/8 runtime `account_state` | Finite `accountSummaryAsync` | No | Completes naturally | Kept separate per PAPER/LIVE execution connection | One per broker/runtime need |
| FINITE RPC — positions/open/completed orders/executions | Stage 8/9 startup, reconciliation, and ledger recovery | Finite `reqPositionsAsync`, `reqAllOpenOrdersAsync`, `reqCompletedOrdersAsync`, `reqExecutionsAsync` | No | Completes naturally | Never shared across execution environments/accounts | Startup and reconnect reconciliation only |
| EXECUTION — place order | Stage 7/8 protected-order submission | Immediate `placeOrder` messages | No | Managed by order lifecycle | Never shared | Not placed behind a Stocker-owned data FIFO |
| EXECUTION — cancel order | Stage 8 order management | Immediate `cancelOrder` message | No | Not applicable | Never shared | Not placed behind a Stocker-owned data FIFO |
| FINITE READ — dashboard diagnostics | Stage 10 read service | Runtime/store/ledger read only | No | Not applicable | Reads the current projection | Browser refresh creates zero IBKR requests |

No direct `cancelHistoricalData` call is needed because every historical request is finite with `keepUpToDate=False`. No TWS API path uses Client Portal Web API rate limits.

## Installed ib_async pacing

The installed and locked version is `ib_async 2.1.0`. Its client owns a single outbound message queue and defaults to `Client.MaxRequests = 45` during a one-second `RequestsInterval`. Stocker reuses that mechanism. For a configured Stocker API line budget `B`, the adapter configures the existing ceiling as `min(ib_async ceiling, max(1, B // 2))`; it does not add a second token bucket or scatter sleeps through Stage 4/5/runtime code.

At the default budget of 100, the effective library limit remains 45 requests/second, preserving headroom below the approximately 50 requests/second relationship for 100 market-data lines. Raising Stocker's budget does not silently exceed the installed library ceiling. Lowering it lowers the same central throttle.

## Market-data ownership and separation

`market_data_line_budget` means the maximum simultaneous lines Stocker may consume on that actual connection. Its default is 100. It is explicitly labelled **Stocker API line budget**, never IBKR account maximum. TWS displays and other API clients may consume the account's shared allowance, so the operator can lower the value. PAPER and LIVE budgets are not added together.

The runtime continues to select one preferred `_market_data_broker` for generic Stage 4/5/activity data. PAPER/LIVE account state, orders, fills, positions, risk, strategy state, and P&L remain isolated.

## Stage 4 result

Stage 4 first reuses a persisted context keyed by underlying `conId`, target session, and calculation version. A cache hit makes zero option-chain, option-qualification, or option-market-data requests. On a cold context it requests the security definition, qualifies the frozen selector's expiry/strike candidates, and opens quote streams only for the selector's nearest common-strike call/put set. It still uses generic tick 101 for open interest and tick 13 (`modelGreeks`) for model IV. It was not changed to snapshot or generic tick 106.

Every selected option stream is registered, bounded, reference-counted, and released in `finally`. A normal context uses two simultaneous option lines; the selector can produce four only for an exact logarithmic-distance tie. Contexts and instruments are currently processed sequentially, so those lines do not multiply by watchlist size.

## Stage 5 measurement

For a cold unique stock/session under the current Session HARD schedule:

- Stage 4 PRE context: one prior-session historical request.
- Stage 5: at most one 5-minute and one 1-minute request at each of 15 checkpoints.
- Cold upper request shape: `1 + (2 × 15) = 31` historical requests per unique stock/session.
- A 50-stock cold session upper shape: 1,550 finite historical requests, spread across checkpoints and sequential instrument processing—not a simultaneous burst.
- A same-key cache hit makes zero new historical requests. Overlapping PAPER/LIVE/strategy runs reuse the physical-data cache and do not multiply this count.

The normal runtime does not issue identical historical requests concurrently because checkpoint cycles and instrument processing are already serialized. Consequently no in-flight-coalescing framework was added. A broker-local semaphore limits any unexpected external concurrency to four.

## Scanner result

Activity Shortlist V1 makes three finite component scans per unique market/cap/session screen. Components run sequentially. The installed `ib_async.reqScannerDataAsync` cancels after normal finite completion, but an outer timeout can interrupt it before that cancellation. Stocker therefore uses the same library's low-level scanner request/future and cancels in `finally`, including timeout. The adapter also enforces a broker-local maximum of ten active API scans. Each component's requested and accepted row count is capped at 50. Scanner parameters are cached for the connection/session and cleared on disconnect.

## Reconnect result

Disconnect first cancels every connection-local physical stream once, then clears the subscription registry, scanner-parameter cache, qualification cache, and option-chain cache. Old request IDs are discarded. The runtime's existing startup sequence reconnects and reconciles execution state before permitting new entries; the persisted frozen Activity Shortlist screen and Stage 4/5 caches are reused. A deterministic 20-stream disconnect/reconnect test ends at zero—not 40—streams.

## Capacity scenarios

| Scenario | Configured memberships | Unique physical stocks | Scanner requests | Cold history upper shape | Peak scanners | Expected peak quote lines |
|---|---:|---:|---:|---:|---:|---:|
| A: NASDAQ/HARD/MID, 50 | 50 | 50 | 3 | 1,550 | 1 | 2 option, 0 underlying |
| B: same physical 50 across PAPER/LIVE/strategy | 150 | 50 | 3 | 1,550 | 1 | 2 option, 0 underlying |
| C: four distinct 50-stock screens | 200 | 200 | 12 total, sequential | 6,200 | 1 | 2 option, 0 underlying |
| D: reconnect with 20 temporary streams | 20 | 20 | 0 | 0 | 0 | 20 before disconnect; 0 after reconnect |
| Realistic four-run overlap (three shared, one distinct) | 200 | 100 | 6 | 3,100 | 1 | 2 option, 0 underlying |

These counters come from a deterministic fake broker that executes scanner, qualification, Stage 4, Stage 5 cache, capacity-rejection, execution-safety, and reconnect operations; they are not constants copied into the result. The expected Stage 4 peak is two option lines; the selector's rare exact-tie ceiling is four. Underlying position/diagnostic snapshots are finite and sequential. The design therefore reasonably operates within a 100-line Stocker budget with substantial line headroom; the meaningful load is finite historical throughput, smoothed over checkpoints rather than converted into permanent quote subscriptions.

## Sources

- IBKR market-data lines and sharing: <https://ibkrcampus.com/docs/general/market-data-subscriptions/market-data-lines/introduction>
- TWS API market-data semantics: <https://interactivebrokers.github.io/tws-api/market_data.html>
- TWS API scanner limits and quote separation: <https://interactivebrokers.github.io/tws-api/market_scanners.html>
- TWS API historical-data limitations: <https://interactivebrokers.github.io/tws-api/historical_limitations.html>
