# Observation-only book flow — incremental Saxo addendum

Feature version: `SAXO_SAMPLED_BOOK_FLOW_V1`. Baseline: `b066386` on `codex/saxo-only`.
The work extends the existing Prices consumer, recorder and Markets view. No second data pipeline,
new service, strategy rule, capture trigger or execution control is introduced.

## Subscription and data contract

Saxo's current reference lists ordinary single-instrument Prices subscriptions as **Personal:
Subscribe**, while grouped InfoPrices subscriptions require **Restricted: Subscribe**. SLRNO keeps
one ordinary subscription per actual underlying UIC on its existing server-owned connection. The
grouped subscription endpoint is not allowlisted or required. Repeated consumers share the existing
subscription; replacement explicitly removes the old local generation. Requested cadence is 1,000 ms;
the returned `RefreshRate` and measured inter-receipt intervals are separate fields.
[Prices subscription](https://www.developer.saxo/openapi/referencedocs/trade/v1/prices/post__trade__subscriptions),
[restricted InfoPrices subscription](https://www.developer.saxo/openapi/referencedocs/trade/v1/infoprices/post__trade__subscriptions).

Quote, PriceInfoDetails and MarketDepth remain requested, alongside the already-required
InstrumentPriceDetails. Current Quote.BidSize/AskSize take precedence, including explicit nulls.
The existing paper size gate retains compatibility with previously supported legacy fields only
when the corresponding Quote field is absent; diagnostics use current Quote sizes exclusively.
[Quote schema](https://www.developer.saxo/openapi/referencedocs/trade/v1/prices/put__trade__subscriptions_contextid_referenceid_marginimpact/schema-tradablequote),
[PriceInfoDetails and obsolete sizes](https://www.developer.saxo/openapi/referencedocs/trade/v1/prices/put__trade__subscriptions_contextid_referenceid_marginimpact/schema-priceinfodetails).

Every delivered price message is retained unabridged with environment/UIC/contract, subscription
generation, opaque message ID, local sequence, UTC receipt and valid provider timestamps. The same
rows carry derived observations and availability/quality metadata. NoNewData heartbeats are control
observations, not trades or price changes. Health uses the granted subscription inactivity timeout;
last data receipt, last field change and last depth change are distinct. Field omissions retain prior
values; nulls clear them and arrays replace them. Reconstruction gaps invalidate lookbacks. The
existing stricter trading quote-age gate is unchanged.
[Streaming semantics](https://www.developer.saxo/openapi/learn/plain-websocket-streaming).

## Fixed formulas

All metrics use reconstructed information received by their calculation timestamp. Queued messages
keep their original receipt times, but calculations using a later initial snapshot are never backdated.
Prices must match the verified instrument tick grid within 0.000001 tick. No interpolation, liquidity
padding, historical backfill of L2, or use of later observations is permitted.

Let B/A be best bid/ask, qB/qA the current Quote sizes, and tick the reference tick size.
For N in **1, 3, 5, 10**, DB(N)/DA(N) are sums of the first N received bid/ask depth sizes.

| Feature | Formula, units and lookback |
|---|---|
| Spread | `(A − B) / tick`, ticks, current observation |
| Cumulative depth | `DB(N)`, `DA(N)`, provider-reported amount units, current observation |
| Depth imbalance | `(DB(N) − DA(N)) / (DB(N) + DA(N))`, dimensionless, current observation |
| Order-count imbalance | Same ratio applied to summed BidOrders/AskOrders; only integer nonnegative counts with UsingOrders=true |
| Size-weighted midpoint | `(A × qB + B × qA) / (qB + qA)`, instrument price units |
| Midpoint displacement | `(weighted midpoint − (A+B)/2) / tick`, ticks |
| Observed depth changes | Current DB/DA minus the as-of values 5, 30 and 60 seconds earlier, for each N |
| Common-price changes | Sum of size differences only for tick prices visible at both endpoints; reports the number matched |
| Persistence | Fraction of the previous 60 seconds with positive, negative or zero depth imbalance, for each N |

Persistence holds each received observation until the next one, weighted by elapsed
time, including quiet intervals whose subscription remained healthy. BID_HEAVY means strictly positive
imbalance, ASK_HEAVY strictly negative, BALANCED exactly zero; these are descriptive labels only.
The compact panel shows five-level changes/persistence; all four N values are available in its details
and saved observations. No thresholds are tuned and no combined score is computed.

Fewer than N valid levels makes that N unavailable. Missing sizes/counts stay missing; a zero or
invalid denominator returns null. Missing baselines, intervening missing/invalid depth, a timeout,
delay change, subscription replacement or feed gap returns INSUFFICIENT_HISTORY. The calculation
scans at most 512 retained observations for its 60-second horizon; unusually high delivered rates
can therefore shorten feature coverage without changing raw retention. Delayed observations are
labelled DELAYED. Missing L2 leaves L1 observations available, with an L2_UNAVAILABLE flag.
[Depth fields and feed-right dependence](https://www.developer.saxo/openapi/referencedocs/trade/v1/prices/put__trade__subscriptions_contextid_referenceid_marginimpact/schema-marketdepth).

Price levels entering or leaving the visible book are not executions or cancellations. LastTraded
and LastTradedSize are displayed as the latest observation only: repetitions create no executions,
and sizes are never summed. Volume is the reported field only. Instrument/session-specific volume
semantics remain unverified, so volume changes are always unavailable in this release. A decrease
is flagged RESET_OR_CORRECTION_UNCLASSIFIED, never made positive or assigned to a trade price.
No aggressor volume, footprint, cumulative delta, absorption or iceberg claims are generated.

## Retention and operational boundaries

Raw and derived rows share the existing 15-minute RAM buffer, memory accounting, advancing
reconstruction checkpoint, queue, compressed event segments and archive quota. The checkpoint
also retains its last derived observation. Features create no event. Existing frozen-trigger events,
including skips, preserve the same prehistory, 60-minute minimum, owned-trade extension and five-minute
tail. Overlapping events share segments. Only that event's underlying and linked subscribed options
append; completed captures receive no further continuous data.

Manifests include feature version, actual coverage, available fields, quality flags and granted/measured
cadence. Archive failure retains the existing safe stop/incomplete semantics and does not block the
independent risk manager. Malformed optional diagnostic groups retain their raw message and report
unavailable features. Persistent recording remains permission-gated; nothing enables it automatically.

## Verification and outstanding access

Implemented/offline-tested: fixed feature formulas, tick-price shifts, absent counts/levels, zero
denominators, repeated latest trades, volume decreases, feed gaps, quiet-book heartbeats, duplicate
consumers/replacement, checkpoint reconstruction, event overlap/routing, completed-capture boundaries,
storage limits, unchanged frozen decisions and current Quote-size precedence. Existing live endpoint,
budget and concurrent-capacity tests remain applicable. Browser fixtures check the Markets-only panel,
mobile layout, distinct cadence labels, quiet ladders and scroll/focus/expanded-state preservation.

| Market | Authenticated feed | Received L2 levels/fields | Delay | Granted/observed cadence |
|---|---|---|---|---|
| CL | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| GC | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| NG | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| NQ | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| SI | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |

No authenticated Saxo access was available. Outstanding actions remain the environment-specific
OAuth grant and UIC selection, actual per-market L2/count/quote entitlements, applicable recording
permission and instrument-specific volume/session semantics. No subscriptions or agreements were
purchased/accepted. No execution was armed, no test order sent and no production service changed.
Fixtures prove implementation behavior, not account access.

Validation: **436 tests passed**, with seven existing NumPy/dependency warnings; the ten new book-flow
tests also pass independently. Ruff formatting/lint (198 files), Mypy (116 source files), locked
Playwright browser checks and the isolated server-only installation smoke pass. A final targeted
rerun covers the provider-envelope timestamp extraction and incomplete-manifest feature metadata.
The generated two-virtual-hour workload is [machine-readable here](saxo-book-flow-benchmark.json);
it is not authenticated evidence. Screenshots are explicitly labelled offline fixtures.

Generated workload: five contracts at one observation per second for two virtual hours, 36,005
messages, three events sharing two archive segments. All five buffers retained 900 seconds with
zero recording gaps and an empty final write queue. The accelerated run took 145.7 seconds.

| Resource/check | Measured result |
|---|---|
| Accounted buffer high-water | 25.71 MiB / 32 MiB limit |
| Whole-process peak RSS | 88.22 MiB |
| Compressed event archive | 4.30 MiB / 2 GiB limit |
| Ingest latency | 8.55 ms p99; 18.01 ms maximum |
| Maximum event-loop delay | 96.78 ms under accelerated workload |

These are local generated-workload measurements, including feature calculation and archive writes;
they do not establish provider cadence, production resource usage or authenticated recorder/L2 status.

## Standards review

Two findings, both fixed and independently rechecked. A P1 capture-routing regression could append
an unrelated market and omit option messages; routing now stays inside the affected-instrument and
CAPTURING guard, independently of feature availability. A P2 malformed optional-depth shape could
escape the diagnostic reducer; group validation and an unavailable fallback now retain the raw row
without interrupting the shared feed. New regressions reproduce both failures and their fixes.

## Spec review

One P1 finding, the capture-routing issue above, fixed and rechecked. No remaining concrete spec
defects or scope creep were found in the bounded review. Ordinary Prices subscriptions, descriptive
features, heartbeat health, missing-data labels and Markets-only rendering were checked.

Review totals: Standards 2 resolved / 0 open (initial worst P1); Spec 1 resolved / 0 open (initial worst P1).
