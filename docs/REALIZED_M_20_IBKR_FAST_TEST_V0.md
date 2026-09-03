# IBKR REALIZED_M_20 Historical Data Fast Test V0

Date: 2026-09-03  
Mode: `research_only=true`, `order_placement=disabled`  
Connection: existing Stocker TWS/IB Gateway socket adapter, PAPER session, isolated client ID 9200

## Result

IBKR returned enough one-minute history in one finite request per stock, the existing Stocker
SQLite history cache served a second identical read without another broker request, and the
frozen calculation produced materially equivalent results. Across 21 matched Session HARD
cases, `REALIZED_M_20_RETURN` had Pearson correlation 0.9899 and median absolute percentage
difference 0.6581%. Frozen PRE_MOVE qualification agreed on all 21 cases.

Decision: **IBKR_HISTORY_SUITABLE_FOR_REALIZED_M**

## Locked method

No strategy or execution code was changed. The test independently applies the frozen method:

1. Convert each signal timestamp to `America/New_York` and select the same minute from strictly
   prior sessions.
2. Calculate `abs(close_Tplus14 / open_T0 - 1)`.
3. Take the median of the latest 20 valid sessions, requiring at least 10.
4. Calculate price M as `current_P0 * REALIZED_M_20_RETURN`.
5. Calculate the frozen aligned prior-provider PRE_MOVE value and classify using the unchanged
   threshold `PRE_MOVE_M > 0.475764059845861`.

The research implementation recalculated the legacy ledger's stored M price to a maximum absolute
difference of `4.44e-15`, confirming that it reproduced the existing frozen producer.

## Retrieval experiment

The representative set was selected by data characteristics rather than profitability:

- WULF: lower-priced stock.
- CRWD: higher-priced stock.
- TSLA: high-volatility stock.
- AAPL: ordinary large-cap stock.
- OKLO: frequent Session HARD research name.

Request settings were `durationStr="40 D"`, `barSizeSetting="1 min"`,
`whatToShow="TRADES"`, `useRTH=true`, `formatDate=2`, and `keepUpToDate=false`. This was a
finite historical request, not a live subscription. Current official TWS API documentation defines
these request fields and states that `useRTH` restricts results to regular trading hours. Its current
maximum-duration table permits up to 365 day-duration units for one-minute bars, so the tested
40-day request is supported: [request fields](https://ibkrcampus.com/docs/tws-api/doc/market-data-historical/historical-bars/requesting-historical-bars),
[maximum duration](https://ibkrcampus.com/docs/tws-api/doc/market-data-historical/historical-bars/max-duration-per-bar-size).

| Symbol | Requests | Bars | Completed prior sessions | Oldest UTC | Newest UTC | Duplicates | Missing expected RTH minutes | Failures / retries | Measured seconds |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| WULF | 1 | 15,420 | 39 | 2025-06-18 13:30 | 2025-08-14 19:59 | 0 | 0 | 0 / 0 | 33.309 |
| CRWD | 1 | 15,240 | 39 | 2024-11-27 14:30 | 2025-01-28 20:59 | 0 | 0 | 0 / 0 | 44.159 |
| TSLA | 1 | 15,600 | 39 | 2025-04-29 13:30 | 2025-06-25 19:59 | 0 | 0 | 0 / 0 | 51.944 |
| AAPL | 1 | 15,600 | 39 | 2025-12-26 14:30 | 2026-02-24 20:59 | 0 | 0 | 0 / 0 | 59.170 |
| OKLO | 1 | 15,600 | 39 | 2025-03-28 13:30 | 2025-05-23 19:59 | 0 | 0 | 0 / 0 | 51.722 |

The five sequential requests took 240.305 seconds in total. The lower bar counts for WULF and CRWD
are explained by exchange-calendar early closes in their windows; calendar-aware completeness found
no missing RTH minute. No chunking was needed and no undocumented limit was inferred.

The TWS documentation's historical pacing page is specifically scoped to bars of 30 seconds or
less, so its small-bar rules were not applied to this one-minute test:
[small-bar pacing scope](https://ibkrcampus.com/docs/tws-api/doc/market-data-historical/historical-data-limitations/pacing-violations-for-small-bars-30-secs-or-less).

## Cache behavior

The experiment used the existing `IbkrHistoryCache` and `IbkrHistoryService`; it did not create a
parallel data layer. Each cold lookup was `NOT_READY`, one validated IBKR response was stored under
`conId + bar size + whatToShow + useRTH + UTC timestamp`, and a second identical required-range
lookup was `READY` for every symbol.

| Measure | First load | Second identical cache-first load |
|---|---:|---:|
| IBKR historical requests | 5 | 0 |
| Cache misses | 5 | 0 |
| Cache hits | 0 | 5 |

Normal Stocker Stage 4/5 callers already follow this cache-first pattern: inspect exact required
timestamps, request only when `NOT_READY`, persist, then read the complete snapshot. Daily operation
should therefore request only missing/new bars rather than repeat the 40-day backfill.

## IBKR REALIZED_M_20 cases

| Symbol | Session | T0 | Valid prior sessions | IBKR REALIZED_M_20 return | IBKR REALIZED_M_20 price |
|---|---|---|---:|---:|---:|
| AAPL | 2026-02-24 | 2026-02-24T15:00:00+00:00 | 20 | 0.00335624 | 0.919760 |
| CRWD | 2025-01-28 | 2025-01-28T15:00:00+00:00 | 20 | 0.00426119 | 1.649294 |
| CRWD | 2025-01-28 | 2025-01-28T15:10:00+00:00 | 20 | 0.00534465 | 2.097721 |
| CRWD | 2025-01-28 | 2025-01-28T15:20:00+00:00 | 20 | 0.00322850 | 1.280133 |
| OKLO | 2025-05-23 | 2025-05-23T14:00:00+00:00 | 20 | 0.01004395 | 0.487132 |
| OKLO | 2025-05-23 | 2025-05-23T14:10:00+00:00 | 20 | 0.01024925 | 0.508260 |
| OKLO | 2025-05-23 | 2025-05-23T14:20:00+00:00 | 20 | 0.01027488 | 0.507476 |
| OKLO | 2025-05-23 | 2025-05-23T14:30:00+00:00 | 20 | 0.00742587 | 0.370328 |
| OKLO | 2025-05-23 | 2025-05-23T14:40:00+00:00 | 20 | 0.01078348 | 0.530817 |
| OKLO | 2025-05-23 | 2025-05-23T14:50:00+00:00 | 20 | 0.00762561 | 0.380959 |
| OKLO | 2025-05-23 | 2025-05-23T15:00:00+00:00 | 20 | 0.00665831 | 0.327922 |
| OKLO | 2025-05-23 | 2025-05-23T15:10:00+00:00 | 20 | 0.00366487 | 0.180898 |
| OKLO | 2025-05-23 | 2025-05-23T15:20:00+00:00 | 20 | 0.00693039 | 0.358093 |
| OKLO | 2025-05-23 | 2025-05-23T15:30:00+00:00 | 20 | 0.00427727 | 0.218825 |
| OKLO | 2025-05-23 | 2025-05-23T16:00:00+00:00 | 20 | 0.00596390 | 0.290979 |
| TSLA | 2025-06-25 | 2025-06-25T14:10:00+00:00 | 20 | 0.00193296 | 0.629721 |
| TSLA | 2025-06-25 | 2025-06-25T14:20:00+00:00 | 20 | 0.00306522 | 0.993837 |
| TSLA | 2025-06-25 | 2025-06-25T14:30:00+00:00 | 20 | 0.00290731 | 0.934787 |
| WULF | 2025-08-14 | 2025-08-14T14:00:00+00:00 | 20 | 0.01161462 | 0.085890 |
| WULF | 2025-08-14 | 2025-08-14T14:30:00+00:00 | 20 | 0.00507828 | 0.039966 |
| WULF | 2025-08-14 | 2025-08-14T15:00:00+00:00 | 20 | 0.00620791 | 0.048701 |

## Provider equivalence

| Metric | Result |
|---|---:|
| Comparisons | 21 |
| Pearson correlation | 0.989911 |
| Spearman correlation | 0.985714 |
| Median ratio, IBKR / existing | 0.999002 |
| Median absolute percentage difference | 0.658057% |
| Within +/-5% | 90.476190% |
| Within +/-10% | 90.476190% |
| Within +/-20% | 95.238095% |

WULF matched exactly for all three test cases. Median absolute return differences were 0.134% for
TSLA, 0.664% for OKLO, and 2.005% for AAPL. CRWD was the only material row-level outlier: the legacy
provider file starts on 2025-01-01 and supplies only 16 valid prior sessions for this case, while
IBKR supplies the intended 20. The legacy CRWD OHLC price scale is also exactly four times the IBKR
scale on the median overlapping bar. The constant scale factor cancels in percentage returns; the
different available-session set explains the remaining 17.160% median and 35.062% maximum CRWD
return difference.

No split-fitting, scaling, or provider adjustment was applied. On the other four stocks the median
provider/IBKR close ratio was 1.0. Small same-minute OHLC discrepancies are expected between feeds,
and IBKR states that its historical feed is filtered, adjusted, and compressed:
[historical filtering](https://ibkrcampus.com/docs/tws-api/doc/market-data-historical/historical-data-limitations/historical-data-filtering).
All timestamps were normalized to UTC before exact matching and to `America/New_York` only for
session/minute selection. The provider files also include extended-hours rows; the frozen calculation
and IBKR request both used RTH observations only.

## PRE_MOVE equivalence

| Metric | Agreement |
|---|---:|
| Pearson correlation | 0.997365 |
| Spearman correlation | 0.996104 |
| Median absolute difference | 0.003915 |
| Both qualify | 13 |
| Both reject | 8 |
| IBKR-only qualify | 0 |
| Existing-provider-only qualify | 0 |
| Classification agreement | 100.000% |
| Jaccard overlap | 1.000000 |

## Practical scaling

The observed cold shape is one finite request and roughly 15,240-15,600 one-minute bars per symbol,
followed by zero broker requests for an identical cache-first read. At the measured sequential mean
of 48.061 seconds, cold backfill is the clear bottleneck: it should be performed once, ahead of the
causal deadline, and reused. Stocker's existing broker-local historical concurrency ceiling is four;
this test deliberately remained sequential and does not claim a fourfold speed-up or an official
IBKR concurrency limit.

For 100 symbols, the measured request/bar shape extrapolates to 100 initial requests and about 1.55
million cached bars—not repeated 20-session downloads at every scan. This is operationally practical
as `initial backfill once -> local cache -> incremental daily updates`, provided initial onboarding is
scheduled rather than placed on the live checkpoint critical path. No hundreds-symbol stress test was
performed.

## Explicit answers

1. **Can IBKR provide enough one-minute history for REALIZED_M_20?** Yes. Every test stock returned
   39 completed prior sessions in one supported 40-day request.
2. **Can the app calculate REALIZED_M_20 entirely itself?** Yes. Only IBKR one-minute OHLC and the
   frozen current P0 are required; the application reproduced the formula independently.
3. **Can initial history be cached so subsequent scans do not redownload 20+ days?** Yes. Five cold
   requests became zero requests on five identical cache-first reads.
4. **Does IBKR produce materially equivalent REALIZED_M_20 and PRE_MOVE classifications?** Yes for
   this representative fast test: high return correlations, 0.658% median absolute percentage
   difference, and 100% agreement across 21 frozen classifications.
5. **Is this practical for eventual scanning of hundreds of US stocks?** Yes with one-time scheduled
   backfill and incremental cache updates. The obvious constraint is cold retrieval latency, not data
   sufficiency or formula equivalence.

## Reproduction artifacts

- `research/realized_m_20_ibkr_fast_v0/contract.json`
- `research/realized_m_20_ibkr_fast_v0/run_experiment.py`
- `research/realized_m_20_ibkr_fast_v0/artifacts/raw/retrieval_metrics.json`
- `research/realized_m_20_ibkr_fast_v0/artifacts/primary/comparisons.csv`
- `research/realized_m_20_ibkr_fast_v0/artifacts/primary/equivalence_summary.json`
- `research/realized_m_20_ibkr_fast_v0/artifacts/primary/source_differences.json`

No order was placed. Session HARD, Structure D, REALIZED_M_20, PRE_MOVE, thresholds, stops, targets,
and execution handling were unchanged.
