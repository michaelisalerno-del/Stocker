# Fix options for exact historical EODHD observation dates

## Recommendation

Keep the completed V0 blocker intact and create a retrieval-only V0.1 amendment. Replace
symbol-month `tradetime` queries with bounded, contract-centric EOD histories:

1. Discover immutable contract identities through `/mp/unicornbay/options/contracts`, restricted
   by the frozen symbol, expiration, and strike ranges.
2. Fetch each necessary contract's history through
   `/mp/unicornbay/options/eod?filter[contract]=...`.
3. Define the EOD observation date from the JSON:API resource-ID suffix.
4. Require the New York dates of `bid_date` and `ask_date` to equal that observation date and
   require expiration/DTE arithmetic to agree.
5. Select only the exact required previous session locally. Never use `tradetime` as the chain
   date.

EODHD's official guide explicitly presents `filter[contract]` as a single-contract EOD time
series and says its timestamps are in `bid_date`. The same guide presents `tradetime` queries as
an activity scan. The official OpenAPI exposes `filter[contract]` but no observation-date or
`bid_date` filter.

Sources:

- [EODHD practical options guide](https://eodhd.com/financial-academy/stock-options/a-practical-us-options-api-guide-from-activity-scan-to-key-strikes-via-eodhd-api)
- [Official options EOD OpenAPI path](https://github.com/EodHistoricalData/EODHD-openapi/blob/main/paths/mp_unicornbay_options_eod.yaml)
- [Official usage examples explaining `tradetime`](https://eodhd.com/financial-academy/stock-options/us-stock-options-api-usage-examples)

## Pair-selection safeguards

- Enumerate expiries in the frozen nearest-first order and strikes by the frozen ATM-distance
  order using immutable contract metadata and the previous unadjusted close.
- Retrieve candidate histories progressively until the earliest expiry with an exact-date common
  call/put strike is established.
- Apply open-interest, spread, and IV tie-breaks only from exact-date observations.
- Once the expiry and pair are selected, mark the pair unavailable if quality fails; do not fall
  back to a later expiry.
- Cache each contract history once and deduplicate by contract plus observation date.
- Quarantine all non-required dates from canonical and analytical panels. Audit zero same-day,
  future, and protected-date joins.

## Bounded feasibility gate

Before cohort retrieval, run a small fixed contract-history probe across frozen high/medium/low
density symbols. Measure unique contracts, rows per history, bytes, paging, and exact-date hit
rate. Project the full request plan and stop if it can exceed 3,000,000 raw option records or
20 GB. Split oversized work by expiration, strike, type, and finally exact contract; never accept
offset truncation.

This changes retrieval mechanics only. It does not change the cohort, dates, pair rule, outcomes,
models, gates, or research-only safety contract.

## Bounded live probe result

A retrieval-only V0.1 probe subsequently tested AAL, MSTR, and WULF at the earliest, midpoint,
and latest shared required option dates. The live API exposed expired contract identities and
exact-date contract histories. It returned 3,624 provider records (1,029,673 bytes) through 27
recorded HTTP attempts; all nine stock-dates had exact observations and eight passed the frozen
pair-quality rule. The one unavailable pair failed the existing spread gate without fallback.

The live responses omitted the documented `meta.total` while providing contiguous
`page[offset]`, `page[limit]`, and `links.next`. The downloader therefore validates next-link
offset continuity, requires final-page next-link absence, preserves the 10,000 offset ceiling,
and revalidates the same conditions on resume. This probe supports the retrieval route only; it
does not replace the V0 decision or answer the movement question.

## Other valid routes

1. Ask EODHD support for a documented resource-observation/`bid_date` filter or a date-partitioned
   bounded export. This would allow the original V0 request plan to run unchanged. The official
   product page lists only the contracts, EOD, and underlying-symbol endpoints and directs API
   questions to support.
2. If EODHD cannot support either route, version a provider-substitution experiment. Intrinio's
   official Options Chain EOD endpoint has an explicit `date` parameter, and ORATS' historical
   strikes endpoint has an explicit `tradeDate`. That would answer a provider-neutral prior-close
   IV question, not the original EODHD-specific question.

Sources:

- [EODHD options product](https://eodhd.com/lp/us-stock-options-api/)
- [Intrinio Options Chain EOD](https://docs.intrinio.com/documentation/web_api/get_options_chain_eod_v2)
- [ORATS Historical Data API](https://orats.com/docs/historical-data-api)

## Non-solutions

- Reinterpreting `tradetime` as the observation date.
- Forward-filling or backfilling a nearby chain.
- Filtering returned future snapshots after having treated the `tradetime` request as complete.
- Using the undocumented legacy options endpoint without a new schema and chronology audit.
- Relaxing the exact previous-session, pair-quality, or coverage gates.
