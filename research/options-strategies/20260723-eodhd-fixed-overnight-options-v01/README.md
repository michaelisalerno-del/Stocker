# EODHD Fixed Overnight Options Strategy Quick Screen V0.1

This is the bounded repair of V0. It preserves all frozen signal, contract,
liquidity, cost, control, bootstrap, support, and economic decision rules.

The repair changes two pipeline behaviours only:

1. A provider response is filtered to its exact requested option observation
   date before canonicalisation. Other returned dates are discarded and
   audited; their presence is not itself leakage.
2. S1 and S3 execute independently. S2 and its `2→3→2` veto remain
   `blocked_direction_mapping_unavailable` because no audited structural
   orientation-to-price-direction mapping exists.

The protected boundary applies to market and quote observation dates. A
contract expiration after 2025-08-22 is permitted metadata when its quote
observation is on or before that date. No protected exit or settlement outcome
is opened.

DTE remains the frozen calendar-day difference between expiration and option
observation date. The provider's sometimes one-day-shorter `dte` field is
audited but is not used for selection or economics.

Individual exact-date records without bid/ask timestamps are audited and
discarded before canonicalisation because they cannot satisfy the frozen quote
requirements. They do not invalidate other contracts in the same chain.

This is research-only retrospective economic feasibility work. It does not
simulate intraday option fills, access a broker, place orders, claim realised
P&L, provide prospective validation, or define a deployable strategy.

## Run

From the repository root:

```bash
python research/options-strategies/20260723-eodhd-fixed-overnight-options-v01/run_screen_v01.py
python research/options-strategies/20260723-eodhd-fixed-overnight-options-v01/audit_screen_v01.py
```

The default stock source is the existing local EODHD five-minute store under
`~/StockerLocal/data/processed/source=eodhd/instrument_type=stock`. Raw option
records remain under the ignored `data/vendor/eodhd/options` tree. Small,
reviewable outputs are written to `artifacts/primary`.

Cached V0 raw responses are always reprocessed first. Only genuinely missing
bounded requests are eligible for download. Credentials are never written or
printed.
