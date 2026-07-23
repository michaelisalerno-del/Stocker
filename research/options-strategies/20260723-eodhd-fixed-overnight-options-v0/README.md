# EODHD Fixed Overnight Options Strategy Quick Screen V0

This bounded retrospective experiment asks whether Stocker’s frozen 15:30
structural information can support three simple options constructions when the
only option observations are EODHD end-of-day quotes:

1. S1: next-session ATM straddle after `BROAD_CONFLICT`.
2. S2: next-session oriented debit spread, with the one frozen `2→3→2` veto.
3. S3: DTE-1 ATM straddle through the following expiration session.

Contract identities are selected from the exact previous US trading session.
Entry and exit economics use only the prescribed bid/ask sides. Option daily
highs and lows are prohibited. The runner stops rather than broadening its
contract/date scope when cache or resource bounds are insufficient.
The untuned repository concentration convention caps any assessment month at
30% of a strategy's trades; stock caps remain the strategy-specific support
gates in `contract.json`.

This is research-only retrospective economic feasibility work. It does not
simulate intraday option fills, access a broker, place orders, claim realised
P&L, provide prospective validation, or define a deployable strategy.

## Run

From the repository root:

```bash
python research/options-strategies/20260723-eodhd-fixed-overnight-options-v0/run_screen_v0.py
python research/options-strategies/20260723-eodhd-fixed-overnight-options-v0/audit_screen_v0.py
```

The default stock source is the existing local EODHD five-minute store under
`~/StockerLocal/data/processed/source=eodhd/instrument_type=stock`. Raw option
records remain under the ignored `data/vendor/eodhd/options` tree. Small,
reviewable outputs are written to `artifacts/primary`.

If the exact bounded option cache is incomplete and `EODHD_API_TOKEN` is not
available, the run writes its signal/date/gap evidence and returns
`blocked_missing_eodhd_api_token`. It never prints the token.
