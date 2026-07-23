# EODHD Fixed Overnight Options Strategy Quick Screen V0

## Decision

`blocked_chronology_or_leakage_failure`

S1, S2, the frozen S2 `2→3→2` veto, and S3 are all marked `blocked`.
The run failed closed before contract construction: requested option observation date 2024-10-08 but provider returned 2024-10-08,2024-10-10.
Independently, S2 cannot start because the repository has no audited
orientation-to-price-direction mapping.

## Frozen stock signal

- Cohort: 20 stocks.
- Regular-session stock-state rows: 8240.
- Ordinal-72 structural rows: 7786.
- Period support: `{"assessment": 3139, "development": 4647}`.
- Route states: `{"BROAD_CONFLICT": 1148, "DOMINANT_ROUTE": 1940, "LOW_ROUTE_SUPPORT": 2014, "NARROWING": 185, "OTHER": 2499}`.
- Broad-conflict candidates before IV: 1144.
- Source exclusions: `{"required_causal_feature_reconstruction_failed": 226, "source_data_unavailable": 226}`.
- Protected market or option rows materialised: 0.

Ordinal 72 is the completed-bar count: zero-based bar 71 starts at 15:25 New
York time and becomes available at 15:30. Normal sessions retain six complete
five-minute bars before the scheduled close; sessions without the required
late bar are retained as unavailable rows.

## Options coverage

- Required option date rows: 6864.
- Missing/incomplete strategy-date chains: 5644.
- Prior cached contract-history provider records: 3624.
- Experiment-specific bounded provider records acquired: 3530.
- Experiment-specific complete exact-date query receipts: 142.
- Safe canonical pre-boundary option observations: 3795.
- Current invocation network records: 0.
- Current invocation network bytes: 0.
- Cumulative bounded logical response bytes: 3228447.

Contract histories whose OCC expiry could cross the protected boundary were
not opened. No contract was reselected, replaced, or forward-filled.

## Economic results

No strategy trade, P&L, matched-control return, or bootstrap estimate was
computed under the blocker. The required metric tables preserve the frozen
schema and mark all values blocked. Ten bootstrap draws are recorded as the
frozen configuration, not presented as inference.

This report is a retrospective research screen. It does not claim intraday
option execution, IBKR fills, realised profits, prospective validation, or a
deployable strategy.

## Independent audit

Passed with no unexplained discrepancy.
