# EODHD Fixed Overnight Options Strategy Quick Screen V0.1

## Decision

`no_eodhd_options_strategy_feasibility`

Constructed 395 treated strategy trades from 2009 causally preselected contract pairs. S2 remains blocked because the repository has no audited orientation-to-price-direction mapping.

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
- Remaining unavailable strategy-date chains: 2781.
- Cached raw responses examined: 5786.
- Cached responses recovered: 3005.
- Exact-date records recovered: 144283.
- V0 cached exact-date records: 3529.
- V0.1 acquired exact-date records: 147606.
- Other-date records discarded before materialisation: 8594.
- Post-boundary records discarded before materialisation: 104.
- Provider DTE disagreements audited: 13665.
- Unquotable exact-date rows discarded: 6852.
- Safe canonical pre-boundary option observations: 144283.
- Cumulative V0.1 repair download records: 156199.
- Cumulative V0.1 repair download bytes: 143319539.

Contract expiration beyond 2025-08-22 was retained only as causal metadata.
No post-boundary quote observation was materialised, and no contract was
reselected, replaced, or forward-filled.

## Economic results

The strategy, monthly, stock, matched-control, veto, bootstrap, and concentration artifacts contain the frozen, untuned economics. All option entries/exits use the prescribed bid/ask sides.

| Strategy | Period | Status | Trades | Mean P&L | Mean return | Median return | Win rate | $1 mean P&L | $1 mean return |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | development | not_supported | 198 | $-18.54 | -12.38% | -16.87% | 15.15% | $-19.54 | -13.28% |
| S1 | assessment | not_supported | 196 | $-22.88 | -8.04% | -12.91% | 18.88% | $-23.88 | -8.77% |
| S2_ALL | development | blocked_direction_mapping_unavailable | NA | NA | NA | NA | NA | NA | NA |
| S2_ALL | assessment | blocked_direction_mapping_unavailable | NA | NA | NA | NA | NA | NA | NA |
| S2_VETO | development | blocked_direction_mapping_unavailable | NA | NA | NA | NA | NA | NA | NA |
| S2_VETO | assessment | blocked_direction_mapping_unavailable | NA | NA | NA | NA | NA | NA | NA |
| S3 | development | insufficient_support | 1 | $-39.00 | -66.67% | -66.67% | 0.00% | $-40.00 | -67.80% |
| S3 | assessment | insufficient_support | 0 | NA | NA | NA | NA | NA | NA |

The `$1` columns are the frozen descriptive commission sensitivity per contract,
per side, per leg. Blocked S2 rows contain no fabricated numerical results.

### Assessment months

| Strategy | Month | Trades | Mean return | Mean P&L |
|---|---|---:|---:|---:|
| S1 | 2025-01 | 11 | 21.58% | $32.64 |
| S1 | 2025-02 | 20 | -17.88% | $-90.90 |
| S1 | 2025-03 | 18 | 0.91% | $120.72 |
| S1 | 2025-04 | 17 | -15.00% | $-20.88 |
| S1 | 2025-05 | 26 | -2.89% | $-30.69 |
| S1 | 2025-06 | 32 | -7.96% | $-27.41 |
| S1 | 2025-07 | 38 | -11.07% | $-70.11 |
| S1 | 2025-08 | 34 | -13.73% | $-14.82 |

### Matched controls and concentration

| Strategy | Status | Matched coverage | Matched excess | Max stock | Max month | Top-5% positive P&L |
|---|---|---:|---:|---:|---:|---:|
| S1 | not_supported | 0.00% | NA | 12.76% | 19.39% | 90.55% |
| S2_ALL | blocked_direction_mapping_unavailable | NA | NA | NA | NA | NA |
| S2_VETO | blocked_direction_mapping_unavailable | NA | NA | NA | NA | NA |
| S3 | insufficient_support | 0.00% | NA | NA | NA | NA |

### Fixed-seed whole-session bootstrap

| Statistic | Interval | Lower | Upper | Draws |
|---|---:|---:|---:|---:|
| s1_mean_return_on_debit | 80.00% | -12.73% | -9.45% | 10 |
| s1_mean_return_on_debit | 90.00% | -12.88% | -9.28% | 10 |
| s1_mean_return_on_debit | 95.00% | -12.95% | -9.19% | 10 |
| s1_matched_control_excess | 80.00% | NA | NA | 10 |
| s1_matched_control_excess | 90.00% | NA | NA | 10 |
| s1_matched_control_excess | 95.00% | NA | NA | 10 |

Ten draws are a coarse stability diagnostic, not precise inference. `NA`
matched-control intervals reflect failed matching coverage, not zero excess.

### DTE and spread diagnostics

| Strategy | Diagnostic | Group | Trades | Mean return | Median return |
|---|---|---|---:|---:|---:|
| S1 | entry_dte_bin | 10-12 | 68 | -9.82% | -10.07% |
| S1 | entry_dte_bin | 13-14 | 8 | -15.24% | -14.27% |
| S1 | entry_dte_bin | 7-9 | 120 | -6.56% | -15.38% |
| S1 | spread_group | tight | 161 | -8.71% | -13.56% |
| S1 | spread_group | wide | 35 | -4.96% | -10.22% |

This report is a retrospective research screen. It does not claim intraday
option execution, IBKR fills, realised profits, prospective validation, or a
deployable strategy.

## Independent audit

Passed with no unexplained discrepancy.
