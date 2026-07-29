# Prior-close EODHD contract-history probe V0.1

This isolated retrieval-only amendment tests the contract-centric fix for the completed V0 date
blocker. It does not modify V0 artifacts or run the movement models.

The fixed sample uses AAL, MSTR, and WULF at the earliest, midpoint, and latest shared required
prior-close dates: 2024-01-16, 2024-10-31, and 2025-08-21. Contract identities are discovered from
`/mp/unicornbay/options/contracts`; each candidate history is then fetched from
`/mp/unicornbay/options/eod` with `filter[contract]`. The observation date comes only from the
resource-ID suffix and must agree with the New York dates of `bid_date` and `ask_date`.

## Result

The probe completed with 3,624 provider records and 1,029,673 bytes across 27 recorded HTTP
attempts. All nine stock-dates had exact-date call/put observations. Eight passed the frozen
primary pair-quality rule. The earliest WULF pair failed the call relative-spread gate and was
left unavailable without falling back to another expiry.

This supports the contract-history retrieval route, not the binding movement hypothesis. The V0
decision remains `blocked_historical_options_date_unavailable` until a separately versioned,
coverage-gated cohort rerun is completed. No intraday option fill, option P&L, executable return,
strategy result, prospective validation, or trading-utility claim is produced.

Raw responses and completion manifests are stored below the ignored
`data/vendor/eodhd/options/contract-history-probe-v01/` cache. Tracked artifacts contain only the
redacted request plan, aggregate manifest, selected-pair results, audit, determinism check, and
report.
