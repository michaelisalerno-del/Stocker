# Prior-Close Options IV Movement Screen V0 — blocked pre-download

Primary decision: `blocked_missing_eodhd_api_token`

The frozen V0.2 clean-advance population reconstructed exactly (87,443 rows;
34,577 assessment rows). The request plan contains 400 symbol-month chunks for
20 frozen symbols and 402 exact prior trading sessions
from 2024-01-16 through 2025-08-21. It estimates
2,141 total requests (2 setup and 2,139
EOD pages) and 2,009,250 raw records, totalling approximately
1,406,475,000 bytes within the frozen resource caps. 3 symbol-date joins cross an
inferred unadjusted-price corporate-action boundary and are preregistered as unavailable.

`EODHD_API_TOKEN` was not present. No provider preflight, mapping call, cohort download, option-pair
selection, structural join, underlying movement inference, model fit, bootstrap draw, route-null
refit, or plot was performed. The public demo token was not substituted.

This is a retrospective research-only feasibility screen. It contains no intraday option fill,
option P&L, executable return, profitable-straddle claim, directional edge, prospective validation,
trading utility, or deployable strategy.
