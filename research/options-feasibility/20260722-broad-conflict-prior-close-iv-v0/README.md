# Broad-Conflict Excess Movement vs Prior-Close Options IV Screen V0

This bounded retrospective experiment asks whether frozen broad route conflict adds information
about **15-minute underlying movement** beyond the movement amount implied by the exact previous
US trading session's EODHD closing option chain. It follows the completed Broad-Conflict
Advance-Hazard Dense-Checkpoint Quick Screen V0.2 without modifying its frozen artifacts.

The options observations are end-of-day context only. The experiment does not reconstruct an
intraday option fill, calculate option P&L, test an executable return or straddle, promote an
options strategy, or establish prospective/trading utility.

## Current bounded result

The verified `/mp/unicornbay/options/eod` schema filters `tradetime`, which is last-trade
activity, but exposes no historical EOD observation-date filter. A live ten-record preflight for
2025-08-21 returned resource and quote observation dates from 2025-09-03 through 2025-09-16.
The experiment therefore stopped before bulk retrieval with
`blocked_historical_options_date_unavailable`. The two setup responses are content-addressed in
the ignored cache and independently rehashed; no options-movement inference was run.

## Frozen flow

1. `run_screen_v0.py --prepare` reconstructs the frozen clean-advance identities, derives exact
   prior sessions and unadjusted closes, and writes the bounded symbol-month request plan.
2. `download_options.py` verifies provider coverage and performs a ten-record historical
   preflight before any sequential, resumable cohort download. It reads only
   `EODHD_API_TOKEN`; data and pacing may be configured with `EODHD_OPTIONS_DATA_DIR` and
   `EODHD_OPTIONS_REQUESTS_PER_MINUTE`.
3. `build_options_panel.py` canonicalizes cached EOD observations, selects the frozen ATM pair,
   and builds the causal movement panel.
4. `run_screen_v0.py --analyse` fits only O0, O1, R0 and R1 after coverage gates pass.
5. `audit_screen_v0.py` independently checks the fail-closed output.

With no `EODHD_API_TOKEN`, preparation still writes the exact request plan and mocked downloader
tests, then records `blocked_missing_eodhd_api_token`. The public demo token is never substituted.

## Data locations

Tracked small artifacts are under `artifacts/primary/`. Raw and canonical vendor data belong under
`data/vendor/eodhd/options/{raw,canonical,manifests}` (or `EODHD_OPTIONS_DATA_DIR`) and are ignored
by Git. No credential is written to any artifact or log.
