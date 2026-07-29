# Daily Stock × Options Regime Context Quick Screen V0

## Decision

Overall decision: `blocked_quick_resource_limit`.

The bounded recovery stopped before exceeding the frozen 500,000-record ceiling: 499,969 provider records across 2,193 completed requests and 154,465,109 raw bytes. The compact provider responses omitted the authoritative EOD observation identity required by the repaired exact-date filter, so zero records were canonicalised and no recovered cache was admitted. No second download was attempted.

## Frozen scope and reconstruction

- Development: 2024-01-01 through 2024-12-31.
- Assessment: 2025-01-01 through 2025-08-22.
- Frozen cohort: 20 stocks.
- Clean structural rows: 87,443
  (52,866 development and
  34,577 assessment).
- Assessment clean-completion positives: 451.
- Structural reconstruction: zero row, route-state, and target mismatches; maximum shared
  feature difference 0.0e+00.
- Protected market/options observations materialised:
  0/
  0.

## Daily stock context

- Raw stock-session rows: 7,903; complete dimension rows:
  7,519.
- Assessment support: 20 stocks,
  159 sessions,
  8 months; feature retention
  97.6%.
- Scaling and the four-component diagonal GMM were fitted on 2024 only. All four assessment
  regimes exceeded the 5% posterior-mass, eight-stock, four-month inference gates.

- Regime 0: posterior mass 0.147; hard support 445 rows, 20 stocks, 123 sessions, 8 months. Centroid: compression -0.569, volatility acceleration +0.283, directional efficiency +0.577, extension +1.288, rejection -0.421, relative strength +1.376.
- Regime 1: posterior mass 0.274; hard support 835 rows, 20 stocks, 155 sessions, 8 months. Centroid: compression -0.424, volatility acceleration +0.613, directional efficiency -0.304, extension +0.083, rejection +0.179, relative strength +0.006.
- Regime 2: posterior mass 0.123; hard support 370 rows, 20 stocks, 102 sessions, 8 months. Centroid: compression +0.008, volatility acceleration -0.121, directional efficiency +0.652, extension -0.896, rejection -0.486, relative strength -0.802.
- Regime 3: posterior mass 0.456; hard support 1422 rows, 20 stocks, 158 sessions, 8 months. Centroid: compression +0.323, volatility acceleration -0.375, directional efficiency -0.128, extension -0.033, rejection +0.157, relative strength -0.098.

## Previous-close options context and bounded recovery

- Repaired exact-date cache: 144,283
  rows across 2,551
  stock-dates; maximum cached DTE
  16.
- Cache records reused across requested stock-dates: 139,622.
- Valid front pairs: 1,250 development
  and 929 assessment stock-sessions.
- Front-pair assessment census: 10,422 clean
  checkpoint rows, 154 sessions,
  20 stocks, 8 months,
  2,159 BROAD_CONFLICT rows, and
  2,915 LOW_ROUTE_SUPPORT rows.
- Back-pair stock-sessions: 0.
- Exact gap manifest: 8,145 component rows;
  7,798 require bounded acquisition.
- Bounded plan: 7,556 exact
  stock-date requests. Status `blocked_quick_resource_limit`; network requests
  2193; new records
  499969; new bytes
  154465109.

## Downstream results

The daily options dimensions/regimes, joined cross-market panel, six mismatch distributions,
S0/S1/S2, O0/O1/O2, both Ridge diagnostics, monthly/checkpoint comparisons, persistence
horizons, regime-pair census, DTE-horizon mapping, ten session-bootstrap draws, three
options-null refits, three route-null refits, concentration analysis, and both plots were not
produced. This is a quick-resource blocker, not evidence for or against any increment.

## Component statuses

- `daily_stock_regime_status`: `supported`
- `daily_options_regime_status`: `blocked`
- `test_a_daily_stock_increment_status`: `blocked`
- `test_a_daily_options_increment_status`: `blocked`
- `test_b_daily_stock_increment_status`: `blocked`
- `test_b_intraday_route_increment_status`: `blocked`
- `mismatch_status`: `blocked`
- `persistence_horizon_status`: `blocked`

## Audit and reproducibility

- Independent fail-closed audit: `passed`.
- Stock posterior reconstruction: 100 rows, maximum difference
  1.55e-15.
- Determinism rebuild: not applicable after the terminal stop; recorded as blocked,
  with no redownload, bootstrap, or null repetition.

No result here establishes option profitability, intraday option fills, economic or
directional edge, prospective validation, trading utility, or a deployable strategy.
