# Daily Stock × Options Regime Context Quick Screen V0

## Decision

Overall decision: `blocked_insufficient_daily_options_coverage`.

The repaired previous-close cache passed the front-pair row gate but contained no 46–90 DTE
back-expiry observation in either period. Consequently `front_term_urgency` had zero finite
development values, its required development median did not exist, and the frozen eight-
dimension options surface and four-state options GMM could not be fitted without changing the
preregistered design. The run stopped before all cross-market model fitting.

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

- Raw stock-session rows: 7,903; complete dimension rows: 7,782.
- Assessment support: 20 stocks,
  160 sessions,
  8 months; feature retention
  100.0%.
- Scaling and the four-component diagonal GMM were fitted on 2024 only. All four assessment
  regimes exceeded the 5% posterior-mass, eight-stock, four-month inference gates.

- Regime 0: posterior mass 0.141; hard support 439 rows, 20 stocks, 126 sessions, 8 months. Centroid: compression -0.532, volatility acceleration +0.250, directional efficiency +0.572, extension +1.289, rejection -0.425, relative strength +1.404.
- Regime 1: posterior mass 0.265; hard support 821 rows, 20 stocks, 155 sessions, 8 months. Centroid: compression -0.462, volatility acceleration +0.678, directional efficiency -0.329, extension +0.079, rejection +0.211, relative strength +0.015.
- Regime 2: posterior mass 0.119; hard support 374 rows, 20 stocks, 105 sessions, 8 months. Centroid: compression +0.000, volatility acceleration -0.034, directional efficiency +0.640, extension -0.916, rejection -0.482, relative strength -0.830.
- Regime 3: posterior mass 0.474; hard support 1512 rows, 20 stocks, 159 sessions, 8 months. Centroid: compression +0.313, volatility acceleration -0.337, directional efficiency -0.102, extension -0.026, rejection +0.148, relative strength -0.087.

## Previous-close options context and bounded recovery

- Repaired exact-date cache: 144,283
  rows across 2,551
  stock-dates; maximum cached DTE
  16.
- Cache records reused across requested stock-dates: 141,351.
- Valid front pairs: 1,260 development
  and 941 assessment stock-sessions.
- Front-pair assessment census: 10,595 clean
  checkpoint rows, 155 sessions,
  20 stocks, 8 months,
  2,180 BROAD_CONFLICT rows, and
  2,943 LOW_ROUTE_SUPPORT rows.
- Back-pair stock-sessions: 0.
- Exact gap manifest: 8,147 component rows;
  7,822 require bounded acquisition.
- Bounded plan: 7,578 exact
  stock-date requests. Status `token_unavailable`; network requests
  0; new records
  0; new bytes
  0.

## Downstream results

The daily options dimensions/regimes, joined cross-market panel, six mismatch distributions,
S0/S1/S2, O0/O1/O2, both Ridge diagnostics, monthly/checkpoint comparisons, persistence
horizons, regime-pair census, DTE-horizon mapping, ten session-bootstrap draws, three
options-null refits, three route-null refits, concentration analysis, and both plots were not
produced. This is a coverage blocker, not evidence for or against any increment.

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
- Stock posterior reconstruction: 100 rows, maximum difference 1.11e-15.
- Determinism rebuild: not applicable after the options-coverage stop; recorded as blocked,
  with no redownload, bootstrap, or null repetition.

No result here establishes option profitability, intraday option fills, economic or
directional edge, prospective validation, trading utility, or a deployable strategy.
