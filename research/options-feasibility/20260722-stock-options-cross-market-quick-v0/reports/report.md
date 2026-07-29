# Stock ↔ Options Cross-Market Information Quick Screen V0

Overall decision: `blocked_insufficient_cached_options_coverage`.

Research-only boundary: previous-close options only; no intraday option quotes,
option P&L, execution, broker integration, prospective validation, strategy
promotion, or production-runtime change.

The current canonical cache contains 3,624
vendor records. Only
18 exact-date history
observations were materialised;
24
cached observations at or beyond the protected boundary were skipped as raw JSON
text before row decoding. It reconstructs
9
exact previous-close chains and 8 valid ATM pairs
across 3 symbols and 3 signal
dates. Those pairs join 86 frozen clean structural rows, including
39 assessment rows from 1 session,
3 stocks, and 1 month
(2025-08).

The exact frozen structural reconstruction contains
87,443 eligible rows:
52,866 development and
34,577 assessment, with
451 assessment
Test A positives. Row-identity and route-state mismatches are zero and the
maximum shared-feature difference is
0.0e+00.
Protected rows materialised: 0.

Exact valid-pair assessment row coverage is
0.112792%, below the fixed 50% gate.
The joined assessment sample also misses the row, session, stock, month, Test A
positive, Test B positive, BROAD_CONFLICT, LOW_ROUTE_SUPPORT, and concentration
gates. Therefore S0/S1/S2, O0/O1/O2, R0/R1, the ten bootstrap draws, the three
options-null refits, and the three route-null refits were not run.

## Exact cache gap

- Missing exact-chain stock-dates: 8,028.
- Cached pair-quality failures with no permitted fallback:
  1.
- Estimated additional viable contract-discovery/history requests:
  56,196.
- Estimated additional provider records:
  6,920,136.
- New requests or downloads in this experiment: 0.

The row-level stock/date/month request manifest is `options_coverage_gap.csv`.

## Test A

Options-to-stock status: `insufficient_support`.
Disagreement status: `insufficient_support`.
No S0, S1, or S2 metrics, monthly/subgroup results, increments, bootstrap
intervals, or options-feature null comparisons exist. The
binding question—whether previous-close options information improves the stock
system's clean two-to-three-bar registered-loop completion forecast—remains
unanswered.

## Test B

Stock-to-options-movement status: `insufficient_support`.
Route-increment status: `insufficient_support`.
No O0, O1, O2, R0, or R1 metrics, increments, monthly results, bootstrap
intervals, or route-feature null comparisons exist. The binding
question—whether compressed-transition and route-competition features improve
prediction that 15-minute underlying movement exceeds the previous-close
options expectation—remains unanswered.

The below-cache-threshold route outcomes are descriptive diagnostics only:

- BROAD_CONFLICT: 8 rows; mean residual 0.00156141; exceed rate 20.0000%.
- LOW_ROUTE_SUPPORT: 6 rows; mean residual -0.00077095; exceed rate 33.3333%.
- BROAD_CONFLICT minus LOW_ROUTE_SUPPORT: mean residual 0.00233235; median residual -0.00001492; exceed-rate difference -13.3333%; upper-decile residual 0.03013086.
- BROAD_CONFLICT top-5%-row positive-residual contribution:
  97.2945%.
- LOW_ROUTE_SUPPORT top-5%-row positive-residual contribution:
  84.3303%.

## Concentration and reproducibility

Maximum assessment stock share of weighted joined rows:
33.3333%, above the 15% ceiling.

Determinism check: `passed`.
Selected-contract mismatches: 0.
Joined-row mismatches: 0.
Maximum feature difference:
0.0e+00.
Maximum probability difference:
0.0e+00.

Independent lightweight audit: `passed`.

No result is option profitability, an intraday option fill, executable option
return, economic edge, prospective validation, trading utility, or a deployable
strategy.
