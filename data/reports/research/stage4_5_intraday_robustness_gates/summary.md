# Stage 4.5 Intraday Robustness Gates

Stage 4.5 promotes the Stage 4.4 cost-stress and concentration diagnostics into
official intraday candidate gates. No new strategy templates, data fetches, ML,
broker, paper, live trading, or dashboard code were added.

## Tests

- focused robustness/classification/report tests: 19 passed
- full pytest suite: 174 passed
- ruff check .: passed
- git diff --check: passed before final summary generation

Status: `passed`

## Classification Counts

- Old classification counts: `{"candidate_intraday_test": 1, "rejected_costs_kill_edge": 6, "rejected_no_edge": 18}`
- New classification counts: `{"interesting_intraday_needs_more_tests": 1, "rejected_costs_kill_edge": 6, "rejected_no_edge": 18}`
- Old candidate count: 1
- New candidate count: 0
- Downgraded symbols: CRM

## CRM

- Old classification: `candidate_intraday_test`
- New classification: `interesting_intraday_needs_more_tests`
- Robustness failure reasons: `['failed_cost_stress', 'negative_median_trade', 'split_concentrated', 'trade_concentrated']`

## Cost Stress

```json
{
  "first_nonpositive_multiplier_counts": {
    "1.0": 20,
    "1.5": 4,
    "3.0": 1
  },
  "symbols_nonpositive_at_base_costs": 20,
  "symbols_nonpositive_by_1_5x_costs": 24,
  "symbols_surviving_1_5x_costs": 1
}
```

## Concentration

```json
{
  "median_profit_factor": 0.7216018599313033,
  "median_round_trip_trade_count": 91.0,
  "negative_median_trade_count": 22,
  "split_concentrated_count": 21,
  "top5_winner_concentrated_count": 5
}
```

## Recommendation

No robust opening-range candidate remains. Move to VWAP reclaim/rejection as the next hypothesis family, using the new robustness gates from the start.
