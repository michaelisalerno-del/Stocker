# Research Process

The first research funnel is designed to reject bad ideas quickly.

1. Data audit.
   Confirm timestamps, duplicates, timezone handling, OHLC sanity, volume sanity,
   corporate actions, missing sessions, gaps, large price jumps, and vendor-specific
   quirks. Generate `data/reports/audits/<SYMBOL>_<TIMEFRAME>_audit.md` before using
   the dataset for research.

2. Baseline tests.
   Compare any idea against simple buy/hold, random, no-trade, and naive directional
   baselines before adding complexity. Stage 2 supports buy-and-hold, always-flat,
   random entry/exit, simple SMA momentum, and simple mean-reversion reports.

3. Simple statistical hypothesis.
   Write down the hypothesis before coding the strategy. Define the expected effect,
   instrument universe, timeframe, costs, parameter space, validation method, and
   invalidation rules. Store examples under `research/hypotheses/examples/`.

4. Cost-adjusted vectorized backtest.
   Apply spread, commission, and slippage assumptions from the start. Ideas that only
   work before costs are rejected. The Stage 3 vectorized evaluator reports gross
   return, net return, total costs, trades, exposure, drawdown, volatility, and a
   Sharpe-like metric.

5. Walk-forward evaluation.
   Use chronological train/test splits. Do not tune on the whole history and call it
   evidence. Do not use random train/test splits for trading data because future
   regimes would leak backward into model selection.

6. Train-side parameter selection.
   Choose parameters from training-side evidence only. The best test-return row is
   recorded as a diagnostic, not as the selected result. If no setting passes the
   train gates, the experiment is rejected even if one test window looks lucky.

7. Benchmark and null gates.
   Compare the selected result with cash and same-window buy-and-hold after costs.
   Then run a small deterministic circular-shift null timing set. A result that is
   positive but fails buy-and-hold or the null p75 is rejected.

8. Leakage checks.
   Check timestamp integrity, split overlap, embargo gaps, generated signal quality,
   and suspicious signal/next-return correlation. Leakage errors reject the result.

9. Regime split.
   Check whether results survive different volatility, trend, liquidity, and market
   session regimes. Regime labels must be based only on prior bars.

10. Event-driven backtest.
   Only candidates that survive earlier filters deserve slower accounting, order, and
   event simulation.

11. Paper trading.
   Verify data freshness, state reconciliation, risk checks, and operational behavior
   without capital at risk.

12. Tiny live test.
   Only after paper evidence and operational safety exist, trade the smallest practical
   size.

13. Scale only after evidence.
   Increase size only when research, backtest, paper, and tiny-live evidence agree.

## Stage 3 Experiment Flow

Run the current disciplined harness with:

```bash
uv run stocker data catalog
uv run stocker data audit --symbol AAPL --timeframe 1d
uv run stocker research baseline --symbol AAPL --timeframe 1d
uv run stocker research run \
  --hypothesis research/hypotheses/examples/moving_average_momentum.yaml \
  --symbol AAPL \
  --timeframe 1d
```

The research runner writes Markdown and JSON reports to `data/reports/research/` and
updates `index.md` plus `index.json`.

Classifications are intentionally conservative:

- `rejected_data_issue`
- `rejected_insufficient_data`
- `rejected_no_edge`
- `rejected_costs_kill_edge`
- `rejected_unstable_parameters`
- `rejected_walkforward_failure`
- `rejected_too_few_trades`
- `interesting_needs_more_tests`
- `candidate_paper_test`

Most ideas should be rejected. A paper-test candidate must use a train-selected
parameter set, survive costs, beat buy-and-hold, pass the deterministic null timing
gate, pass leakage checks, survive multiple walk-forward splits, show nearby
parameter support, include meaningful trade counts, keep tolerable drawdown, and work
across more than one tiny favorable period.

## Universe-Level Stage 3 Flow

Research should usually run against a qualified universe, not one manually selected
symbol:

```bash
uv run stocker universe qualify \
  --universe universes/manual/us_test_5.yaml \
  --timeframe 1d \
  --source eodhd \
  --min-history-days 10 \
  --min-last-close 5 \
  --min-median-dollar-volume-60d 1000000 \
  --output data/universes/research_ready/us_test_5_1d.json

uv run stocker research run-universe \
  --hypothesis research/hypotheses/examples/moving_average_momentum.yaml \
  --qualified-universe data/universes/research_ready/us_test_5_1d.json \
  --config configs/research.example.yaml \
  --max-symbols 5
```

The universe report aggregates symbol classifications, actual classification reasons,
benchmark and null pass counts, median net return, median excess versus benchmark,
median excess versus null, median drawdown, trade counts, stability scores, and links
to each symbol-level report. Failed symbols are recorded without stopping the full
universe run unless `--fail-fast` is set.

This process is still not paper trading or live trading. The server-side execution
boundary remains intentionally separate from Mac research.
