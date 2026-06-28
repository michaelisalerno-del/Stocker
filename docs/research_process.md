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
   instrument universe, timeframe, and failure condition.

4. Cost-adjusted vectorized backtest.
   Apply spread, commission, and slippage assumptions from the start. Ideas that only
   work before costs are rejected.

5. Walk-forward evaluation.
   Use chronological train/test splits. Do not tune on the whole history and call it
   evidence.

6. Regime split.
   Check whether results survive different volatility, trend, liquidity, and market
   session regimes.

7. Event-driven backtest.
   Only candidates that survive earlier filters deserve slower accounting, order, and
   event simulation.

8. Paper trading.
   Verify data freshness, state reconciliation, risk checks, and operational behavior
   without capital at risk.

9. Tiny live test.
   Only after paper evidence and operational safety exist, trade the smallest practical
   size.

10. Scale only after evidence.
   Increase size only when research, backtest, paper, and tiny-live evidence agree.
