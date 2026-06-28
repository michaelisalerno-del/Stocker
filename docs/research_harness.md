# Research Harness

Stage 3 turns Stocker into a disciplined research harness. It is not a live trading
system and it is not an optimizer for profit. Its job is to make weak ideas fail early
for clear reasons.

## Written Hypotheses

Every experiment starts with a YAML hypothesis. The file describes the market
universe, timeframe, signal family, entry and exit logic, holding period, direction,
cost assumptions, parameter space, validation method, expected edge reason, and
invalidation rules.

Example hypotheses live in:

```text
research/hypotheses/examples/
```

These examples are educational test vehicles. They do not claim edge.

## Walk-Forward Testing

Trading data is ordered in time. Random train/test splits are invalid because they let
future regimes influence past decisions. Stocker uses chronological walk-forward
splits:

- Rolling windows keep train and test sizes fixed.
- Expanding windows grow the training history over time.
- Fixed windows create one explicit train/test split.
- Embargo bars leave a gap between train and test to reduce leakage from adjacent bars.

The split engine records deterministic split IDs and boundaries, and tests enforce
that train rows always come before test rows.

## Parameter Stability

The harness generates small guarded parameter grids. A hard sweep limit prevents
accidental brute-force optimization. Results are scored for stability by comparing the
best setting with the rest of the parameter neighborhood.

The main question is:

```text
Does this only work at one magic setting, or across a stable region?
```

An isolated winner is treated as suspicious.

## Strategy Templates

The initial templates are deliberately simple:

- Moving-average momentum.
- Mean reversion after a large move.
- Volatility breakout.

Templates accept OHLCV data and parameters, then return target positions. They do not
place orders, connect to brokers, or know about execution state.

## Costs

Vectorized evaluation applies spread, commission, and slippage assumptions from the
hypothesis. Reports include gross return, net return, total costs, trade count,
exposure, win rate, profit factor, max drawdown, volatility, Sharpe-like metric,
equity curve, drawdown curve, and simplified trades.

Ideas that only work before costs should be rejected.

## Leakage Checks

The harness includes helpers for suspicious research mistakes:

- Same-bar close signals without execution lag.
- Target columns included in feature columns.
- Feature names suggesting future data.
- Train/test overlap.
- Feature timestamps after prediction timestamps.

These checks are intentionally conservative and should be expanded before adding more
complex modelling.

## Regime Analysis

Regime labels use only prior-bar information. Current labels include volatility,
trend, range, and drawdown regimes. The experiment report summarizes performance by
trend regime so a result that depends on one tiny period is easier to spot.

## Classifications

Research reports classify results as:

- `rejected_data_issue`
- `rejected_no_edge`
- `rejected_costs_kill_edge`
- `rejected_unstable_parameters`
- `rejected_walkforward_failure`
- `interesting_needs_more_tests`
- `candidate_paper_test`

`candidate_paper_test` is intentionally hard to reach. The test result must be
positive after costs, pass multiple walk-forward splits, show acceptable parameter
stability, have nearby settings that also work, include enough trades to matter, avoid
excessive drawdown, and not depend on one tiny regime.

## Commands

Import and audit data first:

```bash
uv run stocker data catalog
uv run stocker data audit --symbol AAPL --timeframe 1d
uv run stocker research baseline --symbol AAPL --timeframe 1d
```

Run a hypothesis:

```bash
uv run stocker research run \
  --hypothesis research/hypotheses/examples/moving_average_momentum.yaml \
  --symbol AAPL \
  --timeframe 1d
```

Reports are written to:

```text
data/reports/research/
```

The directory includes one Markdown report, one JSON report, and updated index files
for comparing experiments.
