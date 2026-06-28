# Research Harness

Stage 3 turns Stocker into a disciplined research harness. It is not a live trading
system and it is not an optimizer for profit. Its job is to make weak ideas fail early
for clear reasons.

## Written Hypotheses

Every experiment starts with a YAML hypothesis. The file describes the market
universe, timeframe, signal family, entry and exit logic, holding period, direction,
cost assumptions, bounded parameter space, walk-forward method, expected edge reason,
minimum evidence, and invalidation rules.

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

## Honest Parameter Selection

Parameter selection is based on training-side evidence only. The runner prefers
settings with positive train net return, enough train-side trades, acceptable
train-side drawdown, and non-isolated train behavior when nearby settings are
available.

The highest test-return setting is still reported as `best_test_diagnostic`, but it
is diagnostic only. It does not drive classification. If no parameter set passes the
train-side gates, the runner picks a deterministic fallback only so reports remain
complete, and the experiment is rejected.

## Strategy Templates

The initial templates are deliberately simple:

- Moving-average momentum.
- Pullback in uptrend.
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

## Benchmarks And Null Timing

Single-symbol reports include a zero-position cash benchmark and a cost-aware
buy-and-hold benchmark over the same test windows. A positive strategy that fails to
beat buy-and-hold after costs is rejected with the `failed_benchmark` reason.

The runner also evaluates a tiny deterministic null timing set by circularly shifting
the selected target positions. This preserves broad exposure and trade structure but
disrupts exact timing. The default null set is intentionally small and deterministic,
not a brute-force random search. A selected result that fails to beat the null p75 is
rejected with the `failed_null_timing` reason.

## Leakage Checks

The runner wires leakage checks into each experiment report. It checks:

- Duplicate, missing, or non-monotonic timestamps.
- Train/test overlap and embargo violations.
- Empty or NaN-heavy generated signals and target positions.
- Suspiciously high correlation between generated signal or target position and
  next-bar returns.

Leakage errors reject through the conservative classification path. Warnings remain
visible in JSON and Markdown so they can be reviewed before promoting any result.

## Regime Analysis

Regime labels use only prior-bar information. Current labels include volatility,
trend, range, and drawdown regimes. The experiment report summarizes performance by
trend regime so a result that depends on one tiny period is easier to spot.

## Classifications

Research reports classify results as:

- `rejected_data_issue`
- `rejected_insufficient_data`
- `rejected_no_edge`
- `rejected_costs_kill_edge`
- `rejected_unstable_parameters`
- `rejected_walkforward_failure`
- `rejected_too_few_trades`
- `interesting_needs_more_tests`
- `candidate_paper_test`

`candidate_paper_test` is intentionally hard to reach. The test result must be
positive after costs, come from train-selected parameters, beat buy-and-hold, pass the
deterministic null timing gate, pass multiple walk-forward splits, show acceptable
parameter stability, have nearby settings that also work, include enough trades to
matter, avoid excessive drawdown, and not depend on one tiny regime.

Most results should still be rejected. This harness is still research only. It does
not paper trade, live trade, connect to brokers, run dashboards, train ML models, or
mine strategies automatically.

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
  --symbol AAPL.US \
  --timeframe 1d \
  --source eodhd \
  --config configs/research.example.yaml
```

Run the same written hypothesis across a qualified universe:

```bash
uv run stocker research run-universe \
  --hypothesis research/hypotheses/examples/moving_average_momentum.yaml \
  --qualified-universe data/universes/research_ready/us_test_5_1d.json \
  --config configs/research.example.yaml \
  --max-symbols 5
```

Reports are written to:

```text
data/reports/research/
data/reports/research/universe/
```

The directory includes one Markdown report, one JSON report, and updated index files
for comparing single-symbol experiments and universe-level runs.
