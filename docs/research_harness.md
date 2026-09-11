# Research Harness

Stage 3 turns Stocker into a disciplined research harness. It is not a live trading
system and it is not an optimizer for profit. Its job is to make weak ideas fail early
for clear reasons.

## Written Hypotheses

Every experiment starts with a YAML hypothesis. The file describes the market
universe, timeframe, signal family, entry and exit logic, holding period, holding
policy, direction, cost assumptions, bounded parameter space, walk-forward method,
expected edge reason, minimum evidence, and invalidation rules.

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

## Indicator Context And Warmup

Walk-forward train and test windows may use historical rows before the scoring
window to warm up rolling indicators and preserve fixed-hold carryover positions
that were opened before `eval_start` but remain active inside the scored window. The
required lookback comes from the selected strategy template and parameter set. For
example, a moving-average template with a 200-bar slow window can use up to 201 prior
rows before a 125-bar test window.

The policy is strict:

- No future rows after the evaluation window are used.
- Context rows before the window are never scored.
- Returns, trades, drawdown, exposure, and costs are counted only inside the actual
  train or test rows.
- Each train/test window starts flat for accounting purposes.
- A first-bar target position created from historical context is allowed and pays
  normal entry costs.

This avoids falsely rejecting rolling-indicator hypotheses just because the isolated
test slice is shorter than the indicator warmup period. It does not make rejection
less common.

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

## Holding Policy

The default preference is intraday and session-flat. The written hypothesis carries a
`holding_policy` block that states whether overnight and weekend exposure are allowed,
how many sessions a position may be held, how close to the close new entries are
allowed, and what stricter evidence is required before swing ideas can be considered.

Daily-bar templates remain useful swing research vehicles for market context and
universe screening, but daily data does not prove a strategy can be traded flat by
the close. Daily multi-bar holds are marked as swing exposure, not preferred
intraday evidence. Intraday data can be checked for entries too close to the close,
failure to flatten before the close, overnight holds, and weekend holds.

Reports include:

- Session-flat compliance.
- Maximum holding bars and estimated holding sessions.
- Overnight and weekend exposure counts.
- Gap, overnight, weekend, and intraday return contribution where measurable.
- Holding-policy violations and warning reasons.

Weekend exposure is stricter than ordinary overnight exposure. Conditional swing
exposure requires exceptional evidence before it can become a candidate. Swing
candidates must clear higher thresholds for benchmark excess, null excess, trade
count, drawdown, gap dependence, and overnight/weekend exposure. A profitable result
can still be rejected if most gains come from overnight or weekend gaps.

## Benchmarks And Null Timing

Single-symbol reports include a zero-position cash benchmark and a cost-aware
buy-and-hold benchmark over the same walk-forward test windows used for the selected
result. Buy-and-hold is always a long market baseline, including for `short_only` and
`long_short` hypotheses; the hypothesis direction still controls strategy and null
position evaluation. A positive strategy that fails to beat buy-and-hold after costs
is rejected with the `failed_benchmark` reason.

The runner also evaluates a tiny deterministic null timing set by circularly shifting
the selected target positions inside the same walk-forward test windows and indicator
context policy used by the grid result. This preserves broad exposure and trade
structure but disrupts exact timing. The default null set is intentionally small and
deterministic, not a brute-force random search. A selected result that fails to beat
the null p75 is rejected with the `failed_null_timing` reason.

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
- `rejected_overnight_risk`
- `rejected_weekend_risk`
- `rejected_holding_policy_violation`
- `interesting_needs_more_tests`
- `interesting_intraday_needs_more_tests`
- `interesting_swing_needs_more_tests`
- `candidate_intraday_test`
- `candidate_swing_exceptional`
- `candidate_paper_test`

Candidate classifications are intentionally hard to reach. An intraday candidate must
be positive after costs, come from train-selected parameters, beat buy-and-hold, pass
the deterministic null timing gate, pass multiple walk-forward splits, show acceptable
parameter stability, include enough trades to matter, avoid excessive drawdown, and
remain session-flat. A swing candidate must pass all normal gates and the stricter
exceptional swing thresholds; otherwise it remains interesting or rejected for holding
risk.

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

Run a bounded local real-data research smoke:

```bash
bash scripts/research_smoke_local.sh
```

Without `EODHD_API_TOKEN`, the script skips live fetch and explains whether existing
local data under `data_smoke_research/` is available. With a token, it fetches the
committed `universes/manual/us_test_5.yaml` sample, qualifies a tiny research-ready
universe, runs `moving_average_momentum.yaml` across at most five symbols, and prints
the qualified universe path plus the universe Markdown and JSON report paths.
Rejected results are expected and do not mean the smoke failed.

Reports are written to:

```text
data/reports/research/
data/reports/research/universe/
```

The directory includes one Markdown report, one JSON report, and updated index files
for comparing single-symbol experiments and universe-level runs.


## Historical README walkthroughs (preserved 2026-09-11)

The following material was moved from README at reviewed baseline fcb2c73.
It records earlier research stages and their original claims, not a current release
verification. Research data sources must never feed operational IBKR PRE history.
Use the current README for installation and check commands. Paths in these command
examples are relative to the repository root.

## Bootstrap On Mac

```bash
bash scripts/bootstrap_mac.sh
```

The script checks for Homebrew, `uv`, and Python. It does not silently install global
software. If `uv` is missing, it prints the install command and exits.

After bootstrap:

```bash
uv run stocker check
uv run stocker data import-csv \
  --file tests/fixtures/market_data/clean_ohlcv.csv \
  --symbol AAPL \
  --source manual \
  --timeframe 1d \
  --instrument-type stock \
  --timezone America/New_York \
  --currency USD
uv run stocker data audit --symbol AAPL --timeframe 1d
uv run stocker research baseline --symbol AAPL --timeframe 1d
uv run stocker research run \
  --hypothesis research/hypotheses/examples/moving_average_momentum.yaml \
  --symbol AAPL \
  --timeframe 1d
uv run pytest
uv run jupyter lab apps/desktop/notebooks
```

## Stage 2 IBKR Diagnostic

Stage 2 uses the maintained `ib_async` client through one read-only Stocker adapter. Start and
authenticate IB Gateway or TWS yourself, enable its socket API, review
`configs/ibkr.example.yaml`, then run:

```bash
uv run stocker ibkr-check \
  --run-config configs/run.example.yaml \
  --ibkr-config configs/ibkr.example.yaml \
  --symbol AAPL \
  --primary-exchange NASDAQ
```

The command masks the connected account in normal output and always disconnects. IB Gateway/TWS
handles authentication; Stocker does not store or automate broker login credentials. If a session
exposes multiple accounts, copy the example to the ignored `configs/ibkr.local.yaml` and set
`expected_account` there so Stocker can select the intended account explicitly.

## Data Pipeline

The data pipeline supports local CSV import and EODHD vendor ingestion. Both paths
normalize to the same Stocker OHLCV schema, validate the result, write partitioned
Parquet under
`data/processed/source=.../instrument_type=.../symbol=.../timeframe=.../data.parquet`,
update `data/catalog.json`, and can generate audit and baseline reports.

EODHD fetch commands load `configs/research.example.yaml` by default. The config
controls the data directory, default currency, vendor base URL, token environment
variable, retry count, timeout, and default raw-response saving behavior. Dry-runs do
not require a token. Live fetches require `data_vendors.eodhd.enabled: true` unless
you explicitly pass `--enable-disabled-vendor`.

EODHD credentials are read only from the configured environment variable, normally
`EODHD_API_TOKEN`:

```bash
export EODHD_API_TOKEN="your_token_here"
uv run stocker data fetch-eodhd-eod \
  --config configs/research.example.yaml \
  --symbol AAPL.US \
  --from 2015-01-01 \
  --to 2026-06-28 \
  --period d \
  --instrument-type stock \
  --merge \
  --save-raw \
  --audit
```

See [docs/data_pipeline.md](data_pipeline.md) before researching any edge.
See [docs/vendors/eodhd.md](vendors/eodhd.md) for EODHD-specific commands.
See [docs/universes.md](universes.md) for universe generation, batch fetch, and
research-ready exports.

Local EODHD smoke test:

```bash
bash scripts/smoke_eodhd_local.sh
```

Without `EODHD_API_TOKEN`, the script runs safe dry-runs and skips the live fetch.
With a token, it fetches a tiny EOD sample into `data_smoke/`, then runs catalog,
audit, vendor QA, and baseline checks.

Local Stage 3 research smoke:

```bash
bash scripts/research_smoke_local.sh
```

Without `EODHD_API_TOKEN`, the script skips live fetch and explains how to provide
existing local data. With a token, it fetches a bounded `us_test_5` sample, qualifies
a tiny research-ready universe, runs one moving-average momentum universe report, and
prints the qualified universe path plus Markdown/JSON report paths. Rejections are
expected and are not smoke failures.

## Stage 3 Research Readiness

- `bash scripts/check.sh` passes locally.
- GitHub Actions passes.
- EODHD dry-runs work without a token.
- One small real EODHD EOD fetch works with `EODHD_API_TOKEN`.
- `uv run stocker data catalog` sees the fetched dataset.
- Audit and EODHD QA reports are generated.
- Baseline reports can consume the EODHD dataset.
- Re-running the fetch with `--merge` does not duplicate rows.
- A research-ready universe export exists for the symbols under test.

## Research Harness

Stage 3 adds a hypothesis-first research harness. Each experiment starts from a YAML
hypothesis, loads an audited local Parquet dataset, builds chronological walk-forward
splits, evaluates a small guarded parameter grid, warms rolling indicators with
pre-window historical context while scoring only actual train/test rows, applies
explicit costs, checks train-side parameter selection, keeps the best test-return row
as diagnostic only, runs leakage checks, compares against cash and same-window long
buy-and-hold, applies a same-window deterministic null timing test, separates
preferred intraday/session-flat evidence from stricter swing evidence, reports
overnight/weekend/gap contribution where measurable, summarizes performance by simple
historical regimes, and writes Markdown/JSON reports under `data/reports/research/`.

The initial templates are deliberately basic: moving-average momentum, pullback in
uptrend, mean reversion after a large down day, and volatility breakout. They are test
vehicles for the harness, not claims of edge.

Most research results should still be rejected. This stage is not paper trading, live
trading, broker integration, dashboard work, ML, or automatic strategy mining.
Daily-bar results are useful for context and universe research, but they do not prove
session-flat tradability; swing candidates need exceptional evidence.

Example:

```bash
uv run stocker research run \
  --hypothesis research/hypotheses/examples/moving_average_momentum.yaml \
  --symbol AAPL.US \
  --timeframe 1d \
  --source eodhd \
  --config configs/research.example.yaml
```

Universe example:

```bash
uv run stocker research run-universe \
  --hypothesis research/hypotheses/examples/moving_average_momentum.yaml \
  --qualified-universe data/universes/research_ready/us_test_5_1d.json \
  --config configs/research.example.yaml \
  --max-symbols 5
```

See [docs/research_harness.md](research_harness.md) for the full workflow and
classification rules.

## Universe Workflow

Stage 2.7 adds a stock-universe data manager so research starts from a reproducible
symbol set instead of ad hoc tickers:

```bash
uv run stocker universe validate --universe universes/manual/us_test_5.yaml
uv run stocker universe fetch \
  --universe universes/manual/us_test_5.yaml \
  --from 2024-01-01 \
  --to 2024-02-01 \
  --timeframe 1d \
  --source eodhd \
  --dry-run \
  --max-symbols 2
```

The intended research flow is EODHD screener or manual YAML, batch EODHD fetch, audit/QA for
each dataset, local liquidity and history qualification, then a research-ready JSON
export consumed by `stocker research run-universe`. FMP work remains deferred; the operational
dashboard is available as described below.
