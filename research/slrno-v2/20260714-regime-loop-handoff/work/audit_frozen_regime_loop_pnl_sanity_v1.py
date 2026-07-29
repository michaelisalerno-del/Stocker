#!/usr/bin/env python3
"""Independent audit of the offline regime/loop P&L sanity test."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


WORK = Path(__file__).resolve().parent
CONTRACT = WORK / "contracts/20260712-frozen-regime-loop-pnl-sanity-v1.json"
PRE_SCORE = WORK / "contracts/20260712-frozen-regime-loop-pnl-sanity-v1-pre-score.json"
RUNNER = WORK / "run_frozen_regime_loop_pnl_sanity_v1.py"
PRICE_ROOT = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710")
PREDICTIONS = {
    2025: PRICE_ROOT / "price_predictions_2025.parquet",
    2023: PRICE_ROOT / "price_predictions_2023.parquet",
}
TRAIN_PANEL = PRICE_ROOT / "anchor_panel_train_2024.parquet"
PARAMETERS = PRICE_ROOT / "outcome_model_parameters.npz"
FEATURE_MANIFEST = PRICE_ROOT / "feature_manifest.json"
RAW_ROOTS = {
    2025: Path("/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock"),
    2023: Path("/private/tmp/stocker_eodhd_pre2024_intraday_20260710/source=eodhd/instrument_type=stock"),
}
ROOT = Path("/private/tmp/stocker_frozen_regime_loop_pnl_sanity_v1_20260712")

SEED = 20260712
HORIZONS = (6, 12, 24)
REPRESENTATIONS = ("state_context", "raw_history", "loop_scores")
COSTS = (0, 1, 2, 5, 10)
PRIMARY_COST = 5
UNIVERSE_SIZE = 20
SESSION_BARS = 78
NUMERIC = (
    "b0_entry_numeric",
    "b0_entry_high_stress",
    "entry_time_sin",
    "entry_time_cos",
    "current_bar_log_return",
    "return_sum_6",
    "mean_abs_return_12",
    "session_return",
    "bar_range_pct",
)
LOOPS = tuple(f"loop_score_{index:02d}" for index in range(1, 21))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def source_paths(symbols: list[str]) -> dict[str, Path]:
    values = {
        "contract": CONTRACT,
        "runner": RUNNER,
        "predictions_2025": PREDICTIONS[2025],
        "predictions_2023": PREDICTIONS[2023],
        "threshold_panel_2024": TRAIN_PANEL,
        "model_parameters": PARAMETERS,
        "feature_manifest": FEATURE_MANIFEST,
    }
    for year in (2025, 2023):
        for symbol in symbols:
            values[f"provider_{year}_{symbol}"] = provider_path(RAW_ROOTS[year], symbol)
    return values


def one_hot(values: np.ndarray, width: int) -> sparse.csr_matrix:
    integer = np.asarray(values, dtype=int)
    return sparse.csr_matrix(
        (
            np.ones(len(integer), dtype=np.float32),
            (np.arange(len(integer)), integer),
        ),
        shape=(len(integer), width),
    )


def reconstruct_thresholds() -> pd.DataFrame:
    columns = {
        "state",
        "history_token",
        "bar_index_in_session",
        *(f"exact_{horizon}" for horizon in HORIZONS),
        *NUMERIC,
        *LOOPS,
    }
    panel = pd.read_parquet(TRAIN_PANEL, columns=sorted(columns))
    if panel["bar_index_in_session"].astype(int).gt(53).any() or not all(
        panel[f"exact_{horizon}"].astype(bool).all() for horizon in HORIZONS
    ):
        raise AssertionError("threshold cohort mismatch")
    manifest = json.loads(FEATURE_MANIFEST.read_text())
    medians = pd.Series(manifest["numeric_medians"]).reindex(NUMERIC)
    numeric = (
        panel.loc[:, NUMERIC]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(medians)
        .to_numpy(np.float32)
    )
    context = sparse.hstack(
        (one_hot(panel["state"].to_numpy(int), 8), sparse.csr_matrix(numeric)),
        format="csr",
    )
    raw_features = {
        "state_context": context,
        "raw_history": sparse.hstack(
            (context, one_hot(panel["history_token"].to_numpy(int), 648)),
            format="csr",
        ),
        "loop_scores": sparse.hstack(
            (context, sparse.csr_matrix(panel.loc[:, LOOPS].to_numpy(np.float32))),
            format="csr",
        ),
    }
    parameters = np.load(PARAMETERS)
    rows: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        scaled = raw_features[representation].multiply(
            1.0 / parameters[f"{representation}__scaler_scale"]
        )
        for horizon in HORIZONS:
            coefficient = parameters[
                f"{representation}__future_range_bps__h{horizon}__coef"
            ]
            intercept = parameters[
                f"{representation}__future_range_bps__h{horizon}__intercept"
            ][0]
            prediction = np.asarray(scaled @ coefficient).ravel() + intercept
            rows.append(
                {
                    "representation": representation,
                    "horizon": horizon,
                    "training_rows": len(panel),
                    "prediction_mean_bps": prediction.mean(),
                    "prediction_p75_bps": np.quantile(prediction, 0.75, method="linear"),
                    "prediction_p90_bps": np.quantile(prediction, 0.90, method="linear"),
                }
            )
    return pd.DataFrame(rows)


def load_predictions(year: int, symbols: list[str]) -> pd.DataFrame:
    columns = [
        "anchor_id", "symbol_norm", "session_date", "quarter", "start_timestamp",
        "state", "history_token",
    ]
    for horizon in HORIZONS:
        columns.append(f"signed_return_bps_target_{horizon}")
        for representation in REPRESENTATIONS:
            columns.extend(
                [
                    f"{representation}__direction_probability_{horizon}",
                    f"{representation}__future_range_bps_prediction_{horizon}",
                ]
            )
    frame = pd.read_parquet(PREDICTIONS[year], columns=columns)
    frame = frame.loc[frame["symbol_norm"].isin(symbols)].copy()
    frame["start_timestamp"] = pd.to_datetime(frame["start_timestamp"], utc=True)
    return frame.sort_values(
        ["symbol_norm", "session_date", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)


def load_tape(year: int, symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        frame = pd.read_parquet(
            provider_path(RAW_ROOTS[year], symbol),
            columns=["timestamp", "open", "high", "low", "close"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.loc[
            frame["timestamp"].ge(pd.Timestamp(f"{year}-01-01", tz="UTC"))
            & frame["timestamp"].lt(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        ].dropna()
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        minutes = local.dt.hour * 60 + local.dt.minute
        frame = frame.loc[minutes.ge(570) & minutes.lt(960)].copy()
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        frame["session_date"] = local.dt.strftime("%Y-%m-%d")
        frame["symbol_norm"] = symbol
        frame["bar_ordinal"] = frame.groupby("session_date", sort=False).cumcount()
        frames.append(frame)
    tape = pd.concat(frames, ignore_index=True).sort_values(
        ["symbol_norm", "session_date", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    tape["tape_position"] = np.arange(len(tape))
    return tape


def independent_breakout(
    tape: pd.DataFrame,
    positions: np.ndarray,
    horizon: int,
    upper: np.ndarray,
    lower: np.ndarray,
    exit_price: np.ndarray,
) -> dict[str, np.ndarray]:
    opens = tape["open"].to_numpy(float)
    highs = tape["high"].to_numpy(float)
    lows = tape["low"].to_numpy(float)
    status = np.full(len(positions), "no_trigger", dtype=object)
    direction = np.zeros(len(positions), dtype=np.int8)
    entry_step = np.zeros(len(positions), dtype=int)
    entry_price = np.full(len(positions), np.nan)
    for row, position in enumerate(positions):
        for step in range(1, horizon + 1):
            index = int(position + step)
            open_value = opens[index]
            if open_value >= upper[row] and open_value <= lower[row]:
                status[row] = "ambiguous_same_bar"
                break
            if open_value >= upper[row]:
                status[row] = "filled"
                direction[row] = 1
                entry_step[row] = step
                entry_price[row] = max(upper[row], open_value)
                break
            if open_value <= lower[row]:
                status[row] = "filled"
                direction[row] = -1
                entry_step[row] = step
                entry_price[row] = min(lower[row], open_value)
                break
            up = highs[index] >= upper[row]
            down = lows[index] <= lower[row]
            if up and down:
                status[row] = "ambiguous_same_bar"
                break
            if up:
                status[row] = "filled"
                direction[row] = 1
                entry_step[row] = step
                entry_price[row] = upper[row]
                break
            if down:
                status[row] = "filled"
                direction[row] = -1
                entry_step[row] = step
                entry_price[row] = lower[row]
                break
    filled = status == "filled"
    gross = np.full(len(positions), np.nan)
    gross[filled] = direction[filled] * (
        exit_price[filled] / entry_price[filled] - 1.0
    )
    return {
        "status": status,
        "direction": direction,
        "entry_step": entry_step,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": gross,
        "gross_return_bps": 10000.0 * gross,
        "holding_bars": np.where(filled, horizon - entry_step + 1, 0),
    }


def verify_execution(
    year: int,
    predictions: pd.DataFrame,
    tape: pd.DataFrame,
    observed: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    source = predictions.merge(observed[["anchor_id"]], on="anchor_id", validate="one_to_one")
    lookup = tape.rename(columns={"timestamp": "start_timestamp"})[
        ["symbol_norm", "start_timestamp", "session_date", "bar_ordinal", "tape_position", "open", "high", "low", "close"]
    ]
    frame = source.merge(
        lookup,
        on=["symbol_norm", "start_timestamp"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_tape"),
    ).sort_values(["symbol_norm", "session_date", "start_timestamp"], kind="stable").reset_index(drop=True)
    observed = observed.sort_values(
        ["symbol_norm", "session_date", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)
    if not frame["anchor_id"].equals(observed["anchor_id"]):
        raise AssertionError("execution anchor alignment mismatch")
    maximum = 0.0

    def close(left: Any, right: Any, name: str, atol: float = 1e-10) -> None:
        nonlocal maximum
        a = np.asarray(left, float)
        b = np.asarray(right, float)
        if not np.allclose(a, b, rtol=1e-10, atol=atol, equal_nan=True):
            raise AssertionError(f"{name} mismatch")
        finite = np.abs(a - b)[np.isfinite(a - b)]
        maximum = max(maximum, float(finite.max(initial=0.0)))

    for source_column, observed_column in (
        ("bar_ordinal", "bar_ordinal"),
        ("tape_position", "tape_position"),
        ("open", "anchor_open"),
        ("high", "anchor_high"),
        ("low", "anchor_low"),
        ("close", "anchor_close"),
    ):
        close(frame[source_column], observed[observed_column], f"{year} {observed_column}")
    positions = frame["tape_position"].to_numpy(int)
    for horizon in HORIZONS:
        future = positions + horizon
        next_open = tape["open"].to_numpy(float)[positions + 1]
        exit_close = tape["close"].to_numpy(float)[future]
        close(next_open, observed[f"next_open_{horizon}"], f"{year} next open h{horizon}")
        close(exit_close, observed[f"exit_close_{horizon}"], f"{year} exit h{horizon}")
        close(
            10000.0 * np.log(exit_close / frame["close"].to_numpy(float)),
            frame[f"signed_return_bps_target_{horizon}"],
            f"{year} frozen outcome h{horizon}",
            atol=1e-7,
        )
        breakout = independent_breakout(
            tape,
            positions,
            horizon,
            frame["high"].to_numpy(float),
            frame["low"].to_numpy(float),
            exit_close,
        )
        for name, values in breakout.items():
            column = f"breakout_{name}_{horizon}"
            if name == "status":
                if not np.array_equal(values.astype(str), observed[column].astype(str)):
                    raise AssertionError(f"{year} {column} mismatch")
            else:
                close(values, observed[column], f"{year} {column}", atol=1e-8)
    return frame, maximum


def strategies() -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        values.extend(
            [
                dict(strategy=f"direction_{representation}_all", family="directional_falsification", representation=representation, gated=False),
                dict(strategy=f"direction_{representation}_range_p75", family="directional_falsification", representation=representation, gated=True),
            ]
        )
    values.append(dict(strategy="breakout_all", family="causal_breakout", representation="none", gated=False))
    for representation in REPRESENTATIONS:
        values.append(dict(strategy=f"breakout_{representation}_range_p75", family="causal_breakout", representation=representation, gated=True))
    return values


def greedy(frame: pd.DataFrame, eligible: np.ndarray, horizon: int) -> list[int]:
    accepted: list[int] = []
    candidates = frame.loc[eligible, ["symbol_norm", "session_date", "bar_ordinal"]]
    for _, group in candidates.groupby(["symbol_norm", "session_date"], sort=False):
        blocked_until = -1
        for position, ordinal in zip(group.index, group["bar_ordinal"], strict=True):
            if int(ordinal) >= blocked_until:
                accepted.append(int(position))
                blocked_until = int(ordinal) + horizon
    return accepted


def verify_ledger(
    year: int,
    source: pd.DataFrame,
    execution: pd.DataFrame,
    thresholds: pd.DataFrame,
    ledger: pd.DataFrame,
) -> int:
    frame = source.merge(execution, on="anchor_id", validate="one_to_one", suffixes=("", "_execution"))
    frame = frame.sort_values(["symbol_norm", "session_date", "start_timestamp"], kind="stable").reset_index(drop=True)
    threshold = {(row.representation, int(row.horizon)): float(row.prediction_p75_bps) for row in thresholds.itertuples(index=False)}
    checked = 0
    for horizon in HORIZONS:
        for definition in strategies():
            representation = definition["representation"]
            if definition["gated"]:
                gate = frame[f"{representation}__future_range_bps_prediction_{horizon}"].to_numpy(float)
                cutoff = threshold[(representation, horizon)]
                eligible = gate >= cutoff
            else:
                gate = np.full(len(frame), np.nan)
                cutoff = math.nan
                eligible = np.ones(len(frame), dtype=bool)
            positions = greedy(frame, eligible, horizon)
            expected_ids = frame.loc[positions, "anchor_id"].to_numpy(int)
            selected = ledger.loc[
                ledger["period"].astype(str).eq(str(year))
                & ledger["strategy"].eq(definition["strategy"])
                & ledger["horizon"].eq(horizon)
            ].sort_values(["symbol_norm", "session_date", "start_timestamp"], kind="stable")
            if not np.array_equal(expected_ids, selected["anchor_id"].to_numpy(int)):
                raise AssertionError(f"accepted-signal mismatch {year} {definition['strategy']} h{horizon}")
            if definition["gated"]:
                if not np.allclose(selected["gate_threshold_bps"], cutoff) or not np.allclose(selected["gate_value_bps"], gate[positions]):
                    raise AssertionError("gate value mismatch")
            if definition["family"] == "directional_falsification":
                probability = frame.loc[positions, f"{representation}__direction_probability_{horizon}"].to_numpy(float)
                direction = np.where(probability >= 0.5, 1, -1)
                entry = frame.loc[positions, f"next_open_{horizon}"].to_numpy(float)
                exit_price = frame.loc[positions, f"exit_close_{horizon}"].to_numpy(float)
                gross = direction * (exit_price / entry - 1.0)
                if not np.array_equal(direction, selected["direction"].to_numpy(int)) or not np.allclose(gross, selected["gross_return"], atol=1e-12):
                    raise AssertionError("directional fill mismatch")
            else:
                for name in ("status", "direction", "entry_step", "entry_price", "exit_price", "gross_return", "gross_return_bps", "holding_bars"):
                    expected = frame.loc[positions, f"breakout_{name}_{horizon}"]
                    observed = selected[name]
                    if name == "status":
                        if not np.array_equal(expected.astype(str), observed.astype(str)):
                            raise AssertionError("breakout ledger status mismatch")
                    elif not np.allclose(expected, observed, rtol=1e-10, atol=1e-9, equal_nan=True):
                        raise AssertionError(f"breakout ledger {name} mismatch")
            checked += len(selected)
    return checked


def stats(values: np.ndarray) -> dict[str, float]:
    daily = np.asarray(values, float)
    equity = np.cumprod(1.0 + daily)
    cumulative = equity[-1] - 1.0
    annualized = (1.0 + cumulative) ** (252.0 / len(daily)) - 1.0
    volatility = np.std(daily, ddof=1) * np.sqrt(252.0)
    sharpe = np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252.0) if np.std(daily, ddof=1) > 0 else np.nan
    path = np.r_[1.0, equity]
    drawdown = path / np.maximum.accumulate(path) - 1.0
    return dict(
        cumulative_return=cumulative,
        annualized_return=annualized,
        annualized_volatility=volatility,
        descriptive_sharpe_zero_rate=sharpe,
        maximum_drawdown=drawdown.min(initial=0.0),
        mean_daily_return=daily.mean(),
    )


def daily_for(
    signals: pd.DataFrame,
    sessions: list[str],
    cost: int,
    deleted: str | None = None,
) -> tuple[pd.Series, np.ndarray]:
    trades = signals.loc[signals["status"].eq("filled")].copy()
    divisor = UNIVERSE_SIZE
    if deleted is not None:
        trades = trades.loc[trades["symbol_norm"].ne(deleted)].copy()
        divisor -= 1
    net = trades["gross_return"].to_numpy(float) - 2.0 * cost / 10000.0
    if (net <= -1.0).any():
        raise AssertionError("invalid net sleeve return")
    trades["log_growth"] = np.log1p(net)
    sleeve = np.expm1(
        trades.groupby(["session_date", "symbol_norm"], sort=False)["log_growth"].sum()
    )
    daily = (sleeve.groupby("session_date").sum() / divisor).reindex(sessions, fill_value=0.0)
    return daily, 10000.0 * net


def recompute_metrics(
    ledger: pd.DataFrame, sessions_by_period: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    quarters: list[dict[str, Any]] = []
    deletions: list[dict[str, Any]] = []
    group_columns = ["period", "strategy", "family", "representation", "horizon"]
    for keys, signals in ledger.groupby(group_columns, sort=False):
        period, strategy, family, representation, horizon = keys
        sessions = sessions_by_period[str(period)]
        symbols = sorted(ledger.loc[ledger["period"].astype(str).eq(str(period)), "symbol_norm"].unique())
        for cost in COSTS:
            daily, net_bps = daily_for(signals, sessions, cost)
            filled = signals["status"].eq("filled")
            positive = net_bps[net_bps > 0]
            negative = net_bps[net_bps < 0]
            profit_factor = positive.sum() / -negative.sum() if len(negative) else (np.inf if len(positive) else np.nan)
            metrics.append(
                dict(
                    period=period,
                    strategy=strategy,
                    family=family,
                    representation=representation,
                    horizon=int(horizon),
                    cost_bps_per_side=cost,
                    accepted_signals=len(signals),
                    filled_trades=int(filled.sum()),
                    no_trigger_signals=int(signals["status"].eq("no_trigger").sum()),
                    ambiguous_signals=int(signals["status"].eq("ambiguous_same_bar").sum()),
                    fill_rate=filled.mean(),
                    mean_net_trade_bps=net_bps.mean() if len(net_bps) else np.nan,
                    median_net_trade_bps=np.median(net_bps) if len(net_bps) else np.nan,
                    win_rate=(net_bps > 0).mean() if len(net_bps) else np.nan,
                    profit_factor=profit_factor,
                    exposure_fraction=signals.loc[filled, "holding_bars"].sum() / (SESSION_BARS * UNIVERSE_SIZE * len(sessions)),
                    **stats(daily.to_numpy(float)),
                )
            )
            daily_frame = pd.DataFrame(
                dict(period=period, strategy=strategy, family=family, representation=representation, horizon=int(horizon), cost_bps_per_side=cost, session_date=daily.index.astype(str), daily_return=daily.to_numpy(float))
            )
            daily_frames.append(daily_frame)
            dates = pd.to_datetime(daily_frame["session_date"])
            daily_frame["quarter"] = dates.dt.year.astype(str) + "_q" + dates.dt.quarter.astype(str)
            for quarter, selected in daily_frame.groupby("quarter", sort=True):
                quarters.append(dict(period=period, strategy=strategy, horizon=int(horizon), cost_bps_per_side=cost, quarter=quarter, session_dates=len(selected), **stats(selected["daily_return"].to_numpy(float))))
            for symbol in symbols:
                deleted_daily, _ = daily_for(signals, sessions, cost, symbol)
                deletions.append(dict(period=period, strategy=strategy, horizon=int(horizon), cost_bps_per_side=cost, deleted_symbol=symbol, session_dates=len(deleted_daily), **stats(deleted_daily.to_numpy(float))))
    return pd.DataFrame(metrics), pd.concat(daily_frames, ignore_index=True), pd.DataFrame(quarters), pd.DataFrame(deletions)


def block_interval(values: np.ndarray, offset: int) -> tuple[float, float, float]:
    data = np.asarray(values, float)
    rng = np.random.default_rng(SEED + offset)
    starts = np.arange(len(data) - 5 + 1)
    count = math.ceil(len(data) / 5)
    chosen = rng.choice(starts, size=(5000, count), replace=True)
    positions = (chosen[:, :, None] + np.arange(5)[None, None, :]).reshape(5000, -1)[:, : len(data)]
    sample = data[positions].mean(axis=1)
    low, high = np.quantile(sample, [0.025, 0.975], method="linear")
    return data.mean(), low, high


def recompute_bootstraps(daily: pd.DataFrame) -> pd.DataFrame:
    candidate_name = "breakout_loop_scores_range_p75"
    comparisons = (
        ("candidate_minus_ungated_breakout", "breakout_all"),
        ("candidate_minus_state_context", "breakout_state_context_range_p75"),
        ("candidate_minus_raw_history", "breakout_raw_history_range_p75"),
    )
    rows: list[dict[str, Any]] = []
    for period_index, period in enumerate(("2025", "2023")):
        for horizon_index, horizon in enumerate(HORIZONS):
            common = daily.loc[daily["period"].astype(str).eq(period) & daily["horizon"].eq(horizon) & daily["cost_bps_per_side"].eq(PRIMARY_COST)]
            candidate = common.loc[common["strategy"].eq(candidate_name)].sort_values("session_date")
            observed, low, high = block_interval(candidate["daily_return"], period_index * 100 + horizon_index * 10)
            rows.append(dict(period=period, horizon=horizon, comparison="candidate_absolute", candidate=candidate_name, baseline="zero", session_dates=len(candidate), mean_daily_return=observed, ci_lower=low, ci_upper=high))
            for comparison_index, (comparison, baseline_name) in enumerate(comparisons, start=1):
                baseline = common.loc[common["strategy"].eq(baseline_name)].sort_values("session_date")
                difference = candidate["daily_return"].to_numpy(float) - baseline["daily_return"].to_numpy(float)
                observed, low, high = block_interval(difference, period_index * 100 + horizon_index * 10 + comparison_index)
                rows.append(dict(period=period, horizon=horizon, comparison=comparison, candidate=candidate_name, baseline=baseline_name, session_dates=len(candidate), mean_daily_return=observed, ci_lower=low, ci_upper=high))
    return pd.DataFrame(rows)


def frame_close(observed: pd.DataFrame, expected: pd.DataFrame, name: str) -> float:
    if list(observed.columns) != list(expected.columns) or len(observed) != len(expected):
        raise AssertionError(f"{name} shape mismatch")
    maximum = 0.0
    for column in observed.columns:
        if pd.api.types.is_numeric_dtype(observed[column]):
            left = observed[column].to_numpy(float)
            right = expected[column].to_numpy(float)
            if not np.allclose(left, right, rtol=1e-9, atol=1e-10, equal_nan=True):
                raise AssertionError(f"{name} {column} mismatch")
            finite = np.abs(left - right)[np.isfinite(left - right)]
            maximum = max(maximum, float(finite.max(initial=0.0)))
        elif not observed[column].astype(str).equals(expected[column].astype(str)):
            raise AssertionError(f"{name} {column} mismatch")
    return maximum


def main() -> None:
    check_rows: list[dict[str, Any]] = []

    def record(name: str, detail: Any) -> None:
        check_rows.append(dict(name=name, pass_=True, detail=detail))

    contract = json.loads(CONTRACT.read_text())
    symbols = contract["overlap_and_portfolio"]["fixed_universe"]
    pre_score = json.loads(PRE_SCORE.read_text())
    actual_hashes = {name: digest(path) for name, path in source_paths(symbols).items()}
    if actual_hashes != pre_score["sha256"]:
        raise AssertionError("source hash mismatch")
    record("frozen_source_hashes", len(actual_hashes))
    source_hashes = json.loads((ROOT / "source_hashes.json").read_text())
    if source_hashes["sha256"] != actual_hashes or source_hashes["pre_score_manifest_sha256"] != digest(PRE_SCORE):
        raise AssertionError("artifact source binding mismatch")
    record("artifact_source_binding", digest(PRE_SCORE))
    if not (contract["research_only"] and not contract["live_ordering_enabled"] and contract["order_placement"] == "disabled"):
        raise AssertionError("safety contract mismatch")
    record("research_only_boundary", True)

    expected_thresholds = reconstruct_thresholds()
    observed_thresholds = pd.read_csv(ROOT / "prediction_thresholds_2024.csv")
    record("exact_2024_thresholds", frame_close(observed_thresholds, expected_thresholds, "thresholds"))

    predictions: dict[int, pd.DataFrame] = {}
    sessions_by_period: dict[str, list[str]] = {}
    execution: dict[int, pd.DataFrame] = {}
    maximum_execution_error = 0.0
    for year in (2025, 2023):
        predictions[year] = load_predictions(year, symbols)
        tape = load_tape(year, symbols)
        sessions_by_period[str(year)] = sorted(tape["session_date"].unique())
        execution[year] = pd.read_parquet(ROOT / f"execution_anchors_{year}.parquet")
        _, error = verify_execution(year, predictions[year], tape, execution[year])
        maximum_execution_error = max(maximum_execution_error, error)
    record("exact_provider_execution_reconstruction", maximum_execution_error)
    if any(len(sessions) != 250 for sessions in sessions_by_period.values()):
        raise AssertionError("full session grid mismatch")
    record("full_zero_fill_session_grids", {key: len(value) for key, value in sessions_by_period.items()})

    ledger = pd.read_parquet(ROOT / "accepted_signal_ledger.parquet")
    checked = sum(
        verify_ledger(year, predictions[year], execution[year], observed_thresholds, ledger)
        for year in (2025, 2023)
    )
    if checked != len(ledger):
        raise AssertionError("ledger row count mismatch")
    record("exact_gates_overlap_and_signal_ledger", checked)
    if not np.allclose(
        ledger.loc[ledger["status"].eq("filled"), "gross_return_bps"],
        10000.0 * ledger.loc[ledger["status"].eq("filled"), "gross_return"],
    ):
        raise AssertionError("simple-return bps mismatch")
    record("cash_consistent_simple_trade_returns", True)

    metrics, daily, quarters, deletions = recompute_metrics(ledger, sessions_by_period)
    metric_errors = {
        "pnl_metrics": frame_close(pd.read_csv(ROOT / "pnl_metrics.csv"), metrics, "pnl metrics"),
        "daily_returns": frame_close(pd.read_parquet(ROOT / "daily_portfolio_returns.parquet"), daily, "daily returns"),
        "quarters": frame_close(pd.read_csv(ROOT / "quarter_metrics.csv"), quarters, "quarters"),
        "stock_deletions": frame_close(pd.read_csv(ROOT / "stock_deletion_metrics.csv"), deletions, "deletions"),
    }
    record("exact_costs_compounding_and_all_metric_slices", metric_errors)
    if not daily.groupby(["period", "strategy", "horizon", "cost_bps_per_side"]).size().eq(250).all():
        raise AssertionError("daily zero-fill grid incomplete")
    record("inactive_sleeves_and_dates_zero_filled", True)

    expected_bootstrap = recompute_bootstraps(daily)
    observed_bootstrap = pd.read_csv(ROOT / "primary_bootstraps.csv")
    record("exact_moving_block_bootstraps", frame_close(observed_bootstrap, expected_bootstrap, "bootstraps"))
    decision = json.loads((ROOT / "decision.json").read_text())
    candidate = "breakout_loop_scores_range_p75"
    primary = metrics.loc[metrics.strategy.eq(candidate) & metrics.cost_bps_per_side.eq(PRIMARY_COST)]
    q = quarters.loc[quarters.strategy.eq(candidate) & quarters.cost_bps_per_side.eq(PRIMARY_COST)]
    d = deletions.loc[deletions.strategy.eq(candidate) & deletions.cost_bps_per_side.eq(PRIMARY_COST)]
    b_abs = expected_bootstrap[expected_bootstrap.comparison.eq("candidate_absolute")]
    b_all = expected_bootstrap[expected_bootstrap.comparison.eq("candidate_minus_ungated_breakout")]
    b_state = expected_bootstrap[expected_bootstrap.comparison.eq("candidate_minus_state_context")]
    b_history = expected_bootstrap[expected_bootstrap.comparison.eq("candidate_minus_raw_history")]
    gate_checks = {
        "positive_annualized_return_each_period_and_horizon": bool(primary.annualized_return.gt(0).all() and len(primary) == 6),
        "positive_mean_net_trade_bps_each_period_and_horizon": bool(primary.mean_net_trade_bps.gt(0).all() and len(primary) == 6),
        "daily_return_bootstrap_lower_bound_above_zero_each_period_and_horizon": bool(b_abs.ci_lower.gt(0).all() and len(b_abs) == 6),
        "positive_return_each_quarter_each_period_and_horizon": bool(q.cumulative_return.gt(0).all() and len(q) == 24),
        "positive_return_under_every_leave_one_stock_out_deletion_each_period_and_horizon": bool(d.cumulative_return.gt(0).all() and len(d) == 6 * UNIVERSE_SIZE),
        "paired_daily_return_advantage_over_ungated_breakout_bootstrap_lower_bound_above_zero_each_period_and_horizon": bool(b_all.ci_lower.gt(0).all() and len(b_all) == 6),
        "paired_daily_return_advantage_over_state_context_gate_bootstrap_lower_bound_above_zero_each_period_and_horizon": bool(b_state.ci_lower.gt(0).all() and len(b_state) == 6),
        "paired_daily_return_advantage_over_raw_history_gate_bootstrap_lower_bound_above_zero_each_period_and_horizon": bool(b_history.ci_lower.gt(0).all() and len(b_history) == 6),
    }
    gate_checks["pass"] = all(gate_checks.values())
    if gate_checks != decision["checks"] or decision["decision"] != "pnl_translation_not_supported":
        raise AssertionError("decision mismatch")
    record("exact_predeclared_decision", gate_checks)
    if not (decision["research_only"] and not decision["live_ordering_enabled"] and decision["order_placement"] == "disabled" and not decision["strategy_promotion"]):
        raise AssertionError("decision safety failure")
    record("decision_safety_boundary", True)

    result = {
        "audit": "frozen_regime_loop_pnl_sanity_v1_independent",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "all_passed": True,
        "passed": len(check_rows),
        "failed": 0,
        "checks": [
            {"name": row["name"], "pass": row["pass_"], "detail": row["detail"]}
            for row in check_rows
        ],
    }
    audit_path = ROOT / "independent_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    files = sorted(path for path in ROOT.iterdir() if path.is_file() and path.name != "artifact_manifest.json")
    (ROOT / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "research_only": True,
                "live_ordering_enabled": False,
                "order_placement": "disabled",
                "files": [
                    {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
                    for path in files
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
