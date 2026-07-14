#!/usr/bin/env python3
"""Offline research-only P&L sanity test for frozen regime/loop forecasts.

No live, paper, demo, broker, order-submission, or deployment path exists.
"""

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
CONTRACT_PATH = WORK / "contracts/20260712-frozen-regime-loop-pnl-sanity-v1.json"
PRE_SCORE_PATH = WORK / "contracts/20260712-frozen-regime-loop-pnl-sanity-v1-pre-score.json"
PRICE_ROOT = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710")
PREDICTION_PATHS = {
    2025: PRICE_ROOT / "price_predictions_2025.parquet",
    2023: PRICE_ROOT / "price_predictions_2023.parquet",
}
TRAIN_PANEL_PATH = PRICE_ROOT / "anchor_panel_train_2024.parquet"
MODEL_PATH = PRICE_ROOT / "outcome_model_parameters.npz"
FEATURE_MANIFEST_PATH = PRICE_ROOT / "feature_manifest.json"
RAW_ROOTS = {
    2025: Path(
        "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/"
        "instrument_type=stock"
    ),
    2023: Path(
        "/private/tmp/stocker_eodhd_pre2024_intraday_20260710/"
        "source=eodhd/instrument_type=stock"
    ),
}
OUT = Path("/private/tmp/stocker_frozen_regime_loop_pnl_sanity_v1_20260712")

SEED = 20260712
HORIZONS = (6, 12, 24)
REPRESENTATIONS = ("state_context", "raw_history", "loop_scores")
COSTS = (0, 1, 2, 5, 10)
PRIMARY_COST = 5
UNIVERSE_SIZE = 20
SESSION_BARS = 78
K = 8
TOKEN_COUNT = 648
NUMERIC_CONTROLS = (
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
LOOP_COLUMNS = tuple(f"loop_score_{index:02d}" for index in range(1, 21))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n")


def provider_path(root: Path, symbol: str) -> Path:
    stored = "VTI.US" if symbol == "VTI" else symbol
    return root / f"symbol={stored}" / "timeframe=5m" / "data.parquet"


def source_paths(symbols: list[str]) -> dict[str, Path]:
    paths = {
        "contract": CONTRACT_PATH,
        "runner": Path(__file__).resolve(),
        "predictions_2025": PREDICTION_PATHS[2025],
        "predictions_2023": PREDICTION_PATHS[2023],
        "threshold_panel_2024": TRAIN_PANEL_PATH,
        "model_parameters": MODEL_PATH,
        "feature_manifest": FEATURE_MANIFEST_PATH,
    }
    for year in (2025, 2023):
        for symbol in symbols:
            paths[f"provider_{year}_{symbol}"] = provider_path(RAW_ROOTS[year], symbol)
    return paths


def load_contract_and_verify_hashes(symbols: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    if not (
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
        and contract["broker_connection_enabled"] is False
        and contract["paper_or_demo_execution_enabled"] is False
    ):
        raise AssertionError("research-only execution boundary drift")
    if contract["horizons_bars"] != list(HORIZONS):
        raise AssertionError("horizon contract drift")
    if contract["pnl"]["cost_grid_bps_per_side"] != list(COSTS):
        raise AssertionError("cost-grid contract drift")
    paths = source_paths(symbols)
    actual = {name: sha256(path) for name, path in paths.items()}
    if actual != pre_score["sha256"]:
        raise AssertionError("pre-score source hash mismatch")
    return contract, pre_score


def state_matrix(values: np.ndarray) -> sparse.csr_matrix:
    states = np.asarray(values, dtype=int)
    return sparse.csr_matrix(
        (
            np.ones(len(states), dtype=np.float32),
            (np.arange(len(states)), states),
        ),
        shape=(len(states), K),
    )


def token_matrix(values: np.ndarray) -> sparse.csr_matrix:
    tokens = np.asarray(values, dtype=int)
    return sparse.csr_matrix(
        (
            np.ones(len(tokens), dtype=np.float32),
            (np.arange(len(tokens)), tokens),
        ),
        shape=(len(tokens), TOKEN_COUNT),
    )


def build_training_features(
    panel: pd.DataFrame, representation: str, medians: pd.Series
) -> sparse.csr_matrix:
    numeric = (
        panel.loc[:, NUMERIC_CONTROLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(medians)
        .to_numpy(np.float32)
    )
    context = sparse.hstack(
        (state_matrix(panel["state"].to_numpy(int)), sparse.csr_matrix(numeric)),
        format="csr",
    )
    if representation == "state_context":
        return context
    if representation == "raw_history":
        return sparse.hstack(
            (context, token_matrix(panel["history_token"].to_numpy(int))), format="csr"
        )
    if representation == "loop_scores":
        return sparse.hstack(
            (context, sparse.csr_matrix(panel.loc[:, LOOP_COLUMNS].to_numpy(np.float32))),
            format="csr",
        )
    raise AssertionError(f"unknown representation {representation}")


def reconstruct_thresholds() -> pd.DataFrame:
    required = {
        "state",
        "history_token",
        "bar_index_in_session",
        *(f"exact_{horizon}" for horizon in HORIZONS),
        *NUMERIC_CONTROLS,
        *LOOP_COLUMNS,
    }
    panel = pd.read_parquet(TRAIN_PANEL_PATH, columns=sorted(required))
    if panel["bar_index_in_session"].astype(int).gt(53).any() or not all(
        panel[f"exact_{horizon}"].astype(bool).all() for horizon in HORIZONS
    ):
        raise AssertionError("2024 threshold cohort lacks exact frozen support")
    manifest = json.loads(FEATURE_MANIFEST_PATH.read_text())
    medians = pd.Series(manifest["numeric_medians"]).reindex(NUMERIC_CONTROLS)
    parameters = np.load(MODEL_PATH)
    rows: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        raw = build_training_features(panel, representation, medians)
        scale = parameters[f"{representation}__scaler_scale"]
        if raw.shape[1] != len(scale):
            raise AssertionError("threshold feature-width mismatch")
        scaled = raw.multiply(1.0 / scale).tocsr()
        for horizon in HORIZONS:
            coefficients = parameters[
                f"{representation}__future_range_bps__h{horizon}__coef"
            ]
            intercept = float(
                parameters[
                    f"{representation}__future_range_bps__h{horizon}__intercept"
                ][0]
            )
            prediction = np.asarray(scaled @ coefficients).ravel() + intercept
            if not np.isfinite(prediction).all():
                raise AssertionError("invalid reconstructed 2024 prediction")
            rows.append(
                {
                    "representation": representation,
                    "horizon": horizon,
                    "training_rows": len(panel),
                    "prediction_mean_bps": float(prediction.mean()),
                    "prediction_p75_bps": float(
                        np.quantile(prediction, 0.75, method="linear")
                    ),
                    "prediction_p90_bps": float(
                        np.quantile(prediction, 0.90, method="linear")
                    ),
                }
            )
    return pd.DataFrame(rows)


def prediction_columns() -> list[str]:
    columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "start_timestamp",
        "state",
        "history_token",
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
    return columns


def load_predictions(year: int) -> pd.DataFrame:
    frame = pd.read_parquet(PREDICTION_PATHS[year], columns=prediction_columns())
    frame["start_timestamp"] = pd.to_datetime(
        frame["start_timestamp"], utc=True, errors="raise"
    )
    frame["session_date"] = frame["session_date"].astype(str)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    if set(pd.to_datetime(frame["session_date"]).dt.year.unique()) != {year}:
        raise AssertionError(f"prediction year boundary failure {year}")
    if frame["anchor_id"].duplicated().any():
        raise AssertionError(f"duplicate prediction anchor {year}")
    probability_columns = [
        column for column in frame.columns if "direction_probability" in column
    ]
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(float)).all():
        raise AssertionError("non-finite frozen prediction")
    probabilities = frame.loc[:, probability_columns].to_numpy(float)
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise AssertionError("invalid direction probability")
    return frame.sort_values(
        ["symbol_norm", "session_date", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)


def load_provider_tape(year: int, symbols: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        path = provider_path(RAW_ROOTS[year], symbol)
        frame = pd.read_parquet(
            path, columns=["timestamp", "open", "high", "low", "close"]
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        frame = frame.loc[
            frame["timestamp"].ge(start) & frame["timestamp"].lt(end)
        ].dropna(subset=["timestamp", "open", "high", "low", "close"])
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        frame = frame.loc[minute.ge(570) & minute.lt(960)].copy()
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        frame["session_date"] = local.dt.strftime("%Y-%m-%d")
        frame["symbol_norm"] = symbol
        frame["bar_ordinal"] = frame.groupby("session_date", sort=False).cumcount()
        if frame.empty or frame["timestamp"].duplicated().any():
            raise AssertionError(f"invalid provider tape {year} {symbol}")
        if frame[["open", "high", "low", "close"]].le(0.0).any().any():
            raise AssertionError(f"non-positive provider price {year} {symbol}")
        rows.append(frame)
    tape = pd.concat(rows, ignore_index=True)
    tape = tape.sort_values(
        ["symbol_norm", "session_date", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    tape["tape_position"] = np.arange(len(tape), dtype=np.int64)
    return tape


def breakout_execution(
    tape: pd.DataFrame,
    positions: np.ndarray,
    horizon: int,
    upper: np.ndarray,
    lower: np.ndarray,
    exit_close: np.ndarray,
) -> dict[str, np.ndarray]:
    steps = np.arange(1, horizon + 1, dtype=int)
    indices = positions[:, None] + steps[None, :]
    opens = tape["open"].to_numpy(float)[indices]
    highs = tape["high"].to_numpy(float)[indices]
    lows = tape["low"].to_numpy(float)[indices]
    upper_hit = highs >= upper[:, None]
    lower_hit = lows <= lower[:, None]
    open_upper = opens >= upper[:, None]
    open_lower = opens <= lower[:, None]
    event = np.zeros((len(positions), horizon), dtype=np.int8)
    event[open_upper & ~open_lower] = 1
    event[open_lower & ~open_upper] = -1
    event[open_upper & open_lower] = 2
    unresolved_at_open = ~(open_upper | open_lower)
    event[unresolved_at_open & upper_hit & ~lower_hit] = 1
    event[unresolved_at_open & lower_hit & ~upper_hit] = -1
    event[unresolved_at_open & upper_hit & lower_hit] = 2
    has_event = (event != 0).any(axis=1)
    first_event_step = np.where(
        has_event, (event != 0).argmax(axis=1) + 1, 0
    ).astype(int)
    first_code = np.zeros(len(positions), dtype=np.int8)
    event_rows = np.flatnonzero(has_event)
    first_code[event_rows] = event[
        event_rows, first_event_step[event_rows] - 1
    ]
    ambiguous = first_code == 2
    long_fill = first_code == 1
    short_fill = first_code == -1
    filled = long_fill | short_fill
    direction = np.zeros(len(positions), dtype=np.int8)
    direction[long_fill] = 1
    direction[short_fill] = -1
    entry_step = np.where(filled, first_event_step, 0)
    entry = np.full(len(positions), np.nan, dtype=float)
    filled_rows = np.flatnonzero(filled)
    triggering_open = opens[
        filled_rows, entry_step[filled_rows].astype(int) - 1
    ]
    long_rows = np.flatnonzero(long_fill)
    short_rows = np.flatnonzero(short_fill)
    entry[long_rows] = np.maximum(
        upper[long_rows], opens[long_rows, entry_step[long_rows] - 1]
    )
    entry[short_rows] = np.minimum(
        lower[short_rows], opens[short_rows, entry_step[short_rows] - 1]
    )
    if len(filled_rows) and not np.allclose(entry[filled_rows], np.where(
        direction[filled_rows] == 1,
        np.maximum(upper[filled_rows], triggering_open),
        np.minimum(lower[filled_rows], triggering_open),
    )):
        raise AssertionError("gap-fill reconstruction failure")
    gross_return = np.full(len(positions), np.nan, dtype=float)
    gross_return[filled] = direction[filled] * (
        exit_close[filled] / entry[filled] - 1.0
    )
    gross_bps = 10000.0 * gross_return
    status = np.full(len(positions), "no_trigger", dtype=object)
    status[ambiguous] = "ambiguous_same_bar"
    status[filled] = "filled"
    holding_bars = np.where(filled, horizon - entry_step + 1, 0).astype(int)
    return {
        "status": status,
        "direction": direction,
        "entry_step": entry_step.astype(int),
        "entry_price": entry,
        "exit_price": exit_close,
        "gross_return": gross_return,
        "gross_return_bps": gross_bps,
        "holding_bars": holding_bars,
    }


def attach_execution_prices(
    year: int, predictions: pd.DataFrame, tape: pd.DataFrame
) -> pd.DataFrame:
    keys = ["symbol_norm", "start_timestamp"]
    lookup = tape[
        [
            "symbol_norm",
            "timestamp",
            "session_date",
            "bar_ordinal",
            "tape_position",
            "open",
            "high",
            "low",
            "close",
        ]
    ].rename(
        columns={
            "timestamp": "start_timestamp",
            "session_date": "tape_session_date",
            "open": "anchor_open",
            "high": "anchor_high",
            "low": "anchor_low",
            "close": "anchor_close",
        }
    )
    frame = (
        predictions.reset_index(names="prediction_position")
        .merge(lookup, on=keys, how="left", validate="one_to_one")
        .sort_values("prediction_position", kind="stable")
        .reset_index(drop=True)
    )
    if frame["tape_position"].isna().any() or not frame["session_date"].eq(
        frame["tape_session_date"]
    ).all():
        raise AssertionError(f"prediction-to-tape join failure {year}")
    positions = frame["tape_position"].to_numpy(int)
    tape_symbols = tape["symbol_norm"].to_numpy(str)
    tape_sessions = tape["session_date"].to_numpy(str)
    tape_timestamps = tape["timestamp"].to_numpy()
    for horizon in HORIZONS:
        future_positions = positions + horizon
        if future_positions.max() >= len(tape):
            raise AssertionError("future tape position overflow")
        exact = (
            (tape_symbols[future_positions] == frame["symbol_norm"].to_numpy(str))
            & (tape_sessions[future_positions] == frame["session_date"].to_numpy(str))
            & (
                tape_timestamps[future_positions] - frame["start_timestamp"].to_numpy()
                == np.timedelta64(5 * horizon, "m")
            )
        )
        if not exact.all():
            raise AssertionError(f"inexact execution horizon {year} h{horizon}")
        next_open = tape["open"].to_numpy(float)[positions + 1]
        exit_close = tape["close"].to_numpy(float)[future_positions]
        frame[f"next_open_{horizon}"] = next_open
        frame[f"exit_close_{horizon}"] = exit_close
        close_to_close = 10000.0 * np.log(
            exit_close / frame["anchor_close"].to_numpy(float)
        )
        if not np.allclose(
            close_to_close,
            frame[f"signed_return_bps_target_{horizon}"].to_numpy(float),
            rtol=1e-9,
            atol=1e-7,
        ):
            raise AssertionError(f"provider outcome mismatch {year} h{horizon}")
        breakout = breakout_execution(
            tape,
            positions,
            horizon,
            frame["anchor_high"].to_numpy(float),
            frame["anchor_low"].to_numpy(float),
            exit_close,
        )
        for name, values in breakout.items():
            frame[f"breakout_{name}_{horizon}"] = values
    if frame["bar_ordinal"].astype(int).gt(53).any():
        raise AssertionError("execution anchor after bar 53")
    return frame


def strategy_definitions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        rows.append(
            {
                "strategy": f"direction_{representation}_all",
                "family": "directional_falsification",
                "representation": representation,
                "gated": False,
            }
        )
        rows.append(
            {
                "strategy": f"direction_{representation}_range_p75",
                "family": "directional_falsification",
                "representation": representation,
                "gated": True,
            }
        )
    rows.append(
        {
            "strategy": "breakout_all",
            "family": "causal_breakout",
            "representation": "none",
            "gated": False,
        }
    )
    for representation in REPRESENTATIONS:
        rows.append(
            {
                "strategy": f"breakout_{representation}_range_p75",
                "family": "causal_breakout",
                "representation": representation,
                "gated": True,
            }
        )
    return rows


def greedy_nonoverlap(
    frame: pd.DataFrame, eligible: np.ndarray, horizon: int
) -> np.ndarray:
    accepted = np.zeros(len(frame), dtype=bool)
    candidates = frame.loc[
        eligible, ["symbol_norm", "session_date", "bar_ordinal"]
    ]
    for _, group in candidates.groupby(["symbol_norm", "session_date"], sort=False):
        last_exit = -1
        for position, ordinal in zip(group.index, group["bar_ordinal"], strict=True):
            ordinal = int(ordinal)
            if ordinal >= last_exit:
                accepted[int(position)] = True
                last_exit = ordinal + horizon
    return accepted


def build_signal_ledger(
    year: int, frame: pd.DataFrame, thresholds: pd.DataFrame
) -> pd.DataFrame:
    threshold_lookup = {
        (row.representation, int(row.horizon)): float(row.prediction_p75_bps)
        for row in thresholds.itertuples(index=False)
    }
    ledgers: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        for strategy in strategy_definitions():
            representation = str(strategy["representation"])
            threshold = (
                threshold_lookup[(representation, horizon)]
                if bool(strategy["gated"])
                else math.nan
            )
            gate_value = (
                frame[
                    f"{representation}__future_range_bps_prediction_{horizon}"
                ].to_numpy(float)
                if bool(strategy["gated"])
                else np.full(len(frame), np.nan)
            )
            eligible = (
                gate_value >= threshold
                if bool(strategy["gated"])
                else np.ones(len(frame), dtype=bool)
            )
            accepted = greedy_nonoverlap(frame, eligible, horizon)
            selected = frame.loc[
                accepted,
                [
                    "anchor_id",
                    "symbol_norm",
                    "session_date",
                    "quarter",
                    "start_timestamp",
                    "bar_ordinal",
                    "anchor_open",
                    "anchor_high",
                    "anchor_low",
                    "anchor_close",
                ],
            ].copy()
            positions = np.flatnonzero(accepted)
            selected["period"] = str(year)
            selected["strategy"] = str(strategy["strategy"])
            selected["family"] = str(strategy["family"])
            selected["representation"] = representation
            selected["horizon"] = horizon
            selected["gate_threshold_bps"] = threshold
            selected["gate_value_bps"] = gate_value[positions]
            selected["signal_exit_ordinal"] = (
                selected["bar_ordinal"].to_numpy(int) + horizon
            )
            if strategy["family"] == "directional_falsification":
                probability = frame.loc[
                    accepted,
                    f"{representation}__direction_probability_{horizon}",
                ].to_numpy(float)
                direction = np.where(probability >= 0.5, 1, -1).astype(np.int8)
                entry = frame.loc[accepted, f"next_open_{horizon}"].to_numpy(float)
                exit_price = frame.loc[accepted, f"exit_close_{horizon}"].to_numpy(float)
                selected["direction_probability"] = probability
                selected["status"] = "filled"
                selected["direction"] = direction
                selected["entry_step"] = 1
                selected["entry_price"] = entry
                selected["exit_price"] = exit_price
                selected["gross_return"] = direction * (
                    exit_price / entry - 1.0
                )
                selected["gross_return_bps"] = (
                    10000.0 * selected["gross_return"].to_numpy(float)
                )
                selected["holding_bars"] = horizon
            else:
                selected["direction_probability"] = np.nan
                for column in (
                    "status",
                    "direction",
                    "entry_step",
                    "entry_price",
                    "exit_price",
                    "gross_return",
                    "gross_return_bps",
                    "holding_bars",
                ):
                    selected[column] = frame.loc[
                        accepted, f"breakout_{column}_{horizon}"
                    ].to_numpy()
            ledgers.append(selected)
    ledger = pd.concat(ledgers, ignore_index=True)
    if ledger.duplicated(["period", "strategy", "horizon", "anchor_id"]).any():
        raise AssertionError("duplicate accepted signal")
    filled = ledger["status"].eq("filled")
    if not np.isfinite(
        ledger.loc[filled, ["gross_return", "gross_return_bps"]].to_numpy(float)
    ).all():
        raise AssertionError("non-finite filled trade P&L")
    return ledger


def portfolio_statistics(daily: np.ndarray) -> dict[str, float]:
    values = np.asarray(daily, dtype=float)
    equity = np.cumprod(1.0 + values)
    cumulative = float(equity[-1] - 1.0) if len(equity) else 0.0
    annualized = (
        float((1.0 + cumulative) ** (252.0 / len(values)) - 1.0)
        if len(values) and cumulative > -1.0
        else math.nan
    )
    volatility = float(np.std(values, ddof=1) * math.sqrt(252.0)) if len(values) > 1 else 0.0
    sharpe = (
        float(np.mean(values) / np.std(values, ddof=1) * math.sqrt(252.0))
        if len(values) > 1 and np.std(values, ddof=1) > 0.0
        else math.nan
    )
    running_max = np.maximum.accumulate(np.r_[1.0, equity])
    drawdown = np.r_[1.0, equity] / running_max - 1.0
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "descriptive_sharpe_zero_rate": sharpe,
        "maximum_drawdown": float(drawdown.min(initial=0.0)),
        "mean_daily_return": float(values.mean()) if len(values) else 0.0,
    }


def daily_returns_for_signals(
    signals: pd.DataFrame,
    sessions: list[str],
    cost: int,
    deleted_symbol: str | None = None,
) -> tuple[pd.Series, np.ndarray]:
    frame = signals.loc[signals["status"].eq("filled")].copy()
    divisor = UNIVERSE_SIZE
    if deleted_symbol is not None:
        frame = frame.loc[frame["symbol_norm"].ne(deleted_symbol)].copy()
        divisor -= 1
    frame["net_return"] = (
        frame["gross_return"].to_numpy(float) - 2.0 * cost / 10000.0
    )
    if frame["net_return"].le(-1.0).any():
        raise AssertionError("a sleeve trade lost at least its full collateral")
    frame["log_growth"] = np.log1p(frame["net_return"].to_numpy(float))
    sleeve_growth = frame.groupby(["session_date", "symbol_norm"], sort=False)[
        "log_growth"
    ].sum()
    sleeve_return = np.expm1(sleeve_growth)
    daily = (
        sleeve_return.groupby("session_date").sum() / divisor
    ).reindex(sessions, fill_value=0.0)
    return daily, 10000.0 * frame["net_return"].to_numpy(float)


def evaluate_ledgers(
    ledger: pd.DataFrame, sessions_by_period: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    quarter_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    group_columns = ["period", "strategy", "family", "representation", "horizon"]
    for keys, signals in ledger.groupby(group_columns, sort=False):
        period, strategy, family, representation, horizon = keys
        sessions = sessions_by_period[str(period)]
        symbols = sorted(ledger.loc[ledger["period"].eq(period), "symbol_norm"].unique())
        for cost in COSTS:
            daily, net_bps = daily_returns_for_signals(signals, sessions, cost)
            stats = portfolio_statistics(daily.to_numpy(float))
            filled = signals["status"].eq("filled")
            positive = net_bps[net_bps > 0.0]
            negative = net_bps[net_bps < 0.0]
            if len(negative) and -negative.sum() > 0.0:
                profit_factor = float(positive.sum() / -negative.sum())
            elif len(positive):
                profit_factor = math.inf
            else:
                profit_factor = math.nan
            exposure = float(
                signals.loc[filled, "holding_bars"].to_numpy(float).sum()
                / (SESSION_BARS * UNIVERSE_SIZE * len(sessions))
            )
            metric_rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    "family": family,
                    "representation": representation,
                    "horizon": int(horizon),
                    "cost_bps_per_side": cost,
                    "accepted_signals": len(signals),
                    "filled_trades": int(filled.sum()),
                    "no_trigger_signals": int(signals["status"].eq("no_trigger").sum()),
                    "ambiguous_signals": int(
                        signals["status"].eq("ambiguous_same_bar").sum()
                    ),
                    "fill_rate": float(filled.mean()),
                    "mean_net_trade_bps": float(net_bps.mean()) if len(net_bps) else math.nan,
                    "median_net_trade_bps": float(np.median(net_bps)) if len(net_bps) else math.nan,
                    "win_rate": float((net_bps > 0.0).mean()) if len(net_bps) else math.nan,
                    "profit_factor": profit_factor,
                    "exposure_fraction": exposure,
                    **stats,
                }
            )
            daily_frame = pd.DataFrame(
                {
                    "period": period,
                    "strategy": strategy,
                    "family": family,
                    "representation": representation,
                    "horizon": int(horizon),
                    "cost_bps_per_side": cost,
                    "session_date": daily.index.astype(str),
                    "daily_return": daily.to_numpy(float),
                }
            )
            daily_rows.append(daily_frame)
            daily_frame["quarter"] = (
                pd.to_datetime(daily_frame["session_date"]).dt.year.astype(str)
                + "_q"
                + pd.to_datetime(daily_frame["session_date"]).dt.quarter.astype(str)
            )
            for quarter, selected in daily_frame.groupby("quarter", sort=True):
                quarter_stats = portfolio_statistics(selected["daily_return"].to_numpy(float))
                quarter_rows.append(
                    {
                        "period": period,
                        "strategy": strategy,
                        "horizon": int(horizon),
                        "cost_bps_per_side": cost,
                        "quarter": quarter,
                        "session_dates": len(selected),
                        **quarter_stats,
                    }
                )
            for deleted_symbol in symbols:
                deletion_daily, _ = daily_returns_for_signals(
                    signals, sessions, cost, deleted_symbol
                )
                deletion_stats = portfolio_statistics(
                    deletion_daily.to_numpy(float)
                )
                deletion_rows.append(
                    {
                        "period": period,
                        "strategy": strategy,
                        "horizon": int(horizon),
                        "cost_bps_per_side": cost,
                        "deleted_symbol": deleted_symbol,
                        "session_dates": len(deletion_daily),
                        **deletion_stats,
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(daily_rows, ignore_index=True),
        pd.DataFrame(quarter_rows),
        pd.DataFrame(deletion_rows),
    )


def moving_block_interval(
    values: np.ndarray, seed_offset: int, draws: int = 5000, block: int = 5
) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED + seed_offset)
    starts = np.arange(len(data) - block + 1)
    blocks = math.ceil(len(data) / block)
    selected = rng.choice(starts, size=(draws, blocks), replace=True)
    positions = (selected[:, :, None] + np.arange(block)[None, None, :]).reshape(
        draws, -1
    )[:, : len(data)]
    sampled = data[positions].mean(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975], method="linear")
    return float(data.mean()), float(lower), float(upper)


def bootstrap_primary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_name = "breakout_loop_scores_range_p75"
    comparison_baselines = (
        ("candidate_minus_ungated_breakout", "breakout_all"),
        (
            "candidate_minus_state_context",
            "breakout_state_context_range_p75",
        ),
        ("candidate_minus_raw_history", "breakout_raw_history_range_p75"),
    )
    for period_index, period in enumerate(("2025", "2023")):
        for horizon_index, horizon in enumerate(HORIZONS):
            common = daily.loc[
                daily["period"].astype(str).eq(period)
                & daily["horizon"].eq(horizon)
                & daily["cost_bps_per_side"].eq(PRIMARY_COST)
            ]
            candidate = common.loc[
                common["strategy"].eq(candidate_name)
            ].sort_values("session_date")
            observed, lower, upper = moving_block_interval(
                candidate["daily_return"].to_numpy(float),
                period_index * 100 + horizon_index * 10,
            )
            rows.append(
                {
                    "period": period,
                    "horizon": horizon,
                    "comparison": "candidate_absolute",
                    "candidate": candidate_name,
                    "baseline": "zero",
                    "session_dates": len(candidate),
                    "mean_daily_return": observed,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
            for comparison_index, (comparison, baseline_name) in enumerate(
                comparison_baselines, start=1
            ):
                baseline = common.loc[
                    common["strategy"].eq(baseline_name)
                ].sort_values("session_date")
                if not candidate["session_date"].reset_index(drop=True).equals(
                    baseline["session_date"].reset_index(drop=True)
                ):
                    raise AssertionError("paired daily alignment failure")
                difference = candidate["daily_return"].to_numpy(float) - baseline[
                    "daily_return"
                ].to_numpy(float)
                observed, lower, upper = moving_block_interval(
                    difference,
                    period_index * 100
                    + horizon_index * 10
                    + comparison_index,
                )
                rows.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "comparison": comparison,
                        "candidate": candidate_name,
                        "baseline": baseline_name,
                        "session_dates": len(candidate),
                        "mean_daily_return": observed,
                        "ci_lower": lower,
                        "ci_upper": upper,
                    }
                )
    return pd.DataFrame(rows)


def make_decision(
    metrics: pd.DataFrame,
    quarters: pd.DataFrame,
    deletions: pd.DataFrame,
    bootstraps: pd.DataFrame,
) -> dict[str, Any]:
    candidate = "breakout_loop_scores_range_p75"
    primary = metrics.loc[
        metrics["strategy"].eq(candidate)
        & metrics["cost_bps_per_side"].eq(PRIMARY_COST)
    ]
    quarter = quarters.loc[
        quarters["strategy"].eq(candidate)
        & quarters["cost_bps_per_side"].eq(PRIMARY_COST)
    ]
    deletion = deletions.loc[
        deletions["strategy"].eq(candidate)
        & deletions["cost_bps_per_side"].eq(PRIMARY_COST)
    ]
    absolute_bootstrap = bootstraps.loc[
        bootstraps["comparison"].eq("candidate_absolute")
    ]
    paired_bootstrap = bootstraps.loc[
        bootstraps["comparison"].eq("candidate_minus_state_context")
    ]
    ungated_bootstrap = bootstraps.loc[
        bootstraps["comparison"].eq("candidate_minus_ungated_breakout")
    ]
    history_bootstrap = bootstraps.loc[
        bootstraps["comparison"].eq("candidate_minus_raw_history")
    ]
    checks = {
        "positive_annualized_return_each_period_and_horizon": bool(
            primary["annualized_return"].gt(0.0).all() and len(primary) == 6
        ),
        "positive_mean_net_trade_bps_each_period_and_horizon": bool(
            primary["mean_net_trade_bps"].gt(0.0).all() and len(primary) == 6
        ),
        "daily_return_bootstrap_lower_bound_above_zero_each_period_and_horizon": bool(
            absolute_bootstrap["ci_lower"].gt(0.0).all()
            and len(absolute_bootstrap) == 6
        ),
        "positive_return_each_quarter_each_period_and_horizon": bool(
            quarter["cumulative_return"].gt(0.0).all() and len(quarter) == 24
        ),
        "positive_return_under_every_leave_one_stock_out_deletion_each_period_and_horizon": bool(
            deletion["cumulative_return"].gt(0.0).all()
            and len(deletion) == 6 * UNIVERSE_SIZE
        ),
        "paired_daily_return_advantage_over_ungated_breakout_bootstrap_lower_bound_above_zero_each_period_and_horizon": bool(
            ungated_bootstrap["ci_lower"].gt(0.0).all()
            and len(ungated_bootstrap) == 6
        ),
        "paired_daily_return_advantage_over_state_context_gate_bootstrap_lower_bound_above_zero_each_period_and_horizon": bool(
            paired_bootstrap["ci_lower"].gt(0.0).all()
            and len(paired_bootstrap) == 6
        ),
        "paired_daily_return_advantage_over_raw_history_gate_bootstrap_lower_bound_above_zero_each_period_and_horizon": bool(
            history_bootstrap["ci_lower"].gt(0.0).all()
            and len(history_bootstrap) == 6
        ),
    }
    checks["pass"] = bool(all(checks.values()))
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "broker_connection_enabled": False,
        "paper_or_demo_execution_enabled": False,
        "primary_candidate": candidate,
        "primary_cost_bps_per_side": PRIMARY_COST,
        "checks": checks,
        "decision": (
            "development_pnl_hypothesis_retained_but_not_validated"
            if checks["pass"]
            else "pnl_translation_not_supported"
        ),
        "economic_edge_claim": False,
        "strategy_promotion": False,
        "prospective_validation_claim": False,
        "directional_falsification_can_promote": False,
    }


def main() -> None:
    pre_contract = json.loads(CONTRACT_PATH.read_text())
    fixed_universe = list(pre_contract["overlap_and_portfolio"]["fixed_universe"])
    predictions = {
        year: load_predictions(year)
        .loc[lambda frame: frame["symbol_norm"].isin(fixed_universe)]
        .reset_index(drop=True)
        for year in (2025, 2023)
    }
    symbols_2025 = sorted(predictions[2025]["symbol_norm"].unique())
    symbols_2023 = sorted(predictions[2023]["symbol_norm"].unique())
    if symbols_2025 != symbols_2023 or len(symbols_2025) != UNIVERSE_SIZE:
        raise AssertionError("common 20-stock universe drift")
    symbols = symbols_2025
    contract, pre_score = load_contract_and_verify_hashes(symbols)
    if symbols != contract["overlap_and_portfolio"]["fixed_universe"]:
        raise AssertionError("contract fixed-universe drift")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)

    thresholds = reconstruct_thresholds()
    thresholds.to_csv(OUT / "prediction_thresholds_2024.csv", index=False)
    execution_frames: dict[int, pd.DataFrame] = {}
    signal_frames: list[pd.DataFrame] = []
    sessions_by_period: dict[str, list[str]] = {}
    for year in (2025, 2023):
        tape = load_provider_tape(year, symbols)
        execution = attach_execution_prices(year, predictions[year], tape)
        execution_frames[year] = execution
        sessions_by_period[str(year)] = sorted(tape["session_date"].unique())
        signal_frames.append(build_signal_ledger(year, execution, thresholds))
        audit_columns = [
            "anchor_id",
            "symbol_norm",
            "session_date",
            "start_timestamp",
            "bar_ordinal",
            "tape_position",
            "anchor_open",
            "anchor_high",
            "anchor_low",
            "anchor_close",
        ]
        for horizon in HORIZONS:
            audit_columns.extend(
                [
                    f"next_open_{horizon}",
                    f"exit_close_{horizon}",
                    f"breakout_status_{horizon}",
                    f"breakout_direction_{horizon}",
                    f"breakout_entry_step_{horizon}",
                    f"breakout_entry_price_{horizon}",
                    f"breakout_exit_price_{horizon}",
                    f"breakout_gross_return_{horizon}",
                    f"breakout_gross_return_bps_{horizon}",
                    f"breakout_holding_bars_{horizon}",
                ]
            )
        execution.loc[:, audit_columns].to_parquet(
            OUT / f"execution_anchors_{year}.parquet", index=False
        )
    ledger = pd.concat(signal_frames, ignore_index=True)
    ledger.to_parquet(OUT / "accepted_signal_ledger.parquet", index=False)
    metrics, daily, quarters, deletions = evaluate_ledgers(
        ledger, sessions_by_period
    )
    bootstraps = bootstrap_primary(daily)
    decision = make_decision(metrics, quarters, deletions, bootstraps)
    metrics.to_csv(OUT / "pnl_metrics.csv", index=False)
    daily.to_parquet(OUT / "daily_portfolio_returns.parquet", index=False)
    quarters.to_csv(OUT / "quarter_metrics.csv", index=False)
    deletions.to_csv(OUT / "stock_deletion_metrics.csv", index=False)
    bootstraps.to_csv(OUT / "primary_bootstraps.csv", index=False)
    write_json(OUT / "decision.json", decision)
    write_json(
        OUT / "source_hashes.json",
        {**pre_score, "pre_score_manifest_sha256": sha256(PRE_SCORE_PATH)},
    )
    write_json(
        OUT / "execution_manifest.json",
        {
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "broker_connection_enabled": False,
            "paper_or_demo_execution_enabled": False,
            "provider_volume_label": "historical_volume_not_used",
            "symbols": symbols,
            "horizons": list(HORIZONS),
            "costs_bps_per_side": list(COSTS),
            "primary_cost_bps_per_side": PRIMARY_COST,
            "strategies": strategy_definitions(),
            "sessions_by_period": {
                period: len(sessions) for period, sessions in sessions_by_period.items()
            },
            "bid_ask_quotes_used": False,
            "market_impact_modeled": False,
            "short_locate_or_borrow_modeled": False,
            "overnight_positions": False,
        },
    )
    primary_rows = metrics.loc[
        metrics["cost_bps_per_side"].eq(PRIMARY_COST)
        & metrics["strategy"].isin(
            [
                "direction_loop_scores_all",
                "direction_loop_scores_range_p75",
                "breakout_all",
                "breakout_state_context_range_p75",
                "breakout_raw_history_range_p75",
                "breakout_loop_scores_range_p75",
            ]
        )
    ]
    summary = {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "broker_connection_enabled": False,
        "paper_or_demo_execution_enabled": False,
        "symbols": len(symbols),
        "prediction_rows": {str(year): len(frame) for year, frame in predictions.items()},
        "accepted_signal_rows": len(ledger),
        "thresholds": thresholds.to_dict(orient="records"),
        "primary_cost_metrics": primary_rows.to_dict(orient="records"),
        "primary_bootstraps": bootstraps.to_dict(orient="records"),
        "decision": decision,
    }
    write_json(OUT / "summary.json", summary)
    files = sorted(path for path in OUT.iterdir() if path.is_file())
    write_json(
        OUT / "artifact_manifest.json",
        {
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "files": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in files
            ],
        },
    )
    print(json.dumps(safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
