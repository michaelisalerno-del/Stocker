"""Research-only price-consequence test for frozen loop probabilities.

Fits outcome models on 2024 and scores 2025 plus backward-portability 2023.
No P&L, cost, order, broker, runtime, or deployment path is available.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
STATE_ROOT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
BACKWARD_ROOT = Path(
    "/private/tmp/stocker_sealed_backward_2023_complete_detector_20260710"
)
PATH_ROOT = Path("/private/tmp/stocker_causal_loop_prefix_path_forecast_20260710")
RAW_CURRENT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/"
    "instrument_type=stock"
)
RAW_BACKWARD = Path(
    "/private/tmp/stocker_eodhd_pre2024_intraday_20260710/"
    "source=eodhd/instrument_type=stock"
)
RUN_2024 = STATE_ROOT / "train_2024_filtered_runs.csv"
RUN_2025 = STATE_ROOT / "test_2025_filtered_runs.csv"
RUN_2023 = BACKWARD_ROOT / "backward_2023_filtered_runs.parquet"
PATH_PARAMETERS = PATH_ROOT / "model_parameters.npz"
PATH_GATES = PATH_ROOT / "gates.json"
CYCLE_SOURCE = STATE_ROOT / "fixed_cycle_shuffled_nulls.csv"
OUT = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710")

SEED = 20260710
K = 8
HORIZONS = (6, 12, 24)
MAX_START_BAR = 53
REPRESENTATIONS = ("state_context", "raw_history", "loop_scores")
CONTINUOUS_TARGETS = (
    "signed_return_bps",
    "absolute_return_bps",
    "future_range_bps",
)
PRICE_CONTROLS = (
    "current_bar_log_return",
    "return_sum_6",
    "mean_abs_return_12",
    "session_return",
    "bar_range_pct",
)
NUMERIC_CONTROLS = (
    "b0_entry_numeric",
    "b0_entry_high_stress",
    "entry_time_sin",
    "entry_time_cos",
    *PRICE_CONTROLS,
)
MIN_BIN_SUPPORT = 500
EPSILON = 1e-12


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("price_consequence_path_base", HERE / "run_causal_loop_prefix_path_forecast.py")


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provider_path(root: Path, symbol: str) -> Path:
    stored = "VTI.US" if symbol == "VTI" else symbol
    return root / f"symbol={stored}" / "timeframe=5m" / "data.parquet"


def prepare_symbol_prices(symbol: str, root: Path, year: int) -> pd.DataFrame:
    path = provider_path(root, symbol)
    frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
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
    if frame.empty or frame["timestamp"].duplicated().any():
        raise AssertionError(f"invalid provider tape for {symbol} {year}")
    frame["symbol_norm"] = symbol
    frame["bar_index_in_session"] = frame.groupby(
        "session_date", sort=False
    ).cumcount()
    grouped = frame.groupby("session_date", sort=False)
    previous_close = grouped["close"].shift(1)
    first_bar = frame["bar_index_in_session"].eq(0)
    frame["current_bar_log_return"] = np.log(
        frame["close"] / previous_close.where(~first_bar, frame["open"])
    )
    frame["return_sum_6"] = grouped["current_bar_log_return"].transform(
        lambda values: values.rolling(6, min_periods=1).sum()
    )
    frame["mean_abs_return_12"] = grouped["current_bar_log_return"].transform(
        lambda values: values.abs().rolling(12, min_periods=1).mean()
    )
    session_open = grouped["open"].transform("first")
    frame["session_return"] = np.log(frame["close"] / session_open)
    frame["bar_range_pct"] = (frame["high"] - frame["low"]) / frame["open"]

    timestamp_group = grouped["timestamp"]
    close_group = grouped["close"]
    high_group = grouped["high"]
    low_group = grouped["low"]
    for horizon in HORIZONS:
        future_close = close_group.shift(-horizon)
        exact = np.ones(len(frame), dtype=bool)
        highs = []
        lows = []
        for step in range(1, horizon + 1):
            shifted_time = timestamp_group.shift(-step)
            exact &= (
                (shifted_time - frame["timestamp"])
                .eq(pd.Timedelta(minutes=5 * step))
                .fillna(False)
                .to_numpy(bool)
            )
            highs.append(high_group.shift(-step).to_numpy(dtype=float))
            lows.append(low_group.shift(-step).to_numpy(dtype=float))
        signed = 10000.0 * np.log(future_close.to_numpy(float) / frame["close"].to_numpy(float))
        high_matrix = np.column_stack(highs)
        low_matrix = np.column_stack(lows)
        has_forward_values = np.isfinite(high_matrix).any(axis=1)
        forward_high = np.full(len(frame), np.nan, dtype=float)
        forward_low = np.full(len(frame), np.nan, dtype=float)
        forward_high[has_forward_values] = np.nanmax(
            high_matrix[has_forward_values], axis=1
        )
        forward_low[has_forward_values] = np.nanmin(
            low_matrix[has_forward_values], axis=1
        )
        future_range = (
            10000.0
            * (forward_high - forward_low)
            / frame["close"].to_numpy(dtype=float)
        )
        signed[~exact] = np.nan
        future_range[~exact] = np.nan
        frame[f"exact_{horizon}"] = exact
        frame[f"direction_{horizon}"] = np.where(
            exact, (signed > 0.0).astype(float), np.nan
        )
        frame[f"signed_return_bps_{horizon}"] = signed
        frame[f"absolute_return_bps_{horizon}"] = np.abs(signed)
        frame[f"future_range_bps_{horizon}"] = future_range
    keep = [
        "symbol_norm",
        "session_date",
        "timestamp",
        "bar_index_in_session",
        *PRICE_CONTROLS,
    ]
    for horizon in HORIZONS:
        keep.extend(
            [
                f"exact_{horizon}",
                f"direction_{horizon}",
                f"signed_return_bps_{horizon}",
                f"absolute_return_bps_{horizon}",
                f"future_range_bps_{horizon}",
            ]
        )
    return frame.loc[:, keep]


def prepare_price_panel(symbols: list[str], root: Path, year: int) -> pd.DataFrame:
    parts = [prepare_symbol_prices(symbol, root, year) for symbol in symbols]
    panel = pd.concat(parts, ignore_index=True).sort_values(
        ["symbol_norm", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    return panel


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def history_path_probability(
    anchors: pd.DataFrame,
    route: tuple[int, ...],
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    probability = np.ones(len(anchors), dtype=float)
    previous_state_2 = anchors["previous_state_2"].to_numpy(dtype=int)
    previous_state_1 = anchors["previous_state_1"].to_numpy(dtype=int)
    current_state = np.full(len(anchors), route[0], dtype=int)
    for destination in route[1:]:
        tokens = base.history_tokens(previous_state_2, previous_state_1, current_state)
        logits = (
            parameters["history_intercept"][None, :]
            + parameters["history_coef"][:, tokens].T
        )
        probability *= softmax(logits)[:, int(destination)]
        previous_state_2, previous_state_1, current_state = (
            previous_state_1,
            current_state,
            np.full(len(anchors), int(destination), dtype=int),
        )
    return probability


def add_loop_scores(
    anchors: pd.DataFrame,
    cycles: pd.DataFrame,
    parameters: dict[str, np.ndarray],
) -> pd.DataFrame:
    output = anchors.copy()
    for cycle_index, cycle in enumerate(cycles.itertuples(index=False), start=1):
        core = tuple(int(state) for state in cycle.core)
        values = np.zeros(len(output), dtype=float)
        for current in sorted(set(core)):
            mask = output["state"].eq(current).to_numpy()
            selected = output.loc[mask].reset_index(drop=True)
            probability = np.zeros(len(selected), dtype=float)
            for route in base.oriented_paths(core, current):
                probability += history_path_probability(selected, route, parameters)
            values[mask] = probability
        if values.max(initial=0.0) > 1.0 + 1e-9 or values.min(initial=0.0) < 0:
            raise AssertionError("invalid frozen loop score")
        output[f"loop_score_{cycle_index:02d}"] = np.clip(values, 0.0, 1.0)
    return output


def prepare_anchors(
    run_path: Path,
    raw_root: Path,
    year: int,
    period: str,
    cycles: pd.DataFrame,
    parameters: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    runs = base.load_runs(run_path, year, period)
    runs["start_timestamp"] = pd.to_datetime(
        runs["start_timestamp"], utc=True, errors="raise"
    )
    symbols = sorted(runs["symbol_norm"].astype(str).unique())
    prices = prepare_price_panel(symbols, raw_root, year)
    prices = prices.rename(columns={"timestamp": "start_timestamp"})
    keys = ["symbol_norm", "session_date", "start_timestamp"]
    if runs.duplicated(keys).any() or prices.duplicated(keys).any():
        raise AssertionError(f"duplicate price/run merge key in {period}")
    merged = runs.merge(prices, on=keys, how="inner", validate="one_to_one")
    exact = np.ones(len(merged), dtype=bool)
    for horizon in HORIZONS:
        exact &= merged[f"exact_{horizon}"].astype(bool).to_numpy()
    eligible = exact & merged["bar_index_in_session"].astype(int).le(MAX_START_BAR).to_numpy()
    output = merged.loc[eligible].copy().sort_values(
        ["symbol_norm", "session_date", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)
    if output.empty:
        raise AssertionError(f"empty price-consequence anchors for {period}")
    if output["bar_index_in_session"].astype(int).gt(MAX_START_BAR).any():
        raise AssertionError("session cutoff failure")
    if pd.to_datetime(output["session_date"]).dt.year.ne(year).any() or year >= 2026:
        raise AssertionError("year boundary failure")
    outcome_columns = []
    for horizon in HORIZONS:
        outcome_columns.extend(
            [
                f"direction_{horizon}",
                f"signed_return_bps_{horizon}",
                f"absolute_return_bps_{horizon}",
                f"future_range_bps_{horizon}",
            ]
        )
    if not np.isfinite(output[outcome_columns].to_numpy(dtype=float)).all():
        raise AssertionError("non-finite price outcome")
    output["history_token"] = base.history_tokens(
        output["previous_state_2"].to_numpy(dtype=int),
        output["previous_state_1"].to_numpy(dtype=int),
        output["state"].to_numpy(dtype=int),
    )
    output["anchor_id"] = np.arange(len(output), dtype=np.int64)
    output = add_loop_scores(output, cycles, parameters)
    audit = {
        "period": period,
        "year": year,
        "run_rows": len(runs),
        "provider_bar_rows": len(prices),
        "exact_merged_rows_before_cutoff": int(exact.sum()),
        "anchors": len(output),
        "symbols": len(symbols),
        "dates": int(output["session_date"].nunique()),
        "quarters": int(output["quarter"].nunique()),
        "minimum_start_bar": int(output["bar_index_in_session"].min()),
        "maximum_start_bar": int(output["bar_index_in_session"].max()),
    }
    return output, audit


def feature_matrices(
    train: pd.DataFrame,
    tests: dict[str, pd.DataFrame],
) -> tuple[
    dict[str, sparse.csr_matrix],
    dict[str, dict[str, sparse.csr_matrix]],
    dict[str, Any],
]:
    train_numeric = train.loc[:, list(NUMERIC_CONTROLS)].apply(
        pd.to_numeric, errors="coerce"
    )
    medians = train_numeric.median(axis=0)

    def build(frame: pd.DataFrame) -> dict[str, sparse.csr_matrix]:
        numeric = frame.loc[:, list(NUMERIC_CONTROLS)].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(medians)
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise AssertionError("non-finite causal control")
        current = frame["state"].to_numpy(dtype=int)
        state = sparse.csr_matrix(np.eye(K, dtype=np.float32)[current])
        context = sparse.hstack(
            (state, sparse.csr_matrix(numeric.to_numpy(dtype=np.float32))),
            format="csr",
        )
        history = base.token_matrix(frame["history_token"].to_numpy(dtype=int))
        loop_columns = [f"loop_score_{index:02d}" for index in range(1, 21)]
        loop = sparse.csr_matrix(frame[loop_columns].to_numpy(dtype=np.float32))
        return {
            "state_context": context,
            "raw_history": sparse.hstack((context, history), format="csr"),
            "loop_scores": sparse.hstack((context, loop), format="csr"),
        }

    raw_train = build(train)
    raw_tests = {period: build(frame) for period, frame in tests.items()}
    scaled_train: dict[str, sparse.csr_matrix] = {}
    scaled_tests: dict[str, dict[str, sparse.csr_matrix]] = {
        period: {} for period in tests
    }
    scalers: dict[str, StandardScaler] = {}
    for representation in REPRESENTATIONS:
        scaler = StandardScaler(with_mean=False)
        scaled_train[representation] = scaler.fit_transform(raw_train[representation]).tocsr()
        for period in tests:
            scaled_tests[period][representation] = scaler.transform(
                raw_tests[period][representation]
            ).tocsr()
        scalers[representation] = scaler
    audit = {
        "numeric_controls": list(NUMERIC_CONTROLS),
        "numeric_medians": medians.to_dict(),
        "feature_widths": {
            representation: raw_train[representation].shape[1]
            for representation in REPRESENTATIONS
        },
        "scalers": scalers,
    }
    return scaled_train, scaled_tests, audit


def fit_models(
    train: pd.DataFrame,
    tests: dict[str, pd.DataFrame],
    train_x: dict[str, sparse.csr_matrix],
    test_x: dict[str, dict[str, sparse.csr_matrix]],
    scalers: dict[str, StandardScaler],
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    predictions = {
        period: frame.loc[
            :,
            [
                "anchor_id",
                "symbol_norm",
                "session_date",
                "quarter",
                "start_timestamp",
                "state",
                "history_token",
            ],
        ].copy()
        for period, frame in tests.items()
    }
    model_parameters: dict[str, np.ndarray] = {}
    for representation, scaler in scalers.items():
        model_parameters[f"{representation}__scaler_scale"] = scaler.scale_.copy()
        model_parameters[f"{representation}__scaler_mean"] = scaler.mean_.copy()
        model_parameters[f"{representation}__scaler_var"] = scaler.var_.copy()
    for horizon in HORIZONS:
        train_direction = train[f"direction_{horizon}"].to_numpy(dtype=int)
        for period, frame in tests.items():
            predictions[period][f"direction_target_{horizon}"] = frame[
                f"direction_{horizon}"
            ].to_numpy(dtype=int)
            for target in CONTINUOUS_TARGETS:
                predictions[period][f"{target}_target_{horizon}"] = frame[
                    f"{target}_{horizon}"
                ].to_numpy(dtype=float)
        for representation in REPRESENTATIONS:
            direction_model = LogisticRegression(
                C=0.2,
                solver="lbfgs",
                max_iter=500,
                random_state=SEED,
            )
            direction_model.fit(train_x[representation], train_direction)
            positive_index = int(np.flatnonzero(direction_model.classes_ == 1)[0])
            prefix = f"{representation}__direction__h{horizon}"
            model_parameters[f"{prefix}__classes"] = direction_model.classes_.copy()
            model_parameters[f"{prefix}__coef"] = direction_model.coef_.copy()
            model_parameters[f"{prefix}__intercept"] = direction_model.intercept_.copy()
            model_parameters[f"{prefix}__n_iter"] = direction_model.n_iter_.copy()
            for period in tests:
                probability = direction_model.predict_proba(
                    test_x[period][representation]
                )[:, positive_index]
                predictions[period][
                    f"{representation}__direction_probability_{horizon}"
                ] = np.clip(probability, EPSILON, 1.0 - EPSILON)
            for target in CONTINUOUS_TARGETS:
                ridge = Ridge(alpha=10.0, solver="lsqr")
                ridge.fit(
                    train_x[representation],
                    train[f"{target}_{horizon}"].to_numpy(dtype=float),
                )
                ridge_prefix = f"{representation}__{target}__h{horizon}"
                model_parameters[f"{ridge_prefix}__coef"] = ridge.coef_.copy()
                model_parameters[f"{ridge_prefix}__intercept"] = np.asarray(
                    [ridge.intercept_]
                )
                for period in tests:
                    predictions[period][
                        f"{representation}__{target}_prediction_{horizon}"
                    ] = ridge.predict(test_x[period][representation])
    return predictions, model_parameters


def direction_losses(target: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    probability = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return {
        "log_loss": -(
            target * np.log(probability)
            + (1.0 - target) * np.log(1.0 - probability)
        ),
        "brier": np.square(probability - target),
    }


def calibration(
    period: str,
    representation: str,
    horizon: int,
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[list[dict[str, Any]], float, float]:
    bin_id = np.minimum((probability * 10.0).astype(int), 9)
    rows = []
    ece = 0.0
    supported_errors = []
    for index in range(10):
        mask = bin_id == index
        count = int(mask.sum())
        mean_probability = float(probability[mask].mean()) if count else math.nan
        event_rate = float(target[mask].mean()) if count else math.nan
        error = abs(mean_probability - event_rate) if count else math.nan
        if count:
            ece += count / len(target) * error
        supported = count >= MIN_BIN_SUPPORT
        if supported:
            supported_errors.append(error)
        rows.append(
            {
                "period": period,
                "representation": representation,
                "horizon": horizon,
                "bin": index,
                "count": count,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "absolute_error": error,
                "supported": supported,
            }
        )
    return rows, float(ece), max(supported_errors) if supported_errors else math.nan


def daily_decile_spread(
    frame: pd.DataFrame, prediction: np.ndarray, outcome: np.ndarray
) -> pd.DataFrame:
    working = pd.DataFrame(
        {
            "session_date": frame["session_date"].astype(str).to_numpy(),
            "prediction": prediction,
            "outcome": outcome,
        }
    )
    rows = []
    for session_date, group in working.groupby("session_date", sort=True):
        ordered = group.sort_values("prediction", kind="stable")
        count = max(1, int(math.floor(0.10 * len(ordered))))
        spread = float(ordered.iloc[-count:]["outcome"].mean() - ordered.iloc[:count]["outcome"].mean())
        rows.append({"session_date": session_date, "spread": spread, "tail_count": count})
    return pd.DataFrame(rows)


def paired_comparison(
    prediction_frame: pd.DataFrame,
    period: str,
    candidate: str,
    baseline_representation: str,
    target_name: str,
    loss_name: str,
    candidate_loss: np.ndarray,
    baseline_loss: np.ndarray,
    seed_offset: int,
) -> dict[str, Any]:
    difference = candidate_loss - baseline_loss
    daily = (
        pd.DataFrame(
            {
                "session_date": prediction_frame["session_date"],
                "difference": difference,
            }
        )
        .groupby("session_date", sort=True)["difference"]
        .mean()
        .to_numpy(dtype=float)
    )
    mean, low, high = base.moving_block_bounds(daily, SEED + seed_offset)
    horizons = pd.Series(difference).groupby(
        prediction_frame["horizon"].reset_index(drop=True)
    ).mean()
    quarters = pd.Series(difference).groupby(
        prediction_frame["quarter"].reset_index(drop=True)
    ).mean()
    deletions = {
        symbol: float(
            difference[
                prediction_frame["symbol_norm"].astype(str).ne(symbol).to_numpy()
            ].mean()
        )
        for symbol in sorted(prediction_frame["symbol_norm"].astype(str).unique())
    }
    baseline_mean = float(baseline_loss.mean())
    return {
        "period": period,
        "candidate": candidate,
        "baseline": baseline_representation,
        "target": target_name,
        "loss": loss_name,
        "row_mean_difference": float(difference.mean()),
        "daily_mean_difference": mean,
        "daily_ci_low": low,
        "daily_ci_high": high,
        "baseline_mean_loss": baseline_mean,
        "relative_improvement": float(-difference.mean() / baseline_mean),
        "negative_horizon_count": int((horizons < 0.0).sum()),
        "negative_quarter_count": int((quarters < 0.0).sum()),
        "leave_one_symbol_max_difference": max(deletions.values()),
        "leave_one_symbol_all_negative": bool(max(deletions.values()) < 0.0),
    }


def evaluate_period(predictions: pd.DataFrame, period: str, seed_offset: int) -> dict[str, Any]:
    long_parts = []
    for horizon in HORIZONS:
        part = predictions.loc[
            :,
            ["anchor_id", "symbol_norm", "session_date", "quarter", "state"],
        ].copy()
        part["horizon"] = horizon
        part["direction_target"] = predictions[f"direction_target_{horizon}"]
        for target in CONTINUOUS_TARGETS:
            part[f"{target}_target"] = predictions[f"{target}_target_{horizon}"]
        for representation in REPRESENTATIONS:
            part[f"{representation}__direction_probability"] = predictions[
                f"{representation}__direction_probability_{horizon}"
            ]
            for target in CONTINUOUS_TARGETS:
                part[f"{representation}__{target}_prediction"] = predictions[
                    f"{representation}__{target}_prediction_{horizon}"
                ]
        long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True).sort_values(
        ["anchor_id", "horizon"], kind="stable"
    ).reset_index(drop=True)

    direction_metric_rows = []
    continuous_metric_rows = []
    calibration_rows = []
    decile_rows = []
    direction_loss_arrays: dict[str, dict[str, np.ndarray]] = {}
    continuous_loss_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {
        target: {} for target in CONTINUOUS_TARGETS
    }
    calibration_summary: dict[str, dict[int, tuple[float, float]]] = {
        representation: {} for representation in REPRESENTATIONS
    }
    correlations: dict[str, dict[str, dict[int, float]]] = {
        target: {representation: {} for representation in REPRESENTATIONS}
        for target in CONTINUOUS_TARGETS
    }
    decile_summary: dict[str, dict[int, tuple[float, float, float]]] = {
        representation: {} for representation in REPRESENTATIONS
    }

    direction_target = long["direction_target"].to_numpy(dtype=int)
    for representation in REPRESENTATIONS:
        probability = long[
            f"{representation}__direction_probability"
        ].to_numpy(dtype=float)
        direction_loss_arrays[representation] = direction_losses(
            direction_target, probability
        )
        for horizon in HORIZONS:
            mask = long["horizon"].eq(horizon).to_numpy()
            target_h = direction_target[mask]
            probability_h = probability[mask]
            rows, ece, maximum = calibration(
                period, representation, horizon, target_h, probability_h
            )
            calibration_rows.extend(rows)
            calibration_summary[representation][horizon] = (ece, maximum)
            direction_metric_rows.append(
                {
                    "period": period,
                    "representation": representation,
                    "horizon": horizon,
                    "anchors": int(mask.sum()),
                    "positive_rate": float(target_h.mean()),
                    "log_loss": float(
                        direction_loss_arrays[representation]["log_loss"][mask].mean()
                    ),
                    "brier": float(
                        direction_loss_arrays[representation]["brier"][mask].mean()
                    ),
                    "auc": float(roc_auc_score(target_h, probability_h)),
                    "ece": ece,
                    "maximum_supported_bin_error": maximum,
                }
            )

    for target in CONTINUOUS_TARGETS:
        outcome = long[f"{target}_target"].to_numpy(dtype=float)
        for representation in REPRESENTATIONS:
            prediction = long[
                f"{representation}__{target}_prediction"
            ].to_numpy(dtype=float)
            continuous_loss_arrays[target][representation] = {
                "mse": np.square(prediction - outcome),
                "mae": np.abs(prediction - outcome),
            }
            for horizon in HORIZONS:
                mask = long["horizon"].eq(horizon).to_numpy()
                correlation = float(
                    np.corrcoef(prediction[mask], outcome[mask])[0, 1]
                )
                correlations[target][representation][horizon] = correlation
                continuous_metric_rows.append(
                    {
                        "period": period,
                        "representation": representation,
                        "target": target,
                        "horizon": horizon,
                        "anchors": int(mask.sum()),
                        "outcome_mean": float(outcome[mask].mean()),
                        "prediction_mean": float(prediction[mask].mean()),
                        "mse": float(
                            continuous_loss_arrays[target][representation]["mse"][mask].mean()
                        ),
                        "mae": float(
                            continuous_loss_arrays[target][representation]["mae"][mask].mean()
                        ),
                        "pearson_correlation": correlation,
                    }
                )
                if target == "signed_return_bps":
                    daily = daily_decile_spread(
                        long.loc[mask], prediction[mask], outcome[mask]
                    )
                    mean, low, high = base.moving_block_bounds(
                        daily["spread"].to_numpy(dtype=float),
                        SEED
                        + seed_offset
                        + 7000
                        + REPRESENTATIONS.index(representation) * 100
                        + horizon,
                    )
                    decile_summary[representation][horizon] = (mean, low, high)
                    decile_rows.append(
                        {
                            "period": period,
                            "representation": representation,
                            "horizon": horizon,
                            "dates": len(daily),
                            "mean_top_minus_bottom_bps": mean,
                            "daily_ci_low": low,
                            "daily_ci_high": high,
                        }
                    )

    comparison_rows = []
    comparison_specs = (
        ("loop_scores", "state_context"),
        ("raw_history", "state_context"),
        ("loop_scores", "raw_history"),
    )
    comparison_counter = 0
    for candidate, baseline_representation in comparison_specs:
        for loss_name in ("log_loss", "brier"):
            comparison_rows.append(
                paired_comparison(
                    long,
                    period,
                    candidate,
                    baseline_representation,
                    "direction",
                    loss_name,
                    direction_loss_arrays[candidate][loss_name],
                    direction_loss_arrays[baseline_representation][loss_name],
                    seed_offset + comparison_counter,
                )
            )
            comparison_counter += 1
        for target in CONTINUOUS_TARGETS:
            for loss_name in ("mse", "mae"):
                comparison_rows.append(
                    paired_comparison(
                        long,
                        period,
                        candidate,
                        baseline_representation,
                        target,
                        loss_name,
                        continuous_loss_arrays[target][candidate][loss_name],
                        continuous_loss_arrays[target][baseline_representation][loss_name],
                        seed_offset + comparison_counter,
                    )
                )
                comparison_counter += 1
    comparisons = pd.DataFrame(comparison_rows)

    def paired_gate(target: str, primary: str, secondary: str) -> dict[str, Any]:
        rows = comparisons.loc[
            comparisons["candidate"].eq("loop_scores")
            & comparisons["baseline"].eq("state_context")
            & comparisons["target"].eq(target)
            & comparisons["loss"].isin([primary, secondary])
        ].set_index("loss")
        robust = bool(
            rows["daily_ci_high"].lt(0.0).all()
            and rows["negative_horizon_count"].eq(3).all()
            and rows["negative_quarter_count"].eq(4).all()
            and rows["leave_one_symbol_all_negative"].all()
        )
        return {
            "intervals_horizons_quarters_deletions_pass": robust,
            "relative_primary_improvement": float(
                rows.loc[primary, "relative_improvement"]
            ),
            "relative_primary_pass": bool(
                rows.loc[primary, "relative_improvement"] >= 0.0025
            ),
        }

    direction_gate = paired_gate("direction", "log_loss", "brier")
    direction_metrics = pd.DataFrame(direction_metric_rows)
    loop_direction = direction_metrics.loc[
        direction_metrics["representation"].eq("loop_scores")
    ].set_index("horizon")
    state_direction = direction_metrics.loc[
        direction_metrics["representation"].eq("state_context")
    ].set_index("horizon")
    auc_pass = bool(
        loop_direction["auc"].ge(0.52).all()
        and (loop_direction["auc"] > state_direction["auc"]).all()
    )
    ece_pass = all(
        calibration_summary["loop_scores"][horizon][0]
        <= calibration_summary["state_context"][horizon][0]
        for horizon in HORIZONS
    )
    max_error_pass = all(
        calibration_summary["loop_scores"][horizon][1]
        <= calibration_summary["state_context"][horizon][1] + 0.01
        for horizon in HORIZONS
    )
    direction_gate.update(
        {
            "auc_pass": auc_pass,
            "ece_pass": ece_pass,
            "maximum_supported_bin_error_pass": max_error_pass,
        }
    )
    direction_gate["pass"] = bool(
        direction_gate["intervals_horizons_quarters_deletions_pass"]
        and direction_gate["relative_primary_pass"]
        and auc_pass
        and ece_pass
        and max_error_pass
    )

    signed_gate = paired_gate("signed_return_bps", "mse", "mae")
    correlation_pass = all(
        correlations["signed_return_bps"]["loop_scores"][horizon] >= 0.01
        for horizon in HORIZONS
    )
    decile_pass = all(
        decile_summary["loop_scores"][horizon][1] > 0.0
        for horizon in HORIZONS
    )
    signed_gate.update(
        {
            "correlation_pass": correlation_pass,
            "decile_spread_pass": decile_pass,
        }
    )
    signed_gate["pass"] = bool(
        signed_gate["intervals_horizons_quarters_deletions_pass"]
        and signed_gate["relative_primary_pass"]
        and correlation_pass
        and decile_pass
    )

    movement_gates = {}
    for target in ("absolute_return_bps", "future_range_bps"):
        gate = paired_gate(target, "mse", "mae")
        correlation_better = all(
            correlations[target]["loop_scores"][horizon]
            > correlations[target]["state_context"][horizon]
            for horizon in HORIZONS
        )
        gate["correlation_better_all_horizons"] = correlation_better
        gate["pass"] = bool(
            gate["intervals_horizons_quarters_deletions_pass"]
            and gate["relative_primary_pass"]
            and correlation_better
        )
        movement_gates[target] = gate

    representation_check = {}
    for target, primary in (
        ("direction", "log_loss"),
        ("signed_return_bps", "mse"),
        ("absolute_return_bps", "mse"),
        ("future_range_bps", "mse"),
    ):
        row = comparisons.loc[
            comparisons["candidate"].eq("loop_scores")
            & comparisons["baseline"].eq("raw_history")
            & comparisons["target"].eq(target)
            & comparisons["loss"].eq(primary)
        ].iloc[0]
        representation_check[target] = {
            "loop_minus_history_mean_loss": float(row["row_mean_difference"]),
            "loop_relative_to_history": float(row["relative_improvement"]),
            "loop_not_more_than_0_25pct_worse": bool(
                row["relative_improvement"] >= -0.0025
            ),
        }

    anchor_support = predictions[
        ["anchor_id", "symbol_norm", "session_date", "quarter", "state"]
    ].drop_duplicates("anchor_id")
    support = {
        "anchors": len(anchor_support),
        "stocks": int(anchor_support["symbol_norm"].nunique()),
        "dates": int(anchor_support["session_date"].nunique()),
        "quarters": int(anchor_support["quarter"].nunique()),
        "states": int(anchor_support["state"].nunique()),
        "direction_classes_each_horizon": bool(
            all(
                set(predictions[f"direction_target_{horizon}"].unique()) == {0, 1}
                for horizon in HORIZONS
            )
        ),
    }
    support["pass"] = bool(
        support["anchors"] >= 60_000
        and support["stocks"] >= 18
        and support["dates"] >= 200
        and support["quarters"] == 4
        and support["states"] == K
        and support["direction_classes_each_horizon"]
    )
    return {
        "long_predictions": long,
        "direction_metrics": pd.DataFrame(direction_metric_rows),
        "continuous_metrics": pd.DataFrame(continuous_metric_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "decile_spreads": pd.DataFrame(decile_rows),
        "comparisons": comparisons,
        "support": support,
        "direction_gate": direction_gate,
        "signed_return_gate": signed_gate,
        "movement_gates": movement_gates,
        "representation_check": representation_check,
    }


def self_tests() -> None:
    assert base.canonical_cycle((2, 5, 1)) == base.canonical_cycle((5, 1, 2))
    close = np.asarray([100.0, 101.0])
    signed = 10000.0 * np.log(close[1] / close[0])
    assert signed > 0.0 and abs(signed - 99.5033) < 1e-3
    example = pd.DataFrame(
        {
            "session_date": ["2025-01-02"] * 20,
        }
    )
    prediction = np.arange(20, dtype=float)
    outcome = np.arange(20, dtype=float)
    spread = daily_decile_spread(example, prediction, outcome)
    assert len(spread) == 1 and spread.iloc[0]["tail_count"] == 2
    assert spread.iloc[0]["spread"] == 18.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    self_tests()
    if args.self_test_only:
        print("self-tests passed")
        return

    path_gates = json.loads(PATH_GATES.read_text())
    if path_gates.get("history_retained") is not True:
        raise AssertionError("frozen path predecessor not retained")
    OUT.mkdir(parents=True, exist_ok=True)
    parameters = dict(np.load(PATH_PARAMETERS))
    cycles = base.load_cycles()
    cycles.drop(columns="core").to_csv(OUT / "fixed_cycles.csv", index=False)

    period_specs = {
        "train_2024": (RUN_2024, RAW_CURRENT, 2024),
        "2025": (RUN_2025, RAW_CURRENT, 2025),
        "2023": (RUN_2023, RAW_BACKWARD, 2023),
    }
    source_paths = {
        "path_model_parameters.npz": PATH_PARAMETERS,
        "path_gates.json": PATH_GATES,
        "fixed_cycle_shuffled_nulls.csv": CYCLE_SOURCE,
        "train_2024_filtered_runs.csv": RUN_2024,
        "test_2025_filtered_runs.csv": RUN_2025,
        "backward_2023_filtered_runs.parquet": RUN_2023,
    }
    for period, (run_path, raw_root, year) in period_specs.items():
        runs = base.load_runs(run_path, year, period)
        for symbol in sorted(runs["symbol_norm"].astype(str).unique()):
            source_paths[f"provider_{year}_{symbol}.parquet"] = provider_path(
                raw_root, symbol
            )
    write_json(
        OUT / "source_hashes.json",
        {name: sha256(path) for name, path in source_paths.items()},
    )

    panels = {}
    panel_audits = []
    for period, (run_path, raw_root, year) in period_specs.items():
        panel, audit = prepare_anchors(
            run_path, raw_root, year, period, cycles, parameters
        )
        panels[period] = panel
        panel_audits.append(audit)
        panel.to_parquet(OUT / f"anchor_panel_{period}.parquet", index=False)
    pd.DataFrame(panel_audits).to_csv(OUT / "panel_audit.csv", index=False)

    train_x, test_x, feature_audit = feature_matrices(
        panels["train_2024"], {"2025": panels["2025"], "2023": panels["2023"]}
    )
    predictions, model_parameters = fit_models(
        panels["train_2024"],
        {"2025": panels["2025"], "2023": panels["2023"]},
        train_x,
        test_x,
        feature_audit["scalers"],
    )
    np.savez_compressed(OUT / "outcome_model_parameters.npz", **model_parameters)
    write_json(
        OUT / "feature_manifest.json",
        {
            "representations": list(REPRESENTATIONS),
            "numeric_controls": list(NUMERIC_CONTROLS),
            "numeric_medians": feature_audit["numeric_medians"],
            "feature_widths": feature_audit["feature_widths"],
            "loop_score_columns": [
                f"loop_score_{index:02d}" for index in range(1, 21)
            ],
            "horizons": list(HORIZONS),
            "direction_model": {
                "class": "LogisticRegression",
                "C": 0.2,
                "solver": "lbfgs",
                "max_iter": 500,
                "random_state": SEED,
            },
            "continuous_model": {
                "class": "Ridge",
                "alpha": 10.0,
                "solver": "lsqr",
            },
            "volume_label": "historical_volume_not_used",
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
        },
    )

    all_direction = []
    all_continuous = []
    all_calibration = []
    all_deciles = []
    all_comparisons = []
    gates = {"periods": {}}
    for period, seed_offset in (("2025", 1000), ("2023", 2000)):
        predictions[period].to_parquet(
            OUT / f"price_predictions_{period}.parquet", index=False
        )
        evaluation = evaluate_period(predictions[period], period, seed_offset)
        evaluation["long_predictions"].to_parquet(
            OUT / f"price_scoring_long_{period}.parquet", index=False
        )
        all_direction.append(evaluation["direction_metrics"])
        all_continuous.append(evaluation["continuous_metrics"])
        all_calibration.append(evaluation["calibration"])
        all_deciles.append(evaluation["decile_spreads"])
        all_comparisons.append(evaluation["comparisons"])
        gates["periods"][period] = {
            "support": evaluation["support"],
            "direction": evaluation["direction_gate"],
            "signed_return": evaluation["signed_return_gate"],
            "movement": evaluation["movement_gates"],
            "representation_check": evaluation["representation_check"],
        }
    direction_metrics = pd.concat(all_direction, ignore_index=True)
    continuous_metrics = pd.concat(all_continuous, ignore_index=True)
    calibration_frame = pd.concat(all_calibration, ignore_index=True)
    decile_frame = pd.concat(all_deciles, ignore_index=True)
    comparison_frame = pd.concat(all_comparisons, ignore_index=True)
    direction_metrics.to_csv(OUT / "direction_metrics.csv", index=False)
    continuous_metrics.to_csv(OUT / "continuous_metrics.csv", index=False)
    calibration_frame.to_csv(OUT / "direction_calibration.csv", index=False)
    decile_frame.to_csv(OUT / "signed_return_decile_spreads.csv", index=False)
    comparison_frame.to_csv(OUT / "paired_comparisons.csv", index=False)

    gates["directional_consequence_retained"] = bool(
        all(gates["periods"][period]["support"]["pass"] for period in ("2025", "2023"))
        and all(gates["periods"][period]["direction"]["pass"] for period in ("2025", "2023"))
        and all(gates["periods"][period]["signed_return"]["pass"] for period in ("2025", "2023"))
    )
    gates["absolute_movement_retained"] = bool(
        all(gates["periods"][period]["support"]["pass"] for period in ("2025", "2023"))
        and all(
            gates["periods"][period]["movement"]["absolute_return_bps"]["pass"]
            for period in ("2025", "2023")
        )
    )
    gates["range_movement_retained"] = bool(
        all(gates["periods"][period]["support"]["pass"] for period in ("2025", "2023"))
        and all(
            gates["periods"][period]["movement"]["future_range_bps"]["pass"]
            for period in ("2025", "2023")
        )
    )
    gates["movement_consequence_retained"] = bool(
        gates["absolute_movement_retained"] or gates["range_movement_retained"]
    )
    gates["economic_edge_claim"] = False
    gates["research_only"] = True
    gates["live_ordering_enabled"] = False
    gates["order_placement"] = "disabled"
    write_json(OUT / "gates.json", gates)
    summary = {
        "algorithm": "frozen_loop_score_incremental_price_consequence",
        "training_period": 2024,
        "scoring_periods": [2025, 2023],
        "horizons": list(HORIZONS),
        "gates": gates,
        "direction_metrics": direction_metrics.to_dict(orient="records"),
        "continuous_metrics": continuous_metrics.to_dict(orient="records"),
        "interpretation": (
            "Predictive-information research only. No P&L, costs, order, strategy, "
            "tradability, or deployment claim."
        ),
    }
    write_json(OUT / "summary.json", summary)
    lines = [
        "# Frozen loop-score price-consequence test",
        "",
        f"- Directional consequence retained: {gates['directional_consequence_retained']}",
        f"- Movement consequence retained: {gates['movement_consequence_retained']}",
        "- Economic edge claim: false.",
        "- Research only; live ordering and order placement disabled.",
        "",
        "## Direction metrics",
        "",
        "```text",
        direction_metrics.to_string(index=False),
        "```",
        "",
        "## Continuous metrics",
        "",
        "```text",
        continuous_metrics.to_string(index=False),
        "```",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps(safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
