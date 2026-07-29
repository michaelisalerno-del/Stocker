"""Frozen, research-only inference core for loop-movement shadow validation.

This module contains no outcome evaluation, order, broker, position, P&L, or
deployment surface.  It reproduces the causal feature/state/path/movement
inference lineage without fitting or calibrating any parameter.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


K = 8
END_STATE = K
MAX_DURATION = 24
HORIZONS = (6, 12, 24)
TOKEN_COUNT = (K + 1) * (K + 1) * K
EMISSION_FEATURES = (
    "regime_log_activity_3",
    "regime_log_activity_12",
    "regime_activity_acceleration",
    "signed_efficiency_6",
    "signed_efficiency_12",
    "regime_log_bar_range",
    "close_location_value",
    "regime_wick_balance",
    "log_relative_historical_volume",
    "log_relative_cumulative_historical_volume",
    "regime_log_market_dispersion",
    "regime_stock_minus_market_scaled",
    "vti__signed_efficiency_12",
    "regime_market_breadth_centered",
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
REPRESENTATIONS = ("state_context", "loop_scores")
MOVEMENT_TARGETS = ("absolute_return_bps", "future_range_bps")


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


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provider_path(root: Path, symbol: str) -> Path:
    stored = "VTI.US" if symbol == "VTI" else symbol
    return root / f"symbol={stored}" / "timeframe=5m" / "data.parquet"


def _read_provider(
    symbol: str,
    root: Path,
    start: pd.Timestamp,
    as_of: pd.Timestamp,
    columns: list[str],
) -> pd.DataFrame:
    path = provider_path(root, symbol)
    if not path.is_file():
        raise FileNotFoundError(path)
    filters = [
        ("timestamp", ">=", start.to_pydatetime()),
        ("timestamp", "<=", as_of.to_pydatetime()),
    ]
    frame = pd.read_parquet(path, columns=columns, filters=filters)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].gt(as_of).any():
        raise AssertionError(f"provider filter admitted a future row for {symbol}")
    return frame


def _rolling_feature(
    frame: pd.DataFrame,
    column: str,
    window: int,
    operation: str,
    min_periods: int = 1,
) -> pd.Series:
    grouped = frame.groupby("session_date", sort=False)[column]
    if operation == "sum":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=min_periods).sum()
        )
    if operation == "mean":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=min_periods).mean()
        )
    if operation == "std":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=min_periods).std()
        )
    raise ValueError(operation)


def prepare_symbol_bars(
    symbol: str,
    root: Path,
    start: pd.Timestamp,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Build the exact causal bar features, never returning a post-as-of row."""

    frame = _read_provider(
        symbol,
        root,
        start,
        as_of,
        ["timestamp", "open", "high", "low", "close", "volume"],
    )
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    frame = frame.loc[minute.ge(570) & minute.lt(960)].copy()
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    frame["session_date"] = local.dt.strftime("%Y-%m-%d")
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if frame.empty or frame["timestamp"].duplicated().any():
        raise AssertionError(f"invalid regular-session tape for {symbol}")

    frame["symbol_norm"] = symbol
    frame["bar_index_in_session"] = frame.groupby(
        "session_date", sort=False
    ).cumcount()
    group = frame.groupby("session_date", sort=False)
    previous_close = group["close"].shift(1)
    first_bar = frame["bar_index_in_session"].eq(0)
    frame["bar_log_return"] = np.log(
        frame["close"] / previous_close.where(~first_bar, frame["open"])
    )
    frame["current_bar_log_return"] = frame["bar_log_return"]
    frame["abs_bar_log_return"] = frame["bar_log_return"].abs()
    denominator = frame["high"] - frame["low"]
    frame["bar_range_pct"] = denominator / frame["open"]
    frame["close_location_value"] = (
        2.0 * frame["close"] - frame["high"] - frame["low"]
    ) / denominator.replace(0.0, np.nan)
    frame["upper_wick_pct_of_range"] = (
        frame["high"] - frame[["open", "close"]].max(axis=1)
    ) / denominator.replace(0.0, np.nan)
    frame["lower_wick_pct_of_range"] = (
        frame[["open", "close"]].min(axis=1) - frame["low"]
    ) / denominator.replace(0.0, np.nan)
    for window in (3, 6, 12):
        return_sum = _rolling_feature(frame, "bar_log_return", window, "sum")
        absolute_sum = _rolling_feature(frame, "abs_bar_log_return", window, "sum")
        frame[f"return_sum_{window}"] = return_sum
        frame[f"mean_abs_return_{window}"] = _rolling_feature(
            frame, "abs_bar_log_return", window, "mean"
        )
        frame[f"return_std_{window}"] = _rolling_feature(
            frame, "bar_log_return", window, "std", min_periods=2
        )
        frame[f"signed_efficiency_{window}"] = return_sum / absolute_sum.replace(
            0.0, np.nan
        )

    session_open = group["open"].transform("first")
    frame["session_return"] = np.log(frame["close"] / session_open)
    frame["cumulative_historical_volume"] = group["volume"].cumsum()
    daily = group.agg(
        session_open=("open", "first"),
        session_high=("high", "max"),
        session_low=("low", "min"),
        session_close=("close", "last"),
        session_historical_volume=("volume", "sum"),
    ).reset_index()
    daily["session_return_daily"] = np.log(
        daily["session_close"] / daily["session_open"]
    )
    daily["session_range_daily"] = (
        daily["session_high"] - daily["session_low"]
    ) / daily["session_open"]
    daily["prior_session_close"] = daily["session_close"].shift(1)
    daily["prior_session_return"] = daily["session_return_daily"].shift(1)
    daily["prior_session_range"] = daily["session_range_daily"].shift(1)
    daily["prior_session_historical_volume"] = daily[
        "session_historical_volume"
    ].shift(1)
    daily["prior_volume_baseline"] = (
        daily["session_historical_volume"].expanding(min_periods=10).mean().shift(1)
    )
    daily["prior_session_log_relative_volume"] = np.log1p(
        daily["prior_session_historical_volume"]
        / daily["prior_volume_baseline"].replace(0.0, np.nan)
    )
    frame = frame.merge(
        daily[
            [
                "session_date",
                "prior_session_close",
                "prior_session_return",
                "prior_session_range",
                "prior_session_log_relative_volume",
            ]
        ],
        on="session_date",
        how="left",
        validate="many_to_one",
    )
    frame["gap_log_return"] = np.log(
        session_open.to_numpy(float)
        / pd.to_numeric(frame["prior_session_close"], errors="coerce").to_numpy(float)
    )
    frame["historical_volume_baseline_at_bar"] = frame.groupby(
        "bar_index_in_session", sort=False
    )["volume"].transform(
        lambda values: values.expanding(min_periods=10).mean().shift(1)
    )
    frame["historical_cumulative_volume_baseline_at_bar"] = frame.groupby(
        "bar_index_in_session", sort=False
    )["cumulative_historical_volume"].transform(
        lambda values: values.expanding(min_periods=10).mean().shift(1)
    )
    frame["log_relative_historical_volume"] = np.log1p(
        frame["volume"]
        / frame["historical_volume_baseline_at_bar"].replace(0.0, np.nan)
    )
    frame["log_relative_cumulative_historical_volume"] = np.log1p(
        frame["cumulative_historical_volume"]
        / frame["historical_cumulative_volume_baseline_at_bar"].replace(0.0, np.nan)
    )
    return frame


def add_market_features(panel: pd.DataFrame, vti: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    grouped = frame.groupby("timestamp", sort=False)
    for source in (
        "bar_log_return",
        "return_sum_6",
        "return_sum_12",
        "bar_range_pct",
        "log_relative_historical_volume",
    ):
        frame[f"market_median__{source}"] = grouped[source].transform("median")
    frame["market_breadth_bar_positive"] = grouped["bar_log_return"].transform(
        lambda values: float((values > 0.0).mean())
    )
    frame["market_breadth_return_6_positive"] = grouped["return_sum_6"].transform(
        lambda values: float((values > 0.0).mean())
    )
    frame["market_dispersion_return_6"] = grouped["return_sum_6"].transform("std")
    frame["stock_minus_market_return_6"] = (
        frame["return_sum_6"] - frame["market_median__return_sum_6"]
    )
    keep = [
        "timestamp",
        "bar_log_return",
        "return_sum_6",
        "return_sum_12",
        "signed_efficiency_12",
        "bar_range_pct",
        "log_relative_historical_volume",
    ]
    vti_features = vti[keep].rename(
        columns={column: f"vti__{column}" for column in keep if column != "timestamp"}
    )
    before = len(frame)
    frame = frame.merge(vti_features, on="timestamp", how="left", validate="many_to_one")
    if len(frame) != before:
        raise AssertionError("VTI merge changed panel row count")
    frame["stock_minus_vti_return_6"] = (
        frame["return_sum_6"] - frame["vti__return_sum_6"]
    )
    return frame


def causal_rank_score(series: pd.Series, min_periods: int = 5) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    history: list[float] = []
    for index, value in enumerate(values):
        if not math.isfinite(value):
            continue
        history.append(value)
        if len(history) < min_periods:
            continue
        sample = np.asarray(history, dtype=float)
        percentile = (
            float((sample < value).sum()) + 0.5 * float((sample == value).sum())
        ) / len(sample)
        output[index] = (percentile - 0.5) * 2.0
    return pd.Series(output, index=series.index)


def confirm_states(
    raw_states: list[str], confirm_sessions: int = 5, min_hold_sessions: int = 15
) -> list[str]:
    current = "unknown"
    held = 0
    pending: str | None = None
    pending_count = 0
    output: list[str] = []
    for raw_state in raw_states:
        raw = raw_state or "unknown"
        if current == "unknown" and raw != "unknown":
            current, held, pending, pending_count = raw, 1, None, 0
        elif raw == current or raw == "unknown":
            held += 1
            pending, pending_count = None, 0
        elif held < min_hold_sessions:
            held += 1
        else:
            if pending == raw:
                pending_count += 1
            else:
                pending, pending_count = raw, 1
            if pending_count >= confirm_sessions:
                current, held, pending, pending_count = raw, 1, None, 0
            else:
                held += 1
        output.append(current)
    return output


def build_causal_b0(
    symbols: list[str], root: Path, start: pd.Timestamp, as_of: pd.Timestamp
) -> pd.DataFrame:
    daily_parts: list[pd.DataFrame] = []
    for symbol in symbols:
        bars = _read_provider(
            symbol,
            root,
            start,
            as_of,
            ["timestamp", "open", "high", "low", "close"],
        )
        bars = bars.dropna(subset=["timestamp", "open", "high", "low", "close"])
        bars = bars.sort_values("timestamp", kind="mergesort")
        bars["session_date"] = bars["timestamp"].dt.strftime("%Y-%m-%d")
        daily = bars.groupby("session_date", sort=True).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            bar_count=("close", "size"),
        ).reset_index()
        daily["symbol_norm"] = symbol
        daily_parts.append(daily)
    panel = pd.concat(daily_parts, ignore_index=True).sort_values(
        ["symbol_norm", "session_date"], kind="mergesort"
    )
    grouped = panel.groupby("symbol_norm", sort=False)
    panel["daily_return"] = grouped["close"].pct_change(fill_method=None)
    panel["ret_20d"] = grouped["close"].pct_change(20, fill_method=None)
    panel["ma_20d"] = grouped["close"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    panel["above_20d_ma"] = np.where(
        panel["ma_20d"].notna(), panel["close"].gt(panel["ma_20d"]).astype(float), np.nan
    )
    panel["rolling_20d_high"] = grouped["close"].transform(
        lambda values: values.rolling(20, min_periods=20).max()
    )
    panel["drawdown_20d"] = panel["close"] / panel["rolling_20d_high"] - 1.0
    panel["realized_vol_20d"] = grouped["daily_return"].transform(
        lambda values: values.rolling(20, min_periods=20).std()
    )
    market = panel.groupby("session_date", sort=True).agg(
        broad_symbol_count=("symbol_norm", "nunique"),
        broad_median_ret_20d=("ret_20d", "median"),
        broad_breadth_20d_up=(
            "ret_20d",
            lambda values: float((values.dropna() > 0.0).mean())
            if values.notna().any()
            else math.nan,
        ),
        broad_breadth_above_20d_ma=("above_20d_ma", "mean"),
        broad_median_drawdown_20d=("drawdown_20d", "median"),
        broad_median_realized_vol_20d=("realized_vol_20d", "median"),
    ).reset_index()
    broad_columns = (
        "broad_median_ret_20d",
        "broad_breadth_20d_up",
        "broad_breadth_above_20d_ma",
        "broad_median_drawdown_20d",
        "broad_median_realized_vol_20d",
    )
    for column in broad_columns:
        market[f"{column}_prior"] = pd.to_numeric(
            market[column], errors="coerce"
        ).shift(1)
    direction_inputs = (
        "broad_median_ret_20d_prior",
        "broad_breadth_20d_up_prior",
        "broad_breadth_above_20d_ma_prior",
        "broad_median_drawdown_20d_prior",
    )
    score_columns = []
    for column in direction_inputs:
        output = f"score__{column}"
        market[output] = causal_rank_score(market[column], min_periods=5)
        score_columns.append(output)
    market["b0_direction_score_raw"] = market[score_columns].mean(axis=1)
    market["b0_stress_score_raw"] = causal_rank_score(
        market["broad_median_realized_vol_20d_prior"], min_periods=5
    )
    market["b0_direction_score"] = market["b0_direction_score_raw"].rolling(
        15, min_periods=7
    ).mean()
    market["b0_stress_score"] = market["b0_stress_score_raw"].rolling(
        15, min_periods=7
    ).mean()
    market["b0_raw_state"] = np.select(
        [
            market["b0_direction_score"].le(-0.12),
            market["b0_direction_score"].ge(0.12),
        ],
        ["weak_broad_tape", "strong_broad_tape"],
        default="neutral_broad_tape",
    )
    market.loc[market["b0_direction_score"].isna(), "b0_raw_state"] = "unknown"
    market["causal_slow_b0"] = confirm_states(
        market["b0_raw_state"].astype(str).tolist()
    )
    market["b0_stress_box"] = np.where(
        market["b0_stress_score"].ge(0.0), "high_stress", "normal_stress"
    )
    market.loc[market["b0_stress_score"].isna(), "b0_stress_box"] = "unknown"
    return market


def add_emission_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    activity_3 = pd.to_numeric(frame["mean_abs_return_3"], errors="coerce").clip(
        lower=0.0
    )
    activity_12 = pd.to_numeric(frame["mean_abs_return_12"], errors="coerce").clip(
        lower=0.0
    )
    frame["regime_log_activity_3"] = np.log1p(10000.0 * activity_3)
    frame["regime_log_activity_12"] = np.log1p(10000.0 * activity_12)
    frame["regime_activity_acceleration"] = (
        frame["regime_log_activity_3"] - frame["regime_log_activity_12"]
    )
    frame["regime_log_bar_range"] = np.log1p(
        10000.0 * pd.to_numeric(frame["bar_range_pct"], errors="coerce").clip(lower=0.0)
    )
    frame["regime_wick_balance"] = pd.to_numeric(
        frame["upper_wick_pct_of_range"], errors="coerce"
    ) - pd.to_numeric(frame["lower_wick_pct_of_range"], errors="coerce")
    frame["regime_log_market_dispersion"] = np.log1p(
        10000.0
        * pd.to_numeric(frame["market_dispersion_return_6"], errors="coerce")
        .abs()
        .clip(lower=0.0)
    )
    denominator = (6.0 * activity_12.replace(0.0, np.nan)).clip(lower=1e-8)
    frame["regime_stock_minus_market_scaled"] = np.tanh(
        pd.to_numeric(frame["stock_minus_market_return_6"], errors="coerce")
        / denominator
    )
    frame["regime_market_breadth_centered"] = pd.to_numeric(
        frame["market_breadth_return_6_positive"], errors="coerce"
    ) - 0.5
    missing = [name for name in EMISSION_FEATURES if name not in frame]
    if missing:
        raise AssertionError(f"missing frozen emission features: {missing}")
    return frame


def prepare_causal_panel(
    symbols: list[str],
    provider_root: Path,
    as_of: pd.Timestamp,
    minimum_symbols: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    parts = [prepare_symbol_bars(symbol, provider_root, start, as_of) for symbol in symbols]
    panel = pd.concat(parts, ignore_index=True).sort_values(
        ["symbol_norm", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    vti = prepare_symbol_bars("VTI", provider_root, start, as_of)
    if not vti["timestamp"].eq(as_of).any():
        raise AssertionError("VTI lacks the exact provider as-of bar")
    active_symbols = sorted(
        panel.loc[panel["timestamp"].eq(as_of), "symbol_norm"].astype(str).unique()
    )
    if len(active_symbols) < minimum_symbols:
        raise AssertionError(
            f"only {len(active_symbols)} frozen stocks have the as-of bar; "
            f"need {minimum_symbols}"
        )
    panel = add_market_features(panel, vti)
    b0 = build_causal_b0(symbols, provider_root, start, as_of)
    keep = [
        "session_date",
        "causal_slow_b0",
        "b0_direction_score",
        "b0_stress_score",
        "b0_stress_box",
    ]
    panel = panel.merge(b0[keep], on="session_date", how="left", validate="many_to_one")
    panel["b0_state_numeric"] = panel["causal_slow_b0"].map(
        {
            "weak_broad_tape": -1.0,
            "neutral_broad_tape": 0.0,
            "strong_broad_tape": 1.0,
        }
    )
    panel["b0_high_stress"] = panel["b0_stress_box"].eq("high_stress").astype(float)
    panel = add_emission_features(panel)
    session_date = as_of.tz_convert("America/New_York").strftime("%Y-%m-%d")
    current = panel.loc[panel["session_date"].eq(session_date)].copy()
    current = current.sort_values(["symbol_norm", "timestamp"], kind="mergesort")
    current = current.reset_index(drop=True)
    if current.empty or current["timestamp"].gt(as_of).any():
        raise AssertionError("empty or future current-session panel")
    digest_columns = [
        "symbol_norm",
        "timestamp",
        "bar_index_in_session",
        *EMISSION_FEATURES,
        "b0_state_numeric",
        "b0_high_stress",
        *PRICE_CONTROLS,
    ]
    digest_frame = current[digest_columns].copy()
    digest_frame["timestamp"] = digest_frame["timestamp"].astype(str)
    causal_input_digest = sha256_bytes(
        digest_frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
    )
    return current, {
        "as_of": as_of,
        "session_date": session_date,
        "active_symbols": active_symbols,
        "active_symbol_count": len(active_symbols),
        "current_session_rows": len(current),
        "causal_input_sha256": causal_input_digest,
    }


def group_positions(frame: pd.DataFrame) -> list[np.ndarray]:
    return [
        group.index.to_numpy(dtype=int)
        for _, group in frame.groupby(["symbol_norm", "session_date"], sort=False)
    ]


def scale_emissions(frame: pd.DataFrame, preprocessing: pd.DataFrame) -> np.ndarray:
    if preprocessing["feature"].astype(str).tolist() != list(EMISSION_FEATURES):
        raise AssertionError("frozen emission preprocessing order drifted")
    raw = frame.loc[:, list(EMISSION_FEATURES)].apply(pd.to_numeric, errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    medians = preprocessing["imputer_median"].to_numpy(dtype=float)
    values = raw.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(medians, np.nonzero(missing)[1])
    center = preprocessing["scaler_center"].to_numpy(dtype=float)
    scale = preprocessing["scaler_scale"].to_numpy(dtype=float)
    scaled = ((values - center) / scale).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise AssertionError("non-finite frozen emission input")
    return scaled


def log_emission(scaled: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    means = model["means"]
    variances = model["variances"]
    output = np.empty((len(scaled), K), dtype=np.float64)
    constant = np.log(2.0 * np.pi * variances)
    for state in range(K):
        output[:, state] = -0.5 * np.sum(
            constant[state]
            + np.square(scaled - means[state]) / variances[state],
            axis=1,
        )
    return output


def propagate(alpha: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    hazard = model["duration_hazard"]
    stay = alpha * (1.0 - hazard)
    predicted = np.zeros_like(alpha)
    predicted[:, 1:] += stay[:, :-1]
    predicted[:, -1] += stay[:, -1]
    exit_mass = np.sum(alpha * hazard, axis=1)
    predicted[:, 0] += exit_mass @ model["transitions"]
    total = predicted.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise AssertionError("semi-Markov propagation lost probability mass")
    return predicted / total


def causal_filter(
    emissions: np.ndarray,
    positions_by_session: list[np.ndarray],
    model: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.full(len(emissions), -1, dtype=np.int16)
    ages = np.zeros(len(emissions), dtype=np.int16)
    confidence = np.full(len(emissions), np.nan, dtype=float)
    for positions in positions_by_session:
        alpha: np.ndarray | None = None
        causal_age = 0
        previous_state = -1
        for position in positions:
            if alpha is None:
                prior = np.zeros((K, MAX_DURATION), dtype=float)
                prior[:, 0] = model["initial"]
            else:
                prior = propagate(alpha, model)
            emission = emissions[position]
            scaled_emission = np.exp(emission - np.max(emission))
            posterior = prior * scaled_emission[:, None]
            posterior_sum = posterior.sum()
            if not np.isfinite(posterior_sum) or posterior_sum <= 0.0:
                raise AssertionError("semi-Markov posterior underflow")
            alpha = posterior / posterior_sum
            state_probability = alpha.sum(axis=1)
            state = int(np.argmax(state_probability))
            causal_age = causal_age + 1 if state == previous_state else 1
            previous_state = state
            labels[position] = state
            ages[position] = min(causal_age, MAX_DURATION)
            confidence[position] = float(state_probability[state])
    if (labels < 0).any() or not np.isfinite(confidence).all():
        raise AssertionError("causal filter left an unassigned row")
    return labels, ages, confidence


def _rle(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    return [
        (int(start), int(end), int(labels[start]))
        for start, end in zip(starts, ends, strict=True)
    ]


def history_tokens(
    previous_state_2: np.ndarray,
    previous_state_1: np.ndarray,
    current_state: np.ndarray,
) -> np.ndarray:
    previous_state_2 = np.asarray(previous_state_2, dtype=int)
    previous_state_1 = np.asarray(previous_state_1, dtype=int)
    current_state = np.asarray(current_state, dtype=int)
    if (
        previous_state_2.min(initial=0) < 0
        or previous_state_2.max(initial=0) > END_STATE
        or previous_state_1.min(initial=0) < 0
        or previous_state_1.max(initial=0) > END_STATE
        or current_state.min(initial=0) < 0
        or current_state.max(initial=0) >= K
    ):
        raise AssertionError("invalid history-token state")
    return ((previous_state_2 * (K + 1) + previous_state_1) * K + current_state)


def build_runs(
    frame: pd.DataFrame,
    labels: np.ndarray,
    ages: np.ndarray,
    confidence: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    run_id = 0
    for positions in group_positions(frame):
        local_labels = labels[positions]
        previous_states: list[int] = []
        for start, end, state in _rle(local_labels):
            first = int(positions[start])
            timestamp = pd.Timestamp(frame.at[first, "timestamp"])
            local = timestamp.tz_convert("America/New_York")
            entry_minutes = (
                local.hour * 60.0 + local.minute + local.second / 60.0 - 570.0
            )
            phase = 2.0 * np.pi * entry_minutes / 390.0
            row = {
                "run_id": run_id,
                "symbol_norm": str(frame.at[first, "symbol_norm"]),
                "session_date": str(frame.at[first, "session_date"]),
                "start_timestamp": timestamp,
                "bar_index_in_session": int(frame.at[first, "bar_index_in_session"]),
                "state": int(state),
                "age_at_entry": int(ages[first]),
                "state_posterior_probability": float(confidence[first]),
                "previous_state_1": previous_states[-1]
                if len(previous_states) >= 1
                else END_STATE,
                "previous_state_2": previous_states[-2]
                if len(previous_states) >= 2
                else END_STATE,
                "b0_entry_numeric": float(
                    pd.to_numeric(frame.at[first, "b0_state_numeric"], errors="coerce")
                )
                if pd.notna(frame.at[first, "b0_state_numeric"])
                else 0.0,
                "b0_entry_high_stress": float(
                    pd.to_numeric(frame.at[first, "b0_high_stress"], errors="coerce")
                )
                if pd.notna(frame.at[first, "b0_high_stress"])
                else 0.0,
                "entry_time_sin": float(np.sin(phase)),
                "entry_time_cos": float(np.cos(phase)),
            }
            for control in PRICE_CONTROLS:
                row[control] = float(
                    pd.to_numeric(frame.at[first, control], errors="coerce")
                )
            for feature in EMISSION_FEATURES:
                row[f"emission__{feature}"] = float(
                    pd.to_numeric(frame.at[first, feature], errors="coerce")
                )
            rows.append(row)
            previous_states.append(int(state))
            run_id += 1
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise AssertionError("no causal state runs")
    runs["history_token"] = history_tokens(
        runs["previous_state_2"].to_numpy(dtype=int),
        runs["previous_state_1"].to_numpy(dtype=int),
        runs["state"].to_numpy(dtype=int),
    )
    dates = pd.to_datetime(runs["session_date"], errors="raise")
    runs["quarter"] = dates.dt.year.astype(str) + "_q" + dates.dt.quarter.astype(str)
    return runs


def canonical_cycle(core: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in core)
    if not values:
        raise ValueError("empty cycle")
    return min(values[index:] + values[:index] for index in range(len(values)))


def oriented_paths(core: tuple[int, ...], current: int) -> list[tuple[int, ...]]:
    paths = {
        core[index:] + core[:index] + (int(current),)
        for index, state in enumerate(core)
        if int(state) == int(current)
    }
    return sorted(paths)


def load_cycles(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    if len(source) != 20 or "cycle" not in source:
        raise AssertionError("frozen cycle source must contain twenty cycles")
    rows = []
    seen: set[tuple[int, ...]] = set()
    for index, value in enumerate(source["cycle"].astype(str), start=1):
        closed = tuple(int(part) for part in value.split("->"))
        if len(closed) < 3 or closed[0] != closed[-1]:
            raise AssertionError(f"invalid frozen cycle {value}")
        core = canonical_cycle(closed[:-1])
        if core in seen:
            raise AssertionError(f"duplicate frozen cycle {value}")
        seen.add(core)
        rows.append({"cycle_index": index, "core": core})
    return pd.DataFrame(rows)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _history_path_probability(
    anchors: pd.DataFrame,
    route: tuple[int, ...],
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    probability = np.ones(len(anchors), dtype=float)
    previous_state_2 = anchors["previous_state_2"].to_numpy(dtype=int)
    previous_state_1 = anchors["previous_state_1"].to_numpy(dtype=int)
    current_state = np.full(len(anchors), route[0], dtype=int)
    for destination in route[1:]:
        tokens = history_tokens(previous_state_2, previous_state_1, current_state)
        logits = (
            parameters["history_intercept"][None, :]
            + parameters["history_coef"][:, tokens].T
        )
        probability *= _softmax(logits)[:, int(destination)]
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
    for row in cycles.itertuples(index=False):
        core = tuple(int(state) for state in row.core)
        values = np.zeros(len(output), dtype=float)
        for current in sorted(set(core)):
            mask = output["state"].eq(current).to_numpy()
            selected = output.loc[mask].reset_index(drop=True)
            probability = np.zeros(len(selected), dtype=float)
            for route in oriented_paths(core, current):
                probability += _history_path_probability(selected, route, parameters)
            values[mask] = probability
        if values.max(initial=0.0) > 1.0 + 1e-9 or values.min(initial=0.0) < 0.0:
            raise AssertionError("invalid frozen loop probability")
        output[f"loop_score_{int(row.cycle_index):02d}"] = np.clip(values, 0.0, 1.0)
    return output


def assign_session_states(
    panel: pd.DataFrame,
    preprocessing: pd.DataFrame,
    state_parameters: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Attach the exact frozen causal state, age, and posterior confidence."""

    scaled = scale_emissions(panel, preprocessing)
    emissions = log_emission(scaled, state_parameters)
    labels, ages, confidence = causal_filter(
        emissions, group_positions(panel), state_parameters
    )
    return panel.assign(
        state=labels,
        age=ages,
        state_posterior_probability=confidence,
    )


def make_session_runs(
    panel: pd.DataFrame,
    preprocessing: pd.DataFrame,
    state_parameters: dict[str, np.ndarray],
    cycles: pd.DataFrame,
    path_parameters: dict[str, np.ndarray],
) -> pd.DataFrame:
    assigned = assign_session_states(panel, preprocessing, state_parameters)
    runs = build_runs(
        assigned,
        assigned["state"].to_numpy(dtype=np.int16),
        assigned["age"].to_numpy(dtype=np.int16),
        assigned["state_posterior_probability"].to_numpy(dtype=float),
    )
    return add_loop_scores(runs, cycles, path_parameters)


def _raw_feature_matrix(
    frame: pd.DataFrame, representation: str, manifest: dict[str, Any]
) -> sparse.csr_matrix:
    numeric_columns = list(manifest["numeric_controls"])
    if numeric_columns != list(NUMERIC_CONTROLS):
        raise AssertionError("frozen numeric-control manifest drifted")
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    medians = pd.Series(manifest["numeric_medians"])
    numeric = numeric.fillna(medians)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise AssertionError("non-finite movement control")
    current = frame["state"].to_numpy(dtype=int)
    state = sparse.csr_matrix(np.eye(K, dtype=np.float32)[current])
    context = sparse.hstack(
        (state, sparse.csr_matrix(numeric.to_numpy(dtype=np.float32))), format="csr"
    )
    if representation == "state_context":
        return context
    if representation == "loop_scores":
        loop_columns = list(manifest["loop_score_columns"])
        loop = sparse.csr_matrix(frame[loop_columns].to_numpy(dtype=np.float32))
        return sparse.hstack((context, loop), format="csr")
    raise ValueError(representation)


def movement_predictions(
    anchors: pd.DataFrame,
    manifest: dict[str, Any],
    parameters: dict[str, np.ndarray],
) -> pd.DataFrame:
    output = anchors.copy()
    for representation in REPRESENTATIONS:
        raw = _raw_feature_matrix(output, representation, manifest)
        expected_width = int(manifest["feature_widths"][representation])
        if raw.shape[1] != expected_width:
            raise AssertionError(f"{representation} feature width drifted")
        scaler = StandardScaler(with_mean=False)
        scaler.scale_ = parameters[f"{representation}__scaler_scale"].copy()
        scaler.mean_ = parameters[f"{representation}__scaler_mean"].copy()
        scaler.var_ = parameters[f"{representation}__scaler_var"].copy()
        scaler.n_features_in_ = expected_width
        scaled = scaler.transform(raw).tocsr()
        for target in MOVEMENT_TARGETS:
            for horizon in HORIZONS:
                prefix = f"{representation}__{target}__h{horizon}"
                model = Ridge(alpha=10.0, solver="lsqr")
                model.coef_ = parameters[f"{prefix}__coef"].copy()
                model.intercept_ = parameters[f"{prefix}__intercept"][0].copy()
                model.n_features_in_ = expected_width
                output[
                    f"{representation}__{target}_prediction_{horizon}"
                ] = model.predict(scaled)

    digest_columns = [
        "state",
        *NUMERIC_CONTROLS,
        *manifest["loop_score_columns"],
    ]
    digests = []
    for row in output[digest_columns].itertuples(index=False, name=None):
        values = [safe(value) for value in row]
        digests.append(sha256_bytes(canonical_json_bytes(values)))
    output["frozen_feature_sha256"] = digests
    return output


def moving_block_bounds(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 10:
        return math.nan, math.nan, math.nan
    block = min(5, len(clean))
    blocks = np.asarray(
        [clean[start : start + block] for start in range(len(clean) - block + 1)]
    )
    needed = int(math.ceil(len(clean) / block))
    rng = np.random.default_rng(seed)
    draws = np.empty(5000, dtype=float)
    for index in range(len(draws)):
        selected = rng.integers(0, len(blocks), size=needed)
        draws[index] = blocks[selected].reshape(-1)[: len(clean)].mean()
    return (
        float(clean.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def inference_self_tests() -> None:
    assert canonical_cycle((2, 5, 1)) == canonical_cycle((5, 1, 2))
    assert oriented_paths((1, 2, 1), 1) == [(1, 1, 2, 1), (1, 2, 1, 1)]
    tokens = history_tokens(
        np.asarray([END_STATE, 1]),
        np.asarray([END_STATE, 2]),
        np.asarray([0, 3]),
    )
    assert np.array_equal(tokens, np.asarray([640, 91]))
    model = {
        "duration_hazard": np.full((K, MAX_DURATION), 0.2),
        "transitions": np.full((K, K), 1.0 / K),
    }
    model["duration_hazard"][:, -1] = 1.0
    alpha = np.zeros((K, MAX_DURATION), dtype=float)
    alpha[0, 0] = 1.0
    propagated = propagate(alpha, model)
    assert abs(float(propagated.sum()) - 1.0) < 1e-12
