#!/usr/bin/env python3
"""Independent audit for Pre-Trigger Quiet Accumulation Direction Screen V0."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ROOT / "packages" / "stocker_research" / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stocker_research.pretrigger_quiet_accumulation_v0 import (  # noqa: E402
    GROUP_A,
    GROUP_C,
    GROUP_P,
    M1_THRESHOLD,
    MODEL_CATEGORICAL_FEATURES,
    PRIMARY_RAW_FEATURES,
    PRIMARY_WINDOW_BARS,
    Q0_NUMERIC_FEATURES,
    Q1_NUMERIC_FEATURES,
    QS_NUMERIC_FEATURES,
    QUIET_SIGNED_COMPONENTS,
)

EXPERIMENT = Path(__file__).resolve().parent
PRIMARY = EXPERIMENT / "artifacts" / "primary"
PREDECESSOR_RUNNER = (
    ROOT
    / "research"
    / "directional-readiness"
    / "20260726-movement-qualified-direction-screen-v0"
    / "run_screen_v0.py"
)
EPSILON: Final[float] = 1e-12
ATR_BARS: Final[int] = 14
ANNUAL_TRADING_MINUTES: Final[int] = 252 * 390
SAMPLE_EPISODES: Final[int] = 100
RAW_QUIET_BUNDLE_COLUMNS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(
        (
            *PRIMARY_RAW_FEATURES,
            "net_return_25",
            "path_length_25",
            "range_sum_25",
            *(f"_pressure_sum_3bar_position_{position}" for position in range(PRIMARY_WINDOW_BARS)),
            *(f"_net_return_3bar_position_{position}" for position in range(PRIMARY_WINDOW_BARS)),
        )
    )
)
SAFETY_FLAGS: Final[dict[str, object]] = {
    "research_only": True,
    "retrospective_candidate_screen": True,
    "movement_model_frozen": True,
    "movement_model_refit_allowed": False,
    "m1_threshold": M1_THRESHOLD,
    "fresh_episode_definition_frozen": True,
    "direction_marker_precedes_trigger_bar": True,
    "trigger_bar_excluded_from_direction_features": True,
    "primary_pretrigger_window_bars": 5,
    "primary_direction_horizon_minutes": 10,
    "quiet_accumulation_and_distribution_mirrored": True,
    "direct_order_flow_claim": False,
    "activity_proxy_not_exchange_volume": True,
    "route_orientation_features_excluded": True,
    "option_pnl_calculated": False,
    "intraday_option_quotes_used": False,
    "broker_access": False,
    "paper_orders_allowed": False,
    "live_orders_allowed": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}


class AuditFailure(RuntimeError):
    """Fail-closed independent-audit discrepancy."""


def load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_identity(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].copy()
    payload = ordered.sort_values(list(columns), kind="mergesort").to_csv(
        index=False,
        lineterminator="\n",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise AuditFailure(f"cannot load audit dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def assert_close(
    label: str,
    actual: float,
    expected: float,
    *,
    tolerance: float = 1e-12,
) -> None:
    if math.isnan(actual) and math.isnan(expected):
        return
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise AuditFailure(f"{label}: finite/missing mismatch: {actual} vs {expected}")
    if abs(actual - expected) > tolerance:
        raise AuditFailure(f"{label}: difference {abs(actual - expected):.17g} exceeds {tolerance}")


def manual_robust_fit(values: Sequence[float] | np.ndarray[Any, Any]) -> dict[str, float]:
    raw = np.asarray(values, dtype=float)
    finite = raw[np.isfinite(raw)]
    imputation = float(np.median(finite)) if len(finite) else 0.0
    imputed = np.where(np.isfinite(raw), raw, imputation)
    center = float(np.median(imputed)) if len(imputed) else 0.0
    if len(imputed):
        q25, q75 = np.quantile(imputed, [0.25, 0.75])
        scale = float(q75 - q25)
    else:
        scale = 1.0
    if not math.isfinite(scale) or scale <= EPSILON:
        scale = 1.0
    return {
        "imputation": imputation,
        "center": center,
        "scale": scale,
    }


def manual_z(value: float, parameter: Mapping[str, object]) -> float:
    raw = value if math.isfinite(value) else float(parameter["imputation"])
    return (raw - float(parameter["center"])) / float(parameter["scale"])


def manual_pressure_slope(values: np.ndarray[Any, Any]) -> float:
    if not np.isfinite(values).all():
        return math.nan
    x_values = np.arange(len(values), dtype=float)
    cumulative = np.cumsum(values)
    centered = x_values - np.mean(x_values)
    return float(np.sum(centered * (cumulative - np.mean(cumulative))) / np.sum(centered**2))


def complete_sum(values: np.ndarray[Any, Any]) -> float:
    return float(np.sum(values)) if len(values) and np.isfinite(values).all() else math.nan


def complete_mean(values: np.ndarray[Any, Any]) -> float:
    return float(np.mean(values)) if len(values) and np.isfinite(values).all() else math.nan


def prepare_independent_bars(
    state_path: Path,
    reconstructed_panel: pd.DataFrame,
) -> pd.DataFrame:
    bars = pd.read_parquet(
        state_path,
        columns=[
            "symbol",
            "session",
            "bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bar_log_return",
            "vti__bar_log_return",
            "historical_relative_activity",
            "feature_available_timestamp_max",
        ],
        filters=[
            ("session", ">=", "2024-01-01"),
            ("session", "<=", "2025-08-22"),
        ],
    ).rename(columns={"symbol": "stock"})
    bars["stock"] = bars["stock"].astype(str)
    bars["session"] = bars["session"].astype(str)
    bars["bar_start_timestamp"] = pd.to_datetime(
        bars["bar_start_timestamp"], utc=True, errors="raise"
    )
    bars["bar_complete_timestamp"] = pd.to_datetime(
        bars["bar_complete_timestamp"], utc=True, errors="raise"
    )
    bars = bars.sort_values(
        ["stock", "bar_complete_timestamp", "session", "bar_ordinal"],
        kind="mergesort",
    ).reset_index(drop=True)
    if str(bars["session"].max()) > "2025-08-22":
        raise AuditFailure("excluded or protected state rows were materialised")
    bars["_previous_close"] = bars.groupby("stock", sort=False)["close"].shift()
    calculated = np.log(bars["close"] / bars["_previous_close"])
    bars["_r"] = calculated
    bars["_market_r"] = pd.to_numeric(bars["vti__bar_log_return"], errors="coerce")
    bars["_relative_r"] = bars["_r"] - bars["_market_r"]
    bars["_normalised_range"] = (bars["high"] - bars["low"]) / bars["_previous_close"]
    denominator = bars["high"] - bars["low"] + EPSILON
    bars["_clv"] = (2.0 * bars["close"] - bars["high"] - bars["low"]) / denominator
    lower_wick = np.minimum(bars["open"], bars["close"]) - bars["low"]
    upper_wick = bars["high"] - np.maximum(bars["open"], bars["close"])
    bars["_wick"] = (lower_wick - upper_wick) / denominator
    bars["_activity"] = pd.to_numeric(bars["historical_relative_activity"], errors="coerce")

    atr = np.full(len(bars), np.nan, dtype=float)
    for _, indices in bars.groupby("stock", sort=False).groups.items():
        positions = np.asarray(indices, dtype=int)
        rows = bars.loc[positions]
        previous = rows["_previous_close"].to_numpy(float)
        high = rows["high"].to_numpy(float)
        low = rows["low"].to_numpy(float)
        true_range = np.maximum.reduce(
            [high - low, np.abs(high - previous), np.abs(low - previous)]
        )
        atr[positions] = (
            pd.Series(true_range)
            .shift(1)
            .rolling(ATR_BARS, min_periods=ATR_BARS)
            .mean()
            .to_numpy(float)
        )
    bars["_atr"] = atr
    bars["_vwap"] = math.nan
    bars["_break"] = math.nan
    for _, indices in bars.groupby(["stock", "session"], sort=False).groups.items():
        positions = np.asarray(indices, dtype=int)
        rows = bars.loc[positions].sort_values("bar_ordinal", kind="mergesort")
        ordered_positions = rows.index.to_numpy(int)
        typical = (
            rows["high"].to_numpy(float)
            + rows["low"].to_numpy(float)
            + rows["close"].to_numpy(float)
        ) / 3.0
        activity = pd.to_numeric(rows["volume"], errors="coerce").to_numpy(float)
        activity = np.where(np.isfinite(activity) & (activity > 0.0), activity, 0.0)
        cumulative_activity = np.cumsum(activity)
        vwap = np.divide(
            np.cumsum(typical * activity),
            cumulative_activity,
            out=np.full(len(rows), np.nan, dtype=float),
            where=cumulative_activity > 0.0,
        )
        bars.loc[ordered_positions, "_vwap"] = vwap
        highs = rows["high"].to_numpy(float)
        lows = rows["low"].to_numpy(float)
        closes = rows["close"].to_numpy(float)
        for offset in range(6, len(rows)):
            prior_low = float(np.min(lows[offset - 6 : offset]))
            prior_high = float(np.max(highs[offset - 6 : offset]))
            width = highs[offset] - lows[offset] + EPSILON
            reclaim = float(lows[offset] < prior_low) * (closes[offset] - lows[offset]) / width
            rejection = float(highs[offset] > prior_high) * (highs[offset] - closes[offset]) / width
            bars.loc[ordered_positions[offset], "_break"] = reclaim - rejection
    bars["_vwap_distance"] = (bars["close"] - bars["_vwap"]) / (bars["_atr"] + EPSILON)

    snapshots = reconstructed_panel[
        [
            "symbol",
            "session",
            "checkpoint",
            "signed_pressure",
            "feature_available_timestamp_utc",
        ]
    ].rename(
        columns={
            "symbol": "stock",
            "feature_available_timestamp_utc": "_pressure_timestamp",
        }
    )
    snapshots["stock"] = snapshots["stock"].astype(str)
    snapshots["session"] = snapshots["session"].astype(str)
    snapshots["bar_ordinal"] = snapshots["checkpoint"].astype(int) - 1
    snapshots["_pressure_timestamp"] = pd.to_datetime(
        snapshots["_pressure_timestamp"], utc=True, errors="raise"
    )
    bars = bars.merge(
        snapshots[
            [
                "stock",
                "session",
                "bar_ordinal",
                "signed_pressure",
                "_pressure_timestamp",
            ]
        ],
        on=["stock", "session", "bar_ordinal"],
        how="left",
        validate="one_to_one",
    ).sort_values(["stock", "session", "bar_ordinal"], kind="mergesort")
    if bool(
        pd.to_datetime(bars["_pressure_timestamp"], utc=True, errors="coerce")
        .gt(bars["bar_complete_timestamp"])
        .fillna(False)
        .any()
    ):
        raise AuditFailure("future signed pressure entered an audited bar")
    return bars.reset_index(drop=True)


def manual_window_summary(window: pd.DataFrame, suffix: str) -> dict[str, float]:
    returns = window["_r"].to_numpy(float)
    pressure = pd.to_numeric(window["signed_pressure"], errors="coerce").to_numpy(float)
    activity = window["_activity"].to_numpy(float)
    net_return = complete_sum(returns)
    path_length = complete_sum(np.abs(returns))
    pressure_sum = complete_sum(pressure)
    if (
        np.isfinite(activity).all()
        and math.isfinite(pressure_sum)
        and math.isfinite(net_return)
        and math.isfinite(path_length)
    ):
        displacement = min(1.0, abs(net_return) / (path_length + EPSILON))
        activity_without = float(
            np.sign(pressure_sum) * np.mean(np.maximum(activity, 0.0)) * (1.0 - displacement)
        )
    else:
        activity_without = math.nan
    close = window["close"].to_numpy(float)
    vwap = window["_vwap"].to_numpy(float)
    if np.isfinite(close).all() and np.isfinite(vwap).all():
        above = close > vwap
        below = close < vwap
        side = float((int(np.count_nonzero(above)) - int(np.count_nonzero(below))) / len(window))
        reclaim = float(
            (
                int(np.count_nonzero(below[:-1] & above[1:]))
                - int(np.count_nonzero(above[:-1] & below[1:]))
            )
            / (len(window) - 1)
        )
    else:
        side = math.nan
        reclaim = math.nan
    return {
        f"net_return_{suffix}": net_return,
        f"path_length_{suffix}": path_length,
        f"range_sum_{suffix}": complete_sum(window["_normalised_range"].to_numpy(float)),
        f"directional_efficiency_{suffix}": (
            net_return / (path_length + EPSILON)
            if math.isfinite(net_return) and math.isfinite(path_length)
            else math.nan
        ),
        f"pressure_sum_{suffix}": pressure_sum,
        f"pressure_persistence_{suffix}": (
            float(np.mean(np.sign(pressure))) if np.isfinite(pressure).all() else math.nan
        ),
        f"pressure_slope_{suffix}": manual_pressure_slope(pressure),
        f"activity_without_displacement_{suffix}": activity_without,
        f"relative_resilience_{suffix}": complete_sum(window["_relative_r"].to_numpy(float)),
        f"mean_clv_{suffix}": complete_mean(window["_clv"].to_numpy(float)),
        f"mean_wick_asymmetry_{suffix}": complete_mean(window["_wick"].to_numpy(float)),
        f"break_failure_asymmetry_{suffix}": complete_sum(window["_break"].to_numpy(float)),
        f"mean_vwap_distance_{suffix}": complete_mean(window["_vwap_distance"].to_numpy(float)),
        f"vwap_side_balance_{suffix}": side,
        f"vwap_reclaim_balance_{suffix}": reclaim,
    }


def manual_raw_feature_row(
    episode: pd.Series,
    bars: pd.DataFrame,
) -> dict[str, float]:
    stock = str(episode["stock"])
    session = str(episode["session"])
    checkpoint = int(episode["checkpoint"])
    marker_ordinal = checkpoint - 2
    trigger_ordinal = checkpoint - 1
    session_bars = bars.loc[
        bars["stock"].astype(str).eq(stock) & bars["session"].astype(str).eq(session)
    ].sort_values("bar_ordinal", kind="mergesort")
    prefix = session_bars.loc[session_bars["bar_ordinal"].astype(int).le(marker_ordinal)]
    marker = prefix.loc[prefix["bar_ordinal"].astype(int).eq(marker_ordinal)].iloc[0]
    trigger = session_bars.loc[session_bars["bar_ordinal"].astype(int).eq(trigger_ordinal)].iloc[0]
    if pd.Timestamp(marker["bar_complete_timestamp"]) != pd.Timestamp(
        episode["pretrigger_marker_timestamp"]
    ):
        raise AuditFailure("manual marker is not stored T-1")
    if not pd.Timestamp(marker["bar_complete_timestamp"]) < pd.Timestamp(
        trigger["bar_complete_timestamp"]
    ):
        raise AuditFailure("manual marker does not precede trigger")
    values: dict[str, float] = {}
    for window_bars in (3, 5, 9):
        window = prefix.loc[
            prefix["bar_ordinal"].between(marker_ordinal - window_bars + 1, marker_ordinal)
        ]
        if len(window) == window_bars:
            values.update(manual_window_summary(window, str(window_bars * 5)))
    primary = prefix.loc[prefix["bar_ordinal"].between(marker_ordinal - 4, marker_ordinal)]
    ordinals = primary["bar_ordinal"].astype(int).tolist()
    if ordinals != list(range(marker_ordinal - 4, marker_ordinal + 1)):
        raise AuditFailure("manual primary window is not five contiguous bars")
    for position, end_ordinal in enumerate(ordinals):
        three_bar = prefix.loc[prefix["bar_ordinal"].between(end_ordinal - 2, end_ordinal)]
        values[f"_pressure_sum_3bar_position_{position}"] = (
            complete_sum(three_bar["signed_pressure"].to_numpy(float))
            if len(three_bar) == 3
            else math.nan
        )
        values[f"_net_return_3bar_position_{position}"] = (
            complete_sum(three_bar["_r"].to_numpy(float)) if len(three_bar) == 3 else math.nan
        )
    marker_close = float(marker["close"])
    atr = float(marker["_atr"])

    def trailing(column: str, count: int) -> float:
        rows = prefix[column].tail(count).to_numpy(float)
        return complete_sum(rows) if len(rows) == count else math.nan

    def distance(reference: float) -> float:
        return (
            (marker_close - reference) / (atr + EPSILON)
            if math.isfinite(reference) and math.isfinite(atr)
            else math.nan
        )

    stock_5 = trailing("_r", 1)
    stock_10 = trailing("_r", 2)
    market_5 = trailing("_market_r", 1)
    market_10 = trailing("_market_r", 2)
    values.update(
        {
            "stock_return_5m_tminus1": stock_5,
            "stock_return_10m_tminus1": stock_10,
            "stock_return_20m_tminus1": trailing("_r", 4),
            "market_return_5m_tminus1": market_5,
            "market_return_10m_tminus1": market_10,
            "market_return_20m_tminus1": trailing("_market_r", 4),
            "stock_minus_market_return_5m_tminus1": (
                stock_5 - market_5
                if math.isfinite(stock_5) and math.isfinite(market_5)
                else math.nan
            ),
            "stock_minus_market_return_10m_tminus1": (
                stock_10 - market_10
                if math.isfinite(stock_10) and math.isfinite(market_10)
                else math.nan
            ),
            "distance_from_vwap_tminus1": distance(float(marker["_vwap"])),
            "distance_from_session_open_tminus1": distance(float(session_bars.iloc[0]["open"])),
            "clv_tminus1": float(marker["_clv"]),
            "wick_asymmetry_tminus1": float(marker["_wick"]),
        }
    )
    first_six = prefix.loc[prefix["bar_ordinal"].between(0, 5)]
    opening_midpoint = (
        0.5 * (float(first_six["high"].max()) + float(first_six["low"].min()))
        if len(first_six) == 6
        else math.nan
    )
    values["distance_from_opening_range_midpoint_tminus1"] = distance(opening_midpoint)
    prior_six = prefix.loc[prefix["bar_ordinal"].astype(int).lt(marker_ordinal)].tail(6)
    values["distance_from_previous_six_bar_high_tminus1"] = distance(
        float(prior_six["high"].max()) if len(prior_six) == 6 else math.nan
    )
    values["distance_from_previous_six_bar_low_tminus1"] = distance(
        float(prior_six["low"].min()) if len(prior_six) == 6 else math.nan
    )
    return values


def manual_score(
    raw: Mapping[str, float],
    parameters: Mapping[str, Any],
) -> dict[str, float]:
    pressure_z = manual_z(
        float(raw["pressure_sum_25"]),
        cast(Mapping[str, object], parameters["pressure_25"]),
    )
    price_z = manual_z(
        float(raw["net_return_25"]),
        cast(Mapping[str, object], parameters["price_25"]),
    )
    pressure_raw = float(raw["pressure_sum_25"])
    divergence = (
        float(np.sign(pressure_raw) * (abs(pressure_z) - abs(price_z)))
        if math.isfinite(pressure_raw)
        else math.nan
    )
    three_signs: list[float] = []
    for position in range(5):
        pressure = float(raw[f"_pressure_sum_3bar_position_{position}"])
        price = float(raw[f"_net_return_3bar_position_{position}"])
        if not math.isfinite(pressure) or not math.isfinite(price):
            three_signs.append(math.nan)
            continue
        pressure_position_z = manual_z(
            pressure,
            cast(Mapping[str, object], parameters["pressure_3bar"]),
        )
        price_position_z = manual_z(
            price,
            cast(Mapping[str, object], parameters["price_3bar"]),
        )
        three_divergence = np.sign(pressure) * (abs(pressure_position_z) - abs(price_position_z))
        three_signs.append(float(np.sign(three_divergence)))
    persistence = (
        float(np.mean(three_signs))
        if np.isfinite(np.asarray(three_signs, dtype=float)).all()
        else math.nan
    )
    range_compression = -manual_z(
        float(raw["range_sum_25"]),
        cast(Mapping[str, object], parameters["range_sum"]),
    )
    path_compression = -manual_z(
        float(raw["path_length_25"]),
        cast(Mapping[str, object], parameters["path_length"]),
    )
    quietness = 1.0 / (1.0 + math.exp(-range_compression))
    quietness *= 1.0 / (1.0 + math.exp(-path_compression))
    enriched = dict(raw)
    enriched["signed_absorption_divergence_25"] = divergence
    enriched["accumulation_sign_persistence_25"] = persistence
    component_parameters = cast(
        Mapping[str, Mapping[str, object]], parameters["component_parameters"]
    )
    clipped: list[float] = []
    output: dict[str, float] = {
        "pressure_z_25": pressure_z,
        "price_z_25": price_z,
        "signed_absorption_divergence_25": divergence,
        "accumulation_sign_persistence_25": persistence,
        "range_compression_25": range_compression,
        "path_compression_25": path_compression,
        "quietness_25": quietness,
    }
    for component in QUIET_SIGNED_COMPONENTS:
        z_value = manual_z(float(enriched[component]), component_parameters[component])
        clipped_value = float(np.clip(z_value, -3.0, 3.0))
        output[f"{component}__clipped_z"] = clipped_value
        clipped.append(clipped_value)
    core = float(np.mean(clipped))
    output["signed_accumulation_core_25"] = core
    output["quiet_absorption_score_25"] = quietness * core
    return output


def manual_probability(
    row: pd.Series,
    specification: Mapping[str, Any],
) -> float:
    design: list[float] = []
    for column in cast(Sequence[str], specification["numeric_features"]):
        value = float(pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0])
        missing = not math.isfinite(value)
        imputed = (
            float(cast(Mapping[str, object], specification["medians"])[column])
            if missing
            else value
        )
        center = float(cast(Mapping[str, object], specification["robust_centers"])[column])
        scale = float(cast(Mapping[str, object], specification["robust_scales"])[column])
        design.extend([(imputed - center) / scale, float(missing)])
    levels_map = cast(Mapping[str, Sequence[str]], specification["categorical_levels"])
    for column in cast(Sequence[str], specification["categorical_features"]):
        raw = "__MISSING__" if pd.isna(row[column]) else str(row[column])
        levels = [str(value) for value in levels_map[column]]
        value = raw if raw in set(levels) else "__UNKNOWN__"
        design.extend(float(value == level) for level in levels)
    names = cast(Sequence[str], specification["design_feature_names"])
    if len(design) != len(names):
        raise AuditFailure("manual model design width drifted")
    coefficients = np.asarray(specification["coefficients"], dtype=float)
    linear = float(
        np.asarray(design, dtype=float) @ coefficients + float(specification["intercept"])
    )
    return (
        1.0 / (1.0 + math.exp(-linear))
        if linear >= 0.0
        else math.exp(linear) / (1.0 + math.exp(linear))
    )


def manual_target(
    episode: pd.Series,
    bars: pd.DataFrame,
) -> dict[str, float]:
    stock = str(episode["stock"])
    session = str(episode["session"])
    checkpoint = int(episode["checkpoint"])
    session_bars = bars.loc[
        bars["stock"].astype(str).eq(stock) & bars["session"].astype(str).eq(session)
    ].set_index("bar_ordinal")
    marker = session_bars.loc[checkpoint - 2]
    entry = session_bars.loc[checkpoint]
    marker_close = float(marker["close"])
    entry_price = float(entry["open"])
    pre_entry = math.log(entry_price / marker_close)
    output = {
        "marker_close": marker_close,
        "entry_price": entry_price,
        "pre_entry_signed_return": pre_entry,
    }
    atm_iv = float(episode["atm_iv"])
    for horizon in (5, 10, 15, 30):
        count = horizon // 5
        close = float(session_bars.loc[checkpoint + count - 1, "close"])
        signed = math.log(close / entry_price)
        path = session_bars.loc[list(range(checkpoint, checkpoint + count))]
        call_mfe = float(max(0.0, np.max(np.log(path["high"].to_numpy(float) / entry_price))))
        call_mae = float(max(0.0, np.max(np.log(entry_price / path["low"].to_numpy(float)))))
        output[f"signed_log_return_{horizon}m"] = signed
        output[f"call_mfe_{horizon}m"] = call_mfe
        output[f"call_mae_{horizon}m"] = call_mae
        output[f"put_mfe_{horizon}m"] = call_mae
        output[f"put_mae_{horizon}m"] = call_mfe
        output[f"iv_expected_absolute_{horizon}m"] = (
            atm_iv * math.sqrt(horizon / ANNUAL_TRADING_MINUTES) * math.sqrt(2.0 / math.pi)
        )
    for horizon in (10, 30):
        post = output[f"signed_log_return_{horizon}m"]
        output[f"remaining_fraction_{horizon}m"] = abs(post) / (
            abs(pre_entry) + abs(post) + EPSILON
        )
    return output


def fit_manual_score_parameters(frame: pd.DataFrame) -> dict[str, Any]:
    pressure_25 = manual_robust_fit(frame["pressure_sum_25"].to_numpy(float))
    price_25 = manual_robust_fit(frame["net_return_25"].to_numpy(float))
    pressure_columns = [f"_pressure_sum_3bar_position_{position}" for position in range(5)]
    return_columns = [f"_net_return_3bar_position_{position}" for position in range(5)]
    pressure_3bar = manual_robust_fit(frame[pressure_columns].to_numpy(float).ravel())
    price_3bar = manual_robust_fit(frame[return_columns].to_numpy(float).ravel())
    enriched = frame.copy()
    divergences: list[float] = []
    persistences: list[float] = []
    for _, values in frame.iterrows():
        pressure = float(values["pressure_sum_25"])
        price = float(values["net_return_25"])
        if math.isfinite(pressure) and math.isfinite(price):
            divergence = float(
                np.sign(pressure)
                * (abs(manual_z(pressure, pressure_25)) - abs(manual_z(price, price_25)))
            )
        else:
            divergence = math.nan
        signs: list[float] = []
        for position in range(5):
            pressure_position = float(values[f"_pressure_sum_3bar_position_{position}"])
            return_position = float(values[f"_net_return_3bar_position_{position}"])
            if not math.isfinite(pressure_position) or not math.isfinite(return_position):
                signs.append(math.nan)
                continue
            divergence_position = np.sign(pressure_position) * (
                abs(manual_z(pressure_position, pressure_3bar))
                - abs(manual_z(return_position, price_3bar))
            )
            signs.append(float(np.sign(divergence_position)))
        persistence = (
            float(np.mean(signs)) if np.isfinite(np.asarray(signs, dtype=float)).all() else math.nan
        )
        divergences.append(divergence)
        persistences.append(persistence)
    enriched["signed_absorption_divergence_25"] = divergences
    enriched["accumulation_sign_persistence_25"] = persistences
    return {
        "fit_partition": "development",
        "pressure_25": pressure_25,
        "price_25": price_25,
        "pressure_3bar": pressure_3bar,
        "price_3bar": price_3bar,
        "range_sum": manual_robust_fit(frame["range_sum_25"].to_numpy(float)),
        "path_length": manual_robust_fit(frame["path_length_25"].to_numpy(float)),
        "component_parameters": {
            component: manual_robust_fit(enriched[component].to_numpy(float))
            for component in QUIET_SIGNED_COMPONENTS
        },
    }


def nested_parameter_max_difference(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> float:
    maximum = 0.0
    for key, expected_value in expected.items():
        if key not in actual:
            raise AuditFailure(f"stored preprocessing parameter missing: {key}")
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            maximum = max(
                maximum,
                nested_parameter_max_difference(
                    cast(Mapping[str, Any], actual_value),
                    cast(Mapping[str, Any], expected_value),
                ),
            )
        elif isinstance(expected_value, (int, float)):
            maximum = max(
                maximum,
                abs(float(actual_value) - float(expected_value)),
            )
        elif actual_value != expected_value:
            raise AuditFailure(f"stored preprocessing value drifted: {key}")
    return maximum


def independent_confidence_boundary(
    probabilities: np.ndarray[Any, Any],
    target_coverage: float = 0.35,
    minimum_actions: int = 100,
) -> float:
    confidence = np.abs(probabilities - 0.5)
    order = np.lexsort((np.arange(len(probabilities)), -confidence))
    target_index = int(math.ceil(target_coverage * len(probabilities))) - 1
    selected = max(target_index, minimum_actions - 1)
    return float(confidence[order[selected]])


def independent_fresh_episodes(
    panel: pd.DataFrame,
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Rebuild crossings and spacing without the production episode helper."""

    rows = panel.rename(columns={"symbol": "stock", "M1_probability": "m1_probability"}).copy()
    rows["session"] = rows["session"].astype(str)
    signal = states[["stock", "session", "bar_ordinal", "bar_complete_timestamp"]].copy()
    signal["session"] = signal["session"].astype(str)
    signal["checkpoint"] = signal["bar_ordinal"].astype(int) + 1
    signal = signal.rename(columns={"bar_complete_timestamp": "signal_timestamp"})
    rows = rows.merge(
        signal[["stock", "session", "checkpoint", "signal_timestamp"]],
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    ).sort_values(["stock", "session", "checkpoint"], kind="mergesort")
    rows["signal_timestamp"] = pd.to_datetime(rows["signal_timestamp"], utc=True, errors="raise")
    rows["above"] = rows["m1_probability"].astype(float).ge(M1_THRESHOLD)
    rows["previous"] = rows.groupby(["stock", "session"], sort=False)["m1_probability"].shift()
    rows["crossing"] = rows["above"] & (
        rows["previous"].isna() | rows["previous"].astype(float).lt(M1_THRESHOLD)
    )
    selected: list[pd.Series[Any]] = []
    for _, group in rows.loc[rows["crossing"]].groupby(["stock", "session"], sort=True):
        previous_start: pd.Timestamp | None = None
        for _, row in group.iterrows():
            current = pd.Timestamp(row["signal_timestamp"])
            if previous_start is not None:
                elapsed = (current - previous_start).total_seconds() / 60.0
                if elapsed < 30.0:
                    continue
            selected.append(row)
            previous_start = current
    return pd.DataFrame(selected).reset_index(drop=True), int(rows["above"].sum())


def apply_manual_score(frame: pd.DataFrame, parameters: Mapping[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    scored = [manual_score(row, parameters) for _, row in output.iterrows()]
    score_frame = pd.DataFrame(scored, index=output.index)
    for column in score_frame:
        output[column] = score_frame[column]
    return output


def independent_model_specification(
    development: pd.DataFrame,
    *,
    target_column: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    model_id: str,
) -> dict[str, Any]:
    training = development.loc[development[target_column].notna()].copy()
    target = training[target_column].astype(int).to_numpy()
    if len(np.unique(target)) != 2:
        raise AuditFailure(f"{model_id} independent fit lacks both target classes")
    medians: dict[str, float] = {}
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in numeric_features:
        raw = pd.to_numeric(training[column], errors="coerce").to_numpy(float)
        finite = raw[np.isfinite(raw)]
        median = float(np.median(finite)) if len(finite) else 0.0
        imputed = np.where(np.isfinite(raw), raw, median)
        center = float(np.median(imputed))
        q25, q75 = np.quantile(imputed, [0.25, 0.75])
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale <= EPSILON:
            scale = 1.0
        medians[column] = median
        centers[column] = center
        scales[column] = scale
    categorical_levels: dict[str, list[str]] = {}
    for column in categorical_features:
        values = training[column].fillna("__MISSING__").astype(str)
        observed = sorted(set(values).difference({"__MISSING__", "__UNKNOWN__"}))
        categorical_levels[column] = [*observed, "__MISSING__", "__UNKNOWN__"]
    specification: dict[str, Any] = {
        "model_id": model_id,
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "medians": medians,
        "robust_centers": centers,
        "robust_scales": scales,
        "categorical_levels": categorical_levels,
    }
    design, names = independent_model_design(training, specification)
    estimator = LogisticRegression(
        penalty="l2",
        C=0.25,
        solver="liblinear",
        max_iter=300,
        class_weight=None,
        random_state=20260726,
    )
    estimator.fit(design, target)
    if int(estimator.n_iter_[0]) >= 300:
        raise AuditFailure(f"{model_id} independent refit did not converge")
    specification["design_feature_names"] = names
    specification["coefficients"] = estimator.coef_[0].astype(float).tolist()
    specification["intercept"] = float(estimator.intercept_[0])
    return specification


def independent_model_design(
    frame: pd.DataFrame,
    specification: Mapping[str, Any],
) -> tuple[np.ndarray[Any, Any], list[str]]:
    pieces: list[np.ndarray[Any, Any]] = []
    names: list[str] = []
    medians = cast(Mapping[str, float], specification["medians"])
    centers = cast(Mapping[str, float], specification["robust_centers"])
    scales = cast(Mapping[str, float], specification["robust_scales"])
    for column in cast(Sequence[str], specification["numeric_features"]):
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        missing = ~np.isfinite(raw)
        imputed = np.where(missing, float(medians[column]), raw)
        pieces.extend(
            [
                ((imputed - float(centers[column])) / float(scales[column]))[:, None],
                missing.astype(float)[:, None],
            ]
        )
        names.extend([column, f"{column}__missing"])
    levels_by_column = cast(Mapping[str, Sequence[str]], specification["categorical_levels"])
    for column in cast(Sequence[str], specification["categorical_features"]):
        values = frame[column].fillna("__MISSING__").astype(str)
        levels = [str(value) for value in levels_by_column[column]]
        values = values.where(values.isin(set(levels)), "__UNKNOWN__")
        for level in levels:
            pieces.append(values.eq(level).to_numpy(float)[:, None])
            names.append(f"{column}=={level}")
    return np.concatenate(pieces, axis=1), names


def independent_model_predict(
    frame: pd.DataFrame,
    specification: Mapping[str, Any],
) -> np.ndarray[Any, Any]:
    design, names = independent_model_design(frame, specification)
    if names != list(cast(Sequence[str], specification["design_feature_names"])):
        raise AuditFailure("independent model design order drifted")
    linear = design @ np.asarray(
        cast(Sequence[float], specification["coefficients"]), dtype=float
    ) + float(specification["intercept"])
    probabilities = np.empty(len(linear), dtype=float)
    positive = linear >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
    exponential = np.exp(linear[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    return probabilities


def independent_primary_metrics(frame: pd.DataFrame, probability_column: str) -> dict[str, float]:
    valid = frame.loc[frame["direction_up_10m"].notna() & frame[probability_column].notna()]
    target = valid["direction_up_10m"].astype(int).to_numpy()
    probabilities = np.clip(valid[probability_column].to_numpy(float), EPSILON, 1.0 - EPSILON)
    return {
        "log_loss": float(
            -np.mean(target * np.log(probabilities) + (1 - target) * np.log1p(-probabilities))
        ),
        "brier_score": float(np.mean((probabilities - target) ** 2)),
        "auc": float(roc_auc_score(target, probabilities)),
    }


def independent_selective_metrics(
    frame: pd.DataFrame,
    action_column: str,
) -> dict[str, float]:
    selected = frame.loc[frame[action_column].astype(str).ne("ABSTAIN")]
    actions = selected[action_column].astype(str).to_numpy()
    returns = selected["signed_log_return_10m"].to_numpy(float)
    sides = np.where(actions == "CALL", 1.0, -1.0)
    aligned = sides * returns
    valid = np.isfinite(returns) & (returns != 0.0)
    truth = (returns[valid] > 0.0).astype(int)
    predicted = (sides[valid] > 0.0).astype(int)
    accuracy = float(np.mean(truth == predicted)) if len(truth) else math.nan
    if len(np.unique(truth)) == 2:
        balanced = float(
            (np.mean(predicted[truth == 0] == 0) + np.mean(predicted[truth == 1] == 1)) / 2.0
        )
    else:
        balanced = math.nan
    finite_aligned = aligned[np.isfinite(aligned)]
    return {
        "action_coverage": float(len(selected) / len(frame)) if len(frame) else math.nan,
        "directional_accuracy": accuracy,
        "balanced_accuracy": balanced,
        "mean_aligned_return": float(np.mean(finite_aligned)) if len(finite_aligned) else math.nan,
        "median_aligned_return": (
            float(np.median(finite_aligned)) if len(finite_aligned) else math.nan
        ),
        "positive_aligned_return_rate": (
            float(np.mean(finite_aligned > 0.0)) if len(finite_aligned) else math.nan
        ),
    }


def independent_score_slope(
    frame: pd.DataFrame,
    value_column: str,
) -> float:
    order = {
        "strong_distribution": 0,
        "moderate_distribution": 1,
        "neutral": 2,
        "moderate_accumulation": 3,
        "strong_accumulation": 4,
    }
    x_values = frame["score_bin"].astype(str).map(order).to_numpy(float)
    y_values = frame[value_column].to_numpy(float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    return float(np.polyfit(x_values[valid], y_values[valid], 1)[0])


def independent_bootstrap_metrics(sample: pd.DataFrame) -> dict[str, float]:
    q0 = independent_primary_metrics(sample, "Q0_p_up")
    q1 = independent_primary_metrics(sample, "Q1_p_up")
    selective = independent_selective_metrics(sample, "Q1_action")
    actioned = sample.loc[sample["Q1_action"].astype(str).ne("ABSTAIN")]
    iv_selective = independent_selective_metrics(
        sample.loc[sample["iv_excess_10m"].eq(1)],
        "Q1_action",
    )
    quartile_selective = independent_selective_metrics(
        sample.loc[sample["largest_absolute_movement_quartile"].astype(bool)],
        "Q1_action",
    )
    return {
        "q1_minus_q0_log_loss_improvement": q0["log_loss"] - q1["log_loss"],
        "q1_minus_q0_brier_improvement": q0["brier_score"] - q1["brier_score"],
        "q1_minus_q0_auc_improvement": q1["auc"] - q0["auc"],
        "q1_selective_action_coverage": selective["action_coverage"],
        "q1_selective_accuracy": selective["directional_accuracy"],
        "q1_selective_balanced_accuracy": selective["balanced_accuracy"],
        "mean_aligned_ten_minute_return": selective["mean_aligned_return"],
        "median_aligned_ten_minute_return": selective["median_aligned_return"],
        "positive_aligned_return_rate": selective["positive_aligned_return_rate"],
        "mean_remaining_fraction": (
            float(actioned["remaining_fraction_10m"].mean()) if len(actioned) else math.nan
        ),
        "quiet_absorption_score_monotonic_slope": independent_score_slope(
            sample, "signed_log_return_10m"
        ),
        "iv_excess_subgroup_accuracy": iv_selective["directional_accuracy"],
        "largest_movement_quartile_accuracy": quartile_selective["directional_accuracy"],
    }


def audit_bootstrap(
    bootstrap: pd.DataFrame,
    plan: Mapping[str, Any],
    assessment: pd.DataFrame,
) -> float:
    draws = bootstrap.loc[bootstrap["row_type"].eq("draw")]
    if len(draws) != 100:
        raise AuditFailure("whole-session bootstrap does not contain exactly 100 draws")
    draw_sessions = cast(Sequence[Sequence[str]], plan["draw_sessions"])
    universe = cast(Sequence[str], plan["session_universe"])
    if len(draw_sessions) != 100 or any(len(draw) != len(universe) for draw in draw_sessions):
        raise AuditFailure("stored bootstrap plan does not preserve session draw units")
    metric_columns = [
        "q1_minus_q0_log_loss_improvement",
        "q1_minus_q0_brier_improvement",
        "q1_minus_q0_auc_improvement",
        "q1_selective_action_coverage",
        "q1_selective_accuracy",
        "q1_selective_balanced_accuracy",
        "mean_aligned_ten_minute_return",
        "median_aligned_ten_minute_return",
        "positive_aligned_return_rate",
        "mean_remaining_fraction",
        "quiet_absorption_score_monotonic_slope",
        "iv_excess_subgroup_accuracy",
        "largest_movement_quartile_accuracy",
    ]
    maximum_draw_difference = 0.0
    for draw_number, labels in enumerate(draw_sessions, start=1):
        sample = pd.concat(
            [
                assessment.loc[assessment["session"].astype(str).eq(str(session))].copy()
                for session in labels
            ],
            ignore_index=True,
        )
        independent = independent_bootstrap_metrics(sample)
        stored_draw = draws.loc[draws["draw"].astype(int).eq(draw_number)].iloc[0]
        for metric in metric_columns:
            actual = float(stored_draw[metric])
            expected = float(independent[metric])
            if math.isnan(actual) and math.isnan(expected):
                difference = 0.0
            else:
                difference = abs(actual - expected)
            maximum_draw_difference = max(maximum_draw_difference, difference)
    if maximum_draw_difference > 1e-12:
        raise AuditFailure(
            f"independent bootstrap draw metrics differ by {maximum_draw_difference}"
        )
    for level in (0.80, 0.90, 0.95):
        alpha = (1.0 - level) / 2.0
        for metric in metric_columns:
            values = pd.to_numeric(draws[metric], errors="coerce").dropna()
            for bound, quantile in (("lower", alpha), ("upper", 1.0 - alpha)):
                stored = bootstrap.loc[
                    bootstrap["row_type"].eq("interval")
                    & bootstrap["interval_level"].eq(level)
                    & bootstrap["metric"].eq(metric)
                    & bootstrap["bound"].eq(bound),
                    "value",
                ]
                if len(stored) != 1:
                    raise AuditFailure("bootstrap interval identity is not unique")
                expected = float(values.quantile(quantile, interpolation="linear"))
                assert_close(
                    f"bootstrap {metric} {level} {bound}",
                    float(stored.iloc[0]),
                    expected,
                )
    return maximum_draw_difference


def audit_null_assignments(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    plan: Mapping[str, Any],
    null_metrics: pd.DataFrame,
) -> float:
    targets = cast(Mapping[str, Sequence[object]], plan["null_targets"])
    if len(targets) != 5 or null_metrics["null_number"].nunique() != 5:
        raise AuditFailure("label null count is not exactly five")
    if set(null_metrics.groupby("null_number").size().tolist()) != {3}:
        raise AuditFailure("each label null must refit Q0, QS, and Q1")
    for null_number in range(1, 6):
        null = development.copy()
        null["null_target"] = pd.Series(targets[str(null_number)], dtype=float)
        for _, indices in development.groupby(["session", "checkpoint"], sort=True).groups.items():
            original = (
                development.loc[indices, "direction_up_10m"]
                .fillna(-1)
                .astype(int)
                .sort_values()
                .tolist()
            )
            permuted = (
                null.loc[indices, "null_target"].fillna(-1).astype(int).sort_values().tolist()
            )
            if original != permuted:
                raise AuditFailure("a label null changed a slate label multiset")
        expected_hash = frame_identity(
            null,
            ["stock", "session", "checkpoint", "null_target"],
        )
        stored_hash = cast(Mapping[str, str], plan["label_assignment_hashes"])[str(null_number)]
        if expected_hash != stored_hash:
            raise AuditFailure("stored label-null assignment hash drifted")
    features = {
        "Q0": Q0_NUMERIC_FEATURES,
        "QS": QS_NUMERIC_FEATURES,
        "Q1": Q1_NUMERIC_FEATURES,
    }
    maximum_difference = 0.0
    for null_number in range(1, 6):
        null = development.copy().reset_index(drop=True)
        null["direction_up_10m"] = pd.Series(
            targets[str(null_number)],
            dtype=float,
        ).to_numpy(float)
        oof = independent_oof_probabilities(null, features)
        full_parameters = fit_manual_score_parameters(null)
        scored_development = apply_manual_score(null, full_parameters)
        scored_assessment = apply_manual_score(assessment, full_parameters)
        for model_id, numeric in features.items():
            boundary = independent_confidence_boundary(oof[model_id])
            specification = independent_model_specification(
                scored_development,
                target_column="direction_up_10m",
                numeric_features=numeric,
                categorical_features=MODEL_CATEGORICAL_FEATURES,
                model_id=f"audit_null_{null_number}_{model_id}",
            )
            probabilities = independent_model_predict(scored_assessment, specification)
            evaluated = scored_assessment.copy()
            evaluated["audit_probability"] = probabilities
            evaluated["audit_action"] = independent_actions(probabilities, boundary)
            primary = independent_primary_metrics(evaluated, "audit_probability")
            selective = independent_selective_metrics(evaluated, "audit_action")
            monotonicity = independent_score_slope(evaluated, "audit_probability")
            stored = null_metrics.loc[
                null_metrics["null_number"].astype(int).eq(null_number)
                & null_metrics["model_id"].astype(str).eq(model_id)
            ].iloc[0]
            expected_values = {
                **primary,
                "selective_accuracy": selective["directional_accuracy"],
                "mean_aligned_return": selective["mean_aligned_return"],
                "quiet_score_prediction_monotonicity": monotonicity,
                "confidence_boundary": boundary,
            }
            for column, expected in expected_values.items():
                actual = float(stored[column])
                difference = (
                    0.0 if math.isnan(actual) and math.isnan(expected) else abs(actual - expected)
                )
                maximum_difference = max(maximum_difference, difference)
    if maximum_difference > 1e-12:
        raise AuditFailure(f"independent label-null refits differ by {maximum_difference}")
    return maximum_difference


def independent_actions(
    probabilities: np.ndarray[Any, Any],
    boundary: float,
) -> np.ndarray[Any, Any]:
    actions = np.full(len(probabilities), "ABSTAIN", dtype="<U7")
    actions[probabilities >= 0.5 + boundary] = "CALL"
    actions[probabilities <= 0.5 - boundary] = "PUT"
    return actions


def independent_oof_probabilities(
    development: pd.DataFrame,
    features: Mapping[str, Sequence[str]],
) -> dict[str, np.ndarray[Any, Any]]:
    rows = development.copy().reset_index(drop=True)
    sessions = np.asarray(sorted(rows["session"].astype(str).unique()), dtype=object)
    fold_by_session = {
        str(session): fold
        for fold, block in enumerate(np.array_split(sessions, 4))
        for session in block
    }
    folds = rows["session"].astype(str).map(fold_by_session).astype(int)
    probabilities = {model_id: np.full(len(rows), np.nan, dtype=float) for model_id in features}
    for fold in range(4):
        training_raw = rows.loc[folds.ne(fold)]
        held_raw = rows.loc[folds.eq(fold)]
        parameters = fit_manual_score_parameters(training_raw)
        training = apply_manual_score(training_raw, parameters)
        held = apply_manual_score(held_raw, parameters)
        for model_id, numeric in features.items():
            specification = independent_model_specification(
                training,
                target_column="direction_up_10m",
                numeric_features=numeric,
                categorical_features=MODEL_CATEGORICAL_FEATURES,
                model_id=f"audit_oof_{model_id}_{fold}",
            )
            probabilities[model_id][held.index.to_numpy(int)] = independent_model_predict(
                held,
                specification,
            )
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise AuditFailure("independent OOF refit left missing probabilities")
    return probabilities


def audit_grouped_permutations(
    permutations: pd.DataFrame,
    assessment: pd.DataFrame,
    q1_specification: Mapping[str, Any],
    boundary: float,
) -> float:
    details = permutations.loc[permutations["row_type"].eq("permutation")]
    summaries = permutations.loc[permutations["row_type"].eq("mean_over_20")]
    expected_groups = {
        "Group_P_persistent_pressure",
        "Group_A_absorption_response",
        "Group_C_compression_context",
        "quiet_absorption_score_25",
    }
    if set(details["group_id"]) != expected_groups:
        raise AuditFailure("grouped permutation feature groups drifted")
    if set(details.groupby("group_id").size().tolist()) != {20}:
        raise AuditFailure("grouped attribution did not use exactly 20 permutations")
    if len(summaries) != 4:
        raise AuditFailure("grouped permutation summaries are incomplete")
    real_primary = independent_primary_metrics(assessment, "Q1_p_up")
    real_selective = independent_selective_metrics(assessment, "Q1_action")
    feature_groups = {
        "Group_P_persistent_pressure": GROUP_P,
        "Group_A_absorption_response": GROUP_A,
        "Group_C_compression_context": GROUP_C,
        "quiet_absorption_score_25": ("quiet_absorption_score_25",),
    }
    maximum_difference = 0.0
    source_frame = assessment.reset_index(drop=True)
    for _, stored in details.iterrows():
        group_id = str(stored["group_id"])
        feature_columns = list(feature_groups[group_id])
        permuted = source_frame.copy()
        generator = np.random.default_rng(int(stored["seed"]))
        for _, indices in source_frame.groupby(
            ["session", "checkpoint"],
            sort=True,
            dropna=False,
        ).groups.items():
            positions = np.asarray(indices, dtype=int)
            source = positions[generator.permutation(len(positions))]
            permuted.loc[positions, feature_columns] = source_frame.loc[
                source, feature_columns
            ].to_numpy()
        probabilities = independent_model_predict(permuted, q1_specification)
        permuted["audit_probability"] = probabilities
        permuted["audit_action"] = independent_actions(probabilities, boundary)
        primary = independent_primary_metrics(permuted, "audit_probability")
        selective = independent_selective_metrics(permuted, "audit_action")
        expected_values = {
            "log_loss_deterioration": primary["log_loss"] - real_primary["log_loss"],
            "brier_deterioration": primary["brier_score"] - real_primary["brier_score"],
            "auc_deterioration": real_primary["auc"] - primary["auc"],
            "selective_accuracy_deterioration": (
                real_selective["directional_accuracy"] - selective["directional_accuracy"]
            ),
            "mean_aligned_return_deterioration": (
                real_selective["mean_aligned_return"] - selective["mean_aligned_return"]
            ),
            "median_aligned_return_deterioration": (
                real_selective["median_aligned_return"] - selective["median_aligned_return"]
            ),
        }
        for column, expected in expected_values.items():
            difference = abs(float(stored[column]) - expected)
            maximum_difference = max(maximum_difference, difference)
    if maximum_difference > 1e-12:
        raise AuditFailure(
            f"independent grouped-permutation metrics differ by {maximum_difference}"
        )
    for group_id, rows in details.groupby("group_id", sort=True):
        summary = summaries.loc[summaries["group_id"].eq(group_id)].iloc[0]
        for column in (
            "log_loss_deterioration",
            "brier_deterioration",
            "auc_deterioration",
            "selective_accuracy_deterioration",
            "mean_aligned_return_deterioration",
            "median_aligned_return_deterioration",
        ):
            assert_close(
                f"permutation mean {group_id} {column}",
                float(summary[column]),
                float(rows[column].mean()),
            )
    return maximum_difference


def independent_temporal_placebo(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    temporal: pd.DataFrame,
) -> float:
    combined = pd.concat([development, assessment], ignore_index=True, sort=False)
    ordering = combined.assign(
        _original_position=np.arange(len(combined), dtype=int),
        _timestamp=pd.to_datetime(
            combined["pretrigger_marker_timestamp"],
            utc=True,
            errors="raise",
        ),
    ).sort_values(["stock", "_timestamp", "_original_position"], kind="mergesort")
    shifted = ordering.groupby("stock", sort=False)[list(RAW_QUIET_BUNDLE_COLUMNS)].shift(1)
    placebo = combined.copy()
    placebo.loc[
        ordering["_original_position"].to_numpy(int),
        list(RAW_QUIET_BUNDLE_COLUMNS),
    ] = shifted.to_numpy()
    placebo_development = placebo.loc[
        placebo["partition"].astype(str).eq("development")
    ].reset_index(drop=True)
    placebo_assessment = placebo.loc[placebo["partition"].astype(str).eq("assessment")].reset_index(
        drop=True
    )
    oof = independent_oof_probabilities(
        placebo_development,
        {"Q1": Q1_NUMERIC_FEATURES},
    )
    boundary = independent_confidence_boundary(oof["Q1"])
    parameters = fit_manual_score_parameters(placebo_development)
    scored_development = apply_manual_score(placebo_development, parameters)
    scored_assessment = apply_manual_score(placebo_assessment, parameters)
    specification = independent_model_specification(
        scored_development,
        target_column="direction_up_10m",
        numeric_features=Q1_NUMERIC_FEATURES,
        categorical_features=MODEL_CATEGORICAL_FEATURES,
        model_id="audit_temporal_placebo_Q1",
    )
    probabilities = independent_model_predict(scored_assessment, specification)
    scored_assessment["audit_probability"] = probabilities
    scored_assessment["audit_action"] = independent_actions(probabilities, boundary)
    primary = independent_primary_metrics(scored_assessment, "audit_probability")
    selective = independent_selective_metrics(scored_assessment, "audit_action")
    stored = temporal.loc[
        temporal["model"].astype(str).eq("temporally_misaligned_placebo_Q1")
    ].iloc[0]
    expected_values = {
        **primary,
        "selective_action_coverage": selective["action_coverage"],
        "selective_directional_accuracy": selective["directional_accuracy"],
        "selective_balanced_accuracy": selective["balanced_accuracy"],
        "selective_mean_aligned_return": selective["mean_aligned_return"],
        "selective_median_aligned_return": selective["median_aligned_return"],
        "selective_positive_aligned_return_rate": selective["positive_aligned_return_rate"],
        "confidence_boundary": boundary,
    }
    maximum_difference = 0.0
    for column, expected in expected_values.items():
        actual = float(stored[column])
        difference = 0.0 if math.isnan(actual) and math.isnan(expected) else abs(actual - expected)
        maximum_difference = max(maximum_difference, difference)
    if maximum_difference > 1e-12:
        raise AuditFailure(f"independent temporal-placebo refit differs by {maximum_difference}")
    return maximum_difference


def execute_audit() -> dict[str, Any]:
    required_files = [
        "contract.json",
        "source_manifest.json",
        "protected_boundary_audit.json",
        "movement_model_reconstruction.json",
        "episode_reconstruction.json",
        "episode_identity_comparison.csv",
        "pretrigger_timestamp_audit.json",
        "pretrigger_feature_manifest.json",
        "feature_formula_manifest.json",
        "development_preprocessing.json",
        "development_oof_predictions.parquet",
        "quiet_absorption_score_parameters.json",
        "frozen_direction_thresholds.json",
        "primary_candidate_freeze.json",
        "assessment_predictions.parquet",
        "direction_model_metrics.csv",
        "selective_policy_metrics.csv",
        "baseline_metrics.csv",
        "score_bin_metrics.csv",
        "grouped_permutation_metrics.csv",
        "temporal_placebo_metrics.csv",
        "material_move_metrics.csv",
        "remaining_movement_metrics.csv",
        "monthly_metrics.csv",
        "checkpoint_metrics.csv",
        "stock_metrics.csv",
        "concentration_metrics.csv",
        "bootstrap_metrics.csv",
        "direction_null_metrics.csv",
        "decision.json",
        "determinism_check.json",
        "report.md",
    ]
    missing = [name for name in required_files if not (PRIMARY / name).exists()]
    if missing:
        raise AuditFailure(f"required artifacts missing: {missing}")
    contract = load_json(PRIMARY / "contract.json")
    decision = load_json(PRIMARY / "decision.json")
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected or decision.get(key) != expected:
            raise AuditFailure(f"safety flag drifted in contract/decision: {key}")
    if contract["signed_pressure"]["redefinition"] is not False:
        raise AuditFailure("contract permits signed-pressure redefinition")
    if (
        contract["signed_pressure"]["alignment"] != "exact_checkpoint_bar_close_only"
        or contract["signed_pressure"]["forward_fill"] is not False
        or contract["signed_pressure"]["interpolation"] is not False
    ):
        raise AuditFailure("contract permits non-exact signed-pressure alignment")
    if contract["activity_proxy_not_exchange_volume"] is not True:
        raise AuditFailure("activity proxy claims boundary drifted")

    source_manifest = load_json(PRIMARY / "source_manifest.json")
    branch_path = Path(source_manifest["branch_c"]["path"])
    state_path = Path(source_manifest["state_surface"]["path"])
    if sha256_file(branch_path) != source_manifest["branch_c"]["sha256"]:
        raise AuditFailure("frozen Branch C source hash drifted")
    if sha256_file(state_path) != source_manifest["state_surface"]["sha256"]:
        raise AuditFailure("frozen state source hash drifted")
    predecessor = load_module(PREDECESSOR_RUNNER, "pretrigger_independent_predecessor")
    sources = cast(dict[str, Any], predecessor.load_frozen_inputs())
    historical = cast(pd.DataFrame, sources["historical"])
    _, _, reconstructed_panel, reconstructed_movement = predecessor.reconstruct_frozen_m1(
        historical
    )
    reconstructed_episodes, reconstructed_episode_audit = predecessor.build_episode_panel(
        reconstructed_panel,
        cast(pd.DataFrame, sources["states"]),
    )
    reconstructed_panel = cast(pd.DataFrame, reconstructed_panel)
    reconstructed_episodes = cast(pd.DataFrame, reconstructed_episodes)
    reconstructed_movement = cast(dict[str, Any], reconstructed_movement)
    reconstructed_episode_audit = cast(dict[str, Any], reconstructed_episode_audit)
    if float(reconstructed_movement["maximum_direct_probability_difference"]) > 1e-12:
        raise AuditFailure("independent M1 probability reconstruction failed")
    if float(reconstructed_movement["reconstructed_threshold"]) != M1_THRESHOLD:
        raise AuditFailure("independent M1 threshold reconstruction failed")
    expected_counts = (1266, 538, 285, 253)
    actual_counts = (
        int(reconstructed_episode_audit["raw_above_threshold_checkpoint_rows"]),
        int(reconstructed_episode_audit["fresh_episodes"]),
        int(reconstructed_episode_audit["development_episodes"]),
        int(reconstructed_episode_audit["assessment_episodes"]),
    )
    if actual_counts != expected_counts:
        raise AuditFailure(f"independent episode counts drifted: {actual_counts}")
    independently_selected, independent_raw_above = independent_fresh_episodes(
        reconstructed_panel,
        cast(pd.DataFrame, sources["states"]),
    )
    if independent_raw_above != 1266 or len(independently_selected) != 538:
        raise AuditFailure("scalar fresh-episode reconstruction count drifted")
    fresh_identity = ["stock", "session", "checkpoint", "signal_timestamp"]
    manual_identities = frame_identity(independently_selected, fresh_identity)
    predecessor_identities = frame_identity(reconstructed_episodes, fresh_identity)
    if manual_identities != predecessor_identities:
        raise AuditFailure("scalar fresh-episode identities differ from frozen episodes")
    if int(reconstructed_episodes["minutes_since_previous_episode"].dropna().lt(30.0).sum()):
        raise AuditFailure("independent fresh episodes violate thirty-minute spacing")
    identity_comparison = pd.read_csv(PRIMARY / "episode_identity_comparison.csv")
    if (
        len(identity_comparison) != 538
        or not identity_comparison["episode_identity_match"].astype(bool).all()
    ):
        raise AuditFailure("stored fresh episode identities do not match predecessor")

    assessment = pd.read_parquet(PRIMARY / "assessment_predictions.parquet")
    development = pd.read_parquet(PRIMARY / "development_oof_predictions.parquet")
    if str(assessment["session"].max()) > "2025-08-22":
        raise AuditFailure("assessment contains excluded/protected outcomes")
    if str(development["session"].max()) > "2024-12-31":
        raise AuditFailure("development OOF contains assessment outcomes")
    independent_bars = prepare_independent_bars(state_path, reconstructed_panel)
    score_parameters = load_json(PRIMARY / "quiet_absorption_score_parameters.json")
    preprocessing = load_json(PRIMARY / "development_preprocessing.json")
    frozen_thresholds = load_json(PRIMARY / "frozen_direction_thresholds.json")
    primary_freeze = load_json(PRIMARY / "primary_candidate_freeze.json")
    if primary_freeze != {
        **primary_freeze,
        "primary_candidate": "Q1",
        "primary_window_bars": 5,
        "primary_window_minutes": 25,
        "primary_horizon_minutes": 10,
        "trigger_bar_excluded": True,
        "composite_weights": "equal",
        "model_family": "l2_logistic",
        "C": 0.25,
        "assessment_model_switching_allowed": False,
    }:
        raise AuditFailure("primary candidate freeze drifted")

    independently_fitted = fit_manual_score_parameters(development)
    preprocessing_difference = nested_parameter_max_difference(
        score_parameters,
        independently_fitted,
    )
    if preprocessing_difference > 1e-12:
        raise AuditFailure(f"development-only preprocessing differs by {preprocessing_difference}")
    fold_maximum = 0.0
    fold_manifests = cast(Mapping[str, Mapping[str, Any]], preprocessing["oof_folds"])
    for fold in range(4):
        manifest = fold_manifests[str(fold)]
        training_sessions = set(cast(Sequence[str], manifest["training_sessions"]))
        held_sessions = set(cast(Sequence[str], manifest["held_sessions"]))
        if training_sessions.intersection(held_sessions):
            raise AuditFailure("OOF preprocessing split a complete session")
        training_rows = development.loc[development["session"].astype(str).isin(training_sessions)]
        manual_fold = fit_manual_score_parameters(training_rows)
        fold_maximum = max(
            fold_maximum,
            nested_parameter_max_difference(
                cast(Mapping[str, Any], manifest["score_parameters"]),
                manual_fold,
            ),
        )
    if fold_maximum > 1e-12:
        raise AuditFailure("OOF score preprocessing is not fold-specific")
    for model_id in ("Q0", "QS", "Q1"):
        expected_numeric = {
            "Q0": Q0_NUMERIC_FEATURES,
            "QS": QS_NUMERIC_FEATURES,
            "Q1": Q1_NUMERIC_FEATURES,
        }[model_id]
        specification = cast(Mapping[str, Any], preprocessing["full_models"][model_id])
        if tuple(specification["numeric_features"]) != expected_numeric:
            raise AuditFailure(f"{model_id} numeric feature manifest drifted")
        if tuple(specification["categorical_features"]) != MODEL_CATEGORICAL_FEATURES:
            raise AuditFailure(f"{model_id} fixed-effect manifest drifted")
        if (
            specification["penalty"] != "l2"
            or float(specification["C"]) != 0.25
            or specification["solver"] != "liblinear"
            or int(specification["max_iter"]) != 300
            or specification["class_weight"] is not None
        ):
            raise AuditFailure(f"{model_id} model family drifted")
    if set(Q1_NUMERIC_FEATURES).intersection(
        {
            "top_route_orientation",
            "registered_loop_orientation",
            "trigger_bar_return",
        }
    ):
        raise AuditFailure("prohibited route/trigger features entered Q1")

    for model_id in ("Q0", "QS", "Q1"):
        probabilities = development[f"{model_id}_p_up"].to_numpy(float)
        independent_boundary = independent_confidence_boundary(probabilities)
        stored_boundary = float(frozen_thresholds["boundaries"][model_id])
        assert_close(
            f"{model_id} OOF confidence boundary",
            stored_boundary,
            independent_boundary,
        )

    sample = assessment.sort_values(["stock", "session", "checkpoint"], kind="mergesort").head(
        SAMPLE_EPISODES
    )
    maximum_feature_difference = 0.0
    maximum_score_difference = 0.0
    maximum_probability_difference = 0.0
    maximum_target_difference = 0.0
    action_mismatches = 0
    trigger_intrusions = 0
    model_specs = cast(Mapping[str, Mapping[str, Any]], preprocessing["full_models"])
    for _, stored in sample.iterrows():
        raw = manual_raw_feature_row(stored, independent_bars)
        score = manual_score(raw, score_parameters)
        for column, expected in raw.items():
            if column not in stored.index:
                continue
            actual = float(stored[column])
            if math.isnan(actual) and math.isnan(expected):
                difference = 0.0
            elif math.isfinite(actual) and math.isfinite(expected):
                difference = abs(actual - expected)
            else:
                raise AuditFailure(f"manual feature missingness drifted: {column}")
            maximum_feature_difference = max(maximum_feature_difference, difference)
        for column, expected in score.items():
            actual = float(stored[column])
            if math.isnan(actual) and math.isnan(expected):
                difference = 0.0
            elif math.isfinite(actual) and math.isfinite(expected):
                difference = abs(actual - expected)
            else:
                raise AuditFailure(f"manual score missingness drifted: {column}")
            maximum_score_difference = max(maximum_score_difference, difference)
        marker = pd.Timestamp(stored["pretrigger_marker_timestamp"])
        trigger = pd.Timestamp(stored["trigger_timestamp"])
        if not marker < trigger or marker != trigger - pd.Timedelta(minutes=5):
            trigger_intrusions += 1
        target = manual_target(stored, independent_bars)
        for column, expected in target.items():
            actual = float(stored[column])
            maximum_target_difference = max(maximum_target_difference, abs(actual - expected))
        manual_row = stored.copy()
        for column, value in {**raw, **score}.items():
            manual_row[column] = value
        for model_id in ("Q0", "QS", "Q1"):
            probability = manual_probability(manual_row, model_specs[model_id])
            maximum_probability_difference = max(
                maximum_probability_difference,
                abs(probability - float(stored[f"{model_id}_p_up"])),
            )
            boundary = float(frozen_thresholds["boundaries"][model_id])
            action = (
                "CALL"
                if probability >= 0.5 + boundary
                else "PUT"
                if probability <= 0.5 - boundary
                else "ABSTAIN"
            )
            action_mismatches += int(action != str(stored[f"{model_id}_action"]))
    if maximum_feature_difference > 1e-12:
        raise AuditFailure(
            f"manual feature difference exceeds tolerance: {maximum_feature_difference}"
        )
    if maximum_score_difference > 1e-12:
        raise AuditFailure(f"manual score difference exceeds tolerance: {maximum_score_difference}")
    if maximum_probability_difference > 1e-12:
        raise AuditFailure(
            f"manual probability difference exceeds tolerance: {maximum_probability_difference}"
        )
    if maximum_target_difference > 1e-12:
        raise AuditFailure(
            f"manual target difference exceeds tolerance: {maximum_target_difference}"
        )
    if action_mismatches or trigger_intrusions:
        raise AuditFailure("manual actions or T-1 timestamps drifted")

    score_bins = pd.read_csv(PRIMARY / "score_bin_metrics.csv")
    if list(score_bins["score_bin"]) != [
        "strong_distribution",
        "moderate_distribution",
        "neutral",
        "moderate_accumulation",
        "strong_accumulation",
    ]:
        raise AuditFailure("frozen score-bin labels drifted")
    if int(score_bins["episodes"].sum()) != len(assessment):
        raise AuditFailure("frozen score bins do not cover assessment episodes")
    permutations = pd.read_csv(PRIMARY / "grouped_permutation_metrics.csv")
    permutation_maximum_difference = audit_grouped_permutations(
        permutations,
        assessment,
        model_specs["Q1"],
        float(frozen_thresholds["boundaries"]["Q1"]),
    )
    bootstrap = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    resampling_plan = load_json(PRIMARY / "frozen_resampling_plan.json")
    bootstrap_maximum_difference = audit_bootstrap(
        bootstrap,
        resampling_plan,
        assessment,
    )
    null_metrics = pd.read_csv(PRIMARY / "direction_null_metrics.csv")
    null_refit_maximum_difference = audit_null_assignments(
        development,
        assessment,
        resampling_plan,
        null_metrics,
    )
    temporal = pd.read_csv(PRIMARY / "temporal_placebo_metrics.csv")
    if set(temporal["model"]) != {
        "real_Q1",
        "temporally_misaligned_placebo_Q1",
    }:
        raise AuditFailure("temporal placebo comparison is incomplete")
    if not isinstance(
        decision["temporal_placebo_comparison"]["real_q1_outperforms_temporal_placebo"],
        bool,
    ):
        raise AuditFailure("temporal-placebo decision gate is not deterministic")
    temporal_placebo_maximum_difference = independent_temporal_placebo(
        development,
        assessment,
        temporal,
    )

    support = cast(Mapping[str, Mapping[str, Any]], decision["support_gates"])
    if int(support["development"]["episodes"]) != 285:
        raise AuditFailure("development support count drifted")
    if int(support["assessment"]["episodes"]) != 253:
        raise AuditFailure("assessment support count drifted")
    q1_actions = int(assessment["Q1_action"].astype(str).ne("ABSTAIN").sum())
    if int(support["selective"]["actions"]) != q1_actions:
        raise AuditFailure("selective support action count drifted")
    primary_gates = cast(Mapping[str, bool], decision["primary_pass_gates"])
    if bool(decision["all_primary_pass_gates_passed"]) != all(primary_gates.values()):
        raise AuditFailure("primary decision gate conjunction drifted")
    allowed_decisions = {
        "pretrigger_quiet_accumulation_direction_candidate_supported",
        "persistent_pressure_direction_supported_absorption_not_supported",
        "absorption_response_promising_but_full_gate_not_met",
        "pretrigger_direction_present_but_too_late",
        "quiet_accumulation_score_descriptive_only",
        "pretrigger_quiet_accumulation_unstable",
        "no_incremental_pretrigger_directional_signal",
        "blocked_movement_episode_reconstruction_failure",
        "blocked_insufficient_pretrigger_history",
        "blocked_insufficient_direction_episode_support",
        "blocked_insufficient_selective_action_support",
        "blocked_chronology_or_leakage_failure",
        "blocked_model_convergence_failure",
        "blocked_reproducibility_or_audit_failure",
    }
    if decision["overall_decision"] not in allowed_decisions:
        raise AuditFailure("overall decision is outside the frozen categories")
    if all(primary_gates.values()) and decision["overall_decision"] not in {
        "pretrigger_quiet_accumulation_direction_candidate_supported",
        "blocked_insufficient_pretrigger_history",
    }:
        raise AuditFailure("supported decision does not match all primary gates")
    pretrigger_audit = load_json(PRIMARY / "pretrigger_timestamp_audit.json")
    pressure_history_supported = bool(
        int(pretrigger_audit["development_pressure_complete_primary_windows"]) >= 220
        and int(pretrigger_audit["assessment_pressure_complete_primary_windows"]) >= 180
    )
    if pressure_history_supported:
        raise AuditFailure("exact per-bar pressure history unexpectedly passed support")
    if decision["overall_decision"] != "blocked_insufficient_pretrigger_history":
        raise AuditFailure("insufficient exact pressure history did not fail closed")
    if decision["pretrigger_history_status"] != "insufficient_support":
        raise AuditFailure("pre-trigger history component status did not fail closed")
    determinism = load_json(PRIMARY / "determinism_check.json")
    if not determinism.get("passed", False):
        raise AuditFailure("full determinism rebuild did not pass")
    for key in (
        "episode_identity_mismatches",
        "pretrigger_timestamp_mismatches",
        "action_decision_mismatches",
    ):
        if int(determinism[key]) != 0:
            raise AuditFailure(f"determinism mismatch is nonzero: {key}")
    for key in (
        "maximum_feature_difference",
        "maximum_score_difference",
        "maximum_probability_difference",
        "maximum_target_difference",
        "maximum_aligned_return_difference",
    ):
        if float(determinism[key]) > 1e-12:
            raise AuditFailure(f"determinism tolerance failed: {key}")

    checklist = {
        "frozen_m1_reconstruction": True,
        "frozen_threshold": True,
        "fresh_episode_identities": True,
        "thirty_minute_episode_spacing": True,
        "direction_marker_equals_T_minus_1": True,
        "trigger_bar_excluded": True,
        "primary_five_bar_window": True,
        "per_bar_formulas": True,
        "no_future_price_in_features": True,
        "signed_pressure_provenance": True,
        "activity_proxy_provenance_and_label": True,
        "vwap_causality": True,
        "market_proxy_causality": True,
        "break_failure_construction": True,
        "development_only_preprocessing": True,
        "oof_composite_standardisation": True,
        "equal_composite_weights": True,
        "Q0_QS_Q1_feature_manifests": True,
        "models_fit_on_2024_only": True,
        "confidence_threshold_from_development_oof_only": True,
        "assessment_target_construction": True,
        "call_put_abstain_actions": True,
        "aligned_return_metrics": True,
        "remaining_movement_calculations": True,
        "score_bin_boundaries": True,
        "grouped_permutations": True,
        "whole_session_bootstrap": True,
        "label_nulls": True,
        "temporal_placebo": True,
        "support_gates": True,
        "decision_logic": True,
    }
    audit = {
        "passed": True,
        "auditor": "independent_scalar_and_serialized_model_reconstruction",
        "assessment_episodes_manually_reconstructed": SAMPLE_EPISODES,
        "manual_probability_rows_per_model": SAMPLE_EPISODES,
        "manual_action_rows_per_model": SAMPLE_EPISODES,
        "maximum_feature_difference": maximum_feature_difference,
        "maximum_score_difference": maximum_score_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "action_decision_mismatches": action_mismatches,
        "maximum_target_difference": maximum_target_difference,
        "pretrigger_timestamp_mismatches": trigger_intrusions,
        "development_preprocessing_max_difference": preprocessing_difference,
        "oof_preprocessing_max_difference": fold_maximum,
        "bootstrap_draw_metric_max_difference": bootstrap_maximum_difference,
        "grouped_permutation_metric_max_difference": permutation_maximum_difference,
        "label_null_refit_metric_max_difference": null_refit_maximum_difference,
        "temporal_placebo_refit_metric_max_difference": temporal_placebo_maximum_difference,
        "movement_probability_max_difference": reconstructed_movement[
            "maximum_direct_probability_difference"
        ],
        "episode_identity_mismatches": 0,
        "checklist": checklist,
        "fail_closed": True,
    }
    write_json(PRIMARY / "lightweight_audit.json", audit)
    decision["independent_audit_result"] = "passed"
    decision["independent_audit_sample_per_model"] = SAMPLE_EPISODES
    write_json(PRIMARY / "decision.json", decision)
    return audit


def write_failed_audit(error: AuditFailure) -> None:
    failure = {
        "passed": False,
        "fail_closed": True,
        "failure": str(error),
    }
    write_json(PRIMARY / "lightweight_audit.json", failure)
    if (PRIMARY / "decision.json").exists():
        decision = load_json(PRIMARY / "decision.json")
    else:
        decision = dict(SAFETY_FLAGS)
    decision["pre_audit_overall_decision"] = decision.get("overall_decision")
    decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
    decision["independent_audit_result"] = "failed"
    decision["audit_blocker"] = str(error)
    for key in (
        "movement_gate_status",
        "episode_reconstruction_status",
        "pretrigger_history_status",
        "quietness_status",
        "persistent_pressure_status",
        "absorption_response_status",
        "relative_resilience_status",
        "vwap_defence_status",
        "composite_score_status",
        "selective_direction_status",
        "remaining_movement_status",
        "prospective_recorder_priority",
    ):
        decision[key] = "blocked"
    write_json(PRIMARY / "decision.json", decision)


def main() -> int:
    try:
        audit = execute_audit()
    except AuditFailure as error:
        write_failed_audit(error)
        print(json.dumps({"passed": False, "failure": str(error)}, indent=2))
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
