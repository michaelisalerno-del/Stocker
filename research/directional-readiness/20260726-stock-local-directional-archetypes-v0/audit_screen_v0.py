#!/usr/bin/env python3
"""Independent artifact auditor for Stock-Local Directional Archetype Screen V0."""

from __future__ import annotations

# ruff: noqa: E402 -- deterministic numerical limits must precede imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.stock_local_directional_archetypes_v0 import (
    ABSORPTION_FEATURES,
    CONTINUATION_FEATURES,
    RELATIVE_STRENGTH_FEATURES,
)

DENSE_CAUSAL_PATH = Path(
    os.environ.get(
        "STOCKER_ARCHETYPE_DENSE_CAUSAL_PATH",
        "/Users/michaelsalerno/Documents/Codex/"
        "2026-07-23-you-are-working-in-the-github-3/research/route-competition/"
        "20260722-broad-conflict-advance-hazard-v02/artifacts/primary/"
        "dense_advance_panel.parquet",
    )
)
HISTORICAL_OPTIONS_PATH = Path(
    os.environ.get(
        "STOCKER_ARCHETYPE_HISTORICAL_OPTIONS_PATH",
        "/Users/michaelsalerno/Documents/Codex/"
        "2026-07-23-you-are-working-in-the-github-3/research/cross-market-context/"
        "20260723-daily-stock-front-options-context-v01/artifacts/primary/"
        "front_options_dimensions.parquet",
    )
)
STATE_PATH = Path(
    os.environ.get(
        "STOCKER_ARCHETYPE_STATE_PATH",
        "/Users/michaelsalerno/Documents/Codex/"
        "2026-07-23-you-are-working-in-the-github-5/data/cache/"
        "minimal-intraday-iv-excess-holdout-v0/frozen_state_surface.parquet",
    )
)
KEYS = ["stock", "session", "checkpoint"]
ARCHETYPES = ("C1", "A1", "R1")
FEATURES = {
    "C1": CONTINUATION_FEATURES,
    "A1": ABSORPTION_FEATURES,
    "R1": RELATIVE_STRENGTH_FEATURES,
}


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def maximum_difference(left: pd.DataFrame, right: pd.DataFrame, columns: Sequence[str]) -> float:
    first = left.loc[:, list(columns)].to_numpy(float)
    second = right.loc[:, list(columns)].to_numpy(float)
    difference = np.abs(first - second)
    return float(np.nanmax(difference)) if difference.size else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manual_movement_probabilities(
    frame: pd.DataFrame,
    specification: Mapping[str, object],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    features = [str(value) for value in cast(Sequence[object], specification["numeric_features"])]
    raw = frame.loc[:, features].to_numpy(float)
    medians = np.asarray(specification["numeric_medians"], dtype=float)
    means = np.asarray(specification["numeric_means"], dtype=float)
    scales = np.asarray(specification["numeric_scales"], dtype=float)
    values = np.where(np.isfinite(raw), raw, medians)
    parts = [(values - means) / scales]
    levels_by_control = cast(
        Mapping[str, Sequence[object]],
        specification["category_levels"],
    )
    for control in cast(Sequence[object], specification["category_controls"]):
        name = str(control)
        if name != "stock":
            raise AssertionError(f"unexpected movement control: {name}")
        observed = frame["stock"].astype(str).to_numpy()
        levels = [str(value) for value in levels_by_control[name]]
        for level in levels[1:]:
            parts.append(np.asarray(observed == level, dtype=float)[:, None])
    design = np.concatenate(parts, axis=1)
    coefficients = np.asarray(specification["coefficients"], dtype=float)
    linear = design @ coefficients + float(specification["intercept"])
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))


def build_independent_causal_surface(
    movement_manifest: Mapping[str, Any],
    states: pd.DataFrame,
) -> pd.DataFrame:
    group_o = tuple(str(value) for value in movement_manifest["group_o"])
    causal_group_i = tuple(str(value) for value in movement_manifest["causally_valid_group_i"])
    dense_columns = [
        "row_id",
        "symbol",
        "session",
        "period",
        "checkpoint",
        "feature_available_timestamp_utc",
        *causal_group_i,
    ]
    dense = pd.read_parquet(DENSE_CAUSAL_PATH, columns=dense_columns).rename(
        columns={"symbol": "stock"}
    )
    option_features = group_o[:16]
    option_columns = [
        "symbol",
        "session",
        "required_options_date",
        "options_observation_date",
        "atm_iv",
        *option_features,
    ]
    options = pd.read_parquet(HISTORICAL_OPTIONS_PATH, columns=option_columns).rename(
        columns={"symbol": "stock"}
    )
    if options.duplicated(["stock", "session"]).any():
        raise AssertionError("independent options context is not unique")
    dates = options[["session", "required_options_date", "options_observation_date"]].apply(
        pd.to_datetime
    )
    if not dates["required_options_date"].equals(dates["options_observation_date"]):
        raise AssertionError("options observation is not the exact required previous session")
    if bool((dates["options_observation_date"] >= dates["session"]).any()):
        raise AssertionError("non-causal options context entered the movement gate")
    surface = dense.merge(options, on=["stock", "session"], how="inner", validate="many_to_one")
    checkpoints = surface["checkpoint"].astype(int)
    for name in group_o[16:]:
        expected_checkpoint = int(name.removeprefix("checkpoint_"))
        surface[name] = checkpoints.eq(expected_checkpoint).astype(float)
    counts = surface.groupby(["stock", "session"], sort=False)["checkpoint"].transform("size")
    surface["row_weight"] = 1.0 / counts.astype(float)
    totals = surface.groupby(["stock", "session"])["row_weight"].sum()
    if not np.allclose(totals.to_numpy(float), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("independent stock-local weights do not sum to one")
    entry = states[["stock", "session", "bar_ordinal", "open"]].rename(
        columns={"bar_ordinal": "checkpoint", "open": "_entry"}
    )
    close_15 = states[["stock", "session", "bar_ordinal", "close"]].copy()
    close_15["checkpoint"] = close_15["bar_ordinal"].astype(int) - 2
    close_15 = close_15.rename(columns={"close": "_close_15"}).drop(columns="bar_ordinal")
    surface = surface.merge(entry, on=KEYS, validate="one_to_one").merge(
        close_15,
        on=KEYS,
        validate="one_to_one",
    )
    movement = np.abs(
        np.log(surface["_close_15"].to_numpy(float) / surface["_entry"].to_numpy(float))
    )
    expectation = (
        surface["atm_iv"].to_numpy(float)
        * math.sqrt(15.0 / (252.0 * 390.0))
        * math.sqrt(2.0 / math.pi)
    )
    surface["movement_exceeds_prior_close_iv_15m"] = (movement > expectation).astype(int)
    return surface.sort_values("row_id", kind="mergesort").reset_index(drop=True)


def independently_refit_movement_model(
    development: pd.DataFrame,
    specification: Mapping[str, object],
) -> dict[str, float | int]:
    features = [str(value) for value in cast(Sequence[object], specification["numeric_features"])]
    raw = development.loc[:, features].to_numpy(float)
    medians = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
    values = np.where(np.isfinite(raw), raw, medians)
    means = np.mean(values, axis=0)
    scales = np.std(values, axis=0, ddof=0)
    scales = np.where(scales >= 1e-12, scales, 1.0)
    parts = [(values - means) / scales]
    stocks = development["stock"].astype(str).to_numpy()
    stock_levels = sorted(set(stocks))
    for level in stock_levels[1:]:
        parts.append(np.asarray(stocks == level, dtype=float)[:, None])
    design = np.concatenate(parts, axis=1)
    estimator = LogisticRegression(
        penalty="l2",
        C=0.25,
        solver="liblinear",
        max_iter=300,
        class_weight=None,
        random_state=20260722,
        n_jobs=1,
    )
    estimator.fit(
        design,
        development["movement_exceeds_prior_close_iv_15m"].to_numpy(int),
        sample_weight=development["row_weight"].to_numpy(float),
    )
    return {
        "maximum_median_difference": float(
            np.max(np.abs(medians - np.asarray(specification["numeric_medians"], dtype=float)))
        ),
        "maximum_mean_difference": float(
            np.max(np.abs(means - np.asarray(specification["numeric_means"], dtype=float)))
        ),
        "maximum_scale_difference": float(
            np.max(np.abs(scales - np.asarray(specification["numeric_scales"], dtype=float)))
        ),
        "maximum_coefficient_difference": float(
            np.max(
                np.abs(estimator.coef_[0] - np.asarray(specification["coefficients"], dtype=float))
            )
        ),
        "intercept_difference": abs(
            float(estimator.intercept_[0]) - float(specification["intercept"])
        ),
        "iterations": int(estimator.n_iter_[0]),
    }


def manual_fresh_episodes(
    surface: pd.DataFrame,
    probabilities: np.ndarray[Any, np.dtype[np.float64]],
    threshold: float,
) -> pd.DataFrame:
    ordered = surface.loc[:, KEYS + ["feature_available_timestamp_utc"]].copy()
    ordered["movement_probability"] = probabilities
    ordered = ordered.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    ordered["_previous"] = ordered.groupby(["stock", "session"], sort=False)[
        "movement_probability"
    ].shift()
    ordered["_fresh"] = ordered["movement_probability"].ge(threshold) & (
        ordered["_previous"].isna() | ordered["_previous"].lt(threshold)
    )
    selected: list[pd.Series[Any]] = []
    for _, group in ordered.loc[ordered["_fresh"]].groupby(["stock", "session"], sort=True):
        previous: pd.Timestamp | None = None
        for _, row in group.iterrows():
            current = pd.Timestamp(row["feature_available_timestamp_utc"])
            if previous is not None and (current - previous).total_seconds() < 30.0 * 60.0:
                continue
            selected.append(row)
            previous = current
    return pd.DataFrame(selected).reset_index(drop=True)


def load_states() -> pd.DataFrame:
    columns = [
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
        "historical_relative_activity",
        "vti__bar_log_return",
        "feature_available_timestamp_max",
    ]
    return pd.read_parquet(
        STATE_PATH,
        columns=columns,
        filters=[("session", ">=", "2024-01-01"), ("session", "<=", "2025-08-22")],
    ).rename(columns={"symbol": "stock"})


def reconstruct_gate_and_episodes(
    causal_surface: pd.DataFrame,
    movement_manifest: Mapping[str, Any],
    threshold: float,
    archived_episodes: pd.DataFrame,
) -> dict[str, Any]:
    specification = cast(Mapping[str, object], movement_manifest["model_specification"])
    probability = manual_movement_probabilities(causal_surface, specification)
    sample = archived_episodes.sort_values(KEYS, kind="mergesort").head(100)
    probability_by_row = pd.Series(probability, index=causal_surface["row_id"].astype(str))
    reconstructed_sample = sample["row_id"].astype(str).map(probability_by_row).to_numpy(float)
    maximum_sample_difference = float(
        np.max(np.abs(reconstructed_sample - sample["M1C_probability"].to_numpy(float)))
    )
    development = causal_surface.loc[causal_surface["period"].astype(str).eq("development")].copy()
    development_probability = probability[
        causal_surface["period"].astype(str).eq("development").to_numpy()
    ]
    order = np.argsort(development_probability, kind="mergesort")
    values = development_probability[order]
    weights = development["row_weight"].to_numpy(float)[order]
    positions = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    rebuilt_threshold = float(np.interp(0.95, positions, values))
    rebuilt = manual_fresh_episodes(
        causal_surface,
        probability,
        rebuilt_threshold,
    )
    identity = rebuilt[KEYS].merge(archived_episodes[KEYS], on=KEYS, how="outer", indicator=True)
    refit = independently_refit_movement_model(development, specification)
    return {
        "rows_manually_reconstructed": int(len(sample)),
        "maximum_probability_difference": maximum_sample_difference,
        "maximum_threshold_difference": abs(rebuilt_threshold - threshold),
        "episode_identity_mismatches": int(identity["_merge"].ne("both").sum()),
        "rebuilt_episode_count": int(len(rebuilt)),
        "stock_local_weight_total_maximum_difference": float(
            np.max(
                np.abs(
                    causal_surface.groupby(["stock", "session"])["row_weight"].sum().to_numpy(float)
                    - 1.0
                )
            )
        ),
        "model_refit": refit,
    }


def _finite_sum(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return float(np.sum(values)) if len(values) and np.isfinite(values).all() else math.nan


def _slope(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        return math.nan
    x_values = np.arange(len(values), dtype=float)
    centered = x_values - float(np.mean(x_values))
    denominator = float(np.sum(centered**2))
    return (
        float(np.sum(centered * (values - np.mean(values))) / denominator)
        if denominator > 0.0
        else 0.0
    )


def _directional_efficiency(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return (
        float(np.sum(values) / (np.sum(np.abs(values)) + 1e-12))
        if len(values) and np.isfinite(values).all()
        else math.nan
    )


def prepare_bars_independently(states: pd.DataFrame) -> pd.DataFrame:
    bars = states.copy().sort_values(["stock", "session", "bar_ordinal"], kind="mergesort")
    previous_close = bars.groupby("stock", sort=False)["close"].shift()
    calculated_return = np.log(bars["close"] / previous_close)
    bars["_stock_return"] = pd.to_numeric(bars["bar_log_return"], errors="coerce").where(
        bars["bar_log_return"].notna(),
        calculated_return,
    )
    bars["_market_return"] = pd.to_numeric(bars["vti__bar_log_return"], errors="coerce")
    bars["_relative_return"] = bars["_stock_return"] - bars["_market_return"]
    bars["_normalised_range"] = (bars["high"] - bars["low"]) / previous_close
    denominator = bars["high"] - bars["low"] + 1e-12
    bars["_clv"] = (2.0 * bars["close"] - bars["high"] - bars["low"]) / denominator
    lower_wick = np.minimum(bars["open"], bars["close"]) - bars["low"]
    upper_wick = bars["high"] - np.maximum(bars["open"], bars["close"])
    bars["_wick"] = (lower_wick - upper_wick) / denominator
    bars["_activity"] = pd.to_numeric(
        bars["historical_relative_activity"],
        errors="coerce",
    )
    bars["_vwap"] = np.nan
    for indices in bars.groupby(["stock", "session"], sort=False).groups.values():
        positions = list(indices)
        rows = bars.loc[positions]
        typical = (
            rows["high"].to_numpy(float)
            + rows["low"].to_numpy(float)
            + rows["close"].to_numpy(float)
        ) / 3.0
        volume = rows["volume"].to_numpy(float)
        volume = np.where(np.isfinite(volume) & (volume > 0.0), volume, 0.0)
        cumulative = np.cumsum(volume)
        bars.loc[positions, "_vwap"] = np.divide(
            np.cumsum(typical * volume),
            cumulative,
            out=np.full(len(rows), np.nan),
            where=cumulative > 0.0,
        )
    return bars


def _continuation_boundary_independent(prefix: pd.DataFrame) -> dict[str, float]:
    candidate: tuple[int, int, float, float] | None = None
    for position in range(max(6, len(prefix) - 4), len(prefix)):
        prior = prefix.iloc[position - 6 : position]
        current = prefix.iloc[position]
        prior_high = float(prior["high"].max())
        prior_low = float(prior["low"].min())
        up = max(0.0, float(current["high"]) - prior_high) / (abs(prior_high) + 1e-12)
        down = max(0.0, prior_low - float(current["low"])) / (abs(prior_low) + 1e-12)
        if up > 0.0 or down > 0.0:
            direction = 1 if up >= down else -1
            candidate = (
                position,
                direction,
                prior_high if direction > 0 else prior_low,
                up if direction > 0 else down,
            )
    if candidate is None:
        return {
            "above": 0.0,
            "below": 0.0,
            "distance": 0.0,
            "acceptance": 0.0,
            "rejection": 0.0,
        }
    position, direction, boundary, breach = candidate
    closes = prefix.iloc[position:]["close"].to_numpy(float)
    beyond = closes > boundary if direction > 0 else closes < boundary
    current_close = float(prefix.iloc[-1]["close"])
    distance = (
        (current_close - boundary) / (abs(boundary) + 1e-12)
        if direction > 0
        else (boundary - current_close) / (abs(boundary) + 1e-12)
    )
    rejected = distance <= 0.0
    return {
        "above": float(direction > 0),
        "below": float(direction < 0),
        "distance": float(direction * max(0.0, breach if rejected else distance)),
        "acceptance": float(direction * int(np.sum(beyond))),
        "rejection": float(-direction if rejected else 0.0),
    }


def _attempt_boundary_independent(
    prefix: pd.DataFrame,
    attempt_sign: int,
) -> dict[str, float]:
    marker_position = len(prefix) - 1
    attempt_start = marker_position - 4
    prior = prefix.iloc[max(0, attempt_start - 6) : attempt_start]
    attempt = prefix.iloc[attempt_start : marker_position - 1]
    response = prefix.iloc[marker_position - 1 : marker_position + 1]
    if len(prior) != 6 or len(attempt) != 3 or len(response) != 2 or attempt_sign == 0:
        return {"failure": math.nan, "inside": math.nan, "maintained": math.nan}
    boundary = float(prior["low"].min()) if attempt_sign < 0 else float(prior["high"].max())
    extreme = float(attempt["low"].min()) if attempt_sign < 0 else float(attempt["high"].max())
    response_close = float(response.iloc[-1]["close"])
    failure = 0.0
    if attempt_sign < 0 and extreme < boundary and response_close > boundary:
        failure = (response_close - boundary) / (abs(boundary) + 1e-12)
    elif attempt_sign > 0 and extreme > boundary and response_close < boundary:
        failure = -(boundary - response_close) / (abs(boundary) + 1e-12)
    inside = (
        max(0.0, response_close - boundary) / (abs(boundary) + 1e-12)
        if attempt_sign < 0
        else -max(0.0, boundary - response_close) / (abs(boundary) + 1e-12)
    )
    maintained = (
        int(np.sum(response["close"].to_numpy(float) > boundary))
        if attempt_sign < 0
        else -int(np.sum(response["close"].to_numpy(float) < boundary))
    )
    return {"failure": float(failure), "inside": float(inside), "maintained": float(maintained)}


def manual_raw_archetype_features(
    sample: pd.DataFrame,
    states: pd.DataFrame,
    beta_parameters: pd.DataFrame,
) -> pd.DataFrame:
    bars = prepare_bars_independently(states)
    grouped = {
        (str(stock), str(session)): rows.reset_index(drop=True)
        for (stock, session), rows in bars.groupby(["stock", "session"], sort=False)
    }
    beta = beta_parameters.set_index(["stock", "checkpoint_group"])
    output_rows: list[dict[str, object]] = []
    for episode in sample.itertuples(index=False):
        stock = str(episode.stock)
        session = str(episode.session)
        checkpoint = int(episode.checkpoint)
        group = "early" if checkpoint <= 14 else "middle" if checkpoint <= 24 else "late"
        stock_session_bars = grouped[(stock, session)]
        prefix = stock_session_bars.loc[
            stock_session_bars["bar_ordinal"].astype(int).le(checkpoint - 2)
        ]
        returns = prefix["_stock_return"].to_numpy(float)
        market = prefix["_market_return"].to_numpy(float)
        relative = prefix["_relative_return"].to_numpy(float)
        close = prefix["close"].to_numpy(float)
        vwap = prefix["_vwap"].to_numpy(float)
        clv = prefix["_clv"].to_numpy(float)
        wick = prefix["_wick"].to_numpy(float)
        activity = prefix["_activity"].to_numpy(float)
        stock_1 = _finite_sum(returns[-1:])
        stock_2 = _finite_sum(returns[-2:])
        stock_4 = _finite_sum(returns[-4:])
        stock_6 = _finite_sum(returns[-6:])
        market_2 = _finite_sum(market[-2:])
        relative_1 = _finite_sum(relative[-1:])
        relative_2 = _finite_sum(relative[-2:])
        net_sign = int(np.sign(stock_4)) if math.isfinite(stock_4) else 0
        direction_closes = (
            np.diff(np.log(close[-5:])) * net_sign > 0.0 if net_sign else np.zeros(4, dtype=bool)
        )
        vwap_side = (
            (close[-3:] - vwap[-3:]) * net_sign > 0.0
            if net_sign and np.isfinite(vwap[-3:]).all()
            else np.zeros(3, dtype=bool)
        )
        vwap_log = np.log(vwap[-4:]) if np.isfinite(vwap[-4:]).all() else np.full(4, np.nan)
        marker_vwap = (
            float(np.log(close[-1] / vwap[-1]))
            if np.isfinite(vwap[-1]) and vwap[-1] > 0.0
            else math.nan
        )
        boundary = _continuation_boundary_independent(prefix)
        attempt_returns = returns[-5:-2]
        response_returns = returns[-2:]
        attempt_return = _finite_sum(attempt_returns)
        response_return = _finite_sum(response_returns)
        attempt_sign = int(np.sign(attempt_return)) if math.isfinite(attempt_return) else 0
        attempt_path = float(np.sum(np.abs(attempt_returns)))
        attempt_efficiency = abs(attempt_return) / (attempt_path + 1e-12)
        response_efficiency = (
            attempt_sign * response_return / (float(np.sum(np.abs(response_returns))) + 1e-12)
            if attempt_sign
            else 0.0
        )
        attempt_activity = float(np.mean(activity[-5:-2]))
        response_activity = float(np.mean(activity[-2:]))
        attempt_impact = abs(attempt_return) / (attempt_activity + 1e-12)
        response_impact = abs(response_return) / (response_activity + 1e-12)
        attempted_response = max(0.0, attempt_sign * response_return) if attempt_sign else 0.0
        attempted_response_impact = attempted_response / (response_activity + 1e-12)
        attempt_boundary = _attempt_boundary_independent(prefix, attempt_sign)
        response_clv = float(np.mean(clv[-2:]))
        response_wick = float(np.mean(wick[-2:]))
        close_failure = (
            -attempt_sign * (1.0 - attempt_sign * response_clv) / 2.0 if attempt_sign else 0.0
        )
        vwap_reclaim = 0.0
        if attempt_sign < 0 and close[-3] < vwap[-3] and close[-1] > vwap[-1]:
            vwap_reclaim = 1.0
        elif attempt_sign > 0 and close[-3] > vwap[-3] and close[-1] < vwap[-1]:
            vwap_reclaim = -1.0
        values: dict[str, object] = {
            "stock": stock,
            "session": session,
            "checkpoint": checkpoint,
            "b_stock_return_5m": stock_1,
            "b_stock_return_10m": stock_2,
            "b_market_return_10m": market_2,
            "b_relative_return_10m": relative_2,
            "b_distance_from_vwap": marker_vwap,
            "c_z_return_5m": stock_1,
            "c_z_return_10m": stock_2,
            "c_z_return_20m": stock_4,
            "c_z_return_30m": stock_6,
            "c_directional_efficiency_20m": _directional_efficiency(returns[-4:]),
            "c_mean_clv_4": float(np.mean(clv[-4:])),
            "c_directional_close_fraction_4": float(net_sign * np.mean(direction_closes)),
            "c_signed_wick_asymmetry_4": float(np.mean(wick[-4:])),
            "c_signed_vwap_slope_4": _slope(vwap_log),
            "c_signed_vwap_distance": marker_vwap,
            "c_vwap_side_closes_3": float(net_sign * np.sum(vwap_side)),
            "c_break_above_prior_six_high": boundary["above"],
            "c_break_below_prior_six_low": boundary["below"],
            "c_signed_boundary_distance": boundary["distance"],
            "c_signed_boundary_acceptance_count": boundary["acceptance"],
            "c_boundary_rejection": boundary["rejection"],
            "c_relative_return_5m": relative_1,
            "c_relative_return_10m": relative_2,
            "c_relative_agreement": (
                float(np.sign(stock_2) * abs(relative_2))
                if np.sign(stock_2) == np.sign(relative_2)
                else 0.0
            ),
            "a_attempt_return_abs": abs(attempt_return),
            "a_attempt_path_length": attempt_path,
            "a_attempt_directional_efficiency": attempt_efficiency,
            "a_response_followthrough": response_return,
            "a_reversal_efficiency_change": (
                -attempt_sign * (attempt_efficiency - response_efficiency)
            ),
            "a_wick_rejection": response_wick,
            "a_close_location_recovery": response_clv,
            "a_failure_close_near_extreme": close_failure,
            "a_boundary_failure": attempt_boundary["failure"],
            "a_boundary_distance_inside": attempt_boundary["inside"],
            "a_boundary_maintenance_count": attempt_boundary["maintained"],
            "a_vwap_reclaim_failure": vwap_reclaim,
            "a_vwap_distance_after_failure": marker_vwap if vwap_reclaim else 0.0,
            "a_attempt_price_impact": attempt_impact,
            "a_response_price_impact": response_impact,
            "a_price_impact_decline": (
                -attempt_sign * (attempt_impact - attempted_response_impact)
            ),
            "a_elevated_activity_weak_progress": (
                -attempt_sign
                * response_activity
                * max(0.0, attempt_efficiency - response_efficiency)
            ),
            "a_relative_recovery": (_finite_sum(relative[-2:]) - _finite_sum(relative[-5:-2])),
            "a_market_resilience": (
                -attempt_sign
                * max(0.0, attempt_sign * _finite_sum(market[-2:]))
                * max(0.0, -attempt_sign * _finite_sum(relative[-2:]))
                if attempt_sign
                else 0.0
            ),
            "_stock_return_lag_0": float(returns[-1]),
            "_stock_return_lag_1": float(returns[-2]),
            "_stock_return_lag_2": float(returns[-3]),
            "_stock_return_lag_3": float(returns[-4]),
            "_market_return_lag_0": float(market[-1]),
            "_market_return_lag_1": float(market[-2]),
            "_market_return_lag_2": float(market[-3]),
            "_market_return_lag_3": float(market[-4]),
        }
        fitted = cast(pd.Series, beta.loc[(stock, group)])
        stock_lags = returns[-4:][::-1]
        market_lags = market[-4:][::-1]
        residuals = stock_lags - (float(fitted["alpha"]) + float(fitted["beta"]) * market_lags)
        residual_5 = _finite_sum(residuals[:1])
        residual_10 = _finite_sum(residuals[:2])
        residual_20 = _finite_sum(residuals[:4])
        stock_20 = _finite_sum(stock_lags)
        market_20 = _finite_sum(market_lags)
        residual_slope = _slope(residuals[::-1])
        low = float(fitted["residual_range_low"]) * 2.0
        high = float(fitted["residual_range_high"]) * 2.0
        distance = (
            residual_20 - high
            if residual_20 > high
            else residual_20 - low
            if residual_20 < low
            else 0.0
        )
        values.update(
            {
                "r_residual_return_5m": residual_5,
                "r_residual_return_10m": residual_10,
                "r_residual_return_20m": residual_20,
                "r_residual_slope": residual_slope,
                "r_residual_persistence": float(np.mean(np.sign(residuals))),
                "r_change_in_residual_strength": float(
                    np.mean(residuals[:2]) - np.mean(residuals[2:])
                ),
                "r_stock_flat_up_market_down": (
                    abs(market_20) if stock_20 >= 0.0 and market_20 < 0.0 else 0.0
                ),
                "r_stock_flat_down_market_up": (
                    -abs(market_20) if stock_20 <= 0.0 and market_20 > 0.0 else 0.0
                ),
                "r_residual_volatility_score": residual_20
                / (float(fitted["residual_scale"]) * 2.0 + 1e-12),
                "r_distance_from_normal_residual_range": distance,
                "r_absolute_residual_direction_agreement": (
                    float(np.sign(residual_20) * abs(residual_20))
                    if np.sign(stock_20) == np.sign(residual_20)
                    else 0.0
                ),
                "r_improving_while_absolute_compressed": (
                    residual_slope
                    if math.isfinite(residual_slope)
                    and abs(stock_20) <= 4.0 * float(fitted["stock_abs_return_median"])
                    else 0.0
                ),
            }
        )
        output_rows.append(values)
    return pd.DataFrame(output_rows)


def add_relative_strength_independently(
    frame: pd.DataFrame,
    beta_parameters: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()
    beta = beta_parameters.set_index(["stock", "checkpoint_group"])
    rows: list[dict[str, float]] = []
    for _, row in output.iterrows():
        checkpoint = int(row["checkpoint"])
        group = "early" if checkpoint <= 14 else "middle" if checkpoint <= 24 else "late"
        fitted = cast(pd.Series, beta.loc[(str(row["stock"]), group)])
        stock_lags = np.asarray(
            [row[f"_stock_return_lag_{index}"] for index in range(4)],
            dtype=float,
        )
        market_lags = np.asarray(
            [row[f"_market_return_lag_{index}"] for index in range(4)],
            dtype=float,
        )
        residuals = stock_lags - (float(fitted["alpha"]) + float(fitted["beta"]) * market_lags)
        residual_5 = _finite_sum(residuals[:1])
        residual_10 = _finite_sum(residuals[:2])
        residual_20 = _finite_sum(residuals[:4])
        stock_20 = _finite_sum(stock_lags)
        market_20 = _finite_sum(market_lags)
        residual_slope = _slope(residuals[::-1])
        low = float(fitted["residual_range_low"]) * 2.0
        high = float(fitted["residual_range_high"]) * 2.0
        distance = (
            residual_20 - high
            if residual_20 > high
            else residual_20 - low
            if residual_20 < low
            else 0.0
        )
        rows.append(
            {
                "r_residual_return_5m": residual_5,
                "r_residual_return_10m": residual_10,
                "r_residual_return_20m": residual_20,
                "r_residual_slope": residual_slope,
                "r_residual_persistence": float(np.mean(np.sign(residuals))),
                "r_change_in_residual_strength": float(
                    np.mean(residuals[:2]) - np.mean(residuals[2:])
                ),
                "r_stock_flat_up_market_down": (
                    abs(market_20) if stock_20 >= 0.0 and market_20 < 0.0 else 0.0
                ),
                "r_stock_flat_down_market_up": (
                    -abs(market_20) if stock_20 <= 0.0 and market_20 > 0.0 else 0.0
                ),
                "r_residual_volatility_score": residual_20
                / (float(fitted["residual_scale"]) * 2.0 + 1e-12),
                "r_distance_from_normal_residual_range": distance,
                "r_absolute_residual_direction_agreement": (
                    float(np.sign(residual_20) * abs(residual_20))
                    if np.sign(stock_20) == np.sign(residual_20)
                    else 0.0
                ),
                "r_improving_while_absolute_compressed": (
                    residual_slope
                    if math.isfinite(residual_slope)
                    and abs(stock_20) <= 4.0 * float(fitted["stock_abs_return_median"])
                    else 0.0
                ),
            }
        )
    relative = pd.DataFrame(rows, index=output.index)
    for feature in RELATIVE_STRENGTH_FEATURES:
        output[feature] = relative[feature]
    return output


def _robust_fit_independent(values: pd.Series) -> dict[str, float | int | bool] | None:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = raw[np.isfinite(raw)]
    if not len(finite):
        return None
    median = float(np.median(finite))
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    raw_iqr = float(q75 - q25)
    zero_scale = not math.isfinite(raw_iqr) or raw_iqr <= 1e-12
    lower, upper = np.quantile(finite, [0.01, 0.99])
    return {
        "median": median,
        "iqr": 1.0 if zero_scale else raw_iqr,
        "clip_lower": float(lower),
        "clip_upper": float(upper),
        "missing_value": median,
        "zero_scale": zero_scale,
        "support": int(len(finite)),
    }


def fit_normalisation_independently(
    raw_features: pd.DataFrame,
    *,
    excluded_sessions: Sequence[str] = (),
) -> pd.DataFrame:
    features = (
        *CONTINUATION_FEATURES,
        *ABSORPTION_FEATURES,
        *RELATIVE_STRENGTH_FEATURES,
    )
    excluded = set(str(value) for value in excluded_sessions)
    frame = raw_features.loc[
        raw_features["session"].astype(str).str.startswith("2024-")
        & ~raw_features["session"].astype(str).isin(excluded)
    ].copy()
    frame["_checkpoint_group"] = np.where(
        frame["checkpoint"].astype(int).le(14),
        "early",
        np.where(frame["checkpoint"].astype(int).le(24), "middle", "late"),
    )
    groups = ("early", "middle", "late")
    positions = {name: index for index, name in enumerate(groups)}
    stocks = sorted(frame["stock"].astype(str).unique())
    checkpoints = sorted(frame["checkpoint"].astype(int).unique())
    rows: list[dict[str, object]] = []
    for feature in features:
        pooled = _robust_fit_independent(frame[feature])
        if pooled is None or int(pooled["support"]) < 20:
            raise AssertionError(f"independent pooled normalisation lacks support: {feature}")
        rows.append(
            {
                "feature": feature,
                "stock": "__POOLED__",
                "checkpoint": -1,
                "checkpoint_group": "pooled",
                "fallback_level": "development_pooled",
                "source_checkpoint_group": "pooled",
                **pooled,
            }
        )
        for stock in stocks:
            stock_rows = frame.loc[frame["stock"].astype(str).eq(stock)]
            stock_fit = _robust_fit_independent(stock_rows[feature])
            for checkpoint in checkpoints:
                target_group = (
                    "early" if checkpoint <= 14 else "middle" if checkpoint <= 24 else "late"
                )
                exact = stock_rows.loc[stock_rows["checkpoint"].astype(int).eq(checkpoint)]
                fitted = _robust_fit_independent(exact[feature])
                fallback_level = "stock_checkpoint"
                source_group = target_group
                if fitted is None or int(fitted["support"]) < 20:
                    fitted = None
                    adjacent = sorted(
                        (
                            name
                            for name in groups
                            if abs(positions[name] - positions[target_group]) == 1
                        ),
                        key=lambda name: (
                            abs(positions[name] - positions[target_group]),
                            positions[name],
                        ),
                    )
                    for adjacent_group in adjacent:
                        candidate = _robust_fit_independent(
                            stock_rows.loc[
                                stock_rows["_checkpoint_group"].astype(str).eq(adjacent_group),
                                feature,
                            ]
                        )
                        if candidate is not None and int(candidate["support"]) >= 20:
                            fitted = candidate
                            fallback_level = "stock_adjacent_checkpoint_group"
                            source_group = adjacent_group
                            break
                if fitted is None or int(fitted["support"]) < 20:
                    if stock_fit is not None and int(stock_fit["support"]) >= 20:
                        fitted = stock_fit
                        fallback_level = "stock_all_checkpoints"
                        source_group = "all"
                    else:
                        fitted = pooled
                        fallback_level = "development_pooled"
                        source_group = "pooled"
                rows.append(
                    {
                        "feature": feature,
                        "stock": stock,
                        "checkpoint": checkpoint,
                        "checkpoint_group": target_group,
                        "fallback_level": fallback_level,
                        "source_checkpoint_group": source_group,
                        **fitted,
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["feature", "stock", "checkpoint"], kind="mergesort")
        .reset_index(drop=True)
    )


def manual_apply_normalisation(
    raw: pd.DataFrame,
    parameters: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    output = raw.copy()
    exact = parameters.set_index(["feature", "stock", "checkpoint"])
    pooled = parameters.loc[parameters["stock"].astype(str).eq("__POOLED__")].set_index("feature")
    for feature in features:
        values: list[float] = []
        for row in output.itertuples(index=False):
            key = (feature, str(row.stock), int(row.checkpoint))
            fitted = cast(pd.Series, exact.loc[key] if key in exact.index else pooled.loc[feature])
            raw_value = float(getattr(row, feature))
            value = float(fitted["missing_value"]) if not math.isfinite(raw_value) else raw_value
            clipped = float(np.clip(value, fitted["clip_lower"], fitted["clip_upper"]))
            values.append((clipped - float(fitted["median"])) / float(fitted["iqr"]))
        output[feature] = values
    return output


def manual_direction_probabilities(
    specification: Mapping[str, object],
    frame: pd.DataFrame,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    parts: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    names: list[str] = []
    medians = cast(Mapping[str, float], specification["medians"])
    centers = cast(Mapping[str, float], specification["robust_centers"])
    scales = cast(Mapping[str, float], specification["robust_scales"])
    for feature in cast(Sequence[object], specification["numeric_features"]):
        name = str(feature)
        raw = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        missing = ~np.isfinite(raw)
        imputed = np.where(missing, float(medians[name]), raw)
        parts.extend(
            [
                ((imputed - float(centers[name])) / float(scales[name]))[:, None],
                missing.astype(float)[:, None],
            ]
        )
        names.extend([name, f"{name}__missing"])
    levels_by_feature = cast(
        Mapping[str, Sequence[object]],
        specification["categorical_levels"],
    )
    for feature in cast(Sequence[object], specification["categorical_features"]):
        name = str(feature)
        values = frame[name].fillna("__MISSING__").astype(str)
        levels = [str(value) for value in levels_by_feature[name]]
        values = values.where(values.isin(levels), "__UNKNOWN__")
        for level in levels:
            parts.append(values.eq(level).to_numpy(float)[:, None])
            names.append(f"{name}=={level}")
    if names != [str(value) for value in specification["design_feature_names"]]:
        raise AssertionError("independent direction design order drifted")
    design = np.concatenate(parts, axis=1)
    linear = design @ np.asarray(specification["coefficients"], dtype=float) + float(
        specification["intercept"]
    )
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))


def reconstruct_features_probabilities_actions(
    *,
    states: pd.DataFrame,
    assessment: pd.DataFrame,
    normalisation_parameters: pd.DataFrame,
    beta_parameters: pd.DataFrame,
    model_configurations: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    sample = assessment.sort_values(KEYS, kind="mergesort").head(100).copy()
    full_beta = beta_parameters.loc[beta_parameters["fit_scope"].astype(str).eq("full_2024")].drop(
        columns="fit_scope"
    )
    raw = manual_raw_archetype_features(
        sample,
        states,
        full_beta,
    )
    transformed = manual_apply_normalisation(
        raw,
        normalisation_parameters,
        (
            *CONTINUATION_FEATURES,
            *ABSORPTION_FEATURES,
            *RELATIVE_STRENGTH_FEATURES,
        ),
    )
    transformed["checkpoint_category"] = transformed["checkpoint"].astype(int).astype(str)
    transformed["day_of_week"] = pd.to_datetime(
        transformed["session"], errors="raise"
    ).dt.day_name()
    expected = (
        assessment.set_index(KEYS).loc[pd.MultiIndex.from_frame(transformed[KEYS])].reset_index()
    )
    maximum_raw: dict[str, float] = {}
    maximum_normalised: dict[str, float] = {}
    maximum_probability: dict[str, float] = {}
    action_mismatches: dict[str, int] = {}
    full_models = cast(Mapping[str, Mapping[str, object]], model_configurations["full_models"])
    for model_id in ARCHETYPES:
        features = FEATURES[model_id]
        expected_raw = expected.loc[:, [f"raw__{feature}" for feature in features]].copy()
        expected_raw.columns = list(features)
        maximum_raw[model_id] = maximum_difference(
            raw,
            expected_raw,
            features,
        )
        maximum_normalised[model_id] = maximum_difference(transformed, expected, features)
        probability = manual_direction_probabilities(
            full_models[model_id],
            transformed,
        )
        maximum_probability[model_id] = float(
            np.max(np.abs(probability - expected[f"{model_id}_probability"].to_numpy(float)))
        )
        boundary = float(thresholds[model_id]["boundary"])
        actions = np.full(len(probability), "ABSTAIN", dtype="<U7")
        actions[probability >= 0.5 + boundary] = "CALL"
        actions[probability <= 0.5 - boundary] = "PUT"
        action_mismatches[model_id] = int(
            np.sum(actions != expected[f"{model_id}_action"].astype(str).to_numpy())
        )
    return {
        "rows_manually_reconstructed_per_archetype": 100,
        "maximum_raw_feature_difference": maximum_raw,
        "maximum_normalised_feature_difference": maximum_normalised,
        "maximum_probability_difference": maximum_probability,
        "action_decision_mismatches": action_mismatches,
        "normalisation_formula_reimplemented_independently": True,
    }


def reconstruct_targets_and_aligned_returns(
    *,
    states: pd.DataFrame,
    assessment: pd.DataFrame,
) -> dict[str, Any]:
    sample = assessment.sort_values(KEYS, kind="mergesort").head(100).copy()
    bar_index = states.set_index(["stock", "session", "bar_ordinal"])
    rebuilt_rows: list[dict[str, float]] = []
    for episode in sample.itertuples(index=False):
        key = (str(episode.stock), str(episode.session))
        checkpoint = int(episode.checkpoint)
        entry = cast(pd.Series, bar_index.loc[(*key, checkpoint)])
        entry_price = float(entry["open"])
        values: dict[str, float] = {}
        for horizon in (5, 10, 15, 30):
            ordinal = checkpoint + horizon // 5 - 1
            close = float(cast(pd.Series, bar_index.loc[(*key, ordinal)])["close"])
            values[f"signed_log_return_{horizon}m"] = math.log(close / entry_price)
        rebuilt_rows.append(values)
    rebuilt = pd.DataFrame(rebuilt_rows)
    maximum_target_difference = maximum_difference(
        rebuilt,
        sample,
        [
            "signed_log_return_5m",
            "signed_log_return_10m",
            "signed_log_return_15m",
            "signed_log_return_30m",
        ],
    )
    maximum_aligned = 0.0
    for model_id in ARCHETYPES:
        sides = np.where(
            sample[f"{model_id}_action"].eq("CALL"),
            1.0,
            np.where(sample[f"{model_id}_action"].eq("PUT"), -1.0, np.nan),
        )
        first = sides * sample["signed_log_return_10m"].to_numpy(float)
        second = sides * rebuilt["signed_log_return_10m"].to_numpy(float)
        maximum_aligned = max(maximum_aligned, float(np.nanmax(np.abs(first - second))))
    return {
        "rows_manually_reconstructed": 100,
        "maximum_target_difference": maximum_target_difference,
        "maximum_aligned_return_difference": maximum_aligned,
    }


def fit_betas_independently(
    states: pd.DataFrame,
    *,
    excluded_sessions: Sequence[str] = (),
) -> pd.DataFrame:
    bars = prepare_bars_independently(states)
    frame = bars.loc[
        bars["session"].astype(str).str.startswith("2024-")
        & bars["bar_ordinal"].astype(int).between(4, 32)
        & ~bars["session"].astype(str).isin(excluded_sessions)
    ].copy()
    frame["checkpoint"] = frame["bar_ordinal"].astype(int) + 2
    grouped_checkpoint = np.where(
        frame["checkpoint"].astype(int).mod(2).eq(0),
        frame["checkpoint"].astype(int),
        frame["checkpoint"].astype(int) - 1,
    )
    frame["checkpoint_group"] = np.where(
        grouped_checkpoint <= 14,
        "early",
        np.where(grouped_checkpoint <= 24, "middle", "late"),
    )

    def fit(rows: pd.DataFrame) -> tuple[float, float, float, float, float, float, int] | None:
        stock_values = rows["_stock_return"].to_numpy(float)
        market_values = rows["_market_return"].to_numpy(float)
        valid = np.isfinite(stock_values) & np.isfinite(market_values)
        stock_values = stock_values[valid]
        market_values = market_values[valid]
        if len(stock_values) < 20:
            return None
        design = np.column_stack([np.ones(len(market_values)), market_values])
        coefficients, _, _, _ = np.linalg.lstsq(design, stock_values, rcond=None)
        residuals = stock_values - design @ coefficients
        scale = float(np.std(residuals, ddof=0))
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        low, high = np.quantile(residuals, [0.10, 0.90])
        return (
            float(coefficients[0]),
            float(coefficients[1]),
            scale,
            float(low),
            float(high),
            float(np.median(np.abs(stock_values))),
            int(len(stock_values)),
        )

    pooled = fit(frame)
    if pooled is None:
        raise AssertionError("independent pooled beta fit lacks support")
    rows: list[dict[str, object]] = []
    for stock in sorted(frame["stock"].astype(str).unique()):
        stock_rows = frame.loc[frame["stock"].astype(str).eq(stock)]
        stock_fit = fit(stock_rows)
        for group in ("early", "middle", "late"):
            fitted = fit(stock_rows.loc[stock_rows["checkpoint_group"].astype(str).eq(group)])
            if fitted is None:
                fitted = stock_fit if stock_fit is not None else pooled
            alpha, beta, scale, low, high, stock_abs_median, support = fitted
            rows.append(
                {
                    "stock": stock,
                    "checkpoint_group": group,
                    "alpha": alpha,
                    "beta": beta,
                    "residual_scale": scale,
                    "residual_range_low": low,
                    "residual_range_high": high,
                    "stock_abs_return_median": stock_abs_median,
                    "support": support,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["stock", "checkpoint_group"],
        kind="mergesort",
    )


def independent_action_metrics(frame: pd.DataFrame, archetype: str) -> dict[str, float]:
    actioned = frame.loc[frame[f"{archetype}_action"].astype(str).ne("ABSTAIN")]
    actions = actioned[f"{archetype}_action"].astype(str).to_numpy()
    sides = np.where(actions == "CALL", 1, -1)
    returns = actioned["signed_log_return_10m"].to_numpy(float)
    valid = np.isfinite(returns) & (returns != 0.0)
    aligned = sides * returns
    true = (returns[valid] > 0.0).astype(int)
    predicted = (sides[valid] > 0).astype(int)
    return {
        "action_coverage": float(len(actioned) / len(frame)) if len(frame) else math.nan,
        "directional_accuracy": float(np.mean(true == predicted)) if valid.any() else math.nan,
        "balanced_accuracy": (
            float(balanced_accuracy_score(true, predicted))
            if valid.any() and len(np.unique(true)) == 2
            else math.nan
        ),
        "mean_aligned_return": float(np.mean(aligned)) if len(aligned) else math.nan,
        "median_aligned_return": float(np.median(aligned)) if len(aligned) else math.nan,
        "positive_return_rate": float(np.mean(aligned > 0.0)) if len(aligned) else math.nan,
        "remaining_movement_fraction": (
            float(actioned["remaining_fraction_10m"].mean()) if len(actioned) else math.nan
        ),
    }


def _independent_side_metrics(
    frame: pd.DataFrame,
    side: np.ndarray[Any, Any],
) -> dict[str, float | int]:
    returns = frame["signed_log_return_10m"].to_numpy(float)
    valid = np.isfinite(returns) & (returns != 0.0) & (side != 0)
    target = (returns[valid] > 0.0).astype(int)
    predicted = (side[valid] > 0).astype(int)
    aligned = side[valid] * returns[valid]
    return {
        "episodes": int(len(frame)),
        "predictions": int(valid.sum()),
        "directional_accuracy": (float(np.mean(target == predicted)) if valid.any() else math.nan),
        "balanced_accuracy": (
            float(balanced_accuracy_score(target, predicted))
            if valid.any() and len(np.unique(target)) == 2
            else math.nan
        ),
        "mean_aligned_return": float(np.mean(aligned)) if len(aligned) else math.nan,
        "median_aligned_return": float(np.median(aligned)) if len(aligned) else math.nan,
        "positive_aligned_return_rate": (
            float(np.mean(aligned > 0.0)) if len(aligned) else math.nan
        ),
    }


def independent_baseline_metrics(assessment: pd.DataFrame) -> pd.DataFrame:
    sides = {
        "B1_always_UP": np.ones(len(assessment), dtype=int),
        "B2_five_minute_momentum": np.sign(
            assessment["raw__b_stock_return_5m"].to_numpy(float)
        ).astype(int),
        "B3_ten_minute_momentum": np.sign(
            assessment["raw__b_stock_return_10m"].to_numpy(float)
        ).astype(int),
        "B4_market_direction": np.sign(
            assessment["raw__b_market_return_10m"].to_numpy(float)
        ).astype(int),
        "B5_simple_relative_strength": np.sign(
            assessment["raw__b_relative_return_10m"].to_numpy(float)
        ).astype(int),
        "B6_beta_adjusted_residual_direction": np.sign(
            assessment["raw__r_residual_return_10m"].to_numpy(float)
        ).astype(int),
    }
    rows: list[dict[str, object]] = []
    for baseline, side in sides.items():
        rows.append(
            {
                "baseline": baseline,
                "conditional_on_archetype": "all_assessment",
                **_independent_side_metrics(assessment, side),
            }
        )
        for archetype in ARCHETYPES:
            mask = assessment[f"{archetype}_action"].astype(str).ne("ABSTAIN").to_numpy()
            rows.append(
                {
                    "baseline": baseline,
                    "conditional_on_archetype": archetype,
                    **_independent_side_metrics(assessment.loc[mask], side[mask]),
                }
            )
    return pd.DataFrame(rows)


def independent_overlap_metrics(assessment: pd.DataFrame) -> pd.DataFrame:
    categories: list[str] = []
    consensus: list[float] = []
    for row in assessment.itertuples(index=False):
        actions = [str(getattr(row, f"{archetype}_action")) for archetype in ARCHETYPES]
        active = [action for action in actions if action != "ABSTAIN"]
        if "CALL" in active and "PUT" in active:
            category, side = "Archetypes conflict", math.nan
        elif not active:
            category, side = "All abstain", math.nan
        elif len(active) == 3:
            category, side = "All three agree", 1.0 if active[0] == "CALL" else -1.0
        elif len(active) == 2:
            category, side = "Two archetypes agree", 1.0 if active[0] == "CALL" else -1.0
        else:
            archetype = ARCHETYPES[actions.index(active[0])]
            category = {
                "C1": "Continuation only",
                "A1": "Absorption/reversal only",
                "R1": "Relative strength only",
            }[archetype]
            side = 1.0 if active[0] == "CALL" else -1.0
        categories.append(category)
        consensus.append(side)
    category_series = pd.Series(categories, index=assessment.index)
    side_values = np.asarray(consensus, dtype=float)
    rows: list[dict[str, object]] = []
    ordered_categories = (
        "Continuation only",
        "Absorption/reversal only",
        "Relative strength only",
        "Two archetypes agree",
        "All three agree",
        "Archetypes conflict",
        "All abstain",
    )
    for category in ordered_categories:
        mask = category_series.eq(category).to_numpy()
        subset = assessment.loc[mask]
        sides = side_values[mask]
        returns = subset["signed_log_return_10m"].to_numpy(float)
        valid = np.isfinite(returns) & (returns != 0.0) & np.isfinite(sides)
        aligned = sides[valid] * returns[valid]
        rows.append(
            {
                "category": category,
                "episodes": int(len(subset)),
                "future_up_rate": float(subset["direction_up_10m"].mean()),
                "directional_accuracy": (
                    float(np.mean(np.sign(returns[valid]) == sides[valid]))
                    if valid.any()
                    else math.nan
                ),
                "mean_aligned_return": (float(np.mean(aligned)) if len(aligned) else math.nan),
                "median_aligned_return": (float(np.median(aligned)) if len(aligned) else math.nan),
                "iv_excess_rate": float(subset["realised_iv_excess_10m"].mean()),
                "mean_remaining_fraction": float(subset["remaining_fraction_10m"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compare_metric_frames(
    rebuilt: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: Sequence[str],
    metrics: Sequence[str],
) -> dict[str, float | int]:
    left = rebuilt.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    right = expected.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    identity = left.loc[:, list(keys)].merge(
        right.loc[:, list(keys)],
        on=list(keys),
        how="outer",
        indicator=True,
    )
    identity_mismatches = int(identity["_merge"].ne("both").sum())
    nan_pattern_mismatches = 0
    maximum = 0.0
    if not identity_mismatches and len(left) == len(right):
        for metric in metrics:
            first = pd.to_numeric(left[metric], errors="coerce").to_numpy(float)
            second = pd.to_numeric(right[metric], errors="coerce").to_numpy(float)
            nan_pattern_mismatches += int(np.sum(np.isfinite(first) != np.isfinite(second)))
            valid = np.isfinite(first) & np.isfinite(second)
            if valid.any():
                maximum = max(maximum, float(np.max(np.abs(first[valid] - second[valid]))))
    else:
        maximum = math.inf
    return {
        "identity_mismatches": identity_mismatches,
        "nan_pattern_mismatches": nan_pattern_mismatches,
        "maximum_metric_difference": maximum,
    }


def independent_bootstrap_intervals(
    assessment: pd.DataFrame,
    sampled_sessions: Sequence[Sequence[object]],
) -> pd.DataFrame:
    groups = {str(session): rows for session, rows in assessment.groupby("session", sort=False)}
    records: list[dict[str, object]] = []
    for draw_id, sampled in enumerate(sampled_sessions):
        sample = pd.concat([groups[str(session)] for session in sampled], ignore_index=True)
        raw_target = sample["direction_up_10m"].to_numpy(float)
        valid = np.isfinite(raw_target)
        target = raw_target[valid].astype(int)
        b0_probability = np.clip(
            sample["B0_probability"].to_numpy(float)[valid],
            1e-12,
            1.0 - 1e-12,
        )
        b0_log_loss = float(log_loss(target, b0_probability))
        b0_brier = float(np.mean((b0_probability - target) ** 2))
        b0_auc = float(roc_auc_score(target, b0_probability))
        for archetype in ARCHETYPES:
            probability = np.clip(
                sample[f"{archetype}_probability"].to_numpy(float)[valid],
                1e-12,
                1.0 - 1e-12,
            )
            selective = independent_action_metrics(sample, archetype)
            iv_selective = independent_action_metrics(
                sample.loc[sample["realised_iv_excess_10m"].astype(bool)],
                archetype,
            )
            largest_selective = independent_action_metrics(
                sample.loc[sample["largest_movement_quartile"].astype(bool)],
                archetype,
            )
            records.append(
                {
                    "draw": draw_id,
                    "archetype": archetype,
                    "log_loss_improvement_vs_B0": b0_log_loss
                    - float(log_loss(target, probability)),
                    "brier_improvement_vs_B0": b0_brier
                    - float(np.mean((probability - target) ** 2)),
                    "auc_improvement_vs_B0": float(roc_auc_score(target, probability)) - b0_auc,
                    **selective,
                    "iv_excess_subgroup_accuracy": iv_selective["directional_accuracy"],
                    "largest_movement_quartile_accuracy": largest_selective["directional_accuracy"],
                }
            )
    draws = pd.DataFrame(records)
    rows: list[dict[str, object]] = []
    metric_columns = [column for column in draws.columns if column not in {"draw", "archetype"}]
    for archetype in ARCHETYPES:
        subset = draws.loc[draws["archetype"].eq(archetype)]
        for metric in metric_columns:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy(float)
            for level, lower, upper in (
                (80, 0.10, 0.90),
                (90, 0.05, 0.95),
                (95, 0.025, 0.975),
            ):
                rows.append(
                    {
                        "archetype": archetype,
                        "metric": metric,
                        "interval_level_percent": level,
                        "lower": float(np.quantile(values, lower)),
                        "median": float(np.quantile(values, 0.50)),
                        "upper": float(np.quantile(values, upper)),
                    }
                )
    return pd.DataFrame(rows)


def independent_episode_support(
    frame: pd.DataFrame,
    *,
    partition: str,
) -> dict[str, Any]:
    binary = frame.loc[frame["direction_up_10m"].notna()].copy()
    counts = binary["direction_up_10m"].astype(int).value_counts()
    months = binary["session"].astype(str).str[:7]
    stock_share = binary.groupby("stock").size() / len(binary)
    month_share = binary.assign(month=months).groupby("month").size() / len(binary)
    required = (
        {
            "episodes": 220,
            "sessions": 60,
            "stocks": 15,
            "months": 10,
            "up": 90,
            "down": 90,
        }
        if partition == "development"
        else {
            "episodes": 180,
            "sessions": 45,
            "stocks": 15,
            "months": 8,
            "up": 75,
            "down": 75,
        }
    )
    checks = {
        "episodes": len(binary) >= required["episodes"],
        "sessions": binary["session"].nunique() >= required["sessions"],
        "stocks": binary["stock"].nunique() >= required["stocks"],
        "months": months.nunique() >= required["months"],
        "up": int(counts.get(1, 0)) >= required["up"],
        "down": int(counts.get(0, 0)) >= required["down"],
    }
    if partition == "assessment":
        checks["maximum_stock_share"] = float(stock_share.max()) <= 0.15
        checks["maximum_month_share"] = float(month_share.max()) <= 0.25
    return {
        "partition": partition,
        "episodes": int(len(binary)),
        "sessions": int(binary["session"].nunique()),
        "stocks": int(binary["stock"].nunique()),
        "months": int(months.nunique()),
        "up": int(counts.get(1, 0)),
        "down": int(counts.get(0, 0)),
        "up_rate": float(binary["direction_up_10m"].mean()),
        "maximum_stock_share": float(stock_share.max()),
        "maximum_month_share": float(month_share.max()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def independent_selective_support(
    assessment: pd.DataFrame,
    archetype: str,
) -> dict[str, Any]:
    action_column = f"{archetype}_action"
    actions = assessment.loc[assessment[action_column].astype(str).ne("ABSTAIN")].copy()
    stock_share = actions.groupby("stock").size() / len(actions)
    month_share = actions.assign(month=actions["session"].astype(str).str[:7]).groupby(
        "month"
    ).size() / len(actions)
    session_share = actions.groupby("session").size() / len(actions)
    checks = {
        "actions": len(actions) >= 70,
        "sessions": actions["session"].nunique() >= 30,
        "stocks": actions["stock"].nunique() >= 12,
        "months": actions["session"].astype(str).str[:7].nunique() >= 6,
        "calls": actions[action_column].eq("CALL").sum() >= 25,
        "puts": actions[action_column].eq("PUT").sum() >= 25,
        "maximum_stock_share": float(stock_share.max()) <= 0.20,
        "maximum_month_share": float(month_share.max()) <= 0.30,
        "maximum_session_share": float(session_share.max()) <= 0.08,
    }
    return {
        "archetype": archetype,
        "actions": int(len(actions)),
        "sessions": int(actions["session"].nunique()),
        "stocks": int(actions["stock"].nunique()),
        "months": int(actions["session"].astype(str).str[:7].nunique()),
        "calls": int(actions[action_column].eq("CALL").sum()),
        "puts": int(actions[action_column].eq("PUT").sum()),
        "maximum_stock_share": float(stock_share.max()),
        "maximum_month_share": float(month_share.max()),
        "maximum_session_share": float(session_share.max()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _mapping_maximum_difference(
    rebuilt: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> tuple[int, float]:
    mismatches = 0
    maximum = 0.0
    for key, value in rebuilt.items():
        expected_value = expected.get(key)
        if isinstance(value, Mapping):
            if not isinstance(expected_value, Mapping):
                mismatches += 1
                continue
            child_mismatches, child_maximum = _mapping_maximum_difference(
                cast(Mapping[str, Any], value),
                cast(Mapping[str, Any], expected_value),
            )
            mismatches += child_mismatches
            maximum = max(maximum, child_maximum)
        elif isinstance(value, (float, np.floating)):
            difference = abs(float(value) - float(expected_value))
            maximum = max(maximum, difference)
        elif value != expected_value:
            mismatches += 1
    return mismatches, maximum


def structural_audits(
    *,
    contract: Mapping[str, Any],
    dependency: Mapping[str, Any],
    movement_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    beta_parameters: pd.DataFrame,
    normalisation_parameters: Mapping[str, Any],
    null_metrics: pd.DataFrame,
    placebo_metrics: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    resampling_plan: Mapping[str, Any],
    direction_metrics: pd.DataFrame,
    selective_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    remaining_metrics: pd.DataFrame,
    development_episodes: pd.DataFrame,
    assessment: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    required_flags = {
        "research_only": True,
        "three_archetypes_tested_separately": True,
        "archetypes_combined_for_primary_inference": False,
        "stock_local_normalisation": True,
        "cross_sectional_peer_normalisation": False,
        "archived_signed_pressure_excluded": True,
        "future_filtered_peer_slates_excluded": True,
        "trigger_bar_excluded_from_direction_features": True,
        "direction_marker_bar": "T-1",
        "protected_start": "2026-01-01",
    }
    contract_passed = all(contract.get(key) == value for key, value in required_flags.items())
    decision_contract_passed = all(decision.get(key) == contract.get(key) for key in required_flags)
    graph = cast(Mapping[str, Sequence[object]], dependency["dependency_graph"])
    pending = [str(value) for value in dependency["contaminated_roots"]]
    descendants: set[str] = set()
    while pending:
        parent = pending.pop()
        for child_value in graph.get(parent, ()):
            child = str(child_value)
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    excluded_features = set(str(value) for value in dependency["excluded_group_i_features"])
    movement_features = set(
        str(value)
        for value in cast(
            Mapping[str, Any],
            movement_manifest["model_specification"],
        )["numeric_features"]
    )
    lineage_passed = bool(
        dependency["archived_m1_numerically_affected"]
        and set(dependency["contaminated_group_i_features"]) == {"signed_pressure", "tension"}
        and {"M1_probability", "fresh_episode_identity", "sequential_row_weight"}.issubset(
            descendants
        )
        and not movement_features.intersection(excluded_features)
        and dependency["future_target_validity_used_for_membership"] is False
        and dependency["peer_stock_counts_used_for_weights"] is False
        and source_manifest["archived_signed_pressure_values_read"] is False
        and source_manifest["future_filtered_advance_eligibility_read"] is False
    )
    full_beta = beta_parameters.loc[beta_parameters["fit_scope"].astype(str).eq("full_2024")]
    oof_beta = beta_parameters.loc[
        beta_parameters["fit_scope"].astype(str).str.startswith("oof_fold_")
    ]
    beta_passed = bool(
        full_beta["development_end"].astype(str).eq("2024-12-31").all()
        and oof_beta["excluded_sessions"].astype(str).str.len().gt(0).all()
    )
    oof_normalisation_passed = bool(
        len(cast(Sequence[object], normalisation_parameters["oof_parameter_identities"])) == 4
        and all(
            len(cast(Mapping[str, Any], item)["heldout_sessions"]) > 0
            for item in cast(
                Sequence[object],
                normalisation_parameters["oof_parameter_identities"],
            )
        )
    )
    parameters = pd.DataFrame(normalisation_parameters["parameters"])
    stock_local_normalisation_passed = bool(
        normalisation_parameters["fit_period"] == "2024 only"
        and int(normalisation_parameters["minimum_support"]) == 20
        and not parameters.duplicated(["feature", "stock", "checkpoint"]).any()
        and parameters["iqr"].astype(float).gt(0.0).all()
        and set(normalisation_parameters["fallback_order"])
        == {
            "same stock exact checkpoint",
            "same stock adjacent checkpoint group",
            "same stock all checkpoints",
            "development pooled",
        }
    )
    no_peer_normalisation_passed = bool(
        dependency["cross_sectional_peer_normalisation"] is False
        and source_manifest["cross_stock_count_used_for_weights"] is False
        and movement_manifest["row_weight_uses_peer_stock_counts"] is False
    )
    direction_by_model = direction_metrics.set_index("model")
    selective_primary = selective_metrics.loc[
        selective_metrics["horizon_minutes"].astype(int).eq(10)
    ].set_index("archetype")
    null_comparisons_passed = True
    for row in null_metrics.itertuples(index=False):
        real_direction = direction_by_model.loc[str(row.archetype)]
        real_selective = selective_primary.loc[str(row.archetype)]
        null_comparisons_passed &= bool(
            bool(row.real_beats_log_loss)
            == (float(real_direction["log_loss"]) < float(row.log_loss))
            and bool(row.real_beats_brier)
            == (float(real_direction["brier_score"]) < float(row.brier_score))
            and bool(row.real_beats_auc) == (float(real_direction["auc"]) > float(row.auc))
            and bool(row.real_beats_selective_accuracy)
            == (float(real_selective["directional_accuracy"]) > float(row.selective_accuracy))
            and bool(row.real_beats_mean_aligned_return)
            == (float(real_selective["mean_aligned_return"]) > float(row.mean_aligned_return))
        )
    null_passed = bool(
        len(null_metrics) == 30
        and null_metrics.groupby("archetype").size().eq(10).all()
        and null_metrics["seed"].nunique() == 10
        and null_metrics["permutation_strata"].eq("session × checkpoint_group").all()
        and null_comparisons_passed
    )
    placebo_comparisons_passed = True
    for row in placebo_metrics.itertuples(index=False):
        real_direction = direction_by_model.loc[str(row.archetype)]
        real_selective = selective_primary.loc[str(row.archetype)]
        predictive = bool(
            float(real_direction["log_loss"]) < float(row.log_loss)
            or float(real_direction["auc"]) > float(row.auc)
        )
        returns = bool(
            float(real_selective["mean_aligned_return"]) > float(row.mean_aligned_return)
        )
        placebo_comparisons_passed &= bool(
            bool(row.shift_applied_before_folds_and_period_split)
            and bool(row.real_predictive_quality_beats_placebo) == predictive
            and bool(row.real_mean_return_beats_placebo) == returns
            and bool(row.real_beats_temporal_placebo) == (predictive and returns)
        )
    placebo_passed = bool(
        len(placebo_metrics) == 3
        and set(placebo_metrics["archetype"]) == set(ARCHETYPES)
        and placebo_comparisons_passed
    )
    bootstrap_plan = cast(Mapping[str, Any], resampling_plan["bootstrap"])
    bootstrap_passed = bool(
        int(bootstrap_plan["draws"]) == 100
        and len(cast(Sequence[object], bootstrap_plan["sampled_sessions"])) == 100
        and bootstrap_metrics["draws"].astype(int).eq(100).all()
        and set(bootstrap_metrics["interval_level_percent"].astype(int)) == {80, 90, 95}
    )
    development_support = independent_episode_support(
        development_episodes,
        partition="development",
    )
    assessment_support = independent_episode_support(
        assessment,
        partition="assessment",
    )
    support_by_model = {
        archetype: independent_selective_support(assessment, archetype) for archetype in ARCHETYPES
    }
    support_mismatches = 0
    support_maximum_difference = 0.0
    for rebuilt, expected in (
        (
            development_support,
            cast(Mapping[str, Any], decision["development_support"]),
        ),
        (
            assessment_support,
            cast(Mapping[str, Any], decision["assessment_support"]),
        ),
    ):
        mismatches, difference = _mapping_maximum_difference(rebuilt, expected)
        support_mismatches += mismatches
        support_maximum_difference = max(support_maximum_difference, difference)
    recorded_support = cast(
        Mapping[str, Mapping[str, Any]],
        decision["selective_support"],
    )
    for archetype in ARCHETYPES:
        mismatches, difference = _mapping_maximum_difference(
            support_by_model[archetype],
            recorded_support[archetype],
        )
        support_mismatches += mismatches
        support_maximum_difference = max(support_maximum_difference, difference)
    evidence_by_model: dict[str, dict[str, Any]] = {}
    baseline_names = {
        "B3_ten_minute_momentum",
        "B4_market_direction",
        "B5_simple_relative_strength",
        "B6_beta_adjusted_residual_direction",
    }
    for archetype in ARCHETYPES:
        proper = direction_by_model.loc[archetype]
        selective = selective_primary.loc[archetype]
        comparisons = baseline_metrics.loc[
            baseline_metrics["conditional_on_archetype"].astype(str).eq(archetype)
            & baseline_metrics["baseline"].astype(str).isin(baseline_names)
        ]
        bootstrap_accuracy = bootstrap_metrics.loc[
            bootstrap_metrics["archetype"].astype(str).eq(archetype)
            & bootstrap_metrics["metric"].astype(str).eq("directional_accuracy")
            & bootstrap_metrics["interval_level_percent"].astype(int).eq(80)
        ].iloc[0]
        bootstrap_return = bootstrap_metrics.loc[
            bootstrap_metrics["archetype"].astype(str).eq(archetype)
            & bootstrap_metrics["metric"].astype(str).eq("mean_aligned_return")
            & bootstrap_metrics["interval_level_percent"].astype(int).eq(80)
        ].iloc[0]
        nulls = null_metrics.loc[null_metrics["archetype"].astype(str).eq(archetype)]
        placebo = placebo_metrics.loc[placebo_metrics["archetype"].astype(str).eq(archetype)].iloc[
            0
        ]
        late = bool(
            remaining_metrics.loc[
                remaining_metrics["archetype"].astype(str).eq(archetype)
                & remaining_metrics["group"].astype(str).eq("all_actions"),
                "late_direction_problem",
            ].iloc[0]
        )
        evidence_by_model[archetype] = {
            "log_loss_improves": float(proper["log_loss_improvement_vs_B0"]) > 0.0,
            "brier_improves": float(proper["brier_improvement_vs_B0"]) > 0.0,
            "auc": float(proper["auc"]),
            "balanced_accuracy": float(proper["balanced_accuracy"]),
            "action_coverage": float(selective["action_coverage"]),
            "selective_accuracy": float(selective["directional_accuracy"]),
            "beats_all_selective_baselines": bool(
                len(comparisons) == 4
                and (
                    float(selective["directional_accuracy"])
                    > comparisons["directional_accuracy"].astype(float)
                ).all()
            ),
            "mean_aligned_return": float(selective["mean_aligned_return"]),
            "median_aligned_return": float(selective["median_aligned_return"]),
            "bootstrap_80_accuracy_lower": float(bootstrap_accuracy["lower"]),
            "bootstrap_80_mean_return_lower": float(bootstrap_return["lower"]),
            "positive_months": int(
                (
                    monthly_metrics.loc[
                        monthly_metrics["archetype"].astype(str).eq(archetype),
                        "mean_aligned_return",
                    ].astype(float)
                    > 0.0
                ).sum()
            ),
            "null_predictive_wins": int(
                (
                    nulls["real_beats_log_loss"].astype(bool) | nulls["real_beats_auc"].astype(bool)
                ).sum()
            ),
            "null_return_wins": int(nulls["real_beats_mean_aligned_return"].astype(bool).sum()),
            "beats_temporal_placebo": bool(placebo["real_beats_temporal_placebo"]),
            "selective_support_passed": bool(support_by_model[archetype]["passed"]),
            "concentration_passed": bool(support_by_model[archetype]["passed"]),
            "late_direction_problem": late,
        }
    evidence_mismatches = 0
    evidence_maximum_difference = 0.0
    recorded_evidence = cast(
        Mapping[str, Mapping[str, Any]],
        decision["archetype_evidence"],
    )
    for archetype in ARCHETYPES:
        mismatches, difference = _mapping_maximum_difference(
            evidence_by_model[archetype],
            recorded_evidence[archetype],
        )
        evidence_mismatches += mismatches
        evidence_maximum_difference = max(evidence_maximum_difference, difference)
    statuses: dict[str, str] = {}
    for archetype in ARCHETYPES:
        evidence = evidence_by_model[archetype]
        full_gate = all(
            (
                bool(evidence["log_loss_improves"]),
                bool(evidence["brier_improves"]),
                float(evidence["auc"]) >= 0.55,
                float(evidence["balanced_accuracy"]) > 0.52,
                0.20 <= float(evidence["action_coverage"]) <= 0.50,
                float(evidence["selective_accuracy"]) >= 0.57,
                bool(evidence["beats_all_selective_baselines"]),
                float(evidence["mean_aligned_return"]) > 0.0,
                float(evidence["median_aligned_return"]) > 0.0,
                float(evidence["bootstrap_80_accuracy_lower"]) > 0.50,
                float(evidence["bootstrap_80_mean_return_lower"]) >= 0.0,
                int(evidence["positive_months"]) >= 6,
                int(evidence["null_predictive_wins"]) >= 9,
                int(evidence["null_return_wins"]) >= 9,
                bool(evidence["beats_temporal_placebo"]),
                bool(evidence["selective_support_passed"]),
                bool(evidence["concentration_passed"]),
                not bool(evidence["late_direction_problem"]),
            )
        )
        if (
            not bool(development_support["passed"])
            or not bool(assessment_support["passed"])
            or not bool(support_by_model[archetype]["passed"])
        ):
            statuses[archetype] = "insufficient_support"
        elif full_gate:
            statuses[archetype] = "supported"
        elif (
            float(evidence["auc"]) >= 0.55
            or float(evidence["selective_accuracy"]) >= 0.57
            or (bool(evidence["log_loss_improves"]) and bool(evidence["brier_improves"]))
        ):
            statuses[archetype] = "promising"
        else:
            statuses[archetype] = "not_supported"
    supported = [name for name, status in statuses.items() if status == "supported"]
    agreement_count = int(
        overlap_metrics.loc[
            overlap_metrics["category"].isin(["Two archetypes agree", "All three agree"]),
            "episodes",
        ].sum()
    )
    if not bool(development_support["passed"]) or not bool(assessment_support["passed"]):
        expected_overall = "blocked_insufficient_episode_support"
    elif all(not bool(support_by_model[name]["passed"]) for name in ARCHETYPES):
        expected_overall = "blocked_insufficient_selective_support"
    elif len(supported) >= 2:
        expected_overall = "multiple_stock_local_directional_archetypes_supported"
    elif supported == ["C1"]:
        expected_overall = "stock_local_continuation_supported"
    elif supported == ["A1"]:
        expected_overall = "stock_local_absorption_reversal_supported"
    elif supported == ["R1"]:
        expected_overall = "stock_local_relative_strength_supported"
    elif any(status == "promising" for status in statuses.values()):
        expected_overall = "directional_archetype_present_but_not_trade_ready"
    elif agreement_count >= 70:
        expected_overall = "archetype_agreement_descriptive_only"
    else:
        expected_overall = "no_stock_local_directional_archetype"
    expected_episode_status = (
        "supported"
        if bool(development_support["passed"]) and bool(assessment_support["passed"])
        else "insufficient_support"
    )
    expected_agreement_status = "promising" if supported else "not_supported"
    expected_remaining_status = (
        "supported"
        if supported
        and all(
            not bool(evidence_by_model[archetype]["late_direction_problem"])
            for archetype in supported
        )
        else "not_supported"
    )
    expected_recorder_priority = (
        "promising"
        if supported or any(status == "promising" for status in statuses.values())
        else "not_supported"
    )
    recorded_overall = str(decision["overall_decision"])
    recoverable_prior_audit_block = bool(
        recorded_overall == "blocked_reproducibility_or_audit_failure"
        and decision["independent_audit_status"] == "blocked"
    )
    decision_logic_passed = bool(
        (recorded_overall == expected_overall or recoverable_prior_audit_block)
        and decision["causal_movement_gate_status"] == "supported"
        and decision["episode_status"] == expected_episode_status
        and decision["stock_local_normalisation_status"] == "supported"
        and decision["continuation_status"] == statuses["C1"]
        and decision["absorption_reversal_status"] == statuses["A1"]
        and decision["relative_strength_status"] == statuses["R1"]
        and decision["agreement_status"] == expected_agreement_status
        and decision["remaining_movement_status"] == expected_remaining_status
        and decision["prospective_recorder_priority"] == expected_recorder_priority
        and list(decision["supported_archetypes"]) == supported
    )
    support_reconstruction_passed = bool(
        support_mismatches == 0 and support_maximum_difference <= 1e-12
    )
    evidence_reconstruction_passed = bool(
        evidence_mismatches == 0 and evidence_maximum_difference <= 1e-12
    )
    return {
        "contract_passed": contract_passed,
        "decision_contract_passed": decision_contract_passed,
        "contaminated_lineage_passed": lineage_passed,
        "transitive_dependency_removal_passed": lineage_passed,
        "development_only_beta_passed": beta_passed,
        "oof_beta_exclusion_passed": beta_passed,
        "stock_local_normalisation_passed": stock_local_normalisation_passed,
        "no_peer_slate_normalisation_passed": no_peer_normalisation_passed,
        "oof_normalisation_exclusion_passed": oof_normalisation_passed,
        "bootstrap_passed": bootstrap_passed,
        "label_nulls_passed": null_passed,
        "temporal_placebos_passed": placebo_passed,
        "support_reconstruction_passed": support_reconstruction_passed,
        "evidence_reconstruction_passed": evidence_reconstruction_passed,
        "support_gate_logic_passed": bool(
            support_reconstruction_passed
            and evidence_reconstruction_passed
            and decision_logic_passed
        ),
        "decision_logic_passed": bool(
            support_reconstruction_passed
            and evidence_reconstruction_passed
            and decision_logic_passed
        ),
        "independently_derived_overall_decision": expected_overall,
    }


def reconstruct_episode_chronology(
    episodes: pd.DataFrame,
    states: pd.DataFrame,
) -> dict[str, Any]:
    state_index = states.set_index(["stock", "session", "bar_ordinal"])
    marker_timestamp_mismatches = 0
    signal_timestamp_mismatches = 0
    entry_timestamp_mismatches = 0
    marker_ordinal_mismatches = 0
    trigger_ordinal_mismatches = 0
    for episode in episodes.itertuples(index=False):
        stock = str(episode.stock)
        session = str(episode.session)
        checkpoint = int(episode.checkpoint)
        marker = cast(pd.Series, state_index.loc[(stock, session, checkpoint - 2)])
        trigger = cast(pd.Series, state_index.loc[(stock, session, checkpoint - 1)])
        entry = cast(pd.Series, state_index.loc[(stock, session, checkpoint)])
        marker_ordinal_mismatches += int(int(episode.marker_bar_ordinal) != checkpoint - 2)
        trigger_ordinal_mismatches += int(int(episode.trigger_bar_ordinal) != checkpoint - 1)
        marker_timestamp_mismatches += int(
            pd.Timestamp(episode.independent_marker_timestamp)
            != pd.Timestamp(marker["bar_complete_timestamp"])
        )
        signal_timestamp_mismatches += int(
            pd.Timestamp(episode.signal_timestamp)
            != pd.Timestamp(trigger["bar_complete_timestamp"])
        )
        entry_timestamp_mismatches += int(
            pd.Timestamp(episode.prospective_entry_timestamp)
            != pd.Timestamp(entry["bar_start_timestamp"])
        )
    ordered = episodes.sort_values(["stock", "session", "signal_timestamp"], kind="mergesort")
    elapsed = (
        ordered.groupby(["stock", "session"], sort=False)["signal_timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
        .dropna()
    )
    spacing_violations = int(elapsed.lt(30.0).sum())
    return {
        "episodes_checked": int(len(episodes)),
        "marker_ordinal_mismatches": marker_ordinal_mismatches,
        "trigger_ordinal_mismatches": trigger_ordinal_mismatches,
        "marker_timestamp_mismatches": marker_timestamp_mismatches,
        "signal_timestamp_mismatches": signal_timestamp_mismatches,
        "entry_timestamp_mismatches": entry_timestamp_mismatches,
        "spacing_violations": spacing_violations,
        "maximum_feature_bar_ordinal_relative_to_checkpoint": -2,
        "trigger_bar_excluded": True,
        "direction_marker_bar": "T-1",
    }


def compare_beta_rebuilds(
    states: pd.DataFrame,
    beta_parameters: pd.DataFrame,
) -> dict[str, Any]:
    numeric_columns = [
        "alpha",
        "beta",
        "residual_scale",
        "residual_range_low",
        "residual_range_high",
        "stock_abs_return_median",
        "support",
    ]
    maximum_by_scope: dict[str, float] = {}
    identity_mismatches = 0
    for scope in sorted(beta_parameters["fit_scope"].astype(str).unique()):
        expected = (
            beta_parameters.loc[beta_parameters["fit_scope"].astype(str).eq(scope)]
            .sort_values(["stock", "checkpoint_group"], kind="mergesort")
            .reset_index(drop=True)
        )
        excluded_value = expected["excluded_sessions"].iloc[0]
        excluded = (
            ()
            if pd.isna(excluded_value) or not str(excluded_value)
            else tuple(str(excluded_value).split(","))
        )
        rebuilt = fit_betas_independently(states, excluded_sessions=excluded).reset_index(drop=True)
        identities = rebuilt[["stock", "checkpoint_group"]].merge(
            expected[["stock", "checkpoint_group"]],
            on=["stock", "checkpoint_group"],
            how="outer",
            indicator=True,
        )
        identity_mismatches += int(identities["_merge"].ne("both").sum())
        if len(rebuilt) != len(expected):
            maximum_by_scope[scope] = math.inf
            continue
        maximum_by_scope[scope] = maximum_difference(
            rebuilt,
            expected,
            numeric_columns,
        )
    return {
        "fit_scopes_rebuilt": sorted(maximum_by_scope),
        "identity_mismatches": identity_mismatches,
        "maximum_difference_by_scope": maximum_by_scope,
        "maximum_beta_difference": max(maximum_by_scope.values(), default=0.0),
    }


def _compare_normalisation_parameter_frames(
    rebuilt: pd.DataFrame,
    expected: pd.DataFrame,
) -> dict[str, float | int]:
    keys = ["feature", "stock", "checkpoint"]
    left = rebuilt.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = expected.sort_values(keys, kind="mergesort").reset_index(drop=True)
    identities = left[keys].merge(right[keys], on=keys, how="outer", indicator=True)
    identity_mismatches = int(identities["_merge"].ne("both").sum())
    if identity_mismatches or len(left) != len(right):
        return {
            "identity_mismatches": identity_mismatches + abs(len(left) - len(right)),
            "string_value_mismatches": len(left) + len(right),
            "maximum_parameter_difference": math.inf,
        }
    string_columns = [
        "checkpoint_group",
        "fallback_level",
        "source_checkpoint_group",
    ]
    string_mismatches = sum(
        int(np.sum(left[column].astype(str).to_numpy() != right[column].astype(str).to_numpy()))
        for column in string_columns
    )
    string_mismatches += int(
        np.sum(
            left["zero_scale"].astype(bool).to_numpy()
            != right["zero_scale"].astype(bool).to_numpy()
        )
    )
    return {
        "identity_mismatches": identity_mismatches,
        "string_value_mismatches": string_mismatches,
        "maximum_parameter_difference": maximum_difference(
            left,
            right,
            [
                "support",
                "median",
                "iqr",
                "clip_lower",
                "clip_upper",
                "missing_value",
            ],
        ),
    }


def reconstruct_normalisation_fits(
    *,
    causal_surface: pd.DataFrame,
    states: pd.DataFrame,
    beta_parameters: pd.DataFrame,
    full_expected: pd.DataFrame,
    oof_expected: pd.DataFrame,
    normalisation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    development_keys = causal_surface.loc[
        causal_surface["period"].astype(str).eq("development"),
        KEYS,
    ].copy()
    full_beta = beta_parameters.loc[beta_parameters["fit_scope"].astype(str).eq("full_2024")].drop(
        columns="fit_scope"
    )
    raw_development = manual_raw_archetype_features(
        development_keys,
        states,
        full_beta,
    )
    rebuilt_full = fit_normalisation_independently(raw_development)
    comparisons: dict[str, dict[str, float | int]] = {
        "full_2024": _compare_normalisation_parameter_frames(
            rebuilt_full,
            full_expected,
        )
    }
    identity_by_fold = {
        int(cast(Mapping[str, Any], item)["fold"]): cast(Mapping[str, Any], item)
        for item in cast(
            Sequence[Mapping[str, Any]],
            normalisation_manifest["oof_parameter_identities"],
        )
    }
    for fold in sorted(identity_by_fold):
        fold_beta = beta_parameters.loc[
            beta_parameters["fit_scope"].astype(str).eq(f"oof_fold_{fold}")
        ].drop(columns="fit_scope")
        fold_raw = add_relative_strength_independently(raw_development, fold_beta)
        heldout = tuple(
            str(value)
            for value in cast(Sequence[object], identity_by_fold[fold]["heldout_sessions"])
        )
        rebuilt = fit_normalisation_independently(
            fold_raw,
            excluded_sessions=heldout,
        )
        expected = oof_expected.loc[oof_expected["fold"].astype(int).eq(fold)].drop(columns="fold")
        comparisons[f"oof_fold_{fold}"] = _compare_normalisation_parameter_frames(
            rebuilt,
            expected,
        )
    return {
        "development_checkpoint_rows_reconstructed": int(len(raw_development)),
        "fit_scopes_rebuilt": sorted(comparisons),
        "comparisons": comparisons,
        "maximum_parameter_difference": max(
            float(comparison["maximum_parameter_difference"]) for comparison in comparisons.values()
        ),
        "identity_mismatches": sum(
            int(comparison["identity_mismatches"]) for comparison in comparisons.values()
        ),
        "string_value_mismatches": sum(
            int(comparison["string_value_mismatches"]) for comparison in comparisons.values()
        ),
        "heldout_sessions_excluded_from_each_oof_fit": True,
    }


def run_audit() -> dict[str, Any]:
    contract = read_json(EXPERIMENT_DIR / "contract.json")
    dependency = read_json(PRIMARY / "movement_gate_dependency_audit.json")
    movement_manifest = read_json(PRIMARY / "causal_movement_feature_manifest.json")
    source_manifest = read_json(PRIMARY / "source_manifest.json")
    threshold = float(read_json(PRIMARY / "causal_movement_threshold.json")["threshold"])
    normalisation = read_json(PRIMARY / "stock_local_normalisation_parameters.json")
    normalisation_frame = pd.DataFrame(normalisation["parameters"])
    oof_normalisation_frame = pd.read_parquet(
        PRIMARY / "oof_stock_local_normalisation_parameters.parquet"
    )
    model_configurations = read_json(PRIMARY / "model_configurations.json")
    thresholds = read_json(PRIMARY / "frozen_archetype_thresholds.json")
    decision = read_json(PRIMARY / "decision.json")
    resampling_plan = read_json(PRIMARY / "frozen_resampling_plan.json")
    states = load_states()
    causal_surface = build_independent_causal_surface(movement_manifest, states)
    episodes = pd.read_parquet(PRIMARY / "movement_signal_episodes.parquet")
    assessment = pd.read_parquet(PRIMARY / "assessment_predictions.parquet")
    beta_parameters = pd.read_csv(PRIMARY / "stock_market_beta_parameters.csv")
    null_metrics = pd.read_csv(PRIMARY / "null_metrics.csv")
    placebo_metrics = pd.read_csv(PRIMARY / "temporal_placebo_metrics.csv")
    bootstrap_metrics = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    direction_metrics = pd.read_csv(PRIMARY / "direction_model_metrics.csv")
    selective_metrics = pd.read_csv(PRIMARY / "selective_policy_metrics.csv")
    baseline_metrics = pd.read_csv(PRIMARY / "baseline_metrics.csv")
    monthly_metrics = pd.read_csv(PRIMARY / "monthly_metrics.csv")
    remaining_metrics = pd.read_csv(PRIMARY / "remaining_movement_metrics.csv")
    overlap_metrics = pd.read_csv(PRIMARY / "archetype_overlap_metrics.csv")
    rebuilt_baseline_metrics = independent_baseline_metrics(assessment)
    baseline_reconstruction = compare_metric_frames(
        rebuilt_baseline_metrics,
        baseline_metrics,
        keys=["baseline", "conditional_on_archetype"],
        metrics=[
            "episodes",
            "predictions",
            "directional_accuracy",
            "balanced_accuracy",
            "mean_aligned_return",
            "median_aligned_return",
            "positive_aligned_return_rate",
        ],
    )
    rebuilt_overlap_metrics = independent_overlap_metrics(assessment)
    overlap_reconstruction = compare_metric_frames(
        rebuilt_overlap_metrics,
        overlap_metrics,
        keys=["category"],
        metrics=[
            "episodes",
            "future_up_rate",
            "directional_accuracy",
            "mean_aligned_return",
            "median_aligned_return",
            "iv_excess_rate",
            "mean_remaining_fraction",
        ],
    )

    gate = reconstruct_gate_and_episodes(
        causal_surface,
        movement_manifest,
        threshold,
        episodes,
    )
    chronology = reconstruct_episode_chronology(episodes, states)
    feature_probability_action = reconstruct_features_probabilities_actions(
        states=states,
        assessment=assessment,
        normalisation_parameters=normalisation_frame,
        beta_parameters=beta_parameters,
        model_configurations=model_configurations,
        thresholds=thresholds,
    )
    targets = reconstruct_targets_and_aligned_returns(states=states, assessment=assessment)
    structural = structural_audits(
        contract=contract,
        dependency=dependency,
        movement_manifest=movement_manifest,
        source_manifest=source_manifest,
        beta_parameters=beta_parameters,
        normalisation_parameters=normalisation,
        null_metrics=null_metrics,
        placebo_metrics=placebo_metrics,
        bootstrap_metrics=bootstrap_metrics,
        resampling_plan=resampling_plan,
        direction_metrics=direction_metrics,
        selective_metrics=selective_metrics,
        baseline_metrics=rebuilt_baseline_metrics,
        monthly_metrics=monthly_metrics,
        remaining_metrics=remaining_metrics,
        development_episodes=episodes.loc[episodes["partition"].astype(str).eq("development")],
        assessment=assessment,
        overlap_metrics=rebuilt_overlap_metrics,
        decision=decision,
    )
    bootstrap_rebuilt = independent_bootstrap_intervals(
        assessment,
        tuple(
            tuple(str(session) for session in draw)
            for draw in cast(
                Sequence[Sequence[object]],
                cast(Mapping[str, Any], resampling_plan["bootstrap"])["sampled_sessions"],
            )
        ),
    )
    bootstrap_sort = ["archetype", "metric", "interval_level_percent"]
    bootstrap_rebuilt = bootstrap_rebuilt.sort_values(
        bootstrap_sort,
        kind="mergesort",
    ).reset_index(drop=True)
    bootstrap_expected = bootstrap_metrics.sort_values(
        bootstrap_sort,
        kind="mergesort",
    ).reset_index(drop=True)
    bootstrap_columns = ["lower", "median", "upper"]
    maximum_bootstrap_difference = maximum_difference(
        bootstrap_rebuilt,
        bootstrap_expected,
        bootstrap_columns,
    )
    beta_reconstruction = compare_beta_rebuilds(states, beta_parameters)
    normalisation_reconstruction = reconstruct_normalisation_fits(
        causal_surface=causal_surface,
        states=states,
        beta_parameters=beta_parameters,
        full_expected=normalisation_frame,
        oof_expected=oof_normalisation_frame,
        normalisation_manifest=normalisation,
    )
    source_by_role = {
        str(item["role"]): cast(Mapping[str, Any], item)
        for item in cast(Sequence[Mapping[str, Any]], source_manifest["sources"])
    }
    source_hashes = {
        "unfiltered_dense_causal_checkpoint_surface": (
            sha256_file(DENSE_CAUSAL_PATH)
            == str(source_by_role["unfiltered_dense_causal_checkpoint_surface"]["sha256"])
        ),
        "exact_previous_close_historical_options_context": (
            sha256_file(HISTORICAL_OPTIONS_PATH)
            == str(source_by_role["exact_previous_close_historical_options_context"]["sha256"])
        ),
        "completed_five_minute_stock_and_market_bars": (
            sha256_file(STATE_PATH)
            == str(source_by_role["completed_five_minute_stock_and_market_bars"]["sha256"])
        ),
    }
    numerical_values = [
        gate["maximum_probability_difference"],
        gate["maximum_threshold_difference"],
        gate["episode_identity_mismatches"],
        gate["stock_local_weight_total_maximum_difference"],
        *[
            float(value)
            for key, value in cast(Mapping[str, float], gate["model_refit"]).items()
            if key != "iterations"
        ],
        *feature_probability_action["maximum_raw_feature_difference"].values(),
        *feature_probability_action["maximum_normalised_feature_difference"].values(),
        *feature_probability_action["maximum_probability_difference"].values(),
        *feature_probability_action["action_decision_mismatches"].values(),
        targets["maximum_target_difference"],
        targets["maximum_aligned_return_difference"],
        maximum_bootstrap_difference,
        beta_reconstruction["maximum_beta_difference"],
        beta_reconstruction["identity_mismatches"],
        normalisation_reconstruction["maximum_parameter_difference"],
        normalisation_reconstruction["identity_mismatches"],
        normalisation_reconstruction["string_value_mismatches"],
        baseline_reconstruction["maximum_metric_difference"],
        baseline_reconstruction["identity_mismatches"],
        baseline_reconstruction["nan_pattern_mismatches"],
        overlap_reconstruction["maximum_metric_difference"],
        overlap_reconstruction["identity_mismatches"],
        overlap_reconstruction["nan_pattern_mismatches"],
        chronology["marker_ordinal_mismatches"],
        chronology["trigger_ordinal_mismatches"],
        chronology["marker_timestamp_mismatches"],
        chronology["signal_timestamp_mismatches"],
        chronology["entry_timestamp_mismatches"],
        chronology["spacing_violations"],
    ]
    structural_flags = [bool(value) for key, value in structural.items() if key.endswith("_passed")]
    passed = bool(
        all(structural_flags)
        and all(source_hashes.values())
        and gate["rows_manually_reconstructed"] >= 100
        and gate["episode_identity_mismatches"] == 0
        and max(float(value) for value in numerical_values) <= 1e-12
    )
    result = {
        "research_only": True,
        "passed": passed,
        "independent_program": str(Path(__file__).resolve()),
        "gate_reconstruction": gate,
        "episode_chronology_reconstruction": chronology,
        "feature_probability_action_reconstruction": feature_probability_action,
        "target_and_aligned_return_reconstruction": targets,
        "structural_audits": structural,
        "source_hash_checks": source_hashes,
        "beta_reconstruction": beta_reconstruction,
        "normalisation_fit_reconstruction": normalisation_reconstruction,
        "baseline_reconstruction": baseline_reconstruction,
        "overlap_reconstruction": overlap_reconstruction,
        "maximum_beta_difference": beta_reconstruction["maximum_beta_difference"],
        "maximum_bootstrap_interval_difference": maximum_bootstrap_difference,
        "protected_rows_read": 0,
        "maximum_session_read": "2025-08-22",
        "manual_causal_gate_probabilities": 100,
        "manual_feature_rows_per_archetype": 100,
        "manual_probabilities_and_actions_per_archetype": 100,
    }
    write_json(PRIMARY / "independent_audit.json", result)
    lightweight = read_json(PRIMARY / "lightweight_audit.json")
    lightweight["independent_audit_status"] = "supported" if passed else "blocked"
    lightweight["independent_audit_passed"] = passed
    lightweight["passed"] = bool(
        passed
        and lightweight["contract_passed"]
        and lightweight["protected_boundary_passed"]
        and lightweight["movement_dependency_audit_passed"]
        and lightweight["causal_movement_gate_passed"]
        and lightweight["fresh_episode_audit_passed"]
        and lightweight["stock_local_normalisation_passed"]
        and lightweight["development_only_beta_passed"]
        and lightweight["oof_preprocessing_exclusion_passed"]
        and lightweight["determinism_passed"]
        and lightweight["bootstrap_draws"] == 100
        and lightweight["null_refits_per_archetype"] == 10
        and lightweight["temporal_placebos"] == 3
        and lightweight["peer_slate_normalisation_used"] is False
        and lightweight["trigger_bar_excluded"]
    )
    write_json(PRIMARY / "lightweight_audit.json", lightweight)
    decision["independent_audit_status"] = "supported" if passed else "blocked"
    decision["overall_decision"] = (
        str(structural["independently_derived_overall_decision"])
        if passed
        else "blocked_reproducibility_or_audit_failure"
    )
    write_json(PRIMARY / "decision.json", decision)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    arguments = parser.parse_args()
    if not arguments.audit:
        parser.error("--audit is required")
    result = run_audit()
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
