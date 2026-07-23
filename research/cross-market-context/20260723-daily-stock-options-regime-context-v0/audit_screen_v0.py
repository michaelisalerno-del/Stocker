#!/usr/bin/env python3
"""Independent fail-closed audit for the Daily Stock x Options Context V0 screen."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"
sys.path.insert(0, str(REPO_ROOT / "packages" / "stocker_data" / "src"))

from stocker_data.calendars import get_market_calendar  # noqa: E402

PROTECTED_START = pd.Timestamp("2025-08-23")
EXPECTED_BLOCKER = "blocked_insufficient_daily_options_coverage"
DAILY_HISTORY_START = pd.Timestamp("2023-10-01")
DAILY_STOCK_RAW_FEATURES = (
    "daily_range_5_to_20",
    "daily_rv_5_to_20",
    "daily_range_overlap_5",
    "daily_efficiency_5",
    "daily_efficiency_10",
    "daily_sign_persistence_5",
    "daily_extension_20",
    "daily_extreme_wick_3",
    "daily_close_location_5",
    "daily_relative_return_5",
    "daily_activity_5_to_20",
)
DAILY_OPTIONS_RAW_FEATURES = (
    "atm_iv",
    "straddle_mid_pct",
    "call_put_iv_gap",
    "skew_25d",
    "front_term_urgency",
    "combined_relative_spread",
    "iv_minus_realised_20d",
    "near_spot_oi_concentration",
    "call_put_oi_imbalance",
)
DAILY_OPTIONS_DIMENSIONS = (
    "options_implied_tension",
    "options_premium_richness",
    "options_downside_asymmetry",
    "options_front_urgency",
    "options_liquidity_stress",
    "options_positioning_concentration",
    "options_directional_positioning",
    "options_surface_disagreement",
)
MISMATCH_FEATURES = (
    "mismatch_compression_vs_iv",
    "mismatch_volatility_vs_urgency",
    "mismatch_route_vs_premium",
    "mismatch_transition_vs_urgency",
    "mismatch_direction_agreement",
    "mismatch_complacent_conflict",
)
DENSE_CHECKPOINTS = (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34)
EXPECTED_SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "quick_daily_context_screen": True,
    "daily_stock_dimensions": True,
    "daily_options_dimensions": True,
    "soft_daily_stock_regimes": True,
    "soft_daily_options_regimes": True,
    "cross_market_mismatch_test": True,
    "previous_close_options_only": True,
    "intraday_option_quotes_used": False,
    "option_pnl_calculated": False,
    "underlying_movement_outcomes_opened": True,
    "directional_outcomes_primary": False,
    "options_loop_discovery_enabled": False,
    "economic_strategy_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "prospective_validation": False,
}
REQUIRED_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "chronology_audit.csv",
    "structural_panel_reconstruction.json",
    "daily_stock_raw_features.parquet",
    "daily_stock_dimensions.parquet",
    "daily_stock_feature_manifest.json",
    "daily_stock_regime_mapping.json",
    "daily_stock_regime_diagnostics.csv",
    "daily_options_raw_features.parquet",
    "daily_options_dimensions.parquet",
    "daily_options_feature_manifest.json",
    "daily_options_regime_mapping.json",
    "daily_options_regime_diagnostics.csv",
    "daily_options_coverage_gap.csv",
    "daily_cross_market_panel.parquet",
    "mismatch_feature_manifest.json",
    "model_configurations.json",
    "model_coefficients.json",
    "assessment_predictions.parquet",
    "test_a_metrics.csv",
    "test_a_monthly_metrics.csv",
    "test_a_regime_metrics.csv",
    "test_b_metrics.csv",
    "test_b_monthly_metrics.csv",
    "test_b_regime_metrics.csv",
    "continuous_residual_metrics.csv",
    "persistence_horizon_metrics.csv",
    "regime_pair_persistence_metrics.csv",
    "dte_horizon_mapping.csv",
    "bootstrap_metrics.csv",
    "options_null_metrics.csv",
    "route_null_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "lightweight_audit.json",
    "determinism_check.json",
    "report.md",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_safety(value: Mapping[str, object], label: str) -> None:
    mismatches = {
        key: (value.get(key), expected)
        for key, expected in EXPECTED_SAFETY_FLAGS.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise AssertionError(f"{label} safety flags differ: {mismatches}")


def z(values: pd.Series, manifest: Mapping[str, Any], name: str) -> pd.Series:
    scale = cast(Mapping[str, Any], cast(Mapping[str, Any], manifest["scales"])[name])
    return (pd.to_numeric(values, errors="raise") - float(scale["center"])) / float(scale["scale"])


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = numerator / denominator.where(denominator.abs().gt(1e-12))
    return values.where(np.isfinite(values), np.nan)


def independent_full_daily_bars() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild exact regular-session daily bars from the recorded local sources."""

    source = read_json(PRIMARY / "source_manifest.json")
    source_values = cast(Mapping[str, Any], source["sources"])
    root = Path(str(source_values["full_regular_session_underlying_root"]))
    cohort = [str(value) for value in cast(list[object], source["cohort"])]
    schedule = (
        get_market_calendar("XNYS")
        .schedule(start_date="2023-10-01", end_date="2025-08-22")
        .rename_axis("session")
        .reset_index()
    )
    schedule["session"] = pd.to_datetime(schedule["session"], errors="raise").dt.date
    source_audit = cast(Mapping[str, Any], source["full_regular_session_underlying"])
    expected_hashes = cast(Mapping[str, Any], source_audit["source_sha256_by_stock"])
    pieces: list[pd.DataFrame] = []
    hash_mismatches = 0
    protected_rows = 0
    for symbol in cohort:
        path = root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        hash_mismatches += int(file_sha256(path) != str(expected_hashes[symbol]))
        raw = pd.read_parquet(
            path,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
            filters=[
                ("timestamp", ">=", pd.Timestamp(DAILY_HISTORY_START, tz="UTC")),
                ("timestamp", "<", pd.Timestamp(PROTECTED_START, tz="UTC")),
            ],
        )
        timestamps = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
        protected_rows += int(timestamps.ge(pd.Timestamp(PROTECTED_START, tz="UTC")).sum())
        working = raw.assign(
            timestamp=timestamps,
            symbol=symbol,
            session=timestamps.dt.tz_convert("America/New_York").dt.date,
        ).merge(schedule, on="session", how="inner", validate="many_to_one")
        working = working.loc[
            working["timestamp"].ge(working["market_open"])
            & working["timestamp"].lt(working["market_close"])
        ].copy()
        elapsed = (working["timestamp"] - working["market_open"]).dt.total_seconds()
        working = working.loc[elapsed.mod(300.0).eq(0.0)].copy()
        pieces.append(working)
    bars = pd.concat(pieces, ignore_index=True)
    finite = np.isfinite(bars[["open", "high", "low", "close"]].to_numpy(float)).all(axis=1)
    complete = (
        bars.assign(_finite=finite)
        .groupby(["symbol", "session"], observed=True)["_finite"]
        .transform("all")
    )
    bars["_activity_valid"] = np.isfinite(
        pd.to_numeric(bars["volume"], errors="coerce").to_numpy(float)
    )
    daily = (
        bars.loc[complete]
        .sort_values(["symbol", "session", "timestamp"], kind="mergesort")
        .groupby(["symbol", "session"], sort=True, observed=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            activity=("volume", "sum"),
            activity_complete=("_activity_valid", "all"),
            bars=("timestamp", "size"),
        )
        .reset_index()
    )
    daily.loc[~daily["activity_complete"].astype(bool), "activity"] = np.nan
    daily = daily.drop(columns="activity_complete")
    daily["session"] = daily["session"].astype(str)
    if hash_mismatches or protected_rows:
        raise AssertionError(
            f"full underlying source hash/protected mismatch: {hash_mismatches}/{protected_rows}"
        )
    return daily, {
        "source_hash_mismatches": hash_mismatches,
        "protected_market_rows_materialised": protected_rows,
        "complete_daily_stock_sessions": len(daily),
    }


def independent_stock_raw_features(daily_bars: pd.DataFrame) -> pd.DataFrame:
    """Calculate the frozen raw surface without importing its implementation."""

    pieces: list[pd.DataFrame] = []
    for symbol, source in daily_bars.groupby("symbol", sort=True, observed=True):
        ordered = source.sort_values("session", kind="mergesort").reset_index(drop=True).copy()
        raw_open = ordered["open"].to_numpy(float)
        raw_high = ordered["high"].to_numpy(float)
        raw_low = ordered["low"].to_numpy(float)
        raw_close = ordered["close"].to_numpy(float)
        multiplier = 1.0
        adjusted_open = np.empty(len(ordered))
        adjusted_high = np.empty(len(ordered))
        adjusted_low = np.empty(len(ordered))
        adjusted_close = np.empty(len(ordered))
        boundaries = np.zeros(len(ordered), dtype=int)
        for index in range(len(ordered)):
            if index:
                open_to_prior_close = raw_open[index] / raw_close[index - 1]
                if open_to_prior_close < 0.55 or open_to_prior_close > 1.80:
                    boundaries[index] = 1
                    multiplier = adjusted_close[index - 1] / raw_open[index]
            adjusted_open[index] = raw_open[index] * multiplier
            adjusted_high[index] = raw_high[index] * multiplier
            adjusted_low[index] = raw_low[index] * multiplier
            adjusted_close[index] = raw_close[index] * multiplier
        open_values = pd.Series(adjusted_open)
        high_values = pd.Series(adjusted_high)
        low_values = pd.Series(adjusted_low)
        close_values = pd.Series(adjusted_close)
        prior_close = close_values.shift(1)
        true_range = pd.concat(
            [
                high_values - low_values,
                (high_values - prior_close).abs(),
                (low_values - prior_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        log_return = pd.Series(np.log(close_values / prior_close))
        range_5 = true_range.rolling(5, min_periods=5).mean()
        range_20 = true_range.rolling(20, min_periods=15).mean()
        rv_5 = log_return.rolling(5, min_periods=5).std(ddof=1)
        rv_20 = log_return.rolling(20, min_periods=15).std(ddof=1)
        prior_high = high_values.shift(1)
        prior_low = low_values.shift(1)
        overlap = pd.Series(
            np.minimum(high_values, prior_high) - np.maximum(low_values, prior_low)
        ).clip(lower=0.0)
        minimum_range = pd.Series(np.minimum(high_values - low_values, prior_high - prior_low))
        absolute_return = log_return.abs()
        directional_5 = safe_ratio(
            log_return.rolling(5, min_periods=5).sum().abs(),
            absolute_return.rolling(5, min_periods=5).sum(),
        )
        directional_10 = safe_ratio(
            log_return.rolling(10, min_periods=10).sum().abs(),
            absolute_return.rolling(10, min_periods=10).sum(),
        )
        ema_20 = close_values.ewm(span=20, adjust=False, min_periods=15).mean()
        daily_range = high_values - low_values
        upper_wick = high_values - pd.Series(np.maximum(open_values, close_values))
        lower_wick = pd.Series(np.minimum(open_values, close_values)) - low_values
        extreme_wick = safe_ratio(
            pd.Series(np.maximum(upper_wick, lower_wick)),
            daily_range,
        )
        rolling_high = high_values.rolling(5, min_periods=5).max()
        rolling_low = low_values.rolling(5, min_periods=5).min()
        ordered["unadjusted_close"] = raw_close
        ordered["inferred_corporate_action_boundary"] = boundaries
        ordered["daily_range_5_to_20"] = safe_ratio(range_5, range_20)
        ordered["daily_rv_5_to_20"] = safe_ratio(rv_5, rv_20)
        ordered["daily_range_overlap_5"] = (
            safe_ratio(overlap, minimum_range).rolling(4, min_periods=4).mean()
        )
        ordered["daily_efficiency_5"] = directional_5
        ordered["daily_efficiency_10"] = directional_10
        ordered["daily_sign_persistence_5"] = (
            pd.Series(np.sign(log_return)).rolling(5, min_periods=5).mean().abs()
        )
        ordered["daily_extension_20"] = safe_ratio(close_values - ema_20, range_20)
        ordered["daily_extreme_wick_3"] = extreme_wick.rolling(3, min_periods=3).mean()
        ordered["daily_close_location_5"] = safe_ratio(
            close_values - rolling_low, rolling_high - rolling_low
        )
        ordered["stock_log_return_5"] = log_return.rolling(5, min_periods=5).sum()
        ordered["daily_activity_5_to_20"] = safe_ratio(
            ordered["activity"].rolling(5, min_periods=5).mean(),
            ordered["activity"].rolling(20, min_periods=15).mean(),
        )
        ordered["realised_volatility_20d"] = rv_20 * math.sqrt(252.0)
        ordered["symbol"] = str(symbol)
        pieces.append(ordered)
    output = pd.concat(pieces, ignore_index=True)
    output["daily_relative_return_5"] = np.nan
    for _session, group in output.groupby("session", sort=True, observed=True):
        for index in group.index:
            peers = group.loc[group.index != index, "stock_log_return_5"].dropna()
            own = output.at[index, "stock_log_return_5"]
            output.at[index, "daily_relative_return_5"] = (
                math.nan if pd.isna(own) else float(own) - float(peers.median())
            )
    return output


def audit_stock_raw_features() -> dict[str, Any]:
    stored = pd.read_parquet(PRIMARY / "daily_stock_raw_features.parquet")
    daily, source_check = independent_full_daily_bars()
    rebuilt = independent_stock_raw_features(daily).rename(
        columns={"session": "stock_information_date"}
    )
    joined = stored.merge(
        rebuilt.loc[
            :,
            [
                "symbol",
                "stock_information_date",
                "unadjusted_close",
                "realised_volatility_20d",
                "inferred_corporate_action_boundary",
                *DAILY_STOCK_RAW_FEATURES,
            ],
        ],
        on=["symbol", "stock_information_date"],
        how="left",
        validate="many_to_one",
        suffixes=("_stored", "_rebuilt"),
        indicator=True,
    )
    source_unavailable = joined["_merge"].ne("both")
    stored_value_columns = [
        "unadjusted_close_stored",
        "realised_volatility_20d_stored",
        *(f"{feature}_stored" for feature in DAILY_STOCK_RAW_FEATURES),
    ]
    unexplained_missing = int(
        joined.loc[source_unavailable, stored_value_columns].notna().any(axis=1).sum()
    )
    maximum_difference = 0.0
    for column in (
        "unadjusted_close",
        "realised_volatility_20d",
        *DAILY_STOCK_RAW_FEATURES,
    ):
        left = pd.to_numeric(joined[f"{column}_stored"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(joined[f"{column}_rebuilt"], errors="coerce").to_numpy(float)
        both_nan = np.isnan(left) & np.isnan(right)
        finite = np.isfinite(left) & np.isfinite(right)
        if bool((~both_nan & ~finite).any()):
            maximum_difference = math.inf
            break
        if bool(finite.any()):
            maximum_difference = max(
                maximum_difference,
                float(np.max(np.abs(left[finite] - right[finite]))),
            )
    comparable_boundaries = joined["_merge"].eq("both")
    boundaries = int(
        (
            joined.loc[comparable_boundaries, "inferred_corporate_action_boundary_stored"].astype(
                int
            )
            != joined.loc[
                comparable_boundaries, "inferred_corporate_action_boundary_rebuilt"
            ].astype(int)
        ).sum()
    )
    if unexplained_missing or boundaries or maximum_difference > 1e-12:
        raise AssertionError(
            "independent daily stock raw reconstruction differs: "
            f"{unexplained_missing}/{boundaries}/{maximum_difference}"
        )
    return {
        **source_check,
        "raw_rows": len(stored),
        "missing_source_rows": int(source_unavailable.sum()),
        "unexplained_missing_source_rows": unexplained_missing,
        "corporate_action_boundary_mismatches": boundaries,
        "maximum_raw_feature_difference": maximum_difference,
    }


def audit_stock_dimensions() -> dict[str, Any]:
    raw_all = pd.read_parquet(PRIMARY / "daily_stock_raw_features.parquet")
    stored_dimensions = pd.read_parquet(PRIMARY / "daily_stock_dimensions.parquet")
    manifest = read_json(PRIMARY / "daily_stock_feature_manifest.json")
    assert_safety(manifest, "daily_stock_feature_manifest")
    stored = stored_dimensions.merge(
        raw_all,
        on=["symbol", "session"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_raw"),
    )
    if len(stored) != len(stored_dimensions):
        raise AssertionError("daily stock dimension row identity changed")
    raw = stored
    directional = 0.5 * (
        z(raw["daily_efficiency_5"], manifest, "daily_efficiency_5")
        + z(raw["daily_efficiency_10"], manifest, "daily_efficiency_10")
    )
    expected = pd.DataFrame(index=raw.index)
    expected["daily_compression"] = (
        -z(raw["daily_range_5_to_20"], manifest, "daily_range_5_to_20")
        - z(raw["daily_rv_5_to_20"], manifest, "daily_rv_5_to_20")
        + z(raw["daily_range_overlap_5"], manifest, "daily_range_overlap_5")
    ) / 3.0
    expected["daily_directional_efficiency"] = directional
    expected["daily_trend_persistence"] = 0.5 * (
        z(raw["daily_sign_persistence_5"], manifest, "daily_sign_persistence_5")
        + z(raw["daily_extension_20"].abs(), manifest, "abs_daily_extension_20")
    )
    expected["daily_extension"] = z(raw["daily_extension_20"], manifest, "daily_extension_20")
    expected["daily_rejection"] = 0.5 * (
        z(raw["daily_extreme_wick_3"], manifest, "daily_extreme_wick_3")
        - (
            directional
            - float(
                cast(
                    Mapping[str, Any],
                    cast(Mapping[str, Any], manifest["scales"])["daily_directional_efficiency"],
                )["center"]
            )
        )
        / float(
            cast(
                Mapping[str, Any],
                cast(Mapping[str, Any], manifest["scales"])["daily_directional_efficiency"],
            )["scale"]
        )
    )
    expected["daily_volatility_acceleration"] = z(
        raw["daily_rv_5_to_20"], manifest, "daily_rv_5_to_20"
    )
    expected["daily_relative_strength"] = z(
        raw["daily_relative_return_5"], manifest, "daily_relative_return_5"
    )
    expected["daily_activity_acceleration"] = z(
        raw["daily_activity_5_to_20"], manifest, "daily_activity_5_to_20"
    )
    columns = list(expected.columns)
    maximum_difference = float(
        np.nanmax(
            np.abs(expected.to_numpy(dtype=float) - stored.loc[:, columns].to_numpy(dtype=float))
        )
    )
    if maximum_difference > 1e-12:
        raise AssertionError(f"daily stock dimensions differ by {maximum_difference}")
    if not (pd.to_datetime(raw["stock_information_date"]) < pd.to_datetime(raw["session"])).all():
        raise AssertionError("same-day daily stock context detected")
    support = cast(Mapping[str, Any], manifest["support"])
    if not bool(support["passed"]):
        raise AssertionError("daily stock support gate did not pass")
    return {
        "raw_rows": len(raw_all),
        "dimension_rows": len(stored),
        "maximum_dimension_difference": maximum_difference,
        "development_scaling_only": manifest["fitted_period"] == "development_2024_only",
    }


def audit_stock_regime() -> dict[str, Any]:
    frame = pd.read_parquet(PRIMARY / "daily_stock_dimensions.parquet")
    mapping = read_json(PRIMARY / "daily_stock_regime_mapping.json")
    assert_safety(mapping, "daily_stock_regime_mapping")
    canonical_names = list(mapping["canonical_dimensions"])
    centroids = [
        cast(Mapping[str, Any], centroid)
        for centroid in cast(list[object], mapping["canonical_centroids"])
    ]
    keys = [tuple(float(centroid[name]) for name in canonical_names) for centroid in centroids]
    if keys != sorted(keys):
        raise AssertionError("daily stock regime IDs are not canonical lexicographic order")
    input_columns = list(mapping["input_columns"])
    weights = np.asarray(mapping["canonical_weights"], dtype=float)
    means = np.asarray(
        [[float(centroid[name]) for name in input_columns] for centroid in centroids],
        dtype=float,
    )
    covariances = np.asarray(mapping["canonical_covariances"], dtype=float)
    sample = frame.head(100)
    x = sample.loc[:, input_columns].to_numpy(dtype=float)
    log_density = np.empty((len(sample), 4), dtype=float)
    for regime in range(4):
        delta = x - means[regime]
        log_density[:, regime] = (
            math.log(weights[regime])
            - 0.5 * np.log(2.0 * math.pi * covariances[regime]).sum()
            - 0.5 * ((delta * delta) / covariances[regime]).sum(axis=1)
        )
    log_density -= log_density.max(axis=1, keepdims=True)
    manual = np.exp(log_density)
    manual /= manual.sum(axis=1, keepdims=True)
    stored = sample.loc[:, [f"daily_stock_regime_p_{regime}" for regime in range(4)]].to_numpy(
        dtype=float
    )
    maximum_difference = float(np.max(np.abs(manual - stored)))
    if maximum_difference > 1e-12:
        raise AssertionError(f"manual stock posterior differs by {maximum_difference}")
    if not bool(mapping["converged"]) or mapping["fitted_period"] != "development_2024_only":
        raise AssertionError("stock GMM convergence or fit-period audit failed")
    return {
        "manual_posterior_rows": len(sample),
        "maximum_manual_posterior_difference": maximum_difference,
        "canonical_ordering": True,
    }


def finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def relative_spread(row: Mapping[str, Any]) -> float:
    bid = finite_number(row.get("bid"))
    ask = finite_number(row.get("ask"))
    midpoint = finite_number(row.get("midpoint"))
    if bid is None or ask is None or midpoint is None or midpoint <= 0.0 or ask < bid:
        return math.inf
    return (ask - bid) / midpoint


def combined_relative_spread(call: Mapping[str, Any], put: Mapping[str, Any]) -> float:
    call_midpoint = finite_number(call.get("midpoint"))
    put_midpoint = finite_number(put.get("midpoint"))
    if call_midpoint is None or put_midpoint is None:
        return math.inf
    denominator = call_midpoint + put_midpoint
    if denominator <= 0.0:
        return math.inf
    return (
        relative_spread(call) * call_midpoint + relative_spread(put) * put_midpoint
    ) / denominator


def independent_atm_pair(
    chain: pd.DataFrame,
    *,
    previous_close: float,
    minimum_dte: int,
    maximum_dte: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], pd.DataFrame]:
    """Reapply the frozen rank without importing the implementation under audit."""

    working = chain.copy()
    working["expiration_date"] = pd.to_datetime(working["expiration_date"], errors="raise").dt.date
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="raise").dt.date
    working["dte"] = (
        pd.to_datetime(working["expiration_date"]) - pd.to_datetime(working["trade_date"])
    ).dt.days
    working["strike"] = pd.to_numeric(working["strike"], errors="raise")
    working["option_type"] = working["option_type"].astype(str).str.casefold()
    eligible = working.loc[
        working["dte"].between(minimum_dte, maximum_dte, inclusive="both")
        & working["strike"].gt(0.0)
        & working["option_type"].isin(["call", "put"])
    ]
    expiries: list[tuple[int, date]] = []
    for expiration, expiration_frame in eligible.groupby(
        "expiration_date", sort=True, observed=True
    ):
        calls = set(expiration_frame.loc[expiration_frame["option_type"].eq("call"), "strike"])
        puts = set(expiration_frame.loc[expiration_frame["option_type"].eq("put"), "strike"])
        if calls.intersection(puts):
            expiries.append(
                (
                    int(expiration_frame["dte"].min()),
                    cast(date, expiration),
                )
            )
    if not expiries:
        raise AssertionError("stored available pair has no independently eligible expiry")
    _, selected_expiration = min(expiries, key=lambda value: (value[0], value[1]))
    front = eligible.loc[eligible["expiration_date"].eq(selected_expiration)].copy()
    records = cast(list[dict[str, Any]], front.to_dict(orient="records"))
    candidates: list[
        tuple[
            tuple[float, float, float, float, float, str, str],
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []
    calls = [row for row in records if row["option_type"] == "call"]
    puts = [row for row in records if row["option_type"] == "put"]
    for call in calls:
        for put in puts:
            strike = float(call["strike"])
            if strike != float(put["strike"]):
                continue
            call_oi = finite_number(call.get("open_interest"))
            put_oi = finite_number(put.get("open_interest"))
            minimum_oi = min(
                -math.inf if call_oi is None else call_oi,
                -math.inf if put_oi is None else put_oi,
            )
            call_iv = finite_number(call.get("implied_volatility"))
            put_iv = finite_number(put.get("implied_volatility"))
            iv_gap = math.inf if call_iv is None or put_iv is None else abs(call_iv - put_iv)
            rank = (
                abs(math.log(strike / previous_close)),
                -minimum_oi,
                combined_relative_spread(call, put),
                iv_gap,
                strike,
                str(call["contract_id"]),
                str(put["contract_id"]),
            )
            candidates.append((rank, call, put))
    if not candidates:
        raise AssertionError("stored available pair has no independent call/put candidate")
    _rank, call, put = min(candidates, key=lambda value: value[0])
    return call, put, front


def audit_options_and_chronology(*, expect_coverage_blocker: bool) -> dict[str, Any]:
    raw = pd.read_parquet(PRIMARY / "daily_options_raw_features.parquet")
    gaps = pd.read_csv(PRIMARY / "daily_options_coverage_gap.csv")
    chronology = pd.read_csv(PRIMARY / "chronology_audit.csv")
    manifest = read_json(PRIMARY / "daily_options_feature_manifest.json")
    assert_safety(manifest, "daily_options_feature_manifest")
    if not chronology["chronology_passed"].astype(bool).all():
        raise AssertionError("chronology audit contains a failure")
    available = raw.loc[raw["pair_available"].astype(bool)]
    if not (
        available["options_observation_date"].astype(str)
        == available["required_options_date"].astype(str)
    ).all():
        raise AssertionError("options observation was not exact D-1")
    if not (
        pd.to_datetime(available["options_observation_date"]) < pd.to_datetime(available["session"])
    ).all():
        raise AssertionError("same-day options context detected")
    if pd.to_datetime(available["options_observation_date"]).ge(PROTECTED_START).any():
        raise AssertionError("protected options observation materialised")
    front_dte = (
        pd.to_datetime(available["front_expiration_date"])
        - pd.to_datetime(available["options_observation_date"])
    ).dt.days
    if not front_dte.between(7, 45).all():
        raise AssertionError("front pair escaped the frozen DTE window")
    back_count = int(available["front_term_urgency"].notna().sum())
    if expect_coverage_blocker:
        unsupported = cast(
            list[str],
            cast(Mapping[str, Any], manifest["support"])["unsupported_development_raw_features"],
        )
        if back_count != 0 or unsupported != ["front_term_urgency"]:
            raise AssertionError("the recorded back-surface blocker is not exact")
        back_gaps = gaps.loc[gaps["gap_component"].eq("back_atm_pair")]
        if len(back_gaps) != len(available):
            raise AssertionError("back-expiry gap manifest does not cover every valid front pair")
    elif manifest.get("status") != "fitted" or back_count == 0:
        raise AssertionError("successful options surface is not fitted or has no back pairs")
    forbidden = [column for column in raw.columns if "pnl" in column.casefold()]
    if forbidden:
        raise AssertionError(f"option PnL columns unexpectedly exist: {forbidden}")
    source = read_json(PRIMARY / "source_manifest.json")
    cache_path = Path(
        str(cast(Mapping[str, Any], source["sources"])["repaired_exact_date_options_cache"])
    )
    cache = pd.read_parquet(cache_path)
    cache["trade_date"] = pd.to_datetime(cache["trade_date"], errors="raise").dt.date
    cache["underlying_symbol"] = cache["underlying_symbol"].astype(str)
    cache_groups = {
        (str(symbol), cast(date, observation_date)): group.copy()
        for (symbol, observation_date), group in cache.groupby(
            ["underlying_symbol", "trade_date"], sort=False, observed=True
        )
    }
    stock_raw = pd.read_parquet(PRIMARY / "daily_stock_raw_features.parquet")
    stock_lookup = stock_raw.set_index(["symbol", "session"])
    maximum_raw_feature_difference = 0.0
    pair_identity_mismatches = 0
    skew_identity_mismatches = 0
    sample = available.head(100)
    for row in sample.itertuples(index=False):
        observation_date = date.fromisoformat(str(row.options_observation_date))
        chain = cache_groups[(str(row.symbol), observation_date)]
        call, put, front = independent_atm_pair(
            chain,
            previous_close=float(row.previous_close_underlying_price),
            minimum_dte=7,
            maximum_dte=45,
        )
        pair_identity_mismatches += int(
            str(call["contract_id"]) != str(row.front_call_contract_id)
            or str(put["contract_id"]) != str(row.front_put_contract_id)
        )
        call_iv = float(call["implied_volatility"])
        put_iv = float(put["implied_volatility"])
        call_midpoint = float(call["midpoint"])
        put_midpoint = float(put["midpoint"])
        expected = {
            "atm_iv": (call_iv + put_iv) / 2.0,
            "straddle_mid_pct": (call_midpoint + put_midpoint)
            / float(row.previous_close_underlying_price),
            "call_put_iv_gap": call_iv - put_iv,
            "combined_relative_spread": combined_relative_spread(call, put),
        }
        if not pd.isna(row.back_call_contract_id) and not pd.isna(row.back_put_contract_id):
            back_call, back_put, _back = independent_atm_pair(
                chain,
                previous_close=float(row.previous_close_underlying_price),
                minimum_dte=46,
                maximum_dte=90,
            )
            pair_identity_mismatches += int(
                str(back_call["contract_id"]) != str(row.back_call_contract_id)
                or str(back_put["contract_id"]) != str(row.back_put_contract_id)
            )
            expected["front_term_urgency"] = (
                expected["atm_iv"]
                - (float(back_call["implied_volatility"]) + float(back_put["implied_volatility"]))
                / 2.0
            )
        realised = float(
            stock_lookup.loc[(str(row.symbol), str(row.session)), "realised_volatility_20d"]
        )
        expected["iv_minus_realised_20d"] = expected["atm_iv"] - realised
        valid_oi = front.loc[pd.to_numeric(front["open_interest"], errors="coerce").ge(0.0)].copy()
        valid_oi["_oi"] = pd.to_numeric(valid_oi["open_interest"], errors="raise")
        total_oi = float(valid_oi["_oi"].sum())
        near = valid_oi["strike"].between(
            float(row.previous_close_underlying_price) * 0.95,
            float(row.previous_close_underlying_price) * 1.05,
            inclusive="both",
        )
        expected["near_spot_oi_concentration"] = float(valid_oi.loc[near, "_oi"].sum()) / total_oi
        call_oi = float(valid_oi.loc[valid_oi["option_type"].eq("call"), "_oi"].sum())
        put_oi = float(valid_oi.loc[valid_oi["option_type"].eq("put"), "_oi"].sum())
        expected["call_put_oi_imbalance"] = math.log((call_oi + 1.0) / (put_oi + 1.0))
        for feature, expected_value in expected.items():
            maximum_raw_feature_difference = max(
                maximum_raw_feature_difference,
                abs(float(getattr(row, feature)) - expected_value),
            )
        for option_type, target, stored_id in (
            ("put", -0.25, row.skew_put_contract_id),
            ("call", 0.25, row.skew_call_contract_id),
        ):
            candidates = []
            for candidate in cast(list[dict[str, Any]], front.to_dict(orient="records")):
                if str(candidate["option_type"]) != option_type:
                    continue
                delta = finite_number(candidate.get("delta"))
                iv = finite_number(candidate.get("implied_volatility"))
                if delta is None or iv is None or iv <= 0.0:
                    continue
                error = abs(delta - target)
                if error <= 0.10:
                    candidates.append(((error, str(candidate["contract_id"])), candidate))
            independent_id = (
                None
                if not candidates
                else str(min(candidates, key=lambda value: value[0])[1]["contract_id"])
            )
            stored_value = None if pd.isna(stored_id) else str(stored_id)
            skew_identity_mismatches += int(independent_id != stored_value)
    if pair_identity_mismatches or skew_identity_mismatches:
        raise AssertionError(
            "independent option-pair/skew identity reconstruction differs: "
            f"{pair_identity_mismatches}/{skew_identity_mismatches}"
        )
    if maximum_raw_feature_difference > 1e-12:
        raise AssertionError(
            "independent options raw-feature reconstruction differs by "
            f"{maximum_raw_feature_difference}"
        )
    return {
        "required_stock_sessions": len(raw),
        "front_pair_stock_sessions": len(available),
        "back_pair_stock_sessions": back_count,
        "gap_rows": len(gaps),
        "bounded_download_gap_rows": int(gaps["bounded_download_required"].astype(bool).sum()),
        "chronology_rows": len(chronology),
        "same_day_options_rows": int(chronology["same_day_options_used"].astype(bool).sum()),
        "independent_pair_rows": len(sample),
        "pair_identity_mismatches": pair_identity_mismatches,
        "skew_identity_mismatches": skew_identity_mismatches,
        "maximum_raw_feature_difference": maximum_raw_feature_difference,
    }


def audit_options_dimensions_and_regime() -> dict[str, Any]:
    raw = pd.read_parquet(PRIMARY / "daily_options_raw_features.parquet")
    dimensions = pd.read_parquet(PRIMARY / "daily_options_dimensions.parquet")
    manifest = read_json(PRIMARY / "daily_options_feature_manifest.json")
    mapping = read_json(PRIMARY / "daily_options_regime_mapping.json")
    available = raw.loc[raw["pair_available"].astype(bool)].copy()
    stored = dimensions.merge(
        available.loc[:, ["symbol", "session", *DAILY_OPTIONS_RAW_FEATURES]],
        on=["symbol", "session"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_raw"),
    )
    imputation = cast(Mapping[str, Any], manifest["imputation_medians"])
    scales = cast(Mapping[str, Any], manifest["scales"])
    values: dict[str, pd.Series] = {}
    for feature in DAILY_OPTIONS_RAW_FEATURES:
        numeric = pd.to_numeric(stored[f"{feature}_raw"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        values[feature] = numeric.fillna(float(imputation[feature]))

    def option_z(feature: str, value: pd.Series | None = None) -> pd.Series:
        scale = cast(Mapping[str, Any], scales[feature])
        source = values[feature] if value is None else value
        return (source - float(scale["center"])) / float(scale["scale"])

    expected = pd.DataFrame(index=stored.index)
    z_atm = option_z("atm_iv")
    z_straddle = option_z("straddle_mid_pct")
    z_gap = option_z("call_put_iv_gap")
    z_skew = option_z("skew_25d")
    z_spread = option_z("combined_relative_spread")
    z_iv_minus_rv = option_z("iv_minus_realised_20d")
    expected["options_implied_tension"] = (z_atm + z_straddle + z_iv_minus_rv) / 3.0
    expected["options_premium_richness"] = (z_straddle + z_iv_minus_rv) / 2.0
    expected["options_downside_asymmetry"] = (z_skew - z_gap) / 2.0
    expected["options_front_urgency"] = option_z("front_term_urgency")
    expected["options_liquidity_stress"] = z_spread
    expected["options_positioning_concentration"] = option_z("near_spot_oi_concentration")
    expected["options_directional_positioning"] = option_z("call_put_oi_imbalance")
    expected["options_surface_disagreement"] = (
        option_z("abs_call_put_iv_gap", values["call_put_iv_gap"].abs())
        + option_z("abs_skew_25d", values["skew_25d"].abs())
        + z_spread
    ) / 3.0
    maximum_dimension_difference = float(
        np.max(
            np.abs(
                expected.loc[:, list(DAILY_OPTIONS_DIMENSIONS)].to_numpy(float)
                - stored.loc[:, list(DAILY_OPTIONS_DIMENSIONS)].to_numpy(float)
            )
        )
    )
    if maximum_dimension_difference > 1e-12:
        raise AssertionError(f"daily options dimensions differ by {maximum_dimension_difference}")
    canonical_names = [str(value) for value in cast(list[object], mapping["canonical_dimensions"])]
    centroids = [
        cast(Mapping[str, Any], value)
        for value in cast(list[object], mapping["canonical_centroids"])
    ]
    canonical_keys = [
        tuple(float(centroid[name]) for name in canonical_names) for centroid in centroids
    ]
    if canonical_keys != sorted(canonical_keys):
        raise AssertionError("daily options regime IDs are not canonical lexicographic order")
    input_columns = [str(value) for value in cast(list[object], mapping["input_columns"])]
    means = np.asarray(mapping["canonical_input_means"], dtype=float)
    covariances = np.asarray(mapping["canonical_covariances"], dtype=float)
    weights = np.asarray(mapping["canonical_weights"], dtype=float)
    sample = stored.head(100)
    x = sample.loc[:, input_columns].to_numpy(float)
    log_density = np.empty((len(sample), 4), dtype=float)
    for regime in range(4):
        delta = x - means[regime]
        log_density[:, regime] = (
            math.log(weights[regime])
            - 0.5 * np.log(2.0 * math.pi * covariances[regime]).sum()
            - 0.5 * ((delta * delta) / covariances[regime]).sum(axis=1)
        )
    log_density -= log_density.max(axis=1, keepdims=True)
    manual = np.exp(log_density)
    manual /= manual.sum(axis=1, keepdims=True)
    stored_probabilities = sample.loc[
        :, [f"daily_options_regime_p_{regime}" for regime in range(4)]
    ].to_numpy(float)
    maximum_posterior_difference = float(np.max(np.abs(manual - stored_probabilities)))
    if maximum_posterior_difference > 1e-12:
        raise AssertionError(f"manual options posterior differs by {maximum_posterior_difference}")
    if (
        manifest["fitted_period"] != "development_2024_only"
        or mapping["fitted_period"] != "development_2024_only"
        or not bool(mapping["converged"])
    ):
        raise AssertionError("options development-only fit or convergence differs")
    return {
        "dimension_rows": len(stored),
        "maximum_dimension_difference": maximum_dimension_difference,
        "manual_posterior_rows": len(sample),
        "maximum_manual_posterior_difference": maximum_posterior_difference,
        "canonical_ordering": True,
    }


def audit_common_artifacts() -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (PRIMARY / name).is_file()]
    if missing:
        raise AssertionError(f"required artifacts missing: {missing}")
    contract = read_json(PRIMARY / "contract.json")
    decision = read_json(PRIMARY / "decision.json")
    source = read_json(PRIMARY / "source_manifest.json")
    protected = read_json(PRIMARY / "protected_boundary_audit.json")
    reconstruction = read_json(PRIMARY / "structural_panel_reconstruction.json")
    for label, value in (
        ("contract", contract),
        ("decision", decision),
        ("source_manifest", source),
        ("protected_boundary_audit", protected),
        ("structural_panel_reconstruction", reconstruction),
    ):
        assert_safety(value, label)
    if int(protected["protected_market_rows_materialised"]) != 0:
        raise AssertionError("protected market rows materialised")
    if int(protected["protected_option_observations_materialised"]) != 0:
        raise AssertionError("protected options rows materialised")
    if not bool(reconstruction["passed"]):
        raise AssertionError("structural reconstruction did not pass")
    for field in (
        "row_identity_mismatches",
        "route_state_mismatches",
        "target_mismatches",
    ):
        if int(reconstruction[field]) != 0:
            raise AssertionError(f"structural reconstruction {field} is nonzero")
    if float(reconstruction["maximum_shared_feature_difference"]) > 1e-12:
        raise AssertionError("shared structural feature reconstruction differs")
    dense_path = Path(str(cast(Mapping[str, Any], source["sources"])["dense_panel"]))
    dense = pd.read_parquet(
        dense_path,
        columns=[
            "row_id",
            "symbol",
            "session",
            "period",
            "checkpoint",
            "registered_completion_next_1_bar",
            "any_prefix_one_transition_from_completion",
            "first_completion_lead",
            "completion_in_bars_2_or_3",
            "advance_eligible",
            "row_weight",
        ],
    )
    if tuple(sorted(dense["checkpoint"].astype(int).unique())) != DENSE_CHECKPOINTS:
        raise AssertionError("frozen dense checkpoints differ")
    if pd.to_datetime(dense["session"], errors="raise").ge(PROTECTED_START).any():
        raise AssertionError("protected structural rows materialised")
    eligibility = dense["registered_completion_next_1_bar"].fillna(0).astype(int).eq(0) & dense[
        "any_prefix_one_transition_from_completion"
    ].fillna(0).astype(int).eq(0)
    stored_eligibility = dense["advance_eligible"].astype(int).eq(1)
    eligibility_mismatches = int(eligibility.ne(stored_eligibility).sum())
    target = pd.to_numeric(dense["first_completion_lead"], errors="raise").isin([2, 3]).astype(int)
    target_mismatches = int(target.ne(dense["completion_in_bars_2_or_3"].astype(int)).sum())
    clean = dense.loc[eligibility].copy()
    stock_counts = clean.groupby(["period", "session"], sort=False)["symbol"].transform("nunique")
    row_counts = clean.groupby(["period", "session", "symbol"], sort=False)["symbol"].transform(
        "size"
    )
    expected_weights = 1.0 / (stock_counts.to_numpy(float) * row_counts.to_numpy(float))
    maximum_weight_difference = float(
        np.max(np.abs(expected_weights - clean["row_weight"].to_numpy(float)))
    )
    session_weight = clean.groupby(["period", "session"], observed=True)["row_weight"].sum()
    maximum_session_weight_difference = float(np.max(np.abs(session_weight.to_numpy(float) - 1.0)))
    if (
        eligibility_mismatches
        or target_mismatches
        or len(clean) != int(reconstruction["clean_advance_rows"])
        or maximum_weight_difference > 1e-12
        or maximum_session_weight_difference > 1e-12
    ):
        raise AssertionError(
            "independent clean target, eligibility, or candidate-normalised weights differ"
        )
    return {
        "required_artifacts": len(REQUIRED_ARTIFACTS),
        "structural_rows": int(reconstruction["clean_advance_rows"]),
        "structural_maximum_difference": float(reconstruction["maximum_shared_feature_difference"]),
        "structural_eligibility_mismatches": eligibility_mismatches,
        "clean_target_mismatches": target_mismatches,
        "maximum_candidate_weight_difference": maximum_weight_difference,
        "maximum_session_weight_difference": maximum_session_weight_difference,
        "overall_decision": decision["overall_decision"],
    }


def audit_blocker_and_artifacts() -> dict[str, Any]:
    common = audit_common_artifacts()
    decision = read_json(PRIMARY / "decision.json")
    source = read_json(PRIMARY / "source_manifest.json")
    if decision["overall_decision"] != EXPECTED_BLOCKER:
        raise AssertionError("coverage blocker was not preserved")
    if int(source["newly_downloaded_records"]) != 0:
        raise AssertionError("source manifest unexpectedly records new option rows")
    if int(source["newly_downloaded_bytes"]) != 0:
        raise AssertionError("source manifest unexpectedly records new option bytes")
    model_configuration = read_json(PRIMARY / "model_configurations.json")
    if model_configuration.get("status") != "not_produced":
        raise AssertionError("cross-market models were fitted despite the coverage blocker")
    return {
        **common,
        "new_option_records": int(source["newly_downloaded_records"]),
    }


def independent_model_prediction(
    frame: pd.DataFrame, specification: Mapping[str, Any]
) -> np.ndarray:
    features = [str(value) for value in cast(list[object], specification["numeric_features"])]
    raw = frame.loc[:, features].to_numpy(float)
    medians = np.asarray(specification["numeric_medians"], dtype=float)
    means = np.asarray(specification["numeric_means"], dtype=float)
    scales = np.asarray(specification["numeric_scales"], dtype=float)
    values = np.where(np.isfinite(raw), raw, medians)
    parts = [(values - means) / scales]
    controls = {
        "stock": frame["symbol"].astype(str),
        "checkpoint": frame["checkpoint"].astype(int).astype(str),
        "month_of_year": pd.to_datetime(frame["session"], errors="raise").dt.strftime("%m"),
        "route_state": frame["route_resolution_state"].astype(str),
    }
    level_mapping = cast(Mapping[str, list[object]], specification["category_levels"])
    for control_value in cast(list[object], specification["category_controls"]):
        control = str(control_value)
        observed = controls[control].to_numpy()
        levels = [str(value) for value in level_mapping[control]]
        for level in levels[1:]:
            parts.append(np.asarray(observed == level, dtype=float)[:, None])
    design = np.concatenate(parts, axis=1)
    coefficients = np.asarray(specification["coefficients"], dtype=float)
    linear = design @ coefficients + float(specification["intercept"])
    if specification["kind"] == "ridge":
        return np.asarray(linear, dtype=float)
    return np.asarray(
        1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))),
        dtype=float,
    )


def audit_success_artifacts() -> dict[str, Any]:
    common = audit_common_artifacts()
    decision = read_json(PRIMARY / "decision.json")
    if str(decision["overall_decision"]).startswith("blocked_"):
        raise AssertionError("success auditor received a blocked decision")
    panel = pd.read_parquet(PRIMARY / "daily_cross_market_panel.parquet")
    predictions = pd.read_parquet(PRIMARY / "assessment_predictions.parquet")
    if panel.empty or predictions.empty:
        raise AssertionError("successful joined panel or predictions are empty")
    if not (
        pd.to_datetime(panel["stock_information_date"]) < pd.to_datetime(panel["session"])
    ).all():
        raise AssertionError("successful panel contains same-day stock context")
    if not (
        panel["options_observation_date"].astype(str) == panel["stock_information_date"].astype(str)
    ).all():
        raise AssertionError("successful panel options and stock clocks differ")

    mismatch_manifest = read_json(PRIMARY / "mismatch_feature_manifest.json")
    standardization = cast(Mapping[str, Mapping[str, Any]], mismatch_manifest["standardization"])

    def mismatch_z(name: str, sample: pd.DataFrame) -> pd.Series:
        values = standardization[name]
        return (pd.to_numeric(sample[name], errors="raise") - float(values["mean"])) / float(
            values["scale"]
        )

    sample_panel = panel.head(100)
    compression = mismatch_z("daily_compression", sample_panel)
    tension = mismatch_z("options_implied_tension", sample_panel)
    volatility = mismatch_z("daily_volatility_acceleration", sample_panel)
    urgency = mismatch_z("options_front_urgency", sample_panel)
    route = mismatch_z("prefix_family_entropy", sample_panel)
    richness = mismatch_z("options_premium_richness", sample_panel)
    transition = mismatch_z("transition_probability", sample_panel)
    pressure = mismatch_z("signed_pressure", sample_panel)
    positioning = mismatch_z("options_directional_positioning", sample_panel)
    expected_mismatch = pd.DataFrame(
        {
            "mismatch_compression_vs_iv": compression - tension,
            "mismatch_volatility_vs_urgency": volatility - urgency,
            "mismatch_route_vs_premium": route - richness,
            "mismatch_transition_vs_urgency": transition - urgency,
            "mismatch_direction_agreement": pressure * positioning,
            "mismatch_complacent_conflict": sample_panel["BROAD_CONFLICT"].astype(int) * (-tension),
        }
    )
    maximum_mismatch_difference = float(
        np.max(
            np.abs(
                expected_mismatch.loc[:, list(MISMATCH_FEATURES)].to_numpy(float)
                - sample_panel.loc[:, list(MISMATCH_FEATURES)].to_numpy(float)
            )
        )
    )
    if maximum_mismatch_difference > 1e-12:
        raise AssertionError(f"mismatch features differ by {maximum_mismatch_difference}")

    configurations = read_json(PRIMARY / "model_configurations.json")
    coefficients = read_json(PRIMARY / "model_coefficients.json")
    models = cast(Mapping[str, Mapping[str, Any]], configurations["models"])
    coefficient_models = cast(Mapping[str, Mapping[str, Any]], coefficients["models"])
    if set(models) != {"S0", "S1", "S2", "O0", "O1", "O2", "R0", "R1"}:
        raise AssertionError("successful fitted model IDs differ")
    if (
        int(configurations["primary_classifier_fit_count"]) != 6
        or int(configurations["ridge_fit_count"]) != 2
    ):
        raise AssertionError("successful model fit counts differ")
    stock_manifest = read_json(PRIMARY / "daily_stock_feature_manifest.json")
    options_manifest = read_json(PRIMARY / "daily_options_feature_manifest.json")
    stock_context = {
        *cast(list[str], stock_manifest["dimensions"]),
        *(f"daily_stock_regime_p_{value}" for value in range(4)),
        "daily_stock_regime_entropy",
        "daily_stock_regime_margin",
    }
    options_context = {
        *cast(list[str], options_manifest["dimensions"]),
        *cast(list[str], options_manifest["missing_indicators"]),
        *(f"daily_options_regime_p_{value}" for value in range(4)),
        "daily_options_regime_entropy",
        "daily_options_regime_margin",
    }
    feature_sets = {
        model: set(str(value) for value in specification["numeric_features"])
        for model, specification in models.items()
    }
    if (
        feature_sets["S1"] - feature_sets["S0"] != stock_context
        or feature_sets["S2"] - feature_sets["S1"] != options_context.union(MISMATCH_FEATURES)
        or feature_sets["O1"] - feature_sets["O0"] != stock_context
        or not set(MISMATCH_FEATURES).issubset(feature_sets["O2"] - feature_sets["O1"])
    ):
        raise AssertionError("S0/S1/S2 or O0/O1/O2 feature nesting differs")
    prediction_sample = predictions.head(100)
    manual_differences: dict[str, float] = {}
    for model, specification in models.items():
        if coefficient_models[model]["coefficients"] != specification["coefficients"]:
            raise AssertionError(f"{model} coefficient artifacts differ")
        manual = independent_model_prediction(prediction_sample, specification)
        stored_prediction = prediction_sample[f"{model}_prediction"].to_numpy(float)
        manual_differences[model] = float(np.max(np.abs(manual - stored_prediction)))
    if len(prediction_sample) < 100 or max(manual_differences.values()) > 1e-12:
        raise AssertionError(f"manual model reconstruction differs: {manual_differences}")

    expected_iv_15m = (
        predictions["atm_iv"].to_numpy(float)
        * math.sqrt(15.0 / (252.0 * 390.0))
        * math.sqrt(2.0 / math.pi)
    )
    maximum_iv_expectation_difference = float(
        np.max(np.abs(expected_iv_15m - predictions["iv_expected_absolute_15m"].to_numpy(float)))
    )
    expected_target = (
        predictions["absolute_log_return_15m"].to_numpy(float) > expected_iv_15m
    ).astype(int)
    iv_target_mismatches = int(
        (expected_target != predictions["movement_exceeds_prior_close_iv_15m"].to_numpy(int)).sum()
    )
    if maximum_iv_expectation_difference > 1e-12 or iv_target_mismatches:
        raise AssertionError("15-minute IV-relative target differs")

    dense_path = Path(
        str(
            cast(Mapping[str, Any], read_json(PRIMARY / "source_manifest.json")["sources"])[
                "dense_panel"
            ]
        )
    )
    dense = pd.read_parquet(
        dense_path,
        columns=[
            "row_id",
            "advance_eligible",
            "registered_completion_next_1_bar",
            "any_prefix_one_transition_from_completion",
            "completion_in_bars_2_or_3",
            "row_weight",
        ],
    )
    clean = dense.loc[
        dense["advance_eligible"].astype(bool)
        & ~dense["registered_completion_next_1_bar"].astype(bool)
        & ~dense["any_prefix_one_transition_from_completion"].astype(bool)
    ]
    target_check = panel.loc[
        :, ["row_id", "registered_completion_clean_bars_2_or_3", "row_weight"]
    ].merge(clean, on="row_id", how="left", validate="one_to_one", suffixes=("", "_dense"))
    clean_target_mismatches = int(
        (
            target_check["registered_completion_clean_bars_2_or_3"].astype(int)
            != target_check["completion_in_bars_2_or_3"].astype(int)
        ).sum()
    )
    maximum_weight_difference = float(
        np.max(
            np.abs(
                target_check["row_weight"].to_numpy(float)
                - target_check["row_weight_dense"].to_numpy(float)
            )
        )
    )
    if clean_target_mismatches or maximum_weight_difference > 1e-12:
        raise AssertionError("clean target or candidate-normalised weight differs")

    bootstrap = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    options_null = pd.read_csv(PRIMARY / "options_null_metrics.csv")
    route_null = pd.read_csv(PRIMARY / "route_null_metrics.csv")
    determinism = read_json(PRIMARY / "determinism_check.json")
    if (
        set(bootstrap["draws"].astype(int)) != {10}
        or set(np.round(bootstrap["confidence"].astype(float), 2)) != {0.8, 0.9, 0.95}
        or len(options_null) != 3
        or len(route_null) != 3
        or options_null["seed"].astype(int).tolist() != [20260723, 20260724, 20260725]
        or route_null["seed"].astype(int).tolist() != [20260726, 20260727, 20260728]
        or not bool(determinism["passed"])
        or int(determinism.get("persistence_result_mismatches", 1)) != 0
        or int(determinism.get("decision_mismatches", 1)) != 0
    ):
        raise AssertionError("bootstrap, null, or expanded determinism audit differs")
    return {
        **common,
        "joined_rows": len(panel),
        "manual_probability_rows_per_model": len(prediction_sample),
        "maximum_manual_prediction_difference_by_model": manual_differences,
        "maximum_mismatch_difference": maximum_mismatch_difference,
        "maximum_iv_expectation_difference": maximum_iv_expectation_difference,
        "iv_target_mismatches": iv_target_mismatches,
        "clean_target_mismatches": clean_target_mismatches,
        "maximum_weight_difference": maximum_weight_difference,
        "bootstrap_rows": len(bootstrap),
        "options_null_refits": len(options_null),
        "route_null_refits": len(route_null),
        "determinism_passed": True,
    }


def render_blocked_report(*, audit_passed: bool) -> str:
    """Render the complete blocker report without implying downstream estimates."""

    decision = read_json(PRIMARY / "decision.json")
    source = read_json(PRIMARY / "source_manifest.json")
    reconstruction = read_json(PRIMARY / "structural_panel_reconstruction.json")
    stock_manifest = read_json(PRIMARY / "daily_stock_feature_manifest.json")
    stock_mapping = read_json(PRIMARY / "daily_stock_regime_mapping.json")
    options_manifest = read_json(PRIMARY / "daily_options_feature_manifest.json")
    protected = read_json(PRIMARY / "protected_boundary_audit.json")
    diagnostics = pd.read_csv(PRIMARY / "daily_stock_regime_diagnostics.csv")
    assessment_regimes = diagnostics.loc[
        diagnostics["diagnostic_type"].eq("regime_summary") & diagnostics["period"].eq("assessment")
    ].sort_values("regime")
    regime_lines: list[str] = []
    centroids = cast(list[Mapping[str, Any]], stock_mapping["canonical_centroids"])
    for position, row in enumerate(assessment_regimes.itertuples(index=False)):
        centroid = centroids[position]
        regime_lines.append(
            "- Regime "
            f"{position}: posterior mass {float(row.posterior_mass):.3f}; "
            f"hard support {int(row.hard_top_regime_rows)} rows, "
            f"{int(row.stocks)} stocks, {int(row.sessions)} sessions, "
            f"{int(row.months)} months. Centroid: compression "
            f"{float(centroid['daily_compression']):+.3f}, volatility acceleration "
            f"{float(centroid['daily_volatility_acceleration']):+.3f}, directional "
            f"efficiency {float(centroid['daily_directional_efficiency']):+.3f}, "
            f"extension {float(centroid['daily_extension']):+.3f}, rejection "
            f"{float(centroid['daily_rejection']):+.3f}, relative strength "
            f"{float(centroid['daily_relative_strength']):+.3f}."
        )
    coverage = cast(Mapping[str, Any], options_manifest["support"])
    download = cast(Mapping[str, Any], source.get("bounded_download", {}))
    cache_reprocessing = cast(Mapping[str, Any], source["options_cache_reprocessing"])
    statuses = "\n".join(
        f"- `{key}`: `{decision[key]}`"
        for key in (
            "daily_stock_regime_status",
            "daily_options_regime_status",
            "test_a_daily_stock_increment_status",
            "test_a_daily_options_increment_status",
            "test_b_daily_stock_increment_status",
            "test_b_intraday_route_increment_status",
            "mismatch_status",
            "persistence_horizon_status",
        )
    )
    stock_support = cast(Mapping[str, Any], stock_manifest["support"])
    stock_raw_rows = len(pd.read_parquet(PRIMARY / "daily_stock_raw_features.parquet"))
    stock_dimension_rows = len(pd.read_parquet(PRIMARY / "daily_stock_dimensions.parquet"))
    independent_audit = read_json(PRIMARY / "independent_audit.json")
    audit_checks = cast(Mapping[str, Any], independent_audit.get("checks", {}))
    stock_regime_check = cast(Mapping[str, Any], audit_checks.get("stock_regime", {}))
    posterior_difference = float(
        stock_regime_check.get("maximum_manual_posterior_difference", math.nan)
    )
    return f"""# Daily Stock × Options Regime Context Quick Screen V0

## Decision

Overall decision: `{decision["overall_decision"]}`.

The repaired previous-close cache passed the front-pair row gate but contained no 46–90 DTE
back-expiry observation in either period. Consequently `front_term_urgency` had zero finite
development values, its required development median did not exist, and the frozen eight-
dimension options surface and four-state options GMM could not be fitted without changing the
preregistered design. The run stopped before all cross-market model fitting.

## Frozen scope and reconstruction

- Development: 2024-01-01 through 2024-12-31.
- Assessment: 2025-01-01 through 2025-08-22.
- Frozen cohort: 20 stocks.
- Clean structural rows: {int(reconstruction["clean_advance_rows"]):,}
  ({int(reconstruction["development_clean_rows"]):,} development and
  {int(reconstruction["assessment_clean_rows"]):,} assessment).
- Assessment clean-completion positives: {int(reconstruction["assessment_clean_positives"]):,}.
- Structural reconstruction: zero row, route-state, and target mismatches; maximum shared
  feature difference {float(reconstruction["maximum_shared_feature_difference"]):.1e}.
- Protected market/options observations materialised:
  {int(protected["protected_market_rows_materialised"])}/
  {int(protected["protected_option_observations_materialised"])}.

## Daily stock context

- Raw stock-session rows: {stock_raw_rows:,}; complete dimension rows:
  {stock_dimension_rows:,}.
- Assessment support: {int(stock_support["assessment_stocks"])} stocks,
  {int(stock_support["assessment_sessions"])} sessions,
  {int(stock_support["assessment_months"])} months; feature retention
  {float(stock_support["daily_stock_feature_retention"]):.1%}.
- Scaling and the four-component diagonal GMM were fitted on 2024 only. All four assessment
  regimes exceeded the 5% posterior-mass, eight-stock, four-month inference gates.

{"\n".join(regime_lines)}

## Previous-close options context and bounded recovery

- Repaired exact-date cache: {int(cache_reprocessing["cache_rows_loaded"]):,}
  rows across {int(cache_reprocessing["cache_stock_dates"]):,}
  stock-dates; maximum cached DTE
  {int(cache_reprocessing["maximum_cached_dte"])}.
- Cache records reused across requested stock-dates: {int(source["options_records_reused"]):,}.
- Valid front pairs: {int(coverage["development_front_pair_stock_sessions"]):,} development
  and {int(coverage["assessment_front_pair_stock_sessions"]):,} assessment stock-sessions.
- Front-pair assessment census: {int(coverage["assessment_clean_checkpoint_rows"]):,} clean
  checkpoint rows, {int(coverage["assessment_sessions"])} sessions,
  {int(coverage["assessment_stocks"])} stocks, {int(coverage["assessment_months"])} months,
  {int(coverage["broad_conflict_rows"]):,} BROAD_CONFLICT rows, and
  {int(coverage["low_route_support_rows"]):,} LOW_ROUTE_SUPPORT rows.
- Back-pair stock-sessions: {int(coverage["back_pair_stock_sessions"])}.
- Exact gap manifest: {int(source["options_gap_rows"]):,} component rows;
  {int(source["bounded_download_required_gap_rows"]):,} require bounded acquisition.
- Bounded plan: {int(download.get("planned_exact_stock_date_requests", 0)):,} exact
  stock-date requests. Status `{download.get("status", "not_run")}`; network requests
  {int(download.get("network_requests_made", 0))}; new records
  {int(download.get("newly_downloaded_records", 0))}; new bytes
  {int(download.get("newly_downloaded_bytes", 0))}.

## Downstream results

The daily options dimensions/regimes, joined cross-market panel, six mismatch distributions,
S0/S1/S2, O0/O1/O2, both Ridge diagnostics, monthly/checkpoint comparisons, persistence
horizons, regime-pair census, DTE-horizon mapping, ten session-bootstrap draws, three
options-null refits, three route-null refits, concentration analysis, and both plots were not
produced. This is a coverage blocker, not evidence for or against any increment.

## Component statuses

{statuses}

## Audit and reproducibility

- Independent fail-closed audit: `{"passed" if audit_passed else "failed"}`.
- Stock posterior reconstruction: 100 rows, maximum difference
  {posterior_difference:.2e}.
- Determinism rebuild: not applicable after the options-coverage stop; recorded as blocked,
  with no redownload, bootstrap, or null repetition.

No result here establishes option profitability, intraday option fills, economic or
directional edge, prospective validation, trading utility, or a deployable strategy.
"""


def run() -> dict[str, Any]:
    decision = read_json(PRIMARY / "decision.json")
    overall_decision = str(decision["overall_decision"])
    coverage_blocked = overall_decision == EXPECTED_BLOCKER
    any_blocker = overall_decision.startswith("blocked_")
    checks: dict[str, Any] = {}
    errors: list[str] = []
    audits: list[tuple[str, Any]] = [("common_artifacts", audit_common_artifacts)]
    if overall_decision not in {
        "blocked_structural_panel_reconstruction_failure",
        "blocked_daily_stock_feature_failure",
        "blocked_protected_boundary_failure",
    }:
        audits.extend(
            [
                ("stock_raw_features", audit_stock_raw_features),
                ("stock_dimensions", audit_stock_dimensions),
                ("stock_regime", audit_stock_regime),
            ]
        )
    if coverage_blocked:
        audits.extend(
            [
                (
                    "options_and_chronology",
                    lambda: audit_options_and_chronology(expect_coverage_blocker=True),
                ),
                ("blocker_and_artifacts", audit_blocker_and_artifacts),
            ]
        )
    elif not any_blocker:
        audits.extend(
            [
                (
                    "options_and_chronology",
                    lambda: audit_options_and_chronology(expect_coverage_blocker=False),
                ),
                ("options_dimensions_and_regime", audit_options_dimensions_and_regime),
                ("successful_models_and_artifacts", audit_success_artifacts),
            ]
        )
    for name, audit in audits:
        try:
            checks[name] = audit()
        except Exception as error:  # fail closed and preserve every discrepancy
            errors.append(f"{name}: {type(error).__name__}: {error}")
    passed = not errors
    result = {
        **EXPECTED_SAFETY_FLAGS,
        "passed": passed,
        "scope": (
            "independent_fail_closed_audit_at_daily_options_coverage_blocker"
            if coverage_blocked
            else (
                "independent_fail_closed_success_audit"
                if not any_blocker
                else "independent_fail_closed_stage_blocker_audit"
            )
        ),
        "checks": checks,
        "errors": errors,
        "downstream_model_audits": (
            "not_run_due_to_preregistered_coverage_blocker"
            if coverage_blocked
            else ("completed" if not any_blocker else "not_run_due_to_stage_blocker")
        ),
        "artifact_hashes": {
            name: file_sha256(PRIMARY / name)
            for name in REQUIRED_ARTIFACTS
            if name not in {"lightweight_audit.json", "report.md"}
        },
    }
    write_json(PRIMARY / "lightweight_audit.json", result)
    write_json(PRIMARY / "independent_audit.json", result)
    if not passed:
        decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
        decision["blocker_detail"] = "; ".join(errors)
        for key in (
            "daily_stock_regime_status",
            "daily_options_regime_status",
            "test_a_daily_stock_increment_status",
            "test_a_daily_options_increment_status",
            "test_b_daily_stock_increment_status",
            "test_b_intraday_route_increment_status",
            "mismatch_status",
            "persistence_horizon_status",
        ):
            decision[key] = "blocked"
        write_json(PRIMARY / "decision.json", decision)
    report_path = PRIMARY / "report.md"
    if coverage_blocked:
        report = render_blocked_report(audit_passed=passed)
        report_path.write_text(report, encoding="utf-8")
        (REPORTS / "report.md").write_text(report, encoding="utf-8")
    elif not any_blocker:
        report = report_path.read_text(encoding="utf-8")
        report = report.replace(
            "- Independent audit: pending standalone auditor execution.",
            f"- Independent audit: `{'passed' if passed else 'failed'}`.",
        )
        report_path.write_text(report, encoding="utf-8")
        (REPORTS / "report.md").write_text(report, encoding="utf-8")
    return result


def main() -> None:
    result = run()
    print(json.dumps({"passed": result["passed"], "errors": result["errors"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
