"""Frozen helpers for the Daily Stock x Options Regime Context Quick Screen V0."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from stocker_research.broad_conflict_options_iv_screen_v0 import (
    _select_atm_pair,
    calculate_primary_option_features,
    previous_trading_session,
    select_primary_atm_pair,
)

DAILY_STOCK_RAW_FEATURES: Final[tuple[str, ...]] = (
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
DAILY_OPTIONS_RAW_FEATURES: Final[tuple[str, ...]] = (
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
DAILY_OPTIONS_MISSING_INDICATORS: Final[tuple[str, ...]] = (
    "skew_missing",
    "back_expiry_missing",
    "oi_concentration_missing",
    "call_put_oi_imbalance_missing",
)
MISMATCH_FEATURES: Final[tuple[str, ...]] = (
    "mismatch_compression_vs_iv",
    "mismatch_volatility_vs_urgency",
    "mismatch_route_vs_premium",
    "mismatch_transition_vs_urgency",
    "mismatch_direction_agreement",
    "mismatch_complacent_conflict",
)
MISMATCH_STANDARDIZATION_INPUTS: Final[tuple[str, ...]] = (
    "daily_compression",
    "options_implied_tension",
    "daily_volatility_acceleration",
    "options_front_urgency",
    "prefix_family_entropy",
    "options_premium_richness",
    "transition_probability",
    "signed_pressure",
    "options_directional_positioning",
)
FROZEN_COHORT: Final[tuple[str, ...]] = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)
DEVELOPMENT_START: Final[date] = date(2024, 1, 1)
DEVELOPMENT_END: Final[date] = date(2024, 12, 31)
ASSESSMENT_START: Final[date] = date(2025, 1, 1)
ASSESSMENT_END: Final[date] = date(2025, 8, 22)
PROTECTED_START: Final[date] = date(2025, 8, 23)
SAFETY_FLAGS: Final[dict[str, bool | str]] = {
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
OVERALL_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "daily_stock_and_options_context_supported_bidirectionally",
        "daily_options_improve_stock_completion_only",
        "daily_stock_context_improves_iv_excess_only",
        "cross_market_mismatch_supported_only",
        "daily_context_descriptive_only",
        "no_daily_context_increment",
        "blocked_structural_panel_reconstruction_failure",
        "blocked_daily_stock_feature_failure",
        "blocked_daily_stock_regime_failure",
        "blocked_daily_options_schema_failure",
        "blocked_insufficient_daily_options_coverage",
        "blocked_daily_options_regime_failure",
        "blocked_protected_boundary_failure",
        "blocked_chronology_or_leakage_failure",
        "blocked_quick_resource_limit",
        "blocked_model_convergence_failure",
        "blocked_reproducibility_or_audit_failure",
    }
)


@dataclass(frozen=True)
class MeanStandardization:
    """One development-only arithmetic z-score transform."""

    mean: float
    scale: float


def previous_us_trading_session(signal_date: date) -> date:
    """Return the exact previous NYSE trading session."""

    return previous_trading_session(signal_date)


def validate_daily_context_chronology(
    *,
    signal_date: date,
    stock_information_date: date,
    options_observation_date: date,
) -> None:
    """Require both daily clocks to end at the exact session before the signal."""

    required = previous_us_trading_session(signal_date)
    if (
        stock_information_date != required
        or options_observation_date != required
        or stock_information_date >= signal_date
        or options_observation_date >= signal_date
    ):
        raise ValueError(
            "daily stock and options context must use the exact previous US trading session"
        )


def reject_protected_observations(
    frame: pd.DataFrame,
    *,
    date_columns: Sequence[str],
) -> None:
    """Reject materialised market/option observations at the protected boundary."""

    for column in date_columns:
        if column not in frame.columns:
            raise ValueError(f"protected-date column missing: {column}")
        values = pd.to_datetime(frame[column], errors="raise").dt.date
        if values.ge(PROTECTED_START).any():
            raise ValueError(f"protected observation materialised in {column}")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = numerator / denominator.where(denominator.abs().gt(1e-12))
    return values.where(np.isfinite(values), np.nan)


def _stock_raw_for_symbol(source: pd.DataFrame) -> pd.DataFrame:
    ordered = source.sort_values("session", kind="mergesort").reset_index(drop=True).copy()
    open_values = ordered["open"].to_numpy(float)
    high_values = ordered["high"].to_numpy(float)
    low_values = ordered["low"].to_numpy(float)
    close_values = ordered["close"].to_numpy(float)
    if not (
        np.isfinite(open_values).all()
        and np.isfinite(high_values).all()
        and np.isfinite(low_values).all()
        and np.isfinite(close_values).all()
        and bool((open_values > 0.0).all())
        and bool((low_values > 0.0).all())
        and bool((high_values >= low_values).all())
    ):
        raise ValueError("daily OHLC values must be finite, positive, and ordered")

    adjusted_open = np.empty(len(ordered), dtype=float)
    adjusted_high = np.empty(len(ordered), dtype=float)
    adjusted_low = np.empty(len(ordered), dtype=float)
    adjusted_close = np.empty(len(ordered), dtype=float)
    boundaries = np.zeros(len(ordered), dtype=np.int64)
    multiplier = 1.0
    for index in range(len(ordered)):
        if index:
            ratio = open_values[index] / close_values[index - 1]
            if ratio < 0.55 or ratio > 1.80:
                boundaries[index] = 1
                multiplier = adjusted_close[index - 1] / open_values[index]
        adjusted_open[index] = open_values[index] * multiplier
        adjusted_high[index] = high_values[index] * multiplier
        adjusted_low[index] = low_values[index] * multiplier
        adjusted_close[index] = close_values[index] * multiplier

    ordered["unadjusted_close"] = close_values
    ordered["adjusted_open"] = adjusted_open
    ordered["adjusted_high"] = adjusted_high
    ordered["adjusted_low"] = adjusted_low
    ordered["adjusted_close"] = adjusted_close
    ordered["inferred_corporate_action_boundary"] = boundaries

    adjusted_open_series = ordered["adjusted_open"]
    adjusted_high_series = ordered["adjusted_high"]
    adjusted_low_series = ordered["adjusted_low"]
    close = ordered["adjusted_close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            adjusted_high_series - adjusted_low_series,
            (adjusted_high_series - previous_close).abs(),
            (adjusted_low_series - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    log_return = cast(pd.Series, np.log(close / previous_close))
    range_5 = true_range.rolling(5, min_periods=5).mean()
    range_20 = true_range.rolling(20, min_periods=15).mean()
    rv_5 = log_return.rolling(5, min_periods=5).std(ddof=1)
    rv_20 = log_return.rolling(20, min_periods=15).std(ddof=1)

    prior_high = adjusted_high_series.shift(1)
    prior_low = adjusted_low_series.shift(1)
    overlap = pd.Series(
        np.minimum(adjusted_high_series, prior_high) - np.maximum(adjusted_low_series, prior_low),
        index=ordered.index,
        dtype=float,
    ).clip(lower=0.0)
    minimum_range = pd.Series(
        np.minimum(adjusted_high_series - adjusted_low_series, prior_high - prior_low),
        index=ordered.index,
        dtype=float,
    )
    overlap_fraction = _safe_ratio(overlap, minimum_range)

    absolute_return = log_return.abs()
    net_5 = log_return.rolling(5, min_periods=5).sum().abs()
    net_10 = log_return.rolling(10, min_periods=10).sum().abs()
    efficiency_5 = _safe_ratio(net_5, absolute_return.rolling(5, min_periods=5).sum())
    efficiency_10 = _safe_ratio(net_10, absolute_return.rolling(10, min_periods=10).sum())
    sign_persistence = (
        pd.Series(np.sign(log_return), index=ordered.index, dtype=float)
        .rolling(5, min_periods=5)
        .mean()
        .abs()
    )

    ema_20 = close.ewm(span=20, adjust=False, min_periods=15).mean()
    extension = _safe_ratio(close - ema_20, range_20)
    daily_range = adjusted_high_series - adjusted_low_series
    upper_wick = adjusted_high_series - pd.Series(
        np.maximum(adjusted_open_series, close),
        index=ordered.index,
        dtype=float,
    )
    lower_wick = (
        pd.Series(
            np.minimum(adjusted_open_series, close),
            index=ordered.index,
            dtype=float,
        )
        - adjusted_low_series
    )
    extreme_wick = _safe_ratio(
        pd.Series(
            np.maximum(upper_wick, lower_wick),
            index=ordered.index,
            dtype=float,
        ),
        daily_range,
    )
    rolling_high = adjusted_high_series.rolling(5, min_periods=5).max()
    rolling_low = adjusted_low_series.rolling(5, min_periods=5).min()

    ordered["daily_range_5_to_20"] = _safe_ratio(range_5, range_20)
    ordered["daily_rv_5_to_20"] = _safe_ratio(rv_5, rv_20)
    ordered["daily_range_overlap_5"] = overlap_fraction.rolling(4, min_periods=4).mean()
    ordered["daily_efficiency_5"] = efficiency_5
    ordered["daily_efficiency_10"] = efficiency_10
    ordered["daily_sign_persistence_5"] = sign_persistence
    ordered["daily_extension_20"] = extension
    ordered["daily_extreme_wick_3"] = extreme_wick.rolling(3, min_periods=3).mean()
    ordered["daily_close_location_5"] = _safe_ratio(close - rolling_low, rolling_high - rolling_low)
    ordered["stock_log_return_5"] = log_return.rolling(5, min_periods=5).sum()
    ordered["daily_activity_5_to_20"] = _safe_ratio(
        ordered["activity"].rolling(5, min_periods=5).mean(),
        ordered["activity"].rolling(20, min_periods=15).mean(),
    )
    ordered["realised_volatility_20d"] = rv_20 * math.sqrt(252.0)
    ordered["valid_trailing_sessions_20"] = (
        true_range.rolling(20, min_periods=1).count().clip(upper=20).astype(int)
    )
    return ordered


def calculate_daily_stock_raw_features(daily_bars: pd.DataFrame) -> pd.DataFrame:
    """Calculate the eleven frozen causal daily stock features."""

    required = {"symbol", "session", "open", "high", "low", "close", "activity"}
    if missing := sorted(required.difference(daily_bars.columns)):
        raise ValueError(f"daily stock bars missing columns: {missing}")
    if daily_bars.duplicated(["symbol", "session"]).any():
        raise ValueError("daily stock bars require one row per stock-session")
    pieces = [
        _stock_raw_for_symbol(group)
        for _symbol, group in daily_bars.groupby("symbol", sort=True, observed=True)
    ]
    if not pieces:
        raise ValueError("daily stock bars are empty")
    output = pd.concat(pieces, ignore_index=True)
    output["daily_relative_return_5"] = np.nan
    for _session, group in output.groupby("session", sort=True, observed=True):
        for index in group.index:
            peers = group.loc[group.index != index, "stock_log_return_5"].dropna()
            own = _finite_number(output.at[index, "stock_log_return_5"])
            if own is None:
                relative = math.nan
            elif peers.empty:
                relative = 0.0
            else:
                relative = own - float(peers.median())
            output.at[index, "daily_relative_return_5"] = relative
    for feature in DAILY_STOCK_RAW_FEATURES:
        output[f"{feature}_missing"] = output[feature].isna().astype(int)
    return output.sort_values(["symbol", "session"], kind="mergesort").reset_index(drop=True)


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nearest_delta(
    frame: pd.DataFrame, *, option_type: str, target: float
) -> Mapping[str, Any] | None:
    candidates: list[tuple[tuple[float, str], Mapping[str, Any]]] = []
    records = cast(list[dict[str, Any]], frame.to_dict(orient="records"))
    for row in records:
        if str(row.get("option_type", "")).casefold() != option_type:
            continue
        delta = _finite_number(row.get("delta"))
        iv = _finite_number(row.get("implied_volatility"))
        if delta is None or iv is None or iv <= 0.0:
            continue
        error = abs(delta - target)
        if error > 0.10:
            continue
        candidates.append(((error, str(row.get("contract_id", ""))), row))
    return None if not candidates else min(candidates, key=lambda value: value[0])[1]


def select_daily_options_surface(
    chain: pd.DataFrame,
    *,
    previous_close: float,
    realised_volatility_20d: float,
) -> dict[str, object]:
    """Select the frozen previous-close pair and calculate nine raw features."""

    if chain.empty:
        return {"pair_available": False, "pair_reason": "missing_exact_chain"}
    working = chain.copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="raise").dt.date
    observed_dates = tuple(sorted(set(working["trade_date"])))
    if len(observed_dates) != 1:
        raise ValueError("daily options chain must contain one exact observation date")
    if observed_dates[0] >= date(2025, 8, 23):
        raise ValueError("daily options observation crossed protected boundary")
    working["expiration_date"] = pd.to_datetime(working["expiration_date"], errors="coerce").dt.date
    working["dte"] = working["expiration_date"].map(
        lambda value: (value - observed_dates[0]).days if isinstance(value, date) else math.nan
    )
    front = select_primary_atm_pair(working, previous_close=previous_close)
    if not front.available:
        return {
            "pair_available": False,
            "pair_reason": front.reason,
            "options_observation_date": observed_dates[0],
            "front_expiration_date": front.expiration_date,
            "front_call_contract_id": front.call_contract_id,
            "front_put_contract_id": front.put_contract_id,
        }
    primary = calculate_primary_option_features(front, previous_close=previous_close)
    if front.expiration_date is None:
        raise AssertionError("available front pair lacks expiration")
    front_chain = working.loc[working["expiration_date"].eq(front.expiration_date)].copy()
    put_25 = _nearest_delta(front_chain, option_type="put", target=-0.25)
    call_25 = _nearest_delta(front_chain, option_type="call", target=0.25)
    if put_25 is None or call_25 is None:
        skew = math.nan
        skew_missing = 1
    else:
        put_iv = _finite_number(put_25.get("implied_volatility"))
        call_iv = _finite_number(call_25.get("implied_volatility"))
        if put_iv is None or call_iv is None:
            raise AssertionError("delta candidates require finite IV")
        skew = put_iv - call_iv
        skew_missing = 0

    back = _select_atm_pair(
        working,
        previous_close=previous_close,
        minimum_dte=46,
        maximum_dte=90,
    )
    if back.available:
        back_primary = calculate_primary_option_features(back, previous_close=previous_close)
        front_term_urgency = float(primary["atm_iv"]) - float(back_primary["atm_iv"])
        back_missing = 0
    else:
        front_term_urgency = math.nan
        back_missing = 1

    open_interest = pd.to_numeric(front_chain["open_interest"], errors="coerce")
    valid_oi = open_interest.notna() & open_interest.ge(0.0)
    valid_front = front_chain.loc[valid_oi].copy()
    valid_front["_valid_open_interest"] = open_interest.loc[valid_oi].to_numpy(float)
    total_oi = float(valid_front["_valid_open_interest"].sum())
    if total_oi <= 0.0:
        concentration = math.nan
        imbalance = math.nan
        concentration_missing = 1
        imbalance_missing = 1
    else:
        near = valid_front["strike"].between(
            previous_close * 0.95, previous_close * 1.05, inclusive="both"
        )
        concentration = float(valid_front.loc[near, "_valid_open_interest"].sum()) / total_oi
        option_type = valid_front["option_type"].astype(str).str.casefold()
        call_oi = float(valid_front.loc[option_type.eq("call"), "_valid_open_interest"].sum())
        put_oi = float(valid_front.loc[option_type.eq("put"), "_valid_open_interest"].sum())
        imbalance = math.log((call_oi + 1.0) / (put_oi + 1.0))
        concentration_missing = 0
        imbalance_missing = 0
    realised = float(realised_volatility_20d)
    if not math.isfinite(realised) or realised < 0.0:
        raise ValueError("realised volatility must be finite and non-negative")
    request_ids = tuple(sorted(set(working.get("request_id", pd.Series(dtype=str)).astype(str))))
    output: dict[str, object] = {
        "pair_available": True,
        "pair_reason": "selected",
        "options_observation_date": observed_dates[0],
        "previous_close_underlying_price": previous_close,
        "front_expiration_date": front.expiration_date,
        "front_strike": front.strike,
        "front_call_contract_id": front.call_contract_id,
        "front_put_contract_id": front.put_contract_id,
        "back_expiration_date": back.expiration_date,
        "back_strike": back.strike,
        "back_call_contract_id": back.call_contract_id,
        "back_put_contract_id": back.put_contract_id,
        "skew_put_contract_id": None if put_25 is None else str(put_25.get("contract_id")),
        "skew_call_contract_id": None if call_25 is None else str(call_25.get("contract_id")),
        "previous_close_chain_request_ids": request_ids,
        "atm_iv": float(primary["atm_iv"]),
        "straddle_mid_pct": float(primary["straddle_mid_pct"]),
        "call_put_iv_gap": float(primary["call_put_iv_gap"]),
        "skew_25d": skew,
        "front_term_urgency": front_term_urgency,
        "combined_relative_spread": float(primary["combined_relative_spread"]),
        "iv_minus_realised_20d": float(primary["atm_iv"]) - realised,
        "near_spot_oi_concentration": concentration,
        "call_put_oi_imbalance": imbalance,
        "skew_missing": skew_missing,
        "back_expiry_missing": back_missing,
        "oi_concentration_missing": concentration_missing,
        "call_put_oi_imbalance_missing": imbalance_missing,
    }
    return output


def assert_safety_flags(values: Mapping[str, object]) -> None:
    """Require the complete frozen research-only boundary."""

    for name, expected in SAFETY_FLAGS.items():
        if (
            name not in values
            or values[name] != expected
            or type(values[name]) is not type(expected)
        ):
            raise ValueError(f"daily-context safety flag differs: {name}")


def fit_mismatch_standardization(
    development: pd.DataFrame,
) -> dict[str, MeanStandardization]:
    """Fit the nine mismatch z-score inputs on 2024 only."""

    sessions = pd.to_datetime(development["session"], errors="raise")
    if development.empty or not bool(sessions.dt.year.eq(2024).all()):
        raise ValueError("mismatch standardization must use 2024 development rows only")
    if missing := sorted(set(MISMATCH_STANDARDIZATION_INPUTS).difference(development.columns)):
        raise ValueError(f"mismatch standardization inputs missing: {missing}")
    result: dict[str, MeanStandardization] = {}
    for column in MISMATCH_STANDARDIZATION_INPUTS:
        values = pd.to_numeric(development[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        if not math.isfinite(mean):
            raise ValueError(f"mismatch input lacks finite development support: {column}")
        if not math.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        result[column] = MeanStandardization(mean=mean, scale=scale)
    return result


def _mean_z(
    frame: pd.DataFrame, column: str, values: Mapping[str, MeanStandardization]
) -> pd.Series:
    parameter = values[column]
    return (pd.to_numeric(frame[column], errors="coerce") - parameter.mean) / parameter.scale


def add_mismatch_features(
    frame: pd.DataFrame, standardization: Mapping[str, MeanStandardization]
) -> pd.DataFrame:
    """Add exactly the six frozen stock/options mismatch features."""

    if set(standardization) != set(MISMATCH_STANDARDIZATION_INPUTS):
        raise ValueError("mismatch standardization surface differs")
    if "BROAD_CONFLICT" not in frame:
        raise ValueError("mismatch frame requires BROAD_CONFLICT")
    output = frame.copy()
    compression = _mean_z(output, "daily_compression", standardization)
    tension = _mean_z(output, "options_implied_tension", standardization)
    volatility = _mean_z(output, "daily_volatility_acceleration", standardization)
    urgency = _mean_z(output, "options_front_urgency", standardization)
    route = _mean_z(output, "prefix_family_entropy", standardization)
    richness = _mean_z(output, "options_premium_richness", standardization)
    transition = _mean_z(output, "transition_probability", standardization)
    pressure = _mean_z(output, "signed_pressure", standardization)
    positioning = _mean_z(output, "options_directional_positioning", standardization)
    output["mismatch_compression_vs_iv"] = compression - tension
    output["mismatch_volatility_vs_urgency"] = volatility - urgency
    output["mismatch_route_vs_premium"] = route - richness
    output["mismatch_transition_vs_urgency"] = transition - urgency
    output["mismatch_direction_agreement"] = pressure * positioning
    output["mismatch_complacent_conflict"] = output["BROAD_CONFLICT"].fillna(0).astype(int) * (
        -tension
    )
    return output


def iv_horizon_outcomes(
    *,
    entry_price: float,
    atm_iv: float,
    close_15m: float | None,
    same_session_close: float | None,
    next_session_close: float | None,
    third_session_close: float | None,
    remaining_regular_session_minutes: int,
) -> dict[str, float | int]:
    """Calculate underlying-only movement and prior-close IV residuals."""

    if (
        not math.isfinite(entry_price)
        or entry_price <= 0.0
        or not math.isfinite(atm_iv)
        or atm_iv <= 0.0
        or remaining_regular_session_minutes <= 0
    ):
        raise ValueError("IV horizon inputs must be finite and positive")
    horizon_inputs = {
        "15m": (close_15m, 15.0 / 390.0),
        "to_close": (
            same_session_close,
            float(remaining_regular_session_minutes) / 390.0,
        ),
        "next_close": (
            next_session_close,
            1.0 + float(remaining_regular_session_minutes) / 390.0,
        ),
        "third_close": (
            third_session_close,
            3.0 + float(remaining_regular_session_minutes) / 390.0,
        ),
    }
    output: dict[str, float | int] = {
        "entry_price": entry_price,
        "remaining_regular_session_minutes": remaining_regular_session_minutes,
    }
    for name, (close, horizon_days) in horizon_inputs.items():
        sigma = atm_iv * math.sqrt(horizon_days / 252.0)
        expected = sigma * math.sqrt(2.0 / math.pi)
        output[f"iv_sigma_{name}"] = sigma
        output[f"iv_expected_absolute_{name}"] = expected
        if close is None or not math.isfinite(close) or close <= 0.0:
            movement = math.nan
            residual = math.nan
        else:
            movement = abs(math.log(close / entry_price))
            residual = movement - expected
        output[f"absolute_log_return_{name}"] = movement
        output[f"iv_absolute_residual_{name}"] = residual
    residual_15m = float(output["iv_absolute_residual_15m"])
    output["movement_exceeds_prior_close_iv_15m"] = (
        0 if math.isnan(residual_15m) else int(residual_15m > 0.0)
    )
    return output


def permute_bundle_within_slates(
    frame: pd.DataFrame, *, columns: Sequence[str], seed: int
) -> pd.DataFrame:
    """Permute one complete bundle among stocks within each causal slate."""

    strata = ("period", "session", "checkpoint")
    required = {*strata, "symbol", *columns}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"bundle permutation inputs missing: {missing}")
    if frame.duplicated([*strata, "symbol"]).any():
        raise ValueError("bundle permutation requires one row per stock and slate")
    output = frame.copy()
    rng = np.random.default_rng(seed)
    for _key, slate in frame.groupby(list(strata), sort=True, observed=True):
        target_indices = slate.sort_values("symbol", kind="mergesort").index.to_numpy()
        source_indices = rng.permutation(target_indices)
        output.loc[target_indices, list(columns)] = frame.loc[
            source_indices, list(columns)
        ].to_numpy()
    return output


def choose_daily_context_decision(
    *,
    blocker: str | None,
    test_a_daily_stock_supported: bool,
    test_a_daily_options_supported: bool,
    test_b_daily_stock_supported: bool,
    test_b_intraday_route_supported: bool,
    mismatch_supported: bool,
    descriptive: bool,
) -> str:
    """Choose exactly one frozen overall decision."""

    if blocker is not None:
        if blocker not in OVERALL_DECISIONS or not blocker.startswith("blocked_"):
            raise ValueError(f"unknown daily-context blocker: {blocker}")
        return blocker
    test_a = test_a_daily_stock_supported and test_a_daily_options_supported
    test_b = test_b_daily_stock_supported and test_b_intraday_route_supported
    if test_a and test_b:
        return "daily_stock_and_options_context_supported_bidirectionally"
    if test_a_daily_options_supported and not test_b:
        return "daily_options_improve_stock_completion_only"
    if test_b_daily_stock_supported and not test_a:
        return "daily_stock_context_improves_iv_excess_only"
    if mismatch_supported:
        return "cross_market_mismatch_supported_only"
    if descriptive:
        return "daily_context_descriptive_only"
    return "no_daily_context_increment"


__all__ = [
    "ASSESSMENT_END",
    "ASSESSMENT_START",
    "DAILY_OPTIONS_MISSING_INDICATORS",
    "DAILY_OPTIONS_RAW_FEATURES",
    "DAILY_STOCK_RAW_FEATURES",
    "DEVELOPMENT_END",
    "DEVELOPMENT_START",
    "FROZEN_COHORT",
    "MISMATCH_FEATURES",
    "MISMATCH_STANDARDIZATION_INPUTS",
    "MeanStandardization",
    "OVERALL_DECISIONS",
    "PROTECTED_START",
    "SAFETY_FLAGS",
    "add_mismatch_features",
    "assert_safety_flags",
    "choose_daily_context_decision",
    "calculate_daily_stock_raw_features",
    "fit_mismatch_standardization",
    "iv_horizon_outcomes",
    "permute_bundle_within_slates",
    "previous_us_trading_session",
    "reject_protected_observations",
    "select_daily_options_surface",
    "validate_daily_context_chronology",
]
