"""Frozen front-options context, mismatch, support, and decision utilities for V0.1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from stocker_research.daily_stock_options_context_v0 import (
    ASSESSMENT_END,
    DEVELOPMENT_START,
    FROZEN_COHORT,
    PROTECTED_START,
)
from stocker_research.front_options_soft_regimes_v01 import (
    FRONT_OPTIONS_MISSING_INDICATORS,
    FRONT_OPTIONS_RAW_FEATURES,
)

SAFETY_FLAGS: Final[dict[str, object]] = {
    "research_only": True,
    "quick_context_screen": True,
    "branches_run_independently": True,
    "daily_stock_context_test": True,
    "front_options_only_context_test": True,
    "back_expiry_bulk_download_enabled": False,
    "back_expiry_schema_preflight_only": True,
    "previous_close_options_only": True,
    "intraday_option_quotes_used": False,
    "option_pnl_calculated": False,
    "underlying_movement_outcomes_opened": True,
    "directional_outcomes_primary": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "prospective_validation": False,
}
FRONT_MISMATCH_FEATURES: Final[tuple[str, ...]] = (
    "mismatch_compression_vs_front_iv",
    "mismatch_daily_volatility_vs_front_iv",
    "mismatch_route_vs_front_premium",
    "mismatch_direction_agreement",
    "mismatch_complacent_broad_conflict",
)
MISMATCH_BASE_COLUMNS: Final[tuple[str, ...]] = (
    "daily_compression",
    "daily_volatility_acceleration",
    "front_options_implied_tension",
    "prefix_family_entropy",
    "front_options_premium_richness",
    "signed_pressure",
    "front_options_directional_positioning",
)
FRONT_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "session",
    "period",
    "required_options_date",
    "options_observation_date",
    "previous_close_underlying_price",
    "front_expiration_date",
    "front_strike",
    "front_call_contract_id",
    "front_put_contract_id",
    "skew_put_contract_id",
    "skew_call_contract_id",
    "previous_close_chain_request_ids",
)
ANNUAL_TRADING_MINUTES: Final[int] = 252 * 390


@dataclass(frozen=True)
class MeanScale:
    """One development-only mean and population scale."""

    mean: float
    scale: float


def assert_safety_flags(value: Mapping[str, object]) -> None:
    """Require every V0.1 research-only flag to be exact."""

    mismatches = {
        key: (expected, value.get(key))
        for key, expected in SAFETY_FLAGS.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"front-context safety flags differ: {mismatches}")


def prepare_front_options_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Project the predecessor surface to exact-date front-only rows."""

    required = {
        "pair_available",
        *FRONT_IDENTITY_COLUMNS,
        *FRONT_OPTIONS_RAW_FEATURES,
        "skew_missing",
        "oi_concentration_missing",
        "call_put_oi_imbalance_missing",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"predecessor front-options fields missing: {missing}")
    available = frame.loc[frame["pair_available"].astype(bool)].copy()
    if available.empty:
        raise ValueError("no valid predecessor front-options pairs")
    observation = pd.to_datetime(available["options_observation_date"], errors="raise")
    required_date = pd.to_datetime(available["required_options_date"], errors="raise")
    signal = pd.to_datetime(available["session"], errors="raise")
    if not observation.lt(signal).all():
        raise ValueError("same-day or future front-options observation detected")
    if not observation.eq(required_date).all():
        raise ValueError("front-options observation is not the exact required D-1 date")
    if observation.dt.date.ge(PROTECTED_START).any():
        raise ValueError("protected front-options observation materialised")
    available["skew_25d_missing"] = available["skew_missing"].astype(int)
    available["near_spot_oi_concentration_missing"] = available["oi_concentration_missing"].astype(
        int
    )
    available["call_put_oi_imbalance_missing"] = available["call_put_oi_imbalance_missing"].astype(
        int
    )
    columns = [
        *FRONT_IDENTITY_COLUMNS,
        *FRONT_OPTIONS_RAW_FEATURES,
        *FRONT_OPTIONS_MISSING_INDICATORS,
    ]
    output = available.loc[:, columns].copy()
    forbidden = {"front_term_urgency", "back_atm_iv", "term_structure"}
    if forbidden.intersection(output.columns):
        raise AssertionError("front-only projection retained a back-expiry field")
    return output.sort_values(["symbol", "session"], kind="mergesort").reset_index(drop=True)


def fit_front_mismatch_standardization(
    development: pd.DataFrame,
) -> dict[str, MeanScale]:
    """Fit all mismatch z-scores on development rows only."""

    if development.empty:
        raise ValueError("mismatch development frame is empty")
    sessions = pd.to_datetime(development["session"], errors="raise")
    if not sessions.dt.year.eq(2024).all():
        raise ValueError("mismatch standardization must be fitted on 2024 only")
    if missing := sorted(set(MISMATCH_BASE_COLUMNS).difference(development.columns)):
        raise ValueError(f"mismatch bases missing: {missing}")
    result: dict[str, MeanScale] = {}
    for column in MISMATCH_BASE_COLUMNS:
        values = pd.to_numeric(development[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        if not math.isfinite(mean):
            raise ValueError(f"mismatch base lacks finite support: {column}")
        if not math.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        result[column] = MeanScale(mean=mean, scale=scale)
    return result


def add_front_mismatch_features(
    frame: pd.DataFrame,
    standardization: Mapping[str, MeanScale],
) -> pd.DataFrame:
    """Add exactly the five preregistered front-only mismatch features."""

    if missing := sorted(set(MISMATCH_BASE_COLUMNS).difference(frame.columns)):
        raise ValueError(f"mismatch bases missing: {missing}")
    if missing := sorted(set(MISMATCH_BASE_COLUMNS).difference(standardization)):
        raise ValueError(f"mismatch standardizations missing: {missing}")
    if "BROAD_CONFLICT" not in frame:
        raise ValueError("BROAD_CONFLICT indicator is required")
    output = frame.copy()

    def z(column: str) -> pd.Series:
        fitted = standardization[column]
        return (pd.to_numeric(output[column], errors="coerce") - fitted.mean) / fitted.scale

    compression = z("daily_compression")
    volatility = z("daily_volatility_acceleration")
    tension = z("front_options_implied_tension")
    route_entropy = z("prefix_family_entropy")
    premium = z("front_options_premium_richness")
    pressure = z("signed_pressure")
    positioning = z("front_options_directional_positioning")
    output["mismatch_compression_vs_front_iv"] = compression - tension
    output["mismatch_daily_volatility_vs_front_iv"] = volatility - tension
    output["mismatch_route_vs_front_premium"] = route_entropy - premium
    output["mismatch_direction_agreement"] = pressure * positioning
    output["mismatch_complacent_broad_conflict"] = (
        pd.to_numeric(output["BROAD_CONFLICT"], errors="raise") * -tension
    )
    return output


def iv_excess_15m(
    *,
    entry_price: float,
    close_15m: float,
    atm_iv: float,
) -> dict[str, float | int]:
    """Calculate the exact 15-minute IV-relative underlying movement outcome."""

    row = iv_excess_15m_frame(
        entry_price=[entry_price],
        close_15m=[close_15m],
        atm_iv=[atm_iv],
    ).iloc[0]
    return {
        "entry_price": float(row["entry_price"]),
        "close_15m": float(row["close_15m"]),
        "absolute_log_return_15m": float(row["absolute_log_return_15m"]),
        "iv_sigma_15m": float(row["iv_sigma_15m"]),
        "iv_expected_absolute_15m": float(row["iv_expected_absolute_15m"]),
        "movement_exceeds_prior_close_iv_15m": int(row["movement_exceeds_prior_close_iv_15m"]),
        "iv_absolute_residual_15m": float(row["iv_absolute_residual_15m"]),
    }


def iv_excess_15m_frame(
    *,
    entry_price: Sequence[float],
    close_15m: Sequence[float],
    atm_iv: Sequence[float],
) -> pd.DataFrame:
    """Vectorize the exact 15-minute IV-relative movement calculation."""

    entry = np.asarray(entry_price, dtype=float)
    close = np.asarray(close_15m, dtype=float)
    iv = np.asarray(atm_iv, dtype=float)
    if not (
        entry.ndim == close.ndim == iv.ndim == 1
        and len(entry) == len(close) == len(iv)
        and np.isfinite(entry).all()
        and np.isfinite(close).all()
        and np.isfinite(iv).all()
        and bool((entry > 0.0).all())
        and bool((close > 0.0).all())
        and bool((iv > 0.0).all())
    ):
        raise ValueError("15-minute IV outcome inputs must be finite and positive")
    movement = np.abs(np.log(close / entry))
    sigma = iv * math.sqrt(15.0 / ANNUAL_TRADING_MINUTES)
    expected = sigma * math.sqrt(2.0 / math.pi)
    return pd.DataFrame(
        {
            "entry_price": entry,
            "close_15m": close,
            "absolute_log_return_15m": movement,
            "iv_sigma_15m": sigma,
            "iv_expected_absolute_15m": expected,
            "movement_exceeds_prior_close_iv_15m": (movement > expected).astype(int),
            "iv_absolute_residual_15m": movement - expected,
        }
    )


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    """Return a deterministic weighted quantile."""

    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if not (
        0.0 <= quantile <= 1.0
        and len(value_array) > 0
        and len(value_array) == len(weight_array)
        and np.isfinite(value_array).all()
        and np.isfinite(weight_array).all()
        and bool((weight_array > 0.0).all())
    ):
        raise ValueError("weighted quantile inputs are invalid")
    order = np.argsort(value_array, kind="mergesort")
    ordered_values = value_array[order]
    ordered_weights = weight_array[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= ordered_weights.sum()
    return float(np.interp(quantile, cumulative, ordered_values))


def route_state_iv_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise weighted IV-relative movement by the four frozen report groups."""

    required = {
        "route_resolution_state",
        "row_weight",
        "absolute_log_return_15m",
        "iv_expected_absolute_15m",
        "iv_absolute_residual_15m",
        "movement_exceeds_prior_close_iv_15m",
        "iv_sigma_15m",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"route-state IV metrics missing: {missing}")
    working = frame.copy()
    working["_route_group"] = (
        working["route_resolution_state"]
        .astype(str)
        .where(
            working["route_resolution_state"]
            .astype(str)
            .isin({"BROAD_CONFLICT", "LOW_ROUTE_SUPPORT", "NARROWING"}),
            "OTHER",
        )
    )
    rows: list[dict[str, float | int | str]] = []
    for route_group in ("BROAD_CONFLICT", "LOW_ROUTE_SUPPORT", "NARROWING", "OTHER"):
        group = working.loc[working["_route_group"].eq(route_group)]
        weights = group["row_weight"].to_numpy(float)
        residual = group["iv_absolute_residual_15m"].to_numpy(float)
        movement = group["absolute_log_return_15m"].to_numpy(float)
        expected = group["iv_expected_absolute_15m"].to_numpy(float)
        sigma = group["iv_sigma_15m"].to_numpy(float)
        exceeds = group["movement_exceeds_prior_close_iv_15m"].to_numpy(float)
        total = float(weights.sum())
        positive = np.maximum(residual, 0.0)
        top_count = max(1, math.ceil(len(group) * 0.05))
        top_index = np.argsort(residual, kind="mergesort")[-top_count:]
        positive_total = float(np.sum(weights * positive))
        rows.append(
            {
                "route_state": route_group,
                "rows": len(group),
                "mean_absolute_movement": float(np.sum(weights * movement) / total),
                "median_absolute_movement": weighted_quantile(
                    movement.tolist(), weights.tolist(), 0.5
                ),
                "mean_iv_expectation": float(np.sum(weights * expected) / total),
                "mean_iv_residual": float(np.sum(weights * residual) / total),
                "median_iv_residual": weighted_quantile(residual.tolist(), weights.tolist(), 0.5),
                "exceed_iv_rate": float(np.sum(weights * exceeds) / total),
                "iv_sigma_ratio": float(np.sum(weights * movement) / np.sum(weights * sigma)),
                "upper_decile_iv_residual": weighted_quantile(
                    residual.tolist(), weights.tolist(), 0.9
                ),
                "top_5pct_positive_residual_contribution": (
                    float(np.sum(weights[top_index] * positive[top_index]) / positive_total)
                    if positive_total > 0.0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def choose_overall_decision(
    *,
    daily_stock_context_status: str,
    front_options_completion_status: str,
    stock_to_iv_excess_status: str,
) -> str:
    """Map independent branch statuses to one preregistered overall decision."""

    statuses = (
        daily_stock_context_status,
        front_options_completion_status,
        stock_to_iv_excess_status,
    )
    supported = tuple(status == "supported" for status in statuses)
    if supported == (True, True, True):
        return "daily_stock_and_front_options_context_supported"
    if supported == (True, False, False):
        return "daily_stock_context_supported_only"
    if supported == (False, True, False):
        return "front_options_completion_context_supported_only"
    if supported == (False, False, True):
        return "stock_structure_improves_iv_excess_only"
    if sum(supported) >= 2:
        return "multiple_partial_context_increments_supported"
    if all(status == "insufficient_support" for status in statuses):
        return "all_model_branches_insufficient_support"
    if daily_stock_context_status == "descriptive_only":
        return "daily_stock_context_descriptive_only"
    if front_options_completion_status == "descriptive_only":
        return "front_options_context_descriptive_only"
    return "no_context_increment"


def branch_availability(
    *,
    structural_ready: bool,
    daily_stock_ready: bool,
    front_options_ready: bool,
) -> dict[str, bool]:
    """Return the independent data prerequisites for the three model branches."""

    return {
        "branch_a": structural_ready and daily_stock_ready,
        "branch_b": structural_ready and daily_stock_ready and front_options_ready,
        "branch_c": structural_ready and daily_stock_ready and front_options_ready,
    }


__all__ = [
    "ANNUAL_TRADING_MINUTES",
    "ASSESSMENT_END",
    "DEVELOPMENT_START",
    "FRONT_IDENTITY_COLUMNS",
    "FRONT_MISMATCH_FEATURES",
    "FROZEN_COHORT",
    "MISMATCH_BASE_COLUMNS",
    "MeanScale",
    "PROTECTED_START",
    "SAFETY_FLAGS",
    "add_front_mismatch_features",
    "assert_safety_flags",
    "branch_availability",
    "choose_overall_decision",
    "fit_front_mismatch_standardization",
    "iv_excess_15m",
    "iv_excess_15m_frame",
    "prepare_front_options_raw",
    "route_state_iv_metrics",
    "weighted_quantile",
]
