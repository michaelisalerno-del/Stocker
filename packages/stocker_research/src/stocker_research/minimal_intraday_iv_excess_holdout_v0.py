"""Frozen minimal intraday-H0 to IV-excess holdout validation helpers.

This module contains only retrospective research mechanics.  It does not model
option P&L, use intraday option quotes, or expose any execution surface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from stocker_research.broad_conflict_options_iv_screen_v0 import (
    calculate_primary_option_features,
    select_primary_atm_pair,
    validate_exact_previous_session_join,
)
from stocker_research.stock_layer_iv_excess_attribution_v0 import (
    CHECKPOINTS,
    GROUP_D,
    GROUP_I,
    GROUP_M,
    GROUP_O,
    GROUP_R,
)
from stocker_research.stock_options_cross_market_quick_v0 import (
    FrozenCrossMarketModel,
    binary_metrics,
    fit_cross_market_model,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

TRAINING_START: Final[pd.Timestamp] = pd.Timestamp("2024-01-01")
TRAINING_END: Final[pd.Timestamp] = pd.Timestamp("2024-12-31")
PRIOR_REFERENCE_START: Final[pd.Timestamp] = pd.Timestamp("2025-01-01")
PRIOR_REFERENCE_END: Final[pd.Timestamp] = pd.Timestamp("2025-08-22")
HOLDOUT_START: Final[pd.Timestamp] = pd.Timestamp("2025-09-01")
HOLDOUT_END: Final[pd.Timestamp] = pd.Timestamp("2025-12-31")
PROTECTED_START: Final[pd.Timestamp] = pd.Timestamp("2026-01-01")
TARGET_COLUMN: Final[str] = "movement_exceeds_prior_close_iv_15m"
ANNUAL_TRADING_MINUTES: Final[int] = 252 * 390
HORIZONS: Final[tuple[int, ...]] = (5, 10, 15, 30)
BOOTSTRAP_SEED: Final[int] = 20260764
NULL_SEEDS: Final[tuple[int, int, int]] = (20260765, 20260766, 20260767)

SAFETY_FLAGS: Final[dict[str, object]] = {
    "research_only": True,
    "frozen_holdout_validation": True,
    "holdout_start": "2025-09-01",
    "holdout_end": "2025-12-31",
    "training_end": "2024-12-31",
    "prior_reference_period_not_used_for_tuning": True,
    "previous_close_options_only": True,
    "minimal_options_plus_intraday_h0_model": True,
    "daily_stock_features_excluded": True,
    "route_competition_features_excluded": True,
    "route_state_features_excluded": True,
    "hand_built_mismatch_features_excluded": True,
    "top_5_percent_tail_frozen": True,
    "option_pnl_calculated": False,
    "intraday_option_quotes_used": False,
    "directional_outcomes_primary": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}

EXCLUDED_FEATURES: Final[tuple[str, ...]] = (
    *GROUP_D,
    *GROUP_R,
    *GROUP_M,
    "route_resolution_state",
)


@dataclass(frozen=True)
class ModelGateInputs:
    """Frozen model-transfer gate inputs."""

    log_loss_improvement: float
    brier_improvement: float
    auc_improvement: float
    average_precision_improvement: float
    bootstrap_80_log_loss_lower: float
    bootstrap_80_brier_lower: float
    bootstrap_80_average_precision_lower: float
    positive_log_loss_months: int
    materially_adverse_checkpoint_groups: int
    real_exceeds_all_nulls: bool
    support_passed: bool

    @property
    def passed(self) -> bool:
        """Return whether all ten preregistered model conditions pass."""

        return bool(
            self.log_loss_improvement > 0.0
            and self.brier_improvement > 0.0
            and self.auc_improvement >= 0.0
            and self.average_precision_improvement > 0.0
            and self.bootstrap_80_log_loss_lower >= 0.0
            and self.bootstrap_80_brier_lower >= 0.0
            and self.bootstrap_80_average_precision_lower >= 0.0
            and self.positive_log_loss_months >= 3
            and self.materially_adverse_checkpoint_groups == 0
            and self.real_exceeds_all_nulls
            and self.support_passed
        )


@dataclass(frozen=True)
class TailGateInputs:
    """Frozen top-five-percent tail gate inputs."""

    mean_iv_residual: float
    median_iv_residual: float
    exceed_iv_rate: float
    bootstrap_80_mean_lower: float
    bootstrap_80_median_lower: float
    positive_mean_months: int
    positive_median_months: int
    m1_minus_m0_mean: float
    bootstrap_80_difference_lower: float
    concentration_passed: bool
    support_passed: bool

    @property
    def passed(self) -> bool:
        """Return whether every preregistered binding tail condition passes."""

        return bool(
            self.mean_iv_residual > 0.0
            and self.median_iv_residual > 0.0
            and self.exceed_iv_rate > 0.50
            and self.bootstrap_80_mean_lower >= 0.0
            and self.bootstrap_80_median_lower >= 0.0
            and self.positive_mean_months >= 3
            and self.positive_median_months >= 3
            and self.m1_minus_m0_mean > 0.0
            and self.bootstrap_80_difference_lower >= 0.0
            and self.concentration_passed
            and self.support_passed
        )


@dataclass(frozen=True)
class MinimalModels:
    """The exactly two frozen primary model fits."""

    m0: FrozenCrossMarketModel
    m1: FrozenCrossMarketModel


def assert_safety_flags(value: Mapping[str, object]) -> None:
    """Fail closed if any binding research or execution flag differs."""

    mismatches = {
        key: {"expected": expected, "actual": value.get(key)}
        for key, expected in SAFETY_FLAGS.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"minimal holdout safety flags differ: {mismatches}")


def _normalized_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise", utc=True)
    return parsed.dt.tz_convert(None).dt.normalize()


def validate_holdout_dates(values: pd.Series) -> None:
    """Require every materialised observation to lie in the authorized holdout."""

    dates = _normalized_dates(values)
    if bool(dates.ge(PROTECTED_START).any()):
        raise ValueError("protected observation dated 2026-01-01 or later")
    if bool((dates.lt(HOLDOUT_START) | dates.gt(HOLDOUT_END)).any()):
        raise ValueError("holdout observation lies outside 2025-09-01 through 2025-12-31")


def assert_2024_only_fitting(values: pd.Series) -> None:
    """Require preprocessing and coefficient fitting dates to be 2024 only."""

    dates = _normalized_dates(values)
    if dates.empty or bool((dates.lt(TRAINING_START) | dates.gt(TRAINING_END)).any()):
        raise ValueError("model preprocessing and fitting are frozen to 2024")


def validate_exact_previous_session_options(
    *,
    signal_date: date,
    required_options_date: date,
    actual_options_date: date,
) -> None:
    """Reject same-day, future, stale, and otherwise non-exact option observations."""

    if actual_options_date == signal_date:
        raise ValueError("same-day option observation rejected")
    if actual_options_date > signal_date:
        raise ValueError("future option observation rejected")
    if actual_options_date < required_options_date:
        raise ValueError("stale option observation rejected; forward filling is forbidden")
    if actual_options_date > required_options_date:
        raise ValueError("non-exact previous-session option observation rejected")
    validate_exact_previous_session_join(
        signal_date=signal_date,
        required_options_date=required_options_date,
        actual_options_date=actual_options_date,
    )


def exact_date_option_records(
    frame: pd.DataFrame,
    *,
    required_date: date,
    date_column: str = "trade_date",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter provider output to the requested exact date before materialisation."""

    if date_column not in frame:
        raise ValueError(f"option response lacks authoritative {date_column}")
    observed = pd.to_datetime(frame[date_column], errors="raise", utc=True)
    normalized = observed.dt.tz_convert(None).dt.normalize()
    required = pd.Timestamp(required_date)
    protected = normalized.ge(PROTECTED_START)
    exact = normalized.eq(required)
    retained = frame.loc[exact & ~protected].copy().reset_index(drop=True)
    if bool(
        pd.to_datetime(retained[date_column], errors="raise", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
        .ge(PROTECTED_START)
        .any()
    ):
        raise AssertionError("protected option observation survived exact-date filtering")
    return retained, {
        "records_returned": int(len(frame)),
        "exact_date_records_retained": int(len(retained)),
        "extra_date_records_rejected": int((~exact).sum()),
        "protected_date_records_rejected": int(protected.sum()),
    }


def build_group_o(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact frozen options context plus checkpoint surface."""

    output = frame.copy()
    if "checkpoint" in output:
        checkpoints = pd.to_numeric(output["checkpoint"], errors="raise").astype(int)
        for checkpoint in CHECKPOINTS:
            column = f"checkpoint_{checkpoint}"
            expected = checkpoints.eq(checkpoint).astype(float)
            if column in output:
                actual = pd.to_numeric(output[column], errors="raise").astype(float)
                if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
                    raise ValueError(f"checkpoint indicator drifted: {column}")
            else:
                output[column] = expected
    missing = sorted(set(GROUP_O).difference(output.columns))
    if missing:
        raise ValueError(f"Group O construction missing frozen features: {missing}")
    return output.loc[:, list(GROUP_O)].copy()


def build_group_i(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact frozen intraday-H0 feature bundle."""

    missing = sorted(set(GROUP_I).difference(frame.columns))
    if missing:
        raise ValueError(f"Group I construction missing frozen features: {missing}")
    values = frame.loc[:, list(GROUP_I)].copy()
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("Group I must be fully reconstructed and finite")
    return values


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nearest_delta(
    frame: pd.DataFrame,
    *,
    option_type: str,
    target: float,
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
        distance = abs(delta - target)
        if distance <= 0.10:
            candidates.append(((distance, str(row.get("contract_id", ""))), row))
    return None if not candidates else min(candidates, key=lambda item: item[0])[1]


def select_minimal_front_options_surface(
    chain: pd.DataFrame,
    *,
    previous_close: float,
    realised_volatility_20d: float,
) -> dict[str, object]:
    """Apply the frozen front-only pair and raw-feature rule without back expiry."""

    if chain.empty:
        return {"pair_available": False, "pair_reason": "missing_exact_chain"}
    working = chain.copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="raise").dt.date
    observed_dates = tuple(sorted(set(working["trade_date"])))
    if len(observed_dates) != 1:
        raise ValueError("front-options chain must contain one exact observation date")
    observed = observed_dates[0]
    if observed >= PROTECTED_START.date():
        raise ValueError("front-options observation crossed the protected boundary")
    working["expiration_date"] = pd.to_datetime(working["expiration_date"], errors="coerce").dt.date
    working["dte"] = working["expiration_date"].map(
        lambda value: (value - observed).days if isinstance(value, date) else math.nan
    )
    front = select_primary_atm_pair(working, previous_close=previous_close)
    if not front.available:
        return {
            "pair_available": False,
            "pair_reason": front.reason,
            "options_observation_date": observed,
            "front_expiration_date": front.expiration_date,
            "front_strike": front.strike,
            "front_call_contract_id": front.call_contract_id,
            "front_put_contract_id": front.put_contract_id,
        }
    primary = calculate_primary_option_features(front, previous_close=previous_close)
    if front.expiration_date is None:
        raise AssertionError("available frozen front pair lacks expiration")
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
            raise AssertionError("frozen skew contracts require finite IV")
        skew = put_iv - call_iv
        skew_missing = 0
    open_interest = pd.to_numeric(front_chain["open_interest"], errors="coerce")
    valid = open_interest.notna() & open_interest.ge(0.0)
    valid_front = front_chain.loc[valid].copy()
    valid_front["_valid_open_interest"] = open_interest.loc[valid].to_numpy(float)
    total_oi = float(valid_front["_valid_open_interest"].sum())
    if total_oi <= 0.0:
        concentration = math.nan
        imbalance = math.nan
        concentration_missing = 1
        imbalance_missing = 1
    else:
        near = valid_front["strike"].between(
            previous_close * 0.95,
            previous_close * 1.05,
            inclusive="both",
        )
        concentration = float(valid_front.loc[near, "_valid_open_interest"].sum()) / total_oi
        option_type = valid_front["option_type"].astype(str).str.casefold()
        call_oi = float(valid_front.loc[option_type.eq("call"), "_valid_open_interest"].sum())
        put_oi = float(valid_front.loc[option_type.eq("put"), "_valid_open_interest"].sum())
        imbalance = math.log((call_oi + 1.0) / (put_oi + 1.0))
        concentration_missing = 0
        imbalance_missing = 0
    if not math.isfinite(realised_volatility_20d) or realised_volatility_20d < 0.0:
        raise ValueError("realised volatility must be finite and non-negative")
    return {
        "pair_available": True,
        "pair_reason": "selected",
        "options_observation_date": observed,
        "previous_close_underlying_price": previous_close,
        "front_expiration_date": front.expiration_date,
        "front_strike": front.strike,
        "front_call_contract_id": front.call_contract_id,
        "front_put_contract_id": front.put_contract_id,
        "skew_put_contract_id": None if put_25 is None else str(put_25.get("contract_id")),
        "skew_call_contract_id": None if call_25 is None else str(call_25.get("contract_id")),
        "atm_iv": float(primary["atm_iv"]),
        "straddle_mid_pct": float(primary["straddle_mid_pct"]),
        "call_put_iv_gap": float(primary["call_put_iv_gap"]),
        "skew_25d": skew,
        "combined_relative_spread": float(primary["combined_relative_spread"]),
        "iv_minus_realised_20d": float(primary["atm_iv"]) - realised_volatility_20d,
        "near_spot_oi_concentration": concentration,
        "call_put_oi_imbalance": imbalance,
        "skew_25d_missing": skew_missing,
        "near_spot_oi_concentration_missing": concentration_missing,
        "call_put_oi_imbalance_missing": imbalance_missing,
    }


def validate_no_excluded_features(features: Sequence[str]) -> None:
    """Fail if a daily, route, route-state, or mismatch feature enters a model."""

    observed = set(features)
    excluded = sorted(observed.intersection(EXCLUDED_FEATURES))
    if excluded:
        raise ValueError(f"excluded feature entered minimal model: {excluded}")
    allowed = set(GROUP_O).union(GROUP_I)
    unknown = sorted(observed.difference(allowed))
    if unknown:
        raise ValueError(f"non-frozen feature entered minimal model: {unknown}")


def minimal_feature_manifest() -> dict[str, object]:
    """Serialize the exact two-model feature surface and explicit exclusions."""

    validate_no_excluded_features([*GROUP_O, *GROUP_I])
    return {
        **SAFETY_FLAGS,
        "feature_source": "Stock-Layer Attribution and IV-Excess Tail Quick Screen V0",
        "group_O": {
            "description": "previous-close front-options context and frozen controls",
            "numeric_features": list(GROUP_O),
            "categorical_controls": ["stock"],
        },
        "group_I": {
            "description": "frozen intraday H0 stock condition",
            "numeric_features": list(GROUP_I),
        },
        "models": {
            "M0": {
                "numeric_features": list(GROUP_O),
                "categorical_controls": ["stock"],
            },
            "M1": {
                "numeric_features": [*GROUP_O, *GROUP_I],
                "categorical_controls": ["stock"],
            },
        },
        "excluded_groups": {
            "daily_stock": list(GROUP_D),
            "route_competition": list(GROUP_R),
            "route_state": ["route_resolution_state"],
            "hand_built_mismatch": list(GROUP_M),
        },
    }


def weighted_quantile(
    values: Sequence[float] | FloatArray,
    weights: Sequence[float] | FloatArray,
    quantile: float,
) -> float:
    """Return a deterministic midpoint-CDF weighted quantile."""

    data = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    if (
        data.ndim != 1
        or mass.ndim != 1
        or len(data) == 0
        or len(data) != len(mass)
        or not np.isfinite(data).all()
        or not np.isfinite(mass).all()
        or bool((mass <= 0.0).any())
        or not 0.0 <= quantile <= 1.0
    ):
        raise ValueError("weighted quantile requires finite values and positive aligned weights")
    order = np.argsort(data, kind="mergesort")
    sorted_values = data[order]
    sorted_weights = mass[order]
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return float(
        np.interp(
            quantile,
            positions,
            sorted_values,
            left=sorted_values[0],
            right=sorted_values[-1],
        )
    )


def freeze_tail_thresholds(
    *,
    m0_probabilities: Sequence[float] | FloatArray,
    m1_probabilities: Sequence[float] | FloatArray,
    weights: Sequence[float] | FloatArray,
) -> dict[str, float | str]:
    """Freeze both weighted 2024 top-five-percent probability thresholds."""

    return {
        "quantile": 0.95,
        "method": "deterministic_midpoint_cdf_weighted_quantile",
        "fitted_period": "2024-01-01_through_2024-12-31",
        "M0_top_5_percent_threshold": weighted_quantile(m0_probabilities, weights, 0.95),
        "M1_top_5_percent_threshold": weighted_quantile(m1_probabilities, weights, 0.95),
    }


def frozen_tail_membership(
    probabilities: Sequence[float] | FloatArray,
    threshold: float,
) -> BoolArray:
    """Apply an already-frozen numeric threshold without rank forcing."""

    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all() or not math.isfinite(threshold):
        raise ValueError("tail membership requires finite probabilities and threshold")
    return np.asarray(values >= threshold, dtype=np.bool_)


def add_movement_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct the frozen 5/10/15/30-minute movement and IV diagnostics."""

    required = {"entry_price", "atm_iv", *(f"close_{horizon}m" for horizon in HORIZONS)}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"movement construction inputs missing: {missing}")
    output = frame.copy()
    entry = pd.to_numeric(output["entry_price"], errors="raise").to_numpy(float)
    atm_iv = pd.to_numeric(output["atm_iv"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(entry).all()
        or bool((entry <= 0.0).any())
        or not np.isfinite(atm_iv).all()
        or bool((atm_iv <= 0.0).any())
    ):
        raise ValueError("entry prices and prior-close ATM IV must be finite and positive")
    for horizon in HORIZONS:
        close = pd.to_numeric(output[f"close_{horizon}m"], errors="raise").to_numpy(float)
        if not np.isfinite(close).all() or bool((close <= 0.0).any()):
            raise ValueError(f"{horizon}-minute closes must be finite and positive")
        movement = np.abs(np.log(close / entry))
        sigma = atm_iv * math.sqrt(horizon / ANNUAL_TRADING_MINUTES)
        expectation = sigma * math.sqrt(2.0 / math.pi)
        output[f"absolute_log_return_{horizon}m"] = movement
        output[f"iv_sigma_{horizon}m"] = sigma
        output[f"iv_expected_absolute_{horizon}m"] = expectation
        output[f"iv_absolute_residual_{horizon}m"] = movement - expectation
        output[f"movement_exceeds_prior_close_iv_{horizon}m"] = (movement > expectation).astype(int)
    return output


def fit_minimal_models(development: pd.DataFrame) -> MinimalModels:
    """Fit exactly M0 and M1 with frozen weighted L2 logistic regression."""

    assert_2024_only_fitting(development["session"])
    validate_no_excluded_features(GROUP_O)
    validate_no_excluded_features([*GROUP_O, *GROUP_I])
    m0 = fit_cross_market_model(
        development,
        model_id="M0",
        numeric_features=GROUP_O,
        category_control_names=("stock",),
        target_column=TARGET_COLUMN,
        kind="logistic",
    )
    m1 = fit_cross_market_model(
        development,
        model_id="M1",
        numeric_features=(*GROUP_O, *GROUP_I),
        category_control_names=("stock",),
        target_column=TARGET_COLUMN,
        kind="logistic",
    )
    return MinimalModels(m0=m0, m1=m1)


def model_specification(model: FrozenCrossMarketModel) -> dict[str, object]:
    """Serialize every value needed to reconstruct one fitted model."""

    return {
        "model_id": model.model_id,
        "kind": model.kind,
        "numeric_features": list(model.numeric_features),
        "category_controls": list(model.category_controls),
        "numeric_medians": model.numeric_medians.astype(float).tolist(),
        "numeric_means": model.numeric_means.astype(float).tolist(),
        "numeric_scales": model.numeric_scales.astype(float).tolist(),
        "category_levels": {key: list(values) for key, values in model.category_levels.items()},
        "design_columns": list(model.design_columns),
        "coefficients": model.coefficients.astype(float).tolist(),
        "intercept": float(model.intercept),
        "iterations": int(model.iterations),
    }


def fixed_session_bootstrap_multiplicities(
    sessions: pd.Series,
    *,
    draws: int = 10,
    seed: int = BOOTSTRAP_SEED,
) -> list[IntArray]:
    """Return exactly ten fixed-seed whole-session multiplicity vectors."""

    if draws != 10:
        raise ValueError("holdout bootstrap requires exactly 10 draws")
    labels = sessions.astype(str).to_numpy()
    unique = np.asarray(sorted(set(labels)), dtype=object)
    if unique.size == 0:
        raise ValueError("bootstrap sessions are empty")
    rng = np.random.default_rng(seed)
    output: list[IntArray] = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        output.append(np.asarray([int(counts.get(value, 0)) for value in labels], dtype=np.int64))
    return output


def intraday_h0_bundle_null(
    frame: pd.DataFrame,
    *,
    group_i_columns: Sequence[str] = GROUP_I,
    seed: int,
) -> pd.DataFrame:
    """Permute each complete H0 bundle among stocks within each frozen slate."""

    columns = tuple(group_i_columns)
    missing = sorted(
        {"period", "session", "checkpoint", "symbol", *columns}.difference(frame.columns)
    )
    if missing:
        raise ValueError(f"H0 null inputs missing: {missing}")
    output = frame.copy()
    rng = np.random.default_rng(seed)
    grouped = output.groupby(["period", "session", "checkpoint"], sort=True, observed=True)
    for _, indices in grouped.indices.items():
        target = np.asarray(indices, dtype=np.int64)
        source = target[rng.permutation(len(target))]
        output.loc[output.index[target], list(columns)] = frame.iloc[source][
            list(columns)
        ].to_numpy()
    return output


def tail_overlap_metrics(
    m1_membership: Sequence[bool] | BoolArray,
    m0_membership: Sequence[bool] | BoolArray,
) -> dict[str, float | int]:
    """Calculate frozen M1/M0 tail set overlap."""

    m1 = np.asarray(m1_membership, dtype=np.bool_)
    m0 = np.asarray(m0_membership, dtype=np.bool_)
    if m1.ndim != 1 or m0.ndim != 1 or len(m1) != len(m0):
        raise ValueError("tail memberships must be aligned one-dimensional arrays")
    intersection = int(np.sum(m1 & m0))
    union = int(np.sum(m1 | m0))
    return {
        "intersection_rows": intersection,
        "union_rows": union,
        "jaccard_overlap": float(intersection / union) if union else math.nan,
        "M1_only_rows": int(np.sum(m1 & ~m0)),
        "M0_only_rows": int(np.sum(m0 & ~m1)),
    }


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    """Calculate a candidate-weighted mean."""

    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    if (
        len(values) == 0
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or bool((weights <= 0.0).any())
    ):
        raise ValueError(f"weighted mean inputs invalid: {column}")
    return float(np.sum(weights * values) / np.sum(weights))


def tail_metrics(frame: pd.DataFrame, *, model: str) -> dict[str, float | int | str]:
    """Calculate the complete frozen top-five-percent tail metric surface."""

    if frame.empty:
        raise ValueError("tail metrics require at least one selected row")
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    movement = pd.to_numeric(frame["absolute_log_return_15m"], errors="raise").to_numpy(float)
    residual = pd.to_numeric(frame["iv_absolute_residual_15m"], errors="raise").to_numpy(float)
    expectation = pd.to_numeric(frame["iv_expected_absolute_15m"], errors="raise").to_numpy(float)
    sigma = pd.to_numeric(frame["iv_sigma_15m"], errors="raise").to_numpy(float)
    total_weight = float(weights.sum())
    order = np.argsort(residual, kind="mergesort")
    trim_count = math.floor(len(frame) * 0.10)
    kept = (
        order[trim_count : len(frame) - trim_count]
        if trim_count > 0 and len(frame) - 2 * trim_count > 0
        else order
    )
    top_count = max(1, math.ceil(len(frame) * 0.05))
    top = order[-top_count:]
    positive = np.maximum(residual, 0.0)
    positive_total = float(np.sum(weights * positive))
    months = pd.to_datetime(frame["session"], errors="raise").dt.to_period("M").astype(str)
    session_share = frame["session"].astype(str).value_counts(normalize=True)
    return {
        "model": model,
        "tail": "frozen_top_5_percent",
        "rows": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "months": int(months.nunique()),
        "mean_absolute_movement": float(np.sum(weights * movement) / total_weight),
        "median_absolute_movement": weighted_quantile(movement, weights, 0.50),
        "mean_iv_expectation": float(np.sum(weights * expectation) / total_weight),
        "mean_iv_residual": float(np.sum(weights * residual) / total_weight),
        "median_iv_residual": weighted_quantile(residual, weights, 0.50),
        "trimmed_10pct_mean_iv_residual": float(
            np.sum(weights[kept] * residual[kept]) / np.sum(weights[kept])
        ),
        "exceed_iv_rate": weighted_mean(frame, TARGET_COLUMN),
        "positive_residual_rate": float(
            np.sum(weights * (residual > 0.0).astype(float)) / total_weight
        ),
        "iv_sigma_ratio": float(np.sum(weights * movement) / np.sum(weights * sigma)),
        "iv_residual_percentile_05": weighted_quantile(residual, weights, 0.05),
        "iv_residual_percentile_25": weighted_quantile(residual, weights, 0.25),
        "iv_residual_percentile_75": weighted_quantile(residual, weights, 0.75),
        "iv_residual_percentile_95": weighted_quantile(residual, weights, 0.95),
        "top_5pct_positive_residual_contribution": (
            float(np.sum(weights[top] * positive[top]) / positive_total)
            if positive_total > 0.0
            else math.nan
        ),
        "maximum_stock_share": float(
            frame["symbol"].astype(str).value_counts(normalize=True).max()
        ),
        "maximum_month_share": float(months.value_counts(normalize=True).max()),
        "maximum_session_share": float(session_share.max()),
    }


def tail_comparison_metrics(
    m1: Mapping[str, object],
    m0: Mapping[str, object],
) -> dict[str, float]:
    """Calculate frozen M1-minus-M0 top-tail differences."""

    def difference(key: str) -> float:
        return float(cast(Any, m1[key])) - float(cast(Any, m0[key]))

    return {
        "mean_iv_residual_difference": difference("mean_iv_residual"),
        "median_iv_residual_difference": difference("median_iv_residual"),
        "exceed_iv_rate_difference": difference("exceed_iv_rate"),
        "absolute_movement_difference": difference("mean_absolute_movement"),
        "iv_sigma_ratio_difference": difference("iv_sigma_ratio"),
        "positive_residual_rate_difference": difference("positive_residual_rate"),
        "tail_concentration_difference": difference("top_5pct_positive_residual_contribution"),
    }


def movement_timing_metrics(frame: pd.DataFrame, *, model: str = "M1") -> pd.DataFrame:
    """Report frozen-horizon residuals and 30-minute realization diagnostics."""

    if frame.empty:
        raise ValueError("movement timing requires a non-empty frozen tail")
    eventual = pd.to_numeric(frame["absolute_log_return_30m"], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    denominator = float(np.sum(weights * eventual))
    rows: list[dict[str, object]] = []
    movement_matrix = np.column_stack(
        [
            pd.to_numeric(frame[f"absolute_log_return_{horizon}m"], errors="raise").to_numpy(float)
            for horizon in HORIZONS
        ]
    )
    maximum_bucket = np.argmax(movement_matrix, axis=1)
    for index, horizon in enumerate(HORIZONS):
        residual = pd.to_numeric(
            frame[f"iv_absolute_residual_{horizon}m"], errors="raise"
        ).to_numpy(float)
        exceeds = pd.to_numeric(
            frame[f"movement_exceeds_prior_close_iv_{horizon}m"], errors="raise"
        ).to_numpy(float)
        movement = movement_matrix[:, index]
        rows.append(
            {
                "model": model,
                "horizon_minutes": horizon,
                "mean_iv_residual": float(np.sum(weights * residual) / weights.sum()),
                "median_iv_residual": weighted_quantile(residual, weights, 0.50),
                "exceed_iv_rate": float(np.sum(weights * exceeds) / weights.sum()),
                "percent_eventual_30m_movement_realized": (
                    float(np.sum(weights * movement) / denominator) if denominator > 0 else math.nan
                ),
                "maximum_absolute_excursion_bucket_share": float(
                    np.sum(weights * (maximum_bucket == index)) / weights.sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def model_metric_row(
    frame: pd.DataFrame,
    *,
    model: str,
    probability_column: str,
) -> dict[str, float | int | str]:
    """Calculate the requested holdout metric row for one model."""

    metrics = binary_metrics(
        frame,
        target_column=TARGET_COLUMN,
        probability_column=probability_column,
        boundaries={"top_decile": 1.0, "top_quintile": 1.0},
    )
    return {
        "model": model,
        **{
            key: value
            for key, value in metrics.items()
            if key
            not in {
                "top_decile_precision",
                "top_decile_lift",
                "top_quintile_precision",
                "top_quintile_lift",
            }
        },
    }


def model_increment(
    m0: Mapping[str, object],
    m1: Mapping[str, object],
) -> dict[str, float | str]:
    """Return M1-minus-M0 improvements with proper-score signs normalized."""

    return {
        "comparison": "M1_minus_M0",
        "log_loss_improvement": float(cast(Any, m0["log_loss"])) - float(cast(Any, m1["log_loss"])),
        "brier_improvement": float(cast(Any, m0["brier_score"]))
        - float(cast(Any, m1["brier_score"])),
        "auc_improvement": float(cast(Any, m1["auc"])) - float(cast(Any, m0["auc"])),
        "average_precision_improvement": float(cast(Any, m1["average_precision"]))
        - float(cast(Any, m0["average_precision"])),
        "expected_calibration_error_improvement": float(cast(Any, m0["expected_calibration_error"]))
        - float(cast(Any, m1["expected_calibration_error"])),
        "calibration_intercept_absolute_error_improvement": abs(
            float(cast(Any, m0["calibration_intercept"]))
        )
        - abs(float(cast(Any, m1["calibration_intercept"]))),
        "calibration_slope_absolute_error_improvement": abs(
            float(cast(Any, m0["calibration_slope"])) - 1.0
        )
        - abs(float(cast(Any, m1["calibration_slope"])) - 1.0),
    }


def joined_support(frame: pd.DataFrame) -> dict[str, object]:
    """Evaluate every preregistered joined-holdout support gate."""

    weights = pd.to_numeric(frame["row_weight"], errors="raise")
    total = float(weights.sum())
    stock_weight = frame.assign(_weight=weights).groupby("symbol")["_weight"].sum()
    months = pd.to_datetime(frame["session"], errors="raise").dt.to_period("M").astype(str)
    month_weight = frame.assign(_month=months, _weight=weights).groupby("_month")["_weight"].sum()
    gates = {
        "rows_at_least_5000": len(frame) >= 5_000,
        "sessions_at_least_60": frame["session"].nunique() >= 60,
        "stocks_at_least_15": frame["symbol"].nunique() >= 15,
        "all_four_months": months.nunique() == 4,
        "positive_outcomes_at_least_1000": int(frame[TARGET_COLUMN].sum()) >= 1_000,
        "maximum_stock_weight_share_at_most_0_12": float(stock_weight.max() / total) <= 0.12,
        "maximum_month_weight_share_at_most_0_35": float(month_weight.max() / total) <= 0.35,
    }
    return {
        "rows": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "months": int(months.nunique()),
        "positive_outcomes": int(frame[TARGET_COLUMN].sum()),
        "maximum_stock_weight_share": float(stock_weight.max() / total),
        "maximum_month_weight_share": float(month_weight.max() / total),
        "gates": gates,
        "passed": all(gates.values()),
    }


def frozen_tail_support(frame: pd.DataFrame) -> dict[str, object]:
    """Evaluate frozen M1 top-five-percent support and concentration gates."""

    months = pd.to_datetime(frame["session"], errors="raise").dt.to_period("M").astype(str)
    stock_share = float(frame["symbol"].astype(str).value_counts(normalize=True).max())
    month_share = float(months.value_counts(normalize=True).max())
    session_share = float(frame["session"].astype(str).value_counts(normalize=True).max())
    gates = {
        "rows_at_least_250": len(frame) >= 250,
        "sessions_at_least_40": frame["session"].nunique() >= 40,
        "stocks_at_least_12": frame["symbol"].nunique() >= 12,
        "all_four_months": months.nunique() == 4,
        "maximum_stock_share_at_most_0_18": stock_share <= 0.18,
        "maximum_month_share_at_most_0_40": month_share <= 0.40,
        "maximum_session_share_at_most_0_08": session_share <= 0.08,
    }
    return {
        "rows": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "months": int(months.nunique()),
        "maximum_stock_share": stock_share,
        "maximum_month_share": month_share,
        "maximum_session_share": session_share,
        "gates": gates,
        "passed": all(gates.values()),
    }


def decide_experiment(
    *,
    model: ModelGateInputs,
    tail: TailGateInputs,
) -> dict[str, object]:
    """Map frozen gate results to exactly one authorized decision category."""

    model_status = (
        "insufficient_support"
        if not model.support_passed
        else ("supported" if model.passed else "not_supported")
    )
    tail_status = (
        "insufficient_support"
        if not tail.support_passed
        else ("supported" if tail.passed else "not_supported")
    )
    if model.passed and tail.passed:
        overall = "minimal_intraday_h0_iv_excess_tail_validated"
    elif model.passed:
        overall = "minimal_intraday_h0_model_validated_but_tail_not_validated"
    elif tail.passed:
        overall = "positive_frozen_tail_without_model_increment"
    elif (
        model.log_loss_improvement > 0.0
        or model.brier_improvement > 0.0
        or model.average_precision_improvement > 0.0
        or tail.mean_iv_residual > 0.0
    ):
        overall = "minimal_model_descriptive_only"
    else:
        overall = "minimal_model_does_not_transfer"
    options_comparison_status = (
        "insufficient_support"
        if not tail.support_passed
        else (
            "supported"
            if tail.m1_minus_m0_mean > 0.0 and tail.bootstrap_80_difference_lower >= 0.0
            else "not_supported"
        )
    )
    return {
        **SAFETY_FLAGS,
        "overall_decision": overall,
        "minimal_model_status": model_status,
        "frozen_top_5pct_status": tail_status,
        "options_only_tail_comparison_status": options_comparison_status,
        "movement_timing_status": (
            "descriptive_only" if tail.support_passed else "insufficient_support"
        ),
        "holdout_options_coverage_status": (
            "supported" if model.support_passed else "insufficient_support"
        ),
        "model_gate": {**asdict(model), "passed": model.passed},
        "tail_gate": {**asdict(tail), "passed": tail.passed},
    }
