"""Frozen grouped stock-layer attribution helpers for the IV-excess quick screen."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from stocker_research.stock_options_cross_market_quick_v0 import (
    FrozenCrossMarketModel,
    binary_metrics,
    fit_cross_market_model,
)

SAFETY_FLAGS: Final[dict[str, object]] = {
    "research_only": True,
    "quick_grouped_ablation_screen": True,
    "previous_close_options_only": True,
    "frozen_joined_panel": True,
    "daily_stock_group_test": True,
    "intraday_compressed_transition_group_test": True,
    "route_competition_group_test": True,
    "cross_market_mismatch_group_test": True,
    "iv_excess_tail_test": True,
    "option_pnl_calculated": False,
    "intraday_option_quotes_used": False,
    "directional_outcomes_primary": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "prospective_validation": False,
}


def _as_float(value: object) -> float:
    """Narrow pandas' deliberately broad scalar typing to a runtime float."""

    return float(cast(Any, value))


CHECKPOINTS: Final[tuple[int, ...]] = tuple(range(6, 35, 2))
CHECKPOINT_FEATURES: Final[tuple[str, ...]] = tuple(
    f"checkpoint_{checkpoint}" for checkpoint in CHECKPOINTS
)
GROUP_O: Final[tuple[str, ...]] = (
    "front_options_implied_tension",
    "front_options_premium_richness",
    "front_options_downside_asymmetry",
    "front_options_liquidity_stress",
    "front_options_positioning_concentration",
    "front_options_directional_positioning",
    "front_options_surface_disagreement",
    "front_options_regime_p_0",
    "front_options_regime_p_1",
    "front_options_regime_p_2",
    "front_options_regime_p_3",
    "front_options_regime_entropy",
    "front_options_regime_margin",
    "skew_25d_missing",
    "near_spot_oi_concentration_missing",
    "call_put_oi_imbalance_missing",
    *CHECKPOINT_FEATURES,
)
GROUP_D: Final[tuple[str, ...]] = (
    "daily_compression",
    "daily_directional_efficiency",
    "daily_trend_persistence",
    "daily_extension",
    "daily_rejection",
    "daily_volatility_acceleration",
    "daily_relative_strength",
    "daily_activity_acceleration",
    "daily_stock_regime_p_0",
    "daily_stock_regime_p_1",
    "daily_stock_regime_p_2",
    "daily_stock_regime_p_3",
    "daily_stock_regime_entropy",
    "daily_stock_regime_margin",
)
GROUP_I: Final[tuple[str, ...]] = (
    "arousal",
    "conviction",
    "tension",
    "signed_pressure",
    "posterior_entropy",
    "transition_probability",
    "persistence_probability",
    "expected_state_age",
    "top_state_probability",
    "top_second_margin",
    "prior_6_mean_range",
    "prior_6_price_travel",
    "prior_6_absolute_net_movement",
    "prior_6_activity_proxy",
    "recent_vs_earlier_range_ratio",
    "recent_vs_earlier_activity_ratio",
    "current_bar_range_vs_prior_6",
    "current_bar_activity_vs_prior_6",
    "current_bar_body_fraction",
    "current_bar_extreme_wick_fraction",
    "any_registered_completion_prior_6",
    "any_registered_completion_prior_12",
    "same_identity_active_prefix_with_prior_completion",
    "any_hidden_event_prior_6",
    "hidden_2_3_2_prior_6",
    "bars_since_latest_registered_completion",
)
GROUP_R: Final[tuple[str, ...]] = (
    "active_prefix_count",
    "active_prefix_family_count",
    "top_prefix_depth_fraction",
    "second_prefix_depth_fraction",
    "top_minus_second_prefix_depth",
    "prefix_family_entropy",
    "orientation_disagreement_fraction",
    "new_prefixes_last_1_bar",
    "invalidated_prefixes_last_1_bar",
    "active_prefix_count_change_last_1_bar",
    "active_prefix_count_change_last_3_bars",
    "top_prefix_depth_change_last_1_bar",
    "top_prefix_depth_change_last_3_bars",
    "matching_recent_loop_prefix_count",
    "recent_loop_memory_weighted_top_depth",
)
GROUP_M: Final[tuple[str, ...]] = (
    "mismatch_compression_vs_front_iv",
    "mismatch_daily_volatility_vs_front_iv",
    "mismatch_route_vs_front_premium",
    "mismatch_direction_agreement",
    "mismatch_complacent_broad_conflict",
)
ROUTE_STATE_LEVELS: Final[tuple[str, ...]] = (
    "BROAD_CONFLICT",
    "NARROWING",
    "DOMINANT_ROUTE",
    "LOW_ROUTE_SUPPORT",
    "OTHER",
)
FEATURE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "O": GROUP_O,
    "D": GROUP_D,
    "I": GROUP_I,
    "R": GROUP_R,
    "M": GROUP_M,
}
MODEL_FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "G0": GROUP_O,
    "G1": (*GROUP_O, *GROUP_D),
    "G2": (*GROUP_O, *GROUP_D, *GROUP_I),
    "G3": (*GROUP_O, *GROUP_D, *GROUP_I, *GROUP_R),
    "G4": (*GROUP_O, *GROUP_D, *GROUP_I, *GROUP_R, *GROUP_M),
}
MODEL_CONTROLS: Final[dict[str, tuple[str, ...]]] = {
    "G0": ("stock",),
    "G1": ("stock",),
    "G2": ("stock",),
    "G3": ("stock", "route_state"),
    "G4": ("stock", "route_state"),
}
TARGET_COLUMN: Final[str] = "movement_exceeds_prior_close_iv_15m"
ANNUAL_TRADING_MINUTES: Final[int] = 252 * 390
PROTECTED_START: Final[pd.Timestamp] = pd.Timestamp("2025-08-23")
MODEL_ORDER: Final[tuple[str, ...]] = ("G0", "G1", "G2", "G3", "G4")
ADJACENT_COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    ("G0", "G1"),
    ("G1", "G2"),
    ("G2", "G3"),
    ("G3", "G4"),
)
GROUP_MODEL_STEPS: Final[dict[str, tuple[str, str]]] = {
    "D": ("G0", "G1"),
    "I": ("G1", "G2"),
    "R": ("G2", "G3"),
    "M": ("G3", "G4"),
}
GROUP_PERMUTATION_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "D": GROUP_D,
    "I": GROUP_I,
    "R": (*GROUP_R, "route_resolution_state"),
    "M": GROUP_M,
}
PERMUTATION_SEEDS: Final[dict[str, tuple[int, ...]]] = {
    "D": (20260731, 20260732, 20260733, 20260734, 20260735),
    "I": (20260736, 20260737, 20260738, 20260739, 20260740),
    "R": (20260741, 20260742, 20260743, 20260744, 20260745),
    "M": (20260746, 20260747, 20260748, 20260749, 20260750),
}
NULL_SEEDS: Final[dict[str, tuple[int, ...]]] = {
    "D": (20260751, 20260752, 20260753),
    "I": (20260754, 20260755, 20260756),
    "R": (20260757, 20260758, 20260759),
    "M": (20260760, 20260761, 20260762),
}
BOOTSTRAP_SEED: Final[int] = 20260763
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
FRONT_CONTEXT_FEATURES: Final[tuple[str, ...]] = GROUP_O[:16]
FRONT_CONTRACT_COLUMNS: Final[tuple[str, ...]] = (
    "options_observation_date",
    "front_expiration_date",
    "front_strike",
    "front_call_contract_id",
    "front_put_contract_id",
    "skew_put_contract_id",
    "skew_call_contract_id",
)


@dataclass(frozen=True)
class LadderResult:
    """Five frozen primary models and their aligned predictions and metrics."""

    development: pd.DataFrame
    assessment: pd.DataFrame
    models: Mapping[str, FrozenCrossMarketModel]
    thresholds: Mapping[str, Mapping[str, float]]
    metrics: pd.DataFrame


@dataclass(frozen=True)
class NullRefitResult:
    """Exactly twelve group-specific null refits and their serializable models."""

    metrics: pd.DataFrame
    models: Mapping[str, FrozenCrossMarketModel]


@dataclass(frozen=True)
class StabilityTables:
    """Frozen monthly, checkpoint/context, and route-state increment tables."""

    monthly: pd.DataFrame
    checkpoint_and_context: pd.DataFrame
    route_state: pd.DataFrame
    development_medians: Mapping[str, float]


@dataclass(frozen=True)
class TailTables:
    """Final-model tails, G0 comparison, overlap, capture, and concentration."""

    metrics: pd.DataFrame
    comparison: pd.DataFrame
    overlap: pd.DataFrame
    incremental_capture: pd.DataFrame
    concentration: pd.DataFrame


@dataclass(frozen=True)
class TailGateResult:
    """Binding final-tail and G0-versus-G4 comparison statuses."""

    final_status: str
    options_only_vs_stock_status: str
    gates: Mapping[str, object]


def assert_safety_flags(value: Mapping[str, object]) -> None:
    """Fail when a contract or decision weakens a binding safety flag."""

    mismatches = {
        key: (expected, value.get(key))
        for key, expected in SAFETY_FLAGS.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"stock-layer attribution safety flags differ: {mismatches}")


def validate_feature_groups() -> None:
    """Require exact group disjointness and the frozen nested model ladder."""

    seen: set[str] = set()
    for name, features in FEATURE_GROUPS.items():
        if len(features) != len(set(features)):
            raise ValueError(f"feature group {name} contains a duplicate")
        overlap = seen.intersection(features)
        if overlap:
            raise ValueError(f"feature group {name} overlaps earlier groups: {sorted(overlap)}")
        seen.update(features)
    expected = {
        "G0": GROUP_O,
        "G1": (*GROUP_O, *GROUP_D),
        "G2": (*GROUP_O, *GROUP_D, *GROUP_I),
        "G3": (*GROUP_O, *GROUP_D, *GROUP_I, *GROUP_R),
        "G4": (*GROUP_O, *GROUP_D, *GROUP_I, *GROUP_R, *GROUP_M),
    }
    if expected != MODEL_FEATURES:
        raise ValueError("frozen G0-G4 feature ladder drifted")


def _maximum_numeric_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    if len(left) != len(right):
        return math.inf
    maximum = 0.0
    for column in columns:
        left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        both_nan = np.isnan(left_values) & np.isnan(right_values)
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if bool((~both_nan & ~finite).any()):
            return math.inf
        if bool(finite.any()):
            maximum = max(
                maximum,
                float(np.max(np.abs(left_values[finite] - right_values[finite]))),
            )
    return maximum


def reconstruct_frozen_branch_c_panel(
    panel: pd.DataFrame,
    *,
    dense_panel: pd.DataFrame,
    daily_stock_context: pd.DataFrame,
    front_options_context: pd.DataFrame,
    front_options_raw: pd.DataFrame,
) -> dict[str, object]:
    """Independently cross-check the frozen joined Branch C panel against its layers."""

    validate_feature_groups()
    required = {
        "row_id",
        "symbol",
        "session",
        "period",
        "checkpoint",
        "checkpoint_timestamp_utc",
        "stock_information_date",
        *FRONT_CONTRACT_COLUMNS,
        "entry_price",
        "close_15m",
        "atm_iv",
        "row_weight",
        *MODEL_FEATURES["G4"],
        "route_resolution_state",
        "iv_sigma_15m",
        "iv_expected_absolute_15m",
        "absolute_log_return_15m",
        "iv_absolute_residual_15m",
        TARGET_COLUMN,
    }
    if missing := sorted(required.difference(panel.columns)):
        raise ValueError(f"frozen Branch C panel missing columns: {missing}")
    frozen = panel.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    if frozen["row_id"].duplicated().any():
        raise ValueError("blocked_frozen_branch_c_panel_reconstruction_failure: duplicate row_id")

    dense_columns = (
        "row_id",
        "symbol",
        "session",
        "period",
        "checkpoint",
        "checkpoint_timestamp_utc",
        "row_weight",
        *GROUP_I,
        *CHECKPOINT_FEATURES,
        *GROUP_R,
        "route_resolution_state",
    )
    dense_reference = dense_panel.loc[:, list(dense_columns)].drop_duplicates("row_id")
    dense_joined = frozen.loc[:, list(dense_columns)].merge(
        dense_reference,
        on="row_id",
        how="left",
        validate="one_to_one",
        suffixes=("_frozen", "_reference"),
        indicator=True,
    )
    identity_columns = ("symbol", "session", "period", "checkpoint", "checkpoint_timestamp_utc")
    row_identity_mismatches = int(dense_joined["_merge"].ne("both").sum())
    for column in identity_columns:
        row_identity_mismatches += int(
            dense_joined[f"{column}_frozen"]
            .astype(str)
            .ne(dense_joined[f"{column}_reference"].astype(str))
            .sum()
        )
    dense_left = dense_joined.rename(
        columns={
            f"{column}_frozen": column
            for column in ("row_weight", *GROUP_I, *CHECKPOINT_FEATURES, *GROUP_R)
        }
    )
    dense_right = dense_joined.rename(
        columns={
            f"{column}_reference": column
            for column in ("row_weight", *GROUP_I, *CHECKPOINT_FEATURES, *GROUP_R)
        }
    )
    dense_difference = _maximum_numeric_difference(
        dense_left,
        dense_right,
        ("row_weight", *GROUP_I, *CHECKPOINT_FEATURES, *GROUP_R),
    )
    route_state_mismatches = int(
        dense_joined["route_resolution_state_frozen"]
        .astype(str)
        .ne(dense_joined["route_resolution_state_reference"].astype(str))
        .sum()
    )

    daily_reference = daily_stock_context.loc[
        :,
        ["symbol", "session", "period", *GROUP_D],
    ].drop_duplicates(["symbol", "session", "period"])
    daily_joined = frozen.loc[:, ["row_id", "symbol", "session", "period", *GROUP_D]].merge(
        daily_reference,
        on=["symbol", "session", "period"],
        how="left",
        validate="many_to_one",
        suffixes=("_frozen", "_reference"),
        indicator=True,
    )
    daily_left = daily_joined.rename(columns={f"{column}_frozen": column for column in GROUP_D})
    daily_right = daily_joined.rename(columns={f"{column}_reference": column for column in GROUP_D})
    daily_difference = _maximum_numeric_difference(
        daily_left,
        daily_right,
        GROUP_D,
    )

    front_reference = front_options_context.loc[
        :,
        ["symbol", "session", "period", *FRONT_CONTEXT_FEATURES],
    ].drop_duplicates(["symbol", "session", "period"])
    front_joined = frozen.loc[
        :,
        ["row_id", "symbol", "session", "period", *FRONT_CONTEXT_FEATURES],
    ].merge(
        front_reference,
        on=["symbol", "session", "period"],
        how="left",
        validate="many_to_one",
        suffixes=("_frozen", "_reference"),
        indicator=True,
    )
    front_left = front_joined.rename(
        columns={f"{column}_frozen": column for column in FRONT_CONTEXT_FEATURES}
    )
    front_right = front_joined.rename(
        columns={f"{column}_reference": column for column in FRONT_CONTEXT_FEATURES}
    )
    front_difference = _maximum_numeric_difference(
        front_left,
        front_right,
        FRONT_CONTEXT_FEATURES,
    )

    frozen_contracts = frozen.loc[
        :,
        ["symbol", "session", *FRONT_CONTRACT_COLUMNS],
    ].drop_duplicates(["symbol", "session"])
    raw_contracts = front_options_raw.loc[
        :,
        ["symbol", "session", *FRONT_CONTRACT_COLUMNS],
    ].drop_duplicates(["symbol", "session"])
    contract_joined = frozen_contracts.merge(
        raw_contracts,
        on=["symbol", "session"],
        how="left",
        validate="one_to_one",
        suffixes=("_frozen", "_reference"),
        indicator=True,
    )
    selected_contract_mismatches = int(contract_joined["_merge"].ne("both").sum())
    for column in FRONT_CONTRACT_COLUMNS:
        selected_contract_mismatches += int(
            contract_joined[f"{column}_frozen"]
            .astype(str)
            .ne(contract_joined[f"{column}_reference"].astype(str))
            .sum()
        )

    development = frozen.loc[frozen["period"].astype(str).eq("development")]

    def standardize(column: str) -> pd.Series:
        mean = float(pd.to_numeric(development[column], errors="raise").mean())
        scale = float(pd.to_numeric(development[column], errors="raise").std(ddof=0))
        if not math.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        return (pd.to_numeric(frozen[column], errors="raise") - mean) / scale

    tension = standardize("front_options_implied_tension")
    rebuilt_mismatch = pd.DataFrame(
        {
            "mismatch_compression_vs_front_iv": (standardize("daily_compression") - tension),
            "mismatch_daily_volatility_vs_front_iv": (
                standardize("daily_volatility_acceleration") - tension
            ),
            "mismatch_route_vs_front_premium": (
                standardize("prefix_family_entropy") - standardize("front_options_premium_richness")
            ),
            "mismatch_direction_agreement": (
                standardize("signed_pressure")
                * standardize("front_options_directional_positioning")
            ),
            "mismatch_complacent_broad_conflict": (
                frozen["route_resolution_state"].astype(str).eq("BROAD_CONFLICT").astype(float)
                * -tension
            ),
        }
    )
    mismatch_difference = _maximum_numeric_difference(
        frozen.loc[:, list(GROUP_M)],
        rebuilt_mismatch,
        GROUP_M,
    )

    rebuilt_outcomes = calculate_iv_excess_outcomes(
        entry_price=frozen["entry_price"].to_numpy(float).tolist(),
        close_15m=frozen["close_15m"].to_numpy(float).tolist(),
        atm_iv=frozen["atm_iv"].to_numpy(float).tolist(),
    )
    outcome_columns = (
        "absolute_log_return_15m",
        "iv_sigma_15m",
        "iv_expected_absolute_15m",
        "iv_absolute_residual_15m",
    )
    maximum_outcome_difference = _maximum_numeric_difference(
        frozen.loc[:, list(outcome_columns)].reset_index(drop=True),
        rebuilt_outcomes.loc[:, list(outcome_columns)],
        outcome_columns,
    )
    target_mismatches = int(
        frozen[TARGET_COLUMN]
        .to_numpy(int)
        .__ne__(rebuilt_outcomes[TARGET_COLUMN].to_numpy(int))
        .sum()
    )
    periods = frozen["period"].astype(str)
    assessment = frozen.loc[periods.eq("assessment")]
    development_rows = int(periods.eq("development").sum())
    observed_cohort = tuple(sorted(frozen["symbol"].astype(str).unique()))
    maximum_feature_difference = max(
        dense_difference,
        daily_difference,
        front_difference,
        mismatch_difference,
    )
    result: dict[str, object] = {
        **SAFETY_FLAGS,
        "rows": int(len(frozen)),
        "development_rows": development_rows,
        "assessment_rows": int(len(assessment)),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_months": int(pd.to_datetime(assessment["session"]).dt.to_period("M").nunique()),
        "assessment_positive_outcomes": int(assessment[TARGET_COLUMN].sum()),
        "cohort": list(observed_cohort),
        "row_identity_mismatches": row_identity_mismatches,
        "route_state_mismatches": route_state_mismatches,
        "selected_contract_mismatches": selected_contract_mismatches,
        "maximum_dense_feature_difference": dense_difference,
        "maximum_daily_feature_difference": daily_difference,
        "maximum_front_feature_difference": front_difference,
        "maximum_mismatch_feature_difference": mismatch_difference,
        "maximum_candidate_weight_difference": dense_difference,
        "maximum_feature_difference": maximum_feature_difference,
        "maximum_outcome_difference": maximum_outcome_difference,
        "target_mismatches": target_mismatches,
    }
    result["passed"] = bool(
        len(frozen) == 24_130
        and development_rows == 13_865
        and len(assessment) == 10_265
        and assessment["session"].nunique() == 154
        and assessment["symbol"].nunique() == 20
        and pd.to_datetime(assessment["session"]).dt.to_period("M").nunique() == 8
        and int(assessment[TARGET_COLUMN].sum()) == 2_921
        and observed_cohort == FROZEN_COHORT
        and row_identity_mismatches == 0
        and route_state_mismatches == 0
        and selected_contract_mismatches == 0
        and maximum_feature_difference <= 1e-12
        and maximum_outcome_difference <= 1e-12
        and target_mismatches == 0
    )
    return result


def calculate_iv_excess_outcomes(
    *,
    entry_price: Sequence[float],
    close_15m: Sequence[float],
    atm_iv: Sequence[float],
) -> pd.DataFrame:
    """Calculate the frozen 15-minute underlying movement and IV-excess outcomes."""

    entry = np.asarray(entry_price, dtype=float)
    close = np.asarray(close_15m, dtype=float)
    iv = np.asarray(atm_iv, dtype=float)
    valid = (
        entry.ndim == close.ndim == iv.ndim == 1
        and len(entry) == len(close) == len(iv)
        and np.isfinite(entry).all()
        and np.isfinite(close).all()
        and np.isfinite(iv).all()
        and bool((entry > 0.0).all())
        and bool((close > 0.0).all())
        and bool((iv > 0.0).all())
    )
    if not valid:
        raise ValueError("15-minute IV-excess inputs must be aligned, finite, and positive")
    movement = np.abs(np.log(close / entry))
    sigma = iv * math.sqrt(15.0 / ANNUAL_TRADING_MINUTES)
    expectation = sigma * math.sqrt(2.0 / math.pi)
    return pd.DataFrame(
        {
            "entry_price": entry,
            "close_15m": close,
            "absolute_log_return_15m": movement,
            "iv_sigma_15m": sigma,
            "iv_expected_absolute_15m": expectation,
            TARGET_COLUMN: (movement > expectation).astype(int),
            "iv_absolute_residual_15m": movement - expectation,
        }
    )


def development_prediction_thresholds(
    probabilities: Sequence[float],
) -> dict[str, float]:
    """Freeze all binding prediction-tail thresholds on development probabilities."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("development prediction thresholds require finite probabilities")
    return {
        "top_decile": float(np.quantile(values, 0.90)),
        "top_quintile": float(np.quantile(values, 0.80)),
        "top_5pct": float(np.quantile(values, 0.95)),
        "top_2pct": float(np.quantile(values, 0.98)),
    }


def fit_model_ladder(panel: pd.DataFrame) -> LadderResult:
    """Fit exactly G0-G4 once with 2024-only preprocessing and frozen settings."""

    validate_feature_groups()
    required = {
        "row_id",
        "period",
        "session",
        "symbol",
        "checkpoint",
        "route_resolution_state",
        "row_weight",
        TARGET_COLUMN,
        *(feature for features in FEATURE_GROUPS.values() for feature in features),
    }
    if missing := sorted(required.difference(panel.columns)):
        raise ValueError(f"model ladder panel missing columns: {missing}")
    development = (
        panel.loc[panel["period"].astype(str).eq("development")]
        .sort_values("row_id", kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    assessment = (
        panel.loc[panel["period"].astype(str).eq("assessment")]
        .sort_values("row_id", kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    if development.empty or assessment.empty:
        raise ValueError("model ladder requires development and assessment rows")
    development_sessions = pd.to_datetime(development["session"], errors="raise")
    assessment_sessions = pd.to_datetime(assessment["session"], errors="raise")
    if not bool(development_sessions.dt.year.eq(2024).all()):
        raise ValueError("blocked_chronology_or_leakage_failure: development is not 2024 only")
    if not bool(assessment_sessions.dt.year.eq(2025).all()):
        raise ValueError("blocked_chronology_or_leakage_failure: assessment is not 2025 only")
    models: dict[str, FrozenCrossMarketModel] = {}
    thresholds: dict[str, Mapping[str, float]] = {}
    metric_rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        model = fit_cross_market_model(
            development,
            model_id=model_id,
            numeric_features=MODEL_FEATURES[model_id],
            category_control_names=MODEL_CONTROLS[model_id],
            target_column=TARGET_COLUMN,
            kind="logistic",
        )
        models[model_id] = model
        development_probability = model.predict(development)
        assessment_probability = model.predict(assessment)
        probability_column = f"{model_id}_probability"
        development[probability_column] = development_probability
        assessment[probability_column] = assessment_probability
        model_thresholds = development_prediction_thresholds(development_probability.tolist())
        thresholds[model_id] = model_thresholds
        values = binary_metrics(
            assessment,
            target_column=TARGET_COLUMN,
            probability_column=probability_column,
            boundaries=model_thresholds,
        )
        metric_rows.append({"model": model_id, **values})
    return LadderResult(
        development=development,
        assessment=assessment,
        models=models,
        thresholds=thresholds,
        metrics=pd.DataFrame(metric_rows),
    )


def _metric_increment(
    earlier: Mapping[str, Any],
    later: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "log_loss_improvement": float(earlier["log_loss"]) - float(later["log_loss"]),
        "brier_improvement": float(earlier["brier_score"]) - float(later["brier_score"]),
        "auc_improvement": float(later["auc"]) - float(earlier["auc"]),
        "average_precision_improvement": (
            float(later["average_precision"]) - float(earlier["average_precision"])
        ),
        "expected_calibration_error_improvement": (
            float(earlier["expected_calibration_error"])
            - float(later["expected_calibration_error"])
        ),
        "top_decile_precision_improvement": (
            float(later["top_decile_precision"]) - float(earlier["top_decile_precision"])
        ),
        "top_quintile_precision_improvement": (
            float(later["top_quintile_precision"]) - float(earlier["top_quintile_precision"])
        ),
        "mean_realised_class_probability_improvement": (
            float(later["mean_probability_realised_class"])
            - float(earlier["mean_probability_realised_class"])
        ),
    }


def adjacent_increment_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return every binding adjacent G0-G4 increment with positive meaning improvement."""

    indexed = metrics.set_index("model")
    if missing := sorted(set(MODEL_ORDER).difference(indexed.index.astype(str))):
        raise ValueError(f"adjacent metrics missing models: {missing}")
    rows: list[dict[str, float | str]] = []
    for earlier, later in ADJACENT_COMPARISONS:
        earlier_row = cast(dict[str, Any], cast(pd.Series, indexed.loc[earlier]).to_dict())
        later_row = cast(dict[str, Any], cast(pd.Series, indexed.loc[later]).to_dict())
        rows.append(
            {
                "comparison": f"{later}-{earlier}",
                "earlier_model": earlier,
                "later_model": later,
                **_metric_increment(earlier_row, later_row),
            }
        )
    return pd.DataFrame(rows)


def _prediction_metrics(
    frame: pd.DataFrame,
    *,
    model_id: str,
    thresholds: Mapping[str, Mapping[str, float]],
) -> dict[str, float | int]:
    return binary_metrics(
        frame,
        target_column=TARGET_COLUMN,
        probability_column=f"{model_id}_probability",
        boundaries=thresholds[model_id],
    )


def _comparison_increment(
    frame: pd.DataFrame,
    *,
    earlier: str,
    later: str,
    thresholds: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    earlier_metrics = _prediction_metrics(frame, model_id=earlier, thresholds=thresholds)
    later_metrics = _prediction_metrics(frame, model_id=later, thresholds=thresholds)
    return _metric_increment(earlier_metrics, later_metrics)


def _stability_increment_table(
    assessment: pd.DataFrame,
    *,
    thresholds: Mapping[str, Mapping[str, float]],
    scope: str,
    groups: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = groups.astype(str)
    ordered_groups = tuple(dict.fromkeys(labels.tolist()))
    for group in ordered_groups:
        subset = assessment.loc[labels.eq(group)]
        if subset.empty:
            continue
        for earlier, later in ADJACENT_COMPARISONS:
            earlier_metrics = _prediction_metrics(
                subset,
                model_id=earlier,
                thresholds=thresholds,
            )
            later_metrics = _prediction_metrics(
                subset,
                model_id=later,
                thresholds=thresholds,
            )
            rows.append(
                {
                    "comparison": f"{later}-{earlier}",
                    "earlier_model": earlier,
                    "later_model": later,
                    "scope": scope,
                    "group": group,
                    "rows": len(subset),
                    "sessions": int(subset["session"].nunique()),
                    "stocks": int(subset["symbol"].nunique()),
                    "positive_outcomes": int(subset[TARGET_COLUMN].sum()),
                    "earlier_log_loss": float(earlier_metrics["log_loss"]),
                    "later_log_loss": float(later_metrics["log_loss"]),
                    "earlier_brier": float(earlier_metrics["brier_score"]),
                    "later_brier": float(later_metrics["brier_score"]),
                    "earlier_auc": float(earlier_metrics["auc"]),
                    "later_auc": float(later_metrics["auc"]),
                    "earlier_average_precision": float(earlier_metrics["average_precision"]),
                    "later_average_precision": float(later_metrics["average_precision"]),
                    **_metric_increment(earlier_metrics, later_metrics),
                }
            )
    return pd.DataFrame(rows)


def build_stability_tables(result: LadderResult) -> StabilityTables:
    """Evaluate every adjacent increment on all preregistered stability groups."""

    assessment = result.assessment
    development = result.development
    monthly_groups = (
        pd.to_datetime(assessment["session"], errors="raise").dt.to_period("M").astype(str)
    )
    monthly = _stability_increment_table(
        assessment,
        thresholds=result.thresholds,
        scope="month",
        groups=monthly_groups,
    )
    checkpoint_groups = pd.Series(
        np.select(
            [
                assessment["checkpoint"].astype(int).between(6, 14),
                assessment["checkpoint"].astype(int).between(16, 24),
                assessment["checkpoint"].astype(int).between(26, 34),
            ],
            ["early_6_14", "middle_16_24", "late_26_34"],
            default="invalid",
        ),
        index=assessment.index,
        dtype="string",
    )
    if checkpoint_groups.eq("invalid").any():
        raise ValueError("assessment contains a checkpoint outside the frozen groups")
    checkpoint_tables = [
        _stability_increment_table(
            assessment,
            thresholds=result.thresholds,
            scope="checkpoint_group",
            groups=checkpoint_groups,
        )
    ]
    median_columns = {
        "prior_close_atm_iv": "atm_iv",
        "front_options_implied_tension": "front_options_implied_tension",
        "posterior_transition_probability": "transition_probability",
    }
    medians: dict[str, float] = {}
    for scope, column in median_columns.items():
        median = float(pd.to_numeric(development[column], errors="raise").median())
        medians[scope] = median
        labels = pd.Series(
            np.where(
                pd.to_numeric(assessment[column], errors="raise").le(median),
                "low",
                "high",
            ),
            index=assessment.index,
            dtype="string",
        )
        checkpoint_tables.append(
            _stability_increment_table(
                assessment,
                thresholds=result.thresholds,
                scope=scope,
                groups=labels,
            )
        )
    route_groups = (
        assessment["route_resolution_state"]
        .astype(str)
        .where(
            assessment["route_resolution_state"]
            .astype(str)
            .isin({"BROAD_CONFLICT", "NARROWING", "LOW_ROUTE_SUPPORT"}),
            "OTHER",
        )
    )
    route_state = _stability_increment_table(
        assessment,
        thresholds=result.thresholds,
        scope="route_state",
        groups=route_groups,
    )
    return StabilityTables(
        monthly=monthly,
        checkpoint_and_context=pd.concat(checkpoint_tables, ignore_index=True),
        route_state=route_state,
        development_medians=medians,
    )


def grouped_permutation_attribution(result: LadderResult) -> pd.DataFrame:
    """Run five fixed-seed, frozen-G4 within-slate permutations per added stock group."""

    assessment = result.assessment
    base = _prediction_metrics(assessment, model_id="G4", thresholds=result.thresholds)
    rows: list[dict[str, object]] = []
    for group in ("D", "I", "R", "M"):
        for permutation, seed in enumerate(PERMUTATION_SEEDS[group]):
            permuted = permute_group_within_slates(
                assessment,
                columns=GROUP_PERMUTATION_COLUMNS[group],
                slate_columns=("session", "checkpoint"),
                seed=seed,
            )
            permuted["G4_probability"] = result.models["G4"].predict(permuted)
            metrics = _prediction_metrics(
                permuted,
                model_id="G4",
                thresholds=result.thresholds,
            )
            rows.append(
                {
                    "group": group,
                    "permutation": permutation,
                    "seed": seed,
                    "within_slate": "session_x_checkpoint",
                    "bundle_columns": "|".join(GROUP_PERMUTATION_COLUMNS[group]),
                    "refit": False,
                    "log_loss_deterioration": float(metrics["log_loss"]) - float(base["log_loss"]),
                    "brier_deterioration": float(metrics["brier_score"])
                    - float(base["brier_score"]),
                    "auc_deterioration": float(base["auc"]) - float(metrics["auc"]),
                    "average_precision_deterioration": float(base["average_precision"])
                    - float(metrics["average_precision"]),
                    "top_decile_precision_deterioration": (
                        float(base["top_decile_precision"]) - float(metrics["top_decile_precision"])
                    ),
                }
            )
    output = pd.DataFrame(rows)
    if len(output) != 20 or not output.groupby("group").size().eq(5).all():
        raise AssertionError("grouped attribution must contain five permutations per group")
    return output


def group_null_refits(panel: pd.DataFrame, result: LadderResult) -> NullRefitResult:
    """Run exactly three frozen-design null refits for each added feature group."""

    real_increments = adjacent_increment_metrics(result.metrics).set_index("comparison")
    earlier_metrics = result.metrics.set_index("model")
    rows: list[dict[str, object]] = []
    models: dict[str, FrozenCrossMarketModel] = {}
    for group in ("D", "I", "R", "M"):
        earlier, later = GROUP_MODEL_STEPS[group]
        comparison = f"{later}-{earlier}"
        real = cast(pd.Series, real_increments.loc[comparison])
        for null_refit, seed in enumerate(NULL_SEEDS[group]):
            permuted = permute_group_within_slates(
                panel,
                columns=GROUP_PERMUTATION_COLUMNS[group],
                slate_columns=("period", "session", "checkpoint"),
                seed=seed,
            )
            development = (
                permuted.loc[permuted["period"].astype(str).eq("development")]
                .sort_values("row_id", kind="mergesort")
                .reset_index(drop=True)
            )
            assessment = (
                permuted.loc[permuted["period"].astype(str).eq("assessment")]
                .sort_values("row_id", kind="mergesort")
                .reset_index(drop=True)
                .copy()
            )
            null_model_id = f"{later}_null_{group}_{null_refit}"
            model = fit_cross_market_model(
                development,
                model_id=null_model_id,
                numeric_features=MODEL_FEATURES[later],
                category_control_names=MODEL_CONTROLS[later],
                target_column=TARGET_COLUMN,
                kind="logistic",
            )
            models[null_model_id] = model
            development_probabilities = model.predict(development)
            assessment[f"{later}_probability"] = model.predict(assessment)
            null_thresholds = {
                **result.thresholds,
                later: development_prediction_thresholds(development_probabilities.tolist()),
            }
            null_later_metrics = _prediction_metrics(
                assessment,
                model_id=later,
                thresholds=null_thresholds,
            )
            null_increment = _metric_increment(
                cast(
                    dict[str, Any],
                    cast(pd.Series, earlier_metrics.loc[earlier]).to_dict(),
                ),
                null_later_metrics,
            )
            rows.append(
                {
                    "group": group,
                    "comparison": comparison,
                    "null_refit": null_refit,
                    "seed": seed,
                    "null_model_id": null_model_id,
                    "within_slate": "period_x_session_x_checkpoint",
                    "permuted_columns": "|".join(GROUP_PERMUTATION_COLUMNS[group]),
                    **{
                        name: value
                        for name, value in null_increment.items()
                        if name
                        in {
                            "log_loss_improvement",
                            "brier_improvement",
                            "auc_improvement",
                            "average_precision_improvement",
                        }
                    },
                    "real_log_loss_improvement": _as_float(real["log_loss_improvement"]),
                    "real_brier_improvement": _as_float(real["brier_improvement"]),
                    "real_auc_improvement": _as_float(real["auc_improvement"]),
                    "real_average_precision_improvement": _as_float(
                        real["average_precision_improvement"]
                    ),
                    "real_exceeds_null_log_loss_improvement": (
                        _as_float(real["log_loss_improvement"])
                        > null_increment["log_loss_improvement"]
                    ),
                    "real_exceeds_null_brier_improvement": (
                        _as_float(real["brier_improvement"]) > null_increment["brier_improvement"]
                    ),
                    "real_exceeds_null_auc_improvement": (
                        _as_float(real["auc_improvement"]) > null_increment["auc_improvement"]
                    ),
                    "real_exceeds_null_average_precision_improvement": (
                        _as_float(real["average_precision_improvement"])
                        > null_increment["average_precision_improvement"]
                    ),
                }
            )
    output = pd.DataFrame(rows)
    if len(output) != 12 or len(models) != 12:
        raise AssertionError("group null design must fit exactly twelve null models")
    if not output.groupby("group").size().eq(3).all():
        raise AssertionError("group null design must fit exactly three models per group")
    return NullRefitResult(metrics=output, models=models)


def _tail_comparison_values(assessment: pd.DataFrame) -> dict[str, float]:
    g4 = tail_metrics(
        assessment.loc[assessment["G4_top_decile"].astype(bool)],
        model="G4",
        tail="top_decile",
    )
    g0 = tail_metrics(
        assessment.loc[assessment["G0_top_decile"].astype(bool)],
        model="G0",
        tail="top_decile",
    )
    return {
        "mean_iv_residual_difference": _as_float(g4["mean_iv_residual"])
        - _as_float(g0["mean_iv_residual"]),
        "median_iv_residual_difference": _as_float(g4["median_iv_residual"])
        - _as_float(g0["median_iv_residual"]),
        "exceed_iv_rate_difference": _as_float(g4["exceed_iv_rate"])
        - _as_float(g0["exceed_iv_rate"]),
        "iv_sigma_ratio_difference": _as_float(g4["iv_sigma_ratio"])
        - _as_float(g0["iv_sigma_ratio"]),
    }


def _concentration_rows(
    frame: pd.DataFrame,
    *,
    population: str,
) -> list[dict[str, object]]:
    weights = pd.to_numeric(frame["row_weight"], errors="raise")
    total_weight = float(weights.sum())
    rows: list[dict[str, object]] = []
    for concentration_type, values in (
        ("stock", frame["symbol"].astype(str)),
        (
            "month",
            pd.to_datetime(frame["session"], errors="raise").dt.to_period("M").astype(str),
        ),
    ):
        working = pd.DataFrame(
            {
                "group": values,
                "row_weight": weights,
                "iv_absolute_residual_15m": pd.to_numeric(
                    frame["iv_absolute_residual_15m"], errors="raise"
                ),
            }
        )
        for group, subset in working.groupby("group", sort=True, observed=True):
            weighted_rows = float(subset["row_weight"].sum())
            rows.append(
                {
                    "population": population,
                    "concentration_type": concentration_type,
                    "group": str(group),
                    "rows": int(len(subset)),
                    "weighted_rows": weighted_rows,
                    "row_share": float(len(subset) / len(frame)),
                    "weighted_share": weighted_rows / total_weight,
                    "mean_iv_residual": float(
                        np.average(
                            subset["iv_absolute_residual_15m"].to_numpy(float),
                            weights=subset["row_weight"].to_numpy(float),
                        )
                    ),
                }
            )
    return rows


def build_tail_tables(assessment_with_memberships: pd.DataFrame) -> TailTables:
    """Build the binding G4 tails plus the separately frozen G0 top-decile comparison."""

    tail_rows: list[dict[str, object]] = []
    for tail in ("top_decile", "top_quintile", "top_5pct", "top_2pct"):
        selected = assessment_with_memberships[f"G4_{tail}"].astype(bool)
        tail_rows.append(
            tail_metrics(
                assessment_with_memberships.loc[selected],
                model="G4",
                tail=tail,
            )
        )
    g0_selected = assessment_with_memberships["G0_top_decile"].astype(bool)
    tail_rows.append(
        tail_metrics(
            assessment_with_memberships.loc[g0_selected],
            model="G0",
            tail="top_decile",
        )
    )
    metrics = pd.DataFrame(tail_rows)
    indexed = metrics.set_index(["model", "tail"])
    g4 = cast(pd.Series, indexed.loc[("G4", "top_decile")])
    g0 = cast(pd.Series, indexed.loc[("G0", "top_decile")])
    comparison = pd.DataFrame(
        [
            {
                "comparison": "G4_top_decile-G0_top_decile",
                "mean_iv_residual_difference": _as_float(g4["mean_iv_residual"])
                - _as_float(g0["mean_iv_residual"]),
                "median_iv_residual_difference": _as_float(g4["median_iv_residual"])
                - _as_float(g0["median_iv_residual"]),
                "exceed_iv_rate_difference": _as_float(g4["exceed_iv_rate"])
                - _as_float(g0["exceed_iv_rate"]),
                "absolute_movement_difference": _as_float(g4["mean_absolute_movement"])
                - _as_float(g0["mean_absolute_movement"]),
                "iv_sigma_ratio_difference": _as_float(g4["iv_sigma_ratio"])
                - _as_float(g0["iv_sigma_ratio"]),
                "positive_residual_rate_difference": _as_float(g4["positive_residual_rate"])
                - _as_float(g0["positive_residual_rate"]),
                "top_5pct_contribution_difference": _as_float(
                    g4["top_5pct_positive_residual_contribution"]
                )
                - _as_float(g0["top_5pct_positive_residual_contribution"]),
                "G4_maximum_stock_share": _as_float(g4["maximum_stock_share"]),
                "G0_maximum_stock_share": _as_float(g0["maximum_stock_share"]),
                "G4_maximum_month_share": _as_float(g4["maximum_month_share"]),
                "G0_maximum_month_share": _as_float(g0["maximum_month_share"]),
            }
        ]
    )
    overlap = pd.DataFrame(
        [
            tail_overlap(
                assessment_with_memberships,
                "G0_top_decile",
                "G4_top_decile",
            )
        ]
    )
    captures = pd.DataFrame(
        [
            incremental_tail_capture(
                assessment_with_memberships,
                earlier_model=earlier,
                later_model=later,
            )
            for earlier, later in ADJACENT_COMPARISONS
        ]
    )
    concentration_rows = _concentration_rows(
        assessment_with_memberships,
        population="assessment",
    )
    concentration_rows.extend(
        _concentration_rows(
            assessment_with_memberships.loc[
                assessment_with_memberships["G4_top_decile"].astype(bool)
            ],
            population="G4_top_decile",
        )
    )
    concentration_rows.extend(
        _concentration_rows(
            assessment_with_memberships.loc[
                assessment_with_memberships["G0_top_decile"].astype(bool)
            ],
            population="G0_top_decile",
        )
    )
    return TailTables(
        metrics=metrics,
        comparison=comparison,
        overlap=overlap,
        incremental_capture=captures,
        concentration=pd.DataFrame(concentration_rows),
    )


def assessment_support(frame: pd.DataFrame) -> dict[str, object]:
    """Apply the pooled assessment support and weighted concentration gates."""

    weights = pd.to_numeric(frame["row_weight"], errors="raise")
    total = float(weights.sum())
    stock_share = (
        pd.DataFrame({"symbol": frame["symbol"].astype(str), "weight": weights})
        .groupby("symbol", sort=True, observed=True)["weight"]
        .sum()
        .div(total)
    )
    months = pd.to_datetime(frame["session"], errors="raise").dt.to_period("M").astype(str)
    month_share = (
        pd.DataFrame({"month": months, "weight": weights})
        .groupby("month", sort=True, observed=True)["weight"]
        .sum()
        .div(total)
    )
    result: dict[str, object] = {
        "rows": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "months": int(months.nunique()),
        "positive_outcomes": int(frame[TARGET_COLUMN].sum()),
        "base_rate": _weighted_mean(frame, TARGET_COLUMN),
        "maximum_weighted_stock_share": float(stock_share.max()),
        "maximum_weighted_month_share": float(month_share.max()),
    }
    result["passed"] = bool(
        len(frame) >= 8_000
        and frame["session"].nunique() >= 130
        and frame["symbol"].nunique() >= 15
        and months.nunique() == 8
        and int(frame[TARGET_COLUMN].sum()) >= 2_000
        and _as_float(result["maximum_weighted_stock_share"]) <= 0.12
        and _as_float(result["maximum_weighted_month_share"]) <= 0.20
    )
    return result


def top_decile_support(frame: pd.DataFrame) -> dict[str, object]:
    """Apply the binding unweighted G4 top-decile support gates."""

    months = pd.to_datetime(frame["session"], errors="raise").dt.to_period("M").astype(str)
    stock_share = frame["symbol"].astype(str).value_counts(normalize=True)
    month_share = months.value_counts(normalize=True)
    result: dict[str, object] = {
        "rows": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "months": int(months.nunique()),
        "maximum_stock_share": float(stock_share.max()),
        "maximum_month_share": float(month_share.max()),
    }
    result["passed"] = bool(
        len(frame) >= 700
        and frame["session"].nunique() >= 100
        and frame["symbol"].nunique() >= 15
        and months.nunique() >= 6
        and _as_float(result["maximum_stock_share"]) <= 0.15
        and _as_float(result["maximum_month_share"]) <= 0.25
    )
    return result


def _bootstrap_lower(
    bootstrap: pd.DataFrame,
    statistic: str,
    confidence: float = 0.80,
) -> float:
    selected = bootstrap.loc[
        bootstrap["statistic"].astype(str).eq(statistic)
        & np.isclose(bootstrap["confidence"].to_numpy(float), confidence)
    ]
    if len(selected) != 1:
        raise ValueError(f"bootstrap interval unavailable: {statistic}/{confidence}")
    return float(selected.iloc[0]["lower"])


def evaluate_group_gates(
    *,
    adjacent: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint_and_context: pd.DataFrame,
    null_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    support: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Apply the ten independent pass gates to D, I, R, and M."""

    adjacent_index = adjacent.set_index("comparison")
    output: dict[str, dict[str, object]] = {}
    for group in ("D", "I", "R", "M"):
        earlier, later = GROUP_MODEL_STEPS[group]
        comparison = f"{later}-{earlier}"
        point = cast(pd.Series, adjacent_index.loc[comparison])
        prefix = f"{later}_minus_{earlier}"
        monthly_rows = monthly.loc[monthly["comparison"].eq(comparison)]
        checkpoint_rows = checkpoint_and_context.loc[
            checkpoint_and_context["comparison"].eq(comparison)
            & checkpoint_and_context["scope"].eq("checkpoint_group")
        ]
        null_rows = null_metrics.loc[null_metrics["group"].eq(group)]
        positive_months = int(monthly_rows["log_loss_improvement"].gt(0.0).sum())
        materially_adverse = int(
            (
                checkpoint_rows["log_loss_improvement"].lt(-1e-12)
                | checkpoint_rows["brier_improvement"].lt(-1e-12)
            ).sum()
        )
        null_log_count = int(null_rows["real_exceeds_null_log_loss_improvement"].astype(bool).sum())
        null_brier_count = int(null_rows["real_exceeds_null_brier_improvement"].astype(bool).sum())
        gates: dict[str, object] = {
            "log_loss_improvement": _as_float(point["log_loss_improvement"]),
            "brier_improvement": _as_float(point["brier_improvement"]),
            "auc_improvement": _as_float(point["auc_improvement"]),
            "average_precision_improvement": _as_float(point["average_precision_improvement"]),
            "log_loss_improved": _as_float(point["log_loss_improvement"]) > 0.0,
            "brier_improved": _as_float(point["brier_improvement"]) > 0.0,
            "auc_not_reduced": _as_float(point["auc_improvement"]) >= 0.0,
            "average_precision_improved": (_as_float(point["average_precision_improvement"]) > 0.0),
            "bootstrap_80_log_loss_lower": _bootstrap_lower(
                bootstrap,
                f"{prefix}_log_loss_improvement",
            ),
            "bootstrap_80_brier_lower": _bootstrap_lower(
                bootstrap,
                f"{prefix}_brier_improvement",
            ),
            "bootstrap_80_average_precision_lower": _bootstrap_lower(
                bootstrap,
                f"{prefix}_average_precision_improvement",
            ),
            "positive_log_loss_months": positive_months,
            "months": int(len(monthly_rows)),
            "materially_adverse_checkpoint_groups": materially_adverse,
            "real_exceeds_null_log_loss_count": null_log_count,
            "real_exceeds_null_brier_count": null_brier_count,
            "support_and_concentration_pass": bool(support["passed"]),
        }
        gates["passed"] = bool(
            gates["log_loss_improved"]
            and gates["brier_improved"]
            and gates["auc_not_reduced"]
            and gates["average_precision_improved"]
            and _as_float(gates["bootstrap_80_log_loss_lower"]) >= 0.0
            and _as_float(gates["bootstrap_80_brier_lower"]) >= 0.0
            and _as_float(gates["bootstrap_80_average_precision_lower"]) >= 0.0
            and positive_months >= 5
            and materially_adverse == 0
            and (null_log_count == 3 or null_brier_count == 3)
            and bool(gates["support_and_concentration_pass"])
        )
        status = (
            "insufficient_support"
            if not bool(support["passed"])
            else ("supported" if gates["passed"] else "not_supported")
        )
        output[group] = {"status": status, "gates": gates}
    return output


def evaluate_tail_gate(
    *,
    assessment_with_memberships: pd.DataFrame,
    model_metrics: pd.DataFrame,
    tails: TailTables,
    bootstrap: pd.DataFrame,
) -> TailGateResult:
    """Apply the binding top-decile feasibility and G0-versus-G4 gates."""

    selected = assessment_with_memberships.loc[
        assessment_with_memberships["G4_top_decile"].astype(bool)
    ]
    support = top_decile_support(selected)
    indexed = tails.metrics.set_index(["model", "tail"])
    g4 = cast(pd.Series, indexed.loc[("G4", "top_decile")])
    comparison = tails.comparison.iloc[0]
    base_rate = _as_float(model_metrics.set_index("model").loc["G4", "base_rate"])
    months = pd.to_datetime(selected["session"], errors="raise").dt.to_period("M").astype(str)
    positive_months = 0
    for _, month_frame in selected.groupby(months, sort=True, observed=True):
        if _weighted_mean(month_frame, "iv_absolute_residual_15m") > 0.0:
            positive_months += 1
    mean_lower = _bootstrap_lower(
        bootstrap,
        "G4_top_decile_mean_iv_residual",
    )
    difference_lower = _bootstrap_lower(
        bootstrap,
        "G4_minus_G0_top_decile_mean_iv_residual_difference",
    )
    largest_rows_not_dominant = _as_float(g4["top_5pct_positive_residual_contribution"]) <= 0.50
    gates: dict[str, object] = {
        "mean_iv_residual": _as_float(g4["mean_iv_residual"]),
        "median_iv_residual": _as_float(g4["median_iv_residual"]),
        "exceed_iv_rate": _as_float(g4["exceed_iv_rate"]),
        "pooled_assessment_base_rate": base_rate,
        "bootstrap_80_mean_iv_residual_lower": mean_lower,
        "G4_minus_G0_mean_iv_residual_difference": _as_float(
            comparison["mean_iv_residual_difference"]
        ),
        "bootstrap_80_G4_minus_G0_mean_iv_residual_lower": difference_lower,
        "positive_mean_residual_months": positive_months,
        "maximum_stock_share": _as_float(g4["maximum_stock_share"]),
        "maximum_month_share": _as_float(g4["maximum_month_share"]),
        "top_5pct_positive_residual_contribution": _as_float(
            g4["top_5pct_positive_residual_contribution"]
        ),
        "largest_5pct_rows_not_dominant": largest_rows_not_dominant,
        "support": support,
    }
    gates["passed"] = bool(
        _as_float(gates["mean_iv_residual"]) > 0.0
        and _as_float(gates["median_iv_residual"]) > 0.0
        and _as_float(gates["exceed_iv_rate"]) > base_rate
        and mean_lower >= 0.0
        and _as_float(gates["G4_minus_G0_mean_iv_residual_difference"]) > 0.0
        and difference_lower >= 0.0
        and positive_months >= 5
        and largest_rows_not_dominant
        and bool(support["passed"])
    )
    final_status = (
        "insufficient_support"
        if not bool(support["passed"])
        else ("supported" if gates["passed"] else "not_supported")
    )
    options_status = (
        "insufficient_support"
        if not bool(support["passed"])
        else (
            "supported"
            if float(comparison["mean_iv_residual_difference"]) > 0.0 and difference_lower >= 0.0
            else "not_supported"
        )
    )
    return TailGateResult(
        final_status=final_status,
        options_only_vs_stock_status=options_status,
        gates=gates,
    )


def shared_session_bootstrap(
    result: LadderResult,
    assessment_with_memberships: pd.DataFrame,
) -> pd.DataFrame:
    """Build all required intervals from ten shared fixed-prediction session draws."""

    required_memberships = {"G0_top_decile", "G4_top_decile"}
    if missing := sorted(required_memberships.difference(assessment_with_memberships.columns)):
        raise ValueError(f"bootstrap tail memberships missing: {missing}")
    draws = session_bootstrap_multiplicities(
        assessment_with_memberships["session"],
        draws=10,
        seed=BOOTSTRAP_SEED,
    )
    draw_values: dict[str, list[float]] = {}

    def record(statistic: str, value: float) -> None:
        draw_values.setdefault(statistic, []).append(value)

    for multiplicity in draws:
        selected = multiplicity > 0
        boot = assessment_with_memberships.loc[selected].copy()
        boot["row_weight"] = boot["row_weight"].to_numpy(float) * multiplicity[selected].astype(
            float
        )
        for earlier, later in ADJACENT_COMPARISONS:
            increment = _comparison_increment(
                boot,
                earlier=earlier,
                later=later,
                thresholds=result.thresholds,
            )
            prefix = f"{later}_minus_{earlier}"
            for metric in (
                "log_loss_improvement",
                "brier_improvement",
                "auc_improvement",
                "average_precision_improvement",
                "top_decile_precision_improvement",
            ):
                record(f"{prefix}_{metric}", float(increment[metric]))
        try:
            g4_tail = tail_metrics(
                boot.loc[boot["G4_top_decile"].astype(bool)],
                model="G4",
                tail="top_decile",
            )
            g0_tail = tail_metrics(
                boot.loc[boot["G0_top_decile"].astype(bool)],
                model="G0",
                tail="top_decile",
            )
        except ValueError:
            for statistic in (
                "G4_top_decile_mean_iv_residual",
                "G4_top_decile_median_iv_residual",
                "G4_top_decile_exceed_iv_rate",
                "G4_minus_G0_top_decile_mean_iv_residual_difference",
                "G4_minus_G0_top_decile_median_iv_residual_difference",
                "G4_minus_G0_top_decile_exceed_iv_rate_difference",
                "G4_minus_G0_top_decile_iv_sigma_ratio_difference",
            ):
                record(statistic, math.nan)
            continue
        record("G4_top_decile_mean_iv_residual", _as_float(g4_tail["mean_iv_residual"]))
        record("G4_top_decile_median_iv_residual", _as_float(g4_tail["median_iv_residual"]))
        record("G4_top_decile_exceed_iv_rate", _as_float(g4_tail["exceed_iv_rate"]))
        record(
            "G4_minus_G0_top_decile_mean_iv_residual_difference",
            _as_float(g4_tail["mean_iv_residual"]) - _as_float(g0_tail["mean_iv_residual"]),
        )
        record(
            "G4_minus_G0_top_decile_median_iv_residual_difference",
            _as_float(g4_tail["median_iv_residual"]) - _as_float(g0_tail["median_iv_residual"]),
        )
        record(
            "G4_minus_G0_top_decile_exceed_iv_rate_difference",
            _as_float(g4_tail["exceed_iv_rate"]) - _as_float(g0_tail["exceed_iv_rate"]),
        )
        record(
            "G4_minus_G0_top_decile_iv_sigma_ratio_difference",
            _as_float(g4_tail["iv_sigma_ratio"]) - _as_float(g0_tail["iv_sigma_ratio"]),
        )

    point_increments = adjacent_increment_metrics(result.metrics).set_index("comparison")
    point_values: dict[str, float] = {}
    for earlier, later in ADJACENT_COMPARISONS:
        point = cast(pd.Series, point_increments.loc[f"{later}-{earlier}"])
        prefix = f"{later}_minus_{earlier}"
        for metric in (
            "log_loss_improvement",
            "brier_improvement",
            "auc_improvement",
            "average_precision_improvement",
            "top_decile_precision_improvement",
        ):
            point_values[f"{prefix}_{metric}"] = _as_float(point[metric])
    g4_point = tail_metrics(
        assessment_with_memberships.loc[assessment_with_memberships["G4_top_decile"].astype(bool)],
        model="G4",
        tail="top_decile",
    )
    point_values.update(
        {
            "G4_top_decile_mean_iv_residual": _as_float(g4_point["mean_iv_residual"]),
            "G4_top_decile_median_iv_residual": _as_float(g4_point["median_iv_residual"]),
            "G4_top_decile_exceed_iv_rate": _as_float(g4_point["exceed_iv_rate"]),
        }
    )
    comparison_point = _tail_comparison_values(assessment_with_memberships)
    for name, value in comparison_point.items():
        point_values[f"G4_minus_G0_top_decile_{name}"] = value
    rows: list[dict[str, object]] = []
    for statistic, values in draw_values.items():
        array = np.asarray(values, dtype=float)
        if np.isfinite(array).sum() == 0:
            raise ValueError(f"bootstrap statistic has no finite draws: {statistic}")
        for confidence in (0.80, 0.90, 0.95):
            alpha = (1.0 - confidence) / 2.0
            rows.append(
                {
                    "statistic": statistic,
                    "confidence": confidence,
                    "point_estimate": point_values[statistic],
                    "lower": float(np.nanquantile(array, alpha)),
                    "upper": float(np.nanquantile(array, 1.0 - alpha)),
                    "draws": 10,
                    "fixed_prediction": True,
                    "whole_session_resampling": True,
                    "shared_session_draws": True,
                    "coarse_quick_screen_diagnostic": True,
                }
            )
    return pd.DataFrame(rows)


def apply_tail_memberships(
    frame: pd.DataFrame,
    thresholds: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Apply model-specific development-frozen thresholds without assessment tuning."""

    output = frame.copy()
    for model, model_thresholds in thresholds.items():
        probability_column = f"{model}_probability"
        if probability_column not in output:
            raise ValueError(f"tail probability missing: {probability_column}")
        probabilities = pd.to_numeric(output[probability_column], errors="raise")
        for tail in ("top_decile", "top_quintile", "top_5pct", "top_2pct"):
            if tail not in model_thresholds:
                raise ValueError(f"tail threshold missing: {model}/{tail}")
            output[f"{model}_{tail}"] = probabilities.ge(float(model_thresholds[tail]))
    return output


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    """Return the deterministic midpoint-CDF weighted quantile used by V0.1."""

    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    valid = (
        0.0 <= quantile <= 1.0
        and value_array.ndim == weight_array.ndim == 1
        and len(value_array) > 0
        and len(value_array) == len(weight_array)
        and np.isfinite(value_array).all()
        and np.isfinite(weight_array).all()
        and bool((weight_array > 0.0).all())
    )
    if not valid:
        raise ValueError("weighted quantile inputs are invalid")
    order = np.argsort(value_array, kind="mergesort")
    ordered_values = value_array[order]
    ordered_weights = weight_array[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= ordered_weights.sum()
    return float(np.interp(quantile, cumulative, ordered_values))


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    if (
        len(values) == 0
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or bool((weights <= 0.0).any())
    ):
        raise ValueError(f"weighted mean inputs are invalid: {column}")
    return float(np.sum(weights * values) / np.sum(weights))


def tail_metrics(
    frame: pd.DataFrame,
    *,
    model: str,
    tail: str,
) -> dict[str, object]:
    """Summarise one frozen prediction tail using underlying-movement diagnostics."""

    required = {
        "symbol",
        "session",
        "row_weight",
        "absolute_log_return_15m",
        "iv_expected_absolute_15m",
        "iv_sigma_15m",
        "iv_absolute_residual_15m",
        TARGET_COLUMN,
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"tail metrics missing columns: {missing}")
    if frame.empty:
        raise ValueError("tail metrics require at least one row")
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    residual = pd.to_numeric(frame["iv_absolute_residual_15m"], errors="raise").to_numpy(float)
    movement = pd.to_numeric(frame["absolute_log_return_15m"], errors="raise").to_numpy(float)
    expectation = pd.to_numeric(frame["iv_expected_absolute_15m"], errors="raise").to_numpy(float)
    sigma = pd.to_numeric(frame["iv_sigma_15m"], errors="raise").to_numpy(float)
    positive = np.maximum(residual, 0.0)
    top_count = max(1, math.ceil(len(frame) * 0.05))
    top_indices = np.argsort(residual, kind="mergesort")[-top_count:]
    positive_total = float(np.sum(weights * positive))
    trim_count = math.floor(len(frame) * 0.10)
    residual_order = np.argsort(residual, kind="mergesort")
    kept = (
        residual_order[trim_count : len(frame) - trim_count]
        if trim_count > 0 and len(frame) - 2 * trim_count > 0
        else residual_order
    )
    stock_share = float(frame["symbol"].astype(str).value_counts(normalize=True).max())
    month = pd.to_datetime(frame["session"], errors="raise").dt.to_period("M").astype(str)
    month_share = float(month.value_counts(normalize=True).max())
    excursion_candidates = (
        "maximum_absolute_excursion_15m",
        "max_absolute_excursion_15m",
        "maximum_absolute_excursion",
    )
    excursion_column = next(
        (column for column in excursion_candidates if column in frame.columns),
        None,
    )
    total_weight = float(weights.sum())
    return {
        "model": model,
        "tail": tail,
        "rows": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "months": int(month.nunique()),
        "mean_absolute_movement": float(np.sum(weights * movement) / total_weight),
        "median_absolute_movement": weighted_quantile(
            movement.tolist(),
            weights.tolist(),
            0.5,
        ),
        "mean_iv_expectation": float(np.sum(weights * expectation) / total_weight),
        "mean_iv_residual": float(np.sum(weights * residual) / total_weight),
        "median_iv_residual": weighted_quantile(
            residual.tolist(),
            weights.tolist(),
            0.5,
        ),
        "exceed_iv_rate": _weighted_mean(frame, TARGET_COLUMN),
        "iv_sigma_ratio": float(np.sum(weights * movement) / np.sum(weights * sigma)),
        "mean_maximum_absolute_excursion": (
            math.nan if excursion_column is None else _weighted_mean(frame, excursion_column)
        ),
        "maximum_absolute_excursion_available": excursion_column is not None,
        "trimmed_10pct_mean_iv_residual": float(
            np.sum(weights[kept] * residual[kept]) / np.sum(weights[kept])
        ),
        "positive_residual_rate": float(
            np.sum(weights * (residual > 0.0).astype(float)) / total_weight
        ),
        "top_5pct_positive_residual_contribution": (
            float(np.sum(weights[top_indices] * positive[top_indices]) / positive_total)
            if positive_total > 0.0
            else math.nan
        ),
        "maximum_stock_share": stock_share,
        "maximum_month_share": month_share,
    }


def tail_overlap(
    frame: pd.DataFrame,
    first_membership: str,
    second_membership: str,
) -> dict[str, float | int]:
    """Report the binding G0/G4 top-tail set overlap."""

    first = frame[first_membership].astype(bool).to_numpy()
    second = frame[second_membership].astype(bool).to_numpy()
    intersection = int(np.sum(first & second))
    union = int(np.sum(first | second))
    return {
        "intersection_rows": intersection,
        "union_rows": union,
        "jaccard_overlap": (float(intersection / union) if union else math.nan),
        "G4_only_rows": int(np.sum(second & ~first)),
        "G0_only_rows": int(np.sum(first & ~second)),
    }


def incremental_tail_capture(
    frame: pd.DataFrame,
    *,
    earlier_model: str,
    later_model: str,
) -> dict[str, float | int | str]:
    """Describe target and residual rows entering and leaving an adjacent top decile."""

    earlier = frame[f"{earlier_model}_top_decile"].astype(bool)
    later = frame[f"{later_model}_top_decile"].astype(bool)
    entering = later & ~earlier
    leaving = earlier & ~later
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="raise")

    def subset_mean(mask: pd.Series) -> float:
        return (
            math.nan
            if not bool(mask.any())
            else _weighted_mean(frame.loc[mask], "iv_absolute_residual_15m")
        )

    entering_positive = int(target.loc[entering].sum())
    leaving_positive = int(target.loc[leaving].sum())
    return {
        "comparison": f"{later_model}-{earlier_model}",
        "entering_rows": int(entering.sum()),
        "leaving_rows": int(leaving.sum()),
        "new_positive_targets_entering_top_decile": entering_positive,
        "positive_targets_leaving_top_decile": leaving_positive,
        "net_change_captured_positive_targets": entering_positive - leaving_positive,
        "mean_iv_residual_entering_top_decile": subset_mean(entering),
        "mean_iv_residual_leaving_top_decile": subset_mean(leaving),
    }


def permute_group_within_slates(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    slate_columns: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Permute a complete selected group as one bundle within frozen stock slates."""

    required = {*columns, *slate_columns}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"group permutation columns missing: {missing}")
    if not columns:
        raise ValueError("group permutation requires at least one selected column")
    output = frame.copy()
    rng = np.random.default_rng(seed)
    grouped = frame.groupby(list(slate_columns), sort=True, observed=True).indices
    source_arrays = {column: frame[column].to_numpy(copy=True) for column in columns}
    output_arrays = {column: values.copy() for column, values in source_arrays.items()}
    for positions_value in grouped.values():
        positions = np.asarray(positions_value, dtype=int)
        source_positions = rng.permutation(positions)
        for column in columns:
            output_arrays[column][positions] = source_arrays[column][source_positions]
    for column in columns:
        output[column] = output_arrays[column]
    return output


def session_bootstrap_multiplicities(
    sessions: pd.Series,
    *,
    draws: int,
    seed: int,
) -> list[np.ndarray]:
    """Return exactly ten shared fixed-seed whole-session multiplicity vectors."""

    if draws != 10:
        raise ValueError("stock-layer bootstrap requires exactly 10 draws")
    labels = sessions.astype(str).to_numpy()
    unique = np.asarray(sorted(set(labels)), dtype=object)
    if unique.size == 0:
        raise ValueError("bootstrap sessions are empty")
    rng = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        output.append(np.asarray([int(counts.get(value, 0)) for value in labels], dtype=np.int64))
    return output


def validate_protected_boundary(frame: pd.DataFrame) -> dict[str, object]:
    """Reject protected observations and non-prior-close option chronology."""

    required = {"session", "stock_information_date", "options_observation_date"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"protected-boundary columns missing: {missing}")
    sessions = pd.to_datetime(frame["session"], errors="raise")
    stock_dates = pd.to_datetime(frame["stock_information_date"], errors="raise")
    option_dates = pd.to_datetime(frame["options_observation_date"], errors="raise")
    protected_market = int(
        sessions.ge(PROTECTED_START).sum() + stock_dates.ge(PROTECTED_START).sum()
    )
    protected_options = int(option_dates.ge(PROTECTED_START).sum())
    chronology_mismatches = int((~option_dates.lt(sessions)).sum())
    result = {
        **SAFETY_FLAGS,
        "protected_start": PROTECTED_START.date().isoformat(),
        "maximum_market_observation_date": str(max(sessions.max(), stock_dates.max()).date()),
        "maximum_option_observation_date": str(option_dates.max().date()),
        "protected_market_rows_materialised": protected_market,
        "protected_option_observations_materialised": protected_options,
        "same_or_future_option_observations": chronology_mismatches,
    }
    result["passed"] = bool(
        protected_market == 0 and protected_options == 0 and chronology_mismatches == 0
    )
    if not result["passed"]:
        raise ValueError(f"blocked_chronology_or_leakage_failure: {result}")
    return result


def choose_overall_decision(
    *,
    group_statuses: Mapping[str, str],
    final_tail_status: str,
    full_bundle_increment_reproduced: bool,
) -> str:
    """Map independent group and tail gates to one frozen decision category."""

    expected_groups = ("D", "I", "R", "M")
    if set(group_statuses) != set(expected_groups):
        raise ValueError("decision requires statuses for D, I, R, and M")
    supported = [group for group in expected_groups if group_statuses[group] == "supported"]
    if supported and final_tail_status != "supported":
        return "stock_layers_improve_ranking_but_not_positive_iv_tail"
    if final_tail_status == "supported":
        if len(supported) >= 2:
            return "multiple_stock_layers_contribute_to_iv_excess"
        if supported == ["D"]:
            return "daily_stock_context_drives_iv_excess_increment"
        if supported == ["I"]:
            return "intraday_compressed_transition_drives_iv_excess_increment"
        if supported == ["R"]:
            return "route_competition_drives_iv_excess_increment"
        if supported == ["M"]:
            return "cross_market_mismatch_adds_iv_excess_increment"
        return "positive_iv_excess_tail_without_localised_group"
    if full_bundle_increment_reproduced:
        return "stock_bundle_increment_not_reliably_localised"
    return "no_reproducible_group_increment"


__all__ = [
    "ANNUAL_TRADING_MINUTES",
    "ADJACENT_COMPARISONS",
    "CHECKPOINTS",
    "FEATURE_GROUPS",
    "FROZEN_COHORT",
    "GROUP_MODEL_STEPS",
    "GROUP_PERMUTATION_COLUMNS",
    "GROUP_D",
    "GROUP_I",
    "GROUP_M",
    "GROUP_O",
    "GROUP_R",
    "LadderResult",
    "MODEL_CONTROLS",
    "MODEL_FEATURES",
    "MODEL_ORDER",
    "NULL_SEEDS",
    "NullRefitResult",
    "PERMUTATION_SEEDS",
    "ROUTE_STATE_LEVELS",
    "SAFETY_FLAGS",
    "StabilityTables",
    "TARGET_COLUMN",
    "TailGateResult",
    "TailTables",
    "assessment_support",
    "apply_tail_memberships",
    "adjacent_increment_metrics",
    "assert_safety_flags",
    "build_stability_tables",
    "build_tail_tables",
    "calculate_iv_excess_outcomes",
    "choose_overall_decision",
    "development_prediction_thresholds",
    "evaluate_group_gates",
    "evaluate_tail_gate",
    "fit_model_ladder",
    "group_null_refits",
    "grouped_permutation_attribution",
    "incremental_tail_capture",
    "permute_group_within_slates",
    "reconstruct_frozen_branch_c_panel",
    "session_bootstrap_multiplicities",
    "shared_session_bootstrap",
    "tail_metrics",
    "tail_overlap",
    "top_decile_support",
    "validate_protected_boundary",
    "validate_feature_groups",
    "weighted_quantile",
]
