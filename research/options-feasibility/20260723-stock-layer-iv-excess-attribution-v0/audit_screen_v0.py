#!/usr/bin/env python3
"""Independently audit Stock-Layer Attribution and IV-Excess Tail Quick Screen V0."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
PREDECESSOR_PRIMARY = (
    REPO_ROOT
    / "research"
    / "cross-market-context"
    / "20260723-daily-stock-front-options-context-v01"
    / "artifacts"
    / "primary"
)
DAILY_PREDECESSOR_PRIMARY = (
    REPO_ROOT
    / "research"
    / "cross-market-context"
    / "20260723-daily-stock-options-regime-context-v0"
    / "artifacts"
    / "primary"
)
DENSE_PANEL = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-broad-conflict-advance-hazard-v02"
    / "artifacts"
    / "primary"
    / "dense_advance_panel.parquet"
)
EXPECTED_PANEL_SHA256 = "f62ef0144c12c813cbc665ba6d5ba1a235a6f77101a04b9f491c77b24c295529"
TARGET = "movement_exceeds_prior_close_iv_15m"
BOOTSTRAP_SEED = 20260763

for package_name in (
    "stocker_research",
    "stocker_data",
    "stocker_core",
    "stocker_backtest",
    "stocker_execution",
):
    package_path = str(REPO_ROOT / "packages" / package_name / "src")
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

from stocker_research.broad_conflict_advance_hazard_v02 import (  # noqa: E402
    DENSE_CHECKPOINTS,
    DENSE_H0_FEATURES,
    ROUTE_FEATURES,
)
from stocker_research.daily_soft_regimes_v0 import DAILY_STOCK_DIMENSIONS  # noqa: E402
from stocker_research.daily_stock_front_options_context_v01 import (  # noqa: E402
    FRONT_MISMATCH_FEATURES,
)
from stocker_research.front_options_soft_regimes_v01 import (  # noqa: E402
    FRONT_OPTIONS_DIMENSIONS,
    FRONT_OPTIONS_MISSING_INDICATORS,
)

SAFETY_FLAGS: dict[str, object] = {
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
CHECKPOINT_FEATURES = tuple(f"checkpoint_{value}" for value in DENSE_CHECKPOINTS)
GROUP_O = (
    *FRONT_OPTIONS_DIMENSIONS,
    *(f"front_options_regime_p_{value}" for value in range(4)),
    "front_options_regime_entropy",
    "front_options_regime_margin",
    *FRONT_OPTIONS_MISSING_INDICATORS,
    *CHECKPOINT_FEATURES,
)
GROUP_D = (
    *DAILY_STOCK_DIMENSIONS,
    *(f"daily_stock_regime_p_{value}" for value in range(4)),
    "daily_stock_regime_entropy",
    "daily_stock_regime_margin",
)
GROUP_I = tuple(value for value in DENSE_H0_FEATURES if value not in CHECKPOINT_FEATURES)
GROUP_R = tuple(ROUTE_FEATURES)
GROUP_M = tuple(FRONT_MISMATCH_FEATURES)
FRONT_CONTRACT_COLUMNS = (
    "front_expiration_date",
    "front_strike",
    "front_call_contract_id",
    "front_put_contract_id",
    "skew_put_contract_id",
    "skew_call_contract_id",
)
FEATURE_GROUPS = {
    "O": GROUP_O,
    "D": GROUP_D,
    "I": GROUP_I,
    "R": GROUP_R,
    "M": GROUP_M,
}
MODEL_FEATURES = {
    "G0": GROUP_O,
    "G1": (*GROUP_O, *GROUP_D),
    "G2": (*GROUP_O, *GROUP_D, *GROUP_I),
    "G3": (*GROUP_O, *GROUP_D, *GROUP_I, *GROUP_R),
    "G4": (*GROUP_O, *GROUP_D, *GROUP_I, *GROUP_R, *GROUP_M),
}
MODEL_CONTROLS = {
    "G0": ("stock",),
    "G1": ("stock",),
    "G2": ("stock",),
    "G3": ("stock", "route_state"),
    "G4": ("stock", "route_state"),
}
MODEL_GROUPS = {
    "G0": ("O",),
    "G1": ("O", "D"),
    "G2": ("O", "D", "I"),
    "G3": ("O", "D", "I", "R"),
    "G4": ("O", "D", "I", "R", "M"),
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_difference(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != right_values.shape:
        return math.inf
    both_nan = np.isnan(left_values) & np.isnan(right_values)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if bool((~both_nan & ~finite).any()):
        return math.inf
    return (
        0.0
        if not bool(finite.any())
        else float(np.max(np.abs(left_values[finite] - right_values[finite])))
    )


def manual_probability(frame: pd.DataFrame, specification: Mapping[str, Any]) -> np.ndarray:
    features = tuple(str(value) for value in specification["numeric_features"])
    raw = frame.loc[:, list(features)].to_numpy(float)
    medians = np.asarray(specification["numeric_medians"], dtype=float)
    means = np.asarray(specification["numeric_means"], dtype=float)
    scales = np.asarray(specification["numeric_scales"], dtype=float)
    values = np.where(np.isfinite(raw), raw, medians)
    parts = [np.asarray((values - means) / scales, dtype=np.float64)]
    controls = {
        "stock": frame["symbol"].astype(str),
        "route_state": frame["route_resolution_state"].astype(str),
    }
    levels = cast(Mapping[str, Sequence[object]], specification["category_levels"])
    for control_value in cast(Sequence[object], specification["category_controls"]):
        control = str(control_value)
        observed = controls[control].to_numpy()
        control_levels = tuple(str(value) for value in levels[control])
        for level in control_levels[1:]:
            parts.append(np.asarray(observed == level, dtype=np.float64)[:, None])
    design = np.concatenate(parts, axis=1)
    coefficients = np.asarray(specification["coefficients"], dtype=float)
    linear = design @ coefficients + float(specification["intercept"])
    return np.asarray(
        1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))),
        dtype=np.float64,
    )


def independently_fit_model(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    *,
    numeric_features: Sequence[str],
    category_controls: Sequence[str],
) -> dict[str, object]:
    """Independently rebuild preprocessing, coefficients, and assessment probabilities."""

    features = tuple(numeric_features)
    raw = development.loc[:, list(features)].to_numpy(float)
    finite = np.where(np.isfinite(raw), raw, np.nan)
    medians = np.nanmedian(finite, axis=0)
    values = np.where(np.isfinite(raw), raw, medians)
    means = np.asarray(values.mean(axis=0), dtype=np.float64)
    scales = np.asarray(values.std(axis=0, ddof=0), dtype=np.float64)
    scales = np.where(scales >= 1e-12, scales, 1.0)
    development_parts = [np.asarray((values - means) / scales, dtype=np.float64)]
    assessment_raw = assessment.loc[:, list(features)].to_numpy(float)
    assessment_values = np.where(np.isfinite(assessment_raw), assessment_raw, medians)
    assessment_parts = [np.asarray((assessment_values - means) / scales, dtype=np.float64)]
    development_controls = {
        "stock": development["symbol"].astype(str),
        "route_state": development["route_resolution_state"].astype(str),
    }
    assessment_controls = {
        "stock": assessment["symbol"].astype(str),
        "route_state": assessment["route_resolution_state"].astype(str),
    }
    levels: dict[str, tuple[str, ...]] = {}
    design_columns = list(features)
    for control in category_controls:
        observed = development_controls[control].to_numpy()
        control_levels = tuple(sorted(set(observed)))
        levels[control] = control_levels
        assessment_observed = assessment_controls[control].to_numpy()
        for level in control_levels[1:]:
            development_parts.append(np.asarray(observed == level, dtype=np.float64)[:, None])
            assessment_parts.append(
                np.asarray(assessment_observed == level, dtype=np.float64)[:, None]
            )
            design_columns.append(f"control_{control}__{level}")
    design = np.concatenate(development_parts, axis=1)
    assessment_design = np.concatenate(assessment_parts, axis=1)
    estimator = LogisticRegression(
        penalty="l2",
        C=0.25,
        solver="liblinear",
        max_iter=300,
        class_weight=None,
        random_state=20260722,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")
        warnings.filterwarnings("error", category=ConvergenceWarning)
        estimator.fit(
            design,
            development[TARGET].to_numpy(int),
            sample_weight=development["row_weight"].to_numpy(float),
        )
    return {
        "numeric_medians": np.asarray(medians, dtype=np.float64),
        "numeric_means": means,
        "numeric_scales": scales,
        "category_levels": levels,
        "design_columns": tuple(design_columns),
        "coefficients": np.asarray(estimator.coef_[0], dtype=np.float64),
        "intercept": float(estimator.intercept_[0]),
        "iterations": int(np.max(estimator.n_iter_)),
        "assessment_probabilities": np.asarray(
            estimator.predict_proba(assessment_design)[:, 1],
            dtype=np.float64,
        ),
    }


def binary_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    *,
    top_decile_threshold: float,
) -> dict[str, float]:
    selected = probabilities >= top_decile_threshold
    selected_weight = float(np.sum(weights[selected]))
    top_precision = (
        math.nan
        if selected_weight <= 0.0
        else float(np.sum(weights[selected] * target[selected]) / selected_weight)
    )
    return {
        "log_loss": float(log_loss(target, probabilities, sample_weight=weights, labels=[0, 1])),
        "brier_score": float(np.sum(weights * np.square(probabilities - target)) / weights.sum()),
        "auc": float(roc_auc_score(target, probabilities, sample_weight=weights)),
        "average_precision": float(
            average_precision_score(target, probabilities, sample_weight=weights)
        ),
        "top_decile_precision": top_precision,
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= ordered_weights.sum()
    return float(np.interp(quantile, cumulative, ordered_values))


def tail_values(frame: pd.DataFrame) -> dict[str, float]:
    weights = frame["row_weight"].to_numpy(float)
    residual = frame["iv_absolute_residual_15m"].to_numpy(float)
    movement = frame["absolute_log_return_15m"].to_numpy(float)
    sigma = frame["iv_sigma_15m"].to_numpy(float)
    target = frame[TARGET].to_numpy(float)
    return {
        "mean_iv_residual": float(np.sum(weights * residual) / weights.sum()),
        "median_iv_residual": weighted_quantile(residual, weights, 0.5),
        "exceed_iv_rate": float(np.sum(weights * target) / weights.sum()),
        "iv_sigma_ratio": float(np.sum(weights * movement) / np.sum(weights * sigma)),
    }


def permute_bundle(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    slate_columns: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    output = frame.copy()
    rng = np.random.default_rng(seed)
    grouped = frame.groupby(list(slate_columns), sort=True, observed=True).indices
    source = {column: frame[column].to_numpy(copy=True) for column in columns}
    permuted = {column: values.copy() for column, values in source.items()}
    for positions_value in grouped.values():
        positions = np.asarray(positions_value, dtype=int)
        source_positions = rng.permutation(positions)
        for column in columns:
            permuted[column][positions] = source[column][source_positions]
    for column in columns:
        output[column] = permuted[column]
    return output


def check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    evidence: Mapping[str, Any],
) -> None:
    checks.append({"check": name, "passed": bool(passed), "evidence": dict(evidence)})


def panel_layer_differences(panel: pd.DataFrame) -> dict[str, float | int]:
    dense = pd.read_parquet(DENSE_PANEL).drop_duplicates("row_id")
    dense_columns = ("row_weight", *GROUP_I, *CHECKPOINT_FEATURES, *GROUP_R)
    joined = panel.loc[:, ["row_id", "route_resolution_state", *dense_columns]].merge(
        dense.loc[:, ["row_id", "route_resolution_state", *dense_columns]],
        on="row_id",
        how="left",
        validate="one_to_one",
        suffixes=("_panel", "_source"),
        indicator=True,
    )
    dense_difference = max(
        maximum_difference(joined[f"{column}_panel"], joined[f"{column}_source"])
        for column in dense_columns
    )
    route_state_mismatches = int(
        joined["route_resolution_state_panel"]
        .astype(str)
        .ne(joined["route_resolution_state_source"].astype(str))
        .sum()
    )
    daily = pd.read_parquet(
        DAILY_PREDECESSOR_PRIMARY / "daily_stock_dimensions.parquet"
    ).drop_duplicates(["symbol", "session", "period"])
    daily_joined = panel.loc[:, ["symbol", "session", "period", *GROUP_D]].merge(
        daily.loc[:, ["symbol", "session", "period", *GROUP_D]],
        on=["symbol", "session", "period"],
        how="left",
        validate="many_to_one",
        suffixes=("_panel", "_source"),
    )
    daily_difference = max(
        maximum_difference(daily_joined[f"{column}_panel"], daily_joined[f"{column}_source"])
        for column in GROUP_D
    )
    front_features = GROUP_O[:16]
    front = pd.read_parquet(
        PREDECESSOR_PRIMARY / "front_options_dimensions.parquet"
    ).drop_duplicates(["symbol", "session", "period"])
    front_joined = panel.loc[
        :,
        ["symbol", "session", "period", *front_features],
    ].merge(
        front.loc[:, ["symbol", "session", "period", *front_features]],
        on=["symbol", "session", "period"],
        how="left",
        validate="many_to_one",
        suffixes=("_panel", "_source"),
    )
    front_difference = max(
        maximum_difference(front_joined[f"{column}_panel"], front_joined[f"{column}_source"])
        for column in front_features
    )
    raw_front = pd.read_parquet(
        PREDECESSOR_PRIMARY / "front_options_raw_features.parquet"
    ).drop_duplicates(["symbol", "session", "period"])
    contract_joined = (
        panel.loc[
            :,
            ["symbol", "session", "period", *FRONT_CONTRACT_COLUMNS],
        ]
        .drop_duplicates(["symbol", "session", "period"])
        .merge(
            raw_front.loc[
                :,
                ["symbol", "session", "period", *FRONT_CONTRACT_COLUMNS],
            ],
            on=["symbol", "session", "period"],
            how="left",
            validate="one_to_one",
            suffixes=("_panel", "_source"),
            indicator=True,
        )
    )
    selected_contract_mismatches = int(contract_joined["_merge"].ne("both").sum())
    selected_contract_mismatches += int(
        maximum_difference(
            contract_joined["front_strike_panel"],
            contract_joined["front_strike_source"],
        )
        > 1e-12
    )
    for column in FRONT_CONTRACT_COLUMNS:
        if column == "front_strike":
            continue
        selected_contract_mismatches += int(
            contract_joined[f"{column}_panel"]
            .astype(str)
            .ne(contract_joined[f"{column}_source"].astype(str))
            .sum()
        )

    development = panel.loc[panel["period"].astype(str).eq("development")]

    def standardize(column: str) -> pd.Series:
        mean = float(pd.to_numeric(development[column], errors="raise").mean())
        scale = float(pd.to_numeric(development[column], errors="raise").std(ddof=0))
        if not math.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        return (pd.to_numeric(panel[column], errors="raise") - mean) / scale

    tension = standardize("front_options_implied_tension")
    rebuilt_mismatch = pd.DataFrame(
        {
            "mismatch_compression_vs_front_iv": standardize("daily_compression") - tension,
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
                panel["route_resolution_state"].astype(str).eq("BROAD_CONFLICT").astype(float)
                * -tension
            ),
        }
    )
    mismatch_difference = max(
        maximum_difference(panel[column], rebuilt_mismatch[column]) for column in GROUP_M
    )
    return {
        "missing_dense_rows": int(joined["_merge"].ne("both").sum()),
        "maximum_dense_and_weight_difference": dense_difference,
        "route_state_mismatches": route_state_mismatches,
        "maximum_daily_difference": daily_difference,
        "maximum_front_difference": front_difference,
        "selected_contract_mismatches": selected_contract_mismatches,
        "maximum_mismatch_feature_difference": mismatch_difference,
    }


def bootstrap_reconstruction(
    assessment: pd.DataFrame,
    thresholds: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    labels = assessment["session"].astype(str).to_numpy()
    unique = np.asarray(sorted(set(labels)), dtype=object)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: dict[str, list[float]] = {}

    def record(name: str, value: float) -> None:
        values.setdefault(name, []).append(value)

    for _ in range(10):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        multiplicity = np.asarray([int(counts.get(value, 0)) for value in labels], dtype=int)
        selected = multiplicity > 0
        boot = assessment.loc[selected].copy()
        boot["row_weight"] = boot["row_weight"].to_numpy(float) * multiplicity[selected].astype(
            float
        )
        target = boot[TARGET].to_numpy(int)
        weights = boot["row_weight"].to_numpy(float)
        for earlier, later in (("G0", "G1"), ("G1", "G2"), ("G2", "G3"), ("G3", "G4")):
            earlier_metrics = binary_metrics(
                target,
                boot[f"{earlier}_probability"].to_numpy(float),
                weights,
                top_decile_threshold=float(thresholds[earlier]["top_decile"]),
            )
            later_metrics = binary_metrics(
                target,
                boot[f"{later}_probability"].to_numpy(float),
                weights,
                top_decile_threshold=float(thresholds[later]["top_decile"]),
            )
            prefix = f"{later}_minus_{earlier}"
            record(
                f"{prefix}_log_loss_improvement",
                earlier_metrics["log_loss"] - later_metrics["log_loss"],
            )
            record(
                f"{prefix}_brier_improvement",
                earlier_metrics["brier_score"] - later_metrics["brier_score"],
            )
            record(
                f"{prefix}_auc_improvement",
                later_metrics["auc"] - earlier_metrics["auc"],
            )
            record(
                f"{prefix}_average_precision_improvement",
                later_metrics["average_precision"] - earlier_metrics["average_precision"],
            )
            record(
                f"{prefix}_top_decile_precision_improvement",
                later_metrics["top_decile_precision"] - earlier_metrics["top_decile_precision"],
            )
        g4 = tail_values(boot.loc[boot["G4_top_decile"].astype(bool)])
        g0 = tail_values(boot.loc[boot["G0_top_decile"].astype(bool)])
        record("G4_top_decile_mean_iv_residual", g4["mean_iv_residual"])
        record("G4_top_decile_median_iv_residual", g4["median_iv_residual"])
        record("G4_top_decile_exceed_iv_rate", g4["exceed_iv_rate"])
        for name in (
            "mean_iv_residual",
            "median_iv_residual",
            "exceed_iv_rate",
            "iv_sigma_ratio",
        ):
            record(f"G4_minus_G0_top_decile_{name}_difference", g4[name] - g0[name])
    rows: list[dict[str, object]] = []
    for statistic, statistic_values in values.items():
        array = np.asarray(statistic_values, dtype=float)
        for confidence in (0.80, 0.90, 0.95):
            alpha = (1.0 - confidence) / 2.0
            rows.append(
                {
                    "statistic": statistic,
                    "confidence": confidence,
                    "lower": float(np.quantile(array, alpha)),
                    "upper": float(np.quantile(array, 1.0 - alpha)),
                }
            )
    return pd.DataFrame(rows)


def independently_rebuild_decision(
    assessment: pd.DataFrame,
    *,
    thresholds: Mapping[str, Mapping[str, float]],
    null_metrics: pd.DataFrame,
    rebuilt_bootstrap: pd.DataFrame,
) -> dict[str, object]:
    """Recompute every binding group and tail gate without stored decision booleans."""

    target = assessment[TARGET].to_numpy(int)
    weights = assessment["row_weight"].to_numpy(float)
    months = pd.to_datetime(assessment["session"], errors="raise").dt.to_period("M").astype(str)
    stock_weight_share = (
        pd.DataFrame({"symbol": assessment["symbol"].astype(str), "weight": weights})
        .groupby("symbol", sort=True, observed=True)["weight"]
        .sum()
        .div(weights.sum())
    )
    month_weight_share = (
        pd.DataFrame({"month": months, "weight": weights})
        .groupby("month", sort=True, observed=True)["weight"]
        .sum()
        .div(weights.sum())
    )
    support_passed = bool(
        len(assessment) >= 8_000
        and assessment["session"].nunique() >= 130
        and assessment["symbol"].nunique() >= 15
        and months.nunique() == 8
        and int(assessment[TARGET].sum()) >= 2_000
        and float(stock_weight_share.max()) <= 0.12
        and float(month_weight_share.max()) <= 0.20
    )

    def model_metrics(frame: pd.DataFrame, model: str) -> dict[str, float]:
        return binary_metrics(
            frame[TARGET].to_numpy(int),
            frame[f"{model}_probability"].to_numpy(float),
            frame["row_weight"].to_numpy(float),
            top_decile_threshold=float(thresholds[model]["top_decile"]),
        )

    def bootstrap_lower(statistic: str) -> float:
        row = rebuilt_bootstrap.loc[
            rebuilt_bootstrap["statistic"].astype(str).eq(statistic)
            & np.isclose(rebuilt_bootstrap["confidence"].to_numpy(float), 0.80)
        ]
        if len(row) != 1:
            raise AssertionError(f"missing independent bootstrap statistic: {statistic}")
        return float(row.iloc[0]["lower"])

    model_step = {
        "D": ("G0", "G1"),
        "I": ("G1", "G2"),
        "R": ("G2", "G3"),
        "M": ("G3", "G4"),
    }
    group_statuses: dict[str, str] = {}
    group_gate_evidence: dict[str, dict[str, object]] = {}
    for group, (earlier, later) in model_step.items():
        earlier_metrics = model_metrics(assessment, earlier)
        later_metrics = model_metrics(assessment, later)
        point = {
            "log_loss_improvement": earlier_metrics["log_loss"] - later_metrics["log_loss"],
            "brier_improvement": (earlier_metrics["brier_score"] - later_metrics["brier_score"]),
            "auc_improvement": later_metrics["auc"] - earlier_metrics["auc"],
            "average_precision_improvement": (
                later_metrics["average_precision"] - earlier_metrics["average_precision"]
            ),
        }
        positive_months = 0
        for _, frame in assessment.groupby(months, sort=True, observed=True):
            month_earlier = model_metrics(frame, earlier)
            month_later = model_metrics(frame, later)
            if month_earlier["log_loss"] - month_later["log_loss"] > 0.0:
                positive_months += 1
        adverse_checkpoint_groups = 0
        for lower, upper in ((6, 14), (16, 24), (26, 34)):
            frame = assessment.loc[assessment["checkpoint"].between(lower, upper)]
            checkpoint_earlier = model_metrics(frame, earlier)
            checkpoint_later = model_metrics(frame, later)
            if (
                checkpoint_earlier["log_loss"] - checkpoint_later["log_loss"] < -1e-12
                or checkpoint_earlier["brier_score"] - checkpoint_later["brier_score"] < -1e-12
            ):
                adverse_checkpoint_groups += 1
        group_nulls = null_metrics.loc[null_metrics["group"].astype(str).eq(group)]
        null_log_count = int(
            group_nulls["log_loss_improvement"].lt(point["log_loss_improvement"]).sum()
        )
        null_brier_count = int(
            group_nulls["brier_improvement"].lt(point["brier_improvement"]).sum()
        )
        prefix = f"{later}_minus_{earlier}"
        gate_values = {
            "log_loss_improved": point["log_loss_improvement"] > 0.0,
            "brier_improved": point["brier_improvement"] > 0.0,
            "auc_not_reduced": point["auc_improvement"] >= 0.0,
            "average_precision_improved": point["average_precision_improvement"] > 0.0,
            "bootstrap_80_log_loss_non_negative": (
                bootstrap_lower(f"{prefix}_log_loss_improvement") >= 0.0
            ),
            "bootstrap_80_brier_non_negative": (
                bootstrap_lower(f"{prefix}_brier_improvement") >= 0.0
            ),
            "bootstrap_80_average_precision_non_negative": (
                bootstrap_lower(f"{prefix}_average_precision_improvement") >= 0.0
            ),
            "positive_log_loss_in_at_least_five_months": positive_months >= 5,
            "no_materially_adverse_checkpoint_group": adverse_checkpoint_groups == 0,
            "real_log_loss_or_brier_exceeds_all_nulls": (
                null_log_count == 3 or null_brier_count == 3
            ),
            "support_and_concentration_pass": support_passed,
        }
        passed = bool(all(gate_values.values()))
        status = (
            "insufficient_support"
            if not support_passed
            else ("supported" if passed else "not_supported")
        )
        group_statuses[group] = status
        group_gate_evidence[group] = {
            "point": point,
            "positive_log_loss_months": positive_months,
            "materially_adverse_checkpoint_groups": adverse_checkpoint_groups,
            "real_exceeds_null_log_loss_count": null_log_count,
            "real_exceeds_null_brier_count": null_brier_count,
            "gate_values": gate_values,
            "passed": passed,
            "status": status,
        }

    g4_tail_frame = assessment.loc[assessment["G4_top_decile"].astype(bool)]
    g0_tail_frame = assessment.loc[assessment["G0_top_decile"].astype(bool)]
    g4_tail = tail_values(g4_tail_frame)
    g0_tail = tail_values(g0_tail_frame)
    g4_months = (
        pd.to_datetime(g4_tail_frame["session"], errors="raise").dt.to_period("M").astype(str)
    )
    positive_tail_months = sum(
        float(
            np.average(
                frame["iv_absolute_residual_15m"].to_numpy(float),
                weights=frame["row_weight"].to_numpy(float),
            )
        )
        > 0.0
        for _, frame in g4_tail_frame.groupby(g4_months, sort=True, observed=True)
    )
    tail_stock_share = float(g4_tail_frame["symbol"].astype(str).value_counts(normalize=True).max())
    tail_month_share = float(g4_months.value_counts(normalize=True).max())
    residual = g4_tail_frame["iv_absolute_residual_15m"].to_numpy(float)
    tail_weights = g4_tail_frame["row_weight"].to_numpy(float)
    positive_residual = np.maximum(residual, 0.0)
    top_count = max(1, math.ceil(len(g4_tail_frame) * 0.05))
    top_indices = np.argsort(residual, kind="mergesort")[-top_count:]
    positive_total = float(np.sum(tail_weights * positive_residual))
    top_contribution = float(
        np.sum(tail_weights[top_indices] * positive_residual[top_indices]) / positive_total
    )
    tail_support = bool(
        len(g4_tail_frame) >= 700
        and g4_tail_frame["session"].nunique() >= 100
        and g4_tail_frame["symbol"].nunique() >= 15
        and g4_months.nunique() >= 6
        and tail_stock_share <= 0.15
        and tail_month_share <= 0.25
    )
    mean_difference = g4_tail["mean_iv_residual"] - g0_tail["mean_iv_residual"]
    tail_gate_values = {
        "mean_iv_residual_positive": g4_tail["mean_iv_residual"] > 0.0,
        "median_iv_residual_positive": g4_tail["median_iv_residual"] > 0.0,
        "exceed_rate_above_base": (
            g4_tail["exceed_iv_rate"] > float(np.sum(weights * target) / weights.sum())
        ),
        "bootstrap_80_mean_iv_residual_non_negative": (
            bootstrap_lower("G4_top_decile_mean_iv_residual") >= 0.0
        ),
        "mean_residual_exceeds_G0_tail": mean_difference > 0.0,
        "bootstrap_80_mean_difference_non_negative": (
            bootstrap_lower("G4_minus_G0_top_decile_mean_iv_residual_difference") >= 0.0
        ),
        "positive_mean_residual_in_at_least_five_months": positive_tail_months >= 5,
        "not_dominated_by_stock_month_or_largest_rows": (
            tail_stock_share <= 0.15 and tail_month_share <= 0.25 and top_contribution <= 0.50
        ),
        "tail_support_pass": tail_support,
    }
    tail_passed = bool(all(tail_gate_values.values()))
    final_tail_status = (
        "insufficient_support"
        if not tail_support
        else ("supported" if tail_passed else "not_supported")
    )
    options_tail_status = (
        "insufficient_support"
        if not tail_support
        else (
            "supported"
            if mean_difference > 0.0
            and bootstrap_lower("G4_minus_G0_top_decile_mean_iv_residual_difference") >= 0.0
            else "not_supported"
        )
    )

    g0_metrics = model_metrics(assessment, "G0")
    g4_metrics = model_metrics(assessment, "G4")
    full_bundle_increment_reproduced = bool(
        g0_metrics["log_loss"] - g4_metrics["log_loss"] > 0.0
        and g0_metrics["brier_score"] - g4_metrics["brier_score"] > 0.0
        and g4_metrics["auc"] - g0_metrics["auc"] >= 0.0
        and g4_metrics["average_precision"] - g0_metrics["average_precision"] > 0.0
    )
    supported = [group for group in ("D", "I", "R", "M") if group_statuses[group] == "supported"]
    if supported and final_tail_status != "supported":
        overall = "stock_layers_improve_ranking_but_not_positive_iv_tail"
    elif final_tail_status == "supported":
        if len(supported) >= 2:
            overall = "multiple_stock_layers_contribute_to_iv_excess"
        elif supported == ["D"]:
            overall = "daily_stock_context_drives_iv_excess_increment"
        elif supported == ["I"]:
            overall = "intraday_compressed_transition_drives_iv_excess_increment"
        elif supported == ["R"]:
            overall = "route_competition_drives_iv_excess_increment"
        elif supported == ["M"]:
            overall = "cross_market_mismatch_adds_iv_excess_increment"
        else:
            overall = "positive_iv_excess_tail_without_localised_group"
    elif full_bundle_increment_reproduced:
        overall = "stock_bundle_increment_not_reliably_localised"
    else:
        overall = "no_reproducible_group_increment"
    if not support_passed:
        overall = "blocked_insufficient_support"
    return {
        "assessment_support_passed": support_passed,
        "group_statuses": group_statuses,
        "group_gate_evidence": group_gate_evidence,
        "tail_gate_values": tail_gate_values,
        "tail_passed": tail_passed,
        "final_tail_status": final_tail_status,
        "options_tail_status": options_tail_status,
        "overall_decision": overall,
        "full_bundle_increment_reproduced": full_bundle_increment_reproduced,
        "tail_evidence": {
            "positive_mean_residual_months": positive_tail_months,
            "maximum_stock_share": tail_stock_share,
            "maximum_month_share": tail_month_share,
            "top_5pct_positive_residual_contribution": top_contribution,
            "mean_iv_residual_difference": mean_difference,
        },
    }


def main() -> None:
    checks: list[dict[str, Any]] = []
    contract = read_json(PRIMARY / "contract.json")
    decision = read_json(PRIMARY / "decision.json")
    source = read_json(PRIMARY / "source_manifest.json")
    coefficients = read_json(PRIMARY / "model_coefficients.json")
    configurations = read_json(PRIMARY / "model_configurations.json")
    thresholds_artifact = read_json(PRIMARY / "tail_thresholds.json")
    thresholds = cast(
        Mapping[str, Mapping[str, float]],
        thresholds_artifact["prediction_thresholds"],
    )
    panel_path = Path(cast(Mapping[str, Any], source["sources"])["frozen_branch_c_panel"])
    panel = (
        pd.read_parquet(panel_path).sort_values("row_id", kind="mergesort").reset_index(drop=True)
    )
    assessment = (
        panel.loc[panel["period"].astype(str).eq("assessment")]
        .sort_values("row_id", kind="mergesort")
        .reset_index(drop=True)
    )
    development = (
        panel.loc[panel["period"].astype(str).eq("development")]
        .sort_values("row_id", kind="mergesort")
        .reset_index(drop=True)
    )
    predictions = (
        pd.read_parquet(PRIMARY / "assessment_predictions.parquet")
        .sort_values("row_id", kind="mergesort")
        .reset_index(drop=True)
    )

    safety_mismatches = {
        artifact: {
            key: (expected, value.get(key))
            for key, expected in SAFETY_FLAGS.items()
            if value.get(key) != expected
        }
        for artifact, value in (("contract", contract), ("decision", decision))
    }
    check(
        checks,
        "contract_and_decision_safety_flags",
        not any(safety_mismatches.values()),
        {"mismatches": safety_mismatches},
    )
    dates = cast(Mapping[str, Any], source["dates"])
    option_dates = pd.to_datetime(panel["options_observation_date"], errors="raise")
    sessions = pd.to_datetime(panel["session"], errors="raise")
    chronology_mismatches = int(
        option_dates.ne(pd.to_datetime(panel["required_options_date"], errors="raise")).sum()
        + (~option_dates.lt(sessions)).sum()
    )
    protected_market = int(sessions.ge(pd.Timestamp("2025-08-23")).sum())
    protected_options = int(option_dates.ge(pd.Timestamp("2025-08-23")).sum())
    check(
        checks,
        "dates_protected_boundary_and_exact_previous_close_chronology",
        dates
        == {
            "development_start": "2024-01-01",
            "development_end": "2024-12-31",
            "assessment_start": "2025-01-01",
            "assessment_end": "2025-08-22",
            "protected_start": "2025-08-23",
        }
        and chronology_mismatches == 0
        and protected_market == 0
        and protected_options == 0,
        {
            "dates": dates,
            "chronology_mismatches": chronology_mismatches,
            "protected_market_rows": protected_market,
            "protected_option_rows": protected_options,
        },
    )
    layer_differences = panel_layer_differences(panel)
    check(
        checks,
        "frozen_branch_c_panel_and_candidate_weights",
        sha256_file(panel_path) == EXPECTED_PANEL_SHA256
        and len(panel) == 24_130
        and len(assessment) == 10_265
        and all(float(value) <= 1e-12 for value in layer_differences.values()),
        {
            "panel_sha256": sha256_file(panel_path),
            "rows": len(panel),
            "assessment_rows": len(assessment),
            **layer_differences,
        },
    )
    expected = panel["atm_iv"].to_numpy(float) * math.sqrt(15.0 / (252.0 * 390.0))
    expected_absolute = expected * math.sqrt(2.0 / math.pi)
    movement = np.abs(
        np.log(panel["close_15m"].to_numpy(float) / panel["entry_price"].to_numpy(float))
    )
    target = (movement > expected_absolute).astype(int)
    outcome_difference = max(
        maximum_difference(panel["absolute_log_return_15m"], movement),
        maximum_difference(panel["iv_sigma_15m"], expected),
        maximum_difference(panel["iv_expected_absolute_15m"], expected_absolute),
        maximum_difference(
            panel["iv_absolute_residual_15m"],
            movement - expected_absolute,
        ),
    )
    target_mismatches = int(np.sum(panel[TARGET].to_numpy(int) != target))
    check(
        checks,
        "fifteen_minute_target_and_continuous_iv_residual",
        outcome_difference <= 1e-12 and target_mismatches == 0,
        {
            "maximum_outcome_difference": outcome_difference,
            "target_mismatches": target_mismatches,
        },
    )
    manifest = read_json(PRIMARY / "feature_group_manifest.json")
    manifest_groups = cast(Mapping[str, Mapping[str, Any]], manifest["groups"])
    observed_features = {
        group: tuple(str(value) for value in manifest_groups[group]["numeric_features"])
        for group in FEATURE_GROUPS
    }
    feature_union = [feature for features in observed_features.values() for feature in features]
    check(
        checks,
        "feature_group_membership_and_disjointness",
        observed_features == FEATURE_GROUPS
        and len(feature_union) == len(set(feature_union))
        and len(GROUP_D) == 14
        and len(GROUP_I) == 26
        and len(GROUP_R) == 15
        and len(GROUP_M) == 5,
        {
            "group_sizes": {group: len(features) for group, features in observed_features.items()},
            "duplicates": len(feature_union) - len(set(feature_union)),
        },
    )
    configuration_models = cast(Mapping[str, Mapping[str, Any]], configurations["models"])
    serialized_models = cast(Mapping[str, Mapping[str, Any]], coefficients["models"])
    configuration_mismatches = {
        model: {
            "features": tuple(configuration_models[model]["numeric_features"])
            != MODEL_FEATURES[model],
            "controls": tuple(configuration_models[model]["category_controls"])
            != MODEL_CONTROLS[model],
            "groups": tuple(configuration_models[model]["groups"]) != MODEL_GROUPS[model],
            "preprocessing": configuration_models[model]["preprocessing_fitted_period"]
            != "development_2024_only",
            "penalty": configuration_models[model]["penalty"] != "l2",
            "C": float(configuration_models[model]["C"]) != 0.25,
            "solver": configuration_models[model]["solver"] != "liblinear",
            "max_iter": int(configuration_models[model]["max_iter"]) != 300,
            "class_weight": configuration_models[model]["class_weight"] is not None,
            "n_jobs": int(configuration_models[model]["n_jobs"]) != 1,
            "target": configuration_models[model]["target"] != TARGET,
            "development_rows": int(configuration_models[model]["development_rows"])
            != len(development),
            "assessment_rows": int(configuration_models[model]["assessment_rows"])
            != len(assessment),
        }
        for model in MODEL_FEATURES
    }
    check(
        checks,
        "model_ladder_and_development_only_preprocessing",
        configurations["primary_logistic_models_fitted"] == 5
        and not any(any(value.values()) for value in configuration_mismatches.values()),
        {"mismatches": configuration_mismatches},
    )
    manual_differences: dict[str, float] = {}
    independent_fit_differences: dict[str, dict[str, float | int]] = {}
    development_threshold_differences: dict[str, float] = {}
    for model, specification in serialized_models.items():
        probabilities = manual_probability(assessment, specification)
        manual_differences[model] = maximum_difference(
            probabilities,
            predictions[f"{model}_probability"],
        )
        independent = independently_fit_model(
            development,
            assessment,
            numeric_features=MODEL_FEATURES[model],
            category_controls=MODEL_CONTROLS[model],
        )
        independent_levels = cast(Mapping[str, Sequence[object]], independent["category_levels"])
        serialized_levels = cast(Mapping[str, Sequence[object]], specification["category_levels"])
        independent_fit_differences[model] = {
            "numeric_median_difference": maximum_difference(
                independent["numeric_medians"],
                specification["numeric_medians"],
            ),
            "numeric_mean_difference": maximum_difference(
                independent["numeric_means"],
                specification["numeric_means"],
            ),
            "numeric_scale_difference": maximum_difference(
                independent["numeric_scales"],
                specification["numeric_scales"],
            ),
            "category_level_mismatches": int(
                {
                    key: tuple(str(value) for value in values)
                    for key, values in independent_levels.items()
                }
                != {
                    key: tuple(str(value) for value in values)
                    for key, values in serialized_levels.items()
                }
            ),
            "design_column_mismatches": int(
                tuple(cast(Sequence[object], independent["design_columns"]))
                != tuple(cast(Sequence[object], specification["design_columns"]))
            ),
            "coefficient_difference": maximum_difference(
                independent["coefficients"],
                specification["coefficients"],
            ),
            "intercept_difference": abs(
                float(independent["intercept"]) - float(specification["intercept"])
            ),
            "iteration_difference": abs(
                int(independent["iterations"]) - int(specification["iterations"])
            ),
            "assessment_probability_difference": maximum_difference(
                independent["assessment_probabilities"],
                predictions[f"{model}_probability"],
            ),
        }
        development_probabilities = manual_probability(development, specification)
        observed_thresholds = thresholds[model]
        development_threshold_differences[model] = max(
            abs(float(observed_thresholds[name]) - float(np.quantile(development_probabilities, q)))
            for name, q in (
                ("top_decile", 0.90),
                ("top_quintile", 0.80),
                ("top_5pct", 0.95),
                ("top_2pct", 0.98),
            )
        )
    check(
        checks,
        "all_model_coefficients_and_manual_probability_reconstruction",
        max(manual_differences.values()) <= 1e-12
        and all(
            all(float(value) <= 1e-12 for value in differences.values())
            for differences in independent_fit_differences.values()
        ),
        {
            "rows_per_model": len(assessment),
            "manual_probability_difference_by_model": manual_differences,
            "independent_fit_difference_by_model": independent_fit_differences,
            "models_independently_refitted": 5,
        },
    )
    check(
        checks,
        "development_frozen_tail_thresholds",
        max(development_threshold_differences.values()) <= 1e-12,
        {"maximum_difference_by_model": development_threshold_differences},
    )
    predecessor = pd.read_parquet(PREDECESSOR_PRIMARY / "assessment_predictions.parquet")
    predecessor = predecessor.loc[predecessor["C0_prediction"].notna()].sort_values(
        "row_id", kind="mergesort"
    )
    check(
        checks,
        "predecessor_g0_c0_and_g4_c1_equivalence",
        maximum_difference(predictions["G0_probability"], predecessor["C0_prediction"]) <= 1e-12
        and maximum_difference(predictions["G4_probability"], predecessor["C1_prediction"])
        <= 1e-12,
        {
            "G0_probability_difference": maximum_difference(
                predictions["G0_probability"],
                predecessor["C0_prediction"],
            ),
            "G4_probability_difference": maximum_difference(
                predictions["G4_probability"],
                predecessor["C1_prediction"],
            ),
        },
    )
    stored_metrics = pd.read_csv(PRIMARY / "grouped_model_metrics.csv").set_index("model")
    metric_differences: dict[str, float] = {}
    assessment_target = assessment[TARGET].to_numpy(int)
    assessment_weights = assessment["row_weight"].to_numpy(float)
    for model in MODEL_FEATURES:
        rebuilt = binary_metrics(
            assessment_target,
            predictions[f"{model}_probability"].to_numpy(float),
            assessment_weights,
            top_decile_threshold=float(thresholds[model]["top_decile"]),
        )
        metric_differences[model] = max(
            abs(float(stored_metrics.loc[model, name]) - value) for name, value in rebuilt.items()
        )
    check(
        checks,
        "log_loss_brier_auc_average_precision_and_top_decile",
        max(metric_differences.values()) <= 1e-12,
        {"maximum_difference_by_model": metric_differences},
    )
    membership_mismatches = {
        model: sum(
            int(
                predictions[f"{model}_{name}"]
                .astype(bool)
                .ne(predictions[f"{model}_probability"].ge(float(thresholds[model][name])))
                .sum()
            )
            for name in ("top_decile", "top_quintile", "top_5pct", "top_2pct")
        )
        for model in MODEL_FEATURES
    }
    check(
        checks,
        "g0_g4_and_all_model_tail_membership",
        sum(membership_mismatches.values()) == 0,
        {"mismatches_by_model": membership_mismatches},
    )
    stored_tail = pd.read_csv(PRIMARY / "tail_metrics.csv").set_index(["model", "tail"])
    tail_differences: dict[str, float] = {}
    for model, tail in (("G0", "top_decile"), ("G4", "top_decile")):
        rebuilt = tail_values(predictions.loc[predictions[f"{model}_{tail}"].astype(bool)])
        tail_differences[f"{model}_{tail}"] = max(
            abs(float(stored_tail.loc[(model, tail), name]) - value)
            for name, value in rebuilt.items()
        )
    check(
        checks,
        "tail_iv_residual_calculations",
        max(tail_differences.values()) <= 1e-12,
        {"maximum_difference_by_tail": tail_differences},
    )

    permutation_artifact = pd.read_csv(PRIMARY / "grouped_permutation_attribution.csv")
    g4_specification = serialized_models["G4"]
    base = binary_metrics(
        assessment_target,
        predictions["G4_probability"].to_numpy(float),
        assessment_weights,
        top_decile_threshold=float(thresholds["G4"]["top_decile"]),
    )
    permutation_differences: list[float] = []
    for row in permutation_artifact.itertuples(index=False):
        columns = str(row.bundle_columns).split("|")
        permuted = permute_bundle(
            assessment,
            columns=columns,
            slate_columns=("session", "checkpoint"),
            seed=int(row.seed),
        )
        probabilities = manual_probability(permuted, g4_specification)
        rebuilt = binary_metrics(
            assessment_target,
            probabilities,
            assessment_weights,
            top_decile_threshold=float(thresholds["G4"]["top_decile"]),
        )
        observed = {
            "log_loss_deterioration": rebuilt["log_loss"] - base["log_loss"],
            "brier_deterioration": rebuilt["brier_score"] - base["brier_score"],
            "auc_deterioration": base["auc"] - rebuilt["auc"],
            "average_precision_deterioration": (
                base["average_precision"] - rebuilt["average_precision"]
            ),
            "top_decile_precision_deterioration": (
                base["top_decile_precision"] - rebuilt["top_decile_precision"]
            ),
        }
        permutation_differences.append(
            max(abs(float(getattr(row, name)) - value) for name, value in observed.items())
        )
    check(
        checks,
        "grouped_frozen_model_permutations",
        len(permutation_artifact) == 20 and max(permutation_differences) <= 1e-12,
        {
            "rows": len(permutation_artifact),
            "maximum_metric_difference": max(permutation_differences),
        },
    )

    null_artifact = pd.read_csv(PRIMARY / "group_null_metrics.csv")
    null_models = cast(Mapping[str, Mapping[str, Any]], coefficients["null_models"])
    null_differences: list[float] = []
    null_boolean_mismatches = 0
    primary_probabilities = {
        model: manual_probability(assessment, serialized_models[model]) for model in MODEL_FEATURES
    }
    for row in null_artifact.itertuples(index=False):
        columns = str(row.permuted_columns).split("|")
        permuted = permute_bundle(
            panel,
            columns=columns,
            slate_columns=("period", "session", "checkpoint"),
            seed=int(row.seed),
        )
        permuted_assessment = (
            permuted.loc[permuted["period"].astype(str).eq("assessment")]
            .sort_values("row_id", kind="mergesort")
            .reset_index(drop=True)
        )
        later = str(row.comparison).split("-")[0]
        earlier = str(row.comparison).split("-")[1]
        later_probabilities = manual_probability(
            permuted_assessment,
            null_models[str(row.null_model_id)],
        )
        earlier_metrics = binary_metrics(
            assessment_target,
            primary_probabilities[earlier],
            assessment_weights,
            top_decile_threshold=float(thresholds[earlier]["top_decile"]),
        )
        later_metrics = binary_metrics(
            assessment_target,
            later_probabilities,
            assessment_weights,
            top_decile_threshold=float(thresholds[later]["top_decile"]),
        )
        rebuilt = {
            "log_loss_improvement": earlier_metrics["log_loss"] - later_metrics["log_loss"],
            "brier_improvement": (earlier_metrics["brier_score"] - later_metrics["brier_score"]),
            "auc_improvement": later_metrics["auc"] - earlier_metrics["auc"],
            "average_precision_improvement": (
                later_metrics["average_precision"] - earlier_metrics["average_precision"]
            ),
        }
        primary_later_metrics = binary_metrics(
            assessment_target,
            primary_probabilities[later],
            assessment_weights,
            top_decile_threshold=float(thresholds[later]["top_decile"]),
        )
        rebuilt_real = {
            "log_loss_improvement": (
                earlier_metrics["log_loss"] - primary_later_metrics["log_loss"]
            ),
            "brier_improvement": (
                earlier_metrics["brier_score"] - primary_later_metrics["brier_score"]
            ),
            "auc_improvement": primary_later_metrics["auc"] - earlier_metrics["auc"],
            "average_precision_improvement": (
                primary_later_metrics["average_precision"] - earlier_metrics["average_precision"]
            ),
        }
        null_differences.append(
            max(
                *(abs(float(getattr(row, name)) - value) for name, value in rebuilt.items()),
                *(
                    abs(float(getattr(row, f"real_{name}")) - value)
                    for name, value in rebuilt_real.items()
                ),
            )
        )
        null_boolean_mismatches += sum(
            bool(getattr(row, f"real_exceeds_null_{name}")) != (rebuilt_real[name] > rebuilt[name])
            for name in rebuilt
        )
    check(
        checks,
        "every_group_specific_null",
        len(null_artifact) == 12
        and len(null_models) == 12
        and max(null_differences) <= 1e-12
        and null_boolean_mismatches == 0,
        {
            "rows": len(null_artifact),
            "models": len(null_models),
            "maximum_metric_difference": max(null_differences),
            "comparison_boolean_mismatches": null_boolean_mismatches,
            "null_models_refitted_by_auditor": 0,
        },
    )

    rebuilt_bootstrap = bootstrap_reconstruction(predictions, thresholds)
    stored_bootstrap = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    bootstrap_joined = stored_bootstrap.merge(
        rebuilt_bootstrap,
        on=["statistic", "confidence"],
        how="outer",
        validate="one_to_one",
        suffixes=("_stored", "_rebuilt"),
        indicator=True,
    )
    bootstrap_difference = max(
        maximum_difference(bootstrap_joined["lower_stored"], bootstrap_joined["lower_rebuilt"]),
        maximum_difference(bootstrap_joined["upper_stored"], bootstrap_joined["upper_rebuilt"]),
    )
    check(
        checks,
        "shared_whole_session_bootstrap",
        bootstrap_joined["_merge"].eq("both").all() and bootstrap_difference <= 1e-12,
        {
            "artifact_rows": len(stored_bootstrap),
            "maximum_interval_difference": bootstrap_difference,
            "draws": 10,
            "models_refitted": 0,
        },
    )
    determinism = read_json(PRIMARY / "determinism_check.json")
    check(
        checks,
        "determinism",
        determinism["passed"] is True
        and determinism["joined_row_mismatches"] == 0
        and float(determinism["maximum_model_probability_difference"]) <= 1e-12
        and determinism["tail_membership_mismatches"] == 0
        and determinism["bootstrap_repeated"] is False
        and determinism["grouped_permutations_repeated"] is False
        and determinism["null_refits_repeated"] is False,
        determinism,
    )
    independent_decision = independently_rebuild_decision(
        predictions,
        thresholds=thresholds,
        null_metrics=null_artifact,
        rebuilt_bootstrap=rebuilt_bootstrap,
    )
    independent_group_statuses = cast(Mapping[str, str], independent_decision["group_statuses"])
    stored_group_statuses = {
        "D": str(decision["daily_stock_group_status"]),
        "I": str(decision["intraday_h0_group_status"]),
        "R": str(decision["route_competition_group_status"]),
        "M": str(decision["mismatch_group_status"]),
    }
    stored_group_gate_passes = {
        group: bool(decision["group_gates"][group]["gates"]["passed"])
        for group in ("D", "I", "R", "M")
    }
    independent_group_gate_evidence = cast(
        Mapping[str, Mapping[str, Any]],
        independent_decision["group_gate_evidence"],
    )
    independent_group_gate_passes = {
        group: bool(independent_group_gate_evidence[group]["passed"])
        for group in ("D", "I", "R", "M")
    }
    check(
        checks,
        "decision_logic",
        independent_group_statuses == stored_group_statuses
        and independent_group_gate_passes == stored_group_gate_passes
        and bool(independent_decision["tail_passed"])
        == bool(decision["final_tail_gates"]["passed"])
        and independent_decision["final_tail_status"] == decision["final_top_decile_status"]
        and independent_decision["options_tail_status"]
        == decision["options_only_vs_stock_tail_status"]
        and independent_decision["full_bundle_increment_reproduced"]
        == decision["full_bundle_increment_reproduced"]
        and independent_decision["overall_decision"] == decision["overall_decision"],
        {
            "independent_reconstruction": independent_decision,
            "stored_group_statuses": stored_group_statuses,
            "stored_group_gate_passes": stored_group_gate_passes,
            "stored_final_tail_status": decision["final_top_decile_status"],
            "stored_options_tail_status": decision["options_only_vs_stock_tail_status"],
            "stored_overall_decision": decision["overall_decision"],
        },
    )

    failed = [item["check"] for item in checks if not item["passed"]]
    audit: dict[str, Any] = {
        **SAFETY_FLAGS,
        "audit_kind": "independent_artifact_source_and_fixed_prediction_reconstruction",
        "checks": checks,
        "checks_run": len(checks),
        "checks_passed": len(checks) - len(failed),
        "checks_failed": len(failed),
        "failed_checks": failed,
        "models_manually_reconstructed": 5,
        "models_independently_refitted_for_coefficient_audit": 5,
        "manual_probability_rows_per_model": len(assessment),
        "null_models_refitted": 0,
        "provider_requests_made": 0,
        "passed": not failed,
    }
    write_json(PRIMARY / "independent_audit.json", audit)
    if audit["passed"]:
        decision["independent_audit_status"] = "passed"
    else:
        decision["independent_audit_status"] = "blocked"
        decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
        for key in (
            "daily_stock_group_status",
            "intraday_h0_group_status",
            "route_competition_group_status",
            "mismatch_group_status",
            "final_top_decile_status",
            "options_only_vs_stock_tail_status",
        ):
            decision[key] = "blocked"
    write_json(PRIMARY / "decision.json", decision)
    print("passed" if audit["passed"] else "blocked_reproducibility_or_audit_failure")
    if not audit["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
