#!/usr/bin/env python3
"""Independently audit the frozen Daily Stock + Front-Options V0.1 screen."""

from __future__ import annotations

import hashlib
import json
import math
import runpy
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
PREDECESSOR_PRIMARY = (
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
for package in ("stocker_research", "stocker_data"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package / "src"))

from stocker_research.broad_conflict_advance_hazard_v02 import (  # noqa: E402
    DENSE_CHECKPOINTS,
    DENSE_H0_FEATURES,
    ROUTE_FEATURES,
)
from stocker_research.daily_soft_regimes_v0 import (  # noqa: E402
    DAILY_STOCK_DIMENSIONS,
)
from stocker_research.daily_stock_options_context_v0 import (  # noqa: E402
    permute_bundle_within_slates,
    previous_us_trading_session,
    select_daily_options_surface,
)
from stocker_research.front_options_soft_regimes_v01 import (  # noqa: E402
    FRONT_OPTIONS_DIMENSIONS,
    FRONT_OPTIONS_MISSING_INDICATORS,
    FRONT_OPTIONS_RAW_FEATURES,
)
from stocker_research.stock_options_cross_market_quick_v0 import (  # noqa: E402
    reconstruct_clean_structural_panel,
)

SAFETY_FLAGS: dict[str, object] = {
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
FRONT_MISMATCH_FEATURES = (
    "mismatch_compression_vs_front_iv",
    "mismatch_daily_volatility_vs_front_iv",
    "mismatch_route_vs_front_premium",
    "mismatch_direction_agreement",
    "mismatch_complacent_broad_conflict",
)
CHECKPOINT_FEATURES = tuple(f"checkpoint_{value}" for value in DENSE_CHECKPOINTS)
H0_NON_CLOCK_FEATURES = tuple(
    value for value in DENSE_H0_FEATURES if value not in CHECKPOINT_FEATURES
)
STOCK_REGIME_FEATURES = (
    *(f"daily_stock_regime_p_{value}" for value in range(4)),
    "daily_stock_regime_entropy",
    "daily_stock_regime_margin",
)
FRONT_REGIME_FEATURES = (
    *(f"front_options_regime_p_{value}" for value in range(4)),
    "front_options_regime_entropy",
    "front_options_regime_margin",
)
STOCK_CONTEXT_FEATURES = (*DAILY_STOCK_DIMENSIONS, *STOCK_REGIME_FEATURES)
FRONT_CONTEXT_FEATURES = (
    *FRONT_OPTIONS_DIMENSIONS,
    *FRONT_REGIME_FEATURES,
    *FRONT_OPTIONS_MISSING_INDICATORS,
)
EXPECTED_MODEL_FEATURES = {
    "A0": (*DENSE_H0_FEATURES, *ROUTE_FEATURES),
    "A1": (*DENSE_H0_FEATURES, *ROUTE_FEATURES, *STOCK_CONTEXT_FEATURES),
    "B0": (*DENSE_H0_FEATURES, *ROUTE_FEATURES, *STOCK_CONTEXT_FEATURES),
    "B1": (
        *DENSE_H0_FEATURES,
        *ROUTE_FEATURES,
        *STOCK_CONTEXT_FEATURES,
        *FRONT_CONTEXT_FEATURES,
        *FRONT_MISMATCH_FEATURES,
    ),
    "C0": (*FRONT_CONTEXT_FEATURES, *CHECKPOINT_FEATURES),
    "C1": (
        *FRONT_CONTEXT_FEATURES,
        *CHECKPOINT_FEATURES,
        *STOCK_CONTEXT_FEATURES,
        *H0_NON_CLOCK_FEATURES,
        *ROUTE_FEATURES,
        *FRONT_MISMATCH_FEATURES,
    ),
}
EXPECTED_CONTROLS = {
    "A0": ("stock", "route_state"),
    "A1": ("stock", "route_state"),
    "B0": ("stock", "route_state"),
    "B1": ("stock", "route_state"),
    "C0": ("stock",),
    "C1": ("stock", "route_state"),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_difference(
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


def manual_diag_gmm(
    frame: pd.DataFrame,
    mapping: Mapping[str, Any],
    prefix: str,
) -> pd.DataFrame:
    columns = [str(value) for value in cast(Sequence[Any], mapping["input_columns"])]
    values = frame.loc[:, columns].to_numpy(float)
    medians = np.asarray(mapping["input_medians"], dtype=float)
    values = np.where(np.isfinite(values), values, medians)
    means = np.asarray(mapping["canonical_input_means"], dtype=float)
    covariance = np.asarray(mapping["canonical_covariances"], dtype=float)
    weights = np.asarray(mapping["canonical_weights"], dtype=float)
    width = means.shape[1]
    log_density = np.empty((len(frame), 4), dtype=float)
    for regime in range(4):
        variance = covariance[regime]
        delta = values - means[regime]
        log_density[:, regime] = math.log(weights[regime]) - 0.5 * (
            width * math.log(2.0 * math.pi)
            + np.log(variance).sum()
            + (np.square(delta) / variance).sum(axis=1)
        )
    log_density -= log_density.max(axis=1, keepdims=True)
    probability = np.exp(log_density)
    probability /= probability.sum(axis=1, keepdims=True)
    output = pd.DataFrame(index=frame.index)
    for regime in range(4):
        output[f"{prefix}_p_{regime}"] = probability[:, regime]
    clipped = np.clip(probability, 1e-15, 1.0)
    output[f"{prefix}_entropy"] = -np.sum(probability * np.log(clipped), axis=1)
    ordered = np.sort(probability, axis=1)
    output[f"{prefix}_top_probability"] = ordered[:, -1]
    output[f"{prefix}_margin"] = ordered[:, -1] - ordered[:, -2]
    output[prefix] = np.argmax(probability, axis=1)
    return output


def manual_daily_stock_dimensions(
    raw: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    scale_values = cast(Mapping[str, Mapping[str, Any]], manifest["scales"])

    def z(column: str, *, values: pd.Series | None = None) -> pd.Series:
        scale = scale_values[column]
        source = (
            pd.to_numeric(raw[column], errors="coerce")
            if values is None
            else pd.to_numeric(values, errors="coerce")
        )
        return (source - float(scale["center"])) / float(scale["scale"])

    output = raw.loc[:, ["symbol", "session", "period"]].copy()
    directional = 0.5 * (z("daily_efficiency_5") + z("daily_efficiency_10"))
    output["daily_compression"] = (
        -z("daily_range_5_to_20") - z("daily_rv_5_to_20") + z("daily_range_overlap_5")
    ) / 3.0
    output["daily_directional_efficiency"] = directional
    output["daily_trend_persistence"] = 0.5 * (
        z("daily_sign_persistence_5")
        + z(
            "abs_daily_extension_20",
            values=pd.to_numeric(raw["daily_extension_20"]).abs(),
        )
    )
    output["daily_extension"] = z("daily_extension_20")
    output["daily_rejection"] = 0.5 * (
        z("daily_extreme_wick_3") - z("daily_directional_efficiency", values=directional)
    )
    output["daily_volatility_acceleration"] = z("daily_rv_5_to_20")
    output["daily_relative_strength"] = z("daily_relative_return_5")
    output["daily_activity_acceleration"] = z("daily_activity_5_to_20")
    return output


def manual_front_dimensions(
    raw: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    medians = cast(Mapping[str, Any], manifest["imputation_medians"])
    scales = cast(Mapping[str, Mapping[str, Any]], manifest["scales"])
    imputed = raw.copy()
    for feature in FRONT_OPTIONS_RAW_FEATURES:
        imputed[feature] = (
            pd.to_numeric(imputed[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(float(medians[feature]))
        )

    def z(column: str, values: pd.Series | None = None) -> pd.Series:
        source = imputed[column] if values is None else values
        return (pd.to_numeric(source, errors="coerce") - float(scales[column]["center"])) / float(
            scales[column]["scale"]
        )

    output = raw.loc[
        :,
        [
            "symbol",
            "session",
            "period",
            *FRONT_OPTIONS_MISSING_INDICATORS,
        ],
    ].copy()
    z_atm = z("atm_iv")
    z_straddle = z("straddle_mid_pct")
    z_iv_rv = z("iv_minus_realised_20d")
    z_gap = z("call_put_iv_gap")
    z_skew = z("skew_25d")
    z_spread = z("combined_relative_spread")
    output["front_options_implied_tension"] = (z_atm + z_straddle + z_iv_rv) / 3.0
    output["front_options_premium_richness"] = (z_straddle + z_iv_rv) / 2.0
    output["front_options_downside_asymmetry"] = (z_skew - z_gap) / 2.0
    output["front_options_liquidity_stress"] = z_spread
    output["front_options_positioning_concentration"] = z("near_spot_oi_concentration")
    output["front_options_directional_positioning"] = z("call_put_oi_imbalance")
    output["front_options_surface_disagreement"] = (
        z("abs_call_put_iv_gap", imputed["call_put_iv_gap"].abs())
        + z("abs_skew_25d", imputed["skew_25d"].abs())
        + z_spread
    ) / 3.0
    return output


def manual_mismatches(
    panel: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    parameters = cast(Mapping[str, Mapping[str, Any]], manifest["standardization"])

    def z(column: str) -> pd.Series:
        return (
            pd.to_numeric(panel[column], errors="coerce") - float(parameters[column]["mean"])
        ) / float(parameters[column]["scale"])

    output = pd.DataFrame(index=panel.index)
    tension = z("front_options_implied_tension")
    output["mismatch_compression_vs_front_iv"] = z("daily_compression") - tension
    output["mismatch_daily_volatility_vs_front_iv"] = z("daily_volatility_acceleration") - tension
    output["mismatch_route_vs_front_premium"] = z("prefix_family_entropy") - z(
        "front_options_premium_richness"
    )
    output["mismatch_direction_agreement"] = z("signed_pressure") * z(
        "front_options_directional_positioning"
    )
    output["mismatch_complacent_broad_conflict"] = panel["BROAD_CONFLICT"].astype(int) * -tension
    return output


def manual_model_probability(
    frame: pd.DataFrame,
    specification: Mapping[str, Any],
) -> np.ndarray:
    features = [str(value) for value in cast(Sequence[Any], specification["numeric_features"])]
    raw = frame.loc[:, features].to_numpy(float)
    medians = np.asarray(specification["numeric_medians"], dtype=float)
    means = np.asarray(specification["numeric_means"], dtype=float)
    scales = np.asarray(specification["numeric_scales"], dtype=float)
    values = np.where(np.isfinite(raw), raw, medians)
    parts = [(values - means) / scales]
    controls = {
        "stock": frame["symbol"].astype(str),
        "route_state": frame["route_resolution_state"].astype(str),
    }
    levels = cast(Mapping[str, Sequence[Any]], specification["category_levels"])
    for control_value in cast(Sequence[Any], specification["category_controls"]):
        control = str(control_value)
        observed = controls[control].to_numpy()
        for level in [str(value) for value in levels[control]][1:]:
            parts.append((observed == level).astype(float)[:, None])
    design = np.concatenate(parts, axis=1)
    linear = design @ np.asarray(specification["coefficients"], dtype=float) + float(
        specification["intercept"]
    )
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0)))


def manual_binary_metrics(
    frame: pd.DataFrame,
    target: str,
    prediction: str,
) -> dict[str, float]:
    y = pd.to_numeric(frame[target], errors="raise").to_numpy(int)
    probability = pd.to_numeric(frame[prediction], errors="raise").to_numpy(float)
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    total = weights.sum()
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return {
        "log_loss": float(
            -np.sum(weights * (y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped))) / total
        ),
        "brier_score": float(np.sum(weights * np.square(probability - y)) / total),
        "auc": float(roc_auc_score(y, probability, sample_weight=weights)),
        "average_precision": float(average_precision_score(y, probability, sample_weight=weights)),
    }


def check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    branch: str = "global",
    evidence: Mapping[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "check": name,
            "branch": branch,
            "passed": bool(passed),
            "evidence": dict(evidence or {}),
        }
    )


def main() -> None:
    checks: list[dict[str, Any]] = []
    contract = read_json(PRIMARY / "contract.json")
    decision = read_json(PRIMARY / "decision.json")
    protected = read_json(PRIMARY / "protected_boundary_audit.json")
    reconstruction = read_json(PRIMARY / "structural_panel_reconstruction.json")
    stock_reconstruction = read_json(PRIMARY / "daily_stock_reconstruction.json")
    pair_reconstruction = read_json(PRIMARY / "front_options_pair_reconstruction.json")
    source = read_json(PRIMARY / "source_manifest.json")
    preflight = read_json(PRIMARY / "back_expiry_schema_preflight.json")
    configurations = read_json(PRIMARY / "model_configurations.json")
    coefficients = read_json(PRIMARY / "model_coefficients.json")
    determinism = read_json(PRIMARY / "determinism_check.json")
    for name, artifact in (
        ("contract", contract),
        ("decision", decision),
    ):
        mismatches = {
            key: (expected, artifact.get(key))
            for key, expected in SAFETY_FLAGS.items()
            if artifact.get(key) != expected
        }
        check(
            checks,
            f"{name}_safety_flags",
            not mismatches,
            evidence={"mismatches": mismatches},
        )
    check(
        checks,
        "dates_and_protected_boundary",
        bool(
            contract["development_start"] == "2024-01-01"
            and contract["development_end"] == "2024-12-31"
            and contract["assessment_start"] == "2025-01-01"
            and contract["assessment_end"] == "2025-08-22"
            and contract["protected_start"] == "2025-08-23"
            and protected["protected_market_rows_materialised"] == 0
            and protected["protected_option_observations_materialised"] == 0
            and protected["preflight_protected_option_observations_materialised"] == 0
            and protected["preflight_raw_response_persisted"] is False
            and preflight["protected_records_persisted"] == 0
            and preflight["raw_response_persisted"] is False
            and preflight["raw_response_cache_path"] is None
        ),
        evidence={
            **protected,
            "preflight_protected_records_returned": preflight["protected_records_returned"],
        },
    )

    dense = pd.read_parquet(DENSE_PANEL)
    manual_eligibility = dense["registered_completion_next_1_bar"].fillna(0).astype(int).eq(
        0
    ) & dense["any_prefix_one_transition_from_completion"].fillna(0).astype(int).eq(0)
    manual_target = (
        pd.to_numeric(dense["first_completion_lead"], errors="raise").isin([2, 3]).astype(int)
    )
    structural, package_reconstruction = reconstruct_clean_structural_panel(dense)
    reference = dense.loc[dense["advance_eligible"].astype(int).eq(1)]
    direct_eligibility_mismatches = int(
        manual_eligibility.ne(dense["advance_eligible"].astype(int).eq(1)).sum()
    )
    direct_target_mismatches = int(
        manual_target.loc[reference.index]
        .ne(reference["completion_in_bars_2_or_3"].astype(int))
        .sum()
    )
    check(
        checks,
        "structural_panel_and_clean_completion_target",
        bool(
            package_reconstruction["passed"]
            and direct_eligibility_mismatches == 0
            and direct_target_mismatches == 0
            and reconstruction["row_identity_mismatches"] == 0
            and reconstruction["route_state_mismatches"] == 0
            and reconstruction["target_mismatches"] == 0
            and reconstruction["maximum_shared_feature_difference"] <= 1e-12
        ),
        branch="A",
        evidence={
            "rows": len(structural),
            "direct_eligibility_mismatches": direct_eligibility_mismatches,
            "direct_target_mismatches": direct_target_mismatches,
        },
    )
    manual_weight = reference.copy()
    stock_counts = manual_weight.groupby(["period", "session"], observed=True)["symbol"].transform(
        "nunique"
    )
    row_counts = manual_weight.groupby(["period", "session", "symbol"], observed=True)[
        "symbol"
    ].transform("size")
    expected_weight = 1.0 / (stock_counts * row_counts)
    weight_difference = float(
        np.max(np.abs(reference["row_weight"].to_numpy(float) - expected_weight))
    )
    check(
        checks,
        "candidate_normalised_weights",
        weight_difference <= 1e-12,
        branch="A",
        evidence={"maximum_difference": weight_difference},
    )

    stock_raw = pd.read_parquet(PREDECESSOR_PRIMARY / "daily_stock_raw_features.parquet")
    stock_frozen = pd.read_parquet(
        PREDECESSOR_PRIMARY / "daily_stock_dimensions.parquet"
    ).sort_values(["symbol", "session"], kind="mergesort")
    stock_manifest = read_json(PREDECESSOR_PRIMARY / "daily_stock_feature_manifest.json")
    stock_mapping = read_json(PREDECESSOR_PRIMARY / "daily_stock_regime_mapping.json")
    stock_manual_all = manual_daily_stock_dimensions(stock_raw, stock_manifest)
    stock_manual = stock_manual_all.loc[
        stock_manual_all.loc[:, list(DAILY_STOCK_DIMENSIONS)].notna().all(axis=1)
    ].sort_values(["symbol", "session"], kind="mergesort")
    stock_dimension_difference = maximum_difference(
        stock_frozen,
        stock_manual,
        DAILY_STOCK_DIMENSIONS,
    )
    stock_posterior = manual_diag_gmm(
        stock_manual,
        stock_mapping,
        "daily_stock_regime",
    )
    stock_posterior_columns = (
        *(f"daily_stock_regime_p_{value}" for value in range(4)),
        "daily_stock_regime_entropy",
        "daily_stock_regime_top_probability",
        "daily_stock_regime_margin",
    )
    stock_posterior_difference = maximum_difference(
        stock_frozen,
        stock_posterior,
        stock_posterior_columns,
    )
    stock_hard_mismatches = int(
        np.sum(
            stock_frozen["daily_stock_regime"].to_numpy(int)
            != stock_posterior["daily_stock_regime"].to_numpy(int)
        )
    )
    check(
        checks,
        "daily_stock_context_and_frozen_regime_reconstruction",
        bool(
            stock_dimension_difference <= 1e-12
            and stock_posterior_difference <= 1e-12
            and stock_hard_mismatches == 0
            and stock_reconstruction["passed"]
        ),
        branch="A",
        evidence={
            "maximum_dimension_difference": stock_dimension_difference,
            "maximum_posterior_difference": stock_posterior_difference,
            "hard_regime_mismatches": stock_hard_mismatches,
        },
    )

    front_raw = pd.read_parquet(PRIMARY / "front_options_raw_features.parquet")
    front_dimensions = pd.read_parquet(PRIMARY / "front_options_dimensions.parquet").sort_values(
        ["symbol", "session"], kind="mergesort"
    )
    front_manifest = read_json(PRIMARY / "front_options_feature_manifest.json")
    front_mapping = read_json(PRIMARY / "front_options_regime_mapping.json")
    chronology = pd.read_csv(PRIMARY / "chronology_audit.csv")
    signal_dates = sorted(set(front_raw["session"].astype(str)))
    expected_previous = {
        signal: previous_us_trading_session(date.fromisoformat(signal)).isoformat()
        for signal in signal_dates
    }
    chronology_previous_mismatches = int(
        (
            front_raw["session"].astype(str).map(expected_previous)
            != front_raw["options_observation_date"].astype(str)
        ).sum()
    )
    check(
        checks,
        "exact_previous_session_options_chronology",
        bool(
            chronology_previous_mismatches == 0
            and chronology["chronology_passed"].astype(bool).all()
            and not chronology["same_day_or_future_options_used"].astype(bool).any()
        ),
        branch="B",
        evidence={
            "rows": len(front_raw),
            "previous_session_mismatches": chronology_previous_mismatches,
            "same_or_future_rows": int(
                chronology["same_day_or_future_options_used"].astype(bool).sum()
            ),
        },
    )
    check(
        checks,
        "front_raw_feature_surface",
        tuple(FRONT_OPTIONS_RAW_FEATURES)
        == (
            "atm_iv",
            "straddle_mid_pct",
            "call_put_iv_gap",
            "skew_25d",
            "combined_relative_spread",
            "iv_minus_realised_20d",
            "near_spot_oi_concentration",
            "call_put_oi_imbalance",
        )
        and not {
            "front_term_urgency",
            "back_atm_iv",
            "term_structure",
        }.intersection(front_raw.columns),
        branch="B",
    )

    cache_path = Path(
        cast(Mapping[str, Any], source["sources"])["repaired_exact_date_options_cache"]
    )
    cache = pd.read_parquet(
        cache_path,
        filters=[("trade_date", "<", date(2025, 8, 23))],
    )
    cache["trade_date"] = pd.to_datetime(cache["trade_date"]).dt.date
    cache_groups = {
        (str(symbol), trade_date): group
        for (symbol, trade_date), group in cache.groupby(
            ["underlying_symbol", "trade_date"], observed=True, sort=False
        )
    }
    stock_lookup = stock_raw.set_index(["symbol", "session"])
    pair_mismatches = 0
    pair_feature_difference = 0.0
    for row in front_raw.itertuples(index=False):
        observation = date.fromisoformat(str(row.options_observation_date))
        chain = cache_groups[(str(row.symbol), observation)]
        stock_row = stock_lookup.loc[(str(row.symbol), str(row.session))]
        selected = select_daily_options_surface(
            chain,
            previous_close=float(stock_row["unadjusted_close"]),
            realised_volatility_20d=float(stock_row["realised_volatility_20d"]),
        )
        pair_mismatches += int(
            selected["front_call_contract_id"] != row.front_call_contract_id
            or selected["front_put_contract_id"] != row.front_put_contract_id
            or selected["front_expiration_date"] != row.front_expiration_date
            or float(cast(Any, selected["front_strike"])) != float(row.front_strike)
            or selected["skew_put_contract_id"] != row.skew_put_contract_id
            or selected["skew_call_contract_id"] != row.skew_call_contract_id
        )
        for feature in FRONT_OPTIONS_RAW_FEATURES:
            expected = float(cast(Any, selected[feature]))
            observed = float(cast(Any, getattr(row, feature)))
            if math.isnan(expected) and math.isnan(observed):
                continue
            pair_feature_difference = max(pair_feature_difference, abs(expected - observed))
    check(
        checks,
        "front_atm_pair_and_raw_feature_reconstruction",
        bool(
            pair_mismatches == 0
            and pair_feature_difference <= 1e-12
            and pair_reconstruction["selection_rebuilt_from_cached_chains"] is True
            and pair_reconstruction["selected_contract_mismatches"] == 0
            and pair_reconstruction["maximum_front_raw_feature_difference"] <= 1e-12
            and pair_reconstruction["selected_pairs"] == len(front_raw)
        ),
        branch="B",
        evidence={
            "rows": len(front_raw),
            "selected_contract_mismatches": pair_mismatches,
            "maximum_raw_feature_difference": pair_feature_difference,
            "protected_cache_rows_loaded": int(
                pd.Series(cache["trade_date"]).ge(date(2025, 8, 23)).sum()
            ),
        },
    )

    front_manual = manual_front_dimensions(front_raw, front_manifest).sort_values(
        ["symbol", "session"], kind="mergesort"
    )
    front_dimension_difference = maximum_difference(
        front_dimensions,
        front_manual,
        FRONT_OPTIONS_DIMENSIONS,
    )
    front_posterior = manual_diag_gmm(
        front_manual,
        front_mapping,
        "front_options_regime",
    )
    front_posterior_columns = (
        *(f"front_options_regime_p_{value}" for value in range(4)),
        "front_options_regime_entropy",
        "front_options_regime_top_probability",
        "front_options_regime_margin",
    )
    front_posterior_difference = maximum_difference(
        front_dimensions,
        front_posterior,
        front_posterior_columns,
    )
    front_hard_mismatches = int(
        np.sum(
            front_dimensions["front_options_regime"].to_numpy(int)
            != front_posterior["front_options_regime"].to_numpy(int)
        )
    )
    canonical_dimensions = [
        str(value) for value in cast(Sequence[Any], front_mapping["canonical_dimensions"])
    ]
    ordering_keys = [
        tuple(float(cast(Mapping[str, Any], centroid)[name]) for name in canonical_dimensions)
        for centroid in cast(Sequence[Mapping[str, Any]], front_mapping["canonical_centroids"])
    ]
    check(
        checks,
        "seven_front_dimensions_and_canonical_front_regime",
        bool(
            front_dimension_difference <= 1e-12
            and front_posterior_difference <= 1e-12
            and front_hard_mismatches == 0
            and ordering_keys == sorted(ordering_keys)
            and front_mapping["fitted_period"] == "development_2024_only"
        ),
        branch="B",
        evidence={
            "maximum_dimension_difference": front_dimension_difference,
            "maximum_posterior_difference": front_posterior_difference,
            "hard_regime_mismatches": front_hard_mismatches,
        },
    )

    panel = pd.read_parquet(PRIMARY / "front_options_cross_market_panel.parquet").sort_values(
        "row_id", kind="mergesort"
    )
    mismatch_manifest = read_json(PRIMARY / "mismatch_feature_manifest.json")
    manual_mismatch = manual_mismatches(panel, mismatch_manifest)
    mismatch_difference = maximum_difference(
        panel,
        manual_mismatch,
        FRONT_MISMATCH_FEATURES,
    )
    check(
        checks,
        "five_cross_market_mismatch_features",
        mismatch_difference <= 1e-12 and len(FRONT_MISMATCH_FEATURES) == 5,
        branch="B",
        evidence={"maximum_difference": mismatch_difference},
    )
    movement = panel.loc[panel["entry_price"].notna()].copy()
    expected_movement = np.abs(
        np.log(movement["close_15m"].to_numpy(float) / movement["entry_price"].to_numpy(float))
    )
    expected_iv = (
        movement["atm_iv"].to_numpy(float)
        * math.sqrt(15.0 / (252.0 * 390.0))
        * math.sqrt(2.0 / math.pi)
    )
    movement_difference = float(
        np.max(np.abs(movement["absolute_log_return_15m"].to_numpy(float) - expected_movement))
    )
    iv_difference = float(
        np.max(np.abs(movement["iv_expected_absolute_15m"].to_numpy(float) - expected_iv))
    )
    target_mismatches = int(
        np.sum(
            movement["movement_exceeds_prior_close_iv_15m"].to_numpy(int)
            != (expected_movement > expected_iv).astype(int)
        )
    )
    check(
        checks,
        "fifteen_minute_iv_excess_target",
        movement_difference <= 1e-12 and iv_difference <= 1e-12 and target_mismatches == 0,
        branch="C",
        evidence={
            "rows": len(movement),
            "movement_difference": movement_difference,
            "iv_expectation_difference": iv_difference,
            "target_mismatches": target_mismatches,
        },
    )

    model_configs = cast(Mapping[str, Mapping[str, Any]], configurations["models"])
    feature_mismatches: dict[str, Any] = {}
    for model_id, expected in EXPECTED_MODEL_FEATURES.items():
        observed = tuple(
            str(value) for value in cast(Sequence[Any], model_configs[model_id]["numeric_features"])
        )
        controls = tuple(
            str(value)
            for value in cast(Sequence[Any], model_configs[model_id]["category_controls"])
        )
        if observed != expected or controls != EXPECTED_CONTROLS[model_id]:
            feature_mismatches[model_id] = {
                "features_match": observed == expected,
                "controls_match": controls == EXPECTED_CONTROLS[model_id],
            }
    check(
        checks,
        "a0_a1_b0_b1_c0_c1_feature_surfaces",
        not feature_mismatches and int(configurations["primary_classifier_fits"]) == 6,
        evidence={"mismatches": feature_mismatches},
    )

    predictions = pd.read_parquet(PRIMARY / "assessment_predictions.parquet")
    stock_context = stock_frozen
    branch_a_panel = structural.merge(
        stock_context,
        on=["symbol", "session", "period"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_stock"),
    )
    branch_a_assessment = branch_a_panel.loc[branch_a_panel["period"].eq("assessment")].merge(
        predictions.loc[
            predictions["A0_prediction"].notna(),
            ["row_id", "A0_prediction", "A1_prediction"],
        ],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    joined_assessment = panel.loc[panel["period"].eq("assessment")].merge(
        predictions.loc[
            predictions["B0_prediction"].notna(),
            [
                "row_id",
                "B0_prediction",
                "B1_prediction",
                "C0_prediction",
                "C1_prediction",
            ],
        ],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    coefficient_values = cast(Mapping[str, Mapping[str, Any]], coefficients["models"])
    probability_differences: dict[str, float] = {}
    for model_id in ("A0", "A1", "B0", "B1", "C0", "C1"):
        source_frame = branch_a_assessment if model_id.startswith("A") else joined_assessment
        sample = source_frame.sort_values("row_id", kind="mergesort").head(100)
        manual = manual_model_probability(
            sample,
            coefficient_values[model_id],
        )
        probability_differences[model_id] = float(
            np.max(np.abs(manual - sample[f"{model_id}_prediction"].to_numpy(float)))
        )
    check(
        checks,
        "model_coefficients_and_manual_probability_reconstruction",
        all(value <= 1e-12 for value in probability_differences.values()),
        evidence={
            "rows_per_model": 100,
            "maximum_difference_by_model": probability_differences,
        },
    )

    metric_differences: dict[str, dict[str, float]] = {}
    for branch, frame, target in (
        (
            "a",
            branch_a_assessment,
            "registered_completion_clean_bars_2_or_3",
        ),
        (
            "b",
            joined_assessment,
            "registered_completion_clean_bars_2_or_3",
        ),
        (
            "c",
            joined_assessment,
            "movement_exceeds_prior_close_iv_15m",
        ),
    ):
        artifact = pd.read_csv(PRIMARY / f"branch_{branch}_metrics.csv").set_index("model")
        models = (
            ("A0", "A1") if branch == "a" else (("B0", "B1") if branch == "b" else ("C0", "C1"))
        )
        for model_id in models:
            manual = manual_binary_metrics(
                frame,
                target,
                f"{model_id}_prediction",
            )
            metric_differences[model_id] = {
                name: abs(float(artifact.loc[model_id, name]) - value)
                for name, value in manual.items()
            }
    check(
        checks,
        "log_loss_brier_auc_and_average_precision",
        all(
            difference <= 1e-12
            for values in metric_differences.values()
            for difference in values.values()
        ),
        evidence={"maximum_differences": metric_differences},
    )

    bootstrap = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    unique_sessions = np.asarray(
        sorted(
            set(branch_a_assessment["session"].astype(str))
            | set(joined_assessment["session"].astype(str))
        ),
        dtype=object,
    )
    rng = np.random.default_rng(20260723)
    bootstrap_values: dict[str, list[float]] = {
        f"{prefix}_{metric}_improvement": []
        for prefix in ("A1_minus_A0", "B1_minus_B0", "C1_minus_C0")
        for metric in ("log_loss", "brier", "auc", "average_precision")
    }
    bootstrap_values.update(
        {
            "BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_mean_iv_residual": [],
            "BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_median_iv_residual": [],
            "BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_exceed_iv_rate": [],
        }
    )

    def resampled(frame: pd.DataFrame, counts: Mapping[str, int]) -> pd.DataFrame:
        output = frame.copy()
        output["row_weight"] = output["row_weight"].to_numpy(float) * output["session"].astype(
            str
        ).map(counts).fillna(0).to_numpy(float)
        return output.loc[output["row_weight"].gt(0.0)]

    def append_increment(
        *,
        prefix: str,
        frame: pd.DataFrame,
        target: str,
        old_model: str,
        new_model: str,
    ) -> None:
        old = manual_binary_metrics(frame, target, f"{old_model}_prediction")
        new = manual_binary_metrics(frame, target, f"{new_model}_prediction")
        bootstrap_values[f"{prefix}_log_loss_improvement"].append(old["log_loss"] - new["log_loss"])
        bootstrap_values[f"{prefix}_brier_improvement"].append(
            old["brier_score"] - new["brier_score"]
        )
        bootstrap_values[f"{prefix}_auc_improvement"].append(new["auc"] - old["auc"])
        bootstrap_values[f"{prefix}_average_precision_improvement"].append(
            new["average_precision"] - old["average_precision"]
        )

    def weighted_mean(frame: pd.DataFrame, column: str) -> float:
        weights = frame["row_weight"].to_numpy(float)
        values = frame[column].to_numpy(float)
        return float(np.sum(weights * values) / np.sum(weights))

    def weighted_median(frame: pd.DataFrame, column: str) -> float:
        values = frame[column].to_numpy(float)
        weights = frame["row_weight"].to_numpy(float)
        order = np.argsort(values, kind="mergesort")
        ordered_values = values[order]
        ordered_weights = weights[order]
        cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
        cumulative /= ordered_weights.sum()
        return float(np.interp(0.5, cumulative, ordered_values))

    for _ in range(10):
        sampled = rng.choice(unique_sessions, size=len(unique_sessions), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        weighted_a = resampled(branch_a_assessment, counts)
        weighted_joined = resampled(joined_assessment, counts)
        append_increment(
            prefix="A1_minus_A0",
            frame=weighted_a,
            target="registered_completion_clean_bars_2_or_3",
            old_model="A0",
            new_model="A1",
        )
        append_increment(
            prefix="B1_minus_B0",
            frame=weighted_joined,
            target="registered_completion_clean_bars_2_or_3",
            old_model="B0",
            new_model="B1",
        )
        append_increment(
            prefix="C1_minus_C0",
            frame=weighted_joined,
            target="movement_exceeds_prior_close_iv_15m",
            old_model="C0",
            new_model="C1",
        )
        broad = weighted_joined.loc[weighted_joined["BROAD_CONFLICT"].eq(1)]
        low = weighted_joined.loc[weighted_joined["LOW_ROUTE_SUPPORT"].eq(1)]
        bootstrap_values["BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_mean_iv_residual"].append(
            weighted_mean(broad, "iv_absolute_residual_15m")
            - weighted_mean(low, "iv_absolute_residual_15m")
        )
        bootstrap_values["BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_median_iv_residual"].append(
            weighted_median(broad, "iv_absolute_residual_15m")
            - weighted_median(low, "iv_absolute_residual_15m")
        )
        bootstrap_values["BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_exceed_iv_rate"].append(
            weighted_mean(broad, "movement_exceeds_prior_close_iv_15m")
            - weighted_mean(low, "movement_exceeds_prior_close_iv_15m")
        )
    bootstrap_difference = 0.0
    for statistic, values in bootstrap_values.items():
        for confidence in (0.80, 0.90, 0.95):
            tail = (1.0 - confidence) / 2.0
            row = bootstrap.loc[
                bootstrap["statistic"].eq(statistic)
                & np.isclose(bootstrap["confidence"], confidence)
            ].iloc[0]
            bootstrap_difference = max(
                bootstrap_difference,
                abs(float(row["lower"]) - float(np.quantile(values, tail))),
                abs(float(row["upper"]) - float(np.quantile(values, 1.0 - tail))),
            )
    check(
        checks,
        "shared_whole_session_bootstrap",
        bool(
            len(bootstrap) == 45
            and set(bootstrap["draws"].astype(int)) == {10}
            and bootstrap["fixed_prediction"].astype(bool).all()
            and bootstrap["whole_session_resampling"].astype(bool).all()
            and bootstrap_difference <= 1e-12
        ),
        evidence={
            "artifact_rows": len(bootstrap),
            "reconstructed_interval_maximum_difference": bootstrap_difference,
        },
    )

    front_null = pd.read_csv(PRIMARY / "front_options_null_metrics.csv")
    stock_null = pd.read_csv(PRIMARY / "stock_structure_null_metrics.csv")
    front_real = manual_binary_metrics(
        joined_assessment,
        "registered_completion_clean_bars_2_or_3",
        "B0_prediction",
    )
    front_added = manual_binary_metrics(
        joined_assessment,
        "registered_completion_clean_bars_2_or_3",
        "B1_prediction",
    )
    stock_real = manual_binary_metrics(
        joined_assessment,
        "movement_exceeds_prior_close_iv_15m",
        "C0_prediction",
    )
    stock_added = manual_binary_metrics(
        joined_assessment,
        "movement_exceeds_prior_close_iv_15m",
        "C1_prediction",
    )
    real_front_increment = {
        "log_loss_improvement": front_real["log_loss"] - front_added["log_loss"],
        "brier_improvement": front_real["brier_score"] - front_added["brier_score"],
        "auc_improvement": front_added["auc"] - front_real["auc"],
        "average_precision_improvement": front_added["average_precision"]
        - front_real["average_precision"],
    }
    real_stock_increment = {
        "log_loss_improvement": stock_real["log_loss"] - stock_added["log_loss"],
        "brier_improvement": stock_real["brier_score"] - stock_added["brier_score"],
        "auc_improvement": stock_added["auc"] - stock_real["auc"],
        "average_precision_improvement": stock_added["average_precision"]
        - stock_real["average_precision"],
    }

    def null_flags_match(
        table: pd.DataFrame,
        real: Mapping[str, float],
    ) -> bool:
        return all(
            bool(row[f"real_exceeds_null_{metric}"]) == (real[metric] > float(row[metric]))
            for row in table.to_dict(orient="records")
            for metric in real
        )

    subset_sessions = sorted(set(panel["session"].astype(str)))[:3]
    permutation_subset = panel.loc[panel["session"].astype(str).isin(subset_sessions)].copy()
    front_bundle = [
        column
        for column in (
            *FRONT_OPTIONS_RAW_FEATURES,
            *FRONT_CONTEXT_FEATURES,
            "front_options_regime",
            "options_observation_date",
            "front_call_contract_id",
            "front_put_contract_id",
        )
        if column in permutation_subset
    ]
    stock_bundle = [
        column
        for column in (
            *STOCK_CONTEXT_FEATURES,
            *H0_NON_CLOCK_FEATURES,
            *ROUTE_FEATURES,
            "route_resolution_state",
        )
        if column in permutation_subset
    ]

    def bundle_preserved(columns: Sequence[str], seeds: Sequence[int]) -> bool:
        for seed in seeds:
            permuted = permute_bundle_within_slates(
                permutation_subset,
                columns=columns,
                seed=seed,
            )
            if not permuted["row_id"].equals(permutation_subset["row_id"]):
                return False
            if not permuted["row_weight"].equals(permutation_subset["row_weight"]):
                return False
            for _key, original in permutation_subset.groupby(
                ["period", "session", "checkpoint"],
                sort=True,
                observed=True,
            ):
                changed = permuted.loc[original.index]
                original_hashes = sorted(
                    pd.util.hash_pandas_object(original.loc[:, list(columns)], index=False).tolist()
                )
                changed_hashes = sorted(
                    pd.util.hash_pandas_object(changed.loc[:, list(columns)], index=False).tolist()
                )
                if original_hashes != changed_hashes:
                    return False
        return True

    null_passed = bool(
        len(front_null) == 3
        and front_null["seed"].astype(int).tolist() == [20260723, 20260724, 20260725]
        and len(stock_null) == 3
        and stock_null["seed"].astype(int).tolist() == [20260726, 20260727, 20260728]
        and null_flags_match(front_null, real_front_increment)
        and null_flags_match(stock_null, real_stock_increment)
        and bundle_preserved(front_bundle, [20260723, 20260724, 20260725])
        and bundle_preserved(stock_bundle, [20260726, 20260727, 20260728])
    )
    check(
        checks,
        "front_options_and_stock_structure_null_designs",
        null_passed,
        evidence={
            "front_null_refits": len(front_null),
            "stock_null_refits": len(stock_null),
            "null_models_refitted_by_auditor": 0,
        },
    )

    preflight_parameters = cast(Mapping[str, Any], preflight["parameters"])
    preflight_passed = bool(
        preflight["status"] == "supported_noncompact_schema"
        and preflight["request_count"] == 1
        and preflight["record_count"] <= 100
        and preflight_parameters["compact"] == 0
        and preflight_parameters["page[limit]"] <= 100
        and "api_token" not in preflight_parameters
        and preflight["credential_recorded"] is False
        and preflight["canonical_cache_modified"] is False
        and preflight["raw_response_persisted"] is False
        and preflight["protected_records_persisted"] == 0
        and preflight["exact_date_filtering_possible"] is True
        and preflight["back_expiry_dte_records"] > 0
        and cast(Mapping[str, Any], source["back_expiry_preflight"])["canonical_records_admitted"]
        == 0
    )
    check(
        checks,
        "single_request_back_expiry_preflight_isolation",
        preflight_passed,
        evidence={
            "status": preflight["status"],
            "request_count": preflight["request_count"],
            "record_count": preflight["record_count"],
            "raw_response_sha256": preflight["raw_response_sha256"],
        },
    )
    check(
        checks,
        "branch_isolation",
        bool(
            len(branch_a_assessment) > len(joined_assessment)
            and configurations["models"]["A0"]["population"] == "full_structural_stock_context"
            and configurations["models"]["B0"]["population"] == "joined_front_options"
            and configurations["models"]["C0"]["population"]
            == "joined_front_options_with_15m_outcome"
        ),
        evidence={
            "branch_a_assessment_rows": len(branch_a_assessment),
            "joined_assessment_rows": len(joined_assessment),
        },
    )
    check(
        checks,
        "determinism",
        bool(
            determinism["passed"]
            and determinism["selected_contract_mismatches"] == 0
            and (
                determinism["pair_reconstruction_applicable"] is False
                or determinism["pair_selection_rebuilt_from_cached_chains"] is True
            )
            and determinism["maximum_front_raw_feature_difference"] <= 1e-12
            and determinism["joined_row_mismatches"] == 0
            and determinism["front_options_regime_mapping_mismatches"] == 0
            and determinism["maximum_feature_difference"] <= 1e-12
            and determinism["maximum_probability_difference"] <= 1e-12
            and determinism["bootstrap_repeated"] is False
            and determinism["null_draws_repeated"] is False
            and determinism["options_redownloaded"] is False
        ),
        evidence=determinism,
    )

    def metric_increment(
        old: Mapping[str, float],
        new: Mapping[str, float],
    ) -> dict[str, float]:
        return {
            "log_loss_improvement": old["log_loss"] - new["log_loss"],
            "brier_improvement": old["brier_score"] - new["brier_score"],
            "auc_improvement": new["auc"] - old["auc"],
            "average_precision_improvement": (new["average_precision"] - old["average_precision"]),
        }

    def positive_months(
        frame: pd.DataFrame,
        *,
        target: str,
        old_model: str,
        new_model: str,
    ) -> int:
        count = 0
        months = pd.to_datetime(frame["session"]).dt.to_period("M")
        for month in sorted(set(months.astype(str))):
            group = frame.loc[months.astype(str).eq(month)]
            old = manual_binary_metrics(group, target, f"{old_model}_prediction")
            new = manual_binary_metrics(group, target, f"{new_model}_prediction")
            count += int(old["log_loss"] - new["log_loss"] > 0.0)
        return count

    def independent_increment_gate(
        *,
        prefix: str,
        frame: pd.DataFrame,
        target: str,
        old_model: str,
        new_model: str,
        null_table: pd.DataFrame | None,
        inspect_checkpoints: bool,
    ) -> dict[str, Any]:
        old = manual_binary_metrics(frame, target, f"{old_model}_prediction")
        new = manual_binary_metrics(frame, target, f"{new_model}_prediction")
        increment = metric_increment(old, new)
        lower_log_loss = float(
            np.quantile(bootstrap_values[f"{prefix}_log_loss_improvement"], 0.10)
        )
        lower_brier = float(np.quantile(bootstrap_values[f"{prefix}_brier_improvement"], 0.10))
        month_count = positive_months(
            frame,
            target=target,
            old_model=old_model,
            new_model=new_model,
        )
        gate: dict[str, Any] = {
            **increment,
            "log_loss_improved": increment["log_loss_improvement"] > 0.0,
            "brier_improved": increment["brier_improvement"] > 0.0,
            "auc_not_reduced": increment["auc_improvement"] >= 0.0,
            "average_precision_improved": (increment["average_precision_improvement"] > 0.0),
            "bootstrap_80_log_loss_lower": lower_log_loss,
            "bootstrap_80_brier_lower": lower_brier,
            "bootstrap_80_proper_score_lowers_non_negative": (
                lower_log_loss >= 0.0 and lower_brier >= 0.0
            ),
            "positive_assessment_months": month_count,
            "positive_in_at_least_four_months": month_count >= 4,
        }
        if inspect_checkpoints:
            adverse = 0
            checkpoint_groups = (
                tuple(range(6, 15, 2)),
                tuple(range(16, 25, 2)),
                tuple(range(26, 35, 2)),
            )
            for checkpoints in checkpoint_groups:
                group = frame.loc[frame["checkpoint"].astype(int).isin(checkpoints)]
                old_group = manual_binary_metrics(
                    group,
                    target,
                    f"{old_model}_prediction",
                )
                new_group = manual_binary_metrics(
                    group,
                    target,
                    f"{new_model}_prediction",
                )
                adverse += int(
                    old_group["log_loss"] - new_group["log_loss"] < -1e-12
                    or old_group["brier_score"] - new_group["brier_score"] < -1e-12
                )
            gate["materially_adverse_checkpoint_groups"] = adverse
            gate["no_checkpoint_group_materially_adverse"] = adverse == 0
        if null_table is not None:
            gate["proper_score_increment_exceeds_all_three_nulls"] = bool(
                len(null_table) == 3
                and all(
                    increment["log_loss_improvement"] > float(row["log_loss_improvement"])
                    and increment["brier_improvement"] > float(row["brier_improvement"])
                    for row in null_table.to_dict(orient="records")
                )
            )
        required = (
            "log_loss_improved",
            "brier_improved",
            "auc_not_reduced",
            "average_precision_improved",
            "bootstrap_80_proper_score_lowers_non_negative",
            "positive_in_at_least_four_months",
        )
        gate["passed"] = all(bool(gate[name]) for name in required)
        if inspect_checkpoints:
            gate["passed"] = bool(gate["passed"] and gate["no_checkpoint_group_materially_adverse"])
        if null_table is not None:
            gate["passed"] = bool(
                gate["passed"] and gate["proper_score_increment_exceeds_all_three_nulls"]
            )
        return gate

    independent_gates: dict[str, dict[str, Any]] = {
        "daily_stock_context": independent_increment_gate(
            prefix="A1_minus_A0",
            frame=branch_a_assessment,
            target="registered_completion_clean_bars_2_or_3",
            old_model="A0",
            new_model="A1",
            null_table=None,
            inspect_checkpoints=True,
        ),
        "front_options_completion": independent_increment_gate(
            prefix="B1_minus_B0",
            frame=joined_assessment,
            target="registered_completion_clean_bars_2_or_3",
            old_model="B0",
            new_model="B1",
            null_table=front_null,
            inspect_checkpoints=False,
        ),
        "stock_to_iv_excess": independent_increment_gate(
            prefix="C1_minus_C0",
            frame=joined_assessment,
            target="movement_exceeds_prior_close_iv_15m",
            old_model="C0",
            new_model="C1",
            null_table=stock_null,
            inspect_checkpoints=False,
        ),
    }
    broad = joined_assessment.loc[joined_assessment["BROAD_CONFLICT"].eq(1)]
    low = joined_assessment.loc[joined_assessment["LOW_ROUTE_SUPPORT"].eq(1)]
    broad_gate: dict[str, Any] = {
        "broad_rows": len(broad),
        "low_route_support_rows": len(low),
        "mean_iv_residual_difference": (
            weighted_mean(broad, "iv_absolute_residual_15m")
            - weighted_mean(low, "iv_absolute_residual_15m")
        ),
        "median_iv_residual_difference": (
            weighted_median(broad, "iv_absolute_residual_15m")
            - weighted_median(low, "iv_absolute_residual_15m")
        ),
        "exceed_iv_rate_difference": (
            weighted_mean(broad, "movement_exceeds_prior_close_iv_15m")
            - weighted_mean(low, "movement_exceeds_prior_close_iv_15m")
        ),
        "bootstrap_80_mean_residual_lower": float(
            np.quantile(
                bootstrap_values["BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_mean_iv_residual"],
                0.10,
            )
        ),
    }
    broad_gate["supported"] = bool(
        broad_gate["broad_rows"] >= 200
        and broad_gate["low_route_support_rows"] >= 200
        and broad_gate["mean_iv_residual_difference"] > 0.0
        and broad_gate["median_iv_residual_difference"] > 0.0
        and broad_gate["exceed_iv_rate_difference"] > 0.0
        and broad_gate["bootstrap_80_mean_residual_lower"] >= 0.0
    )
    independent_gates["broad_conflict_iv_residual"] = broad_gate

    stored_gates = cast(Mapping[str, Mapping[str, Any]], decision["gates"])
    gate_mismatches: dict[str, dict[str, tuple[Any, Any]]] = {}
    for gate_name, expected_gate in independent_gates.items():
        observed_gate = stored_gates.get(gate_name, {})
        differences: dict[str, tuple[Any, Any]] = {}
        for field, expected in expected_gate.items():
            observed = observed_gate.get(field)
            if isinstance(expected, float):
                equal = isinstance(observed, (float, int)) and math.isclose(
                    expected,
                    float(observed),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            else:
                equal = observed == expected
            if not equal:
                differences[field] = (expected, observed)
        if differences:
            gate_mismatches[gate_name] = differences

    front_assessment = front_dimensions.loc[front_dimensions["period"].eq("assessment")]
    all_front_regimes_supported = True
    for regime in range(4):
        hard = front_assessment.loc[front_assessment["front_options_regime"].astype(int).eq(regime)]
        all_front_regimes_supported &= bool(
            float(front_assessment[f"front_options_regime_p_{regime}"].mean()) >= 0.05
            and hard["symbol"].nunique() >= 8
            and pd.to_datetime(hard["session"]).dt.to_period("M").nunique() >= 4
        )
    expected_statuses = {
        "daily_stock_context_status": (
            "supported" if independent_gates["daily_stock_context"]["passed"] else "not_supported"
        ),
        "front_options_regime_status": (
            "supported" if all_front_regimes_supported else "descriptive_only"
        ),
        "front_options_completion_status": (
            "supported"
            if independent_gates["front_options_completion"]["passed"]
            else "not_supported"
        ),
        "stock_to_iv_excess_status": (
            "supported" if independent_gates["stock_to_iv_excess"]["passed"] else "not_supported"
        ),
        "broad_conflict_iv_residual_status": (
            "supported" if broad_gate["supported"] else "descriptive_only"
        ),
        "back_expiry_preflight_status": str(preflight["status"]),
    }
    status_mismatches = {
        name: (expected, decision.get(name))
        for name, expected in expected_statuses.items()
        if decision.get(name) != expected
    }
    branch_supported = (
        expected_statuses["daily_stock_context_status"] == "supported",
        expected_statuses["front_options_completion_status"] == "supported",
        expected_statuses["stock_to_iv_excess_status"] == "supported",
    )
    if branch_supported == (True, True, True):
        expected_overall = "daily_stock_and_front_options_context_supported"
    elif branch_supported == (True, False, False):
        expected_overall = "daily_stock_context_supported_only"
    elif branch_supported == (False, True, False):
        expected_overall = "front_options_completion_context_supported_only"
    elif branch_supported == (False, False, True):
        expected_overall = "stock_structure_improves_iv_excess_only"
    elif sum(branch_supported) >= 2:
        expected_overall = "multiple_partial_context_increments_supported"
    else:
        expected_overall = "no_context_increment"
    check(
        checks,
        "decision_logic",
        not gate_mismatches
        and not status_mismatches
        and decision["overall_decision"] == expected_overall,
        evidence={
            "gate_mismatches": gate_mismatches,
            "status_mismatches": status_mismatches,
            "expected_overall": expected_overall,
            "observed_overall": decision["overall_decision"],
        },
    )

    failures = [row for row in checks if not bool(row["passed"])]
    result: dict[str, Any] = {
        **SAFETY_FLAGS,
        "audit_kind": "independent_artifact_and_source_reconstruction",
        "checks": checks,
        "checks_run": len(checks),
        "checks_passed": len(checks) - len(failures),
        "checks_failed": len(failures),
        "failed_checks": [row["check"] for row in failures],
        "model_probability_rows_per_model": 100,
        "models_manually_reconstructed": 6,
        "null_models_refitted": 0,
        "options_requests_made": 0,
        "passed": not failures,
    }
    write_json(PRIMARY / "lightweight_audit.json", result)
    write_json(PRIMARY / "independent_audit.json", result)
    decision["independent_audit_status"] = "passed" if not failures else "failed"
    if failures:
        failed_branches = {
            branch
            for row in failures
            for branch in str(row["branch"]).split(",")
            if branch != "global"
        }
        if "A" in failed_branches:
            decision["daily_stock_context_status"] = "blocked"
        if "B" in failed_branches:
            decision["front_options_completion_status"] = "blocked"
            decision["front_options_regime_status"] = "blocked"
        if "C" in failed_branches:
            decision["stock_to_iv_excess_status"] = "blocked"
            decision["broad_conflict_iv_residual_status"] = "blocked"
        if any(str(row["branch"]) == "global" for row in failures):
            decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
        else:
            remaining = (
                decision["daily_stock_context_status"] == "supported",
                decision["front_options_completion_status"] == "supported",
                decision["stock_to_iv_excess_status"] == "supported",
            )
            decision["overall_decision"] = (
                "daily_stock_and_front_options_context_supported"
                if remaining == (True, True, True)
                else (
                    "daily_stock_context_supported_only"
                    if remaining == (True, False, False)
                    else (
                        "front_options_completion_context_supported_only"
                        if remaining == (False, True, False)
                        else (
                            "stock_structure_improves_iv_excess_only"
                            if remaining == (False, False, True)
                            else (
                                "multiple_partial_context_increments_supported"
                                if sum(remaining) >= 2
                                else "no_context_increment"
                            )
                        )
                    )
                )
            )
    else:
        decision.update(expected_statuses)
        decision["overall_decision"] = expected_overall
    write_json(PRIMARY / "decision.json", decision)
    runner_namespace = runpy.run_path(
        str(EXPERIMENT_DIR / "run_screen_v01.py"),
        run_name="front_options_v01_report_renderer",
    )
    render_report = cast(Any, runner_namespace["render_report_from_artifacts"])
    report = str(render_report())
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")
    reports_dir = EXPERIMENT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "report.md").write_text(report, encoding="utf-8")
    print("passed" if not failures else "failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
