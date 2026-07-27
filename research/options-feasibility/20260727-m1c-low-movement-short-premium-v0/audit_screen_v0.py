#!/usr/bin/env python3
"""Independently audit the frozen M1C low-movement screen V0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
PREDECESSOR_PRIMARY = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0"
    / "artifacts"
    / "primary"
)
TOLERANCE = 1e-12
EPSILON = 1e-12
HORIZONS = (5, 10, 15, 30, 60)
CHECKPOINTS = tuple(range(6, 35, 2))
ANNUAL_TRADING_MINUTES = 252 * 390
MATCHED_SEEDS = tuple(range(2026072701, 2026072721))
PERMUTATION_SEEDS = tuple(range(2026072731, 2026072741))
BOOTSTRAP_SEEDS = {"assessment": 2026072751, "stress": 2026072752}
CHECKPOINT_GROUP_ORDER = {"early": 0, "middle": 1, "late": 2}
STATUS_KEYS = (
    "m1c_reconstruction_status",
    "low_tail_threshold_status",
    "checkpoint_low_movement_status",
    "fresh_quiet_episode_status",
    "m0_comparison_status",
    "score_monotonicity_status",
    "surprise_mover_status",
    "range_containment_status",
    "long_premium_veto_status",
    "short_premium_recorder_priority",
)
CLAIMS: dict[str, bool | str | int] = {
    "research_only": True,
    "retrospective_low_movement_screen": True,
    "m1c_frozen": True,
    "m1c_causal_feature_surface": True,
    "archived_contaminated_m1_excluded": True,
    "cross_sectional_future_filtered_features_excluded": True,
    "low_tail_thresholds_fit_on_2024_only": True,
    "primary_low_tail": "bottom_10_percent",
    "primary_horizon_minutes": 15,
    "option_pnl_calculated": False,
    "intraday_option_quotes_used": False,
    "short_option_pnl_claim": False,
    "defined_risk_structures_only_for_future_recording": True,
    "naked_short_options_authorised": False,
    "broker_access": False,
    "paper_orders_allowed": False,
    "live_orders_allowed": False,
    "strategy_promotion": False,
    "protected_start": "2026-01-01",
}


class AuditFailure(RuntimeError):
    """An independently detected fail-closed discrepancy."""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period, Path)):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise AuditFailure(detail)


def maximum_difference(left: Sequence[float], right: Sequence[float]) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    require(first.shape == second.shape, "numeric comparison shapes differ")
    finite = np.isfinite(first) & np.isfinite(second)
    require(
        bool(np.array_equal(np.isfinite(first), np.isfinite(second))),
        "numeric missing-value masks differ",
    )
    return float(np.max(np.abs(first[finite] - second[finite]))) if finite.any() else 0.0


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    data = np.asarray(values, dtype=float)
    mass = np.asarray(weights, dtype=float)
    require(
        data.ndim == 1
        and mass.ndim == 1
        and len(data) > 0
        and len(data) == len(mass)
        and np.isfinite(data).all()
        and np.isfinite(mass).all()
        and bool((mass > 0.0).all())
        and 0.0 <= quantile <= 1.0,
        "invalid independent weighted-quantile inputs",
    )
    order = np.argsort(data, kind="mergesort")
    ordered = data[order]
    ordered_mass = mass[order]
    positions = (np.cumsum(ordered_mass) - 0.5 * ordered_mass) / ordered_mass.sum()
    return float(
        np.interp(
            quantile,
            positions,
            ordered,
            left=ordered[0],
            right=ordered[-1],
        )
    )


def weights(frame: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
    column = "_analysis_weight" if "_analysis_weight" in frame else "row_weight"
    output = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    require(
        len(output) > 0 and np.isfinite(output).all() and bool((output > 0.0).all()),
        "invalid independent analysis weights",
    )
    return output


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    mass = weights(frame)
    valid = np.isfinite(values)
    require(bool(valid.any()), f"no finite values for {column}")
    return float(np.average(values[valid], weights=mass[valid]))


def weighted_percentile(frame: pd.DataFrame, column: str, quantile: float) -> float:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    mass = weights(frame)
    valid = np.isfinite(values)
    require(bool(valid.any()), f"no finite values for {column}")
    return weighted_quantile(values[valid], mass[valid], quantile)


def summary_metrics(frame: pd.DataFrame, horizon: int = 15) -> dict[str, float]:
    available = frame.loc[frame[f"available_{horizon}m"].astype(bool)].copy()
    require(not available.empty, "an independently audited metric population is empty")
    remains = f"movement_remains_below_iv_{horizon}m"
    residual = f"terminal_iv_residual_{horizon}m"
    absolute = f"absolute_return_{horizon}m"
    excursion = f"maximum_absolute_excursion_{horizon}m"
    ratio = f"excursion_sigma_ratio_{horizon}m"
    return {
        "remains_below_iv_rate": weighted_mean(available, remains),
        "mean_iv_residual": weighted_mean(available, residual),
        "median_iv_residual": weighted_percentile(available, residual, 0.50),
        "mean_absolute_movement": weighted_mean(available, absolute),
        "p95_absolute_movement": weighted_percentile(available, absolute, 0.95),
        "mean_maximum_excursion": weighted_mean(available, excursion),
        "breach_1_5_sigma_rate": float(
            np.average(
                available[ratio].gt(1.5).to_numpy(float),
                weights=weights(available),
            )
        ),
        "breach_2_0_sigma_rate": float(
            np.average(
                available[ratio].gt(2.0).to_numpy(float),
                weights=weights(available),
            )
        ),
    }


def null_metrics(frame: pd.DataFrame, baseline: Mapping[str, float]) -> dict[str, float]:
    values = summary_metrics(frame)
    return {
        "remains_below_iv_rate": values["remains_below_iv_rate"],
        "npv_lift": values["remains_below_iv_rate"] - baseline["remains_below_iv_rate"],
        "mean_iv_residual": values["mean_iv_residual"],
        "median_iv_residual": values["median_iv_residual"],
        "mean_maximum_excursion": values["mean_maximum_excursion"],
        "breach_1_5_sigma_rate": values["breach_1_5_sigma_rate"],
        "breach_2_0_sigma_rate": values["breach_2_0_sigma_rate"],
    }


def load_analytic() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(PRIMARY / "checkpoint_predictions.parquet")
    movement = pd.read_parquet(PRIMARY / "movement_outcomes.parquet")
    paths = pd.read_parquet(PRIMARY / "path_excursion_outcomes.parquet")
    episodes = pd.read_parquet(PRIMARY / "fresh_quiet_episodes.parquet")
    for name, frame in (
        ("predictions", predictions),
        ("movement", movement),
        ("paths", paths),
    ):
        require(not frame["row_id"].astype(str).duplicated().any(), f"duplicate {name} row IDs")
    identity = {"period", "stock", "session", "checkpoint", "entry_timestamp"}
    movement_payload = movement.drop(columns=[column for column in identity if column in movement])
    path_drop = {
        *identity,
        "entry_price",
        "available_horizons",
        *(f"available_{horizon}m" for horizon in HORIZONS),
    }
    path_payload = paths.drop(columns=[column for column in path_drop if column in paths])
    analytic = predictions.merge(
        movement_payload,
        on="row_id",
        how="left",
        validate="one_to_one",
    ).merge(
        path_payload,
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    require(len(analytic) == len(predictions), "outcome merge changed checkpoint support")
    return analytic, predictions, movement, episodes


def manual_probabilities(
    frame: pd.DataFrame,
    specification: Mapping[str, Any],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    features = [str(value) for value in specification["numeric_features"]]
    medians = np.asarray(specification["numeric_medians"], dtype=float)
    means = np.asarray(specification["numeric_means"], dtype=float)
    scales = np.asarray(specification["numeric_scales"], dtype=float)
    raw = frame[features].to_numpy(float)
    numeric = (np.where(np.isfinite(raw), raw, medians) - means) / scales
    parts = [numeric]
    generated_columns = list(features)
    for control in specification["category_controls"]:
        observed = frame[str(control)].astype(str).to_numpy()
        levels = [str(value) for value in specification["category_levels"][str(control)]]
        for level in levels[1:]:
            parts.append((observed == level).astype(float)[:, None])
            generated_columns.append(f"control_{control}__{level}")
    require(
        generated_columns == [str(value) for value in specification["design_columns"]],
        "independent frozen design-column order differs",
    )
    design = np.concatenate(parts, axis=1)
    coefficients = np.asarray(specification["coefficients"], dtype=float)
    require(
        design.shape[1] == len(coefficients),
        "independent frozen design has the wrong width",
    )
    linear = design @ coefficients + float(specification["intercept"])
    return np.asarray(
        1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))),
        dtype=float,
    )


def evenly_spaced_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    require(len(frame) >= count, f"fewer than {count} rows are available for manual audit")
    ordered = frame.sort_values(["session", "stock", "checkpoint"], kind="mergesort")
    indices = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
    return ordered.iloc[indices].copy()


def audit_sources(source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    checked: list[dict[str, str]] = []
    for payload in source_manifest["sources"]:
        path = Path(str(payload["path"]))
        require(path.is_file(), f"frozen source is missing: {path}")
        observed = sha256_file(path)
        require(observed == payload["sha256"], f"frozen source hash changed: {path}")
        checked.append({"path": str(path), "sha256": observed})
    for key in ("predecessor_runner", "predecessor_feature_manifest"):
        payload = source_manifest[key]
        path = REPO_ROOT / str(payload["path"])
        observed = sha256_file(path)
        require(observed == payload["sha256"], f"{key} hash changed")
        checked.append({"path": str(path), "sha256": observed})
    return {"sources_checked": len(checked), "sources": checked}


def audit_manifest_and_probabilities(
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(PRIMARY / "m1c_feature_manifest.json")
    predecessor = read_json(PREDECESSOR_PRIMARY / "causal_movement_feature_manifest.json")
    require(manifest["group_o"] == predecessor["group_o"], "Group O manifest differs")
    require(
        manifest["causally_valid_group_i"] == predecessor["causally_valid_group_i"],
        "causal Group I manifest differs",
    )
    require(
        manifest["model_specification"] == predecessor["model_specification"],
        "M1C coefficients or preprocessing differ from the predecessor",
    )
    require(
        manifest["m0_model_specification"] == predecessor["m0_model_specification"],
        "M0 coefficients or preprocessing differ from the predecessor",
    )
    expected_order = [*manifest["group_o"], *manifest["causally_valid_group_i"]]
    require(manifest["feature_order"] == expected_order, "causal feature order differs")
    forbidden = set(manifest["contaminated_and_peer_normalised_features_excluded"])
    require(
        not forbidden.intersection(expected_order),
        "a contaminated feature appears in the M1C feature order",
    )
    require(
        {"signed_pressure", "tension"}.issubset(forbidden),
        "signed pressure or contaminated tension is absent from the exclusion manifest",
    )
    sample = evenly_spaced_sample(predictions, 100)
    manual_m1c = manual_probabilities(sample, manifest["model_specification"])
    manual_m0 = manual_probabilities(sample, manifest["m0_model_specification"])
    m1c_difference = maximum_difference(manual_m1c, sample["M1C_probability"])
    m0_difference = maximum_difference(manual_m0, sample["M0_probability"])
    require(m1c_difference <= TOLERANCE, "100-row M1C probability audit failed")
    require(m0_difference <= TOLERANCE, "100-row M0 probability audit failed")
    comparison = pd.read_csv(PRIMARY / "m1c_probability_comparison.csv")
    require(
        set(comparison["row_id"].astype(str)) == set(predictions["row_id"].astype(str)),
        "probability comparison row identities differ",
    )
    comparison_difference = float(comparison["absolute_probability_difference"].max())
    require(comparison_difference <= TOLERANCE, "full probability comparison failed")
    expected_weight = 1.0 / predictions["stock_local_checkpoints_in_session"].to_numpy(float)
    weight_difference = maximum_difference(expected_weight, predictions["row_weight"])
    grouped_weight = predictions.groupby(["stock", "session"], sort=True)["row_weight"].sum()
    group_weight_difference = maximum_difference(grouped_weight, np.ones(len(grouped_weight)))
    require(
        weight_difference <= TOLERANCE and group_weight_difference <= TOLERANCE,
        "candidate-normalised weights differ",
    )
    return manifest, {
        "manual_m1c_probabilities": 100,
        "manual_m0_probabilities": 100,
        "maximum_m1c_probability_difference": m1c_difference,
        "maximum_m0_probability_difference": m0_difference,
        "maximum_full_comparison_difference": comparison_difference,
        "maximum_row_weight_difference": weight_difference,
        "maximum_stock_session_weight_sum_difference": group_weight_difference,
    }


def audit_thresholds(predictions: pd.DataFrame) -> dict[str, Any]:
    artifact = read_json(PRIMARY / "frozen_low_tail_thresholds.json")
    decile_artifact = read_json(PRIMARY / "frozen_score_deciles.json")
    development = predictions.loc[predictions["period"].eq("development")]
    require(
        development["session"].astype(str).between("2024-01-01", "2024-12-31").all(),
        "threshold-development rows are not confined to 2024",
    )
    mass = development["row_weight"].to_numpy(float)
    quantiles = {
        "bottom_5_percent": 0.05,
        "bottom_10_percent": 0.10,
        "bottom_20_percent": 0.20,
    }
    threshold_differences: list[float] = []
    membership_mismatches = 0
    for model, column, prefix in (
        ("M1C", "M1C_probability", "m1c"),
        ("M0", "M0_probability", "m0"),
    ):
        for label, quantile in quantiles.items():
            reconstructed = weighted_quantile(
                development[column].to_numpy(float),
                mass,
                quantile,
            )
            stored = float(artifact[model][label])
            threshold_differences.append(abs(reconstructed - stored))
            expected_membership = predictions[column].le(stored).to_numpy(bool)
            actual_membership = predictions[f"{prefix}_{label}"].to_numpy(bool)
            membership_mismatches += int(np.count_nonzero(expected_membership != actual_membership))
    reconstructed_deciles = [
        weighted_quantile(
            development["M1C_probability"].to_numpy(float),
            mass,
            quantile,
        )
        for quantile in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ]
    decile_difference = maximum_difference(reconstructed_deciles, decile_artifact["boundaries"])
    expected_deciles = (
        np.searchsorted(
            np.asarray(decile_artifact["boundaries"], dtype=float),
            predictions["M1C_probability"].to_numpy(float),
            side="left",
        )
        + 1
    )
    decile_mismatches = int(
        np.count_nonzero(expected_deciles != predictions["m1c_score_decile"].to_numpy(int))
    )
    sample = evenly_spaced_sample(predictions, 100)
    sample_membership_mismatches = int(
        np.count_nonzero(
            sample["M1C_probability"].le(float(artifact["M1C"]["bottom_10_percent"])).to_numpy(bool)
            != sample["m1c_bottom_10_percent"].to_numpy(bool)
        )
    )
    maximum_threshold_difference = max(threshold_differences)
    require(maximum_threshold_difference <= TOLERANCE, "2024 low-tail thresholds differ")
    require(decile_difference <= TOLERANCE, "2024 decile boundaries differ")
    require(membership_mismatches == 0, "frozen low-tail membership differs")
    require(decile_mismatches == 0, "frozen score-decile membership differs")
    require(sample_membership_mismatches == 0, "100-row bottom-tail audit differs")
    return {
        "fit_rows": int(len(development)),
        "fit_start": str(development["session"].min()),
        "fit_end": str(development["session"].max()),
        "maximum_threshold_difference": maximum_threshold_difference,
        "maximum_decile_boundary_difference": decile_difference,
        "all_tail_membership_mismatches": membership_mismatches,
        "all_decile_membership_mismatches": decile_mismatches,
        "manual_bottom_tail_memberships": 100,
        "manual_bottom_tail_membership_mismatches": sample_membership_mismatches,
    }


def completed_bar_source(source_manifest: Mapping[str, Any]) -> Path:
    matches = [
        Path(str(payload["path"]))
        for payload in source_manifest["sources"]
        if payload["role"] == "completed_five_minute_stock_and_market_bars"
    ]
    require(len(matches) == 1, "completed-bar source manifest is ambiguous")
    return matches[0]


def load_unprotected_bars(path: Path) -> pd.DataFrame:
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
        "vti__bar_log_return",
    ]
    bars = pd.read_parquet(
        path,
        columns=columns,
        filters=[("session", "<", "2026-01-01")],
    )
    bars["session"] = bars["session"].astype(str)
    require(bars["session"].max() < "2026-01-01", "protected bar outcomes were materialised")
    return bars


def audit_chronology(
    predictions: pd.DataFrame,
    bars: pd.DataFrame,
) -> dict[str, Any]:
    sessions = sorted(bars["session"].unique())
    previous_session = {sessions[index]: sessions[index - 1] for index in range(1, len(sessions))}
    required_dates = predictions["required_options_date"].astype(str)
    observed_dates = predictions["options_observation_date"].astype(str)
    exact_previous = predictions["session"].astype(str).map(previous_session)
    previous_date_mismatches = int(
        np.count_nonzero(
            (required_dates != observed_dates).to_numpy(bool)
            | (required_dates != exact_previous.astype(str)).to_numpy(bool)
        )
    )
    checkpoint_timestamp = pd.to_datetime(
        predictions["checkpoint_timestamp_utc"], utc=True, errors="raise"
    )
    feature_timestamp = pd.to_datetime(
        predictions["feature_available_timestamp_utc"], utc=True, errors="raise"
    )
    entry_timestamp = pd.to_datetime(predictions["entry_timestamp"], utc=True, errors="raise")
    timing_mismatches = int(
        np.count_nonzero(
            (
                (feature_timestamp != entry_timestamp)
                | (checkpoint_timestamp + pd.Timedelta(minutes=5) != entry_timestamp)
                | (feature_timestamp > entry_timestamp)
            ).to_numpy(bool)
        )
    )
    dte = (
        pd.to_datetime(predictions["front_expiration_date"], errors="raise")
        - pd.to_datetime(predictions["session"], errors="raise")
    ).dt.days
    dte_mismatches = int(np.count_nonzero(dte.to_numpy(int) != predictions["option_dte"]))
    require(previous_date_mismatches == 0, "exact previous-close IV chronology differs")
    require(timing_mismatches == 0, "completed-bar entry chronology differs")
    require(dte_mismatches == 0, "option DTE differs")
    return {
        "previous_close_date_mismatches": previous_date_mismatches,
        "completed_bar_timing_mismatches": timing_mismatches,
        "option_dte_mismatches": dte_mismatches,
        "maximum_checkpoint_session": str(predictions["session"].max()),
        "protected_rows_read": 0,
    }


def audit_outcomes(
    predictions: pd.DataFrame,
    movement: pd.DataFrame,
    paths: pd.DataFrame,
    bars: pd.DataFrame,
) -> dict[str, Any]:
    candidates = predictions.loc[predictions["period"].isin(["assessment", "stress"])]
    sample = evenly_spaced_sample(candidates, 100)
    movement_indexed = movement.set_index("row_id")
    paths_indexed = paths.set_index("row_id")
    grouped_bars = {
        (str(stock), str(session)): group.sort_values("bar_ordinal", kind="mergesort").set_index(
            "bar_ordinal"
        )
        for (stock, session), group in bars.groupby(["symbol", "session"], sort=False)
    }
    differences: dict[str, float] = {
        "entry_timestamp": 0.0,
        "entry_price": 0.0,
        "terminal_return": 0.0,
        "maximum_up_excursion": 0.0,
        "maximum_down_excursion": 0.0,
        "maximum_absolute_excursion": 0.0,
        "realised_path_range": 0.0,
        "iv_sigma": 0.0,
        "iv_expected_absolute": 0.0,
        "terminal_iv_residual": 0.0,
        "excursion_sigma_ratio": 0.0,
        "path_range_sigma_ratio": 0.0,
    }
    availability_mismatches = 0
    categorical_mismatches = 0
    verified_by_horizon = {str(horizon): 0 for horizon in HORIZONS}
    for prediction in sample.itertuples(index=False):
        row_id = str(prediction.row_id)
        move = movement_indexed.loc[row_id]
        path = paths_indexed.loc[row_id]
        session_bars = grouped_bars[(str(prediction.stock), str(prediction.session))]
        checkpoint = int(prediction.checkpoint)
        entry_bar = session_bars.loc[checkpoint]
        entry_price = float(entry_bar["open"])
        entry_timestamp = pd.Timestamp(entry_bar["bar_start_timestamp"])
        stored_entry_timestamp = pd.Timestamp(move["entry_timestamp"])
        differences["entry_timestamp"] = max(
            differences["entry_timestamp"],
            abs((entry_timestamp - stored_entry_timestamp).total_seconds()),
        )
        differences["entry_price"] = max(
            differences["entry_price"],
            abs(entry_price - float(move["entry_price"])),
        )
        for horizon in HORIZONS:
            ordinals = list(range(checkpoint, checkpoint + horizon // 5))
            available = all(ordinal in session_bars.index for ordinal in ordinals)
            availability_mismatches += int(bool(move[f"available_{horizon}m"]) != available)
            availability_mismatches += int(bool(path[f"available_{horizon}m"]) != available)
            if not available:
                continue
            verified_by_horizon[str(horizon)] += 1
            future = session_bars.loc[ordinals]
            terminal_close = float(future.iloc[-1]["close"])
            signed_return = math.log(terminal_close / entry_price)
            absolute_return = abs(signed_return)
            highs = future["high"].to_numpy(float)
            lows = future["low"].to_numpy(float)
            upward = np.log(highs / entry_price)
            downward = np.log(lows / entry_price)
            up_index = int(np.argmax(upward))
            down_index = int(np.argmin(downward))
            maximum_up = float(upward[up_index])
            maximum_down = float(downward[down_index])
            if maximum_up >= abs(maximum_down):
                maximum_absolute = maximum_up
                absolute_index = up_index
            else:
                maximum_absolute = abs(maximum_down)
                absolute_index = down_index
            realised_range = math.log(float(np.max(highs)) / float(np.min(lows)))
            sigma = float(prediction.atm_iv) * math.sqrt(horizon / ANNUAL_TRADING_MINUTES)
            expected = sigma * math.sqrt(2.0 / math.pi)
            residual = absolute_return - expected
            excursion_ratio = maximum_absolute / (sigma + EPSILON)
            range_ratio = realised_range / (2.0 * sigma + EPSILON)
            comparisons = {
                "terminal_return": (
                    signed_return,
                    float(move[f"signed_return_{horizon}m"]),
                ),
                "maximum_up_excursion": (
                    maximum_up,
                    float(path[f"maximum_up_excursion_{horizon}m"]),
                ),
                "maximum_down_excursion": (
                    maximum_down,
                    float(path[f"maximum_down_excursion_{horizon}m"]),
                ),
                "maximum_absolute_excursion": (
                    maximum_absolute,
                    float(path[f"maximum_absolute_excursion_{horizon}m"]),
                ),
                "realised_path_range": (
                    realised_range,
                    float(path[f"realised_path_range_{horizon}m"]),
                ),
                "iv_sigma": (sigma, float(move[f"iv_sigma_{horizon}m"])),
                "iv_expected_absolute": (
                    expected,
                    float(move[f"iv_expected_absolute_{horizon}m"]),
                ),
                "terminal_iv_residual": (
                    residual,
                    float(move[f"terminal_iv_residual_{horizon}m"]),
                ),
                "excursion_sigma_ratio": (
                    excursion_ratio,
                    float(path[f"excursion_sigma_ratio_{horizon}m"]),
                ),
                "path_range_sigma_ratio": (
                    range_ratio,
                    float(path[f"path_range_sigma_ratio_{horizon}m"]),
                ),
            }
            for name, (expected_value, actual_value) in comparisons.items():
                differences[name] = max(
                    differences[name],
                    abs(expected_value - actual_value),
                )
            expected_times = {
                f"time_to_maximum_up_excursion_{horizon}m": (up_index + 1) * 5,
                f"time_to_maximum_down_excursion_{horizon}m": (down_index + 1) * 5,
                f"time_to_maximum_absolute_excursion_{horizon}m": (absolute_index + 1) * 5,
            }
            categorical_mismatches += sum(
                int(float(path[column]) != float(value)) for column, value in expected_times.items()
            )
            categorical_mismatches += int(
                bool(move[f"movement_exceeds_iv_{horizon}m"]) != bool(absolute_return > expected)
            )
            categorical_mismatches += int(
                bool(move[f"movement_remains_below_iv_{horizon}m"])
                != bool(absolute_return <= expected)
            )
            categorical_mismatches += int(
                bool(path[f"crossed_above_and_below_entry_{horizon}m"])
                != bool((highs > entry_price).any() and (lows < entry_price).any())
            )
    require(availability_mismatches == 0, "manual horizon availability differs")
    require(categorical_mismatches == 0, "manual path classifications differ")
    require(max(differences.values()) <= TOLERANCE, "manual outcome reconstruction differs")
    require(verified_by_horizon["15"] >= 100, "fewer than 100 fifteen-minute outcomes audited")
    return {
        "manual_probabilistic_outcome_rows": 100,
        "manual_fifteen_minute_outcomes": verified_by_horizon["15"],
        "manual_maximum_excursion_outcomes": verified_by_horizon["15"],
        "manual_iv_residuals": verified_by_horizon["15"],
        "verified_available_outcomes_by_horizon": verified_by_horizon,
        "availability_mismatches": availability_mismatches,
        "classification_mismatches": categorical_mismatches,
        "maximum_differences": differences,
    }


def reconstruct_episodes(
    analytic: pd.DataFrame,
    *,
    threshold: float,
    probability_column: str,
) -> pd.DataFrame:
    ordered = analytic.loc[analytic["period"].isin(["assessment", "stress"])].copy()
    ordered["entry_timestamp"] = pd.to_datetime(ordered["entry_timestamp"], utc=True)
    ordered = ordered.sort_values(["stock", "session", "checkpoint"], kind="mergesort").reset_index(
        drop=True
    )
    previous = ordered.groupby(["stock", "session"], sort=False)[probability_column].shift()
    crossings = ordered.loc[
        ordered[probability_column].le(threshold) & (previous.isna() | previous.gt(threshold))
    ].copy()
    crossings["_previous_probability"] = previous.loc[crossings.index]
    selected: list[int] = []
    episode_number: dict[int, int] = {}
    elapsed_minutes: dict[int, float] = {}
    for _, group in crossings.groupby(["stock", "session"], sort=True):
        previous_start: pd.Timestamp | None = None
        number = 0
        for index, row in group.iterrows():
            current = pd.Timestamp(row["entry_timestamp"])
            elapsed = (
                math.nan
                if previous_start is None
                else (current - previous_start).total_seconds() / 60.0
            )
            if math.isfinite(elapsed) and elapsed < 30.0:
                continue
            selected.append(int(index))
            number += 1
            episode_number[int(index)] = number
            elapsed_minutes[int(index)] = elapsed
            previous_start = current
    output = crossings.loc[selected].copy()
    output["episode_number"] = [episode_number[index] for index in selected]
    output["minutes_since_previous_quiet_episode"] = [elapsed_minutes[index] for index in selected]
    output["previous_probability"] = output["_previous_probability"]
    output["current_probability"] = output[probability_column]
    return output.reset_index(drop=True)


def audit_episodes(
    analytic: pd.DataFrame,
    stored_episodes: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    thresholds = read_json(PRIMARY / "frozen_low_tail_thresholds.json")
    reconstructed: dict[str, pd.DataFrame] = {}
    identity_mismatches = 0
    for tail in ("bottom_5_percent", "bottom_10_percent", "bottom_20_percent"):
        frame = reconstruct_episodes(
            analytic,
            threshold=float(thresholds["M1C"][tail]),
            probability_column="M1C_probability",
        )
        reconstructed[tail] = frame
        stored = stored_episodes.loc[stored_episodes["tail"].eq(tail)]
        identity_mismatches += len(
            set(frame["row_id"].astype(str)).symmetric_difference(set(stored["row_id"].astype(str)))
        )
    m0 = reconstruct_episodes(
        analytic,
        threshold=float(thresholds["M0"]["bottom_10_percent"]),
        probability_column="M0_probability",
    )
    binding = reconstructed["bottom_10_percent"].sort_values("row_id", kind="mergesort")
    stored_binding = (
        stored_episodes.loc[stored_episodes["tail"].eq("bottom_10_percent")]
        .sort_values("row_id", kind="mergesort")
        .set_index("row_id")
    )
    sample = evenly_spaced_sample(binding, 50)
    field_mismatches = 0
    maximum_probability_difference = 0.0
    maximum_elapsed_difference = 0.0
    for row in sample.itertuples(index=False):
        stored = stored_binding.loc[str(row.row_id)]
        field_mismatches += int(int(row.episode_number) != int(stored["episode_number"]))
        field_mismatches += int(
            pd.Timestamp(row.entry_timestamp) != pd.Timestamp(stored["entry_timestamp"])
        )
        field_mismatches += int(str(row.available_horizons) != str(stored["available_horizons"]))
        probability_pairs = (
            (row.current_probability, stored["current_m1c_probability"]),
            (row.previous_probability, stored["previous_m1c_probability"]),
        )
        for expected, actual in probability_pairs:
            if pd.isna(expected) and pd.isna(actual):
                continue
            maximum_probability_difference = max(
                maximum_probability_difference,
                abs(float(expected) - float(actual)),
            )
        expected_elapsed = float(row.minutes_since_previous_quiet_episode)
        actual_elapsed = float(stored["minutes_since_previous_quiet_episode"])
        if not (math.isnan(expected_elapsed) and math.isnan(actual_elapsed)):
            maximum_elapsed_difference = max(
                maximum_elapsed_difference,
                abs(expected_elapsed - actual_elapsed),
            )
    require(identity_mismatches == 0, "fresh quiet episode identities differ")
    require(field_mismatches == 0, "50-row fresh episode field audit differs")
    require(
        maximum_probability_difference <= TOLERANCE and maximum_elapsed_difference <= TOLERANCE,
        "50-row fresh episode numeric audit differs",
    )
    spacing = binding["minutes_since_previous_quiet_episode"].dropna()
    require(bool(spacing.ge(30.0).all()), "fresh episodes violate thirty-minute spacing")
    return (
        reconstructed,
        m0,
        {
            "all_tail_episode_identity_mismatches": identity_mismatches,
            "manual_fresh_quiet_episodes": 50,
            "manual_fresh_episode_field_mismatches": field_mismatches,
            "maximum_episode_probability_difference": maximum_probability_difference,
            "maximum_episode_spacing_difference": maximum_elapsed_difference,
            "minimum_noninitial_spacing_minutes": float(spacing.min()),
        },
    )


def month_ordinal(value: object) -> int:
    return int(pd.Period(str(value), freq="M").ordinal)


def match_cell(row: pd.Series) -> tuple[object, ...]:
    return tuple(
        row[column]
        for column in ("period", "stock", "month", "checkpoint_group", "atm_iv_quartile")
    )


def fallback_distance(
    wanted: tuple[object, ...],
    candidate: tuple[object, ...],
) -> tuple[int, int, int, int]:
    return (
        int(str(wanted[1]) != str(candidate[1])),
        abs(month_ordinal(wanted[2]) - month_ordinal(candidate[2])),
        abs(CHECKPOINT_GROUP_ORDER[str(wanted[3])] - CHECKPOINT_GROUP_ORDER[str(candidate[3])]),
        abs(int(wanted[4]) - int(candidate[4])),
    )


def independent_matched_selection(
    population: pd.DataFrame,
    real_tail: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rng = np.random.default_rng(seed)
    tail_ids = set(real_tail["row_id"].astype(str))
    candidates = population.loc[~population["row_id"].astype(str).isin(tail_ids)]
    buckets: dict[tuple[object, ...], list[int]] = {}
    for index, row in candidates.iterrows():
        buckets.setdefault(match_cell(row), []).append(int(index))
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[int] = []
    exact = 0
    fallback = 0
    maximum_fallback = 0
    for _, row in real_tail.sort_values("row_id", kind="mergesort").iterrows():
        wanted = match_cell(row)
        bucket = buckets.get(wanted, [])
        if bucket:
            selected.append(bucket.pop())
            exact += 1
            continue
        candidates_with_rows = [key for key, values in buckets.items() if values]
        require(bool(candidates_with_rows), "matched selection exhausted candidates")
        ranked = sorted(
            (
                fallback_distance(wanted, key),
                tuple(str(value) for value in key),
                key,
            )
            for key in candidates_with_rows
        )
        distance, _, selected_key = ranked[0]
        selected.append(buckets[selected_key].pop())
        fallback += 1
        maximum_fallback = max(maximum_fallback, sum(distance))
    output = population.loc[selected].copy().reset_index(drop=True)
    return output, {
        "selected_rows": len(output),
        "exact_match_rows": exact,
        "nearest_cell_fallback_rows": fallback,
        "maximum_fallback_distance": maximum_fallback,
    }


def independent_permutation(frame: pd.DataFrame, seed: int) -> pd.Series:
    output = frame["M1C_probability"].copy()
    rng = np.random.default_rng(seed)
    for labels in frame.groupby(["session", "checkpoint_group"], sort=True).groups.values():
        group_labels = list(labels)
        output.loc[group_labels] = rng.permutation(
            frame.loc[group_labels, "M1C_probability"].to_numpy(float)
        )
    return output


def compare_metric_row(
    actual: Mapping[str, Any],
    expected: Mapping[str, float],
    columns: Sequence[str],
) -> float:
    return max(abs(float(actual[column]) - float(expected[column])) for column in columns)


def audit_nulls(
    analytic: pd.DataFrame,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, Any]]:
    plan = read_json(PRIMARY / "frozen_resampling_plan.json")
    matched_table = pd.read_csv(PRIMARY / "matched_random_null_metrics.csv")
    permutation_table = pd.read_csv(PRIMARY / "probability_permutation_null_metrics.csv")
    thresholds = read_json(PRIMARY / "frozen_low_tail_thresholds.json")
    metric_columns = (
        "remains_below_iv_rate",
        "npv_lift",
        "mean_iv_residual",
        "median_iv_residual",
        "mean_maximum_excursion",
        "breach_1_5_sigma_rate",
        "breach_2_0_sigma_rate",
    )
    maximum_matched_difference = 0.0
    maximum_permutation_difference = 0.0
    matched_identity_mismatches = 0
    permutation_identity_mismatches = 0
    matched_wins: dict[str, dict[str, int]] = {}
    permutation_wins: dict[str, dict[str, int]] = {}
    for period in ("assessment", "stress"):
        population = (
            analytic.loc[analytic["period"].eq(period)]
            .sort_values("row_id", kind="mergesort")
            .reset_index(drop=True)
        )
        real_tail = population.loc[population["m1c_bottom_10_percent"].astype(bool)]
        baseline = summary_metrics(population)
        real = null_metrics(real_tail, baseline)
        matched_wins[period] = {column: 0 for column in metric_columns}
        permutation_wins[period] = {column: 0 for column in metric_columns}
        require(len(plan["matched_random"][period]) == 20, "matched plan does not have 20 draws")
        require(
            len(plan["probability_permutation"][period]) == 10,
            "permutation plan does not have 10 draws",
        )
        for draw, seed in enumerate(MATCHED_SEEDS, start=1):
            selected, selection_audit = independent_matched_selection(
                population,
                real_tail,
                seed,
            )
            stored_plan = plan["matched_random"][period][draw - 1]
            matched_identity_mismatches += len(
                set(selected["row_id"].astype(str)).symmetric_difference(
                    set(str(value) for value in stored_plan["selected_row_ids"])
                )
            )
            require(
                selection_audit
                == {
                    key: int(stored_plan[key])
                    for key in (
                        "selected_rows",
                        "exact_match_rows",
                        "nearest_cell_fallback_rows",
                        "maximum_fallback_distance",
                    )
                },
                "matched fallback audit differs",
            )
            computed = null_metrics(selected, baseline)
            table_row = matched_table.loc[
                matched_table["period"].eq(period) & matched_table["draw"].eq(draw)
            ].iloc[0]
            maximum_matched_difference = max(
                maximum_matched_difference,
                compare_metric_row(table_row, computed, metric_columns),
            )
            for column in metric_columns:
                higher = column in {"remains_below_iv_rate", "npv_lift"}
                beat = (
                    real[column] > computed[column] if higher else real[column] < computed[column]
                )
                matched_wins[period][column] += int(beat)
        for draw, seed in enumerate(PERMUTATION_SEEDS, start=1):
            permuted = independent_permutation(population, seed)
            null_tail = population.loc[permuted.le(float(thresholds["M1C"]["bottom_10_percent"]))]
            stored_plan = plan["probability_permutation"][period][draw - 1]
            permutation_identity_mismatches += len(
                set(null_tail["row_id"].astype(str)).symmetric_difference(
                    set(str(value) for value in stored_plan["tail_row_ids"])
                )
            )
            computed = null_metrics(null_tail, baseline)
            table_row = permutation_table.loc[
                permutation_table["period"].eq(period) & permutation_table["draw"].eq(draw)
            ].iloc[0]
            maximum_permutation_difference = max(
                maximum_permutation_difference,
                compare_metric_row(table_row, computed, metric_columns),
            )
            for column in metric_columns:
                higher = column in {"remains_below_iv_rate", "npv_lift"}
                beat = (
                    real[column] > computed[column] if higher else real[column] < computed[column]
                )
                permutation_wins[period][column] += int(beat)
    require(matched_identity_mismatches == 0, "matched-null row identities differ")
    require(permutation_identity_mismatches == 0, "permutation-null row identities differ")
    require(maximum_matched_difference <= TOLERANCE, "matched-null metrics differ")
    require(maximum_permutation_difference <= TOLERANCE, "permutation-null metrics differ")
    return (
        matched_wins,
        permutation_wins,
        {
            "matched_draws_per_period": 20,
            "permutation_draws_per_period": 10,
            "matched_identity_mismatches": matched_identity_mismatches,
            "permutation_identity_mismatches": permutation_identity_mismatches,
            "maximum_matched_metric_difference": maximum_matched_difference,
            "maximum_permutation_metric_difference": maximum_permutation_difference,
        },
    )


def resample_sessions(frame: pd.DataFrame, sampled_sessions: Sequence[str]) -> pd.DataFrame:
    multiplicities = Counter(str(value) for value in sampled_sessions)
    output = frame.loc[frame["session"].astype(str).isin(multiplicities)].copy()
    output["_analysis_weight"] = output["row_weight"].to_numpy(float) * output["session"].astype(
        str
    ).map(multiplicities).to_numpy(float)
    return output


def bootstrap_statistics(
    population: pd.DataFrame,
    fresh: pd.DataFrame,
) -> dict[str, float]:
    full = summary_metrics(population)
    m1c = summary_metrics(population.loc[population["m1c_bottom_10_percent"].astype(bool)])
    m0 = summary_metrics(population.loc[population["m0_bottom_10_percent"].astype(bool)])
    episodes = summary_metrics(fresh)
    return {
        "m1c_bottom_tail_remains_below_iv_rate": m1c["remains_below_iv_rate"],
        "bottom_tail_npv_lift": m1c["remains_below_iv_rate"] - full["remains_below_iv_rate"],
        "mean_terminal_iv_residual": m1c["mean_iv_residual"],
        "median_terminal_iv_residual": m1c["median_iv_residual"],
        "mean_terminal_absolute_movement": m1c["mean_absolute_movement"],
        "p95_terminal_absolute_movement": m1c["p95_absolute_movement"],
        "breach_1_5_sigma_rate": m1c["breach_1_5_sigma_rate"],
        "breach_2_0_sigma_rate": m1c["breach_2_0_sigma_rate"],
        "m1c_minus_m0_remains_below_iv_difference": (
            m1c["remains_below_iv_rate"] - m0["remains_below_iv_rate"]
        ),
        "m1c_minus_m0_mean_residual_difference": (m1c["mean_iv_residual"] - m0["mean_iv_residual"]),
        "m1c_minus_m0_1_5_sigma_breach_difference": (
            m1c["breach_1_5_sigma_rate"] - m0["breach_1_5_sigma_rate"]
        ),
        "fresh_quiet_episode_remains_below_iv_rate": episodes["remains_below_iv_rate"],
        "fresh_quiet_episode_mean_residual": episodes["mean_iv_residual"],
        "fresh_episode_1_5_sigma_breach_difference_vs_full": (
            episodes["breach_1_5_sigma_rate"] - full["breach_1_5_sigma_rate"]
        ),
    }


def audit_bootstrap(
    analytic: pd.DataFrame,
    episodes: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    plan = read_json(PRIMARY / "frozen_resampling_plan.json")["bootstrap"]
    stored = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    rebuilt_rows: list[dict[str, Any]] = []
    maximum_difference_value = 0.0
    plan_mismatches = 0
    for period in ("assessment", "stress"):
        population = analytic.loc[analytic["period"].eq(period)].copy()
        fresh = (
            episodes["bottom_10_percent"]
            .loc[episodes["bottom_10_percent"]["period"].eq(period)]
            .copy()
        )
        unique_sessions = tuple(sorted(population["session"].astype(str).unique()))
        rng = np.random.default_rng(BOOTSTRAP_SEEDS[period])
        regenerated = [
            [str(value) for value in rng.choice(unique_sessions, size=len(unique_sessions))]
            for _ in range(100)
        ]
        plan_mismatches += sum(
            int(left != right)
            for left, right in zip(regenerated, plan[period]["draws"], strict=True)
        )
        estimate = bootstrap_statistics(population, fresh)
        values = {statistic: [] for statistic in estimate}
        for sampled_sessions in plan[period]["draws"]:
            statistics = bootstrap_statistics(
                resample_sessions(population, sampled_sessions),
                resample_sessions(fresh, sampled_sessions),
            )
            for statistic, value in statistics.items():
                values[statistic].append(value)
        for statistic, draw_values in values.items():
            array = np.asarray(draw_values, dtype=float)
            rebuilt = {
                "period": period,
                "statistic": statistic,
                "estimate": estimate[statistic],
                "draws": 100,
                "seed": BOOTSTRAP_SEEDS[period],
                "lower_80": float(np.quantile(array, 0.10)),
                "upper_80": float(np.quantile(array, 0.90)),
                "lower_90": float(np.quantile(array, 0.05)),
                "upper_90": float(np.quantile(array, 0.95)),
                "lower_95": float(np.quantile(array, 0.025)),
                "upper_95": float(np.quantile(array, 0.975)),
            }
            rebuilt_rows.append(rebuilt)
            actual = stored.loc[
                stored["period"].eq(period) & stored["statistic"].eq(statistic)
            ].iloc[0]
            maximum_difference_value = max(
                maximum_difference_value,
                compare_metric_row(
                    actual,
                    rebuilt,
                    (
                        "estimate",
                        "lower_80",
                        "upper_80",
                        "lower_90",
                        "upper_90",
                        "lower_95",
                        "upper_95",
                    ),
                ),
            )
    require(plan_mismatches == 0, "whole-session bootstrap identities differ")
    require(maximum_difference_value <= TOLERANCE, "whole-session bootstrap metrics differ")
    return pd.DataFrame(rebuilt_rows), {
        "draws_per_period": 100,
        "whole_session_plan_mismatches": plan_mismatches,
        "maximum_bootstrap_metric_difference": maximum_difference_value,
    }


def support_gate(frame: pd.DataFrame, period: str, population: str) -> dict[str, Any]:
    rows = len(frame)
    months = frame["session"].astype(str).str[:7]
    expected_months = (
        {f"2025-{month:02d}" for month in range(1, 9)}
        if period == "assessment"
        else {f"2025-{month:02d}" for month in range(9, 13)}
    )
    if population == "checkpoint" and period == "assessment":
        minimum_rows, minimum_sessions, minimum_stocks = 500, 60, 15
        stock_limit, month_limit, session_limit = 0.15, 0.25, 0.05
    elif population == "checkpoint":
        minimum_rows, minimum_sessions, minimum_stocks = 300, 45, 15
        stock_limit, month_limit, session_limit = 0.15, 0.35, 0.07
    else:
        minimum_rows, minimum_sessions, minimum_stocks = 100, 40, 12
        stock_limit, month_limit, session_limit = 0.20, 0.35, 0.08
    maximum_stock = float(frame.groupby("stock").size().max() / rows)
    maximum_month = float(months.value_counts().max() / rows)
    maximum_session = float(frame.groupby("session").size().max() / rows)
    checks = {
        "minimum_rows": rows >= minimum_rows,
        "minimum_sessions": frame["session"].nunique() >= minimum_sessions,
        "minimum_stocks": frame["stock"].nunique() >= minimum_stocks,
        "every_period_month_represented": expected_months.issubset(set(months)),
        "maximum_stock_share": maximum_stock <= stock_limit,
        "maximum_month_share": maximum_month <= month_limit,
        "maximum_session_share": maximum_session <= session_limit,
    }
    return {
        "period": period,
        "population": population,
        "rows": int(rows),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["stock"].nunique()),
        "months": len(set(months).intersection(expected_months)),
        "maximum_stock_share": maximum_stock,
        "maximum_month_share": maximum_month,
        "maximum_session_share": maximum_session,
        "checks": checks,
        "passed": all(checks.values()),
    }


def audit_support(
    analytic: pd.DataFrame,
    episodes: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    stored = read_json(PRIMARY / "panel_support.json")
    checkpoint_support: dict[str, dict[str, Any]] = {}
    episode_support: dict[str, dict[str, Any]] = {}
    maximum_share_difference = 0.0
    for period in ("assessment", "stress"):
        population = analytic.loc[analytic["period"].eq(period)]
        tail = population.loc[population["m1c_bottom_10_percent"].astype(bool)]
        fresh = episodes["bottom_10_percent"].loc[
            episodes["bottom_10_percent"]["period"].eq(period)
        ]
        checkpoint_support[period] = support_gate(tail, period, "checkpoint")
        episode_support[period] = support_gate(fresh, period, "fresh_episode")
        for population_name, rebuilt in (
            ("checkpoint_bottom_10_percent", checkpoint_support[period]),
            ("fresh_bottom_10_percent", episode_support[period]),
        ):
            actual = stored[population_name][period]
            require(rebuilt["checks"] == actual["checks"], f"{population_name} checks differ")
            require(rebuilt["passed"] == actual["passed"], f"{population_name} result differs")
            require(
                (rebuilt["rows"], rebuilt["sessions"], rebuilt["stocks"], rebuilt["months"])
                == (actual["rows"], actual["sessions"], actual["stocks"], actual["months"]),
                f"{population_name} support counts differ",
            )
            maximum_share_difference = max(
                maximum_share_difference,
                max(
                    abs(float(rebuilt[column]) - float(actual[column]))
                    for column in (
                        "maximum_stock_share",
                        "maximum_month_share",
                        "maximum_session_share",
                    )
                ),
            )
    require(maximum_share_difference <= TOLERANCE, "support concentration shares differ")
    return (
        checkpoint_support,
        episode_support,
        {
            "maximum_support_share_difference": maximum_share_difference,
            "checkpoint_support": checkpoint_support,
            "fresh_episode_support": episode_support,
        },
    )


def containment_summary(
    frame: pd.DataFrame,
    horizon: int,
    multiplier: float,
) -> dict[str, float]:
    available = frame.loc[frame[f"available_{horizon}m"].astype(bool)].copy()
    require(not available.empty, "containment population is empty")
    up = available[f"maximum_up_excursion_{horizon}m"].to_numpy(float)
    down = np.abs(available[f"maximum_down_excursion_{horizon}m"].to_numpy(float))
    boundary = multiplier * available[f"iv_sigma_{horizon}m"].to_numpy(float)
    up_breach = up > boundary
    down_breach = down > boundary
    contained = ~(up_breach | down_breach)
    mass = weights(available)
    return {
        "containment_rate": float(np.average(contained.astype(float), weights=mass)),
        "one_sided_breach_rate": float(
            np.average((up_breach ^ down_breach).astype(float), weights=mass)
        ),
        "two_sided_breach_rate": float(
            np.average((up_breach & down_breach).astype(float), weights=mass)
        ),
        "any_breach_rate": float(np.average((up_breach | down_breach).astype(float), weights=mass)),
    }


def audit_containment(
    analytic: pd.DataFrame,
    episodes: Mapping[str, pd.DataFrame],
    m0_episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stored = pd.read_csv(PRIMARY / "containment_metrics.csv")
    rows: list[dict[str, Any]] = []
    maximum_metric_difference = 0.0
    for period in ("assessment", "stress"):
        populations = {
            "full_checkpoint_population": analytic.loc[analytic["period"].eq(period)],
            "M1C_bottom_10_fresh_episode": episodes["bottom_10_percent"].loc[
                episodes["bottom_10_percent"]["period"].eq(period)
            ],
            "M0_bottom_10_fresh_episode": m0_episodes.loc[m0_episodes["period"].eq(period)],
        }
        for population_name, frame in populations.items():
            for horizon in (15, 30, 60):
                for multiplier in (1.0, 1.5, 2.0):
                    metrics = containment_summary(frame, horizon, multiplier)
                    row = {
                        "period": period,
                        "population": population_name,
                        "horizon_minutes": horizon,
                        "sigma_boundary": multiplier,
                        **metrics,
                    }
                    rows.append(row)
                    actual = stored.loc[
                        stored["period"].eq(period)
                        & stored["population"].eq(population_name)
                        & stored["horizon_minutes"].eq(horizon)
                        & stored["sigma_boundary"].eq(multiplier)
                    ].iloc[0]
                    maximum_metric_difference = max(
                        maximum_metric_difference,
                        compare_metric_row(actual, metrics, tuple(metrics)),
                    )
    require(maximum_metric_difference <= TOLERANCE, "containment metrics differ")
    return pd.DataFrame(rows), {
        "rows_recomputed": len(rows),
        "maximum_containment_metric_difference": maximum_metric_difference,
    }


def decile_direction(analytic: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    correct: list[bool] = []
    for period in ("assessment", "stress"):
        frame = analytic.loc[analytic["period"].eq(period)]
        rates: list[float] = []
        candidate_weights: list[float] = []
        for decile in range(1, 11):
            decile_rows = frame.loc[frame["m1c_score_decile"].eq(decile)]
            rates.append(weighted_mean(decile_rows, "movement_exceeds_iv_15m"))
            candidate_weights.append(float(decile_rows["row_weight"].sum()))
        x = np.arange(1, 11, dtype=float)
        y = np.asarray(rates, dtype=float)
        mass = np.asarray(candidate_weights, dtype=float)
        ranked = rankdata(y, method="average").astype(float)
        x_mean = float(np.average(x, weights=mass))
        rank_mean = float(np.average(ranked, weights=mass))
        covariance = float(np.average((x - x_mean) * (ranked - rank_mean), weights=mass))
        denominator = math.sqrt(
            float(np.average((x - x_mean) ** 2, weights=mass))
            * float(np.average((ranked - rank_mean) ** 2, weights=mass))
        )
        spearman = covariance / denominator
        slope = float(np.polyfit(x, y, 1, w=np.sqrt(mass))[0])
        adjacent = int(np.count_nonzero(np.diff(y) >= 0.0))
        direction = bool(spearman > 0.0 and slope > 0.0 and y[-1] - y[0] > 0.0 and adjacent >= 5)
        correct.append(direction)
        diagnostics[period] = {
            "weighted_spearman": spearman,
            "linear_slope": slope,
            "monotonic_adjacent_steps": adjacent,
            "top_minus_bottom_decile": float(y[-1] - y[0]),
            "correct_overall_direction": direction,
        }
    return all(correct), diagnostics


def low_movement_gate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for period in ("assessment", "stress"):
        values = evidence[period]
        prefix = f"{period}_"
        checks[f"{prefix}remains_below_at_least_85_percent"] = (
            values["remains_below_iv_rate"] >= 0.85
        )
        checks[f"{prefix}npv_lift_at_least_8_points"] = values["npv_lift"] >= 0.08
        checks[f"{prefix}mean_residual_negative"] = values["mean_iv_residual"] < 0.0
        checks[f"{prefix}median_residual_negative"] = values["median_iv_residual"] < 0.0
        checks[f"{prefix}bootstrap_80_npv_lift_lower_above_zero"] = (
            values["bootstrap_80_npv_lift_lower"] > 0.0
        )
        checks[f"{prefix}bootstrap_80_mean_residual_upper_below_zero"] = (
            values["bootstrap_80_mean_residual_upper"] < 0.0
        )
        for name in (
            "m1c_beats_m0_remains_below",
            "m1c_beats_m0_mean_residual",
            "m1c_beats_m0_1_5_sigma_breach",
            "support_and_concentration",
            "not_dependent_on_one_stock",
        ):
            checks[f"{prefix}{name}"] = bool(values[name])
        checks[f"{prefix}matched_npv_lift_wins"] = values["matched_npv_lift_wins"] >= 18
        checks[f"{prefix}matched_mean_residual_wins"] = values["matched_mean_residual_wins"] >= 18
        checks[f"{prefix}permutation_npv_lift_wins"] = values["permutation_npv_lift_wins"] >= 9
        checks[f"{prefix}permutation_mean_residual_wins"] = (
            values["permutation_mean_residual_wins"] >= 9
        )
        checks[f"{prefix}monthly_negative_residual_support"] = (
            values["negative_residual_months"] >= values["required_negative_residual_months"]
        )
    checks["score_decile_direction_correct"] = bool(evidence["score_decile_direction_correct"])
    checks["protected_boundary_passed"] = bool(evidence["protected_boundary_passed"])
    checks["chronology_audit_passed"] = bool(evidence["chronology_audit_passed"])
    return {"checks": checks, "passed": all(checks.values())}


def readiness_gate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "binding_low_movement_veto_passed": bool(evidence["veto_gate_passed"]),
        "surprise_movers_not_concentrated": bool(evidence["surprise_movers_not_concentrated"]),
        "thirty_minute_containment_favourable": bool(
            evidence["thirty_minute_containment_favourable"]
        ),
    }
    for period in ("assessment", "stress"):
        values = evidence[period]
        prefix = f"{period}_"
        for name in (
            "fresh_1_5_sigma_lower_than_full",
            "fresh_1_5_sigma_lower_than_m0",
            "fresh_2_sigma_lower_than_full",
            "support_passed",
        ):
            checks[f"{prefix}{name}"] = bool(values[name])
        checks[f"{prefix}bootstrap_80_1_5_sigma_difference_upper_below_zero"] = (
            values["bootstrap_80_1_5_sigma_difference_upper"] < 0.0
        )
        checks[f"{prefix}two_sigma_containment_at_least_80_percent"] = (
            values["two_sigma_containment_rate"] >= 0.80
        )
    return {"checks": checks, "passed": all(checks.values())}


def table_value(
    frame: pd.DataFrame,
    *,
    column: str,
    **matches: object,
) -> float:
    mask = pd.Series(True, index=frame.index)
    for match_column, value in matches.items():
        mask &= frame[match_column].eq(value)
    selected = frame.loc[mask]
    require(len(selected) == 1, f"expected one row for {matches}")
    return float(selected.iloc[0][column])


def audit_gates_and_decision(
    analytic: pd.DataFrame,
    episodes: Mapping[str, pd.DataFrame],
    containment: pd.DataFrame,
    bootstrap: pd.DataFrame,
    matched_wins: Mapping[str, Mapping[str, int]],
    permutation_wins: Mapping[str, Mapping[str, int]],
    checkpoint_support: Mapping[str, Mapping[str, Any]],
    episode_support: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    decision = read_json(PRIMARY / "decision.json")
    monotonic, monotonic_diagnostics = decile_direction(analytic)
    gate_evidence: dict[str, Any] = {
        "score_decile_direction_correct": monotonic,
        "protected_boundary_passed": True,
        "chronology_audit_passed": True,
    }
    descriptive: list[bool] = []
    comparison_supported: list[bool] = []
    for period in ("assessment", "stress"):
        population = analytic.loc[analytic["period"].eq(period)]
        tail = population.loc[population["m1c_bottom_10_percent"].astype(bool)]
        m0_tail = population.loc[population["m0_bottom_10_percent"].astype(bool)]
        baseline = summary_metrics(population)
        m1c = summary_metrics(tail)
        m0 = summary_metrics(m0_tail)
        monthly_negative = sum(
            summary_metrics(frame)["mean_iv_residual"] < 0.0
            for _, frame in tail.groupby(tail["session"].astype(str).str[:7], sort=True)
        )
        leave_one_out_passed = True
        for stock in sorted(population["stock"].astype(str).unique()):
            reduced = population.loc[population["stock"].astype(str).ne(stock)]
            reduced_tail = reduced.loc[reduced["m1c_bottom_10_percent"].astype(bool)]
            reduced_full_metrics = summary_metrics(reduced)
            reduced_tail_metrics = summary_metrics(reduced_tail)
            leave_one_out_passed &= bool(
                reduced_tail_metrics["remains_below_iv_rate"]
                - reduced_full_metrics["remains_below_iv_rate"]
                > 0.0
                and reduced_tail_metrics["mean_iv_residual"] < 0.0
            )
        gate_evidence[period] = {
            "remains_below_iv_rate": m1c["remains_below_iv_rate"],
            "npv_lift": m1c["remains_below_iv_rate"] - baseline["remains_below_iv_rate"],
            "mean_iv_residual": m1c["mean_iv_residual"],
            "median_iv_residual": m1c["median_iv_residual"],
            "bootstrap_80_npv_lift_lower": table_value(
                bootstrap,
                column="lower_80",
                period=period,
                statistic="bottom_tail_npv_lift",
            ),
            "bootstrap_80_mean_residual_upper": table_value(
                bootstrap,
                column="upper_80",
                period=period,
                statistic="mean_terminal_iv_residual",
            ),
            "m1c_beats_m0_remains_below": (
                m1c["remains_below_iv_rate"] > m0["remains_below_iv_rate"]
            ),
            "m1c_beats_m0_mean_residual": (m1c["mean_iv_residual"] < m0["mean_iv_residual"]),
            "m1c_beats_m0_1_5_sigma_breach": (
                m1c["breach_1_5_sigma_rate"] < m0["breach_1_5_sigma_rate"]
            ),
            "matched_npv_lift_wins": matched_wins[period]["npv_lift"],
            "matched_mean_residual_wins": matched_wins[period]["mean_iv_residual"],
            "permutation_npv_lift_wins": permutation_wins[period]["npv_lift"],
            "permutation_mean_residual_wins": permutation_wins[period]["mean_iv_residual"],
            "negative_residual_months": int(monthly_negative),
            "required_negative_residual_months": 6 if period == "assessment" else 3,
            "support_and_concentration": bool(checkpoint_support[period]["passed"]),
            "not_dependent_on_one_stock": leave_one_out_passed,
        }
        descriptive.append(
            gate_evidence[period]["npv_lift"] > 0.0
            and gate_evidence[period]["mean_iv_residual"] < 0.0
        )
        comparison_supported.append(
            gate_evidence[period]["m1c_beats_m0_remains_below"]
            and gate_evidence[period]["m1c_beats_m0_mean_residual"]
            and gate_evidence[period]["m1c_beats_m0_1_5_sigma_breach"]
        )
    veto = low_movement_gate(gate_evidence)
    readiness_evidence: dict[str, Any] = {"veto_gate_passed": veto["passed"]}
    thirty_favourable = True
    surprise_not_concentrated = True
    for period in ("assessment", "stress"):
        fresh = episodes["bottom_10_percent"].loc[
            episodes["bottom_10_percent"]["period"].eq(period)
        ]
        surprises = fresh.loc[fresh["excursion_sigma_ratio_15m"].ge(1.5)]
        maximum_stock_share = (
            float(surprises.groupby("stock").size().max() / len(surprises))
            if len(surprises)
            else 0.0
        )
        month = surprises["session"].astype(str).str[:7]
        maximum_month_share = (
            float(month.value_counts().max() / len(surprises)) if len(surprises) else 0.0
        )
        surprise_not_concentrated &= maximum_stock_share <= 0.50 and maximum_month_share <= 0.60
        fresh_15_1_5 = table_value(
            containment,
            column="any_breach_rate",
            period=period,
            population="M1C_bottom_10_fresh_episode",
            horizon_minutes=15,
            sigma_boundary=1.5,
        )
        m0_15_1_5 = table_value(
            containment,
            column="any_breach_rate",
            period=period,
            population="M0_bottom_10_fresh_episode",
            horizon_minutes=15,
            sigma_boundary=1.5,
        )
        full_15_1_5 = table_value(
            containment,
            column="any_breach_rate",
            period=period,
            population="full_checkpoint_population",
            horizon_minutes=15,
            sigma_boundary=1.5,
        )
        fresh_15_2 = table_value(
            containment,
            column="any_breach_rate",
            period=period,
            population="M1C_bottom_10_fresh_episode",
            horizon_minutes=15,
            sigma_boundary=2.0,
        )
        full_15_2 = table_value(
            containment,
            column="any_breach_rate",
            period=period,
            population="full_checkpoint_population",
            horizon_minutes=15,
            sigma_boundary=2.0,
        )
        readiness_evidence[period] = {
            "fresh_1_5_sigma_lower_than_full": fresh_15_1_5 < full_15_1_5,
            "fresh_1_5_sigma_lower_than_m0": fresh_15_1_5 < m0_15_1_5,
            "fresh_2_sigma_lower_than_full": fresh_15_2 < full_15_2,
            "bootstrap_80_1_5_sigma_difference_upper": table_value(
                bootstrap,
                column="upper_80",
                period=period,
                statistic="fresh_episode_1_5_sigma_breach_difference_vs_full",
            ),
            "two_sigma_containment_rate": table_value(
                containment,
                column="containment_rate",
                period=period,
                population="M1C_bottom_10_fresh_episode",
                horizon_minutes=15,
                sigma_boundary=2.0,
            ),
            "support_passed": bool(episode_support[period]["passed"]),
        }
        fresh_30 = table_value(
            containment,
            column="any_breach_rate",
            period=period,
            population="M1C_bottom_10_fresh_episode",
            horizon_minutes=30,
            sigma_boundary=1.5,
        )
        m0_30 = table_value(
            containment,
            column="any_breach_rate",
            period=period,
            population="M0_bottom_10_fresh_episode",
            horizon_minutes=30,
            sigma_boundary=1.5,
        )
        full_30 = table_value(
            containment,
            column="any_breach_rate",
            period=period,
            population="full_checkpoint_population",
            horizon_minutes=30,
            sigma_boundary=1.5,
        )
        thirty_favourable &= fresh_30 < m0_30 and fresh_30 < full_30
    readiness_evidence["surprise_movers_not_concentrated"] = surprise_not_concentrated
    readiness_evidence["thirty_minute_containment_favourable"] = thirty_favourable
    readiness = readiness_gate(readiness_evidence)
    require(
        veto["checks"] == decision["binding_low_movement_veto_gate"]["checks"],
        "independent veto-gate checks differ",
    )
    require(
        readiness["checks"] == decision["short_premium_readiness_gate"]["checks"],
        "independent readiness-gate checks differ",
    )
    checkpoint_passed = all(
        checkpoint_support[period]["passed"] for period in ("assessment", "stress")
    )
    episode_passed = all(episode_support[period]["passed"] for period in ("assessment", "stress"))
    if not checkpoint_passed:
        overall = "blocked_insufficient_low_tail_support"
    elif not episode_passed:
        overall = "blocked_insufficient_fresh_quiet_episode_support"
    elif veto["passed"] and readiness["passed"]:
        overall = "m1c_low_movement_veto_supported_and_short_premium_recording_prioritised"
    elif veto["passed"]:
        overall = "m1c_low_movement_veto_supported_short_premium_readiness_unproven"
    elif all(descriptive):
        overall = "m1c_bottom_tail_below_iv_descriptive_only"
    else:
        overall = "m1c_low_movement_veto_not_supported"
    expected_statuses = {
        "m1c_reconstruction_status": "supported",
        "low_tail_threshold_status": "supported",
        "checkpoint_low_movement_status": (
            "supported"
            if veto["passed"]
            else ("promising" if all(descriptive) else "not_supported")
        ),
        "fresh_quiet_episode_status": "supported" if episode_passed else "insufficient_support",
        "m0_comparison_status": ("supported" if all(comparison_supported) else "not_supported"),
        "score_monotonicity_status": "supported" if monotonic else "not_supported",
        "surprise_mover_status": (
            "supported"
            if all(
                readiness_evidence[period]["fresh_1_5_sigma_lower_than_full"]
                for period in ("assessment", "stress")
            )
            and surprise_not_concentrated
            else "descriptive_only"
        ),
        "range_containment_status": (
            "supported"
            if readiness["passed"]
            else (
                "promising"
                if all(
                    readiness_evidence[period]["two_sigma_containment_rate"] >= 0.80
                    for period in ("assessment", "stress")
                )
                else "not_supported"
            )
        ),
        "long_premium_veto_status": (
            "supported"
            if veto["passed"]
            else ("descriptive_only" if all(descriptive) else "not_supported")
        ),
        "short_premium_recorder_priority": (
            "supported" if readiness["passed"] else "not_supported"
        ),
    }
    require(overall == decision["overall_decision"], "independent final decision differs")
    require(
        expected_statuses == {key: decision[key] for key in STATUS_KEYS},
        "independent component statuses differ",
    )
    return {
        "score_monotonicity": monotonic_diagnostics,
        "veto_gate": veto,
        "readiness_gate": readiness,
        "overall_decision": overall,
        "component_statuses": expected_statuses,
    }


def audit_claims_and_protected_boundary(frames: Sequence[pd.DataFrame]) -> dict[str, Any]:
    contract = read_json(EXPERIMENT_DIR / "contract.json")
    decision = read_json(PRIMARY / "decision.json")
    prospective_paths = [
        path
        for path in (
            PRIMARY / "prospective_short_premium_recording_contract.json",
            PRIMARY / "prospective_short_premium_recording_blocker.json",
        )
        if path.exists()
    ]
    require(len(prospective_paths) == 1, "exactly one prospective artifact must exist")
    prospective = read_json(prospective_paths[0])
    for name, artifact in (
        ("contract", contract),
        ("decision", decision),
        ("prospective artifact", prospective),
    ):
        mismatches = {
            key: (artifact.get(key), value)
            for key, value in CLAIMS.items()
            if artifact.get(key) != value
        }
        require(not mismatches, f"{name} claims differ: {mismatches}")
    maximum_session = max(
        str(frame["session"].astype(str).max())
        for frame in frames
        if "session" in frame and not frame.empty
    )
    require(maximum_session < "2026-01-01", "a generated artifact contains protected outcomes")
    protected = read_json(PRIMARY / "protected_boundary_audit.json")
    require(bool(protected["passed"]), "protected-boundary artifact does not pass")
    require(int(protected["protected_rows_read"]) == 0, "protected rows were read")
    return {
        "contract_claims_passed": True,
        "prospective_artifact": str(prospective_paths[0].relative_to(REPO_ROOT)),
        "maximum_generated_session": maximum_session,
        "protected_rows_read": 0,
    }


def update_pass_artifacts(audit: Mapping[str, Any]) -> None:
    decision_path = PRIMARY / "decision.json"
    decision = read_json(decision_path)
    decision["independent_audit_status"] = "supported"
    decision["independent_audit_path"] = str(
        (PRIMARY / "independent_audit.json").relative_to(REPO_ROOT)
    )
    write_json(decision_path, decision)
    lightweight_path = PRIMARY / "lightweight_audit.json"
    lightweight = read_json(lightweight_path)
    lightweight["independent_audit_status"] = "supported"
    lightweight["independent_audit_passed"] = True
    lightweight["passed"] = bool(lightweight.get("passed", False) and audit["passed"])
    write_json(lightweight_path, lightweight)
    report_path = PRIMARY / "report.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "\n## Independent audit\n"
    if marker in report:
        report = report.split(marker, maxsplit=1)[0].rstrip() + "\n"
    report += (
        "\n## Independent audit\n\n"
        "- Status: `supported`.\n"
        "- Manually reconstructed 100 M1C/M0 probabilities, 100 bottom-tail memberships, "
        "100 fifteen-minute outcomes/excursions/IV residuals, and 50 fresh quiet episodes.\n"
        "- Independently rebuilt all fixed null identities, 100 whole-session bootstrap draws "
        "per period, both binding gates, and the final decision with no unexplained discrepancy.\n"
    )
    report_path.write_text(report, encoding="utf-8")


def write_failure_artifacts(detail: str) -> None:
    failure = {
        **CLAIMS,
        "status": "blocked",
        "passed": False,
        "overall_decision": "blocked_reproducibility_or_audit_failure",
        "discrepancy": detail,
    }
    write_json(PRIMARY / "independent_audit.json", failure)
    if (PRIMARY / "decision.json").exists():
        decision = read_json(PRIMARY / "decision.json")
        decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
        decision["independent_audit_status"] = "blocked"
        decision["audit_blocker"] = detail
        for key in STATUS_KEYS:
            decision[key] = "blocked"
        write_json(PRIMARY / "decision.json", decision)
    write_json(
        PRIMARY / "prospective_short_premium_recording_blocker.json",
        {
            **CLAIMS,
            "authorisation": "blocked",
            "overall_decision": "blocked_reproducibility_or_audit_failure",
            "blocker": detail,
        },
    )
    contract_path = PRIMARY / "prospective_short_premium_recording_contract.json"
    if contract_path.exists():
        contract_path.unlink()


def run_audit() -> dict[str, Any]:
    required = (
        "contract.json",
        "source_manifest.json",
        "protected_boundary_audit.json",
        "m1c_reconstruction.json",
        "m1c_feature_manifest.json",
        "m1c_probability_comparison.csv",
        "panel_support.json",
        "frozen_low_tail_thresholds.json",
        "frozen_score_deciles.json",
        "checkpoint_predictions.parquet",
        "fresh_quiet_episodes.parquet",
        "movement_outcomes.parquet",
        "path_excursion_outcomes.parquet",
        "matched_random_null_metrics.csv",
        "probability_permutation_null_metrics.csv",
        "bootstrap_metrics.csv",
        "containment_metrics.csv",
        "decision.json",
        "determinism_check.json",
    )
    missing = [
        name
        for name in required
        if not ((EXPERIMENT_DIR / name).exists() or (PRIMARY / name).exists())
    ]
    require(not missing, f"required audit artifacts are missing: {missing}")
    source_manifest = read_json(PRIMARY / "source_manifest.json")
    source_audit = audit_sources(source_manifest)
    analytic, predictions, movement, stored_episodes = load_analytic()
    paths = pd.read_parquet(PRIMARY / "path_excursion_outcomes.parquet")
    claims_audit = audit_claims_and_protected_boundary(
        (predictions, movement, paths, stored_episodes)
    )
    manifest, probability_audit = audit_manifest_and_probabilities(predictions)
    threshold_audit = audit_thresholds(predictions)
    bars = load_unprotected_bars(completed_bar_source(source_manifest))
    chronology_audit = audit_chronology(predictions, bars)
    outcome_audit = audit_outcomes(predictions, movement, paths, bars)
    episodes, m0_episodes, episode_audit = audit_episodes(analytic, stored_episodes)
    matched_wins, permutation_wins, null_audit = audit_nulls(analytic)
    bootstrap, bootstrap_audit = audit_bootstrap(analytic, episodes)
    checkpoint_support, episode_support, support_audit = audit_support(analytic, episodes)
    containment, containment_audit = audit_containment(analytic, episodes, m0_episodes)
    gate_audit = audit_gates_and_decision(
        analytic,
        episodes,
        containment,
        bootstrap,
        matched_wins,
        permutation_wins,
        checkpoint_support,
        episode_support,
    )
    determinism = read_json(PRIMARY / "determinism_check.json")
    require(bool(determinism["passed"]), "runner determinism artifact does not pass")
    reconstruction = read_json(PRIMARY / "m1c_reconstruction.json")
    require(reconstruction["status"] == "supported", "M1C reconstruction is not supported")
    artifact = {
        **CLAIMS,
        "status": "supported",
        "passed": True,
        "audit_implementation": "independent_no_runner_import",
        "source_audit": source_audit,
        "claims_and_protected_boundary_audit": claims_audit,
        "feature_manifest_audit": {
            "feature_order": manifest["feature_order"],
            "contaminated_features_excluded": (
                manifest["contaminated_and_peer_normalised_features_excluded"]
            ),
            "coefficients_and_preprocessing_match_predecessor": True,
        },
        "probability_audit": probability_audit,
        "threshold_and_membership_audit": threshold_audit,
        "chronology_audit": chronology_audit,
        "outcome_and_excursion_audit": outcome_audit,
        "fresh_quiet_episode_audit": episode_audit,
        "null_audit": null_audit,
        "bootstrap_audit": bootstrap_audit,
        "support_gate_audit": support_audit,
        "containment_audit": containment_audit,
        "gate_and_decision_audit": gate_audit,
        "determinism_artifact_passed": True,
        "unexplained_discrepancies": 0,
    }
    write_json(PRIMARY / "independent_audit.json", artifact)
    update_pass_artifacts(artifact)
    return {
        "status": artifact["status"],
        "passed": artifact["passed"],
        "overall_decision": gate_audit["overall_decision"],
        "manual_m1c_probabilities": probability_audit["manual_m1c_probabilities"],
        "manual_fifteen_minute_outcomes": outcome_audit["manual_fifteen_minute_outcomes"],
        "manual_fresh_quiet_episodes": episode_audit["manual_fresh_quiet_episodes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="run the independent audit")
    arguments = parser.parse_args()
    if not arguments.audit:
        parser.error("--audit is required")
    try:
        result = run_audit()
    except (AuditFailure, KeyError, ValueError, TypeError, OSError) as error:
        write_failure_artifacts(str(error))
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "decision": "blocked_reproducibility_or_audit_failure",
                    "detail": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
