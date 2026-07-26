#!/usr/bin/env python3
"""Independently audit Movement-Qualified Directional Readiness V0."""

from __future__ import annotations

# ruff: noqa: E402 -- deterministic numerical limits precede imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import hashlib
import importlib.util
import json
import math
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings(
    "ignore",
    message="'penalty' was deprecated.*",
    category=FutureWarning,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
V0_RUNNER = (
    REPO_ROOT
    / "research/options-feasibility/20260723-minimal-intraday-iv-excess-holdout-v0"
    / "run_screen_v0.py"
)
M1_THRESHOLD = 0.49588519865576763
HORIZONS = (5, 10, 15, 30)
MODEL_IDS = ("D0", "D1", "D2")
NULL_SEEDS = (20260731, 20260732, 20260733, 20260734, 20260735)
BOOTSTRAP_SEED = 20260726
SAFETY_EXPECTED: dict[str, Any] = {
    "research_only": True,
    "retrospective_directional_screen": True,
    "movement_model_frozen": True,
    "movement_model_refit_allowed": False,
    "m1_threshold_frozen": M1_THRESHOLD,
    "direction_model_is_second_stage": True,
    "direction_models_trained_on_2024_only": True,
    "assessment_start": "2025-01-01",
    "assessment_end": "2025-08-22",
    "opened_movement_holdout_excluded": True,
    "rows_from_2026_onward_protected": True,
    "primary_direction_horizon_minutes": 10,
    "call_put_abstain_policy": True,
    "option_pnl_calculated": False,
    "intraday_option_quotes_used": False,
    "broker_access": False,
    "paper_orders_allowed": False,
    "live_orders_allowed": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
REQUIRED_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "movement_model_reconstruction.json",
    "movement_feature_manifest.json",
    "movement_signal_episodes.parquet",
    "episode_construction_audit.json",
    "direction_target_audit.json",
    "direction_feature_manifest.json",
    "orientation_source_audit.json",
    "development_orientation_map.csv",
    "orientation_crossfit_audit.json",
    "model_configurations.json",
    "development_oof_predictions.parquet",
    "frozen_direction_thresholds.json",
    "primary_candidate_freeze.json",
    "assessment_predictions.parquet",
    "direction_model_metrics.csv",
    "selective_policy_metrics.csv",
    "baseline_metrics.csv",
    "monthly_metrics.csv",
    "checkpoint_metrics.csv",
    "route_state_metrics.csv",
    "stock_metrics.csv",
    "material_move_metrics.csv",
    "remaining_movement_metrics.csv",
    "excursion_metrics.csv",
    "bootstrap_metrics.csv",
    "direction_null_metrics.csv",
    "concentration_metrics.csv",
    "layer_attribution.csv",
    "decision.json",
    "lightweight_audit.json",
    "determinism_check.json",
    "report.md",
)


class AuditFailure(RuntimeError):
    """An unexplained independent-audit discrepancy."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    content = (
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise AuditFailure(f"cannot independently load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def assert_close(name: str, actual: float, expected: float, tolerance: float) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected):
        if math.isnan(actual) and math.isnan(expected):
            return
        raise AuditFailure(f"{name}: non-finite mismatch {actual} != {expected}")
    if abs(actual - expected) > tolerance:
        raise AuditFailure(f"{name}: {actual} != {expected} within {tolerance}")


def manual_design(
    specification: Mapping[str, Any], frame: pd.DataFrame
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], tuple[str, ...]]:
    pieces: list[np.ndarray[Any, np.dtype[np.float64]]] = []
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
    categorical_levels = cast(Mapping[str, Sequence[str]], specification["categorical_levels"])
    for column in cast(Sequence[str], specification["categorical_features"]):
        values = frame[column].fillna("__MISSING__").astype(str)
        levels = tuple(str(value) for value in categorical_levels[column])
        values = values.where(values.isin(set(levels)), "__UNKNOWN__")
        for level in levels:
            pieces.append(values.eq(level).to_numpy(float)[:, None])
            names.append(f"{column}=={level}")
    return np.concatenate(pieces, axis=1), tuple(names)


def manual_probabilities(
    specification: Mapping[str, Any], frame: pd.DataFrame
) -> np.ndarray[Any, np.dtype[np.float64]]:
    design, names = manual_design(specification, frame)
    expected_names = tuple(str(value) for value in specification["design_feature_names"])
    if names != expected_names:
        raise AuditFailure("manual model design order differs from the frozen order")
    linear = design @ np.asarray(specification["coefficients"], dtype=float) + float(
        specification["intercept"]
    )
    probability = np.empty(len(linear), dtype=float)
    positive = linear >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
    exponential = np.exp(linear[~positive])
    probability[~positive] = exponential / (1.0 + exponential)
    return probability


def refit_independent_model(
    frame: pd.DataFrame,
    target_column: str,
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    training = frame.loc[frame[target_column].notna()].copy()
    numeric = tuple(str(value) for value in specification["numeric_features"])
    categorical = tuple(str(value) for value in specification["categorical_features"])
    medians: dict[str, float] = {}
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in numeric:
        raw = pd.to_numeric(training[column], errors="coerce").to_numpy(float)
        finite = raw[np.isfinite(raw)]
        median = float(np.median(finite)) if len(finite) else 0.0
        imputed = np.where(np.isfinite(raw), raw, median)
        center = float(np.median(imputed))
        q25, q75 = np.quantile(imputed, [0.25, 0.75])
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        medians[column] = median
        centers[column] = center
        scales[column] = scale
    levels: dict[str, list[str]] = {}
    for column in categorical:
        values = training[column].fillna("__MISSING__").astype(str)
        observed = sorted(set(values).difference({"__MISSING__", "__UNKNOWN__"}))
        levels[column] = [*observed, "__MISSING__", "__UNKNOWN__"]
    independent_spec = {
        "numeric_features": list(numeric),
        "categorical_features": list(categorical),
        "medians": medians,
        "robust_centers": centers,
        "robust_scales": scales,
        "categorical_levels": levels,
    }
    design, names = manual_design(independent_spec, training)
    target = pd.to_numeric(training[target_column], errors="raise").to_numpy(int)
    estimator = LogisticRegression(
        penalty="l2",
        C=0.25,
        solver="liblinear",
        max_iter=300,
        class_weight=None,
        random_state=20260726,
    ).fit(design, target)
    independent_spec.update(
        {
            "design_feature_names": list(names),
            "coefficients": estimator.coef_[0].tolist(),
            "intercept": float(estimator.intercept_[0]),
            "iterations": int(estimator.n_iter_[0]),
        }
    )
    return independent_spec


def reconstruct_episodes(panel: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    source = panel.rename(columns={"symbol": "stock"}).copy()
    source["partition"] = np.where(
        source["session"].astype(str).le("2024-12-31"),
        "development",
        "assessment",
    )
    source["above"] = source["M1_probability"].to_numpy(float) >= M1_THRESHOLD
    source = source.sort_values(["stock", "session", "checkpoint"], kind="mergesort").reset_index(
        drop=True
    )
    source["previous_checkpoint_probability"] = source.groupby(["stock", "session"], sort=False)[
        "M1_probability"
    ].shift()
    source["crossing"] = source["above"] & (
        source["previous_checkpoint_probability"].isna()
        | source["previous_checkpoint_probability"].lt(M1_THRESHOLD)
    )
    state_times = states[
        [
            "stock",
            "session",
            "bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
        ]
    ]
    close_lookup = {
        (str(row.stock), str(row.session), int(row.bar_ordinal) + 1): pd.Timestamp(
            row.bar_complete_timestamp
        )
        for row in state_times.itertuples(index=False)
    }
    entry_lookup = {
        (str(row.stock), str(row.session), int(row.bar_ordinal)): pd.Timestamp(
            row.bar_start_timestamp
        )
        for row in state_times.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for (stock, session), group in source.loc[source["crossing"]].groupby(
        ["stock", "session"], sort=True
    ):
        previous_start: pd.Timestamp | None = None
        episode_number = 0
        for row in group.sort_values("checkpoint").itertuples(index=False):
            key = (str(stock), str(session), int(row.checkpoint))
            signal = close_lookup[key]
            elapsed = (
                math.nan
                if previous_start is None
                else (signal - previous_start).total_seconds() / 60.0
            )
            if math.isfinite(elapsed) and elapsed < 30.0:
                continue
            episode_number += 1
            rows.append(
                {
                    "stock": str(stock),
                    "session": str(session),
                    "checkpoint": int(row.checkpoint),
                    "signal_timestamp": signal,
                    "prospective_entry_timestamp": entry_lookup[key],
                    "m1_probability": float(row.M1_probability),
                    "previous_checkpoint_probability": float(row.previous_checkpoint_probability),
                    "episode_number": episode_number,
                    "minutes_since_previous_episode": elapsed,
                    "partition": str(row.partition),
                }
            )
            previous_start = signal
    return (
        pd.DataFrame(rows)
        .sort_values(["stock", "session", "checkpoint"], kind="mergesort")
        .reset_index(drop=True)
    )


def parse_orientation_path(value: object) -> tuple[int, ...]:
    text = str(value)
    if "__o_" not in text:
        return ()
    return tuple(int(token) for token in text.split("__o_", maxsplit=1)[1].split("-"))


def audit_route_features(
    sample: pd.DataFrame,
    ledger_path: Path,
    orientation_map: pd.DataFrame,
) -> tuple[float, int]:
    ledger = pd.read_parquet(
        ledger_path,
        columns=[
            "ledger_kind",
            "symbol",
            "session",
            "bar_ordinal",
            "semantic_loop_id",
            "orientation_id",
            "progress_states",
            "transitions_remaining",
            "available_timestamp_utc",
        ],
    ).rename(columns={"symbol": "stock"})
    state_sign = {
        int(row.state): int(row.orientation_sign) for row in orientation_map.itertuples(index=False)
    }
    maximum = 0.0
    future_rows = 0
    for episode in sample.itertuples(index=False):
        active = ledger.loc[
            ledger["ledger_kind"].astype(str).eq("active_prefix")
            & ledger["stock"].astype(str).eq(str(episode.stock))
            & ledger["session"].astype(str).eq(str(episode.session))
            & ledger["bar_ordinal"].astype(int).eq(int(episode.checkpoint))
        ].copy()
        future_rows += int(
            (
                pd.to_datetime(active["available_timestamp_utc"], utc=True)
                > pd.Timestamp(episode.signal_timestamp)
            ).sum()
        )
        records: list[tuple[int, float, str]] = []
        for prefix in active.itertuples(index=False):
            path = parse_orientation_path(prefix.orientation_id)
            progress = int(prefix.progress_states)
            remaining = int(prefix.transitions_remaining)
            next_state = path[progress] if progress < len(path) else -1
            sign = state_sign.get(next_state, 0)
            denominator = progress + remaining - 1
            depth = (
                float(np.clip((progress - 1) / denominator, 0.0, 1.0)) if denominator > 0 else 0.0
            )
            records.append((sign, depth, str(prefix.semantic_loop_id)))
        records.sort(key=lambda value: (-value[1], value[2]))
        signs = np.asarray([value[0] for value in records], dtype=int)
        depths = np.asarray([value[1] for value in records], dtype=float)
        positive = int(np.sum(signs > 0))
        negative = int(np.sum(signs < 0))
        neutral = int(np.sum(signs == 0))
        weighted_positive = float(np.sum(depths[signs > 0]))
        weighted_negative = float(np.sum(depths[signs < 0]))
        top = int(signs[0]) if len(signs) else 0
        second = int(signs[1]) if len(signs) > 1 else 0
        nonneutral = positive + negative
        weight_total = weighted_positive + weighted_negative
        margin = (
            (weighted_positive - weighted_negative) / weight_total
            if weight_total > 0.0
            else ((positive - negative) / nonneutral if nonneutral > 0 else 0.0)
        )
        agreement = max(positive, negative) / len(signs) if len(signs) else 0.0
        completed = ledger.loc[
            ledger["ledger_kind"].astype(str).eq("registered_completion")
            & ledger["stock"].astype(str).eq(str(episode.stock))
            & ledger["session"].astype(str).eq(str(episode.session))
            & ledger["bar_ordinal"].astype(int).le(int(episode.checkpoint))
        ]
        completion_records: list[tuple[int, int]] = []
        for completion in completed.itertuples(index=False):
            path = parse_orientation_path(completion.orientation_id)
            completion_records.append(
                (
                    int(completion.bar_ordinal),
                    state_sign.get(path[-1], 0) if path else 0,
                )
            )
        completion_records.sort()
        recent = completion_records[-1][1] if completion_records else 0
        same_memory = sum(
            math.exp(-(int(episode.checkpoint) - ordinal) / 6.0)
            for ordinal, sign in completion_records
            if top and sign == top
        )
        expected = {
            "positive_active_prefix_count": positive,
            "negative_active_prefix_count": negative,
            "neutral_active_prefix_count": neutral,
            "depth_weighted_positive_orientation": weighted_positive,
            "depth_weighted_negative_orientation": weighted_negative,
            "top_route_orientation": top,
            "second_route_orientation": second,
            "orientation_margin": margin,
            "orientation_agreement": agreement,
            "orientation_disagreement": 1.0 - agreement,
            "narrowing_route_orientation": (
                top if str(episode.route_resolution_state) == "NARROWING" else 0
            ),
            "recent_completed_loop_orientation": recent,
            "recent_same_orientation_loop_memory_score": same_memory,
            "dominant_route_pressure_agreement": top * int(np.sign(float(episode.signed_pressure))),
        }
        for column, value in expected.items():
            maximum = max(maximum, abs(float(getattr(episode, column)) - value))
    return maximum, future_rows


def selective_values(frame: pd.DataFrame, action_column: str) -> dict[str, float]:
    actioned = frame.loc[frame[action_column].astype(str).ne("ABSTAIN")]
    actions = actioned[action_column].astype(str).to_numpy()
    returns = actioned["signed_log_return_10m"].to_numpy(float)
    sides = np.where(actions == "CALL", 1, -1)
    valid = np.isfinite(returns) & (returns != 0.0)
    truth = (returns[valid] > 0.0).astype(int)
    predicted = (sides[valid] > 0).astype(int)
    favourable = np.where(
        sides > 0,
        actioned["upside_mfe_10m"].to_numpy(float),
        actioned["downside_mfe_10m"].to_numpy(float),
    )
    adverse = np.where(
        sides > 0,
        actioned["upside_mae_10m"].to_numpy(float),
        actioned["downside_mae_10m"].to_numpy(float),
    )
    aligned = sides * returns
    return {
        "action_coverage": len(actioned) / len(frame),
        "directional_accuracy": float(np.mean(predicted == truth)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "mean_aligned_return": float(np.mean(aligned)),
        "median_aligned_return": float(np.median(aligned)),
        "positive_aligned_return_rate": float(np.mean(aligned > 0.0)),
        "favourable_adverse_excursion_ratio": float(np.mean(favourable)) / float(np.mean(adverse)),
    }


def audit_targets_and_price_features(
    assessment: pd.DataFrame, states: pd.DataFrame
) -> dict[str, float | int]:
    groups = {
        (str(stock), str(session)): group.sort_values("bar_ordinal").set_index("bar_ordinal")
        for (stock, session), group in states.groupby(["stock", "session"], sort=False)
    }
    maximum_target = 0.0
    maximum_d0 = 0.0
    timestamp_mismatches = 0
    for episode in assessment.itertuples(index=False):
        bars = groups[(str(episode.stock), str(episode.session))]
        checkpoint = int(episode.checkpoint)
        signal = bars.loc[checkpoint - 1]
        entry = bars.loc[checkpoint]
        if pd.Timestamp(signal["bar_complete_timestamp"]) != pd.Timestamp(episode.signal_timestamp):
            timestamp_mismatches += 1
        if pd.Timestamp(entry["bar_start_timestamp"]) != pd.Timestamp(
            episode.prospective_entry_timestamp
        ):
            timestamp_mismatches += 1
        entry_price = float(entry["open"])
        expected_targets = {"entry_price": entry_price}
        for horizon in HORIZONS:
            close = float(bars.loc[checkpoint + horizon // 5 - 1, "close"])
            signed = math.log(close / entry_price)
            expected_targets[f"signed_log_return_{horizon}m"] = signed
            expected_targets[f"absolute_log_return_{horizon}m"] = abs(signed)
            expected_targets[f"iv_expected_absolute_{horizon}m"] = (
                float(episode.atm_iv) * math.sqrt(horizon / (252 * 390)) * math.sqrt(2.0 / math.pi)
            )
        before = math.log(float(signal["close"]) / float(bars.loc[0, "open"]))
        gap = math.log(entry_price / float(signal["close"]))
        for horizon in (10, 30):
            after = abs(expected_targets[f"signed_log_return_{horizon}m"])
            denominator = abs(before) + abs(gap) + after
            expected_targets[f"fraction_eventual_{horizon}m_move_after_entry"] = (
                after / denominator if denominator else math.nan
            )
        for column, value in expected_targets.items():
            actual = float(getattr(episode, column))
            if math.isnan(value) and math.isnan(actual):
                continue
            maximum_target = max(maximum_target, abs(actual - value))

        available = bars.loc[bars.index.astype(int) < checkpoint]
        opens = available["open"].to_numpy(float)
        highs = available["high"].to_numpy(float)
        lows = available["low"].to_numpy(float)
        closes = available["close"].to_numpy(float)
        bar_returns = np.log(closes / np.concatenate(([opens[0]], closes[:-1])))
        market_returns = available["vti__bar_log_return"].to_numpy(float)
        current_range = highs[-1] - lows[-1]
        upper_wick = (
            (highs[-1] - max(opens[-1], closes[-1])) / current_range if current_range else 0.0
        )
        lower_wick = (
            (min(opens[-1], closes[-1]) - lows[-1]) / current_range if current_range else 0.0
        )
        expected_d0 = {
            "signed_return_1bar": float(np.sum(bar_returns[-1:])),
            "signed_return_2bar": float(np.sum(bar_returns[-2:])),
            "signed_return_3bar": float(np.sum(bar_returns[-3:])),
            "signed_return_6bar": float(np.sum(bar_returns[-6:])),
            "current_candle_body_direction": math.log(closes[-1] / opens[-1]),
            "wick_imbalance": float(lower_wick - upper_wick),
            "market_return_1bar": float(np.sum(market_returns[-1:])),
            "market_return_2bar": float(np.sum(market_returns[-2:])),
            "market_return_3bar": float(np.sum(market_returns[-3:])),
            "stock_minus_market_return_3bar": float(
                np.sum(bar_returns[-3:]) - np.sum(market_returns[-3:])
            ),
            "market_breadth_direction": float(available.iloc[-1]["market_breadth_bar_positive"]),
        }
        for column, value in expected_d0.items():
            maximum_d0 = max(maximum_d0, abs(float(getattr(episode, column)) - value))
    return {
        "target_rows_checked": int(len(assessment)),
        "maximum_target_difference": maximum_target,
        "maximum_d0_feature_difference": maximum_d0,
        "signal_or_entry_timestamp_mismatches": timestamp_mismatches,
    }


def audit_d1_features(
    assessment: pd.DataFrame,
    historical: pd.DataFrame,
    behaviour: pd.DataFrame,
) -> dict[str, float | int]:
    panel = historical.rename(columns={"symbol": "stock"}).sort_values(
        ["stock", "session", "checkpoint"], kind="mergesort"
    )
    panel_groups = {
        (str(stock), str(session)): group.reset_index(drop=True)
        for (stock, session), group in panel.groupby(["stock", "session"], sort=False)
    }
    exhaustion = {
        (str(row.stock), str(row.session), int(row.checkpoint)): float(row.signed_exhaustion)
        for row in behaviour.itertuples(index=False)
    }
    maximum = 0.0
    nan_mismatches = 0
    for episode in assessment.itertuples(index=False):
        group = panel_groups[(str(episode.stock), str(episode.session))]
        positions = group.index[group["checkpoint"].astype(int).eq(int(episode.checkpoint))]
        if len(positions) != 1:
            raise AuditFailure("D1 source checkpoint identity is not unique")
        position = int(positions[0])
        current = group.iloc[position]
        previous = group.iloc[position - 1] if position else None
        current_exhaustion = exhaustion.get(
            (str(episode.stock), str(episode.session), int(episode.checkpoint)),
            math.nan,
        )
        previous_exhaustion = (
            exhaustion.get(
                (
                    str(episode.stock),
                    str(episode.session),
                    int(previous["checkpoint"]),
                ),
                math.nan,
            )
            if previous is not None
            else math.nan
        )
        pressure = float(current["signed_pressure"])
        memory = float(current["recent_loop_memory_weighted_top_depth"])
        expected = {
            "signed_pressure": pressure,
            "signed_pressure_change": (
                pressure - float(previous["signed_pressure"]) if previous is not None else math.nan
            ),
            "signed_exhaustion": current_exhaustion,
            "signed_exhaustion_change": (
                current_exhaustion - previous_exhaustion
                if math.isfinite(current_exhaustion) and math.isfinite(previous_exhaustion)
                else math.nan
            ),
            "compression_release_direction": float(
                current["raw_component__signed_progress_acceleration"]
            ),
            "pressure_x_conviction": pressure * float(current["conviction"]),
            "pressure_x_tension": pressure * float(current["tension"]),
            "pressure_x_arousal": pressure * float(current["arousal"]),
            "signed_activity_imbalance": float(current["raw_component__return_gap"]),
            "signed_structural_memory": float(np.sign(pressure) * memory),
        }
        for column, value in expected.items():
            actual = float(getattr(episode, column))
            if math.isnan(actual) or math.isnan(value):
                nan_mismatches += int(not (math.isnan(actual) and math.isnan(value)))
            else:
                maximum = max(maximum, abs(actual - value))
    return {
        "rows_checked": int(len(assessment)),
        "maximum_d1_feature_difference": maximum,
        "d1_missingness_mismatches": nan_mismatches,
    }


def audit_model_probabilities_and_thresholds(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    configurations: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    maximum_probability = 0.0
    maximum_coefficient = 0.0
    maximum_preprocessing = 0.0
    maximum_oof_refit_coefficient = 0.0
    maximum_oof_refit_preprocessing = 0.0
    maximum_threshold = 0.0
    oof_rows = 0
    oof_refits = 0
    oof_session_overlap_count = 0
    manual_assessment_rows: dict[str, int] = {}
    fitted_model_ids = tuple(
        model_id
        for model_id in MODEL_IDS
        if bool(cast(Mapping[str, Any], configurations[model_id]).get("fitted", True))
    )
    for model_id in fitted_model_ids:
        configuration = cast(Mapping[str, Any], configurations[model_id])
        full_specification = cast(Mapping[str, Any], configuration["full_development_model"])
        sample = assessment.head(100)
        manual = manual_probabilities(full_specification, sample)
        maximum_probability = max(
            maximum_probability,
            float(np.max(np.abs(manual - sample[f"{model_id}_probability"].to_numpy(float)))),
        )
        manual_assessment_rows[model_id] = len(sample)
        refitted = refit_independent_model(development, "direction_up_10m", full_specification)
        maximum_coefficient = max(
            maximum_coefficient,
            float(
                np.max(
                    np.abs(
                        np.asarray(refitted["coefficients"], dtype=float)
                        - np.asarray(full_specification["coefficients"], dtype=float)
                    )
                )
            ),
            abs(float(refitted["intercept"]) - float(full_specification["intercept"])),
        )
        for field in ("medians", "robust_centers", "robust_scales"):
            expected = cast(Mapping[str, float], full_specification[field])
            actual = cast(Mapping[str, float], refitted[field])
            maximum_preprocessing = max(
                maximum_preprocessing,
                max(abs(float(actual[key]) - float(expected[key])) for key in expected),
            )
        if refitted["categorical_levels"] != full_specification["categorical_levels"]:
            raise AuditFailure(f"{model_id} frozen categorical levels drifted")
        for fold_entry in cast(Sequence[Mapping[str, Any]], configuration["oof_models"]):
            fold = int(fold_entry["fold"])
            training = development.loc[development["fold"].astype(int).ne(fold)]
            held_out = development.loc[development["fold"].astype(int).eq(fold)]
            overlap = set(training["session"].astype(str)).intersection(
                set(held_out["session"].astype(str))
            )
            oof_session_overlap_count += len(overlap)
            if overlap:
                raise AuditFailure(f"{model_id} fold {fold} leaks a held-out session")
            if int(training["session"].nunique()) != int(fold_entry["training_sessions"]):
                raise AuditFailure(f"{model_id} fold {fold} training-session count drifted")
            if int(held_out["session"].nunique()) != int(fold_entry["held_out_sessions"]):
                raise AuditFailure(f"{model_id} fold {fold} held-out-session count drifted")
            if str(held_out["session"].min()) != str(fold_entry["held_out_start"]) or str(
                held_out["session"].max()
            ) != str(fold_entry["held_out_end"]):
                raise AuditFailure(f"{model_id} fold {fold} held-out calendar block drifted")
            specification = cast(Mapping[str, Any], fold_entry["specification"])
            fold_refit = refit_independent_model(
                training,
                "direction_up_10m",
                specification,
            )
            maximum_oof_refit_coefficient = max(
                maximum_oof_refit_coefficient,
                float(
                    np.max(
                        np.abs(
                            np.asarray(fold_refit["coefficients"], dtype=float)
                            - np.asarray(specification["coefficients"], dtype=float)
                        )
                    )
                ),
                abs(float(fold_refit["intercept"]) - float(specification["intercept"])),
            )
            for field in ("medians", "robust_centers", "robust_scales"):
                expected = cast(Mapping[str, float], specification[field])
                actual = cast(Mapping[str, float], fold_refit[field])
                maximum_oof_refit_preprocessing = max(
                    maximum_oof_refit_preprocessing,
                    max(abs(float(actual[key]) - float(expected[key])) for key in expected),
                )
            if fold_refit["categorical_levels"] != specification["categorical_levels"]:
                raise AuditFailure(f"{model_id} fold {fold} categorical levels drifted")
            probabilities = manual_probabilities(specification, held_out)
            maximum_probability = max(
                maximum_probability,
                float(
                    np.max(
                        np.abs(probabilities - held_out[f"{model_id}_probability"].to_numpy(float))
                    )
                ),
            )
            oof_rows += len(held_out)
            oof_refits += 1
        confidence = np.abs(development[f"{model_id}_probability"].to_numpy(float) - 0.5)
        order = np.lexsort((np.arange(len(confidence)), -confidence))
        required_actions = max(150, math.ceil(0.35 * len(confidence)))
        boundary = float(confidence[order[required_actions - 1]])
        stored_boundary = float(cast(Mapping[str, Any], thresholds[model_id])["boundary"])
        maximum_threshold = max(maximum_threshold, abs(boundary - stored_boundary))
    return {
        "manual_assessment_rows_per_model": manual_assessment_rows,
        "manual_oof_rows": oof_rows,
        "independent_oof_refits": oof_refits,
        "expected_oof_refits": 4 * len(fitted_model_ids),
        "fitted_model_ids": list(fitted_model_ids),
        "oof_session_overlap_count": oof_session_overlap_count,
        "maximum_probability_difference": maximum_probability,
        "maximum_refit_coefficient_difference": maximum_coefficient,
        "maximum_preprocessing_difference": maximum_preprocessing,
        "maximum_oof_refit_coefficient_difference": maximum_oof_refit_coefficient,
        "maximum_oof_refit_preprocessing_difference": maximum_oof_refit_preprocessing,
        "maximum_threshold_difference": maximum_threshold,
    }


def audit_bootstrap(
    assessment: pd.DataFrame, stored: pd.DataFrame, primary_candidate: str
) -> dict[str, float | int]:
    sessions = np.asarray(sorted(assessment["session"].astype(str).unique()), dtype=object)
    groups = {str(session): group for session, group in assessment.groupby("session", sort=False)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    reconstructed: dict[str, list[float]] = {}
    maximum = 0.0
    for draw in range(100):
        sampled_sessions = rng.choice(sessions, size=len(sessions), replace=True).astype(str)
        stored_identity = stored.loc[
            stored["record_type"].astype(str).eq("sample_identity")
            & pd.to_numeric(stored["draw"], errors="coerce").eq(draw),
            "sampled_sessions_json",
        ]
        if len(stored_identity) != 1 or json.loads(str(stored_identity.iloc[0])) != list(
            sampled_sessions
        ):
            raise AuditFailure(f"bootstrap sample identity mismatch: {draw}")
        sample = pd.concat([groups[session] for session in sampled_sessions], ignore_index=True)
        binary: dict[str, dict[str, float]] = {}
        valid = sample["direction_up_10m"].notna()
        labels = sample.loc[valid, "direction_up_10m"].to_numpy(int)
        for model_id in MODEL_IDS:
            probabilities = sample.loc[valid, f"{model_id}_probability"].to_numpy(float)
            binary[model_id] = {
                "auc": float(roc_auc_score(labels, probabilities)),
                "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
            }
        selective = selective_values(sample, f"{primary_candidate}_action")
        values = {
            **{f"{model_id}_auc": binary[model_id]["auc"] for model_id in MODEL_IDS},
            **{f"{model_id}_log_loss": binary[model_id]["log_loss"] for model_id in MODEL_IDS},
            "D1_minus_D0_log_loss_improvement": (
                binary["D0"]["log_loss"] - binary["D1"]["log_loss"]
            ),
            "D2_minus_D1_log_loss_improvement": (
                binary["D1"]["log_loss"] - binary["D2"]["log_loss"]
            ),
            "D2_minus_D0_auc_improvement": (binary["D2"]["auc"] - binary["D0"]["auc"]),
            **{f"selective_{key}": value for key, value in selective.items()},
        }
        for metric, value in values.items():
            reconstructed.setdefault(metric, []).append(value)
            match = stored.loc[
                stored["record_type"].eq("draw")
                & stored["draw"].eq(draw)
                & stored["metric"].eq(metric),
                "value",
            ]
            if len(match) != 1:
                raise AuditFailure(f"bootstrap draw missing: {draw} {metric}")
            maximum = max(maximum, abs(value - float(match.iloc[0])))
    maximum_interval = 0.0
    for metric, values in reconstructed.items():
        array = np.asarray(values, dtype=float)
        for confidence in (0.80, 0.90, 0.95):
            alpha = 0.5 * (1.0 - confidence)
            lower, upper = np.nanquantile(array, [alpha, 1.0 - alpha])
            match = stored.loc[
                stored["record_type"].eq("interval")
                & stored["metric"].eq(metric)
                & np.isclose(stored["confidence_level"], confidence)
            ]
            if len(match) != 1:
                raise AuditFailure(f"bootstrap interval missing: {metric} {confidence}")
            maximum_interval = max(
                maximum_interval,
                abs(float(lower) - float(match.iloc[0]["lower"])),
                abs(float(upper) - float(match.iloc[0]["upper"])),
            )
    return {
        "draws_checked": 100,
        "maximum_bootstrap_draw_difference": maximum,
        "maximum_bootstrap_interval_difference": maximum_interval,
    }


def independent_permutation(frame: pd.DataFrame, seed: int) -> pd.Series[Any]:
    output = frame["direction_up_10m"].copy()
    rng = np.random.default_rng(seed)
    groups = frame.groupby(["session", "checkpoint_group"], sort=True, dropna=False).groups
    for indices in groups.values():
        positions = list(indices)
        output.loc[positions] = rng.permutation(
            frame.loc[positions, "direction_up_10m"].to_numpy(copy=True)
        )
    return output


def audit_nulls(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    stored: pd.DataFrame,
    thresholds: Mapping[str, Any],
) -> dict[str, float | int]:
    maximum_coefficient = 0.0
    maximum_metric = 0.0
    fitted_models = 0
    labels = assessment["direction_up_10m"].to_numpy(float)
    valid = np.isfinite(labels)
    labels_valid = labels[valid].astype(int)
    for null_index, seed in enumerate(NULL_SEEDS):
        permuted = development.copy()
        permuted["null_direction_up_10m"] = independent_permutation(permuted, seed)
        expected_labels = [
            None if pd.isna(value) else int(value)
            for value in permuted["null_direction_up_10m"].tolist()
        ]
        stored_labels = stored.loc[
            stored["null_refit"].astype(int).eq(null_index),
            "permuted_labels_json",
        ].drop_duplicates()
        if len(stored_labels) != 1 or json.loads(str(stored_labels.iloc[0])) != expected_labels:
            raise AuditFailure(f"null label-slate identity mismatch {null_index}")
        for model_id in MODEL_IDS:
            row = stored.loc[
                stored["null_refit"].astype(int).eq(null_index)
                & stored["model"].astype(str).eq(model_id)
            ]
            if len(row) != 1 or int(row.iloc[0]["seed"]) != seed:
                raise AuditFailure(f"null identity mismatch {null_index} {model_id}")
            frozen_specification = cast(
                Mapping[str, Any],
                json.loads(str(row.iloc[0]["model_specification_json"])),
            )
            refitted = refit_independent_model(
                permuted, "null_direction_up_10m", frozen_specification
            )
            maximum_coefficient = max(
                maximum_coefficient,
                float(
                    np.max(
                        np.abs(
                            np.asarray(refitted["coefficients"], dtype=float)
                            - np.asarray(frozen_specification["coefficients"], dtype=float)
                        )
                    )
                ),
                abs(float(refitted["intercept"]) - float(frozen_specification["intercept"])),
            )
            probabilities = manual_probabilities(refitted, assessment)
            metrics = {
                "log_loss": float(log_loss(labels_valid, probabilities[valid], labels=[0, 1])),
                "brier_score": float(brier_score_loss(labels_valid, probabilities[valid])),
                "auc": float(roc_auc_score(labels_valid, probabilities[valid])),
            }
            boundary = float(cast(Mapping[str, Any], thresholds[model_id])["boundary"])
            null_frame = assessment.copy()
            null_frame["_action"] = np.where(
                probabilities >= 0.5 + boundary,
                "CALL",
                np.where(probabilities <= 0.5 - boundary, "PUT", "ABSTAIN"),
            )
            selective = selective_values(null_frame, "_action")
            expected = {
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "auc": metrics["auc"],
                "selective_directional_accuracy": selective["directional_accuracy"],
                "selective_mean_aligned_return_10m": selective["mean_aligned_return"],
            }
            for column, value in expected.items():
                maximum_metric = max(maximum_metric, abs(value - float(row.iloc[0][column])))
            fitted_models += 1
    return {
        "null_slates_checked": len(NULL_SEEDS),
        "null_models_refitted": fitted_models,
        "maximum_null_coefficient_difference": maximum_coefficient,
        "maximum_null_metric_difference": maximum_metric,
    }


def audit_baselines(assessment: pd.DataFrame, stored: pd.DataFrame) -> float:
    development = pd.read_parquet(PRIMARY / "development_oof_predictions.parquet")
    development_prior = float(development["direction_up_10m"].dropna().mean())
    sides = {
        "B0": np.full(len(assessment), 1 if development_prior >= 0.5 else -1, dtype=int),
        "B1": np.ones(len(assessment), dtype=int),
        "B2": np.sign(assessment["signed_return_2bar"].to_numpy(float)).astype(int),
        "B3": np.sign(assessment["signed_return_1bar"].to_numpy(float)).astype(int),
        "B4": np.sign(np.nan_to_num(assessment["market_return_2bar"].to_numpy(float))).astype(int),
    }
    labels = assessment["direction_up_10m"].to_numpy(float)
    maximum = 0.0
    for baseline, predicted_side in sides.items():
        valid = np.isfinite(labels) & (predicted_side != 0)
        accuracy = float(
            np.mean((predicted_side[valid] > 0).astype(int) == labels[valid].astype(int))
        )
        row = stored.loc[stored["baseline"].astype(str).eq(baseline)]
        if len(row) != 1:
            raise AuditFailure(f"baseline row missing: {baseline}")
        maximum = max(
            maximum,
            abs(accuracy - float(row.iloc[0]["directional_accuracy"])),
        )
    return maximum


def audit_support_and_decision(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    primary_candidate = str(decision["primary_candidate"])
    actions = assessment.loc[assessment[f"{primary_candidate}_action"].astype(str).ne("ABSTAIN")]
    independent = {
        "development_episodes": int(len(development)),
        "development_sessions": int(development["session"].nunique()),
        "development_stocks": int(development["stock"].nunique()),
        "development_months": int(development["session"].str[:7].nunique()),
        "development_up": int(development["direction_up_10m"].eq(1).sum()),
        "development_down": int(development["direction_up_10m"].eq(0).sum()),
        "assessment_episodes": int(len(assessment)),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["stock"].nunique()),
        "assessment_months": int(assessment["session"].str[:7].nunique()),
        "assessment_up": int(assessment["direction_up_10m"].eq(1).sum()),
        "assessment_down": int(assessment["direction_up_10m"].eq(0).sum()),
        "actions": int(len(actions)),
        "action_sessions": int(actions["session"].nunique()),
        "action_stocks": int(actions["stock"].nunique()),
        "action_months": int(actions["session"].str[:7].nunique()),
        "calls": int(actions[f"{primary_candidate}_action"].astype(str).eq("CALL").sum()),
        "puts": int(actions[f"{primary_candidate}_action"].astype(str).eq("PUT").sum()),
        "maximum_episode_stock_share": float(
            assessment.groupby("stock").size().max() / len(assessment)
        ),
        "maximum_action_stock_share": float(actions.groupby("stock").size().max() / len(actions)),
        "maximum_action_month_share": float(
            actions.assign(_month=actions["session"].astype(str).str[:7])
            .groupby("_month")
            .size()
            .max()
            / len(actions)
        ),
        "maximum_action_session_share": float(
            actions.groupby("session").size().max() / len(actions)
        ),
    }
    stored_support = cast(Mapping[str, Any], decision["support_gates"])
    comparisons = {
        "development_episodes": stored_support["development"]["episodes"],
        "development_sessions": stored_support["development"]["sessions"],
        "development_stocks": stored_support["development"]["stocks"],
        "development_months": stored_support["development"]["months"],
        "development_up": stored_support["development"]["up"],
        "development_down": stored_support["development"]["down"],
        "assessment_episodes": stored_support["assessment"]["episodes"],
        "assessment_sessions": stored_support["assessment"]["sessions"],
        "assessment_stocks": stored_support["assessment"]["stocks"],
        "assessment_months": stored_support["assessment"]["months"],
        "assessment_up": stored_support["assessment"]["up"],
        "assessment_down": stored_support["assessment"]["down"],
        "actions": stored_support["selective"]["actions"],
        "action_sessions": stored_support["selective"]["sessions"],
        "action_stocks": stored_support["selective"]["stocks"],
        "action_months": stored_support["selective"]["months"],
        "calls": stored_support["selective"]["calls"],
        "puts": stored_support["selective"]["puts"],
    }
    mismatches = {
        key: {"independent": independent[key], "stored": value}
        for key, value in comparisons.items()
        if independent[key] != value
    }
    evidence = cast(Mapping[str, Any], decision["gate_evidence"])
    expected_decision = (
        str(evidence["blocker"])
        if evidence.get("blocker")
        else (
            "blocked_insufficient_direction_episode_support"
            if not bool(evidence["episode_support_passed"])
            else (
                "blocked_insufficient_selective_action_support"
                if not bool(evidence["selective_support_passed"])
                else (
                    "route_orientation_adds_directional_value_but_full_gate_not_met"
                    if bool(evidence["d2_adds_value"])
                    else (
                        "signed_behaviour_direction_supported_route_orientation_not_supported"
                        if bool(evidence["d1_adds_value"])
                        and bool(evidence["signed_behaviour_supported"])
                        else (
                            "directional_candidate_unstable"
                            if bool(evidence["directional_information_present"])
                            and bool(evidence["stability_failed"])
                            else (
                                "directional_information_present_but_not_trade_ready"
                                if bool(evidence["directional_information_present"])
                                else "no_incremental_directional_signal"
                            )
                        )
                    )
                )
            )
        )
    )
    if expected_decision != decision["overall_decision"]:
        raise AuditFailure(
            f"decision logic mismatch: {expected_decision} != {decision['overall_decision']}"
        )
    return {
        **independent,
        "stored_support_mismatches": mismatches,
        "decision_reconstructed": expected_decision,
        "decision_match": expected_decision == decision["overall_decision"],
    }


def execute_audit() -> dict[str, Any]:
    missing_artifacts = [name for name in REQUIRED_ARTIFACTS if not (PRIMARY / name).is_file()]
    if missing_artifacts:
        raise AuditFailure(f"required artifacts missing: {missing_artifacts}")
    contract = read_json(EXPERIMENT_DIR / "contract.json")
    artifact_contract = read_json(PRIMARY / "contract.json")
    decision = read_json(PRIMARY / "decision.json")
    primary_freeze = read_json(PRIMARY / "primary_candidate_freeze.json")
    for name, artifact in (
        ("root contract", contract),
        ("artifact contract", artifact_contract),
        ("decision", decision),
        ("primary candidate freeze", primary_freeze),
    ):
        mismatches = {
            key: {"expected": value, "actual": artifact.get(key)}
            for key, value in SAFETY_EXPECTED.items()
            if artifact.get(key) != value
        }
        if mismatches:
            raise AuditFailure(f"{name} safety flags drifted: {mismatches}")
    if contract != artifact_contract:
        raise AuditFailure("root and artifact contract copies differ")

    source_manifest = read_json(PRIMARY / "source_manifest.json")
    d2_available = bool(primary_freeze["D2_causally_available"])
    sources = cast(Mapping[str, Mapping[str, Any]], source_manifest["sources"])
    branch_c_path = Path(sources["frozen_branch_c_panel"]["path"])
    state_path = Path(sources["frozen_five_minute_state_surface"]["path"])
    behaviour_path = Path(sources["audited_behavioural_dimensions"]["path"])
    route_path_raw = Path(sources["audited_route_ledger"]["path"])
    route_path = route_path_raw if route_path_raw.is_absolute() else REPO_ROOT / route_path_raw
    centroid_path_raw = Path(sources["audited_state_centroids"]["path"])
    centroid_path = (
        centroid_path_raw if centroid_path_raw.is_absolute() else REPO_ROOT / centroid_path_raw
    )
    required_sources = [
        ("frozen_branch_c_panel", branch_c_path),
        ("frozen_five_minute_state_surface", state_path),
        ("audited_behavioural_dimensions", behaviour_path),
    ]
    if d2_available:
        required_sources.extend(
            [
                ("audited_route_ledger", route_path),
                ("audited_state_centroids", centroid_path),
            ]
        )
    for key, path in required_sources:
        expected_hash = str(sources[key]["sha256"])
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise AuditFailure(f"{key} source hash drifted")

    historical = pd.read_parquet(branch_c_path)
    state_columns = [
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
        "market_breadth_bar_positive",
        "vti__bar_log_return",
    ]
    states = pd.read_parquet(
        state_path,
        columns=state_columns,
        filters=[
            ("session", ">=", "2024-01-01"),
            ("session", "<=", "2025-08-22"),
        ],
    ).rename(columns={"symbol": "stock"})
    states["bar_start_timestamp"] = pd.to_datetime(
        states["bar_start_timestamp"], utc=True, errors="raise"
    )
    states["bar_complete_timestamp"] = pd.to_datetime(
        states["bar_complete_timestamp"], utc=True, errors="raise"
    )
    behaviour = pd.read_parquet(
        behaviour_path,
        columns=[
            "symbol",
            "session",
            "decision_ordinal",
            "signed_exhaustion",
        ],
        filters=[
            ("session", ">=", "2024-01-01"),
            ("session", "<=", "2025-08-22"),
        ],
    ).rename(
        columns={
            "symbol": "stock",
            "decision_ordinal": "checkpoint",
        }
    )
    for frame_name, frame in (
        ("historical", historical),
        ("states", states),
        ("behaviour", behaviour),
    ):
        minimum = frame["session"].astype(str).min()
        maximum = frame["session"].astype(str).max()
        if minimum < "2024-01-01" or maximum > "2025-08-22":
            raise AuditFailure(f"{frame_name} chronology escaped the authorized boundary")

    v0 = load_module(V0_RUNNER, "independent_direction_audit_m1")
    models, _development_panel, _reference, reconstruction = v0.reconstruct_historical_models(
        historical
    )
    thresholds_m1 = cast(Mapping[str, Any], reconstruction["thresholds"])
    assert_close(
        "M1 threshold",
        float(thresholds_m1["M1_top_5_percent_threshold"]),
        M1_THRESHOLD,
        1e-15,
    )
    scored_panel = historical.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    scored_panel["M1_probability"] = models.m1.predict(scored_panel)
    reconstructed_episodes = reconstruct_episodes(scored_panel, states)
    stored_episodes = pd.read_parquet(PRIMARY / "movement_signal_episodes.parquet")
    stored_episodes["signal_timestamp"] = pd.to_datetime(
        stored_episodes["signal_timestamp"], utc=True
    )
    stored_episodes["prospective_entry_timestamp"] = pd.to_datetime(
        stored_episodes["prospective_entry_timestamp"], utc=True
    )
    stored_episodes = stored_episodes.sort_values(
        ["stock", "session", "checkpoint"], kind="mergesort"
    ).reset_index(drop=True)
    identity_columns = [
        "stock",
        "session",
        "checkpoint",
        "episode_number",
        "partition",
    ]
    if not reconstructed_episodes[identity_columns].equals(stored_episodes[identity_columns]):
        raise AuditFailure("independent M1 episode identities differ")
    if not reconstructed_episodes[["signal_timestamp", "prospective_entry_timestamp"]].equals(
        stored_episodes[["signal_timestamp", "prospective_entry_timestamp"]]
    ):
        raise AuditFailure("independent signal or entry timestamps differ")
    maximum_episode_probability = float(
        np.max(
            np.abs(
                reconstructed_episodes["m1_probability"].to_numpy(float)
                - stored_episodes["m1_probability"].to_numpy(float)
            )
        )
    )
    previous_left = reconstructed_episodes["previous_checkpoint_probability"].to_numpy(float)
    previous_right = stored_episodes["previous_checkpoint_probability"].to_numpy(float)
    if not np.array_equal(np.isnan(previous_left), np.isnan(previous_right)):
        raise AuditFailure("previous checkpoint M1 missingness differs")
    maximum_previous_probability = float(
        np.max(
            np.abs(
                previous_left[np.isfinite(previous_left)]
                - previous_right[np.isfinite(previous_right)]
            )
        )
    )
    spacing_violations = int(
        reconstructed_episodes["minutes_since_previous_episode"].dropna().lt(30.0).sum()
    )
    raw_above = int((scored_panel["M1_probability"].to_numpy(float) >= M1_THRESHOLD).sum())
    stored_episode_audit = read_json(PRIMARY / "episode_construction_audit.json")
    if raw_above != stored_episode_audit["raw_above_threshold_checkpoint_rows"]:
        raise AuditFailure("raw M1 threshold row count differs")

    development = pd.read_parquet(PRIMARY / "development_oof_predictions.parquet")
    assessment = pd.read_parquet(PRIMARY / "assessment_predictions.parquet")
    if (
        development["session"].astype(str).max() > "2024-12-31"
        or assessment["session"].astype(str).min() < "2025-01-01"
        or assessment["session"].astype(str).max() > "2025-08-22"
    ):
        raise AuditFailure("direction development/assessment chronology drifted")
    if development.groupby("session")["fold"].nunique().gt(1).any():
        raise AuditFailure("an OOF session appears in multiple folds")

    target_and_d0 = audit_targets_and_price_features(assessment, states)
    d1_audit = audit_d1_features(assessment, historical, behaviour)
    maximum_orientation = 0.0
    orientation_sign_mismatches = 0
    route_feature_difference = 0.0
    future_route_rows = 0
    route_rows_checked = 0
    if d2_available:
        centroids = pd.read_csv(centroid_path)
        expected_orientation = (
            centroids.loc[
                centroids["feature"]
                .astype(str)
                .isin(["signed_efficiency_6", "signed_efficiency_12"])
            ]
            .groupby("state", sort=True)["raw_feature_centroid"]
            .mean()
            .rename("mean_raw_signed_efficiency")
            .reset_index()
        )
        expected_orientation["orientation_sign"] = np.where(
            expected_orientation["mean_raw_signed_efficiency"].to_numpy(float) >= 0.0,
            1,
            -1,
        )
        orientation_map = pd.read_csv(PRIMARY / "development_orientation_map.csv")
        orientation_join = expected_orientation.merge(
            orientation_map[["state", "mean_raw_signed_efficiency", "orientation_sign"]],
            on="state",
            validate="one_to_one",
            suffixes=("_independent", "_stored"),
        )
        maximum_orientation = float(
            np.max(
                np.abs(
                    orientation_join["mean_raw_signed_efficiency_independent"].to_numpy(float)
                    - orientation_join["mean_raw_signed_efficiency_stored"].to_numpy(float)
                )
            )
        )
        orientation_sign_mismatches = int(
            (
                orientation_join["orientation_sign_independent"].to_numpy(int)
                != orientation_join["orientation_sign_stored"].to_numpy(int)
            ).sum()
        )
        route_feature_difference, future_route_rows = audit_route_features(
            assessment.head(100), route_path, orientation_map
        )
        route_rows_checked = 100

    configurations = read_json(PRIMARY / "model_configurations.json")
    threshold_artifact = read_json(PRIMARY / "frozen_direction_thresholds.json")
    direction_thresholds = cast(Mapping[str, Any], threshold_artifact["thresholds"])
    model_audit = audit_model_probabilities_and_thresholds(
        development, assessment, configurations, direction_thresholds
    )
    action_mismatches = 0
    maximum_aligned = 0.0
    for model_id in cast(Sequence[str], model_audit["fitted_model_ids"]):
        full_specification = cast(
            Mapping[str, Any],
            configurations[model_id]["full_development_model"],
        )
        probability = manual_probabilities(full_specification, assessment)
        boundary = float(cast(Mapping[str, Any], direction_thresholds[model_id])["boundary"])
        action = np.where(
            probability >= 0.5 + boundary,
            "CALL",
            np.where(probability <= 0.5 - boundary, "PUT", "ABSTAIN"),
        )
        stored_action = assessment[f"{model_id}_action"].astype(str).to_numpy()
        action_mismatches += int(np.count_nonzero(action != stored_action))
        side = np.where(action == "CALL", 1.0, np.where(action == "PUT", -1.0, np.nan))
        for horizon in HORIZONS:
            aligned = side * assessment[f"signed_log_return_{horizon}m"].to_numpy(float)
            stored_aligned = assessment[f"{model_id}_aligned_return_{horizon}m"].to_numpy(float)
            if not np.array_equal(np.isnan(aligned), np.isnan(stored_aligned)):
                raise AuditFailure("aligned return missingness differs")
            finite = np.isfinite(aligned)
            maximum_aligned = max(
                maximum_aligned,
                float(np.max(np.abs(aligned[finite] - stored_aligned[finite]))),
            )

    baseline_difference = audit_baselines(assessment, pd.read_csv(PRIMARY / "baseline_metrics.csv"))
    bootstrap_audit = audit_bootstrap(
        assessment,
        pd.read_csv(PRIMARY / "bootstrap_metrics.csv"),
        str(primary_freeze["primary_candidate"]),
    )
    null_audit = audit_nulls(
        development,
        assessment,
        pd.read_csv(PRIMARY / "direction_null_metrics.csv"),
        direction_thresholds,
    )
    support_and_decision = audit_support_and_decision(development, assessment, decision)

    actioned = assessment.loc[
        assessment[f"{primary_freeze['primary_candidate']}_action"].astype(str).ne("ABSTAIN")
    ]
    numerator = float(np.mean(np.abs(actioned["return_after_entry_10m"].to_numpy(float))))
    denominator = float(
        np.mean(
            np.abs(actioned["return_realised_before_signal"].to_numpy(float))
            + np.abs(actioned["return_signal_to_entry"].to_numpy(float))
            + np.abs(actioned["return_after_entry_10m"].to_numpy(float))
        )
    )
    remaining_ratio = numerator / denominator
    assert_close(
        "remaining movement ratio",
        remaining_ratio,
        float(decision["mean_absolute_remaining_fraction_10m"]),
        1e-12,
    )
    if bool(remaining_ratio < 0.50) != bool(decision["late_direction_problem"]):
        raise AuditFailure("late-direction flag differs")

    protected = read_json(PRIMARY / "protected_boundary_audit.json")
    determinism = read_json(PRIMARY / "determinism_check.json")
    report_primary = PRIMARY / "report.md"
    report_copy = EXPERIMENT_DIR / "reports/report.md"
    if sha256_file(report_primary) != sha256_file(report_copy):
        raise AuditFailure("report copies differ")
    plots = sorted(PRIMARY.glob("*.png"))
    if len(plots) > 3:
        raise AuditFailure("more than three plots were materialized")
    statuses = cast(Mapping[str, Any], decision["component_statuses"])
    if len(statuses) != 8 or not set(statuses.values()).issubset(
        {"supported", "promising", "not_supported", "insufficient_support", "blocked"}
    ):
        raise AuditFailure("component statuses are invalid")

    checks = {
        "required_artifacts": not missing_artifacts,
        "safety_contract": True,
        "M1_reconstruction": bool(reconstruction["passed"]),
        "frozen_M1_threshold": True,
        "episode_start_construction": True,
        "thirty_minute_spacing": spacing_violations == 0,
        "signal_and_entry_timestamps": True,
        "future_bar_feature_prevention": bool(
            pd.to_datetime(assessment["maximum_feature_source_timestamp"], utc=True)
            .le(pd.to_datetime(assessment["signal_timestamp"], utc=True))
            .all()
        ),
        "ten_minute_target": target_and_d0["maximum_target_difference"] <= 1e-12,
        "development_assessment_chronology": True,
        "opened_holdout_excluded": protected["september_through_december_2025_rows"] == 0,
        "rows_from_2026_rejected": protected["rows_from_2026_onward"] == 0,
        "D0_feature_construction": target_and_d0["maximum_d0_feature_difference"] <= 1e-12,
        "D1_feature_construction": d1_audit["maximum_d1_feature_difference"] <= 1e-12
        and d1_audit["d1_missingness_mismatches"] == 0,
        "D2_orientation_construction": (
            route_feature_difference <= 1e-12
            if d2_available
            else decision["overall_decision"] == "blocked_missing_auditable_orientation"
        ),
        "orientation_source_outcome_free": not read_json(PRIMARY / "orientation_source_audit.json")[
            "outcome_fitted"
        ],
        "orientation_crossfit_not_required": (
            read_json(PRIMARY / "orientation_crossfit_audit.json")["status"]
            == (
                "not_required_outcome_free_audited_orientation"
                if d2_available
                else "blocked_missing_auditable_orientation"
            )
        ),
        "preprocessing_2024_only": (
            model_audit["maximum_preprocessing_difference"] <= 1e-12
            and model_audit["maximum_oof_refit_preprocessing_difference"] <= 1e-12
        ),
        "development_oof_session_exclusion": (
            model_audit["independent_oof_refits"] == model_audit["expected_oof_refits"]
            and model_audit["oof_session_overlap_count"] == 0
            and model_audit["maximum_oof_refit_coefficient_difference"] <= 1e-12
        ),
        "full_model_independent_refit": (
            model_audit["maximum_refit_coefficient_difference"] <= 1e-12
        ),
        "confidence_threshold_2024_oof_only": model_audit["maximum_threshold_difference"] <= 1e-12,
        "assessment_probabilities": model_audit["maximum_probability_difference"] <= 1e-12,
        "baselines": baseline_difference <= 1e-12,
        "selective_actions": action_mismatches == 0,
        "aligned_returns": maximum_aligned <= 1e-12,
        "remaining_movement": True,
        "bootstrap": bootstrap_audit["maximum_bootstrap_draw_difference"] <= 1e-12,
        "null_tests": null_audit["maximum_null_metric_difference"] <= 1e-12,
        "support_gates": not support_and_decision["stored_support_mismatches"],
        "decision_logic": support_and_decision["decision_match"],
        "determinism": bool(determinism["passed"]),
        "report_copies": True,
        "plot_limit": len(plots) <= 3,
    }
    passed = all(checks.values())
    audit = {
        **SAFETY_EXPECTED,
        "auditor": "audit_screen_v0.py",
        "auditor_imported_runner": False,
        "independent_artifact_reload": True,
        "passed": passed,
        "checks": checks,
        "M1": {
            "threshold": M1_THRESHOLD,
            "raw_above_threshold_rows": raw_above,
            "episodes_checked": len(reconstructed_episodes),
            "maximum_episode_probability_difference": maximum_episode_probability,
            "maximum_previous_checkpoint_probability_difference": (maximum_previous_probability),
            "spacing_violations": spacing_violations,
        },
        "targets_and_D0": target_and_d0,
        "D1": d1_audit,
        "D2": {
            "status": "available" if d2_available else "blocked_missing_auditable_orientation",
            "route_rows_manually_checked": route_rows_checked,
            "maximum_route_feature_difference": route_feature_difference,
            "future_available_route_rows": future_route_rows,
            "maximum_orientation_centroid_difference": maximum_orientation,
            "orientation_sign_mismatches": orientation_sign_mismatches,
        },
        "models": model_audit,
        "actions": {
            "action_decision_mismatches": action_mismatches,
            "maximum_aligned_return_difference": maximum_aligned,
            "maximum_baseline_difference": baseline_difference,
        },
        "remaining_movement": {
            "mean_absolute_remaining_fraction_10m": remaining_ratio,
            "late_direction_problem": remaining_ratio < 0.50,
        },
        "bootstrap": bootstrap_audit,
        "nulls": null_audit,
        "support_and_decision": support_and_decision,
        "plots": [path.name for path in plots],
        "report_sha256": sha256_file(report_primary),
        "unexplained_discrepancies": 0 if passed else 1,
    }
    if not passed:
        raise AuditFailure(f"one or more audit checks failed: {checks}")
    write_json(PRIMARY / "lightweight_audit.json", audit)
    return audit


def main() -> int:
    try:
        audit = execute_audit()
    except Exception as error:
        failure = {
            **SAFETY_EXPECTED,
            "auditor": "audit_screen_v0.py",
            "auditor_imported_runner": False,
            "passed": False,
            "failure": f"{type(error).__name__}: {error}",
            "fail_closed_decision": "blocked_reproducibility_or_audit_failure",
        }
        write_json(PRIMARY / "lightweight_audit.json", failure)
        decision = read_json(PRIMARY / "decision.json")
        decision["pre_audit_decision"] = decision.get("overall_decision")
        decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
        decision["independent_audit_failure"] = failure["failure"]
        write_json(PRIMARY / "decision.json", decision)
        print(failure["failure"], file=sys.stderr)
        return 1
    print(json.dumps(_json_safe(audit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
