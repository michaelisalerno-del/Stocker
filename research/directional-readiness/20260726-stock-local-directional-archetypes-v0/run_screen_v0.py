#!/usr/bin/env python3
"""Run Stock-Local Directional Archetype Screen V0."""

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
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.behavioural_state_dimensions_v0 import (
    bar_component_frame,
    opening_raw_components,
)
from stocker_research.minimal_intraday_iv_excess_holdout_v0 import (
    ANNUAL_TRADING_MINUTES,
    GROUP_I,
    GROUP_O,
    TARGET_COLUMN,
    build_group_o,
    validate_exact_previous_session_options,
)
from stocker_research.minimal_intraday_iv_excess_holdout_v0 import (
    model_specification as movement_model_specification,
)
from stocker_research.movement_qualified_direction_v0 import (
    aligned_returns,
    assign_contiguous_session_folds,
    attach_direction_targets,
    binary_direction_metrics,
    fit_direction_model,
    freeze_confidence_boundary,
    selective_policy_metrics,
    session_bootstrap_samples,
)
from stocker_research.stock_local_directional_archetypes_v0 import (
    ABSORPTION_FEATURES,
    BASELINE_FEATURES,
    CONTINUATION_FEATURES,
    RELATIVE_STRENGTH_FEATURES,
    add_relative_strength_features,
    apply_selective_policy,
    apply_stock_local_normalisation,
    archetype_decision,
    assign_stock_local_session_weights,
    build_movement_dependency_audit,
    build_raw_archetype_features,
    checkpoint_group,
    construct_fresh_episodes,
    fit_stock_local_normalisation,
    fit_stock_market_betas,
    prepare_completed_bars,
    remaining_fraction,
    shift_features_to_next_episode,
    weighted_quantile,
)
from stocker_research.stock_options_cross_market_quick_v0 import (
    fit_cross_market_model,
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
DENSE_MODEL_CONFIG_PATH = Path(
    os.environ.get(
        "STOCKER_ARCHETYPE_DENSE_MODEL_CONFIG_PATH",
        "/Users/michaelsalerno/Documents/Codex/"
        "2026-07-23-you-are-working-in-the-github-3/research/route-competition/"
        "20260722-broad-conflict-advance-hazard-v02/artifacts/primary/"
        "model_configurations.json",
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
STRESS_OPTIONS_PATH = Path(
    os.environ.get(
        "STOCKER_ARCHETYPE_STRESS_OPTIONS_PATH",
        str(
            REPO_ROOT / "research/options-feasibility/"
            "20260724-minimal-intraday-iv-excess-holdout-v01/artifacts/primary/"
            "holdout_selected_option_pairs.parquet"
        ),
    )
)
ARCHIVED_EPISODES_PATH = Path(
    os.environ.get(
        "STOCKER_ARCHETYPE_ARCHIVED_EPISODES_PATH",
        str(
            REPO_ROOT / "research/directional-readiness/"
            "20260726-movement-qualified-direction-screen-v0/artifacts/primary/"
            "movement_signal_episodes.parquet"
        ),
    )
)

EXPECTED_DENSE_CAUSAL_SHA256 = "a916b792e15e8630dadc09bed64d71be5533ce9f3b2bd93af06605d0faaa0cc3"
EXPECTED_DENSE_MODEL_CONFIG_SHA256 = (
    "9521b093f01313a4993a9e101ef0e214ab32933809585ef267551747762b49c2"
)
EXPECTED_HISTORICAL_OPTIONS_SHA256 = (
    "4bc6fd0ce6972210949a5447fd06ca0ffaa258cb953d5e3447c1c07afab85b40"
)
EXPECTED_STATE_SHA256 = "68b1cc53c1570d53054d685966eef96f533d8760368ebfc148766bb8f3a6bcc0"
EXPECTED_STRESS_OPTIONS_SHA256 = "0b3f16cb06ae00df06dc34041a87d78b17478f1245810f54b5cc2d0f38d27e97"
EXPECTED_ARCHIVED_EPISODES_SHA256 = (
    "8db3debfbbd360f94f8b171d0bac02d673bfc31b951c15a6713ba5ccaf939aba"
)
ARCHIVED_M1_THRESHOLD = 0.49588519865576763
DEVELOPMENT_START = "2024-01-01"
DEVELOPMENT_END = "2024-12-31"
ASSESSMENT_START = "2025-01-01"
ASSESSMENT_END = "2025-08-22"
STRESS_START = "2025-09-01"
STRESS_END = "2025-12-31"
PROTECTED_START = "2026-01-01"
BOOTSTRAP_DRAWS = 100
BOOTSTRAP_SEED = 20260726
NULL_SEEDS = tuple(range(2026072601, 2026072611))
MODEL_IDS = ("B0", "C1", "A1", "R1")
ARCHETYPE_IDS = ("C1", "A1", "R1")
CATEGORICAL_FEATURES = ("stock", "checkpoint_category", "day_of_week")
CAUSAL_GROUP_I = (
    "arousal",
    "conviction",
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
)
AROUSAL_COMPONENTS = ("activity_effort", "range_effort", "travel_effort")
CONVICTION_COMPONENTS = (
    "absolute_efficiency",
    "close_retention",
    "directional_persistence",
)
CAUSAL_COMPONENTS = (*AROUSAL_COMPONENTS, *CONVICTION_COMPONENTS)
LOCAL_GROUP_I = CAUSAL_GROUP_I[2:]
DENSE_CHECKPOINTS = tuple(range(6, 35, 2))
PEER_NORMALISED_GROUP_I = (
    "posterior_entropy",
    "transition_probability",
    "persistence_probability",
    "expected_state_age",
    "top_state_probability",
    "top_second_margin",
    "any_registered_completion_prior_6",
    "any_registered_completion_prior_12",
    "same_identity_active_prefix_with_prior_completion",
    "any_hidden_event_prior_6",
    "hidden_2_3_2_prior_6",
    "bars_since_latest_registered_completion",
)
FUTURE_CONTAMINATED_GROUP_I = ("signed_pressure", "tension")
FEATURES_BY_MODEL: dict[str, tuple[str, ...]] = {
    "B0": BASELINE_FEATURES,
    "C1": CONTINUATION_FEATURES,
    "A1": ABSORPTION_FEATURES,
    "R1": RELATIVE_STRENGTH_FEATURES,
}


class ScreenBlocked(RuntimeError):
    """Fail-closed experiment blocker."""

    def __init__(self, decision: str, detail: str) -> None:
        super().__init__(detail)
        self.decision = decision
        self.detail = detail


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
    if isinstance(value, (pd.Timestamp, pd.Period, Path)):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_identity(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    values = pd.util.hash_pandas_object(frame.loc[:, list(columns)], index=False).to_numpy(
        np.uint64
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def load_contract() -> dict[str, Any]:
    contract = cast(
        dict[str, Any],
        json.loads((EXPERIMENT_DIR / "contract.json").read_text(encoding="utf-8")),
    )
    required = {
        "research_only": True,
        "retrospective_directional_candidate_screen": True,
        "three_archetypes_tested_separately": True,
        "archetypes_combined_for_primary_inference": False,
        "stock_local_normalisation": True,
        "cross_sectional_peer_normalisation": False,
        "archived_signed_pressure_excluded": True,
        "future_filtered_peer_slates_excluded": True,
        "trigger_bar_excluded_from_direction_features": True,
        "direction_marker_bar": "T-1",
        "primary_direction_horizon_minutes": 10,
        "development_end": DEVELOPMENT_END,
        "assessment_end": ASSESSMENT_END,
        "opened_holdout_directionally_excluded": True,
        "protected_start": PROTECTED_START,
        "option_pnl_calculated": False,
        "broker_access": False,
        "paper_orders_allowed": False,
        "live_orders_allowed": False,
        "strategy_promotion": False,
        "production_runtime_modified": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            f"contract mismatch: {mismatches}",
        )
    return contract


def dependency_graph() -> dict[str, tuple[str, ...]]:
    """Return the explicit archived movement dependency graph."""

    return {
        "future_filtered_peer_slate": (
            "future_filtered_peer_median_signed_progress",
            "future_filtered_peer_median_absolute_progress",
            "future_dependent_checkpoint_membership",
        ),
        "future_filtered_peer_median_signed_progress": ("raw_signed_progress",),
        "future_filtered_peer_median_absolute_progress": ("raw_component__absolute_progress",),
        "raw_signed_progress": ("raw_component__signed_progress",),
        "raw_component__signed_progress": ("z_component__signed_progress",),
        "z_component__signed_progress": ("signed_pressure",),
        "raw_component__absolute_progress": ("z_component__absolute_progress",),
        "z_component__absolute_progress": ("tension",),
        "signed_pressure": ("M1_probability",),
        "tension": ("M1_probability",),
        "M1_probability": ("frozen_threshold_membership",),
        "frozen_threshold_membership": ("fresh_episode_identity",),
        "future_dependent_checkpoint_membership": (
            "eligible_stocks_in_session",
            "eligible_advance_rows_for_stock_session",
            "sequential_row_weight",
            "archived_checkpoint_universe",
        ),
        "eligible_stocks_in_session": ("sequential_row_weight",),
        "eligible_advance_rows_for_stock_session": ("sequential_row_weight",),
        "sequential_row_weight": (
            "M1_coefficients",
            "archived_weighted_threshold",
        ),
        "archived_checkpoint_universe": (
            "M1_coefficients",
            "archived_weighted_threshold",
            "archived_episode_identity",
        ),
        "M1_coefficients": ("M1_probability",),
        "archived_weighted_threshold": ("frozen_threshold_membership",),
        "future_bar_target_validity": ("archived_branch_c_checkpoint_membership",),
        "archived_branch_c_checkpoint_membership": (
            "M1_coefficients",
            "archived_weighted_threshold",
        ),
        "contemporaneous_peer_slate": ("regime_cross_sectional_market_features",),
        "regime_cross_sectional_market_features": ("regime_emission_inputs",),
        "regime_emission_inputs": (
            "posterior_entropy",
            "transition_probability",
            "persistence_probability",
            "expected_state_age",
            "top_state_probability",
            "top_second_margin",
            "structural_memory_features",
        ),
        "structural_memory_features": tuple(PEER_NORMALISED_GROUP_I[6:]),
    }


def movement_metrics(frame: pd.DataFrame, probability_column: str) -> dict[str, float | int]:
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").to_numpy(int)
    probability = np.clip(
        pd.to_numeric(frame[probability_column], errors="raise").to_numpy(float),
        1e-12,
        1.0 - 1e-12,
    )
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    return {
        "rows": int(len(frame)),
        "stocks": int(frame["symbol"].nunique()),
        "sessions": int(frame["session"].nunique()),
        "log_loss": float(log_loss(target, probability, sample_weight=weights)),
        "brier_score": float(brier_score_loss(target, probability, sample_weight=weights)),
        "auc": float(roc_auc_score(target, probability, sample_weight=weights)),
        "average_precision": float(
            average_precision_score(target, probability, sample_weight=weights)
        ),
    }


def _local_raw_features(completed: pd.DataFrame) -> dict[str, float]:
    components = bar_component_frame(completed)
    trailing = components.iloc[-6:].reset_index(drop=True)
    ranges = trailing["true_range_bps"].to_numpy(float)
    returns = trailing["return_bps"].to_numpy(float)
    activity = trailing["historical_relative_activity"].to_numpy(float)
    mean_range = float(np.mean(ranges))
    mean_activity = float(np.mean(activity))
    current = trailing.iloc[-1]
    width = float(current["high"] - current["low"])
    body = abs(float(current["close"] - current["open"])) / max(width, 1e-12)
    return {
        "prior_6_mean_range": mean_range,
        "prior_6_price_travel": float(np.sum(np.abs(returns))),
        "prior_6_absolute_net_movement": abs(float(np.sum(returns))),
        "prior_6_activity_proxy": mean_activity,
        "recent_vs_earlier_range_ratio": float(np.mean(ranges[3:]))
        / max(float(np.mean(ranges[:3])), 1e-12),
        "recent_vs_earlier_activity_ratio": float(np.mean(activity[3:]))
        / max(float(np.mean(activity[:3])), 1e-12),
        "current_bar_range_vs_prior_6": float(current["true_range_bps"]) / max(mean_range, 1e-12),
        "current_bar_activity_vs_prior_6": float(current["historical_relative_activity"])
        / max(mean_activity, 1e-12),
        "current_bar_body_fraction": min(max(body, 0.0), 1.0),
        "current_bar_extreme_wick_fraction": max(
            float(current["upper_wick_fraction"]),
            float(current["lower_wick_fraction"]),
        ),
    }


def _scale_causal_group_i(
    raw: pd.DataFrame,
    scaling: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply the predecessor's 2024-frozen non-peer Group-I definitions."""

    output = raw.copy()
    component_scaling = cast(
        Mapping[str, Mapping[str, Mapping[str, float]]],
        scaling["component_development_scaling"],
    )
    for component in CAUSAL_COMPONENTS:
        output[f"_z_{component}"] = np.nan
    for checkpoint, indices in output.groupby("checkpoint", sort=True).groups.items():
        frozen_checkpoint = component_scaling[str(int(checkpoint))]
        positions = list(indices)
        for component in CAUSAL_COMPONENTS:
            frozen = frozen_checkpoint[component]
            values = (
                output.loc[positions, f"raw_component__{component}"].to_numpy(float)
                - float(frozen["center"])
            ) / float(frozen["scale"])
            output.loc[positions, f"_z_{component}"] = np.clip(
                values,
                float(frozen.get("clip_lower", -5.0)),
                float(frozen.get("clip_upper", 5.0)),
            )
    output["arousal"] = output[[f"_z_{name}" for name in AROUSAL_COMPONENTS]].mean(axis=1)
    output["conviction"] = output[[f"_z_{name}" for name in CONVICTION_COMPONENTS]].mean(axis=1)
    local_scaling = cast(
        Mapping[str, Mapping[str, Mapping[str, float]]],
        scaling["local_development_scaling"],
    )
    for feature in LOCAL_GROUP_I:
        output[feature] = np.nan
    for (stock, checkpoint), indices in output.groupby(
        ["stock", "checkpoint"], sort=True
    ).groups.items():
        key = f"{stock}|{int(checkpoint)}"
        if key not in local_scaling:
            raise ScreenBlocked(
                "blocked_reproducibility_or_audit_failure",
                f"frozen local Group-I scaling is unavailable for {key}",
            )
        positions = list(indices)
        for feature in LOCAL_GROUP_I:
            frozen = local_scaling[key][feature]
            values = (
                output.loc[positions, f"raw_local__{feature}"].to_numpy(float)
                - float(frozen["center"])
            ) / float(frozen["scale"])
            output.loc[positions, feature] = np.clip(values, -5.0, 5.0)
    causal_values = output.loc[:, list(CAUSAL_GROUP_I)].to_numpy(float)
    if not np.isfinite(causal_values).all():
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "causal Group-I reconstruction produced non-finite values",
        )
    return output.drop(columns=[f"_z_{name}" for name in CAUSAL_COMPONENTS])


def _opening_range_baselines(states: pd.DataFrame) -> dict[tuple[str, str, int], float]:
    rows: list[dict[str, object]] = []
    for (stock, session), session_rows in states.groupby(["stock", "session"], sort=False):
        ordered = session_rows.sort_values("bar_ordinal", kind="mergesort")
        ordinals = set(ordered["bar_ordinal"].astype(int))
        for checkpoint in DENSE_CHECKPOINTS:
            if not set(range(checkpoint)).issubset(ordinals):
                continue
            prefix = ordered.loc[ordered["bar_ordinal"].astype(int).lt(checkpoint)]
            session_open = float(prefix.iloc[0]["open"])
            value = (
                10_000.0 * (float(prefix["high"].max()) - float(prefix["low"].min())) / session_open
            )
            rows.append(
                {
                    "stock": str(stock),
                    "session": str(session),
                    "checkpoint": checkpoint,
                    "opening_range_bps": value,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["stock", "checkpoint", "session"], kind="mergesort")
    frame["trailing_median"] = frame.groupby(["stock", "checkpoint"], sort=False)[
        "opening_range_bps"
    ].transform(lambda values: values.expanding(min_periods=1).median().shift(1))
    return {
        (str(row.stock), str(row.session), int(row.checkpoint)): float(row.trailing_median)
        for row in frame.itertuples(index=False)
        if math.isfinite(float(row.trailing_median)) and float(row.trailing_median) > 0.0
    }


def _reconstruct_stress_raw_surface(
    states: pd.DataFrame,
    option_context: pd.DataFrame,
) -> pd.DataFrame:
    """Build opened-period checkpoint features from completed stock bars only."""

    context_keys = {
        (str(row.stock), str(row.session))
        for row in option_context[["stock", "session"]].itertuples(index=False)
    }
    baselines = _opening_range_baselines(states)
    records: list[dict[str, object]] = []
    for (stock, session), session_rows in states.groupby(["stock", "session"], sort=False):
        key = (str(stock), str(session))
        if key not in context_keys or not STRESS_START <= str(session) <= STRESS_END:
            continue
        ordered = session_rows.sort_values("bar_ordinal", kind="mergesort")
        ordinals = set(ordered["bar_ordinal"].astype(int))
        for checkpoint in DENSE_CHECKPOINTS:
            baseline = baselines.get((str(stock), str(session), checkpoint))
            if baseline is None or not set(range(checkpoint)).issubset(ordinals):
                continue
            completed = ordered.loc[ordered["bar_ordinal"].astype(int).lt(checkpoint)]
            activity = completed["historical_relative_activity"].to_numpy(float)
            if len(completed) != checkpoint or not np.isfinite(activity).all():
                continue
            components = bar_component_frame(completed)
            first_close = float(components.iloc[checkpoint // 2 - 1]["close"])
            session_open = float(components.iloc[0]["open"])
            last_close = float(components.iloc[-1]["close"])
            progress = 10_000.0 * (last_close / session_open - 1.0)
            earlier = 10_000.0 * (first_close / session_open - 1.0)
            recent = 10_000.0 * (last_close / first_close - 1.0)
            raw_components = opening_raw_components(
                components,
                trailing_opening_range_median_bps=baseline,
                signed_progress_bps=progress,
                signed_progress_acceleration_bps=recent - earlier,
                return_gap_bps=progress,
            )
            local = _local_raw_features(completed)
            current = completed.iloc[-1]
            record: dict[str, object] = {
                "row_id": f"{stock}|{session}|{checkpoint}",
                "stock": str(stock),
                "session": str(session),
                "period": "opened_retrospective_stress",
                "checkpoint": checkpoint,
                "checkpoint_bar_ordinal_zero_based": checkpoint - 1,
                "checkpoint_timestamp_utc": current["bar_start_timestamp"],
                "feature_available_timestamp_utc": current["bar_complete_timestamp"],
            }
            for component in CAUSAL_COMPONENTS:
                record[f"raw_component__{component}"] = float(raw_components[component])
            for feature in LOCAL_GROUP_I:
                record[f"raw_local__{feature}"] = float(local[feature])
            records.append(record)
    if not records:
        raise ScreenBlocked(
            "blocked_no_supported_causal_movement_gate",
            "opened stress surface has no causal checkpoint rows",
        )
    return pd.DataFrame(records)


def _attach_movement_gate_targets(
    surface: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    """Attach future movement labels without changing causal row membership."""

    output = surface.copy()
    entry = states[["stock", "session", "bar_ordinal", "open"]].rename(
        columns={"bar_ordinal": "checkpoint", "open": "_movement_entry_price"}
    )
    close_15 = states[["stock", "session", "bar_ordinal", "close"]].copy()
    close_15["checkpoint"] = close_15["bar_ordinal"].astype(int) - 2
    close_15 = close_15.rename(columns={"close": "_movement_close_15m"}).drop(columns="bar_ordinal")
    output = output.merge(
        entry,
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    ).merge(
        close_15,
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    entry_values = pd.to_numeric(output["_movement_entry_price"], errors="coerce").to_numpy(float)
    close_values = pd.to_numeric(output["_movement_close_15m"], errors="coerce").to_numpy(float)
    atm_iv = pd.to_numeric(output["atm_iv"], errors="coerce").to_numpy(float)
    valid = (
        np.isfinite(entry_values)
        & (entry_values > 0.0)
        & np.isfinite(close_values)
        & (close_values > 0.0)
        & np.isfinite(atm_iv)
        & (atm_iv > 0.0)
    )
    movement = np.full(len(output), np.nan, dtype=float)
    expectation = np.full(len(output), np.nan, dtype=float)
    movement[valid] = np.abs(np.log(close_values[valid] / entry_values[valid]))
    expectation[valid] = (
        atm_iv[valid] * math.sqrt(15.0 / ANNUAL_TRADING_MINUTES) * math.sqrt(2.0 / math.pi)
    )
    target = np.full(len(output), np.nan, dtype=float)
    target[valid] = (movement[valid] > expectation[valid]).astype(float)
    output["absolute_log_return_15m"] = movement
    output["iv_expected_absolute_15m"] = expectation
    output[TARGET_COLUMN] = target
    output["movement_target_available"] = valid
    return output


def _join_causal_option_context(
    raw_surface: pd.DataFrame,
    option_context: pd.DataFrame,
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    option_columns = [
        "stock",
        "session",
        "required_options_date",
        "options_observation_date",
        "atm_iv",
        *GROUP_O[:16],
    ]
    context = option_context.loc[:, option_columns].drop_duplicates(
        ["stock", "session"], keep=False
    )
    if len(context) != option_context[["stock", "session"]].drop_duplicates().shape[0]:
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "previous-close options context is not unique by stock-session",
        )
    for row in context[["session", "required_options_date", "options_observation_date"]].itertuples(
        index=False
    ):
        validate_exact_previous_session_options(
            signal_date=pd.Timestamp(row.session).date(),
            required_options_date=pd.Timestamp(row.required_options_date).date(),
            actual_options_date=pd.Timestamp(row.options_observation_date).date(),
        )
    joined = raw_surface.merge(
        context,
        on=["stock", "session"],
        how="inner",
        validate="many_to_one",
    )
    group_o = build_group_o(joined.rename(columns={"stock": "symbol"}))
    joined.loc[:, list(GROUP_O)] = group_o.to_numpy(float)
    joined = assign_stock_local_session_weights(joined)
    joined["symbol"] = joined["stock"].astype(str)
    joined = _attach_movement_gate_targets(joined, states)
    audit = {
        "causal_checkpoint_rows_before_options": int(len(raw_surface)),
        "causal_checkpoint_rows_after_previous_close_options": int(len(joined)),
        "causal_stock_sessions_after_previous_close_options": int(
            joined[["stock", "session"]].drop_duplicates().shape[0]
        ),
        "movement_target_available_rows": int(joined["movement_target_available"].sum()),
        "movement_target_unavailable_rows": int((~joined["movement_target_available"]).sum()),
        "row_membership_uses_future_target_validity": False,
        "weights_use_future_target_validity": False,
        "weights_use_peer_stock_counts": False,
        "weight_definition": "1 / causal checkpoints in the same stock-session",
    }
    return joined.sort_values("row_id", kind="mergesort").reset_index(drop=True), audit


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_paths = (
        DENSE_CAUSAL_PATH,
        DENSE_MODEL_CONFIG_PATH,
        HISTORICAL_OPTIONS_PATH,
        STATE_PATH,
        STRESS_OPTIONS_PATH,
        ARCHIVED_EPISODES_PATH,
    )
    for path in source_paths:
        if not path.is_file():
            raise ScreenBlocked(
                "blocked_reproducibility_or_audit_failure",
                f"required source is missing: {path}",
            )
    expected_hashes = {
        DENSE_CAUSAL_PATH: EXPECTED_DENSE_CAUSAL_SHA256,
        DENSE_MODEL_CONFIG_PATH: EXPECTED_DENSE_MODEL_CONFIG_SHA256,
        HISTORICAL_OPTIONS_PATH: EXPECTED_HISTORICAL_OPTIONS_SHA256,
        STATE_PATH: EXPECTED_STATE_SHA256,
        STRESS_OPTIONS_PATH: EXPECTED_STRESS_OPTIONS_SHA256,
        ARCHIVED_EPISODES_PATH: EXPECTED_ARCHIVED_EPISODES_SHA256,
    }
    observed_hashes = {path: sha256_file(path) for path in source_paths}
    if any(observed_hashes[path] != expected for path, expected in expected_hashes.items()):
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "a frozen source hash drifted",
        )
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
        "bar_log_return",
        "historical_relative_activity",
        "vti__bar_log_return",
        "feature_available_timestamp_max",
    ]
    states = pd.read_parquet(
        STATE_PATH,
        columns=state_columns,
        filters=[
            ("session", ">=", DEVELOPMENT_START),
            ("session", "<=", STRESS_END),
        ],
    ).rename(columns={"symbol": "stock"})
    if states["session"].astype(str).max() > STRESS_END:
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "state read crossed the opened stress boundary",
        )
    dense_columns = [
        "row_id",
        "symbol",
        "session",
        "period",
        "checkpoint",
        "checkpoint_bar_ordinal_zero_based",
        "checkpoint_timestamp_utc",
        "feature_available_timestamp_utc",
        *(f"raw_component__{name}" for name in CAUSAL_COMPONENTS),
        *(f"raw_local__{name}" for name in LOCAL_GROUP_I),
        *CAUSAL_GROUP_I,
    ]
    dense = pd.read_parquet(DENSE_CAUSAL_PATH, columns=dense_columns).rename(
        columns={"symbol": "stock"}
    )
    if dense["session"].astype(str).max() != ASSESSMENT_END:
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "causal dense surface chronology drifted",
        )
    scaling = cast(
        dict[str, Any],
        json.loads(DENSE_MODEL_CONFIG_PATH.read_text(encoding="utf-8")),
    )
    raw_historical_columns = [
        "row_id",
        "stock",
        "session",
        "period",
        "checkpoint",
        "checkpoint_bar_ordinal_zero_based",
        "checkpoint_timestamp_utc",
        "feature_available_timestamp_utc",
        *(f"raw_component__{name}" for name in CAUSAL_COMPONENTS),
        *(f"raw_local__{name}" for name in LOCAL_GROUP_I),
    ]
    reconstructed_dense = _scale_causal_group_i(dense.loc[:, raw_historical_columns], scaling)
    maximum_group_i_reconstruction_difference = float(
        np.max(
            np.abs(
                reconstructed_dense.loc[:, list(CAUSAL_GROUP_I)].to_numpy(float)
                - dense.loc[:, list(CAUSAL_GROUP_I)].to_numpy(float)
            )
        )
    )
    if maximum_group_i_reconstruction_difference > 1e-12:
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "causal Group-I reconstruction differed from the frozen full checkpoint surface",
        )
    historical_options = pd.read_parquet(HISTORICAL_OPTIONS_PATH).rename(
        columns={"symbol": "stock"}
    )
    historical, historical_surface_audit = _join_causal_option_context(
        reconstructed_dense,
        historical_options,
        states,
    )
    stress_options = pd.read_parquet(STRESS_OPTIONS_PATH).rename(columns={"symbol": "stock"})
    if "pair_available" in stress_options:
        stress_options = stress_options.loc[stress_options["pair_available"].astype(bool)].copy()
    stress_raw = _reconstruct_stress_raw_surface(states, stress_options)
    stress_scaled = _scale_causal_group_i(stress_raw, scaling)
    stress, stress_surface_audit = _join_causal_option_context(
        stress_scaled,
        stress_options,
        states,
    )
    if (
        historical["session"].astype(str).min() < DEVELOPMENT_START
        or historical["session"].astype(str).max() != ASSESSMENT_END
        or stress["session"].astype(str).min() < STRESS_START
        or stress["session"].astype(str).max() > STRESS_END
    ):
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "causal movement surfaces crossed a frozen chronology boundary",
        )
    manifest = {
        "research_only": True,
        "sources": [
            {
                "role": "unfiltered_dense_causal_checkpoint_surface",
                "path": str(DENSE_CAUSAL_PATH),
                "sha256": observed_hashes[DENSE_CAUSAL_PATH],
                "columns_read_exclude_archived_signed_pressure": True,
                "columns_read_exclude_future_labels_and_eligibility": True,
                "rows_read": int(len(dense)),
            },
            {
                "role": "frozen_causal_group_i_scaling_definitions",
                "path": str(DENSE_MODEL_CONFIG_PATH),
                "sha256": observed_hashes[DENSE_MODEL_CONFIG_PATH],
                "maximum_reconstruction_difference": maximum_group_i_reconstruction_difference,
            },
            {
                "role": "exact_previous_close_historical_options_context",
                "path": str(HISTORICAL_OPTIONS_PATH),
                "sha256": observed_hashes[HISTORICAL_OPTIONS_PATH],
                "stock_sessions": int(len(historical_options)),
            },
            {
                "role": "completed_five_minute_stock_and_market_bars",
                "path": str(STATE_PATH),
                "sha256": observed_hashes[STATE_PATH],
                "rows_read": int(len(states)),
                "maximum_session_read": str(states["session"].max()),
            },
            {
                "role": "exact_previous_close_opened_stress_options_context",
                "path": str(STRESS_OPTIONS_PATH),
                "sha256": observed_hashes[STRESS_OPTIONS_PATH],
                "stock_sessions": int(len(stress_options)),
                "maximum_session": str(stress["session"].max()),
            },
            {
                "role": "archived_episode_identities_for_impact_comparison_only",
                "path": str(ARCHIVED_EPISODES_PATH),
                "sha256": observed_hashes[ARCHIVED_EPISODES_PATH],
                "archived_pressure_values_read": False,
            },
        ],
        "causal_historical_surface": historical_surface_audit,
        "causal_opened_stress_surface": stress_surface_audit,
        "archived_signed_pressure_values_read": False,
        "future_filtered_advance_eligibility_read": False,
        "future_bar_target_validity_used_for_row_membership": False,
        "cross_stock_count_used_for_weights": False,
        "options_downloaded": False,
        "broker_access": False,
        "protected_rows_read": 0,
    }
    return historical, stress, states, manifest


def phase_zero(
    historical: pd.DataFrame,
    stress: pd.DataFrame,
) -> tuple[Any, Any, float, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    graph = dependency_graph()
    audit = build_movement_dependency_audit(
        graph=graph,
        contaminated_roots=(
            "future_filtered_peer_slate",
            "future_bar_target_validity",
        ),
        group_i_features=GROUP_I,
        peer_normalised_features=PEER_NORMALISED_GROUP_I,
    )
    if not audit["archived_m1_numerically_affected"] or set(
        audit["contaminated_group_i_features"]
    ) != set(FUTURE_CONTAMINATED_GROUP_I):
        raise ScreenBlocked(
            "blocked_movement_gate_dependency_audit_failure",
            "archived M1 contamination did not match the audited lineage",
        )
    development = historical.loc[historical["period"].astype(str).eq("development")].copy()
    assessment = historical.loc[historical["period"].astype(str).eq("assessment")].copy()
    development_fit = development.loc[development[TARGET_COLUMN].notna()].copy()
    assessment_eval = assessment.loc[assessment[TARGET_COLUMN].notna()].copy()
    stress_eval = stress.loc[stress[TARGET_COLUMN].notna()].copy()
    m0 = fit_cross_market_model(
        development_fit,
        model_id="M0",
        numeric_features=GROUP_O,
        category_control_names=("stock",),
        target_column=TARGET_COLUMN,
        kind="logistic",
    )
    m1c = fit_cross_market_model(
        development_fit,
        model_id="M1C",
        numeric_features=(*GROUP_O, *CAUSAL_GROUP_I),
        category_control_names=("stock",),
        target_column=TARGET_COLUMN,
        kind="logistic",
    )
    development["M0_probability"] = m0.predict(development)
    development["M1C_probability"] = m1c.predict(development)
    assessment["M0_probability"] = m0.predict(assessment)
    assessment["M1C_probability"] = m1c.predict(assessment)
    stress_scored = stress.copy()
    stress_scored["M0_probability"] = m0.predict(stress_scored)
    stress_scored["M1C_probability"] = m1c.predict(stress_scored)
    assessment_eval = assessment.loc[assessment[TARGET_COLUMN].notna()].copy()
    stress_eval = stress_scored.loc[stress_scored[TARGET_COLUMN].notna()].copy()
    threshold = weighted_quantile(
        development["M1C_probability"].to_numpy(float),
        development["row_weight"].to_numpy(float),
        0.95,
    )
    metric_rows: list[dict[str, object]] = []
    for period, frame in (
        ("assessment", assessment_eval),
        ("opened_retrospective_stress", stress_eval),
    ):
        for model_id in ("M0", "M1C"):
            metric_rows.append(
                {
                    "period": period,
                    "model": model_id,
                    **movement_metrics(frame, f"{model_id}_probability"),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    for metric in ("log_loss", "brier_score", "auc", "average_precision"):
        metrics[f"{metric}_improvement_vs_M0"] = 0.0
    for period in metrics["period"].astype(str).unique():
        period_rows = metrics.loc[metrics["period"].astype(str).eq(period)]
        baseline = period_rows.loc[period_rows["model"].eq("M0")].iloc[0]
        m1c_index = period_rows.loc[period_rows["model"].eq("M1C")].index[0]
        metrics.loc[m1c_index, "log_loss_improvement_vs_M0"] = float(baseline["log_loss"]) - float(
            metrics.loc[m1c_index, "log_loss"]
        )
        metrics.loc[m1c_index, "brier_score_improvement_vs_M0"] = float(
            baseline["brier_score"]
        ) - float(metrics.loc[m1c_index, "brier_score"])
        metrics.loc[m1c_index, "auc_improvement_vs_M0"] = float(
            metrics.loc[m1c_index, "auc"]
        ) - float(baseline["auc"])
        metrics.loc[m1c_index, "average_precision_improvement_vs_M0"] = float(
            metrics.loc[m1c_index, "average_precision"]
        ) - float(baseline["average_precision"])
    gate_checks: dict[str, bool] = {}
    for period in ("assessment", "opened_retrospective_stress"):
        period_rows = metrics.loc[metrics["period"].eq(period)].set_index("model")
        m0_row = period_rows.loc["M0"]
        m1c_row = period_rows.loc["M1C"]
        gate_checks[f"{period}_log_loss_improved"] = bool(
            float(m1c_row["log_loss"]) < float(m0_row["log_loss"])
        )
        gate_checks[f"{period}_brier_improved"] = bool(
            float(m1c_row["brier_score"]) < float(m0_row["brier_score"])
        )
        gate_checks[f"{period}_auc_or_ap_improved"] = bool(
            float(m1c_row["auc"]) > float(m0_row["auc"])
            or float(m1c_row["average_precision"]) > float(m0_row["average_precision"])
        )
    concentration: dict[str, dict[str, float | int | bool]] = {}
    for period, frame in (
        ("assessment", assessment_eval),
        ("opened_retrospective_stress", stress_eval),
    ):
        target = frame[TARGET_COLUMN].to_numpy(float)
        gain = (
            target * np.log(np.clip(frame["M1C_probability"].to_numpy(float), 1e-12, 1.0))
            + (1.0 - target)
            * np.log(np.clip(1.0 - frame["M1C_probability"].to_numpy(float), 1e-12, 1.0))
            - target * np.log(np.clip(frame["M0_probability"].to_numpy(float), 1e-12, 1.0))
            - (1.0 - target)
            * np.log(np.clip(1.0 - frame["M0_probability"].to_numpy(float), 1e-12, 1.0))
        ) * frame["row_weight"].to_numpy(float)
        by_stock = frame.assign(_gain=gain).groupby("stock", sort=True)["_gain"].sum()
        positive = by_stock.clip(lower=0.0)
        maximum_positive_stock_share = (
            float(positive.max() / positive.sum()) if positive.sum() > 0.0 else math.inf
        )
        positive_stocks = int((by_stock > 0.0).sum())
        passed = bool(maximum_positive_stock_share < 0.50 and positive_stocks >= 10)
        concentration[period] = {
            "maximum_positive_gain_stock_share": maximum_positive_stock_share,
            "positive_gain_stocks": positive_stocks,
            "passed": passed,
        }
        gate_checks[f"{period}_effect_not_concentrated_in_one_stock"] = passed
    phase_audit = {
        **audit,
        "passed": bool(all(gate_checks.values())),
        "archived_m1_threshold": ARCHIVED_M1_THRESHOLD,
        "m1c_required": True,
        "causal_group_i_features": list(CAUSAL_GROUP_I),
        "future_contaminated_group_i_features": list(FUTURE_CONTAMINATED_GROUP_I),
        "other_peer_normalised_group_i_features": list(PEER_NORMALISED_GROUP_I),
        "group_o_unchanged": True,
        "interactions_in_frozen_m1": [],
        "causal_checkpoint_membership": (
            "completed stock bars through T plus exact previous-close options context"
        ),
        "future_target_validity_used_for_membership": False,
        "stock_local_weight_definition": (
            "one divided by causal checkpoints in the same stock-session"
        ),
        "peer_stock_counts_used_for_weights": False,
        "development_full_threshold_rows": int(len(development)),
        "development_labelled_fit_rows": int(len(development_fit)),
        "assessment_metric_rows": int(len(assessment_eval)),
        "opened_stress_metric_rows": int(len(stress_eval)),
        "concentration_by_period": concentration,
        "gate_checks_before_episode_support": gate_checks,
    }
    if not all(gate_checks.values()):
        raise ScreenBlocked(
            "blocked_no_supported_causal_movement_gate",
            f"M1C failed its movement gate checks: {gate_checks}",
        )
    scored = pd.concat([development, assessment], ignore_index=True).sort_values(
        "row_id", kind="mergesort"
    )
    return m0, m1c, threshold, scored, metrics, phase_audit


def build_episodes(
    scored: pd.DataFrame,
    states: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    causal = scored.copy()
    causal["partition"] = np.where(
        causal["session"].astype(str).le(DEVELOPMENT_END),
        "development",
        "assessment",
    )
    times = states[
        [
            "stock",
            "session",
            "bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
        ]
    ].copy()
    signal = times.rename(columns={"bar_complete_timestamp": "signal_timestamp"})
    signal["checkpoint"] = signal["bar_ordinal"].astype(int) + 1
    entry = times.rename(columns={"bar_start_timestamp": "prospective_entry_timestamp"})
    entry["checkpoint"] = entry["bar_ordinal"].astype(int)
    causal = causal.merge(
        signal[["stock", "session", "checkpoint", "signal_timestamp"]],
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    ).merge(
        entry[
            [
                "stock",
                "session",
                "checkpoint",
                "prospective_entry_timestamp",
            ]
        ],
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    if causal[["signal_timestamp", "prospective_entry_timestamp"]].isna().any().any():
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "a causal movement checkpoint lacks a completed-bar timestamp",
        )
    source_available = pd.to_datetime(
        causal["feature_available_timestamp_utc"], utc=True, errors="raise"
    )
    signal_available = pd.to_datetime(causal["signal_timestamp"], utc=True, errors="raise")
    if not source_available.equals(signal_available):
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "movement feature availability does not equal trigger close",
        )
    episode_input = causal[
        [
            "stock",
            "session",
            "checkpoint",
            "signal_timestamp",
            "prospective_entry_timestamp",
            "M1C_probability",
            "partition",
        ]
    ].rename(columns={"M1C_probability": "movement_probability"})
    episodes = construct_fresh_episodes(episode_input, threshold=threshold)
    metadata_columns = [
        "stock",
        "session",
        "checkpoint",
        "row_id",
        "row_weight",
        "atm_iv",
        "M0_probability",
        "M1C_probability",
    ]
    episodes = episodes.merge(
        causal[metadata_columns],
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    targets = attach_direction_targets(episodes, states)
    marker = states[["stock", "session", "bar_ordinal", "bar_complete_timestamp", "close"]].rename(
        columns={
            "bar_ordinal": "marker_bar_ordinal",
            "bar_complete_timestamp": "independent_marker_timestamp",
            "close": "marker_close",
        }
    )
    targets = targets.merge(
        marker,
        on=["stock", "session", "marker_bar_ordinal"],
        how="left",
        validate="many_to_one",
    )
    targets["pre_entry_return"] = np.log(targets["entry_price"] / targets["marker_close"])
    targets["remaining_fraction_10m"] = [
        remaining_fraction(before, after)
        for before, after in zip(
            targets["pre_entry_return"].to_numpy(float),
            targets["signed_log_return_10m"].to_numpy(float),
            strict=True,
        )
    ]
    marker_times = pd.to_datetime(targets["independent_marker_timestamp"], utc=True, errors="raise")
    trigger_times = pd.to_datetime(targets["signal_timestamp"], utc=True, errors="raise")
    if bool(marker_times.ge(trigger_times).any()):
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "direction marker did not precede trigger T",
        )
    ordered = episode_input.sort_values(["stock", "session", "checkpoint"], kind="mergesort").copy()
    ordered["above"] = ordered["movement_probability"].to_numpy(float) >= threshold
    ordered["previous"] = ordered.groupby(["stock", "session"], sort=False)[
        "movement_probability"
    ].shift()
    ordered["fresh_crossing"] = ordered["above"] & (
        ordered["previous"].isna() | ordered["previous"].lt(threshold)
    )
    archived = pd.read_parquet(
        ARCHIVED_EPISODES_PATH,
        columns=["stock", "session", "checkpoint", "m1_probability"],
    )
    impact = episodes[["stock", "session", "checkpoint", "M1C_probability"]].merge(
        archived,
        on=["stock", "session", "checkpoint"],
        how="outer",
        indicator=True,
    )
    impact["episode_impact"] = impact["_merge"].map(
        {
            "left_only": "M1C_only",
            "right_only": "archived_M1_only",
            "both": "shared",
        }
    )
    impact = impact.drop(columns="_merge")
    spacing_violations = int(episodes["minutes_since_previous_episode"].dropna().lt(30.0).sum())
    audit = {
        "passed": spacing_violations == 0,
        "raw_checkpoint_rows": int(len(ordered)),
        "raw_threshold_rows": int(ordered["above"].sum()),
        "fresh_unspaced_crossings": int(ordered["fresh_crossing"].sum()),
        "fresh_episodes": int(len(episodes)),
        "development_episodes": int(episodes["partition"].eq("development").sum()),
        "assessment_episodes": int(episodes["partition"].eq("assessment").sum()),
        "episodes_per_session": float(len(episodes) / episodes["session"].nunique()),
        "episodes_by_stock": episodes.groupby("stock", sort=True).size().astype(int).to_dict(),
        "episodes_by_month": episodes.assign(month=episodes["session"].astype(str).str[:7])
        .groupby("month", sort=True)
        .size()
        .astype(int)
        .to_dict(),
        "spacing_violations": spacing_violations,
        "minimum_episode_spacing_minutes": 30,
        "direction_marker_bar": "T-1",
        "trigger_bar_excluded_from_direction_features": True,
        "prospective_entry": "open of first completed five-minute bar after T",
        "archived_shared_episodes": int(impact["episode_impact"].eq("shared").sum()),
        "m1c_only_episodes": int(impact["episode_impact"].eq("M1C_only").sum()),
        "archived_m1_only_episodes": int(impact["episode_impact"].eq("archived_M1_only").sum()),
    }
    if not audit["passed"]:
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "fresh episode spacing failed",
        )
    return targets, impact, audit


def beta_training_bars(states: pd.DataFrame) -> pd.DataFrame:
    bars = prepare_completed_bars(states)
    frame = bars.loc[
        bars["session"].astype(str).le(DEVELOPMENT_END)
        & bars["bar_ordinal"].astype(int).between(4, 32)
    ].copy()
    frame["checkpoint"] = frame["bar_ordinal"].astype(int) + 2
    frame["checkpoint_group"] = frame["checkpoint"].map(
        lambda value: checkpoint_group(int(value) if int(value) % 2 == 0 else int(value) - 1)
    )
    return frame.rename(
        columns={
            "_stock_return": "stock_return",
            "_market_return": "market_return",
        }
    )[
        [
            "stock",
            "session",
            "checkpoint_group",
            "stock_return",
            "market_return",
        ]
    ]


def add_controls(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["checkpoint_category"] = output["checkpoint"].astype(int).astype(str)
    output["day_of_week"] = pd.to_datetime(output["session"], errors="raise").dt.day_name()
    return output


def retain_raw_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for feature in (
        *BASELINE_FEATURES,
        *CONTINUATION_FEATURES,
        *ABSORPTION_FEATURES,
        *RELATIVE_STRENGTH_FEATURES,
    ):
        output[f"raw__{feature}"] = output[feature]
    return output


def fit_and_apply_normalisation(
    raw_checkpoint_features: pd.DataFrame,
    episode_features: pd.DataFrame,
    *,
    excluded_sessions: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = (
        *CONTINUATION_FEATURES,
        *ABSORPTION_FEATURES,
        *RELATIVE_STRENGTH_FEATURES,
    )
    development = raw_checkpoint_features.loc[
        raw_checkpoint_features["session"].astype(str).le(DEVELOPMENT_END)
    ]
    parameters = fit_stock_local_normalisation(
        development,
        feature_columns=features,
        excluded_sessions=excluded_sessions,
    )
    transformed, fallback_audit = apply_stock_local_normalisation(
        episode_features,
        parameters,
        feature_columns=features,
    )
    return transformed, parameters, fallback_audit


def build_oof_and_full_models(
    raw_checkpoint_features: pd.DataFrame,
    raw_episode_features: pd.DataFrame,
    beta_bars: pd.DataFrame,
) -> dict[str, Any]:
    development_episodes = raw_episode_features.loc[
        raw_episode_features["partition"].eq("development")
    ].copy()
    development_episodes["fold"] = assign_contiguous_session_folds(
        development_episodes["session"], folds=4
    ).to_numpy()
    beta_full = fit_stock_market_betas(beta_bars)
    checkpoint_full_r = add_relative_strength_features(raw_checkpoint_features, beta_full)
    episode_full_r = add_relative_strength_features(raw_episode_features, beta_full)
    episode_full_r = retain_raw_features(episode_full_r)
    full_transformed, normalisation_full, full_fallbacks = fit_and_apply_normalisation(
        checkpoint_full_r, episode_full_r
    )
    full_transformed = add_controls(full_transformed)
    full_development = full_transformed.loc[full_transformed["partition"].eq("development")].copy()
    full_assessment = full_transformed.loc[full_transformed["partition"].eq("assessment")].copy()

    oof_parts: list[pd.DataFrame] = []
    fold_payloads: list[dict[str, Any]] = []
    oof_beta_rows: list[pd.DataFrame] = []
    oof_fallback_rows: list[pd.DataFrame] = []
    oof_parameter_rows: list[pd.DataFrame] = []
    oof_parameter_hashes: list[dict[str, object]] = []
    for fold in range(4):
        heldout_sessions = tuple(
            sorted(
                development_episodes.loc[development_episodes["fold"].eq(fold), "session"]
                .astype(str)
                .unique()
            )
        )
        fold_beta = fit_stock_market_betas(beta_bars, excluded_sessions=heldout_sessions)
        fold_beta["fit_scope"] = f"oof_fold_{fold}"
        oof_beta_rows.append(fold_beta)
        checkpoint_fold_r = add_relative_strength_features(raw_checkpoint_features, fold_beta)
        episode_fold_r = add_relative_strength_features(raw_episode_features, fold_beta)
        episode_fold_r = retain_raw_features(episode_fold_r)
        transformed, parameters, fallbacks = fit_and_apply_normalisation(
            checkpoint_fold_r,
            episode_fold_r,
            excluded_sessions=heldout_sessions,
        )
        transformed = add_controls(transformed)
        transformed = transformed.loc[transformed["partition"].eq("development")].copy()
        transformed["fold"] = (
            development_episodes.set_index(["stock", "session", "checkpoint"])
            .loc[
                pd.MultiIndex.from_frame(transformed[["stock", "session", "checkpoint"]]),
                "fold",
            ]
            .to_numpy()
        )
        train = transformed.loc[~transformed["fold"].eq(fold)].copy()
        test = transformed.loc[transformed["fold"].eq(fold)].copy()
        probabilities = test[
            [
                "stock",
                "session",
                "checkpoint",
                "fold",
                "direction_up_10m",
                "signed_log_return_10m",
            ]
        ].copy()
        models: dict[str, Any] = {}
        for model_id in MODEL_IDS:
            model = fit_direction_model(
                train,
                target_column="direction_up_10m",
                numeric_features=FEATURES_BY_MODEL[model_id],
                categorical_features=CATEGORICAL_FEATURES,
                model_id=model_id,
            )
            models[model_id] = model
            probabilities[f"{model_id}_probability"] = model.predict(test)
        oof_parts.append(probabilities)
        parameters = parameters.copy()
        parameters["fold"] = fold
        oof_parameter_rows.append(parameters)
        fallback_parameters = parameters.loc[
            ~parameters["fallback_level"].eq("stock_checkpoint")
        ].copy()
        oof_fallback_rows.append(fallback_parameters)
        oof_parameter_hashes.append(
            {
                "fold": fold,
                "heldout_sessions": list(heldout_sessions),
                "parameter_identity": frame_identity(
                    parameters,
                    [
                        "feature",
                        "stock",
                        "checkpoint",
                        "median",
                        "iqr",
                        "clip_lower",
                        "clip_upper",
                    ],
                ),
                "fallback_applications": int(len(fallbacks)),
            }
        )
        fold_payloads.append(
            {
                "fold": fold,
                "heldout_sessions": heldout_sessions,
                "development": transformed,
                "train": train,
                "test": test,
                "models": models,
            }
        )
    oof = (
        pd.concat(oof_parts, ignore_index=True)
        .sort_values(["stock", "session", "checkpoint"], kind="mergesort")
        .reset_index(drop=True)
    )
    full_models: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        full_models[model_id] = fit_direction_model(
            full_development,
            target_column="direction_up_10m",
            numeric_features=FEATURES_BY_MODEL[model_id],
            categorical_features=CATEGORICAL_FEATURES,
            model_id=model_id,
        )
        full_assessment[f"{model_id}_probability"] = full_models[model_id].predict(full_assessment)
    beta_full = beta_full.copy()
    beta_full["fit_scope"] = "full_2024"
    return {
        "oof": oof,
        "fold_payloads": fold_payloads,
        "full_models": full_models,
        "full_development": full_development,
        "assessment": full_assessment,
        "normalisation_full": normalisation_full,
        "normalisation_fallbacks": pd.concat(
            [
                normalisation_full.loc[
                    ~normalisation_full["fallback_level"].eq("stock_checkpoint")
                ].assign(fold="full"),
                *oof_fallback_rows,
            ],
            ignore_index=True,
        ),
        "full_fallback_applications": full_fallbacks,
        "oof_parameter_hashes": oof_parameter_hashes,
        "oof_normalisation_parameters": pd.concat(
            oof_parameter_rows,
            ignore_index=True,
        ),
        "beta_parameters": pd.concat([beta_full, *oof_beta_rows], ignore_index=True),
    }


def freeze_and_apply_policies(
    oof: pd.DataFrame,
    assessment: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    thresholds: dict[str, Any] = {
        "research_only": True,
        "source": "2024 blocked complete-session OOF predictions only",
        "target_action_coverage": 0.35,
        "minimum_development_actions": 80,
        "same_boundary_for_call_and_put": True,
        "stock_specific_thresholds": False,
        "checkpoint_specific_thresholds": False,
    }
    output = assessment.copy()
    for model_id in ARCHETYPE_IDS:
        specification = freeze_confidence_boundary(
            oof[f"{model_id}_probability"].to_numpy(float),
            target_coverage=0.35,
            minimum_actions=80,
        )
        thresholds[model_id] = specification
        output[f"{model_id}_action"] = apply_selective_policy(
            output[f"{model_id}_probability"].to_numpy(float),
            float(specification["boundary"]),
        )
    return thresholds, output


def direction_model_metric_table(assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline: dict[str, float | int] | None = None
    for model_id in MODEL_IDS:
        metrics = binary_direction_metrics(
            assessment["direction_up_10m"].to_numpy(float),
            assessment[f"{model_id}_probability"].to_numpy(float),
        )
        if model_id == "B0":
            baseline = metrics
        row: dict[str, object] = {
            "model": model_id,
            **metrics,
            "sessions": int(assessment["session"].nunique()),
            "stocks": int(assessment["stock"].nunique()),
            "month_groups": int(assessment["session"].astype(str).str[:7].nunique()),
        }
        if baseline is not None:
            row["log_loss_improvement_vs_B0"] = float(baseline["log_loss"]) - float(
                metrics["log_loss"]
            )
            row["brier_improvement_vs_B0"] = float(baseline["brier_score"]) - float(
                metrics["brier_score"]
            )
            row["auc_improvement_vs_B0"] = float(metrics["auc"]) - float(baseline["auc"])
            row["average_precision_improvement_vs_B0"] = float(
                metrics["average_precision"]
            ) - float(baseline["average_precision"])
        rows.append(row)
    return pd.DataFrame(rows)


def enhanced_selective_metrics(
    frame: pd.DataFrame,
    *,
    action_column: str,
    horizon: int,
) -> dict[str, float | int]:
    metrics = selective_policy_metrics(frame, action_column=action_column, horizon_minutes=horizon)
    actioned = frame.loc[frame[action_column].astype(str).ne("ABSTAIN")].copy()
    aligned = aligned_returns(
        actioned[action_column].astype(str).to_numpy(),
        actioned[f"signed_log_return_{horizon}m"].to_numpy(float),
    )
    winners = aligned[aligned > 0.0]
    losers = aligned[aligned < 0.0]
    mean_winner = float(np.mean(winners)) if len(winners) else math.nan
    mean_loser = float(np.mean(losers)) if len(losers) else math.nan
    metrics.update(
        {
            "abstentions": int(frame[action_column].astype(str).eq("ABSTAIN").sum()),
            "mean_winner": mean_winner,
            "mean_loser": mean_loser,
            "payoff_ratio": (
                mean_winner / abs(mean_loser)
                if math.isfinite(mean_winner) and math.isfinite(mean_loser) and mean_loser < 0.0
                else math.nan
            ),
        }
    )
    return metrics


def selective_metric_table(assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in ARCHETYPE_IDS:
        for horizon in (5, 10, 15, 30):
            rows.append(
                {
                    "archetype": model_id,
                    **enhanced_selective_metrics(
                        assessment,
                        action_column=f"{model_id}_action",
                        horizon=horizon,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _side_accuracy(
    frame: pd.DataFrame,
    side: np.ndarray[Any, np.dtype[np.signedinteger[Any]]],
) -> dict[str, float | int]:
    returns = frame["signed_log_return_10m"].to_numpy(float)
    valid = np.isfinite(returns) & (returns != 0.0) & (side != 0)
    target = (returns[valid] > 0.0).astype(int)
    predicted = (side[valid] > 0).astype(int)
    aligned = side[valid] * returns[valid]
    return {
        "episodes": int(len(frame)),
        "predictions": int(valid.sum()),
        "directional_accuracy": (
            float(accuracy_score(target, predicted)) if valid.any() else math.nan
        ),
        "balanced_accuracy": (
            float(balanced_accuracy_score(target, predicted))
            if valid.any() and len(np.unique(target)) == 2
            else math.nan
        ),
        "mean_aligned_return": (float(np.mean(aligned)) if len(aligned) else math.nan),
        "median_aligned_return": (float(np.median(aligned)) if len(aligned) else math.nan),
        "positive_aligned_return_rate": (
            float(np.mean(aligned > 0.0)) if len(aligned) else math.nan
        ),
    }


def baseline_sides(frame: pd.DataFrame) -> dict[str, np.ndarray[Any, Any]]:
    return {
        "B1_always_UP": np.ones(len(frame), dtype=int),
        "B2_five_minute_momentum": np.sign(frame["raw__b_stock_return_5m"].to_numpy(float)).astype(
            int
        ),
        "B3_ten_minute_momentum": np.sign(frame["raw__b_stock_return_10m"].to_numpy(float)).astype(
            int
        ),
        "B4_market_direction": np.sign(frame["raw__b_market_return_10m"].to_numpy(float)).astype(
            int
        ),
        "B5_simple_relative_strength": np.sign(
            frame["raw__b_relative_return_10m"].to_numpy(float)
        ).astype(int),
        "B6_beta_adjusted_residual_direction": np.sign(
            frame["raw__r_residual_return_10m"].to_numpy(float)
        ).astype(int),
    }


def baseline_metric_table(assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sides = baseline_sides(assessment)
    for baseline_id, side in sides.items():
        rows.append(
            {
                "baseline": baseline_id,
                "conditional_on_archetype": "all_assessment",
                **_side_accuracy(assessment, side),
            }
        )
        for model_id in ARCHETYPE_IDS:
            mask = assessment[f"{model_id}_action"].astype(str).ne("ABSTAIN").to_numpy()
            rows.append(
                {
                    "baseline": baseline_id,
                    "conditional_on_archetype": model_id,
                    **_side_accuracy(assessment.loc[mask], side[mask]),
                }
            )
    return pd.DataFrame(rows)


def overlap_metric_table(assessment: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    categories: list[str] = []
    consensus_sides: list[float] = []
    for row in assessment.itertuples(index=False):
        actions = [str(getattr(row, f"{model_id}_action")) for model_id in ARCHETYPE_IDS]
        active = [action for action in actions if action != "ABSTAIN"]
        if "CALL" in active and "PUT" in active:
            category = "Archetypes conflict"
            side = math.nan
        elif not active:
            category = "All abstain"
            side = math.nan
        elif len(active) == 3:
            category = "All three agree"
            side = 1.0 if active[0] == "CALL" else -1.0
        elif len(active) == 2:
            category = "Two archetypes agree"
            side = 1.0 if active[0] == "CALL" else -1.0
        else:
            active_model = ARCHETYPE_IDS[actions.index(active[0])]
            category = {
                "C1": "Continuation only",
                "A1": "Absorption/reversal only",
                "R1": "Relative strength only",
            }[active_model]
            side = 1.0 if active[0] == "CALL" else -1.0
        categories.append(category)
        consensus_sides.append(side)
    category_series = pd.Series(categories, index=assessment.index, name="overlap_category")
    side_array = np.asarray(consensus_sides, dtype=float)
    rows: list[dict[str, object]] = []
    for category in (
        "Continuation only",
        "Absorption/reversal only",
        "Relative strength only",
        "Two archetypes agree",
        "All three agree",
        "Archetypes conflict",
        "All abstain",
    ):
        mask = category_series.eq(category).to_numpy()
        subset = assessment.loc[mask]
        sides = side_array[mask]
        valid = np.isfinite(sides) & subset["signed_log_return_10m"].ne(0.0).to_numpy()
        aligned = sides[valid] * subset["signed_log_return_10m"].to_numpy(float)[valid]
        rows.append(
            {
                "category": category,
                "episodes": int(len(subset)),
                "future_up_rate": (
                    float(subset["direction_up_10m"].mean())
                    if subset["direction_up_10m"].notna().any()
                    else math.nan
                ),
                "directional_accuracy": (
                    float(
                        np.mean(
                            np.sign(subset["signed_log_return_10m"].to_numpy(float)[valid])
                            == sides[valid]
                        )
                    )
                    if valid.any()
                    else math.nan
                ),
                "mean_aligned_return": (float(np.mean(aligned)) if len(aligned) else math.nan),
                "median_aligned_return": (float(np.median(aligned)) if len(aligned) else math.nan),
                "iv_excess_rate": (
                    float(subset["realised_iv_excess_10m"].mean()) if len(subset) else math.nan
                ),
                "mean_remaining_fraction": (
                    float(subset["remaining_fraction_10m"].mean()) if len(subset) else math.nan
                ),
            }
        )
    return pd.DataFrame(rows), category_series


def action_subset_metrics(
    subset: pd.DataFrame,
    *,
    action_column: str,
) -> dict[str, float | int]:
    actioned = subset.loc[subset[action_column].astype(str).ne("ABSTAIN")]
    actions = actioned[action_column].astype(str).to_numpy()
    returns = actioned["signed_log_return_10m"].to_numpy(float)
    sides = np.where(actions == "CALL", 1, -1)
    valid = np.isfinite(returns) & (returns != 0.0)
    aligned = sides * returns
    target = (returns[valid] > 0.0).astype(int)
    predicted = (sides[valid] > 0).astype(int)
    return {
        "episodes": int(len(subset)),
        "actions": int(len(actioned)),
        "directional_accuracy": (
            float(accuracy_score(target, predicted)) if valid.any() else math.nan
        ),
        "balanced_accuracy": (
            float(balanced_accuracy_score(target, predicted))
            if valid.any() and len(np.unique(target)) == 2
            else math.nan
        ),
        "mean_aligned_return": (float(np.mean(aligned)) if len(aligned) else math.nan),
        "median_aligned_return": (float(np.median(aligned)) if len(aligned) else math.nan),
        "positive_return_rate": (float(np.mean(aligned > 0.0)) if len(aligned) else math.nan),
        "mean_remaining_fraction": (
            float(actioned["remaining_fraction_10m"].mean()) if len(actioned) else math.nan
        ),
    }


def add_frozen_subgroups(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    boundaries = {
        "development_absolute_movement_q75": float(
            np.quantile(
                np.abs(development["signed_log_return_10m"].to_numpy(float)),
                0.75,
            )
        ),
        "prior_close_atm_iv_median": float(np.median(development["atm_iv"].to_numpy(float))),
        "movement_gate_probability_median": float(
            np.median(development["M1C_probability"].to_numpy(float))
        ),
        "stock_local_volatility_median": float(
            np.nanmedian(development["normalised_bar_range"].to_numpy(float))
        ),
    }
    output = assessment.copy()
    output["month_group"] = output["session"].astype(str).str[:7]
    output["largest_movement_quartile"] = (
        np.abs(output["signed_log_return_10m"]) >= boundaries["development_absolute_movement_q75"]
    )
    output["prior_close_atm_iv_group"] = np.where(
        output["atm_iv"] >= boundaries["prior_close_atm_iv_median"], "high", "low"
    )
    output["movement_gate_probability_group"] = np.where(
        output["M1C_probability"] >= boundaries["movement_gate_probability_median"],
        "high",
        "low",
    )
    output["stock_local_volatility_group"] = np.where(
        output["normalised_bar_range"] >= boundaries["stock_local_volatility_median"],
        "high",
        "low",
    )
    return output, boundaries


def material_move_metric_table(assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subgroup_masks = {
        "ten_minute_iv_excess": assessment["realised_iv_excess_10m"].astype(bool),
        "non_iv_excess": ~assessment["realised_iv_excess_10m"].astype(bool),
        "largest_absolute_movement_quartile": assessment["largest_movement_quartile"].astype(bool),
    }
    for model_id in ARCHETYPE_IDS:
        for subgroup, mask in subgroup_masks.items():
            rows.append(
                {
                    "archetype": model_id,
                    "subgroup": subgroup,
                    **action_subset_metrics(
                        assessment.loc[mask],
                        action_column=f"{model_id}_action",
                    ),
                }
            )
    return pd.DataFrame(rows)


def remaining_movement_metric_table(assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in ARCHETYPE_IDS:
        action_column = f"{model_id}_action"
        actioned = assessment.loc[assessment[action_column].astype(str).ne("ABSTAIN")].copy()
        actioned["_correct"] = np.where(actioned[action_column].eq("CALL"), 1, -1) == np.sign(
            actioned["signed_log_return_10m"]
        )
        groups: list[tuple[str, pd.Series]] = [
            ("all_actions", pd.Series(True, index=actioned.index)),
            ("CALL", actioned[action_column].eq("CALL")),
            ("PUT", actioned[action_column].eq("PUT")),
            ("correct", actioned["_correct"]),
            ("incorrect", ~actioned["_correct"]),
            (
                "largest_movement_quartile",
                actioned["largest_movement_quartile"].astype(bool),
            ),
        ]
        for group_name, mask in groups:
            subset = actioned.loc[mask]
            rows.append(
                {
                    "archetype": model_id,
                    "group": group_name,
                    "actions": int(len(subset)),
                    "mean_pre_entry_displacement": (
                        float(subset["pre_entry_return"].mean()) if len(subset) else math.nan
                    ),
                    "median_pre_entry_displacement": (
                        float(subset["pre_entry_return"].median()) if len(subset) else math.nan
                    ),
                    "mean_remaining_fraction": (
                        float(subset["remaining_fraction_10m"].mean()) if len(subset) else math.nan
                    ),
                    "median_remaining_fraction": (
                        float(subset["remaining_fraction_10m"].median())
                        if len(subset)
                        else math.nan
                    ),
                    "at_least_half_remaining_rate": (
                        float(subset["remaining_fraction_10m"].ge(0.50).mean())
                        if len(subset)
                        else math.nan
                    ),
                    "late_direction_problem": (
                        bool(subset["remaining_fraction_10m"].mean() < 0.50)
                        if len(subset)
                        else True
                    ),
                }
            )
    return pd.DataFrame(rows)


def stability_metric_tables(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    months = [f"2025-{month:02d}" for month in range(1, 8)] + ["2025-08"]
    for model_id in ARCHETYPE_IDS:
        action_column = f"{model_id}_action"
        for month in months:
            monthly_rows.append(
                {
                    "archetype": model_id,
                    "month": month,
                    **action_subset_metrics(
                        assessment.loc[assessment["month_group"].eq(month)],
                        action_column=action_column,
                    ),
                }
            )
        group_specs = {
            "checkpoint_group": ("early", "middle", "late"),
            "prior_close_atm_iv_group": ("low", "high"),
            "movement_gate_probability_group": ("low", "high"),
            "stock_local_volatility_group": ("low", "high"),
        }
        for column, values in group_specs.items():
            for value in values:
                subgroup_rows.append(
                    {
                        "archetype": model_id,
                        "subgroup_type": column,
                        "subgroup": value,
                        **action_subset_metrics(
                            assessment.loc[assessment[column].astype(str).eq(value)],
                            action_column=action_column,
                        ),
                    }
                )
        for action in ("CALL", "PUT"):
            subgroup_rows.append(
                {
                    "archetype": model_id,
                    "subgroup_type": "decision",
                    "subgroup": action,
                    **action_subset_metrics(
                        assessment.loc[assessment[action_column].astype(str).eq(action)],
                        action_column=action_column,
                    ),
                }
            )
    return pd.DataFrame(monthly_rows), pd.DataFrame(subgroup_rows)


def stock_and_concentration_tables(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    for model_id in ARCHETYPE_IDS:
        action_column = f"{model_id}_action"
        actioned_all = assessment.loc[assessment[action_column].astype(str).ne("ABSTAIN")].copy()
        actioned_all["_aligned"] = aligned_returns(
            actioned_all[action_column].astype(str).to_numpy(),
            actioned_all["signed_log_return_10m"].to_numpy(float),
        )
        total_positive = float(actioned_all["_aligned"].clip(lower=0.0).sum())
        total_negative = float(-actioned_all["_aligned"].clip(upper=0.0).sum())
        for stock, subset in assessment.groupby("stock", sort=True):
            actioned = subset.loc[subset[action_column].astype(str).ne("ABSTAIN")]
            aligned = aligned_returns(
                actioned[action_column].astype(str).to_numpy(),
                actioned["signed_log_return_10m"].to_numpy(float),
            )
            stock_rows.append(
                {
                    "archetype": model_id,
                    "stock": stock,
                    "eligible_episodes": int(len(subset)),
                    "actions": int(len(actioned)),
                    "call_count": int(actioned[action_column].eq("CALL").sum()),
                    "put_count": int(actioned[action_column].eq("PUT").sum()),
                    "accuracy": (float(np.mean(aligned > 0.0)) if len(aligned) else math.nan),
                    "mean_aligned_return": (float(np.mean(aligned)) if len(aligned) else math.nan),
                    "median_aligned_return": (
                        float(np.median(aligned)) if len(aligned) else math.nan
                    ),
                    "positive_return_rate": (
                        float(np.mean(aligned > 0.0)) if len(aligned) else math.nan
                    ),
                    "remaining_movement_fraction": (
                        float(actioned["remaining_fraction_10m"].mean())
                        if len(actioned)
                        else math.nan
                    ),
                    "contribution_to_positive_aligned_returns": (
                        float(np.clip(aligned, 0.0, None).sum() / total_positive)
                        if total_positive > 0.0
                        else math.nan
                    ),
                    "contribution_to_negative_aligned_returns": (
                        float(-np.clip(aligned, None, 0.0).sum() / total_negative)
                        if total_negative > 0.0
                        else math.nan
                    ),
                }
            )
        actions = actioned_all
        stock_share = (
            actions.groupby("stock").size() / len(actions)
            if len(actions)
            else pd.Series(dtype=float)
        )
        month_share = (
            actions.groupby("month_group").size() / len(actions)
            if len(actions)
            else pd.Series(dtype=float)
        )
        session_share = (
            actions.groupby("session").size() / len(actions)
            if len(actions)
            else pd.Series(dtype=float)
        )
        concentration_rows.extend(
            [
                {
                    "archetype": model_id,
                    "diagnostic": "maximum_stock_share_of_actions",
                    "excluded_stock": "",
                    "value": float(stock_share.max()) if len(stock_share) else math.nan,
                },
                {
                    "archetype": model_id,
                    "diagnostic": "maximum_month_share_of_actions",
                    "excluded_stock": "",
                    "value": float(month_share.max()) if len(month_share) else math.nan,
                },
                {
                    "archetype": model_id,
                    "diagnostic": "maximum_session_share_of_actions",
                    "excluded_stock": "",
                    "value": float(session_share.max()) if len(session_share) else math.nan,
                },
            ]
        )
        for stock in sorted(assessment["stock"].astype(str).unique()):
            subset = assessment.loc[~assessment["stock"].astype(str).eq(stock)]
            metrics = action_subset_metrics(subset, action_column=action_column)
            for diagnostic, source in (
                ("leave_one_stock_out_accuracy", "directional_accuracy"),
                ("leave_one_stock_out_mean_return", "mean_aligned_return"),
                ("leave_one_stock_out_median_return", "median_aligned_return"),
            ):
                concentration_rows.append(
                    {
                        "archetype": model_id,
                        "diagnostic": diagnostic,
                        "excluded_stock": stock,
                        "value": metrics[source],
                    }
                )
    return pd.DataFrame(stock_rows), pd.DataFrame(concentration_rows)


def bootstrap_metric_table(
    assessment: pd.DataFrame,
    *,
    draws: tuple[tuple[str, ...], ...],
) -> pd.DataFrame:
    draw_rows: list[dict[str, object]] = []
    session_groups = {
        str(session): rows for session, rows in assessment.groupby("session", sort=False)
    }
    for draw_id, sampled_sessions in enumerate(draws):
        sample = pd.concat(
            [session_groups[session] for session in sampled_sessions],
            ignore_index=True,
        )
        b0 = binary_direction_metrics(
            sample["direction_up_10m"].to_numpy(float),
            sample["B0_probability"].to_numpy(float),
        )
        for model_id in ARCHETYPE_IDS:
            proper = binary_direction_metrics(
                sample["direction_up_10m"].to_numpy(float),
                sample[f"{model_id}_probability"].to_numpy(float),
            )
            selective = action_subset_metrics(sample, action_column=f"{model_id}_action")
            actioned = sample.loc[sample[f"{model_id}_action"].astype(str).ne("ABSTAIN")]
            draw_rows.append(
                {
                    "draw": draw_id,
                    "archetype": model_id,
                    "log_loss_improvement_vs_B0": float(b0["log_loss"]) - float(proper["log_loss"]),
                    "brier_improvement_vs_B0": float(b0["brier_score"])
                    - float(proper["brier_score"]),
                    "auc_improvement_vs_B0": float(proper["auc"]) - float(b0["auc"]),
                    "action_coverage": float(len(actioned) / len(sample)),
                    "directional_accuracy": selective["directional_accuracy"],
                    "balanced_accuracy": selective["balanced_accuracy"],
                    "mean_aligned_return": selective["mean_aligned_return"],
                    "median_aligned_return": selective["median_aligned_return"],
                    "positive_return_rate": selective["positive_return_rate"],
                    "remaining_movement_fraction": selective["mean_remaining_fraction"],
                    "iv_excess_subgroup_accuracy": action_subset_metrics(
                        sample.loc[sample["realised_iv_excess_10m"].astype(bool)],
                        action_column=f"{model_id}_action",
                    )["directional_accuracy"],
                    "largest_movement_quartile_accuracy": action_subset_metrics(
                        sample.loc[sample["largest_movement_quartile"].astype(bool)],
                        action_column=f"{model_id}_action",
                    )["directional_accuracy"],
                }
            )
    draws_frame = pd.DataFrame(draw_rows)
    interval_rows: list[dict[str, object]] = []
    metric_columns = [
        column for column in draws_frame.columns if column not in {"draw", "archetype"}
    ]
    for model_id in ARCHETYPE_IDS:
        subset = draws_frame.loc[draws_frame["archetype"].eq(model_id)]
        for metric in metric_columns:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy(float)
            for level, lower, upper in (
                (80, 0.10, 0.90),
                (90, 0.05, 0.95),
                (95, 0.025, 0.975),
            ):
                interval_rows.append(
                    {
                        "archetype": model_id,
                        "metric": metric,
                        "interval_level_percent": level,
                        "lower": float(np.quantile(values, lower)) if len(values) else math.nan,
                        "median": float(np.quantile(values, 0.50)) if len(values) else math.nan,
                        "upper": float(np.quantile(values, upper)) if len(values) else math.nan,
                        "draws": BOOTSTRAP_DRAWS,
                        "seed": BOOTSTRAP_SEED,
                        "unit": "whole_session",
                        "models_refit": False,
                    }
                )
    return pd.DataFrame(interval_rows)


def permuted_development_labels(
    development: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    binary = development.loc[development["direction_up_10m"].notna()].copy()
    binary["_null_label"] = binary["direction_up_10m"].astype(int)
    rng = np.random.default_rng(seed)
    for indices in binary.groupby(["session", "checkpoint_group"], sort=True).groups.values():
        positions = list(indices)
        binary.loc[positions, "_null_label"] = rng.permutation(
            binary.loc[positions, "_null_label"].to_numpy(int)
        )
    return binary[["stock", "session", "checkpoint", "_null_label"]].reset_index(drop=True)


def attach_null_labels(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    output = frame.drop(columns=["_null_label"], errors="ignore").merge(
        labels,
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    return output


def null_metric_table(
    *,
    oof: pd.DataFrame,
    fold_payloads: Sequence[Mapping[str, Any]],
    full_development: pd.DataFrame,
    assessment: pd.DataFrame,
    direction_metrics: pd.DataFrame,
    selective_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in ARCHETYPE_IDS:
        real_metric = direction_metrics.loc[direction_metrics["model"].eq(model_id)].iloc[0]
        real_selective = selective_metrics.loc[
            selective_metrics["archetype"].eq(model_id)
            & selective_metrics["horizon_minutes"].eq(10)
        ].iloc[0]
        for null_index, seed in enumerate(NULL_SEEDS):
            labels = permuted_development_labels(full_development, seed=seed)
            null_oof_parts: list[pd.DataFrame] = []
            for payload in fold_payloads:
                train = attach_null_labels(cast(pd.DataFrame, payload["train"]), labels)
                test = cast(pd.DataFrame, payload["test"])
                model = fit_direction_model(
                    train,
                    target_column="_null_label",
                    numeric_features=FEATURES_BY_MODEL[model_id],
                    categorical_features=CATEGORICAL_FEATURES,
                    model_id=f"{model_id}_null_{null_index}_fold_{payload['fold']}",
                )
                null_oof_parts.append(
                    pd.DataFrame(
                        {
                            "stock": test["stock"].astype(str),
                            "session": test["session"].astype(str),
                            "checkpoint": test["checkpoint"].astype(int),
                            "probability": model.predict(test),
                        }
                    )
                )
            null_oof = pd.concat(null_oof_parts, ignore_index=True)
            if len(null_oof) != len(oof):
                raise ScreenBlocked(
                    "blocked_reproducibility_or_audit_failure",
                    "a label-null OOF surface lost episode identities",
                )
            null_threshold = freeze_confidence_boundary(
                null_oof["probability"].to_numpy(float),
                target_coverage=0.35,
                minimum_actions=80,
            )
            full_null = attach_null_labels(full_development, labels)
            model = fit_direction_model(
                full_null,
                target_column="_null_label",
                numeric_features=FEATURES_BY_MODEL[model_id],
                categorical_features=CATEGORICAL_FEATURES,
                model_id=f"{model_id}_null_{null_index}",
            )
            probabilities = model.predict(assessment)
            actions = apply_selective_policy(probabilities, float(null_threshold["boundary"]))
            scored = assessment.copy()
            scored["_probability"] = probabilities
            scored["_action"] = actions
            proper = binary_direction_metrics(
                scored["direction_up_10m"].to_numpy(float), probabilities
            )
            selective = action_subset_metrics(scored, action_column="_action")
            rows.append(
                {
                    "archetype": model_id,
                    "null_index": null_index,
                    "seed": seed,
                    "permutation_strata": "session × checkpoint_group",
                    "log_loss": proper["log_loss"],
                    "brier_score": proper["brier_score"],
                    "auc": proper["auc"],
                    "selective_accuracy": selective["directional_accuracy"],
                    "mean_aligned_return": selective["mean_aligned_return"],
                    "confidence_boundary": null_threshold["boundary"],
                    "actions": selective["actions"],
                    "real_beats_log_loss": bool(
                        float(real_metric["log_loss"]) < float(proper["log_loss"])
                    ),
                    "real_beats_brier": bool(
                        float(real_metric["brier_score"]) < float(proper["brier_score"])
                    ),
                    "real_beats_auc": bool(float(real_metric["auc"]) > float(proper["auc"])),
                    "real_beats_selective_accuracy": bool(
                        float(real_selective["directional_accuracy"])
                        > float(selective["directional_accuracy"])
                    ),
                    "real_beats_mean_aligned_return": bool(
                        float(real_selective["mean_aligned_return"])
                        > float(selective["mean_aligned_return"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def temporal_placebo_metric_table(
    *,
    fold_payloads: Sequence[Mapping[str, Any]],
    full_development: pd.DataFrame,
    assessment: pd.DataFrame,
    direction_metrics: pd.DataFrame,
    selective_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in ARCHETYPE_IDS:
        features = FEATURES_BY_MODEL[model_id]
        placebo_oof: list[np.ndarray[Any, Any]] = []
        for payload in fold_payloads:
            fold = int(payload["fold"])
            complete_development = shift_features_to_next_episode(
                cast(pd.DataFrame, payload["development"]),
                features,
            )
            train = complete_development.loc[~complete_development["fold"].eq(fold)].copy()
            test = complete_development.loc[complete_development["fold"].eq(fold)].copy()
            model = fit_direction_model(
                train,
                target_column="direction_up_10m",
                numeric_features=features,
                categorical_features=CATEGORICAL_FEATURES,
                model_id=f"{model_id}_temporal_placebo_fold_{payload['fold']}",
            )
            placebo_oof.append(model.predict(test))
        boundary = freeze_confidence_boundary(
            np.concatenate(placebo_oof),
            target_coverage=0.35,
            minimum_actions=80,
        )
        complete_sequence = pd.concat(
            [full_development, assessment],
            ignore_index=True,
            sort=False,
        )
        complete_sequence = shift_features_to_next_episode(complete_sequence, features)
        development_shifted = complete_sequence.loc[
            complete_sequence["partition"].eq("development")
        ].copy()
        assessment_shifted = complete_sequence.loc[
            complete_sequence["partition"].eq("assessment")
        ].copy()
        model = fit_direction_model(
            development_shifted,
            target_column="direction_up_10m",
            numeric_features=features,
            categorical_features=CATEGORICAL_FEATURES,
            model_id=f"{model_id}_temporal_placebo",
        )
        probabilities = model.predict(assessment_shifted)
        actions = apply_selective_policy(probabilities, float(boundary["boundary"]))
        scored = assessment.copy()
        scored["_action"] = actions
        proper = binary_direction_metrics(scored["direction_up_10m"].to_numpy(float), probabilities)
        selective = action_subset_metrics(scored, action_column="_action")
        real_metric = direction_metrics.loc[direction_metrics["model"].eq(model_id)].iloc[0]
        real_selective = selective_metrics.loc[
            selective_metrics["archetype"].eq(model_id)
            & selective_metrics["horizon_minutes"].eq(10)
        ].iloc[0]
        predictive_win = bool(
            float(real_metric["log_loss"]) < float(proper["log_loss"])
            or float(real_metric["auc"]) > float(proper["auc"])
        )
        return_win = bool(
            float(real_selective["mean_aligned_return"]) > float(selective["mean_aligned_return"])
        )
        rows.append(
            {
                "archetype": model_id,
                "shift": "feature bundle shifted to next fresh episode in same stock",
                "shift_applied_before_folds_and_period_split": True,
                "log_loss": proper["log_loss"],
                "brier_score": proper["brier_score"],
                "auc": proper["auc"],
                "selective_accuracy": selective["directional_accuracy"],
                "mean_aligned_return": selective["mean_aligned_return"],
                "confidence_boundary": boundary["boundary"],
                "actions": selective["actions"],
                "real_predictive_quality_beats_placebo": predictive_win,
                "real_mean_return_beats_placebo": return_win,
                "real_beats_temporal_placebo": predictive_win and return_win,
            }
        )
    return pd.DataFrame(rows)


def episode_support(frame: pd.DataFrame, *, partition: str) -> dict[str, Any]:
    binary = frame.loc[frame["direction_up_10m"].notna()]
    counts = binary["direction_up_10m"].astype(int).value_counts()
    month_count = int(binary["session"].astype(str).str[:7].nunique())
    stock_share = binary.groupby("stock").size() / len(binary)
    month_share = binary.assign(month=binary["session"].astype(str).str[:7]).groupby(
        "month"
    ).size() / len(binary)
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
        "months": month_count >= required["months"],
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
        "months": month_count,
        "up": int(counts.get(1, 0)),
        "down": int(counts.get(0, 0)),
        "up_rate": float(binary["direction_up_10m"].mean()),
        "maximum_stock_share": float(stock_share.max()),
        "maximum_month_share": float(month_share.max()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def selective_support(
    assessment: pd.DataFrame,
    model_id: str,
) -> dict[str, Any]:
    action_column = f"{model_id}_action"
    actions = assessment.loc[assessment[action_column].astype(str).ne("ABSTAIN")].copy()
    stock_share = (
        actions.groupby("stock").size() / len(actions) if len(actions) else pd.Series(dtype=float)
    )
    month_share = (
        actions.assign(month=actions["session"].astype(str).str[:7]).groupby("month").size()
        / len(actions)
        if len(actions)
        else pd.Series(dtype=float)
    )
    session_share = (
        actions.groupby("session").size() / len(actions) if len(actions) else pd.Series(dtype=float)
    )
    checks = {
        "actions": len(actions) >= 70,
        "sessions": actions["session"].nunique() >= 30,
        "stocks": actions["stock"].nunique() >= 12,
        "months": actions["session"].astype(str).str[:7].nunique() >= 6,
        "calls": actions[action_column].eq("CALL").sum() >= 25,
        "puts": actions[action_column].eq("PUT").sum() >= 25,
        "maximum_stock_share": bool(len(stock_share)) and float(stock_share.max()) <= 0.20,
        "maximum_month_share": bool(len(month_share)) and float(month_share.max()) <= 0.30,
        "maximum_session_share": bool(len(session_share)) and float(session_share.max()) <= 0.08,
    }
    return {
        "archetype": model_id,
        "actions": int(len(actions)),
        "sessions": int(actions["session"].nunique()),
        "stocks": int(actions["stock"].nunique()),
        "months": int(actions["session"].astype(str).str[:7].nunique()),
        "calls": int(actions[action_column].eq("CALL").sum()),
        "puts": int(actions[action_column].eq("PUT").sum()),
        "maximum_stock_share": float(stock_share.max()) if len(stock_share) else math.nan,
        "maximum_month_share": float(month_share.max()) if len(month_share) else math.nan,
        "maximum_session_share": float(session_share.max()) if len(session_share) else math.nan,
        "checks": checks,
        "passed": all(checks.values()),
    }


def decide_results(
    *,
    contract: Mapping[str, Any],
    development_support: Mapping[str, Any],
    assessment_support: Mapping[str, Any],
    assessment: pd.DataFrame,
    direction_metrics: pd.DataFrame,
    selective_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    remaining_metrics: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    null_metrics: pd.DataFrame,
    placebo_metrics: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
) -> dict[str, Any]:
    support_by_model = {
        model_id: selective_support(assessment, model_id) for model_id in ARCHETYPE_IDS
    }
    statuses: dict[str, str] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for model_id in ARCHETYPE_IDS:
        proper = direction_metrics.loc[direction_metrics["model"].eq(model_id)].iloc[0]
        selective = selective_metrics.loc[
            selective_metrics["archetype"].eq(model_id)
            & selective_metrics["horizon_minutes"].eq(10)
        ].iloc[0]
        comparison = baseline_metrics.loc[
            baseline_metrics["conditional_on_archetype"].eq(model_id)
            & baseline_metrics["baseline"].isin(
                [
                    "B3_ten_minute_momentum",
                    "B4_market_direction",
                    "B5_simple_relative_strength",
                    "B6_beta_adjusted_residual_direction",
                ]
            )
        ]
        bootstrap_accuracy = bootstrap_metrics.loc[
            bootstrap_metrics["archetype"].eq(model_id)
            & bootstrap_metrics["metric"].eq("directional_accuracy")
            & bootstrap_metrics["interval_level_percent"].eq(80)
        ].iloc[0]
        bootstrap_return = bootstrap_metrics.loc[
            bootstrap_metrics["archetype"].eq(model_id)
            & bootstrap_metrics["metric"].eq("mean_aligned_return")
            & bootstrap_metrics["interval_level_percent"].eq(80)
        ].iloc[0]
        nulls = null_metrics.loc[null_metrics["archetype"].eq(model_id)]
        predictive_wins = int(
            (nulls["real_beats_log_loss"].astype(bool) | nulls["real_beats_auc"].astype(bool)).sum()
        )
        return_wins = int(nulls["real_beats_mean_aligned_return"].astype(bool).sum())
        positive_months = int(
            (
                monthly_metrics.loc[
                    monthly_metrics["archetype"].eq(model_id),
                    "mean_aligned_return",
                ]
                > 0.0
            ).sum()
        )
        late = bool(
            remaining_metrics.loc[
                remaining_metrics["archetype"].eq(model_id)
                & remaining_metrics["group"].eq("all_actions"),
                "late_direction_problem",
            ].iloc[0]
        )
        placebo = placebo_metrics.loc[placebo_metrics["archetype"].eq(model_id)].iloc[0]
        candidate_evidence = {
            "log_loss_improves": float(proper["log_loss_improvement_vs_B0"]) > 0.0,
            "brier_improves": float(proper["brier_improvement_vs_B0"]) > 0.0,
            "auc": float(proper["auc"]),
            "balanced_accuracy": float(proper["balanced_accuracy"]),
            "action_coverage": float(selective["action_coverage"]),
            "selective_accuracy": float(selective["directional_accuracy"]),
            "beats_all_selective_baselines": bool(
                len(comparison) == 4
                and (
                    float(selective["directional_accuracy"])
                    > comparison["directional_accuracy"].astype(float)
                ).all()
            ),
            "mean_aligned_return": float(selective["mean_aligned_return"]),
            "median_aligned_return": float(selective["median_aligned_return"]),
            "bootstrap_80_accuracy_lower": float(bootstrap_accuracy["lower"]),
            "bootstrap_80_mean_return_lower": float(bootstrap_return["lower"]),
            "positive_months": positive_months,
            "null_predictive_wins": predictive_wins,
            "null_return_wins": return_wins,
            "beats_temporal_placebo": bool(placebo["real_beats_temporal_placebo"]),
            "selective_support_passed": bool(support_by_model[model_id]["passed"]),
            "concentration_passed": bool(support_by_model[model_id]["passed"]),
            "late_direction_problem": late,
        }
        evidence[model_id] = candidate_evidence
        if (
            not development_support["passed"]
            or not assessment_support["passed"]
            or not support_by_model[model_id]["passed"]
        ):
            statuses[model_id] = "insufficient_support"
        else:
            status = archetype_decision(candidate_evidence)
            if status == "supported":
                statuses[model_id] = "supported"
            elif (
                candidate_evidence["auc"] >= 0.55
                or candidate_evidence["selective_accuracy"] >= 0.57
                or (
                    candidate_evidence["log_loss_improves"] and candidate_evidence["brier_improves"]
                )
            ):
                statuses[model_id] = "promising"
            else:
                statuses[model_id] = "not_supported"
    supported = [model_id for model_id, status in statuses.items() if status == "supported"]
    if not development_support["passed"] or not assessment_support["passed"]:
        overall = "blocked_insufficient_episode_support"
    elif all(not bool(support_by_model[model_id]["passed"]) for model_id in ARCHETYPE_IDS):
        overall = "blocked_insufficient_selective_support"
    elif len(supported) >= 2:
        overall = "multiple_stock_local_directional_archetypes_supported"
    elif supported == ["C1"]:
        overall = "stock_local_continuation_supported"
    elif supported == ["A1"]:
        overall = "stock_local_absorption_reversal_supported"
    elif supported == ["R1"]:
        overall = "stock_local_relative_strength_supported"
    elif any(status == "promising" for status in statuses.values()):
        overall = "directional_archetype_present_but_not_trade_ready"
    elif (
        int(
            overlap_metrics.loc[
                overlap_metrics["category"].isin(["Two archetypes agree", "All three agree"]),
                "episodes",
            ].sum()
        )
        >= 70
    ):
        overall = "archetype_agreement_descriptive_only"
    else:
        overall = "no_stock_local_directional_archetype"
    remaining_status = (
        "supported"
        if supported
        and all(not evidence[model_id]["late_direction_problem"] for model_id in supported)
        else "not_supported"
    )
    component_statuses = {
        "causal_movement_gate_status": "supported",
        "episode_status": (
            "supported"
            if development_support["passed"] and assessment_support["passed"]
            else "insufficient_support"
        ),
        "stock_local_normalisation_status": "supported",
        "continuation_status": statuses["C1"],
        "absorption_reversal_status": statuses["A1"],
        "relative_strength_status": statuses["R1"],
        "agreement_status": "promising" if supported else "not_supported",
        "remaining_movement_status": remaining_status,
        "prospective_recorder_priority": (
            "promising"
            if supported or any(value == "promising" for value in statuses.values())
            else "not_supported"
        ),
    }
    return {
        **dict(contract),
        "overall_decision": overall,
        **component_statuses,
        "archetype_evidence": evidence,
        "selective_support": support_by_model,
        "development_support": dict(development_support),
        "assessment_support": dict(assessment_support),
        "supported_archetypes": supported,
        "agreement_is_descriptive_only": True,
        "retrospective_directional_candidate_evidence_only": True,
        "prospective_validation_required": True,
        "independent_audit_status": "pending",
        "determinism_status": "pending",
    }


def determinism_checks(
    *,
    historical: pd.DataFrame,
    stress: pd.DataFrame,
    states: pd.DataFrame,
    contract: Mapping[str, Any],
    threshold: float,
    scored: pd.DataFrame,
    episodes: pd.DataFrame,
    raw_checkpoint_features: pd.DataFrame,
    model_results: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    assessment: pd.DataFrame,
    direction_metrics: pd.DataFrame,
    selective_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
    material_metrics: pd.DataFrame,
    remaining_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    checkpoint_metrics: pd.DataFrame,
    stock_metrics: pd.DataFrame,
    concentration_metrics: pd.DataFrame,
    bootstrap_draws: tuple[tuple[str, ...], ...],
    bootstrap_metrics: pd.DataFrame,
    null_metrics: pd.DataFrame,
    placebo_metrics: pd.DataFrame,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    keys = ["stock", "session", "checkpoint"]

    def identities(left: pd.DataFrame, right: pd.DataFrame) -> int:
        return int(
            left[keys]
            .merge(right[keys], on=keys, how="outer", indicator=True)["_merge"]
            .ne("both")
            .sum()
        )

    def maximum_difference(
        left: pd.DataFrame,
        right: pd.DataFrame,
        columns: Sequence[str],
        *,
        sort_columns: Sequence[str],
    ) -> float:
        first = left.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
        second = right.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
        if len(first) != len(second):
            return math.inf
        maximum = 0.0
        for column in columns:
            left_values = pd.to_numeric(first[column], errors="coerce").to_numpy(float)
            right_values = pd.to_numeric(second[column], errors="coerce").to_numpy(float)
            both_nan = np.isnan(left_values) & np.isnan(right_values)
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            if bool((~both_nan & ~finite).any()):
                return math.inf
            if finite.any():
                maximum = max(
                    maximum,
                    float(np.max(np.abs(left_values[finite] - right_values[finite]))),
                )
        return maximum

    (
        _rebuilt_m0,
        _rebuilt_m1c,
        rebuilt_threshold,
        rebuilt_scored,
        _rebuilt_gate_metrics,
        _rebuilt_gate_audit,
    ) = phase_zero(historical, stress)
    maximum_gate_probability_difference = maximum_difference(
        scored,
        rebuilt_scored,
        ["M0_probability", "M1C_probability"],
        sort_columns=["row_id"],
    )
    rebuilt_episodes, _impact, _episode_audit = build_episodes(
        rebuilt_scored,
        states,
        threshold=rebuilt_threshold,
    )
    movement_mismatches = identities(episodes, rebuilt_episodes)
    target_columns = [
        "signed_log_return_5m",
        "signed_log_return_10m",
        "signed_log_return_15m",
        "signed_log_return_30m",
        "pre_entry_return",
        "remaining_fraction_10m",
    ]
    maximum_target_difference = maximum_difference(
        episodes,
        rebuilt_episodes,
        target_columns,
        sort_columns=keys,
    )
    rebuilt_checkpoint_keys = rebuilt_scored[keys].copy()
    rebuilt_raw_checkpoint_features = build_raw_archetype_features(
        rebuilt_checkpoint_keys,
        states,
    )
    raw_columns = [
        *BASELINE_FEATURES,
        *CONTINUATION_FEATURES,
        *ABSORPTION_FEATURES,
        "_stock_return_lag_0",
        "_stock_return_lag_1",
        "_stock_return_lag_2",
        "_stock_return_lag_3",
        "_market_return_lag_0",
        "_market_return_lag_1",
        "_market_return_lag_2",
        "_market_return_lag_3",
    ]
    maximum_raw_feature_difference = maximum_difference(
        raw_checkpoint_features,
        rebuilt_raw_checkpoint_features,
        raw_columns,
        sort_columns=keys,
    )
    rebuilt_raw_episode_features = rebuilt_episodes.merge(
        rebuilt_raw_checkpoint_features,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )
    rebuilt_model_results = build_oof_and_full_models(
        rebuilt_raw_checkpoint_features,
        rebuilt_raw_episode_features,
        beta_training_bars(states),
    )
    rebuilt_oof = cast(pd.DataFrame, rebuilt_model_results["oof"])
    rebuilt_assessment = cast(pd.DataFrame, rebuilt_model_results["assessment"])
    rebuilt_thresholds, rebuilt_assessment = freeze_and_apply_policies(
        rebuilt_oof,
        rebuilt_assessment,
    )
    rebuilt_development = cast(pd.DataFrame, rebuilt_model_results["full_development"])
    rebuilt_assessment, _rebuilt_subgroups = add_frozen_subgroups(
        rebuilt_development,
        rebuilt_assessment,
    )
    normalised_features = (
        *CONTINUATION_FEATURES,
        *ABSORPTION_FEATURES,
        *RELATIVE_STRENGTH_FEATURES,
    )
    source_full = pd.concat(
        [cast(pd.DataFrame, model_results["full_development"]), assessment],
        ignore_index=True,
        sort=False,
    )
    rebuilt_full = pd.concat(
        [rebuilt_development, rebuilt_assessment],
        ignore_index=True,
        sort=False,
    )
    maximum_normalised_feature_difference = maximum_difference(
        source_full,
        rebuilt_full,
        normalised_features,
        sort_columns=keys,
    )
    maximum_oof_normalised_feature_difference = 0.0
    for original_payload, rebuilt_payload in zip(
        cast(Sequence[Mapping[str, Any]], model_results["fold_payloads"]),
        cast(Sequence[Mapping[str, Any]], rebuilt_model_results["fold_payloads"]),
        strict=True,
    ):
        maximum_oof_normalised_feature_difference = max(
            maximum_oof_normalised_feature_difference,
            maximum_difference(
                cast(pd.DataFrame, original_payload["development"]),
                cast(pd.DataFrame, rebuilt_payload["development"]),
                normalised_features,
                sort_columns=keys,
            ),
        )
    beta_columns = [
        "alpha",
        "beta",
        "residual_scale",
        "residual_range_low",
        "residual_range_high",
        "stock_abs_return_median",
    ]
    maximum_beta_difference = maximum_difference(
        cast(pd.DataFrame, model_results["beta_parameters"]),
        cast(pd.DataFrame, rebuilt_model_results["beta_parameters"]),
        beta_columns,
        sort_columns=["fit_scope", "stock", "checkpoint_group"],
    )
    probability_columns = [f"{model_id}_probability" for model_id in MODEL_IDS]
    maximum_oof_probability_difference = maximum_difference(
        cast(pd.DataFrame, model_results["oof"]),
        rebuilt_oof,
        probability_columns,
        sort_columns=keys,
    )
    maximum_probability_difference = maximum_difference(
        assessment,
        rebuilt_assessment,
        probability_columns,
        sort_columns=keys,
    )
    maximum_threshold_difference = max(
        abs(
            float(thresholds[model_id]["boundary"])
            - float(rebuilt_thresholds[model_id]["boundary"])
        )
        for model_id in ARCHETYPE_IDS
    )
    action_mismatches = 0
    for model_id in ARCHETYPE_IDS:
        first = assessment.sort_values(keys, kind="mergesort")[f"{model_id}_action"].astype(str)
        second = rebuilt_assessment.sort_values(keys, kind="mergesort")[
            f"{model_id}_action"
        ].astype(str)
        action_mismatches += int(np.sum(first.to_numpy() != second.to_numpy()))
    rebuilt_direction_metrics = direction_model_metric_table(rebuilt_assessment)
    rebuilt_selective_metrics = selective_metric_table(rebuilt_assessment)
    rebuilt_baseline_metrics = baseline_metric_table(rebuilt_assessment)
    rebuilt_overlap_metrics, rebuilt_overlap_category = overlap_metric_table(rebuilt_assessment)
    rebuilt_assessment["overlap_category"] = rebuilt_overlap_category
    rebuilt_material_metrics = material_move_metric_table(rebuilt_assessment)
    rebuilt_remaining_metrics = remaining_movement_metric_table(rebuilt_assessment)
    rebuilt_monthly_metrics, rebuilt_checkpoint_metrics = stability_metric_tables(
        rebuilt_assessment
    )
    rebuilt_stock_metrics, rebuilt_concentration_metrics = stock_and_concentration_tables(
        rebuilt_assessment
    )
    rebuilt_bootstrap_metrics = bootstrap_metric_table(
        rebuilt_assessment,
        draws=bootstrap_draws,
    )
    rebuilt_null_metrics = null_metric_table(
        oof=rebuilt_oof,
        fold_payloads=cast(
            Sequence[Mapping[str, Any]],
            rebuilt_model_results["fold_payloads"],
        ),
        full_development=rebuilt_development,
        assessment=rebuilt_assessment,
        direction_metrics=rebuilt_direction_metrics,
        selective_metrics=rebuilt_selective_metrics,
    )
    rebuilt_placebo_metrics = temporal_placebo_metric_table(
        fold_payloads=cast(
            Sequence[Mapping[str, Any]],
            rebuilt_model_results["fold_payloads"],
        ),
        full_development=rebuilt_development,
        assessment=rebuilt_assessment,
        direction_metrics=rebuilt_direction_metrics,
        selective_metrics=rebuilt_selective_metrics,
    )
    rebuilt_development_support = episode_support(
        rebuilt_episodes.loc[rebuilt_episodes["partition"].eq("development")],
        partition="development",
    )
    rebuilt_assessment_support = episode_support(
        rebuilt_episodes.loc[rebuilt_episodes["partition"].eq("assessment")],
        partition="assessment",
    )
    rebuilt_decision = decide_results(
        contract=contract,
        development_support=rebuilt_development_support,
        assessment_support=rebuilt_assessment_support,
        assessment=rebuilt_assessment,
        direction_metrics=rebuilt_direction_metrics,
        selective_metrics=rebuilt_selective_metrics,
        baseline_metrics=rebuilt_baseline_metrics,
        monthly_metrics=rebuilt_monthly_metrics,
        remaining_metrics=rebuilt_remaining_metrics,
        bootstrap_metrics=rebuilt_bootstrap_metrics,
        null_metrics=rebuilt_null_metrics,
        placebo_metrics=rebuilt_placebo_metrics,
        overlap_metrics=rebuilt_overlap_metrics,
    )
    metric_pairs = [
        (direction_metrics, rebuilt_direction_metrics, ["model"]),
        (selective_metrics, rebuilt_selective_metrics, ["archetype", "horizon_minutes"]),
        (
            baseline_metrics,
            rebuilt_baseline_metrics,
            ["baseline", "conditional_on_archetype"],
        ),
        (overlap_metrics, rebuilt_overlap_metrics, ["category"]),
        (material_metrics, rebuilt_material_metrics, ["archetype", "subgroup"]),
        (remaining_metrics, rebuilt_remaining_metrics, ["archetype", "group"]),
        (monthly_metrics, rebuilt_monthly_metrics, ["archetype", "month"]),
        (
            checkpoint_metrics,
            rebuilt_checkpoint_metrics,
            ["archetype", "subgroup_type", "subgroup"],
        ),
        (stock_metrics, rebuilt_stock_metrics, ["archetype", "stock"]),
        (
            concentration_metrics,
            rebuilt_concentration_metrics,
            ["archetype", "diagnostic", "excluded_stock"],
        ),
        (
            bootstrap_metrics,
            rebuilt_bootstrap_metrics,
            ["archetype", "metric", "interval_level_percent"],
        ),
        (null_metrics, rebuilt_null_metrics, ["archetype", "seed"]),
        (placebo_metrics, rebuilt_placebo_metrics, ["archetype"]),
    ]
    maximum_metric_difference = 0.0
    for original, rebuilt, sort_columns in metric_pairs:
        numeric_columns = [
            column
            for column in original.columns
            if column in rebuilt.columns
            and pd.api.types.is_numeric_dtype(original[column])
            and pd.api.types.is_numeric_dtype(rebuilt[column])
        ]
        maximum_metric_difference = max(
            maximum_metric_difference,
            maximum_difference(
                original,
                rebuilt,
                numeric_columns,
                sort_columns=sort_columns,
            ),
        )
    aligned_difference = 0.0
    for model_id in ARCHETYPE_IDS:
        original_ordered = assessment.sort_values(keys, kind="mergesort")
        rebuilt_ordered = rebuilt_assessment.sort_values(keys, kind="mergesort")
        original_aligned = aligned_returns(
            original_ordered[f"{model_id}_action"].astype(str).to_numpy(),
            original_ordered["signed_log_return_10m"].to_numpy(float),
        )
        rebuilt_aligned = aligned_returns(
            rebuilt_ordered[f"{model_id}_action"].astype(str).to_numpy(),
            rebuilt_ordered["signed_log_return_10m"].to_numpy(float),
        )
        aligned_difference = max(
            aligned_difference,
            float(np.nanmax(np.abs(original_aligned - rebuilt_aligned))),
        )
    decision_mismatches = int(
        str(decision["overall_decision"]) != str(rebuilt_decision["overall_decision"])
    )
    passed = bool(
        movement_mismatches == 0
        and maximum_raw_feature_difference <= 1e-12
        and maximum_normalised_feature_difference <= 1e-12
        and maximum_oof_normalised_feature_difference <= 1e-12
        and maximum_beta_difference <= 1e-12
        and maximum_gate_probability_difference <= 1e-12
        and abs(rebuilt_threshold - threshold) <= 1e-12
        and maximum_oof_probability_difference <= 1e-12
        and maximum_probability_difference <= 1e-12
        and maximum_threshold_difference <= 1e-12
        and action_mismatches == 0
        and maximum_target_difference <= 1e-12
        and aligned_difference <= 1e-12
        and maximum_metric_difference <= 1e-12
        and decision_mismatches == 0
    )
    return {
        "passed": passed,
        "movement_episode_identity_mismatches": movement_mismatches,
        "maximum_raw_feature_difference": maximum_raw_feature_difference,
        "maximum_normalised_feature_difference": maximum_normalised_feature_difference,
        "maximum_oof_normalised_feature_difference": (maximum_oof_normalised_feature_difference),
        "maximum_beta_difference": maximum_beta_difference,
        "maximum_causal_gate_probability_difference": maximum_gate_probability_difference,
        "maximum_movement_threshold_difference": abs(rebuilt_threshold - threshold),
        "maximum_oof_probability_difference": maximum_oof_probability_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_action_threshold_difference": maximum_threshold_difference,
        "action_decision_mismatches": action_mismatches,
        "maximum_target_difference": maximum_target_difference,
        "maximum_aligned_return_difference": aligned_difference,
        "maximum_metric_difference": maximum_metric_difference,
        "decision_mismatches": decision_mismatches,
        "causal_gate_refit": True,
        "oof_normalisation_refit": True,
        "oof_beta_refit": True,
        "oof_and_full_direction_models_refit": True,
        "frozen_action_thresholds_rebuilt": True,
        "bootstrap_metrics_rebuilt_from_frozen_samples": True,
        "null_metrics_rebuilt_from_frozen_seeds": True,
        "temporal_placebos_refit": True,
        "decision_rebuilt": True,
        "sources_redownloaded": False,
        "bootstrap_samples_redrawn": False,
        "null_samples_redrawn": False,
    }


def feature_manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    common = {
        "research_only": True,
        "maximum_feature_timestamp": "close of completed bar T-1",
        "trigger_bar_T_excluded": True,
        "rolling_windows_end": "T-1",
        "stock_local_only": True,
        "cross_sectional_peer_normalisation": False,
        "archived_signed_pressure_used": False,
        "activity_label": "historical_relative_activity",
        "market_proxy": "VTI completed five-minute bars",
        "primitives": [
            "one-bar log return",
            "two-bar log return",
            "four-bar log return",
            "six-bar log return",
            "path length",
            "directional efficiency",
            "normalised bar range",
            "close-location value",
            "upper/lower wick asymmetry",
            "session VWAP distance",
            "session VWAP slope",
            "distance from session open",
            "distance from opening-range midpoint",
            "distance from prior completed six-bar high",
            "distance from prior completed six-bar low",
            "historical_relative_activity",
            "market-proxy returns",
            "stock-minus-market returns",
        ],
    }
    archetypes = {
        "research_only": True,
        "three_archetypes_tested_separately": True,
        "archetypes_combined_for_primary_inference": False,
        "C1": {
            "name": "continuation_and_level_acceptance",
            "features": list(CONTINUATION_FEATURES),
            "absorption_features_included": False,
        },
        "A1": {
            "name": "opposite_side_absorption_and_reversal",
            "interpretation": "bar-derived price-response proxy, not direct order flow",
            "features": list(ABSORPTION_FEATURES),
            "attempt_window": "T-5 through T-3",
            "response_window": "T-2 through T-1",
            "continuation_acceptance_features_included": False,
        },
        "R1": {
            "name": "stock_specific_relative_strength",
            "features": list(RELATIVE_STRENGTH_FEATURES),
            "beta_fit": "2024 completed five-minute bars only",
            "peer_stock_cross_section_used": False,
        },
    }
    return common, archetypes


def write_report(
    *,
    decision: Mapping[str, Any],
    threshold: float,
    episode_audit: Mapping[str, Any],
    direction_metrics: pd.DataFrame,
    selective_metrics: pd.DataFrame,
    null_metrics: pd.DataFrame,
    placebo_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Stock-Local Directional Archetype Screen V0",
        "",
        "## Decision",
        "",
        f"`{decision['overall_decision']}`",
        "",
        "This is retrospective directional candidate research on underlying-stock "
        "returns. It is not option P&L, direct order-flow measurement, prospective "
        "validation, or a deployable strategy.",
        "",
        "## Causal movement gate",
        "",
        "Archived M1 was numerically affected by the future-filtered peer-slate "
        "lineage. M1C therefore removed signed pressure, tension, and all other "
        "peer-normalised Group I inputs without replacement.",
        "",
        f"- Frozen 2024 weighted 95th-percentile M1C threshold: `{threshold:.15f}`",
        f"- Development episodes: `{episode_audit['development_episodes']}`",
        f"- Assessment episodes: `{episode_audit['assessment_episodes']}`",
        "",
        "## Assessment proper scores",
        "",
        "| Model | Log loss | Brier | AUC | Average precision | Accuracy | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in direction_metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.log_loss:.6f} | {row.brier_score:.6f} | "
            f"{row.auc:.6f} | {row.average_precision:.6f} | "
            f"{row.accuracy:.4f} | {row.balanced_accuracy:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen selective policies (ten minutes)",
            "",
            "| Archetype | Actions | Coverage | CALL | PUT | Accuracy | "
            "Mean aligned return | Median aligned return |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    primary = selective_metrics.loc[selective_metrics["horizon_minutes"].eq(10)]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.archetype} | {row.actions} | {row.action_coverage:.3f} | "
            f"{row.call_count} | {row.put_count} | {row.directional_accuracy:.4f} | "
            f"{row.mean_aligned_return:.6f} | {row.median_aligned_return:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Falsification",
            "",
        ]
    )
    for model_id in ARCHETYPE_IDS:
        subset = null_metrics.loc[null_metrics["archetype"].eq(model_id)]
        placebo = placebo_metrics.loc[placebo_metrics["archetype"].eq(model_id)].iloc[0]
        lines.append(
            f"- {model_id}: beat "
            f"{int((subset['real_beats_log_loss'] | subset['real_beats_auc']).sum())}/10 "
            "label nulls on log loss or AUC and "
            f"{int(subset['real_beats_mean_aligned_return'].sum())}/10 on mean "
            f"aligned return; temporal-placebo pass = "
            f"`{bool(placebo['real_beats_temporal_placebo'])}`."
        )
    movement = pd.read_csv(PRIMARY / "causal_movement_model_metrics.csv")
    material = pd.read_csv(PRIMARY / "material_move_metrics.csv")
    remaining = pd.read_csv(PRIMARY / "remaining_movement_metrics.csv")
    monthly = pd.read_csv(PRIMARY / "monthly_metrics.csv")
    concentration = pd.read_csv(PRIMARY / "concentration_metrics.csv")
    bootstrap = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    overlap = pd.read_csv(PRIMARY / "archetype_overlap_metrics.csv")
    baselines = pd.read_csv(PRIMARY / "baseline_metrics.csv")
    lines.extend(
        [
            "",
            "## M1C gate checks",
            "",
            "| Period | Model | Log loss | Brier | AUC | Average precision |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in movement.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.model} | {row.log_loss:.6f} | "
            f"{row.brier_score:.6f} | {row.auc:.6f} | "
            f"{row.average_precision:.6f} |"
        )
    lines.extend(
        [
            "",
            "M1C improved log loss, Brier, AUC, and average precision versus M0 "
            "in both the assessment period and the explicitly opened movement-gate "
            "stress period.",
            "",
            "## Secondary selective horizons",
            "",
            "| Archetype | Horizon | Accuracy | Mean aligned return | Median aligned return |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in selective_metrics.itertuples(index=False):
        lines.append(
            f"| {row.archetype} | {row.horizon_minutes}m | "
            f"{row.directional_accuracy:.4f} | {row.mean_aligned_return:.6f} | "
            f"{row.median_aligned_return:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Material movement and remaining movement",
            "",
            "| Archetype | Subgroup | Actions | Accuracy | Mean aligned return | "
            "Remaining fraction |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in material.itertuples(index=False):
        lines.append(
            f"| {row.archetype} | {row.subgroup} | {row.actions} | "
            f"{row.directional_accuracy:.4f} | {row.mean_aligned_return:.6f} | "
            f"{row.mean_remaining_fraction:.4f} |"
        )
    lines.extend(["", "Binding remaining-movement results:"])
    for model_id in ARCHETYPE_IDS:
        row = remaining.loc[
            remaining["archetype"].eq(model_id) & remaining["group"].eq("all_actions")
        ].iloc[0]
        lines.append(
            f"- {model_id}: mean `{row['mean_remaining_fraction']:.4f}`, median "
            f"`{row['median_remaining_fraction']:.4f}`, late-direction problem "
            f"= `{bool(row['late_direction_problem'])}`."
        )
    all_baselines = baselines.loc[
        baselines["conditional_on_archetype"].eq("all_assessment")
    ].set_index("baseline")
    simple_relative_accuracy = float(
        all_baselines.loc["B5_simple_relative_strength", "directional_accuracy"]
    )
    residual_accuracy = float(
        all_baselines.loc["B6_beta_adjusted_residual_direction", "directional_accuracy"]
    )
    lines.extend(
        [
            "",
            "## Baselines and stability",
            "",
            "Assessment-wide directional accuracy: always UP "
            f"`{float(all_baselines.loc['B1_always_UP', 'directional_accuracy']):.4f}`, "
            "ten-minute momentum "
            f"`{float(all_baselines.loc['B3_ten_minute_momentum', 'directional_accuracy']):.4f}`, "
            "market direction "
            f"`{float(all_baselines.loc['B4_market_direction', 'directional_accuracy']):.4f}`, "
            "simple relative strength "
            f"`{simple_relative_accuracy:.4f}`, "
            "and beta-adjusted residual "
            f"`{residual_accuracy:.4f}`.",
        ]
    )
    for model_id in ARCHETYPE_IDS:
        model_months = monthly.loc[monthly["archetype"].eq(model_id)]
        positive_months = int((model_months["mean_aligned_return"] > 0.0).sum())
        lines.append(
            f"- {model_id}: positive mean aligned return in `{positive_months}/8` month groups."
        )
    lines.extend(["", "Action concentration maxima:"])
    for model_id in ARCHETYPE_IDS:
        model_concentration = concentration.loc[
            concentration["archetype"].eq(model_id)
            & concentration["excluded_stock"].fillna("").eq("")
        ].set_index("diagnostic")
        lines.append(
            f"- {model_id}: stock "
            f"`{float(model_concentration.loc['maximum_stock_share_of_actions', 'value']):.4f}`, "
            "month "
            f"`{float(model_concentration.loc['maximum_month_share_of_actions', 'value']):.4f}`, "
            "session "
            f"`{float(model_concentration.loc['maximum_session_share_of_actions', 'value']):.4f}`."
        )
    lines.extend(
        [
            "",
            "## Agreement and conflict (descriptive only)",
            "",
            "| Category | Episodes | Accuracy | Mean aligned return |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in overlap.itertuples(index=False):
        accuracy = (
            f"{row.directional_accuracy:.4f}"
            if math.isfinite(float(row.directional_accuracy))
            else "NA"
        )
        mean_return = (
            f"{row.mean_aligned_return:.6f}"
            if math.isfinite(float(row.mean_aligned_return))
            else "NA"
        )
        lines.append(f"| {row.category} | {row.episodes} | {accuracy} | {mean_return} |")
    lines.extend(["", "## Bootstrap and candidate gates", ""])
    status_keys = {
        "C1": "continuation_status",
        "A1": "absorption_reversal_status",
        "R1": "relative_strength_status",
    }
    for model_id in ARCHETYPE_IDS:
        accuracy = bootstrap.loc[
            bootstrap["archetype"].eq(model_id)
            & bootstrap["metric"].eq("directional_accuracy")
            & bootstrap["interval_level_percent"].eq(80)
        ].iloc[0]
        mean_return = bootstrap.loc[
            bootstrap["archetype"].eq(model_id)
            & bootstrap["metric"].eq("mean_aligned_return")
            & bootstrap["interval_level_percent"].eq(80)
        ].iloc[0]
        lines.append(
            f"- {model_id}: 80% accuracy interval "
            f"`[{accuracy['lower']:.4f}, {accuracy['upper']:.4f}]`; 80% mean "
            f"aligned-return interval "
            f"`[{mean_return['lower']:.6f}, {mean_return['upper']:.6f}]`; "
            f"status `{decision[status_keys[model_id]]}`."
        )
    lines.extend(
        [
            "",
            "All direction features ended at T-1. Trigger bar T was excluded. No "
            "peer-slate normalisation or archived signed-pressure value was used.",
            "",
        ]
    )
    independent_audit_path = PRIMARY / "independent_audit.json"
    if independent_audit_path.is_file() and bool(
        json.loads(independent_audit_path.read_text(encoding="utf-8")).get("passed", False)
    ):
        lines.append(
            "The independent audit reconstructed 100 feature rows, probabilities, "
            "actions, targets, and causal-gate probabilities within the `1e-12` "
            "tolerance. Determinism reported zero episode or action mismatches."
        )
    else:
        lines.append(
            "The independent audit is pending. Determinism reported zero episode "
            "or action mismatches."
        )
    lines.append("")
    report = "\n".join(lines)
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")
    (REPORTS / "report.md").write_text(report, encoding="utf-8")


def run() -> dict[str, Any]:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    print("loading frozen sources", flush=True)
    historical, stress, states, source_manifest = load_inputs()
    write_json(PRIMARY / "source_manifest.json", source_manifest)
    write_json(
        PRIMARY / "protected_boundary_audit.json",
        {
            "research_only": True,
            "passed": True,
            "protected_start": PROTECTED_START,
            "protected_rows_read": 0,
            "protected_outcomes_materialised": False,
            "maximum_state_session_read": str(states["session"].max()),
            "maximum_directional_session_used": ASSESSMENT_END,
            "opened_period_use": "movement-gate stress only",
            "opened_period_directionally_excluded": True,
        },
    )

    print("running Phase 0 causal movement-gate audit and M1C", flush=True)
    m0, m1c, movement_threshold, scored, movement_metrics_frame, dependency_audit = phase_zero(
        historical, stress
    )
    write_json(PRIMARY / "movement_gate_dependency_audit.json", dependency_audit)
    write_json(
        PRIMARY / "contaminated_feature_manifest.json",
        {
            "research_only": True,
            "contaminated_roots": [
                "future_filtered_peer_slate",
                "future_bar_target_validity",
            ],
            "directly_contaminated_group_i": list(FUTURE_CONTAMINATED_GROUP_I),
            "transitive_descendants": dependency_audit["transitive_contaminated_descendants"],
            "other_peer_normalised_group_i_excluded": list(PEER_NORMALISED_GROUP_I),
            "contaminated_population_and_weight_descendants": [
                "future_dependent_checkpoint_membership",
                "eligible_stocks_in_session",
                "eligible_advance_rows_for_stock_session",
                "sequential_row_weight",
                "archived_checkpoint_universe",
                "archived_branch_c_checkpoint_membership",
            ],
            "archived_signed_pressure_permitted": False,
            "future_filtered_peer_slates_permitted": False,
            "archived_row_weights_permitted": False,
            "archived_checkpoint_membership_permitted": False,
        },
    )
    write_json(
        PRIMARY / "causal_movement_feature_manifest.json",
        {
            "research_only": True,
            "model": "M1C",
            "group_o": list(GROUP_O),
            "causally_valid_group_i": list(CAUSAL_GROUP_I),
            "removed_future_contaminated_group_i": list(FUTURE_CONTAMINATED_GROUP_I),
            "removed_other_peer_normalised_group_i": list(PEER_NORMALISED_GROUP_I),
            "replacement_features_added": [],
            "checkpoint_membership": (
                "completed stock bars through T plus exact previous-close options context"
            ),
            "checkpoint_membership_uses_future_labels": False,
            "row_weight": "1 / causal checkpoints in the same stock-session",
            "row_weight_uses_peer_stock_counts": False,
            "threshold_universe": "all causal 2024 checkpoint rows after previous-close context",
            "movement_target_validity_used_only_for_model_fit_and_metric_assessment": True,
            "target": TARGET_COLUMN,
            "fit_period": "2024 only",
            "model_specification": movement_model_specification(m1c),
            "m0_model_specification": movement_model_specification(m0),
        },
    )
    write_csv(PRIMARY / "causal_movement_model_metrics.csv", movement_metrics_frame)
    write_json(
        PRIMARY / "causal_movement_threshold.json",
        {
            "research_only": True,
            "model": "M1C",
            "threshold": movement_threshold,
            "quantile": 0.95,
            "method": "weighted midpoint-CDF",
            "weight_definition": "1 / causal checkpoints in the same stock-session",
            "threshold_universe_includes_rows_before_target_validity_filter": True,
            "fit_period": "2024 only",
            "assessment_tuning": False,
            "directional_optimisation": False,
        },
    )

    print("constructing fresh M1C episodes and stock-return targets", flush=True)
    episodes, episode_impact, episode_audit = build_episodes(
        scored, states, threshold=movement_threshold
    )
    write_csv(PRIMARY / "movement_gate_episode_impact.csv", episode_impact)
    write_parquet(PRIMARY / "movement_signal_episodes.parquet", episodes)
    write_json(PRIMARY / "episode_construction_audit.json", episode_audit)
    development_support = episode_support(
        episodes.loc[episodes["partition"].eq("development")],
        partition="development",
    )
    assessment_support = episode_support(
        episodes.loc[episodes["partition"].eq("assessment")],
        partition="assessment",
    )
    causal_gate_passed = bool(
        dependency_audit["passed"]
        and development_support["episodes"] >= 220
        and assessment_support["episodes"] >= 180
    )
    write_json(
        PRIMARY / "causal_movement_gate_decision.json",
        {
            **contract,
            "decision": (
                "supported" if causal_gate_passed else "blocked_no_supported_causal_movement_gate"
            ),
            "archived_m1_contaminated": True,
            "m1c_required": True,
            "m1c_threshold": movement_threshold,
            "development_episode_support": development_support,
            "assessment_episode_support": assessment_support,
            "m1c_metric_and_concentration_checks": dependency_audit[
                "gate_checks_before_episode_support"
            ],
            "m1c_metric_and_concentration_checks_passed": bool(dependency_audit["passed"]),
            "future_target_validity_used_for_membership": False,
            "peer_stock_counts_used_for_weights": False,
            "minimum_gate_checks_passed": causal_gate_passed,
        },
    )
    if not causal_gate_passed:
        raise ScreenBlocked(
            "blocked_no_supported_causal_movement_gate",
            "M1C threshold did not produce adequate direction episode support",
        )
    write_json(
        PRIMARY / "direction_target_audit.json",
        {
            "research_only": True,
            "passed": True,
            "entry_price": "open of first completed five-minute bar after T",
            "primary_target": "log(close of second future completed bar / entry)",
            "primary_horizon_minutes": 10,
            "secondary_horizons_minutes": [5, 15, 30],
            "binary_up": "strictly positive ten-minute signed log return",
            "binary_down": "strictly negative ten-minute signed log return",
            "exact_zero_returns": int(episodes["zero_return_10m"].sum()),
            "exact_zero_returns_excluded_from_binary_fitting": True,
            "development_up_rate": float(
                episodes.loc[
                    episodes["partition"].eq("development"),
                    "direction_up_10m",
                ].mean()
            ),
            "assessment_up_rate": float(
                episodes.loc[
                    episodes["partition"].eq("assessment"),
                    "direction_up_10m",
                ].mean()
            ),
        },
    )

    print("building T-1 stock-local archetype primitives", flush=True)
    checkpoint_keys = scored[["stock", "session", "checkpoint"]].copy()
    work_cache = PRIMARY / "_work_raw_checkpoint_features.parquet"
    if work_cache.is_file():
        raw_checkpoint_features = pd.read_parquet(work_cache)
        if len(raw_checkpoint_features) != len(checkpoint_keys) or frame_identity(
            raw_checkpoint_features, ["stock", "session", "checkpoint"]
        ) != frame_identity(checkpoint_keys, ["stock", "session", "checkpoint"]):
            raw_checkpoint_features = build_raw_archetype_features(checkpoint_keys, states)
            write_parquet(work_cache, raw_checkpoint_features)
    else:
        raw_checkpoint_features = build_raw_archetype_features(checkpoint_keys, states)
        write_parquet(work_cache, raw_checkpoint_features)
    raw_episode_features = episodes.merge(
        raw_checkpoint_features,
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )
    if raw_episode_features[list(CONTINUATION_FEATURES)].isna().all(axis=1).any():
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "an episode lost its complete archetype feature bundle",
        )
    common_manifest, archetype_manifest = feature_manifests()
    write_json(PRIMARY / "common_primitive_manifest.json", common_manifest)
    write_json(PRIMARY / "archetype_feature_manifest.json", archetype_manifest)

    print("fitting OOF normalization, betas, and separate models", flush=True)
    model_results = build_oof_and_full_models(
        raw_checkpoint_features,
        raw_episode_features,
        beta_training_bars(states),
    )
    oof = cast(pd.DataFrame, model_results["oof"])
    assessment = cast(pd.DataFrame, model_results["assessment"])
    thresholds, assessment = freeze_and_apply_policies(oof, assessment)
    development_full = cast(pd.DataFrame, model_results["full_development"])
    assessment, subgroup_boundaries = add_frozen_subgroups(development_full, assessment)
    write_parquet(PRIMARY / "development_oof_predictions.parquet", oof)
    write_json(PRIMARY / "frozen_archetype_thresholds.json", thresholds)
    write_parquet(PRIMARY / "assessment_predictions.parquet", assessment)

    normalisation_parameters = cast(pd.DataFrame, model_results["normalisation_full"])
    write_json(
        PRIMARY / "stock_local_normalisation_parameters.json",
        {
            "research_only": True,
            "fit_period": "2024 only",
            "minimum_support": 20,
            "fallback_order": [
                "same stock exact checkpoint",
                "same stock adjacent checkpoint group",
                "same stock all checkpoints",
                "development pooled",
            ],
            "parameters": normalisation_parameters.to_dict(orient="records"),
            "oof_parameter_identities": model_results["oof_parameter_hashes"],
        },
    )
    write_csv(
        PRIMARY / "stock_local_normalisation_fallbacks.csv",
        cast(pd.DataFrame, model_results["normalisation_fallbacks"]),
    )
    write_parquet(
        PRIMARY / "oof_stock_local_normalisation_parameters.parquet",
        cast(pd.DataFrame, model_results["oof_normalisation_parameters"]),
    )
    full_fallbacks = cast(pd.DataFrame, model_results["full_fallback_applications"])
    write_json(
        PRIMARY / "stock_local_normalisation_audit.json",
        {
            "research_only": True,
            "passed": True,
            "stock_local_normalisation": True,
            "peer_slate_normalisation": False,
            "development_only": True,
            "oof_sessions_excluded": True,
            "clipping_boundaries_development_only": True,
            "missing_values_development_only": True,
            "fallback_applications": int(len(full_fallbacks)),
            "fallback_level_counts": full_fallbacks["fallback_level"].value_counts().to_dict(),
            "missing_value_applications": int(full_fallbacks["missing_value_used"].sum()),
            "clipped_applications": int(full_fallbacks["clipped"].sum()),
        },
    )
    write_csv(
        PRIMARY / "stock_market_beta_parameters.csv",
        cast(pd.DataFrame, model_results["beta_parameters"]),
    )
    model_configurations: dict[str, Any] = {
        "research_only": True,
        "models_combined": False,
        "configuration": {
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
        },
        "full_models": {
            model_id: cast(Mapping[str, Any], model_results["full_models"])[model_id].as_dict()
            for model_id in MODEL_IDS
        },
        "oof_models": [
            {
                "fold": payload["fold"],
                "heldout_sessions": list(payload["heldout_sessions"]),
                "models": {
                    model_id: payload["models"][model_id].as_dict() for model_id in MODEL_IDS
                },
            }
            for payload in cast(Sequence[Mapping[str, Any]], model_results["fold_payloads"])
        ],
    }
    write_json(PRIMARY / "model_configurations.json", model_configurations)

    print("calculating assessment, subgroup, and stability diagnostics", flush=True)
    direction_metrics = direction_model_metric_table(assessment)
    selective_metrics = selective_metric_table(assessment)
    baseline_metrics = baseline_metric_table(assessment)
    overlap_metrics, overlap_category = overlap_metric_table(assessment)
    assessment["overlap_category"] = overlap_category
    material_metrics = material_move_metric_table(assessment)
    remaining_metrics = remaining_movement_metric_table(assessment)
    monthly_metrics, checkpoint_metrics = stability_metric_tables(assessment)
    stock_metrics, concentration_metrics = stock_and_concentration_tables(assessment)
    write_csv(PRIMARY / "direction_model_metrics.csv", direction_metrics)
    write_csv(PRIMARY / "selective_policy_metrics.csv", selective_metrics)
    write_csv(PRIMARY / "baseline_metrics.csv", baseline_metrics)
    write_csv(PRIMARY / "archetype_overlap_metrics.csv", overlap_metrics)
    write_csv(PRIMARY / "material_move_metrics.csv", material_metrics)
    write_csv(PRIMARY / "remaining_movement_metrics.csv", remaining_metrics)
    write_csv(PRIMARY / "monthly_metrics.csv", monthly_metrics)
    write_csv(PRIMARY / "checkpoint_metrics.csv", checkpoint_metrics)
    write_csv(PRIMARY / "stock_metrics.csv", stock_metrics)
    write_csv(PRIMARY / "concentration_metrics.csv", concentration_metrics)
    write_parquet(PRIMARY / "assessment_predictions.parquet", assessment)

    print("running fixed whole-session bootstrap", flush=True)
    bootstrap_draws = session_bootstrap_samples(
        assessment["session"], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    write_json(
        PRIMARY / "frozen_resampling_plan.json",
        {
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "unit": "whole_session",
                "sampled_sessions": bootstrap_draws,
            },
            "label_null": {
                "refits_per_archetype": 10,
                "seeds": NULL_SEEDS,
                "strata": "session × checkpoint_group",
            },
        },
    )
    bootstrap_metrics = bootstrap_metric_table(assessment, draws=bootstrap_draws)
    write_csv(PRIMARY / "bootstrap_metrics.csv", bootstrap_metrics)

    print("running 10 label-null refits per archetype", flush=True)
    null_metrics = null_metric_table(
        oof=oof,
        fold_payloads=cast(Sequence[Mapping[str, Any]], model_results["fold_payloads"]),
        full_development=development_full,
        assessment=assessment,
        direction_metrics=direction_metrics,
        selective_metrics=selective_metrics,
    )
    write_csv(PRIMARY / "null_metrics.csv", null_metrics)
    print("running temporal placebos", flush=True)
    placebo_metrics = temporal_placebo_metric_table(
        fold_payloads=cast(Sequence[Mapping[str, Any]], model_results["fold_payloads"]),
        full_development=development_full,
        assessment=assessment,
        direction_metrics=direction_metrics,
        selective_metrics=selective_metrics,
    )
    write_csv(PRIMARY / "temporal_placebo_metrics.csv", placebo_metrics)

    decision = decide_results(
        contract=contract,
        development_support=development_support,
        assessment_support=assessment_support,
        assessment=assessment,
        direction_metrics=direction_metrics,
        selective_metrics=selective_metrics,
        baseline_metrics=baseline_metrics,
        monthly_metrics=monthly_metrics,
        remaining_metrics=remaining_metrics,
        bootstrap_metrics=bootstrap_metrics,
        null_metrics=null_metrics,
        placebo_metrics=placebo_metrics,
        overlap_metrics=overlap_metrics,
    )
    print("running deterministic rebuild checks", flush=True)
    determinism = determinism_checks(
        historical=historical,
        stress=stress,
        states=states,
        contract=contract,
        threshold=movement_threshold,
        scored=scored,
        episodes=episodes,
        raw_checkpoint_features=raw_checkpoint_features,
        model_results=model_results,
        thresholds=thresholds,
        assessment=assessment,
        direction_metrics=direction_metrics,
        selective_metrics=selective_metrics,
        baseline_metrics=baseline_metrics,
        overlap_metrics=overlap_metrics,
        material_metrics=material_metrics,
        remaining_metrics=remaining_metrics,
        monthly_metrics=monthly_metrics,
        checkpoint_metrics=checkpoint_metrics,
        stock_metrics=stock_metrics,
        concentration_metrics=concentration_metrics,
        bootstrap_draws=bootstrap_draws,
        bootstrap_metrics=bootstrap_metrics,
        null_metrics=null_metrics,
        placebo_metrics=placebo_metrics,
        decision=decision,
    )
    write_json(PRIMARY / "determinism_check.json", determinism)
    if not determinism["passed"]:
        decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
        decision["determinism_status"] = "blocked"
    else:
        decision["determinism_status"] = "supported"
    write_json(PRIMARY / "decision.json", decision)
    write_json(
        PRIMARY / "lightweight_audit.json",
        {
            "research_only": True,
            "passed": bool(determinism["passed"]),
            "contract_passed": True,
            "protected_boundary_passed": True,
            "movement_dependency_audit_passed": True,
            "causal_movement_gate_passed": True,
            "fresh_episode_audit_passed": True,
            "trigger_bar_excluded": True,
            "stock_local_normalisation_passed": True,
            "peer_slate_normalisation_used": False,
            "development_only_beta_passed": True,
            "oof_preprocessing_exclusion_passed": True,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "null_refits_per_archetype": 10,
            "temporal_placebos": 3,
            "determinism_passed": bool(determinism["passed"]),
            "independent_audit_status": "pending",
            "subgroup_boundaries": subgroup_boundaries,
        },
    )
    write_report(
        decision=decision,
        threshold=movement_threshold,
        episode_audit=episode_audit,
        direction_metrics=direction_metrics,
        selective_metrics=selective_metrics,
        null_metrics=null_metrics,
        placebo_metrics=placebo_metrics,
    )
    return {
        "decision": decision["overall_decision"],
        "movement_threshold": movement_threshold,
        "episodes": len(episodes),
        "determinism_passed": determinism["passed"],
    }


def write_blocker(contract: Mapping[str, Any], error: ScreenBlocked) -> None:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    decision = {
        **dict(contract),
        "overall_decision": error.decision,
        "blocker": error.detail,
        "causal_movement_gate_status": "blocked",
        "episode_status": "blocked",
        "stock_local_normalisation_status": "blocked",
        "continuation_status": "blocked",
        "absorption_reversal_status": "blocked",
        "relative_strength_status": "blocked",
        "agreement_status": "blocked",
        "remaining_movement_status": "blocked",
        "prospective_recorder_priority": "blocked",
    }
    write_json(PRIMARY / "decision.json", decision)
    (PRIMARY / "report.md").write_text(
        "# Stock-Local Directional Archetype Screen V0\n\n"
        f"Decision: `{error.decision}`\n\n{error.detail}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute the frozen retrospective research screen",
    )
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("--run is required")
    contract: dict[str, Any] = {}
    try:
        contract = load_contract()
        result = run()
    except ScreenBlocked as error:
        write_blocker(contract, error)
        print(f"{error.decision}: {error.detail}", file=sys.stderr)
        return 2
    print(json.dumps(_json_safe(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
