#!/usr/bin/env python3
"""Run the preregistered M1C Tail Phase V1 structural assessment."""

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

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

import numpy as np
import pandas as pd

EXPERIMENT_DIR: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = EXPERIMENT_DIR.parents[2]
PRIMARY: Final[Path] = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS: Final[Path] = EXPERIMENT_DIR / "reports"
SOURCE_EXPERIMENT: Final[Path] = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0"
)
SOURCE_PRIMARY: Final[Path] = SOURCE_EXPERIMENT / "artifacts" / "primary"
SOURCE_RUNNER: Final[Path] = SOURCE_EXPERIMENT / "run_screen_v0.py"

for _package in ("stocker_prospective", "stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_prospective.contract import M1C_FROZEN_THRESHOLD
from stocker_prospective.direction import FrozenDirectionRuntime
from stocker_prospective.direction_features import (
    FrozenDirectionFeatureBuilder,
    checkpoint_group,
)
from stocker_prospective.frozen_m1c import FrozenM1CRuntime
from stocker_prospective.m1c_features import FROZEN_CHECKPOINTS
from stocker_prospective.tail_phase_v1 import (
    M1C_TAIL_PHASE_V1_VERSION,
    assert_tail_phase_unprotected_sessions,
)
from stocker_research.m1c_tail_phase_v1 import (
    FrozenConsumedMedianV1,
    apply_frozen_consumed_bucket_v1,
    attach_canonical_tail_outcomes_v1,
    attach_frozen_a1_and_regime_v1,
    build_tail_phase_checkpoint_rows_v1,
    construct_fresh_tail_episodes_v1,
    freeze_movement_consumed_median_v1,
    score_frozen_m1c_checkpoint_rows_v1,
)

DEVELOPMENT_START: Final[str] = "2024-01-01"
DEVELOPMENT_END: Final[str] = "2024-12-31"
ASSESSMENT_START: Final[str] = "2025-01-01"
ASSESSMENT_END: Final[str] = "2025-08-22"
STRESS_START: Final[str] = "2025-09-01"
STRESS_END: Final[str] = "2025-12-31"
PROTECTED_START: Final[str] = "2026-01-01"
BOOTSTRAP_SEED: Final[int] = 20260728
BOOTSTRAP_DRAWS: Final[int] = 1000
CONFIDENCE_LEVEL: Final[float] = 0.95
MINIMUM_CELL_ROWS: Final[int] = 30
MINIMUM_CELL_SESSIONS: Final[int] = 10
MINIMUM_A1_ACTION_ROWS: Final[int] = 30
RUN_COMMAND: Final[str] = (
    "rtk uv run python research/directional-readiness/"
    "20260728-m1c-tail-phase-v1/run_tail_phase_v1.py"
)
PHASE_ORDER: Final[tuple[str, ...]] = (
    "FIRST_ENTRY",
    "PERSISTENT",
    "RE_ENTRY",
)
BUCKET_ORDER: Final[tuple[str, ...]] = ("LOW_OR_EQUAL", "HIGH")
PRIMARY_QUANTITIES: Final[dict[str, str]] = {
    "future_absolute_10m": "future_10m_absolute_return_v1",
    "future_absolute_15m": "future_15m_absolute_movement_v1",
    "post_share_of_local_range_v1": "post_share_of_local_range_v1",
}


class TailPhaseRunBlocked(RuntimeError):
    """A fail-closed operational or chronology blocker."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "tail_phase_v1_frozen_source_loader",
        SOURCE_RUNNER,
    )
    if specification is None or specification.loader is None:
        raise TailPhaseRunBlocked("unable to load the frozen source runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_permitted_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Call only the prior study's hash-verified loader, never its fitting phases."""

    module = _load_source_module()
    loader = getattr(module, "load_inputs", None)
    if not callable(loader):
        raise TailPhaseRunBlocked("frozen source loader is unavailable")
    historical, stress, bars, source_manifest = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]],
        loader(),
    )
    for frame in (historical, stress, bars):
        assert_tail_phase_unprotected_sessions(frame["session"])
    maximum = max(
        str(historical["session"].max()),
        str(stress["session"].max()),
        str(bars["session"].max()),
    )
    if maximum > STRESS_END:
        raise TailPhaseRunBlocked("a source crossed the opened stress boundary")
    if int(source_manifest.get("protected_rows_read", -1)) != 0:
        raise TailPhaseRunBlocked("source manifest does not prove zero protected rows")
    source_manifest = dict(source_manifest)
    dense_source = next(
        (
            Path(str(item["path"]))
            for item in cast(list[dict[str, Any]], source_manifest["sources"])
            if item.get("role") == "unfiltered_dense_causal_checkpoint_surface"
        ),
        None,
    )
    if dense_source is None:
        raise TailPhaseRunBlocked("unfiltered causal checkpoint identity is unavailable")
    dense_sessions = pd.read_parquet(dense_source, columns=["session"])["session"].astype(str)
    if bool(dense_sessions.ge(PROTECTED_START).any()):
        raise TailPhaseRunBlocked("unfiltered checkpoint identity crossed into protected data")
    joined_sessions = historical["session"].astype(str)
    coverage: dict[str, dict[str, int | float | str]] = {}
    for period, start, end in (
        ("development", DEVELOPMENT_START, DEVELOPMENT_END),
        ("assessment", ASSESSMENT_START, ASSESSMENT_END),
    ):
        before = int(dense_sessions.between(start, end).sum())
        after = int(joined_sessions.between(start, end).sum())
        missing = before - after
        coverage[period] = {
            "canonical_causal_checkpoint_rows": before,
            "rows_with_previous_close_option_context": after,
            "rows_without_previous_close_option_context": missing,
            "context_coverage_rate": float(after / before) if before else math.nan,
            "missing_row_state": "M1C_probability_undefined_not_scored",
        }
    coverage["stress"] = {
        "canonical_causal_checkpoint_rows": int(
            source_manifest["causal_opened_stress_surface"]["causal_checkpoint_rows_before_options"]
        ),
        "rows_with_previous_close_option_context": int(len(stress)),
        "rows_without_previous_close_option_context": 0,
        "context_coverage_rate": 1.0,
        "missing_row_state": ("source_membership_conditioned_on_available_previous_close_options"),
    }
    source_manifest["tail_phase_v1_source_coverage"] = coverage
    regression_fixture = SOURCE_PRIMARY / "assessment_predictions.parquet"
    regression_sessions = pd.read_parquet(
        regression_fixture,
        columns=["session"],
    )["session"].astype(str)
    if bool(regression_sessions.ge(PROTECTED_START).any()):
        raise TailPhaseRunBlocked("M1C regression fixture crossed into protected data")
    source_manifest["sources"] = [
        *cast(list[dict[str, Any]], source_manifest["sources"]),
        {
            "role": "frozen_m1c_probability_regression_fixture",
            "path": str(regression_fixture),
            "sha256": sha256_file(regression_fixture),
            "rows_read": int(len(regression_sessions)),
            "maximum_session": str(regression_sessions.max()),
        },
    ]
    return historical, stress, bars, source_manifest


def load_frozen_runtimes() -> tuple[
    FrozenM1CRuntime,
    FrozenDirectionFeatureBuilder,
    FrozenDirectionRuntime,
]:
    m1c = FrozenM1CRuntime.from_artifacts(
        feature_manifest_path=SOURCE_PRIMARY / "causal_movement_feature_manifest.json",
        threshold_path=SOURCE_PRIMARY / "causal_movement_threshold.json",
    )
    direction_builder = FrozenDirectionFeatureBuilder.from_beta_artifact(
        SOURCE_PRIMARY / "stock_market_beta_parameters.csv"
    )
    direction = FrozenDirectionRuntime.from_artifacts(
        model_configurations_path=SOURCE_PRIMARY / "model_configurations.json",
        normalisation_path=SOURCE_PRIMARY / "stock_local_normalisation_parameters.json",
        thresholds_path=SOURCE_PRIMARY / "frozen_archetype_thresholds.json",
    )
    return m1c, direction_builder, direction


def _partition_inputs(
    historical: pd.DataFrame,
    stress: pd.DataFrame,
) -> pd.DataFrame:
    historical_work = historical.copy()
    historical_dates = historical_work["session"].astype(str)
    historical_work["partition"] = np.where(
        historical_dates.le(DEVELOPMENT_END),
        "development",
        "assessment",
    )
    stress_work = stress.copy()
    stress_work["partition"] = "stress"
    combined = pd.concat([historical_work, stress_work], ignore_index=True)
    combined = combined.sort_values(
        ["stock", "session", "checkpoint"],
        kind="mergesort",
    ).reset_index(drop=True)
    if combined.duplicated(["stock", "session", "checkpoint"]).any():
        raise TailPhaseRunBlocked("checkpoint identity is not unique")
    sessions = combined["session"].astype(str)
    permitted = (
        sessions.between(DEVELOPMENT_START, DEVELOPMENT_END)
        | sessions.between(ASSESSMENT_START, ASSESSMENT_END)
        | sessions.between(STRESS_START, STRESS_END)
    )
    if not bool(permitted.all()):
        raise TailPhaseRunBlocked("checkpoint chronology contains an unregistered gap or boundary")
    return combined


def _validate_m1c_regression(scored: pd.DataFrame) -> dict[str, Any]:
    archived = pd.read_parquet(
        SOURCE_PRIMARY / "assessment_predictions.parquet",
        columns=["stock", "session", "checkpoint", "M1C_probability"],
    )
    comparison = archived.merge(
        scored[["stock", "session", "checkpoint", "M1C_probability"]],
        on=["stock", "session", "checkpoint"],
        how="inner",
        suffixes=("_archived", "_v1"),
        validate="one_to_one",
    )
    if comparison.empty:
        raise TailPhaseRunBlocked("no frozen M1C regression rows were available")
    differences = np.abs(
        comparison["M1C_probability_archived"].to_numpy(float)
        - comparison["M1C_probability_v1"].to_numpy(float)
    )
    maximum = float(np.max(differences))
    if maximum > 1e-12:
        raise TailPhaseRunBlocked("frozen M1C probability regression failed")
    return {
        "rows_compared": int(len(comparison)),
        "maximum_absolute_probability_difference": maximum,
        "tolerance": 1e-12,
        "passed": True,
    }


def _add_directional_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    signed = pd.to_numeric(output["future_10m_signed_return_v1"], errors="coerce")
    momentum = pd.to_numeric(
        output["pre_entry_stock_signed_return_10m_v1"],
        errors="coerce",
    )
    valid = signed.notna() & momentum.notna() & signed.ne(0.0) & momentum.ne(0.0)
    output["direction_up_10m_v1"] = signed.gt(0.0).where(signed.notna())
    output["recent_momentum_continuation_v1"] = (np.sign(signed) == np.sign(momentum)).where(valid)
    output["recent_momentum_reversal_v1"] = (np.sign(signed) != np.sign(momentum)).where(valid)
    action = output["A1_action_v1"].astype("string")
    is_call = action.eq("CALL").fillna(False)
    is_put = action.eq("PUT").fillna(False)
    acted = action.isin(["CALL", "PUT"]) & signed.notna() & signed.ne(0.0)
    predicted_up = is_call
    output["A1_acted_v1"] = acted
    output["A1_correct_v1"] = (predicted_up == signed.gt(0.0)).where(acted)
    output["A1_aligned_10m_return_v1"] = np.where(
        is_call,
        signed,
        np.where(is_put, -signed, np.nan),
    )
    return output


def build_research_frames(
    historical: pd.DataFrame,
    stress: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    m1c_runtime: FrozenM1CRuntime,
    direction_builder: FrozenDirectionFeatureBuilder,
    direction_runtime: FrozenDirectionRuntime,
) -> tuple[pd.DataFrame, pd.DataFrame, FrozenConsumedMedianV1, dict[str, Any]]:
    """Build all V1 predictors first, then attach outcomes without changing definitions."""

    raw = _partition_inputs(historical, stress)
    scored = score_frozen_m1c_checkpoint_rows_v1(raw, runtime=m1c_runtime)
    regression = _validate_m1c_regression(scored)
    phase = build_tail_phase_checkpoint_rows_v1(scored, bars)
    frozen_median = freeze_movement_consumed_median_v1(phase)
    phase = apply_frozen_consumed_bucket_v1(
        phase,
        frozen_median=frozen_median.value,
    )
    phase["checkpoint_group_v1"] = phase["checkpoint"].map(
        lambda value: checkpoint_group(int(value))
    )
    phase["month_v1"] = phase["session"].astype(str).str[:7]
    phase["time_of_day_v1"] = pd.to_datetime(
        phase["signal_timestamp"],
        utc=True,
        errors="raise",
    ).dt.strftime("%H:%M:%S%z")

    high_tail = phase.loc[phase["m1c_high_tail_v1"].astype(bool)].copy()
    high_tail = attach_canonical_tail_outcomes_v1(high_tail, bars)
    high_tail = attach_frozen_a1_and_regime_v1(
        high_tail,
        bars,
        feature_builder=direction_builder,
        direction_runtime=direction_runtime,
    )
    high_tail = _add_directional_diagnostics(high_tail)

    identity = ["stock", "session", "checkpoint"]
    original_columns = set(phase.columns)
    attached_columns = [
        name for name in high_tail.columns if name not in original_columns or name in identity
    ]
    checkpoint = phase.merge(
        high_tail[attached_columns],
        on=identity,
        how="left",
        validate="one_to_one",
    )
    episodes = construct_fresh_tail_episodes_v1(checkpoint)
    episodes["phase_at_trigger_v1"] = episodes["m1c_tail_phase_v1"]
    episodes["tail_run_age_at_trigger_v1"] = episodes["tail_run_age_minutes_v1"]
    episodes["movement_consumed_at_trigger_v1"] = episodes["movement_consumed_v1"]
    episodes["phase_updated_at_direction_decision_v1"] = False
    episodes["m1c_probability_updated_at_direction_decision_v1"] = False

    if not bool(
        checkpoint["m1c_high_tail_v1"]
        .astype(bool)
        .equals(checkpoint["M1C_probability"].ge(M1C_FROZEN_THRESHOLD))
    ):
        raise TailPhaseRunBlocked("tail membership drifted from the exact frozen threshold")
    return checkpoint, episodes, frozen_median, regression


def _analysis_frames(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
) -> list[tuple[str, str, pd.DataFrame]]:
    frames: list[tuple[str, str, pd.DataFrame]] = []
    checkpoint_tail = checkpoint.loc[_bool_series(checkpoint["m1c_high_tail_v1"])].copy()
    for period in ("assessment", "stress"):
        frames.append(
            (
                period,
                "high_tail_checkpoint",
                checkpoint_tail.loc[checkpoint_tail["partition"].eq(period)].copy(),
            )
        )
        frames.append(
            (
                period,
                "fresh_high_tail_episode",
                episodes.loc[episodes["partition"].eq(period)].copy(),
            )
        )
    return frames


def _support_status(frame: pd.DataFrame) -> str:
    if len(frame) < MINIMUM_CELL_ROWS:
        return "blocked_insufficient_support"
    if frame["session"].nunique() < MINIMUM_CELL_SESSIONS:
        return "blocked_insufficient_support"
    return "descriptive_support_available"


def _bool_series(values: pd.Series) -> pd.Series:
    return values.astype("boolean").fillna(False)


def _stable_seed(label: str) -> int:
    suffix = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return (BOOTSTRAP_SEED + suffix) % (2**32 - 1)


def _cluster_bootstrap_estimate(
    frame: pd.DataFrame,
    *,
    value_column: str,
    estimator: Callable[[np.ndarray[Any, np.dtype[np.float64]]], float],
    label: str,
) -> dict[str, Any]:
    work = frame.loc[:, ["session", value_column]].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.loc[np.isfinite(work[value_column].to_numpy(float))]
    sessions = tuple(sorted(work["session"].astype(str).unique()))
    result: dict[str, Any] = {
        "n": int(len(work)),
        "sessions": int(len(sessions)),
        "estimate": (
            float(estimator(work[value_column].to_numpy(float))) if len(work) else math.nan
        ),
        "lower_95": math.nan,
        "upper_95": math.nan,
        "status": _support_status(work),
    }
    if result["status"] != "descriptive_support_available":
        return result
    values_by_session = {
        session: work.loc[
            work["session"].astype(str).eq(session),
            value_column,
        ].to_numpy(float)
        for session in sessions
    }
    rng = np.random.default_rng(_stable_seed(label))
    estimates = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        values = np.concatenate([values_by_session[str(session)] for session in sampled])
        estimates[draw] = estimator(values)
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    result["lower_95"] = float(np.quantile(estimates, alpha))
    result["upper_95"] = float(np.quantile(estimates, 1.0 - alpha))
    return result


def _subgroups(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    output: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", frame)]
    for phase in PHASE_ORDER:
        output.append(
            (
                "phase",
                phase,
                frame.loc[frame["m1c_tail_phase_v1"].eq(phase)],
            )
        )
    for bucket in BUCKET_ORDER:
        output.append(
            (
                "movement_consumed_bucket",
                bucket,
                frame.loc[frame["movement_consumed_bucket_v1"].eq(bucket)],
            )
        )
    for phase in PHASE_ORDER:
        for bucket in BUCKET_ORDER:
            output.append(
                (
                    "phase_x_movement_consumed_bucket",
                    f"{phase}|{bucket}",
                    frame.loc[
                        frame["m1c_tail_phase_v1"].eq(phase)
                        & frame["movement_consumed_bucket_v1"].eq(bucket)
                    ],
                )
            )
    return output


def structural_counts(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, level, frame in _analysis_frames(checkpoint, episodes):
        work = frame.copy()
        work["phase_x_bucket_v1"] = (
            work["m1c_tail_phase_v1"].astype(str)
            + "|"
            + work["movement_consumed_bucket_v1"].astype(str)
        )
        work["month_v1"] = work["session"].astype(str).str[:7]
        dimensions = {
            "m1c_tail_phase_v1": "m1c_tail_phase_v1",
            "movement_consumed_bucket_v1": "movement_consumed_bucket_v1",
            "phase_x_movement_consumed_bucket_v1": "phase_x_bucket_v1",
            "stock": "stock",
            "month": "month_v1",
            "session": "session",
            "checkpoint": "checkpoint",
            "time_of_day": "time_of_day_v1",
        }
        for dimension, column in dimensions.items():
            counts = work.groupby(column, dropna=False, sort=True).size()
            for value, count in counts.items():
                rows.append(
                    {
                        "period": period,
                        "analysis_level": level,
                        "dimension": dimension,
                        "value": str(value),
                        "count": int(count),
                        "share": float(count / len(work)) if len(work) else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def missingness_table(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    source_manifest: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions: tuple[tuple[str, str | None, str | None], ...] = (
        (
            "phase_history",
            "phase_history_complete_v1",
            "phase_missing_reason_v1",
        ),
        (
            "movement_consumed",
            "movement_consumed_complete_v1",
            "movement_consumed_missing_reason_v1",
        ),
        (
            "post_share_of_local_range",
            "post_share_of_local_range_complete_v1",
            "post_share_of_local_range_missing_reason_v1",
        ),
        ("frozen_A1", "A1_complete_v1", "A1_missing_reason_v1"),
    )
    for period, level, frame in _analysis_frames(checkpoint, episodes):
        for name, complete_column, reason_column in definitions:
            if complete_column not in frame:
                continue
            complete = _bool_series(frame[complete_column])
            rows.append(
                {
                    "period": period,
                    "analysis_level": level,
                    "field": name,
                    "reason": "complete",
                    "count": int(complete.sum()),
                    "rate": float(complete.mean()) if len(frame) else math.nan,
                }
            )
            missing = frame.loc[~complete]
            if missing.empty:
                continue
            reasons = (
                missing[reason_column].fillna("not_materialised_or_unknown").astype(str)
                if reason_column is not None and reason_column in missing
                else pd.Series(["unknown"] * len(missing), index=missing.index)
            )
            for reason, count in reasons.value_counts(dropna=False, sort=False).items():
                rows.append(
                    {
                        "period": period,
                        "analysis_level": level,
                        "field": name,
                        "reason": str(reason),
                        "count": int(count),
                        "rate": float(count / len(frame)) if len(frame) else math.nan,
                    }
                )
    coverage = cast(
        Mapping[str, Mapping[str, Any]],
        source_manifest.get("tail_phase_v1_source_coverage", {}),
    )
    for period in ("development", "assessment", "stress"):
        item = coverage.get(period)
        if item is None:
            continue
        before = int(item["canonical_causal_checkpoint_rows"])
        coverage_complete_count = int(item["rows_with_previous_close_option_context"])
        coverage_missing_count = int(item["rows_without_previous_close_option_context"])
        rows.extend(
            [
                {
                    "period": period,
                    "analysis_level": "source_checkpoint_coverage",
                    "field": "previous_close_option_context",
                    "reason": "complete",
                    "count": coverage_complete_count,
                    "rate": (float(coverage_complete_count / before) if before else math.nan),
                },
                {
                    "period": period,
                    "analysis_level": "source_checkpoint_coverage",
                    "field": "previous_close_option_context",
                    "reason": str(item["missing_row_state"]),
                    "count": coverage_missing_count,
                    "rate": (float(coverage_missing_count / before) if before else math.nan),
                },
            ]
        )
    return pd.DataFrame(rows)


def movement_summary_table(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = {
        "future_10m_absolute_underlying_return": "future_10m_absolute_return_v1",
        "future_10m_signed_underlying_return": "future_10m_signed_return_v1",
        "future_15m_absolute_movement": "future_15m_absolute_movement_v1",
        "future_15m_iv_residual": "future_15m_iv_residual_v1",
        "future_15m_exceed_iv_rate": "future_15m_exceed_iv_v1",
        "post_share_of_local_range_v1": "post_share_of_local_range_v1",
        "maximum_up_excursion_10m": "maximum_up_excursion_10m",
        "maximum_down_excursion_10m": "maximum_down_excursion_10m",
    }
    rows: list[dict[str, Any]] = []
    for period, level, frame in _analysis_frames(checkpoint, episodes):
        for subgroup_type, subgroup, group in _subgroups(frame):
            for metric, column in metric_columns.items():
                if column not in group:
                    continue
                values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "period": period,
                        "analysis_level": level,
                        "subgroup_type": subgroup_type,
                        "subgroup": subgroup,
                        "metric": metric,
                        "n": int(len(values)),
                        "sessions": int(
                            group.loc[
                                pd.to_numeric(group[column], errors="coerce").notna(),
                                "session",
                            ].nunique()
                        ),
                        "mean": float(np.mean(values)) if len(values) else math.nan,
                        "median": float(np.median(values)) if len(values) else math.nan,
                        "q05": float(np.quantile(values, 0.05)) if len(values) else math.nan,
                        "q10": float(np.quantile(values, 0.10)) if len(values) else math.nan,
                        "q90": float(np.quantile(values, 0.90)) if len(values) else math.nan,
                        "q95": float(np.quantile(values, 0.95)) if len(values) else math.nan,
                        "status": _support_status(
                            group.loc[pd.to_numeric(group[column], errors="coerce").notna()]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_estimates(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    estimators: dict[
        str,
        Callable[[np.ndarray[Any, np.dtype[np.float64]]], float],
    ] = {
        "mean": lambda values: float(np.mean(values)),
        "median": lambda values: float(np.median(values)),
    }
    for period, level, frame in _analysis_frames(checkpoint, episodes):
        for subgroup_type, subgroup, group in _subgroups(frame):
            for quantity, column in PRIMARY_QUANTITIES.items():
                for estimate_name, estimator in estimators.items():
                    label = "|".join(
                        (
                            period,
                            level,
                            subgroup_type,
                            subgroup,
                            quantity,
                            estimate_name,
                        )
                    )
                    result = _cluster_bootstrap_estimate(
                        group,
                        value_column=column,
                        estimator=estimator,
                        label=label,
                    )
                    rows.append(
                        {
                            "period": period,
                            "analysis_level": level,
                            "subgroup_type": subgroup_type,
                            "subgroup": subgroup,
                            "quantity": quantity,
                            "estimate_type": estimate_name,
                            **result,
                        }
                    )
    return pd.DataFrame(rows)


def _comparison_bootstrap(
    frame: pd.DataFrame,
    *,
    value_column: str,
    selector_a: pd.Series,
    selector_b: pd.Series,
    estimator: Callable[[np.ndarray[Any, np.dtype[np.float64]]], float],
    label: str,
) -> dict[str, Any]:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    finite = np.isfinite(values.to_numpy(float))
    group_a = frame.loc[selector_a & finite, ["session"]].copy()
    group_a["value"] = values.loc[group_a.index].to_numpy(float)
    group_b = frame.loc[selector_b & finite, ["session"]].copy()
    group_b["value"] = values.loc[group_b.index].to_numpy(float)
    status = (
        "descriptive_support_available"
        if _support_status(group_a) == "descriptive_support_available"
        and _support_status(group_b) == "descriptive_support_available"
        else "blocked_insufficient_support"
    )
    estimate = (
        float(estimator(group_a["value"].to_numpy(float)))
        - float(estimator(group_b["value"].to_numpy(float)))
        if len(group_a) and len(group_b)
        else math.nan
    )
    result: dict[str, Any] = {
        "n_a": int(len(group_a)),
        "n_b": int(len(group_b)),
        "sessions_a": int(group_a["session"].nunique()),
        "sessions_b": int(group_b["session"].nunique()),
        "estimate_a_minus_b": estimate,
        "lower_95": math.nan,
        "upper_95": math.nan,
        "status": status,
    }
    if status != "descriptive_support_available":
        return result
    sessions = tuple(
        sorted(set(group_a["session"].astype(str)).union(group_b["session"].astype(str)))
    )
    a_values = {
        session: group_a.loc[
            group_a["session"].astype(str).eq(session),
            "value",
        ].to_numpy(float)
        for session in sessions
    }
    b_values = {
        session: group_b.loc[
            group_b["session"].astype(str).eq(session),
            "value",
        ].to_numpy(float)
        for session in sessions
    }
    rng = np.random.default_rng(_stable_seed(label))
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        sampled_a = [a_values[str(session)] for session in sampled if len(a_values[str(session)])]
        sampled_b = [b_values[str(session)] for session in sampled if len(b_values[str(session)])]
        if not sampled_a or not sampled_b:
            continue
        estimates.append(
            float(estimator(np.concatenate(sampled_a)))
            - float(estimator(np.concatenate(sampled_b)))
        )
    if len(estimates) != BOOTSTRAP_DRAWS:
        result["status"] = "blocked_insufficient_support"
        return result
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    result["lower_95"] = float(np.quantile(estimates, alpha))
    result["upper_95"] = float(np.quantile(estimates, 1.0 - alpha))
    return result


def key_comparisons(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    estimators: dict[
        str,
        Callable[[np.ndarray[Any, np.dtype[np.float64]]], float],
    ] = {
        "mean": lambda values: float(np.mean(values)),
        "median": lambda values: float(np.median(values)),
    }
    for period, level, frame in _analysis_frames(checkpoint, episodes):
        comparisons: list[tuple[str, str, str, pd.Series, pd.Series]] = []
        for phase_a, phase_b in (
            ("FIRST_ENTRY", "PERSISTENT"),
            ("FIRST_ENTRY", "RE_ENTRY"),
            ("PERSISTENT", "RE_ENTRY"),
        ):
            comparisons.append(
                (
                    f"{phase_a}_minus_{phase_b}",
                    phase_a,
                    phase_b,
                    frame["m1c_tail_phase_v1"].eq(phase_a),
                    frame["m1c_tail_phase_v1"].eq(phase_b),
                )
            )
        comparisons.append(
            (
                "LOW_OR_EQUAL_minus_HIGH",
                "LOW_OR_EQUAL",
                "HIGH",
                frame["movement_consumed_bucket_v1"].eq("LOW_OR_EQUAL"),
                frame["movement_consumed_bucket_v1"].eq("HIGH"),
            )
        )
        for comparison, label_a, label_b, selector_a, selector_b in comparisons:
            for quantity, column in PRIMARY_QUANTITIES.items():
                for estimate_name, estimator in estimators.items():
                    result = _comparison_bootstrap(
                        frame,
                        value_column=column,
                        selector_a=selector_a,
                        selector_b=selector_b,
                        estimator=estimator,
                        label=f"{period}|{level}|{comparison}|{quantity}|{estimate_name}",
                    )
                    rows.append(
                        {
                            "period": period,
                            "analysis_level": level,
                            "comparison": comparison,
                            "group_a": label_a,
                            "group_b": label_b,
                            "quantity": quantity,
                            "estimate_type": estimate_name,
                            **result,
                        }
                    )
    return pd.DataFrame(rows)


def leave_one_out_diagnostics(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, level, frame in _analysis_frames(checkpoint, episodes):
        work = frame.copy()
        work["month_v1"] = work["session"].astype(str).str[:7]
        subgroup_specs = [
            *[
                (
                    "phase",
                    phase,
                    work["m1c_tail_phase_v1"].eq(phase),
                )
                for phase in PHASE_ORDER
            ],
            *[
                (
                    "movement_consumed_bucket",
                    bucket,
                    work["movement_consumed_bucket_v1"].eq(bucket),
                )
                for bucket in BUCKET_ORDER
            ],
        ]
        for exclusion_dimension, exclusion_column in (
            ("month", "month_v1"),
            ("stock", "stock"),
        ):
            for excluded in sorted(work[exclusion_column].dropna().astype(str).unique()):
                retained = work.loc[~work[exclusion_column].astype(str).eq(excluded)]
                for subgroup_type, subgroup, selector in subgroup_specs:
                    group = retained.loc[selector.reindex(retained.index, fill_value=False)]
                    for quantity, column in PRIMARY_QUANTITIES.items():
                        values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
                        values = values[np.isfinite(values)]
                        rows.append(
                            {
                                "period": period,
                                "analysis_level": level,
                                "exclusion_dimension": exclusion_dimension,
                                "excluded": excluded,
                                "subgroup_type": subgroup_type,
                                "subgroup": subgroup,
                                "quantity": quantity,
                                "n": int(len(values)),
                                "sessions": int(
                                    group.loc[
                                        pd.to_numeric(
                                            group[column],
                                            errors="coerce",
                                        ).notna(),
                                        "session",
                                    ].nunique()
                                ),
                                "mean": (float(np.mean(values)) if len(values) else math.nan),
                                "median": (float(np.median(values)) if len(values) else math.nan),
                                "status": _support_status(
                                    group.loc[
                                        pd.to_numeric(
                                            group[column],
                                            errors="coerce",
                                        ).notna()
                                    ]
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def concentration_table(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, level, frame in _analysis_frames(checkpoint, episodes):
        work = frame.copy()
        work["month_v1"] = work["session"].astype(str).str[:7]
        for grouping_name, grouping_column in (
            ("month", "month_v1"),
            ("stock", "stock"),
            ("checkpoint", "checkpoint"),
        ):
            for phase in PHASE_ORDER:
                phase_rows = work.loc[work["m1c_tail_phase_v1"].eq(phase)]
                counts = phase_rows.groupby(grouping_column, sort=True).size()
                for value, count in counts.items():
                    rows.append(
                        {
                            "period": period,
                            "analysis_level": level,
                            "subgroup_type": "phase",
                            "subgroup": phase,
                            "concentration_dimension": grouping_name,
                            "value": str(value),
                            "count": int(count),
                            "share_within_subgroup": (
                                float(count / len(phase_rows)) if len(phase_rows) else math.nan
                            ),
                        }
                    )
            for bucket in BUCKET_ORDER:
                bucket_rows = work.loc[work["movement_consumed_bucket_v1"].eq(bucket)]
                counts = bucket_rows.groupby(grouping_column, sort=True).size()
                for value, count in counts.items():
                    rows.append(
                        {
                            "period": period,
                            "analysis_level": level,
                            "subgroup_type": "movement_consumed_bucket",
                            "subgroup": bucket,
                            "concentration_dimension": grouping_name,
                            "value": str(value),
                            "count": int(count),
                            "share_within_subgroup": (
                                float(count / len(bucket_rows)) if len(bucket_rows) else math.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def directional_diagnostics(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, level, frame in _analysis_frames(checkpoint, episodes):
        for subgroup_type, subgroup, group in _subgroups(frame):
            interaction = subgroup_type == "phase_x_movement_consumed_bucket"
            interaction_supported = _support_status(group) == "descriptive_support_available"
            direction = group.loc[
                pd.to_numeric(
                    group["future_10m_signed_return_v1"],
                    errors="coerce",
                ).notna()
                & pd.to_numeric(
                    group["future_10m_signed_return_v1"],
                    errors="coerce",
                ).ne(0.0)
            ]
            momentum = group.loc[group["recent_momentum_continuation_v1"].notna()]
            acted = group.loc[_bool_series(group["A1_acted_v1"])]
            call = acted.loc[acted["A1_action_v1"].eq("CALL")]
            put = acted.loc[acted["A1_action_v1"].eq("PUT")]
            action_counts = (
                group["A1_action_v1"].fillna("UNKNOWN_INCOMPLETE").astype(str).value_counts()
            )
            report_metrics = not interaction or interaction_supported
            a1_supported = bool(
                len(acted) >= MINIMUM_A1_ACTION_ROWS
                and acted["session"].nunique() >= MINIMUM_CELL_SESSIONS
            )
            report_a1_metrics = report_metrics and (not interaction or a1_supported)
            rows.append(
                {
                    "period": period,
                    "analysis_level": level,
                    "subgroup_type": subgroup_type,
                    "subgroup": subgroup,
                    "rows": int(len(group)),
                    "sessions": int(group["session"].nunique()),
                    "up_count": int(_bool_series(direction["direction_up_10m_v1"]).sum()),
                    "down_count": int((~_bool_series(direction["direction_up_10m_v1"])).sum()),
                    "up_rate": (
                        float(_bool_series(direction["direction_up_10m_v1"]).mean())
                        if len(direction) and report_metrics
                        else math.nan
                    ),
                    "recent_momentum_valid": int(len(momentum)),
                    "recent_momentum_continuation_rate": (
                        float(momentum["recent_momentum_continuation_v1"].astype(bool).mean())
                        if len(momentum) and report_metrics
                        else math.nan
                    ),
                    "recent_momentum_reversal_rate": (
                        float(momentum["recent_momentum_reversal_v1"].astype(bool).mean())
                        if len(momentum) and report_metrics
                        else math.nan
                    ),
                    "A1_CALL_count": int(action_counts.get("CALL", 0)),
                    "A1_PUT_count": int(action_counts.get("PUT", 0)),
                    "A1_ABSTAIN_count": int(action_counts.get("ABSTAIN", 0)),
                    "A1_UNKNOWN_count": int(action_counts.get("UNKNOWN_INCOMPLETE", 0)),
                    "A1_acted_count": int(len(acted)),
                    "A1_accuracy": (
                        float(acted["A1_correct_v1"].astype(bool).mean())
                        if len(acted) and report_a1_metrics
                        else math.nan
                    ),
                    "A1_mean_aligned_10m_return": (
                        float(acted["A1_aligned_10m_return_v1"].mean())
                        if len(acted) and report_a1_metrics
                        else math.nan
                    ),
                    "A1_median_aligned_10m_return": (
                        float(acted["A1_aligned_10m_return_v1"].median())
                        if len(acted) and report_a1_metrics
                        else math.nan
                    ),
                    "A1_CALL_accuracy": (
                        float(call["A1_correct_v1"].astype(bool).mean())
                        if len(call) and report_a1_metrics
                        else math.nan
                    ),
                    "A1_CALL_mean_aligned_10m_return": (
                        float(call["A1_aligned_10m_return_v1"].mean())
                        if len(call) and report_a1_metrics
                        else math.nan
                    ),
                    "A1_CALL_median_aligned_10m_return": (
                        float(call["A1_aligned_10m_return_v1"].median())
                        if len(call) and report_a1_metrics
                        else math.nan
                    ),
                    "A1_PUT_accuracy": (
                        float(put["A1_correct_v1"].astype(bool).mean())
                        if len(put) and report_a1_metrics
                        else math.nan
                    ),
                    "A1_PUT_mean_aligned_10m_return": (
                        float(put["A1_aligned_10m_return_v1"].mean())
                        if len(put) and report_a1_metrics
                        else math.nan
                    ),
                    "A1_PUT_median_aligned_10m_return": (
                        float(put["A1_aligned_10m_return_v1"].median())
                        if len(put) and report_a1_metrics
                        else math.nan
                    ),
                    "status": (
                        "descriptive_support_available"
                        if a1_supported and report_metrics
                        else "blocked_insufficient_support"
                    ),
                    "interpretation": (
                        "frozen_A1_descriptive_comparison_only"
                        if report_a1_metrics
                        else "blocked_insufficient_support"
                    ),
                }
            )
    return pd.DataFrame(rows)


def regime_diagnostics(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise only the existing fixed 10-minute pre-entry market context."""

    rows: list[dict[str, Any]] = []
    for period, level, frame in _analysis_frames(checkpoint, episodes):
        for phase in PHASE_ORDER:
            phase_rows = frame.loc[frame["m1c_tail_phase_v1"].eq(phase)]
            for alignment in ("ALIGNED", "OPPOSED", "FLAT_OR_UNKNOWN"):
                group = phase_rows.loc[phase_rows["stock_market_alignment_v1"].eq(alignment)]
                market = pd.to_numeric(
                    group["pre_entry_broad_market_signed_return_10m_v1"],
                    errors="coerce",
                )
                stock = pd.to_numeric(
                    group["pre_entry_stock_signed_return_10m_v1"],
                    errors="coerce",
                )
                rows.append(
                    {
                        "period": period,
                        "analysis_level": level,
                        "phase": phase,
                        "stock_market_alignment": alignment,
                        "n": int(len(group)),
                        "sessions": int(group["session"].nunique()),
                        "mean_pre_entry_market_return_10m": (
                            float(market.mean()) if market.notna().any() else math.nan
                        ),
                        "median_pre_entry_market_return_10m": (
                            float(market.median()) if market.notna().any() else math.nan
                        ),
                        "mean_pre_entry_stock_return_10m": (
                            float(stock.mean()) if stock.notna().any() else math.nan
                        ),
                        "median_pre_entry_stock_return_10m": (
                            float(stock.median()) if stock.notna().any() else math.nan
                        ),
                        "fixed_pre_entry_horizon_minutes": 10,
                        "sector_context_status": "out_of_scope_not_available",
                        "market_volatility_state_status": ("out_of_scope_not_available"),
                        "status": _support_status(group),
                    }
                )
    return pd.DataFrame(rows)


def frozen_config_payload(frozen: FrozenConsumedMedianV1) -> dict[str, Any]:
    return {
        "schema_version": M1C_TAIL_PHASE_V1_VERSION,
        "research_id": "M1C Tail Phase V1",
        "m1c_threshold": M1C_FROZEN_THRESHOLD,
        "frozen_checkpoints": list(FROZEN_CHECKPOINTS),
        "underlying_bar_minutes": 5,
        "movement_consumed_lookback_minutes": 15,
        "movement_consumed_median_2024": frozen.value,
        "movement_consumed_median_provenance": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "complete_observations": frozen.complete_observations,
            "predictor_values_only": frozen.predictor_values_only,
        },
        "chronology": {
            "development": {
                "start": DEVELOPMENT_START,
                "end": DEVELOPMENT_END,
            },
            "assessment": {
                "start": ASSESSMENT_START,
                "end": ASSESSMENT_END,
            },
            "stress": {"start": STRESS_START, "end": STRESS_END},
            "protected": {"start": PROTECTED_START, "end": None},
        },
        "model_identifiers": {
            "movement": "M1C/frozen-m1c-v0",
            "direction": "A1/frozen-comparison-unchanged",
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "confidence_level": CONFIDENCE_LEVEL,
        },
    }


def canonical_implementation_payload() -> dict[str, Any]:
    return {
        "authoritative_implementation": (
            "committed 20260726 frozen archetype artifacts plus later no-fit "
            "prospective runtimes at repository HEAD"
        ),
        "authority_evidence": [
            "committed machine-readable M1C and A1 artifacts",
            "known-row regression tests for frozen M1C and A1",
            "later recorder commits bind to the no-fit runtime schemas",
            "hash-verified local source loader is newer than the stale alternate checkout",
        ],
        "code_paths": {
            "frozen_m1c_probabilities": (
                "packages/stocker_prospective/src/stocker_prospective/frozen_m1c.py"
            ),
            "top_5_percent_threshold": (
                "packages/stocker_prospective/src/stocker_prospective/contract.py"
            ),
            "session_and_checkpoint_sequence": (
                "packages/stocker_prospective/src/stocker_prospective/m1c_features.py"
            ),
            "fresh_episode_runtime": (
                "packages/stocker_prospective/src/stocker_prospective/frozen_m1c.py"
                "#FreshEpisodeTracker"
            ),
            "fresh_episode_retrospective": (
                "packages/stocker_research/src/stocker_research/"
                "stock_local_directional_archetypes_v0.py#construct_fresh_episodes"
            ),
            "previous_close_option_implied_movement": (
                "packages/stocker_research/src/stocker_research/"
                "m1c_low_movement_v0.py#iv_expected_absolute"
            ),
            "ten_and_fifteen_minute_outcomes": (
                "packages/stocker_research/src/stocker_research/"
                "m1c_low_movement_v0.py#calculate_checkpoint_outcomes"
            ),
            "frozen_a1": ("packages/stocker_prospective/src/stocker_prospective/direction.py"),
            "ibkr_prospective_records": (
                "packages/stocker_prospective/src/stocker_prospective/recorder_repository.py"
            ),
            "chronology_guards": [
                "packages/stocker_prospective/src/stocker_prospective/tail_phase_v1.py",
                str(SOURCE_RUNNER.relative_to(REPO_ROOT)),
            ],
            "report_and_artifact_conventions": (
                "research/directional-readiness/<date>-<study>/artifacts/primary"
            ),
        },
        "duplicated_m1c_logic": False,
        "m1c_refit": False,
        "a1_refit": False,
    }


CHECKPOINT_ARTIFACT_COLUMNS: Final[tuple[str, ...]] = (
    "row_id",
    "stock",
    "session",
    "partition",
    "checkpoint",
    "checkpoint_group_v1",
    "time_of_day_v1",
    "feature_available_timestamp_utc",
    "signal_timestamp",
    "prospective_entry_timestamp",
    "prospective_entry_timestamp_complete_v1",
    "M1C_probability",
    "m1c_high_tail_threshold_v1",
    "m1c_high_tail_v1",
    "m1c_model_version_v1",
    "m1c_model_hash_v1",
    "m1c_feature_hash_v1",
    "m1c_missing_feature_count_v1",
    "tail_phase_schema_version_v1",
    "m1c_tail_phase_v1",
    "tail_entry_number_v1",
    "tail_run_length_checkpoints_v1",
    "tail_run_age_minutes_v1",
    "prior_tail_entries_v1",
    "previous_checkpoint_above_tail_v1",
    "minutes_since_previous_tail_exit_v1",
    "phase_history_complete_v1",
    "phase_missing_reason_v1",
    "movement_consumed_v1",
    "movement_consumed_numerator_v1",
    "movement_consumed_denominator_v1",
    "movement_consumed_complete_v1",
    "movement_consumed_missing_reason_v1",
    "movement_consumed_bucket_v1",
    "movement_consumed_frozen_median_v1",
    "available_10m",
    "future_10m_absolute_return_v1",
    "future_10m_signed_return_v1",
    "available_15m",
    "future_15m_absolute_movement_v1",
    "future_15m_iv_residual_v1",
    "future_15m_exceed_iv_v1",
    "maximum_up_excursion_10m",
    "maximum_down_excursion_10m",
    "maximum_absolute_excursion_10m",
    "realised_path_range_10m",
    "maximum_up_excursion_15m",
    "maximum_down_excursion_15m",
    "maximum_absolute_excursion_15m",
    "realised_path_range_15m",
    "post_share_of_local_range_v1",
    "post_share_of_local_range_complete_v1",
    "post_share_of_local_range_missing_reason_v1",
    "direction_up_10m_v1",
    "recent_momentum_continuation_v1",
    "recent_momentum_reversal_v1",
    "A1_complete_v1",
    "A1_missing_reason_v1",
    "A1_probability_up_v1",
    "A1_confidence_v1",
    "A1_action_v1",
    "A1_boundary_v1",
    "A1_feature_hash_v1",
    "A1_model_hash_v1",
    "A1_preprocessing_hash_v1",
    "A1_maximum_feature_timestamp_v1",
    "A1_acted_v1",
    "A1_correct_v1",
    "A1_aligned_10m_return_v1",
    "pre_entry_stock_signed_return_10m_v1",
    "pre_entry_broad_market_signed_return_10m_v1",
    "stock_market_alignment_v1",
    "pre_entry_sector_signed_return_v1",
    "stock_sector_alignment_v1",
    "sector_context_status_v1",
    "market_volatility_state_status_v1",
    "source_surface_id_v1",
    "source_provenance_v1",
    "protected_outcomes_accessed_v1",
)

EPISODE_EXTRA_COLUMNS: Final[tuple[str, ...]] = (
    "episode_id",
    "existing_fresh_episode_identifier",
    "episode_number",
    "minutes_since_previous_episode",
    "above_frozen_threshold",
    "previous_checkpoint_probability",
    "fresh_crossing",
    "trigger_bar_ordinal",
    "marker_bar_ordinal",
    "direction_marker_bar",
    "trigger_bar_excluded_from_direction_features",
    "phase_at_trigger_v1",
    "tail_run_age_at_trigger_v1",
    "movement_consumed_at_trigger_v1",
    "phase_updated_at_direction_decision_v1",
    "m1c_probability_updated_at_direction_decision_v1",
)


def artifact_views(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    source_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_payload = json.dumps(
        {
            "loader": str(SOURCE_RUNNER.relative_to(REPO_ROOT)),
            "protected_rows_read": source_manifest.get("protected_rows_read"),
            "archived_signed_pressure_values_read": source_manifest.get(
                "archived_signed_pressure_values_read"
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    for frame in (checkpoint, episodes):
        frame["source_surface_id_v1"] = (
            "20260726-stock-local-directional-archetypes-v0/hash-verified-load_inputs"
        )
        frame["source_provenance_v1"] = source_payload
        frame["protected_outcomes_accessed_v1"] = False
    missing = sorted(set(CHECKPOINT_ARTIFACT_COLUMNS).difference(checkpoint.columns))
    if missing:
        raise TailPhaseRunBlocked(f"checkpoint artifact schema is incomplete: {missing}")
    checkpoint_output = checkpoint.loc[:, list(CHECKPOINT_ARTIFACT_COLUMNS)].copy()
    episode_columns = [
        *[name for name in CHECKPOINT_ARTIFACT_COLUMNS if name in episodes],
        *[name for name in EPISODE_EXTRA_COLUMNS if name in episodes],
    ]
    missing_episode = sorted(set(EPISODE_EXTRA_COLUMNS).difference(episode_columns))
    if missing_episode:
        raise TailPhaseRunBlocked(f"episode artifact schema is incomplete: {missing_episode}")
    episode_output = episodes.loc[:, episode_columns].copy()
    return checkpoint_output, episode_output


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], _json_safe(frame.to_dict(orient="records")))


def _composition_payload(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for period, level, frame in _analysis_frames(checkpoint, episodes):
        counts = frame["m1c_tail_phase_v1"].fillna("UNKNOWN_INCOMPLETE").value_counts()
        payload[f"{period}|{level}"] = {
            "total_high_tail_rows": int(len(frame)),
            "counts": {str(key): int(value) for key, value in counts.items()},
            "shares": {
                str(key): float(value / len(frame)) if len(frame) else math.nan
                for key, value in counts.items()
            },
            "sessions": int(frame["session"].nunique()),
            "stocks": int(frame["stock"].nunique()),
        }
    return payload


def _comparison_extract(
    comparisons: pd.DataFrame,
    *,
    period: str,
    level: str,
    comparison: str,
    quantity: str,
) -> dict[str, Any] | None:
    selected = comparisons.loc[
        comparisons["period"].eq(period)
        & comparisons["analysis_level"].eq(level)
        & comparisons["comparison"].eq(comparison)
        & comparisons["quantity"].eq(quantity)
        & comparisons["estimate_type"].eq("mean")
    ]
    return None if selected.empty else _records(selected.iloc[[0]])[0]


def _breadth_payload(
    leave_one_out: pd.DataFrame,
    concentration: pd.DataFrame,
) -> dict[str, Any]:
    concentration_summary: list[dict[str, Any]] = []
    for keys, group in concentration.groupby(
        [
            "period",
            "analysis_level",
            "subgroup_type",
            "subgroup",
            "concentration_dimension",
        ],
        sort=True,
    ):
        period, level, subgroup_type, subgroup, dimension = keys
        concentration_summary.append(
            {
                "period": str(period),
                "analysis_level": str(level),
                "subgroup_type": str(subgroup_type),
                "subgroup": str(subgroup),
                "dimension": str(dimension),
                "maximum_share": float(group["share_within_subgroup"].max()),
            }
        )
    leave_ranges: list[dict[str, Any]] = []
    for keys, group in leave_one_out.groupby(
        [
            "period",
            "analysis_level",
            "exclusion_dimension",
            "subgroup_type",
            "subgroup",
            "quantity",
        ],
        sort=True,
    ):
        finite = group.loc[np.isfinite(group["mean"].to_numpy(float)), "mean"]
        period, level, exclusion, subgroup_type, subgroup, quantity = keys
        leave_ranges.append(
            {
                "period": str(period),
                "analysis_level": str(level),
                "exclusion_dimension": str(exclusion),
                "subgroup_type": str(subgroup_type),
                "subgroup": str(subgroup),
                "quantity": str(quantity),
                "minimum_mean": float(finite.min()) if len(finite) else math.nan,
                "maximum_mean": float(finite.max()) if len(finite) else math.nan,
                "exclusions": int(len(group)),
            }
        )
    return {
        "maximum_concentration_shares": concentration_summary,
        "leave_one_out_estimate_ranges": leave_ranges,
    }


def _structural_conclusion(
    comparisons: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> str:
    candidates = comparisons.loc[
        comparisons["analysis_level"].eq("fresh_high_tail_episode")
        & comparisons["estimate_type"].eq("mean")
        & comparisons["comparison"].isin(
            [
                "FIRST_ENTRY_minus_PERSISTENT",
                "FIRST_ENTRY_minus_RE_ENTRY",
                "PERSISTENT_minus_RE_ENTRY",
            ]
        )
        & comparisons["status"].eq("descriptive_support_available")
    ].copy()
    stable_separation = False
    for keys, group in candidates.groupby(["comparison", "quantity"], sort=True):
        if set(group["period"]) != {"assessment", "stress"}:
            continue
        comparison, quantity = (str(keys[0]), str(keys[1]))
        estimates = group["estimate_a_minus_b"].to_numpy(float)
        lower = group["lower_95"].to_numpy(float)
        upper = group["upper_95"].to_numpy(float)
        excludes_zero = (lower > 0.0) | (upper < 0.0)
        same_sign = bool(np.all(np.sign(estimates) == np.sign(estimates[0])))
        if not bool(excludes_zero.all()) or not same_sign:
            continue
        group_a, group_b = comparison.split("_minus_", maxsplit=1)
        breadth = leave_one_out.loc[
            leave_one_out["analysis_level"].eq("fresh_high_tail_episode")
            & leave_one_out["subgroup_type"].eq("phase")
            & leave_one_out["subgroup"].isin([group_a, group_b])
            & leave_one_out["quantity"].eq(quantity)
        ]
        pivot = breadth.pivot_table(
            index=["period", "exclusion_dimension", "excluded"],
            columns="subgroup",
            values="mean",
        ).dropna()
        if group_a not in pivot or group_b not in pivot:
            continue
        pivot["difference"] = pivot[group_a] - pivot[group_b]
        expected_sign = int(np.sign(estimates[0]))
        broad = True
        for period in ("assessment", "stress"):
            for dimension in ("month", "stock"):
                values = pivot.loc[
                    (pivot.index.get_level_values("period") == period)
                    & (pivot.index.get_level_values("exclusion_dimension") == dimension),
                    "difference",
                ]
                if values.empty or float((np.sign(values) == expected_sign).mean()) < 0.75:
                    broad = False
        if broad:
            stable_separation = True
            break
    if stable_separation:
        return "tail_phase_structural_separation_observed"
    if candidates.empty:
        return "blocked_insufficient_support"
    return "tail_phase_descriptive_only"


def build_summary(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    frozen_median: FrozenConsumedMedianV1,
    comparisons: pd.DataFrame,
    directional: pd.DataFrame,
    missingness: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    concentration: pd.DataFrame,
    m1c_regression: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    composition = _composition_payload(checkpoint, episodes)
    phase_comparisons = comparisons.loc[
        comparisons["estimate_type"].eq("mean")
        & comparisons["comparison"].str.contains(
            "FIRST_ENTRY|PERSISTENT|RE_ENTRY",
            regex=True,
        )
    ]
    consumed_comparisons = comparisons.loc[
        comparisons["estimate_type"].eq("mean")
        & comparisons["comparison"].eq("LOW_OR_EQUAL_minus_HIGH")
    ]
    directional_primary = directional.loc[
        directional["subgroup_type"].isin(["overall", "phase", "movement_consumed_bucket"])
    ]
    conclusion = _structural_conclusion(comparisons, leave_one_out)
    recommendation = (
        "include_phase_as_preregistered_logging_only_interaction"
        if conclusion == "tail_phase_structural_separation_observed"
        else "retain_phase_logging_and_treat_any_interaction_as_secondary_preregistered"
    )
    answers = {
        "1_phase_composition": composition,
        "2_assessment_vs_stress": {
            "assessment_checkpoint": composition.get(
                "assessment|high_tail_checkpoint",
                {},
            ),
            "stress_checkpoint": composition.get("stress|high_tail_checkpoint", {}),
        },
        "3_first_entry_remaining_movement": _records(phase_comparisons),
        "4_consumed_status_and_remaining_movement": _records(consumed_comparisons),
        "5_breadth": _breadth_payload(leave_one_out, concentration),
        "6_frozen_a1_by_phase": _records(
            directional_primary.loc[directional_primary["subgroup_type"].eq("phase")]
        ),
        "7_a1_support_interpretation": (
            "retrospective_descriptive_only_even_when_cell_support_is_adequate"
        ),
        "8_continuation_reversal_mixture": _records(
            directional_primary.loc[
                directional_primary["subgroup_type"].isin(["overall", "phase"])
            ][
                [
                    "period",
                    "analysis_level",
                    "subgroup_type",
                    "subgroup",
                    "recent_momentum_valid",
                    "recent_momentum_continuation_rate",
                    "recent_momentum_reversal_rate",
                ]
            ]
        ),
        "9_prospective_interaction": recommendation,
        "10_remaining_unknown": [
            "prospective bid and ask spreads",
            "queue position and fill probability",
            "trade impact and slippage",
            "contract-selection realism",
            "whether descriptive A1 phase differences persist prospectively",
        ],
    }
    return {
        "research_id": "M1C Tail Phase V1",
        "schema_version": M1C_TAIL_PHASE_V1_VERSION,
        "conclusion": conclusion,
        "m1c_threshold": M1C_FROZEN_THRESHOLD,
        "movement_consumed_median_2024": frozen_median.value,
        "movement_consumed_median_complete_observations": (frozen_median.complete_observations),
        "movement_consumed_median_predictor_values_only": True,
        "chronology": {
            "development": [DEVELOPMENT_START, DEVELOPMENT_END],
            "assessment": [ASSESSMENT_START, ASSESSMENT_END],
            "stress": [STRESS_START, STRESS_END],
            "protected_start": PROTECTED_START,
        },
        "support_rules": {
            "minimum_cell_rows": MINIMUM_CELL_ROWS,
            "minimum_cell_sessions": MINIMUM_CELL_SESSIONS,
            "minimum_a1_action_rows": MINIMUM_A1_ACTION_ROWS,
        },
        "composition": composition,
        "key_phase_comparisons": _records(phase_comparisons),
        "key_consumed_comparisons": _records(consumed_comparisons),
        "directional_diagnostics": _records(directional_primary),
        "missingness": _records(missingness),
        "source_checkpoint_coverage": source_manifest.get(
            "tail_phase_v1_source_coverage",
            {},
        ),
        "m1c_regression": dict(m1c_regression),
        "answers": answers,
        "claims": {
            "directional_edge": False,
            "options_edge": False,
            "tradeable": False,
            "validated_strategy": False,
            "option_pnl_calculated": False,
        },
        "protected_outcomes_accessed": False,
        "orders_enabled_or_placed": False,
    }


def _format_number(value: object, *, digits: int = 6) -> str:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def _composition_line(summary: Mapping[str, Any], key: str) -> str:
    composition = cast(Mapping[str, Any], summary["composition"])
    item = cast(Mapping[str, Any], composition.get(key, {}))
    counts = cast(Mapping[str, Any], item.get("counts", {}))
    shares = cast(Mapping[str, Any], item.get("shares", {}))
    pieces = [
        f"{phase} {int(counts.get(phase, 0))} "
        f"({_format_number(100.0 * float(shares.get(phase, 0.0)), digits=1)}%)"
        for phase in (*PHASE_ORDER, "UNKNOWN_INCOMPLETE")
        if phase in counts
    ]
    return ", ".join(pieces) if pieces else "no rows"


def _comparison_line(
    comparisons: pd.DataFrame,
    *,
    period: str,
    comparison: str,
    quantity: str,
    level: str = "fresh_high_tail_episode",
) -> str:
    item = _comparison_extract(
        comparisons,
        period=period,
        level=level,
        comparison=comparison,
        quantity=quantity,
    )
    if item is None:
        return "unavailable"
    return (
        f"{_format_number(item['estimate_a_minus_b'])} "
        f"(95% session-cluster CI "
        f"[{_format_number(item['lower_95'])}, "
        f"{_format_number(item['upper_95'])}], "
        f"n={item['n_a']} vs {item['n_b']}, {item['status']})"
    )


def render_report(
    summary: Mapping[str, Any],
    comparisons: pd.DataFrame,
    directional: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    concentration: pd.DataFrame,
) -> str:
    median = _format_number(summary["movement_consumed_median_2024"], digits=16)
    conclusion = str(summary["conclusion"])
    source_coverage = cast(
        Mapping[str, Mapping[str, Any]],
        summary.get("source_checkpoint_coverage", {}),
    )
    development_coverage = source_coverage.get("development", {})
    assessment_coverage = source_coverage.get("assessment", {})
    assessment_a1 = directional.loc[
        directional["period"].eq("assessment")
        & directional["analysis_level"].eq("high_tail_checkpoint")
        & directional["subgroup_type"].eq("phase")
    ]
    a1_lines = []
    for row in assessment_a1.itertuples(index=False):
        item = cast(Any, row)
        a1_lines.append(
            f"- {item.subgroup}: actions {int(item.A1_acted_count)}, accuracy "
            f"{_format_number(item.A1_accuracy, digits=4)}, mean aligned 10-minute "
            f"return {_format_number(item.A1_mean_aligned_10m_return)}, "
            f"`{item.status}`."
        )
    overall_direction = directional.loc[
        directional["analysis_level"].eq("high_tail_checkpoint")
        & directional["subgroup_type"].eq("overall")
    ].set_index("period")
    mixture_lines = []
    for period in ("assessment", "stress"):
        if period not in overall_direction.index:
            continue
        item = cast(pd.Series, overall_direction.loc[period])
        mixture_lines.append(
            f"{period.title()} continuation "
            f"{_format_number(item['recent_momentum_continuation_rate'], digits=4)} "
            f"and reversal "
            f"{_format_number(item['recent_momentum_reversal_rate'], digits=4)}"
        )
    breadth = leave_one_out.loc[
        leave_one_out["analysis_level"].eq("fresh_high_tail_episode")
        & leave_one_out["subgroup_type"].eq("phase")
        & leave_one_out["subgroup"].isin(["FIRST_ENTRY", "RE_ENTRY"])
        & leave_one_out["quantity"].eq("future_absolute_15m")
    ]
    breadth_pivot = breadth.pivot_table(
        index=["period", "exclusion_dimension", "excluded"],
        columns="subgroup",
        values="mean",
    ).dropna()
    breadth_pivot["first_minus_reentry"] = breadth_pivot["FIRST_ENTRY"] - breadth_pivot["RE_ENTRY"]
    breadth_positive = int(breadth_pivot["first_minus_reentry"].gt(0.0).sum())
    breadth_total = int(len(breadth_pivot))
    reentry_concentration = concentration.loc[
        concentration["analysis_level"].eq("fresh_high_tail_episode")
        & concentration["subgroup_type"].eq("phase")
        & concentration["subgroup"].eq("RE_ENTRY")
        & concentration["concentration_dimension"].eq("checkpoint")
    ]
    maximum_checkpoint_concentration = (
        float(reentry_concentration["share_within_subgroup"].max())
        if len(reentry_concentration)
        else math.nan
    )
    lines = [
        "# M1C Tail Phase V1",
        "",
        f"Decision: `{conclusion}`.",
        "",
        "This fixed retrospective study used the unchanged frozen M1C probability "
        f"threshold `{M1C_FROZEN_THRESHOLD}` and a 15-minute stock-local causal "
        f"movement-consumed split frozen from 2024 predictors at `{median}` "
        f"(n={summary['movement_consumed_median_complete_observations']}). No 2024 "
        "outcome selected that split.",
        "",
        "## Structural predictability",
        "",
        "Assessment checkpoint composition: "
        + _composition_line(summary, "assessment|high_tail_checkpoint")
        + ".",
        "",
        "Stress checkpoint composition: "
        + _composition_line(summary, "stress|high_tail_checkpoint")
        + ".",
        "",
        "Assessment fresh-episode composition: "
        + _composition_line(summary, "assessment|fresh_high_tail_episode")
        + ". Persistence rows are not treated as independent episode support.",
        "",
        "Stress fresh-episode composition: "
        + _composition_line(summary, "stress|fresh_high_tail_episode")
        + ".",
        "",
        "The phase mix was similar across assessment and stress: persistence was "
        "about half of checkpoint observations, first entries about two-fifths, and "
        "re-entries the smaller remainder.",
        "",
        "## Absolute-movement predictability",
        "",
        "FIRST_ENTRY minus PERSISTENT future absolute 10-minute movement:",
        "",
        "- Assessment: "
        + _comparison_line(
            comparisons,
            period="assessment",
            comparison="FIRST_ENTRY_minus_PERSISTENT",
            quantity="future_absolute_10m",
            level="high_tail_checkpoint",
        )
        + ".",
        "- Stress: "
        + _comparison_line(
            comparisons,
            period="stress",
            comparison="FIRST_ENTRY_minus_PERSISTENT",
            quantity="future_absolute_10m",
            level="high_tail_checkpoint",
        )
        + ".",
        "",
        "FIRST_ENTRY minus RE_ENTRY future absolute 15-minute movement:",
        "",
        "- Assessment: "
        + _comparison_line(
            comparisons,
            period="assessment",
            comparison="FIRST_ENTRY_minus_RE_ENTRY",
            quantity="future_absolute_15m",
        )
        + ".",
        "- Stress: "
        + _comparison_line(
            comparisons,
            period="stress",
            comparison="FIRST_ENTRY_minus_RE_ENTRY",
            quantity="future_absolute_15m",
        )
        + ".",
        "",
        "The FIRST_ENTRY-versus-RE_ENTRY 15-minute difference kept the same "
        f"positive sign in {breadth_positive}/{breadth_total} leave-one-month and "
        "leave-one-stock estimates. Re-entry support was nevertheless concentrated "
        "by checkpoint (maximum checkpoint share "
        f"{_format_number(maximum_checkpoint_concentration, digits=4)}), so this is "
        "structural timing evidence rather than a direction or trading claim.",
        "",
        "LOW_OR_EQUAL minus HIGH movement-consumed future absolute 10-minute movement:",
        "",
        "- Assessment: "
        + _comparison_line(
            comparisons,
            period="assessment",
            comparison="LOW_OR_EQUAL_minus_HIGH",
            quantity="future_absolute_10m",
        )
        + ".",
        "- Stress: "
        + _comparison_line(
            comparisons,
            period="stress",
            comparison="LOW_OR_EQUAL_minus_HIGH",
            quantity="future_absolute_10m",
        )
        + ".",
        "",
        "## Timing and remaining movement",
        "",
        "`post_share_of_local_range_v1` uses a non-overlapping pre-trigger 15-minute "
        "range and future 10-minute range. It is descriptive, bounded, and is not "
        "an option-profitability measure. Session-cluster intervals, leave-one-month-"
        "out, leave-one-stock-out, and month/stock/checkpoint concentration tables "
        "are included as machine-readable artifacts.",
        "",
        "FIRST_ENTRY minus RE_ENTRY `post_share_of_local_range_v1`:",
        "",
        "- Assessment: "
        + _comparison_line(
            comparisons,
            period="assessment",
            comparison="FIRST_ENTRY_minus_RE_ENTRY",
            quantity="post_share_of_local_range_v1",
        )
        + ".",
        "- Stress: "
        + _comparison_line(
            comparisons,
            period="stress",
            comparison="FIRST_ENTRY_minus_RE_ENTRY",
            quantity="post_share_of_local_range_v1",
        )
        + ". Both intervals span zero, so V1 did not establish a phase difference "
        "in this bounded share diagnostic.",
        "",
        "The frozen consumed split was highly imbalanced inside later high-M1C "
        "episodes (assessment n=8 LOW_OR_EQUAL versus 409 HIGH; stress n=9 versus "
        "516), so the consumed-bucket remaining-movement comparison is "
        "`blocked_insufficient_support` rather than negative evidence.",
        "",
        "## Directional predictability",
        "",
        "Direction remains secondary. Frozen A1 was applied unchanged; no A1 "
        "threshold or coefficient was fitted or selected here.",
        "",
        *(a1_lines or ["- No adequately materialised A1 phase rows."]),
        "",
        "; ".join(mixture_lines)
        + ". The near-even continuation/reversal mix is consistent with, but does "
        "not prove, a phase-mixture explanation for weak direction.",
        "",
        "Any apparent phase-specific A1 difference is retrospective and exploratory. "
        "It is not evidence of a directional edge and does not define a combined "
        "A1-plus-phase rule.",
        "",
        "## Option profitability",
        "",
        "Not tested. Previous-close IV is used only as the canonical movement scale. "
        "No option P&L, bid/ask fill, contract-selection return, or tradeability claim "
        "is produced.",
        "",
        "## Execution realism",
        "",
        "This is five-minute underlying-bar research. It cannot answer spread, queue, "
        "fill probability, slippage, or trade-impact questions. Prospective Tail "
        "Phase fields are logging-only and do not alter recorder priority, promotion, "
        "subscriptions, direction, contracts, capacity, or episode inclusion.",
        "",
        "## Operational blockers and scope limits",
        "",
        "- The canonical historical causal checkpoint surface had "
        f"{int(development_coverage.get('rows_without_previous_close_option_context', 0))} "
        "development and "
        f"{int(assessment_coverage.get('rows_without_previous_close_option_context', 0))} "
        "assessment rows without exact previous-close option context. M1C is undefined "
        "for those rows, so they were not scored or silently labelled outside-tail; "
        "their exclusion is reported in source coverage and missingness artifacts. "
        "The observed structural result is conditional on the valid frozen-M1C surface.",
        "- The external Group-O package producer is outside this repository. It must "
        "supply and receipt-hash the canonical previous-close implied 15-minute "
        "movement; until the engineering-transfer checks verify that handoff, "
        "prospective consumed buckets may correctly remain `UNKNOWN_INCOMPLETE` "
        "without interrupting M1C recording.",
        "- Sector context was not present causally in the existing historical surface, "
        "so it is explicitly out of scope; no external data was acquired.",
        "- No existing frozen market-volatility state was present on this surface.",
        "- Re-entry or interaction cells below 30 rows or 10 sessions are labelled "
        "`blocked_insufficient_support`; thresholds were not relaxed.",
        "- The first 20 IBKR/EODHD transfer sessions remain `engineering_transfer` "
        "and may verify logging mechanics only.",
        "",
        "## Answers to the ten preregistered questions",
        "",
        "1. The exact phase composition is reported above and in `structural_counts_v1.csv`.",
        "2. Assessment and stress use identical definitions; their observed "
        "composition is reported separately without post-hoc subgroup selection.",
        "3. FIRST_ENTRY comparisons against PERSISTENT and RE_ENTRY are reported with "
        "session-cluster support and intervals above.",
        "4. The fixed LOW_OR_EQUAL versus HIGH comparison reports whether greater "
        "pre-trigger consumption corresponds to less remaining movement.",
        "5. Breadth is reported through leave-one-month, leave-one-stock, and "
        "concentration artifacts; outliers were retained.",
        "6. Frozen A1 action counts, accuracy, and aligned returns are reported by "
        "phase and consumed bucket.",
        "7. No apparent A1 improvement is confirmatory in this retrospective V1; "
        "small interactions are explicitly blocked.",
        "8. Continuation and reversal rates are descriptive checks for a mixture "
        "explanation, not a directional claim.",
        "9. Phase may be retained as a preregistered, logging-only prospective "
        "interaction according to the summary recommendation; it is not a gate.",
        "10. Prospective bid/ask, fills, and impact remain unknown.",
        "",
        "Protected 2026 historical outcomes were not opened, calculated, displayed, "
        "or inspected. No order-routing path was enabled and no order was placed.",
        "",
    ]
    return "\n".join(lines)


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip("\r\n")


def build_provenance(
    checkpoint: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    source_manifest: Mapping[str, Any],
    output_paths: Sequence[Path],
) -> dict[str, Any]:
    status = _git_output("status", "--short")
    high_tail = checkpoint.loc[_bool_series(checkpoint["m1c_high_tail_v1"])]
    output_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in output_paths
        if path.is_file()
    }
    return {
        "research_id": "M1C Tail Phase V1",
        "schema_version": M1C_TAIL_PHASE_V1_VERSION,
        "current_branch": _git_output("branch", "--show-current"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "dirty_working_tree": bool(status),
        "dirty_status": status.splitlines(),
        "relevant_configuration_hashes": {
            "v1_contract": sha256_file(EXPERIMENT_DIR / "contract.json"),
            "v1_frozen_config": sha256_file(PRIMARY / "frozen_config_v1.json"),
            "prospective_tail_phase_schema_manifest": sha256_file(
                REPO_ROOT
                / "research"
                / "prospective"
                / "frozen-m1c-microstructure-recorder-v0"
                / "schema_manifest.json"
            ),
            "frozen_m1c_feature_manifest": sha256_file(
                SOURCE_PRIMARY / "causal_movement_feature_manifest.json"
            ),
            "frozen_m1c_threshold": sha256_file(SOURCE_PRIMARY / "causal_movement_threshold.json"),
            "frozen_a1_models": sha256_file(SOURCE_PRIMARY / "model_configurations.json"),
            "frozen_a1_normalisation": sha256_file(
                SOURCE_PRIMARY / "stock_local_normalisation_parameters.json"
            ),
            "frozen_a1_thresholds": sha256_file(
                SOURCE_PRIMARY / "frozen_archetype_thresholds.json"
            ),
            "frozen_stock_market_betas": sha256_file(
                SOURCE_PRIMARY / "stock_market_beta_parameters.csv"
            ),
        },
        "input_artifact_identities": source_manifest.get("sources", []),
        "row_counts": {
            "checkpoint_rows": int(len(checkpoint)),
            "raw_high_tail_checkpoint_rows": int(len(high_tail)),
            "valid_phase_high_tail_checkpoint_rows": int(
                high_tail["m1c_tail_phase_v1"].isin(PHASE_ORDER).sum()
            ),
            "fresh_episode_rows": int(len(episodes)),
            "by_partition": {
                str(key): int(value)
                for key, value in checkpoint.groupby("partition").size().items()
            },
            "high_tail_by_partition": {
                str(key): int(value) for key, value in high_tail.groupby("partition").size().items()
            },
            "episodes_by_partition": {
                str(key): int(value) for key, value in episodes.groupby("partition").size().items()
            },
        },
        "exclusion_counts_and_reasons": {
            "development_missing_previous_close_option_context": int(
                cast(
                    Mapping[str, Mapping[str, Any]],
                    source_manifest.get("tail_phase_v1_source_coverage", {}),
                )
                .get("development", {})
                .get("rows_without_previous_close_option_context", 0)
            ),
            "assessment_missing_previous_close_option_context": int(
                cast(
                    Mapping[str, Mapping[str, Any]],
                    source_manifest.get("tail_phase_v1_source_coverage", {}),
                )
                .get("assessment", {})
                .get("rows_without_previous_close_option_context", 0)
            ),
            "outside_tail_not_in_structural_outcome_analysis": int(
                (~_bool_series(checkpoint["m1c_high_tail_v1"])).sum()
            ),
            "high_tail_unknown_incomplete_phase": int(
                high_tail["m1c_tail_phase_v1"].eq("UNKNOWN_INCOMPLETE").sum()
            ),
            "high_tail_incomplete_movement_consumed": int(
                (~_bool_series(high_tail["movement_consumed_complete_v1"])).sum()
            ),
            "high_tail_unavailable_future_10m": int(
                (~_bool_series(high_tail["available_10m"])).sum()
            ),
            "high_tail_unavailable_future_15m": int(
                (~_bool_series(high_tail["available_15m"])).sum()
            ),
            "high_tail_incomplete_frozen_a1": int(
                (~_bool_series(high_tail["A1_complete_v1"])).sum()
            ),
            "outliers_removed": 0,
            "stocks_removed_after_outcome_review": 0,
            "months_removed_after_outcome_review": 0,
        },
        "missingness_counts": {
            "phase_missing_reasons": {
                str(key): int(value)
                for key, value in high_tail["phase_missing_reason_v1"]
                .fillna("complete")
                .value_counts()
                .items()
            },
            "movement_consumed_missing_reasons": {
                str(key): int(value)
                for key, value in high_tail["movement_consumed_missing_reason_v1"]
                .fillna("complete")
                .value_counts()
                .items()
            },
            "post_share_missing_reasons": {
                str(key): int(value)
                for key, value in high_tail["post_share_of_local_range_missing_reason_v1"]
                .fillna("complete")
                .value_counts()
                .items()
            },
        },
        "exact_commands_used": [RUN_COMMAND],
        "date_boundaries": {
            "development": [DEVELOPMENT_START, DEVELOPMENT_END],
            "assessment": [ASSESSMENT_START, ASSESSMENT_END],
            "stress": [STRESS_START, STRESS_END],
            "protected_start": PROTECTED_START,
        },
        "maximum_session_opened": str(
            max(
                checkpoint["session"].astype(str).max(),
                episodes["session"].astype(str).max(),
            )
        ),
        "protected_data_confirmation": {
            "protected_data_opened": False,
            "protected_outcomes_calculated": False,
            "protected_outcomes_displayed": False,
            "protected_outcomes_inspected": False,
            "source_manifest_protected_rows_read": source_manifest.get("protected_rows_read"),
        },
        "causality_confirmation": {
            "archived_signed_pressure_used": False,
            "archived_tension_used": False,
            "future_filtered_peer_slate_used": False,
            "peer_normalisation_used_for_v1_features": False,
            "m1c_refit": False,
            "a1_refit": False,
            "fresh_episode_definition_changed": False,
        },
        "execution_confirmation": {
            "broker_access": False,
            "order_routing_enabled": False,
            "orders_submitted": False,
        },
        "output_hashes": output_hashes,
    }


def run() -> dict[str, Any]:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    print("loading hash-verified permitted inputs", flush=True)
    historical, stress, bars, source_manifest = load_permitted_inputs()
    print("loading unchanged frozen M1C and A1 runtimes", flush=True)
    m1c_runtime, direction_builder, direction_runtime = load_frozen_runtimes()
    print("building causal phase, consumed, outcome, and A1 rows", flush=True)
    checkpoint, episodes, frozen_median, m1c_regression = build_research_frames(
        historical,
        stress,
        bars,
        m1c_runtime=m1c_runtime,
        direction_builder=direction_builder,
        direction_runtime=direction_runtime,
    )

    frozen_config_path = PRIMARY / "frozen_config_v1.json"
    canonical_path = PRIMARY / "canonical_implementation_v1.json"
    source_path = PRIMARY / "source_manifest_v1.json"
    write_json(frozen_config_path, frozen_config_payload(frozen_median))
    write_json(canonical_path, canonical_implementation_payload())
    write_json(source_path, dict(source_manifest))

    checkpoint_output, episode_output = artifact_views(
        checkpoint,
        episodes,
        source_manifest=source_manifest,
    )
    checkpoint_path = PRIMARY / "checkpoint_results_v1.parquet"
    episode_path = PRIMARY / "fresh_episode_results_v1.parquet"
    write_parquet(checkpoint_path, checkpoint_output)
    write_parquet(episode_path, episode_output)

    print("running fixed summaries and 1,000-draw session-cluster bootstrap", flush=True)
    counts = structural_counts(checkpoint, episodes)
    missingness = missingness_table(
        checkpoint,
        episodes,
        source_manifest=source_manifest,
    )
    movement = movement_summary_table(checkpoint, episodes)
    bootstrap = bootstrap_estimates(checkpoint, episodes)
    comparisons = key_comparisons(checkpoint, episodes)
    leave_one_out = leave_one_out_diagnostics(checkpoint, episodes)
    concentration = concentration_table(checkpoint, episodes)
    directional = directional_diagnostics(checkpoint, episodes)
    regime = regime_diagnostics(checkpoint, episodes)
    table_paths = {
        "structural_counts_v1.csv": counts,
        "missingness_v1.csv": missingness,
        "movement_summaries_v1.csv": movement,
        "cluster_bootstrap_v1.csv": bootstrap,
        "key_comparisons_v1.csv": comparisons,
        "leave_one_out_v1.csv": leave_one_out,
        "concentration_v1.csv": concentration,
        "directional_diagnostics_v1.csv": directional,
        "regime_diagnostics_v1.csv": regime,
    }
    for name, frame in table_paths.items():
        write_csv(PRIMARY / name, frame)

    summary = build_summary(
        checkpoint,
        episodes,
        frozen_median=frozen_median,
        comparisons=comparisons,
        directional=directional,
        missingness=missingness,
        leave_one_out=leave_one_out,
        concentration=concentration,
        m1c_regression=m1c_regression,
        source_manifest=source_manifest,
    )
    summary_path = PRIMARY / "summary_v1.json"
    write_json(summary_path, summary)
    report = render_report(
        summary,
        comparisons,
        directional,
        leave_one_out,
        concentration,
    )
    report_path = PRIMARY / "report.md"
    reports_path = REPORTS / "report.md"
    report_path.write_text(report, encoding="utf-8")
    reports_path.write_text(report, encoding="utf-8")

    output_paths = [
        frozen_config_path,
        canonical_path,
        source_path,
        checkpoint_path,
        episode_path,
        *[PRIMARY / name for name in table_paths],
        summary_path,
        report_path,
        reports_path,
    ]
    provenance = build_provenance(
        checkpoint,
        episodes,
        source_manifest=source_manifest,
        output_paths=output_paths,
    )
    write_json(PRIMARY / "provenance_manifest_v1.json", provenance)
    write_json(
        PRIMARY / "operational_failure_v1.json",
        {
            "research_id": "M1C Tail Phase V1",
            "status": "no_unresolved_operational_failure",
            "conclusion": summary["conclusion"],
            "protected_outcomes_accessed": False,
            "orders_enabled_or_placed": False,
        },
    )
    print(f"completed: {summary['conclusion']}", flush=True)
    return summary


def main() -> None:
    try:
        run()
    except Exception as error:
        PRIMARY.mkdir(parents=True, exist_ok=True)
        failure = {
            "research_id": "M1C Tail Phase V1",
            "conclusion": "operational_failure",
            "blocker": type(error).__name__,
            "detail": str(error),
            "protected_outcomes_accessed": False,
            "orders_enabled_or_placed": False,
            "safest_next_action": (
                "repair the named technical blocker and rerun the same frozen command"
            ),
        }
        write_json(PRIMARY / "operational_failure_v1.json", failure)
        (PRIMARY / "report.md").write_text(
            "# M1C Tail Phase V1\n\n"
            "Decision: `operational_failure`.\n\n"
            f"Blocker: `{type(error).__name__}` — {error}\n\n"
            "Protected 2026 outcomes were not accessed. No order was enabled or placed.\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
