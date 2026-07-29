"""Fail-closed reconstruction helpers for dense five-minute signed pressure.

The archived sparse pressure is an equal-weight mean of four development-scaled
components.  This module preserves that lineage and makes population causality
explicit.  It does not define an alternative pressure primitive.
"""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DENSE_CHECKPOINTS: Final[tuple[int, ...]] = tuple(range(1, 35))
SPARSE_CHECKPOINTS: Final[tuple[int, ...]] = tuple(range(6, 35, 2))
SPARSE_TOLERANCE: Final[float] = 1e-12
EPSILON: Final[float] = 1e-12
DEVELOPMENT_START: Final[pd.Timestamp] = pd.Timestamp("2024-01-01")
DEVELOPMENT_END: Final[pd.Timestamp] = pd.Timestamp("2024-12-31")
ASSESSMENT_START: Final[pd.Timestamp] = pd.Timestamp("2025-01-01")
ASSESSMENT_END: Final[pd.Timestamp] = pd.Timestamp("2025-08-22")
EXCLUDED_START: Final[pd.Timestamp] = pd.Timestamp("2025-09-01")
PROTECTED_START: Final[pd.Timestamp] = pd.Timestamp("2026-01-01")
PRESSURE_COMPONENTS: Final[tuple[str, ...]] = (
    "signed_progress",
    "signed_efficiency",
    "mean_close_location",
    "boundary_slope",
)
PHASE1_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "dense_signed_pressure_reconstruction_supported",
        "blocked_signed_pressure_lineage_not_found",
        "blocked_dense_pressure_upstream_dependency",
        "blocked_sparse_pressure_compatibility_failure",
        "blocked_dense_pressure_causality_failure",
        "blocked_insufficient_dense_pressure_coverage",
        "blocked_dense_pressure_reproducibility_failure",
    }
)
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DependencyClassification:
    """Classification of a binding upstream dependency."""

    classification: str
    causality_status: str
    changed_slates: int
    changed_rows: int


@dataclass(frozen=True, slots=True)
class SignedPressureLineage:
    """Exact source-level lineage of the archived signed-pressure primitive."""

    source_file: str
    function_name: str
    function_signature: str
    route_call_site_file: str
    route_call_site_function: str
    input_columns: tuple[str, ...]
    component_weights: str
    formula_hash: str
    source_sha256: str
    route_source_sha256: str


@dataclass(frozen=True, slots=True)
class CoverageGateResult:
    """Fixed dense-pressure episode support gate."""

    passed: bool
    development_complete_episodes: int
    development_sessions: int
    development_stocks: int
    development_months: int
    assessment_complete_episodes: int
    assessment_sessions: int
    assessment_stocks: int
    assessment_month_groups: int


def sha256_file(path: Path) -> str:
    """Hash a source or artifact without modifying it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-like mapping with stable key ordering."""

    import json

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def discover_signed_pressure_lineage(
    behavioural_source: Path,
    route_call_site: Path,
) -> SignedPressureLineage:
    """Locate the exact formula and its sparse-panel call site."""

    source_text = behavioural_source.read_text(encoding="utf-8")
    route_text = route_call_site.read_text(encoding="utf-8")
    source_tree = ast.parse(source_text)
    route_tree = ast.parse(route_text)
    pressure_node = next(
        (
            node
            for node in ast.walk(source_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_signed_pressure"
        ),
        None,
    )
    route_node = next(
        (
            node
            for node in ast.walk(route_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "add_development_frozen_baseline_features"
        ),
        None,
    )
    if pressure_node is None or route_node is None:
        raise ValueError("exact signed-pressure lineage was not found")
    formula = ast.get_source_segment(source_text, pressure_node)
    route_formula = ast.get_source_segment(route_text, route_node)
    if formula is None or route_formula is None:
        raise ValueError("signed-pressure source segment was unavailable")
    required_formula_tokens = (
        '"z_signed_progress"',
        '"z_signed_efficiency"',
        '"z_mean_close_location"',
        '"z_boundary_slope"',
        "values.mean(axis=1)",
    )
    required_route_tokens = (
        'groupby(["session", "checkpoint"]',
        "progress - np.median(progress)",
        "fit_component_scaling(",
        'panel["signed_pressure"]',
    )
    if not all(token in formula for token in required_formula_tokens) or not all(
        token in route_formula for token in required_route_tokens
    ):
        raise ValueError("signed-pressure lineage is ambiguous or has changed")
    arguments = ", ".join(argument.arg for argument in pressure_node.args.args)
    formula_hash = hashlib.sha256(
        (formula + "\n" + "\n".join(required_route_tokens)).encode()
    ).hexdigest()
    return SignedPressureLineage(
        source_file=str(behavioural_source),
        function_name="_signed_pressure",
        function_signature=f"_signed_pressure({arguments})",
        route_call_site_file=str(route_call_site),
        route_call_site_function="add_development_frozen_baseline_features",
        input_columns=tuple(f"z_{component}" for component in PRESSURE_COMPONENTS),
        component_weights="equal_arithmetic_mean",
        formula_hash=formula_hash,
        source_sha256=sha256_file(behavioural_source),
        route_source_sha256=sha256_file(route_call_site),
    )


def validate_authorized_sessions(
    frame: pd.DataFrame,
    *,
    session_column: str = "session",
) -> dict[str, object]:
    """Reject opened-holdout and protected sessions before reconstruction."""

    if session_column not in frame:
        raise ValueError(f"session column missing: {session_column}")
    sessions = pd.to_datetime(frame[session_column], errors="raise").dt.tz_localize(None)
    if sessions.empty:
        raise ValueError("authorized historical rows are required")
    excluded = sessions.between(EXCLUDED_START, pd.Timestamp("2025-12-31"), inclusive="both")
    protected = sessions.ge(PROTECTED_START)
    before_start = sessions.lt(DEVELOPMENT_START)
    after_assessment = sessions.gt(ASSESSMENT_END)
    if (
        bool(excluded.any())
        or bool(protected.any())
        or bool(before_start.any())
        or bool(after_assessment.any())
    ):
        raise ValueError("opened-holdout, protected, or out-of-contract session encountered")
    return {
        "passed": True,
        "minimum_session": str(sessions.min().date()),
        "maximum_session": str(sessions.max().date()),
        "opened_holdout_rows": int(excluded.sum()),
        "protected_rows": int(protected.sum()),
    }


def _required_trace_columns() -> tuple[str, ...]:
    return (
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "open",
        "high",
        "low",
        "close",
        "historical_relative_activity",
    )


def build_dense_bar_grid(trace: pd.DataFrame) -> pd.DataFrame:
    """Build the checkpoint 1..34 completed-bar grid without filling missing bars."""

    missing = sorted(set(_required_trace_columns()).difference(trace.columns))
    if missing:
        raise ValueError(f"dense-grid source columns missing: {missing}")
    validate_authorized_sessions(trace)
    ordered = trace.loc[:, list(_required_trace_columns())].copy()
    ordered["session"] = ordered["session"].astype(str)
    ordered["bar_ordinal"] = pd.to_numeric(ordered["bar_ordinal"], errors="raise").astype(int)
    duplicates = ordered.duplicated(["symbol", "session", "bar_ordinal"], keep=False)
    if bool(duplicates.any()):
        raise ValueError("duplicate completed bars make checkpoint alignment ambiguous")
    ordered = ordered.sort_values(
        ["symbol", "session", "bar_ordinal"], kind="mergesort"
    ).reset_index(drop=True)
    grouped = ordered.groupby(["symbol", "session"], sort=False)
    ordered["_prefix_count"] = grouped.cumcount() + 1
    ordered["_prefix_contiguous"] = ordered["_prefix_count"].eq(ordered["bar_ordinal"] + 1)
    ordered["_activity_missing"] = ~np.isfinite(
        pd.to_numeric(ordered["historical_relative_activity"], errors="coerce").to_numpy(
            dtype=float
        )
    )
    ordered["_prefix_activity_missing"] = grouped["_activity_missing"].cumsum().gt(0)
    start = pd.to_datetime(ordered["bar_start_timestamp"], utc=True)
    complete = pd.to_datetime(ordered["bar_complete_timestamp"], utc=True)
    first_start = grouped["bar_start_timestamp"].transform("first")
    expected_complete = pd.to_datetime(first_start, utc=True) + pd.to_timedelta(
        5 * (ordered["bar_ordinal"] + 1), unit="min"
    )
    ordered["_timestamp_aligned"] = (complete - start).eq(pd.Timedelta(minutes=5)) & complete.eq(
        expected_complete
    )
    ordered["checkpoint"] = ordered["bar_ordinal"] + 1
    current = ordered.loc[
        ordered["checkpoint"].isin(DENSE_CHECKPOINTS),
        [
            "symbol",
            "session",
            "checkpoint",
            "bar_complete_timestamp",
            "open",
            "close",
            "_prefix_contiguous",
            "_prefix_activity_missing",
            "_timestamp_aligned",
        ],
    ].copy()
    session_open = (
        ordered.loc[ordered["bar_ordinal"].eq(0), ["symbol", "session", "open"]]
        .rename(columns={"open": "session_open"})
        .copy()
    )
    current = current.merge(
        session_open,
        on=["symbol", "session"],
        how="left",
        validate="many_to_one",
    )
    sessions = ordered.loc[:, ["symbol", "session"]].drop_duplicates().copy()
    sessions["_cross"] = 1
    checkpoints = pd.DataFrame({"checkpoint": DENSE_CHECKPOINTS, "_cross": 1})
    grid = sessions.merge(checkpoints, on="_cross", how="inner").drop(columns="_cross")
    grid = grid.merge(
        current,
        on=["symbol", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    observed_keys = set(
        ordered.loc[:, ["symbol", "session", "bar_ordinal"]].itertuples(index=False, name=None)
    )
    grid["bar_present"] = grid["bar_complete_timestamp"].notna()
    grid["prefix_contiguous"] = grid["_prefix_contiguous"].eq(True)
    grid["activity_proxy_history_present"] = grid["_prefix_activity_missing"].eq(False)
    grid["timestamp_aligned"] = grid["_timestamp_aligned"].eq(True)
    future_flags: list[bool] = []
    for symbol, session, checkpoint in grid.loc[:, ["symbol", "session", "checkpoint"]].itertuples(
        index=False, name=None
    ):
        future_flags.append(
            all(
                (str(symbol), str(session), int(checkpoint) + offset) in observed_keys
                for offset in range(3)
            )
        )
    grid["three_future_bars_available"] = future_flags
    grid["current_history_available"] = (
        grid["bar_present"]
        & grid["prefix_contiguous"]
        & grid["activity_proxy_history_present"]
        & grid["timestamp_aligned"]
    )
    grid["partition"] = np.where(
        pd.to_datetime(grid["session"]).le(DEVELOPMENT_END),
        "development",
        "assessment",
    )
    grid["missing_reason_code"] = np.select(
        [
            ~grid["bar_present"],
            ~grid["prefix_contiguous"],
            ~grid["timestamp_aligned"],
            ~grid["activity_proxy_history_present"],
        ],
        [
            "completed_bar_missing",
            "noncontiguous_completed_bar_prefix",
            "ambiguous_timestamp_alignment",
            "activity_proxy_history_missing",
        ],
        default="",
    )
    return (
        grid.drop(columns=["_prefix_contiguous", "_prefix_activity_missing", "_timestamp_aligned"])
        .sort_values(["session", "checkpoint", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def causal_progress_surface(grid: pd.DataFrame) -> pd.DataFrame:
    """Center open-to-checkpoint progress on the current-bar causal stock slate."""

    required = {
        "symbol",
        "session",
        "checkpoint",
        "close",
        "session_open",
        "current_history_available",
        "three_future_bars_available",
    }
    missing = sorted(required.difference(grid.columns))
    if missing:
        raise ValueError(f"causal-progress columns missing: {missing}")
    output = grid.loc[grid["current_history_available"].astype(bool)].copy()
    output["raw_progress_bps"] = 10_000.0 * (
        pd.to_numeric(output["close"], errors="raise")
        / pd.to_numeric(output["session_open"], errors="raise")
        - 1.0
    )
    output["causal_current_slate_median_bps"] = output.groupby(
        ["session", "checkpoint"], sort=False
    )["raw_progress_bps"].transform("median")
    output["causal_signed_progress_bps"] = (
        output["raw_progress_bps"] - output["causal_current_slate_median_bps"]
    )
    future_subset = output.loc[output["three_future_bars_available"].astype(bool)].copy()
    future_subset["archived_future_filtered_median_bps"] = future_subset.groupby(
        ["session", "checkpoint"], sort=False
    )["raw_progress_bps"].transform("median")
    future_subset["archived_future_filtered_signed_progress_bps"] = (
        future_subset["raw_progress_bps"] - future_subset["archived_future_filtered_median_bps"]
    )
    output = output.merge(
        future_subset.loc[
            :,
            [
                "symbol",
                "session",
                "checkpoint",
                "archived_future_filtered_median_bps",
                "archived_future_filtered_signed_progress_bps",
            ],
        ],
        on=["symbol", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    return output


def classify_cross_sectional_dependency(frame: pd.DataFrame) -> DependencyClassification:
    """Detect future-filtered membership in a session/checkpoint normalization slate."""

    required = {
        "stock",
        "session",
        "checkpoint",
        "current_history_available",
        "three_future_bars_available",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"dependency-classification columns missing: {missing}")
    available = frame.loc[frame["current_history_available"].astype(bool)].copy()
    future_filtered = available.loc[~available["three_future_bars_available"].astype(bool)]
    changed_slates = int(
        future_filtered.loc[:, ["session", "checkpoint"]].drop_duplicates().shape[0]
    )
    if future_filtered.empty:
        return DependencyClassification(
            classification="A",
            causality_status="causal_current_bar_population",
            changed_slates=0,
            changed_rows=0,
        )
    return DependencyClassification(
        classification="D",
        causality_status="future_dependent_population_membership",
        changed_slates=changed_slates,
        changed_rows=int(len(future_filtered)),
    )


def _least_squares_slope(values: FloatArray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    centered = x - float(x.mean())
    denominator = float(centered @ centered)
    if denominator <= EPSILON:
        return 0.0
    return float(centered @ (values - float(values.mean())) / denominator)


def exact_pressure_raw_components(
    bars: pd.DataFrame,
    *,
    centered_signed_progress_bps: float,
) -> dict[str, float]:
    """Extract the four unchanged raw inputs used by signed pressure at any prefix."""

    required = {
        "open",
        "high",
        "low",
        "close",
        "historical_relative_activity",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"pressure-component columns missing: {missing}")
    if bars.empty:
        raise ValueError("at least one completed bar is required")
    values = bars.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=float)
    activity = pd.to_numeric(bars["historical_relative_activity"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all() or not np.isfinite(activity).all():
        raise ValueError("pressure-component inputs must be finite")
    if bool((values <= 0.0).any()) or bool((activity < 0.0).any()):
        raise ValueError("pressure-component prices must be positive and activity non-negative")
    open_ = values[:, 0]
    high = values[:, 1]
    low = values[:, 2]
    close = values[:, 3]
    width = high - low
    if bool((width < 0.0).any()):
        raise ValueError("bar high must not be below bar low")
    previous_close = np.roll(close, 1)
    previous_close[0] = open_[0]
    returns = 10_000.0 * (close / previous_close - 1.0)
    signed_efficiency = float(returns.sum() / max(float(np.abs(returns).sum()), EPSILON))
    close_location = np.full(len(bars), 0.5, dtype=np.float64)
    nonzero = width > EPSILON
    close_location[nonzero] = (close[nonzero] - low[nonzero]) / width[nonzero]
    opening_range = float(high.max() - low.min())
    boundary_slope = 0.5 * (
        _least_squares_slope(high) / max(opening_range, EPSILON)
        + _least_squares_slope(low) / max(opening_range, EPSILON)
    )
    return {
        "signed_progress": float(centered_signed_progress_bps),
        "signed_efficiency": signed_efficiency,
        "mean_close_location": float(np.clip(close_location, 0.0, 1.0).mean()),
        "boundary_slope": boundary_slope,
    }


def standardized_signed_pressure(components: Mapping[str, float]) -> float:
    """Apply the frozen equal-weight arithmetic mean to four standardized inputs."""

    missing = sorted(set(PRESSURE_COMPONENTS).difference(components))
    if missing:
        raise ValueError(f"standardized pressure components missing: {missing}")
    values = np.asarray([components[name] for name in PRESSURE_COMPONENTS], dtype=float)
    if not np.isfinite(values).all():
        return math.nan
    return float(values.mean())


def apply_component_scale(
    value: float,
    parameters: Mapping[str, object],
) -> float:
    """Apply the archived median/IQR scale and clipping rule."""

    center = float(cast(Any, parameters["center"]))
    scale = float(cast(Any, parameters["scale"]))
    lower = float(cast(Any, parameters["clip_lower"]))
    upper = float(cast(Any, parameters["clip_upper"]))
    if not all(math.isfinite(number) for number in (value, center, scale, lower, upper)):
        return math.nan
    if scale <= 0.0:
        raise ValueError("component scale must be positive")
    return float(np.clip((value - center) / scale, lower, upper))


def compare_causal_candidate_to_sparse(
    sparse: pd.DataFrame,
    progress: pd.DataFrame,
    component_scaling: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> pd.DataFrame:
    """Quantify the exact compatibility cost of replacing future-filtered membership."""

    source = sparse.copy()
    if "stock" in source and "symbol" not in source:
        source = source.rename(columns={"stock": "symbol"})
    required_sparse = {
        "symbol",
        "session",
        "checkpoint",
        "signed_pressure",
        "z_component__signed_progress",
        "raw_component__signed_progress",
    }
    missing_sparse = sorted(required_sparse.difference(source.columns))
    if missing_sparse:
        raise ValueError(f"sparse comparison columns missing: {missing_sparse}")
    joined = source.merge(
        progress.loc[
            :,
            [
                "symbol",
                "session",
                "checkpoint",
                "causal_signed_progress_bps",
                "archived_future_filtered_signed_progress_bps",
            ],
        ],
        on=["symbol", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    if bool(joined["causal_signed_progress_bps"].isna().any()):
        raise ValueError("causal signed-progress candidate missing for sparse row")
    causal_z: list[float] = []
    for checkpoint, value in joined.loc[:, ["checkpoint", "causal_signed_progress_bps"]].itertuples(
        index=False, name=None
    ):
        parameters = component_scaling[str(int(checkpoint))]["signed_progress"]
        causal_z.append(apply_component_scale(float(value), parameters))
    joined["causal_candidate_z_signed_progress"] = causal_z
    joined["dense_pressure"] = (
        joined["signed_pressure"]
        + (joined["causal_candidate_z_signed_progress"] - joined["z_component__signed_progress"])
        / 4.0
    )
    joined = joined.rename(columns={"signed_pressure": "existing_sparse_pressure"})
    joined["absolute_difference"] = (
        joined["dense_pressure"] - joined["existing_sparse_pressure"]
    ).abs()
    joined["exceeds_tolerance"] = joined["absolute_difference"].gt(SPARSE_TOLERANCE)
    joined["dense_pressure_valid"] = False
    joined["comparison_basis"] = "causal_current_slate_diagnostic_not_authorized_dense_pressure"
    joined["archived_raw_component_reproduced"] = np.isclose(
        joined["archived_future_filtered_signed_progress_bps"],
        joined["raw_component__signed_progress"],
        atol=SPARSE_TOLERANCE,
        rtol=0.0,
        equal_nan=False,
    )
    return joined.sort_values(["session", "checkpoint", "symbol"], kind="mergesort").reset_index(
        drop=True
    )


def sparse_compatibility_summary(comparison: pd.DataFrame) -> dict[str, object]:
    """Summarize sparse compatibility under the binding 1e-12 tolerance."""

    differences = pd.to_numeric(comparison["absolute_difference"], errors="coerce")
    finite = differences.dropna()
    rows_exceeding = int(finite.gt(SPARSE_TOLERANCE).sum())
    return {
        "joined_rows": int(len(comparison)),
        "missing_dense_rows": int(differences.isna().sum()),
        "missing_sparse_rows": int(comparison["existing_sparse_pressure"].isna().sum()),
        "exact_matches": int(finite.eq(0.0).sum()),
        "maximum_absolute_difference": float(finite.max()) if len(finite) else math.nan,
        "mean_absolute_difference": float(finite.mean()) if len(finite) else math.nan,
        "median_absolute_difference": float(finite.median()) if len(finite) else math.nan,
        "p95_absolute_difference": (float(finite.quantile(0.95)) if len(finite) else math.nan),
        "rows_exceeding_1e_12": rows_exceeding,
        "passed": bool(
            len(finite) == len(comparison)
            and rows_exceeding == 0
            and float(finite.max()) <= SPARSE_TOLERANCE
        ),
    }


def invalid_dense_pressure_surface(
    grid: pd.DataFrame,
    *,
    formula_hash: str,
    preprocessing_hash: str,
    source_lineage_version: str,
) -> pd.DataFrame:
    """Materialize explicit invalid rows after a class-D dependency blocks pressure."""

    materialized = grid.loc[grid["bar_present"].astype(bool)].copy()
    result = pd.DataFrame(
        {
            "stock": materialized["symbol"].astype(str),
            "session": materialized["session"].astype(str),
            "bar_timestamp": pd.to_datetime(materialized["bar_complete_timestamp"], utc=True),
            "checkpoint_index": materialized["checkpoint"].astype(int),
            "signed_pressure": np.nan,
            "upstream_current_history_present": materialized["current_history_available"].astype(
                bool
            ),
            "upstream_activity_proxy_present": materialized[
                "activity_proxy_history_present"
            ].astype(bool),
            "upstream_cross_sectional_normalization_present": False,
            "pressure_valid": False,
            "missing_reason_code": "blocked_future_dependent_cross_sectional_population",
            "partition": materialized["partition"].astype(str),
            "source_lineage_version": source_lineage_version,
            "formula_hash": formula_hash,
            "preprocessing_hash": preprocessing_hash,
        }
    )
    return result.sort_values(
        ["session", "checkpoint_index", "stock"], kind="mergesort"
    ).reset_index(drop=True)


def pressure_window_audit(
    episodes: pd.DataFrame,
    dense_pressure: pd.DataFrame,
) -> pd.DataFrame:
    """Count exact T-5..T-1 valid dense pressure bars for each fresh episode."""

    required_episodes = {"stock", "session", "checkpoint", "partition"}
    missing = sorted(required_episodes.difference(episodes.columns))
    if missing:
        raise ValueError(f"episode coverage columns missing: {missing}")
    valid = dense_pressure.loc[dense_pressure["pressure_valid"].astype(bool)].copy()
    valid_keys = set(
        valid.loc[:, ["stock", "session", "checkpoint_index"]].itertuples(index=False, name=None)
    )
    rows: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        stock = str(episode.stock)
        session = str(episode.session)
        checkpoint = int(cast(Any, episode.checkpoint))
        required_checkpoints = tuple(range(checkpoint - 5, checkpoint))
        present = sum(
            (stock, session, required_checkpoint) in valid_keys
            for required_checkpoint in required_checkpoints
        )
        rows.append(
            {
                "stock": stock,
                "session": session,
                "month": session[:7],
                "checkpoint": checkpoint,
                "partition": str(episode.partition),
                "required_pressure_checkpoints": ",".join(
                    str(value) for value in required_checkpoints
                ),
                "valid_pressure_bars": int(present),
                "complete_five_bar_pressure_window": bool(present == 5),
                "missing_window_reason": (
                    "" if present == 5 else "binding_dense_pressure_dependency_blocked"
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_coverage_gate(coverage: pd.DataFrame) -> CoverageGateResult:
    """Apply the frozen development and assessment complete-window support gates."""

    complete = coverage.loc[coverage["complete_five_bar_pressure_window"].astype(bool)].copy()
    development = complete.loc[complete["partition"].eq("development")]
    assessment = complete.loc[complete["partition"].eq("assessment")]
    development_months = int(development["month"].nunique())
    assessment_months = int(assessment["month"].nunique())
    result = CoverageGateResult(
        passed=False,
        development_complete_episodes=int(len(development)),
        development_sessions=int(development["session"].nunique()),
        development_stocks=int(development["stock"].nunique()),
        development_months=development_months,
        assessment_complete_episodes=int(len(assessment)),
        assessment_sessions=int(assessment["session"].nunique()),
        assessment_stocks=int(assessment["stock"].nunique()),
        assessment_month_groups=assessment_months,
    )
    passed = bool(
        result.development_complete_episodes >= 220
        and result.development_sessions >= 60
        and result.development_stocks >= 15
        and result.development_months >= 10
        and result.assessment_complete_episodes >= 180
        and result.assessment_sessions >= 45
        and result.assessment_stocks >= 15
        and result.assessment_month_groups == 8
    )
    return CoverageGateResult(
        passed=passed,
        development_complete_episodes=result.development_complete_episodes,
        development_sessions=result.development_sessions,
        development_stocks=result.development_stocks,
        development_months=result.development_months,
        assessment_complete_episodes=result.assessment_complete_episodes,
        assessment_sessions=result.assessment_sessions,
        assessment_stocks=result.assessment_stocks,
        assessment_month_groups=result.assessment_month_groups,
    )


def phase1_decision(
    *,
    lineage_found: bool,
    binding_dependency_class: str | None,
    compatibility_passed: bool,
    causality_passed: bool,
    coverage_passed: bool,
    reproducibility_passed: bool,
) -> str:
    """Apply the prescribed fail-closed Phase 1 precedence."""

    if not lineage_found:
        decision = "blocked_signed_pressure_lineage_not_found"
    elif binding_dependency_class == "D":
        decision = "blocked_dense_pressure_upstream_dependency"
    elif not compatibility_passed:
        decision = "blocked_sparse_pressure_compatibility_failure"
    elif not causality_passed:
        decision = "blocked_dense_pressure_causality_failure"
    elif not coverage_passed:
        decision = "blocked_insufficient_dense_pressure_coverage"
    elif not reproducibility_passed:
        decision = "blocked_dense_pressure_reproducibility_failure"
    else:
        decision = "dense_signed_pressure_reconstruction_supported"
    if decision not in PHASE1_DECISIONS:
        raise AssertionError("invalid Phase 1 decision")
    return decision


def assert_phase2_authorized(phase1_primary_decision: str) -> None:
    """Prevent the frozen directional rerun unless reconstruction is supported."""

    if phase1_primary_decision != "dense_signed_pressure_reconstruction_supported":
        raise RuntimeError(
            "Phase 2 is not authorized because dense signed-pressure reconstruction did not pass"
        )
