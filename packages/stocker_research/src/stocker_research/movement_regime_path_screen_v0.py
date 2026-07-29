"""Reusable primitives for the bounded Movement-Regime-Path V0 screen.

The module deliberately contains only compact topology, fixed linear-model,
chronology, weighting, null-shift, bootstrap, and decision-gate primitives.
It has no broker, execution, order, account, position, or runtime dependency.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

RESEARCH_ONLY = True
FEASIBILITY_SCREEN = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_INTEGRATION_REQUIRED = False
STRATEGY_PROMOTION = False
PRODUCTION_RUNTIME_MODIFIED = False

SAFETY_FLAGS: dict[str, object] = {
    "research_only": RESEARCH_ONLY,
    "feasibility_screen": FEASIBILITY_SCREEN,
    "execution_enabled": EXECUTION_ENABLED,
    "order_placement": ORDER_PLACEMENT,
    "broker_integration_required": BROKER_INTEGRATION_REQUIRED,
    "strategy_promotion": STRATEGY_PROMOTION,
    "production_runtime_modified": PRODUCTION_RUNTIME_MODIFIED,
}

DECISIONS = (
    "promising_probability_chain_for_intensive_v1",
    "movement_predictable_but_no_structural_increment",
    "structural_increment_without_directional_value",
    "no_incremental_probability_chain",
    "blocked_missing_required_frozen_artifacts",
    "blocked_chronology_or_leakage_failure",
    "blocked_insufficient_probability_chain_support",
    "blocked_model_convergence_failure",
    "blocked_reproducibility_or_audit_failure",
)

FORBIDDEN_FEATURE_EXACT = {
    "symbol",
    "symbol_identity",
    "month",
    "future_state",
    "future_path",
    "future_movement",
    "exact_loop_id",
    "excursion_resolution_label",
}
FORBIDDEN_FEATURE_FRAGMENTS = (
    "loop_id",
    "selected_loop",
    "profitable_loop",
    "payoff",
    "future_",
    "outcome",
    "excursion_resolution",
    "model_score",
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def fixed_ordinal_rows(
    frame: pd.DataFrame,
    *,
    ordinals: Sequence[int] = (12, 36),
    ordinal_column: str = "bar_ordinal",
) -> pd.DataFrame:
    """Extract only exact declared ordinals in stable natural order."""

    if tuple(int(value) for value in ordinals) != (12, 36):
        raise ValueError("V0 permits exactly decision ordinals 12 and 36")
    if ordinal_column not in frame:
        raise ValueError(f"missing ordinal column: {ordinal_column}")
    selected = frame.loc[pd.to_numeric(frame[ordinal_column], errors="raise").isin(ordinals)]
    sort_columns = [
        column for column in ("session", ordinal_column, "symbol") if column in selected.columns
    ]
    return selected.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def assert_protected_boundary(
    timestamps: Sequence[object] | pd.Series,
    *,
    cutoff: str | pd.Timestamp = "2025-08-23T00:00:00Z",
) -> None:
    """Fail if any materialised timestamp reaches the protected boundary."""

    values = pd.to_datetime(pd.Series(timestamps), utc=True, errors="raise")
    boundary = pd.Timestamp(cutoff)
    if boundary.tzinfo is None:
        boundary = boundary.tz_localize("UTC")
    if values.ge(boundary).any():
        raise ValueError("protected timestamp materialised")


def assert_allowed_feature_names(feature_names: Sequence[str]) -> None:
    """Reject identity, future, loop-selection, and economic-history fields."""

    invalid = []
    for feature in feature_names:
        lowered = feature.lower()
        if lowered in FORBIDDEN_FEATURE_EXACT or any(
            fragment in lowered for fragment in FORBIDDEN_FEATURE_FRAGMENTS
        ):
            invalid.append(feature)
    if invalid:
        raise ValueError(f"forbidden V0 causal feature(s): {sorted(invalid)}")


def leave_one_out_median(values: Sequence[float]) -> FloatArray:
    """Return the exact median of all other contemporaneous rows."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("leave-one-out median needs at least two finite values")
    output = np.empty(len(array), dtype=np.float64)
    for position in range(len(array)):
        output[position] = float(np.median(np.delete(array, position)))
    return output


@dataclass(frozen=True, slots=True)
class PathTopology:
    """Broad, label-invariant topology of one bounded future hard-state path."""

    transition_count: int
    transition_burst: bool
    short_closure: bool
    first_return_step: int | None
    first_closure_unique_states: int | None


def classify_state_path(
    origin_state: int,
    future_states: Sequence[int],
    *,
    source_gap: bool = False,
    crosses_session: bool = False,
) -> PathTopology:
    """Classify transitions and a broad short closure over exactly 24 bars."""

    if len(future_states) != 24:
        raise ValueError("V0 structural path must contain exactly 24 future states")
    if source_gap or crosses_session:
        return PathTopology(0, False, False, None, None)
    states = [int(origin_state), *(int(value) for value in future_states)]
    transition_count = sum(
        left != right for left, right in zip(states[:-1], states[1:], strict=True)
    )
    departure_step: int | None = None
    return_step: int | None = None
    for step, state in enumerate(states[1:], start=1):
        if departure_step is None and state != int(origin_state):
            departure_step = step
            continue
        if departure_step is not None and state == int(origin_state):
            return_step = step
            break
    unique_count: int | None = None
    if departure_step is not None and return_step is not None:
        unique_count = len(set(states[departure_step - 1 : return_step + 1]))
    closure = bool(
        departure_step is not None
        and return_step is not None
        and transition_count >= 2
        and unique_count is not None
        and unique_count <= 3
    )
    return PathTopology(
        transition_count=transition_count,
        transition_burst=transition_count >= 2,
        short_closure=closure,
        first_return_step=return_step,
        first_closure_unique_states=unique_count,
    )


def movement_thresholds(
    frame: pd.DataFrame,
    *,
    year_column: str = "year",
    ordinal_column: str = "decision_ordinal",
    target_column: str = "absolute_movement_bps",
) -> dict[int, float]:
    """Fit the fixed q75 thresholds on 2024 rows only, separately by clock."""

    training = frame.loc[pd.to_numeric(frame[year_column], errors="raise").eq(2024)]
    if training.empty:
        raise ValueError("movement thresholds require 2024 training rows")
    thresholds: dict[int, float] = {}
    for ordinal in (12, 36):
        values = pd.to_numeric(
            training.loc[training[ordinal_column].eq(ordinal), target_column], errors="coerce"
        ).dropna()
        if values.empty:
            raise ValueError(f"no 2024 movement rows for ordinal {ordinal}")
        thresholds[ordinal] = float(values.quantile(0.75, interpolation="linear"))
    return thresholds


def equal_slate_weights(slate_ids: pd.Series) -> FloatArray:
    """Give every represented slate total weight one."""

    if slate_ids.empty or slate_ids.isna().any():
        raise ValueError("slate weights require non-empty, non-missing slate ids")
    sizes = slate_ids.groupby(slate_ids, sort=True).transform("size").to_numpy(dtype=float)
    weights = np.asarray(1.0 / sizes, dtype=np.float64)
    totals = pd.Series(weights).groupby(slate_ids.reset_index(drop=True), sort=True).sum()
    if not np.allclose(totals.to_numpy(float), 1.0, atol=1e-12):
        raise AssertionError("slate weights do not total one")
    return weights


def bounded_monthly_smoke_population(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Select complete slates round-robin across months for non-scientific smoke runs."""

    if max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    required = {"slate_id", "session"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"smoke population is missing columns: {missing}")
    if frame.empty:
        return frame.copy()

    slates = (
        frame.groupby("slate_id", sort=True, as_index=False)
        .agg(session=("session", "min"), slate_size=("slate_id", "size"))
        .sort_values(["session", "slate_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    slates["month"] = pd.to_datetime(slates["session"], errors="raise").dt.strftime("%Y-%m")
    monthly_slates = {
        str(month): group.reset_index(drop=True)
        for month, group in slates.groupby("month", sort=True)
    }
    minimum_rows = 0
    for group in monthly_slates.values():
        minimum_rows += int(cast(Any, group.loc[0, "slate_size"]))
    if max_rows < minimum_rows:
        raise ValueError(
            "--max-rows must accommodate at least one complete slate per observed month "
            f"({minimum_rows} rows)"
        )

    selected: list[str] = []
    positions = {month: 0 for month in monthly_slates}
    used_rows = 0
    while True:
        made_progress = False
        for month, group in monthly_slates.items():
            position = positions[month]
            if position >= len(group):
                continue
            row = group.iloc[position]
            slate_size = int(row["slate_size"])
            if used_rows + slate_size > max_rows:
                continue
            selected.append(str(row["slate_id"]))
            positions[month] += 1
            used_rows += slate_size
            made_progress = True
        if not made_progress:
            break

    result = frame.loc[frame["slate_id"].astype(str).isin(selected)].copy()
    sort_columns = [
        column for column in ("session", "decision_ordinal", "symbol") if column in result.columns
    ]
    return result.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class FrozenLinearModel:
    """JSON-serializable standardized fixed L2 linear model."""

    model_id: str
    kind: Literal["logistic", "ridge"]
    feature_names: tuple[str, ...]
    means: FloatArray
    scales: FloatArray
    coefficients: FloatArray
    intercept: float
    training_rows: int
    training_slates: int
    iterations: int
    converged: bool
    c: float | None = None
    alpha: float | None = None

    def transform(self, frame: pd.DataFrame) -> FloatArray:
        values = frame.loc[:, list(self.feature_names)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{self.model_id} prediction features contain non-finite values")
        return np.asarray((values - self.means) / self.scales, dtype=np.float64)

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        linear = self.intercept + self.transform(frame) @ self.coefficients
        if self.kind == "ridge":
            return np.asarray(linear, dtype=np.float64)
        clipped = np.clip(linear, -709.0, 709.0)
        return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "training_rows": self.training_rows,
            "training_slates": self.training_slates,
            "iterations": self.iterations,
            "converged": self.converged,
            "C": self.c,
            "alpha": self.alpha,
            "penalty": "l2",
            "solver": "liblinear" if self.kind == "logistic" else "cholesky",
            "max_iter": 250 if self.kind == "logistic" else None,
            "class_weight": None,
            "n_jobs": 1,
        }


def _standardization(frame: pd.DataFrame, features: Sequence[str]) -> tuple[FloatArray, FloatArray]:
    names = tuple(features)
    assert_allowed_feature_names(names)
    values = frame.loc[:, list(names)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("fixed linear model requires complete finite features")
    means = np.asarray(values.mean(axis=0), dtype=np.float64)
    scales = np.asarray(values.std(axis=0, ddof=0), dtype=np.float64)
    scales = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, 1.0)
    return means, scales


def fit_fixed_logistic(
    frame: pd.DataFrame,
    target: Sequence[int] | pd.Series,
    *,
    features: Sequence[str],
    slate_column: str,
    model_id: str,
    random_state: int = 20260720,
) -> FrozenLinearModel:
    """Fit the preregistered deterministic C=1 L2 liblinear model."""

    names = tuple(features)
    means, scales = _standardization(frame, names)
    design = (frame.loc[:, list(names)].to_numpy(dtype=np.float64) - means) / scales
    labels = np.asarray(target, dtype=np.int64)
    if labels.shape != (len(frame),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{model_id} requires aligned binary classes")
    weights = equal_slate_weights(frame[slate_column].astype(str))
    estimator = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=random_state,
        n_jobs=1,
    )
    # The frozen contract explicitly requires parameters deprecated only in newer
    # scikit-learn releases; suppress those API-lifecycle notices, not convergence.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")
        estimator.fit(design, labels, sample_weight=weights)
    iterations = int(np.max(estimator.n_iter_))
    converged = iterations < 250
    if not converged:
        raise RuntimeError(f"{model_id} failed to converge")
    return FrozenLinearModel(
        model_id=model_id,
        kind="logistic",
        feature_names=names,
        means=means,
        scales=scales,
        coefficients=np.asarray(estimator.coef_[0], dtype=np.float64),
        intercept=float(estimator.intercept_[0]),
        training_rows=len(frame),
        training_slates=int(frame[slate_column].nunique()),
        iterations=iterations,
        converged=True,
        c=1.0,
    )


def fit_fixed_ridge(
    frame: pd.DataFrame,
    target: Sequence[float] | pd.Series,
    *,
    features: Sequence[str],
    slate_column: str,
    model_id: str,
) -> FrozenLinearModel:
    """Fit the preregistered deterministic alpha=1 Ridge model."""

    names = tuple(features)
    means, scales = _standardization(frame, names)
    design = (frame.loc[:, list(names)].to_numpy(dtype=np.float64) - means) / scales
    values = np.asarray(target, dtype=np.float64)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError(f"{model_id} requires aligned finite regression targets")
    weights = equal_slate_weights(frame[slate_column].astype(str))
    estimator = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky")
    estimator.fit(design, values, sample_weight=weights)
    return FrozenLinearModel(
        model_id=model_id,
        kind="ridge",
        feature_names=names,
        means=means,
        scales=scales,
        coefficients=np.asarray(estimator.coef_, dtype=np.float64),
        intercept=float(estimator.intercept_),
        training_rows=len(frame),
        training_slates=int(frame[slate_column].nunique()),
        iterations=1,
        converged=True,
        alpha=1.0,
    )


@dataclass(frozen=True, slots=True)
class MonthFold:
    """One expanding one-month 2024 out-of-fold split."""

    fold_id: str
    score_month: str
    training_months: tuple[str, ...]
    train_indices: IntArray
    score_indices: IntArray


def expanding_month_folds(
    frame: pd.DataFrame,
    *,
    session_column: str = "session",
) -> list[MonthFold]:
    """Return Jan-June -> July, then expand monthly through December."""

    sessions = pd.to_datetime(frame[session_column], errors="raise")
    if sessions.dt.tz is not None:
        sessions = sessions.dt.tz_localize(None)
    if not sessions.dt.year.eq(2024).all():
        raise ValueError("OOF folds may contain 2024 rows only")
    months = sessions.dt.to_period("M").astype(str)
    output: list[MonthFold] = []
    for month_number in range(7, 13):
        score_month = f"2024-{month_number:02d}"
        train_mask = months.lt(score_month)
        score_mask = months.eq(score_month)
        train_months = tuple(sorted(months.loc[train_mask].unique()))
        if len(train_months) < 6 or not score_mask.any():
            continue
        train_indices = np.flatnonzero(train_mask.to_numpy()).astype(np.int64)
        score_indices = np.flatnonzero(score_mask.to_numpy()).astype(np.int64)
        if sessions.iloc[train_indices].max() >= sessions.iloc[score_indices].min():
            raise AssertionError("chronological fold overlaps")
        output.append(
            MonthFold(
                fold_id=f"fold_{month_number - 6:02d}",
                score_month=score_month,
                training_months=train_months,
                train_indices=train_indices,
                score_indices=score_indices,
            )
        )
    return output


def assert_stacking_chronology(
    frame: pd.DataFrame,
    *,
    prediction_columns: Sequence[str],
    session_column: str = "session",
) -> None:
    """Prove each populated upstream prediction was trained strictly earlier."""

    score_dates = pd.to_datetime(frame[session_column], errors="raise")
    for prediction in prediction_columns:
        provenance = f"{prediction}__trained_through"
        if provenance not in frame:
            raise AssertionError(f"missing OOF provenance column {provenance}")
        populated = frame[prediction].notna()
        trained_through = pd.to_datetime(frame.loc[populated, provenance], errors="raise")
        if trained_through.ge(score_dates.loc[populated]).any():
            raise AssertionError(f"in-sample stacked feature detected: {prediction}")


def probability_chain(
    p_move: Sequence[float],
    p_up_given_move: Sequence[float],
    predicted_absolute_movement_bps: Sequence[float],
) -> dict[str, FloatArray]:
    """Compose hurdle probabilities and the signed ranking proxy."""

    move = np.asarray(p_move, dtype=np.float64)
    up = np.asarray(p_up_given_move, dtype=np.float64)
    size = np.asarray(predicted_absolute_movement_bps, dtype=np.float64)
    if move.shape != up.shape or move.shape != size.shape:
        raise ValueError("probability-chain vectors must align")
    if np.any((move < 0.0) | (move > 1.0) | (up < 0.0) | (up > 1.0)):
        raise ValueError("probability-chain inputs must be probabilities")
    p_long = move * up
    p_short = move * (1.0 - up)
    p_neutral = 1.0 - move
    return {
        "p_long": np.asarray(p_long, dtype=np.float64),
        "p_short": np.asarray(p_short, dtype=np.float64),
        "p_neutral": np.asarray(p_neutral, dtype=np.float64),
        "score": np.asarray(move * (2.0 * up - 1.0) * size, dtype=np.float64),
    }


def circular_shift_session_blocks(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    draw: int,
    seed: int,
    session_column: str = "session",
    ordinal_column: str = "decision_ordinal",
    symbol_column: str = "symbol",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Circularly shift whole same-ordinal cross-sections with identical membership."""

    output = frame.copy()
    rng = np.random.default_rng(seed + draw)
    manifest: list[dict[str, Any]] = []
    group_keys: dict[tuple[int, tuple[str, ...]], list[str]] = {}
    for (ordinal, session), slate in frame.groupby(
        [ordinal_column, session_column], sort=True, observed=True
    ):
        membership = tuple(sorted(slate[symbol_column].astype(str)))
        group_keys.setdefault((int(str(ordinal)), membership), []).append(str(session))
    for (ordinal, membership), raw_sessions in sorted(group_keys.items()):
        sessions = sorted(set(raw_sessions))
        offset = 0 if len(sessions) <= 1 else int(rng.integers(1, len(sessions)))
        for destination_position, destination in enumerate(sessions):
            source = sessions[(destination_position - offset) % len(sessions)]
            destination_index = frame.index[
                frame[ordinal_column].eq(ordinal)
                & frame[session_column].astype(str).eq(destination)
            ]
            source_block = frame.loc[
                frame[ordinal_column].eq(ordinal) & frame[session_column].astype(str).eq(source),
                [symbol_column, *value_columns],
            ].copy()
            source_block = source_block.set_index(symbol_column).loc[list(membership)]
            destination_symbols = frame.loc[destination_index, symbol_column].astype(str)
            values = source_block.loc[destination_symbols, list(value_columns)].to_numpy()
            output.loc[destination_index, list(value_columns)] = values
            manifest.append(
                {
                    "draw": draw,
                    "decision_ordinal": ordinal,
                    "destination_session": destination,
                    "source_session": source,
                    "offset": offset,
                    "membership_size": len(membership),
                }
            )
    return output, manifest


def sampled_sessions(
    sessions: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> list[list[str]]:
    """Return fixed-seed whole-session bootstrap samples."""

    unique = np.asarray(sorted(set(str(value) for value in sessions)), dtype=object)
    if len(unique) == 0:
        raise ValueError("bootstrap requires at least one session")
    rng = np.random.default_rng(seed)
    return [
        cast(list[str], rng.choice(unique, size=len(unique), replace=True).astype(str).tolist())
        for _ in range(draws)
    ]


def _both_improve(evidence: dict[str, Any], prefix: str) -> bool:
    return bool(
        float(evidence[f"{prefix}_brier_improvement"]) > 0.0
        and float(evidence[f"{prefix}_log_loss_improvement"]) > 0.0
    )


def decide_screen(evidence: dict[str, Any]) -> str:
    """Apply the exact fail-closed V0 decision ladder."""

    blocker = evidence.get("blocker")
    if blocker:
        if blocker not in DECISIONS or not str(blocker).startswith("blocked_"):
            raise ValueError(f"unknown blocker decision: {blocker}")
        return str(blocker)
    movement = _both_improve(evidence, "p1_minus_p0")
    structural = _both_improve(evidence, "b1_minus_b0")
    directional = _both_improve(evidence, "d1_minus_d0")
    promising = bool(
        movement
        and structural
        and directional
        and int(evidence["b1_positive_months"]) >= 5
        and int(evidence["d1_positive_months"]) >= 5
        and float(evidence["b1_bootstrap_90_lower"]) >= 0.0
        and float(evidence["d1_bootstrap_90_lower"]) >= 0.0
        and float(evidence["b1_null_percentile"]) >= 0.90
        and float(evidence["d1_null_percentile"]) >= 0.90
        and float(evidence["path_spearman"]) > float(evidence["observable_spearman"])
        and float(evidence["path_top_one_minus_median"])
        > float(evidence["observable_top_one_minus_median"])
        and bool(evidence["concentration_passed"])
        and bool(evidence["exact_rerun_passed"])
        and bool(evidence["independent_audit_passed"])
    )
    if promising:
        return "promising_probability_chain_for_intensive_v1"
    if structural and not directional:
        return "structural_increment_without_directional_value"
    if movement and not structural:
        return "movement_predictable_but_no_structural_increment"
    return "no_incremental_probability_chain"


__all__ = [
    "DECISIONS",
    "SAFETY_FLAGS",
    "FrozenLinearModel",
    "MonthFold",
    "PathTopology",
    "assert_allowed_feature_names",
    "assert_protected_boundary",
    "assert_stacking_chronology",
    "circular_shift_session_blocks",
    "classify_state_path",
    "decide_screen",
    "equal_slate_weights",
    "expanding_month_folds",
    "fit_fixed_logistic",
    "fit_fixed_ridge",
    "fixed_ordinal_rows",
    "leave_one_out_median",
    "movement_thresholds",
    "probability_chain",
    "sampled_sessions",
]
