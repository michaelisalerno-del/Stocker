"""Pure helpers for the Route-Competition Completion-Hazard Quick Screen V0.

The public surface is retrospective structural research infrastructure only. It
contains no return, direction, execution, broker, order, position, deployment,
or strategy-promotion integration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

CHECKPOINTS: Final[tuple[int, ...]] = (6, 10, 14, 18, 22, 26, 30, 34)
PROTECTED_START: Final[pd.Timestamp] = pd.Timestamp("2025-08-23", tz="UTC")
ROUTE_FEATURES: Final[tuple[str, ...]] = (
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
BASELINE_FEATURES: Final[tuple[str, ...]] = (
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
    *(f"checkpoint_{checkpoint}" for checkpoint in CHECKPOINTS),
)
H1_FEATURES: Final[tuple[str, ...]] = (*BASELINE_FEATURES, *ROUTE_FEATURES)


@dataclass(frozen=True)
class FittedHazardModel:
    """Serializable development-fitted standardisation and binary hazard model."""

    features: tuple[str, ...]
    scaler: StandardScaler
    estimator: LogisticRegression

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, list(self.features)].to_numpy(dtype=float)
        return np.asarray(
            self.estimator.predict_proba(self.scaler.transform(matrix))[:, 1],
            dtype=float,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.features),
            "scaler_mean": self.scaler.mean_.astype(float).tolist(),
            "scaler_scale": self.scaler.scale_.astype(float).tolist(),
            "coefficient": self.estimator.coef_[0].astype(float).tolist(),
            "intercept": float(self.estimator.intercept_[0]),
            "n_iter": int(self.estimator.n_iter_[0]),
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "random_state": 20260722,
            "n_jobs": 1,
        }


def completion_targets(*, checkpoint: int, completion_ordinals: Sequence[int]) -> dict[str, int]:
    """Return strict next-one and next-three registered-completion indicators."""

    future = {int(value) for value in completion_ordinals if int(value) > int(checkpoint)}
    return {
        "registered_completion_next_3_bars": int(
            any(int(checkpoint) < value <= int(checkpoint) + 3 for value in future)
        ),
        "registered_completion_next_1_bar": int(int(checkpoint) + 1 in future),
    }


def reject_protected_dates(frame: pd.DataFrame, *, column: str = "session") -> None:
    """Reject any analytical row on or after the protected boundary."""

    if column not in frame:
        raise ValueError(f"{column} column is required")
    timestamps = pd.to_datetime(frame[column], utc=True, errors="raise")
    if bool(timestamps.ge(PROTECTED_START).any()):
        raise ValueError("protected date 2025-08-23 or later materialised")


def _prefix_depth(progress_states: Any, transitions_remaining: Any) -> float:
    progress = max(int(progress_states), 1)
    remaining = max(int(transitions_remaining), 0)
    required_transitions = progress + remaining - 1
    if required_transitions <= 0:
        return 0.0
    return min(max((progress - 1) / required_transitions, 0.0), 1.0)


def _prefix_snapshot(prefix_ledger: pd.DataFrame, bar_ordinal: int) -> pd.DataFrame:
    required = {
        "bar_ordinal",
        "semantic_loop_id",
        "motif_type",
        "orientation_id",
        "progress_states",
        "transitions_remaining",
    }
    missing = sorted(required.difference(prefix_ledger.columns))
    if missing:
        raise ValueError(f"prefix ledger missing columns: {missing}")

    current = prefix_ledger.loc[
        pd.to_numeric(prefix_ledger["bar_ordinal"], errors="raise").eq(bar_ordinal)
    ].copy()
    if current.empty:
        return current.assign(prefix_depth_fraction=pd.Series(dtype=float))

    current["prefix_depth_fraction"] = [
        _prefix_depth(progress, remaining)
        for progress, remaining in zip(
            current["progress_states"],
            current["transitions_remaining"],
            strict=True,
        )
    ]
    current["semantic_loop_id"] = current["semantic_loop_id"].astype(str)
    current["orientation_id"] = current["orientation_id"].astype("string")
    return (
        current.sort_values("prefix_depth_fraction", ascending=False, kind="stable")
        .drop_duplicates(["semantic_loop_id", "orientation_id"], keep="first")
        .reset_index(drop=True)
    )


def _prefix_keys(snapshot: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(row.semantic_loop_id), str(row.orientation_id))
        for row in snapshot.itertuples(index=False)
    }


def _top_depth(snapshot: pd.DataFrame) -> float:
    if snapshot.empty:
        return 0.0
    return float(snapshot["prefix_depth_fraction"].max())


def _orientation_signature(value: Any) -> str | None:
    """Return a cross-route orientation anchor from a frozen orientation ID."""

    if pd.isna(value):
        return None
    text = str(value)
    if not text:
        return None
    marker = "__o_"
    if marker not in text:
        return text
    path = text.split(marker, maxsplit=1)[1]
    anchor = path.split("-", maxsplit=1)[0]
    return f"anchor_state_{anchor}" if anchor else None


def route_competition_features_from_ledger(
    prefix_ledger: pd.DataFrame,
    registered_completions: pd.DataFrame,
    *,
    checkpoint: int,
) -> dict[str, float]:
    """Construct the fixed causal route bundle at one completed-bar ordinal."""

    completion_columns = {"completion_bar_ordinal", "semantic_loop_id"}
    missing = sorted(completion_columns.difference(registered_completions.columns))
    if missing:
        raise ValueError(f"registered completion ledger missing columns: {missing}")

    current = _prefix_snapshot(prefix_ledger, checkpoint)
    previous_one = _prefix_snapshot(prefix_ledger, checkpoint - 1)
    previous_three = _prefix_snapshot(prefix_ledger, checkpoint - 3)

    depths = sorted((float(value) for value in current["prefix_depth_fraction"]), reverse=True)
    top_depth = depths[0] if depths else 0.0
    second_depth = depths[1] if len(depths) > 1 else 0.0

    family_counts = current["motif_type"].astype(str).value_counts()
    if len(family_counts) < 2:
        family_entropy = 0.0
    else:
        probabilities = family_counts / float(family_counts.sum())
        family_entropy = -sum(float(p) * math.log(float(p)) for p in probabilities)

    orientations = current["orientation_id"].map(_orientation_signature).dropna()
    if len(orientations) < 2 or len(orientations) != len(current):
        orientation_disagreement = 0.0
    else:
        orientation_disagreement = 1.0 - (
            float(orientations.value_counts().max()) / float(len(orientations))
        )

    current_keys = _prefix_keys(current)
    previous_one_keys = _prefix_keys(previous_one)
    current_count = float(len(current))

    completion_ordinals = pd.to_numeric(
        registered_completions["completion_bar_ordinal"], errors="raise"
    )
    causal_completions = registered_completions.loc[completion_ordinals.lt(checkpoint)].copy()
    causal_ordinals = pd.to_numeric(causal_completions["completion_bar_ordinal"], errors="raise")
    recent_identities = set(
        causal_completions.loc[causal_ordinals.ge(checkpoint - 6), "semantic_loop_id"].astype(str)
    )
    older_identities = set(
        causal_completions.loc[
            causal_ordinals.ge(checkpoint - 12) & causal_ordinals.lt(checkpoint - 6),
            "semantic_loop_id",
        ].astype(str)
    )

    matching_recent = float(current["semantic_loop_id"].astype(str).isin(recent_identities).sum())
    weighted_depths = []
    for row in current.itertuples(index=False):
        identity = str(row.semantic_loop_id)
        multiplier = 1.0 + float(identity in recent_identities)
        multiplier += 0.5 * float(identity in older_identities)
        weighted_depths.append(float(cast(Any, row.prefix_depth_fraction)) * multiplier)

    values = {
        "active_prefix_count": current_count,
        "active_prefix_family_count": float(current["motif_type"].nunique()),
        "top_prefix_depth_fraction": top_depth,
        "second_prefix_depth_fraction": second_depth,
        "top_minus_second_prefix_depth": top_depth - second_depth,
        "prefix_family_entropy": float(family_entropy),
        "orientation_disagreement_fraction": float(orientation_disagreement),
        "new_prefixes_last_1_bar": float(len(current_keys - previous_one_keys)),
        "invalidated_prefixes_last_1_bar": float(len(previous_one_keys - current_keys)),
        "active_prefix_count_change_last_1_bar": current_count - float(len(previous_one)),
        "active_prefix_count_change_last_3_bars": current_count - float(len(previous_three)),
        "top_prefix_depth_change_last_1_bar": top_depth - _top_depth(previous_one),
        "top_prefix_depth_change_last_3_bars": top_depth - _top_depth(previous_three),
        "matching_recent_loop_prefix_count": matching_recent,
        "recent_loop_memory_weighted_top_depth": max(weighted_depths, default=0.0),
    }
    return {feature: float(values[feature]) for feature in ROUTE_FEATURES}


def freeze_route_thresholds(
    development: pd.DataFrame,
) -> dict[str, tuple[float, float, float]]:
    """Freeze the three route-state quartile boundaries using development only."""

    columns = (
        "top_prefix_depth_fraction",
        "top_minus_second_prefix_depth",
        "prefix_family_entropy",
    )
    missing = sorted(set(columns).difference(development.columns))
    if missing:
        raise ValueError(f"development route panel missing columns: {missing}")
    result: dict[str, tuple[float, float, float]] = {}
    for column in columns:
        values = pd.to_numeric(development[column], errors="raise")
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"non-finite development route values: {column}")
        quantiles = values.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
        result[column] = tuple(float(value) for value in quantiles)  # type: ignore[assignment]
    return result


def assign_frozen_quartile(
    values: pd.Series,
    boundaries: tuple[float, float, float],
) -> pd.Series:
    """Apply frozen quartiles without inspecting assessment distributions."""

    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("quartile values must be finite")
    bins = np.searchsorted(np.asarray(boundaries, dtype=float), numeric, side="left")
    labels = np.asarray(("Q1", "Q2", "Q3", "Q4"), dtype=object)[bins]
    return pd.Series(labels, index=values.index, dtype="string")


def assign_route_resolution_state(
    frame: pd.DataFrame,
    thresholds: Mapping[str, Sequence[float]],
) -> pd.Series:
    """Assign the five frozen descriptive route-resolution states."""

    required = {
        "active_prefix_count",
        "active_prefix_count_change_last_3_bars",
        "top_prefix_depth_fraction",
        "top_minus_second_prefix_depth",
        "prefix_family_entropy",
        "depth_margin_change_last_3_bars",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"route-resolution panel missing columns: {missing}")
    for column in (
        "top_prefix_depth_fraction",
        "top_minus_second_prefix_depth",
        "prefix_family_entropy",
    ):
        if column not in thresholds or len(thresholds[column]) != 3:
            raise ValueError(f"missing frozen quartiles for {column}")

    labels = pd.Series("OTHER", index=frame.index, dtype="string")
    entropy_high = float(thresholds["prefix_family_entropy"][2])
    margin_low = float(thresholds["top_minus_second_prefix_depth"][0])
    depth_high = float(thresholds["top_prefix_depth_fraction"][2])
    margin_high = float(thresholds["top_minus_second_prefix_depth"][2])

    broad = frame["prefix_family_entropy"].ge(entropy_high) & frame[
        "top_minus_second_prefix_depth"
    ].le(margin_low)
    narrowing = frame["active_prefix_count_change_last_3_bars"].lt(0) & frame[
        "depth_margin_change_last_3_bars"
    ].gt(0)
    dominant = frame["top_prefix_depth_fraction"].ge(depth_high) & frame[
        "top_minus_second_prefix_depth"
    ].ge(margin_high)
    low_support = frame["active_prefix_count"].le(2)

    labels.loc[low_support] = "LOW_ROUTE_SUPPORT"
    labels.loc[dominant] = "DOMINANT_ROUTE"
    labels.loc[narrowing] = "NARROWING"
    labels.loc[broad] = "BROAD_CONFLICT"
    return labels


def session_bootstrap_multiplicities(
    sessions: pd.Series,
    *,
    draws: int,
    seed: int,
) -> list[np.ndarray]:
    """Return whole-session bootstrap multiplicities aligned to input rows."""

    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    labels = sessions.astype(str).to_numpy()
    unique = np.asarray(sorted(set(labels)), dtype=object)
    if unique.size == 0:
        raise ValueError("bootstrap sessions are empty")
    rng = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        result.append(np.asarray([int(counts.get(value, 0)) for value in labels]))
    return result


def permute_route_bundle(
    frame: pd.DataFrame,
    *,
    route_features: Sequence[str],
    strata: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Permute one intact route bundle among stocks inside every causal slate."""

    required = set(route_features).union(strata).union({"symbol"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"route permutation panel missing columns: {missing}")
    if frame.duplicated([*strata, "symbol"]).any():
        raise ValueError("route permutation requires one stock row per slate")

    result = frame.copy()
    rng = np.random.default_rng(seed)
    columns = list(route_features)
    for _, group in frame.groupby(list(strata), sort=True, dropna=False):
        indices = group.index.to_numpy()
        donors = rng.permutation(indices)
        result.loc[indices, columns] = frame.loc[donors, columns].to_numpy()
    return result


def route_increment_passes(gates: Mapping[str, object]) -> bool:
    """Apply the eight binding H1 decision gates without approximation."""

    required = {
        "log_loss_improvement",
        "brier_improvement",
        "auc_improvement",
        "bootstrap_80_log_loss_lower",
        "bootstrap_80_brier_lower",
        "positive_months",
        "materially_adverse_checkpoints",
        "real_exceeds_all_nulls",
        "concentration_passed",
    }
    missing = sorted(required.difference(gates))
    if missing:
        raise ValueError(f"route increment gates missing: {missing}")
    return bool(
        float(cast(Any, gates["log_loss_improvement"])) > 0.0
        and float(cast(Any, gates["brier_improvement"])) > 0.0
        and float(cast(Any, gates["auc_improvement"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_log_loss_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_brier_lower"])) >= 0.0
        and int(cast(Any, gates["positive_months"])) >= 5
        and int(cast(Any, gates["materially_adverse_checkpoints"])) == 0
        and bool(gates["real_exceeds_all_nulls"])
        and bool(gates["concentration_passed"])
    )


def choose_primary_decision(
    *,
    blocker: str | None,
    h1_passed: bool,
    route_narrowing_ordered: bool,
    h0_meaningful: bool,
) -> str:
    """Choose exactly one preregistered primary decision, with blockers first."""

    blockers = {
        "blocked_population_reconstruction_failure",
        "blocked_prefix_reconstruction_failure",
        "blocked_insufficient_support",
        "blocked_protected_boundary_failure",
        "blocked_chronology_or_leakage_failure",
        "blocked_quick_route_competition_resource_limit",
        "blocked_model_convergence_failure",
        "blocked_reproducibility_or_audit_failure",
    }
    if blocker is not None:
        if blocker not in blockers:
            raise ValueError(f"unknown blocker: {blocker}")
        return blocker
    if h1_passed:
        return "route_competition_improves_completion_hazard"
    if route_narrowing_ordered:
        return "descriptive_route_narrowing_only"
    if h0_meaningful:
        return "compressed_transition_baseline_only"
    return "no_route_competition_increment"


def fit_hazard_model(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    target: str = "registered_completion_next_3_bars",
    weight_column: str = "row_weight",
) -> FittedHazardModel:
    """Fit one fixed deterministic weighted binary L2 logistic model."""

    required = {target, weight_column, *features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"hazard model columns missing: {missing}")
    labels = frame[target].to_numpy(dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError("hazard model fitting requires both classes")
    matrix = frame.loc[:, list(features)].to_numpy(dtype=float)
    weights = frame[weight_column].to_numpy(dtype=float)
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(weights).all()
        or bool((weights <= 0.0).any())
    ):
        raise ValueError("hazard model fitting requires finite features and weights")
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    transformed = scaler.fit_transform(matrix)
    estimator = LogisticRegression(
        penalty="l2",
        C=0.25,
        solver="liblinear",
        max_iter=300,
        class_weight=None,
        n_jobs=1,
        random_state=20260722,
    )
    estimator.fit(transformed, labels, sample_weight=weights)
    if bool(np.any(estimator.n_iter_ >= 300)):
        raise ValueError("hazard model did not converge")
    return FittedHazardModel(features=features, scaler=scaler, estimator=estimator)


def reconstruct_hazard_probability(
    frame: pd.DataFrame, specification: Mapping[str, object]
) -> np.ndarray:
    """Manually reconstruct probabilities from a serialized model."""

    feature_values = cast(Sequence[object], specification["feature_names"])
    features = [str(value) for value in feature_values]
    matrix = frame.loc[:, features].to_numpy(dtype=float)
    mean = np.asarray(specification["scaler_mean"], dtype=float)
    scale = np.asarray(specification["scaler_scale"], dtype=float)
    coefficient = np.asarray(specification["coefficient"], dtype=float)
    intercept = float(cast(Any, specification["intercept"]))
    if not (
        matrix.shape[1] == len(mean) == len(scale) == len(coefficient)
        and np.isfinite(matrix).all()
        and np.isfinite(mean).all()
        and np.isfinite(scale).all()
        and np.isfinite(coefficient).all()
        and np.isfinite(intercept)
        and bool((scale > 0.0).all())
    ):
        raise ValueError("serialized hazard model is invalid")
    logits = ((matrix - mean) / scale) @ coefficient + intercept
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(logits, -709.0, 709.0))), dtype=float)


def _calibration_fit(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    if set(labels.astype(int)) != {0, 1}:
        return float("nan"), float("nan")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones(len(labels), dtype=float), logits])
    beta = np.asarray([0.0, 1.0], dtype=float)
    for _ in range(50):
        fitted = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -35.0, 35.0)))
        gradient = design.T @ (weights * (labels - fitted))
        curvature = weights * fitted * (1.0 - fitted)
        information = design.T @ (curvature[:, None] * design)
        information += np.eye(2, dtype=float) * 1e-12
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta += step
        if float(np.max(np.abs(step))) <= 1e-10:
            break
    return float(beta[0]), float(beta[1])


def binary_hazard_metrics(
    labels: Sequence[int] | pd.Series,
    probabilities: Sequence[float] | pd.Series,
    weights: Sequence[float] | pd.Series,
) -> dict[str, float | int]:
    """Calculate the preregistered weighted binary evaluation surface."""

    target = np.asarray(labels, dtype=int)
    prediction = np.asarray(probabilities, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    if not (
        len(target) == len(prediction) == len(sample_weight)
        and len(target) > 0
        and np.isfinite(prediction).all()
        and np.isfinite(sample_weight).all()
        and bool((sample_weight > 0.0).all())
        and bool(((prediction >= 0.0) & (prediction <= 1.0)).all())
    ):
        raise ValueError("hazard metrics require aligned finite inputs")
    total_weight = float(sample_weight.sum())
    brier = float(np.sum(sample_weight * np.square(prediction - target)) / total_weight)
    realised = np.where(target == 1, prediction, 1.0 - prediction)
    base_rate = float(np.sum(sample_weight * target) / total_weight)
    if set(target) == {0, 1}:
        auc = float(roc_auc_score(target, prediction, sample_weight=sample_weight))
        average_precision = float(
            average_precision_score(target, prediction, sample_weight=sample_weight)
        )
    else:
        auc = float("nan")
        average_precision = float("nan")
    intercept, slope = _calibration_fit(target, prediction, sample_weight)
    bin_id = np.minimum((prediction * 10.0).astype(int), 9)
    calibration_error = 0.0
    for value in range(10):
        mask = bin_id == value
        if not bool(mask.any()):
            continue
        bin_weight = sample_weight[mask]
        bin_total = float(bin_weight.sum())
        predicted = float(np.sum(bin_weight * prediction[mask]) / bin_total)
        observed = float(np.sum(bin_weight * target[mask]) / bin_total)
        calibration_error += bin_total / total_weight * abs(predicted - observed)
    return {
        "log_loss": float(log_loss(target, prediction, sample_weight=sample_weight, labels=[0, 1])),
        "brier_score": brier,
        "auc": auc,
        "average_precision": average_precision,
        "expected_calibration_error": float(calibration_error),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "base_rate": base_rate,
        "mean_probability_realised_class": float(np.sum(sample_weight * realised) / total_weight),
        "rows": int(len(target)),
        "positive_outcomes": int(target.sum()),
    }
