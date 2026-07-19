"""Causal structural forecasts for cluster-invariant excursion resolution.

This module deliberately has no price-payoff, execution, broker, order, or
strategy dependency.  Its targets are structural event families and their
arrival times inside an already frozen excursion-event lineage.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SAFETY_FLAGS: Final[dict[str, object]] = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "economic_outcomes_used": False,
    "payoff_selection_used": False,
    "production_runtime_modified": False,
    "strategy_promotion": False,
}

TARGET_CLASSES: Final[tuple[str, ...]] = (
    "RETURN_TO_ORIGIN",
    "PARTIAL_RETURN",
    "CONTINUE_AWAY",
    "ROTATE_TO_NEW_REGION",
    "SESSION_END",
    "UNAVAILABLE",
)

UNAVAILABLE_FAMILIES: Final[frozenset[str]] = frozenset(
    {"UNAVAILABLE_SOURCE", "UNAVAILABLE_STRUCTURAL_GAP"}
)


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(cast(Any, value))


def canonical_target_family(value: object) -> str | None:
    """Map a Part A resolution to the frozen Part B target or censoring."""

    family = str(value)
    if family in TARGET_CLASSES:
        return family
    if family in UNAVAILABLE_FAMILIES:
        return "UNAVAILABLE"
    if family == "UNRESOLVED_AT_HORIZON":
        return None
    raise ValueError(f"unsupported Part A event family: {family}")


def _json_vector(value: object, *, expected: int) -> np.ndarray:
    parsed = np.asarray(json.loads(str(value)), dtype=np.float64)
    if parsed.shape != (expected,):
        raise ValueError(f"expected vector of length {expected}, got {parsed.shape}")
    return parsed


def _mahalanobis(vector: np.ndarray, origin: np.ndarray, precision: np.ndarray) -> float:
    delta = vector - origin
    value = float(delta @ precision @ delta)
    return math.sqrt(max(value, 0.0))


def _cosine_angle(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(math.acos(cosine))


def _directional_consistency(path: Sequence[np.ndarray]) -> float:
    if len(path) < 2:
        return 0.0
    increments = [path[index] - path[index - 1] for index in range(1, len(path))]
    length = float(sum(np.linalg.norm(value) for value in increments))
    if length <= 0.0:
        return 0.0
    return float(np.linalg.norm(path[-1] - path[0]) / length)


def _hard_state_persistence(frame: pd.DataFrame) -> pd.Series:
    output = pd.Series(index=frame.index, dtype="int64")
    keys = ["symbol", "session", "segment_id", "model_lineage"]
    for _, group in frame.groupby(keys, sort=False, dropna=False):
        ordered = group.sort_values("bar_ordinal", kind="stable")
        states = ordered["hard_map_state"].to_numpy()
        values = np.ones(len(ordered), dtype=np.int64)
        for index in range(1, len(values)):
            if states[index] == states[index - 1]:
                values[index] = values[index - 1] + 1
        output.loc[ordered.index] = values
    return output.astype("int64")


def _prior_event_context(events: pd.DataFrame) -> dict[str, dict[str, float]]:
    context: dict[str, dict[str, float]] = {}
    for _, session_events in events.groupby(["symbol", "session"], sort=False):
        ordered = session_events.sort_values(
            ["confirmation_bar_ordinal", "event_id"], kind="stable"
        )
        prior: list[Any] = []
        consecutive_returns = 0
        for event in ordered.itertuples(index=False):
            eligible = [
                value
                for value in prior
                if _as_int(value.resolution_bar_ordinal) <= _as_int(event.confirmation_bar_ordinal)
            ]
            latest = eligible[-1] if eligible else None
            previous = canonical_target_family(latest.event_family) if latest is not None else None
            context[str(event.event_id)] = {
                "previous_is_return": float(previous == "RETURN_TO_ORIGIN"),
                "previous_is_partial": float(previous == "PARTIAL_RETURN"),
                "previous_is_continue": float(previous == "CONTINUE_AWAY"),
                "previous_is_rotate": float(previous == "ROTATE_TO_NEW_REGION"),
                "previous_is_session_end": float(previous == "SESSION_END"),
                "previous_is_unavailable": float(previous == "UNAVAILABLE"),
                "bars_since_previous_excursion": (
                    float(
                        _as_int(event.confirmation_bar_ordinal)
                        - _as_int(latest.resolution_bar_ordinal)
                    )
                    if latest is not None
                    else np.nan
                ),
                "consecutive_return_count": float(consecutive_returns),
                "session_event_count_so_far": float(len(prior)),
            }
            resolved = canonical_target_family(event.event_family)
            if resolved == "RETURN_TO_ORIGIN":
                consecutive_returns += 1
            elif resolved is not None:
                consecutive_returns = 0
            prior.append(event)
    return context


def build_active_excursion_rows(
    events: pd.DataFrame,
    emission: pd.DataFrame,
    posterior: pd.DataFrame,
    *,
    precision: np.ndarray,
    emission_features: Sequence[str],
    primary_model_lineage: str,
    scheduled_session_bars: int = 78,
) -> pd.DataFrame:
    """Build one causal forecast row per completed active-excursion bar.

    Rows start one completed bar after confirmation and stop strictly before
    the frozen Part A resolution.  Unresolved-at-horizon events remain in the
    timing population as right-censored observations and have no class target.
    """

    z_columns = [f"z__{name}" for name in emission_features]
    missing_emission = [column for column in z_columns if column not in emission]
    if missing_emission:
        raise KeyError(f"missing emission columns: {missing_emission}")
    if precision.shape != (len(z_columns), len(z_columns)):
        raise ValueError("precision matrix does not match emission feature count")

    posterior_primary = posterior.loc[posterior["model_lineage"].eq(primary_model_lineage)].copy()
    posterior_primary["hard_state_persistence_bars"] = _hard_state_persistence(posterior_primary)
    posterior_columns = [
        "decision_id",
        "posterior_entropy",
        "expected_state_age",
        "departure_probability",
        "hard_hysteretic_disagreement",
        "posterior_velocity",
        "hard_state_persistence_bars",
        "availability_timestamp",
    ] + [
        str(column)
        for column in posterior_primary.columns
        if str(column).startswith("posterior_state_")
    ]
    posterior_small = posterior_primary[posterior_columns].copy()
    posterior_state_columns = [
        str(column) for column in posterior_small if str(column).startswith("posterior_state_")
    ]
    posterior_small["posterior_max_probability"] = posterior_small[posterior_state_columns].max(
        axis=1
    )
    posterior_small = posterior_small.drop(columns=posterior_state_columns)
    posterior_small = posterior_small.rename(
        columns={"availability_timestamp": "posterior_availability_timestamp"}
    )

    merged = emission.merge(
        posterior_small,
        how="left",
        on="decision_id",
        validate="one_to_one",
    )
    merged = merged.sort_values(["symbol", "session", "segment_id", "bar_ordinal"], kind="stable")
    grouped = {
        tuple(str(value) for value in key): group.reset_index(drop=True)
        for key, group in merged.groupby(
            ["symbol", "session", "segment_id"], sort=False, dropna=False
        )
    }
    history = _prior_event_context(events)
    records: list[dict[str, Any]] = []
    market_names = (
        "regime_log_market_dispersion",
        "regime_stock_minus_market_scaled",
        "vti__signed_efficiency_12",
        "regime_market_breadth_centered",
    )
    market_z = [f"z__{name}" for name in market_names]
    market_delta = [f"delta_z__{name}" for name in market_names]

    for event in events.sort_values(["period", "event_id"], kind="stable").itertuples(index=False):
        key = (str(event.symbol), str(event.session), str(event.segment_id))
        segment = grouped.get(key)
        if segment is None:
            raise ValueError(f"missing trajectory segment for event {event.event_id}")
        confirmation = _as_int(event.confirmation_bar_ordinal)
        resolution = _as_int(event.resolution_bar_ordinal)
        active = segment.loc[
            segment["bar_ordinal"].gt(confirmation) & segment["bar_ordinal"].lt(resolution)
        ].copy()
        if active.empty:
            continue
        causal_path = segment.loc[
            segment["bar_ordinal"].ge(confirmation) & segment["bar_ordinal"].lt(resolution)
        ].copy()
        causal_path = causal_path.sort_values("bar_ordinal", kind="stable")
        vectors = causal_path[z_columns].to_numpy(dtype=np.float64)
        ordinals = causal_path["bar_ordinal"].to_numpy(dtype=np.int64)
        origin = _json_vector(event.frozen_origin_vector, expected=len(z_columns))
        direction = _json_vector(event.departure_direction_vector, expected=len(z_columns))
        distances = np.asarray(
            [_mahalanobis(vector, origin, precision) for vector in vectors],
            dtype=np.float64,
        )
        target = canonical_target_family(event.event_family)
        event_history = history[str(event.event_id)]
        for path_index in range(1, len(causal_path)):
            row = causal_path.iloc[path_index]
            ordinal = int(ordinals[path_index])
            if ordinal <= confirmation:
                continue
            prefix_vectors = [vectors[index] for index in range(path_index + 1)]
            prefix_distances = distances[: path_index + 1]
            current_distance = float(prefix_distances[-1])
            maximum_distance = float(np.max(prefix_distances))
            velocity = float(prefix_distances[-1] - prefix_distances[-2])
            acceleration = (
                float(prefix_distances[-1] - 2.0 * prefix_distances[-2] + prefix_distances[-3])
                if len(prefix_distances) >= 3
                else 0.0
            )
            increments = [
                prefix_vectors[index] - prefix_vectors[index - 1]
                for index in range(1, len(prefix_vectors))
            ]
            path_length = float(sum(np.linalg.norm(value) for value in increments))
            curvature = (
                _cosine_angle(increments[-2], increments[-1]) if len(increments) >= 2 else 0.0
            )
            decision_timestamp = pd.Timestamp(row["decision_timestamp"])
            emission_available = pd.Timestamp(row["availability_timestamp"])
            posterior_available_raw = row.get("posterior_availability_timestamp", pd.NaT)
            posterior_available = pd.Timestamp(posterior_available_raw)
            if pd.isna(posterior_available):
                feature_available = emission_available
            else:
                feature_available = max(emission_available, posterior_available)
            session_fraction = float(np.clip(ordinal / (scheduled_session_bars - 1), 0, 1))
            if session_fraction < 1.0 / 3.0:
                clock_phase = "EARLY"
            elif session_fraction < 2.0 / 3.0:
                clock_phase = "MIDDLE"
            else:
                clock_phase = "LATE"
            if velocity < -1e-12:
                trend = "INWARD"
            elif velocity > 1e-12:
                trend = "OUTWARD"
            else:
                trend = "FLAT"
            departure_distance = _as_float(event.departure_distance)
            distance_ratio = current_distance / max(departure_distance, 1e-12)
            if distance_ratio < 1.0:
                distance_bucket = 0
            elif distance_ratio < 1.5:
                distance_bucket = 1
            else:
                distance_bucket = 2
            record: dict[str, Any] = {
                "run_id": str(event.run_id),
                "git_sha": str(event.git_sha),
                "contract_hash": "pending_part_b_contract",
                "data_snapshot_hash": str(event.period_data_snapshot_hash),
                "panel_hash": str(event.panel_hash),
                "feature_manifest_hash": "pending_forecast_feature_manifest",
                "trajectory_representation": "E",
                "model_lineage": primary_model_lineage,
                "event_definition_hash": str(event.event_definition_hash),
                "decision_id": str(row["decision_id"]),
                "event_id": str(event.event_id),
                "symbol": str(event.symbol),
                "session": str(event.session),
                "segment_id": str(event.segment_id),
                "decision_timestamp": decision_timestamp,
                "onset_timestamp": _as_timestamp(event.onset_timestamp),
                "resolution_timestamp": _as_timestamp(event.resolution_timestamp),
                "event_family": str(event.event_family),
                "period": str(event.period),
                "target_family": target,
                "target_observed": target is not None,
                "right_censored": target is None,
                "bars_until_resolution": resolution - ordinal,
                "resolution_within_3_bars": bool(resolution - ordinal <= 3),
                "resolution_within_6_bars": bool(resolution - ordinal <= 6),
                "resolution_within_12_bars": bool(resolution - ordinal <= 12),
                "source_artifact": "emission_trajectory_ledger.parquet",
                "source_hash": str(event.source_hash),
                "feature_available_timestamp": feature_available,
                "bar_ordinal": ordinal,
                "confirmation_bar_ordinal": confirmation,
                "resolution_bar_ordinal": resolution,
                "current_distance": current_distance,
                "maximum_distance_so_far": maximum_distance,
                "distance_velocity": velocity,
                "distance_acceleration": acceleration,
                "retracement_fraction_so_far": (
                    max(maximum_distance - current_distance, 0.0) / max(maximum_distance, 1e-12)
                ),
                "path_length_since_departure": path_length,
                "directional_consistency_since_departure": _directional_consistency(prefix_vectors),
                "angle_to_departure_vector": _cosine_angle(prefix_vectors[-1] - origin, direction),
                "local_curvature": curvature,
                "bars_since_departure": ordinal - confirmation,
                "bars_remaining_in_session": max(scheduled_session_bars - 1 - ordinal, 0),
                "short_trajectory_velocity": float(row["short_trajectory_velocity"]),
                "short_trajectory_acceleration": float(row["short_trajectory_acceleration"]),
                "local_path_length": float(row["local_path_length"]),
                "local_directional_consistency": float(row["local_directional_consistency"]),
                "posterior_entropy": float(row["posterior_entropy"]),
                "expected_state_age": float(row["expected_state_age"]),
                "departure_probability": float(row["departure_probability"]),
                "hard_hysteretic_disagreement": float(row["hard_hysteretic_disagreement"]),
                "posterior_velocity": float(row["posterior_velocity"]),
                "posterior_max_probability": float(row["posterior_max_probability"]),
                "hard_state_persistence_bars": float(row["hard_state_persistence_bars"]),
                "market_dispersion_z": float(row[market_z[0]]),
                "stock_minus_market_z": float(row[market_z[1]]),
                "market_efficiency_z": float(row[market_z[2]]),
                "market_breadth_z": float(row[market_z[3]]),
                "market_trajectory_velocity": float(
                    np.linalg.norm(row[market_delta].to_numpy(dtype=np.float64))
                ),
                **event_history,
                "clock_phase": clock_phase,
                "clock_phase_early": float(clock_phase == "EARLY"),
                "clock_phase_middle": float(clock_phase == "MIDDLE"),
                "clock_phase_late": float(clock_phase == "LATE"),
                "session_fraction": session_fraction,
                "distance_bucket": distance_bucket,
                "distance_trend": trend,
                "distance_trend_inward": float(trend == "INWARD"),
                "distance_trend_flat": float(trend == "FLAT"),
                "distance_trend_outward": float(trend == "OUTWARD"),
            }
            for feature in emission_features:
                record[f"missing__{feature}"] = bool(row[f"missing__{feature}"])
            records.append(record)
    output = pd.DataFrame.from_records(records)
    if output.empty:
        return output
    if not output["feature_available_timestamp"].le(output["decision_timestamp"]).all():
        raise ValueError("forecast feature availability exceeds decision time")
    if not output["bar_ordinal"].gt(output["confirmation_bar_ordinal"]).all():
        raise ValueError("forecast rows must begin after departure confirmation")
    if not output["bar_ordinal"].lt(output["resolution_bar_ordinal"]).all():
        raise ValueError("forecast rows must end before resolution")
    if output.duplicated(["event_id", "decision_id"]).any():
        raise ValueError("duplicate active-excursion forecast row")
    return output.sort_values(
        ["period", "decision_timestamp", "symbol", "event_id"], kind="stable"
    ).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class FoldPreprocessor:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> FoldPreprocessor:
        matrix = np.asarray(values, dtype=np.float64)
        medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(matrix), matrix, medians)
        means = np.mean(filled, axis=0)
        scales = np.std(filled, axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        return cls(medians=medians, means=means, scales=scales)

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        filled = np.where(np.isfinite(matrix), matrix, self.medians)
        return np.asarray((filled - self.means) / self.scales, dtype=np.float64)

    @property
    def hash(self) -> str:
        digest = hashlib.sha256()
        for values in (self.medians, self.means, self.scales):
            digest.update(np.asarray(values, dtype="<f8").tobytes())
        return digest.hexdigest()


@dataclass(slots=True)
class MultinomialEstimator:
    classes: tuple[str, ...]
    features: tuple[str, ...]
    preprocessor: FoldPreprocessor
    estimator: LogisticRegression | None
    fallback_probabilities: np.ndarray
    regularization_c: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(
            frame.loc[:, list(self.features)].to_numpy(dtype=np.float64)
        )
        if self.estimator is None:
            return np.repeat(self.fallback_probabilities[None, :], len(frame), axis=0)
        raw = self.estimator.predict_proba(matrix)
        expanded = np.full((len(frame), len(self.classes)), 1e-12, dtype=np.float64)
        indices = {value: index for index, value in enumerate(self.classes)}
        for source_index, value in enumerate(self.estimator.classes_):
            expanded[:, indices[str(value)]] = raw[:, source_index]
        expanded /= expanded.sum(axis=1, keepdims=True)
        return expanded

    @property
    def coefficient_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.preprocessor.hash.encode("ascii"))
        if self.estimator is not None:
            digest.update(np.asarray(self.estimator.coef_, dtype="<f8").tobytes())
            digest.update(np.asarray(self.estimator.intercept_, dtype="<f8").tobytes())
            digest.update("\x1f".join(map(str, self.estimator.classes_)).encode("utf-8"))
        else:
            digest.update(self.fallback_probabilities.astype("<f8").tobytes())
        return digest.hexdigest()


def fit_multinomial(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    target_column: str,
    classes: Sequence[str] = TARGET_CLASSES,
    regularization_c: float = 1.0,
    sample_weight: np.ndarray | None = None,
    maximum_iterations: int = 1000,
) -> MultinomialEstimator:
    ordered_classes = tuple(classes)
    selected_features = tuple(features)
    matrix = frame.loc[:, list(selected_features)].to_numpy(dtype=np.float64)
    preprocessor = FoldPreprocessor.fit(matrix)
    target = frame[target_column].astype(str).to_numpy()
    counts = np.asarray([(target == value).sum() for value in ordered_classes], dtype=float)
    fallback = (counts + 1.0) / (counts.sum() + len(ordered_classes))
    unique = np.unique(target)
    estimator: LogisticRegression | None = None
    if len(unique) >= 2:
        estimator = LogisticRegression(
            C=float(regularization_c),
            max_iter=int(maximum_iterations),
            solver="lbfgs",
            random_state=0,
        )
        estimator.fit(preprocessor.transform(matrix), target, sample_weight=sample_weight)
    return MultinomialEstimator(
        classes=ordered_classes,
        features=selected_features,
        preprocessor=preprocessor,
        estimator=estimator,
        fallback_probabilities=fallback,
        regularization_c=float(regularization_c),
    )


def balanced_event_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_id", sort=False)["event_id"].transform("size")
    return 1.0 / counts.to_numpy(dtype=np.float64)


def frequency_probabilities(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    target_column: str,
    classes: Sequence[str] = TARGET_CLASSES,
    group_columns: Sequence[str] = (),
    alpha: float = 1.0,
) -> np.ndarray:
    ordered_classes = tuple(classes)
    global_counts = train[target_column].value_counts()
    global_probability = np.asarray(
        [float(global_counts.get(value, 0)) + alpha for value in ordered_classes],
        dtype=np.float64,
    )
    global_probability /= global_probability.sum()
    if not group_columns:
        return np.repeat(global_probability[None, :], len(predict), axis=0)
    table: dict[tuple[object, ...], np.ndarray] = {}
    grouper: str | list[str] = (
        str(group_columns[0]) if len(group_columns) == 1 else list(group_columns)
    )
    for key, group in train.groupby(grouper, sort=True, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        counts = group[target_column].value_counts()
        probability = np.asarray(
            [float(counts.get(value, 0)) + alpha for value in ordered_classes],
            dtype=np.float64,
        )
        probability /= probability.sum()
        table[tuple(values)] = probability
    rows = []
    for values in predict.loc[:, list(group_columns)].itertuples(index=False, name=None):
        rows.append(table.get(tuple(values), global_probability))
    return np.asarray(rows, dtype=np.float64)


def validate_probabilities(probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("probability matrix must be two-dimensional")
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("class probabilities must sum to one")


def multiclass_losses(
    targets: Sequence[str],
    probabilities: np.ndarray,
    *,
    classes: Sequence[str] = TARGET_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    validate_probabilities(probabilities)
    index = {value: offset for offset, value in enumerate(classes)}
    encoded = np.asarray([index[str(value)] for value in targets], dtype=np.int64)
    clipped = np.clip(probabilities[np.arange(len(encoded)), encoded], 1e-15, 1.0)
    log_loss = -np.log(clipped)
    truth = np.zeros_like(probabilities)
    truth[np.arange(len(encoded)), encoded] = 1.0
    brier = np.sum((probabilities - truth) ** 2, axis=1)
    return log_loss, brier


def ranking_metrics(
    targets: Sequence[str],
    probabilities: np.ndarray,
    *,
    classes: Sequence[str] = TARGET_CLASSES,
) -> dict[str, float]:
    validate_probabilities(probabilities)
    index = {value: offset for offset, value in enumerate(classes)}
    encoded = np.asarray([index[str(value)] for value in targets], dtype=np.int64)
    ordering = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.empty(len(encoded), dtype=np.int64)
    for row_index, target_index in enumerate(encoded):
        ranks[row_index] = int(np.flatnonzero(ordering[row_index] == target_index)[0]) + 1
    return {
        "top_one_accuracy": float(np.mean(ranks == 1)),
        "top_two_hit_rate": float(np.mean(ranks <= 2)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
    }


def calibration_table(
    targets: Sequence[str],
    probabilities: np.ndarray,
    *,
    classes: Sequence[str] = TARGET_CLASSES,
    bins: int = 10,
    minimum_bin_count: int = 25,
) -> tuple[pd.DataFrame, float, float]:
    validate_probabilities(probabilities)
    target_array = np.asarray([str(value) for value in targets], dtype=object)
    records: list[dict[str, Any]] = []
    weighted_error = 0.0
    total = 0
    maximum_supported = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for class_index, class_name in enumerate(classes):
        predicted = probabilities[:, class_index]
        observed = target_array == class_name
        assignments = np.minimum(np.searchsorted(edges, predicted, side="right") - 1, bins - 1)
        assignments = np.maximum(assignments, 0)
        for bin_index in range(bins):
            mask = assignments == bin_index
            count = int(mask.sum())
            if count == 0:
                continue
            mean_probability = float(np.mean(predicted[mask]))
            observed_rate = float(np.mean(observed[mask]))
            error = abs(mean_probability - observed_rate)
            weighted_error += error * count
            total += count
            if count >= minimum_bin_count:
                maximum_supported = max(maximum_supported, error)
            records.append(
                {
                    "event_family": class_name,
                    "bin": bin_index,
                    "lower": float(edges[bin_index]),
                    "upper": float(edges[bin_index + 1]),
                    "count": count,
                    "mean_probability": mean_probability,
                    "observed_rate": observed_rate,
                    "absolute_error": error,
                    "supported": count >= minimum_bin_count,
                }
            )
    ece = weighted_error / max(total, 1)
    return pd.DataFrame.from_records(records), float(ece), float(maximum_supported)


def per_class_metrics(
    targets: Sequence[str],
    probabilities: np.ndarray,
    *,
    classes: Sequence[str] = TARGET_CLASSES,
) -> pd.DataFrame:
    validate_probabilities(probabilities)
    target_array = np.asarray([str(value) for value in targets], dtype=object)
    predicted = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
    records = []
    for class_name in classes:
        truth = target_array == class_name
        chosen = predicted == class_name
        true_positive = int(np.sum(truth & chosen))
        records.append(
            {
                "event_family": class_name,
                "support": int(truth.sum()),
                "precision": true_positive / max(int(chosen.sum()), 1),
                "recall": true_positive / max(int(truth.sum()), 1),
            }
        )
    return pd.DataFrame.from_records(records)


def confusion_matrix_frame(
    targets: Sequence[str],
    probabilities: np.ndarray,
    *,
    classes: Sequence[str] = TARGET_CLASSES,
) -> pd.DataFrame:
    target_array = np.asarray([str(value) for value in targets], dtype=object)
    predicted = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
    rows = []
    for actual in classes:
        for chosen in classes:
            rows.append(
                {
                    "actual_family": actual,
                    "predicted_family": chosen,
                    "count": int(np.sum((target_array == actual) & (predicted == chosen))),
                }
            )
    return pd.DataFrame.from_records(rows)


def constant_hazard_competing_risk(
    next_bar_probabilities: np.ndarray,
    *,
    no_event_index: int,
    horizons: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert next-bar competing probabilities to cumulative incidence.

    Current causal features are held fixed; no future feature path is used.
    The returned incidence is monotonic in each requested horizon and the
    incidence sum plus survival is exactly one.
    """

    validate_probabilities(next_bar_probabilities)
    if not 0 <= no_event_index < next_bar_probabilities.shape[1]:
        raise ValueError("invalid no-event column")
    event_columns = [
        index for index in range(next_bar_probabilities.shape[1]) if index != no_event_index
    ]
    no_event = next_bar_probabilities[:, no_event_index]
    hazards = next_bar_probabilities[:, event_columns]
    total_hazard = hazards.sum(axis=1)
    shares = np.divide(
        hazards,
        total_hazard[:, None],
        out=np.zeros_like(hazards),
        where=total_hazard[:, None] > 0.0,
    )
    incidence = np.empty((len(next_bar_probabilities), len(horizons), len(event_columns)))
    survival = np.empty((len(next_bar_probabilities), len(horizons)))
    for horizon_index, horizon in enumerate(horizons):
        if horizon <= 0:
            raise ValueError("horizons must be positive")
        survival[:, horizon_index] = np.power(no_event, int(horizon))
        incidence[:, horizon_index, :] = shares * (1.0 - survival[:, horizon_index, None])
    return incidence, survival


def timing_metrics(
    frame: pd.DataFrame,
    cumulative_resolution: Mapping[int, np.ndarray],
    predicted_median_bars: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    records = []
    brier_values = []
    for horizon, predicted in sorted(cumulative_resolution.items()):
        observed = (
            frame["target_observed"].to_numpy(dtype=bool)
            & frame["bars_until_resolution"].le(horizon).to_numpy(dtype=bool)
        ).astype(float)
        brier = (np.asarray(predicted, dtype=float) - observed) ** 2
        brier_values.append(float(np.mean(brier)))
        records.append(
            {
                "horizon_bars": int(horizon),
                "integrated_component_brier": float(np.mean(brier)),
                "predicted_cumulative_incidence": float(np.mean(predicted)),
                "observed_cumulative_incidence": float(np.mean(observed)),
            }
        )
    observed_mask = frame["target_observed"].to_numpy(dtype=bool)
    absolute = np.abs(
        predicted_median_bars[observed_mask]
        - frame.loc[observed_mask, "bars_until_resolution"].to_numpy(dtype=float)
    )
    median_absolute = float(np.median(absolute)) if len(absolute) else float("nan")
    output = pd.DataFrame.from_records(records)
    output["median_absolute_bars_to_event_error"] = median_absolute
    return output, float(np.mean(brier_values))


def paired_block_bootstrap(
    frame: pd.DataFrame,
    *,
    candidate_loss: np.ndarray,
    baseline_loss: np.ndarray,
    group_column: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    if len(frame) != len(candidate_loss) or len(frame) != len(baseline_loss):
        raise ValueError("loss arrays must match frame")
    differences = np.asarray(candidate_loss) - np.asarray(baseline_loss)
    groups = frame[group_column].astype(str).to_numpy()
    unique = np.asarray(sorted(set(groups)), dtype=object)
    by_group = {value: differences[groups == value] for value in unique}
    rng = np.random.default_rng(seed)
    records = []
    for draw in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        values = np.concatenate([by_group[str(value)] for value in sampled])
        records.append({"draw": draw, "paired_loss_difference": float(np.mean(values))})
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True, slots=True)
class PartBGateMetrics:
    source_blocked: bool
    candidate_beats_log_loss: bool
    candidate_beats_brier: bool
    log_loss_upper_below_zero: bool
    brier_upper_below_zero: bool
    relative_log_loss_improvement: float
    favourable_quarters: int
    all_stock_deletions_favourable: bool
    calibration_not_worse: bool
    improved_major_classes: int
    return_only_gain: bool
    median_correct_lead_time_bars: float
    sensitivity_directionally_similar: bool
    binary_support_sufficient: bool
    binary_gate_pass: bool
    timing_gate_pass: bool
    pooled_improvement_present: bool


def decide_part_b(metrics: PartBGateMetrics) -> str:
    if metrics.source_blocked:
        return "excursion_forecast_experiment_blocked"
    multiclass = (
        metrics.candidate_beats_log_loss
        and metrics.candidate_beats_brier
        and metrics.log_loss_upper_below_zero
        and metrics.brier_upper_below_zero
        and metrics.relative_log_loss_improvement >= 0.005
        and metrics.favourable_quarters >= 3
        and metrics.all_stock_deletions_favourable
        and metrics.calibration_not_worse
        and metrics.improved_major_classes >= 2
        and not metrics.return_only_gain
        and metrics.median_correct_lead_time_bars >= 2.0
        and metrics.sensitivity_directionally_similar
    )
    if multiclass:
        return "cluster_invariant_excursion_forecast_validated"
    if metrics.binary_support_sufficient and metrics.binary_gate_pass:
        return "cluster_invariant_return_probability_validated"
    if metrics.timing_gate_pass:
        return "excursion_resolution_timing_validated"
    if metrics.pooled_improvement_present:
        return "excursion_structural_forecast_weak"
    return "no_predictable_excursion_resolution_structure"


def first_eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(["event_id", "decision_timestamp", "decision_id"], kind="stable")
        .drop_duplicates("event_id", keep="first")
        .reset_index(drop=True)
    )


def last_eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values(["event_id", "decision_timestamp", "decision_id"], kind="stable")
        .drop_duplicates("event_id", keep="last")
        .reset_index(drop=True)
    )


__all__ = [
    "SAFETY_FLAGS",
    "TARGET_CLASSES",
    "FoldPreprocessor",
    "MultinomialEstimator",
    "PartBGateMetrics",
    "balanced_event_weights",
    "build_active_excursion_rows",
    "calibration_table",
    "canonical_target_family",
    "confusion_matrix_frame",
    "constant_hazard_competing_risk",
    "decide_part_b",
    "first_eligible_rows",
    "fit_multinomial",
    "frequency_probabilities",
    "last_eligible_rows",
    "multiclass_losses",
    "paired_block_bootstrap",
    "per_class_metrics",
    "ranking_metrics",
    "timing_metrics",
    "validate_probabilities",
]
