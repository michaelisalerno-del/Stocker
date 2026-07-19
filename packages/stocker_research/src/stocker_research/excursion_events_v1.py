"""Cluster-invariant excursion detection and deterministic Part A gates V1."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from stocker_research.continuous_trajectory_v1 import (
    diagonal_distance,
    jensen_shannon_distance,
    mahalanobis_distance,
)
from stocker_research.excursion_origin_v1 import OriginSurface

FloatArray = NDArray[np.float64]


class EventFamily(StrEnum):
    RETURN_TO_ORIGIN = "RETURN_TO_ORIGIN"
    PARTIAL_RETURN = "PARTIAL_RETURN"
    CONTINUE_AWAY = "CONTINUE_AWAY"
    ROTATE_TO_NEW_REGION = "ROTATE_TO_NEW_REGION"
    REMAIN_LOCAL = "REMAIN_LOCAL"
    SESSION_END = "SESSION_END"
    UNAVAILABLE_SOURCE = "UNAVAILABLE_SOURCE"
    UNAVAILABLE_STRUCTURAL_GAP = "UNAVAILABLE_STRUCTURAL_GAP"
    UNRESOLVED_AT_HORIZON = "UNRESOLVED_AT_HORIZON"


_PRECEDENCE = (
    EventFamily.UNAVAILABLE_SOURCE,
    EventFamily.UNAVAILABLE_STRUCTURAL_GAP,
    EventFamily.RETURN_TO_ORIGIN,
    EventFamily.ROTATE_TO_NEW_REGION,
    EventFamily.CONTINUE_AWAY,
    EventFamily.PARTIAL_RETURN,
    EventFamily.SESSION_END,
    EventFamily.UNRESOLVED_AT_HORIZON,
)


@dataclass(frozen=True, slots=True)
class DistanceCalibration:
    """Development-frozen geometry scales used by one event definition."""

    emission_scale: FloatArray
    emission_q90: float
    posterior_q90: float
    mahalanobis_precision: FloatArray

    def __post_init__(self) -> None:
        scale = np.asarray(self.emission_scale, dtype=float)
        precision = np.asarray(self.mahalanobis_precision, dtype=float)
        if scale.ndim != 1 or not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("emission geometry scales must be finite and positive")
        if precision.shape != (len(scale), len(scale)) or not np.isfinite(precision).all():
            raise ValueError("Mahalanobis precision differs from emission dimensions")
        if self.emission_q90 <= 0.0 or self.posterior_q90 <= 0.0:
            raise ValueError("hybrid normalizers must be positive")


@dataclass(frozen=True, slots=True)
class ExcursionConfig:
    """One preregistered mutually exclusive excursion definition."""

    candidate_id: str
    representation: str
    distance_metric: str
    departure_threshold: float
    confirmation_bars: int
    velocity_condition: bool
    minimum_departure_velocity: float
    return_ratio: float
    rotation_persistence: int
    rotation_separation_ratio: float
    rotation_maximum_velocity: float
    continuation_ratio: float
    partial_retracement_fraction: float
    horizon_bars: int
    lockout_bars: int = 1

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("event candidate ID cannot be empty")
        if self.representation not in {"E", "P", "H"}:
            raise ValueError("representation must be E P or H")
        if self.distance_metric not in {
            "DIAGONAL",
            "SHRINKAGE_MAHALANOBIS",
            "JENSEN_SHANNON",
            "EQUAL_WEIGHT_HYBRID",
        }:
            raise ValueError("unsupported excursion distance metric")
        if self.departure_threshold <= 0.0 or self.minimum_departure_velocity < 0.0:
            raise ValueError("departure thresholds are invalid")
        if self.confirmation_bars not in {1, 2}:
            raise ValueError("confirmation must use one or two completed bars")
        if not 0.0 < self.return_ratio < 1.0:
            raise ValueError("return ratio must lie inside (0, 1)")
        if self.rotation_persistence not in {1, 2, 3}:
            raise ValueError("rotation persistence must be one two or three bars")
        if self.rotation_separation_ratio <= 0.0 or self.rotation_maximum_velocity < 0.0:
            raise ValueError("rotation configuration is invalid")
        if self.continuation_ratio <= 1.0:
            raise ValueError("continuation ratio must exceed one")
        if not 0.0 <= self.partial_retracement_fraction <= 1.0:
            raise ValueError("partial retracement threshold must be in [0, 1]")
        if self.horizon_bars <= 0 or self.lockout_bars < 0:
            raise ValueError("event horizon or lockout is invalid")


@dataclass(frozen=True, slots=True)
class ExcursionDetection:
    events: pd.DataFrame
    decision_mapping: pd.DataFrame
    coincident_conditions: pd.DataFrame
    departure_candidates: pd.DataFrame


def event_definition_hash(
    config: ExcursionConfig,
    *,
    calibration: DistanceCalibration | None = None,
    origin_definition_id: str | None = None,
    posterior_origin_definition_id: str | None = None,
) -> str:
    """Hash every declared and fitted field that can change event identity."""

    payload_value: dict[str, Any] = {"config": asdict(config)}
    if calibration is not None:
        payload_value["calibration"] = {
            "emission_scale": np.asarray(calibration.emission_scale, dtype=np.float64).tolist(),
            "emission_q90": calibration.emission_q90,
            "posterior_q90": calibration.posterior_q90,
            "mahalanobis_precision": np.asarray(
                calibration.mahalanobis_precision, dtype=np.float64
            ).tolist(),
        }
    payload_value["origin_definition_id"] = origin_definition_id
    payload_value["posterior_origin_definition_id"] = posterior_origin_definition_id
    payload = json.dumps(payload_value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _event_id(
    *,
    symbol: str,
    session: str,
    segment_id: str,
    onset_timestamp: pd.Timestamp,
    origin_id: str,
    definition_hash: str,
) -> str:
    payload = "|".join(
        (
            symbol,
            session,
            segment_id,
            pd.Timestamp(onset_timestamp).isoformat(),
            origin_id,
            definition_hash,
        )
    )
    return f"excursion_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _json_vector(values: FloatArray | None) -> str:
    if values is None:
        return "[]"
    return json.dumps(
        [float(value) for value in np.asarray(values, dtype=float)],
        separators=(",", ":"),
        allow_nan=False,
    )


def _validated_posterior(values: FloatArray | None, rows: int) -> FloatArray | None:
    if values is None:
        return None
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != rows:
        raise ValueError("posterior trajectory differs from decision rows")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("posterior trajectory contains invalid probabilities")
    totals = matrix.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("posterior trajectory contains an empty row")
    return np.asarray(matrix / totals[:, None], dtype=np.float64)


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(cast(Any, value))


def _emission_distance(
    current: FloatArray,
    origin: FloatArray,
    *,
    calibration: DistanceCalibration,
    metric: str,
) -> float:
    if metric == "SHRINKAGE_MAHALANOBIS":
        return mahalanobis_distance(current, origin, calibration.mahalanobis_precision)
    return diagonal_distance(current, origin, calibration.emission_scale)


def _distance(
    current_emission: FloatArray,
    origin_emission: FloatArray,
    current_posterior: FloatArray | None,
    origin_posterior: FloatArray | None,
    *,
    calibration: DistanceCalibration,
    config: ExcursionConfig,
) -> float:
    emission = _emission_distance(
        current_emission,
        origin_emission,
        calibration=calibration,
        metric=config.distance_metric,
    )
    if config.representation == "E":
        return emission
    if current_posterior is None or origin_posterior is None:
        return math.nan
    posterior = jensen_shannon_distance(current_posterior, origin_posterior)
    if config.representation == "P":
        return posterior
    return 0.5 * (emission / calibration.emission_q90) + 0.5 * (
        posterior / calibration.posterior_q90
    )


def _point_distance(
    current_emission: FloatArray,
    previous_emission: FloatArray,
    current_posterior: FloatArray | None,
    previous_posterior: FloatArray | None,
    *,
    calibration: DistanceCalibration,
    config: ExcursionConfig,
) -> float:
    if config.representation == "P":
        if current_posterior is None or previous_posterior is None:
            return math.nan
        return jensen_shannon_distance(current_posterior, previous_posterior)
    return _emission_distance(
        current_emission,
        previous_emission,
        calibration=calibration,
        metric=config.distance_metric,
    )


def _required_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "decision_id",
        "symbol",
        "session",
        "segment_id",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "decision_timestamp",
        "segment_end_reason",
        "session_source_complete",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"excursion decision frame lacks columns: {missing}")
    result = frame.copy().reset_index(drop=True)
    for column in ("bar_start_timestamp", "bar_complete_timestamp", "decision_timestamp"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if result["decision_id"].duplicated().any():
        raise ValueError("excursion decision IDs must be unique")
    return result


def _origin_identity(
    emission_origins: OriginSurface,
    posterior_origins: OriginSurface | None,
    position: int,
    representation: str,
) -> str:
    if representation == "E" or posterior_origins is None:
        return emission_origins.origin_ids[position]
    if representation == "P":
        return posterior_origins.origin_ids[position]
    payload = f"{emission_origins.origin_ids[position]}|{posterior_origins.origin_ids[position]}"
    return f"origin_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _boundary_family(reason: str, source_complete: bool) -> EventFamily:
    normalized = reason.lower()
    if "source_gap" in normalized:
        return EventFamily.UNAVAILABLE_STRUCTURAL_GAP
    if "scheduled_session_end" in normalized and source_complete:
        return EventFamily.SESSION_END
    return EventFamily.UNAVAILABLE_SOURCE


def _choose_condition(conditions: list[EventFamily]) -> EventFamily:
    condition_set = set(conditions)
    for family in _PRECEDENCE:
        if family in condition_set:
            return family
    raise AssertionError("resolution condition list is empty")


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "candidate_id",
            "event_definition_hash",
            "trajectory_representation",
            "distance_metric",
            "symbol",
            "session",
            "segment_id",
            "frozen_origin_id",
            "frozen_origin_vector",
            "frozen_posterior_origin",
            "departure_direction_vector",
            "first_detectable_bar_ordinal",
            "confirmation_bar_ordinal",
            "onset_bar_ordinal",
            "resolution_bar_ordinal",
            "first_detectable_timestamp",
            "confirmation_timestamp",
            "onset_timestamp",
            "resolution_timestamp",
            "event_family",
            "departure_distance",
            "confirmation_distance",
            "maximum_excursion_distance",
            "resolution_distance",
            "retracement_fraction",
            "bars_from_first_detectable_to_resolution",
            "bars_from_confirmation_to_resolution",
            "coincident_conditions_json",
        ]
    )


def detect_excursions(
    frame: pd.DataFrame,
    *,
    emission_vectors: FloatArray,
    emission_origins: OriginSurface,
    posterior_vectors: FloatArray | None,
    posterior_origins: OriginSurface | None,
    calibration: DistanceCalibration,
    config: ExcursionConfig,
) -> ExcursionDetection:
    """Detect non-overlapping excursions and resolve by frozen scientific precedence."""

    decisions = _required_frame(frame)
    emissions = np.asarray(emission_vectors, dtype=np.float64)
    if emissions.ndim != 2 or len(emissions) != len(decisions):
        raise ValueError("emission trajectory differs from decision rows")
    if emission_origins.centers.shape != emissions.shape:
        raise ValueError("emission origin surface differs from trajectory")
    posteriors = _validated_posterior(posterior_vectors, len(decisions))
    if config.representation in {"P", "H"}:
        if posteriors is None or posterior_origins is None:
            raise ValueError("posterior or hybrid events require posterior origins")
        if posterior_origins.centers.shape != posteriors.shape:
            raise ValueError("posterior origin surface differs from trajectory")
    definition_hash = event_definition_hash(
        config,
        calibration=calibration,
        origin_definition_id=emission_origins.definition_id,
        posterior_origin_definition_id=(
            None if posterior_origins is None else posterior_origins.definition_id
        ),
    )
    event_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    grouped = decisions.groupby(["symbol", "session", "segment_id"], sort=False)
    for (raw_symbol, raw_session, raw_segment), group_frame in grouped:
        positions = group_frame.index.to_numpy(dtype=np.int64)
        symbol = str(raw_symbol)
        session = str(raw_session)
        segment_id = str(raw_segment)
        local_index = 0
        while local_index < len(positions):
            onset_position = int(positions[local_index])
            origin_eligible = bool(emission_origins.eligible[onset_position])
            if config.representation in {"P", "H"} and posterior_origins is not None:
                origin_eligible &= bool(posterior_origins.eligible[onset_position])
            if not origin_eligible or not np.isfinite(emissions[onset_position]).all():
                local_index += 1
                continue
            origin_emission = (
                emissions[onset_position] * 0.0 + emission_origins.centers[onset_position]
            )
            origin_posterior = (
                None
                if posterior_origins is None
                else np.asarray(posterior_origins.centers[onset_position], dtype=float)
            )
            current_posterior = None if posteriors is None else posteriors[onset_position]
            onset_distance = _distance(
                emissions[onset_position],
                origin_emission,
                current_posterior,
                origin_posterior,
                calibration=calibration,
                config=config,
            )
            previous_distance = 0.0
            if local_index > 0:
                previous_position = int(positions[local_index - 1])
                previous_posterior = None if posteriors is None else posteriors[previous_position]
                previous_distance = _distance(
                    emissions[previous_position],
                    origin_emission,
                    previous_posterior,
                    origin_posterior,
                    calibration=calibration,
                    config=config,
                )
                if not math.isfinite(previous_distance):
                    previous_distance = 0.0
            onset_velocity = onset_distance - previous_distance
            qualifies = (
                math.isfinite(onset_distance) and onset_distance >= config.departure_threshold
            )
            if config.velocity_condition:
                qualifies &= onset_velocity >= config.minimum_departure_velocity
            if not qualifies:
                local_index += 1
                continue

            confirmation_local = local_index + config.confirmation_bars - 1
            confirmed = confirmation_local < len(positions)
            confirmation_distance = onset_distance
            if confirmed:
                for check_local in range(local_index, confirmation_local + 1):
                    check_position = int(positions[check_local])
                    check_posterior = None if posteriors is None else posteriors[check_position]
                    check_distance = _distance(
                        emissions[check_position],
                        origin_emission,
                        check_posterior,
                        origin_posterior,
                        calibration=calibration,
                        config=config,
                    )
                    if (
                        not math.isfinite(check_distance)
                        or check_distance < config.departure_threshold
                    ):
                        confirmed = False
                        break
                    confirmation_distance = check_distance
            candidate_key = (
                f"{config.candidate_id}|{symbol}|{session}|{segment_id}|"
                f"{decisions.at[onset_position, 'bar_complete_timestamp']}"
            )
            candidate_rows.append(
                {
                    "departure_candidate_id": (
                        "departure_"
                        + hashlib.sha256(candidate_key.encode("utf-8")).hexdigest()[:24]
                    ),
                    "candidate_id": config.candidate_id,
                    "symbol": symbol,
                    "session": session,
                    "segment_id": segment_id,
                    "decision_id": str(decisions.at[onset_position, "decision_id"]),
                    "onset_bar_ordinal": _as_int(decisions.at[onset_position, "bar_ordinal"]),
                    "onset_timestamp": decisions.at[onset_position, "bar_complete_timestamp"],
                    "departure_distance": float(onset_distance),
                    "departure_velocity": float(onset_velocity),
                    "confirmation_bars": config.confirmation_bars,
                    "confirmed": bool(confirmed),
                    "frozen_origin_id": _origin_identity(
                        emission_origins,
                        posterior_origins,
                        onset_position,
                        config.representation,
                    ),
                }
            )
            if not confirmed:
                local_index += 1
                continue

            confirmation_position = int(positions[confirmation_local])
            origin_id = _origin_identity(
                emission_origins,
                posterior_origins,
                onset_position,
                config.representation,
            )
            onset_timestamp = _as_timestamp(decisions.at[onset_position, "bar_complete_timestamp"])
            event_id = _event_id(
                symbol=symbol,
                session=session,
                segment_id=segment_id,
                onset_timestamp=onset_timestamp,
                origin_id=origin_id,
                definition_hash=definition_hash,
            )
            departure_direction = emissions[onset_position] - origin_emission
            direction_norm = float(np.linalg.norm(departure_direction))
            if direction_norm > 0.0:
                departure_direction = departure_direction / direction_norm
            maximum_distance = max(float(onset_distance), float(confirmation_distance))
            resolution_position = confirmation_position
            resolution_local = confirmation_local
            resolution_distance = float(confirmation_distance)
            retracement = 0.0
            resolved_family: EventFamily | None = None
            resolved_conditions: list[EventFamily] = []

            evaluation_locals = list(range(confirmation_local + 1, len(positions)))
            if not evaluation_locals:
                evaluation_locals = [confirmation_local]
            for current_local in evaluation_locals:
                current_position = int(positions[current_local])
                current_emission = emissions[current_position]
                current_posterior = None if posteriors is None else posteriors[current_position]
                conditions: list[EventFamily] = []
                current_distance = _distance(
                    current_emission,
                    origin_emission,
                    current_posterior,
                    origin_posterior,
                    calibration=calibration,
                    config=config,
                )
                unavailable = not math.isfinite(current_distance)
                if unavailable:
                    conditions.append(EventFamily.UNAVAILABLE_SOURCE)
                else:
                    prior_maximum = maximum_distance
                    maximum_distance = max(maximum_distance, float(current_distance))
                    resolution_distance = float(current_distance)
                    retracement = (
                        max(0.0, (maximum_distance - float(current_distance)) / maximum_distance)
                        if maximum_distance > 0.0
                        else 0.0
                    )
                    if current_distance <= config.return_ratio * config.departure_threshold:
                        conditions.append(EventFamily.RETURN_TO_ORIGIN)

                    rotation_start = current_local - config.rotation_persistence + 1
                    if rotation_start >= confirmation_local + 1 or (
                        config.rotation_persistence == 1 and current_local >= confirmation_local + 1
                    ):
                        rotation_positions = positions[rotation_start : current_local + 1]
                        local_velocities: list[float] = []
                        for previous_raw, current_raw in zip(
                            rotation_positions[:-1], rotation_positions[1:], strict=True
                        ):
                            previous_position = int(previous_raw)
                            next_position = int(current_raw)
                            local_velocities.append(
                                _point_distance(
                                    emissions[next_position],
                                    emissions[previous_position],
                                    None if posteriors is None else posteriors[next_position],
                                    None if posteriors is None else posteriors[previous_position],
                                    calibration=calibration,
                                    config=config,
                                )
                            )
                        rotation_velocity = (
                            float(np.mean(local_velocities)) if local_velocities else 0.0
                        )
                        region_emission = np.median(emissions[rotation_positions], axis=0)
                        region_posterior = (
                            None
                            if posteriors is None
                            else np.median(posteriors[rotation_positions], axis=0)
                        )
                        region_separation = _distance(
                            region_emission,
                            origin_emission,
                            region_posterior,
                            origin_posterior,
                            calibration=calibration,
                            config=config,
                        )
                        if (
                            math.isfinite(rotation_velocity)
                            and rotation_velocity <= config.rotation_maximum_velocity
                            and region_separation
                            >= config.rotation_separation_ratio * config.departure_threshold
                        ):
                            conditions.append(EventFamily.ROTATE_TO_NEW_REGION)

                    outward_velocity = float(current_distance) - float(prior_maximum)
                    continuation_boundary = config.continuation_ratio * max(
                        config.departure_threshold,
                        float(onset_distance),
                    )
                    if current_distance >= continuation_boundary and outward_velocity > 0.0:
                        conditions.append(EventFamily.CONTINUE_AWAY)

                bars_after_confirmation = current_local - confirmation_local
                at_horizon = bars_after_confirmation >= config.horizon_bars
                at_segment_end = current_local == len(positions) - 1
                if at_segment_end:
                    boundary = _boundary_family(
                        str(decisions.at[current_position, "segment_end_reason"]),
                        bool(decisions.at[current_position, "session_source_complete"]),
                    )
                    conditions.append(boundary)
                if at_horizon or at_segment_end:
                    if not unavailable and retracement >= config.partial_retracement_fraction:
                        conditions.append(EventFamily.PARTIAL_RETURN)
                    if at_horizon:
                        conditions.append(EventFamily.UNRESOLVED_AT_HORIZON)

                if conditions:
                    ordered_conditions = [family for family in _PRECEDENCE if family in conditions]
                    chosen = _choose_condition(ordered_conditions)
                    condition_rows.append(
                        {
                            "event_id": event_id,
                            "decision_id": str(decisions.at[current_position, "decision_id"]),
                            "bar_ordinal": _as_int(decisions.at[current_position, "bar_ordinal"]),
                            "decision_timestamp": decisions.at[
                                current_position, "decision_timestamp"
                            ],
                            "chosen_family": chosen.value,
                            "conditions_json": json.dumps(
                                [family.value for family in ordered_conditions],
                                separators=(",", ":"),
                            ),
                        }
                    )
                    resolved_family = chosen
                    resolved_conditions = ordered_conditions
                    resolution_position = current_position
                    resolution_local = current_local
                    break

            if resolved_family is None:
                raise AssertionError("confirmed excursion did not resolve at horizon or boundary")
            event_rows.append(
                {
                    "event_id": event_id,
                    "candidate_id": config.candidate_id,
                    "event_definition_hash": definition_hash,
                    "trajectory_representation": config.representation,
                    "distance_metric": config.distance_metric,
                    "symbol": symbol,
                    "session": session,
                    "segment_id": segment_id,
                    "frozen_origin_id": origin_id,
                    "frozen_origin_vector": _json_vector(origin_emission),
                    "frozen_posterior_origin": _json_vector(origin_posterior),
                    "departure_direction_vector": _json_vector(departure_direction),
                    "first_detectable_bar_ordinal": _as_int(
                        decisions.at[onset_position, "bar_ordinal"]
                    ),
                    "confirmation_bar_ordinal": _as_int(
                        decisions.at[confirmation_position, "bar_ordinal"]
                    ),
                    "onset_bar_ordinal": _as_int(decisions.at[onset_position, "bar_ordinal"]),
                    "resolution_bar_ordinal": _as_int(
                        decisions.at[resolution_position, "bar_ordinal"]
                    ),
                    "first_detectable_timestamp": decisions.at[
                        onset_position, "bar_complete_timestamp"
                    ],
                    "confirmation_timestamp": decisions.at[
                        confirmation_position, "decision_timestamp"
                    ],
                    "onset_timestamp": onset_timestamp,
                    "resolution_timestamp": decisions.at[
                        resolution_position, "bar_complete_timestamp"
                    ],
                    "event_family": resolved_family.value,
                    "departure_distance": float(onset_distance),
                    "confirmation_distance": float(confirmation_distance),
                    "maximum_excursion_distance": float(maximum_distance),
                    "resolution_distance": float(resolution_distance),
                    "retracement_fraction": float(retracement),
                    "bars_from_first_detectable_to_resolution": int(resolution_local - local_index),
                    "bars_from_confirmation_to_resolution": int(
                        resolution_local - confirmation_local
                    ),
                    "coincident_conditions_json": json.dumps(
                        [family.value for family in resolved_conditions],
                        separators=(",", ":"),
                    ),
                }
            )
            for mapped_local in range(local_index, resolution_local + 1):
                mapped_position = int(positions[mapped_local])
                mapping_rows.append(
                    {
                        "event_id": event_id,
                        "decision_id": str(decisions.at[mapped_position, "decision_id"]),
                        "symbol": symbol,
                        "session": session,
                        "segment_id": segment_id,
                        "bar_ordinal": _as_int(decisions.at[mapped_position, "bar_ordinal"]),
                        "decision_timestamp": decisions.at[mapped_position, "decision_timestamp"],
                        "mapping_role": (
                            "FIRST_DETECTABLE"
                            if mapped_local == local_index
                            else (
                                "DEPARTURE_CONFIRMATION"
                                if mapped_local == confirmation_local
                                else (
                                    "RESOLUTION"
                                    if mapped_local == resolution_local
                                    else "ACTIVE_EXCURSION"
                                )
                            )
                        ),
                    }
                )
            local_index = resolution_local + config.lockout_bars + 1

    events = _empty_events() if not event_rows else pd.DataFrame(event_rows)
    mapping = pd.DataFrame(
        mapping_rows,
        columns=[
            "event_id",
            "decision_id",
            "symbol",
            "session",
            "segment_id",
            "bar_ordinal",
            "decision_timestamp",
            "mapping_role",
        ],
    )
    coincident = pd.DataFrame(
        condition_rows,
        columns=[
            "event_id",
            "decision_id",
            "bar_ordinal",
            "decision_timestamp",
            "chosen_family",
            "conditions_json",
        ],
    )
    candidates = pd.DataFrame(
        candidate_rows,
        columns=[
            "departure_candidate_id",
            "candidate_id",
            "symbol",
            "session",
            "segment_id",
            "decision_id",
            "onset_bar_ordinal",
            "onset_timestamp",
            "departure_distance",
            "departure_velocity",
            "confirmation_bars",
            "confirmed",
            "frozen_origin_id",
        ],
    )
    if not events.empty and not events["event_id"].is_unique:
        raise AssertionError("excursion event IDs are not unique")
    return ExcursionDetection(
        events=events,
        decision_mapping=mapping,
        coincident_conditions=coincident,
        departure_candidates=candidates,
    )


@dataclass(frozen=True, slots=True)
class PartAGateMetrics:
    representation: str
    cross_lineage_agreement: float
    cross_seed_agreement: float
    cross_sample_agreement: float
    cross_k_agreement: float
    maximum_validation_share_shift_pp: float
    unique_development_events: int
    unique_validation_events: int
    stock_count: int
    month_count: int
    maximum_stock_share: float
    maximum_month_share: float
    median_timing_disagreement_bars: float
    posterior_hybrid_validated: bool
    secondary_gate_narrow_failure: bool
    source_blocked: bool
    exact_rerun_pass: bool
    independent_audit_pass: bool


def decide_part_a(metrics: PartAGateMetrics) -> str:
    """Apply the frozen Part A hierarchy without forecast or outcome information."""

    if metrics.source_blocked or not metrics.exact_rerun_pass or not metrics.independent_audit_pass:
        return "cluster_invariant_event_experiment_blocked"
    stable = (
        metrics.cross_lineage_agreement >= 0.70
        and metrics.cross_seed_agreement >= 0.65
        and metrics.cross_sample_agreement >= 0.60
        and metrics.cross_k_agreement >= 0.55
        and metrics.maximum_validation_share_shift_pp <= 7.5
        and metrics.maximum_stock_share <= 0.20
        and metrics.maximum_month_share <= 0.25
        and metrics.median_timing_disagreement_bars <= 2.0
    )
    if not stable:
        return "cluster_invariant_excursion_events_not_stable"
    support = (
        metrics.unique_development_events >= 2000
        and metrics.unique_validation_events >= 1000
        and metrics.stock_count >= 15
        and metrics.month_count >= 8
    )
    if not support:
        return "cluster_invariant_event_population_too_sparse"
    if metrics.representation == "E" and not metrics.posterior_hybrid_validated:
        return "emission_space_excursion_events_validated"
    if metrics.secondary_gate_narrow_failure:
        return "cluster_invariant_events_valid_with_required_sensitivity"
    return "cluster_invariant_excursion_events_validated"


def part_b_authorized(
    decision: str,
    *,
    exact_rerun_pass: bool,
    independent_audit_pass: bool,
    binding_hash: str,
) -> bool:
    """Hard-close Part B unless the final frozen Part A lineage is complete."""

    allowed = {
        "cluster_invariant_excursion_events_validated",
        "cluster_invariant_events_valid_with_required_sensitivity",
        "emission_space_excursion_events_validated",
    }
    return bool(
        decision in allowed
        and exact_rerun_pass
        and independent_audit_pass
        and len(binding_hash) == 64
    )


__all__ = [
    "DistanceCalibration",
    "EventFamily",
    "ExcursionConfig",
    "ExcursionDetection",
    "PartAGateMetrics",
    "decide_part_a",
    "detect_excursions",
    "event_definition_hash",
    "part_b_authorized",
]
