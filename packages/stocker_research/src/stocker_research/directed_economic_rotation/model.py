"""Deterministic prequential multi-label family activation models."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .graph import (
    GraphSettings,
    MaturedRotationExample,
    PastOnlyRotationGraph,
)

OWN_FEATURES = (
    "state_unknown",
    "state_decaying",
    "state_retired",
    "destination_max_p_edge_active",
    "destination_max_p_on_next",
    "destination_mean_p_off_next",
    "destination_posterior_mean_scaled",
    "destination_posterior_mean_change_scaled",
    "destination_active_probability_change",
    "destination_state_age_scaled",
    "sessions_since_destination_active_scaled",
    "destination_support_scaled",
    "destination_uncertainty_scaled",
    "past_only_activation_base_rate",
)
SYSTEM_FEATURES = (
    "active_family_fraction",
    "decaying_family_fraction",
    "newly_retired_family_fraction",
    "newly_decaying_family_fraction",
    "unknown_family_fraction",
    "active_family_entropy",
    "market_wide_family_transition_fraction",
)
DIRECTED_FEATURES = (
    "directed_active_log_lift_score",
    "directed_newly_decaying_log_lift_score",
    "directed_newly_retired_log_lift_score",
    "maximum_supported_positive_directed_log_lift",
    "supported_source_edge_fraction",
)
MODEL_SCHEMAS = {
    "M1_destination_own_history": OWN_FEATURES,
    "M2_undirected_system_state": (*OWN_FEATURES, *SYSTEM_FEATURES),
    "M3_directed_family_rotation": (*OWN_FEATURES, *SYSTEM_FEATURES, *DIRECTED_FEATURES),
}


@dataclass(frozen=True)
class PrequentialSettings:
    run_id: str
    target_window_sessions: int = 3
    learning_rate: float = 0.05
    ridge_penalty: float = 0.05
    feature_clip: float = 5.0
    coefficient_clip: float = 4.0
    minimum_training_rows: int = 30
    minimum_training_activations: int = 4
    minimum_lift_over_base: float = 1.25
    maximum_interval_width: float = 0.5
    graph: GraphSettings = field(default_factory=GraphSettings)

    def __post_init__(self) -> None:
        if self.target_window_sessions not in {1, 3, 5}:
            raise ValueError("target window is outside the registered family")
        if min(self.learning_rate, self.ridge_penalty, self.feature_clip) <= 0.0:
            raise ValueError("online model settings must be positive")
        if self.coefficient_clip <= 0.0:
            raise ValueError("coefficient clip must be positive")


class OnlineRidgeLogit:
    """Small deterministic online logistic head around a causal base-rate offset."""

    def __init__(self, feature_names: tuple[str, ...], settings: PrequentialSettings) -> None:
        self.feature_names = feature_names
        self.settings = settings
        self.weights = np.zeros(len(feature_names), dtype=float)
        self.updates = 0

    def _vector(self, features: dict[str, float]) -> np.ndarray:
        vector = np.asarray(
            np.asarray([features.get(name, 0.0) for name in self.feature_names], dtype=float),
            dtype=float,
        )
        return np.asarray(
            np.clip(vector, -self.settings.feature_clip, self.settings.feature_clip),
            dtype=float,
        )

    def predict(self, features: dict[str, float], base_rate: float) -> float:
        offset = float(logit(np.clip(base_rate, 1e-6, 1.0 - 1e-6)))
        return float(expit(offset + np.dot(self.weights, self._vector(features))))

    def update(self, features: dict[str, float], target: bool, base_rate: float) -> None:
        vector = self._vector(features)
        probability = self.predict(features, base_rate)
        rate = self.settings.learning_rate / math.sqrt(self.updates + 1.0)
        gradient = (
            float(target) - probability
        ) * vector - self.settings.ridge_penalty * self.weights
        self.weights = np.clip(
            self.weights + rate * gradient,
            -self.settings.coefficient_clip,
            self.settings.coefficient_clip,
        )
        self.updates += 1


@dataclass
class _Pending:
    example: MaturedRotationExample
    base_rate: float
    features: dict[str, dict[str, float]]
    trained: bool = False


def _stable_id(prefix: str, values: tuple[object, ...]) -> str:
    payload = json.dumps([str(value) for value in values], separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _source_vector(rows: pd.DataFrame) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for row in rows.to_dict(orient="records"):
        events: set[str] = set()
        if bool(row.get("source_active", False)):
            events.add("active")
        if bool(row.get("newly_decaying", False)):
            events.add("newly_decaying")
        if bool(row.get("newly_retired", False)):
            events.add("newly_retired")
        if events:
            result[str(row["destination_family"])] = frozenset(events)
    return result


def _own_features(row: dict[str, Any], base_rate: float) -> dict[str, float]:
    state = str(row["operational_state"])
    return {
        "state_unknown": float(state == "unknown"),
        "state_decaying": float(state == "decaying"),
        "state_retired": float(state == "retired"),
        "destination_max_p_edge_active": _finite(row.get("max_p_edge_active")),
        "destination_max_p_on_next": _finite(row.get("max_p_on_next")),
        "destination_mean_p_off_next": _finite(row.get("mean_p_off_next")),
        "destination_posterior_mean_scaled": _finite(row.get("posterior_mean_net_bps")) / 100.0,
        "destination_posterior_mean_change_scaled": _finite(row.get("posterior_mean_change_bps"))
        / 100.0,
        "destination_active_probability_change": _finite(row.get("active_probability_change")),
        "destination_state_age_scaled": min(_finite(row.get("state_age_sessions")) / 20.0, 5.0),
        "sessions_since_destination_active_scaled": min(
            _finite(row.get("sessions_since_active")) / 20.0, 5.0
        ),
        "destination_support_scaled": min(
            math.log1p(max(_finite(row.get("effective_sample_size")), 0.0)) / 5.0, 5.0
        ),
        "destination_uncertainty_scaled": _finite(row.get("posterior_std_net_bps")) / 100.0,
        "past_only_activation_base_rate": base_rate,
    }


def _system_features(states: pd.DataFrame, events: pd.DataFrame) -> dict[str, float]:
    count = max(len(states), 1)
    state = states["operational_state"].astype(str)
    transition = events.get("source_state_transition", pd.Series(False, index=events.index))
    active_probabilities = np.clip(
        pd.to_numeric(states["max_p_edge_active"], errors="coerce").fillna(0.0).to_numpy(float),
        0.0,
        1.0,
    )
    total = float(active_probabilities.sum())
    if total > 0.0 and len(active_probabilities) > 1:
        weights = active_probabilities / total
        entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0)))) / math.log(
            len(active_probabilities)
        )
    else:
        entropy = 0.0
    return {
        "active_family_fraction": float(state.eq("active").sum() / count),
        "decaying_family_fraction": float(state.eq("decaying").sum() / count),
        "newly_retired_family_fraction": float(events["newly_retired"].sum() / count),
        "newly_decaying_family_fraction": float(events["newly_decaying"].sum() / count),
        "unknown_family_fraction": float(state.eq("unknown").sum() / count),
        "active_family_entropy": entropy,
        "market_wide_family_transition_fraction": float(transition.sum() / count),
    }


def _shift_interval(
    base_probability: float,
    model_probability: float,
    interval: tuple[float, float],
) -> tuple[float, float]:
    delta = float(
        logit(np.clip(model_probability, 1e-6, 1.0 - 1e-6))
        - logit(np.clip(base_probability, 1e-6, 1.0 - 1e-6))
    )
    return (
        float(expit(logit(np.clip(interval[0], 1e-6, 1.0 - 1e-6)) + delta)),
        float(expit(logit(np.clip(interval[1], 1e-6, 1.0 - 1e-6)) + delta)),
    )


def _prediction_state(
    *,
    row: dict[str, Any],
    probability: float,
    base_rate: float,
    interval: tuple[float, float],
    support: int,
    activations: int,
    settings: PrequentialSettings,
) -> tuple[str, str]:
    reasons: list[str] = []
    if str(row["operational_state"]) == "active":
        reasons.append("destination_already_active")
    if support < settings.minimum_training_rows:
        reasons.append("insufficient_training_rows")
    if activations < settings.minimum_training_activations:
        reasons.append("insufficient_activation_events")
    if probability < base_rate * settings.minimum_lift_over_base:
        reasons.append("predicted_lift_too_low")
    if interval[1] - interval[0] > settings.maximum_interval_width:
        reasons.append("uncertainty_too_high")
    return ("nominated", "") if not reasons else ("abstain", "|".join(reasons))


def _add_system_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    keys = ["period", "forecast_session", "target_window_sessions", "model_name"]
    result["probability_no_activation"] = np.nan
    result["probability_multiple_activation"] = np.nan
    result["predicted_activation_count"] = 0
    for _, index in result.groupby(keys, sort=False, observed=True).groups.items():
        positions = list(index)
        probabilities = np.clip(
            result.loc[positions, "predicted_activation_probability"].to_numpy(float),
            1e-9,
            1.0 - 1e-9,
        )
        no_activation = float(np.prod(1.0 - probabilities))
        exactly_one = float(
            sum(
                probability * np.prod(np.delete(1.0 - probabilities, item))
                for item, probability in enumerate(probabilities)
            )
        )
        result.loc[positions, "probability_no_activation"] = no_activation
        result.loc[positions, "probability_multiple_activation"] = max(
            0.0, 1.0 - no_activation - exactly_one
        )
        result.loc[positions, "predicted_activation_count"] = int(
            result.loc[positions, "prediction_state"].eq("nominated").sum()
        )
    return result


def run_prequential_rotation(
    family_states: pd.DataFrame,
    source_events: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    settings: PrequentialSettings,
) -> pd.DataFrame:
    """Run update-before-freeze online cause-specific hazards per period."""

    window_targets = targets.loc[
        targets["target_window_sessions"].eq(settings.target_window_sessions)
    ].copy()
    target_lookup = {
        (int(str(row.period)), str(row.forecast_session), str(row.destination_family)): row
        for row in window_targets.itertuples(index=False)
    }
    records: list[dict[str, object]] = []
    for period, period_states in family_states.groupby("period", sort=True, observed=True):
        graph = PastOnlyRotationGraph(settings.graph)
        families = sorted(period_states["destination_family"].astype(str).unique())
        models = {
            model_name: {family: OnlineRidgeLogit(schema, settings) for family in families}
            for model_name, schema in MODEL_SCHEMAS.items()
        }
        pending: list[_Pending] = []
        sessions = sorted(period_states["score_session"].astype(str).unique())
        period_event_rows = source_events.loc[source_events["period"].eq(period)].copy()
        for session in sessions:
            current_states = period_states.loc[
                period_states["score_session"].astype(str).eq(session)
            ].sort_values("destination_family", kind="stable")
            current_events = period_event_rows.loc[
                period_event_rows["score_session"].astype(str).eq(session)
            ].sort_values("destination_family", kind="stable")
            if current_states.empty:
                continue
            freeze = pd.Timestamp(current_states["forecast_freeze_timestamp"].iloc[0])
            matured = [item for item in pending if not item.trained]
            for item in sorted(
                matured,
                key=lambda value: (
                    value.example.label_availability_timestamp,
                    value.example.example_id,
                ),
            ):
                if item.example.label_availability_timestamp >= freeze:
                    continue
                for model_name, matured_features in item.features.items():
                    models[model_name][item.example.destination_family].update(
                        matured_features,
                        item.example.activation_target,
                        item.base_rate,
                    )
                item.trained = True
            graph.update_matured(
                [item.example for item in pending],
                as_of=freeze,
            )
            source_vector = _source_vector(current_events)
            system = _system_features(current_states, current_events)
            for untyped_row in current_states.to_dict(orient="records"):
                raw_row = {str(key): value for key, value in untyped_row.items()}
                destination = str(raw_row["destination_family"])
                base_rate = graph.base_rate(destination)
                support, activations = graph.base_counts(destination)
                own = _own_features(raw_row, base_rate)
                directed = graph.directed_features(destination, source_vector)
                all_features = {**own, **system, **directed}
                model_features = {
                    name: {feature: all_features[feature] for feature in schema}
                    for name, schema in MODEL_SCHEMAS.items()
                }
                target = target_lookup.get((int(str(period)), session, destination))
                target_available = bool(target.target_available) if target is not None else False
                target_value = (
                    bool(target.activation_target)
                    if target is not None and target_available
                    else pd.NA
                )
                for model_name in ("M0_activation_base_rate", *MODEL_SCHEMAS):
                    if model_name == "M0_activation_base_rate":
                        probability = base_rate
                        features: dict[str, float] = {}
                        schema: tuple[str, ...] = ()
                    else:
                        features = model_features[model_name]
                        schema = MODEL_SCHEMAS[model_name]
                        probability = models[model_name][destination].predict(features, base_rate)
                    interval = _shift_interval(
                        base_rate,
                        probability,
                        graph.base_interval(destination),
                    )
                    prediction_state, reasons = _prediction_state(
                        row=raw_row,
                        probability=probability,
                        base_rate=base_rate,
                        interval=interval,
                        support=support,
                        activations=activations,
                        settings=settings,
                    )
                    records.append(
                        {
                            "run_id": settings.run_id,
                            "forecast_id": _stable_id(
                                "rotation-forecast",
                                (
                                    settings.run_id,
                                    model_name,
                                    period,
                                    session,
                                    destination,
                                    settings.target_window_sessions,
                                ),
                            ),
                            "period": int(str(period)),
                            "forecast_session": session,
                            "forecast_timestamp": freeze,
                            "forecast_freeze_timestamp": freeze,
                            "feature_availability_timestamp": raw_row.get(
                                "feature_availability_timestamp"
                            ),
                            "training_cutoff": graph.latest_label_availability,
                            "destination_family": destination,
                            "destination_current_economic_state": raw_row["operational_state"],
                            "target_window_sessions": settings.target_window_sessions,
                            "model_name": model_name,
                            "predicted_activation_probability": probability,
                            "activation_base_rate": base_rate,
                            "predicted_lift_over_base": probability / max(base_rate, 1e-12),
                            "probability_interval_lower": interval[0],
                            "probability_interval_upper": interval[1],
                            "prediction_state": prediction_state,
                            "reason_codes": reasons,
                            "training_rows": support,
                            "training_activations": activations,
                            "source_family_state_vector_json": json.dumps(
                                {key: sorted(value) for key, value in source_vector.items()},
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "frozen_feature_values_json": json.dumps(
                                features, sort_keys=True, separators=(",", ":")
                            ),
                            "feature_schema_json": json.dumps(schema, separators=(",", ":")),
                            "target_available": target_available,
                            "activation_target": target_value,
                            "target_status": target.target_status
                            if target is not None
                            else "missing",
                            "label_availability_timestamp": (
                                target.label_availability_timestamp
                                if target is not None
                                else pd.NaT
                            ),
                            "first_activation_session": (
                                target.first_activation_session if target is not None else pd.NA
                            ),
                            "target_episode_ids": (
                                target.target_episode_ids if target is not None else ""
                            ),
                            "observed_activation_count": (
                                target.observed_activation_count if target is not None else 0
                            ),
                            "multiple_activation_flag": (
                                target.multiple_activation_flag if target is not None else False
                            ),
                            "no_activation_flag": (
                                target.no_activation_flag if target is not None else False
                            ),
                        }
                    )
                if target is not None and target_available:
                    example = MaturedRotationExample(
                        example_id=_stable_id(
                            "rotation-example",
                            (
                                period,
                                session,
                                destination,
                                settings.target_window_sessions,
                            ),
                        ),
                        period=int(str(period)),
                        forecast_session=session,
                        destination_family=destination,
                        activation_target=bool(target.activation_target),
                        label_availability_timestamp=pd.Timestamp(
                            str(target.label_availability_timestamp)
                        ),
                        source_events=source_vector,
                    )
                    pending.append(
                        _Pending(
                            example=example,
                            base_rate=base_rate,
                            features=model_features,
                        )
                    )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    result["activation_target"] = result["activation_target"].astype("boolean")
    result = _add_system_probabilities(result)
    return result.sort_values(
        ["period", "forecast_session", "destination_family", "model_name"], kind="stable"
    ).reset_index(drop=True)


def shift_source_events(events: pd.DataFrame, *, sessions: int) -> pd.DataFrame:
    """Apply the registered intentionally wrong source lag inside each period/family."""

    if sessions < 1:
        raise ValueError("wrong lag must be positive")
    result = events.copy().sort_values(
        ["period", "destination_family", "score_session"], kind="stable"
    )
    columns = [
        "source_active",
        "newly_active",
        "newly_decaying",
        "newly_retired",
        "source_state_transition",
    ]
    shifted = result.groupby(["period", "destination_family"], sort=False, observed=True)[
        columns
    ].shift(sessions)
    result.loc[:, columns] = shifted.to_numpy(dtype=bool, na_value=False)
    return result.reset_index(drop=True)


def permute_source_events(events: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Permute source-state histories within periods without consulting any target."""

    result = events.copy().sort_values(
        ["period", "destination_family", "score_session"], kind="stable"
    )
    columns = [
        "source_active",
        "newly_active",
        "newly_decaying",
        "newly_retired",
        "source_state_transition",
    ]
    rng = np.random.default_rng(seed)
    for _, index in result.groupby(
        ["period", "destination_family"], sort=True, observed=True
    ).groups.items():
        positions = np.asarray(list(index), dtype=int)
        order = rng.permutation(len(positions))
        values = result.loc[positions, columns].to_numpy(copy=True)
        result.loc[positions, columns] = values[order]
    return result.reset_index(drop=True)


__all__ = [
    "MODEL_SCHEMAS",
    "OnlineRidgeLogit",
    "PrequentialSettings",
    "permute_source_events",
    "run_prequential_rotation",
    "shift_source_events",
]
