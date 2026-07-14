"""Explicit chronological settlement, forecast freezing, and admission joins."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from stocker_research.dynamic_loop_edge_state.decision import (
    DecisionThresholds,
    classify_edge_state,
)
from stocker_research.dynamic_loop_edge_state.online_state import (
    BOCPDSettings,
    HierarchicalPayoffModel,
    HierarchicalSettings,
    PayoffObservation,
)


@dataclass(frozen=True)
class WalkForwardSettings:
    run_id: str
    model_name: str
    model_version: str
    configuration_hash: str
    feature_schema_version: str
    cost_model_version: str
    horizon_bars: int
    session_bars: int
    required_features: tuple[str, ...] = ()
    include_leading_features: bool = True
    random_seed: int = 20260714


@dataclass
class _Moments:
    count: int = 0
    mean: float = 0.0
    squared_deviation: float = 0.0


class ExpandingFeatureScaler:
    """Past-only Welford standardisation used before updating on the current row."""

    def __init__(self, feature_names: tuple[str, ...]) -> None:
        self.feature_names = feature_names
        self._moments = {name: _Moments() for name in feature_names}

    def transform(self, values: dict[str, float]) -> tuple[dict[str, float], float]:
        transformed: dict[str, float] = {}
        distances: list[float] = []
        for name in self.feature_names:
            value = float(values[name])
            moments = self._moments[name]
            if moments.count < 2:
                z_score = 0.0
            else:
                variance = moments.squared_deviation / (moments.count - 1)
                z_score = (value - moments.mean) / math.sqrt(max(variance, 1e-12))
            transformed[name] = float(np.clip(z_score, -10.0, 10.0))
            distances.append(z_score**2)
        ood = math.sqrt(float(np.mean(distances))) if distances else 0.0
        return transformed, ood

    def update(self, values: dict[str, float]) -> None:
        for name in self.feature_names:
            value = float(values[name])
            moments = self._moments[name]
            moments.count += 1
            difference = value - moments.mean
            moments.mean += difference / moments.count
            moments.squared_deviation += difference * (value - moments.mean)


def _validate_calendar(session_calendar: pd.DataFrame) -> pd.DataFrame:
    required = {"score_session", "decision_timestamp"}
    missing = sorted(required - set(session_calendar.columns))
    if missing:
        raise ValueError(f"missing calendar columns: {missing}")
    calendar = session_calendar.loc[:, ["score_session", "decision_timestamp"]].copy()
    calendar["score_session"] = calendar["score_session"].astype(str)
    calendar["decision_timestamp"] = pd.to_datetime(
        calendar["decision_timestamp"], utc=True, errors="raise"
    )
    calendar = calendar.sort_values("decision_timestamp", kind="stable").reset_index(drop=True)
    if calendar["score_session"].duplicated().any():
        raise ValueError("duplicate score session")
    if not calendar["decision_timestamp"].is_monotonic_increasing:
        raise ValueError("decision timestamps are not chronological")
    return calendar


def _validate_panel(payoff_panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session",
        "loop_id",
        "orientation",
        "horizon",
        "robust_net_payoff_bps",
        "effective_sample_size",
        "independent_stock_ids",
        "raw_fill_count",
        "data_availability_timestamp",
    }
    missing = sorted(required - set(payoff_panel.columns))
    if missing:
        raise ValueError(f"missing payoff-panel columns: {missing}")
    panel = payoff_panel.copy()
    panel["session"] = panel["session"].astype(str)
    panel["data_availability_timestamp"] = pd.to_datetime(
        panel["data_availability_timestamp"], utc=True, errors="raise"
    )
    keys = ["session", "loop_id", "orientation", "horizon"]
    if panel.duplicated(keys).any():
        raise ValueError("duplicate session payoff cell")
    return panel.sort_values(keys, kind="stable").reset_index(drop=True)


def _feature_lookup(
    feature_panel: pd.DataFrame,
    required_features: tuple[str, ...],
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    required = {
        "score_session",
        "loop_id",
        "orientation",
        "horizon",
        "feature_availability_timestamp",
        *required_features,
    }
    missing = sorted(required - set(feature_panel.columns))
    if missing:
        raise ValueError(f"missing feature-panel columns: {missing}")
    frame = feature_panel.copy()
    frame["score_session"] = frame["score_session"].astype(str)
    frame["feature_availability_timestamp"] = pd.to_datetime(
        frame["feature_availability_timestamp"], utc=True, errors="raise"
    )
    keys = ["score_session", "loop_id", "orientation", "horizon"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate scoring feature cell")
    rows: list[dict[str, Any]] = [
        {str(key): value for key, value in raw_row.items()}
        for raw_row in frame.to_dict(orient="records")
    ]
    return {
        (
            str(row["score_session"]),
            str(row["loop_id"]),
            str(row["orientation"]),
            int(row["horizon"]),
        ): row
        for row in rows
    }


def _parse_stocks(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, (list, tuple, np.ndarray)):
        parsed = list(value)
    else:
        raise ValueError("independent_stock_ids must be JSON or a sequence")
    return tuple(sorted({str(item) for item in parsed}))


def run_causal_walk_forward(
    *,
    session_calendar: pd.DataFrame,
    payoff_panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    cell_keys: list[tuple[str, str, int]],
    bocpd_settings: BOCPDSettings,
    hierarchy_settings: HierarchicalSettings,
    decision_thresholds: DecisionThresholds,
    settings: WalkForwardSettings,
) -> pd.DataFrame:
    """Score sessions in explicit update → feature → freeze order."""

    calendar = _validate_calendar(session_calendar)
    panel = _validate_panel(payoff_panel)
    lookup = _feature_lookup(feature_panel, settings.required_features)
    model = HierarchicalPayoffModel(bocpd_settings, hierarchy_settings)
    scaler = ExpandingFeatureScaler(settings.required_features)
    updated_sessions: set[str] = set()
    previous_states = {key: "unknown" for key in cell_keys}
    settled_observation_count = 0
    latest_availability: pd.Timestamp | None = None
    latest_source_session: str | None = None
    rows: list[dict[str, object]] = []
    metadata = {
        "run_id": settings.run_id,
        "model_name": settings.model_name,
        "model_version": settings.model_version,
        "configuration_hash": settings.configuration_hash,
        "feature_schema_version": settings.feature_schema_version,
        "cost_model_version": settings.cost_model_version,
        "fixed_horizon_bars": settings.horizon_bars,
        "random_seed": settings.random_seed,
    }
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))

    for calendar_row in calendar.to_dict(orient="records"):
        score_session = str(calendar_row["score_session"])
        decision_timestamp = pd.Timestamp(calendar_row["decision_timestamp"])
        pending_sessions = sorted(
            session
            for session in panel.loc[panel["session"].lt(score_session), "session"].unique()
            if session not in updated_sessions
        )
        blocked_by_unresolved = False
        for source_session in pending_sessions:
            session_rows = panel.loc[panel["session"].eq(source_session)].copy()
            if not session_rows["data_availability_timestamp"].lt(decision_timestamp).all():
                blocked_by_unresolved = True
                break
            observations = [
                PayoffObservation(
                    cell_key=(
                        str(row["loop_id"]),
                        str(row["orientation"]),
                        int(row["horizon"]),
                    ),
                    session=source_session,
                    net_payoff_bps=float(row["robust_net_payoff_bps"]),
                    effective_sample_size=float(row["effective_sample_size"]),
                    independent_stocks=_parse_stocks(row["independent_stock_ids"]),
                    raw_fills=int(row["raw_fill_count"]),
                    availability_timestamp=pd.Timestamp(row["data_availability_timestamp"]),
                )
                for row in session_rows.to_dict(orient="records")
            ]
            model.update_session(source_session, observations)
            updated_sessions.add(source_session)
            settled_observation_count += len(observations)
            session_latest = session_rows["data_availability_timestamp"].max()
            latest_availability = (
                session_latest
                if latest_availability is None
                else max(latest_availability, session_latest)
            )
            latest_source_session = source_session

        eligible_feature_updates: list[dict[str, float]] = []
        for cell_key in sorted(cell_keys):
            feature_key = (score_session, cell_key[0], cell_key[1], cell_key[2])
            feature_row = lookup.get(feature_key)
            feature_values: dict[str, float] = {}
            feature_timestamp: pd.Timestamp | None = None
            features_available = not settings.required_features
            if feature_row is not None:
                candidate_timestamp = pd.Timestamp(feature_row["feature_availability_timestamp"])
                candidate_values = {
                    name: float(feature_row[name]) for name in settings.required_features
                }
                features_available = candidate_timestamp <= decision_timestamp and all(
                    math.isfinite(value) for value in candidate_values.values()
                )
                if features_available:
                    feature_values = candidate_values
                    feature_timestamp = candidate_timestamp
                    eligible_feature_updates.append(candidate_values)
            if features_available and settings.required_features:
                transformed, ood_score = scaler.transform(feature_values)
                transformed["out_of_distribution_score"] = ood_score
            else:
                transformed, ood_score = {}, 0.0
            forecast, support = model.forecast(
                cell_key,
                horizon_bars=settings.horizon_bars,
                session_bars=settings.session_bars,
                leading_features=transformed,
                out_of_distribution_score=ood_score,
                include_leading_features=settings.include_leading_features,
            )
            decision = classify_edge_state(
                forecast,
                support,
                decision_thresholds,
                previous_state=previous_states[cell_key],
                unresolved_outcomes=blocked_by_unresolved,
                required_features_available=features_available,
            )
            previous_states[cell_key] = decision.edge_state
            row: dict[str, object] = {
                "loop_id": cell_key[0],
                "orientation": cell_key[1],
                "score_session": score_session,
                "decision_timestamp": decision_timestamp,
                "horizon": cell_key[2],
                **asdict(forecast),
                "effective_sessions": support.effective_sessions,
                "independent_stocks": support.independent_stocks,
                "raw_fills": support.raw_fills,
                "effective_sample_size": support.effective_sample_size,
                "edge_state": decision.edge_state,
                "admit_new_entry": decision.admit_new_entry,
                "reason_codes": "|".join(decision.reason_codes),
                "existing_position_action": decision.existing_position_action,
                "required_features_available": bool(features_available),
                "feature_max_availability_timestamp": feature_timestamp,
                "settled_observation_count": settled_observation_count,
                "training_latest_source_session": latest_source_session,
                "training_latest_availability_timestamp": latest_availability,
                "prediction_frozen_at": decision_timestamp,
                "run_id": settings.run_id,
                "model_name": settings.model_name,
                "model_version": settings.model_version,
                "configuration_hash": settings.configuration_hash,
                "feature_schema_version": settings.feature_schema_version,
                "cost_model_version": settings.cost_model_version,
                "run_metadata_json": metadata_json,
            }
            for feature_name in settings.required_features:
                row[f"z__{feature_name}"] = transformed.get(feature_name, math.nan)
            rows.append(row)
        for values in eligible_feature_updates:
            scaler.update(values)
    return (
        pd.DataFrame(rows)
        .sort_values(["score_session", "loop_id", "orientation", "horizon"], kind="stable")
        .reset_index(drop=True)
    )


def apply_frozen_admission(
    opportunities: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Join a frozen session forecast to new entries without touching exits."""

    opportunity_required = {
        "opportunity_id",
        "score_session",
        "loop_id",
        "orientation",
        "horizon",
        "opportunity_decision_timestamp",
    }
    forecast_required = {
        "score_session",
        "loop_id",
        "orientation",
        "horizon",
        "decision_timestamp",
        "admit_new_entry",
        "edge_state",
        "reason_codes",
        "run_id",
        "model_name",
    }
    missing_opportunity = sorted(opportunity_required - set(opportunities.columns))
    missing_forecast = sorted(forecast_required - set(forecasts.columns))
    if missing_opportunity or missing_forecast:
        raise ValueError(
            f"missing admission columns: opportunities={missing_opportunity}, "
            f"forecasts={missing_forecast}"
        )
    keys = ["score_session", "loop_id", "orientation", "horizon"]
    forecast_columns = [
        *keys,
        "decision_timestamp",
        "admit_new_entry",
        "edge_state",
        "reason_codes",
        "run_id",
        "model_name",
        "configuration_hash",
    ]
    joined = opportunities.merge(
        forecasts.loc[:, forecast_columns],
        on=keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_forecast"),
    )
    joined["opportunity_decision_timestamp"] = pd.to_datetime(
        joined["opportunity_decision_timestamp"], utc=True, errors="raise"
    )
    joined["decision_timestamp"] = pd.to_datetime(
        joined["decision_timestamp"], utc=True, errors="coerce"
    )
    invalid_clock = joined["decision_timestamp"].notna() & joined["decision_timestamp"].gt(
        joined["opportunity_decision_timestamp"]
    )
    if invalid_clock.any():
        raise ValueError("forecast was not frozen before opportunity decision")
    missing = joined["decision_timestamp"].isna()
    joined["accepted"] = joined["admit_new_entry"].fillna(False).astype(bool)
    joined["decision"] = np.where(joined["accepted"], "accepted", "rejected")
    joined.loc[missing, "edge_state"] = "unknown"
    joined.loc[missing, "reason_codes"] = "missing_features"
    joined["forecast_frozen_before_payoff"] = ~missing
    joined["existing_position_action"] = "unchanged_existing_exit_rule"
    return joined


__all__ = [
    "ExpandingFeatureScaler",
    "WalkForwardSettings",
    "apply_frozen_admission",
    "run_causal_walk_forward",
]
