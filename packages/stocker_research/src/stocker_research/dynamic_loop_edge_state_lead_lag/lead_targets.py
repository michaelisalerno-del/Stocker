"""Immutable forecast records and explicit-calendar payoff lead joins."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

CELL_COLUMNS = ("period", "loop_id", "orientation", "horizon")
FORECAST_KEY_COLUMNS = (
    "model_name",
    "period",
    "score_session",
    "loop_id",
    "orientation",
    "horizon",
)
OUTCOME_KEY_COLUMNS = ("period", "session", "loop_id", "orientation", "horizon")
REGISTERED_LEADS = (0, 1, 2, 3, 5)


def _stable_id(prefix: str, values: Sequence[object]) -> str:
    payload = json.dumps([str(value) for value in values], separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


@dataclass(frozen=True)
class LeadRegistration:
    leads: tuple[int, ...] = REGISTERED_LEADS
    primary_lead: int = 1

    def __post_init__(self) -> None:
        if self.leads != REGISTERED_LEADS:
            raise ValueError(f"registered leads must remain {REGISTERED_LEADS}")
        if self.primary_lead != 1:
            raise ValueError("primary lead must remain one session")


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def build_frozen_forecast_ledger(
    forecasts: pd.DataFrame,
    metadata: Mapping[str, object],
) -> pd.DataFrame:
    """Copy V2 forecasts into immutable, deterministic attribution records."""

    required = {
        *FORECAST_KEY_COLUMNS,
        "prediction_frozen_at",
        "decision_timestamp",
        "feature_max_availability_timestamp",
        "training_latest_availability_timestamp",
        "p_next_payoff_positive",
        "p_edge_positive",
        "p_edge_active",
        "p_change_now",
        "p_on_next",
        "p_off_next",
        "p_survive_horizon",
        "posterior_mean_net_bps",
        "posterior_lower_bound_net_bps",
        "posterior_run_length_mean",
        "edge_state",
        "reason_codes",
        "effective_sessions",
        "independent_stocks",
        "effective_sample_size",
        "run_id",
        "model_version",
        "configuration_hash",
        "feature_schema_version",
    }
    _require(forecasts, required, "forecast")
    required_metadata = {
        "run_id",
        "git_sha",
        "contract_hash",
        "data_snapshot_hash",
        "experiment_version",
    }
    missing_metadata = sorted(required_metadata - set(metadata))
    if missing_metadata:
        raise ValueError(f"missing forecast metadata: {missing_metadata}")
    frame = forecasts.copy()
    if frame.duplicated(list(FORECAST_KEY_COLUMNS)).any():
        raise ValueError("duplicate frozen forecast key")
    freeze = pd.to_datetime(frame["prediction_frozen_at"], utc=True, errors="raise")
    decision = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="raise")
    feature_time = pd.to_datetime(
        frame["feature_max_availability_timestamp"], utc=True, errors="coerce"
    )
    training_time = pd.to_datetime(
        frame["training_latest_availability_timestamp"], utc=True, errors="coerce"
    )
    if not freeze.eq(decision).all():
        raise ValueError("forecast freeze differs from decision timestamp")
    if (feature_time.notna() & feature_time.ge(freeze)).any():
        raise ValueError("feature availability is not strictly before forecast freeze")
    if (training_time.notna() & training_time.ge(freeze)).any():
        raise ValueError("training availability is not strictly before forecast freeze")
    frame["source_run_id"] = frame["run_id"].astype(str)
    frame["run_id"] = str(metadata["run_id"])
    frame["git_sha"] = str(metadata["git_sha"])
    frame["contract_hash"] = str(metadata["contract_hash"])
    frame["data_snapshot_hash"] = str(metadata["data_snapshot_hash"])
    frame["experiment_version"] = str(metadata["experiment_version"])
    frame["forecast_effective_session"] = frame["score_session"].astype(str)
    frame["forecast_creation_timestamp"] = freeze
    frame["forecast_freeze_timestamp"] = freeze
    frame["stock_id"] = pd.NA
    frame["independent_session_support"] = frame["effective_sessions"].astype(float)
    frame["independent_stock_support"] = frame["independent_stocks"].astype(int)
    feature_columns = sorted(
        str(column) for column in frame.columns if str(column).startswith("z__")
    )
    feature_records = frame.loc[:, feature_columns].to_dict(orient="records")
    frame["frozen_feature_values_json"] = [
        json.dumps(
            {name: (None if pd.isna(value) else float(value)) for name, value in row.items()},
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in feature_records
    ]
    frame["feature_availability_timestamps_json"] = [
        json.dumps(
            {
                name.removeprefix("z__"): (
                    None if pd.isna(row[name]) else pd.Timestamp(availability).isoformat()
                )
                for name in feature_columns
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for row, availability in zip(
            feature_records, frame["feature_max_availability_timestamp"], strict=True
        )
    ]
    frame["forecast_id"] = [
        _stable_id(
            "forecast",
            (
                source_run_id,
                model_name,
                period,
                score_session,
                loop_id,
                orientation,
                horizon,
            ),
        )
        for (
            source_run_id,
            model_name,
            period,
            score_session,
            loop_id,
            orientation,
            horizon,
        ) in frame[["source_run_id", *FORECAST_KEY_COLUMNS]].itertuples(index=False, name=None)
    ]
    if not frame["forecast_id"].is_unique:
        raise ValueError("forecast IDs are not unique")
    return frame.sort_values(list(FORECAST_KEY_COLUMNS), kind="stable").reset_index(drop=True)


def build_settled_outcome_ledger(
    payoff_panel: pd.DataFrame,
    metadata: Mapping[str, object],
) -> pd.DataFrame:
    """Create one immutable outcome record per settled session/cell observation."""

    required = {
        *OUTCOME_KEY_COLUMNS,
        "robust_net_payoff_bps",
        "robust_gross_payoff_bps",
        "cost_contribution_bps",
        "independent_stock_count",
        "independent_stock_ids",
        "effective_sample_size",
        "data_availability_timestamp",
        "source_data_id",
    }
    _require(payoff_panel, required, "outcome")
    frame = payoff_panel.copy()
    if frame.empty:
        for column in (
            "outcome_id",
            "run_id",
            "contract_hash",
            "data_snapshot_hash",
            "target_payoff_positive",
        ):
            frame[column] = pd.Series(dtype="object")
        return frame
    if frame.duplicated(list(OUTCOME_KEY_COLUMNS)).any():
        raise ValueError("duplicate settled outcome key")
    availability = pd.to_datetime(frame["data_availability_timestamp"], utc=True, errors="raise")
    if frame["robust_net_payoff_bps"].isna().any():
        raise ValueError("settled outcome payoff cannot be missing")
    frame["data_availability_timestamp"] = availability
    frame["target_payoff_positive"] = frame["robust_net_payoff_bps"].gt(0.0)
    frame["run_id"] = str(metadata["run_id"])
    frame["contract_hash"] = str(metadata["contract_hash"])
    frame["data_snapshot_hash"] = str(metadata["data_snapshot_hash"])
    frame["outcome_id"] = [
        _stable_id("outcome", (*key, source_data_id))
        for *key, source_data_id in frame[[*OUTCOME_KEY_COLUMNS, "source_data_id"]].itertuples(
            index=False, name=None
        )
    ]
    if not frame["outcome_id"].is_unique:
        raise ValueError("outcome IDs are not unique")
    return frame.sort_values(list(OUTCOME_KEY_COLUMNS), kind="stable").reset_index(drop=True)


def _calendar_lead_map(
    cell_calendar: pd.DataFrame,
    registration: LeadRegistration,
) -> pd.DataFrame:
    _require(cell_calendar, {"period", "score_session", *CELL_COLUMNS[1:]}, "calendar")
    sessions = (
        cell_calendar.loc[:, ["period", "score_session"]]
        .drop_duplicates()
        .sort_values(["period", "score_session"], kind="stable")
    )
    rows: list[pd.DataFrame] = []
    for lead in registration.leads:
        shifted = sessions.copy()
        shifted["target_lead_sessions"] = lead
        shifted["target_session"] = shifted.groupby("period", observed=True)["score_session"].shift(
            -lead
        )
        rows.append(shifted)
    return pd.concat(rows, ignore_index=True)


def _opportunity_summary(opportunities: pd.DataFrame) -> pd.DataFrame:
    keys = ["period", "score_session", "loop_id", "orientation", "horizon"]
    _require(opportunities, {*keys, "status", "settlement_timestamp"}, "opportunity")
    if opportunities.empty:
        return pd.DataFrame(
            columns=[
                *keys,
                "target_opportunity_count",
                "target_filled_opportunity_count",
                "target_max_settlement_timestamp",
            ]
        )
    frame = opportunities.copy()
    frame["settlement_timestamp"] = pd.to_datetime(
        frame["settlement_timestamp"], utc=True, errors="coerce"
    )
    return (
        frame.groupby(keys, sort=True, observed=True)
        .agg(
            target_opportunity_count=("status", "size"),
            target_filled_opportunity_count=("status", lambda values: values.eq("filled").sum()),
            target_max_settlement_timestamp=("settlement_timestamp", "max"),
        )
        .reset_index()
    )


def build_lead_target_joins(
    forecasts: pd.DataFrame,
    outcomes: pd.DataFrame,
    cell_calendar: pd.DataFrame,
    opportunities: pd.DataFrame,
    registration: LeadRegistration,
    *,
    snapshot_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Join every immutable forecast to exact calendar-lead session evidence."""

    _require(forecasts, {"forecast_id", *FORECAST_KEY_COLUMNS}, "frozen forecast")
    lead_map = _calendar_lead_map(cell_calendar, registration)
    expanded = forecasts.merge(
        lead_map,
        on=["period", "score_session"],
        how="left",
        validate="many_to_many",
    )
    available = cell_calendar.loc[
        :, ["period", "score_session", "loop_id", "orientation", "horizon"]
    ].drop_duplicates()
    available["target_cell_available"] = True
    expanded = expanded.merge(
        available.rename(columns={"score_session": "target_session"}),
        on=["period", "target_session", "loop_id", "orientation", "horizon"],
        how="left",
        validate="many_to_one",
    )
    outcome_columns = [
        "period",
        "session",
        "loop_id",
        "orientation",
        "horizon",
        "outcome_id",
        "robust_net_payoff_bps",
        "robust_gross_payoff_bps",
        "cost_contribution_bps",
        "target_payoff_positive",
        "independent_stock_count",
        "independent_stock_ids",
        "effective_sample_size",
        "data_availability_timestamp",
    ]
    expanded = expanded.merge(
        outcomes.loc[:, outcome_columns].rename(
            columns={
                "session": "target_session",
                "outcome_id": "target_outcome_id",
                "robust_net_payoff_bps": "target_robust_net_bps",
                "robust_gross_payoff_bps": "target_robust_gross_bps",
                "cost_contribution_bps": "target_cost_contribution_bps",
                "independent_stock_count": "target_independent_stocks",
                "independent_stock_ids": "target_independent_stock_ids",
                "effective_sample_size": "target_effective_sample_size",
                "data_availability_timestamp": "target_payoff_availability_timestamp",
            }
        ),
        on=["period", "target_session", "loop_id", "orientation", "horizon"],
        how="left",
        validate="many_to_one",
    )
    summary = _opportunity_summary(opportunities).rename(
        columns={"score_session": "target_session"}
    )
    expanded = expanded.merge(
        summary,
        on=["period", "target_session", "loop_id", "orientation", "horizon"],
        how="left",
        validate="many_to_one",
    )
    expanded["target_opportunity_count"] = (
        pd.to_numeric(expanded["target_opportunity_count"], errors="coerce")
        .astype("Int64")
        .fillna(0)
        .astype(int)
    )
    expanded["target_filled_opportunity_count"] = (
        pd.to_numeric(expanded["target_filled_opportunity_count"], errors="coerce")
        .astype("Int64")
        .fillna(0)
        .astype(int)
    )
    expanded["target_status"] = "missing_source_data"
    boundary = expanded["target_session"].isna()
    unavailable = ~boundary & expanded["target_cell_available"].ne(True)
    settled = expanded["target_outcome_id"].notna()
    no_opportunity = (
        ~boundary & ~unavailable & ~settled & expanded["target_opportunity_count"].eq(0)
    )
    unfilled = (
        ~boundary
        & ~unavailable
        & ~settled
        & expanded["target_opportunity_count"].gt(0)
        & expanded["target_filled_opportunity_count"].eq(0)
    )
    unresolved = pd.Series(False, index=expanded.index)
    if snapshot_timestamp is not None:
        snapshot = pd.Timestamp(snapshot_timestamp)
        if snapshot.tzinfo is None:
            raise ValueError("snapshot timestamp must be timezone-aware")
        settlement = pd.to_datetime(
            expanded["target_max_settlement_timestamp"], utc=True, errors="coerce"
        )
        unresolved = ~settled & settlement.gt(snapshot)
    expanded.loc[boundary, "target_status"] = "period_boundary"
    expanded.loc[unavailable, "target_status"] = "cell_unavailable"
    expanded.loc[no_opportunity, "target_status"] = "no_opportunity"
    expanded.loc[unfilled, "target_status"] = "opportunity_unfilled_no_payoff"
    expanded.loc[unresolved, "target_status"] = "unresolved_at_snapshot"
    expanded.loc[settled, "target_status"] = "payoff_settled"
    expanded["target_payoff_available"] = expanded["target_status"].eq("payoff_settled")
    expanded["target_payoff_positive"] = expanded["target_payoff_positive"].astype("boolean")
    expanded["target_episode_state"] = pd.NA
    expanded["target_episode_id"] = pd.NA
    expanded["target_episode_onset_within_lead"] = pd.NA
    expanded["target_episode_survival"] = pd.NA
    expanded["lead_join_id"] = [
        _stable_id("lead", (forecast_id, lead, target_session, status))
        for forecast_id, lead, target_session, status in expanded[
            ["forecast_id", "target_lead_sessions", "target_session", "target_status"]
        ].itertuples(index=False, name=None)
    ]
    if not expanded["lead_join_id"].is_unique:
        raise ValueError("lead join IDs are not unique")
    return expanded.sort_values(
        [*FORECAST_KEY_COLUMNS, "target_lead_sessions"], kind="stable"
    ).reset_index(drop=True)


__all__ = [
    "LeadRegistration",
    "build_frozen_forecast_ledger",
    "build_lead_target_joins",
    "build_settled_outcome_ledger",
]
