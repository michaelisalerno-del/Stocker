"""Explicit-calendar future family activation targets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REGISTERED_WINDOWS = (1, 3, 5)


@dataclass(frozen=True)
class ActivationRegistration:
    windows: tuple[int, ...] = REGISTERED_WINDOWS
    primary_window: int = 3

    def __post_init__(self) -> None:
        if self.windows != REGISTERED_WINDOWS:
            raise ValueError(f"activation windows must remain {REGISTERED_WINDOWS}")
        if self.primary_window != 3:
            raise ValueError("primary activation window must remain three sessions")


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def build_activation_targets(
    *,
    forecast_states: pd.DataFrame,
    calendar: pd.DataFrame,
    episode_intervals: pd.DataFrame,
    payoff_support: pd.DataFrame,
    registration: ActivationRegistration,
) -> pd.DataFrame:
    """Attach future onsets without exposing them to the frozen forecast feature set."""

    _require(
        forecast_states,
        {
            "period",
            "score_session",
            "destination_family",
            "operational_state",
            "forecast_freeze_timestamp",
        },
        "forecast-state",
    )
    _require(calendar, {"period", "score_session"}, "calendar")
    _require(
        episode_intervals,
        {
            "period",
            "destination_family",
            "episode_id",
            "episode_onset_session",
            "episode_end_session",
            "label_availability_timestamp",
        },
        "episode",
    )
    _require(
        payoff_support,
        {"period", "session", "destination_family", "data_availability_timestamp"},
        "payoff-support",
    )
    periods: dict[int, list[str]] = {
        int(str(period)): group["score_session"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
        for period, group in calendar.groupby("period", sort=True, observed=True)
    }
    indices = {
        (period, session): index
        for period, sessions in periods.items()
        for index, session in enumerate(sessions)
    }
    episodes = episode_intervals.copy()
    episodes["episode_onset_session"] = episodes["episode_onset_session"].astype(str)
    episodes["label_availability_timestamp"] = pd.to_datetime(
        episodes["label_availability_timestamp"], utc=True, errors="coerce"
    )
    support = payoff_support.copy()
    support["session"] = support["session"].astype(str)
    support["data_availability_timestamp"] = pd.to_datetime(
        support["data_availability_timestamp"], utc=True, errors="coerce"
    )
    rows: list[dict[str, object]] = []
    for forecast in forecast_states.to_dict(orient="records"):
        period = int(forecast["period"])
        session = str(forecast["score_session"])
        family = str(forecast["destination_family"])
        current_state = str(forecast["operational_state"])
        session_index = indices.get((period, session))
        for window in registration.windows:
            record: dict[str, object] = {
                "period": period,
                "forecast_session": session,
                "destination_family": family,
                "destination_current_state": current_state,
                "forecast_freeze_timestamp": pd.Timestamp(forecast["forecast_freeze_timestamp"]),
                "target_window_sessions": window,
                "target_start_session": pd.NA,
                "target_end_session": pd.NA,
                "target_status": "period_boundary",
                "target_available": False,
                "activation_target": pd.NA,
                "first_activation_session": pd.NA,
                "target_episode_ids": "",
                "target_support_sessions": 0,
                "label_availability_timestamp": pd.NaT,
            }
            sessions = periods.get(period, [])
            if session_index is None or session_index + window >= len(sessions):
                rows.append(record)
                continue
            targets = sessions[session_index + 1 : session_index + window + 1]
            record["target_start_session"] = targets[0]
            record["target_end_session"] = targets[-1]
            if current_state == "active":
                record["target_status"] = "current_active_not_candidate"
                rows.append(record)
                continue
            support_rows = support.loc[
                support["period"].eq(period)
                & support["destination_family"].eq(family)
                & support["session"].isin(targets)
            ]
            record["target_support_sessions"] = int(support_rows["session"].nunique())
            if support_rows.empty:
                record["target_status"] = "insufficient_future_support"
                rows.append(record)
                continue
            onsets = episodes.loc[
                episodes["period"].eq(period)
                & episodes["destination_family"].eq(family)
                & episodes["episode_onset_session"].isin(targets)
            ].sort_values("episode_onset_session", kind="stable")
            activation = not onsets.empty
            record["target_status"] = "activation_observed" if activation else "no_activation"
            record["target_available"] = True
            record["activation_target"] = activation
            record["label_availability_timestamp"] = support_rows[
                "data_availability_timestamp"
            ].max()
            if activation:
                record["first_activation_session"] = str(onsets["episode_onset_session"].iloc[0])
                record["target_episode_ids"] = "|".join(
                    onsets["episode_id"].astype(str).drop_duplicates().sort_values()
                )
                record["label_availability_timestamp"] = max(
                    pd.Timestamp(str(record["label_availability_timestamp"])),
                    pd.Timestamp(onsets["label_availability_timestamp"].max()),
                )
            rows.append(record)
    result = pd.DataFrame.from_records(rows)
    if result.empty:
        return result
    result["activation_target"] = result["activation_target"].astype("boolean")
    keys = ["period", "forecast_session", "target_window_sessions"]
    available_positive = result["target_available"] & result["activation_target"].fillna(False)
    result["observed_activation_count"] = (
        available_positive.astype(int)
        .groupby([result[key] for key in keys], sort=False)
        .transform("sum")
    )
    result["multiple_activation_flag"] = result["observed_activation_count"].gt(1)
    group_available = (
        result["target_available"]
        .astype(int)
        .groupby([result[key] for key in keys], sort=False)
        .transform("sum")
    )
    result["no_activation_flag"] = group_available.gt(0) & result["observed_activation_count"].eq(0)
    return result.sort_values(
        ["period", "forecast_session", "target_window_sessions", "destination_family"],
        kind="stable",
    ).reset_index(drop=True)


__all__ = ["ActivationRegistration", "build_activation_targets"]
