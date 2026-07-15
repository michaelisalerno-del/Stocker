"""Hindsight-only episode targets and descriptive lead attribution."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

CELL = ("period", "loop_id", "orientation", "horizon")


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def _calendar_positions(calendar: pd.DataFrame) -> pd.DataFrame:
    sessions = (
        calendar.loc[:, ["period", "score_session"]]
        .drop_duplicates()
        .sort_values(["period", "score_session"], kind="stable")
    )
    sessions["calendar_session_index"] = sessions.groupby("period", observed=True).cumcount()
    return sessions


def attach_hindsight_episode_targets(
    joins: pd.DataFrame,
    episode_states: pd.DataFrame,
    episodes: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Attach evaluation-only episode labels after forecasts have been frozen."""

    _require(
        joins,
        {*CELL, "score_session", "target_session", "target_lead_sessions"},
        "lead join",
    )
    _require(
        episode_states,
        {*CELL, "score_session", "hindsight_payoff_state"},
        "episode state",
    )
    _require(
        episodes,
        {
            *CELL,
            "episode_id",
            "hindsight_estimated_onset",
            "hindsight_estimated_end",
        },
        "episode",
    )
    result = joins.drop(
        columns=[
            "target_episode_state",
            "target_episode_onset_within_lead",
            "target_episode_survival",
        ],
        errors="ignore",
    ).copy()
    states = episode_states.loc[:, [*CELL, "score_session", "hindsight_payoff_state"]].rename(
        columns={
            "score_session": "target_session",
            "hindsight_payoff_state": "target_episode_state",
        }
    )
    result = result.merge(
        states,
        on=[*CELL, "target_session"],
        how="left",
        validate="many_to_one",
    )
    positions = _calendar_positions(calendar)
    forecast_positions = positions.rename(
        columns={
            "score_session": "score_session",
            "calendar_session_index": "forecast_calendar_index",
        }
    )
    target_positions = positions.rename(
        columns={
            "score_session": "target_session",
            "calendar_session_index": "target_calendar_index",
        }
    )
    result = result.merge(
        forecast_positions,
        on=["period", "score_session"],
        how="left",
        validate="many_to_one",
    ).merge(
        target_positions,
        on=["period", "target_session"],
        how="left",
        validate="many_to_one",
    )
    result["target_episode_onset_within_lead"] = False
    result["target_episode_survival"] = False
    result["target_episode_id"] = pd.Series(pd.NA, index=result.index, dtype="string")

    indexed_episodes = episodes.merge(
        positions.rename(
            columns={
                "score_session": "hindsight_estimated_onset",
                "calendar_session_index": "episode_onset_index",
            }
        ),
        on=["period", "hindsight_estimated_onset"],
        how="left",
        validate="many_to_one",
    ).merge(
        positions.rename(
            columns={
                "score_session": "hindsight_estimated_end",
                "calendar_session_index": "episode_end_index",
            }
        ),
        on=["period", "hindsight_estimated_end"],
        how="left",
        validate="many_to_one",
    )
    for raw_episode in indexed_episodes.itertuples(index=False):
        episode: Any = raw_episode
        cell = (
            result["period"].eq(episode.period)
            & result["loop_id"].eq(episode.loop_id)
            & result["orientation"].eq(episode.orientation)
            & result["horizon"].eq(episode.horizon)
        )
        lead_zero = result["target_lead_sessions"].eq(0)
        onset = cell & (
            (lead_zero & result["forecast_calendar_index"].eq(episode.episode_onset_index))
            | (
                ~lead_zero
                & result["forecast_calendar_index"].lt(episode.episode_onset_index)
                & result["target_calendar_index"].ge(episode.episode_onset_index)
            )
        )
        survival = (
            cell
            & result["forecast_calendar_index"].ge(episode.episode_onset_index)
            & result["forecast_calendar_index"].le(episode.episode_end_index)
            & result["target_calendar_index"].le(episode.episode_end_index)
        )
        target_inside = (
            cell
            & result["target_calendar_index"].ge(episode.episode_onset_index)
            & result["target_calendar_index"].le(episode.episode_end_index)
        )
        result.loc[onset, "target_episode_onset_within_lead"] = True
        result.loc[survival, "target_episode_survival"] = True
        result.loc[target_inside, "target_episode_id"] = str(episode.episode_id)
    return result


def _stock_ids(values: pd.Series) -> set[str]:
    result: set[str] = set()
    for value in values.dropna():
        decoded = json.loads(str(value))
        result.update(str(item) for item in decoded)
    return result


def build_episode_attribution(
    forecasts: pd.DataFrame,
    episode_states: pd.DataFrame,
    episodes: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    lookback_sessions: int = 5,
    positive_probability: float = 0.5,
) -> pd.DataFrame:
    """Classify episodes using the contract-frozen, evaluation-only rule."""

    _require(
        forecasts,
        {
            *CELL,
            "score_session",
            "model_name",
            "p_next_payoff_positive",
        },
        "episode forecast",
    )
    _require(
        episode_states,
        {
            *CELL,
            "score_session",
            "robust_net_payoff_bps",
            "independent_stock_ids",
        },
        "episode payoff state",
    )
    positions = _calendar_positions(calendar)
    indexed_forecasts = forecasts.merge(
        positions,
        on=["period", "score_session"],
        how="left",
        validate="many_to_one",
    )
    indexed_episodes = episodes.merge(
        positions.rename(
            columns={
                "score_session": "hindsight_estimated_onset",
                "calendar_session_index": "episode_onset_index",
            }
        ),
        on=["period", "hindsight_estimated_onset"],
        how="left",
        validate="many_to_one",
    ).merge(
        positions.rename(
            columns={
                "score_session": "hindsight_estimated_end",
                "calendar_session_index": "episode_end_index",
            }
        ),
        on=["period", "hindsight_estimated_end"],
        how="left",
        validate="many_to_one",
    )
    records: list[dict[str, object]] = []
    for raw_episode in indexed_episodes.itertuples(index=False):
        episode: Any = raw_episode
        cell = (
            indexed_forecasts["period"].eq(episode.period)
            & indexed_forecasts["loop_id"].eq(episode.loop_id)
            & indexed_forecasts["orientation"].eq(episode.orientation)
            & indexed_forecasts["horizon"].eq(episode.horizon)
        )
        pre = indexed_forecasts.loc[
            cell
            & indexed_forecasts["calendar_session_index"].ge(
                episode.episode_onset_index - lookback_sessions
            )
            & indexed_forecasts["calendar_session_index"].le(episode.episode_onset_index)
        ]

        def first_positive(
            model_name: str,
            current: pd.DataFrame = pre,
            onset_index: float = float(episode.episode_onset_index),
        ) -> tuple[Any, float]:
            selected = current.loc[
                current["model_name"].eq(model_name)
                & current["p_next_payoff_positive"].gt(positive_probability)
            ].sort_values("calendar_session_index", kind="stable")
            if selected.empty:
                return pd.NA, np.nan
            row = selected.iloc[0]
            return row["score_session"], float(onset_index - row["calendar_session_index"])

        full_date, full_lead = first_positive("hierarchical_change_point")
        control_date, control_lead = first_positive("hierarchical_payoff_history_change_point")
        if pd.isna(full_date) and pd.isna(control_date):
            classification = "unpredicted"
        elif pd.notna(full_date) and pd.notna(control_date) and full_date == control_date:
            classification = "simultaneous"
        elif pd.notna(full_date) and (pd.isna(control_date) or full_lead > control_lead):
            classification = "structurally_led"
        elif pd.notna(control_date) and (pd.isna(full_date) or control_lead > full_lead):
            classification = "payoff_history_led"
        else:
            classification = "ambiguous"

        paired = pre.pivot_table(
            index=[*CELL, "score_session", "calendar_session_index"],
            columns="model_name",
            values="p_next_payoff_positive",
            aggfunc="first",
        ).reset_index()
        if {
            "hierarchical_change_point",
            "hierarchical_payoff_history_change_point",
        } <= set(paired):
            increment = (
                paired["hierarchical_change_point"]
                - paired["hierarchical_payoff_history_change_point"]
            )
            positive_increment = bool(increment.gt(0.0).any())
        else:
            positive_increment = False

        payoff = episode_states.loc[
            episode_states["period"].eq(episode.period)
            & episode_states["loop_id"].eq(episode.loop_id)
            & episode_states["orientation"].eq(episode.orientation)
            & episode_states["horizon"].eq(episode.horizon)
            & episode_states["score_session"].ge(episode.hindsight_estimated_onset)
            & episode_states["score_session"].le(episode.hindsight_estimated_end)
        ].copy()
        if pd.notna(full_date):
            available = payoff.loc[payoff["score_session"].gt(full_date)]
        else:
            available = payoff.iloc[0:0]
        total_payoff = float(payoff["robust_net_payoff_bps"].sum())
        available_payoff = float(available["robust_net_payoff_bps"].sum())
        record = dict(episode._asdict())
        record.update(
            {
                "full_model_first_positive_lead_forecast": full_date,
                "no_feature_first_positive_lead_forecast": control_date,
                "full_forecast_lead_sessions_relative_to_onset": full_lead,
                "control_forecast_lead_sessions_relative_to_onset": control_lead,
                "full_features_fired_before_no_features": classification == "structurally_led",
                "positive_feature_increment_before_onset": positive_increment,
                "payoff_available_after_full_forecast_bps": available_payoff,
                "episode_share_captured_after_full_forecast": (
                    available_payoff / total_payoff if not np.isclose(total_payoff, 0.0) else np.nan
                ),
                "episode_independent_stocks": len(_stock_ids(payoff["independent_stock_ids"])),
                "episode_attribution_class": classification,
                "preceded_by_rising_breadth": bool(
                    getattr(episode, "breadth_increased_before_onset", False)
                ),
                "preceded_by_rising_coherence": bool(
                    getattr(episode, "coherence_increased_before_onset", False)
                ),
                "neither_precursor_appeared": not bool(
                    getattr(episode, "breadth_increased_before_onset", False)
                )
                and not bool(getattr(episode, "coherence_increased_before_onset", False)),
                "hindsight_labels_used_as_features": False,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


__all__ = ["attach_hindsight_episode_targets", "build_episode_attribution"]
