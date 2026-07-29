"""Frozen exploratory-census reconstruction.

This module is intentionally small and independent of the experiment runner so
the gate can be audited directly from the immutable input ledgers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _session_text(values: pd.Series) -> pd.Series:
    """Return canonical trading-session labels without changing boundaries."""

    converted = pd.to_datetime(values, errors="raise")
    return converted.dt.strftime("%Y-%m-%d")


def _calendar_positions(calendar: pd.DataFrame) -> dict[str, Mapping[str, int]]:
    normalized = calendar.loc[:, ["period", "score_session"]].copy()
    normalized["period"] = normalized["period"].astype(str)
    normalized["score_session"] = _session_text(normalized["score_session"])
    normalized = normalized.drop_duplicates().sort_values(["period", "score_session"])
    result: dict[str, Mapping[str, int]] = {}
    for period, group in normalized.groupby("period", sort=True):
        sessions = group["score_session"].tolist()
        result[str(period)] = {session: index for index, session in enumerate(sessions)}
    return result


def _same_regime_episode_members(
    pair_episodes: pd.DataFrame,
    calendar: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Union overlapping or one-session-adjacent pair episodes by regime.

    Adjacency is measured on the explicit within-period trading calendar.  A
    missing calendar session therefore cannot be crossed, and periods can never
    be bridged.
    """

    positions = _calendar_positions(calendar)
    episodes = pair_episodes.copy()
    episodes["period"] = episodes["period"].astype(str)
    episodes["_onset"] = _session_text(episodes["hindsight_estimated_onset"])
    episodes["_end"] = _session_text(episodes["hindsight_estimated_end"])

    unions: list[pd.DataFrame] = []
    for (period, _orientation), group in episodes.groupby(
        ["period", "orientation"], sort=True, dropna=False
    ):
        period_positions = positions.get(str(period))
        if period_positions is None:
            raise ValueError(f"calendar has no sessions for period {period!r}")
        indexed = group.copy()
        try:
            indexed["_start_index"] = indexed["_onset"].map(period_positions.__getitem__)
            indexed["_end_index"] = indexed["_end"].map(period_positions.__getitem__)
        except KeyError as exc:
            raise ValueError(
                f"episode boundary {exc.args[0]!r} is absent from the {period} calendar"
            ) from exc
        indexed = indexed.sort_values(["_start_index", "_end_index", "episode_id"], kind="stable")

        current_indices: list[Any] = []
        current_end = -1
        for row_index, row in indexed.iterrows():
            start_index = int(row["_start_index"])
            end_index = int(row["_end_index"])
            if current_indices and start_index > current_end + 1:
                unions.append(indexed.loc[current_indices].copy())
                current_indices = []
                current_end = -1
            current_indices.append(row_index)
            current_end = max(current_end, end_index)
        if current_indices:
            unions.append(indexed.loc[current_indices].copy())
    return unions


def reproduce_exploratory_census(
    hindsight_states: pd.DataFrame,
    pair_episodes: pd.DataFrame,
    calendar: pd.DataFrame,
) -> dict[str, Any]:
    """Reproduce the registered census without adapting any frozen definition.

    A strictly positive pair is *exactly* a row whose frozen
    ``hindsight_payoff_state`` is ``"positive"``.  ``decaying`` is not positive;
    absent rows remain absent.  Same-regime episodes use the pair-episode ledger
    as supplied and an explicit zero/one trading-session union.
    """

    _require_columns(
        hindsight_states,
        {"period", "score_session", "loop_id", "orientation", "hindsight_payoff_state"},
        "hindsight_states",
    )
    _require_columns(
        pair_episodes,
        {
            "episode_id",
            "period",
            "loop_id",
            "orientation",
            "hindsight_estimated_onset",
            "hindsight_estimated_end",
            "total_episode_payoff_bps",
        },
        "pair_episodes",
    )
    _require_columns(calendar, {"period", "score_session"}, "calendar")

    positive = hindsight_states.loc[
        hindsight_states["hindsight_payoff_state"].eq("positive")
    ].copy()
    positive["period"] = positive["period"].astype(str)
    positive["score_session"] = _session_text(positive["score_session"])
    positive["_pair"] = positive["loop_id"].astype(str) + "|" + positive["orientation"].astype(str)
    per_session = (
        positive.groupby(["period", "score_session"], observed=True)["_pair"]
        .nunique()
        .rename("positive_pairs")
        .reset_index()
    )
    positive_sessions = int(len(per_session))
    multi_pair_sessions = int(per_session["positive_pairs"].ge(2).sum())

    period_results: dict[str, dict[str, float | int]] = {}
    for period, group in per_session.groupby("period", sort=True):
        count = int(len(group))
        multi = int(group["positive_pairs"].ge(2).sum())
        period_results[str(period)] = {
            "positive_sessions": count,
            "multi_pair_positive_sessions": multi,
            "multi_pair_positive_session_share": float(multi / count),
        }

    unions = _same_regime_episode_members(pair_episodes, calendar)
    union_records: list[dict[str, Any]] = []
    leader_shares: list[float] = []
    unavailable = 0
    majority = 0
    over_80 = 0
    for members in unions:
        period = str(members["period"].iloc[0])
        loop_count = int(members["loop_id"].nunique())
        union_records.append({"period": period, "loop_count": loop_count})
        if loop_count <= 1:
            continue
        # This is the frozen exploratory definition: first aggregate source
        # episode totals by loop, then retain strictly positive loop totals.
        loop_payoff = members.groupby("loop_id", observed=True)["total_episode_payoff_bps"].sum()
        positive_loop_payoff = loop_payoff[loop_payoff.gt(0.0)]
        denominator = float(positive_loop_payoff.sum())
        if positive_loop_payoff.empty or not np.isfinite(denominator) or denominator <= 0.0:
            unavailable += 1
            continue
        share = float(positive_loop_payoff.max() / denominator)
        leader_shares.append(share)
        majority += int(share > 0.5)
        over_80 += int(share > 0.8)

    union_frame = pd.DataFrame.from_records(union_records)
    multi_loop = union_frame["loop_count"].gt(1)
    period_multi_shares: dict[str, float] = {}
    for period, group in union_frame.groupby("period", sort=True):
        period_multi_shares[str(period)] = float(group["loop_count"].gt(1).mean())

    return {
        "strict_positive_pair_rows": int(len(positive)),
        "positive_sessions": positive_sessions,
        "multi_pair_positive_sessions": multi_pair_sessions,
        "multi_pair_positive_session_share": float(multi_pair_sessions / positive_sessions),
        "periods": period_results,
        "same_regime_episodes": int(len(union_frame)),
        "single_loop_same_regime_episodes": int((~multi_loop).sum()),
        "multi_loop_same_regime_episodes": int(multi_loop.sum()),
        "multi_loop_same_regime_share_by_period": period_multi_shares,
        "multi_loop_leader_share_available": int(len(leader_shares)),
        "multi_loop_leader_share_unavailable": int(unavailable),
        "multi_loop_leader_positive_payoff_share_median": float(np.median(leader_shares))
        if leader_shares
        else float("nan"),
        "multi_loop_majority_leader_episodes": int(majority),
        "multi_loop_over_80pct_leader_episodes": int(over_80),
    }
