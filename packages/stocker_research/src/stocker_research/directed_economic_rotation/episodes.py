"""Outcome-only family episode unions and family payoff support."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .taxonomy import FamilyTaxonomy


@dataclass
class _EpisodeUnion:
    onset: str
    end: str
    onset_index: int
    end_index: int
    ids: list[str]
    payoff: float


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def _stable_id(values: tuple[object, ...]) -> str:
    payload = json.dumps([str(value) for value in values], separators=(",", ":"))
    return f"family-episode-{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def build_family_payoff_support(
    session_panel: pd.DataFrame,
    taxonomy: FamilyTaxonomy,
) -> pd.DataFrame:
    """Map observed pair payoff cells to equal-pair family support without filling gaps."""

    _require(
        session_panel,
        {
            "period",
            "session",
            "loop_id",
            "orientation",
            "data_availability_timestamp",
            "robust_net_payoff_bps",
            "independent_stock_count",
            "effective_sample_size",
        },
        "session payoff",
    )
    mapped = taxonomy.map_pairs(session_panel)
    mapped = mapped.loc[mapped["family_mapping_status"].eq("mapped")].copy()
    mapped["data_availability_timestamp"] = pd.to_datetime(
        mapped["data_availability_timestamp"], utc=True, errors="raise"
    )
    rows: list[dict[str, object]] = []
    keys = ["period", "session", "destination_family"]
    for key, group in mapped.groupby(keys, sort=True, observed=True):
        payoff = pd.to_numeric(group["robust_net_payoff_bps"], errors="coerce")
        rows.append(
            {
                "period": int(str(key[0])),
                "session": str(key[1]),
                "destination_family": str(key[2]),
                "data_availability_timestamp": group["data_availability_timestamp"].max(),
                "robust_net_payoff_bps": float(payoff.mean()),
                "independent_stock_count": int(
                    pd.to_numeric(group["independent_stock_count"], errors="coerce").sum()
                ),
                "effective_sample_size": float(
                    pd.to_numeric(group["effective_sample_size"], errors="coerce").sum()
                ),
                "observed_pair_cells": int(len(group)),
                "positive_payoff": bool(float(payoff.mean()) > 0.0),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_family_episode_intervals(
    pair_episodes: pd.DataFrame,
    taxonomy: FamilyTaxonomy,
    calendar: pd.DataFrame,
    payoff_support: pd.DataFrame,
) -> pd.DataFrame:
    """Union overlapping or adjacent positive pair episodes within each family."""

    _require(
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
        "pair episode",
    )
    _require(
        calendar,
        {"period", "score_session", "forecast_freeze_timestamp"},
        "calendar",
    )
    _require(
        payoff_support,
        {"period", "session", "destination_family", "data_availability_timestamp"},
        "payoff support",
    )
    mapped = taxonomy.map_pairs(pair_episodes)
    mapped = mapped.loc[
        mapped["family_mapping_status"].eq("mapped")
        & mapped["hindsight_estimated_onset"].notna()
        & mapped["hindsight_estimated_end"].notna()
    ].copy()
    columns = [
        "episode_id",
        "period",
        "destination_family",
        "episode_onset_session",
        "episode_end_session",
        "duration_sessions",
        "source_pair_episode_ids",
        "source_pair_count",
        "total_episode_payoff_bps",
        "label_availability_timestamp",
    ]
    if mapped.empty:
        return pd.DataFrame(columns=columns)
    period_sessions = {
        int(str(period)): group["score_session"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
        for period, group in calendar.groupby("period", sort=True, observed=True)
    }
    support = payoff_support.copy()
    support["data_availability_timestamp"] = pd.to_datetime(
        support["data_availability_timestamp"], utc=True, errors="coerce"
    )
    rows: list[dict[str, object]] = []
    for (period_value, family), group in mapped.groupby(
        ["period", "destination_family"], sort=True, observed=True
    ):
        period = int(str(period_value))
        family_name = str(family)
        sessions = period_sessions.get(period, [])
        index = {session: item for item, session in enumerate(sessions)}
        candidates: list[_EpisodeUnion] = []
        for record in group.to_dict(orient="records"):
            onset = str(record["hindsight_estimated_onset"])
            end = str(record["hindsight_estimated_end"])
            if onset not in index or end not in index or index[end] < index[onset]:
                continue
            candidates.append(
                _EpisodeUnion(
                    onset=onset,
                    end=end,
                    onset_index=index[onset],
                    end_index=index[end],
                    ids=[str(record["episode_id"])],
                    payoff=float(record["total_episode_payoff_bps"]),
                )
            )
        candidates.sort(key=lambda value: (value.onset_index, value.ids))
        unions: list[_EpisodeUnion] = []
        for candidate in candidates:
            if not unions or candidate.onset_index > unions[-1].end_index + 1:
                unions.append(candidate)
                continue
            current = unions[-1]
            if candidate.end_index > current.end_index:
                current.end = candidate.end
                current.end_index = candidate.end_index
            current.ids.extend(candidate.ids)
            current.payoff += candidate.payoff
        for union in unions:
            onset = union.onset
            end = union.end
            ids = sorted(set(union.ids))
            observed = support.loc[
                support["period"].eq(period)
                & support["destination_family"].eq(family_name)
                & support["session"]
                .astype(str)
                .isin(sessions[union.onset_index : union.end_index + 1])
            ]
            end_of_session = pd.Timestamp(f"{end}T23:59:59Z")
            availability = end_of_session
            if not observed.empty:
                observed_max = observed["data_availability_timestamp"].max()
                if not pd.isna(observed_max):
                    availability = max(availability, pd.Timestamp(observed_max))
            rows.append(
                {
                    "episode_id": _stable_id((period, family_name, onset, end, *ids)),
                    "period": period,
                    "destination_family": family_name,
                    "episode_onset_session": onset,
                    "episode_end_session": end,
                    "duration_sessions": union.end_index - union.onset_index + 1,
                    "source_pair_episode_ids": "|".join(ids),
                    "source_pair_count": len(ids),
                    "total_episode_payoff_bps": union.payoff,
                    "label_availability_timestamp": availability,
                }
            )
    return (
        pd.DataFrame.from_records(rows, columns=columns)
        .sort_values(["period", "destination_family", "episode_onset_session"], kind="stable")
        .reset_index(drop=True)
    )


__all__ = ["build_family_episode_intervals", "build_family_payoff_support"]
