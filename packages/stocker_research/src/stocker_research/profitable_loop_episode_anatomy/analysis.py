"""Deterministic, descriptive transformations for episode-anatomy research.

All functions consume in-memory frames or immutable files supplied by callers.
Nothing in this module can place orders or alter application runtime state.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


def frozen_run_git_head(
    contract: Mapping[str, Any],
    checkout_head: str,
    *,
    frozen_is_ancestor: bool,
) -> str:
    """Return the contract's immutable Git identity for deterministic outputs.

    A rerun from a later descendant checkout must retain the registered source
    identity rather than writing a volatile checkout SHA into byte-compared run
    metadata.  Divergent checkouts fail closed.
    """

    lineage = contract.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("contract lineage is missing")
    frozen = lineage.get("starting_commit")
    if not isinstance(frozen, str) or not frozen:
        raise ValueError("contract starting_commit is missing")
    if checkout_head != frozen and not frozen_is_ancestor:
        raise ValueError("checkout does not descend from frozen starting_commit")
    return frozen


def _canonical_session(values: pd.Series) -> pd.Series:
    converted = pd.to_datetime(values, errors="raise")
    return converted.dt.strftime("%Y-%m-%d")


def _winsorized_mean(values: pd.Series, fraction: float = 0.10) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(clean):
        return math.nan
    lower, upper = np.quantile(clean, [fraction, 1.0 - fraction])
    return float(np.clip(clean, lower, upper).mean())


def collapse_stock_contributions(fills: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated fills to one capped contribution per stock and cell."""

    aliases = {
        "score_session": "session",
        "loop_id": "loop",
        "stock_id": "stock",
        "symbol_norm": "stock",
        "primary_total_cost_bps": "cost_bps",
        "primary_net_payoff_bps": "net_payoff_bps",
    }
    frame = fills.copy()
    for source, destination in aliases.items():
        if destination not in frame and source in frame:
            frame[destination] = frame[source]
    required = {
        "period",
        "session",
        "loop",
        "orientation",
        "stock",
        "gross_payoff_bps",
        "cost_bps",
        "net_payoff_bps",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"fills missing required columns: {missing}")
    keys = ["period", "session", "loop", "orientation"]
    frame["session"] = _canonical_session(frame["session"])
    stock = (
        frame.groupby([*keys, "stock"], observed=True, dropna=False)
        .agg(
            stock_gross_payoff_bps=("gross_payoff_bps", "mean"),
            stock_cost_bps=("cost_bps", "mean"),
            stock_net_payoff_bps=("net_payoff_bps", "mean"),
            stock_raw_fill_count=("net_payoff_bps", "size"),
        )
        .reset_index()
    )
    stock["stock_net_payoff_bps"] = stock["stock_net_payoff_bps"].clip(-500.0, 500.0)
    # The frozen source reconstructs gross from the capped net contribution and
    # the observed cost; it does not cap gross independently.
    stock["stock_gross_payoff_bps"] = stock["stock_net_payoff_bps"] + stock["stock_cost_bps"]

    records: list[dict[str, Any]] = []
    for key, group in stock.groupby(keys, observed=True, dropna=False, sort=True):
        period, session, loop, orientation = key
        count = int(len(group))
        records.append(
            {
                "period": period,
                "session": session,
                "loop": loop,
                "parent_loop": loop,
                "orientation": orientation,
                "regime": orientation,
                "pair": f"{loop}|{orientation}",
                "occurrence_count": count,
                "independent_stock_count": count,
                "raw_fill_count": int(group["stock_raw_fill_count"].sum()),
                "effective_sample_size": float(count),
                "robust_gross_payoff_bps": _winsorized_mean(group["stock_gross_payoff_bps"]),
                "total_costs_bps": float(group["stock_cost_bps"].sum()),
                "robust_net_payoff_bps": _winsorized_mean(group["stock_net_payoff_bps"]),
                "median_net_payoff_bps": float(group["stock_net_payoff_bps"].median()),
                "stock_ids": "|".join(sorted(group["stock"].astype(str).unique())),
            }
        )
    return pd.DataFrame.from_records(records)


def build_synchronized_panel(
    eligibility_grid: pd.DataFrame,
    payoff_panel: pd.DataFrame,
    hindsight_states: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build an eligibility-preserving panel; absent opportunities stay missing."""

    grid = eligibility_grid.copy()
    payoff = payoff_panel.copy()
    aliases = {
        "score_session": "session",
        "loop_id": "loop",
        "independent_stock_count": "independent_stock_count",
        "cost_contribution_bps": "total_costs_bps",
    }
    for frame in (grid, payoff):
        for source, destination in aliases.items():
            if destination not in frame and source in frame:
                frame[destination] = frame[source]
    keys = ["period", "session", "loop", "orientation", "horizon"]
    missing_grid = sorted(set(keys).difference(grid.columns))
    if missing_grid:
        raise ValueError(f"eligibility grid missing columns: {missing_grid}")
    grid["session"] = _canonical_session(grid["session"])
    payoff["session"] = _canonical_session(payoff["session"])
    duplicate_payoff = payoff.duplicated(keys, keep=False)
    if duplicate_payoff.any():
        raise ValueError("payoff panel has duplicate synchronized cells")

    payoff_columns = [column for column in payoff.columns if column not in grid or column in keys]
    merged = grid.merge(payoff.loc[:, payoff_columns], on=keys, how="left", validate="one_to_one")
    merged["parent_loop"] = merged["loop"]
    merged["regime"] = merged["orientation"]
    merged["current_regime"] = merged["orientation"]
    merged["pair"] = merged["loop"].astype(str) + "|" + merged["orientation"].astype(str)
    merged["eligible"] = True

    label_keys = keys
    if hindsight_states is not None:
        labels = hindsight_states.copy()
        for source, destination in aliases.items():
            if destination not in labels and source in labels:
                labels[destination] = labels[source]
        labels["session"] = _canonical_session(labels["session"])
        labels = labels.loc[:, [*label_keys, "hindsight_payoff_state"]]
        if labels.duplicated(label_keys).any():
            raise ValueError("hindsight state panel has duplicate synchronized cells")
        merged = merged.merge(labels, on=label_keys, how="left", validate="one_to_one")
    else:
        merged["hindsight_payoff_state"] = pd.NA
    merged["positive_pair_available"] = merged["hindsight_payoff_state"].notna()
    merged["positive_pair_flag"] = merged["hindsight_payoff_state"].eq("positive").fillna(False)
    return merged.sort_values(keys, kind="stable").reset_index(drop=True)


def _calendar_maps(
    calendar: pd.DataFrame,
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    frame = calendar.copy()
    frame["period"] = frame["period"].astype(str)
    frame["score_session"] = _canonical_session(frame["score_session"])
    frame = frame.drop_duplicates().sort_values(["period", "score_session"])
    lists: dict[str, list[str]] = {}
    maps: dict[str, dict[str, int]] = {}
    for period, group in frame.groupby("period", sort=True):
        sessions = group["score_session"].tolist()
        lists[str(period)] = sessions
        maps[str(period)] = {value: index for index, value in enumerate(sessions)}
    return lists, maps


def _union_episodes(
    pair: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    episode_level: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    _, maps = _calendar_maps(calendar)
    records: list[dict[str, Any]] = []
    grouping = ["period", *group_columns]
    for _group_key, group in pair.groupby(grouping, sort=True, dropna=False, observed=True):
        ordered = group.sort_values(["onset", "end", "pair_episode_id"], kind="stable")
        batches: list[list[int]] = []
        current: list[int] = []
        current_end: int | None = None
        for index, row in ordered.iterrows():
            period = str(row["period"])
            start = maps.get(period, {}).get(str(row["onset"]))
            end = maps.get(period, {}).get(str(row["end"]))
            # A boundary absent from the explicit calendar is retained as an
            # isolated episode; it cannot silently bridge known sessions.
            connect = (
                bool(current)
                and start is not None
                and current_end is not None
                and start <= current_end + 1
            )
            if current and not connect:
                batches.append(current)
                current = []
                current_end = None
            current.append(cast(int, index))
            current_end = end if current_end is None or end is None else max(current_end, end)
        if current:
            batches.append(current)

        for batch in batches:
            members = ordered.loc[batch]
            record_number = len(records) + 1
            loops = sorted(members["loop"].astype(str).unique())
            regimes = sorted(members["regime"].astype(str).unique())
            records.append(
                {
                    "episode_id": f"{episode_level}_{record_number:04d}",
                    "episode_level": episode_level,
                    "period": str(members["period"].iloc[0]),
                    "onset": min(members["onset"]),
                    "end": max(members["end"]),
                    "source_pair_episode_ids": "|".join(
                        sorted(members["pair_episode_id"].astype(str))
                    ),
                    "loops": "|".join(loops),
                    "regimes": "|".join(regimes),
                    "loop_count": len(loops),
                    "regime_count": len(regimes),
                    "source_pair_episode_count": len(members),
                }
            )
    return pd.DataFrame.from_records(records)


def build_episode_ledgers(
    pair_episodes: pd.DataFrame, calendar: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return distinct pair, same-regime, and shared-market ledgers."""

    source = pair_episodes.copy()
    source["period"] = source["period"].astype(str)
    source["onset"] = _canonical_session(source["hindsight_estimated_onset"])
    source["end"] = _canonical_session(source["hindsight_estimated_end"])
    source = source.rename(
        columns={
            "episode_id": "pair_episode_id",
            "loop_id": "loop",
            "orientation": "regime",
        }
    )
    source["orientation"] = source["regime"]
    source["pair"] = source["loop"].astype(str) + "|" + source["regime"].astype(str)
    pair = source.copy()
    pair["episode_id"] = pair["pair_episode_id"]
    pair["episode_level"] = "pair"
    pair["source_pair_episode_ids"] = pair["pair_episode_id"].astype(str)
    same = _union_episodes(source, calendar, episode_level="same_regime", group_columns=["regime"])
    shared = _union_episodes(source, calendar, episode_level="shared_market", group_columns=[])
    return pair.reset_index(drop=True), same.reset_index(drop=True), shared.reset_index(drop=True)


def attach_episode_membership(
    panel: pd.DataFrame,
    pair_ledger: pd.DataFrame,
    same_regime_ledger: pd.DataFrame,
    shared_market_ledger: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact, calendar-aware episode memberships to each panel row.

    A row may legitimately belong to more than one independently defined level,
    but memberships at a single level are never guessed.  Any overlapping IDs
    are retained as a sorted pipe-delimited set so that ties remain explicit.
    """

    result = panel.copy()
    result["session"] = _canonical_session(result["session"])
    calendar_frame = calendar.copy()
    calendar_frame["period"] = calendar_frame["period"].astype(str)
    calendar_frame["score_session"] = _canonical_session(calendar_frame["score_session"])
    sessions_by_period = {
        str(period): sorted(group["score_session"].drop_duplicates().astype(str))
        for period, group in calendar_frame.groupby("period", sort=True)
    }

    def build_map(
        ledger: pd.DataFrame,
        *,
        key_column: str | None,
        ledger_key_column: str | None,
    ) -> dict[tuple[str, ...], list[str]]:
        memberships: dict[tuple[str, ...], list[str]] = {}
        for row in ledger.itertuples(index=False):
            period = str(row.period)
            sessions = sessions_by_period.get(period, [])
            if not sessions:
                continue
            positions = {session: index for index, session in enumerate(sessions)}
            onset = str(row.onset)
            end = str(row.end)
            if onset not in positions or end not in positions:
                continue
            ledger_key = ""
            if ledger_key_column is not None:
                ledger_key = str(getattr(row, ledger_key_column))
            for session in sessions[positions[onset] : positions[end] + 1]:
                key = (period, session, ledger_key) if key_column is not None else (period, session)
                memberships.setdefault(key, []).append(str(row.episode_id))
        return memberships

    pair_map = build_map(pair_ledger, key_column="pair", ledger_key_column="pair")
    same_source = same_regime_ledger.copy()
    same_source["regime_key"] = same_source["regimes"].astype(str)
    same_map = build_map(
        same_source,
        key_column="regime",
        ledger_key_column="regime_key",
    )
    shared_map = build_map(
        shared_market_ledger,
        key_column=None,
        ledger_key_column=None,
    )

    def lookup(mapping: dict[tuple[str, ...], list[str]], key: tuple[str, ...]) -> object:
        values = sorted(set(mapping.get(key, [])))
        return "|".join(values) if values else pd.NA

    result["hindsight_pair_episode_id"] = pd.Series(
        [
            lookup(pair_map, (str(row.period), str(row.session), str(row.pair)))
            for row in result.itertuples()
        ],
        index=result.index,
        dtype="string",
    )
    result["same_regime_episode_id"] = pd.Series(
        [
            lookup(same_map, (str(row.period), str(row.session), str(row.regime)))
            for row in result.itertuples()
        ],
        index=result.index,
        dtype="string",
    )
    result["shared_session_episode_id"] = pd.Series(
        [lookup(shared_map, (str(row.period), str(row.session))) for row in result.itertuples()],
        index=result.index,
        dtype="string",
    )
    return result


def decompose_payoff_components(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply the registered robust common/regime/residual additive identity."""

    frame = panel.copy()
    required = {"period", "session", "regime", "robust_net_payoff_bps", "eligible"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"component panel missing columns: {missing}")
    supported = frame["eligible"].fillna(False) & frame["robust_net_payoff_bps"].notna()
    frame["common_component"] = math.nan
    frame["regime_component"] = math.nan
    frame["loop_excess_component"] = math.nan
    if not supported.any():
        return frame
    active = frame.loc[supported].copy()
    group_session = ["period", "session"]
    active["common_component"] = active.groupby(group_session, observed=True)[
        "robust_net_payoff_bps"
    ].transform("median")
    active["_after_common"] = active["robust_net_payoff_bps"] - active["common_component"]
    active["regime_component"] = active.groupby([*group_session, "regime"], observed=True)[
        "_after_common"
    ].transform("median")
    active["loop_excess_component"] = (
        active["robust_net_payoff_bps"] - active["common_component"] - active["regime_component"]
    )
    columns = ["common_component", "regime_component", "loop_excess_component"]
    frame.loc[supported, columns] = active[columns].to_numpy()
    reconciliation = (
        frame.loc[supported, columns].sum(axis=1) - frame.loc[supported, "robust_net_payoff_bps"]
    ).abs()
    if bool(reconciliation.gt(1e-10).any()):
        raise AssertionError("payoff component identity failed")
    return frame


def recompute_component_summary_after_stock_removal(
    occurrences: pd.DataFrame,
    *,
    member_pairs: Iterable[str],
    removed_stocks: Iterable[str],
) -> dict[str, float | int]:
    """Rebuild robust pair payoffs and C/R/L after removing named stocks.

    Callers must provide the full supported pair population for the relevant
    sessions, not merely the target episode pairs.  This keeps the common
    component on its registered all-pair support base.
    """

    required = {
        "period",
        "session",
        "pair",
        "loop",
        "regime",
        "stock",
        "net_payoff_bps",
    }
    missing = sorted(required.difference(occurrences.columns))
    if missing:
        raise ValueError(f"occurrence rows missing required columns: {missing}")
    removed = {str(stock) for stock in removed_stocks}
    remaining = occurrences[~occurrences["stock"].astype(str).isin(removed)].copy()
    keys = ["period", "session", "pair", "loop", "regime"]
    rebuilt = (
        remaining.groupby(keys, observed=True, dropna=False)["net_payoff_bps"]
        .agg(_winsorized_mean)
        .rename("robust_net_payoff_bps")
        .reset_index()
    )
    if rebuilt.empty:
        return {
            "supported_pair_cells": 0,
            "robust_net_payoff_sum": math.nan,
            "robust_positive_payoff_sum": math.nan,
            "common_component_mean": math.nan,
            "regime_component_mean": math.nan,
            "loop_excess_component_mean": math.nan,
            "common_positive_component_share": math.nan,
            "regime_positive_component_share": math.nan,
            "loop_excess_positive_component_share": math.nan,
            "component_identity_max_absolute_error": math.nan,
        }
    rebuilt["eligible"] = True
    rebuilt["orientation"] = rebuilt["regime"]
    decomposed = decompose_payoff_components(rebuilt)
    scope = decomposed[decomposed["pair"].astype(str).isin({str(pair) for pair in member_pairs})]
    if scope.empty:
        return {
            "supported_pair_cells": 0,
            "robust_net_payoff_sum": math.nan,
            "robust_positive_payoff_sum": math.nan,
            "common_component_mean": math.nan,
            "regime_component_mean": math.nan,
            "loop_excess_component_mean": math.nan,
            "common_positive_component_share": math.nan,
            "regime_positive_component_share": math.nan,
            "loop_excess_positive_component_share": math.nan,
            "component_identity_max_absolute_error": math.nan,
        }
    component_columns = [
        "common_component",
        "regime_component",
        "loop_excess_component",
    ]
    positive_masses = scope[component_columns].clip(lower=0.0).sum()
    denominator = float(positive_masses.sum())
    identity_error = (scope[component_columns].sum(axis=1) - scope["robust_net_payoff_bps"]).abs()
    return {
        "supported_pair_cells": int(len(scope)),
        "robust_net_payoff_sum": float(scope["robust_net_payoff_bps"].sum()),
        "robust_positive_payoff_sum": float(scope["robust_net_payoff_bps"].clip(lower=0.0).sum()),
        "common_component_mean": float(scope["common_component"].mean()),
        "regime_component_mean": float(scope["regime_component"].mean()),
        "loop_excess_component_mean": float(scope["loop_excess_component"].mean()),
        "common_positive_component_share": float(positive_masses["common_component"]) / denominator
        if denominator > 0
        else math.nan,
        "regime_positive_component_share": float(positive_masses["regime_component"]) / denominator
        if denominator > 0
        else math.nan,
        "loop_excess_positive_component_share": float(positive_masses["loop_excess_component"])
        / denominator
        if denominator > 0
        else math.nan,
        "component_identity_max_absolute_error": float(identity_error.max()),
    }


def classify_episode(loop_count: int, leader_positive_payoff_share: float) -> str:
    """Return the frozen mutually exclusive primary anatomy category."""

    if loop_count < 1:
        raise ValueError("loop_count must be positive")
    if loop_count == 1:
        return "SINGLE_LOOP_EPISODE"
    if not math.isfinite(leader_positive_payoff_share):
        return "UNKNOWN_MULTI_LOOP"
    if leader_positive_payoff_share > 0.80:
        return "EXTREME_LOOP_DOMINANCE"
    if leader_positive_payoff_share > 0.50:
        return "MAJORITY_LOOP_DOMINANCE"
    return "DIFFUSE_MULTI_LOOP"


def _ranked_loop_totals(group: pd.DataFrame) -> pd.DataFrame:
    totals = (
        group.groupby("loop", observed=True)
        .agg(
            positive_payoff=("positive_payoff", "sum"),
            occurrence_count=("occurrence_count", "sum"),
        )
        .reset_index()
    )
    totals["positive_payoff"] = totals["positive_payoff"].clip(lower=0.0)
    return totals.sort_values(["positive_payoff", "loop"], ascending=[False, True])


def early_leader_checkpoints(rows: pd.DataFrame) -> pd.DataFrame:
    """Describe provisional leaders using prefixes only.

    The final ranking is computed separately and used only for the registered
    retrospective match columns.  It never affects provisional ranks or shares.
    """

    required = {"episode_id", "session", "loop", "positive_payoff", "occurrence_count"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"early leader input missing columns: {missing}")
    checkpoint_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for episode_id, episode in rows.groupby("episode_id", sort=True, observed=True):
        sessions = sorted(episode["session"].astype(str).unique())
        if "final_positive_payoff" in episode:
            final = (
                episode[["loop", "final_positive_payoff"]]
                .drop_duplicates("loop")
                .rename(columns={"final_positive_payoff": "positive_payoff"})
            )
            final["occurrence_count"] = final["loop"].map(
                episode.groupby("loop", observed=True)["occurrence_count"].sum()
            )
            final = final.sort_values(["positive_payoff", "loop"], ascending=[False, True])
        else:
            final = _ranked_loop_totals(episode)
        realised_final = _ranked_loop_totals(episode)
        total_payoff = float(final["positive_payoff"].sum())
        realised_total_payoff = float(realised_final["positive_payoff"].sum())
        total_occurrences = float(final["occurrence_count"].sum())
        payoff_shares = final["positive_payoff"] / total_payoff if total_payoff > 0 else math.nan
        occurrence_shares = (
            final["occurrence_count"] / total_occurrences if total_occurrences > 0 else math.nan
        )
        max_payoff = float(final["positive_payoff"].max()) if len(final) else math.nan
        leaders = (
            sorted(final.loc[final["positive_payoff"].eq(max_payoff), "loop"].astype(str))
            if total_payoff > 0
            else []
        )
        leader_tie = len(leaders) > 1
        leader_share = float(max_payoff / total_payoff) if total_payoff > 0 else math.nan
        realised_final_leader_payoff = float(
            realised_final.loc[
                realised_final["loop"].astype(str).isin(leaders), "positive_payoff"
            ].sum()
        )
        if total_occurrences > 0 and not leader_tie and leaders:
            leader_occurrences = float(
                final.loc[final["loop"].astype(str).eq(leaders[0]), "occurrence_count"].iloc[0]
            )
            leader_occurrence_share = leader_occurrences / total_occurrences
            leader_efficiency = (
                leader_share / leader_occurrence_share if leader_occurrence_share > 0 else math.nan
            )
        else:
            leader_occurrence_share = math.nan
            leader_efficiency = math.nan
        summary_records.append(
            {
                "episode_id": episode_id,
                "leader_loops": "|".join(leaders),
                "leader_tie": leader_tie,
                "leader_positive_payoff_share": leader_share,
                "leader_occurrence_share": leader_occurrence_share,
                "leader_efficiency": leader_efficiency,
                "positive_payoff_share_sum": float(payoff_shares.sum())
                if isinstance(payoff_shares, pd.Series)
                else math.nan,
                "occurrence_share_sum": float(occurrence_shares.sum())
                if isinstance(occurrence_shares, pd.Series)
                else math.nan,
            }
        )

        duration = len(sessions)
        checkpoint_lengths = {
            "first_session": 1,
            "first_two": 2,
            "first_three": 3,
            "first_25pct": max(1, int(math.ceil(duration * 0.25))),
            "first_50pct": max(1, int(math.ceil(duration * 0.50))),
        }
        for checkpoint, prefix_length in checkpoint_lengths.items():
            if prefix_length > duration:
                continue
            prefix_sessions = set(sessions[:prefix_length])
            prefix = episode[episode["session"].astype(str).isin(prefix_sessions)]
            provisional = _ranked_loop_totals(prefix)
            provisional_max = float(provisional["positive_payoff"].max())
            prefix_payoff = float(provisional["positive_payoff"].sum())
            provisional_leaders = (
                sorted(
                    provisional.loc[
                        provisional["positive_payoff"].eq(provisional_max), "loop"
                    ].astype(str)
                )
                if prefix_payoff > 0
                else []
            )
            prefix_occurrences = float(provisional["occurrence_count"].sum())
            provisional_share = provisional_max / prefix_payoff if prefix_payoff > 0 else math.nan
            provisional_occurrence = float(
                provisional.loc[
                    provisional["loop"].astype(str).isin(provisional_leaders), "occurrence_count"
                ].sum()
            )
            provisional_occurrence_share = (
                provisional_occurrence / prefix_occurrences
                if prefix_occurrences > 0 and provisional_leaders
                else math.nan
            )
            provisional_efficiency = (
                provisional_share / provisional_occurrence_share
                if provisional_occurrence_share > 0
                else math.nan
            )
            provisional_ranks = provisional.set_index("loop")["positive_payoff"].rank(
                method="min", ascending=False
            )
            top_three = set(provisional_ranks[provisional_ranks.le(3)].index.astype(str))
            final_leader_set = set(leaders)
            rank_frame = (
                final[["loop", "positive_payoff"]]
                .rename(columns={"positive_payoff": "final_payoff"})
                .merge(
                    provisional[["loop", "positive_payoff"]].rename(
                        columns={"positive_payoff": "provisional_payoff"}
                    ),
                    on="loop",
                    how="outer",
                )
                .fillna(0.0)
            )
            final_ranks = rank_frame["final_payoff"].rank(method="average", ascending=False)
            prefix_ranks = rank_frame["provisional_payoff"].rank(method="average", ascending=False)
            rank_correlation = (
                float(final_ranks.corr(prefix_ranks, method="spearman"))
                if len(rank_frame) >= 2 and final_ranks.nunique() > 1 and prefix_ranks.nunique() > 1
                else math.nan
            )
            after = episode[~episode["session"].astype(str).isin(prefix_sessions)]
            remaining = float(after["positive_payoff"].clip(lower=0.0).sum())
            final_leader_remaining = float(
                after.loc[after["loop"].astype(str).isin(final_leader_set), "positive_payoff"]
                .clip(lower=0.0)
                .sum()
            )
            checkpoint_records.append(
                {
                    "episode_id": episode_id,
                    "checkpoint": checkpoint,
                    "prefix_sessions": prefix_length,
                    "provisional_leaders": "|".join(provisional_leaders),
                    "provisional_leader_tie": len(provisional_leaders) > 1,
                    "top_one_match": bool(set(provisional_leaders) & final_leader_set)
                    if provisional_leaders
                    else pd.NA,
                    "top_three_inclusion": bool(top_three & final_leader_set)
                    if provisional_leaders
                    else pd.NA,
                    "rank_correlation_with_final": rank_correlation,
                    "provisional_leader_payoff_share": provisional_share,
                    "provisional_leader_occurrence_share": provisional_occurrence_share,
                    "provisional_leader_efficiency": provisional_efficiency,
                    "payoff_remaining": remaining,
                    "fraction_final_payoff_remaining": remaining / realised_total_payoff
                    if realised_total_payoff > 0
                    else math.nan,
                    "final_leader_payoff_remaining": final_leader_remaining,
                    "fraction_final_leader_payoff_remaining": final_leader_remaining
                    / realised_final_leader_payoff
                    if realised_final_leader_payoff > 0
                    else math.nan,
                }
            )
    result = pd.DataFrame.from_records(checkpoint_records)
    result.attrs["episode_summary"] = pd.DataFrame.from_records(summary_records)
    return result


def decode_history_token(token: int) -> dict[str, str]:
    """Decode the frozen current/previous-two completed-state token."""

    if token < 0:
        raise ValueError("history token must be non-negative")
    current = token % 8
    quotient = token // 8
    previous_1 = quotient % 9
    previous_2 = quotient // 9
    result = {
        "current_regime": f"state_{current}",
        "previous_regime": "unavailable" if previous_1 == 8 else f"state_{previous_1}",
    }
    if previous_1 != 8:
        result["regime_history_2"] = f"state_{previous_1}>state_{current}"
    else:
        result["regime_history_2"] = "unavailable"
    if previous_1 != 8 and previous_2 != 8:
        result["regime_history_3"] = f"state_{previous_2}>state_{previous_1}>state_{current}"
    else:
        result["regime_history_3"] = "unavailable"
    return result


def four_way_counterfactual(
    rows: pd.DataFrame,
    *,
    target_loop: str,
    current_regime: str,
    target_sequence: str,
    sequence_column: str,
) -> pd.DataFrame:
    """Build the registered target-loop x target-sequence four-way table."""

    filtered = rows[rows["regime"].astype(str).eq(current_regime)].copy()
    is_loop = filtered["loop"].astype(str).eq(target_loop)
    is_sequence = filtered[sequence_column].astype(str).eq(target_sequence)
    filtered["counterfactual_group"] = np.select(
        [is_loop & is_sequence, is_loop & ~is_sequence, ~is_loop & is_sequence],
        ["1", "2", "3"],
        default="4",
    )
    records: list[dict[str, Any]] = []
    for group_id in ["1", "2", "3", "4"]:
        group = filtered[filtered["counterfactual_group"].eq(group_id)]
        record: dict[str, Any] = {
            "counterfactual_group": group_id,
            "target_loop": target_loop,
            "regime": current_regime,
            "target_sequence": target_sequence,
            "sequence_column": sequence_column,
            "rows": int(len(group)),
            "independent_sessions": int(group["session"].nunique()) if "session" in group else 0,
            "independent_stocks": int(group["stock"].nunique()) if "stock" in group else 0,
            "mean_net_payoff_bps": float(group["robust_net_payoff_bps"].mean())
            if len(group)
            else math.nan,
            "median_net_payoff_bps": float(group["robust_net_payoff_bps"].median())
            if len(group)
            else math.nan,
            "positive_rate": float(group["robust_net_payoff_bps"].gt(0).mean())
            if len(group)
            else math.nan,
        }
        for component in [
            "common_component",
            "regime_component",
            "loop_excess_component",
        ]:
            record[component] = (
                float(group[component].mean()) if component in group and len(group) else math.nan
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _conditional_multi_share(frame: pd.DataFrame) -> float:
    counts = (
        frame.loc[frame["eligible"].fillna(False)]
        .groupby(["period", "session"], observed=True)["positive_pair_flag"]
        .sum()
    )
    positive = counts[counts.gt(0)]
    return float(positive.ge(2).mean()) if len(positive) else math.nan


def poisson_binomial_null(rows: pd.DataFrame) -> pd.DataFrame:
    """Exact pair-specific-independence distribution per eligible session."""

    frame = rows.copy()
    frame["period"] = frame["period"].astype(str)
    eligible = frame[frame["eligible"].fillna(False)].copy()
    rates = eligible.groupby(["period", "pair"], observed=True)["positive_pair_flag"].mean()
    records: list[dict[str, Any]] = []
    for (period, session), group in eligible.groupby(["period", "session"], sort=True):
        probabilities = np.array(
            [float(rates.loc[(str(period), pair)]) for pair in group["pair"]], dtype=float
        )
        distribution = np.array([1.0])
        for probability in probabilities:
            distribution = np.convolve(distribution, np.array([1.0 - probability, probability]))
        record: dict[str, Any] = {
            "period": str(period),
            "session": session,
            "eligible_pair_count": int(len(probabilities)),
            "observed_positive_pair_count": int(group["positive_pair_flag"].sum()),
            "probability_at_least_two": float(distribution[2:].sum())
            if len(distribution) > 2
            else 0.0,
        }
        for index, probability in enumerate(distribution):
            record[f"probability_exactly_{index}"] = float(probability)
        records.append(record)
    return pd.DataFrame.from_records(records)


def block_circular_pair_shift(
    rows: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
    block_length: int,
    retain_shifted_rows: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Circularly shift each pair inside its period and eligibility sequence."""

    if resamples < 1 or block_length < 1:
        raise ValueError("resamples and block_length must be positive")
    frame = rows.loc[:, ["period", "session", "pair", "eligible", "positive_pair_flag"]].copy()
    frame["period"] = frame["period"].astype(str)
    frame["positive_pair_flag"] = frame["positive_pair_flag"].fillna(False).astype(bool)
    frame["eligible"] = frame["eligible"].fillna(False).astype(bool)
    frame = frame.sort_values(["period", "pair", "session"], kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    outputs: list[pd.DataFrame] = []
    compact_records: list[dict[str, Any]] = []
    null_shares: list[float] = []
    count_preserved = True
    group_indices = [
        group.index.to_numpy()
        for _, group in frame.groupby(["period", "pair"], sort=True, observed=True)
    ]
    if not retain_shifted_rows:
        eligible_mask = frame["eligible"].to_numpy(dtype=bool)
        session_keys = frame["period"].astype(str) + "\x1f" + frame["session"].astype(str)
        session_codes, unique_sessions = pd.factorize(session_keys, sort=True)
        for replicate in range(resamples):
            shifted_values = np.zeros(len(frame), dtype=bool)
            for indices in group_indices:
                eligible_indices = indices[eligible_mask[indices]]
                values = frame.loc[eligible_indices, "positive_pair_flag"].to_numpy(dtype=bool)
                if len(values):
                    offset = int(rng.integers(0, max(1, math.ceil(len(values) / block_length))))
                    offset = (offset * block_length) % len(values)
                    shifted_values[eligible_indices] = np.roll(values, offset)
            session_counts = np.bincount(
                session_codes[eligible_mask],
                weights=shifted_values[eligible_mask].astype(int),
                minlength=len(unique_sessions),
            )
            positive_counts = session_counts[session_counts > 0]
            share = float(np.mean(positive_counts >= 2)) if len(positive_counts) else math.nan
            null_shares.append(share)
            compact_records.append(
                {
                    "replicate": replicate,
                    "multi_pair_positive_session_share": share,
                    "positive_sessions": int(np.sum(session_counts > 0)),
                    "exactly_one": int(np.sum(session_counts == 1)),
                    "exactly_two": int(np.sum(session_counts == 2)),
                    "exactly_three": int(np.sum(session_counts == 3)),
                    "four_or_more": int(np.sum(session_counts >= 4)),
                    "maximum_positive_pairs": int(session_counts.max()),
                }
            )
            # Each assignment above is a circular permutation of the complete
            # eligible sequence, so its positive count is preserved exactly.
        ledger = pd.DataFrame.from_records(compact_records)
        finite_null = np.asarray(
            [value for value in null_shares if math.isfinite(value)], dtype=float
        )
        observed = _conditional_multi_share(frame)
        null_mean = float(finite_null.mean()) if len(finite_null) else math.nan
        return ledger, {
            "resamples": resamples,
            "seed": seed,
            "block_length": block_length,
            "observed_multi_pair_positive_session_share": observed,
            "block_null_mean_multi_pair_positive_session_share": null_mean,
            "block_null_lower_95": float(np.quantile(finite_null, 0.025))
            if len(finite_null)
            else math.nan,
            "block_null_upper_95": float(np.quantile(finite_null, 0.975))
            if len(finite_null)
            else math.nan,
            "observed_minus_null_share": observed - null_mean,
            "one_sided_empirical_p": float(
                (1 + np.sum(finite_null >= observed)) / (1 + len(finite_null))
            )
            if len(finite_null)
            else math.nan,
            "eligibility_mask_preserved": True,
            "pair_positive_counts_preserved": bool(count_preserved),
            "period_boundaries_preserved": True,
        }
    for replicate in range(resamples):
        shifted = frame.copy()
        shifted["positive_pair_flag"] = False
        for indices in group_indices:
            eligible_indices = indices[frame.loc[indices, "eligible"].to_numpy(dtype=bool)]
            values = frame.loc[eligible_indices, "positive_pair_flag"].to_numpy(dtype=bool)
            if len(values):
                offset = int(rng.integers(0, max(1, math.ceil(len(values) / block_length))))
                offset = (offset * block_length) % len(values)
                shifted.loc[eligible_indices, "positive_pair_flag"] = np.roll(values, offset)
        shifted["replicate"] = replicate
        share = _conditional_multi_share(shifted)
        null_shares.append(share)
        expected_counts = frame.groupby(["period", "pair"], observed=True)[
            "positive_pair_flag"
        ].sum()
        replicate_counts = shifted.groupby(["period", "pair"], observed=True)[
            "positive_pair_flag"
        ].sum()
        count_preserved = count_preserved and replicate_counts.equals(expected_counts)
        if retain_shifted_rows:
            outputs.append(shifted)
        else:
            session_counts = (
                shifted.loc[shifted["eligible"]]
                .groupby(["period", "session"], observed=True)["positive_pair_flag"]
                .sum()
            )
            compact_records.append(
                {
                    "replicate": replicate,
                    "multi_pair_positive_session_share": share,
                    "positive_sessions": int(session_counts.gt(0).sum()),
                    "exactly_one": int(session_counts.eq(1).sum()),
                    "exactly_two": int(session_counts.eq(2).sum()),
                    "exactly_three": int(session_counts.eq(3).sum()),
                    "four_or_more": int(session_counts.ge(4).sum()),
                    "maximum_positive_pairs": int(session_counts.max()),
                }
            )
    ledger = (
        pd.concat(outputs, ignore_index=True)
        if retain_shifted_rows
        else pd.DataFrame.from_records(compact_records)
    )
    finite_null = np.asarray([value for value in null_shares if math.isfinite(value)], dtype=float)
    observed = _conditional_multi_share(frame)
    null_mean = float(finite_null.mean()) if len(finite_null) else math.nan
    summary = {
        "resamples": resamples,
        "seed": seed,
        "block_length": block_length,
        "observed_multi_pair_positive_session_share": observed,
        "block_null_mean_multi_pair_positive_session_share": null_mean,
        "block_null_lower_95": float(np.quantile(finite_null, 0.025))
        if len(finite_null)
        else math.nan,
        "block_null_upper_95": float(np.quantile(finite_null, 0.975))
        if len(finite_null)
        else math.nan,
        "observed_minus_null_share": observed - null_mean,
        "one_sided_empirical_p": float(
            (1 + np.sum(finite_null >= observed)) / (1 + len(finite_null))
        )
        if len(finite_null)
        else math.nan,
        "eligibility_mask_preserved": True,
        "pair_positive_counts_preserved": bool(count_preserved),
        "period_boundaries_preserved": (
            set(ledger["period"]) == set(frame["period"]) if retain_shifted_rows else True
        ),
    }
    return ledger, summary


_FORBIDDEN_CAUSAL_INDICATORS = {
    "mfe_bps",
    "mae_bps",
    "post_entry_mfe_bps",
    "post_entry_mae_bps",
    "hindsight_episode_id",
    "hindsight_payoff_state",
    "target_episode_state",
}


def validate_causal_indicators(columns: Iterable[str], rows: pd.DataFrame | None = None) -> None:
    """Fail closed on realised outcomes, hindsight labels, or late features."""

    normalized = {str(column).lower() for column in columns}
    forbidden = sorted(normalized & _FORBIDDEN_CAUSAL_INDICATORS)
    if forbidden:
        raise ValueError(f"forbidden causal indicators: {forbidden}")
    future = sorted(column for column in normalized if column.startswith("future_"))
    if future:
        raise ValueError(f"future causal indicators are forbidden: {future}")
    if rows is not None:
        decision = pd.to_datetime(rows["decision_timestamp"], utc=True)
        available = pd.to_datetime(rows["feature_availability_timestamp"], utc=True)
        if bool(available.gt(decision).fillna(False).any()):
            raise ValueError("causal indicator became available after decision timestamp")


def concentration_attribution(rows: pd.DataFrame) -> pd.DataFrame:
    """Compute episode stock concentration and descriptive removal attribution."""

    records: list[dict[str, Any]] = []
    for episode_id, group in rows.groupby("episode_id", sort=True, observed=True):
        stock = (
            group.assign(positive_payoff=group["positive_payoff"].clip(lower=0.0))
            .groupby("stock", observed=True)["positive_payoff"]
            .sum()
            .sort_values(ascending=False)
        )
        total = float(stock.sum())
        shares = stock / total if total > 0 else stock * math.nan
        sector_available = (
            "sector" in group
            and group["sector"].notna().any()
            and not group["sector"].astype(str).eq("unavailable").all()
        )
        records.append(
            {
                "episode_id": episode_id,
                "total_positive_payoff": total,
                "top_one_share": float(shares.iloc[0]) if len(shares) else math.nan,
                "top_five_share": float(shares.head(5).sum()) if len(shares) else math.nan,
                "herfindahl_index": float(np.square(shares).sum()) if len(shares) else math.nan,
                "after_remove_best_stock": float(stock.iloc[1:].sum()) if len(stock) else math.nan,
                "after_remove_top_five_stocks": float(stock.iloc[5:].sum())
                if len(stock)
                else math.nan,
                "stock_concentrated": bool(len(shares) and shares.iloc[0] > 0.30),
                "sector_status": "available" if sector_available else "unavailable",
            }
        )
    return pd.DataFrame.from_records(records)


def common_factor_diagnostic(
    rows: pd.DataFrame, *, factor_counts: Sequence[int] = (1, 2, 3)
) -> pd.DataFrame:
    """Fixed-count, period-specific PCA variance diagnostic."""

    records: list[dict[str, Any]] = []
    for period, group in rows.groupby("period", sort=True, observed=True):
        matrix = group.pivot_table(
            index="session", columns="pair", values="robust_net_payoff_bps", aggfunc="mean"
        )
        if matrix.empty:
            continue
        centered = matrix - matrix.mean(axis=0)
        values = centered.fillna(0.0).to_numpy(dtype=float)
        _, singular, _ = np.linalg.svd(values, full_matrices=False)
        variances = np.square(singular)
        denominator = float(variances.sum())
        explained = variances / denominator if denominator > 0 else np.zeros_like(variances)
        maximum = min(values.shape)
        for count in factor_counts:
            if count < 1 or count > maximum:
                continue
            records.append(
                {
                    "period": str(period),
                    "factor_count": int(count),
                    "cumulative_variance_explained": float(explained[:count].sum()),
                    "sessions": int(values.shape[0]),
                    "pairs": int(values.shape[1]),
                    "period_specific_loadings": True,
                    "changes_primary_decomposition": False,
                }
            )
    return pd.DataFrame.from_records(records)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_rerun_identity(primary: Path, rerun: Path) -> dict[str, Any]:
    """Compare deterministic machine-readable artifacts byte for byte."""

    suffixes = {".csv", ".json", ".parquet"}
    ignored = {"artifact_manifest.json", "exact_rerun_identity.json", "independent_audit.json"}
    primary_files = {
        path.relative_to(primary).as_posix(): path
        for path in primary.rglob("*")
        if path.is_file() and path.suffix in suffixes and path.name not in ignored
    }
    rerun_files = {
        path.relative_to(rerun).as_posix(): path
        for path in rerun.rglob("*")
        if path.is_file() and path.suffix in suffixes and path.name not in ignored
    }
    missing = sorted(set(primary_files) - set(rerun_files))
    extra = sorted(set(rerun_files) - set(primary_files))
    mismatches = sorted(
        name
        for name in set(primary_files) & set(rerun_files)
        if _file_hash(primary_files[name]) != _file_hash(rerun_files[name])
    )
    return {
        "byte_identical": not missing and not extra and not mismatches,
        "missing": missing,
        "extra": extra,
        "hash_mismatches": mismatches,
        "compared_files": len(set(primary_files) & set(rerun_files)),
    }
