"""Independent audit for Profitable Loop Episode Anatomy V1.

The auditor reconstructs headline quantities from frozen source ledgers and
detailed artifacts.  It never imports the experiment runner or its summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
CONTRACT_PATH = WORK / "contracts/20260717-profitable-loop-episode-anatomy-v1.json"
DEFAULT_PRIMARY = WORK / "artifacts/20260717-profitable-loop-episode-anatomy-v1/primary"
DEFAULT_RERUN = WORK / "artifacts/20260717-profitable-loop-episode-anatomy-v1/exact_rerun"
IGNORED_IDENTITY_FILES = {
    "artifact_manifest.json",
    "exact_rerun_identity.json",
    "independent_audit.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def source_paths(contract: dict[str, Any]) -> dict[str, Path]:
    root = (CONTRACT_PATH.parent / contract["inputs"]["root"]).resolve()
    return {name: root / name for name in contract["inputs"]["files"]}


def independent_census(
    states: pd.DataFrame, episodes: pd.DataFrame, calendar: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    positive = states[states["hindsight_payoff_state"].eq("positive")].copy()
    positive["pair"] = positive["loop_id"].astype(str) + "|" + positive["orientation"].astype(str)
    counts = positive.groupby(["period", "score_session"])["pair"].nunique()
    period_counts = {
        str(period_value): {
            "positive_sessions": int(len(group)),
            "multi_pair_positive_sessions": int(group.ge(2).sum()),
            "multi_pair_share": float(group.ge(2).mean()),
        }
        for period_value, group in counts.groupby(level="period")
    }

    calendars: dict[str, dict[str, int]] = {}
    calendar_frame = calendar.copy()
    calendar_frame["period"] = calendar_frame["period"].astype(str)
    calendar_frame["score_session"] = pd.to_datetime(calendar_frame["score_session"]).dt.strftime(
        "%Y-%m-%d"
    )
    for calendar_period_value, group in calendar_frame.drop_duplicates().groupby("period"):
        period = str(calendar_period_value)
        sessions = sorted(str(value) for value in group["score_session"].unique())
        calendars[period] = {session: index for index, session in enumerate(sessions)}

    source = episodes.copy()
    source["period"] = source["period"].astype(str)
    source["onset"] = pd.to_datetime(source["hindsight_estimated_onset"]).dt.strftime("%Y-%m-%d")
    source["end"] = pd.to_datetime(source["hindsight_estimated_end"]).dt.strftime("%Y-%m-%d")
    unions: list[dict[str, Any]] = []
    for (episode_period_value, orientation_value), group in source.groupby(
        ["period", "orientation"], sort=True
    ):
        period = str(episode_period_value)
        orientation = str(orientation_value)
        positions = calendars[period]
        ordered = group.assign(
            start_index=group["onset"].map(positions),
            end_index=group["end"].map(positions),
        ).sort_values(["start_index", "end_index", "episode_id"])
        current: list[int] = []
        current_end = -1
        for index, episode_row in ordered.iterrows():
            if current and int(episode_row["start_index"]) > current_end + 1:
                members = ordered.loc[current]
                unions.append(
                    {
                        "period": period,
                        "orientation": orientation,
                        "ids": "|".join(sorted(members["episode_id"])),
                        "loops": int(members["loop_id"].nunique()),
                    }
                )
                current = []
                current_end = -1
            current.append(cast(int, index))
            current_end = max(current_end, int(episode_row["end_index"]))
        if current:
            members = ordered.loc[current]
            unions.append(
                {
                    "period": period,
                    "orientation": orientation,
                    "ids": "|".join(sorted(members["episode_id"])),
                    "loops": int(members["loop_id"].nunique()),
                }
            )
    union_frame = pd.DataFrame.from_records(unions)
    leader_shares: list[float] = []
    unavailable = 0
    for leader_row in union_frame[union_frame["loops"].gt(1)].itertuples():
        ids = str(leader_row.ids).split("|")
        members = source[source["episode_id"].isin(ids)]
        payoff = members.groupby("loop_id")["total_episode_payoff_bps"].sum()
        payoff = payoff[payoff.gt(0)]
        if payoff.empty or payoff.sum() <= 0:
            unavailable += 1
        else:
            leader_shares.append(float(payoff.max() / payoff.sum()))
    result = {
        "strict_positive_rows": int(len(positive)),
        "positive_sessions": int(len(counts)),
        "multi_pair_positive_sessions": int(counts.ge(2).sum()),
        "multi_pair_share": float(counts.ge(2).mean()),
        "periods": period_counts,
        "same_regime_episodes": int(len(union_frame)),
        "single_loop": int(union_frame["loops"].eq(1).sum()),
        "multi_loop": int(union_frame["loops"].gt(1).sum()),
        "leader_share_available": len(leader_shares),
        "leader_share_unavailable": unavailable,
        "leader_share_median": float(np.median(leader_shares)),
        "majority": int(np.sum(np.asarray(leader_shares) > 0.5)),
        "over_80pct": int(np.sum(np.asarray(leader_shares) > 0.8)),
    }
    return result, union_frame


def independent_episode_ledgers(
    episodes: pd.DataFrame, calendar: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Rebuild all three frozen episode levels without runner code."""

    calendar_frame = calendar.copy()
    calendar_frame["period"] = calendar_frame["period"].astype(str)
    calendar_frame["score_session"] = pd.to_datetime(calendar_frame["score_session"]).dt.strftime(
        "%Y-%m-%d"
    )
    calendars = {
        str(period): sorted(group["score_session"].drop_duplicates().astype(str))
        for period, group in calendar_frame.groupby("period", sort=True)
    }
    positions = {
        period: {session: index for index, session in enumerate(sessions)}
        for period, sessions in calendars.items()
    }
    pair = episodes.copy()
    pair["period"] = pair["period"].astype(str)
    pair["onset"] = pd.to_datetime(pair["hindsight_estimated_onset"]).dt.strftime("%Y-%m-%d")
    pair["end"] = pd.to_datetime(pair["hindsight_estimated_end"]).dt.strftime("%Y-%m-%d")
    pair = pair.rename(
        columns={"episode_id": "pair_episode_id", "loop_id": "loop", "orientation": "regime"}
    )
    pair["orientation"] = pair["regime"]
    pair["pair"] = pair["loop"].astype(str) + "|" + pair["regime"].astype(str)
    pair["episode_id"] = pair["pair_episode_id"]
    pair["source_pair_episode_ids"] = pair["pair_episode_id"]

    def union(group_columns: list[str], prefix: str) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        grouping = ["period", *group_columns]
        for group_key, group in pair.groupby(grouping, sort=True, observed=True):
            period = str(group_key[0] if isinstance(group_key, tuple) else group_key)
            ordered = group.assign(
                start_index=group["onset"].map(positions[period]),
                end_index=group["end"].map(positions[period]),
            ).sort_values(["start_index", "end_index", "pair_episode_id"])
            batches: list[list[int]] = []
            current: list[int] = []
            current_end = -1
            for index, row in ordered.iterrows():
                if current and int(row["start_index"]) > current_end + 1:
                    batches.append(current)
                    current = []
                    current_end = -1
                current.append(cast(int, index))
                current_end = max(current_end, int(row["end_index"]))
            if current:
                batches.append(current)
            for batch in batches:
                members = ordered.loc[batch]
                records.append(
                    {
                        "episode_id": f"{prefix}_{len(records) + 1:04d}",
                        "period": period,
                        "onset": str(members["onset"].min()),
                        "end": str(members["end"].max()),
                        "source_pair_episode_ids": "|".join(
                            sorted(members["pair_episode_id"].astype(str))
                        ),
                        "loops": "|".join(sorted(members["loop"].astype(str).unique())),
                        "regimes": "|".join(sorted(members["regime"].astype(str).unique())),
                        "loop_count": int(members["loop"].nunique()),
                        "regime_count": int(members["regime"].nunique()),
                    }
                )
        return pd.DataFrame.from_records(records)

    same = union(["regime"], "same_regime")
    shared = union([], "shared_market")
    return pair.reset_index(drop=True), same, shared, calendars


def expected_membership(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    calendars: dict[str, list[str]],
    *,
    panel_key: str | None,
    ledger_key: str | None,
) -> pd.Series:
    mapping: dict[tuple[str, ...], list[str]] = {}
    for row in ledger.itertuples(index=False):
        period = str(row.period)
        sessions = calendars[period]
        start = sessions.index(str(row.onset))
        end = sessions.index(str(row.end))
        key_value = str(getattr(row, ledger_key)) if ledger_key else ""
        for session in sessions[start : end + 1]:
            key = (period, session, key_value) if panel_key else (period, session)
            mapping.setdefault(key, []).append(str(row.episode_id))
    values: list[object] = []
    for row in panel.itertuples(index=False):
        key = (
            (str(row.period), str(row.session), str(getattr(row, panel_key)))
            if panel_key
            else (str(row.period), str(row.session))
        )
        ids = sorted(set(mapping.get(key, [])))
        values.append("|".join(ids) if ids else pd.NA)
    return pd.Series(values, index=panel.index, dtype="string")


def decode_token(token: int) -> tuple[str, str]:
    current = token % 8
    quotient = token // 8
    previous_1 = quotient % 9
    previous_2 = quotient // 9
    history_2 = "unavailable" if previous_1 == 8 else f"state_{previous_1}>state_{current}"
    history_3 = (
        "unavailable"
        if previous_1 == 8 or previous_2 == 8
        else f"state_{previous_2}>state_{previous_1}>state_{current}"
    )
    return history_2, history_3


def independent_occurrences(trades: pd.DataFrame) -> pd.DataFrame:
    filled = trades[
        trades["model_name"].eq("no_payoff_state_filter")
        & trades["horizon"].eq(24)
        & trades["status"].eq("filled")
    ].copy()
    filled["period"] = filled["period"].astype(str)
    filled["session"] = pd.to_datetime(filled["score_session"]).dt.strftime("%Y-%m-%d")
    filled["loop"] = filled["loop_id"].astype(str)
    filled["regime"] = filled["orientation"].astype(str)
    filled["stock"] = filled["stock_id"].astype(str)
    decoded = [decode_token(int(token)) for token in filled["history_token"]]
    filled["regime_history_2"] = [value[0] for value in decoded]
    filled["regime_history_3"] = [value[1] for value in decoded]
    filled["net_payoff_bps"] = filled["primary_net_payoff_bps"].clip(-500.0, 500.0)

    def unique_context(values: pd.Series) -> str:
        clean = sorted(values.dropna().astype(str).unique())
        return clean[0] if len(clean) == 1 else "unavailable"

    result = (
        filled.groupby(
            ["period", "session", "loop", "regime", "stock"],
            observed=True,
            dropna=False,
        )
        .agg(
            net_payoff_bps=("net_payoff_bps", "mean"),
            raw_fill_count=("fill_id", "size"),
            history_token_count=("history_token", "nunique"),
            history_2_variant_count_within_occurrence=("regime_history_2", "nunique"),
            history_3_variant_count_within_occurrence=("regime_history_3", "nunique"),
            regime_history_2=("regime_history_2", unique_context),
            regime_history_3=("regime_history_3", unique_context),
            clock_phase=("state_change_phase", unique_context),
            month=("month", "first"),
        )
        .reset_index()
    )
    result["history_mixed_within_occurrence"] = result["history_token_count"].gt(1)
    result["history_2_mixed_within_occurrence"] = result[
        "history_2_variant_count_within_occurrence"
    ].gt(1)
    result["history_3_mixed_within_occurrence"] = result[
        "history_3_variant_count_within_occurrence"
    ].gt(1)
    result["pair"] = result["loop"] + "|" + result["regime"]
    result["positive_payoff"] = result["net_payoff_bps"].clip(lower=0.0)
    return result


def support_gate(group: pd.DataFrame) -> bool:
    shares = group["stock"].value_counts(normalize=True)
    return bool(
        len(group) >= 30
        and group["session"].nunique() >= 15
        and group["stock"].nunique() >= 8
        and group["month"].nunique() >= 3
        and len(shares)
        and shares.max() <= 0.30
    )


def independent_bh(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(array), np.nan)
    finite = np.flatnonzero(np.isfinite(array))
    if not len(finite):
        return result
    order = finite[np.argsort(array[finite])]
    adjusted = np.minimum.accumulate(
        (array[order] * len(order) / np.arange(1, len(order) + 1))[::-1]
    )[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def independent_sequence_seed(*parts: object, base_seed: int) -> int:
    identity = "|".join(str(part) for part in parts).encode()
    offset = int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")
    return (base_seed + offset) % (2**32)


def independent_session_block_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    seed: int,
    resamples: int,
    block_length: int,
) -> tuple[float, float, float, int]:
    sessions = sorted(set(left["session"].astype(str)) | set(right["session"].astype(str)))
    if not sessions:
        return math.nan, math.nan, math.nan, 0

    def aggregate(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        by_session = frame.groupby(frame["session"].astype(str))["net_payoff_bps"].agg(
            ["sum", "count"]
        )
        sums = by_session["sum"].reindex(sessions, fill_value=0.0).to_numpy(dtype=float)
        counts = by_session["count"].reindex(sessions, fill_value=0).to_numpy(dtype=float)
        return sums, counts

    left_sum, left_count = aggregate(left)
    right_sum, right_count = aggregate(right)
    generator = np.random.default_rng(seed)
    width = len(sessions)
    blocks = math.ceil(width / block_length)
    offset = np.arange(block_length)
    values: list[float] = []
    for _ in range(resamples):
        starts = generator.integers(0, width, size=blocks)
        selected = ((starts[:, None] + offset[None, :]) % width).ravel()[:width]
        left_n = float(left_count[selected].sum())
        right_n = float(right_count[selected].sum())
        if left_n and right_n:
            values.append(
                float(left_sum[selected].sum() / left_n - right_sum[selected].sum() / right_n)
            )
    if not values:
        return math.nan, math.nan, math.nan, 0
    array = np.asarray(values, dtype=float)
    tail = min(int(np.sum(array <= 0.0)), int(np.sum(array >= 0.0)))
    p_value = min(1.0, 2.0 * (1.0 + tail) / (1.0 + len(array)))
    return (
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
        p_value,
        len(array),
    )


def audit_episode_phase(relative: int, duration: int) -> str:
    if relative == 0:
        return "onset"
    if duration > 1 and relative == duration - 1:
        return "decay"
    fraction = relative / max(duration - 1, 1)
    if fraction <= 1 / 3:
        return "early"
    if fraction <= 2 / 3:
        return "middle"
    return "late"


def independent_block_null(states: pd.DataFrame, resamples: int) -> dict[str, float]:
    frame = states[["period", "score_session", "loop_id", "orientation"]].copy()
    frame["period"] = frame["period"].astype(str)
    frame["session"] = pd.to_datetime(frame["score_session"]).dt.strftime("%Y-%m-%d")
    frame["pair"] = frame["loop_id"].astype(str) + "|" + frame["orientation"].astype(str)
    frame["positive"] = states["hindsight_payoff_state"].eq("positive").to_numpy()
    frame = frame.sort_values(["period", "pair", "session"]).reset_index(drop=True)
    group_indices = [
        group.index.to_numpy() for _, group in frame.groupby(["period", "pair"], sort=True)
    ]
    session_codes, sessions = pd.factorize(frame["period"] + "\x1f" + frame["session"], sort=True)
    rng = np.random.default_rng(20260717)
    shares = np.empty(resamples, dtype=float)
    for replicate in range(resamples):
        shifted = np.zeros(len(frame), dtype=bool)
        for indices in group_indices:
            sequence = frame.loc[indices, "positive"].to_numpy(dtype=bool)
            offset = int(rng.integers(0, max(1, math.ceil(len(sequence) / 5))))
            shifted[indices] = np.roll(sequence, (offset * 5) % len(sequence))
        session_counts = np.bincount(
            session_codes, weights=shifted.astype(int), minlength=len(sessions)
        )
        positive_counts = session_counts[session_counts > 0]
        shares[replicate] = float(np.mean(positive_counts >= 2))
    observed_counts = frame.groupby(["period", "session"])["positive"].sum()
    observed = float(observed_counts[observed_counts.gt(0)].ge(2).mean())
    return {
        "observed": observed,
        "null_mean": float(shares.mean()),
        "lower_95": float(np.quantile(shares, 0.025)),
        "upper_95": float(np.quantile(shares, 0.975)),
        "one_sided_p": float((1 + np.sum(shares >= observed)) / (1 + len(shares))),
    }


def independent_coactivation_network(states: pd.DataFrame, resamples: int) -> pd.DataFrame:
    frame = states[["period", "score_session", "loop_id", "orientation"]].copy()
    frame["period"] = frame["period"].astype(str)
    frame["session"] = pd.to_datetime(frame["score_session"]).dt.strftime("%Y-%m-%d")
    frame["pair"] = frame["loop_id"].astype(str) + "|" + frame["orientation"].astype(str)
    frame["positive"] = states["hindsight_payoff_state"].eq("positive").to_numpy()
    pairs = sorted(frame["pair"].unique())
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    edges = [(left, right) for left in range(len(pairs)) for right in range(left + 1, len(pairs))]
    observed = np.zeros((len(pairs), len(pairs)), dtype=float)
    eligible = np.zeros_like(observed)
    period_matrices: list[tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]] = []
    for _, group in frame.groupby("period", sort=True):
        sessions = sorted(group["session"].unique())
        session_index = {session: index for index, session in enumerate(sessions)}
        values = np.full((len(sessions), len(pairs)), np.nan)
        for row in group.itertuples(index=False):
            values[session_index[str(row.session)], pair_index[str(row.pair)]] = float(
                bool(row.positive)
            )
        mask = np.isfinite(values)
        binary = np.nan_to_num(values, nan=0.0)
        observed += binary.T @ binary
        eligible += mask.astype(float).T @ mask.astype(float)
        indices = [np.flatnonzero(mask[:, column]) for column in range(len(pairs))]
        sequences = [binary[index, column].copy() for column, index in enumerate(indices)]
        period_matrices.append((binary, indices, sequences))
    simulations = np.zeros((len(edges), resamples), dtype=float)
    rng = np.random.default_rng(20260717)
    for replicate in range(resamples):
        total = np.zeros_like(observed)
        for template, indices, sequences in period_matrices:
            shifted = np.zeros_like(template)
            for column, (eligible_indices, sequence) in enumerate(
                zip(indices, sequences, strict=True)
            ):
                if not len(sequence):
                    continue
                offset = int(rng.integers(0, max(1, math.ceil(len(sequence) / 5))))
                shifted[eligible_indices, column] = np.roll(sequence, (offset * 5) % len(sequence))
            total += shifted.T @ shifted
        for edge_number, (left, right) in enumerate(edges):
            simulations[edge_number, replicate] = total[left, right]
    records: list[dict[str, Any]] = []
    for edge_number, (left, right) in enumerate(edges):
        null = simulations[edge_number]
        observed_value = int(observed[left, right])
        records.append(
            {
                "pair_left": pairs[left],
                "pair_right": pairs[right],
                "eligible_sessions": int(eligible[left, right]),
                "observed_coactivations": observed_value,
                "block_null_expected_coactivations": float(null.mean()),
                "block_null_95th_percentile": float(np.quantile(null, 0.95)),
                "excess_coactivations": float(observed_value - null.mean()),
                "one_sided_empirical_p": float(
                    (1 + np.sum(null >= observed_value)) / (1 + len(null))
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def artifact_identity(primary: Path, rerun: Path) -> dict[str, Any]:
    suffixes = {".parquet", ".csv", ".json"}
    left = {
        path.relative_to(primary).as_posix(): sha256(path)
        for path in primary.rglob("*")
        if path.is_file() and path.suffix in suffixes and path.name not in IGNORED_IDENTITY_FILES
    }
    right = {
        path.relative_to(rerun).as_posix(): sha256(path)
        for path in rerun.rglob("*")
        if path.is_file() and path.suffix in suffixes and path.name not in IGNORED_IDENTITY_FILES
    }
    return {
        "compared_files": len(set(left) & set(right)),
        "missing": sorted(set(left) - set(right)),
        "extra": sorted(set(right) - set(left)),
        "mismatches": sorted(name for name in set(left) & set(right) if left[name] != right[name]),
    }


def audit(primary: Path, rerun: Path) -> dict[str, Any]:
    contract: dict[str, Any] = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    paths = source_paths(contract)
    source_hashes = {name: sha256(path) for name, path in paths.items()}
    source_hash_pass = source_hashes == contract["inputs"]["files"]
    states = pd.read_parquet(paths["v2_hindsight_episode_states.parquet"])
    source_episodes = pd.read_parquet(paths["v2_hindsight_episode_diagnostics.parquet"])
    features = pd.read_parquet(paths["v2_causal_feature_panel.parquet"])
    census, independent_unions = independent_census(
        states, source_episodes, features[["period", "score_session"]]
    )
    independent_pair, independent_same, independent_shared, calendars = independent_episode_ledgers(
        source_episodes, features[["period", "score_session"]]
    )
    census_pass = (
        census["positive_sessions"] == 322
        and census["multi_pair_positive_sessions"] == 210
        and census["same_regime_episodes"] == 107
        and census["single_loop"] == 65
        and census["multi_loop"] == 42
        and census["majority"] == 33
        and census["over_80pct"] == 11
    )

    null = independent_block_null(states, 2000)
    stored_null = json.loads((primary / "block_circular_null_summary.json").read_text())
    null_pass = (
        abs(null["null_mean"] - stored_null["block_null_mean_multi_pair_positive_session_share"])
        < 1e-12
        and abs(null["one_sided_p"] - stored_null["one_sided_empirical_p"]) < 1e-12
    )
    panel = pd.read_parquet(primary / "session_regime_loop_orientation_panel.parquet")
    independent_network = independent_coactivation_network(states, 2000)
    stored_network = pd.read_parquet(primary / "coactivation_excess_table.parquet")
    network_compare = stored_network.merge(
        independent_network,
        on=["pair_left", "pair_right"],
        how="outer",
        suffixes=("_stored", "_expected"),
        indicator=True,
    )
    network_errors: list[float] = []
    for column in [
        "eligible_sessions",
        "observed_coactivations",
        "block_null_expected_coactivations",
        "block_null_95th_percentile",
        "excess_coactivations",
        "one_sided_empirical_p",
    ]:
        network_errors.append(
            float(
                (network_compare[f"{column}_stored"] - network_compare[f"{column}_expected"])
                .abs()
                .max()
            )
        )
    node_supported = panel[panel["robust_net_payoff_bps"].notna()].copy()
    node_supported["positive_payoff"] = node_supported["robust_net_payoff_bps"].clip(lower=0.0)
    expected_node = (
        node_supported.groupby("pair", observed=True)
        .agg(
            node_total_positive_payoff=("positive_payoff", "sum"),
            node_regime=("regime", "first"),
            node_loop=("loop", "first"),
            node_supported_sessions=("session", "nunique"),
        )
        .reset_index()
    )
    leader_records: list[dict[str, Any]] = []
    for episode in independent_same.itertuples(index=False):
        episode_members = independent_pair[
            independent_pair["pair_episode_id"].isin(
                str(episode.source_pair_episode_ids).split("|")
            )
        ]
        loop_payoff = (
            episode_members.groupby("loop", observed=True)["total_episode_payoff_bps"]
            .sum()
            .clip(lower=0.0)
        )
        leader_value = float(loop_payoff.max()) if len(loop_payoff) else math.nan
        node_leaders = (
            set(loop_payoff[loop_payoff.eq(leader_value)].index.astype(str))
            if math.isfinite(leader_value) and leader_value > 0
            else set()
        )
        for member in episode_members[["pair", "loop"]].drop_duplicates().itertuples(index=False):
            leader_records.append(
                {
                    "pair": str(member.pair),
                    "episode_id": str(episode.episode_id),
                    "is_leader": str(member.loop) in node_leaders,
                }
            )
    independent_leaders = pd.DataFrame.from_records(leader_records)
    leader_node = (
        independent_leaders.groupby("pair", observed=True)
        .agg(
            node_episode_appearances=("episode_id", "nunique"),
            node_leader_episodes=("is_leader", "sum"),
        )
        .reset_index()
    )
    leader_node["node_leader_frequency"] = (
        leader_node["node_leader_episodes"] / leader_node["node_episode_appearances"]
    )
    expected_node = expected_node.merge(leader_node, on="pair", how="left", validate="one_to_one")
    expected_node[["node_episode_appearances", "node_leader_episodes"]] = expected_node[
        ["node_episode_appearances", "node_leader_episodes"]
    ].fillna(0)
    network_node_pass = True
    stored_network = pd.read_parquet(primary / "coactivation_excess_table.parquet")
    for side in ["left", "right"]:
        side_expected = expected_node.add_suffix(f"_{side}").rename(
            columns={f"pair_{side}": f"pair_{side}"}
        )
        side_compare = stored_network.merge(
            side_expected,
            on=f"pair_{side}",
            how="left",
            suffixes=("_stored", "_expected"),
            validate="many_to_one",
        )
        for column in [
            "node_total_positive_payoff",
            "node_supported_sessions",
            "node_episode_appearances",
            "node_leader_episodes",
            "node_leader_frequency",
        ]:
            difference = (
                side_compare[f"{column}_{side}_stored"] - side_compare[f"{column}_{side}_expected"]
            ).abs()
            finite = difference.dropna()
            if len(finite):
                network_errors.append(float(finite.max()))
        network_node_pass &= bool(
            side_compare[f"node_regime_{side}_stored"]
            .astype(str)
            .eq(side_compare[f"node_regime_{side}_expected"].astype(str))
            .all()
            and side_compare[f"node_loop_{side}_stored"]
            .astype(str)
            .eq(side_compare[f"node_loop_{side}_expected"].astype(str))
            .all()
        )
    network_pass = bool(
        network_compare["_merge"].eq("both").all()
        and max(network_errors) <= 1e-12
        and network_node_pass
    )
    expected_pair_membership = expected_membership(
        panel,
        independent_pair,
        calendars,
        panel_key="pair",
        ledger_key="pair",
    )
    expected_same_membership = expected_membership(
        panel,
        independent_same,
        calendars,
        panel_key="regime",
        ledger_key="regimes",
    )
    expected_shared_membership = expected_membership(
        panel,
        independent_shared,
        calendars,
        panel_key=None,
        ledger_key=None,
    )

    def memberships_match(column: str, expected: pd.Series) -> bool:
        actual = panel[column].astype("string").fillna("<missing>")
        return bool(actual.equals(expected.astype("string").fillna("<missing>")))

    episode_membership_pass = bool(
        memberships_match("hindsight_pair_episode_id", expected_pair_membership)
        and memberships_match("same_regime_episode_id", expected_same_membership)
        and memberships_match("shared_session_episode_id", expected_shared_membership)
    )
    supported = panel[panel["robust_net_payoff_bps"].notna()].copy()
    common = supported.groupby(["period", "session"])["robust_net_payoff_bps"].transform("median")
    after_common = supported["robust_net_payoff_bps"] - common
    regime = after_common.groupby(
        [supported["period"], supported["session"], supported["regime"]]
    ).transform("median")
    residual = supported["robust_net_payoff_bps"] - common - regime
    component_errors = {
        "common": float((common - supported["common_component"]).abs().max()),
        "regime": float((regime - supported["regime_component"]).abs().max()),
        "loop_excess": float((residual - supported["loop_excess_component"]).abs().max()),
        "identity": float(
            (
                supported["common_component"]
                + supported["regime_component"]
                + supported["loop_excess_component"]
                - supported["robust_net_payoff_bps"]
            )
            .abs()
            .max()
        ),
    }
    components_pass = max(component_errors.values()) <= 1e-10

    pair_mapping_pass = bool(
        (panel["pair"] == panel["loop"].astype(str) + "|" + panel["orientation"].astype(str)).all()
        and (panel["regime"].astype(str) == panel["orientation"].astype(str)).all()
    )
    trades = pd.read_parquet(paths["v2_trade_decisions.parquet"])
    independent_occurrence = independent_occurrences(trades)
    history_expectations: dict[tuple[str, str, str, str], tuple[bool, bool, bool, str, str]] = {}
    for key, group in independent_occurrence.groupby(
        ["period", "session", "loop", "regime"], sort=True, observed=True
    ):
        history_2 = group["regime_history_2"].astype(str)
        history_3 = group["regime_history_3"].astype(str)
        within_mixed = bool(group["history_mixed_within_occurrence"].any())
        within_mixed_2 = bool(group["history_2_mixed_within_occurrence"].any())
        within_mixed_3 = bool(group["history_3_mixed_within_occurrence"].any())
        history_2_available = bool(
            not within_mixed_2 and history_2.ne("unavailable").all() and history_2.nunique() == 1
        )
        history_3_available = bool(
            not within_mixed_3 and history_3.ne("unavailable").all() and history_3.nunique() == 1
        )
        history_mixed = bool(within_mixed or history_2.nunique() > 1 or history_3.nunique() > 1)
        history_key = (str(key[0]), str(key[1]), str(key[2]), str(key[3]))
        history_expectations[history_key] = (
            history_2_available,
            history_3_available,
            history_mixed,
            str(history_2.iloc[0]) if history_2_available else "unavailable",
            str(history_3.iloc[0]) if history_3_available else "unavailable",
        )
    expected_history = [
        history_expectations.get(
            (str(row.period), str(row.session), str(row.loop), str(row.regime)),
            (False, False, False, "unavailable", "unavailable"),
        )
        for row in panel.itertuples(index=False)
    ]
    expected_history_2_available = pd.Series(
        [value[0] for value in expected_history], index=panel.index
    )
    expected_history_3_available = pd.Series(
        [value[1] for value in expected_history], index=panel.index
    )
    expected_history_mixed = pd.Series([value[2] for value in expected_history], index=panel.index)
    expected_history_2 = pd.Series([value[3] for value in expected_history], index=panel.index)
    expected_history_3 = pd.Series([value[4] for value in expected_history], index=panel.index)
    missingness_pass = bool(
        panel.loc[panel["robust_net_payoff_bps"].isna(), "robust_net_payoff_bps"].isna().all()
        and not panel.loc[~panel["positive_pair_available"], "positive_pair_flag"].any()
        and panel["history_available"].astype(bool).equals(expected_history_2_available)
        and panel["history_3_available"].astype(bool).equals(expected_history_3_available)
        and panel["history_mixed"].equals(expected_history_mixed)
        and panel["regime_history_2"].astype(str).equals(expected_history_2)
        and panel["regime_history_3"].astype(str).equals(expected_history_3)
    )

    stored_pair = pd.read_parquet(primary / "pair_positive_episode_ledger.parquet")
    same = pd.read_parquet(primary / "same_regime_episode_ledger.parquet")
    shared = pd.read_parquet(primary / "shared_market_episode_ledger.parquet")

    def episode_signatures(frame: pd.DataFrame) -> set[tuple[str, str, str, str, str]]:
        return {
            (
                str(row.episode_id),
                str(row.period),
                str(row.onset),
                str(row.end),
                str(row.source_pair_episode_ids),
            )
            for row in frame.itertuples(index=False)
        }

    episode_ledger_identity_pass = bool(
        episode_signatures(stored_pair) == episode_signatures(independent_pair)
        and episode_signatures(same) == episode_signatures(independent_same)
        and episode_signatures(shared) == episode_signatures(independent_shared)
    )
    occurrence = pd.read_parquet(primary / "occurrence_share_table.parquet")
    mixed_history = independent_occurrence["history_mixed_within_occurrence"]
    mixed_history_2 = independent_occurrence["history_2_mixed_within_occurrence"]
    mixed_history_3 = independent_occurrence["history_3_mixed_within_occurrence"]
    occurrence_identity_pass = bool(
        len(independent_occurrence)
        == int(contract["occurrence"]["expected_stock_capped_occurrences"])
        and int(independent_occurrence["raw_fill_count"].sum())
        == int(contract["occurrence"]["expected_raw_fills"])
        and not independent_occurrence.duplicated(
            ["period", "session", "loop", "regime", "stock"]
        ).any()
        and independent_occurrence.loc[mixed_history_2, "regime_history_2"].eq("unavailable").all()
        and independent_occurrence.loc[mixed_history_3, "regime_history_3"].eq("unavailable").all()
    )
    leader_identity_errors: list[float] = []
    occurrence_errors: list[float] = []
    payoff_occurrence_identity_errors: list[float] = []
    component_episode_errors: list[float] = []
    leader_identity_pass = True
    for episode in independent_same.itertuples(index=False):
        ids = str(episode.source_pair_episode_ids).split("|")
        members = independent_pair[independent_pair["pair_episode_id"].isin(ids)]
        loop_net_payoff = members.groupby("loop", observed=True)["total_episode_payoff_bps"].sum()
        payoff = loop_net_payoff.clip(lower=0.0)
        total_positive = float(payoff.sum())
        leader_value = float(payoff.max()) if total_positive > 0 else math.nan
        episode_leaders = (
            sorted(payoff[payoff.eq(leader_value)].index.astype(str))
            if math.isfinite(leader_value) and leader_value > 0
            else []
        )
        expected_leader = "|".join(episode_leaders) if episode_leaders else "unavailable"
        expected_share = leader_value / total_positive if total_positive > 0 else math.nan
        expected_category = (
            "SINGLE_LOOP_EPISODE"
            if int(cast(Any, episode.loop_count)) == 1
            else "UNKNOWN_MULTI_LOOP"
            if not math.isfinite(expected_share)
            else "EXTREME_LOOP_DOMINANCE"
            if expected_share > 0.8
            else "MAJORITY_LOOP_DOMINANCE"
            if expected_share > 0.5
            else "DIFFUSE_MULTI_LOOP"
        )
        stored_episode = same[same["episode_id"].eq(episode.episode_id)].iloc[0]
        leader_identity_pass &= bool(
            str(stored_episode["final_dominant_loop"]) == expected_leader
            and str(stored_episode["anatomy_category"]) == expected_category
        )
        if math.isfinite(expected_share):
            leader_identity_errors.append(
                abs(float(stored_episode["final_leader_share"]) - expected_share)
            )

        period_sessions = calendars[str(episode.period)]
        start = period_sessions.index(str(episode.onset))
        end = period_sessions.index(str(episode.end))
        episode_sessions = period_sessions[start : end + 1]
        member_pairs = set(members["pair"].astype(str))
        episode_occurrence = independent_occurrence[
            independent_occurrence["period"].eq(str(episode.period))
            & independent_occurrence["session"].isin(episode_sessions)
            & independent_occurrence["pair"].isin(member_pairs)
        ]
        occurrence_counts = episode_occurrence.groupby("loop", observed=True).size()
        occurrence_total = int(occurrence_counts.sum())
        supported_episode_cells = supported[
            supported["period"].astype(str).eq(str(episode.period))
            & supported["session"].isin(episode_sessions)
            & supported["pair"].isin(member_pairs)
        ]
        supported_cell_keys = set(
            zip(
                supported_episode_cells["session"].astype(str),
                supported_episode_cells["pair"].astype(str),
                strict=False,
            )
        )
        occurrence_cell_keys = set(
            zip(
                episode_occurrence["session"].astype(str),
                episode_occurrence["pair"].astype(str),
                strict=False,
            )
        )
        occurrence_population_complete = bool(
            supported_cell_keys and supported_cell_keys.issubset(occurrence_cell_keys)
        )
        stored_loops = occurrence[
            occurrence["episode_level"].eq("same_regime")
            & occurrence["episode_id"].eq(episode.episode_id)
        ]
        for loop_name in sorted(members["loop"].astype(str).unique()):
            stored_loop = stored_loops[stored_loops["loop"].eq(loop_name)].iloc[0]
            expected_occurrence_share = (
                float(occurrence_counts.get(loop_name, 0)) / occurrence_total
                if occurrence_total and occurrence_population_complete
                else math.nan
            )
            expected_payoff_share = (
                float(payoff.get(loop_name, 0.0)) / total_positive
                if total_positive > 0
                else math.nan
            )
            for actual, expected in [
                (stored_loop["occurrence_share"], expected_occurrence_share),
                (stored_loop["positive_payoff_share"], expected_payoff_share),
            ]:
                if math.isfinite(expected):
                    occurrence_errors.append(abs(float(actual) - expected))
                else:
                    occurrence_errors.append(0.0 if pd.isna(actual) else math.inf)
            occurrence_errors.append(
                0.0
                if bool(stored_loop["occurrence_population_complete"])
                == occurrence_population_complete
                else math.inf
            )
            occurrence_count = int(occurrence_counts.get(loop_name, 0))
            expected_mean_payoff = (
                float(loop_net_payoff.get(loop_name, 0.0)) / occurrence_count
                if occurrence_count > 0 and occurrence_population_complete
                else math.nan
            )
            if math.isfinite(expected_mean_payoff):
                payoff_occurrence_identity_errors.extend(
                    [
                        abs(
                            float(stored_loop["mean_payoff_per_occurrence"]) - expected_mean_payoff
                        ),
                        abs(float(stored_loop["payoff_occurrence_identity_error"])),
                        abs(
                            float(stored_loop["total_loop_payoff"])
                            - occurrence_count * float(stored_loop["mean_payoff_per_occurrence"])
                        ),
                    ]
                )
            else:
                payoff_occurrence_identity_errors.append(
                    0.0
                    if pd.isna(stored_loop["mean_payoff_per_occurrence"])
                    and pd.isna(stored_loop["payoff_occurrence_identity_error"])
                    else math.inf
                )
            if expected_occurrence_share > 0 and math.isfinite(expected_payoff_share):
                expected_efficiency = expected_payoff_share / expected_occurrence_share
                occurrence_errors.append(
                    abs(float(stored_loop["leader_efficiency"]) - expected_efficiency)
                )

        selected_components = supported[
            supported["period"].astype(str).eq(str(episode.period))
            & supported["session"].isin(episode_sessions)
            & supported["pair"].isin(member_pairs)
        ]
        masses = np.array(
            [
                selected_components[column].clip(lower=0.0).sum()
                for column in [
                    "common_component",
                    "regime_component",
                    "loop_excess_component",
                ]
            ],
            dtype=float,
        )
        expected_component_shares = masses / masses.sum() if masses.sum() > 0 else masses * np.nan
        actual_component_shares = stored_episode[
            [
                "common_positive_contribution_share",
                "regime_positive_contribution_share",
                "loop_excess_positive_contribution_share",
            ]
        ].to_numpy(dtype=float)
        finite_component = np.isfinite(expected_component_shares)
        if finite_component.any():
            component_episode_errors.extend(
                np.abs(
                    actual_component_shares[finite_component]
                    - expected_component_shares[finite_component]
                ).tolist()
            )

    leader = occurrence[occurrence["is_final_leader"] & occurrence["leader_efficiency"].notna()]
    efficiency_error = float(
        (leader["leader_efficiency"] - leader["positive_payoff_share"] / leader["occurrence_share"])
        .abs()
        .max()
    )
    anatomy_pass = (
        len(same) == len(independent_unions)
        and int(same["anatomy_category"].eq("SINGLE_LOOP_EPISODE").sum()) == 65
        and int(same["anatomy_category"].eq("EXTREME_LOOP_DOMINANCE").sum()) == 11
        and int(same["anatomy_category"].eq("MAJORITY_LOOP_DOMINANCE").sum()) == 22
        and int(same["anatomy_category"].eq("DIFFUSE_MULTI_LOOP").sum()) == 8
        and efficiency_error <= 1e-12
        and leader_identity_pass
        and (not leader_identity_errors or max(leader_identity_errors) <= 1e-12)
        and (not occurrence_errors or max(occurrence_errors) <= 1e-12)
        and (
            not payoff_occurrence_identity_errors or max(payoff_occurrence_identity_errors) <= 1e-12
        )
        and (not component_episode_errors or max(component_episode_errors) <= 1e-12)
    )

    timeline = pd.read_parquet(primary / "episode_session_timeline.parquet")
    sampled_timeline_checks: list[dict[str, Any]] = []
    calendar_sets = {
        str(period): sorted(group["score_session"].astype(str).unique())
        for period, group in features.groupby("period")
    }
    for episode in same.head(10).itertuples():
        sample = timeline[timeline["episode_id"].eq(episode.episode_id)]
        sessions = calendar_sets[str(episode.period)]
        start = sessions.index(str(episode.onset))
        end = sessions.index(str(episode.end))
        expected_rows = min(len(sessions), end + 11) - max(0, start - 10)
        sampled_timeline_checks.append(
            {
                "episode_id": episode.episode_id,
                "rows": len(sample),
                "expected_rows": expected_rows,
                "onset_rows": int(sample["relative_session_to_onset"].eq(0).sum()),
                "pass": len(sample) == expected_rows
                and int(sample["relative_session_to_onset"].eq(0).sum()) == 1,
            }
        )
    timeline_pass = all(item["pass"] for item in sampled_timeline_checks)

    early = pd.read_parquet(primary / "early_leader_table.parquet")
    early_identity_pass = True
    early_numeric_errors: list[float] = []
    early_error_details: list[dict[str, Any]] = []
    for episode in independent_same.itertuples(index=False):
        ids = str(episode.source_pair_episode_ids).split("|")
        members = independent_pair[independent_pair["pair_episode_id"].isin(ids)]
        member_pairs = set(members["pair"].astype(str))
        sessions = calendars[str(episode.period)]
        start = sessions.index(str(episode.onset))
        end = sessions.index(str(episode.end))
        episode_sessions = sessions[start : end + 1]
        selected = supported[
            supported["period"].astype(str).eq(str(episode.period))
            & supported["session"].isin(episode_sessions)
            & supported["pair"].isin(member_pairs)
        ].copy()
        selected["positive_payoff"] = selected["robust_net_payoff_bps"].clip(lower=0.0)
        episode_occurrences = independent_occurrence[
            independent_occurrence["period"].eq(str(episode.period))
            & independent_occurrence["session"].isin(episode_sessions)
            & independent_occurrence["pair"].isin(member_pairs)
        ]
        early_supported_cells = set(
            zip(
                selected["session"].astype(str),
                selected["pair"].astype(str),
                strict=False,
            )
        )
        early_occurrence_cells = set(
            zip(
                episode_occurrences["session"].astype(str),
                episode_occurrences["pair"].astype(str),
                strict=False,
            )
        )
        early_occurrence_complete = bool(
            early_supported_cells and early_supported_cells.issubset(early_occurrence_cells)
        )
        final_payoff = (
            members.groupby("loop", observed=True)["total_episode_payoff_bps"].sum().clip(lower=0.0)
        )
        total_final = float(final_payoff.sum())
        final_max = float(final_payoff.max()) if total_final > 0 else math.nan
        final_leaders = (
            set(final_payoff[final_payoff.eq(final_max)].index.astype(str))
            if math.isfinite(final_max) and final_max > 0
            else set()
        )
        realised_total = float(selected["positive_payoff"].sum())
        realised_final_leader = float(
            selected.loc[selected["loop"].astype(str).isin(final_leaders), "positive_payoff"].sum()
        )
        duration = len(episode_sessions)
        checkpoints = {
            "first_session": 1,
            "first_two": 2,
            "first_three": 3,
            "first_25pct": max(1, int(math.ceil(duration * 0.25))),
            "first_50pct": max(1, int(math.ceil(duration * 0.50))),
        }
        for checkpoint, prefix_length in checkpoints.items():
            if prefix_length > duration:
                continue
            prefix_sessions = episode_sessions[:prefix_length]
            prefix = selected[selected["session"].isin(prefix_sessions)]
            provisional = prefix.groupby("loop", observed=True)["positive_payoff"].sum()
            for loop_name in members["loop"].astype(str).unique():
                if loop_name not in provisional:
                    provisional.loc[loop_name] = 0.0
            prefix_total = float(provisional.sum())
            provisional_max = float(provisional.max()) if len(provisional) else math.nan
            provisional_leaders = (
                set(provisional[provisional.eq(provisional_max)].index.astype(str))
                if prefix_total > 0
                else set()
            )
            provisional_ranks = provisional.rank(method="min", ascending=False)
            top_three = set(provisional_ranks[provisional_ranks.le(3)].index.astype(str))
            final_rank_frame = pd.DataFrame(
                {
                    "final_payoff": final_payoff,
                    "provisional_payoff": provisional.reindex(final_payoff.index, fill_value=0.0),
                }
            )
            final_ranks = final_rank_frame["final_payoff"].rank(method="average", ascending=False)
            prefix_ranks = final_rank_frame["provisional_payoff"].rank(
                method="average", ascending=False
            )
            expected_rank_correlation = (
                float(final_ranks.corr(prefix_ranks, method="spearman"))
                if len(final_rank_frame) >= 2
                and final_ranks.nunique() > 1
                and prefix_ranks.nunique() > 1
                else math.nan
            )
            prefix_occurrences = independent_occurrence[
                independent_occurrence["period"].eq(str(episode.period))
                & independent_occurrence["session"].isin(prefix_sessions)
                & independent_occurrence["pair"].isin(member_pairs)
            ]
            occurrence_by_loop = prefix_occurrences.groupby("loop", observed=True).size()
            total_prefix_occurrences = int(occurrence_by_loop.sum())
            provisional_share = provisional_max / prefix_total if prefix_total > 0 else math.nan
            provisional_occurrence_share = (
                float(occurrence_by_loop.reindex(sorted(provisional_leaders), fill_value=0).sum())
                / total_prefix_occurrences
                if total_prefix_occurrences > 0
                and provisional_leaders
                and early_occurrence_complete
                else math.nan
            )
            provisional_efficiency = (
                provisional_share / provisional_occurrence_share
                if provisional_occurrence_share > 0
                else math.nan
            )
            after = selected[~selected["session"].isin(prefix_sessions)]
            remaining = float(after["positive_payoff"].sum())
            leader_remaining = float(
                after.loc[after["loop"].astype(str).isin(final_leaders), "positive_payoff"].sum()
            )
            stored_early = early[
                early["episode_id"].eq(episode.episode_id) & early["checkpoint"].eq(checkpoint)
            ].iloc[0]
            expected_provisional = (
                "|".join(sorted(provisional_leaders)) if provisional_leaders else ""
            )
            early_identity_pass &= bool(
                str(stored_early["provisional_leaders"]) == expected_provisional
                and (
                    pd.isna(stored_early["top_one_match"])
                    if not provisional_leaders
                    else bool(stored_early["top_one_match"])
                    == bool(provisional_leaders & final_leaders)
                )
                and (
                    pd.isna(stored_early["top_three_inclusion"])
                    if not provisional_leaders
                    else bool(stored_early["top_three_inclusion"])
                    == bool(top_three & final_leaders)
                )
            )
            for field, actual, expected in [
                (
                    "rank_correlation_with_final",
                    stored_early["rank_correlation_with_final"],
                    expected_rank_correlation,
                ),
                (
                    "provisional_leader_payoff_share",
                    stored_early["provisional_leader_payoff_share"],
                    provisional_share,
                ),
                (
                    "provisional_leader_occurrence_share",
                    stored_early["provisional_leader_occurrence_share"],
                    provisional_occurrence_share,
                ),
                (
                    "provisional_leader_efficiency",
                    stored_early["provisional_leader_efficiency"],
                    provisional_efficiency,
                ),
                ("payoff_remaining", stored_early["payoff_remaining"], remaining),
                (
                    "fraction_final_payoff_remaining",
                    stored_early["fraction_final_payoff_remaining"],
                    remaining / realised_total if realised_total > 0 else math.nan,
                ),
                (
                    "final_leader_payoff_remaining",
                    stored_early["final_leader_payoff_remaining"],
                    leader_remaining,
                ),
                (
                    "fraction_final_leader_payoff_remaining",
                    stored_early["fraction_final_leader_payoff_remaining"],
                    leader_remaining / realised_final_leader
                    if realised_final_leader > 0
                    else math.nan,
                ),
            ]:
                if math.isfinite(expected):
                    error = abs(float(actual) - expected)
                    early_numeric_errors.append(error)
                    early_error_details.append(
                        {
                            "episode_id": str(episode.episode_id),
                            "checkpoint": checkpoint,
                            "field": field,
                            "actual": float(actual),
                            "expected": expected,
                            "absolute_error": error,
                        }
                    )
                elif not pd.isna(actual):
                    early_numeric_errors.append(math.inf)
                    early_error_details.append(
                        {
                            "episode_id": str(episode.episode_id),
                            "checkpoint": checkpoint,
                            "field": field,
                            "actual": float(actual),
                            "expected": "unavailable",
                            "absolute_error": math.inf,
                        }
                    )
    payoff_remaining_pass = bool(
        early["payoff_remaining"].ge(-1e-12).all()
        and early["fraction_final_payoff_remaining"].dropna().ge(-1e-12).all()
        and early["fraction_final_payoff_remaining"].dropna().le(1 + 1e-12).all()
        and early["fraction_final_leader_payoff_remaining"].dropna().le(1 + 1e-12).all()
    )
    early_prefix_pass = bool(
        early.loc[early["checkpoint"].eq("first_session"), "prefix_sessions"].eq(1).all()
        and early.loc[early["checkpoint"].eq("first_two"), "prefix_sessions"].eq(2).all()
        and early.loc[early["checkpoint"].eq("first_three"), "prefix_sessions"].eq(3).all()
        and early_identity_pass
        and (not early_numeric_errors or max(early_numeric_errors) <= 1e-10)
    )

    persistence = pd.read_parquet(primary / "leader_persistence_table.parquet")
    persistence_pass = True
    for episode in independent_same.itertuples(index=False):
        ids = str(episode.source_pair_episode_ids).split("|")
        members = independent_pair[independent_pair["pair_episode_id"].isin(ids)]
        member_pairs = set(members["pair"].astype(str))
        sessions = calendars[str(episode.period)]
        start = sessions.index(str(episode.onset))
        end = sessions.index(str(episode.end))
        episode_sessions = sessions[start : end + 1]
        leaders_by_session: dict[str, set[str]] = {}
        support_by_session: dict[str, bool] = {}
        positive_by_session: dict[str, bool] = {}
        for session in episode_sessions:
            current = panel[
                panel["period"].astype(str).eq(str(episode.period))
                & panel["session"].eq(session)
                & panel["pair"].isin(member_pairs)
            ]
            support_by_session[session] = bool(current["robust_net_payoff_bps"].notna().any())
            positive_by_session[session] = bool(
                (current["positive_pair_flag"] & current["positive_pair_available"]).any()
            )
            payoff = (
                current[current["robust_net_payoff_bps"].notna()]
                .assign(
                    positive_payoff=lambda frame: frame["robust_net_payoff_bps"].clip(lower=0.0)
                )
                .groupby("loop", observed=True)["positive_payoff"]
                .sum()
            )
            leaders_by_session[session] = (
                set(payoff[payoff.eq(payoff.max())].index.astype(str))
                if len(payoff) and payoff.sum() > 0 and positive_by_session[session]
                else set()
            )
        for index, session in enumerate(episode_sessions):
            for lag in [1, 2, 3]:
                stored_row = persistence[
                    persistence["episode_id"].eq(episode.episode_id)
                    & persistence["session"].eq(session)
                    & persistence["lag"].eq(lag)
                ].iloc[0]
                if index + lag >= len(episode_sessions):
                    expected_status = "episode_boundary"
                    expected_top_one: object = math.nan
                else:
                    future = episode_sessions[index + lag]
                    if not support_by_session[session] or not support_by_session[future]:
                        expected_status = "missing_support"
                        expected_top_one = math.nan
                    elif not positive_by_session[session] or not leaders_by_session[session]:
                        expected_status = "no_positive_pair_current_session"
                        expected_top_one = math.nan
                    elif not positive_by_session[future] or not leaders_by_session[future]:
                        expected_status = "no_positive_pair_next_session"
                        expected_top_one = math.nan
                    else:
                        expected_status = "same_shared_episode"
                        expected_top_one = bool(
                            leaders_by_session[session] & leaders_by_session[future]
                        )
                persistence_pass &= str(stored_row["status"]) == expected_status
                if isinstance(expected_top_one, bool):
                    persistence_pass &= bool(stored_row["top_one_persistence"]) == expected_top_one
                else:
                    persistence_pass &= pd.isna(stored_row["top_one_persistence"])

    no_filter = trades[
        trades["model_name"].eq("no_payoff_state_filter") & trades["status"].eq("filled")
    ]
    token_sample = no_filter[["history_token", "previous_state_1", "state"]].head(500)
    decoded_current = token_sample["history_token"].astype(int) % 8
    decoded_previous = (token_sample["history_token"].astype(int) // 8) % 9
    sequence_decode_pass = bool(
        decoded_current.eq(token_sample["state"].astype(int)).all()
        and decoded_previous.eq(token_sample["previous_state_1"].astype(int)).all()
    )
    sequence_census = pd.read_parquet(primary / "regime_sequence_census.parquet")
    sequence = pd.read_parquet(primary / "sequence_increment_table.parquet")
    loop_increment = pd.read_parquet(primary / "loop_increment_within_sequence_table.parquet")
    interaction_stability = pd.read_parquet(primary / "sequence_interaction_stability.parquet")
    sequence_errors: list[float] = []
    sequence_bootstrap_errors: list[float] = []
    sequence_support_pass = True
    sequence_clock_pass = True
    bootstrap_resamples = int(contract["support"]["bootstrap_resamples"])
    bootstrap_block_length = int(contract["support"]["bootstrap_block_length_sessions"])
    bootstrap_seed = int(contract["support"]["bootstrap_seed"])
    for row in sequence.itertuples(index=False):
        sequence_column = f"regime_history_{int(cast(Any, row.sequence_length))}"
        same_regime = independent_occurrence[
            independent_occurrence["period"].eq(str(row.period))
            & independent_occurrence["regime"].eq(str(row.regime))
            & ~independent_occurrence[sequence_column].eq("unavailable")
        ]
        is_loop = same_regime["loop"].eq(str(row.loop))
        is_sequence = same_regime[sequence_column].eq(str(row.sequence))
        target = same_regime[is_loop & is_sequence]
        other_sequence = same_regime[is_loop & ~is_sequence]
        other_loop = same_regime[~is_loop & is_sequence]
        expected_sequence_increment = float(
            target["net_payoff_bps"].mean() - other_sequence["net_payoff_bps"].mean()
        )
        expected_loop_increment = float(
            target["net_payoff_bps"].mean() - other_loop["net_payoff_bps"].mean()
        )
        if math.isfinite(expected_sequence_increment):
            sequence_errors.append(
                abs(float(cast(Any, row.sequence_increment_bps)) - expected_sequence_increment)
            )
        if math.isfinite(expected_loop_increment):
            sequence_errors.append(
                abs(
                    float(cast(Any, row.loop_increment_within_sequence_bps))
                    - expected_loop_increment
                )
            )
        sequence_support_pass &= bool(row.target_support) == support_gate(target)
        expected_comparison_support = bool(
            support_gate(target) and support_gate(other_sequence) and support_gate(other_loop)
        )
        sequence_support_pass &= (
            bool(row.increment_comparison_supported) == expected_comparison_support
        )
        if expected_comparison_support:
            seed_parts = (
                str(row.period),
                str(row.loop),
                str(row.regime),
                int(cast(Any, row.sequence_length)),
                str(row.sequence),
            )
            expected_sequence_bootstrap = independent_session_block_difference(
                target,
                other_sequence,
                seed=independent_sequence_seed(
                    *seed_parts, "sequence_increment", base_seed=bootstrap_seed
                ),
                resamples=bootstrap_resamples,
                block_length=bootstrap_block_length,
            )
            expected_loop_bootstrap = independent_session_block_difference(
                target,
                other_loop,
                seed=independent_sequence_seed(
                    *seed_parts, "loop_increment", base_seed=bootstrap_seed
                ),
                resamples=bootstrap_resamples,
                block_length=bootstrap_block_length,
            )
        else:
            expected_sequence_bootstrap = (math.nan, math.nan, math.nan, 0)
            expected_loop_bootstrap = (math.nan, math.nan, math.nan, 0)
        stored_bootstrap = [
            row.sequence_increment_bootstrap_lower_95,
            row.sequence_increment_bootstrap_upper_95,
            row.sequence_increment_p_value,
            row.sequence_increment_bootstrap_valid_resamples,
            row.loop_increment_bootstrap_lower_95,
            row.loop_increment_bootstrap_upper_95,
            row.loop_increment_p_value,
            row.loop_increment_bootstrap_valid_resamples,
        ]
        for actual, expected in zip(
            stored_bootstrap,
            [*expected_sequence_bootstrap, *expected_loop_bootstrap],
            strict=True,
        ):
            if isinstance(expected, int):
                sequence_bootstrap_errors.append(abs(int(cast(Any, actual)) - expected))
            elif math.isfinite(expected):
                sequence_bootstrap_errors.append(abs(float(cast(Any, actual)) - expected))
            else:
                sequence_bootstrap_errors.append(0.0 if pd.isna(actual) else math.inf)
        target_clock = target["clock_phase"].astype("string")
        clock_available = target_clock.notna() & ~target_clock.eq("unavailable")
        clock_counts = target_clock[clock_available].value_counts().sort_index()
        expected_dominant = (
            sorted(clock_counts[clock_counts.eq(clock_counts.max())].index.astype(str))[0]
            if len(clock_counts)
            else "unavailable"
        )
        expected_counts_json = json.dumps(
            {str(key): int(value) for key, value in clock_counts.items()},
            sort_keys=True,
            separators=(",", ":"),
        )
        sequence_clock_pass &= (
            int(cast(Any, row.target_clock_phase_available_rows)) == int(clock_available.sum())
            and int(cast(Any, row.target_clock_phase_missing_rows)) == int((~clock_available).sum())
            and str(row.target_dominant_clock_phase) == expected_dominant
            and str(row.target_clock_phase_counts_json) == expected_counts_json
        )

    for row in sequence_census.itertuples(index=False):
        column = f"regime_history_{int(cast(Any, row.sequence_length))}"
        target = independent_occurrence[
            independent_occurrence["period"].eq(str(row.period))
            & independent_occurrence["loop"].eq(str(row.loop))
            & independent_occurrence["regime"].eq(str(row.regime))
            & independent_occurrence[column].eq(str(row.sequence))
        ]
        phases = target["clock_phase"].astype("string")
        available = phases.notna() & ~phases.eq("unavailable")
        sequence_clock_pass &= int(cast(Any, row.clock_phase_available_rows)) == int(
            available.sum()
        )

    four_way = pd.read_parquet(primary / "four_way_counterfactual_tables.parquet")
    four_way_errors: list[float] = []
    for row in four_way.itertuples(index=False):
        sequence_column = f"regime_history_{int(cast(Any, row.sequence_length))}"
        same_regime = independent_occurrence[
            independent_occurrence["period"].eq(str(row.period))
            & independent_occurrence["regime"].eq(str(row.regime))
            & ~independent_occurrence[sequence_column].eq("unavailable")
        ]
        is_loop = same_regime["loop"].eq(str(row.loop))
        is_sequence = same_regime[sequence_column].eq(str(row.sequence))
        masks = {
            "1": is_loop & is_sequence,
            "2": is_loop & ~is_sequence,
            "3": ~is_loop & is_sequence,
            "4": ~is_loop & ~is_sequence,
        }
        expected_group = same_regime[masks[str(row.counterfactual_group)]]
        four_way_errors.append(abs(int(cast(Any, row.rows)) - len(expected_group)))
        if len(expected_group):
            four_way_errors.append(
                abs(
                    float(cast(Any, row.mean_net_payoff_bps))
                    - float(expected_group["net_payoff_bps"].mean())
                )
            )
        sequence_support_pass &= bool(row.supported) == support_gate(expected_group)
        phases = expected_group["clock_phase"].astype("string")
        available = phases.notna() & ~phases.eq("unavailable")
        sequence_clock_pass &= int(cast(Any, row.clock_phase_available_rows)) == int(
            available.sum()
        )

    sequence_pass = bool(
        set(sequence["sequence_length"]) == {2, 3}
        and not sequence["sequence"].astype(str).str.contains("future", case=False).any()
        and not interaction_stability["multiplicity_controlled_interaction_both_periods"].any()
        and len(loop_increment) == len(sequence)
        and np.allclose(
            loop_increment["loop_increment_within_sequence_bps"],
            sequence["loop_increment_within_sequence_bps"],
            equal_nan=True,
        )
        and (not sequence_errors or max(sequence_errors) <= 1e-12)
        and (not sequence_bootstrap_errors or max(sequence_bootstrap_errors) <= 1e-12)
        and (not four_way_errors or max(four_way_errors) <= 1e-12)
        and sequence_support_pass
        and sequence_clock_pass
        and sequence_decode_pass
        and np.allclose(
            sequence["sequence_increment_fdr_q_value"],
            independent_bh(
                sequence["sequence_increment_p_value"].where(
                    sequence["increment_comparison_supported"]
                )
            ),
            equal_nan=True,
        )
        and np.allclose(
            sequence["loop_increment_fdr_q_value"],
            independent_bh(
                sequence["loop_increment_p_value"].where(sequence["increment_comparison_supported"])
            ),
            equal_nan=True,
        )
        and not sequence.loc[
            ~sequence["increment_comparison_supported"], "interaction_fdr_pass"
        ].any()
    )
    four_way_pass = bool(
        set(four_way["counterfactual_group"].astype(str)) == {"1", "2", "3", "4"}
        and (
            four_way["pair"] == four_way["loop"].astype(str) + "|" + four_way["regime"].astype(str)
        ).all()
    )

    named = pd.read_parquet(primary / "named_pair_deep_dive_tables.parquet")
    named_pairs_pass = set(named["pair"]) == {
        "cycle_04|state_4",
        "cycle_07|state_5",
        "cycle_04|state_2",
        "cycle_07|state_6",
    }
    named_errors: list[float] = []
    named_raw_p_values: list[float] = []
    raw_route = pd.read_parquet(paths["causal_route_signal_path_events.parquet"])
    raw_route["pair"] = (
        raw_route["candidate"]
        .astype(str)
        .str.replace("|state2", "|state_2", regex=False)
        .str.replace("|state4", "|state_4", regex=False)
        .str.replace("|state5", "|state_5", regex=False)
        .str.replace("|state6", "|state_6", regex=False)
    )
    raw_route["period"] = raw_route["period"].astype(str)
    raw_route["session"] = pd.to_datetime(raw_route["session_date"]).dt.strftime("%Y-%m-%d")
    route_expected_records: list[dict[str, Any]] = []
    for pair_episode in independent_pair.itertuples(index=False):
        period_sessions = calendars[str(pair_episode.period)]
        start = period_sessions.index(str(pair_episode.onset))
        end = period_sessions.index(str(pair_episode.end))
        selected_route = raw_route[
            raw_route["period"].eq(str(pair_episode.period))
            & raw_route["pair"].eq(str(pair_episode.pair))
            & raw_route["session"].isin(period_sessions[start : end + 1])
        ].copy()
        if selected_route.empty:
            continue
        positions = {session: index for index, session in enumerate(period_sessions)}

        selected_route["episode_phase"] = [
            audit_episode_phase(positions[session] - start, end - start + 1)
            for session in selected_route["session"]
        ]
        for (phase_value, topology), group in selected_route.groupby(
            ["episode_phase", "path_topology"], sort=True, observed=True
        ):
            route_expected_records.append(
                {
                    "period": str(pair_episode.period),
                    "episode_id": str(pair_episode.episode_id),
                    "pair": str(pair_episode.pair),
                    "episode_phase": str(phase_value),
                    "path_topology": str(topology),
                    "frequency": len(group),
                    "event_payoff": float(
                        group["terminal_route_event_next_open_else_fixed__net_bps"].mean()
                    ),
                    "after_payoff": float(
                        -group[
                            "terminal_route_event_next_open_else_fixed__paired_difference_bps"
                        ].mean()
                    ),
                }
            )
    route_expected = pd.DataFrame.from_records(route_expected_records)
    route = pd.read_parquet(primary / "route_topology_anatomy.parquet")
    route_compare = route.merge(
        route_expected,
        on=["period", "episode_id", "pair", "episode_phase", "path_topology"],
        how="outer",
        suffixes=("_stored", "_expected"),
        indicator=True,
    )
    route_errors = [
        float(
            (route_compare["frequency_stored"] - route_compare["frequency_expected"]).abs().max()
        ),
        float(
            (route_compare["payoff_before_route_event_bps"] - route_compare["event_payoff"])
            .abs()
            .max()
        ),
        float(
            (route_compare["payoff_after_route_event_bps"] - route_compare["after_payoff"])
            .abs()
            .max()
        ),
    ]
    route_pass = bool(
        route["outcome_only"].all()
        and route_compare["_merge"].eq("both").all()
        and max(route_errors) <= 1e-10
    )

    for named_row in named.itertuples(index=False):
        pair = str(named_row.pair)
        pair_episodes = independent_pair[independent_pair["pair"].eq(pair)]
        component_parts: list[pd.DataFrame] = []
        for pair_episode in pair_episodes.itertuples(index=False):
            component_parts.append(
                supported[
                    supported["period"].astype(str).eq(str(pair_episode.period))
                    & supported["pair"].eq(pair)
                    & supported["session"].between(str(pair_episode.onset), str(pair_episode.end))
                ]
            )
        components = pd.concat(component_parts, ignore_index=True)
        for actual, column in [
            (named_row.common_component_mean, "common_component"),
            (named_row.regime_component_mean, "regime_component"),
            (named_row.loop_specific_excess_mean, "loop_excess_component"),
        ]:
            named_errors.append(abs(float(cast(Any, actual)) - float(components[column].mean())))
        pair_occurrences = independent_occurrence[independent_occurrence["pair"].eq(pair)]
        named_pairs_pass &= int(cast(Any, named_row.pair_episode_count)) == len(pair_episodes)
        named_pairs_pass &= int(cast(Any, named_row.target_pair_component_rows)) == len(components)
        named_pairs_pass &= int(cast(Any, named_row.occurrence_support_rows)) == len(
            pair_occurrences
        )
        nonzero = pair_occurrences.loc[pair_occurrences["net_payoff_bps"].ne(0), "net_payoff_bps"]
        p_value = (
            float(stats.wilcoxon(nonzero, alternative="two-sided").pvalue)
            if len(nonzero) >= 2
            else math.nan
        )
        named_raw_p_values.append(p_value)
        if math.isfinite(p_value):
            named_errors.append(abs(float(cast(Any, named_row.named_family_raw_p_value)) - p_value))
        pair_route = route[route["pair"].eq(pair)]
        topology_counts = pair_route.groupby("path_topology")["frequency"].sum()
        topology_total = int(pair_route["frequency"].sum())
        for actual, topology in [
            (named_row.exact_completion_rows, "exact_parent_completion"),
            (named_row.incompatible_transition_rows, "incompatible_first_transition"),
            (named_row.expected_leg_diversion_rows, "expected_leg_then_diversion"),
        ]:
            named_errors.append(abs(int(cast(Any, actual)) - int(topology_counts.get(topology, 0))))
        if topology_total:
            named_errors.append(
                abs(
                    float(cast(Any, named_row.exact_completion_share))
                    - int(topology_counts.get("exact_parent_completion", 0)) / topology_total
                )
            )

    finite_named = [
        (index, value) for index, value in enumerate(named_raw_p_values) if math.isfinite(value)
    ]
    expected_holm = np.full(len(named_raw_p_values), np.nan)
    if finite_named:
        order = sorted(finite_named, key=lambda item: item[1])
        running = 0.0
        for rank, (index, value) in enumerate(order):
            running = max(running, value * (len(order) - rank))
            expected_holm[index] = min(running, 1.0)
    named_errors.extend(
        np.abs(named["named_family_holm_adjusted_p_value"].to_numpy() - expected_holm)[
            np.isfinite(expected_holm)
        ].tolist()
    )
    named_pass = bool(named_pairs_pass and (not named_errors or max(named_errors) <= 1e-10))
    raw_path = pd.read_parquet(paths["sequential_path_diagnostics.parquet"])
    raw_path["period"] = raw_path["period"].astype(str)
    raw_path["session"] = pd.to_datetime(raw_path["session_date"]).dt.strftime("%Y-%m-%d")
    raw_path["pair"] = (
        raw_path["top_loop"].astype(str)
        + "|state_"
        + raw_path["anchor_state"].astype(int).astype(str)
    )
    expected_path_counts: list[dict[str, Any]] = []
    for episode in independent_same.itertuples(index=False):
        ids = str(episode.source_pair_episode_ids).split("|")
        members = independent_pair[independent_pair["pair_episode_id"].isin(ids)]
        selected_path = raw_path[
            raw_path["period"].eq(str(episode.period))
            & raw_path["pair"].isin(set(members["pair"].astype(str)))
            & raw_path["session"].between(str(episode.onset), str(episode.end))
        ]
        if len(selected_path):
            expected_path_counts.append(
                {"episode_id": str(episode.episode_id), "expected_rows": len(selected_path)}
            )
    expected_path = pd.DataFrame.from_records(expected_path_counts)
    deterioration = pd.read_parquet(primary / "sequential_path_deterioration.parquet")
    stored_path = (
        deterioration.groupby("episode_id", observed=True)["rows"]
        .sum()
        .rename("stored_rows")
        .reset_index()
    )
    path_compare = expected_path.merge(stored_path, on="episode_id", how="outer", indicator=True)
    independent_path_rows: list[pd.DataFrame] = []
    for episode in independent_same.itertuples(index=False):
        member_ids = str(episode.source_pair_episode_ids).split("|")
        members = independent_pair[independent_pair["pair_episode_id"].isin(member_ids)]
        loop_payoff = members.groupby("loop", observed=True)["total_episode_payoff_bps"].sum()
        positive_loop_payoff = loop_payoff[loop_payoff.gt(0.0)]
        leader_value = float(positive_loop_payoff.max()) if len(positive_loop_payoff) else math.nan
        path_leaders = (
            set(positive_loop_payoff[positive_loop_payoff.eq(leader_value)].index.astype(str))
            if math.isfinite(leader_value)
            else set()
        )
        period_sessions = calendars[str(episode.period)]
        start = period_sessions.index(str(episode.onset))
        end = period_sessions.index(str(episode.end))
        positions = {session: index for index, session in enumerate(period_sessions)}
        for pair in members["pair"].astype(str).unique():
            selected_path = raw_path[
                raw_path["period"].eq(str(episode.period))
                & raw_path["pair"].eq(pair)
                & raw_path["session"].isin(period_sessions[start : end + 1])
            ].copy()
            if selected_path.empty:
                continue
            path_panel = panel[
                [
                    "period",
                    "session",
                    "pair",
                    "robust_net_payoff_bps",
                    "positive_pair_flag",
                    "positive_pair_available",
                ]
            ].copy()
            path_panel["period"] = path_panel["period"].astype(str)
            selected_path = selected_path.merge(
                path_panel[
                    [
                        "period",
                        "session",
                        "pair",
                        "robust_net_payoff_bps",
                        "positive_pair_flag",
                        "positive_pair_available",
                    ]
                ],
                on=["period", "session", "pair"],
                how="left",
                validate="many_to_one",
            )
            selected_path["episode_id"] = str(episode.episode_id)
            selected_path["episode_phase"] = [
                audit_episode_phase(positions[session] - start, end - start + 1)
                for session in selected_path["session"]
            ]
            selected_path["episode_role"] = np.select(
                [
                    selected_path["top_loop"].astype(str).isin(path_leaders),
                    selected_path["robust_net_payoff_bps"].gt(0)
                    | (
                        selected_path["positive_pair_flag"].fillna(False)
                        & selected_path["positive_pair_available"].fillna(False)
                    ),
                ],
                ["final_episode_leader", "non_leading_positive"],
                default="negative_or_neutral_same_episode",
            )
            selected_path["negative_path_tail"] = selected_path["path_class"].isin(
                ["timing_failure", "no_usable_move"]
            )
            independent_path_rows.append(selected_path)
    independent_path = pd.concat(independent_path_rows, ignore_index=True)
    path_numeric_errors: list[float] = []
    path_direction_pass = True
    path_keys = [
        "period",
        "episode_id",
        "pair",
        "episode_role",
        "episode_phase",
        "path_class",
    ]
    for path_key, group in independent_path.groupby(path_keys, sort=True, observed=True):
        path_period = str(path_key[0])
        path_episode_id = str(path_key[1])
        path_pair = str(path_key[2])
        path_role = str(path_key[3])
        path_phase = str(path_key[4])
        path_class = str(path_key[5])
        stored_row = deterioration[
            deterioration["period"].astype(str).eq(path_period)
            & deterioration["episode_id"].eq(path_episode_id)
            & deterioration["pair"].eq(path_pair)
            & deterioration["episode_role"].eq(path_role)
            & deterioration["episode_phase"].eq(path_phase)
            & deterioration["path_class"].eq(path_class)
        ].iloc[0]
        role_population = independent_path[
            independent_path["episode_id"].eq(path_episode_id)
            & independent_path["pair"].eq(path_pair)
            & independent_path["episode_role"].eq(path_role)
        ]
        middle = role_population[role_population["episode_phase"].eq("middle")]
        late = role_population[role_population["episode_phase"].eq("late")]
        decay = role_population[role_population["episode_phase"].eq("decay")]
        middle_rate = float(middle["negative_path_tail"].mean()) if len(middle) else math.nan
        late_rate = float(late["negative_path_tail"].mean()) if len(late) else math.nan
        decay_rate = float(decay["negative_path_tail"].mean()) if len(decay) else math.nan
        expected_path_values = {
            "rows": float(len(group)),
            "negative_tail_frequency": float(group["negative_path_tail"].mean()),
            "timing_failure_frequency": float(group["path_class"].eq("timing_failure").mean()),
            "no_usable_move_frequency": float(group["path_class"].eq("no_usable_move").mean()),
            "median_mfe_bps_outcome_only": float(group["mfe_bps"].median()),
            "median_mae_bps_outcome_only": float(group["mae_bps"].median()),
            "mean_remaining_fixed_payoff_bps": float(group["net_return_bps"].mean()),
            "negative_tail_middle_rate": middle_rate,
            "negative_tail_late_rate": late_rate,
            "negative_tail_at_decay_rate": decay_rate,
        }
        for field, expected in expected_path_values.items():
            actual = float(cast(Any, stored_row[field]))
            if math.isfinite(expected):
                path_numeric_errors.append(abs(actual - expected))
            else:
                path_direction_pass &= pd.isna(stored_row[field])
        expected_rise = bool(late_rate > middle_rate) if len(middle) and len(late) else pd.NA
        path_direction_pass &= (
            pd.isna(stored_row["deterioration_rises_before_leader_decay"])
            if pd.isna(expected_rise)
            else bool(stored_row["deterioration_rises_before_leader_decay"]) == bool(expected_rise)
        )
    path_pass = bool(
        path_compare["_merge"].eq("both").all()
        and path_compare["expected_rows"].eq(path_compare["stored_rows"]).all()
        and {
            "final_episode_leader",
            "non_leading_positive",
            "negative_or_neutral_same_episode",
        }.issubset(set(deterioration["episode_role"]))
        and {"onset", "early", "middle", "late", "decay"}.issubset(
            set(deterioration["episode_phase"])
        )
        and deterioration["raw_path_score_status"].eq("unavailable_in_retained_diagnostic").all()
        and not deterioration["causal_input"].any()
        and path_direction_pass
        and (not path_numeric_errors or max(path_numeric_errors) <= 1e-10)
    )
    associations = pd.read_parquet(primary / "component_specific_indicator_association.parquet")
    forbidden = {"mfe_bps", "mae_bps", "hindsight_episode_id", "hindsight_payoff_state"}
    manifestation = pd.read_parquet(primary / "indicator_manifestation_tables.parquet")
    association_support_pass = True
    for row in associations.itertuples(index=False):
        population = panel
        if str(row.period) != "all":
            population = population[population["period"].astype(str).eq(str(row.period))]
        data = population[
            population[str(row.indicator)].notna() & population[str(row.component)].notna()
        ]
        association_keys = data[["period", "session", "loop", "regime"]].drop_duplicates().copy()
        association_keys["period"] = association_keys["period"].astype(str)
        matched = independent_occurrence.merge(
            association_keys,
            on=["period", "session", "loop", "regime"],
            how="inner",
            validate="many_to_one",
        )
        shares = matched["stock"].value_counts(normalize=True)
        expected_support = bool(
            len(data) >= 30
            and data["session"].nunique() >= 15
            and matched["stock"].nunique() >= 8
            and data["session"].astype(str).str[:7].nunique() >= 3
            and len(shares)
            and shares.max() <= 0.30
        )
        association_support_pass &= bool(row.support_passed) == expected_support
        association_support_pass &= (
            int(cast(Any, row.independent_stocks)) == matched["stock"].nunique()
        )

    def bh(values: pd.Series) -> np.ndarray:
        array = values.to_numpy(dtype=float)
        adjusted = np.full(len(array), np.nan)
        finite = np.flatnonzero(np.isfinite(array))
        order = finite[np.argsort(array[finite])]
        if len(order):
            ranked = np.minimum.accumulate(
                (array[order] * len(order) / np.arange(1, len(order) + 1))[::-1]
            )[::-1]
            adjusted[order] = np.minimum(ranked, 1.0)
        return adjusted

    association_q_error = float(
        np.nanmax(np.abs(associations["fdr_q_value"].to_numpy() - bh(associations["p_value"])))
    )
    manifestation_q_error = float(
        np.nanmax(np.abs(manifestation["fdr_q_value"].to_numpy() - bh(manifestation["p_value"])))
    )
    within_pair_manifestations = manifestation[
        manifestation["comparison"].eq("profitable_vs_unprofitable_same_pair")
    ]
    within_pair_identity_pass = bool(
        len(within_pair_manifestations)
        and within_pair_manifestations["pair"].astype(str).str.contains(r"\|").all()
        and (
            within_pair_manifestations["pair"].astype(str).str.split("|").str[0]
            == within_pair_manifestations["loop"].astype(str)
        ).all()
        and (
            within_pair_manifestations["pair"].astype(str).str.split("|").str[1]
            == within_pair_manifestations["regime"].astype(str)
        ).all()
    )
    indicator_pass = bool(
        not (set(associations["indicator"].astype(str)) & forbidden)
        and {
            "session_block_bootstrap_lower_95",
            "session_block_bootstrap_upper_95",
            "period_direction_consistent",
        }.issubset(associations.columns)
        and {
            "median_difference",
            "standardised_mean_difference",
            "distribution_overlap",
            "fdr_q_value",
        }.issubset(manifestation.columns)
        and association_support_pass
        and association_q_error <= 1e-12
        and manifestation_q_error <= 1e-12
        and within_pair_identity_pass
        and (
            associations["reportable_after_support_and_fdr"]
            == (associations["support_passed"] & associations["fdr_q_value"].le(0.05))
        ).all()
        and (
            manifestation["reportable_after_support_and_fdr"]
            == (manifestation["comparison_support_passed"] & manifestation["fdr_q_value"].le(0.05))
        ).all()
    )

    concentration = pd.read_parquet(primary / "stock_and_cohort_concentration.parquet")
    concentration_tolerance = 1e-12
    concentration_errors: list[float] = []
    for concentration_row in concentration.itertuples(index=False):
        ledger = (
            independent_same
            if concentration_row.episode_level == "same_regime"
            else independent_shared
        )
        episode_record = ledger[ledger["episode_id"].eq(concentration_row.episode_id)].iloc[0]
        ids = str(episode_record["source_pair_episode_ids"]).split("|")
        members = independent_pair[independent_pair["pair_episode_id"].isin(ids)]
        selected = independent_occurrence[
            independent_occurrence["period"].eq(str(episode_record["period"]))
            & independent_occurrence["session"].between(
                str(episode_record["onset"]), str(episode_record["end"])
            )
            & independent_occurrence["pair"].isin(set(members["pair"].astype(str)))
        ]
        stock_payoff = (
            selected.groupby("stock", observed=True)["positive_payoff"]
            .sum()
            .sort_values(ascending=False)
        )
        total = float(stock_payoff.sum())
        if total > 0:
            shares = stock_payoff / total
            expected_values = [
                float(shares.iloc[0]),
                float(shares.head(5).sum()),
                float(np.square(shares).sum()),
                float(total - stock_payoff.iloc[0]),
                float(total - stock_payoff.head(5).sum()),
            ]
            actual_values = [
                float(cast(Any, concentration_row.top_one_share)),
                float(cast(Any, concentration_row.top_five_share)),
                float(cast(Any, concentration_row.herfindahl_index)),
                float(cast(Any, concentration_row.after_remove_best_stock)),
                float(cast(Any, concentration_row.after_remove_top_five_stocks)),
            ]
            concentration_errors.extend(
                abs(actual - expected)
                for actual, expected in zip(actual_values, expected_values, strict=True)
            )
    concentration_pass = bool(
        concentration["top_one_share"]
        .dropna()
        .between(-concentration_tolerance, 1 + concentration_tolerance)
        .all()
        and concentration["top_five_share"]
        .dropna()
        .between(-concentration_tolerance, 1 + concentration_tolerance)
        .all()
        and concentration["herfindahl_index"]
        .dropna()
        .between(-concentration_tolerance, 1 + concentration_tolerance)
        .all()
        and concentration["beta_cohort_status"].eq("unavailable").all()
        and (not concentration_errors or max(concentration_errors) <= 1e-10)
    )
    cohort = pd.read_parquet(primary / "cohort_contribution_table.parquet")
    leave_one = pd.read_parquet(primary / "leave_one_stock_out_attribution.parquet")
    stock_removal_errors: list[float] = []
    stock_removal_pass = True
    for removal_row in leave_one.itertuples(index=False):
        ledger = (
            independent_same if removal_row.episode_level == "same_regime" else independent_shared
        )
        episode_record = ledger[ledger["episode_id"].eq(removal_row.episode_id)].iloc[0]
        member_ids = str(episode_record["source_pair_episode_ids"]).split("|")
        members = independent_pair[independent_pair["pair_episode_id"].isin(member_ids)]
        member_pairs = set(members["pair"].astype(str))
        episode_sessions = calendars[str(episode_record["period"])]
        start = episode_sessions.index(str(episode_record["onset"]))
        end = episode_sessions.index(str(episode_record["end"]))
        selected_sessions = episode_sessions[start : end + 1]
        all_occurrences = independent_occurrence[
            independent_occurrence["period"].eq(str(episode_record["period"]))
            & independent_occurrence["session"].isin(selected_sessions)
        ].copy()
        member_occurrences = all_occurrences[all_occurrences["pair"].isin(member_pairs)]
        supported_episode_panel = supported[
            supported["period"].astype(str).eq(str(episode_record["period"]))
            & supported["session"].isin(selected_sessions)
        ]
        supported_cells = set(
            zip(
                supported_episode_panel["session"].astype(str),
                supported_episode_panel["pair"].astype(str),
                strict=False,
            )
        )
        occurrence_cells = set(
            zip(
                all_occurrences["session"].astype(str),
                all_occurrences["pair"].astype(str),
                strict=False,
            )
        )
        complete = bool(supported_cells and supported_cells.issubset(occurrence_cells))
        stock_removal_pass &= bool(removal_row.component_summary_recalculated) == complete
        removed_stocks = set(str(removal_row.stock).split(";"))
        removed_positive = float(
            member_occurrences.loc[
                member_occurrences["stock"].astype(str).isin(removed_stocks), "positive_payoff"
            ].sum()
        )
        stock_removal_errors.append(
            abs(float(cast(Any, removal_row.removed_positive_payoff)) - removed_positive)
        )
        if not complete:
            stock_removal_pass &= pd.isna(removal_row.common_component_mean)
            continue
        remaining_occurrences = all_occurrences[
            ~all_occurrences["stock"].astype(str).isin(removed_stocks)
        ].copy()

        def audit_winsor_mean(values: pd.Series) -> float:
            clean = values.dropna().astype(float).to_numpy()
            if not len(clean):
                return math.nan
            lower, upper = np.quantile(clean, [0.10, 0.90])
            return float(np.clip(clean, lower, upper).mean())

        rebuilt = (
            remaining_occurrences.groupby(
                ["period", "session", "pair", "loop", "regime"],
                observed=True,
                dropna=False,
            )["net_payoff_bps"]
            .agg(audit_winsor_mean)
            .rename("robust_net_payoff_bps")
            .reset_index()
        )
        rebuilt["common_component"] = rebuilt.groupby(["period", "session"], observed=True)[
            "robust_net_payoff_bps"
        ].transform("median")
        rebuilt["after_common"] = rebuilt["robust_net_payoff_bps"] - rebuilt["common_component"]
        rebuilt["regime_component"] = rebuilt.groupby(
            ["period", "session", "regime"], observed=True
        )["after_common"].transform("median")
        rebuilt["loop_excess_component"] = (
            rebuilt["robust_net_payoff_bps"]
            - rebuilt["common_component"]
            - rebuilt["regime_component"]
        )
        scope = rebuilt[rebuilt["pair"].isin(member_pairs)]
        removal_masses = (
            scope[["common_component", "regime_component", "loop_excess_component"]]
            .clip(lower=0.0)
            .sum()
        )
        mass_total = float(removal_masses.sum())
        expected_removal_values = {
            "supported_pair_cells": float(len(scope)),
            "robust_net_payoff_sum": float(scope["robust_net_payoff_bps"].sum()),
            "robust_positive_payoff_sum": float(
                scope["robust_net_payoff_bps"].clip(lower=0.0).sum()
            ),
            "common_component_mean": float(scope["common_component"].mean()),
            "regime_component_mean": float(scope["regime_component"].mean()),
            "loop_excess_component_mean": float(scope["loop_excess_component"].mean()),
            "common_positive_component_share": float(removal_masses["common_component"])
            / mass_total
            if mass_total > 0
            else math.nan,
            "regime_positive_component_share": float(removal_masses["regime_component"])
            / mass_total
            if mass_total > 0
            else math.nan,
            "loop_excess_positive_component_share": float(removal_masses["loop_excess_component"])
            / mass_total
            if mass_total > 0
            else math.nan,
        }
        for field, expected in expected_removal_values.items():
            actual = float(cast(Any, getattr(removal_row, field)))
            if math.isfinite(expected):
                stock_removal_errors.append(abs(actual - expected))
        identity_error = (
            scope[["common_component", "regime_component", "loop_excess_component"]].sum(axis=1)
            - scope["robust_net_payoff_bps"]
        ).abs()
        expected_identity_error = float(identity_error.max()) if len(identity_error) else math.nan
        if math.isfinite(expected_identity_error):
            stock_removal_errors.append(
                abs(
                    float(cast(Any, removal_row.component_identity_max_absolute_error))
                    - expected_identity_error
                )
            )
    cohort_pass = bool(
        set(cohort["cohort_dimension"])
        == {"stock", "month", "quarter", "period", "liquidity_cohort"}
        and cohort["positive_payoff_share"].dropna().between(-1e-12, 1 + 1e-12).all()
        and leave_one["row_deletion_only_not_retrained_model"].all()
        and leave_one["positive_payoff_after_removal"].ge(-1e-10).all()
        and not leave_one["model_retrained"].any()
        and set(leave_one["removal_scope"])
        == {"leave_one_stock_out", "remove_best_stock", "remove_top_five_stocks"}
        and stock_removal_pass
        and (not stock_removal_errors or max(stock_removal_errors) <= 1e-10)
    )

    component_persistence = pd.read_csv(primary / "component_persistence_table.csv")
    component_persistence_pass = bool(
        set(component_persistence["component"]) == {"common", "regime", "loop_excess"}
        and set(component_persistence["lag_sessions"]) == {1, 2, 3}
        and not component_persistence["raw_fill_weighting_used"].any()
    )

    factor = pd.read_csv(primary / "common_factor_diagnostic.csv")
    factor_checks: list[float] = []
    for period, group in supported.groupby("period"):
        matrix = group.pivot_table(
            index="session", columns="pair", values="robust_net_payoff_bps", aggfunc="mean"
        )
        centered = matrix - matrix.mean(axis=0)
        _, singular, _ = np.linalg.svd(centered.fillna(0).to_numpy(), full_matrices=False)
        explained = np.square(singular) / np.square(singular).sum()
        stored = factor[factor["period"].astype(str).eq(str(period))]
        for factor_row in stored.itertuples():
            factor_checks.append(
                abs(
                    float(explained[: int(cast(Any, factor_row.factor_count))].sum())
                    - float(cast(Any, factor_row.cumulative_variance_explained))
                )
            )
    factor_pass = bool(factor_checks and max(factor_checks) < 1e-12)
    factor_loadings = pd.read_parquet(primary / "common_factor_loadings.parquet")
    factor_loading_pass = bool(
        set(factor_loadings["period"].astype(str)) == {"2023", "2025"}
        and factor_loadings["fit_population"].eq("same_period_only").all()
        and factor["first_component_common_rank_correlation"].notna().all()
    )

    manifest = json.loads((primary / "artifact_manifest.json").read_text())
    manifest_mismatches = sorted(
        name for name, expected in manifest.items() if sha256(primary / name) != expected
    )
    plot_names = sorted(
        path.relative_to(primary).as_posix() for path in (primary / "plots").glob("*.png")
    )
    plot_hash_pass = len(plot_names) == 16 and all(name in manifest for name in plot_names)
    identity = artifact_identity(primary, rerun)
    identity_pass = not identity["missing"] and not identity["extra"] and not identity["mismatches"]
    trace_columns = {
        "run_id",
        "period",
        "session",
        "episode_id",
        "pair",
        "loop",
        "orientation",
        "regime",
        "component",
        "source_artifact",
        "source_hash",
    }
    trace_failures: list[str] = []
    scoped_trace_artifacts = {
        "same_regime_episode_ledger.parquet",
        "shared_market_episode_ledger.parquet",
        "episode_session_timeline.parquet",
        "early_leader_table.parquet",
        "component_episode_attribution.parquet",
        "stock_and_cohort_concentration.parquet",
        "cohort_contribution_table.parquet",
        "leave_one_stock_out_attribution.parquet",
    }
    for artifact_path in sorted(primary.glob("*.parquet")):
        artifact = pd.read_parquet(artifact_path)
        if (
            not trace_columns.issubset(artifact.columns)
            or artifact[list(trace_columns)].isna().any().any()
        ):
            trace_failures.append(artifact_path.name)
        elif artifact_path.name in scoped_trace_artifacts and any(
            artifact[column].astype(str).str.startswith("all_").any()
            for column in [
                "period",
                "session",
                "episode_id",
                "pair",
                "loop",
                "orientation",
                "regime",
            ]
        ):
            trace_failures.append(f"{artifact_path.name}:unscoped_identity")
    for artifact_path in sorted(primary.glob("*.csv")):
        artifact = pd.read_csv(artifact_path)
        if (
            not trace_columns.issubset(artifact.columns)
            or artifact[list(trace_columns)].isna().any().any()
        ):
            trace_failures.append(artifact_path.name)
    traceability_pass = not trace_failures

    safety = contract["safety"]
    changed = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPO, text=True
    ).splitlines()
    forbidden_paths = (
        "packages/stocker_execution/",
        "apps/",
        "stocker_launcher.py",
    )
    safety_paths_pass = not any(
        line[3:].startswith(forbidden_paths) for line in changed if len(line) > 3
    )
    safety_pass = bool(
        safety["research_only"]
        and not safety["live_ordering_enabled"]
        and not safety["broker_connection_enabled"]
        and not safety["paper_or_demo_execution_enabled"]
        and not safety["deployment_enabled"]
        and safety_paths_pass
    )

    run_metadata = json.loads((primary / "run_metadata.json").read_text())
    frozen_git_head = str(contract["lineage"]["starting_commit"])
    audit_checkout_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    frozen_is_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", frozen_git_head, audit_checkout_head],
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )

    checks = {
        "contract_identity": contract["contract_id"]
        == "20260717-profitable-loop-episode-anatomy-v1",
        "git_identity": run_metadata["git_head"] == frozen_git_head and frozen_is_ancestor,
        "data_snapshot_source_hashes": source_hash_pass,
        "pair_and_orientation_mapping": pair_mapping_pass,
        "positive_pair_labels_and_census": census_pass,
        "eligibility_and_missingness": missingness_pass and episode_membership_pass,
        "episode_construction": (timeline_pass and anatomy_pass and episode_ledger_identity_pass),
        "block_null_generation": null_pass,
        "coactivation_network_statistics": network_pass,
        "common_regime_loop_excess": components_pass,
        "component_persistence": component_persistence_pass,
        "occurrence_payoff_leader_efficiency": bool(
            occurrence_identity_pass
            and efficiency_error <= 1e-12
            and (
                not payoff_occurrence_identity_errors
                or max(payoff_occurrence_identity_errors) <= 1e-12
            )
        ),
        "early_leader_prefixes": early_prefix_pass,
        "leader_persistence": persistence_pass,
        "payoff_remaining": payoff_remaining_pass,
        "regime_sequence_construction": sequence_pass,
        "four_way_counterfactual": four_way_pass,
        "named_pair_deep_dives": named_pass,
        "route_topology_outcome_only": route_pass,
        "sequential_path_deterioration": path_pass,
        "indicator_availability": indicator_pass,
        "concentration_tables": concentration_pass and cohort_pass,
        "principal_component_diagnostic": factor_pass and factor_loading_pass,
        "machine_readable_hashes": not manifest_mismatches,
        "row_traceability": traceability_pass,
        "plot_hashes": plot_hash_pass,
        "primary_exact_rerun_identity": identity_pass,
        "research_only_safety": safety_pass,
    }
    result = {
        "audit_id": "profitable-loop-episode-anatomy-v1-independent-audit",
        "contract_hash": sha256(CONTRACT_PATH),
        "frozen_git_head": frozen_git_head,
        "audit_checkout_head": audit_checkout_head,
        "frozen_git_head_is_ancestor": frozen_is_ancestor,
        "source_hashes": source_hashes,
        "independent_census": census,
        "independent_block_null": null,
        "component_max_absolute_errors": component_errors,
        "episode_component_max_absolute_error": max(component_episode_errors, default=0.0),
        "leader_identity_max_absolute_error": max(leader_identity_errors, default=0.0),
        "occurrence_share_max_absolute_error": max(occurrence_errors, default=0.0),
        "leader_efficiency_max_absolute_error": efficiency_error,
        "payoff_occurrence_identity_max_absolute_error": max(
            payoff_occurrence_identity_errors, default=0.0
        ),
        "stock_capped_occurrence_count": len(independent_occurrence),
        "raw_fill_occurrence_count": int(independent_occurrence["raw_fill_count"].sum()),
        "mixed_history_stock_occurrence_count": int(mixed_history.sum()),
        "sequence_bootstrap_max_absolute_error": max(sequence_bootstrap_errors, default=0.0),
        "early_leader_max_absolute_error": max(early_numeric_errors, default=0.0),
        "early_leader_largest_errors": sorted(
            early_error_details,
            key=lambda item: float(item["absolute_error"]),
            reverse=True,
        )[:10],
        "sequence_max_absolute_error": max(sequence_errors, default=0.0),
        "four_way_max_absolute_error": max(four_way_errors, default=0.0),
        "coactivation_network_max_absolute_error": max(network_errors, default=0.0),
        "named_pair_max_absolute_error": max(named_errors, default=0.0),
        "route_max_absolute_error": max(route_errors, default=0.0),
        "concentration_max_absolute_error": max(concentration_errors, default=0.0),
        "stock_removal_component_max_absolute_error": max(stock_removal_errors, default=0.0),
        "path_deterioration_max_absolute_error": max(path_numeric_errors, default=0.0),
        "sampled_episode_timelines": sampled_timeline_checks,
        "factor_max_absolute_error": max(factor_checks),
        "artifact_manifest_mismatches": manifest_mismatches,
        "traceability_failures": trace_failures,
        "plot_files": plot_names,
        "primary_exact_rerun": identity,
        "changed_paths_reviewed": changed,
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(primary / "independent_audit.json", result)
    if not result["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise AssertionError(f"independent audit failed: {failed}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--exact-rerun", type=Path, default=DEFAULT_RERUN)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    audited = audit(arguments.primary.resolve(), arguments.exact_rerun.resolve())
    print(json.dumps(audited, indent=2, sort_keys=True, default=str))
