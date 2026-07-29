"""Run the frozen, read-only Profitable Loop Episode Anatomy V1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stocker-profitable-loop-anatomy-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
PACKAGE_SOURCE = REPO / "packages/stocker_research/src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from stocker_research.profitable_loop_episode_anatomy import (  # noqa: E402
    attach_episode_membership,
    block_circular_pair_shift,
    build_episode_ledgers,
    build_synchronized_panel,
    classify_episode,
    collapse_stock_contributions,
    common_factor_diagnostic,
    concentration_attribution,
    decode_history_token,
    decompose_payoff_components,
    early_leader_checkpoints,
    exact_rerun_identity,
    frozen_run_git_head,
    poisson_binomial_null,
    recompute_component_summary_after_stock_removal,
    reproduce_exploratory_census,
)

CONTRACT_PATH = WORK / "contracts/20260717-profitable-loop-episode-anatomy-v1.json"
DEFAULT_OUTPUT = WORK / "artifacts/20260717-profitable-loop-episode-anatomy-v1/primary"
RUN_ID = "profitable-loop-episode-anatomy-v1-frozen-run"
RUN_TIMESTAMP = "2026-07-17T00:00:00+00:00"
TRACE_COLUMNS = [
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
]
NAMED_PAIRS = {
    "cycle_04|state_4": "named",
    "cycle_07|state_5": "named",
    "cycle_04|state_2": "control",
    "cycle_07|state_6": "control",
}
INDICATORS = [
    "structural_breadth",
    "breadth_change",
    "top_loop_score",
    "top_second_margin",
    "loop_score_entropy",
    "transition_surprise",
    "positive_stock_fraction",
    "payoff_dispersion",
    "dispersion_change",
    "cost_pressure",
    "market_return",
    "market_volatility",
    "liquidity_pressure",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(safe_json(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def mode_or_unavailable(values: pd.Series) -> object:
    clean = values.dropna()
    if clean.empty:
        return pd.NA
    counts = clean.astype(str).value_counts()
    return sorted(counts[counts.eq(counts.max())].index)[0]


def unique_or_unavailable(values: pd.Series) -> str:
    """Retain context only when every fill in one stock occurrence agrees."""

    clean = sorted(values.dropna().astype(str).unique())
    return clean[0] if len(clean) == 1 else "unavailable"


def unavailable_count(values: pd.Series) -> int:
    strings = values.astype("string")
    return int((strings.isna() | strings.eq("unavailable")).sum())


def entropy(shares: Iterable[float]) -> float:
    values = np.asarray([value for value in shares if value > 0 and math.isfinite(value)])
    return float(-(values * np.log(values)).sum()) if len(values) else math.nan


def finite_median(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.median()) if len(clean) else math.nan


def finite_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else math.nan


def add_trace(
    frame: pd.DataFrame,
    *,
    source_artifact: str,
    source_hash: str,
    component: str = "descriptive",
) -> pd.DataFrame:
    result = frame.copy()
    if "episode_id" not in result:
        membership_columns = [
            column
            for column in [
                "hindsight_pair_episode_id",
                "same_regime_episode_id",
                "shared_session_episode_id",
            ]
            if column in result
        ]
        if membership_columns:
            result["episode_id"] = result[membership_columns].apply(
                lambda row: (
                    "|".join(
                        f"{column}={row[column]}"
                        for column in membership_columns
                        if pd.notna(row[column])
                    )
                    or "unavailable"
                ),
                axis=1,
            )
    defaults: dict[str, Any] = {
        "run_id": RUN_ID,
        "period": "all_periods",
        "session": "all_sessions",
        "episode_id": "all_episodes",
        "pair": "all_pairs",
        "loop": "all_loops",
        "orientation": "all_orientations",
        "regime": "all_regimes",
        "component": component,
        "source_artifact": source_artifact,
        "source_hash": source_hash,
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = default
        else:
            result[column] = result[column].fillna(default)
    ordered = [
        *TRACE_COLUMNS,
        *[column for column in result.columns if column not in TRACE_COLUMNS],
    ]
    return result.loc[:, ordered]


def write_frame(
    output: Path,
    name: str,
    frame: pd.DataFrame,
    *,
    source_artifact: str,
    source_hash: str,
    component: str = "descriptive",
) -> pd.DataFrame:
    traced = add_trace(
        frame,
        source_artifact=source_artifact,
        source_hash=source_hash,
        component=component,
    )
    path = output / name
    if path.suffix == ".parquet":
        traced.to_parquet(path, index=False, compression="zstd")
    elif path.suffix == ".csv":
        traced.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")
    else:
        raise ValueError(f"unsupported table format: {path}")
    return traced


def verify_contract() -> tuple[dict[str, Any], dict[str, Path], dict[str, str], str]:
    contract: dict[str, Any] = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("registered_before_extended_scoring") is not True:
        raise AssertionError("contract is not frozen before extended scoring")
    required_safety = {
        "research_only": True,
        "read_only_inputs": True,
        "live_ordering_enabled": False,
        "order_placement_enabled": False,
        "broker_connection_enabled": False,
        "ig_integration_enabled": False,
        "paper_or_demo_execution_enabled": False,
        "position_management_changed": False,
        "existing_exit_logic_changed": False,
        "deployment_enabled": False,
        "application_runtime_changed": False,
        "api_keys_or_secrets_used": False,
    }
    drift = [
        key for key, expected in required_safety.items() if contract["safety"].get(key) != expected
    ]
    if drift:
        raise AssertionError(f"research-only safety drift: {drift}")
    root = (CONTRACT_PATH.parent / contract["inputs"]["root"]).resolve()
    paths = {name: root / name for name in contract["inputs"]["files"]}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"frozen inputs are missing: {missing}")
    hashes = {name: sha256(path) for name, path in paths.items()}
    mismatches = sorted(
        name for name, expected in contract["inputs"]["files"].items() if hashes[name] != expected
    )
    if mismatches:
        raise AssertionError(f"frozen source hash mismatch: {mismatches}")
    snapshot = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract, paths, hashes, snapshot


def stock_occurrence_ledger(trades: pd.DataFrame) -> pd.DataFrame:
    filled = trades[
        trades["model_name"].eq("no_payoff_state_filter")
        & trades["horizon"].eq(24)
        & trades["status"].eq("filled")
    ].copy()
    filled["session"] = pd.to_datetime(filled["score_session"]).dt.strftime("%Y-%m-%d")
    filled["loop"] = filled["loop_id"]
    filled["regime"] = filled["orientation"]
    filled["stock"] = filled["stock_id"]
    decoded = [decode_history_token(int(token)) for token in filled["history_token"]]
    for column in ["previous_regime", "regime_history_2", "regime_history_3"]:
        filled[column] = [item[column] for item in decoded]
    filled["gross_payoff_bps"] = filled["gross_payoff_bps"].clip(-500.0, 500.0)
    filled["net_payoff_bps"] = filled["primary_net_payoff_bps"].clip(-500.0, 500.0)
    filled["twice_cost_net_payoff_bps"] = filled["gross_payoff_bps"] - 20.0
    # The occurrence identity is frozen at stock/session/loop/orientation.  In
    # particular, history variants may not split one stock into several
    # economic occurrences.  Context that disagrees across repeated fills is
    # retained as explicitly unavailable.
    keys = ["period", "session", "loop", "regime", "stock"]
    result = (
        filled.groupby(keys, observed=True, dropna=False)
        .agg(
            gross_payoff_bps=("gross_payoff_bps", "mean"),
            net_payoff_bps=("net_payoff_bps", "mean"),
            twice_cost_net_payoff_bps=("twice_cost_net_payoff_bps", "mean"),
            raw_fill_count=("fill_id", "size"),
            history_token_count=("history_token", "nunique"),
            history_2_variant_count_within_occurrence=("regime_history_2", "nunique"),
            history_3_variant_count_within_occurrence=("regime_history_3", "nunique"),
            previous_regime=("previous_regime", unique_or_unavailable),
            regime_history_2=("regime_history_2", unique_or_unavailable),
            regime_history_3=("regime_history_3", unique_or_unavailable),
            month=("month", "first"),
            quarter=("quarter", "first"),
            volume_bucket=("volume_bucket", mode_or_unavailable),
            clock_phase=("state_change_phase", unique_or_unavailable),
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
    result["pair"] = result["loop"].astype(str) + "|" + result["regime"].astype(str)
    result["positive_payoff"] = result["net_payoff_bps"].clip(lower=0.0)
    result["positive_occurrence"] = result["net_payoff_bps"].gt(0.0)
    return result


def add_history_to_panel(panel: pd.DataFrame, occurrences: pd.DataFrame) -> pd.DataFrame:
    keys = ["period", "session", "loop", "orientation"]
    history = occurrences.rename(columns={"regime": "orientation"})
    aggregate = (
        history.groupby(keys, observed=True)
        .agg(
            previous_regime=("previous_regime", mode_or_unavailable),
            regime_history_2=("regime_history_2", mode_or_unavailable),
            regime_history_3=("regime_history_3", mode_or_unavailable),
            clock_phase=("clock_phase", mode_or_unavailable),
            history_2_variant_count=("regime_history_2", "nunique"),
            history_2_unavailable_count=("regime_history_2", unavailable_count),
            history_3_variant_count=("regime_history_3", "nunique"),
            history_3_unavailable_count=("regime_history_3", unavailable_count),
            history_mixed_within_occurrence=("history_mixed_within_occurrence", "any"),
            history_2_mixed_within_occurrence=(
                "history_2_mixed_within_occurrence",
                "any",
            ),
            history_3_mixed_within_occurrence=(
                "history_3_mixed_within_occurrence",
                "any",
            ),
            occurrence_count=("stock", "size"),
        )
        .reset_index()
    )
    result = panel.merge(aggregate, on=keys, how="left", validate="one_to_one")
    result["regime_history_4"] = pd.NA
    result["state_age"] = math.nan
    result["completed_prior_dwell"] = math.nan
    result["same_orientation_repeat_count"] = math.nan
    mixed_within_occurrence = (
        result["history_mixed_within_occurrence"].astype("boolean").fillna(False)
    )
    history_2_mixed_within_occurrence = (
        result["history_2_mixed_within_occurrence"].astype("boolean").fillna(False)
    )
    history_3_mixed_within_occurrence = (
        result["history_3_mixed_within_occurrence"].astype("boolean").fillna(False)
    )
    occurrence_count = result["occurrence_count"].fillna(0)
    history_2_variants = result["history_2_variant_count"].fillna(0)
    history_3_variants = result["history_3_variant_count"].fillna(0)
    history_2_unavailable = result["history_2_unavailable_count"].fillna(0)
    history_3_unavailable = result["history_3_unavailable_count"].fillna(0)
    result["history_mixed"] = (
        history_2_variants.gt(1) | history_3_variants.gt(1) | mixed_within_occurrence
    ).astype(bool)
    result["history_variant_count"] = pd.concat(
        [history_2_variants, history_3_variants], axis=1
    ).max(axis=1)
    history_2_fail_closed = (
        occurrence_count.eq(0)
        | history_2_variants.ne(1)
        | history_2_unavailable.gt(0)
        | history_2_mixed_within_occurrence
    )
    history_3_fail_closed = (
        occurrence_count.eq(0)
        | history_3_variants.ne(1)
        | history_3_unavailable.gt(0)
        | history_3_mixed_within_occurrence
    )
    result.loc[
        history_2_fail_closed,
        ["previous_regime", "regime_history_2"],
    ] = "unavailable"
    result.loc[history_3_fail_closed, "regime_history_3"] = "unavailable"
    result["history_available"] = ~history_2_fail_closed
    result["history_3_available"] = ~history_3_fail_closed
    result["state_age_available"] = False
    result["completed_prior_dwell_available"] = False
    result["repeat_count_available"] = False
    return result


def winsor_mean(values: pd.Series) -> float:
    clean = values.dropna().astype(float).sort_values().to_numpy()
    if not len(clean):
        return math.nan
    lower, upper = np.quantile(clean, [0.10, 0.90])
    return float(np.clip(clean, lower, upper).mean())


def component_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    supported = panel[panel["robust_net_payoff_bps"].notna()].copy()
    records: list[pd.DataFrame] = []
    for (_period, _session), group in supported.groupby(["period", "session"], sort=True):
        common = winsor_mean(group["robust_net_payoff_bps"])
        current = group.copy()
        current["common_component_sensitivity"] = common
        current["after_common"] = current["robust_net_payoff_bps"] - common
        regime_values = current.groupby("regime", observed=True)["after_common"].transform(
            winsor_mean
        )
        current["regime_component_sensitivity"] = regime_values
        current["loop_excess_component_sensitivity"] = (
            current["robust_net_payoff_bps"] - common - regime_values
        )
        records.append(current)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def calendar_lookup(
    calendar: pd.DataFrame,
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    frame = calendar.copy()
    frame["period"] = frame["period"].astype(str)
    frame["score_session"] = pd.to_datetime(frame["score_session"]).dt.strftime("%Y-%m-%d")
    frame = frame.drop_duplicates().sort_values(["period", "score_session"])
    lists: dict[str, list[str]] = {}
    positions: dict[str, dict[str, int]] = {}
    for period_value, group in frame.groupby("period", sort=True):
        period = str(period_value)
        sessions = [str(value) for value in group["score_session"].tolist()]
        lists[period] = sessions
        positions[period] = {session: index for index, session in enumerate(sessions)}
    return lists, positions


def episode_members(pair_ledger: pd.DataFrame, episode: pd.Series) -> pd.DataFrame:
    ids = str(episode["source_pair_episode_ids"]).split("|")
    return pair_ledger[pair_ledger["pair_episode_id"].astype(str).isin(ids)].copy()


def positive_component_timing(
    component_rows: pd.DataFrame,
    column: str,
    session_positions: Mapping[str, int],
) -> tuple[float, float]:
    by_session = component_rows.groupby("session", observed=True)[column].median()
    positive_sessions = [
        session_positions[str(session)]
        for session, value in by_session.items()
        if value > 0 and str(session) in session_positions
    ]
    if not positive_sessions:
        return math.nan, math.nan
    return float(min(positive_sessions)), float(max(positive_sessions))


def episode_metrics(
    ledger: pd.DataFrame,
    pair_ledger: pd.DataFrame,
    panel: pd.DataFrame,
    occurrences: pd.DataFrame,
    calendars: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episode_records: list[dict[str, Any]] = []
    loop_records: list[dict[str, Any]] = []
    session_loop_records: list[dict[str, Any]] = []
    for _, episode in ledger.iterrows():
        members = episode_members(pair_ledger, episode)
        member_pair_values = sorted(set(members["pair"].astype(str)))
        member_loop_values = sorted(set(members["loop"].astype(str)))
        member_regime_values = sorted(set(members["regime"].astype(str)))
        member_pairs = set(member_pair_values)
        period = str(episode["period"])
        sessions = calendars[period]
        start = sessions.index(str(episode["onset"]))
        end = sessions.index(str(episode["end"]))
        episode_sessions = sessions[start : end + 1]
        pre_sessions = sessions[max(0, start - 10) : start]
        pair_totals = members.groupby("loop", observed=True)["total_episode_payoff_bps"].sum()
        positive_totals = pair_totals[pair_totals.gt(0.0)].sort_values(ascending=False)
        regime_positive_totals = (
            members.groupby("regime", observed=True)["total_episode_payoff_bps"]
            .sum()
            .clip(lower=0.0)
        )
        regime_leader_value = (
            float(regime_positive_totals.max()) if len(regime_positive_totals) else math.nan
        )
        regime_leaders = (
            sorted(
                regime_positive_totals[regime_positive_totals.eq(regime_leader_value)].index.astype(
                    str
                )
            )
            if math.isfinite(regime_leader_value) and regime_leader_value > 0
            else []
        )
        total_positive = float(positive_totals.sum())
        leader_value = float(positive_totals.max()) if len(positive_totals) else math.nan
        leader_loops = (
            sorted(positive_totals[positive_totals.eq(leader_value)].index.astype(str))
            if len(positive_totals)
            else []
        )
        leader_share = leader_value / total_positive if total_positive > 0 else math.nan
        second_value = (
            float(positive_totals.iloc[len(leader_loops)])
            if len(positive_totals) > len(leader_loops)
            else math.nan
        )
        second_loops = (
            sorted(positive_totals[positive_totals.eq(second_value)].index.astype(str))
            if math.isfinite(second_value)
            else []
        )
        episode_occ = occurrences[
            occurrences["period"].astype(str).eq(period)
            & occurrences["session"].isin(episode_sessions)
            & occurrences["pair"].isin(member_pairs)
        ]
        pre_occ = occurrences[
            occurrences["period"].astype(str).eq(period)
            & occurrences["session"].isin(pre_sessions)
            & occurrences["pair"].isin(member_pairs)
        ]
        pre_component_rows = panel[
            panel["period"].astype(str).eq(period)
            & panel["session"].isin(pre_sessions)
            & panel["pair"].isin(member_pairs)
            & panel["robust_net_payoff_bps"].notna()
        ]
        pre_payoff_by_loop = pre_component_rows.groupby("loop", observed=True)[
            "robust_net_payoff_bps"
        ].sum()
        pre_supported_cells = set(
            zip(
                pre_component_rows["session"].astype(str),
                pre_component_rows["pair"].astype(str),
                strict=False,
            )
        )
        pre_available_cells = set(
            zip(
                pre_occ["session"].astype(str),
                pre_occ["pair"].astype(str),
                strict=False,
            )
        )
        pre_occurrence_population_complete = bool(
            pre_supported_cells and pre_supported_cells.issubset(pre_available_cells)
        )
        occurrence_by_loop = episode_occ.groupby("loop", observed=True).size()
        positive_occurrence_by_loop = episode_occ.groupby("loop", observed=True)[
            "positive_occurrence"
        ].sum()
        occurrence_total = int(occurrence_by_loop.sum())
        pre_occurrence_by_loop = pre_occ.groupby("loop", observed=True).size()
        pre_total = int(pre_occurrence_by_loop.sum())
        stock_positive = episode_occ.groupby("stock", observed=True)["positive_payoff"].sum()
        stock_total = float(stock_positive.sum())
        stock_top = float(stock_positive.max() / stock_total) if stock_total > 0 else math.nan

        component_rows = panel[
            panel["period"].astype(str).eq(period)
            & panel["session"].isin(episode_sessions)
            & panel["pair"].isin(member_pairs)
            & panel["robust_net_payoff_bps"].notna()
        ]
        supported_occurrence_cells = set(
            zip(
                component_rows["session"].astype(str),
                component_rows["pair"].astype(str),
                strict=False,
            )
        )
        available_occurrence_cells = set(
            zip(
                episode_occ["session"].astype(str),
                episode_occ["pair"].astype(str),
                strict=False,
            )
        )
        occurrence_coverage_fraction = (
            len(supported_occurrence_cells & available_occurrence_cells)
            / len(supported_occurrence_cells)
            if supported_occurrence_cells
            else math.nan
        )
        occurrence_population_complete = bool(
            supported_occurrence_cells
            and supported_occurrence_cells.issubset(available_occurrence_cells)
        )
        common_by_session = component_rows.groupby("session", observed=True)[
            "common_component"
        ].first()
        regime_by_session = component_rows.groupby(["session", "regime"], observed=True)[
            "regime_component"
        ].first()
        component_columns = {
            "common": "common_component",
            "regime": "regime_component",
            "loop_excess": "loop_excess_component",
        }
        # Every component is expanded over the same supported pair rows before
        # aggregation.  This preserves the per-pair additive identity and avoids
        # mixing one-common-row, one-regime-row, and one-loop-row denominators.
        positive_component_totals = {
            name: float(component_rows[column].clip(lower=0.0).sum())
            for name, column in component_columns.items()
        }
        component_denominator = sum(positive_component_totals.values())
        observed_positive_total = float(
            component_rows["robust_net_payoff_bps"].clip(lower=0.0).sum()
        )
        positive_pair_component_totals = {
            name: float(
                component_rows.loc[component_rows["robust_net_payoff_bps"].gt(0), column].sum()
            )
            for name, column in component_columns.items()
        }
        component_variances = {
            name: float(component_rows[column].var(ddof=0))
            for name, column in component_columns.items()
        }
        marginal_variance_total = sum(
            value for value in component_variances.values() if math.isfinite(value)
        )
        session_positions = {session: index for index, session in enumerate(episode_sessions)}

        common_first, common_last = positive_component_timing(
            component_rows, "common_component", session_positions
        )
        regime_first, regime_last = positive_component_timing(
            component_rows, "regime_component", session_positions
        )
        loop_first, loop_last = positive_component_timing(
            component_rows, "loop_excess_component", session_positions
        )
        common_signed_share = (
            positive_pair_component_totals["common"] / observed_positive_total
            if observed_positive_total > 0
            else math.nan
        )
        regime_signed_share = (
            positive_pair_component_totals["regime"] / observed_positive_total
            if observed_positive_total > 0
            else math.nan
        )
        loop_signed_share = (
            positive_pair_component_totals["loop_excess"] / observed_positive_total
            if observed_positive_total > 0
            else math.nan
        )
        common_positive_all = bool(len(common_by_session) and common_by_session.gt(0).all())
        positive_regimes = int(
            members.groupby("regime", observed=True)["total_episode_payoff_bps"].sum().gt(0).sum()
        )
        anatomy = classify_episode(int(episode["loop_count"]), leader_share)
        leader_component_rows = component_rows[
            component_rows["loop"].astype(str).isin(leader_loops)
        ]
        peer_component_rows = component_rows[~component_rows["loop"].astype(str).isin(leader_loops)]
        leader_supported_positive = float(
            leader_component_rows["robust_net_payoff_bps"].clip(lower=0.0).sum()
        )
        leader_positive_loop_excess = float(
            leader_component_rows["loop_excess_component"].clip(lower=0.0).sum()
        )
        peer_positive_fraction = (
            float(peer_component_rows["robust_net_payoff_bps"].gt(0).mean())
            if len(peer_component_rows)
            else math.nan
        )
        episode_records.append(
            {
                **{str(key): value for key, value in episode.to_dict().items()},
                "session": f"{episode['onset']}..{episode['end']}",
                "pair": ";".join(member_pair_values),
                "loop": ";".join(member_loop_values),
                "orientation": ";".join(member_regime_values),
                "regime": ";".join(member_regime_values),
                "duration_sessions": end - start + 1,
                "final_dominant_loop": "|".join(leader_loops) if leader_loops else "unavailable",
                "leader_tie": len(leader_loops) > 1,
                "final_dominant_regime": "|".join(regime_leaders)
                if regime_leaders
                else "unavailable",
                "dominant_regime_tie": len(regime_leaders) > 1,
                "second_place_loop": "|".join(second_loops) if second_loops else "unavailable",
                "final_leader_share": leader_share,
                "leader_margin": leader_value - second_value
                if math.isfinite(leader_value) and math.isfinite(second_value)
                else math.nan,
                "loop_payoff_entropy": entropy((positive_totals / total_positive).tolist())
                if total_positive > 0
                else math.nan,
                "loop_occurrence_entropy": entropy((occurrence_by_loop / occurrence_total).tolist())
                if occurrence_total > 0 and occurrence_population_complete
                else math.nan,
                "regime_payoff_entropy": entropy(
                    (
                        members.groupby("regime", observed=True)["total_episode_payoff_bps"]
                        .sum()
                        .clip(lower=0.0)
                        / max(
                            float(
                                members.groupby("regime", observed=True)["total_episode_payoff_bps"]
                                .sum()
                                .clip(lower=0.0)
                                .sum()
                            ),
                            1e-12,
                        )
                    ).tolist()
                ),
                "regime_occurrence_entropy": entropy(
                    (episode_occ.groupby("regime").size() / max(len(episode_occ), 1)).tolist()
                )
                if occurrence_population_complete
                else math.nan,
                "occurrence_coverage_fraction": occurrence_coverage_fraction,
                "occurrence_population_complete": occurrence_population_complete,
                "anatomy_category": anatomy,
                "common_component_mean": float(common_by_session.mean()),
                "regime_component_mean": float(regime_by_session.mean()),
                "loop_excess_component_mean": float(component_rows["loop_excess_component"].mean()),
                "common_component_positive_pair_share": float(
                    component_rows["common_component"].gt(0).mean()
                ),
                "regime_component_positive_pair_share": float(
                    component_rows["regime_component"].gt(0).mean()
                ),
                "loop_excess_component_positive_pair_share": float(
                    component_rows["loop_excess_component"].gt(0).mean()
                ),
                "common_positive_contribution_share": positive_component_totals["common"]
                / component_denominator
                if component_denominator > 0
                else math.nan,
                "regime_positive_contribution_share": positive_component_totals["regime"]
                / component_denominator
                if component_denominator > 0
                else math.nan,
                "loop_excess_positive_contribution_share": positive_component_totals["loop_excess"]
                / component_denominator
                if component_denominator > 0
                else math.nan,
                "common_signed_contribution_to_positive_pair_payoff": common_signed_share,
                "regime_signed_contribution_to_positive_pair_payoff": regime_signed_share,
                "loop_excess_signed_contribution_to_positive_pair_payoff": loop_signed_share,
                "common_marginal_variance_share_noncausal": component_variances["common"]
                / marginal_variance_total
                if marginal_variance_total > 0
                else math.nan,
                "regime_marginal_variance_share_noncausal": component_variances["regime"]
                / marginal_variance_total
                if marginal_variance_total > 0
                else math.nan,
                "loop_excess_marginal_variance_share_noncausal": component_variances["loop_excess"]
                / marginal_variance_total
                if marginal_variance_total > 0
                else math.nan,
                "variance_decomposition_is_causal": False,
                "common_first_positive_relative_session": common_first,
                "common_last_positive_relative_session": common_last,
                "regime_first_positive_relative_session": regime_first,
                "regime_last_positive_relative_session": regime_last,
                "loop_excess_first_positive_relative_session": loop_first,
                "loop_excess_last_positive_relative_session": loop_last,
                "common_onset_value": finite_median(
                    component_rows.loc[
                        component_rows["session"].eq(episode_sessions[0]), "common_component"
                    ]
                ),
                "common_decay_value": finite_median(
                    component_rows.loc[
                        component_rows["session"].eq(episode_sessions[-1]), "common_component"
                    ]
                ),
                "regime_onset_value": finite_median(
                    component_rows.loc[
                        component_rows["session"].eq(episode_sessions[0]), "regime_component"
                    ]
                ),
                "regime_decay_value": finite_median(
                    component_rows.loc[
                        component_rows["session"].eq(episode_sessions[-1]), "regime_component"
                    ]
                ),
                "loop_excess_onset_value": finite_median(
                    component_rows.loc[
                        component_rows["session"].eq(episode_sessions[0]),
                        "loop_excess_component",
                    ]
                ),
                "loop_excess_decay_value": finite_median(
                    component_rows.loc[
                        component_rows["session"].eq(episode_sessions[-1]),
                        "loop_excess_component",
                    ]
                ),
                "common_component_persistence": float(common_by_session.autocorr(lag=1))
                if len(common_by_session) > 2
                else math.nan,
                "stock_top_one_positive_contribution_share": stock_top,
                "stock_or_cohort_concentrated": bool(math.isfinite(stock_top) and stock_top > 0.30),
                "shared_regime_activation": bool(
                    int(episode["loop_count"]) >= 2
                    and positive_component_totals["regime"] > 0.5 * component_denominator
                ),
                "multi_regime_shared_episode": bool(positive_regimes >= 2 and common_positive_all),
                "loop_specific_activation": bool(
                    leader_supported_positive > 0
                    and leader_positive_loop_excess > 0.5 * leader_supported_positive
                    and (not math.isfinite(peer_positive_fraction) or peer_positive_fraction < 0.5)
                ),
                "leader_positive_loop_excess_share": leader_positive_loop_excess
                / leader_supported_positive
                if leader_supported_positive > 0
                else math.nan,
                "same_regime_peer_positive_fraction": peer_positive_fraction,
                "regime_sequence_interaction_candidate": False,
            }
        )
        all_loops = sorted(set(members["loop"].astype(str)))
        for loop in all_loops:
            net_payoff = float(pair_totals.get(loop, 0.0))
            payoff = float(positive_totals.get(loop, 0.0))
            occurrence = int(occurrence_by_loop.get(loop, 0))
            occurrence_share = (
                occurrence / occurrence_total
                if occurrence_total and occurrence_population_complete
                else math.nan
            )
            payoff_share = payoff / total_positive if total_positive > 0 else math.nan
            baseline_share = (
                float(pre_occurrence_by_loop.get(loop, 0)) / pre_total
                if pre_total and pre_occurrence_population_complete
                else math.nan
            )
            loop_occ = episode_occ[episode_occ["loop"].astype(str).eq(loop)]
            pre_loop = pre_occ[pre_occ["loop"].astype(str).eq(loop)]
            loop_regimes = sorted(
                members.loc[members["loop"].astype(str).eq(loop), "regime"].astype(str).unique()
            )
            loop_pairs = sorted(
                members.loc[members["loop"].astype(str).eq(loop), "pair"].astype(str).unique()
            )
            mean_payoff_per_occurrence = (
                net_payoff / occurrence
                if occurrence > 0 and occurrence_population_complete
                else math.nan
            )
            pre_occurrence = int(pre_occurrence_by_loop.get(loop, 0))
            pre_payoff_per_occurrence = (
                float(pre_payoff_by_loop.get(loop, 0.0)) / pre_occurrence
                if pre_occurrence > 0 and pre_occurrence_population_complete
                else math.nan
            )
            loop_records.append(
                {
                    "period": period,
                    "episode_id": episode["episode_id"],
                    "episode_level": episode["episode_level"],
                    "session": f"{episode['onset']}..{episode['end']}",
                    "pair": ";".join(loop_pairs),
                    "loop": loop,
                    "orientation": ";".join(loop_regimes),
                    "regime": ";".join(loop_regimes),
                    "occurrence_count": occurrence,
                    "occurrence_share": occurrence_share,
                    "positive_occurrence_count": int(positive_occurrence_by_loop.get(loop, 0)),
                    "positive_occurrence_share": int(positive_occurrence_by_loop.get(loop, 0))
                    / max(int(positive_occurrence_by_loop.sum()), 1)
                    if occurrence_population_complete
                    else math.nan,
                    "total_loop_payoff": net_payoff,
                    "total_positive_payoff": payoff,
                    "positive_payoff_share": payoff_share,
                    "mean_payoff_per_occurrence": mean_payoff_per_occurrence,
                    "median_payoff_per_occurrence": float(loop_occ["net_payoff_bps"].median()),
                    "raw_stock_mean_payoff_per_occurrence": float(
                        loop_occ["net_payoff_bps"].mean()
                    ),
                    "payoff_occurrence_identity_error": net_payoff
                    - occurrence * mean_payoff_per_occurrence
                    if occurrence > 0 and occurrence_population_complete
                    else math.nan,
                    "independent_stock_share": loop_occ["stock"].nunique()
                    / max(episode_occ["stock"].nunique(), 1),
                    "occurrence_coverage_fraction": occurrence_coverage_fraction,
                    "occurrence_population_complete": occurrence_population_complete,
                    "pre_episode_occurrence_population_complete": (
                        pre_occurrence_population_complete
                    ),
                    "normal_occurrence_share_outside_episode": baseline_share,
                    "occurrence_share_lift": occurrence_share - baseline_share
                    if math.isfinite(occurrence_share) and math.isfinite(baseline_share)
                    else math.nan,
                    "payoff_per_occurrence_lift": mean_payoff_per_occurrence
                    - pre_payoff_per_occurrence
                    if math.isfinite(mean_payoff_per_occurrence)
                    and math.isfinite(pre_payoff_per_occurrence)
                    else math.nan,
                    "raw_stock_payoff_per_occurrence_lift": float(loop_occ["net_payoff_bps"].mean())
                    - float(pre_loop["net_payoff_bps"].mean())
                    if len(loop_occ) and len(pre_loop)
                    else math.nan,
                    "leader_efficiency": payoff_share / occurrence_share
                    if occurrence_population_complete
                    and occurrence_share > 0
                    and math.isfinite(payoff_share)
                    else math.nan,
                    "is_final_leader": loop in leader_loops,
                }
            )
        episode_panel = panel[
            panel["period"].astype(str).eq(period)
            & panel["session"].isin(episode_sessions)
            & panel["pair"].isin(member_pairs)
        ]
        for session_key in episode_sessions:
            for loop_key in all_loops:
                raw_group = episode_panel[
                    episode_panel["session"].eq(session_key)
                    & episode_panel["loop"].astype(str).eq(loop_key)
                ]
                group = raw_group[raw_group["robust_net_payoff_bps"].notna()]
                matching_occ = episode_occ[
                    episode_occ["session"].eq(session_key) & episode_occ["loop"].eq(loop_key)
                ]
                session_loop_records.append(
                    {
                        "period": period,
                        "episode_id": episode["episode_id"],
                        "episode_level": episode["episode_level"],
                        "session": session_key,
                        "pair": ";".join(sorted(set(raw_group["pair"].astype(str))))
                        if len(raw_group)
                        else ";".join(
                            sorted(
                                set(
                                    members.loc[
                                        members["loop"].astype(str).eq(loop_key), "pair"
                                    ].astype(str)
                                )
                            )
                        ),
                        "loop": loop_key,
                        "orientation": ";".join(
                            sorted(
                                set(
                                    members.loc[
                                        members["loop"].astype(str).eq(loop_key), "regime"
                                    ].astype(str)
                                )
                            )
                        ),
                        "regime": ";".join(
                            sorted(
                                set(
                                    members.loc[
                                        members["loop"].astype(str).eq(loop_key), "regime"
                                    ].astype(str)
                                )
                            )
                        ),
                        "positive_payoff": float(
                            group["robust_net_payoff_bps"].clip(lower=0.0).sum()
                        )
                        if len(group)
                        else math.nan,
                        "occurrence_count": int(len(matching_occ)),
                        "final_positive_payoff": float(positive_totals.get(loop_key, 0.0)),
                        "support_available": bool(len(group)),
                        "strict_positive_pair_count": int(
                            (
                                raw_group["positive_pair_flag"]
                                & raw_group["positive_pair_available"]
                            ).sum()
                        ),
                    }
                )
    return (
        pd.DataFrame.from_records(episode_records),
        pd.DataFrame.from_records(loop_records),
        pd.DataFrame.from_records(session_loop_records),
    )


def episode_phase(relative: int, duration: int) -> str:
    if relative < 0:
        return "pre"
    if relative >= duration:
        return "post"
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


def episode_timelines(
    episodes: pd.DataFrame,
    pair_ledger: pd.DataFrame,
    panel: pd.DataFrame,
    episode_loops: pd.DataFrame,
    occurrences: pd.DataFrame,
    calendars: dict[str, list[str]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    anatomy_map = episodes.set_index("episode_id")["anatomy_category"].to_dict()
    for _, episode in episodes.iterrows():
        members = episode_members(pair_ledger, episode)
        member_pair_values = sorted(set(members["pair"].astype(str)))
        member_loop_values = sorted(set(members["loop"].astype(str)))
        member_regime_values = sorted(set(members["regime"].astype(str)))
        member_pairs = set(member_pair_values)
        period = str(episode["period"])
        sessions = calendars[period]
        start = sessions.index(str(episode["onset"]))
        end = sessions.index(str(episode["end"]))
        duration = end - start + 1
        for index in range(max(0, start - 10), min(len(sessions), end + 11)):
            session = sessions[index]
            relative = index - start
            current = panel[
                panel["period"].astype(str).eq(period)
                & panel["session"].eq(session)
                & panel["pair"].isin(member_pairs)
            ]
            supported = current[current["robust_net_payoff_bps"].notna()]
            positive = current[current["positive_pair_flag"] & current["positive_pair_available"]]
            loop_payoff = (
                supported.assign(positive_payoff=supported["robust_net_payoff_bps"].clip(lower=0.0))
                .groupby("loop", observed=True)["positive_payoff"]
                .sum()
                .sort_values(ascending=False)
            )
            leader_value = float(loop_payoff.max()) if len(loop_payoff) else math.nan
            leaders = (
                sorted(loop_payoff[loop_payoff.eq(leader_value)].index.astype(str))
                if math.isfinite(leader_value) and leader_value > 0
                else []
            )
            leader = "|".join(leaders) if leaders else "unavailable"
            leader_payoff_share = (
                float(leader_value / loop_payoff.sum()) if loop_payoff.sum() > 0 else math.nan
            )
            current_occ = occurrences[
                occurrences["period"].astype(str).eq(period)
                & occurrences["session"].eq(session)
                & occurrences["pair"].isin(member_pairs)
            ]
            loop_occ = current_occ.groupby("loop", observed=True).size()
            leader_occ_share = (
                float(loop_occ.reindex(leaders, fill_value=0).sum() / loop_occ.sum())
                if loop_occ.sum() > 0 and leaders
                else math.nan
            )
            lower_payoffs = loop_payoff[loop_payoff.lt(leader_value)] if leaders else loop_payoff
            second_value = float(lower_payoffs.max()) if len(lower_payoffs) else math.nan
            second_loops = (
                sorted(lower_payoffs[lower_payoffs.eq(second_value)].index.astype(str))
                if math.isfinite(second_value)
                else []
            )
            second = "|".join(second_loops) if second_loops else "unavailable"
            regime_positive = positive.groupby("regime", observed=True).size()
            regime_payoff = (
                supported.assign(positive_payoff=supported["robust_net_payoff_bps"].clip(lower=0.0))
                .groupby("regime", observed=True)["positive_payoff"]
                .sum()
            )
            regime_leader_value = float(regime_payoff.max()) if len(regime_payoff) else math.nan
            regime_leaders = (
                sorted(regime_payoff[regime_payoff.eq(regime_leader_value)].index.astype(str))
                if math.isfinite(regime_leader_value) and regime_leader_value > 0
                else []
            )
            regime_components = supported.groupby("regime", observed=True)[
                "regime_component"
            ].first()
            loop_components = supported.groupby("loop", observed=True)[
                "loop_excess_component"
            ].max()
            history = supported["regime_history_3"].dropna()
            records.append(
                {
                    "period": period,
                    "episode_id": episode["episode_id"],
                    "episode_level": episode["episode_level"],
                    "pair": ";".join(member_pair_values),
                    "loop": ";".join(member_loop_values),
                    "orientation": ";".join(member_regime_values),
                    "regime": ";".join(member_regime_values),
                    "relative_session_to_onset": relative,
                    "session": session,
                    "calendar_session": session,
                    "episode_phase": episode_phase(relative, duration),
                    "anatomy_category": anatomy_map[episode["episode_id"]],
                    "number_of_eligible_pairs": int(current["positive_pair_available"].sum()),
                    "number_of_positive_pairs": int(len(positive)),
                    "number_of_positive_regimes": int(len(regime_positive)),
                    "common_component": float(supported["common_component"].iloc[0])
                    if len(supported)
                    else math.nan,
                    "dominant_regime_component": float(regime_components.max())
                    if len(regime_components)
                    else math.nan,
                    "dominant_loop_specific_excess": float(loop_components.max())
                    if len(loop_components)
                    else math.nan,
                    "leading_loop": leader,
                    "leading_loop_tie": len(leaders) > 1,
                    "leading_regime": "|".join(regime_leaders) if regime_leaders else "unavailable",
                    "leading_regime_tie": len(regime_leaders) > 1,
                    "leader_occurrence_share": leader_occ_share,
                    "leader_positive_payoff_share": leader_payoff_share,
                    "leader_efficiency": leader_payoff_share / leader_occ_share
                    if leader_occ_share > 0 and math.isfinite(leader_payoff_share)
                    else math.nan,
                    "second_place_loop": second,
                    "leader_margin": float(leader_value - second_value)
                    if math.isfinite(leader_value) and math.isfinite(second_value)
                    else math.nan,
                    "loop_payoff_entropy": entropy((loop_payoff / loop_payoff.sum()).tolist())
                    if loop_payoff.sum() > 0
                    else math.nan,
                    "loop_occurrence_entropy": entropy((loop_occ / loop_occ.sum()).tolist())
                    if loop_occ.sum() > 0
                    else math.nan,
                    "regime_payoff_entropy": entropy((regime_payoff / regime_payoff.sum()).tolist())
                    if regime_payoff.sum() > 0
                    else math.nan,
                    "regime_occurrence_entropy": entropy(
                        (
                            current_occ.groupby("regime", observed=True).size()
                            / max(len(current_occ), 1)
                        ).tolist()
                    ),
                    "dominant_prior_regime_sequence": mode_or_unavailable(history),
                    "state_history_summaries": "|".join(sorted(history.astype(str).unique())),
                    "recurrence_phase_summaries": "unavailable",
                    "range_summary": "unavailable_on_matched_panel",
                    "volatility_summary": finite_median(supported["market_volatility"])
                    if "market_volatility" in supported and len(supported)
                    else math.nan,
                    "vwap_summary": "unavailable_on_matched_panel",
                    "opening_range_summary": "unavailable_on_matched_panel",
                    "relative_strength_summary": "unavailable_on_matched_panel",
                    "independent_stock_support": int(current_occ["stock"].nunique()),
                    "window_left_truncated": start < 10,
                    "window_right_truncated": end + 10 >= len(sessions),
                    "missingness_flags": (
                        "state_age|prior_dwell|repeat_count|vwap|opening_range|relative_strength"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def leader_persistence(session_loops: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for episode_id, episode in session_loops.groupby("episode_id", sort=True):
        sessions = sorted(episode["session"].unique())
        rank_tables: dict[str, pd.Series] = {}
        leader_sets: dict[str, set[str]] = {}
        top_three_sets: dict[str, set[str]] = {}
        shares: dict[str, float] = {}
        support: dict[str, bool] = {}
        strict_positive: dict[str, bool] = {}
        for session in sessions:
            current = episode[episode["session"].eq(session)].copy()
            support[session] = bool(current["support_available"].any())
            strict_positive[session] = bool(current["strict_positive_pair_count"].sum() > 0)
            payoff = current.set_index("loop")["positive_payoff"].dropna().astype(float)
            total = float(payoff.sum())
            if support[session] and strict_positive[session] and total > 0:
                ranks = payoff.rank(method="average", ascending=False)
                leaders = set(payoff[payoff.eq(payoff.max())].index.astype(str))
                session_top_three = set(ranks[ranks.le(3)].index.astype(str))
                rank_tables[session] = ranks
                leader_sets[session] = leaders
                top_three_sets[session] = session_top_three
                shares[session] = float(payoff.max() / total)
            else:
                rank_tables[session] = pd.Series(dtype=float)
                leader_sets[session] = set()
                top_three_sets[session] = set()
                shares[session] = math.nan
        for index, session in enumerate(sessions):
            for lag in [1, 2, 3]:
                if index + lag >= len(sessions):
                    records.append(
                        {
                            "episode_id": episode_id,
                            "session": session,
                            "lag": lag,
                            "status": "episode_boundary",
                            "current_leaders": "|".join(sorted(leader_sets[session])),
                            "future_leaders": "unavailable",
                            "current_leader_tie": len(leader_sets[session]) > 1,
                            "future_leader_tie": pd.NA,
                            "top_one_persistence": math.nan,
                            "top_three_persistence": math.nan,
                            "rank_correlation": math.nan,
                            "leader_share_persistence": math.nan,
                        }
                    )
                    continue
                future_session = sessions[index + lag]
                current_leaders = leader_sets[session]
                future_leaders = leader_sets[future_session]
                if not support[session] or not support[future_session]:
                    status = "missing_support"
                    top_one = math.nan
                    top_three_persists: float | bool = math.nan
                    correlation = math.nan
                elif not strict_positive[session] or not current_leaders:
                    status = "no_positive_pair_current_session"
                    top_one = math.nan
                    top_three_persists = math.nan
                    correlation = math.nan
                elif not strict_positive[future_session] or not future_leaders:
                    status = "no_positive_pair_next_session"
                    top_one = math.nan
                    top_three_persists = math.nan
                    correlation = math.nan
                else:
                    status = "same_shared_episode"
                    top_one = bool(current_leaders & future_leaders)
                    top_three_persists = bool(current_leaders & top_three_sets[future_session])
                    current_rank = rank_tables[session]
                    future_rank = rank_tables[future_session]
                    common = sorted(set(current_rank.index) & set(future_rank.index))
                    if (
                        len(common) >= 2
                        and current_rank.loc[common].nunique() > 1
                        and future_rank.loc[common].nunique() > 1
                    ):
                        correlation = float(
                            stats.spearmanr(
                                current_rank.loc[common], future_rank.loc[common]
                            ).statistic
                        )
                    else:
                        correlation = math.nan
                records.append(
                    {
                        "episode_id": episode_id,
                        "session": session,
                        "lag": lag,
                        "status": status,
                        "current_leaders": "|".join(sorted(current_leaders))
                        if current_leaders
                        else "unavailable",
                        "future_leaders": "|".join(sorted(future_leaders))
                        if future_leaders
                        else "unavailable",
                        "current_leader_tie": len(current_leaders) > 1,
                        "future_leader_tie": len(future_leaders) > 1,
                        "top_one_persistence": top_one,
                        "top_three_persistence": top_three_persists,
                        "rank_correlation": correlation,
                        "leader_share_persistence": shares[future_session] - shares[session]
                        if math.isfinite(shares[future_session]) and math.isfinite(shares[session])
                        else math.nan,
                    }
                )
    return pd.DataFrame.from_records(records)


def component_persistence(panel: pd.DataFrame) -> pd.DataFrame:
    """Describe lag persistence separately for common, regime, and loop excess."""

    common_grid = panel[["period", "session"]].drop_duplicates()
    common_values = (
        panel[panel["common_component"].notna()]
        .groupby(["period", "session"], observed=True)["common_component"]
        .first()
        .reset_index()
    )
    common = common_grid.merge(
        common_values, on=["period", "session"], how="left", validate="one_to_one"
    ).rename(columns={"common_component": "value"})
    common["component_name"] = "common"
    common["entity"] = "all_pairs"
    regime_grid = panel[["period", "session", "regime"]].drop_duplicates()
    regime_values = (
        panel[panel["regime_component"].notna()]
        .groupby(["period", "session", "regime"], observed=True)["regime_component"]
        .first()
        .reset_index()
    )
    regime = regime_grid.merge(
        regime_values,
        on=["period", "session", "regime"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"regime_component": "value", "regime": "entity"})
    regime["component_name"] = "regime"
    loop = panel[["period", "session", "pair", "loop_excess_component"]].rename(
        columns={"loop_excess_component": "value", "pair": "entity"}
    )
    loop["component_name"] = "loop_excess"
    combined = pd.concat([common, regime, loop], ignore_index=True)
    records: list[dict[str, Any]] = []
    for (period, component_name), component_frame in combined.groupby(
        ["period", "component_name"], sort=True
    ):
        for lag in [1, 2, 3]:
            pairs: list[pd.DataFrame] = []
            for _entity, entity in component_frame.groupby("entity", sort=True):
                ordered = entity.sort_values("session")
                paired = pd.DataFrame(
                    {"current": ordered["value"], "future": ordered["value"].shift(-lag)}
                ).dropna()
                pairs.append(paired)
            values = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
            correlation = (
                float(stats.spearmanr(values["current"], values["future"]).statistic)
                if len(values) >= 3
                and values["current"].nunique() > 1
                and values["future"].nunique() > 1
                else math.nan
            )
            records.append(
                {
                    "period": str(period),
                    "component": component_name,
                    "lag_sessions": lag,
                    "paired_rows": len(values),
                    "rank_correlation": correlation,
                    "median_current": float(values["current"].median())
                    if len(values)
                    else math.nan,
                    "median_future": float(values["future"].median()) if len(values) else math.nan,
                    "raw_fill_weighting_used": False,
                }
            )
    return pd.DataFrame.from_records(records)


def support_summary(group: pd.DataFrame) -> dict[str, Any]:
    stock_counts = (
        group["stock"].value_counts(normalize=True) if len(group) else pd.Series(dtype=float)
    )
    return {
        "rows": int(len(group)),
        "independent_sessions": int(group["session"].nunique()),
        "independent_stocks": int(group["stock"].nunique()),
        "months": int(group["month"].nunique()),
        "max_stock_row_share": float(stock_counts.max()) if len(stock_counts) else math.nan,
        "supported": bool(
            len(group) >= 30
            and group["session"].nunique() >= 15
            and group["stock"].nunique() >= 8
            and group["month"].nunique() >= 3
            and len(stock_counts)
            and stock_counts.max() <= 0.30
        ),
    }


def bh_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result.tolist()
    order = finite[np.argsort(values[finite])]
    adjusted = np.minimum.accumulate(
        (values[order] * len(order) / np.arange(1, len(order) + 1))[::-1]
    )[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values with finite values only."""

    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result.tolist()
    order = finite[np.argsort(values[finite])]
    adjusted = np.maximum.accumulate(values[order] * (len(order) - np.arange(len(order))))
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def clock_phase_context(group: pd.DataFrame, *, prefix: str = "") -> dict[str, Any]:
    """Summarise the frozen clock context without expanding sequence identity."""

    values = group["clock_phase"].astype("string")
    available = values.notna() & ~values.eq("unavailable")
    counts = values[available].value_counts().sort_index()
    dominant = (
        sorted(counts[counts.eq(counts.max())].index.astype(str))[0]
        if len(counts)
        else "unavailable"
    )
    return {
        f"{prefix}clock_phase_available_rows": int(available.sum()),
        f"{prefix}clock_phase_missing_rows": int((~available).sum()),
        f"{prefix}clock_phase_availability_rate": float(available.mean())
        if len(group)
        else math.nan,
        f"{prefix}dominant_clock_phase": dominant,
        f"{prefix}clock_phase_counts_json": json.dumps(
            {str(key): int(value) for key, value in counts.items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def sequence_bootstrap_seed(*parts: object, base_seed: int = 20260717) -> int:
    """Derive the registered stable per-contrast bootstrap seed."""

    identity = "|".join(str(part) for part in parts).encode()
    offset = int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")
    return (base_seed + offset) % (2**32)


def session_block_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    seed: int,
    resamples: int = 1000,
    block_length: int = 5,
) -> dict[str, float | int]:
    """Bootstrap a row-mean difference while clustering whole sessions."""

    sessions = sorted(set(left["session"].astype(str)) | set(right["session"].astype(str)))
    if not sessions or resamples < 1 or block_length < 1:
        return {
            "lower_95": math.nan,
            "upper_95": math.nan,
            "two_sided_p_value": math.nan,
            "valid_resamples": 0,
        }

    def sums_and_counts(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        grouped = frame.groupby(frame["session"].astype(str))["net_payoff_bps"].agg(
            ["sum", "count"]
        )
        return (
            grouped["sum"].reindex(sessions, fill_value=0.0).to_numpy(dtype=float),
            grouped["count"].reindex(sessions, fill_value=0).to_numpy(dtype=float),
        )

    left_sum, left_count = sums_and_counts(left)
    right_sum, right_count = sums_and_counts(right)
    rng = np.random.default_rng(seed)
    session_count = len(sessions)
    block_count = math.ceil(session_count / block_length)
    offsets = np.arange(block_length)
    differences: list[float] = []
    for _ in range(resamples):
        starts = rng.integers(0, session_count, size=block_count)
        sampled = ((starts[:, None] + offsets[None, :]) % session_count).ravel()[:session_count]
        left_n = float(left_count[sampled].sum())
        right_n = float(right_count[sampled].sum())
        if left_n <= 0 or right_n <= 0:
            continue
        differences.append(
            float(left_sum[sampled].sum() / left_n - right_sum[sampled].sum() / right_n)
        )
    if not differences:
        return {
            "lower_95": math.nan,
            "upper_95": math.nan,
            "two_sided_p_value": math.nan,
            "valid_resamples": 0,
        }
    values = np.asarray(differences, dtype=float)
    tail = min(int(np.sum(values <= 0.0)), int(np.sum(values >= 0.0)))
    p_value = min(1.0, 2.0 * (1.0 + tail) / (1.0 + len(values)))
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "two_sided_p_value": p_value,
        "valid_resamples": int(len(values)),
    }


def sequence_analysis(
    occurrences: pd.DataFrame, component_panel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    join_columns = ["period", "session", "loop", "regime"]
    components = component_panel.loc[
        component_panel["robust_net_payoff_bps"].notna(),
        [*join_columns, "common_component", "regime_component", "loop_excess_component"],
    ]
    rows = occurrences.merge(components, on=join_columns, how="left", validate="many_to_one")
    census_records: list[dict[str, Any]] = []
    four_way_records: list[dict[str, Any]] = []
    increment_records: list[dict[str, Any]] = []
    for length, sequence_column in [(2, "regime_history_2"), (3, "regime_history_3")]:
        available = rows[~rows[sequence_column].astype(str).eq("unavailable")].copy()
        for (period_value, loop_value, regime_value, sequence_value), target in available.groupby(
            ["period", "loop", "regime", sequence_column], sort=True, observed=True
        ):
            period = str(period_value)
            loop = str(loop_value)
            regime = str(regime_value)
            sequence = str(sequence_value)
            summary = support_summary(target)
            census_records.append(
                {
                    "period": str(period),
                    "loop": loop,
                    "regime": regime,
                    "pair": f"{loop}|{regime}",
                    "sequence_length": length,
                    "sequence": sequence,
                    **summary,
                    "occurrence_rate": len(target)
                    / max(
                        len(
                            available[
                                available["loop"].astype(str).eq(loop)
                                & available["regime"].astype(str).eq(regime)
                            ]
                        ),
                        1,
                    ),
                    "mean_net_payoff_bps": float(target["net_payoff_bps"].mean()),
                    "median_net_payoff_bps": float(target["net_payoff_bps"].median()),
                    "positive_rate": float(target["net_payoff_bps"].gt(0).mean()),
                    "total_positive_payoff": float(target["positive_payoff"].sum()),
                    "twice_cost_mean_net_payoff_bps": float(
                        target["twice_cost_net_payoff_bps"].mean()
                    ),
                    **clock_phase_context(target),
                }
            )
            same_regime = available[
                available["period"].astype(str).eq(period)
                & available["regime"].astype(str).eq(regime)
            ].copy()
            is_loop = same_regime["loop"].astype(str).eq(loop)
            is_sequence = same_regime[sequence_column].astype(str).eq(sequence)
            same_regime["counterfactual_group"] = np.select(
                [is_loop & is_sequence, is_loop & ~is_sequence, ~is_loop & is_sequence],
                ["1", "2", "3"],
                default="4",
            )
            groups: dict[str, pd.DataFrame] = {}
            group_supports: dict[str, dict[str, Any]] = {}
            for group_id in ["1", "2", "3", "4"]:
                group = same_regime[same_regime["counterfactual_group"].eq(group_id)]
                groups[group_id] = group
                group_support = support_summary(group)
                group_supports[group_id] = group_support
                four_way_records.append(
                    {
                        "period": str(period),
                        "loop": loop,
                        "regime": regime,
                        "pair": f"{loop}|{regime}",
                        "sequence_length": length,
                        "sequence": sequence,
                        "counterfactual_group": group_id,
                        **group_support,
                        "mean_net_payoff_bps": float(group["net_payoff_bps"].mean()),
                        "median_net_payoff_bps": float(group["net_payoff_bps"].median()),
                        "positive_rate": float(group["net_payoff_bps"].gt(0).mean()),
                        "total_positive_payoff": float(group["positive_payoff"].sum()),
                        "occurrence_rate": len(group) / max(len(same_regime), 1),
                        "common_component": float(group["common_component"].mean()),
                        "regime_component": float(group["regime_component"].mean()),
                        "loop_excess_component": float(group["loop_excess_component"].mean()),
                        "twice_cost_mean_net_payoff_bps": float(
                            group["twice_cost_net_payoff_bps"].mean()
                        ),
                        **clock_phase_context(group),
                    }
                )
            target_other = groups["2"]
            other_target = groups["3"]
            sequence_increment = float(
                target["net_payoff_bps"].mean() - target_other["net_payoff_bps"].mean()
            )
            loop_increment = float(
                target["net_payoff_bps"].mean() - other_target["net_payoff_bps"].mean()
            )
            comparison_supported = bool(
                summary["supported"]
                and group_supports["2"]["supported"]
                and group_supports["3"]["supported"]
            )
            if comparison_supported:
                seed_parts = (period, loop, regime, length, sequence)
                sequence_bootstrap = session_block_difference(
                    target,
                    target_other,
                    seed=sequence_bootstrap_seed(*seed_parts, "sequence_increment"),
                )
                loop_bootstrap = session_block_difference(
                    target,
                    other_target,
                    seed=sequence_bootstrap_seed(*seed_parts, "loop_increment"),
                )
            else:
                sequence_bootstrap = {
                    "lower_95": math.nan,
                    "upper_95": math.nan,
                    "two_sided_p_value": math.nan,
                    "valid_resamples": 0,
                }
                loop_bootstrap = dict(sequence_bootstrap)
            increment_records.append(
                {
                    "period": str(period),
                    "loop": loop,
                    "regime": regime,
                    "pair": f"{loop}|{regime}",
                    "sequence_length": length,
                    "sequence": sequence,
                    "sequence_increment_bps": sequence_increment,
                    "loop_increment_within_sequence_bps": loop_increment,
                    "both_directionally_positive": bool(
                        sequence_increment > 0 and loop_increment > 0
                    ),
                    "target_support": summary["supported"],
                    "other_sequence_same_loop_support": group_supports["2"]["supported"],
                    "other_loop_target_sequence_support": group_supports["3"]["supported"],
                    "other_loop_other_sequence_support": group_supports["4"]["supported"],
                    "increment_comparison_supported": comparison_supported,
                    "all_four_groups_supported": bool(
                        summary["supported"]
                        and all(
                            group_supports[group_id]["supported"] for group_id in ["2", "3", "4"]
                        )
                    ),
                    "sequence_increment_bootstrap_lower_95": sequence_bootstrap["lower_95"],
                    "sequence_increment_bootstrap_upper_95": sequence_bootstrap["upper_95"],
                    "sequence_increment_bootstrap_valid_resamples": sequence_bootstrap[
                        "valid_resamples"
                    ],
                    "sequence_increment_p_value": sequence_bootstrap["two_sided_p_value"],
                    "loop_increment_bootstrap_lower_95": loop_bootstrap["lower_95"],
                    "loop_increment_bootstrap_upper_95": loop_bootstrap["upper_95"],
                    "loop_increment_bootstrap_valid_resamples": loop_bootstrap["valid_resamples"],
                    "loop_increment_p_value": loop_bootstrap["two_sided_p_value"],
                    **clock_phase_context(target, prefix="target_"),
                }
            )
    census = pd.DataFrame.from_records(census_records)
    four_way = pd.DataFrame.from_records(four_way_records)
    increments = pd.DataFrame.from_records(increment_records)
    supported_inference = increments["increment_comparison_supported"].astype(bool)
    increments["sequence_increment_fdr_q_value"] = bh_adjust(
        increments["sequence_increment_p_value"].where(supported_inference).tolist()
    )
    increments["loop_increment_fdr_q_value"] = bh_adjust(
        increments["loop_increment_p_value"].where(supported_inference).tolist()
    )
    increments["interaction_fdr_pass"] = (
        supported_inference
        & increments["sequence_increment_fdr_q_value"].le(0.05)
        & increments["loop_increment_fdr_q_value"].le(0.05)
    )
    stability_keys = ["loop", "regime", "sequence_length", "sequence"]
    stability_records: list[dict[str, Any]] = []
    for key, group in increments.groupby(stability_keys, sort=True, observed=True):
        direction = group.set_index("period")["both_directionally_positive"].to_dict()
        supported = group.set_index("period")["increment_comparison_supported"].to_dict()
        multiplicity = group.set_index("period")["interaction_fdr_pass"].to_dict()
        stability_records.append(
            {
                "loop": key[0],
                "regime": key[1],
                "pair": f"{key[0]}|{key[1]}",
                "sequence_length": key[2],
                "sequence": key[3],
                "positive_in_2023": bool(direction.get("2023", False)),
                "positive_in_2025": bool(direction.get("2025", False)),
                "supported_in_2023": bool(supported.get("2023", False)),
                "supported_in_2025": bool(supported.get("2025", False)),
                "fdr_pass_in_2023": bool(multiplicity.get("2023", False)),
                "fdr_pass_in_2025": bool(multiplicity.get("2025", False)),
                "same_positive_direction_both_periods": bool(
                    direction.get("2023", False) and direction.get("2025", False)
                ),
                "adequately_supported_same_direction_both_periods": bool(
                    direction.get("2023", False)
                    and direction.get("2025", False)
                    and supported.get("2023", False)
                    and supported.get("2025", False)
                ),
                "multiplicity_controlled_interaction_both_periods": bool(
                    direction.get("2023", False)
                    and direction.get("2025", False)
                    and supported.get("2023", False)
                    and supported.get("2025", False)
                    and multiplicity.get("2023", False)
                    and multiplicity.get("2025", False)
                ),
            }
        )
    stability = pd.DataFrame.from_records(stability_records)
    return census, four_way, increments, stability


def block_network(
    null_rows: pd.DataFrame, *, resamples: int, seed: int, block_length: int
) -> pd.DataFrame:
    frame = null_rows.copy()
    frame["period"] = frame["period"].astype(str)
    pairs = sorted(frame["pair"].unique())
    edge_indices = [
        (left, right) for left in range(len(pairs)) for right in range(left + 1, len(pairs))
    ]
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    observed_matrix = np.zeros((len(pairs), len(pairs)), dtype=float)
    eligible_matrix = np.zeros((len(pairs), len(pairs)), dtype=float)
    matrices: list[tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]] = []
    for _period, group in frame.groupby("period", sort=True):
        sessions = sorted(group["session"].astype(str).unique())
        session_index = {session: index for index, session in enumerate(sessions)}
        values = np.full((len(sessions), len(pairs)), np.nan)
        for row in group.itertuples():
            values[session_index[str(row.session)], pair_index[str(row.pair)]] = float(
                bool(row.positive_pair_flag)
            )
        eligible = np.isfinite(values)
        binary = np.nan_to_num(values, nan=0.0)
        observed_matrix += binary.T @ binary
        eligible_matrix += eligible.astype(float).T @ eligible.astype(float)
        eligible_indices = [np.flatnonzero(eligible[:, column]) for column in range(len(pairs))]
        sequences = [
            binary[indices, column].copy() for column, indices in enumerate(eligible_indices)
        ]
        matrices.append((binary, eligible_indices, sequences))
    simulations = np.zeros((len(edge_indices), resamples), dtype=float)
    rng = np.random.default_rng(seed)
    for replicate in range(resamples):
        total_coactivation = np.zeros_like(observed_matrix)
        for template, eligible_indices, sequences in matrices:
            shifted = np.zeros_like(template)
            for column, (indices, sequence) in enumerate(
                zip(eligible_indices, sequences, strict=True)
            ):
                if not len(sequence):
                    continue
                offset = int(rng.integers(0, max(1, math.ceil(len(sequence) / block_length))))
                offset = (offset * block_length) % len(sequence)
                shifted[indices, column] = np.roll(sequence, offset)
            total_coactivation += shifted.T @ shifted
        for edge_number, (left, right) in enumerate(edge_indices):
            simulations[edge_number, replicate] = total_coactivation[left, right]
    records: list[dict[str, Any]] = []
    for edge_number, (left_index, right_index) in enumerate(edge_indices):
        null = simulations[edge_number]
        observed = int(observed_matrix[left_index, right_index])
        upper = float(np.quantile(null, 0.95))
        records.append(
            {
                "pair_left": pairs[left_index],
                "pair_right": pairs[right_index],
                "eligible_sessions": int(eligible_matrix[left_index, right_index]),
                "observed_coactivations": observed,
                "block_null_expected_coactivations": float(null.mean()),
                "block_null_sd": float(null.std(ddof=1)),
                "block_null_95th_percentile": upper,
                "excess_coactivations": float(observed - null.mean()),
                "one_sided_empirical_p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
                "display_edge": bool(observed >= 5 and observed > upper),
            }
        )
    return pd.DataFrame.from_records(records)


def add_network_node_metadata(
    network: pd.DataFrame,
    panel: pd.DataFrame,
    same_regime_loops: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the frozen descriptive node attributes to every network edge."""

    supported = panel[panel["robust_net_payoff_bps"].notna()].copy()
    supported["positive_payoff"] = supported["robust_net_payoff_bps"].clip(lower=0.0)
    node = (
        supported.groupby("pair", observed=True)
        .agg(
            node_total_positive_payoff=("positive_payoff", "sum"),
            node_regime=("regime", "first"),
            node_loop=("loop", "first"),
            node_supported_sessions=("session", "nunique"),
        )
        .reset_index()
    )
    same_scope = same_regime_loops.copy()
    leader = (
        same_scope.groupby("pair", observed=True)
        .agg(
            node_episode_appearances=("episode_id", "nunique"),
            node_leader_episodes=("is_final_leader", "sum"),
        )
        .reset_index()
    )
    leader["node_leader_frequency"] = leader["node_leader_episodes"] / leader[
        "node_episode_appearances"
    ].replace(0, np.nan)
    node = node.merge(leader, on="pair", how="left", validate="one_to_one")
    node["node_episode_appearances"] = node["node_episode_appearances"].fillna(0).astype(int)
    node["node_leader_episodes"] = node["node_leader_episodes"].fillna(0).astype(int)
    result = network.copy()
    for side in ["left", "right"]:
        renamed = node.rename(
            columns={
                "pair": f"pair_{side}",
                **{column: f"{column}_{side}" for column in node.columns if column != "pair"},
            }
        )
        result = result.merge(renamed, on=f"pair_{side}", how="left", validate="many_to_one")
    result["pair"] = result["pair_left"].astype(str) + ";" + result["pair_right"].astype(str)
    result["loop"] = (
        result["node_loop_left"].astype(str) + ";" + result["node_loop_right"].astype(str)
    )
    result["orientation"] = (
        result["node_regime_left"].astype(str) + ";" + result["node_regime_right"].astype(str)
    )
    result["regime"] = result["orientation"]
    return result


def session_block_rank_interval(
    data: pd.DataFrame,
    indicator: str,
    component: str,
    *,
    seed_label: str,
    resamples: int = 1000,
) -> tuple[float, float]:
    if len(data) < 3 or data[indicator].nunique() < 2 or data[component].nunique() < 2:
        return math.nan, math.nan
    ranked_x = data[indicator].rank(method="average").to_numpy(dtype=float)
    ranked_y = data[component].rank(method="average").to_numpy(dtype=float)
    sessions = sorted(data["session"].astype(str).unique())
    blocks = [sessions[index : index + 5] for index in range(0, len(sessions), 5)]
    row_indices = {
        session: np.flatnonzero(data["session"].astype(str).eq(session).to_numpy())
        for session in sessions
    }
    seed = int(hashlib.sha256(seed_label.encode()).hexdigest()[:8], 16) ^ 20260717
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        selected_blocks = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate(
            [
                row_indices[session]
                for block_index in selected_blocks
                for session in blocks[int(block_index)]
            ]
        )
        if len(indices) < 3:
            continue
        estimate = float(np.corrcoef(ranked_x[indices], ranked_y[indices])[0, 1])
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return math.nan, math.nan
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def distribution_overlap(left: pd.Series, right: pd.Series) -> float:
    left_values = left.dropna().to_numpy(dtype=float)
    right_values = right.dropna().to_numpy(dtype=float)
    if not len(left_values) or not len(right_values):
        return math.nan
    lower = min(float(left_values.min()), float(right_values.min()))
    upper = max(float(left_values.max()), float(right_values.max()))
    if lower == upper:
        return 1.0
    bins = np.linspace(lower, upper, 11)
    left_hist, _ = np.histogram(left_values, bins=bins)
    right_hist, _ = np.histogram(right_values, bins=bins)
    left_density = left_hist / left_hist.sum()
    right_density = right_hist / right_hist.sum()
    return float(np.minimum(left_density, right_density).sum())


def panel_population_support(panel_rows: pd.DataFrame, occurrences: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen support gate using exact occurrence-stock identities."""

    if panel_rows.empty:
        return {
            "rows": 0,
            "independent_sessions": 0,
            "independent_stocks": 0,
            "months": 0,
            "max_stock_row_share": math.nan,
            "supported": False,
        }
    keys = panel_rows[["period", "session", "loop", "regime"]].drop_duplicates().copy()
    keys["period"] = keys["period"].astype(str)
    occurrence_keys = occurrences.copy()
    occurrence_keys["period"] = occurrence_keys["period"].astype(str)
    matched = occurrence_keys.merge(
        keys,
        on=["period", "session", "loop", "regime"],
        how="inner",
        validate="many_to_one",
    )
    stock_shares = matched["stock"].value_counts(normalize=True)
    sessions = int(panel_rows["session"].nunique())
    stocks = int(matched["stock"].nunique())
    months = int(panel_rows["session"].astype(str).str[:7].nunique())
    max_stock_share = float(stock_shares.max()) if len(stock_shares) else math.nan
    return {
        "rows": int(len(panel_rows)),
        "independent_sessions": sessions,
        "independent_stocks": stocks,
        "months": months,
        "max_stock_row_share": max_stock_share,
        "supported": bool(
            len(panel_rows) >= 30
            and sessions >= 15
            and stocks >= 8
            and months >= 3
            and len(stock_shares)
            and max_stock_share <= 0.30
        ),
    }


def indicator_associations(panel: pd.DataFrame, occurrences: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    components = ["common_component", "regime_component", "loop_excess_component"]
    for period_label, period_frame in [
        ("all", panel),
        *[(str(period), group) for period, group in panel.groupby("period", sort=True)],
    ]:
        for indicator in INDICATORS:
            for component in components:
                data = period_frame[
                    ["period", "session", "loop", "regime", indicator, component]
                ].dropna()
                support = panel_population_support(data, occurrences)
                if len(data) < 3 or data[indicator].nunique() < 2 or data[component].nunique() < 2:
                    coefficient = p_value = math.nan
                else:
                    test = stats.spearmanr(data[indicator], data[component])
                    coefficient = float(test.statistic)
                    p_value = float(test.pvalue)
                lower = data[indicator].quantile(1 / 3) if len(data) else math.nan
                upper = data[indicator].quantile(2 / 3) if len(data) else math.nan
                low_values = data.loc[data[indicator].le(lower), component]
                high_values = data.loc[data[indicator].ge(upper), component]
                pooled_sd = float(data[component].std(ddof=1))
                effect = (
                    float((high_values.mean() - low_values.mean()) / pooled_sd)
                    if pooled_sd > 0 and len(low_values) and len(high_values)
                    else math.nan
                )
                interval_low, interval_high = session_block_rank_interval(
                    data,
                    indicator,
                    component,
                    seed_label=f"{period_label}|{indicator}|{component}",
                )
                records.append(
                    {
                        "period": period_label,
                        "indicator": indicator,
                        "component": component,
                        "rows": len(data),
                        "independent_sessions": data["session"].nunique(),
                        "independent_stocks": support["independent_stocks"],
                        "months": support["months"],
                        "max_stock_row_share": support["max_stock_row_share"],
                        "support_passed": support["supported"],
                        "spearman_rank_correlation": coefficient,
                        "session_block_bootstrap_lower_95": interval_low,
                        "session_block_bootstrap_upper_95": interval_high,
                        "p_value": p_value,
                        "lower_third_boundary": lower,
                        "upper_third_boundary": upper,
                        "standardised_upper_minus_lower_effect": effect,
                        "median_upper_minus_lower": float(
                            high_values.median() - low_values.median()
                        )
                        if len(low_values) and len(high_values)
                        else math.nan,
                        "upper_lower_distribution_overlap": distribution_overlap(
                            low_values, high_values
                        ),
                        "missing_fraction": 1.0 - len(data) / max(len(period_frame), 1),
                        "stock_consistency_status": "exact_stock_support_gate_applied",
                    }
                )
    result = pd.DataFrame.from_records(records)
    result["fdr_q_value"] = bh_adjust(result["p_value"].tolist())
    result["reportable_after_support_and_fdr"] = result["support_passed"] & result[
        "fdr_q_value"
    ].le(0.05)
    direction: dict[tuple[str, str], bool] = {}
    for key, group in result[result["period"].isin(["2023", "2025"])].groupby(
        ["indicator", "component"], sort=True
    ):
        estimates = group.set_index("period")["spearman_rank_correlation"].to_dict()
        direction[(str(key[0]), str(key[1]))] = bool(
            math.isfinite(estimates.get("2023", math.nan))
            and math.isfinite(estimates.get("2025", math.nan))
            and np.sign(estimates["2023"]) == np.sign(estimates["2025"])
        )
    result["period_direction_consistent"] = [
        direction.get((str(row.indicator), str(row.component)), False)
        for row in result.itertuples()
    ]
    result["comparisons_examined"] = len(result)
    return result


def manifestation_table(timeline: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    phases = ["pre", "onset", "early", "middle", "late", "decay", "post"]
    numeric_fields = [
        "common_component",
        "dominant_regime_component",
        "dominant_loop_specific_excess",
        "volatility_summary",
        "leader_efficiency",
    ]
    for level, level_frame in timeline.groupby("episode_level", sort=True):
        for field in numeric_fields:
            for phase in phases:
                values = level_frame.loc[level_frame["episode_phase"].eq(phase), field].dropna()
                records.append(
                    {
                        "episode_level": level,
                        "indicator": field,
                        "episode_phase": phase,
                        "rows": len(values),
                        "median": float(values.median()),
                        "mean": float(values.mean()),
                        "standard_deviation": float(values.std(ddof=1)),
                    }
                )
    return pd.DataFrame.from_records(records)


def indicator_manifestation_table(
    panel: pd.DataFrame,
    episodes: pd.DataFrame,
    pair_ledger: pd.DataFrame,
    calendars: dict[str, list[str]],
    occurrences: pd.DataFrame,
) -> pd.DataFrame:
    observations: list[pd.DataFrame] = []
    for _, episode in episodes.iterrows():
        members = episode_members(pair_ledger, episode)
        member_pairs = set(members["pair"].astype(str))
        period = str(episode["period"])
        sessions = calendars[period]
        start = sessions.index(str(episode["onset"]))
        end = sessions.index(str(episode["end"]))
        duration = end - start + 1
        selected_sessions = sessions[max(0, start - 10) : end + 1]
        selected = panel[
            panel["period"].astype(str).eq(period)
            & panel["session"].isin(selected_sessions)
            & panel["pair"].isin(member_pairs)
        ].copy()
        if selected.empty:
            continue
        positions = {session: index for index, session in enumerate(sessions)}
        selected["episode_phase"] = [
            episode_phase(positions[str(session)] - start, duration)
            for session in selected["session"]
        ]
        leader_loops = str(episode["final_dominant_loop"]).split("|")
        selected["is_final_leader"] = selected["loop"].astype(str).isin(leader_loops)
        selected["profitable_manifestation"] = selected["robust_net_payoff_bps"].gt(0)
        selected["episode_id"] = episode["episode_id"]
        observations.append(selected)
    population = pd.concat(observations, ignore_index=True)
    comparisons = {
        "onset_vs_pre": (
            population["episode_phase"].eq("onset"),
            population["episode_phase"].eq("pre"),
        ),
        "middle_vs_pre": (
            population["episode_phase"].eq("middle"),
            population["episode_phase"].eq("pre"),
        ),
        "decay_vs_middle": (
            population["episode_phase"].eq("decay"),
            population["episode_phase"].eq("middle"),
        ),
        "dominant_vs_other_positive": (
            population["is_final_leader"]
            & population["profitable_manifestation"]
            & ~population["episode_phase"].eq("pre"),
            ~population["is_final_leader"]
            & population["profitable_manifestation"]
            & ~population["episode_phase"].eq("pre"),
        ),
    }
    comparison_scopes: list[tuple[str, pd.Series, pd.Series, str]] = [
        (name, masks[0], masks[1], "all_pairs") for name, masks in comparisons.items()
    ]
    for pair in sorted(population["pair"].astype(str).unique()):
        pair_mask = population["pair"].astype(str).eq(pair)
        comparison_scopes.append(
            (
                "profitable_vs_unprofitable_same_pair",
                pair_mask
                & population["profitable_manifestation"]
                & ~population["episode_phase"].eq("pre"),
                pair_mask
                & ~population["profitable_manifestation"]
                & population["robust_net_payoff_bps"].notna()
                & ~population["episode_phase"].eq("pre"),
                pair,
            )
        )
    records: list[dict[str, Any]] = []
    for period_label, period_mask in [
        ("all", pd.Series(True, index=population.index)),
        *[
            (str(period), population["period"].astype(str).eq(str(period)))
            for period in sorted(population["period"].unique())
        ],
    ]:
        for comparison, left_mask, right_mask, pair_scope in comparison_scopes:
            if pair_scope == "all_pairs":
                loop_scope = "all_loops"
                regime_scope = "all_regimes"
            else:
                loop_scope, regime_scope = pair_scope.split("|", maxsplit=1)
            for indicator in INDICATORS:
                left = population.loc[period_mask & left_mask, indicator].dropna()
                right = population.loc[period_mask & right_mask, indicator].dropna()
                left_population = population.loc[
                    period_mask & left_mask & population[indicator].notna()
                ]
                right_population = population.loc[
                    period_mask & right_mask & population[indicator].notna()
                ]
                left_support = panel_population_support(left_population, occurrences)
                right_support = panel_population_support(right_population, occurrences)
                pooled = pd.concat([left, right], ignore_index=True)
                pooled_sd = float(pooled.std(ddof=1))
                p_value = (
                    float(stats.mannwhitneyu(left, right, alternative="two-sided").pvalue)
                    if len(left) and len(right)
                    else math.nan
                )
                records.append(
                    {
                        "period": period_label,
                        "pair": pair_scope,
                        "loop": loop_scope,
                        "orientation": regime_scope,
                        "regime": regime_scope,
                        "comparison": comparison,
                        "indicator": indicator,
                        "left_rows": len(left),
                        "right_rows": len(right),
                        "left_independent_sessions": left_support["independent_sessions"],
                        "right_independent_sessions": right_support["independent_sessions"],
                        "left_independent_stocks": left_support["independent_stocks"],
                        "right_independent_stocks": right_support["independent_stocks"],
                        "left_months": left_support["months"],
                        "right_months": right_support["months"],
                        "left_max_stock_row_share": left_support["max_stock_row_share"],
                        "right_max_stock_row_share": right_support["max_stock_row_share"],
                        "comparison_support_passed": bool(
                            left_support["supported"] and right_support["supported"]
                        ),
                        "left_median": float(left.median()) if len(left) else math.nan,
                        "right_median": float(right.median()) if len(right) else math.nan,
                        "median_difference": float(left.median() - right.median())
                        if len(left) and len(right)
                        else math.nan,
                        "standardised_mean_difference": float(
                            (left.mean() - right.mean()) / pooled_sd
                        )
                        if pooled_sd > 0 and len(left) and len(right)
                        else math.nan,
                        "distribution_overlap": distribution_overlap(left, right),
                        "p_value": p_value,
                        "stock_consistency_status": "exact_stock_support_gate_applied",
                        "causal_timestamp_rule": "source_feature_available_no_later_than_decision",
                    }
                )
    result = pd.DataFrame.from_records(records)
    result["fdr_q_value"] = bh_adjust(result["p_value"].tolist())
    result["reportable_after_support_and_fdr"] = result["comparison_support_passed"] & result[
        "fdr_q_value"
    ].le(0.05)
    consistency: dict[tuple[str, str], bool] = {}
    for key, group in result[result["period"].isin(["2023", "2025"])].groupby(
        ["comparison", "indicator"], sort=True
    ):
        effects = group.set_index("period")["median_difference"].to_dict()
        consistency[(str(key[0]), str(key[1]))] = bool(
            math.isfinite(effects.get("2023", math.nan))
            and math.isfinite(effects.get("2025", math.nan))
            and np.sign(effects["2023"]) == np.sign(effects["2025"])
        )
    result["period_direction_consistent"] = [
        consistency.get((str(row.comparison), str(row.indicator)), False)
        for row in result.itertuples()
    ]
    result["comparisons_examined"] = len(result)
    return result


def enriched_factor_diagnostic(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    supported = panel[panel["robust_net_payoff_bps"].notna()].copy()
    summary = common_factor_diagnostic(supported)
    loadings_records: list[dict[str, Any]] = []
    period_records: list[dict[str, Any]] = []
    loading_vectors: dict[str, pd.Series] = {}
    for period, group in supported.groupby("period", sort=True):
        matrix = group.pivot_table(
            index="session", columns="pair", values="robust_net_payoff_bps", aggfunc="mean"
        )
        centered = matrix - matrix.mean(axis=0)
        values = centered.fillna(0.0).to_numpy(dtype=float)
        left, singular, right = np.linalg.svd(values, full_matrices=False)
        scores = left[:, 0] * singular[0]
        common = (
            group.drop_duplicates("session")
            .set_index("session")["common_component"]
            .reindex(matrix.index)
        )
        relation = float(stats.spearmanr(scores, common).statistic)
        if math.isfinite(relation) and relation < 0:
            scores = -scores
            right[0] = -right[0]
            relation = -relation
        loadings = pd.Series(right[0], index=matrix.columns)
        loading_vectors[str(period)] = loadings
        for pair, loading in loadings.items():
            loop, regime = str(pair).split("|")
            loadings_records.append(
                {
                    "period": str(period),
                    "pair": pair,
                    "loop": loop,
                    "orientation": regime,
                    "regime": regime,
                    "factor": 1,
                    "loading": float(loading),
                    "fit_population": "same_period_only",
                }
            )
        raw_correlations: list[float] = []
        residual_correlations: list[float] = []
        period_grid = panel[panel["period"].eq(period)]
        for _pair, pair_rows in period_grid.groupby("pair", sort=True):
            ordered = pair_rows.sort_values("session")
            raw_pairs = pd.DataFrame(
                {
                    "current": ordered["robust_net_payoff_bps"],
                    "future": ordered["robust_net_payoff_bps"].shift(-1),
                }
            ).dropna()
            residual_pairs = pd.DataFrame(
                {
                    "current": ordered["loop_excess_component"],
                    "future": ordered["loop_excess_component"].shift(-1),
                }
            ).dropna()
            if (
                len(raw_pairs) >= 3
                and raw_pairs["current"].nunique() > 1
                and raw_pairs["future"].nunique() > 1
            ):
                raw_estimate = float(
                    stats.spearmanr(raw_pairs["current"], raw_pairs["future"]).statistic
                )
                if math.isfinite(raw_estimate):
                    raw_correlations.append(raw_estimate)
            if (
                len(residual_pairs) >= 3
                and residual_pairs["current"].nunique() > 1
                and residual_pairs["future"].nunique() > 1
            ):
                residual_estimate = float(
                    stats.spearmanr(residual_pairs["current"], residual_pairs["future"]).statistic
                )
                if math.isfinite(residual_estimate):
                    residual_correlations.append(residual_estimate)
        period_records.append(
            {
                "period": str(period),
                "first_component_common_rank_correlation": relation,
                "median_raw_pair_lag1_rank_correlation": float(np.median(raw_correlations))
                if raw_correlations
                else math.nan,
                "median_loop_excess_lag1_rank_correlation": float(np.median(residual_correlations))
                if residual_correlations
                else math.nan,
                "residual_stability_change": float(
                    np.median(residual_correlations) - np.median(raw_correlations)
                )
                if residual_correlations and raw_correlations
                else math.nan,
            }
        )
    loading_stability = math.nan
    if set(loading_vectors) >= {"2023", "2025"}:
        common_pairs = sorted(
            set(loading_vectors["2023"].index) & set(loading_vectors["2025"].index)
        )
        if len(common_pairs) >= 3:
            loading_stability = float(
                stats.spearmanr(
                    loading_vectors["2023"].reindex(common_pairs),
                    loading_vectors["2025"].reindex(common_pairs),
                ).statistic
            )
    diagnostics = pd.DataFrame.from_records(period_records)
    diagnostics["first_factor_loading_rank_correlation_2023_2025"] = loading_stability
    summary = summary.merge(diagnostics, on="period", how="left", validate="many_to_one")
    return summary, pd.DataFrame.from_records(loadings_records)


def cohort_contribution_tables(
    episodes: pd.DataFrame,
    pair_ledger: pd.DataFrame,
    occurrences: pd.DataFrame,
    panel: pd.DataFrame,
    calendars: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contribution_records: list[dict[str, Any]] = []
    leave_one_records: list[dict[str, Any]] = []
    dimensions = {
        "stock": "stock",
        "month": "month",
        "quarter": "quarter",
        "period": "period",
        "liquidity_cohort": "volume_bucket",
    }
    for _, episode in episodes.iterrows():
        members = episode_members(pair_ledger, episode)
        period = str(episode["period"])
        period_sessions = calendars[period]
        start = period_sessions.index(str(episode["onset"]))
        end = period_sessions.index(str(episode["end"]))
        episode_sessions = period_sessions[start : end + 1]
        member_pair_values = sorted(set(members["pair"].astype(str)))
        member_loop_values = sorted(set(members["loop"].astype(str)))
        member_regime_values = sorted(set(members["regime"].astype(str)))
        trace_scope = {
            "period": period,
            "session": f"{episode['onset']}..{episode['end']}",
            "episode_id": episode["episode_id"],
            "pair": ";".join(member_pair_values),
            "loop": ";".join(member_loop_values),
            "orientation": ";".join(member_regime_values),
            "regime": ";".join(member_regime_values),
        }
        selected = occurrences[
            occurrences["period"].astype(str).eq(period)
            & occurrences["session"].isin(episode_sessions)
            & occurrences["pair"].isin(set(member_pair_values))
        ].copy()
        all_session_occurrences = occurrences[
            occurrences["period"].astype(str).eq(period)
            & occurrences["session"].isin(episode_sessions)
        ].copy()
        supported_session_panel = panel[
            panel["period"].astype(str).eq(period)
            & panel["session"].isin(episode_sessions)
            & panel["robust_net_payoff_bps"].notna()
        ]
        supported_cells = set(
            zip(
                supported_session_panel["session"].astype(str),
                supported_session_panel["pair"].astype(str),
                strict=False,
            )
        )
        occurrence_cells = set(
            zip(
                all_session_occurrences["session"].astype(str),
                all_session_occurrences["pair"].astype(str),
                strict=False,
            )
        )
        occurrence_population_complete = bool(
            supported_cells and supported_cells.issubset(occurrence_cells)
        )
        total = float(selected["positive_payoff"].sum())
        for dimension, column in dimensions.items():
            grouped = selected.groupby(column, observed=True)["positive_payoff"].agg(
                ["sum", "size"]
            )
            grouped = grouped.sort_values("sum", ascending=False)
            shares = grouped["sum"] / total if total > 0 else grouped["sum"] * math.nan
            hhi = float(np.square(shares).sum()) if len(shares) else math.nan
            for rank, (value, row) in enumerate(grouped.iterrows(), start=1):
                contribution_records.append(
                    {
                        **trace_scope,
                        "episode_level": episode["episode_level"],
                        "cohort_dimension": dimension,
                        "cohort_value": str(value),
                        "rank": rank,
                        "occurrence_rows": int(row["size"]),
                        "positive_payoff": float(row["sum"]),
                        "positive_payoff_share": float(row["sum"] / total)
                        if total > 0
                        else math.nan,
                        "dimension_herfindahl_index": hhi,
                        "occurrence_population_complete": occurrence_population_complete,
                    }
                )
        stock_payoff = (
            selected.groupby("stock", observed=True)["positive_payoff"]
            .sum()
            .sort_values(ascending=False)
        )
        removal_scenarios: list[tuple[str, list[str]]] = [
            ("leave_one_stock_out", [str(stock)]) for stock in stock_payoff.index
        ]
        if len(stock_payoff):
            removal_scenarios.extend(
                [
                    ("remove_best_stock", [str(stock_payoff.index[0])]),
                    ("remove_top_five_stocks", [str(stock) for stock in stock_payoff.index[:5]]),
                ]
            )
        for removal_scope, removed_stocks in removal_scenarios:
            removed_positive_payoff = float(stock_payoff.reindex(removed_stocks).fillna(0.0).sum())
            if occurrence_population_complete:
                component_summary = recompute_component_summary_after_stock_removal(
                    all_session_occurrences,
                    member_pairs=member_pair_values,
                    removed_stocks=removed_stocks,
                )
            else:
                component_summary = {
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
            leave_one_records.append(
                {
                    **trace_scope,
                    "episode_level": episode["episode_level"],
                    "removal_scope": removal_scope,
                    "stock": ";".join(removed_stocks),
                    "removed_stock_count": len(removed_stocks),
                    "removed_positive_payoff": removed_positive_payoff,
                    "positive_payoff_after_removal": total - removed_positive_payoff,
                    **component_summary,
                    "occurrence_population_complete": occurrence_population_complete,
                    "component_summary_recalculated": occurrence_population_complete,
                    "row_deletion_only_not_retrained_model": True,
                    "model_retrained": False,
                }
            )
    return (
        pd.DataFrame.from_records(contribution_records),
        pd.DataFrame.from_records(leave_one_records),
    )


def route_tables(
    route: pd.DataFrame,
    panel: pd.DataFrame,
    pair_ledger: pd.DataFrame,
    calendars: dict[str, list[str]],
) -> pd.DataFrame:
    frame = route.copy()
    frame["pair"] = (
        frame["candidate"]
        .astype(str)
        .str.replace("|state2", "|state_2", regex=False)
        .str.replace("|state4", "|state_4", regex=False)
        .str.replace("|state5", "|state_5", regex=False)
        .str.replace("|state6", "|state_6", regex=False)
    )
    frame["loop"] = frame["pair"].str.split("|").str[0]
    frame["regime"] = frame["pair"].str.split("|").str[1]
    frame["orientation"] = frame["regime"]
    frame["session"] = pd.to_datetime(frame["session_date"]).dt.strftime("%Y-%m-%d")
    component_lookup = panel.loc[
        panel["robust_net_payoff_bps"].notna(),
        ["period", "session", "pair", "loop_excess_component"],
    ].copy()
    frame = frame.merge(
        component_lookup,
        on=["period", "session", "pair"],
        how="left",
        validate="many_to_one",
    )
    records: list[dict[str, Any]] = []
    for episode in pair_ledger.itertuples(index=False):
        period = str(episode.period)
        sessions = calendars[period]
        start = sessions.index(str(episode.onset))
        end = sessions.index(str(episode.end))
        selected = frame[
            frame["period"].astype(str).eq(period)
            & frame["pair"].eq(str(episode.pair))
            & frame["session"].isin(sessions[start : end + 1])
        ].copy()
        if selected.empty:
            continue
        positions = {session: index for index, session in enumerate(sessions)}
        selected["episode_phase"] = [
            episode_phase(positions[session] - start, end - start + 1)
            for session in selected["session"]
        ]
        episode_total = len(selected)
        episode_positive_total = float(selected["fixed_net_bps"].clip(lower=0).sum())
        event_payoff_column = "terminal_route_event_next_open_else_fixed__net_bps"
        event_difference_column = "terminal_route_event_next_open_else_fixed__paired_difference_bps"
        for (phase_value, topology_value), group in selected.groupby(
            ["episode_phase", "path_topology"], sort=True, observed=True
        ):
            event_payoff = pd.to_numeric(group[event_payoff_column], errors="coerce")
            event_minus_fixed = pd.to_numeric(group[event_difference_column], errors="coerce")
            records.append(
                {
                    "period": period,
                    "episode_id": str(episode.episode_id),
                    "episode_level": "pair",
                    "episode_onset": str(episode.onset),
                    "episode_end": str(episode.end),
                    "pair": str(episode.pair),
                    "loop": str(episode.loop),
                    "regime": str(episode.regime),
                    "orientation": str(episode.orientation),
                    "episode_phase": str(phase_value),
                    "path_topology": str(topology_value),
                    "frequency": len(group),
                    "independent_stock_count": int(group["symbol_norm"].nunique()),
                    "frequency_share": len(group) / episode_total,
                    "mean_fixed_close_net_payoff_bps": float(group["fixed_net_bps"].mean()),
                    "median_fixed_close_net_payoff_bps": float(group["fixed_net_bps"].median()),
                    "total_positive_fixed_close_payoff_bps": float(
                        group["fixed_net_bps"].clip(lower=0).sum()
                    ),
                    "positive_payoff_share": float(group["fixed_net_bps"].clip(lower=0).sum())
                    / episode_positive_total
                    if episode_positive_total > 0
                    else math.nan,
                    "median_terminal_event_position": finite_median(
                        group["terminal_event_position"]
                    ),
                    "median_bars_consumed": finite_median(
                        group["terminal_event_position"] - group["entry_ordinal"]
                    ),
                    "median_route_detection_bars_after_entry": finite_median(
                        group[
                            "terminal_route_event_next_open_else_fixed__detection_bars_after_entry"
                        ]
                    ),
                    "payoff_before_route_event_bps": finite_mean(event_payoff),
                    "payoff_after_route_event_bps": finite_mean(-event_minus_fixed),
                    "route_payoff_semantics": (
                        "before=realised net through event next-open; "
                        "after=fixed-close net minus event next-open net"
                    ),
                    "loop_excess_component": finite_mean(group["loop_excess_component"]),
                    "loop_excess_component_available_rows": int(
                        group["loop_excess_component"].notna().sum()
                    ),
                    "outcome_only": True,
                }
            )
    return pd.DataFrame.from_records(records)


def path_deterioration(
    diagnostics: pd.DataFrame,
    _trades: pd.DataFrame,
    pair_ledger: pd.DataFrame,
    episodes: pd.DataFrame,
    panel: pd.DataFrame,
    calendars: dict[str, list[str]],
) -> pd.DataFrame:
    # The diagnostic already has one realised path per anchor. Joining it to
    # every compatible V2 pair would duplicate that outcome, so retain the
    # diagnostic's own top-loop/current-state identity.
    frame = diagnostics.copy()
    frame["loop_id"] = frame["top_loop"].astype(str)
    frame["orientation"] = "state_" + frame["anchor_state"].astype(int).astype(str)
    frame["session"] = pd.to_datetime(frame["session_date"]).dt.strftime("%Y-%m-%d")
    frame["pair"] = frame["loop_id"].astype(str) + "|" + frame["orientation"].astype(str)
    episode_lookup: list[dict[str, Any]] = []
    for _, episode in episodes.iterrows():
        members = episode_members(pair_ledger, episode)
        for pair in members["pair"].unique():
            episode_lookup.append(
                {
                    "episode_id": episode["episode_id"],
                    "period": str(episode["period"]),
                    "pair": pair,
                    "onset": episode["onset"],
                    "end": episode["end"],
                    "leader_loops": str(episode["final_dominant_loop"]),
                }
            )
    joined_rows: list[pd.DataFrame] = []
    for lookup in episode_lookup:
        selected = frame[
            frame["period"].astype(str).eq(lookup["period"])
            & frame["pair"].eq(lookup["pair"])
            & frame["session"].between(lookup["onset"], lookup["end"])
        ]
        if selected.empty:
            continue
        selected = selected.merge(
            panel[
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
        period_sessions = calendars[str(lookup["period"])]
        positions = {session: index for index, session in enumerate(period_sessions)}
        start = positions[str(lookup["onset"])]
        end = positions[str(lookup["end"])]
        selected["episode_phase"] = [
            episode_phase(positions[session] - start, end - start + 1)
            for session in selected["session"]
        ]
        leader_loops = set(str(lookup["leader_loops"]).split("|"))
        selected["episode_role"] = np.select(
            [
                selected["loop_id"].astype(str).isin(leader_loops),
                selected["robust_net_payoff_bps"].gt(0)
                | (
                    selected["positive_pair_flag"].fillna(False)
                    & selected["positive_pair_available"].fillna(False)
                ),
            ],
            ["final_episode_leader", "non_leading_positive"],
            default="negative_or_neutral_same_episode",
        )
        selected["episode_id"] = lookup["episode_id"]
        selected["negative_path_tail"] = selected["path_class"].isin(
            ["timing_failure", "no_usable_move"]
        )
        joined_rows.append(selected)
    if not joined_rows:
        return pd.DataFrame()
    population = pd.concat(joined_rows, ignore_index=True)
    records: list[dict[str, Any]] = []
    for (period, episode_id, pair, role, phase, path_class), group in population.groupby(
        ["period", "episode_id", "pair", "episode_role", "episode_phase", "path_class"],
        sort=True,
        observed=True,
    ):
        episode_id_text = str(episode_id)
        pair_text = str(pair)
        role_text = str(role)
        episode_population = population[
            population["episode_id"].eq(episode_id_text)
            & population["pair"].eq(pair_text)
            & population["episode_role"].eq(role_text)
        ]
        middle = episode_population[episode_population["episode_phase"].eq("middle")]
        late = episode_population[episode_population["episode_phase"].eq("late")]
        decay = episode_population[episode_population["episode_phase"].eq("decay")]
        deterioration_available = bool(len(middle) and len(late))
        middle_rate = float(middle["negative_path_tail"].mean()) if len(middle) else math.nan
        late_rate = float(late["negative_path_tail"].mean()) if len(late) else math.nan
        decay_rate = float(decay["negative_path_tail"].mean()) if len(decay) else math.nan
        records.append(
            {
                "period": str(period),
                "episode_id": episode_id_text,
                "episode_level": "same_regime",
                "pair": pair_text,
                "loop": str(group["loop_id"].iloc[0]),
                "orientation": str(group["orientation"].iloc[0]),
                "regime": str(group["orientation"].iloc[0]),
                "episode_role": role_text,
                "episode_phase": str(phase),
                "path_class": str(path_class),
                "rows": len(group),
                "negative_tail_frequency": float(group["negative_path_tail"].mean()),
                "timing_failure_frequency": float(group["path_class"].eq("timing_failure").mean()),
                "no_usable_move_frequency": float(group["path_class"].eq("no_usable_move").mean()),
                "median_mfe_bps_outcome_only": float(group["mfe_bps"].median()),
                "median_mae_bps_outcome_only": float(group["mae_bps"].median()),
                "mean_remaining_fixed_payoff_bps": float(group["net_return_bps"].mean()),
                "raw_one_to_three_bar_path_score": math.nan,
                "raw_path_score_status": "unavailable_in_retained_diagnostic",
                "negative_tail_middle_rate": middle_rate,
                "negative_tail_late_rate": late_rate,
                "negative_tail_at_decay_rate": decay_rate,
                "deterioration_rises_before_leader_decay": bool(late_rate > middle_rate)
                if deterioration_available
                else pd.NA,
                "deterioration_higher_at_decay_than_late": bool(decay_rate > late_rate)
                if len(late) and len(decay)
                else pd.NA,
                "deterioration_before_decay_available": deterioration_available,
                "causal_input": False,
            }
        )
    return pd.DataFrame.from_records(records)


def named_deep_dive(
    pair_ledger: pd.DataFrame,
    episodes: pd.DataFrame,
    loops: pd.DataFrame,
    route: pd.DataFrame,
    named_reference: pd.DataFrame,
    control_reference: pd.DataFrame,
    envelope: pd.DataFrame,
    occurrences: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    references = pd.concat([named_reference, control_reference], ignore_index=True)
    references["pair"] = (
        references["loop_id"].astype(str) + "|" + references["orientation"].astype(str)
    )
    envelope = envelope.copy()
    envelope["pair"] = envelope["loop_id"].astype(str) + "|" + envelope["orientation"].astype(str)
    for pair, role in NAMED_PAIRS.items():
        loop, regime = pair.split("|")
        pair_source = pair_ledger[pair_ledger["pair"].eq(pair)]
        matching_episode_ids = []
        for _, episode in episodes.iterrows():
            members = episode_members(pair_ledger, episode)
            if pair in set(members["pair"]):
                matching_episode_ids.append(episode["episode_id"])
        matching_loops = loops[
            loops["episode_id"].isin(matching_episode_ids) & loops["loop"].eq(loop)
        ]
        component_parts: list[pd.DataFrame] = []
        for pair_episode in pair_source.itertuples(index=False):
            current = panel[
                panel["period"].astype(str).eq(str(pair_episode.period))
                & panel["pair"].eq(pair)
                & panel["session"].between(str(pair_episode.onset), str(pair_episode.end))
                & panel["robust_net_payoff_bps"].notna()
            ].copy()
            current["pair_episode_id"] = str(pair_episode.episode_id)
            component_parts.append(current)
        pair_components = (
            pd.concat(component_parts, ignore_index=True)
            if component_parts
            else pd.DataFrame(
                columns=["common_component", "regime_component", "loop_excess_component"]
            )
        )
        pair_route = route[route["pair"].eq(pair)]
        pair_occurrences = occurrences[occurrences["pair"].eq(pair)]
        ref = references[references["pair"].eq(pair)]
        f0 = envelope[envelope["pair"].eq(pair) & envelope["fill_model"].eq("F0")]
        f10 = envelope[envelope["pair"].eq(pair) & envelope["fill_model"].eq("F10")]
        topology_counts = pair_route.groupby("path_topology", observed=True)["frequency"].sum()
        topology_total = max(int(pair_route["frequency"].sum()), 1)
        sequence_counts = pair_occurrences["regime_history_3"].value_counts().head(5)
        clock_counts = pair_occurrences["clock_phase"].value_counts()
        nonzero_payoffs = pair_occurrences.loc[
            pair_occurrences["net_payoff_bps"].ne(0), "net_payoff_bps"
        ]
        named_p_value = (
            float(stats.wilcoxon(nonzero_payoffs, alternative="two-sided").pvalue)
            if len(nonzero_payoffs) >= 2
            else math.nan
        )
        occurrence_support = support_summary(pair_occurrences)
        route_payoff_by_topology = "|".join(
            f"{topology}:{finite_mean(group['mean_fixed_close_net_payoff_bps'])}"
            for topology, group in pair_route.groupby("path_topology", sort=True)
        )
        records.append(
            {
                "pair": pair,
                "loop": loop,
                "orientation": regime,
                "regime": regime,
                "named_control_identity": role,
                "pair_episode_count": len(pair_source),
                "pair_episode_dates": "|".join(
                    f"{row.onset}:{row.end}" for row in pair_source.itertuples()
                ),
                "common_component_mean": float(pair_components["common_component"].mean()),
                "regime_component_mean": float(pair_components["regime_component"].mean()),
                "loop_specific_excess_mean": float(pair_components["loop_excess_component"].mean()),
                "target_pair_component_rows": len(pair_components),
                "component_population": "target_pair_rows_inside_its_frozen_pair_episodes",
                "mean_occurrence_share": float(matching_loops["occurrence_share"].mean()),
                "mean_positive_payoff_share": float(matching_loops["positive_payoff_share"].mean()),
                "median_payoff_efficiency": float(matching_loops["leader_efficiency"].median()),
                "same_regime_episode_count": int(len(set(matching_episode_ids))),
                "reference_rows_2025": len(ref),
                "exact_completion_rows": int(topology_counts.get("exact_parent_completion", 0)),
                "exact_completion_share": int(topology_counts.get("exact_parent_completion", 0))
                / topology_total,
                "incompatible_transition_rows": int(
                    topology_counts.get("incompatible_first_transition", 0)
                ),
                "incompatible_transition_share": int(
                    topology_counts.get("incompatible_first_transition", 0)
                )
                / topology_total,
                "expected_leg_diversion_rows": int(
                    topology_counts.get("expected_leg_then_diversion", 0)
                ),
                "expected_leg_diversion_share": int(
                    topology_counts.get("expected_leg_then_diversion", 0)
                )
                / topology_total,
                "route_fixed_close_payoff_by_topology": route_payoff_by_topology,
                "f0_mean_net_payoff_bps": float(f0["net_payoff_bps"].mean()),
                "f10_mean_net_payoff_bps": float(f10["net_payoff_bps"].mean()),
                "stock_concentration_top_share": float(
                    ref.groupby("symbol")["original_net_payoff_bps"].sum().clip(lower=0).max()
                    / ref.groupby("symbol")["original_net_payoff_bps"].sum().clip(lower=0).sum()
                )
                if ref.groupby("symbol")["original_net_payoff_bps"].sum().clip(lower=0).sum() > 0
                else math.nan,
                "month_count": int(ref["month"].nunique()),
                "occurrence_support_rows": occurrence_support["rows"],
                "occurrence_support_sessions": occurrence_support["independent_sessions"],
                "occurrence_support_stocks": occurrence_support["independent_stocks"],
                "occurrence_support_months": occurrence_support["months"],
                "occurrence_support_passed": occurrence_support["supported"],
                "named_family_raw_p_value": named_p_value,
                "dominant_prior_regime_sequences": "|".join(
                    f"{sequence}:{int(count)}" for sequence, count in sequence_counts.items()
                ),
                "clock_phase_composition": "|".join(
                    f"{phase}:{int(count)}" for phase, count in clock_counts.items()
                ),
                "state_age_status": "unavailable_on_primary_pair_panel",
                "recurrence_phase_status": "unavailable_on_primary_pair_panel",
                "clock_phase_status": "available_only_as_frozen_state_change_phase_summary",
            }
        )
    result = pd.DataFrame.from_records(records)
    result["named_family_holm_adjusted_p_value"] = holm_adjust(
        result["named_family_raw_p_value"].tolist()
    )
    result["named_family_comparisons_examined"] = len(result)
    result["named_family_holm_significant"] = (
        result["named_family_holm_adjusted_p_value"].le(0.05) & result["occurrence_support_passed"]
    )
    return result


def make_plots(
    output: Path,
    block: pd.DataFrame,
    null_summary: dict[str, Any],
    episodes: pd.DataFrame,
    loops: pd.DataFrame,
    timeline: pd.DataFrame,
    early: pd.DataFrame,
    four_way: pd.DataFrame,
    associations: pd.DataFrame,
    network: pd.DataFrame,
    factor: pd.DataFrame,
) -> None:
    plot_root = output / "plots"
    plot_root.mkdir()

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(plot_root / name, dpi=120, metadata={"Software": "Stocker research"})
        plt.close()

    plt.figure(figsize=(6, 4))
    observed = [
        null_summary["observed_exactly_one"],
        null_summary["observed_exactly_two"],
        null_summary["observed_exactly_three"],
        null_summary["observed_four_or_more"],
    ]
    expected = [
        block["exactly_one"].mean(),
        block["exactly_two"].mean(),
        block["exactly_three"].mean(),
        block["four_or_more"].mean(),
    ]
    x = np.arange(4)
    plt.bar(x - 0.2, observed, 0.4, label="observed")
    plt.bar(x + 0.2, expected, 0.4, label="block null")
    plt.xticks(x, ["1", "2", "3", "4+"])
    plt.legend()
    plt.title("Positive pairs per session")
    save("01_observed_vs_block_null_positive_pair_count.png")

    plt.figure(figsize=(6, 4))
    periods = ["2023", "2025"]
    shares = [null_summary["period_multi_pair_share"][period] for period in periods]
    plt.bar(periods, shares, color=["#5865f2", "#f97316"])
    plt.ylim(0, 1)
    plt.title("Multi-pair positive-session rate")
    save("02_multi_pair_rate_by_period.png")

    plt.figure(figsize=(8, 4))
    episodes["anatomy_category"].value_counts().plot.bar(color="#334155")
    plt.title("Episode anatomy categories")
    save("03_episode_anatomy_category_counts.png")

    plt.figure(figsize=(6, 4))
    episodes["final_leader_share"].dropna().hist(bins=15, color="#0f766e")
    plt.title("Leader positive-payoff share")
    save("04_leader_payoff_share_distribution.png")

    leaders = loops[loops["is_final_leader"]]
    plt.figure(figsize=(6, 4))
    plt.scatter(leaders["occurrence_share"], leaders["positive_payoff_share"], alpha=0.6)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("occurrence share")
    plt.ylabel("positive-payoff share")
    save("05_leader_occurrence_vs_payoff_share.png")

    plt.figure(figsize=(6, 4))
    leaders["leader_efficiency"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=15)
    plt.axvline(1.0, color="black", linestyle="--")
    plt.title("Leader efficiency")
    save("06_leader_efficiency_distribution.png")

    plt.figure(figsize=(8, 4))
    shared = timeline[timeline["episode_level"].eq("shared_market")]
    component_timeline = shared.groupby("relative_session_to_onset")[
        [
            "common_component",
            "dominant_regime_component",
            "dominant_loop_specific_excess",
        ]
    ].median()
    component_timeline.plot(ax=plt.gca())
    plt.title("Median component timeline")
    save("07_component_timeline.png")

    plt.figure(figsize=(7, 4))
    early.groupby("checkpoint")["top_one_match"].mean().reindex(
        ["first_session", "first_two", "first_three", "first_25pct", "first_50pct"]
    ).plot.bar()
    plt.ylim(0, 1)
    plt.title("Provisional leader top-one match")
    save("08_early_vs_final_leader_accuracy.png")

    plt.figure(figsize=(7, 4))
    early.groupby("checkpoint")["fraction_final_payoff_remaining"].median().plot.bar()
    plt.ylim(0, 1)
    plt.title("Episode payoff remaining")
    save("09_payoff_remaining_by_checkpoint.png")

    plt.figure(figsize=(8, 4))
    timeline.groupby("relative_session_to_onset")[
        [
            "loop_payoff_entropy",
            "regime_payoff_entropy",
        ]
    ].median().plot(ax=plt.gca())
    plt.title("Entropy around onset")
    save("10_entropy_around_episode_onset.png")

    for index, loop in [(11, "cycle_04"), (12, "cycle_07")]:
        plt.figure(figsize=(8, 4))
        named_ids = loops.loc[loops["loop"].eq(loop), "episode_id"].unique()
        named = timeline[timeline["episode_id"].isin(named_ids)]
        named.groupby("relative_session_to_onset")[
            [
                "common_component",
                "dominant_loop_specific_excess",
            ]
        ].median().plot(ax=plt.gca())
        plt.title(f"{loop} anatomy timeline")
        save(f"{index:02d}_{loop}_anatomy_timeline.png")

    plt.figure(figsize=(8, 4))
    supported = four_way[four_way["supported"]]
    counterfactual_plot = supported.groupby("counterfactual_group")["mean_net_payoff_bps"].mean()
    if len(counterfactual_plot):
        counterfactual_plot.plot.bar()
    else:
        plt.text(0.5, 0.5, "No supported counterfactual groups", ha="center", va="center")
        plt.xticks([])
        plt.yticks([])
    plt.title("Prior-sequence four-way counterfactual")
    save("13_prior_regime_sequence_counterfactual.png")

    plt.figure(figsize=(9, 5))
    pivot = associations[associations["period"].eq("all")].pivot(
        index="indicator", columns="component", values="spearman_rank_correlation"
    )
    plt.imshow(pivot.fillna(0), aspect="auto", cmap="coolwarm", vmin=-0.3, vmax=0.3)
    plt.yticks(range(len(pivot)), [str(value) for value in pivot.index])
    plt.xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns], rotation=20)
    plt.colorbar(label="rank correlation")
    plt.title("Indicator association by component")
    save("14_indicator_effect_by_component.png")

    plt.figure(figsize=(8, 8))
    display = network[network["display_edge"]]
    nodes = sorted(set(display["pair_left"]) | set(display["pair_right"]))
    angles = np.linspace(0, 2 * np.pi, max(len(nodes), 1), endpoint=False)
    positions = {
        node: (math.cos(angle), math.sin(angle)) for node, angle in zip(nodes, angles, strict=False)
    }
    for row in display.itertuples():
        left = positions[row.pair_left]
        right = positions[row.pair_right]
        width = 0.5 + min(max(float(cast(Any, row.excess_coactivations)), 0.0), 10.0) / 4.0
        plt.plot([left[0], right[0]], [left[1], right[1]], alpha=0.4, linewidth=width)
    left_node = display[
        [
            "pair_left",
            "node_total_positive_payoff_left",
            "node_regime_left",
            "node_leader_frequency_left",
        ]
    ].rename(
        columns={
            "pair_left": "pair",
            "node_total_positive_payoff_left": "total_positive_payoff",
            "node_regime_left": "regime",
            "node_leader_frequency_left": "leader_frequency",
        }
    )
    right_node = display[
        [
            "pair_right",
            "node_total_positive_payoff_right",
            "node_regime_right",
            "node_leader_frequency_right",
        ]
    ].rename(
        columns={
            "pair_right": "pair",
            "node_total_positive_payoff_right": "total_positive_payoff",
            "node_regime_right": "regime",
            "node_leader_frequency_right": "leader_frequency",
        }
    )
    node_metadata = pd.concat([left_node, right_node], ignore_index=True).drop_duplicates("pair")
    node_metadata = node_metadata.set_index("pair")
    maximum_node_payoff = finite_mean(pd.Series([node_metadata["total_positive_payoff"].max()]))
    regime_values = sorted(node_metadata["regime"].dropna().astype(str).unique())
    colour_map = {
        regime: plt.get_cmap("tab20")(index % 20) for index, regime in enumerate(regime_values)
    }
    for node, position in positions.items():
        metadata = node_metadata.loc[node]
        payoff = float(metadata["total_positive_payoff"])
        size = (
            60.0 + 440.0 * math.sqrt(max(payoff, 0.0) / maximum_node_payoff)
            if maximum_node_payoff > 0
            else 60.0
        )
        regime = str(metadata["regime"])
        leader_frequency = float(metadata["leader_frequency"])
        plt.scatter(*position, s=size, color=colour_map.get(regime, "grey"), alpha=0.8)
        label = (
            f"{node}\nleader={leader_frequency:.2f}" if math.isfinite(leader_frequency) else node
        )
        plt.text(position[0], position[1], label, fontsize=6, ha="center", va="center")
    plt.axis("off")
    plt.title("Null-adjusted co-activation network")
    save("15_coactivation_network.png")

    plt.figure(figsize=(7, 4))
    for period, group in factor.groupby("period"):
        plt.plot(
            group["factor_count"], group["cumulative_variance_explained"], marker="o", label=period
        )
    plt.xticks([1, 2, 3])
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Fixed common-factor variance diagnostic")
    save("16_common_factor_variance_explanation.png")


def run(output: Path, exact_rerun_of: Path | None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    contract, paths, hashes, snapshot = verify_contract()
    contract_hash = sha256(CONTRACT_PATH)
    write_json(output / "frozen_experiment_contract.json", contract)
    source_manifest = {
        "run_id": RUN_ID,
        "contract_id": contract["contract_id"],
        "contract_hash": contract_hash,
        "data_snapshot_id": snapshot,
        "sources": {
            name: {"path": str(path.relative_to(REPO)), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "missing_originals": contract["inputs"]["unavailable_originals"],
    }
    write_json(output / "source_identity_manifest.json", source_manifest)

    states = pd.read_parquet(paths["v2_hindsight_episode_states.parquet"])
    source_pair_episodes = pd.read_parquet(paths["v2_hindsight_episode_diagnostics.parquet"])
    features = pd.read_parquet(paths["v2_causal_feature_panel.parquet"])
    payoff = pd.read_parquet(paths["v2_session_payoff_panel.parquet"])
    trades = pd.read_parquet(paths["v2_trade_decisions.parquet"])
    calendar = features[["period", "score_session"]].drop_duplicates()
    census = reproduce_exploratory_census(states, source_pair_episodes, calendar)
    gate = contract["positive_pair"]["census_gate"]
    gate_actual = {
        "positive_sessions": census["positive_sessions"],
        "multi_pair_positive_sessions": census["multi_pair_positive_sessions"],
        "same_regime_episodes": census["same_regime_episodes"],
        "single_loop_same_regime_episodes": census["single_loop_same_regime_episodes"],
        "multi_loop_same_regime_episodes": census["multi_loop_same_regime_episodes"],
    }
    if gate_actual != gate:
        write_json(output / "reproduced_exploratory_census.json", {**census, "gate_passed": False})
        raise AssertionError(f"exploratory census gate failed: {gate_actual} != {gate}")
    census["gate_passed"] = True
    census["run_id"] = RUN_ID
    census["source_artifact"] = (
        "v2_hindsight_episode_states.parquet|v2_hindsight_episode_diagnostics.parquet"
    )
    census["source_hash"] = (
        f"{hashes['v2_hindsight_episode_states.parquet']}|{hashes['v2_hindsight_episode_diagnostics.parquet']}"
    )
    write_json(output / "reproduced_exploratory_census.json", census)

    occurrences = stock_occurrence_ledger(trades)
    occurrence_contract = contract["occurrence"]
    raw_occurrences = int(occurrences["raw_fill_count"].sum())
    stock_capped_occurrences = int(len(occurrences))
    if raw_occurrences != int(occurrence_contract["expected_raw_fills"]):
        raise AssertionError("frozen raw occurrence count changed")
    if stock_capped_occurrences != int(occurrence_contract["expected_stock_capped_occurrences"]):
        raise AssertionError("frozen stock-capped occurrence identity changed")
    if occurrences.duplicated(["period", "session", "loop", "regime", "stock"]).any():
        raise AssertionError("stock occurrence identity is not unique")
    collapsed_source = trades[
        trades["model_name"].eq("no_payoff_state_filter") & trades["status"].eq("filled")
    ][
        [
            "period",
            "score_session",
            "loop_id",
            "orientation",
            "stock_id",
            "gross_payoff_bps",
            "primary_total_cost_bps",
            "primary_net_payoff_bps",
        ]
    ].copy()
    collapsed = collapse_stock_contributions(
        collapsed_source.rename(
            columns={
                "score_session": "session",
                "loop_id": "loop",
                "stock_id": "stock",
                "primary_total_cost_bps": "cost_bps",
                "primary_net_payoff_bps": "net_payoff_bps",
            }
        )
    )
    panel = build_synchronized_panel(features, payoff, states)
    panel = add_history_to_panel(panel, occurrences)
    panel = decompose_payoff_components(panel)
    pair_ledger, same_ledger, shared_ledger = build_episode_ledgers(source_pair_episodes, calendar)
    pair_ledger["session"] = (
        pair_ledger["onset"].astype(str) + ".." + pair_ledger["end"].astype(str)
    )
    panel = attach_episode_membership(
        panel,
        pair_ledger,
        same_ledger,
        shared_ledger,
        calendar,
    )
    aggregation_check = collapsed.merge(
        payoff.rename(
            columns={
                "score_session": "session",
                "loop_id": "loop",
            }
        )[
            [
                "period",
                "session",
                "loop",
                "orientation",
                "robust_net_payoff_bps",
            ]
        ],
        on=["period", "session", "loop", "orientation"],
        how="outer",
        suffixes=("_rebuilt", "_source"),
        indicator=True,
    )
    if aggregation_check["_merge"].eq("left_only").any():
        raise AssertionError("rebuilt stock-collapsed cells are absent from source payoff panel")
    aggregation_max_error = float(
        (
            aggregation_check["robust_net_payoff_bps_rebuilt"]
            - aggregation_check["robust_net_payoff_bps_source"]
        )
        .abs()
        .max()
    )
    if aggregation_max_error > 1e-10:
        raise AssertionError(
            f"stock-collapsed payoff reconstruction drift: {aggregation_max_error}"
        )
    panel["named_control_identity"] = panel["pair"].map(NAMED_PAIRS).fillna("other")
    panel["data_availability_flags"] = np.where(
        panel["robust_net_payoff_bps"].notna(), "payoff_available", "no_opportunity_missing"
    )
    panel = write_frame(
        output,
        "session_regime_loop_orientation_panel.parquet",
        panel,
        source_artifact="v2_causal_feature_panel.parquet|v2_session_payoff_panel.parquet|v2_hindsight_episode_states.parquet",
        source_hash=f"{hashes['v2_causal_feature_panel.parquet']}|{hashes['v2_session_payoff_panel.parquet']}|{hashes['v2_hindsight_episode_states.parquet']}",
    )
    robust_panel = panel[panel["robust_net_payoff_bps"].notna()].copy()
    write_frame(
        output,
        "robust_pair_payoff_panel.parquet",
        robust_panel,
        source_artifact="v2_session_payoff_panel.parquet",
        source_hash=hashes["v2_session_payoff_panel.parquet"],
        component="robust_equal_stock_payoff",
    )

    calendars, _ = calendar_lookup(calendar)
    same_metrics, same_loops, same_session_loops = episode_metrics(
        same_ledger, pair_ledger, panel, occurrences, calendars
    )
    shared_metrics, shared_loops, shared_session_loops = episode_metrics(
        shared_ledger, pair_ledger, panel, occurrences, calendars
    )
    all_episode_metrics = pd.concat([same_metrics, shared_metrics], ignore_index=True)
    all_episode_loops = pd.concat([same_loops, shared_loops], ignore_index=True)
    all_session_loops = pd.concat([same_session_loops, shared_session_loops], ignore_index=True)
    occurrence_identity = all_episode_loops["payoff_occurrence_identity_error"].dropna().abs()
    if len(occurrence_identity) and float(occurrence_identity.max()) > 1e-10:
        raise AssertionError("total loop payoff != occurrence count * mean payoff per occurrence")
    write_frame(
        output,
        "pair_positive_episode_ledger.parquet",
        pair_ledger,
        source_artifact="v2_hindsight_episode_diagnostics.parquet",
        source_hash=hashes["v2_hindsight_episode_diagnostics.parquet"],
    )
    write_frame(
        output,
        "same_regime_episode_ledger.parquet",
        same_metrics,
        source_artifact="v2_hindsight_episode_diagnostics.parquet",
        source_hash=hashes["v2_hindsight_episode_diagnostics.parquet"],
    )
    write_frame(
        output,
        "shared_market_episode_ledger.parquet",
        shared_metrics,
        source_artifact="v2_hindsight_episode_diagnostics.parquet",
        source_hash=hashes["v2_hindsight_episode_diagnostics.parquet"],
    )

    timeline = episode_timelines(
        all_episode_metrics, pair_ledger, panel, all_episode_loops, occurrences, calendars
    )
    write_frame(
        output,
        "episode_session_timeline.parquet",
        timeline,
        source_artifact="derived_episode_ledgers|session_regime_loop_orientation_panel.parquet",
        source_hash=f"{hashes['v2_hindsight_episode_diagnostics.parquet']}|{hashes['v2_session_payoff_panel.parquet']}",
    )
    timeline_subsets = {
        "timeline_all_shared_episodes.parquet": timeline["episode_level"].eq("shared_market"),
        "timeline_same_regime_episodes.parquet": timeline["episode_level"].eq("same_regime"),
        "timeline_named_pair_episodes.parquet": timeline["episode_id"].isin(
            all_episode_loops.loc[
                all_episode_loops["loop"].isin(["cycle_04", "cycle_07"]), "episode_id"
            ]
        ),
        "timeline_single_loop_episodes.parquet": timeline["anatomy_category"].eq(
            "SINGLE_LOOP_EPISODE"
        ),
        "timeline_majority_dominant_episodes.parquet": timeline["anatomy_category"].isin(
            ["MAJORITY_LOOP_DOMINANCE", "EXTREME_LOOP_DOMINANCE"]
        ),
        "timeline_diffuse_episodes.parquet": timeline["anatomy_category"].eq("DIFFUSE_MULTI_LOOP"),
    }
    for name, mask in timeline_subsets.items():
        write_frame(
            output,
            name,
            timeline.loc[mask].copy(),
            source_artifact="episode_session_timeline.parquet",
            source_hash=hashes["v2_hindsight_episode_diagnostics.parquet"],
        )

    eligibility = (
        panel.groupby(["period", "pair", "loop", "orientation", "regime"], observed=True)
        .agg(
            calendar_cells=("session", "size"),
            payoff_available_cells=("robust_net_payoff_bps", "count"),
            positive_label_available_cells=("positive_pair_available", "sum"),
            positive_cells=("positive_pair_flag", "sum"),
        )
        .reset_index()
    )
    eligibility["missing_payoff_cells"] = (
        eligibility["calendar_cells"] - eligibility["payoff_available_cells"]
    )
    write_frame(
        output,
        "pair_eligibility_and_missingness.csv",
        eligibility,
        source_artifact="v2_causal_feature_panel.parquet|v2_hindsight_episode_states.parquet",
        source_hash=f"{hashes['v2_causal_feature_panel.parquet']}|{hashes['v2_hindsight_episode_states.parquet']}",
    )
    positive_rate = eligibility.copy()
    positive_rate["pair_specific_positive_rate"] = positive_rate["positive_cells"] / positive_rate[
        "positive_label_available_cells"
    ].replace(0, np.nan)
    write_frame(
        output,
        "pair_specific_positive_rate.csv",
        positive_rate,
        source_artifact="v2_hindsight_episode_states.parquet",
        source_hash=hashes["v2_hindsight_episode_states.parquet"],
    )

    null_rows = states[
        ["period", "score_session", "loop_id", "orientation", "hindsight_payoff_state"]
    ].copy()
    null_rows["period"] = null_rows["period"].astype(str)
    null_rows["session"] = pd.to_datetime(null_rows["score_session"]).dt.strftime("%Y-%m-%d")
    null_rows["pair"] = (
        null_rows["loop_id"].astype(str) + "|" + null_rows["orientation"].astype(str)
    )
    null_rows["eligible"] = True
    null_rows["positive_pair_flag"] = null_rows["hindsight_payoff_state"].eq("positive")
    poisson = poisson_binomial_null(null_rows)
    write_frame(
        output,
        "poisson_binomial_null_results.parquet",
        poisson,
        source_artifact="v2_hindsight_episode_states.parquet",
        source_hash=hashes["v2_hindsight_episode_states.parquet"],
        component="null_pair_specific_independence",
    )
    block, block_summary = block_circular_pair_shift(
        null_rows,
        resamples=int(contract["nulls"]["block_circular_pair_shift"]["resamples"]),
        seed=int(contract["nulls"]["block_circular_pair_shift"]["seed"]),
        block_length=int(contract["nulls"]["block_circular_pair_shift"]["block_length_sessions"]),
        retain_shifted_rows=False,
    )
    observed_counts = null_rows.groupby(["period", "session"])["positive_pair_flag"].sum()
    block_summary.update(
        {
            "observed_exactly_one": int(observed_counts.eq(1).sum()),
            "observed_exactly_two": int(observed_counts.eq(2).sum()),
            "observed_exactly_three": int(observed_counts.eq(3).sum()),
            "observed_four_or_more": int(observed_counts.ge(4).sum()),
            "observed_maximum_positive_pairs": int(observed_counts.max()),
            "period_multi_pair_share": {
                period: float(group[group.gt(0)].ge(2).mean())
                for period, group in observed_counts.groupby(level="period")
            },
        }
    )
    write_frame(
        output,
        "block_circular_null_results.parquet",
        block,
        source_artifact="v2_hindsight_episode_states.parquet",
        source_hash=hashes["v2_hindsight_episode_states.parquet"],
        component="null_block_circular_pair_shift",
    )
    write_json(
        output / "block_circular_null_summary.json",
        {
            **block_summary,
            "run_id": RUN_ID,
            "source_hash": hashes["v2_hindsight_episode_states.parquet"],
        },
    )
    network = block_network(null_rows, resamples=2000, seed=20260717, block_length=5)
    network = add_network_node_metadata(network, panel, same_loops)
    write_frame(
        output,
        "coactivation_excess_table.parquet",
        network,
        source_artifact="v2_hindsight_episode_states.parquet",
        source_hash=hashes["v2_hindsight_episode_states.parquet"],
        component="null_adjusted_coactivation",
    )
    write_frame(
        output,
        "coactivation_network_edge_table.csv",
        network,
        source_artifact="v2_hindsight_episode_states.parquet",
        source_hash=hashes["v2_hindsight_episode_states.parquet"],
        component="null_adjusted_coactivation",
    )

    common = panel.loc[panel["robust_net_payoff_bps"].notna()].drop_duplicates(
        ["period", "session"]
    )[["period", "session", "common_component"]]
    regime = panel.loc[panel["robust_net_payoff_bps"].notna()].drop_duplicates(
        ["period", "session", "regime"]
    )[["period", "session", "regime", "orientation", "regime_component"]]
    loop_excess = panel.loc[
        panel["robust_net_payoff_bps"].notna(),
        ["period", "session", "pair", "loop", "orientation", "regime", "loop_excess_component"],
    ]
    write_frame(
        output,
        "common_component_table.parquet",
        common,
        source_artifact="robust_pair_payoff_panel.parquet",
        source_hash=hashes["v2_session_payoff_panel.parquet"],
        component="common",
    )
    write_frame(
        output,
        "regime_component_table.parquet",
        regime,
        source_artifact="robust_pair_payoff_panel.parquet",
        source_hash=hashes["v2_session_payoff_panel.parquet"],
        component="regime",
    )
    write_frame(
        output,
        "loop_specific_excess_table.parquet",
        loop_excess,
        source_artifact="robust_pair_payoff_panel.parquet",
        source_hash=hashes["v2_session_payoff_panel.parquet"],
        component="loop_excess",
    )
    component_episode_columns = [
        "period",
        "session",
        "episode_id",
        "pair",
        "loop",
        "orientation",
        "regime",
        "episode_level",
        "common_component_positive_pair_share",
        "regime_component_positive_pair_share",
        "loop_excess_component_positive_pair_share",
        "common_positive_contribution_share",
        "regime_positive_contribution_share",
        "loop_excess_positive_contribution_share",
        "common_signed_contribution_to_positive_pair_payoff",
        "regime_signed_contribution_to_positive_pair_payoff",
        "loop_excess_signed_contribution_to_positive_pair_payoff",
        "common_marginal_variance_share_noncausal",
        "regime_marginal_variance_share_noncausal",
        "loop_excess_marginal_variance_share_noncausal",
        "variance_decomposition_is_causal",
        "common_first_positive_relative_session",
        "common_last_positive_relative_session",
        "regime_first_positive_relative_session",
        "regime_last_positive_relative_session",
        "loop_excess_first_positive_relative_session",
        "loop_excess_last_positive_relative_session",
        "common_onset_value",
        "common_decay_value",
        "regime_onset_value",
        "regime_decay_value",
        "loop_excess_onset_value",
        "loop_excess_decay_value",
    ]
    write_frame(
        output,
        "component_episode_attribution.parquet",
        all_episode_metrics[component_episode_columns],
        source_artifact="robust_pair_payoff_panel.parquet|derived_episode_ledgers",
        source_hash=f"{hashes['v2_session_payoff_panel.parquet']}|{hashes['v2_hindsight_episode_diagnostics.parquet']}",
        component="episode_component_attribution_noncausal",
    )
    sensitivity = component_sensitivity(panel)
    write_frame(
        output,
        "component_sensitivity_results.parquet",
        sensitivity,
        source_artifact="robust_pair_payoff_panel.parquet",
        source_hash=hashes["v2_session_payoff_panel.parquet"],
        component="winsorised_mean_sensitivity",
    )

    write_frame(
        output,
        "occurrence_share_table.parquet",
        all_episode_loops,
        source_artifact="v2_trade_decisions.parquet|v2_hindsight_episode_diagnostics.parquet",
        source_hash=f"{hashes['v2_trade_decisions.parquet']}|{hashes['v2_hindsight_episode_diagnostics.parquet']}",
        component="occurrence",
    )
    write_frame(
        output,
        "payoff_share_table.parquet",
        all_episode_loops,
        source_artifact="v2_trade_decisions.parquet|v2_hindsight_episode_diagnostics.parquet",
        source_hash=f"{hashes['v2_trade_decisions.parquet']}|{hashes['v2_hindsight_episode_diagnostics.parquet']}",
        component="positive_payoff",
    )
    write_frame(
        output,
        "leader_efficiency_table.parquet",
        all_episode_loops[all_episode_loops["is_final_leader"]],
        source_artifact="v2_trade_decisions.parquet|v2_hindsight_episode_diagnostics.parquet",
        source_hash=f"{hashes['v2_trade_decisions.parquet']}|{hashes['v2_hindsight_episode_diagnostics.parquet']}",
        component="leader_efficiency",
    )
    write_frame(
        output,
        "episode_anatomy_classification.csv",
        all_episode_metrics,
        source_artifact="derived_episode_ledgers",
        source_hash=hashes["v2_hindsight_episode_diagnostics.parquet"],
        component="episode_anatomy",
    )

    early = early_leader_checkpoints(all_session_loops)
    early = early.merge(
        all_episode_metrics[
            [
                "episode_id",
                "period",
                "episode_level",
                "session",
                "pair",
                "loop",
                "orientation",
                "regime",
                "occurrence_coverage_fraction",
                "occurrence_population_complete",
            ]
        ],
        on="episode_id",
        how="left",
        validate="many_to_one",
    )
    early.loc[
        ~early["occurrence_population_complete"].fillna(False),
        ["provisional_leader_occurrence_share", "provisional_leader_efficiency"],
    ] = math.nan
    write_frame(
        output,
        "early_leader_table.parquet",
        early,
        source_artifact="robust_pair_payoff_panel.parquet|derived_episode_ledgers",
        source_hash=f"{hashes['v2_session_payoff_panel.parquet']}|{hashes['v2_hindsight_episode_diagnostics.parquet']}",
        component="early_leader",
    )
    persistence = leader_persistence(all_session_loops)
    persistence = persistence.merge(
        all_episode_metrics[
            ["episode_id", "period", "episode_level", "pair", "loop", "orientation", "regime"]
        ],
        on="episode_id",
        how="left",
        validate="many_to_one",
    )
    write_frame(
        output,
        "leader_persistence_table.parquet",
        persistence,
        source_artifact="robust_pair_payoff_panel.parquet|derived_episode_ledgers",
        source_hash=f"{hashes['v2_session_payoff_panel.parquet']}|{hashes['v2_hindsight_episode_diagnostics.parquet']}",
        component="leader_persistence",
    )
    component_persistence_result = component_persistence(panel)
    write_frame(
        output,
        "component_persistence_table.csv",
        component_persistence_result,
        source_artifact="session_regime_loop_orientation_panel.parquet",
        source_hash=hashes["v2_session_payoff_panel.parquet"],
        component="component_persistence",
    )
    write_frame(
        output,
        "capturable_payoff_by_checkpoint.csv",
        early[
            [
                "period",
                "session",
                "episode_id",
                "episode_level",
                "pair",
                "loop",
                "orientation",
                "regime",
                "checkpoint",
                "payoff_remaining",
                "fraction_final_payoff_remaining",
                "final_leader_payoff_remaining",
                "fraction_final_leader_payoff_remaining",
            ]
        ],
        source_artifact="early_leader_table.parquet",
        source_hash=hashes["v2_session_payoff_panel.parquet"],
        component="payoff_remaining",
    )

    sequence_census, four_way, sequence_increments, sequence_stability = sequence_analysis(
        occurrences, panel
    )
    write_frame(
        output,
        "regime_sequence_census.parquet",
        sequence_census,
        source_artifact="v2_trade_decisions.parquet",
        source_hash=hashes["v2_trade_decisions.parquet"],
        component="regime_sequence",
    )
    write_frame(
        output,
        "four_way_counterfactual_tables.parquet",
        four_way,
        source_artifact="v2_trade_decisions.parquet|robust_pair_payoff_panel.parquet",
        source_hash=f"{hashes['v2_trade_decisions.parquet']}|{hashes['v2_session_payoff_panel.parquet']}",
        component="four_way_counterfactual",
    )
    write_frame(
        output,
        "sequence_increment_table.parquet",
        sequence_increments,
        source_artifact="v2_trade_decisions.parquet",
        source_hash=hashes["v2_trade_decisions.parquet"],
        component="sequence_increment",
    )
    write_frame(
        output,
        "loop_increment_within_sequence_table.parquet",
        sequence_increments[
            [
                "period",
                "pair",
                "loop",
                "regime",
                "sequence_length",
                "sequence",
                "loop_increment_within_sequence_bps",
                "loop_increment_p_value",
                "loop_increment_fdr_q_value",
                "increment_comparison_supported",
                "both_directionally_positive",
            ]
        ],
        source_artifact="v2_trade_decisions.parquet",
        source_hash=hashes["v2_trade_decisions.parquet"],
        component="loop_increment_within_sequence",
    )
    write_frame(
        output,
        "sequence_interaction_stability.parquet",
        sequence_stability,
        source_artifact="sequence_increment_table.parquet",
        source_hash=hashes["v2_trade_decisions.parquet"],
        component="sequence_interaction_stability",
    )

    route = route_tables(
        pd.read_parquet(paths["causal_route_signal_path_events.parquet"]),
        panel,
        pair_ledger,
        calendars,
    )
    named = named_deep_dive(
        pair_ledger,
        same_metrics,
        same_loops,
        route,
        pd.read_parquet(paths["t0_historical_named_reference_ledger.parquet"]),
        pd.read_parquet(paths["t0_historical_control_reference_ledger.parquet"]),
        pd.read_parquet(paths["t0_payoff_envelope_ledger.parquet"]),
        occurrences,
        panel,
    )
    for index, named_row in named.iterrows():
        matching_ids: list[str] = []
        for _, episode in same_metrics.iterrows():
            members = episode_members(pair_ledger, episode)
            if named_row["pair"] in set(members["pair"]):
                matching_ids.append(str(episode["episode_id"]))
        named_early = early[
            early["episode_id"].isin(matching_ids) & early["checkpoint"].eq("first_session")
        ]
        named.loc[index, "early_first_session_top_one_match_rate"] = float(
            named_early["top_one_match"].mean()
        )
        named.loc[index, "early_first_session_median_payoff_remaining"] = float(
            named_early["fraction_final_payoff_remaining"].median()
        )
    write_frame(
        output,
        "named_pair_deep_dive_tables.parquet",
        named,
        source_artifact="multiple_frozen_named_and_episode_ledgers",
        source_hash=snapshot,
        component="named_pair_anatomy",
    )
    write_frame(
        output,
        "route_topology_anatomy.parquet",
        route,
        source_artifact="causal_route_signal_path_events.parquet",
        source_hash=hashes["causal_route_signal_path_events.parquet"],
        component="route_outcome_only",
    )
    deterioration = path_deterioration(
        pd.read_parquet(paths["sequential_path_diagnostics.parquet"]),
        trades,
        pair_ledger,
        same_metrics,
        panel,
        calendars,
    )
    write_frame(
        output,
        "sequential_path_deterioration.parquet",
        deterioration,
        source_artifact="sequential_path_diagnostics.parquet",
        source_hash=hashes["sequential_path_diagnostics.parquet"],
        component="path_outcome_only",
    )

    manifestation = indicator_manifestation_table(
        panel, all_episode_metrics, pair_ledger, calendars, occurrences
    )
    associations = indicator_associations(panel, occurrences)
    write_frame(
        output,
        "indicator_manifestation_tables.parquet",
        manifestation,
        source_artifact="episode_session_timeline.parquet",
        source_hash=hashes["v2_causal_feature_panel.parquet"],
        component="indicator_manifestation",
    )
    write_frame(
        output,
        "component_specific_indicator_association.parquet",
        associations,
        source_artifact="v2_causal_feature_panel.parquet|robust_pair_payoff_panel.parquet",
        source_hash=f"{hashes['v2_causal_feature_panel.parquet']}|{hashes['v2_session_payoff_panel.parquet']}",
        component="indicator_association",
    )
    factor, factor_loadings = enriched_factor_diagnostic(panel)
    write_frame(
        output,
        "common_factor_diagnostic.csv",
        factor,
        source_artifact="robust_pair_payoff_panel.parquet",
        source_hash=hashes["v2_session_payoff_panel.parquet"],
        component="secondary_pca",
    )
    write_frame(
        output,
        "common_factor_loadings.parquet",
        factor_loadings,
        source_artifact="robust_pair_payoff_panel.parquet",
        source_hash=hashes["v2_session_payoff_panel.parquet"],
        component="secondary_pca_loadings",
    )

    concentration_rows: list[pd.DataFrame] = []
    for _, episode in all_episode_metrics.iterrows():
        members = episode_members(pair_ledger, episode)
        selected = occurrences[
            occurrences["period"].astype(str).eq(str(episode["period"]))
            & occurrences["session"].between(str(episode["onset"]), str(episode["end"]))
            & occurrences["pair"].isin(set(members["pair"]))
        ][["stock", "positive_payoff"]].copy()
        selected["episode_id"] = episode["episode_id"]
        selected["sector"] = pd.NA
        concentration_rows.append(selected)
    concentration_input = pd.concat(concentration_rows, ignore_index=True)
    concentration = concentration_attribution(concentration_input).merge(
        all_episode_metrics[
            [
                "episode_id",
                "period",
                "episode_level",
                "session",
                "pair",
                "loop",
                "orientation",
                "regime",
                "occurrence_coverage_fraction",
                "occurrence_population_complete",
            ]
        ],
        on="episode_id",
        how="left",
        validate="one_to_one",
    )
    concentration["liquidity_cohort_status"] = "source_volume_bucket_available_in_occurrence_ledger"
    concentration["volatility_cohort_status"] = "unavailable_without_post_registration_binning"
    concentration["beta_cohort_status"] = "unavailable"
    write_frame(
        output,
        "stock_and_cohort_concentration.parquet",
        concentration,
        source_artifact="v2_trade_decisions.parquet|derived_episode_ledgers",
        source_hash=f"{hashes['v2_trade_decisions.parquet']}|{hashes['v2_hindsight_episode_diagnostics.parquet']}",
        component="concentration",
    )
    cohort_contributions, leave_one_stock_out = cohort_contribution_tables(
        all_episode_metrics, pair_ledger, occurrences, panel, calendars
    )
    write_frame(
        output,
        "cohort_contribution_table.parquet",
        cohort_contributions,
        source_artifact="v2_trade_decisions.parquet|derived_episode_ledgers",
        source_hash=f"{hashes['v2_trade_decisions.parquet']}|{hashes['v2_hindsight_episode_diagnostics.parquet']}",
        component="cohort_contribution",
    )
    write_frame(
        output,
        "leave_one_stock_out_attribution.parquet",
        leave_one_stock_out,
        source_artifact="v2_trade_decisions.parquet|derived_episode_ledgers",
        source_hash=f"{hashes['v2_trade_decisions.parquet']}|{hashes['v2_hindsight_episode_diagnostics.parquet']}",
        component="row_deletion_attribution_not_retrained_model",
    )

    missing_rows = pd.DataFrame(
        [
            {
                "field_or_source": item,
                "status": "unavailable",
                "handling": "kept unavailable; no approximation",
            }
            for item in contract["inputs"]["unavailable_originals"]
        ]
        + [
            {
                "field_or_source": "regime_history_4",
                "status": "unavailable",
                "handling": "length-four sensitivity not run",
            },
            {
                "field_or_source": "primary-panel state_age",
                "status": "unavailable",
                "handling": "post-anchor admission age rejected",
            },
            {
                "field_or_source": "completed prior dwell",
                "status": "unavailable",
                "handling": "not joined from unmatched Atlas population",
            },
            {
                "field_or_source": "same-orientation repeat count",
                "status": "unavailable",
                "handling": "not joined from unmatched Atlas population",
            },
            {
                "field_or_source": "VWAP and typical-price distance",
                "status": "unavailable",
                "handling": "not present on matched 2023/2025 h24 panel",
            },
            {
                "field_or_source": "opening-range position",
                "status": "unavailable",
                "handling": "not present on matched 2023/2025 h24 panel",
            },
            {
                "field_or_source": "sector and beta cohorts",
                "status": "unavailable",
                "handling": "sector source is literal unavailable; no name inference",
            },
            {
                "field_or_source": "raw one-to-three-bar sequential path score",
                "status": "unavailable",
                "handling": (
                    "retained diagnostic has path classes and outcome tails, not the raw score"
                ),
            },
            {
                "field_or_source": "mixed prior-state history within stock occurrence",
                "status": "partially unavailable",
                "handling": (
                    f"{int(occurrences['history_mixed_within_occurrence'].sum())} of "
                    f"{len(occurrences)} stock-capped occurrences fail closed to unavailable "
                    "sequence context"
                ),
            },
        ]
    )
    write_frame(
        output,
        "missing_data_and_blocker_report.csv",
        missing_rows,
        source_artifact="source_identity_manifest.json",
        source_hash=snapshot,
        component="missingness",
    )

    same_categories = same_metrics["anatomy_category"].value_counts().to_dict()
    leader_rows = all_episode_loops[
        all_episode_loops["is_final_leader"] & all_episode_loops["leader_efficiency"].notna()
    ]
    early_summary_by_checkpoint = (
        early.groupby("checkpoint")
        .agg(
            episodes=("episode_id", "size"),
            top_one_match_rate=("top_one_match", "mean"),
            top_three_rate=("top_three_inclusion", "mean"),
            median_fraction_payoff_remaining=("fraction_final_payoff_remaining", "median"),
            median_fraction_leader_payoff_remaining=(
                "fraction_final_leader_payoff_remaining",
                "median",
            ),
        )
        .reset_index()
    )
    supported_interactions = sequence_stability[
        sequence_stability["multiplicity_controlled_interaction_both_periods"]
    ]
    multi_episode_ids = same_metrics.loc[
        same_metrics["loop_count"].gt(1) & same_metrics["final_leader_share"].notna(),
        "episode_id",
    ]
    multi_leaders = same_loops[
        same_loops["episode_id"].isin(multi_episode_ids)
        & same_loops["is_final_leader"]
        & same_loops["leader_efficiency"].notna()
    ]
    multi_early_summary = (
        early[early["episode_id"].isin(multi_episode_ids)]
        .groupby("checkpoint")
        .agg(
            episodes=("episode_id", "size"),
            top_one_match_rate=("top_one_match", "mean"),
            top_three_rate=("top_three_inclusion", "mean"),
            median_fraction_payoff_remaining=("fraction_final_payoff_remaining", "median"),
            median_fraction_leader_payoff_remaining=(
                "fraction_final_leader_payoff_remaining",
                "median",
            ),
        )
        .reset_index()
    )
    summary = {
        "run_id": RUN_ID,
        "census_gate_passed": True,
        "stock_collapsed_payoff_max_reconstruction_error_bps": aggregation_max_error,
        "raw_fill_occurrences": raw_occurrences,
        "stock_capped_occurrences": stock_capped_occurrences,
        "mixed_history_stock_occurrences": int(
            occurrences["history_mixed_within_occurrence"].sum()
        ),
        "source_payoff_cells_before_trade_decision_warmup": int(
            aggregation_check["_merge"].eq("right_only").sum()
        ),
        "positive_sessions": census["positive_sessions"],
        "multi_pair_positive_sessions": census["multi_pair_positive_sessions"],
        "observed_multi_pair_share": census["multi_pair_positive_session_share"],
        "block_null_multi_pair_share": block_summary[
            "block_null_mean_multi_pair_positive_session_share"
        ],
        "observed_minus_block_null_share": block_summary["observed_minus_null_share"],
        "block_null_one_sided_p": block_summary["one_sided_empirical_p"],
        "pair_episode_count": len(pair_ledger),
        "same_regime_episode_count": len(same_metrics),
        "shared_market_episode_count": len(shared_metrics),
        "same_regime_anatomy_counts": same_categories,
        "same_regime_single_loop_share": float(
            same_metrics["anatomy_category"].eq("SINGLE_LOOP_EPISODE").mean()
        ),
        "same_regime_multi_loop_share": float(
            (~same_metrics["anatomy_category"].eq("SINGLE_LOOP_EPISODE")).mean()
        ),
        "median_leader_payoff_share": float(same_metrics["final_leader_share"].median()),
        "median_leader_occurrence_share": float(leader_rows["occurrence_share"].median()),
        "median_leader_efficiency": float(leader_rows["leader_efficiency"].median()),
        "leader_efficiency_supported_episode_rows": int(len(leader_rows)),
        "same_regime_occurrence_complete_episode_count": int(
            same_metrics["occurrence_population_complete"].sum()
        ),
        "leaders_efficiency_above_one_share": float(leader_rows["leader_efficiency"].gt(1).mean()),
        "multi_loop_median_leader_payoff_share": float(
            same_metrics.loc[same_metrics["loop_count"].gt(1), "final_leader_share"].median()
        ),
        "multi_loop_median_leader_occurrence_share": float(
            multi_leaders["occurrence_share"].median()
        ),
        "multi_loop_median_leader_efficiency": float(multi_leaders["leader_efficiency"].median()),
        "multi_loop_leader_efficiency_supported_episode_rows": int(len(multi_leaders)),
        "multi_loop_leaders_efficiency_above_one_share": float(
            multi_leaders["leader_efficiency"].gt(1).mean()
        ),
        "median_common_component": float(common["common_component"].median()),
        "median_regime_component": float(regime["regime_component"].median()),
        "median_loop_excess_component": float(loop_excess["loop_excess_component"].median()),
        "common_positive_contribution_share_median_episode": float(
            all_episode_metrics["common_positive_contribution_share"].median()
        ),
        "regime_positive_contribution_share_median_episode": float(
            all_episode_metrics["regime_positive_contribution_share"].median()
        ),
        "loop_excess_positive_contribution_share_median_episode": float(
            all_episode_metrics["loop_excess_positive_contribution_share"].median()
        ),
        "multi_loop_component_positive_contribution_share_medians": {
            "common": float(
                same_metrics.loc[
                    same_metrics["loop_count"].gt(1),
                    "common_positive_contribution_share",
                ].median()
            ),
            "regime": float(
                same_metrics.loc[
                    same_metrics["loop_count"].gt(1),
                    "regime_positive_contribution_share",
                ].median()
            ),
            "loop_excess": float(
                same_metrics.loc[
                    same_metrics["loop_count"].gt(1),
                    "loop_excess_positive_contribution_share",
                ].median()
            ),
        },
        "component_lag1_persistence": component_persistence_result[
            component_persistence_result["lag_sessions"].eq(1)
        ][["period", "component", "paired_rows", "rank_correlation"]].to_dict(orient="records"),
        "early_leader": early_summary_by_checkpoint.to_dict(orient="records"),
        "multi_loop_early_leader": multi_early_summary.to_dict(orient="records"),
        "leader_persistence": persistence.groupby("lag")
        .agg(top_one=("top_one_persistence", "mean"), top_three=("top_three_persistence", "mean"))
        .reset_index()
        .to_dict(orient="records"),
        "supported_sequence_groups": int(sequence_census["supported"].sum()),
        "multiplicity_controlled_same_direction_both_period_interactions": len(
            supported_interactions
        ),
        "named_pairs": named.to_dict(orient="records"),
        "display_network_edges": int(network["display_edge"].sum()),
        "stock_concentrated_episode_share": float(concentration["stock_concentrated"].mean()),
        "multi_loop_stock_concentrated_episode_share": float(
            concentration.loc[
                concentration["episode_id"].isin(
                    same_metrics.loc[same_metrics["loop_count"].gt(1), "episode_id"]
                ),
                "stock_concentrated",
            ].mean()
        ),
        "shared_regime_activation_episode_count": int(
            same_metrics["shared_regime_activation"].sum()
        ),
        "loop_specific_activation_episode_count": int(
            same_metrics["loop_specific_activation"].sum()
        ),
        "indicator_fdr_significant_comparisons": int(
            associations["reportable_after_support_and_fdr"].sum()
        ),
        "indicator_fdr_significant_period_consistent_comparisons": int(
            (
                associations["reportable_after_support_and_fdr"]
                & associations["period_direction_consistent"]
            ).sum()
        ),
        "factor_diagnostic": factor.to_dict(orient="records"),
        "missing_data_items": len(missing_rows),
    }
    if (
        block_summary["observed_minus_null_share"] <= 0
        or block_summary["one_sided_empirical_p"] > 0.05
    ):
        decision = "coactivation_not_above_null"
    elif concentration["stock_concentrated"].mean() > 0.5:
        decision = "episode_results_stock_or_period_concentrated"
    elif supported_interactions.empty:
        decision = "mixed_episode_anatomy_no_single_mechanism"
    else:
        decision = "shared_episode_with_dominant_loop_supported_descriptively"
    summary["scientific_decision"] = decision
    write_json(output / "scientific_summary.json", summary)

    checkout_git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    frozen_starting_commit = str(contract["lineage"]["starting_commit"])
    frozen_is_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", frozen_starting_commit, checkout_git_head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
    )
    run_metadata = {
        "run_id": RUN_ID,
        "fixed_run_timestamp": RUN_TIMESTAMP,
        "contract_id": contract["contract_id"],
        "contract_hash": contract_hash,
        "data_snapshot_id": snapshot,
        "git_head": frozen_run_git_head(
            contract,
            checkout_git_head,
            frozen_is_ancestor=frozen_is_ancestor,
        ),
        "git_identity_kind": "contract_starting_commit",
        "git_branch": "agent/slrno-research-handoff",
        "python": sys.version.split()[0],
        "primary_horizon": 24,
        "read_only_research": True,
        "predictive_model_built": False,
        "trading_rule_built": False,
    }
    write_json(output / "run_metadata.json", run_metadata)
    make_plots(
        output,
        block,
        block_summary,
        all_episode_metrics,
        all_episode_loops,
        timeline,
        early,
        four_way,
        associations,
        network,
        factor,
    )
    manifest = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    write_json(output / "artifact_manifest.json", manifest)
    if exact_rerun_of is not None:
        identity = exact_rerun_identity(exact_rerun_of, output)
        write_json(output / "exact_rerun_identity.json", identity)
        if not identity["byte_identical"]:
            raise AssertionError(f"exact rerun differs: {identity}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exact-rerun-of", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(
        arguments.output.resolve(),
        arguments.exact_rerun_of.resolve() if arguments.exact_rerun_of else None,
    )
    print(json.dumps(safe_json(result), indent=2, sort_keys=True))
