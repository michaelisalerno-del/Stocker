#!/usr/bin/env python3
# ruff: noqa: E501
"""Run the frozen research-only Fixed One-Bar Entry Latency V1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
PACKAGE_SOURCE = REPO / "packages/stocker_research/src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from stocker_research.fixed_one_bar_entry_latency import (  # noqa: E402
    ProspectiveLatencyLedger,
    score_fixed_latency,
)
from stocker_research.fixed_one_bar_entry_latency.metrics import (  # noqa: E402
    build_exact_paired_population,
    paired_breakdowns,
    paired_summary,
    session_block_bootstrap,
)

CONTRACT_PATH = WORK / "contracts/20260716-fixed-one-bar-entry-latency-v1.json"
DEFAULT_OUTPUT = WORK / "artifacts/20260716-fixed-one-bar-entry-latency-v1/primary"
DEFAULT_REPORT = WORK / "reports/20260716-fixed-one-bar-entry-latency-v1.md"
MODEL_VERSION = "fixed_one_bar_entry_latency_v1.0.0"
RUN_TIMESTAMP = "2026-07-16T00:00:00+00:00"
MACHINE_SUFFIXES = {".parquet", ".csv", ".json"}
IDENTITY_EXCLUSIONS = {
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


def stable_hash(value: object, *, length: int | None = None) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    result = hashlib.sha256(encoded).hexdigest()
    return result if length is None else result[:length]


def _json_safe(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(cast(Any, value))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _resolved(value: str) -> Path:
    return (CONTRACT_PATH.parent / value).resolve()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def load_and_verify_contract() -> tuple[dict[str, Any], str, dict[str, str]]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not contract["registered_before_scoring"]:
        raise AssertionError("experiment contract was not frozen before scoring")
    if contract["t1_fixed_latency"]["primary_latency_bars"] != 1:
        raise AssertionError("primary T1 latency drift")
    if contract["secondary_diagnostics"]["t2_may_replace_t1"] is not False:
        raise AssertionError("T2 was allowed to replace the primary endpoint")
    if contract["stresses"]["unbounded_search_allowed"] is not False:
        raise AssertionError("unbounded search enabled")
    input_hashes: dict[str, str] = {}
    for name, specification in contract["inputs"].items():
        if not isinstance(specification, Mapping) or "path" not in specification:
            continue
        path = _resolved(str(specification["path"]))
        actual = sha256(path)
        if actual != str(specification["sha256"]):
            raise AssertionError(f"frozen input drift: {name}")
        input_hashes[str(name)] = actual
    return contract, sha256(CONTRACT_PATH), input_hashes


def verify_provider_hashes(contract: Mapping[str, Any]) -> dict[str, str]:
    manifest = json.loads(
        _resolved(contract["inputs"]["provider_2025_hash_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )["sha256"]
    root = Path(str(contract["inputs"]["provider_2025_root"]))
    hashes: dict[str, str] = {}
    for key, expected in sorted(manifest.items()):
        if not key.startswith("provider_2025_"):
            continue
        symbol = key.removeprefix("provider_2025_")
        path = root / f"symbol={symbol}" / "timeframe=5m/data.parquet"
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"hash-pinned provider drift for {symbol}")
        hashes[symbol] = actual
    if len(hashes) != 20:
        raise AssertionError("expected frozen twenty-symbol 2025 provider universe")
    return hashes


def verify_file_hash(path: Path, expected: str) -> bool:
    """Accept archival provider input only when it is byte-identical."""

    return path.is_file() and sha256(path) == expected


def _source_reference_path(contract: Mapping[str, Any], track: str) -> Path:
    key = (
        "clean_anchor_named_source"
        if track == "track_a_named_family"
        else "clean_anchor_control_source"
    )
    return _resolved(str(contract["inputs"][key]["path"]))


def build_source_population(contract: Mapping[str, Any], *, track: str) -> pd.DataFrame:
    """Independently reconcile immutable source rows against V2 and veto ledgers."""

    if track not in {"track_a_named_family", "track_b_prior_only"}:
        raise ValueError(f"unsupported population track: {track}")
    reference = pd.read_parquet(_source_reference_path(contract, track))
    policy = pd.read_parquet(_resolved(contract["inputs"]["sequential_veto_policy"]["path"]))
    policy = policy.loc[
        policy["track"].eq(track) & policy["policy"].eq("static_anchor_good_to_bad_odds_veto")
    ].copy()
    if track == "track_b_prior_only":
        policy = policy.loc[
            policy["population_role"].isin(["neutral_control", "negative_control"])
        ].copy()
    v2 = pd.read_parquet(_resolved(contract["inputs"]["v2_trade_decisions"]["path"]))
    v2 = v2.loc[v2["model_name"].eq("no_payoff_state_filter")].copy()
    if policy["opportunity_id"].duplicated().any() or v2["opportunity_id"].duplicated().any():
        raise AssertionError("ambiguous frozen source identity")
    policy_ids = set(policy["opportunity_id"].astype(str))
    reference_ids = set(reference["opportunity_id"].astype(str))
    if policy_ids != reference_ids:
        raise AssertionError("clean-anchor source identity drift")
    trade_columns = [
        "opportunity_id",
        "anchor_id",
        "symbol_norm",
        "session_date",
        "period",
        "start_timestamp",
        "anchor_open",
        "anchor_high",
        "anchor_low",
        "anchor_close",
        "direction",
        "entry_step",
        "entry_timestamp",
        "entry_price",
        "exit_timestamp",
        "exit_price",
        "gross_payoff_bps",
        "primary_total_cost_bps",
        "primary_net_payoff_bps",
        "loop_id",
        "orientation",
        "state",
        "history_token",
        "dollar_volume_proxy",
        "liquidity_proxy_status",
        "sector",
        "month",
        "quarter",
        "run_id",
        "configuration_hash",
        "status",
        "horizon",
        "strategy",
    ]
    source = policy.loc[
        :,
        [
            "experiment_run_id",
            "opportunity_id",
            "event_lineage_id",
            "period",
            "session_date",
            "stock",
            "target_loop",
            "orientation",
            "population_role",
        ],
    ].merge(
        v2.loc[:, trade_columns],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
        suffixes=("_policy", "_v2"),
    )
    if source["anchor_id"].isna().any():
        raise AssertionError("policy opportunity lacks exact V2 row")
    checks = {
        "stock": source["stock"].astype(str).eq(source["symbol_norm"].astype(str)),
        "period": source["period_policy"].astype(str).eq(source["period_v2"].astype(str)),
        "loop": source["target_loop"].astype(str).eq(source["loop_id"].astype(str)),
        "orientation": source["orientation_policy"]
        .astype(str)
        .eq(source["orientation_v2"].astype(str)),
    }
    failed = [name for name, values in checks.items() if not bool(values.all())]
    if failed:
        raise AssertionError(f"source identity mismatch: {failed}")
    source = source.rename(
        columns={
            "symbol_norm": "symbol",
            "period_policy": "period",
            "session_date_policy": "session_date",
            "target_loop": "loop_id_frozen",
            "orientation_policy": "orientation",
            "start_timestamp": "anchor_timestamp",
            "entry_timestamp": "original_entry_timestamp",
            "entry_price": "original_entry_price",
            "exit_timestamp": "original_terminal_timestamp",
            "exit_price": "original_terminal_price",
            "gross_payoff_bps": "original_gross_payoff_bps",
            "primary_total_cost_bps": "original_total_cost_bps",
            "primary_net_payoff_bps": "original_net_payoff_bps",
            "run_id": "source_run_id",
        }
    )
    source["loop_id"] = source.pop("loop_id_frozen")
    source["population_track"] = track
    if track == "track_a_named_family":
        source["population_role"] = "named_candidate"
    source["source_artifact_hash"] = stable_hash(
        {
            "v2": contract["inputs"]["v2_trade_decisions"]["sha256"],
            "policy": contract["inputs"]["sequential_veto_policy"]["sha256"],
        }
    )
    source["source_opportunity_hash"] = [
        stable_hash(
            {
                "opportunity_id": row.opportunity_id,
                "anchor_id": row.anchor_id,
                "direction": row.direction,
                "entry_timestamp": row.original_entry_timestamp,
                "entry_price": row.original_entry_price,
                "terminal_timestamp": row.original_terminal_timestamp,
                "terminal_price": row.original_terminal_price,
            }
        )
        for row in source.itertuples(index=False)
    ]
    for column in ["anchor_timestamp", "original_entry_timestamp", "original_terminal_timestamp"]:
        source[column] = pd.to_datetime(source[column], utc=True, errors="raise")
    if not source["direction"].isin([-1, 1]).all():
        raise AssertionError("ambiguous direction survived source construction")
    if not source["status"].eq("filled").all() or not source["horizon"].eq(24).all():
        raise AssertionError("source execution or horizon drift")
    reference = reference.set_index("opportunity_id")
    source_indexed = source.set_index("opportunity_id")
    reference = reference.loc[source_indexed.index]
    reconciliations = {
        "anchor_id": source_indexed["anchor_id"].astype(str).eq(reference["anchor_id"].astype(str)),
        "direction": source_indexed["direction"].astype(int).eq(reference["direction"].astype(int)),
        "entry_timestamp": source_indexed["original_entry_timestamp"].eq(
            pd.to_datetime(reference["original_entry_timestamp"], utc=True)
        ),
        "terminal_timestamp": source_indexed["original_terminal_timestamp"].eq(
            pd.to_datetime(reference["original_terminal_timestamp"], utc=True)
        ),
        "entry_price": np.isclose(
            source_indexed["original_entry_price"].to_numpy(float),
            reference["original_entry_price"].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ),
        "terminal_price": np.isclose(
            source_indexed["original_terminal_price"].to_numpy(float),
            reference["original_exit_price"].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ),
    }
    failed_reconciliation = [
        name
        for name, values in reconciliations.items()
        if not bool(np.asarray(values, dtype=bool).all())
    ]
    if failed_reconciliation:
        raise AssertionError(f"clean-anchor source field drift: {failed_reconciliation}")
    drop = [
        "period_v2",
        "session_date_v2",
        "orientation_v2",
        "stock",
        "experiment_run_id",
    ]
    source = source.drop(columns=drop)
    return source.sort_values(
        ["period", "session_date", "symbol", "anchor_timestamp", "opportunity_id"],
        kind="stable",
    ).reset_index(drop=True)


def load_provider_frames(
    contract: Mapping[str, Any], symbols: Iterable[str]
) -> dict[str, pd.DataFrame]:
    root = Path(str(contract["inputs"]["provider_2025_root"]))
    frames: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(str(value) for value in symbols)):
        path = root / f"symbol={symbol}" / "timeframe=5m/data.parquet"
        frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frames[symbol] = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return frames


def score_population(
    source: pd.DataFrame,
    providers: Mapping[str, pd.DataFrame],
    *,
    latency_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate an unconditional exact post-fill latency on every source row."""

    expected_records: list[dict[str, object]] = []
    scored_records: list[dict[str, object]] = []
    restarted_records: list[dict[str, object]] = []
    for row in source.itertuples(index=False):
        identity = {
            "opportunity_id": str(row.opportunity_id),
            "anchor_id": str(row.anchor_id),
            "event_lineage_id": str(row.event_lineage_id),
            "period": _as_int(row.period),
            "session_date": str(row.session_date),
            "symbol": str(row.symbol),
            "loop_id": str(row.loop_id),
            "orientation": str(row.orientation),
            "direction": _as_int(row.direction),
            "original_terminal_timestamp": _as_timestamp(row.original_terminal_timestamp),
        }
        t0 = _as_timestamp(row.original_entry_timestamp)
        expected = t0 + pd.Timedelta(minutes=5 * latency_bars)
        if _as_int(row.period) != 2025:
            status = "provider_2023_hash_pinned_tape_unavailable"
            expected_records.append(
                {
                    **identity,
                    "t1_expected_timestamp": expected,
                    "t1_status": status,
                    "expected_latency_bars": latency_bars,
                }
            )
            scored_records.append(
                {
                    **identity,
                    "t1_status": status,
                    "t1_expected_timestamp": expected,
                    "t1_entry_timestamp": pd.NaT,
                    "t1_entry_price": np.nan,
                    "t1_gross_return_bps": np.nan,
                    "t1_total_cost_bps": np.nan,
                    "t1_net_return_bps": np.nan,
                    "paired_difference_bps": np.nan,
                }
            )
            restarted_records.append(
                {
                    **identity,
                    "restarted_status": status,
                    "restarted_exit_timestamp": pd.NaT,
                    "restarted_net_return_bps": np.nan,
                }
            )
            continue
        frame = providers[str(row.symbol)]
        anchor = _as_timestamp(row.anchor_timestamp)
        path_end = max(
            _as_timestamp(row.original_terminal_timestamp),
            expected + pd.Timedelta(minutes=125),
        )
        bars = frame.loc[frame["timestamp"].between(anchor, path_end, inclusive="both")].copy()
        result = score_fixed_latency(
            bars,
            anchor_timestamp=anchor,
            entry_step=_as_int(row.entry_step),
            t0_entry_timestamp=t0,
            t0_entry_price=_as_float(row.original_entry_price),
            original_terminal_timestamp=_as_timestamp(row.original_terminal_timestamp),
            original_terminal_price=_as_float(row.original_terminal_price),
            direction=_as_int(row.direction),
            source_t0_gross_return_bps=_as_float(row.original_gross_payoff_bps),
            source_t0_net_return_bps=_as_float(row.original_net_payoff_bps),
            latency_bars=latency_bars,
            cost_bps_per_side=5.0,
            restarted_horizon_bars=24,
        )
        values = asdict(result)
        status = str(values.pop("status"))
        expected_records.append(
            {
                **identity,
                "t1_expected_timestamp": expected,
                "t1_status": status,
                "expected_latency_bars": latency_bars,
            }
        )
        scored_records.append({**identity, "t1_status": status, **values})
        restarted_records.append(
            {
                **identity,
                "restarted_status": (
                    "available" if result.restarted_net_return_bps is not None else status
                ),
                "t1_entry_timestamp": result.t1_entry_timestamp,
                "t1_entry_price": result.t1_entry_price,
                "restarted_exit_timestamp": result.restarted_exit_timestamp,
                "restarted_terminal_price": result.restarted_terminal_price,
                "restarted_gross_return_bps": result.restarted_gross_return_bps,
                "restarted_net_return_bps": result.restarted_net_return_bps,
            }
        )
    return (
        pd.DataFrame(expected_records),
        pd.DataFrame(scored_records),
        pd.DataFrame(restarted_records),
    )


def attach_evaluation_context(frame: pd.DataFrame, contract: Mapping[str, Any]) -> pd.DataFrame:
    """Attach descriptive context and hindsight episodes only after scoring."""

    result = frame.copy()
    result["direction_label"] = np.where(result["direction"].eq(1), "long", "short")
    result["month"] = result["session_date"].astype(str).str[:7]
    result["quarter_label"] = pd.to_datetime(result["session_date"]).dt.to_period("Q").astype(str)
    local = pd.to_datetime(result["original_entry_timestamp"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    result["clock_phase"] = np.select(
        [local.dt.hour.lt(11), local.dt.hour.lt(14)],
        ["early", "middle"],
        default="late",
    )
    result["anchor_regime"] = "state_" + result["state"].astype(int).astype(str)
    result["hindsight_episode_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["hindsight_episode_status"] = "outside_hindsight_positive_episode"
    episodes = pd.read_parquet(_resolved(contract["inputs"]["episode_diagnostics"]["path"]))
    episodes = episodes.copy()
    episodes["onset"] = pd.to_datetime(episodes["hindsight_estimated_onset"]).dt.date
    episodes["end"] = pd.to_datetime(episodes["hindsight_estimated_end"]).dt.date
    for index, row in result.iterrows():
        date = pd.Timestamp(row["session_date"]).date()
        candidates = episodes.loc[
            episodes["period"].eq(int(row["period"]))
            & episodes["loop_id"].eq(str(row["loop_id"]))
            & episodes["orientation"].eq(str(row["orientation"]))
            & episodes["onset"].le(date)
            & episodes["end"].ge(date)
        ]
        if len(candidates) == 1:
            result.at[index, "hindsight_episode_id"] = str(candidates.iloc[0]["episode_id"])
            result.at[index, "hindsight_episode_status"] = "matched_positive_episode"
        elif len(candidates) > 1:
            result.at[index, "hindsight_episode_status"] = "ambiguous_hindsight_episode"
    return result


def _performance(values: pd.Series) -> dict[str, float | int]:
    net = pd.to_numeric(values, errors="coerce").dropna()
    positive = float(net.loc[net.gt(0.0)].sum())
    loss = float(-net.loc[net.lt(0.0)].sum())
    cumulative = net.cumsum()
    drawdown = float((cumulative - cumulative.cummax()).min()) if len(net) else 0.0
    return {
        "trade_count": int(len(net)),
        "net_payoff_bps": float(net.sum()),
        "mean_net_payoff_bps": float(net.mean()) if len(net) else np.nan,
        "median_net_payoff_bps": float(net.median()) if len(net) else np.nan,
        "positive_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
        "profit_factor": positive / loss if loss > 0.0 else np.nan,
        "maximum_drawdown_bps": drawdown,
    }


def build_primary_metrics(
    source: pd.DataFrame,
    all_pairs: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    dimensions: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = paired_breakdowns(paired, dimensions=dimensions)
    counts: list[dict[str, object]] = []
    slices: list[tuple[str, str, pd.Series, pd.Series]] = [
        (
            "all",
            "all",
            pd.Series(True, index=source.index),
            pd.Series(True, index=all_pairs.index),
        )
    ]
    for dimension in dimensions:
        if dimension not in source or dimension not in all_pairs:
            continue
        values = sorted(set(source[dimension].dropna().astype(str)))
        for value in values:
            slices.append(
                (
                    dimension,
                    value,
                    source[dimension].astype(str).eq(value),
                    all_pairs[dimension].astype(str).eq(value),
                )
            )
    for slice_type, slice_value, source_mask, all_mask in slices:
        subset = all_pairs.loc[all_mask]
        counts.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                "source_opportunities": int(source_mask.sum()),
                "exact_t0_opportunities": int(subset["t0_available"].sum()),
                "exact_t1_opportunities": int(subset["t1_available"].sum()),
                "paired_opportunities": int(subset["paired_available"].sum()),
                "pairing_rate": float(subset["paired_available"].mean()) if len(subset) else np.nan,
            }
        )
    metrics = metrics.rename(columns={"paired_opportunities": "scored_paired_opportunities"})
    metrics = pd.DataFrame(counts).merge(
        metrics,
        on=["slice_type", "slice_value"],
        how="left",
        validate="one_to_one",
    )
    scored_count = metrics["scored_paired_opportunities"]
    count_mismatch = scored_count.notna() & metrics["paired_opportunities"].ne(scored_count)
    if bool(count_mismatch.any()):
        raise AssertionError("paired metric count differs from exact availability accounting")
    metrics = metrics.drop(columns="scored_paired_opportunities")
    bootstrap_rows: list[dict[str, object]] = []
    bootstrap_slices: list[tuple[str, str, pd.DataFrame]] = [("all", "all", paired)]
    for dimension in ["period", "loop_id", "orientation", "direction_label"]:
        if dimension not in paired:
            continue
        for slice_key, group in paired.groupby(dimension, dropna=False, sort=True):
            bootstrap_slices.append((dimension, str(slice_key), group))
    for slice_type, slice_value, group in bootstrap_slices:
        interval = session_block_bootstrap(
            group,
            resamples=2000,
            block_length=5,
            seed=20260716,
        )
        by_session = group.groupby(["period", "session_date"], sort=True)[
            "paired_difference_bps"
        ].mean()
        bootstrap_rows.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                "sessions": int(len(by_session)),
                "sessions_improved_fraction": float(by_session.gt(0.0).mean()),
                **interval,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    metrics = metrics.merge(
        bootstrap,
        on=["slice_type", "slice_value"],
        how="left",
        validate="one_to_one",
    )
    return metrics, bootstrap


def build_full_sample_levels(source: pd.DataFrame, all_pairs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    slices = [("all", "all", pd.Series(True, index=source.index))]
    for dimension in ["period", "loop_id", "orientation", "direction_label"]:
        for value in sorted(source[dimension].dropna().astype(str).unique()):
            slices.append((dimension, value, source[dimension].astype(str).eq(value)))
    for slice_type, slice_value, mask in slices:
        group = source.loc[mask].sort_values(
            ["original_entry_timestamp", "opportunity_id"], kind="stable"
        )
        t0 = _performance(group["original_net_payoff_bps"])
        records.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                "timing": "T0_full_source",
                "source_opportunities": int(len(group)),
                "gross_payoff_bps": float(group["original_gross_payoff_bps"].sum()),
                "total_costs_bps": float(group["original_total_cost_bps"].sum()),
                **t0,
            }
        )
        ids = set(group["opportunity_id"].astype(str))
        t1_group = all_pairs.loc[
            all_pairs["opportunity_id"].astype(str).isin(ids) & all_pairs["t1_available"]
        ].sort_values(["t1_entry_timestamp", "opportunity_id"], kind="stable")
        t1_performance = _performance(t1_group["t1_net_return_bps"])
        records.append(
            {
                "slice_type": slice_type,
                "slice_value": slice_value,
                "timing": "T1_available_unpaired_context",
                "source_opportunities": int(len(group)),
                "gross_payoff_bps": float(t1_group["t1_gross_return_bps"].sum()),
                "total_costs_bps": float(t1_group["t1_total_cost_bps"].sum()),
                **t1_performance,
            }
        )
    return pd.DataFrame(records)


def build_entry_decomposition(paired: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    slices: list[tuple[str, str, pd.DataFrame]] = [("all", "all", paired)]
    for dimension in ["period", "loop_id", "orientation", "direction_label"]:
        for value, group in paired.groupby(dimension, dropna=False, sort=True):
            slices.append((dimension, str(value), group))
    if "t0_net_return_bps" in paired:
        t0_payoff = pd.to_numeric(paired["t0_net_return_bps"], errors="coerce")
        slices.extend(
            [
                ("t0_outcome", "positive", paired.loc[t0_payoff.gt(0.0)]),
                ("t0_outcome", "nonpositive", paired.loc[t0_payoff.le(0.0)]),
            ]
        )
    for slice_type, slice_value, group in slices:
        move = pd.to_numeric(group["direction_adjusted_entry_move_bps"], errors="coerce")
        delta = pd.to_numeric(group["paired_difference_bps"], errors="coerce")
        valid = move.notna() & delta.notna()
        correlation = spearmanr(move.loc[valid], delta.loc[valid]) if valid.sum() >= 3 else None
        for sign_name, sign_mask in [
            ("all", pd.Series(True, index=group.index)),
            ("adverse_before_t1", move.lt(0.0)),
            ("favourable_before_t1", move.gt(0.0)),
            ("unchanged_before_t1", np.isclose(move, 0.0)),
        ]:
            selected = group.loc[sign_mask]
            selected_move = pd.to_numeric(
                selected["direction_adjusted_entry_move_bps"], errors="coerce"
            )
            selected_delta = pd.to_numeric(selected["paired_difference_bps"], errors="coerce")
            records.append(
                {
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "entry_move_cell": sign_name,
                    "opportunities": int(len(selected)),
                    "mean_direction_adjusted_entry_move_bps": float(selected_move.mean()),
                    "median_direction_adjusted_entry_move_bps": float(selected_move.median()),
                    "mean_paired_difference_bps": float(selected_delta.mean()),
                    "total_paired_difference_bps": float(selected_delta.sum()),
                    "opportunities_improved_fraction": float(selected_delta.gt(0.0).mean())
                    if len(selected)
                    else np.nan,
                    "move_delta_spearman_rho": float(correlation.statistic)
                    if correlation is not None
                    else np.nan,
                    "move_delta_spearman_pvalue": float(correlation.pvalue)
                    if correlation is not None
                    else np.nan,
                }
            )
    return pd.DataFrame(records)


def build_cost_stress(named_paired: pd.DataFrame, control_paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    populations = [("named", named_paired), ("control", control_paired)]
    for population, frame in populations:
        slices: list[tuple[str, str, pd.DataFrame]] = [("all", "all", frame)]
        for dimension in ["period", "loop_id", "orientation"]:
            for value, group in frame.groupby(dimension, dropna=False, sort=True):
                slices.append((dimension, str(value), group))
        for slice_type, slice_value, group in slices:
            t0_gross = pd.to_numeric(group["t0_gross_return_bps"], errors="coerce")
            t1_gross = pd.to_numeric(group["t1_gross_return_bps"], errors="coerce")
            rows.append(
                {
                    "population": population,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "stress": "frozen_costs",
                    "opportunities": int(len(group)),
                    "t0_net_payoff_bps": float((t0_gross - 10.0).sum()),
                    "t1_net_payoff_bps": float((t1_gross - 10.0).sum()),
                    "paired_difference_bps": float((t1_gross - t0_gross).sum()),
                }
            )
            rows.append(
                {
                    "population": population,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "stress": "twice_costs",
                    "opportunities": int(len(group)),
                    "t0_net_payoff_bps": float((t0_gross - 20.0).sum()),
                    "t1_net_payoff_bps": float((t1_gross - 20.0).sum()),
                    "paired_difference_bps": float((t1_gross - t0_gross).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_concentration(paired: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    dimensions = [
        "symbol",
        "loop_id",
        "orientation",
        "direction_label",
        "period",
        "month",
        "quarter_label",
        "clock_phase",
        "anchor_regime",
        "hindsight_episode_id",
    ]
    for dimension in dimensions:
        contribution = paired.groupby(dimension, dropna=False)["paired_difference_bps"].sum()
        absolute = contribution.abs()
        denominator = float(absolute.sum())
        shares = absolute / denominator if denominator > 0.0 else absolute * np.nan
        ordered = shares.sort_values(ascending=False)
        hhi = float(np.square(shares).sum()) if denominator > 0.0 else np.nan
        for contributor, value in contribution.items():
            records.append(
                {
                    "dimension": dimension,
                    "contributor": str(contributor),
                    "contribution_bps": float(value),
                    "absolute_contribution_share": float(shares.get(contributor, np.nan))
                    if denominator > 0.0
                    else np.nan,
                    "top_one_absolute_share": float(ordered.iloc[0]) if len(ordered) else np.nan,
                    "top_five_absolute_share": float(ordered.head(5).sum())
                    if len(ordered)
                    else np.nan,
                    "herfindahl": hhi,
                }
            )
    return pd.DataFrame(records)


def build_deletion_stress(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    stock = paired.groupby("symbol")["paired_difference_bps"].sum().sort_values(ascending=False)
    episode_rows = paired.loc[paired["hindsight_episode_id"].notna()].copy()
    episode = (
        episode_rows.groupby("hindsight_episode_id")["paired_difference_bps"]
        .sum()
        .sort_values(ascending=False)
    )
    best_stock = [str(stock.index[0])] if len(stock) else []
    top_stocks = list(stock.head(5).index.astype(str))
    best_episode = [str(episode.index[0])] if len(episode) else []
    top_episodes = list(episode.head(5).index.astype(str))
    deletions = {
        "remove_best_stock": (best_stock, "available"),
        "remove_top_five_stocks": (top_stocks, "available"),
        "remove_best_episode": (
            best_episode,
            "available" if best_episode else "no_hindsight_positive_episode_available",
        ),
        "remove_top_five_episodes": (
            top_episodes,
            "available" if top_episodes else "no_hindsight_positive_episode_available",
        ),
    }
    for name, (contributors, availability) in deletions.items():
        dimension = "symbol" if "stock" in name else "hindsight_episode_id"
        mask = ~paired[dimension].astype("string").isin(contributors)
        group = paired.loc[mask]
        rows.append(
            {
                "stress": name,
                "removed_contributors": "|".join(contributors),
                "paired_opportunities": int(len(group)),
                "paired_total_difference_bps": float(group["paired_difference_bps"].sum()),
                "paired_mean_difference_bps": float(group["paired_difference_bps"].mean()),
                "status": (
                    "immutable_population_deletion_no_replacement"
                    if availability == "available"
                    else availability
                ),
            }
        )
    threshold = paired.groupby("period")["dollar_volume_proxy"].transform("median")
    liquid = paired.loc[paired["dollar_volume_proxy"].ge(threshold)]
    rows.append(
        {
            "stress": "minimum_liquidity_within_period_median",
            "removed_contributors": "",
            "paired_opportunities": int(len(liquid)),
            "paired_total_difference_bps": float(liquid["paired_difference_bps"].sum()),
            "paired_mean_difference_bps": float(liquid["paired_difference_bps"].mean()),
            "status": "frozen_causal_dollar_volume_proxy",
        }
    )
    loo: list[dict[str, object]] = []
    for symbol in sorted(paired["symbol"].unique()):
        group = paired.loc[paired["symbol"].ne(symbol)]
        loo.append(
            {
                "excluded_symbol": str(symbol),
                "paired_opportunities": int(len(group)),
                "paired_total_difference_bps": float(group["paired_difference_bps"].sum()),
                "paired_mean_difference_bps": float(group["paired_difference_bps"].mean()),
                "status": "exact_deterministic_recalculation_no_trainable_state",
                "fully_rebuilt_stock_dependent_model": False,
                "reason": "no trained or stock-dependent timing model exists; all remaining exact rows are recomputed",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(loo)


def build_nulls(
    paired: pd.DataFrame,
    control_paired: pd.DataFrame,
    t2_paired: pd.DataFrame,
    *,
    trading_sessions: Sequence[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(20260716)
    delta = paired["paired_difference_bps"].to_numpy(float)
    draws = np.array(
        [float((delta * rng.integers(0, 2, size=len(delta))).sum()) for _ in range(500)]
    )
    shifted = paired.sort_values(
        ["period", "symbol", "session_date", "opportunity_id"], kind="stable"
    ).copy()
    shifted["entry_price_ratio"] = shifted["t1_entry_price"] / shifted["t0_entry_price"]
    by_symbol = shifted.groupby(["period", "symbol"], sort=False)
    shifted["prior_session_entry_price_ratio"] = by_symbol["entry_price_ratio"].shift(1)
    shifted["prior_session_date"] = by_symbol["session_date"].shift(1)
    ordered_sessions = sorted({str(value) for value in trading_sessions})
    previous_session = {
        session: ordered_sessions[index - 1] if index > 0 else None
        for index, session in enumerate(ordered_sessions)
    }
    shifted["expected_prior_trading_session"] = (
        shifted["session_date"].astype(str).map(previous_session)
    )
    valid = shifted.loc[
        shifted["prior_session_entry_price_ratio"].notna()
        & shifted["prior_session_date"]
        .astype(str)
        .eq(shifted["expected_prior_trading_session"].astype(str))
    ].copy()
    valid["null_t1_entry_price"] = (
        valid["t0_entry_price"] * valid["prior_session_entry_price_ratio"]
    )
    null_t0_gross = (
        10_000.0 * valid["direction"] * (valid["terminal_price"] / valid["t0_entry_price"] - 1.0)
    )
    null_shifted_gross = (
        10_000.0
        * valid["direction"]
        * (valid["terminal_price"] / valid["null_t1_entry_price"] - 1.0)
    )
    shifted_increment = null_shifted_gross - null_t0_gross
    return pd.DataFrame(
        [
            {
                "null_test": "random_T0_or_T1_timing_500_repetitions",
                "opportunities": int(len(paired)),
                "null_mean_increment_bps": float(draws.mean()),
                "null_lower_95_bps": float(np.quantile(draws, 0.025)),
                "null_upper_95_bps": float(np.quantile(draws, 0.975)),
                "actual_T1_increment_bps": float(delta.sum()),
                "actual_percentile_within_null": float(np.mean(draws <= delta.sum())),
            },
            {
                "null_test": "prior_session_entry_displacement",
                "opportunities": int(len(valid)),
                "shifted_increment_bps": float(shifted_increment.sum()),
                "shifted_mean_increment_bps": float(shifted_increment.mean()),
                "status": "non_executable_exact_prior_trading_session_price_ratio_displacement",
            },
            {
                "null_test": "timestamp_offset_falsification_T2",
                "opportunities": int(len(t2_paired)),
                "actual_T1_increment_bps": float(delta.sum()),
                "wrong_offset_T2_increment_bps": float(t2_paired["paired_difference_bps"].sum()),
                "detectably_different": bool(
                    not np.isclose(
                        float(delta.sum()),
                        float(t2_paired["paired_difference_bps"].sum()),
                        rtol=0.0,
                        atol=1e-10,
                    )
                ),
            },
            {
                "null_test": "frozen_control_orientations_equal_clock",
                "opportunities": int(len(control_paired)),
                "control_increment_bps": float(control_paired["paired_difference_bps"].sum()),
                "control_mean_increment_bps": float(control_paired["paired_difference_bps"].mean()),
            },
        ]
    )


def prospective_schema() -> dict[str, object]:
    return {
        "schema_version": "fixed_one_bar_entry_latency_prospective_v1",
        "research_only": True,
        "execution_enabled": False,
        "opened_periods_forbidden_in_holdout": [2023, 2025],
        "opportunity_ledger": {
            "immutable": True,
            "required_fields": [
                "run_id",
                "git_sha",
                "contract_hash",
                "data_snapshot_hash",
                "source_run_id",
                "source_artifact_hash",
                "source_opportunity_hash",
                "opportunity_id",
                "anchor_id",
                "event_lineage_id",
                "symbol",
                "session",
                "loop_id",
                "orientation",
                "frozen_direction",
                "anchor_timestamp",
                "t0_entry_timestamp",
                "t0_entry_price",
                "expected_t1_timestamp",
                "original_terminal_timestamp",
                "provider_data_hash",
                "forecast_freeze_timestamp",
            ],
        },
        "timing_ledger": {
            "create_only": True,
            "separate_from_opportunity": True,
            "exact_t1": "t0_entry_timestamp plus five minutes",
        },
        "outcome_ledger": {
            "create_only": True,
            "separate_from_forecast_and_timing": True,
            "settles_after_original_terminal": True,
        },
        "safety": {
            "broker_connection_enabled": False,
            "order_placement_enabled": False,
            "position_management_enabled": False,
            "existing_exit_management_enabled": False,
            "deployment_enabled": False,
        },
    }


def build_missing_2023_report(contract: Mapping[str, Any]) -> dict[str, object]:
    manifest_path = _resolved(contract["inputs"]["provider_2023_hash_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]
    expected = {
        key.removeprefix("provider_2023_"): value
        for key, value in manifest.items()
        if key.startswith("provider_2023_")
    }
    current_root = Path(str(contract["inputs"]["provider_2025_root"]))
    current_matches: list[str] = []
    for symbol, expected_hash in sorted(expected.items()):
        current = current_root / f"symbol={symbol}" / "timeframe=5m/data.parquet"
        if verify_file_hash(current, expected_hash):
            current_matches.append(symbol)
    expired = Path(str(contract["inputs"]["expired_provider_2023_root"]))
    return {
        "status": "unavailable_no_matching_original_2023_provider_files",
        "required_symbols": len(expected),
        "expected_hash_manifest": str(manifest_path),
        "expected_hash_manifest_sha256": sha256(manifest_path),
        "expired_original_root": str(expired),
        "expired_original_root_exists": expired.exists(),
        "pre_score_search_roots": contract["provider_2023_archive_search"][
            "pre_score_search_roots"
        ],
        "pre_score_candidate_files_hashed": contract["provider_2023_archive_search"][
            "pre_score_candidate_files_hashed"
        ],
        "pre_score_exact_hash_matches": contract["provider_2023_archive_search"][
            "pre_score_exact_hash_matches"
        ],
        "current_2025_files_matching_2023_hashes": current_matches,
        "fresh_download_allowed": False,
        "approximate_or_imputed_reconstruction_allowed": False,
        "t1_outcomes_imputed": False,
    }


def annotate(
    frame: pd.DataFrame,
    *,
    run_id: str,
    contract_hash: str,
    data_snapshot_hash: str,
) -> pd.DataFrame:
    result = frame.copy()
    for column, value in [
        ("run_id", run_id),
        ("contract_hash", contract_hash),
        ("data_snapshot_hash", data_snapshot_hash),
        ("model_version", MODEL_VERSION),
    ]:
        if column not in result:
            result[column] = value
    return result


def artifact_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }


def verify_exact_rerun(output: Path, primary: Path) -> dict[str, object]:
    expected = {
        str(path.relative_to(primary)): path
        for path in primary.rglob("*")
        if path.is_file()
        and (path.suffix in MACHINE_SUFFIXES or path.suffix == ".png")
        and path.name not in IDENTITY_EXCLUSIONS
    }
    actual = {
        str(path.relative_to(output)): path
        for path in output.rglob("*")
        if path.is_file()
        and (path.suffix in MACHINE_SUFFIXES or path.suffix == ".png")
        and path.name not in IDENTITY_EXCLUSIONS
    }
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatches = sorted(
        name
        for name in set(expected) & set(actual)
        if sha256(expected[name]) != sha256(actual[name])
    )
    return {
        "byte_identical": not missing and not extra and not mismatches,
        "compared_files_including_plots": len(expected),
        "missing_files": missing,
        "extra_files": extra,
        "hash_mismatches": mismatches,
    }


def make_plots(output: Path, paired: pd.DataFrame, concentration: pd.DataFrame) -> list[Path]:
    plot_root = output / "plots"
    plot_root.mkdir()
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(paired["t0_net_return_bps"], paired["t1_net_return_bps"], s=10, alpha=0.4)
    limits = [
        float(min(paired["t0_net_return_bps"].min(), paired["t1_net_return_bps"].min())),
        float(max(paired["t0_net_return_bps"].max(), paired["t1_net_return_bps"].max())),
    ]
    ax.plot(limits, limits, color="black", linewidth=1)
    ax.set(xlabel="T0 net payoff (bps)", ylabel="T1 net payoff (bps)")
    ax.set_title("Exact paired T0 versus T1")
    paths.append(plot_root / "t0_vs_t1_paired_payoff.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(paired["paired_difference_bps"], bins=40, color="#2563eb", alpha=0.8)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set(xlabel="T1 minus T0 (bps)", ylabel="opportunities")
    ax.set_title("Fixed one-bar paired delta")
    paths.append(plot_root / "paired_delta_distribution.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    ordered = paired.sort_values(["original_entry_timestamp", "opportunity_id"], kind="stable")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ordered["original_entry_timestamp"], ordered["t0_net_return_bps"].cumsum(), label="T0")
    ax.plot(ordered["original_entry_timestamp"], ordered["t1_net_return_bps"].cumsum(), label="T1")
    ax.legend()
    ax.set(xlabel="entry time", ylabel="cumulative net payoff (bps)")
    ax.set_title("Same-population cumulative T0 and T1")
    paths.append(plot_root / "cumulative_t0_t1.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    loop = paired.groupby("loop_id")["paired_difference_bps"].agg(["mean", "sum"])
    fig, ax = plt.subplots(figsize=(6, 4))
    loop["mean"].plot.bar(ax=ax, color="#0f766e")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set(xlabel="named loop", ylabel="mean T1 minus T0 (bps)")
    ax.set_title("Paired delta by named loop")
    paths.append(plot_root / "paired_delta_by_named_loop.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        paired["direction_adjusted_entry_move_bps"],
        paired["paired_difference_bps"],
        s=10,
        alpha=0.4,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set(xlabel="direction-adjusted T0 to T1 move (bps)", ylabel="T1 minus T0 (bps)")
    ax.set_title("Entry-price decomposition")
    paths.append(plot_root / "entry_move_vs_paired_delta.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    clock = paired.groupby("clock_phase")["paired_difference_bps"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    clock.plot.bar(ax=ax, color="#7c3aed")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set(xlabel="T0 clock phase", ylabel="mean paired delta (bps)")
    ax.set_title("Latency effect by session clock")
    paths.append(plot_root / "paired_delta_by_clock_phase.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    stock = concentration.loc[concentration["dimension"].eq("symbol")].copy()
    stock = stock.sort_values(
        "contribution_bps", key=lambda values: values.abs(), ascending=False
    ).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(stock["contributor"], stock["contribution_bps"], color="#ea580c")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.tick_params(axis="x", rotation=45)
    ax.set(ylabel="paired delta contribution (bps)")
    ax.set_title("Largest stock contributions")
    paths.append(plot_root / "delta_concentration_by_stock.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    examples = [
        (
            "adverse move improves T1",
            paired.loc[paired["direction_adjusted_entry_move_bps"].lt(0.0)].sort_values(
                "paired_difference_bps", ascending=False
            ),
        ),
        (
            "favourable move harms T1",
            paired.loc[paired["direction_adjusted_entry_move_bps"].gt(0.0)].sort_values(
                "paired_difference_bps"
            ),
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, (title, candidates) in zip(axes, examples, strict=True):
        if candidates.empty:
            axis.text(0.5, 0.5, "no eligible example", ha="center", va="center")
            axis.set_title(title)
            continue
        row = candidates.iloc[0]
        axis.plot(
            [0, 1, 2],
            [
                float(row["t0_entry_price"]),
                float(row["t1_entry_price"]),
                float(row["terminal_price"]),
            ],
            marker="o",
        )
        axis.set_xticks([0, 1, 2], ["T0", "T1", "terminal"])
        axis.set_title(
            f"{title}\n{row['symbol']} {row['loop_id']} delta={row['paired_difference_bps']:.1f}"
        )
        axis.set_ylabel("provider price")
    paths.append(plot_root / "representative_latency_examples.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)
    return paths


def scientific_decision() -> str:
    return "experiment_blocked_by_missing_2023_provider_tape"


def _available_interpretation(
    primary: pd.Series, loop_metrics: pd.DataFrame, control: pd.Series
) -> str:
    loop_deltas = loop_metrics.set_index("slice_value")["paired_total_difference_bps"]
    cycle_04 = float(loop_deltas.get("cycle_04", np.nan))
    cycle_07 = float(loop_deltas.get("cycle_07", np.nan))
    named_mean = float(primary["paired_mean_difference_bps"])
    control_mean = float(control["paired_mean_difference_bps"])
    if cycle_04 > 0.0 and cycle_07 > 0.0:
        if control_mean >= named_mean:
            return "general_one_bar_timing_effect_not_loop_specific"
        return "named_loop_entry_signal_appears_early"
    if cycle_04 > 0.0 >= cycle_07:
        return "cycle_04_specific_latency_descriptive_only"
    if cycle_07 > 0.0 >= cycle_04:
        return "cycle_07_specific_latency_descriptive_only"
    return "same_terminal_latency_not_supported"


def write_report(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    named_source: pd.DataFrame,
    primary_metrics: pd.DataFrame,
    control_metrics: pd.DataFrame,
    decomposition: pd.DataFrame,
    cost_stress: pd.DataFrame,
    t2_metrics: pd.DataFrame,
    deletion: pd.DataFrame,
    concentration: pd.DataFrame,
    missing_2023: Mapping[str, Any],
) -> None:
    primary = primary_metrics.loc[
        primary_metrics["slice_type"].eq("all") & primary_metrics["slice_value"].eq("all")
    ].iloc[0]
    loop_rows = primary_metrics.loc[primary_metrics["slice_type"].eq("loop_id")]
    period_rows = primary_metrics.loc[primary_metrics["slice_type"].eq("period")]
    direction_rows = primary_metrics.loc[primary_metrics["slice_type"].eq("direction_label")]
    control = control_metrics.loc[
        control_metrics["slice_type"].eq("all") & control_metrics["slice_value"].eq("all")
    ].iloc[0]
    interpretation = _available_interpretation(primary, loop_rows, control)
    cost_primary = cost_stress.loc[
        cost_stress["population"].eq("named")
        & cost_stress["slice_type"].eq("all")
        & cost_stress["stress"].eq("frozen_costs")
    ].iloc[0]
    twice = cost_stress.loc[
        cost_stress["population"].eq("named")
        & cost_stress["slice_type"].eq("all")
        & cost_stress["stress"].eq("twice_costs")
    ].iloc[0]
    entry = decomposition.loc[
        decomposition["slice_type"].eq("all")
        & decomposition["slice_value"].eq("all")
        & decomposition["entry_move_cell"].eq("all")
    ].iloc[0]
    adverse = decomposition.loc[
        decomposition["slice_type"].eq("all")
        & decomposition["entry_move_cell"].eq("adverse_before_t1")
    ].iloc[0]
    profitable = decomposition.loc[
        decomposition["slice_type"].eq("t0_outcome")
        & decomposition["slice_value"].eq("positive")
        & decomposition["entry_move_cell"].eq("all")
    ].iloc[0]
    profitable_adverse = decomposition.loc[
        decomposition["slice_type"].eq("t0_outcome")
        & decomposition["slice_value"].eq("positive")
        & decomposition["entry_move_cell"].eq("adverse_before_t1")
    ].iloc[0]
    stock_concentration = concentration.loc[concentration["dimension"].eq("symbol")].iloc[0]
    loop_lines = "\n".join(
        f"| {row.slice_value} | {_as_int(row.paired_opportunities)} | {_as_float(row.t0_net_payoff_bps):.2f} | {_as_float(row.t1_net_payoff_bps):.2f} | {_as_float(row.paired_total_difference_bps):.2f} | {_as_float(row.paired_mean_difference_bps):.2f} |"
        for row in loop_rows.itertuples(index=False)
    )
    period_lines = "\n".join(
        f"| {row.slice_value} | {_as_int(row.source_opportunities)} | {_as_int(row.paired_opportunities)} | {_as_float(row.t0_net_payoff_bps):.2f} | {_as_float(row.t1_net_payoff_bps):.2f} | {_as_float(row.paired_total_difference_bps):.2f} |"
        for row in period_rows.itertuples(index=False)
    )
    direction_lines = "\n".join(
        f"| {row.slice_value} | {_as_int(row.paired_opportunities)} | {_as_float(row.t0_net_payoff_bps):.2f} | {_as_float(row.t1_net_payoff_bps):.2f} | {_as_float(row.paired_total_difference_bps):.2f} |"
        for row in direction_rows.itertuples(index=False)
    )
    control_rows = control_metrics.loc[control_metrics["slice_type"].eq("orientation")]
    control_lines = "\n".join(
        f"| {row.slice_value} | {_as_int(row.paired_opportunities)} | {_as_float(row.t0_net_payoff_bps):.2f} | {_as_float(row.t1_net_payoff_bps):.2f} | {_as_float(row.paired_total_difference_bps):.2f} | {_as_float(row.paired_mean_difference_bps):.2f} |"
        for row in control_rows.itertuples(index=False)
    )
    t2 = t2_metrics.loc[t2_metrics["slice_type"].eq("all")].iloc[0]
    deletion_lines = "\n".join(
        f"- {row.stress}: {_as_int(row.paired_opportunities)} pairs, delta {_as_float(row.paired_total_difference_bps):.2f} bps."
        for row in deletion.itertuples(index=False)
        if str(row.stress).startswith("remove_")
    )
    counts = named_source.groupby(["period", "loop_id", "orientation"]).size().to_dict()
    report = f"""# Fixed One-Bar Entry Latency V1

## Scientific decision

**{metadata["scientific_decision"]}**

The exact 2025 post-fill latency test completed, but the registered two-period decision is blocked because none of the original 2023 provider files matched their frozen SHA-256 values. The available 2025 interpretation is **{interpretation}**. This is opened retrospective research, not trading approval.

## 1. Exact hypothesis and prior boundary

The hypothesis is deliberately narrow: for the same immutable named-loop opportunity, frozen OCO direction, costs, and original terminal, T1 waits until the actual T0 breakout-fill bar completes and enters at the exact next provider open. No price sign, anchor veto, range rule, payoff-state gate, or fitted model is used.

This exact T0/T1 comparison had not previously been tested. Clean Anchor Price Acceptance V1 used `anchor+10m` for every opportunity. The source T0 is instead `anchor + 5 * entry_step` and `entry_step` ranges from 1 to 24. In 2025, 206 of 809 named rows have `entry_step > 1`; on those rows the Clean Anchor clock was not one bar after T0 and could precede the causal breakout direction. Its +27,936.20 bps level is therefore context, not the registered latency result.

## 2. Scientific status, frozen populations, and controls

- `cycle_04|state_4`: 2023={counts.get((2023, "cycle_04", "state_4"), 0)}, 2025={counts.get((2025, "cycle_04", "state_4"), 0)}.
- `cycle_07|state_5`: 2023={counts.get((2023, "cycle_07", "state_5"), 0)}, 2025={counts.get((2025, "cycle_07", "state_5"), 0)}.
- Controls remain separate: `cycle_04|state_2` and `cycle_07|state_6`.
- No failed loop is replaced; overlap and capacity are never refilled.

## 3. Source identity and exact clocks

T0 is the stored V2 `no_payoff_state_filter` OCO fill. Its timestamp is the start of the trigger bar; its price is the frozen breakout threshold or opening gap fill, not necessarily the provider open. T1 is exactly `T0 timestamp + 5 minutes`, at that provider bar's open. T2 is `T0 + 10 minutes` and remains a secondary shape diagnostic. All primary rows retain the stored terminal `anchor + 125 minutes`, priced at the close of the provider bar beginning five minutes earlier. Entry and exit each cost 5 bps.

The exact source and delayed rows are paired by opportunity, anchor, event lineage, symbol, session, loop, orientation, direction, and terminal. Missing T1 rows remain explicit; T0 metrics below are recomputed only on exact pairs.

## 4. 2023 archival status and paired population

The expired root `{missing_2023["expired_original_root"]}` does not exist. The pre-score archival search hashed {missing_2023["pre_score_candidate_files_hashed"]} candidate files and found {missing_2023["pre_score_exact_hash_matches"]} matches across {missing_2023["required_symbols"]} required symbols. No fresh download, approximate field, or imputation was used. All 854 named 2023 T1 outcomes remain missing.

Across both frozen source periods there are {int(primary["source_opportunities"])} named opportunities and {int(primary["exact_t0_opportunities"])} exact stored T0 rows. The immutable provider evidence yields {int(primary["exact_t1_opportunities"])} exact T1 rows and {int(primary["paired_opportunities"])} pairs ({float(primary["pairing_rate"]):.1%}); every missing 2023 T1 remains explicit rather than becoming a zero.

## 5. Primary T0, T1, and paired result

- T0: {float(primary["t0_net_payoff_bps"]):.2f} bps total, {float(primary["t0_mean_net_payoff_bps"]):.2f} bps per pair.
- T1: {float(primary["t1_net_payoff_bps"]):.2f} bps total, {float(primary["t1_mean_net_payoff_bps"]):.2f} bps per pair.
- T1 minus T0: **{float(primary["paired_total_difference_bps"]):.2f} bps**, mean {float(primary["paired_mean_difference_bps"]):.2f}, median {float(primary["paired_median_difference_bps"]):.2f}.
- Opportunities improved: {float(primary["opportunities_improved_fraction"]):.1%}; sessions improved: {float(primary["sessions_improved_fraction"]):.1%}.
- Five-session-block 95% interval for the session-mean delta: [{float(primary["bootstrap_lower_95_bps"]):.2f}, {float(primary["bootstrap_upper_95_bps"]):.2f}] bps.

## 6. Named-loop results

| named loop | pairs | T0 net bps | T1 net bps | delta bps | mean delta |
|---|---:|---:|---:|---:|---:|
{loop_lines}

## 7. Period and direction results

| period | source rows | exact pairs | T0 paired net bps | T1 net bps | delta bps |
|---|---:|---:|---:|---:|---:|
{period_lines}

| direction | pairs | T0 net bps | T1 net bps | delta bps |
|---|---:|---:|---:|---:|
{direction_lines}

The 2023 row is unavailable by construction and is never interpreted as no effect. The 2025 row is the complete exact archival result currently available.

## 8. Frozen control results

| control orientation | pairs | T0 net bps | T1 net bps | delta bps | mean delta |
|---|---:|---:|---:|---:|---:|
{control_lines}

The named-versus-control comparison determines whether latency is loop-specific or a general execution-clock effect; controls are never pooled into the primary endpoint.

## 9. Entry-price decomposition

The mean direction-adjusted T0-to-T1 entry move is {float(entry["mean_direction_adjusted_entry_move_bps"]):.2f} bps and the median is {float(entry["median_direction_adjusted_entry_move_bps"]):.2f} bps. Negative values mean price moved against the frozen direction and offered a better delayed price. Such adverse moves occur in {int(adverse["opportunities"])} of {int(entry["opportunities"])} pairs ({int(adverse["opportunities"]) / max(1, int(entry["opportunities"])):.1%}). Among T0-profitable rows, {int(profitable_adverse["opportunities"])} of {int(profitable["opportunities"])} ({int(profitable_adverse["opportunities"]) / max(1, int(profitable["opportunities"])):.1%}) move adversely before T1. The entry-move/delta Spearman relationship is {float(entry["move_delta_spearman_rho"]):.3f}; the exact return-convention reconciliation error is checked row by row and audited independently.

This decomposition is diagnostic only. It does not create an inverse acceptance rule or select a subset.

## 10. Costs, T2, and restarted horizon

At frozen costs, paired T0 and T1 levels are {float(cost_primary["t0_net_payoff_bps"]):.2f} and {float(cost_primary["t1_net_payoff_bps"]):.2f} bps, with delta {float(cost_primary["paired_difference_bps"]):.2f}. At twice costs they are {float(twice["t0_net_payoff_bps"]):.2f} and {float(twice["t1_net_payoff_bps"]):.2f}, with unchanged delta {float(twice["paired_difference_bps"]):.2f} because this frozen model charges identical fixed-bps entry and exit costs.

T2 has {int(t2["paired_opportunities"])} exact pairs and leaves a mean {float(t2["mean_exposure_bars_remaining"]):.2f} bars (minimum {int(t2["minimum_exposure_bars_remaining"])}) before the original terminal; T2 minus T0 is {float(t2["t2_minus_t0_bps"]):.2f} bps and T2 minus T1 on the common population is {float(t2["t2_minus_t1_bps"]):.2f} bps. T2 cannot replace the T1 endpoint. Restarted-h24 outcomes are exported separately and never enter the same-terminal conclusion.

## 11. Concentration, deletions, and leave-one-stock-out

The largest stock contributes {float(stock_concentration["top_one_absolute_share"]):.1%} of absolute paired delta and the top five contribute {float(stock_concentration["top_five_absolute_share"]):.1%}; stock HHI is {float(stock_concentration["herfindahl"]):.3f}.

{deletion_lines}

Because the latency rule has no trained or stock-dependent state, a conventional model rebuild is not applicable. `leave_one_stock_out_results.csv` exactly recomputes every remaining deterministic pair after excluding each stock and labels this honestly rather than calling row deletion a trained-model rebuild.

## 12. Failure cases and interpretation

The main failure modes are missing exact T1 opens, T1 at or after the original terminal, absent 2023 provider evidence, period heterogeneity, control replication, and concentration. A positive standalone T1 level is not incremental evidence; only paired T1-minus-T0 counts. Hindsight episodes appear only in concentration and attribution outputs.

The available evidence is classified as **{interpretation}**. The formal result remains **{metadata["scientific_decision"]}** because success required both 2023 and 2025 plus prospective confirmation.

## 13. Exact recommendation

The single most valuable next step is an execution-free prospective cohort using the immutable T0 opportunity ledger, exact create-only T1 timing record, and later separate outcome settlement. Do not deploy or tune the delay. If an original 2023 tape matching the registered per-symbol hashes is recovered, rerun this unchanged contract to close the archival two-period question.

## Reproducibility

- Run ID: `{metadata["run_id"]}`
- Git SHA: `{metadata["git_sha"]}`
- Contract SHA-256: `{metadata["contract_hash"]}`
- Data snapshot SHA-256: `{metadata["data_snapshot_hash"]}`
- Command: `{metadata["command"]}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8", newline="\n")


def _add_population_fields(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    fields = source.loc[
        :,
        [
            "opportunity_id",
            "population_track",
            "population_role",
            "source_run_id",
            "source_artifact_hash",
            "source_opportunity_hash",
        ],
    ]
    return frame.merge(fields, on="opportunity_id", how="left", validate="one_to_one")


def _t2_artifact(all_rows: pd.DataFrame) -> pd.DataFrame:
    rename = {
        column: column.replace("t1_", "t2_", 1)
        for column in all_rows.columns
        if column.startswith("t1_")
    }
    rename["paired_difference_bps"] = "t2_minus_t0_bps"
    return all_rows.rename(columns=rename)


def run_historical(*, output: Path, report_path: Path, exact_rerun_of: Path | None) -> None:
    contract, contract_hash, input_hashes = load_and_verify_contract()
    provider_hashes = verify_provider_hashes(contract)
    data_snapshot_hash = stable_hash(
        {"contract": contract_hash, "inputs": input_hashes, "providers": provider_hashes}
    )
    run_id = "fixed-latency-" + stable_hash(
        {"contract": contract_hash, "data": data_snapshot_hash, "model": MODEL_VERSION},
        length=24,
    )
    named = build_source_population(contract, track="track_a_named_family")
    controls = build_source_population(contract, track="track_b_prior_only")
    if len(named) != 1663 or len(controls) != 641:
        raise AssertionError("frozen source population drift")
    if (
        not named["original_total_cost_bps"].eq(10.0).all()
        or not controls["original_total_cost_bps"].eq(10.0).all()
    ):
        raise AssertionError("source cost-model drift")
    providers = load_provider_frames(contract, pd.concat([named["symbol"], controls["symbol"]]))

    named_expected, named_t1, named_restarted = score_population(named, providers, latency_bars=1)
    control_expected, control_t1, control_restarted = score_population(
        controls, providers, latency_bars=1
    )
    named_all_raw, named_paired_raw, named_unavailable = build_exact_paired_population(
        named, named_t1
    )
    control_all_raw, control_paired_raw, control_unavailable = build_exact_paired_population(
        controls, control_t1
    )
    named_all = attach_evaluation_context(named_all_raw, contract)
    named_paired = named_all.loc[named_all["paired_available"]].reset_index(drop=True)
    control_all = attach_evaluation_context(control_all_raw, contract)
    control_paired = control_all.loc[control_all["paired_available"]].reset_index(drop=True)

    _, named_t2, _ = score_population(named, providers, latency_bars=2)
    named_t2_all, named_t2_paired, _ = build_exact_paired_population(named, named_t2)
    _, control_t2, _ = score_population(controls, providers, latency_bars=2)
    control_t2_all, control_t2_paired, _ = build_exact_paired_population(controls, control_t2)

    context_named = attach_evaluation_context(named, contract)
    context_controls = attach_evaluation_context(controls, contract)
    dimensions = [
        "period",
        "loop_id",
        "orientation",
        "direction_label",
        "symbol",
        "month",
        "quarter_label",
        "clock_phase",
        "anchor_regime",
    ]
    primary_metrics, bootstrap = build_primary_metrics(
        context_named, named_all, named_paired, dimensions=dimensions
    )
    control_metrics, control_bootstrap = build_primary_metrics(
        context_controls, control_all, control_paired, dimensions=dimensions
    )
    full_levels = pd.concat(
        [
            build_full_sample_levels(context_named, named_all).assign(population="named"),
            build_full_sample_levels(context_controls, control_all).assign(population="control"),
        ],
        ignore_index=True,
    )
    decomposition = build_entry_decomposition(named_paired)
    cost_stress = build_cost_stress(named_paired, control_paired)
    concentration = build_concentration(named_paired)
    deletion, loo = build_deletion_stress(named_paired)

    t2_common = named_paired.loc[
        named_paired["opportunity_id"].isin(named_t2_paired["opportunity_id"])
    ][["opportunity_id", "t1_net_return_bps"]].merge(
        named_t2_paired[["opportunity_id", "t1_net_return_bps"]],
        on="opportunity_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_T1", "_T2"),
    )
    t2_summary = paired_summary(named_t2_paired)
    t2_metrics = pd.DataFrame(
        [
            {
                "slice_type": "all",
                "slice_value": "all",
                "source_opportunities": len(named),
                "paired_opportunities": len(named_t2_paired),
                "availability_rate": len(named_t2_paired) / len(named),
                "t2_minus_t0_bps": t2_summary["paired_total_difference_bps"],
                "t2_mean_minus_t0_bps": t2_summary["paired_mean_difference_bps"],
                "t2_minus_t1_bps": float(
                    (t2_common["t1_net_return_bps_T2"] - t2_common["t1_net_return_bps_T1"]).sum()
                ),
                "mean_exposure_bars_remaining": float(
                    named_t2_paired["exposure_bars_remaining"].mean()
                ),
                "minimum_exposure_bars_remaining": int(
                    named_t2_paired["exposure_bars_remaining"].min()
                )
                if len(named_t2_paired)
                else 0,
                "common_T1_T2_opportunities": len(t2_common),
                "primary_endpoint_changed": False,
            }
        ]
    )
    trading_sessions = sorted(
        {
            str(value)
            for frame in providers.values()
            for value in pd.to_datetime(frame["timestamp"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.date.unique()
        }
    )
    nulls = build_nulls(
        named_paired,
        control_paired,
        named_t2_paired,
        trading_sessions=trading_sessions,
    )
    missing_2023 = build_missing_2023_report(contract)
    source_clock_audit = pd.DataFrame(
        [
            {
                "population": population,
                "period": _as_int(period),
                "source_opportunities": int(len(group)),
                "entry_step_one": int(group["entry_step"].eq(1).sum()),
                "entry_step_greater_than_one": int(group["entry_step"].gt(1).sum()),
                "clean_anchor_plus_10_equals_exact_t1": int(
                    (pd.to_datetime(group["anchor_timestamp"], utc=True) + pd.Timedelta(minutes=10))
                    .eq(
                        pd.to_datetime(group["original_entry_timestamp"], utc=True)
                        + pd.Timedelta(minutes=5)
                    )
                    .sum()
                ),
                "clean_anchor_context_is_exact_latency_for_all": False,
            }
            for population, frame in [("named", named), ("control", controls)]
            for period, group in frame.groupby("period", sort=True)
        ]
    )

    t0_columns = [
        "source_run_id",
        "source_artifact_hash",
        "source_opportunity_hash",
        "opportunity_id",
        "anchor_id",
        "event_lineage_id",
        "population_track",
        "population_role",
        "symbol",
        "period",
        "session_date",
        "loop_id",
        "orientation",
        "direction",
        "anchor_timestamp",
        "entry_step",
        "original_entry_timestamp",
        "original_entry_price",
        "original_terminal_timestamp",
        "original_terminal_price",
        "original_gross_payoff_bps",
        "original_total_cost_bps",
        "original_net_payoff_bps",
    ]
    t0_ledger = pd.concat([named[t0_columns], controls[t0_columns]], ignore_index=True)
    t1_expected = pd.concat(
        [
            _add_population_fields(named_expected, named),
            _add_population_fields(control_expected, controls),
        ],
        ignore_index=True,
    )
    all_t1_raw = pd.concat(
        [named_all_raw.assign(population="named"), control_all_raw.assign(population="control")],
        ignore_index=True,
    )
    t1_available = all_t1_raw.loc[all_t1_raw["t1_available"]].copy()
    t1_unavailable = all_t1_raw.loc[~all_t1_raw["t1_available"]].copy()
    restarted = pd.concat(
        [
            _add_population_fields(named_restarted, named).assign(population="named"),
            _add_population_fields(control_restarted, controls).assign(population="control"),
        ],
        ignore_index=True,
    )
    t2_artifact = pd.concat(
        [
            _t2_artifact(named_t2_all).assign(population="named"),
            _t2_artifact(control_t2_all).assign(population="control"),
        ],
        ignore_index=True,
    )

    metadata = {
        "run_id": run_id,
        "git_sha": _git("rev-parse", "HEAD"),
        "repository_branch": _git("branch", "--show-current"),
        "contract_id": contract["contract_id"],
        "contract_hash": contract_hash,
        "data_snapshot_hash": data_snapshot_hash,
        "input_hashes": input_hashes,
        "provider_hashes": provider_hashes,
        "model_version": MODEL_VERSION,
        "feature_schema_version": "fixed_one_bar_entry_latency_v1",
        "fixed_horizon_bars": 24,
        "primary_latency_bars": 1,
        "secondary_latency_bars": 2,
        "cost_bps_per_side": 5.0,
        "random_seed": 20260716,
        "generated_at": RUN_TIMESTAMP,
        "scored_periods": [2023, 2025],
        "provider_2023_status": missing_2023["status"],
        "scientific_status": contract["scientific_status"],
        "scientific_decision": scientific_decision(),
        "command": (
            "PYTHONPATH=packages/stocker_research/src .venv/bin/python "
            "research/slrno-v2/20260714-regime-loop-handoff/work/"
            "run_fixed_one_bar_entry_latency_v1.py --output <OUTPUT> --report <REPORT>"
        ),
        "safety": contract["safety"],
    }
    output.mkdir(parents=True)
    shutil.copyfile(CONTRACT_PATH, output / "frozen_experiment_contract.json")
    detailed = {
        "source_named_opportunity_ledger.parquet": named,
        "source_control_opportunity_ledger.parquet": controls,
        "t0_entry_ledger.parquet": t0_ledger,
        "t1_expected_entry_ledger.parquet": t1_expected,
        "t1_available_entry_ledger.parquet": t1_available,
        "t1_unavailable_ledger.parquet": t1_unavailable,
        "exact_paired_t0_t1_ledger.parquet": named_paired,
        "control_exact_paired_t0_t1_ledger.parquet": control_paired,
        "restarted_h24_diagnostic_ledger.parquet": restarted,
        "t2_sensitivity_ledger.parquet": t2_artifact,
        "entry_price_decomposition.parquet": named_paired.loc[
            :,
            [
                "opportunity_id",
                "anchor_id",
                "event_lineage_id",
                "period",
                "session_date",
                "symbol",
                "loop_id",
                "orientation",
                "direction",
                "t0_entry_timestamp",
                "t0_entry_price",
                "t1_entry_timestamp",
                "t1_entry_price",
                "original_terminal_timestamp",
                "terminal_price",
                "direction_adjusted_entry_move_bps",
                "exact_entry_price_effect_bps",
                "paired_difference_bps",
                "reconciliation_error_bps",
                "first_bar_signed_close_bps",
                "first_bar_favourable_excursion_bps",
                "first_bar_adverse_excursion_bps",
            ],
        ],
    }
    summaries = {
        "primary_paired_metrics.csv": primary_metrics,
        "session_block_bootstrap_metrics.csv": bootstrap,
        "period_breakdowns.csv": primary_metrics.loc[primary_metrics["slice_type"].eq("period")],
        "named_loop_breakdowns.csv": primary_metrics.loc[
            primary_metrics["slice_type"].eq("loop_id")
        ],
        "control_breakdowns.csv": control_metrics,
        "direction_breakdowns.csv": primary_metrics.loc[
            primary_metrics["slice_type"].eq("direction_label")
        ],
        "entry_price_decomposition_summary.csv": decomposition,
        "cost_stress_results.csv": cost_stress,
        "t2_sensitivity_metrics.csv": t2_metrics,
        "deletion_stress_results.csv": deletion,
        "leave_one_stock_out_results.csv": loo,
        "concentration_results.csv": concentration,
        "full_sample_levels.csv": full_levels,
        "null_test_results.csv": nulls,
        "source_clock_audit.csv": source_clock_audit,
        "control_session_block_bootstrap_metrics.csv": control_bootstrap,
    }
    for filename, frame in detailed.items():
        write_parquet(
            output / filename,
            annotate(
                frame,
                run_id=run_id,
                contract_hash=contract_hash,
                data_snapshot_hash=data_snapshot_hash,
            ),
        )
    for filename, frame in summaries.items():
        write_csv(
            output / filename,
            annotate(
                frame,
                run_id=run_id,
                contract_hash=contract_hash,
                data_snapshot_hash=data_snapshot_hash,
            ),
        )
    write_json(output / "missing_2023_input_report.json", missing_2023)
    write_json(output / "prospective_immutable_ledger_schema.json", prospective_schema())
    plot_paths = make_plots(output, named_paired, concentration)
    metadata["artifact_names"] = sorted([*detailed, *summaries])
    metadata["plot_names"] = sorted(path.name for path in plot_paths)
    write_json(output / "run_metadata.json", metadata)
    if exact_rerun_of is not None:
        identity = verify_exact_rerun(output, exact_rerun_of)
        write_json(output / "exact_rerun_identity.json", identity)
        if not bool(identity["byte_identical"]):
            raise AssertionError(f"exact rerun failed: {identity}")
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    write_report(
        report_path,
        metadata=metadata,
        named_source=named,
        primary_metrics=primary_metrics,
        control_metrics=control_metrics,
        decomposition=decomposition,
        cost_stress=cost_stress,
        t2_metrics=t2_metrics,
        deletion=deletion,
        concentration=concentration,
        missing_2023=missing_2023,
    )


def run_prospective(args: argparse.Namespace) -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if args.prospective_root is None or args.record_json is None:
        raise ValueError("prospective mode requires --prospective-root and --record-json")
    ledger = ProspectiveLatencyLedger(
        Path(args.prospective_root),
        opened_periods=set(contract["prospective_logging"]["opened_periods_forbidden_in_holdout"]),
    )
    record = json.loads(Path(args.record_json).read_text(encoding="utf-8"))
    if args.mode == "prospective-opportunity":
        path = ledger.append_opportunity(record, holdout=bool(args.holdout))
    elif args.mode == "prospective-timing":
        path = ledger.append_timing(record)
    else:
        path = ledger.append_outcome(record)
    print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "historical",
            "prospective-opportunity",
            "prospective-timing",
            "prospective-settle",
        ],
        default="historical",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--exact-rerun-of", type=Path)
    parser.add_argument("--prospective-root", type=Path)
    parser.add_argument("--record-json", type=Path)
    parser.add_argument("--holdout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "historical":
        run_historical(
            output=Path(args.output),
            report_path=Path(args.report),
            exact_rerun_of=Path(args.exact_rerun_of) if args.exact_rerun_of else None,
        )
    else:
        run_prospective(args)


if __name__ == "__main__":
    main()
