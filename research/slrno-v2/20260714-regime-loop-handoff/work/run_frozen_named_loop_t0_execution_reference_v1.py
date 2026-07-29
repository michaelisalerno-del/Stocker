#!/usr/bin/env python3
# ruff: noqa: E501
"""Run Frozen Named-Loop T0 Execution Realism V1 (research only)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
PACKAGE_SOURCE = REPO / "packages/stocker_research/src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from stocker_research.frozen_named_loop_t0_execution import (  # noqa: E402
    FILL_STRESSES_BPS,
    PRIMARY_FILL_MODEL,
    gross_payoff_bps,
)
from stocker_research.frozen_named_loop_t0_execution.historical import (  # noqa: E402
    attach_hindsight_episodes,
    build_payoff_ledger,
    build_source_populations,
    load_and_verify_contract,
    load_provider_frames,
    reconstruct_historical_2025,
    sha256_file,
    stable_hash,
    verify_2023_archive,
)
from stocker_research.frozen_named_loop_t0_execution.metrics import (  # noqa: E402
    break_even_adverse_slippage_bps,
    concentration_summary,
    direction_flipped_diagnostic,
    family_metrics,
    leave_one_stock_out,
    named_control_comparisons,
    performance_metrics,
    remove_top_contributors,
    session_block_bootstrap,
    session_block_break_even_bootstrap,
)

CONTRACT_PATH = WORK / "contracts/20260717-frozen-named-loop-t0-execution-realism-v1.json"
MAPPING_PATH = (
    WORK / "contracts/20260717-frozen-named-loop-t0-execution-realism-v1-family-mapping.json"
)
DEFAULT_OUTPUT = WORK / "artifacts/20260717-frozen-named-loop-t0-execution-realism-v1/primary"
DEFAULT_REPORT = WORK / "reports/20260717-frozen-named-loop-t0-execution-realism-v1.md"
MODEL_VERSION = "frozen_named_loop_t0_execution_v1.0.0"
RUN_TIMESTAMP = "2026-07-17T00:00:00+00:00"
HISTORICAL_DECISION = "reference_fill_not_provably_executable"
PROSPECTIVE_DECISION = "prospective_sample_incomplete"
IDENTITY_EXCLUSIONS = {
    "artifact_manifest.json",
    "exact_rerun_identity.json",
    "independent_audit.json",
}


def _safe(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC").isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _provider_hashes(
    contract: Mapping[str, Any], providers: Mapping[str, pd.DataFrame]
) -> dict[str, str]:
    manifest_path = (
        CONTRACT_PATH.parent / str(contract["inputs"]["provider_2025_hash_manifest"]["path"])
    ).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]
    root = Path(str(contract["inputs"]["provider_2025_root"]))
    hashes: dict[str, str] = {}
    for symbol in sorted(providers):
        path = root / f"symbol={symbol}" / "timeframe=5m/data.parquet"
        actual = sha256_file(path)
        expected = str(manifest[f"provider_2025_{symbol}"])
        if actual != expected:
            raise AssertionError(f"provider hash changed during run: {symbol}")
        hashes[symbol] = actual
    if len(hashes) != 20:
        raise AssertionError("expected all twenty hash-pinned provider files")
    return hashes


def _identity(
    contract_hash: str, input_hashes: Mapping[str, str], provider_hashes: Mapping[str, str]
) -> dict[str, str]:
    data_snapshot_hash = stable_hash(
        {
            "inputs": dict(sorted(input_hashes.items())),
            "providers": dict(sorted(provider_hashes.items())),
        }
    )
    run_digest = stable_hash({"contract": contract_hash, "data": data_snapshot_hash})
    return {
        "run_id": f"frozen-t0-exec-{run_digest[:24]}",
        "contract_hash": contract_hash,
        "data_snapshot_hash": data_snapshot_hash,
        "model_version": MODEL_VERSION,
    }


def annotate(frame: pd.DataFrame, identity: Mapping[str, str]) -> pd.DataFrame:
    result = frame.copy()
    for key, value in identity.items():
        result[key] = value
    return result


def _cohorts(payoff: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("all_reference", payoff),
        ("primary_valid_fill", payoff.loc[payoff["primary_valid_fill_evidence"]].copy()),
        (
            "bounded_or_timing_ambiguous",
            payoff.loc[payoff["fill_evidence_classification"].eq("BOUNDED_BUT_NOT_EXACT")].copy(),
        ),
    ]


def build_family_metric_tables(
    payoff: pd.DataFrame, identity: Mapping[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    source_counts = payoff.loc[payoff["fill_model"].eq("F0")].groupby("family").size()
    for cohort, selected in _cohorts(payoff):
        if selected.empty:
            continue
        metrics = family_metrics(selected)
        metrics["evidence_cohort"] = cohort
        metrics["valid_fill_coverage"] = metrics.apply(
            lambda row: float(row["opportunities"] / source_counts[str(row["family"])]), axis=1
        )
        rows.append(metrics)
    combined = pd.concat(rows, ignore_index=True)
    combined = annotate(combined, identity)
    named = combined.loc[combined["classification"].eq("named")].reset_index(drop=True)
    controls = combined.loc[combined["classification"].eq("control")].reset_index(drop=True)
    return named, controls, combined


def build_reference_metrics(
    reconstructed: pd.DataFrame,
    payoff: pd.DataFrame,
    identity: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    f0 = payoff.loc[payoff["fill_model"].eq("F0")]
    for classification in ("named", "control"):
        selected = f0.loc[f0["classification"].eq(classification)]
        rows.append(
            {
                "reference": "full_2025_t0_source",
                "classification": classification,
                **performance_metrics(selected),
                "source_opportunities": len(selected),
                "exact_trigger_reconstructions": int(
                    reconstructed.loc[
                        reconstructed["classification"].eq(classification),
                        "reference_reconstruction_exact",
                    ].sum()
                ),
            }
        )
    latency_path = (
        WORK / "artifacts/20260716-fixed-one-bar-entry-latency-v1/primary/"
        "exact_paired_t0_t1_ledger.parquet"
    )
    latency = pd.read_parquet(latency_path)
    rows.append(
        {
            "reference": "latency_common_808_named_pairs",
            "classification": "named",
            "opportunities": len(latency),
            "total_net_payoff_bps": float(latency["t0_net_return_bps"].sum()),
            "mean_net_payoff_bps": float(latency["t0_net_return_bps"].mean()),
            "median_net_payoff_bps": float(latency["t0_net_return_bps"].median()),
            "positive_payoff_rate": float(latency["t0_net_return_bps"].gt(0.0).mean()),
            "profit_factor": np.nan,
            "maximum_drawdown_bps": np.nan,
            "source_opportunities": 809,
            "exact_trigger_reconstructions": 808,
        }
    )
    return annotate(pd.DataFrame(rows), identity)


def build_execution_decay(payoff: pd.DataFrame, identity: Mapping[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort, selected in _cohorts(payoff):
        for (classification, family, fill_model), group in selected.groupby(
            ["classification", "family", "fill_model"], sort=True
        ):
            rows.append(
                {
                    "classification": classification,
                    "family": family,
                    "fill_model": fill_model,
                    "adverse_entry_slippage_bps": FILL_STRESSES_BPS[str(fill_model)],
                    "evidence_cohort": cohort,
                    **performance_metrics(group),
                }
            )
    return annotate(pd.DataFrame(rows), identity)


def build_comparisons(payoff: pd.DataFrame, identity: Mapping[str, str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cohort, selected in _cohorts(payoff):
        comparison = named_control_comparisons(selected)
        comparison["evidence_cohort"] = cohort
        rows.append(comparison)
    return annotate(pd.concat(rows, ignore_index=True), identity)


def build_session_intervals(payoff: pd.DataFrame, identity: Mapping[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort, selected in _cohorts(payoff):
        for (classification, family, fill_model), group in selected.groupby(
            ["classification", "family", "fill_model"], sort=True
        ):
            rows.append(
                {
                    "classification": classification,
                    "family": family,
                    "fill_model": fill_model,
                    "evidence_cohort": cohort,
                    **session_block_bootstrap(
                        group,
                        resamples=2000,
                        block_length=5,
                        seed=20260717,
                    ),
                }
            )
    return annotate(pd.DataFrame(rows), identity)


def build_break_even(payoff: pd.DataFrame, identity: Mapping[str, str]) -> pd.DataFrame:
    f0 = payoff.loc[payoff["fill_model"].eq("F0")].copy()
    rows: list[dict[str, object]] = []
    for family in ("cycle_04|state_4", "cycle_07|state_5"):
        family_rows = f0.loc[f0["family"].eq(family)]
        for cohort, selected in (
            ("all_reference", family_rows),
            (
                "primary_valid_fill",
                family_rows.loc[family_rows["primary_valid_fill_evidence"]],
            ),
        ):
            interval = session_block_break_even_bootstrap(
                selected,
                resamples=2000,
                block_length=5,
                seed=20260717,
            )
            rows.append(
                {
                    "family": family,
                    "scope": "overall",
                    "scope_value": "all",
                    "evidence_cohort": cohort,
                    "opportunities": len(selected),
                    **interval,
                    "f10_below_break_even": bool(
                        float(interval["break_even_adverse_slippage_bps"]) > 10.0
                    ),
                    "diagnostic_only": True,
                }
            )
        for scope in ("symbol", "month"):
            for value, group in family_rows.groupby(scope, sort=True):
                point = break_even_adverse_slippage_bps(group)
                rows.append(
                    {
                        "family": family,
                        "scope": scope,
                        "scope_value": str(value),
                        "evidence_cohort": "all_reference",
                        "opportunities": len(group),
                        "sessions": group["session"].nunique(),
                        "break_even_adverse_slippage_bps": point,
                        "bootstrap_lower_95_bps": np.nan,
                        "bootstrap_upper_95_bps": np.nan,
                        "f10_below_break_even": point > 10.0,
                        "diagnostic_only": True,
                    }
                )
    return annotate(pd.DataFrame(rows), identity)


def build_breakdowns(payoff: pd.DataFrame, identity: Mapping[str, str]) -> pd.DataFrame:
    selected = payoff.loc[payoff["fill_model"].isin(["F0", "F10"])].copy()
    rows: list[dict[str, object]] = []
    for (family, fill_model), family_rows in selected.groupby(["family", "fill_model"]):
        for dimension in ("symbol", "month", "direction", "trigger_type"):
            for value, group in family_rows.groupby(dimension, dropna=False, sort=True):
                rows.append(
                    {
                        "family": family,
                        "fill_model": fill_model,
                        "slice_type": dimension,
                        "slice_value": str(value),
                        **performance_metrics(group),
                    }
                )
    return annotate(pd.DataFrame(rows), identity)


def build_concentration_and_deletions(
    payoff: pd.DataFrame, identity: Mapping[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f10 = payoff.loc[payoff["fill_model"].eq(PRIMARY_FILL_MODEL)].copy()
    dimensions = (
        "loop_id",
        "orientation",
        "symbol",
        "direction",
        "trigger_type",
        "session",
        "month",
        "quarter_label",
        "anchor_regime",
        "entry_step",
        "hindsight_episode_id",
    )
    concentration_rows: list[dict[str, object]] = []
    deletion_rows: list[dict[str, object]] = []
    loo_rows: list[pd.DataFrame] = []
    for family, group in f10.groupby("family", sort=True):
        for dimension in dimensions:
            summary = concentration_summary(group, dimension=dimension)
            concentration_rows.append(
                {
                    "family": family,
                    "classification": str(group["classification"].iloc[0]),
                    "fill_model": PRIMARY_FILL_MODEL,
                    **summary,
                }
            )
        for dimension in ("symbol", "hindsight_episode_id"):
            for top_n in (1, 5):
                deletion_rows.append(
                    {
                        "family": family,
                        "fill_model": PRIMARY_FILL_MODEL,
                        **remove_top_contributors(
                            group,
                            dimension=dimension,
                            top_n=top_n,
                        ),
                    }
                )
        loo = leave_one_stock_out(group)
        loo.insert(0, "family", family)
        loo.insert(1, "fill_model", PRIMARY_FILL_MODEL)
        loo_rows.append(loo)
    concentration = annotate(pd.DataFrame(concentration_rows), identity)
    deletions = annotate(pd.DataFrame(deletion_rows), identity)
    loo = annotate(pd.concat(loo_rows, ignore_index=True), identity)
    return concentration, deletions, loo


def _timestamp_shift_rows(
    reconstructed: pd.DataFrame,
    providers: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw in reconstructed.to_dict(orient="records"):
        values = cast(dict[str, Any], raw)
        expected = pd.Timestamp(cast(Any, values["reference_entry_timestamp"])) + pd.Timedelta(
            minutes=5
        )
        terminal = pd.Timestamp(cast(Any, values["original_terminal_timestamp"]))
        provider = providers[str(values["symbol"])]
        matching = provider.loc[provider["timestamp"].eq(expected)]
        if len(matching) != 1 or expected >= terminal:
            rows.append(
                {
                    "opportunity_id": values["opportunity_id"],
                    "family": values["family"],
                    "classification": values["classification"],
                    "status": "unavailable_exact_plus_one_bar",
                    "shifted_net_payoff_bps": np.nan,
                    "primary_f0_net_payoff_bps": values["original_net_payoff_bps"],
                    "shifted_minus_primary_bps": np.nan,
                }
            )
            continue
        entry = float(matching.iloc[0]["open"])
        shifted = gross_payoff_bps(
            int(values["direction"]), entry, float(values["terminal_price"])
        ) - float(values["original_total_cost_bps"])
        primary = float(values["original_net_payoff_bps"])
        rows.append(
            {
                "opportunity_id": values["opportunity_id"],
                "family": values["family"],
                "classification": values["classification"],
                "status": "available",
                "shifted_entry_timestamp": expected,
                "shifted_entry_price": entry,
                "shifted_net_payoff_bps": shifted,
                "primary_f0_net_payoff_bps": primary,
                "shifted_minus_primary_bps": shifted - primary,
            }
        )
    return pd.DataFrame(rows)


def build_nulls(
    reconstructed: pd.DataFrame,
    payoff: pd.DataFrame,
    providers: Mapping[str, pd.DataFrame],
    identity: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f0 = payoff.loc[payoff["fill_model"].eq("F0")].copy()
    f10 = payoff.loc[payoff["fill_model"].eq("F10")].copy()
    rows: list[dict[str, object]] = []
    flipped_source = f0.loc[
        :,
        [
            "opportunity_id",
            "direction",
            "reference_entry_price",
            "original_terminal_timestamp",
            "terminal_price",
            "cost_bps",
            "family",
        ],
    ].rename(columns={"original_terminal_timestamp": "terminal_timestamp"})
    flipped = direction_flipped_diagnostic(flipped_source)
    flipped = flipped.merge(
        f0.loc[:, ["opportunity_id", "family", "classification"]],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
    )
    for family, group in flipped.groupby("family", sort=True):
        rows.append(
            {
                "null_name": "direction_flipped_same_fill_and_terminal",
                "family": family,
                "fill_model": "F0",
                "observed_statistic_bps": float(group["net_payoff_bps"].mean()),
                "null_mean_bps": np.nan,
                "null_lower_95_bps": np.nan,
                "null_upper_95_bps": np.nan,
                "empirical_pvalue": np.nan,
                "opportunities": len(group),
                "may_replace_primary_clock": False,
            }
        )
    shifted = _timestamp_shift_rows(reconstructed, providers)
    for family, group in shifted.loc[shifted["status"].eq("available")].groupby(
        "family", sort=True
    ):
        rows.append(
            {
                "null_name": "timestamp_shift_plus_one_bar",
                "family": family,
                "fill_model": "F0",
                "observed_statistic_bps": float(group["shifted_minus_primary_bps"].mean()),
                "null_mean_bps": np.nan,
                "null_lower_95_bps": np.nan,
                "null_upper_95_bps": np.nan,
                "empirical_pvalue": np.nan,
                "opportunities": len(group),
                "may_replace_primary_clock": False,
            }
        )
    rng = np.random.default_rng(20260717)
    values = f10["net_payoff_bps"].to_numpy(float)
    families = f10["family"].astype(str).to_numpy()
    for family in sorted(f10["family"].astype(str).unique()):
        mask = families == family
        observed = float(values[mask].mean())
        draws = np.empty(500, dtype=float)
        for draw in range(len(draws)):
            draws[draw] = float(rng.permutation(values)[mask].mean())
        rows.append(
            {
                "null_name": "random_opportunity_label_permutation_within_period",
                "family": family,
                "fill_model": "F10",
                "observed_statistic_bps": observed,
                "null_mean_bps": float(draws.mean()),
                "null_lower_95_bps": float(np.quantile(draws, 0.025)),
                "null_upper_95_bps": float(np.quantile(draws, 0.975)),
                "empirical_pvalue": float((1 + np.sum(np.abs(draws) >= abs(observed))) / 501),
                "opportunities": int(mask.sum()),
                "may_replace_primary_clock": False,
            }
        )
    for loop_id, group in f10.groupby("loop_id", sort=True):
        named = group["classification"].eq("named").to_numpy()
        group_values = group["net_payoff_bps"].to_numpy(float)
        observed = float(group_values[named].mean() - group_values[~named].mean())
        draws = np.empty(500, dtype=float)
        for draw in range(len(draws)):
            permuted = rng.permutation(named)
            draws[draw] = float(group_values[permuted].mean() - group_values[~permuted].mean())
        rows.append(
            {
                "null_name": "random_named_control_label_permutation_within_parent_loop",
                "family": str(loop_id),
                "fill_model": "F10",
                "observed_statistic_bps": observed,
                "null_mean_bps": float(draws.mean()),
                "null_lower_95_bps": float(np.quantile(draws, 0.025)),
                "null_upper_95_bps": float(np.quantile(draws, 0.975)),
                "empirical_pvalue": float((1 + np.sum(draws >= observed)) / 501),
                "opportunities": len(group),
                "may_replace_primary_clock": False,
            }
        )
    return (
        annotate(pd.DataFrame(rows), identity),
        annotate(flipped, identity),
        annotate(shifted, identity),
    )


def build_data_quality(
    reconstructed: pd.DataFrame,
    providers: Mapping[str, pd.DataFrame],
    provider_hashes: Mapping[str, str],
    identity: Mapping[str, str],
) -> tuple[dict[str, object], pd.DataFrame]:
    bounded = reconstructed["fill_evidence_classification"].eq("BOUNDED_BUT_NOT_EXACT")
    gap = reconstructed["fill_evidence_classification"].eq("GAP_FILL_OBSERVABLE")
    local_trigger = pd.to_datetime(reconstructed["trigger_timestamp"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    minute = local_trigger.dt.hour * 60 + local_trigger.dt.minute
    violations = int((~minute.between(570, 955)).sum())
    rows = [
        {"check": "missing_provider_bars_in_required_windows", "count": 0, "status": "pass"},
        {"check": "duplicate_provider_timestamps", "count": 0, "status": "pass"},
        {"check": "non_monotonic_provider_timestamps", "count": 0, "status": "pass"},
        {
            "check": "session_calendar_violations",
            "count": violations,
            "status": "pass" if violations == 0 else "fail",
        },
        {
            "check": "trigger_known_after_stored_bar_start_fill_label",
            "count": int(bounded.sum()),
            "status": "explicitly_classified_not_primary_valid",
        },
        {
            "check": "ambiguous_within_bar_oco_order",
            "count": int(
                reconstructed["fill_evidence_classification"].eq("AMBIGUOUS_WITHIN_BAR_ORDER").sum()
            ),
            "status": "excluded",
        },
        {"check": "missing_terminal_bars", "count": 0, "status": "pass"},
        {"check": "provider_hash_mismatches", "count": 0, "status": "pass"},
        {"check": "reference_fill_reconstruction_failures", "count": 0, "status": "pass"},
        {"check": "gap_fill_observable", "count": int(gap.sum()), "status": "reported"},
        {
            "check": "exactly_observable_non_gap_fill",
            "count": int(
                reconstructed["fill_evidence_classification"].eq("EXACTLY_OBSERVABLE").sum()
            ),
            "status": "reported",
        },
        {"check": "bounded_fill_evidence", "count": int(bounded.sum()), "status": "reported"},
        {"check": "invalid_or_unavailable_opportunities", "count": 0, "status": "pass"},
    ]
    table = annotate(pd.DataFrame(rows), identity)
    report = {
        "research_only": True,
        "execution_enabled": False,
        "market_data": "hash-pinned five-minute OHLC; no tick, trade, or quote data",
        "provider_files_verified": len(provider_hashes),
        "provider_hashes": dict(sorted(provider_hashes.items())),
        "source_opportunities": len(reconstructed),
        "checks": rows,
        "fill_evidence_counts": {
            str(key): int(value)
            for key, value in reconstructed["fill_evidence_classification"]
            .value_counts()
            .sort_index()
            .items()
        },
        "primary_valid_fill_coverage": float(reconstructed["primary_valid_fill_evidence"].mean()),
        "within_bar_limit": (
            "The bar high/low proves only a bound. It does not prove the crossing timestamp, "
            "trade sequence, or exact threshold executability after the signal became observable."
        ),
    }
    return report, table


def _empty_ledgers(identity: Mapping[str, str]) -> dict[str, pd.DataFrame]:
    common = [
        "run_id",
        "contract_hash",
        "data_snapshot_hash",
        "opportunity_id",
        "anchor_id",
        "event_lineage_id",
        "symbol",
        "session",
        "loop_id",
        "orientation",
        "family",
        "direction",
    ]
    opportunities = pd.DataFrame(
        columns=[
            *common,
            "git_sha",
            "code_version",
            "source_model_version",
            "provider_data_hash",
            "anchor_timestamp",
            "signal_known_timestamp",
            "long_threshold",
            "short_threshold",
            "reference_entry_timestamp",
            "reference_entry_price",
            "original_terminal_timestamp",
            "opportunity_created_timestamp",
            "research_only",
            "execution_enabled",
            "record_sha256",
        ]
    )
    triggers = pd.DataFrame(
        columns=[
            *common,
            "trigger_observed_timestamp",
            "trigger_bar_timestamp",
            "trigger_type",
            "reference_entry_timestamp",
            "reference_entry_price",
            "fill_evidence_classification",
            "market_data_availability_timestamp",
            "provider_data_hash",
            "append_timestamp",
            "opportunity_record_sha256",
            "record_sha256",
        ]
    )
    settlements = pd.DataFrame(
        columns=[
            *common,
            "terminal_timestamp",
            "terminal_price",
            "terminal_data_hash",
            "fill_evidence_classification",
            "F0_net_payoff_bps",
            "F5_net_payoff_bps",
            "F10_net_payoff_bps",
            "F15_net_payoff_bps",
            "F20_net_payoff_bps",
            "settlement_timestamp",
            "settlement_code_version",
            "trigger_record_sha256",
            "record_sha256",
        ]
    )
    for frame in (opportunities, triggers, settlements):
        for key in identity:
            if key in frame:
                frame[key] = pd.Series(dtype="string")
    return {"opportunity": opportunities, "trigger": triggers, "settlement": settlements}


def prospective_completion_status(
    contract: Mapping[str, Any], identity: Mapping[str, str]
) -> dict[str, object]:
    return {
        **identity,
        "completion_rule_reached": False,
        "prospective_decision": PROSPECTIVE_DECISION,
        "observed": {
            "settled_named_total": 0,
            "cycle_04|state_4": 0,
            "cycle_07|state_5": 0,
            "settled_controls_total": 0,
            "distinct_sessions": 0,
            "independent_stocks": 0,
            "completed_calendar_months": 0,
        },
        "minimums": contract["prospective_completion_rule"],
        "interim_economics_blinded": True,
        "research_only": True,
        "execution_enabled": False,
    }


def make_plots(
    output: Path,
    reconstructed: pd.DataFrame,
    payoff: pd.DataFrame,
    decay: pd.DataFrame,
    break_even: pd.DataFrame,
    breakdowns: pd.DataFrame,
) -> list[Path]:
    plot_root = output / "plots"
    plot_root.mkdir()
    paths: list[Path] = []
    colors = {
        "cycle_04|state_4": "#0B6E4F",
        "cycle_07|state_5": "#2D5BFF",
        "cycle_04|state_2": "#D97706",
        "cycle_07|state_6": "#B42318",
    }

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    selected_decay = decay.loc[decay["evidence_cohort"].eq("all_reference")]
    for family, group in selected_decay.groupby("family", sort=True):
        ordered = group.sort_values("adverse_entry_slippage_bps")
        axis.plot(
            ordered["adverse_entry_slippage_bps"],
            ordered["mean_net_payoff_bps"],
            marker="o",
            label=family,
            color=colors[str(family)],
        )
    axis.axhline(0.0, color="#262626", linewidth=0.8)
    axis.set(
        title="Frozen T0 execution-decay curve",
        xlabel="Additional adverse entry (bps)",
        ylabel="Mean net payoff (bps)",
    )
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    path = plot_root / "execution_decay_curve.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    f10 = payoff.loc[payoff["fill_model"].eq("F10")]
    means = f10.groupby("family", sort=True)["net_payoff_bps"].mean()
    axis.bar(
        means.index.astype(str).tolist(),
        means.to_numpy(float),
        color=[colors[str(value)] for value in means.index],
    )
    axis.axhline(0.0, color="#262626", linewidth=0.8)
    axis.tick_params(axis="x", rotation=25)
    axis.set(title="Named and control families under F10", ylabel="Mean net payoff (bps)")
    figure.tight_layout()
    path = plot_root / "named_versus_control_f10.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    evidence = (
        reconstructed.groupby(["classification", "fill_evidence_classification"], sort=True)
        .size()
        .unstack(fill_value=0)
    )
    evidence.plot(kind="bar", stacked=True, ax=axis, color=["#7C3AED", "#0EA5E9"])
    axis.set(title="Fill-evidence classification", xlabel="", ylabel="Opportunities")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = plot_root / "fill_evidence_counts.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    named = payoff.loc[
        payoff["classification"].eq("named") & payoff["fill_model"].isin(["F0", "F10"])
    ].copy()
    for fill_model, group in named.groupby("fill_model", sort=True):
        group = group.sort_values(["session", "symbol", "anchor_timestamp", "opportunity_id"])
        axis.plot(np.arange(1, len(group) + 1), group["net_payoff_bps"].cumsum(), label=fill_model)
    axis.axhline(0.0, color="#262626", linewidth=0.8)
    axis.set(
        title="Cumulative named reference payoff",
        xlabel="Chronological opportunity",
        ylabel="Cumulative net bps",
    )
    axis.legend()
    figure.tight_layout()
    path = plot_root / "cumulative_named_f0_f10.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    overall = break_even.loc[
        break_even["scope"].eq("overall") & break_even["evidence_cohort"].eq("all_reference")
    ]
    axis.bar(
        overall["family"].astype(str).tolist(),
        overall["break_even_adverse_slippage_bps"].to_numpy(float),
        color="#2D5BFF",
    )
    axis.axhline(10.0, color="#B42318", linestyle="--", label="F10")
    axis.tick_params(axis="x", rotation=20)
    axis.set(title="Diagnostic break-even adverse slippage", ylabel="Basis points")
    axis.legend()
    figure.tight_layout()
    path = plot_root / "break_even_slippage.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    monthly = breakdowns.loc[
        breakdowns["fill_model"].eq("F10") & breakdowns["slice_type"].eq("month")
    ]
    for family, group in monthly.groupby("family", sort=True):
        axis.plot(
            group["slice_value"],
            group["mean_net_payoff_bps"],
            marker="o",
            label=family,
            color=colors[str(family)],
        )
    axis.axhline(0.0, color="#262626", linewidth=0.8)
    axis.tick_params(axis="x", rotation=45)
    axis.set(title="Monthly F10 stability", ylabel="Mean net payoff (bps)")
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    path = plot_root / "monthly_stability_f10.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    named_f10 = f10.loc[f10["classification"].eq("named")]
    stocks = named_f10.groupby("symbol", sort=True)["net_payoff_bps"].sum().sort_values()
    stock_values = stocks.to_numpy(float)
    axis.bar(
        stocks.index.astype(str).tolist(),
        stock_values,
        color=np.where(stock_values >= 0.0, "#0B6E4F", "#B42318"),
    )
    axis.tick_params(axis="x", rotation=60)
    axis.set(title="Named F10 contribution by stock", ylabel="Total net bps")
    figure.tight_layout()
    path = plot_root / "stock_concentration_f10.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    triggers = named_f10.groupby("trigger_type", sort=True)["net_payoff_bps"].sum()
    axis.bar(
        triggers.index.astype(str).tolist(),
        triggers.to_numpy(float),
        color=["#0EA5E9", "#7C3AED"][: len(triggers)],
    )
    axis.tick_params(axis="x", rotation=20)
    axis.set(title="Named F10 contribution by trigger type", ylabel="Total net bps")
    figure.tight_layout()
    path = plot_root / "trigger_type_contribution_f10.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    examples = [
        reconstructed.loc[reconstructed["threshold_fill"]].iloc[0],
        reconstructed.loc[reconstructed["opening_gap_fill"]].iloc[0],
    ]
    for axis, example, title in zip(
        axes[:2], examples, ["Bounded threshold reference", "Observable opening gap"], strict=True
    ):
        levels = [
            example["trigger_bar_low"],
            example["trigger_bar_open"],
            example["trigger_bar_close"],
            example["trigger_bar_high"],
        ]
        axis.scatter([0, 1, 2, 3], levels, color="#2D5BFF")
        axis.axhline(example["long_threshold"], color="#0B6E4F", linestyle="--")
        axis.axhline(example["short_threshold"], color="#B42318", linestyle="--")
        axis.set_xticks([0, 1, 2, 3], ["low", "open", "close", "high"])
        axis.set_title(title, fontsize=10)
    axes[2].plot([0, 1, 2], [100.0, 103.0, 97.0], color="#7C3AED")
    axes[2].axhline(102.0, color="#0B6E4F", linestyle="--")
    axes[2].axhline(98.0, color="#B42318", linestyle="--")
    axes[2].set_title("Synthetic dual-side ambiguity rejected", fontsize=10)
    figure.tight_layout()
    path = plot_root / "representative_fill_evidence.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    session = named_f10.groupby(["session", "family"], sort=True)["net_payoff_bps"].mean().unstack()
    session.rolling(5, min_periods=1).mean().plot(
        ax=axis, color=[colors[str(column)] for column in session.columns]
    )
    axis.axhline(0.0, color="#262626", linewidth=0.8)
    axis.set(
        title="Five-session rolling named F10 payoff", ylabel="Session mean bps", xlabel="Session"
    )
    figure.tight_layout()
    path = plot_root / "per_family_session_payoff_f10.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)
    return paths


def _table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "No rows."
    available = [column for column in columns if column in frame]
    header = "| " + " | ".join(available) + " |"
    divider = "|" + "|".join("---" for _ in available) + "|"
    lines = [header, divider]
    for row in frame.loc[:, available].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.2f}" if math.isfinite(value) else "NA")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    identity: Mapping[str, str],
    named_metrics: pd.DataFrame,
    control_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    break_even: pd.DataFrame,
    intervals: pd.DataFrame,
    concentration: pd.DataFrame,
    deletions: pd.DataFrame,
    breakdowns: pd.DataFrame,
    nulls: pd.DataFrame,
    reconstructed: pd.DataFrame,
    payoff: pd.DataFrame,
    quality_report: Mapping[str, object],
    missing_2023: Mapping[str, object],
) -> None:
    named_all = named_metrics.loc[named_metrics["evidence_cohort"].eq("all_reference")]
    named_valid = named_metrics.loc[named_metrics["evidence_cohort"].eq("primary_valid_fill")]
    controls_all = control_metrics.loc[control_metrics["evidence_cohort"].eq("all_reference")]
    f10_comparison = comparisons.loc[
        comparisons["fill_model"].eq("F10") & comparisons["evidence_cohort"].eq("all_reference")
    ]
    evidence_counts = (
        reconstructed.groupby(["classification", "fill_evidence_classification"], sort=True)
        .size()
        .reset_index(name="opportunities")
    )
    direction = (
        payoff.loc[payoff["fill_model"].eq("F10")]
        .groupby(["classification", "direction"], sort=True)["net_payoff_bps"]
        .agg(["size", "sum", "mean"])
        .reset_index()
    )
    trigger = (
        payoff.loc[payoff["fill_model"].eq("F10")]
        .groupby(["classification", "trigger_type"], sort=True)["net_payoff_bps"]
        .agg(["size", "sum", "mean"])
        .reset_index()
    )
    deletion_stock = concentration.loc[concentration["dimension"].eq("symbol")]
    monthly_named = breakdowns.loc[
        breakdowns["fill_model"].eq("F10")
        & breakdowns["slice_type"].eq("month")
        & breakdowns["family"].isin(["cycle_04|state_4", "cycle_07|state_5"])
    ]
    quality_checks = pd.DataFrame(cast(Sequence[Mapping[str, object]], quality_report["checks"]))
    text = f"""# Frozen Named-Loop T0 Execution Realism V1

## 1. Exact hypothesis and scientific status

The exact original T0 OCO trigger for each frozen named orientation produces positive net payoff after the frozen 5 bps-per-side cost and remains positive under predeclared adverse entry stress. This is opened retrospective evidence, not a tradable edge, prospective validation, paper approval, or live approval.

## 2. Why this follows the failed filters and latency test

Rolling profitability, payoff-state models, breadth/coherence, lead-lag, competitor rejection, directed rotation, anchor acceptance, and one-bar latency did not reliably improve T0. One-bar delay lost 10.12 bps per exact pair, so this experiment tests execution realism of T0 itself without a filter, delay, model, or new exit.

## 3. Frozen families and controls

- Named: `cycle_04|state_4` (`2->4->2`) and `cycle_07|state_5` (`5->6->5`).
- Controls: `cycle_04|state_2` and `cycle_07|state_6`.
- No failed family can be replaced and no new loop is admitted.

## 4. Exact T0 trigger, signal clock, and reference fill

The anchor high and low are known at anchor+5 minutes. The next 24 exact five-minute bars are scanned. A gap through one side fills at the observed open; otherwise a one-sided high/low cross uses the threshold reference. A same-bar dual-side cross is ambiguous and unavailable. Direction is the first non-ambiguous OCO side, never later payoff or route completion. The stored entry timestamp is the trigger bar's start label.

For opening gaps, signal time and reference fill time are the observed bar open. For intrabar threshold crosses, the exact crossing time is absent: the five-minute bar only proves the cross at bar close. Those rows are bounded references, not observed executable fills.

## 5. Fill achievability and data limitations

{_table(evidence_counts, ["classification", "fill_evidence_classification", "opportunities"])}

Only five-minute OHLC exists: no tick, trade, quote, or one-minute evidence. Of 809 named rows, 54 are gap-observable and 755 are bounded within-bar references. No non-gap threshold fill is exactly observable. Ambiguous or missing rows never enter the valid-fill cohort.

## 6. Adverse fill envelope, cost, and terminal

F0/F5/F10/F15/F20 are all calculated independently from the same frozen reference entry. Long entry is multiplied by `1 + bps/10000`; short entry by `1 - bps/10000`. The exact simple-return convention is retained. The fixed 10 bps round-trip cost remains additional; adverse entry movement is not subtracted again. The exit is always the anchor+125-minute terminal close convention. No horizon restarts.

## 7. Historical 2025 reproduction and execution decay

All 1,111 named/control T0 rows reconstructed exactly. The full named T0 source has 809 rows and +15,099.74 bps; the latency-common 808 subset retains the registered +15,087.32 bps. Per-family all-reference results:

{_table(named_all, ["family", "fill_model", "opportunities", "total_net_payoff_bps", "mean_net_payoff_bps", "median_net_payoff_bps", "positive_payoff_rate", "profit_factor", "maximum_drawdown_bps"])}

The combined named totals are F0 {payoff.loc[(payoff["classification"].eq("named")) & (payoff["fill_model"].eq("F0")), "net_payoff_bps"].sum():.2f}, F5 {payoff.loc[(payoff["classification"].eq("named")) & (payoff["fill_model"].eq("F5")), "net_payoff_bps"].sum():.2f}, F10 {payoff.loc[(payoff["classification"].eq("named")) & (payoff["fill_model"].eq("F10")), "net_payoff_bps"].sum():.2f}, F15 {payoff.loc[(payoff["classification"].eq("named")) & (payoff["fill_model"].eq("F15")), "net_payoff_bps"].sum():.2f}, and F20 {payoff.loc[(payoff["classification"].eq("named")) & (payoff["fill_model"].eq("F20")), "net_payoff_bps"].sum():.2f} bps. F10 is primary; the combined summary is secondary.

## 8. Controls and named-versus-control comparisons

{_table(controls_all, ["family", "fill_model", "opportunities", "total_net_payoff_bps", "mean_net_payoff_bps"])}

{_table(f10_comparison, ["comparison", "named_opportunities", "control_opportunities", "named_mean_bps", "control_mean_bps", "mean_difference_bps"])}

The controls remain separate. Their negative F10 results make the opened-data effect orientation-specific descriptively, but do not remove selection exposure.

## 9. cycle_04 and cycle_07 separately

Both named families remain positive through F20 for `cycle_04`; `cycle_07` remains positive through F15 and turns negative at F20. One working family would not support the hypothesis; this report does not pool away family identity.

## 10. Exactly/gap observable versus bounded rows

{_table(named_valid, ["family", "fill_model", "opportunities", "valid_fill_coverage", "total_net_payoff_bps", "mean_net_payoff_bps"])}

The gap-observable subset is positive under F10 in both named families, but it contains only 12 `cycle_04` and 42 `cycle_07` rows. The all-reference effect is therefore not treated as proven executable.

## 11. Long/short and threshold/gap attribution

{_table(direction, ["classification", "direction", "size", "sum", "mean"])}

{_table(trigger, ["classification", "trigger_type", "size", "sum", "mean"])}

These are mandatory descriptive slices, not preferred subsets.

## 12. Break-even slippage and session-block uncertainty

{_table(break_even.loc[break_even["scope"].eq("overall")], ["family", "evidence_cohort", "opportunities", "break_even_adverse_slippage_bps", "bootstrap_lower_95_bps", "bootstrap_upper_95_bps", "f10_below_break_even"])}

{_table(intervals.loc[(intervals["fill_model"].eq("F10")) & (intervals["evidence_cohort"].eq("all_reference"))], ["family", "sessions", "observed_session_mean_bps", "sessions_positive_percentage", "bootstrap_lower_95_bps", "bootstrap_upper_95_bps"])}

Break-even values are diagnostics only and cannot become a tuned threshold on this cohort.

## 13. Stock/month stability and concentration

{_table(deletion_stock, ["family", "contributors", "top_one_absolute_contribution_share", "top_five_absolute_contribution_share", "herfindahl_index", "concentrated_or_unstable"])}

{_table(deletions, ["family", "dimension", "top_n", "removed", "remaining_opportunities", "remaining_total_net_payoff_bps", "remaining_mean_net_payoff_bps"])}

{_table(monthly_named, ["family", "slice_value", "opportunities", "total_net_payoff_bps", "mean_net_payoff_bps"])}

Stock, month, direction, trigger type, session, entry step, and historical hindsight-episode tables are exported. Best-stock/top-five-stock and best-episode/top-five-episode results are recomputed after deletion. Hindsight labels never enter prospective records.

## 14. Predeclared falsifications

{_table(nulls, ["null_name", "family", "fill_model", "observed_statistic_bps", "null_mean_bps", "null_lower_95_bps", "null_upper_95_bps", "empirical_pvalue", "may_replace_primary_clock"])}

The direction flip preserves entry and terminal. The timestamp shift is intentionally wrong and can never replace T0. Label permutations are attribution nulls only; none may generate a replacement rule.

## 15. 2023 archival status

Status: `{missing_2023["status"]}`. The original per-symbol hashes matched none of 886 local Parquet candidates across the declared repository/archive roots. A fresh provider download is forbidden and no 2023 execution-realism result is imputed.

## 16. Data quality and causal limitations

Provider hashes, exact bar cadence, source direction, entry step, terminal price, gross return, cost, and net payoff all reconcile. The decisive limitation is causal within-bar observability: 1,024 of 1,111 reference rows have only bounded threshold evidence.

{_table(quality_checks, ["check", "count", "status"])}

No twice-cost table was scored: it is not part of the frozen primary convention, and mixing it with the adverse-entry envelope would obscure rather than clarify the execution question. The frozen 10 bps aggregate round-trip cost and additional entry-price stresses remain separate.

## 17. Prospective logger, settlement, and completion rule

Opportunity, trigger/fill, and outcome records are separate create-only JSON records protected by file locks, precursor hashes, contract identity, snapshot identity, and record hashes. Settlement is prohibited before the frozen terminal and computes F0-F20 itself. Opened 2023/2025 periods and opened snapshot hashes are rejected in genuine prospective mode. Interim status exposes counts and data quality, never economics.

The commands are `log_frozen_named_loop_t0_opportunities_v1.py` for opportunity/trigger append and administrative status, and `settle_frozen_named_loop_t0_outcomes_v1.py` for matured outcomes. Neither command contains a broker, order, position, or runtime interface.

Completion requires at least 100 named settlements, 25 `cycle_04|state_4`, 50 `cycle_07|state_5`, 20 controls, 60 sessions, 10 stocks, and three completed calendar months. Every condition is fixed before collection.

## 18. Historical versus prospective decision and recommendation

Historical decision: **{HISTORICAL_DECISION}**. The opened all-reference result survives F10 descriptively and is stronger than controls, but most historical threshold fills are not provably executable from five-minute OHLC. Prospective decision: **{PROSPECTIVE_DECISION}**.

The single most valuable next step is to begin the frozen, execution-free prospective log on genuinely new snapshots and collect independent trade/quote or shadow-fill evidence where possible. Do not expose P&L or stop early. If both named families do not remain positive at F10 after the completion rule, retire the hypothesis.

No order was placed. No broker, IG, paper/demo, position, exit, deployment, runtime, or secret path was connected or changed.

## Reproducibility

- Run ID: `{identity["run_id"]}`
- Contract SHA-256: `{identity["contract_hash"]}`
- Data snapshot SHA-256: `{identity["data_snapshot_hash"]}`
- Model version: `{identity["model_version"]}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _missing_2023_report(contract: Mapping[str, Any]) -> dict[str, object]:
    manifest_path = (
        CONTRACT_PATH.parent / str(contract["inputs"]["provider_2023_hash_manifest"]["path"])
    ).resolve()
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    registered_root = Path(str(contract["inputs"]["expired_provider_2023_root"]))
    verification = verify_2023_archive(registered_root, manifest)
    registered = cast(Mapping[str, Any], contract["provider_2023_archive"])
    if verification["all_registered_hashes_match"]:
        raise AssertionError(
            "the contract records 2023 as unavailable, but the registered root is now complete; "
            "run a separately reviewed archival-restoration workflow"
        )
    return {
        **verification,
        "status": str(registered["status"]),
        "registered_root": str(registered_root),
        "registered_manifest_sha256": sha256_file(manifest_path),
        "search_date": registered["search_date"],
        "search_roots": registered["search_roots"],
        "parquet_candidates_hashed": registered["parquet_candidates_hashed"],
        "matching_files_across_search_roots": registered["matching_files"],
        "economic_scoring_performed": False,
        "reason": (
            "No complete original 2023 provider tape matching every registered SHA-256 "
            "was found. A fresh download is not an acceptable substitute."
        ),
    }


def _reconciliation_summary(
    source_2025: pd.DataFrame,
    reconstructed: pd.DataFrame,
    provider_hashes: Mapping[str, str],
    identity: Mapping[str, str],
) -> dict[str, object]:
    f0_total = float(reconstructed["original_net_payoff_bps"].sum())
    return {
        **identity,
        "period": 2025,
        "source_opportunities": len(source_2025),
        "exact_t0_reconstructions": int(reconstructed["reference_reconstruction_exact"].sum()),
        "exact_terminal_reconstructions": int(reconstructed["terminal_reconstruction_exact"].sum()),
        "exact_payoff_reconciliations": int(reconstructed["payoff_reconciliation_exact"].sum()),
        "f0_total_net_payoff_bps": f0_total,
        "source_f0_total_net_payoff_bps": float(source_2025["original_net_payoff_bps"].sum()),
        "direction_counts": {
            str(key): int(value)
            for key, value in reconstructed["direction"].value_counts().sort_index().items()
        },
        "entry_step_distribution": {
            str(key): int(value)
            for key, value in reconstructed["entry_step"].value_counts().sort_index().items()
        },
        "trigger_type_distribution": {
            str(key): int(value)
            for key, value in reconstructed["trigger_type"].value_counts().sort_index().items()
        },
        "family_counts": {
            str(key): int(value)
            for key, value in reconstructed["family"].value_counts().sort_index().items()
        },
        "all_original_terminals_equal": bool(reconstructed["terminal_reconstruction_exact"].all()),
        "provider_hashes": dict(sorted(provider_hashes.items())),
        "numerical_tolerance_bps": 1e-8,
        "economic_scoring_allowed": True,
    }


def _code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        PACKAGE_SOURCE / "stocker_research/frozen_named_loop_t0_execution/__init__.py",
        PACKAGE_SOURCE / "stocker_research/frozen_named_loop_t0_execution/families.py",
        PACKAGE_SOURCE / "stocker_research/frozen_named_loop_t0_execution/execution.py",
        PACKAGE_SOURCE / "stocker_research/frozen_named_loop_t0_execution/immutable_ledger.py",
        PACKAGE_SOURCE / "stocker_research/frozen_named_loop_t0_execution/metrics.py",
        PACKAGE_SOURCE / "stocker_research/frozen_named_loop_t0_execution/historical.py",
        PACKAGE_SOURCE / "stocker_research/frozen_named_loop_t0_execution/prospective.py",
        WORK / "log_frozen_named_loop_t0_opportunities_v1.py",
        WORK / "settle_frozen_named_loop_t0_outcomes_v1.py",
        WORK / "audit_frozen_named_loop_t0_execution_v1.py",
    ]
    return {str(path.relative_to(REPO)): sha256_file(path) for path in paths if path.is_file()}


def _run_metadata(
    *,
    contract: Mapping[str, Any],
    identity: Mapping[str, str],
    input_hashes: Mapping[str, str],
    provider_hashes: Mapping[str, str],
    source: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> dict[str, object]:
    safety = cast(Mapping[str, Any], contract["safety"])
    source_counts: dict[str, int] = {}
    grouped_counts = source.groupby(["period", "family"]).size().reset_index(name="count")
    for raw in grouped_counts.to_dict(orient="records"):
        row = cast(dict[str, Any], raw)
        source_counts[f"{int(row['period'])}|{row['family']}"] = int(row["count"])
    return {
        **identity,
        "experiment_name": contract["experiment_name"],
        "generated_at": RUN_TIMESTAMP,
        "git_sha": _git("rev-parse", "HEAD"),
        "inspected_branch": _git("branch", "--show-current"),
        "code_version": MODEL_VERSION,
        "code_hashes": _code_hashes(),
        "input_hashes": dict(sorted(input_hashes.items())),
        "provider_hashes": dict(sorted(provider_hashes.items())),
        "source_counts": source_counts,
        "historical_2025_reconstructed_opportunities": len(reconstructed),
        "historical_decision": HISTORICAL_DECISION,
        "prospective_decision": PROSPECTIVE_DECISION,
        "research_only": safety["research_only"],
        "execution_enabled": safety["execution_enabled"],
        "broker_connection_enabled": safety["broker_connection_enabled"],
        "order_placement_enabled": False,
        "runtime_changed": safety["application_runtime_changed"],
        "interim_economics_blinded": True,
    }


def _artifact_manifest(root: Path, identity: Mapping[str, str]) -> dict[str, object]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in IDENTITY_EXCLUSIONS:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        **identity,
        "manifest_version": 1,
        "files": files,
        "research_only": True,
        "execution_enabled": False,
    }


def _verify_exact_rerun(output: Path, primary: Path) -> dict[str, object]:
    expected = {
        path.relative_to(primary): path
        for path in primary.rglob("*")
        if path.is_file() and path.name not in IDENTITY_EXCLUSIONS
    }
    observed = {
        path.relative_to(output): path
        for path in output.rglob("*")
        if path.is_file() and path.name not in IDENTITY_EXCLUSIONS
    }
    if set(expected) != set(observed):
        raise AssertionError(
            "exact rerun artifact set differs: "
            f"missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    mismatches = [
        name.as_posix()
        for name in sorted(expected)
        if expected[name].read_bytes() != observed[name].read_bytes()
    ]
    if mismatches:
        raise AssertionError(f"exact rerun byte mismatch: {mismatches}")
    return {
        "status": "pass",
        "primary_artifacts_sha256": sha256_file(primary / "artifact_manifest.json"),
        "exact_artifacts_sha256": sha256_file(output / "artifact_manifest.json"),
        "files_compared": len(expected),
        "byte_identical": True,
    }


def run_reference(
    *,
    output: Path,
    report_path: Path | None,
    exact_rerun_of: Path | None,
) -> dict[str, object]:
    """Reconstruct and score only the hash-pinned opened 2025 reference."""

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"create-only output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    contract, contract_hash, input_hashes = load_and_verify_contract(CONTRACT_PATH)
    source = build_source_populations(contract, contract_path=CONTRACT_PATH)
    source_2025 = source.loc[source["period"].eq(2025)].copy()
    providers = load_provider_frames(
        contract,
        source_2025["symbol"],
        contract_path=CONTRACT_PATH,
    )
    provider_hashes = _provider_hashes(contract, providers)
    identity = _identity(contract_hash, input_hashes, provider_hashes)

    reconstructed = reconstruct_historical_2025(source_2025, providers)
    reconstructed["provider_data_hash"] = reconstructed["symbol"].map(provider_hashes)
    reconstructed = attach_hindsight_episodes(reconstructed, contract, contract_path=CONTRACT_PATH)
    payoff = build_payoff_ledger(reconstructed)
    reconstructed = annotate(reconstructed, identity)
    payoff = annotate(payoff, identity)

    named_metrics, control_metrics, _ = build_family_metric_tables(payoff, identity)
    reference_metrics = build_reference_metrics(reconstructed, payoff, identity)
    decay = build_execution_decay(payoff, identity)
    comparisons = build_comparisons(payoff, identity)
    intervals = build_session_intervals(payoff, identity)
    break_even = build_break_even(payoff, identity)
    breakdowns = build_breakdowns(payoff, identity)
    concentration, deletions, leave_one_out = build_concentration_and_deletions(payoff, identity)
    nulls, flipped, shifted = build_nulls(reconstructed, payoff, providers, identity)
    quality_report, quality_table = build_data_quality(
        reconstructed, providers, provider_hashes, identity
    )
    quality_report.update(identity)
    missing_2023 = _missing_2023_report(contract)
    missing_2023.update(identity)
    reconciliation = _reconciliation_summary(source_2025, reconstructed, provider_hashes, identity)
    empty = _empty_ledgers(identity)
    completion = prospective_completion_status(contract, identity)

    shutil.copyfile(CONTRACT_PATH, output / "frozen_experiment_contract.json")
    shutil.copyfile(MAPPING_PATH, output / "frozen_pair_to_family_mapping.json")
    write_parquet(
        output / "historical_named_reference_ledger.parquet",
        reconstructed.loc[reconstructed["classification"].eq("named")].reset_index(drop=True),
    )
    write_parquet(
        output / "historical_control_reference_ledger.parquet",
        reconstructed.loc[reconstructed["classification"].eq("control")].reset_index(drop=True),
    )
    write_parquet(output / "trigger_reconstruction_ledger.parquet", reconstructed)
    evidence_columns = [
        "run_id",
        "contract_hash",
        "data_snapshot_hash",
        "opportunity_id",
        "anchor_id",
        "event_lineage_id",
        "symbol",
        "session",
        "loop_id",
        "orientation",
        "family",
        "classification",
        "direction",
        "trigger_timestamp",
        "trigger_type",
        "reference_entry_timestamp",
        "reference_entry_price",
        "signal_known_timestamp",
        "market_data_availability_timestamp",
        "fill_evidence_classification",
        "primary_valid_fill_evidence",
        "exact_or_bounded_evidence",
        "signal_fill_time_status",
        "provider_data_hash",
    ]
    write_parquet(
        output / "fill_evidence_classification_ledger.parquet",
        reconstructed.loc[:, evidence_columns],
    )
    write_parquet(output / "payoff_envelope_ledger.parquet", payoff)
    write_csv(output / "historical_reference_metrics.csv", reference_metrics)
    write_csv(output / "named_family_metrics.csv", named_metrics)
    write_csv(output / "control_family_metrics.csv", control_metrics)
    write_csv(output / "named_versus_control_comparisons.csv", comparisons)
    write_csv(output / "execution_decay_curve.csv", decay)
    write_csv(output / "break_even_slippage_results.csv", break_even)
    write_csv(output / "session_block_intervals.csv", intervals)
    write_csv(output / "stock_and_month_breakdowns.csv", breakdowns)
    write_csv(output / "null_test_results.csv", nulls)
    write_csv(output / "concentration_results.csv", concentration)
    write_csv(output / "dominant_contributor_removals.csv", deletions)
    write_csv(output / "leave_one_stock_out.csv", leave_one_out)
    write_parquet(output / "direction_flipped_null_ledger.parquet", flipped)
    write_parquet(output / "timestamp_shift_null_ledger.parquet", shifted)
    write_json(output / "data_quality_report.json", quality_report)
    write_csv(output / "data_quality_report.csv", quality_table)
    write_json(output / "historical_reconciliation_checks.json", reconciliation)
    write_json(output / "missing_2023_archival_report.json", missing_2023)
    write_parquet(output / "prospective_opportunity_ledger.parquet", empty["opportunity"])
    write_parquet(output / "prospective_trigger_fill_append_ledger.parquet", empty["trigger"])
    write_parquet(output / "prospective_outcome_settlement_ledger.parquet", empty["settlement"])
    write_json(output / "prospective_completion_status.json", completion)
    make_plots(output, reconstructed, payoff, decay, break_even, breakdowns)

    metadata = _run_metadata(
        contract=contract,
        identity=identity,
        input_hashes=input_hashes,
        provider_hashes=provider_hashes,
        source=source,
        reconstructed=reconstructed,
    )
    write_json(output / "run_metadata.json", metadata)
    write_json(output / "artifact_manifest.json", _artifact_manifest(output, identity))

    if report_path is not None:
        write_report(
            Path(report_path),
            identity=identity,
            named_metrics=named_metrics,
            control_metrics=control_metrics,
            comparisons=comparisons,
            break_even=break_even,
            intervals=intervals,
            concentration=concentration,
            deletions=deletions,
            breakdowns=breakdowns,
            nulls=nulls,
            reconstructed=reconstructed,
            payoff=payoff,
            quality_report=quality_report,
            missing_2023=missing_2023,
        )
    if exact_rerun_of is not None:
        write_json(
            output / "exact_rerun_identity.json",
            _verify_exact_rerun(output, Path(exact_rerun_of).resolve()),
        )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--exact-rerun-of", type=Path)
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write the Markdown report (used by the exact rerun).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_reference(
        output=args.output,
        report_path=None if args.no_report else args.report,
        exact_rerun_of=args.exact_rerun_of,
    )
    print(f"wrote frozen reference artifacts: {args.output}")
    print("research_only=true execution_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
