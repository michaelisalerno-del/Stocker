#!/usr/bin/env python3
# ruff: noqa: E501
"""Run the frozen, research-only Clean Anchor Price Acceptance V1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stocker_research.clean_anchor_price_acceptance import (
    ProspectiveAcceptanceLedger,
    build_variant_decisions,
    calculate_price_acceptance,
    calculate_remaining_payoff,
    select_first_post_anchor_bar,
)
from stocker_research.clean_anchor_price_acceptance.metrics import (
    acceptance_diagnostics,
    four_cell_interaction,
    paired_difference_rows,
    paired_variant_comparison,
    session_block_bootstrap,
    veto_accounting,
)

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
CONTRACT_PATH = WORK / "contracts/20260716-clean-anchor-price-acceptance-v1.json"
DEFAULT_OUTPUT = WORK / "artifacts/20260716-clean-anchor-price-acceptance-v1/primary"
DEFAULT_REPORT = WORK / "reports/20260716-clean-anchor-price-acceptance-v1.md"
MODEL_VERSION = "clean_anchor_price_acceptance_v1.0.0"
RUN_TIMESTAMP = "2026-07-16T00:00:00+00:00"
MACHINE_SUFFIXES = {".parquet", ".csv", ".json"}
IDENTITY_EXCLUSIONS = {
    "artifact_manifest.json",
    "exact_rerun_identity.json",
    "independent_audit.json",
}
VARIANT_A = "A_same_clock_base"
VARIANT_B = "B_anchor_veto_only"
VARIANT_C = "C_price_acceptance_only"
VARIANT_D = "D_anchor_veto_plus_price_acceptance"
VARIANT_E = "E_anchor_veto_plus_price_acceptance_plus_range"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object, *, length: int | None = None) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    result = hashlib.sha256(payload).hexdigest()
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


def _resolved(value: str) -> Path:
    return (CONTRACT_PATH.parent / value).resolve()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def load_and_verify_contract() -> tuple[dict[str, Any], str, dict[str, str]]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_hash = sha256(CONTRACT_PATH)
    if not contract["registered_before_scoring"]:
        raise AssertionError("experiment contract was not frozen before scoring")
    paths = {
        "v2_trade_decisions": _resolved(contract["inputs"]["v2_trade_decisions"]["path"]),
        "static_anchor_policy_ledger": _resolved(
            contract["inputs"]["static_anchor_policy_ledger"]["path"]
        ),
        "anchor_mass_ledger": _resolved(contract["inputs"]["anchor_mass_ledger"]["path"]),
        "episode_states": _resolved(contract["inputs"]["episode_states"]["path"]),
        "episode_diagnostics": _resolved(contract["inputs"]["episode_diagnostics"]["path"]),
        "provider_hash_manifest": _resolved(contract["inputs"]["provider_hash_manifest"]["path"]),
        "range_prediction_ledger": _resolved(contract["inputs"]["range_prediction_ledger"]["path"]),
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    expected = {
        "v2_trade_decisions": contract["inputs"]["v2_trade_decisions"]["sha256"],
        "static_anchor_policy_ledger": contract["inputs"]["static_anchor_policy_ledger"]["sha256"],
        "anchor_mass_ledger": contract["inputs"]["anchor_mass_ledger"]["sha256"],
        "episode_states": contract["inputs"]["episode_states"]["sha256"],
        "episode_diagnostics": contract["inputs"]["episode_diagnostics"]["sha256"],
        "provider_hash_manifest": contract["inputs"]["provider_hash_manifest"]["sha256"],
        "range_prediction_ledger": contract["inputs"]["range_prediction_ledger"]["sha256"],
    }
    drift = {
        name: (expected[name], hashes[name]) for name in expected if hashes[name] != expected[name]
    }
    if drift:
        raise AssertionError(f"frozen input drift: {drift}")
    return contract, contract_hash, hashes


def verify_provider_hashes(contract: Mapping[str, Any]) -> dict[str, str]:
    manifest = json.loads(
        _resolved(contract["inputs"]["provider_hash_manifest"]["path"]).read_text(encoding="utf-8")
    )["sha256"]
    root = Path(contract["inputs"]["provider_2025_root"])
    provider_hashes: dict[str, str] = {}
    for key, expected in sorted(manifest.items()):
        if not key.startswith("provider_2025_"):
            continue
        symbol = key.removeprefix("provider_2025_")
        path = root / f"symbol={symbol}" / "timeframe=5m/data.parquet"
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"hash-pinned provider drift for {symbol}")
        provider_hashes[symbol] = actual
    if len(provider_hashes) != 20:
        raise AssertionError("expected the frozen twenty-symbol 2025 provider universe")
    return provider_hashes


def _population_policy_rows(policy: pd.DataFrame, *, track: str) -> pd.DataFrame:
    selected = policy.loc[
        policy["track"].eq(track) & policy["policy"].eq("static_anchor_good_to_bad_odds_veto")
    ].copy()
    if selected["opportunity_id"].duplicated().any():
        raise AssertionError(f"duplicate frozen policy identities for {track}")
    return selected


def build_source_population(
    contract: Mapping[str, Any],
    *,
    track: str,
) -> pd.DataFrame:
    """Reconstruct exact source identities without touching target payoff."""

    policy = pd.read_parquet(_resolved(contract["inputs"]["static_anchor_policy_ledger"]["path"]))
    masses = pd.read_parquet(_resolved(contract["inputs"]["anchor_mass_ledger"]["path"]))
    trades = pd.read_parquet(_resolved(contract["inputs"]["v2_trade_decisions"]["path"]))
    selected = _population_policy_rows(policy, track=track)
    no_filter = trades.loc[trades["model_name"].eq("no_payoff_state_filter")].copy()
    if no_filter["opportunity_id"].duplicated().any():
        raise AssertionError("V2 no-filter opportunity identities are not unique")
    mass_columns = [
        "checkpoint_id",
        "track",
        "anchor_good_mass",
        "anchor_bad_mass",
        "anchor_unknown_mass",
        "good_to_bad_odds",
        "anchor_probability",
    ]
    mass = masses.loc[:, mass_columns]
    if mass.duplicated(["checkpoint_id", "track"]).any():
        raise AssertionError("anchor-mass checkpoint identity is ambiguous within track")
    selected = selected.merge(
        mass,
        on=["checkpoint_id", "track"],
        how="left",
        validate="one_to_one",
    )
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
        "entry_timestamp",
        "exit_timestamp",
        "entry_price",
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
        "score_session",
        "month",
        "quarter",
        "sector",
        "run_id",
        "configuration_hash",
    ]
    source = selected.merge(
        no_filter.loc[:, trade_columns],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
        suffixes=("_policy", "_v2"),
    )
    if source["anchor_id"].isna().any():
        raise AssertionError("frozen policy row lacks exact V2 source opportunity")
    if not source["stock"].astype(str).eq(source["symbol_norm"].astype(str)).all():
        raise AssertionError("stock identity mismatch between frozen experiments")
    if not source["target_loop"].astype(str).eq(source["loop_id"].astype(str)).all():
        raise AssertionError("loop identity mismatch between frozen experiments")
    if not source["orientation_policy"].astype(str).eq(source["orientation_v2"].astype(str)).all():
        raise AssertionError("orientation mismatch between frozen experiments")
    odds = np.where(
        pd.to_numeric(source["anchor_bad_mass"], errors="coerce").eq(0.0),
        np.inf,
        pd.to_numeric(source["anchor_good_mass"], errors="coerce")
        / pd.to_numeric(source["anchor_bad_mass"], errors="coerce"),
    )
    reconstructed_pass = pd.Series(odds, index=source.index).gt(1.0)
    frozen_pass = source["decision_status"].eq("retained")
    if not reconstructed_pass.eq(frozen_pass).all():
        raise AssertionError("static anchor veto does not reproduce frozen decision")
    source = source.rename(
        columns={
            "symbol_norm": "symbol",
            "target_loop": "loop_id_frozen",
            "orientation_policy": "orientation",
            "start_timestamp": "anchor_timestamp",
            "entry_timestamp": "original_entry_timestamp",
            "exit_timestamp": "original_terminal_timestamp",
            "primary_net_payoff_bps": "original_net_payoff_bps",
            "gross_payoff_bps": "original_gross_payoff_bps",
            "primary_total_cost_bps": "original_total_cost_bps",
            "run_id": "v2_run_id",
            "experiment_run_id": "sequential_veto_run_id",
        }
    )
    source["loop_id"] = source.pop("loop_id_frozen")
    source["population_track"] = track
    source["static_anchor_veto_pass"] = frozen_pass.to_numpy(bool)
    source["static_anchor_veto_score"] = odds
    source["static_anchor_veto_threshold"] = 1.0
    source["static_anchor_veto_reason_codes"] = np.where(
        source["static_anchor_veto_pass"],
        np.where(
            pd.to_numeric(source["anchor_bad_mass"], errors="coerce").eq(0.0),
            "static_anchor_no_bad_mass",
            "static_anchor_good_to_bad_odds_above_1",
        ),
        "static_anchor_good_to_bad_odds_at_or_below_1",
    )
    source["source_artifact_hash"] = stable_hash(
        {
            "v2": contract["inputs"]["v2_trade_decisions"]["sha256"],
            "veto": contract["inputs"]["static_anchor_policy_ledger"]["sha256"],
        }
    )
    source["anchor_timestamp"] = pd.to_datetime(source["anchor_timestamp"], utc=True)
    source["original_entry_timestamp"] = pd.to_datetime(
        source["original_entry_timestamp"], utc=True
    )
    source["original_terminal_timestamp"] = pd.to_datetime(
        source["original_terminal_timestamp"], utc=True
    )
    source["session_date"] = source["session_date_policy"].astype(str)
    source["period"] = source["period_policy"].astype(int)
    source = source.drop(
        columns=[
            "session_date_policy",
            "session_date_v2",
            "period_policy",
            "period_v2",
            "orientation_v2",
        ]
    )
    return source.sort_values(
        ["period", "session_date", "symbol", "anchor_timestamp", "opportunity_id"],
        kind="stable",
    ).reset_index(drop=True)


def load_provider_frames(
    contract: Mapping[str, Any], symbols: Iterable[str]
) -> dict[str, pd.DataFrame]:
    root = Path(contract["inputs"]["provider_2025_root"])
    result: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(symbols)):
        path = root / f"symbol={symbol}" / "timeframe=5m/data.parquet"
        frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        result[symbol] = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return result


def score_population(
    source: pd.DataFrame,
    providers: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Freeze one checkpoint, one acceptance rule, and exact-clock outcomes."""

    feature_records: list[dict[str, object]] = []
    outcome_records: list[dict[str, object]] = []
    delay_records: list[dict[str, object]] = []
    for row in source.itertuples(index=False):
        base = {
            "opportunity_id": str(row.opportunity_id),
            "period": int(row.period),
            "symbol": str(row.symbol),
            "loop_id": str(row.loop_id),
            "orientation": str(row.orientation),
            "event_lineage_id": str(row.event_lineage_id),
        }
        if int(row.period) != 2025:
            feature_records.append(
                {
                    **base,
                    "checkpoint_status": "provider_2023_hash_pinned_tape_unavailable",
                    "price_acceptance_status": "provider_2023_hash_pinned_tape_unavailable",
                    "price_acceptance_pass": False,
                }
            )
            outcome_records.append(
                {**base, "outcome_status": "provider_2023_hash_pinned_tape_unavailable"}
            )
            delay_records.append(
                {**base, "outcome_status": "provider_2023_hash_pinned_tape_unavailable"}
            )
            continue
        anchor = pd.Timestamp(row.anchor_timestamp)
        terminal = pd.Timestamp(row.original_terminal_timestamp)
        frame = providers[str(row.symbol)]
        bars = frame.loc[
            frame["timestamp"].between(anchor, anchor + pd.Timedelta(minutes=250), inclusive="both")
        ].copy()
        anchor_rows = bars.loc[bars["timestamp"].eq(anchor)]
        terminal_valid = terminal == anchor + pd.Timedelta(minutes=125)
        if len(anchor_rows) != 1:
            status = "missing_or_ambiguous_anchor_bar"
        elif not math.isclose(
            float(anchor_rows.iloc[0]["close"]),
            float(row.anchor_close),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            status = "anchor_reference_price_mismatch"
        elif not terminal_valid:
            status = "original_terminal_clock_mismatch"
        else:
            status = "available"
        if status == "available":
            checkpoint = select_first_post_anchor_bar(bars, anchor_timestamp=anchor)
            acceptance = calculate_price_acceptance(
                checkpoint,
                anchor_reference_price=float(row.anchor_close),
                direction=int(row.direction),
            )
        else:
            checkpoint = select_first_post_anchor_bar(pd.DataFrame(), anchor_timestamp=anchor)
            acceptance = calculate_price_acceptance(
                checkpoint,
                anchor_reference_price=float(row.anchor_close),
                direction=int(row.direction),
            )
        checkpoint_status = checkpoint.status if status == "available" else status
        acceptance_status = acceptance.status if status == "available" else status
        flipped = calculate_price_acceptance(
            checkpoint,
            anchor_reference_price=float(row.anchor_close),
            direction=-int(row.direction),
        )
        feature_records.append(
            {
                **base,
                "checkpoint_status": checkpoint_status,
                "checkpoint_bar_start_timestamp": checkpoint.bar_start_timestamp,
                "checkpoint_freeze_timestamp": checkpoint.freeze_timestamp,
                "checkpoint_open": checkpoint.open,
                "checkpoint_high": checkpoint.high,
                "checkpoint_low": checkpoint.low,
                "checkpoint_close": checkpoint.close,
                "price_acceptance_status": acceptance_status,
                "signed_close_return_bps": acceptance.signed_close_return_bps,
                "favourable_excursion_bps": acceptance.favourable_excursion_bps,
                "adverse_excursion_bps": acceptance.adverse_excursion_bps,
                "acceptance_balance_bps": acceptance.acceptance_balance_bps,
                "price_acceptance_pass": acceptance.price_acceptance_pass,
                "close_sign_only_pass": (
                    acceptance.signed_close_return_bps is not None
                    and acceptance.signed_close_return_bps > 0.0
                ),
                "direction_flipped_acceptance_pass": flipped.price_acceptance_pass,
                "feature_max_availability_timestamp": checkpoint.freeze_timestamp,
                "forecast_freeze_timestamp": checkpoint.freeze_timestamp,
                "future_path_feature_count": 0,
            }
        )
        outcome = calculate_remaining_payoff(
            bars,
            anchor_timestamp=anchor,
            original_terminal_timestamp=terminal,
            direction=int(row.direction),
            cost_bps_per_side=5.0,
        )
        delayed = calculate_remaining_payoff(
            bars,
            anchor_timestamp=anchor,
            original_terminal_timestamp=terminal,
            direction=int(row.direction),
            cost_bps_per_side=5.0,
            additional_delay_bars=1,
        )
        outcome_records.append(
            {
                **base,
                "outcome_status": outcome.status,
                "entry_timestamp": outcome.entry_timestamp,
                "entry_price": outcome.entry_price,
                "exit_timestamp": outcome.exit_timestamp,
                "exit_price": outcome.exit_price,
                "gross_payoff_bps": outcome.gross_payoff_bps,
                "entry_cost_bps": outcome.entry_cost_bps,
                "exit_cost_bps": outcome.exit_cost_bps,
                "total_cost_bps": outcome.total_cost_bps,
                "net_payoff_bps": outcome.net_payoff_bps,
                "remaining_mfe_bps": outcome.remaining_mfe_bps,
                "remaining_mae_bps": outcome.remaining_mae_bps,
                "restarted_exit_timestamp": outcome.restarted_exit_timestamp,
                "restarted_gross_payoff_bps": outcome.restarted_gross_payoff_bps,
                "restarted_net_payoff_bps": outcome.restarted_net_payoff_bps,
            }
        )
        delay_records.append(
            {
                **base,
                "outcome_status": delayed.status,
                "entry_timestamp": delayed.entry_timestamp,
                "exit_timestamp": delayed.exit_timestamp,
                "gross_payoff_bps": delayed.gross_payoff_bps,
                "total_cost_bps": delayed.total_cost_bps,
                "net_payoff_bps": delayed.net_payoff_bps,
                "acceptance_recomputed": False,
            }
        )
    return (
        pd.DataFrame(feature_records),
        pd.DataFrame(outcome_records),
        pd.DataFrame(delay_records),
    )


def attach_episode_identity(
    source: pd.DataFrame,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    episodes = pd.read_parquet(_resolved(contract["inputs"]["episode_diagnostics"]["path"])).copy()
    episodes["onset"] = pd.to_datetime(episodes["hindsight_estimated_onset"]).dt.date
    episodes["end"] = pd.to_datetime(episodes["hindsight_estimated_end"]).dt.date
    result = source.copy()
    result["episode_id"] = "outside_hindsight_positive_episode"
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
            result.at[index, "episode_id"] = str(candidates.iloc[0]["episode_id"])
        elif len(candidates) > 1:
            result.at[index, "episode_id"] = "ambiguous_hindsight_episode"
    return result


def build_scored_source(
    source: pd.DataFrame,
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    delay: pd.DataFrame,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    join = ["opportunity_id", "period", "symbol", "loop_id", "orientation", "event_lineage_id"]
    frame = source.merge(features, on=join, how="left", validate="one_to_one")
    frame = frame.merge(outcomes, on=join, how="left", validate="one_to_one")
    frame = frame.merge(
        delay.loc[:, [*join, "outcome_status", "entry_timestamp", "net_payoff_bps"]].rename(
            columns={
                "outcome_status": "additional_delay_status",
                "entry_timestamp": "additional_delay_entry_timestamp",
                "net_payoff_bps": "additional_delay_net_payoff_bps",
            }
        ),
        on=join,
        how="left",
        validate="one_to_one",
    )
    frame["source_available"] = frame["outcome_status"].eq("available") & frame[
        "price_acceptance_status"
    ].eq("available")
    frame["availability_status"] = np.where(
        frame["source_available"], "available", frame["outcome_status"]
    )
    frame["range_permission_available"] = False
    frame["range_permission_pass"] = False
    frame["predicted_remaining_range_bps"] = np.nan
    frame["range_permission_reason_codes"] = "immutable_causal_range_prediction_unavailable"
    frame["original_payoff_before_delayed_entry_bps"] = frame["original_net_payoff_bps"]
    frame["fraction_original_payoff_remaining"] = np.where(
        pd.to_numeric(frame["original_net_payoff_bps"], errors="coerce").abs().gt(1e-12),
        pd.to_numeric(frame["net_payoff_bps"], errors="coerce")
        / pd.to_numeric(frame["original_net_payoff_bps"], errors="coerce"),
        np.nan,
    )
    frame["month"] = frame["session_date"].astype(str).str[:7]
    minute = pd.to_datetime(frame["anchor_timestamp"], utc=True).dt.tz_convert("America/New_York")
    frame["clock_phase"] = np.select(
        [minute.dt.hour.lt(11), minute.dt.hour.lt(14)],
        ["early", "middle"],
        default="late",
    )
    frame["anchor_regime"] = "state_" + frame["state"].astype(int).astype(str)
    frame["direction_label"] = np.where(frame["direction"].eq(1), "long", "short")
    frame = attach_episode_identity(frame, contract)
    return frame


def _max_drawdown(values: pd.Series) -> float:
    cumulative = pd.to_numeric(values, errors="coerce").fillna(0.0).cumsum()
    return float((cumulative - cumulative.cummax()).min()) if len(cumulative) else 0.0


def variant_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    slices: list[tuple[str, str, pd.Series]] = [
        ("all", "all", pd.Series(True, index=decisions.index)),
    ]
    for column in [
        "period",
        "loop_id",
        "orientation",
        "direction_label",
        "symbol",
        "month",
        "clock_phase",
        "anchor_regime",
    ]:
        for value in sorted(decisions[column].dropna().astype(str).unique()):
            slices.append((column, value, decisions[column].astype(str).eq(value)))
    for slice_type, slice_value, mask in slices:
        subset = decisions.loc[mask]
        for variant, group in subset.groupby("variant", sort=True):
            available = group.loc[group["decision"].ne("unavailable")]
            admitted = available.loc[available["admitted"]]
            net = pd.to_numeric(admitted["net_payoff_bps"], errors="coerce").dropna()
            gross = pd.to_numeric(admitted["gross_payoff_bps"], errors="coerce").dropna()
            costs = pd.to_numeric(admitted["total_cost_bps"], errors="coerce").dropna()
            losses = float(-net.loc[net.lt(0.0)].sum())
            records.append(
                {
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "variant": str(variant),
                    "eligible_source_opportunities": int(len(group)),
                    "available_delayed_entries": int(len(available)),
                    "admitted_opportunities": int(len(admitted)),
                    "rejected_opportunities": int(available["decision"].eq("rejected").sum()),
                    "coverage": float(len(admitted) / len(available)) if len(available) else np.nan,
                    "gross_payoff_bps": float(gross.sum()) if len(gross) else np.nan,
                    "net_payoff_bps": float(net.sum()) if len(net) else np.nan,
                    "net_per_admitted_opportunity_bps": float(net.mean()) if len(net) else np.nan,
                    "median_net_payoff_bps": float(net.median()) if len(net) else np.nan,
                    "positive_payoff_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
                    "profit_factor": float(net.loc[net.gt(0.0)].sum() / losses)
                    if losses > 0.0
                    else np.nan,
                    "total_cost_bps": float(costs.sum()) if len(costs) else np.nan,
                    "maximum_drawdown_bps": _max_drawdown(available["policy_net_payoff_bps"]),
                    "average_remaining_mfe_bps": float(
                        pd.to_numeric(admitted["remaining_mfe_bps"], errors="coerce").mean()
                    ),
                    "average_remaining_mae_bps": float(
                        pd.to_numeric(admitted["remaining_mae_bps"], errors="coerce").mean()
                    ),
                    "mean_fraction_original_payoff_remaining": float(
                        pd.to_numeric(
                            admitted["fraction_original_payoff_remaining"], errors="coerce"
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(records)


def build_paired_metrics(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = decisions.loc[decisions["decision"].ne("unavailable")].copy()
    comparisons = [
        (VARIANT_D, VARIANT_A, "D_vs_A_primary"),
        (VARIANT_D, VARIANT_B, "D_vs_B_price_increment_after_anchor"),
        (VARIANT_D, VARIANT_C, "D_vs_C_anchor_increment_after_price"),
        (VARIANT_B, VARIANT_A, "B_vs_A_anchor_veto"),
        (VARIANT_C, VARIANT_A, "C_vs_A_price_acceptance"),
    ]
    rows: list[dict[str, object]] = []
    differences: list[pd.DataFrame] = []
    slices: list[tuple[str, str, pd.Series]] = [
        ("all", "all", pd.Series(True, index=available.index))
    ]
    for column in ["period", "loop_id"]:
        for value in sorted(available[column].astype(str).unique()):
            slices.append((column, value, available[column].astype(str).eq(value)))
    for slice_type, slice_value, mask in slices:
        subset = available.loc[mask]
        for treatment, control, label in comparisons:
            result = paired_variant_comparison(subset, treatment=treatment, control=control)
            rows.append(
                {
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "comparison": label,
                    **result,
                }
            )
            if slice_type == "all" and label == "D_vs_A_primary":
                frame = paired_difference_rows(subset, treatment=treatment, control=control)
                frame["comparison"] = label
                differences.append(frame)
    primary = pd.concat(differences, ignore_index=True)
    bootstrap = session_block_bootstrap(primary, resamples=2000, block_length=5, seed=20260716)
    rows[0].update(bootstrap)
    return pd.DataFrame(rows), primary


def build_four_cell(source: pd.DataFrame) -> pd.DataFrame:
    available = source.loc[source["source_available"]].copy()
    result = four_cell_interaction(available, group_columns=["period", "loop_id"])
    enriched: list[dict[str, object]] = []
    for row in result.to_dict("records"):
        anchor_pass = str(row["interaction_cell"]).startswith("anchor_pass")
        acceptance_pass = str(row["interaction_cell"]).endswith("acceptance_pass")
        group = available.loc[
            available["period"].eq(int(row["period"]))
            & available["loop_id"].eq(str(row["loop_id"]))
            & available["static_anchor_veto_pass"].eq(anchor_pass)
            & available["price_acceptance_pass"].eq(acceptance_pass)
        ]
        stock_contribution = group.groupby("symbol")["net_payoff_bps"].sum()
        episode_contribution = group.groupby("episode_id")["net_payoff_bps"].sum()
        best_stock = stock_contribution.idxmax() if len(stock_contribution) else None
        best_episode = episode_contribution.idxmax() if len(episode_contribution) else None
        row.update(
            {
                "additional_bar_delay_net_payoff_bps": float(
                    pd.to_numeric(group["additional_delay_net_payoff_bps"], errors="coerce").sum()
                ),
                "net_after_removing_best_stock_bps": float(
                    pd.to_numeric(
                        group.loc[group["symbol"].ne(best_stock), "net_payoff_bps"], errors="coerce"
                    ).sum()
                ),
                "net_after_removing_best_episode_bps": float(
                    pd.to_numeric(
                        group.loc[group["episode_id"].ne(best_episode), "net_payoff_bps"],
                        errors="coerce",
                    ).sum()
                ),
                "availability_status": "available",
            }
        )
        enriched.append(row)
    cells = [
        "anchor_fail|acceptance_fail",
        "anchor_fail|acceptance_pass",
        "anchor_pass|acceptance_fail",
        "anchor_pass|acceptance_pass",
    ]
    for loop_id in ["cycle_04", "cycle_07"]:
        source_count = int(
            source.loc[source["period"].eq(2023) & source["loop_id"].eq(loop_id)].shape[0]
        )
        for cell in cells:
            enriched.append(
                {
                    "period": 2023,
                    "loop_id": loop_id,
                    "interaction_cell": cell,
                    "source_opportunities_before_checkpoint": source_count,
                    "opportunities": 0,
                    "independent_stocks": 0,
                    "availability_status": "provider_2023_hash_pinned_tape_unavailable",
                }
            )
    return pd.DataFrame(enriched).sort_values(
        ["period", "loop_id", "interaction_cell"], kind="stable"
    )


def build_continuous_diagnostics(source: pd.DataFrame) -> pd.DataFrame:
    available = acceptance_diagnostics(
        source.loc[source["source_available"]], round_trip_cost_bps=10.0
    )
    rows: list[dict[str, object]] = []
    slices = [("all", "all", pd.Series(True, index=available.index))]
    for column in ["period", "loop_id"]:
        for value in sorted(available[column].astype(str).unique()):
            slices.append((column, value, available[column].astype(str).eq(value)))
    for slice_type, slice_value, mask in slices:
        group = available.loc[mask]
        correlation = spearmanr(group["acceptance_balance_bps"], group["net_payoff_bps"])
        rows.append(
            {
                "diagnostic": "spearman_continuous",
                "slice_type": slice_type,
                "slice_value": slice_value,
                "opportunities": int(len(group)),
                "spearman_rho": float(correlation.statistic),
                "spearman_pvalue": float(correlation.pvalue),
            }
        )
        for bin_name, bin_group in group.groupby("acceptance_bin", sort=False):
            rows.append(
                {
                    "diagnostic": "fixed_cost_bin",
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "acceptance_bin": str(bin_name),
                    "opportunities": int(len(bin_group)),
                    "mean_net_payoff_bps": float(bin_group["net_payoff_bps"].mean()),
                    "median_net_payoff_bps": float(bin_group["net_payoff_bps"].median()),
                    "positive_payoff_rate": float(bin_group["net_payoff_bps"].gt(0.0).mean()),
                    "net_payoff_bps": float(bin_group["net_payoff_bps"].sum()),
                }
            )
    return pd.DataFrame(rows)


def build_veto_tables(source: pd.DataFrame) -> pd.DataFrame:
    available = source.loc[source["source_available"]].copy()
    rules = {
        "B_vs_A_anchor_veto": available["static_anchor_veto_pass"],
        "C_vs_A_price_acceptance": available["price_acceptance_pass"],
        "D_vs_A_primary": available["static_anchor_veto_pass"] & available["price_acceptance_pass"],
    }
    return pd.DataFrame(
        [
            {"comparison": label, **veto_accounting(available, admitted=mask)}
            for label, mask in rules.items()
        ]
    )


def _policy_net(source: pd.DataFrame, admitted: pd.Series, *, payoff_column: str) -> float:
    values = pd.to_numeric(source[payoff_column], errors="coerce")
    return float(values.loc[admitted.fillna(False)].sum())


def build_nulls(source: pd.DataFrame) -> pd.DataFrame:
    available = (
        source.loc[source["source_available"]]
        .sort_values(["period", "symbol", "anchor_timestamp", "opportunity_id"], kind="stable")
        .copy()
    )
    d_pass = available["static_anchor_veto_pass"] & available["price_acceptance_pass"]
    actual = _policy_net(available, d_pass, payoff_column="net_payoff_bps")
    rng = np.random.default_rng(20260716)
    count = int(d_pass.sum())
    random_results = np.array(
        [
            float(
                available.iloc[rng.choice(len(available), size=count, replace=False)][
                    "net_payoff_bps"
                ].sum()
            )
            for _ in range(500)
        ]
    )
    available["prior_acceptance"] = available.groupby(["period", "symbol"], observed=True)[
        "price_acceptance_pass"
    ].shift(1)
    rules = {
        "registered_D": d_pass,
        "time_shifted_prior_opportunity_acceptance": available["static_anchor_veto_pass"]
        & available["prior_acceptance"].fillna(False),
        "direction_flipped_acceptance": available["static_anchor_veto_pass"]
        & available["direction_flipped_acceptance_pass"],
        "close_sign_only": available["static_anchor_veto_pass"] & available["close_sign_only_pass"],
    }
    rows = [
        {
            "null_test": name,
            "admitted_opportunities": int(mask.sum()),
            "coverage": float(mask.mean()),
            "net_payoff_bps": _policy_net(available, mask, payoff_column="net_payoff_bps"),
        }
        for name, mask in rules.items()
    ]
    rows.append(
        {
            "null_test": "random_coverage_matched_500_seeds",
            "admitted_opportunities": count,
            "coverage": float(d_pass.mean()),
            "net_payoff_bps": float(random_results.mean()),
            "null_lower_95_bps": float(np.quantile(random_results, 0.025)),
            "null_upper_95_bps": float(np.quantile(random_results, 0.975)),
            "actual_D_percentile_within_null": float(np.mean(random_results <= actual)),
        }
    )
    return pd.DataFrame(rows)


def build_stresses(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = source.loc[source["source_available"]].copy()
    d_pass = available["static_anchor_veto_pass"] & available["price_acceptance_pass"]
    admitted = available.loc[d_pass].copy()
    rows: list[dict[str, object]] = [
        {
            "stress_test": "twice_costs",
            "admitted_opportunities": int(len(admitted)),
            "net_payoff_bps": float((admitted["gross_payoff_bps"] - 20.0).sum()),
            "status": "entry_and_exit_costs_doubled_only",
        },
        {
            "stress_test": "one_additional_bar_execution_delay",
            "admitted_opportunities": int(
                admitted["additional_delay_net_payoff_bps"].notna().sum()
            ),
            "net_payoff_bps": float(admitted["additional_delay_net_payoff_bps"].sum()),
            "status": "first_bar_acceptance_frozen_original_terminal_retained",
        },
    ]
    contribution = admitted.groupby("symbol")["net_payoff_bps"].sum().sort_values(ascending=False)
    best_stock = str(contribution.index[0]) if len(contribution) else None
    top_five = set(contribution.head(5).index.astype(str))
    episode = admitted.groupby("episode_id")["net_payoff_bps"].sum().sort_values(ascending=False)
    best_episode = str(episode.index[0]) if len(episode) else None
    top_five_episodes = set(episode.head(5).index.astype(str))
    deletions = {
        "remove_best_stock": admitted["symbol"].ne(best_stock),
        "remove_top_five_stocks": ~admitted["symbol"].isin(top_five),
        "remove_best_episode": admitted["episode_id"].ne(best_episode),
        "remove_top_five_episodes": ~admitted["episode_id"].isin(top_five_episodes),
    }
    for name, mask in deletions.items():
        rows.append(
            {
                "stress_test": name,
                "admitted_opportunities": int(mask.sum()),
                "net_payoff_bps": float(admitted.loc[mask, "net_payoff_bps"].sum()),
                "status": "immutable_population_deletion_no_replacement",
            }
        )
    threshold = available.groupby("period")["dollar_volume_proxy"].transform("median")
    liquid = available["dollar_volume_proxy"].ge(threshold)
    rows.append(
        {
            "stress_test": "minimum_liquidity_within_period_median",
            "admitted_opportunities": int((d_pass & liquid).sum()),
            "net_payoff_bps": _policy_net(
                available, d_pass & liquid, payoff_column="net_payoff_bps"
            ),
            "status": "frozen_causal_dollar_volume_proxy",
        }
    )
    rows.extend(
        [
            {
                "stress_test": "median_session_aggregation",
                "status": "blocked_missing_hash_pinned_classification_rebuild_inputs",
                "result_imputed": False,
            },
            {
                "stress_test": "fully_rebuilt_leave_one_stock_out",
                "status": "blocked_missing_hash_pinned_V1_V2_rebuild_inputs",
                "result_imputed": False,
            },
            {
                "stress_test": "2023_provider_scoring",
                "status": "blocked_provider_2023_hash_pinned_tape_unavailable",
                "result_imputed": False,
            },
            {
                "stress_test": "range_permission_variant_E",
                "status": "blocked_immutable_causal_range_prediction_unavailable",
                "result_imputed": False,
            },
        ]
    )
    twice = admitted.loc[:, ["opportunity_id", "period", "symbol", "loop_id"]].copy()
    twice["gross_payoff_bps"] = admitted["gross_payoff_bps"].to_numpy(float)
    twice["stressed_total_cost_bps"] = 20.0
    twice["stressed_net_payoff_bps"] = twice["gross_payoff_bps"] - 20.0
    delay = admitted.loc[
        :,
        [
            "opportunity_id",
            "period",
            "symbol",
            "loop_id",
            "checkpoint_freeze_timestamp",
            "additional_delay_entry_timestamp",
            "original_terminal_timestamp",
            "additional_delay_status",
            "additional_delay_net_payoff_bps",
        ],
    ].copy()
    delay["price_acceptance_decision_recomputed"] = False
    return pd.DataFrame(rows), twice, delay


def build_concentration(source: pd.DataFrame) -> pd.DataFrame:
    available = source.loc[source["source_available"]].copy()
    d_pass = available["static_anchor_veto_pass"] & available["price_acceptance_pass"]
    available["D_policy_contribution_bps"] = np.where(d_pass, available["net_payoff_bps"], 0.0)
    available["D_vs_A_increment_bps"] = (
        available["D_policy_contribution_bps"] - available["net_payoff_bps"]
    )
    rows: list[dict[str, object]] = []
    dimensions = [
        "symbol",
        "loop_id",
        "orientation",
        "direction_label",
        "period",
        "month",
        "anchor_regime",
        "clock_phase",
        "episode_id",
    ]
    for measure in ["D_policy_contribution_bps", "D_vs_A_increment_bps"]:
        for dimension in dimensions:
            contribution = available.groupby(dimension, dropna=False)[measure].sum()
            absolute = contribution.abs()
            denominator = float(absolute.sum())
            shares = absolute / denominator if denominator > 0.0 else absolute * np.nan
            hhi = float(np.square(shares).sum()) if denominator > 0.0 else np.nan
            ordered = shares.sort_values(ascending=False)
            for key, value in contribution.items():
                rows.append(
                    {
                        "measure": measure,
                        "dimension": dimension,
                        "contributor": str(key),
                        "contribution_bps": float(value),
                        "absolute_contribution_share": float(shares.loc[key])
                        if denominator > 0.0
                        else np.nan,
                        "top_one_absolute_share": float(ordered.iloc[0])
                        if len(ordered)
                        else np.nan,
                        "top_five_absolute_share": float(ordered.head(5).sum())
                        if len(ordered)
                        else np.nan,
                        "herfindahl": hhi,
                    }
                )
    return pd.DataFrame(rows)


def prospective_schema() -> dict[str, object]:
    return {
        "schema_version": "clean_anchor_price_acceptance_prospective_v1",
        "immutable_forecast": True,
        "outcomes_create_only": True,
        "research_only": True,
        "execution_enabled": False,
        "opened_periods_forbidden_in_holdout": [2023, 2025],
        "required_fields": [
            "run_id",
            "git_sha",
            "contract_hash",
            "data_snapshot_hash",
            "opportunity_id",
            "event_lineage_id",
            "symbol",
            "session",
            "loop_id",
            "orientation",
            "frozen_direction",
            "anchor_timestamp",
            "anchor_reference_price",
            "static_anchor_veto_score",
            "static_anchor_veto_pass",
            "checkpoint_timestamp",
            "checkpoint_open",
            "checkpoint_high",
            "checkpoint_low",
            "checkpoint_close",
            "signed_close_return_bps",
            "favourable_excursion_bps",
            "adverse_excursion_bps",
            "acceptance_balance_bps",
            "price_acceptance_pass",
            "next_entry_timestamp",
            "original_terminal_timestamp",
            "variant_decisions",
            "feature_availability_timestamps",
            "training_cutoff",
            "forecast_freeze_timestamp",
        ],
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
        path.name: sha256(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }


def verify_exact_rerun(output: Path, primary: Path) -> dict[str, object]:
    expected = {
        path.name: path
        for path in primary.iterdir()
        if path.is_file()
        and path.suffix in MACHINE_SUFFIXES
        and path.name not in IDENTITY_EXCLUSIONS
    }
    actual = {
        path.name: path
        for path in output.iterdir()
        if path.is_file()
        and path.suffix in MACHINE_SUFFIXES
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
        "compared_machine_files": len(expected),
        "missing_files": missing,
        "extra_files": extra,
        "hash_mismatches": mismatches,
    }


def make_plots(
    output: Path,
    source: pd.DataFrame,
    decisions: pd.DataFrame,
    four_cell: pd.DataFrame,
    paired: pd.DataFrame,
) -> list[Path]:
    plot_root = output / "plots"
    plot_root.mkdir()
    paths: list[Path] = []

    available = source.loc[source["source_available"]]
    cell = (
        four_cell.loc[four_cell["period"].eq(2025)]
        .groupby("interaction_cell")["mean_net_payoff_bps"]
        .mean()
    )
    matrix = np.array(
        [
            [
                cell.get("anchor_fail|acceptance_fail", np.nan),
                cell.get("anchor_fail|acceptance_pass", np.nan),
            ],
            [
                cell.get("anchor_pass|acceptance_fail", np.nan),
                cell.get("anchor_pass|acceptance_pass", np.nan),
            ],
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xticks([0, 1], ["acceptance fail", "acceptance pass"])
    ax.set_yticks([0, 1], ["anchor fail", "anchor pass"])
    for y in range(2):
        for x in range(2):
            ax.text(x, y, f"{matrix[y, x]:.1f}", ha="center", va="center")
    ax.set_title("2025 mean remaining net bps")
    fig.colorbar(image, ax=ax)
    paths.append(plot_root / "anchor_acceptance_four_cell.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        available["acceptance_balance_bps"],
        available["net_payoff_bps"],
        s=9,
        alpha=0.35,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set(xlabel="acceptance balance (bps)", ylabel="remaining net payoff (bps)")
    ax.set_title("Acceptance balance versus constant-terminal payoff")
    paths.append(plot_root / "acceptance_balance_vs_remaining_payoff.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    all_metrics = variant_metrics(decisions)
    all_metrics = all_metrics.loc[
        all_metrics["slice_type"].eq("all")
        & all_metrics["variant"].isin([VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D])
    ]
    fig, left = plt.subplots(figsize=(8, 4))
    x = np.arange(len(all_metrics))
    left.bar(x, all_metrics["net_payoff_bps"], color="#3b82f6")
    left.set_xticks(x, [str(value).split("_")[0] for value in all_metrics["variant"]])
    left.set_ylabel("net payoff (bps)")
    right = left.twinx()
    right.plot(x, all_metrics["coverage"], color="#dc2626", marker="o")
    right.set_ylabel("coverage")
    left.set_title("Same-clock variants")
    paths.append(plot_root / "variant_payoff_and_coverage.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    session = paired.groupby("session_date", sort=True)["difference_bps"].sum().cumsum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(session.index), session.to_numpy(float))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set(ylabel="cumulative D minus A (bps)", xlabel="session")
    ax.set_title("Primary paired session increment")
    paths.append(plot_root / "variant_D_vs_A_paired_session.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    policy = decisions.loc[decisions["variant"].isin([VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D])]
    cumulative = (
        policy.pivot_table(
            index="anchor_timestamp",
            columns="variant",
            values="policy_net_payoff_bps",
            aggfunc="sum",
        )
        .sort_index()
        .fillna(0.0)
        .cumsum()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    for column in cumulative:
        ax.plot(cumulative.index, cumulative[column], label=column.split("_")[0])
    ax.legend(ncol=4)
    ax.set(ylabel="cumulative policy net bps", xlabel="anchor time")
    ax.set_title("Same-clock cumulative retained payoff")
    paths.append(plot_root / "variant_cumulative_payoff.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)

    d = decisions.loc[decisions["variant"].eq(VARIANT_D) & decisions["admitted"]]
    grouped = d.groupby(["period", "loop_id"])["net_payoff_bps"].sum().unstack(fill_value=0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped.plot.bar(ax=ax)
    ax.set(ylabel="Variant D net payoff (bps)", xlabel="period")
    ax.set_title("Variant D by named loop and period")
    paths.append(plot_root / "variant_D_by_loop_period.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120, metadata={"Software": MODEL_VERSION})
    plt.close(fig)
    return paths


def scientific_decision() -> str:
    """The missing frozen 2023 tape blocks the registered two-period decision."""

    return "experiment_blocked_by_missing_anchor_or_bar_identity"


def write_report(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    source: pd.DataFrame,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    four_cell: pd.DataFrame,
    veto: pd.DataFrame,
    continuous: pd.DataFrame,
    stress: pd.DataFrame,
    nulls: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    def metric(variant: str, field: str) -> float:
        row = metrics.loc[metrics["slice_type"].eq("all") & metrics["variant"].eq(variant)]
        return float(row.iloc[0][field])

    primary = paired.loc[
        paired["comparison"].eq("D_vs_A_primary") & paired["slice_type"].eq("all")
    ].iloc[0]
    price_increment = paired.loc[
        paired["comparison"].eq("D_vs_B_price_increment_after_anchor")
        & paired["slice_type"].eq("all")
    ].iloc[0]
    anchor_increment = paired.loc[
        paired["comparison"].eq("D_vs_C_anchor_increment_after_price")
        & paired["slice_type"].eq("all")
    ].iloc[0]
    primary_veto = veto.loc[veto["comparison"].eq("D_vs_A_primary")].iloc[0]
    rho = continuous.loc[
        continuous["diagnostic"].eq("spearman_continuous") & continuous["slice_type"].eq("all")
    ].iloc[0]
    twice = stress.loc[stress["stress_test"].eq("twice_costs")].iloc[0]
    delay = stress.loc[stress["stress_test"].eq("one_additional_bar_execution_delay")].iloc[0]
    source_counts = source.groupby(["period", "loop_id"]).size().to_dict()
    text = f"""# Clean Anchor Price Acceptance V1

## Scientific decision

**{metadata["scientific_decision"]}**

The exact 2025 attribution ran and is reported below, but the registered two-period hypothesis cannot be decided: the hash-pinned 2023 five-minute provider tape no longer exists. All 854 frozen 2023 named candidates remain explicit missing evidence, never zero and never reconstructed from a similar field. These opened retrospective surfaces cannot establish a tradable edge in any event.

## 1. Exact hypothesis and prior boundary

The registered primary comparison asks whether the exact frozen Sequential Loop Competitor Veto Track-A anchor decision plus one completed-bar directional acceptance rule improves **remaining** net payoff relative to the same named candidates at the same delayed entry and original terminal clock. This exact paired interaction had not been tested. Selective Payoff Equations V1 used a different 2024 OCO population, fitted many one-to-three-bar variables, and restarted horizons; Sequential Veto tested the anchor veto and evolving structural exclusion separately; payoff-state, lead-lag, and rotation work tested different state questions.

## 2. Frozen populations and inputs

- `cycle_04|state_4`: 2023={source_counts.get((2023, "cycle_04"), 0)}, 2025={source_counts.get((2025, "cycle_04"), 0)}.
- `cycle_07|state_5`: 2023={source_counts.get((2023, "cycle_07"), 0)}, 2025={source_counts.get((2025, "cycle_07"), 0)}.
- Controls remain separately labelled in machine-readable outputs.
- Source opportunity identity is the frozen V2 `opportunity_id`, joined one-to-one to the frozen Sequential Veto `event_lineage_id` and policy row.
- The anchor reference is V2 `anchor_close`, independently required to equal the hash-pinned provider close at `start_timestamp`.

## 3. Frozen anchor veto

The anchor score is `anchor_good_mass / anchor_bad_mass`, infinity when bad mass is zero. The exact frozen threshold is **1.0**: pass strictly above 1.0, reject at or below 1.0. Good/bad/unknown classifications, smoothing, support, and policy decisions come unchanged from Sequential Loop Competitor Veto V1. The veto is not updated after price is observed.

## 4. Checkpoint, price sign, and economic clock

The only checkpoint bar starts exactly five minutes after the anchor and freezes at its close ten minutes after anchor. Entry is the exact provider open at that freeze; a later row cannot substitute for a missing bar. For a long (short), returns and excursions are direction-adjusted exactly as registered. Admission requires signed close return > 0 and favourable excursion > adverse excursion. All A--D variants enter on the same clock and exit at the original `anchor + 125 minutes` terminal close, with 5 bps charged at entry and exit. Restarted h24 outcomes are separate diagnostics.

The retained causal range ledger is empty, so Variant E is unavailable. No replacement range model was fit.

## 5. 2025 variant results (constant terminal)

| Variant | admitted | coverage | net bps | mean/admitted bps |
|---|---:|---:|---:|---:|
| A same-clock base | {int(metric(VARIANT_A, "admitted_opportunities"))} | {metric(VARIANT_A, "coverage"):.1%} | {metric(VARIANT_A, "net_payoff_bps"):.2f} | {metric(VARIANT_A, "net_per_admitted_opportunity_bps"):.2f} |
| B anchor veto | {int(metric(VARIANT_B, "admitted_opportunities"))} | {metric(VARIANT_B, "coverage"):.1%} | {metric(VARIANT_B, "net_payoff_bps"):.2f} | {metric(VARIANT_B, "net_per_admitted_opportunity_bps"):.2f} |
| C price acceptance | {int(metric(VARIANT_C, "admitted_opportunities"))} | {metric(VARIANT_C, "coverage"):.1%} | {metric(VARIANT_C, "net_payoff_bps"):.2f} | {metric(VARIANT_C, "net_per_admitted_opportunity_bps"):.2f} |
| D anchor + acceptance | {int(metric(VARIANT_D, "admitted_opportunities"))} | {metric(VARIANT_D, "coverage"):.1%} | {metric(VARIANT_D, "net_payoff_bps"):.2f} | {metric(VARIANT_D, "net_per_admitted_opportunity_bps"):.2f} |
| E + range | 0 | unavailable | unavailable | unavailable |

## 6. Primary paired result and interaction

Variant D minus A is **{float(primary["paired_total_difference_bps"]):.2f} bps**, or {float(primary["paired_mean_difference_bps"]):.2f} bps per paired opportunity. The five-session-block 95% interval for the session-mean increment is [{float(primary["bootstrap_lower_95_bps"]):.2f}, {float(primary["bootstrap_upper_95_bps"]):.2f}] bps. Price acceptance after the anchor veto contributes {float(price_increment["paired_total_difference_bps"]):.2f} bps; the anchor veto after price acceptance contributes {float(anchor_increment["paired_total_difference_bps"]):.2f} bps.

The four-cell table is exported in `four_cell_interaction.csv`. The 2023 cells are explicitly unavailable because no causal checkpoint OHLC survives; they are not inferred.

## 7. Veto value, continuous relationship, and nulls

Variant D avoided {float(primary_veto["losses_avoided_bps"]):.2f} bps of losses while rejecting {float(primary_veto["profits_mistakenly_rejected_bps"]):.2f} bps of winners, for veto value {float(primary_veto["veto_value_bps"]):.2f} bps. The predeclared continuous acceptance diagnostic has Spearman rho {float(rho["spearman_rho"]):.3f} (p={float(rho["spearman_pvalue"]):.3g}). Fixed zero/cost bins, random coverage, prior-opportunity time shift, flipped direction, and close-sign-only controls are exported without selecting a replacement rule.

## 8. Stress, concentration, and failure cases

- Twice-cost Variant D: {float(twice["net_payoff_bps"]):.2f} bps.
- One additional execution bar, with the original acceptance frozen and terminal unchanged: {float(delay["net_payoff_bps"]):.2f} bps.
- Fully rebuilt leave-one-stock-out is blocked because the immutable 2023 V1/V2 rebuild inputs and provider tape are absent; no aggregate deletion is mislabeled as a rebuild.
- Stock, loop, direction, period, month, regime, clock, and hindsight-episode concentration are in `concentration_results.csv`. Hindsight episodes are outcome diagnostics only.
- A favourable first bar can still fail later; acceptance is a deterministic sign, not a route-completion prediction.

## 9. Interpretation and exact recommendation

The 2025 result can say only whether the loop supplied a candidate, the frozen anchor veto removed contamination, and one completed bar supplied an incremental sign on that opened period. It cannot satisfy the registered requirement for positive and stable 2023 and 2025 evidence because one period is unscorable. The exact next recommendation is to start the immutable prospective logger on a genuinely new hash-pinned five-minute snapshot with both named loops unchanged, then settle outcomes create-only after their original terminals; do not tune the rule on 2025.

## Reproducibility

- Run ID: `{metadata["run_id"]}`
- Git SHA at execution: `{metadata["git_sha"]}`
- Contract SHA-256: `{metadata["contract_hash"]}`
- Data snapshot SHA-256: `{metadata["data_snapshot_hash"]}`
- Exact command: `{metadata["command"]}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def run_historical(
    *,
    output: Path,
    report_path: Path,
    exact_rerun_of: Path | None,
) -> None:
    contract, contract_hash, input_hashes = load_and_verify_contract()
    provider_hashes = verify_provider_hashes(contract)
    data_snapshot_hash = stable_hash(
        {"contract": contract_hash, "inputs": input_hashes, "providers": provider_hashes}
    )
    run_id = "clean-anchor-" + stable_hash(
        {"contract": contract_hash, "data": data_snapshot_hash, "model": MODEL_VERSION},
        length=24,
    )
    named = build_source_population(contract, track="track_a_named_family")
    controls = build_source_population(contract, track="track_b_prior_only")
    expected_named = contract["populations"]["named_source_counts_before_price_scoring"]["all"]
    if len(named) != int(expected_named):
        raise AssertionError(f"named population drift: {len(named)} != {expected_named}")
    providers = load_provider_frames(contract, pd.concat([named["symbol"], controls["symbol"]]))
    named_features, named_outcomes, named_delay = score_population(named, providers)
    control_features, control_outcomes, control_delay = score_population(controls, providers)
    scored = build_scored_source(named, named_features, named_outcomes, named_delay, contract)
    scored_controls = build_scored_source(
        controls, control_features, control_outcomes, control_delay, contract
    )
    decisions = build_variant_decisions(scored)
    control_decisions = build_variant_decisions(scored_controls)
    metrics = variant_metrics(decisions)
    control_metrics = variant_metrics(control_decisions)
    paired, paired_rows = build_paired_metrics(decisions)
    four_cell = build_four_cell(scored)
    veto = build_veto_tables(scored)
    continuous = build_continuous_diagnostics(scored)
    nulls = build_nulls(scored)
    stress, twice_cost, additional_delay = build_stresses(scored)
    concentration = build_concentration(scored)
    range_ledger = scored.loc[
        :,
        [
            "opportunity_id",
            "period",
            "symbol",
            "loop_id",
            "orientation",
            "predicted_remaining_range_bps",
            "range_permission_available",
            "range_permission_pass",
            "range_permission_reason_codes",
        ],
    ]
    static_veto = scored.loc[
        :,
        [
            "opportunity_id",
            "event_lineage_id",
            "period",
            "symbol",
            "loop_id",
            "orientation",
            "checkpoint_id",
            "anchor_good_mass",
            "anchor_bad_mass",
            "anchor_unknown_mass",
            "static_anchor_veto_score",
            "static_anchor_veto_threshold",
            "static_anchor_veto_pass",
            "static_anchor_veto_reason_codes",
        ],
    ]
    restarted = scored.loc[
        :,
        [
            "opportunity_id",
            "period",
            "symbol",
            "loop_id",
            "orientation",
            "entry_timestamp",
            "restarted_exit_timestamp",
            "restarted_gross_payoff_bps",
            "restarted_net_payoff_bps",
        ],
    ]
    loo = pd.DataFrame(
        [
            {
                "status": "blocked_missing_hash_pinned_V1_V2_rebuild_inputs",
                "result_imputed": False,
                "blocker": "2023 provider tape and ephemeral V1/V2 rebuild surfaces are unavailable",
                "exact_command_after_restore": (
                    "PYTHONPATH=packages/stocker_research/src .venv/bin/python "
                    "research/slrno-v2/20260714-regime-loop-handoff/work/"
                    "run_clean_anchor_price_acceptance_v1.py"
                ),
            }
        ]
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
        "feature_schema_version": "clean_anchor_price_acceptance_v1",
        "fixed_horizon_bars": 24,
        "checkpoint_bars": 1,
        "cost_bps_per_side": 5.0,
        "random_seed": 20260716,
        "scored_periods": [2023, 2025],
        "generated_at": RUN_TIMESTAMP,
        "scientific_status": contract["scientific_status"],
        "scientific_decision": scientific_decision(),
        "range_permission_status": "unavailable_empty_immutable_prediction_ledger",
        "provider_2023_status": "unavailable_no_surviving_hash_pinned_five_minute_tape",
        "command": (
            "PYTHONPATH=packages/stocker_research/src .venv/bin/python "
            "research/slrno-v2/20260714-regime-loop-handoff/work/"
            "run_clean_anchor_price_acceptance_v1.py --output <OUTPUT> --report <REPORT>"
        ),
        "safety": contract["safety"],
    }
    output.mkdir(parents=True)
    shutil.copyfile(CONTRACT_PATH, output / "frozen_experiment_contract.json")
    detailed = {
        "named_source_opportunity_ledger.parquet": scored,
        "control_source_opportunity_ledger.parquet": scored_controls,
        "static_anchor_veto_ledger.parquet": static_veto,
        "first_bar_checkpoint_ledger.parquet": named_features.loc[
            :,
            [
                "opportunity_id",
                "period",
                "symbol",
                "loop_id",
                "orientation",
                "checkpoint_status",
                "checkpoint_bar_start_timestamp",
                "checkpoint_freeze_timestamp",
                "checkpoint_open",
                "checkpoint_high",
                "checkpoint_low",
                "checkpoint_close",
                "feature_max_availability_timestamp",
            ],
        ],
        "price_acceptance_feature_ledger.parquet": named_features,
        "range_permission_ledger.parquet": range_ledger,
        "variant_decision_ledger.parquet": decisions,
        "constant_terminal_remaining_payoff_ledger.parquet": named_outcomes,
        "restarted_horizon_diagnostic_ledger.parquet": restarted,
        "additional_bar_delay_ledger.parquet": additional_delay,
        "twice_cost_ledger.parquet": twice_cost,
        "primary_paired_opportunity_differences.parquet": paired_rows,
        "control_variant_decision_ledger.parquet": control_decisions,
    }
    summaries = {
        "four_cell_interaction.csv": four_cell,
        "variant_metrics.csv": metrics,
        "control_variant_metrics.csv": control_metrics,
        "paired_comparison_metrics.csv": paired,
        "veto_accounting.csv": veto,
        "continuous_acceptance_diagnostics.csv": continuous,
        "stress_test_results.csv": stress,
        "leave_one_stock_out_results.csv": loo,
        "concentration_results.csv": concentration,
        "null_test_results.csv": nulls,
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
    write_json(output / "prospective_immutable_forecast_ledger_schema.json", prospective_schema())
    plot_paths = make_plots(output, scored, decisions, four_cell, paired_rows)
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
        source=scored,
        metrics=metrics,
        paired=paired,
        four_cell=four_cell,
        veto=veto,
        continuous=continuous,
        stress=stress,
        nulls=nulls,
        concentration=concentration,
    )


def run_prospective(args: argparse.Namespace) -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if args.prospective_root is None or args.record_json is None:
        raise ValueError("prospective mode requires --prospective-root and --record-json")
    ledger = ProspectiveAcceptanceLedger(
        Path(args.prospective_root),
        opened_periods=set(contract["opened_data_status"]["opened_periods"]),
    )
    record = json.loads(Path(args.record_json).read_text(encoding="utf-8"))
    if args.mode == "prospective-forecast":
        path = ledger.append_forecast(record, holdout=bool(args.holdout))
    else:
        path = ledger.append_outcome(record)
    print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["historical", "prospective-forecast", "prospective-settle"],
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
