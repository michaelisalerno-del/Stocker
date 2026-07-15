# ruff: noqa: E402, E501
"""Research-only sequential competitive-loop exclusion experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stocker-mplconfig")

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[4]
_PACKAGE_SOURCE = _REPO_BOOTSTRAP / "packages/stocker_research/src"
if str(_PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_SOURCE))

from stocker_research.sequential_loop_competitor_veto import (
    CensusConfig,
    DecisionConfig,
    PayoffClassConfig,
    ProspectiveCompetitorLedger,
    RollingTrainingOnlyCensus,
    apply_irreversible_decisions,
    build_registered_checkpoints,
    classify_decision,
    classify_payoff_families,
    clock_bin,
    initial_posterior,
    paired_economic_contribution,
    paired_predictive_metrics,
    remaining_payoff,
    summarise_posterior,
    update_posterior,
)

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[3]
CONTRACT_PATH = WORK / "contracts/20260715-sequential-loop-competitor-veto-v1.json"
DEFAULT_OUTPUT = WORK / "artifacts/20260715-sequential-loop-competitor-veto-v1/primary"
DEFAULT_REPORT = WORK / "reports/20260715-sequential-loop-competitor-veto-v1.md"
V2_ROOT = WORK / "artifacts/20260714-dynamic-loop-edge-state-v2/primary"
MODEL_VERSION = "sequential_loop_competitor_veto_v1.0.0"
RUN_TIMESTAMP = "2026-07-15T00:00:00+00:00"
EVENT_CHECKPOINTS = {
    "first_completed_transition",
    "second_completed_transition",
    "exact_parent_completion",
    "first_route_diversion",
    "first_incompatible_transition",
}
SCORING_COLUMNS = [
    "anchor_id",
    "symbol_norm",
    "session_date",
    "start_timestamp",
    "cycle_index",
    "cycle_id",
    "cycle",
    "transition_length",
    "state",
    "current_state",
    "history_token",
    "loop_probability",
    "loop_occurs",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object, *, length: int = 24) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


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
    path.write_text(json.dumps(safe_json(value), indent=2, sort_keys=True) + "\n")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def input_paths(contract: Mapping[str, Any]) -> dict[str, Path]:
    paths = {
        "contract": CONTRACT_PATH,
        "scoring_2023": Path(contract["inputs"]["scoring_predictions"]["2023"]["path"]),
        "scoring_2025": Path(contract["inputs"]["scoring_predictions"]["2025"]["path"]),
        "accepted_signal_ledger": Path(contract["inputs"]["accepted_signal_ledger"]["path"]),
        "execution_anchors_2023": Path(contract["inputs"]["execution_anchors"]["2023"]["path"]),
        "execution_anchors_2025": Path(contract["inputs"]["execution_anchors"]["2025"]["path"]),
        "v1_source_hashes": Path(contract["inputs"]["v1_source_hashes"]["path"]),
        "v2_forecasts": V2_ROOT / "causal_edge_state_forecasts.parquet",
        "v2_trade_decisions": V2_ROOT / "trade_decisions.parquet",
        "v2_session_panel": V2_ROOT / "session_payoff_panel.parquet",
        "v2_episode_states": V2_ROOT / "hindsight_episode_states.parquet",
        "v2_episode_diagnostics": V2_ROOT / "hindsight_episode_diagnostics.parquet",
        "v2_rebuild_runner": WORK / contract["inputs"]["v2_rebuild_runner"]["path"],
        "v2_rebuild_config": WORK / contract["inputs"]["v2_rebuild_config"]["path"],
    }
    return paths


def verify_contract_and_inputs() -> tuple[dict[str, Any], dict[str, str], str]:
    contract = json.loads(CONTRACT_PATH.read_text())
    safety = contract["safety"]
    if not (
        safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["order_placement"] == "disabled"
        and safety["broker_connection_enabled"] is False
        and safety["deployment_enabled"] is False
        and safety["application_runtime_changed"] is False
    ):
        raise AssertionError("research-only safety contract drift")
    paths = input_paths(contract)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    actual = {name: sha256(path) for name, path in paths.items()}
    expected = {
        "scoring_2023": contract["inputs"]["scoring_predictions"]["2023"]["sha256"],
        "scoring_2025": contract["inputs"]["scoring_predictions"]["2025"]["sha256"],
        "accepted_signal_ledger": contract["inputs"]["accepted_signal_ledger"]["sha256"],
        "execution_anchors_2023": contract["inputs"]["execution_anchors"]["2023"]["sha256"],
        "execution_anchors_2025": contract["inputs"]["execution_anchors"]["2025"]["sha256"],
        "v1_source_hashes": contract["inputs"]["v1_source_hashes"]["sha256"],
        "v2_forecasts": contract["inputs"]["v2_artifacts"]["causal_edge_state_forecasts.parquet"],
        "v2_trade_decisions": contract["inputs"]["v2_artifacts"]["trade_decisions.parquet"],
        "v2_session_panel": contract["inputs"]["v2_artifacts"]["session_payoff_panel.parquet"],
        "v2_episode_states": contract["inputs"]["v2_artifacts"]["hindsight_episode_states.parquet"],
        "v2_episode_diagnostics": contract["inputs"]["v2_artifacts"][
            "hindsight_episode_diagnostics.parquet"
        ],
        "v2_rebuild_runner": contract["inputs"]["v2_rebuild_runner"]["sha256"],
        "v2_rebuild_config": contract["inputs"]["v2_rebuild_config"]["sha256"],
    }
    changed = sorted(name for name, digest in expected.items() if actual[name] != digest)
    if changed:
        raise AssertionError(f"frozen input hash mismatch: {changed}")
    source_hashes = json.loads(paths["v1_source_hashes"].read_text())["sha256"]
    provider_root = Path(contract["inputs"]["provider_2025_root"])
    for symbol in contract["population"].get("symbols", []):
        provider = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        if provider.exists() and sha256(provider) != source_hashes[f"provider_2025_{symbol}"]:
            raise AssertionError(f"2025 provider drift: {symbol}")
    data_snapshot = hashlib.sha256(
        json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return contract, actual, data_snapshot


def _bar_ordinal(timestamp: pd.Series) -> pd.Series:
    local = pd.to_datetime(timestamp, utc=True, errors="raise").dt.tz_convert("America/New_York")
    return ((local.dt.hour * 60 + local.dt.minute - 570) // 5).astype(int)


def load_structural_surfaces(
    paths: Mapping[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for period in (2023, 2025):
        frame = pd.read_parquet(paths[f"scoring_{period}"], columns=SCORING_COLUMNS)
        frame["period"] = period
        frame["start_timestamp"] = pd.to_datetime(frame["start_timestamp"], utc=True)
        frame["session_date"] = frame["session_date"].astype(str)
        frame["bar_ordinal"] = _bar_ordinal(frame["start_timestamp"])
        frame["orientation"] = "state_" + frame["state"].astype(int).astype(str)
        frames.append(frame)
    scoring = pd.concat(frames, ignore_index=True)
    if scoring.duplicated(["period", "anchor_id", "cycle_id"]).any():
        raise AssertionError("duplicate anchor/cycle structural row")
    catalogue = (
        scoring[["cycle_index", "cycle_id", "cycle", "transition_length"]]
        .drop_duplicates()
        .sort_values("cycle_index", kind="stable")
        .reset_index(drop=True)
    )
    if len(catalogue) != 20 or catalogue["cycle_index"].duplicated().any():
        raise AssertionError("frozen twenty-cycle catalogue drift")
    run_columns = [
        "period",
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "bar_ordinal",
        "state",
        "history_token",
    ]
    runs = scoring[run_columns].drop_duplicates(["period", "anchor_id"])
    if runs.duplicated(["period", "symbol_norm", "session_date", "bar_ordinal"]).any():
        raise AssertionError("ambiguous state-run boundary")
    runs = runs.sort_values(
        ["period", "symbol_norm", "session_date", "bar_ordinal"], kind="stable"
    ).reset_index(drop=True)
    grouped = runs.groupby(["period", "symbol_norm", "session_date"], sort=False)
    for number in (1, 2):
        runs[f"transition_{number}_state"] = grouped["state"].shift(-number)
        next_ordinal = grouped["bar_ordinal"].shift(-number)
        lag = next_ordinal - runs["bar_ordinal"]
        runs[f"transition_{number}_lag"] = lag.where(lag.le(24))
        runs[f"transition_{number}_state"] = runs[f"transition_{number}_state"].where(lag.le(24))
    evidence = runs[
        [
            "period",
            "anchor_id",
            "transition_1_state",
            "transition_1_lag",
            "transition_2_state",
            "transition_2_lag",
        ]
    ].rename(
        columns={
            "transition_1_state": "first_transition_state",
            "transition_1_lag": "first_transition_lag",
            "transition_2_state": "second_transition_state",
            "transition_2_lag": "second_transition_lag",
        }
    )
    examples = scoring.merge(
        evidence, on=["period", "anchor_id"], how="left", validate="many_to_one"
    )
    examples["current_state"] = examples["state"].astype(int)
    examples["loop_id"] = examples["cycle_id"].astype(str)
    return scoring, runs, examples


def load_opportunities(
    paths: Mapping[str, Path],
    scoring: pd.DataFrame,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    ledger = pd.read_parquet(paths["accepted_signal_ledger"])
    ledger = ledger.loc[
        ledger["period"].astype(str).isin(["2023", "2025"])
        & ledger["strategy"].eq("breakout_loop_scores_range_p75")
        & ledger["horizon"].eq(24)
    ].copy()
    ledger["period"] = ledger["period"].astype(int)
    ledger["session_date"] = ledger["session_date"].astype(str)
    ledger["start_timestamp"] = pd.to_datetime(ledger["start_timestamp"], utc=True)
    sessions = {
        period: sorted(runs.loc[runs["period"].eq(period), "session_date"].astype(str).unique())
        for period in (2023, 2025)
    }
    session_index = {
        (period, session): index
        for period, values in sessions.items()
        for index, session in enumerate(values)
    }
    ledger["session_index"] = [
        session_index[(int(period), str(session))]
        for period, session in ledger[["period", "session_date"]].itertuples(index=False, name=None)
    ]
    ledger = ledger.loc[ledger["session_index"].ge(60)].copy()
    top = scoring.sort_values(
        ["period", "anchor_id", "loop_probability", "cycle_index"],
        ascending=[True, True, False, True],
        kind="stable",
    ).drop_duplicates(["period", "anchor_id"])
    top = top[
        [
            "period",
            "anchor_id",
            "cycle_id",
            "cycle",
            "state",
            "history_token",
            "loop_probability",
            "bar_ordinal",
        ]
    ].rename(
        columns={
            "cycle_id": "top_loop",
            "cycle": "top_loop_cycle",
            "loop_probability": "top_loop_probability",
            "state": "anchor_state",
            "bar_ordinal": "structural_bar_ordinal",
        }
    )
    ledger = ledger.merge(top, on=["period", "anchor_id"], how="left", validate="one_to_one")
    if ledger["top_loop"].isna().any():
        raise AssertionError("source opportunity missing frozen structural anchor")
    ledger["orientation"] = "state_" + ledger["anchor_state"].astype(int).astype(str)
    ledger["source_bar_ordinal"] = ledger["bar_ordinal"].astype(int)
    ledger["bar_ordinal"] = ledger["structural_bar_ordinal"].astype(int)
    ledger["opportunity_id"] = (
        ledger["period"].astype(str) + "-" + ledger["anchor_id"].astype(str) + "-h24"
    )
    ledger["event_lineage_id"] = [
        "lineage-" + stable_hash([period, anchor, symbol, timestamp])
        for period, anchor, symbol, timestamp in ledger[
            ["period", "anchor_id", "symbol_norm", "start_timestamp"]
        ].itertuples(index=False, name=None)
    ]
    ledger["decision_timestamp"] = ledger["start_timestamp"] + pd.Timedelta(minutes=5)
    ledger["source_entry_timestamp"] = ledger["start_timestamp"] + pd.to_timedelta(
        5 * ledger["entry_step"].astype(int), unit="m"
    )
    ledger["terminal_timestamp"] = ledger["start_timestamp"] + pd.Timedelta(minutes=125)
    ledger["original_net_payoff_bps"] = np.where(
        ledger["status"].eq("filled"), ledger["gross_return_bps"].astype(float) - 10.0, np.nan
    )
    ledger["population_role"] = "general_source"
    target = (ledger["top_loop"].eq("cycle_04") & ledger["orientation"].eq("state_4")) | (
        ledger["top_loop"].eq("cycle_07") & ledger["orientation"].eq("state_5")
    )
    negative = ledger["top_loop"].eq("cycle_07") & ledger["orientation"].eq("state_6")
    neutral = ledger["top_loop"].eq("cycle_04") & ledger["orientation"].eq("state_2")
    ledger.loc[target, "population_role"] = "named_target"
    ledger.loc[negative, "population_role"] = "negative_control"
    ledger.loc[neutral, "population_role"] = "neutral_control"
    if ledger["opportunity_id"].duplicated().any():
        raise AssertionError("duplicate immutable source opportunity")
    return ledger.sort_values(
        ["period", "session_date", "symbol_norm", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)


def load_payoff_classes(paths: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasts = pd.read_parquet(paths["v2_forecasts"])
    classes = classify_payoff_families(forecasts, PayoffClassConfig())
    baseline = forecasts.loc[
        forecasts["model_name"].isin(["v1_60_session_selector", "payoff_only_change_point"])
    ].copy()
    return classes, baseline


def load_frozen_v2_rebuilder(paths: Mapping[str, Path]) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "frozen_dynamic_loop_edge_state_v2",
        paths["v2_rebuild_runner"],
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V2 rebuild runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_v2_rebuild_context(paths: Mapping[str, Path]) -> dict[str, Any]:
    v2 = load_frozen_v2_rebuilder(paths)
    config = v2.load_config()
    ledger, sessions, _, _ = v2.load_recovered_v1_analysis(config)
    calendars = v2.build_session_calendars(sessions)
    surface = v2.build_trade_surface(ledger, config)
    required_features = tuple(
        name
        for name in config["features"]["leading_feature_logit_weights"]
        if name != "out_of_distribution_score"
    )
    return {
        "v2": v2,
        "config": config,
        "configuration_hash": sha256(paths["v2_rebuild_config"]),
        "calendars": calendars,
        "surface": surface,
        "required_features": required_features,
    }


def rebuild_payoff_classes(
    context: Mapping[str, Any],
    *,
    excluded_stocks: set[str] | None = None,
    aggregation: str = "winsorised_mean",
    hazard: float | None = None,
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    v2 = context["v2"]
    config = context["config"]
    surface = context["surface"].copy()
    excluded = set(excluded_stocks or ())
    if excluded:
        surface = surface.loc[~surface["symbol_norm"].astype(str).isin(excluded)].copy()
        surface = v2.rebuild_surface_context_for_universe(
            surface,
            universe_size=int(surface["symbol_norm"].nunique()),
        )
    primary_panel, median_panel = v2.aggregate_payoff_panels(surface, config)
    payoff_panel = median_panel if aggregation == "median" else primary_panel
    cell_keys = {
        int(period): sorted(
            {
                (str(row.loop_id), str(row.orientation), int(row.horizon))
                for row in surface.loc[surface["period"].eq(int(period))].itertuples(index=False)
            }
        )
        for period in config["evaluation"]["periods"]
    }
    features = v2.build_feature_panel(
        surface,
        payoff_panel,
        context["calendars"],
        cell_keys,
        context["required_features"],
    )
    forecasts = v2.run_change_point_model(
        model_name="hierarchical_payoff_history_change_point",
        config=config,
        configuration_hash=context["configuration_hash"],
        run_id=run_id,
        calendars=context["calendars"],
        payoff_panel=payoff_panel,
        feature_panel=features,
        cell_keys_by_period=cell_keys,
        enable_hierarchy=True,
        include_leading_features=False,
        hazard=hazard,
    )
    return classify_payoff_families(forecasts, PayoffClassConfig()), {
        "excluded_stocks": sorted(excluded),
        "aggregation": aggregation,
        "hazard": hazard,
        "surface_rows": len(surface),
        "surface_stocks": int(surface["symbol_norm"].nunique()),
        "session_panel_rows": len(payoff_panel),
        "forecast_rows": len(forecasts),
        "all_stock_dependent_inputs_rebuilt": True,
    }


def load_execution_anchors(paths: Mapping[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "bar_ordinal",
        "next_open_24",
        "exit_close_24",
    ]
    for period in (2023, 2025):
        frame = pd.read_parquet(paths[f"execution_anchors_{period}"], columns=columns)
        frame["period"] = period
        frame["start_timestamp"] = pd.to_datetime(frame["start_timestamp"], utc=True)
        frame["session_date"] = frame["session_date"].astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_2025_bars(
    contract: Mapping[str, Any], symbols: Sequence[str]
) -> dict[tuple[str, str], pd.DataFrame]:
    root = Path(contract["inputs"]["provider_2025_root"])
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        path = root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        frame = pd.read_parquet(
            path,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        frame = frame.loc[local.dt.year.eq(2025) & minute.ge(570) & minute.lt(960)].copy()
        frame["session_date"] = (
            frame["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        )
        frame["symbol_norm"] = symbol
        frames.append(frame)
    tape = pd.concat(frames, ignore_index=True)
    return {
        (str(symbol), str(session)): group.sort_values("timestamp", kind="stable").reset_index(
            drop=True
        )
        for (symbol, session), group in tape.groupby(["symbol_norm", "session_date"], sort=False)
    }


def _class_maps(
    class_index: pd.DataFrame,
    period: int,
    session: str,
    candidates: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, float], dict[str, float], dict[str, bool]]:
    classes: dict[str, str] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    support: dict[str, bool] = {}
    for candidate in candidates.itertuples(index=False):
        key = (period, session, str(candidate.cycle_id), str(candidate.orientation))
        if key in class_index.index:
            row = class_index.loc[key]
            classes[str(candidate.cycle_id)] = str(row["payoff_class"])
            means[str(candidate.cycle_id)] = float(row["posterior_mean_net_bps"])
            stds[str(candidate.cycle_id)] = float(row["posterior_std_net_bps"])
            support[str(candidate.cycle_id)] = bool(row["payoff_class_support_pass"])
        else:
            classes[str(candidate.cycle_id)] = "unknown"
            means[str(candidate.cycle_id)] = 0.0
            stds[str(candidate.cycle_id)] = 120.0
            support[str(candidate.cycle_id)] = False
    return classes, means, stds, support


def _summary_payload(summary: Any) -> dict[str, object]:
    return {
        "all_good_loop_mass": summary.good_mass,
        "bad_loop_mass": summary.bad_mass,
        "unknown_loop_mass": summary.unknown_mass,
        "good_to_bad_odds": summary.good_to_bad_odds,
        "good_to_non_good_odds": summary.good_to_non_good_odds,
        "compatible_good_count": summary.compatible_good_count,
        "compatible_bad_count": summary.compatible_bad_count,
        "compatible_unknown_count": summary.compatible_unknown_count,
        "posterior_entropy": summary.posterior_entropy,
        "normalised_entropy": summary.normalised_entropy,
        "conservative_expected_remaining_net_bps": summary.conservative_remaining_net_bps,
        "posterior_expected_remaining_net_bps": summary.expected_remaining_net_bps,
        "posterior_expected_remaining_std_bps": summary.expected_remaining_std_bps,
        "p_positive_remaining": summary.p_positive_remaining,
        "target_compatible": summary.target_compatible,
    }


def _outcome_for_checkpoint(
    opportunity: Any,
    checkpoint: Any,
    execution_by_run: Mapping[tuple[int, int], Any],
    run_id_by_start: Mapping[tuple[int, str, str, pd.Timestamp], int],
    bars_2025: Mapping[tuple[str, str], pd.DataFrame],
    *,
    cost_bps_per_side: float = 5.0,
    execution_delay_bars: int = 0,
) -> dict[str, object]:
    base = {
        "outcome_status": "unfilled_no_outcome",
        "remaining_entry_timestamp": pd.NaT,
        "remaining_entry_price": math.nan,
        "constant_terminal_gross_bps": math.nan,
        "constant_terminal_net_bps": math.nan,
        "restarted_exit_timestamp": pd.NaT,
        "restarted_gross_bps": math.nan,
        "restarted_net_bps": math.nan,
        "remaining_mfe_bps": math.nan,
        "remaining_mae_bps": math.nan,
    }
    if str(opportunity.status) != "filled" or int(opportunity.direction) not in (-1, 1):
        return base
    checkpoint_time = pd.Timestamp(checkpoint.checkpoint_timestamp)
    terminal = pd.Timestamp(opportunity.terminal_timestamp)
    if checkpoint_time >= terminal:
        base["outcome_status"] = "too_late"
        return base
    if int(opportunity.period) == 2025:
        bars = bars_2025.get(
            (str(opportunity.symbol_norm), str(opportunity.session_date)), pd.DataFrame()
        )
        result = remaining_payoff(
            bars,
            direction=int(opportunity.direction),
            checkpoint_timestamp=checkpoint_time,
            terminal_timestamp=terminal,
            cost_bps_per_side=cost_bps_per_side,
            execution_delay_bars=execution_delay_bars,
        )
        return {
            "outcome_status": result.status,
            "remaining_entry_timestamp": result.entry_timestamp,
            "remaining_entry_price": result.entry_price,
            "constant_terminal_gross_bps": result.constant_terminal_gross_bps,
            "constant_terminal_net_bps": result.constant_terminal_net_bps,
            "restarted_exit_timestamp": result.restarted_exit_timestamp,
            "restarted_gross_bps": result.restarted_gross_bps,
            "restarted_net_bps": result.restarted_net_bps,
            "remaining_mfe_bps": result.remaining_mfe_bps,
            "remaining_mae_bps": result.remaining_mae_bps,
        }
    if str(checkpoint.checkpoint_type).startswith("fixed_bar_") or execution_delay_bars:
        base["outcome_status"] = "missing_source_data"
        return base
    start = checkpoint_time - pd.Timedelta(minutes=5)
    key = (
        int(opportunity.period),
        str(opportunity.symbol_norm),
        str(opportunity.session_date),
        start,
    )
    run_anchor = run_id_by_start.get(key)
    if run_anchor is None or (int(opportunity.period), int(run_anchor)) not in execution_by_run:
        base["outcome_status"] = "missing_source_data"
        return base
    source = execution_by_run[(int(opportunity.period), int(run_anchor))]
    entry = float(source.next_open_24)
    terminal_price = float(opportunity.exit_price)
    if not np.isfinite(entry) or not np.isfinite(terminal_price):
        base["outcome_status"] = "missing_source_data"
        return base
    gross = 10_000.0 * int(opportunity.direction) * (terminal_price / entry - 1.0)
    restarted_price = float(source.exit_close_24)
    restarted_gross = (
        10_000.0 * int(opportunity.direction) * (restarted_price / entry - 1.0)
        if np.isfinite(restarted_price)
        else math.nan
    )
    costs = 2.0 * cost_bps_per_side
    base.update(
        {
            "outcome_status": "available",
            "remaining_entry_timestamp": checkpoint_time,
            "remaining_entry_price": entry,
            "constant_terminal_gross_bps": gross,
            "constant_terminal_net_bps": gross - costs,
            "restarted_exit_timestamp": checkpoint_time + pd.Timedelta(minutes=120),
            "restarted_gross_bps": restarted_gross,
            "restarted_net_bps": restarted_gross - costs,
        }
    )
    return base


def build_checkpoint_ledgers(
    contract: Mapping[str, Any],
    opportunities: pd.DataFrame,
    scoring: pd.DataFrame,
    runs: pd.DataFrame,
    examples: pd.DataFrame,
    classes: pd.DataFrame,
    execution: pd.DataFrame,
    bars_2025: Mapping[tuple[str, str], pd.DataFrame],
    run_id: str,
    *,
    census_config: CensusConfig | None = None,
) -> dict[str, pd.DataFrame]:
    scoring_index = scoring.set_index(["period", "anchor_id"]).sort_index()
    candidate_cache: dict[tuple[int, int], pd.DataFrame] = {}

    def candidates_for(key: tuple[int, int]) -> pd.DataFrame:
        if key not in candidate_cache:
            selected = scoring_index.loc[key]
            if isinstance(selected, pd.Series):
                selected = selected.to_frame().T
            candidate_cache[key] = selected.reset_index(drop=True).sort_values(
                "cycle_index", kind="stable"
            )
        return candidate_cache[key]

    run_groups = {
        (int(period), str(symbol), str(session)): group.sort_values("bar_ordinal", kind="stable")
        for (period, symbol, session), group in runs.groupby(
            ["period", "symbol_norm", "session_date"], sort=False
        )
    }
    run_id_by_start = {
        (
            int(row.period),
            str(row.symbol_norm),
            str(row.session_date),
            pd.Timestamp(row.start_timestamp),
        ): int(row.anchor_id)
        for row in runs.itertuples(index=False)
    }
    execution_by_run = {
        (int(row.period), int(row.anchor_id)): row for row in execution.itertuples(index=False)
    }
    class_index = classes.set_index(["period", "score_session", "loop_id", "orientation"])
    cycles = {
        str(row.cycle_id): str(row.cycle)
        for row in scoring[["cycle_id", "cycle"]].drop_duplicates().itertuples(index=False)
    }
    checkpoint_rows: list[dict[str, object]] = []
    posterior_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    elimination_rows: list[dict[str, object]] = []
    anchor_set_rows: list[dict[str, object]] = []
    census_rows: list[dict[str, object]] = []
    primary_keys: set[tuple[str, str]] = set()
    decision_config = DecisionConfig()

    registered_census_config = census_config or CensusConfig()
    rolling_census = {
        period: RollingTrainingOnlyCensus(
            examples,
            period=period,
            config=registered_census_config,
        )
        for period in (2023, 2025)
    }
    for (period, session), opportunity_group in opportunities.groupby(
        ["period", "session_date"], sort=False
    ):
        census = rolling_census[int(period)]
        census.advance_before(str(session))
        for (loop, orientation, state, clock), (rows, successes) in census.clock_counts.items():
            census_rows.append(
                {
                    "period": int(period),
                    "score_session": str(session),
                    "training_latest_session": census.training_sessions[-1]
                    if census.training_sessions
                    else None,
                    "loop_id": loop,
                    "orientation": orientation,
                    "current_state": state,
                    "clock_phase": clock,
                    "training_rows": rows,
                    "training_occurrences": successes,
                    "smoothed_occurrence_probability": (successes + 1.0) / (rows + 2.0),
                }
            )
        for opportunity in opportunity_group.itertuples(index=False):
            candidate_key = (int(opportunity.period), int(opportunity.anchor_id))
            candidates = candidates_for(candidate_key)
            masses = {
                str(row.cycle_id): float(row.loop_probability)
                for row in candidates.itertuples(index=False)
            }
            initial = initial_posterior(masses)
            anchor_state = int(opportunity.anchor_state)
            orientation = str(opportunity.orientation)
            clock_lifts = {
                str(row.cycle_id): census.clock_lift(
                    str(row.cycle_id), orientation, anchor_state, int(opportunity.bar_ordinal)
                )
                for row in candidates.itertuples(index=False)
            }
            full_anchor = update_posterior(
                initial,
                cycles,
                anchor_state,
                (),
                evidence_likelihoods=clock_lifts,
            )
            general_classes, means, stds, support = _class_maps(
                class_index, int(period), str(session), candidates
            )
            for candidate in candidates.itertuples(index=False):
                anchor_set_rows.append(
                    {
                        "run_id": run_id,
                        "opportunity_id": str(opportunity.opportunity_id),
                        "event_lineage_id": str(opportunity.event_lineage_id),
                        "period": int(period),
                        "session_date": str(session),
                        "stock": str(opportunity.symbol_norm),
                        "anchor_id": int(opportunity.anchor_id),
                        "loop_id": str(candidate.cycle_id),
                        "orientation": orientation,
                        "cycle": str(candidate.cycle),
                        "initial_compatibility_score": float(candidate.loop_probability),
                        "initial_posterior_probability": full_anchor.known[str(candidate.cycle_id)],
                        "payoff_class": general_classes[str(candidate.cycle_id)],
                        "training_expected_net_bps": means[str(candidate.cycle_id)],
                        "training_payoff_std_bps": stds[str(candidate.cycle_id)],
                        "support_pass": support[str(candidate.cycle_id)],
                    }
                )
            state_group = run_groups[(int(period), str(opportunity.symbol_norm), str(session))]
            state_window = state_group.loc[
                state_group["bar_ordinal"].ge(int(opportunity.bar_ordinal))
                & state_group["bar_ordinal"].le(int(opportunity.bar_ordinal) + 24)
            ]
            checkpoints = build_registered_checkpoints(
                state_window,
                anchor_ordinal=int(opportunity.bar_ordinal),
                terminal_ordinal=int(opportunity.bar_ordinal) + 24,
                target_cycle=str(opportunity.top_loop_cycle),
                anchor_state=anchor_state,
            )
            seen_eliminations: set[str] = set()
            anchor_summaries: dict[str, Any] = {}
            opportunity_checkpoint_rows: list[dict[str, object]] = []
            for checkpoint in checkpoints.itertuples(index=False):
                observed = tuple(json.loads(str(checkpoint.observed_transitions_json)))
                latest_runs = state_window.loc[
                    state_window["bar_ordinal"].le(
                        int(opportunity.bar_ordinal) + int(checkpoint.bars_since_anchor)
                    )
                ]
                latest_run = latest_runs.iloc[-1]
                latest_candidates = candidates_for((int(period), int(latest_run["anchor_id"])))
                latest_scores = dict(
                    latest_candidates[["cycle_id", "loop_probability"]].itertuples(
                        index=False, name=None
                    )
                )
                evidence_likelihoods: dict[str, float] = {}
                timing_likelihoods: dict[str, float] = {}
                score_ratios: dict[str, float] = {}
                for candidate in candidates.itertuples(index=False):
                    loop = str(candidate.cycle_id)
                    timing = census.timing_likelihood(
                        loop,
                        orientation,
                        anchor_state,
                        observed,
                        int(checkpoint.bars_since_anchor),
                    )
                    raw_ratio = (
                        float(latest_scores.get(loop, float(candidate.loop_probability))) + 1e-6
                    ) / (float(candidate.loop_probability) + 1e-6)
                    score_ratio = min(4.0, max(0.25, raw_ratio))
                    timing_likelihoods[loop] = timing
                    score_ratios[loop] = score_ratio
                    evidence_likelihoods[loop] = timing * score_ratio
                compatibility = update_posterior(initial, cycles, anchor_state, observed)
                full = update_posterior(
                    full_anchor,
                    cycles,
                    anchor_state,
                    observed,
                    evidence_likelihoods=evidence_likelihoods,
                )
                bars_remaining = max(0, 24 - int(checkpoint.bars_since_anchor))
                outcome = _outcome_for_checkpoint(
                    opportunity,
                    checkpoint,
                    execution_by_run,
                    run_id_by_start,
                    bars_2025,
                )
                delayed_outcome = _outcome_for_checkpoint(
                    opportunity,
                    checkpoint,
                    execution_by_run,
                    run_id_by_start,
                    bars_2025,
                    execution_delay_bars=1,
                )
                checkpoint_id = "checkpoint-" + stable_hash(
                    [
                        opportunity.opportunity_id,
                        checkpoint.checkpoint_type,
                        checkpoint.checkpoint_timestamp,
                    ]
                )
                outcome_id = "outcome-" + stable_hash(checkpoint_id)
                outcome_payload = {
                    "run_id": run_id,
                    "outcome_id": outcome_id,
                    "checkpoint_id": checkpoint_id,
                    "opportunity_id": str(opportunity.opportunity_id),
                    "event_lineage_id": str(opportunity.event_lineage_id),
                    "period": int(period),
                    "session_date": str(session),
                    "stock": str(opportunity.symbol_norm),
                    "checkpoint_type": str(checkpoint.checkpoint_type),
                    "checkpoint_timestamp": pd.Timestamp(checkpoint.checkpoint_timestamp),
                    "terminal_timestamp": pd.Timestamp(opportunity.terminal_timestamp),
                    "original_net_payoff_bps": float(opportunity.original_net_payoff_bps)
                    if np.isfinite(float(opportunity.original_net_payoff_bps))
                    else math.nan,
                    **outcome,
                    "one_bar_delay_constant_terminal_net_bps": delayed_outcome[
                        "constant_terminal_net_bps"
                    ],
                }
                remaining_net = outcome_payload["constant_terminal_net_bps"]
                outcome_payload["target_positive_remaining"] = (
                    bool(float(remaining_net) > 0.0)
                    if remaining_net is not None and np.isfinite(float(remaining_net))
                    else pd.NA
                )
                original_net = outcome_payload["original_net_payoff_bps"]
                outcome_payload["capturable_payoff_fraction"] = (
                    float(remaining_net) / float(original_net)
                    if remaining_net is not None
                    and np.isfinite(float(remaining_net))
                    and np.isfinite(float(original_net))
                    and float(original_net) > 0.0
                    else math.nan
                )
                outcome_rows.append(outcome_payload)
                tracks = ["track_b_prior_only"]
                if str(opportunity.population_role) == "named_target":
                    tracks.append("track_a_named_family")
                for track in tracks:
                    track_classes = dict(general_classes)
                    if track == "track_a_named_family":
                        track_classes[str(opportunity.top_loop)] = "good"
                    summary = summarise_posterior(
                        full,
                        track_classes,
                        means,
                        stds,
                        bars_remaining=bars_remaining,
                        target_loop=(
                            str(opportunity.top_loop) if track == "track_a_named_family" else None
                        ),
                    )
                    compatibility_summary = summarise_posterior(
                        compatibility,
                        track_classes,
                        means,
                        stds,
                        bars_remaining=bars_remaining,
                        target_loop=(
                            str(opportunity.top_loop) if track == "track_a_named_family" else None
                        ),
                    )
                    decision = classify_decision(summary, decision_config)
                    eliminated_now = set(full.eliminated) - seen_eliminations
                    payload = {
                        "run_id": run_id,
                        "checkpoint_id": checkpoint_id,
                        "outcome_id": outcome_id,
                        "opportunity_id": str(opportunity.opportunity_id),
                        "event_lineage_id": str(opportunity.event_lineage_id),
                        "track": track,
                        "population_role": str(opportunity.population_role),
                        "period": int(period),
                        "session_date": str(session),
                        "stock": str(opportunity.symbol_norm),
                        "anchor_id": int(opportunity.anchor_id),
                        "target_loop": str(opportunity.top_loop),
                        "target_orientation": orientation,
                        "current_state": anchor_state,
                        "state_history": str(opportunity.history_token),
                        "clock_phase": clock_bin(int(opportunity.bar_ordinal)),
                        "checkpoint_type": str(checkpoint.checkpoint_type),
                        "checkpoint_timestamp": pd.Timestamp(checkpoint.checkpoint_timestamp),
                        "feature_max_availability_timestamp": pd.Timestamp(
                            checkpoint.feature_max_availability_timestamp
                        ),
                        "bars_consumed": int(checkpoint.bars_since_anchor),
                        "bars_remaining": bars_remaining,
                        "resolution_lag_ratio": int(checkpoint.bars_since_anchor) / 24.0,
                        "observed_transitions_json": str(checkpoint.observed_transitions_json),
                        "compatible_loop_count": int(
                            sum(value > 0.0 for value in full.known.values())
                        ),
                        "number_competitors_eliminated": len(full.eliminated),
                        "number_bad_competitors_eliminated": sum(
                            track_classes.get(loop) == "bad" for loop in full.eliminated
                        ),
                        "target_good_loop_mass": float(
                            full.known.get(str(opportunity.top_loop), 0.0)
                        )
                        if track == "track_a_named_family"
                        else 0.0,
                        "compatibility_only_p_positive_remaining": compatibility_summary.p_positive_remaining,
                        "proposed_decision": decision,
                        "reason_codes": f"posterior_{decision}",
                        "posterior_vector_json": json.dumps(
                            {**full.known, "unknown": full.unknown},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "eliminated_loops_json": json.dumps(
                            list(full.eliminated), separators=(",", ":")
                        ),
                        "completed_loops_json": json.dumps(
                            list(full.completed), separators=(",", ":")
                        ),
                        **_summary_payload(summary),
                        "target_remaining_net_bps": remaining_net,
                        "target_positive_remaining": outcome_payload["target_positive_remaining"],
                        "capturable_payoff_fraction": outcome_payload["capturable_payoff_fraction"],
                    }
                    if str(checkpoint.checkpoint_type) == "anchor_freeze":
                        anchor_summaries[track] = summary
                    anchor_summary = anchor_summaries.get(track, summary)
                    payload["anchor_probability"] = anchor_summary.p_positive_remaining
                    payload["anchor_good_mass"] = anchor_summary.good_mass
                    payload["anchor_bad_mass"] = anchor_summary.bad_mass
                    payload["anchor_unknown_mass"] = anchor_summary.unknown_mass
                    payload["bad_mass_change_from_anchor"] = (
                        summary.bad_mass - anchor_summary.bad_mass
                    )
                    payload["good_mass_change_from_anchor"] = (
                        summary.good_mass - anchor_summary.good_mass
                    )
                    payload["unknown_mass_change_from_anchor"] = (
                        summary.unknown_mass - anchor_summary.unknown_mass
                    )
                    payload["entropy_change_from_anchor"] = (
                        summary.normalised_entropy - anchor_summary.normalised_entropy
                    )
                    opportunity_checkpoint_rows.append(payload)
                    for candidate in candidates.itertuples(index=False):
                        loop = str(candidate.cycle_id)
                        posterior_rows.append(
                            {
                                "run_id": run_id,
                                "checkpoint_id": checkpoint_id,
                                "opportunity_id": str(opportunity.opportunity_id),
                                "event_lineage_id": str(opportunity.event_lineage_id),
                                "track": track,
                                "period": int(period),
                                "session_date": str(session),
                                "checkpoint_timestamp": pd.Timestamp(
                                    checkpoint.checkpoint_timestamp
                                ),
                                "checkpoint_type": str(checkpoint.checkpoint_type),
                                "loop_id": loop,
                                "orientation": orientation,
                                "cycle": str(candidate.cycle),
                                "payoff_class": track_classes[loop],
                                "posterior_probability": float(full.known[loop]),
                                "compatibility_status": full.statuses[loop],
                                "clock_lift": clock_lifts[loop],
                                "timing_likelihood": timing_likelihoods[loop],
                                "score_likelihood_ratio": score_ratios[loop],
                                "training_expected_net_bps": means[loop],
                                "training_payoff_std_bps": stds[loop],
                            }
                        )
                    posterior_rows.append(
                        {
                            "run_id": run_id,
                            "checkpoint_id": checkpoint_id,
                            "opportunity_id": str(opportunity.opportunity_id),
                            "event_lineage_id": str(opportunity.event_lineage_id),
                            "track": track,
                            "period": int(period),
                            "session_date": str(session),
                            "checkpoint_timestamp": pd.Timestamp(checkpoint.checkpoint_timestamp),
                            "checkpoint_type": str(checkpoint.checkpoint_type),
                            "loop_id": "__unknown__",
                            "orientation": orientation,
                            "cycle": "unknown",
                            "payoff_class": "unknown",
                            "posterior_probability": full.unknown,
                            "compatibility_status": "residual_unknown",
                            "clock_lift": 1.0,
                            "timing_likelihood": 1.0,
                            "score_likelihood_ratio": 1.0,
                            "training_expected_net_bps": 0.0,
                            "training_payoff_std_bps": 120.0,
                        }
                    )
                for loop in sorted(eliminated_now):
                    elimination_rows.append(
                        {
                            "run_id": run_id,
                            "opportunity_id": str(opportunity.opportunity_id),
                            "event_lineage_id": str(opportunity.event_lineage_id),
                            "checkpoint_id": checkpoint_id,
                            "period": int(period),
                            "session_date": str(session),
                            "loop_id": loop,
                            "checkpoint_type": str(checkpoint.checkpoint_type),
                            "checkpoint_timestamp": pd.Timestamp(checkpoint.checkpoint_timestamp),
                            "bars_consumed": int(checkpoint.bars_since_anchor),
                            "elimination_reason": "observed_transition_prefix_incompatible",
                            "payoff_class": general_classes.get(loop, "unknown"),
                        }
                    )
                seen_eliminations.update(eliminated_now)
            opportunity_frame = pd.DataFrame(opportunity_checkpoint_rows)
            for track, group in opportunity_frame.groupby("track", sort=False):
                eligible = group.loc[
                    group["checkpoint_type"].isin(EVENT_CHECKPOINTS)
                    & group["number_competitors_eliminated"].gt(0)
                    & group["bars_remaining"].ge(6)
                ].sort_values(["checkpoint_timestamp", "checkpoint_type"], kind="stable")
                if not eligible.empty:
                    primary_keys.add((str(eligible.iloc[0]["checkpoint_id"]), str(track)))
            checkpoint_rows.extend(opportunity_checkpoint_rows)

    checkpoints = pd.DataFrame(checkpoint_rows)
    checkpoints["is_primary_resolution"] = [
        (str(checkpoint), str(track)) in primary_keys
        for checkpoint, track in checkpoints[["checkpoint_id", "track"]].itertuples(
            index=False, name=None
        )
    ]
    decisions = apply_irreversible_decisions(
        checkpoints,
        identity_columns=("opportunity_id", "track"),
    )
    return {
        "anchor_sets": pd.DataFrame(anchor_set_rows),
        "posterior": pd.DataFrame(posterior_rows),
        "checkpoints": decisions,
        "outcomes": pd.DataFrame(outcome_rows),
        "eliminations": pd.DataFrame(elimination_rows),
        "census": pd.DataFrame(census_rows).drop_duplicates(
            ["period", "score_session", "loop_id", "orientation", "current_state", "clock_phase"]
        ),
    }


def _auc(target: np.ndarray, probability: np.ndarray) -> float:
    positives = target == 1.0
    negatives = target == 0.0
    if not positives.any() or not negatives.any():
        return math.nan
    order = np.argsort(probability, kind="stable")
    ranks = np.empty(len(probability), dtype=float)
    ranks[order] = np.arange(1, len(probability) + 1, dtype=float)
    _, inverse, counts = np.unique(probability, return_inverse=True, return_counts=True)
    for group in np.flatnonzero(counts > 1):
        positions = np.flatnonzero(inverse == group)
        ranks[positions] = float(np.mean(ranks[positions]))
    positive_ranks = float(ranks[positives].sum())
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    return (positive_ranks - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def probability_metrics(
    target: pd.Series,
    probability: pd.Series,
    payoff: pd.Series,
) -> dict[str, float | int]:
    frame = pd.DataFrame({"target": target, "probability": probability, "payoff": payoff}).dropna()
    if frame.empty:
        return {
            "observable_rows": 0,
            "brier_score": math.nan,
            "log_loss": math.nan,
            "auc": math.nan,
            "ece": math.nan,
            "calibration_intercept": math.nan,
            "calibration_slope": math.nan,
            "rank_payoff_correlation": math.nan,
            "probability_weighted_payoff_bps": math.nan,
        }
    y = frame["target"].astype(float).to_numpy()
    p = np.clip(frame["probability"].astype(float).to_numpy(), 1e-12, 1.0 - 1e-12)
    value = frame["payoff"].astype(float).to_numpy()
    bins = np.minimum(4, np.floor(p * 5.0).astype(int))
    ece = 0.0
    for bin_value in range(5):
        selected = bins == bin_value
        if selected.any():
            ece += float(selected.mean()) * abs(float(y[selected].mean() - p[selected].mean()))
    design = np.column_stack([np.ones(len(p)), p])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    probability_rank = pd.Series(p).rank(method="average").to_numpy()
    payoff_rank = pd.Series(value).rank(method="average").to_numpy()
    rank_correlation = (
        float(np.corrcoef(probability_rank, payoff_rank)[0, 1])
        if len(p) > 1 and float(np.std(probability_rank)) > 0.0 and float(np.std(payoff_rank)) > 0.0
        else math.nan
    )
    return {
        "observable_rows": int(len(frame)),
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "auc": _auc(y, p),
        "ece": ece,
        "calibration_intercept": float(coefficients[0]),
        "calibration_slope": float(coefficients[1]),
        "rank_payoff_correlation": rank_correlation,
        "probability_weighted_payoff_bps": float(np.sum(value * p)),
    }


def build_static_comparators(
    checkpoints: pd.DataFrame,
    opportunities: pd.DataFrame,
    v2_baselines: pd.DataFrame,
) -> pd.DataFrame:
    primary = checkpoints.loc[
        checkpoints["track"].eq("track_b_prior_only") & checkpoints["is_primary_resolution"]
    ].copy()
    opportunity_columns = [
        "opportunity_id",
        "top_loop_probability",
        "top_loop",
        "orientation",
        "original_net_payoff_bps",
        "status",
        "bar_ordinal",
    ]
    primary = primary.merge(
        opportunities[opportunity_columns],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
    )
    denominator = primary["anchor_good_mass"] + primary["anchor_bad_mass"]
    primary["static_good_to_bad_probability"] = np.where(
        denominator.gt(0.0), primary["anchor_good_mass"] / denominator, np.nan
    )
    primary["sequential_probability"] = primary["p_positive_remaining"]
    primary["absolute_target_loop_probability"] = primary["top_loop_probability"]

    baseline = v2_baselines[
        [
            "period",
            "score_session",
            "loop_id",
            "orientation",
            "model_name",
            "p_next_payoff_positive",
        ]
    ].copy()
    pivot = baseline.pivot(
        index=["period", "score_session", "loop_id", "orientation"],
        columns="model_name",
        values="p_next_payoff_positive",
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "score_session": "session_date",
            "loop_id": "top_loop",
            "v1_60_session_selector": "v1_probability",
            "payoff_only_change_point": "payoff_only_probability",
        }
    )
    primary = primary.merge(
        pivot,
        on=["period", "session_date", "top_loop", "orientation"],
        how="left",
        validate="many_to_one",
    )
    return primary.sort_values(
        ["period", "session_date", "stock", "checkpoint_timestamp"], kind="stable"
    ).reset_index(drop=True)


def evaluate_registered_pairs(
    base_pairs: pd.DataFrame,
    scoring: pd.DataFrame,
    runs: pd.DataFrame,
    examples: pd.DataFrame,
    classes: pd.DataFrame,
    *,
    census_config: CensusConfig | None = None,
    excluded_stocks: set[str] | None = None,
) -> pd.DataFrame:
    """Rebuild stock-dependent priors/classes on the frozen primary checkpoint set."""

    excluded = set(excluded_stocks or ())
    pairs = base_pairs.loc[~base_pairs["stock"].astype(str).isin(excluded)].copy()
    scoring_index = scoring.set_index(["period", "anchor_id"]).sort_index()
    cache: dict[tuple[int, int], pd.DataFrame] = {}

    def candidates_for(key: tuple[int, int]) -> pd.DataFrame:
        if key not in cache:
            selected = scoring_index.loc[key]
            if isinstance(selected, pd.Series):
                selected = selected.to_frame().T
            cache[key] = selected.reset_index(drop=True).sort_values("cycle_index", kind="stable")
        return cache[key]

    run_groups = {
        (int(period), str(symbol), str(session)): group.sort_values("bar_ordinal", kind="stable")
        for (period, symbol, session), group in runs.groupby(
            ["period", "symbol_norm", "session_date"], sort=False
        )
        if str(symbol) not in excluded
    }
    class_index = classes.set_index(["period", "score_session", "loop_id", "orientation"])
    cycles = {
        str(row.cycle_id): str(row.cycle)
        for row in scoring[["cycle_id", "cycle"]].drop_duplicates().itertuples(index=False)
    }
    registered_config = census_config or CensusConfig()
    censuses = {
        period: RollingTrainingOnlyCensus(
            examples,
            period=period,
            config=registered_config,
            excluded_stocks=excluded,
        )
        for period in (2023, 2025)
    }
    rows: list[dict[str, object]] = []
    for (period, session), group in pairs.groupby(["period", "session_date"], sort=False):
        census = censuses[int(period)]
        census.advance_before(str(session))
        for pair in group.itertuples(index=False):
            candidates = candidates_for((int(pair.period), int(pair.anchor_id)))
            initial = initial_posterior(
                {
                    str(row.cycle_id): float(row.loop_probability)
                    for row in candidates.itertuples(index=False)
                }
            )
            anchor_state = int(pair.current_state)
            orientation = str(pair.orientation)
            clock_lifts = {
                str(row.cycle_id): census.clock_lift(
                    str(row.cycle_id),
                    orientation,
                    anchor_state,
                    int(pair.bar_ordinal),
                )
                for row in candidates.itertuples(index=False)
            }
            full_anchor = update_posterior(
                initial,
                cycles,
                anchor_state,
                (),
                evidence_likelihoods=clock_lifts,
            )
            payoff_classes, means, stds, _ = _class_maps(
                class_index, int(period), str(session), candidates
            )
            anchor_likelihoods = {
                str(row.cycle_id): census.timing_likelihood(
                    str(row.cycle_id), orientation, anchor_state, (), 0
                )
                for row in candidates.itertuples(index=False)
            }
            anchor_snapshot = update_posterior(
                full_anchor,
                cycles,
                anchor_state,
                (),
                evidence_likelihoods=anchor_likelihoods,
            )
            anchor_summary = summarise_posterior(
                anchor_snapshot,
                payoff_classes,
                means,
                stds,
                bars_remaining=24,
            )
            observed = tuple(
                int(value) for value in json.loads(str(pair.observed_transitions_json))
            )
            state_group = run_groups[(int(period), str(pair.stock), str(pair.session_date))]
            latest = state_group.loc[
                state_group["bar_ordinal"].le(int(pair.bar_ordinal) + int(pair.bars_consumed))
            ].iloc[-1]
            latest_candidates = candidates_for((int(period), int(latest["anchor_id"])))
            latest_scores = dict(
                latest_candidates[["cycle_id", "loop_probability"]].itertuples(
                    index=False, name=None
                )
            )
            likelihoods: dict[str, float] = {}
            for candidate in candidates.itertuples(index=False):
                loop = str(candidate.cycle_id)
                timing = census.timing_likelihood(
                    loop,
                    orientation,
                    anchor_state,
                    observed,
                    int(pair.bars_consumed),
                )
                score_ratio = (
                    float(latest_scores.get(loop, float(candidate.loop_probability))) + 1e-6
                ) / (float(candidate.loop_probability) + 1e-6)
                likelihoods[loop] = timing * min(4.0, max(0.25, score_ratio))
            sequential_snapshot = update_posterior(
                full_anchor,
                cycles,
                anchor_state,
                observed,
                evidence_likelihoods=likelihoods,
            )
            sequential_summary = summarise_posterior(
                sequential_snapshot,
                payoff_classes,
                means,
                stds,
                bars_remaining=int(pair.bars_remaining),
            )
            rows.append(
                {
                    "opportunity_id": str(pair.opportunity_id),
                    "period": int(period),
                    "session_date": str(session),
                    "stock": str(pair.stock),
                    "anchor_probability": anchor_summary.p_positive_remaining,
                    "sequential_probability": sequential_summary.p_positive_remaining,
                    "target_positive": pair.target_positive_remaining,
                    "target_remaining_net_bps": pair.target_remaining_net_bps,
                    "bars_consumed": int(pair.bars_consumed),
                    "bars_remaining": int(pair.bars_remaining),
                    "all_stock_dependent_inputs_rebuilt": True,
                }
            )
    return pd.DataFrame(rows)


def build_model_comparison_metrics(comparators: pd.DataFrame) -> pd.DataFrame:
    prediction_columns = {
        "absolute_target_loop_probability": "absolute_target_loop_probability",
        "static_good_mass_anchor": "anchor_good_mass",
        "static_good_to_bad_anchor": "static_good_to_bad_probability",
        "static_anchor_expected_positive": "anchor_probability",
        "sequential_compatibility_only": "compatibility_only_p_positive_remaining",
        "sequential_transition_dwell_clock": "sequential_probability",
        "v1_60_session_selector": "v1_probability",
        "payoff_only_bocpd": "payoff_only_probability",
    }
    rows: list[dict[str, object]] = []
    slices: list[tuple[str, pd.DataFrame]] = [("all", comparators)]
    slices.extend(
        (str(period), group) for period, group in comparators.groupby("period", sort=True)
    )
    for period_label, frame in slices:
        for model, column in prediction_columns.items():
            metrics = probability_metrics(
                frame["target_positive_remaining"],
                frame[column],
                frame["target_remaining_net_bps"],
            )
            rows.append(
                {
                    "period_slice": period_label,
                    "model": model,
                    "eligible_forecasts": len(frame),
                    **metrics,
                }
            )
    paired = comparators.rename(
        columns={
            "target_positive_remaining": "target_positive",
        }
    )
    paired_metrics = paired_predictive_metrics(
        paired[
            [
                "session_date",
                "target_positive",
                "target_remaining_net_bps",
                "anchor_probability",
                "sequential_probability",
            ]
        ]
    )
    rows.append(
        {
            "period_slice": "all",
            "model": "primary_paired_sequence_minus_anchor",
            "eligible_forecasts": len(comparators),
            **paired_metrics,
        }
    )
    for period, group in paired.groupby("period", sort=True):
        metrics = paired_predictive_metrics(
            group[
                [
                    "session_date",
                    "target_positive",
                    "target_remaining_net_bps",
                    "anchor_probability",
                    "sequential_probability",
                ]
            ]
        )
        rows.append(
            {
                "period_slice": str(period),
                "model": "primary_paired_sequence_minus_anchor",
                "eligible_forecasts": len(group),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _first_policy_rows(checkpoints: pd.DataFrame, track: str) -> pd.DataFrame:
    frame = checkpoints.loc[checkpoints["track"].eq(track)].copy()
    frame = frame.sort_values(
        ["opportunity_id", "checkpoint_timestamp", "checkpoint_type"], kind="stable"
    )
    resolution = frame.loc[frame["is_primary_resolution"]].drop_duplicates(
        "opportunity_id", keep="first"
    )
    resolution["policy_checkpoint_source"] = "primary_first_elimination"
    unresolved_ids = set(frame["opportunity_id"].astype(str)) - set(
        resolution["opportunity_id"].astype(str)
    )
    if unresolved_ids:
        fallback = (
            frame.loc[frame["opportunity_id"].astype(str).isin(unresolved_ids)]
            .sort_values(
                ["opportunity_id", "checkpoint_timestamp", "checkpoint_type"],
                kind="stable",
            )
            .drop_duplicates("opportunity_id", keep="last")
        )
        fallback["policy_checkpoint_source"] = "latest_registered_unresolved"
        resolution = pd.concat([resolution, fallback], ignore_index=True)
    return resolution.sort_values("opportunity_id", kind="stable").reset_index(drop=True)


def build_veto_accounting(
    checkpoints: pd.DataFrame,
    opportunities: pd.DataFrame,
    *,
    track: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = opportunities.loc[opportunities["status"].eq("filled")].copy()
    policy_checkpoint = _first_policy_rows(checkpoints, track)
    fields = [
        "opportunity_id",
        "checkpoint_id",
        "checkpoint_timestamp",
        "checkpoint_type",
        "policy_checkpoint_source",
        "bars_consumed",
        "bars_remaining",
        "bad_loop_mass",
        "good_to_bad_odds",
        "anchor_good_mass",
        "anchor_bad_mass",
        "unknown_loop_mass",
        "decision_state",
        "target_remaining_net_bps",
        "capturable_payoff_fraction",
    ]
    joined = source.merge(
        policy_checkpoint[fields],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for opportunity in joined.itertuples(index=False):
        source_payoff = float(opportunity.original_net_payoff_bps)
        has_checkpoint = not pd.isna(opportunity.checkpoint_id)
        anchor_bad_mass = float(opportunity.anchor_bad_mass)
        anchor_odds = (
            float(opportunity.anchor_good_mass) / anchor_bad_mass
            if has_checkpoint and anchor_bad_mass > 0.0
            else math.inf
        )
        policies = {
            "base_no_rejection": "retained",
            "bad_mass_veto": (
                "rejected"
                if has_checkpoint and float(opportunity.bad_loop_mass) >= 0.5
                else "retained"
            ),
            "static_anchor_good_to_bad_odds_veto": (
                "rejected" if has_checkpoint and anchor_odds <= 1.0 else "retained"
            ),
            "sequential_good_to_bad_odds_veto": (
                "rejected"
                if has_checkpoint and float(opportunity.good_to_bad_odds) <= 1.0
                else "retained"
            ),
            "full_sequential_conservative_veto": (
                "retained"
                if has_checkpoint and str(opportunity.decision_state) == "admit"
                else "rejected"
            ),
            "delayed_admit_after_resolution": (
                "delayed_and_admitted"
                if has_checkpoint
                and str(opportunity.decision_state) == "admit"
                and np.isfinite(float(opportunity.target_remaining_net_bps))
                else "unresolved_until_too_late"
            ),
        }
        for policy, status in policies.items():
            if policy == "delayed_admit_after_resolution":
                policy_payoff = (
                    float(opportunity.target_remaining_net_bps)
                    if status == "delayed_and_admitted"
                    else 0.0
                )
                economic_clock = "constant_terminal_next_open"
            else:
                policy_payoff = source_payoff if status == "retained" else 0.0
                economic_clock = "selection_only_original_clock"
            rows.append(
                {
                    "opportunity_id": str(opportunity.opportunity_id),
                    "event_lineage_id": str(opportunity.event_lineage_id),
                    "track": track,
                    "policy": policy,
                    "period": int(opportunity.period),
                    "session_date": str(opportunity.session_date),
                    "stock": str(opportunity.symbol_norm),
                    "target_loop": str(opportunity.top_loop),
                    "orientation": str(opportunity.orientation),
                    "population_role": str(opportunity.population_role),
                    "decision_status": status,
                    "checkpoint_id": opportunity.checkpoint_id,
                    "checkpoint_type": opportunity.checkpoint_type,
                    "checkpoint_timestamp": opportunity.checkpoint_timestamp,
                    "bars_consumed": opportunity.bars_consumed,
                    "bars_remaining": opportunity.bars_remaining,
                    "source_net_payoff_bps": source_payoff,
                    "policy_net_payoff_bps": policy_payoff,
                    "veto_value_bps": policy_payoff - source_payoff,
                    "capturable_payoff_fraction": opportunity.capturable_payoff_fraction,
                    "economic_clock": economic_clock,
                    "replacement_opportunity_id": pd.NA,
                    "existing_position_action": "unchanged",
                }
            )
    accounting = pd.DataFrame(rows)
    metrics: list[dict[str, object]] = []
    for (policy, period), group in accounting.groupby(["policy", "period"], sort=True):
        rejected = ~group["decision_status"].isin(["retained", "delayed_and_admitted"])
        losses_avoided = -group.loc[
            rejected & group["source_net_payoff_bps"].lt(0.0), "source_net_payoff_bps"
        ].sum()
        profits_rejected = group.loc[
            rejected & group["source_net_payoff_bps"].gt(0.0), "source_net_payoff_bps"
        ].sum()
        metrics.append(
            {
                "track": track,
                "policy": policy,
                "period_slice": str(period),
                "source_opportunities": len(group),
                "retained_or_admitted": int((~rejected).sum()),
                "coverage": float((~rejected).mean()),
                "source_net_payoff_bps": float(group["source_net_payoff_bps"].sum()),
                "policy_net_payoff_bps": float(group["policy_net_payoff_bps"].sum()),
                "veto_value_bps": float(group["veto_value_bps"].sum()),
                "losses_avoided_bps": float(losses_avoided),
                "profits_mistakenly_rejected_bps": float(profits_rejected),
                "net_negative_veto_value_bps": float(losses_avoided - profits_rejected),
                "mean_bars_consumed": float(
                    pd.to_numeric(group["bars_consumed"], errors="coerce").mean()
                ),
            }
        )
    for policy, group in accounting.groupby("policy", sort=True):
        rejected = ~group["decision_status"].isin(["retained", "delayed_and_admitted"])
        losses_avoided = -group.loc[
            rejected & group["source_net_payoff_bps"].lt(0.0), "source_net_payoff_bps"
        ].sum()
        profits_rejected = group.loc[
            rejected & group["source_net_payoff_bps"].gt(0.0), "source_net_payoff_bps"
        ].sum()
        metrics.append(
            {
                "track": track,
                "policy": policy,
                "period_slice": "all",
                "source_opportunities": len(group),
                "retained_or_admitted": int((~rejected).sum()),
                "coverage": float((~rejected).mean()),
                "source_net_payoff_bps": float(group["source_net_payoff_bps"].sum()),
                "policy_net_payoff_bps": float(group["policy_net_payoff_bps"].sum()),
                "veto_value_bps": float(group["veto_value_bps"].sum()),
                "losses_avoided_bps": float(losses_avoided),
                "profits_mistakenly_rejected_bps": float(profits_rejected),
                "net_negative_veto_value_bps": float(losses_avoided - profits_rejected),
                "mean_bars_consumed": float(
                    pd.to_numeric(group["bars_consumed"], errors="coerce").mean()
                ),
            }
        )
    delayed = accounting.loc[accounting["policy"].eq("delayed_admit_after_resolution")].copy()
    return accounting, pd.DataFrame(metrics), delayed


def build_competitor_census(
    anchor_sets: pd.DataFrame,
    eliminations: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> pd.DataFrame:
    named = opportunities.loc[opportunities["population_role"].eq("named_target")][
        [
            "opportunity_id",
            "top_loop",
            "anchor_state",
            "bar_ordinal",
            "original_net_payoff_bps",
        ]
    ].copy()
    named["clock_phase"] = named["bar_ordinal"].map(lambda value: clock_bin(int(value)))
    rows = anchor_sets.merge(named, on="opportunity_id", how="inner", validate="many_to_one")
    rows = rows.loc[~rows["loop_id"].eq(rows["top_loop"])].copy()
    first_elimination = (
        eliminations.sort_values(
            ["opportunity_id", "checkpoint_timestamp", "loop_id"], kind="stable"
        )
        .drop_duplicates(["opportunity_id", "loop_id"], keep="first")[
            ["opportunity_id", "loop_id", "checkpoint_type", "bars_consumed"]
        ]
        .rename(
            columns={
                "checkpoint_type": "usual_elimination_checkpoint",
                "bars_consumed": "elimination_bars_consumed",
            }
        )
    )
    rows = rows.merge(
        first_elimination,
        on=["opportunity_id", "loop_id"],
        how="left",
        validate="one_to_one",
    )
    rows["profitable_target_outcome"] = rows["original_net_payoff_bps"].gt(0.0)
    group_columns = ["top_loop", "loop_id", "payoff_class"]
    summaries: list[dict[str, object]] = []
    for keys, group in rows.groupby(group_columns, sort=True):
        eliminated = group["elimination_bars_consumed"].notna()
        profitable = group["profitable_target_outcome"]
        profitable_eliminated = eliminated & profitable
        losing_eliminated = eliminated & ~profitable
        summaries.append(
            {
                "target_loop": keys[0],
                "competitor_loop": keys[1],
                "competitor_payoff_class": keys[2],
                "compatible_opportunities": group["opportunity_id"].nunique(),
                "mean_anchor_posterior_mass": float(group["initial_posterior_probability"].mean()),
                "frequency_profitable_target": int(profitable.sum()),
                "frequency_losing_target": int((~profitable).sum()),
                "elimination_rate": float(eliminated.mean()),
                "profitable_target_elimination_rate": float(eliminated.loc[profitable].mean()),
                "losing_target_elimination_rate": float(eliminated.loc[~profitable].mean()),
                "median_elimination_bars": float(
                    group.loc[eliminated, "elimination_bars_consumed"].median()
                ),
                "profitable_target_median_elimination_bars": float(
                    group.loc[profitable_eliminated, "elimination_bars_consumed"].median()
                ),
                "losing_target_median_elimination_bars": float(
                    group.loc[losing_eliminated, "elimination_bars_consumed"].median()
                ),
                "survives_until_too_late_rate": float(
                    (~eliminated | group["elimination_bars_consumed"].gt(18)).mean()
                ),
                "mean_payoff_when_eliminated_bps": float(
                    group.loc[eliminated, "original_net_payoff_bps"].mean()
                ),
                "mean_payoff_when_survives_bps": float(
                    group.loc[~eliminated, "original_net_payoff_bps"].mean()
                ),
                "clock_phase_distribution_json": json.dumps(
                    group["clock_phase"].value_counts(normalize=True).sort_index().to_dict(),
                    sort_keys=True,
                ),
                "regime_distribution_json": json.dumps(
                    group["anchor_state"]
                    .astype(str)
                    .value_counts(normalize=True)
                    .sort_index()
                    .to_dict(),
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(summaries)


def build_concentration(comparators: pd.DataFrame, accounting: pd.DataFrame) -> pd.DataFrame:
    primary = comparators.copy()
    primary["net_contribution_bps"] = paired_economic_contribution(primary)
    primary["analysis_scope"] = "primary_paired_economic_increment"
    delayed = accounting.loc[accounting["policy"].eq("delayed_admit_after_resolution")].copy()
    delayed["net_contribution_bps"] = delayed["policy_net_payoff_bps"]
    delayed["analysis_scope"] = "delayed_admission_policy_payoff"
    rows: list[dict[str, object]] = []
    for selected in (primary, delayed):
        scope = str(selected["analysis_scope"].iloc[0])
        for dimension, column in {
            "stock": "stock",
            "loop": "target_loop",
            "orientation": "orientation",
            "period": "period",
            "population": "population_role",
        }.items():
            contributions = selected.groupby(column, dropna=False)["net_contribution_bps"].sum(
                min_count=1
            )
            order = contributions.abs().sort_values(ascending=False).index
            contributions = contributions.reindex(order)
            absolute_total = float(contributions.abs().sum())
            for rank, (key, value) in enumerate(contributions.items(), start=1):
                rows.append(
                    {
                        "analysis_scope": scope,
                        "dimension": dimension,
                        "key": str(key),
                        "rank": rank,
                        "net_contribution_bps": float(value),
                        "absolute_contribution_share": (
                            abs(float(value)) / absolute_total if absolute_total > 0.0 else math.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_stress_results(
    comparators: pd.DataFrame,
    outcomes: pd.DataFrame,
    accounting: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def paired_result(label: str, frame: pd.DataFrame, target_column: str) -> None:
        current = frame.copy()
        current["target_positive"] = (
            current[target_column].gt(0.0).where(current[target_column].notna(), pd.NA)
        )
        current["target_remaining_net_bps"] = current[target_column]
        metrics = paired_predictive_metrics(
            current[
                [
                    "session_date",
                    "target_positive",
                    "target_remaining_net_bps",
                    "anchor_probability",
                    "sequential_probability",
                ]
            ],
            bootstrap_resamples=500,
        )
        rows.append({"stress_test": label, **metrics})

    paired_result("registered_primary", comparators, "target_remaining_net_bps")
    twice_cost = comparators.copy()
    twice_cost["twice_cost_net_bps"] = twice_cost["target_remaining_net_bps"] - 10.0
    paired_result("twice_costs", twice_cost, "twice_cost_net_bps")
    delayed = outcomes[
        ["checkpoint_id", "one_bar_delay_constant_terminal_net_bps"]
    ].drop_duplicates("checkpoint_id")
    one_bar = comparators.merge(delayed, on="checkpoint_id", how="left", validate="one_to_one")
    paired_result(
        "one_additional_bar_execution_delay",
        one_bar,
        "one_bar_delay_constant_terminal_net_bps",
    )
    for period, group in comparators.groupby("period", sort=True):
        paired_result(f"period_{period}", group, "target_remaining_net_bps")

    stock_contribution = (
        accounting.loc[accounting["policy"].eq("delayed_admit_after_resolution")]
        .groupby("stock")["policy_net_payoff_bps"]
        .sum()
        .sort_values(ascending=False)
    )
    for count in (1, 5):
        excluded = set(stock_contribution.head(count).index.astype(str))
        subset = comparators.loc[~comparators["stock"].astype(str).isin(excluded)]
        paired_result(
            f"remove_top_{count}_contributing_stocks_attribution_only",
            subset,
            "target_remaining_net_bps",
        )
    episode_contribution = comparators.sort_values(
        "target_remaining_net_bps", ascending=False, kind="stable"
    )
    for count in (1, 5):
        excluded_ids = set(episode_contribution.head(count)["opportunity_id"].astype(str))
        subset = comparators.loc[~comparators["opportunity_id"].astype(str).isin(excluded_ids)]
        paired_result(
            f"remove_top_{count}_opportunity_episodes",
            subset,
            "target_remaining_net_bps",
        )
    for label, reason in {
        "median_session_aggregation": "pending_fully_rebuilt_v2_classification",
        "leave_one_stock_out": "pending_fully_rebuilt_stock_dependent_states",
        "weakest_liquidity_exclusion": "not_executable_no_hash_pinned_2023_liquidity_field",
        "minimum_two_bar_state_dwell": "reported_in_checkpoint_shape_diagnostics",
        "coarse_clock_bins": "pending_full_census_rebuild",
        "prior_smoothing_alpha_0.5": "pending_full_census_rebuild",
        "prior_smoothing_alpha_2.0": "pending_full_census_rebuild",
    }.items():
        rows.append({"stress_test": label, "status": reason})
    return pd.DataFrame(rows)


def _paired_stress_row(
    label: str,
    pairs: pd.DataFrame,
    *,
    detail: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metrics = paired_predictive_metrics(
        pairs[
            [
                "session_date",
                "target_positive",
                "target_remaining_net_bps",
                "anchor_probability",
                "sequential_probability",
            ]
        ],
        bootstrap_resamples=500,
    )
    return {
        "stress_test": label,
        **metrics,
        "rebuild_detail_json": json.dumps(
            safe_json(dict(detail or {})), sort_keys=True, separators=(",", ":")
        ),
    }


def attach_hindsight_episodes(
    pairs: pd.DataFrame,
    episode_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    frame = pairs.copy()
    frame["episode_id"] = pd.NA
    frame["target_episode_state"] = "non_positive_or_unlabelled"
    positive = episode_diagnostics.loc[
        pd.to_numeric(episode_diagnostics["total_episode_payoff_bps"], errors="coerce").gt(0.0)
    ].copy()
    for episode in positive.itertuples(index=False):
        selected = (
            frame["period"].eq(int(episode.period))
            & frame["top_loop"].eq(str(episode.loop_id))
            & frame["orientation"].eq(str(episode.orientation))
            & frame["session_date"].astype(str).ge(str(episode.hindsight_estimated_onset))
            & frame["session_date"].astype(str).le(str(episode.hindsight_estimated_end))
        )
        frame.loc[selected, "episode_id"] = str(episode.episode_id)
        frame.loc[selected, "target_episode_state"] = "hindsight_positive_episode"
    return frame


def run_rebuilt_sensitivities(
    *,
    paths: Mapping[str, Path],
    base_pairs: pd.DataFrame,
    scoring: pd.DataFrame,
    runs: pd.DataFrame,
    examples: pd.DataFrame,
    primary_classes: pd.DataFrame,
    accounting: pd.DataFrame,
    bars_2025: Mapping[tuple[str, str], pd.DataFrame],
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    loo_rows: list[dict[str, object]] = []
    context = prepare_v2_rebuild_context(paths)

    primary_rebuild = evaluate_registered_pairs(
        base_pairs, scoring, runs, examples, primary_classes
    )
    if not np.allclose(
        primary_rebuild["anchor_probability"],
        base_pairs.set_index("opportunity_id")
        .loc[primary_rebuild["opportunity_id"], "anchor_probability"]
        .to_numpy(float),
        atol=1e-12,
        rtol=0.0,
    ) or not np.allclose(
        primary_rebuild["sequential_probability"],
        base_pairs.set_index("opportunity_id")
        .loc[primary_rebuild["opportunity_id"], "sequential_probability"]
        .to_numpy(float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError("compact registered-pair rebuild differs from primary ledger")
    rows.append(
        _paired_stress_row(
            "registered_primary_compact_rebuild",
            primary_rebuild,
            detail={"all_stock_dependent_inputs_rebuilt": True},
        )
    )

    median_classes, median_detail = rebuild_payoff_classes(
        context,
        aggregation="median",
        run_id=run_id,
    )
    median_pairs = evaluate_registered_pairs(base_pairs, scoring, runs, examples, median_classes)
    rows.append(
        _paired_stress_row("median_session_aggregation", median_pairs, detail=median_detail)
    )
    print(json.dumps({"rebuild": "median", "status": "complete"}), flush=True)

    for hazard in context["config"]["change_point"]["predeclared_hazard_sensitivities"]:
        hazard_classes, hazard_detail = rebuild_payoff_classes(
            context,
            hazard=float(hazard),
            run_id=run_id,
        )
        hazard_pairs = evaluate_registered_pairs(
            base_pairs, scoring, runs, examples, hazard_classes
        )
        rows.append(
            _paired_stress_row(
                f"v2_hazard_{float(hazard):.6f}",
                hazard_pairs,
                detail=hazard_detail,
            )
        )
        print(
            json.dumps({"rebuild": f"hazard_{float(hazard):.6f}", "status": "complete"}),
            flush=True,
        )

    for label, config in {
        "coarse_clock_bins": CensusConfig(coarse_clock=True),
        "prior_smoothing_alpha_0.5": CensusConfig(smoothing_alpha=0.5),
        "prior_smoothing_alpha_2.0": CensusConfig(smoothing_alpha=2.0),
    }.items():
        pairs = evaluate_registered_pairs(
            base_pairs,
            scoring,
            runs,
            examples,
            primary_classes,
            census_config=config,
        )
        rows.append(
            _paired_stress_row(
                label,
                pairs,
                detail={
                    "census_rebuilt": True,
                    "coarse_clock": config.coarse_clock,
                    "smoothing_alpha": config.smoothing_alpha,
                },
            )
        )

    two_bar = primary_rebuild.loc[primary_rebuild["bars_consumed"].ge(2)].copy()
    rows.append(
        _paired_stress_row(
            "minimum_two_bar_state_dwell",
            two_bar,
            detail={
                "sensitivity_filter_only": True,
                "primary_state_definition_unchanged": True,
            },
        )
    )

    stock_contribution = (
        accounting.loc[
            accounting["policy"].eq("delayed_admit_after_resolution")
            & accounting["track"].eq("track_b_prior_only")
        ]
        .groupby("stock")["policy_net_payoff_bps"]
        .sum()
        .sort_values(ascending=False)
    )
    best_stock = str(stock_contribution.index[0])
    top_five = set(stock_contribution.head(5).index.astype(str))
    symbols = sorted(base_pairs["stock"].astype(str).unique())
    for index, stock in enumerate(symbols, start=1):
        rebuilt_classes, detail = rebuild_payoff_classes(
            context,
            excluded_stocks={stock},
            run_id=run_id,
        )
        pairs = evaluate_registered_pairs(
            base_pairs,
            scoring,
            runs,
            examples,
            rebuilt_classes,
            excluded_stocks={stock},
        )
        metrics = paired_predictive_metrics(
            pairs[
                [
                    "session_date",
                    "target_positive",
                    "target_remaining_net_bps",
                    "anchor_probability",
                    "sequential_probability",
                ]
            ],
            bootstrap_resamples=500,
        )
        loo_rows.append(
            {
                "excluded_stock": stock,
                **metrics,
                **detail,
            }
        )
        if stock == best_stock:
            rows.append(
                {
                    "stress_test": "remove_best_stock_fully_rebuilt",
                    **metrics,
                    "rebuild_detail_json": json.dumps(
                        safe_json(detail), sort_keys=True, separators=(",", ":")
                    ),
                }
            )
        print(
            json.dumps(
                {"rebuild": "leave_one_stock_out", "stock": stock, "index": index},
                sort_keys=True,
            ),
            flush=True,
        )

    if top_five:
        joint_classes, detail = rebuild_payoff_classes(
            context,
            excluded_stocks=top_five,
            run_id=run_id,
        )
        joint_pairs = evaluate_registered_pairs(
            base_pairs,
            scoring,
            runs,
            examples,
            joint_classes,
            excluded_stocks=top_five,
        )
        rows.append(
            _paired_stress_row(
                "remove_top_five_stocks_fully_rebuilt",
                joint_pairs,
                detail=detail,
            )
        )

    liquidity = {
        stock: float(
            pd.concat(
                [
                    frame.assign(dollar_volume=frame["close"] * frame["volume"])["dollar_volume"]
                    for (candidate, _), frame in bars_2025.items()
                    if candidate == stock and "volume" in frame
                ],
                ignore_index=True,
            ).median()
        )
        for stock in symbols
    }
    weakest_count = max(1, int(math.ceil(0.2 * len(symbols))))
    weakest = set(sorted(liquidity, key=liquidity.get)[:weakest_count])
    liquidity_classes, liquidity_detail = rebuild_payoff_classes(
        context,
        excluded_stocks=weakest,
        run_id=run_id,
    )
    liquidity_pairs = evaluate_registered_pairs(
        base_pairs.loc[base_pairs["period"].eq(2025)],
        scoring,
        runs,
        examples,
        liquidity_classes,
        excluded_stocks=weakest,
    )
    rows.append(
        _paired_stress_row(
            "weakest_liquidity_exclusion",
            liquidity_pairs,
            detail={
                **liquidity_detail,
                "period_scope": 2025,
                "liquidity_measure": "median_5m_close_times_volume",
            },
        )
    )

    episodes = pd.read_parquet(paths["v2_episode_diagnostics"])
    episode_pairs = attach_hindsight_episodes(base_pairs, episodes)
    contribution = (
        episode_pairs.dropna(subset=["episode_id"])
        .groupby("episode_id")["target_remaining_net_bps"]
        .sum()
        .sort_values(ascending=False)
    )
    for count in (1, 5):
        removed = set(contribution.head(count).index.astype(str))
        subset = episode_pairs.loc[~episode_pairs["episode_id"].astype(str).isin(removed)].copy()
        subset = subset.rename(columns={"target_positive_remaining": "target_positive"})
        rows.append(
            _paired_stress_row(
                f"remove_top_{count}_hindsight_episodes",
                subset,
                detail={
                    "episode_labels_evaluation_only": True,
                    "removed_episode_ids": sorted(removed),
                },
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(loo_rows), episode_pairs


def build_named_and_general_results(
    comparators: pd.DataFrame,
    checkpoints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    named_rows: list[dict[str, object]] = []
    named = checkpoints.loc[
        checkpoints["track"].eq("track_a_named_family") & checkpoints["is_primary_resolution"]
    ].copy()
    for (target_loop, period), group in named.groupby(["target_loop", "period"], sort=True):
        target = pd.to_numeric(group["target_remaining_net_bps"], errors="coerce")
        named_rows.append(
            {
                "target_loop": target_loop,
                "period": int(period),
                "eligible_resolutions": len(group),
                "observable_remaining_payoffs": int(target.notna().sum()),
                "mean_remaining_net_bps": float(target.mean()),
                "positive_remaining_rate": float(target.gt(0.0).mean()),
                "mean_bad_mass_change": float(group["bad_mass_change_from_anchor"].mean()),
                "mean_good_mass_change": float(group["good_mass_change_from_anchor"].mean()),
                "mean_resolution_lag_ratio": float(group["resolution_lag_ratio"].mean()),
                "mean_capturable_payoff_fraction": float(
                    pd.to_numeric(group["capturable_payoff_fraction"], errors="coerce").mean()
                ),
            }
        )
    general_rows: list[dict[str, object]] = []
    for period, group in comparators.groupby("period", sort=True):
        metrics = paired_predictive_metrics(
            group.rename(columns={"target_positive_remaining": "target_positive"})[
                [
                    "session_date",
                    "target_positive",
                    "target_remaining_net_bps",
                    "anchor_probability",
                    "sequential_probability",
                ]
            ]
        )
        general_rows.append({"period": int(period), **metrics})
    return pd.DataFrame(named_rows), pd.DataFrame(general_rows)


def make_plots(
    output: Path,
    checkpoints: pd.DataFrame,
    comparators: pd.DataFrame,
    accounting_metrics: pd.DataFrame,
) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/stocker-mplconfig")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: list[Path] = []

    def save(name: str) -> Path:
        path = output / name
        plt.tight_layout()
        plt.savefig(path, dpi=130, metadata={"Software": MODEL_VERSION})
        plt.close()
        plot_paths.append(path)
        return path

    primary = checkpoints.loc[
        checkpoints["track"].eq("track_b_prior_only") & checkpoints["is_primary_resolution"]
    ].copy()
    if not primary.empty:
        chosen = primary.sort_values(
            "target_remaining_net_bps", ascending=False, kind="stable"
        ).iloc[0]
        timeline = checkpoints.loc[
            checkpoints["opportunity_id"].eq(chosen["opportunity_id"])
            & checkpoints["track"].eq("track_b_prior_only")
        ].sort_values("bars_consumed", kind="stable")
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(timeline["bars_consumed"], timeline["all_good_loop_mass"], label="good")
        plt.plot(timeline["bars_consumed"], timeline["bad_loop_mass"], label="bad")
        plt.plot(timeline["bars_consumed"], timeline["unknown_loop_mass"], label="unknown")
        plt.xlabel("Bars consumed")
        plt.ylabel("Posterior mass")
        plt.title("Representative successful resolution")
        plt.legend()
        save("posterior_timeline_success.png")

        false_rows = primary.loc[primary["target_remaining_net_bps"].lt(0.0)]
        if not false_rows.empty:
            chosen = false_rows.sort_values("target_remaining_net_bps", kind="stable").iloc[0]
            timeline = checkpoints.loc[
                checkpoints["opportunity_id"].eq(chosen["opportunity_id"])
                & checkpoints["track"].eq("track_b_prior_only")
            ].sort_values("bars_consumed", kind="stable")
            plt.figure(figsize=(7.2, 4.2))
            plt.plot(timeline["bars_consumed"], timeline["all_good_loop_mass"], label="good")
            plt.plot(timeline["bars_consumed"], timeline["bad_loop_mass"], label="bad")
            plt.plot(timeline["bars_consumed"], timeline["unknown_loop_mass"], label="unknown")
            plt.xlabel("Bars consumed")
            plt.ylabel("Posterior mass")
            plt.title("Representative false resolution")
            plt.legend()
            save("posterior_timeline_false.png")

    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(
        comparators["resolution_lag_ratio"],
        comparators["target_remaining_net_bps"],
        s=8,
        alpha=0.35,
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Competitor-resolution lag ratio")
    plt.ylabel("Constant-terminal remaining net bps")
    plt.title("Resolution timing versus remaining payoff")
    save("resolution_lag_vs_remaining_payoff.png")

    overall = accounting_metrics.loc[accounting_metrics["period_slice"].eq("all")]
    plt.figure(figsize=(6.4, 4.2))
    plt.scatter(overall["coverage"], overall["veto_value_bps"], s=45)
    for row in overall.itertuples(index=False):
        plt.annotate(str(row.policy), (row.coverage, row.veto_value_bps), fontsize=7)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Coverage")
    plt.ylabel("Veto value bps")
    plt.title("Veto coverage versus value")
    save("veto_coverage_vs_value.png")

    plt.figure(figsize=(6.0, 5.0))
    plt.scatter(
        comparators["anchor_probability"],
        comparators["sequential_probability"],
        s=8,
        alpha=0.3,
    )
    plt.plot([0, 1], [0, 1], color="black", linewidth=0.8)
    plt.xlabel("Anchor positive-payoff probability")
    plt.ylabel("Sequential positive-payoff probability")
    plt.title("Static anchor versus sequential posterior")
    save("static_anchor_vs_sequential_probability.png")
    return plot_paths


def _annotate_table(
    frame: pd.DataFrame,
    *,
    run_id: str,
    contract_hash: str,
    data_snapshot_id: str,
) -> pd.DataFrame:
    result = frame.copy()
    annotations = {
        "experiment_run_id": run_id,
        "contract_hash": contract_hash,
        "data_snapshot_id": data_snapshot_id,
        "experiment_model_version": MODEL_VERSION,
    }
    for name, value in reversed(list(annotations.items())):
        if name in result.columns:
            result[name] = value
        else:
            result.insert(0, name, value)
    return result


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def scientific_decision(
    model_metrics: pd.DataFrame,
    veto_metrics: pd.DataFrame,
    comparators: pd.DataFrame,
) -> str:
    paired = model_metrics.loc[
        model_metrics["model"].eq("primary_paired_sequence_minus_anchor")
        & model_metrics["period_slice"].eq("all")
    ]
    if paired.empty or int(paired.iloc[0].get("paired_rows", 0)) == 0:
        return "insufficient_support"
    row = paired.iloc[0]
    brier = float(row["brier_improvement"])
    economic = float(row["paired_economic_increment_bps"])
    if brier <= 0.0:
        return "static_anchor_information_sufficient_no_sequence_increment"
    if economic <= 0.0:
        return "competitor_posterior_not_payoff_informative"
    lag = float(pd.to_numeric(comparators["resolution_lag_ratio"], errors="coerce").median())
    if lag >= 0.5:
        return "competitor_elimination_predictive_but_too_late"
    veto = veto_metrics.loc[
        veto_metrics["policy"].eq("full_sequential_conservative_veto")
        & veto_metrics["period_slice"].eq("all")
    ]
    if not veto.empty and float(veto.iloc[0]["net_negative_veto_value_bps"]) > 0.0:
        return "sequential_competitor_veto_supported_prospectively_only_required"
    return "rejection_value_supported_admission_unknown"


def write_report(
    report_path: Path,
    *,
    metadata: Mapping[str, Any],
    opportunities: pd.DataFrame,
    comparators: pd.DataFrame,
    model_metrics: pd.DataFrame,
    veto_metrics: pd.DataFrame,
    competitor_census: pd.DataFrame,
    named_results: pd.DataFrame,
    stress_results: pd.DataFrame,
    concentration: pd.DataFrame,
    audit: Mapping[str, Any] | None = None,
) -> None:
    paired = model_metrics.loc[
        model_metrics["model"].eq("primary_paired_sequence_minus_anchor")
        & model_metrics["period_slice"].eq("all")
    ].iloc[0]
    decision = str(metadata["scientific_decision"])
    full_veto = veto_metrics.loc[
        veto_metrics["policy"].eq("full_sequential_conservative_veto")
        & veto_metrics["period_slice"].eq("all")
    ].iloc[0]
    static_odds = veto_metrics.loc[
        veto_metrics["policy"].eq("static_anchor_good_to_bad_odds_veto")
        & veto_metrics["period_slice"].eq("all")
    ].iloc[0]
    sequential_odds = veto_metrics.loc[
        veto_metrics["policy"].eq("sequential_good_to_bad_odds_veto")
        & veto_metrics["period_slice"].eq("all")
    ].iloc[0]
    delayed = veto_metrics.loc[
        veto_metrics["policy"].eq("delayed_admit_after_resolution")
        & veto_metrics["period_slice"].eq("all")
    ].iloc[0]
    major = (
        competitor_census.sort_values(
            ["target_loop", "compatible_opportunities", "mean_anchor_posterior_mass"],
            ascending=[True, False, False],
            kind="stable",
        )
        .groupby("target_loop", sort=True)
        .head(5)
    )
    major_lines = (
        "\n".join(
            f"- {row.target_loop}: {row.competitor_loop} ({row.competitor_payoff_class}), "
            f"n={row.compatible_opportunities}, anchor mass={row.mean_anchor_posterior_mass:.3f}, "
            f"elimination rate={row.elimination_rate:.1%}, median={row.median_elimination_bars:.1f} bars"
            for row in major.itertuples(index=False)
        )
        or "- No supported named-family competitor rows."
    )
    period_rows = model_metrics.loc[
        model_metrics["model"].eq("primary_paired_sequence_minus_anchor")
        & ~model_metrics["period_slice"].eq("all")
    ]
    period_lines = "\n".join(
        f"- {row.period_slice}: n={int(row.paired_rows)}, Brier improvement="
        f"{row.brier_improvement:.6f}, economic increment={row.paired_economic_increment_bps:.2f} bps"
        for row in period_rows.itertuples(index=False)
    )
    concentration_top = concentration.loc[
        concentration["analysis_scope"].eq("primary_paired_economic_increment")
        & concentration["rank"].eq(1)
    ].sort_values("dimension", kind="stable")
    concentration_lines = "\n".join(
        f"- {row.dimension}: {row.key}, contribution={row.net_contribution_bps:.2f} bps, "
        f"absolute share={row.absolute_contribution_share:.1%}"
        for row in concentration_top.itertuples(index=False)
    )
    audit_text = (
        f"{audit.get('status')} ({audit.get('passed_checks')}/{audit.get('total_checks')} checks)"
        if audit
        else "pending separate independent auditor"
    )
    report = f"""# Sequential Loop Competitor Veto V1

## 1–4. Hypothesis, prior boundary, status, and frozen targets

The registered hypothesis is that positive remaining h24 payoff becomes identifiable when causal state, transition, dwell, and clock evidence eliminates bad compatible loops while good-loop mass survives. This differs from prior static top-loop, absolute-score, full-vector, realised-route, and early price-bar tests: it represents a normalised compatible set, explicit unknown mass, past-only economic classes, and one-versus-rest elimination through time.

The exact sequential good/bad/unknown veto had **not** previously been tested. All 2023/2025 results are opened-data mechanism attribution, never validation or trading approval. The frozen named targets are `cycle_04|state_4` (`2->4->2`) and `cycle_07|state_5` (`5->6->5`); the negative and neutral controls remain `cycle_07|state_6` and `cycle_04|state_2`. No replacement family was selected.

## 5–11. Classification, dictionary, census, posterior, causality, and checkpoints

The general track classifies each loop/orientation from the frozen V2 no-leading-feature hierarchy before the scoring session: GOOD requires support and a one-sided 90% lower bound above zero; BAD requires a one-sided 90% upper bound below zero; otherwise UNKNOWN. Economic support remains session × loop × orientation × h24, never raw fills.

The frozen twenty-loop dictionary is rotated to the anchor state. Anchor known mass is the frozen structural score and residual mass remains explicitly unknown. A known loop reaches zero only after a completed observable transition makes every matching rotation impossible. Clock and transition-timing likelihoods use strictly earlier sessions within the same period, Beta smoothing, and empirical-Bayes pooling. Forecast inputs exclude direction, payoff, realised loop/child, future state, MFE/MAE, and hindsight episode labels.

Registered checkpoints are anchor freeze; completed transitions one and two; parent completion, route diversion, and incompatibility when causally observable; and fixed bars 1, 2, 3, and 6. The primary endpoint was frozen as the first event checkpoint eliminating at least one competitor with at least six bars remaining. No checkpoint was selected after scoring.

## 12–18. Anchor baseline, sequence, veto, delayed admission, and payoff clocks

Primary paired rows: **{int(paired.paired_rows):,}**. Sequential-minus-anchor Brier improvement was **{paired.brier_improvement:.6f}** (session-block 95% interval {paired.brier_interval_lower:.6f} to {paired.brier_interval_upper:.6f}); log-loss improvement was **{paired.log_loss_improvement:.6f}**; probability-weighted paired economic increment was **{paired.paired_economic_increment_bps:.2f} bps**.

{period_lines}

The negative-veto accounting keeps one immutable source row and never refills overlap/capacity. The full conservative veto avoided **{full_veto.losses_avoided_bps:.2f} bps** of losses and mistakenly rejected **{full_veto.profits_mistakenly_rejected_bps:.2f} bps** of profits, for **{full_veto.net_negative_veto_value_bps:.2f} bps** net veto value at **{full_veto.coverage:.1%}** coverage. This is a selection-only diagnostic on the original entry clock.

The static anchor good-to-bad odds veto produced **{static_odds.net_negative_veto_value_bps:.2f} bps** veto value at **{static_odds.coverage:.1%}** coverage. The identically thresholded sequential odds veto produced **{sequential_odds.net_negative_veto_value_bps:.2f} bps** at **{sequential_odds.coverage:.1%}**, an incremental **{sequential_odds.net_negative_veto_value_bps - static_odds.net_negative_veto_value_bps:.2f} bps** over the anchor comparator.

The separate executable translation waits until causal admission, enters at the next provider open, and retains the original anchor-plus-24 terminal clock. It produced **{delayed.policy_net_payoff_bps:.2f} bps** at **{delayed.coverage:.1%}** coverage. Restarted-h24 payoff is exported only as a secondary table and is never mixed with constant-terminal payoff.

## 19–26. Earliness, capture, errors, targets, competitors, regime, clock, and costs

Median resolution-lag ratio was **{comparators.resolution_lag_ratio.median():.3f}**. Among positive original opportunities with observable remaining paths, mean capturable payoff fraction was **{pd.to_numeric(comparators.capturable_payoff_fraction, errors="coerce").mean():.3f}**. MFE/MAE are outcome diagnostics only and never posterior inputs.

Major named-family competitors:

{major_lines}

Named-family result rows and the full regime × clock census are machine-readable. Competitor prevalence is reported by anchor regime and clock phase; elimination timing is split by profitable versus losing target outcomes. Unsupported competitor pairs remain unknown.

Twice-cost and one-bar-delay results are in `stress_test_results.csv`. The 2023 intermediate tape is unavailable: event-checkpoint next opens and terminal closes are reconstructed from hash-pinned frozen execution anchors, while 2023 fixed-bar payoffs, one-bar delays, and MFE/MAE remain missing—not zero.

## 27–31. Robustness, concentration, chatter, and failure cases

{concentration_lines}

The stress ledger distinguishes fully rebuilt analyses from attribution-only exclusions and unavailable liquidity tests. A minimum-two-bar event filter is a sensitivity only; it never changes the primary detector state definition. Fixed-bar rows in 2023 cannot support complete economic timing diagnostics. The general prior-only track can also remain almost entirely UNKNOWN where frozen V2 support is sparse, which is a substantive failure mode rather than an invitation to weaken thresholds.

Independent audit: **{audit_text}**.

## 32–34. Scientific decision and recommendation

Scientific decision: **`{decision}`**.

This retrospective result answers whether competitors can be rejected early enough only under the frozen causal and coverage tests above. Structural identification alone is not treated as economic correctness, and no opened-data result authorises live or paper trading.

Exact next recommendation: prospectively log the immutable anchor and checkpoint posterior ledger on genuinely unopened sessions, then settle constant-terminal outcomes separately; do not alter the target families, checkpoints, thresholds, smoothing, or class rules until that prospective cohort is complete.

## Reproducibility and field assumptions

- Run ID: `{metadata["run_id"]}`
- Git SHA: `{metadata["git_sha"]}`
- Contract SHA-256: `{metadata["contract_hash"]}`
- Data snapshot: `{metadata["data_snapshot_id"]}`
- Source opportunities: {len(opportunities):,}; filled: {int(opportunities.status.eq("filled").sum()):,}
- Decision timestamp: end of the anchor bar (`start_timestamp + 5 minutes`). Transition evidence freezes only after the first bar in the new state completes.
- Original terminal: anchor start plus 125 minutes, matching next-open entry plus 24 completed five-minute bars.
- The frozen source ledger's `bar_ordinal` differs from the structural scoring ordinal for some 2023 rows; checkpoint identity uses the scoring anchor ID/timestamp and recomputed regular-session ordinal, while the source ordinal is retained separately.
- Structural loop prediction and economic payoff classification remain separate layers.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")


def build_prospective_schema() -> dict[str, object]:
    return {
        "schema_version": "sequential_loop_competitor_prospective_v1",
        "forecast_create_only": True,
        "outcome_create_only": True,
        "execution_enabled": False,
        "required_forecast_fields": sorted(
            [
                "run_id",
                "git_sha",
                "contract_hash",
                "data_snapshot_hash",
                "model_version",
                "forecast_id",
                "event_lineage_id",
                "opportunity_id",
                "anchor_id",
                "stock",
                "session",
                "decision_timestamp",
                "checkpoint_timestamp",
                "checkpoint_type",
                "bars_since_anchor",
                "bars_remaining",
                "current_state",
                "state_history",
                "clock_phase",
                "compatible_loop_set",
                "loop_posterior",
                "good_loop_mass",
                "bad_loop_mass",
                "unknown_loop_mass",
                "entropy",
                "competitor_eliminations",
                "decision_state",
                "reason_codes",
                "freeze_timestamp",
                "feature_availability_timestamps",
                "training_cutoff",
            ]
        ),
        "holdout_rejects_opened_years": [2023, 2024, 2025, 2026],
        "settlement_is_separate_from_forecast": True,
    }


def artifact_manifest(output: Path) -> dict[str, str]:
    return {
        path.name: sha256(path)
        for path in sorted(output.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name not in {"artifact_manifest.json"}
    }


def verify_exact_rerun(output: Path, primary: Path) -> dict[str, object]:
    extensions = {".parquet", ".csv", ".json"}
    excluded = {"artifact_manifest.json", "independent_audit.json"}
    output_files = {
        path.name: path
        for path in output.iterdir()
        if path.is_file() and path.suffix in extensions and path.name not in excluded
    }
    primary_files = {
        path.name: path
        for path in primary.iterdir()
        if path.is_file() and path.suffix in extensions and path.name not in excluded
    }
    missing = sorted(set(primary_files) - set(output_files))
    extra = sorted(set(output_files) - set(primary_files))
    mismatches = sorted(
        name
        for name in set(output_files) & set(primary_files)
        if sha256(output_files[name]) != sha256(primary_files[name])
    )
    return {
        "primary_path": str(primary),
        "compared_machine_readable_files": len(primary_files),
        "missing_files": missing,
        "extra_files": extra,
        "hash_mismatches": mismatches,
        "byte_identical": not missing and not extra and not mismatches,
    }


def run_historical(
    *,
    output: Path,
    report_path: Path,
    exact_rerun_of: Path | None,
) -> None:
    contract, source_hashes, data_snapshot_id = verify_contract_and_inputs()
    contract_hash = sha256(CONTRACT_PATH)
    git_sha = git_value("rev-parse", "HEAD")
    branch = git_value("branch", "--show-current")
    run_id = "sequential-veto-" + stable_hash(
        [contract_hash, data_snapshot_id, git_sha, MODEL_VERSION]
    )
    paths = input_paths(contract)
    scoring, runs, examples = load_structural_surfaces(paths)
    opportunities = load_opportunities(paths, scoring, runs)
    classes, v2_baselines = load_payoff_classes(paths)
    execution = load_execution_anchors(paths)
    symbols = sorted(opportunities["symbol_norm"].astype(str).unique())
    source_catalogue = json.loads(paths["v1_source_hashes"].read_text())["sha256"]
    provider_root = Path(contract["inputs"]["provider_2025_root"])
    for symbol in symbols:
        provider = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        expected = source_catalogue.get(f"provider_2025_{symbol}")
        if expected is None or sha256(provider) != expected:
            raise AssertionError(f"hash-pinned 2025 provider mismatch: {symbol}")
    bars_2025 = load_2025_bars(contract, symbols)
    ledgers = build_checkpoint_ledgers(
        contract,
        opportunities,
        scoring,
        runs,
        examples,
        classes,
        execution,
        bars_2025,
        run_id,
    )
    comparators = build_static_comparators(ledgers["checkpoints"], opportunities, v2_baselines)
    model_metrics = build_model_comparison_metrics(comparators)
    general_accounting, general_veto_metrics, delayed = build_veto_accounting(
        ledgers["checkpoints"], opportunities, track="track_b_prior_only"
    )
    named_opportunities = opportunities.loc[
        opportunities["population_role"].eq("named_target")
    ].copy()
    named_accounting, named_veto_metrics, _ = build_veto_accounting(
        ledgers["checkpoints"], named_opportunities, track="track_a_named_family"
    )
    accounting = pd.concat([general_accounting, named_accounting], ignore_index=True)
    veto_metrics = pd.concat([general_veto_metrics, named_veto_metrics], ignore_index=True)
    competitor_census = build_competitor_census(
        ledgers["anchor_sets"], ledgers["eliminations"], opportunities
    )
    concentration = build_concentration(comparators, general_accounting)
    stress_results = build_stress_results(comparators, ledgers["outcomes"], general_accounting)
    named_results, general_results = build_named_and_general_results(
        comparators, ledgers["checkpoints"]
    )
    rebuilt_stress, loo_results, episode_attribution = run_rebuilt_sensitivities(
        paths=paths,
        base_pairs=comparators,
        scoring=scoring,
        runs=runs,
        examples=examples,
        primary_classes=classes,
        accounting=general_accounting,
        bars_2025=bars_2025,
        run_id=run_id,
    )
    rebuilt_labels = set(rebuilt_stress["stress_test"].astype(str))
    rebuilt_labels.add("leave_one_stock_out")
    loo_summary = pd.DataFrame(
        [
            {
                "stress_test": "leave_one_stock_out",
                "paired_rows": int(loo_results["paired_rows"].min()),
                "brier_improvement": float(loo_results["brier_improvement"].mean()),
                "paired_economic_increment_bps": float(
                    loo_results["paired_economic_increment_bps"].mean()
                ),
                "rebuild_detail_json": json.dumps(
                    {
                        "stocks": int(loo_results["excluded_stock"].nunique()),
                        "all_stock_dependent_inputs_rebuilt": bool(
                            loo_results["all_stock_dependent_inputs_rebuilt"].all()
                        ),
                        "directionally_positive_brier_fraction": float(
                            loo_results["brier_improvement"].gt(0.0).mean()
                        ),
                        "directionally_positive_economic_fraction": float(
                            loo_results["paired_economic_increment_bps"].gt(0.0).mean()
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        ]
    )
    stress_results = pd.concat(
        [
            stress_results.loc[~stress_results["stress_test"].astype(str).isin(rebuilt_labels)],
            rebuilt_stress,
            loo_summary,
        ],
        ignore_index=True,
        sort=False,
    )
    decision = scientific_decision(model_metrics, general_veto_metrics, comparators)

    output.mkdir(parents=True, exist_ok=True)
    detailed_tables = {
        "training_only_loop_payoff_classifications.parquet": classes,
        "regime_clock_loop_census.parquet": ledgers["census"],
        "pairwise_target_competitor_census.parquet": competitor_census,
        "anchor_compatible_loop_sets.parquet": ledgers["anchor_sets"],
        "sequential_checkpoint_posterior_ledger.parquet": ledgers["posterior"],
        "loop_elimination_events.parquet": ledgers["eliminations"],
        "good_bad_unknown_mass_timeline.parquet": ledgers["checkpoints"],
        "static_anchor_comparator_predictions.parquet": comparators,
        "sequential_rejection_decisions.parquet": accounting,
        "sequential_delayed_admission_decisions.parquet": delayed,
        "constant_terminal_remaining_payoff_outcomes.parquet": ledgers["outcomes"][
            [
                column
                for column in ledgers["outcomes"].columns
                if not column.startswith("restarted_")
            ]
        ],
        "restarted_horizon_sensitivity_outcomes.parquet": ledgers["outcomes"][
            [
                column
                for column in ledgers["outcomes"].columns
                if column.startswith("restarted_")
                or column
                in {
                    "run_id",
                    "outcome_id",
                    "checkpoint_id",
                    "opportunity_id",
                    "event_lineage_id",
                    "period",
                    "session_date",
                    "stock",
                    "checkpoint_type",
                    "checkpoint_timestamp",
                }
            ]
        ],
        "veto_accounting.parquet": accounting,
        "capturable_payoff_analysis.parquet": comparators[
            [
                "opportunity_id",
                "event_lineage_id",
                "period",
                "session_date",
                "stock",
                "target_loop",
                "orientation",
                "checkpoint_id",
                "checkpoint_timestamp",
                "bars_consumed",
                "bars_remaining",
                "resolution_lag_ratio",
                "target_remaining_net_bps",
                "original_net_payoff_bps",
                "capturable_payoff_fraction",
            ]
        ],
        "hindsight_episode_attribution.parquet": episode_attribution,
    }
    summary_tables = {
        "model_comparison_metrics.csv": model_metrics,
        "veto_accounting_summary.csv": veto_metrics,
        "stress_test_results.csv": stress_results,
        "leave_one_stock_out_results.csv": loo_results,
        "concentration_results.csv": concentration,
        "named_family_mechanism_results.csv": named_results,
        "general_prior_only_classification_results.csv": general_results,
    }
    for filename, frame in detailed_tables.items():
        annotated = _annotate_table(
            frame,
            run_id=run_id,
            contract_hash=contract_hash,
            data_snapshot_id=data_snapshot_id,
        )
        _write_parquet(output / filename, annotated)
    for filename, frame in summary_tables.items():
        annotated = _annotate_table(
            frame,
            run_id=run_id,
            contract_hash=contract_hash,
            data_snapshot_id=data_snapshot_id,
        )
        _write_csv(output / filename, annotated)
    write_json(
        output / "prospective_immutable_forecast_ledger_schema.json", build_prospective_schema()
    )
    plot_paths = make_plots(output, ledgers["checkpoints"], comparators, general_veto_metrics)
    metadata = {
        "run_id": run_id,
        "git_sha": git_sha,
        "repository_branch": branch,
        "contract_id": contract["contract_id"],
        "contract_hash": contract_hash,
        "data_snapshot_id": data_snapshot_id,
        "source_hashes": source_hashes,
        "model_version": MODEL_VERSION,
        "fixed_horizon_bars": 24,
        "cost_bps_per_side": 5.0,
        "random_seed": int(contract["random_seed"]),
        "periods": [2023, 2025],
        "scientific_status": contract["scientific_status"],
        "scientific_decision": decision,
        "generated_at": RUN_TIMESTAMP,
        "command": (
            "PYTHONPATH=packages/stocker_research/src .venv/bin/python "
            "research/slrno-v2/20260714-regime-loop-handoff/work/"
            "run_sequential_loop_competitor_veto_v1.py --output <OUTPUT>"
        ),
        "decision_timestamp_convention": "anchor bar close; transition after first new-state bar close",
        "training_cutoff": "strictly earlier completed sessions within period",
        "provider_2023_limitation": contract["inputs"]["provider_2023_status"],
        "safety": contract["safety"],
        "artifact_names": sorted([*detailed_tables, *summary_tables]),
        "plot_names": sorted(path.name for path in plot_paths),
    }
    write_json(output / "run_metadata.json", metadata)
    if exact_rerun_of is not None:
        identity = verify_exact_rerun(output, exact_rerun_of)
        write_json(output / "exact_rerun_identity.json", identity)
        if not identity["byte_identical"]:
            raise AssertionError(f"exact rerun identity failed: {identity}")
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    write_report(
        report_path,
        metadata=metadata,
        opportunities=opportunities,
        comparators=comparators,
        model_metrics=model_metrics,
        veto_metrics=general_veto_metrics,
        competitor_census=competitor_census,
        named_results=named_results,
        stress_results=stress_results,
        concentration=concentration,
    )


def run_prospective(args: argparse.Namespace) -> None:
    if args.prospective_root is None or args.record_json is None:
        raise ValueError("prospective mode requires --prospective-root and --record-json")
    contract = json.loads(CONTRACT_PATH.read_text())
    ledger = ProspectiveCompetitorLedger(
        Path(args.prospective_root),
        opened_periods=set(contract["opened_data_status"]["opened_periods"]),
    )
    record = json.loads(Path(args.record_json).read_text())
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
            exact_rerun_of=(Path(args.exact_rerun_of) if args.exact_rerun_of is not None else None),
        )
    else:
        run_prospective(args)


if __name__ == "__main__":
    main()
