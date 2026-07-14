#!/usr/bin/env python3
# ruff: noqa: E402, E501
"""Research-only causal temporary loop-payoff state experiment V2.

This runner consumes the frozen V1 causal loop forecasts and execution ledger.
It cannot connect to a broker, place an order, mutate a position, or change the
frozen trade exit. Structural-loop forecasting and economic admission remain
separate outputs throughout the run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[4]
_PACKAGE_SOURCE = _REPO_BOOTSTRAP / "packages/stocker_research/src"
if str(_PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_SOURCE))

from stocker_research.dynamic_loop_edge_state.decision import (
    DecisionThresholds,
    classify_edge_state,
)
from stocker_research.dynamic_loop_edge_state.online_state import (
    BOCPDSettings,
    EdgeForecast,
    HierarchicalSettings,
    SupportEvidence,
)
from stocker_research.dynamic_loop_edge_state.session_payoff import (
    AggregationSettings,
    aggregate_session_payoffs,
)
from stocker_research.dynamic_loop_edge_state.walkforward import (
    WalkForwardSettings,
    apply_frozen_admission,
    run_causal_walk_forward,
)

WORK = Path(__file__).resolve().parent
REPO = _REPO_BOOTSTRAP
CONFIG_PATH = WORK / "contracts/20260714-dynamic-loop-edge-state-v2.json"
V1_RUNNER_PATH = WORK / "run_dynamic_loop_context_edge_v1.py"
DEFAULT_OUT = WORK / "artifacts/20260714-dynamic-loop-edge-state-v2/primary"
DEFAULT_REPORT = WORK / "reports/20260714-dynamic-loop-edge-state-v2.md"
LOOP_COLUMNS = tuple(f"loop_score_{index:02d}" for index in range(1, 21))
PRIMARY_MODELS = (
    "v1_60_session_selector",
    "ewma_short_memory",
    "payoff_only_change_point",
    "hierarchical_change_point",
)


def required_artifact_names() -> tuple[str, ...]:
    return (
        "session_payoff_panel.parquet",
        "causal_edge_state_forecasts.parquet",
        "trade_decisions.parquet",
        "model_comparison_metrics.csv",
        "calibration_results.csv",
        "change_point_diagnostics.csv",
        "hindsight_episode_diagnostics.parquet",
        "stress_test_results.csv",
        "run_metadata.json",
    )


def derive_execution_clock(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive conservative causal timestamps from the frozen five-minute clock."""

    required = {"start_timestamp", "entry_step", "horizon", "status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing execution-clock columns: {missing}")
    result = pd.DataFrame(index=frame.index)
    start = pd.to_datetime(frame["start_timestamp"], utc=True, errors="raise")
    horizon = pd.to_numeric(frame["horizon"], errors="raise").astype(int)
    entry_step = pd.to_numeric(frame["entry_step"], errors="raise").astype(int)
    result["decision_timestamp"] = start + pd.Timedelta(minutes=5)
    entry = start + pd.to_timedelta(5 * entry_step, unit="m")
    result["entry_timestamp"] = entry.where(frame["status"].eq("filled"), pd.NaT)
    # Provider timestamps mark bar starts; an anchor+horizon close is one bar later.
    result["exit_timestamp"] = start + pd.to_timedelta(5 * (horizon + 1), unit="m")
    result["settlement_timestamp"] = result["exit_timestamp"]
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n")


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    safety = config["safety"]
    if not (
        safety["research_only"] is True
        and safety["live_ordering_enabled"] is False
        and safety["order_placement"] == "disabled"
        and safety["broker_connection_enabled"] is False
        and safety["paper_or_demo_execution_enabled"] is False
        and safety["deployment_enabled"] is False
        and safety["application_position_or_exit_logic_changed"] is False
    ):
        raise AssertionError("research-only safety boundary drift")
    if config["registered_target"]["fixed_horizon_bars"] != 24:
        raise AssertionError("registered horizon drift")
    if config["stress_tests"]["unbounded_parameter_search_allowed"] is not False:
        raise AssertionError("unbounded search enabled")
    return config


def verify_recovered_v1_inputs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Verify the sealed V1-derived recovery surface before any V2 scoring.

    The original V1 anchor panels and 2023 provider tape were ephemeral
    ``/private/tmp`` inputs.  The sealed V1 output and a prior scoring artifact
    retain the exact top-loop probabilities plus the full 250-session causal
    anchor context.  This adapter refuses to run unless every registered file
    hash and the unchanged V1 runner/contract hashes match.
    """

    specifications = config["source"]["recovered_inputs"]
    paths: dict[str, Path] = {}
    verified: dict[str, str] = {}
    missing: list[str] = []
    for name, specification in specifications.items():
        path = Path(str(specification["path"]))
        paths[str(name)] = path
        if not path.exists():
            missing.append(f"{name}={path}")
            continue
        actual = sha256(path)
        expected = str(specification["sha256"])
        if actual != expected:
            raise AssertionError(
                f"recovered V1 input hash drift for {name}: {actual} != {expected}"
            )
        verified[f"recovered_{name}"] = actual
    if missing:
        raise FileNotFoundError("missing hash-pinned V1 recovery inputs: " + ", ".join(missing))

    original_manifest = json.loads(paths["v1_source_hashes"].read_text())
    if not (
        original_manifest["research_only"] is True
        and original_manifest["live_ordering_enabled"] is False
        and original_manifest["order_placement"] == "disabled"
        and original_manifest["created_before_scoring"] is True
    ):
        raise AssertionError("sealed V1 safety/provenance manifest drift")
    original_hashes = {str(key): str(value) for key, value in original_manifest["sha256"].items()}
    if sha256(V1_RUNNER_PATH) != original_hashes["runner"]:
        raise AssertionError("V1 runner is not the hash-pinned frozen baseline")
    v1_contract_path = WORK / "contracts/20260713-dynamic-loop-context-edge-v1.json"
    if sha256(v1_contract_path) != original_hashes["contract"]:
        raise AssertionError("V1 contract is not the hash-pinned frozen baseline")
    verified.update({f"v1_original_{key}": value for key, value in original_hashes.items()})
    return paths, {
        "contract_id": "20260714-dynamic-loop-edge-state-v2-recovered-inputs",
        "created_before_scoring": True,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "input_adapter": config["source"]["input_adapter"],
        "sha256": verified,
    }


def _reconstruct_anchor_context(path: Path, period: int) -> pd.DataFrame:
    """Reconstruct the V1 causal anchor fields from a sealed derived artifact."""

    columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "cycle_index",
        "cycle_id",
        "cycle",
        "state",
        "history_token",
        "loop_probability",
        "mean_abs_return_12",
        "session_return",
        "bar_range_pct",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if frame.duplicated(["anchor_id", "cycle_index"]).any():
        raise AssertionError(f"duplicate recovered anchor/cycle score in {period}")
    cycle_map = (
        frame.loc[:, ["cycle_index", "cycle_id", "cycle"]]
        .drop_duplicates()
        .sort_values("cycle_index", kind="stable")
    )
    if cycle_map["cycle_index"].duplicated().any() or len(cycle_map) != 20:
        raise AssertionError(f"recovered loop catalogue drift in {period}")
    expected_indices = np.arange(20, dtype=int)
    if not np.array_equal(cycle_map["cycle_index"].to_numpy(int), expected_indices):
        raise AssertionError(f"recovered loop ordering drift in {period}")

    scores = frame.pivot(index="anchor_id", columns="cycle_index", values="loop_probability")
    scores = scores.reindex(columns=expected_indices, fill_value=0.0).fillna(0.0)
    values = scores.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise AssertionError(f"invalid recovered loop probabilities in {period}")
    top_index = np.argmax(values, axis=1)
    ordered = np.sort(values, axis=1)
    total = values.sum(axis=1)
    probabilities = np.divide(
        values,
        total[:, None],
        out=np.full_like(values, 1.0 / values.shape[1]),
        where=total[:, None] > 0.0,
    )
    log_probabilities = np.log(np.where(probabilities > 0.0, probabilities, 1.0))
    entropy = -np.sum(probabilities * log_probabilities, axis=1) / math.log(values.shape[1])

    base = (
        frame.drop(columns=["cycle_index", "cycle_id", "cycle", "loop_probability"])
        .drop_duplicates("anchor_id")
        .set_index("anchor_id")
        .reindex(scores.index)
    )
    if base.isna().all(axis=1).any():
        raise AssertionError(f"recovered anchor context join failure in {period}")
    cycle_ids = cycle_map.set_index("cycle_index")["cycle_id"].to_numpy(str)
    cycles = cycle_map.set_index("cycle_index")["cycle"].to_numpy(str)
    base["period"] = period
    base["top_loop_index"] = top_index + 1
    base["top_loop"] = cycle_ids[top_index]
    base["top_loop_cycle"] = cycles[top_index]
    base["top_loop_probability"] = values[np.arange(len(values)), top_index]
    base["top_loop_score"] = base["top_loop_probability"]
    base["top_second_margin"] = ordered[:, -1] - ordered[:, -2]
    base["compatibility_mass"] = total
    base["loop_score_entropy"] = entropy
    token = base["history_token"].to_numpy(int)
    decoded_state = token % 8
    decoded_previous = (token % 72) // 8
    if not np.array_equal(decoded_state, base["state"].to_numpy(int)):
        raise AssertionError(f"recovered history-token state mismatch in {period}")
    base["previous_state_1"] = decoded_previous
    base["start_timestamp"] = pd.to_datetime(base["start_timestamp"], utc=True, errors="raise")
    base["session_date"] = base["session_date"].astype(str)
    return base.reset_index()


def load_recovered_v1_analysis(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[int, list[str]], pd.DataFrame, dict[str, Any]]:
    """Load a hash-verified 250-session surface and the unchanged V1 states."""

    paths, pre_score = verify_recovered_v1_inputs(config)
    contexts = pd.concat(
        [
            _reconstruct_anchor_context(paths[f"loop_scoring_{period}"], period)
            for period in config["evaluation"]["periods"]
        ],
        ignore_index=True,
    )
    sessions_by_period = {
        int(period): sorted(
            contexts.loc[contexts["period"].eq(int(period)), "session_date"].unique()
        )
        for period in config["evaluation"]["periods"]
    }
    for period, sessions in sessions_by_period.items():
        if len(sessions) != 250:
            raise AssertionError(
                f"expected 250 recovered sessions in {period}, got {len(sessions)}"
            )

    horizon = int(config["registered_target"]["fixed_horizon_bars"])
    ledger = pd.read_parquet(paths["accepted_signal_ledger"])
    ledger = ledger.loc[
        ledger["period"]
        .astype(str)
        .isin([str(period) for period in config["evaluation"]["periods"]])
        & ledger["strategy"].eq(config["source"]["source_strategy"])
        & ledger["horizon"].eq(horizon)
    ].copy()
    ledger["period"] = ledger["period"].astype(int)
    ledger["start_timestamp"] = pd.to_datetime(ledger["start_timestamp"], utc=True, errors="raise")
    if ledger.duplicated(["period", "horizon", "anchor_id"]).any():
        raise AssertionError("duplicate recovered frozen signal")
    ledger = ledger.merge(
        contexts,
        on=["period", "anchor_id", "symbol_norm", "session_date", "start_timestamp"],
        how="left",
        validate="one_to_one",
    )
    if ledger["top_loop"].isna().any():
        raise AssertionError("recovered signal-to-anchor context join failure")
    ledger["session_index"] = ledger.apply(
        lambda row: sessions_by_period[int(row["period"])].index(str(row["session_date"])),
        axis=1,
    ).astype(int)
    ledger["net_return_bps"] = np.where(
        ledger["status"].eq("filled"),
        ledger["gross_return_bps"].to_numpy(float)
        - float(config["session_aggregation"]["entry_cost_bps"])
        - float(config["session_aggregation"]["exit_cost_bps"]),
        np.nan,
    )

    sealed = pd.read_parquet(
        paths["v1_scored_signal_ledger"],
        columns=[
            "period",
            "horizon",
            "anchor_id",
            "top_loop",
            "top_loop_probability",
            "state",
            "history_token",
            "volume_ratio",
            "volume_bucket",
        ],
    )
    sealed["period"] = sealed["period"].astype(int)
    sealed = sealed.loc[sealed["horizon"].eq(horizon)].copy()
    if sealed.duplicated(["period", "horizon", "anchor_id"]).any():
        raise AssertionError("duplicate sealed V1 score")
    expected = ledger.loc[
        ledger["session_index"].ge(int(config["support"]["warmup_completed_sessions"])),
        [
            "period",
            "horizon",
            "anchor_id",
            "top_loop",
            "top_loop_probability",
            "state",
            "history_token",
        ],
    ]
    comparison = expected.merge(
        sealed,
        on=["period", "horizon", "anchor_id"],
        how="outer",
        suffixes=("_recovered", "_sealed"),
        indicator=True,
        validate="one_to_one",
    )
    if not comparison["_merge"].eq("both").all():
        raise AssertionError("recovered evaluation surface differs from sealed V1 row set")
    if not (
        comparison["top_loop_recovered"].eq(comparison["top_loop_sealed"]).all()
        and np.array_equal(
            comparison["top_loop_probability_recovered"].to_numpy(float),
            comparison["top_loop_probability_sealed"].to_numpy(float),
        )
        and comparison["state_recovered"].eq(comparison["state_sealed"]).all()
        and comparison["history_token_recovered"].eq(comparison["history_token_sealed"]).all()
    ):
        raise AssertionError("recovered loop/orientation scores are not exactly V1-equivalent")

    volume = sealed.loc[:, ["period", "horizon", "anchor_id", "volume_ratio", "volume_bucket"]]
    ledger = ledger.merge(
        volume,
        on=["period", "horizon", "anchor_id"],
        how="left",
        validate="one_to_one",
    )
    ledger["volume_bucket"] = ledger["volume_bucket"].fillna("unavailable_warmup")
    primary_states = pd.read_parquet(paths["v1_primary_cell_states"])
    pre_score["recovery_equivalence"] = {
        "sealed_rows_checked": int(len(comparison)),
        "top_loop_exact": True,
        "top_probability_exact": True,
        "state_and_history_token_exact": True,
    }
    return ledger, sessions_by_period, primary_states, pre_score


def _load_v1_module() -> Any:
    spec = importlib.util.spec_from_file_location("dynamic_loop_context_edge_v1", V1_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen V1 runner: {V1_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _session_open(session: str) -> pd.Timestamp:
    return pd.Timestamp(f"{session} 09:30", tz="America/New_York").tz_convert("UTC")


def _session_close(session: str) -> pd.Timestamp:
    return pd.Timestamp(f"{session} 16:00", tz="America/New_York").tz_convert("UTC")


def build_session_calendars(
    sessions_by_period: Mapping[int, Sequence[str]],
) -> dict[int, pd.DataFrame]:
    calendars: dict[int, pd.DataFrame] = {}
    for period, sessions in sessions_by_period.items():
        calendars[int(period)] = pd.DataFrame(
            {
                "period": int(period),
                "score_session": list(sessions),
                "session_index": np.arange(len(sessions), dtype=int),
                "decision_timestamp": [_session_open(session) for session in sessions],
                "session_close_timestamp": [_session_close(session) for session in sessions],
            }
        )
    return calendars


def _load_confidence_rows(
    contract: Mapping[str, Any],
    period: int,
) -> pd.DataFrame:
    path = Path(contract["inputs"]["anchor_panels"][str(period)])
    frame = pd.read_parquet(
        path,
        columns=["anchor_id", *LOOP_COLUMNS],
    )
    scores = frame.loc[:, LOOP_COLUMNS].to_numpy(float)
    if not np.isfinite(scores).all() or (scores < 0.0).any():
        raise AssertionError(f"invalid causal loop scores in {period}")
    ordered = np.sort(scores, axis=1)
    total = scores.sum(axis=1)
    probabilities = np.divide(
        scores,
        total[:, None],
        out=np.full_like(scores, 1.0 / scores.shape[1]),
        where=total[:, None] > 0.0,
    )
    entropy = -np.sum(
        np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0),
        axis=1,
    ) / math.log(scores.shape[1])
    return pd.DataFrame(
        {
            "period": period,
            "anchor_id": frame["anchor_id"].to_numpy(int),
            "top_loop_score": ordered[:, -1],
            "top_second_margin": ordered[:, -1] - ordered[:, -2],
            "compatibility_mass": total,
            "loop_score_entropy": entropy,
        }
    )


def _provider_volume_lookup(
    contract: Mapping[str, Any],
    period: int,
    symbols: Sequence[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    root = Path(contract["inputs"]["provider_roots"][str(period)])
    for symbol in symbols:
        path = root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        frame = pd.read_parquet(path, columns=["timestamp", "volume"])
        frame["start_timestamp"] = pd.to_datetime(frame.pop("timestamp"), utc=True)
        frame["symbol_norm"] = symbol
        local = frame["start_timestamp"].dt.tz_convert("America/New_York")
        frame = frame.loc[local.dt.year.eq(period)].copy()
        rows.append(frame.loc[:, ["symbol_norm", "start_timestamp", "volume"]])
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["symbol_norm", "start_timestamp"]).any():
        raise AssertionError("duplicate provider volume timestamp")
    return result


def _causal_transition_surprise(surface: pd.DataFrame) -> pd.Series:
    result = pd.Series(index=surface.index, dtype=float)
    state_values = sorted(
        set(surface["state"].astype(int)) | set(surface["previous_state_1"].astype(int))
    )
    state_to_index = {state: index for index, state in enumerate(state_values)}
    for _, period_frame in surface.groupby("period", sort=True):
        counts = np.ones((len(state_values), len(state_values)), dtype=float)
        for _, session_frame in period_frame.groupby("session_date", sort=True):
            previous = session_frame["previous_state_1"].astype(int).map(state_to_index).to_numpy()
            current = session_frame["state"].astype(int).map(state_to_index).to_numpy()
            probabilities = counts[previous, current] / counts[previous].sum(axis=1)
            result.loc[session_frame.index] = -np.log(np.maximum(probabilities, 1e-12))
            for previous_index, current_index in zip(previous, current, strict=True):
                counts[previous_index, current_index] += 1.0
    return result.astype(float)


def build_trade_surface(
    ledger: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Adapt the hash-verified V1 ledger without a parallel trade schema."""

    horizon = int(config["registered_target"]["fixed_horizon_bars"])
    surface = ledger.loc[ledger["horizon"].eq(horizon)].copy()
    surface["period"] = surface["period"].astype(int)
    surface["session"] = surface["session_date"].astype(str)
    surface["loop_id"] = surface["top_loop"].astype(str)
    surface["orientation"] = "state_" + surface["state"].astype(int).astype(str)
    symbols = sorted(surface["symbol_norm"].astype(str).unique())
    # Raw volume was not retained in the sealed V1 score ledger.  The causal
    # historical-volume ratio was retained for the 190 scored sessions, so
    # liquidity stresses use price × volume-ratio as an explicitly labelled
    # activity proxy rather than fabricating raw dollar volume.
    surface["dollar_volume_proxy"] = surface["anchor_close"].to_numpy(float) * surface[
        "volume_ratio"
    ].to_numpy(float)
    surface["liquidity_proxy_status"] = np.where(
        np.isfinite(surface["volume_ratio"].to_numpy(float)),
        "anchor_price_times_causal_volume_ratio",
        "unavailable_in_warmup",
    )
    surface["liquidity_pressure"] = np.where(
        np.isfinite(surface["volume_ratio"].to_numpy(float)),
        1.0 / (1.0 + np.maximum(surface["volume_ratio"].to_numpy(float), 0.0)),
        np.nan,
    )
    surface["transition_surprise"] = _causal_transition_surprise(surface)
    cell_keys = ["period", "session", "loop_id", "orientation", "horizon"]
    breadth = (
        surface.groupby(cell_keys, observed=True)["symbol_norm"]
        .nunique()
        .div(len(symbols))
        .rename("structural_breadth")
        .reset_index()
    )
    market = (
        surface.groupby(["period", "session"], observed=True)
        .agg(
            market_return=("session_return", "mean"),
            market_volatility=("mean_abs_return_12", "mean"),
        )
        .reset_index()
    )
    surface = surface.merge(breadth, on=cell_keys, how="left", validate="many_to_one")
    surface = surface.merge(
        market,
        on=["period", "session"],
        how="left",
        validate="many_to_one",
    )
    clock = derive_execution_clock(surface)
    for column in clock:
        surface[column] = clock[column]
    surface["feature_availability_timestamp"] = surface["decision_timestamp"]
    surface["stock_id"] = surface["symbol_norm"].astype(str)
    surface["fill_id"] = (
        surface["period"].astype(str)
        + "-"
        + surface["anchor_id"].astype(str)
        + "-h"
        + surface["horizon"].astype(str)
    )
    surface["gross_payoff_bps"] = surface["gross_return_bps"].astype(float)
    cost_config = config["session_aggregation"]
    surface["entry_cost_bps"] = float(cost_config["entry_cost_bps"])
    surface["exit_cost_bps"] = float(cost_config["exit_cost_bps"])
    # The frozen source has no quote/component data. Zero means unavailable, not estimated zero.
    for column in (
        "spread_cost_bps",
        "slippage_cost_bps",
        "commission_cost_bps",
        "financing_cost_bps",
        "fx_cost_bps",
        "other_cost_bps",
    ):
        surface[column] = 0.0
    surface["cost_data_status"] = "aggregate_5bps_per_side_no_component_quotes"
    surface["net_payoff_bps"] = np.where(
        surface["status"].eq("filled"),
        surface["gross_payoff_bps"].to_numpy(float)
        - float(cost_config["entry_cost_bps"])
        - float(cost_config["exit_cost_bps"]),
        np.nan,
    )
    return surface.sort_values(
        ["period", "session", "symbol_norm", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)


def aggregate_payoff_panels(
    surface: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregation = config["session_aggregation"]
    filled = surface.loc[surface["status"].eq("filled")].copy()
    primary = aggregate_session_payoffs(
        filled,
        AggregationSettings(
            method=str(aggregation["primary_method"]),
            winsor_fraction_each_tail=float(aggregation["winsor_fraction_each_tail"]),
            stock_contribution_cap_bps=float(aggregation["stock_contribution_cap_bps"]),
        ),
    )
    alternative = aggregate_session_payoffs(
        filled,
        AggregationSettings(
            method=str(aggregation["alternative_method"]),
            winsor_fraction_each_tail=float(aggregation["winsor_fraction_each_tail"]),
            stock_contribution_cap_bps=float(aggregation["stock_contribution_cap_bps"]),
        ),
    )
    for panel in (primary, alternative):
        panel.insert(0, "period", pd.to_datetime(panel["session"]).dt.year.astype(int))
        panel["source_data_id"] = panel["source_fill_ids"].map(
            lambda value: hashlib.sha256(str(value).encode()).hexdigest()[:16]
        )
    return primary, alternative


def build_feature_panel(
    surface: pd.DataFrame,
    payoff_panel: pd.DataFrame,
    calendars: Mapping[int, pd.DataFrame],
    cell_keys_by_period: Mapping[int, Sequence[tuple[str, str, int]]],
    required_features: tuple[str, ...],
) -> pd.DataFrame:
    """Create one-session-lagged compact features with explicit availability."""

    keys = ["period", "session", "loop_id", "orientation", "horizon"]
    structural = (
        surface.groupby(keys, observed=True)
        .agg(
            structural_breadth=("structural_breadth", "mean"),
            top_loop_score=("top_loop_score", "mean"),
            top_second_margin=("top_second_margin", "mean"),
            loop_score_entropy=("loop_score_entropy", "mean"),
            transition_surprise=("transition_surprise", "mean"),
            market_return=("market_return", "mean"),
            market_volatility=("market_volatility", "mean"),
            liquidity_pressure=("liquidity_pressure", "mean"),
            structural_availability_timestamp=("decision_timestamp", "max"),
        )
        .reset_index()
    )
    payoff = payoff_panel.loc[
        :,
        [
            *keys,
            "positive_stock_fraction",
            "cross_stock_payoff_dispersion_bps",
            "cost_contribution_bps",
            "robust_gross_payoff_bps",
            "data_availability_timestamp",
        ],
    ].rename(
        columns={
            "cross_stock_payoff_dispersion_bps": "payoff_dispersion",
            "data_availability_timestamp": "payoff_availability_timestamp",
        }
    )
    events = structural.merge(payoff, on=keys, how="inner", validate="one_to_one")
    events["cost_pressure"] = events["cost_contribution_bps"] / (
        events["robust_gross_payoff_bps"].abs() + 1.0
    )
    events = events.sort_values(
        ["period", "loop_id", "orientation", "horizon", "session"], kind="stable"
    )
    group = events.groupby(["period", "loop_id", "orientation", "horizon"], observed=True)
    events["breadth_change"] = group["structural_breadth"].diff().fillna(0.0)
    events["dispersion_change"] = group["payoff_dispersion"].diff().fillna(0.0)
    events["feature_availability_timestamp"] = events[
        ["structural_availability_timestamp", "payoff_availability_timestamp"]
    ].max(axis=1)
    rows: list[dict[str, object]] = []
    for period, calendar in calendars.items():
        period_events = events.loc[events["period"].eq(period)].copy()
        for cell_key in cell_keys_by_period[period]:
            history = period_events.loc[
                period_events["loop_id"].eq(cell_key[0])
                & period_events["orientation"].eq(cell_key[1])
                & period_events["horizon"].eq(cell_key[2])
            ].sort_values("session", kind="stable")
            for calendar_row in calendar.itertuples(index=False):
                eligible = history.loc[
                    history["session"].lt(str(calendar_row.score_session))
                    & history["feature_availability_timestamp"].lt(
                        pd.Timestamp(calendar_row.decision_timestamp)
                    )
                ]
                row: dict[str, object] = {
                    "period": period,
                    "score_session": str(calendar_row.score_session),
                    "loop_id": cell_key[0],
                    "orientation": cell_key[1],
                    "horizon": cell_key[2],
                    "feature_source_session": None,
                    "feature_availability_timestamp": pd.NaT,
                }
                if not eligible.empty:
                    source = eligible.iloc[-1]
                    row["feature_source_session"] = str(source["session"])
                    row["feature_availability_timestamp"] = source["feature_availability_timestamp"]
                    for name in required_features:
                        row[name] = float(source[name])
                else:
                    for name in required_features:
                        row[name] = math.nan
                rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["period", "score_session", "loop_id", "orientation"], kind="stable")
        .reset_index(drop=True)
    )


def bocpd_settings_from_config(
    config: Mapping[str, Any],
    *,
    hazard: float | None = None,
) -> BOCPDSettings:
    observation = config["observation_model"]
    selected_hazard = (
        float(config["change_point"]["primary_hazard_probability_per_observed_session"])
        if hazard is None
        else float(hazard)
    )
    return BOCPDSettings(
        hazard_probability=selected_hazard,
        degrees_of_freedom_floor=float(observation["student_t_degrees_of_freedom_floor"]),
        prior_mean_net_bps=float(observation["prior_mean_net_bps"]),
        prior_mean_variance_bps2=float(observation["prior_mean_variance_bps2"]),
        scale_prior_bps=float(observation["scale_prior_bps"]),
        outlier_clip_predictive_scales=float(observation["outlier_clip_predictive_scales"]),
        max_run_length_sessions=int(observation["max_run_length_sessions"]),
        minimum_likelihood=float(observation["minimum_likelihood"]),
    )


def hierarchy_settings_from_config(
    config: Mapping[str, Any],
    *,
    payoff_only: bool,
) -> HierarchicalSettings:
    hierarchy = config["hierarchy"]
    return HierarchicalSettings(
        pooling_strength_sessions=(
            0.0 if payoff_only else float(hierarchy["pooling_strength_sessions"])
        ),
        minimum_shared_cells_per_session=int(hierarchy["minimum_shared_cells_per_session"]),
        sparse_uncertainty_inflation_bps=(
            0.0 if payoff_only else float(hierarchy["sparse_uncertainty_inflation_bps"])
        ),
        lower_bound_confidence=float(config["thresholds"]["lower_bound_confidence"]),
        feature_logit_weights=(
            {} if payoff_only else dict(config["features"]["leading_feature_logit_weights"])
        ),
    )


def decision_thresholds_from_config(
    config: Mapping[str, Any],
    *,
    active_probability: float | None = None,
    survival_probability: float | None = None,
) -> DecisionThresholds:
    support = config["support"]
    threshold = config["thresholds"]
    return DecisionThresholds(
        minimum_independent_sessions=int(support["minimum_independent_sessions"]),
        minimum_independent_stocks=int(support["minimum_independent_stocks"]),
        minimum_effective_sample_size=float(support["minimum_effective_sample_size"]),
        maximum_posterior_std_net_bps=float(support["maximum_posterior_std_net_bps"]),
        active_probability=(
            float(threshold["active_probability"])
            if active_probability is None
            else float(active_probability)
        ),
        survival_probability=(
            float(threshold["survival_probability"])
            if survival_probability is None
            else float(survival_probability)
        ),
        decaying_termination_probability=float(threshold["decaying_termination_probability"]),
        retired_positive_probability=float(threshold["retired_positive_probability"]),
        change_reset_probability=float(config["change_point"]["change_reset_probability"]),
        out_of_distribution_threshold=float(threshold["out_of_distribution_score"]),
    )


def run_change_point_model(
    *,
    model_name: str,
    config: Mapping[str, Any],
    configuration_hash: str,
    run_id: str,
    calendars: Mapping[int, pd.DataFrame],
    payoff_panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    cell_keys_by_period: Mapping[int, Sequence[tuple[str, str, int]]],
    payoff_only: bool,
    hazard: float | None = None,
    active_probability: float | None = None,
    survival_probability: float | None = None,
) -> pd.DataFrame:
    feature_weights = config["features"]["leading_feature_logit_weights"]
    required_features = (
        ()
        if payoff_only
        else tuple(name for name in feature_weights if name != "out_of_distribution_score")
    )
    rows: list[pd.DataFrame] = []
    for period in config["evaluation"]["periods"]:
        period = int(period)
        period_features = feature_panel.loc[feature_panel["period"].eq(period)].copy()
        period_panel = payoff_panel.loc[payoff_panel["period"].eq(period)].copy()
        forecast = run_causal_walk_forward(
            session_calendar=calendars[period],
            payoff_panel=period_panel,
            feature_panel=period_features,
            cell_keys=list(cell_keys_by_period[period]),
            bocpd_settings=bocpd_settings_from_config(config, hazard=hazard),
            hierarchy_settings=hierarchy_settings_from_config(config, payoff_only=payoff_only),
            decision_thresholds=decision_thresholds_from_config(
                config,
                active_probability=active_probability,
                survival_probability=survival_probability,
            ),
            settings=WalkForwardSettings(
                run_id=run_id,
                model_name=model_name,
                model_version=str(config["model_version"]),
                configuration_hash=configuration_hash,
                feature_schema_version=str(config["source"]["feature_schema_version"]),
                cost_model_version=str(config["source"]["cost_model_version"]),
                horizon_bars=int(config["registered_target"]["fixed_horizon_bars"]),
                session_bars=int(config["registered_target"]["session_bars"]),
                required_features=required_features,
                include_leading_features=not payoff_only,
                random_seed=int(config["evaluation"]["fixed_random_seed"]),
            ),
        )
        forecast.insert(0, "period", period)
        rows.append(forecast)
    combined = pd.concat(rows, ignore_index=True)
    warmup = int(config["support"]["warmup_completed_sessions"])
    calendar_index = pd.concat(calendars.values(), ignore_index=True).loc[
        :, ["period", "score_session", "session_index"]
    ]
    combined = combined.merge(
        calendar_index,
        on=["period", "score_session"],
        how="left",
        validate="many_to_one",
    )
    return combined.loc[combined["session_index"].ge(warmup)].reset_index(drop=True)


def run_ewma_model(
    *,
    config: Mapping[str, Any],
    configuration_hash: str,
    run_id: str,
    calendars: Mapping[int, pd.DataFrame],
    payoff_panel: pd.DataFrame,
    cell_keys_by_period: Mapping[int, Sequence[tuple[str, str, int]]],
) -> pd.DataFrame:
    """Causal short-memory EWMA with uncertainty and the shared admission rules."""

    half_life = float(config["ewma"]["half_life_observed_sessions"])
    alpha = 1.0 - math.exp(math.log(0.5) / half_life)
    horizon = int(config["registered_target"]["fixed_horizon_bars"])
    session_bars = int(config["registered_target"]["session_bars"])
    thresholds = decision_thresholds_from_config(config)
    output: list[dict[str, object]] = []
    metadata = {
        "run_id": run_id,
        "model_name": "ewma_short_memory",
        "model_version": config["model_version"],
        "configuration_hash": configuration_hash,
        "feature_schema_version": config["source"]["feature_schema_version"],
        "cost_model_version": config["source"]["cost_model_version"],
        "fixed_horizon_bars": horizon,
        "random_seed": config["evaluation"]["fixed_random_seed"],
    }
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    for period in config["evaluation"]["periods"]:
        period = int(period)
        calendar = calendars[period]
        panel = payoff_panel.loc[payoff_panel["period"].eq(period)].copy()
        states: dict[tuple[str, str, int], dict[str, Any]] = {}
        previous_states = {key: "unknown" for key in cell_keys_by_period[period]}
        updated_sessions: set[str] = set()
        settled_count = 0
        latest_availability: pd.Timestamp | None = None
        latest_source: str | None = None
        for calendar_row in calendar.itertuples(index=False):
            score_session = str(calendar_row.score_session)
            decision_timestamp = pd.Timestamp(calendar_row.decision_timestamp)
            pending = sorted(
                session
                for session in panel.loc[panel["session"].lt(score_session), "session"].unique()
                if session not in updated_sessions
            )
            unresolved = False
            for source_session in pending:
                session_rows = panel.loc[panel["session"].eq(source_session)]
                if not session_rows["data_availability_timestamp"].lt(decision_timestamp).all():
                    unresolved = True
                    break
                for row in session_rows.itertuples(index=False):
                    key = (str(row.loop_id), str(row.orientation), int(row.horizon))
                    state = states.setdefault(
                        key,
                        {
                            "mean": 0.0,
                            "variance": float(config["observation_model"]["scale_prior_bps"]) ** 2,
                            "count": 0,
                            "observations": [],
                        },
                    )
                    value = float(row.robust_net_payoff_bps)
                    if state["count"] == 0:
                        state["mean"] = value
                    else:
                        difference = value - state["mean"]
                        state["mean"] += alpha * difference
                        state["variance"] = (1.0 - alpha) * (
                            state["variance"] + alpha * difference**2
                        )
                    state["count"] += 1
                    state["observations"].append(row)
                updated_sessions.add(source_session)
                settled_count += len(session_rows)
                latest_availability = session_rows["data_availability_timestamp"].max()
                latest_source = source_session
            for key in cell_keys_by_period[period]:
                state = states.get(key)
                if state is None:
                    support = SupportEvidence(0.0, 0, 0, 0.0)
                    mean = 0.0
                    std = float(config["observation_model"]["scale_prior_bps"])
                    p_positive = 0.5
                    run_length = 0.0
                else:
                    count = int(state["count"])
                    run_length = min(float(count), 1.0 / alpha)
                    window = min(count, max(1, int(math.ceil(run_length))))
                    recent = state["observations"][-window:]
                    stocks = {
                        stock for item in recent for stock in json.loads(item.independent_stock_ids)
                    }
                    support = SupportEvidence(
                        effective_sessions=run_length,
                        independent_stocks=len(stocks),
                        raw_fills=sum(int(item.raw_fill_count) for item in recent),
                        effective_sample_size=sum(
                            float(item.effective_sample_size) for item in recent
                        ),
                    )
                    mean = float(state["mean"])
                    std = math.sqrt(
                        max(
                            float(state["variance"]) / max(run_length, 1.0),
                            float(config["ewma"]["minimum_variance_bps2"]),
                        )
                    )
                    p_positive = float(
                        student_t.cdf(
                            mean / std,
                            df=float(
                                config["observation_model"]["student_t_degrees_of_freedom_floor"]
                            ),
                        )
                    )
                p_off = alpha * 0.5 + (1.0 - alpha) * (1.0 - p_positive)
                survival = (1.0 - p_off) ** (horizon / session_bars)
                quantile = student_t.ppf(
                    float(config["thresholds"]["lower_bound_confidence"]),
                    df=float(config["observation_model"]["student_t_degrees_of_freedom_floor"]),
                )
                forecast = EdgeForecast(
                    p_change_now=alpha,
                    posterior_run_length_mean=run_length,
                    posterior_run_length_mode=run_length,
                    posterior_mean_net_bps=mean,
                    posterior_std_net_bps=std,
                    posterior_lower_bound_net_bps=mean - quantile * std,
                    p_edge_positive=p_positive,
                    p_edge_active=p_positive,
                    p_on_next=alpha * 0.5 + (1.0 - alpha) * p_positive,
                    p_off_next=p_off,
                    p_survive_horizon=survival,
                    out_of_distribution_score=0.0,
                )
                decision = classify_edge_state(
                    forecast,
                    support,
                    thresholds,
                    previous_state=previous_states[key],
                    unresolved_outcomes=unresolved,
                )
                previous_states[key] = decision.edge_state
                output.append(
                    {
                        "period": period,
                        "loop_id": key[0],
                        "orientation": key[1],
                        "score_session": score_session,
                        "decision_timestamp": decision_timestamp,
                        "horizon": key[2],
                        **forecast.__dict__,
                        "effective_sessions": support.effective_sessions,
                        "independent_stocks": support.independent_stocks,
                        "raw_fills": support.raw_fills,
                        "effective_sample_size": support.effective_sample_size,
                        "edge_state": decision.edge_state,
                        "admit_new_entry": decision.admit_new_entry,
                        "reason_codes": "|".join(decision.reason_codes),
                        "existing_position_action": decision.existing_position_action,
                        "required_features_available": True,
                        "feature_max_availability_timestamp": pd.NaT,
                        "settled_observation_count": settled_count,
                        "training_latest_source_session": latest_source,
                        "training_latest_availability_timestamp": latest_availability,
                        "prediction_frozen_at": decision_timestamp,
                        "run_id": run_id,
                        "model_name": "ewma_short_memory",
                        "model_version": config["model_version"],
                        "configuration_hash": configuration_hash,
                        "feature_schema_version": config["source"]["feature_schema_version"],
                        "cost_model_version": config["source"]["cost_model_version"],
                        "run_metadata_json": metadata_json,
                        "session_index": int(calendar_row.session_index),
                    }
                )
    result = pd.DataFrame(output)
    warmup = int(config["support"]["warmup_completed_sessions"])
    return result.loc[result["session_index"].ge(warmup)].reset_index(drop=True)


def build_v1_forecasts(
    primary_states: pd.DataFrame,
    calendars: Mapping[int, pd.DataFrame],
    config: Mapping[str, Any],
    configuration_hash: str,
    run_id: str,
) -> pd.DataFrame:
    """Translate the unchanged V1 binary state ledger to the common score schema."""

    horizon = int(config["registered_target"]["fixed_horizon_bars"])
    frame = primary_states.loc[primary_states["horizon"].eq(horizon)].copy()
    frame["period"] = frame["period"].astype(int)
    frame["score_session"] = frame.pop("session_date").astype(str)
    frame["loop_id"] = frame["top_loop"].astype(str)
    frame["orientation"] = frame["cell_key"].astype(str).str.rsplit("|c", n=1).str[-1]
    frame["orientation"] = "state_" + frame["orientation"]
    calendar = pd.concat(calendars.values(), ignore_index=True).loc[
        :, ["period", "score_session", "decision_timestamp"]
    ]
    frame = frame.merge(
        calendar,
        on=["period", "score_session"],
        how="left",
        validate="many_to_one",
    )
    frame = frame.sort_values(["period", "cell_key", "score_session"], kind="stable").reset_index(
        drop=True
    )
    previous = frame.groupby(["period", "cell_key"], observed=True)["active"].shift()
    changed = previous.notna() & previous.ne(frame["active"])
    active = frame["active"].astype(bool)
    p_binary = np.where(active, 0.999, 0.001)
    prior_scale = float(config["observation_model"]["scale_prior_bps"])
    lower_quantile = student_t.ppf(
        float(config["thresholds"]["lower_bound_confidence"]),
        df=float(config["observation_model"]["student_t_degrees_of_freedom_floor"]),
    )
    metadata = {
        "run_id": run_id,
        "model_name": "v1_60_session_selector",
        "model_version": "dynamic_loop_context_edge_v1_frozen",
        "configuration_hash": configuration_hash,
        "feature_schema_version": "v1_loop_current_regime",
        "cost_model_version": config["source"]["cost_model_version"],
        "fixed_horizon_bars": horizon,
        "random_seed": 20260713,
    }
    frame["p_change_now"] = np.where(changed, 1.0, 0.0)
    frame["posterior_run_length_mean"] = frame["active_age_sessions"].astype(float)
    frame["posterior_run_length_mode"] = frame["active_age_sessions"].astype(float)
    frame["posterior_mean_net_bps"] = frame["estimate_net_bps"].astype(float)
    frame["posterior_std_net_bps"] = prior_scale
    frame["posterior_lower_bound_net_bps"] = (
        frame["estimate_net_bps"].astype(float) - lower_quantile * prior_scale
    )
    frame["p_edge_positive"] = p_binary
    frame["p_edge_active"] = p_binary
    frame["p_on_next"] = np.where(active, 0.999, 0.001)
    frame["p_off_next"] = np.where(active, 0.001, 0.999)
    frame["p_survive_horizon"] = np.where(active, 0.999, 0.001)
    frame["out_of_distribution_score"] = 0.0
    frame["effective_sessions"] = 60.0
    frame["independent_stocks"] = 0
    frame["raw_fills"] = frame["cell_support"].astype(int)
    frame["effective_sample_size"] = frame["cell_support"].astype(float)
    frame["edge_state"] = np.where(active, "active", "retired")
    frame["admit_new_entry"] = active
    frame["reason_codes"] = np.where(active, "v1_estimate_positive", "v1_inactive")
    frame["existing_position_action"] = "not_applicable"
    frame["required_features_available"] = True
    frame["feature_max_availability_timestamp"] = pd.NaT
    frame["settled_observation_count"] = frame["cell_support"].astype(int)
    frame["training_latest_source_session"] = None
    frame["training_latest_availability_timestamp"] = pd.NaT
    frame["prediction_frozen_at"] = frame["decision_timestamp"]
    frame["run_id"] = run_id
    frame["model_name"] = "v1_60_session_selector"
    frame["model_version"] = "dynamic_loop_context_edge_v1_frozen"
    frame["configuration_hash"] = configuration_hash
    frame["feature_schema_version"] = "v1_loop_current_regime"
    frame["cost_model_version"] = config["source"]["cost_model_version"]
    frame["run_metadata_json"] = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    columns = [
        "period",
        "loop_id",
        "orientation",
        "score_session",
        "decision_timestamp",
        "horizon",
        "p_edge_positive",
        "p_edge_active",
        "p_change_now",
        "p_on_next",
        "p_off_next",
        "p_survive_horizon",
        "posterior_mean_net_bps",
        "posterior_std_net_bps",
        "posterior_lower_bound_net_bps",
        "posterior_run_length_mean",
        "posterior_run_length_mode",
        "effective_sessions",
        "independent_stocks",
        "raw_fills",
        "effective_sample_size",
        "out_of_distribution_score",
        "edge_state",
        "admit_new_entry",
        "reason_codes",
        "existing_position_action",
        "required_features_available",
        "feature_max_availability_timestamp",
        "settled_observation_count",
        "training_latest_source_session",
        "training_latest_availability_timestamp",
        "prediction_frozen_at",
        "run_id",
        "model_name",
        "model_version",
        "configuration_hash",
        "feature_schema_version",
        "cost_model_version",
        "run_metadata_json",
        "session_index",
    ]
    return (
        frame.loc[:, columns]
        .sort_values(["period", "score_session", "loop_id", "orientation"], kind="stable")
        .reset_index(drop=True)
    )


def build_trade_decisions(
    surface: pd.DataFrame,
    forecasts: pd.DataFrame,
    calendars: Mapping[int, pd.DataFrame],
    config: Mapping[str, Any],
    configuration_hash: str,
    run_id: str,
) -> pd.DataFrame:
    warmup = int(config["support"]["warmup_completed_sessions"])
    calendar = pd.concat(calendars.values(), ignore_index=True).loc[
        :, ["period", "score_session", "session_index", "decision_timestamp"]
    ]
    opportunities = (
        surface.drop(columns="session_index", errors="ignore")
        .rename(
            columns={
                "session": "score_session",
                "decision_timestamp": "opportunity_decision_timestamp",
            }
        )
        .merge(
            calendar.loc[:, ["period", "score_session", "session_index"]],
            on=["period", "score_session"],
            how="left",
            validate="many_to_one",
        )
    )
    opportunities = opportunities.loc[opportunities["session_index"].ge(warmup)].copy()
    opportunities["opportunity_id"] = (
        opportunities["period"].astype(str)
        + "-"
        + opportunities["anchor_id"].astype(str)
        + "-h"
        + opportunities["horizon"].astype(str)
    )
    decisions: list[pd.DataFrame] = []
    for model_name in PRIMARY_MODELS:
        model_forecasts = forecasts.loc[forecasts["model_name"].eq(model_name)].copy()
        joined = apply_frozen_admission(opportunities, model_forecasts)
        joined["model_name"] = model_name
        decisions.append(joined)
    no_filter = opportunities.copy()
    no_filter["decision_timestamp"] = no_filter["score_session"].map(
        {
            str(row.score_session): row.decision_timestamp
            for frame in calendars.values()
            for row in frame.itertuples(index=False)
        }
    )
    no_filter["admit_new_entry"] = True
    no_filter["accepted"] = True
    no_filter["decision"] = "accepted"
    no_filter["edge_state"] = "active"
    no_filter["reason_codes"] = "accepted_no_payoff_state_filter"
    no_filter["run_id"] = run_id
    no_filter["model_name"] = "no_payoff_state_filter"
    no_filter["configuration_hash"] = configuration_hash
    no_filter["forecast_frozen_before_payoff"] = True
    no_filter["existing_position_action"] = "unchanged_existing_exit_rule"
    decisions.append(no_filter)
    combined = pd.concat(decisions, ignore_index=True, sort=False)
    combined["accepted_filled"] = combined["accepted"] & combined["status"].eq("filled")
    combined["sector"] = "unavailable"
    combined["month"] = combined["score_session"].astype(str).str[:7]
    combined["year"] = combined["period"].astype(int)
    combined["primary_total_cost_bps"] = np.where(combined["status"].eq("filled"), 10.0, 0.0)
    combined["primary_net_payoff_bps"] = np.where(
        combined["status"].eq("filled"),
        combined["gross_return_bps"].astype(float) - 10.0,
        np.nan,
    )
    if not combined["forecast_frozen_before_payoff"].fillna(False).all():
        raise AssertionError("unfrozen opportunity decision")
    return combined.sort_values(
        ["model_name", "period", "score_session", "symbol_norm", "start_timestamp"],
        kind="stable",
    ).reset_index(drop=True)


def _portfolio_statistics(daily: np.ndarray) -> dict[str, float]:
    values = np.asarray(daily, dtype=float)
    equity = np.cumprod(1.0 + values)
    cumulative = float(equity[-1] - 1.0) if len(equity) else 0.0
    volatility = float(values.std(ddof=1) * math.sqrt(252.0)) if len(values) > 1 else 0.0
    sharpe = (
        float(values.mean() / values.std(ddof=1) * math.sqrt(252.0))
        if len(values) > 1 and values.std(ddof=1) > 0.0
        else math.nan
    )
    if len(equity):
        padded = np.r_[1.0, equity]
        drawdown = padded / np.maximum.accumulate(padded) - 1.0
        maximum_drawdown = float(drawdown.min())
    else:
        maximum_drawdown = 0.0
    return {
        "cumulative_return": cumulative,
        "annualized_volatility": volatility,
        "descriptive_sharpe_zero_rate": sharpe,
        "maximum_drawdown": maximum_drawdown,
    }


def daily_portfolio_returns(
    decisions: pd.DataFrame,
    *,
    cost_multiplier: float,
    universe_size: int,
) -> pd.DataFrame:
    sessions = sorted(decisions["score_session"].astype(str).unique())
    selected = decisions.loc[decisions["accepted_filled"]].copy()
    selected["stressed_net_return"] = (
        selected["gross_return_bps"].to_numpy(float)
        - cost_multiplier * selected["primary_total_cost_bps"].to_numpy(float)
    ) / 10_000.0
    sleeve_rows: list[dict[str, object]] = []
    for (session, stock), group in selected.groupby(
        ["score_session", "symbol_norm"], sort=True, observed=True
    ):
        returns = group.sort_values("start_timestamp", kind="stable")[
            "stressed_net_return"
        ].to_numpy(float)
        sleeve_rows.append(
            {
                "score_session": str(session),
                "symbol_norm": str(stock),
                "sleeve_return": float(np.prod(1.0 + returns) - 1.0),
            }
        )
    sleeves = pd.DataFrame(sleeve_rows)
    if sleeves.empty:
        portfolio = pd.Series(0.0, index=sessions)
    else:
        portfolio = (
            sleeves.groupby("score_session")["sleeve_return"].sum().div(universe_size)
        ).reindex(sessions, fill_value=0.0)
    return pd.DataFrame(
        {"score_session": sessions, "daily_portfolio_return": portfolio.to_numpy(float)}
    )


def trading_summary(
    decisions: pd.DataFrame,
    *,
    cost_multiplier: float,
    universe_size: int,
) -> dict[str, float | int]:
    filled = decisions.loc[decisions["accepted_filled"]].copy()
    gross = filled["gross_return_bps"].to_numpy(float)
    net = gross - cost_multiplier * filled["primary_total_cost_bps"].to_numpy(float)
    daily = daily_portfolio_returns(
        decisions, cost_multiplier=cost_multiplier, universe_size=universe_size
    )
    statistics = _portfolio_statistics(daily["daily_portfolio_return"].to_numpy(float))
    session_count = max(decisions["score_session"].nunique(), 1)
    exposure = float(filled["holding_bars"].sum() / (session_count * 78.0 * universe_size))
    return {
        "opportunity_count": int(len(decisions)),
        "accepted_signal_count": int(decisions["accepted"].sum()),
        "rejected_signal_count": int((~decisions["accepted"]).sum()),
        "accepted_trade_count": int(len(filled)),
        "gross_pnl_bps": float(gross.sum()) if len(gross) else 0.0,
        "total_cost_bps": float(cost_multiplier * filled["primary_total_cost_bps"].sum()),
        "net_pnl_bps": float(net.sum()) if len(net) else 0.0,
        "net_return_per_accepted_trade_bps": float(net.mean()) if len(net) else math.nan,
        "net_return_per_opportunity_bps": float(net.sum() / len(decisions))
        if len(decisions)
        else math.nan,
        "hit_rate": float((net > 0.0).mean()) if len(net) else math.nan,
        "turnover_round_trips_per_session": float(2.0 * len(filled) / session_count),
        "exposure_fraction": exposure,
        "coverage": float(decisions["accepted"].mean()) if len(decisions) else 0.0,
        "abstention_rate": float((~decisions["accepted"]).mean()) if len(decisions) else 0.0,
        **statistics,
    }


def evaluate_prediction_models(
    forecasts: pd.DataFrame,
    payoff_panel: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = payoff_panel.rename(columns={"session": "score_session"}).copy()
    target["positive_target"] = target["robust_net_payoff_bps"].gt(0.0)
    keys = ["period", "score_session", "loop_id", "orientation", "horizon"]
    scored = forecasts.merge(
        target.loc[:, [*keys, "robust_net_payoff_bps", "positive_target"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("pooled", scored)]
    scopes.extend(
        (str(period), scored.loc[scored["period"].eq(period)])
        for period in config["evaluation"]["periods"]
    )
    bins = np.linspace(0.0, 1.0, int(config["evaluation"]["calibration_bins"]) + 1)
    for model_name in PRIMARY_MODELS:
        for scope, frame in scopes:
            frame = frame.loc[frame["model_name"].eq(model_name)].copy()
            if frame.empty:
                continue
            probabilities = np.clip(frame["p_edge_positive"].to_numpy(float), 1e-6, 1 - 1e-6)
            targets = frame["positive_target"].to_numpy(bool)
            log_loss = float(
                -np.mean(targets * np.log(probabilities) + (~targets) * np.log(1.0 - probabilities))
            )
            brier = float(np.mean((probabilities - targets.astype(float)) ** 2))
            predicted = frame["edge_state"].eq("active").to_numpy()
            made_prediction = ~frame["edge_state"].eq("unknown").to_numpy()
            true_positive = int(np.sum(predicted & targets))
            false_positive = int(np.sum(predicted & ~targets))
            false_negative = int(np.sum(~predicted & targets))
            correct = predicted == targets
            std = np.maximum(frame["posterior_std_net_bps"].to_numpy(float), 1e-6)
            continuous_log_density = float(
                np.mean(
                    student_t.logpdf(
                        (
                            frame["robust_net_payoff_bps"].to_numpy(float)
                            - frame["posterior_mean_net_bps"].to_numpy(float)
                        )
                        / std,
                        df=float(config["observation_model"]["student_t_degrees_of_freedom_floor"]),
                    )
                    - np.log(std)
                )
            )
            calibration_index = np.clip(np.digitize(probabilities, bins) - 1, 0, len(bins) - 2)
            ece = 0.0
            for bin_index in range(len(bins) - 1):
                selected = calibration_index == bin_index
                count = int(selected.sum())
                if not count:
                    continue
                mean_probability = float(probabilities[selected].mean())
                observed_rate = float(targets[selected].mean())
                ece += count / len(frame) * abs(mean_probability - observed_rate)
                calibration_rows.append(
                    {
                        "model_name": model_name,
                        "scope": scope,
                        "probability_bin_lower": float(bins[bin_index]),
                        "probability_bin_upper": float(bins[bin_index + 1]),
                        "rows": count,
                        "mean_probability": mean_probability,
                        "observed_positive_rate": observed_rate,
                        "absolute_calibration_error": abs(mean_probability - observed_rate),
                    }
                )
            metric_rows.append(
                {
                    "model_name": model_name,
                    "scope": scope,
                    "predictive_rows": int(len(frame)),
                    "predictive_log_loss": log_loss,
                    "prequential_predictive_log_density": continuous_log_density,
                    "brier_score": brier,
                    "expected_calibration_error": float(ece),
                    "activation_precision": true_positive / (true_positive + false_positive)
                    if true_positive + false_positive
                    else math.nan,
                    "activation_recall": true_positive / (true_positive + false_negative)
                    if true_positive + false_negative
                    else math.nan,
                    "false_activation_rate": false_positive / max(int((~targets).sum()), 1),
                    "false_retirement_rate": float(
                        (frame["edge_state"].eq("retired") & frame["positive_target"]).sum()
                        / max(int(targets.sum()), 1)
                    ),
                    "abstention_rate_predictive": float((~made_prediction).mean()),
                    "predictive_coverage": float(made_prediction.mean()),
                    "conditional_accuracy": float(correct[made_prediction].mean())
                    if made_prediction.any()
                    else math.nan,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(calibration_rows), scored


def model_comparison_metrics(
    prediction_metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    universe_size = 20
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("pooled", decisions)]
    scopes.extend(
        (str(period), decisions.loc[decisions["period"].eq(period)])
        for period in config["evaluation"]["periods"]
    )
    for model_name in (*PRIMARY_MODELS, "no_payoff_state_filter"):
        for scope, scope_frame in scopes:
            frame = scope_frame.loc[scope_frame["model_name"].eq(model_name)].copy()
            trading = trading_summary(frame, cost_multiplier=1.0, universe_size=universe_size)
            prediction = prediction_metrics.loc[
                prediction_metrics["model_name"].eq(model_name)
                & prediction_metrics["scope"].eq(scope)
            ]
            row: dict[str, object] = {"model_name": model_name, "scope": scope, **trading}
            if not prediction.empty:
                row.update(prediction.iloc[0].to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def trading_slices(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    dimensions = {
        "year": "year",
        "month": "month",
        "loop": "loop_id",
        "orientation": "orientation",
        "sector": "sector",
        "stock": "symbol_norm",
    }
    if "state_change_phase" in decisions:
        dimensions["state_change_phase"] = "state_change_phase"
    rows: list[dict[str, object]] = []
    for model_name, model_frame in decisions.groupby("model_name", sort=True):
        for dimension, column in dimensions.items():
            for value, frame in model_frame.groupby(column, sort=True, dropna=False):
                filled = frame.loc[frame["accepted_filled"]]
                net = filled["primary_net_payoff_bps"].to_numpy(float)
                rows.append(
                    {
                        "model_name": model_name,
                        "dimension": dimension,
                        "value": str(value),
                        "opportunities": int(len(frame)),
                        "accepted_signals": int(frame["accepted"].sum()),
                        "accepted_trades": int(len(filled)),
                        "gross_pnl_bps": float(filled["gross_return_bps"].sum()),
                        "total_cost_bps": float(filled["primary_total_cost_bps"].sum()),
                        "net_pnl_bps": float(net.sum()) if len(net) else 0.0,
                        "mean_net_trade_bps": float(net.mean()) if len(net) else math.nan,
                        "hit_rate": float((net > 0.0).mean()) if len(net) else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def add_state_change_phase(
    decisions: pd.DataFrame,
    forecasts: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    result = decisions.copy()
    result["state_change_phase"] = "other"
    changes = forecasts.loc[
        forecasts["model_name"].eq("hierarchical_change_point")
        & forecasts["p_change_now"].ge(threshold),
        ["period", "loop_id", "orientation", "score_session"],
    ].copy()
    change_lookup: dict[tuple[int, str, str], set[str]] = {}
    for key, frame in changes.groupby(["period", "loop_id", "orientation"], observed=True):
        change_lookup[(int(key[0]), str(key[1]), str(key[2]))] = set(
            frame["score_session"].astype(str)
        )
    for key, indices in result.groupby(
        ["period", "loop_id", "orientation"], observed=True
    ).groups.items():
        sessions = sorted(result.loc[indices, "score_session"].astype(str).unique())
        positions = {session: index for index, session in enumerate(sessions)}
        change_sessions = change_lookup.get((int(key[0]), str(key[1]), str(key[2])), set())
        phase_by_session: dict[str, str] = {}
        for session in sessions:
            if session in change_sessions:
                phase_by_session[session] = "change_session"
                continue
            prior_distances = [
                positions[session] - positions[changed]
                for changed in change_sessions
                if changed in positions and positions[changed] < positions[session]
            ]
            if prior_distances and min(prior_distances) <= 3:
                phase_by_session[session] = "after_change_1_3"
            else:
                phase_by_session[session] = "other"
        result.loc[indices, "state_change_phase"] = result.loc[indices, "score_session"].map(
            phase_by_session
        )
    return result


def identify_hindsight_episodes(
    payoff_panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    feature_panel: pd.DataFrame,
    calendars: Mapping[int, pd.DataFrame],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create hindsight labels for evaluation only; never returned to a model."""

    warmup = int(config["support"]["warmup_completed_sessions"])
    neutral_band = float(config["evaluation"]["episode_neutral_band_bps"])
    smooth = int(config["evaluation"]["episode_smoothing_sessions"])
    calendar_index = pd.concat(calendars.values(), ignore_index=True).loc[
        :, ["period", "score_session", "session_index"]
    ]
    panel = payoff_panel.rename(columns={"session": "score_session"}).merge(
        calendar_index,
        on=["period", "score_session"],
        how="left",
        validate="many_to_one",
    )
    panel = panel.loc[panel["session_index"].ge(warmup)].copy()
    state_rows: list[pd.DataFrame] = []
    episode_rows: list[dict[str, object]] = []
    full = forecasts.loc[forecasts["model_name"].eq("hierarchical_change_point")].copy()
    episode_counter = 0
    group_keys = ["period", "loop_id", "orientation", "horizon"]
    for key, group in panel.groupby(group_keys, sort=True, observed=True):
        group = group.sort_values("session_index", kind="stable").copy()
        group["hindsight_smoothed_net_bps"] = (
            group["robust_net_payoff_bps"].rolling(smooth, center=True, min_periods=2).mean()
        )
        smoothed = group["hindsight_smoothed_net_bps"].to_numpy(float)
        labels = np.where(
            smoothed > neutral_band,
            "positive",
            np.where(smoothed < -neutral_band, "negative", "neutral"),
        ).astype(object)
        positive_positions = np.flatnonzero(labels == "positive")
        for position in positive_positions:
            next_values = smoothed[position + 1 : position + 4]
            if (
                len(next_values)
                and np.nanmean(next_values) < smoothed[position]
                and (next_values <= neutral_band).any()
            ):
                labels[position] = "decaying"
        group["hindsight_payoff_state"] = labels
        state_rows.append(group)
        episode_mask = np.isin(labels, ["positive", "decaying"])
        starts = np.flatnonzero(episode_mask & ~np.r_[False, episode_mask[:-1]])
        ends = np.flatnonzero(episode_mask & ~np.r_[episode_mask[1:], False])
        for start, end in zip(starts, ends, strict=True):
            segment = group.iloc[start : end + 1].copy()
            if len(segment) < 2:
                continue
            episode_counter += 1
            episode_id = f"episode_{episode_counter:04d}"
            onset = str(segment["score_session"].iloc[0])
            episode_end = str(segment["score_session"].iloc[-1])
            period = int(key[0])
            loop_id = str(key[1])
            orientation = str(key[2])
            model = full.loc[
                full["period"].eq(period)
                & full["loop_id"].eq(loop_id)
                & full["orientation"].eq(orientation)
                & full["score_session"].ge(onset)
                & full["score_session"].le(episode_end)
            ].sort_values("session_index", kind="stable")
            active = model.loc[model["edge_state"].eq("active")]
            activation_date = str(active["score_session"].iloc[0]) if not active.empty else None
            onset_index = int(segment["session_index"].iloc[0])
            end_index = int(segment["session_index"].iloc[-1])
            activation_delay = (
                int(active["session_index"].iloc[0]) - onset_index if not active.empty else math.nan
            )
            after = full.loc[
                full["period"].eq(period)
                & full["loop_id"].eq(loop_id)
                & full["orientation"].eq(orientation)
                & full["session_index"].ge(end_index)
                & full["edge_state"].isin(["decaying", "retired"])
            ].sort_values("session_index", kind="stable")
            termination_date = str(after["score_session"].iloc[0]) if not after.empty else None
            termination_delay = (
                int(after["session_index"].iloc[0]) - end_index if not after.empty else math.nan
            )
            state_by_session = dict(
                zip(model["score_session"].astype(str), model["edge_state"], strict=True)
            )
            captured_mask = segment["score_session"].map(state_by_session).eq("active")
            captured = float(segment.loc[captured_mask, "robust_net_payoff_bps"].sum())
            missed = float(segment.loc[~captured_mask, "robust_net_payoff_bps"].sum())
            duration = end_index - onset_index + 1
            feature_history = feature_panel.loc[
                feature_panel["period"].eq(period)
                & feature_panel["loop_id"].eq(loop_id)
                & feature_panel["orientation"].eq(orientation)
                & feature_panel["score_session"].le(onset)
            ].sort_values("score_session", kind="stable")
            recent_features = feature_history.tail(3)
            earlier_features = feature_history.iloc[:-3].tail(3)

            def feature_change(
                name: str,
                recent: pd.DataFrame = recent_features,
                earlier: pd.DataFrame = earlier_features,
            ) -> float:
                if recent.empty or earlier.empty:
                    return math.nan
                return float(recent[name].mean() - earlier[name].mean())

            breadth_change = feature_change("structural_breadth")
            coherence_change = feature_change("top_second_margin")
            dispersion_change = feature_change("payoff_dispersion")
            surprise_change = feature_change("transition_surprise")
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "loop_id": loop_id,
                    "orientation": orientation,
                    "period": period,
                    "horizon": int(key[3]),
                    "hindsight_estimated_onset": onset,
                    "hindsight_estimated_end": episode_end,
                    "duration_sessions": duration,
                    "observed_payoff_sessions": int(len(segment)),
                    "mean_session_payoff_bps": float(segment["robust_net_payoff_bps"].mean()),
                    "total_episode_payoff_bps": float(segment["robust_net_payoff_bps"].sum()),
                    "causal_activation_date": activation_date,
                    "activation_delay_sessions": activation_delay,
                    "causal_retirement_or_decay_date": termination_date,
                    "termination_delay_sessions": termination_delay,
                    "percentage_episode_captured": float(captured_mask.mean()),
                    "net_payoff_captured_bps": captured,
                    "net_payoff_missed_bps": missed,
                    "activated_too_late": bool(
                        not math.isfinite(float(activation_delay))
                        or float(activation_delay) / max(duration, 1) > 0.5
                    ),
                    "remained_active_too_long": bool(
                        math.isfinite(float(termination_delay)) and float(termination_delay) > 3.0
                    ),
                    "breadth_change_before_onset": breadth_change,
                    "breadth_increased_before_onset": bool(
                        math.isfinite(breadth_change) and breadth_change > 0.0
                    ),
                    "coherence_change_before_onset": coherence_change,
                    "coherence_increased_before_onset": bool(
                        math.isfinite(coherence_change) and coherence_change > 0.0
                    ),
                    "dispersion_change_before_decay": dispersion_change,
                    "dispersion_increased_before_decay": bool(
                        math.isfinite(dispersion_change) and dispersion_change > 0.0
                    ),
                    "structural_surprise_change_before_decay": surprise_change,
                    "structural_surprise_increased_before_decay": bool(
                        math.isfinite(surprise_change) and surprise_change > 0.0
                    ),
                }
            )
    states = pd.concat(state_rows, ignore_index=True) if state_rows else pd.DataFrame()
    return pd.DataFrame(episode_rows), states


def change_point_diagnostics(
    forecasts: pd.DataFrame,
    episodes: pd.DataFrame,
    hindsight_states: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    threshold = float(config["change_point"]["change_reset_probability"])
    rows: list[dict[str, object]] = []
    for model_name in PRIMARY_MODELS:
        model = forecasts.loc[forecasts["model_name"].eq(model_name)].copy()
        state_change = model.groupby(["period", "loop_id", "orientation"], observed=True)[
            "edge_state"
        ].transform(lambda values: values.ne(values.shift()) & values.shift().notna())
        detected = model.loc[model["p_change_now"].ge(threshold) | state_change].copy()
        activation_delays: list[float] = []
        termination_delays: list[float] = []
        lag_ratios: list[float] = []
        captured_payoff = 0.0
        missed_payoff = 0.0
        captured_fractions: list[float] = []
        detected_episodes = 0
        boundary_keys: list[tuple[int, str, str, int]] = []
        for episode in episodes.itertuples(index=False):
            episode_model = model.loc[
                model["period"].eq(int(episode.period))
                & model["loop_id"].eq(str(episode.loop_id))
                & model["orientation"].eq(str(episode.orientation))
            ].sort_values("session_index", kind="stable")
            onset_index_rows = episode_model.loc[
                episode_model["score_session"].eq(str(episode.hindsight_estimated_onset))
            ]
            end_index_rows = episode_model.loc[
                episode_model["score_session"].eq(str(episode.hindsight_estimated_end))
            ]
            if onset_index_rows.empty or end_index_rows.empty:
                continue
            onset_index = int(onset_index_rows["session_index"].iloc[0])
            end_index = int(end_index_rows["session_index"].iloc[0])
            boundary_keys.extend(
                [
                    (
                        int(episode.period),
                        str(episode.loop_id),
                        str(episode.orientation),
                        onset_index,
                    ),
                    (
                        int(episode.period),
                        str(episode.loop_id),
                        str(episode.orientation),
                        end_index,
                    ),
                ]
            )
            within = episode_model.loc[
                episode_model["session_index"].between(onset_index, end_index)
            ]
            activation = within.loc[within["edge_state"].eq("active")]
            if not activation.empty:
                delay = float(int(activation["session_index"].iloc[0]) - onset_index)
                activation_delays.append(delay)
                lag_ratios.append(delay / max(int(episode.duration_sessions), 1))
                detected_episodes += 1
                captured_fractions.append(float(within["edge_state"].eq("active").mean()))
            after = episode_model.loc[
                episode_model["session_index"].ge(end_index)
                & episode_model["edge_state"].isin(["decaying", "retired"])
            ]
            if not after.empty:
                termination_delays.append(float(int(after["session_index"].iloc[0]) - end_index))
            state_segment = hindsight_states.loc[
                hindsight_states["period"].eq(int(episode.period))
                & hindsight_states["loop_id"].eq(str(episode.loop_id))
                & hindsight_states["orientation"].eq(str(episode.orientation))
                & hindsight_states["score_session"].between(
                    str(episode.hindsight_estimated_onset),
                    str(episode.hindsight_estimated_end),
                )
            ].merge(
                within.loc[:, ["score_session", "edge_state"]],
                on="score_session",
                how="left",
                validate="one_to_one",
            )
            is_active = state_segment["edge_state"].eq("active")
            captured_payoff += float(state_segment.loc[is_active, "robust_net_payoff_bps"].sum())
            missed_payoff += float(state_segment.loc[~is_active, "robust_net_payoff_bps"].sum())
        false_changes = 0
        for change in detected.itertuples(index=False):
            nearby = any(
                period == int(change.period)
                and loop_id == str(change.loop_id)
                and orientation == str(change.orientation)
                and abs(index - int(change.session_index)) <= 2
                for period, loop_id, orientation, index in boundary_keys
            )
            false_changes += int(not nearby)
        decaying = model.loc[model["edge_state"].eq("decaying")].merge(
            hindsight_states.loc[
                :,
                [
                    "period",
                    "score_session",
                    "loop_id",
                    "orientation",
                    "horizon",
                    "robust_net_payoff_bps",
                ],
            ],
            on=["period", "score_session", "loop_id", "orientation", "horizon"],
            how="inner",
        )
        rows.append(
            {
                "model_name": model_name,
                "hindsight_positive_episodes": int(len(episodes)),
                "episodes_detected": detected_episodes,
                "fraction_hindsight_positive_episodes_detected": detected_episodes
                / max(len(episodes), 1),
                "mean_activation_delay_sessions": float(np.mean(activation_delays))
                if activation_delays
                else math.nan,
                "median_activation_delay_sessions": float(np.median(activation_delays))
                if activation_delays
                else math.nan,
                "mean_termination_delay_sessions": float(np.mean(termination_delays))
                if termination_delays
                else math.nan,
                "detected_change_points": int(len(detected)),
                "false_change_points": false_changes,
                "mean_fraction_episode_captured_after_activation": float(
                    np.mean(captured_fractions)
                )
                if captured_fractions
                else 0.0,
                "net_payoff_captured_after_activation_bps": captured_payoff,
                "net_payoff_missed_before_activation_bps": missed_payoff,
                "loss_incurred_during_decaying_bps": float(
                    decaying.loc[
                        decaying["robust_net_payoff_bps"].lt(0.0),
                        "robust_net_payoff_bps",
                    ].sum()
                ),
                "detection_lag_ratio": float(np.mean(lag_ratios)) if lag_ratios else math.nan,
            }
        )
    return pd.DataFrame(rows)


def concentration_analysis(
    slices: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, frame in slices.loc[slices["dimension"].eq("stock")].groupby(
        "model_name", sort=True
    ):
        positive = frame["net_pnl_bps"].clip(lower=0.0)
        best = frame.sort_values("net_pnl_bps", ascending=False).head(5)
        rows.append(
            {
                "model_name": model_name,
                "concentration_type": "best_stocks",
                "top_item": str(best["value"].iloc[0]) if not best.empty else None,
                "top_item_net_pnl_bps": float(best["net_pnl_bps"].iloc[0])
                if not best.empty
                else 0.0,
                "top_five_share_of_positive_contribution": float(
                    best["net_pnl_bps"].clip(lower=0.0).sum() / positive.sum()
                )
                if positive.sum() > 0.0
                else math.nan,
            }
        )
    if not episodes.empty:
        best_episode = episodes.sort_values("total_episode_payoff_bps", ascending=False).iloc[0]
        positive_total = episodes["total_episode_payoff_bps"].clip(lower=0.0).sum()
        rows.append(
            {
                "model_name": "hierarchical_change_point",
                "concentration_type": "best_episodes",
                "top_item": best_episode["episode_id"],
                "top_item_net_pnl_bps": float(best_episode["total_episode_payoff_bps"]),
                "top_five_share_of_positive_contribution": float(
                    episodes.nlargest(5, "total_episode_payoff_bps")["total_episode_payoff_bps"]
                    .clip(lower=0.0)
                    .sum()
                    / positive_total
                )
                if positive_total > 0.0
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _replace_policy(
    base_decisions: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    label: str,
    admission_override: pd.Series | None = None,
) -> pd.DataFrame:
    base = base_decisions.loc[base_decisions["model_name"].eq("hierarchical_change_point")].copy()
    keys = ["period", "score_session", "loop_id", "orientation", "horizon"]
    policy = forecasts.loc[:, [*keys, "admit_new_entry"]].copy()
    if admission_override is not None:
        policy["admit_new_entry"] = admission_override.to_numpy(bool)
    policy = policy.rename(columns={"admit_new_entry": "stress_admit"})
    base = base.drop(columns=["stress_admit"], errors="ignore").merge(
        policy,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    base["accepted"] = base["stress_admit"].fillna(False).astype(bool)
    base["accepted_filled"] = base["accepted"] & base["status"].eq("filled")
    base["model_name"] = label
    return base


def _delayed_policy(decisions: pd.DataFrame) -> pd.DataFrame:
    keys = ["period", "loop_id", "orientation", "horizon", "score_session"]
    policy = (
        decisions.loc[:, [*keys, "accepted"]].drop_duplicates(keys).sort_values(keys, kind="stable")
    )
    policy["delayed_accept"] = policy.groupby(
        ["period", "loop_id", "orientation", "horizon"], observed=True
    )["accepted"].shift(1, fill_value=False)
    result = decisions.merge(
        policy.loc[:, [*keys, "delayed_accept"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    result["accepted"] = result["delayed_accept"].fillna(False).astype(bool)
    result["accepted_filled"] = result["accepted"] & result["status"].eq("filled")
    return result


def stress_test_results(
    decisions: pd.DataFrame,
    forecasts: pd.DataFrame,
    episodes: pd.DataFrame,
    config: Mapping[str, Any],
    sensitivity_decisions: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def record(
        stress_name: str,
        frame: pd.DataFrame,
        *,
        model_name: str,
        cost_multiplier: float = 1.0,
        detail: str = "",
    ) -> None:
        rows.append(
            {
                "stress_test": stress_name,
                "model_name": model_name,
                "detail": detail,
                "cost_multiplier": cost_multiplier,
                **trading_summary(
                    frame,
                    cost_multiplier=cost_multiplier,
                    universe_size=20,
                ),
            }
        )

    for model_name in (*PRIMARY_MODELS, "no_payoff_state_filter"):
        base = decisions.loc[decisions["model_name"].eq(model_name)].copy()
        record("primary_cost", base, model_name=model_name)
        record(
            "twice_estimated_costs",
            base,
            model_name=model_name,
            cost_multiplier=float(config["session_aggregation"]["cost_stress_multiplier"]),
        )
        record(
            "one_session_entry_delay",
            _delayed_policy(base),
            model_name=model_name,
        )

    full = decisions.loc[decisions["model_name"].eq("hierarchical_change_point")].copy()
    for stock in sorted(full["symbol_norm"].astype(str).unique()):
        record(
            "leave_one_stock_out",
            full.loc[~full["symbol_norm"].eq(stock)],
            model_name="hierarchical_change_point",
            detail=f"excluded={stock}",
        )
    filled = full.loc[full["accepted_filled"]].copy()
    month_net = filled.groupby("month")["primary_net_payoff_bps"].sum()
    if not month_net.empty:
        best_month = str(month_net.idxmax())
        record(
            "remove_best_month",
            full.loc[~full["month"].eq(best_month)],
            model_name="hierarchical_change_point",
            detail=f"removed={best_month}",
        )
    if not episodes.empty:
        best_episode = episodes.nlargest(1, "total_episode_payoff_bps").iloc[0]
        in_episode = (
            full["period"].eq(int(best_episode["period"]))
            & full["loop_id"].eq(str(best_episode["loop_id"]))
            & full["orientation"].eq(str(best_episode["orientation"]))
            & full["score_session"].between(
                str(best_episode["hindsight_estimated_onset"]),
                str(best_episode["hindsight_estimated_end"]),
            )
        )
        record(
            "remove_best_profitable_episode",
            full.loc[~in_episode],
            model_name="hierarchical_change_point",
            detail=f"removed={best_episode['episode_id']}",
        )
    liquid_values = full["dollar_volume_proxy"].replace([np.inf, -np.inf], np.nan).dropna()
    if not liquid_values.empty:
        minimum_threshold = float(
            liquid_values.quantile(float(config["stress_tests"]["minimum_liquidity_quantile"]))
        )
        record(
            "minimum_liquidity_subset",
            full.loc[full["dollar_volume_proxy"].ge(minimum_threshold)],
            model_name="hierarchical_change_point",
            detail=f"threshold={minimum_threshold:.6g}",
        )
        stock_liquidity = full.groupby("symbol_norm")["dollar_volume_proxy"].median()
        cutoff = float(
            stock_liquidity.quantile(float(config["stress_tests"]["exclude_least_liquid_quantile"]))
        )
        excluded = sorted(stock_liquidity.loc[stock_liquidity.le(cutoff)].index.astype(str))
        record(
            "exclude_least_liquid_stocks",
            full.loc[~full["symbol_norm"].isin(excluded)],
            model_name="hierarchical_change_point",
            detail="excluded=" + ",".join(excluded),
        )
    turnover = full["volume_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    if not turnover.empty:
        cutoff = float(
            turnover.quantile(
                1.0 - float(config["stress_tests"]["exclude_highest_turnover_quantile"])
            )
        )
        record(
            "exclude_highest_turnover_opportunities",
            full.loc[full["volume_ratio"].le(cutoff) | full["volume_ratio"].isna()],
            model_name="hierarchical_change_point",
            detail=f"maximum_volume_ratio={cutoff:.6g}",
        )

    main_forecast = forecasts.loc[forecasts["model_name"].eq("hierarchical_change_point")].copy()
    support = config["support"]
    ood_threshold = float(config["thresholds"]["out_of_distribution_score"])
    for active_threshold in config["thresholds"]["nearby_active_probability"]:
        admitted = (
            main_forecast["p_edge_active"].ge(float(active_threshold))
            & main_forecast["posterior_lower_bound_net_bps"].gt(0.0)
            & main_forecast["p_survive_horizon"].ge(
                float(config["thresholds"]["survival_probability"])
            )
            & main_forecast["effective_sessions"].ge(int(support["minimum_independent_sessions"]))
            & main_forecast["independent_stocks"].ge(int(support["minimum_independent_stocks"]))
            & main_forecast["effective_sample_size"].ge(
                float(support["minimum_effective_sample_size"])
            )
            & main_forecast["out_of_distribution_score"].le(ood_threshold)
            & main_forecast["required_features_available"].astype(bool)
        )
        policy = _replace_policy(
            decisions,
            main_forecast,
            label=f"active_threshold_{active_threshold}",
            admission_override=admitted,
        )
        record(
            "nearby_active_probability_threshold",
            policy,
            model_name="hierarchical_change_point",
            detail=f"threshold={active_threshold}",
        )
    for survival_threshold in config["thresholds"]["nearby_survival_probability"]:
        admitted = (
            main_forecast["p_edge_active"].ge(float(config["thresholds"]["active_probability"]))
            & main_forecast["posterior_lower_bound_net_bps"].gt(0.0)
            & main_forecast["p_survive_horizon"].ge(float(survival_threshold))
            & main_forecast["effective_sessions"].ge(int(support["minimum_independent_sessions"]))
            & main_forecast["independent_stocks"].ge(int(support["minimum_independent_stocks"]))
            & main_forecast["effective_sample_size"].ge(
                float(support["minimum_effective_sample_size"])
            )
            & main_forecast["out_of_distribution_score"].le(ood_threshold)
            & main_forecast["required_features_available"].astype(bool)
        )
        policy = _replace_policy(
            decisions,
            main_forecast,
            label=f"survival_threshold_{survival_threshold}",
            admission_override=admitted,
        )
        record(
            "nearby_survival_probability_threshold",
            policy,
            model_name="hierarchical_change_point",
            detail=f"threshold={survival_threshold}",
        )
    for name, frame in sensitivity_decisions.items():
        record(name, frame, model_name="hierarchical_change_point")
    return pd.DataFrame(rows)


def representative_episode_plots(
    episodes: pd.DataFrame,
    hindsight_states: pd.DataFrame,
    forecasts: pd.DataFrame,
    decisions: pd.DataFrame,
    output: Path,
    count: int,
) -> list[str]:
    if episodes.empty or count <= 0:
        return []
    names: list[str] = []
    selected = episodes.nlargest(count, "total_episode_payoff_bps")
    full_forecasts = forecasts.loc[forecasts["model_name"].eq("hierarchical_change_point")]
    full_decisions = decisions.loc[decisions["model_name"].eq("hierarchical_change_point")]
    state_colours = {
        "unknown": "#a8a29e",
        "active": "#16a34a",
        "decaying": "#f59e0b",
        "retired": "#dc2626",
    }
    for episode in selected.itertuples(index=False):
        padding = 10
        period = int(episode.period)
        loop_id = str(episode.loop_id)
        orientation = str(episode.orientation)
        forecast = full_forecasts.loc[
            full_forecasts["period"].eq(period)
            & full_forecasts["loop_id"].eq(loop_id)
            & full_forecasts["orientation"].eq(orientation)
        ].sort_values("session_index", kind="stable")
        onset_rows = forecast.loc[
            forecast["score_session"].eq(str(episode.hindsight_estimated_onset))
        ]
        end_rows = forecast.loc[forecast["score_session"].eq(str(episode.hindsight_estimated_end))]
        if onset_rows.empty or end_rows.empty:
            continue
        start_index = int(onset_rows["session_index"].iloc[0]) - padding
        end_index = int(end_rows["session_index"].iloc[0]) + padding
        forecast = forecast.loc[forecast["session_index"].between(start_index, end_index)]
        states = hindsight_states.loc[
            hindsight_states["period"].eq(period)
            & hindsight_states["loop_id"].eq(loop_id)
            & hindsight_states["orientation"].eq(orientation)
            & hindsight_states["session_index"].between(start_index, end_index)
        ]
        joined = forecast.merge(
            states.loc[:, ["score_session", "robust_net_payoff_bps"]],
            on="score_session",
            how="left",
            validate="one_to_one",
        )
        dates = pd.to_datetime(joined["score_session"])
        figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
        axes[0].axhline(0.0, color="#44403c", linewidth=0.8)
        axes[0].plot(dates, joined["robust_net_payoff_bps"], color="#1d4ed8", label="session net")
        axes[0].plot(
            dates, joined["posterior_mean_net_bps"], color="#111827", label="posterior mean"
        )
        lower = joined["posterior_mean_net_bps"] - joined["posterior_std_net_bps"]
        upper = joined["posterior_mean_net_bps"] + joined["posterior_std_net_bps"]
        axes[0].fill_between(
            dates, lower, upper, color="#94a3b8", alpha=0.3, label="±1 posterior std"
        )
        axes[0].legend(loc="upper left", ncol=3)
        axes[0].set_ylabel("net bps")
        axes[1].plot(dates, joined["p_edge_active"], label="p_active")
        axes[1].plot(dates, joined["p_survive_horizon"], label="p_survive")
        axes[1].plot(dates, joined["p_change_now"], label="p_change")
        axes[1].legend(loc="upper left", ncol=3)
        axes[1].set_ylim(-0.03, 1.03)
        axes[2].plot(dates, joined["p_on_next"], label="p_on_next", color="#16a34a")
        axes[2].plot(dates, joined["p_off_next"], label="p_off_next", color="#dc2626")
        axes[2].legend(loc="upper left", ncol=2)
        axes[2].set_ylim(-0.03, 1.03)
        for state, colour in state_colours.items():
            mask = joined["edge_state"].eq(state)
            axes[3].scatter(dates[mask], np.full(mask.sum(), 0.5), color=colour, label=state, s=22)
        opportunity = full_decisions.loc[
            full_decisions["period"].eq(period)
            & full_decisions["loop_id"].eq(loop_id)
            & full_decisions["orientation"].eq(orientation)
            & full_decisions["score_session"].isin(joined["score_session"])
        ]
        counts = opportunity.groupby(["score_session", "accepted"]).size().unstack(fill_value=0)
        if True in counts:
            axes[3].bar(
                pd.to_datetime(counts.index),
                counts[True],
                alpha=0.25,
                color="#16a34a",
                label="accepted opportunities",
            )
        if False in counts:
            axes[3].bar(
                pd.to_datetime(counts.index),
                -counts[False],
                alpha=0.2,
                color="#dc2626",
                label="rejected opportunities",
            )
        axes[3].set_ylabel("state / count")
        axes[3].legend(loc="upper left", ncol=3, fontsize=8)
        for axis in axes:
            axis.axvline(
                pd.Timestamp(episode.hindsight_estimated_onset),
                color="#16a34a",
                linestyle="--",
                linewidth=1,
            )
            axis.axvline(
                pd.Timestamp(episode.hindsight_estimated_end),
                color="#dc2626",
                linestyle="--",
                linewidth=1,
            )
            for change_date in dates.loc[joined["p_change_now"].ge(0.45)]:
                axis.axvline(change_date, color="#7c3aed", alpha=0.25, linewidth=0.8)
            axis.grid(alpha=0.15)
        figure.suptitle(f"{episode.episode_id}: {loop_id} / {orientation} / {period}")
        figure.tight_layout()
        name = f"{episode.episode_id}-{loop_id}-{orientation}-{period}.png"
        figure.savefig(output / name, dpi=140)
        plt.close(figure)
        names.append(name)
    return names


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"

    def render(value: object) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(render(row[column]) for column in columns) + " |"
        for _, row in frame.loc[:, columns].iterrows()
    ]
    return "\n".join([header, divider, *body])


def hypothesis_assessment(
    comparison: pd.DataFrame,
    changes: pd.DataFrame,
    stresses: pd.DataFrame,
    concentration: pd.DataFrame,
) -> tuple[str, list[str]]:
    pooled = comparison.loc[comparison["scope"].eq("pooled")].set_index("model_name")
    change = changes.set_index("model_name")
    evidence: list[str] = []
    checks: list[bool] = []
    if {
        "hierarchical_change_point",
        "payoff_only_change_point",
    } <= set(pooled.index):
        calibrated = float(pooled.loc["hierarchical_change_point", "brier_score"]) < float(
            pooled.loc["payoff_only_change_point", "brier_score"]
        )
        checks.append(calibrated)
        evidence.append(
            "breadth/coherence improved Brier over payoff-only"
            if calibrated
            else "breadth/coherence did not improve Brier over payoff-only"
        )
    if {"hierarchical_change_point", "v1_60_session_selector"} <= set(change.index):
        full_lag = float(change.loc["hierarchical_change_point", "detection_lag_ratio"])
        v1_lag = float(change.loc["v1_60_session_selector", "detection_lag_ratio"])
        faster = math.isfinite(full_lag) and (not math.isfinite(v1_lag) or full_lag < v1_lag)
        checks.append(faster)
        evidence.append(
            "V2 reduced detection-lag ratio versus V1"
            if faster
            else "V2 did not reduce detection-lag ratio versus V1"
        )
    if "hierarchical_change_point" in pooled.index:
        positive_net = float(pooled.loc["hierarchical_change_point", "net_pnl_bps"]) > 0.0
        checks.append(positive_net)
        evidence.append(
            "V2 net payoff after costs was positive"
            if positive_net
            else "V2 net payoff after costs was non-positive"
        )
    twice = stresses.loc[
        stresses["stress_test"].eq("twice_estimated_costs")
        & stresses["model_name"].eq("hierarchical_change_point")
    ]
    if not twice.empty:
        robust_cost = float(twice["net_pnl_bps"].iloc[0]) > 0.0
        checks.append(robust_cost)
        evidence.append(
            "V2 survived twice-cost stress" if robust_cost else "V2 failed twice-cost stress"
        )
    concentration_row = concentration.loc[
        concentration["model_name"].eq("hierarchical_change_point")
        & concentration["concentration_type"].eq("best_stocks")
    ]
    if not concentration_row.empty:
        share = float(concentration_row["top_five_share_of_positive_contribution"].iloc[0])
        diversified = math.isfinite(share) and share < 0.75
        checks.append(diversified)
        evidence.append(
            "positive contribution was not dominated by five stocks"
            if diversified
            else "positive contribution was concentrated or absent"
        )
    passed = sum(checks)
    if checks and passed == len(checks):
        return "supported", evidence
    if passed >= 2:
        return "partially_supported", evidence
    return "rejected", evidence


def write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    changes: pd.DataFrame,
    stresses: pd.DataFrame,
    episodes: pd.DataFrame,
    decisions: pd.DataFrame,
    concentration: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    status, evidence = hypothesis_assessment(comparison, changes, stresses, concentration)
    pooled = comparison.loc[comparison["scope"].eq("pooled")].copy()
    model_table = pooled.loc[
        :,
        [
            "model_name",
            "predictive_log_loss",
            "brier_score",
            "expected_calibration_error",
            "detection_lag_ratio" if "detection_lag_ratio" in pooled else "scope",
            "accepted_trade_count",
            "net_pnl_bps",
            "net_return_per_accepted_trade_bps",
            "coverage",
            "descriptive_sharpe_zero_rate",
            "maximum_drawdown",
        ],
    ]
    if "detection_lag_ratio" not in model_table:
        model_table = model_table.merge(
            changes.loc[:, ["model_name", "detection_lag_ratio"]],
            on="model_name",
            how="left",
        )
    calibration_pooled = calibration.loc[calibration["scope"].eq("pooled")]
    twice = stresses.loc[stresses["stress_test"].eq("twice_estimated_costs")]
    delayed = stresses.loc[stresses["stress_test"].eq("one_session_entry_delay")]
    deletions = stresses.loc[stresses["stress_test"].eq("leave_one_stock_out")]
    full_decisions = decisions.loc[decisions["model_name"].eq("hierarchical_change_point")]
    reasons = full_decisions.loc[~full_decisions["accepted"], "reason_codes"].value_counts().head(8)
    episode_activation = (
        episodes["activation_delay_sessions"].dropna()
        if not episodes.empty
        else pd.Series(dtype=float)
    )
    breadth_rate = (
        float(episodes["breadth_increased_before_onset"].mean()) if not episodes.empty else math.nan
    )
    coherence_rate = (
        float(episodes["coherence_increased_before_onset"].mean())
        if not episodes.empty
        else math.nan
    )
    dispersion_rate = (
        float(episodes["dispersion_increased_before_decay"].mean())
        if not episodes.empty
        else math.nan
    )
    surprise_rate = (
        float(episodes["structural_surprise_increased_before_decay"].mean())
        if not episodes.empty
        else math.nan
    )
    best_stock = concentration.loc[
        concentration["model_name"].eq("hierarchical_change_point")
        & concentration["concentration_type"].eq("best_stocks")
    ]
    report = f"""# Dynamic loop edge-state V2

Date: 2026-07-14

Decision: **`temporary_payoff_state_hypothesis_{status}`**

Scientific status: causal retrospective development on already-opened 2023 and 2025 surfaces; not prospective validation and not strategy approval.

Safety: `research_only: true`; `live_ordering_enabled: false`; `order_placement: disabled`. No broker, paper/demo, deployment, position-management, or frozen-exit code was changed.

## 1. Hypothesis

A loop orientation may enter and leave a temporary latent net-payoff state. Structural loop occurrence remains a separate prediction target from economic payoff and admission.

## 2. Existing V1 baseline

V1 is preserved byte-for-byte. It uses 60 completed sessions, raw filled-trade support of 20, a fixed 50-trade pseudocount, and activation when the shrunk net mean is above zero. Its exact summary was reproduced before V2 scoring. At 24 bars V1 averaged -0.01 bps/trade in 2025 and +1.25 bps/trade in backward-2023 and rejected the overall hypothesis.

## 3. Data and field definitions

The source is the frozen `breakout_loop_scores_range_p75` accepted-signal ledger. `loop_id` is the top causal parent cycle; `orientation` is the current causal state within that rotation-invariant parent. Stock is `symbol_norm`; signal bar start is `start_timestamp`; decision is that five-minute bar's close; entry is the triggering-bar start proxy; exit/settlement is the anchor+24 bar close. Gross payoff is the frozen directional simple return. Net payoff subtracts the frozen 5 bps entry and 5 bps exit assumptions.

The original V1 anchor panels and 2023 provider files were ephemeral and expired after the exact V1 rerun. V2 therefore uses a registered, hash-verified recovery adapter: the surviving accepted-trade ledger, the prior 250-session causal loop-scoring artifact, and V1's sealed score/state ledgers. Before any V2 score, the adapter verifies hashes and proves exact equality of top loop, top probability, state, and history token for every one of V1's scored rows. This preserves historical predictions; it does not regenerate them from revised data. Raw volume was not retained, so liquidity stresses use the documented anchor-price × causal-volume-ratio activity proxy and must not be read as true dollar-volume tests.

No bid/ask, spread, slippage, commission, financing, borrow, market-impact, or FX component observations exist in this source. Those component columns are retained as unavailable zero fields rather than fabricated estimates. The provider metadata labels these US symbols' currency `GBP`; V1 computes dimensionless returns, but this inconsistency remains a data-quality warning. Sector metadata is unavailable, so the sector slice is explicitly `unavailable`.

## 4. Registered horizon

The only confirmatory horizon is **24 five-minute bars** (about 120 minutes), selected from the prior V1 follow-up rather than re-optimised here. No horizon search was run.

## 5. Decision-time and settlement-time conventions

Each loop/orientation forecast is frozen at regular-session open, before any current-session anchor or payoff. Only complete session observations whose maximum settlement timestamp is strictly earlier than that open can update the state. Current-session outcomes, unresolved outcomes, and later feature rows cannot train the current gate. Entry trigger time is known only to its five-minute bar, so `entry_timestamp` is the triggering bar start proxy; payoff availability waits until the fixed exit bar closes.

## 6. Session-level aggregation

The statistical unit is session × loop × orientation × 24 bars. Multiple fills first collapse to one capped contribution per stock. The primary observation is an equal-stock mean after 10% winsorisation per tail and a ±500 bps stock cap. Raw fill count, independent-stock count, and Kish equal-weight ESS remain separate. A no-opportunity session is absent, never zero. Median aggregation is a predeclared sensitivity.

## 7. Model implementation

Four frozen selectors were compared: V1, a 10-observation-half-life EWMA with support and uncertainty, payoff-only Student-t BOCPD, and the full hierarchical Student-t BOCPD. The primary BOCPD hazard is 0.05 per observed session (broad geometric mean 20 sessions), with 1/30 and 1/14 sensitivities. Run-length branches are bounded at 120 sessions. A Normal-Inverse-Gamma update supplies Student-t predictives, and one observation is clipped at four branch-predictive scales for robust sufficient-statistic updates.

Separate `p_on_next`, `p_off_next`, and `p_survive_horizon` outputs drive `unknown`, `active`, `decaying`, and `retired` states. Only `active` admits a new entry; the existing frozen exit is always retained.

## 8. Hierarchical pooling approximation

The shared environment is an online winsorised mean across eligible loop/orientation session cells. Each cell retains its own BOCPD. The published mean is an empirical-Bayes blend whose cell weight increases with current-run independent sessions relative to a frozen 12-session pooling strength. Shared and cell uncertainty are combined, with extra sparse-cell variance. This is a practical approximation: it does not learn dynamic loop loadings or a joint covariance matrix, and population contamination remains possible despite the sensitivity and leave-one-stock-out checks.

## 9. Leakage controls

The processing order is explicit: settle complete prior sessions; update shared state; update cells; transform current lagged features against past-only moments; forecast; freeze; then join to current opportunities. Hindsight episode labels are generated only after forecasts and are never model inputs. Focused tests cover settlement, same-session exclusion, appended-future invariance, rolling-scaler isolation, session boundaries, shared-state timing, costs, correlated fills, metadata, and unchanged exits.

## 10. Test results

Before the final historical run, 28 focused V2 tests passed and the exact V1 summary SHA-256 matched its archived exact rerun. Final repository-suite and static-check results are recorded in the run note and handoff response.

## 11. Model comparison

{_markdown_table(model_table, list(model_table.columns))}

## 12. Probability calibration

Positive target: robust session net payoff strictly above zero after 10 bps round trip. Calibration used fixed decile bins; rows were scored prequentially. The machine-readable calibration table has {len(calibration_pooled):,} pooled bin rows. An abstaining model is `unknown`, not a correct positive prediction.

## 13. Activation and termination delays

{_markdown_table(changes, list(changes.columns))}

## 14. Detection-lag ratio

`detection_lag_ratio = causal activation delay / hindsight positive episode length`. Missing ratios mean the model never activated during a qualifying hindsight episode; they are not treated as zero delay.

## 15. Trading results after costs

The table in section 11 reports frozen-exit results after 5 bps per side. Accepted/rejected counts include all frozen signal opportunities; trade counts include only hypothetical fills. The payoff-state model is only an admission overlay and does not refill later overlapping opportunities.

## 16. Twice-cost stress

{_markdown_table(twice, ["model_name", "accepted_trade_count", "net_pnl_bps", "net_return_per_accepted_trade_bps", "cumulative_return", "maximum_drawdown"])}

## 17. Delayed-entry stress

{_markdown_table(delayed, ["model_name", "accepted_trade_count", "net_pnl_bps", "net_return_per_accepted_trade_bps", "cumulative_return", "maximum_drawdown"])}

## 18. Leave-one-stock-out

The full model produced {len(deletions)} leave-one-stock-out rows. Net P&L range: {float(deletions["net_pnl_bps"].min()) if not deletions.empty else math.nan:.2f} to {float(deletions["net_pnl_bps"].max()) if not deletions.empty else math.nan:.2f} bps; positive deletions: {int(deletions["net_pnl_bps"].gt(0).sum())}/{len(deletions)}.

## 19. Episode analysis

The hindsight diagnostic found {len(episodes)} positive/decaying episodes. Median finite causal activation delay was {float(episode_activation.median()) if not episode_activation.empty else math.nan:.2f} sessions. These are evaluation labels, not predictions. A loop is called predicted only when its frozen session-open forecast activated before the associated payoff was observed.

## 20. Failure cases

Most common full-model rejection combinations:

{chr(10).join(f"- `{reason}`: {count}" for reason, count in reasons.items()) if not reasons.empty else "- No rejected opportunities."}

## 21. Concentration analysis

Best-stock diagnostic: {safe(best_stock.iloc[0].to_dict()) if not best_stock.empty else "unavailable"}. Machine-readable stock, loop, orientation, month, episode, and other slices are exported. Sector concentration cannot be assessed on this frozen source.

## 22. Did breadth/coherence lead payoff changes?

Breadth increased before {breadth_rate:.1%} of hindsight episodes and top-versus-second coherence increased before {coherence_rate:.1%}. Dispersion increased before decay in {dispersion_rate:.1%}; structural surprise increased in {surprise_rate:.1%}. These are descriptive lead diagnostics. The stronger causal test is whether the full model improved Brier/log loss and delay over payoff-only; see sections 11 and 13.

## 23. Hypothesis assessment

**{status.replace("_", " ")}.**

{chr(10).join(f"- {item}." for item in evidence)}

Higher filtered P&L alone is not treated as success. Calibration, delay, costs, delayed entry, abstention, and concentration are part of this decision.

## 24. Exact next recommendation

Freeze this V2 implementation and log it prospectively on genuinely new sessions without execution. Do not retune thresholds on 2023/2025. The single highest-value next experiment is a sealed prospective comparison of payoff-only versus breadth/coherence hierarchy, with immutable forecasts and enough independent session/stock support to estimate activation and termination calibration.

## Reproducibility

- Run ID: `{metadata["run_id"]}`
- Git SHA: `{metadata["git_sha"]}`
- Branch: `{metadata["repository_branch"]}`
- Configuration SHA-256: `{metadata["configuration_hash"]}`
- Data snapshot SHA-256: `{metadata["data_snapshot_identifier"]}`
- Command: `{metadata["command"]}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    return status


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_run_metadata(
    *,
    config: Mapping[str, Any],
    configuration_hash: str,
    pre_score: Mapping[str, Any],
    run_id: str,
    output: Path,
    report: Path,
    command: str,
) -> dict[str, Any]:
    source_hashes = dict(pre_score["sha256"])
    data_snapshot = hash_json(source_hashes)
    universe = list(config.get("population", []))
    if not universe:
        v1_contract = json.loads(
            (WORK / "contracts/20260713-dynamic-loop-context-edge-v1.json").read_text()
        )
        universe = list(v1_contract["population"]["symbols"])
    return {
        "run_id": run_id,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "git_sha": _git_value("rev-parse", "HEAD"),
        "repository_branch": _git_value("branch", "--show-current"),
        "data_snapshot_identifier": data_snapshot,
        "source_hashes": source_hashes,
        "universe_snapshot_identifier": hash_json(universe),
        "symbols": universe,
        "configuration_hash": configuration_hash,
        "configuration_path": str(CONFIG_PATH),
        "model_version": config["model_version"],
        "cost_model_version": config["source"]["cost_model_version"],
        "feature_schema_version": config["source"]["feature_schema_version"],
        "fixed_horizon_bars": config["registered_target"]["fixed_horizon_bars"],
        "change_point_hazard_prior": config["change_point"],
        "observation_model_settings": config["observation_model"],
        "thresholds": config["thresholds"],
        "minimum_support_requirements": config["support"],
        "random_seed": config["evaluation"]["fixed_random_seed"],
        "training_and_evaluation_dates": config["evaluation"]["period_date_ranges"],
        "decision_timestamp_convention": config["decision_clock"],
        "generated_artifact_paths": [str(output / name) for name in required_artifact_names()]
        + [str(report)],
        "command": command,
    }


def artifact_manifest(output: Path, report: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    files.append(
        {
            "name": str(report),
            "bytes": report.stat().st_size,
            "sha256": sha256(report),
        }
    )
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")

    config = load_config()
    configuration_hash = sha256(CONFIG_PATH)
    ledger, sessions_by_period, primary_states, pre_score = load_recovered_v1_analysis(config)
    calendars = build_session_calendars(sessions_by_period)
    surface = build_trade_surface(ledger, config)
    primary_panel, median_panel = aggregate_payoff_panels(surface, config)
    cell_keys_by_period = {
        period: sorted(
            {
                (str(row.loop_id), str(row.orientation), int(row.horizon))
                for row in surface.loc[surface["period"].eq(period)].itertuples(index=False)
            }
        )
        for period in config["evaluation"]["periods"]
    }
    required_features = tuple(
        name
        for name in config["features"]["leading_feature_logit_weights"]
        if name != "out_of_distribution_score"
    )
    feature_panel = build_feature_panel(
        surface,
        primary_panel,
        calendars,
        cell_keys_by_period,
        required_features,
    )
    data_snapshot_identifier = hash_json(pre_score["sha256"])
    run_id = (
        f"20260714-dynamic-loop-edge-state-v2-"
        f"{configuration_hash[:8]}-{data_snapshot_identifier[:8]}"
    )

    payoff_only = run_change_point_model(
        model_name="payoff_only_change_point",
        config=config,
        configuration_hash=configuration_hash,
        run_id=run_id,
        calendars=calendars,
        payoff_panel=primary_panel,
        feature_panel=feature_panel,
        cell_keys_by_period=cell_keys_by_period,
        payoff_only=True,
    )
    hierarchical = run_change_point_model(
        model_name="hierarchical_change_point",
        config=config,
        configuration_hash=configuration_hash,
        run_id=run_id,
        calendars=calendars,
        payoff_panel=primary_panel,
        feature_panel=feature_panel,
        cell_keys_by_period=cell_keys_by_period,
        payoff_only=False,
    )
    ewma = run_ewma_model(
        config=config,
        configuration_hash=configuration_hash,
        run_id=run_id,
        calendars=calendars,
        payoff_panel=primary_panel,
        cell_keys_by_period=cell_keys_by_period,
    )
    v1_forecasts = build_v1_forecasts(primary_states, calendars, config, configuration_hash, run_id)
    forecasts = pd.concat(
        [v1_forecasts, ewma, payoff_only, hierarchical], ignore_index=True, sort=False
    )
    forecasts["source_data_snapshot_id"] = data_snapshot_identifier
    decisions = build_trade_decisions(
        surface, forecasts, calendars, config, configuration_hash, run_id
    )
    prediction_metrics, calibration, scored_targets = evaluate_prediction_models(
        forecasts, primary_panel, config
    )
    comparison = model_comparison_metrics(prediction_metrics, decisions, config)
    episodes, hindsight_states = identify_hindsight_episodes(
        primary_panel, forecasts, feature_panel, calendars, config
    )
    changes = change_point_diagnostics(forecasts, episodes, hindsight_states, config)
    comparison = comparison.merge(
        changes.loc[:, ["model_name", "detection_lag_ratio"]],
        on="model_name",
        how="left",
    )
    decisions = add_state_change_phase(
        decisions,
        forecasts,
        float(config["change_point"]["change_reset_probability"]),
    )
    slices = trading_slices(decisions)
    concentration = concentration_analysis(slices, episodes)

    median_feature_panel = build_feature_panel(
        surface,
        median_panel,
        calendars,
        cell_keys_by_period,
        required_features,
    )
    median_forecast = run_change_point_model(
        model_name="hierarchical_change_point_median",
        config=config,
        configuration_hash=configuration_hash,
        run_id=run_id,
        calendars=calendars,
        payoff_panel=median_panel,
        feature_panel=median_feature_panel,
        cell_keys_by_period=cell_keys_by_period,
        payoff_only=False,
    )
    sensitivity_decisions: dict[str, pd.DataFrame] = {
        "alternative_robust_session_aggregation_median": _replace_policy(
            decisions,
            median_forecast,
            label="hierarchical_change_point_median",
        )
    }
    for hazard in config["change_point"]["predeclared_hazard_sensitivities"]:
        hazard_forecast = run_change_point_model(
            model_name=f"hierarchical_change_point_hazard_{float(hazard):.6f}",
            config=config,
            configuration_hash=configuration_hash,
            run_id=run_id,
            calendars=calendars,
            payoff_panel=primary_panel,
            feature_panel=feature_panel,
            cell_keys_by_period=cell_keys_by_period,
            payoff_only=False,
            hazard=float(hazard),
        )
        sensitivity_decisions[f"change_point_hazard_{float(hazard):.6f}"] = _replace_policy(
            decisions,
            hazard_forecast,
            label=f"hierarchical_change_point_hazard_{float(hazard):.6f}",
        )
    stresses = stress_test_results(
        decisions,
        forecasts,
        episodes,
        config,
        sensitivity_decisions,
    )

    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    metadata = build_run_metadata(
        config=config,
        configuration_hash=configuration_hash,
        pre_score=pre_score,
        run_id=run_id,
        output=output,
        report=report_path,
        command=command,
    )
    output.mkdir(parents=True)
    primary_panel = primary_panel.copy()
    primary_panel["run_id"] = run_id
    primary_panel["configuration_hash"] = configuration_hash
    primary_panel["model_version"] = config["model_version"]
    primary_panel.to_parquet(output / "session_payoff_panel.parquet", index=False)
    forecasts.to_parquet(output / "causal_edge_state_forecasts.parquet", index=False)
    decisions.to_parquet(output / "trade_decisions.parquet", index=False)
    comparison.to_csv(output / "model_comparison_metrics.csv", index=False)
    calibration.to_csv(output / "calibration_results.csv", index=False)
    changes.to_csv(output / "change_point_diagnostics.csv", index=False)
    episodes.to_parquet(output / "hindsight_episode_diagnostics.parquet", index=False)
    stresses.to_csv(output / "stress_test_results.csv", index=False)
    feature_panel.to_parquet(output / "causal_feature_panel.parquet", index=False)
    scored_targets.to_parquet(output / "prequential_scored_targets.parquet", index=False)
    hindsight_states.to_parquet(output / "hindsight_episode_states.parquet", index=False)
    slices.to_csv(output / "trading_slices.csv", index=False)
    concentration.to_csv(output / "concentration_analysis.csv", index=False)
    plot_names = representative_episode_plots(
        episodes,
        hindsight_states,
        forecasts,
        decisions,
        output,
        int(config["evaluation"]["representative_episode_plot_count"]),
    )
    metadata["representative_episode_plots"] = [str(output / name) for name in plot_names]
    status = write_report(
        report_path,
        config=config,
        comparison=comparison,
        calibration=calibration,
        changes=changes,
        stresses=stresses,
        episodes=episodes,
        decisions=decisions,
        concentration=concentration,
        metadata=metadata,
    )
    metadata["hypothesis_assessment"] = status
    write_json(output / "run_metadata.json", metadata)
    write_json(output / "artifact_manifest.json", artifact_manifest(output, report_path))


if __name__ == "__main__":
    main()
