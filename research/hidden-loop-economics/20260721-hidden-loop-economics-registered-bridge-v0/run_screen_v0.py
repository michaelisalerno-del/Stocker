#!/usr/bin/env python3
"""Run the Frozen Hidden-Loop Economics and Registered-Loop Bridge screen V0."""

from __future__ import annotations

# ruff: noqa: E402 -- numerical thread limits must be fixed before imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-hidden-loop-economics-mpl")

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, trim_mean

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.hidden_loop_economics_registered_bridge_v0 import (
    FROZEN_FAMILIES,
    OTHER_FAMILY,
    benjamini_hochberg,
    binary_model_metrics,
    bridge_feature_sets,
    choose_primary_decision,
    cohort_relative_signed_return_bps,
    completion_momentum_direction,
    deduplicate_hidden_events,
    eligible_matched_controls,
    expanding_logistic_crossfit,
    fit_weighted_logistic,
    net_after_friction_bps,
    opening_pressure_direction,
    opposite_opening_pressure_direction,
    permute_feature_within_slates,
    reconstruct_serialised_probability,
    registered_completion_targets,
    registered_loop_bridge_target,
    reject_protected_dates,
    score_event_horizons,
    session_block_bootstrap_indices,
    stock_clock_session_permutation,
)
from stocker_research.loop_prefix_automaton_v2 import FirstNextLoopEventEngine

CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
PREDECESSOR_DIR = (
    REPO_ROOT
    / "research"
    / "unregistered-loop-families"
    / "20260721-opening-trajectory-unregistered-families-v0"
)
PREDECESSOR_PRIMARY = PREDECESSOR_DIR / "artifacts" / "primary"
PATH_LEDGER_PATH = PREDECESSOR_PRIMARY / "unregistered_path_ledger.parquet"
FAMILY_MAPPING_PATH = PREDECESSOR_PRIMARY / "hidden_family_mapping.json"
PREDECESSOR_COEFFICIENTS_PATH = PREDECESSOR_PRIMARY / "model_coefficients.json"
PREDECESSOR_CONFIGURATIONS_PATH = PREDECESSOR_PRIMARY / "model_configurations.json"
PREDECESSOR_PREDICTIONS_PATH = PREDECESSOR_PRIMARY / "assessment_predictions.parquet"
PREDECESSOR_RUNNER_PATH = PREDECESSOR_DIR / "run_screen_v0.py"
V2_RUNNER_PATH = (
    REPO_ROOT
    / "research"
    / "loop-funnel"
    / "20260721-emotion-regime-coarse-loop-family-v0"
    / "run_screen_v0.py"
)

DEVELOPMENT_START = pd.Timestamp("2024-01-01T00:00:00Z")
ASSESSMENT_START = pd.Timestamp("2025-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
DECISION_SYMBOLS = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)
SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "frozen_hidden_families": True,
    "post_completion_economic_diagnostic": True,
    "registered_loop_bridge_test": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "prospective_validation": False,
}
MODEL_SEED = 20260721
ECONOMIC_BOOTSTRAP_SEED = 20260721
LEAD_NULL_SEED = 20260722
BRIDGE_BOOTSTRAP_SEED = 20260723
BRIDGE_NULL_SEED = 20260724
ECONOMIC_BOOTSTRAP_DRAWS = 50
LEAD_NULL_DRAWS = 50
BRIDGE_BOOTSTRAP_DRAWS = 50
BRIDGE_NULL_DRAWS = 10


class ScreenBlocker(RuntimeError):
    """Fail-closed quick-screen blocker with an allowed decision code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].astype(str).sort_values(list(columns), kind="mergesort")
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ScreenBlocker("blocked_reproducibility_or_audit_failure", f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                f"contract safety flag differs: {key}",
            )
    if tuple(contract["frozen_hidden_family_identities"]) != FROZEN_FAMILIES:
        raise ScreenBlocker(
            "blocked_hidden_event_population_not_reconstructable",
            "contract hidden-family identities differ from the frozen helper",
        )
    return contract


def predecessor_model_specification(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise the predecessor FittedBinary serialisation."""

    return {
        "feature_names": list(raw["features"]),
        "scaler_mean": list(raw["scaler_mean"]),
        "scaler_scale": list(raw["scaler_scale"]),
        "coefficient": list(raw["coefficient"][0]),
        "intercept": float(raw["intercept"][0]),
    }


def reconstruct_frozen_population(
    predecessor: ModuleType,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    opening, opening_manifest, feature_manifest = predecessor.load_opening_panel()
    family_mapping = read_json(FAMILY_MAPPING_PATH)
    if tuple(family_mapping["selected_families"]) != FROZEN_FAMILIES:
        raise ScreenBlocker(
            "blocked_hidden_event_population_not_reconstructable",
            "predecessor hidden-family mapping differs",
        )
    raw_events = pd.read_parquet(PATH_LEDGER_PATH)
    raw_events["event_timestamp_utc"] = pd.to_datetime(
        raw_events["event_timestamp_utc"], utc=True, errors="raise"
    )
    raw_events["decision_timestamp_utc"] = pd.to_datetime(
        raw_events["decision_timestamp_utc"], utc=True, errors="raise"
    )
    events = deduplicate_hidden_events(raw_events)
    identity = ["symbol", "session", "event_timestamp_utc", "family_id"]
    if len(events) != 4317 or frame_hash(events, identity) != frame_hash(raw_events, identity):
        raise ScreenBlocker(
            "blocked_hidden_event_population_not_reconstructable",
            "predecessor event identities do not reconstruct exactly",
        )
    configurations = read_json(PREDECESSOR_CONFIGURATIONS_PATH)
    coefficients = read_json(PREDECESSOR_COEFFICIENTS_PATH)["primary_models"]
    t0_features = tuple(str(value) for value in configurations["T0_features"])
    t1_features = tuple(str(value) for value in configurations["T1_features"])
    for model_name in ("U0", "U1"):
        model_specification = predecessor_model_specification(coefficients[model_name])
        opening[f"{model_name}_probability"] = reconstruct_serialised_probability(
            opening, model_specification
        )
    archived_predictions = pd.read_parquet(PREDECESSOR_PREDICTIONS_PATH)
    comparison = archived_predictions.merge(
        opening.loc[
            opening["year"].eq(2025),
            ["symbol", "session", "decision_ordinal", "U0_probability", "U1_probability"],
        ],
        on=["symbol", "session", "decision_ordinal"],
        how="left",
        suffixes=("_archived", "_reconstructed"),
        validate="one_to_one",
    )
    differences = []
    for name in ("U0", "U1"):
        differences.append(
            float(
                np.max(
                    np.abs(
                        comparison[f"{name}_probability_archived"].to_numpy(dtype=float)
                        - comparison[f"{name}_probability_reconstructed"].to_numpy(dtype=float)
                    )
                )
            )
        )
    maximum_probability_difference = max(differences)
    if maximum_probability_difference > 1e-12:
        raise ScreenBlocker(
            "blocked_hidden_event_population_not_reconstructable",
            f"frozen U0/U1 probability difference={maximum_probability_difference}",
        )
    join_columns = [
        "symbol",
        "session",
        "decision_ordinal",
        "repo_bar_start_ordinal",
        "feature_available_timestamp_utc",
        "signed_pressure",
        "transition_probability",
        "row_weight",
        "unregistered_event",
        "U0_probability",
        "U1_probability",
        *t1_features,
    ]
    join_columns = list(dict.fromkeys(join_columns))
    events = events.merge(
        opening.loc[:, join_columns],
        on=["symbol", "session", "decision_ordinal"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_opening"),
    )
    if events.loc[:, ["signed_pressure", "U0_probability", "U1_probability"]].isna().any().any():
        raise ScreenBlocker(
            "blocked_hidden_event_population_not_reconstructable",
            "event source values are unavailable",
        )
    events["hidden_family_class"] = np.where(
        events["family_id"].isin(FROZEN_FAMILIES), events["family_id"], OTHER_FAMILY
    )
    events["period"] = np.where(events["year"].eq(2024), "development", "assessment")
    events["event_id"] = (
        events["symbol"].astype(str)
        + "|"
        + events["session"].astype(str)
        + "|"
        + events["event_timestamp_utc"].astype(str)
        + "|"
        + events["family_id"].astype(str)
    )
    reconstruction = {
        **SAFETY_FLAGS,
        "predecessor_path_rows": len(raw_events),
        "unique_event_rows": len(events),
        "deduplicated_rows_removed": len(raw_events) - len(events),
        "identity_columns": identity,
        "event_identity_sha256": frame_hash(events, identity),
        "maximum_shared_field_difference": max(
            float(feature_manifest["maximum_feature_difference"]),
            maximum_probability_difference,
        ),
        "tolerance": 1e-12,
        "development_events": int(events["period"].eq("development").sum()),
        "assessment_events": int(events["period"].eq("assessment").sum()),
        "frozen_family_development_events": int(
            (events["period"].eq("development") & events["family_id"].isin(FROZEN_FAMILIES)).sum()
        ),
        "frozen_family_assessment_events": int(
            (events["period"].eq("assessment") & events["family_id"].isin(FROZEN_FAMILIES)).sum()
        ),
        "opening_population_manifest": opening_manifest,
        "passed": True,
    }
    return opening, events, reconstruction, t0_features, t1_features


def load_market_bars(provider_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    materialised_rows: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    source_files: list[str] = []
    for symbol in DECISION_SYMBOLS:
        source_path = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        if not source_path.is_file():
            raise ScreenBlocker(
                "blocked_economic_outcome_construction", f"missing 5m source for {symbol}"
            )
        frame = pd.read_parquet(
            source_path,
            columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"],
            filters=[
                ("timestamp", ">=", DEVELOPMENT_START.to_pydatetime()),
                ("timestamp", "<", PROTECTED_START.to_pydatetime()),
            ],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        if frame["timestamp"].ge(PROTECTED_START).any():
            raise ScreenBlocker(
                "blocked_protected_boundary_failure", "protected market row materialised"
            )
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        materialised = frame.loc[:, ["symbol", "timestamp"]].copy()
        materialised["year_month"] = local.dt.strftime("%Y-%m")
        materialised_rows.append(materialised)
        minutes = local.dt.hour * 60 + local.dt.minute
        frame = frame.loc[minutes.ge(570) & minutes.lt(960)].copy()
        frame["session"] = local.loc[frame.index].dt.strftime("%Y-%m-%d")
        frame["year_month"] = frame["session"].str[:7]
        numeric = frame.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=float)
        valid_ohlc = (
            np.isfinite(numeric).all(axis=1)
            & (numeric > 0.0).all(axis=1)
            & (numeric[:, 1] >= np.maximum(numeric[:, 0], numeric[:, 3]))
            & (numeric[:, 2] <= np.minimum(numeric[:, 0], numeric[:, 3]))
        )
        frame["qa_valid"] = valid_ohlc
        if frame.duplicated(["symbol", "timestamp"]).any():
            raise ScreenBlocker(
                "blocked_economic_outcome_construction", f"duplicate market bars for {symbol}"
            )
        source_hashes[symbol] = sha256_file(source_path)
        source_files.append(str(source_path))
        rows.append(frame)
    bars = pd.concat(rows, ignore_index=True).sort_values(["symbol", "timestamp"], kind="mergesort")
    materialised = pd.concat(materialised_rows, ignore_index=True).sort_values(
        ["symbol", "timestamp"], kind="mergesort"
    )
    reject_protected_dates(bars, column="timestamp")
    reject_protected_dates(materialised, column="timestamp")
    materialised_row_counts = (
        materialised.groupby(["symbol", "year_month"], sort=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    regular_row_counts = (
        bars.groupby(["symbol", "year_month"], sort=True).size().rename("rows").reset_index()
    )
    source_manifest = {
        **SAFETY_FLAGS,
        "provider": "EODHD",
        "timeframe": "5m",
        "symbols": list(DECISION_SYMBOLS),
        "minimum_timestamp_read": str(materialised["timestamp"].min()),
        "maximum_timestamp_read": str(materialised["timestamp"].max()),
        "rows_by_symbol_and_month": materialised_row_counts.to_dict(orient="records"),
        "materialised_rows": int(len(materialised)),
        "regular_session_minimum_timestamp": str(bars["timestamp"].min()),
        "regular_session_maximum_timestamp": str(bars["timestamp"].max()),
        "regular_session_rows_by_symbol_and_month": regular_row_counts.to_dict(orient="records"),
        "regular_session_rows": int(len(bars)),
        "source_hashes": source_hashes,
        "source_files_touched": source_files,
        "protected_files_touched": [],
        "date_predicate_applied_before_materialisation": True,
        "protected_rows_materialised": 0,
    }
    return bars.reset_index(drop=True), source_manifest


def build_registered_completions(
    v2_runner: ModuleType, provider_root: Path
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    preprocessing, parameters = v2_runner.load_frozen_model()
    v2_runner.MAX_TARGET_BAR_ORDINAL = 30
    states, source_context = v2_runner.build_v2_state_panel(
        provider_root, preprocessing, parameters
    )
    dictionary, dictionary_manifest = v2_runner.load_loop_dictionary()
    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    rows: list[dict[str, Any]] = []
    for (symbol, session), group in states.groupby(["symbol", "session"], sort=True):
        ordered = group.sort_values("bar_ordinal", kind="mergesort")
        hard = ordered["causal_hard_state"].to_numpy(dtype=int)
        event_rows = ordered.loc[np.concatenate(([True], hard[1:] != hard[:-1]))]
        trace = engine.scan_state_events(
            event_rows["causal_hard_state"].astype(int).tolist(),
            bar_ordinals=event_rows["bar_ordinal"].astype(int).tolist(),
            event_timestamps=[
                value.to_pydatetime()
                for value in pd.to_datetime(event_rows["bar_start_timestamp"], utc=True)
            ],
            available_timestamps=[
                value.to_pydatetime()
                for value in pd.to_datetime(event_rows["bar_complete_timestamp"], utc=True)
            ],
        )
        for event in trace.registered_completions:
            motif = getattr(event.motif_type, "value", str(event.motif_type))
            rows.append(
                {
                    "symbol": str(symbol),
                    "session": str(session),
                    "completion_bar_ordinal": int(event.completion_bar_ordinal),
                    "completion_timestamp_utc": pd.Timestamp(
                        event.completion_state_event_timestamp
                    ),
                    "completion_available_timestamp_utc": pd.Timestamp(
                        event.completion_available_timestamp
                    ),
                    "semantic_loop_id": str(event.semantic_loop_id),
                    "primitive_loop_id": str(event.primitive_loop_id),
                    "orientation_id": str(event.orientation_id),
                    "motif_type": str(motif),
                    "repeat_depth": int(event.repeat_depth),
                    "nested_completion": bool(event.nested_completion),
                    "tied_completion": bool(event.tied_completion),
                }
            )
    completions = pd.DataFrame(rows).drop_duplicates(
        ["symbol", "session", "completion_bar_ordinal", "semantic_loop_id", "orientation_id"]
    )
    completions = completions.sort_values(
        ["symbol", "session", "completion_bar_ordinal", "semantic_loop_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return completions, source_context, dictionary_manifest


def _exact_return(
    lookup: pd.DataFrame,
    *,
    symbol: str,
    entry_timestamp: pd.Timestamp,
    exit_bar_start: pd.Timestamp,
) -> tuple[float, float, float] | None:
    try:
        entry = lookup.loc[(symbol, entry_timestamp)]
        exit_row = lookup.loc[(symbol, exit_bar_start)]
    except KeyError:
        return None
    if not bool(entry["qa_valid"]) or not bool(exit_row["qa_valid"]):
        return None
    entry_price = float(entry["open"])
    exit_price = float(exit_row["close"])
    if (
        not np.isfinite(entry_price)
        or not np.isfinite(exit_price)
        or min(entry_price, exit_price) <= 0
    ):
        return None
    return entry_price, exit_price, 10_000.0 * (exit_price / entry_price - 1.0)


def _close_to_close_return(
    lookup: pd.DataFrame,
    *,
    symbol: str,
    start_bar: pd.Timestamp,
    end_bar: pd.Timestamp,
) -> float | None:
    try:
        start = lookup.loc[(symbol, start_bar)]
        end = lookup.loc[(symbol, end_bar)]
    except KeyError:
        return None
    if not bool(start["qa_valid"]) or not bool(end["qa_valid"]):
        return None
    start_price = float(start["close"])
    end_price = float(end["close"])
    if min(start_price, end_price) <= 0.0 or not np.isfinite([start_price, end_price]).all():
        return None
    return 10_000.0 * (end_price / start_price - 1.0)


def build_economic_ledgers(
    events: pd.DataFrame, opening: pd.DataFrame, bars: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    lookup = bars.set_index(["symbol", "timestamp"], verify_integrity=True)
    session_bars = {
        (str(symbol), str(session)): group.sort_values("timestamp", kind="mergesort")
        for (symbol, session), group in bars.groupby(["symbol", "session"], sort=False)
    }
    all_events = events.loc[
        :,
        [
            "symbol",
            "session",
            "event_timestamp_utc",
            "event_available_timestamp_utc",
            "decision_timestamp_utc",
        ],
    ].copy()
    economic_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    excluded_zero_direction = 0
    excluded_horizon = 0
    for event in events.itertuples(index=False):
        direction = opening_pressure_direction(float(event.signed_pressure))
        if direction is None:
            excluded_zero_direction += 1
            continue
        group = session_bars.get((str(event.symbol), str(event.session)))
        if group is None:
            excluded_horizon += 1
            continue
        scored = score_event_horizons(
            group,
            completion_timestamp=pd.Timestamp(event.event_timestamp_utc),
            direction=direction,
            horizons=(6, 12),
        )
        if set(scored.get("horizon_bars", pd.Series(dtype=int)).astype(int)) != {6, 12}:
            excluded_horizon += 1
        decision_bar = pd.Timestamp(event.decision_timestamp_utc) - pd.Timedelta(minutes=5)
        completion_bar = pd.Timestamp(event.event_timestamp_utc)
        stock_momentum = _close_to_close_return(
            lookup,
            symbol=str(event.symbol),
            start_bar=decision_bar,
            end_bar=completion_bar,
        )
        other_momentum = [
            value
            for symbol in DECISION_SYMBOLS
            if symbol != str(event.symbol)
            and (
                value := _close_to_close_return(
                    lookup,
                    symbol=symbol,
                    start_bar=decision_bar,
                    end_bar=completion_bar,
                )
            )
            is not None
        ]
        momentum_direction = (
            completion_momentum_direction(stock_momentum, other_momentum)
            if stock_momentum is not None
            else None
        )
        for outcome in scored.itertuples(index=False):
            entry_timestamp = pd.Timestamp(outcome.entry_timestamp_utc)
            exit_bar_start = pd.Timestamp(outcome.exit_bar_start_timestamp_utc)
            other_returns = [
                value[2]
                for symbol in DECISION_SYMBOLS
                if symbol != str(event.symbol)
                and (
                    value := _exact_return(
                        lookup,
                        symbol=symbol,
                        entry_timestamp=entry_timestamp,
                        exit_bar_start=exit_bar_start,
                    )
                )
                is not None
            ]
            if len(other_returns) < 15:
                excluded_horizon += 1
                continue
            primary_signed = float(outcome.signed_return_bps)
            cohort_signed = cohort_relative_signed_return_bps(
                stock_raw_return_bps=float(outcome.raw_return_bps),
                other_stock_raw_returns_bps=other_returns,
                direction=direction,
            )
            opposite_signed = -primary_signed
            momentum_signed = (
                float(momentum_direction) * float(outcome.raw_return_bps)
                if momentum_direction is not None
                else math.nan
            )
            slate = opening.loc[
                opening["session"].astype(str).eq(str(event.session))
                & opening["decision_ordinal"].eq(int(event.decision_ordinal)),
                ["symbol", "signed_pressure"],
            ].copy()
            candidate_values: list[dict[str, Any]] = []
            for candidate in slate.itertuples(index=False):
                prices = _exact_return(
                    lookup,
                    symbol=str(candidate.symbol),
                    entry_timestamp=entry_timestamp,
                    exit_bar_start=exit_bar_start,
                )
                if prices is not None:
                    candidate_values.append(
                        {
                            "symbol": str(candidate.symbol),
                            "signed_pressure": float(candidate.signed_pressure),
                            "entry_price": prices[0],
                            "exit_price": prices[1],
                            "raw_return_bps": prices[2],
                        }
                    )
            candidates = pd.DataFrame(candidate_values)
            controls = eligible_matched_controls(
                candidates,
                all_events.loc[all_events["session"].astype(str).eq(str(event.session))],
                focal_symbol=str(event.symbol),
                decision_timestamp=pd.Timestamp(event.decision_timestamp_utc),
                completion_timestamp=pd.Timestamp(event.event_available_timestamp_utc),
            )
            matched = len(controls) >= 5
            matched_mean = math.nan
            if matched:
                controls["signed_return_bps"] = controls["direction"].to_numpy(
                    dtype=float
                ) * controls["raw_return_bps"].to_numpy(dtype=float)
                matched_mean = float(controls["signed_return_bps"].mean())
                for control in controls.itertuples(index=False):
                    control_rows.append(
                        {
                            "event_id": str(event.event_id),
                            "period": str(event.period),
                            "hidden_family_class": str(event.hidden_family_class),
                            "horizon_bars": int(outcome.horizon_bars),
                            "control_symbol": str(control.symbol),
                            "control_direction": int(control.direction),
                            "entry_timestamp_utc": entry_timestamp,
                            "exit_timestamp_utc": pd.Timestamp(outcome.exit_timestamp_utc),
                            "entry_price": float(control.entry_price),
                            "exit_price": float(control.exit_price),
                            "raw_return_bps": float(control.raw_return_bps),
                            "signed_return_bps": float(control.signed_return_bps),
                            "net_return_20bps": float(control.signed_return_bps) - 20.0,
                        }
                    )
            economic_rows.append(
                {
                    "event_id": str(event.event_id),
                    "symbol": str(event.symbol),
                    "session": str(event.session),
                    "event_month": str(event.year_month),
                    "period": str(event.period),
                    "source_checkpoint": int(event.decision_ordinal),
                    "decision_timestamp_utc": pd.Timestamp(event.decision_timestamp_utc),
                    "event_completion_timestamp_utc": pd.Timestamp(event.event_timestamp_utc),
                    "event_completion_available_timestamp_utc": pd.Timestamp(
                        event.event_available_timestamp_utc
                    ),
                    "family_id": str(event.family_id),
                    "hidden_family_class": str(event.hidden_family_class),
                    "source_structural_path": str(event.oriented_path),
                    "U0_probability": float(event.U0_probability),
                    "U1_probability": float(event.U1_probability),
                    "horizon_bars": int(outcome.horizon_bars),
                    "entry_timestamp_utc": entry_timestamp,
                    "entry_price": float(outcome.entry_price),
                    "exit_timestamp_utc": pd.Timestamp(outcome.exit_timestamp_utc),
                    "exit_bar_start_timestamp_utc": exit_bar_start,
                    "exit_price": float(outcome.exit_price),
                    "raw_return_bps": float(outcome.raw_return_bps),
                    "opening_pressure_direction": int(direction),
                    "completion_momentum_direction": momentum_direction,
                    "opposite_opening_pressure_direction": opposite_opening_pressure_direction(
                        direction
                    ),
                    "opening_pressure_signed_return_bps": primary_signed,
                    "opening_pressure_net_return_0bps": net_after_friction_bps(
                        primary_signed, friction_bps=0.0
                    ),
                    "opening_pressure_net_return_20bps": net_after_friction_bps(
                        primary_signed, friction_bps=20.0
                    ),
                    "opening_pressure_positive_return_0bps": primary_signed > 0.0,
                    "opening_pressure_positive_return_20bps": primary_signed - 20.0 > 0.0,
                    "cohort_relative_signed_return_bps": cohort_signed,
                    "cohort_relative_net_return_20bps": net_after_friction_bps(
                        cohort_signed, friction_bps=20.0
                    ),
                    "completion_momentum_signed_return_bps": momentum_signed,
                    "completion_momentum_net_return_20bps": (
                        momentum_signed - 20.0 if np.isfinite(momentum_signed) else math.nan
                    ),
                    "opposite_pressure_signed_return_bps": opposite_signed,
                    "opposite_pressure_net_return_20bps": opposite_signed - 20.0,
                    "matched_control_count": int(len(controls)),
                    "matched_control_eligibility_rule": (
                        "no_unregistered_completion_by_focal_completion"
                    ),
                    "matched_control_available": bool(matched),
                    "matched_control_mean_signed_return_bps": matched_mean,
                    "event_excess_vs_matched_control_bps": (
                        primary_signed - matched_mean if matched else math.nan
                    ),
                }
            )
    economics = pd.DataFrame(economic_rows).sort_values(
        ["period", "session", "event_completion_timestamp_utc", "symbol", "horizon_bars"],
        kind="mergesort",
    )
    controls = pd.DataFrame(control_rows).sort_values(
        ["event_id", "horizon_bars", "control_symbol"], kind="mergesort"
    )
    diagnostic = {
        "excluded_zero_or_unavailable_opening_pressure": excluded_zero_direction,
        "excluded_insufficient_or_invalid_horizon": excluded_horizon,
        "economic_rows": len(economics),
        "control_rows": len(controls),
    }
    return economics.reset_index(drop=True), controls.reset_index(drop=True), diagnostic


def _maximum_positive_contribution_share(frame: pd.DataFrame, column: str) -> float:
    positive = frame.loc[frame[column].gt(0.0), ["event_month", column]]
    total = float(positive[column].sum())
    if total <= 0.0:
        return math.nan
    return float(positive.groupby("event_month", sort=True)[column].sum().max() / total)


def economic_metric_row(
    frame: pd.DataFrame,
    *,
    scope: str,
    period: str,
    horizon_bars: int,
    direction_name: str,
    return_column: str,
) -> dict[str, Any]:
    values = frame[return_column].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "scope": scope,
            "period": period,
            "horizon_bars": horizon_bars,
            "direction": direction_name,
            "events": 0,
        }
    event_frame = frame.loc[frame[return_column].notna()].copy()
    standard_error = (
        float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else math.nan
    )
    matched = event_frame["event_excess_vs_matched_control_bps"].dropna()
    return {
        "scope": scope,
        "period": period,
        "horizon_bars": horizon_bars,
        "direction": direction_name,
        "events": int(event_frame["event_id"].nunique()),
        "sessions": int(event_frame["session"].nunique()),
        "stocks": int(event_frame["symbol"].nunique()),
        "mean_bps": float(np.mean(values)),
        "median_bps": float(np.median(values)),
        "trimmed_mean_10pct_bps": float(trim_mean(values, proportiontocut=0.1)),
        "positive_rate": float(np.mean(values > 0.0)),
        "standard_error_bps": standard_error,
        "raw_opening_pressure_return_after_20bps": float(
            event_frame["opening_pressure_net_return_20bps"].mean()
        ),
        "cohort_relative_return_after_20bps": float(
            event_frame["cohort_relative_net_return_20bps"].mean()
        ),
        "excess_vs_matched_control_bps": float(matched.mean()) if len(matched) else math.nan,
        "matched_control_coverage": float(event_frame["matched_control_available"].mean()),
        "maximum_stock_share": float(
            event_frame.groupby("symbol", sort=True).size().max() / len(event_frame)
        ),
        "maximum_month_share": _maximum_positive_contribution_share(
            event_frame, "opening_pressure_net_return_20bps"
        ),
    }


def economic_metric_panels(
    economics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directions = {
        "opening_pressure_direction": "opening_pressure_net_return_20bps",
        "completion_momentum_direction": "completion_momentum_net_return_20bps",
        "opposite_opening_pressure_direction": "opposite_pressure_net_return_20bps",
    }
    selected = economics["hidden_family_class"].isin(FROZEN_FAMILIES)
    main_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for period in ("development", "assessment", "all"):
        period_mask = pd.Series(True, index=economics.index)
        if period != "all":
            period_mask = economics["period"].eq(period)
        for horizon in (6, 12):
            for direction_name, return_column in directions.items():
                subset = economics.loc[
                    period_mask & selected & economics["horizon_bars"].eq(horizon)
                ]
                main_rows.append(
                    economic_metric_row(
                        subset,
                        scope="FOUR_FROZEN_FAMILIES_POOLED",
                        period=period,
                        horizon_bars=horizon,
                        direction_name=direction_name,
                        return_column=return_column,
                    )
                )
                for family in (*FROZEN_FAMILIES, OTHER_FAMILY):
                    family_subset = economics.loc[
                        period_mask
                        & economics["hidden_family_class"].eq(family)
                        & economics["horizon_bars"].eq(horizon)
                    ]
                    family_rows.append(
                        economic_metric_row(
                            family_subset,
                            scope=family,
                            period=period,
                            horizon_bars=horizon,
                            direction_name=direction_name,
                            return_column=return_column,
                        )
                    )
    assessment_primary = economics.loc[
        economics["period"].eq("assessment") & selected & economics["horizon_bars"].eq(12)
    ]
    for month, subset in assessment_primary.groupby("event_month", sort=True):
        monthly_rows.append(
            economic_metric_row(
                subset,
                scope="FOUR_FROZEN_FAMILIES_POOLED",
                period=str(month),
                horizon_bars=12,
                direction_name="opening_pressure_direction",
                return_column="opening_pressure_net_return_20bps",
            )
        )
    for period in ("development", "assessment"):
        for checkpoint in (6, 12):
            subset = economics.loc[
                economics["period"].eq(period)
                & selected
                & economics["horizon_bars"].eq(12)
                & economics["source_checkpoint"].eq(checkpoint)
            ]
            row = economic_metric_row(
                subset,
                scope="FOUR_FROZEN_FAMILIES_POOLED",
                period=period,
                horizon_bars=12,
                direction_name="opening_pressure_direction",
                return_column="opening_pressure_net_return_20bps",
            )
            row["source_checkpoint"] = checkpoint
            checkpoint_rows.append(row)
    return (
        pd.DataFrame(main_rows),
        pd.DataFrame(family_rows),
        pd.DataFrame(monthly_rows),
        pd.DataFrame(checkpoint_rows),
    )


def bootstrap_economics(economics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assessment = economics.loc[
        economics["period"].eq("assessment")
        & economics["hidden_family_class"].isin(FROZEN_FAMILIES)
        & economics["horizon_bars"].eq(12)
    ].reset_index(drop=True)
    bootstrap_indices = session_block_bootstrap_indices(
        assessment,
        draws=ECONOMIC_BOOTSTRAP_DRAWS,
        seed=ECONOMIC_BOOTSTRAP_SEED,
    )
    metrics = {
        "primary_net_return_20bps": "opening_pressure_net_return_20bps",
        "cohort_relative_net_return_20bps": "cohort_relative_net_return_20bps",
        "matched_control_excess": "event_excess_vs_matched_control_bps",
    }
    draw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    scopes: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "FOUR_FROZEN_FAMILIES_POOLED": lambda frame: pd.Series(True, index=frame.index),
        **{
            family: lambda frame, family=family: frame["hidden_family_class"].eq(family)
            for family in FROZEN_FAMILIES
        },
    }
    p_values: dict[str, float] = {}
    for scope, scope_filter in scopes.items():
        for metric, column in metrics.items():
            values: list[float] = []
            for draw, indices in enumerate(bootstrap_indices, start=1):
                sampled = assessment.iloc[indices]
                sampled = sampled.loc[scope_filter(sampled)]
                value = float(sampled[column].dropna().mean())
                values.append(value)
                draw_rows.append(
                    {
                        "row_type": "draw",
                        "scope": scope,
                        "metric": metric,
                        "draw": draw,
                        "value": value,
                    }
                )
            array = np.asarray(values, dtype=float)
            finite = array[np.isfinite(array)]
            summary: dict[str, Any] = {
                "row_type": "summary",
                "scope": scope,
                "metric": metric,
                "draw": 0,
                "value": float(np.mean(finite)),
                "draws": len(finite),
                "p_value_one_sided": float((1 + np.sum(finite <= 0.0)) / (1 + len(finite))),
            }
            for confidence in (80, 90, 95):
                alpha = (100 - confidence) / 200
                summary[f"interval_{confidence}_lower"] = float(np.quantile(finite, alpha))
                summary[f"interval_{confidence}_upper"] = float(np.quantile(finite, 1.0 - alpha))
            summary_rows.append(summary)
            if scope in FROZEN_FAMILIES and metric == "primary_net_return_20bps":
                p_values[scope] = float(summary["p_value_one_sided"])
    ordered_p = [p_values[family] for family in FROZEN_FAMILIES]
    q_values = benjamini_hochberg(ordered_p)
    multiplicity = pd.DataFrame(
        {
            "hidden_family_class": list(FROZEN_FAMILIES),
            "hypothesis": "assessment_primary_net_return_20bps_positive",
            "p_value": ordered_p,
            "q_value": q_values,
            "q_le_0_10": [value <= 0.10 for value in q_values],
        }
    )
    return pd.DataFrame([*draw_rows, *summary_rows]), multiplicity


def _bootstrap_summary(bootstrap: pd.DataFrame, *, scope: str, metric: str) -> pd.Series:
    subset = bootstrap.loc[
        bootstrap["row_type"].eq("summary")
        & bootstrap["scope"].eq(scope)
        & bootstrap["metric"].eq(metric)
    ]
    if len(subset) != 1:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "bootstrap summary is ambiguous"
        )
    return subset.iloc[0]


def evaluate_economic_gate(
    economics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    multiplicity: pd.DataFrame,
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    primary = economics.loc[
        economics["horizon_bars"].eq(12) & economics["hidden_family_class"].isin(FROZEN_FAMILIES)
    ]
    development = primary.loc[primary["period"].eq("development")]
    assessment = primary.loc[primary["period"].eq("assessment")]
    assessment_months = assessment.groupby("event_month", sort=True)[
        "opening_pressure_net_return_20bps"
    ].mean()
    checkpoints = assessment.groupby("source_checkpoint", sort=True)[
        "opening_pressure_net_return_20bps"
    ].mean()
    pooled_net = _bootstrap_summary(
        bootstrap,
        scope="FOUR_FROZEN_FAMILIES_POOLED",
        metric="primary_net_return_20bps",
    )
    pooled_relative = _bootstrap_summary(
        bootstrap,
        scope="FOUR_FROZEN_FAMILIES_POOLED",
        metric="cohort_relative_net_return_20bps",
    )
    pooled_excess = _bootstrap_summary(
        bootstrap,
        scope="FOUR_FROZEN_FAMILIES_POOLED",
        metric="matched_control_excess",
    )
    maximum_stock_share = float(assessment.groupby("symbol").size().max() / len(assessment))
    maximum_month_share = _maximum_positive_contribution_share(
        assessment, "opening_pressure_net_return_20bps"
    )
    support_checks = {
        "assessment_events_at_least_1000": int(assessment["event_id"].nunique()) >= 1000,
        "sessions_at_least_100": int(assessment["session"].nunique()) >= 100,
        "stocks_at_least_15": int(assessment["symbol"].nunique()) >= 15,
        "months_at_least_6": int(assessment["event_month"].nunique()) >= 6,
        "matched_control_coverage_at_least_80pct": float(
            assessment["matched_control_available"].mean()
        )
        >= 0.80,
    }
    gate_checks = {
        "assessment_primary_net_positive": float(
            assessment["opening_pressure_net_return_20bps"].mean()
        )
        > 0.0,
        "assessment_cohort_relative_net_positive": float(
            assessment["cohort_relative_net_return_20bps"].mean()
        )
        > 0.0,
        "primary_net_80pct_lower_nonnegative": float(pooled_net["interval_80_lower"]) >= 0.0,
        "cohort_relative_80pct_lower_nonnegative": float(pooled_relative["interval_80_lower"])
        >= 0.0,
        "matched_control_excess_positive": float(
            assessment["event_excess_vs_matched_control_bps"].mean()
        )
        > 0.0,
        "matched_control_80pct_lower_nonnegative": float(pooled_excess["interval_80_lower"]) >= 0.0,
        "development_assessment_same_sign": bool(
            np.sign(development["opening_pressure_net_return_20bps"].mean())
            == np.sign(assessment["opening_pressure_net_return_20bps"].mean())
        ),
        "at_least_five_positive_assessment_months": int((assessment_months > 0.0).sum()) >= 5,
        "neither_checkpoint_materially_adverse": bool((checkpoints >= -5.0).all()),
        "maximum_stock_share_at_most_15pct": maximum_stock_share <= 0.15,
        "maximum_month_positive_return_share_at_most_30pct": maximum_month_share <= 0.30,
    }
    family_rows: list[dict[str, Any]] = []
    q_map = multiplicity.set_index("hidden_family_class")["q_value"].to_dict()
    for family in FROZEN_FAMILIES:
        family_assessment = assessment.loc[assessment["hidden_family_class"].eq(family)]
        family_development = development.loc[development["hidden_family_class"].eq(family)]
        family_bootstrap = _bootstrap_summary(
            bootstrap, scope=family, metric="primary_net_return_20bps"
        )
        family_excess = _bootstrap_summary(bootstrap, scope=family, metric="matched_control_excess")
        family_month_share = _maximum_positive_contribution_share(
            family_assessment, "opening_pressure_net_return_20bps"
        )
        family_checkpoint = family_assessment.groupby("source_checkpoint")[
            "opening_pressure_net_return_20bps"
        ].mean()
        passed = bool(
            len(family_assessment) >= 50
            and family_assessment["opening_pressure_net_return_20bps"].mean() > 0.0
            and family_assessment["cohort_relative_net_return_20bps"].mean() > 0.0
            and float(family_bootstrap["interval_80_lower"]) >= 0.0
            and family_assessment["event_excess_vs_matched_control_bps"].mean() > 0.0
            and float(family_excess["interval_80_lower"]) >= 0.0
            and np.sign(family_development["opening_pressure_net_return_20bps"].mean())
            == np.sign(family_assessment["opening_pressure_net_return_20bps"].mean())
            and bool((family_checkpoint >= -5.0).all())
            and family_assessment.groupby("symbol").size().max() / len(family_assessment) <= 0.15
            and family_month_share <= 0.30
            and float(q_map[family]) <= 0.10
        )
        family_rows.append(
            {
                "hidden_family_class": family,
                "assessment_events": len(family_assessment),
                "rough_screen_positive": passed,
                "q_value": float(q_map[family]),
            }
        )
    support_passed = all(support_checks.values())
    status = (
        "supported"
        if support_passed and all(gate_checks.values())
        else "not_supported"
        if support_passed
        else "insufficient_support"
    )
    gate = {
        "support_checks": support_checks,
        "gate_checks": gate_checks,
        "support_passed": support_passed,
        "rough_screen_positive": support_passed and all(gate_checks.values()),
        "assessment_events": int(assessment["event_id"].nunique()),
        "development_events": int(development["event_id"].nunique()),
        "maximum_stock_share": maximum_stock_share,
        "maximum_month_positive_return_share": maximum_month_share,
    }
    return status, gate, pd.DataFrame(family_rows)


def build_hidden_registered_leads(events: pd.DataFrame, completions: pd.DataFrame) -> pd.DataFrame:
    completion_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in completions.groupby(["symbol", "session"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        candidates = completion_groups.get(
            (str(event.symbol), str(event.session)),
            pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id", "motif_type"]),
        )
        targets = registered_completion_targets(int(event.completion_bar_ordinal), candidates)
        bars_to_first = targets["bars_to_first_registered_completion"]
        first_in_twelve = bars_to_first is not None and int(bars_to_first) <= 12
        first_row: pd.Series | None = None
        first_rows = pd.DataFrame()
        if first_in_twelve:
            eligible = candidates.loc[
                candidates["completion_bar_ordinal"].gt(int(event.completion_bar_ordinal))
                & candidates["completion_bar_ordinal"].le(int(event.completion_bar_ordinal) + 12)
            ].sort_values(["completion_bar_ordinal", "semantic_loop_id"], kind="mergesort")
            if not eligible.empty:
                first_bar_ordinal = int(eligible.iloc[0]["completion_bar_ordinal"])
                first_rows = eligible.loc[
                    eligible["completion_bar_ordinal"].eq(first_bar_ordinal)
                ].drop_duplicates("semantic_loop_id", keep="first")
                first_row = first_rows.iloc[0]
        first_semantic_ids = (
            sorted(first_rows["semantic_loop_id"].astype(str).tolist())
            if not first_rows.empty
            else []
        )
        first_motif_types = (
            sorted(first_rows["motif_type"].astype(str).unique().tolist())
            if not first_rows.empty
            else []
        )
        timestamp = pd.Timestamp(event.event_timestamp_utc)
        local = timestamp.tz_convert("America/New_York")
        clock_bin_minute = (local.minute // 30) * 30
        clock_bin = f"{local.hour:02d}:{clock_bin_minute:02d}"
        rows.append(
            {
                "event_id": str(event.event_id),
                "symbol": str(event.symbol),
                "session": str(event.session),
                "period": str(event.period),
                "event_month": str(event.year_month),
                "hidden_family_class": str(event.hidden_family_class),
                "hidden_completion_timestamp_utc": timestamp,
                "completion_bar_ordinal": int(event.completion_bar_ordinal),
                "clock_bin": clock_bin,
                "registered_within_6_bars": bool(targets["registered_within_6_bars"]),
                "registered_within_12_bars": bool(targets["registered_within_12_bars"]),
                "bars_to_first_registered_completion": (
                    int(bars_to_first) if first_in_twelve else math.nan
                ),
                "first_registered_semantic_loop_id": (
                    str(first_row["semantic_loop_id"]) if first_row is not None else None
                ),
                "first_registered_primitive_loop_id": (
                    str(first_row["primitive_loop_id"]) if first_row is not None else None
                ),
                "first_registered_motif_type": (
                    str(first_row["motif_type"]) if first_row is not None else None
                ),
                "first_registered_repeat_depth": (
                    int(first_row["repeat_depth"]) if first_row is not None else None
                ),
                "first_registered_semantic_loop_ids_json": json.dumps(first_semantic_ids),
                "first_registered_motif_types_json": json.dumps(first_motif_types),
                "first_registered_tied_identity_count": len(first_semantic_ids),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["period", "session", "hidden_completion_timestamp_utc", "symbol"], kind="mergesort"
    )


def transition_tables(lead: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transitioned = lead.loc[lead["registered_within_12_bars"]].copy()
    broad_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    for event in transitioned.itertuples(index=False):
        for motif_type in json.loads(str(event.first_registered_motif_types_json)):
            broad_rows.append(
                {
                    "event_id": str(event.event_id),
                    "period": str(event.period),
                    "hidden_family_class": str(event.hidden_family_class),
                    "first_registered_motif_type": str(motif_type),
                }
            )
        for semantic_loop_id in json.loads(str(event.first_registered_semantic_loop_ids_json)):
            exact_rows.append(
                {
                    "event_id": str(event.event_id),
                    "period": str(event.period),
                    "hidden_family_class": str(event.hidden_family_class),
                    "first_registered_semantic_loop_id": str(semantic_loop_id),
                }
            )
    broad_events = pd.DataFrame(broad_rows)
    exact_events = pd.DataFrame(exact_rows)
    broad = (
        broad_events.groupby(
            ["period", "hidden_family_class", "first_registered_motif_type"],
            sort=True,
            dropna=False,
        )
        .size()
        .rename("transitions")
        .reset_index()
    )
    exact_counts = (
        exact_events.groupby(
            ["period", "hidden_family_class", "first_registered_semantic_loop_id"],
            sort=True,
            dropna=False,
        )
        .size()
        .rename("transitions")
        .reset_index()
    )
    all_pairs = exact_counts.pivot_table(
        index=["hidden_family_class", "first_registered_semantic_loop_id"],
        columns="period",
        values="transitions",
        fill_value=0,
    ).reset_index()
    if "development" not in all_pairs:
        all_pairs["development"] = 0
    if "assessment" not in all_pairs:
        all_pairs["assessment"] = 0
    analytical_rows: list[dict[str, Any]] = []
    for pair in all_pairs.itertuples(index=False):
        support_eligible = bool(
            str(pair.hidden_family_class) in FROZEN_FAMILIES
            and int(pair.development) >= 30
            and int(pair.assessment) >= 20
        )
        row: dict[str, Any] = {
            "hidden_family_class": str(pair.hidden_family_class),
            "registered_semantic_loop_id": str(pair.first_registered_semantic_loop_id),
            "development_transitions": int(pair.development),
            "assessment_transitions": int(pair.assessment),
            "support_eligible": support_eligible,
            "development_odds_ratio": math.nan,
            "development_p_value": math.nan,
            "assessment_odds_ratio": math.nan,
            "assessment_p_value": math.nan,
        }
        semantic_loop_id = str(pair.first_registered_semantic_loop_id)
        if support_eligible:
            for period in ("development", "assessment"):
                population = transitioned.loc[
                    transitioned["period"].eq(period)
                    & transitioned["hidden_family_class"].isin(FROZEN_FAMILIES)
                ]
                family_mask = population["hidden_family_class"].eq(str(pair.hidden_family_class))
                loop_mask = population["first_registered_semantic_loop_ids_json"].map(
                    lambda value, semantic_loop_id=semantic_loop_id: (
                        semantic_loop_id in json.loads(str(value))
                    )
                )
                table = [
                    [int((family_mask & loop_mask).sum()), int((family_mask & ~loop_mask).sum())],
                    [int((~family_mask & loop_mask).sum()), int((~family_mask & ~loop_mask).sum())],
                ]
                odds_ratio, p_value = fisher_exact(table, alternative="greater")
                row[f"{period}_odds_ratio"] = float(odds_ratio)
                row[f"{period}_p_value"] = float(p_value)
        analytical_rows.append(row)
    exact = pd.DataFrame(analytical_rows)
    exact["assessment_q_value"] = math.nan
    supported_mask = exact["support_eligible"]
    if supported_mask.any():
        exact.loc[supported_mask, "assessment_q_value"] = benjamini_hochberg(
            exact.loc[supported_mask, "assessment_p_value"].astype(float).tolist()
        )
    exact["same_sign_replication"] = supported_mask & (
        (exact["development_odds_ratio"] - 1.0) * (exact["assessment_odds_ratio"] - 1.0)
    ).gt(0.0)
    exact["supported_replication"] = exact["same_sign_replication"] & exact[
        "assessment_q_value"
    ].le(0.10)
    return broad, exact


def structural_lead_null(
    lead: pd.DataFrame,
    completions: pd.DataFrame,
    opening: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed = lead.loc[
        lead["period"].eq("assessment") & lead["hidden_family_class"].isin(FROZEN_FAMILIES)
    ].reset_index(drop=True)
    eligible_sessions = opening.loc[
        opening["year"].eq(2025), ["symbol", "session"]
    ].drop_duplicates()
    eligible_sessions["period"] = "assessment"
    completion_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in completions.groupby(["symbol", "session"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    for draw in range(1, LEAD_NULL_DRAWS + 1):
        permuted = stock_clock_session_permutation(
            observed,
            eligible_sessions,
            seed=LEAD_NULL_SEED + draw - 1,
        )
        targets: list[dict[str, bool | int | str | None]] = []
        motifs: list[str] = []
        for event in permuted.itertuples(index=False):
            candidates = completion_groups.get(
                (str(event.symbol), str(event.session)),
                pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id", "motif_type"]),
            )
            target = registered_completion_targets(int(event.completion_bar_ordinal), candidates)
            targets.append(target)
            if target["registered_within_12_bars"]:
                first_bar = int(event.completion_bar_ordinal) + int(
                    target["bars_to_first_registered_completion"]
                )
                motifs.extend(
                    sorted(
                        candidates.loc[
                            candidates["completion_bar_ordinal"].eq(first_bar), "motif_type"
                        ]
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                )
        within_six = np.asarray(
            [bool(target["registered_within_6_bars"]) for target in targets], dtype=bool
        )
        within_twelve = np.asarray(
            [bool(target["registered_within_12_bars"]) for target in targets], dtype=bool
        )
        bars_to = np.asarray(
            [
                float(target["bars_to_first_registered_completion"])
                if target["registered_within_12_bars"]
                else math.nan
                for target in targets
            ],
            dtype=float,
        )
        rows.extend(
            [
                {
                    "draw": draw,
                    "metric": "registered_completion_rate_6",
                    "value": within_six.mean(),
                },
                {
                    "draw": draw,
                    "metric": "registered_completion_rate_12",
                    "value": within_twelve.mean(),
                },
                {"draw": draw, "metric": "mean_bars_to_registered", "value": np.nanmean(bars_to)},
            ]
        )
        for motif in ("primitive", "repeat", "composite"):
            rows.append(
                {
                    "draw": draw,
                    "metric": f"transition_count_{motif}",
                    "value": int(sum(value == motif for value in motifs)),
                }
            )
    null = pd.DataFrame(rows)
    return null, summarize_structural_lead_null(lead, null)


def summarize_structural_lead_null(lead: pd.DataFrame, null: pd.DataFrame) -> dict[str, Any]:
    """Reconstruct the structural-lead summary from a frozen null ledger."""

    observed = lead.loc[
        lead["period"].eq("assessment") & lead["hidden_family_class"].isin(FROZEN_FAMILIES)
    ].reset_index(drop=True)
    observed_rate_six = float(observed["registered_within_6_bars"].mean())
    null_six = null.loc[null["metric"].eq("registered_completion_rate_6"), "value"].to_numpy(
        dtype=float
    )
    percentile = float(100.0 * np.mean(null_six <= observed_rate_six))
    summary = {
        "observed_assessment_events": len(observed),
        "observed_registered_completion_rate_6": observed_rate_six,
        "observed_registered_completion_rate_12": float(
            observed["registered_within_12_bars"].mean()
        ),
        "observed_mean_bars_to_registered_within_12": float(
            observed["bars_to_first_registered_completion"].mean()
        ),
        "null_draws": LEAD_NULL_DRAWS,
        "six_bar_null_90th_percentile": float(np.quantile(null_six, 0.90)),
        "observed_six_bar_percentile": percentile,
        "observed_exceeds_null_90th_percentile": observed_rate_six
        > float(np.quantile(null_six, 0.90)),
    }
    return summary


def evaluate_registered_lead_gate(
    lead: pd.DataFrame, null_summary: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    assessment = lead.loc[
        lead["period"].eq("assessment") & lead["hidden_family_class"].isin(FROZEN_FAMILIES)
    ]
    support = {
        "assessment_events_at_least_1000": len(assessment) >= 1000,
        "sessions_at_least_100": assessment["session"].nunique() >= 100,
        "stocks_at_least_15": assessment["symbol"].nunique() >= 15,
        "registered_completions_within_12_at_least_100": int(
            assessment["registered_within_12_bars"].sum()
        )
        >= 100,
    }
    support_passed = all(support.values())
    lead_passed = bool(null_summary["observed_exceeds_null_90th_percentile"])
    status = (
        "supported"
        if support_passed and lead_passed
        else "not_supported"
        if support_passed
        else "insufficient_support"
    )
    return status, {
        "support_checks": support,
        "support_passed": support_passed,
        "structural_lead_gate_passed": support_passed and lead_passed,
        **null_summary,
    }


def construct_bridge_panels(
    opening: pd.DataFrame,
    completions: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    t1_features: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    completion_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in completions.groupby(["symbol", "session"], sort=False)
    }
    bar_keys = set(zip(bars["symbol"].astype(str), bars["timestamp"], strict=True))
    panel = opening.copy()
    targets: list[float] = []
    source_available: list[bool] = []
    for row in panel.itertuples(index=False):
        origin_start = pd.Timestamp(row.bar_start_timestamp)
        required = [
            (str(row.symbol), origin_start + pd.Timedelta(minutes=5 * offset))
            for offset in range(1, 13)
        ]
        available = all(key in bar_keys for key in required)
        source_available.append(available)
        if not available:
            targets.append(math.nan)
            continue
        candidates = completion_groups.get(
            (str(row.symbol), str(row.session)),
            pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id", "motif_type"]),
        )
        targets.append(
            float(registered_loop_bridge_target(int(row.repo_bar_start_ordinal), candidates))
        )
    panel["registered_completion_within_12_bars"] = targets
    panel["bridge_target_source_available"] = source_available
    panel = panel.loc[panel["bridge_target_source_available"]].copy()
    slate_counts = panel.groupby("slate_id", sort=True).size()
    panel["bridge_row_weight"] = panel["slate_id"].map((1.0 / slate_counts).to_dict())
    panel["row_weight"] = panel["bridge_row_weight"]
    panel["actual_hidden_event_within_6_bars"] = panel["unregistered_event"].eq(1.0)

    crossfit_source = panel.loc[
        panel["year"].eq(2024)
        & panel["unregistered_event"].notna()
        & panel.loc[:, list(t1_features)].notna().all(axis=1)
    ].copy()
    crossfit_source["row_weight"] = opening.loc[crossfit_source.index, "row_weight"]
    predictions, fold_manifest = expanding_logistic_crossfit(
        crossfit_source,
        features=t1_features,
        target="unregistered_event",
        folds=4,
        warmup_fraction=0.2,
    )
    crossfit_source["oof_p_unregistered_within_6_bars"] = predictions
    first_prediction_session = str(fold_manifest["prediction_session_start"].min())
    after_warmup = crossfit_source["session"].astype(str).ge(first_prediction_session)
    coverage_after_warmup = float(
        crossfit_source.loc[after_warmup, "oof_p_unregistered_within_6_bars"].notna().mean()
    )
    development = crossfit_source.loc[
        crossfit_source["oof_p_unregistered_within_6_bars"].notna()
    ].copy()
    development["p_unregistered_within_6_bars"] = development["oof_p_unregistered_within_6_bars"]
    development["row_weight"] = development["bridge_row_weight"]
    assessment = panel.loc[panel["year"].eq(2025)].copy()
    assessment["p_unregistered_within_6_bars"] = assessment["U1_probability"]
    assessment["oof_p_unregistered_within_6_bars"] = math.nan
    if development.empty or assessment.empty:
        raise ScreenBlocker("blocked_bridge_crossfit_failure", "bridge panels are empty")
    if not (
        development["session"].astype(str).lt(assessment["session"].astype(str).min()).all()
        and fold_manifest.apply(
            lambda row: str(row["train_session_end"]) < str(row["prediction_session_start"]),
            axis=1,
        ).all()
    ):
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "bridge crossfit chronology is not causal"
        )
    target_manifest = {
        **SAFETY_FLAGS,
        "target": "registered_completion_within_12_bars",
        "strictly_after_opening_decision": True,
        "horizon_completed_bars": 12,
        "rows_before_source_availability": len(opening),
        "rows_after_source_availability": len(panel),
        "source_unavailable_rows": len(opening) - len(panel),
        "development_rows": int(panel["year"].eq(2024).sum()),
        "assessment_rows": int(panel["year"].eq(2025).sum()),
        "assessment_positive_targets": int(
            assessment["registered_completion_within_12_bars"].sum()
        ),
        "assessment_base_rate": float(assessment["registered_completion_within_12_bars"].mean()),
    }
    crossfit_manifest = {
        **SAFETY_FLAGS,
        "target": "unregistered_event_within_6_bars",
        "feature_specification": list(t1_features),
        "folds": 4,
        "warmup_fraction": 0.2,
        "fold_manifest": fold_manifest.to_dict(orient="records"),
        "source_rows": len(crossfit_source),
        "predicted_rows": len(development),
        "first_prediction_session": first_prediction_session,
        "coverage_after_warmup": coverage_after_warmup,
        "assessment_feature": "exact frozen 2024-trained predecessor U1 probability",
        "registered_target_used_to_fit_U1": False,
        "passed": coverage_after_warmup >= 0.90,
    }
    if coverage_after_warmup < 0.90:
        raise ScreenBlocker(
            "blocked_bridge_crossfit_failure", "development crossfit coverage is below 90%"
        )
    return development, assessment, target_manifest, crossfit_manifest


def bridge_metric_row(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    model: str,
    subgroup: str,
) -> dict[str, Any]:
    metrics = binary_model_metrics(
        frame["registered_completion_within_12_bars"],
        probabilities,
        frame["row_weight"],
    )
    return {
        "model": model,
        "subgroup": subgroup,
        **metrics,
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
    }


def bridge_metric_panels(
    assessment: pd.DataFrame,
    *,
    development_probability_median: float,
    transition_probability_median: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    subgroups: list[tuple[str, pd.Series]] = [
        ("pooled_assessment", pd.Series(True, index=assessment.index)),
        (
            "high_u1_probability",
            assessment["p_unregistered_within_6_bars"].ge(development_probability_median),
        ),
        (
            "low_u1_probability",
            assessment["p_unregistered_within_6_bars"].lt(development_probability_median),
        ),
        (
            "high_transition_probability",
            assessment["transition_probability"].ge(transition_probability_median),
        ),
        (
            "low_transition_probability",
            assessment["transition_probability"].lt(transition_probability_median),
        ),
        (
            "actual_hidden_event_within_6_bars",
            assessment["actual_hidden_event_within_6_bars"],
        ),
        (
            "no_hidden_event_within_6_bars",
            ~assessment["actual_hidden_event_within_6_bars"],
        ),
    ]
    for subgroup, mask in subgroups:
        subset = assessment.loc[mask]
        for model in ("B0", "B1"):
            metric_rows.append(
                bridge_metric_row(
                    subset,
                    subset[f"{model}_probability"].to_numpy(dtype=float),
                    model=model,
                    subgroup=subgroup,
                )
            )
    for month, subset in assessment.groupby("year_month", sort=True):
        for model in ("B0", "B1"):
            row = bridge_metric_row(
                subset,
                subset[f"{model}_probability"].to_numpy(dtype=float),
                model=model,
                subgroup=str(month),
            )
            row["year_month"] = str(month)
            monthly_rows.append(row)
    for checkpoint, subset in assessment.groupby("decision_ordinal", sort=True):
        for model in ("B0", "B1"):
            row = bridge_metric_row(
                subset,
                subset[f"{model}_probability"].to_numpy(dtype=float),
                model=model,
                subgroup=f"ordinal_{int(checkpoint)}",
            )
            row["decision_ordinal"] = int(checkpoint)
            checkpoint_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(monthly_rows), pd.DataFrame(checkpoint_rows)


def metric_increment(frame: pd.DataFrame) -> dict[str, float]:
    b0 = binary_model_metrics(
        frame["registered_completion_within_12_bars"],
        frame["B0_probability"],
        frame["row_weight"],
    )
    b1 = binary_model_metrics(
        frame["registered_completion_within_12_bars"],
        frame["B1_probability"],
        frame["row_weight"],
    )
    return {
        "log_loss_improvement": float(b0["log_loss"]) - float(b1["log_loss"]),
        "brier_improvement": float(b0["brier_score"]) - float(b1["brier_score"]),
        "auc_improvement": float(b1["auc"]) - float(b0["auc"]),
        "average_precision_improvement": float(b1["average_precision"])
        - float(b0["average_precision"]),
    }


def bridge_permutation_sha256(frame: pd.DataFrame) -> str:
    """Hash a permuted hidden-probability panel with its immutable row keys."""

    columns = [
        "symbol",
        "session",
        "decision_ordinal",
        "slate_id",
        "p_unregistered_within_6_bars",
    ]
    hashed = pd.util.hash_pandas_object(frame.loc[:, columns], index=False).to_numpy(
        dtype=np.uint64
    )
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def attach_bridge_null_permutation_trace(
    development: pd.DataFrame, assessment: pd.DataFrame, null: pd.DataFrame
) -> pd.DataFrame:
    """Attach reproducible permutation fingerprints without additional model refits."""

    if null["draw"].astype(int).tolist() != list(range(1, BRIDGE_NULL_DRAWS + 1)):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "bridge-null draw ledger is incomplete"
        )
    result = null.copy()
    seeds: list[int] = []
    development_hashes: list[str] = []
    assessment_hashes: list[str] = []
    for draw in result["draw"].astype(int):
        seed = BRIDGE_NULL_SEED + draw - 1
        seeds.append(seed)
        development_hashes.append(
            bridge_permutation_sha256(
                permute_feature_within_slates(
                    development,
                    feature="p_unregistered_within_6_bars",
                    seed=seed,
                )
            )
        )
        assessment_hashes.append(
            bridge_permutation_sha256(
                permute_feature_within_slates(
                    assessment,
                    feature="p_unregistered_within_6_bars",
                    seed=seed,
                )
            )
        )
    result["seed"] = seeds
    result["development_permutation_sha256"] = development_hashes
    result["assessment_permutation_sha256"] = assessment_hashes
    return result


def bridge_bootstrap(assessment: pd.DataFrame) -> pd.DataFrame:
    indices = session_block_bootstrap_indices(
        assessment,
        draws=BRIDGE_BOOTSTRAP_DRAWS,
        seed=BRIDGE_BOOTSTRAP_SEED,
    )
    rows: list[dict[str, Any]] = []
    draw_values: dict[str, list[float]] = {
        "log_loss_improvement": [],
        "brier_improvement": [],
        "auc_improvement": [],
        "average_precision_improvement": [],
    }
    for draw, positions in enumerate(indices, start=1):
        increments = metric_increment(assessment.iloc[positions])
        for metric, value in increments.items():
            draw_values[metric].append(value)
            rows.append(
                {
                    "row_type": "draw",
                    "draw": draw,
                    "metric": metric,
                    "value": value,
                }
            )
    for metric, values in draw_values.items():
        array = np.asarray(values, dtype=float)
        row: dict[str, Any] = {
            "row_type": "summary",
            "draw": 0,
            "metric": metric,
            "value": float(np.mean(array)),
            "draws": len(array),
        }
        for confidence in (80, 90, 95):
            alpha = (100 - confidence) / 200
            row[f"interval_{confidence}_lower"] = float(np.quantile(array, alpha))
            row[f"interval_{confidence}_upper"] = float(np.quantile(array, 1.0 - alpha))
        rows.append(row)
    return pd.DataFrame(rows)


def bridge_null(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    *,
    b1_features: tuple[str, ...],
    real_increment: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    for draw in range(1, BRIDGE_NULL_DRAWS + 1):
        seed = BRIDGE_NULL_SEED + draw - 1
        permuted_development = permute_feature_within_slates(
            development,
            feature="p_unregistered_within_6_bars",
            seed=seed,
        )
        permuted_assessment = permute_feature_within_slates(
            assessment,
            feature="p_unregistered_within_6_bars",
            seed=seed,
        )
        null_model = fit_weighted_logistic(
            permuted_development,
            features=b1_features,
            target="registered_completion_within_12_bars",
        )
        null_scored = assessment.copy()
        null_scored["B1_probability"] = null_model.predict_probability(permuted_assessment)
        increments = metric_increment(null_scored)
        rows.append(
            {
                "draw": draw,
                "seed": seed,
                "development_permutation_sha256": bridge_permutation_sha256(permuted_development),
                "assessment_permutation_sha256": bridge_permutation_sha256(permuted_assessment),
                **increments,
            }
        )
    null = pd.DataFrame(rows)
    exceeded = {
        metric: int((float(real_increment[metric]) > null[metric]).sum())
        for metric in real_increment
    }
    return null, exceeded


def evaluate_bridge_gate(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    checkpoint_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null_exceeded: dict[str, int],
    crossfit_manifest: dict[str, Any],
    *,
    registered_lead_status: str,
) -> tuple[str, dict[str, Any]]:
    increments = metric_increment(assessment)
    bootstrap_summary = bootstrap.loc[bootstrap["row_type"].eq("summary")].set_index("metric")
    monthly_pivot = monthly_metrics.pivot(index="year_month", columns="model", values="log_loss")
    monthly_increment = monthly_pivot["B0"] - monthly_pivot["B1"]
    checkpoint_pivot = checkpoint_metrics.pivot(
        index="decision_ordinal", columns="model", values="log_loss"
    )
    checkpoint_increment = checkpoint_pivot["B0"] - checkpoint_pivot["B1"]
    class_share = float(
        assessment["registered_completion_within_12_bars"].value_counts(normalize=True).max()
    )
    maximum_stock_share = float(assessment.groupby("symbol").size().max() / len(assessment))
    support_checks = {
        "assessment_rows_at_least_5500": len(assessment) >= 5500,
        "sessions_at_least_140": assessment["session"].nunique() >= 140,
        "stocks_at_least_15": assessment["symbol"].nunique() >= 15,
        "eight_assessment_months": assessment["year_month"].nunique() == 8,
        "positive_targets_at_least_500": int(
            assessment["registered_completion_within_12_bars"].sum()
        )
        >= 500,
        "crossfit_coverage_at_least_90pct": float(crossfit_manifest["coverage_after_warmup"])
        >= 0.90,
        "no_target_class_above_80pct": class_share <= 0.80,
    }
    gate_checks = {
        "B1_improves_log_loss": increments["log_loss_improvement"] > 0.0,
        "B1_improves_brier": increments["brier_improvement"] > 0.0,
        "B1_does_not_reduce_auc": increments["auc_improvement"] >= 0.0,
        "log_loss_90pct_lower_nonnegative": float(
            bootstrap_summary.loc["log_loss_improvement", "interval_90_lower"]
        )
        >= 0.0,
        "brier_90pct_lower_nonnegative": float(
            bootstrap_summary.loc["brier_improvement", "interval_90_lower"]
        )
        >= 0.0,
        "five_positive_assessment_months": int((monthly_increment > 0.0).sum()) >= 5,
        "neither_checkpoint_materially_adverse": bool((checkpoint_increment >= -0.001).all()),
        "real_increment_exceeds_nine_nulls": max(
            null_exceeded["log_loss_improvement"], null_exceeded["brier_improvement"]
        )
        >= 9,
        "maximum_stock_share_at_most_10pct": maximum_stock_share <= 0.10,
    }
    support_passed = all(support_checks.values())
    predictive_conditions_passed = support_passed and all(gate_checks.values())
    if not support_passed:
        status = "insufficient_support"
        label = "insufficient_support"
    elif predictive_conditions_passed and registered_lead_status == "supported":
        status = "supported"
        label = "confirmed_hidden_to_registered_sequence"
    elif predictive_conditions_passed:
        status = "descriptive_only"
        label = "calibration_increment_without_confirmed_realised_lead_census"
    else:
        status = "not_supported"
        label = "no_useful_registered_loop_bridge_increment"
    return status, {
        "support_checks": support_checks,
        "gate_checks": gate_checks,
        "support_passed": support_passed,
        "predictive_conditions_passed": predictive_conditions_passed,
        "bridge_label": label,
        "increments": increments,
        "monthly_log_loss_improvements": {
            str(index): float(value) for index, value in monthly_increment.items()
        },
        "checkpoint_log_loss_improvements": {
            str(index): float(value) for index, value in checkpoint_increment.items()
        },
        "null_draws_exceeded": null_exceeded,
        "maximum_stock_share": maximum_stock_share,
        "maximum_target_class_share": class_share,
        "development_fit_rows": len(development),
        "assessment_rows": len(assessment),
    }


def build_concentration_metrics(economics: pd.DataFrame, assessment: pd.DataFrame) -> pd.DataFrame:
    primary = economics.loc[
        economics["period"].eq("assessment")
        & economics["hidden_family_class"].isin(FROZEN_FAMILIES)
        & economics["horizon_bars"].eq(12)
    ]
    rows: list[dict[str, Any]] = []
    for symbol, group in primary.groupby("symbol", sort=True):
        rows.append(
            {
                "part": "economic",
                "dimension": "stock_event_share",
                "value_id": str(symbol),
                "share": len(group) / len(primary),
                "rows": len(group),
            }
        )
    positive_total = float(primary["opening_pressure_net_return_20bps"].clip(lower=0.0).sum())
    for month, group in primary.groupby("event_month", sort=True):
        contribution = float(group["opening_pressure_net_return_20bps"].clip(lower=0.0).sum())
        rows.append(
            {
                "part": "economic",
                "dimension": "month_positive_return_share",
                "value_id": str(month),
                "share": contribution / positive_total if positive_total > 0.0 else math.nan,
                "rows": len(group),
            }
        )
    for symbol, group in assessment.groupby("symbol", sort=True):
        rows.append(
            {
                "part": "bridge",
                "dimension": "stock_row_share",
                "value_id": str(symbol),
                "share": len(group) / len(assessment),
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def create_plots(
    economics: pd.DataFrame,
    assessment: pd.DataFrame,
    output: Path,
) -> list[str]:
    primary = economics.loc[
        economics["period"].eq("assessment")
        & economics["hidden_family_class"].isin(FROZEN_FAMILIES)
        & economics["horizon_bars"].eq(12)
    ]
    summary = primary.groupby("hidden_family_class", sort=False).agg(
        net_return=("opening_pressure_net_return_20bps", "mean"),
        matched_excess=("event_excess_vs_matched_control_bps", "mean"),
    )
    summary = summary.reindex(FROZEN_FAMILIES)
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(summary))
    axis.bar(positions - 0.18, summary["net_return"], width=0.36, label="12-bar net (20 bps)")
    axis.bar(positions + 0.18, summary["matched_excess"], width=0.36, label="matched excess")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, [value.rsplit("__", 1)[-1] for value in summary.index])
    axis.set_ylabel("Mean signed return (bps)")
    axis.set_title("Frozen hidden-family assessment economics")
    axis.legend(frameon=False)
    figure.tight_layout()
    first = output / "hidden_family_economics.png"
    figure.savefig(first, dpi=160)
    plt.close(figure)

    plotted = assessment.copy()
    plotted["probability_bin"] = pd.qcut(
        plotted["p_unregistered_within_6_bars"], q=10, labels=False, duplicates="drop"
    )
    calibration = plotted.groupby("probability_bin", sort=True).agg(
        mean_hidden_probability=("p_unregistered_within_6_bars", "mean"),
        registered_rate=("registered_completion_within_12_bars", "mean"),
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        calibration["mean_hidden_probability"],
        calibration["registered_rate"],
        marker="o",
        color="#6b3fa0",
    )
    axis.set_xlabel("Frozen U1 hidden-event probability")
    axis.set_ylabel("Registered completion within 12 bars")
    axis.set_title("Predicted hidden activity and later registered loops")
    figure.tight_layout()
    second = output / "hidden_probability_registered_completion.png"
    figure.savefig(second, dpi=160)
    plt.close(figure)
    return [str(first), str(second)]


def run_determinism_check(
    *,
    output: Path,
    events: pd.DataFrame,
    opening: pd.DataFrame,
    bars: pd.DataFrame,
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    b0_features: tuple[str, ...],
    b1_features: tuple[str, ...],
    b0_model: Any,
    b1_model: Any,
    primary_decision: str,
    economic_status: str,
    registered_lead_status: str,
    predictive_bridge_status: str,
) -> dict[str, Any]:
    frozen_economics = pd.read_parquet(output / "hidden_event_economic_ledger.parquet")
    frozen_development = pd.read_parquet(output / "bridge_development_panel.parquet")
    frozen_assessment = pd.read_parquet(output / "bridge_assessment_predictions.parquet")
    regenerated_economics, _, _ = build_economic_ledgers(events, opening, bars)
    keys = ["event_id", "horizon_bars"]
    comparison = frozen_economics.merge(
        regenerated_economics,
        on=keys,
        how="outer",
        suffixes=("_frozen", "_regenerated"),
        indicator=True,
        validate="one_to_one",
    )
    identity_match = bool(comparison["_merge"].eq("both").all())
    return_columns = [
        "entry_price",
        "exit_price",
        "opening_pressure_signed_return_bps",
        "cohort_relative_signed_return_bps",
        "event_excess_vs_matched_control_bps",
    ]
    return_differences = []
    return_missingness_match = True
    for column in return_columns:
        left = comparison[f"{column}_frozen"].to_numpy(dtype=float)
        right = comparison[f"{column}_regenerated"].to_numpy(dtype=float)
        left_missing = np.isnan(left)
        right_missing = np.isnan(right)
        return_missingness_match = return_missingness_match and bool(
            np.array_equal(left_missing, right_missing)
        )
        finite = ~(left_missing | right_missing)
        return_differences.append(
            float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        )
    refit_b0 = fit_weighted_logistic(
        frozen_development,
        features=b0_features,
        target="registered_completion_within_12_bars",
    )
    refit_b1 = fit_weighted_logistic(
        frozen_development,
        features=b1_features,
        target="registered_completion_within_12_bars",
    )
    regenerated_b0 = refit_b0.predict_probability(frozen_assessment)
    regenerated_b1 = refit_b1.predict_probability(frozen_assessment)
    probability_difference = max(
        float(
            np.max(
                np.abs(regenerated_b0 - frozen_assessment["B0_probability"].to_numpy(dtype=float))
            )
        ),
        float(
            np.max(
                np.abs(regenerated_b1 - frozen_assessment["B1_probability"].to_numpy(dtype=float))
            )
        ),
    )
    coefficient_difference = max(
        float(np.max(np.abs(refit_b0.estimator.coef_ - b0_model.estimator.coef_))),
        float(np.max(np.abs(refit_b1.estimator.coef_ - b1_model.estimator.coef_))),
        float(np.max(np.abs(refit_b0.estimator.intercept_ - b0_model.estimator.intercept_))),
        float(np.max(np.abs(refit_b1.estimator.intercept_ - b1_model.estimator.intercept_))),
    )
    economic_bootstrap = pd.read_csv(output / "economic_bootstrap_metrics.csv")
    multiplicity = pd.read_csv(output / "economic_multiplicity_results.csv")
    reproduced_economic_status, reproduced_economic_gate, _ = evaluate_economic_gate(
        regenerated_economics, economic_bootstrap, multiplicity
    )
    frozen_lead = pd.read_parquet(output / "hidden_to_registered_lead_ledger.parquet")
    frozen_lead_null = pd.read_csv(output / "structural_lead_null_metrics.csv")
    reproduced_lead_summary = summarize_structural_lead_null(frozen_lead, frozen_lead_null)
    reproduced_registered_lead_status, reproduced_registered_lead_gate = (
        evaluate_registered_lead_gate(frozen_lead, reproduced_lead_summary)
    )
    regenerated_development = frozen_development.copy()
    regenerated_assessment = frozen_assessment.copy()
    regenerated_development["B0_probability"] = refit_b0.predict_probability(
        regenerated_development
    )
    regenerated_development["B1_probability"] = refit_b1.predict_probability(
        regenerated_development
    )
    regenerated_assessment["B0_probability"] = regenerated_b0
    regenerated_assessment["B1_probability"] = regenerated_b1
    bridge_configuration = read_json(output / "bridge_model_configurations.json")
    _, regenerated_bridge_monthly, regenerated_bridge_checkpoint = bridge_metric_panels(
        regenerated_assessment,
        development_probability_median=float(
            bridge_configuration["development_probability_median"]
        ),
        transition_probability_median=float(
            bridge_configuration["development_transition_probability_median"]
        ),
    )
    bridge_bootstrap_metrics = pd.read_csv(output / "bridge_bootstrap_metrics.csv")
    bridge_null_metrics = pd.read_csv(output / "bridge_null_metrics.csv")
    regenerated_increment = metric_increment(regenerated_assessment)
    regenerated_null_exceeded = {
        metric: int((float(regenerated_increment[metric]) > bridge_null_metrics[metric]).sum())
        for metric in regenerated_increment
    }
    crossfit_manifest = read_json(output / "bridge_crossfit_manifest.json")
    reproduced_predictive_bridge_status, reproduced_predictive_bridge_gate = evaluate_bridge_gate(
        regenerated_development,
        regenerated_assessment,
        regenerated_bridge_monthly,
        regenerated_bridge_checkpoint,
        bridge_bootstrap_metrics,
        regenerated_null_exceeded,
        crossfit_manifest,
        registered_lead_status=reproduced_registered_lead_status,
    )
    original_mean = float(frozen_economics["opening_pressure_net_return_20bps"].mean())
    regenerated_mean = float(regenerated_economics["opening_pressure_net_return_20bps"].mean())
    pooled_metric_difference = abs(original_mean - regenerated_mean)
    maximum_return_difference = max(return_differences, default=0.0)
    reproduced_decision = choose_primary_decision(
        economic_status=reproduced_economic_status,
        registered_lead_status=reproduced_registered_lead_status,
        predictive_bridge_status=reproduced_predictive_bridge_status,
    )
    statuses_match = bool(
        reproduced_economic_status == economic_status
        and reproduced_registered_lead_status == registered_lead_status
        and reproduced_predictive_bridge_status == predictive_bridge_status
    )
    passed = bool(
        identity_match
        and return_missingness_match
        and probability_difference <= 1e-12
        and coefficient_difference <= 1e-12
        and maximum_return_difference <= 1e-10
        and pooled_metric_difference <= 1e-10
        and statuses_match
        and primary_decision == reproduced_decision
        and len(development) == len(frozen_development)
        and len(assessment) == len(frozen_assessment)
    )
    return {
        **SAFETY_FLAGS,
        "event_identities_match": identity_match,
        "return_missingness_match": return_missingness_match,
        "maximum_probability_difference": probability_difference,
        "maximum_model_coefficient_difference": coefficient_difference,
        "refit_model_configurations": {
            "B0": refit_b0.as_dict(),
            "B1": refit_b1.as_dict(),
        },
        "maximum_return_difference_bps": maximum_return_difference,
        "pooled_metric_difference_bps": pooled_metric_difference,
        "archived_final_decision": primary_decision,
        "final_decision_reproduced": reproduced_decision,
        "final_decision_match": primary_decision == reproduced_decision,
        "status_reconstruction_match": statuses_match,
        "reproduced_statuses": {
            "economic_status": reproduced_economic_status,
            "registered_lead_status": reproduced_registered_lead_status,
            "predictive_bridge_status": reproduced_predictive_bridge_status,
        },
        "reproduced_gates": {
            "economic_gate": reproduced_economic_gate,
            "registered_lead_gate": reproduced_registered_lead_gate,
            "predictive_bridge_gate": reproduced_predictive_bridge_gate,
        },
        "probability_tolerance": 1e-12,
        "return_tolerance_bps": 1e-10,
        "full_bootstrap_or_null_rerun": False,
        "passed": passed,
    }


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, rows: int = 20) -> str:
    if frame.empty:
        return "No rows."
    selected = frame.loc[:, list(columns)].head(rows)

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(str(column) for column in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body])


def build_report(
    *,
    decision: dict[str, Any],
    reconstruction: dict[str, Any],
    source_manifest: dict[str, Any],
    economic_metrics: pd.DataFrame,
    family_metrics: pd.DataFrame,
    monthly_economic: pd.DataFrame,
    checkpoint_economic: pd.DataFrame,
    economic_bootstrap: pd.DataFrame,
    multiplicity: pd.DataFrame,
    lead: pd.DataFrame,
    lead_gate: dict[str, Any],
    exact_pairs: pd.DataFrame,
    bridge_metrics: pd.DataFrame,
    bridge_monthly: pd.DataFrame,
    bridge_checkpoint: pd.DataFrame,
    bridge_bootstrap_metrics: pd.DataFrame,
    bridge_gate: dict[str, Any],
    determinism: dict[str, Any],
) -> str:
    pooled = economic_metrics.loc[
        economic_metrics["scope"].eq("FOUR_FROZEN_FAMILIES_POOLED")
        & economic_metrics["period"].isin(["development", "assessment"])
        & economic_metrics["horizon_bars"].eq(12)
        & economic_metrics["direction"].eq("opening_pressure_direction")
    ]
    family_primary = family_metrics.loc[
        family_metrics["period"].eq("assessment")
        & family_metrics["horizon_bars"].eq(12)
        & family_metrics["direction"].eq("opening_pressure_direction")
    ]
    lead_assessment = lead.loc[
        lead["period"].eq("assessment") & lead["hidden_family_class"].isin(FROZEN_FAMILIES)
    ]
    bridge_pooled = bridge_metrics.loc[bridge_metrics["subgroup"].eq("pooled_assessment")]
    economic_ci = economic_bootstrap.loc[
        economic_bootstrap["row_type"].eq("summary")
        & economic_bootstrap["scope"].eq("FOUR_FROZEN_FAMILIES_POOLED")
    ]
    bridge_ci = bridge_bootstrap_metrics.loc[bridge_bootstrap_metrics["row_type"].eq("summary")]
    supported_pairs = exact_pairs.loc[exact_pairs["supported_replication"]]
    pooled_table = markdown_table(
        pooled,
        [
            "period",
            "events",
            "mean_bps",
            "cohort_relative_return_after_20bps",
            "excess_vs_matched_control_bps",
            "matched_control_coverage",
        ],
    )
    family_table = markdown_table(
        family_primary,
        [
            "scope",
            "events",
            "mean_bps",
            "cohort_relative_return_after_20bps",
            "excess_vs_matched_control_bps",
            "maximum_stock_share",
            "maximum_month_share",
        ],
    )
    monthly_table = markdown_table(
        monthly_economic, ["period", "events", "mean_bps", "positive_rate"]
    )
    economic_checkpoint_table = markdown_table(
        checkpoint_economic,
        ["period", "source_checkpoint", "events", "mean_bps"],
    )
    interval_columns = [
        "metric",
        "value",
        "interval_80_lower",
        "interval_80_upper",
        "interval_90_lower",
        "interval_90_upper",
        "interval_95_lower",
        "interval_95_upper",
    ]
    economic_interval_table = markdown_table(economic_ci, interval_columns)
    multiplicity_table = markdown_table(
        multiplicity, ["hidden_family_class", "p_value", "q_value", "q_le_0_10"]
    )
    supported_pair_table = markdown_table(
        supported_pairs,
        [
            "hidden_family_class",
            "registered_semantic_loop_id",
            "development_transitions",
            "assessment_transitions",
            "assessment_q_value",
        ],
    )
    bridge_table = markdown_table(
        bridge_pooled,
        [
            "model",
            "rows",
            "base_rate",
            "log_loss",
            "brier_score",
            "auc",
            "average_precision",
            "expected_calibration_error",
        ],
    )
    bridge_month_table = markdown_table(
        bridge_monthly, ["year_month", "model", "log_loss", "brier_score", "auc"]
    )
    bridge_checkpoint_table = markdown_table(
        bridge_checkpoint,
        ["decision_ordinal", "model", "log_loss", "brier_score", "auc"],
    )
    bridge_interval_table = markdown_table(bridge_ci, interval_columns)
    null_comparisons = json.dumps(bridge_gate["null_draws_exceeded"], sort_keys=True)
    development_events = reconstruction["frozen_family_development_events"]
    assessment_events = reconstruction["frozen_family_assessment_events"]
    maximum_shared_difference = reconstruction["maximum_shared_field_difference"]
    maximum_probability_difference = determinism["maximum_probability_difference"]
    maximum_return_difference = determinism["maximum_return_difference_bps"]
    return f"""# Frozen Hidden-Loop Economics and Registered-Loop Bridge Quick Screen V0

Decision: `{decision["primary_decision"]}`

- Economic status: `{decision["economic_status"]}`
- Registered-lead status: `{decision["registered_lead_status"]}`
- Predictive-bridge status: `{decision["predictive_bridge_status"]}`
- Scope: retrospective research-only quick feasibility screen.
- Synthetic friction is not realised P&L.
- Protected rows materialised: `{source_manifest["protected_rows_materialised"]}`
- Frozen-family support: {development_events} development / {assessment_events} assessment events.

## Binding question A — post-completion economics

{pooled_table}

Family assessment results (12-bar opening-pressure direction, after 20 bps):

{family_table}

Monthly stability:

{monthly_table}

Checkpoint stability:

{economic_checkpoint_table}

Bootstrap intervals:

{economic_interval_table}

Family multiplicity:

{multiplicity_table}

## Binding question B1 — realised hidden-to-registered lead

- Assessment six-bar completion rate: {lead_assessment["registered_within_6_bars"].mean():.6f}
- Assessment twelve-bar completion rate: {lead_assessment["registered_within_12_bars"].mean():.6f}
- Six-bar structural-null percentile: {lead_gate["observed_six_bar_percentile"]:.2f}
- Observed rate exceeds null 90th percentile: {lead_gate["observed_exceeds_null_90th_percentile"]}

Supported exact transitions:

{supported_pair_table}

## Binding question B2 — predictive bridge

{bridge_table}

Monthly metrics:

{bridge_month_table}

Checkpoint metrics:

{bridge_checkpoint_table}

Fixed-prediction session-bootstrap intervals:

{bridge_interval_table}

Null draws exceeded: `{null_comparisons}`.

## Integrity

- Population reconstruction: passed; maximum difference {maximum_shared_difference:.3g}.
- Determinism: `{determinism["passed"]}`.
- Maximum probability difference: {maximum_probability_difference:.3g}.
- Maximum return difference: {maximum_return_difference:.3g} bps.
- This is not prospective validation, achieved P&L, a deployable model,
  strategy promotion, or permission to trade.
"""


def execute_screen(output: Path, *, provider_root: Path) -> dict[str, Any]:
    contract = load_contract()
    if any(
        int(contract["hard_limits"][key]) != expected
        for key, expected in (
            ("maximum_session_bootstrap_draws", 50),
            ("maximum_bridge_null_refits", 10),
            ("maximum_structural_lead_null_draws", 50),
            ("maximum_plots", 2),
        )
    ):
        raise ScreenBlocker(
            "blocked_quick_hidden_loop_screen_resource_limit", "contract hard limits differ"
        )
    output.mkdir(parents=True, exist_ok=True)
    predecessor = load_module(PREDECESSOR_RUNNER_PATH, "hidden_economics_predecessor_runner")
    v2_runner = load_module(V2_RUNNER_PATH, "hidden_economics_v2_runner")

    print("reconstructing frozen hidden-event population", flush=True)
    opening, events, reconstruction, t0_features, t1_features = reconstruct_frozen_population(
        predecessor
    )
    reject_protected_dates(opening)
    reject_protected_dates(events)
    write_json(output / "contract.json", contract)
    write_json(output / "hidden_event_population_reconstruction.json", reconstruction)

    print("loading bounded five-minute economic source", flush=True)
    bars, raw_source_manifest = load_market_bars(provider_root)
    print("reconstructing registered semantic-loop completions", flush=True)
    completions, v2_source_context, dictionary_manifest = build_registered_completions(
        v2_runner, provider_root
    )
    write_parquet(output / "registered_completion_ledger.parquet", completions)
    source_manifest = {
        **SAFETY_FLAGS,
        "minimum_timestamp_read": raw_source_manifest["minimum_timestamp_read"],
        "maximum_timestamp_read": raw_source_manifest["maximum_timestamp_read"],
        "rows_by_symbol_and_month": raw_source_manifest["rows_by_symbol_and_month"],
        "source_hashes": raw_source_manifest["source_hashes"],
        "protected_files_touched": raw_source_manifest["protected_files_touched"],
        "protected_rows_materialised": 0,
        "date_predicate_applied_before_materialisation": True,
        "raw_market_source": raw_source_manifest,
        "v2_state_source": v2_source_context,
        "semantic_dictionary": dictionary_manifest,
        "predecessor_artifacts": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in (
                PATH_LEDGER_PATH,
                FAMILY_MAPPING_PATH,
                PREDECESSOR_COEFFICIENTS_PATH,
                PREDECESSOR_CONFIGURATIONS_PATH,
                PREDECESSOR_PREDICTIONS_PATH,
            )
        },
    }
    protected_audit = {
        **SAFETY_FLAGS,
        "development_period": ["2024-01-01", "2024-12-31"],
        "assessment_period": ["2025-01-01", "2025-08-22"],
        "protected_start": "2025-08-23",
        "minimum_timestamp_read": source_manifest["minimum_timestamp_read"],
        "maximum_timestamp_read": source_manifest["maximum_timestamp_read"],
        "rows_by_symbol_and_month": source_manifest["rows_by_symbol_and_month"],
        "source_hashes": source_manifest["source_hashes"],
        "protected_files_touched": [],
        "protected_rows_materialised": 0,
        "passed": pd.Timestamp(source_manifest["maximum_timestamp_read"]) < PROTECTED_START,
    }
    if not protected_audit["passed"]:
        raise ScreenBlocker(
            "blocked_protected_boundary_failure", "source boundary cannot be proved"
        )
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "protected_boundary_audit.json", protected_audit)

    print("constructing post-completion economics and matched controls", flush=True)
    economics, controls, economic_diagnostic = build_economic_ledgers(events, opening, bars)
    write_parquet(output / "hidden_event_economic_ledger.parquet", economics)
    write_parquet(output / "matched_control_ledger.parquet", controls)
    economic_metrics, family_metrics, monthly_economic, checkpoint_economic = (
        economic_metric_panels(economics)
    )
    economic_bootstrap, multiplicity = bootstrap_economics(economics)
    economic_status, economic_gate, family_gate = evaluate_economic_gate(
        economics, economic_bootstrap, multiplicity
    )
    family_metrics = family_metrics.merge(
        family_gate,
        left_on="scope",
        right_on="hidden_family_class",
        how="left",
    )
    write_csv(output / "economic_metrics.csv", economic_metrics)
    write_csv(output / "family_economic_metrics.csv", family_metrics)
    write_csv(output / "monthly_economic_metrics.csv", monthly_economic)
    write_csv(output / "economic_checkpoint_metrics.csv", checkpoint_economic)
    write_csv(output / "economic_bootstrap_metrics.csv", economic_bootstrap)
    write_csv(output / "economic_multiplicity_results.csv", multiplicity)

    print("building realised hidden-to-registered lead census and 50-draw null", flush=True)
    lead = build_hidden_registered_leads(events, completions)
    transition_counts, exact_pairs = transition_tables(lead)
    lead_null, lead_null_summary = structural_lead_null(lead, completions, opening)
    registered_lead_status, lead_gate = evaluate_registered_lead_gate(lead, lead_null_summary)
    write_parquet(output / "hidden_to_registered_lead_ledger.parquet", lead)
    write_csv(output / "hidden_to_registered_transition_counts.csv", transition_counts)
    write_csv(output / "hidden_to_registered_exact_pairs.csv", exact_pairs)
    write_csv(output / "structural_lead_null_metrics.csv", lead_null)

    print("generating causal development U1 crossfit and fitting B0/B1", flush=True)
    development, assessment, target_manifest, crossfit_manifest = construct_bridge_panels(
        opening, completions, bars, t1_features=t1_features
    )
    b0_features, b1_features = bridge_feature_sets(t0_features)
    b0_model = fit_weighted_logistic(
        development,
        features=b0_features,
        target="registered_completion_within_12_bars",
    )
    b1_model = fit_weighted_logistic(
        development,
        features=b1_features,
        target="registered_completion_within_12_bars",
    )
    development["B0_probability"] = b0_model.predict_probability(development)
    development["B1_probability"] = b1_model.predict_probability(development)
    assessment["B0_probability"] = b0_model.predict_probability(assessment)
    assessment["B1_probability"] = b1_model.predict_probability(assessment)
    predecessor_configuration = read_json(PREDECESSOR_CONFIGURATIONS_PATH)
    development_probability_median = float(development["p_unregistered_within_6_bars"].median())
    transition_probability_median = float(
        predecessor_configuration["development_transition_probability_median"]
    )
    bridge_metrics, bridge_monthly, bridge_checkpoint = bridge_metric_panels(
        assessment,
        development_probability_median=development_probability_median,
        transition_probability_median=transition_probability_median,
    )
    bridge_bootstrap_metrics = bridge_bootstrap(assessment)
    real_increment = metric_increment(assessment)
    print("running 10 within-slate bridge-null refits", flush=True)
    bridge_null_metrics, null_exceeded = bridge_null(
        development,
        assessment,
        b1_features=b1_features,
        real_increment=real_increment,
    )
    predictive_bridge_status, bridge_gate = evaluate_bridge_gate(
        development,
        assessment,
        bridge_monthly,
        bridge_checkpoint,
        bridge_bootstrap_metrics,
        null_exceeded,
        crossfit_manifest,
        registered_lead_status=registered_lead_status,
    )
    bridge_configurations = {
        **SAFETY_FLAGS,
        "B0": {
            "features": list(b0_features),
            "feature_count": len(b0_features),
            "role": "exact predecessor T0 surface",
        },
        "B1": {
            "features": list(b1_features),
            "feature_count": len(b1_features),
            "increment": "p_unregistered_within_6_bars",
        },
        "model": contract["bridge"]["model"],
        "development_probability_median": development_probability_median,
        "development_transition_probability_median": transition_probability_median,
        "primary_model_fits": 2,
        "determinism_refits": 2,
        "bridge_null_refits": 10,
        "limited_development_crossfit_fits": 4,
    }
    bridge_coefficients = {
        **SAFETY_FLAGS,
        "B0": b0_model.as_dict(),
        "B1": b1_model.as_dict(),
    }
    write_json(output / "bridge_target_manifest.json", target_manifest)
    write_json(output / "bridge_crossfit_manifest.json", crossfit_manifest)
    write_json(output / "bridge_model_configurations.json", bridge_configurations)
    write_json(output / "bridge_model_coefficients.json", bridge_coefficients)
    write_parquet(output / "bridge_development_panel.parquet", development)
    write_parquet(output / "bridge_assessment_predictions.parquet", assessment)
    write_csv(output / "bridge_metrics.csv", bridge_metrics)
    write_csv(output / "bridge_monthly_metrics.csv", bridge_monthly)
    write_csv(output / "bridge_checkpoint_metrics.csv", bridge_checkpoint)
    write_csv(output / "bridge_bootstrap_metrics.csv", bridge_bootstrap_metrics)
    write_csv(output / "bridge_null_metrics.csv", bridge_null_metrics)

    concentration = build_concentration_metrics(economics, assessment)
    write_csv(output / "concentration_metrics.csv", concentration)
    plot_paths = create_plots(economics, assessment, output)

    primary_decision = choose_primary_decision(
        economic_status=economic_status,
        registered_lead_status=registered_lead_status,
        predictive_bridge_status=predictive_bridge_status,
    )
    decision = {
        **SAFETY_FLAGS,
        "primary_decision": primary_decision,
        "economic_status": economic_status,
        "registered_lead_status": registered_lead_status,
        "predictive_bridge_status": predictive_bridge_status,
        "economic_gate": economic_gate,
        "registered_lead_gate": lead_gate,
        "predictive_bridge_gate": bridge_gate,
        "economic_construction_diagnostic": economic_diagnostic,
        "family_gate_results": family_gate.to_dict(orient="records"),
        "plots": plot_paths,
        "retrospective_only": True,
        "achieved_pnl": False,
        "permission_to_trade": False,
    }
    write_json(output / "decision.json", decision)

    print("performing fast deterministic panel/model reconstruction", flush=True)
    determinism = run_determinism_check(
        output=output,
        events=events,
        opening=opening,
        bars=bars,
        development=development,
        assessment=assessment,
        b0_features=b0_features,
        b1_features=b1_features,
        b0_model=b0_model,
        b1_model=b1_model,
        primary_decision=primary_decision,
        economic_status=economic_status,
        registered_lead_status=registered_lead_status,
        predictive_bridge_status=predictive_bridge_status,
    )
    write_json(output / "determinism_check.json", determinism)
    if not determinism["passed"]:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "fast determinism check failed"
        )

    report = build_report(
        decision=decision,
        reconstruction=reconstruction,
        source_manifest=source_manifest,
        economic_metrics=economic_metrics,
        family_metrics=family_metrics,
        monthly_economic=monthly_economic,
        checkpoint_economic=checkpoint_economic,
        economic_bootstrap=economic_bootstrap,
        multiplicity=multiplicity,
        lead=lead,
        lead_gate=lead_gate,
        exact_pairs=exact_pairs,
        bridge_metrics=bridge_metrics,
        bridge_monthly=bridge_monthly,
        bridge_checkpoint=bridge_checkpoint,
        bridge_bootstrap_metrics=bridge_bootstrap_metrics,
        bridge_gate=bridge_gate,
        determinism=determinism,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")

    print("running lightweight independent audit", flush=True)
    auditor = load_module(EXPERIMENT_DIR / "audit_screen_v0.py", "hidden_economics_auditor")
    audit = auditor.run_audit(output, write=True)
    if not bool(audit.get("passed")):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "lightweight independent audit failed"
        )
    decision["lightweight_audit_passed"] = True
    decision["determinism_check_passed"] = True
    write_json(output / "decision.json", decision)
    return decision


def finalize_existing_screen(output: Path) -> dict[str, Any]:
    """Finish presentation and audit from already frozen bounded-run artifacts."""

    decision = read_json(output / "decision.json")
    reconstruction = read_json(output / "hidden_event_population_reconstruction.json")
    source_manifest = read_json(output / "source_manifest.json")
    determinism = read_json(output / "determinism_check.json")
    economic_metrics = pd.read_csv(output / "economic_metrics.csv")
    family_metrics = pd.read_csv(output / "family_economic_metrics.csv")
    monthly_economic = pd.read_csv(output / "monthly_economic_metrics.csv")
    checkpoint_economic = pd.read_csv(output / "economic_checkpoint_metrics.csv")
    economic_bootstrap = pd.read_csv(output / "economic_bootstrap_metrics.csv")
    multiplicity = pd.read_csv(output / "economic_multiplicity_results.csv")
    lead = pd.read_parquet(output / "hidden_to_registered_lead_ledger.parquet")
    transition_counts, exact_pairs = transition_tables(lead)
    write_csv(output / "hidden_to_registered_transition_counts.csv", transition_counts)
    write_csv(output / "hidden_to_registered_exact_pairs.csv", exact_pairs)
    bridge_metrics = pd.read_csv(output / "bridge_metrics.csv")
    bridge_monthly = pd.read_csv(output / "bridge_monthly_metrics.csv")
    bridge_checkpoint = pd.read_csv(output / "bridge_checkpoint_metrics.csv")
    bridge_bootstrap_metrics = pd.read_csv(output / "bridge_bootstrap_metrics.csv")
    report = build_report(
        decision=decision,
        reconstruction=reconstruction,
        source_manifest=source_manifest,
        economic_metrics=economic_metrics,
        family_metrics=family_metrics,
        monthly_economic=monthly_economic,
        checkpoint_economic=checkpoint_economic,
        economic_bootstrap=economic_bootstrap,
        multiplicity=multiplicity,
        lead=lead,
        lead_gate=decision["registered_lead_gate"],
        exact_pairs=exact_pairs,
        bridge_metrics=bridge_metrics,
        bridge_monthly=bridge_monthly,
        bridge_checkpoint=bridge_checkpoint,
        bridge_bootstrap_metrics=bridge_bootstrap_metrics,
        bridge_gate=decision["predictive_bridge_gate"],
        determinism=determinism,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    auditor = load_module(EXPERIMENT_DIR / "audit_screen_v0.py", "hidden_economics_auditor")
    audit = auditor.run_audit(output, write=True)
    if not bool(audit.get("passed")):
        raise RuntimeError("lightweight audit failed: " + ", ".join(audit.get("failed_checks", [])))
    decision["lightweight_audit_passed"] = True
    decision["determinism_check_passed"] = True
    write_json(output / "decision.json", decision)
    return decision


def refresh_derived_existing_screen(output: Path, *, provider_root: Path) -> dict[str, Any]:
    """Refresh corrected economic/lead derivatives without repeating V2 reconstruction."""

    predecessor = load_module(PREDECESSOR_RUNNER_PATH, "hidden_economics_refresh_predecessor")
    opening, events, reconstruction, t0_features, _ = reconstruct_frozen_population(predecessor)
    bars, refreshed_raw_source_manifest = load_market_bars(provider_root)
    source_manifest = read_json(output / "source_manifest.json")
    for key in (
        "minimum_timestamp_read",
        "maximum_timestamp_read",
        "rows_by_symbol_and_month",
        "source_hashes",
        "protected_files_touched",
        "protected_rows_materialised",
        "date_predicate_applied_before_materialisation",
    ):
        source_manifest[key] = refreshed_raw_source_manifest[key]
    source_manifest["raw_market_source"] = refreshed_raw_source_manifest
    write_json(output / "source_manifest.json", source_manifest)
    protected_audit = read_json(output / "protected_boundary_audit.json")
    protected_audit.update(
        {
            "minimum_timestamp_read": refreshed_raw_source_manifest["minimum_timestamp_read"],
            "maximum_timestamp_read": refreshed_raw_source_manifest["maximum_timestamp_read"],
            "rows_by_symbol_and_month": refreshed_raw_source_manifest["rows_by_symbol_and_month"],
            "source_hashes": refreshed_raw_source_manifest["source_hashes"],
            "protected_files_touched": refreshed_raw_source_manifest["protected_files_touched"],
            "protected_rows_materialised": refreshed_raw_source_manifest[
                "protected_rows_materialised"
            ],
            "passed": pd.Timestamp(refreshed_raw_source_manifest["maximum_timestamp_read"])
            < PROTECTED_START,
        }
    )
    if not bool(protected_audit["passed"]):
        raise ScreenBlocker(
            "blocked_protected_boundary_failure", "source boundary cannot be proved"
        )
    write_json(output / "protected_boundary_audit.json", protected_audit)
    existing_economics = pd.read_parquet(output / "hidden_event_economic_ledger.parquet")
    corrected_columns = {
        "event_completion_available_timestamp_utc",
        "opening_pressure_positive_return_0bps",
        "opening_pressure_positive_return_20bps",
        "matched_control_eligibility_rule",
    }
    if corrected_columns.issubset(existing_economics.columns):
        economics = existing_economics
        controls = pd.read_parquet(output / "matched_control_ledger.parquet")
        economic_metrics = pd.read_csv(output / "economic_metrics.csv")
        family_metrics = pd.read_csv(output / "family_economic_metrics.csv")
        monthly_economic = pd.read_csv(output / "monthly_economic_metrics.csv")
        checkpoint_economic = pd.read_csv(output / "economic_checkpoint_metrics.csv")
        economic_bootstrap = pd.read_csv(output / "economic_bootstrap_metrics.csv")
        multiplicity = pd.read_csv(output / "economic_multiplicity_results.csv")
        economic_status, economic_gate, family_gate = evaluate_economic_gate(
            economics, economic_bootstrap, multiplicity
        )
        economic_diagnostic = {
            "economic_rows": len(economics),
            "control_rows": len(controls),
            "excluded_zero_or_unavailable_opening_pressure": 0,
            "excluded_insufficient_or_invalid_horizon": 0,
        }
    else:
        economics, controls, economic_diagnostic = build_economic_ledgers(events, opening, bars)
        economic_metrics, family_metrics, monthly_economic, checkpoint_economic = (
            economic_metric_panels(economics)
        )
        economic_bootstrap, multiplicity = bootstrap_economics(economics)
        economic_status, economic_gate, family_gate = evaluate_economic_gate(
            economics, economic_bootstrap, multiplicity
        )
        family_metrics = family_metrics.merge(
            family_gate,
            left_on="scope",
            right_on="hidden_family_class",
            how="left",
        )
        write_parquet(output / "hidden_event_economic_ledger.parquet", economics)
        write_parquet(output / "matched_control_ledger.parquet", controls)
        write_csv(output / "economic_metrics.csv", economic_metrics)
        write_csv(output / "family_economic_metrics.csv", family_metrics)
        write_csv(output / "monthly_economic_metrics.csv", monthly_economic)
        write_csv(output / "economic_checkpoint_metrics.csv", checkpoint_economic)
        write_csv(output / "economic_bootstrap_metrics.csv", economic_bootstrap)
        write_csv(output / "economic_multiplicity_results.csv", multiplicity)

    lead = pd.read_parquet(output / "hidden_to_registered_lead_ledger.parquet")
    transition_counts, exact_pairs = transition_tables(lead)
    lead_null = pd.read_csv(output / "structural_lead_null_metrics.csv")
    lead_null_summary = summarize_structural_lead_null(lead, lead_null)
    registered_lead_status, lead_gate = evaluate_registered_lead_gate(lead, lead_null_summary)
    write_csv(output / "hidden_to_registered_transition_counts.csv", transition_counts)
    write_csv(output / "hidden_to_registered_exact_pairs.csv", exact_pairs)

    development = pd.read_parquet(output / "bridge_development_panel.parquet")
    assessment = pd.read_parquet(output / "bridge_assessment_predictions.parquet")
    bridge_monthly = pd.read_csv(output / "bridge_monthly_metrics.csv")
    bridge_checkpoint = pd.read_csv(output / "bridge_checkpoint_metrics.csv")
    bridge_bootstrap_metrics = pd.read_csv(output / "bridge_bootstrap_metrics.csv")
    bridge_null_metrics = pd.read_csv(output / "bridge_null_metrics.csv")
    bridge_null_metrics = attach_bridge_null_permutation_trace(
        development, assessment, bridge_null_metrics
    )
    write_csv(output / "bridge_null_metrics.csv", bridge_null_metrics)
    crossfit_manifest = read_json(output / "bridge_crossfit_manifest.json")
    real_increment = metric_increment(assessment)
    null_exceeded = {
        metric: int((float(real_increment[metric]) > bridge_null_metrics[metric]).sum())
        for metric in real_increment
    }
    predictive_bridge_status, bridge_gate = evaluate_bridge_gate(
        development,
        assessment,
        bridge_monthly,
        bridge_checkpoint,
        bridge_bootstrap_metrics,
        null_exceeded,
        crossfit_manifest,
        registered_lead_status=registered_lead_status,
    )
    concentration = build_concentration_metrics(economics, assessment)
    write_csv(output / "concentration_metrics.csv", concentration)
    plot_paths = create_plots(economics, assessment, output)

    primary_decision = choose_primary_decision(
        economic_status=economic_status,
        registered_lead_status=registered_lead_status,
        predictive_bridge_status=predictive_bridge_status,
    )
    decision = {
        **SAFETY_FLAGS,
        "primary_decision": primary_decision,
        "economic_status": economic_status,
        "registered_lead_status": registered_lead_status,
        "predictive_bridge_status": predictive_bridge_status,
        "economic_gate": economic_gate,
        "registered_lead_gate": lead_gate,
        "predictive_bridge_gate": bridge_gate,
        "economic_construction_diagnostic": economic_diagnostic,
        "family_gate_results": family_gate.to_dict(orient="records"),
        "plots": plot_paths,
        "retrospective_only": True,
        "achieved_pnl": False,
        "permission_to_trade": False,
    }
    write_json(output / "decision.json", decision)
    b0_features, b1_features = bridge_feature_sets(t0_features)
    b0_model = fit_weighted_logistic(
        development,
        features=b0_features,
        target="registered_completion_within_12_bars",
    )
    b1_model = fit_weighted_logistic(
        development,
        features=b1_features,
        target="registered_completion_within_12_bars",
    )
    determinism = run_determinism_check(
        output=output,
        events=events,
        opening=opening,
        bars=bars,
        development=development,
        assessment=assessment,
        b0_features=b0_features,
        b1_features=b1_features,
        b0_model=b0_model,
        b1_model=b1_model,
        primary_decision=primary_decision,
        economic_status=economic_status,
        registered_lead_status=registered_lead_status,
        predictive_bridge_status=predictive_bridge_status,
    )
    write_json(output / "determinism_check.json", determinism)
    if not determinism["passed"]:
        raise RuntimeError("corrected derivative determinism check failed")
    return finalize_existing_screen(output)


def write_blocker(output: Path, blocker: ScreenBlocker) -> None:
    output.mkdir(parents=True, exist_ok=True)
    contract = read_json(CONTRACT_PATH)
    decision = {
        **SAFETY_FLAGS,
        "primary_decision": blocker.code,
        "economic_status": "insufficient_support",
        "registered_lead_status": "insufficient_support",
        "predictive_bridge_status": "insufficient_support",
        "blocker": blocker.detail,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "decision.json", decision)
    report = (
        "# Frozen Hidden-Loop Economics and Registered-Loop Bridge Quick Screen V0\n\n"
        f"Decision: `{blocker.code}`\n\nBlocked fail-closed: {blocker.detail}\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=Path.home()
        / "StockerLocal"
        / "data"
        / "processed"
        / "source=eodhd"
        / "instrument_type=stock",
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="resume report/audit from completed frozen artifacts without rerunning calculations",
    )
    parser.add_argument(
        "--refresh-derived-existing",
        action="store_true",
        help="refresh economics/lead derivatives without repeating V2 state reconstruction",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    output = arguments.output.expanduser().resolve()
    provider_root = arguments.provider_root.expanduser().resolve()
    try:
        if arguments.refresh_derived_existing:
            decision = refresh_derived_existing_screen(output, provider_root=provider_root)
        elif arguments.finalize_existing:
            decision = finalize_existing_screen(output)
        else:
            decision = execute_screen(output, provider_root=provider_root)
        print(canonical_json(decision), end="")
        return 0
    except ScreenBlocker as blocker:
        write_blocker(output, blocker)
        print(blocker.code)
        print(blocker.detail, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
