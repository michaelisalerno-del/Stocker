#!/usr/bin/env python3
"""Run Movement-Qualified Directional Readiness Quick Screen V0."""

from __future__ import annotations

# ruff: noqa: E402 -- numerical thread limits must precede numerical imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-directional-readiness-v0-mpl")

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

warnings.filterwarnings(
    "ignore",
    message="'penalty' was deprecated.*",
    category=FutureWarning,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"

for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.movement_qualified_direction_v0 import (
    D0_FEATURES,
    D1_FEATURES,
    D2_FEATURES,
    HORIZONS,
    M1_THRESHOLD,
    aligned_returns,
    apply_selective_policy,
    assign_contiguous_session_folds,
    attach_direction_targets,
    audited_state_orientation_map,
    baseline_predictions,
    binary_direction_metrics,
    build_d0_features,
    build_route_orientation_features,
    build_signed_behavioural_features,
    construct_fresh_episodes,
    decide_direction_candidate,
    fit_direction_model,
    freeze_confidence_boundary,
    movement_gate,
    permute_labels_within_slates,
    selective_policy_metrics,
    session_bootstrap_samples,
    validate_protected_boundary,
)

V0_DIR = REPO_ROOT / "research/options-feasibility/20260723-minimal-intraday-iv-excess-holdout-v0"
V0_RUNNER = V0_DIR / "run_screen_v0.py"
V0_PRIMARY = V0_DIR / "artifacts" / "primary"
ROUTE_PRIMARY = (
    REPO_ROOT
    / "research/route-competition/20260722-route-competition-hazard-quick-v0"
    / "artifacts/primary"
)
ROUTE_LEDGER = ROUTE_PRIMARY / "route_competition_ledger.parquet"
ROUTE_AUDIT = ROUTE_PRIMARY / "lightweight_audit.json"
ORIENTATION_EXPERIMENT = (
    REPO_ROOT / "research/regime-loop-behaviour/20260721-regime-loop-behaviour-quick-screen-v0"
)
ORIENTATION_CONTRACT = ORIENTATION_EXPERIMENT / "contract.json"
ORIENTATION_AUDIT = ORIENTATION_EXPERIMENT / "artifacts/primary/independent_audit.json"
CENTROIDS = (
    REPO_ROOT
    / "research/slrno-v2/20260714-regime-loop-handoff/work/artifacts"
    / "20260719-right-censored-regime-refit-v2/primary/full_refit_cluster_centroids.csv"
)
BRANCH_C_CANDIDATES = (
    REPO_ROOT
    / "research/cross-market-context/20260723-daily-stock-front-options-context-v01"
    / "artifacts/primary/front_options_cross_market_panel.parquet",
    Path(
        "/Users/michaelsalerno/Documents/Codex/"
        "2026-07-23-you-are-working-in-the-github-3/research/cross-market-context/"
        "20260723-daily-stock-front-options-context-v01/artifacts/primary/"
        "front_options_cross_market_panel.parquet"
    ),
)
STATE_CANDIDATES = (
    REPO_ROOT / "data/cache/minimal-intraday-iv-excess-holdout-v0/frozen_state_surface.parquet",
    Path(
        "/Users/michaelsalerno/Documents/Codex/"
        "2026-07-23-you-are-working-in-the-github-5/data/cache/"
        "minimal-intraday-iv-excess-holdout-v0/frozen_state_surface.parquet"
    ),
)
BEHAVIOUR_CANDIDATES = (
    REPO_ROOT / "research/observable-behavioural-state/"
    "20260721-behavioural-state-dimensions-screen-v0/artifacts/primary/"
    "compact_decision_panel.parquet",
    Path(
        "/Users/michaelsalerno/Documents/Codex/"
        "2026-07-21-you-are-working-in-the-github-2/research/"
        "observable-behavioural-state/"
        "20260721-behavioural-state-dimensions-screen-v0/artifacts/primary/"
        "compact_decision_panel.parquet"
    ),
)

EXPECTED_BRANCH_C_SHA256 = "f62ef0144c12c813cbc665ba6d5ba1a235a6f77101a04b9f491c77b24c295529"
EXPECTED_STATE_SHA256 = "68b1cc53c1570d53054d685966eef96f533d8760368ebfc148766bb8f3a6bcc0"
DEVELOPMENT_START = "2024-01-01"
DEVELOPMENT_END = "2024-12-31"
ASSESSMENT_START = "2025-01-01"
ASSESSMENT_END = "2025-08-22"
BOOTSTRAP_DRAWS = 100
BOOTSTRAP_SEED = 20260726
NULL_SEEDS = (20260731, 20260732, 20260733, 20260734, 20260735)
MODEL_IDS = ("D0", "D1", "D2")
CATEGORICAL_CONTROLS = ("stock", "checkpoint_category", "day_of_week")
SAFETY_KEYS = (
    "research_only",
    "retrospective_directional_screen",
    "movement_model_frozen",
    "movement_model_refit_allowed",
    "m1_threshold_frozen",
    "direction_model_is_second_stage",
    "direction_models_trained_on_2024_only",
    "assessment_start",
    "assessment_end",
    "opened_movement_holdout_excluded",
    "rows_from_2026_onward_protected",
    "primary_direction_horizon_minutes",
    "call_put_abstain_policy",
    "option_pnl_calculated",
    "intraday_option_quotes_used",
    "broker_access",
    "paper_orders_allowed",
    "live_orders_allowed",
    "strategy_promotion",
    "production_runtime_modified",
)


class ScreenBlocked(RuntimeError):
    """A fail-closed research-screen blocker."""

    def __init__(self, decision: str, detail: str) -> None:
        super().__init__(detail)
        self.decision = decision
        self.detail = detail


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period, Path)):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_identity(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    hashes = pd.util.hash_pandas_object(selected, index=False).to_numpy(np.uint64)
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ScreenBlocked(
            "blocked_movement_model_reconstruction_failure",
            f"cannot load predecessor runner {path}",
        )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def first_real_parquet(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            if handle.read(4) == b"PAR1":
                return path
    raise ScreenBlocked(
        "blocked_reproducibility_or_audit_failure",
        f"no materialized frozen parquet among {[str(value) for value in candidates]}",
    )


def load_contract() -> dict[str, Any]:
    contract = cast(
        dict[str, Any],
        json.loads((EXPERIMENT_DIR / "contract.json").read_text(encoding="utf-8")),
    )
    expected: dict[str, Any] = {
        "research_only": True,
        "retrospective_directional_screen": True,
        "movement_model_frozen": True,
        "movement_model_refit_allowed": False,
        "m1_threshold_frozen": M1_THRESHOLD,
        "direction_model_is_second_stage": True,
        "direction_models_trained_on_2024_only": True,
        "assessment_start": ASSESSMENT_START,
        "assessment_end": ASSESSMENT_END,
        "opened_movement_holdout_excluded": True,
        "rows_from_2026_onward_protected": True,
        "primary_direction_horizon_minutes": 10,
        "call_put_abstain_policy": True,
        "option_pnl_calculated": False,
        "intraday_option_quotes_used": False,
        "broker_access": False,
        "paper_orders_allowed": False,
        "live_orders_allowed": False,
        "strategy_promotion": False,
        "production_runtime_modified": False,
    }
    mismatches = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            f"contract safety mismatch: {mismatches}",
        )
    return contract


def safety_flags(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {key: contract[key] for key in SAFETY_KEYS}


def load_frozen_inputs() -> dict[str, Any]:
    branch_c_path = first_real_parquet(BRANCH_C_CANDIDATES)
    state_path = first_real_parquet(STATE_CANDIDATES)
    behaviour_path = first_real_parquet(BEHAVIOUR_CANDIDATES)
    if sha256_file(branch_c_path) != EXPECTED_BRANCH_C_SHA256:
        raise ScreenBlocked(
            "blocked_movement_model_reconstruction_failure",
            "frozen Branch C panel hash drifted",
        )
    if sha256_file(state_path) != EXPECTED_STATE_SHA256:
        raise ScreenBlocked(
            "blocked_movement_model_reconstruction_failure",
            "frozen five-minute state surface hash drifted",
        )
    historical = pd.read_parquet(branch_c_path)
    validate_protected_boundary(historical["session"])
    if historical["session"].astype(str).max() != ASSESSMENT_END:
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "Branch C assessment boundary is not 2025-08-22",
        )
    state_columns = [
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "upper_wick_pct_of_range",
        "lower_wick_pct_of_range",
        "market_breadth_bar_positive",
        "vti__bar_log_return",
        "feature_available_timestamp_max",
    ]
    states = pd.read_parquet(
        state_path,
        columns=state_columns,
        filters=[
            ("session", ">=", DEVELOPMENT_START),
            ("session", "<=", ASSESSMENT_END),
        ],
    ).rename(columns={"symbol": "stock"})
    validate_protected_boundary(states["session"])
    states["bar_start_timestamp"] = pd.to_datetime(
        states["bar_start_timestamp"], utc=True, errors="raise"
    )
    states["bar_complete_timestamp"] = pd.to_datetime(
        states["bar_complete_timestamp"], utc=True, errors="raise"
    )
    states = states.sort_values(["stock", "session", "bar_ordinal"], kind="mergesort").reset_index(
        drop=True
    )
    if states.duplicated(["stock", "session", "bar_ordinal"]).any():
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "five-minute state surface has duplicate bar identities",
        )
    behaviour = pd.read_parquet(
        behaviour_path,
        columns=[
            "symbol",
            "session",
            "decision_ordinal",
            "feature_available_timestamp_utc",
            "signed_exhaustion",
        ],
        filters=[
            ("session", ">=", DEVELOPMENT_START),
            ("session", "<=", ASSESSMENT_END),
        ],
    ).rename(
        columns={
            "symbol": "stock",
            "decision_ordinal": "checkpoint",
            "feature_available_timestamp_utc": "exhaustion_available_timestamp",
        }
    )
    validate_protected_boundary(behaviour["session"])
    if not set(behaviour["checkpoint"].astype(int).unique()).issubset({6, 12}):
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "audited signed exhaustion has an unexpected checkpoint",
        )
    return {
        "branch_c_path": branch_c_path,
        "state_path": state_path,
        "behaviour_path": behaviour_path,
        "historical": historical,
        "states": states,
        "behaviour": behaviour,
    }


def reconstruct_frozen_m1(
    historical: pd.DataFrame,
) -> tuple[ModuleType, Any, pd.DataFrame, dict[str, Any]]:
    v0 = load_module(V0_RUNNER, "direction_screen_frozen_m1_runner")
    models, development, reference, reconstruction = v0.reconstruct_historical_models(historical)
    thresholds = cast(dict[str, Any], reconstruction.pop("thresholds"))
    reconstructed_threshold = float(thresholds["M1_top_5_percent_threshold"])
    predecessor_thresholds = json.loads(
        (V0_PRIMARY / "frozen_tail_thresholds.json").read_text(encoding="utf-8")
    )
    predecessor_threshold = float(predecessor_thresholds["M1_top_5_percent_threshold"])
    maximum_threshold_difference = max(
        abs(reconstructed_threshold - M1_THRESHOLD),
        abs(predecessor_threshold - M1_THRESHOLD),
    )
    if maximum_threshold_difference > 1e-15:
        raise ScreenBlocked(
            "blocked_movement_model_reconstruction_failure",
            "frozen M1 threshold failed exact reconstruction",
        )
    panel = historical.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    panel["M1_probability"] = models.m1.predict(panel)
    reference_check = reference[["row_id", "M1_probability"]].merge(
        panel[["row_id", "M1_probability"]],
        on="row_id",
        validate="one_to_one",
        suffixes=("_reconstruction", "_direct"),
    )
    maximum_probability_difference = float(
        np.max(
            np.abs(
                reference_check["M1_probability_reconstruction"].to_numpy(float)
                - reference_check["M1_probability_direct"].to_numpy(float)
            )
        )
    )
    if maximum_probability_difference > 1e-12:
        raise ScreenBlocked(
            "blocked_movement_model_reconstruction_failure",
            "frozen M1 predictions did not reproduce",
        )
    audit = {
        **reconstruction,
        "passed": True,
        "movement_model": "M1",
        "movement_model_role": "eligibility_gate_only",
        "movement_model_refit_for_direction": False,
        "development_rows": int(len(development)),
        "assessment_rows": int(len(reference)),
        "reconstructed_threshold": reconstructed_threshold,
        "predecessor_threshold": predecessor_threshold,
        "contract_threshold": M1_THRESHOLD,
        "maximum_threshold_difference": maximum_threshold_difference,
        "maximum_direct_probability_difference": maximum_probability_difference,
        "model_specification": v0.model_specification(models.m1),
    }
    return v0, models, panel, audit


def build_episode_panel(
    panel: pd.DataFrame,
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    causal = panel.rename(columns={"symbol": "stock"}).copy()
    causal["session"] = causal["session"].astype(str)
    validate_protected_boundary(causal["session"])
    causal["partition"] = np.where(
        causal["session"].le(DEVELOPMENT_END), "development", "assessment"
    )
    signal_times = states[
        [
            "stock",
            "session",
            "bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
        ]
    ].copy()
    signal_times["checkpoint"] = signal_times["bar_ordinal"].astype(int) + 1
    signal_times = signal_times.rename(columns={"bar_complete_timestamp": "signal_timestamp"})
    entry_times = states[["stock", "session", "bar_ordinal", "bar_start_timestamp"]].copy()
    entry_times["checkpoint"] = entry_times["bar_ordinal"].astype(int)
    entry_times = entry_times.rename(columns={"bar_start_timestamp": "prospective_entry_timestamp"})
    causal = causal.merge(
        signal_times[["stock", "session", "checkpoint", "signal_timestamp"]],
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    ).merge(
        entry_times[["stock", "session", "checkpoint", "prospective_entry_timestamp"]],
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    if causal[["signal_timestamp", "prospective_entry_timestamp"]].isna().any().any():
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "an M1 checkpoint lacks independently reconstructed timestamps",
        )
    source_available = pd.to_datetime(
        causal["feature_available_timestamp_utc"], utc=True, errors="raise"
    )
    if not source_available.equals(
        pd.to_datetime(causal["signal_timestamp"], utc=True, errors="raise")
    ):
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "M1 feature availability differs from the signal close",
        )
    causal = causal.rename(columns={"M1_probability": "m1_probability"})
    episode_input = causal[
        [
            "stock",
            "session",
            "checkpoint",
            "signal_timestamp",
            "prospective_entry_timestamp",
            "m1_probability",
            "partition",
        ]
    ].copy()
    episodes = construct_fresh_episodes(episode_input)
    safe_metadata = [
        column
        for column in causal.columns
        if column
        not in {
            "signal_timestamp",
            "prospective_entry_timestamp",
            "partition",
            "m1_probability",
            "M1_probability",
            "entry_price",
            "close_15m",
            "absolute_log_return_15m",
            "iv_expected_absolute_15m",
            "movement_exceeds_prior_close_iv_15m",
            "iv_absolute_residual_15m",
        }
        and not column.startswith("future_")
        and not column.startswith("target_")
    ]
    safe_metadata = list(dict.fromkeys(["stock", "session", "checkpoint", *safe_metadata]))
    episodes = episodes.merge(
        causal[safe_metadata],
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    ordered = episode_input.sort_values(["stock", "session", "checkpoint"], kind="mergesort").copy()
    ordered["above"] = movement_gate(ordered["m1_probability"].to_numpy(float))
    ordered["previous"] = ordered.groupby(["stock", "session"], sort=False)[
        "m1_probability"
    ].shift()
    ordered["crossing"] = ordered["above"] & (
        ordered["previous"].isna() | ordered["previous"].lt(M1_THRESHOLD)
    )
    audit = {
        "passed": True,
        "raw_checkpoint_rows": int(len(ordered)),
        "raw_above_threshold_checkpoint_rows": int(ordered["above"].sum()),
        "raw_above_threshold_development_rows": int(
            (ordered["above"] & ordered["partition"].eq("development")).sum()
        ),
        "raw_above_threshold_assessment_rows": int(
            (ordered["above"] & ordered["partition"].eq("assessment")).sum()
        ),
        "fresh_unspaced_crossings": int(ordered["crossing"].sum()),
        "fresh_episodes": int(len(episodes)),
        "development_episodes": int(episodes["partition"].astype(str).eq("development").sum()),
        "assessment_episodes": int(episodes["partition"].astype(str).eq("assessment").sum()),
        "sessions": int(episodes["session"].nunique()),
        "episodes_per_session": float(len(episodes) / episodes["session"].nunique()),
        "maximum_episodes_per_stock_session": int(
            episodes.groupby(["stock", "session"], sort=False).size().max()
        ),
        "minimum_episode_spacing_minutes": 30,
        "spacing_violations": int(
            episodes["minutes_since_previous_episode"].dropna().lt(30.0).sum()
        ),
        "signal_is_completed_checkpoint_bar_close": True,
        "entry_is_next_five_minute_bar_open": True,
    }
    return episodes, audit


def _route_subset(episodes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = pd.read_parquet(
        ROUTE_LEDGER,
        columns=[
            "ledger_kind",
            "symbol",
            "session",
            "bar_ordinal",
            "semantic_loop_id",
            "orientation_id",
            "progress_states",
            "transitions_remaining",
            "available_timestamp_utc",
        ],
    ).rename(columns={"symbol": "stock"})
    validate_protected_boundary(ledger["session"])
    keys = episodes[["stock", "session", "checkpoint", "signal_timestamp"]].copy()
    active = ledger.loc[ledger["ledger_kind"].astype(str).eq("active_prefix")].merge(
        keys,
        left_on=["stock", "session", "bar_ordinal"],
        right_on=["stock", "session", "checkpoint"],
        how="inner",
        validate="many_to_one",
    )
    maxima = (
        keys.groupby(["stock", "session"], sort=False)["checkpoint"]
        .max()
        .rename("maximum_checkpoint")
        .reset_index()
    )
    completions = ledger.loc[ledger["ledger_kind"].astype(str).eq("registered_completion")].merge(
        maxima, on=["stock", "session"], how="inner", validate="many_to_one"
    )
    completions = completions.loc[
        completions["bar_ordinal"].astype(int) <= completions["maximum_checkpoint"].astype(int)
    ].copy()
    active_available = pd.to_datetime(active["available_timestamp_utc"], utc=True, errors="raise")
    active_signal = pd.to_datetime(active["signal_timestamp"], utc=True, errors="raise")
    if bool(active_available.gt(active_signal).any()):
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "an active-prefix orientation was unavailable at signal time",
        )
    subset = pd.concat(
        [
            active[ledger.columns],
            completions[ledger.columns],
        ],
        ignore_index=True,
    ).drop_duplicates()
    audit = {
        "active_prefix_rows_used": int(len(active)),
        "registered_completion_rows_used": int(len(completions)),
        "maximum_active_prefix_availability_lag_seconds": float(
            (active_signal - active_available).dt.total_seconds().max()
        )
        if len(active)
        else 0.0,
        "future_available_active_prefix_rows": 0,
        "ledger_completed_count_semantics": (
            "one-based completed-bar count; checkpoint N is available at the "
            "close of zero-based price bar N-1"
        ),
    }
    return subset, audit


def build_direction_features(
    episodes: pd.DataFrame,
    panel: pd.DataFrame,
    states: pd.DataFrame,
    behaviour: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    targeted = attach_direction_targets(episodes, states)
    d0 = build_d0_features(targeted, states)
    checkpoint_source = panel.rename(columns={"symbol": "stock"}).copy()
    checkpoint_source = checkpoint_source.merge(
        behaviour[
            [
                "stock",
                "session",
                "checkpoint",
                "exhaustion_available_timestamp",
                "signed_exhaustion",
            ]
        ],
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    exhaustion_available = pd.to_datetime(
        checkpoint_source["exhaustion_available_timestamp"],
        utc=True,
        errors="coerce",
    )
    checkpoint_available = pd.to_datetime(
        checkpoint_source["feature_available_timestamp_utc"],
        utc=True,
        errors="raise",
    )
    if bool(exhaustion_available.gt(checkpoint_available).fillna(False).any()):
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "signed exhaustion was unavailable at a checkpoint",
        )
    d1_all = build_signed_behavioural_features(checkpoint_source)
    d1 = targeted[["stock", "session", "checkpoint"]].merge(
        d1_all,
        on=["stock", "session", "checkpoint"],
        how="left",
        validate="one_to_one",
    )
    features = d0.copy()
    if (
        not features[["stock", "session", "checkpoint"]]
        .reset_index(drop=True)
        .equals(d1[["stock", "session", "checkpoint"]].reset_index(drop=True))
    ):
        raise ScreenBlocked(
            "blocked_reproducibility_or_audit_failure",
            "D1 row identities drifted from episode order",
        )
    for column in D1_FEATURES:
        features[column] = d1[column].to_numpy()
    orientation_map = pd.DataFrame(
        columns=["state", "mean_raw_signed_efficiency", "orientation_sign", "source_rule"]
    )
    orientation_source: dict[str, Any]
    try:
        orientation_contract = json.loads(ORIENTATION_CONTRACT.read_text(encoding="utf-8"))
        orientation_audit_source = json.loads(ORIENTATION_AUDIT.read_text(encoding="utf-8"))
        route_audit_source = json.loads(ROUTE_AUDIT.read_text(encoding="utf-8"))
        if not bool(orientation_audit_source.get("passed")) or not bool(
            route_audit_source.get("passed")
        ):
            raise ValueError("predecessor orientation or route-ledger audit did not pass")
        orientation_map = audited_state_orientation_map(pd.read_csv(CENTROIDS))
        route_subset, route_audit = _route_subset(targeted)
        route_input = targeted[
            ["stock", "session", "checkpoint", "signed_pressure", "route_resolution_state"]
        ].copy()
        d2 = build_route_orientation_features(route_input, route_subset, orientation_map)
        features = features.merge(
            d2,
            on=["stock", "session", "checkpoint"],
            how="left",
            validate="one_to_one",
        )
        orientation_source = {
            "passed": True,
            "status": "audited_outcome_free_orientation_reused",
            "source": str(ORIENTATION_CONTRACT.relative_to(REPO_ROOT)),
            "source_audit": str(ORIENTATION_AUDIT.relative_to(REPO_ROOT)),
            "source_rule": orientation_contract["orientation_sign_rule"],
            "construction": (
                "sign of the arithmetic mean of frozen development-fitted raw "
                "signed_efficiency_6 and signed_efficiency_12 centroids for the "
                "next required state; zero maps to +1"
            ),
            "outcome_fitted": False,
            "future_outcomes_used": False,
            "valid_cohort": sorted(features["stock"].astype(str).unique()),
            "valid_start": DEVELOPMENT_START,
            "valid_end": ASSESSMENT_END,
            "centroid_source": str(CENTROIDS.relative_to(REPO_ROOT)),
            "orientation_states": int(len(orientation_map)),
            **route_audit,
        }
    except (FileNotFoundError, KeyError, ValueError) as error:
        # D2 is optional by contract. Preserve the D0/D1 panel and freeze D1 as
        # primary; compatibility-only neutral columns are never fitted as D2.
        for column in D2_FEATURES:
            features[column] = 0.0
        orientation_source = {
            "passed": False,
            "status": "blocked_missing_auditable_orientation",
            "blocker": str(error),
            "source": None,
            "source_audit": None,
            "source_rule": None,
            "construction": "D2 not constructed; D0 and D1 remain causally available",
            "outcome_fitted": False,
            "future_outcomes_used": False,
            "valid_cohort": sorted(features["stock"].astype(str).unique()),
            "valid_start": DEVELOPMENT_START,
            "valid_end": ASSESSMENT_END,
            "orientation_states": 0,
        }
    features["checkpoint_category"] = features["checkpoint"].astype(int).astype(str)
    features["day_of_week"] = pd.to_datetime(features["session"], errors="raise").dt.day_name()
    features["checkpoint_group"] = pd.cut(
        features["checkpoint"].astype(int),
        bins=[5, 14, 24, 34],
        labels=["early_6_14", "middle_16_24", "late_26_34"],
        include_lowest=True,
    ).astype(str)
    features["assessment_month_group"] = features["session"].astype(str).str[:7]
    maximum_feature_timestamp = pd.to_datetime(
        features["maximum_feature_source_timestamp"], utc=True, errors="raise"
    )
    signal_timestamp = pd.to_datetime(features["signal_timestamp"], utc=True, errors="raise")
    future_feature_rows = int(maximum_feature_timestamp.gt(signal_timestamp).sum())
    if future_feature_rows:
        raise ScreenBlocked(
            "blocked_chronology_or_leakage_failure",
            "D0 used a bar after the signal close",
        )
    target_audit = {
        "passed": True,
        "primary_horizon_minutes": 10,
        "entry_definition": "open of zero-based price bar checkpoint",
        "signal_definition": "close of zero-based price bar checkpoint-1",
        "secondary_horizons_minutes": [5, 15, 30],
        "zero_return_10m_count": int(features["zero_return_10m"].sum()),
        "future_bar_features": future_feature_rows,
        "target_rows": int(len(features)),
        "direction_up_rows": int(features["direction_up_10m"].eq(1).sum()),
        "direction_down_rows": int(features["direction_up_10m"].eq(0).sum()),
        "option_pnl_calculated": False,
        "intraday_option_quotes_used": False,
    }
    return features, orientation_map, orientation_source, target_audit


def feature_specifications() -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        "D0": {
            "numeric": D0_FEATURES,
            "categorical": CATEGORICAL_CONTROLS,
        },
        "D1": {
            "numeric": (*D0_FEATURES, *D1_FEATURES),
            "categorical": CATEGORICAL_CONTROLS,
        },
        "D2": {
            "numeric": (*D0_FEATURES, *D1_FEATURES, *D2_FEATURES),
            "categorical": (*CATEGORICAL_CONTROLS, "route_resolution_state"),
        },
    }


def fit_direction_stack(
    features: pd.DataFrame,
    *,
    d2_available: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    development = features.loc[features["partition"].astype(str).eq("development")].copy()
    assessment = features.loc[features["partition"].astype(str).eq("assessment")].copy()
    development["fold"] = assign_contiguous_session_folds(
        development["session"], folds=4
    ).to_numpy()
    specifications = feature_specifications()
    configurations: dict[str, Any] = {
        "shared": {
            "development_period": [DEVELOPMENT_START, DEVELOPMENT_END],
            "assessment_period": [ASSESSMENT_START, ASSESSMENT_END],
            "blocked_oof_folds": 4,
            "fold_unit": "complete_session",
            "fold_shape": "contiguous_calendar_blocks",
            "target": "direction_up_10m",
            "zero_returns_excluded_from_fitting": True,
            "M1_probability_used_as_direction_feature": False,
        }
    }
    thresholds: dict[str, Any] = {}
    fitted_model_ids = MODEL_IDS if d2_available else ("D0", "D1")
    for model_id in fitted_model_ids:
        numeric = specifications[model_id]["numeric"]
        categorical = specifications[model_id]["categorical"]
        if "m1_probability" in numeric:
            raise AssertionError("M1 probability cannot be a direction feature")
        probabilities = np.full(len(development), np.nan, dtype=float)
        fold_models: list[dict[str, Any]] = []
        for fold in range(4):
            training = development.loc[development["fold"].ne(fold)]
            held_out = development.loc[development["fold"].eq(fold)]
            if set(training["session"]).intersection(set(held_out["session"])):
                raise ScreenBlocked(
                    "blocked_chronology_or_leakage_failure",
                    "an OOF session appears in training",
                )
            model = fit_direction_model(
                training,
                target_column="direction_up_10m",
                numeric_features=numeric,
                categorical_features=categorical,
                model_id=f"{model_id}_fold_{fold}",
            )
            probabilities[development["fold"].eq(fold).to_numpy()] = model.predict(held_out)
            fold_models.append(
                {
                    "fold": fold,
                    "training_sessions": int(training["session"].nunique()),
                    "held_out_sessions": int(held_out["session"].nunique()),
                    "held_out_start": str(held_out["session"].min()),
                    "held_out_end": str(held_out["session"].max()),
                    "specification": model.as_dict(),
                }
            )
        if not np.isfinite(probabilities).all():
            raise ScreenBlocked(
                "blocked_model_convergence_failure",
                f"{model_id} OOF probabilities are incomplete",
            )
        development[f"{model_id}_probability"] = probabilities
        full = fit_direction_model(
            development,
            target_column="direction_up_10m",
            numeric_features=numeric,
            categorical_features=categorical,
            model_id=f"{model_id}_full_2024",
        )
        assessment[f"{model_id}_probability"] = full.predict(assessment)
        thresholds[model_id] = freeze_confidence_boundary(probabilities)
        thresholds[model_id]["fixed_probability_sensitivities"] = {
            "55_percent_probability": 0.05,
            "60_percent_probability": 0.10,
            "65_percent_probability": 0.15,
        }
        configurations[model_id] = {
            "numeric_features": list(numeric),
            "categorical_features": list(categorical),
            "oof_models": fold_models,
            "full_development_model": full.as_dict(),
        }
    if not d2_available:
        configurations["D2"] = {
            "status": "blocked_missing_auditable_orientation",
            "fitted": False,
            "compatibility_probability_alias": "D1",
        }
        development["D2_probability"] = development["D1_probability"].to_numpy(float)
        assessment["D2_probability"] = assessment["D1_probability"].to_numpy(float)
        thresholds["D2"] = {
            **cast(dict[str, Any], thresholds["D1"]),
            "status": "blocked_missing_auditable_orientation",
            "source": "compatibility_alias_to_D1_not_a_fitted_D2_threshold",
        }
    configurations["primary_models_fitted"] = len(fitted_model_ids)
    configurations["oof_models_fitted"] = 4 * len(fitted_model_ids)
    configurations["D2_causally_available"] = d2_available
    return development, assessment, configurations, thresholds


def add_frozen_subgroups(
    development: pd.DataFrame, assessment: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = assessment.copy()
    medians = {
        "atm_iv": float(np.nanmedian(development["atm_iv"].to_numpy(float))),
        "m1_probability": float(np.nanmedian(development["m1_probability"].to_numpy(float))),
        "transition_probability": float(
            np.nanmedian(development["transition_probability"].to_numpy(float))
        ),
    }
    for column, median in medians.items():
        result[f"{column}_half"] = np.where(
            pd.to_numeric(result[column], errors="coerce").le(median), "low", "high"
        )
    for horizon in HORIZONS:
        medians[f"absolute_return_{horizon}m_q75"] = float(
            np.nanquantile(
                development[f"absolute_log_return_{horizon}m"].to_numpy(float),
                0.75,
            )
        )
    return result, medians


def _population_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "episodes": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["stock"].nunique()),
        "months": int(frame["session"].astype(str).str[:7].nunique()),
    }


def _safe_binary_metrics(frame: pd.DataFrame, probability_column: str) -> dict[str, float | int]:
    valid = frame.loc[frame["direction_up_10m"].notna() & frame[probability_column].notna()]
    if valid.empty:
        return {
            "log_loss": math.nan,
            "brier_score": math.nan,
            "auc": math.nan,
            "average_precision": math.nan,
            "accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "matthews_correlation_coefficient": math.nan,
            "up_base_rate": math.nan,
            "predicted_up_rate": math.nan,
            "calibration_intercept": math.nan,
            "calibration_slope": math.nan,
            "expected_calibration_error": math.nan,
            "episodes": 0,
        }
    return binary_direction_metrics(
        valid["direction_up_10m"].to_numpy(float),
        valid[probability_column].to_numpy(float),
    )


def direction_metric_tables(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    rows: list[dict[str, Any]] = []
    assessment_metrics: dict[str, dict[str, float | int]] = {}
    for period, frame in (
        ("development_oof", development),
        ("retrospective_assessment", assessment),
    ):
        for model_id in MODEL_IDS:
            metrics = _safe_binary_metrics(frame, f"{model_id}_probability")
            rows.append(
                {
                    "record_type": "model",
                    "period": period,
                    "model": model_id,
                    **metrics,
                    **{
                        key: value
                        for key, value in _population_counts(frame).items()
                        if key != "episodes"
                    },
                }
            )
            if period == "retrospective_assessment":
                assessment_metrics[model_id] = metrics
    comparisons = (("D1", "D0"), ("D2", "D1"), ("D2", "D0"))
    for candidate, baseline in comparisons:
        candidate_metrics = assessment_metrics[candidate]
        baseline_metrics = assessment_metrics[baseline]
        rows.append(
            {
                "record_type": "increment",
                "period": "retrospective_assessment",
                "model": f"{candidate}_minus_{baseline}",
                "log_loss_improvement": float(baseline_metrics["log_loss"])
                - float(candidate_metrics["log_loss"]),
                "brier_improvement": float(baseline_metrics["brier_score"])
                - float(candidate_metrics["brier_score"]),
                "auc_improvement": float(candidate_metrics["auc"]) - float(baseline_metrics["auc"]),
                "average_precision_improvement": float(candidate_metrics["average_precision"])
                - float(baseline_metrics["average_precision"]),
                "accuracy_improvement": float(candidate_metrics["accuracy"])
                - float(baseline_metrics["accuracy"]),
                "balanced_accuracy_improvement": float(candidate_metrics["balanced_accuracy"])
                - float(baseline_metrics["balanced_accuracy"]),
            }
        )
    return pd.DataFrame(rows), assessment_metrics


def apply_frozen_policies(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    thresholds: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development_output = development.copy()
    assessment_output = assessment.copy()
    for model_id in MODEL_IDS:
        boundary = float(cast(Mapping[str, Any], thresholds[model_id])["boundary"])
        for frame in (development_output, assessment_output):
            action_column = f"{model_id}_action"
            frame[action_column] = apply_selective_policy(
                frame[f"{model_id}_probability"].to_numpy(float), boundary
            )
            for horizon in HORIZONS:
                frame[f"{model_id}_aligned_return_{horizon}m"] = aligned_returns(
                    frame[action_column].to_numpy(),
                    frame[f"signed_log_return_{horizon}m"].to_numpy(float),
                )
    return development_output, assessment_output


def selective_metric_table(
    assessment: pd.DataFrame,
    thresholds: Mapping[str, Any],
    primary_candidate: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    rows: list[dict[str, Any]] = []
    primary_metrics: dict[str, dict[str, float | int]] = {}
    for model_id in MODEL_IDS:
        metrics = selective_policy_metrics(
            assessment,
            action_column=f"{model_id}_action",
            horizon_minutes=10,
        )
        rows.append(
            {
                "model": model_id,
                "policy": "development_oof_35_percent_minimum_150",
                "boundary": float(cast(Mapping[str, Any], thresholds[model_id])["boundary"]),
                "primary_horizon": True,
                **metrics,
            }
        )
        if model_id == primary_candidate:
            primary_metrics["10"] = metrics
    for horizon in HORIZONS:
        if horizon == 10:
            continue
        metrics = selective_policy_metrics(
            assessment,
            action_column=f"{primary_candidate}_action",
            horizon_minutes=horizon,
        )
        rows.append(
            {
                "model": primary_candidate,
                "policy": "development_oof_35_percent_minimum_150",
                "boundary": float(
                    cast(Mapping[str, Any], thresholds[primary_candidate])["boundary"]
                ),
                "primary_horizon": False,
                **metrics,
            }
        )
        primary_metrics[str(horizon)] = metrics
    for label, boundary in (
        ("fixed_55_percent_probability", 0.05),
        ("fixed_60_percent_probability", 0.10),
        ("fixed_65_percent_probability", 0.15),
    ):
        sensitivity = assessment.copy()
        sensitivity["_sensitivity_action"] = apply_selective_policy(
            sensitivity[f"{primary_candidate}_probability"].to_numpy(float),
            boundary,
        )
        metrics = selective_policy_metrics(
            sensitivity,
            action_column="_sensitivity_action",
            horizon_minutes=10,
        )
        rows.append(
            {
                "model": primary_candidate,
                "policy": label,
                "boundary": boundary,
                "primary_horizon": False,
                **metrics,
            }
        )
    return pd.DataFrame(rows), primary_metrics


def baseline_metric_table(
    assessment: pd.DataFrame, development_up_rate: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    baselines = baseline_predictions(assessment, development_up_rate=development_up_rate)
    rows: list[dict[str, Any]] = []
    accuracies: dict[str, float] = {}
    side_columns: dict[str, np.ndarray[Any, np.dtype[Any]]] = {
        "B0": baselines["B0_side"].to_numpy(int),
        "B1": baselines["B1_side"].to_numpy(int),
        "B2": baselines["B2_side"].to_numpy(int),
        "B3": baselines["B3_side"].to_numpy(int),
        "B4": baselines["B4_side"].to_numpy(int),
    }
    labels = assessment["direction_up_10m"].to_numpy(float)
    signed = assessment["signed_log_return_10m"].to_numpy(float)
    for baseline, sides in side_columns.items():
        if baseline == "B0":
            probabilities = baselines["B0_probability"].to_numpy(float)
        else:
            probabilities = np.where(sides > 0, 1.0 - 1e-12, np.where(sides < 0, 1e-12, 0.5))
        valid = np.isfinite(labels) & (sides != 0)
        side_accuracy = (
            float(np.mean((sides[valid] > 0).astype(int) == labels[valid].astype(int)))
            if valid.any()
            else math.nan
        )
        accuracies[baseline] = side_accuracy
        proper = binary_direction_metrics(labels, probabilities)
        aligned = sides.astype(float) * signed
        rows.append(
            {
                "baseline": baseline,
                "definition": {
                    "B0": "frozen_2024_up_rate",
                    "B1": "always_long",
                    "B2": "recent_completed_10m_stock_return_sign",
                    "B3": "recent_completed_5m_stock_return_sign",
                    "B4": "recent_completed_10m_market_proxy_return_sign",
                }[baseline],
                **proper,
                "side_eligible_episodes": int(valid.sum()),
                "directional_accuracy": side_accuracy,
                "mean_aligned_return_10m": float(np.nanmean(aligned)),
                "median_aligned_return_10m": float(np.nanmedian(aligned)),
                "positive_aligned_return_rate": float(np.nanmean(aligned > 0.0)),
            }
        )
    return pd.DataFrame(rows), accuracies


def _combined_subset_metrics(
    subset: pd.DataFrame,
    *,
    primary_candidate: str,
    scope: str,
    group: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scope": scope,
        "group": group,
        **_population_counts(subset),
    }
    if subset.empty:
        return row
    row.update(_safe_binary_metrics(subset, f"{primary_candidate}_probability"))
    selective = selective_policy_metrics(
        subset,
        action_column=f"{primary_candidate}_action",
        horizon_minutes=10,
    )
    row.update({f"selective_{key}": value for key, value in selective.items()})
    return row


def stability_tables(
    assessment: pd.DataFrame, primary_candidate: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_rows = [
        _combined_subset_metrics(
            group,
            primary_candidate=primary_candidate,
            scope="assessment_month",
            group=str(month),
        )
        for month, group in assessment.groupby("assessment_month_group", sort=True)
    ]
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint, group in assessment.groupby("checkpoint", sort=True):
        checkpoint_rows.append(
            _combined_subset_metrics(
                group,
                primary_candidate=primary_candidate,
                scope="checkpoint",
                group=str(int(checkpoint)),
            )
        )
    for column, scope in (
        ("checkpoint_group", "checkpoint_group"),
        ("atm_iv_half", "previous_close_atm_iv"),
        ("m1_probability_half", "m1_probability_tail"),
        ("transition_probability_half", "transition_probability"),
    ):
        for value, group in assessment.groupby(column, sort=True):
            checkpoint_rows.append(
                _combined_subset_metrics(
                    group,
                    primary_candidate=primary_candidate,
                    scope=scope,
                    group=str(value),
                )
            )
    for action in ("CALL", "PUT"):
        group = assessment.loc[assessment[f"{primary_candidate}_action"].astype(str).eq(action)]
        checkpoint_rows.append(
            _combined_subset_metrics(
                group,
                primary_candidate=primary_candidate,
                scope="predicted_side",
                group=action,
            )
        )
    route_rows = [
        _combined_subset_metrics(
            group,
            primary_candidate=primary_candidate,
            scope="route_resolution_state",
            group=str(state),
        )
        for state, group in assessment.groupby("route_resolution_state", sort=True)
    ]
    known_states = {
        "BROAD_CONFLICT",
        "NARROWING",
        "LOW_ROUTE_SUPPORT",
    }
    other = assessment.loc[~assessment["route_resolution_state"].astype(str).isin(known_states)]
    route_rows.append(
        _combined_subset_metrics(
            other,
            primary_candidate=primary_candidate,
            scope="route_resolution_rollup",
            group="OTHER_ROUTE_STATES",
        )
    )
    return (
        pd.DataFrame(monthly_rows),
        pd.DataFrame(checkpoint_rows),
        pd.DataFrame(route_rows),
    )


def material_move_table(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    primary_candidate: str,
    frozen_boundaries: Mapping[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        expectation = assessment[f"iv_expected_absolute_{horizon}m"].to_numpy(float)
        absolute = assessment[f"absolute_log_return_{horizon}m"].to_numpy(float)
        available = np.isfinite(expectation)
        for group_name, mask in (
            ("realised_iv_excess", available & (absolute > expectation)),
            ("non_iv_excess", available & (absolute <= expectation)),
        ):
            group = assessment.loc[mask].copy()
            if group.empty:
                continue
            metrics = selective_policy_metrics(
                group,
                action_column=f"{primary_candidate}_action",
                horizon_minutes=horizon,
            )
            rows.append(
                {
                    "scope": "previous_close_iv_expectation",
                    "group": group_name,
                    "horizon_minutes": horizon,
                    "descriptive_only": True,
                    **metrics,
                }
            )
        boundary = float(frozen_boundaries[f"absolute_return_{horizon}m_q75"])
        largest = assessment.loc[absolute >= boundary].copy()
        if not largest.empty:
            metrics = selective_policy_metrics(
                largest,
                action_column=f"{primary_candidate}_action",
                horizon_minutes=horizon,
            )
            rows.append(
                {
                    "scope": "largest_absolute_movement_quartile",
                    "group": "top_2024_frozen_quartile",
                    "horizon_minutes": horizon,
                    "development_q75_boundary": boundary,
                    "descriptive_only": True,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def remaining_and_excursion_tables(
    assessment: pd.DataFrame, primary_candidate: str
) -> tuple[pd.DataFrame, pd.DataFrame, bool, float]:
    action_column = f"{primary_candidate}_action"
    actioned = assessment.loc[assessment[action_column].astype(str).ne("ABSTAIN")].copy()
    side = np.where(actioned[action_column].astype(str).eq("CALL"), 1, -1)
    actioned["predicted_side"] = side
    actioned["time_of_maximum_favourable_excursion_10m"] = np.where(
        side > 0,
        actioned["time_of_upside_mfe_10m"],
        actioned["time_of_upside_mae_10m"],
    )
    actioned["time_of_maximum_adverse_excursion_10m"] = np.where(
        side > 0,
        actioned["time_of_upside_mae_10m"],
        actioned["time_of_upside_mfe_10m"],
    )
    actioned["time_of_maximum_favourable_excursion_30m"] = np.where(
        side > 0,
        actioned["time_of_upside_mfe_30m"],
        actioned["time_of_upside_mae_30m"],
    )
    actioned["time_of_maximum_adverse_excursion_30m"] = np.where(
        side > 0,
        actioned["time_of_upside_mae_30m"],
        actioned["time_of_upside_mfe_30m"],
    )
    remaining_columns = [
        "stock",
        "session",
        "checkpoint",
        "signal_timestamp",
        "prospective_entry_timestamp",
        action_column,
        "predicted_side",
        "return_realised_before_signal",
        "return_signal_to_entry",
        "return_after_entry_10m",
        "return_after_entry_30m",
        "fraction_eventual_10m_move_after_entry",
        "fraction_eventual_30m_move_after_entry",
        "time_of_maximum_favourable_excursion_10m",
        "time_of_maximum_adverse_excursion_10m",
        "time_of_maximum_favourable_excursion_30m",
        "time_of_maximum_adverse_excursion_30m",
    ]
    remaining = actioned[remaining_columns].rename(columns={action_column: "action"})
    numerator = float(np.nanmean(np.abs(actioned["return_after_entry_10m"].to_numpy(float))))
    denominator = float(
        np.nanmean(
            np.abs(actioned["return_realised_before_signal"].to_numpy(float))
            + np.abs(actioned["return_signal_to_entry"].to_numpy(float))
            + np.abs(actioned["return_after_entry_10m"].to_numpy(float))
        )
    )
    remaining_ratio = numerator / denominator if denominator > 0.0 else math.nan
    late_direction_problem = bool(not math.isfinite(remaining_ratio) or remaining_ratio < 0.50)
    summary = pd.DataFrame(
        [
            {
                "stock": "__SUMMARY__",
                "session": "__SUMMARY__",
                "checkpoint": -1,
                "action": "ALL",
                "return_realised_before_signal": float(
                    actioned["return_realised_before_signal"].mean()
                ),
                "return_signal_to_entry": float(actioned["return_signal_to_entry"].mean()),
                "return_after_entry_10m": float(actioned["return_after_entry_10m"].mean()),
                "return_after_entry_30m": float(actioned["return_after_entry_30m"].mean()),
                "fraction_eventual_10m_move_after_entry": float(
                    actioned["fraction_eventual_10m_move_after_entry"].mean()
                ),
                "fraction_eventual_30m_move_after_entry": float(
                    actioned["fraction_eventual_30m_move_after_entry"].mean()
                ),
                "mean_absolute_remaining_fraction_10m": remaining_ratio,
                "late_direction_problem": late_direction_problem,
            }
        ]
    )
    remaining = pd.concat([summary, remaining], ignore_index=True, sort=False)
    excursion_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        metrics = selective_policy_metrics(
            assessment,
            action_column=action_column,
            horizon_minutes=horizon,
        )
        excursion_rows.append(
            {
                "horizon_minutes": horizon,
                "model": primary_candidate,
                "mean_favourable_excursion": metrics["mean_favourable_excursion"],
                "mean_adverse_excursion": metrics["mean_adverse_excursion"],
                "favourable_adverse_excursion_ratio": metrics["favourable_adverse_excursion_ratio"],
            }
        )
    return (
        remaining,
        pd.DataFrame(excursion_rows),
        late_direction_problem,
        remaining_ratio,
    )


def stock_and_concentration_tables(
    assessment: pd.DataFrame, primary_candidate: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    action_column = f"{primary_candidate}_action"
    actioned = assessment.loc[assessment[action_column].astype(str).ne("ABSTAIN")].copy()
    actioned["_aligned"] = actioned[f"{primary_candidate}_aligned_return_10m"].to_numpy(float)
    total_positive = float(actioned["_aligned"].clip(lower=0.0).sum())
    total_negative = float(-actioned["_aligned"].clip(upper=0.0).sum())
    rows: list[dict[str, Any]] = []
    for stock, population in assessment.groupby("stock", sort=True):
        actions = population.loc[population[action_column].astype(str).ne("ABSTAIN")]
        aligned = actions[f"{primary_candidate}_aligned_return_10m"].to_numpy(float)
        valid = actions["direction_up_10m"].notna()
        predicted = actions.loc[valid, action_column].astype(str)
        truth = actions.loc[valid, "direction_up_10m"].to_numpy(int)
        accuracy = (
            float(np.mean((predicted.eq("CALL").to_numpy(int)) == truth))
            if valid.any()
            else math.nan
        )
        calls = int(actions[action_column].astype(str).eq("CALL").sum())
        puts = int(actions[action_column].astype(str).eq("PUT").sum())
        rows.append(
            {
                "scope": "stock",
                "stock": str(stock),
                "episodes": int(len(population)),
                "actions": int(len(actions)),
                "direction_accuracy": accuracy,
                "mean_aligned_return": float(np.nanmean(aligned)) if len(aligned) else math.nan,
                "median_aligned_return": float(np.nanmedian(aligned)) if len(aligned) else math.nan,
                "call_count": calls,
                "put_count": puts,
                "call_put_balance": calls / puts if puts else math.nan,
                "contribution_to_total_positive_aligned_return": (
                    float(np.nansum(np.maximum(aligned, 0.0))) / total_positive
                    if total_positive > 0.0
                    else math.nan
                ),
                "contribution_to_total_negative_aligned_return": (
                    float(np.nansum(np.maximum(-aligned, 0.0))) / total_negative
                    if total_negative > 0.0
                    else math.nan
                ),
            }
        )
    for excluded_stock in sorted(assessment["stock"].astype(str).unique()):
        retained = assessment.loc[assessment["stock"].astype(str).ne(excluded_stock)]
        metrics = selective_policy_metrics(
            retained,
            action_column=action_column,
            horizon_minutes=10,
        )
        rows.append(
            {
                "scope": "leave_one_stock_out",
                "stock": excluded_stock,
                "episodes": int(len(retained)),
                "actions": metrics["actions"],
                "direction_accuracy": metrics["directional_accuracy"],
                "mean_aligned_return": metrics["mean_aligned_return"],
                "median_aligned_return": metrics["median_aligned_return"],
            }
        )
    episode_stock_share = float(assessment.groupby("stock").size().max() / len(assessment))
    action_stock_share = float(actioned.groupby("stock").size().max() / len(actioned))
    action_month_share = float(
        actioned.groupby("assessment_month_group").size().max() / len(actioned)
    )
    action_session_share = float(actioned.groupby("session").size().max() / len(actioned))
    concentrations = {
        "maximum_stock_share_of_episodes": episode_stock_share,
        "maximum_stock_share_of_actions": action_stock_share,
        "maximum_month_share_of_actions": action_month_share,
        "maximum_session_share_of_actions": action_session_share,
    }
    concentration = pd.DataFrame(
        [
            {
                "metric": name,
                "value": value,
                "gate": {
                    "maximum_stock_share_of_episodes": 0.15,
                    "maximum_stock_share_of_actions": 0.20,
                    "maximum_month_share_of_actions": 0.30,
                    "maximum_session_share_of_actions": 0.08,
                }[name],
                "passed": value
                <= {
                    "maximum_stock_share_of_episodes": 0.15,
                    "maximum_stock_share_of_actions": 0.20,
                    "maximum_month_share_of_actions": 0.30,
                    "maximum_session_share_of_actions": 0.08,
                }[name],
            }
            for name, value in concentrations.items()
        ]
    )
    return pd.DataFrame(rows), concentration, concentrations


def bootstrap_metric_values(
    sample: pd.DataFrame,
    primary_candidate: str,
) -> dict[str, float]:
    """Calculate the frozen bootstrap metric vector for one session sample."""

    metrics = {
        model_id: _safe_binary_metrics(sample, f"{model_id}_probability") for model_id in MODEL_IDS
    }
    values = {
        **{f"{model_id}_auc": float(metrics[model_id]["auc"]) for model_id in MODEL_IDS},
        **{f"{model_id}_log_loss": float(metrics[model_id]["log_loss"]) for model_id in MODEL_IDS},
        "D1_minus_D0_log_loss_improvement": (
            float(metrics["D0"]["log_loss"]) - float(metrics["D1"]["log_loss"])
        ),
        "D2_minus_D1_log_loss_improvement": (
            float(metrics["D1"]["log_loss"]) - float(metrics["D2"]["log_loss"])
        ),
        "D2_minus_D0_auc_improvement": (float(metrics["D2"]["auc"]) - float(metrics["D0"]["auc"])),
    }
    selective = selective_policy_metrics(
        sample,
        action_column=f"{primary_candidate}_action",
        horizon_minutes=10,
    )
    for metric in (
        "action_coverage",
        "directional_accuracy",
        "balanced_accuracy",
        "mean_aligned_return",
        "median_aligned_return",
        "positive_aligned_return_rate",
        "favourable_adverse_excursion_ratio",
    ):
        values[f"selective_{metric}"] = float(selective[metric])
    return values


def bootstrap_table(
    assessment: pd.DataFrame, primary_candidate: str
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    samples = session_bootstrap_samples(
        assessment["session"], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    groups = {str(session): group for session, group in assessment.groupby("session", sort=False)}
    draw_values: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []

    def add(draw: int, metric: str, value: float) -> None:
        draw_values.setdefault(metric, []).append(value)
        rows.append(
            {
                "record_type": "draw",
                "draw": draw,
                "metric": metric,
                "value": value,
                "confidence_level": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "sampled_sessions_json": None,
            }
        )

    for draw, sampled_sessions in enumerate(samples):
        rows.append(
            {
                "record_type": "sample_identity",
                "draw": draw,
                "metric": "__sampled_sessions__",
                "value": np.nan,
                "confidence_level": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "sampled_sessions_json": json.dumps(
                    [str(session) for session in sampled_sessions],
                    separators=(",", ":"),
                ),
            }
        )
        sample = pd.concat(
            [groups[session] for session in sampled_sessions],
            ignore_index=True,
        )
        for metric, value in bootstrap_metric_values(sample, primary_candidate).items():
            add(draw, metric, value)
    intervals: dict[str, dict[str, float]] = {}
    for metric, values in sorted(draw_values.items()):
        array = np.asarray(values, dtype=float)
        intervals[metric] = {}
        for confidence in (0.80, 0.90, 0.95):
            alpha = 0.5 * (1.0 - confidence)
            lower, upper = np.nanquantile(array, [alpha, 1.0 - alpha])
            label = f"{int(confidence * 100)}"
            intervals[metric][f"{label}_lower"] = float(lower)
            intervals[metric][f"{label}_upper"] = float(upper)
            rows.append(
                {
                    "record_type": "interval",
                    "draw": np.nan,
                    "metric": metric,
                    "value": np.nan,
                    "confidence_level": confidence,
                    "lower": float(lower),
                    "upper": float(upper),
                    "sampled_sessions_json": None,
                }
            )
    return pd.DataFrame(rows), intervals


def null_refit_table(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    thresholds: Mapping[str, Any],
    primary_candidate: str,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    specifications = feature_specifications()
    real_metrics = {
        model_id: _safe_binary_metrics(assessment, f"{model_id}_probability")
        for model_id in MODEL_IDS
    }
    real_selective = {
        model_id: selective_policy_metrics(
            assessment,
            action_column=f"{model_id}_action",
            horizon_minutes=10,
        )
        for model_id in MODEL_IDS
    }
    rows: list[dict[str, Any]] = []
    for null_index, seed in enumerate(NULL_SEEDS):
        permuted = development.copy()
        permuted["null_direction_up_10m"] = permute_labels_within_slates(
            permuted,
            label_column="direction_up_10m",
            strata=("session", "checkpoint_group"),
            seed=seed,
        )
        permuted_labels_json = json.dumps(
            [
                None if pd.isna(value) else int(value)
                for value in permuted["null_direction_up_10m"].tolist()
            ],
            separators=(",", ":"),
        )
        for model_id in MODEL_IDS:
            specification = specifications[model_id]
            null_model = fit_direction_model(
                permuted,
                target_column="null_direction_up_10m",
                numeric_features=specification["numeric"],
                categorical_features=specification["categorical"],
                model_id=f"{model_id}_null_{null_index}",
            )
            probability = null_model.predict(assessment)
            null_frame = assessment.copy()
            null_frame["_null_probability"] = probability
            null_frame["_null_action"] = apply_selective_policy(
                probability,
                float(cast(Mapping[str, Any], thresholds[model_id])["boundary"]),
            )
            metrics = _safe_binary_metrics(null_frame, "_null_probability")
            selective = selective_policy_metrics(
                null_frame,
                action_column="_null_action",
                horizon_minutes=10,
            )
            real = real_metrics[model_id]
            real_policy = real_selective[model_id]
            rows.append(
                {
                    "null_refit": null_index,
                    "seed": seed,
                    "model": model_id,
                    "strata": "development_session_x_checkpoint_group",
                    "labels_permuted_among_stocks": True,
                    "permuted_labels_json": permuted_labels_json,
                    "assessment_design_frozen": True,
                    "log_loss": metrics["log_loss"],
                    "brier_score": metrics["brier_score"],
                    "auc": metrics["auc"],
                    "selective_directional_accuracy": selective["directional_accuracy"],
                    "selective_mean_aligned_return_10m": selective["mean_aligned_return"],
                    "real_exceeds_null_log_loss": float(real["log_loss"])
                    < float(metrics["log_loss"]),
                    "real_exceeds_null_brier": float(real["brier_score"])
                    < float(metrics["brier_score"]),
                    "real_exceeds_null_auc": float(real["auc"]) > float(metrics["auc"]),
                    "real_exceeds_null_selective_accuracy": float(
                        real_policy["directional_accuracy"]
                    )
                    > float(selective["directional_accuracy"]),
                    "real_exceeds_null_mean_aligned_return": float(
                        real_policy["mean_aligned_return"]
                    )
                    > float(selective["mean_aligned_return"]),
                    "model_specification_json": json.dumps(
                        _json_safe(null_model.as_dict()),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    primary_rows = frame.loc[frame["model"].eq(primary_candidate)]
    summary: dict[str, int | bool] = {
        "null_refits": len(NULL_SEEDS),
        "real_exceeds_log_loss_count": int(primary_rows["real_exceeds_null_log_loss"].sum()),
        "real_exceeds_brier_count": int(primary_rows["real_exceeds_null_brier"].sum()),
        "real_exceeds_auc_count": int(primary_rows["real_exceeds_null_auc"].sum()),
        "real_exceeds_selective_accuracy_count": int(
            primary_rows["real_exceeds_null_selective_accuracy"].sum()
        ),
        "real_exceeds_mean_aligned_return_count": int(
            primary_rows["real_exceeds_null_mean_aligned_return"].sum()
        ),
    }
    summary["exceeds_all_five_on_log_loss_or_auc"] = bool(
        summary["real_exceeds_log_loss_count"] == 5 or summary["real_exceeds_auc_count"] == 5
    )
    return frame, summary


def layer_attribution_table(
    development: pd.DataFrame,
    assessment_metrics: Mapping[str, Mapping[str, float | int]],
    selective_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    frozen_rows = selective_metrics.loc[
        selective_metrics["policy"].eq("development_oof_35_percent_minimum_150")
        & selective_metrics["horizon_minutes"].eq(10)
    ].set_index("model")
    rows: list[dict[str, Any]] = []
    layer_flags: dict[str, bool] = {}
    development_metrics = {
        model_id: _safe_binary_metrics(development, f"{model_id}_probability")
        for model_id in MODEL_IDS
    }
    for candidate, baseline, layer in (
        ("D1", "D0", "signed_behaviour"),
        ("D2", "D1", "route_orientation"),
    ):
        proper = assessment_metrics
        row = {
            "layer": layer,
            "comparison": f"{candidate}_minus_{baseline}",
            "log_loss_improvement": float(proper[baseline]["log_loss"])
            - float(proper[candidate]["log_loss"]),
            "brier_improvement": float(proper[baseline]["brier_score"])
            - float(proper[candidate]["brier_score"]),
            "auc_improvement": float(proper[candidate]["auc"]) - float(proper[baseline]["auc"]),
            "selective_accuracy_improvement": float(
                frozen_rows.loc[candidate, "directional_accuracy"]
            )
            - float(frozen_rows.loc[baseline, "directional_accuracy"]),
            "aligned_return_improvement": float(frozen_rows.loc[candidate, "mean_aligned_return"])
            - float(frozen_rows.loc[baseline, "mean_aligned_return"]),
            "development_oof_log_loss_improvement": float(development_metrics[baseline]["log_loss"])
            - float(development_metrics[candidate]["log_loss"]),
            "development_oof_brier_improvement": float(development_metrics[baseline]["brier_score"])
            - float(development_metrics[candidate]["brier_score"]),
            "development_oof_auc_improvement": float(development_metrics[candidate]["auc"])
            - float(development_metrics[baseline]["auc"]),
        }
        row["assessment_increment_non_adverse"] = bool(
            row["log_loss_improvement"] >= 0.0
            and row["brier_improvement"] >= 0.0
            and row["auc_improvement"] >= 0.0
            and row["selective_accuracy_improvement"] >= 0.0
            and row["aligned_return_improvement"] >= 0.0
            and any(
                float(row[name]) > 0.0
                for name in (
                    "log_loss_improvement",
                    "brier_improvement",
                    "auc_improvement",
                    "selective_accuracy_improvement",
                    "aligned_return_improvement",
                )
            )
        )
        row["development_oof_increment_consistent"] = bool(
            row["development_oof_log_loss_improvement"] >= 0.0
            and row["development_oof_brier_improvement"] >= 0.0
            and row["development_oof_auc_improvement"] >= 0.0
        )
        row["adds_value"] = bool(
            row["assessment_increment_non_adverse"] and row["development_oof_increment_consistent"]
        )
        layer_flags[layer] = bool(row["adds_value"])
        rows.append(row)
    return pd.DataFrame(rows), layer_flags


def support_gates(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    primary_candidate: str,
    concentrations: Mapping[str, float],
) -> dict[str, Any]:
    action_column = f"{primary_candidate}_action"
    actions = assessment.loc[assessment[action_column].astype(str).ne("ABSTAIN")]
    development_up = int(development["direction_up_10m"].eq(1).sum())
    development_down = int(development["direction_up_10m"].eq(0).sum())
    assessment_up = int(assessment["direction_up_10m"].eq(1).sum())
    assessment_down = int(assessment["direction_up_10m"].eq(0).sum())
    expected_months = {
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
        "2025-07",
        "2025-08",
    }
    development_gates = {
        "episodes_at_least_250": len(development) >= 250,
        "sessions_at_least_60": development["session"].nunique() >= 60,
        "stocks_at_least_15": development["stock"].nunique() >= 15,
        "months_at_least_10": development["session"].str[:7].nunique() >= 10,
        "up_at_least_100": development_up >= 100,
        "down_at_least_100": development_down >= 100,
    }
    assessment_gates = {
        "episodes_at_least_250": len(assessment) >= 250,
        "sessions_at_least_50": assessment["session"].nunique() >= 50,
        "stocks_at_least_15": assessment["stock"].nunique() >= 15,
        "all_eight_month_groups": set(assessment["assessment_month_group"].astype(str).unique())
        == expected_months,
        "up_at_least_100": assessment_up >= 100,
        "down_at_least_100": assessment_down >= 100,
        "no_stock_above_15_percent": concentrations["maximum_stock_share_of_episodes"] <= 0.15,
        "no_month_above_25_percent": float(
            assessment.groupby("assessment_month_group").size().max() / len(assessment)
        )
        <= 0.25,
    }
    calls = int(actions[action_column].astype(str).eq("CALL").sum())
    puts = int(actions[action_column].astype(str).eq("PUT").sum())
    selective_gates = {
        "actions_at_least_100": len(actions) >= 100,
        "sessions_at_least_35": actions["session"].nunique() >= 35,
        "stocks_at_least_12": actions["stock"].nunique() >= 12,
        "month_groups_at_least_6": actions["assessment_month_group"].nunique() >= 6,
        "calls_at_least_35": calls >= 35,
        "puts_at_least_35": puts >= 35,
        "no_stock_above_20_percent": concentrations["maximum_stock_share_of_actions"] <= 0.20,
        "no_month_above_30_percent": concentrations["maximum_month_share_of_actions"] <= 0.30,
        "no_session_above_8_percent": concentrations["maximum_session_share_of_actions"] <= 0.08,
    }
    return {
        "development": {
            "episodes": int(len(development)),
            "sessions": int(development["session"].nunique()),
            "stocks": int(development["stock"].nunique()),
            "months": int(development["session"].str[:7].nunique()),
            "up": development_up,
            "down": development_down,
            "gates": development_gates,
            "passed": all(development_gates.values()),
        },
        "assessment": {
            "episodes": int(len(assessment)),
            "sessions": int(assessment["session"].nunique()),
            "stocks": int(assessment["stock"].nunique()),
            "months": int(assessment["assessment_month_group"].nunique()),
            "up": assessment_up,
            "down": assessment_down,
            "gates": assessment_gates,
            "passed": all(assessment_gates.values()),
        },
        "selective": {
            "actions": int(len(actions)),
            "sessions": int(actions["session"].nunique()),
            "stocks": int(actions["stock"].nunique()),
            "months": int(actions["assessment_month_group"].nunique()),
            "calls": calls,
            "puts": puts,
            "gates": selective_gates,
            "passed": all(selective_gates.values()),
        },
        "episode_support_passed": bool(
            all(development_gates.values()) and all(assessment_gates.values())
        ),
        "selective_support_passed": bool(all(selective_gates.values())),
    }


def build_gate_evidence(
    *,
    d2_available: bool,
    assessment_metrics: Mapping[str, Mapping[str, float | int]],
    primary_candidate: str,
    primary_selective: Mapping[str, float | int],
    support: Mapping[str, Any],
    bootstrap_intervals: Mapping[str, Mapping[str, float]],
    baseline_accuracies: Mapping[str, float],
    null_summary: Mapping[str, int | bool],
    late_direction_problem: bool,
    layer_flags: Mapping[str, bool],
    positive_months: int,
) -> dict[str, Any]:
    """Build the frozen pass-gate evidence from scored artifacts."""

    primary_metrics = assessment_metrics[primary_candidate]
    return {
        "blocker": None if d2_available else "blocked_missing_auditable_orientation",
        "episode_support_passed": support["episode_support_passed"],
        "selective_support_passed": support["selective_support_passed"],
        "assessment_log_loss_improves_vs_d0": float(primary_metrics["log_loss"])
        < float(assessment_metrics["D0"]["log_loss"]),
        "assessment_brier_improves_vs_d0": float(primary_metrics["brier_score"])
        < float(assessment_metrics["D0"]["brier_score"]),
        "assessment_auc": primary_metrics["auc"],
        "assessment_balanced_accuracy": primary_metrics["balanced_accuracy"],
        "action_coverage": primary_selective["action_coverage"],
        "selective_accuracy": primary_selective["directional_accuracy"],
        "mean_aligned_return_10m": primary_selective["mean_aligned_return"],
        "median_aligned_return_10m": primary_selective["median_aligned_return"],
        "bootstrap_80_accuracy_lower": bootstrap_intervals["selective_directional_accuracy"][
            "80_lower"
        ],
        "bootstrap_80_mean_return_lower": bootstrap_intervals["selective_mean_aligned_return"][
            "80_lower"
        ],
        "positive_months": positive_months,
        "beats_momentum_and_market": float(primary_selective["directional_accuracy"])
        > max(baseline_accuracies["B2"], baseline_accuracies["B4"]),
        "exceeds_all_nulls_log_loss_or_auc": null_summary["exceeds_all_five_on_log_loss_or_auc"],
        "late_direction_problem": late_direction_problem,
        "d1_adds_value": layer_flags["signed_behaviour"],
        "d2_adds_value": layer_flags["route_orientation"],
        "signed_behaviour_supported": bool(
            layer_flags["signed_behaviour"]
            and float(assessment_metrics["D1"]["auc"]) >= 0.55
            and float(assessment_metrics["D1"]["balanced_accuracy"]) > 0.52
        ),
        "directional_information_present": bool(
            float(primary_metrics["auc"]) >= 0.55
            or (
                float(primary_selective["directional_accuracy"]) >= 0.55
                and float(primary_selective["mean_aligned_return"]) > 0.0
            )
        ),
        "stability_failed": bool(
            positive_months < 6
            or bootstrap_intervals["selective_directional_accuracy"]["80_lower"] <= 0.50
            or bootstrap_intervals["selective_mean_aligned_return"]["80_lower"] < 0.0
        ),
    }


def bootstrap_intervals_from_frozen_draws(
    bootstrap_metrics: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Rebuild bootstrap intervals without drawing another resample."""

    draws = bootstrap_metrics.loc[bootstrap_metrics["record_type"].astype(str).eq("draw")]
    intervals: dict[str, dict[str, float]] = {}
    for metric, group in draws.groupby("metric", sort=True):
        values = pd.to_numeric(group["value"], errors="raise").to_numpy(float)
        intervals[str(metric)] = {}
        for confidence in (0.80, 0.90, 0.95):
            alpha = 0.5 * (1.0 - confidence)
            lower, upper = np.nanquantile(values, [alpha, 1.0 - alpha])
            label = str(int(confidence * 100))
            intervals[str(metric)][f"{label}_lower"] = float(lower)
            intervals[str(metric)][f"{label}_upper"] = float(upper)
    return intervals


def rebuild_bootstrap_from_frozen_samples(
    assessment: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    *,
    primary_candidate: str,
) -> tuple[dict[str, dict[str, float]], float, int]:
    """Rebuild draw metrics from persisted session identities without RNG."""

    identities = bootstrap_metrics.loc[
        bootstrap_metrics["record_type"].astype(str).eq("sample_identity")
    ].sort_values("draw", kind="mergesort")
    groups = {str(session): group for session, group in assessment.groupby("session", sort=False)}
    rebuilt_rows: list[dict[str, Any]] = []
    maximum_draw_difference = 0.0
    identity_mismatches = int(len(identities) != BOOTSTRAP_DRAWS)
    for identity in identities.itertuples(index=False):
        draw = int(identity.draw)
        sampled_sessions = [
            str(session)
            for session in cast(
                Sequence[object],
                json.loads(str(identity.sampled_sessions_json)),
            )
        ]
        if len(sampled_sessions) != assessment["session"].nunique() or any(
            session not in groups for session in sampled_sessions
        ):
            identity_mismatches += 1
            continue
        sample = pd.concat(
            [groups[session] for session in sampled_sessions],
            ignore_index=True,
        )
        for metric, value in bootstrap_metric_values(sample, primary_candidate).items():
            rebuilt_rows.append(
                {
                    "record_type": "draw",
                    "draw": draw,
                    "metric": metric,
                    "value": value,
                }
            )
            stored = bootstrap_metrics.loc[
                bootstrap_metrics["record_type"].astype(str).eq("draw")
                & pd.to_numeric(bootstrap_metrics["draw"], errors="coerce").eq(draw)
                & bootstrap_metrics["metric"].astype(str).eq(metric),
                "value",
            ]
            if len(stored) != 1:
                identity_mismatches += 1
            else:
                maximum_draw_difference = max(
                    maximum_draw_difference,
                    abs(value - float(stored.iloc[0])),
                )
    rebuilt_draws = pd.DataFrame(rebuilt_rows)
    return (
        bootstrap_intervals_from_frozen_draws(rebuilt_draws),
        maximum_draw_difference,
        identity_mismatches,
    )


def null_summary_from_frozen_refits(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    null_metrics: pd.DataFrame,
    thresholds: Mapping[str, Any],
    *,
    primary_candidate: str,
    assessment_metrics: Mapping[str, Mapping[str, float | int]],
    real_selective_metrics: pd.DataFrame,
) -> tuple[dict[str, int | bool], float, float, int]:
    """Refit nulls from persisted label slates without invoking the RNG."""

    specifications = feature_specifications()
    real_selective = real_selective_metrics.loc[
        real_selective_metrics["policy"].astype(str).eq("development_oof_35_percent_minimum_150")
        & pd.to_numeric(real_selective_metrics["horizon_minutes"], errors="coerce").eq(10)
    ].set_index("model")
    counts = {
        "real_exceeds_log_loss_count": 0,
        "real_exceeds_brier_count": 0,
        "real_exceeds_auc_count": 0,
        "real_exceeds_selective_accuracy_count": 0,
        "real_exceeds_mean_aligned_return_count": 0,
    }
    maximum_metric_difference = 0.0
    maximum_coefficient_difference = 0.0
    identity_mismatches = 0
    for null_index in range(len(NULL_SEEDS)):
        slate_rows = null_metrics.loc[
            pd.to_numeric(null_metrics["null_refit"], errors="coerce").eq(null_index)
        ]
        serialized = slate_rows["permuted_labels_json"].drop_duplicates()
        if len(serialized) != 1:
            identity_mismatches += 1
            continue
        raw_labels = cast(Sequence[object], json.loads(str(serialized.iloc[0])))
        if len(raw_labels) != len(development):
            identity_mismatches += 1
            continue
        permuted = development.copy()
        permuted["null_direction_up_10m"] = pd.Series(
            [np.nan if value is None else float(value) for value in raw_labels],
            index=permuted.index,
            dtype=float,
        )
        for model_id in MODEL_IDS:
            stored = slate_rows.loc[slate_rows["model"].astype(str).eq(model_id)]
            if len(stored) != 1:
                identity_mismatches += 1
                continue
            specification = specifications[model_id]
            model = fit_direction_model(
                permuted,
                target_column="null_direction_up_10m",
                numeric_features=specification["numeric"],
                categorical_features=specification["categorical"],
                model_id=f"{model_id}_null_{null_index}",
            )
            frozen_specification = cast(
                Mapping[str, Any],
                json.loads(str(stored.iloc[0]["model_specification_json"])),
            )
            maximum_coefficient_difference = max(
                maximum_coefficient_difference,
                float(
                    np.max(
                        np.abs(
                            model.coefficients
                            - np.asarray(
                                frozen_specification["coefficients"],
                                dtype=float,
                            )
                        )
                    )
                ),
                abs(model.intercept - float(frozen_specification["intercept"])),
            )
            probability = model.predict(assessment)
            null_frame = assessment.copy()
            null_frame["_null_probability"] = probability
            null_frame["_null_action"] = apply_selective_policy(
                probability,
                float(cast(Mapping[str, Any], thresholds[model_id])["boundary"]),
            )
            metrics = _safe_binary_metrics(null_frame, "_null_probability")
            selective = selective_policy_metrics(
                null_frame,
                action_column="_null_action",
                horizon_minutes=10,
            )
            rebuilt_values = {
                "log_loss": float(metrics["log_loss"]),
                "brier_score": float(metrics["brier_score"]),
                "auc": float(metrics["auc"]),
                "selective_directional_accuracy": float(selective["directional_accuracy"]),
                "selective_mean_aligned_return_10m": float(selective["mean_aligned_return"]),
            }
            for column, value in rebuilt_values.items():
                maximum_metric_difference = max(
                    maximum_metric_difference,
                    abs(value - float(stored.iloc[0][column])),
                )
            if model_id == primary_candidate:
                real = assessment_metrics[model_id]
                real_policy = real_selective.loc[model_id]
                counts["real_exceeds_log_loss_count"] += int(
                    float(real["log_loss"]) < rebuilt_values["log_loss"]
                )
                counts["real_exceeds_brier_count"] += int(
                    float(real["brier_score"]) < rebuilt_values["brier_score"]
                )
                counts["real_exceeds_auc_count"] += int(float(real["auc"]) > rebuilt_values["auc"])
                counts["real_exceeds_selective_accuracy_count"] += int(
                    float(real_policy["directional_accuracy"])
                    > rebuilt_values["selective_directional_accuracy"]
                )
                counts["real_exceeds_mean_aligned_return_count"] += int(
                    float(real_policy["mean_aligned_return"])
                    > rebuilt_values["selective_mean_aligned_return_10m"]
                )
    summary: dict[str, int | bool] = {
        "null_refits": len(NULL_SEEDS),
        **counts,
    }
    summary["exceeds_all_five_on_log_loss_or_auc"] = bool(
        summary["real_exceeds_log_loss_count"] == 5 or summary["real_exceeds_auc_count"] == 5
    )
    return (
        summary,
        maximum_metric_difference,
        maximum_coefficient_difference,
        identity_mismatches,
    )


def compare_metric_tables(
    original: pd.DataFrame,
    rebuilt: pd.DataFrame,
) -> tuple[int, float]:
    """Return structural mismatches and maximum numeric metric difference."""

    if list(original.columns) != list(rebuilt.columns) or original.shape != rebuilt.shape:
        return 1, math.inf
    structural_mismatches = 0
    maximum = 0.0
    for column in original.columns:
        left_numeric = pd.to_numeric(original[column], errors="coerce")
        right_numeric = pd.to_numeric(rebuilt[column], errors="coerce")
        numeric = left_numeric.notna() | right_numeric.notna()
        if bool(numeric.any()):
            left = left_numeric.to_numpy(float)
            right = right_numeric.to_numpy(float)
            if not np.array_equal(np.isnan(left), np.isnan(right)):
                structural_mismatches += 1
                continue
            finite = np.isfinite(left) & np.isfinite(right)
            if bool(finite.any()):
                maximum = max(maximum, float(np.max(np.abs(left[finite] - right[finite]))))
            if not np.array_equal(np.isposinf(left), np.isposinf(right)) or not np.array_equal(
                np.isneginf(left), np.isneginf(right)
            ):
                structural_mismatches += 1
            nonnumeric = left_numeric.isna() & right_numeric.isna()
            left_text = original.loc[nonnumeric, column].fillna("__NA__").astype(str).to_numpy()
            right_text = rebuilt.loc[nonnumeric, column].fillna("__NA__").astype(str).to_numpy()
            structural_mismatches += int(np.count_nonzero(left_text != right_text))
        else:
            left_text = original[column].fillna("__NA__").astype(str).to_numpy()
            right_text = rebuilt[column].fillna("__NA__").astype(str).to_numpy()
            structural_mismatches += int(np.count_nonzero(left_text != right_text))
    return structural_mismatches, maximum


def create_plots(
    assessment: pd.DataFrame,
    primary_candidate: str,
    excursion_metrics: pd.DataFrame,
) -> None:
    valid = assessment.loc[assessment["direction_up_10m"].notna()].copy()
    target = valid["direction_up_10m"].to_numpy(int)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
    colors = {"D0": "#7a7a7a", "D1": "#147d92", "D2": "#b44c43"}
    for model_id in MODEL_IDS:
        probability = valid[f"{model_id}_probability"].to_numpy(float)
        false_positive, true_positive, _ = roc_curve(target, probability)
        precision, recall, _ = precision_recall_curve(target, probability)
        calibration_true, calibration_predicted = calibration_curve(
            target, probability, n_bins=8, strategy="quantile"
        )
        axes[0].plot(
            false_positive,
            true_positive,
            label=model_id,
            color=colors[model_id],
        )
        axes[1].plot(
            recall,
            precision,
            label=model_id,
            color=colors[model_id],
        )
        axes[2].plot(
            calibration_predicted,
            calibration_true,
            marker="o",
            label=model_id,
            color=colors[model_id],
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#aaaaaa")
    axes[0].set(title="ROC", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].axhline(target.mean(), linestyle="--", color="#aaaaaa")
    axes[1].set(title="Precision–recall", xlabel="Recall", ylabel="Precision")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="#aaaaaa")
    axes[2].set(title="Calibration", xlabel="Mean predicted UP", ylabel="Observed UP")
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(alpha=0.18)
    figure.suptitle("2025 retrospective directional comparison")
    figure.tight_layout()
    figure.savefig(PRIMARY / "direction_model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    probability = assessment[f"{primary_candidate}_probability"].to_numpy(float)
    returns = assessment["signed_log_return_10m"].to_numpy(float)
    labels = assessment["direction_up_10m"].to_numpy(float)
    confidence_rows: list[dict[str, float]] = []
    for boundary in np.linspace(0.0, 0.30, 61):
        actions = apply_selective_policy(probability, float(boundary))
        actioned = actions != "ABSTAIN"
        valid_direction = actioned & np.isfinite(labels)
        side = np.where(actions == "CALL", 1.0, np.where(actions == "PUT", -1.0, np.nan))
        confidence_rows.append(
            {
                "boundary": float(boundary),
                "coverage": float(actioned.mean()),
                "accuracy": float(
                    np.mean((side[valid_direction] > 0).astype(int) == labels[valid_direction])
                )
                if valid_direction.any()
                else math.nan,
                "aligned_return": float(np.nanmean(side[actioned] * returns[actioned]))
                if actioned.any()
                else math.nan,
            }
        )
    confidence = pd.DataFrame(confidence_rows)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    axes[0].plot(confidence["boundary"], confidence["coverage"], color="#147d92")
    axes[1].plot(confidence["boundary"], confidence["accuracy"], color="#b44c43")
    axes[2].plot(confidence["boundary"], confidence["aligned_return"], color="#5c7d37")
    axes[0].set(title="Action coverage", ylabel="Fraction")
    axes[1].set(title="Directional accuracy", ylabel="Fraction")
    axes[2].set(title="Mean aligned return", ylabel="Log return")
    for axis in axes:
        axis.set_xlabel("Symmetric confidence boundary")
        axis.grid(alpha=0.18)
    figure.suptitle(f"{primary_candidate} confidence sensitivity (descriptive)")
    figure.tight_layout()
    figure.savefig(PRIMARY / "confidence_selectivity.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    actioned = assessment.loc[assessment[f"{primary_candidate}_action"].astype(str).ne("ABSTAIN")]
    horizons = np.asarray(HORIZONS, dtype=int)
    remaining = [
        (
            float(actioned[f"fraction_eventual_{horizon}m_move_after_entry"].mean())
            if horizon in (10, 30)
            else math.nan
        )
        for horizon in horizons
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(horizons.astype(str), remaining, color="#147d92")
    axes[0].axhline(0.5, linestyle="--", color="#b44c43")
    axes[0].set(
        title="Remaining movement after entry",
        xlabel="Horizon (minutes)",
        ylabel="Mean path-decomposed fraction",
    )
    axes[1].plot(
        excursion_metrics["horizon_minutes"],
        excursion_metrics["mean_favourable_excursion"],
        marker="o",
        label="Favourable",
        color="#5c7d37",
    )
    axes[1].plot(
        excursion_metrics["horizon_minutes"],
        excursion_metrics["mean_adverse_excursion"],
        marker="o",
        label="Adverse",
        color="#b44c43",
    )
    axes[1].set(
        title="Action-aligned excursion",
        xlabel="Horizon (minutes)",
        ylabel="Mean log excursion",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(
        PRIMARY / "remaining_movement_excursion.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def build_report(
    *,
    movement_audit: Mapping[str, Any],
    episode_audit: Mapping[str, Any],
    support: Mapping[str, Any],
    assessment_metrics: Mapping[str, Mapping[str, float | int]],
    layer_attribution: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    thresholds: Mapping[str, Any],
    selective_metrics: pd.DataFrame,
    material_metrics: pd.DataFrame,
    remaining_ratio: float,
    late_direction_problem: bool,
    monthly_metrics: pd.DataFrame,
    bootstrap_intervals: Mapping[str, Mapping[str, float]],
    null_summary: Mapping[str, int | bool],
    decision: Mapping[str, Any],
) -> str:
    primary_candidate = str(decision["primary_candidate"])
    primary_selective = selective_metrics.loc[
        selective_metrics["model"].eq(primary_candidate)
        & selective_metrics["policy"].eq("development_oof_35_percent_minimum_150")
        & selective_metrics["horizon_minutes"].eq(10)
    ].iloc[0]
    lines = [
        "# Movement-Qualified Directional Readiness Quick Screen V0",
        "",
        "## Claims boundary",
        "",
        "This is retrospective directional candidate evidence on underlying-stock "
        "returns. The frozen M1 movement model is unchanged and used only as an "
        "eligibility gate. No option P&L, intraday option quotes, broker access, "
        "execution claim, prospective validation, or deployable strategy claim is made.",
        "",
        "## Frozen movement gate and episodes",
        "",
        f"- M1 reconstruction passed: `{movement_audit['passed']}`.",
        f"- Frozen threshold: `{M1_THRESHOLD:.17g}`.",
        f"- Raw above-threshold checkpoint rows: "
        f"{int(episode_audit['raw_above_threshold_checkpoint_rows']):,}.",
        f"- Fresh 30-minute-spaced episodes: {int(episode_audit['fresh_episodes']):,}.",
        f"- Episodes per session: {float(episode_audit['episodes_per_session']):.4f}.",
        f"- Development support passed: `{support['development']['passed']}`; "
        f"{support['development']['episodes']} episodes, "
        f"{support['development']['up']} UP / {support['development']['down']} DOWN.",
        f"- Assessment support passed: `{support['assessment']['passed']}`; "
        f"{support['assessment']['episodes']} episodes, "
        f"{support['assessment']['up']} UP / {support['assessment']['down']} DOWN.",
        "",
        "## Direction models — 2025-01-01 through 2025-08-22",
        "",
        "| Model | Log loss | Brier | AUC | AP | Accuracy | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_IDS:
        metrics = assessment_metrics[model_id]
        lines.append(
            f"| {model_id} | {float(metrics['log_loss']):.6f} | "
            f"{float(metrics['brier_score']):.6f} | "
            f"{float(metrics['auc']):.6f} | "
            f"{float(metrics['average_precision']):.6f} | "
            f"{float(metrics['accuracy']):.6f} | "
            f"{float(metrics['balanced_accuracy']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Layer attribution",
            "",
        ]
    )
    for row in layer_attribution.itertuples(index=False):
        lines.append(
            f"- {row.comparison}: log-loss improvement "
            f"{float(row.log_loss_improvement):.6f}, Brier improvement "
            f"{float(row.brier_improvement):.6f}, AUC increment "
            f"{float(row.auc_improvement):.6f}, selective-accuracy increment "
            f"{float(row.selective_accuracy_improvement):.6f}, aligned-return "
            f"increment {float(row.aligned_return_improvement):.8f}; adds value "
            f"`{bool(row.adds_value)}`."
        )
    lines.extend(
        [
            "",
            "## Frozen CALL / PUT / ABSTAIN policy",
            "",
            f"- Primary candidate frozen before assessment scoring: `{primary_candidate}`.",
            f"- OOF confidence boundary: "
            f"{float(cast(Mapping[str, Any], thresholds[primary_candidate])['boundary']):.8f}.",
            f"- Actions: {int(primary_selective['actions'])} "
            f"({int(primary_selective['call_count'])} CALL / "
            f"{int(primary_selective['put_count'])} PUT); coverage "
            f"{float(primary_selective['action_coverage']):.2%}.",
            f"- Ten-minute directional accuracy: "
            f"{float(primary_selective['directional_accuracy']):.2%}; balanced "
            f"accuracy {float(primary_selective['balanced_accuracy']):.2%}.",
            f"- Mean / median aligned ten-minute return: "
            f"{float(primary_selective['mean_aligned_return']):.8f} / "
            f"{float(primary_selective['median_aligned_return']):.8f}.",
            f"- Positive aligned-return rate: "
            f"{float(primary_selective['positive_aligned_return_rate']):.2%}.",
            f"- Selective support passed: `{support['selective']['passed']}`.",
            "",
            "Secondary horizon results remain descriptive:",
            "",
            "| Horizon | Accuracy | Mean aligned return | Median aligned return |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in (
        selective_metrics.loc[
            selective_metrics["model"].eq(primary_candidate)
            & selective_metrics["policy"].eq("development_oof_35_percent_minimum_150")
        ]
        .sort_values("horizon_minutes")
        .itertuples(index=False)
    ):
        lines.append(
            f"| {int(row.horizon_minutes)} | "
            f"{float(row.directional_accuracy):.4f} | "
            f"{float(row.mean_aligned_return):.8f} | "
            f"{float(row.median_aligned_return):.8f} |"
        )
    baseline_summary = baseline_metrics.set_index("baseline")
    iv_row = material_metrics.loc[
        material_metrics["group"].eq("realised_iv_excess")
        & material_metrics["horizon_minutes"].eq(10)
    ]
    quartile_row = material_metrics.loc[
        material_metrics["scope"].eq("largest_absolute_movement_quartile")
        & material_metrics["horizon_minutes"].eq(10)
    ]
    positive_months = int(monthly_metrics["selective_mean_aligned_return"].gt(0.0).sum())
    lines.extend(
        [
            "",
            "## Economic-readiness diagnostics on the underlying",
            "",
            f"- Realised IV-excess ten-minute selective accuracy / mean aligned "
            f"return: {float(iv_row.iloc[0]['directional_accuracy']):.4f} / "
            f"{float(iv_row.iloc[0]['mean_aligned_return']):.8f}."
            if not iv_row.empty
            else "- Realised IV-excess ten-minute subgroup unavailable.",
            f"- Largest-movement-quartile selective accuracy / mean aligned return: "
            f"{float(quartile_row.iloc[0]['directional_accuracy']):.4f} / "
            f"{float(quartile_row.iloc[0]['mean_aligned_return']):.8f}."
            if not quartile_row.empty
            else "- Largest-movement quartile unavailable.",
            f"- Mean absolute remaining-movement fraction at ten minutes: "
            f"{remaining_ratio:.4f}; late-direction problem `{late_direction_problem}`.",
            f"- Positive mean aligned return in {positive_months} of eight assessment months.",
            f"- Recent-ten-minute momentum accuracy: "
            f"{float(baseline_summary.loc['B2', 'directional_accuracy']):.4f}; "
            f"market-direction accuracy: "
            f"{float(baseline_summary.loc['B4', 'directional_accuracy']):.4f}.",
            "",
            "## Uncertainty and nulls",
            "",
            f"- Selective accuracy 80% interval: "
            f"[{bootstrap_intervals['selective_directional_accuracy']['80_lower']:.4f}, "
            f"{bootstrap_intervals['selective_directional_accuracy']['80_upper']:.4f}].",
            f"- Mean aligned return 80% interval: "
            f"[{bootstrap_intervals['selective_mean_aligned_return']['80_lower']:.8f}, "
            f"{bootstrap_intervals['selective_mean_aligned_return']['80_upper']:.8f}].",
            f"- Real candidate exceeded nulls on log loss "
            f"{null_summary['real_exceeds_log_loss_count']}/5 and AUC "
            f"{null_summary['real_exceeds_auc_count']}/5.",
            "",
            "## Decision",
            "",
            f"Overall decision: `{decision['overall_decision']}`.",
            "",
        ]
    )
    for status_name, value in cast(Mapping[str, Any], decision["component_statuses"]).items():
        lines.append(f"- {status_name}: `{value}`.")
    lines.extend(
        [
            "",
            "Clean confirmation must occur prospectively through the live recorder. "
            "This screen does not establish option profitability or live/paper readiness.",
            "",
        ]
    )
    return "\n".join(lines)


def maximum_numeric_difference(
    left: pd.DataFrame, right: pd.DataFrame, columns: Sequence[str]
) -> float:
    maximum = 0.0
    for column in columns:
        left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(left_values), np.isnan(right_values)):
            return math.inf
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if finite.any():
            maximum = max(
                maximum,
                float(np.max(np.abs(left_values[finite] - right_values[finite]))),
            )
    return maximum


def build_core() -> dict[str, Any]:
    inputs = load_frozen_inputs()
    v0, _, panel, movement_audit = reconstruct_frozen_m1(cast(pd.DataFrame, inputs["historical"]))
    episodes, episode_audit = build_episode_panel(panel, cast(pd.DataFrame, inputs["states"]))
    features, orientation_map, orientation_source, target_audit = build_direction_features(
        episodes,
        panel,
        cast(pd.DataFrame, inputs["states"]),
        cast(pd.DataFrame, inputs["behaviour"]),
    )
    d2_available = bool(orientation_source["passed"])
    development, assessment, configurations, thresholds = fit_direction_stack(
        features,
        d2_available=d2_available,
    )
    assessment, frozen_boundaries = add_frozen_subgroups(development, assessment)
    return {
        "inputs": inputs,
        "v0": v0,
        "panel": panel,
        "movement_audit": movement_audit,
        "episodes": episodes,
        "episode_audit": episode_audit,
        "features": features,
        "orientation_map": orientation_map,
        "orientation_source": orientation_source,
        "target_audit": target_audit,
        "development": development,
        "assessment": assessment,
        "configurations": configurations,
        "thresholds": thresholds,
        "frozen_boundaries": frozen_boundaries,
        "d2_available": d2_available,
    }


def execute() -> dict[str, Any]:
    contract = load_contract()
    flags = safety_flags(contract)
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_json(PRIMARY / "contract.json", contract)

    core = build_core()
    inputs = cast(Mapping[str, Any], core["inputs"])
    v0 = cast(ModuleType, core["v0"])
    movement_audit = cast(dict[str, Any], core["movement_audit"])
    episodes = cast(pd.DataFrame, core["episodes"])
    episode_audit = cast(dict[str, Any], core["episode_audit"])
    features = cast(pd.DataFrame, core["features"])
    orientation_map = cast(pd.DataFrame, core["orientation_map"])
    orientation_source = cast(dict[str, Any], core["orientation_source"])
    target_audit = cast(dict[str, Any], core["target_audit"])
    development = cast(pd.DataFrame, core["development"])
    assessment = cast(pd.DataFrame, core["assessment"])
    configurations = cast(dict[str, Any], core["configurations"])
    thresholds = cast(dict[str, Any], core["thresholds"])
    frozen_boundaries = cast(dict[str, float], core["frozen_boundaries"])
    d2_available = bool(core["d2_available"])

    branch_c_path = cast(Path, inputs["branch_c_path"])
    state_path = cast(Path, inputs["state_path"])
    behaviour_path = cast(Path, inputs["behaviour_path"])
    source_manifest = {
        **flags,
        "development_period": [DEVELOPMENT_START, DEVELOPMENT_END],
        "retrospective_assessment_period": [ASSESSMENT_START, ASSESSMENT_END],
        "excluded_opened_movement_holdout": ["2025-09-01", "2025-12-31"],
        "protected_period_not_materialized": ["2026-01-01", "onward"],
        "sources": {
            "frozen_branch_c_panel": {
                "path": str(branch_c_path),
                "sha256": sha256_file(branch_c_path),
                "rows": int(len(cast(pd.DataFrame, inputs["historical"]))),
                "role": "frozen_M1_Group_O_plus_Group_I_checkpoint_surface",
            },
            "frozen_five_minute_state_surface": {
                "path": str(state_path),
                "sha256": sha256_file(state_path),
                "read_filter": [DEVELOPMENT_START, ASSESSMENT_END],
                "materialized_rows": int(len(cast(pd.DataFrame, inputs["states"]))),
                "role": "causal_direction_features_and_underlying_targets",
            },
            "audited_behavioural_dimensions": {
                "path": str(behaviour_path),
                "sha256": sha256_file(behaviour_path),
                "read_filter": [DEVELOPMENT_START, ASSESSMENT_END],
                "role": "sparse_audited_signed_exhaustion",
            },
            "audited_route_ledger": {
                "path": str(ROUTE_LEDGER.relative_to(REPO_ROOT)),
                "sha256": sha256_file(ROUTE_LEDGER) if ROUTE_LEDGER.is_file() else None,
                "available": ROUTE_LEDGER.is_file(),
                "role": "causal_active_prefix_and_registered_loop_orientation",
            },
            "audited_state_centroids": {
                "path": str(CENTROIDS.relative_to(REPO_ROOT)),
                "sha256": sha256_file(CENTROIDS) if CENTROIDS.is_file() else None,
                "available": CENTROIDS.is_file(),
                "role": "outcome_free_semantic_orientation_contract",
            },
        },
        "new_options_data_downloaded": False,
        "IBKR_accessed": False,
        "broker_accessed": False,
    }
    write_json(PRIMARY / "source_manifest.json", source_manifest)
    write_json(
        PRIMARY / "protected_boundary_audit.json",
        {
            **flags,
            "passed": True,
            "earliest_materialized_session": str(features["session"].astype(str).min()),
            "latest_materialized_session": str(features["session"].astype(str).max()),
            "september_through_december_2025_rows": int(
                features["session"].astype(str).between("2025-09-01", "2025-12-31").sum()
            ),
            "rows_from_2026_onward": int(features["session"].astype(str).ge("2026-01-01").sum()),
            "state_surface_predicate_pushdown_end": ASSESSMENT_END,
            "protected_rows_materialized": False,
        },
    )
    write_json(
        PRIMARY / "movement_model_reconstruction.json",
        {**flags, **movement_audit},
    )
    write_json(
        PRIMARY / "movement_feature_manifest.json",
        {
            **flags,
            "M0": {
                "numeric_features": list(v0.GROUP_O),
                "categorical_controls": ["stock"],
            },
            "M1": {
                "numeric_features": [*v0.GROUP_O, *v0.GROUP_I],
                "categorical_controls": ["stock"],
            },
            "M1_exactly_Group_O_plus_Group_I": True,
            "direction_features_added_to_M1": False,
            "M1_probability_direction_feature": False,
        },
    )
    write_parquet(PRIMARY / "movement_signal_episodes.parquet", episodes)
    episode_audit["episodes_by_partition"] = (
        episodes.groupby("partition", sort=True)
        .size()
        .rename("episodes")
        .reset_index()
        .to_dict(orient="records")
    )
    episode_audit["episodes_by_stock_and_month"] = (
        episodes.assign(month=episodes["session"].astype(str).str[:7])
        .groupby(["stock", "month"], sort=True)
        .size()
        .rename("episodes")
        .reset_index()
        .to_dict(orient="records")
    )
    episode_audit["episodes_per_stock_session_distribution"] = (
        episodes.groupby(["stock", "session"], sort=True)
        .size()
        .value_counts()
        .sort_index()
        .rename_axis("episodes_per_stock_session")
        .rename("stock_sessions")
        .reset_index()
        .to_dict(orient="records")
    )
    write_json(
        PRIMARY / "episode_construction_audit.json",
        {**flags, **episode_audit},
    )
    write_json(PRIMARY / "direction_target_audit.json", {**flags, **target_audit})
    feature_manifest = {
        **flags,
        "common_controls": {
            "categorical": list(CATEGORICAL_CONTROLS),
            "missing_indicators": "explicit_for_every_numeric_feature",
            "preprocessing_fit": "2024_development_only",
            "robust_scale": "median_and_interquartile_range",
        },
        "D0": {
            "numeric_features": list(D0_FEATURES),
            "source": "deterministic_completed_five_minute_bars",
        },
        "D1": {
            "numeric_features": [*D0_FEATURES, *D1_FEATURES],
            "incremental_features": list(D1_FEATURES),
            "unsigned_dimensions_used_only_in_signed_interactions": True,
        },
        "D2": {
            "numeric_features": [*D0_FEATURES, *D1_FEATURES, *D2_FEATURES],
            "incremental_features": list(D2_FEATURES),
            "categorical_features": [
                *CATEGORICAL_CONTROLS,
                "route_resolution_state",
            ],
            "orientation_contract": "audited_outcome_free_state_centroid_sign",
            "causally_available": d2_available,
            "status": ("available" if d2_available else "blocked_missing_auditable_orientation"),
        },
        "future_prices_as_features": False,
        "option_returns_as_features": False,
        "M1_probability_as_direction_feature": False,
        "broad_technical_indicator_search": False,
    }
    write_json(PRIMARY / "direction_feature_manifest.json", feature_manifest)
    write_json(
        PRIMARY / "orientation_source_audit.json",
        {**flags, **orientation_source},
    )
    orientation_output = orientation_map.copy()
    orientation_output["orientation_source"] = (
        "audited_outcome_free_frozen_state_centroids"
        if d2_available
        else "blocked_missing_auditable_orientation"
    )
    orientation_output["development_outcomes_used"] = False
    write_csv(PRIMARY / "development_orientation_map.csv", orientation_output)
    write_json(
        PRIMARY / "orientation_crossfit_audit.json",
        {
            **flags,
            "passed": d2_available,
            "status": (
                "not_required_outcome_free_audited_orientation"
                if d2_available
                else "blocked_missing_auditable_orientation"
            ),
            "orientation_map_fitted_from_direction_outcomes": False,
            "development_episode_outcomes_used": False,
            "assessment_outcomes_used": False,
            "cross_fitting_required": False,
            "fallback_empirical_bayes_contract": {
                "prior_equivalent_sample_size": 50,
                "minimum_raw_support": 20,
                "complete_session_blocked_folds": 4,
            },
        },
    )
    write_json(
        PRIMARY / "model_configurations.json",
        {**flags, **configurations},
    )
    write_parquet(PRIMARY / "development_oof_predictions.parquet", development)
    write_json(
        PRIMARY / "frozen_direction_thresholds.json",
        {
            **flags,
            "thresholds": thresholds,
            "written_before_assessment_scoring": True,
            "assessment_outcomes_used": False,
        },
    )

    primary_candidate = "D2" if d2_available else "D1"
    primary_freeze = {
        **flags,
        "primary_candidate": primary_candidate,
        "fallback_candidate": "D1",
        "D2_causally_available": d2_available,
        "orientation_audit_passed": d2_available,
        "selection_rule": (
            "D2 when causal audited orientation passes; otherwise D1. "
            "Assessment performance is never used to switch."
        ),
        "written_before_assessment_scoring": True,
        "assessment_metrics_inspected": False,
    }
    write_json(PRIMARY / "primary_candidate_freeze.json", primary_freeze)

    development, assessment = apply_frozen_policies(development, assessment, thresholds)
    direction_metrics, assessment_metric_map = direction_metric_tables(development, assessment)
    selective_metrics, primary_selective_map = selective_metric_table(
        assessment, thresholds, primary_candidate
    )
    development_up_rate = float(development["direction_up_10m"].dropna().mean())
    baseline_metrics, baseline_accuracies = baseline_metric_table(assessment, development_up_rate)
    monthly_metrics, checkpoint_metrics, route_metrics = stability_tables(
        assessment, primary_candidate
    )
    material_metrics = material_move_table(
        development,
        assessment,
        primary_candidate,
        frozen_boundaries,
    )
    (
        remaining_metrics,
        excursion_metrics,
        late_direction_problem,
        remaining_ratio,
    ) = remaining_and_excursion_tables(assessment, primary_candidate)
    stock_metrics, concentration_metrics, concentrations = stock_and_concentration_tables(
        assessment, primary_candidate
    )
    bootstrap_metrics, bootstrap_intervals = bootstrap_table(assessment, primary_candidate)
    null_metrics, null_summary = null_refit_table(
        development, assessment, thresholds, primary_candidate
    )
    attribution, layer_flags = layer_attribution_table(
        development, assessment_metric_map, selective_metrics
    )
    support = support_gates(development, assessment, primary_candidate, concentrations)

    write_parquet(PRIMARY / "assessment_predictions.parquet", assessment)
    write_csv(PRIMARY / "direction_model_metrics.csv", direction_metrics)
    write_csv(PRIMARY / "selective_policy_metrics.csv", selective_metrics)
    write_csv(PRIMARY / "baseline_metrics.csv", baseline_metrics)
    write_csv(PRIMARY / "monthly_metrics.csv", monthly_metrics)
    write_csv(PRIMARY / "checkpoint_metrics.csv", checkpoint_metrics)
    write_csv(PRIMARY / "route_state_metrics.csv", route_metrics)
    write_csv(PRIMARY / "stock_metrics.csv", stock_metrics)
    write_csv(PRIMARY / "material_move_metrics.csv", material_metrics)
    write_csv(PRIMARY / "remaining_movement_metrics.csv", remaining_metrics)
    write_csv(PRIMARY / "excursion_metrics.csv", excursion_metrics)
    write_csv(PRIMARY / "bootstrap_metrics.csv", bootstrap_metrics)
    write_csv(PRIMARY / "direction_null_metrics.csv", null_metrics)
    write_csv(PRIMARY / "concentration_metrics.csv", concentration_metrics)
    write_csv(PRIMARY / "layer_attribution.csv", attribution)

    primary_selective = primary_selective_map["10"]
    positive_months = int(monthly_metrics["selective_mean_aligned_return"].gt(0.0).sum())
    evidence = build_gate_evidence(
        d2_available=d2_available,
        assessment_metrics=assessment_metric_map,
        primary_candidate=primary_candidate,
        primary_selective=primary_selective,
        support=support,
        bootstrap_intervals=bootstrap_intervals,
        baseline_accuracies=baseline_accuracies,
        null_summary=null_summary,
        late_direction_problem=late_direction_problem,
        layer_flags=layer_flags,
        positive_months=positive_months,
    )
    overall_decision = decide_direction_candidate(evidence)
    d0_metrics = assessment_metric_map["D0"]
    attribution_indexed = attribution.set_index("layer")

    def layer_status(layer: str) -> str:
        row = attribution_indexed.loc[layer]
        if bool(row["adds_value"]):
            return "supported"
        if bool(row["assessment_increment_non_adverse"]) or (
            float(row["log_loss_improvement"]) > 0.0
            or float(row["auc_improvement"]) > 0.0
            or float(row["selective_accuracy_improvement"]) > 0.0
            or float(row["aligned_return_improvement"]) > 0.0
        ):
            return "promising"
        return "not_supported"

    if not support["selective_support_passed"]:
        selective_status = "insufficient_support"
    elif (
        float(primary_selective["directional_accuracy"]) >= 0.55
        and float(primary_selective["mean_aligned_return"]) > 0.0
    ):
        selective_status = "supported"
    elif float(primary_selective["mean_aligned_return"]) > 0.0:
        selective_status = "promising"
    else:
        selective_status = "not_supported"
    if float(d0_metrics["auc"]) >= 0.55 and float(d0_metrics["balanced_accuracy"]) > 0.52:
        price_status = "supported"
    elif float(d0_metrics["auc"]) > 0.50:
        price_status = "promising"
    else:
        price_status = "not_supported"
    component_statuses = {
        "movement_gate_status": "supported",
        "episode_construction_status": (
            "supported" if support["episode_support_passed"] else "insufficient_support"
        ),
        "price_direction_status": price_status,
        "signed_behaviour_status": layer_status("signed_behaviour"),
        "route_orientation_status": (
            layer_status("route_orientation") if d2_available else "blocked"
        ),
        "selective_policy_status": selective_status,
        "remaining_movement_status": (
            "supported" if not late_direction_problem else "not_supported"
        ),
        "forward_readiness_status": "not_supported",
    }
    decision = {
        **flags,
        "overall_decision": overall_decision,
        "claims_label": "retrospective directional candidate evidence",
        "primary_candidate": primary_candidate,
        "gate_evidence": evidence,
        "support_gates": support,
        "component_statuses": component_statuses,
        "late_direction_problem": late_direction_problem,
        "mean_absolute_remaining_fraction_10m": remaining_ratio,
        "null_summary": null_summary,
        "bootstrap_intervals": bootstrap_intervals,
        "frozen_subgroup_boundaries": frozen_boundaries,
        "assessment_used_for_model_or_threshold_selection": False,
        "prospective_confirmation_required": True,
    }

    create_plots(assessment, primary_candidate, excursion_metrics)
    report = build_report(
        movement_audit=movement_audit,
        episode_audit=episode_audit,
        support=support,
        assessment_metrics=assessment_metric_map,
        layer_attribution=attribution,
        baseline_metrics=baseline_metrics,
        thresholds=thresholds,
        selective_metrics=selective_metrics,
        material_metrics=material_metrics,
        remaining_ratio=remaining_ratio,
        late_direction_problem=late_direction_problem,
        monthly_metrics=monthly_metrics,
        bootstrap_intervals=bootstrap_intervals,
        null_summary=null_summary,
        decision=decision,
    )
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")
    (REPORTS / "report.md").write_text(report, encoding="utf-8")

    core_rebuilt = build_core()
    episodes_rebuilt = cast(pd.DataFrame, core_rebuilt["episodes"])
    features_rebuilt = cast(pd.DataFrame, core_rebuilt["features"])
    development_rebuilt = cast(pd.DataFrame, core_rebuilt["development"])
    assessment_rebuilt = cast(pd.DataFrame, core_rebuilt["assessment"])
    thresholds_rebuilt = cast(dict[str, Any], core_rebuilt["thresholds"])
    development_rebuilt, assessment_rebuilt = apply_frozen_policies(
        development_rebuilt, assessment_rebuilt, thresholds_rebuilt
    )
    primary_candidate_rebuilt = "D2" if bool(core_rebuilt["d2_available"]) else "D1"
    direction_metrics_rebuilt, assessment_metric_map_rebuilt = direction_metric_tables(
        development_rebuilt,
        assessment_rebuilt,
    )
    selective_metrics_rebuilt, primary_selective_map_rebuilt = selective_metric_table(
        assessment_rebuilt,
        thresholds_rebuilt,
        primary_candidate_rebuilt,
    )
    development_up_rate_rebuilt = float(development_rebuilt["direction_up_10m"].dropna().mean())
    baseline_metrics_rebuilt, baseline_accuracies_rebuilt = baseline_metric_table(
        assessment_rebuilt,
        development_up_rate_rebuilt,
    )
    (
        monthly_metrics_rebuilt,
        checkpoint_metrics_rebuilt,
        route_metrics_rebuilt,
    ) = stability_tables(assessment_rebuilt, primary_candidate_rebuilt)
    material_metrics_rebuilt = material_move_table(
        development_rebuilt,
        assessment_rebuilt,
        primary_candidate_rebuilt,
        cast(dict[str, float], core_rebuilt["frozen_boundaries"]),
    )
    (
        remaining_metrics_rebuilt,
        excursion_metrics_rebuilt,
        late_direction_problem_rebuilt,
        remaining_ratio_rebuilt,
    ) = remaining_and_excursion_tables(assessment_rebuilt, primary_candidate_rebuilt)
    (
        stock_metrics_rebuilt,
        concentration_metrics_rebuilt,
        concentrations_rebuilt,
    ) = stock_and_concentration_tables(assessment_rebuilt, primary_candidate_rebuilt)
    attribution_rebuilt, layer_flags_rebuilt = layer_attribution_table(
        development_rebuilt,
        assessment_metric_map_rebuilt,
        selective_metrics_rebuilt,
    )
    support_rebuilt = support_gates(
        development_rebuilt,
        assessment_rebuilt,
        primary_candidate_rebuilt,
        concentrations_rebuilt,
    )
    (
        bootstrap_intervals_rebuilt,
        maximum_bootstrap_draw_metric_difference,
        bootstrap_sample_identity_mismatches,
    ) = rebuild_bootstrap_from_frozen_samples(
        assessment_rebuilt,
        bootstrap_metrics,
        primary_candidate=primary_candidate_rebuilt,
    )
    (
        null_summary_rebuilt,
        maximum_null_metric_difference,
        maximum_null_coefficient_difference,
        null_slate_identity_mismatches,
    ) = null_summary_from_frozen_refits(
        development_rebuilt,
        assessment_rebuilt,
        null_metrics,
        thresholds_rebuilt,
        primary_candidate=primary_candidate_rebuilt,
        assessment_metrics=assessment_metric_map_rebuilt,
        real_selective_metrics=selective_metrics_rebuilt,
    )
    positive_months_rebuilt = int(
        monthly_metrics_rebuilt["selective_mean_aligned_return"].gt(0.0).sum()
    )
    evidence_rebuilt = build_gate_evidence(
        d2_available=bool(core_rebuilt["d2_available"]),
        assessment_metrics=assessment_metric_map_rebuilt,
        primary_candidate=primary_candidate_rebuilt,
        primary_selective=primary_selective_map_rebuilt["10"],
        support=support_rebuilt,
        bootstrap_intervals=bootstrap_intervals_rebuilt,
        baseline_accuracies=baseline_accuracies_rebuilt,
        null_summary=null_summary_rebuilt,
        late_direction_problem=late_direction_problem_rebuilt,
        layer_flags=layer_flags_rebuilt,
        positive_months=positive_months_rebuilt,
    )
    decision_rebuild = decide_direction_candidate(evidence_rebuilt)
    metric_table_pairs = (
        (direction_metrics, direction_metrics_rebuilt),
        (selective_metrics, selective_metrics_rebuilt),
        (baseline_metrics, baseline_metrics_rebuilt),
        (monthly_metrics, monthly_metrics_rebuilt),
        (checkpoint_metrics, checkpoint_metrics_rebuilt),
        (route_metrics, route_metrics_rebuilt),
        (material_metrics, material_metrics_rebuilt),
        (remaining_metrics, remaining_metrics_rebuilt),
        (excursion_metrics, excursion_metrics_rebuilt),
        (stock_metrics, stock_metrics_rebuilt),
        (concentration_metrics, concentration_metrics_rebuilt),
        (attribution, attribution_rebuilt),
    )
    metric_comparisons = [
        compare_metric_tables(original, rebuilt) for original, rebuilt in metric_table_pairs
    ]
    metric_identity_mismatches = int(sum(item[0] for item in metric_comparisons))
    maximum_metric_difference = float(max(item[1] for item in metric_comparisons))
    support_match = json.dumps(_json_safe(support_rebuilt), sort_keys=True) == json.dumps(
        _json_safe(support), sort_keys=True
    )
    bootstrap_interval_match = json.dumps(
        _json_safe(bootstrap_intervals_rebuilt), sort_keys=True
    ) == json.dumps(_json_safe(bootstrap_intervals), sort_keys=True)
    null_summary_match = json.dumps(_json_safe(null_summary_rebuilt), sort_keys=True) == json.dumps(
        _json_safe(null_summary), sort_keys=True
    )
    evidence_match = json.dumps(_json_safe(evidence_rebuilt), sort_keys=True) == json.dumps(
        _json_safe(evidence), sort_keys=True
    )
    decision_match = (
        primary_candidate_rebuilt == primary_candidate and decision_rebuild == overall_decision
    )
    identity_columns = ["stock", "session", "checkpoint", "episode_number"]
    identity_match = (
        episodes[identity_columns]
        .reset_index(drop=True)
        .equals(episodes_rebuilt[identity_columns].reset_index(drop=True))
    )
    feature_columns = [*D0_FEATURES, *D1_FEATURES, *D2_FEATURES]
    maximum_feature_difference = maximum_numeric_difference(
        features, features_rebuilt, feature_columns
    )
    maximum_probability_difference = max(
        *[
            float(
                np.max(
                    np.abs(
                        development[f"{model_id}_probability"].to_numpy(float)
                        - development_rebuilt[f"{model_id}_probability"].to_numpy(float)
                    )
                )
            )
            for model_id in MODEL_IDS
        ],
        *[
            float(
                np.max(
                    np.abs(
                        assessment[f"{model_id}_probability"].to_numpy(float)
                        - assessment_rebuilt[f"{model_id}_probability"].to_numpy(float)
                    )
                )
            )
            for model_id in MODEL_IDS
        ],
    )
    action_decision_mismatches = int(
        sum(
            np.count_nonzero(
                assessment[f"{model_id}_action"].astype(str).to_numpy()
                != assessment_rebuilt[f"{model_id}_action"].astype(str).to_numpy()
            )
            for model_id in MODEL_IDS
        )
    )
    target_columns = [f"signed_log_return_{horizon}m" for horizon in HORIZONS]
    maximum_target_difference = maximum_numeric_difference(
        features, features_rebuilt, target_columns
    )
    aligned_columns = [
        f"{model_id}_aligned_return_{horizon}m" for model_id in MODEL_IDS for horizon in HORIZONS
    ]
    maximum_aligned_return_difference = maximum_numeric_difference(
        assessment, assessment_rebuilt, aligned_columns
    )
    maximum_threshold_difference = max(
        abs(
            float(cast(Mapping[str, Any], thresholds[model_id])["boundary"])
            - float(cast(Mapping[str, Any], thresholds_rebuilt[model_id])["boundary"])
        )
        for model_id in MODEL_IDS
    )
    determinism = {
        **flags,
        "passed": bool(
            identity_match
            and maximum_feature_difference <= 1e-12
            and maximum_probability_difference <= 1e-12
            and action_decision_mismatches == 0
            and maximum_target_difference <= 1e-12
            and maximum_aligned_return_difference <= 1e-12
            and maximum_threshold_difference <= 1e-12
            and metric_identity_mismatches == 0
            and maximum_metric_difference <= 1e-12
            and support_match
            and bootstrap_interval_match
            and maximum_bootstrap_draw_metric_difference <= 1e-12
            and bootstrap_sample_identity_mismatches == 0
            and null_summary_match
            and maximum_null_metric_difference <= 1e-12
            and maximum_null_coefficient_difference <= 1e-12
            and null_slate_identity_mismatches == 0
            and evidence_match
            and decision_match
            and abs(remaining_ratio - remaining_ratio_rebuilt) <= 1e-12
        ),
        "episode_identity_mismatches": 0 if identity_match else len(episodes),
        "maximum_feature_difference": maximum_feature_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "action_decision_mismatches": action_decision_mismatches,
        "maximum_target_difference": maximum_target_difference,
        "maximum_aligned_return_difference": maximum_aligned_return_difference,
        "maximum_direction_threshold_difference": maximum_threshold_difference,
        "metric_identity_mismatches": metric_identity_mismatches,
        "maximum_metric_difference": maximum_metric_difference,
        "support_rebuild_match": support_match,
        "bootstrap_interval_rebuild_match": bootstrap_interval_match,
        "maximum_bootstrap_draw_metric_difference": (maximum_bootstrap_draw_metric_difference),
        "bootstrap_sample_identity_mismatches": bootstrap_sample_identity_mismatches,
        "null_summary_rebuild_match": null_summary_match,
        "maximum_null_metric_difference": maximum_null_metric_difference,
        "maximum_null_coefficient_difference": maximum_null_coefficient_difference,
        "null_slate_identity_mismatches": null_slate_identity_mismatches,
        "gate_evidence_rebuild_match": evidence_match,
        "maximum_remaining_movement_ratio_difference": abs(
            remaining_ratio - remaining_ratio_rebuilt
        ),
        "bootstrap_draws_redrawn": False,
        "null_samples_redrawn": False,
        "frozen_bootstrap_seed": BOOTSTRAP_SEED,
        "frozen_null_seeds": list(NULL_SEEDS),
        "metrics_rebuilt_from_frozen_predictions": True,
        "decision_rebuild": decision_rebuild,
        "decision_match": decision_match,
    }
    if not determinism["passed"]:
        decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
        decision["determinism_failure"] = determinism
    write_json(PRIMARY / "determinism_check.json", determinism)
    write_json(PRIMARY / "decision.json", decision)
    write_json(
        PRIMARY / "lightweight_audit.json",
        {
            **flags,
            "passed": False,
            "status": "pending_independent_auditor",
            "runner_self_checks_passed": bool(determinism["passed"]),
            "independent_probability_reconstructions": 0,
        },
    )
    return {
        "decision": decision,
        "episodes": episode_audit,
        "support": support,
        "determinism": determinism,
    }


def write_blocked_decision(error: ScreenBlocked) -> None:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    write_json(
        PRIMARY / "decision.json",
        {
            **safety_flags(contract),
            "overall_decision": error.decision,
            "blocker": error.detail,
            "component_statuses": {
                "movement_gate_status": "blocked",
                "episode_construction_status": "blocked",
                "price_direction_status": "blocked",
                "signed_behaviour_status": "blocked",
                "route_orientation_status": "blocked",
                "selective_policy_status": "blocked",
                "remaining_movement_status": "blocked",
                "forward_readiness_status": "blocked",
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="print the compact machine-readable run summary",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        result = execute()
    except ScreenBlocked as error:
        write_blocked_decision(error)
        print(f"{error.decision}: {error.detail}", file=sys.stderr)
        return 2
    if arguments.print_summary:
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
