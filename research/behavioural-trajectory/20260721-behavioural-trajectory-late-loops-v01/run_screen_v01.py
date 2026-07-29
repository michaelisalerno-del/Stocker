#!/usr/bin/env python3
"""Run Behavioural-Trajectory Funnel V0.1 under the fixed quick-screen limits."""

from __future__ import annotations

# ruff: noqa: E402 -- numerical thread limits must be set before numerical imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-behavioural-trajectory-v01-mpl")

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.behavioural_state_dimensions_v0 import (
    RobustComponentScale,
    apply_component_scaling,
    bar_component_frame,
    derive_behavioural_dimensions,
    derive_exhaustion_inputs,
    fit_component_scaling,
    opening_raw_components,
)
from stocker_research.behavioural_trajectory_late_loops_v01 import (
    ANCHORS_BY_CHECKPOINT,
    SCREEN_SCOPES,
    TRAJECTORY_INTERACTION_FEATURES,
    build_trajectory_regime_interactions,
    decide_quick_screen,
    late_loop_subgroup,
    map_six_bar_structural_target,
    permute_trajectory_bundle_within_slates,
    phase_label,
    reject_protected_dates,
    session_block_bootstrap_draws,
    structural_history_controls,
    trajectory_feature_values,
)
from stocker_research.emotion_regime_coarse_loop_funnel_v0 import (
    INTERACTION_FEATURES,
    STATE_PROBABILITY_FEATURES,
    build_interactions,
    multiclass_brier,
    prediction_entropy,
)
from stocker_research.loop_prefix_automaton_v2 import FirstNextLoopEventEngine

CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
AUDITOR_PATH = EXPERIMENT_DIR / "audit_screen_v01.py"

COARSE_DIR = (
    REPO_ROOT / "research" / "loop-funnel" / "20260721-emotion-regime-coarse-loop-family-v0"
)
COARSE_PRIMARY = COARSE_DIR / "artifacts" / "primary"
COARSE_PANEL = COARSE_PRIMARY / "decision_panel.parquet"
COARSE_SOURCE = COARSE_PRIMARY / "source_manifest.json"
COARSE_AUDIT = COARSE_PRIMARY / "lightweight_audit.json"
COARSE_DETERMINISM = COARSE_PRIMARY / "determinism_check.json"
COARSE_DECISION = COARSE_PRIMARY / "decision.json"
COARSE_RUNNER = COARSE_DIR / "run_screen_v0.py"

OBSERVABLE_DIR = (
    REPO_ROOT
    / "research"
    / "observable-behavioural-state"
    / "20260721-behavioural-state-dimensions-screen-v0"
)
OBSERVABLE_PRIMARY = OBSERVABLE_DIR / "artifacts" / "primary"
OBSERVABLE_SCALING = OBSERVABLE_PRIMARY / "behavioural_component_scaling.json"
OBSERVABLE_AUDIT = OBSERVABLE_PRIMARY / "independent_audit.json"
OBSERVABLE_RUNNER = OBSERVABLE_DIR / "run_screen_v0.py"
OBSERVABLE_COMPONENT_LEDGER = OBSERVABLE_PRIMARY / "behavioural_component_ledger.parquet"
EXPECTED_COMPONENT_LEDGER_SHA256 = (
    "96fad90eeeb6edf8b2f82ba809fb6939ad3ce5b7d5bf8193abacd90f471a91cb"
)

BLOCKED_DIR = (
    REPO_ROOT
    / "research"
    / "behavioural-trajectory"
    / "20260721-behavioural-trajectory-regime-funnel-quick-v0"
)
BLOCKED_PRIMARY = BLOCKED_DIR / "artifacts" / "primary"
BLOCKED_DECISION = BLOCKED_PRIMARY / "decision.json"
BLOCKED_ANCHORS = BLOCKED_PRIMARY / "trajectory_anchor_manifest.json"

SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "corrected_even_trajectory_anchors": True,
    "opening_and_later_session_checkpoints": True,
    "late_no_open_loop_subgroup": True,
    "structural_outcomes_only": True,
    "economic_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
TARGET_CLASSES = (
    "REGISTERED_COMPLETION",
    "UNREGISTERED_LOOP",
    "NO_REGISTERED_COMPLETION",
)
CHECKPOINTS = (6, 12, 24, 36)
BEHAVIOURS = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "signed_exhaustion",
)
PRIMARY_FORMS = ("change", "acceleration", "reversal")
DESCRIPTIVE_FORMS = ("recent_change", "monotonic_persistence", "peak_displacement")
PRIMARY_TRAJECTORY_FEATURES = tuple(
    f"{behaviour}_{form}" for behaviour in BEHAVIOURS for form in PRIMARY_FORMS
)
ALL_TRAJECTORY_FEATURES = tuple(
    f"{behaviour}_{form}"
    for behaviour in BEHAVIOURS
    for form in (*PRIMARY_FORMS, *DESCRIPTIVE_FORMS)
)
BASE_COMPONENTS = (
    "activity_effort",
    "range_effort",
    "travel_effort",
    "signed_progress",
    "absolute_progress",
    "signed_efficiency",
    "absolute_efficiency",
    "close_retention",
    "directional_persistence",
    "new_high_fraction",
    "new_low_fraction",
    "up_extreme_rejection",
    "down_extreme_rejection",
    "extreme_rejection",
    "compression",
    "normalised_high_slope",
    "normalised_low_slope",
    "boundary_slope",
    "activity_acceleration",
    "range_acceleration",
    "effort_acceleration",
    "signed_progress_acceleration",
    "return_gap",
    "activity_gap",
    "range_gap",
    "mean_close_location",
)
DERIVED_COMPONENTS = ("aligned_progress_acceleration", "directional_rejection")
STRUCTURAL_CONTROL_FEATURES = (
    "registered_completion_count_before_decision",
    "bars_since_last_registered_completion",
    "bars_since_last_registered_completion_missing",
    "active_registered_prefix_count_at_decision",
)
CHECKPOINT_FEATURES = ("checkpoint_12", "checkpoint_24", "checkpoint_36")
REGIME_FEATURES = (
    *STATE_PROBABILITY_FEATURES,
    "posterior_entropy",
    "top_state_probability",
    "top_second_margin",
    "expected_state_age",
    "persistence_probability",
    "transition_probability",
    "remaining_session_bars",
)
T0_FEATURES = (
    *REGIME_FEATURES,
    *CHECKPOINT_FEATURES,
    *STRUCTURAL_CONTROL_FEATURES,
    *BEHAVIOURS,
    *INTERACTION_FEATURES,
)
T1_FEATURES = (*T0_FEATURES, *PRIMARY_TRAJECTORY_FEATURES)
T2_FEATURES = (*T1_FEATURES, *TRAJECTORY_INTERACTION_FEATURES)
MODEL_FEATURES = {"T0": T0_FEATURES, "T1": T1_FEATURES, "T2": T2_FEATURES}
MODEL_COMPARISONS = (("T0", "T1"), ("T1", "T2"))

START = pd.Timestamp("2024-01-01T00:00:00Z")
DEVELOPMENT_END = pd.Timestamp("2025-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
EXPECTED_SESSION_BARS = 78
MAX_TARGET_BAR_ORDINAL = 41
MAX_ROWS = 35_000
MODEL_SEED = 20260721
BOOTSTRAP_SEED = 20260722
NULL_SEED = 20260723
BOOTSTRAP_DRAWS = 25
NULL_DRAWS = 5


class ScreenBlocker(RuntimeError):
    """Fail-closed blocker with one preregistered decision code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_script(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ScreenBlocker("blocked_predecessor_population_not_reconstructable", str(path))
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_contract() -> dict[str, Any]:
    contract = cast(dict[str, Any], json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected or contract.get("safety", {}).get(key) != expected:
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure", f"contract safety flag differs: {key}"
            )
    return contract


def verify_predecessors() -> dict[str, Any]:
    required = (
        COARSE_PANEL,
        COARSE_SOURCE,
        COARSE_AUDIT,
        COARSE_DETERMINISM,
        COARSE_DECISION,
        OBSERVABLE_SCALING,
        OBSERVABLE_AUDIT,
        BLOCKED_DECISION,
        BLOCKED_ANCHORS,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ScreenBlocker(
            "blocked_predecessor_population_not_reconstructable",
            f"predecessor artifacts missing: {missing}",
        )
    coarse_decision = json.loads(COARSE_DECISION.read_text(encoding="utf-8"))
    blocked_decision = json.loads(BLOCKED_DECISION.read_text(encoding="utf-8"))
    coarse_audit = json.loads(COARSE_AUDIT.read_text(encoding="utf-8"))
    coarse_determinism = json.loads(COARSE_DETERMINISM.read_text(encoding="utf-8"))
    if coarse_decision.get("decision") != "descriptive_coarse_funnel_only":
        raise ScreenBlocker(
            "blocked_predecessor_population_not_reconstructable",
            "coarse predecessor decision differs",
        )
    if blocked_decision.get("decision") != "blocked_insufficient_trajectory_support":
        raise ScreenBlocker(
            "blocked_predecessor_population_not_reconstructable",
            "blocked trajectory predecessor decision differs",
        )
    if not coarse_audit.get("passed") or not coarse_determinism.get("passed"):
        raise ScreenBlocker(
            "blocked_predecessor_population_not_reconstructable",
            "coarse predecessor audit or determinism did not pass",
        )
    return {
        **SAFETY_FLAGS,
        "experiments": [
            "Observable Behavioural-State Dimensions Screen V0",
            "Emotion × Regime-Mix Coarse Loop-Family Funnel V0",
            "Behavioural-Trajectory × Regime-Mix Funnel Quick Screen V0",
        ],
        "files": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
            for path in required
        ],
        "coarse_decision": coarse_decision["decision"],
        "blocked_v0_decision": blocked_decision["decision"],
    }


def resolve_frozen_component_ledger() -> Path:
    """Find the already-local materialised LFS artifact without network access."""

    candidates = [OBSERVABLE_COMPONENT_LEDGER]
    frozen_root = os.environ.get("STOCKER_FROZEN_REPO_ROOT")
    if frozen_root:
        candidates.append(
            Path(frozen_root)
            / "research"
            / "observable-behavioural-state"
            / "20260721-behavioural-state-dimensions-screen-v0"
            / "artifacts"
            / "primary"
            / "behavioural_component_ledger.parquet"
        )
    for sibling in sorted(REPO_ROOT.parent.glob("2026-07-21-you-are-working-in-the-github-*")):
        candidates.append(
            sibling
            / "research"
            / "observable-behavioural-state"
            / "20260721-behavioural-state-dimensions-screen-v0"
            / "artifacts"
            / "primary"
            / "behavioural_component_ledger.parquet"
        )
    for candidate in candidates:
        if (
            candidate.is_file()
            and candidate.stat().st_size > 1_000_000
            and sha256_file(candidate) == EXPECTED_COMPONENT_LEDGER_SHA256
        ):
            return candidate.resolve()
    raise ScreenBlocker(
        "blocked_predecessor_population_not_reconstructable",
        "the frozen behavioural component ledger is not materialised in any local checkout",
    )


def predecessor_population() -> tuple[pd.DataFrame, dict[str, Any]]:
    coarse = pd.read_parquet(COARSE_PANEL)
    reject_protected_dates(coarse)
    opening = coarse.loc[coarse["decision_ordinal"].isin((6, 12))].copy()
    opening = opening.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    if len(opening) != len(coarse) or set(opening["symbol"].astype(str)) == set():
        raise ScreenBlocker(
            "blocked_predecessor_population_not_reconstructable",
            "coarse panel is not the frozen two-checkpoint population",
        )
    pairs = opening.loc[:, ["symbol", "session"]].drop_duplicates().reset_index(drop=True)
    later_parts: list[pd.DataFrame] = []
    for checkpoint in (24, 36):
        part = pairs.copy()
        part["decision_ordinal"] = checkpoint
        later_parts.append(part)
    later = pd.concat(later_parts, ignore_index=True)
    keys = pd.concat(
        [opening.loc[:, ["symbol", "session", "decision_ordinal"]], later],
        ignore_index=True,
    ).drop_duplicates()
    keys["repo_bar_start_ordinal"] = keys["decision_ordinal"].astype(int) - 1
    keys["year"] = keys["session"].astype(str).str[:4].astype(int)
    keys["year_month"] = keys["session"].astype(str).str[:7]
    keys["slate_id"] = (
        keys["session"].astype(str)
        + "|"
        + keys["decision_ordinal"].astype(int).map(lambda value: f"{value:02d}")
    )
    keys = keys.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    if len(keys) > MAX_ROWS:
        raise ScreenBlocker(
            "blocked_quick_trajectory_late_loop_resource_limit",
            f"combined decision population exceeds {MAX_ROWS}: {len(keys)}",
        )
    manifest = {
        **SAFETY_FLAGS,
        "opening_rows_exact_predecessor": len(opening),
        "unique_stock_sessions": len(pairs),
        "combined_rows_before_trajectory_availability": len(keys),
        "rows_by_checkpoint": {
            str(key): int(value)
            for key, value in keys.groupby("decision_ordinal", sort=True).size().items()
        },
        "stocks": sorted(keys["symbol"].astype(str).unique().tolist()),
        "minimum_session": str(keys["session"].min()),
        "maximum_session": str(keys["session"].max()),
        "protected_rows_materialised": 0,
        "opening_population_sha256": sha256_file(COARSE_PANEL),
    }
    return keys, manifest


def attach_state_features(
    population: pd.DataFrame,
    states: pd.DataFrame,
    coarse_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    state_columns = [
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "causal_hard_state",
        "expected_state_age",
        "transition_probability",
        "persistence_probability",
        "posterior_entropy_reproduced",
        *STATE_PROBABILITY_FEATURES,
    ]
    decision_states = states.loc[:, state_columns].copy()
    panel = population.merge(
        decision_states,
        left_on=["symbol", "session", "repo_bar_start_ordinal"],
        right_on=["symbol", "session", "bar_ordinal"],
        how="left",
        validate="one_to_one",
    )
    if panel["bar_complete_timestamp"].isna().any():
        raise ScreenBlocker(
            "blocked_predecessor_population_not_reconstructable",
            "causal V2 state is unavailable for a combined-clock decision",
        )
    panel["feature_available_timestamp_utc"] = pd.to_datetime(
        panel["bar_complete_timestamp"], utc=True, errors="raise"
    )
    panel["decision_bar_start_timestamp_utc"] = pd.to_datetime(
        panel["bar_start_timestamp"], utc=True, errors="raise"
    )
    panel["decision_time_america_new_york"] = (
        panel["feature_available_timestamp_utc"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    expected_clocks = {6: "10:00", 12: "10:30", 24: "11:30", 36: "12:30"}
    clocks = panel.groupby("decision_ordinal", sort=True)["decision_time_america_new_york"].unique()
    for checkpoint, expected in expected_clocks.items():
        if clocks.loc[checkpoint].tolist() != [expected]:
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure",
                f"checkpoint {checkpoint} local clock differs",
            )
    panel["posterior_entropy"] = panel["posterior_entropy_reproduced"].astype(float)
    probabilities = panel.loc[:, list(STATE_PROBABILITY_FEATURES)].to_numpy(dtype=float)
    ordered = np.sort(probabilities, axis=1)
    panel["top_state_probability"] = ordered[:, -1]
    panel["top_second_margin"] = ordered[:, -1] - ordered[:, -2]
    panel["remaining_session_bars"] = EXPECTED_SESSION_BARS - panel["decision_ordinal"].astype(int)
    for checkpoint in (12, 24, 36):
        panel[f"checkpoint_{checkpoint}"] = panel["decision_ordinal"].eq(checkpoint).astype(float)
    frozen_columns = [
        *STATE_PROBABILITY_FEATURES,
        "posterior_entropy",
        "top_state_probability",
        "top_second_margin",
        "expected_state_age",
        "persistence_probability",
        "transition_probability",
        "remaining_session_bars",
    ]
    opening = panel.loc[panel["decision_ordinal"].isin((6, 12))]
    authority = coarse_panel.loc[:, ["symbol", "session", "decision_ordinal", *frozen_columns]]
    comparison = opening.merge(
        authority,
        on=["symbol", "session", "decision_ordinal"],
        suffixes=("", "_frozen"),
        validate="one_to_one",
    )
    differences = [
        np.max(np.abs(comparison[column] - comparison[f"{column}_frozen"]))
        for column in frozen_columns
    ]
    maximum_difference = float(max(differences, default=0.0))
    if maximum_difference > 1e-12 or len(comparison) != len(opening):
        raise ScreenBlocker(
            "blocked_predecessor_population_not_reconstructable",
            f"opening V2 reconstruction differs by {maximum_difference}",
        )
    return panel, {
        **SAFETY_FLAGS,
        "checkpoint_clocks_america_new_york": expected_clocks,
        "maximum_opening_state_feature_difference": maximum_difference,
        "passed": True,
    }


def generalized_range_baselines(
    bars: pd.DataFrame,
    anchors: Sequence[int],
) -> dict[tuple[str, int], float]:
    rows: list[dict[str, Any]] = []
    for session, session_frame in bars.groupby("session", sort=True):
        ordered = session_frame.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
        for anchor in anchors:
            prefix = ordered.iloc[: int(anchor)]
            opening_range = (
                10_000.0
                * (float(prefix["high"].max()) - float(prefix["low"].min()))
                / float(prefix.iloc[0]["open"])
            )
            rows.append(
                {"session": str(session), "anchor_ordinal": int(anchor), "range_bps": opening_range}
            )
    frame = pd.DataFrame(rows).sort_values(["anchor_ordinal", "session"], kind="mergesort")
    frame["trailing_median"] = frame.groupby("anchor_ordinal", sort=False)["range_bps"].transform(
        lambda values: values.expanding(min_periods=1).median().shift(1)
    )
    return {
        (str(row.session), int(row.anchor_ordinal)): float(row.trailing_median)
        for row in frame.itertuples(index=False)
        if np.isfinite(float(row.trailing_median)) and float(row.trailing_median) > 0.0
    }


def _cross_sectional_gap(values: np.ndarray, *, leave_one_out: bool) -> np.ndarray:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("cross-sectional behavioural inputs require two finite stocks")
    if leave_one_out:
        return np.asarray(
            [value - np.median(np.delete(values, index)) for index, value in enumerate(values)],
            dtype=float,
        )
    return values - float(np.median(values))


def build_behavioural_anchors(
    population: pd.DataFrame,
    *,
    provider_root: Path,
    observable: ModuleType,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Build causal raw components, anchor-specific scaling, levels, and trajectories."""

    anchors = tuple(
        sorted({anchor for triplet in ANCHORS_BY_CHECKPOINT.values() for anchor in triplet})
    )
    records: list[dict[str, Any]] = []
    missingness: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    minimum_timestamp: pd.Timestamp | None = None
    maximum_timestamp: pd.Timestamp | None = None
    for symbol in sorted(population["symbol"].astype(str).unique()):
        path = observable.provider_path(provider_root, symbol)
        raw = observable.bounded_source(path)
        if bool(pd.to_datetime(raw["timestamp"], utc=True).ge(PROTECTED_START).any()):
            raise ScreenBlocker(
                "blocked_protected_boundary_failure", f"protected raw row materialised for {symbol}"
            )
        bars, gaps = observable.prepare_symbol_bars(raw, symbol=symbol)
        sessions = {
            str(session): part.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
            for session, part in bars.groupby("session", sort=True)
        }
        baselines = generalized_range_baselines(bars, anchors)
        source_records.append(
            {
                "symbol": symbol,
                "logical_path": str(path),
                "bounded_rows": len(raw),
                "complete_session_gaps": len(gaps),
            }
        )
        raw_minimum = pd.Timestamp(raw["timestamp"].min())
        raw_maximum = pd.Timestamp(raw["timestamp"].max())
        minimum_timestamp = (
            raw_minimum if minimum_timestamp is None else min(minimum_timestamp, raw_minimum)
        )
        maximum_timestamp = (
            raw_maximum if maximum_timestamp is None else max(maximum_timestamp, raw_maximum)
        )
        requested = population.loc[population["symbol"].eq(symbol)]
        for decision in requested.itertuples(index=False):
            checkpoint = int(decision.decision_ordinal)
            session = str(decision.session)
            for anchor_role, anchor in zip(
                ("E0", "E1", "E2"), ANCHORS_BY_CHECKPOINT[checkpoint], strict=True
            ):
                if session not in sessions or (session, anchor) not in baselines:
                    missingness.append(
                        {
                            "symbol": symbol,
                            "session": session,
                            "month": session[:7],
                            "decision_ordinal": checkpoint,
                            "anchor_ordinal": anchor,
                            "anchor_role": anchor_role,
                            "emotion": "ALL",
                            "reason": "causal_anchor_source_unavailable",
                        }
                    )
                    continue
                prefix = sessions[session].iloc[:anchor].copy()
                activity = prefix["historical_relative_activity"].to_numpy(dtype=float)
                if len(prefix) != anchor or not np.isfinite(activity).all():
                    missingness.append(
                        {
                            "symbol": symbol,
                            "session": session,
                            "month": session[:7],
                            "decision_ordinal": checkpoint,
                            "anchor_ordinal": anchor,
                            "anchor_role": anchor_role,
                            "emotion": "ALL",
                            "reason": "causal_historical_activity_unavailable",
                        }
                    )
                    continue
                component_bars = bar_component_frame(prefix)
                calculated = opening_raw_components(
                    component_bars,
                    trailing_opening_range_median_bps=baselines[(session, anchor)],
                    signed_progress_bps=0.0,
                    signed_progress_acceleration_bps=0.0,
                    return_gap_bps=0.0,
                )
                returns = component_bars["return_bps"].to_numpy(dtype=float)
                half = anchor // 2
                latest_complete = pd.Timestamp(prefix.iloc[-1]["bar_complete_timestamp"])
                decision_available = pd.Timestamp(decision.feature_available_timestamp_utc)
                if latest_complete > decision_available:
                    raise ScreenBlocker(
                        "blocked_behavioural_trajectory_not_causal",
                        f"future bar used for {symbol}/{session}/{checkpoint}/{anchor}",
                    )
                records.append(
                    {
                        "symbol": symbol,
                        "session": session,
                        "year": int(decision.year),
                        "year_month": str(decision.year_month),
                        "decision_ordinal": checkpoint,
                        "slate_id": str(decision.slate_id),
                        "anchor_ordinal": anchor,
                        "anchor_role": anchor_role,
                        "anchor_available_timestamp_utc": latest_complete,
                        "decision_available_timestamp_utc": decision_available,
                        "latest_input_bar_complete_timestamp_utc": latest_complete,
                        "completed_bar_count": anchor,
                        "calculated_open_to_anchor_return_bps": 10_000.0
                        * (
                            float(component_bars.iloc[-1]["close"])
                            / float(component_bars.iloc[0]["open"])
                            - 1.0
                        ),
                        "earlier_half_return_bps": float(returns[:half].sum()),
                        "recent_half_return_bps": float(returns[half:].sum()),
                        **calculated,
                    }
                )
    raw_components = pd.DataFrame(records)
    if raw_components.empty:
        raise ScreenBlocker(
            "blocked_insufficient_trajectory_support", "no behavioural anchor was available"
        )
    raw_components["anchor_slate_id"] = (
        raw_components["slate_id"]
        + "|A"
        + raw_components["anchor_ordinal"].astype(int).map(lambda value: f"{value:02d}")
    )
    for _, indices in raw_components.groupby("anchor_slate_id", sort=True).groups.items():
        index = list(indices)
        open_return = raw_components.loc[index, "calculated_open_to_anchor_return_bps"].to_numpy(
            dtype=float
        )
        earlier = raw_components.loc[index, "earlier_half_return_bps"].to_numpy(dtype=float)
        recent = raw_components.loc[index, "recent_half_return_bps"].to_numpy(dtype=float)
        activity = raw_components.loc[index, "activity_effort"].to_numpy(dtype=float)
        range_effort = raw_components.loc[index, "range_effort"].to_numpy(dtype=float)
        signed = _cross_sectional_gap(open_return, leave_one_out=True)
        raw_components.loc[index, "signed_progress"] = signed
        raw_components.loc[index, "absolute_progress"] = np.abs(signed)
        raw_components.loc[index, "return_gap"] = _cross_sectional_gap(
            open_return, leave_one_out=False
        )
        raw_components.loc[index, "signed_progress_acceleration"] = _cross_sectional_gap(
            recent, leave_one_out=False
        ) - _cross_sectional_gap(earlier, leave_one_out=False)
        raw_components.loc[index, "activity_gap"] = _cross_sectional_gap(
            activity, leave_one_out=False
        )
        raw_components.loc[index, "range_gap"] = _cross_sectional_gap(
            range_effort, leave_one_out=False
        )
    frozen_component_path = resolve_frozen_component_ledger()
    frozen_components = pd.read_parquet(
        frozen_component_path,
        columns=["symbol", "session", "decision_ordinal", *BASE_COMPONENTS],
    )
    final_mask = raw_components["anchor_role"].eq("E2") & raw_components["decision_ordinal"].isin(
        (6, 12)
    )
    final_rows = raw_components.loc[
        final_mask, ["symbol", "session", "decision_ordinal", *BASE_COMPONENTS]
    ]
    frozen_join = final_rows.merge(
        frozen_components,
        on=["symbol", "session", "decision_ordinal"],
        suffixes=("_reconstructed", "_frozen"),
        how="inner",
        validate="one_to_one",
    )
    if len(frozen_join) != int(final_mask.sum()):
        raise ScreenBlocker(
            "blocked_predecessor_population_not_reconstructable",
            "frozen final-anchor component population differs",
        )
    raw_component_differences = {
        component: float(
            np.max(
                np.abs(
                    frozen_join[f"{component}_reconstructed"] - frozen_join[f"{component}_frozen"]
                )
            )
        )
        for component in BASE_COMPONENTS
    }
    frozen_indexed = frozen_components.set_index(
        ["symbol", "session", "decision_ordinal"], verify_integrity=True
    )
    final_index = pd.MultiIndex.from_frame(
        raw_components.loc[final_mask, ["symbol", "session", "decision_ordinal"]]
    )
    for component in BASE_COMPONENTS:
        raw_components.loc[final_mask, component] = frozen_indexed.loc[
            final_index, component
        ].to_numpy(dtype=float)
    raw_components["scale_group"] = raw_components["decision_ordinal"].astype(
        int
    ) * 100 + raw_components["anchor_ordinal"].astype(int)
    development = raw_components["year"].eq(2024)
    base_scaling = fit_component_scaling(
        raw_components.loc[development],
        components=BASE_COMPONENTS,
        checkpoint_column="scale_group",
    )
    scaled = apply_component_scaling(
        raw_components,
        base_scaling,
        components=BASE_COMPONENTS,
        checkpoint_column="scale_group",
    )
    scaled["signed_pressure"] = scaled[
        ["z_signed_progress", "z_signed_efficiency", "z_mean_close_location", "z_boundary_slope"]
    ].mean(axis=1)
    scaled = derive_exhaustion_inputs(scaled)
    derived_scaling = fit_component_scaling(
        scaled.loc[development],
        components=DERIVED_COMPONENTS,
        checkpoint_column="scale_group",
    )
    scaled = apply_component_scaling(
        scaled,
        derived_scaling,
        components=DERIVED_COMPONENTS,
        checkpoint_column="scale_group",
    )
    dimensions = derive_behavioural_dimensions(scaled)
    for behaviour in BEHAVIOURS:
        scaled[behaviour] = dimensions[behaviour]
    scaling_manifest = serialize_anchor_scaling(base_scaling, derived_scaling)
    reproduction = reproduce_frozen_final_anchors(scaled, scaling_manifest)
    key_columns = ["symbol", "session", "decision_ordinal"]
    trajectory_rows: list[dict[str, Any]] = []
    for key, rows in scaled.groupby(key_columns, sort=True):
        ordered = rows.set_index("anchor_role")
        if set(ordered.index) != {"E0", "E1", "E2"}:
            continue
        record: dict[str, Any] = dict(zip(key_columns, key, strict=True))
        for behaviour in BEHAVIOURS:
            e0 = float(ordered.loc["E0", behaviour])
            e1 = float(ordered.loc["E1", behaviour])
            e2 = float(ordered.loc["E2", behaviour])
            record[f"{behaviour}_E0"] = e0
            record[f"{behaviour}_E1"] = e1
            record[f"{behaviour}_E2"] = e2
            record[behaviour] = e2
            for form, value in trajectory_feature_values(e0, e1, e2).items():
                record[f"{behaviour}_{form}"] = value
        trajectory_rows.append(record)
    trajectories = pd.DataFrame(trajectory_rows)
    retained = population.merge(trajectories, on=key_columns, how="inner", validate="one_to_one")
    retention = len(retained) / len(population)
    if retention < 0.95:
        raise ScreenBlocker(
            "blocked_insufficient_trajectory_support",
            f"trajectory retention is {retention:.6f}",
        )
    missing_frame = pd.DataFrame(
        missingness,
        columns=[
            "symbol",
            "session",
            "month",
            "decision_ordinal",
            "anchor_ordinal",
            "anchor_role",
            "emotion",
            "reason",
        ],
    )
    source = {
        **SAFETY_FLAGS,
        "provider": "EODHD",
        "timeframe": "5m",
        "raw_data_downloaded": False,
        "minimum_timestamp_read": str(minimum_timestamp),
        "maximum_timestamp_read": str(maximum_timestamp),
        "protected_rows_materialised": 0,
        "sources": source_records,
    }
    support = {
        **SAFETY_FLAGS,
        "eligible_rows": len(population),
        "complete_trajectory_rows": len(retained),
        "trajectory_retention": retention,
        "missing_anchor_rows": len(missing_frame),
        "final_anchor_reproduction": reproduction,
        "frozen_component_ledger_path": str(frozen_component_path),
        "frozen_component_ledger_sha256": EXPECTED_COMPONENT_LEDGER_SHA256,
        "maximum_pre_authority_raw_component_difference": max(
            raw_component_differences.values(), default=0.0
        ),
        "pre_authority_raw_component_differences": raw_component_differences,
        "final_anchor_component_authority": (
            "frozen observable behavioural component ledger; preserves its broader "
            "causal cross-sectional signed-progress context"
        ),
    }
    return retained, scaled, scaling_manifest, missing_frame, {"source": source, "support": support}


def serialize_anchor_scaling(
    base: Mapping[int, Mapping[str, RobustComponentScale]],
    derived: Mapping[int, Mapping[str, RobustComponentScale]],
) -> dict[str, Any]:
    def serialize(
        values: Mapping[int, Mapping[str, RobustComponentScale]],
    ) -> dict[str, dict[str, Mapping[str, float | str]]]:
        result: dict[str, dict[str, Mapping[str, float | str]]] = {}
        for scale_group, components in sorted(values.items()):
            checkpoint, anchor = divmod(int(scale_group), 100)
            key = f"checkpoint_{checkpoint}_anchor_{anchor}"
            result[key] = {
                component: frozen.as_dict() for component, frozen in sorted(components.items())
            }
        return result

    return {
        **SAFETY_FLAGS,
        "fit_interval": "2024-01-01_through_2024-12-31_only",
        "method": "checkpoint_and_anchor_specific_median_iqr",
        "clip": [-5.0, 5.0],
        "base_components": serialize(base),
        "pressure_aligned_components": serialize(derived),
        "new_final_anchor_groups": [
            "checkpoint_24_anchor_24",
            "checkpoint_36_anchor_36",
        ],
    }


def reproduce_frozen_final_anchors(
    ledger: pd.DataFrame,
    scaling_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_panel = pd.read_parquet(
        COARSE_PANEL,
        columns=["symbol", "session", "decision_ordinal", *BEHAVIOURS],
    )
    final = ledger.loc[
        ledger["anchor_role"].eq("E2") & ledger["decision_ordinal"].isin((6, 12)),
        ["symbol", "session", "decision_ordinal", *BEHAVIOURS],
    ]
    comparison = final.merge(
        frozen_panel,
        on=["symbol", "session", "decision_ordinal"],
        suffixes=("", "_frozen"),
        validate="one_to_one",
    )
    maximum_level_difference = float(
        max(
            np.max(np.abs(comparison[behaviour] - comparison[f"{behaviour}_frozen"]))
            for behaviour in BEHAVIOURS
        )
    )
    frozen_scaling = json.loads(OBSERVABLE_SCALING.read_text(encoding="utf-8"))
    parameter_differences: list[float] = []
    for checkpoint in (6, 12):
        key = f"checkpoint_{checkpoint}_anchor_{checkpoint}"
        for family in ("base_components", "pressure_aligned_components"):
            actual = scaling_manifest[family][key]
            expected = frozen_scaling[family][str(checkpoint)]
            for component in expected:
                for field in ("center", "scale", "clip_lower", "clip_upper"):
                    parameter_differences.append(
                        abs(float(actual[component][field]) - float(expected[component][field]))
                    )
    maximum_scaling_difference = float(max(parameter_differences, default=0.0))
    if maximum_level_difference > 1e-12 or maximum_scaling_difference > 1e-12:
        raise ScreenBlocker(
            "blocked_behavioural_trajectory_not_causal",
            "final-anchor frozen behavioural values or scaling do not reproduce: "
            f"levels={maximum_level_difference}, scaling={maximum_scaling_difference}",
        )
    return {
        "rows_compared": len(comparison),
        "maximum_behavioural_level_difference": maximum_level_difference,
        "maximum_scaling_parameter_difference": maximum_scaling_difference,
        "tolerance": 1e-12,
        "passed": True,
    }


def build_targets_and_controls(
    decisions: pd.DataFrame,
    states: pd.DataFrame,
    dictionary: Any,
    coarse: ModuleType,
) -> pd.DataFrame:
    """Scan each causal state path once and derive targets plus structural history."""

    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    requested = set(
        zip(decisions["symbol"].astype(str), decisions["session"].astype(str), strict=True)
    )
    records: list[dict[str, Any]] = []
    for (symbol_value, session_value), group in states.groupby(["symbol", "session"], sort=True):
        symbol = str(symbol_value)
        session = str(session_value)
        if (symbol, session) not in requested:
            continue
        ordered = group.sort_values("bar_ordinal", kind="mergesort")
        needed = ordered.loc[ordered["bar_ordinal"].le(MAX_TARGET_BAR_ORDINAL)].copy()
        if needed["bar_ordinal"].astype(int).tolist() != list(range(MAX_TARGET_BAR_ORDINAL + 1)):
            raise ScreenBlocker(
                "blocked_predecessor_population_not_reconstructable",
                f"V2 state path is incomplete for {symbol}/{session}",
            )
        hard = needed["causal_hard_state"].to_numpy(dtype=int)
        event_mask = np.concatenate(([True], hard[1:] != hard[:-1]))
        event_rows = needed.loc[event_mask]
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
        event_ordinals = np.asarray([event.bar_ordinal for event in trace.state_events], dtype=int)
        completion_ordinals = [
            int(event.completion_bar_ordinal) for event in trace.registered_completions
        ]
        opening_completion_count = sum(value <= 11 for value in completion_ordinals)
        checkpoints = decisions.loc[
            decisions["symbol"].eq(symbol) & decisions["session"].eq(session)
        ]
        for decision in checkpoints.itertuples(index=False):
            checkpoint = int(decision.decision_ordinal)
            origin = checkpoint - 1
            candidates = np.flatnonzero(event_ordinals <= origin)
            if not len(candidates):
                raise ScreenBlocker(
                    "blocked_chronology_or_leakage_failure", "decision has no causal state event"
                )
            event_index = int(candidates[-1])
            outcome = coarse.resolve_first_loop_target(
                engine,
                trace,
                decision_id=f"{symbol}|{session}|{checkpoint:02d}",
                decision_event_index=event_index,
                decision_bar_ordinal=origin,
                decision_timestamp=pd.Timestamp(
                    decision.feature_available_timestamp_utc
                ).to_pydatetime(),
                session_end_bar_ordinal=EXPECTED_SESSION_BARS - 1,
                horizon_bars=6,
                source_available=True,
                symbol=symbol,
                session=session,
            )
            target = map_six_bar_structural_target(str(outcome["raw_outcome"]), horizon_bars=6)
            controls = structural_history_controls(
                registered_completion_bar_ordinals=completion_ordinals,
                decision_bar_ordinal=origin,
                active_registered_prefix_count=len(trace.prefixes_after_event[event_index]),
            )
            horizon = needed.loc[needed["bar_ordinal"].between(0, origin + 6)]
            records.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "decision_ordinal": checkpoint,
                    **outcome,
                    "target_class": target,
                    "scoring_eligible": target is not None,
                    **controls,
                    "opening_registered_completion_count_through_ordinal_12": (
                        opening_completion_count
                    ),
                    "phase": phase_label(checkpoint),
                    "late_loop_subgroup": late_loop_subgroup(
                        checkpoint,
                        opening_registered_completion_count=opening_completion_count,
                    ),
                    "state_path_through_horizon": horizon["causal_hard_state"].astype(int).tolist(),
                    "bar_ordinals_through_horizon": horizon["bar_ordinal"].astype(int).tolist(),
                    "decision_event_index": event_index,
                }
            )
    result = pd.DataFrame(records).sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    if len(result) != len(decisions):
        raise ScreenBlocker(
            "blocked_predecessor_population_not_reconstructable",
            f"target/control rows differ: {len(result)} versus {len(decisions)}",
        )
    return result.reset_index(drop=True)


def assemble_panel(
    trajectories: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    keys = ["symbol", "session", "decision_ordinal"]
    target_columns = [
        *keys,
        "raw_outcome",
        "semantic_loop_id",
        "orientation",
        "oriented_loop_key",
        "motif_type",
        "bars_until_completion",
        "state_events_until_completion",
        "target_excluded",
        "tied_semantic_loop_ids",
        "source_available",
        "target_class",
        "scoring_eligible",
        *STRUCTURAL_CONTROL_FEATURES,
        "opening_registered_completion_count_through_ordinal_12",
        "phase",
        "late_loop_subgroup",
        "state_path_through_horizon",
        "bar_ordinals_through_horizon",
        "decision_event_index",
    ]
    panel = trajectories.merge(
        targets.loc[:, target_columns], on=keys, how="inner", validate="one_to_one"
    )
    reject_protected_dates(panel)
    scoring = panel["scoring_eligible"].astype(bool)
    panel.loc[scoring, "row_weight"] = 1.0 / panel.loc[scoring].groupby("slate_id", sort=True)[
        "symbol"
    ].transform("size")
    development = panel["year"].eq(2024) & scoring
    baseline_interactions, baseline_bounds = build_interactions(
        panel.loc[development], fit_bounds=True
    )
    del baseline_interactions
    interactions, _ = build_interactions(panel, bounds=baseline_bounds)
    for feature in INTERACTION_FEATURES:
        panel[feature] = interactions[feature]
    trajectory_dev, trajectory_bounds = build_trajectory_regime_interactions(
        panel.loc[development], fit_bounds=True
    )
    del trajectory_dev
    trajectory_interactions, _ = build_trajectory_regime_interactions(
        panel, bounds=trajectory_bounds
    )
    for feature in TRAJECTORY_INTERACTION_FEATURES:
        panel[feature] = trajectory_interactions[feature]
    required = set((*T2_FEATURES, "row_weight"))
    if panel.loc[scoring, list(required)].isna().any().any():
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "a scoring model input is missing"
        )
    baseline_manifest = {
        feature: {"q01": bounds[0], "q99": bounds[1]}
        for feature, bounds in sorted(baseline_bounds.items())
    }
    trajectory_manifest = {
        feature: {"q01": bounds[0], "q99": bounds[1]}
        for feature, bounds in sorted(trajectory_bounds.items())
    }
    return panel, baseline_manifest, trajectory_manifest


@dataclass(slots=True)
class FittedMultinomial:
    name: str
    features: tuple[str, ...]
    scaler: StandardScaler
    estimator: LogisticRegression
    class_order: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, list(self.features)].to_numpy(dtype=float)
        return np.asarray(self.estimator.predict_proba(self.scaler.transform(matrix)), dtype=float)

    def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "features": list(self.features),
            "class_order": list(self.class_order),
            "estimator_classes": self.estimator.classes_.astype(int).tolist(),
            "coefficient": self.estimator.coef_.tolist(),
            "intercept": self.estimator.intercept_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "scaler_variance": self.scaler.var_.tolist(),
            "n_iter": self.estimator.n_iter_.astype(int).tolist(),
        }


def fit_multinomial(name: str, development: pd.DataFrame) -> FittedMultinomial:
    features = MODEL_FEATURES[name]
    class_index = {label: index for index, label in enumerate(TARGET_CLASSES)}
    target = development["target_class"].map(class_index)
    if target.isna().any() or set(target.astype(int)) != set(range(3)):
        raise ScreenBlocker(
            "blocked_insufficient_trajectory_support", "development class support is incomplete"
        )
    matrix = development.loc[:, list(features)].to_numpy(dtype=float)
    weights = development["row_weight"].to_numpy(dtype=float)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled = scaler.fit_transform(matrix)
    kwargs: dict[str, Any] = {
        "penalty": "l2",
        "C": 0.25,
        "solver": "lbfgs",
        "max_iter": 300,
        "class_weight": None,
        "random_state": MODEL_SEED,
        "n_jobs": 1,
    }
    # scikit-learn 1.7 removed the no-longer-needed public multi_class parameter;
    # lbfgs with three classes still fits the required multinomial objective.
    if "multi_class" in LogisticRegression().get_params():
        kwargs["multi_class"] = "multinomial"
    estimator = LogisticRegression(**kwargs)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            estimator.fit(scaled, target.to_numpy(dtype=int), sample_weight=weights)
    except ConvergenceWarning as error:
        raise ScreenBlocker(
            "blocked_model_convergence_failure", f"{name} emitted a convergence warning"
        ) from error
    if not np.array_equal(estimator.classes_, np.arange(3)) or np.any(estimator.n_iter_ >= 300):
        raise ScreenBlocker("blocked_model_convergence_failure", f"{name} did not converge")
    return FittedMultinomial(name, features, scaler, estimator, TARGET_CLASSES)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def probability_diagnostics(
    target_indices: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.empty(len(target_indices), dtype=int)
    for index, target in enumerate(target_indices):
        ranks[index] = int(np.flatnonzero(order[index] == target)[0]) + 1
    realised = probabilities[np.arange(len(target_indices)), target_indices]
    return ranks, realised, prediction_entropy(probabilities)


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(weights.sum())
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if not mask.any():
            continue
        result += (
            float(weights[mask].sum())
            / total
            * abs(
                weighted_mean(correct[mask].astype(float), weights[mask])
                - weighted_mean(confidence[mask], weights[mask])
            )
        )
    return float(result)


def metric_row(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    model: str,
) -> dict[str, Any]:
    class_index = {label: index for index, label in enumerate(TARGET_CLASSES)}
    targets = frame["target_class"].map(class_index).to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    ranks, realised, entropy = probability_diagnostics(targets, probabilities)
    mean_entropy = weighted_mean(entropy, weights)
    support = frame["target_class"].value_counts().reindex(TARGET_CLASSES, fill_value=0)
    return {
        "model": model,
        "multiclass_log_loss": float(
            log_loss(targets, probabilities, labels=np.arange(3), sample_weight=weights)
        ),
        "multiclass_brier": multiclass_brier(targets, probabilities, weights),
        "top_one_accuracy": weighted_mean((ranks <= 1).astype(float), weights),
        "top_two_accuracy": weighted_mean((ranks <= 2).astype(float), weights),
        "mean_reciprocal_rank": weighted_mean(1.0 / ranks, weights),
        "mean_probability_realised_class": weighted_mean(realised, weights),
        "expected_calibration_error": expected_calibration_error(targets, probabilities, weights),
        "prediction_entropy": mean_entropy,
        "effective_candidate_count": math.exp(mean_entropy),
        "rows": len(frame),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "class_support": json.dumps(
            {key: int(value) for key, value in support.items()}, sort_keys=True
        ),
    }


def score_models(
    assessment: pd.DataFrame,
    models: Mapping[str, FittedMultinomial],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = assessment.loc[
        :,
        [
            "symbol",
            "session",
            "year_month",
            "decision_ordinal",
            "slate_id",
            "phase",
            "late_loop_subgroup",
            "posterior_entropy",
            "transition_probability",
            "target_class",
            "row_weight",
        ],
    ].copy()
    metrics: list[dict[str, Any]] = []
    for name, model in models.items():
        probabilities = model.predict(assessment)
        for index, label in enumerate(TARGET_CLASSES):
            predictions[f"{name}_probability_{label}"] = probabilities[:, index]
        metrics.append(metric_row(assessment, probabilities, model=name))
    return predictions, pd.DataFrame(metrics)


def probabilities_for(predictions: pd.DataFrame, model: str) -> np.ndarray:
    return predictions.loc[
        :, [f"{model}_probability_{label}" for label in TARGET_CLASSES]
    ].to_numpy(dtype=float)


def subset_predictions(
    predictions: pd.DataFrame,
    mask: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    subset = predictions.loc[mask].copy()
    return subset, subset.index.to_numpy(dtype=int)


def scope_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "pooled": pd.Series(True, index=frame.index),
        "opening": frame["phase"].eq("OPENING_PHASE"),
        "later": frame["phase"].eq("LATER_PHASE"),
        "late_no_open": frame["late_loop_subgroup"].eq("LATE_NO_OPEN_REGISTERED_LOOP"),
    }


def grouped_model_metrics(
    predictions: pd.DataFrame,
    *,
    group_type: str,
    group_values: Mapping[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, mask in group_values.items():
        subset = predictions.loc[mask]
        if subset.empty:
            continue
        for model in MODEL_FEATURES:
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": value,
                    **metric_row(subset, probabilities_for(subset, model), model=model),
                }
            )
    return pd.DataFrame(rows)


def all_breakdown_metrics(
    predictions: pd.DataFrame,
    *,
    development_entropy_median: float,
    development_transition_median: float,
) -> dict[str, pd.DataFrame]:
    phase = grouped_model_metrics(
        predictions,
        group_type="phase",
        group_values={
            "OPENING_PHASE": predictions["phase"].eq("OPENING_PHASE"),
            "LATER_PHASE": predictions["phase"].eq("LATER_PHASE"),
        },
    )
    entropy = grouped_model_metrics(
        predictions,
        group_type="posterior_entropy_split",
        group_values={
            "LOW": predictions["posterior_entropy"].le(development_entropy_median),
            "HIGH": predictions["posterior_entropy"].gt(development_entropy_median),
        },
    )
    transition = grouped_model_metrics(
        predictions,
        group_type="transition_probability_split",
        group_values={
            "LOW": predictions["transition_probability"].le(development_transition_median),
            "HIGH": predictions["transition_probability"].gt(development_transition_median),
        },
    )
    phase_metrics = pd.concat([phase, entropy, transition], ignore_index=True)
    checkpoint_metrics = grouped_model_metrics(
        predictions,
        group_type="decision_ordinal",
        group_values={
            str(checkpoint): predictions["decision_ordinal"].eq(checkpoint)
            for checkpoint in CHECKPOINTS
        },
    )
    late_metrics = grouped_model_metrics(
        predictions,
        group_type="late_loop_subgroup",
        group_values={
            subgroup: predictions["late_loop_subgroup"].eq(subgroup)
            for subgroup in (
                "LATE_NO_OPEN_REGISTERED_LOOP",
                "LATE_AFTER_OPEN_REGISTERED_LOOP",
            )
        },
    )
    monthly_parts: list[pd.DataFrame] = []
    scopes = scope_masks(predictions)
    for scope, scope_mask in scopes.items():
        for month in sorted(predictions["year_month"].astype(str).unique()):
            month_mask = scope_mask & predictions["year_month"].eq(month)
            monthly_parts.append(
                grouped_model_metrics(
                    predictions,
                    group_type="assessment_month",
                    group_values={f"{scope}|{month}": month_mask},
                )
            )
    monthly_metrics = pd.concat(monthly_parts, ignore_index=True)
    class_metrics = grouped_model_metrics(
        predictions,
        group_type="realised_target_class",
        group_values={target: predictions["target_class"].eq(target) for target in TARGET_CLASSES},
    )
    return {
        "phase_metrics": phase_metrics,
        "checkpoint_metrics": checkpoint_metrics,
        "late_loop_subgroup_metrics": late_metrics,
        "monthly_metrics": monthly_metrics,
        "class_metrics": class_metrics,
    }


def support_and_concentration(
    panel: pd.DataFrame,
    assessment: pd.DataFrame,
    *,
    eligible_population_rows: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, bool]]:
    retention = len(panel) / eligible_population_rows
    masks = {
        **scope_masks(assessment),
        "late_after_open": assessment["late_loop_subgroup"].eq("LATE_AFTER_OPEN_REGISTERED_LOOP"),
    }
    concentration_rows: list[dict[str, Any]] = []
    concentration_pass: dict[str, bool] = {}
    scope_support: dict[str, Any] = {}
    for scope, mask in masks.items():
        rows = assessment.loc[mask]
        stock_shares = rows["symbol"].value_counts(normalize=True)
        class_shares = rows["target_class"].value_counts(normalize=True)
        max_stock = float(stock_shares.max()) if not stock_shares.empty else math.nan
        max_class = float(class_shares.max()) if not class_shares.empty else math.nan
        passed = bool(max_stock <= 0.10 and max_class <= 0.75) if len(rows) else False
        concentration_pass[scope] = passed
        concentration_rows.extend(
            [
                {
                    "population": scope,
                    "gate": "maximum_stock_share",
                    "value": max_stock,
                    "threshold": 0.10,
                    "passed": bool(max_stock <= 0.10) if len(rows) else False,
                },
                {
                    "population": scope,
                    "gate": "maximum_target_class_share",
                    "value": max_class,
                    "threshold": 0.75,
                    "passed": bool(max_class <= 0.75) if len(rows) else False,
                },
            ]
        )
        support = rows["target_class"].value_counts().reindex(TARGET_CLASSES, fill_value=0)
        scope_support[scope] = {
            "rows": len(rows),
            "sessions": int(rows["session"].nunique()),
            "stocks": int(rows["symbol"].nunique()),
            "months": int(rows["year_month"].nunique()),
            "class_support": {key: int(value) for key, value in support.items()},
            "maximum_stock_share": max_stock,
            "maximum_target_class_share": max_class,
        }
    pooled = scope_support["pooled"]
    pooled_pass = bool(
        pooled["rows"] >= 10_000
        and pooled["sessions"] >= 140
        and pooled["stocks"] >= 15
        and pooled["months"] == 8
        and min(pooled["class_support"].values()) >= 100
        and concentration_pass["pooled"]
        and retention >= 0.95
    )
    later = scope_support["later"]
    later_pass = bool(
        later["rows"] >= 4_500
        and later["sessions"] >= 140
        and later["stocks"] >= 15
        and min(later["class_support"].values()) >= 75
        and concentration_pass["later"]
    )
    late_no = scope_support["late_no_open"]
    late_no_pass = bool(
        late_no["rows"] >= 1_500
        and late_no["sessions"] >= 100
        and late_no["stocks"] >= 15
        and min(late_no["class_support"].values()) >= 40
        and concentration_pass["late_no_open"]
    )
    if not pooled_pass:
        raise ScreenBlocker(
            "blocked_insufficient_trajectory_support",
            f"pooled assessment support failed: {pooled}",
        )
    supported = {
        "pooled": True,
        "opening": concentration_pass["opening"],
        "later": later_pass,
        "late_no_open": late_no_pass,
    }
    manifest = {
        **SAFETY_FLAGS,
        "eligible_four_checkpoint_rows": eligible_population_rows,
        "retained_rows": len(panel),
        "trajectory_retention": retention,
        "assessment": scope_support,
        "pooled_support_passed": pooled_pass,
        "later_support_passed": later_pass,
        "late_no_open_support_passed": late_no_pass,
        "screen_supported_populations": supported,
    }
    return manifest, pd.DataFrame(concentration_rows), supported


def comparison_increment(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
) -> dict[str, float]:
    baseline_metrics = metric_row(frame, probabilities_for(frame, baseline), model=baseline)
    candidate_metrics = metric_row(frame, probabilities_for(frame, candidate), model=candidate)
    return {
        "log_loss_improvement": float(
            baseline_metrics["multiclass_log_loss"] - candidate_metrics["multiclass_log_loss"]
        ),
        "brier_improvement": float(
            baseline_metrics["multiclass_brier"] - candidate_metrics["multiclass_brier"]
        ),
        "top_two_change": float(
            candidate_metrics["top_two_accuracy"] - baseline_metrics["top_two_accuracy"]
        ),
        "top_one_change": float(
            candidate_metrics["top_one_accuracy"] - baseline_metrics["top_one_accuracy"]
        ),
        "realised_class_probability_change": float(
            candidate_metrics["mean_probability_realised_class"]
            - baseline_metrics["mean_probability_realised_class"]
        ),
        "prediction_entropy_reduction": float(
            baseline_metrics["prediction_entropy"] - candidate_metrics["prediction_entropy"]
        ),
        "effective_candidate_count_change": float(
            candidate_metrics["effective_candidate_count"]
            - baseline_metrics["effective_candidate_count"]
        ),
    }


def real_increments(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, mask in scope_masks(predictions).items():
        subset = predictions.loc[mask]
        for baseline, candidate in MODEL_COMPARISONS:
            rows.append(
                {
                    "population": scope,
                    "comparison": f"{candidate}_minus_{baseline}",
                    **comparison_increment(subset, baseline=baseline, candidate=candidate),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = predictions.reset_index(drop=True)
    draws = session_block_bootstrap_draws(base, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for draw_index, draw in enumerate(draws):
        sample = base.iloc[draw.row_indices].reset_index(drop=True)
        for scope, mask in scope_masks(sample).items():
            subset = sample.loc[mask]
            if subset.empty:
                continue
            for baseline, candidate in MODEL_COMPARISONS:
                rows.append(
                    {
                        "record_type": "draw",
                        "draw": draw_index,
                        "population": scope,
                        "comparison": f"{candidate}_minus_{baseline}",
                        "sampled_session_count": len(draw.sampled_sessions),
                        **comparison_increment(subset, baseline=baseline, candidate=candidate),
                    }
                )
    draw_frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    metrics = ("log_loss_improvement", "brier_improvement", "top_two_change")
    for (population, comparison), group in draw_frame.groupby(
        ["population", "comparison"], sort=True
    ):
        key = f"{population}|{comparison}"
        summary[key] = {}
        for metric in metrics:
            summary[key][metric] = {}
            values = group[metric].to_numpy(dtype=float)
            for level in (0.80, 0.90, 0.95):
                tail = (1.0 - level) / 2.0
                lower = float(np.quantile(values, tail, method="linear"))
                upper = float(np.quantile(values, 1.0 - tail, method="linear"))
                summary[key][metric][f"{level:.2f}"] = {"lower": lower, "upper": upper}
                summary_rows.append(
                    {
                        "record_type": "interval",
                        "draw": pd.NA,
                        "population": population,
                        "comparison": comparison,
                        "metric": metric,
                        "interval_level": level,
                        "lower": lower,
                        "upper": upper,
                    }
                )
    return pd.concat([draw_frame, pd.DataFrame(summary_rows)], ignore_index=True), summary


def trajectory_null(
    panel: pd.DataFrame,
    real_predictions: pd.DataFrame,
    *,
    trajectory_bounds: Mapping[str, tuple[float, float]],
    real: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scoring = panel.loc[panel["scoring_eligible"].astype(bool)].copy().reset_index(drop=True)
    real_lookup = {
        (str(row.population), str(row.comparison)): row for row in real.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    serialized_models: list[dict[str, Any]] = []
    for draw in range(NULL_DRAWS):
        parts: list[pd.DataFrame] = []
        scoring["_row_order"] = np.arange(len(scoring))
        for year in (2024, 2025):
            part = scoring.loc[scoring["year"].eq(year)].copy()
            part = permute_trajectory_bundle_within_slates(
                part,
                features=ALL_TRAJECTORY_FEATURES,
                seed=NULL_SEED + draw * 10 + (year - 2024),
            )
            parts.append(part)
        permuted = pd.concat(parts, ignore_index=True).sort_values("_row_order", kind="mergesort")
        trajectory_interactions, _ = build_trajectory_regime_interactions(
            permuted, bounds=trajectory_bounds
        )
        for feature in TRAJECTORY_INTERACTION_FEATURES:
            permuted[feature] = trajectory_interactions[feature]
        development = permuted.loc[permuted["year"].eq(2024)]
        assessment = permuted.loc[permuted["year"].eq(2025)].copy()
        t1 = fit_multinomial("T1", development)
        t2 = fit_multinomial("T2", development)
        null_predictions = assessment.loc[
            :,
            [
                "symbol",
                "session",
                "year_month",
                "decision_ordinal",
                "slate_id",
                "phase",
                "late_loop_subgroup",
                "posterior_entropy",
                "transition_probability",
                "target_class",
                "row_weight",
            ],
        ].copy()
        key_columns = ["symbol", "session", "decision_ordinal"]
        frozen_t0 = real_predictions.loc[
            :,
            [
                *key_columns,
                *[f"T0_probability_{label}" for label in TARGET_CLASSES],
            ],
        ]
        null_predictions = null_predictions.merge(
            frozen_t0, on=key_columns, how="left", validate="one_to_one"
        )
        for model in (t1, t2):
            probabilities = model.predict(assessment)
            for index, label in enumerate(TARGET_CLASSES):
                null_predictions[f"{model.name}_probability_{label}"] = probabilities[:, index]
        for scope, mask in scope_masks(null_predictions).items():
            subset = null_predictions.loc[mask]
            for baseline, candidate in MODEL_COMPARISONS:
                rows.append(
                    {
                        "record_type": "draw",
                        "draw": draw,
                        "population": scope,
                        "comparison": f"{candidate}_minus_{baseline}",
                        **comparison_increment(subset, baseline=baseline, candidate=candidate),
                    }
                )
        serialized_models.append({"draw": draw, "T1": t1.serialize(), "T2": t2.serialize()})
    draw_frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for (population, comparison), group in draw_frame.groupby(
        ["population", "comparison"], sort=True
    ):
        real_row = real_lookup[(str(population), str(comparison))]
        key = f"{population}|{comparison}"
        summary[key] = {}
        for metric in ("log_loss_improvement", "brier_improvement", "top_two_change"):
            real_value = float(getattr(real_row, metric))
            count = int((real_value > group[metric].to_numpy(dtype=float)).sum())
            summary[key][metric] = {
                "real_increment": real_value,
                "null_draws_exceeded": count,
                "exceeds_at_least_0_of_5": count >= 0,
                "exceeds_at_least_3_of_5": count >= 3,
                "exceeds_at_least_4_of_5": count >= 4,
                "exceeds_all_5": count == 5,
            }
            summary_rows.append(
                {
                    "record_type": "comparison",
                    "draw": pd.NA,
                    "population": population,
                    "comparison": comparison,
                    "metric": metric,
                    **summary[key][metric],
                }
            )
    return (
        pd.concat([draw_frame, pd.DataFrame(summary_rows)], ignore_index=True),
        summary,
        serialized_models,
    )


def trajectory_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    populations = {
        "development": panel.loc[panel["year"].eq(2024) & panel["scoring_eligible"]],
        "assessment": panel.loc[panel["year"].eq(2025) & panel["scoring_eligible"]],
    }
    continuous_forms = ("change", "acceleration", "recent_change", "peak_displacement")
    correlation_forms = (*PRIMARY_FORMS, *DESCRIPTIVE_FORMS)
    for population_name, population in populations.items():
        for behaviour in BEHAVIOURS:
            level = population[behaviour].astype(float)
            change = population[f"{behaviour}_change"].astype(float)
            statistics = {
                "level_mean": float(level.mean()),
                "level_standard_deviation": float(level.std(ddof=0)),
                "change_mean": float(change.mean()),
                "change_standard_deviation": float(change.std(ddof=0)),
                "acceleration_mean": float(population[f"{behaviour}_acceleration"].mean()),
                "acceleration_standard_deviation": float(
                    population[f"{behaviour}_acceleration"].std(ddof=0)
                ),
                "reversal_frequency": float(population[f"{behaviour}_reversal"].mean()),
                "persistence_frequency": float(
                    population[f"{behaviour}_monotonic_persistence"].ne(0).mean()
                ),
                "peak_displacement_mean": float(
                    population[f"{behaviour}_peak_displacement"].mean()
                ),
                "peak_displacement_standard_deviation": float(
                    population[f"{behaviour}_peak_displacement"].std(ddof=0)
                ),
                "level_change_opposite_sign_frequency": float(
                    ((np.sign(level) * np.sign(change)) < 0).mean()
                ),
                "reverses_before_decision_frequency": float(
                    population[f"{behaviour}_reversal"].mean()
                ),
                "at_local_peak_at_decision_frequency": float(
                    np.isclose(
                        population[f"{behaviour}_peak_displacement"].to_numpy(dtype=float),
                        0.0,
                        atol=1e-12,
                    ).mean()
                ),
            }
            for statistic, value in statistics.items():
                rows.append(
                    {
                        "record_type": "summary",
                        "population": population_name,
                        "emotion": behaviour,
                        "trajectory": None,
                        "statistic": statistic,
                        "value": value,
                        "quintile": pd.NA,
                        "target_class": None,
                    }
                )
            for form in correlation_forms:
                feature = f"{behaviour}_{form}"
                rows.append(
                    {
                        "record_type": "correlation",
                        "population": population_name,
                        "emotion": behaviour,
                        "trajectory": form,
                        "statistic": "correlation_with_current_level",
                        "value": float(population[[behaviour, feature]].corr().iloc[0, 1]),
                        "quintile": pd.NA,
                        "target_class": None,
                    }
                )
                for other in correlation_forms:
                    other_feature = f"{behaviour}_{other}"
                    rows.append(
                        {
                            "record_type": "correlation",
                            "population": population_name,
                            "emotion": behaviour,
                            "trajectory": form,
                            "statistic": f"correlation_with_{other}",
                            "value": float(population[[feature, other_feature]].corr().iloc[0, 1]),
                            "quintile": pd.NA,
                            "target_class": None,
                        }
                    )
    development = populations["development"]
    assessment = populations["assessment"]
    for behaviour in BEHAVIOURS:
        for form in continuous_forms:
            feature = f"{behaviour}_{form}"
            bounds = np.quantile(
                development[feature].to_numpy(dtype=float),
                [0.2, 0.4, 0.6, 0.8],
                method="linear",
            )
            for population_name, population in (
                ("development", development),
                ("assessment", assessment),
            ):
                quintiles = (
                    np.searchsorted(bounds, population[feature].to_numpy(dtype=float), side="right")
                    + 1
                )
                for quintile in range(1, 6):
                    mask = quintiles == quintile
                    for target in TARGET_CLASSES:
                        value = (
                            float(population.loc[mask, "target_class"].eq(target).mean())
                            if mask.any()
                            else math.nan
                        )
                        rows.append(
                            {
                                "record_type": "development_frozen_quintile_target_rate",
                                "population": population_name,
                                "emotion": behaviour,
                                "trajectory": form,
                                "statistic": "target_class_rate",
                                "value": value,
                                "quintile": quintile,
                                "target_class": target,
                                "development_q20": float(bounds[0]),
                                "development_q40": float(bounds[1]),
                                "development_q60": float(bounds[2]),
                                "development_q80": float(bounds[3]),
                            }
                        )
    return pd.DataFrame(rows)


def monthly_positive_counts(monthly: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for scope in SCREEN_SCOPES:
        result[scope] = {}
        scope_rows = monthly.loc[monthly["group_value"].astype(str).str.startswith(f"{scope}|")]
        for baseline, candidate in MODEL_COMPARISONS:
            pivot = scope_rows.pivot(
                index="group_value", columns="model", values="multiclass_log_loss"
            )
            result[scope][f"{candidate}_minus_{baseline}"] = int(
                (pivot[baseline] - pivot[candidate] > 0.0).sum()
            )
    return result


def derive_screen_decision(
    real: pd.DataFrame,
    *,
    bootstrap_summary: Mapping[str, Any],
    null_summary: Mapping[str, Any],
    monthly_counts: Mapping[str, Mapping[str, int]],
    concentration_pass: Mapping[str, bool],
    supported: Mapping[str, bool],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, bool]]:
    lookup = {
        (str(row.population), str(row.comparison)): row for row in real.itertuples(index=False)
    }
    gates: list[dict[str, Any]] = []
    positives: dict[str, dict[str, bool]] = {"T1": {}, "T2": {}}
    for scope in SCREEN_SCOPES:
        for baseline, candidate in MODEL_COMPARISONS:
            comparison = f"{candidate}_minus_{baseline}"
            key = f"{scope}|{comparison}"
            increment = lookup[(scope, comparison)]
            month_threshold = 5 if scope in {"pooled", "opening"} else 4
            bootstrap_log = float(bootstrap_summary[key]["log_loss_improvement"]["0.80"]["lower"])
            bootstrap_brier = float(bootstrap_summary[key]["brier_improvement"]["0.80"]["lower"])
            null_log = int(null_summary[key]["log_loss_improvement"]["null_draws_exceeded"])
            null_brier = int(null_summary[key]["brier_improvement"]["null_draws_exceeded"])
            conditions = {
                "supported": bool(supported[scope]),
                "log_loss_improves": float(increment.log_loss_improvement) > 0.0,
                "brier_improves": float(increment.brier_improvement) > 0.0,
                "bootstrap_80_log_loss_lower_non_negative": bootstrap_log >= 0.0,
                "bootstrap_80_brier_lower_non_negative": bootstrap_brier >= 0.0,
                "top_two_decline_within_0_002": float(increment.top_two_change) >= -0.002,
                "positive_month_count": int(monthly_counts[scope][comparison]) >= month_threshold,
                "real_log_loss_or_brier_exceeds_four_of_five_nulls": (
                    null_log >= 4 or null_brier >= 4
                ),
                "concentration_gates_pass": bool(concentration_pass[scope]),
            }
            positive = all(conditions.values())
            positives[candidate][scope] = positive
            gates.append(
                {
                    "population": scope,
                    "comparison": comparison,
                    "conditions": conditions,
                    "bootstrap_80_log_loss_lower": bootstrap_log,
                    "bootstrap_80_brier_lower": bootstrap_brier,
                    "positive_log_loss_months": monthly_counts[scope][comparison],
                    "required_positive_months": month_threshold,
                    "log_loss_null_draws_exceeded": null_log,
                    "brier_null_draws_exceeded": null_brier,
                    "rough_screen_positive": positive,
                }
            )
    point_estimate_improves = bool(
        ((real["log_loss_improvement"] > 0.0) | (real["brier_improvement"] > 0.0)).any()
    )
    decision = decide_quick_screen(
        t1_positive=positives["T1"],
        t2_positive=positives["T2"],
        point_estimate_improves=point_estimate_improves,
    )
    artifact = {
        **SAFETY_FLAGS,
        "decision": decision,
        "primary_decision": decision,
        "feasibility_only": True,
        "validation_or_promotion": False,
        "T1_screen_positive": positives["T1"],
        "T2_screen_positive": positives["T2"],
        "point_estimate_improves_somewhere": point_estimate_improves,
        "gates": gates,
        "binding_answers": {
            "corrected_trajectories_improve_next_structural_event": bool(
                any(positives["T1"].values()) or any(positives["T2"].values())
            ),
            "increment_stronger_for_later_or_late_no_open": bool(
                positives["T1"]["later"]
                or positives["T1"]["late_no_open"]
                or positives["T2"]["later"]
                or positives["T2"]["late_no_open"]
            ),
        },
    }
    return artifact, positives["T1"], positives["T2"]


def determinism_check(
    panel: pd.DataFrame,
    original_models: Mapping[str, Mapping[str, Any]],
    original_predictions: pd.DataFrame,
    original_pooled: pd.DataFrame,
    *,
    decision: str,
    bootstrap_summary: Mapping[str, Any],
    null_summary: Mapping[str, Any],
    concentration_pass: Mapping[str, bool],
    supported: Mapping[str, bool],
) -> dict[str, Any]:
    scoring = panel.loc[panel["scoring_eligible"].astype(bool)]
    development = scoring.loc[scoring["year"].eq(2024)]
    assessment = scoring.loc[scoring["year"].eq(2025)]
    repeated = {name: fit_multinomial(name, development) for name in MODEL_FEATURES}
    repeated_predictions, repeated_pooled = score_models(assessment, repeated)
    probability_difference = 0.0
    coefficient_difference = 0.0
    preprocessing_difference = 0.0
    class_order_equal = True
    for name in MODEL_FEATURES:
        original_model = original_models[name]
        probability_difference = max(
            probability_difference,
            float(
                np.max(
                    np.abs(
                        probabilities_for(original_predictions, name)
                        - probabilities_for(repeated_predictions, name)
                    )
                )
            ),
        )
        coefficient_difference = max(
            coefficient_difference,
            float(
                np.max(
                    np.abs(
                        np.asarray(original_model["coefficient"], dtype=float)
                        - repeated[name].estimator.coef_
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        np.asarray(original_model["intercept"], dtype=float)
                        - repeated[name].estimator.intercept_
                    )
                )
            ),
        )
        preprocessing_difference = max(
            preprocessing_difference,
            float(
                np.max(
                    np.abs(
                        np.asarray(original_model["scaler_mean"], dtype=float)
                        - repeated[name].scaler.mean_
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        np.asarray(original_model["scaler_scale"], dtype=float)
                        - repeated[name].scaler.scale_
                    )
                )
            ),
        )
        class_order_equal &= (
            tuple(str(value) for value in original_model["class_order"])
            == repeated[name].class_order
        )
    numeric_metrics = [
        "multiclass_log_loss",
        "multiclass_brier",
        "top_one_accuracy",
        "top_two_accuracy",
        "mean_reciprocal_rank",
        "mean_probability_realised_class",
        "expected_calibration_error",
        "prediction_entropy",
        "effective_candidate_count",
    ]
    original_indexed = original_pooled.set_index("model")
    repeated_indexed = repeated_pooled.set_index("model")
    metric_difference = float(
        max(
            abs(
                float(original_indexed.loc[name, metric])
                - float(repeated_indexed.loc[name, metric])
            )
            for name in MODEL_FEATURES
            for metric in numeric_metrics
        )
    )
    repeated_monthly = all_breakdown_metrics(
        repeated_predictions,
        development_entropy_median=float(development["posterior_entropy"].median()),
        development_transition_median=float(development["transition_probability"].median()),
    )["monthly_metrics"]
    repeated_decision, _, _ = derive_screen_decision(
        real_increments(repeated_predictions),
        bootstrap_summary=bootstrap_summary,
        null_summary=null_summary,
        monthly_counts=monthly_positive_counts(repeated_monthly),
        concentration_pass=concentration_pass,
        supported=supported,
    )
    decision_equal = str(repeated_decision["decision"]) == decision
    passed = bool(
        class_order_equal
        and probability_difference <= 1e-12
        and coefficient_difference <= 1e-12
        and preprocessing_difference <= 1e-12
        and metric_difference <= 1e-12
        and decision_equal
    )
    if not passed:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "fast primary-model determinism failed"
        )
    return {
        **SAFETY_FLAGS,
        "passed": passed,
        "class_order_equal": class_order_equal,
        "maximum_preprocessing_parameter_difference": preprocessing_difference,
        "maximum_coefficient_difference": coefficient_difference,
        "maximum_probability_difference": probability_difference,
        "maximum_pooled_metric_difference": metric_difference,
        "probability_tolerance": 1e-12,
        "bootstrap_repeated": False,
        "null_repeated": False,
        "decision_equal": decision_equal,
        "decision": decision,
        "regenerated_decision": repeated_decision["decision"],
    }


def anchor_manifest(ledger: pd.DataFrame, reproduction: Mapping[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for (checkpoint, anchor, role), rows in ledger.groupby(
        ["decision_ordinal", "anchor_ordinal", "anchor_role"], sort=True
    ):
        available = pd.to_datetime(rows["anchor_available_timestamp_utc"], utc=True)
        latest = pd.to_datetime(rows["latest_input_bar_complete_timestamp_utc"], utc=True)
        decision = pd.to_datetime(rows["decision_available_timestamp_utc"], utc=True)
        records.append(
            {
                "decision_ordinal": int(checkpoint),
                "anchor_ordinal": int(anchor),
                "anchor_role": str(role),
                "even_completed_bar_count": int(anchor) % 2 == 0,
                "rows": len(rows),
                "minimum_available_timestamp_utc": str(available.min()),
                "maximum_available_timestamp_utc": str(available.max()),
                "maximum_latest_input_minus_anchor_availability_seconds": float(
                    (latest - available).dt.total_seconds().max()
                ),
                "future_input_rows": int((latest > decision).sum()),
                "causal": bool((latest <= decision).all()),
            }
        )
    return {
        **SAFETY_FLAGS,
        "anchor_triplets": {str(key): list(value) for key, value in ANCHORS_BY_CHECKPOINT.items()},
        "completed_bar_convention": (
            "ordinal N uses bars 0 through N-1 and is available at the completion of bar N-1"
        ),
        "records": records,
        "final_anchor_reproduction": dict(reproduction),
        "passed": bool(all(row["causal"] for row in records) and reproduction["passed"]),
    }


def model_configuration_artifact() -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "model_count": 3,
        "configuration": {
            "penalty": "l2",
            "C": 0.25,
            "solver": "lbfgs",
            "max_iter": 300,
            "multi_class": "multinomial",
            "class_weight": None,
            "random_state": MODEL_SEED,
            "n_jobs": 1,
            "preprocessing_fit": "2024_only",
            "coefficients_fit": "2024_only",
            "row_weight": "1 / eligible_stocks_in_session_checkpoint",
        },
        "models": {
            name: {"features": list(features), "feature_count": len(features)}
            for name, features in MODEL_FEATURES.items()
        },
        "class_order": list(TARGET_CLASSES),
    }


def feature_manifest() -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "behavioural_dimensions": list(BEHAVIOURS),
        "primary_trajectory_features": list(PRIMARY_TRAJECTORY_FEATURES),
        "descriptive_only_trajectory_features": [
            f"{behaviour}_{form}" for behaviour in BEHAVIOURS for form in DESCRIPTIVE_FORMS
        ],
        "trajectory_formulas": {
            "change": "E2-E0",
            "acceleration": "(E2-E1)-(E1-E0)",
            "reversal": "1 iff consecutive changes have opposite non-zero signs; else 0",
            "recent_change": "E2-E1",
            "monotonic_persistence": "1 increasing, -1 decreasing, 0 otherwise",
            "peak_displacement": "E2-max(E0,E1,E2)",
        },
        "structural_history_controls": list(STRUCTURAL_CONTROL_FEATURES),
        "checkpoint_indicators": list(CHECKPOINT_FEATURES),
        "soft_regime_features": list(REGIME_FEATURES),
        "fitted_descriptive_trajectory_fields": False,
        "forbidden_economic_fields_opened": False,
    }


def plot_increments(real: pd.DataFrame, path: Path) -> None:
    populations = ("opening", "later", "late_no_open")
    comparisons = ("T1_minus_T0", "T2_minus_T1")
    metrics = ("log_loss_improvement", "brier_improvement")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for axis, population in zip(axes, populations, strict=True):
        subset = real.loc[real["population"].eq(population)].set_index("comparison")
        x = np.arange(len(metrics))
        width = 0.34
        for offset, comparison in enumerate(comparisons):
            values = [float(subset.loc[comparison, metric]) for metric in metrics]
            axis.bar(
                x + (offset - 0.5) * width,
                values,
                width,
                label=comparison.replace("_minus_", " − "),
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, ["Log loss", "Brier"])
        axis.set_title(population.replace("_", " ").title())
        axis.set_ylabel("Improvement (positive is favourable)")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Corrected behavioural-trajectory proper-score increments")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    for column in selected.select_dtypes(include=["number"]).columns:
        selected[column] = selected[column].map(
            lambda value: f"{value:.9f}" if pd.notna(value) else "NA"
        )
    headers = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    )
    return "\n".join((headers, rule, body))


def render_report(
    *,
    decision: Mapping[str, Any],
    support: Mapping[str, Any],
    pooled: pd.DataFrame,
    real: pd.DataFrame,
    bootstrap_summary: Mapping[str, Any],
    null_summary: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    target_counts: Mapping[str, int],
    determinism: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> str:
    increment_table = markdown_table(
        real,
        (
            "population",
            "comparison",
            "log_loss_improvement",
            "brier_improvement",
            "top_two_change",
            "prediction_entropy_reduction",
        ),
    )
    pooled_table = markdown_table(
        pooled,
        (
            "model",
            "multiclass_log_loss",
            "multiclass_brier",
            "top_one_accuracy",
            "top_two_accuracy",
            "mean_probability_realised_class",
            "prediction_entropy",
            "effective_candidate_count",
        ),
    )
    return f"""# Behavioural-Trajectory Funnel V0.1 — Corrected Anchors and Later Loops

Decision: `{decision["decision"]}`.

This was a retrospective, observable-only, structural quick feasibility screen. It did not
open economic outcomes, test price direction or trading rules, enable execution, or provide
prospective validation.

## Population and causal anchors

- Development: 2024-01-01 through 2024-12-31.
- Assessment: 2025-01-01 through 2025-08-22.
- Protected rows materialised: 0.
- Corrected anchors: 6→2/4/6, 12→4/8/12, 24→8/16/24, 36→12/24/36.
- Final-anchor rows compared: {reproduction["rows_compared"]}.
- Maximum final-level difference: {reproduction["maximum_behavioural_level_difference"]:.3g}.
- Maximum final-scaling difference: {reproduction["maximum_scaling_parameter_difference"]:.3g}.
- Trajectory retention: {support["trajectory_retention"]:.6f}.
- Assessment support: `{json.dumps(support["assessment"], sort_keys=True)}`.
- Assessment structural targets: `{json.dumps(dict(target_counts), sort_keys=True)}`.

## Pooled assessment metrics

{pooled_table}

## Preregistered increments

{increment_table}

The bootstrap used exactly 25 fixed-prediction whole-session draws. The null used exactly five
fixed-seed trajectory-bundle permutations with T1/T2 refits. Full 80%, 90%, and 95% intervals
and the five-draw comparisons are in `bootstrap_metrics.csv` and `null_metrics.csv`.

Bootstrap summary keys: `{", ".join(sorted(bootstrap_summary))}`.

Null summary keys: `{", ".join(sorted(null_summary))}`.

## Verification

- Fast determinism check: `{determinism["passed"]}`; maximum probability difference
  `{determinism["maximum_probability_difference"]:.3g}`.
- Independent lightweight audit: `{audit["passed"]}`.

The result is descriptive feasibility evidence only. It is not economic-edge evidence, a
trading strategy, a deployable model, or a claim of achieved P&L.
"""


def metric_increment_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (group_type, group_value), group in frame.groupby(["group_type", "group_value"], sort=True):
        models = group.set_index("model")
        for baseline, candidate in MODEL_COMPARISONS:
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "comparison": f"{candidate}_minus_{baseline}",
                    "log_loss_improvement": float(
                        models.loc[baseline, "multiclass_log_loss"]
                        - models.loc[candidate, "multiclass_log_loss"]
                    ),
                    "brier_improvement": float(
                        models.loc[baseline, "multiclass_brier"]
                        - models.loc[candidate, "multiclass_brier"]
                    ),
                    "top_one_change": float(
                        models.loc[candidate, "top_one_accuracy"]
                        - models.loc[baseline, "top_one_accuracy"]
                    ),
                    "top_two_change": float(
                        models.loc[candidate, "top_two_accuracy"]
                        - models.loc[baseline, "top_two_accuracy"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def render_detailed_appendix(output: Path, decision: Mapping[str, Any]) -> str:
    checkpoint = metric_increment_table(pd.read_csv(output / "checkpoint_metrics.csv"))
    phase = metric_increment_table(pd.read_csv(output / "phase_metrics.csv"))
    late = metric_increment_table(pd.read_csv(output / "late_loop_subgroup_metrics.csv"))
    classes = metric_increment_table(pd.read_csv(output / "class_metrics.csv"))
    bootstrap = pd.read_csv(output / "bootstrap_metrics.csv")
    intervals = bootstrap.loc[
        bootstrap["record_type"].eq("interval")
        & bootstrap["metric"].isin(("log_loss_improvement", "brier_improvement"))
    ].copy()
    intervals["interval"] = intervals.apply(
        lambda row: f"[{float(row['lower']):.9f}, {float(row['upper']):.9f}]", axis=1
    )
    interval_table = intervals.pivot(
        index=["population", "comparison", "metric"],
        columns="interval_level",
        values="interval",
    ).reset_index()
    interval_table.columns = [
        "population",
        "comparison",
        "metric",
        "80%",
        "90%",
        "95%",
    ]
    null = pd.read_csv(output / "null_metrics.csv")
    null_summary = null.loc[
        null["record_type"].eq("comparison")
        & null["metric"].isin(("log_loss_improvement", "brier_improvement")),
        [
            "population",
            "comparison",
            "metric",
            "real_increment",
            "null_draws_exceeded",
        ],
    ]
    concentration = pd.read_csv(output / "concentration_metrics.csv")
    diagnostics = pd.read_csv(output / "trajectory_diagnostics.csv")
    diagnostic_summary = diagnostics.loc[
        diagnostics["record_type"].eq("summary")
        & diagnostics["population"].eq("assessment")
        & diagnostics["statistic"].isin(
            (
                "change_mean",
                "acceleration_mean",
                "reversal_frequency",
                "persistence_frequency",
                "level_change_opposite_sign_frequency",
                "at_local_peak_at_decision_frequency",
            )
        ),
        ["emotion", "statistic", "value"],
    ]
    missingness = pd.read_csv(output / "trajectory_missingness.csv")
    return f"""

## Stability and subgroup appendix

Positive log-loss months by preregistered population and comparison:
`{json.dumps(decision["monthly_positive_log_loss_counts"], sort_keys=True)}`.

### Checkpoint increments

{markdown_table(checkpoint, tuple(checkpoint.columns))}

### Phase, posterior-entropy, and transition-probability increments

{markdown_table(phase, tuple(phase.columns))}

### Later opening-history subgroup increments

{markdown_table(late, tuple(late.columns))}

### Realised-target-class increments

{markdown_table(classes, tuple(classes.columns))}

### Session-bootstrap proper-score intervals

{markdown_table(interval_table, tuple(interval_table.columns))}

### Five-draw trajectory-null comparisons

{markdown_table(null_summary, tuple(null_summary.columns))}

### Concentration

{markdown_table(concentration, tuple(concentration.columns))}

### Assessment trajectory diagnostics

{markdown_table(diagnostic_summary, tuple(diagnostic_summary.columns))}

Missing anchor records: {len(missingness)}. The complete-case missingness ledger is
`trajectory_missingness.csv`; no alternative anchor was substituted.
"""


def execute_screen(output: Path, *, provider_root: Path) -> dict[str, Any]:
    contract = load_contract()
    predecessor_manifest = verify_predecessors()
    output.mkdir(parents=True, exist_ok=True)
    population, population_manifest = predecessor_population()
    coarse_panel = pd.read_parquet(COARSE_PANEL)
    coarse = load_script("_trajectory_v01_coarse_runner", COARSE_RUNNER)
    observable = load_script("_trajectory_v01_observable_runner", OBSERVABLE_RUNNER)
    coarse.MAX_TARGET_BAR_ORDINAL = MAX_TARGET_BAR_ORDINAL
    preprocessing, parameters = coarse.load_frozen_model()
    states, v2_source = coarse.build_v2_state_panel(provider_root, preprocessing, parameters)
    decisions, state_reconstruction = attach_state_features(population, states, coarse_panel)
    trajectory_panel, ledger, scaling, missingness, behaviour_context = build_behavioural_anchors(
        decisions,
        provider_root=provider_root,
        observable=observable,
    )
    dictionary, dictionary_manifest = coarse.load_loop_dictionary()
    targets = build_targets_and_controls(trajectory_panel, states, dictionary, coarse)
    panel, baseline_bounds_manifest, trajectory_bounds_manifest = assemble_panel(
        trajectory_panel, targets
    )
    scoring = panel.loc[panel["scoring_eligible"].astype(bool)].copy()
    development = scoring.loc[scoring["year"].eq(2024)].copy()
    assessment = scoring.loc[scoring["year"].eq(2025)].copy()
    if len(panel) > MAX_ROWS:
        raise ScreenBlocker(
            "blocked_quick_trajectory_late_loop_resource_limit", f"panel rows={len(panel)}"
        )
    support, concentration, supported = support_and_concentration(
        panel, assessment, eligible_population_rows=len(population)
    )
    models = {name: fit_multinomial(name, development) for name in MODEL_FEATURES}
    predictions, pooled = score_models(assessment, models)
    entropy_median = float(development["posterior_entropy"].median())
    transition_median = float(development["transition_probability"].median())
    breakdowns = all_breakdown_metrics(
        predictions,
        development_entropy_median=entropy_median,
        development_transition_median=transition_median,
    )
    real = real_increments(predictions)
    bootstrap, bootstrap_summary = bootstrap_metrics(predictions)
    trajectory_bounds = {
        feature: (float(values["q01"]), float(values["q99"]))
        for feature, values in trajectory_bounds_manifest.items()
    }
    null, null_summary, null_models = trajectory_null(
        panel,
        predictions,
        trajectory_bounds=trajectory_bounds,
        real=real,
    )
    diagnostics = trajectory_diagnostics(panel)
    monthly_counts = monthly_positive_counts(breakdowns["monthly_metrics"])
    concentration_pass = {
        scope: bool(concentration.loc[concentration["population"].eq(scope), "passed"].all())
        for scope in SCREEN_SCOPES
    }
    decision, _, _ = derive_screen_decision(
        real,
        bootstrap_summary=bootstrap_summary,
        null_summary=null_summary,
        monthly_counts=monthly_counts,
        concentration_pass=concentration_pass,
        supported=supported,
    )
    determinism = determinism_check(
        panel,
        {name: model.serialize() for name, model in models.items()},
        predictions,
        pooled,
        decision=str(decision["decision"]),
        bootstrap_summary=bootstrap_summary,
        null_summary=null_summary,
        concentration_pass=concentration_pass,
        supported=supported,
    )
    anchors = anchor_manifest(ledger, behaviour_context["support"]["final_anchor_reproduction"])
    target_counts = {
        str(key): int(value)
        for key, value in assessment["target_class"].value_counts().sort_index().items()
    }
    protected_audit = {
        **SAFETY_FLAGS,
        "development_start": "2024-01-01",
        "development_end_inclusive": "2024-12-31",
        "assessment_start": "2025-01-01",
        "assessment_end_inclusive": "2025-08-22",
        "protected_start": "2025-08-23",
        "minimum_timestamp_read": behaviour_context["source"]["minimum_timestamp_read"],
        "maximum_timestamp_read": behaviour_context["source"]["maximum_timestamp_read"],
        "protected_rows_materialised": 0,
        "passed": True,
    }
    source_manifest = {
        **SAFETY_FLAGS,
        "predecessors": predecessor_manifest,
        "population_reconstruction": population_manifest,
        "opening_state_reconstruction": state_reconstruction,
        "behavioural_source": behaviour_context["source"],
        "behavioural_anchor_support": behaviour_context["support"],
        "v2_source": v2_source,
        "loop_dictionary": dictionary_manifest,
        "raw_data_downloaded": False,
        "protected_rows_materialised": 0,
    }
    interaction_manifest = {
        **SAFETY_FLAGS,
        "behavioural_level_regime_interactions": baseline_bounds_manifest,
        "trajectory_regime_interactions": trajectory_bounds_manifest,
        "trajectory_interaction_count": len(TRAJECTORY_INTERACTION_FEATURES),
        "fit_interval": "2024_only",
        "clip_quantiles": [0.01, 0.99],
    }
    coefficient_artifact = {
        **SAFETY_FLAGS,
        "primary_models": {name: model.serialize() for name, model in models.items()},
        "null_models": null_models,
    }
    decision.update(
        {
            "development_rows": len(development),
            "assessment_rows": len(assessment),
            "assessment_target_counts": target_counts,
            "monthly_positive_log_loss_counts": monthly_counts,
            "support": support,
            "determinism_check_passed": True,
            "lightweight_audit_passed": None,
        }
    )
    write_json(output / "contract.json", contract)
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "protected_boundary_audit.json", protected_audit)
    write_json(output / "checkpoint_anchor_manifest.json", anchors)
    write_json(output / "trajectory_anchor_scaling.json", scaling)
    write_csv(output / "trajectory_missingness.csv", missingness)
    feature_document = feature_manifest()
    feature_document["development_frozen_reporting_medians"] = {
        "posterior_entropy": entropy_median,
        "transition_probability": transition_median,
    }
    write_json(output / "feature_manifest.json", feature_document)
    write_json(output / "interaction_manifest.json", interaction_manifest)
    write_parquet(output / "decision_panel.parquet", panel)
    write_parquet(output / "trajectory_ledger.parquet", ledger)
    write_json(output / "model_configurations.json", model_configuration_artifact())
    write_json(output / "model_coefficients.json", coefficient_artifact)
    write_parquet(output / "assessment_predictions.parquet", predictions)
    write_csv(output / "pooled_metrics.csv", pooled)
    for artifact, frame in breakdowns.items():
        write_csv(output / f"{artifact}.csv", frame)
    write_csv(output / "trajectory_diagnostics.csv", diagnostics)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "null_metrics.csv", null)
    write_csv(output / "concentration_metrics.csv", concentration)
    write_json(output / "decision.json", decision)
    write_json(output / "determinism_check.json", determinism)
    plot_increments(real, output / "proper_score_increments.png")
    auditor = load_script("_trajectory_v01_auditor", AUDITOR_PATH)
    audit = cast(dict[str, Any], auditor.audit(output))
    if not audit.get("passed"):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "independent lightweight audit failed"
        )
    decision["lightweight_audit_passed"] = True
    write_json(output / "decision.json", decision)
    report = render_report(
        decision=decision,
        support=support,
        pooled=pooled,
        real=real,
        bootstrap_summary=bootstrap_summary,
        null_summary=null_summary,
        reproduction=behaviour_context["support"]["final_anchor_reproduction"],
        target_counts=target_counts,
        determinism=determinism,
        audit=audit,
    ) + render_detailed_appendix(output, decision)
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


def bootstrap_summary_from_artifact(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in frame.loc[frame["record_type"].eq("interval")].itertuples(index=False):
        key = f"{row.population}|{row.comparison}"
        summary.setdefault(key, {}).setdefault(str(row.metric), {})[
            f"{float(row.interval_level):.2f}"
        ] = {"lower": float(row.lower), "upper": float(row.upper)}
    return summary


def null_summary_from_artifact(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in frame.loc[frame["record_type"].eq("comparison")].itertuples(index=False):
        key = f"{row.population}|{row.comparison}"
        summary.setdefault(key, {})[str(row.metric)] = {
            "real_increment": float(row.real_increment),
            "null_draws_exceeded": int(row.null_draws_exceeded),
            "exceeds_at_least_0_of_5": bool(row.exceeds_at_least_0_of_5),
            "exceeds_at_least_3_of_5": bool(row.exceeds_at_least_3_of_5),
            "exceeds_at_least_4_of_5": bool(row.exceeds_at_least_4_of_5),
            "exceeds_all_5": bool(row.exceeds_all_5),
        }
    return summary


def resume_audit_and_report(output: Path) -> dict[str, Any]:
    """Resume metric, deterministic-refit, audit, and report work from frozen outputs."""

    panel = pd.read_parquet(output / "decision_panel.parquet")
    predictions = pd.read_parquet(output / "assessment_predictions.parquet")
    bootstrap_frame = pd.read_csv(output / "bootstrap_metrics.csv")
    null_frame = pd.read_csv(output / "null_metrics.csv")
    concentration = pd.read_csv(output / "concentration_metrics.csv")
    _, population_manifest = predecessor_population()
    assessment = panel.loc[panel["scoring_eligible"] & panel["year"].eq(2025)]
    development = panel.loc[panel["scoring_eligible"] & panel["year"].eq(2024)]
    entropy_median = float(development["posterior_entropy"].median())
    transition_median = float(development["transition_probability"].median())
    pooled = pd.DataFrame(
        [
            metric_row(assessment, probabilities_for(predictions, name), model=name)
            for name in MODEL_FEATURES
        ]
    )
    breakdowns = all_breakdown_metrics(
        predictions,
        development_entropy_median=entropy_median,
        development_transition_median=transition_median,
    )
    monthly = breakdowns["monthly_metrics"]
    support, reconstructed_concentration, supported = support_and_concentration(
        panel,
        assessment,
        eligible_population_rows=int(
            population_manifest["combined_rows_before_trajectory_availability"]
        ),
    )
    require_concentration = concentration.sort_values(
        ["population", "gate"], kind="mergesort"
    ).reset_index(drop=True)
    actual_concentration = reconstructed_concentration.sort_values(
        ["population", "gate"], kind="mergesort"
    ).reset_index(drop=True)
    concentration_identity = require_concentration[["population", "gate", "passed"]].equals(
        actual_concentration[["population", "gate", "passed"]]
    ) and np.allclose(
        require_concentration[["value", "threshold"]].to_numpy(dtype=float),
        actual_concentration[["value", "threshold"]].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-15,
    )
    if not concentration_identity:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            "stored and reconstructed concentration metrics differ",
        )
    real = real_increments(predictions)
    bootstrap_summary = bootstrap_summary_from_artifact(bootstrap_frame)
    null_summary = null_summary_from_artifact(null_frame)
    monthly_counts = monthly_positive_counts(monthly)
    concentration_pass = {
        scope: bool(concentration.loc[concentration["population"].eq(scope), "passed"].all())
        for scope in SCREEN_SCOPES
    }
    decision, _, _ = derive_screen_decision(
        real,
        bootstrap_summary=bootstrap_summary,
        null_summary=null_summary,
        monthly_counts=monthly_counts,
        concentration_pass=concentration_pass,
        supported=supported,
    )
    target_counts = {
        str(key): int(value)
        for key, value in assessment["target_class"].value_counts().sort_index().items()
    }
    decision.update(
        {
            "development_rows": len(development),
            "assessment_rows": len(assessment),
            "assessment_target_counts": target_counts,
            "monthly_positive_log_loss_counts": monthly_counts,
            "support": support,
            "determinism_check_passed": None,
            "lightweight_audit_passed": None,
        }
    )
    source_manifest = read_json_file(output / "source_manifest.json")
    frozen_component_path = resolve_frozen_component_ledger()
    source_manifest["population_reconstruction"] = population_manifest
    source_manifest["frozen_behavioural_component_authority"] = {
        "path": str(frozen_component_path),
        "sha256": EXPECTED_COMPONENT_LEDGER_SHA256,
        "raw_data_downloaded": False,
        "purpose": (
            "exact ordinal-6/12 final-anchor component and scaling authority, including "
            "the frozen causal cross-sectional signed-progress context"
        ),
    }
    source_manifest["trajectory_support"] = support
    write_json(output / "source_manifest.json", source_manifest)
    feature_document = read_json_file(output / "feature_manifest.json")
    feature_document["development_frozen_reporting_medians"] = {
        "posterior_entropy": entropy_median,
        "transition_probability": transition_median,
    }
    write_json(output / "feature_manifest.json", feature_document)
    determinism = determinism_check(
        panel,
        cast(
            dict[str, Mapping[str, Any]],
            read_json_file(output / "model_coefficients.json")["primary_models"],
        ),
        predictions,
        pooled,
        decision=str(decision["decision"]),
        bootstrap_summary=bootstrap_summary,
        null_summary=null_summary,
        concentration_pass=concentration_pass,
        supported=supported,
    )
    decision["determinism_check_passed"] = bool(determinism["passed"])
    write_csv(output / "pooled_metrics.csv", pooled)
    for artifact, frame in breakdowns.items():
        write_csv(output / f"{artifact}.csv", frame)
    write_json(output / "determinism_check.json", determinism)
    write_json(output / "decision.json", decision)
    auditor = load_script("_trajectory_v01_resume_auditor", AUDITOR_PATH)
    audit = cast(dict[str, Any], auditor.audit(output))
    if not audit.get("passed"):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "independent lightweight audit failed"
        )
    decision["lightweight_audit_passed"] = True
    write_json(output / "decision.json", decision)
    anchor_document = read_json_file(output / "checkpoint_anchor_manifest.json")
    reproduction = cast(dict[str, Any], anchor_document["final_anchor_reproduction"])
    report = render_report(
        decision=decision,
        support=support,
        pooled=pooled,
        real=real,
        bootstrap_summary=bootstrap_summary,
        null_summary=null_summary,
        reproduction=reproduction,
        target_counts=target_counts,
        determinism=determinism,
        audit=audit,
    ) + render_detailed_appendix(output, decision)
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


def read_json_file(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_blocker(output: Path, blocker: ScreenBlocker) -> None:
    output.mkdir(parents=True, exist_ok=True)
    decision = {
        **SAFETY_FLAGS,
        "decision": blocker.code,
        "primary_decision": blocker.code,
        "blocker_detail": blocker.detail,
        "feasibility_only": True,
        "validation_or_promotion": False,
    }
    write_json(output / "decision.json", decision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-audit", action="store_true")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    provider_root = args.provider_root.expanduser().resolve()
    try:
        decision = (
            resume_audit_and_report(output)
            if args.resume_audit
            else execute_screen(output, provider_root=provider_root)
        )
        print(canonical_json(decision), end="")
        return 0
    except ScreenBlocker as blocker:
        write_blocker(output, blocker)
        print(blocker.code)
        print(blocker.detail, file=sys.stderr)
        return 2
    except Exception as error:
        blocker = ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            f"unexpected fail-closed error: {type(error).__name__}: {error}",
        )
        write_blocker(output, blocker)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
