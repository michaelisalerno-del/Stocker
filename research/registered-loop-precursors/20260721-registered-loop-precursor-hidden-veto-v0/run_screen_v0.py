#!/usr/bin/env python3
"""Run the Registered-Loop Precursors and Hidden-Diversion Veto screen V0."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-registered-precursor-veto-mpl")

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.hidden_loop_economics_registered_bridge_v0 import (
    binary_model_metrics,
    expanding_logistic_crossfit,
    fit_weighted_logistic,
    reconstruct_serialised_probability,
)
from stocker_research.loop_prefix_automaton_v2 import FirstNextLoopEventEngine
from stocker_research.opening_trajectory_unregistered_families_v0 import (
    canonical_unregistered_path,
    pool_hidden_family,
)
from stocker_research.registered_loop_precursor_hidden_veto_v0 import (
    FROZEN_HIDDEN_FAMILIES,
    OPENING_KEYS,
    OTHER_HIDDEN_FAMILY,
    assign_hidden_risk,
    benjamini_hochberg,
    candidate_threshold,
    choose_primary_decision,
    deduplicate_registered_completions,
    exact_precursor_identity_eligible,
    freeze_hidden_risk_thresholds,
    opening_panel_differences,
    permute_hidden_probability_within_slates,
    precursor_window_features,
    reject_protected_dates,
    sample_matched_pseudo_completions,
    session_block_bootstrap_indices,
    veto_feature_frame,
)

CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
BRIDGE_DIR = (
    REPO_ROOT
    / "research"
    / "hidden-loop-economics"
    / "20260721-hidden-loop-economics-registered-bridge-v0"
)
BRIDGE_PRIMARY = BRIDGE_DIR / "artifacts" / "primary"
BRIDGE_RUNNER = BRIDGE_DIR / "run_screen_v0.py"
OPENING_DIR = (
    REPO_ROOT
    / "research"
    / "unregistered-loop-families"
    / "20260721-opening-trajectory-unregistered-families-v0"
)
OPENING_PRIMARY = OPENING_DIR / "artifacts" / "primary"
OPENING_RUNNER = OPENING_DIR / "run_screen_v0.py"
V2_RUNNER = (
    REPO_ROOT
    / "research"
    / "loop-funnel"
    / "20260721-emotion-regime-coarse-loop-family-v0"
    / "run_screen_v0.py"
)

DEVELOPMENT_START = pd.Timestamp("2024-01-01T00:00:00Z")
ASSESSMENT_START = pd.Timestamp("2025-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
LOOKBACKS = (3, 6, 12)
PRECURSOR_NULL_DRAWS = 25
BOOTSTRAP_DRAWS = 25
VETO_NULL_DRAWS = 5
PRECURSOR_NULL_SEED = 20260721
BOOTSTRAP_SEED = 20260722
VETO_NULL_SEED = 20260723
SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "registered_loop_precursor_census": True,
    "hidden_route_diversion_veto_test": True,
    "economic_outcomes_opened": False,
    "directional_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
BOOLEAN_PRECURSORS = (
    "same_registered_identity",
    "same_registered_broad_family_different_identity",
    "different_registered_broad_family",
    "any_prior_registered_completion",
    "hidden_5_6_5",
    "hidden_2_3_2",
    "hidden_2_5_2",
    "hidden_4_7_4",
    "hidden_other_unregistered_family",
    "any_hidden_unregistered_completion",
    "active_prefix_immediately_before_completion",
    "active_prefix_any",
    "matching_prefix_any",
    "other_prefix_any",
    "any_regime_transition",
    "no_identified_structural_precursor",
)
BROAD_GATE_PRECURSORS = (
    "matching_prefix_any",
    "other_prefix_any",
    "any_prior_registered_completion",
    "any_hidden_unregistered_completion",
    "any_regime_transition",
)
V0_FEATURES = ("logit_B0_probability", "checkpoint_12")
V1_FEATURES = (*V0_FEATURES, "logit_U1_probability")


class ScreenBlocker(RuntimeError):
    """Fail-closed blocker carrying one preregistered decision code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


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
    hard = contract["hard_limits"]
    required_limits = {
        "n_jobs": 1,
        "maximum_new_assessment_models": 2,
        "maximum_expanding_development_folds": 4,
        "session_bootstrap_draws": 25,
        "precursor_null_draws": 25,
        "hidden_probability_null_refits": 5,
        "maximum_plots": 2,
    }
    if any(hard.get(key) != value for key, value in required_limits.items()):
        raise ScreenBlocker(
            "blocked_quick_precursor_veto_resource_limit",
            "contract hard speed limits differ from the executable",
        )
    return contract


def reconstruct_opening_population() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    tuple[str, ...],
]:
    predecessor = load_module(OPENING_RUNNER, "precursor_veto_opening_predecessor")
    bridge = load_module(BRIDGE_RUNNER, "precursor_veto_bridge_predecessor")
    opening, hidden_events, predecessor_reconstruction, t0_features, _ = (
        bridge.reconstruct_frozen_population(predecessor)
    )
    completions = pd.read_parquet(BRIDGE_PRIMARY / "registered_completion_ledger.parquet")
    completion_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in completions.groupby(["symbol", "session"], sort=False)
    }
    targets: list[int] = []
    for row in opening.itertuples(index=False):
        group = completion_groups.get(
            (str(row.symbol), str(row.session)),
            pd.DataFrame(columns=["completion_bar_ordinal"]),
        )
        ordinal = group["completion_bar_ordinal"].to_numpy(dtype=int)
        target = np.any(
            (ordinal > int(row.repo_bar_start_ordinal))
            & (ordinal <= int(row.repo_bar_start_ordinal) + 12)
        )
        targets.append(int(target))
    opening["registered_completion_within_12_bars"] = targets
    opening["actual_hidden_event_within_6_bars"] = opening["unregistered_event"].eq(1.0)
    slate_counts = opening.groupby("slate_id", sort=True).size()
    opening["bridge_row_weight"] = opening["slate_id"].map((1.0 / slate_counts).to_dict())
    opening["row_weight"] = opening["bridge_row_weight"]
    opening["p_unregistered_within_6_bars"] = opening["U1_probability"]

    coefficients = read_json(BRIDGE_PRIMARY / "bridge_model_coefficients.json")
    for model in ("B0", "B1"):
        opening[f"{model}_probability"] = reconstruct_serialised_probability(
            opening, coefficients[model]
        )
    development_archived = pd.read_parquet(BRIDGE_PRIMARY / "bridge_development_panel.parquet")
    assessment_archived = pd.read_parquet(BRIDGE_PRIMARY / "bridge_assessment_predictions.parquet")
    development_archived["period"] = "development"
    assessment_archived["period"] = "assessment"
    opening["period"] = np.where(opening["year"].eq(2024), "development", "assessment")

    shared_fields = tuple(
        dict.fromkeys(
            [
                *t0_features,
                "registered_completion_within_12_bars",
                "actual_hidden_event_within_6_bars",
                "row_weight",
            ]
        )
    )
    assessment_reconstructed = opening.loc[opening["year"].eq(2025)].copy()
    assessment_comparison = opening_panel_differences(
        assessment_archived,
        assessment_reconstructed,
        shared_fields=shared_fields,
        probability_fields=("B0_probability", "B1_probability", "U1_probability"),
    )

    timestamp_fields = tuple(
        field
        for field in (
            "decision_timestamp_utc",
            "feature_available_timestamp_utc",
            "decision_bar_start_timestamp_utc",
            "bar_start_timestamp",
            "bar_complete_timestamp",
        )
        if field in opening.columns
        and field in assessment_archived.columns
        and field in development_archived.columns
    )

    def timestamp_mismatches(archived: pd.DataFrame, reconstructed: pd.DataFrame) -> int:
        comparison = archived.loc[:, [*OPENING_KEYS, *timestamp_fields]].merge(
            reconstructed.loc[:, [*OPENING_KEYS, *timestamp_fields]],
            on=list(OPENING_KEYS),
            how="outer",
            suffixes=("_archived", "_reconstructed"),
            indicator=True,
            validate="one_to_one",
        )
        mismatches = int((~comparison["_merge"].eq("both")).sum())
        for field in timestamp_fields:
            left = pd.to_datetime(comparison[f"{field}_archived"], utc=True, errors="raise")
            right = pd.to_datetime(comparison[f"{field}_reconstructed"], utc=True, errors="raise")
            mismatches += int((left != right).sum())
        return mismatches

    timestamp_mismatch_count = timestamp_mismatches(
        assessment_archived, assessment_reconstructed
    ) + timestamp_mismatches(
        development_archived,
        opening.loc[opening["year"].eq(2024)].merge(
            development_archived.loc[:, list(OPENING_KEYS)],
            on=list(OPENING_KEYS),
            how="inner",
            validate="one_to_one",
        ),
    )
    development_reconstructed = development_archived.loc[
        :, list(development_archived.columns)
    ].merge(
        opening.loc[
            :,
            [
                "symbol",
                "session",
                "decision_ordinal",
                "U1_probability",
            ],
        ],
        on=["symbol", "session", "decision_ordinal"],
        how="left",
        suffixes=("_archived", "_reconstructed"),
        validate="one_to_one",
    )
    development_probability_difference = 0.0
    for field in ("B0_probability", "B1_probability"):
        reconstructed = reconstruct_serialised_probability(
            development_archived, coefficients[field.removesuffix("_probability")]
        )
        development_probability_difference = max(
            development_probability_difference,
            float(np.max(np.abs(development_archived[field].to_numpy(float) - reconstructed))),
        )
    development_probability_difference = max(
        development_probability_difference,
        float(
            np.max(
                np.abs(
                    development_reconstructed["U1_probability_archived"].to_numpy(float)
                    - development_reconstructed["U1_probability_reconstructed"].to_numpy(float)
                )
            )
        ),
    )
    development_cross_fitted_u1_alias_difference = float(
        np.max(
            np.abs(
                development_archived["p_unregistered_within_6_bars"].to_numpy(float)
                - development_archived["oof_p_unregistered_within_6_bars"].to_numpy(float)
            ),
            initial=0.0,
        )
    )
    development_probability_difference = max(
        development_probability_difference,
        development_cross_fitted_u1_alias_difference,
    )
    target_manifest = read_json(BRIDGE_PRIMARY / "bridge_target_manifest.json")
    if len(opening) != int(target_manifest["rows_after_source_availability"]) or int(
        opening["registered_completion_within_12_bars"].sum()
    ) != int(
        opening.loc[opening["year"].eq(2024), "registered_completion_within_12_bars"].sum()
        + target_manifest["assessment_positive_targets"]
    ):
        raise ScreenBlocker(
            "blocked_predecessor_panel_not_reconstructable",
            "registered target population differs from predecessor",
        )
    maximum_shared = float(assessment_comparison["maximum_shared_field_difference"])
    maximum_probability = max(
        float(assessment_comparison["maximum_probability_difference"]),
        development_probability_difference,
    )
    reconstruction = {
        **SAFETY_FLAGS,
        "passed": maximum_shared <= 1e-12
        and maximum_probability <= 1e-12
        and timestamp_mismatch_count == 0,
        "opening_rows": int(len(opening)),
        "development_rows": int(opening["year"].eq(2024).sum()),
        "assessment_rows": int(opening["year"].eq(2025).sum()),
        "sessions": int(opening["session"].nunique()),
        "stocks": int(opening["symbol"].nunique()),
        "rows_by_checkpoint": {
            str(key): int(value)
            for key, value in opening["decision_ordinal"].value_counts().sort_index().items()
        },
        "maximum_shared_field_difference": maximum_shared,
        "maximum_probability_difference": maximum_probability,
        "decision_timestamp_fields": list(timestamp_fields),
        "decision_timestamp_mismatches": timestamp_mismatch_count,
        "development_cross_fitted_U1_reused_from_predecessor": True,
        "development_cross_fitted_U1_alias_maximum_difference": (
            development_cross_fitted_u1_alias_difference
        ),
        "development_cross_fitted_U1_rows": int(
            development_archived["p_unregistered_within_6_bars"].notna().sum()
        ),
        "population_key_sha256": frame_hash(opening, ["symbol", "session", "decision_ordinal"]),
        "predecessor_reconstruction": predecessor_reconstruction,
        "tolerance": 1e-12,
    }
    if not reconstruction["passed"]:
        raise ScreenBlocker(
            "blocked_predecessor_panel_not_reconstructable",
            f"opening reconstruction differences={maximum_shared}/{maximum_probability}",
        )
    return (
        opening,
        development_archived,
        assessment_archived,
        completions,
        reconstruction,
        tuple(str(value) for value in t0_features),
    )


def build_completion_event_ledger(
    completions: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    year: int,
    period: str,
) -> pd.DataFrame:
    """Build one deduplicated registered-completion ledger for a frozen period."""

    ledger = deduplicate_registered_completions(
        completions, decisions.loc[decisions["year"].eq(year)]
    )
    ledger["event_id"] = (
        ledger["symbol"].astype(str)
        + "|"
        + ledger["session"].astype(str)
        + "|"
        + ledger["completion_timestamp_utc"].astype(str)
        + "|"
        + ledger["semantic_loop_id"].astype(str)
    )
    ledger["clock_bin"] = (
        pd.to_datetime(ledger["completion_timestamp_utc"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.floor("30min")
        .dt.strftime("%H:%M")
    )
    ledger["period"] = period
    return ledger


def load_v2_states(
    provider_root: Path,
) -> tuple[pd.DataFrame, Any, dict[str, Any], dict[str, Any]]:
    runner = load_module(V2_RUNNER, "precursor_veto_v2_runner")
    preprocessing, parameters = runner.load_frozen_model()
    runner.MAX_TARGET_BAR_ORDINAL = 30
    states, source = runner.build_v2_state_panel(provider_root, preprocessing, parameters)
    dictionary, dictionary_manifest = runner.load_loop_dictionary()
    states["posterior_entropy"] = states["posterior_entropy_reproduced"].astype(float)
    probability_columns = [f"state_p_{state}" for state in range(8)]
    states["top_state_probability"] = states.loc[:, probability_columns].max(axis=1)
    states = states.sort_values(["symbol", "session", "bar_ordinal"], kind="mergesort")
    reject_protected_dates(states, column="bar_start_timestamp")
    return states.reset_index(drop=True), dictionary, source, dictionary_manifest


def build_null_eligibility(
    states: pd.DataFrame, completions: pd.DataFrame, *, period: str
) -> pd.DataFrame:
    registered_keys = set(
        zip(
            completions["symbol"].astype(str),
            completions["session"].astype(str),
            completions["completion_bar_ordinal"].astype(int),
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    if period == "development":
        period_states = states.loc[states["session"].astype(str).lt("2025-01-01")]
    elif period == "assessment":
        period_states = states.loc[states["session"].astype(str).ge("2025-01-01")]
    else:
        raise ValueError(f"unknown null-eligibility period: {period}")
    for (symbol, session), group in period_states.groupby(["symbol", "session"], sort=True):
        ordinals = set(group["bar_ordinal"].astype(int))
        for row in group.itertuples(index=False):
            ordinal = int(row.bar_ordinal)
            full_history = set(range(ordinal - 12, ordinal)).issubset(ordinals)
            timestamp = pd.Timestamp(row.bar_start_timestamp)
            rows.append(
                {
                    "symbol": str(symbol),
                    "session": str(session),
                    "year_month": str(session)[:7],
                    "clock_bin": timestamp.tz_convert("America/New_York")
                    .floor("30min")
                    .strftime("%H:%M"),
                    "completion_bar_ordinal": ordinal,
                    "completion_timestamp_utc": timestamp,
                    "full_prior_history": bool(full_history),
                    "registered_completion_at_timestamp": (str(symbol), str(session), ordinal)
                    in registered_keys,
                }
            )
    return pd.DataFrame(rows)


def build_null_targets(
    assessment_events: pd.DataFrame,
    development_events: pd.DataFrame,
    assessment_eligible: pd.DataFrame,
    development_eligible: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    draws: list[pd.DataFrame] = []
    manifests: dict[str, Any] = {}
    period_sources = (
        ("development", development_events, development_eligible),
        ("assessment", assessment_events, assessment_eligible),
    )
    for draw in range(PRECURSOR_NULL_DRAWS):
        for period, events, eligible in period_sources:
            full_history_events = events.loc[
                events["completion_bar_ordinal"].astype(int).ge(12)
            ].copy()
            sampled = sample_matched_pseudo_completions(
                full_history_events, eligible, seed=PRECURSOR_NULL_SEED + draw
            )
            sampled["record_type"] = "null"
            sampled["draw"] = draw
            sampled["event_id"] = sampled["source_event_id"].astype(str) + f"|NULL|{draw:02d}"
            sampled["period"] = period
            draws.append(sampled)
            if draw == 0:
                manifests[period] = {
                    "observed_events": int(len(events)),
                    "matched_full_history_events": int(len(full_history_events)),
                    "unmatched_incomplete_history_events": int(
                        len(events) - len(full_history_events)
                    ),
                    "matched_coverage": float(len(full_history_events) / len(events)),
                    "all_registered_completions_matched": len(full_history_events) == len(events),
                }
    return pd.concat(draws, ignore_index=True), {
        "draws": PRECURSOR_NULL_DRAWS,
        "seed": PRECURSOR_NULL_SEED,
        "draw_definition": "each draw samples both development and assessment events",
        "periods": manifests,
        "assessment_inference_restricted_to_full_history_subset": True,
    }


def _trace_prefix_frame(group: pd.DataFrame, trace: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = group.sort_values("bar_ordinal", kind="mergesort")
    hard = ordered["causal_hard_state"].to_numpy(dtype=int)
    event_index = np.cumsum(np.concatenate(([True], hard[1:] != hard[:-1]))).astype(int) - 1
    for position, (_, bar) in enumerate(ordered.iterrows()):
        for prefix in trace.prefixes_after_event[int(event_index[position])]:
            rows.append(
                {
                    "bar_ordinal": int(bar["bar_ordinal"]),
                    "semantic_loop_id": str(prefix.semantic_loop_id),
                    "orientation_id": str(prefix.orientation_id),
                    "progress_states": int(prefix.progress_states),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["bar_ordinal", "semantic_loop_id", "orientation_id", "progress_states"],
    )


def _trace_hidden_frame(trace: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in trace.unregistered_completions:
        canonical = canonical_unregistered_path(event.full_path)
        rows.append(
            {
                "completion_bar_ordinal": int(event.completion_bar_ordinal),
                "family_id": str(canonical.family_id),
                "hidden_family_class": pool_hidden_family(
                    canonical.family_id, FROZEN_HIDDEN_FAMILIES
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["completion_bar_ordinal", "family_id", "hidden_family_class"])
    return (
        pd.DataFrame(rows)
        .drop_duplicates(["completion_bar_ordinal", "family_id"])
        .sort_values(["completion_bar_ordinal", "family_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_precursor_feature_ledger(
    states: pd.DataFrame,
    dictionary: Any,
    completions: pd.DataFrame,
    observed_development: pd.DataFrame,
    observed_assessment: pd.DataFrame,
    null_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    development_specs = observed_development.copy()
    development_specs["record_type"] = "observed"
    development_specs["draw"] = -1
    development_specs["source_event_id"] = development_specs["event_id"]
    assessment_specs = observed_assessment.copy()
    assessment_specs["record_type"] = "observed"
    assessment_specs["draw"] = -1
    assessment_specs["source_event_id"] = assessment_specs["event_id"]
    specs = pd.concat(
        [development_specs, assessment_specs, null_targets], ignore_index=True, sort=False
    )
    specs_by_group = {
        (str(symbol), str(session)): group
        for (symbol, session), group in specs.groupby(["symbol", "session"], sort=False)
    }
    registered_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in completions.groupby(["symbol", "session"], sort=False)
    }
    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    feature_rows: list[dict[str, Any]] = []
    trace_registered_rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, int, str, str, int], dict[str, Any]] = {}
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
            trace_registered_rows.append(
                {
                    "symbol": str(symbol),
                    "session": str(session),
                    "completion_bar_ordinal": int(event.completion_bar_ordinal),
                    "semantic_loop_id": str(event.semantic_loop_id),
                    "orientation_id": str(event.orientation_id),
                }
            )
        target_group = specs_by_group.get((str(symbol), str(session)))
        if target_group is None:
            continue
        prefix_frame = _trace_prefix_frame(ordered, trace)
        hidden_frame = _trace_hidden_frame(trace)
        registered_frame = registered_groups.get(
            (str(symbol), str(session)),
            pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id", "motif_type"]),
        )
        state_frame = ordered.loc[
            :,
            [
                "bar_ordinal",
                "causal_hard_state",
                "transition_probability",
                "posterior_entropy",
                "top_state_probability",
                "expected_state_age",
            ],
        ]
        for target in target_group.itertuples(index=False):
            for lookback in LOOKBACKS:
                cache_key = (
                    str(symbol),
                    str(session),
                    int(target.completion_bar_ordinal),
                    str(target.semantic_loop_id),
                    str(target.motif_type),
                    lookback,
                )
                if cache_key not in cache:
                    cache[cache_key] = precursor_window_features(
                        pd.Series(
                            {
                                "completion_bar_ordinal": int(target.completion_bar_ordinal),
                                "semantic_loop_id": str(target.semantic_loop_id),
                                "motif_type": str(target.motif_type),
                            }
                        ),
                        registered_frame,
                        hidden_frame,
                        prefix_frame,
                        state_frame,
                        lookback_bars=lookback,
                    )
                row = {
                    "record_type": str(target.record_type),
                    "period": str(target.period),
                    "draw": int(target.draw),
                    "event_id": str(target.event_id),
                    "source_event_id": str(target.source_event_id),
                    "symbol": str(symbol),
                    "session": str(session),
                    "year_month": str(target.year_month),
                    "clock_bin": str(target.clock_bin),
                    "decision_ordinal": int(target.decision_ordinal),
                    "completion_bar_ordinal": int(target.completion_bar_ordinal),
                    "completion_timestamp_utc": pd.Timestamp(target.completion_timestamp_utc),
                    "target_semantic_loop_id": str(target.semantic_loop_id),
                    "target_motif_type": str(target.motif_type),
                    **cache[cache_key],
                }
                feature_rows.append(row)
    frozen_identity = [
        "symbol",
        "session",
        "completion_bar_ordinal",
        "semantic_loop_id",
        "orientation_id",
    ]
    trace_registered = pd.DataFrame(trace_registered_rows).drop_duplicates(frozen_identity)
    frozen_registered = completions.loc[:, frozen_identity].drop_duplicates(frozen_identity)
    trace_hash = frame_hash(trace_registered, frozen_identity)
    frozen_hash = frame_hash(frozen_registered, frozen_identity)
    if len(trace_registered) != len(frozen_registered) or trace_hash != frozen_hash:
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure",
            "registered semantic event trace differs from frozen bridge ledger",
        )
    ledger = pd.DataFrame(feature_rows).sort_values(
        ["period", "record_type", "draw", "event_id", "lookback_bars"], kind="mergesort"
    )
    return ledger.reset_index(drop=True), {
        "trace_registered_rows": int(len(trace_registered)),
        "frozen_registered_rows": int(len(frozen_registered)),
        "trace_identity_sha256": trace_hash,
        "frozen_identity_sha256": frozen_hash,
        "prefix_taxonomy_rows": int(len(ledger)),
    }


def precursor_census_tables(
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed = ledger.loc[
        ledger["record_type"].eq("observed") & ledger["period"].eq("assessment")
    ].copy()
    null = ledger.loc[ledger["record_type"].eq("null") & ledger["period"].eq("assessment")].copy()
    matched_ids = set(null["source_event_id"].astype(str))
    matched_observed = observed.loc[observed["event_id"].astype(str).isin(matched_ids)]
    census_rows: list[dict[str, Any]] = []
    for lookback in LOOKBACKS:
        group = observed.loc[observed["lookback_bars"].eq(lookback)]
        for precursor in BOOLEAN_PRECURSORS:
            census_rows.append(
                {
                    "lookback_bars": lookback,
                    "precursor_type": precursor,
                    "events": int(len(group)),
                    "occurrences": int(group[precursor].astype(bool).sum()),
                    "observed_prevalence": float(group[precursor].astype(bool).mean()),
                }
            )
    census = pd.DataFrame(census_rows)
    nearest = (
        observed.groupby(["lookback_bars", "nearest_precursor_label"], sort=True)
        .size()
        .rename("events")
        .reset_index()
    )
    nearest["prevalence"] = nearest["events"] / nearest.groupby("lookback_bars")[
        "events"
    ].transform("sum")

    null_rows: list[dict[str, Any]] = []
    for lookback in LOOKBACKS:
        observed_group = matched_observed.loc[matched_observed["lookback_bars"].eq(lookback)]
        for precursor in BOOLEAN_PRECURSORS:
            observed_prevalence = float(observed_group[precursor].astype(bool).mean())
            draw_values = (
                null.loc[null["lookback_bars"].eq(lookback)]
                .groupby("draw", sort=True)[precursor]
                .mean()
                .reindex(range(PRECURSOR_NULL_DRAWS))
            )
            if draw_values.isna().any():
                raise ScreenBlocker(
                    "blocked_reproducibility_or_audit_failure",
                    "precursor null draw is incomplete",
                )
            for draw, value in draw_values.items():
                null_rows.append(
                    {
                        "record_type": "draw",
                        "draw": int(draw),
                        "lookback_bars": lookback,
                        "precursor_type": precursor,
                        "observed_prevalence": observed_prevalence,
                        "null_prevalence": float(value),
                        "mean_null_prevalence": math.nan,
                        "observed_minus_null_enrichment": observed_prevalence - float(value),
                        "null_percentile": math.nan,
                    }
                )
            values = draw_values.to_numpy(float)
            mean_null = float(values.mean())
            null_rows.append(
                {
                    "record_type": "summary",
                    "draw": -1,
                    "lookback_bars": lookback,
                    "precursor_type": precursor,
                    "observed_prevalence": observed_prevalence,
                    "null_prevalence": math.nan,
                    "mean_null_prevalence": mean_null,
                    "observed_minus_null_enrichment": observed_prevalence - mean_null,
                    "null_percentile": float(100.0 * np.mean(values <= observed_prevalence)),
                    "null_90th_percentile": float(np.quantile(values, 0.90)),
                }
            )
    null_metrics = pd.DataFrame(null_rows)

    def stability_table(group_column: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        groups = sorted(matched_observed[group_column].unique().tolist())
        for value in groups:
            for lookback in LOOKBACKS:
                observed_group = matched_observed.loc[
                    matched_observed[group_column].eq(value)
                    & matched_observed["lookback_bars"].eq(lookback)
                ]
                null_group = null.loc[
                    null[group_column].eq(value) & null["lookback_bars"].eq(lookback)
                ]
                for precursor in BOOLEAN_PRECURSORS:
                    observed_prevalence = float(observed_group[precursor].astype(bool).mean())
                    draw_prevalence = null_group.groupby("draw", sort=True)[precursor].mean()
                    mean_null = float(draw_prevalence.mean())
                    rows.append(
                        {
                            group_column: value,
                            "lookback_bars": lookback,
                            "precursor_type": precursor,
                            "events": int(len(observed_group)),
                            "observed_prevalence": observed_prevalence,
                            "mean_null_prevalence": mean_null,
                            "enrichment": observed_prevalence - mean_null,
                        }
                    )
        return pd.DataFrame(rows)

    monthly = stability_table("year_month")
    checkpoint = stability_table("decision_ordinal")
    return census, nearest, monthly, checkpoint, null_metrics


def exact_transition_tables(
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    relationship_rows: list[dict[str, Any]] = []
    for row in ledger.itertuples(index=False):
        completed = json.loads(str(row.registered_precursors_json))
        relationships = {(str(item["kind"]), str(item["identity"])) for item in completed}
        for kind, identity in sorted(relationships):
            relationship_rows.append(
                {
                    "record_type": str(row.record_type),
                    "period": str(row.period),
                    "draw": int(row.draw),
                    "event_id": str(row.event_id),
                    "source_event_id": str(row.source_event_id),
                    "symbol": str(row.symbol),
                    "session": str(row.session),
                    "year_month": str(row.year_month),
                    "lookback_bars": int(row.lookback_bars),
                    "precursor_kind": kind,
                    "precursor_identity": identity,
                    "target_semantic_loop_id": str(row.target_semantic_loop_id),
                }
            )
    relationships = pd.DataFrame(relationship_rows)
    observed = relationships.loc[relationships["record_type"].eq("observed")]
    counts_rows: list[dict[str, Any]] = []
    group_keys = [
        "period",
        "lookback_bars",
        "precursor_kind",
        "precursor_identity",
        "target_semantic_loop_id",
    ]
    for key, group in observed.groupby(group_keys, sort=True):
        period, lookback, kind, precursor, target = key
        unique = group.drop_duplicates("event_id")
        stock_share = unique["symbol"].value_counts(normalize=True)
        month_share = unique["year_month"].value_counts(normalize=True)
        denominator = int(
            ledger.loc[
                ledger["record_type"].eq("observed")
                & ledger["period"].eq(period)
                & ledger["lookback_bars"].eq(lookback)
            ]["event_id"].nunique()
        )
        counts_rows.append(
            {
                "period": period,
                "lookback_bars": int(lookback),
                "precursor_kind": kind,
                "precursor_identity": precursor,
                "target_semantic_loop_id": target,
                "occurrences": int(unique["event_id"].nunique()),
                "eligible_events": denominator,
                "prevalence": float(unique["event_id"].nunique() / denominator),
                "sessions": int(unique["session"].nunique()),
                "stocks": int(unique["symbol"].nunique()),
                "months": int(unique["year_month"].nunique()),
                "maximum_stock_share": float(stock_share.max()),
                "maximum_month_share": float(month_share.max()),
            }
        )
    counts = pd.DataFrame(counts_rows)
    if counts.empty:
        return counts, pd.DataFrame()
    assessment = counts.loc[
        counts["period"].eq("assessment")
        & counts.apply(
            lambda row: exact_precursor_identity_eligible(
                str(row["precursor_kind"]), str(row["precursor_identity"])
            ),
            axis=1,
        )
    ].copy()
    development_lookup = counts.loc[counts["period"].eq("development")].set_index(
        ["lookback_bars", "precursor_kind", "precursor_identity", "target_semantic_loop_id"]
    )
    assessment_null = relationships.loc[
        relationships["record_type"].eq("null") & relationships["period"].eq("assessment")
    ]
    development_null = relationships.loc[
        relationships["record_type"].eq("null") & relationships["period"].eq("development")
    ]
    assessment_matched_source_ids = set(
        ledger.loc[
            ledger["record_type"].eq("null") & ledger["period"].eq("assessment"),
            "source_event_id",
        ].astype(str)
    )
    development_matched_source_ids = set(
        ledger.loc[
            ledger["record_type"].eq("null") & ledger["period"].eq("development"),
            "source_event_id",
        ].astype(str)
    )
    multiplicity_rows: list[dict[str, Any]] = []
    assessment_matched_denominator = int(
        ledger.loc[
            ledger["record_type"].eq("observed")
            & ledger["period"].eq("assessment")
            & ledger["lookback_bars"].eq(12)
            & ledger["complete_prior_history"].astype(bool)
        ]["event_id"].nunique()
    )
    development_matched_denominator = int(
        ledger.loc[
            ledger["record_type"].eq("observed")
            & ledger["period"].eq("development")
            & ledger["lookback_bars"].eq(12)
            & ledger["complete_prior_history"].astype(bool)
        ]["event_id"].nunique()
    )
    for row in assessment.itertuples(index=False):
        key = (
            int(row.lookback_bars),
            str(row.precursor_kind),
            str(row.precursor_identity),
            str(row.target_semantic_loop_id),
        )
        development = development_lookup.loc[key] if key in development_lookup.index else None
        matched_observed_group = observed.loc[
            observed["period"].eq("assessment")
            & observed["lookback_bars"].eq(row.lookback_bars)
            & observed["precursor_kind"].eq(row.precursor_kind)
            & observed["precursor_identity"].eq(row.precursor_identity)
            & observed["target_semantic_loop_id"].eq(row.target_semantic_loop_id)
            & observed["event_id"].astype(str).isin(assessment_matched_source_ids)
        ]
        matched_observed_count = int(matched_observed_group["event_id"].nunique())
        matched_observed_prevalence = matched_observed_count / assessment_matched_denominator
        assessment_draw_counts = []
        development_draw_counts = []
        for draw in range(PRECURSOR_NULL_DRAWS):
            assessment_draw = assessment_null.loc[
                assessment_null["draw"].eq(draw)
                & assessment_null["lookback_bars"].eq(row.lookback_bars)
                & assessment_null["precursor_kind"].eq(row.precursor_kind)
                & assessment_null["precursor_identity"].eq(row.precursor_identity)
                & assessment_null["target_semantic_loop_id"].eq(row.target_semantic_loop_id)
            ]
            development_draw = development_null.loc[
                development_null["draw"].eq(draw)
                & development_null["lookback_bars"].eq(row.lookback_bars)
                & development_null["precursor_kind"].eq(row.precursor_kind)
                & development_null["precursor_identity"].eq(row.precursor_identity)
                & development_null["target_semantic_loop_id"].eq(row.target_semantic_loop_id)
            ]
            assessment_draw_counts.append(int(assessment_draw["event_id"].nunique()))
            development_draw_counts.append(int(development_draw["event_id"].nunique()))
        observed_count = int(row.occurrences)
        p_value = float(
            (1 + sum(value >= matched_observed_count for value in assessment_draw_counts)) / 26.0
        )
        mean_null_prevalence = float(
            np.mean(assessment_draw_counts) / assessment_matched_denominator
        )
        development_occurrences = int(development["occurrences"]) if development is not None else 0
        development_observed_group = observed.loc[
            observed["period"].eq("development")
            & observed["lookback_bars"].eq(row.lookback_bars)
            & observed["precursor_kind"].eq(row.precursor_kind)
            & observed["precursor_identity"].eq(row.precursor_identity)
            & observed["target_semantic_loop_id"].eq(row.target_semantic_loop_id)
            & observed["event_id"].astype(str).isin(development_matched_source_ids)
        ]
        development_matched_occurrences = int(development_observed_group["event_id"].nunique())
        development_matched_prevalence = (
            development_matched_occurrences / development_matched_denominator
        )
        development_mean_null_prevalence = float(
            np.mean(development_draw_counts) / development_matched_denominator
        )
        development_enrichment = development_matched_prevalence - development_mean_null_prevalence
        assessment_enrichment = matched_observed_prevalence - mean_null_prevalence
        support = (
            development_occurrences >= 30
            and observed_count >= 20
            and int(row.sessions) >= 10
            and int(row.stocks) >= 8
        )
        multiplicity_rows.append(
            {
                **{column: getattr(row, column) for column in assessment.columns},
                "development_occurrences": development_occurrences,
                "development_matched_occurrences": development_matched_occurrences,
                "development_matched_prevalence": development_matched_prevalence,
                "development_mean_null_occurrences": float(np.mean(development_draw_counts)),
                "development_mean_null_prevalence": development_mean_null_prevalence,
                "development_enrichment": development_enrichment,
                "matched_assessment_occurrences": matched_observed_count,
                "matched_assessment_prevalence": matched_observed_prevalence,
                "mean_null_occurrences": float(np.mean(assessment_draw_counts)),
                "mean_null_prevalence": mean_null_prevalence,
                "assessment_enrichment": float(assessment_enrichment),
                "same_direction_development_assessment": bool(
                    development is not None
                    and development_enrichment > 0.0
                    and assessment_enrichment > 0.0
                ),
                "support_passed": support,
                "p_value": p_value,
            }
        )
    multiplicity = pd.DataFrame(multiplicity_rows)
    eligible_mask = multiplicity["support_passed"].astype(bool)
    multiplicity["q_value"] = math.nan
    if eligible_mask.any():
        multiplicity.loc[eligible_mask, "q_value"] = benjamini_hochberg(
            multiplicity.loc[eligible_mask, "p_value"].astype(float).tolist()
        )
    multiplicity["q_le_0_10"] = multiplicity["q_value"].le(0.10)
    return counts, multiplicity


def evaluate_precursor_gate(
    event_ledger: pd.DataFrame,
    feature_ledger: pd.DataFrame,
    null_metrics: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    multiplicity: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    assessment_12_bar = feature_ledger.loc[
        feature_ledger["record_type"].eq("observed")
        & feature_ledger["period"].eq("assessment")
        & feature_ledger["lookback_bars"].eq(12)
    ]
    complete_history_events = int(assessment_12_bar["complete_prior_history"].astype(bool).sum())
    complete_rate = float(complete_history_events / len(assessment_12_bar))
    support = {
        "unique_completions_at_least_500": len(event_ledger) >= 500,
        "sessions_at_least_120": event_ledger["session"].nunique() >= 120,
        "stocks_at_least_15": event_ledger["symbol"].nunique() >= 15,
        "months_at_least_6": event_ledger["year_month"].nunique() >= 6,
        "complete_12_bar_history_at_least_90pct": complete_rate >= 0.90,
    }
    broad_results: list[dict[str, Any]] = []
    summaries = null_metrics.loc[
        null_metrics["record_type"].eq("summary")
        & null_metrics["precursor_type"].isin(BROAD_GATE_PRECURSORS)
    ]
    for row in summaries.itertuples(index=False):
        checkpoint_group = checkpoint.loc[
            checkpoint["lookback_bars"].eq(row.lookback_bars)
            & checkpoint["precursor_type"].eq(row.precursor_type)
        ]
        month_group = monthly.loc[
            monthly["lookback_bars"].eq(row.lookback_bars)
            & monthly["precursor_type"].eq(row.precursor_type)
        ]
        checks = {
            "positive_enrichment": float(row.observed_minus_null_enrichment) > 0.0,
            "above_90th_null_percentile": float(row.observed_prevalence)
            > float(row.null_90th_percentile),
            "same_sign_both_checkpoints": len(checkpoint_group) == 2
            and bool(checkpoint_group["enrichment"].gt(0.0).all()),
            "positive_in_five_months": int(month_group["enrichment"].gt(0.0).sum()) >= 5,
        }
        broad_results.append(
            {
                "lookback_bars": int(row.lookback_bars),
                "precursor_type": str(row.precursor_type),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    exact_passed = False
    exact_supported = multiplicity.loc[
        multiplicity.get("support_passed", pd.Series(False, index=multiplicity.index)).astype(bool)
        & multiplicity.get("q_le_0_10", pd.Series(False, index=multiplicity.index)).astype(bool)
        & multiplicity.get(
            "same_direction_development_assessment",
            pd.Series(False, index=multiplicity.index),
        ).astype(bool)
        & multiplicity.get("maximum_stock_share", pd.Series(1.0, index=multiplicity.index)).le(0.30)
        & multiplicity.get("maximum_month_share", pd.Series(1.0, index=multiplicity.index)).le(0.30)
    ]
    exact_passed = not exact_supported.empty
    support_passed = all(support.values())
    structure_passed = any(value["passed"] for value in broad_results) or exact_passed
    any_descriptive = bool(
        summaries["observed_minus_null_enrichment"].gt(0.0).any() if not summaries.empty else False
    )
    status = (
        "insufficient_support"
        if not support_passed
        else "supported"
        if structure_passed
        else "descriptive_only"
        if any_descriptive
        else "not_supported"
    )
    return status, {
        "support_checks": support,
        "support_passed": support_passed,
        "unique_registered_completions": int(len(event_ledger)),
        "complete_12_bar_history_rate": complete_rate,
        "stock_clock_null_matched_completions": complete_history_events,
        "stock_clock_null_unmatched_completions": int(len(event_ledger) - complete_history_events),
        "broad_precursor_results": broad_results,
        "exact_precursor_relationship_passed": exact_passed,
        "exact_supported_relationships": exact_supported.to_dict(orient="records"),
        "registered_precursor_structure_passed": support_passed and structure_passed,
    }


def build_b0_crossfit(
    opening: pd.DataFrame, t0_features: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    development = opening.loc[
        opening["year"].eq(2024)
        & opening.loc[:, list(t0_features)].notna().all(axis=1)
        & opening["registered_completion_within_12_bars"].notna()
    ].copy()
    predictions, manifest = expanding_logistic_crossfit(
        development,
        features=t0_features,
        target="registered_completion_within_12_bars",
        folds=4,
        warmup_fraction=0.2,
        weight_column="row_weight",
    )
    development["B0_oof_probability"] = predictions
    valid = development.loc[development["B0_oof_probability"].notna()].copy()
    threshold = candidate_threshold(valid["B0_oof_probability"])
    if not manifest.apply(
        lambda row: str(row.train_session_end) < str(row.prediction_session_start), axis=1
    ).all():
        raise ScreenBlocker("blocked_chronology_or_leakage_failure", "B0 expanding folds overlap")
    return valid, manifest, threshold


def attach_candidate_mechanisms(
    assessment: pd.DataFrame,
    path_ledger: pd.DataFrame,
    completions: pd.DataFrame,
) -> pd.DataFrame:
    paths = path_ledger.loc[
        path_ledger["year"].eq(2025),
        [
            "symbol",
            "session",
            "decision_ordinal",
            "family_id",
            "event_timestamp_utc",
            "event_available_timestamp_utc",
            "completion_bar_ordinal",
        ],
    ].copy()
    paths = paths.rename(columns={"completion_bar_ordinal": "hidden_completion_bar_ordinal"})
    paths = paths.drop_duplicates(["symbol", "session", "decision_ordinal"], keep="first")
    result = assessment.merge(
        paths,
        on=["symbol", "session", "decision_ordinal"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_hidden"),
    )
    result["hidden_family_class"] = result["family_id"].map(
        lambda value: (
            pool_hidden_family(str(value), FROZEN_HIDDEN_FAMILIES) if pd.notna(value) else None
        )
    )
    completion_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in completions.groupby(["symbol", "session"], sort=False)
    }
    first_ordinals: list[float] = []
    first_timestamps: list[pd.Timestamp | pd.NaT] = []
    first_ids: list[str | None] = []
    for row in result.itertuples(index=False):
        group = completion_groups.get((str(row.symbol), str(row.session)), pd.DataFrame())
        if group.empty:
            eligible = group
        else:
            eligible = group.loc[
                group["completion_bar_ordinal"].gt(int(row.repo_bar_start_ordinal))
                & group["completion_bar_ordinal"].le(int(row.repo_bar_start_ordinal) + 12)
            ].sort_values(["completion_bar_ordinal", "semantic_loop_id"], kind="mergesort")
        if eligible.empty:
            first_ordinals.append(math.nan)
            first_timestamps.append(pd.NaT)
            first_ids.append(None)
        else:
            first = eligible.iloc[0]
            first_ordinals.append(float(first["completion_bar_ordinal"]))
            first_timestamps.append(pd.Timestamp(first["completion_available_timestamp_utc"]))
            first_ids.append(str(first["semantic_loop_id"]))
    result["first_registered_completion_bar_ordinal"] = first_ordinals
    result["first_registered_completion_timestamp_utc"] = first_timestamps
    result["first_registered_semantic_loop_id"] = first_ids
    hidden_ordinal = pd.to_numeric(result["hidden_completion_bar_ordinal"], errors="coerce")
    registered_ordinal = pd.to_numeric(
        result["first_registered_completion_bar_ordinal"], errors="coerce"
    )
    result["hidden_event_before_registered_completion"] = (
        hidden_ordinal.notna() & registered_ordinal.notna() & hidden_ordinal.lt(registered_ordinal)
    )
    result["hidden_event_with_no_later_registered_completion"] = result[
        "actual_hidden_event_within_6_bars"
    ].astype(bool) & result["registered_completion_within_12_bars"].eq(0)
    result["registered_completion_with_no_hidden_event"] = result[
        "registered_completion_within_12_bars"
    ].eq(1) & ~result["actual_hidden_event_within_6_bars"].astype(bool)
    return result


def build_candidate_populations(
    opening: pd.DataFrame,
    development_archived: pd.DataFrame,
    assessment_archived: pd.DataFrame,
    completions: pd.DataFrame,
    t0_features: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    b0_valid, fold_manifest, threshold = build_b0_crossfit(opening, t0_features)
    development_u1 = development_archived.loc[
        :,
        [
            "symbol",
            "session",
            "decision_ordinal",
            "slate_id",
            "p_unregistered_within_6_bars",
            "registered_completion_within_12_bars",
            "actual_hidden_event_within_6_bars",
            "row_weight",
        ],
    ].rename(columns={"p_unregistered_within_6_bars": "development_cross_fitted_U1_probability"})
    development = b0_valid.merge(
        development_u1,
        on=["symbol", "session", "decision_ordinal"],
        how="inner",
        suffixes=("", "_archived"),
        validate="one_to_one",
    )
    development["U1_for_veto"] = development["development_cross_fitted_U1_probability"]
    development["high_candidate"] = development["B0_oof_probability"].ge(threshold)
    development_candidates = development.loc[development["high_candidate"]].copy()
    hidden_thresholds = freeze_hidden_risk_thresholds(development_candidates["U1_for_veto"])
    assigned_dev = assign_hidden_risk(
        development_candidates["U1_for_veto"].reset_index(drop=True), hidden_thresholds
    )
    development_candidates = development_candidates.reset_index(drop=True)
    development_candidates[["hidden_risk_group", "hidden_risk_quintile"]] = assigned_dev
    development_candidates["period"] = "development"
    development_candidates["candidate_B0_probability"] = development_candidates[
        "B0_oof_probability"
    ]

    assessment = assessment_archived.copy()
    assessment["U1_for_veto"] = assessment["U1_probability"]
    assessment["high_candidate"] = assessment["B0_probability"].ge(threshold)
    assessment_candidates = (
        assessment.loc[assessment["high_candidate"]].copy().reset_index(drop=True)
    )
    assigned_assessment = assign_hidden_risk(
        assessment_candidates["U1_for_veto"], hidden_thresholds
    )
    assessment_candidates[["hidden_risk_group", "hidden_risk_quintile"]] = assigned_assessment
    assessment_candidates["period"] = "assessment"
    assessment_candidates["candidate_B0_probability"] = assessment_candidates["B0_probability"]
    path_ledger = pd.read_parquet(OPENING_PRIMARY / "unregistered_path_ledger.parquet")
    assessment_candidates = attach_candidate_mechanisms(
        assessment_candidates, path_ledger, completions
    )

    threshold_manifest = {
        **SAFETY_FLAGS,
        "definition": "80th percentile of valid 2024 expanding OOF B0 probabilities",
        "threshold": threshold,
        "development_source_rows": int(len(b0_valid)),
        "development_candidate_rows_before_U1_intersection": int(
            b0_valid["B0_oof_probability"].ge(threshold).sum()
        ),
        "development_candidate_rows_with_valid_cross_fitted_U1": int(len(development_candidates)),
        "folds": 4,
        "fold_manifest": fold_manifest.to_dict(orient="records"),
        "assessment_outcomes_inspected_before_freeze": False,
    }
    hidden_manifest = {
        **SAFETY_FLAGS,
        **hidden_thresholds,
        "source": "2024 high-candidate population with valid cross-fitted U1",
        "source_rows": int(len(development_candidates)),
        "applied_unchanged_to_assessment": True,
    }
    return (
        development_candidates,
        assessment_candidates,
        threshold_manifest,
        hidden_manifest,
        development,
        b0_valid,
    )


def weighted_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return math.nan
    weights = frame["row_weight"].to_numpy(float)
    values = frame[column].astype(float).to_numpy()
    return float(np.sum(weights * values) / np.sum(weights))


def coefficient_standard_errors(model: Any, frame: pd.DataFrame) -> list[float]:
    matrix = frame.loc[:, list(model.features)].to_numpy(float)
    transformed = model.scaler.transform(matrix)
    design = np.column_stack([np.ones(len(frame)), transformed])
    probabilities = model.predict_probability(frame)
    weights = frame["row_weight"].to_numpy(float) * probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    penalty = np.zeros(information.shape[0])
    penalty[1:] = 1.0 / 0.25
    information += np.diag(penalty)
    covariance = np.linalg.pinv(information, hermitian=True)
    return np.sqrt(np.maximum(np.diag(covariance), 0.0)).astype(float).tolist()


def fit_veto_models(
    development: pd.DataFrame, assessment: pd.DataFrame
) -> tuple[Any, Any, pd.DataFrame, dict[str, Any]]:
    development = development.copy().reset_index(drop=True)
    assessment = assessment.copy().reset_index(drop=True)
    dev_v1 = veto_feature_frame(
        development,
        include_hidden_risk=True,
        b0_column="B0_oof_probability",
        u1_column="U1_for_veto",
    )
    assess_v1 = veto_feature_frame(
        assessment,
        include_hidden_risk=True,
        b0_column="B0_probability",
        u1_column="U1_for_veto",
    )
    for column in V1_FEATURES:
        development[column] = dev_v1[column].to_numpy()
        assessment[column] = assess_v1[column].to_numpy()
    try:
        v0 = fit_weighted_logistic(
            development,
            features=V0_FEATURES,
            target="registered_completion_within_12_bars",
        )
        v1 = fit_weighted_logistic(
            development,
            features=V1_FEATURES,
            target="registered_completion_within_12_bars",
        )
    except ValueError as error:
        raise ScreenBlocker("blocked_model_convergence_failure", str(error)) from error
    assessment["V0_probability"] = v0.predict_probability(assessment)
    assessment["V1_probability"] = v1.predict_probability(assessment)
    v0_spec = v0.as_dict()
    v1_spec = v1.as_dict()
    v0_spec["standard_error_intercept_then_coefficients"] = coefficient_standard_errors(
        v0, development
    )
    v1_spec["standard_error_intercept_then_coefficients"] = coefficient_standard_errors(
        v1, development
    )
    specifications = {**SAFETY_FLAGS, "V0": v0_spec, "V1": v1_spec}
    return v0, v1, assessment, specifications


def model_metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for model in ("V0", "V1"):
        metric = binary_model_metrics(
            predictions["registered_completion_within_12_bars"],
            predictions[f"{model}_probability"],
            predictions["row_weight"],
        )
        rows.append(
            {
                "model": model,
                **metric,
                "sessions": int(predictions["session"].nunique()),
                "stocks": int(predictions["symbol"].nunique()),
            }
        )
        for month, group in predictions.groupby("year_month", sort=True):
            monthly_rows.append(
                {
                    "year_month": month,
                    "model": model,
                    **binary_model_metrics(
                        group["registered_completion_within_12_bars"],
                        group[f"{model}_probability"],
                        group["row_weight"],
                    ),
                }
            )
        for checkpoint, group in predictions.groupby("decision_ordinal", sort=True):
            checkpoint_rows.append(
                {
                    "decision_ordinal": int(checkpoint),
                    "model": model,
                    **binary_model_metrics(
                        group["registered_completion_within_12_bars"],
                        group[f"{model}_probability"],
                        group["row_weight"],
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(monthly_rows), pd.DataFrame(checkpoint_rows)


def candidate_group_tables(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_group(grouping: str, value: Any, group: pd.DataFrame) -> None:
        rows.append(
            {
                "grouping": grouping,
                "group": str(value),
                "rows": int(len(group)),
                "sessions": int(group["session"].nunique()),
                "stocks": int(group["symbol"].nunique()),
                "registered_completions": int(group["registered_completion_within_12_bars"].sum()),
                "actual_hidden_events": int(
                    group["actual_hidden_event_within_6_bars"].astype(bool).sum()
                ),
                "registered_completion_rate": weighted_rate(
                    group, "registered_completion_within_12_bars"
                ),
                "actual_hidden_event_rate": weighted_rate(
                    group, "actual_hidden_event_within_6_bars"
                ),
            }
        )

    append_group("pooled", "all", predictions)
    for column in (
        "decision_ordinal",
        "year_month",
        "symbol",
        "hidden_risk_group",
        "hidden_risk_quintile",
        "registered_completion_within_12_bars",
        "actual_hidden_event_within_6_bars",
    ):
        for value, group in predictions.groupby(column, sort=True):
            append_group(column, value, group)
    return pd.DataFrame(rows)


def realised_diversion_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    successful = predictions.loc[predictions["registered_completion_within_12_bars"].eq(1)]
    failed = predictions.loc[predictions["registered_completion_within_12_bars"].eq(0)]
    success_rate = weighted_rate(successful, "actual_hidden_event_within_6_bars")
    failed_rate = weighted_rate(failed, "actual_hidden_event_within_6_bars")
    rows.extend(
        [
            {
                "scope": "pooled",
                "metric": "hidden_event_rate_successful",
                "value": success_rate,
                "rows": int(len(successful)),
            },
            {
                "scope": "pooled",
                "metric": "hidden_event_rate_failed",
                "value": failed_rate,
                "rows": int(len(failed)),
            },
            {
                "scope": "pooled",
                "metric": "failed_minus_successful_hidden_event_rate",
                "value": failed_rate - success_rate,
                "rows": int(len(predictions)),
            },
        ]
    )
    for family in (*FROZEN_HIDDEN_FAMILIES, OTHER_HIDDEN_FAMILY):
        column = predictions["hidden_family_class"].astype(str).eq(family).astype(float)
        temporary = predictions.assign(_family_event=column)
        family_successful = temporary.loc[temporary["registered_completion_within_12_bars"].eq(1)]
        family_failed = temporary.loc[temporary["registered_completion_within_12_bars"].eq(0)]
        family_success_rate = weighted_rate(family_successful, "_family_event")
        family_failed_rate = weighted_rate(family_failed, "_family_event")
        rows.extend(
            [
                {
                    "scope": family,
                    "metric": "hidden_family_rate_successful",
                    "value": family_success_rate,
                    "rows": int(family_successful["_family_event"].sum()),
                },
                {
                    "scope": family,
                    "metric": "hidden_family_rate_failed",
                    "value": family_failed_rate,
                    "rows": int(family_failed["_family_event"].sum()),
                },
                {
                    "scope": family,
                    "metric": "hidden_family_rate_failed_minus_successful",
                    "value": family_failed_rate - family_success_rate,
                    "rows": int(column.sum()),
                },
            ]
        )
    for column in (
        "hidden_event_before_registered_completion",
        "hidden_event_with_no_later_registered_completion",
        "registered_completion_with_no_hidden_event",
    ):
        rows.append(
            {
                "scope": "pooled",
                "metric": column,
                "value": weighted_rate(predictions, column),
                "rows": int(predictions[column].astype(bool).sum()),
            }
        )

    def stability(column: str) -> pd.DataFrame:
        output = []
        for value, group in predictions.groupby(column, sort=True):
            failed_group = group.loc[group["registered_completion_within_12_bars"].eq(0)]
            success_group = group.loc[group["registered_completion_within_12_bars"].eq(1)]
            high = group.loc[group["hidden_risk_group"].eq("high")]
            low = group.loc[group["hidden_risk_group"].eq("low")]
            output.append(
                {
                    column: value,
                    "rows": int(len(group)),
                    "high_minus_low_completion_rate": weighted_rate(
                        high, "registered_completion_within_12_bars"
                    )
                    - weighted_rate(low, "registered_completion_within_12_bars"),
                    "failed_minus_successful_hidden_event_rate": weighted_rate(
                        failed_group, "actual_hidden_event_within_6_bars"
                    )
                    - weighted_rate(success_group, "actual_hidden_event_within_6_bars"),
                }
            )
        return pd.DataFrame(output)

    return pd.DataFrame(rows), stability("year_month"), stability("decision_ordinal")


def metric_increments(predictions: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    model_metrics = {}
    for model in ("V0", "V1"):
        model_metrics[model] = binary_model_metrics(
            predictions["registered_completion_within_12_bars"],
            predictions[f"{model}_probability"],
            predictions["row_weight"],
        )
    metrics["log_loss_improvement"] = float(
        model_metrics["V0"]["log_loss"] - model_metrics["V1"]["log_loss"]
    )
    metrics["brier_improvement"] = float(
        model_metrics["V0"]["brier_score"] - model_metrics["V1"]["brier_score"]
    )
    metrics["auc_improvement"] = float(model_metrics["V1"]["auc"] - model_metrics["V0"]["auc"])
    return metrics


def bootstrap_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    point = metric_increments(predictions)
    high = predictions.loc[predictions["hidden_risk_group"].eq("high")]
    low = predictions.loc[predictions["hidden_risk_group"].eq("low")]
    failed = predictions.loc[predictions["registered_completion_within_12_bars"].eq(0)]
    successful = predictions.loc[predictions["registered_completion_within_12_bars"].eq(1)]
    point["high_minus_low_completion_rate"] = weighted_rate(
        high, "registered_completion_within_12_bars"
    ) - weighted_rate(low, "registered_completion_within_12_bars")
    point["failed_minus_successful_hidden_event_rate"] = weighted_rate(
        failed, "actual_hidden_event_within_6_bars"
    ) - weighted_rate(successful, "actual_hidden_event_within_6_bars")
    draw_rows: list[dict[str, Any]] = []
    draw_values: dict[str, list[float]] = defaultdict(list)
    for draw, indices in enumerate(
        session_block_bootstrap_indices(predictions, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    ):
        sampled = predictions.iloc[indices].reset_index(drop=True)
        values = metric_increments(sampled)
        sampled_high = sampled.loc[sampled["hidden_risk_group"].eq("high")]
        sampled_low = sampled.loc[sampled["hidden_risk_group"].eq("low")]
        sampled_failed = sampled.loc[sampled["registered_completion_within_12_bars"].eq(0)]
        sampled_success = sampled.loc[sampled["registered_completion_within_12_bars"].eq(1)]
        values["high_minus_low_completion_rate"] = weighted_rate(
            sampled_high, "registered_completion_within_12_bars"
        ) - weighted_rate(sampled_low, "registered_completion_within_12_bars")
        values["failed_minus_successful_hidden_event_rate"] = weighted_rate(
            sampled_failed, "actual_hidden_event_within_6_bars"
        ) - weighted_rate(sampled_success, "actual_hidden_event_within_6_bars")
        for metric, value in values.items():
            draw_values[metric].append(value)
            draw_rows.append(
                {
                    "record_type": "draw",
                    "draw": draw,
                    "metric": metric,
                    "value": value,
                    "point_estimate": point[metric],
                    "interval_level": math.nan,
                    "lower": math.nan,
                    "upper": math.nan,
                }
            )
    for metric, values in draw_values.items():
        for level in (0.80, 0.90, 0.95):
            alpha = (1.0 - level) / 2.0
            draw_rows.append(
                {
                    "record_type": "interval",
                    "draw": -1,
                    "metric": metric,
                    "value": math.nan,
                    "point_estimate": point[metric],
                    "interval_level": level,
                    "lower": float(np.quantile(values, alpha)),
                    "upper": float(np.quantile(values, 1.0 - alpha)),
                }
            )
    return pd.DataFrame(draw_rows)


def veto_null_metrics(
    development_all: pd.DataFrame,
    assessment_all: pd.DataFrame,
    threshold: float,
    v0_predictions: pd.Series,
    real_increment: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    exceeded = {metric: 0 for metric in real_increment}
    for draw in range(VETO_NULL_DRAWS):
        dev = permute_hidden_probability_within_slates(
            development_all, seed=VETO_NULL_SEED + draw, feature="U1_for_veto"
        )
        assess = permute_hidden_probability_within_slates(
            assessment_all, seed=VETO_NULL_SEED + draw, feature="U1_for_veto"
        )
        dev_candidate = (
            dev.loc[dev["B0_oof_probability"].ge(threshold)].copy().reset_index(drop=True)
        )
        assess_candidate = (
            assess.loc[assess["B0_probability"].ge(threshold)].copy().reset_index(drop=True)
        )
        dev_features = veto_feature_frame(
            dev_candidate,
            include_hidden_risk=True,
            b0_column="B0_oof_probability",
            u1_column="U1_for_veto",
        )
        assess_features = veto_feature_frame(
            assess_candidate,
            include_hidden_risk=True,
            b0_column="B0_probability",
            u1_column="U1_for_veto",
        )
        for column in V1_FEATURES:
            dev_candidate[column] = dev_features[column].to_numpy()
            assess_candidate[column] = assess_features[column].to_numpy()
        model = fit_weighted_logistic(
            dev_candidate,
            features=V1_FEATURES,
            target="registered_completion_within_12_bars",
        )
        assess_candidate["V0_probability"] = v0_predictions.to_numpy()
        assess_candidate["V1_probability"] = model.predict_probability(assess_candidate)
        increments = metric_increments(assess_candidate)
        fingerprint = hashlib.sha256(
            np.asarray(assess_candidate["U1_for_veto"], dtype=np.float64).tobytes()
        ).hexdigest()
        development_fingerprint = hashlib.sha256(
            np.asarray(dev_candidate["U1_for_veto"], dtype=np.float64).tobytes()
        ).hexdigest()
        for metric, value in increments.items():
            if real_increment[metric] > value:
                exceeded[metric] += 1
            rows.append(
                {
                    "draw": draw,
                    "metric": metric,
                    "null_increment": value,
                    "real_increment": real_increment[metric],
                    "real_exceeds_null": real_increment[metric] > value,
                    "assessment_permutation_sha256": fingerprint,
                    "development_permutation_sha256": development_fingerprint,
                    "V1_model_json": json.dumps(model.as_dict(), sort_keys=True),
                }
            )
    return pd.DataFrame(rows), exceeded


def concentration_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension, column in (
        ("stock", "symbol"),
        ("month", "year_month"),
        ("checkpoint", "decision_ordinal"),
        ("target", "registered_completion_within_12_bars"),
    ):
        counts = predictions[column].value_counts(normalize=True, sort=False)
        for value, share in counts.items():
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "rows": int(predictions[column].eq(value).sum()),
                    "share": float(share),
                }
            )
    return pd.DataFrame(rows)


def interval_value(bootstrap: pd.DataFrame, metric: str, level: float, field: str) -> float:
    row = bootstrap.loc[
        bootstrap["record_type"].eq("interval")
        & bootstrap["metric"].eq(metric)
        & bootstrap["interval_level"].eq(level)
    ]
    return float(row.iloc[0][field])


def evaluate_veto_and_diversion(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null_exceeded: dict[str, int],
    hidden_coefficient: float,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    high = predictions.loc[predictions["hidden_risk_group"].eq("high")]
    low = predictions.loc[predictions["hidden_risk_group"].eq("low")]
    high_low = weighted_rate(high, "registered_completion_within_12_bars") - weighted_rate(
        low, "registered_completion_within_12_bars"
    )
    failed = predictions.loc[predictions["registered_completion_within_12_bars"].eq(0)]
    success = predictions.loc[predictions["registered_completion_within_12_bars"].eq(1)]
    failed_success = weighted_rate(failed, "actual_hidden_event_within_6_bars") - weighted_rate(
        success, "actual_hidden_event_within_6_bars"
    )
    maximum_stock_share = float(predictions["symbol"].value_counts(normalize=True).max())
    maximum_class_share = float(
        predictions["registered_completion_within_12_bars"].value_counts(normalize=True).max()
    )
    support = {
        "rows_at_least_1000": len(predictions) >= 1000,
        "sessions_at_least_120": predictions["session"].nunique() >= 120,
        "stocks_at_least_15": predictions["symbol"].nunique() >= 15,
        "registered_completions_at_least_150": int(
            predictions["registered_completion_within_12_bars"].sum()
        )
        >= 150,
        "actual_hidden_events_at_least_150": int(
            predictions["actual_hidden_event_within_6_bars"].astype(bool).sum()
        )
        >= 150,
        "high_hidden_risk_rows_at_least_200": len(high) >= 200,
        "low_hidden_risk_rows_at_least_200": len(low) >= 200,
        "no_stock_above_10pct": maximum_stock_share <= 0.10,
        "no_target_class_above_85pct": maximum_class_share <= 0.85,
    }
    metric_lookup = metrics.set_index("model")
    increments = {
        "log_loss_improvement": float(
            metric_lookup.loc["V0", "log_loss"] - metric_lookup.loc["V1", "log_loss"]
        ),
        "brier_improvement": float(
            metric_lookup.loc["V0", "brier_score"] - metric_lookup.loc["V1", "brier_score"]
        ),
        "auc_improvement": float(metric_lookup.loc["V1", "auc"] - metric_lookup.loc["V0", "auc"]),
    }
    checkpoint_opposite = bool(checkpoint["high_minus_low_completion_rate"].gt(0.02).any())
    predictive_checks = {
        "V1_improves_log_loss": increments["log_loss_improvement"] > 0.0,
        "V1_improves_brier": increments["brier_improvement"] > 0.0,
        "V1_auc_not_reduced_more_than_0_002": increments["auc_improvement"] >= -0.002,
        "V1_hidden_coefficient_negative": hidden_coefficient < 0.0,
        "log_loss_80pct_lower_nonnegative": interval_value(
            bootstrap, "log_loss_improvement", 0.80, "lower"
        )
        >= 0.0,
        "brier_80pct_lower_nonnegative": interval_value(
            bootstrap, "brier_improvement", 0.80, "lower"
        )
        >= 0.0,
        "high_hidden_completion_rate_lower": high_low < 0.0,
        "high_minus_low_80pct_upper_nonpositive": interval_value(
            bootstrap, "high_minus_low_completion_rate", 0.80, "upper"
        )
        <= 0.0,
        "negative_in_five_months": int(monthly["high_minus_low_completion_rate"].lt(0.0).sum())
        >= 5,
        "neither_checkpoint_materially_opposite": not checkpoint_opposite,
        "proper_score_increment_exceeds_four_of_five_nulls": null_exceeded["log_loss_improvement"]
        >= 4
        and null_exceeded["brier_improvement"] >= 4,
        "concentration_gates_pass": support["no_stock_above_10pct"]
        and support["no_target_class_above_85pct"],
    }
    realised_checks = {
        "hidden_events_more_common_among_failures": failed_success > 0.0,
        "failed_minus_successful_80pct_lower_nonnegative": interval_value(
            bootstrap, "failed_minus_successful_hidden_event_rate", 0.80, "lower"
        )
        >= 0.0,
        "same_positive_sign_both_checkpoints": len(checkpoint) == 2
        and bool(checkpoint["failed_minus_successful_hidden_event_rate"].gt(0.0).all()),
        "positive_in_five_months": int(
            monthly["failed_minus_successful_hidden_event_rate"].gt(0.0).sum()
        )
        >= 5,
    }
    support_passed = all(support.values())
    predictive_passed = support_passed and all(predictive_checks.values())
    realised_passed = support_passed and all(realised_checks.values())
    predictive_status = (
        "insufficient_support"
        if not support_passed
        else "supported"
        if predictive_passed
        else "descriptive_only"
        if hidden_coefficient < 0.0 or high_low < 0.0
        else "not_supported"
    )
    realised_status = (
        "insufficient_support"
        if not support_passed
        else "supported"
        if realised_passed
        else "descriptive_only"
        if failed_success > 0.0
        else "not_supported"
    )
    return (
        predictive_status,
        realised_status,
        {
            "support_checks": support,
            "support_passed": support_passed,
            "gate_checks": predictive_checks,
            "predictive_hidden_veto_passed": predictive_passed,
            "increments": increments,
            "high_minus_low_completion_rate": high_low,
            "hidden_coefficient": hidden_coefficient,
            "null_draws_exceeded": null_exceeded,
            "maximum_stock_share": maximum_stock_share,
            "maximum_target_class_share": maximum_class_share,
        },
        {
            "support_checks": support,
            "support_passed": support_passed,
            "gate_checks": realised_checks,
            "realised_diversion_passed": realised_passed,
            "failed_minus_successful_hidden_event_rate": failed_success,
        },
    )


def veto_removal_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    high = predictions["hidden_risk_group"].eq("high")
    success = predictions["registered_completion_within_12_bars"].eq(1)
    failed = ~success
    weights = predictions["row_weight"].to_numpy(float)

    def weighted_fraction(mask: pd.Series, denominator: pd.Series) -> float:
        denominator_weight = float(weights[denominator.to_numpy()].sum())
        return float(weights[(mask & denominator).to_numpy()].sum() / denominator_weight)

    success_removed = weighted_fraction(high, success)
    failed_removed = weighted_fraction(high, failed)
    return {
        "candidate_rows": int(len(predictions)),
        "vetoed_rows": int(high.sum()),
        "candidate_retention": float(
            1.0 - weighted_fraction(high, pd.Series(True, index=high.index))
        ),
        "successful_completions_removed": success_removed,
        "failed_candidates_removed": failed_removed,
        "failure_removal_to_success_removal_ratio": (
            failed_removed / success_removed if success_removed > 0.0 else math.inf
        ),
    }


def create_plots(
    precursor_null: pd.DataFrame,
    group_metrics: pd.DataFrame,
    output: Path,
) -> list[str]:
    paths: list[str] = []
    summary = precursor_null.loc[
        precursor_null["record_type"].eq("summary")
        & precursor_null["precursor_type"].isin(BROAD_GATE_PRECURSORS)
    ].copy()
    if not summary.empty:
        labels = [
            f"{row.precursor_type}\n{int(row.lookback_bars)}b"
            for row in summary.itertuples(index=False)
        ]
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.bar(np.arange(len(summary)), summary["observed_minus_null_enrichment"])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(len(summary)), labels, rotation=70, ha="right", fontsize=7)
        axis.set_ylabel("Observed minus matched-null prevalence")
        axis.set_title("Registered-loop precursor enrichment")
        figure.tight_layout()
        path = output / "precursor_enrichment_matched_null.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    quintiles = group_metrics.loc[group_metrics["grouping"].eq("hidden_risk_quintile")].copy()
    if not quintiles.empty:
        quintiles["group_numeric"] = quintiles["group"].astype(int)
        quintiles = quintiles.sort_values("group_numeric")
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(
            quintiles["group_numeric"],
            quintiles["registered_completion_rate"],
            marker="o",
        )
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.set_xlabel("Frozen development U1 quintile")
        axis.set_ylabel("Registered completion rate")
        axis.set_title("High-B0 candidates by hidden-event risk")
        figure.tight_layout()
        path = output / "registered_completion_by_hidden_risk_quintile.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    return paths


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small deterministic Markdown table without optional dependencies."""

    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        output = f"{value:.9g}" if isinstance(value, float) else str(value)
        return output.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def build_report(
    decision: dict[str, Any],
    precursor_census: pd.DataFrame,
    nearest: pd.DataFrame,
    precursor_null: pd.DataFrame,
    group_metrics: pd.DataFrame,
    veto_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    realised: pd.DataFrame,
) -> str:
    precursor_pivot = precursor_census.pivot(
        index="precursor_type", columns="lookback_bars", values="observed_prevalence"
    ).reset_index()
    null_summary = precursor_null.loc[precursor_null["record_type"].eq("summary")]
    quartiles = group_metrics.loc[
        group_metrics["grouping"].eq("hidden_risk_group"),
        ["group", "rows", "registered_completion_rate"],
    ]
    quintiles = group_metrics.loc[
        group_metrics["grouping"].eq("hidden_risk_quintile"),
        ["group", "rows", "registered_completion_rate"],
    ]
    intervals = bootstrap.loc[
        bootstrap["record_type"].eq("interval") & bootstrap["interval_level"].eq(0.80)
    ]
    realised_pooled = realised.loc[realised["scope"].eq("pooled")]
    realised_families = realised.loc[~realised["scope"].eq("pooled")]
    unique_completions = decision["precursor_gate"]["unique_registered_completions"]
    complete_history_rate = decision["precursor_gate"]["complete_12_bar_history_rate"]
    unmatched_completions = decision["precursor_gate"]["stock_clock_null_unmatched_completions"]
    exact_supported = pd.DataFrame(decision["precursor_gate"]["exact_supported_relationships"])
    if exact_supported.empty:
        exact_markdown = "No exact precursor relationship passed every support and stability gate."
    else:
        exact_columns = [
            "lookback_bars",
            "precursor_kind",
            "precursor_identity",
            "target_semantic_loop_id",
            "development_enrichment",
            "assessment_enrichment",
            "q_value",
        ]
        exact_markdown = markdown_table(exact_supported.loc[:, exact_columns])
    null_columns = [
        "lookback_bars",
        "precursor_type",
        "observed_prevalence",
        "mean_null_prevalence",
        "observed_minus_null_enrichment",
        "null_percentile",
    ]
    null_markdown = markdown_table(null_summary.loc[:, null_columns])
    return f"""# Registered-Loop Precursors and Hidden-Diversion Veto Quick Screen V0

Decision: `{decision["primary_decision"]}`

- Precursor status: `{decision["precursor_status"]}`
- Predictive-veto status: `{decision["predictive_veto_status"]}`
- Realised-diversion status: `{decision["realised_diversion_status"]}`
- Protected rows materialised: `{decision["protected_rows_materialised"]}`
- Scope: retrospective, research-only, structural feasibility screen.
- Economic and directional outcomes remained closed.

## Registered-loop precursor census

Unique assessment registered completions: {unique_completions}.
Complete 12-bar history: {complete_history_rate:.9f}.
The stock-and-clock null could not match {unmatched_completions} early completions while also
requiring twelve prior bars; null inference is restricted to the complete-history subset and
the preregistered 90% complete-history support gate fails closed.

{markdown_table(precursor_pivot)}

Nearest precursor distribution:

{markdown_table(nearest)}

Matched-null summaries:

{null_markdown}

Exact precursor relationships passing all gates:

{exact_markdown}

## Frozen high-B0 candidate veto

Candidate threshold: `{decision["candidate_threshold"]:.15g}`.

Hidden-risk groups:

{markdown_table(quartiles)}

Hidden-risk quintiles:

{markdown_table(quintiles)}

Model metrics:

{markdown_table(veto_metrics)}

80% fixed-prediction session-bootstrap intervals:

{markdown_table(intervals[["metric", "point_estimate", "lower", "upper"]])}

Realised structural diversion diagnostics:

{markdown_table(realised_pooled)}

Realised hidden-family rates by successful and failed candidate status:

{markdown_table(realised_families)}

## Boundary

These findings are not prospective validation, economic-edge evidence, directional edge,
trading utility, a deployable strategy, or permission to trade.
"""


def determinism_check(output: Path) -> dict[str, Any]:
    """Reload frozen artifacts and independently reproduce deterministic outputs."""

    keys = list(OPENING_KEYS)
    threshold_manifest = read_json(output / "candidate_threshold_manifest.json")
    hidden_manifest = read_json(output / "hidden_risk_thresholds.json")
    model_specs = read_json(output / "veto_model_coefficients.json")
    decision = read_json(output / "decision.json")
    oof = pd.read_parquet(output / "b0_development_oof_predictions.parquet")
    candidates = pd.read_parquet(output / "candidate_population.parquet")
    predictions = pd.read_parquet(output / "veto_assessment_predictions.parquet")
    feature_ledger = pd.read_parquet(output / "precursor_feature_ledger.parquet")
    precursor_census = pd.read_csv(output / "precursor_census.csv")
    archived_metrics = pd.read_csv(output / "veto_metrics.csv")
    development_archived = pd.read_parquet(BRIDGE_PRIMARY / "bridge_development_panel.parquet")
    assessment_archived = pd.read_parquet(BRIDGE_PRIMARY / "bridge_assessment_predictions.parquet")

    recalculated_threshold = candidate_threshold(oof["B0_oof_probability"])
    threshold_difference = abs(recalculated_threshold - float(threshold_manifest["threshold"]))
    expected_development = oof.loc[oof["B0_oof_probability"].ge(recalculated_threshold)].merge(
        development_archived.loc[:, [*keys, "p_unregistered_within_6_bars"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    expected_assessment = assessment_archived.loc[
        assessment_archived["B0_probability"].ge(recalculated_threshold)
    ]

    def membership_mismatches(expected: pd.DataFrame, archived: pd.DataFrame) -> int:
        comparison = (
            expected.loc[:, keys]
            .drop_duplicates()
            .merge(
                archived.loc[:, keys].drop_duplicates(),
                on=keys,
                how="outer",
                indicator=True,
                validate="one_to_one",
            )
        )
        return int((~comparison["_merge"].eq("both")).sum())

    development_candidates = candidates.loc[candidates["period"].eq("development")].copy()
    assessment_candidates = candidates.loc[candidates["period"].eq("assessment")].copy()
    candidate_membership_mismatches = membership_mismatches(
        expected_development, development_candidates
    ) + membership_mismatches(expected_assessment, assessment_candidates)

    recalculated_hidden = freeze_hidden_risk_thresholds(
        expected_development["p_unregistered_within_6_bars"]
    )
    hidden_differences = [
        abs(float(recalculated_hidden["low_maximum"]) - float(hidden_manifest["low_maximum"])),
        abs(float(recalculated_hidden["high_minimum"]) - float(hidden_manifest["high_minimum"])),
        *np.abs(
            np.asarray(recalculated_hidden["quintile_boundaries"], dtype=float)
            - np.asarray(hidden_manifest["quintile_boundaries"], dtype=float)
        ).tolist(),
    ]
    maximum_threshold_difference = max([threshold_difference, *hidden_differences])
    threshold_mismatches = int(
        sum(value > 1e-12 for value in [threshold_difference, *hidden_differences])
    )

    refit_development = development_candidates.reset_index(drop=True)
    refit_assessment = predictions.copy().reset_index(drop=True)
    development_features = veto_feature_frame(
        refit_development,
        include_hidden_risk=True,
        b0_column="candidate_B0_probability",
        u1_column="U1_for_veto",
    )
    assessment_features = veto_feature_frame(
        refit_assessment,
        include_hidden_risk=True,
        b0_column="candidate_B0_probability",
        u1_column="U1_for_veto",
    )
    for column in V1_FEATURES:
        refit_development[column] = development_features[column].to_numpy()
        refit_assessment[column] = assessment_features[column].to_numpy()
    refit_v0 = fit_weighted_logistic(
        refit_development,
        features=V0_FEATURES,
        target="registered_completion_within_12_bars",
    )
    refit_v1 = fit_weighted_logistic(
        refit_development,
        features=V1_FEATURES,
        target="registered_completion_within_12_bars",
    )
    refit_probabilities = {
        "V0": refit_v0.predict_probability(refit_assessment),
        "V1": refit_v1.predict_probability(refit_assessment),
    }
    probability_difference = max(
        float(
            np.max(
                np.abs(
                    refit_probabilities[model] - predictions[f"{model}_probability"].to_numpy(float)
                )
            )
        )
        for model in ("V0", "V1")
    )
    coefficient_differences: list[float] = []
    for model_name, fitted in (("V0", refit_v0), ("V1", refit_v1)):
        fitted_spec = fitted.as_dict()
        archived_spec = model_specs[model_name]
        coefficient_differences.extend(
            [
                abs(float(fitted_spec["intercept"]) - float(archived_spec["intercept"])),
                *np.abs(
                    np.asarray(fitted_spec["coefficient"], dtype=float)
                    - np.asarray(archived_spec["coefficient"], dtype=float)
                ).tolist(),
                *np.abs(
                    np.asarray(fitted_spec["scaler_mean"], dtype=float)
                    - np.asarray(archived_spec["scaler_mean"], dtype=float)
                ).tolist(),
                *np.abs(
                    np.asarray(fitted_spec["scaler_scale"], dtype=float)
                    - np.asarray(archived_spec["scaler_scale"], dtype=float)
                ).tolist(),
            ]
        )
    coefficient_difference = max(coefficient_differences, default=0.0)

    def regenerate_nearest_label(row: Any) -> str:
        completed = json.loads(str(row.registered_precursors_json))
        if completed:
            return "NEAREST_COMPLETED_LOOP_EVENT"
        prefixes = json.loads(str(row.prefix_precursors_json))
        if any(
            str(prefix["semantic_loop_id"]) == str(row.target_semantic_loop_id)
            for prefix in prefixes
        ):
            return "ACTIVE_MATCHING_PREFIX"
        if prefixes:
            return "OTHER_ACTIVE_PREFIX"
        transition_states = json.loads(str(row.transition_state_path_json))
        transition_count = sum(
            int(right["bar_ordinal"]) == int(left["bar_ordinal"]) + 1
            and int(right["bar_ordinal"]) >= int(row.window_start_bar_ordinal)
            and int(left["causal_hard_state"]) != int(right["causal_hard_state"])
            for left, right in zip(transition_states[:-1], transition_states[1:], strict=True)
        )
        if transition_count > 0:
            return "REGIME_TRANSITION"
        return "NO_IDENTIFIED_STRUCTURAL_PRECURSOR"

    regenerated_labels = feature_ledger.apply(regenerate_nearest_label, axis=1)
    precursor_label_mismatches = int(
        regenerated_labels.ne(feature_ledger["nearest_precursor_label"].astype(str)).sum()
    )
    observed = feature_ledger.loc[
        feature_ledger["record_type"].eq("observed") & feature_ledger["period"].eq("assessment")
    ]
    recalculated_rows = []
    for lookback in LOOKBACKS:
        group = observed.loc[observed["lookback_bars"].eq(lookback)]
        for precursor in BOOLEAN_PRECURSORS:
            recalculated_rows.append(
                {
                    "lookback_bars": lookback,
                    "precursor_type": precursor,
                    "observed_prevalence": float(group[precursor].astype(bool).mean()),
                }
            )
    recalculated = pd.DataFrame(recalculated_rows).merge(
        precursor_census,
        on=["lookback_bars", "precursor_type"],
        suffixes=("_recalculated", "_archived"),
        validate="one_to_one",
    )
    census_difference = float(
        np.max(
            np.abs(
                recalculated["observed_prevalence_recalculated"].to_numpy(float)
                - recalculated["observed_prevalence_archived"].to_numpy(float)
            )
        )
    )

    metric_differences: list[float] = []
    pooled_metric_mismatches = 0
    metric_fields = (
        "log_loss",
        "brier_score",
        "auc",
        "average_precision",
        "calibration_intercept",
        "calibration_slope",
        "expected_calibration_error",
        "base_rate",
        "mean_probability_realised_class",
    )
    archived_metric_lookup = archived_metrics.set_index("model")
    for model in ("V0", "V1"):
        recalculated_metric = binary_model_metrics(
            refit_assessment["registered_completion_within_12_bars"],
            refit_probabilities[model],
            refit_assessment["row_weight"],
        )
        for field in metric_fields:
            difference = abs(
                float(recalculated_metric[field]) - float(archived_metric_lookup.loc[model, field])
            )
            metric_differences.append(difference)
            pooled_metric_mismatches += int(difference > 1e-12)
    maximum_metric_difference = max(metric_differences, default=0.0)

    reconstructed_decision = choose_primary_decision(
        precursor_status=str(decision["precursor_status"]),
        predictive_veto_status=str(decision["predictive_veto_status"]),
        realised_diversion_status=str(decision["realised_diversion_status"]),
    )
    final_decision_match = reconstructed_decision == str(decision["primary_decision"])
    passed = (
        probability_difference <= 1e-12
        and coefficient_difference <= 1e-12
        and census_difference <= 1e-12
        and precursor_label_mismatches == 0
        and candidate_membership_mismatches == 0
        and threshold_mismatches == 0
        and pooled_metric_mismatches == 0
        and final_decision_match
    )
    return {
        **SAFETY_FLAGS,
        "passed": passed,
        "maximum_probability_difference": probability_difference,
        "maximum_coefficient_difference": coefficient_difference,
        "maximum_precursor_census_difference": census_difference,
        "maximum_threshold_difference": maximum_threshold_difference,
        "maximum_pooled_metric_difference": maximum_metric_difference,
        "precursor_label_mismatches": precursor_label_mismatches,
        "candidate_membership_mismatches": candidate_membership_mismatches,
        "threshold_mismatches": threshold_mismatches,
        "pooled_metric_mismatches": pooled_metric_mismatches,
        "final_decision_match": final_decision_match,
        "reloaded_frozen_panels": True,
        "bootstrap_repeated": False,
        "precursor_null_repeated": False,
        "hidden_probability_null_repeated": False,
        "refitted_models": 2,
    }


def execute_screen(output: Path, *, provider_root: Path) -> dict[str, Any]:
    contract = load_contract()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    print("reconstructing exact frozen opening decision population", flush=True)
    (
        opening,
        development_archived,
        assessment_archived,
        completions,
        opening_reconstruction,
        t0_features,
    ) = reconstruct_opening_population()
    reject_protected_dates(opening)
    write_json(output / "opening_panel_reconstruction.json", opening_reconstruction)

    print("reconstructing bounded V2 state/prefix surface", flush=True)
    states, dictionary, v2_source, dictionary_manifest = load_v2_states(provider_root)
    assessment_events = build_completion_event_ledger(
        completions, opening, year=2025, period="assessment"
    )
    development_events = build_completion_event_ledger(
        completions, opening, year=2024, period="development"
    )
    write_parquet(output / "registered_completion_event_ledger.parquet", assessment_events)
    assessment_eligible = build_null_eligibility(states, completions, period="assessment")
    development_eligible = build_null_eligibility(states, completions, period="development")
    null_targets, null_manifest = build_null_targets(
        assessment_events,
        development_events,
        assessment_eligible,
        development_eligible,
    )

    print("building observed and 25-draw matched precursor census", flush=True)
    feature_ledger, trace_manifest = build_precursor_feature_ledger(
        states,
        dictionary,
        completions,
        development_events,
        assessment_events,
        null_targets,
    )
    write_parquet(output / "precursor_feature_ledger.parquet", feature_ledger)
    precursor_census, nearest, precursor_monthly, precursor_checkpoint, precursor_null = (
        precursor_census_tables(feature_ledger)
    )
    exact_counts, exact_multiplicity = exact_transition_tables(feature_ledger)
    write_csv(output / "precursor_census.csv", precursor_census)
    write_csv(output / "nearest_precursor_census.csv", nearest)
    write_csv(output / "precursor_monthly_metrics.csv", precursor_monthly)
    write_csv(output / "precursor_checkpoint_metrics.csv", precursor_checkpoint)
    write_csv(output / "precursor_null_metrics.csv", precursor_null)
    write_csv(output / "exact_precursor_transition_counts.csv", exact_counts)
    write_csv(output / "exact_precursor_multiplicity.csv", exact_multiplicity)
    precursor_status, precursor_gate = evaluate_precursor_gate(
        assessment_events,
        feature_ledger,
        precursor_null,
        precursor_monthly,
        precursor_checkpoint,
        exact_multiplicity,
    )

    print("cross-fitting B0 and freezing candidate/U1 thresholds", flush=True)
    (
        development_candidates,
        assessment_candidates,
        threshold_manifest,
        hidden_manifest,
        development_all,
        b0_development_oof,
    ) = build_candidate_populations(
        opening,
        development_archived,
        assessment_archived,
        completions,
        t0_features,
    )
    write_json(output / "candidate_threshold_manifest.json", threshold_manifest)
    write_json(output / "hidden_risk_thresholds.json", hidden_manifest)
    write_parquet(
        output / "b0_development_oof_predictions.parquet",
        b0_development_oof.loc[
            :,
            [
                "symbol",
                "session",
                "decision_ordinal",
                "slate_id",
                "row_weight",
                "registered_completion_within_12_bars",
                "B0_oof_probability",
            ],
        ],
    )
    threshold = float(threshold_manifest["threshold"])

    print("fitting the two frozen candidate-only veto models", flush=True)
    _, _, predictions, coefficient_specs = fit_veto_models(
        development_candidates, assessment_candidates
    )
    model_configurations = {
        **SAFETY_FLAGS,
        "V0": {"features": list(V0_FEATURES)},
        "V1": {"features": list(V1_FEATURES)},
        "actual_hidden_event_excluded": True,
        "model": contract["candidate_veto"]["model"],
        "primary_model_fits": 2,
        "development_B0_crossfit_fits": 4,
        "determinism_refits": 2,
        "hidden_probability_null_refits": 5,
    }
    write_json(output / "veto_model_configurations.json", model_configurations)
    write_json(output / "veto_model_coefficients.json", coefficient_specs)
    metrics, model_monthly, model_checkpoint = model_metric_tables(predictions)
    group_metrics = candidate_group_tables(predictions)
    realised, stability_monthly, stability_checkpoint = realised_diversion_tables(predictions)
    veto_monthly = model_monthly.merge(
        stability_monthly, on="year_month", how="left", validate="many_to_one"
    )
    veto_checkpoint = model_checkpoint.merge(
        stability_checkpoint, on="decision_ordinal", how="left", validate="many_to_one"
    )
    bootstrap = bootstrap_metrics(predictions)
    real_increment = metric_increments(predictions)

    assessment_all = assessment_archived.copy()
    assessment_all["U1_for_veto"] = assessment_all["U1_probability"]
    development_all = development_all.copy()
    print("running exactly five within-slate hidden-probability null refits", flush=True)
    null_metrics, null_exceeded = veto_null_metrics(
        development_all,
        assessment_all,
        threshold,
        predictions["V0_probability"],
        real_increment,
    )
    hidden_index = coefficient_specs["V1"]["feature_names"].index("logit_U1_probability")
    hidden_coefficient = float(coefficient_specs["V1"]["coefficient"][hidden_index])
    hidden_standard_error = float(
        coefficient_specs["V1"]["standard_error_intercept_then_coefficients"][hidden_index + 1]
    )
    predictive_status, realised_status, predictive_gate, realised_gate = (
        evaluate_veto_and_diversion(
            predictions,
            metrics,
            stability_monthly,
            stability_checkpoint,
            bootstrap,
            null_exceeded,
            hidden_coefficient,
        )
    )
    removal = veto_removal_metrics(predictions)
    coefficient_specs["V1_hidden_risk_summary"] = {
        "standardised_coefficient": hidden_coefficient,
        "descriptive_standard_error": hidden_standard_error,
        "negative": hidden_coefficient < 0.0,
        "assessment_months_with_negative_high_minus_low_rate": int(
            stability_monthly["high_minus_low_completion_rate"].lt(0.0).sum()
        ),
    }
    write_json(output / "veto_model_coefficients.json", coefficient_specs)

    candidate_columns = [
        "period",
        "symbol",
        "session",
        "year_month",
        "decision_ordinal",
        "slate_id",
        "row_weight",
        "candidate_B0_probability",
        "U1_for_veto",
        "hidden_risk_group",
        "hidden_risk_quintile",
        "registered_completion_within_12_bars",
        "actual_hidden_event_within_6_bars",
    ]
    development_export = development_candidates.loc[:, candidate_columns].copy()
    assessment_export = predictions.loc[:, candidate_columns].copy()
    candidate_population = pd.concat(
        [development_export, assessment_export], ignore_index=True, sort=False
    )
    write_parquet(output / "candidate_population.parquet", candidate_population)
    write_parquet(output / "veto_assessment_predictions.parquet", predictions)
    write_csv(output / "candidate_group_metrics.csv", group_metrics)
    write_csv(output / "veto_metrics.csv", metrics)
    write_csv(output / "veto_monthly_metrics.csv", veto_monthly)
    write_csv(output / "veto_checkpoint_metrics.csv", veto_checkpoint)
    write_csv(output / "realised_diversion_metrics.csv", realised)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "veto_null_metrics.csv", null_metrics)
    concentration = concentration_metrics(predictions)
    write_csv(output / "concentration_metrics.csv", concentration)

    primary_decision = choose_primary_decision(
        precursor_status=precursor_status,
        predictive_veto_status=predictive_status,
        realised_diversion_status=realised_status,
    )
    decision = {
        **SAFETY_FLAGS,
        "primary_decision": primary_decision,
        "precursor_status": precursor_status,
        "predictive_veto_status": predictive_status,
        "realised_diversion_status": realised_status,
        "precursor_gate": precursor_gate,
        "predictive_veto_gate": predictive_gate,
        "realised_diversion_gate": realised_gate,
        "candidate_threshold": threshold,
        "hidden_risk_thresholds": hidden_manifest,
        "V1_hidden_risk_coefficient": hidden_coefficient,
        "V1_hidden_risk_standard_error": hidden_standard_error,
        "candidate_veto_removal": removal,
        "protected_rows_materialised": 0,
        "lightweight_audit_passed": False,
        "determinism_check_passed": False,
        "retrospective_only": True,
        "prospective_validation": False,
        "permission_to_trade": False,
    }
    source_manifest = {
        **SAFETY_FLAGS,
        "development_period": ["2024-01-01", "2024-12-31"],
        "assessment_period": ["2025-01-01", "2025-08-22"],
        "minimum_timestamp_read": v2_source["minimum_timestamp_read"],
        "maximum_timestamp_read": v2_source["maximum_timestamp_read"],
        "protected_rows_materialised": 0,
        "date_predicate_applied_before_materialisation": True,
        "v2_state_source": v2_source,
        "semantic_dictionary": dictionary_manifest,
        "trace_reconstruction": trace_manifest,
        "precursor_null": null_manifest,
        "predecessor_artifacts": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in (
                BRIDGE_PRIMARY / "bridge_development_panel.parquet",
                BRIDGE_PRIMARY / "bridge_assessment_predictions.parquet",
                BRIDGE_PRIMARY / "bridge_model_coefficients.json",
                BRIDGE_PRIMARY / "registered_completion_ledger.parquet",
                OPENING_PRIMARY / "unregistered_path_ledger.parquet",
                OPENING_PRIMARY / "hidden_family_mapping.json",
            )
        },
    }
    protected_audit = {
        **SAFETY_FLAGS,
        "development_start": "2024-01-01",
        "development_end_inclusive": "2024-12-31",
        "assessment_start": "2025-01-01",
        "assessment_end_inclusive": "2025-08-22",
        "protected_start": "2025-08-23",
        "minimum_timestamp_read": source_manifest["minimum_timestamp_read"],
        "maximum_timestamp_read": source_manifest["maximum_timestamp_read"],
        "protected_rows_materialised": 0,
        "passed": pd.Timestamp(source_manifest["maximum_timestamp_read"]) < PROTECTED_START,
    }
    if not protected_audit["passed"]:
        raise ScreenBlocker(
            "blocked_protected_boundary_failure", "protected boundary cannot be proved"
        )
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "protected_boundary_audit.json", protected_audit)
    write_json(output / "decision.json", decision)

    print("performing fast deterministic model/census reconstruction", flush=True)
    check = determinism_check(output)
    decision["determinism_check_passed"] = bool(check["passed"])
    if not check["passed"]:
        decision["primary_decision"] = "blocked_reproducibility_or_audit_failure"
    write_json(output / "determinism_check.json", check)
    write_json(output / "decision.json", decision)
    plots = create_plots(precursor_null, group_metrics, output)
    decision["plots"] = plots
    write_json(output / "decision.json", decision)
    report = build_report(
        decision,
        precursor_census,
        nearest,
        precursor_null,
        group_metrics,
        metrics,
        bootstrap,
        realised,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


def write_blocker(output: Path, blocker: ScreenBlocker) -> None:
    output.mkdir(parents=True, exist_ok=True)
    decision = {
        **SAFETY_FLAGS,
        "primary_decision": blocker.code,
        "precursor_status": "insufficient_support",
        "predictive_veto_status": "insufficient_support",
        "realised_diversion_status": "insufficient_support",
        "blocker_detail": blocker.detail,
        "protected_rows_materialised": 0,
        "determinism_check_passed": False,
        "lightweight_audit_passed": False,
    }
    write_json(output / "decision.json", decision)


def finalize_existing(output: Path) -> dict[str, Any]:
    """Render reports from completed frozen artifacts without repeating calculations."""

    decision = read_json(output / "decision.json")
    report = build_report(
        decision,
        pd.read_csv(output / "precursor_census.csv"),
        pd.read_csv(output / "nearest_precursor_census.csv"),
        pd.read_csv(output / "precursor_null_metrics.csv"),
        pd.read_csv(output / "candidate_group_metrics.csv"),
        pd.read_csv(output / "veto_metrics.csv"),
        pd.read_csv(output / "bootstrap_metrics.csv"),
        pd.read_csv(output / "realised_diversion_metrics.csv"),
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


def refresh_existing_determinism(output: Path) -> dict[str, Any]:
    """Rerun only the preregistered fast determinism check and refresh reports."""

    decision = read_json(output / "decision.json")
    decision["primary_decision"] = choose_primary_decision(
        precursor_status=str(decision["precursor_status"]),
        predictive_veto_status=str(decision["predictive_veto_status"]),
        realised_diversion_status=str(decision["realised_diversion_status"]),
    )
    write_json(output / "decision.json", decision)
    check = determinism_check(output)
    decision["determinism_check_passed"] = bool(check["passed"])
    if not check["passed"]:
        decision["primary_decision"] = "blocked_reproducibility_or_audit_failure"
    write_json(output / "determinism_check.json", check)
    write_json(output / "decision.json", decision)
    return finalize_existing(output)


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
        help="render reports from completed artifacts without repeating bootstrap or nulls",
    )
    parser.add_argument(
        "--refresh-determinism-existing",
        action="store_true",
        help="rerun only deterministic refits/census checks and refresh reports",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    output = arguments.output.expanduser().resolve()
    if arguments.refresh_determinism_existing:
        decision = refresh_existing_determinism(output)
        print(canonical_json(decision), end="")
        return 0
    if arguments.finalize_existing:
        decision = finalize_existing(output)
        print(canonical_json(decision), end="")
        return 0
    try:
        decision = execute_screen(
            output, provider_root=arguments.provider_root.expanduser().resolve()
        )
        print(canonical_json(decision), end="")
        return 0
    except ScreenBlocker as blocker:
        write_blocker(output, blocker)
        print(blocker.code)
        print(blocker.detail, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
