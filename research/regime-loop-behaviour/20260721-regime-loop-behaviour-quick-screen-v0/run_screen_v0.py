#!/usr/bin/env python3
"""Run Regime x Loop Prefix x Behavioural Context Quick Screen V0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from stocker_research.loop_dictionary_v2 import (
    LoopDictionary,
    decompose_closed_path,
)
from stocker_research.loop_prefix_automaton_v2 import FirstNextLoopEventEngine
from stocker_research.regime_gap_segmentation_v2 import causal_segment_groups
from stocker_research.regime_loop_behaviour_quick_v0 import (
    BEHAVIOURAL_DIMENSIONS,
    INTERACTION_FEATURES,
    SAFETY_FLAGS,
    active_prefix_records,
    apply_interaction_clipping,
    assert_decision_time_causality,
    assert_no_protected_rows,
    assign_candidate_weights,
    causal_checkpoint_filter,
    compute_interactions,
    decide_screen,
    fit_candidate_logistic,
    fit_interaction_clipping,
    normalize_first_event_outcome,
    permute_behavioural_bundle_within_slates,
    session_block_bootstrap_draws,
    target_candidate_rows,
    verify_behavioural_ledger_reconstruction,
)
from stocker_research.regime_panel_v2 import (
    EMISSION_FEATURES,
    NATURAL_KEY,
    RegimePanelConfig,
    build_regime_panel,
    canonical_frame_hash,
)
from stocker_research.regime_validity_v2 import (
    EmissionPreprocessing,
    SemiMarkovParameters,
    gaussian_log_emissions,
    transform_emissions,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
DEFAULT_EXACT = EXPERIMENT_DIR / "artifacts" / "exact_rerun"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
AUDITOR_PATH = EXPERIMENT_DIR / "audit_screen_v0.py"

SLRNO_WORK = REPO_ROOT / "research" / "slrno-v2" / "20260714-regime-loop-handoff" / "work"
REFIT_DIR = SLRNO_WORK / "artifacts" / "20260719-right-censored-regime-refit-v2" / "primary"
EVENT_DIR = SLRNO_WORK / "artifacts" / "20260718-loop-event-semantics-v2" / "primary"
PARAMETERS_PATH = REFIT_DIR / "full_refit_parameters.npz"
PREPROCESSING_PATH = REFIT_DIR / "full_refit_preprocessing.csv"
CENTROIDS_PATH = REFIT_DIR / "full_refit_cluster_centroids.csv"
PANEL_HASHES_PATH = REFIT_DIR / "panel_hashes.json"
DICTIONARY_PATH = EVENT_DIR / "semantic_loop_dictionary_v2.csv"

BEHAVIOURAL_RELATIVE = Path(
    "research/observable-behavioural-state/"
    "20260721-behavioural-state-dimensions-screen-v0/artifacts"
)
OPENING_RELATIVE = Path(
    "research/opening-regime-path/"
    "20260720-opening-regime-path-direction-screen-v0/artifacts/primary/"
    "opening_state_path_ledger.parquet"
)

START = pd.Timestamp("2024-01-01T00:00:00Z")
ASSESSMENT_START = pd.Timestamp("2025-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
READ_END_INCLUSIVE = PROTECTED_START - pd.Timedelta(microseconds=1)
STATE_COUNT = 8
EXPECTED_SESSION_BARS = 78
HORIZON_BARS = 6
MAX_STATE_BAR_ORDINAL = 17
MAX_CANDIDATE_ROWS = 50_000
BOOTSTRAP_DRAWS = 100
BOOTSTRAP_SEED = 20260721
NULL_DRAWS = 25
NULL_SEED = 20260722

EXPECTED_PANEL_HASH = "801c0bf9d69ecdd58b21fb2ba4392137048b466668344ebfc4c8faf6a0d3e2f1"
EXPECTED_MODEL_HASH = "4fc1a02dce9ac2311dabaeb4623a559d37286dfe58baffef53828cc7415a3425"
EXPECTED_DICTIONARY_HASH = "497142c8d0ab880e59385da123d9eb2189469e9e3a4a631e0f63eb6fc77030d3"
EXPECTED_BEHAVIOURAL_HASHES = {
    "behavioural_dimension_ledger.parquet": (
        "cd5e7cee343952638bb17d4e4ea5d58918cdd1b13477275c312ea21e37d2dee0"
    ),
    "compact_decision_panel.parquet": (
        "589edb9b3dcbe5ea9dd91d1330054c03c5fee4b3baff0ea030559dd754bec479"
    ),
    "behavioural_component_scaling.json": (
        "6b54a3ce150dcce054fed7a370e411d9d555f03d77f022dcb7feae52100b25ba"
    ),
}

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
REGIME_CONTEXT_SYMBOLS = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "AXTI",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "OKLO",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)
DECISION_ORDINAL_TO_BAR = {6: 5, 12: 11}

POSTERIOR_FEATURES = tuple(f"posterior_state_{state}" for state in range(STATE_COUNT))
STRUCTURAL_NUMERIC_FEATURES = (
    *POSTERIOR_FEATURES,
    "posterior_entropy",
    "top_state_probability",
    "top_versus_second_margin",
    "expected_state_age",
    "current_persistence_probability",
    "current_transition_probability",
    "candidate_orientation_sign",
    "candidate_path_length",
    "repeat_depth",
    "prefix_matched_length",
    "prefix_completion_fraction",
    "prefix_age",
    "probability_of_next_required_state",
    "remaining_session_bars",
    "checkpoint_indicator",
)
STRUCTURAL_CATEGORICAL_FEATURES = ("semantic_loop_id", "candidate_class")
TARGET = "candidate_completes_first_within_6_bars"

SCIENTIFIC_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "input_artifact_hashes.json",
    "protected_boundary_audit.json",
    "behavioural_ledger_reconstruction.json",
    "v2_prefix_population_reconstruction.json",
    "candidate_feature_manifest.json",
    "interaction_manifest.json",
    "candidate_population.parquet",
    "candidate_outcomes.parquet",
    "model_configurations.json",
    "model_coefficients.json",
    "assessment_predictions.parquet",
    "candidate_metrics.csv",
    "decision_ranking_metrics.csv",
    "monthly_metrics.csv",
    "checkpoint_metrics.csv",
    "candidate_family_metrics.csv",
    "prefix_maturity_metrics.csv",
    "bootstrap_metrics.csv",
    "null_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "independent_audit.json",
    "report.md",
)


class ScreenBlocker(RuntimeError):
    def __init__(self, decision: str, detail: str) -> None:
        super().__init__(detail)
        self.decision = decision
        self.detail = detail


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(dict(value)), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_materialised_parquet(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 4:
        raise ScreenBlocker(
            "blocked_behavioural_ledger_not_reconstructable",
            f"required frozen parquet is absent: {path.name}",
        )
    with path.open("rb") as handle:
        if handle.read(4) != b"PAR1":
            raise ScreenBlocker(
                "blocked_behavioural_ledger_not_reconstructable",
                f"required frozen parquet is not materialised: {path.name}",
            )


def _load_frozen_behavioural(
    materialized_repo: Path,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    roots = {
        "primary": materialized_repo / BEHAVIOURAL_RELATIVE / "primary",
        "exact_rerun": materialized_repo / BEHAVIOURAL_RELATIVE / "exact_rerun",
    }
    joined: dict[str, pd.DataFrame] = {}
    input_hashes: list[dict[str, Any]] = []
    within_errors: dict[str, float] = {}
    join_key = ["symbol", "session", "decision_ordinal", "slate_id"]
    ledger_columns = [*join_key, *BEHAVIOURAL_DIMENSIONS]
    compact_columns = [
        *join_key,
        "decision_bar_start_timestamp_utc",
        "feature_available_timestamp_utc",
        *BEHAVIOURAL_DIMENSIONS,
    ]
    for run_name, root in roots.items():
        ledger_path = root / "behavioural_dimension_ledger.parquet"
        compact_path = root / "compact_decision_panel.parquet"
        scaling_path = root / "behavioural_component_scaling.json"
        for path in (ledger_path, compact_path):
            _require_materialised_parquet(path)
        for path in (ledger_path, compact_path, scaling_path):
            digest = sha256_file(path)
            expected = EXPECTED_BEHAVIOURAL_HASHES[path.name]
            if digest != expected:
                raise ScreenBlocker(
                    "blocked_behavioural_ledger_not_reconstructable",
                    f"frozen {run_name} {path.name} hash differs",
                )
            input_hashes.append(
                {
                    "logical_path": str(BEHAVIOURAL_RELATIVE / run_name / path.name),
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                }
            )
        ledger = pd.read_parquet(ledger_path, columns=ledger_columns)
        compact = pd.read_parquet(compact_path, columns=compact_columns)
        if ledger.duplicated(join_key).any() or compact.duplicated(join_key).any():
            raise ScreenBlocker(
                "blocked_behavioural_ledger_not_reconstructable",
                f"frozen {run_name} behavioural natural key is not unique",
            )
        merged = ledger.merge(
            compact,
            on=join_key,
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_ledger", "_compact"),
        )
        if not merged["_merge"].eq("both").all():
            raise ScreenBlocker(
                "blocked_behavioural_ledger_not_reconstructable",
                f"frozen {run_name} ledger cannot join to decision timestamps",
            )
        errors = []
        for dimension in BEHAVIOURAL_DIMENSIONS:
            errors.append(
                np.abs(
                    merged[f"{dimension}_ledger"].to_numpy(dtype=float)
                    - merged[f"{dimension}_compact"].to_numpy(dtype=float)
                )
            )
        maximum_error = float(np.max(np.column_stack(errors), initial=0.0))
        if maximum_error > 1e-12:
            raise ScreenBlocker(
                "blocked_behavioural_ledger_not_reconstructable",
                f"frozen {run_name} ledger-to-panel values differ by {maximum_error}",
            )
        within_errors[run_name] = maximum_error
        output = merged.loc[
            :,
            [
                *join_key,
                "decision_bar_start_timestamp_utc",
                "feature_available_timestamp_utc",
                *(f"{name}_ledger" for name in BEHAVIOURAL_DIMENSIONS),
            ],
        ].rename(
            columns={
                "feature_available_timestamp_utc": "decision_timestamp",
                **{f"{name}_ledger": name for name in BEHAVIOURAL_DIMENSIONS},
            }
        )
        output["decision_timestamp"] = pd.to_datetime(output["decision_timestamp"], utc=True)
        output["decision_bar_start_timestamp_utc"] = pd.to_datetime(
            output["decision_bar_start_timestamp_utc"], utc=True
        )
        joined[run_name] = output.sort_values(join_key, kind="mergesort").reset_index(drop=True)
    reconstruction = verify_behavioural_ledger_reconstruction(
        joined["primary"].rename(columns={"feature_available_timestamp_utc": "decision_timestamp"}),
        joined["exact_rerun"].rename(
            columns={"feature_available_timestamp_utc": "decision_timestamp"}
        ),
    )
    reconstruction.update(
        {
            "primary_ledger_to_compact_maximum_absolute_error": within_errors["primary"],
            "exact_rerun_ledger_to_compact_maximum_absolute_error": within_errors["exact_rerun"],
            "primary_exact_rerun_file_hashes_match": True,
            "predecessor_decision": "behavioural_descriptions_only_no_predictive_increment",
            "predecessor_commit": "ef9c2e4a6636d5404fbc2767a50ecb10a413f14a",
        }
    )
    return joined["primary"], reconstruction, input_hashes


def _load_structural_inputs() -> tuple[
    EmissionPreprocessing,
    SemiMarkovParameters,
    LoopDictionary,
    dict[int, int],
    list[dict[str, Any]],
]:
    preprocessing_frame = pd.read_csv(PREPROCESSING_PATH)
    preprocessing = EmissionPreprocessing(
        feature_names=tuple(preprocessing_frame["feature"].astype(str)),
        medians=preprocessing_frame["imputer_median"].to_numpy(dtype=float),
        centers=preprocessing_frame["scaler_center"].to_numpy(dtype=float),
        scales=preprocessing_frame["scaler_scale"].to_numpy(dtype=float),
    )
    preprocessing.validate()
    if preprocessing.feature_names != tuple(EMISSION_FEATURES):
        raise ScreenBlocker(
            "blocked_v2_prefix_population_not_reconstructable",
            "frozen V2 preprocessing feature order differs",
        )
    with np.load(PARAMETERS_PATH) as stored:
        parameters = SemiMarkovParameters(
            means=np.asarray(stored["means"]).copy(),
            variances=np.asarray(stored["variances"]).copy(),
            duration_hazard=np.asarray(stored["duration_hazard"]).copy(),
            transitions=np.asarray(stored["transitions"]).copy(),
            initial=np.asarray(stored["initial"]).copy(),
            occupancy=np.asarray(stored["occupancy"]).copy(),
        )
        model_hash = str(np.asarray(stored["state_model_hash"]).item())
    parameters.validate()
    if model_hash != EXPECTED_MODEL_HASH:
        raise ScreenBlocker(
            "blocked_v2_prefix_population_not_reconstructable",
            "frozen repaired V2 model hash differs",
        )
    table = pd.read_csv(DICTIONARY_PATH)
    if (
        len(table) != 20
        or not table["dictionary_hash"].astype(str).eq(EXPECTED_DICTIONARY_HASH).all()
    ):
        raise ScreenBlocker(
            "blocked_v2_prefix_population_not_reconstructable",
            "frozen semantic loop dictionary identity differs",
        )
    definitions = []
    for row in table.sort_values("semantic_loop_id", kind="mergesort").itertuples(index=False):
        path = tuple(int(value) for value in str(row.canonical_path).split("->"))
        definition = decompose_closed_path(path)
        if (
            definition.semantic_loop_id != str(row.semantic_loop_id)
            or definition.motif_type.value != str(row.motif_type)
            or definition.repeat_depth != int(row.repeat_depth)
        ):
            raise ScreenBlocker(
                "blocked_v2_prefix_population_not_reconstructable",
                f"semantic loop definition differs for {row.semantic_loop_id}",
            )
        definitions.append(definition)
    dictionary = LoopDictionary.from_definitions(definitions, version="semantic_loop_dictionary_v2")
    centroids = pd.read_csv(CENTROIDS_PATH)
    direction = centroids.loc[
        centroids["feature"].isin(["signed_efficiency_6", "signed_efficiency_12"])
    ]
    state_direction = direction.groupby("state", sort=True)["raw_feature_centroid"].mean()
    if set(state_direction.index.astype(int)) != set(range(STATE_COUNT)):
        raise ScreenBlocker(
            "blocked_v2_prefix_population_not_reconstructable",
            "frozen state directional centroids are incomplete",
        )
    orientation_sign = {
        int(state): (1 if float(value) >= 0.0 else -1) for state, value in state_direction.items()
    }
    inputs = [
        {
            "logical_path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (
            PARAMETERS_PATH,
            PREPROCESSING_PATH,
            CENTROIDS_PATH,
            PANEL_HASHES_PATH,
            DICTIONARY_PATH,
            EVENT_DIR / "independent_audit.json",
            EVENT_DIR / "decision.json",
            REFIT_DIR / "repair_decision.json",
            REFIT_DIR / "independent_audit.json",
        )
    ]
    return preprocessing, parameters, dictionary, orientation_sign, inputs


def _whole_session_is_valid(frame: pd.DataFrame) -> bool:
    ordered = frame.sort_values("bar_ordinal", kind="mergesort")
    ordinals = pd.to_numeric(ordered["bar_ordinal"], errors="coerce")
    return bool(
        len(ordered) == EXPECTED_SESSION_BARS
        and ordinals.tolist() == list(range(EXPECTED_SESSION_BARS))
        and ordered["segment_id"].astype(str).nunique() == 1
        and ordered["session_source_complete"].astype(bool).all()
        and not ordered["source_data_error_in_session"].astype(bool).any()
        and ordered["expected_session_bars"].astype(int).eq(EXPECTED_SESSION_BARS).all()
        and not pd.to_datetime(ordered["bar_start_timestamp"], utc=True).ge(PROTECTED_START).any()
    )


def _build_structural_panel(
    provider_root: Path,
    preprocessing: EmissionPreprocessing,
    parameters: SemiMarkovParameters,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    built = build_regime_panel(
        RegimePanelConfig(
            provider_root=provider_root,
            symbols=REGIME_CONTEXT_SYMBOLS,
            benchmark_symbol="VTI",
            start=START,
            end=READ_END_INCLUSIVE,
        )
    )
    panel = built.frame.sort_values(list(NATURAL_KEY), kind="mergesort").reset_index(drop=True)
    try:
        assert_no_protected_rows(panel["bar_start_timestamp"])
    except ValueError as error:
        raise ScreenBlocker("blocked_protected_boundary_failure", str(error)) from error
    development = panel.loc[
        pd.to_datetime(panel["bar_start_timestamp"], utc=True).lt(ASSESSMENT_START)
    ]
    panel_hash = canonical_frame_hash(development, columns=(*NATURAL_KEY, *EMISSION_FEATURES))
    if panel_hash != EXPECTED_PANEL_HASH:
        raise ScreenBlocker(
            "blocked_v2_prefix_population_not_reconstructable",
            "reconstructed 2024 V2 panel hash differs",
        )
    valid_keys = {
        (str(symbol), str(session))
        for (symbol, session), frame in panel.loc[panel["symbol"].isin(DECISION_SYMBOLS)].groupby(
            ["symbol", "session"], sort=True
        )
        if _whole_session_is_valid(frame)
    }
    panel_keys = pd.MultiIndex.from_arrays(
        [panel["symbol"].astype(str), panel["session"].astype(str)]
    )
    valid_index = pd.MultiIndex.from_tuples(sorted(valid_keys), names=["symbol", "session"])
    state_panel = panel.loc[
        panel_keys.isin(valid_index) & panel["bar_ordinal"].le(MAX_STATE_BAR_ORDINAL)
    ].copy()
    state_panel = state_panel.sort_values(list(NATURAL_KEY), kind="mergesort").reset_index(
        drop=True
    )
    if state_panel.empty:
        raise ScreenBlocker(
            "blocked_v2_prefix_population_not_reconstructable",
            "no complete V2 checkpoint sessions were reconstructed",
        )
    filtered = causal_checkpoint_filter(
        gaussian_log_emissions(transform_emissions(state_panel, preprocessing), parameters),
        groups=causal_segment_groups(state_panel),
        model=parameters.as_dict(),
    )
    state_panel["causal_hard_state"] = filtered.hard_states
    state_panel["posterior_entropy"] = filtered.posterior_entropy
    state_panel["top_state_probability"] = filtered.top_state_probability
    state_panel["top_versus_second_margin"] = filtered.top_versus_second_margin
    state_panel["expected_state_age"] = filtered.expected_state_age
    state_panel["current_persistence_probability"] = filtered.current_persistence_probability
    state_panel["current_transition_probability"] = filtered.current_transition_probability
    for state in range(STATE_COUNT):
        state_panel[f"posterior_state_{state}"] = filtered.state_probabilities[:, state]
        state_panel[f"next_state_probability_{state}"] = filtered.next_state_probabilities[:, state]
    manifest = {
        **SAFETY_FLAGS,
        "panel_rows_materialised": len(panel),
        "state_rows_materialised": len(state_panel),
        "valid_stock_sessions": len(valid_keys),
        "minimum_timestamp_read": str(pd.to_datetime(panel["bar_start_timestamp"], utc=True).min()),
        "maximum_timestamp_read": str(pd.to_datetime(panel["bar_start_timestamp"], utc=True).max()),
        "protected_rows_materialised": int(
            pd.to_datetime(panel["bar_start_timestamp"], utc=True).ge(PROTECTED_START).sum()
        ),
        "development_panel_hash": panel_hash,
        "state_model_hash": EXPECTED_MODEL_HASH,
        "data_snapshot_hash": built.data_snapshot_hash,
        "source_hashes": built.source_hashes,
        "source_row_counts": built.source_row_counts,
        "panel_row_key_hash": built.row_key_hash,
        "panel_feature_table_hash": built.feature_table_hash,
    }
    del panel
    return state_panel, manifest


def _validate_against_opening_ledger(
    state_panel: pd.DataFrame,
    materialized_repo: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = materialized_repo / OPENING_RELATIVE
    if not path.is_file() or path.stat().st_size < 4 or path.read_bytes()[:4] != b"PAR1":
        raise ScreenBlocker(
            "blocked_v2_prefix_population_not_reconstructable",
            "materialised frozen opening state-path ledger is unavailable",
        )
    columns = [
        "symbol",
        "session",
        "decision_ordinal",
        "repo_bar_start_ordinal",
        "feature_available_timestamp_utc",
        "opening_state_path",
        "current_state",
        "current_posterior",
    ]
    frozen = pd.read_parquet(path, columns=columns)
    frozen["feature_available_timestamp_utc"] = pd.to_datetime(
        frozen["feature_available_timestamp_utc"], utc=True
    )
    by_key = {
        (str(symbol), str(session)): frame.sort_values("bar_ordinal", kind="mergesort")
        for (symbol, session), frame in state_panel.groupby(["symbol", "session"], sort=True)
    }
    state_mismatches = 0
    timestamp_mismatches = 0
    missing = 0
    maximum_probability_error = 0.0
    checked = 0
    for row in frozen.itertuples(index=False):
        key = (str(row.symbol), str(row.session))
        frame = by_key.get(key)
        if frame is None:
            missing += 1
            continue
        origin = int(row.repo_bar_start_ordinal)
        opening = frame.loc[frame["bar_ordinal"].between(0, origin)]
        if len(opening) != origin + 1:
            missing += 1
            continue
        states = tuple(int(value) for value in str(row.opening_state_path).split(","))
        actual_states = tuple(opening["causal_hard_state"].astype(int))
        if states != actual_states or int(row.current_state) != actual_states[-1]:
            state_mismatches += 1
        current = opening.iloc[-1]
        actual_timestamp = pd.Timestamp(current["bar_start_timestamp"]) + pd.Timedelta(minutes=5)
        if actual_timestamp != pd.Timestamp(row.feature_available_timestamp_utc):
            timestamp_mismatches += 1
        expected_probability = np.asarray(
            [float(value) for value in str(row.current_posterior).split(",")], dtype=float
        )
        actual_probability = np.asarray(
            [current[f"posterior_state_{state}"] for state in range(STATE_COUNT)], dtype=float
        )
        maximum_probability_error = max(
            maximum_probability_error,
            float(np.max(np.abs(expected_probability - actual_probability))),
        )
        checked += 1
    if (
        missing
        or state_mismatches
        or timestamp_mismatches
        or maximum_probability_error > 1e-8
        or checked != len(frozen)
    ):
        raise ScreenBlocker(
            "blocked_v2_prefix_population_not_reconstructable",
            "reconstructed V2 checkpoint states differ from the frozen opening ledger",
        )
    audit = {
        **SAFETY_FLAGS,
        "opening_ledger_rows": len(frozen),
        "rows_checked": checked,
        "missing_rows": missing,
        "hard_state_path_mismatches": state_mismatches,
        "decision_timestamp_mismatches": timestamp_mismatches,
        "maximum_posterior_absolute_error": maximum_probability_error,
        "required_maximum_posterior_absolute_error": 1e-8,
        "passed": True,
    }
    identity = {
        "logical_path": str(OPENING_RELATIVE),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    return audit, identity


def _decision_trace(
    frame: pd.DataFrame,
    engine: FirstNextLoopEventEngine,
):
    ordered = frame.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
    compressed = ordered.loc[
        ordered["causal_hard_state"].ne(ordered["causal_hard_state"].shift())
    ].copy()
    starts = pd.to_datetime(compressed["bar_start_timestamp"], utc=True)
    trace = engine.scan_state_events(
        compressed["causal_hard_state"].astype(int).tolist(),
        bar_ordinals=compressed["bar_ordinal"].astype(int).tolist(),
        event_timestamps=[value.to_pydatetime() for value in starts],
        available_timestamps=[
            (value + pd.Timedelta(minutes=5)).to_pydatetime() for value in starts
        ],
    )
    return ordered, trace


def _build_candidate_population(
    behavioural: pd.DataFrame,
    state_panel: pd.DataFrame,
    dictionary: LoopDictionary,
    orientation_sign: Mapping[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(STATE_COUNT)))
    structural_groups = {
        (str(symbol), str(session)): frame
        for (symbol, session), frame in state_panel.groupby(["symbol", "session"], sort=True)
    }
    behavioural_groups = behavioural.groupby(["symbol", "session"], sort=True)
    primary_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    no_active: list[dict[str, Any]] = []
    ties: list[dict[str, Any]] = []
    missing_structural = 0
    outcome_counts: Counter[str] = Counter()
    candidates_before_ties = 0
    for (symbol_value, session_value), decisions in behavioural_groups:
        symbol = str(symbol_value)
        session = str(session_value)
        state_frame = structural_groups.get((symbol, session))
        if state_frame is None:
            missing_structural += len(decisions)
            continue
        ordered, trace = _decision_trace(state_frame, engine)
        event_bars = np.asarray([event.bar_ordinal for event in trace.state_events], dtype=int)
        for behavioural_row in decisions.sort_values(
            "decision_ordinal", kind="mergesort"
        ).itertuples(index=False):
            ordinal = int(behavioural_row.decision_ordinal)
            origin = DECISION_ORDINAL_TO_BAR.get(ordinal)
            if origin is None:
                raise ScreenBlocker(
                    "blocked_chronology_or_leakage_failure",
                    f"unregistered decision ordinal {ordinal}",
                )
            current_rows = ordered.loc[ordered["bar_ordinal"].eq(origin)]
            if len(current_rows) != 1:
                missing_structural += 1
                continue
            current = current_rows.iloc[0]
            source_timestamp = pd.Timestamp(current["bar_start_timestamp"])
            decision_timestamp = source_timestamp + pd.Timedelta(minutes=5)
            if decision_timestamp != pd.Timestamp(behavioural_row.decision_timestamp):
                raise ScreenBlocker(
                    "blocked_chronology_or_leakage_failure",
                    "behavioural and V2 decision timestamp differs for "
                    f"{symbol} {session} {ordinal}",
                )
            decision_event_index = int(np.searchsorted(event_bars, origin, side="right") - 1)
            if decision_event_index < 0:
                missing_structural += 1
                continue
            prefix_records = active_prefix_records(
                trace,
                decision_event_index=decision_event_index,
                decision_bar_ordinal=origin,
            )
            decision_id = f"{symbol}|{session}|{ordinal:02d}"
            slate_id = f"{session}|{ordinal:02d}"
            if not prefix_records:
                no_active.append(
                    {
                        "decision_id": decision_id,
                        "symbol": symbol,
                        "session": session,
                        "decision_ordinal": ordinal,
                        "reason": "no_active_registered_prefix",
                    }
                )
                continue
            outcome = engine.outcome_for_decision(
                trace,
                decision_id=decision_id,
                decision_event_index=decision_event_index,
                decision_bar_ordinal=origin,
                decision_timestamp=source_timestamp.to_pydatetime(),
                decision_available_timestamp=decision_timestamp.to_pydatetime(),
                horizon_bars=HORIZON_BARS,
                session_end_bar_ordinal=EXPECTED_SESSION_BARS - 1,
                symbol=symbol,
                session=session,
                source_hashes=(("state_model_hash", EXPECTED_MODEL_HASH),),
            )
            outcome = normalize_first_event_outcome(
                outcome,
                decision_bar_ordinal=origin,
                horizon_bars=HORIZON_BARS,
                session_end_bar_ordinal=EXPECTED_SESSION_BARS - 1,
            )
            outcome_counts[str(outcome.primary_label)] += 1
            common: dict[str, Any] = {
                "decision_id": decision_id,
                "slate_id": slate_id,
                "symbol": symbol,
                "session": session,
                "year": int(session[:4]),
                "year_month": session[:7],
                "decision_ordinal": ordinal,
                "repo_bar_start_ordinal": origin,
                "checkpoint_indicator": float(ordinal == 12),
                "decision_source_timestamp": source_timestamp,
                "decision_timestamp": decision_timestamp,
                "predictor_max_source_timestamp": source_timestamp,
                "predictor_max_available_timestamp": decision_timestamp,
                "current_state": int(current["causal_hard_state"]),
                "posterior_entropy": float(current["posterior_entropy"]),
                "top_state_probability": float(current["top_state_probability"]),
                "top_versus_second_margin": float(current["top_versus_second_margin"]),
                "expected_state_age": float(current["expected_state_age"]),
                "current_persistence_probability": float(
                    current["current_persistence_probability"]
                ),
                "current_transition_probability": float(current["current_transition_probability"]),
                "remaining_session_bars": int(EXPECTED_SESSION_BARS - origin - 1),
                "state_events_through_horizon": ",".join(
                    str(event.state)
                    for event in trace.state_events
                    if event.bar_ordinal <= origin + HORIZON_BARS
                ),
                "state_event_bars_through_horizon": ",".join(
                    str(event.bar_ordinal)
                    for event in trace.state_events
                    if event.bar_ordinal <= origin + HORIZON_BARS
                ),
                "decision_event_index": decision_event_index,
                "primary_structural_outcome": str(outcome.primary_label),
                "bars_until_first_event": outcome.bars_until_completion,
                **{name: float(getattr(behavioural_row, name)) for name in BEHAVIOURAL_DIMENSIONS},
                **{
                    f"posterior_state_{state}": float(current[f"posterior_state_{state}"])
                    for state in range(STATE_COUNT)
                },
            }
            candidate_frame = pd.DataFrame(prefix_records)
            candidates_before_ties += len(candidate_frame)
            for index, record in candidate_frame.iterrows():
                next_state = int(record["next_required_state"])
                candidate_frame.loc[index, "candidate_orientation_sign"] = int(
                    orientation_sign[next_state]
                )
                candidate_frame.loc[index, "probability_of_next_required_state"] = float(
                    current[f"next_state_probability_{next_state}"]
                )
            targeted, tied = target_candidate_rows(candidate_frame, outcome)
            if tied:
                ties.append(
                    {
                        "decision_id": decision_id,
                        "symbol": symbol,
                        "session": session,
                        "decision_ordinal": ordinal,
                        "candidate_rows_excluded": len(candidate_frame),
                        "tied_semantic_loop_ids": ",".join(outcome.tied_semantic_loop_ids),
                    }
                )
                for record in candidate_frame.to_dict(orient="records"):
                    outcome_rows.append(
                        {
                            **common,
                            **record,
                            "primary_scoring_eligible": False,
                            TARGET: None,
                            "exclusion_reason": "TIED_REGISTERED_COMPLETION",
                        }
                    )
                continue
            for record in targeted.to_dict(orient="records"):
                candidate_id = (
                    f"{decision_id}|{record['semantic_loop_id']}|"
                    f"{record['candidate_orientation']}|{int(record['prefix_start_event_index'])}|"
                    f"{int(record['prefix_matched_length'])}"
                )
                assembled = {
                    **common,
                    **record,
                    "candidate_id": candidate_id,
                    "primary_scoring_eligible": True,
                    "exclusion_reason": "",
                }
                primary_rows.append(assembled)
                outcome_rows.append(assembled.copy())
    if missing_structural:
        raise ScreenBlocker(
            "blocked_v2_prefix_population_not_reconstructable",
            f"{missing_structural} behavioural decisions lack an exact V2 structural row",
        )
    if not primary_rows:
        raise ScreenBlocker(
            "blocked_v2_prefix_population_not_reconstructable",
            "no active registered V2 candidate prefix population was reconstructed",
        )
    candidates = pd.DataFrame.from_records(primary_rows)
    outcomes = pd.DataFrame.from_records(outcome_rows)
    candidates = assign_candidate_weights(candidates)
    if len(candidates) > MAX_CANDIDATE_ROWS:
        raise ScreenBlocker(
            "blocked_resource_limit",
            f"candidate population {len(candidates)} exceeds {MAX_CANDIDATE_ROWS}",
        )
    try:
        assert_decision_time_causality(
            candidates,
            source_column="predictor_max_source_timestamp",
            available_column="predictor_max_available_timestamp",
            decision_column="decision_timestamp",
        )
        assert_no_protected_rows(candidates["decision_timestamp"])
    except ValueError as error:
        label = (
            "blocked_protected_boundary_failure"
            if "protected" in str(error)
            else "blocked_chronology_or_leakage_failure"
        )
        raise ScreenBlocker(label, str(error)) from error
    interaction_values = compute_interactions(candidates)
    candidates.loc[:, list(INTERACTION_FEATURES)] = interaction_values
    development_mask = candidates["year"].eq(2024)
    bounds = fit_interaction_clipping(candidates.loc[development_mask])
    candidates.loc[:, list(INTERACTION_FEATURES)] = apply_interaction_clipping(
        candidates.loc[:, list(INTERACTION_FEATURES)], bounds
    )
    maturity = np.select(
        [
            candidates["prefix_completion_fraction"].le(0.33),
            candidates["prefix_completion_fraction"].lt(0.67),
        ],
        ["early", "middle"],
        default="late",
    )
    candidates["prefix_maturity"] = maturity
    sort_key = [
        "session",
        "decision_ordinal",
        "symbol",
        "semantic_loop_id",
        "candidate_orientation",
        "prefix_start_event_index",
        "prefix_matched_length",
    ]
    candidates = candidates.sort_values(sort_key, kind="mergesort").reset_index(drop=True)
    outcomes = outcomes.sort_values(
        ["session", "decision_ordinal", "symbol", "semantic_loop_id", "candidate_orientation"],
        kind="mergesort",
    ).reset_index(drop=True)
    no_active_frame = pd.DataFrame.from_records(no_active)
    no_active_summary = {
        "total": len(no_active_frame),
        "by_year": {
            str(year): int(count)
            for year, count in no_active_frame["session"]
            .astype(str)
            .str[:4]
            .value_counts(sort=False)
            .sort_index()
            .items()
        },
        "by_decision_ordinal": {
            str(int(ordinal)): int(count)
            for ordinal, count in no_active_frame["decision_ordinal"]
            .value_counts(sort=False)
            .sort_index()
            .items()
        },
        "reason": "no_active_registered_prefix",
    }
    audit = {
        **SAFETY_FLAGS,
        "behavioural_decisions": len(behavioural),
        "active_prefix_decisions_before_tie_exclusion": int(
            len(set(candidates["decision_id"]).union(row["decision_id"] for row in ties))
        ),
        "primary_active_prefix_decisions": int(candidates["decision_id"].nunique()),
        "candidate_rows_before_tie_exclusion": candidates_before_ties,
        "primary_candidate_rows": len(candidates),
        "no_active_registered_prefix_decisions": len(no_active),
        "tied_registered_completion_decisions_excluded": len(ties),
        "tied_candidate_rows_excluded": int(
            sum(int(row["candidate_rows_excluded"]) for row in ties)
        ),
        "candidate_semantic_loop_support": int(candidates["semantic_loop_id"].nunique()),
        "candidate_orientation_support": int(candidates["candidate_orientation"].nunique()),
        "outcome_counts_before_tie_exclusion": dict(sorted(outcome_counts.items())),
        "no_active_registered_prefix": no_active_summary,
        "tie_exclusions": ties,
        "interaction_clipping_bounds": {
            key: {"p01": value[0], "p99": value[1]} for key, value in bounds.items()
        },
        "passed": True,
    }
    return candidates, outcomes, audit


def _support_summary(frame: pd.DataFrame) -> dict[str, Any]:
    positive = frame.loc[frame[TARGET].eq(1)]
    decisions = frame.groupby("decision_id", sort=True).size()
    return {
        "candidate_rows": len(frame),
        "active_prefix_decisions": int(frame["decision_id"].nunique()),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "positive_candidate_completions": int(frame[TARGET].sum()),
        "represented_months": int(frame["year_month"].nunique()),
        "semantic_candidate_loops": int(frame["semantic_loop_id"].nunique()),
        "multi_candidate_decisions": int(decisions.ge(2).sum()),
        "singleton_candidate_decisions": int(decisions.eq(1).sum()),
        "positive_loop_support": int(positive["semantic_loop_id"].nunique()),
        "base_rate": float(frame[TARGET].mean()),
    }


def _concentration_metrics(assessment: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for symbol, count in assessment["symbol"].value_counts(sort=False).sort_index().items():
        share = int(count) / len(assessment)
        records.append(
            {
                "entity_type": "stock_candidate_rows",
                "entity": str(symbol),
                "count": int(count),
                "denominator": len(assessment),
                "share": share,
                "maximum_allowed_share": 0.125,
                "passed": share <= 0.125,
            }
        )
    positives = assessment.loc[assessment[TARGET].eq(1)]
    for loop_id, count in (
        positives["semantic_loop_id"].value_counts(sort=False).sort_index().items()
    ):
        share = int(count) / len(positives) if len(positives) else math.nan
        records.append(
            {
                "entity_type": "loop_positive_completions",
                "entity": str(loop_id),
                "count": int(count),
                "denominator": len(positives),
                "share": share,
                "maximum_allowed_share": 0.30,
                "passed": share <= 0.30,
            }
        )
    table = pd.DataFrame.from_records(records)
    stock_max = float(table.loc[table["entity_type"].eq("stock_candidate_rows"), "share"].max())
    loop_rows = table.loc[table["entity_type"].eq("loop_positive_completions")]
    loop_max = float(loop_rows["share"].max()) if not loop_rows.empty else math.inf
    summary = {
        "maximum_stock_candidate_row_share": stock_max,
        "maximum_loop_positive_completion_share": loop_max,
        "stock_concentration_gate_passed": stock_max <= 0.125,
        "loop_concentration_gate_passed": loop_max <= 0.30,
        "concentration_gates_passed": stock_max <= 0.125 and loop_max <= 0.30,
    }
    return table, summary


def _enforce_support(
    candidates: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    development = candidates.loc[candidates["year"].eq(2024)]
    assessment = candidates.loc[candidates["year"].eq(2025)]
    development_support = _support_summary(development)
    assessment_support = _support_summary(assessment)
    concentration_table, concentration = _concentration_metrics(assessment)
    gates = {
        "active_prefix_decisions": assessment_support["active_prefix_decisions"] >= 1_000,
        "candidate_rows": assessment_support["candidate_rows"] >= 3_000,
        "sessions": assessment_support["sessions"] >= 100,
        "stocks": assessment_support["stocks"] >= 15,
        "positive_candidate_completions": assessment_support["positive_candidate_completions"]
        >= 300,
        "represented_months": assessment_support["represented_months"] >= 6,
        "semantic_candidate_loops": assessment_support["semantic_candidate_loops"] >= 10,
        "multi_candidate_decisions": assessment_support["multi_candidate_decisions"] >= 300,
        "stock_concentration": concentration["stock_concentration_gate_passed"],
        "loop_concentration": concentration["loop_concentration_gate_passed"],
    }
    assessment_support["support_gates"] = gates
    assessment_support.update(concentration)
    assessment_support["all_support_gates_passed"] = all(gates.values())
    assessment_support["failed_support_gates"] = sorted(
        key for key, passed in gates.items() if not passed
    )
    return development_support, assessment_support, concentration_table, concentration


def _calibration_parameters(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    logit = np.log(clipped / (1.0 - clipped))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = beta[0] + beta[1] * logit
        probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0)))
        loss = -np.sum(
            w
            * (
                y * np.log(np.clip(probability, 1e-15, 1.0))
                + (1.0 - y) * np.log(np.clip(1.0 - probability, 1e-15, 1.0))
            )
        )
        residual = w * (probability - y)
        gradient = np.asarray([residual.sum(), np.sum(residual * logit)], dtype=float)
        return float(loss), gradient

    fitted = minimize(
        lambda beta: objective(beta)[0],
        np.asarray([0.0, 1.0]),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    if not fitted.success and np.linalg.norm(fitted.jac) > 1e-6:
        return math.nan, math.nan
    return float(fitted.x[0]), float(fitted.x[1])


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    bins = np.minimum((np.clip(p, 0.0, 1.0) * 10).astype(int), 9)
    total = float(w.sum())
    result = 0.0
    for index in range(10):
        mask = bins == index
        if not mask.any():
            continue
        mass = float(w[mask].sum())
        observed = float(np.average(y[mask], weights=w[mask]))
        predicted = float(np.average(p[mask], weights=w[mask]))
        result += mass / total * abs(observed - predicted)
    return result


def _candidate_metric_row(
    frame: pd.DataFrame,
    *,
    model_id: str,
    probability_column: str,
    stratum_type: str,
    stratum_value: str,
) -> dict[str, Any]:
    y = frame[TARGET].to_numpy(dtype=float)
    p = frame[probability_column].to_numpy(dtype=float)
    w = frame["row_weight"].to_numpy(dtype=float)
    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    intercept, slope = _calibration_parameters(y, p, w)
    auc = float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else math.nan
    return {
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "model_id": model_id,
        "brier_score": float(np.average(np.square(p - y), weights=w)),
        "log_loss": float(
            -np.average(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped), weights=w)
        ),
        "auc": auc,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": _expected_calibration_error(y, p, w),
        "base_rate": float(np.average(y, weights=w)),
        "rows": len(frame),
        "decisions": int(frame["decision_id"].nunique()),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "candidate_loop_support": int(frame["semantic_loop_id"].nunique()),
        "row_weight_sum": float(w.sum()),
    }


def _ranking_detail(frame: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for decision_id, group in frame.groupby("decision_id", sort=True):
        ordered = group.sort_values(
            [probability_column, "semantic_loop_id", "candidate_orientation", "candidate_id"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        candidate_count = len(ordered)
        positives = np.flatnonzero(ordered[TARGET].to_numpy(dtype=int) == 1)
        realised = len(positives) == 1
        rank = int(positives[0] + 1) if realised else None
        records.append(
            {
                "decision_id": str(decision_id),
                "session": str(ordered.iloc[0]["session"]),
                "decision_ordinal": int(ordered.iloc[0]["decision_ordinal"]),
                "candidate_count": candidate_count,
                "realised_candidate_present": realised,
                "top_one_realised_candidate_accuracy": (
                    float(rank == 1) if rank is not None else math.nan
                ),
                "reciprocal_rank": (1.0 / rank if rank is not None else math.nan),
                "probability_assigned_to_realised_candidate": (
                    float(ordered.iloc[positives[0]][probability_column]) if realised else math.nan
                ),
                "top_three_inclusion": (
                    float(rank <= 3) if rank is not None and candidate_count >= 3 else math.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _ranking_metric_rows(
    frame: pd.DataFrame,
    *,
    model_id: str,
    probability_column: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    detail = _ranking_detail(frame, probability_column)
    populations = {
        "multi_candidate_realised": detail[
            detail["candidate_count"].ge(2) & detail["realised_candidate_present"]
        ],
        "singleton_realised": detail[
            detail["candidate_count"].eq(1) & detail["realised_candidate_present"]
        ],
        "multi_candidate_no_realised": detail[
            detail["candidate_count"].ge(2) & ~detail["realised_candidate_present"]
        ],
        "singleton_no_realised": detail[
            detail["candidate_count"].eq(1) & ~detail["realised_candidate_present"]
        ],
    }
    rows = []
    for label, population in populations.items():
        rows.append(
            {
                "model_id": model_id,
                "population": label,
                "decisions": len(population),
                "top_one_realised_candidate_accuracy": float(
                    population["top_one_realised_candidate_accuracy"].mean()
                )
                if population["top_one_realised_candidate_accuracy"].notna().any()
                else math.nan,
                "mean_reciprocal_rank": float(population["reciprocal_rank"].mean())
                if population["reciprocal_rank"].notna().any()
                else math.nan,
                "mean_probability_assigned_to_realised_candidate": float(
                    population["probability_assigned_to_realised_candidate"].mean()
                )
                if population["probability_assigned_to_realised_candidate"].notna().any()
                else math.nan,
                "top_three_inclusion": float(population["top_three_inclusion"].mean())
                if population["top_three_inclusion"].notna().any()
                else math.nan,
                "top_three_eligible_decisions": int(
                    population["top_three_inclusion"].notna().sum()
                ),
            }
        )
    return rows, detail


def _proper_increments(
    metrics: pd.DataFrame,
    *,
    augmented: str,
    baseline: str,
) -> dict[str, float]:
    indexed = metrics.set_index("model_id")
    return {
        "brier_improvement": float(
            indexed.loc[baseline, "brier_score"] - indexed.loc[augmented, "brier_score"]
        ),
        "log_loss_improvement": float(
            indexed.loc[baseline, "log_loss"] - indexed.loc[augmented, "log_loss"]
        ),
        "auc_increment": float(indexed.loc[augmented, "auc"] - indexed.loc[baseline, "auc"]),
    }


def _top_one(frame: pd.DataFrame, probability_column: str) -> float:
    detail = _ranking_detail(frame, probability_column)
    eligible = detail.loc[detail["candidate_count"].ge(2) & detail["realised_candidate_present"]]
    return float(eligible["top_one_realised_candidate_accuracy"].mean())


def _fit_primary_models(
    candidates: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    development = candidates.loc[candidates["year"].eq(2024)].copy()
    assessment = candidates.loc[candidates["year"].eq(2025)].copy()
    specifications = {
        "M0": STRUCTURAL_NUMERIC_FEATURES,
        "M1": (*STRUCTURAL_NUMERIC_FEATURES, *BEHAVIOURAL_DIMENSIONS),
        "M2": (
            *STRUCTURAL_NUMERIC_FEATURES,
            *BEHAVIOURAL_DIMENSIONS,
            *INTERACTION_FEATURES,
        ),
    }
    models: dict[str, Any] = {}
    try:
        for model_id, numeric in specifications.items():
            models[model_id] = fit_candidate_logistic(
                development,
                target_column=TARGET,
                numeric_features=numeric,
                categorical_features=STRUCTURAL_CATEGORICAL_FEATURES,
                model_id=model_id,
            )
    except RuntimeError as error:
        raise ScreenBlocker("blocked_model_convergence_failure", str(error)) from error
    for model_id, model in models.items():
        assessment[f"probability_{model_id}"] = model.predict(assessment)
    configurations = {
        **SAFETY_FLAGS,
        "primary_model_count": 3,
        "fit_period": "2024_only",
        "assessment_period": "2025-01-01_through_2025-08-22",
        "target": TARGET,
        "categorical_features": list(STRUCTURAL_CATEGORICAL_FEATURES),
        "models": {
            model_id: {
                "numeric_features": list(numeric),
                "categorical_features": list(STRUCTURAL_CATEGORICAL_FEATURES),
                "penalty": "l2",
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 250,
                "class_weight": None,
                "n_jobs": 1,
            }
            for model_id, numeric in specifications.items()
        },
    }
    coefficients = {
        **SAFETY_FLAGS,
        "models": {model_id: model.as_dict() for model_id, model in models.items()},
    }
    return models, assessment, {"configurations": configurations, "coefficients": coefficients}


def _all_metric_tables(
    assessment: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    model_ids = ("M0", "M1", "M2")

    def rows_for(frame: pd.DataFrame, stratum_type: str, stratum_value: str):
        return [
            _candidate_metric_row(
                frame,
                model_id=model_id,
                probability_column=f"probability_{model_id}",
                stratum_type=stratum_type,
                stratum_value=stratum_value,
            )
            for model_id in model_ids
        ]

    candidate_metrics = pd.DataFrame.from_records(rows_for(assessment, "pooled", "assessment"))
    monthly_records: list[dict[str, Any]] = []
    for month, frame in assessment.groupby("year_month", sort=True):
        monthly_records.extend(rows_for(frame, "month", str(month)))
    checkpoint_records: list[dict[str, Any]] = []
    for ordinal, frame in assessment.groupby("decision_ordinal", sort=True):
        checkpoint_records.extend(rows_for(frame, "decision_ordinal", str(int(ordinal))))
    family_records: list[dict[str, Any]] = []
    for family, frame in assessment.groupby("candidate_class", sort=True):
        family_records.extend(rows_for(frame, "candidate_class", str(family)))
    maturity_records: list[dict[str, Any]] = []
    for maturity, frame in assessment.groupby("prefix_maturity", sort=True):
        maturity_records.extend(rows_for(frame, "prefix_maturity", str(maturity)))
    ranking_records: list[dict[str, Any]] = []
    ranking_details: dict[str, pd.DataFrame] = {}
    for model_id in model_ids:
        rows, detail = _ranking_metric_rows(
            assessment,
            model_id=model_id,
            probability_column=f"probability_{model_id}",
        )
        ranking_records.extend(rows)
        ranking_details[model_id] = detail
    return (
        candidate_metrics,
        pd.DataFrame.from_records(ranking_records),
        pd.DataFrame.from_records(monthly_records),
        pd.DataFrame.from_records(checkpoint_records),
        pd.DataFrame.from_records(family_records),
        pd.DataFrame.from_records(maturity_records),
        ranking_details,
    )


def _bootstrap_metrics(assessment: pd.DataFrame) -> pd.DataFrame:
    draws = session_block_bootstrap_draws(
        assessment["session"].astype(str).tolist(),
        draws=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
    )
    comparisons = {
        "M1_minus_M0": ("M1", "M0"),
        "M2_minus_M1": ("M2", "M1"),
    }
    values: dict[tuple[str, str], list[float]] = {
        (comparison, metric): []
        for comparison in comparisons
        for metric in ("brier_improvement", "log_loss_improvement", "top_one_accuracy_improvement")
    }
    draw_records: list[dict[str, Any]] = []
    by_session = {
        str(session): frame for session, frame in assessment.groupby("session", sort=True)
    }
    for draw in draws:
        blocks = []
        for replica, session in enumerate(draw.sampled_sessions):
            block = by_session[session].copy()
            block["decision_id"] = (
                block["decision_id"]
                .astype(str)
                .map(lambda value, replica=replica: f"{replica:04d}|{value}")
            )
            blocks.append(block)
        sampled = pd.concat(blocks, ignore_index=True)
        metrics = pd.DataFrame.from_records(
            [
                _candidate_metric_row(
                    sampled,
                    model_id=model_id,
                    probability_column=f"probability_{model_id}",
                    stratum_type="bootstrap",
                    stratum_value=str(draw.draw),
                )
                for model_id in ("M0", "M1", "M2")
            ]
        )
        for comparison, (augmented, baseline) in comparisons.items():
            increments = _proper_increments(metrics, augmented=augmented, baseline=baseline)
            increments["top_one_accuracy_improvement"] = _top_one(
                sampled, f"probability_{augmented}"
            ) - _top_one(sampled, f"probability_{baseline}")
            for metric, value in increments.items():
                values[(comparison, metric)].append(value)
                draw_records.append(
                    {
                        "record_type": "draw",
                        "comparison": comparison,
                        "metric": metric,
                        "draw": draw.draw,
                        "value": value,
                        "interval_level": math.nan,
                        "lower": math.nan,
                        "upper": math.nan,
                    }
                )
    interval_records: list[dict[str, Any]] = []
    for (comparison, metric), raw in sorted(values.items()):
        array = np.asarray(raw, dtype=float)
        for level, lower_q, upper_q in ((0.90, 0.05, 0.95), (0.95, 0.025, 0.975)):
            interval_records.append(
                {
                    "record_type": "interval",
                    "comparison": comparison,
                    "metric": metric,
                    "draw": math.nan,
                    "value": math.nan,
                    "interval_level": level,
                    "lower": float(np.quantile(array, lower_q, method="linear")),
                    "upper": float(np.quantile(array, upper_q, method="linear")),
                }
            )
    return pd.DataFrame.from_records([*draw_records, *interval_records])


def _null_metrics(
    candidates: pd.DataFrame,
    assessment_real: pd.DataFrame,
) -> pd.DataFrame:
    development_mask = candidates["year"].eq(2024)
    real_metrics = pd.DataFrame.from_records(
        [
            _candidate_metric_row(
                assessment_real,
                model_id=model_id,
                probability_column=f"probability_{model_id}",
                stratum_type="real",
                stratum_value="assessment",
            )
            for model_id in ("M0", "M1", "M2")
        ]
    )
    real_increments = {
        "M1_minus_M0": _proper_increments(real_metrics, augmented="M1", baseline="M0"),
        "M2_minus_M1": _proper_increments(real_metrics, augmented="M2", baseline="M1"),
    }
    real_increments["M1_minus_M0"]["top_one_accuracy_improvement"] = _top_one(
        assessment_real, "probability_M1"
    ) - _top_one(assessment_real, "probability_M0")
    real_increments["M2_minus_M1"]["top_one_accuracy_improvement"] = _top_one(
        assessment_real, "probability_M2"
    ) - _top_one(assessment_real, "probability_M1")
    draw_records: list[dict[str, Any]] = []
    values: dict[tuple[str, str], list[float]] = {
        (comparison, metric): []
        for comparison in ("M1_minus_M0", "M2_minus_M1")
        for metric in ("brier_improvement", "log_loss_improvement", "top_one_accuracy_improvement")
    }
    for draw in range(NULL_DRAWS):
        permuted = permute_behavioural_bundle_within_slates(
            candidates,
            seed=NULL_SEED,
            draw=draw,
        )
        raw_interactions = compute_interactions(permuted)
        permuted.loc[:, list(INTERACTION_FEATURES)] = raw_interactions
        bounds = fit_interaction_clipping(
            permuted.loc[development_mask, list(INTERACTION_FEATURES)]
        )
        permuted.loc[:, list(INTERACTION_FEATURES)] = apply_interaction_clipping(
            permuted.loc[:, list(INTERACTION_FEATURES)], bounds
        )
        development = permuted.loc[development_mask]
        assessment = permuted.loc[~development_mask].copy()
        try:
            null_m1 = fit_candidate_logistic(
                development,
                target_column=TARGET,
                numeric_features=(*STRUCTURAL_NUMERIC_FEATURES, *BEHAVIOURAL_DIMENSIONS),
                categorical_features=STRUCTURAL_CATEGORICAL_FEATURES,
                model_id=f"M1_null_{draw:02d}",
            )
            null_m2 = fit_candidate_logistic(
                development,
                target_column=TARGET,
                numeric_features=(
                    *STRUCTURAL_NUMERIC_FEATURES,
                    *BEHAVIOURAL_DIMENSIONS,
                    *INTERACTION_FEATURES,
                ),
                categorical_features=STRUCTURAL_CATEGORICAL_FEATURES,
                model_id=f"M2_null_{draw:02d}",
            )
        except RuntimeError as error:
            raise ScreenBlocker("blocked_model_convergence_failure", str(error)) from error
        assessment["probability_M0"] = assessment_real["probability_M0"].to_numpy()
        assessment["probability_M1"] = null_m1.predict(assessment)
        assessment["probability_M2"] = null_m2.predict(assessment)
        metrics = pd.DataFrame.from_records(
            [
                _candidate_metric_row(
                    assessment,
                    model_id=model_id,
                    probability_column=f"probability_{model_id}",
                    stratum_type="null",
                    stratum_value=str(draw),
                )
                for model_id in ("M0", "M1", "M2")
            ]
        )
        increments = {
            "M1_minus_M0": _proper_increments(metrics, augmented="M1", baseline="M0"),
            "M2_minus_M1": _proper_increments(metrics, augmented="M2", baseline="M1"),
        }
        increments["M1_minus_M0"]["top_one_accuracy_improvement"] = _top_one(
            assessment, "probability_M1"
        ) - _top_one(assessment, "probability_M0")
        increments["M2_minus_M1"]["top_one_accuracy_improvement"] = _top_one(
            assessment, "probability_M2"
        ) - _top_one(assessment, "probability_M1")
        for comparison, metrics_by_name in increments.items():
            for metric, value in metrics_by_name.items():
                values[(comparison, metric)].append(value)
                draw_records.append(
                    {
                        "record_type": "draw",
                        "comparison": comparison,
                        "metric": metric,
                        "draw": draw,
                        "value": value,
                        "real_value": real_increments[comparison][metric],
                        "null_90th_percentile": math.nan,
                        "real_percentile": math.nan,
                    }
                )
    summary_records: list[dict[str, Any]] = []
    for (comparison, metric), raw in sorted(values.items()):
        array = np.asarray(raw, dtype=float)
        real = real_increments[comparison][metric]
        summary_records.append(
            {
                "record_type": "summary",
                "comparison": comparison,
                "metric": metric,
                "draw": math.nan,
                "value": math.nan,
                "real_value": real,
                "null_90th_percentile": float(np.quantile(array, 0.90, method="linear")),
                "real_percentile": float(100.0 * np.mean(array <= real)),
            }
        )
    return pd.DataFrame.from_records([*draw_records, *summary_records])


def _comparison_by_stratum(
    table: pd.DataFrame,
    *,
    augmented: str,
    baseline: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (stratum_type, stratum_value), group in table.groupby(
        ["stratum_type", "stratum_value"], sort=True
    ):
        increments = _proper_increments(group, augmented=augmented, baseline=baseline)
        records.append(
            {
                "stratum_type": str(stratum_type),
                "stratum_value": str(stratum_value),
                **increments,
            }
        )
    return pd.DataFrame.from_records(records)


def _decision_payload(
    candidate_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    checkpoint_metrics: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    null_metrics: pd.DataFrame,
    concentration: Mapping[str, Any],
    development_support: Mapping[str, Any],
    assessment_support: Mapping[str, Any],
) -> dict[str, Any]:
    m1 = _proper_increments(candidate_metrics, augmented="M1", baseline="M0")
    m2 = _proper_increments(candidate_metrics, augmented="M2", baseline="M1")
    ranking = ranking_metrics.loc[
        ranking_metrics["population"].eq("multi_candidate_realised")
    ].set_index("model_id")
    m1["top_one_accuracy_improvement"] = float(
        ranking.loc["M1", "top_one_realised_candidate_accuracy"]
        - ranking.loc["M0", "top_one_realised_candidate_accuracy"]
    )
    m2["top_one_accuracy_improvement"] = float(
        ranking.loc["M2", "top_one_realised_candidate_accuracy"]
        - ranking.loc["M1", "top_one_realised_candidate_accuracy"]
    )
    monthly_m1 = _comparison_by_stratum(monthly_metrics, augmented="M1", baseline="M0")
    monthly_m2 = _comparison_by_stratum(monthly_metrics, augmented="M2", baseline="M1")
    checkpoint_m1 = _comparison_by_stratum(checkpoint_metrics, augmented="M1", baseline="M0")
    checkpoint_m2 = _comparison_by_stratum(checkpoint_metrics, augmented="M2", baseline="M1")
    m1_positive_months = int(
        (monthly_m1["brier_improvement"].gt(0.0) & monthly_m1["log_loss_improvement"].gt(0.0)).sum()
    )
    m2_positive_months = int(
        (monthly_m2["brier_improvement"].gt(0.0) & monthly_m2["log_loss_improvement"].gt(0.0)).sum()
    )

    def checkpoint_adverse(table: pd.DataFrame) -> bool:
        return bool(
            (table["brier_improvement"].lt(-0.001) | table["log_loss_improvement"].lt(-0.005)).any()
        )

    bootstrap_intervals = bootstrap_metrics.loc[
        bootstrap_metrics["record_type"].eq("interval")
        & bootstrap_metrics["interval_level"].eq(0.90)
    ].set_index(["comparison", "metric"])
    null_summary = null_metrics.loc[null_metrics["record_type"].eq("summary")].set_index(
        ["comparison", "metric"]
    )

    def comparison_gates(
        comparison: str,
        increments: Mapping[str, float],
        positive_months: int,
        checkpoints: pd.DataFrame,
    ) -> dict[str, bool]:
        return {
            "brier_improves": increments["brier_improvement"] > 0.0,
            "log_loss_improves": increments["log_loss_improvement"] > 0.0,
            "auc_not_reduced": increments["auc_increment"] >= 0.0,
            "bootstrap_90_brier_lower_non_negative": float(
                bootstrap_intervals.loc[(comparison, "brier_improvement"), "lower"]
            )
            >= 0.0,
            "bootstrap_90_log_loss_lower_non_negative": float(
                bootstrap_intervals.loc[(comparison, "log_loss_improvement"), "lower"]
            )
            >= 0.0,
            "positive_in_at_least_five_months": positive_months >= 5,
            "neither_checkpoint_materially_adverse": not checkpoint_adverse(checkpoints),
            "real_proper_increment_exceeds_null_90th_percentile": (
                increments["brier_improvement"]
                > float(null_summary.loc[(comparison, "brier_improvement"), "null_90th_percentile"])
                or increments["log_loss_improvement"]
                > float(
                    null_summary.loc[(comparison, "log_loss_improvement"), "null_90th_percentile"]
                )
            ),
            "concentration_gates_pass": bool(concentration["concentration_gates_passed"]),
        }

    m1_gates = comparison_gates("M1_minus_M0", m1, m1_positive_months, checkpoint_m1)
    m2_gates = comparison_gates("M2_minus_M1", m2, m2_positive_months, checkpoint_m2)
    m1_passes = all(m1_gates.values())
    m2_passes = all(m2_gates.values())
    m2_materially_adverse = bool(
        m2["brier_improvement"] < -0.001 or m2["log_loss_improvement"] < -0.005
    )
    decision = decide_screen(
        {
            "integrity_blocker": None,
            "m1_passes": m1_passes,
            "m2_passes": m2_passes,
            "m2_materially_adverse": m2_materially_adverse,
        }
    )
    return {
        **SAFETY_FLAGS,
        "decision": decision,
        "m1_passes": m1_passes,
        "m2_passes": m2_passes,
        "m2_materially_adverse": m2_materially_adverse,
        "m1_gates": m1_gates,
        "m2_gates": m2_gates,
        "m1_minus_m0": m1,
        "m2_minus_m1": m2,
        "m1_positive_month_count": m1_positive_months,
        "m2_positive_month_count": m2_positive_months,
        "m1_checkpoint_increments": checkpoint_m1.to_dict(orient="records"),
        "m2_checkpoint_increments": checkpoint_m2.to_dict(orient="records"),
        "development_support": dict(development_support),
        "assessment_support": dict(assessment_support),
        "concentration": dict(concentration),
        "top_one_is_secondary": True,
        "independent_audit_passed": False,
        "exact_rerun_passed": False,
    }


def _calibration_plot(assessment: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1.0)
    colors = {"M0": "#243b53", "M1": "#d97706", "M2": "#0f766e"}
    for model_id in ("M0", "M1", "M2"):
        probability = assessment[f"probability_{model_id}"].to_numpy(dtype=float)
        target = assessment[TARGET].to_numpy(dtype=float)
        weight = assessment["row_weight"].to_numpy(dtype=float)
        bins = np.minimum((np.clip(probability, 0.0, 1.0) * 10).astype(int), 9)
        x_values = []
        y_values = []
        for index in range(10):
            mask = bins == index
            if not mask.any():
                continue
            x_values.append(float(np.average(probability[mask], weights=weight[mask])))
            y_values.append(float(np.average(target[mask], weights=weight[mask])))
        axis.plot(x_values, y_values, marker="o", label=model_id, color=colors[model_id])
    axis.set(xlabel="Mean predicted probability", ylabel="Weighted observed frequency")
    axis.set_title("Assessment candidate calibration")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": "Stocker research"})
    plt.close(fig)


def _ranking_plot(ranking_metrics: pd.DataFrame, path: Path) -> None:
    selected = ranking_metrics.loc[
        ranking_metrics["population"].eq("multi_candidate_realised")
    ].set_index("model_id")
    labels = ["Top-one", "Mean reciprocal rank"]
    x = np.arange(len(labels))
    width = 0.22
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    colors = {"M0": "#243b53", "M1": "#d97706", "M2": "#0f766e"}
    for index, model_id in enumerate(("M0", "M1", "M2")):
        values = [
            selected.loc[model_id, "top_one_realised_candidate_accuracy"],
            selected.loc[model_id, "mean_reciprocal_rank"],
        ]
        axis.bar(x + (index - 1) * width, values, width, label=model_id, color=colors[model_id])
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Decision-level score")
    axis.set_title("Multi-candidate realised-loop ranking")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150, metadata={"Software": "Stocker research"})
    plt.close(fig)


def _report_text(
    decision: Mapping[str, Any],
    candidate_metrics: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    null_metrics: pd.DataFrame,
) -> str:
    pooled = candidate_metrics.set_index("model_id")
    ranking = ranking_metrics.loc[
        ranking_metrics["population"].eq("multi_candidate_realised")
    ].set_index("model_id")
    intervals = bootstrap_metrics.loc[bootstrap_metrics["record_type"].eq("interval")]
    nulls = null_metrics.loc[null_metrics["record_type"].eq("summary")]
    lines = [
        "# Regime × Loop Prefix × Behavioural Context Quick Screen V0",
        "",
        "This retrospective quick feasibility screen is research-only and structural-only. "
        "It evaluates pre-completion context, opens no economic outcome, and enables no execution.",
        "",
        f"Decision: `{decision['decision']}`.",
        "",
        "## Pooled assessment metrics",
        "",
        "| Model | Brier | Log loss | AUC | Top-one | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id in ("M0", "M1", "M2"):
        lines.append(
            f"| {model_id} | {pooled.loc[model_id, 'brier_score']:.8f} | "
            f"{pooled.loc[model_id, 'log_loss']:.8f} | {pooled.loc[model_id, 'auc']:.8f} | "
            f"{ranking.loc[model_id, 'top_one_realised_candidate_accuracy']:.8f} | "
            f"{ranking.loc[model_id, 'mean_reciprocal_rank']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Proper-score increments",
            "",
            f"M1−M0 Brier improvement: {decision['m1_minus_m0']['brier_improvement']:.10f}.",
            f"M1−M0 log-loss improvement: {decision['m1_minus_m0']['log_loss_improvement']:.10f}.",
            f"M2−M1 Brier improvement: {decision['m2_minus_m1']['brier_improvement']:.10f}.",
            f"M2−M1 log-loss improvement: {decision['m2_minus_m1']['log_loss_improvement']:.10f}.",
            "",
            "## Uncertainty and null",
            "",
            "```text",
            intervals.to_string(index=False),
            "```",
            "",
            "```text",
            nulls.to_string(index=False),
            "```",
            "",
            "Monthly, checkpoint, family, and prefix-maturity detail is in the adjacent "
            "CSV artifacts.",
            "",
            "No directional, economic, trading, or prospective claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def _feature_manifest() -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "target": TARGET,
        "all_predictors_available_at_completed_checkpoint_bar": True,
        "structural_numeric_features": list(STRUCTURAL_NUMERIC_FEATURES),
        "structural_categorical_features": list(STRUCTURAL_CATEGORICAL_FEATURES),
        "behavioural_features": list(BEHAVIOURAL_DIMENSIONS),
        "interaction_features": list(INTERACTION_FEATURES),
        "semantic_loop_identity_encoding": "one_hot_by_semantic_loop_id",
        "numeric_discovery_rank_used": False,
        "assessment_completion_rates_used": False,
        "orientation_sign_rule": (
            "sign of mean frozen raw signed_efficiency_6 and signed_efficiency_12 "
            "centroids for the next required state; zero maps to +1"
        ),
        "repeat_depth_rule": "zero for primitive and non-repeat composite candidates",
        "checkpoint_indicator": "one for ordinal 12 and zero for ordinal 6",
        "remaining_session_bars_rule": "78 - repo_bar_start_ordinal - 1",
        "prefix_matched_length_unit": "causal compressed-state transitions",
        "prefix_completion_fraction_rule": "matched_transitions / candidate_path_transitions",
    }


def _interaction_manifest(bounds: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    formulas = {
        "orientation_pressure_alignment": "candidate_orientation_sign * signed_pressure",
        "prefix_conviction": "prefix_completion_fraction * conviction",
        "transition_arousal": "current_transition_probability * arousal",
        "repeat_tension": "repeat_depth * tension",
        "next_leg_exhaustion_alignment": (
            "probability_of_next_required_state * candidate_orientation_sign * signed_exhaustion"
        ),
    }
    return {
        **SAFETY_FLAGS,
        "interaction_count": 5,
        "formulas": formulas,
        "clipping_fit_period": "2024_development_only",
        "clipping_percentiles": [0.01, 0.99],
        "clipping_bounds": {
            key: {"p01": float(value[0]), "p99": float(value[1])} for key, value in bounds.items()
        },
        "additional_interactions_created": False,
    }


def _candidate_population_artifact(candidates: pd.DataFrame) -> pd.DataFrame:
    exclusions = {
        TARGET,
        "primary_structural_outcome",
        "bars_until_first_event",
        "state_events_through_horizon",
        "state_event_bars_through_horizon",
        "primary_scoring_eligible",
        "exclusion_reason",
    }
    return candidates.loc[
        :, [column for column in candidates.columns if column not in exclusions]
    ].copy()


def _candidate_outcomes_artifact(candidate_outcomes: pd.DataFrame) -> pd.DataFrame:
    requested = [
        "candidate_id",
        "decision_id",
        "symbol",
        "session",
        "decision_ordinal",
        "semantic_loop_id",
        "candidate_orientation",
        "prefix_start_event_index",
        "prefix_matched_length",
        "decision_event_index",
        "state_events_through_horizon",
        "state_event_bars_through_horizon",
        "primary_structural_outcome",
        "bars_until_first_event",
        "primary_scoring_eligible",
        TARGET,
        "exclusion_reason",
    ]
    return candidate_outcomes.loc[
        :, [column for column in requested if column in candidate_outcomes.columns]
    ].copy()


def _base_source_manifest(structural_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "provider": "EODHD_local_processed_5m",
        "raw_data_downloaded": False,
        "one_minute_data_read": False,
        "development_start": "2024-01-01",
        "development_end_inclusive": "2024-12-31",
        "assessment_start": "2025-01-01",
        "assessment_end_inclusive": "2025-08-22",
        "protected_start": "2025-08-23",
        "minimum_timestamp_read": structural_manifest["minimum_timestamp_read"],
        "maximum_timestamp_read": structural_manifest["maximum_timestamp_read"],
        "protected_rows_materialised": structural_manifest["protected_rows_materialised"],
        "date_predicate_applied_before_materialisation": True,
        "decision_symbols": list(DECISION_SYMBOLS),
        "regime_context_symbols": list(REGIME_CONTEXT_SYMBOLS),
        "benchmark_symbol": "VTI",
        "decision_ordinals": [6, 12],
        "state_model_hash": EXPECTED_MODEL_HASH,
        "development_panel_hash": EXPECTED_PANEL_HASH,
        "semantic_dictionary_hash": EXPECTED_DICTIONARY_HASH,
        "structural_panel": dict(structural_manifest),
        "behavioural_predecessor_logical_root": str(BEHAVIOURAL_RELATIVE),
        "opening_predecessor_logical_path": str(OPENING_RELATIVE),
    }


def _protected_audit(
    candidates: pd.DataFrame, structural_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    protected_candidate_rows = int(
        pd.to_datetime(candidates["decision_timestamp"], utc=True).ge(PROTECTED_START).sum()
    )
    return {
        **SAFETY_FLAGS,
        "protected_start": str(PROTECTED_START),
        "maximum_timestamp_read": structural_manifest["maximum_timestamp_read"],
        "protected_rows_materialised": structural_manifest["protected_rows_materialised"],
        "candidate_rows_on_or_after_protected_start": protected_candidate_rows,
        "assessment_rows_after_end": protected_candidate_rows,
        "passed": protected_candidate_rows == 0
        and int(structural_manifest["protected_rows_materialised"]) == 0,
    }


def _blocked_metric_table(name: str) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "status": "not_run_due_insufficient_active_prefix_support",
                "artifact": name,
            }
        ]
    )


def _write_support_blocked_run(
    artifact_dir: Path,
    *,
    contract: Mapping[str, Any],
    candidates: pd.DataFrame,
    candidate_outcomes: pd.DataFrame,
    behavioural_audit: Mapping[str, Any],
    prefix_audit: Mapping[str, Any],
    structural_manifest: Mapping[str, Any],
    input_rows: Sequence[Mapping[str, Any]],
    development_support: Mapping[str, Any],
    assessment_support: Mapping[str, Any],
    concentration_table: pd.DataFrame,
) -> dict[str, Any]:
    failed = list(assessment_support["failed_support_gates"])
    detail = f"assessment active-prefix support gates failed: {failed}"
    decision = {
        **SAFETY_FLAGS,
        "decision": "blocked_insufficient_active_prefix_support",
        "blocker_detail": detail,
        "development_support": dict(development_support),
        "assessment_support": dict(assessment_support),
        "primary_models_fitted": 0,
        "bootstrap_draws_run": 0,
        "null_draws_run": 0,
        "plots_created": 0,
        "independent_audit_passed": False,
        "exact_rerun_passed": False,
    }
    bounds_payload = prefix_audit["interaction_clipping_bounds"]
    bounds = {
        key: (float(value["p01"]), float(value["p99"])) for key, value in bounds_payload.items()
    }
    model_configuration = {
        **SAFETY_FLAGS,
        "status": "not_fit_due_insufficient_active_prefix_support",
        "primary_model_count_fitted": 0,
        "registered_primary_model_count": 3,
        "fit_period": "2024_only_if_support_passes",
        "target": TARGET,
        "registered_models": {
            "M0": list(STRUCTURAL_NUMERIC_FEATURES),
            "M1": [*STRUCTURAL_NUMERIC_FEATURES, *BEHAVIOURAL_DIMENSIONS],
            "M2": [
                *STRUCTURAL_NUMERIC_FEATURES,
                *BEHAVIOURAL_DIMENSIONS,
                *INTERACTION_FEATURES,
            ],
        },
    }
    coefficients = {
        **SAFETY_FLAGS,
        "status": "not_fit_due_insufficient_active_prefix_support",
        "models": {},
    }
    source_manifest = _base_source_manifest(structural_manifest)
    input_hashes = {
        **SAFETY_FLAGS,
        "inputs": sorted(
            [dict(row) for row in input_rows], key=lambda row: str(row["logical_path"])
        ),
    }
    write_json(artifact_dir / "contract.json", contract)
    write_json(artifact_dir / "source_manifest.json", source_manifest)
    write_json(artifact_dir / "input_artifact_hashes.json", input_hashes)
    write_json(
        artifact_dir / "protected_boundary_audit.json",
        _protected_audit(candidates, structural_manifest),
    )
    write_json(artifact_dir / "behavioural_ledger_reconstruction.json", behavioural_audit)
    write_json(artifact_dir / "v2_prefix_population_reconstruction.json", prefix_audit)
    write_json(artifact_dir / "candidate_feature_manifest.json", _feature_manifest())
    write_json(artifact_dir / "interaction_manifest.json", _interaction_manifest(bounds))
    write_json(artifact_dir / "model_configurations.json", model_configuration)
    write_json(artifact_dir / "model_coefficients.json", coefficients)
    write_json(artifact_dir / "decision.json", decision)
    _write_parquet(
        _candidate_population_artifact(candidates),
        artifact_dir / "candidate_population.parquet",
    )
    _write_parquet(
        _candidate_outcomes_artifact(candidate_outcomes),
        artifact_dir / "candidate_outcomes.parquet",
    )
    assessment = candidates.loc[candidates["year"].eq(2025)].copy()
    assessment_predictions = assessment.loc[
        :,
        [
            "candidate_id",
            "decision_id",
            "slate_id",
            "symbol",
            "session",
            "year_month",
            "decision_ordinal",
            "semantic_loop_id",
            "candidate_orientation",
            "candidate_class",
            "prefix_maturity",
            "row_weight",
            TARGET,
        ],
    ].copy()
    assessment_predictions["status"] = "not_scored_due_insufficient_active_prefix_support"
    _write_parquet(assessment_predictions, artifact_dir / "assessment_predictions.parquet")
    for filename in (
        "candidate_metrics.csv",
        "decision_ranking_metrics.csv",
        "monthly_metrics.csv",
        "checkpoint_metrics.csv",
        "candidate_family_metrics.csv",
        "prefix_maturity_metrics.csv",
        "bootstrap_metrics.csv",
        "null_metrics.csv",
    ):
        _write_csv(_blocked_metric_table(filename), artifact_dir / filename)
    _write_csv(concentration_table, artifact_dir / "concentration_metrics.csv")
    development_text = json.dumps(_json_value(dict(development_support)), sort_keys=True)
    assessment_text = json.dumps(_json_value(dict(assessment_support)), sort_keys=True)
    report = "\n".join(
        [
            "# Regime × Loop Prefix × Behavioural Context Quick Screen V0",
            "",
            "This retrospective, research-only structural feasibility screen stopped at its "
            "preregistered assessment support gate.",
            "",
            "Decision: `blocked_insufficient_active_prefix_support`.",
            "",
            f"Failed gates: {', '.join(failed)}.",
            "",
            f"Development support: `{development_text}`",
            "",
            f"Assessment support: `{assessment_text}`",
            "",
            "M0–M2, bootstrap uncertainty, and behavioural nulls were not run after the "
            "fail-closed gate. No directional, economic, trading, or prospective claim is made.",
            "",
        ]
    )
    (artifact_dir / "report.md").write_text(report, encoding="utf-8")
    return decision


def execute_run(
    artifact_dir: Path,
    *,
    provider_root: Path,
    materialized_repo: Path,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if any(contract.get(key) != value for key, value in SAFETY_FLAGS.items()):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            "root contract safety flags differ from the mandatory boundary",
        )
    write_json(artifact_dir / "contract.json", contract)
    behavioural, behavioural_audit, behavioural_hashes = _load_frozen_behavioural(materialized_repo)
    preprocessing, parameters, dictionary, orientation_sign, structural_hashes = (
        _load_structural_inputs()
    )
    state_panel, structural_manifest = _build_structural_panel(
        provider_root, preprocessing, parameters
    )
    opening_audit, opening_identity = _validate_against_opening_ledger(
        state_panel, materialized_repo
    )
    candidates, candidate_outcomes, prefix_audit = _build_candidate_population(
        behavioural, state_panel, dictionary, orientation_sign
    )
    prefix_audit["frozen_opening_checkpoint_reconstruction"] = opening_audit
    development_support, assessment_support, concentration_table, concentration = _enforce_support(
        candidates
    )
    all_input_rows = [*behavioural_hashes, *structural_hashes, opening_identity]
    if not bool(assessment_support["all_support_gates_passed"]):
        return _write_support_blocked_run(
            artifact_dir,
            contract=contract,
            candidates=candidates,
            candidate_outcomes=candidate_outcomes,
            behavioural_audit=behavioural_audit,
            prefix_audit=prefix_audit,
            structural_manifest=structural_manifest,
            input_rows=all_input_rows,
            development_support=development_support,
            assessment_support=assessment_support,
            concentration_table=concentration_table,
        )
    models, assessment, model_payload = _fit_primary_models(candidates)
    (
        candidate_metrics,
        ranking_metrics,
        monthly_metrics,
        checkpoint_metrics,
        family_metrics,
        maturity_metrics,
        _,
    ) = _all_metric_tables(assessment)
    bootstrap_metrics = _bootstrap_metrics(assessment)
    null_metrics = _null_metrics(candidates, assessment)
    decision = _decision_payload(
        candidate_metrics,
        monthly_metrics,
        checkpoint_metrics,
        ranking_metrics,
        bootstrap_metrics,
        null_metrics,
        concentration,
        development_support,
        assessment_support,
    )

    bounds_payload = prefix_audit["interaction_clipping_bounds"]
    bounds = {
        key: (float(value["p01"]), float(value["p99"])) for key, value in bounds_payload.items()
    }
    feature_manifest = _feature_manifest()
    interaction_manifest = _interaction_manifest(bounds)
    source_manifest = _base_source_manifest(structural_manifest)
    input_hashes = {
        **SAFETY_FLAGS,
        "inputs": sorted(
            all_input_rows,
            key=lambda row: row["logical_path"],
        ),
    }
    protected_audit = _protected_audit(candidates, structural_manifest)

    population = _candidate_population_artifact(candidates)
    outcomes_artifact = _candidate_outcomes_artifact(candidate_outcomes)
    prediction_columns = [
        "candidate_id",
        "decision_id",
        "slate_id",
        "symbol",
        "session",
        "year_month",
        "decision_ordinal",
        "semantic_loop_id",
        "candidate_orientation",
        "candidate_class",
        "prefix_maturity",
        "row_weight",
        TARGET,
        "probability_M0",
        "probability_M1",
        "probability_M2",
    ]

    write_json(artifact_dir / "source_manifest.json", source_manifest)
    write_json(artifact_dir / "input_artifact_hashes.json", input_hashes)
    write_json(artifact_dir / "protected_boundary_audit.json", protected_audit)
    write_json(artifact_dir / "behavioural_ledger_reconstruction.json", behavioural_audit)
    write_json(artifact_dir / "v2_prefix_population_reconstruction.json", prefix_audit)
    write_json(artifact_dir / "candidate_feature_manifest.json", feature_manifest)
    write_json(artifact_dir / "interaction_manifest.json", interaction_manifest)
    write_json(artifact_dir / "model_configurations.json", model_payload["configurations"])
    write_json(artifact_dir / "model_coefficients.json", model_payload["coefficients"])
    write_json(artifact_dir / "decision.json", decision)
    _write_parquet(population, artifact_dir / "candidate_population.parquet")
    _write_parquet(
        outcomes_artifact,
        artifact_dir / "candidate_outcomes.parquet",
    )
    _write_parquet(
        assessment.loc[:, prediction_columns], artifact_dir / "assessment_predictions.parquet"
    )
    _write_csv(candidate_metrics, artifact_dir / "candidate_metrics.csv")
    _write_csv(ranking_metrics, artifact_dir / "decision_ranking_metrics.csv")
    _write_csv(monthly_metrics, artifact_dir / "monthly_metrics.csv")
    _write_csv(checkpoint_metrics, artifact_dir / "checkpoint_metrics.csv")
    _write_csv(family_metrics, artifact_dir / "candidate_family_metrics.csv")
    _write_csv(maturity_metrics, artifact_dir / "prefix_maturity_metrics.csv")
    _write_csv(bootstrap_metrics, artifact_dir / "bootstrap_metrics.csv")
    _write_csv(null_metrics, artifact_dir / "null_metrics.csv")
    _write_csv(concentration_table, artifact_dir / "concentration_metrics.csv")
    _calibration_plot(assessment, artifact_dir / "calibration_m0_m2.png")
    _ranking_plot(ranking_metrics, artifact_dir / "decision_ranking_m0_m2.png")
    report = _report_text(
        decision,
        candidate_metrics,
        ranking_metrics,
        bootstrap_metrics,
        null_metrics,
    )
    (artifact_dir / "report.md").write_text(report, encoding="utf-8")
    del models
    return decision


def run_independent_auditor(
    artifact_dir: Path,
    *,
    materialized_repo: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(AUDITOR_PATH),
            "--artifacts",
            str(artifact_dir),
            "--materialized-predecessor-repo",
            str(materialized_repo),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    if completed.returncode != 0:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            f"independent audit failed: {completed.stderr[-2000:]}",
        )
    audit = json.loads((artifact_dir / "independent_audit.json").read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            "independent audit did not pass",
        )
    return audit


def compare_exact_runs(primary: Path, exact: Path) -> dict[str, Any]:
    comparisons = []
    for name in SCIENTIFIC_ARTIFACTS:
        left = primary / name
        right = exact / name
        if not left.is_file() or not right.is_file():
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                f"exact-rerun artifact is absent: {name}",
            )
        left_hash = sha256_file(left)
        right_hash = sha256_file(right)
        comparisons.append(
            {
                "artifact": name,
                "primary_sha256": left_hash,
                "exact_rerun_sha256": right_hash,
                "identical": left_hash == right_hash,
            }
        )
    passed = all(row["identical"] for row in comparisons)
    manifest = {
        **SAFETY_FLAGS,
        "fixed_seeds": {"bootstrap": BOOTSTRAP_SEED, "null": NULL_SEED},
        "artifact_comparisons": comparisons,
        "all_scientific_artifacts_identical": passed,
        "passed": passed,
    }
    write_json(primary / "exact_rerun_manifest.json", manifest)
    write_json(exact / "exact_rerun_manifest.json", manifest)
    if not passed:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            "primary and exact-rerun scientific artifacts differ",
        )
    return manifest


def _mark_final_status(
    artifact_dir: Path,
    *,
    exact_passed: bool,
    audit_passed: bool,
) -> None:
    decision_path = artifact_dir / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["independent_audit_passed"] = audit_passed
    decision["exact_rerun_passed"] = exact_passed
    write_json(decision_path, decision)


def run_complete_screen(
    *,
    primary: Path,
    exact: Path,
    provider_root: Path,
    materialized_repo: Path,
) -> dict[str, Any]:
    execute_run(primary, provider_root=provider_root, materialized_repo=materialized_repo)
    run_independent_auditor(primary, materialized_repo=materialized_repo)
    execute_run(exact, provider_root=provider_root, materialized_repo=materialized_repo)
    run_independent_auditor(exact, materialized_repo=materialized_repo)
    _mark_final_status(primary, exact_passed=True, audit_passed=True)
    _mark_final_status(exact, exact_passed=True, audit_passed=True)
    compare_exact_runs(primary, exact)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(primary / "report.md", REPORTS_DIR / "report.md")
    return json.loads((primary / "decision.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--materialized-predecessor-repo",
        type=Path,
        default=REPO_ROOT,
    )
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--exact-rerun", type=Path, default=DEFAULT_EXACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        decision = run_complete_screen(
            primary=args.primary,
            exact=args.exact_rerun,
            provider_root=args.provider_root,
            materialized_repo=args.materialized_predecessor_repo,
        )
    except ScreenBlocker as blocker:
        payload = {**SAFETY_FLAGS, "decision": blocker.decision, "blocker_detail": blocker.detail}
        for directory in (args.primary, args.exact_rerun):
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "decision.json", payload)
        print(json.dumps(payload, sort_keys=True))
        return 2
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
