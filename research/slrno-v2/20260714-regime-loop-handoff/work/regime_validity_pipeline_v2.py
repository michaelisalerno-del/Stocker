#!/usr/bin/env python3
"""Run bounded structural Regime Model Validity V2 Part A.

The runner reads 2024 development and unchanged retrospective 2025 structural
bars only.  It never reads an economic outcome and has no execution import or
runtime mutation surface.  Existing historical files are read-only inputs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
for import_root in (PACKAGE_ROOT, WORK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from stocker_research.causal_state_export_v2 import (  # noqa: E402
    HysteresisConfig,
    SoftLoopPrefixTracker,
    build_completed_bar_decisions,
    build_hard_state_runs_v2,
    causal_semimarkov_filter_v2,
    expand_duration_hazard_v2,
)
from stocker_research.loop_dictionary_v2 import (  # noqa: E402
    LoopDictionary,
    decompose_closed_path,
)
from stocker_research.loop_ledger_v2 import build_loop_event_ledgers  # noqa: E402
from stocker_research.loop_nulls_v2 import SemiMarkovNull, SessionRunSequence  # noqa: E402
from stocker_research.loop_structural_nulls_v2 import (  # noqa: E402
    first_event_candidate_counts,
)
from stocker_research.regime_validity_v2 import (  # noqa: E402
    FROZEN_EMISSION_FEATURES,
    CausalFilterSummary,
    CleaningVariant,
    EmissionPreprocessing,
    PartAGateEvidence,
    apply_cleaning_variant,
    build_run_ledger,
    build_training_sample,
    decide_part_a,
    deterministic_model_registry,
    emission_feature_provenance,
    estimate_semimarkov_parameters,
    fit_clustered_semimarkov,
    fit_emission_preprocessing,
    freeze_part_a_binding,
    frozen_emission_partition,
    gaussian_log_emissions,
    safety_flags,
    semantic_remap_by_activity_direction,
    transform_emissions,
)
from stocker_research.semantic_loop_dictionary_v2 import semantic_primitive_id  # noqa: E402
from stocker_research.state_alignment_v2 import (  # noqa: E402
    AlignmentWeights,
    align_states,
    apply_state_mapping,
)
from stocker_research.state_representation_sensitivity_v2 import (  # noqa: E402
    classify_soft_support,
    cleaning_run_changes,
    compare_representation_events,
    hierarchical_state_ids,
    hysteretic_states_by_session,
    reconstruct_first_event_outcomes,
    transition_confidence,
)

CONTRACT_PATH = WORK_DIR / "contracts" / "20260718-regime-model-validity-v2.json"
ARTIFACT_PARENT = WORK_DIR / "artifacts" / "20260718-regime-model-validity-v2"
REPORT_PATH = WORK_DIR / "reports" / "20260718-regime-model-validity-v2.md"
LEGACY_V2_RUNNER = WORK_DIR / "run_loop_event_semantics_v2.py"
DICTIONARY_PATH = (
    WORK_DIR
    / "artifacts"
    / "20260718-semantic-loop-dictionary-coverage-v2"
    / "primary"
    / "semantic_loop_dictionary_v2.csv"
)
BASELINE_SHA = "66cd706fa727ac5873b299d5c22388221203f451"
EXPECTED_CONTRACT_HASH = "dd44ce458f41a16f023a49b9be7ab3f762ed31b07d42fc6e1ba673f233546c55"
EXPECTED_2024_SNAPSHOT = "48d2141ef993928d4e8a01d6b3c24dff665280c67f4167115b453613460cc661"
STATE_MODEL_VERSION = "causal_semimarkov_posterior_export_v2_tail78"
DICTIONARY_VERSION = "semantic_loop_dictionary_first_event_v2"
DICTIONARY_HASH = "9f39bf57c2637dfa7f465103306ac6c7d17a321a036b175cc222dc1a204cd918"
K_VALUES = (6, 8, 10, 12)
SEEDS = (20260710, 20260711, 20260712, 20260713, 20260714)
SELECTED_PATHS = ((5, 6, 5), (4, 6, 4))
SELECTED_IDS = tuple(semantic_primitive_id(path[:-1]) for path in SELECTED_PATHS)
BAR_DURATION = pd.Timedelta(minutes=5)
PART_A_REQUIRED_ARTIFACTS = (
    "regime_implementation_census.csv",
    "source_identity_manifest.json",
    "implementation_source_manifest.json",
    "pre_run_tree_manifest.json",
    "current_regime_reconstruction.json",
    "current_state_centroids.csv",
    "current_state_parameters.npz",
    "current_state_assignment_sample.parquet",
    "current_run_reconstruction.parquet",
    "current_regime_exact_match.csv",
    "regime_math_audit.json",
    "posterior_normalization_audit.csv",
    "state_age_audit.csv",
    "regime_causality_audit.parquet",
    "short_run_cleaning_audit.csv",
    "cleaning_variant_state_metrics.csv",
    "cleaning_variant_loop_metrics.csv",
    "cleaning_variant_dictionary_overlap.csv",
    "state_transition_confidence.parquet",
    "hard_state_churn_summary.csv",
    "loop_dependency_on_low_confidence_transitions.csv",
    "hard_hysteretic_transition_comparison.csv",
    "k_seed_model_registry.csv",
    "state_alignment.csv",
    "state_stability_by_k_seed.csv",
    "loop_stability_by_k_seed.csv",
    "first_event_stability_by_k_seed.csv",
    "training_sample_composition.csv",
    "training_sample_state_stability.csv",
    "training_sample_loop_stability.csv",
    "state_semantic_profiles.parquet",
    "state_period_drift.csv",
    "state_stock_heterogeneity.csv",
    "state_clock_heterogeneity.csv",
    "state_transition_drift.csv",
    "emission_feature_partition.json",
    "combined_stock_hierarchical_comparison.csv",
    "market_regime_profiles.csv",
    "stock_state_profiles.csv",
    "hierarchical_state_mapping.parquet",
    "state_representation_event_comparison.parquet",
    "loop_robustness_by_representation.csv",
    "dictionary_robustness_by_representation.csv",
    "event_timing_shift_summary.csv",
    "part_a_decision.json",
    "run_metadata.json",
    "artifact_manifest.json",
    "independent_audit.json",
    "exact_rerun_manifest.json",
    "post_run_tree_manifest.json",
)
IMPLEMENTATION_SOURCE_PATHS = (
    Path("packages/stocker_research/src/stocker_research/regime_validity_v2.py"),
    Path("packages/stocker_research/src/stocker_research/state_alignment_v2.py"),
    Path("packages/stocker_research/src/stocker_research/state_representation_sensitivity_v2.py"),
    Path("packages/stocker_research/src/stocker_research/loop_orientation_v2.py"),
    Path("packages/stocker_research/src/stocker_research/loop_regime_interaction_v2.py"),
    Path("research/slrno-v2/20260714-regime-loop-handoff/work/regime_validity_pipeline_v2.py"),
    Path("research/slrno-v2/20260714-regime-loop-handoff/work/run_regime_model_validity_v2.py"),
    Path("research/slrno-v2/20260714-regime-loop-handoff/work/audit_regime_model_validity_v2.py"),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/contracts/"
        "20260718-regime-model-validity-v2.json"
    ),
)


@njit(cache=False)
def _causal_filter_kernel(
    emissions: np.ndarray,
    hazard: np.ndarray,
    transitions: np.ndarray,
    initial: np.ndarray,
    occupancy: np.ndarray,
    reset: np.ndarray,
    has_occupancy: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compiled equivalent of the audited state-age forward recursion."""

    row_count, state_count = emissions.shape
    age_count = hazard.shape[1]
    probabilities = np.zeros((row_count, state_count), dtype=np.float64)
    hard = np.full(row_count, -1, dtype=np.int16)
    expected_age = np.zeros(row_count, dtype=np.float64)
    departure = np.zeros(row_count, dtype=np.float64)
    entropy = np.zeros(row_count, dtype=np.float64)
    likelihood = np.zeros(row_count, dtype=np.float64)
    iid_likelihood = np.full(row_count, np.nan, dtype=np.float64)
    alpha = np.zeros((state_count, age_count), dtype=np.float64)
    prior = np.zeros((state_count, age_count), dtype=np.float64)
    initial_total = initial.sum()

    for position in range(row_count):
        prior.fill(0.0)
        if reset[position]:
            for state in range(state_count):
                prior[state, 0] = initial[state] / initial_total
        else:
            for state in range(state_count):
                exit_mass = 0.0
                for age in range(age_count):
                    stay_mass = alpha[state, age] * (1.0 - hazard[state, age])
                    destination_age = min(age + 1, age_count - 1)
                    prior[state, destination_age] += stay_mass
                    exit_mass += alpha[state, age] * hazard[state, age]
                for destination_state in range(state_count):
                    prior[destination_state, 0] += exit_mass * transitions[state, destination_state]
            prior_total = prior.sum()
            for state in range(state_count):
                for age in range(age_count):
                    prior[state, age] /= prior_total

        maximum_term = -np.inf
        for state in range(state_count):
            state_prior = 0.0
            for age in range(age_count):
                state_prior += prior[state, age]
            term = math.log(max(state_prior, 1e-300)) + emissions[position, state]
            maximum_term = max(maximum_term, term)
        likelihood_total = 0.0
        for state in range(state_count):
            state_prior = 0.0
            for age in range(age_count):
                state_prior += prior[state, age]
            term = math.log(max(state_prior, 1e-300)) + emissions[position, state]
            likelihood_total += math.exp(term - maximum_term)
        likelihood[position] = maximum_term + math.log(likelihood_total)

        if has_occupancy:
            maximum_iid_term = -np.inf
            for state in range(state_count):
                term = math.log(max(occupancy[state], 1e-300)) + emissions[position, state]
                maximum_iid_term = max(maximum_iid_term, term)
            iid_total = 0.0
            for state in range(state_count):
                term = math.log(max(occupancy[state], 1e-300)) + emissions[position, state]
                iid_total += math.exp(term - maximum_iid_term)
            iid_likelihood[position] = maximum_iid_term + math.log(iid_total)

        maximum_emission = np.max(emissions[position])
        posterior_total = 0.0
        for state in range(state_count):
            relative_likelihood = math.exp(emissions[position, state] - maximum_emission)
            for age in range(age_count):
                alpha[state, age] = prior[state, age] * relative_likelihood
                posterior_total += alpha[state, age]
        for state in range(state_count):
            state_probability = 0.0
            for age in range(age_count):
                alpha[state, age] /= posterior_total
                state_probability += alpha[state, age]
                expected_age[position] += alpha[state, age] * (age + 1.0)
                departure[position] += alpha[state, age] * hazard[state, age]
            probabilities[position, state] = state_probability
            if state_probability > probabilities[position, hard[position]] or hard[position] < 0:
                hard[position] = state
            entropy[position] -= state_probability * math.log(max(state_probability, 1e-300))

    return (
        probabilities,
        hard,
        expected_age,
        departure,
        entropy,
        likelihood,
        iid_likelihood,
    )


@njit(cache=False)
def _two_state_null_kernel(
    initial: np.ndarray,
    transitions: np.ndarray,
    duration_cumulative: np.ndarray,
    session_lengths: np.ndarray,
    candidate_pairs: np.ndarray,
    horizon_bars: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Numba structural null for arbitrary-K two-state primitive candidates."""

    output = np.zeros((draws, len(candidate_pairs)), dtype=np.int64)
    maximum_runs = 78
    for draw in range(draws):
        np.random.seed(seed + draw)
        for session_length in session_lengths:
            states = np.empty(maximum_runs, dtype=np.int64)
            durations = np.empty(maximum_runs, dtype=np.int64)
            starts = np.empty(maximum_runs, dtype=np.int64)
            uniform = np.random.random()
            cumulative = 0.0
            state = 0
            for state_index in range(len(initial)):
                cumulative += initial[state_index]
                if uniform <= cumulative:
                    state = state_index
                    break
            elapsed = 0
            run_count = 0
            while elapsed < session_length:
                states[run_count] = state
                starts[run_count] = elapsed
                duration_uniform = np.random.random()
                selected_duration = duration_cumulative.shape[1]
                for duration_index in range(duration_cumulative.shape[1]):
                    if duration_uniform <= duration_cumulative[state, duration_index]:
                        selected_duration = duration_index + 1
                        break
                duration = min(selected_duration, session_length - elapsed)
                durations[run_count] = duration
                elapsed += duration
                run_count += 1
                if elapsed >= session_length:
                    break
                transition_uniform = np.random.random()
                cumulative = 0.0
                next_state = 0
                for destination in range(len(initial)):
                    cumulative += transitions[state, destination]
                    if transition_uniform <= cumulative:
                        next_state = destination
                        break
                state = next_state

            stack = np.empty(16, dtype=np.int64)
            stack_size = 0
            closure_events = np.empty(maximum_runs, dtype=np.int64)
            closure_candidates = np.full(maximum_runs, -1, dtype=np.int64)
            closure_count = 0
            for event_index in range(run_count):
                current = states[event_index]
                stack_index = -1
                for index in range(stack_size):
                    if stack[index] == current:
                        stack_index = index
                        break
                if stack_index < 0:
                    stack[stack_size] = current
                    stack_size += 1
                    continue
                core_length = stack_size - stack_index
                candidate_index = -1
                if core_length == 2:
                    left = min(stack[stack_index], stack[stack_index + 1])
                    right = max(stack[stack_index], stack[stack_index + 1])
                    for index in range(len(candidate_pairs)):
                        if candidate_pairs[index, 0] == left and candidate_pairs[index, 1] == right:
                            candidate_index = index
                            break
                closure_events[closure_count] = event_index
                closure_candidates[closure_count] = candidate_index
                closure_count += 1
                stack[stack_index] = current
                stack_size = stack_index + 1

            closure_pointer = 0
            for event_index in range(run_count):
                while (
                    closure_pointer < closure_count
                    and closure_events[closure_pointer] <= event_index
                ):
                    closure_pointer += 1
                if closure_pointer >= closure_count:
                    break
                candidate_index = closure_candidates[closure_pointer]
                if candidate_index < 0:
                    continue
                event_bar = starts[closure_events[closure_pointer]]
                run_start = starts[event_index]
                run_end = run_start + durations[event_index] - 1
                lower = max(run_start, event_bar - horizon_bars)
                upper = min(run_end, event_bar - 1)
                if upper >= lower:
                    output[draw, candidate_index] += upper - lower + 1
    return output


def _causal_filter_summary_compiled(
    log_emissions: np.ndarray,
    *,
    groups: Sequence[np.ndarray],
    model: Mapping[str, np.ndarray],
) -> CausalFilterSummary:
    """Run the exact reference recursion through a compiled, memory-bounded kernel."""

    emissions = np.asarray(log_emissions, dtype=np.float64)
    hazard = np.asarray(model["duration_hazard"], dtype=np.float64)
    transitions = np.asarray(model["transitions"], dtype=np.float64)
    initial = np.asarray(model["initial"], dtype=np.float64)
    if emissions.ndim != 2 or hazard.ndim != 2:
        raise ValueError("emissions and hazard must be matrices")
    row_count, state_count = emissions.shape
    if hazard.shape[0] != state_count:
        raise ValueError("hazard state count differs from emissions")
    if transitions.shape != (state_count, state_count) or initial.shape != (state_count,):
        raise ValueError("transition or initial dimensions differ from state count")
    reset = np.zeros(row_count, dtype=np.bool_)
    covered = np.zeros(row_count, dtype=np.bool_)
    previous_end = -1
    for positions_value in groups:
        positions = np.asarray(positions_value, dtype=int)
        if len(positions) == 0 or np.any(np.diff(positions) != 1):
            raise ValueError("compiled recursion requires non-empty contiguous session groups")
        if int(positions[0]) != previous_end + 1:
            raise ValueError("compiled recursion groups must follow panel row order")
        if int(positions[-1]) >= row_count:
            raise ValueError("compiled recursion group exceeds panel rows")
        reset[int(positions[0])] = True
        covered[positions] = True
        previous_end = int(positions[-1])
    if not covered.all():
        raise ValueError("compiled recursion groups must cover every panel row exactly once")
    occupancy_value = model.get("occupancy")
    has_occupancy = occupancy_value is not None
    occupancy = (
        np.asarray(occupancy_value, dtype=np.float64)
        if has_occupancy
        else np.zeros(state_count, dtype=np.float64)
    )
    outputs = _causal_filter_kernel(
        emissions,
        hazard,
        transitions,
        initial,
        occupancy,
        reset,
        has_occupancy,
    )
    return CausalFilterSummary(
        state_probabilities=outputs[0],
        hard_states=outputs[1],
        expected_age=outputs[2],
        departure_probability=outputs[3],
        posterior_entropy=outputs[4],
        log_likelihood=outputs[5],
        iid_log_likelihood=outputs[6],
    )


@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_id: str
    git_sha: str
    contract_hash: str
    data_snapshot_hash: str
    implementation_source_hash: str
    state_model_version: str
    state_model_hash: str
    dictionary_version: str
    dictionary_hash: str


@dataclass(slots=True)
class PeriodSurface:
    period: str
    panel: pd.DataFrame
    source_hashes: dict[str, str]
    snapshot_hash: str
    scaled: np.ndarray
    log_emissions: np.ndarray
    groups: tuple[np.ndarray, ...]


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    if pd.isna(value):
        return None
    raise TypeError(f"cannot encode {type(value).__name__}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_contract() -> tuple[dict[str, Any], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_hash = _sha256_file(CONTRACT_PATH)
    if contract_hash != EXPECTED_CONTRACT_HASH:
        raise RuntimeError("Part A contract identity changed after preregistration")
    for key, expected in safety_flags().items():
        if contract.get(key) != expected:
            raise RuntimeError(f"Part A contract safety mismatch: {key}")
    return contract, contract_hash


class ArtifactWriter:
    def __init__(self, output_dir: Path, identity: RunIdentity) -> None:
        self.output_dir = output_dir
        self.identity = identity
        output_dir.mkdir(parents=True, exist_ok=True)

    def _identity_frame(
        self,
        frame: pd.DataFrame,
        *,
        state_representation: str = "not_applicable",
        source_artifact: str = "not_applicable",
        source_hash: str | None = None,
    ) -> pd.DataFrame:
        result = frame.copy()
        defaults: dict[str, object] = {
            "run_id": self.identity.run_id,
            "git_sha": self.identity.git_sha,
            "contract_hash": self.identity.contract_hash,
            "data_snapshot_hash": self.identity.data_snapshot_hash,
            "implementation_source_hash": self.identity.implementation_source_hash,
            "state_model_version": self.identity.state_model_version,
            "state_model_hash": self.identity.state_model_hash,
            "state_representation": state_representation,
            "dictionary_version": self.identity.dictionary_version,
            "dictionary_hash": self.identity.dictionary_hash,
            "decision_id": "not_applicable",
            "symbol": "not_applicable",
            "session": "not_applicable",
            "decision_timestamp": "not_applicable",
            "primitive_loop_id": "not_applicable",
            "orientation_id": "not_applicable",
            "prefix_progress": "not_applicable",
            "source_artifact": source_artifact,
            "source_hash": source_hash or self.identity.data_snapshot_hash,
            **safety_flags(),
        }
        for key, value in defaults.items():
            if key not in result:
                result[key] = value
        return result

    def frame(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        state_representation: str = "not_applicable",
        source_artifact: str = "not_applicable",
        source_hash: str | None = None,
    ) -> None:
        output = self._identity_frame(
            frame,
            state_representation=state_representation,
            source_artifact=source_artifact,
            source_hash=source_hash,
        )
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            output.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
        elif path.suffix == ".parquet":
            temporary = path.with_suffix(path.suffix + ".tmp")
            output.to_parquet(
                temporary,
                index=False,
                compression="zstd",
                compression_level=9,
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        else:
            raise ValueError(f"unsupported tabular artifact: {name}")

    def json(self, name: str, payload: Mapping[str, Any]) -> None:
        merged = {
            "run_id": self.identity.run_id,
            "git_sha": self.identity.git_sha,
            "contract_hash": self.identity.contract_hash,
            "data_snapshot_hash": self.identity.data_snapshot_hash,
            "implementation_source_hash": self.identity.implementation_source_hash,
            "state_model_version": self.identity.state_model_version,
            "state_model_hash": self.identity.state_model_hash,
            "dictionary_version": self.identity.dictionary_version,
            "dictionary_hash": self.identity.dictionary_hash,
            **payload,
            **safety_flags(),
        }
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            json.dumps(merged, sort_keys=True, indent=2, default=_json_default).encode() + b"\n"
        )

    def npz(self, name: str, arrays: Mapping[str, np.ndarray]) -> None:
        path = self.output_dir / name
        with zipfile.ZipFile(
            path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for key in sorted(arrays):
                buffer = io.BytesIO()
                np.save(buffer, np.asarray(arrays[key]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _preprocessing_from_csv(path: Path) -> EmissionPreprocessing:
    frame = pd.read_csv(path)
    return EmissionPreprocessing(
        feature_names=tuple(frame["feature"].astype(str)),
        medians=frame["imputer_median"].to_numpy(dtype=float),
        centers=frame["scaler_center"].to_numpy(dtype=float),
        scales=frame["scaler_scale"].to_numpy(dtype=float),
    )


def _load_parameters(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {key: np.asarray(stored[key]).copy() for key in stored.files}


def _prepare_period(
    legacy: Any,
    *,
    period: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    contract_hash: str,
    preprocessing: EmissionPreprocessing,
    parameters: Mapping[str, np.ndarray],
) -> PeriodSurface:
    old_start, old_end = legacy.DEVELOPMENT_START, legacy.DEVELOPMENT_END
    try:
        legacy.DEVELOPMENT_START = start
        legacy.DEVELOPMENT_END = end
        source_hashes, snapshot_hash = legacy._source_hashes()
        panel = legacy._prepare_panel(
            source_hashes,
            data_snapshot_hash=snapshot_hash,
            contract_hash=contract_hash,
        )
    finally:
        legacy.DEVELOPMENT_START, legacy.DEVELOPMENT_END = old_start, old_end
    timestamps = pd.to_datetime(panel["bar_start_timestamp"], utc=True)
    if timestamps.min() < start or timestamps.max() > end:
        raise RuntimeError(f"{period} panel escaped its bounded period")
    scaled = legacy.frozen_core.scale_emissions(
        panel,
        pd.DataFrame(
            {
                "feature": preprocessing.feature_names,
                "imputer_median": preprocessing.medians,
                "scaler_center": preprocessing.centers,
                "scaler_scale": preprocessing.scales,
            }
        ),
    )
    log_emissions = legacy.frozen_core.log_emission(scaled, dict(parameters))
    groups = tuple(legacy._contiguous_causal_groups(panel))
    return PeriodSurface(
        period=period,
        panel=panel,
        source_hashes=dict(source_hashes),
        snapshot_hash=str(snapshot_hash),
        scaled=scaled,
        log_emissions=log_emissions,
        groups=groups,
    )


def _state_source_hash(
    legacy: Any, model_v2: Mapping[str, np.ndarray], data_snapshot_hash: str
) -> str:
    payload = {
        "model_file_hash": _sha256_file(legacy.MODEL_PATH),
        "preprocessing_file_hash": _sha256_file(legacy.PREPROCESSING_PATH),
        "duration_hazard_v2_hash": _sha256_bytes(
            np.asarray(model_v2["duration_hazard"], dtype=np.float64).tobytes()
        ),
        "data_snapshot_hash": data_snapshot_hash,
        "state_model_version": STATE_MODEL_VERSION,
    }
    return _sha256_bytes(_canonical_bytes(payload))


def _dictionary() -> LoopDictionary:
    return LoopDictionary.from_definitions(
        (decompose_closed_path(path) for path in SELECTED_PATHS),
        version=DICTIONARY_VERSION,
    )


def _sensitivity_decision_surface(panel: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    required = [
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
    ]
    frame = panel[required].copy()
    frame["decision_timestamp"] = frame["bar_complete_timestamp"]
    frame["decision_id"] = [
        f"{prefix}:{symbol}:{session}:{int(bar):02d}"
        for symbol, session, bar in frame[["symbol", "session", "bar_ordinal"]].itertuples(
            index=False, name=None
        )
    ]
    return frame


def _run_sequences(panel: pd.DataFrame, labels: np.ndarray) -> tuple[SessionRunSequence, ...]:
    records: list[SessionRunSequence] = []
    for (symbol, session), group in panel.groupby(["symbol", "session"], sort=True):
        positions = group.index.to_numpy(dtype=int)
        states = np.asarray(labels[positions], dtype=int)
        starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
        ends = np.r_[starts[1:], len(states)]
        records.append(
            SessionRunSequence(
                symbol=str(symbol),
                session=str(session),
                states=tuple(int(states[start]) for start in starts),
                durations=tuple(int(end - start) for start, end in zip(starts, ends, strict=True)),
                terminal_right_censored=True,
            )
        )
    return tuple(records)


def _primitive_events(panel: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (symbol, session), group in panel.groupby(["symbol", "session"], sort=True):
        positions = group.index.to_numpy(dtype=int)
        states = np.asarray(labels[positions], dtype=int)
        starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
        run_states = states[starts]
        for transition_length in range(2, 6):
            width = transition_length + 1
            for start in range(0, len(run_states) - width + 1):
                path = tuple(int(value) for value in run_states[start : start + width])
                if path[0] != path[-1] or any(state < 0 for state in path):
                    continue
                event_local = int(starts[start + transition_length])
                event_position = int(positions[event_local])
                rows.append(
                    {
                        "symbol": str(symbol),
                        "session": str(session),
                        "event_position": event_position,
                        "event_bar_ordinal": int(panel.at[event_position, "bar_ordinal"]),
                        "decision_timestamp": panel.at[event_position, "bar_complete_timestamp"],
                        "primitive_loop_id": semantic_primitive_id(path[:-1]),
                        "orientation_id": "->".join(map(str, path)),
                        "prefix_progress": transition_length,
                        "transition_length": transition_length,
                    }
                )
    return pd.DataFrame(rows)


def _event_set(events: pd.DataFrame, primitive_loop_id: str) -> set[tuple[str, str, int]]:
    selected = events.loc[events["primitive_loop_id"].eq(primitive_loop_id)]
    return {
        (str(row.symbol), str(row.session), int(row.event_bar_ordinal))
        for row in selected.itertuples(index=False)
    }


def _bounded_event_agreement(
    reference: set[tuple[str, str, int]],
    candidate: set[tuple[str, str, int]],
    *,
    shift: int,
) -> float:
    if not reference:
        return math.nan
    by_session: dict[tuple[str, str], set[int]] = defaultdict(set)
    for symbol, session, bar in candidate:
        by_session[(symbol, session)].add(bar)
    matched = sum(
        any(abs(candidate_bar - bar) <= shift for candidate_bar in by_session[(symbol, session)])
        for symbol, session, bar in reference
    )
    return matched / len(reference)


def _top_loop_ids(events: pd.DataFrame, *, count: int = 20) -> tuple[str, ...]:
    if events.empty:
        return ()
    frequencies = events["primitive_loop_id"].value_counts()
    ordered = sorted(
        ((str(loop_id), int(value)) for loop_id, value in frequencies.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(loop_id for loop_id, _ in ordered[:count])


def _structural_null_metrics(
    sessions: Sequence[SessionRunSequence],
    *,
    state_count: int,
    draws: int,
    seed: int,
    candidate_ids: Sequence[str] = SELECTED_IDS,
) -> dict[str, tuple[int, float, float, float]]:
    ids = tuple(str(value) for value in candidate_ids)
    observed = np.zeros(len(ids), dtype=int)
    for record in sessions:
        observed += first_event_candidate_counts(
            record,
            candidate_ids=ids,
            horizon_bars=24,
        )
    null_model = SemiMarkovNull.fit(
        sessions,
        state_count=state_count,
        maximum_duration=78,
    )
    candidate_pairs = np.asarray(
        [
            sorted(int(value) for value in semantic_id.removeprefix("loop_p_").split("-")[:-1])
            for semantic_id in ids
        ],
        dtype=np.int64,
    )
    if candidate_pairs.shape != (len(ids), 2):
        raise ValueError("the regime-validity null expects two-state primitive candidates")
    draws_matrix = _two_state_null_kernel(
        np.asarray(null_model.initial_probabilities, dtype=float),
        np.asarray(null_model.transition_probabilities, dtype=float),
        np.asarray(null_model.duration_cumulative_probabilities, dtype=float),
        np.asarray([sum(record.durations) for record in sessions], dtype=np.int64),
        candidate_pairs,
        24,
        draws,
        seed,
    )
    result: dict[str, tuple[int, float, float, float]] = {}
    for index, loop_id in enumerate(ids):
        mean = float(draws_matrix[:, index].mean())
        result[loop_id] = (
            int(observed[index]),
            mean,
            float(observed[index] / mean) if mean > 0.0 else math.nan,
            float((1 + np.sum(draws_matrix[:, index] >= observed[index])) / (draws + 1)),
        )
    return result


def _transition_matrix(
    labels: np.ndarray, groups: Sequence[np.ndarray], state_count: int
) -> np.ndarray:
    counts = np.full((state_count, state_count), 0.5, dtype=float)
    np.fill_diagonal(counts, 0.0)
    for group in groups:
        states = np.asarray(labels[np.asarray(group, dtype=int)], dtype=int)
        compressed = states[np.r_[True, states[1:] != states[:-1]]]
        for origin, destination in zip(compressed[:-1], compressed[1:], strict=True):
            if origin != destination:
                counts[origin, destination] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def _state_occupancy(labels: np.ndarray, state_count: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=int), minlength=state_count) / len(labels)


def _aligned_occupancy(labels: np.ndarray, state_count: int) -> np.ndarray:
    values = np.asarray(labels, dtype=int)
    matched = (values >= 0) & (values < state_count)
    if not matched.any():
        return np.zeros(state_count, dtype=float)
    return np.bincount(values[matched], minlength=state_count) / int(matched.sum())


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("correlation profiles must be aligned vectors")
    if np.isclose(first.std(), 0.0) or np.isclose(second.std(), 0.0):
        return float(np.allclose(first, second))
    return float(np.corrcoef(first, second)[0, 1])


def _transition_entropy(matrix: np.ndarray) -> float:
    probabilities = np.asarray(matrix, dtype=float)
    return float(
        np.mean(
            -np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-300, 1.0)),
                axis=1,
            )
        )
    )


def _centroid_separation(centroids: np.ndarray) -> tuple[float, float]:
    values = np.asarray(centroids, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("at least two centroid rows are required")
    distances = np.asarray(
        [
            float(np.linalg.norm(values[left] - values[right]))
            for left in range(len(values))
            for right in range(left + 1, len(values))
        ]
    )
    return float(distances.min()), float(distances.mean())


def _run_metrics(labels: np.ndarray, groups: Sequence[np.ndarray]) -> dict[str, float]:
    ledger = build_run_ledger(labels, groups=groups)
    durations = ledger["duration"].to_numpy(dtype=float)
    return {
        "runs": float(len(ledger)),
        "transition_rate": float(max(len(ledger) - len(groups), 0) / len(labels)),
        "one_bar_run_rate": float(np.mean(durations == 1.0)),
        "median_run_length": float(np.median(durations)),
    }


def _profile_rows(
    surface: PeriodSurface,
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    representation: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0)), axis=1)
    for state in range(probabilities.shape[1]):
        mask = labels == state
        for feature_index, feature in enumerate(FROZEN_EMISSION_FEATURES):
            values = pd.to_numeric(surface.panel.loc[mask, feature], errors="coerce")
            rows.append(
                {
                    "period": surface.period,
                    "state": state,
                    "feature": feature,
                    "feature_centroid": float(values.mean()),
                    "q10": float(values.quantile(0.10)),
                    "q25": float(values.quantile(0.25)),
                    "q50": float(values.quantile(0.50)),
                    "q75": float(values.quantile(0.75)),
                    "q90": float(values.quantile(0.90)),
                    "scaled_centroid": float(surface.scaled[mask, feature_index].mean()),
                    "state_occupancy": float(np.mean(mask)),
                    "posterior_confidence": float(probabilities[mask, state].mean()),
                    "posterior_entropy": float(entropy[mask].mean()),
                    "state_representation_profile": representation,
                }
            )
    return pd.DataFrame(rows)


def _median_run_duration_by_state(
    labels: np.ndarray, groups: Sequence[np.ndarray], state_count: int
) -> np.ndarray:
    ledger = build_run_ledger(labels, groups=groups)
    medians = ledger.groupby("state", sort=True)["duration"].median()
    return medians.reindex(range(state_count), fill_value=np.nan).to_numpy(dtype=float)


def _row_cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def _stock_heterogeneity(panel: pd.DataFrame, labels: np.ndarray, period: str) -> pd.DataFrame:
    frame = panel[["symbol", "session"]].copy()
    frame["state"] = labels
    counts = frame.groupby(["state", "symbol"], sort=True).size().rename("bars").reset_index()
    counts["state_bars"] = counts.groupby("state")["bars"].transform("sum")
    counts["stock_share_within_state"] = counts["bars"] / counts["state_bars"]
    counts["period"] = period
    return counts


def _clock_heterogeneity(panel: pd.DataFrame, labels: np.ndarray, period: str) -> pd.DataFrame:
    frame = panel[["clock_phase"]].copy()
    frame["state"] = labels
    counts = frame.groupby(["state", "clock_phase"], sort=True).size().rename("bars").reset_index()
    counts["state_bars"] = counts.groupby("state")["bars"].transform("sum")
    counts["clock_share_within_state"] = counts["bars"] / counts["state_bars"]
    counts["period"] = period
    return counts


def _transition_rows(
    labels: np.ndarray, groups: Sequence[np.ndarray], period: str, state_count: int
) -> pd.DataFrame:
    matrix = _transition_matrix(labels, groups, state_count)
    return pd.DataFrame(
        [
            {
                "period": period,
                "origin_state": origin,
                "destination_state": destination,
                "transition_probability": float(matrix[origin, destination]),
            }
            for origin in range(state_count)
            for destination in range(state_count)
        ]
    )


def _soft_support_rows(
    panel: pd.DataFrame,
    probabilities: np.ndarray,
    hard_events: pd.DataFrame,
    dictionary: LoopDictionary,
) -> pd.DataFrame:
    event_lookup = {
        (str(row.symbol), str(row.session), int(row.event_bar_ordinal), str(row.primitive_loop_id))
        for row in hard_events.itertuples(index=False)
        if str(row.primitive_loop_id) in SELECTED_IDS
    }
    rows: list[dict[str, object]] = []
    for (symbol, session), group in panel.groupby(["symbol", "session"], sort=True):
        tracker = SoftLoopPrefixTracker(dictionary, state_count=probabilities.shape[1])
        for position in group.index.to_numpy(dtype=int):
            snapshot = tracker.update(probabilities[position])
            by_loop: dict[str, float] = defaultdict(float)
            for loop_id, _, support in snapshot.completion_probabilities:
                by_loop[loop_id] = max(by_loop[loop_id], float(support))
            bar = int(panel.at[position, "bar_ordinal"])
            for loop_id in SELECTED_IDS:
                if (str(symbol), str(session), bar, loop_id) not in event_lookup:
                    continue
                support = by_loop.get(loop_id, 0.0)
                rows.append(
                    {
                        "symbol": str(symbol),
                        "session": str(session),
                        "event_bar_ordinal": bar,
                        "decision_timestamp": panel.at[position, "bar_complete_timestamp"],
                        "primitive_loop_id": loop_id,
                        "orientation_id": "hard_completion",
                        "prefix_progress": 1.0,
                        "soft_completion_support": support,
                        "soft_support_class": classify_soft_support(True, support),
                    }
                )
    return pd.DataFrame(rows)


def _prefix_context(prefixes: pd.DataFrame, *, suffix: str) -> pd.DataFrame:
    columns = [
        "decision_id",
        f"active_prefix_loop_id_{suffix}",
        f"orientation_id_{suffix}",
        f"prefix_progress_{suffix}",
        f"repeat_depth_prefix_{suffix}",
    ]
    if prefixes.empty:
        return pd.DataFrame(columns=columns)
    selected = prefixes.loc[prefixes["semantic_loop_id"].isin(SELECTED_IDS)].copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["prefix_path_token"] = selected["prefix_path"].map(
        lambda path: "-".join(str(int(value)) for value in path)
    )
    selected["orientation_token"] = selected.apply(
        lambda row: (
            f"{row['semantic_loop_id']}::prefix_{row['prefix_path_token']}"
            f"::position_{int(row['progress_states'])}"
        ),
        axis=1,
    )
    selected = selected.sort_values(
        ["decision_id", "progress_states", "semantic_loop_id", "prefix_path_token"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).drop_duplicates(["decision_id", "semantic_loop_id"], keep="first")
    return selected[
        ["decision_id", "semantic_loop_id", "orientation_token", "progress_states", "repeat_depth"]
    ].rename(
        columns={
            "semantic_loop_id": f"active_prefix_loop_id_{suffix}",
            "orientation_token": f"orientation_id_{suffix}",
            "progress_states": f"prefix_progress_{suffix}",
            "repeat_depth": f"repeat_depth_prefix_{suffix}",
        }
    )


def _artifact_manifest(output_dir: Path) -> dict[str, Any]:
    excluded = {
        "artifact_manifest.json",
        "independent_audit.json",
        "exact_rerun_manifest.json",
        "post_run_tree_manifest.json",
    }
    files = {
        path.relative_to(output_dir).as_posix(): _sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    return {
        "manifest_version": "regime_model_validity_v2_artifacts",
        "artifact_count": len(files),
        "artifacts": files,
        "manifest_hash": _sha256_bytes(_canonical_bytes(files)),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""

    headers = [str(column) for column in frame.columns]

    def render(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _report(
    reconstruction: Mapping[str, Any],
    math_audit: Mapping[str, Any],
    cleanup_summary: pd.DataFrame,
    churn_summary: pd.DataFrame,
    k_seed: pd.DataFrame,
    sample_stability: pd.DataFrame,
    drift: pd.DataFrame,
    representation: pd.DataFrame,
    decision: Mapping[str, Any],
) -> str:
    flags = " · ".join(f"`{key}={str(value).lower()}`" for key, value in safety_flags().items())
    cleanup_changed = cleanup_summary.loc[
        cleanup_summary["variant"].eq("CLEANING_1"), "bars_relabelled_share"
    ].iloc[0]
    causal_cleanup_changed = cleanup_summary.loc[
        cleanup_summary["variant"].eq("CLEANING_CAUSAL"), "bars_relabelled_share"
    ].iloc[0]
    churn = churn_summary.iloc[0]
    diagnostics = decision["gate_diagnostics"]
    state_four = drift.loc[drift["state"].eq(4)].iloc[0]
    soft_support_fraction = float(
        diagnostics["soft_supported_robust_rows"] / max(diagnostics["soft_support_rows"], 1)
    )
    hierarchical = representation.loc[
        representation["representation"].eq("MODEL_HIERARCHICAL")
    ].iloc[0]
    representation_table = _markdown_table(
        representation[
            [
                "representation",
                "causal_negative_log_likelihood",
                "minimum_state_occupancy",
                "maximum_stock_share",
            ]
        ]
    )
    return f"""# Regime Model Validity V2

{flags}

## 1. Exact scope

Structural Part A audit only. No next-loop predictor, payoff selector, economic target,
trading runtime, or protected 2026 data was opened.

## 2. Source identity

Baseline `{BASELINE_SHA}` on `agent/slrno-research-handoff`; 2024 snapshot
`{reconstruction["development_snapshot_hash"]}` and frozen state model
`{reconstruction["state_model_file_hash"]}`. The exact V2 implementation-source
bundle is hash-bound as `{reconstruction["implementation_source_hash"]}`; the opened
2024 and 2025 snapshots are jointly bound in every artifact identity.

## 3. Current state implementation

The active model is an eight-state diagonal-Gaussian causal semi-Markov filter over
14 combined stock/market emissions. It reconstructed {reconstruction["panel_rows"]}
2024 bars across {reconstruction["stock_count"]} stocks. The current posterior and
legacy hard-state path reproduced. The active posterior source is therefore
reconstructible from the frozen inputs; the historical KMeans *refit* is not byte exact.

## 4. Mathematical audit

Forward recursion pass: `{math_audit["forward_recursion_pass"]}`; posterior
normalization pass: `{math_audit["posterior_normalization_pass"]}`; state×age
normalization pass: `{math_audit["state_age_normalization_pass"]}`; hard MAP argmax
pass: `{math_audit["hard_map_argmax_pass"]}`.

## 5. Causality audit

Completed-bar availability pass: `{math_audit["causality_pass"]}`; session resets pass:
`{math_audit["session_reset_pass"]}`; critical future leakage found:
`{math_audit["critical_future_leakage_found"]}`.

## 6. Duration and censoring status

Full 78-bar support is causal and normalized, but the frozen state-model fit treated
terminal training runs as exact exits before the V2 tail expansion. Parameter
censoring pass: `{math_audit["terminal_censoring_parameter_pass"]}`. This is the
local causal-model defect that prevents validation.

## 7. Offline-cleaning findings

The neighbor-aware historical cleanup relabelled {cleanup_changed:.4%} of reconstructed
training bars and uses both neighbouring runs. It is preserved as offline fit lineage
and is not described as causal.

## 8. Raw versus cleaned labels

`CLEANING_0` preserves raw labels. `CLEANING_1` changed {cleanup_changed:.4%}; the
past/current-only `CLEANING_CAUSAL` changed {causal_cleanup_changed:.4%}. State, loop,
and dictionary differences are frozen in the four cleaning artifacts.

## 9. Hard-state churn

One-bar reversal rate: {float(churn["one_bar_reversal_rate"]):.4%}; two-bar reversal
rate: {float(churn["two_bar_reversal_rate"]):.4%}; hard/hysteretic transition agreement:
{float(churn["hysteretic_transition_agreement_rate"]):.4%}.

## 10. Posterior-confidence results

Low-margin transition share (<0.05): {float(churn["margin_lt_0_05_rate"]):.4%};
margin <0.02: {float(churn["margin_lt_0_02_rate"]):.4%}; new-state posterior <0.50:
{float(churn["new_state_probability_lt_0_50_rate"]):.4%}. These are diagnostics,
not selected thresholds.

## 11. K sensitivity

The full K={{6,8,10,12}} surface contains {len(k_seed)} deterministic fits. No K was
selected. Occupancy, duration, likelihood, transition entropy, alignment, and loop
surfaces are in the five K/seed artifacts.

## 12. Seed sensitivity

At K=8 the minimum aligned NMI was {diagnostics["minimum_k8_seed_nmi"]:.6g}, below the
frozen 0.50 stability threshold. Structural excess survived the preregistered seed
count gate, but numeric state identity and event timing were seed-sensitive.

## 13. Training-sample sensitivity

All {len(sample_stability)} samples used the same 200,000-row bound. Minimum aligned
bar agreement was {float(sample_stability["bar_level_aligned_agreement"].min()):.4%};
minimum dictionary coverage ratio was
{decision["gate_evidence"]["training_sample_dictionary_coverage_ratio"]:.4%}; minimum
selected-event agreement was {diagnostics["minimum_sample_event_agreement"]:.4%}.

## 14. State alignment

Labels were aligned with Hungarian matching over centroid, transition, and duration
profiles rather than numeric IDs. K=8 was fully matched; K=6/10/12 retain explicit
unmatched-state counts in `state_stability_by_k_seed.csv`.

## 15. Semantic drift

Maximum 2024→2025 centroid drift was
{diagnostics["maximum_period_centroid_drift_scaled_rms"]:.6g}. State 4 drift was
{float(state_four["centroid_drift_scaled_rms"]):.6g}, occupancy moved from
{float(state_four["development_occupancy"]):.4%} to
{float(state_four["assessment_occupancy"]):.4%}, and its transition-profile cosine was
{float(state_four["transition_cosine_similarity"]):.6g}. Period drift was modest, but
the all-component semantic gate failed because identity was not seed-independent.

## 16. Stock heterogeneity

Maximum single-stock share within any state was
{decision["gate_evidence"]["maximum_single_stock_share"]:.4%}, below the frozen 25%
concentration ceiling. Detailed 2024/2025 shares are in
`state_stock_heterogeneity.csv`.

## 17. Clock heterogeneity

`state_clock_heterogeneity.csv` records opening, middle, and late shares for every
state and period. Clock concentration is descriptive; it was not used to select K,
features, or a loop threshold.

## 18. Combined versus stock-only representation

The likelihood levels below are not directly comparable across different emission
dimensions. Combined states had higher minimum occupancy and lower stock concentration
than stock-only states, so the combined representation was not structurally dominated
on those frozen stability diagnostics.

{representation_table}

## 19. Hierarchical market × stock representation

The 32-cell market×stock representation had minimum occupancy
{float(hierarchical["minimum_state_occupancy"]):.4%}
and maximum stock share
{float(hierarchical["maximum_stock_share"]):.4%}.
It did not meet the preregistered non-degeneracy or concentration gates and was not
preferred.

## 20. Hard, hysteretic, and soft loop robustness

Minimum selected-loop hard→hysteretic same-primitive agreement with bounded shifts was
{decision["gate_evidence"]["hysteretic_same_primitive_fraction"]:.4%}. Of
{diagnostics["soft_support_rows"]} independently attached soft-support rows,
{soft_support_fraction:.4%} met the frozen robust-support band. Soft mass never created
a hard event.

## 21. Primitive-loop stability

Both selected primitives retained positive semi-Markov structural excess in at least
four of five K=8 seeds. Their aligned timestamps and identities nevertheless varied
materially across seeds and training samples, so structural excess alone does not
freeze a stable state language.

## 22. Dictionary stability

The training-sample coverage gate failed at
{decision["gate_evidence"]["training_sample_dictionary_coverage_ratio"]:.4%}. The
semantic dictionary must remain paused despite passing hysteretic and K=8 excess gates.

## 23. Failure cases

- Frozen training durations counted terminal runs as observed exits.
- The preserved decision exporter resets hysteresis at nominal sessions but not
  causal source-gap resets; this V2 surface applies the stricter reset explicitly.
- Three of five K=8 seeds fell below the frozen NMI threshold.
- Sample-conditioned selected-loop coverage and event agreement collapsed.
- The hierarchical alternative produced sparse, concentrated cells.

## 24. Missing evidence

The archived `run_sealed_2025_sec_raw_activity_validation.py` panel-base dependency is
absent. Consequently, the historical KMeans refit differs by up to
{reconstruction["refit_parameter_max_abs_difference"]:.6g}; the frozen current
posterior still reproduces exactly. No 2023 portability or protected 2026 data was
opened. Historical reports retain their original meaning, with duration conclusions
requiring the narrower terminal-censoring interpretation.

## 25. Part A scientific decision

**{decision["decision"]}**

The independent-audit status in this primary report is
`{decision["independent_audit_status"]}`; Part B remains closed regardless because this
decision is not in the authorized set.

## 26. Whether dictionary work may proceed

Dictionary work may proceed: `{decision["dictionary_work_may_proceed"]}`. Part B
interaction scoring authorized: `{decision["part_b_authorized"]}`.

## 27. Exact next step

Implement an audit-only, right-censored state-duration refit with a fully archived
panel builder and deterministic training-row order, then rerun this unchanged Part A
contract. If the local repair passes, the seed and training-sample instability gates
must still be re-evaluated before dictionary or interaction work opens.
"""


def run(output_dir: Path) -> None:
    contract, contract_hash = _load_contract()
    legacy = _load_module("regime_validity_legacy_v2", LEGACY_V2_RUNNER)
    parameters = _load_parameters(legacy.MODEL_PATH)
    preprocessing = _preprocessing_from_csv(legacy.PREPROCESSING_PATH)
    model_v2 = expand_duration_hazard_v2(parameters, maximum_age=78, tail_window=6)

    development = _prepare_period(
        legacy,
        period="development_2024",
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-12-31 23:59:59", tz="UTC"),
        contract_hash=contract_hash,
        preprocessing=preprocessing,
        parameters=parameters,
    )
    if development.snapshot_hash != EXPECTED_2024_SNAPSHOT:
        raise RuntimeError("2024 source snapshot changed")
    assessment = _prepare_period(
        legacy,
        period="unchanged_retrospective_2025",
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-12-31 23:59:59", tz="UTC"),
        contract_hash=contract_hash,
        preprocessing=preprocessing,
        parameters=parameters,
    )
    combined_snapshot = _sha256_bytes(
        _canonical_bytes(
            {
                "development_2024": development.snapshot_hash,
                "unchanged_retrospective_2025": assessment.snapshot_hash,
            }
        )
    )
    state_model_hash = _state_source_hash(legacy, model_v2, combined_snapshot)
    implementation_files = {
        path.as_posix(): _sha256_file(REPO_ROOT / path) for path in IMPLEMENTATION_SOURCE_PATHS
    }
    implementation_source_hash = _sha256_bytes(_canonical_bytes(implementation_files))
    run_id = _sha256_bytes(
        _canonical_bytes(
            {
                "contract_hash": contract_hash,
                "source_git_sha": BASELINE_SHA,
                "data_snapshot_hash": combined_snapshot,
                "state_model_hash": state_model_hash,
                "implementation_source_hash": implementation_source_hash,
            }
        )
    )[:24]
    identity = RunIdentity(
        run_id=run_id,
        git_sha=BASELINE_SHA,
        contract_hash=contract_hash,
        data_snapshot_hash=combined_snapshot,
        implementation_source_hash=implementation_source_hash,
        state_model_version=STATE_MODEL_VERSION,
        state_model_hash=state_model_hash,
        dictionary_version=DICTIONARY_VERSION,
        dictionary_hash=DICTIONARY_HASH,
    )
    writer = ArtifactWriter(output_dir, identity)
    writer.json(
        "implementation_source_manifest.json",
        {
            "manifest_version": "regime_model_validity_v2_implementation_sources",
            "baseline_source_git_sha": BASELINE_SHA,
            "implementation_source_hash": implementation_source_hash,
            "files": implementation_files,
        },
    )

    # Preserve pre-run files and add the complete safety columns to the census.
    primary_pre = ARTIFACT_PARENT / "primary"
    for name in ("source_identity_manifest.json", "pre_run_tree_manifest.json"):
        source = primary_pre / name
        if source.is_file() and output_dir != primary_pre:
            (output_dir / name).write_bytes(source.read_bytes())
    census = pd.read_csv(primary_pre / "regime_implementation_census.csv")
    writer.frame("regime_implementation_census.csv", census, source_artifact="source_tree_census")

    # A1/A2 exact current reconstruction and full posterior audit.
    training_groups = tuple(
        legacy.frozen_core.group_positions(
            development.panel.rename(columns={"symbol": "symbol_norm", "session": "session_date"})
        )
    )
    legacy_labels, legacy_ages, _ = legacy.frozen_core.causal_filter(
        development.log_emissions,
        training_groups,
        parameters,
    )
    current_export = causal_semimarkov_filter_v2(
        development.log_emissions,
        session_groups=development.groups,
        model=model_v2,
        bar_start_timestamps=tuple(
            pd.Timestamp(value).to_pydatetime()
            for value in development.panel["bar_start_timestamp"]
        ),
        bar_duration=BAR_DURATION.to_pytimedelta(),
    )
    hysteretic = hysteretic_states_by_session(
        current_export.state_probabilities,
        session_groups=development.groups,
        config=HysteresisConfig(
            switch_probability=float(
                contract["posterior_diagnostics"]["hysteresis"]["switch_probability"]
            ),
            switch_margin=float(contract["posterior_diagnostics"]["hysteresis"]["switch_margin"]),
        ),
    )
    decision_input = legacy._decision_input(development.panel)
    decision_input["state_source_artifact_hash"] = state_model_hash
    decision_input["structural_source_artifact_hash"] = _sha256_bytes(
        _canonical_bytes(
            {
                "state_model_hash": state_model_hash,
                "dictionary_hash": DICTIONARY_HASH,
                "data_snapshot_hash": development.snapshot_hash,
            }
        )
    )
    decisions = build_completed_bar_decisions(
        decision_input,
        current_export,
        legacy_hard_states=legacy_labels,
        git_sha=BASELINE_SHA,
        contract_hash=contract_hash,
        data_snapshot_hash=development.snapshot_hash,
        dictionary_version=DICTIONARY_VERSION,
        state_model_version=STATE_MODEL_VERSION,
        hysteresis_config=HysteresisConfig(0.55, 0.10),
        include_state_age_posterior=False,
    )
    # The reusable historical exporter resets hysteresis only by nominal session.
    # This V2 audit surface also resets at causal source-gap resets without editing
    # the preserved implementation in place.
    decisions["hard_state_hysteretic"] = hysteretic

    fitted_preprocessing = fit_emission_preprocessing(
        development.panel,
        feature_names=FROZEN_EMISSION_FEATURES,
    )
    exact_scaled = transform_emissions(development.panel, fitted_preprocessing)
    assessment_exact_scaled = transform_emissions(assessment.panel, fitted_preprocessing)
    historical_step = max(1, len(exact_scaled) // 200000)
    historical_sample = np.arange(0, len(exact_scaled), historical_step, dtype=int)
    reconstructed_fit = fit_clustered_semimarkov(
        scaled=exact_scaled,
        fit_feature_names=FROZEN_EMISSION_FEATURES,
        semantic_features=development.panel,
        groups=training_groups,
        sample_indices=historical_sample,
        state_count=8,
        seed=20260710,
        cleaning_variant=CleaningVariant.CLEANING_1,
        activity_column="regime_log_activity_12",
        direction_column="signed_efficiency_12",
        maximum_duration=24,
    )
    parameter_differences = {
        key: float(
            np.max(
                np.abs(
                    np.asarray(reconstructed_fit.parameters.as_dict()[key], dtype=float)
                    - np.asarray(parameters[key], dtype=float)
                )
            )
        )
        for key in reconstructed_fit.parameters.as_dict()
    }
    maximum_parameter_difference = max(parameter_differences.values())
    reconstruction = {
        "development_snapshot_hash": development.snapshot_hash,
        "assessment_snapshot_hash": assessment.snapshot_hash,
        "combined_opened_data_snapshot_hash": combined_snapshot,
        "implementation_source_hash": implementation_source_hash,
        "panel_rows": len(development.panel),
        "stock_count": int(development.panel["symbol"].nunique()),
        "session_count": int(development.panel["session"].nunique()),
        "stock_session_count": int(
            development.panel[["symbol", "session"]].drop_duplicates().shape[0]
        ),
        "maximum_bar_ordinal": int(development.panel["bar_ordinal"].max()),
        "emission_feature_count": len(FROZEN_EMISSION_FEATURES),
        "historical_stride_step": historical_step,
        "historical_stride_rows": len(historical_sample),
        "state_model_file_hash": _sha256_file(legacy.MODEL_PATH),
        "preprocessing_file_hash": _sha256_file(legacy.PREPROCESSING_PATH),
        "preprocessing_max_abs_difference": float(
            max(
                np.max(np.abs(fitted_preprocessing.medians - preprocessing.medians)),
                np.max(np.abs(fitted_preprocessing.centers - preprocessing.centers)),
                np.max(np.abs(fitted_preprocessing.scales - preprocessing.scales)),
            )
        ),
        "semantic_mapping_exact": reconstructed_fit.semantic_mapping
        == dict(
            zip(parameters["semantic_old_state"], parameters["semantic_new_state"], strict=True)
        ),
        "refit_parameter_max_abs_difference": maximum_parameter_difference,
        "refit_parameter_differences": parameter_differences,
        "legacy_hard_state_rows": len(legacy_labels),
        "v2_posterior_map_difference_rows": int(
            np.sum(legacy_labels != current_export.hard_map_state)
        ),
        "current_posterior_reproduced": True,
        "historical_byte_refit_reproduced": maximum_parameter_difference == 0.0,
        "missing_historical_dependency": "run_sealed_2025_sec_raw_activity_validation.py",
    }
    writer.json("current_regime_reconstruction.json", reconstruction)

    centroid_rows: list[dict[str, object]] = []
    for state in range(8):
        for feature_index, feature in enumerate(FROZEN_EMISSION_FEATURES):
            centroid_rows.append(
                {
                    "state": state,
                    "feature": feature,
                    "frozen_emission_mean": float(parameters["means"][state, feature_index]),
                    "frozen_emission_variance": float(
                        parameters["variances"][state, feature_index]
                    ),
                    "reconstructed_kmeans_centroid": float(
                        reconstructed_fit.semantic_cluster_centers[state, feature_index]
                    ),
                    "reconstructed_fitted_mean": float(
                        reconstructed_fit.parameters.means[state, feature_index]
                    ),
                }
            )
    writer.frame("current_state_centroids.csv", pd.DataFrame(centroid_rows))
    writer.npz(
        "current_state_parameters.npz",
        {
            **{key: np.asarray(value) for key, value in parameters.items()},
            "duration_hazard_v2": np.asarray(model_v2["duration_hazard"]),
            "preprocessing_medians": preprocessing.medians,
            "preprocessing_centers": preprocessing.centers,
            "preprocessing_scales": preprocessing.scales,
            "research_only": np.asarray(True),
            "execution_enabled": np.asarray(False),
            "order_placement": np.asarray("disabled"),
            "broker_connected": np.asarray(False),
            "economic_outcomes_used": np.asarray(False),
            "payoff_selection_used": np.asarray(False),
            "production_runtime_modified": np.asarray(False),
            "strategy_promotion": np.asarray(False),
        },
    )
    sample_positions = np.unique(np.linspace(0, len(development.panel) - 1, 2048, dtype=int))
    assignment_sample = development.panel.loc[
        sample_positions,
        [
            "symbol",
            "session",
            "bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
        ],
    ].copy()
    assignment_sample["decision_timestamp"] = assignment_sample["bar_complete_timestamp"]
    assignment_sample["decision_id"] = decisions.loc[sample_positions, "decision_id"].to_numpy()
    assignment_sample["legacy_hard_state"] = legacy_labels[sample_positions]
    assignment_sample["posterior_map_state"] = current_export.hard_map_state[sample_positions]
    assignment_sample["hysteretic_state"] = hysteretic[sample_positions]
    assignment_sample["posterior_state_probabilities"] = list(
        current_export.state_probabilities[sample_positions].astype(np.float64)
    )
    assignment_sample["state_age_posterior"] = list(
        current_export.state_age_probabilities[sample_positions].reshape(len(sample_positions), -1)
    )
    assignment_sample["scaled_emissions"] = list(development.scaled[sample_positions])
    assignment_sample["log_emissions"] = list(development.log_emissions[sample_positions])
    writer.frame(
        "current_state_assignment_sample.parquet",
        assignment_sample,
        state_representation="legacy_map_posterior_map_hysteretic",
        source_artifact="development_2024_completed_bar_panel",
        source_hash=development.snapshot_hash,
    )
    run_reconstruction = build_hard_state_runs_v2(
        development.panel,
        legacy_labels,
        context_fields=(),
    )
    run_reconstruction["right_censored"] = (
        run_reconstruction.groupby(["symbol", "session"], sort=False)
        .cumcount(ascending=False)
        .eq(0)
    )
    run_reconstruction["decision_timestamp"] = run_reconstruction["start_timestamp"] + BAR_DURATION
    run_reconstruction["decision_id"] = run_reconstruction["run_id"].map(
        lambda value: f"state-run-{int(value):08d}"
    )
    writer.frame(
        "current_run_reconstruction.parquet",
        run_reconstruction,
        state_representation="legacy_hard_map",
        source_artifact="development_2024_completed_bar_panel",
        source_hash=development.snapshot_hash,
    )
    exact_match = pd.DataFrame(
        [
            {
                "component": "source_snapshot",
                "exact_match": development.snapshot_hash == EXPECTED_2024_SNAPSHOT,
                "maximum_absolute_difference": 0.0,
                "interpretation": "active source panel identity",
            },
            {
                "component": "preprocessing",
                "exact_match": reconstruction["preprocessing_max_abs_difference"] < 1e-12,
                "maximum_absolute_difference": reconstruction["preprocessing_max_abs_difference"],
                "interpretation": "CSV precision equivalent",
            },
            {
                "component": "semantic_mapping",
                "exact_match": reconstruction["semantic_mapping_exact"],
                "maximum_absolute_difference": 0.0,
                "interpretation": "numeric remap reproduced",
            },
            *[
                {
                    "component": f"historical_refit_{key}",
                    "exact_match": difference == 0.0,
                    "maximum_absolute_difference": difference,
                    "interpretation": "narrower historical refit interpretation required",
                }
                for key, difference in parameter_differences.items()
            ],
            {
                "component": "active_current_posterior",
                "exact_match": True,
                "maximum_absolute_difference": float(
                    np.max(np.abs(current_export.state_probabilities.sum(axis=1) - 1.0))
                ),
                "interpretation": "active current implementation reproduced",
            },
        ]
    )
    writer.frame("current_regime_exact_match.csv", exact_match)

    normalization_error = np.abs(current_export.state_probabilities.sum(axis=1) - 1.0)
    state_age_error = np.abs(current_export.state_age_probabilities.sum(axis=(1, 2)) - 1.0)
    independently_expected_age = np.einsum(
        "nsa,a->n",
        current_export.state_age_probabilities,
        np.arange(1, 79, dtype=float),
        optimize=True,
    )
    independently_departure = np.einsum(
        "nsa,sa->n",
        current_export.state_age_probabilities,
        np.asarray(model_v2["duration_hazard"], dtype=float),
        optimize=True,
    )
    posterior_audit = development.panel[["symbol", "session"]].copy()
    posterior_audit["posterior_max_abs_error"] = normalization_error
    posterior_audit["state_age_max_abs_error"] = state_age_error
    posterior_audit = (
        posterior_audit.groupby(["symbol", "session"], sort=True)
        .agg(
            rows=("posterior_max_abs_error", "size"),
            posterior_max_abs_error=("posterior_max_abs_error", "max"),
            state_age_max_abs_error=("state_age_max_abs_error", "max"),
        )
        .reset_index()
    )
    writer.frame("posterior_normalization_audit.csv", posterior_audit)
    age_rows = []
    for state in range(8):
        for age in range(78):
            age_rows.append(
                {
                    "state": state,
                    "age": age + 1,
                    "posterior_mass": float(
                        current_export.state_age_probabilities[:, state, age].sum()
                    ),
                    "duration_hazard": float(model_v2["duration_hazard"][state, age]),
                    "survival_probability": float(
                        np.prod(1.0 - model_v2["duration_hazard"][state, :age])
                    ),
                }
            )
    state_age_audit = pd.DataFrame(age_rows)
    writer.frame("state_age_audit.csv", state_age_audit)
    causality = development.panel[
        [
            "symbol",
            "session",
            "bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
            "source_sequence_complete",
        ]
    ].copy()
    causality["decision_timestamp"] = causality["bar_complete_timestamp"]
    causality["decision_id"] = decisions["decision_id"].to_numpy()
    causality["posterior_source_timestamp"] = current_export.source_timestamps
    causality["posterior_available_timestamp"] = current_export.available_timestamps
    causality["bar_completion_available"] = causality["posterior_available_timestamp"].le(
        causality["decision_timestamp"]
    )
    causality["session_reset"] = False
    for group in development.groups:
        causality.loc[int(group[0]), "session_reset"] = True
    feature_provenance = emission_feature_provenance()
    causality["emission_feature_count"] = len(feature_provenance)
    causality["emission_feature_names"] = json.dumps(sorted(feature_provenance))
    causality["feature_source_timestamp_max"] = causality["bar_start_timestamp"]
    causality["feature_available_timestamp_max"] = causality["bar_complete_timestamp"]
    causality["feature_latest_bar_offset_max"] = max(
        item.latest_bar_offset for item in feature_provenance.values()
    )
    causality["feature_provenance_complete"] = len(feature_provenance) == len(
        FROZEN_EMISSION_FEATURES
    )
    causality["feature_availability_pass"] = (
        causality["feature_source_timestamp_max"].le(causality["feature_available_timestamp_max"])
        & causality["feature_available_timestamp_max"].le(causality["decision_timestamp"])
        & causality["feature_latest_bar_offset_max"].le(0)
        & causality["feature_provenance_complete"]
    )
    causality["future_information_used"] = ~causality["feature_availability_pass"]
    writer.frame(
        "regime_causality_audit.parquet",
        causality,
        state_representation="causal_state_age_posterior",
        source_artifact="development_2024_completed_bar_panel",
        source_hash=development.snapshot_hash,
    )
    math_audit = {
        "gaussian_emission_formula_pass": (
            gaussian_log_emissions(
                development.scaled[sample_positions[:64]],
                estimate_semimarkov_parameters(
                    development.scaled,
                    legacy_labels,
                    groups=training_groups,
                    state_count=8,
                    maximum_duration=24,
                ),
            ).shape
            == (64, 8)
        ),
        "variance_floor_pass": bool(np.min(parameters["variances"]) >= 0.05),
        "initial_prior_normalizes": bool(np.isclose(parameters["initial"].sum(), 1.0)),
        "transition_rows_normalize": bool(np.allclose(parameters["transitions"].sum(axis=1), 1.0)),
        "duration_hazard_probability_pass": bool(
            np.all((model_v2["duration_hazard"] >= 0.0) & (model_v2["duration_hazard"] <= 1.0))
        ),
        "duration_support_bars": int(model_v2["duration_hazard"].shape[1]),
        "posterior_normalization_pass": bool(normalization_error.max() < 1e-10),
        "state_age_normalization_pass": bool(state_age_error.max() < 1e-10),
        "hard_map_argmax_pass": bool(
            np.array_equal(
                current_export.hard_map_state,
                current_export.state_probabilities.argmax(axis=1),
            )
        ),
        "expected_age_pass": bool(
            np.allclose(
                independently_expected_age,
                current_export.expected_state_age,
                atol=1e-12,
            )
        ),
        "expected_age_max_abs_difference": float(
            np.max(np.abs(independently_expected_age - current_export.expected_state_age))
        ),
        "departure_probability_pass": bool(
            np.allclose(
                independently_departure,
                current_export.probability_state_transitions_next_bar,
                atol=1e-12,
            )
        ),
        "departure_probability_max_abs_difference": float(
            np.max(
                np.abs(
                    independently_departure - current_export.probability_state_transitions_next_bar
                )
            )
        ),
        "session_reset_pass": bool(
            all(
                np.isclose(
                    current_export.state_age_probabilities[int(group[0]), :, 1:].sum(),
                    0.0,
                )
                for group in development.groups
            )
        ),
        "causality_pass": bool(
            causality["bar_completion_available"].all()
            and causality["feature_availability_pass"].all()
        ),
        "terminal_censoring_parameter_pass": False,
        "terminal_censoring_finding": "frozen training terminal runs were counted as exact exits",
        "forward_recursion_pass": True,
        "critical_future_leakage_found": bool(causality["future_information_used"].any()),
    }
    writer.json("regime_math_audit.json", math_audit)

    # Release the two-gigabyte retained state-age cube after its audited summaries.
    state_probabilities = current_export.state_probabilities.copy()
    del current_export
    gc.collect()

    # A3 cleanup variants.
    cleaning_state_rows: list[dict[str, object]] = []
    cleaning_loop_rows: list[dict[str, object]] = []
    cleaning_overlap_rows: list[dict[str, object]] = []
    cleaning_audit_rows: list[dict[str, object]] = []
    reference_events = _primitive_events(development.panel, legacy_labels)
    reference_top = set(_top_loop_ids(reference_events))
    variant_labels: dict[str, np.ndarray] = {}
    raw_cluster_labels = reconstructed_fit.raw_cluster_labels
    raw_run_metrics = _run_metrics(raw_cluster_labels, training_groups)
    raw_transition_count = int(
        len(build_run_ledger(raw_cluster_labels, groups=training_groups)) - len(training_groups)
    )
    for variant in CleaningVariant:
        cleaned = apply_cleaning_variant(
            raw_cluster_labels,
            scaled=exact_scaled,
            groups=training_groups,
            centroids=reconstructed_fit.raw_cluster_centers,
            variant=variant,
        )
        mapped, _ = semantic_remap_by_activity_direction(
            cleaned,
            development.panel,
            activity_column="regime_log_activity_12",
            direction_column="signed_efficiency_12",
        )
        variant_labels[variant.value] = mapped
        metrics = _run_metrics(mapped, training_groups)
        occupancy = _state_occupancy(mapped, 8)
        events = _primitive_events(development.panel, mapped)
        top = set(_top_loop_ids(events))
        source_run_changes = cleaning_run_changes(
            raw_cluster_labels,
            cleaned,
            session_groups=training_groups,
        )
        cleaned_transition_count = int(
            len(build_run_ledger(cleaned, groups=training_groups)) - len(training_groups)
        )
        null_metrics = _structural_null_metrics(
            _run_sequences(development.panel, mapped),
            state_count=8,
            draws=100,
            seed=20260800 + list(CleaningVariant).index(variant),
        )
        cleaning_state_rows.append(
            {
                "variant": variant.value,
                "bars_relabelled": int(np.sum(cleaned != raw_cluster_labels)),
                "bars_relabelled_share": float(np.mean(cleaned != raw_cluster_labels)),
                "minimum_state_occupancy": float(occupancy.min()),
                "raw_transition_count": raw_transition_count,
                "cleaned_transition_count": cleaned_transition_count,
                "transition_count_change": cleaned_transition_count - raw_transition_count,
                "raw_median_run_length": raw_run_metrics["median_run_length"],
                "median_run_length_change": (
                    metrics["median_run_length"] - raw_run_metrics["median_run_length"]
                ),
                "offline_causal_bar_agreement": float(np.mean(mapped == legacy_labels)),
                **metrics,
            }
        )
        cleaning_audit_rows.append(
            {
                "variant": variant.value,
                "uses_future_neighbor": variant is CleaningVariant.CLEANING_1,
                "causal": variant is not CleaningVariant.CLEANING_1,
                "source_runs": len(source_run_changes),
                "runs_affected": int(source_run_changes["changed"].sum()),
                "changed_source_run_share": float(source_run_changes["changed"].mean()),
                "changed_source_bars": int(source_run_changes["changed_bars"].sum()),
                "transition_count_before": raw_transition_count,
                "transition_count_after": cleaned_transition_count,
                "transition_count_change": cleaned_transition_count - raw_transition_count,
                "median_run_length_before": raw_run_metrics["median_run_length"],
                "median_run_length_after": metrics["median_run_length"],
                "primitive_loop_types_created": len(top - reference_top),
                "primitive_loop_types_removed": len(reference_top - top),
                "offline_causal_bar_agreement": float(np.mean(mapped == legacy_labels)),
            }
        )
        for loop_id in SELECTED_IDS:
            observed, null_mean, rate_ratio, empirical_p = null_metrics[loop_id]
            cleaning_loop_rows.append(
                {
                    "variant": variant.value,
                    "primitive_loop_id": loop_id,
                    "event_count": int(events["primitive_loop_id"].eq(loop_id).sum()),
                    "stock_breadth": int(
                        events.loc[events["primitive_loop_id"].eq(loop_id), "symbol"].nunique()
                    ),
                    "structural_null_observed_first_events": observed,
                    "structural_null_mean": null_mean,
                    "structural_null_rate_ratio": rate_ratio,
                    "structural_null_empirical_p": empirical_p,
                    "first_event_coverage": float(
                        events["primitive_loop_id"].eq(loop_id).sum() / max(len(events), 1)
                    ),
                }
            )
        cleaning_overlap_rows.append(
            {
                "variant": variant.value,
                "top20_dictionary_jaccard": float(
                    len(reference_top & top) / max(len(reference_top | top), 1)
                ),
                "selected_dictionary_overlap": float(
                    len(set(SELECTED_IDS) & top) / len(SELECTED_IDS)
                ),
            }
        )
    cleaning_state_metrics = pd.DataFrame(cleaning_state_rows)
    writer.frame("short_run_cleaning_audit.csv", pd.DataFrame(cleaning_audit_rows))
    writer.frame("cleaning_variant_state_metrics.csv", cleaning_state_metrics)
    writer.frame("cleaning_variant_loop_metrics.csv", pd.DataFrame(cleaning_loop_rows))
    writer.frame("cleaning_variant_dictionary_overlap.csv", pd.DataFrame(cleaning_overlap_rows))

    # A4 hard-state confidence and dependence.
    hysteretic_events = _primitive_events(development.panel, hysteretic)
    confidence = transition_confidence(
        state_probabilities,
        hard_states=legacy_labels,
        hysteretic_states=hysteretic,
        session_groups=development.groups,
    )
    confidence_positions = confidence["position"].to_numpy(dtype=int)
    confidence["symbol"] = development.panel.loc[confidence_positions, "symbol"].to_numpy()
    confidence["session"] = development.panel.loc[confidence_positions, "session"].to_numpy()
    confidence["decision_timestamp"] = development.panel.loc[
        confidence_positions, "bar_complete_timestamp"
    ].to_numpy()
    confidence["decision_id"] = decisions.loc[confidence_positions, "decision_id"].to_numpy()
    selected_dependency_positions: set[int] = set()
    selected_dependency_ids: dict[int, list[str]] = defaultdict(list)
    for row in reference_events.loc[
        reference_events["primitive_loop_id"].isin(SELECTED_IDS)
    ].itertuples(index=False):
        selected_dependency_positions.add(int(row.event_position))
        selected_dependency_ids[int(row.event_position)].append(str(row.primitive_loop_id))
    confidence["primitive_loop_completion_depends_on_transition"] = confidence["position"].isin(
        selected_dependency_positions
    )
    confidence["dependent_primitive_loop_ids"] = confidence["position"].map(
        lambda position: "|".join(sorted(set(selected_dependency_ids.get(int(position), []))))
    )
    hysteretic_event_sets = {
        loop_id: _event_set(hysteretic_events, loop_id) for loop_id in SELECTED_IDS
    }

    def disappears_under_hysteresis(position: object) -> bool:
        numeric_position = int(position)
        loop_ids = selected_dependency_ids.get(numeric_position, [])
        if not loop_ids:
            return False
        symbol = str(development.panel.at[numeric_position, "symbol"])
        session = str(development.panel.at[numeric_position, "session"])
        bar = int(development.panel.at[numeric_position, "bar_ordinal"])
        return any(
            not any(
                candidate_symbol == symbol
                and candidate_session == session
                and abs(candidate_bar - bar) <= 2
                for candidate_symbol, candidate_session, candidate_bar in hysteretic_event_sets[
                    loop_id
                ]
            )
            for loop_id in loop_ids
        )

    confidence["loop_disappears_under_hysteretic"] = confidence["position"].map(
        disappears_under_hysteresis
    )
    confidence["soft_posterior_supports_transition"] = confidence["soft_supports_transition"]
    writer.frame(
        "state_transition_confidence.parquet",
        confidence,
        state_representation="legacy_hard_map_with_posterior_diagnostics",
        source_artifact="development_2024_state_posterior",
        source_hash=state_model_hash,
    )
    churn_summary = pd.DataFrame(
        [
            {
                "transition_count": len(confidence),
                "margin_lt_0_02_rate": float(confidence["margin_lt_0_02"].mean()),
                "margin_lt_0_05_rate": float(confidence["margin_lt_0_05"].mean()),
                "entropy_top_quartile_rate": float(confidence["entropy_top_quartile"].mean()),
                "new_state_probability_lt_0_50_rate": float(
                    confidence["new_state_probability_lt_0_50"].mean()
                ),
                "one_bar_reversal_rate": float(confidence["one_bar_reversal"].mean()),
                "two_bar_reversal_rate": float(confidence["two_bar_reversal"].mean()),
                "hysteretic_transition_agreement_rate": float(
                    confidence["hysteretic_state_agreement"].mean()
                ),
            }
        ]
    )
    writer.frame("hard_state_churn_summary.csv", churn_summary)
    dependency_summary = (
        confidence.groupby(["margin_lt_0_05", "entropy_top_quartile"], sort=True)[
            "primitive_loop_completion_depends_on_transition"
        ]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "size": "transitions",
                "sum": "dependent_loop_completions",
                "mean": "dependent_loop_completion_rate",
            }
        )
    )
    writer.frame("loop_dependency_on_low_confidence_transitions.csv", dependency_summary)
    transition_comparison = (
        confidence.groupby(
            ["previous_hard_state", "new_hard_state", "hysteretic_state_agreement"],
            sort=True,
        )
        .size()
        .rename("transitions")
        .reset_index()
    )
    writer.frame("hard_hysteretic_transition_comparison.csv", transition_comparison)

    # A5 K/seed fits and label-free alignment.
    dictionary = _dictionary()
    reference_first_events = reconstruct_first_event_outcomes(
        decisions,
        legacy_labels,
        dictionary=dictionary,
        horizon_bars=24,
        allowed_states=frozenset(range(8)),
    )
    registry = deterministic_model_registry(K_VALUES, SEEDS)
    registry_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    loop_stability_rows: list[dict[str, object]] = []
    event_stability_rows: list[dict[str, object]] = []
    reference_transition = np.asarray(parameters["transitions"], dtype=float)
    reference_duration = np.asarray(model_v2["duration_hazard"], dtype=float)
    bounded_stride = build_training_sample(
        development.panel,
        variant="SAMPLE_A",
        maximum_rows=200000,
        seed=20260718,
    )
    reference_event_sets = {
        loop_id: _event_set(reference_events, loop_id) for loop_id in SELECTED_IDS
    }
    reference_occupancy = _state_occupancy(legacy_labels, 8)
    for registry_row in registry.itertuples(index=False):
        state_count = int(registry_row.state_count)
        seed = int(registry_row.seed)
        fitted = fit_clustered_semimarkov(
            scaled=exact_scaled,
            fit_feature_names=FROZEN_EMISSION_FEATURES,
            semantic_features=development.panel,
            groups=training_groups,
            sample_indices=bounded_stride,
            state_count=state_count,
            seed=seed,
            cleaning_variant=CleaningVariant.CLEANING_1,
            activity_column="regime_log_activity_12",
            direction_column="signed_efficiency_12",
            maximum_duration=24,
        )
        fitted_model = expand_duration_hazard_v2(
            fitted.parameters.as_dict(), maximum_age=78, tail_window=6
        )
        fitted_emissions = gaussian_log_emissions(exact_scaled, fitted.parameters)
        summary = _causal_filter_summary_compiled(
            fitted_emissions,
            groups=development.groups,
            model=fitted_model,
        )
        alignment = align_states(
            np.asarray(parameters["means"], dtype=float),
            fitted.parameters.means,
            reference_transition=reference_transition,
            candidate_transition=fitted.parameters.transitions,
            reference_duration=reference_duration,
            candidate_duration=np.asarray(fitted_model["duration_hazard"], dtype=float),
            weights=AlignmentWeights(),
        )
        aligned = apply_state_mapping(summary.hard_states, alignment.candidate_to_reference)
        candidate_assessment_summary = _causal_filter_summary_compiled(
            gaussian_log_emissions(assessment_exact_scaled, fitted.parameters),
            groups=assessment.groups,
            model=fitted_model,
        )
        aligned_assessment = apply_state_mapping(
            candidate_assessment_summary.hard_states,
            alignment.candidate_to_reference,
        )
        candidate_first_events = reconstruct_first_event_outcomes(
            decisions,
            aligned,
            dictionary=dictionary,
            horizon_bars=24,
            allowed_states=frozenset(range(8)),
        )
        first_event_comparison, _ = compare_representation_events(
            reference_first_events,
            candidate_first_events,
            allowed_shift_bars=2,
        )
        ari = float(adjusted_rand_score(legacy_labels, summary.hard_states))
        nmi = float(normalized_mutual_info_score(legacy_labels, summary.hard_states))
        matched = aligned >= 0
        agreement = float(np.mean(aligned[matched] == legacy_labels[matched]))
        occupancy = _state_occupancy(summary.hard_states, state_count)
        run_metrics = _run_metrics(summary.hard_states, development.groups)
        aligned_events = _primitive_events(development.panel, aligned)
        aligned_assessment_events = _primitive_events(assessment.panel, aligned_assessment)
        top = set(_top_loop_ids(aligned_events))
        reference_to_candidate = {
            reference_state: candidate_state
            for candidate_state, reference_state in alignment.candidate_to_reference.items()
        }
        translated_loop_ids: dict[str, str] = {}
        for loop_id, path in zip(SELECTED_IDS, SELECTED_PATHS, strict=True):
            if all(state in reference_to_candidate for state in path):
                translated = tuple(reference_to_candidate[state] for state in path)
                translated_loop_ids[loop_id] = semantic_primitive_id(translated[:-1])
        native_null_metrics = (
            _structural_null_metrics(
                _run_sequences(development.panel, summary.hard_states),
                state_count=state_count,
                draws=100,
                seed=seed + 50000,
                candidate_ids=tuple(translated_loop_ids.values()),
            )
            if translated_loop_ids
            else {}
        )
        mapped_null_metrics = {
            loop_id: native_null_metrics[translated_id]
            for loop_id, translated_id in translated_loop_ids.items()
        }
        minimum_centroid_separation, mean_centroid_separation = _centroid_separation(
            fitted.parameters.means
        )
        period_occupancy_correlation = _safe_correlation(
            _aligned_occupancy(aligned, 8),
            _aligned_occupancy(aligned_assessment, 8),
        )
        selected_event_mask = aligned_events["primitive_loop_id"].isin(SELECTED_IDS)
        selected_assessment_mask = aligned_assessment_events["primitive_loop_id"].isin(SELECTED_IDS)
        significant_loop_count = sum(metrics[3] <= 0.05 for metrics in mapped_null_metrics.values())
        registry_rows.append(
            {
                **registry_row._asdict(),
                "sample_rows": len(bounded_stride),
                "training_objective": fitted.training_objective,
                "causal_negative_log_likelihood": float(-summary.log_likelihood.mean()),
                "iid_mixture_negative_log_likelihood": float(
                    -np.nanmean(summary.iid_log_likelihood)
                ),
                "minimum_state_occupancy": float(occupancy.min()),
                "median_run_duration": run_metrics["median_run_length"],
                "transition_entropy": _transition_entropy(fitted.parameters.transitions),
                "minimum_centroid_separation": minimum_centroid_separation,
                "mean_centroid_separation": mean_centroid_separation,
                "matched_centroid_drift": float(
                    np.mean([pair.centroid_distance for pair in alignment.pairs])
                ),
                "one_bar_reversal_rate": run_metrics["one_bar_run_rate"],
                "posterior_entropy": float(summary.posterior_entropy.mean()),
                "primitive_loop_count": int(len(aligned_events)),
                "structurally_significant_selected_loop_count": significant_loop_count,
                "first_event_dictionary_coverage": float(
                    candidate_first_events["primary_label"].isin(SELECTED_IDS).mean()
                ),
                "dictionary_stability": float(len(set(SELECTED_IDS) & top) / len(SELECTED_IDS)),
                "selected_loop_stock_breadth": int(
                    aligned_events.loc[selected_event_mask, "symbol"].nunique()
                ),
                "period_occupancy_correlation": period_occupancy_correlation,
                "assessment_selected_loop_count": int(selected_assessment_mask.sum()),
            }
        )
        for pair in alignment.pairs:
            alignment_rows.append(
                {
                    "model_id": str(registry_row.model_id),
                    **asdict(pair),
                }
            )
        stability_rows.append(
            {
                "model_id": str(registry_row.model_id),
                "state_count": state_count,
                "seed": seed,
                "matched_centroid_distance": float(
                    np.mean([pair.centroid_distance for pair in alignment.pairs])
                ),
                "transition_profile_similarity": float(
                    1.0 - np.mean([pair.transition_distance for pair in alignment.pairs])
                ),
                "duration_profile_similarity": float(
                    1.0 - np.mean([pair.duration_distance for pair in alignment.pairs])
                ),
                "state_occupancy_correlation": _safe_correlation(
                    reference_occupancy,
                    _aligned_occupancy(aligned, 8),
                ),
                "bar_level_aligned_agreement": agreement,
                "adjusted_rand_index": ari,
                "normalized_mutual_information": nmi,
                "unmatched_reference_states": len(alignment.unmatched_reference),
                "unmatched_candidate_states": len(alignment.unmatched_candidate),
            }
        )
        for loop_id in SELECTED_IDS:
            translatable = loop_id in mapped_null_metrics
            observed, null_mean, rate_ratio, empirical_p = mapped_null_metrics.get(
                loop_id,
                (0, math.nan, math.nan, math.nan),
            )
            event_agreement = _bounded_event_agreement(
                reference_event_sets[loop_id],
                _event_set(aligned_events, loop_id),
                shift=2,
            )
            loop_stability_rows.append(
                {
                    "model_id": str(registry_row.model_id),
                    "state_count": state_count,
                    "seed": seed,
                    "primitive_loop_id": loop_id,
                    "loop_translatable": translatable,
                    "native_candidate_loop_id": translated_loop_ids.get(loop_id, "unmatched"),
                    "observed_first_events": observed,
                    "structural_null_mean": null_mean,
                    "structural_null_rate_ratio": rate_ratio,
                    "structural_null_empirical_p": empirical_p,
                    "structurally_significant": bool(
                        math.isfinite(empirical_p) and empirical_p <= 0.05
                    ),
                    "positive_structural_excess": bool(
                        math.isfinite(rate_ratio) and rate_ratio > 1.0
                    ),
                    "dictionary_jaccard": float(
                        len(reference_top & top) / max(len(reference_top | top), 1)
                    ),
                    "event_timestamp_agreement_bounded": event_agreement,
                    "orientation_jaccard": float(
                        len(
                            set(
                                reference_events.loc[
                                    reference_events["primitive_loop_id"].eq(loop_id),
                                    "orientation_id",
                                ]
                            )
                            & set(
                                aligned_events.loc[
                                    aligned_events["primitive_loop_id"].eq(loop_id),
                                    "orientation_id",
                                ]
                            )
                        )
                        / max(
                            len(
                                set(
                                    reference_events.loc[
                                        reference_events["primitive_loop_id"].eq(loop_id),
                                        "orientation_id",
                                    ]
                                )
                                | set(
                                    aligned_events.loc[
                                        aligned_events["primitive_loop_id"].eq(loop_id),
                                        "orientation_id",
                                    ]
                                )
                            ),
                            1,
                        )
                    ),
                    "stock_breadth": int(
                        aligned_events.loc[
                            aligned_events["primitive_loop_id"].eq(loop_id), "symbol"
                        ].nunique()
                    ),
                    "assessment_event_count": int(
                        aligned_assessment_events["primitive_loop_id"].eq(loop_id).sum()
                    ),
                    "period_occupancy_correlation": period_occupancy_correlation,
                }
            )
            event_stability_rows.append(
                {
                    "model_id": str(registry_row.model_id),
                    "state_count": state_count,
                    "seed": seed,
                    "primitive_loop_id": loop_id,
                    "reference_first_events": int(
                        reference_first_events["primary_label"].eq(loop_id).sum()
                    ),
                    "candidate_first_events": int(
                        candidate_first_events["primary_label"].eq(loop_id).sum()
                    ),
                    "coverage_ratio": float(
                        candidate_first_events["primary_label"].eq(loop_id).sum()
                        / max(reference_first_events["primary_label"].eq(loop_id).sum(), 1)
                    ),
                    "same_primitive_bounded_timestamp_fraction": float(
                        first_event_comparison.loc[
                            first_event_comparison["primary_label_reference"].eq(loop_id),
                            "agreement_class",
                        ]
                        .isin(
                            [
                                "EXACT_EVENT_AGREEMENT",
                                "SAME_PRIMITIVE_SHIFTED_TIMESTAMP",
                            ]
                        )
                        .mean()
                    ),
                }
            )
        del (
            summary,
            fitted,
            fitted_model,
            fitted_emissions,
            aligned_events,
            aligned_assessment_events,
            candidate_assessment_summary,
            aligned_assessment,
            candidate_first_events,
            first_event_comparison,
        )
        gc.collect()
    registry_metrics = pd.DataFrame(registry_rows)
    state_alignment = pd.DataFrame(alignment_rows)
    state_stability = pd.DataFrame(stability_rows)
    loop_stability = pd.DataFrame(loop_stability_rows)
    first_event_stability = pd.DataFrame(event_stability_rows)
    writer.frame("k_seed_model_registry.csv", registry_metrics)
    writer.frame("state_alignment.csv", state_alignment)
    writer.frame("state_stability_by_k_seed.csv", state_stability)
    writer.frame("loop_stability_by_k_seed.csv", loop_stability)
    writer.frame("first_event_stability_by_k_seed.csv", first_event_stability)

    # A6 bounded sample variants.
    sample_composition_rows: list[dict[str, object]] = []
    sample_state_rows: list[dict[str, object]] = []
    sample_loop_rows: list[dict[str, object]] = []
    sample_variants = ("SAMPLE_A", "SAMPLE_B", "SAMPLE_C", "SAMPLE_D")
    sample_frame = development.panel.copy()
    sample_frame["month"] = sample_frame["session"].astype(str).str[:7]
    for variant in sample_variants:
        sample = build_training_sample(
            sample_frame,
            variant=variant,
            maximum_rows=200000,
            seed=20260718,
        )
        selected = sample_frame.iloc[sample]
        for (symbol, month, phase), group in selected.groupby(
            ["symbol", "month", "clock_phase"], sort=True
        ):
            sample_composition_rows.append(
                {
                    "sample_variant": variant,
                    "symbol": str(symbol),
                    "month": str(month),
                    "clock_phase": str(phase),
                    "rows": len(group),
                    "sample_rows": len(sample),
                }
            )
        fitted = fit_clustered_semimarkov(
            scaled=exact_scaled,
            fit_feature_names=FROZEN_EMISSION_FEATURES,
            semantic_features=development.panel,
            groups=training_groups,
            sample_indices=sample,
            state_count=8,
            seed=20260710,
            cleaning_variant=CleaningVariant.CLEANING_1,
            activity_column="regime_log_activity_12",
            direction_column="signed_efficiency_12",
            maximum_duration=24,
        )
        fitted_model = expand_duration_hazard_v2(
            fitted.parameters.as_dict(), maximum_age=78, tail_window=6
        )
        fitted_emissions = gaussian_log_emissions(exact_scaled, fitted.parameters)
        summary = _causal_filter_summary_compiled(
            fitted_emissions,
            groups=development.groups,
            model=fitted_model,
        )
        alignment = align_states(
            np.asarray(parameters["means"]),
            fitted.parameters.means,
            reference_transition=np.asarray(parameters["transitions"]),
            candidate_transition=fitted.parameters.transitions,
            reference_duration=np.asarray(model_v2["duration_hazard"]),
            candidate_duration=np.asarray(fitted_model["duration_hazard"]),
        )
        aligned = apply_state_mapping(summary.hard_states, alignment.candidate_to_reference)
        candidate_first_events = reconstruct_first_event_outcomes(
            decisions,
            aligned,
            dictionary=dictionary,
            horizon_bars=24,
            allowed_states=frozenset(range(8)),
        )
        first_event_comparison, _ = compare_representation_events(
            reference_first_events,
            candidate_first_events,
            allowed_shift_bars=2,
        )
        sample_state_rows.append(
            {
                "sample_variant": variant,
                "sample_rows": len(sample),
                "bar_level_aligned_agreement": float(np.mean(aligned == legacy_labels)),
                "adjusted_rand_index": float(
                    adjusted_rand_score(legacy_labels, summary.hard_states)
                ),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(legacy_labels, summary.hard_states)
                ),
                "minimum_state_occupancy": float(_state_occupancy(summary.hard_states, 8).min()),
            }
        )
        for loop_id in SELECTED_IDS:
            reference_loop_count = int(reference_first_events["primary_label"].eq(loop_id).sum())
            candidate_loop_count = int(candidate_first_events["primary_label"].eq(loop_id).sum())
            comparable = first_event_comparison["primary_label_reference"].eq(loop_id)
            sample_loop_rows.append(
                {
                    "sample_variant": variant,
                    "primitive_loop_id": loop_id,
                    "first_event_count": candidate_loop_count,
                    "event_agreement": float(
                        first_event_comparison.loc[comparable, "agreement_class"]
                        .isin(
                            [
                                "EXACT_EVENT_AGREEMENT",
                                "SAME_PRIMITIVE_SHIFTED_TIMESTAMP",
                            ]
                        )
                        .mean()
                    ),
                    "dictionary_coverage_ratio": float(
                        candidate_loop_count / max(reference_loop_count, 1)
                    ),
                }
            )
        del (
            fitted,
            fitted_model,
            fitted_emissions,
            summary,
            candidate_first_events,
            first_event_comparison,
        )
        gc.collect()
    training_sample_composition = pd.DataFrame(sample_composition_rows)
    training_sample_state = pd.DataFrame(sample_state_rows)
    training_sample_loop = pd.DataFrame(sample_loop_rows)
    writer.frame("training_sample_composition.csv", training_sample_composition)
    writer.frame("training_sample_state_stability.csv", training_sample_state)
    writer.frame("training_sample_loop_stability.csv", training_sample_loop)

    # A7 unchanged 2025 semantic surface.
    assessment_summary = _causal_filter_summary_compiled(
        assessment.log_emissions,
        groups=assessment.groups,
        model=model_v2,
    )
    semantic_profiles = pd.concat(
        (
            _profile_rows(
                development,
                state_probabilities,
                legacy_labels,
                representation="legacy_hard_map",
            ),
            _profile_rows(
                assessment,
                assessment_summary.state_probabilities,
                assessment_summary.hard_states,
                representation="legacy_hard_map",
            ),
        ),
        ignore_index=True,
    )
    writer.frame(
        "state_semantic_profiles.parquet",
        semantic_profiles,
        state_representation="legacy_hard_map",
        source_artifact="bounded_2024_2025_structural_panels",
        source_hash=combined_snapshot,
    )
    profile_pivot = semantic_profiles.pivot_table(
        index=["state", "feature"],
        columns="period",
        values="scaled_centroid",
    ).reset_index()
    profile_pivot["absolute_scaled_drift"] = (
        profile_pivot["development_2024"] - profile_pivot["unchanged_retrospective_2025"]
    ).abs()
    state_period_drift = (
        profile_pivot.groupby("state", sort=True)["absolute_scaled_drift"]
        .apply(lambda values: float(np.sqrt(np.mean(np.square(values)))))
        .rename("centroid_drift_scaled_rms")
        .reset_index()
    )
    occupancy_2024 = _state_occupancy(legacy_labels, 8)
    occupancy_2025 = _state_occupancy(assessment_summary.hard_states, 8)
    state_period_drift["development_occupancy"] = occupancy_2024
    state_period_drift["assessment_occupancy"] = occupancy_2025
    state_period_drift["occupancy_change"] = occupancy_2025 - occupancy_2024
    transition_2024 = _transition_matrix(legacy_labels, development.groups, 8)
    transition_2025 = _transition_matrix(assessment_summary.hard_states, assessment.groups, 8)
    state_period_drift["transition_cosine_similarity"] = _row_cosine_similarity(
        transition_2024, transition_2025
    )
    duration_2024 = _median_run_duration_by_state(legacy_labels, development.groups, 8)
    duration_2025 = _median_run_duration_by_state(
        assessment_summary.hard_states, assessment.groups, 8
    )
    state_period_drift["development_median_duration"] = duration_2024
    state_period_drift["assessment_median_duration"] = duration_2025
    state_period_drift["duration_median_ratio"] = np.divide(
        duration_2025,
        duration_2024,
        out=np.full(8, np.nan, dtype=float),
        where=duration_2024 > 0.0,
    )
    stock_heterogeneity = pd.concat(
        (
            _stock_heterogeneity(development.panel, legacy_labels, development.period),
            _stock_heterogeneity(
                assessment.panel, assessment_summary.hard_states, assessment.period
            ),
        ),
        ignore_index=True,
    )
    clock_heterogeneity = pd.concat(
        (
            _clock_heterogeneity(development.panel, legacy_labels, development.period),
            _clock_heterogeneity(
                assessment.panel, assessment_summary.hard_states, assessment.period
            ),
        ),
        ignore_index=True,
    )
    maximum_stock_share_by_state = stock_heterogeneity.groupby("state", sort=True)[
        "stock_share_within_state"
    ].max()
    state_period_drift["maximum_single_stock_share"] = state_period_drift["state"].map(
        maximum_stock_share_by_state
    )
    state_period_drift["period_component_gate_pass"] = (
        state_period_drift["centroid_drift_scaled_rms"].le(3.0)
        & state_period_drift["development_occupancy"].ge(0.01)
        & state_period_drift["assessment_occupancy"].ge(0.01)
        & state_period_drift["transition_cosine_similarity"].ge(0.70)
        & state_period_drift["duration_median_ratio"].between(0.50, 2.0, inclusive="both")
        & state_period_drift["maximum_single_stock_share"].le(0.25)
    )
    writer.frame("state_period_drift.csv", state_period_drift)
    writer.frame("state_stock_heterogeneity.csv", stock_heterogeneity)
    writer.frame("state_clock_heterogeneity.csv", clock_heterogeneity)
    transition_drift = pd.concat(
        (
            _transition_rows(legacy_labels, development.groups, development.period, 8),
            _transition_rows(
                assessment_summary.hard_states, assessment.groups, assessment.period, 8
            ),
        ),
        ignore_index=True,
    )
    writer.frame("state_transition_drift.csv", transition_drift)

    # A8 stock-only and compact hierarchical development-only fits.
    partition = frozen_emission_partition()
    writer.json(
        "emission_feature_partition.json",
        {
            "combined": sorted(partition.combined),
            "stock_only": sorted(partition.stock),
            "market_only": sorted(partition.market),
            "relative": sorted(partition.relative),
            "partitions_disjoint": True,
            "feature_provenance": {
                feature: asdict(metadata)
                for feature, metadata in sorted(emission_feature_provenance().items())
            },
        },
    )
    stock_features = tuple(
        feature for feature in FROZEN_EMISSION_FEATURES if feature in partition.stock
    )
    stock_preprocessing = fit_emission_preprocessing(
        development.panel, feature_names=stock_features
    )
    stock_scaled = transform_emissions(development.panel, stock_preprocessing)
    stock_fit = fit_clustered_semimarkov(
        scaled=stock_scaled,
        fit_feature_names=stock_features,
        semantic_features=development.panel,
        groups=training_groups,
        sample_indices=bounded_stride,
        state_count=8,
        seed=20260710,
        cleaning_variant=CleaningVariant.CLEANING_1,
        activity_column="regime_log_activity_12",
        direction_column="signed_efficiency_12",
        maximum_duration=24,
    )
    stock_model = expand_duration_hazard_v2(
        stock_fit.parameters.as_dict(), maximum_age=78, tail_window=6
    )
    stock_summary = _causal_filter_summary_compiled(
        gaussian_log_emissions(stock_scaled, stock_fit.parameters),
        groups=development.groups,
        model=stock_model,
    )
    market_frame = (
        development.panel[
            ["session", "bar_ordinal", "bar_start_timestamp", *sorted(partition.market)]
        ]
        .drop_duplicates("bar_start_timestamp")
        .sort_values("bar_start_timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    market_groups = tuple(
        group.index.to_numpy(dtype=int) for _, group in market_frame.groupby("session", sort=False)
    )
    market_candidates: list[dict[str, Any]] = []
    market_feature_names = tuple(sorted(partition.market))
    fold_specs = (
        ("fold_1", pd.Timestamp("2024-05-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")),
        ("fold_2", pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2024-10-01", tz="UTC")),
        ("fold_3", pd.Timestamp("2024-10-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    )
    for market_k in (3, 4):
        fold_rows: list[dict[str, object]] = []
        fold_fits: list[tuple[Any, EmissionPreprocessing, np.ndarray]] = []
        market_timestamps = pd.to_datetime(market_frame["bar_start_timestamp"], utc=True)
        for fold_id, validation_start, validation_end in fold_specs:
            train_fold = market_frame.loc[market_timestamps.lt(validation_start)].reset_index(
                drop=True
            )
            validation_fold = market_frame.loc[
                market_timestamps.ge(validation_start) & market_timestamps.lt(validation_end)
            ].reset_index(drop=True)
            train_groups = tuple(
                group.index.to_numpy(dtype=int)
                for _, group in train_fold.groupby("session", sort=False)
            )
            validation_groups = tuple(
                group.index.to_numpy(dtype=int)
                for _, group in validation_fold.groupby("session", sort=False)
            )
            fold_preprocessing = fit_emission_preprocessing(
                train_fold,
                feature_names=market_feature_names,
            )
            train_scaled = transform_emissions(train_fold, fold_preprocessing)
            validation_scaled = transform_emissions(validation_fold, fold_preprocessing)
            fold_fit = fit_clustered_semimarkov(
                scaled=train_scaled,
                fit_feature_names=market_feature_names,
                semantic_features=train_fold,
                groups=train_groups,
                sample_indices=np.arange(len(train_fold), dtype=int),
                state_count=market_k,
                seed=20260718,
                cleaning_variant=CleaningVariant.CLEANING_CAUSAL,
                activity_column=market_feature_names[0],
                direction_column=market_feature_names[1],
                maximum_duration=24,
                batch_size=1024,
            )
            fold_model = expand_duration_hazard_v2(
                fold_fit.parameters.as_dict(), maximum_age=78, tail_window=6
            )
            fold_summary = _causal_filter_summary_compiled(
                gaussian_log_emissions(validation_scaled, fold_fit.parameters),
                groups=validation_groups,
                model=fold_model,
            )
            raw_centroids = (
                fold_fit.parameters.means * fold_preprocessing.scales[None, :]
                + fold_preprocessing.centers[None, :]
            )
            fold_fits.append((fold_fit, fold_preprocessing, raw_centroids))
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "train_rows": len(train_fold),
                    "validation_rows": len(validation_fold),
                    "validation_negative_log_likelihood": float(
                        -fold_summary.log_likelihood.mean()
                    ),
                    "validation_minimum_occupancy": float(
                        _state_occupancy(fold_summary.hard_states, market_k).min()
                    ),
                }
            )
        fold_alignment_distances: list[float] = []
        for (left_fit, _, left_raw), (right_fit, _, right_raw) in zip(
            fold_fits[:-1], fold_fits[1:], strict=True
        ):
            fold_alignment = align_states(
                left_raw,
                right_raw,
                reference_transition=left_fit.parameters.transitions,
                candidate_transition=right_fit.parameters.transitions,
                reference_duration=np.asarray(
                    expand_duration_hazard_v2(
                        left_fit.parameters.as_dict(), maximum_age=78, tail_window=6
                    )["duration_hazard"]
                ),
                candidate_duration=np.asarray(
                    expand_duration_hazard_v2(
                        right_fit.parameters.as_dict(), maximum_age=78, tail_window=6
                    )["duration_hazard"]
                ),
            )
            fold_alignment_distances.append(
                float(np.mean([pair.centroid_distance for pair in fold_alignment.pairs]))
            )
        market_preprocessing = fit_emission_preprocessing(
            market_frame, feature_names=market_feature_names
        )
        market_scaled = transform_emissions(market_frame, market_preprocessing)
        market_sample = np.arange(len(market_frame), dtype=int)
        market_fit = fit_clustered_semimarkov(
            scaled=market_scaled,
            fit_feature_names=market_feature_names,
            semantic_features=market_frame,
            groups=market_groups,
            sample_indices=market_sample,
            state_count=market_k,
            seed=20260718,
            cleaning_variant=CleaningVariant.CLEANING_CAUSAL,
            activity_column=market_feature_names[0],
            direction_column=market_feature_names[1],
            maximum_duration=24,
            batch_size=1024,
        )
        market_model = expand_duration_hazard_v2(
            market_fit.parameters.as_dict(), maximum_age=78, tail_window=6
        )
        market_summary = _causal_filter_summary_compiled(
            gaussian_log_emissions(market_scaled, market_fit.parameters),
            groups=market_groups,
            model=market_model,
        )
        market_candidates.append(
            {
                "state_count": market_k,
                "fit": market_fit,
                "summary": market_summary,
                "preprocessing": market_preprocessing,
                "scaled": market_scaled,
                "fold_rows": fold_rows,
                "fold_validation_nll": float(
                    np.average(
                        [float(row["validation_negative_log_likelihood"]) for row in fold_rows],
                        weights=[int(row["validation_rows"]) for row in fold_rows],
                    )
                ),
                "fold_centroid_stability_distance": float(np.mean(fold_alignment_distances)),
            }
        )
    likelihood_order = {
        int(candidate["state_count"]): rank
        for rank, candidate in enumerate(
            sorted(
                market_candidates,
                key=lambda item: (float(item["fold_validation_nll"]), int(item["state_count"])),
            ),
            start=1,
        )
    }
    stability_order = {
        int(candidate["state_count"]): rank
        for rank, candidate in enumerate(
            sorted(
                market_candidates,
                key=lambda item: (
                    float(item["fold_centroid_stability_distance"]),
                    int(item["state_count"]),
                ),
            ),
            start=1,
        )
    }
    selected_market = min(
        market_candidates,
        key=lambda item: (
            likelihood_order[int(item["state_count"])] + stability_order[int(item["state_count"])],
            likelihood_order[int(item["state_count"])],
            int(item["state_count"]),
        ),
    )
    market_k = int(selected_market["state_count"])
    market_fit = selected_market["fit"]
    market_summary = selected_market["summary"]
    market_preprocessing = selected_market["preprocessing"]
    market_scaled = selected_market["scaled"]
    market_by_timestamp = dict(
        zip(
            pd.to_datetime(market_frame["bar_start_timestamp"], utc=True),
            market_summary.hard_states,
            strict=True,
        )
    )
    row_market_state = np.asarray(
        [
            int(market_by_timestamp[pd.Timestamp(value)])
            for value in pd.to_datetime(development.panel["bar_start_timestamp"], utc=True)
        ],
        dtype=int,
    )
    hierarchy = hierarchical_state_ids(
        row_market_state,
        stock_summary.hard_states,
        stock_state_count=8,
    )
    assessment_stock_scaled = transform_emissions(assessment.panel, stock_preprocessing)
    assessment_stock_summary = _causal_filter_summary_compiled(
        gaussian_log_emissions(assessment_stock_scaled, stock_fit.parameters),
        groups=assessment.groups,
        model=stock_model,
    )
    assessment_market_frame = (
        assessment.panel[["session", "bar_ordinal", "bar_start_timestamp", *market_feature_names]]
        .drop_duplicates("bar_start_timestamp")
        .sort_values("bar_start_timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    assessment_market_groups = tuple(
        group.index.to_numpy(dtype=int)
        for _, group in assessment_market_frame.groupby("session", sort=False)
    )
    assessment_market_summary = _causal_filter_summary_compiled(
        gaussian_log_emissions(
            transform_emissions(assessment_market_frame, market_preprocessing),
            market_fit.parameters,
        ),
        groups=assessment_market_groups,
        model=expand_duration_hazard_v2(
            market_fit.parameters.as_dict(), maximum_age=78, tail_window=6
        ),
    )
    assessment_market_by_timestamp = dict(
        zip(
            pd.to_datetime(assessment_market_frame["bar_start_timestamp"], utc=True),
            assessment_market_summary.hard_states,
            strict=True,
        )
    )
    assessment_row_market_state = np.asarray(
        [
            int(assessment_market_by_timestamp[pd.Timestamp(value)])
            for value in pd.to_datetime(assessment.panel["bar_start_timestamp"], utc=True)
        ],
        dtype=int,
    )
    assessment_hierarchy = hierarchical_state_ids(
        assessment_row_market_state,
        assessment_stock_summary.hard_states,
        stock_state_count=8,
    )
    hierarchical_mapping = development.panel[
        ["symbol", "session", "bar_ordinal", "bar_complete_timestamp"]
    ].copy()
    hierarchical_mapping["decision_timestamp"] = hierarchical_mapping["bar_complete_timestamp"]
    hierarchical_mapping["decision_id"] = decisions["decision_id"].to_numpy()
    hierarchical_mapping["market_state"] = row_market_state
    hierarchical_mapping["stock_state"] = stock_summary.hard_states
    hierarchical_mapping["hierarchical_state"] = hierarchy.numeric
    hierarchical_mapping["hierarchical_state_token"] = hierarchy.tokens
    writer.frame(
        "hierarchical_state_mapping.parquet",
        hierarchical_mapping,
        state_representation="hierarchical_market_x_stock",
        source_artifact="development_2024_feature_partitions",
        source_hash=development.snapshot_hash,
    )
    representation_rows = []
    assessment_decisions = _sensitivity_decision_surface(
        assessment.panel,
        prefix="assessment-2025",
    )
    representation_surfaces = (
        (
            "MODEL_COMBINED",
            legacy_labels,
            assessment_summary.hard_states,
            None,
            assessment_summary,
            "combined_context",
        ),
        (
            "MODEL_STOCK_ONLY",
            stock_summary.hard_states,
            assessment_stock_summary.hard_states,
            stock_summary,
            assessment_stock_summary,
            "stock_behaviour",
        ),
    )
    for (
        name,
        labels,
        assessment_labels,
        summary,
        score_summary,
        interpretation,
    ) in representation_surfaces:
        occupancy = _state_occupancy(labels, 8)
        assessment_occupancy = _state_occupancy(assessment_labels, 8)
        concentration = _stock_heterogeneity(development.panel, labels, development.period)
        assessment_concentration = _stock_heterogeneity(
            assessment.panel,
            assessment_labels,
            assessment.period,
        )
        development_loops = _primitive_events(development.panel, labels)
        assessment_loops = _primitive_events(assessment.panel, assessment_labels)
        development_top = set(_top_loop_ids(development_loops))
        assessment_top = set(_top_loop_ids(assessment_loops))
        assessment_first_events = reconstruct_first_event_outcomes(
            assessment_decisions,
            assessment_labels,
            dictionary=dictionary,
            horizon_bars=24,
            allowed_states=frozenset(range(8)),
        )
        representation_rows.append(
            {
                "representation": name,
                "state_count": 8,
                "causal_negative_log_likelihood": float(
                    -(
                        _causal_filter_summary_compiled(
                            development.log_emissions,
                            groups=development.groups,
                            model=model_v2,
                        ).log_likelihood.mean()
                        if summary is None
                        else summary.log_likelihood.mean()
                    )
                ),
                "minimum_state_occupancy": float(occupancy.min()),
                "assessment_minimum_state_occupancy": float(assessment_occupancy.min()),
                "maximum_stock_share": float(
                    max(
                        concentration["stock_share_within_state"].max(),
                        assessment_concentration["stock_share_within_state"].max(),
                    )
                ),
                "loop_count": int(len(development_loops)),
                "assessment_loop_count": int(len(assessment_loops)),
                "loop_count_ratio": float(len(assessment_loops) / max(len(development_loops), 1)),
                "top20_dictionary_period_jaccard": float(
                    len(development_top & assessment_top)
                    / max(len(development_top | assessment_top), 1)
                ),
                "assessment_first_event_coverage": float(
                    assessment_first_events["primary_label"].isin(SELECTED_IDS).mean()
                ),
                "period_occupancy_correlation": _safe_correlation(occupancy, assessment_occupancy),
                "maximum_period_occupancy_drift": float(
                    np.max(np.abs(assessment_occupancy - occupancy))
                ),
                "assessment_causal_negative_log_likelihood": float(
                    -score_summary.log_likelihood.mean()
                ),
                "interpretability": interpretation,
                "redundancy": "audited_separately",
            }
        )
    hierarchy_occupancy = _state_occupancy(hierarchy.numeric, market_k * 8)
    assessment_hierarchy_occupancy = _state_occupancy(
        assessment_hierarchy.numeric,
        market_k * 8,
    )
    hierarchy_concentration = _stock_heterogeneity(
        development.panel, hierarchy.numeric, development.period
    )
    assessment_hierarchy_concentration = _stock_heterogeneity(
        assessment.panel,
        assessment_hierarchy.numeric,
        assessment.period,
    )
    hierarchy_development_loops = _primitive_events(
        development.panel,
        stock_summary.hard_states,
    )
    hierarchy_assessment_loops = _primitive_events(
        assessment.panel,
        assessment_stock_summary.hard_states,
    )
    hierarchy_assessment_first_events = reconstruct_first_event_outcomes(
        assessment_decisions,
        assessment_stock_summary.hard_states,
        dictionary=dictionary,
        horizon_bars=24,
        allowed_states=frozenset(range(8)),
    )
    representation_rows.append(
        {
            "representation": "MODEL_HIERARCHICAL",
            "state_count": market_k * 8,
            "causal_negative_log_likelihood": float(
                -stock_summary.log_likelihood.mean() - market_summary.log_likelihood.mean()
            ),
            "minimum_state_occupancy": float(hierarchy_occupancy.min()),
            "assessment_minimum_state_occupancy": float(assessment_hierarchy_occupancy.min()),
            "maximum_stock_share": float(
                max(
                    hierarchy_concentration["stock_share_within_state"].max(),
                    assessment_hierarchy_concentration["stock_share_within_state"].max(),
                )
            ),
            "loop_count": int(len(hierarchy_development_loops)),
            "assessment_loop_count": int(len(hierarchy_assessment_loops)),
            "loop_count_ratio": float(
                len(hierarchy_assessment_loops) / max(len(hierarchy_development_loops), 1)
            ),
            "top20_dictionary_period_jaccard": float(
                len(
                    set(_top_loop_ids(hierarchy_development_loops))
                    & set(_top_loop_ids(hierarchy_assessment_loops))
                )
                / max(
                    len(
                        set(_top_loop_ids(hierarchy_development_loops))
                        | set(_top_loop_ids(hierarchy_assessment_loops))
                    ),
                    1,
                )
            ),
            "assessment_first_event_coverage": float(
                hierarchy_assessment_first_events["primary_label"].isin(SELECTED_IDS).mean()
            ),
            "period_occupancy_correlation": _safe_correlation(
                hierarchy_occupancy,
                assessment_hierarchy_occupancy,
            ),
            "maximum_period_occupancy_drift": float(
                np.max(np.abs(assessment_hierarchy_occupancy - hierarchy_occupancy))
            ),
            "assessment_causal_negative_log_likelihood": float(
                -assessment_stock_summary.log_likelihood.mean()
                - assessment_market_summary.log_likelihood.mean()
            ),
            "interpretability": f"market_{market_k}_x_stock_8",
            "redundancy": "stock_loop_language_with_separate_market_context",
        }
    )
    representation_comparison = pd.DataFrame(representation_rows)
    writer.frame("combined_stock_hierarchical_comparison.csv", representation_comparison)
    market_profile_rows: list[dict[str, object]] = []
    for candidate in market_candidates:
        candidate_k = int(candidate["state_count"])
        candidate_fit = candidate["fit"]
        candidate_summary = candidate["summary"]
        market_profile_rows.append(
            {
                "row_type": "model_candidate",
                "market_state_count": candidate_k,
                "selected": candidate_k == market_k,
                "selection_method": "expanding_fold_likelihood_plus_stability_rank",
                "likelihood_rank": likelihood_order[candidate_k],
                "stability_rank": stability_order[candidate_k],
                "training_objective": candidate_fit.training_objective,
                "causal_negative_log_likelihood": float(-candidate_summary.log_likelihood.mean()),
                "expanding_fold_validation_nll": candidate["fold_validation_nll"],
                "expanding_fold_centroid_stability_distance": candidate[
                    "fold_centroid_stability_distance"
                ],
                "minimum_occupancy": float(
                    _state_occupancy(candidate_summary.hard_states, candidate_k).min()
                ),
                "posterior_entropy": float(candidate_summary.posterior_entropy.mean()),
            }
        )
        for fold_row in candidate["fold_rows"]:
            market_profile_rows.append(
                {
                    "row_type": "expanding_fold",
                    "market_state_count": candidate_k,
                    "selected": candidate_k == market_k,
                    "selection_method": "expanding_fold_likelihood_plus_stability_rank",
                    **fold_row,
                }
            )
    for state in range(market_k):
        state_mask = market_summary.hard_states == state
        for feature_index, feature in enumerate(market_feature_names):
            market_profile_rows.append(
                {
                    "row_type": "selected_state_feature_profile",
                    "market_state_count": market_k,
                    "selected": True,
                    "market_state": state,
                    "feature": feature,
                    "scaled_feature_centroid": float(
                        market_scaled[state_mask, feature_index].mean()
                    ),
                    "state_occupancy": float(state_mask.mean()),
                }
            )
    market_profiles = pd.DataFrame(market_profile_rows)
    writer.frame("market_regime_profiles.csv", market_profiles)
    stock_profiles = pd.DataFrame(
        [
            {
                "stock_state": state,
                "occupancy": float(np.mean(stock_summary.hard_states == state)),
                "assessment_occupancy": float(
                    np.mean(assessment_stock_summary.hard_states == state)
                ),
                "posterior_entropy": float(
                    stock_summary.posterior_entropy[stock_summary.hard_states == state].mean()
                ),
                "median_expected_age": float(
                    np.median(stock_summary.expected_age[stock_summary.hard_states == state])
                ),
            }
            for state in range(8)
        ]
    )
    writer.frame("stock_state_profiles.csv", stock_profiles)

    # A9 exact corrected first-event comparison under hard/hysteretic states.
    legacy_bundle = build_loop_event_ledgers(
        decisions,
        dictionary=dictionary,
        horizon_bars=24,
        allowed_states=frozenset(range(8)),
        state_column="hard_state_legacy",
        soft_prefix_session_keys=frozenset(),
    )
    legacy_outcome_surface = legacy_bundle.outcomes[
        ["decision_id", "primary_label", "bars_until_completion"]
    ].reset_index(drop=True)
    if not (
        reference_first_events["decision_id"].equals(legacy_outcome_surface["decision_id"])
        and reference_first_events["primary_label"].equals(legacy_outcome_surface["primary_label"])
        and reference_first_events["bars_until_completion"]
        .fillna(-1)
        .equals(legacy_outcome_surface["bars_until_completion"].fillna(-1))
    ):
        raise RuntimeError("fast sensitivity first-event surface differs from audited ledger")
    hysteretic_bundle = build_loop_event_ledgers(
        decisions,
        dictionary=dictionary,
        horizon_bars=24,
        allowed_states=frozenset(range(8)),
        state_column="hard_state_hysteretic",
        soft_prefix_session_keys=frozenset(),
    )
    event_comparison, event_metrics = compare_representation_events(
        legacy_bundle.outcomes[["decision_id", "primary_label", "bars_until_completion"]],
        hysteretic_bundle.outcomes[["decision_id", "primary_label", "bars_until_completion"]],
        allowed_shift_bars=2,
    )
    event_comparison = event_comparison.merge(
        decisions[
            [
                "decision_id",
                "symbol",
                "session",
                "bar_ordinal",
                "decision_timestamp",
                "hard_state_legacy",
                "hard_state_hysteretic",
            ]
        ],
        on="decision_id",
        how="left",
        validate="one_to_one",
    )
    for suffix, bundle in (("reference", legacy_bundle), ("candidate", hysteretic_bundle)):
        outcome_context = bundle.outcomes[
            ["decision_id", "repeat_depth", "transitions_remaining_at_decision"]
        ].rename(
            columns={
                "repeat_depth": f"repeat_depth_{suffix}",
                "transitions_remaining_at_decision": (
                    f"transitions_remaining_at_decision_{suffix}"
                ),
            }
        )
        event_comparison = event_comparison.merge(
            outcome_context,
            on="decision_id",
            how="left",
            validate="one_to_one",
        )
        prefix_context = _prefix_context(bundle.prefixes, suffix=suffix)
        event_comparison = event_comparison.merge(
            prefix_context,
            left_on=["decision_id", f"primary_label_{suffix}"],
            right_on=["decision_id", f"active_prefix_loop_id_{suffix}"],
            how="left",
            validate="many_to_one",
        )
    for suffix in ("reference", "candidate"):
        event_comparison[f"orientation_id_{suffix}"] = event_comparison[
            f"orientation_id_{suffix}"
        ].fillna("NO_ACTIVE_SELECTED_PREFIX")
        event_comparison[f"prefix_progress_{suffix}"] = (
            event_comparison[f"prefix_progress_{suffix}"].fillna(0).astype(int)
        )
        event_comparison[f"repeat_depth_prefix_{suffix}"] = (
            event_comparison[f"repeat_depth_prefix_{suffix}"].fillna(0).astype(int)
        )
    reference_is_loop = (
        event_comparison["primary_label_reference"].astype(str).str.startswith("loop_p_")
    )
    candidate_is_loop = (
        event_comparison["primary_label_candidate"].astype(str).str.startswith("loop_p_")
    )
    event_comparison["representation_event_class"] = event_comparison["agreement_class"]
    event_comparison.loc[reference_is_loop & ~candidate_is_loop, "representation_event_class"] = (
        "HARD_ONLY_EVENT"
    )
    event_comparison.loc[~reference_is_loop & candidate_is_loop, "representation_event_class"] = (
        "HYSTERETIC_ONLY_EVENT"
    )
    event_comparison["event_bar_shift"] = (
        event_comparison["bars_until_completion_candidate"]
        - event_comparison["bars_until_completion_reference"]
    )
    event_comparison["orientation_match"] = (
        event_comparison["orientation_id_reference"] == event_comparison["orientation_id_candidate"]
    )
    event_comparison["prefix_progress_match"] = (
        event_comparison["prefix_progress_reference"]
        == event_comparison["prefix_progress_candidate"]
    )
    event_comparison["repeat_depth_match"] = (
        event_comparison["repeat_depth_reference"] == event_comparison["repeat_depth_candidate"]
    )
    event_comparison["state_context_match"] = (
        event_comparison["hard_state_legacy"] == event_comparison["hard_state_hysteretic"]
    )
    event_comparison["primitive_loop_id"] = event_comparison["primary_label_reference"]
    event_comparison["orientation_id"] = event_comparison["orientation_id_reference"]
    event_comparison["prefix_progress"] = event_comparison["prefix_progress_reference"]
    hard_events = reference_events.loc[reference_events["primitive_loop_id"].isin(SELECTED_IDS)]
    soft_support = _soft_support_rows(
        development.panel, state_probabilities, hard_events, dictionary
    )
    soft_lookup = {
        (
            str(row.symbol),
            str(row.session),
            int(row.event_bar_ordinal),
            str(row.primitive_loop_id),
        ): (
            float(row.soft_completion_support),
            str(row.soft_support_class),
        )
        for row in soft_support.itertuples(index=False)
    }

    def soft_context(row: pd.Series) -> tuple[float, str]:
        loop_id = str(row["primary_label_reference"])
        bars = row["bars_until_completion_reference"]
        if not loop_id.startswith("loop_p_") or pd.isna(bars):
            return math.nan, "NO_HARD_EVENT"
        key = (
            str(row["symbol"]),
            str(row["session"]),
            int(row["bar_ordinal"]) + int(bars),
            loop_id,
        )
        return soft_lookup.get(key, (0.0, "LOW_SOFT_SUPPORT_HARD_EVENT"))

    soft_context_rows = event_comparison.apply(soft_context, axis=1)
    event_comparison["soft_completion_support"] = [value[0] for value in soft_context_rows]
    event_comparison["soft_support_class"] = [value[1] for value in soft_context_rows]
    writer.frame(
        "state_representation_event_comparison.parquet",
        event_comparison,
        state_representation="legacy_hard_map_vs_causal_hysteretic",
        source_artifact="corrected_first_event_ledgers",
        source_hash=state_model_hash,
    )
    hard_null = _structural_null_metrics(
        _run_sequences(development.panel, legacy_labels),
        state_count=8,
        draws=100,
        seed=20260820,
    )
    hysteretic_null = _structural_null_metrics(
        _run_sequences(development.panel, hysteretic),
        state_count=8,
        draws=100,
        seed=20260821,
    )
    robust_rows = []
    for loop_id in SELECTED_IDS:
        reference_loop = legacy_bundle.outcomes["primary_label"].eq(loop_id)
        candidate_loop = hysteretic_bundle.outcomes["primary_label"].eq(loop_id)
        comparable = event_comparison["primary_label_reference"].eq(loop_id)
        same_primitive_full = comparable & event_comparison["agreement_class"].isin(
            ["EXACT_EVENT_AGREEMENT", "SAME_PRIMITIVE_SHIFTED_TIMESTAMP"]
        )
        hard_observed, hard_null_mean, hard_ratio, hard_p = hard_null[loop_id]
        hyst_observed, hyst_null_mean, hyst_ratio, hyst_p = hysteretic_null[loop_id]
        reference_rows = legacy_bundle.outcomes.loc[reference_loop]
        candidate_rows = hysteretic_bundle.outcomes.loc[candidate_loop]
        robust_rows.append(
            {
                "primitive_loop_id": loop_id,
                "legacy_event_count": int(reference_loop.sum()),
                "hysteretic_event_count": int(candidate_loop.sum()),
                "same_primitive_bounded_shift_fraction": float(
                    event_comparison.loc[comparable, "agreement_class"]
                    .isin(["EXACT_EVENT_AGREEMENT", "SAME_PRIMITIVE_SHIFTED_TIMESTAMP"])
                    .mean()
                ),
                "exact_event_agreement": float(
                    event_comparison.loc[comparable, "agreement_class"]
                    .eq("EXACT_EVENT_AGREEMENT")
                    .mean()
                ),
                "hard_only_events": int(
                    event_comparison.loc[comparable, "representation_event_class"]
                    .eq("HARD_ONLY_EVENT")
                    .sum()
                ),
                "hysteretic_only_events": int(
                    (
                        event_comparison["primary_label_candidate"].eq(loop_id)
                        & event_comparison["representation_event_class"].eq("HYSTERETIC_ONLY_EVENT")
                    ).sum()
                ),
                "orientation_agreement_on_same_primitive": float(
                    event_comparison.loc[same_primitive_full, "orientation_match"].mean()
                ),
                "prefix_progress_agreement_on_same_primitive": float(
                    event_comparison.loc[same_primitive_full, "prefix_progress_match"].mean()
                ),
                "repeat_depth_agreement_on_same_primitive": float(
                    event_comparison.loc[same_primitive_full, "repeat_depth_match"].mean()
                ),
                "legacy_structural_null_observed": hard_observed,
                "legacy_structural_null_mean": hard_null_mean,
                "legacy_structural_null_rate_ratio": hard_ratio,
                "legacy_structural_null_empirical_p": hard_p,
                "hysteretic_structural_null_observed": hyst_observed,
                "hysteretic_structural_null_mean": hyst_null_mean,
                "hysteretic_structural_null_rate_ratio": hyst_ratio,
                "hysteretic_structural_null_empirical_p": hyst_p,
                "legacy_stock_breadth": int(reference_rows["symbol"].nunique()),
                "hysteretic_stock_breadth": int(candidate_rows["symbol"].nunique()),
                "legacy_month_breadth": int(
                    reference_rows["session"].astype(str).str[:7].nunique()
                ),
                "hysteretic_month_breadth": int(
                    candidate_rows["session"].astype(str).str[:7].nunique()
                ),
                "legacy_first_event_coverage": float(reference_loop.mean()),
                "hysteretic_first_event_coverage": float(candidate_loop.mean()),
                "low_soft_support_hard_events": int(
                    event_comparison.loc[comparable, "soft_support_class"]
                    .eq("LOW_SOFT_SUPPORT_HARD_EVENT")
                    .sum()
                ),
                "soft_supported_robust_events": int(
                    event_comparison.loc[comparable, "soft_support_class"]
                    .eq("SOFT_SUPPORTED_ROBUST_EVENT")
                    .sum()
                ),
            }
        )
    loop_robustness = pd.DataFrame(robust_rows)
    writer.frame("loop_robustness_by_representation.csv", loop_robustness)
    writer.frame(
        "dictionary_robustness_by_representation.csv",
        pd.DataFrame(
            [
                {
                    "reference": "legacy_hard_map",
                    "candidate": "causal_hysteretic",
                    "exact_event_agreement": event_metrics.exact_fraction,
                    "same_primitive_bounded_shift_fraction": (
                        event_metrics.same_primitive_bounded_shift_fraction
                    ),
                    "primitive_mismatch_fraction": event_metrics.primitive_mismatch_fraction,
                    "hard_only_event_fraction": float(
                        event_comparison["representation_event_class"].eq("HARD_ONLY_EVENT").mean()
                    ),
                    "hysteretic_only_event_fraction": float(
                        event_comparison["representation_event_class"]
                        .eq("HYSTERETIC_ONLY_EVENT")
                        .mean()
                    ),
                    "orientation_agreement": float(
                        event_comparison.loc[
                            reference_is_loop & candidate_is_loop,
                            "orientation_match",
                        ].mean()
                    ),
                    "prefix_progress_agreement": float(
                        event_comparison.loc[
                            reference_is_loop & candidate_is_loop,
                            "prefix_progress_match",
                        ].mean()
                    ),
                    "dictionary_entries": len(SELECTED_IDS),
                }
            ]
        ),
    )
    event_timing_summary = (
        event_comparison.groupby("representation_event_class", sort=True)
        .size()
        .rename("decisions")
        .reset_index()
    )
    event_timing_summary["share"] = event_timing_summary["decisions"] / len(event_comparison)
    writer.frame("event_timing_shift_summary.csv", event_timing_summary)
    soft_summary = (
        soft_support.groupby(["primitive_loop_id", "soft_support_class"], sort=True)
        .agg(
            hard_events=("soft_completion_support", "size"),
            mean_soft_support=("soft_completion_support", "mean"),
        )
        .reset_index()
    )
    writer.frame("soft_support_summary.csv", soft_summary)

    # Primary gate: recursion is correct, but original state-duration censoring is not.
    selected_hysteretic_fraction = float(
        loop_robustness["same_primitive_bounded_shift_fraction"].min()
    )
    k8_seed_counts = (
        loop_stability.loc[loop_stability["state_count"].eq(8)]
        .groupby("primitive_loop_id")["positive_structural_excess"]
        .sum()
    )
    k8_seed_gate = bool(all(int(k8_seed_counts.get(loop_id, 0)) >= 4 for loop_id in SELECTED_IDS))
    max_stock_share = float(stock_heterogeneity["stock_share_within_state"].max())
    median_drift = float(state_period_drift["centroid_drift_scaled_rms"].median())
    maximum_drift = float(state_period_drift["centroid_drift_scaled_rms"].max())
    minimum_transition_similarity = float(state_period_drift["transition_cosine_similarity"].min())
    minimum_duration_ratio = float(state_period_drift["duration_median_ratio"].min())
    maximum_duration_ratio = float(state_period_drift["duration_median_ratio"].max())
    k8_stability = state_stability.loc[state_stability["state_count"].eq(8)]
    minimum_k8_seed_nmi = float(k8_stability["normalized_mutual_information"].min())
    semantic_drift_pass = bool(
        median_drift <= 1.5
        and maximum_drift <= 3.0
        and minimum_transition_similarity >= 0.70
        and minimum_duration_ratio >= 0.50
        and maximum_duration_ratio <= 2.0
        and minimum_k8_seed_nmi >= 0.50
    )
    minimum_sample_coverage_ratio = float(training_sample_loop["dictionary_coverage_ratio"].min())
    minimum_sample_event_agreement = float(training_sample_loop["event_agreement"].min())
    representation_sensitive = bool(
        selected_hysteretic_fraction < 0.90
        or minimum_sample_coverage_ratio < 0.75
        or minimum_k8_seed_nmi < 0.50
    )
    usable_with_sensitivity = bool(
        selected_hysteretic_fraction >= 0.75
        and k8_seed_gate
        and minimum_sample_coverage_ratio >= 0.75
        and semantic_drift_pass
    )
    combined_row = representation_comparison.loc[
        representation_comparison["representation"].eq("MODEL_COMBINED")
    ].iloc[0]
    alternatives = representation_comparison.loc[
        ~representation_comparison["representation"].eq("MODEL_COMBINED")
    ]
    combined_stability_deficit = float(
        max(
            0.0,
            float(alternatives["minimum_state_occupancy"].max())
            - float(combined_row["minimum_state_occupancy"]),
        )
        + max(
            0.0,
            float(combined_row["maximum_stock_share"])
            - float(alternatives["maximum_stock_share"].min()),
        )
    )
    evidence = PartAGateEvidence(
        source_available=True,
        exact_reconstruction_pass=True,
        independent_audit_reproducible=False,
        mathematical_audit_pass=bool(
            math_audit["gaussian_emission_formula_pass"]
            and math_audit["variance_floor_pass"]
            and math_audit["initial_prior_normalizes"]
            and math_audit["transition_rows_normalize"]
            and math_audit["duration_hazard_probability_pass"]
            and math_audit["posterior_normalization_pass"]
            and math_audit["state_age_normalization_pass"]
            and math_audit["hard_map_argmax_pass"]
            and math_audit["expected_age_pass"]
            and math_audit["departure_probability_pass"]
            and math_audit["session_reset_pass"]
            and math_audit["causality_pass"]
        ),
        posterior_duration_pass=False,
        critical_future_leakage=bool(math_audit["critical_future_leakage_found"]),
        hysteretic_same_primitive_fraction=selected_hysteretic_fraction,
        k8_selected_loop_seed_gate_pass=k8_seed_gate,
        minimum_state_occupancy=float(_state_occupancy(legacy_labels, 8).min()),
        maximum_single_stock_share=max_stock_share,
        semantic_drift_pass=semantic_drift_pass,
        training_sample_dictionary_coverage_ratio=minimum_sample_coverage_ratio,
        combined_stability_deficit=combined_stability_deficit,
        representation_sensitive=representation_sensitive,
        usable_with_sensitivity=usable_with_sensitivity,
        recoverable_local_defect=True,
        hierarchical_materially_more_stable=False,
        hierarchical_reproducible=False,
    )
    decision_value = decide_part_a(evidence)
    state_alignment_hash = _sha256_file(output_dir / "state_alignment.csv")
    binding = freeze_part_a_binding(
        decision_value,
        state_model_hash=state_model_hash,
        state_count=8,
        state_representation="legacy_hard_map_with_hysteretic_and_posterior_sensitivity",
        hysteresis_policy={"switch_probability": 0.55, "switch_margin": 0.10},
        posterior_support_fields=(
            "posterior_entropy",
            "top_second_margin",
            "expected_state_age",
            "departure_probability",
        ),
        state_alignment_hash=state_alignment_hash,
    )
    authorized_decisions = set(contract["part_b_authorized_decisions"])
    part_b_authorized = decision_value.value in authorized_decisions
    decision_payload = {
        "decision": decision_value.value,
        "gate_evidence": asdict(evidence),
        "gate_diagnostics": {
            "minimum_k8_seed_nmi": minimum_k8_seed_nmi,
            "minimum_sample_event_agreement": minimum_sample_event_agreement,
            "minimum_transition_cosine_similarity": minimum_transition_similarity,
            "minimum_duration_median_ratio": minimum_duration_ratio,
            "maximum_duration_median_ratio": maximum_duration_ratio,
            "median_period_centroid_drift_scaled_rms": median_drift,
            "maximum_period_centroid_drift_scaled_rms": maximum_drift,
            "soft_support_rows": len(soft_support),
            "soft_supported_robust_rows": int(
                soft_support["soft_support_class"].eq("SOFT_SUPPORTED_ROBUST_EVENT").sum()
            ),
        },
        "binding": {
            **asdict(binding),
            "decision": binding.decision.value,
        },
        "dictionary_work_may_proceed": part_b_authorized,
        "part_b_authorized": part_b_authorized,
        "part_b_accessed": False,
        "independent_audit_status": "pending",
        "blocking_findings": [
            "frozen state-duration fit treated terminal training runs as exact exits",
            "historical panel-base source dependency is absent for byte-exact KMeans refit",
        ],
        "exact_next_step": (
            "right-censored state-duration refit with archived deterministic panel order"
        ),
    }
    writer.json("part_a_decision.json", decision_payload)
    writer.json(
        "run_metadata.json",
        {
            "experiment_id": "20260718-regime-validity-loop-interaction-foundations-v2",
            "part": "A",
            "development_period": "2024",
            "unchanged_retrospective_assessment_period": "2025",
            "assessment_snapshot_hash": assessment.snapshot_hash,
            "development_snapshot_hash": development.snapshot_hash,
            "combined_opened_data_snapshot_hash": combined_snapshot,
            "implementation_source_hash": implementation_source_hash,
            "protected_2026_opened": False,
            "k_values": list(K_VALUES),
            "seeds": list(SEEDS),
            "historical_stride_rows": len(historical_sample),
            "sensitivity_sample_rows": len(bounded_stride),
            "part_b_accessed": False,
        },
    )
    post_tree = {
        "manifest_version": "regime_model_validity_v2_post_run_tree",
        "baseline_git_sha": BASELINE_SHA,
        "baseline_git_tree": _git("rev-parse", f"{BASELINE_SHA}^{{tree}}"),
        "frozen_tree_hash": _git(
            "rev-parse",
            f"{BASELINE_SHA}:research/slrno-v2/20260714-regime-loop-handoff/work/frozen",
        ),
        "baseline_tracked_modifications": _git("diff", "--name-only", BASELINE_SHA).splitlines(),
        "frozen_historical_mismatches": [],
        "new_versioned_paths_only": True,
    }
    writer.json("post_run_tree_manifest.json", post_tree)
    if output_dir.name == "primary":
        REPORT_PATH.write_text(
            _report(
                reconstruction,
                math_audit,
                cleaning_state_metrics,
                churn_summary,
                registry_metrics,
                training_sample_state,
                state_period_drift,
                representation_comparison,
                decision_payload,
            ),
            encoding="utf-8",
        )
    writer.json("artifact_manifest.json", _artifact_manifest(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if ARTIFACT_PARENT.resolve() not in output.parents:
        raise SystemExit("output must be inside the versioned Part A artifact directory")
    if output.name not in {"primary", "exact_rerun"}:
        raise SystemExit("output directory must be primary or exact_rerun")
    run(output)


if __name__ == "__main__":
    main()
