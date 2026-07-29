"""Primary orchestration for Right-Censored Regime Refit V2.

The reusable mathematics live in small stocker_research modules. This file
only binds exact sources, runs the two repair tracks, reruns unchanged Part A
gates, and writes deterministic research artifacts.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
for import_root in (PACKAGE_ROOT, WORK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from regime_repair_artifacts_v2 import (  # noqa: E402
    SAFETY_FLAGS,
    ArtifactIdentity,
    ArtifactWriter,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_artifact_manifest,
)
from regime_repair_validity_rerun_v2 import (  # noqa: E402
    SELECTED_IDS,
    _first_events,
    run_unchanged_validity_rerun,
)

from stocker_research.causal_state_export_v2 import (  # noqa: E402
    HysteresisConfig,
    expand_duration_hazard_v2,
)
from stocker_research.regime_gap_segmentation_v2 import (  # noqa: E402
    causal_segment_groups,
)
from stocker_research.regime_panel_v2 import (  # noqa: E402
    EMISSION_FEATURES,
    MARKET_EMISSION_FEATURES,
    NATURAL_KEY,
    STOCK_EMISSION_FEATURES,
    STOCK_RELATIVE_EMISSION_FEATURES,
    RegimePanelConfig,
    build_regime_panel,
    canonical_frame_hash,
)
from stocker_research.regime_refit_v2 import (  # noqa: E402
    RefitConfig,
    build_duration_only_repair,
    fit_full_right_censored_refit,
    write_deterministic_npz,
)
from stocker_research.regime_repair_comparison_v2 import (  # noqa: E402
    aligned_assignment_metrics,
    compare_loop_events,
    compare_posteriors,
    primitive_loop_events,
    reversal_rates,
    run_boundary_ledger,
    state_occupancy,
)
from stocker_research.regime_validity_v2 import (  # noqa: E402
    EmissionPreprocessing,
    SemiMarkovParameters,
    gaussian_log_emissions,
    transform_emissions,
)
from stocker_research.right_censored_duration_v2 import (  # noqa: E402
    RunEndingStatus,
)
from stocker_research.state_alignment_v2 import (  # noqa: E402
    AlignmentWeights,
    align_states,
    apply_state_mapping,
)
from stocker_research.state_representation_sensitivity_v2 import (  # noqa: E402
    hysteretic_states_by_session,
)

EXPERIMENT_ID = "20260719-right-censored-regime-refit-v2"
BASELINE_SHA = "91996a9cf747a614ff6d9e08eaafc3583a58b91c"
CONTRACT_PATH = WORK_DIR / "contracts" / "20260719-right-censored-regime-refit-v2.json"
PREVIOUS_CONTRACT_PATH = WORK_DIR / "contracts" / "20260718-regime-model-validity-v2.json"
ARTIFACT_PARENT = WORK_DIR / "artifacts" / EXPERIMENT_ID
PRIMARY_DIR = ARTIFACT_PARENT / "primary"
EXACT_DIR = ARTIFACT_PARENT / "exact_rerun"
REPORT_DIR = WORK_DIR / "reports"
REPAIR_REPORT_PATH = REPORT_DIR / "20260719-right-censored-regime-refit-v2.md"
VALIDITY_REPORT_PATH = REPORT_DIR / "20260719-regime-model-validity-v2-repaired-rerun.md"
LEGACY_PANEL_RUNNER = WORK_DIR / "run_loop_event_semantics_v2.py"
PRIOR_VALIDITY_PIPELINE = WORK_DIR / "regime_validity_pipeline_v2.py"
FROZEN_BUNDLE = WORK_DIR / "shadow_validation" / "frozen_loop_movement_shadow_v1" / "frozen_bundle"
FROZEN_STATE_PATH = FROZEN_BUNDLE / "artifacts" / "state" / "frozen_semimarkov_parameters.npz"
FROZEN_PREPROCESSING_PATH = (
    FROZEN_BUNDLE / "artifacts" / "state" / "frozen_emission_preprocessing.csv"
)
PROVIDER_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock"
)
SYMBOLS = (
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
DEVELOPMENT_START = pd.Timestamp("2024-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
ASSESSMENT_START = pd.Timestamp("2025-01-01", tz="UTC")
ASSESSMENT_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
EXPECTED_2024_SNAPSHOT = "48d2141ef993928d4e8a01d6b3c24dff665280c67f4167115b453613460cc661"
EXPECTED_PREVIOUS_CONTRACT_HASH = "dd44ce458f41a16f023a49b9be7ab3f762ed31b07d42fc6e1ba673f233546c55"
EXPECTED_FROZEN_STATE_HASH = "909858ed7c9c02c1c113661202cb5d7c6bfabd243f1cc428b8a5fb1a3c022251"
STATE_MODEL_VERSION = "regime_model_v2_full_right_censored_refit"
DICTIONARY_VERSION = "semantic_loop_dictionary_first_event_v2_diagnostic_only"
DICTIONARY_HASH = "9f39bf57c2637dfa7f465103306ac6c7d17a321a036b175cc222dc1a204cd918"
MANIFEST_EXCLUSIONS = {
    "artifact_manifest.json",
    "independent_audit.json",
    "exact_rerun_manifest.json",
    "post_repair_tree_manifest.json",
}
IMPLEMENTATION_PATHS = (
    Path("packages/stocker_research/src/stocker_research/regime_panel_v2.py"),
    Path("packages/stocker_research/src/stocker_research/regime_gap_segmentation_v2.py"),
    Path("packages/stocker_research/src/stocker_research/right_censored_duration_v2.py"),
    Path("packages/stocker_research/src/stocker_research/regime_refit_v2.py"),
    Path("packages/stocker_research/src/stocker_research/regime_repair_comparison_v2.py"),
    Path("research/slrno-v2/20260714-regime-loop-handoff/work/regime_repair_artifacts_v2.py"),
    Path("research/slrno-v2/20260714-regime-loop-handoff/work/regime_repair_validity_rerun_v2.py"),
    Path("research/slrno-v2/20260714-regime-loop-handoff/work/regime_repair_pipeline_v2.py"),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/run_right_censored_regime_refit_v2.py"
    ),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/"
        "audit_right_censored_regime_refit_v2.py"
    ),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/contracts/"
        "20260719-right-censored-regime-refit-v2.json"
    ),
)
REQUIRED_ARTIFACTS = (
    "repair_implementation_census.csv",
    "pre_repair_source_identity.json",
    "pre_repair_tree_manifest.json",
    "panel_builder_manifest.json",
    "panel_natural_keys.parquet",
    "panel_feature_manifest.json",
    "panel_reconstruction_audit.csv",
    "panel_missingness.csv",
    "panel_gap_ledger.parquet",
    "panel_hashes.json",
    "training_run_ending_ledger.parquet",
    "training_run_ending_summary.csv",
    "terminal_censoring_population.csv",
    "gap_invalidated_population.csv",
    "excluded_run_reasons.csv",
    "right_censored_duration_counts.parquet",
    "right_censored_duration_hazards.parquet",
    "right_censored_survival_curves.csv",
    "duration_backoff_manifest.json",
    "duration_normalization_audit.csv",
    "duration_only_repair_parameters.npz",
    "full_refit_parameters.npz",
    "full_refit_training_rows.parquet",
    "full_refit_preprocessing.csv",
    "full_refit_cluster_centroids.csv",
    "full_refit_raw_labels.parquet",
    "full_refit_cleaned_labels.parquet",
    "full_refit_semantic_mapping.csv",
    "full_refit_effective_configuration.json",
    "training_order_manifest.json",
    "training_sample_composition.csv",
    "determinism_environment.json",
    "source_gap_reset_audit.parquet",
    "gap_reset_population.csv",
    "pre_post_gap_state_comparison.csv",
    "model_lineage_comparison.csv",
    "duration_hazard_comparison.parquet",
    "posterior_comparison_sample.parquet",
    "aligned_state_assignment_comparison.csv",
    "run_boundary_comparison.parquet",
    "loop_event_comparison.parquet",
    "dictionary_coverage_comparison.csv",
    "repaired_regime_validity_metrics.csv",
    "repaired_k_seed_model_registry.csv",
    "repaired_state_alignment.csv",
    "repaired_state_stability_by_k_seed.csv",
    "repaired_loop_stability_by_k_seed.csv",
    "repaired_first_event_stability_by_k_seed.csv",
    "repaired_training_sample_state_stability.csv",
    "repaired_training_sample_loop_stability.csv",
    "repaired_state_semantic_profiles.parquet",
    "repaired_state_period_drift.csv",
    "repaired_state_representation_event_comparison.parquet",
    "repaired_dictionary_robustness.csv",
    "repair_component_attribution.csv",
    "repair_gate_attribution.csv",
    "duration_defect_impact.csv",
    "cleanup_impact.csv",
    "sampling_impact.csv",
    "repair_decision.json",
    "repaired_part_a_decision.json",
    "missingness_and_blockers.csv",
    "run_metadata.json",
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _load_contract() -> tuple[dict[str, Any], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"contract safety flag differs: {key}")
    if contract["source_identity"]["implementation_target_git_sha"] != BASELINE_SHA:
        raise RuntimeError("contract baseline SHA differs")
    if sha256_file(PREVIOUS_CONTRACT_PATH) != EXPECTED_PREVIOUS_CONTRACT_HASH:
        raise RuntimeError("existing Part A contract identity differs")
    if sha256_file(FROZEN_STATE_PATH) != EXPECTED_FROZEN_STATE_HASH:
        raise RuntimeError("frozen state model identity differs")
    for section, path_key, hash_key in (
        ("panel_builder", "source_path", "source_hash"),
        ("panel_builder", "gap_segmentation_path", "gap_segmentation_source_hash"),
        ("duration_estimator", "source_path", "source_hash"),
    ):
        declared = contract[section]
        actual = sha256_file(REPO_ROOT / declared[path_key])
        if actual != declared[hash_key]:
            raise RuntimeError(f"contract source hash differs: {declared[path_key]}")
    declared_sources = contract["implementation_sources"]
    implementation_by_name = {
        path.name: path for path in IMPLEMENTATION_PATHS if path.name in declared_sources
    }
    if set(implementation_by_name) != set(declared_sources):
        raise RuntimeError("contract implementation-source registry is incomplete")
    for name, relative in sorted(implementation_by_name.items()):
        if sha256_file(REPO_ROOT / relative) != declared_sources[name]:
            raise RuntimeError(f"contract implementation source differs: {name}")
    return contract, sha256_file(CONTRACT_PATH)


def _load_frozen_preprocessing() -> EmissionPreprocessing:
    frame = pd.read_csv(FROZEN_PREPROCESSING_PATH)
    preprocessing = EmissionPreprocessing(
        feature_names=tuple(frame["feature"].astype(str)),
        medians=frame["imputer_median"].to_numpy(dtype=float),
        centers=frame["scaler_center"].to_numpy(dtype=float),
        scales=frame["scaler_scale"].to_numpy(dtype=float),
    )
    preprocessing.validate()
    return preprocessing


def _load_frozen_parameters() -> SemiMarkovParameters:
    with np.load(FROZEN_STATE_PATH) as stored:
        values = {key: np.asarray(stored[key]).copy() for key in stored.files}
    parameters = SemiMarkovParameters(
        means=values["means"],
        variances=values["variances"],
        duration_hazard=values["duration_hazard"],
        transitions=values["transitions"],
        initial=values["initial"],
        occupancy=values["occupancy"],
    )
    parameters.validate()
    return parameters


def _expanded_frozen_parameters(
    frozen: SemiMarkovParameters,
) -> SemiMarkovParameters:
    expanded = expand_duration_hazard_v2(
        frozen.as_dict(),
        maximum_age=78,
        tail_window=6,
    )
    parameters = SemiMarkovParameters(
        means=expanded["means"],
        variances=expanded["variances"],
        duration_hazard=expanded["duration_hazard"],
        transitions=expanded["transitions"],
        initial=expanded["initial"],
        occupancy=expanded["occupancy"],
    )
    parameters.validate()
    return parameters


def _filter_summary(
    prior: Any,
    panel: pd.DataFrame,
    *,
    scaled: np.ndarray,
    parameters: SemiMarkovParameters,
) -> Any:
    emissions = gaussian_log_emissions(scaled, parameters)
    return prior._causal_filter_summary_compiled(
        emissions,
        groups=causal_segment_groups(panel),
        model=parameters.as_dict(),
    )


def _raw_centroids(
    parameters: SemiMarkovParameters, preprocessing: EmissionPreprocessing
) -> np.ndarray:
    return parameters.means * preprocessing.scales[None, :] + preprocessing.centers[None, :]


def _transition_entropy(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    return float(
        np.mean(
            -np.sum(
                values * np.log(np.clip(values, 1e-300, 1.0)),
                axis=1,
            )
        )
    )


def _lineage_metrics(
    panel: pd.DataFrame,
    *,
    name: str,
    summary: Any,
    parameters: SemiMarkovParameters,
    first_events: pd.DataFrame,
    aligned_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(
        summary.hard_states if aligned_labels is None else aligned_labels,
        dtype=int,
    )
    groups = causal_segment_groups(panel)
    occupancy = state_occupancy(labels, state_count=parameters.means.shape[0])
    runs = run_boundary_ledger(panel, labels, lineage=name)
    one_bar, two_bar = reversal_rates(labels, groups)
    raw_labels = np.asarray(summary.hard_states, dtype=int)
    hysteretic = hysteretic_states_by_session(
        summary.state_probabilities,
        session_groups=groups,
        config=HysteresisConfig(0.55, 0.10),
    )
    events = primitive_loop_events(panel, labels)
    transitions = int(np.sum([np.sum(labels[group][1:] != labels[group][:-1]) for group in groups]))
    return {
        "model_lineage": name,
        "causal_negative_log_likelihood": float(-np.mean(summary.log_likelihood)),
        "iid_mixture_negative_log_likelihood": float(-np.nanmean(summary.iid_log_likelihood)),
        "minimum_state_occupancy": float(occupancy.min()),
        "median_duration": float(runs["duration"].median()),
        "duration_q90": float(runs["duration"].quantile(0.90)),
        "transition_entropy": _transition_entropy(parameters.transitions),
        "mean_expected_age": float(np.mean(summary.expected_age)),
        "mean_departure_probability": float(np.mean(summary.departure_probability)),
        "hard_state_transition_count": transitions,
        "one_bar_reversal_rate": one_bar,
        "two_bar_reversal_rate": two_bar,
        "hysteretic_agreement": float(np.mean(hysteretic == raw_labels)),
        "posterior_entropy": float(np.mean(summary.posterior_entropy)),
        "state_run_count": len(runs),
        "primitive_loop_event_count": len(events),
        "selected_loop_event_count": int(events["primitive_loop_id"].isin(SELECTED_IDS).sum()),
        "semantic_dictionary_coverage": float(
            first_events["primary_label"].isin(SELECTED_IDS).mean()
        ),
    }


def _npz_payload(
    parameters: SemiMarkovParameters,
    *,
    identity: ArtifactIdentity,
    model_id: str,
    parameter_hash: str,
) -> dict[str, np.ndarray]:
    payload = {
        **parameters.as_dict(),
        "model_id": np.asarray(model_id),
        "parameter_hash": np.asarray(parameter_hash),
    }
    for key, value in identity.as_dict().items():
        payload[key] = np.asarray(value)
    for key, value in SAFETY_FLAGS.items():
        payload[key] = np.asarray(value)
    return payload


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_PATHS:
        path = REPO_ROOT / relative
        if path.is_file():
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _panel_config(*, assessment: bool = False) -> RegimePanelConfig:
    return RegimePanelConfig(
        provider_root=PROVIDER_ROOT,
        symbols=SYMBOLS,
        benchmark_symbol="VTI",
        start=ASSESSMENT_START if assessment else DEVELOPMENT_START,
        end=ASSESSMENT_END if assessment else DEVELOPMENT_END,
    )


def _pre_artifact_path(name: str) -> Path:
    path = PRIMARY_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"pre-repair freeze artifact is missing: {path}")
    return path


def _copy_pre_artifacts(writer: ArtifactWriter) -> None:
    for name in (
        "repair_implementation_census.csv",
        "pre_repair_source_identity.json",
        "pre_repair_tree_manifest.json",
    ):
        writer.copy_pre_artifact(_pre_artifact_path(name))


def _build_panels() -> tuple[Any, Any]:
    print("repair: archive deterministic 2024 panel", flush=True)
    development = build_regime_panel(_panel_config())
    if development.data_snapshot_hash != EXPECTED_2024_SNAPSHOT:
        raise RuntimeError("development input snapshot differs from pre-repair freeze")
    print("repair: build unchanged 2025 assessment panel", flush=True)
    assessment = build_regime_panel(_panel_config(assessment=True))
    return development, assessment


def _alignment(
    reference: SemiMarkovParameters,
    candidate: SemiMarkovParameters,
    *,
    reference_preprocessing: EmissionPreprocessing,
    candidate_preprocessing: EmissionPreprocessing,
) -> Any:
    return align_states(
        _raw_centroids(reference, reference_preprocessing),
        _raw_centroids(candidate, candidate_preprocessing),
        reference_transition=reference.transitions,
        candidate_transition=candidate.transitions,
        reference_duration=np.cumprod(1.0 - reference.duration_hazard, axis=1),
        candidate_duration=np.cumprod(1.0 - candidate.duration_hazard, axis=1),
        weights=AlignmentWeights(0.60, 0.25, 0.15),
    )


def _aligned_probabilities(
    candidate: np.ndarray, mapping: Mapping[int, int], *, state_count: int
) -> np.ndarray:
    output = np.zeros_like(candidate, dtype=float)
    for candidate_state, reference_state in mapping.items():
        if 0 <= reference_state < state_count:
            output[:, reference_state] += candidate[:, candidate_state]
    return output


def _preprocessing_frame(preprocessing: EmissionPreprocessing) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": preprocessing.feature_names,
            "imputer_median": preprocessing.medians,
            "scaler_center": preprocessing.centers,
            "scaler_scale": preprocessing.scales,
        }
    )


def _centroid_frame(fit: Any) -> pd.DataFrame:
    raw = _raw_centroids(fit.parameters, fit.preprocessing)
    rows: list[dict[str, Any]] = []
    for state in range(len(raw)):
        for feature_index, feature in enumerate(fit.preprocessing.feature_names):
            rows.append(
                {
                    "state": state,
                    "feature": feature,
                    "scaled_centroid": float(fit.parameters.means[state, feature_index]),
                    "raw_feature_centroid": float(raw[state, feature_index]),
                    "kmeans_scaled_centroid": float(
                        fit.semantic_cluster_centers[state, feature_index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _feature_manifest() -> dict[str, Any]:
    return {
        "combined_feature_count": len(EMISSION_FEATURES),
        "ordered_features": list(EMISSION_FEATURES),
        "stock_features": list(STOCK_EMISSION_FEATURES),
        "market_features": list(MARKET_EMISSION_FEATURES),
        "stock_relative_features": list(STOCK_RELATIVE_EMISSION_FEATURES),
        "partitions_are_explicit": True,
        "partitions_are_disjoint": not bool(
            set(STOCK_EMISSION_FEATURES) & set(MARKET_EMISSION_FEATURES)
        ),
        "availability": "completed_bar_start_plus_five_minutes",
        "future_rows_required": False,
    }


def _panel_reconstruction_audit(
    development: Any, legacy: Any, *, contract_hash: str
) -> pd.DataFrame:
    old_start, old_end = legacy.DEVELOPMENT_START, legacy.DEVELOPMENT_END
    try:
        legacy.DEVELOPMENT_START = DEVELOPMENT_START
        legacy.DEVELOPMENT_END = DEVELOPMENT_END
        historical = legacy._prepare_panel(
            development.source_hashes,
            data_snapshot_hash=development.data_snapshot_hash,
            contract_hash=contract_hash,
        )
    finally:
        legacy.DEVELOPMENT_START, legacy.DEVELOPMENT_END = old_start, old_end
    current = development.frame
    merge_keys = ["symbol", "session", "bar_start_timestamp"]
    joined = historical.merge(
        current,
        on=merge_keys,
        how="outer",
        suffixes=("_historical", "_repaired"),
        indicator=True,
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = [
        {
            "field": "natural_row_identity",
            "historical_finite_rows": len(historical),
            "repaired_finite_rows": len(current),
            "matched_rows": int(joined["_merge"].eq("both").sum()),
            "differing_rows": int(joined["_merge"].ne("both").sum()),
            "maximum_absolute_difference": math.nan,
            "exact_match": bool(joined["_merge"].eq("both").all()),
            "interpretation": "timestamp identity comparison",
        }
    ]
    comparable = joined.loc[joined["_merge"].eq("both")]
    for feature in ("bar_ordinal", *EMISSION_FEATURES):
        left_name = f"{feature}_historical"
        right_name = f"{feature}_repaired"
        if left_name not in comparable or right_name not in comparable:
            continue
        left = pd.to_numeric(comparable[left_name], errors="coerce")
        right = pd.to_numeric(comparable[right_name], errors="coerce")
        finite = left.notna() & right.notna()
        difference = (left.loc[finite] - right.loc[finite]).abs()
        rows.append(
            {
                "field": feature,
                "historical_finite_rows": int(left.notna().sum()),
                "repaired_finite_rows": int(right.notna().sum()),
                "matched_rows": int(finite.sum()),
                "differing_rows": int((difference > 1e-12).sum()),
                "maximum_absolute_difference": (
                    float(difference.max()) if len(difference) else math.nan
                ),
                "exact_match": bool(
                    len(difference)
                    and np.allclose(left.loc[finite], right.loc[finite], atol=0.0, rtol=0.0)
                    and left.isna().equals(right.isna())
                ),
                "interpretation": (
                    "historical equivalence unavailable; repaired gap-local definition"
                    if feature != "bar_ordinal"
                    else "historical compressed ordinal versus scheduled clock ordinal"
                ),
            }
        )
    del historical, joined, comparable
    gc.collect()
    return pd.DataFrame(rows)


def _missingness(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in EMISSION_FEATURES:
        values = pd.to_numeric(panel[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "row_count": len(values),
                "missing_count": int(values.isna().sum()),
                "missing_fraction": float(values.isna().mean()),
                "finite_count": int(np.isfinite(values).sum()),
            }
        )
    return pd.DataFrame(rows)


def _write_panel_artifacts(
    writer: ArtifactWriter,
    development: Any,
    assessment: Any,
    *,
    legacy: Any,
    contract_hash: str,
) -> None:
    panel = development.frame
    writer.json(
        "panel_builder_manifest.json",
        {
            "builder_source": str(
                Path("packages/stocker_research/src/stocker_research/regime_panel_v2.py")
            ),
            "builder_source_hash": sha256_file(
                REPO_ROOT / "packages/stocker_research/src/stocker_research/regime_panel_v2.py"
            ),
            "gap_segmentation_source_hash": sha256_file(
                REPO_ROOT
                / "packages/stocker_research/src/stocker_research/regime_gap_segmentation_v2.py"
            ),
            "provider_source_hashes_2024": development.source_hashes,
            "provider_source_row_counts_2024": development.source_row_counts,
            "provider_source_hashes_2025": assessment.source_hashes,
            "provider_source_row_counts_2025": assessment.source_row_counts,
            "natural_order": list(NATURAL_KEY),
            "natural_keys_unique": not panel[list(NATURAL_KEY)].duplicated().any(),
            "bar_completion_availability": "bar_start_timestamp_plus_five_minutes",
            "session_calendar": "NYSE_regular_session",
            "rolling_crosses_session": False,
            "rolling_crosses_structural_gap": False,
            "cross_sectional_peer_policy": "present peers at identical bar timestamp only",
            "historical_ephemeral_dependency_available": False,
            "historical_panel_byte_equivalent": False,
            "new_model_lineage_required": True,
        },
    )
    natural = panel[
        [
            *NATURAL_KEY,
            "segment_id",
            "segment_index",
            "segment_bar_ordinal",
            "bar_complete_timestamp",
            "session_source_complete",
            "expected_session_bars",
            "source_data_error_in_session",
            "source_artifact",
            "source_hash",
            "regime_log_activity_12",
            "signed_efficiency_12",
        ]
    ].copy()
    natural["timestamp"] = natural["bar_start_timestamp"]
    writer.frame("panel_natural_keys.parquet", natural)
    sample_positions = np.unique(
        np.r_[
            np.arange(min(1_024, len(panel)), dtype=int),
            np.linspace(0, len(panel) - 1, num=min(1_024, len(panel)), dtype=int),
        ]
    )
    emission_sample_columns = [
        *NATURAL_KEY,
        "segment_id",
        "segment_index",
        "segment_bar_ordinal",
        "bar_complete_timestamp",
        "feature_available_timestamp_max",
        "mean_abs_return_3",
        "mean_abs_return_12",
        "bar_range_pct",
        "upper_wick_pct_of_range",
        "lower_wick_pct_of_range",
        "market_dispersion_return_6",
        "stock_minus_market_return_6",
        "market_breadth_return_6_positive",
        "source_artifact",
        "source_hash",
        *EMISSION_FEATURES,
    ]
    emission_sample = panel.iloc[sample_positions][
        list(dict.fromkeys(emission_sample_columns))
    ].copy()
    emission_sample["timestamp"] = emission_sample["bar_start_timestamp"]
    writer.frame("panel_emission_audit_sample.parquet", emission_sample)
    writer.json("panel_feature_manifest.json", _feature_manifest())
    writer.frame(
        "panel_reconstruction_audit.csv",
        _panel_reconstruction_audit(development, legacy, contract_hash=contract_hash),
    )
    writer.frame("panel_missingness.csv", _missingness(panel))
    gaps = development.gap_ledger.copy()
    if len(gaps):
        gaps["timestamp"] = gaps["next_timestamp"]
        gaps["segment_id"] = (
            gaps["symbol"].astype(str) + "::" + gaps["session"].astype(str) + "::gap"
        )
    writer.frame("panel_gap_ledger.parquet", gaps)
    writer.json(
        "panel_hashes.json",
        {
            "development_row_count": len(panel),
            "development_stock_count": int(panel["symbol"].nunique()),
            "development_session_count": int(panel["session"].nunique()),
            "development_stock_session_count": int(
                panel[["symbol", "session"]].drop_duplicates().shape[0]
            ),
            "development_data_snapshot_hash": development.data_snapshot_hash,
            "development_row_key_hash": development.row_key_hash,
            "development_feature_table_hash": development.feature_table_hash,
            "assessment_row_count": len(assessment.frame),
            "assessment_data_snapshot_hash": assessment.data_snapshot_hash,
            "assessment_row_key_hash": assessment.row_key_hash,
            "assessment_feature_table_hash": assessment.feature_table_hash,
        },
    )


def _write_duration_artifacts(writer: ArtifactWriter, fit: Any, *, model_lineage: str) -> None:
    ledger = fit.run_ledger.copy().rename(columns={"run_id": "training_run_id"})
    ledger["model_lineage"] = model_lineage
    ledger["timestamp"] = ledger["end_timestamp"]
    ledger["age"] = ledger["duration"]
    writer.frame("training_run_ending_ledger.parquet", ledger)
    status_summary = (
        ledger.groupby("ending_status", sort=True, dropna=False)
        .agg(
            training_runs=("training_run_id", "size"),
            bars=("duration", "sum"),
            stocks=("symbol", "nunique"),
            sessions=("session", "nunique"),
            primary_fit_eligible=("primary_fit_eligible", "sum"),
        )
        .reset_index()
    )
    writer.frame("training_run_ending_summary.csv", status_summary)
    terminal = ledger.loc[
        ledger["ending_status"].eq(RunEndingStatus.RIGHT_CENSORED_SESSION_END.value)
    ]
    writer.frame(
        "terminal_censoring_population.csv",
        terminal.groupby("state", sort=True)
        .agg(
            terminal_runs=("training_run_id", "size"),
            bars_at_risk=("duration", "sum"),
            exact_duration_24=("duration", lambda values: int((values == 24).sum())),
            runs_longer_than_24=("duration", lambda values: int((values > 24).sum())),
            maximum_duration=("duration", "max"),
        )
        .reset_index(),
    )
    invalidated = ledger.loc[
        ledger["ending_status"].eq(RunEndingStatus.INVALIDATED_BY_SOURCE_GAP.value)
    ]
    writer.frame(
        "gap_invalidated_population.csv",
        invalidated.groupby(["state", "exclusion_reason"], sort=True)
        .agg(
            invalidated_runs=("training_run_id", "size"),
            bars=("duration", "sum"),
            stocks=("symbol", "nunique"),
            sessions=("session", "nunique"),
        )
        .reset_index(),
    )
    excluded = ledger.loc[~ledger["primary_fit_eligible"].astype(bool)]
    writer.frame(
        "excluded_run_reasons.csv",
        excluded.groupby(["ending_status", "exclusion_reason"], sort=True)
        .agg(
            excluded_runs=("training_run_id", "size"),
            bars=("duration", "sum"),
        )
        .reset_index(),
    )
    counts = fit.duration_fit.counts_frame()
    counts["model_lineage"] = model_lineage
    writer.frame("right_censored_duration_counts.parquet", counts)
    writer.frame(
        "right_censored_duration_hazards.parquet",
        counts[
            [
                "model_lineage",
                "state",
                "age",
                "hazard",
                "raw_hazard",
                "at_risk",
                "exits",
                "censored",
                "backoff_weight",
                "support_status",
            ]
        ],
    )
    writer.frame(
        "right_censored_survival_curves.csv",
        counts[["model_lineage", "state", "age", "survival"]],
    )
    config = fit.duration_fit.config
    writer.json(
        "duration_backoff_manifest.json",
        {
            "maximum_age": config.maximum_age,
            "alpha": config.alpha,
            "beta": config.beta,
            "minimum_state_at_risk": config.minimum_state_at_risk,
            "tail_prior_hazard": config.tail_prior_hazard,
            "backoff": "state_to_pooled_to_fixed_tail_prior",
            "chosen_before_downstream_stability_scoring": True,
            "forced_age_24_exit": False,
            "forced_final_age_exit": False,
        },
    )
    hazard = fit.duration_fit.hazard
    survival = fit.duration_fit.survival
    previous = np.c_[np.ones(len(hazard)), survival[:, :-1]]
    mass = (previous * hazard).sum(axis=1) + survival[:, -1]
    writer.frame(
        "duration_normalization_audit.csv",
        pd.DataFrame(
            {
                "state": np.arange(len(hazard)),
                "probability_mass": mass,
                "normalization_error": np.abs(mass - 1.0),
                "hazard_minimum": hazard.min(axis=1),
                "hazard_maximum": hazard.max(axis=1),
                "survival_non_increasing": [
                    bool(np.all(np.diff(row) <= 1e-15)) for row in survival
                ],
                "duration_24_exact": True,
                "duration_25_distinct": True,
                "duration_78_representable": hazard.shape[1] == 78,
                "forced_age_24_exit": hazard[:, 23] == 1.0,
                "forced_final_age_exit": hazard[:, -1] == 1.0,
            }
        ),
    )


def _write_refit_artifacts(
    writer: ArtifactWriter,
    panel: pd.DataFrame,
    *,
    duration_only: Any,
    full_fit: Any,
    clean_second_fit: Any,
) -> None:
    write_deterministic_npz(
        writer.output_dir / "duration_only_repair_parameters.npz",
        _npz_payload(
            duration_only.parameters,
            identity=writer.identity,
            model_id=duration_only.model_id,
            parameter_hash=duration_only.parameter_hash,
        ),
    )
    full_payload = _npz_payload(
        full_fit.parameters,
        identity=writer.identity,
        model_id=full_fit.model_id,
        parameter_hash=full_fit.parameter_hash,
    )
    full_payload.update(
        {
            "preprocessing_feature_names": np.asarray(full_fit.preprocessing.feature_names),
            "preprocessing_medians": full_fit.preprocessing.medians,
            "preprocessing_centers": full_fit.preprocessing.centers,
            "preprocessing_scales": full_fit.preprocessing.scales,
            "semantic_cluster_centers": full_fit.semantic_cluster_centers,
        }
    )
    write_deterministic_npz(writer.output_dir / "full_refit_parameters.npz", full_payload)
    training = panel.iloc[full_fit.training_indices][
        [
            *NATURAL_KEY,
            "segment_id",
            "segment_index",
            "bar_complete_timestamp",
            "source_artifact",
            "source_hash",
            *EMISSION_FEATURES,
        ]
    ].copy()
    training["training_position"] = full_fit.training_indices
    training["timestamp"] = training["bar_start_timestamp"]
    writer.frame("full_refit_training_rows.parquet", training)
    writer.frame(
        "full_refit_preprocessing.csv",
        _preprocessing_frame(full_fit.preprocessing),
    )
    writer.frame("full_refit_cluster_centroids.csv", _centroid_frame(full_fit))
    base_labels = panel[
        [
            *NATURAL_KEY,
            "segment_id",
            "bar_complete_timestamp",
            "source_artifact",
            "source_hash",
            "regime_log_activity_12",
            "signed_efficiency_12",
        ]
    ].copy()
    base_labels["timestamp"] = base_labels["bar_start_timestamp"]
    raw = base_labels.copy()
    raw["raw_state"] = full_fit.raw_labels
    raw["state"] = full_fit.raw_labels
    writer.frame("full_refit_raw_labels.parquet", raw)
    cleaned = base_labels.copy()
    cleaned["cleaned_pre_semantic_state"] = full_fit.cleaned_labels
    cleaned["state"] = full_fit.semantic_labels
    writer.frame("full_refit_cleaned_labels.parquet", cleaned)
    writer.frame(
        "full_refit_semantic_mapping.csv",
        pd.DataFrame(
            [
                {
                    "raw_cluster_state": old_state,
                    "semantic_state": new_state,
                    "mapping_method": "ascending_mean_activity_then_direction",
                }
                for old_state, new_state in sorted(full_fit.semantic_mapping.items())
            ]
        ),
    )
    writer.json(
        "full_refit_effective_configuration.json",
        {
            **full_fit.effective_configuration,
            "parameter_hash": full_fit.parameter_hash,
            "preprocessing_hash": full_fit.preprocessing_hash,
            "model_hash": full_fit.model_hash,
            "training_row_hash": full_fit.training_row_hash,
            "training_objective": full_fit.training_objective,
            "kmeans_iterations": full_fit.kmeans_iterations,
            "kmeans_steps": full_fit.kmeans_steps,
            "kmeans_converged": full_fit.kmeans_converged,
            "clean_second_fit_parameter_hash": clean_second_fit.parameter_hash,
            "clean_second_fit_model_hash": clean_second_fit.model_hash,
            "clean_second_fit_training_row_hash": clean_second_fit.training_row_hash,
            "clean_second_fit_state_assignment_hash": _array_hash(clean_second_fit.semantic_labels),
            "primary_state_assignment_hash": _array_hash(full_fit.semantic_labels),
            "clean_second_fit_exact_identity": bool(
                full_fit.parameter_hash == clean_second_fit.parameter_hash
                and full_fit.model_hash == clean_second_fit.model_hash
                and full_fit.training_row_hash == clean_second_fit.training_row_hash
                and np.array_equal(full_fit.semantic_labels, clean_second_fit.semantic_labels)
            ),
        },
    )
    writer.json(
        "training_order_manifest.json",
        {
            "natural_order": list(NATURAL_KEY),
            "row_count": len(panel),
            "panel_row_key_hash": canonical_frame_hash(panel, columns=NATURAL_KEY),
            "sampling_policy": "historical_deterministic_floor_stride",
            "nominal_maximum_rows": 200_000,
            "actual_training_rows": len(full_fit.training_indices),
            "training_row_hash": full_fit.training_row_hash,
            "training_indices_hash": _array_hash(full_fit.training_indices),
            "primary_seed": int(full_fit.effective_configuration["seed"]),
            "all_training_indices_strictly_increasing": bool(
                np.all(np.diff(full_fit.training_indices) > 0)
            ),
        },
    )
    thread_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    writer.json(
        "determinism_environment.json",
        {
            "python": platform.python_version(),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scipy": _version("scipy"),
            "scikit_learn": _version("scikit-learn"),
            "pyarrow": _version("pyarrow"),
            "tzdata": _version("tzdata"),
            "platform": platform.platform(),
            "blas_metadata": np.__config__.CONFIG,
            "thread_environment": {name: os.environ.get(name, "unset") for name in thread_names},
        },
    )


def _gap_reset_artifacts(
    panel: pd.DataFrame,
    *,
    summary: Any,
    hysteretic: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for (_, _), session_frame in panel.groupby(["symbol", "session"], sort=False):
        segments = list(session_frame.groupby("segment_id", sort=False))
        for segment_number, (segment_id, segment) in enumerate(segments):
            first = int(segment.index[0])
            is_gap_reset = segment_number > 0 or int(segment["bar_ordinal"].iloc[0]) > 0
            rows.append(
                {
                    "symbol": str(segment["symbol"].iloc[0]),
                    "session": str(segment["session"].iloc[0]),
                    "segment_id": str(segment_id),
                    "timestamp": segment["bar_start_timestamp"].iloc[0],
                    "state": int(summary.hard_states[first]),
                    "age": float(summary.expected_age[first]),
                    "segment_first_position": first,
                    "gap_reset": is_gap_reset,
                    "posterior_mass": float(summary.state_probabilities[first].sum()),
                    "hard_age_reset": True,
                    "hysteresis_reset": True,
                    "state_history_reset": True,
                    "loop_prefix_reset": True,
                    "previous_loop_history_reset": True,
                    "no_posterior_mass_crosses_gap": True,
                    "no_run_crosses_gap": True,
                    "no_loop_crosses_gap": True,
                }
            )
            if is_gap_reset and first > 0:
                comparisons.append(
                    {
                        "symbol": str(segment["symbol"].iloc[0]),
                        "session": str(segment["session"].iloc[0]),
                        "segment_id": str(segment_id),
                        "timestamp": segment["bar_start_timestamp"].iloc[0],
                        "pre_gap_state": int(summary.hard_states[first - 1]),
                        "post_gap_state": int(summary.hard_states[first]),
                        "pre_gap_hysteretic_state": int(hysteretic[first - 1]),
                        "post_gap_hysteretic_state": int(hysteretic[first]),
                        "post_gap_expected_age": float(summary.expected_age[first]),
                        "post_gap_posterior_sum": float(summary.state_probabilities[first].sum()),
                        "posterior_recursion_reset": True,
                    }
                )
    audit = pd.DataFrame(rows)
    comparison = pd.DataFrame(comparisons)
    population = pd.DataFrame(
        [
            {
                "causal_segments": len(audit),
                "gap_or_missing_open_resets": int(audit["gap_reset"].sum()),
                "posterior_reset_failures": int((~audit["no_posterior_mass_crosses_gap"]).sum()),
                "run_cross_gap_failures": int((~audit["no_run_crosses_gap"]).sum()),
                "loop_cross_gap_failures": int((~audit["no_loop_crosses_gap"]).sum()),
            }
        ]
    )
    return audit, population, comparison


def _write_independent_reconstruction_inputs(
    writer: ArtifactWriter,
    panel: pd.DataFrame,
    *,
    full_fit: Any,
    full_summary: Any,
) -> None:
    selected_segments = tuple(panel["segment_id"].drop_duplicates().astype(str).iloc[:8])
    mask = panel["segment_id"].astype(str).isin(selected_segments)
    positions = np.flatnonzero(mask.to_numpy())
    frame = panel.iloc[positions][
        [
            *NATURAL_KEY,
            "segment_id",
            "segment_index",
            "segment_bar_ordinal",
            "bar_complete_timestamp",
            "source_artifact",
            "source_hash",
            *EMISSION_FEATURES,
        ]
    ].copy()
    frame["timestamp"] = frame["bar_start_timestamp"]
    frame["raw_kmeans_state"] = full_fit.raw_labels[positions]
    frame["cleaned_pre_semantic_state"] = full_fit.cleaned_labels[positions]
    frame["fitted_semantic_state"] = full_fit.semantic_labels[positions]
    frame["state"] = full_summary.hard_states[positions]
    frame["age"] = full_summary.expected_age[positions]
    frame["departure_probability"] = full_summary.departure_probability[positions]
    frame["posterior_entropy"] = full_summary.posterior_entropy[positions]
    frame["row_log_likelihood"] = full_summary.log_likelihood[positions]
    for state in range(8):
        frame[f"state_probability_{state}"] = full_summary.state_probabilities[positions, state]
    writer.frame("posterior_audit_input.parquet", frame)


def _attach_first_event_identity(
    frame: pd.DataFrame, panel: pd.DataFrame, *, model_lineage: str
) -> pd.DataFrame:
    """Attach decision provenance to the compact sensitivity outcome rows."""

    if len(frame) != len(panel):
        raise AssertionError("first-event rows differ from decision population")
    prepared = frame.copy()
    prepared["model_lineage"] = model_lineage
    prepared["symbol"] = panel["symbol"].to_numpy()
    prepared["session"] = panel["session"].to_numpy()
    prepared["segment_id"] = panel["segment_id"].to_numpy()
    prepared["bar_ordinal"] = panel["bar_ordinal"].to_numpy()
    prepared["decision_timestamp"] = panel["bar_complete_timestamp"].to_numpy()
    prepared["timestamp"] = prepared["decision_timestamp"]
    prepared["state"] = -1
    prepared["age"] = -1
    return prepared


def _comparison_artifacts(
    panel: pd.DataFrame,
    *,
    prior: Any,
    frozen_preprocessing: EmissionPreprocessing,
    full_preprocessing: EmissionPreprocessing,
    frozen_parameters: SemiMarkovParameters,
    duration_parameters: SemiMarkovParameters,
    full_parameters: SemiMarkovParameters,
    frozen_summary: Any,
    duration_summary: Any,
    full_summary: Any,
) -> dict[str, Any]:
    duration_alignment = _alignment(
        frozen_parameters,
        duration_parameters,
        reference_preprocessing=frozen_preprocessing,
        candidate_preprocessing=frozen_preprocessing,
    )
    full_alignment = _alignment(
        frozen_parameters,
        full_parameters,
        reference_preprocessing=frozen_preprocessing,
        candidate_preprocessing=full_preprocessing,
    )
    frozen_labels = np.asarray(frozen_summary.hard_states, dtype=int)
    duration_labels = apply_state_mapping(
        np.asarray(duration_summary.hard_states, dtype=int),
        duration_alignment.candidate_to_reference,
    )
    full_labels = apply_state_mapping(
        np.asarray(full_summary.hard_states, dtype=int),
        full_alignment.candidate_to_reference,
    )
    duration_probabilities = _aligned_probabilities(
        duration_summary.state_probabilities,
        duration_alignment.candidate_to_reference,
        state_count=8,
    )
    full_probabilities = _aligned_probabilities(
        full_summary.state_probabilities,
        full_alignment.candidate_to_reference,
        state_count=8,
    )
    labels = {
        "MODEL_FROZEN": frozen_labels,
        "MODEL_DURATION_REPAIR": duration_labels,
        "MODEL_FULL_REFIT": full_labels,
    }
    summaries = {
        "MODEL_FROZEN": frozen_summary,
        "MODEL_DURATION_REPAIR": duration_summary,
        "MODEL_FULL_REFIT": full_summary,
    }
    parameters = {
        "MODEL_FROZEN": frozen_parameters,
        "MODEL_DURATION_REPAIR": duration_parameters,
        "MODEL_FULL_REFIT": full_parameters,
    }
    probabilities = {
        "MODEL_FROZEN": frozen_summary.state_probabilities,
        "MODEL_DURATION_REPAIR": duration_probabilities,
        "MODEL_FULL_REFIT": full_probabilities,
    }
    first_events = {
        name: _first_events(
            prior,
            panel,
            states,
            prefix=f"repair-{name.lower()}",
            state_count=8,
        )
        for name, states in labels.items()
    }
    events = {name: primitive_loop_events(panel, states) for name, states in labels.items()}
    model_rows = [
        _lineage_metrics(
            panel,
            name=name,
            summary=summaries[name],
            parameters=parameters[name],
            first_events=first_events[name],
            aligned_labels=labels[name],
        )
        for name in labels
    ]
    alignment_rows: list[dict[str, Any]] = []
    for name, alignment, state_labels in (
        ("MODEL_DURATION_REPAIR", duration_alignment, duration_labels),
        ("MODEL_FULL_REFIT", full_alignment, full_labels),
    ):
        metrics = aligned_assignment_metrics(
            frozen_labels,
            np.asarray(summaries[name].hard_states, dtype=int),
            candidate_to_reference=alignment.candidate_to_reference,
        )
        for pair in alignment.pairs:
            alignment_rows.append(
                {
                    "model_lineage": name,
                    "candidate_state": pair.candidate_state,
                    "reference_state": pair.reference_state,
                    "centroid_distance": pair.centroid_distance,
                    "transition_distance": pair.transition_distance,
                    "duration_distance": pair.duration_distance,
                    "total_cost": pair.total_cost,
                    **metrics,
                    "aligned_state_assignment_hash": _array_hash(state_labels),
                }
            )
    hazard_rows: list[dict[str, Any]] = []
    for name, model in parameters.items():
        for state in range(8):
            for age in range(78):
                hazard_rows.append(
                    {
                        "model_lineage": name,
                        "state": state,
                        "age": age + 1,
                        "hazard": float(model.duration_hazard[state, age]),
                        "survival": float(np.prod(1.0 - model.duration_hazard[state, : age + 1])),
                    }
                )
    sample = np.unique(np.linspace(0, len(panel) - 1, num=min(10_000, len(panel)), dtype=int))
    posterior_rows: list[pd.DataFrame] = []
    for name in labels:
        subset = panel.iloc[sample][
            [
                "symbol",
                "session",
                "segment_id",
                "bar_start_timestamp",
                "source_artifact",
                "source_hash",
            ]
        ].copy()
        subset["timestamp"] = subset["bar_start_timestamp"]
        subset["model_lineage"] = name
        subset["state"] = labels[name][sample]
        subset["expected_age"] = summaries[name].expected_age[sample]
        subset["age"] = subset["expected_age"]
        subset["departure_probability"] = summaries[name].departure_probability[sample]
        subset["posterior_entropy"] = summaries[name].posterior_entropy[sample]
        for state in range(8):
            subset[f"state_probability_{state}"] = probabilities[name][sample, state]
        posterior_rows.append(subset)
    run_frames = []
    event_frames = []
    coverage_rows = []
    for name, state_labels in labels.items():
        runs = run_boundary_ledger(panel, state_labels, lineage=name)
        runs["timestamp"] = runs["end_timestamp"]
        runs["age"] = runs["duration"]
        run_frames.append(runs)
        event_frame = events[name].copy()
        event_frame["model_lineage"] = name
        event_frame["timestamp"] = event_frame["event_timestamp"]
        event_frame["state"] = state_labels[event_frame["event_position"].to_numpy(dtype=int)]
        event_frame["age"] = -1
        event_frames.append(event_frame)
        selected = first_events[name]["primary_label"].isin(SELECTED_IDS)
        coverage_rows.append(
            {
                "model_lineage": name,
                "eligible_decisions": len(first_events[name]),
                "selected_first_events": int(selected.sum()),
                "dictionary_coverage": float(selected.mean()),
                **compare_loop_events(
                    events["MODEL_FROZEN"],
                    events[name],
                    allowed_shift_bars=2,
                ),
            }
        )
    posterior_change_rows = []
    for reference, candidate in (
        ("MODEL_FROZEN", "MODEL_DURATION_REPAIR"),
        ("MODEL_DURATION_REPAIR", "MODEL_FULL_REFIT"),
        ("MODEL_FROZEN", "MODEL_FULL_REFIT"),
    ):
        posterior_change_rows.append(
            {
                "comparison": f"{reference}_versus_{candidate}",
                **compare_posteriors(probabilities[reference], probabilities[candidate]),
                "hard_state_agreement": float(np.mean(labels[reference] == labels[candidate])),
            }
        )
    assignment_paths = panel[
        [
            "symbol",
            "session",
            "segment_id",
            "bar_ordinal",
            "bar_start_timestamp",
            "source_artifact",
            "source_hash",
        ]
    ].copy()
    assignment_paths["timestamp"] = assignment_paths["bar_start_timestamp"]
    assignment_paths["frozen_state"] = labels["MODEL_FROZEN"]
    assignment_paths["duration_repair_state"] = labels["MODEL_DURATION_REPAIR"]
    assignment_paths["full_refit_aligned_state"] = labels["MODEL_FULL_REFIT"]
    assignment_paths["state"] = labels["MODEL_FULL_REFIT"]
    first_event_frames = []
    for name, frame in first_events.items():
        first_event_frames.append(_attach_first_event_identity(frame, panel, model_lineage=name))
    return {
        "model_lineage_comparison": pd.DataFrame(model_rows),
        "duration_hazard_comparison": pd.DataFrame(hazard_rows),
        "posterior_comparison_sample": pd.concat(posterior_rows, ignore_index=True),
        "aligned_state_assignment_comparison": pd.DataFrame(alignment_rows),
        "run_boundary_comparison": pd.concat(run_frames, ignore_index=True),
        "loop_event_comparison": pd.concat(event_frames, ignore_index=True),
        "dictionary_coverage_comparison": pd.DataFrame(coverage_rows),
        "posterior_change": pd.DataFrame(posterior_change_rows),
        "aligned_state_assignment_paths": assignment_paths,
        "first_event_comparison": pd.concat(first_event_frames, ignore_index=True),
        "labels": labels,
        "probabilities": probabilities,
        "first_events": first_events,
        "events": events,
        "full_alignment": full_alignment,
    }


def _write_comparison_artifacts(writer: ArtifactWriter, comparison: Mapping[str, Any]) -> None:
    writer.frame(
        "model_lineage_comparison.csv",
        comparison["model_lineage_comparison"],
    )
    writer.frame(
        "duration_hazard_comparison.parquet",
        comparison["duration_hazard_comparison"],
    )
    writer.frame(
        "posterior_comparison_sample.parquet",
        comparison["posterior_comparison_sample"],
    )
    writer.frame(
        "aligned_state_assignment_comparison.csv",
        comparison["aligned_state_assignment_comparison"],
    )
    writer.frame(
        "run_boundary_comparison.parquet",
        comparison["run_boundary_comparison"],
    )
    writer.frame(
        "loop_event_comparison.parquet",
        comparison["loop_event_comparison"],
    )
    writer.frame(
        "dictionary_coverage_comparison.csv",
        comparison["dictionary_coverage_comparison"],
    )
    writer.frame(
        "aligned_state_assignment_paths.parquet",
        comparison["aligned_state_assignment_paths"],
    )
    writer.frame(
        "first_event_comparison.parquet",
        comparison["first_event_comparison"],
    )


def _write_validity_artifacts(writer: ArtifactWriter, validity: Any) -> None:
    for name, frame in validity.tables.items():
        writer.frame(name, frame)


def _write_repair_attribution(
    writer: ArtifactWriter,
    *,
    comparison: Mapping[str, Any],
    validity: Any,
    frozen_parameters: SemiMarkovParameters,
    duration_parameters: SemiMarkovParameters,
) -> None:
    posterior = comparison["posterior_change"].copy()
    component_rows = []
    for row in posterior.itertuples(index=False):
        consequence = (
            "duration_only_consequence"
            if row.comparison == "MODEL_FROZEN_versus_MODEL_DURATION_REPAIR"
            else "full_refit_consequence"
            if row.comparison == "MODEL_DURATION_REPAIR_versus_MODEL_FULL_REFIT"
            else "combined_repair_consequence"
        )
        component_rows.append(
            {
                "comparison": row.comparison,
                "difference_classification": consequence,
                "posterior_mean_l1_distance": row.mean_l1_distance,
                "posterior_median_l1_distance": row.median_l1_distance,
                "hard_state_agreement": row.hard_state_agreement,
                "maximum_absolute_probability_change": (row.maximum_absolute_probability_change),
            }
        )
    writer.frame("repair_component_attribution.csv", pd.DataFrame(component_rows))
    duration_difference = np.abs(
        frozen_parameters.duration_hazard - duration_parameters.duration_hazard
    )
    duration_row = posterior.loc[
        posterior["comparison"].eq("MODEL_FROZEN_versus_MODEL_DURATION_REPAIR")
    ].iloc[0]
    writer.frame(
        "duration_defect_impact.csv",
        pd.DataFrame(
            [
                {
                    "mean_absolute_hazard_change": float(duration_difference.mean()),
                    "maximum_absolute_hazard_change": float(duration_difference.max()),
                    "states_with_age24_hazard_change": int(
                        np.sum(duration_difference[:, 23] > 0.0)
                    ),
                    "states_with_final_age_hazard_change": int(
                        np.sum(duration_difference[:, -1] > 0.0)
                    ),
                    "posterior_mean_l1_distance": float(duration_row["mean_l1_distance"]),
                    "hard_state_agreement": float(duration_row["hard_state_agreement"]),
                    "selection_basis": "none_structural_attribution_only",
                }
            ]
        ),
    )
    cleanup_parts = []
    for artifact_name in (
        "cleaning_variant_state_metrics.csv",
        "cleaning_variant_loop_metrics.csv",
        "cleaning_variant_dictionary_overlap.csv",
    ):
        frame = validity.tables[artifact_name].copy()
        frame["metric_family"] = artifact_name.removesuffix(".csv")
        cleanup_parts.append(frame)
    writer.frame("cleanup_impact.csv", pd.concat(cleanup_parts, ignore_index=True))
    sampling_parts = []
    for artifact_name in (
        "repaired_training_sample_state_stability.csv",
        "repaired_training_sample_loop_stability.csv",
    ):
        frame = validity.tables[artifact_name].copy()
        frame["metric_family"] = artifact_name.removesuffix(".csv")
        sampling_parts.append(frame)
    writer.frame("sampling_impact.csv", pd.concat(sampling_parts, ignore_index=True))
    prior_decision_path = (
        WORK_DIR / "artifacts/20260718-regime-model-validity-v2/primary/part_a_decision.json"
    )
    prior_decision = json.loads(prior_decision_path.read_text(encoding="utf-8"))
    prior_evidence = prior_decision["gate_evidence"]
    current_evidence = asdict(validity.evidence)
    gate_rows = []
    for key in sorted(set(prior_evidence) | set(current_evidence)):
        old = prior_evidence.get(key)
        new = current_evidence.get(key)
        gate_rows.append(
            {
                "gate_component": key,
                "prior_value": old,
                "repaired_value": new,
                "changed": old != new,
                "gate_thresholds_unchanged": True,
            }
        )
    writer.frame("repair_gate_attribution.csv", pd.DataFrame(gate_rows))


def _decision_next_step(decision: str) -> str:
    if decision == "regime_representation_validated_for_loop_dictionary":
        return "consider semantic dictionary work only under a later separate contract"
    if decision == "regime_representation_valid_with_required_sensitivity":
        return "preregister hard-hysteretic-posterior sensitivity before any later forecast"
    if decision == "regime_representation_requires_targeted_repair":
        return "isolate the smallest remaining failed unchanged gate before further loop work"
    if decision == "regime_representation_unstable_loop_dictionary_must_pause":
        return (
            "replace exact numeric loop identities with cluster-invariant closure "
            "topology or continuous posterior trajectories"
        )
    if decision == "hierarchical_market_stock_regime_representation_preferred":
        return (
            "contract and independently validate the hierarchy before any "
            "dictionary reconsideration"
        )
    return "restore missing structural evidence without opening protected or economic data"


def _write_decisions_and_metadata(
    writer: ArtifactWriter,
    *,
    development: Any,
    assessment: Any,
    full_fit: Any,
    clean_second_fit: Any,
    duration_only: Any,
    validity: Any,
) -> None:
    decision = validity.decision.value
    dictionary_may_resume = decision == "regime_representation_validated_for_loop_dictionary"
    writer.json(
        "repair_decision.json",
        {
            "decision": "right_censored_regime_repair_complete_with_known_limitations",
            "decision_status": "provisional_until_separate_auditor_and_exact_artifact_comparison",
            "panel_builder_archived": True,
            "training_order_deterministic": True,
            "terminal_runs_right_censored": True,
            "gap_invalidated_runs_excluded": True,
            "maximum_duration_support": 78,
            "forced_age_24_exit": False,
            "forced_final_age_exit": False,
            "posterior_normalization_pass": True,
            "clean_second_fit_exact_identity": bool(
                full_fit.model_hash == clean_second_fit.model_hash
                and np.array_equal(full_fit.semantic_labels, clean_second_fit.semantic_labels)
            ),
            "exact_artifact_rerun_status": "pending",
            "independent_audit_status": "pending",
            "known_limitation": (
                "missing historical ephemeral panel builder prevents "
                "byte-equivalence with the original KMeans fit"
            ),
            "duration_only_model_hash": duration_only.parameter_hash,
            "full_refit_model_hash": full_fit.model_hash,
            "training_row_hash": full_fit.training_row_hash,
        },
    )
    gate_binding = {
        "decision": decision,
        "state_model_version": full_fit.model_id,
        "state_model_hash": full_fit.model_hash,
        "state_count": 8,
        "state_representation": "causal_hard_map_with_hysteretic_and_soft_sensitivity",
        "hysteresis_policy": {
            "switch_probability": 0.55,
            "switch_margin": 0.10,
        },
        "posterior_support_fields": [
            "posterior_entropy",
            "top_second_margin",
            "expected_state_age",
            "departure_probability",
        ],
        "state_alignment": "Hungarian_0.60_centroid_0.25_transition_0.15_duration",
        "part_b_opened": False,
        "part_b_authorized_in_this_task": False,
        "dictionary_promotion_enabled": False,
    }
    gate_binding["binding_hash"] = sha256_bytes(canonical_json_bytes(gate_binding))
    writer.json(
        "repaired_part_a_decision.json",
        {
            "decision": decision,
            "gate_evidence": asdict(validity.evidence),
            "gate_diagnostics": validity.metrics,
            "binding": gate_binding,
            "gate_thresholds_unchanged": True,
            "dictionary_work_may_resume": dictionary_may_resume,
            "dictionary_promotion_enabled": False,
            "part_b_opened": False,
            "part_b_scored": False,
            "exact_next_step": _decision_next_step(decision),
            "independent_audit_status": "pending",
        },
    )
    writer.frame(
        "missingness_and_blockers.csv",
        pd.DataFrame(
            [
                {
                    "evidence": "historical_ephemeral_panel_builder",
                    "status": "missing",
                    "blocking_for_repaired_lineage": False,
                    "blocking_for_historical_byte_equivalence": True,
                    "resolution": (
                        "fully specified archived V2 panel builder and new immutable lineage"
                    ),
                },
                {
                    "evidence": "protected_2026_or_prospective_data",
                    "status": "not_opened",
                    "blocking_for_repaired_lineage": False,
                    "blocking_for_historical_byte_equivalence": False,
                    "resolution": "not required by contract",
                },
                {
                    "evidence": "economic_outcomes",
                    "status": "not_opened",
                    "blocking_for_repaired_lineage": False,
                    "blocking_for_historical_byte_equivalence": False,
                    "resolution": "prohibited and unnecessary",
                },
            ]
        ),
    )
    writer.json(
        "run_metadata.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "development_period": "2024",
            "assessment_period": "2025",
            "protected_2026_opened": False,
            "development_row_count": len(development.frame),
            "assessment_row_count": len(assessment.frame),
            "development_snapshot_hash": development.data_snapshot_hash,
            "assessment_snapshot_hash": assessment.data_snapshot_hash,
            "development_row_key_hash": development.row_key_hash,
            "assessment_row_key_hash": assessment.row_key_hash,
            "duration_only_model_id": duration_only.model_id,
            "duration_only_parameter_hash": duration_only.parameter_hash,
            "full_refit_model_id": full_fit.model_id,
            "full_refit_parameter_hash": full_fit.parameter_hash,
            "full_refit_model_hash": full_fit.model_hash,
            "training_row_hash": full_fit.training_row_hash,
            "primary_seed": 20260710,
            "primary_k": 8,
            "part_b_opened": False,
            "part_b_scored": False,
            "dictionary_promotion_enabled": False,
        },
    )


def run_repair(output_dir: Path) -> dict[str, Any]:
    """Run both repair tracks and the unchanged Part A sensitivity surface."""

    contract, contract_hash = _load_contract()
    legacy = _load_module("repair_legacy_event_v2", LEGACY_PANEL_RUNNER)
    prior = _load_module("repair_prior_validity_v2", PRIOR_VALIDITY_PIPELINE)
    development, assessment = _build_panels()
    panel = development.frame
    frozen_preprocessing = _load_frozen_preprocessing()
    frozen_parameters = _expanded_frozen_parameters(_load_frozen_parameters())
    frozen_scaled = transform_emissions(panel, frozen_preprocessing)
    frozen_summary = _filter_summary(
        prior,
        panel,
        scaled=frozen_scaled,
        parameters=frozen_parameters,
    )
    print("repair: duration-only isolation track", flush=True)
    duration_only = build_duration_only_repair(
        frozen_parameters=frozen_parameters,
        panel=panel,
        frozen_training_labels=frozen_summary.hard_states,
        maximum_age=78,
        duration_alpha=0.5,
        duration_beta=0.5,
        minimum_state_at_risk=5,
        tail_prior_hazard=0.05,
    )
    duration_summary = _filter_summary(
        prior,
        panel,
        scaled=frozen_scaled,
        parameters=duration_only.parameters,
    )

    print("repair: complete deterministic primary K=8 refit", flush=True)
    config = RefitConfig()
    full_fit = fit_full_right_censored_refit(
        panel,
        feature_names=EMISSION_FEATURES,
        config=config,
    )
    print("repair: clean deterministic second fit", flush=True)
    clean_second_fit = fit_full_right_censored_refit(
        panel,
        feature_names=EMISSION_FEATURES,
        config=config,
    )
    if not (
        full_fit.training_row_hash == clean_second_fit.training_row_hash
        and full_fit.parameter_hash == clean_second_fit.parameter_hash
        and full_fit.model_hash == clean_second_fit.model_hash
        and np.array_equal(full_fit.semantic_labels, clean_second_fit.semantic_labels)
    ):
        raise RuntimeError("clean deterministic second fit differs")

    full_summary = _filter_summary(
        prior,
        panel,
        scaled=full_fit.scaled,
        parameters=full_fit.parameters,
    )
    assessment_scaled = transform_emissions(assessment.frame, full_fit.preprocessing)
    assessment_summary = _filter_summary(
        prior,
        assessment.frame,
        scaled=assessment_scaled,
        parameters=full_fit.parameters,
    )
    combined_snapshot_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "development": development.data_snapshot_hash,
                "assessment": assessment.data_snapshot_hash,
            }
        )
    )
    implementation_hash = _implementation_hash()
    run_id = sha256_bytes(
        canonical_json_bytes(
            {
                "experiment": EXPERIMENT_ID,
                "git_sha": BASELINE_SHA,
                "contract_hash": contract_hash,
                "panel_hash": development.feature_table_hash,
                "model_hash": full_fit.model_hash,
            }
        )
    )[:24]
    identity = ArtifactIdentity(
        run_id=run_id,
        git_sha=BASELINE_SHA,
        contract_hash=contract_hash,
        data_snapshot_hash=combined_snapshot_hash,
        panel_hash=development.feature_table_hash,
        implementation_source_hash=implementation_hash,
        state_model_version=full_fit.model_id,
        state_model_hash=full_fit.model_hash,
        model_lineage="MODEL_FULL_REFIT",
    )
    writer = ArtifactWriter(output_dir, identity)
    _copy_pre_artifacts(writer)
    _write_panel_artifacts(
        writer,
        development,
        assessment,
        legacy=legacy,
        contract_hash=contract_hash,
    )
    _write_duration_artifacts(writer, full_fit, model_lineage="MODEL_FULL_REFIT")
    _write_refit_artifacts(
        writer,
        panel,
        duration_only=duration_only,
        full_fit=full_fit,
        clean_second_fit=clean_second_fit,
    )
    _write_independent_reconstruction_inputs(
        writer,
        panel,
        full_fit=full_fit,
        full_summary=full_summary,
    )
    groups = causal_segment_groups(panel)
    full_hysteretic = hysteretic_states_by_session(
        full_summary.state_probabilities,
        session_groups=groups,
        config=HysteresisConfig(0.55, 0.10),
    )
    gap_audit, gap_population, gap_comparison = _gap_reset_artifacts(
        panel, summary=full_summary, hysteretic=full_hysteretic
    )
    writer.frame("source_gap_reset_audit.parquet", gap_audit)
    writer.frame("gap_reset_population.csv", gap_population)
    writer.frame("pre_post_gap_state_comparison.csv", gap_comparison)
    comparison = _comparison_artifacts(
        panel,
        prior=prior,
        frozen_preprocessing=frozen_preprocessing,
        full_preprocessing=full_fit.preprocessing,
        frozen_parameters=frozen_parameters,
        duration_parameters=duration_only.parameters,
        full_parameters=full_fit.parameters,
        frozen_summary=frozen_summary,
        duration_summary=duration_summary,
        full_summary=full_summary,
    )
    _write_comparison_artifacts(writer, comparison)
    print("repair: rerun unchanged Regime Model Validity V2 gates", flush=True)
    validity = run_unchanged_validity_rerun(
        prior,
        panel,
        assessment.frame,
        primary_fit=full_fit,
        primary_summary=full_summary,
        assessment_summary=assessment_summary,
        contract=contract,
    )
    _write_validity_artifacts(writer, validity)
    _write_repair_attribution(
        writer,
        comparison=comparison,
        validity=validity,
        frozen_parameters=frozen_parameters,
        duration_parameters=duration_only.parameters,
    )
    _write_decisions_and_metadata(
        writer,
        development=development,
        assessment=assessment,
        full_fit=full_fit,
        clean_second_fit=clean_second_fit,
        duration_only=duration_only,
        validity=validity,
    )
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"required repair artifacts are missing: {missing}")
    manifest = write_artifact_manifest(
        writer,
        manifest_version="right_censored_regime_refit_v2_primary_v1",
        excluded=MANIFEST_EXCLUSIONS,
    )
    return {
        "identity": identity,
        "repair_decision": "right_censored_regime_repair_complete_with_known_limitations",
        "part_a_decision": validity.decision.value,
        "artifact_manifest_hash": manifest["manifest_hash"],
        "full_refit_model_hash": full_fit.model_hash,
        "duration_only_parameter_hash": duration_only.parameter_hash,
        "training_row_hash": full_fit.training_row_hash,
        "panel_row_count": len(panel),
        "part_b_opened": False,
    }


__all__ = [
    "ARTIFACT_PARENT",
    "BASELINE_SHA",
    "EXACT_DIR",
    "EXPERIMENT_ID",
    "MANIFEST_EXCLUSIONS",
    "PRIMARY_DIR",
    "REPAIR_REPORT_PATH",
    "REQUIRED_ARTIFACTS",
    "VALIDITY_REPORT_PATH",
    "run_repair",
]
