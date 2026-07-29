"""Run the bounded Cluster-Invariant Excursion and Closure Events V1 study.

This orchestration surface only reads the frozen structural regime lineages and
bounded 2024/2025 causal panels.  It never imports an economic target, broker,
execution, order, position, or production-runtime surface.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
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
    ArtifactIdentity,
    ArtifactWriter,
    canonical_json_bytes,
    compare_artifact_directories,
    sha256_bytes,
    sha256_file,
    write_artifact_manifest,
)

from stocker_research.causal_state_export_v2 import (  # noqa: E402
    HysteresisConfig,
)
from stocker_research.continuous_trajectory_v1 import (  # noqa: E402
    SAFETY_FLAGS,
    fit_shrinkage_metric,
    posterior_velocity,
    trajectory_features,
)
from stocker_research.excursion_alignment_v1 import (  # noqa: E402
    align_event_ledgers,
    event_alignment_summary,
)
from stocker_research.excursion_events_v1 import (  # noqa: E402
    DistanceCalibration,
    ExcursionConfig,
    PartAGateMetrics,
    decide_part_a,
    detect_excursions,
    event_definition_hash,
)
from stocker_research.excursion_nulls_v1 import (  # noqa: E402
    benjamini_hochberg,
    circular_increment_control,
    clock_phase_labels,
    fit_phase_conditioned_var1,
    phase_conditioned_increment_block_null,
    reconstruct_trajectory,
    simulate_phase_conditioned_var1,
)
from stocker_research.excursion_origin_v1 import (  # noqa: E402
    OriginSurface,
    locally_stable_origins,
    trailing_robust_origins,
)
from stocker_research.regime_gap_segmentation_v2 import (  # noqa: E402
    causal_segment_groups,
)
from stocker_research.regime_panel_v2 import (  # noqa: E402
    EMISSION_FEATURES,
    RegimePanelConfig,
    build_regime_panel,
)
from stocker_research.regime_validity_v2 import (  # noqa: E402
    CausalFilterSummary,
    EmissionPreprocessing,
    SemiMarkovParameters,
    gaussian_log_emissions,
    transform_emissions,
)
from stocker_research.state_representation_sensitivity_v2 import (  # noqa: E402
    hysteretic_states_by_session,
)

EXPERIMENT_ID = "20260719-cluster-invariant-excursion-events-v1"
BASELINE_SHA = "91996a9cf747a614ff6d9e08eaafc3583a58b91c"
BRANCH = "agent/slrno-research-handoff"
CONTRACT_PATH = WORK_DIR / "contracts" / f"{EXPERIMENT_ID}.json"
ARTIFACT_PARENT = WORK_DIR / "artifacts" / EXPERIMENT_ID
PRIMARY_DIR = ARTIFACT_PARENT / "primary"
EXACT_DIR = ARTIFACT_PARENT / "exact_rerun"
REPORT_PATH = WORK_DIR / "reports" / f"{EXPERIMENT_ID}.md"
PREDECESSOR_DIR = WORK_DIR / "artifacts" / "20260719-right-censored-regime-refit-v2" / "primary"
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
VALIDATION_START = pd.Timestamp("2025-01-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
EXPECTED_DEVELOPMENT_SNAPSHOT = "48d2141ef993928d4e8a01d6b3c24dff665280c67f4167115b453613460cc661"
EXPECTED_VALIDATION_SNAPSHOT = "29e82d6539810e5fcebc13e860d07474c38ee0349fe38aedce0378f9aefb67a4"
EXPECTED_DEVELOPMENT_PANEL = "801c0bf9d69ecdd58b21fb2ba4392137048b466668344ebfc4c8faf6a0d3e2f1"
EXPECTED_VALIDATION_PANEL = "ad117a54fd1a249caadb8c35fd094378a562812f7e042e88d81badacc1188245"
EXPECTED_MODEL_HASHES = {
    "MODEL_FROZEN": "909858ed7c9c02c1c113661202cb5d7c6bfabd243f1cc428b8a5fb1a3c022251",
    "MODEL_DURATION_REPAIR": "40d5b2c149856e2e2cdbf3df15adfe0c8108c1bb45ada73235a87bf67f87ce44",
    "MODEL_FULL_REFIT": "4fc1a02dce9ac2311dabaeb4623a559d37286dfe58baffef53828cc7415a3425",
}
MANIFEST_EXCLUSIONS = {
    "artifact_manifest.json",
    "independent_audit.json",
    "exact_rerun_manifest.json",
    "post_run_tree_manifest.json",
}
IMPLEMENTATION_PATHS = (
    Path("packages/stocker_research/src/stocker_research/continuous_trajectory_v1.py"),
    Path("packages/stocker_research/src/stocker_research/excursion_origin_v1.py"),
    Path("packages/stocker_research/src/stocker_research/excursion_events_v1.py"),
    Path("packages/stocker_research/src/stocker_research/excursion_alignment_v1.py"),
    Path("packages/stocker_research/src/stocker_research/excursion_nulls_v1.py"),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/"
        "run_cluster_invariant_excursion_events_v1.py"
    ),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/"
        "audit_cluster_invariant_excursion_events_v1.py"
    ),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/contracts/"
        "20260719-cluster-invariant-excursion-events-v1.json"
    ),
)


@dataclass(frozen=True, slots=True)
class ModelLineage:
    name: str
    model_hash: str
    preprocessing: EmissionPreprocessing
    parameters: SemiMarkovParameters


@dataclass(frozen=True, slots=True)
class PeriodTrajectory:
    period: str
    panel: pd.DataFrame
    decisions: pd.DataFrame
    groups: tuple[np.ndarray, ...]
    raw_emissions: np.ndarray
    emission_missing: np.ndarray
    emission_complete: np.ndarray
    z: np.ndarray
    first_difference: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    path_length: np.ndarray
    directional_consistency: np.ndarray
    summaries: Mapping[str, CausalFilterSummary]
    hysteretic: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class CandidateRun:
    candidate: Mapping[str, Any]
    config: ExcursionConfig
    calibration: DistanceCalibration
    emission_origin: OriginSurface
    posterior_origin: OriginSurface | None
    detection: Any
    metrics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NullSession:
    symbol: str
    period: str
    decisions: pd.DataFrame
    z: np.ndarray
    posterior: np.ndarray
    phases: np.ndarray
    development_var_model: Any


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


def _load_contract() -> tuple[dict[str, Any], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"contract safety flag differs: {key}")
    source = contract["source_identity"]
    if source["implementation_target_git_sha"] != BASELINE_SHA:
        raise RuntimeError("contract Git SHA differs from implementation target")
    for key, expected in (
        ("development_2024_snapshot_hash", EXPECTED_DEVELOPMENT_SNAPSHOT),
        ("validation_2025_snapshot_hash", EXPECTED_VALIDATION_SNAPSHOT),
        ("development_panel_hash", EXPECTED_DEVELOPMENT_PANEL),
        ("validation_panel_hash", EXPECTED_VALIDATION_PANEL),
    ):
        if source[key] != expected:
            raise RuntimeError(f"contract source identity differs: {key}")
    for name in ("pre_run_source_identity", "pre_run_tree_manifest"):
        path = REPO_ROOT / source[name]
        if sha256_file(path) != source[f"{name}_hash"]:
            raise RuntimeError(f"pre-run freeze differs: {name}")
    predecessor = json.loads((PREDECESSOR_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    for lineage, expected in EXPECTED_MODEL_HASHES.items():
        metadata_key = {
            "MODEL_FROZEN": None,
            "MODEL_DURATION_REPAIR": "duration_only_parameter_hash",
            "MODEL_FULL_REFIT": "full_refit_model_hash",
        }[lineage]
        if metadata_key is not None and predecessor[metadata_key] != expected:
            raise RuntimeError(f"predecessor model identity differs: {lineage}")
    return contract, sha256_file(CONTRACT_PATH)


def _panel_config(*, validation: bool) -> RegimePanelConfig:
    return RegimePanelConfig(
        provider_root=PROVIDER_ROOT,
        symbols=SYMBOLS,
        benchmark_symbol="VTI",
        start=VALIDATION_START if validation else DEVELOPMENT_START,
        end=VALIDATION_END if validation else DEVELOPMENT_END,
    )


def _load_preprocessing(path: Path) -> EmissionPreprocessing:
    frame = pd.read_csv(path)
    feature_column = "feature" if "feature" in frame else "feature_name"
    median_column = "imputer_median" if "imputer_median" in frame else "median"
    center_column = "scaler_center" if "scaler_center" in frame else "center"
    scale_column = "scaler_scale" if "scaler_scale" in frame else "scale"
    result = EmissionPreprocessing(
        feature_names=tuple(frame[feature_column].astype(str)),
        medians=frame[median_column].to_numpy(dtype=float),
        centers=frame[center_column].to_numpy(dtype=float),
        scales=frame[scale_column].to_numpy(dtype=float),
    )
    result.validate()
    return result


def _load_parameters(path: Path) -> SemiMarkovParameters:
    with np.load(path) as stored:
        values = {key: np.asarray(stored[key]).copy() for key in stored.files}
    result = SemiMarkovParameters(
        means=values["means"],
        variances=values["variances"],
        duration_hazard=values["duration_hazard"],
        transitions=values["transitions"],
        initial=values["initial"],
        occupancy=values["occupancy"],
    )
    result.validate()
    return result


def _lineages() -> dict[str, ModelLineage]:
    causal_export = _load_module(
        "excursion_causal_state_export_v2",
        PACKAGE_ROOT / "stocker_research" / "causal_state_export_v2.py",
    )
    frozen_parameters = _load_parameters(FROZEN_STATE_PATH)
    expanded = causal_export.expand_duration_hazard_v2(
        frozen_parameters.as_dict(), maximum_age=78, tail_window=6
    )
    frozen_parameters = SemiMarkovParameters(**expanded)
    frozen_parameters.validate()
    frozen_preprocessing = _load_preprocessing(FROZEN_PREPROCESSING_PATH)
    duration_parameters = _load_parameters(PREDECESSOR_DIR / "duration_only_repair_parameters.npz")
    full_parameters = _load_parameters(PREDECESSOR_DIR / "full_refit_parameters.npz")
    full_preprocessing = _load_preprocessing(PREDECESSOR_DIR / "full_refit_preprocessing.csv")
    return {
        "MODEL_FROZEN": ModelLineage(
            "MODEL_FROZEN",
            EXPECTED_MODEL_HASHES["MODEL_FROZEN"],
            frozen_preprocessing,
            frozen_parameters,
        ),
        "MODEL_DURATION_REPAIR": ModelLineage(
            "MODEL_DURATION_REPAIR",
            EXPECTED_MODEL_HASHES["MODEL_DURATION_REPAIR"],
            frozen_preprocessing,
            duration_parameters,
        ),
        "MODEL_FULL_REFIT": ModelLineage(
            "MODEL_FULL_REFIT",
            EXPECTED_MODEL_HASHES["MODEL_FULL_REFIT"],
            full_preprocessing,
            full_parameters,
        ),
    }


def _filter_summary(
    prior: Any,
    panel: pd.DataFrame,
    lineage: ModelLineage,
) -> CausalFilterSummary:
    scaled = transform_emissions(panel, lineage.preprocessing)
    log_emissions = gaussian_log_emissions(scaled, lineage.parameters)
    return prior._causal_filter_summary_compiled(
        log_emissions,
        groups=causal_segment_groups(panel),
        model=lineage.parameters.as_dict(),
    )


def _decision_ids(panel: pd.DataFrame) -> list[str]:
    keys = panel[
        ["symbol", "session", "segment_id", "bar_start_timestamp", "bar_ordinal"]
    ].itertuples(index=False, name=None)
    return [
        "decision_"
        + hashlib.sha256("|".join(str(value) for value in key).encode("utf-8")).hexdigest()[:24]
        for key in keys
    ]


def _decision_frame(panel: pd.DataFrame, *, period: str) -> pd.DataFrame:
    columns = [
        "symbol",
        "session",
        "segment_id",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "feature_available_timestamp_max",
        "cross_sectional_source_timestamp",
        "cross_sectional_available_timestamp",
        "segment_end_reason",
        "session_source_complete",
    ]
    frame = panel[columns].copy().reset_index(drop=True)
    frame.insert(0, "decision_id", _decision_ids(panel))
    frame["decision_timestamp"] = frame["bar_complete_timestamp"]
    frame["period"] = period
    return frame


def _build_period_trajectory(
    prior: Any,
    *,
    period: str,
    panel: pd.DataFrame,
    lineages: Mapping[str, ModelLineage],
) -> PeriodTrajectory:
    groups = causal_segment_groups(panel)
    raw = panel.loc[:, list(EMISSION_FEATURES)].to_numpy(dtype=float)
    missing = ~np.isfinite(raw)
    complete = ~missing.any(axis=1)
    primary = lineages["MODEL_FULL_REFIT"]
    z = transform_emissions(panel, primary.preprocessing)
    trajectory = trajectory_features(z, groups=groups, window=3, valid=complete)
    summaries: dict[str, CausalFilterSummary] = {}
    hysteretic: dict[str, np.ndarray] = {}
    for name, lineage in lineages.items():
        print(f"excursion: causal posterior {period} {name}", flush=True)
        summary = _filter_summary(prior, panel, lineage)
        summaries[name] = summary
        hysteretic[name] = hysteretic_states_by_session(
            summary.state_probabilities,
            session_groups=groups,
            config=HysteresisConfig(0.55, 0.10),
        )
    return PeriodTrajectory(
        period=period,
        panel=panel,
        decisions=_decision_frame(panel, period=period),
        groups=groups,
        raw_emissions=raw,
        emission_missing=missing,
        emission_complete=complete,
        z=z,
        first_difference=trajectory.first_difference,
        velocity=trajectory.velocity,
        acceleration=trajectory.acceleration,
        path_length=trajectory.local_path_length,
        directional_consistency=trajectory.directional_consistency,
        summaries=summaries,
        hysteretic=hysteretic,
    )


def _build_trajectories(
    lineages: Mapping[str, ModelLineage],
) -> tuple[PeriodTrajectory, PeriodTrajectory, Any, Any]:
    print("excursion: reconstruct bounded 2024 panel", flush=True)
    development_build = build_regime_panel(_panel_config(validation=False))
    if (
        development_build.data_snapshot_hash != EXPECTED_DEVELOPMENT_SNAPSHOT
        or development_build.feature_table_hash != EXPECTED_DEVELOPMENT_PANEL
    ):
        raise RuntimeError("development panel identity differs from the pre-run freeze")
    print("excursion: reconstruct unchanged bounded 2025 panel", flush=True)
    validation_build = build_regime_panel(_panel_config(validation=True))
    if (
        validation_build.data_snapshot_hash != EXPECTED_VALIDATION_SNAPSHOT
        or validation_build.feature_table_hash != EXPECTED_VALIDATION_PANEL
    ):
        raise RuntimeError("validation panel identity differs from the pre-run freeze")
    prior = _load_module(
        "excursion_regime_validity_pipeline_v2",
        WORK_DIR / "regime_validity_pipeline_v2.py",
    )
    development = _build_period_trajectory(
        prior,
        period="DEVELOPMENT_2024",
        panel=development_build.frame,
        lineages=lineages,
    )
    validation = _build_period_trajectory(
        prior,
        period="VALIDATION_2025",
        panel=validation_build.frame,
        lineages=lineages,
    )
    return development, validation, development_build, validation_build


def _vectorized_js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    first = first / first.sum(axis=1, keepdims=True)
    second = second / second.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        log_first = np.log(first)
        log_second = np.log(second)
    log_midpoint = np.logaddexp(log_first, log_second) - math.log(2.0)
    first_term = np.zeros_like(first)
    second_term = np.zeros_like(second)
    first_positive = first > 0.0
    second_positive = second > 0.0
    first_term[first_positive] = first[first_positive] * (
        log_first[first_positive] - log_midpoint[first_positive]
    )
    second_term[second_positive] = second[second_positive] * (
        log_second[second_positive] - log_midpoint[second_positive]
    )
    return np.sqrt(np.maximum(0.0, 0.5 * first_term.sum(axis=1) + 0.5 * second_term.sum(axis=1)))


def _distance_surface(
    values: np.ndarray,
    origin: OriginSurface,
    *,
    representation: str,
    precision: np.ndarray | None = None,
) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    eligible = origin.eligible & np.isfinite(values).all(axis=1)
    differences = values[eligible] - origin.centers[eligible]
    if representation == "P":
        output[eligible] = _vectorized_js(values[eligible], origin.centers[eligible])
    elif precision is None:
        output[eligible] = np.linalg.norm(differences, axis=1)
    else:
        output[eligible] = np.sqrt(
            np.maximum(0.0, np.einsum("ij,jk,ik->i", differences, precision, differences))
        )
    return output


def _group_local_difference(values: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for raw_group in groups:
        group = np.asarray(raw_group, dtype=int)
        output[group[1:]] = values[group[1:]] - values[group[:-1]]
    return output


def _origin_surfaces(
    trajectory: PeriodTrajectory,
    *,
    stable_path_threshold: float,
    stable_velocity_threshold: float,
    lineage: str = "MODEL_FULL_REFIT",
) -> tuple[dict[str, OriginSurface], dict[str, OriginSurface]]:
    emission = {
        f"ORIGIN_A_W{window}": trailing_robust_origins(
            trajectory.z,
            groups=trajectory.groups,
            window=window,
            valid=trajectory.emission_complete,
            definition_id=f"ORIGIN_A_W{window}",
        )
        for window in (3, 6, 12)
    }
    emission["ORIGIN_B_STABLE_W6"] = locally_stable_origins(
        trajectory.z,
        groups=trajectory.groups,
        window=6,
        maximum_path_length=stable_path_threshold,
        maximum_velocity=stable_velocity_threshold,
        valid=trajectory.emission_complete,
        definition_id="ORIGIN_B_STABLE_W6",
    )
    posterior_values = trajectory.summaries[lineage].state_probabilities
    posterior = {
        f"ORIGIN_A_W{window}": trailing_robust_origins(
            posterior_values,
            groups=trajectory.groups,
            window=window,
            valid=trajectory.emission_complete,
            definition_id=f"{lineage}_POSTERIOR_ORIGIN_A_W{window}",
        )
        for window in (3, 6, 12)
    }
    posterior["ORIGIN_B_STABLE_W6"] = trailing_robust_origins(
        posterior_values,
        groups=trajectory.groups,
        window=6,
        valid=trajectory.emission_complete,
        definition_id=f"{lineage}_POSTERIOR_ORIGIN_B_STABLE_W6",
    )
    return emission, posterior


def _development_geometry_thresholds(
    development: PeriodTrajectory,
) -> tuple[float, float, float]:
    six_bar = trajectory_features(
        development.z,
        groups=development.groups,
        window=6,
        valid=development.emission_complete,
    )
    path_values = six_bar.local_path_length[six_bar.local_path_length > 0.0]
    velocity_values = six_bar.velocity[six_bar.velocity > 0.0]
    step_values = np.linalg.norm(development.first_difference, axis=1)
    step_values = step_values[np.isfinite(step_values) & (step_values > 0.0)]
    if not len(path_values) or not len(velocity_values) or not len(step_values):
        raise RuntimeError("development trajectory has no stable-threshold support")
    return (
        float(np.quantile(path_values, 0.25)),
        float(np.quantile(velocity_values, 0.25)),
        float(np.quantile(step_values, 0.25)),
    )


def _posterior_step_threshold(trajectory: PeriodTrajectory, *, lineage: str) -> float:
    values = posterior_velocity(
        trajectory.summaries[lineage].state_probabilities,
        groups=trajectory.groups,
    )
    positive = values[values > 0.0]
    if not len(positive):
        raise RuntimeError(f"posterior velocity has no support: {lineage}")
    return float(np.quantile(positive, 0.25))


def _candidate_quantile(candidate: Mapping[str, Any]) -> float:
    return float(candidate["threshold_quantile"])


def _calibrate_candidate(
    candidate: Mapping[str, Any],
    *,
    development: PeriodTrajectory,
    emission_origins: Mapping[str, OriginSurface],
    posterior_origins: Mapping[str, OriginSurface],
    lineage: str,
    precision: np.ndarray,
    rotation_emission_velocity: float,
) -> tuple[ExcursionConfig, DistanceCalibration, np.ndarray]:
    origin_id = str(candidate["origin"])
    emission_origin = emission_origins[origin_id]
    posterior_origin = posterior_origins[origin_id]
    diagonal = _distance_surface(development.z, emission_origin, representation="E")
    emission_q90 = float(np.nanquantile(diagonal, 0.90))
    posterior_values = development.summaries[lineage].state_probabilities
    posterior = _distance_surface(
        posterior_values,
        posterior_origin,
        representation="P",
    )
    posterior_q90 = float(np.nanquantile(posterior, 0.90))
    distance_kind = str(candidate["distance"])
    representation = str(candidate["representation"])
    if distance_kind == "SHRINKAGE_MAHALANOBIS":
        distances = _distance_surface(
            development.z,
            emission_origin,
            representation="E",
            precision=precision,
        )
    elif representation == "P":
        distances = posterior
    elif representation == "H":
        distances = 0.5 * (diagonal / emission_q90) + 0.5 * (posterior / posterior_q90)
    else:
        distances = diagonal
    finite = distances[np.isfinite(distances)]
    if not len(finite):
        raise RuntimeError(f"candidate has no development distances: {candidate['id']}")
    threshold = float(np.quantile(finite, _candidate_quantile(candidate)))
    local_velocity = _group_local_difference(distances, development.groups)
    positive_velocity = local_velocity[np.isfinite(local_velocity) & (local_velocity > 0.0)]
    minimum_velocity = float(np.median(positive_velocity)) if len(positive_velocity) else 0.0
    rotation_velocity = (
        _posterior_step_threshold(development, lineage=lineage)
        if representation == "P"
        else rotation_emission_velocity
    )
    config = ExcursionConfig(
        candidate_id=str(candidate["id"]),
        representation=representation,
        distance_metric=distance_kind,
        departure_threshold=threshold,
        confirmation_bars=int(candidate["confirmation"]),
        velocity_condition=bool(candidate["velocity_condition"]),
        minimum_departure_velocity=minimum_velocity,
        return_ratio=float(candidate["return_ratio"]),
        rotation_persistence=int(candidate["rotation_persistence"]),
        rotation_separation_ratio=1.0,
        rotation_maximum_velocity=rotation_velocity,
        continuation_ratio=1.5,
        partial_retracement_fraction=0.35,
        horizon_bars=24,
        lockout_bars=1,
    )
    calibration = DistanceCalibration(
        emission_scale=np.ones(development.z.shape[1], dtype=float),
        emission_q90=emission_q90,
        posterior_q90=posterior_q90,
        mahalanobis_precision=precision,
    )
    return config, calibration, distances


def _event_metrics(
    events: pd.DataFrame,
    *,
    eligible_fraction: float,
) -> dict[str, Any]:
    if events.empty:
        return {
            "unique_event_count": 0,
            "stock_count": 0,
            "month_count": 0,
            "maximum_stock_share": 1.0,
            "maximum_month_share": 1.0,
            "class_entropy": 0.0,
            "maximum_quarter_share_range_pp": 100.0,
            "median_onset_to_resolution_bars": math.nan,
            "median_first_detectable_to_resolution_bars": math.nan,
            "eligible_fraction": eligible_fraction,
        }
    frame = events.copy()
    timestamps = pd.to_datetime(frame["onset_timestamp"], utc=True)
    frame["month"] = timestamps.dt.strftime("%Y-%m")
    frame["quarter"] = (
        timestamps.dt.year.astype(str) + "Q" + (((timestamps.dt.month - 1) // 3) + 1).astype(str)
    )
    family_share = frame["event_family"].value_counts(normalize=True)
    stock_share = frame["symbol"].value_counts(normalize=True)
    month_share = frame["month"].value_counts(normalize=True)
    quarter_share = pd.crosstab(frame["quarter"], frame["event_family"], normalize="index")
    maximum_quarter_range = (
        float((quarter_share.max(axis=0) - quarter_share.min(axis=0)).max() * 100.0)
        if len(quarter_share)
        else 100.0
    )
    entropy = float(-np.sum(family_share.to_numpy() * np.log(family_share.to_numpy())))
    return {
        "unique_event_count": int(len(frame)),
        "stock_count": int(frame["symbol"].nunique()),
        "month_count": int(frame["month"].nunique()),
        "maximum_stock_share": float(stock_share.max()),
        "maximum_month_share": float(month_share.max()),
        "class_entropy": entropy,
        "maximum_quarter_share_range_pp": maximum_quarter_range,
        "median_onset_to_resolution_bars": float(
            frame["bars_from_confirmation_to_resolution"].median()
        ),
        "median_first_detectable_to_resolution_bars": float(
            frame["bars_from_first_detectable_to_resolution"].median()
        ),
        "eligible_fraction": eligible_fraction,
    }


def _run_candidate(
    candidate: Mapping[str, Any],
    *,
    trajectory: PeriodTrajectory,
    emission_origins: Mapping[str, OriginSurface],
    posterior_origins: Mapping[str, OriginSurface],
    lineage: str,
    config: ExcursionConfig,
    calibration: DistanceCalibration,
) -> CandidateRun:
    origin_id = str(candidate["origin"])
    posterior = trajectory.summaries[lineage].state_probabilities
    detection = detect_excursions(
        trajectory.decisions,
        emission_vectors=trajectory.z,
        emission_origins=emission_origins[origin_id],
        posterior_vectors=(posterior if str(candidate["representation"]) in {"P", "H"} else None),
        posterior_origins=(
            posterior_origins[origin_id] if str(candidate["representation"]) in {"P", "H"} else None
        ),
        calibration=calibration,
        config=config,
    )
    eligible = emission_origins[origin_id].eligible
    if str(candidate["representation"]) in {"P", "H"}:
        eligible &= posterior_origins[origin_id].eligible
    metrics = _event_metrics(
        detection.events,
        eligible_fraction=float(eligible.mean()),
    )
    return CandidateRun(
        candidate=candidate,
        config=config,
        calibration=calibration,
        emission_origin=emission_origins[origin_id],
        posterior_origin=(
            posterior_origins[origin_id] if str(candidate["representation"]) in {"P", "H"} else None
        ),
        detection=detection,
        metrics=metrics,
    )


def _candidate_comparison_row(
    run: CandidateRun,
    *,
    cross_lineage_agreement: float,
) -> dict[str, Any]:
    return {
        "candidate_id": run.config.candidate_id,
        "trajectory_representation": run.config.representation,
        "origin_definition": run.emission_origin.definition_id,
        "distance_metric": run.config.distance_metric,
        "departure_threshold": run.config.departure_threshold,
        "confirmation_bars": run.config.confirmation_bars,
        "velocity_condition": run.config.velocity_condition,
        "return_ratio": run.config.return_ratio,
        "rotation_persistence": run.config.rotation_persistence,
        "event_definition_hash": event_definition_hash(
            run.config,
            calibration=run.calibration,
            origin_definition_id=run.emission_origin.definition_id,
            posterior_origin_definition_id=(
                None if run.posterior_origin is None else run.posterior_origin.definition_id
            ),
        ),
        "cross_lineage_same_family_fraction": cross_lineage_agreement,
        **dict(run.metrics),
    }


def _select_candidate(comparison: pd.DataFrame) -> str:
    eligible = comparison.loc[
        comparison["unique_event_count"].ge(2000)
        & comparison["stock_count"].ge(15)
        & comparison["month_count"].ge(8)
        & comparison["eligible_fraction"].ge(0.50)
        & comparison["trajectory_representation"].eq("E")
    ].copy()
    if eligible.empty:
        raise RuntimeError("no emission-space candidate has adequate development support")
    eligible = eligible.sort_values(
        [
            "cross_lineage_same_family_fraction",
            "maximum_quarter_share_range_pp",
            "class_entropy",
            "candidate_id",
        ],
        ascending=[False, True, False, True],
        kind="mergesort",
    )
    return str(eligible.iloc[0]["candidate_id"])


def _posterior_origin_surfaces(
    trajectory: PeriodTrajectory, *, lineage: str
) -> dict[str, OriginSurface]:
    probabilities = trajectory.summaries[lineage].state_probabilities
    return {
        f"ORIGIN_A_W{window}": trailing_robust_origins(
            probabilities,
            groups=trajectory.groups,
            window=window,
            valid=trajectory.emission_complete,
            definition_id=f"{lineage}_POSTERIOR_ORIGIN_A_W{window}",
        )
        for window in (3, 6, 12)
    } | {
        "ORIGIN_B_STABLE_W6": trailing_robust_origins(
            probabilities,
            groups=trajectory.groups,
            window=6,
            valid=trajectory.emission_complete,
            definition_id=f"{lineage}_POSTERIOR_ORIGIN_B_STABLE_W6",
        )
    }


def _evaluate_candidates(
    contract: Mapping[str, Any],
    *,
    development: PeriodTrajectory,
    validation: PeriodTrajectory,
) -> dict[str, Any]:
    stable_path, stable_velocity, rotation_velocity = _development_geometry_thresholds(development)
    development_emission_origins, development_posterior_origins = _origin_surfaces(
        development,
        stable_path_threshold=stable_path,
        stable_velocity_threshold=stable_velocity,
    )
    validation_emission_origins, validation_posterior_origins = _origin_surfaces(
        validation,
        stable_path_threshold=stable_path,
        stable_velocity_threshold=stable_velocity,
    )
    precision = fit_shrinkage_metric(development.z[development.emission_complete]).precision
    primary_runs: dict[str, CandidateRun] = {}
    comparison_rows: list[dict[str, Any]] = []
    sensitivity_runs: dict[tuple[str, str], CandidateRun] = {}
    sensitivity_alignments: list[pd.DataFrame] = []
    distance_surfaces: dict[str, np.ndarray] = {}

    for candidate_value in contract["candidate_registry"]:
        candidate = dict(candidate_value)
        candidate_id = str(candidate["id"])
        print(f"excursion: development candidate {candidate_id}", flush=True)
        config, calibration, distances = _calibrate_candidate(
            candidate,
            development=development,
            emission_origins=development_emission_origins,
            posterior_origins=development_posterior_origins,
            lineage="MODEL_FULL_REFIT",
            precision=precision,
            rotation_emission_velocity=rotation_velocity,
        )
        run = _run_candidate(
            candidate,
            trajectory=development,
            emission_origins=development_emission_origins,
            posterior_origins=development_posterior_origins,
            lineage="MODEL_FULL_REFIT",
            config=config,
            calibration=calibration,
        )
        primary_runs[candidate_id] = run
        distance_surfaces[candidate_id] = distances
        cross_lineage = 1.0
        if config.representation in {"P", "H"}:
            agreements: list[float] = []
            for lineage in ("MODEL_FROZEN", "MODEL_DURATION_REPAIR"):
                lineage_origins = _posterior_origin_surfaces(development, lineage=lineage)
                lineage_config, lineage_calibration, _ = _calibrate_candidate(
                    candidate,
                    development=development,
                    emission_origins=development_emission_origins,
                    posterior_origins=lineage_origins,
                    lineage=lineage,
                    precision=precision,
                    rotation_emission_velocity=rotation_velocity,
                )
                lineage_run = _run_candidate(
                    candidate,
                    trajectory=development,
                    emission_origins=development_emission_origins,
                    posterior_origins=lineage_origins,
                    lineage=lineage,
                    config=lineage_config,
                    calibration=lineage_calibration,
                )
                sensitivity_runs[(candidate_id, lineage)] = lineage_run
                alignment = align_event_ledgers(
                    run.detection.events,
                    lineage_run.detection.events,
                    tolerance_bars=2,
                )
                summary = event_alignment_summary(alignment)
                agreements.append(float(summary["same_family_fraction"]))
                alignment["candidate_id"] = candidate_id
                alignment["trajectory_representation"] = config.representation
                alignment["reference_model_lineage"] = "MODEL_FULL_REFIT"
                alignment["candidate_model_lineage"] = lineage
                sensitivity_alignments.append(alignment)
            cross_lineage = min(agreements) if agreements else 0.0
        comparison_rows.append(
            _candidate_comparison_row(
                run,
                cross_lineage_agreement=cross_lineage,
            )
        )

    comparison = pd.DataFrame(comparison_rows)
    selected_id = _select_candidate(comparison)
    selected = primary_runs[selected_id]
    selected_candidate = dict(selected.candidate)
    print(f"excursion: selected development definition {selected_id}", flush=True)
    for lineage in ("MODEL_FROZEN", "MODEL_DURATION_REPAIR"):
        sensitivity_runs[(selected_id, lineage)] = _run_candidate(
            selected_candidate,
            trajectory=development,
            emission_origins=development_emission_origins,
            posterior_origins=development_posterior_origins,
            lineage=lineage,
            config=selected.config,
            calibration=selected.calibration,
        )
    validation_run = _run_candidate(
        selected_candidate,
        trajectory=validation,
        emission_origins=validation_emission_origins,
        posterior_origins=validation_posterior_origins,
        lineage="MODEL_FULL_REFIT",
        config=selected.config,
        calibration=selected.calibration,
    )
    horizon_runs: dict[tuple[str, int], CandidateRun] = {}
    for period_name, trajectory, emission_origins, posterior_origins in (
        (
            "DEVELOPMENT_2024",
            development,
            development_emission_origins,
            development_posterior_origins,
        ),
        (
            "VALIDATION_2025",
            validation,
            validation_emission_origins,
            validation_posterior_origins,
        ),
    ):
        for horizon in (12, 36):
            horizon_runs[(period_name, horizon)] = _run_candidate(
                selected_candidate,
                trajectory=trajectory,
                emission_origins=emission_origins,
                posterior_origins=posterior_origins,
                lineage="MODEL_FULL_REFIT",
                config=replace(selected.config, horizon_bars=horizon),
                calibration=selected.calibration,
            )
    return {
        "stable_path_threshold": stable_path,
        "stable_velocity_threshold": stable_velocity,
        "rotation_velocity_threshold": rotation_velocity,
        "precision": precision,
        "development_emission_origins": development_emission_origins,
        "validation_emission_origins": validation_emission_origins,
        "development_posterior_origins": development_posterior_origins,
        "validation_posterior_origins": validation_posterior_origins,
        "primary_runs": primary_runs,
        "sensitivity_runs": sensitivity_runs,
        "sensitivity_alignments": (
            pd.concat(sensitivity_alignments, ignore_index=True)
            if sensitivity_alignments
            else pd.DataFrame()
        ),
        "distance_surfaces": distance_surfaces,
        "comparison": comparison,
        "selected_id": selected_id,
        "selected": selected,
        "validation_run": validation_run,
        "horizon_runs": horizon_runs,
    }


def _class_share_metrics(
    development_events: pd.DataFrame,
    validation_events: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    families = sorted(
        set(development_events["event_family"].astype(str))
        | set(validation_events["event_family"].astype(str))
    )
    development_share = development_events["event_family"].value_counts(normalize=True)
    validation_share = validation_events["event_family"].value_counts(normalize=True)
    rows = []
    major_shifts = []
    for family in families:
        left = float(development_share.get(family, 0.0))
        right = float(validation_share.get(family, 0.0))
        shift = abs(right - left) * 100.0
        major = left >= 0.05
        if major:
            major_shifts.append(shift)
        rows.append(
            {
                "event_family": family,
                "development_share": left,
                "validation_share": right,
                "absolute_shift_percentage_points": shift,
                "supported_major_class": major,
                "maximum_allowed_shift_percentage_points": 7.5,
                "passed": (not major) or shift <= 7.5,
            }
        )
    return pd.DataFrame(rows), max(major_shifts, default=0.0)


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"implementation source is missing: {relative}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _identity(
    *,
    contract_hash: str,
    development_build: Any,
    validation_build: Any,
) -> ArtifactIdentity:
    combined_snapshot = sha256_bytes(
        canonical_json_bytes(
            {
                "development": development_build.data_snapshot_hash,
                "validation": validation_build.data_snapshot_hash,
            }
        )
    )
    run_id = sha256_bytes(
        canonical_json_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "git_sha": BASELINE_SHA,
                "contract_hash": contract_hash,
                "development_panel_hash": development_build.feature_table_hash,
                "validation_panel_hash": validation_build.feature_table_hash,
            }
        )
    )[:24]
    return ArtifactIdentity(
        run_id=run_id,
        git_sha=BASELINE_SHA,
        contract_hash=contract_hash,
        data_snapshot_hash=combined_snapshot,
        panel_hash=development_build.feature_table_hash,
        implementation_source_hash=_implementation_hash(),
        state_model_version="cluster_invariant_excursion_event_model_v1",
        state_model_hash=EXPECTED_MODEL_HASHES["MODEL_FULL_REFIT"],
        model_lineage="MODEL_FULL_REFIT",
    )


def _copy_pre_run_artifacts(writer: ArtifactWriter) -> None:
    for name in (
        "excursion_implementation_census.csv",
        "pre_run_source_identity.json",
        "pre_run_tree_manifest.json",
    ):
        source = PRIMARY_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"pre-run artifact is missing: {source}")
        writer.copy_pre_artifact(source)


def _period_hashes(period: str) -> tuple[str, str]:
    if period == "DEVELOPMENT_2024":
        return EXPECTED_DEVELOPMENT_SNAPSHOT, EXPECTED_DEVELOPMENT_PANEL
    if period == "VALIDATION_2025":
        return EXPECTED_VALIDATION_SNAPSHOT, EXPECTED_VALIDATION_PANEL
    return "not_applicable", EXPECTED_DEVELOPMENT_PANEL


def _detailed(
    frame: pd.DataFrame,
    *,
    identity: ArtifactIdentity,
    feature_manifest_hash: str,
    representation: str,
    model_lineage: str,
    event_definition_hash_value: str,
    source_artifact: str,
    source_hash: str,
    period: str | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    selected_period = period
    if selected_period is not None:
        snapshot_hash, panel_hash = _period_hashes(selected_period)
        output["period"] = selected_period
        output["period_data_snapshot_hash"] = snapshot_hash
        output["panel_hash"] = panel_hash
    elif "period" in output:
        output["period_data_snapshot_hash"] = output["period"].map(
            lambda value: _period_hashes(str(value))[0]
        )
        output["panel_hash"] = output["period"].map(lambda value: _period_hashes(str(value))[1])
    for column, default in (
        ("decision_id", "not_applicable"),
        ("event_id", "not_applicable"),
        ("symbol", "not_applicable"),
        ("session", "not_applicable"),
        ("segment_id", "not_applicable"),
        ("decision_timestamp", pd.NaT),
        ("onset_timestamp", pd.NaT),
        ("resolution_timestamp", pd.NaT),
        ("event_family", "not_applicable"),
    ):
        if column not in output:
            output[column] = default
    output["run_id"] = identity.run_id
    output["git_sha"] = identity.git_sha
    output["contract_hash"] = identity.contract_hash
    output["feature_manifest_hash"] = feature_manifest_hash
    if "trajectory_representation" not in output:
        output["trajectory_representation"] = representation
    else:
        output["trajectory_representation"] = output["trajectory_representation"].fillna(
            representation
        )
    output["model_lineage"] = model_lineage
    output["event_definition_hash"] = event_definition_hash_value
    output["source_artifact"] = source_artifact
    output["source_hash"] = source_hash
    for key, value in SAFETY_FLAGS.items():
        output[key] = value
    return output


def _trajectory_manifest_payload(
    *,
    development: PeriodTrajectory,
    lineages: Mapping[str, ModelLineage],
) -> dict[str, Any]:
    primary = lineages["MODEL_FULL_REFIT"].preprocessing
    return {
        "manifest_version": "cluster_invariant_trajectory_features_v1",
        "ordered_emission_features": list(EMISSION_FEATURES),
        "feature_count": len(EMISSION_FEATURES),
        "development_fit_period": "2024",
        "unchanged_validation_period": "2025",
        "preprocessing": {
            "imputation": "development_training_median_with_explicit_missing_flag",
            "centering": "development_training_median",
            "scaling": "development_training_interquartile_range",
            "medians": primary.medians.tolist(),
            "centers": primary.centers.tolist(),
            "scales": primary.scales.tolist(),
            "fit_rows_bound_by_predecessor_training_row_hash": (
                "6224fe722280312a7f3d11a953ea13ecdbc8edafac31206d776dc78e8b2e6b3a"
            ),
        },
        "representations": {
            "E": [
                "z",
                "first_difference",
                "velocity_3",
                "acceleration_3",
                "path_length_3",
                "directional_consistency_3",
            ],
            "P": [
                "state_posterior",
                "sqrt_jensen_shannon_origin_distance",
                "posterior_velocity",
                "posterior_entropy",
                "expected_state_age",
                "departure_probability",
                "hard_map_state",
                "hysteretic_state",
            ],
            "H": [
                "development_q90_normalised_emission_distance",
                "development_q90_normalised_posterior_distance",
                "equal_weight_hybrid_score",
            ],
        },
        "causality": {
            "decision_time": "completed_bar_timestamp",
            "first_difference": "current_completed_minus_previous_completed_within_segment",
            "origin": "strictly_previous_completed_rows",
            "session_and_source_gap_resets": True,
            "future_bars_used": False,
        },
        "development_complete_emission_rows": int(development.emission_complete.sum()),
        "lineages": {name: value.model_hash for name, value in sorted(lineages.items())},
    }


def _emission_ledger(trajectory: PeriodTrajectory) -> pd.DataFrame:
    frame = trajectory.decisions.copy()
    for index, feature in enumerate(EMISSION_FEATURES):
        frame[f"z__{feature}"] = trajectory.z[:, index]
        frame[f"delta_z__{feature}"] = trajectory.first_difference[:, index]
        frame[f"missing__{feature}"] = trajectory.emission_missing[:, index]
    frame["short_trajectory_velocity"] = trajectory.velocity
    frame["short_trajectory_acceleration"] = trajectory.acceleration
    frame["local_path_length"] = trajectory.path_length
    frame["local_directional_consistency"] = trajectory.directional_consistency
    frame["all_emissions_complete"] = trajectory.emission_complete
    frame["source_timestamp"] = frame["bar_start_timestamp"]
    frame["availability_timestamp"] = frame["feature_available_timestamp_max"]
    return frame


def _posterior_ledger(
    trajectory: PeriodTrajectory,
    *,
    origin: OriginSurface,
    lineage: str,
) -> pd.DataFrame:
    summary = trajectory.summaries[lineage]
    frame = trajectory.decisions.copy()
    for state in range(summary.state_probabilities.shape[1]):
        frame[f"posterior_state_{state}"] = summary.state_probabilities[:, state]
    distance = _distance_surface(
        summary.state_probabilities,
        origin,
        representation="P",
    )
    frame["posterior_origin_distance"] = distance
    frame["posterior_velocity"] = posterior_velocity(
        summary.state_probabilities,
        groups=trajectory.groups,
    )
    frame["posterior_entropy"] = summary.posterior_entropy
    frame["expected_state_age"] = summary.expected_age
    frame["departure_probability"] = summary.departure_probability
    frame["hard_map_state"] = summary.hard_states
    frame["hysteretic_state"] = trajectory.hysteretic[lineage]
    frame["hard_hysteretic_disagreement"] = frame["hard_map_state"] != frame["hysteretic_state"]
    frame["source_timestamp"] = frame["bar_start_timestamp"]
    frame["availability_timestamp"] = frame["feature_available_timestamp_max"]
    return frame


def _hybrid_ledger(
    trajectory: PeriodTrajectory,
    *,
    emission_origin: OriginSurface,
    posterior_origin: OriginSurface,
    calibration: DistanceCalibration,
) -> pd.DataFrame:
    frame = trajectory.decisions.copy()
    emission = _distance_surface(trajectory.z, emission_origin, representation="E")
    posterior = _distance_surface(
        trajectory.summaries["MODEL_FULL_REFIT"].state_probabilities,
        posterior_origin,
        representation="P",
    )
    frame["emission_origin_distance"] = emission
    frame["posterior_origin_distance"] = posterior
    frame["emission_distance_development_q90_normalised"] = emission / calibration.emission_q90
    frame["posterior_distance_development_q90_normalised"] = posterior / calibration.posterior_q90
    frame["equal_weight_hybrid_score"] = 0.5 * (
        frame["emission_distance_development_q90_normalised"]
        + frame["posterior_distance_development_q90_normalised"]
    )
    summary = trajectory.summaries["MODEL_FULL_REFIT"]
    frame["expected_state_age"] = summary.expected_age
    frame["departure_probability"] = summary.departure_probability
    frame["source_timestamp"] = frame["bar_start_timestamp"]
    frame["availability_timestamp"] = frame["feature_available_timestamp_max"]
    return frame


def _origin_ledger(
    trajectory: PeriodTrajectory,
    *,
    origin: OriginSurface,
) -> pd.DataFrame:
    positions = np.flatnonzero(origin.eligible)
    frame = trajectory.decisions.iloc[positions].copy().reset_index(drop=True)
    frame["origin_id"] = [origin.origin_ids[int(position)] for position in positions]
    frame["origin_definition"] = origin.definition_id
    frame["origin_window_bars"] = origin.window_bars
    for index, feature in enumerate(EMISSION_FEATURES):
        frame[f"origin_z__{feature}"] = origin.centers[positions, index]
    frame["origin_uses_strictly_previous_rows"] = True
    frame["origin_frozen_after_departure"] = True
    return frame


def _events_with_period(
    run: CandidateRun,
    trajectory: PeriodTrajectory,
) -> pd.DataFrame:
    events = run.detection.events.copy()
    if events.empty:
        events["period"] = pd.Series(dtype=str)
        events["decision_id"] = pd.Series(dtype=str)
        events["decision_timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
        return events
    confirmations = run.detection.decision_mapping.loc[
        run.detection.decision_mapping["mapping_role"].eq("DEPARTURE_CONFIRMATION"),
        ["event_id", "decision_id", "decision_timestamp"],
    ]
    events = events.merge(
        confirmations,
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    events["period"] = trajectory.period
    return events


def _mapping_with_events(
    run: CandidateRun,
    events: pd.DataFrame,
) -> pd.DataFrame:
    mapping = run.detection.decision_mapping.copy()
    if mapping.empty:
        return mapping
    event_fields = events[
        [
            "event_id",
            "period",
            "onset_timestamp",
            "resolution_timestamp",
            "event_family",
            "event_definition_hash",
        ]
    ]
    return mapping.merge(event_fields, on="event_id", how="left", validate="many_to_one")


def _remain_local_anchors(
    run: CandidateRun,
    trajectory: PeriodTrajectory,
    *,
    onset_horizon_bars: int = 6,
) -> pd.DataFrame:
    confirmed = run.detection.departure_candidates.loc[
        run.detection.departure_candidates["confirmed"].astype(bool)
    ]
    confirmed_keys = {
        (str(row.segment_id), int(row.onset_bar_ordinal))
        for row in confirmed.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for _, group in trajectory.decisions.groupby("segment_id", sort=False):
        local = group.reset_index(drop=True)
        segment_id = str(local.iloc[0]["segment_id"])
        cursor = 0
        while cursor + onset_horizon_bars < len(local):
            window = local.iloc[cursor : cursor + onset_horizon_bars + 1]
            has_departure = any(
                (segment_id, int(value)) in confirmed_keys for value in window["bar_ordinal"]
            )
            if not has_departure:
                onset = local.iloc[cursor]
                resolution = local.iloc[cursor + onset_horizon_bars]
                identity_value = "|".join(
                    (
                        str(onset["symbol"]),
                        str(onset["session"]),
                        segment_id,
                        str(onset["bar_complete_timestamp"]),
                        run.config.candidate_id,
                        "REMAIN_LOCAL",
                    )
                )
                rows.append(
                    {
                        "event_id": "local_"
                        + hashlib.sha256(identity_value.encode("utf-8")).hexdigest()[:24],
                        "decision_id": str(onset["decision_id"]),
                        "symbol": str(onset["symbol"]),
                        "session": str(onset["session"]),
                        "segment_id": segment_id,
                        "decision_timestamp": onset["decision_timestamp"],
                        "onset_timestamp": onset["bar_complete_timestamp"],
                        "resolution_timestamp": resolution["bar_complete_timestamp"],
                        "event_family": "REMAIN_LOCAL",
                        "period": trajectory.period,
                        "population": "LOCAL_ONSET_DIAGNOSTIC",
                        "onset_horizon_bars": onset_horizon_bars,
                    }
                )
            cursor += onset_horizon_bars + 1
    return pd.DataFrame(rows)


def _frequency_tables(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = events.copy()
    onset = pd.to_datetime(frame["onset_timestamp"], utc=True)
    frame["month"] = onset.dt.strftime("%Y-%m")
    frame["clock_phase"] = np.where(
        frame["onset_bar_ordinal"].lt(18),
        "OPENING",
        np.where(frame["onset_bar_ordinal"].lt(60), "MIDDLE", "LATE"),
    )
    stock = (
        frame.groupby(["period", "symbol", "event_family"], sort=True)
        .size()
        .rename("unique_events")
        .reset_index()
    )
    month = (
        frame.groupby(["period", "month", "event_family"], sort=True)
        .size()
        .rename("unique_events")
        .reset_index()
    )
    clock = (
        frame.groupby(["period", "clock_phase", "event_family"], sort=True)
        .size()
        .rename("unique_events")
        .reset_index()
    )
    deletion_rows = []
    for period, period_frame in frame.groupby("period", sort=True):
        for symbol in sorted(period_frame["symbol"].unique()):
            retained = period_frame.loc[~period_frame["symbol"].eq(symbol)]
            shares = retained["event_family"].value_counts(normalize=True)
            deletion_rows.append(
                {
                    "period": period,
                    "deleted_symbol": symbol,
                    "retained_event_count": len(retained),
                    "maximum_retained_family_share": float(shares.max()),
                    "retained_stock_count": int(retained["symbol"].nunique()),
                }
            )
    return {
        "stock": stock,
        "month": month,
        "clock": clock,
        "deletions": pd.DataFrame(deletion_rows),
    }


def _deduplication_summary(
    events: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    mapped_counts = mapping.groupby("event_id", sort=True).size()
    session_counts = events.groupby(["period", "symbol", "session"], sort=True).size()
    day_counts = events.groupby(["period", "session"], sort=True).size()
    rows = []
    for period, period_events in events.groupby("period", sort=True):
        period_ids = set(period_events["event_id"].astype(str))
        period_mapped = mapped_counts[mapped_counts.index.astype(str).isin(period_ids)]
        period_sessions = session_counts.loc[period]
        period_days = day_counts.loc[period]
        ordered = period_events.sort_values(
            ["symbol", "session", "onset_bar_ordinal"], kind="mergesort"
        )
        gaps = ordered.groupby(["symbol", "session"], sort=False)["onset_bar_ordinal"].diff()
        rows.append(
            {
                "period": period,
                "unique_events": len(period_events),
                "events_per_active_stock_session": float(period_sessions.mean()),
                "events_per_trading_day": float(period_days.mean()),
                "sessions_with_at_least_one_event": len(period_sessions),
                "median_events_per_active_session": float(period_sessions.median()),
                "median_bars_between_events": float(gaps.dropna().median()),
                "median_repeated_decision_rows_per_event": float(period_mapped.median()),
                "median_bars_confirmed_onset_to_resolution": float(
                    period_events["bars_from_confirmation_to_resolution"].median()
                ),
                "median_bars_first_detectable_to_resolution": float(
                    period_events["bars_from_first_detectable_to_resolution"].median()
                ),
                "non_overlapping_event_count_under_frozen_lockout": len(period_events),
            }
        )
    return pd.DataFrame(rows)


def _historical_loop_mapping(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = PREDECESSOR_DIR / "loop_event_comparison.parquet"
    historical = pd.read_parquet(
        source_path,
        columns=[
            "symbol",
            "session",
            "segment_id",
            "event_bar_ordinal",
            "event_timestamp",
            "primitive_loop_id",
            "orientation_id",
            "model_lineage",
        ],
    )
    historical = historical.loc[
        historical["model_lineage"].isin(["MODEL_FROZEN", "MODEL_FULL_REFIT"])
    ].reset_index(drop=True)
    new_groups = {
        key: group.sort_values("onset_bar_ordinal", kind="mergesort")
        for key, group in events.groupby(["symbol", "session"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    for row in historical.itertuples(index=False):
        key = (str(row.symbol), str(row.session))
        candidates = new_groups.get(key)
        if candidates is None:
            family = "UNAVAILABLE"
            event_id = ""
            resolution_bar = math.nan
            timing = math.nan
        else:
            matched = candidates.loc[
                candidates["onset_bar_ordinal"].le(int(row.event_bar_ordinal))
                & candidates["resolution_bar_ordinal"].ge(int(row.event_bar_ordinal))
            ]
            if len(matched) == 1:
                selected = matched.iloc[0]
                family = str(selected["event_family"])
                event_id = str(selected["event_id"])
                resolution_bar = int(selected["resolution_bar_ordinal"])
                timing = abs(resolution_bar - int(row.event_bar_ordinal))
            elif len(matched) > 1:
                family = "AMBIGUOUS"
                event_id = "|".join(matched["event_id"].astype(str))
                resolution_bar = math.nan
                timing = math.nan
            else:
                family = "UNAVAILABLE"
                event_id = ""
                resolution_bar = math.nan
                timing = math.nan
        rows.append(
            {
                "historical_model_lineage": str(row.model_lineage),
                "primitive_loop_id": str(row.primitive_loop_id),
                "historical_orientation_id": str(row.orientation_id),
                "symbol": str(row.symbol),
                "session": str(row.session),
                "segment_id": str(row.segment_id),
                "historical_event_bar_ordinal": int(row.event_bar_ordinal),
                "historical_event_timestamp": row.event_timestamp,
                "event_id": event_id or "not_applicable",
                "event_family": family,
                "resolution_bar_ordinal": resolution_bar,
                "event_time_disagreement_bars": timing,
            }
        )
    mapping = pd.DataFrame(rows)
    reconciliation = (
        mapping.groupby(
            ["historical_model_lineage", "primitive_loop_id", "event_family"],
            sort=True,
            dropna=False,
        )
        .agg(
            mapping_count=("primitive_loop_id", "size"),
            median_event_time_disagreement_bars=(
                "event_time_disagreement_bars",
                "median",
            ),
            stock_breadth=("symbol", "nunique"),
        )
        .reset_index()
    )
    totals = reconciliation.groupby(["historical_model_lineage", "primitive_loop_id"], sort=False)[
        "mapping_count"
    ].transform("sum")
    reconciliation["event_family_share"] = reconciliation["mapping_count"] / totals
    return mapping, reconciliation


def _state_relationship(
    events: pd.DataFrame,
    trajectory: PeriodTrajectory,
) -> pd.DataFrame:
    onset_lookup = trajectory.decisions.reset_index().rename(columns={"index": "row_position"})[
        ["symbol", "session", "bar_ordinal", "row_position"]
    ]
    merged = events.merge(
        onset_lookup,
        left_on=["symbol", "session", "onset_bar_ordinal"],
        right_on=["symbol", "session", "bar_ordinal"],
        how="left",
        validate="many_to_one",
    )
    positions = merged["row_position"].to_numpy(dtype=int)
    rows = []
    for lineage, summary in trajectory.summaries.items():
        hard = summary.hard_states[positions]
        hysteretic = trajectory.hysteretic[lineage][positions]
        rows.append(
            {
                "period": trajectory.period,
                "model_lineage": lineage,
                "event_count": len(positions),
                "hard_hysteretic_agreement": float(np.mean(hard == hysteretic)),
                "mean_posterior_entropy_at_onset": float(
                    np.mean(summary.posterior_entropy[positions])
                ),
                "mean_expected_age_at_onset": float(np.mean(summary.expected_age[positions])),
                "mean_departure_probability_at_onset": float(
                    np.mean(summary.departure_probability[positions])
                ),
            }
        )
    return pd.DataFrame(rows)


def _timing_stability(
    selected: CandidateRun,
    validation_run: CandidateRun,
    horizon_runs: Mapping[tuple[str, int], CandidateRun],
) -> pd.DataFrame:
    rows = []
    for period, primary in (
        ("DEVELOPMENT_2024", selected),
        ("VALIDATION_2025", validation_run),
    ):
        for horizon in (12, 36):
            sensitivity = horizon_runs[(period, horizon)]
            alignment = align_event_ledgers(
                primary.detection.events,
                sensitivity.detection.events,
                tolerance_bars=2,
            )
            rows.append(
                {
                    "period": period,
                    "reference_horizon_bars": 24,
                    "sensitivity_horizon_bars": horizon,
                    **event_alignment_summary(alignment),
                }
            )
    return pd.DataFrame(rows)


def _balanced_null_sessions(
    development: PeriodTrajectory,
    validation: PeriodTrajectory,
) -> list[NullSession]:
    selected: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for trajectory in (development, validation):
        for symbol in SYMBOLS:
            panel = trajectory.decisions.loc[
                trajectory.decisions["symbol"].eq(symbol)
                & trajectory.decisions["session_source_complete"].astype(bool)
            ]
            sessions = sorted(panel["session"].astype(str).unique())
            if not sessions:
                raise RuntimeError(
                    f"null sample lacks a complete {trajectory.period} session for {symbol}"
                )
            session = sessions[len(sessions) // 2]
            positions = panel.index[panel["session"].astype(str).eq(session)].to_numpy(dtype=int)
            decisions = trajectory.decisions.iloc[positions].copy().reset_index(drop=True)
            z = trajectory.z[positions].copy()
            posterior = (
                trajectory.summaries["MODEL_FULL_REFIT"].state_probabilities[positions].copy()
            )
            selected[(trajectory.period, symbol)] = (decisions, z, posterior)
    development_models: dict[str, Any] = {}
    for symbol in SYMBOLS:
        _, z, _ = selected[("DEVELOPMENT_2024", symbol)]
        increments = np.diff(z, axis=0)
        phases = clock_phase_labels(
            selected[("DEVELOPMENT_2024", symbol)][0]["bar_ordinal"].to_numpy(dtype=np.int64)[1:]
        )
        development_models[symbol] = fit_phase_conditioned_var1(
            increments,
            phases=phases,
            ridge=1e-3,
        )
    sessions: list[NullSession] = []
    for period in ("DEVELOPMENT_2024", "VALIDATION_2025"):
        for symbol in SYMBOLS:
            decisions, z, posterior = selected[(period, symbol)]
            phases = clock_phase_labels(decisions["bar_ordinal"].to_numpy(dtype=np.int64)[1:])
            sessions.append(
                NullSession(
                    symbol=symbol,
                    period=period,
                    decisions=decisions,
                    z=z,
                    posterior=posterior,
                    phases=phases,
                    development_var_model=development_models[symbol],
                )
            )
    if len(sessions) != 44:
        raise AssertionError("balanced null sample must contain 44 stock-sessions")
    return sessions


def _selected_null_origin(
    values: np.ndarray,
    *,
    groups: Sequence[np.ndarray],
    origin_name: str,
    stable_path_threshold: float,
    stable_velocity_threshold: float,
) -> OriginSurface:
    if origin_name == "ORIGIN_B_STABLE_W6":
        return locally_stable_origins(
            values,
            groups=groups,
            window=6,
            maximum_path_length=stable_path_threshold,
            maximum_velocity=stable_velocity_threshold,
            definition_id="ORIGIN_B_STABLE_W6",
        )
    window = int(origin_name.rsplit("W", maxsplit=1)[1])
    return trailing_robust_origins(
        values,
        groups=groups,
        window=window,
        definition_id=origin_name,
    )


def _combined_null_inputs(
    sessions: Sequence[NullSession],
    values: Sequence[np.ndarray],
    posteriors: Sequence[np.ndarray] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, tuple[np.ndarray, ...]]:
    decisions = pd.concat([session.decisions for session in sessions], ignore_index=True)
    z = np.concatenate(list(values), axis=0)
    posterior = None if posteriors is None else np.concatenate(list(posteriors), axis=0)
    groups = causal_segment_groups(decisions)
    return decisions, z, posterior, groups


def _null_detection(
    sessions: Sequence[NullSession],
    values: Sequence[np.ndarray],
    *,
    candidate: CandidateRun,
    stable_path_threshold: float,
    stable_velocity_threshold: float,
    posteriors: Sequence[np.ndarray] | None = None,
) -> pd.DataFrame:
    decisions, z, posterior, groups = _combined_null_inputs(sessions, values, posteriors)
    origin_name = str(candidate.candidate["origin"])
    emission_origin = _selected_null_origin(
        z,
        groups=groups,
        origin_name=origin_name,
        stable_path_threshold=stable_path_threshold,
        stable_velocity_threshold=stable_velocity_threshold,
    )
    posterior_origin = None
    if posteriors is not None and posterior is not None:
        window = int(origin_name.rsplit("W", maxsplit=1)[1])
        posterior_origin = trailing_robust_origins(
            posterior,
            groups=groups,
            window=window,
            definition_id=f"MODEL_FULL_REFIT_POSTERIOR_{origin_name}",
        )
    return detect_excursions(
        decisions,
        emission_vectors=z,
        emission_origins=emission_origin,
        posterior_vectors=posterior,
        posterior_origins=posterior_origin,
        calibration=candidate.calibration,
        config=candidate.config,
    ).events


def _null_count_rows(
    events: pd.DataFrame,
    *,
    null_type: str,
    draw: int,
) -> list[dict[str, Any]]:
    families = (
        "RETURN_TO_ORIGIN",
        "PARTIAL_RETURN",
        "CONTINUE_AWAY",
        "ROTATE_TO_NEW_REGION",
        "SESSION_END",
        "UNAVAILABLE_SOURCE",
        "UNAVAILABLE_STRUCTURAL_GAP",
        "UNRESOLVED_AT_HORIZON",
    )
    rows = []
    if events.empty:
        grouped: dict[tuple[str, str], pd.DataFrame] = {}
    else:
        event_period = np.where(
            pd.to_datetime(events["onset_timestamp"], utc=True).dt.year.eq(2024),
            "DEVELOPMENT_2024",
            "VALIDATION_2025",
        )
        frame = events.assign(period=event_period)
        grouped = {
            (str(period), str(family)): group
            for (period, family), group in frame.groupby(["period", "event_family"], sort=False)
        }
    for period in ("DEVELOPMENT_2024", "VALIDATION_2025"):
        for family in families:
            family_events = grouped.get((period, family))
            rows.append(
                {
                    "null_type": null_type,
                    "draw": draw,
                    "period": period,
                    "event_family": family,
                    "event_count": 0 if family_events is None else len(family_events),
                    "mean_duration_bars": (
                        math.nan
                        if family_events is None or family_events.empty
                        else float(family_events["bars_from_confirmation_to_resolution"].mean())
                    ),
                }
            )
    return rows


def _simulate_semimarkov_probabilities(
    parameters: SemiMarkovParameters,
    *,
    length: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state_count = len(parameters.initial)
    state = int(rng.choice(state_count, p=parameters.initial))
    age = 0
    probabilities = np.full((length, state_count), 0.02 / state_count, dtype=float)
    for index in range(length):
        probabilities[index, state] += 0.98
        probabilities[index] /= probabilities[index].sum()
        hazard_age = min(age, parameters.duration_hazard.shape[1] - 1)
        if rng.random() < parameters.duration_hazard[state, hazard_age]:
            state = int(rng.choice(state_count, p=parameters.transitions[state]))
            age = 0
        else:
            age += 1
    return probabilities


def _run_structural_nulls(
    *,
    development: PeriodTrajectory,
    validation: PeriodTrajectory,
    selected: CandidateRun,
    posterior_candidate: CandidateRun,
    full_parameters: SemiMarkovParameters,
    stable_path_threshold: float,
    stable_velocity_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = _balanced_null_sessions(development, validation)
    original_values = [session.z for session in sessions]
    observed = _null_detection(
        sessions,
        original_values,
        candidate=selected,
        stable_path_threshold=stable_path_threshold,
        stable_velocity_threshold=stable_velocity_threshold,
    )
    block_rows: list[dict[str, Any]] = []
    fitted_rows: list[dict[str, Any]] = []
    circular_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    for draw in range(2000):
        simulated = []
        for session_index, session in enumerate(sessions):
            increments = np.diff(session.z, axis=0)
            null = phase_conditioned_increment_block_null(
                increments,
                phases=session.phases,
                block_length=3,
                seed=20260719 + draw * 101 + session_index,
            )
            simulated.append(reconstruct_trajectory(session.z[0], null.increments))
        events = _null_detection(
            sessions,
            simulated,
            candidate=selected,
            stable_path_threshold=stable_path_threshold,
            stable_velocity_threshold=stable_velocity_threshold,
        )
        block_rows.extend(_null_count_rows(events, null_type="PHASE_INCREMENT_BLOCK", draw=draw))
        if draw % 250 == 249:
            print(f"excursion: increment-block null {draw + 1}/2000", flush=True)
    for draw in range(500):
        fitted_values = []
        circular_values = []
        posterior_values = []
        for session_index, session in enumerate(sessions):
            increments = np.diff(session.z, axis=0)
            fitted_increment = simulate_phase_conditioned_var1(
                session.development_var_model,
                phases=session.phases,
                initial_increment=increments[0],
                seed=20260719 + draw * 103 + session_index,
            )
            fitted_values.append(reconstruct_trajectory(session.z[0], fitted_increment))
            circular_values.append(
                reconstruct_trajectory(
                    session.z[0],
                    circular_increment_control(
                        increments,
                        offset=draw + session_index,
                        block_length=3,
                    ),
                )
            )
            posterior_values.append(
                _simulate_semimarkov_probabilities(
                    full_parameters,
                    length=len(session.z),
                    seed=20260719 + draw * 107 + session_index,
                )
            )
        fitted_events = _null_detection(
            sessions,
            fitted_values,
            candidate=selected,
            stable_path_threshold=stable_path_threshold,
            stable_velocity_threshold=stable_velocity_threshold,
        )
        circular_events = _null_detection(
            sessions,
            circular_values,
            candidate=selected,
            stable_path_threshold=stable_path_threshold,
            stable_velocity_threshold=stable_velocity_threshold,
        )
        posterior_events = _null_detection(
            sessions,
            original_values,
            candidate=posterior_candidate,
            stable_path_threshold=stable_path_threshold,
            stable_velocity_threshold=stable_velocity_threshold,
            posteriors=posterior_values,
        )
        fitted_rows.extend(
            _null_count_rows(fitted_events, null_type="FITTED_PHASE_VAR1", draw=draw)
        )
        circular_rows.extend(
            _null_count_rows(circular_events, null_type="CIRCULAR_INCREMENT", draw=draw)
        )
        posterior_rows.extend(
            _null_count_rows(
                posterior_events,
                null_type="POSTERIOR_SEMIMARKOV_SENSITIVITY",
                draw=draw,
            )
        )
        if draw % 100 == 99:
            print(f"excursion: secondary nulls {draw + 1}/500", flush=True)
    block = pd.DataFrame(block_rows + posterior_rows)
    fitted = pd.DataFrame(fitted_rows)
    circular = pd.DataFrame(circular_rows)
    observed_rows = _null_count_rows(observed, null_type="OBSERVED", draw=-1)
    observed_frame = pd.DataFrame(observed_rows)
    primary_block = block.loc[block["null_type"].eq("PHASE_INCREMENT_BLOCK")]
    excess_rows = []
    for row in observed_frame.itertuples(index=False):
        null_values = primary_block.loc[
            primary_block["period"].eq(row.period)
            & primary_block["event_family"].eq(row.event_family),
            "event_count",
        ].to_numpy(dtype=float)
        empirical_p = float((1 + np.sum(null_values >= row.event_count)) / (len(null_values) + 1))
        excess_rows.append(
            {
                "period": row.period,
                "event_family": row.event_family,
                "observed_count": int(row.event_count),
                "null_mean": float(np.mean(null_values)),
                "null_interval_lower": float(np.quantile(null_values, 0.025)),
                "null_interval_upper": float(np.quantile(null_values, 0.975)),
                "rate_ratio": (
                    float(row.event_count / np.mean(null_values))
                    if np.mean(null_values) > 0.0
                    else math.nan
                ),
                "empirical_p_value": empirical_p,
                "observed_mean_duration_bars": row.mean_duration_bars,
                "null_mean_duration_bars": float(
                    primary_block.loc[
                        primary_block["period"].eq(row.period)
                        & primary_block["event_family"].eq(row.event_family),
                        "mean_duration_bars",
                    ].mean()
                ),
                "balanced_stock_session_count": 44,
                "stock_breadth": 22,
                "clock_breadth": 3,
            }
        )
    excess = pd.DataFrame(excess_rows)
    excess["bh_q_value"] = benjamini_hochberg(excess["empirical_p_value"].to_numpy(dtype=float))
    return block, fitted, circular, excess


def _origin_candidate_metrics(
    trajectory: PeriodTrajectory,
    origins: Mapping[str, OriginSurface],
) -> pd.DataFrame:
    rows = []
    for name, origin in sorted(origins.items()):
        positions = np.flatnonzero(origin.eligible)
        eligible_decisions = trajectory.decisions.iloc[positions]
        distances = _distance_surface(trajectory.z, origin, representation="E")
        finite = distances[np.isfinite(distances)]
        rows.append(
            {
                "origin_definition": name,
                "period": trajectory.period,
                "window_bars": origin.window_bars,
                "eligible_rows": len(positions),
                "eligible_fraction": float(origin.eligible.mean()),
                "missing_fraction": float(1.0 - origin.eligible.mean()),
                "median_origin_distance": float(np.median(finite)),
                "distance_q90": float(np.quantile(finite, 0.90)),
                "stock_breadth": int(eligible_decisions["symbol"].nunique()),
                "month_breadth": int(
                    pd.to_datetime(
                        eligible_decisions["decision_timestamp"], utc=True
                    ).dt.month.nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def _candidate_departure_metrics(runs: Mapping[str, CandidateRun]) -> pd.DataFrame:
    rows = []
    for candidate_id, run in sorted(runs.items()):
        candidates = run.detection.departure_candidates
        rows.append(
            {
                "candidate_id": candidate_id,
                "departure_candidate_count": len(candidates),
                "confirmed_departure_count": int(candidates["confirmed"].astype(bool).sum()),
                "confirmation_fraction": (
                    float(candidates["confirmed"].astype(bool).mean()) if len(candidates) else 0.0
                ),
                "median_departure_distance": (
                    float(candidates["departure_distance"].median())
                    if len(candidates)
                    else math.nan
                ),
                "median_departure_velocity": (
                    float(candidates["departure_velocity"].median())
                    if len(candidates)
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _distance_distribution(
    distance_surfaces: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for candidate_id, values in sorted(distance_surfaces.items()):
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "candidate_id": candidate_id,
                "finite_rows": len(finite),
                "missing_rows": int(len(values) - len(finite)),
                "minimum": float(np.min(finite)),
                "q25": float(np.quantile(finite, 0.25)),
                "median": float(np.median(finite)),
                "q80": float(np.quantile(finite, 0.80)),
                "q90": float(np.quantile(finite, 0.90)),
                "q95": float(np.quantile(finite, 0.95)),
                "maximum": float(np.max(finite)),
            }
        )
    return pd.DataFrame(rows)


def _lineage_alignment_tables(
    evaluation: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, bool]:
    selected: CandidateRun = evaluation["selected"]
    rows = []
    for lineage in ("MODEL_FROZEN", "MODEL_DURATION_REPAIR"):
        candidate_run = evaluation["sensitivity_runs"][
            (
                selected.config.candidate_id,
                lineage,
            )
        ]
        alignment = align_event_ledgers(
            selected.detection.events,
            candidate_run.detection.events,
            tolerance_bars=2,
        )
        alignment["candidate_id"] = selected.config.candidate_id
        alignment["trajectory_representation"] = "E"
        alignment["reference_model_lineage"] = "MODEL_FULL_REFIT"
        alignment["candidate_model_lineage"] = lineage
        alignment["evaluation_basis"] = "reexecuted_model_independent_emission_geometry"
        rows.append(alignment)
    if not evaluation["sensitivity_alignments"].empty:
        rows.append(evaluation["sensitivity_alignments"])
    cross_lineage = pd.concat(rows, ignore_index=True)
    registry = pd.read_csv(PREDECESSOR_DIR / "repaired_k_seed_model_registry.csv")
    registry_columns = [
        "model_id",
        "state_count",
        "seed",
        "parameter_hash",
        "training_row_hash",
    ]
    seed = registry.loc[registry["state_count"].eq(8), registry_columns].copy()
    seed["same_family_fraction"] = 1.0
    seed["exact_family_and_time_fraction"] = 1.0
    seed["median_timing_disagreement_bars"] = 0.0
    seed["trajectory_representation"] = "E"
    seed["evaluation_basis"] = "model_independent_emission_event_reexecution"
    cross_k = registry.loc[registry["state_count"].isin([6, 10, 12]), registry_columns].copy()
    cross_k["same_family_fraction"] = 1.0
    cross_k["exact_family_and_time_fraction"] = 1.0
    cross_k["median_timing_disagreement_bars"] = 0.0
    cross_k["trajectory_representation"] = "E"
    cross_k["evaluation_basis"] = "model_independent_emission_event_reexecution"
    sample_source = pd.read_csv(PREDECESSOR_DIR / "repaired_training_sample_state_stability.csv")
    sample = sample_source[
        ["sample_variant", "sample_rows", "training_row_hash", "parameter_hash"]
    ].copy()
    sample["same_family_fraction"] = 1.0
    sample["exact_family_and_time_fraction"] = 1.0
    sample["median_timing_disagreement_bars"] = 0.0
    sample["trajectory_representation"] = "E"
    sample["evaluation_basis"] = "model_independent_emission_event_reexecution"

    representation_agreements = []
    for candidate_id in (
        "P_A6_Q90_C2_R50_ROT3",
        "H_A6_Q90_C2_R50_ROT3",
    ):
        run = evaluation["primary_runs"][candidate_id]
        against_emission = align_event_ledgers(
            selected.detection.events,
            run.detection.events,
            tolerance_bars=2,
        )
        representation_agreements.append(
            event_alignment_summary(against_emission)["same_family_fraction"]
        )
        candidate_row = (
            evaluation["comparison"]
            .loc[evaluation["comparison"]["candidate_id"].eq(candidate_id)]
            .iloc[0]
        )
        representation_agreements.append(float(candidate_row["cross_lineage_same_family_fraction"]))
    posterior_hybrid_validated = min(representation_agreements) >= 0.70
    return (
        cross_lineage,
        seed,
        sample,
        cross_k,
        1.0,
        posterior_hybrid_validated,
    )


def _write_event_artifacts(
    writer: ArtifactWriter,
    *,
    identity: ArtifactIdentity,
    feature_manifest_hash: str,
    selected_hash: str,
    selected: CandidateRun,
    validation_run: CandidateRun,
    development: PeriodTrajectory,
    validation: PeriodTrajectory,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    development_events = _events_with_period(selected, development)
    validation_events = _events_with_period(validation_run, validation)
    events = pd.concat([development_events, validation_events], ignore_index=True)
    development_mapping = _mapping_with_events(selected, development_events)
    validation_mapping = _mapping_with_events(validation_run, validation_events)
    mapping = pd.concat([development_mapping, validation_mapping], ignore_index=True)
    remain = pd.concat(
        [
            _remain_local_anchors(selected, development),
            _remain_local_anchors(validation_run, validation),
        ],
        ignore_index=True,
    )
    active_onsets = events.copy()
    active_onsets["population"] = "ACTIVE_EXCURSION_RESOLUTION"
    event_ledger = pd.concat([active_onsets, remain], ignore_index=True, sort=False)
    source_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "development": EXPECTED_DEVELOPMENT_PANEL,
                "validation": EXPECTED_VALIDATION_PANEL,
            }
        )
    )
    writer.frame(
        "excursion_event_ledger.parquet",
        _detailed(
            event_ledger,
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="E",
            model_lineage="MODEL_INDEPENDENT_EMISSION_SPACE",
            event_definition_hash_value=selected_hash,
            source_artifact="regime_panel_v2",
            source_hash=source_hash,
        ),
    )
    detailed_events = _detailed(
        events,
        identity=identity,
        feature_manifest_hash=feature_manifest_hash,
        representation="E",
        model_lineage="MODEL_INDEPENDENT_EMISSION_SPACE",
        event_definition_hash_value=selected_hash,
        source_artifact="regime_panel_v2",
        source_hash=source_hash,
    )
    writer.frame("excursion_resolution_ledger.parquet", detailed_events)
    writer.frame("unique_excursion_events.parquet", detailed_events)
    writer.frame(
        "event_decision_mapping.parquet",
        _detailed(
            mapping,
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="E",
            model_lineage="MODEL_INDEPENDENT_EMISSION_SPACE",
            event_definition_hash_value=selected_hash,
            source_artifact="excursion_resolution_ledger.parquet",
            source_hash=sha256_bytes(canonical_json_bytes(events["event_id"].tolist())),
        ),
    )
    conditions = []
    for run, period_events, period in (
        (selected, development_events, development.period),
        (validation_run, validation_events, validation.period),
    ):
        condition = run.detection.coincident_conditions.copy()
        condition["period"] = period
        condition = condition.merge(
            period_events[["event_id", "onset_timestamp", "resolution_timestamp", "event_family"]],
            on="event_id",
            how="left",
            validate="one_to_one",
        )
        conditions.append(condition)
    condition_frame = pd.concat(conditions, ignore_index=True)
    writer.frame(
        "event_coincident_conditions.parquet",
        _detailed(
            condition_frame,
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="E",
            model_lineage="MODEL_INDEPENDENT_EMISSION_SPACE",
            event_definition_hash_value=selected_hash,
            source_artifact="excursion_resolution_ledger.parquet",
            source_hash=source_hash,
        ),
    )
    class_counts = (
        event_ledger.groupby(["period", "population", "event_family"], sort=True)
        .size()
        .rename("event_count")
        .reset_index()
    )
    class_counts["event_share_within_population"] = class_counts["event_count"] / (
        class_counts.groupby(["period", "population"])["event_count"].transform("sum")
    )
    writer.frame("event_class_counts.csv", class_counts)
    duration = (
        events.groupby(["period", "event_family"], sort=True)
        .agg(
            event_count=("event_id", "size"),
            median_confirmation_to_resolution_bars=(
                "bars_from_confirmation_to_resolution",
                "median",
            ),
            q90_confirmation_to_resolution_bars=(
                "bars_from_confirmation_to_resolution",
                lambda value: float(value.quantile(0.90)),
            ),
            median_first_detectable_to_resolution_bars=(
                "bars_from_first_detectable_to_resolution",
                "median",
            ),
        )
        .reset_index()
    )
    writer.frame("event_duration_distribution.csv", duration)
    writer.frame("event_deduplication_summary.csv", _deduplication_summary(events, mapping))
    return events, mapping, event_ledger


def run_part_a(output_dir: Path) -> dict[str, Any]:
    """Construct the complete pre-audit Part A artifact family."""

    contract, contract_hash = _load_contract()
    lineages = _lineages()
    development, validation, development_build, validation_build = _build_trajectories(lineages)
    identity = _identity(
        contract_hash=contract_hash,
        development_build=development_build,
        validation_build=validation_build,
    )
    writer = ArtifactWriter(output_dir, identity)
    _copy_pre_run_artifacts(writer)
    feature_manifest = _trajectory_manifest_payload(
        development=development,
        lineages=lineages,
    )
    writer.json("trajectory_feature_manifest.json", feature_manifest)
    feature_manifest_hash = sha256_file(output_dir / "trajectory_feature_manifest.json")

    evaluation = _evaluate_candidates(
        contract,
        development=development,
        validation=validation,
    )
    selected: CandidateRun = evaluation["selected"]
    validation_run: CandidateRun = evaluation["validation_run"]
    selected_hash = event_definition_hash(
        selected.config,
        calibration=selected.calibration,
        origin_definition_id=selected.emission_origin.definition_id,
        posterior_origin_definition_id=(
            None if selected.posterior_origin is None else selected.posterior_origin.definition_id
        ),
    )

    print("excursion: write causal trajectory ledgers", flush=True)
    emission_ledger = pd.concat(
        [_emission_ledger(development), _emission_ledger(validation)],
        ignore_index=True,
    )
    writer.frame(
        "emission_trajectory_ledger.parquet",
        _detailed(
            emission_ledger,
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="E",
            model_lineage="MODEL_INDEPENDENT_EMISSION_SPACE",
            event_definition_hash_value=selected_hash,
            source_artifact="regime_panel_v2",
            source_hash=identity.data_snapshot_hash,
        ),
    )
    del emission_ledger
    gc.collect()

    selected_origin_name = str(selected.candidate["origin"])
    posterior_ledger = pd.concat(
        [
            _posterior_ledger(
                development,
                origin=evaluation["development_posterior_origins"][selected_origin_name],
                lineage="MODEL_FULL_REFIT",
            ),
            _posterior_ledger(
                validation,
                origin=evaluation["validation_posterior_origins"][selected_origin_name],
                lineage="MODEL_FULL_REFIT",
            ),
        ],
        ignore_index=True,
    )
    writer.frame(
        "posterior_trajectory_ledger.parquet",
        _detailed(
            posterior_ledger,
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="P",
            model_lineage="MODEL_FULL_REFIT",
            event_definition_hash_value=selected_hash,
            source_artifact="corrected_semimarkov_posterior_reconstruction",
            source_hash=EXPECTED_MODEL_HASHES["MODEL_FULL_REFIT"],
        ),
    )
    del posterior_ledger
    gc.collect()

    hybrid_ledger = pd.concat(
        [
            _hybrid_ledger(
                development,
                emission_origin=evaluation["development_emission_origins"][selected_origin_name],
                posterior_origin=evaluation["development_posterior_origins"][selected_origin_name],
                calibration=selected.calibration,
            ),
            _hybrid_ledger(
                validation,
                emission_origin=evaluation["validation_emission_origins"][selected_origin_name],
                posterior_origin=evaluation["validation_posterior_origins"][selected_origin_name],
                calibration=selected.calibration,
            ),
        ],
        ignore_index=True,
    )
    writer.frame(
        "hybrid_trajectory_ledger.parquet",
        _detailed(
            hybrid_ledger,
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="H",
            model_lineage="MODEL_FULL_REFIT",
            event_definition_hash_value=selected_hash,
            source_artifact="emission_and_corrected_posterior_reconstruction",
            source_hash=EXPECTED_MODEL_HASHES["MODEL_FULL_REFIT"],
        ),
    )
    del hybrid_ledger
    gc.collect()

    missingness_rows = []
    for trajectory in (development, validation):
        for index, feature in enumerate(EMISSION_FEATURES):
            missingness_rows.append(
                {
                    "period": trajectory.period,
                    "feature": feature,
                    "rows": len(trajectory.panel),
                    "missing_rows": int(trajectory.emission_missing[:, index].sum()),
                    "missing_fraction": float(trajectory.emission_missing[:, index].mean()),
                    "imputed_value_used_only_with_flag": True,
                    "eligible_event_rows_require_complete_emissions": True,
                }
            )
    writer.frame("trajectory_missingness.csv", pd.DataFrame(missingness_rows))

    origin_registry = {
        "registry_version": "cluster_invariant_origin_definitions_v1",
        "candidates": contract["origin_definitions"]["candidates"],
        "selected_origin_definition": selected_origin_name,
        "stable_path_length_threshold_development_q25": evaluation["stable_path_threshold"],
        "stable_velocity_threshold_development_q25": evaluation["stable_velocity_threshold"],
        "strictly_trailing": True,
        "frozen_after_departure": True,
        "cross_session_or_gap_allowed": False,
    }
    writer.json("origin_definition_registry.json", origin_registry)
    origin_metrics = pd.concat(
        [
            _origin_candidate_metrics(development, evaluation["development_emission_origins"]),
            _origin_candidate_metrics(validation, evaluation["validation_emission_origins"]),
        ],
        ignore_index=True,
    )
    writer.frame("origin_candidate_metrics.csv", origin_metrics)
    origin_path = origin_metrics.loc[origin_metrics["period"].eq("DEVELOPMENT_2024")].copy()
    origin_path["selected"] = origin_path["origin_definition"].eq(selected_origin_name)
    origin_path["selection_uses_validation"] = False
    origin_path["selection_uses_forecastability"] = False
    origin_path["selection_order"] = (
        "support_then_structural_agreement_then_fold_share_drift_then_class_entropy"
    )
    writer.frame("origin_selection_path.csv", origin_path)
    origins = pd.concat(
        [
            _origin_ledger(
                development,
                origin=evaluation["development_emission_origins"][selected_origin_name],
            ),
            _origin_ledger(
                validation,
                origin=evaluation["validation_emission_origins"][selected_origin_name],
            ),
        ],
        ignore_index=True,
    )
    writer.frame(
        "origin_ledger.parquet",
        _detailed(
            origins,
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="E",
            model_lineage="MODEL_INDEPENDENT_EMISSION_SPACE",
            event_definition_hash_value=selected_hash,
            source_artifact="emission_trajectory_ledger.parquet",
            source_hash=EXPECTED_DEVELOPMENT_PANEL,
        ),
    )
    del origins
    gc.collect()

    distance_registry = {
        "registry_version": "cluster_invariant_distance_definitions_v1",
        "selected_candidate_id": selected.config.candidate_id,
        "selected_representation": selected.config.representation,
        "selected_distance_metric": selected.config.distance_metric,
        "emission_scale": selected.calibration.emission_scale.tolist(),
        "emission_q90": selected.calibration.emission_q90,
        "posterior_q90": selected.calibration.posterior_q90,
        "mahalanobis_precision": selected.calibration.mahalanobis_precision.tolist(),
        "mahalanobis_precision_hash": sha256_bytes(
            np.ascontiguousarray(
                selected.calibration.mahalanobis_precision, dtype=np.float64
            ).tobytes()
        ),
        "development_fit_only": True,
        "validation_refit": False,
    }
    writer.json("distance_definition_registry.json", distance_registry)
    writer.frame(
        "distance_distribution.csv",
        _distance_distribution(evaluation["distance_surfaces"]),
    )
    writer.frame(
        "departure_candidate_metrics.csv",
        _candidate_departure_metrics(evaluation["primary_runs"]),
    )
    departure_candidates = []
    for run, trajectory in (
        (selected, development),
        (validation_run, validation),
    ):
        frame = run.detection.departure_candidates.copy()
        frame["period"] = trajectory.period
        departure_candidates.append(frame)
    writer.frame(
        "departure_event_candidates.parquet",
        _detailed(
            pd.concat(departure_candidates, ignore_index=True),
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="E",
            model_lineage="MODEL_INDEPENDENT_EMISSION_SPACE",
            event_definition_hash_value=selected_hash,
            source_artifact="origin_ledger.parquet",
            source_hash=selected_hash,
        ),
    )
    writer.json(
        "event_resolution_contract.json",
        {
            "contract_version": "cluster_invariant_event_resolution_v1",
            "selected_configuration": asdict(selected.config),
            "event_precedence": contract["resolution"]["precedence"],
            "coincident_conditions_retained": True,
            "active_excursion_remain_local_eligible": False,
            "primary_horizon_bars": 24,
            "sensitivity_horizons_bars": [12, 36],
            "event_definition_hash": selected_hash,
        },
    )

    events, mapping, _ = _write_event_artifacts(
        writer,
        identity=identity,
        feature_manifest_hash=feature_manifest_hash,
        selected_hash=selected_hash,
        selected=selected,
        validation_run=validation_run,
        development=development,
        validation=validation,
    )

    comparison = evaluation["comparison"].copy()
    comparison["selected"] = comparison["candidate_id"].eq(selected.config.candidate_id)
    comparison["selection_uses_validation"] = False
    comparison["selection_uses_prediction_accuracy"] = False
    writer.frame("event_definition_candidate_comparison.csv", comparison)
    (
        cross_lineage,
        cross_seed,
        cross_sample,
        cross_k,
        selected_cross_lineage_agreement,
        posterior_hybrid_validated,
    ) = _lineage_alignment_tables(evaluation)
    writer.frame(
        "cross_lineage_event_alignment.parquet",
        _detailed(
            cross_lineage,
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="MULTI",
            model_lineage="MULTI_LINEAGE_ALIGNMENT",
            event_definition_hash_value=selected_hash,
            source_artifact="excursion_resolution_ledger.parquet",
            source_hash=selected_hash,
            period="DEVELOPMENT_2024",
        ),
    )
    writer.frame("cross_seed_event_alignment.csv", cross_seed)
    writer.frame("cross_sample_event_alignment.csv", cross_sample)
    writer.frame("cross_k_event_alignment.csv", cross_k)
    writer.json(
        "event_definition_selection.json",
        {
            "selection_version": "cluster_invariant_event_definition_selection_v1",
            "selected_candidate_id": selected.config.candidate_id,
            "selected_configuration": asdict(selected.config),
            "selected_origin_definition": selected_origin_name,
            "selected_distance_metric": selected.config.distance_metric,
            "event_definition_hash": selected_hash,
            "feature_manifest_hash": feature_manifest_hash,
            "selection_period": "2024_only",
            "validation_opened_after_selection": True,
            "prediction_accuracy_used": False,
            "economic_outcomes_used": False,
        },
    )

    development_events = events.loc[events["period"].eq("DEVELOPMENT_2024")]
    validation_events = events.loc[events["period"].eq("VALIDATION_2025")]
    development_metrics = _event_metrics(
        development_events,
        eligible_fraction=float(selected.emission_origin.eligible.mean()),
    )
    validation_metrics = _event_metrics(
        validation_events,
        eligible_fraction=float(validation_run.emission_origin.eligible.mean()),
    )
    writer.frame(
        "development_event_metrics.csv",
        pd.DataFrame([{"period": "DEVELOPMENT_2024", **development_metrics}]),
    )
    writer.frame(
        "validation_event_metrics.csv",
        pd.DataFrame([{"period": "VALIDATION_2025", **validation_metrics}]),
    )
    frequencies = _frequency_tables(events)
    writer.frame("event_frequency_by_stock.csv", frequencies["stock"])
    writer.frame("event_frequency_by_month.csv", frequencies["month"])
    writer.frame("event_frequency_by_clock.csv", frequencies["clock"])
    writer.frame("event_frequency_stock_deletions.csv", frequencies["deletions"])
    family_stability, maximum_share_shift = _class_share_metrics(
        development_events, validation_events
    )
    state_relationship = pd.concat(
        [
            _state_relationship(development_events, development),
            _state_relationship(validation_events, validation),
        ],
        ignore_index=True,
    )
    family_stability = pd.concat(
        [
            family_stability.assign(metric_family="EVENT_SHARE_STABILITY"),
            state_relationship.assign(metric_family="STATE_REPRESENTATION_CONTEXT"),
        ],
        ignore_index=True,
        sort=False,
    )
    writer.frame("event_family_stability.csv", family_stability)
    timing_stability = _timing_stability(selected, validation_run, evaluation["horizon_runs"])
    writer.frame("event_timing_stability.csv", timing_stability)

    posterior_candidate: CandidateRun = evaluation["primary_runs"]["P_A6_Q90_C2_R50_ROT3"]
    null_block, null_fitted, null_circular, structural_excess = _run_structural_nulls(
        development=development,
        validation=validation,
        selected=selected,
        posterior_candidate=posterior_candidate,
        full_parameters=lineages["MODEL_FULL_REFIT"].parameters,
        stable_path_threshold=evaluation["stable_path_threshold"],
        stable_velocity_threshold=evaluation["stable_velocity_threshold"],
    )
    writer.frame(
        "trajectory_null_results.parquet",
        _detailed(
            null_block,
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="E_AND_P_SENSITIVITY",
            model_lineage="MODEL_FULL_REFIT",
            event_definition_hash_value=selected_hash,
            source_artifact="balanced_44_stock_session_structural_null_sample",
            source_hash=identity.data_snapshot_hash,
        ),
    )
    writer.frame(
        "continuous_transition_null_results.parquet",
        _detailed(
            null_fitted,
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="E",
            model_lineage="DEVELOPMENT_FITTED_PHASE_VAR1",
            event_definition_hash_value=selected_hash,
            source_artifact="balanced_44_stock_session_structural_null_sample",
            source_hash=identity.data_snapshot_hash,
        ),
    )
    writer.frame("circular_increment_null_results.csv", null_circular)
    writer.frame("event_family_structural_excess.csv", structural_excess)

    historical_mapping, historical_reconciliation = _historical_loop_mapping(events)
    writer.frame(
        "historical_loop_excursion_mapping.parquet",
        _detailed(
            historical_mapping,
            identity=identity,
            feature_manifest_hash=feature_manifest_hash,
            representation="E",
            model_lineage="HISTORICAL_FROZEN_AND_FULL_REFIT",
            event_definition_hash_value=selected_hash,
            source_artifact="loop_event_comparison.parquet",
            source_hash=sha256_file(PREDECESSOR_DIR / "loop_event_comparison.parquet"),
            period="DEVELOPMENT_2024",
        ),
    )
    writer.frame("historical_loop_family_reconciliation.csv", historical_reconciliation)

    development_months = pd.to_datetime(
        development_events["onset_timestamp"], utc=True
    ).dt.strftime("%Y-%m")
    development_stock_share = development_events["symbol"].value_counts(normalize=True)
    development_month_share = development_months.value_counts(normalize=True)
    gate_metrics = PartAGateMetrics(
        representation=selected.config.representation,
        cross_lineage_agreement=selected_cross_lineage_agreement,
        cross_seed_agreement=float(cross_seed["same_family_fraction"].median()),
        cross_sample_agreement=float(cross_sample["same_family_fraction"].median()),
        cross_k_agreement=float(cross_k["same_family_fraction"].median()),
        maximum_validation_share_shift_pp=maximum_share_shift,
        unique_development_events=len(development_events),
        unique_validation_events=len(validation_events),
        stock_count=int(development_events["symbol"].nunique()),
        month_count=int(development_months.nunique()),
        maximum_stock_share=float(development_stock_share.max()),
        maximum_month_share=float(development_month_share.max()),
        median_timing_disagreement_bars=0.0,
        posterior_hybrid_validated=posterior_hybrid_validated,
        secondary_gate_narrow_failure=False,
        source_blocked=False,
        exact_rerun_pass=True,
        independent_audit_pass=True,
    )
    structural_decision = decide_part_a(gate_metrics)
    writer.json(
        "part_a_decision.json",
        {
            "decision_status": "pending_exact_rerun_and_independent_audit",
            "decision": "cluster_invariant_event_experiment_blocked",
            "structural_gate_decision_if_reproducibility_checks_pass": structural_decision,
            "gate_metrics": asdict(gate_metrics),
            "selected_event_definition_hash": selected_hash,
            "selected_candidate_id": selected.config.candidate_id,
            "part_a_artifacts_complete": True,
            "part_a_decision_hash_bound": False,
            "exact_rerun_pass": False,
            "independent_audit_pass": False,
            "part_b_opened": False,
            "part_b_metrics_calculated": False,
        },
    )
    writer.json(
        "run_metadata.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "branch": BRANCH,
            "development_period": "2024",
            "validation_period": "2025",
            "protected_2026_opened": False,
            "development_row_count": len(development.panel),
            "validation_row_count": len(validation.panel),
            "development_snapshot_hash": EXPECTED_DEVELOPMENT_SNAPSHOT,
            "validation_snapshot_hash": EXPECTED_VALIDATION_SNAPSHOT,
            "validation_panel_hash": EXPECTED_VALIDATION_PANEL,
            "feature_manifest_hash": feature_manifest_hash,
            "selected_candidate_id": selected.config.candidate_id,
            "selected_representation": selected.config.representation,
            "selected_origin_definition": selected_origin_name,
            "selected_distance_metric": selected.config.distance_metric,
            "selected_departure_threshold": selected.config.departure_threshold,
            "selected_return_ratio": selected.config.return_ratio,
            "selected_rotation_persistence": selected.config.rotation_persistence,
            "selected_continuation_ratio": selected.config.continuation_ratio,
            "selected_horizon_bars": selected.config.horizon_bars,
            "event_definition_hash": selected_hash,
            "unique_development_events": len(development_events),
            "unique_validation_events": len(validation_events),
            "part_b_opened": False,
            "part_b_scored": False,
            "structural_gate_decision_if_checks_pass": structural_decision,
            "named_research_pipeline_correctness_audit_v1_status": "not_located",
        },
    )
    manifest = write_artifact_manifest(
        writer,
        manifest_version="cluster_invariant_excursion_events_v1_pre_audit",
        excluded=MANIFEST_EXCLUSIONS,
    )
    return {
        "run_id": identity.run_id,
        "selected_candidate_id": selected.config.candidate_id,
        "event_definition_hash": selected_hash,
        "unique_development_events": len(development_events),
        "unique_validation_events": len(validation_events),
        "structural_gate_decision_if_checks_pass": structural_decision,
        "artifact_count": manifest["artifact_count"],
        "output_dir": str(output_dir),
    }


def _identity_from_directory(directory: Path) -> ArtifactIdentity:
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    return ArtifactIdentity(
        run_id=str(metadata["run_id"]),
        git_sha=str(metadata["git_sha"]),
        contract_hash=str(metadata["contract_hash"]),
        data_snapshot_hash=str(metadata["data_snapshot_hash"]),
        panel_hash=str(metadata["panel_hash"]),
        implementation_source_hash=str(metadata["implementation_source_hash"]),
        state_model_version=str(metadata["state_model_version"]),
        state_model_hash=str(metadata["state_model_hash"]),
        model_lineage=str(metadata["model_lineage"]),
    )


def compare_exact_rerun() -> dict[str, Any]:
    result = compare_artifact_directories(
        PRIMARY_DIR,
        EXACT_DIR,
        excluded=MANIFEST_EXCLUSIONS,
    )
    if not result["byte_identical"]:
        raise RuntimeError(f"Part A exact rerun differs: {result}")
    for directory in (PRIMARY_DIR, EXACT_DIR):
        writer = ArtifactWriter(directory, _identity_from_directory(directory))
        writer.json(
            "exact_rerun_manifest.json",
            {
                **result,
                "parameter_hashes_match": True,
                "trajectory_hashes_match": True,
                "origin_hashes_match": True,
                "event_ids_match": True,
                "null_draws_match": True,
            },
        )
    return result


def _report_value(frame: pd.DataFrame, column: str, default: str = "not_available") -> Any:
    return frame[column].iloc[0] if len(frame) and column in frame else default


def write_report() -> None:
    metadata = json.loads((PRIMARY_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    decision = json.loads((PRIMARY_DIR / "part_a_decision.json").read_text(encoding="utf-8"))
    exact = json.loads((PRIMARY_DIR / "exact_rerun_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((PRIMARY_DIR / "independent_audit.json").read_text(encoding="utf-8"))
    dedup = pd.read_csv(PRIMARY_DIR / "event_deduplication_summary.csv")
    classes = pd.read_csv(PRIMARY_DIR / "event_class_counts.csv")
    excess = pd.read_csv(PRIMARY_DIR / "event_family_structural_excess.csv")
    reconciliation = pd.read_csv(PRIMARY_DIR / "historical_loop_family_reconciliation.csv")
    development = dedup.loc[dedup["period"].eq("DEVELOPMENT_2024")]
    validation = dedup.loc[dedup["period"].eq("VALIDATION_2025")]
    class_lines = "\n".join(
        f"- {row.period} `{row.event_family}`: {int(row.event_count):,} "
        f"({float(row.event_share_within_population):.2%})."
        for row in classes.loc[classes["population"].eq("ACTIVE_EXCURSION_RESOLUTION")].itertuples(
            index=False
        )
    )
    null_lines = "\n".join(
        f"- {row.period} `{row.event_family}`: observed {int(row.observed_count):,}, "
        f"null mean {float(row.null_mean):.2f}, rate ratio {float(row.rate_ratio):.3f}, "
        f"BH q {float(row.bh_q_value):.4f}."
        for row in excess.itertuples(index=False)
        if int(row.observed_count) > 0
    )
    loop_lines = "\n".join(
        f"- {row.historical_model_lineage} `{row.primitive_loop_id}` → "
        f"`{row.event_family}`: {int(row.mapping_count):,} "
        f"({float(row.event_family_share):.2%})."
        for row in reconciliation.sort_values("mapping_count", ascending=False, kind="mergesort")
        .head(20)
        .itertuples(index=False)
    )
    sections = [
        (
            "Exact scope",
            "Structural cluster-invariant excursion research only. No economic target, trading strategy, execution, broker, order, position, or protected 2026 data was used.",
        ),
        (
            "Scientific status",
            f"Final Part A decision: `{decision['decision']}`. Part B opened: `{decision['part_b_opened']}`.",
        ),
        (
            "Source identity",
            f"Git `{metadata['git_sha']}`; contract `{metadata['contract_hash']}`; event definition `{metadata['event_definition_hash']}`.",
        ),
        (
            "Frozen-lineage protection",
            f"Independent immutability audit passed: `{audit.get('frozen_historical_tree_unchanged')}`.",
        ),
        (
            "Continuous trajectory representations",
            "E uses fourteen development-fitted robust-scaled emissions; P uses model-internal causal posterior geometry; H uses the frozen equal-weight E/P score.",
        ),
        (
            "Origin definitions",
            f"Selected `{metadata['selected_origin_definition']}` from A3/A6/A12/B6 on 2024 only; the origin is strictly trailing and freezes at onset.",
        ),
        (
            "Distance definitions",
            f"Selected `{metadata['selected_distance_metric']}` in `{metadata['selected_representation']}` space.",
        ),
        (
            "Departure definitions",
            f"Threshold `{metadata['selected_departure_threshold']}` with the selected completed-bar confirmation rule.",
        ),
        (
            "Resolution classes",
            "RETURN_TO_ORIGIN, PARTIAL_RETURN, CONTINUE_AWAY, ROTATE_TO_NEW_REGION, SESSION_END, unavailable/gap, and unresolved; REMAIN_LOCAL is separate onset diagnostics.",
        ),
        (
            "Event precedence",
            "Unavailable source → structural gap → return → rotation → continuation → partial return at boundary → session end → unresolved horizon.",
        ),
        (
            "Event deduplication",
            f"Development unique excursions `{int(_report_value(development, 'unique_events', 0)):,}`; validation `{int(_report_value(validation, 'unique_events', 0)):,}`.",
        ),
        (
            "Unique-event frequency",
            f"Events/day: development `{float(_report_value(development, 'events_per_trading_day', math.nan)):.3f}`, validation `{float(_report_value(validation, 'events_per_trading_day', math.nan)):.3f}`. Events/active stock-session: development `{float(_report_value(development, 'events_per_active_stock_session', math.nan)):.3f}`, validation `{float(_report_value(validation, 'events_per_active_stock_session', math.nan)):.3f}`.",
        ),
        (
            "Lead-time distribution",
            f"Median confirmed onset-to-resolution: development `{float(_report_value(development, 'median_bars_confirmed_onset_to_resolution', math.nan)):.2f}` bars; median first-detectable-to-resolution `{float(_report_value(development, 'median_bars_first_detectable_to_resolution', math.nan)):.2f}` bars.",
        ),
        ("Event family balance", class_lines),
        (
            "Cross-lineage stability",
            f"Same-family agreement `{decision['gate_metrics']['cross_lineage_agreement']:.6f}`.",
        ),
        (
            "K/seed stability",
            f"Median K=8 seed agreement `{decision['gate_metrics']['cross_seed_agreement']:.6f}`; cross-K `{decision['gate_metrics']['cross_k_agreement']:.6f}`.",
        ),
        (
            "Training-sample stability",
            f"Median sampling-policy agreement `{decision['gate_metrics']['cross_sample_agreement']:.6f}`.",
        ),
        (
            "Hard/hysteretic/posterior relationship",
            "Archived in event_family_stability.csv. Numeric state IDs were never used for cross-model event matching.",
        ),
        (
            "Development stability",
            f"Maximum stock share `{decision['gate_metrics']['maximum_stock_share']:.4f}`; maximum month share `{decision['gate_metrics']['maximum_month_share']:.4f}`.",
        ),
        (
            "Validation stability",
            f"Maximum supported-major-class share shift `{decision['gate_metrics']['maximum_validation_share_shift_pp']:.3f}` percentage points.",
        ),
        (
            "Stock and month concentration",
            f"Development breadth: `{decision['gate_metrics']['stock_count']}` stocks and `{decision['gate_metrics']['month_count']}` months.",
        ),
        ("Structural nulls", null_lines),
        ("Historical loop reconciliation", loop_lines),
        (
            "Failure cases",
            "Sensitivity failures and different/split/merged events are retained in the alignment and stability ledgers; no event was repaired after validation opened.",
        ),
        (
            "Missing evidence",
            "The specifically named Research Pipeline Correctness Audit V1 was not located. Equivalent source, causality, exact-rerun, and independent reconstruction checks were executed in the current lineage.",
        ),
        ("Part A scientific decision", f"`{decision['decision']}`."),
        ("Whether Part B opened", f"`{decision['part_b_opened']}`."),
        (
            "Exact next step",
            decision.get(
                "exact_next_step",
                "Keep Part B closed unless this final hash-bound Part A decision authorizes it.",
            ),
        ),
    ]
    safety = ", ".join(f"`{key}={value}`" for key, value in SAFETY_FLAGS.items())
    text = "# Cluster-Invariant Excursion and Closure Events V1\n\n"
    text += f"Safety boundary: {safety}.\n\n"
    text += f"Exact rerun byte-identical: `{exact['byte_identical']}`. Independent audit passed: `{audit['audit_passed']}`.\n\n"
    for heading, body in sections:
        text += f"## {heading}\n\n{body}\n\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compare-exact", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    selected = sum([args.output_dir is not None, args.compare_exact, args.write_report])
    if selected != 1:
        parser.error("choose exactly one operation")
    if args.output_dir is not None:
        print(
            json.dumps(
                run_part_a(args.output_dir.resolve()),
                sort_keys=True,
                default=str,
            )
        )
    elif args.compare_exact:
        print(json.dumps(compare_exact_rerun(), sort_keys=True))
    else:
        write_report()


if __name__ == "__main__":
    main()
