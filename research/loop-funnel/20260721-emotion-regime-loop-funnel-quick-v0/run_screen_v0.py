#!/usr/bin/env python3
"""Run the bounded Emotion × Regime-Mix Loop Funnel Quick Screen V0."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-emotion-regime-loop-funnel-mpl")

import argparse
import hashlib
import json
import math
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.emotion_regime_loop_funnel_quick_v0 import (
    BEHAVIOURAL_DIMENSIONS,
    INTERACTION_FEATURES,
    PRIMARY_BEHAVIOURAL_FEATURES,
    SAFETY_FLAGS,
    STATE_PROBABILITY_FEATURES,
    BlockedScreen,
    build_interactions,
    decide_funnel,
    join_behavioural_ledger,
    join_v2_posteriors,
    multiclass_brier,
    permute_behavioural_bundle_within_slates,
    pool_target_class,
    prediction_entropy,
    reject_protected_dates,
    resolve_first_loop_target,
    select_exact_loop_classes,
    session_block_bootstrap_draws,
    validate_checkpoint_timing,
)
from stocker_research.loop_dictionary_v2 import LoopDictionary, decompose_closed_path
from stocker_research.loop_prefix_automaton_v2 import FirstNextLoopEventEngine
from stocker_research.regime_gap_segmentation_v2 import causal_segment_groups
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
    causal_filter_summary,
    gaussian_log_emissions,
    transform_emissions,
)

CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS_DIR = EXPERIMENT_DIR / "reports"

BEHAVIOURAL_DIR = (
    REPO_ROOT
    / "research"
    / "observable-behavioural-state"
    / "20260721-behavioural-state-dimensions-screen-v0"
)
BEHAVIOURAL_PRIMARY = BEHAVIOURAL_DIR / "artifacts" / "primary"
BEHAVIOURAL_COMPACT = BEHAVIOURAL_PRIMARY / "compact_decision_panel.parquet"
BEHAVIOURAL_DIMENSION_LEDGER = BEHAVIOURAL_PRIMARY / "behavioural_dimension_ledger.parquet"
BEHAVIOURAL_SCALING = BEHAVIOURAL_PRIMARY / "behavioural_component_scaling.json"
BEHAVIOURAL_AUDIT = BEHAVIOURAL_PRIMARY / "independent_audit.json"

OPENING_DIR = (
    REPO_ROOT
    / "research"
    / "opening-regime-path"
    / "20260720-opening-regime-path-direction-screen-v0"
)
OPENING_PRIMARY = OPENING_DIR / "artifacts" / "primary"
OPENING_PANEL = OPENING_PRIMARY / "opening_decision_panel.parquet"
OPENING_SOURCE_MANIFEST = OPENING_PRIMARY / "source_manifest.json"

SLRNO_WORK = REPO_ROOT / "research" / "slrno-v2" / "20260714-regime-loop-handoff" / "work"
REFIT_PRIMARY = SLRNO_WORK / "artifacts" / "20260719-right-censored-regime-refit-v2" / "primary"
PARAMETERS_PATH = REFIT_PRIMARY / "full_refit_parameters.npz"
PREPROCESSING_PATH = REFIT_PRIMARY / "full_refit_preprocessing.csv"
REFIT_CONFIG_PATH = REFIT_PRIMARY / "full_refit_effective_configuration.json"
DICTIONARY_PATH = (
    SLRNO_WORK
    / "artifacts"
    / "20260718-loop-event-semantics-v2"
    / "primary"
    / "semantic_loop_dictionary_v2.csv"
)

START = pd.Timestamp("2024-01-01T00:00:00Z")
DEVELOPMENT_END_EXCLUSIVE = pd.Timestamp("2025-01-01T00:00:00Z")
PROTECTED_START_UTC = pd.Timestamp("2025-08-23T00:00:00Z")
READ_END_INCLUSIVE = PROTECTED_START_UTC - pd.Timedelta(microseconds=1)
EXPECTED_SESSION_BARS = 78
MAX_TARGET_BAR_ORDINAL = 17
MAX_DECISION_ROWS = 20_000
STATE_COUNT = 8
BOOTSTRAP_DRAWS = 50
NULL_DRAWS = 10
MODEL_SEED = 20260721
BOOTSTRAP_SEED = 20260722
NULL_SEED = 20260723
CHECKPOINT_MATERIAL_ADVERSITY = -0.001

EXPECTED_MODEL_HASH = "4fc1a02dce9ac2311dabaeb4623a559d37286dfe58baffef53828cc7415a3425"
EXPECTED_PANEL_HASH = "801c0bf9d69ecdd58b21fb2ba4392137048b466668344ebfc4c8faf6a0d3e2f1"
BEHAVIOURAL_PREDECESSOR_COMMIT = "ef9c2e4a6636d5404fbc2767a50ecb10a413f14a"
V2_STRUCTURAL_PREDECESSOR_COMMIT = "5e80d972d1e003c8366a1ec6ca170d1077288ead"

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

BEHAVIOURAL_Z_COMPONENTS = (
    "z_activity_effort",
    "z_range_effort",
    "z_travel_effort",
    "z_absolute_efficiency",
    "z_close_retention",
    "z_directional_persistence",
    "z_extreme_rejection",
    "z_absolute_progress",
    "z_compression",
    "z_signed_progress",
    "z_signed_efficiency",
    "z_mean_close_location",
    "z_boundary_slope",
    "z_effort_acceleration",
    "z_aligned_progress_acceleration",
    "z_directional_rejection",
    "z_return_gap",
    "z_activity_gap",
    "z_range_gap",
)

M0_FEATURES = (
    *STATE_PROBABILITY_FEATURES,
    "posterior_entropy",
    "top_state_probability",
    "top_second_margin",
    "expected_state_age",
    "persistence_probability",
    "transition_probability",
    "remaining_session_bars",
    "checkpoint",
)
M1_FEATURES = (*M0_FEATURES, *PRIMARY_BEHAVIOURAL_FEATURES)
M2_FEATURES = (*M1_FEATURES, *INTERACTION_FEATURES)
MODEL_FEATURES = {"M0": M0_FEATURES, "M1": M1_FEATURES, "M2": M2_FEATURES}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected or contract.get("safety", {}).get(key) != expected:
            raise BlockedScreen(
                "blocked_chronology_or_leakage_failure", f"contract safety flag differs: {key}"
            )
    return cast(dict[str, Any], contract)


def load_frozen_model() -> tuple[EmissionPreprocessing, SemiMarkovParameters]:
    preprocessing_frame = pd.read_csv(PREPROCESSING_PATH)
    preprocessing = EmissionPreprocessing(
        feature_names=tuple(preprocessing_frame["feature"].astype(str)),
        medians=preprocessing_frame["imputer_median"].to_numpy(dtype=float),
        centers=preprocessing_frame["scaler_center"].to_numpy(dtype=float),
        scales=preprocessing_frame["scaler_scale"].to_numpy(dtype=float),
    )
    preprocessing.validate()
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
    if preprocessing.feature_names != tuple(EMISSION_FEATURES) or model_hash != EXPECTED_MODEL_HASH:
        raise BlockedScreen(
            "blocked_v2_decision_population_not_reconstructable",
            "frozen V2 preprocessing or state-model identity differs",
        )
    return preprocessing, parameters


def load_behavioural_ledger() -> pd.DataFrame:
    compact_columns = [
        "symbol",
        "session",
        "decision_ordinal",
        "feature_available_timestamp_utc",
        "return_gap",
    ]
    compact = pd.read_parquet(BEHAVIOURAL_COMPACT, columns=compact_columns)
    dimension_columns = [
        "symbol",
        "session",
        "decision_ordinal",
        *BEHAVIOURAL_Z_COMPONENTS,
        *BEHAVIOURAL_DIMENSIONS,
    ]
    dimensions = pd.read_parquet(BEHAVIOURAL_DIMENSION_LEDGER, columns=dimension_columns)
    ledger = compact.merge(
        dimensions,
        on=["symbol", "session", "decision_ordinal"],
        how="inner",
        validate="one_to_one",
    )
    if len(ledger) != len(compact) or len(ledger) != len(dimensions):
        raise BlockedScreen(
            "blocked_behavioural_ledger_not_reconstructable",
            "frozen behavioural compact/dimension ledgers do not align",
        )
    return ledger


def load_opening_decisions() -> pd.DataFrame:
    columns = [
        "symbol",
        "session",
        "year",
        "year_month",
        "decision_ordinal",
        "repo_bar_start_ordinal",
        "slate_id",
        "feature_available_timestamp_utc",
        "current_state",
        "state_model_hash",
        "posterior_entropy",
        "maximum_posterior_probability",
        *(f"posterior_state_{state}" for state in range(STATE_COUNT)),
    ]
    decisions = pd.read_parquet(OPENING_PANEL, columns=columns)
    if len(decisions) > MAX_DECISION_ROWS:
        raise BlockedScreen(
            "blocked_quick_funnel_resource_limit", "frozen V2 decisions exceed 20,000 rows"
        )
    reject_protected_dates(decisions)
    return decisions.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


def build_v2_state_panel(
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
    if panel["bar_start_timestamp"].ge(PROTECTED_START_UTC).any():
        raise BlockedScreen(
            "blocked_protected_boundary_failure", "protected V2 market row materialised"
        )
    development = panel.loc[panel["bar_start_timestamp"].lt(DEVELOPMENT_END_EXCLUSIVE)]
    panel_hash = canonical_frame_hash(development, columns=(*NATURAL_KEY, *EMISSION_FEATURES))
    if panel_hash != EXPECTED_PANEL_HASH:
        raise BlockedScreen(
            "blocked_v2_decision_population_not_reconstructable",
            "frozen 2024 V2 emission panel hash differs",
        )
    states = panel.loc[
        panel["symbol"].isin(DECISION_SYMBOLS) & panel["bar_ordinal"].le(MAX_TARGET_BAR_ORDINAL)
    ].copy()
    states = states.sort_values(list(NATURAL_KEY), kind="mergesort").reset_index(drop=True)
    summary = causal_filter_summary(
        gaussian_log_emissions(transform_emissions(states, preprocessing), parameters),
        groups=causal_segment_groups(states),
        model=parameters.as_dict(),
    )
    states["causal_hard_state"] = summary.hard_states.astype(np.int16)
    states["expected_state_age"] = summary.expected_age
    states["transition_probability"] = summary.departure_probability
    states["persistence_probability"] = 1.0 - summary.departure_probability
    states["posterior_entropy_reproduced"] = summary.posterior_entropy
    for state in range(STATE_COUNT):
        states[f"state_p_{state}"] = summary.state_probabilities[:, state]
    source = {
        **SAFETY_FLAGS,
        "provider": "EODHD",
        "timeframe": "5m",
        "raw_data_downloaded": False,
        "date_predicate_applied_before_materialisation": True,
        "minimum_timestamp_read": str(panel["bar_start_timestamp"].min()),
        "maximum_timestamp_read": str(panel["bar_start_timestamp"].max()),
        "protected_rows_materialised": 0,
        "source_hashes": built.source_hashes,
        "source_row_counts": built.source_row_counts,
        "development_emission_panel_hash": panel_hash,
        "state_rows_filtered": len(states),
        "maximum_target_bar_ordinal": MAX_TARGET_BAR_ORDINAL,
    }
    return states, source


def reproduce_v2_decisions(
    archived: pd.DataFrame,
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    decision_states = states.loc[
        :,
        [
            "symbol",
            "session",
            "bar_ordinal",
            "bar_complete_timestamp",
            "causal_hard_state",
            "expected_state_age",
            "transition_probability",
            "persistence_probability",
            "posterior_entropy_reproduced",
            *STATE_PROBABILITY_FEATURES,
        ],
    ].copy()
    keys = archived.loc[
        :,
        [
            "symbol",
            "session",
            "decision_ordinal",
            "repo_bar_start_ordinal",
            "feature_available_timestamp_utc",
        ],
    ].copy()
    reproduced = keys.merge(
        decision_states,
        left_on=["symbol", "session", "repo_bar_start_ordinal"],
        right_on=["symbol", "session", "bar_ordinal"],
        how="left",
        validate="one_to_one",
    )
    reproduced["bar_complete_timestamp"] = pd.to_datetime(
        reproduced["bar_complete_timestamp"], utc=True, errors="raise"
    )
    reproduced["feature_available_timestamp_utc"] = pd.to_datetime(
        reproduced["feature_available_timestamp_utc"], utc=True, errors="raise"
    )
    if (
        not reproduced["bar_complete_timestamp"]
        .eq(reproduced["feature_available_timestamp_utc"])
        .all()
    ):
        raise BlockedScreen(
            "blocked_v2_decision_population_not_reconstructable",
            "V2 completed-bar timestamp differs from frozen decision timestamp",
        )
    probabilities = reproduced.loc[:, list(STATE_PROBABILITY_FEATURES)].to_numpy(dtype=float)
    ordered = np.sort(probabilities, axis=1)
    reproduced["top_state_probability"] = ordered[:, -1]
    reproduced["top_second_margin"] = ordered[:, -1] - ordered[:, -2]
    reproduced["hard_top_state"] = reproduced["causal_hard_state"].astype(int)
    reproduced["remaining_session_bars"] = EXPECTED_SESSION_BARS - reproduced[
        "decision_ordinal"
    ].astype(int)
    reproduced["checkpoint"] = reproduced["decision_ordinal"].eq(12).astype(float)
    keep = [
        "symbol",
        "session",
        "decision_ordinal",
        "feature_available_timestamp_utc",
        *STATE_PROBABILITY_FEATURES,
        "posterior_entropy_reproduced",
        "top_state_probability",
        "top_second_margin",
        "hard_top_state",
        "expected_state_age",
        "transition_probability",
        "persistence_probability",
        "remaining_session_bars",
        "checkpoint",
    ]
    return join_v2_posteriors(archived, reproduced.loc[:, keep])


def load_loop_dictionary() -> tuple[LoopDictionary, dict[str, Any]]:
    table = pd.read_csv(DICTIONARY_PATH)
    definitions = {}
    for row in table.itertuples(index=False):
        definition = decompose_closed_path(json.loads(str(row.canonical_orientation)))
        if definition.semantic_loop_id != str(row.semantic_loop_id):
            raise BlockedScreen(
                "blocked_v2_decision_population_not_reconstructable",
                "semantic loop dictionary cannot be reconstructed",
            )
        definitions[definition.semantic_loop_id] = definition
    dictionary = LoopDictionary(
        definitions,
        (),
        version=str(table["dictionary_version"].iloc[0]),
    )
    expected_hashes = set(table["dictionary_hash"].astype(str))
    if expected_hashes != {dictionary.dictionary_hash}:
        raise BlockedScreen(
            "blocked_v2_decision_population_not_reconstructable",
            "semantic loop dictionary hash differs",
        )
    manifest = {
        "dictionary_version": dictionary.version,
        "dictionary_hash": dictionary.dictionary_hash,
        "registered_definition_count": len(definitions),
        "semantic_loop_ids": sorted(definitions),
        "source_sha256": sha256_file(DICTIONARY_PATH),
    }
    return dictionary, manifest


def build_targets(
    decisions: pd.DataFrame,
    states: pd.DataFrame,
    dictionary: LoopDictionary,
) -> pd.DataFrame:
    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(STATE_COUNT)))
    required_keys = set(
        zip(decisions["symbol"].astype(str), decisions["session"].astype(str), strict=True)
    )
    rows: list[dict[str, Any]] = []
    for (symbol_value, session_value), group in states.groupby(["symbol", "session"], sort=True):
        symbol = str(symbol_value)
        session = str(session_value)
        if (symbol, session) not in required_keys:
            continue
        ordered = group.sort_values("bar_ordinal", kind="mergesort")
        checkpoints = decisions.loc[
            decisions["symbol"].eq(symbol) & decisions["session"].eq(session)
        ]
        if checkpoints.empty:
            continue
        for checkpoint in checkpoints.itertuples(index=False):
            origin = int(checkpoint.repo_bar_start_ordinal)
            horizon_end = origin + 6
            target_bars = ordered.loc[ordered["bar_ordinal"].le(horizon_end)].copy()
            ordinals = target_bars["bar_ordinal"].astype(int).tolist()
            if ordinals != list(range(horizon_end + 1)):
                rows.append(
                    {
                        "symbol": symbol,
                        "session": session,
                        "decision_ordinal": int(checkpoint.decision_ordinal),
                        "raw_outcome": "UNAVAILABLE",
                        "semantic_loop_id": None,
                        "orientation": None,
                        "oriented_loop_key": None,
                        "motif_type": None,
                        "bars_until_completion": None,
                        "state_events_until_completion": None,
                        "target_excluded": True,
                        "tied_semantic_loop_ids": [],
                        "source_available": False,
                        "state_path_through_horizon": [],
                        "bar_ordinals_through_horizon": [],
                    }
                )
                continue
            hard = target_bars["causal_hard_state"].to_numpy(dtype=int)
            event_mask = np.concatenate(([True], hard[1:] != hard[:-1]))
            event_rows = target_bars.loc[event_mask]
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
            event_ordinals = np.asarray(
                [event.bar_ordinal for event in trace.state_events], dtype=int
            )
            candidates = np.flatnonzero(event_ordinals <= origin)
            decision_event_index = int(candidates[-1]) if len(candidates) else -1
            outcome = resolve_first_loop_target(
                engine,
                trace,
                decision_id=f"{symbol}|{session}|{int(checkpoint.decision_ordinal):02d}",
                decision_event_index=decision_event_index,
                decision_bar_ordinal=origin,
                decision_timestamp=pd.Timestamp(
                    checkpoint.feature_available_timestamp_utc
                ).to_pydatetime(),
                session_end_bar_ordinal=EXPECTED_SESSION_BARS - 1,
                symbol=symbol,
                session=session,
            )
            rows.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "decision_ordinal": int(checkpoint.decision_ordinal),
                    **outcome,
                    "state_path_through_horizon": target_bars["causal_hard_state"]
                    .astype(int)
                    .tolist(),
                    "bar_ordinals_through_horizon": target_bars["bar_ordinal"].astype(int).tolist(),
                }
            )
    targets = pd.DataFrame(rows)
    if len(targets) != len(decisions):
        raise BlockedScreen(
            "blocked_v2_decision_population_not_reconstructable",
            "six-bar target population differs from frozen decisions",
        )
    return targets.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


@dataclass(slots=True)
class FittedMultinomial:
    name: str
    features: tuple[str, ...]
    scaler: StandardScaler
    estimator: LogisticRegression
    class_order: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, list(self.features)].to_numpy(dtype=float)
        scaled = self.scaler.transform(matrix)
        probabilities = self.estimator.predict_proba(scaled)
        expected = np.arange(len(self.class_order), dtype=int)
        if not np.array_equal(self.estimator.classes_, expected):
            raise BlockedScreen(
                "blocked_reproducibility_or_audit_failure", "multinomial class ordering differs"
            )
        return np.asarray(probabilities, dtype=float)

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


def fit_multinomial(
    name: str,
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    class_order: tuple[str, ...],
) -> FittedMultinomial:
    class_index = {label: index for index, label in enumerate(class_order)}
    target = frame["target_class"].map(class_index)
    if target.isna().any():
        raise BlockedScreen(
            "blocked_insufficient_loop_class_support", "development target lies outside class order"
        )
    matrix = frame.loc[:, list(features)].to_numpy(dtype=float)
    weights = frame["row_weight"].to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(weights).all():
        raise BlockedScreen("blocked_chronology_or_leakage_failure", "model input is not finite")
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaled = scaler.fit_transform(matrix)
    estimator = LogisticRegression(
        penalty="l2",
        C=0.25,
        solver="lbfgs",
        max_iter=300,
        class_weight=None,
        random_state=MODEL_SEED,
        n_jobs=1,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            estimator.fit(scaled, target.to_numpy(dtype=int), sample_weight=weights)
    except ConvergenceWarning as error:
        raise BlockedScreen(
            "blocked_model_convergence_failure", f"{name} emitted a convergence warning"
        ) from error
    expected_classes = np.arange(len(class_order), dtype=int)
    if not np.array_equal(estimator.classes_, expected_classes) or np.any(estimator.n_iter_ >= 300):
        raise BlockedScreen(
            "blocked_model_convergence_failure", f"{name} did not converge over every class"
        )
    return FittedMultinomial(name, features, scaler, estimator, class_order)


def assemble_decision_panel(
    v2_decisions: pd.DataFrame,
    targets: pd.DataFrame,
    behavioural_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, tuple[float, float]]]:
    target_columns = [
        "symbol",
        "session",
        "decision_ordinal",
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
        "state_path_through_horizon",
        "bar_ordinals_through_horizon",
    ]
    panel = v2_decisions.merge(
        targets.loc[:, target_columns],
        on=["symbol", "session", "decision_ordinal"],
        how="left",
        validate="one_to_one",
    )
    behavioural_keys = [
        "symbol",
        "session",
        "decision_ordinal",
        "feature_available_timestamp_utc",
    ]
    panel["feature_available_timestamp_utc"] = pd.to_datetime(
        panel["feature_available_timestamp_utc"], utc=True, errors="raise"
    )
    eligible_keys = behavioural_ledger.loc[:, behavioural_keys].copy()
    eligible_keys["feature_available_timestamp_utc"] = pd.to_datetime(
        eligible_keys["feature_available_timestamp_utc"], utc=True, errors="raise"
    )
    eligible_panel = panel.merge(
        eligible_keys,
        on=behavioural_keys,
        how="inner",
        validate="one_to_one",
    )
    if len(eligible_panel) != len(behavioural_ledger):
        raise BlockedScreen(
            "blocked_behavioural_ledger_not_reconstructable",
            "behaviourally eligible keys do not form the frozen predecessor population",
        )
    panel, behavioural_audit = join_behavioural_ledger(eligible_panel, behavioural_ledger)
    behavioural_audit["v2_rows_before_behavioural_eligibility"] = len(v2_decisions)
    behavioural_audit["v2_rows_excluded_by_frozen_behavioural_predecessor"] = len(
        v2_decisions
    ) - len(panel)
    validate_checkpoint_timing(panel)
    reject_protected_dates(panel)
    panel["year"] = panel["session"].astype(str).str[:4].astype(int)
    panel["year_month"] = panel["session"].astype(str).str[:7]
    panel["slate_id"] = (
        panel["session"].astype(str)
        + "|"
        + panel["decision_ordinal"].astype(int).map(lambda value: f"{value:02d}")
    )
    mapping, mapping_manifest = select_exact_loop_classes(panel, require_minimum=False)
    selection_passed = bool(mapping_manifest["selection_support_passed"])
    if selection_passed:
        panel["target_class"] = [
            pool_target_class(str(raw), key if isinstance(key, str) else None, mapping)
            for raw, key in zip(panel["raw_outcome"], panel["oriented_loop_key"], strict=True)
        ]
        panel["scoring_eligible"] = panel["target_class"].notna()
        panel.loc[panel["scoring_eligible"], "row_weight"] = 1.0 / panel.loc[
            panel["scoring_eligible"]
        ].groupby("slate_id", sort=True)["symbol"].transform("size")
        development_eligible = panel["year"].eq(2024) & panel["scoring_eligible"]
    else:
        panel["target_class"] = None
        panel["scoring_eligible"] = False
        panel["row_weight"] = np.nan
        development_eligible = panel["year"].eq(2024) & ~panel["target_excluded"]
    _, interaction_bounds = build_interactions(panel.loc[development_eligible], fit_bounds=True)
    interactions, _ = build_interactions(panel, bounds=interaction_bounds)
    for feature in INTERACTION_FEATURES:
        panel[feature] = interactions[feature]
    mapping_manifest["final_target_classes"] = (
        [
            *mapping.values(),
            "OTHER_REGISTERED_LOOP",
            "UNREGISTERED_LOOP",
            "NO_REGISTERED_COMPLETION",
        ]
        if selection_passed
        else []
    )
    return panel, behavioural_audit, mapping_manifest, interaction_bounds


def target_class_order(mapping_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    selected = cast(Mapping[str, str], mapping_manifest["selected_mapping"])
    ordered_selected = tuple(
        value for _, value in sorted(selected.items(), key=lambda item: int(item[1].split("_")[1]))
    )
    return (
        *ordered_selected,
        "OTHER_REGISTERED_LOOP",
        "UNREGISTERED_LOOP",
        "NO_REGISTERED_COMPLETION",
    )


def support_audit(
    panel: pd.DataFrame,
    *,
    class_order: tuple[str, ...],
) -> tuple[dict[str, Any], pd.DataFrame]:
    scoring = panel.loc[panel["scoring_eligible"]].copy()
    development = scoring.loc[scoring["year"].eq(2024)].copy()
    assessment = scoring.loc[scoring["year"].eq(2025)].copy()
    assessment_class_support = (
        assessment["target_class"].value_counts().reindex(class_order, fill_value=0)
    )
    development_class_support = (
        development["target_class"].value_counts().reindex(class_order, fill_value=0)
    )
    stock_share = assessment["symbol"].value_counts(normalize=True)
    class_share = assessment["target_class"].value_counts(normalize=True)
    selected = [label for label in class_order if label.startswith("LOOP_")]
    gates = {
        "assessment_rows_at_least_3000": len(assessment) >= 3000,
        "assessment_sessions_at_least_100": assessment["session"].nunique() >= 100,
        "assessment_stocks_at_least_15": assessment["symbol"].nunique() >= 15,
        "assessment_months_at_least_6": assessment["year_month"].nunique() >= 6,
        "total_target_classes_at_least_6": len(class_order) >= 6,
        "selected_exact_loop_classes_at_least_4": len(selected) >= 4,
        "selected_assessment_support_at_least_50": bool(
            assessment_class_support.loc[selected].ge(50).all()
        ),
        "maximum_stock_share_at_most_10_percent": float(stock_share.max()) <= 0.10,
        "maximum_class_share_at_most_60_percent": float(class_share.max()) <= 0.60,
        "all_classes_have_development_support": bool(development_class_support.gt(0).all()),
        "all_classes_have_assessment_support": bool(assessment_class_support.gt(0).all()),
    }
    support = {
        **SAFETY_FLAGS,
        "total_rows_including_exclusions": len(panel),
        "development_rows": len(development),
        "development_sessions": int(development["session"].nunique()),
        "development_stocks": int(development["symbol"].nunique()),
        "development_class_support": {
            key: int(value) for key, value in development_class_support.items()
        },
        "assessment_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_months": int(assessment["year_month"].nunique()),
        "assessment_class_support": {
            key: int(value) for key, value in assessment_class_support.items()
        },
        "maximum_assessment_stock_share": float(stock_share.max()),
        "maximum_assessment_class_share": float(class_share.max()),
        "tied_exclusions": int(panel["raw_outcome"].eq("TIED_REGISTERED_COMPLETION").sum()),
        "unavailable_exclusions": int(panel["raw_outcome"].eq("UNAVAILABLE").sum()),
        "gates": gates,
        "passed": all(gates.values()),
    }
    if not support["passed"]:
        raise BlockedScreen(
            "blocked_insufficient_loop_class_support", "one or more frozen support gates failed"
        )
    return support, scoring


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(
        np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float))
    )


def probability_diagnostics(
    target_indices: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.empty(len(target_indices), dtype=int)
    for index, target in enumerate(target_indices):
        ranks[index] = int(np.flatnonzero(order[index] == target)[0]) + 1
    realised = probabilities[np.arange(len(target_indices)), target_indices]
    entropy = prediction_entropy(probabilities)
    return ranks, realised, entropy


def expected_calibration_error(
    target_indices: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == target_indices
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(weights.sum())
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not mask.any():
            continue
        bin_weight = float(weights[mask].sum())
        accuracy = _weighted_mean(correct[mask].astype(float), weights[mask])
        mean_confidence = _weighted_mean(confidence[mask], weights[mask])
        value += bin_weight / total * abs(accuracy - mean_confidence)
    return float(value)


def metric_row(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    model: str,
    class_order: tuple[str, ...],
) -> dict[str, Any]:
    class_index = {label: index for index, label in enumerate(class_order)}
    targets = frame["target_class"].map(class_index).to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    ranks, realised, _ = probability_diagnostics(targets, probabilities)
    auc = math.nan
    if set(targets) == set(range(len(class_order))):
        auc = float(
            roc_auc_score(
                targets,
                probabilities,
                labels=np.arange(len(class_order)),
                multi_class="ovr",
                average="macro",
                sample_weight=weights,
            )
        )
    predicted_distribution = {
        label: _weighted_mean(probabilities[:, index], weights)
        for index, label in enumerate(class_order)
    }
    support = frame["target_class"].value_counts().reindex(class_order, fill_value=0)
    return {
        "model": model,
        "multiclass_log_loss": float(
            log_loss(
                targets,
                probabilities,
                labels=np.arange(len(class_order)),
                sample_weight=weights,
            )
        ),
        "multiclass_brier": multiclass_brier(targets, probabilities, weights),
        "top_one_accuracy": _weighted_mean((ranks <= 1).astype(float), weights),
        "top_three_accuracy": _weighted_mean((ranks <= 3).astype(float), weights),
        "mean_reciprocal_rank": _weighted_mean(1.0 / ranks, weights),
        "macro_ovr_auc": auc,
        "expected_calibration_error": expected_calibration_error(targets, probabilities, weights),
        "rows": len(frame),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "class_support": json.dumps(
            {key: int(value) for key, value in support.items()}, sort_keys=True
        ),
        "predicted_class_distribution": json.dumps(predicted_distribution, sort_keys=True),
    }


def score_models(
    frame: pd.DataFrame,
    models: Mapping[str, FittedMultinomial],
    *,
    class_order: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    scored = frame.copy()
    class_index = {label: index for index, label in enumerate(class_order)}
    targets = scored["target_class"].map(class_index).to_numpy(dtype=int)
    probabilities_by_model: dict[str, np.ndarray] = {}
    pooled_rows: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    for name, model in models.items():
        probabilities = model.predict(scored)
        probabilities_by_model[name] = probabilities
        ranks, realised, entropy = probability_diagnostics(targets, probabilities)
        for index, label in enumerate(class_order):
            scored[f"probability__{name}__{label}"] = probabilities[:, index]
        scored[f"realised_probability__{name}"] = realised
        scored[f"realised_rank__{name}"] = ranks
        scored[f"prediction_entropy__{name}"] = entropy
        scored[f"effective_candidate_count__{name}"] = np.exp(entropy)
        pooled_rows.append(metric_row(scored, probabilities, model=name, class_order=class_order))
        weights = scored["row_weight"].to_numpy(dtype=float)
        funnel_rows.append(
            {
                "model": name,
                "mean_prediction_entropy": _weighted_mean(entropy, weights),
                "median_prediction_entropy": float(np.median(entropy)),
                "mean_effective_candidate_count": _weighted_mean(np.exp(entropy), weights),
                "mean_probability_realised_class": _weighted_mean(realised, weights),
                "realised_class_top_one_percent": 100.0
                * _weighted_mean((ranks <= 1).astype(float), weights),
                "realised_class_top_two_percent": 100.0
                * _weighted_mean((ranks <= 2).astype(float), weights),
                "realised_class_top_three_percent": 100.0
                * _weighted_mean((ranks <= 3).astype(float), weights),
                "rows": len(scored),
            }
        )
    return scored, probabilities_by_model, pd.DataFrame(pooled_rows), pd.DataFrame(funnel_rows)


def sliced_metrics(
    frame: pd.DataFrame,
    models: Mapping[str, FittedMultinomial],
    *,
    class_order: tuple[str, ...],
    group_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, subset in frame.groupby(group_column, sort=True):
        for name, model in models.items():
            row = metric_row(
                subset,
                model.predict(subset),
                model=name,
                class_order=class_order,
            )
            row[group_column] = group
            rows.append(row)
    return pd.DataFrame(rows)


def class_metrics(
    frame: pd.DataFrame,
    probabilities_by_model: Mapping[str, np.ndarray],
    *,
    class_order: tuple[str, ...],
) -> pd.DataFrame:
    class_index = {label: index for index, label in enumerate(class_order)}
    target = frame["target_class"].map(class_index).to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for model, probabilities in probabilities_by_model.items():
        ranks, _, _ = probability_diagnostics(target, probabilities)
        for index, label in enumerate(class_order):
            mask = target == index
            support = int(mask.sum())
            binary = mask.astype(int)
            auc = (
                float(roc_auc_score(binary, probabilities[:, index], sample_weight=weights))
                if binary.min() != binary.max()
                else math.nan
            )
            rows.append(
                {
                    "model": model,
                    "target_class": label,
                    "support": support,
                    "mean_realised_probability": _weighted_mean(
                        probabilities[mask, index], weights[mask]
                    ),
                    "top_one_accuracy": _weighted_mean(
                        (ranks[mask] <= 1).astype(float), weights[mask]
                    ),
                    "top_three_accuracy": _weighted_mean(
                        (ranks[mask] <= 3).astype(float), weights[mask]
                    ),
                    "ovr_auc": auc,
                    "ovr_brier": _weighted_mean(
                        np.square(probabilities[:, index] - binary), weights
                    ),
                }
            )
    return pd.DataFrame(rows)


def _comparison_values(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    class_order: tuple[str, ...],
) -> dict[str, float]:
    baseline_metrics = metric_row(frame, baseline, model="baseline", class_order=class_order)
    candidate_metrics = metric_row(frame, candidate, model="candidate", class_order=class_order)
    class_index = {label: index for index, label in enumerate(class_order)}
    targets = frame["target_class"].map(class_index).to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    baseline_rank, baseline_realised, baseline_entropy = probability_diagnostics(targets, baseline)
    candidate_rank, candidate_realised, candidate_entropy = probability_diagnostics(
        targets, candidate
    )
    return {
        "log_loss_improvement": float(baseline_metrics["multiclass_log_loss"])
        - float(candidate_metrics["multiclass_log_loss"]),
        "brier_improvement": float(baseline_metrics["multiclass_brier"])
        - float(candidate_metrics["multiclass_brier"]),
        "top_three_improvement": _weighted_mean((candidate_rank <= 3).astype(float), weights)
        - _weighted_mean((baseline_rank <= 3).astype(float), weights),
        "realised_probability_improvement": _weighted_mean(
            candidate_realised - baseline_realised, weights
        ),
        "prediction_entropy_reduction": _weighted_mean(
            baseline_entropy - candidate_entropy, weights
        ),
    }


def paired_bootstrap_metrics(
    assessment: pd.DataFrame,
    probabilities: Mapping[str, np.ndarray],
    *,
    class_order: tuple[str, ...],
) -> pd.DataFrame:
    draws = session_block_bootstrap_draws(assessment, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    specifications = {
        "m1_minus_m0": (
            "M0",
            "M1",
            ("log_loss_improvement", "brier_improvement", "top_three_improvement"),
        ),
        "m2_minus_m1": (
            "M1",
            "M2",
            (
                "log_loss_improvement",
                "brier_improvement",
                "top_three_improvement",
                "realised_probability_improvement",
                "prediction_entropy_reduction",
            ),
        ),
    }
    real: dict[str, float] = {}
    values: dict[str, list[float]] = {}
    for prefix, (baseline_name, candidate_name, metrics) in specifications.items():
        actual = _comparison_values(
            assessment,
            probabilities[baseline_name],
            probabilities[candidate_name],
            class_order=class_order,
        )
        for metric in metrics:
            key = f"{prefix}_{metric}"
            real[key] = actual[metric]
            values[key] = []
    for indices in draws:
        sampled = assessment.iloc[indices].reset_index(drop=True)
        for prefix, (baseline_name, candidate_name, metrics) in specifications.items():
            draw_values = _comparison_values(
                sampled,
                probabilities[baseline_name][indices],
                probabilities[candidate_name][indices],
                class_order=class_order,
            )
            for metric in metrics:
                values[f"{prefix}_{metric}"].append(draw_values[metric])
    rows: list[dict[str, Any]] = []
    for metric, draws_for_metric in values.items():
        array = np.asarray(draws_for_metric, dtype=float)
        rows.append(
            {
                "metric": metric,
                "real_value": real[metric],
                "draw_mean": float(array.mean()),
                "interval_90_lower": float(np.quantile(array, 0.05)),
                "interval_90_upper": float(np.quantile(array, 0.95)),
                "interval_95_lower": float(np.quantile(array, 0.025)),
                "interval_95_upper": float(np.quantile(array, 0.975)),
                "draw_count": len(array),
                "seed": BOOTSTRAP_SEED,
                "draw_values": json.dumps(array.tolist()),
            }
        )
    return pd.DataFrame(rows)


def within_slate_null_metrics(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    real_models: Mapping[str, FittedMultinomial],
    real_probabilities: Mapping[str, np.ndarray],
    *,
    class_order: tuple[str, ...],
) -> pd.DataFrame:
    real_m1 = _comparison_values(
        assessment,
        real_probabilities["M0"],
        real_probabilities["M1"],
        class_order=class_order,
    )
    real_m2 = _comparison_values(
        assessment,
        real_probabilities["M1"],
        real_probabilities["M2"],
        class_order=class_order,
    )
    keys = {
        "m1_minus_m0_log_loss_improvement": real_m1["log_loss_improvement"],
        "m1_minus_m0_brier_improvement": real_m1["brier_improvement"],
        "m2_minus_m1_log_loss_improvement": real_m2["log_loss_improvement"],
        "m2_minus_m1_brier_improvement": real_m2["brier_improvement"],
        "m2_minus_m1_top_three_improvement": real_m2["top_three_improvement"],
    }
    null_values = {key: [] for key in keys}
    for draw in range(NULL_DRAWS):
        dev = permute_behavioural_bundle_within_slates(development, seed=NULL_SEED + draw * 2)
        ass = permute_behavioural_bundle_within_slates(assessment, seed=NULL_SEED + draw * 2 + 1)
        _, bounds = build_interactions(dev, fit_bounds=True)
        dev_interactions, _ = build_interactions(dev, bounds=bounds)
        ass_interactions, _ = build_interactions(ass, bounds=bounds)
        for feature in INTERACTION_FEATURES:
            dev[feature] = dev_interactions[feature]
            ass[feature] = ass_interactions[feature]
        m1 = fit_multinomial(
            f"null_{draw}_M1",
            dev,
            features=M1_FEATURES,
            class_order=class_order,
        )
        m2 = fit_multinomial(
            f"null_{draw}_M2",
            dev,
            features=M2_FEATURES,
            class_order=class_order,
        )
        m1_probabilities = m1.predict(ass)
        m2_probabilities = m2.predict(ass)
        comparison_m1 = _comparison_values(
            ass,
            real_models["M0"].predict(ass),
            m1_probabilities,
            class_order=class_order,
        )
        comparison_m2 = _comparison_values(
            ass, m1_probabilities, m2_probabilities, class_order=class_order
        )
        null_values["m1_minus_m0_log_loss_improvement"].append(
            comparison_m1["log_loss_improvement"]
        )
        null_values["m1_minus_m0_brier_improvement"].append(comparison_m1["brier_improvement"])
        null_values["m2_minus_m1_log_loss_improvement"].append(
            comparison_m2["log_loss_improvement"]
        )
        null_values["m2_minus_m1_brier_improvement"].append(comparison_m2["brier_improvement"])
        null_values["m2_minus_m1_top_three_improvement"].append(
            comparison_m2["top_three_improvement"]
        )
    rows: list[dict[str, Any]] = []
    for metric, values in null_values.items():
        array = np.asarray(values, dtype=float)
        real_value = keys[metric]
        rows.append(
            {
                "metric": metric,
                "real_value": real_value,
                "null_mean": float(array.mean()),
                "null_q90": float(np.quantile(array, 0.90)),
                "real_percentile": float(np.mean(array <= real_value)),
                "draw_count": len(array),
                "seed_base": NULL_SEED,
                "draw_values": json.dumps(array.tolist()),
            }
        )
    return pd.DataFrame(rows)


def stability_and_concentration_metrics(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    models: Mapping[str, FittedMultinomial],
    *,
    class_order: tuple[str, ...],
    support: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    entropy_median = float(development["posterior_entropy"].median())
    transition_median = float(development["transition_probability"].median())
    grouped = assessment.copy()
    grouped["entropy_group"] = np.where(
        grouped["posterior_entropy"] <= entropy_median, "low", "high"
    )
    grouped["transition_group"] = np.where(
        grouped["transition_probability"] <= transition_median, "low", "high"
    )
    rows: list[pd.DataFrame] = []
    for column, kind in (
        ("entropy_group", "development_frozen_posterior_entropy"),
        ("transition_group", "development_frozen_transition_probability"),
    ):
        result = sliced_metrics(
            grouped, models, class_order=class_order, group_column=column
        ).rename(columns={column: "group"})
        result["breakdown_type"] = kind
        rows.append(result)
    concentration = pd.concat(rows, ignore_index=True)
    support_rows = pd.DataFrame(
        [
            {
                "breakdown_type": "support_concentration",
                "group": "maximum_assessment_stock_share",
                "model": "ALL",
                "value": support["maximum_assessment_stock_share"],
            },
            {
                "breakdown_type": "support_concentration",
                "group": "maximum_assessment_class_share",
                "model": "ALL",
                "value": support["maximum_assessment_class_share"],
            },
        ]
    )
    concentration = pd.concat([concentration, support_rows], ignore_index=True, sort=False)
    return concentration, {
        "posterior_entropy_development_median": entropy_median,
        "transition_probability_development_median": transition_median,
    }


def _metric_lookup(frame: pd.DataFrame, model: str, metric: str) -> float:
    selected = frame.loc[frame["model"].eq(model), metric]
    if len(selected) != 1:
        raise BlockedScreen(
            "blocked_reproducibility_or_audit_failure",
            f"metric lookup is ambiguous: {model}/{metric}",
        )
    return float(selected.iloc[0])


def derive_decision(
    pooled: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    funnel: pd.DataFrame,
    *,
    support: Mapping[str, Any],
) -> dict[str, Any]:
    def bootstrap_lower(metric: str) -> float:
        selected = bootstrap.loc[bootstrap["metric"].eq(metric), "interval_90_lower"]
        return float(selected.iloc[0])

    def null_row(metric: str) -> pd.Series:
        selected = null.loc[null["metric"].eq(metric)]
        return selected.iloc[0]

    def comparison(prefix: str, baseline: str, candidate: str) -> dict[str, Any]:
        log_improvement = _metric_lookup(pooled, baseline, "multiclass_log_loss") - _metric_lookup(
            pooled, candidate, "multiclass_log_loss"
        )
        brier_improvement = _metric_lookup(pooled, baseline, "multiclass_brier") - _metric_lookup(
            pooled, candidate, "multiclass_brier"
        )
        top_three_improvement = _metric_lookup(
            pooled, candidate, "top_three_accuracy"
        ) - _metric_lookup(pooled, baseline, "top_three_accuracy")
        month_values: dict[str, float] = {}
        for month in sorted(monthly["year_month"].astype(str).unique()):
            subset = monthly.loc[monthly["year_month"].astype(str).eq(month)]
            month_values[month] = _metric_lookup(
                subset, baseline, "multiclass_log_loss"
            ) - _metric_lookup(subset, candidate, "multiclass_log_loss")
        checkpoint_values: dict[str, float] = {}
        for ordinal in sorted(checkpoint["decision_ordinal"].astype(int).unique()):
            subset = checkpoint.loc[checkpoint["decision_ordinal"].astype(int).eq(ordinal)]
            checkpoint_values[str(ordinal)] = _metric_lookup(
                subset, baseline, "multiclass_log_loss"
            ) - _metric_lookup(subset, candidate, "multiclass_log_loss")
        matching_null_metrics = [
            f"{prefix}_log_loss_improvement",
            f"{prefix}_brier_improvement",
        ]
        if prefix == "m2_minus_m1":
            matching_null_metrics.append(f"{prefix}_top_three_improvement")
        null_details = {
            metric: {
                "real_value": float(null_row(metric)["real_value"]),
                "null_q90": float(null_row(metric)["null_q90"]),
                "real_percentile": float(null_row(metric)["real_percentile"]),
            }
            for metric in matching_null_metrics
        }
        gates = {
            "log_loss_improves": log_improvement > 0.0,
            "brier_improves": brier_improvement > 0.0,
            "top_three_not_reduced": top_three_improvement >= 0.0,
            "bootstrap_90_lower_log_loss_non_negative": bootstrap_lower(
                f"{prefix}_log_loss_improvement"
            )
            >= 0.0,
            "bootstrap_90_lower_brier_non_negative": bootstrap_lower(f"{prefix}_brier_improvement")
            >= 0.0,
            "positive_log_loss_months_at_least_five": sum(
                value > 0.0 for value in month_values.values()
            )
            >= 5,
            "neither_checkpoint_materially_adverse": min(checkpoint_values.values())
            >= CHECKPOINT_MATERIAL_ADVERSITY,
            "real_increment_exceeds_at_least_one_matching_null_q90": any(
                details["real_value"] > details["null_q90"] for details in null_details.values()
            ),
            "concentration_gates_pass": bool(support["passed"]),
        }
        return {
            "baseline": baseline,
            "candidate": candidate,
            "log_loss_improvement": log_improvement,
            "brier_improvement": brier_improvement,
            "top_three_improvement": top_three_improvement,
            "monthly_log_loss_improvements": month_values,
            "positive_log_loss_months": sum(value > 0.0 for value in month_values.values()),
            "checkpoint_log_loss_improvements": checkpoint_values,
            "null": null_details,
            "gates": gates,
            "passes": all(gates.values()),
        }

    m1 = comparison("m1_minus_m0", "M0", "M1")
    m2 = comparison("m2_minus_m1", "M1", "M2")
    entropy_by_model = {
        str(row.model): float(row.mean_prediction_entropy) for row in funnel.itertuples(index=False)
    }
    descriptive_change = bool(
        entropy_by_model["M1"] < entropy_by_model["M0"]
        or entropy_by_model["M2"] < entropy_by_model["M1"]
        or m1["top_three_improvement"] > 0.0
        or m2["top_three_improvement"] > 0.0
    )
    category = decide_funnel(
        m1_pass=bool(m1["passes"]),
        m2_pass=bool(m2["passes"]),
        descriptive_change=descriptive_change,
    )
    return {
        **SAFETY_FLAGS,
        "decision": category,
        "binding_question": (
            "Does behavioural condition produce a smaller, better-calibrated set of likely future "
            "loops, and does the soft regime mixture materially improve that filtering?"
        ),
        "m1_versus_m0": m1,
        "m2_versus_m1": m2,
        "descriptive_funnel_change": descriptive_change,
        "support": dict(support),
        "independent_audit_passed": False,
        "determinism_check_passed": False,
    }


def plot_funnel_comparison(scored: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    models = ("M0", "M1", "M2")
    ranks = [scored[f"realised_rank__{model}"].to_numpy(dtype=float) for model in models]
    entropies = [scored[f"prediction_entropy__{model}"].to_numpy(dtype=float) for model in models]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].boxplot(ranks, tick_labels=models, showfliers=False)
    axes[0].set_title("Realised-class rank")
    axes[0].set_ylabel("Rank (lower is better)")
    axes[1].boxplot(entropies, tick_labels=models, showfliers=False)
    axes[1].set_title("Prediction entropy")
    axes[1].set_ylabel("Natural-log entropy")
    figure.suptitle("Six-bar future-loop funnel: fixed 2025 assessment")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def render_report(
    *,
    decision: Mapping[str, Any],
    support: Mapping[str, Any],
    mapping: Mapping[str, Any],
    pooled: pd.DataFrame,
    funnel: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
) -> str:
    lines = [
        "# Emotion × Regime-Mix Loop Funnel Quick Screen V0",
        "",
        "Retrospective, research-only, structural quick feasibility screen. Economic outcomes were "
        "not opened; execution and strategy promotion remained disabled.",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        "## Support",
        "",
        f"- Development: {support['development_rows']} rows, {support['development_sessions']} "
        f"sessions, {support['development_stocks']} stocks.",
        f"- Assessment: {support['assessment_rows']} rows, {support['assessment_sessions']} "
        f"sessions, {support['assessment_stocks']} stocks, {support['assessment_months']} months.",
        f"- Ties excluded: {support['tied_exclusions']}; unavailable excluded: "
        f"{support['unavailable_exclusions']}.",
        f"- Selected exact oriented loops: {len(mapping['selected_mapping'])}.",
        "",
        "## Pooled proper scores and ranking",
        "",
        "```text",
        pooled[
            [
                "model",
                "multiclass_log_loss",
                "multiclass_brier",
                "top_one_accuracy",
                "top_three_accuracy",
                "mean_reciprocal_rank",
                "expected_calibration_error",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Funnel diagnostics",
        "",
        "```text",
        funnel.to_string(index=False),
        "```",
        "",
        "## Binding comparisons",
        "",
        f"- M1 versus M0: log-loss improvement "
        f"{decision['m1_versus_m0']['log_loss_improvement']:.8f}; Brier improvement "
        f"{decision['m1_versus_m0']['brier_improvement']:.8f}; top-three change "
        f"{decision['m1_versus_m0']['top_three_improvement']:.8f}; passes="
        f"{decision['m1_versus_m0']['passes']}.",
        f"- M2 versus M1: log-loss improvement "
        f"{decision['m2_versus_m1']['log_loss_improvement']:.8f}; Brier improvement "
        f"{decision['m2_versus_m1']['brier_improvement']:.8f}; top-three change "
        f"{decision['m2_versus_m1']['top_three_improvement']:.8f}; passes="
        f"{decision['m2_versus_m1']['passes']}.",
        "",
        "## Resampling",
        "",
        f"The paired session bootstrap used {int(bootstrap['draw_count'].max())} fixed draws. The "
        f"within-slate behavioural null used {int(null['draw_count'].max())} fixed draws. Full 90% "
        "and 95% bootstrap intervals and real-result null percentiles are in the CSV artifacts.",
        "",
        "Lower prediction entropy is treated as descriptive unless proper scores also improve. "
        "This "
        "screen is not prospective validation and supplies no evidence about economic or trading "
        "utility.",
        "",
    ]
    return "\n".join(lines)


def serialize_interaction_bounds(
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    return {
        feature: {"q01": float(values[0]), "q99": float(values[1])}
        for feature, values in bounds.items()
    }


def blocked_support_summary(
    panel: pd.DataFrame,
    mapping_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    available = panel.loc[~panel["target_excluded"]].copy()
    development = available.loc[available["year"].eq(2024)]
    assessment = available.loc[available["year"].eq(2025)]
    assessment_stock_share = assessment["symbol"].value_counts(normalize=True)
    raw_class_share = assessment["raw_outcome"].value_counts(normalize=True)
    selected_count = int(mapping_manifest["selected_count"])
    gates = {
        "assessment_rows_at_least_3000": len(assessment) >= 3000,
        "assessment_sessions_at_least_100": assessment["session"].nunique() >= 100,
        "assessment_stocks_at_least_15": assessment["symbol"].nunique() >= 15,
        "assessment_months_at_least_6": assessment["year_month"].nunique() >= 6,
        "selected_exact_loop_classes_at_least_4": selected_count >= 4,
        "total_target_classes_at_least_6": selected_count + 3 >= 6,
        "maximum_stock_share_at_most_10_percent": float(assessment_stock_share.max()) <= 0.10,
        "selected_assessment_support_at_least_50": False,
        "maximum_final_class_share_at_most_60_percent": False,
    }
    return {
        **SAFETY_FLAGS,
        "support_stage": "development_loop_vocabulary_gate",
        "total_rows_including_exclusions": len(panel),
        "development_rows_with_available_untied_target": len(development),
        "development_sessions": int(development["session"].nunique()),
        "development_stocks": int(development["symbol"].nunique()),
        "assessment_rows_with_available_untied_target": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_months": int(assessment["year_month"].nunique()),
        "development_raw_outcome_support": {
            str(key): int(value)
            for key, value in development["raw_outcome"].value_counts().sort_index().items()
        },
        "assessment_raw_outcome_support": {
            str(key): int(value)
            for key, value in assessment["raw_outcome"].value_counts().sort_index().items()
        },
        "selected_exact_oriented_loop_count": selected_count,
        "minimum_required_selected_exact_oriented_loop_count": 4,
        "final_target_class_set_formed": False,
        "assessment_class_support": "not_calculated_because_development_vocabulary_gate_failed",
        "maximum_assessment_stock_share": float(assessment_stock_share.max()),
        "maximum_assessment_raw_outcome_share": float(raw_class_share.max()),
        "maximum_final_class_share": None,
        "tied_exclusions": int(panel["raw_outcome"].eq("TIED_REGISTERED_COMPLETION").sum()),
        "unavailable_exclusions": int(panel["raw_outcome"].eq("UNAVAILABLE").sum()),
        "gates": gates,
        "passed": False,
    }


def write_support_blocked_artifacts(
    output: Path,
    *,
    contract: Mapping[str, Any],
    source_context: Mapping[str, Any],
    behavioural_audit: Mapping[str, Any],
    v2_audit: Mapping[str, Any],
    dictionary_manifest: Mapping[str, Any],
    mapping_manifest: dict[str, Any],
    interaction_bounds: Mapping[str, tuple[float, float]],
    panel: pd.DataFrame,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    support = blocked_support_summary(panel, mapping_manifest)
    mapping_manifest["provisional_development_rank_mapping_not_frozen_for_scoring"] = True
    mapping_manifest["development_support_gate_blocker"] = "blocked_insufficient_loop_class_support"
    mapping_manifest["final_target_classes"] = []
    mapping_manifest["assessment_class_support"] = (
        "not_calculated_because_development_vocabulary_gate_failed"
    )
    decision = {
        **SAFETY_FLAGS,
        "decision": "blocked_insufficient_loop_class_support",
        "blocker_detail": (
            f"only {mapping_manifest['selected_count']} exact oriented loops passed the 2024 "
            "support rules; at least four were required"
        ),
        "support": support,
        "m1_versus_m0": "not_fit_due_development_loop_vocabulary_blocker",
        "m2_versus_m1": "not_fit_due_development_loop_vocabulary_blocker",
        "independent_audit_passed": False,
        "determinism_check_passed": False,
    }
    write_json(output / "contract.json", contract)
    write_json(
        output / "source_manifest.json",
        {
            **SAFETY_FLAGS,
            "dates_read": {
                "start": "2024-01-01",
                "end_inclusive": "2025-08-22",
                "protected_start": "2025-08-23",
            },
            "behavioural_predecessor": {
                "experiment": "Observable Behavioural-State Dimensions Screen V0",
                "commit": BEHAVIOURAL_PREDECESSOR_COMMIT,
                "compact_decision_panel_sha256": sha256_file(BEHAVIOURAL_COMPACT),
                "dimension_ledger_sha256": sha256_file(BEHAVIOURAL_DIMENSION_LEDGER),
                "scaling_sha256": sha256_file(BEHAVIOURAL_SCALING),
                "audit_sha256": sha256_file(BEHAVIOURAL_AUDIT),
            },
            "v2_structural_predecessor": {
                "experiment": (
                    "Opening Regime-Path Direction Screen V0 plus frozen V2 loop semantics"
                ),
                "commit": V2_STRUCTURAL_PREDECESSOR_COMMIT,
                "opening_decision_panel_sha256": sha256_file(OPENING_PANEL),
                "state_parameters_sha256": sha256_file(PARAMETERS_PATH),
                "state_preprocessing_sha256": sha256_file(PREPROCESSING_PATH),
                **dictionary_manifest,
            },
            "market_sources": dict(source_context),
        },
    )
    write_json(
        output / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "read_start": "2024-01-01",
            "read_end_inclusive": "2025-08-22",
            "protected_start": "2025-08-23",
            "maximum_timestamp_materialised": source_context["maximum_timestamp_read"],
            "protected_rows_materialised": source_context["protected_rows_materialised"],
            "passed": source_context["protected_rows_materialised"] == 0,
        },
    )
    write_json(output / "behavioural_ledger_reconstruction.json", behavioural_audit)
    write_json(output / "v2_population_reconstruction.json", v2_audit)
    write_json(output / "target_class_mapping.json", mapping_manifest)
    write_json(
        output / "feature_manifest.json",
        {
            **SAFETY_FLAGS,
            "M0": list(M0_FEATURES),
            "M1": list(M1_FEATURES),
            "M2": list(M2_FEATURES),
            "planned_only_not_fit": True,
            "preprocessing_fit_interval": "2024_only",
            "prefix_features_used": False,
            "future_state_features_used": False,
        },
    )
    write_json(
        output / "interaction_manifest.json",
        {
            **SAFETY_FLAGS,
            "interaction_count": len(INTERACTION_FEATURES),
            "pressure_gating": list(INTERACTION_FEATURES[:8]),
            "exhaustion_gating": list(INTERACTION_FEATURES[8:16]),
            "structural_uncertainty": list(INTERACTION_FEATURES[16:]),
            "development_only_clip_quantiles": [0.01, 0.99],
            "clip_bounds": serialize_interaction_bounds(interaction_bounds),
            "planned_only_not_fit": True,
        },
    )
    panel_columns = list(
        dict.fromkeys(
            [
                "symbol",
                "session",
                "year",
                "year_month",
                "decision_ordinal",
                "repo_bar_start_ordinal",
                "slate_id",
                "feature_available_timestamp_utc",
                "state_model_hash",
                "current_state",
                "hard_top_state",
                *(f"posterior_state_{state}" for state in range(STATE_COUNT)),
                *STATE_PROBABILITY_FEATURES,
                "posterior_entropy",
                "posterior_entropy_reproduced",
                "top_state_probability",
                "top_second_margin",
                "expected_state_age",
                "persistence_probability",
                "transition_probability",
                "remaining_session_bars",
                "checkpoint",
                *BEHAVIOURAL_Z_COMPONENTS,
                "return_gap",
                *BEHAVIOURAL_DIMENSIONS,
                *INTERACTION_FEATURES,
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
                "state_path_through_horizon",
                "bar_ordinals_through_horizon",
                "target_class",
                "scoring_eligible",
                "row_weight",
            ]
        )
    )
    write_parquet(output / "decision_panel.parquet", panel.loc[:, panel_columns])
    write_json(
        output / "model_configurations.json",
        {
            **SAFETY_FLAGS,
            "requested_configuration": contract["models"],
            "features": {name: list(features) for name, features in MODEL_FEATURES.items()},
            "primary_fitted_model_count": 0,
            "status": "not_fit_due_development_loop_vocabulary_blocker",
        },
    )
    write_json(output / "model_coefficients.json", {**SAFETY_FLAGS, "models": {}})
    empty_predictions = pd.DataFrame(
        columns=["symbol", "session", "decision_ordinal", "target_class"]
    )
    write_parquet(output / "assessment_predictions.parquet", empty_predictions)
    empty_metrics = pd.DataFrame(columns=["status", "blocker", "model", "metric", "value"])
    empty_metrics.loc[0] = [
        "not_calculated",
        "blocked_insufficient_loop_class_support",
        None,
        None,
        None,
    ]
    for name in (
        "pooled_metrics.csv",
        "monthly_metrics.csv",
        "checkpoint_metrics.csv",
        "class_metrics.csv",
        "funnel_metrics.csv",
        "bootstrap_metrics.csv",
        "null_metrics.csv",
    ):
        write_csv(output / name, empty_metrics)
    write_csv(
        output / "concentration_metrics.csv",
        pd.DataFrame(
            [
                {
                    "status": "development_vocabulary_gate_failed",
                    "maximum_assessment_stock_share": support["maximum_assessment_stock_share"],
                    "maximum_assessment_raw_outcome_share": support[
                        "maximum_assessment_raw_outcome_share"
                    ],
                    "maximum_final_class_share": None,
                }
            ]
        ),
    )
    write_json(output / "decision.json", decision)
    reloaded = pd.read_parquet(output / "decision_panel.parquet")
    repeated_mapping, repeated_manifest = select_exact_loop_classes(reloaded, require_minimum=False)
    support_gate_reproducibility_passed = bool(
        repeated_mapping == mapping_manifest["selected_mapping"]
        and repeated_manifest["selected_count"] == mapping_manifest["selected_count"]
        and repeated_manifest["selected_count"] < 4
    )
    determinism = {
        **SAFETY_FLAGS,
        "method": "reload_panel_and_repeat_development_loop_vocabulary_gate",
        "prescribed_model_determinism_applicable": False,
        "prescribed_model_determinism_status": (
            "not_applicable_due_development_loop_vocabulary_blocker"
        ),
        "maximum_probability_difference": None,
        "selected_mapping_equal": repeated_mapping == mapping_manifest["selected_mapping"],
        "selected_count_equal": repeated_manifest["selected_count"]
        == mapping_manifest["selected_count"],
        "final_decision_equal": repeated_manifest["selected_count"] < 4,
        "support_gate_reproducibility_passed": support_gate_reproducibility_passed,
        "passed": support_gate_reproducibility_passed,
    }
    write_json(output / "determinism_check.json", determinism)
    decision["determinism_check_passed"] = None
    decision["support_gate_reproducibility_passed"] = support_gate_reproducibility_passed
    write_json(output / "decision.json", decision)
    eligible_text = "\n".join(
        f"- `{key}`: {details['support']} outcomes, {details['sessions']} sessions, "
        f"{details['stocks']} stocks, {details['months']} months, max-stock share "
        f"{details['maximum_stock_share']:.3f}"
        for key, details in mapping_manifest["eligible_oriented_loops"].items()
    )
    report = (
        "# Emotion × Regime-Mix Loop Funnel Quick Screen V0\n\n"
        "Retrospective, research-only, structural quick feasibility screen. Economic outcomes "
        "were not opened and no models were fit after the binding support gate failed.\n\n"
        "Decision: `blocked_insufficient_loop_class_support`\n\n"
        f"Only {mapping_manifest['selected_count']} exact oriented loops met every 2024 support "
        "rule; at least four were required. The final target class set was therefore not formed, "
        "and no 2025 predictive metric, bootstrap, null, or model comparison was calculated.\n\n"
        "## Development-eligible exact oriented loops\n\n"
        f"{eligible_text}\n\n"
        f"Tied completions excluded: {support['tied_exclusions']}; source-unavailable exclusions: "
        f"{support['unavailable_exclusions']}. Protected rows materialised: 0.\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    return decision


def write_artifacts(
    output: Path,
    *,
    contract: Mapping[str, Any],
    source_context: Mapping[str, Any],
    behavioural_audit: Mapping[str, Any],
    v2_audit: Mapping[str, Any],
    dictionary_manifest: Mapping[str, Any],
    mapping_manifest: Mapping[str, Any],
    interaction_bounds: Mapping[str, tuple[float, float]],
    support: Mapping[str, Any],
    panel: pd.DataFrame,
    models: Mapping[str, FittedMultinomial],
    scored: pd.DataFrame,
    pooled: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    per_class: pd.DataFrame,
    funnel: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    concentration: pd.DataFrame,
    frozen_medians: Mapping[str, float],
    decision: Mapping[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    write_json(
        output / "source_manifest.json",
        {
            **SAFETY_FLAGS,
            "dates_read": {
                "start": "2024-01-01",
                "end_inclusive": "2025-08-22",
                "protected_start": "2025-08-23",
            },
            "behavioural_predecessor": {
                "experiment": "Observable Behavioural-State Dimensions Screen V0",
                "commit": BEHAVIOURAL_PREDECESSOR_COMMIT,
                "compact_decision_panel_sha256": sha256_file(BEHAVIOURAL_COMPACT),
                "dimension_ledger_sha256": sha256_file(BEHAVIOURAL_DIMENSION_LEDGER),
                "scaling_sha256": sha256_file(BEHAVIOURAL_SCALING),
                "audit_sha256": sha256_file(BEHAVIOURAL_AUDIT),
            },
            "v2_structural_predecessor": {
                "experiment": (
                    "Opening Regime-Path Direction Screen V0 plus frozen V2 loop semantics"
                ),
                "commit": V2_STRUCTURAL_PREDECESSOR_COMMIT,
                "opening_decision_panel_sha256": sha256_file(OPENING_PANEL),
                "source_manifest_sha256": sha256_file(OPENING_SOURCE_MANIFEST),
                "state_parameters_sha256": sha256_file(PARAMETERS_PATH),
                "state_preprocessing_sha256": sha256_file(PREPROCESSING_PATH),
                "semantic_dictionary_sha256": sha256_file(DICTIONARY_PATH),
                **dictionary_manifest,
            },
            "market_sources": dict(source_context),
        },
    )
    write_json(
        output / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "read_start": "2024-01-01",
            "read_end_inclusive": "2025-08-22",
            "protected_start": "2025-08-23",
            "maximum_timestamp_materialised": source_context["maximum_timestamp_read"],
            "protected_rows_materialised": source_context["protected_rows_materialised"],
            "passed": source_context["protected_rows_materialised"] == 0,
        },
    )
    write_json(output / "behavioural_ledger_reconstruction.json", behavioural_audit)
    write_json(output / "v2_population_reconstruction.json", v2_audit)
    write_json(output / "target_class_mapping.json", mapping_manifest)
    write_json(
        output / "feature_manifest.json",
        {
            **SAFETY_FLAGS,
            "M0": list(M0_FEATURES),
            "M1": list(M1_FEATURES),
            "M2": list(M2_FEATURES),
            "hard_top_state_reporting_only": True,
            "behavioural_dimensions_joined": list(BEHAVIOURAL_DIMENSIONS),
            "primary_behavioural_dimensions": list(PRIMARY_BEHAVIOURAL_FEATURES),
            "excluded_behavioural_dimensions": [
                feature
                for feature in BEHAVIOURAL_DIMENSIONS
                if feature not in PRIMARY_BEHAVIOURAL_FEATURES
            ],
            "prefix_features_used": False,
            "future_state_features_used": False,
            "preprocessing_fit_interval": "2024_only",
            "frozen_split_medians": dict(frozen_medians),
        },
    )
    write_json(
        output / "interaction_manifest.json",
        {
            **SAFETY_FLAGS,
            "interaction_count": len(INTERACTION_FEATURES),
            "pressure_gating": list(INTERACTION_FEATURES[:8]),
            "exhaustion_gating": list(INTERACTION_FEATURES[8:16]),
            "structural_uncertainty": list(INTERACTION_FEATURES[16:]),
            "development_only_clip_quantiles": [0.01, 0.99],
            "clip_bounds": serialize_interaction_bounds(interaction_bounds),
        },
    )
    panel_columns = list(
        dict.fromkeys(
            [
                "symbol",
                "session",
                "year",
                "year_month",
                "decision_ordinal",
                "repo_bar_start_ordinal",
                "slate_id",
                "feature_available_timestamp_utc",
                "state_model_hash",
                "current_state",
                "hard_top_state",
                *(f"posterior_state_{state}" for state in range(STATE_COUNT)),
                *STATE_PROBABILITY_FEATURES,
                "posterior_entropy",
                "posterior_entropy_reproduced",
                "top_state_probability",
                "top_second_margin",
                "expected_state_age",
                "persistence_probability",
                "transition_probability",
                "remaining_session_bars",
                "checkpoint",
                *BEHAVIOURAL_Z_COMPONENTS,
                "return_gap",
                *BEHAVIOURAL_DIMENSIONS,
                *INTERACTION_FEATURES,
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
                "state_path_through_horizon",
                "bar_ordinals_through_horizon",
                "target_class",
                "scoring_eligible",
                "row_weight",
            ]
        )
    )
    write_parquet(output / "decision_panel.parquet", panel.loc[:, panel_columns])
    write_json(
        output / "model_configurations.json",
        {
            **SAFETY_FLAGS,
            "requested_configuration": {
                "penalty": "l2",
                "C": 0.25,
                "solver": "lbfgs",
                "max_iter": 300,
                "multi_class": "multinomial",
                "class_weight": None,
                "random_state": MODEL_SEED,
                "n_jobs": 1,
            },
            "effective_multiclass_handling": (
                "scikit-learn_lbfgs_automatic_multinomial; multi_class keyword removed in "
                "sklearn 1.9"
            ),
            "features": {name: list(features) for name, features in MODEL_FEATURES.items()},
            "primary_fitted_model_count": 3,
            "null_refits": {"draws": NULL_DRAWS, "models_per_draw": 2},
            "row_weight": "1/eligible_stocks_in_session_checkpoint",
            "preprocessor": "StandardScaler_fit_on_2024_only",
        },
    )
    write_json(
        output / "model_coefficients.json",
        {
            **SAFETY_FLAGS,
            "models": {name: model.serialize() for name, model in models.items()},
        },
    )
    prediction_columns = [
        "symbol",
        "session",
        "year_month",
        "decision_ordinal",
        "slate_id",
        "target_class",
        "row_weight",
        "posterior_entropy",
        "transition_probability",
        *(column for column in scored.columns if column.startswith("probability__")),
        *(column for column in scored.columns if column.startswith("realised_probability__")),
        *(column for column in scored.columns if column.startswith("realised_rank__")),
        *(column for column in scored.columns if column.startswith("prediction_entropy__")),
        *(column for column in scored.columns if column.startswith("effective_candidate_count__")),
    ]
    write_parquet(
        output / "assessment_predictions.parquet",
        scored.loc[:, list(dict.fromkeys(prediction_columns))],
    )
    write_csv(output / "pooled_metrics.csv", pooled)
    write_csv(output / "monthly_metrics.csv", monthly)
    write_csv(output / "checkpoint_metrics.csv", checkpoint)
    write_csv(output / "class_metrics.csv", per_class)
    write_csv(output / "funnel_metrics.csv", funnel)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "null_metrics.csv", null)
    write_csv(output / "concentration_metrics.csv", concentration)
    write_json(output / "decision.json", decision)
    plot_funnel_comparison(scored, output / "realised_rank_prediction_entropy_comparison.png")
    report = render_report(
        decision=decision,
        support=support,
        mapping=mapping_manifest,
        pooled=pooled,
        funnel=funnel,
        bootstrap=bootstrap,
        null=null,
    )
    (output / "report.md").write_text(report, encoding="utf-8")


def determinism_check(
    output: Path,
    *,
    original_models: Mapping[str, FittedMultinomial],
    original_probabilities: Mapping[str, np.ndarray],
    class_order: tuple[str, ...],
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    original_decision: Mapping[str, Any],
    support: Mapping[str, Any],
) -> dict[str, Any]:
    reloaded = pd.read_parquet(output / "decision_panel.parquet")
    scoring = reloaded.loc[reloaded["scoring_eligible"]].copy()
    development = scoring.loc[scoring["year"].eq(2024)].copy()
    assessment = scoring.loc[scoring["year"].eq(2025)].copy()
    refit = {
        name: fit_multinomial(
            name,
            development,
            features=features,
            class_order=class_order,
        )
        for name, features in MODEL_FEATURES.items()
    }
    refit_scored, refit_probabilities, refit_pooled, refit_funnel = score_models(
        assessment, refit, class_order=class_order
    )
    probability_difference = max(
        float(np.max(np.abs(refit_probabilities[name] - original_probabilities[name])))
        for name in MODEL_FEATURES
    )
    coefficient_difference = 0.0
    scaler_difference = 0.0
    class_order_equal = True
    for name in MODEL_FEATURES:
        coefficient_difference = max(
            coefficient_difference,
            float(
                np.max(np.abs(refit[name].estimator.coef_ - original_models[name].estimator.coef_))
            ),
            float(
                np.max(
                    np.abs(
                        refit[name].estimator.intercept_
                        - original_models[name].estimator.intercept_
                    )
                )
            ),
        )
        scaler_difference = max(
            scaler_difference,
            float(np.max(np.abs(refit[name].scaler.mean_ - original_models[name].scaler.mean_))),
            float(np.max(np.abs(refit[name].scaler.scale_ - original_models[name].scaler.scale_))),
        )
        class_order_equal = class_order_equal and (
            refit[name].class_order == original_models[name].class_order
            and np.array_equal(
                refit[name].estimator.classes_, original_models[name].estimator.classes_
            )
        )
    original_pooled = pd.read_csv(output / "pooled_metrics.csv")
    numeric_metrics = [
        "multiclass_log_loss",
        "multiclass_brier",
        "top_one_accuracy",
        "top_three_accuracy",
        "mean_reciprocal_rank",
        "expected_calibration_error",
    ]
    pooled_difference = max(
        abs(
            _metric_lookup(original_pooled, name, metric)
            - _metric_lookup(refit_pooled, name, metric)
        )
        for name in MODEL_FEATURES
        for metric in numeric_metrics
    )
    refit_monthly = sliced_metrics(
        refit_scored, refit, class_order=class_order, group_column="year_month"
    )
    refit_checkpoint = sliced_metrics(
        refit_scored, refit, class_order=class_order, group_column="decision_ordinal"
    )
    refit_decision = derive_decision(
        refit_pooled,
        refit_monthly,
        refit_checkpoint,
        bootstrap,
        null,
        refit_funnel,
        support=support,
    )
    decision_equal = refit_decision["decision"] == original_decision["decision"]
    passed = bool(
        probability_difference <= 1e-12
        and coefficient_difference <= 1e-12
        and scaler_difference <= 1e-12
        and pooled_difference <= 1e-12
        and class_order_equal
        and decision_equal
    )
    result = {
        **SAFETY_FLAGS,
        "method": "reload_panel_refit_three_models_without_bootstrap_or_null_rerun",
        "maximum_permitted_probability_difference": 1e-12,
        "maximum_probability_difference": probability_difference,
        "maximum_coefficient_difference": coefficient_difference,
        "maximum_scaler_difference": scaler_difference,
        "maximum_pooled_metric_difference": pooled_difference,
        "class_order_equal": class_order_equal,
        "final_decision_equal": decision_equal,
        "passed": passed,
    }
    write_json(output / "determinism_check.json", result)
    if not passed:
        raise BlockedScreen(
            "blocked_reproducibility_or_audit_failure", "fast determinism check failed"
        )
    return result


def execute_screen(output: Path, *, provider_root: Path) -> dict[str, Any]:
    contract = load_contract()
    required = (
        BEHAVIOURAL_COMPACT,
        BEHAVIOURAL_DIMENSION_LEDGER,
        BEHAVIOURAL_SCALING,
        BEHAVIOURAL_AUDIT,
        OPENING_PANEL,
        OPENING_SOURCE_MANIFEST,
        PARAMETERS_PATH,
        PREPROCESSING_PATH,
        REFIT_CONFIG_PATH,
        DICTIONARY_PATH,
    )
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise BlockedScreen(
            "blocked_v2_decision_population_not_reconstructable",
            f"required frozen artifacts are absent: {missing}",
        )

    archived = load_opening_decisions()
    preprocessing, parameters = load_frozen_model()
    states, source_context = build_v2_state_panel(provider_root, preprocessing, parameters)
    v2_decisions, v2_audit = reproduce_v2_decisions(archived, states)
    dictionary, dictionary_manifest = load_loop_dictionary()
    targets = build_targets(v2_decisions, states, dictionary)
    behavioural_ledger = load_behavioural_ledger()
    panel, behavioural_audit, mapping_manifest, interaction_bounds = assemble_decision_panel(
        v2_decisions, targets, behavioural_ledger
    )
    if len(panel) > MAX_DECISION_ROWS:
        raise BlockedScreen(
            "blocked_quick_funnel_resource_limit", "joined decision panel exceeds 20,000 rows"
        )
    v2_audit.update(
        {
            "semantic_dictionary": dictionary_manifest,
            "target_horizon_completed_five_minute_bars": 6,
            "active_loop_prefix_required": False,
            "raw_target_support": {
                str(key): int(value)
                for key, value in panel["raw_outcome"].value_counts().sort_index().items()
            },
            "tied_registered_completion_exclusions": int(
                panel["raw_outcome"].eq("TIED_REGISTERED_COMPLETION").sum()
            ),
            "source_unavailable_exclusions": int(panel["raw_outcome"].eq("UNAVAILABLE").sum()),
        }
    )
    if not bool(mapping_manifest["selection_support_passed"]):
        blocked_decision = write_support_blocked_artifacts(
            output,
            contract=contract,
            source_context=source_context,
            behavioural_audit=behavioural_audit,
            v2_audit=v2_audit,
            dictionary_manifest=dictionary_manifest,
            mapping_manifest=mapping_manifest,
            interaction_bounds=interaction_bounds,
            panel=panel,
        )
        if str(EXPERIMENT_DIR) not in sys.path:
            sys.path.insert(0, str(EXPERIMENT_DIR))
        from audit_screen_v0 import audit_artifacts

        audit = audit_artifacts(output)
        if not audit.get("passed"):
            raise BlockedScreen(
                "blocked_reproducibility_or_audit_failure",
                "lightweight audit of the support blocker failed",
            )
        blocked_decision["independent_audit_passed"] = True
        write_json(output / "decision.json", blocked_decision)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "report.md").write_text(
            (output / "report.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
        raise BlockedScreen(
            "blocked_insufficient_loop_class_support",
            str(blocked_decision["blocker_detail"]),
        )
    class_order = target_class_order(mapping_manifest)
    support, scoring = support_audit(panel, class_order=class_order)
    mapping_manifest["development_class_support"] = support["development_class_support"]
    mapping_manifest["assessment_class_support"] = support["assessment_class_support"]
    mapping_manifest["tied_exclusions"] = support["tied_exclusions"]
    mapping_manifest["unavailable_exclusions"] = support["unavailable_exclusions"]
    development = scoring.loc[scoring["year"].eq(2024)].copy()
    assessment = scoring.loc[scoring["year"].eq(2025)].copy()
    models = {
        name: fit_multinomial(
            name,
            development,
            features=features,
            class_order=class_order,
        )
        for name, features in MODEL_FEATURES.items()
    }
    scored, probabilities, pooled, funnel = score_models(
        assessment, models, class_order=class_order
    )
    monthly = sliced_metrics(scored, models, class_order=class_order, group_column="year_month")
    checkpoint = sliced_metrics(
        scored, models, class_order=class_order, group_column="decision_ordinal"
    )
    per_class = class_metrics(scored, probabilities, class_order=class_order)
    bootstrap = paired_bootstrap_metrics(scored, probabilities, class_order=class_order)
    null = within_slate_null_metrics(
        development,
        assessment,
        models,
        probabilities,
        class_order=class_order,
    )
    concentration, frozen_medians = stability_and_concentration_metrics(
        development,
        assessment,
        models,
        class_order=class_order,
        support=support,
    )
    decision = derive_decision(
        pooled,
        monthly,
        checkpoint,
        bootstrap,
        null,
        funnel,
        support=support,
    )
    write_artifacts(
        output,
        contract=contract,
        source_context=source_context,
        behavioural_audit=behavioural_audit,
        v2_audit=v2_audit,
        dictionary_manifest=dictionary_manifest,
        mapping_manifest=mapping_manifest,
        interaction_bounds=interaction_bounds,
        support=support,
        panel=panel,
        models=models,
        scored=scored,
        pooled=pooled,
        monthly=monthly,
        checkpoint=checkpoint,
        per_class=per_class,
        funnel=funnel,
        bootstrap=bootstrap,
        null=null,
        concentration=concentration,
        frozen_medians=frozen_medians,
        decision=decision,
    )
    determinism = determinism_check(
        output,
        original_models=models,
        original_probabilities=probabilities,
        class_order=class_order,
        bootstrap=bootstrap,
        null=null,
        original_decision=decision,
        support=support,
    )
    decision["determinism_check_passed"] = bool(determinism["passed"])
    write_json(output / "decision.json", decision)

    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))
    from audit_screen_v0 import audit_artifacts

    audit = audit_artifacts(output)
    if not audit.get("passed"):
        raise BlockedScreen(
            "blocked_reproducibility_or_audit_failure", "lightweight independent audit failed"
        )
    decision["independent_audit_passed"] = True
    write_json(output / "decision.json", decision)
    report = render_report(
        decision=decision,
        support=support,
        mapping=mapping_manifest,
        pooled=pooled,
        funnel=funnel,
        bootstrap=bootstrap,
        null=null,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


def write_blocker(output: Path, blocker: BlockedScreen) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing_path = output / "decision.json"
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("decision") == blocker.code:
            return
    decision = {
        **SAFETY_FLAGS,
        "decision": blocker.code,
        "blocker_detail": blocker.detail,
        "independent_audit_passed": False,
        "determinism_check_passed": False,
    }
    write_json(output / "decision.json", decision)
    if CONTRACT_PATH.is_file():
        write_json(output / "contract.json", load_contract())
    report = (
        "# Emotion × Regime-Mix Loop Funnel Quick Screen V0\n\n"
        f"Decision: `{blocker.code}`\n\n"
        f"Blocked fail-closed: {blocker.detail}\n"
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    provider_root = args.provider_root.expanduser().resolve()
    try:
        decision = execute_screen(output, provider_root=provider_root)
        print(canonical_json(decision), end="")
        return 0
    except BlockedScreen as blocker:
        write_blocker(output, blocker)
        print(blocker.code)
        print(blocker.detail, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
