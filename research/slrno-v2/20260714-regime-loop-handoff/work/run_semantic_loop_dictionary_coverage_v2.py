#!/usr/bin/env python3
# ruff: noqa: E501
"""Run Semantic Loop Dictionary and First-Event Coverage V2.

This runner orchestrates isolated reusable modules.  It reads the frozen 2024
structural ledgers, bounds the unchanged retrospective validation read to 2025,
and never reads or scores an economic outcome.  It does not train a next-loop
predictor and exposes no execution surface.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
for import_root in (PACKAGE_ROOT, WORK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_loop_event_semantics_v2 as frozen_v2  # noqa: E402

from stocker_research.first_event_target_v2 import (  # noqa: E402
    build_first_event_target,
    build_loop_family_mapping,
    decide_target_tractability,
)
from stocker_research.loop_nulls_v2 import (  # noqa: E402
    ClockConditionedSemiMarkovNull,
    SemiMarkovNull,
    SessionRunSequence,
    SimulatedSession,
    circular_session_control,
    first_order_expected_counts,
)
from stocker_research.loop_structural_nulls_v2 import (  # noqa: E402
    FIRST_EVENT_FAMILY_ORDER,
    first_event_candidate_counts,
    information_increment_from_chronological_folds,
    qualify_structural_candidates,
    simulate_first_event_family_null_counts_fast,
    simulate_first_event_null_counts_by_group_fast,
    simulate_first_event_null_counts_fast,
    summarize_null_draws,
)
from stocker_research.loop_tie_resolution_v2 import (  # noqa: E402
    TieResolutionBundle,
    resolve_registered_ties,
)
from stocker_research.prefix_features_v2 import (  # noqa: E402
    build_compressed_prefix_features,
)
from stocker_research.semantic_loop_dictionary_v2 import (  # noqa: E402
    CandidateSupportGates,
    CandidateUniverseBundle,
    DictionarySelectionBundle,
    build_candidate_universe,
    deterministic_dictionary_hash,
    safety_flags,
    select_primary_dictionary,
)
from stocker_research.unregistered_loop_census_v2 import (  # noqa: E402
    UnregisteredVocabularyBundle,
    reconstruct_first_events,
    summarize_unregistered_vocabulary,
)

CONTRACT_PATH = WORK_DIR / "contracts" / "20260718-semantic-loop-dictionary-coverage-v2.json"
ARTIFACT_PARENT = WORK_DIR / "artifacts" / "20260718-semantic-loop-dictionary-coverage-v2"
REPORT_PATH = WORK_DIR / "reports" / "20260718-semantic-loop-dictionary-coverage-v2.md"
OLD_ARTIFACT_PARENT = WORK_DIR / "artifacts" / "20260718-loop-event-semantics-v2"
OLD_PRIMARY = OLD_ARTIFACT_PARENT / "primary"
OLD_EXACT = OLD_ARTIFACT_PARENT / "exact_rerun"
DICTIONARY_VERSION = "semantic_loop_dictionary_first_event_v2"
STATE_MODEL_VERSION = "causal_semimarkov_posterior_export_v2_tail78"
HORIZON_BARS = 24
PRIMARY_DRAWS = 2_000
CLOCK_DRAWS = 500
NULL_SEED = 20_260_718


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    run_id: str
    git_sha: str
    contract_hash: str
    data_snapshot_hash: str
    dictionary_version: str
    dictionary_hash: str
    state_model_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "contract_hash": self.contract_hash,
            "data_snapshot_hash": self.data_snapshot_hash,
            "dictionary_version": self.dictionary_version,
            "dictionary_hash": self.dictionary_hash,
            "state_model_version": self.state_model_version,
        }

    def for_snapshot(self, data_snapshot_hash: str) -> ArtifactIdentity:
        return replace(self, data_snapshot_hash=data_snapshot_hash)


@dataclass(frozen=True, slots=True)
class ArtifactWriter:
    identity: ArtifactIdentity

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        _write_json(path, payload, identity=self.identity)

    def write_frame(
        self,
        path: Path,
        frame: pd.DataFrame,
        *,
        source_artifact: str,
        copy_frame: bool = True,
    ) -> None:
        _write_frame(
            path,
            frame,
            source_artifact=source_artifact,
            copy_frame=copy_frame,
            identity=self.identity,
        )


@dataclass(frozen=True, slots=True)
class PreflightContext:
    output_dir: Path
    primary_dir: Path
    contract: dict[str, Any]
    contract_hash: str
    preflight: dict[str, Any]
    git_sha: str
    branch: str
    development_snapshot_hash: str
    validation_snapshot_hash: str


@dataclass(frozen=True, slots=True)
class RunContext:
    preflight: PreflightContext
    identity: ArtifactIdentity
    validation_identity: ArtifactIdentity

    @property
    def writer(self) -> ArtifactWriter:
        return ArtifactWriter(self.identity)

    @property
    def validation_writer(self) -> ArtifactWriter:
        return ArtifactWriter(self.validation_identity)


@dataclass(slots=True)
class DevelopmentReconstruction:
    decisions: pd.DataFrame
    first_events: pd.DataFrame
    old_dictionary: pd.DataFrame
    old_primitive_ids: tuple[str, ...]
    old_unregistered_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class DictionaryPhase:
    context: RunContext
    candidate_bundle: CandidateUniverseBundle
    sessions: tuple[SessionRunSequence, ...]
    null_results: pd.DataFrame
    clock_results: pd.DataFrame
    development_draws: np.ndarray
    development_clock_draws: np.ndarray
    development_family_draws: np.ndarray
    development_family_clock_draws: np.ndarray
    development_deletion_details: pd.DataFrame
    analytical: pd.DataFrame
    circular: pd.DataFrame
    information: pd.DataFrame
    scored: pd.DataFrame
    selection: DictionarySelectionBundle
    dictionary: pd.DataFrame
    dictionary_hash: str
    selected_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class DevelopmentArtifacts:
    auxiliary_registry: pd.DataFrame
    legacy_mapping: pd.DataFrame
    target_contract: dict[str, Any]
    development_light: pd.DataFrame
    class_counts: pd.DataFrame
    legacy_comparison_summary: pd.DataFrame
    prefix_manifest: pd.DataFrame
    prefix_missingness: pd.DataFrame
    prefix_distribution: pd.DataFrame
    prefix_feature_rows: int
    tie_bundle: TieResolutionBundle
    vocabulary: UnregisteredVocabularyBundle
    unregistered_unique_primitives: int
    motif_distribution: dict[str, int]
    development_decision_count: int


@dataclass(frozen=True, slots=True)
class ValidationPhase:
    validation_decision_count: int
    validation_session_count: int
    validation_null: pd.DataFrame
    validation_clock: pd.DataFrame
    validation_deletion_details: pd.DataFrame
    replication: pd.DataFrame
    rank_stability: pd.DataFrame
    development_coverage: pd.DataFrame
    validation_coverage: pd.DataFrame
    coverage_by_stock: pd.DataFrame
    coverage_by_month: pd.DataFrame
    coverage_by_clock: pd.DataFrame
    coverage_by_state: pd.DataFrame
    stock_deletions: pd.DataFrame
    family_mapping: pd.DataFrame
    family_counts: pd.DataFrame
    exact_vs_family: pd.DataFrame
    family_stability: pd.DataFrame
    scientific_decision: dict[str, Any]
    development_coverage_value: float
    validation_coverage_value: float


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
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"cannot encode {type(value).__name__}")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


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


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    identity: ArtifactIdentity,
) -> None:
    output = {**identity.as_dict(), **payload, **safety_flags()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            output,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ).encode("utf-8")
        + b"\n"
    )


def _identity_frame(
    frame: pd.DataFrame,
    *,
    source_artifact: str,
    copy_frame: bool = True,
    identity: ArtifactIdentity,
) -> pd.DataFrame:
    artifact_identity = identity
    identity_values = artifact_identity.as_dict()
    output = frame.copy() if copy_frame else frame
    if "run_id" in output and not output["run_id"].astype(str).eq(artifact_identity.run_id).all():
        output = output.rename(columns={"run_id": "source_state_run_id"})
    for key, value in identity_values.items():
        output[key] = value
    for key, value in safety_flags().items():
        output[key] = value
    defaults: dict[str, Any] = {
        "decision_id": "not_applicable",
        "symbol": "not_applicable",
        "session": "not_applicable",
        "decision_timestamp": pd.NaT,
        "event_timestamp": pd.NaT,
        "semantic_loop_id": None,
        "primitive_loop_id": None,
        "source_artifact": source_artifact,
        "source_hash": artifact_identity.data_snapshot_hash,
    }
    for key, value in defaults.items():
        if key not in output:
            output[key] = value
    return output


def _write_frame(
    path: Path,
    frame: pd.DataFrame,
    *,
    source_artifact: str,
    copy_frame: bool = True,
    identity: ArtifactIdentity,
) -> None:
    output = _identity_frame(
        frame,
        source_artifact=source_artifact,
        copy_frame=copy_frame,
        identity=identity,
    )
    _serialize_frame(path, output)


def _serialize_frame(path: Path, output: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        output.to_parquet(path, index=False, compression="zstd", compression_level=9)
    elif path.suffix == ".csv":
        output.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    else:
        raise ValueError(f"unsupported tabular artifact {path}")


def _write_mixed_period_frame(
    path: Path,
    frame: pd.DataFrame,
    *,
    source_artifact: str,
    context: RunContext,
) -> None:
    if "period" not in frame:
        raise ValueError("mixed-period artifact requires an explicit period column")
    working = frame.copy()
    working["_source_row_order"] = np.arange(len(working), dtype=int)
    validation_mask = working["period"].astype(str).str.contains("validation_2025")
    if not validation_mask.any() or validation_mask.all():
        raise ValueError("mixed-period artifact must contain development and validation rows")
    development = _identity_frame(
        working.loc[~validation_mask],
        source_artifact=source_artifact,
        identity=context.identity,
    )
    validation = _identity_frame(
        working.loc[validation_mask],
        source_artifact=source_artifact,
        identity=context.validation_identity,
    )
    output = (
        pd.concat([development, validation], ignore_index=True)
        .sort_values("_source_row_order")
        .drop(columns="_source_row_order")
        .reset_index(drop=True)
    )
    _serialize_frame(path, output)


def _verify_declared_manifest(directory: Path) -> None:
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["artifacts"]:
        path = directory / record["file"]
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"audited artifact missing or changed: {path}")
        if _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"audited artifact hash mismatch: {path}")


def _verify_frozen_tree(contract: Mapping[str, Any]) -> tuple[int, str]:
    frozen_blobs = pd.read_csv(OLD_PRIMARY / "frozen_historical_tree_hashes.csv")
    if len(frozen_blobs) != 1_087 or not frozen_blobs["byte_unchanged"].astype(bool).all():
        raise RuntimeError("frozen historical baseline is incomplete")
    for row in frozen_blobs.itertuples(index=False):
        path = REPO_ROOT / str(row.file)
        if not path.is_file() or _sha256_file(path) != str(row.current_sha256):
            raise RuntimeError(f"frozen historical file changed: {row.file}")
    source_identity = contract["source_identity"]
    baseline_commit = str(source_identity["frozen_lineage_baseline_commit"])
    prefix = "research/slrno-v2/20260714-regime-loop-handoff"
    tree_hash = _git("rev-parse", f"{baseline_commit}:{prefix}")
    if tree_hash != str(source_identity["frozen_historical_tree_git_object"]):
        raise RuntimeError("frozen historical tree object differs from the contract")
    return len(frozen_blobs), tree_hash


def _preflight(contract: Mapping[str, Any]) -> dict[str, Any]:
    _verify_declared_manifest(OLD_PRIMARY)
    _verify_declared_manifest(OLD_EXACT)
    primary_manifest = json.loads((OLD_PRIMARY / "artifact_manifest.json").read_text())
    exact_manifest = json.loads((OLD_EXACT / "artifact_manifest.json").read_text())
    primary_hashes = {row["file"]: row["sha256"] for row in primary_manifest["artifacts"]}
    exact_hashes = {row["file"]: row["sha256"] for row in exact_manifest["artifacts"]}
    if primary_hashes != exact_hashes:
        raise RuntimeError("audited primary and exact-rerun manifests differ")
    source_manifest = json.loads(
        (ARTIFACT_PARENT / "primary" / "source_identity_manifest.json").read_text()
    )
    for relative, expected in source_manifest["source_file_hashes"].items():
        if _sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"audited V2 source identity changed: {relative}")
    decisions = pd.read_parquet(
        OLD_PRIMARY / "causal_completed_bar_decisions.parquet",
        columns=[
            "decision_id",
            "structural_event_eligibility",
            "is_run_entry",
            "active_prefix_count",
        ],
    )
    outcomes = pd.read_parquet(
        OLD_PRIMARY / "first_next_loop_outcomes.parquet", columns=["decision_id", "primary_label"]
    )
    completions = pd.read_parquet(
        OLD_PRIMARY / "loop_completion_event_ledger.parquet", columns=["decision_id"]
    )
    comparison = pd.read_parquet(
        OLD_PRIMARY / "legacy_v2_target_comparison_detail.parquet",
        columns=[
            "comparison_available",
            "semantics_differ",
            "registered_event_set_differs",
        ],
    )
    legacy = pd.read_parquet(
        OLD_PRIMARY / "legacy_overlapping_targets.parquet",
        columns=["decision_id", "legacy_positive_count"],
    )
    expected = {
        "completed_bar_decisions": 424_583,
        "eligible_decisions": 398_304,
        "run_entries": 110_949,
        "completion_rows": 487_721,
        "comparable": 398_304,
        "semantic_differences": 369_745,
        "event_set_differences": 175_054,
        "legacy_multiple_positives": 14_331,
        "decisions_with_active_prefix": 317_181,
        "old_ties": 11_003,
        "old_unregistered": 271_500,
    }
    actual = {
        "completed_bar_decisions": len(decisions),
        "eligible_decisions": int(decisions["structural_event_eligibility"].sum()),
        "run_entries": int(decisions["is_run_entry"].sum()),
        "completion_rows": len(completions),
        "comparable": int(comparison["comparison_available"].sum()),
        "semantic_differences": int(comparison["semantics_differ"].sum()),
        "event_set_differences": int(comparison["registered_event_set_differs"].sum()),
        "legacy_multiple_positives": int(legacy["legacy_positive_count"].gt(1).sum()),
        "decisions_with_active_prefix": int(decisions["active_prefix_count"].gt(0).sum()),
        "old_ties": int(outcomes["primary_label"].eq("TIED_REGISTERED_COMPLETION").sum()),
        "old_unregistered": int(outcomes["primary_label"].eq("UNREGISTERED_LOOP").sum()),
    }
    if actual != expected:
        raise RuntimeError(f"audited V2 ledger counts differ: {actual}")
    frozen_count, frozen_tree = _verify_frozen_tree(contract)
    return {
        **actual,
        "declared_artifacts_verified_per_tree": len(primary_hashes),
        "primary_exact_byte_identical": True,
        "frozen_historical_blob_count": frozen_count,
        "frozen_historical_tree_hash": frozen_tree,
    }


def _development_decisions() -> pd.DataFrame:
    columns = [
        "decision_id",
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "decision_timestamp",
        "hard_state_legacy",
        "clock_phase",
        "structural_event_eligibility",
        "source_sequence_complete",
        "source_sequence_missing_reason",
        "source_artifact_hash",
        "run_id",
        "git_sha",
        "contract_hash",
        "data_snapshot_hash",
        "dictionary_version",
        "dictionary_hash",
        "state_model_version",
    ]
    frame = pd.read_parquet(OLD_PRIMARY / "causal_completed_bar_decisions.parquet", columns=columns)
    frame["source_available"] = frame["source_sequence_complete"].astype(bool)
    return frame


def _attach_legacy_overlapping_labels(first_events: pd.DataFrame) -> pd.DataFrame:
    legacy = pd.read_parquet(
        OLD_PRIMARY / "legacy_overlapping_targets.parquet",
        columns=["decision_id", "legacy_positive_semantic_ids"],
    )
    if legacy["decision_id"].duplicated().any():
        raise RuntimeError("legacy overlapping targets are not one-to-one with decisions")
    label_lookup = {
        str(row.decision_id): [str(value) for value in row.legacy_positive_semantic_ids]
        for row in legacy.itertuples(index=False)
    }
    output = first_events.copy()
    output["legacy_overlapping_positive_labels"] = output["decision_id"].map(
        lambda decision_id: label_lookup.get(str(decision_id), [])
    )
    return output


def _information_decisions(first_events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "decision_id",
        "decision_timestamp",
        "symbol",
        "session",
        "bar_ordinal",
        "hard_state_legacy",
        "previous_completed_state_1",
        "previous_completed_state_2",
        "previous_completed_state_3",
        "hard_run_age",
        "structural_event_eligibility",
    ]
    decisions = pd.read_parquet(
        OLD_PRIMARY / "causal_completed_bar_decisions.parquet", columns=columns
    )
    ordered = decisions.sort_values(
        ["symbol", "session", "bar_ordinal", "decision_id"], kind="mergesort"
    ).reset_index(drop=True)
    histories: list[tuple[int, ...]] = [()] * len(ordered)
    for _, group in ordered.groupby(["symbol", "session"], sort=False):
        state_events: list[int] = []
        previous_state: int | None = None
        for index, row in group.iterrows():
            if not bool(row["structural_event_eligibility"]):
                state_events = []
                previous_state = None
                continue
            state = int(row["hard_state_legacy"])
            if state != previous_state:
                state_events.append(state)
                previous_state = state
            histories[int(index)] = tuple(state_events[-8:])
    ordered["recent_state_events"] = histories
    history_fields = [
        "decision_id",
        "primitive_loop_id",
        "previous_completed_primitive_loop",
        "same_primitive_repeat_depth",
        "bars_since_previous_primitive_completion",
    ]
    return ordered.loc[ordered["structural_event_eligibility"].astype(bool)].merge(
        first_events[history_fields],
        on="decision_id",
        how="left",
        validate="one_to_one",
    )


def _prefix_decisions(first_events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "decision_id",
        "symbol",
        "session",
        "bar_ordinal",
        "decision_timestamp",
        "hard_state_legacy",
        "hard_run_age",
        "hard_state_hysteretic",
        "posterior_entropy",
        "top_second_margin",
        "expected_state_age",
        "transition_probability_next_bar",
        "bars_remaining_in_session",
        "clock_phase",
        "structural_event_eligibility",
        "source_artifact_hash",
        "run_id",
        "git_sha",
        "contract_hash",
        "data_snapshot_hash",
        "dictionary_version",
        "dictionary_hash",
        "state_model_version",
    ]
    decisions = pd.read_parquet(
        OLD_PRIMARY / "causal_completed_bar_decisions.parquet", columns=columns
    )
    history_columns = [
        "decision_id",
        "previous_completed_primitive_loop",
        "previous_two_completed_primitive_loops",
        "same_primitive_repeat_depth",
        "bars_since_previous_primitive_completion",
        "state_events_since_previous_primitive_completion",
    ]
    return decisions.merge(
        first_events[history_columns], on="decision_id", how="left", validate="one_to_one"
    )


def _development_sessions() -> tuple[SessionRunSequence, ...]:
    runs = pd.read_parquet(
        OLD_PRIMARY / "structural_session_runs.parquet",
        columns=[
            "symbol",
            "session",
            "state",
            "duration",
            "right_censored",
            "source_sequence_complete",
        ],
    )
    sessions: list[SessionRunSequence] = []
    complete = runs.loc[runs["source_sequence_complete"].astype(bool)]
    for (symbol, session), group in complete.groupby(["symbol", "session"], sort=True):
        sessions.append(
            SessionRunSequence(
                symbol=str(symbol),
                session=str(session),
                states=tuple(group["state"].astype(int)),
                durations=tuple(group["duration"].astype(int)),
                terminal_right_censored=bool(group["right_censored"].iloc[-1]),
            )
        )
    return tuple(sessions)


def _resolve_old_ties() -> Any:
    outcomes = pd.read_parquet(OLD_PRIMARY / "first_next_loop_outcomes.parquet")
    completions = pd.read_parquet(OLD_PRIMARY / "loop_completion_event_ledger.parquet")
    composite_table = pd.read_csv(OLD_PRIMARY / "composite_component_mapping.csv")
    components = {
        str(row.semantic_loop_id): tuple(ast.literal_eval(str(row.component_primitive_ids)))
        for row in composite_table.itertuples(index=False)
    }
    return resolve_registered_ties(outcomes, completions, composite_components=components)


def _candidate_paths(candidate_support: pd.DataFrame) -> tuple[list[tuple[int, ...]], list[int]]:
    paths: list[tuple[int, ...]] = []
    owners: list[int] = []
    for candidate_index, row in candidate_support.reset_index(drop=True).iterrows():
        for raw_path in row["allowed_orientations"]:
            paths.append(tuple(int(value) for value in raw_path))
            owners.append(int(candidate_index))
    return paths, owners


def _analytical_results(
    sessions: Sequence[SessionRunSequence],
    candidate_support: pd.DataFrame,
    first_events: pd.DataFrame,
) -> pd.DataFrame:
    model = SemiMarkovNull.fit(sessions, state_count=8, maximum_duration=78)
    paths, owners = _candidate_paths(candidate_support)
    orientation_expected = first_order_expected_counts(sessions, model, paths)
    expected = np.zeros(len(candidate_support), dtype=float)
    for value, owner in zip(orientation_expected, owners, strict=True):
        expected[owner] += float(value)
    independent = (
        first_events.loc[
            first_events["primitive_loop_id"].notna(),
            ["primitive_loop_id", "event_key"],
        ]
        .groupby("primitive_loop_id")["event_key"]
        .nunique()
    )
    return pd.DataFrame(
        {
            "semantic_loop_id": candidate_support["semantic_loop_id"].to_numpy(),
            "primitive_loop_id": candidate_support["primitive_loop_id"].to_numpy(),
            "observed_independent_completion_events": candidate_support["primitive_loop_id"]
            .map(independent)
            .fillna(0)
            .astype(int),
            "analytical_expected_count": expected,
            "analytical_control": "first_order_oriented_raw_completion_expectation",
        }
    )


def _circular_results(
    sessions: Sequence[SessionRunSequence], candidate_ids: Sequence[str]
) -> pd.DataFrame:
    counts = np.zeros(len(candidate_ids), dtype=np.int64)
    used = 0
    for session in sessions:
        if len(session.states) < 2:
            continue
        digest = hashlib.sha256(f"{session.symbol}|{session.session}".encode()).digest()
        offset = int.from_bytes(digest[:4], "big") % (len(session.states) - 1) + 1
        rotated = circular_session_control(session, offset=offset)
        simulated = SimulatedSession(
            states=rotated.states,
            durations=rotated.durations,
            terminal_right_censored=rotated.terminal_right_censored,
            phase_labels=tuple("not_used" for _ in range(sum(rotated.durations))),
        )
        counts += first_event_candidate_counts(
            simulated, candidate_ids=candidate_ids, horizon_bars=HORIZON_BARS
        )
        used += 1
    return pd.DataFrame(
        {
            "semantic_loop_id": list(candidate_ids),
            "circular_first_event_count": counts,
            "sessions_rotated": used,
            "control": "deterministic_whole_state_duration_block_rotation",
        }
    )


def _score_nulls(
    first_events: pd.DataFrame,
    decisions: pd.DataFrame,
    support: pd.DataFrame,
    sessions: Sequence[SessionRunSequence],
    *,
    draws: int,
    clock_draws: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    candidates = support.loc[support["support_pass"].astype(bool)].copy().reset_index(drop=True)
    candidate_ids = candidates["semantic_loop_id"].astype(str).tolist()
    base = SemiMarkovNull.fit(sessions, state_count=8, maximum_duration=78)
    clock = ClockConditionedSemiMarkovNull.fit(sessions, state_count=8, maximum_duration=78)
    lengths = [sum(session.durations) for session in sessions]
    stock_labels, stock_resolved_draws = simulate_first_event_null_counts_by_group_fast(
        base,
        session_lengths=lengths,
        session_groups=[session.symbol for session in sessions],
        candidate_ids=candidate_ids,
        horizon_bars=HORIZON_BARS,
        draws=draws,
        seed=seed,
    )
    primary_draws = stock_resolved_draws.sum(axis=1)
    broad_clock_draws = simulate_first_event_null_counts_fast(
        clock,
        session_lengths=lengths,
        candidate_ids=candidate_ids,
        horizon_bars=HORIZON_BARS,
        draws=clock_draws,
        seed=seed + 1,
    )
    family_draws = simulate_first_event_family_null_counts_fast(
        base,
        session_lengths=lengths,
        horizon_bars=HORIZON_BARS,
        draws=draws,
        seed=seed,
    )
    family_clock_draws = simulate_first_event_family_null_counts_fast(
        clock,
        session_lengths=lengths,
        horizon_bars=HORIZON_BARS,
        draws=clock_draws,
        seed=seed + 1,
    )
    observed = first_events["primitive_loop_id"].value_counts().reindex(candidate_ids, fill_value=0)
    statistics = summarize_null_draws(observed.to_numpy(), primary_draws, broad_clock_draws)
    statistics.insert(0, "semantic_loop_id", candidate_ids)
    statistics.insert(1, "primitive_loop_id", candidate_ids)
    eligible = decisions.loc[
        decisions["structural_event_eligibility"].astype(bool),
        ["decision_timestamp", "symbol"],
    ].copy()
    total_eligible = len(eligible)
    statistics["eligible_decisions"] = total_eligible
    statistics["observed_rate"] = statistics["observed_count"] / total_eligible

    eligible["_quarter"] = (
        pd.to_datetime(eligible["decision_timestamp"], utc=True).dt.year.astype(str)
        + "Q"
        + ((pd.to_datetime(eligible["decision_timestamp"], utc=True).dt.month - 1) // 3 + 1).astype(
            str
        )
    )
    quarter_totals = eligible["_quarter"].value_counts().to_dict()
    event_quarters = first_events.loc[
        first_events["primitive_loop_id"].notna(),
        ["primitive_loop_id", "decision_timestamp"],
    ].copy()
    event_quarters["_quarter"] = (
        pd.to_datetime(event_quarters["decision_timestamp"], utc=True).dt.year.astype(str)
        + "Q"
        + (
            (pd.to_datetime(event_quarters["decision_timestamp"], utc=True).dt.month - 1) // 3 + 1
        ).astype(str)
    )
    positive_quarters = []
    for row in statistics.itertuples(index=False):
        count = 0
        candidate_events = event_quarters.loc[
            event_quarters["primitive_loop_id"].eq(row.semantic_loop_id)
        ]
        for quarter, quarter_total in sorted(quarter_totals.items()):
            observed_quarter = int(candidate_events["_quarter"].eq(quarter).sum())
            expected_quarter = float(row.semi_markov_null_mean) * quarter_total / total_eligible
            count += observed_quarter > expected_quarter
        positive_quarters.append(count)
    statistics["positive_excess_quarters"] = positive_quarters

    event_by_stock = (
        first_events.loc[first_events["primitive_loop_id"].notna(), ["primitive_loop_id", "symbol"]]
        .groupby(["primitive_loop_id", "symbol"])
        .size()
        .to_dict()
    )
    expected_without_stock = (primary_draws[:, None, :] - stock_resolved_draws).mean(axis=0)
    observed_total = statistics["observed_count"].to_numpy(dtype=float)
    observed_by_stock = np.asarray(
        [
            [float(event_by_stock.get((candidate_id, symbol), 0)) for candidate_id in candidate_ids]
            for symbol in stock_labels
        ],
        dtype=float,
    )
    deletion_ratios = (observed_total[None, :] - observed_by_stock + 0.5) / (
        expected_without_stock + 0.5
    )
    eligible_by_stock = eligible["symbol"].value_counts().to_dict()
    deletion_records = [
        {
            "semantic_loop_id": candidate_id,
            "primitive_loop_id": candidate_id,
            "deleted_stock": symbol,
            "eligible_decisions_without_stock": total_eligible
            - int(eligible_by_stock.get(symbol, 0)),
            "observed_count_without_stock": int(
                observed_total[candidate_index] - observed_by_stock[stock_index, candidate_index]
            ),
            "semi_markov_null_mean_without_stock": float(
                expected_without_stock[stock_index, candidate_index]
            ),
            "semi_markov_rate_ratio_without_stock": float(
                deletion_ratios[stock_index, candidate_index]
            ),
            "recomputation_method": (
                "session_resolved_semi_markov_draw_subtraction_without_rate_scaling"
            ),
        }
        for stock_index, symbol in enumerate(stock_labels)
        for candidate_index, candidate_id in enumerate(candidate_ids)
    ]
    deletion_details = pd.DataFrame.from_records(deletion_records)
    statistics["leave_one_stock_out_minimum_rate_ratio"] = deletion_ratios.min(axis=0)
    statistics["leave_one_stock_out_deletions"] = len(stock_labels)
    statistics["leave_one_stock_out_method"] = (
        "session_resolved_semi_markov_draw_subtraction_without_rate_scaling"
    )
    merged = candidates.merge(statistics, on=["semantic_loop_id", "primitive_loop_id"])
    qualified = qualify_structural_candidates(merged)
    clock_results = qualified[
        [
            "semantic_loop_id",
            "primitive_loop_id",
            "observed_count",
            "clock_null_mean",
            "clock_null_lower",
            "clock_null_upper",
            "clock_null_rate_ratio",
            "clock_null_p",
            "clock_null_q",
            "clock_null_status",
        ]
    ].copy()
    return (
        qualified,
        clock_results,
        primary_draws,
        broad_clock_draws,
        family_draws,
        family_clock_draws,
        deletion_details,
    )


def _validation_snapshot_hash() -> str:
    frozen_v2.DEVELOPMENT_START = pd.Timestamp("2025-01-01", tz="UTC")
    frozen_v2.DEVELOPMENT_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
    _, snapshot_hash = frozen_v2._source_hashes()
    return str(snapshot_hash)


def _validation_surface(
    contract_hash: str,
    identity: ArtifactIdentity,
) -> tuple[pd.DataFrame, str, tuple[SessionRunSequence, ...]]:
    frozen_v2.DEVELOPMENT_START = pd.Timestamp("2025-01-01", tz="UTC")
    frozen_v2.DEVELOPMENT_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
    source_hashes, snapshot_hash = frozen_v2._source_hashes()
    panel = frozen_v2._prepare_panel(
        source_hashes, data_snapshot_hash=snapshot_hash, contract_hash=contract_hash
    )
    timestamps = pd.to_datetime(panel["bar_start_timestamp"], utc=True)
    if timestamps.min() < pd.Timestamp("2025-01-01", tz="UTC") or timestamps.max() >= pd.Timestamp(
        "2026-01-01", tz="UTC"
    ):
        raise RuntimeError("validation surface escaped the frozen 2025 boundary")
    preprocessing = pd.read_csv(frozen_v2.PREPROCESSING_PATH)
    model = {key: value for key, value in np.load(frozen_v2.MODEL_PATH).items()}
    scaled = frozen_v2.frozen_core.scale_emissions(panel, preprocessing)
    log_emissions = frozen_v2.frozen_core.log_emission(scaled, model)
    groups = frozen_v2.frozen_core.group_positions(
        panel.rename(columns={"symbol": "symbol_norm", "session": "session_date"})
    )
    labels, ages, _ = frozen_v2.frozen_core.causal_filter(log_emissions, groups, model)
    decisions = panel[
        [
            "symbol",
            "session",
            "bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
            "clock_phase",
            "source_sequence_complete",
            "source_sequence_missing_reason",
        ]
    ].copy()
    decisions["decision_timestamp"] = decisions["bar_complete_timestamp"]
    decisions["hard_state_legacy"] = labels.astype(int)
    decisions["hard_run_age"] = ages.astype(int)
    decisions["structural_event_eligibility"] = decisions["source_sequence_complete"].astype(bool)
    decisions["source_available"] = decisions["source_sequence_complete"].astype(bool)
    decisions["source_artifact_hash"] = snapshot_hash
    decisions["git_sha"] = identity.git_sha
    decisions["contract_hash"] = contract_hash
    decisions["data_snapshot_hash"] = snapshot_hash
    decisions["dictionary_version"] = DICTIONARY_VERSION
    decisions["dictionary_hash"] = identity.dictionary_hash
    decisions["state_model_version"] = STATE_MODEL_VERSION
    decisions["run_id"] = (
        decisions.groupby(["symbol", "session"], sort=False)["hard_state_legacy"]
        .transform(lambda values: values.ne(values.shift()).cumsum())
        .astype(int)
    )
    decisions["decision_id"] = [
        hashlib.sha256(f"validation|{symbol}|{session}|{bar}".encode()).hexdigest()[:24]
        for symbol, session, bar in zip(
            decisions["symbol"], decisions["session"], decisions["bar_ordinal"], strict=True
        )
    ]
    sessions: list[SessionRunSequence] = []
    for (symbol, session), group in decisions.loc[
        decisions["source_sequence_complete"].astype(bool)
    ].groupby(["symbol", "session"], sort=True):
        states = group["hard_state_legacy"].to_numpy(dtype=int)
        starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
        ends = np.r_[starts[1:], len(states)]
        sessions.append(
            SessionRunSequence(
                symbol=str(symbol),
                session=str(session),
                states=tuple(int(states[start]) for start in starts),
                durations=tuple(int(end - start) for start, end in zip(starts, ends, strict=True)),
                terminal_right_censored=True,
            )
        )
    del panel, scaled, log_emissions, model
    gc.collect()
    return decisions.reset_index(drop=True), snapshot_hash, tuple(sessions)


def _legacy_mapping(selected_ids: set[str]) -> pd.DataFrame:
    old = pd.read_csv(OLD_PRIMARY / "semantic_loop_dictionary_v2.csv")
    rows = []
    for row in old.itertuples(index=False):
        primitive = str(row.primitive_loop_id) if pd.notna(row.primitive_loop_id) else None
        components = ast.literal_eval(str(row.component_primitive_ids))
        rows.append(
            {
                "legacy_semantic_loop_id": str(row.semantic_loop_id),
                "legacy_motif_type": str(row.motif_type),
                "legacy_repeat_depth": int(row.repeat_depth),
                "primitive_loop_id": primitive,
                "component_primitive_ids": components,
                "selected_primary_dictionary_member": primitive in selected_ids,
                "migration_status": (
                    "DIRECT_PRIMITIVE_OR_REPEAT_ROOT"
                    if primitive is not None
                    else "AUXILIARY_COMPOSITE_NO_SINGLE_GLOBAL_ROOT"
                ),
            }
        )
    mapping = pd.DataFrame.from_records(rows)
    if len(mapping) != 20 or mapping["legacy_semantic_loop_id"].duplicated().any():
        raise RuntimeError("legacy dictionary mapping has duplicates or omissions")
    return mapping


def _auxiliary_registry(first_events: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in dictionary.itertuples(index=False):
        path = tuple(int(value) for value in entry.closed_path)
        core = path[:-1]
        observed = first_events.loc[first_events["primitive_loop_id"].eq(entry.primitive_loop_id)]
        maximum_depth = int(observed["current_repeat_depth"].fillna(1).max())
        for depth in range(2, maximum_depth + 1):
            full_path = tuple(core * depth) + (core[0],)
            rows.append(
                {
                    "semantic_motif_id": f"loop_r{depth}_{'-'.join(map(str, path))}",
                    "motif_type": "repeat",
                    "primitive_loop_id": entry.primitive_loop_id,
                    "repeat_depth": depth,
                    "component_primitive_ids": [entry.primitive_loop_id] * depth,
                    "full_path": list(full_path),
                    "closed_path": list(full_path),
                    "allowed_orientations": [list(full_path)],
                    "transition_length": len(full_path) - 1,
                    "legacy_cycle_ids": [],
                    "nested_completion_policy": "auxiliary_under_first_primitive_completion",
                    "primary_class_eligible": False,
                    "metadata_use": "repeat_burst_and_depth",
                }
            )
    composites = first_events.loc[
        first_events["motif_type"].eq("composite") & first_events["transition_length"].le(8)
    ]
    counts = composites["semantic_motif_id"].value_counts()
    for motif_id in sorted(counts.loc[counts.ge(100)].index):
        group = composites.loc[composites["semantic_motif_id"].eq(motif_id)]
        representative = group.sort_values("decision_id").iloc[0]
        rows.append(
            {
                "semantic_motif_id": motif_id,
                "motif_type": "composite",
                "primitive_loop_id": representative["primitive_loop_id"],
                "repeat_depth": 1,
                "component_primitive_ids": representative["component_primitive_ids"],
                "full_path": representative["full_closed_path"],
                "closed_path": representative["full_closed_path"],
                "allowed_orientations": [representative["full_closed_path"]],
                "transition_length": int(representative["transition_length"]),
                "legacy_cycle_ids": [],
                "nested_completion_policy": "never_replaces_earlier_primitive_primary",
                "primary_class_eligible": False,
                "metadata_use": "nested_composite_sequence",
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("semantic_motif_id").reset_index(drop=True)


def _coverage_summary(target: pd.DataFrame, *, period: str) -> pd.DataFrame:
    primitive = target["primitive_loop_id"].notna()
    selected = target["primary_class"].astype(str).str.startswith("loop_p_")
    unavailable = target["primary_class"].isin(["UNAVAILABLE_SOURCE", "UNAVAILABLE_STRUCTURAL_GAP"])
    eligible = ~unavailable
    primitive_count = int(primitive.sum())
    return pd.DataFrame(
        [
            {
                "period": period,
                "total_decisions": len(target),
                "eligible_decisions": int(eligible.sum()),
                "primitive_loop_event_decisions": primitive_count,
                "selected_dictionary_events": int(selected.sum()),
                "selected_dictionary_event_coverage": (
                    float(selected.sum() / primitive_count) if primitive_count else 0.0
                ),
                "other_primitive_loop_count": int(
                    target["primary_class"].eq("OTHER_PRIMITIVE_LOOP").sum()
                ),
                "other_primitive_loop_share": (
                    float(
                        target["primary_class"].eq("OTHER_PRIMITIVE_LOOP").sum() / primitive_count
                    )
                    if primitive_count
                    else 0.0
                ),
                "no_loop_share": float(
                    target["primary_class"].eq("NO_LOOP_WITHIN_HORIZON").sum()
                    / max(int(eligible.sum()), 1)
                ),
                "session_end_share": float(
                    target["primary_class"].eq("SESSION_END").sum() / max(int(eligible.sum()), 1)
                ),
                "genuine_tie_share": float(
                    target["primary_class"].eq("DISTINCT_PRIMITIVE_TIE").sum()
                    / max(int(eligible.sum()), 1)
                ),
                "unavailable_share": float(unavailable.mean()),
            }
        ]
    )


def _dictionary_replication(
    dictionary: pd.DataFrame,
    development_null: pd.DataFrame,
    validation_null: pd.DataFrame,
) -> pd.DataFrame:
    """Join period-specific dictionary evidence without relying on suffix side effects."""

    development = dictionary[
        [
            "semantic_loop_id",
            "development_count",
            "semi_markov_rate_ratio",
            "semi_markov_q",
            "selection_rank",
        ]
    ].rename(
        columns={
            "semi_markov_rate_ratio": "semi_markov_rate_ratio_development",
            "semi_markov_q": "semi_markov_q_development",
        }
    )
    development_excess = development_null[["semantic_loop_id", "excess_count"]].rename(
        columns={"excess_count": "excess_count_development"}
    )
    validation = validation_null[
        [
            "semantic_loop_id",
            "observed_count",
            "semi_markov_rate_ratio",
            "semi_markov_p",
            "semi_markov_q",
            "excess_count",
        ]
    ].rename(
        columns={
            "observed_count": "observed_count_validation",
            "semi_markov_rate_ratio": "semi_markov_rate_ratio_validation",
            "semi_markov_p": "semi_markov_p_validation",
            "semi_markov_q": "semi_markov_q_validation",
            "excess_count": "excess_count_validation",
        }
    )
    return (
        development.merge(
            development_excess,
            on="semantic_loop_id",
            validate="one_to_one",
        )
        .merge(validation, on="semantic_loop_id", validate="one_to_one")
        .sort_values("selection_rank")
        .reset_index(drop=True)
    )


def _coverage_by(target: pd.DataFrame, dimension: str, *, period: str) -> pd.DataFrame:
    records = []
    for value, group in target.groupby(dimension, dropna=False, sort=True):
        primitive = group["primitive_loop_id"].notna()
        selected = group["primary_class"].astype(str).str.startswith("loop_p_")
        records.append(
            {
                "period": period,
                "dimension": dimension,
                "group": str(value),
                "total_decisions": len(group),
                "primitive_loop_events": int(primitive.sum()),
                "selected_dictionary_events": int(selected.sum()),
                "selected_event_coverage": float(selected.sum() / max(int(primitive.sum()), 1)),
                "other_primitive_share": float(
                    group["primary_class"].eq("OTHER_PRIMITIVE_LOOP").sum()
                    / max(int(primitive.sum()), 1)
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _stock_deletion_coverage(target: pd.DataFrame, *, period: str) -> pd.DataFrame:
    selected = target["primary_class"].astype(str).str.startswith("loop_p_")
    ranked = target.loc[selected, "symbol"].value_counts().index.tolist()
    records = []
    for removed in (0, 1, 5):
        retained = target.loc[~target["symbol"].isin(ranked[:removed])]
        primitive = retained["primitive_loop_id"].notna()
        selected_retained = retained["primary_class"].astype(str).str.startswith("loop_p_")
        records.append(
            {
                "period": period,
                "stocks_removed": removed,
                "removed_symbols": ranked[:removed],
                "primitive_loop_events": int(primitive.sum()),
                "selected_dictionary_events": int(selected_retained.sum()),
                "coverage": float(selected_retained.sum() / max(int(primitive.sum()), 1)),
            }
        )
    return pd.DataFrame.from_records(records)


def _family_outputs(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    development_draws: np.ndarray,
    development_clock_draws: np.ndarray,
    validation_draws: np.ndarray,
    validation_clock_draws: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dev_family = build_loop_family_mapping(development).assign(period="development_2024")
    val_family = build_loop_family_mapping(validation).assign(
        period="unchanged_retrospective_validation_2025"
    )
    mapping = pd.concat([dev_family, val_family], ignore_index=True)
    counts = (
        mapping.groupby(["period", "loop_family"], dropna=False)
        .agg(
            decisions=("decision_id", "size"),
            exact_ids=("primitive_loop_id", "nunique"),
        )
        .reset_index()
    )
    counts["share"] = counts["decisions"] / counts.groupby("period")["decisions"].transform("sum")
    exact_vs_family_rows = []
    stability_rows = []
    for period, group in mapping.groupby("period", sort=True):
        primitive = group["primitive_loop_id"].notna()
        exact_selected = group["primary_class"].astype(str).str.startswith("loop_p_")
        exact_counts = group.loc[primitive, "primitive_loop_id"].value_counts()
        probabilities = exact_counts.to_numpy(dtype=float) / max(float(exact_counts.sum()), 1.0)
        family_counts = group.loc[primitive, "loop_family"].value_counts()
        family_probabilities = family_counts.to_numpy(dtype=float) / max(
            float(family_counts.sum()), 1.0
        )
        other = group.loc[group["primary_class"].eq("OTHER_PRIMITIVE_LOOP")]
        exact_vs_family_rows.append(
            {
                "period": period,
                "exact_selected_coverage": float(
                    exact_selected.sum() / max(int(primitive.sum()), 1)
                ),
                "family_resolved_coverage": float(
                    group.loc[primitive, "loop_family"].notna().sum() / max(int(primitive.sum()), 1)
                ),
                "other_events_resolved_by_family_share": float(
                    other["loop_family"].notna().mean() if len(other) else 0.0
                ),
                "exact_identity_entropy_nats": float(
                    -(probabilities * np.log(probabilities)).sum() if len(probabilities) else 0.0
                ),
                "family_entropy_nats": float(
                    -(family_probabilities * np.log(family_probabilities)).sum()
                    if len(family_probabilities)
                    else 0.0
                ),
            }
        )
        for family, family_group in group.loc[primitive].groupby("loop_family", sort=True):
            identities = family_group["primitive_loop_id"].value_counts()
            identity_probabilities = identities.to_numpy(dtype=float) / identities.sum()
            stability_rows.append(
                {
                    "period": period,
                    "loop_family": family,
                    "primitive_events": len(family_group),
                    "exact_id_count": len(identities),
                    "exact_id_entropy_within_family": float(
                        -(identity_probabilities * np.log(identity_probabilities)).sum()
                    ),
                }
            )
    exact_vs_family = pd.DataFrame.from_records(exact_vs_family_rows)
    stability = pd.DataFrame.from_records(stability_rows)
    family_null_rows = []
    for period, draws, clock_draws in (
        ("development_2024", development_draws, development_clock_draws),
        (
            "unchanged_retrospective_validation_2025",
            validation_draws,
            validation_clock_draws,
        ),
    ):
        observed = (
            mapping.loc[
                mapping["period"].eq(period) & mapping["primitive_loop_id"].notna(),
                "loop_family",
            ]
            .value_counts()
            .reindex(FIRST_EVENT_FAMILY_ORDER, fill_value=0)
            .to_numpy(dtype=int)
        )
        statistics = summarize_null_draws(observed, draws, clock_draws)
        statistics.insert(0, "loop_family", FIRST_EVENT_FAMILY_ORDER)
        statistics.insert(0, "period", period)
        family_null_rows.append(statistics)
    family_statistics = pd.concat(family_null_rows, ignore_index=True)
    stability = stability.merge(
        family_statistics,
        on=["period", "loop_family"],
        how="outer",
        validate="one_to_one",
    ).sort_values(["period", "loop_family"])
    return mapping, counts, exact_vs_family, stability


def _artifact_manifest(output_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_manifest.json", "independent_audit.json"}
    records = []
    for path in sorted(output_dir.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name in excluded:
            continue
        records.append(
            {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    return {"artifact_count": len(records), "artifacts": records}


def _filter_generated_git_status(status: str) -> str:
    """Remove this experiment's generated output paths from a status snapshot."""

    generated_paths = (
        "work/artifacts/20260718-semantic-loop-dictionary-coverage-v2/",
        "work/reports/20260718-semantic-loop-dictionary-coverage-v2.md",
    )
    return "\n".join(
        line for line in status.splitlines() if not any(path in line for path in generated_paths)
    )


def _non_generated_git_status() -> str:
    """Return worktree status excluding outputs created by this experiment itself."""

    return _filter_generated_git_status(_git("status", "--short"))


def _report(
    metrics: dict[str, Any],
    tie_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    dictionary: pd.DataFrame,
    development_coverage: pd.DataFrame,
    validation_coverage: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    headings = [
        "Exact scope",
        "Scientific status",
        "Source identity",
        "Audit input reproduction",
        "Previous V2 dictionary",
        "Original tie count",
        "Nested versus genuine tie decomposition",
        "Revised primary event precedence",
        "Unregistered-loop population",
        "Primitive/repeat/composite distribution",
        "Vocabulary concentration",
        "Event-length distribution",
        "Candidate universe",
        "Support and concentration gates",
        "Semi-Markov null",
        "Clock-conditioned null",
        "Analytical null",
        "Circular control",
        "Structural rate ratios",
        "Multiplicity",
        "Increment beyond state history",
        "Selected primary dictionary",
        "Auxiliary motif registry",
        "Dictionary selection path",
        "Dictionary hash",
        "Legacy mapping",
        "Revised first-event class counts",
        "Selected-dictionary development coverage",
        "Validation coverage",
        "OTHER-loop share",
        "Genuine tie share",
        "No-loop and session-end rates",
        "Coverage stability",
        "Loop structural-excess replication",
        "Prefix feature compression",
        "Family-level fallback",
        "Exact versus family target comparison",
        "Failure cases",
        "Missing evidence",
        "Scientific decision",
        "Exact next experiment",
    ]
    details = {
        "Exact scope": "Structural semantic identity, recurrence, information attribution, and unchanged retrospective coverage only. No predictor, payoff model, or strategy was built.",
        "Scientific status": "2024 is development; 2025 is unchanged retrospective structural validation, not untouched or prospective.",
        "Source identity": f"Implementation target `{metrics['git_sha']}` on `{metrics['branch']}`; frozen V2 data `{metrics['development_snapshot_hash']}`; validation snapshot `{metrics['validation_snapshot_hash']}`.",
        "Audit input reproduction": f"All {metrics['old_artifacts_verified']} declared artifacts per old tree and {metrics['frozen_blob_count']} frozen blobs verified before use.",
        "Previous V2 dictionary": "The audited mixed registry had 11 primitives, 8 repeats, and 1 composite. It remains byte frozen.",
        "Original tie count": str(metrics["old_ties"]),
        "Nested versus genuine tie decomposition": tie_summary.to_csv(index=False),
        "Revised primary event precedence": "Source unavailable; structural gap; earliest future causal closure; primitive-root reduction; genuine distinct primitive tie; no-loop; session-end.",
        "Unregistered-loop population": f"{metrics['old_unregistered']} old unregistered decisions were reconstructed without a length cap.",
        "Primitive/repeat/composite distribution": metrics["motif_distribution"],
        "Vocabulary concentration": concentration.head(8).to_csv(index=False),
        "Event-length distribution": "See `unregistered_length_distribution.csv`; no path after the earliest completion was used.",
        "Candidate universe": f"{metrics['candidate_count']} supported primary-length candidates entered null scoring.",
        "Support and concentration gates": "100 decisions, 50 stock-sessions, 10 stocks, 6 months, 3 broad phases, <=20% top-stock and <=30% top-month share were frozen before scoring.",
        "Semi-Markov null": f"{PRIMARY_DRAWS} all-session fixed-seed draws with corrected exact/censored duration support.",
        "Clock-conditioned null": f"{CLOCK_DRAWS} all-session draws using frozen opening/middle/late phases.",
        "Analytical null": "First-order oriented completion expectations are reported as a separate raw-event control.",
        "Circular control": "Whole state-duration blocks were deterministically rotated within sessions.",
        "Structural rate ratios": "See `semi_markov_null_results.parquet` and `dictionary_entry_replication.csv`.",
        "Multiplicity": "One-sided empirical p-values use plus-one correction; BH adjustment is deterministic.",
        "Increment beyond state history": f"{metrics['information_qualified_count']} candidates retained positive chronological log-loss and Brier increment.",
        "Selected primary dictionary": dictionary[
            ["semantic_loop_id", "selection_rank", "development_count"]
        ].to_csv(index=False),
        "Auxiliary motif registry": "Repeats and bounded frequent composites are metadata only and consume no primary slot.",
        "Dictionary selection path": "Forward selection stops at 32, below 0.50 percentage-point marginal development coverage, or exhaustion.",
        "Dictionary hash": f"`{metrics['dictionary_hash']}`",
        "Legacy mapping": "All 20 old semantic entries are mapped exactly once; irreducible composite aliases remain auxiliary.",
        "Revised first-event class counts": "See `first_event_class_counts.csv`.",
        "Selected-dictionary development coverage": development_coverage.to_csv(index=False),
        "Validation coverage": validation_coverage.to_csv(index=False),
        "OTHER-loop share": f"Development {metrics['development_other_share']:.4%}; validation {metrics['validation_other_share']:.4%}.",
        "Genuine tie share": f"Development {metrics['development_tie_share']:.4%}; validation {metrics['validation_tie_share']:.4%}.",
        "No-loop and session-end rates": f"Validation no-loop {metrics['validation_no_loop_share']:.4%}; session-end {metrics['validation_session_end_share']:.4%}.",
        "Coverage stability": f"Exact-ID event coverage changed from {metrics['development_coverage']:.4%} to {metrics['validation_coverage']:.4%}.",
        "Loop structural-excess replication": f"{metrics['rr_above_one_share']:.2%} of selected entries retained validation rate ratio above one.",
        "Prefix feature compression": f"{metrics['prefix_feature_rows']} causally timestamped rows; no future or economic fields.",
        "Family-level fallback": (
            "Topology is mutually exclusive by primitive transition length; repeat/new status "
            "remains auxiliary. Complete-family structural excess:\n"
            f"{metrics['family_structural_excess']}"
        ),
        "Exact versus family target comparison": "See `exact_vs_family_coverage.csv` and `family_stability.csv`.",
        "Failure cases": "Clock-dependent candidates, diffuse OTHER identities, and period-specific rate reversals remain explicit.",
        "Missing evidence": metrics["missing_evidence"],
        "Scientific decision": f"**{decision['decision_label']}** — {decision['decision_reason']}.",
        "Exact next experiment": (
            "A separately preregistered structural forecast of the mutually exclusive first primitive-loop event and arrival time using simple frequency, state-history, active-prefix, and duration-aware competing-risk baselines, with no payoff or economic target. This task does not implement it."
            if decision["next_loop_predictor_justified"]
            else "No forecasting experiment is justified by this result."
        ),
    }
    lines = ["# Semantic Loop Dictionary and First-Event Coverage V2", ""]
    for index, heading in enumerate(headings, start=1):
        lines.extend([f"## {index}. {heading}", "", str(details[heading]).strip(), ""])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _corrected_duration_hazard() -> np.ndarray:
    with np.load(OLD_PRIMARY / "duration_model_v2.npz") as model:
        return np.asarray(model["hazard"], dtype=float).copy()


def _prepare_run(output_dir: Path) -> PreflightContext:
    print("phase=preflight start", flush=True)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_hash = _sha256_file(CONTRACT_PATH)
    preflight = _preflight(contract)
    git_sha = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    expected_sha = contract["source_identity"]["implementation_target_git_sha"]
    if (
        git_sha != expected_sha
        or branch != contract["source_identity"]["implementation_target_branch"]
    ):
        raise RuntimeError("implementation target SHA or branch differs from frozen contract")
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_dir = ARTIFACT_PARENT / "primary"
    static_files = (
        "implementation_delta_census.csv",
        "source_identity_manifest.json",
        "pre_run_tree_manifest.json",
        "audited_input_reconstruction.json",
    )
    if output_dir != primary_dir:
        for name in static_files:
            shutil.copy2(primary_dir / name, output_dir / name)
    context = PreflightContext(
        output_dir=output_dir,
        primary_dir=primary_dir,
        contract=contract,
        contract_hash=contract_hash,
        preflight=preflight,
        git_sha=git_sha,
        branch=branch,
        development_snapshot_hash=str(contract["source_identity"]["audited_v2_data_snapshot_hash"]),
        validation_snapshot_hash=_validation_snapshot_hash(),
    )
    print("phase=preflight pass", flush=True)
    return context


def _reconstruct_development(context: PreflightContext) -> DevelopmentReconstruction:
    decisions = _development_decisions()
    old_dictionary = pd.read_csv(OLD_PRIMARY / "semantic_loop_dictionary_v2.csv")
    old_primitive_ids = tuple(
        old_dictionary["primitive_loop_id"].dropna().astype(str).unique().tolist()
    )
    first_events = reconstruct_first_events(
        decisions,
        horizon_bars=HORIZON_BARS,
        legacy_dictionary_ids=old_primitive_ids,
        current_dictionary_ids=old_primitive_ids,
    )
    first_events = first_events.merge(
        decisions[["decision_id", "clock_phase", "hard_state_legacy"]],
        on="decision_id",
        how="left",
        validate="one_to_one",
    )
    first_events = _attach_legacy_overlapping_labels(first_events)
    old_outcomes = pd.read_parquet(
        OLD_PRIMARY / "first_next_loop_outcomes.parquet",
        columns=["decision_id", "primary_label"],
    )
    old_unregistered_ids = frozenset(
        old_outcomes.loc[
            old_outcomes["primary_label"].eq("UNREGISTERED_LOOP"), "decision_id"
        ].astype(str)
    )
    del old_outcomes
    gc.collect()
    print("phase=uncapped_reconstruction pass", flush=True)
    return DevelopmentReconstruction(
        decisions=decisions,
        first_events=first_events,
        old_dictionary=old_dictionary,
        old_primitive_ids=old_primitive_ids,
        old_unregistered_ids=old_unregistered_ids,
    )


def _score_and_select_dictionary(
    context: PreflightContext,
    reconstruction: DevelopmentReconstruction,
) -> DictionaryPhase:
    gates = CandidateSupportGates()
    candidate_bundle = build_candidate_universe(reconstruction.first_events, gates=gates)
    print("phase=candidate_support pass", flush=True)
    sessions = _development_sessions()
    print("phase=development_sessions pass", flush=True)
    (
        null_results,
        clock_results,
        development_draws,
        development_clock_draws,
        development_family_draws,
        development_family_clock_draws,
        development_deletion_details,
    ) = _score_nulls(
        reconstruction.first_events,
        reconstruction.decisions,
        candidate_bundle.support,
        sessions,
        draws=PRIMARY_DRAWS,
        clock_draws=CLOCK_DRAWS,
        seed=NULL_SEED,
    )
    print("phase=development_simulation pass", flush=True)
    analytical = _analytical_results(
        sessions,
        candidate_bundle.support.loc[
            candidate_bundle.support["support_pass"].astype(bool)
        ].reset_index(drop=True),
        reconstruction.first_events,
    )
    print("phase=analytical_control pass", flush=True)
    circular = _circular_results(sessions, null_results["semantic_loop_id"].astype(str).tolist())
    print("phase=development_nulls pass", flush=True)

    information_input = _information_decisions(reconstruction.first_events)
    information = information_increment_from_chronological_folds(
        information_input,
        candidate_ids=null_results["semantic_loop_id"].astype(str).tolist(),
    )
    information["information_qualified"] = information["information_qualified"].astype(
        bool
    ) & information["quarter_consistency"].ge(0.75)
    print("phase=information_increment pass", flush=True)
    scored = null_results.merge(
        information,
        on=["semantic_loop_id", "primitive_loop_id"],
        how="left",
        validate="one_to_one",
    )
    valid_primitive_events = int(reconstruction.first_events["primitive_loop_id"].notna().sum())
    selection = select_primary_dictionary(
        scored,
        total_valid_primitive_events=valid_primitive_events,
        maximum_entries=32,
        minimum_marginal_coverage=0.005,
    )
    dictionary = selection.dictionary.copy()
    if dictionary.empty:
        raise RuntimeError("no primitive loop passed the frozen dictionary selection")
    dictionary["dictionary_version"] = DICTIONARY_VERSION
    dictionary["eligible_decisions"] = int(
        reconstruction.decisions["structural_event_eligibility"].sum()
    )
    dictionary["structural_rate"] = dictionary["observed_rate"]
    dictionary["information_increment"] = dictionary["oof_log_loss_increment"]
    dictionary["stocks"] = dictionary["stock_breadth"]
    dictionary["sessions"] = dictionary["session_breadth"]
    dictionary["months"] = dictionary["month_breadth"]
    dictionary["quarters"] = dictionary["quarter_breadth"]
    dictionary["complexity"] = dictionary["transition_length"]
    old_by_primitive = (
        reconstruction.old_dictionary.loc[
            reconstruction.old_dictionary["primitive_loop_id"].notna()
        ]
        .groupby("primitive_loop_id")["semantic_loop_id"]
        .agg(lambda values: sorted(values.astype(str)))
    )
    dictionary["legacy_cycle_ids"] = (
        dictionary["primitive_loop_id"]
        .map(old_by_primitive)
        .map(lambda value: value if isinstance(value, list) else [])
    )
    dictionary_hash = deterministic_dictionary_hash(dictionary)
    dictionary["dictionary_hash"] = dictionary_hash
    selected_ids = frozenset(dictionary["primitive_loop_id"].astype(str))

    run_id = hashlib.sha256(
        _canonical_bytes(
            {
                "git_sha": context.git_sha,
                "contract_hash": context.contract_hash,
                "development_snapshot_hash": context.development_snapshot_hash,
                "validation_snapshot_hash": context.validation_snapshot_hash,
                "dictionary_hash": dictionary_hash,
                "state_model_version": STATE_MODEL_VERSION,
            }
        )
    ).hexdigest()[:24]
    identity = ArtifactIdentity(
        run_id=run_id,
        git_sha=context.git_sha,
        contract_hash=context.contract_hash,
        data_snapshot_hash=context.development_snapshot_hash,
        dictionary_version=DICTIONARY_VERSION,
        dictionary_hash=dictionary_hash,
        state_model_version=STATE_MODEL_VERSION,
    )
    run_context = RunContext(
        preflight=context,
        identity=identity,
        validation_identity=identity.for_snapshot(context.validation_snapshot_hash),
    )
    print(f"phase=dictionary_selection pass entries={len(dictionary)}", flush=True)
    return DictionaryPhase(
        context=run_context,
        candidate_bundle=candidate_bundle,
        sessions=sessions,
        null_results=null_results,
        clock_results=clock_results,
        development_draws=development_draws,
        development_clock_draws=development_clock_draws,
        development_family_draws=development_family_draws,
        development_family_clock_draws=development_family_clock_draws,
        development_deletion_details=development_deletion_details,
        analytical=analytical,
        circular=circular,
        information=information,
        scored=scored,
        selection=selection,
        dictionary=dictionary,
        dictionary_hash=dictionary_hash,
        selected_ids=selected_ids,
    )


def _build_development_artifacts(
    reconstruction: DevelopmentReconstruction,
    dictionary_phase: DictionaryPhase,
) -> DevelopmentArtifacts:
    writer = dictionary_phase.context.writer
    output_dir = dictionary_phase.context.preflight.output_dir
    dictionary = dictionary_phase.dictionary
    selected_ids = dictionary_phase.selected_ids
    first_events = reconstruction.first_events
    decisions = reconstruction.decisions

    auxiliary_registry = _auxiliary_registry(first_events, dictionary)
    legacy_mapping = _legacy_mapping(set(selected_ids))
    target_bundle = build_first_event_target(
        first_events,
        selected_primitive_ids=selected_ids,
        horizon_bars=HORIZON_BARS,
        copy_outcomes=False,
    )
    target_contract = dict(target_bundle.target_contract)
    development_target = target_bundle.outcomes
    development_target["month"] = pd.to_datetime(
        development_target["decision_timestamp"], utc=True
    ).dt.strftime("%Y-%m")
    development_target["transition_length_group"] = development_target[
        "primitive_transition_length"
    ].fillna(-1)

    class_counts = (
        development_target.groupby("primary_class", dropna=False)
        .agg(decisions=("decision_id", "size"))
        .reset_index()
    )
    class_counts["share"] = class_counts["decisions"] / len(development_target)
    legacy_comparison = pd.read_parquet(
        OLD_PRIMARY / "first_next_loop_outcomes.parquet",
        columns=["decision_id", "primary_label"],
    )
    legacy_comparison = legacy_comparison.merge(
        development_target[["decision_id", "primary_class"]],
        on="decision_id",
        validate="one_to_one",
    )
    legacy_comparison_summary = (
        legacy_comparison.groupby(["primary_label", "primary_class"], dropna=False)
        .size()
        .rename("decisions")
        .reset_index()
    )
    del legacy_comparison

    prefix_input = _prefix_decisions(development_target)
    prefix_bundle = build_compressed_prefix_features(
        prefix_input,
        primary_dictionary=dictionary,
        auxiliary_registry=auxiliary_registry,
        duration_hazard=_corrected_duration_hazard(),
        include_full_prefixes=False,
    )
    prefix_features = prefix_bundle.features
    prefix_manifest = prefix_bundle.manifest.copy()
    feature_names = prefix_manifest["feature_name"].tolist()
    prefix_missingness = pd.DataFrame(
        {
            "feature_name": feature_names,
            "missing_count": [int(prefix_features[name].isna().sum()) for name in feature_names],
            "missing_share": [float(prefix_features[name].isna().mean()) for name in feature_names],
        }
    )
    numeric_prefix = prefix_features[feature_names].select_dtypes(include=[np.number])
    prefix_distribution = numeric_prefix.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).T
    prefix_distribution.index.name = "feature_name"
    prefix_distribution = prefix_distribution.reset_index()
    prefix_feature_rows = len(prefix_features)

    development_light_columns = [
        "decision_id",
        "symbol",
        "session",
        "decision_timestamp",
        "primary_class",
        "primitive_loop_id",
        "primitive_transition_length",
        "current_repeat_depth",
        "repeat_depth",
        "clock_phase",
        "hard_state_legacy",
        "month",
        "transition_length_group",
    ]
    development_light = development_target[development_light_columns].copy()
    writer.write_frame(
        output_dir / "first_event_outcome_ledger_v2.parquet",
        development_target,
        source_artifact="frozen semantic dictionary",
        copy_frame=False,
    )
    writer.write_frame(
        output_dir / "first_event_tie_details.parquet",
        target_bundle.tie_details,
        source_artifact="first_event_outcome_ledger_v2.parquet",
    )
    writer.write_frame(
        output_dir / "first_event_auxiliary_targets.parquet",
        target_bundle.auxiliary,
        source_artifact="first_event_outcome_ledger_v2.parquet",
        copy_frame=False,
    )
    writer.write_frame(
        output_dir / "compressed_active_prefix_features.parquet",
        prefix_features,
        source_artifact="causal decision and semantic prefix history",
        copy_frame=False,
    )
    print("phase=development_detailed_ledgers pass", flush=True)
    del development_target, prefix_input, prefix_features, numeric_prefix
    gc.collect()

    tie_bundle = _resolve_old_ties()
    if len(tie_bundle.classification) != dictionary_phase.context.preflight.preflight["old_ties"]:
        raise RuntimeError("tie reconstruction did not reproduce the audited population")
    print("phase=tie_resolution pass", flush=True)

    unregistered = reconstruct_first_events(
        decisions,
        horizon_bars=HORIZON_BARS,
        legacy_dictionary_ids=reconstruction.old_primitive_ids,
        current_dictionary_ids=reconstruction.old_primitive_ids,
        decision_ids=reconstruction.old_unregistered_ids,
    )
    unregistered = unregistered.merge(
        decisions[["decision_id", "clock_phase", "hard_state_legacy"]],
        on="decision_id",
        how="left",
        validate="one_to_one",
    )
    if (
        len(unregistered) != dictionary_phase.context.preflight.preflight["old_unregistered"]
        or unregistered["primitive_loop_id"].isna().any()
    ):
        raise RuntimeError("old unregistered first events could not be reconstructed exactly")
    vocabulary = summarize_unregistered_vocabulary(unregistered)
    unregistered_unique_primitives = int(unregistered["primitive_loop_id"].nunique())
    motif_distribution = {
        str(key): int(value)
        for key, value in unregistered["motif_type"].value_counts().to_dict().items()
    }
    writer.write_frame(
        output_dir / "unregistered_event_ledger.parquet",
        unregistered,
        source_artifact="uncapped 2024 reconstruction",
        copy_frame=False,
    )
    del unregistered
    gc.collect()
    print("phase=unregistered_vocabulary pass", flush=True)

    return DevelopmentArtifacts(
        auxiliary_registry=auxiliary_registry,
        legacy_mapping=legacy_mapping,
        target_contract=target_contract,
        development_light=development_light,
        class_counts=class_counts,
        legacy_comparison_summary=legacy_comparison_summary,
        prefix_manifest=prefix_manifest,
        prefix_missingness=prefix_missingness,
        prefix_distribution=prefix_distribution,
        prefix_feature_rows=prefix_feature_rows,
        tie_bundle=tie_bundle,
        vocabulary=vocabulary,
        unregistered_unique_primitives=unregistered_unique_primitives,
        motif_distribution=motif_distribution,
        development_decision_count=len(decisions),
    )


def _score_unchanged_validation(
    dictionary_phase: DictionaryPhase,
    development: DevelopmentArtifacts,
) -> ValidationPhase:
    context = dictionary_phase.context
    validation_decisions, confirmed_validation_hash, validation_sessions = _validation_surface(
        context.preflight.contract_hash,
        context.identity,
    )
    if confirmed_validation_hash != context.preflight.validation_snapshot_hash:
        raise RuntimeError("validation source identity changed between pinning and scoring")
    print("phase=validation_surface pass", flush=True)
    validation_events = reconstruct_first_events(
        validation_decisions,
        horizon_bars=HORIZON_BARS,
        current_dictionary_ids=dictionary_phase.selected_ids,
    )
    validation_events = validation_events.merge(
        validation_decisions[["decision_id", "clock_phase", "hard_state_legacy"]],
        on="decision_id",
        how="left",
        validate="one_to_one",
    )
    validation_bundle = build_first_event_target(
        validation_events,
        selected_primitive_ids=dictionary_phase.selected_ids,
        horizon_bars=HORIZON_BARS,
        copy_outcomes=False,
    )
    validation_target = validation_bundle.outcomes
    validation_target["month"] = pd.to_datetime(
        validation_target["decision_timestamp"], utc=True
    ).dt.strftime("%Y-%m")
    validation_target["transition_length_group"] = validation_target[
        "primitive_transition_length"
    ].fillna(-1)
    validation_support = dictionary_phase.candidate_bundle.support.loc[
        dictionary_phase.candidate_bundle.support["semantic_loop_id"].isin(
            dictionary_phase.selected_ids
        )
    ].copy()
    validation_support["support_pass"] = True
    (
        validation_null,
        validation_clock,
        validation_draws,
        validation_clock_draws,
        validation_family_draws,
        validation_family_clock_draws,
        validation_deletion_details,
    ) = _score_nulls(
        validation_events,
        validation_decisions,
        validation_support,
        validation_sessions,
        draws=PRIMARY_DRAWS,
        clock_draws=CLOCK_DRAWS,
        seed=NULL_SEED + 10_000,
    )
    print("phase=validation_nulls pass", flush=True)
    validation_light = validation_target[development.development_light.columns].copy()
    context.validation_writer.write_frame(
        context.preflight.output_dir / "validation_first_event_outcome_ledger_v2.parquet",
        validation_target,
        source_artifact="bounded 2025 hard-state surface",
        copy_frame=False,
    )
    del validation_bundle, validation_events, validation_target
    gc.collect()

    replication = _dictionary_replication(
        dictionary_phase.dictionary,
        dictionary_phase.null_results,
        validation_null,
    )
    replication["entry_survived"] = replication["observed_count_validation"].gt(0)
    replication["validation_rr_above_one"] = replication["semi_markov_rate_ratio_validation"].gt(
        1.0
    )
    replication["validation_threshold_or_directional"] = replication[
        "semi_markov_rate_ratio_validation"
    ].gt(1.0)
    frequency_correlation = float(
        spearmanr(
            replication["development_count"], replication["observed_count_validation"]
        ).statistic
    )
    excess_correlation = float(
        spearmanr(
            replication["excess_count_development"],
            replication.set_index("semantic_loop_id").loc[
                dictionary_phase.dictionary["semantic_loop_id"],
                "excess_count_validation",
            ],
        ).statistic
    )
    rank_stability = replication[
        [
            "semantic_loop_id",
            "development_count",
            "observed_count_validation",
            "excess_count_development",
            "excess_count_validation",
        ]
    ].copy()
    rank_stability["frequency_rank_correlation"] = frequency_correlation
    rank_stability["structural_excess_rank_correlation"] = excess_correlation
    rank_stability["primitive_semantic_id_stable"] = True

    development_coverage = _coverage_summary(
        development.development_light, period="development_2024"
    )
    validation_coverage = _coverage_summary(
        validation_light, period="unchanged_retrospective_validation_2025"
    )
    coverage_by_stock = pd.concat(
        [
            _coverage_by(development.development_light, "symbol", period="development_2024"),
            _coverage_by(
                validation_light,
                "symbol",
                period="unchanged_retrospective_validation_2025",
            ),
        ],
        ignore_index=True,
    )
    coverage_by_month = pd.concat(
        [
            _coverage_by(development.development_light, "month", period="development_2024"),
            _coverage_by(
                validation_light,
                "month",
                period="unchanged_retrospective_validation_2025",
            ),
        ],
        ignore_index=True,
    )
    coverage_by_clock = pd.concat(
        [
            _coverage_by(development.development_light, "clock_phase", period="development_2024"),
            _coverage_by(
                validation_light,
                "clock_phase",
                period="unchanged_retrospective_validation_2025",
            ),
        ],
        ignore_index=True,
    )
    coverage_by_state = pd.concat(
        [
            _coverage_by(
                development.development_light,
                "hard_state_legacy",
                period="development_2024",
            ),
            _coverage_by(
                validation_light,
                "hard_state_legacy",
                period="unchanged_retrospective_validation_2025",
            ),
            _coverage_by(
                development.development_light,
                "transition_length_group",
                period="development_2024_transition_length",
            ),
            _coverage_by(
                validation_light,
                "transition_length_group",
                period="validation_2025_transition_length",
            ),
        ],
        ignore_index=True,
    )
    stock_deletions = pd.concat(
        [
            _stock_deletion_coverage(development.development_light, period="development_2024"),
            _stock_deletion_coverage(
                validation_light,
                period="unchanged_retrospective_validation_2025",
            ),
        ],
        ignore_index=True,
    )
    print("phase=prefix_compression pass", flush=True)

    family_mapping, family_counts, exact_vs_family, family_stability = _family_outputs(
        development.development_light,
        validation_light,
        dictionary_phase.development_family_draws,
        dictionary_phase.development_family_clock_draws,
        validation_family_draws,
        validation_family_clock_draws,
    )
    development_coverage_value = float(
        development_coverage.iloc[0]["selected_dictionary_event_coverage"]
    )
    validation_coverage_value = float(
        validation_coverage.iloc[0]["selected_dictionary_event_coverage"]
    )
    selected_validation = validation_light.loc[
        validation_light["primary_class"].astype(str).str.startswith("loop_p_")
    ]
    top_stock_share = float(
        selected_validation["symbol"].value_counts(normalize=True).iloc[0]
        if len(selected_validation)
        else 1.0
    )
    other_validation = validation_light.loc[
        validation_light["primary_class"].eq("OTHER_PRIMITIVE_LOOP")
    ]
    other_counts = other_validation["primitive_loop_id"].value_counts()
    other_probabilities = other_counts.to_numpy(dtype=float) / max(float(other_counts.sum()), 1.0)
    other_top_share = float(other_probabilities[0]) if len(other_probabilities) else 0.0
    other_hhi = float(np.square(other_probabilities).sum())
    exact_metrics = exact_vs_family.set_index("period")
    metrics_for_decision = {
        "dictionary_size": len(dictionary_phase.dictionary),
        "development_coverage": development_coverage_value,
        "validation_coverage": validation_coverage_value,
        "entries_rate_ratio_above_one_share": float(replication["validation_rr_above_one"].mean()),
        "entries_threshold_retained_share": float(
            replication["validation_threshold_or_directional"].mean()
        ),
        "top_stock_share": top_stock_share,
        "genuine_tie_rate": float(validation_coverage.iloc[0]["genuine_tie_share"]),
        "other_dominated_by_obvious_candidate": other_top_share > 0.20,
        "semantic_ids_stable": bool(rank_stability["primitive_semantic_id_stable"].all()),
        "exact_dictionary_stable_and_informative": bool(
            dictionary_phase.information.loc[
                dictionary_phase.information["semantic_loop_id"].isin(
                    dictionary_phase.selected_ids
                ),
                "information_qualified",
            ].all()
        ),
        "other_is_diffuse": other_top_share < 0.20 and other_hhi < 0.10,
        "family_reduces_residual_entropy": float(
            exact_metrics.loc["unchanged_retrospective_validation_2025", "family_entropy_nats"]
        )
        < float(
            exact_metrics.loc[
                "unchanged_retrospective_validation_2025",
                "exact_identity_entropy_nats",
            ]
        ),
        "family_coverage_stable_and_higher": bool(
            exact_metrics["family_resolved_coverage"].min() >= 0.99
        ),
        "exact_excess_consistent": float(replication["validation_rr_above_one"].mean()) >= 0.50,
        "coverage_collapsed": validation_coverage_value < development_coverage_value - 0.10,
        "structural_excess_reversed": float(replication["validation_rr_above_one"].mean()) < 0.50,
        "blocked": False,
    }
    scientific_decision = decide_target_tractability(metrics_for_decision)
    return ValidationPhase(
        validation_decision_count=len(validation_decisions),
        validation_session_count=len(validation_sessions),
        validation_null=validation_null,
        validation_clock=validation_clock,
        validation_deletion_details=validation_deletion_details,
        replication=replication,
        rank_stability=rank_stability,
        development_coverage=development_coverage,
        validation_coverage=validation_coverage,
        coverage_by_stock=coverage_by_stock,
        coverage_by_month=coverage_by_month,
        coverage_by_clock=coverage_by_clock,
        coverage_by_state=coverage_by_state,
        stock_deletions=stock_deletions,
        family_mapping=family_mapping,
        family_counts=family_counts,
        exact_vs_family=exact_vs_family,
        family_stability=family_stability,
        scientific_decision=scientific_decision,
        development_coverage_value=development_coverage_value,
        validation_coverage_value=validation_coverage_value,
    )


def _publication_support_tables(
    dictionary_phase: DictionaryPhase,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    support_rejections = dictionary_phase.candidate_bundle.rejections.copy()
    scoring_rejections = dictionary_phase.scored.loc[
        ~dictionary_phase.scored["structurally_qualified"].astype(bool)
        | ~dictionary_phase.scored["information_qualified"].astype(bool),
        [
            "semantic_loop_id",
            "primitive_loop_id",
            "structural_rejection_reason",
            "information_qualified",
        ],
    ].copy()
    scoring_rejections["rejection_reason"] = np.where(
        ~scoring_rejections["information_qualified"].astype(bool),
        "information_increment_gate",
        scoring_rejections["structural_rejection_reason"],
    )
    rejection_reasons = pd.concat(
        [
            support_rejections,
            scoring_rejections[["semantic_loop_id", "primitive_loop_id", "rejection_reason"]],
        ],
        ignore_index=True,
    )
    information_calibration = dictionary_phase.information[
        [
            "semantic_loop_id",
            "candidate_event_prevalence",
            "calibration_error",
            "quarter_consistency",
            "stock_deletion_consistency",
        ]
    ].copy()
    history_baselines = dictionary_phase.information.melt(
        id_vars=["semantic_loop_id"],
        value_vars=[f"b{index}_log_loss" for index in range(6)],
        var_name="baseline",
        value_name="log_loss",
    )
    blockers = pd.DataFrame(
        [
            {
                "evidence": "2025 corrected posterior surface",
                "status": "not_required_for_hard-state structural validation",
                "affected_rows": 0,
                "blocks_experiment": False,
            },
            {
                "evidence": "2026_or_later_rows",
                "status": "protected_not_read",
                "affected_rows": 0,
                "blocks_experiment": False,
            },
            {
                "evidence": "economic_outcomes",
                "status": "forbidden_not_read_or_used",
                "affected_rows": 0,
                "blocks_experiment": False,
            },
            {
                "evidence": "frozen_scientific_decision_hierarchy",
                "status": "uncovered_stable_low_coverage_retained_excess_region",
                "affected_rows": 1,
                "blocks_experiment": True,
            },
        ]
    )
    return rejection_reasons, information_calibration, history_baselines, blockers


def _publish_tabular_artifacts(
    dictionary_phase: DictionaryPhase,
    development: DevelopmentArtifacts,
    validation: ValidationPhase,
) -> None:
    context = dictionary_phase.context
    output_dir = context.preflight.output_dir
    writer = context.writer
    validation_writer = context.validation_writer
    (
        rejection_reasons,
        information_calibration,
        history_baselines,
        blockers,
    ) = _publication_support_tables(dictionary_phase)

    census = pd.read_csv(output_dir / "implementation_delta_census.csv")
    writer.write_frame(
        output_dir / "implementation_delta_census.csv",
        census,
        source_artifact="pre_edit_implementation_delta_census",
    )
    frame_artifacts: list[tuple[str, pd.DataFrame, str]] = [
        (
            "tie_classification.parquet",
            development.tie_bundle.classification,
            "frozen V2 completion ledger",
        ),
        (
            "tie_classification_summary.csv",
            development.tie_bundle.summary,
            "tie_classification.parquet",
        ),
        (
            "nested_completion_mapping.parquet",
            development.tie_bundle.nested_mapping,
            "frozen V2 completion ledger",
        ),
        (
            "tie_primary_label_rewrite.csv",
            development.tie_bundle.primary_rewrite,
            "tie_classification.parquet",
        ),
        (
            "unregistered_primitive_census.parquet",
            development.vocabulary.primitive_census,
            "unregistered_event_ledger.parquet",
        ),
        (
            "unregistered_repeat_census.csv",
            development.vocabulary.repeat_census,
            "unregistered_event_ledger.parquet",
        ),
        (
            "unregistered_composite_census.csv",
            development.vocabulary.composite_census,
            "unregistered_event_ledger.parquet",
        ),
        (
            "unregistered_length_distribution.csv",
            development.vocabulary.length_distribution,
            "unregistered_event_ledger.parquet",
        ),
        (
            "unregistered_vocabulary_concentration.csv",
            development.vocabulary.concentration,
            "unregistered_event_ledger.parquet",
        ),
        (
            "unregistered_tie_census.csv",
            development.vocabulary.tie_census,
            "unregistered_event_ledger.parquet",
        ),
        (
            "candidate_loop_universe.parquet",
            dictionary_phase.candidate_bundle.universe,
            "development first-event ledger",
        ),
        (
            "candidate_support.csv",
            dictionary_phase.candidate_bundle.support,
            "candidate_loop_universe.parquet",
        ),
        (
            "candidate_rejection_reasons.csv",
            rejection_reasons,
            "candidate scoring gates",
        ),
        (
            "semi_markov_null_results.parquet",
            dictionary_phase.null_results,
            "corrected duration and all development sessions",
        ),
        (
            "leave_one_stock_out_null_results.parquet",
            dictionary_phase.development_deletion_details,
            "stock-resolved development semi-Markov draws",
        ),
        (
            "clock_null_results.parquet",
            dictionary_phase.clock_results,
            "broad clock-conditioned development null",
        ),
        (
            "analytical_null_results.csv",
            dictionary_phase.analytical,
            "first-order development state process",
        ),
        (
            "circular_null_results.csv",
            dictionary_phase.circular,
            "whole-session circular control",
        ),
        (
            "candidate_structural_qualification.csv",
            dictionary_phase.scored,
            "null and support results",
        ),
        (
            "candidate_information_increment.csv",
            dictionary_phase.information,
            "chronological structural folds",
        ),
        (
            "candidate_information_calibration.csv",
            information_calibration,
            "chronological structural folds",
        ),
        (
            "candidate_history_baseline_comparison.csv",
            history_baselines,
            "chronological structural folds",
        ),
        (
            "semantic_loop_dictionary_v2.csv",
            dictionary_phase.dictionary,
            "development-only forward selection",
        ),
        (
            "auxiliary_loop_motif_registry_v2.csv",
            development.auxiliary_registry,
            "development structural motifs",
        ),
        (
            "dictionary_selection_path.csv",
            dictionary_phase.selection.selection_path,
            "development-only forward selection",
        ),
        (
            "legacy_to_dictionary_v2_mapping.csv",
            development.legacy_mapping,
            "frozen V2 dictionary",
        ),
        (
            "first_event_class_counts.csv",
            development.class_counts,
            "first_event_outcome_ledger_v2.parquet",
        ),
        (
            "legacy_v2_dictionary_target_comparison.csv",
            development.legacy_comparison_summary,
            "old and new target ledgers",
        ),
        (
            "prefix_feature_missingness.csv",
            development.prefix_missingness,
            "compressed_active_prefix_features.parquet",
        ),
        (
            "prefix_feature_distribution.csv",
            development.prefix_distribution,
            "compressed_active_prefix_features.parquet",
        ),
        (
            "development_coverage.csv",
            validation.development_coverage,
            "development target ledger",
        ),
        (
            "validation_coverage.csv",
            validation.validation_coverage,
            "unchanged retrospective validation ledger",
        ),
        ("coverage_by_stock.csv", validation.coverage_by_stock, "period target ledgers"),
        ("coverage_by_month.csv", validation.coverage_by_month, "period target ledgers"),
        ("coverage_by_clock.csv", validation.coverage_by_clock, "period target ledgers"),
        ("coverage_by_state.csv", validation.coverage_by_state, "period target ledgers"),
        (
            "coverage_stock_deletions.csv",
            validation.stock_deletions,
            "period target ledgers",
        ),
        (
            "dictionary_entry_replication.csv",
            validation.replication,
            "development and validation nulls",
        ),
        (
            "structural_rank_stability.csv",
            validation.rank_stability,
            "development and validation nulls",
        ),
        (
            "loop_family_mapping.parquet",
            validation.family_mapping,
            "period target ledgers",
        ),
        (
            "loop_family_counts.csv",
            validation.family_counts,
            "loop_family_mapping.parquet",
        ),
        (
            "exact_vs_family_coverage.csv",
            validation.exact_vs_family,
            "loop_family_mapping.parquet",
        ),
        (
            "family_stability.csv",
            validation.family_stability,
            "period target ledgers and complete-family nulls",
        ),
        ("missingness_and_blockers.csv", blockers, "run evidence census"),
        (
            "validation_semi_markov_null_results.parquet",
            validation.validation_null,
            "bounded 2025 sessions",
        ),
        (
            "validation_clock_null_results.parquet",
            validation.validation_clock,
            "bounded 2025 sessions",
        ),
        (
            "validation_leave_one_stock_out_null_results.parquet",
            validation.validation_deletion_details,
            "stock-resolved bounded 2025 semi-Markov draws",
        ),
    ]
    mixed_period_artifacts = {
        "coverage_by_stock.csv",
        "coverage_by_month.csv",
        "coverage_by_clock.csv",
        "coverage_by_state.csv",
        "coverage_stock_deletions.csv",
        "loop_family_mapping.parquet",
        "loop_family_counts.csv",
        "exact_vs_family_coverage.csv",
        "family_stability.csv",
    }
    for name, frame, source in frame_artifacts:
        if name in mixed_period_artifacts:
            _write_mixed_period_frame(
                output_dir / name,
                frame,
                source_artifact=source,
                context=context,
            )
        else:
            target_writer = validation_writer if name.startswith("validation_") else writer
            target_writer.write_frame(
                output_dir / name,
                frame,
                source_artifact=source,
            )
    print("phase=tabular_artifacts pass", flush=True)


def _publish_json_report_and_manifests(
    dictionary_phase: DictionaryPhase,
    development: DevelopmentArtifacts,
    validation: ValidationPhase,
) -> None:
    context = dictionary_phase.context
    preflight = context.preflight
    output_dir = preflight.output_dir
    writer = context.writer
    writer.write_json(
        output_dir / "dictionary_hash.json",
        {
            "dictionary_hash": dictionary_phase.dictionary_hash,
            "entry_count": len(dictionary_phase.dictionary),
            "hash_scope": "semantic membership and paths; rank and metrics excluded",
        },
    )
    writer.write_json(
        output_dir / "first_event_target_contract.json",
        development.target_contract,
    )
    writer.write_json(
        output_dir / "compressed_prefix_feature_manifest.json",
        {
            "manifest_version": "compressed_semantic_prefix_features_v2",
            "fields": development.prefix_manifest.to_dict(orient="records"),
            "forbidden_future_fields": [
                "future_loop",
                "future_state",
                "bars_until_future_completion",
                "payoff",
                "return_after_decision",
                "route_outcome",
                "MFE",
                "MAE",
            ],
        },
    )
    writer.write_json(
        output_dir / "loop_family_taxonomy.json",
        {
            "taxonomy_version": "topology_only_loop_family_v2",
            "hierarchy": [
                "NO_LOOP_OR_SESSION_END",
                "DISTINCT_PRIMITIVE_TIE",
                *FIRST_EVENT_FAMILY_ORDER,
            ],
            "repeat_status_is_auxiliary": True,
        },
    )
    run_metadata = {
        "branch": preflight.branch,
        "development_snapshot_hash": preflight.development_snapshot_hash,
        "validation_snapshot_hash": preflight.validation_snapshot_hash,
        "development_decisions": development.development_decision_count,
        "validation_decisions": validation.validation_decision_count,
        "development_sessions": len(dictionary_phase.sessions),
        "validation_sessions": validation.validation_session_count,
        "old_ties": preflight.preflight["old_ties"],
        "old_unregistered": preflight.preflight["old_unregistered"],
        "unregistered_unique_primitives": development.unregistered_unique_primitives,
        "candidate_count": len(dictionary_phase.null_results),
        "structurally_qualified_candidate_count": int(
            dictionary_phase.null_results["structurally_qualified"].sum()
        ),
        "selected_dictionary_size": len(dictionary_phase.dictionary),
        "primary_null_draws": PRIMARY_DRAWS,
        "clock_null_draws": CLOCK_DRAWS,
        "family_null_scope": "all_first_primitive_events",
        "predictor_trained": False,
        "payoff_model_trained": False,
        "economic_columns_read_for_selection": False,
        "frozen_historical_files_unchanged": True,
        "preflight": preflight.preflight,
        "numba_version": __import__("numba").__version__,
    }
    writer.write_json(output_dir / "run_metadata.json", run_metadata)
    writer.write_json(output_dir / "decision.json", validation.scientific_decision)
    frozen_count_after, frozen_tree_after = _verify_frozen_tree(preflight.contract)
    writer.write_json(
        output_dir / "post_run_tree_manifest.json",
        {
            "frozen_historical_blob_count": frozen_count_after,
            "frozen_historical_tree_hash": frozen_tree_after,
            "frozen_historical_files_unchanged": True,
            "git_status_excluding_generated_outputs": _non_generated_git_status(),
        },
    )
    writer.write_json(output_dir / "artifact_manifest.json", _artifact_manifest(output_dir))

    report_metrics = {
        "git_sha": preflight.git_sha,
        "branch": preflight.branch,
        "development_snapshot_hash": preflight.development_snapshot_hash,
        "validation_snapshot_hash": preflight.validation_snapshot_hash,
        "old_artifacts_verified": preflight.preflight["declared_artifacts_verified_per_tree"],
        "frozen_blob_count": preflight.preflight["frozen_historical_blob_count"],
        "old_ties": preflight.preflight["old_ties"],
        "old_unregistered": preflight.preflight["old_unregistered"],
        "motif_distribution": json.dumps(development.motif_distribution, sort_keys=True),
        "candidate_count": len(dictionary_phase.null_results),
        "information_qualified_count": int(
            dictionary_phase.information["information_qualified"].sum()
        ),
        "dictionary_hash": dictionary_phase.dictionary_hash,
        "development_other_share": float(
            validation.development_coverage.iloc[0]["other_primitive_loop_share"]
        ),
        "validation_other_share": float(
            validation.validation_coverage.iloc[0]["other_primitive_loop_share"]
        ),
        "development_tie_share": float(
            validation.development_coverage.iloc[0]["genuine_tie_share"]
        ),
        "validation_tie_share": float(validation.validation_coverage.iloc[0]["genuine_tie_share"]),
        "validation_no_loop_share": float(validation.validation_coverage.iloc[0]["no_loop_share"]),
        "validation_session_end_share": float(
            validation.validation_coverage.iloc[0]["session_end_share"]
        ),
        "development_coverage": validation.development_coverage_value,
        "validation_coverage": validation.validation_coverage_value,
        "rr_above_one_share": float(validation.replication["validation_rr_above_one"].mean()),
        "prefix_feature_rows": development.prefix_feature_rows,
        "family_structural_excess": validation.family_stability[
            [
                "period",
                "loop_family",
                "observed_count",
                "semi_markov_null_mean",
                "semi_markov_rate_ratio",
                "semi_markov_q",
                "clock_null_rate_ratio",
                "clock_null_q",
            ]
        ].to_csv(index=False),
        "missing_evidence": (
            "The frozen decision hierarchy has no authorized label for stable low exact "
            "coverage with retained exact structural excess; forecasting is fail-closed."
        ),
    }
    if output_dir == preflight.primary_dir:
        _report(
            report_metrics,
            development.tie_bundle.summary,
            development.vocabulary.concentration,
            dictionary_phase.dictionary,
            validation.development_coverage,
            validation.validation_coverage,
            validation.scientific_decision,
        )


def run(output_dir: Path) -> None:
    preflight = _prepare_run(output_dir)
    reconstruction = _reconstruct_development(preflight)
    dictionary_phase = _score_and_select_dictionary(preflight, reconstruction)
    development = _build_development_artifacts(reconstruction, dictionary_phase)
    del reconstruction
    gc.collect()
    validation = _score_unchanged_validation(dictionary_phase, development)
    _publish_tabular_artifacts(dictionary_phase, development, validation)
    _publish_json_report_and_manifests(dictionary_phase, development, validation)
    print("phase=run complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output.resolve())


if __name__ == "__main__":
    main()
