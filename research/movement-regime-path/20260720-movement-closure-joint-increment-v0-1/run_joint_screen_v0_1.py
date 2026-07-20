#!/usr/bin/env python3
"""Run Movement x Closure-History Joint Increment V0.1."""

# ruff: noqa: E402 -- the repository source tree is added before local imports.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

_REPO_FOR_IMPORT = Path(__file__).resolve().parents[3]
_RESEARCH_PACKAGE = _REPO_FOR_IMPORT / "packages/stocker_research/src"
if str(_RESEARCH_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_PACKAGE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from stocker_research.movement_closure_joint_screen_v0_1 import (
    SAFETY_FLAGS,
    add_joint_probability_features,
    assert_compact_panel_has_no_forbidden_fields,
    assert_protected_date_boundary,
    assert_upstream_chronology,
    bootstrap_intervals,
    classify_joint_decision,
    evaluate_support,
    exact_active_pair_join,
    fit_fixed_logistic,
    joint_arm_passes,
    logit_probability,
    paired_loss_improvements,
    primary_arm_passes,
    probability_metrics,
    session_block_bootstrap_improvements,
    split_development_assessment,
    whole_session_shift,
    whole_session_shift_feasibility,
    with_equal_slate_weights,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
MOVEMENT_DIR = (
    REPO_ROOT
    / "research/movement-regime-path/20260720-movement-conditioned-regime-path-probability-chain-v0"
)
CLOSURE_ROOT = REPO_ROOT / "research/slrno-v2/20260714-regime-loop-handoff/work"
CLOSURE_DIR = CLOSURE_ROOT / "artifacts/20260720-immediate-pair-closure-history-v1"
MOVEMENT_PRIMARY = MOVEMENT_DIR / "artifacts/primary"
CLOSURE_PRIMARY = CLOSURE_DIR / "primary"
MAPPING_PATH = (
    CLOSURE_ROOT
    / "artifacts/20260719-right-censored-regime-refit-v2/primary/full_refit_semantic_mapping.csv"
)
CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"

START_SESSION = "2024-07-01"
END_EXCLUSIVE_SESSION = "2025-08-23"
STATE_MODEL_HASH = "4fc1a02dce9ac2311dabaeb4623a559d37286dfe58baffef53828cc7415a3425"
STATE_MODEL_ID = "regime_model_v2_full_right_censored_refit"
SEMANTIC_REPRESENTATION = "CAUSAL_HARD_SEMANTIC"
BOOTSTRAP_DRAWS = 500
NULL_DRAWS = 100

MOVEMENT_OOF_PATH = MOVEMENT_PRIMARY / "oof_2024_predictions.parquet"
MOVEMENT_ASSESSMENT_PATH = MOVEMENT_PRIMARY / "scored_2025_predictions.parquet"
MOVEMENT_PANEL_PATH = MOVEMENT_PRIMARY / "compact_decision_panel.parquet"
CLOSURE_POPULATION_PATH = CLOSURE_PRIMARY / "pair_closure_population.parquet"
CLOSURE_OOF_PATH = CLOSURE_PRIMARY / "development_oof_predictions.parquet"
CLOSURE_ASSESSMENT_PATH = CLOSURE_PRIMARY / "assessment_predictions.parquet"

INPUT_PATHS = {
    "movement_contract": MOVEMENT_DIR / "contract.json",
    "movement_compact_population": MOVEMENT_PANEL_PATH,
    "movement_2024_oof_predictions": MOVEMENT_OOF_PATH,
    "movement_2025_frozen_predictions": MOVEMENT_ASSESSMENT_PATH,
    "movement_source_manifest": MOVEMENT_PRIMARY / "source_manifest.json",
    "movement_fold_manifest": MOVEMENT_PRIMARY / "chronological_fold_manifest.json",
    "movement_model_coefficients": MOVEMENT_PRIMARY / "model_coefficients.json",
    "movement_independent_audit": MOVEMENT_PRIMARY / "independent_audit.json",
    "movement_boundary_audit": MOVEMENT_PRIMARY / "protected_boundary_audit.json",
    "closure_contract": CLOSURE_ROOT / "contracts/20260720-immediate-pair-closure-history-v1.json",
    "closure_pair_population": CLOSURE_POPULATION_PATH,
    "closure_2024_oof_predictions": CLOSURE_OOF_PATH,
    "closure_2025_frozen_predictions": CLOSURE_ASSESSMENT_PATH,
    "closure_source_manifest": CLOSURE_PRIMARY / "source_identity_manifest.json",
    "closure_run_metadata": CLOSURE_PRIMARY / "run_metadata.json",
    "closure_model_configuration": CLOSURE_PRIMARY / "model_effective_configuration.json",
    "closure_independent_audit": CLOSURE_PRIMARY / "independent_audit.json",
    "closure_censoring_summary": CLOSURE_PRIMARY / "censoring_summary.csv",
    "semantic_mapping": MAPPING_PATH,
}

A2_FEATURES = (
    "logit_p_close_m5",
    "pair_age_bars",
    "scheduled_bars_remaining",
    "decision_ordinal",
)
A3_FEATURES = (*A2_FEATURES, "logit_p_move", "log1p_predicted_absolute_movement_bps")
B0_FEATURES = (
    "logit_p_move",
    "pair_age_bars",
    "scheduled_bars_remaining",
    "decision_ordinal",
)
B1_FEATURES = (*B0_FEATURES, "logit_p_close_m5", "closure_history_increment")
C1_FEATURES = (
    "logit_p_move",
    "logit_p_close_m5",
    "closure_history_increment",
    "pair_age_bars",
    "scheduled_bars_remaining",
    "decision_ordinal",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"blocked_reproducibility_or_audit_failure: {path.name}")
    return cast(dict[str, Any], value)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _audit_check_passed(audit: dict[str, Any], name: str) -> bool:
    return any(
        item.get("check") == name and item.get("passed") is True
        for item in cast(list[dict[str, Any]], audit.get("checks", []))
    )


def _validate_upstream_evidence(mapping_hash: str) -> dict[str, Any]:
    """Reconcile hash-bound chronology and source-lineage evidence before reads."""

    movement_contract = _read_json(INPUT_PATHS["movement_contract"])
    movement_source = _read_json(INPUT_PATHS["movement_source_manifest"])
    movement_folds = _read_json(INPUT_PATHS["movement_fold_manifest"])
    movement_models = _read_json(INPUT_PATHS["movement_model_coefficients"])
    movement_audit = _read_json(INPUT_PATHS["movement_independent_audit"])
    movement_boundary = _read_json(INPUT_PATHS["movement_boundary_audit"])
    closure_source = _read_json(INPUT_PATHS["closure_source_manifest"])
    closure_run = _read_json(INPUT_PATHS["closure_run_metadata"])
    closure_models = _read_json(INPUT_PATHS["closure_model_configuration"])
    closure_audit = _read_json(INPUT_PATHS["closure_independent_audit"])
    closure_censoring = pd.read_csv(INPUT_PATHS["closure_censoring_summary"])

    fold_rows = [
        cast(dict[str, Any], row)
        for row in cast(list[dict[str, Any]], movement_folds.get("folds", []))
        if row.get("layer") == "movement"
    ]
    fold_by_month = {str(row["score_month"]): row for row in fold_rows}
    expected_months = {f"2024-{month:02d}" for month in range(7, 13)}
    if (
        set(fold_by_month) != expected_months
        or movement_folds.get("all_upstream_predictions_out_of_fold") is not True
        or int(movement_folds.get("in_sample_stacked_features", -1)) != 0
        or movement_audit.get("passed") is not True
        or not _audit_check_passed(movement_audit, "oof_and_stacking_chronology")
    ):
        raise RuntimeError("blocked_chronology_or_leakage_failure: movement evidence")
    for month, row in fold_by_month.items():
        trained = pd.Timestamp(str(row["trained_through"]), tz="UTC")
        score_start = pd.Timestamp(str(row["score_start"]), tz="UTC")
        if row.get("strictly_earlier") is not True or trained >= score_start:
            raise RuntimeError(f"blocked_chronology_or_leakage_failure: movement fold {month}")

    final_models = cast(dict[str, Any], movement_models.get("models", {}))
    p_move_model = cast(dict[str, Any], final_models.get("P1", {}))
    size_model = cast(dict[str, Any], final_models.get("P1_SIZE", {}))
    final_training_rows = int(p_move_model.get("training_rows", -1))
    if (
        final_training_rows <= 0
        or int(size_model.get("training_rows", -2)) != final_training_rows
        or movement_contract.get("chronology", {}).get("final_score_period")
        != "2025 sessions strictly before 2025-08-23"
        or movement_boundary.get("protected_rows_materialised") != 0
    ):
        raise RuntimeError("blocked_chronology_or_leakage_failure: movement final fit")

    required_closure_checks = {
        "expanding_fold_training_counts",
        "assessment_fit_uses_2024_count_only",
    }
    if (
        closure_audit.get("audit_passed") is not True
        or len(cast(list[Any], closure_audit.get("failed_checks", ["missing"]))) != 0
        or not all(_audit_check_passed(closure_audit, name) for name in required_closure_checks)
        or closure_run.get("state_model_hash") != STATE_MODEL_HASH
        or closure_models.get("sensitivity_representation") != SEMANTIC_REPRESENTATION
        or closure_source.get("provider") != "EODHD"
        or closure_source.get("dataset_identity")
        != "StockerLocal/source=eodhd/instrument_type=stock/timeframe=5m"
    ):
        raise RuntimeError("blocked_chronology_or_leakage_failure: closure evidence")
    contract_hash = _sha256(INPUT_PATHS["closure_contract"])
    if closure_run.get("contract_hash") != contract_hash:
        raise RuntimeError("blocked_reproducibility_or_audit_failure: closure contract hash")
    training_count_rows = closure_censoring.loc[
        closure_censoring["period"].astype(str).eq("DEVELOPMENT_2024")
        & closure_censoring["representation"].astype(str).eq(SEMANTIC_REPRESENTATION)
        & closure_censoring["target_available"].astype(str).str.lower().eq("true")
    ]
    if len(training_count_rows) != 1:
        raise RuntimeError("blocked_chronology_or_leakage_failure: closure training count")
    closure_training_rows = int(training_count_rows.iloc[0]["rows"])

    cohort = [str(value) for value in movement_source.get("decision_cohort", [])]
    if len(cohort) != 20 or len(set(cohort)) != 20:
        raise RuntimeError("blocked_join_semantics_failure: movement cohort identity")
    movement_provider = cast(dict[str, Any], movement_source.get("provider_sources", {}))
    closure_development = cast(dict[str, Any], closure_source.get("development", {}))
    closure_assessment = cast(dict[str, Any], closure_source.get("assessment", {}))
    closure_development_hashes = cast(dict[str, str], closure_development.get("source_hashes", {}))
    closure_assessment_hashes = cast(dict[str, str], closure_assessment.get("source_hashes", {}))
    if (
        movement_source.get("frozen_model_hash") != STATE_MODEL_HASH
        or movement_source.get("frozen_development_snapshot_hash")
        != closure_development.get("data_snapshot_hash")
        or movement_source.get("frozen_development_panel_hash")
        != closure_development.get("feature_table_hash")
    ):
        raise RuntimeError("blocked_join_semantics_failure: source snapshot lineage")
    logical_paths: dict[str, str] = {}
    bounded_assessment_hashes: dict[str, str] = {}
    for stock in cohort:
        source = cast(dict[str, Any], movement_provider.get(stock, {}))
        logical = str(source.get("logical_path", ""))
        expected_suffix = f"symbol={stock}/timeframe=5m/data.parquet"
        if (
            not logical.endswith(expected_suffix)
            or source.get("bounded_2024_hash") != closure_development_hashes.get(stock)
            or stock not in closure_assessment_hashes
        ):
            raise RuntimeError(f"blocked_join_semantics_failure: source lineage mismatch {stock}")
        logical_paths[stock] = logical
        bounded_assessment_hashes[stock] = str(source.get("bounded_safe_hash", ""))

    lineage_evidence = {
        "provider": "EODHD",
        "instrument_type": "stock",
        "timeframe": "5m",
        "cohort": cohort,
        "logical_paths": logical_paths,
        "state_model_hash": STATE_MODEL_HASH,
        "semantic_mapping_hash": mapping_hash,
        "shared_development_snapshot_hash": closure_development["data_snapshot_hash"],
        "shared_development_feature_table_hash": closure_development["feature_table_hash"],
        "shared_development_source_hashes": {
            stock: closure_development_hashes[stock] for stock in cohort
        },
        "movement_bounded_assessment_end_exclusive": END_EXCLUSIVE_SESSION,
        "movement_bounded_assessment_hashes": bounded_assessment_hashes,
        "closure_assessment_snapshot_hash": closure_assessment["data_snapshot_hash"],
        "closure_assessment_source_hashes": {
            stock: closure_assessment_hashes[stock] for stock in cohort
        },
    }
    lineage_id = f"joint-source-lineage-v0-1|{_canonical_hash(lineage_evidence)}"
    movement_chronology = {
        "fold_manifest_sha256": _sha256(INPUT_PATHS["movement_fold_manifest"]),
        "model_coefficients_sha256": _sha256(INPUT_PATHS["movement_model_coefficients"]),
        "independent_audit_sha256": _sha256(INPUT_PATHS["movement_independent_audit"]),
        "folds": fold_by_month,
        "final_training_rows": final_training_rows,
        "final_trained_through": "2024-12-31T23:59:59.999999999Z",
    }
    closure_chronology = {
        "run_metadata_sha256": _sha256(INPUT_PATHS["closure_run_metadata"]),
        "model_configuration_sha256": _sha256(INPUT_PATHS["closure_model_configuration"]),
        "independent_audit_sha256": _sha256(INPUT_PATHS["closure_independent_audit"]),
        "development_evaluation_period": "DEVELOPMENT_2024_OOF",
        "assessment_evaluation_period": "ASSESSMENT_2025",
        "assessment_training_rows": closure_training_rows,
        "final_trained_through": "2024-12-31T23:59:59.999999999Z",
    }
    return {
        "lineage_id": lineage_id,
        "lineage_evidence": lineage_evidence,
        "movement_chronology": movement_chronology,
        "movement_chronology_evidence_id": _canonical_hash(movement_chronology),
        "closure_chronology": closure_chronology,
        "closure_chronology_evidence_id": _canonical_hash(closure_chronology),
        "closure_source": closure_source,
    }


def _short_id(parts: Iterable[object]) -> str:
    payload = "|".join(str(value) for value in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _verify_required_inputs() -> dict[str, str]:
    missing = [name for name, path in INPUT_PATHS.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"blocked_missing_frozen_joint_inputs: {missing}")
    return {name: _sha256(path) for name, path in sorted(INPUT_PATHS.items())}


def _load_semantic_mapping() -> tuple[dict[int, int], str]:
    frame = pd.read_csv(MAPPING_PATH)
    mapping = {
        int(row.raw_cluster_state): int(row.semantic_state) for row in frame.itertuples(index=False)
    }
    if set(mapping) != set(range(8)) or set(mapping.values()) != set(range(8)):
        raise RuntimeError("blocked_join_semantics_failure: semantic mapping is not a permutation")
    return mapping, _sha256(MAPPING_PATH)


def _read_movement_surface(
    mapping: dict[int, int], evidence: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_columns = [
        "symbol",
        "session",
        "decision_ordinal",
        "feature_available_timestamp_utc",
        "origin_state",
        "origin_segment_id",
        "history_ordinals_contiguous",
        "future_ordinals_contiguous",
        "source_gap_crossed",
        "session_boundary_crossed",
        "session_source_complete",
        "scheduled_bars_remaining",
        "state_model_hash",
        "p_move",
        "predicted_absolute_movement_bps",
        "movement_threshold_bps",
        "large_move",
    ]
    development = pd.read_parquet(
        MOVEMENT_OOF_PATH,
        columns=[
            *base_columns,
            "p_move__trained_through",
            "predicted_absolute_movement_bps__trained_through",
        ],
        filters=[
            ("session", ">=", START_SESSION),
            ("session", "<", "2025-01-01"),
        ],
    )
    assessment = pd.read_parquet(
        MOVEMENT_ASSESSMENT_PATH,
        columns=base_columns,
        filters=[
            ("session", ">=", "2025-01-01"),
            ("session", "<", END_EXCLUSIVE_SESSION),
        ],
    )
    development["year"] = 2024
    assessment["year"] = 2025
    movement_chronology = cast(dict[str, Any], evidence["movement_chronology"])
    expected_folds = cast(dict[str, dict[str, Any]], movement_chronology["folds"])
    development_month = development["session"].astype(str).str[:7]
    if set(development_month.unique()) != set(expected_folds):
        raise RuntimeError("blocked_chronology_or_leakage_failure: movement score months")
    for month, fold in sorted(expected_folds.items()):
        rows = development.loc[development_month.eq(month)]
        expected_cutoff = pd.Timestamp(str(fold["trained_through"]), tz="UTC")
        if len(rows) != int(fold["scored_rows"]):
            raise RuntimeError(f"blocked_chronology_or_leakage_failure: movement row count {month}")
        for column in (
            "p_move__trained_through",
            "predicted_absolute_movement_bps__trained_through",
        ):
            cutoffs = pd.to_datetime(rows[column], utc=True, errors="raise")
            if not cutoffs.eq(expected_cutoff).all():
                raise RuntimeError(f"blocked_chronology_or_leakage_failure: {column} {month}")
    development["movement_oof"] = (
        development["p_move"].notna() & development["predicted_absolute_movement_bps"].notna()
    )
    assessment["movement_oof"] = False
    development["movement_trained_through"] = pd.to_datetime(
        development.pop("p_move__trained_through"), utc=True, errors="coerce"
    )
    development["movement_size_trained_through"] = pd.to_datetime(
        development.pop("predicted_absolute_movement_bps__trained_through"),
        utc=True,
        errors="coerce",
    )
    final_cutoff = pd.Timestamp(str(movement_chronology["final_trained_through"]))
    assessment["movement_trained_through"] = final_cutoff
    assessment["movement_size_trained_through"] = final_cutoff
    development["movement_frozen_before_outcome"] = False
    assessment["movement_frozen_before_outcome"] = True
    frame = pd.concat([development, assessment], ignore_index=True)
    if not frame["state_model_hash"].astype(str).eq(STATE_MODEL_HASH).all():
        raise RuntimeError("blocked_missing_frozen_joint_inputs: movement model hash differs")
    frame["stock"] = frame.pop("symbol").astype(str)
    frame["fixed_clock_timestamp"] = pd.to_datetime(
        frame.pop("feature_available_timestamp_utc"), utc=True, errors="raise"
    )
    frame["movement_horizon_terminal_timestamp"] = frame["fixed_clock_timestamp"] + pd.Timedelta(
        minutes=120
    )
    frame["current_state_b"] = frame["origin_state"].map(mapping)
    if frame["current_state_b"].isna().any():
        raise RuntimeError("blocked_join_semantics_failure: unmapped movement hard state")
    frame["current_state_b"] = frame["current_state_b"].astype(int)
    frame["representation_id"] = f"{STATE_MODEL_ID}|{STATE_MODEL_HASH}|{SEMANTIC_REPRESENTATION}"
    frame["source_lineage_id"] = str(evidence["lineage_id"])
    frame["movement_chronology_evidence_id"] = str(evidence["movement_chronology_evidence_id"])
    frame["source_gap"] = (
        frame["source_gap_crossed"].astype(bool)
        | frame["session_boundary_crossed"].astype(bool)
        | ~frame["history_ordinals_contiguous"].astype(bool)
        | ~frame["future_ordinals_contiguous"].astype(bool)
        | ~frame["session_source_complete"].astype(bool)
    )
    frame["movement_available"] = (
        frame["p_move"].notna()
        & frame["predicted_absolute_movement_bps"].notna()
        & frame["large_move"].notna()
        & ~frame["source_gap"]
    )
    frame["movement_row_id"] = [
        _short_id(
            (
                STATE_MODEL_HASH,
                stock,
                session,
                ordinal,
                timestamp.isoformat(),
            )
        )
        for stock, session, ordinal, timestamp in frame[
            ["stock", "session", "decision_ordinal", "fixed_clock_timestamp"]
        ].itertuples(index=False, name=None)
    ]
    keep = [
        "movement_row_id",
        "representation_id",
        "source_lineage_id",
        "stock",
        "session",
        "year",
        "decision_ordinal",
        "fixed_clock_timestamp",
        "movement_horizon_terminal_timestamp",
        "origin_segment_id",
        "current_state_b",
        "scheduled_bars_remaining",
        "p_move",
        "predicted_absolute_movement_bps",
        "movement_threshold_bps",
        "large_move",
        "movement_available",
        "source_gap",
        "movement_oof",
        "movement_trained_through",
        "movement_size_trained_through",
        "movement_frozen_before_outcome",
        "movement_chronology_evidence_id",
    ]
    output = (
        frame.loc[:, keep]
        .sort_values(["session", "decision_ordinal", "stock"], kind="mergesort")
        .reset_index(drop=True)
    )
    accounting = {
        "rows": int(len(output)),
        "development_rows": int(output["year"].eq(2024).sum()),
        "assessment_rows": int(output["year"].eq(2025).sum()),
        "minimum_session": str(output["session"].min()),
        "maximum_session": str(output["session"].max()),
        "chronology_evidence_id": str(evidence["movement_chronology_evidence_id"]),
    }
    return output, accounting


def _training_cutoff(score_month: pd.Series) -> pd.Series:
    starts = pd.to_datetime(score_month.astype(str) + "-01", utc=True, errors="raise")
    return starts - pd.Timedelta(nanoseconds=1)


def _read_prediction_pair(path: Path, *, year: int, evidence: dict[str, Any]) -> pd.DataFrame:
    start = START_SESSION if year == 2024 else "2025-01-01"
    end = "2025-01-01" if year == 2024 else END_EXCLUSIVE_SESSION
    frame = pd.read_parquet(
        path,
        columns=[
            "decision_id",
            "representation",
            "session",
            "score_month",
            "evaluation_period",
            "model",
            "probability",
            "training_rows",
        ],
        filters=[
            ("representation", "==", SEMANTIC_REPRESENTATION),
            ("session", ">=", start),
            ("session", "<", end),
        ],
    )
    frame = frame.loc[frame["model"].isin(["M2_IMMEDIATE_PAIR", "M5_LAST_FIVE_STATES"])].copy()
    chronology = cast(dict[str, Any], evidence["closure_chronology"])
    expected_period = str(
        chronology[
            "development_evaluation_period" if year == 2024 else "assessment_evaluation_period"
        ]
    )
    if (
        not frame["evaluation_period"].astype(str).eq(expected_period).all()
        or not frame["score_month"].astype(str).eq(frame["session"].astype(str).str[:7]).all()
    ):
        raise RuntimeError("blocked_chronology_or_leakage_failure: closure score identity")
    duplicate = frame.duplicated(["decision_id", "model"], keep=False)
    if duplicate.any():
        raise RuntimeError("blocked_reproducibility_or_audit_failure: closure prediction duplicate")
    model_counts = frame.groupby("decision_id", sort=True)["model"].nunique()
    if not model_counts.eq(2).all():
        raise RuntimeError("blocked_missing_frozen_joint_inputs: incomplete M2/M5 pair")
    metadata_consistency = frame.groupby("decision_id", sort=True)[
        ["score_month", "evaluation_period", "training_rows"]
    ].nunique()
    if metadata_consistency.gt(1).any().any():
        raise RuntimeError("blocked_chronology_or_leakage_failure: closure metadata differs")
    if year == 2025:
        expected_training_rows = int(chronology["assessment_training_rows"])
        if (
            not pd.to_numeric(frame["training_rows"], errors="raise")
            .eq(expected_training_rows)
            .all()
        ):
            raise RuntimeError(
                "blocked_chronology_or_leakage_failure: closure assessment fit count"
            )
    elif pd.to_numeric(frame["training_rows"], errors="raise").le(0).any():
        raise RuntimeError("blocked_chronology_or_leakage_failure: closure OOF fit count")
    probability = frame.pivot(index="decision_id", columns="model", values="probability")
    probability = probability.rename(
        columns={
            "M2_IMMEDIATE_PAIR": "p_close_m2",
            "M5_LAST_FIVE_STATES": "p_close_m5",
        }
    )
    metadata = frame.groupby("decision_id", sort=True, as_index=True).agg(
        score_month=("score_month", "first"),
        closure_training_rows=("training_rows", "min"),
    )
    output = metadata.join(probability, how="inner").reset_index()
    output["year"] = year
    output["closure_trained_through"] = (
        _training_cutoff(output["score_month"])
        if year == 2024
        else pd.Timestamp(str(chronology["final_trained_through"]))
    )
    output["closure_m2_oof"] = year == 2024
    output["closure_m5_oof"] = year == 2024
    output["closure_frozen_before_outcome"] = year == 2025
    output["closure_chronology_evidence_id"] = str(evidence["closure_chronology_evidence_id"])
    return output


def _read_closure_surface(evidence: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    population = pd.read_parquet(
        CLOSURE_POPULATION_PATH,
        columns=[
            "decision_id",
            "representation",
            "symbol",
            "session",
            "segment_id",
            "decision_timestamp",
            "target_available_timestamp",
            "current_state",
            "previous_state_1",
            "target_available",
            "target_pair_closure",
            "censor_reason",
            "source_provider",
            "source_artifact",
            "source_hash",
            "data_snapshot_hash",
            "period",
        ],
        filters=[
            ("representation", "==", SEMANTIC_REPRESENTATION),
            ("session", ">=", START_SESSION),
            ("session", "<", END_EXCLUSIVE_SESSION),
        ],
    )
    development = _read_prediction_pair(CLOSURE_OOF_PATH, year=2024, evidence=evidence)
    assessment = _read_prediction_pair(CLOSURE_ASSESSMENT_PATH, year=2025, evidence=evidence)
    probabilities = pd.concat([development, assessment], ignore_index=True)
    population = population.merge(
        probabilities,
        left_on="decision_id",
        right_on="decision_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_prediction"),
    )
    closure_source = cast(dict[str, Any], evidence["closure_source"])
    for year, period_key, period_label in (
        (2024, "development", "DEVELOPMENT_2024"),
        (2025, "assessment", "ASSESSMENT_2025"),
    ):
        mask = population["session"].astype(str).str[:4].eq(str(year))
        identity = cast(dict[str, Any], closure_source[period_key])
        source_hashes = cast(dict[str, str], identity["source_hashes"])
        expected_hash = population.loc[mask, "symbol"].astype(str).map(source_hashes)
        expected_artifact = (
            "symbol=" + population.loc[mask, "symbol"].astype(str) + "/timeframe=5m/data.parquet"
        )
        if (
            expected_hash.isna().any()
            or not population.loc[mask, "source_provider"].astype(str).eq("EODHD").all()
            or not population.loc[mask, "source_artifact"].astype(str).eq(expected_artifact).all()
            or not population.loc[mask, "source_hash"].astype(str).eq(expected_hash).all()
            or not population.loc[mask, "data_snapshot_hash"]
            .astype(str)
            .eq(str(identity["data_snapshot_hash"]))
            .all()
            or not population.loc[mask, "period"].astype(str).eq(period_label).all()
        ):
            raise RuntimeError(f"blocked_join_semantics_failure: closure source lineage {year}")
    population["pair_forecast_id"] = population.pop("decision_id").astype(str)
    population["stock"] = population.pop("symbol").astype(str)
    population["pair_forecast_timestamp"] = pd.to_datetime(
        population.pop("decision_timestamp"), utc=True, errors="raise"
    )
    population["closure_resolution_timestamp"] = pd.to_datetime(
        population.pop("target_available_timestamp"), utc=True, errors="coerce"
    )
    population["current_state_b"] = population.pop("current_state").astype(int)
    population["pair_orientation"] = (
        population["previous_state_1"].astype(int).astype(str)
        + "->"
        + population["current_state_b"].astype(str)
    )
    population["immediate_pair_closure"] = population.pop("target_pair_closure")
    population["closure_available"] = population.pop("target_available").astype(bool)
    population["source_gap"] = (
        population["censor_reason"].astype(str).eq("UNAVAILABLE_STRUCTURAL_GAP")
    )
    population["representation_id"] = (
        f"{STATE_MODEL_ID}|{STATE_MODEL_HASH}|{SEMANTIC_REPRESENTATION}"
    )
    population["source_lineage_id"] = str(evidence["lineage_id"])
    population["year"] = pd.to_numeric(
        population["session"].astype(str).str.slice(0, 4), errors="raise"
    ).astype(int)
    population["closure_m2_oof"] = population["closure_m2_oof"].eq(True)
    population["closure_m5_oof"] = population["closure_m5_oof"].eq(True)
    population["closure_frozen_before_outcome"] = population["closure_frozen_before_outcome"].eq(
        True
    )
    keep = [
        "pair_forecast_id",
        "representation_id",
        "source_lineage_id",
        "stock",
        "session",
        "year",
        "segment_id",
        "pair_forecast_timestamp",
        "closure_resolution_timestamp",
        "current_state_b",
        "pair_orientation",
        "p_close_m2",
        "p_close_m5",
        "immediate_pair_closure",
        "closure_available",
        "source_gap",
        "closure_m2_oof",
        "closure_m5_oof",
        "closure_trained_through",
        "closure_frozen_before_outcome",
        "closure_training_rows",
        "closure_chronology_evidence_id",
        "score_month",
        "censor_reason",
        "source_provider",
        "source_artifact",
        "source_hash",
        "data_snapshot_hash",
    ]
    output = (
        population.loc[:, keep]
        .sort_values(
            ["session", "stock", "pair_forecast_timestamp", "pair_forecast_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    accounting = {
        "rows": int(len(output)),
        "development_rows": int(output["year"].eq(2024).sum()),
        "assessment_rows": int(output["year"].eq(2025).sum()),
        "minimum_session": str(output["session"].min()),
        "maximum_session": str(output["session"].max()),
        "available_rows": int(output["closure_available"].sum()),
        "chronology_evidence_id": str(evidence["closure_chronology_evidence_id"]),
    }
    return output, accounting


def _compact_joined_panel(
    movement: pd.DataFrame,
    closure: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    result = exact_active_pair_join(movement, closure, max_rows=10_000)
    frame = add_joint_probability_features(result.frame)
    frame["log1p_predicted_absolute_movement_bps"] = np.log1p(
        frame["predicted_absolute_movement_bps"].to_numpy(dtype=float)
    )
    frame = with_equal_slate_weights(frame)
    frame["year"] = pd.to_numeric(frame["session"].astype(str).str[:4], errors="raise").astype(int)
    frame["closure_m2_oof"] = frame["closure_m2_oof"].astype(bool)
    frame["closure_m5_oof"] = frame["closure_m5_oof"].astype(bool)
    frame["closure_frozen_before_outcome"] = frame["closure_frozen_before_outcome"].astype(bool)
    frame["movement_available"] = frame["movement_available"].astype(bool)
    frame["closure_available"] = frame["closure_available"].astype(bool)
    compact_columns = [
        "joined_row_id",
        "pair_forecast_id",
        "movement_row_id",
        "representation_id",
        "source_lineage_id",
        "stock",
        "session",
        "year",
        "decision_ordinal",
        "slate_id",
        "pair_forecast_timestamp",
        "fixed_clock_timestamp",
        "pair_age_bars",
        "scheduled_bars_remaining",
        "closure_resolution_timestamp",
        "movement_horizon_terminal_timestamp",
        "pair_orientation",
        "p_move",
        "predicted_absolute_movement_bps",
        "p_close_m2",
        "p_close_m5",
        "logit_p_move",
        "logit_p_close_m2",
        "logit_p_close_m5",
        "closure_history_increment",
        "log1p_predicted_absolute_movement_bps",
        "large_move",
        "immediate_pair_closure",
        "joint_large_move_and_closure",
        "closure_available",
        "movement_available",
        "movement_oof",
        "closure_m2_oof",
        "closure_m5_oof",
        "movement_trained_through",
        "movement_size_trained_through",
        "closure_trained_through",
        "movement_frozen_before_outcome",
        "closure_frozen_before_outcome",
        "movement_chronology_evidence_id",
        "closure_chronology_evidence_id",
        "joined_slate_size",
        "row_weight",
    ]
    compact = (
        frame.loc[:, compact_columns]
        .sort_values(
            ["session", "decision_ordinal", "stock", "pair_forecast_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    assert_compact_panel_has_no_forbidden_fields(compact)
    assert_upstream_chronology(compact)
    timestamp_columns = [
        "pair_forecast_timestamp",
        "fixed_clock_timestamp",
        "closure_resolution_timestamp",
        "movement_horizon_terminal_timestamp",
    ]
    for column in timestamp_columns:
        assert_protected_date_boundary(compact[column])
    if not compact["closure_resolution_timestamp"].gt(compact["fixed_clock_timestamp"]).all():
        raise RuntimeError("blocked_join_semantics_failure: non-causal closure outcome")
    if compact.duplicated(["representation_id", "stock", "session", "pair_forecast_id"]).any():
        raise RuntimeError("blocked_join_semantics_failure: duplicate pair forecast")
    return compact, result.accounting


def _fit_models(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    models = {
        "A2": fit_fixed_logistic(
            development,
            development["immediate_pair_closure"],
            features=A2_FEATURES,
            slate_column="slate_id",
            model_id="A2",
        ),
        "A3": fit_fixed_logistic(
            development,
            development["immediate_pair_closure"],
            features=A3_FEATURES,
            slate_column="slate_id",
            model_id="A3",
        ),
        "B0": fit_fixed_logistic(
            development,
            development["large_move"],
            features=B0_FEATURES,
            slate_column="slate_id",
            model_id="B0",
        ),
        "B1": fit_fixed_logistic(
            development,
            development["large_move"],
            features=B1_FEATURES,
            slate_column="slate_id",
            model_id="B1",
        ),
        "C1": fit_fixed_logistic(
            development,
            development["joint_large_move_and_closure"],
            features=C1_FEATURES,
            slate_column="slate_id",
            model_id="C1",
        ),
    }
    if len(models) > 6:
        raise RuntimeError("blocked_quick_screen_resource_limit")
    predictions = assessment.copy()
    predictions["p_A0"] = predictions["p_close_m2"]
    predictions["p_A1"] = predictions["p_close_m5"]
    predictions["p_A2"] = models["A2"].predict(predictions)
    predictions["p_A3"] = models["A3"].predict(predictions)
    predictions["p_B0"] = models["B0"].predict(predictions)
    predictions["p_B1"] = models["B1"].predict(predictions)
    predictions["p_C0"] = predictions["p_move"] * predictions["p_close_m5"]
    predictions["p_C1"] = models["C1"].predict(predictions)
    return predictions, {name: model.as_dict() for name, model in sorted(models.items())}


def _model_configurations() -> dict[str, Any]:
    base = {
        "class": "LogisticRegression",
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": 250,
        "class_weight": None,
        "n_jobs": 1,
        "scaler_fit": "2024_joined_rows_only",
    }
    return {
        **SAFETY_FLAGS,
        "actual_fitted_model_count": 5,
        "hard_cap": 6,
        "models": {
            "A0": {"status": "direct_no_fit", "prediction": "p_close_m2"},
            "A1": {"status": "direct_no_fit", "prediction": "p_close_m5"},
            "A2": {**base, "status": "fitted", "features": list(A2_FEATURES)},
            "A3": {**base, "status": "fitted", "features": list(A3_FEATURES)},
            "A4": {"status": "not_fitted_hard_cap", "rescue_allowed": False},
            "B0": {**base, "status": "fitted", "features": list(B0_FEATURES)},
            "B1": {**base, "status": "fitted", "features": list(B1_FEATURES)},
            "B2": {"status": "not_fitted_hard_cap", "rescue_allowed": False},
            "C0": {"status": "direct_no_fit", "prediction": "p_move * p_close_m5"},
            "C1": {**base, "status": "fitted", "features": list(C1_FEATURES)},
            "C2": {"status": "not_fitted_hard_cap", "selection_allowed": False},
        },
    }


def _metrics(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = {
        "closure": [
            ("A0", "immediate_pair_closure", "p_A0"),
            ("A1", "immediate_pair_closure", "p_A1"),
            ("A2", "immediate_pair_closure", "p_A2"),
            ("A3", "immediate_pair_closure", "p_A3"),
        ],
        "movement": [
            ("B0", "large_move", "p_B0"),
            ("B1", "large_move", "p_B1"),
        ],
        "joint": [
            ("C0", "joint_large_move_and_closure", "p_C0"),
            ("C1", "joint_large_move_and_closure", "p_C1"),
        ],
    }
    tables: dict[str, list[dict[str, Any]]] = {key: [] for key in specs}
    bins: list[pd.DataFrame] = []
    for arm, arm_specs in specs.items():
        for model, target, probability in arm_specs:
            metric, reliability = probability_metrics(
                assessment, target=target, probability=probability, model_id=model
            )
            tables[arm].append(metric)
            bins.append(reliability.assign(arm=arm))
    comparisons = {
        "A1": paired_loss_improvements(
            assessment,
            target="immediate_pair_closure",
            baseline="p_A0",
            candidate="p_A1",
        ),
        "A3": paired_loss_improvements(
            assessment,
            target="immediate_pair_closure",
            baseline="p_A2",
            candidate="p_A3",
        ),
        "B1": paired_loss_improvements(
            assessment, target="large_move", baseline="p_B0", candidate="p_B1"
        ),
        "C1": paired_loss_improvements(
            assessment,
            target="joint_large_move_and_closure",
            baseline="p_C0",
            candidate="p_C1",
        ),
    }
    baseline_names = {"A1": "A0", "A3": "A2", "B1": "B0", "C1": "C0"}
    for rows in tables.values():
        for row in rows:
            model = str(row["model"])
            if model in comparisons:
                row["comparison_baseline"] = baseline_names[model]
                row.update(comparisons[model])
            else:
                row["comparison_baseline"] = ""
                row["brier_improvement"] = float("nan")
                row["log_loss_improvement"] = float("nan")
    return (
        pd.DataFrame(tables["closure"]),
        pd.DataFrame(tables["movement"]),
        pd.DataFrame(tables["joint"]),
        pd.concat(bins, ignore_index=True),
    )


def _training_edges(values: pd.Series, bins: int) -> npt.NDArray[np.float64]:
    raw = np.quantile(values.to_numpy(dtype=float), np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(raw)
    if len(edges) < 2:
        return np.asarray([-np.inf, np.inf], dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    return np.asarray(edges, dtype=float)


def _add_breakdown_groups(development: pd.DataFrame, assessment: pd.DataFrame) -> pd.DataFrame:
    output = assessment.copy()
    output["month"] = pd.to_datetime(output["session"], errors="raise").dt.strftime("%Y-%m")
    age_edges = _training_edges(development["pair_age_bars"], 4)
    close_edges = _training_edges(development["closure_history_increment"], 5)
    move_edges = _training_edges(development["p_move"], 5)
    output["pair_age_quartile"] = (
        pd.cut(output["pair_age_bars"], age_edges, include_lowest=True, labels=False).astype(int)
        + 1
    )
    output["closure_history_increment_quintile"] = (
        pd.cut(
            output["closure_history_increment"], close_edges, include_lowest=True, labels=False
        ).astype(int)
        + 1
    )
    output["movement_probability_quintile"] = (
        pd.cut(output["p_move"], move_edges, include_lowest=True, labels=False).astype(int) + 1
    )
    return output


def _breakdown_metrics(development: pd.DataFrame, assessment: pd.DataFrame) -> pd.DataFrame:
    grouped = _add_breakdown_groups(development, assessment)
    comparisons = [
        ("closure", "A3_vs_A2", "immediate_pair_closure", "p_A2", "p_A3"),
        ("movement", "B1_vs_B0", "large_move", "p_B0", "p_B1"),
        (
            "joint",
            "C1_vs_C0",
            "joint_large_move_and_closure",
            "p_C0",
            "p_C1",
        ),
    ]
    dimensions = [
        "month",
        "decision_ordinal",
        "pair_age_quartile",
        "stock",
        "closure_history_increment_quintile",
        "movement_probability_quintile",
    ]
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        for value, group in grouped.groupby(dimension, sort=True, observed=True):
            for arm, comparison, target, baseline, candidate in comparisons:
                increment = paired_loss_improvements(
                    group, target=target, baseline=baseline, candidate=candidate
                )
                rows.append(
                    {
                        "breakdown": dimension,
                        "value": str(value),
                        "arm": arm,
                        "comparison": comparison,
                        "rows": int(len(group)),
                        "sessions": int(group["session"].nunique()),
                        "stocks": int(group["stock"].nunique()),
                        "outcome_rate": float(group[target].mean()),
                        **increment,
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["breakdown", "value", "arm"], kind="mergesort")
        .reset_index(drop=True)
    )


def _bootstrap_all(assessment: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs = [
        ("A3_vs_A2", "immediate_pair_closure", "p_A2", "p_A3"),
        ("B1_vs_B0", "large_move", "p_B0", "p_B1"),
        ("C1_vs_C0", "joint_large_move_and_closure", "p_C0", "p_C1"),
    ]
    parts: list[pd.DataFrame] = []
    intervals: dict[str, Any] = {}
    for comparison, target, baseline, candidate in specs:
        draws = session_block_bootstrap_improvements(
            assessment,
            target=target,
            baseline=baseline,
            candidate=candidate,
            draws=BOOTSTRAP_DRAWS,
            seed=20260720,
        )
        draws.insert(0, "comparison", comparison)
        parts.append(draws)
        intervals[comparison] = bootstrap_intervals(draws)
    return pd.concat(parts, ignore_index=True), intervals


def _null_models(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    real: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feasibility_parts: list[pd.DataFrame] = []
    feasibility_summary: dict[str, Any] = {}
    for label, frame in (("development_2024", development), ("assessment_2025", assessment)):
        part = whole_session_shift_feasibility(frame)
        part.insert(0, "population", label)
        feasibility_parts.append(part)
        feasibility_summary[label] = {
            "session_blocks": int(part["session_blocks"].sum()),
            "shiftable_blocks": int(part["shiftable_blocks"].sum()),
            "unshiftable_blocks": int(part["unshiftable_blocks"].sum()),
            "membership_groups": int(len(part)),
        }
    feasibility = pd.concat(feasibility_parts, ignore_index=True)
    unshiftable = int(feasibility["unshiftable_blocks"].sum())
    if unshiftable > 0:
        feasibility.insert(0, "null", "whole_session_shift_feasibility")
        feasibility.insert(1, "draw", -1)
        return feasibility, {
            "status": "blocked_join_semantics_failure",
            "blocker": "blocked_join_semantics_failure",
            "reason": (
                "exact stock-membership blocks cannot all receive a non-identity "
                "whole-session shift"
            ),
            "requested_draws_per_primary_increment": NULL_DRAWS,
            "completed_draws_per_primary_increment": 0,
            "feasibility": feasibility_summary,
            "closure_increment": {"status": "not_run_null_semantics_failure"},
            "movement_increment": {"status": "not_run_null_semantics_failure"},
        }
    rows: list[dict[str, Any]] = []
    for draw in range(NULL_DRAWS):
        shifted_dev, dev_manifest = whole_session_shift(
            development,
            value_columns=("p_move", "predicted_absolute_movement_bps"),
            draw=draw,
            seed=20260721,
        )
        shifted_assessment, assessment_manifest = whole_session_shift(
            assessment,
            value_columns=("p_move", "predicted_absolute_movement_bps"),
            draw=draw,
            seed=20260721,
        )
        for shifted in (shifted_dev, shifted_assessment):
            shifted["logit_p_move"] = logit_probability(shifted["p_move"].to_numpy(float))
            shifted["log1p_predicted_absolute_movement_bps"] = np.log1p(
                shifted["predicted_absolute_movement_bps"].to_numpy(float)
            )
        model = fit_fixed_logistic(
            shifted_dev,
            shifted_dev["immediate_pair_closure"],
            features=A3_FEATURES,
            slate_column="slate_id",
            model_id=f"A3_null_{draw:03d}",
            random_state=20260721 + draw,
        )
        shifted_assessment["p_null_candidate"] = model.predict(shifted_assessment)
        improvement = paired_loss_improvements(
            shifted_assessment,
            target="immediate_pair_closure",
            baseline="p_A2",
            candidate="p_null_candidate",
        )
        manifests = [*dev_manifest, *assessment_manifest]
        nonzero = sum(row["source_session"] != row["destination_session"] for row in manifests)
        rows.append(
            {
                "null": "closure_increment",
                "draw": draw,
                **improvement,
                "block_assignments": len(manifests),
                "non_identity_assignments": nonzero,
            }
        )

    for draw in range(NULL_DRAWS):
        shifted_dev, dev_manifest = whole_session_shift(
            development,
            value_columns=("p_close_m2", "p_close_m5", "closure_history_increment"),
            draw=draw,
            seed=20260722,
        )
        shifted_assessment, assessment_manifest = whole_session_shift(
            assessment,
            value_columns=("p_close_m2", "p_close_m5", "closure_history_increment"),
            draw=draw,
            seed=20260722,
        )
        for shifted in (shifted_dev, shifted_assessment):
            shifted["logit_p_close_m2"] = logit_probability(shifted["p_close_m2"].to_numpy(float))
            shifted["logit_p_close_m5"] = logit_probability(shifted["p_close_m5"].to_numpy(float))
            expected_increment = shifted["logit_p_close_m5"] - shifted["logit_p_close_m2"]
            if not np.allclose(
                expected_increment,
                shifted["closure_history_increment"].to_numpy(float),
                atol=1e-12,
            ):
                raise RuntimeError("blocked_reproducibility_or_audit_failure: null block split")
        model = fit_fixed_logistic(
            shifted_dev,
            shifted_dev["large_move"],
            features=B1_FEATURES,
            slate_column="slate_id",
            model_id=f"B1_null_{draw:03d}",
            random_state=20260722 + draw,
        )
        shifted_assessment["p_null_candidate"] = model.predict(shifted_assessment)
        improvement = paired_loss_improvements(
            shifted_assessment,
            target="large_move",
            baseline="p_B0",
            candidate="p_null_candidate",
        )
        manifests = [*dev_manifest, *assessment_manifest]
        nonzero = sum(row["source_session"] != row["destination_session"] for row in manifests)
        rows.append(
            {
                "null": "movement_increment",
                "draw": draw,
                **improvement,
                "block_assignments": len(manifests),
                "non_identity_assignments": nonzero,
            }
        )

    nulls = pd.DataFrame(rows).sort_values(["null", "draw"], kind="mergesort")
    summary: dict[str, Any] = {}
    mapping = {
        "closure_increment": "A3_vs_A2",
        "movement_increment": "B1_vs_B0",
    }
    for null_name, comparison in mapping.items():
        group = nulls.loc[nulls["null"].eq(null_name)]
        summary[null_name] = {}
        for metric in ("brier_improvement", "log_loss_improvement"):
            values = group[metric].to_numpy(dtype=float)
            observed = real[comparison][metric]
            summary[null_name][metric] = {
                "real": observed,
                "null_90th_percentile": float(np.quantile(values, 0.90)),
                "real_percentile": float(np.mean(values <= observed)),
            }
    return nulls.reset_index(drop=True), summary


def _concentration(assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(assessment)
    month = pd.to_datetime(assessment["session"], errors="raise").dt.strftime("%Y-%m")
    groups = {
        "stock": assessment["stock"].astype(str),
        "month": month,
        "pair_orientation": assessment["pair_orientation"].astype(str),
        "joined_slate_size": assessment["joined_slate_size"].astype(int).astype(str),
    }
    for group_type, values in groups.items():
        counts = values.value_counts(sort=False).sort_index()
        for value, count in counts.items():
            rows.append(
                {
                    "group": group_type,
                    "value": str(value),
                    "rows": int(count),
                    "fraction": float(count / total),
                }
            )
    return pd.DataFrame(rows).sort_values(["group", "value"], kind="mergesort")


def _plot_calibration(
    bins: pd.DataFrame,
    *,
    models: Sequence[str],
    title: str,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#777777", linewidth=1.0)
    colors = ("#1f4e79", "#c55a11")
    for model, color in zip(models, colors, strict=True):
        frame = bins.loc[bins["model"].eq(model) & bins["rows"].gt(0)]
        axis.plot(
            frame["mean_prediction"],
            frame["outcome_rate"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=model,
        )
    axis.set(xlabel="Mean predicted probability", ylabel="Observed rate", title=title)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.savefig(path, dpi=140, metadata={"Software": "Stocker research"})
    plt.close(fig)


def _date_boundary_audit(
    movement: pd.DataFrame,
    closure: pd.DataFrame,
    joined: pd.DataFrame,
) -> dict[str, Any]:
    counts: list[dict[str, Any]] = []
    for source, frame in (
        ("movement_rows_inspected", movement),
        ("closure_forecasts_inspected", closure),
        ("joined_rows", joined),
    ):
        months = pd.to_datetime(frame["session"], errors="raise").dt.strftime("%Y-%m")
        for month, count in months.value_counts(sort=False).sort_index().items():
            counts.append({"source": source, "year_month": month, "rows": int(count)})
    timestamp_sources = {
        "movement.fixed_clock_timestamp": movement["fixed_clock_timestamp"],
        "movement.movement_horizon_terminal_timestamp": movement[
            "movement_horizon_terminal_timestamp"
        ],
        "closure.pair_forecast_timestamp": closure["pair_forecast_timestamp"],
        "closure.closure_resolution_timestamp": closure["closure_resolution_timestamp"],
        "joined.pair_forecast_timestamp": joined["pair_forecast_timestamp"],
        "joined.fixed_clock_timestamp": joined["fixed_clock_timestamp"],
        "joined.closure_resolution_timestamp": joined["closure_resolution_timestamp"],
        "joined.movement_horizon_terminal_timestamp": joined["movement_horizon_terminal_timestamp"],
    }
    timestamps = pd.concat(list(timestamp_sources.values()), ignore_index=True).dropna()
    materialized_ranges = {
        name: {
            "minimum": pd.to_datetime(values.dropna(), utc=True).min().isoformat(),
            "maximum": pd.to_datetime(values.dropna(), utc=True).max().isoformat(),
        }
        for name, values in sorted(timestamp_sources.items())
        if values.notna().any()
    }
    return {
        **SAFETY_FLAGS,
        "predicate_start_inclusive": START_SESSION,
        "predicate_end_exclusive": END_EXCLUSIVE_SESSION,
        "minimum_timestamp_read": pd.to_datetime(timestamps, utc=True).min().isoformat(),
        "maximum_timestamp_read": pd.to_datetime(timestamps, utc=True).max().isoformat(),
        "counts_by_year_month": counts,
        "materialized_timestamp_ranges": materialized_ranges,
        "protected_rows_opened": 0,
        "years_opened": [2024, 2025],
        "movement_original_opened_period": "2024-01-01 through 2025-08-22",
        "closure_original_opened_period": "2024-01-01 through 2025-12-31",
        "joint_materialised_period": f"{joined['session'].min()} through {joined['session'].max()}",
    }


def _join_contract(mapping_hash: str, lineage_id: str) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "movement_anchor": "feature_available_timestamp_utc",
        "same_stock": True,
        "same_session": True,
        "same_representation": SEMANTIC_REPRESENTATION,
        "same_state_model_hash": STATE_MODEL_HASH,
        "same_source_lineage_id": lineage_id,
        "semantic_mapping_hash": mapping_hash,
        "active_pair_predicate": (
            "pair_forecast_timestamp <= fixed_clock_timestamp < closure_resolution_timestamp"
        ),
        "state_predicate": "semantic current pair state B equals mapped movement hard state",
        "segment_predicate": "pair segment_id equals movement origin_segment_id",
        "source_gap_allowed": False,
        "outcome_after_fixed_clock_required": True,
        "right_censored_pair_label": "unavailable_not_false",
        "deduplication": (
            "earliest fixed clock by representation_id x stock x session x pair_forecast_id"
        ),
        "pair_age_definition": (
            "integer completed five-minute bars from pair forecast creation to fixed clock"
        ),
    }


def _feature_manifest() -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "epsilon_for_logit_only": 1e-6,
        "features": {
            "A2": list(A2_FEATURES),
            "A3": list(A3_FEATURES),
            "B0": list(B0_FEATURES),
            "B1": list(B1_FEATURES),
            "C1": list(C1_FEATURES),
        },
        "derived": {
            "closure_history_increment": "logit_p_close_m5 - logit_p_close_m2",
            "joint_large_move_and_closure": ("large_move AND immediate_pair_closure"),
        },
        "forbidden": [
            "future signed return",
            "long or short target",
            "P&L",
            "MFE",
            "MAE",
            "profitable-loop label",
            "exact five-state history",
            "payoff history",
            "broker or execution data",
        ],
    }


def _decision_payload(
    assessment: pd.DataFrame,
    closure_metrics: pd.DataFrame,
    movement_metrics: pd.DataFrame,
    joint_metrics: pd.DataFrame,
    breakdown: pd.DataFrame,
    intervals: dict[str, Any],
    null_summary: dict[str, Any],
) -> dict[str, Any]:
    support = evaluate_support(assessment)
    point = {
        "A3_vs_A2": {
            key: float(value)
            for key, value in paired_loss_improvements(
                assessment,
                target="immediate_pair_closure",
                baseline="p_A2",
                candidate="p_A3",
            ).items()
        },
        "B1_vs_B0": {
            key: float(value)
            for key, value in paired_loss_improvements(
                assessment, target="large_move", baseline="p_B0", candidate="p_B1"
            ).items()
        },
        "C1_vs_C0": {
            key: float(value)
            for key, value in paired_loss_improvements(
                assessment,
                target="joint_large_move_and_closure",
                baseline="p_C0",
                candidate="p_C1",
            ).items()
        },
        "A1_vs_A0_direct": {
            key: float(value)
            for key, value in paired_loss_improvements(
                assessment,
                target="immediate_pair_closure",
                baseline="p_A0",
                candidate="p_A1",
            ).items()
        },
    }
    month = breakdown.loc[breakdown["breakdown"].eq("month")]
    month_stability: dict[str, Any] = {}
    for comparison in ("A3_vs_A2", "B1_vs_B0", "C1_vs_C0"):
        group = month.loc[month["comparison"].eq(comparison)]
        month_stability[comparison] = {
            "represented_months": int(len(group)),
            "positive_brier_months": int(group["brier_improvement"].gt(0.0).sum()),
            "positive_log_loss_months": int(group["log_loss_improvement"].gt(0.0).sum()),
        }
    null_blocker = cast(str | None, null_summary.get("blocker"))
    if null_blocker is None:
        arm_a = primary_arm_passes(
            brier_improvement=point["A3_vs_A2"]["brier_improvement"],
            log_loss_improvement=point["A3_vs_A2"]["log_loss_improvement"],
            brier_lower_90=intervals["A3_vs_A2"]["brier"]["lower_90"],
            log_loss_lower_90=intervals["A3_vs_A2"]["log_loss"]["lower_90"],
            positive_months=month_stability["A3_vs_A2"]["positive_brier_months"],
            represented_months=month_stability["A3_vs_A2"]["represented_months"],
            null_percentile=null_summary["closure_increment"]["brier_improvement"][
                "real_percentile"
            ],
            concentration_passed=bool(support["concentration_passed"]),
        )
        arm_b = primary_arm_passes(
            brier_improvement=point["B1_vs_B0"]["brier_improvement"],
            log_loss_improvement=point["B1_vs_B0"]["log_loss_improvement"],
            brier_lower_90=intervals["B1_vs_B0"]["brier"]["lower_90"],
            log_loss_lower_90=intervals["B1_vs_B0"]["log_loss"]["lower_90"],
            positive_months=month_stability["B1_vs_B0"]["positive_brier_months"],
            represented_months=month_stability["B1_vs_B0"]["represented_months"],
            null_percentile=null_summary["movement_increment"]["brier_improvement"][
                "real_percentile"
            ],
            concentration_passed=bool(support["concentration_passed"]),
        )
        arm_c = joint_arm_passes(
            brier_improvement=point["C1_vs_C0"]["brier_improvement"],
            log_loss_improvement=point["C1_vs_C0"]["log_loss_improvement"],
            brier_lower_90=intervals["C1_vs_C0"]["brier"]["lower_90"],
            log_loss_lower_90=intervals["C1_vs_C0"]["log_loss"]["lower_90"],
            positive_months=month_stability["C1_vs_C0"]["positive_brier_months"],
            represented_months=month_stability["C1_vs_C0"]["represented_months"],
            joint_support_status=str(support["joint_support_status"]),
        )
    else:
        arm_a = False
        arm_b = False
        arm_c = False
    blocker = null_blocker or cast(str | None, support["blocker"])
    decision = classify_joint_decision(
        arm_a_pass=arm_a,
        arm_b_pass=arm_b,
        arm_c_pass=arm_c,
        blocker=blocker,
    )
    return {
        **SAFETY_FLAGS,
        "decision": decision,
        "arm_a_pass": arm_a,
        "arm_b_pass": arm_b,
        "arm_c_pass": arm_c,
        "support": support,
        "point_improvements": point,
        "monthly_stability": month_stability,
        "bootstrap_intervals": intervals,
        "null_summary": null_summary,
        "direct_model_metrics": {
            "closure": closure_metrics.to_dict(orient="records"),
            "movement": movement_metrics.to_dict(orient="records"),
            "joint": joint_metrics.to_dict(orient="records"),
        },
        "direction_tested": False,
        "economic_tested": False,
        "execution_tested": False,
    }


def _report(
    decision: dict[str, Any],
    join_accounting: dict[str, int],
    date_audit: dict[str, Any],
) -> str:
    point = decision["point_improvements"]
    support = decision["support"]
    monthly = decision["monthly_stability"]
    intervals = decision["bootstrap_intervals"]
    nulls = decision["null_summary"]
    direct = point["A1_vs_A0_direct"]
    closure = point["A3_vs_A2"]
    movement = point["B1_vs_B0"]
    joint = point["C1_vs_C0"]
    null_blocked = nulls.get("status") == "blocked_join_semantics_failure"
    closure_null_text = (
        "not run: exact membership blocks cannot all be shifted"
        if null_blocked
        else f"percentile {nulls['closure_increment']['brier_improvement']['real_percentile']:.3f}"
    )
    movement_null_text = (
        "not run: exact membership blocks cannot all be shifted"
        if null_blocked
        else f"percentile {nulls['movement_increment']['brier_improvement']['real_percentile']:.3f}"
    )
    lines = [
        "# Movement × Closure-History Joint Increment V0.1",
        "",
        "Research-only, retrospective, representation-specific feasibility screen. "
        "Execution and order placement are disabled. No direction, payoff, or "
        "executable edge was tested.",
        "",
        "## Population and join",
        "",
        f"- Common materialised period: {date_audit['joint_materialised_period']}.",
        f"- Exact joined rows across both years: {join_accounting['exact_joined_rows']}.",
        f"- 2025 assessment joined rows: {support['joined_rows']} "
        f"({support['sessions']} sessions, "
        f"{support['stocks']} stocks).",
        f"- Immediate closures: {support['immediate_closures']}.",
        f"- Large movements: {support['large_moves']}.",
        f"- Joint positives: {support['joint_positive_events']}.",
        f"- Duplicate later clocks removed: {join_accounting['excluded_duplicate_later_clock']}.",
        "",
        "## Results",
        "",
        f"- Direct M5 vs M2: Brier improvement {direct['brier_improvement']:.8f}; "
        f"log-loss improvement {direct['log_loss_improvement']:.8f}.",
        f"- A3 vs A2 closure: Brier {closure['brier_improvement']:.8f}; "
        f"log loss {closure['log_loss_improvement']:.8f}; "
        f"90% Brier interval [{intervals['A3_vs_A2']['brier']['lower_90']:.8f}, "
        f"{intervals['A3_vs_A2']['brier']['upper_90']:.8f}]; "
        f"null {closure_null_text}.",
        f"- B1 vs B0 movement: Brier {movement['brier_improvement']:.8f}; "
        f"log loss {movement['log_loss_improvement']:.8f}; "
        f"90% Brier interval [{intervals['B1_vs_B0']['brier']['lower_90']:.8f}, "
        f"{intervals['B1_vs_B0']['brier']['upper_90']:.8f}]; "
        f"null {movement_null_text}.",
        f"- C1 vs product: Brier {joint['brier_improvement']:.8f}; "
        f"log loss {joint['log_loss_improvement']:.8f}; "
        f"90% Brier interval [{intervals['C1_vs_C0']['brier']['lower_90']:.8f}, "
        f"{intervals['C1_vs_C0']['brier']['upper_90']:.8f}].",
        "",
        "## Monthly stability",
        "",
        f"- A3 vs A2 positive-Brier months: {monthly['A3_vs_A2']['positive_brier_months']}/"
        f"{monthly['A3_vs_A2']['represented_months']}.",
        f"- B1 vs B0 positive-Brier months: {monthly['B1_vs_B0']['positive_brier_months']}/"
        f"{monthly['B1_vs_B0']['represented_months']}.",
        f"- C1 vs C0 positive-Brier months: {monthly['C1_vs_C0']['positive_brier_months']}/"
        f"{monthly['C1_vs_C0']['represented_months']}.",
        "",
        "## Decision",
        "",
        f"`{decision['decision']}`",
        "",
        (
            "The preregistered session-shift null is not identified on this irregular "
            "joined panel: singleton exact-stock membership blocks cannot receive a "
            "non-identity whole-session shift. The screen therefore fails closed."
            if null_blocked
            else "The preregistered session-shift null completed."
        ),
        "",
        f"Arm A pass: `{decision['arm_a_pass']}`. Arm B pass: `{decision['arm_b_pass']}`. "
        f"Arm C pass: `{decision['arm_c_pass']}`.",
        "",
        "The optional A4, B2, and C2 interaction sensitivities were not fitted because the "
        "five required baseline/candidate stackers take precedence under the explicit "
        "six-model cap.",
        "",
        "This result does not claim a directional signal, strategy return, economic "
        "payoff, or executable system.",
        "",
    ]
    return "\n".join(lines)


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = _verify_required_inputs()
    mapping, mapping_hash = _load_semantic_mapping()
    evidence = _validate_upstream_evidence(mapping_hash)
    movement, movement_accounting = _read_movement_surface(mapping, evidence)
    closure, closure_accounting = _read_closure_surface(evidence)
    joined, join_accounting = _compact_joined_panel(movement, closure)
    development, assessment = split_development_assessment(joined)
    assessment_predictions, coefficients = _fit_models(development, assessment)
    closure_metrics, movement_metrics, joint_metrics, calibration_bins = _metrics(
        assessment_predictions
    )
    breakdown = _breakdown_metrics(development, assessment_predictions)
    bootstrap, intervals = _bootstrap_all(assessment_predictions)
    real = {
        "A3_vs_A2": paired_loss_improvements(
            assessment_predictions,
            target="immediate_pair_closure",
            baseline="p_A2",
            candidate="p_A3",
        ),
        "B1_vs_B0": paired_loss_improvements(
            assessment_predictions,
            target="large_move",
            baseline="p_B0",
            candidate="p_B1",
        ),
    }
    nulls, null_summary = _null_models(development, assessment_predictions, real)
    concentration = _concentration(assessment_predictions)
    decision = _decision_payload(
        assessment_predictions,
        closure_metrics,
        movement_metrics,
        joint_metrics,
        breakdown,
        intervals,
        null_summary,
    )
    date_audit = _date_boundary_audit(movement, closure, joined)
    source_manifest = {
        **SAFETY_FLAGS,
        "movement": {
            "experiment": "20260720-movement-conditioned-regime-path-probability-chain-v0",
            "commit": "e111cfd259c1ed9ad988903f6cf9d79c32ea80da",
            "opened_period": "2024-01-01 through 2025-08-22",
            "joint_rows_materialised": movement_accounting,
        },
        "closure": {
            "experiment": "20260720-immediate-pair-closure-history-v1",
            "source_commit": "04c6d45589e0c114dc0b03f6f98b4858bde7dffe",
            "implementation_hash": (
                "1d6b7a57ff37cf9b4f47e1c8dcfa555d4717ac9afcaccd0eefd4968f4d0c112b"
            ),
            "opened_period": "2024-01-01 through 2025-12-31",
            "joint_rows_materialised": closure_accounting,
        },
        "joint_common_start": str(joined["session"].min()),
        "joint_common_end": str(joined["session"].max()),
        "state_model_id": STATE_MODEL_ID,
        "state_model_hash": STATE_MODEL_HASH,
        "representation": SEMANTIC_REPRESENTATION,
        "source_lineage_id": evidence["lineage_id"],
        "lineage_reconciliation": evidence["lineage_evidence"],
        "chronology_evidence": {
            "movement": evidence["movement_chronology"],
            "closure": evidence["closure_chronology"],
        },
        "protected_rows_opened": 0,
    }
    hash_manifest = {
        **SAFETY_FLAGS,
        "artifacts": [
            {
                "logical_name": name,
                "repository_relative_path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": input_hashes[name],
                "size_bytes": path.stat().st_size,
            }
            for name, path in sorted(INPUT_PATHS.items())
        ],
    }
    coefficients_payload = {**SAFETY_FLAGS, "models": coefficients}

    _write_json(output / "source_manifest.json", source_manifest)
    _write_json(output / "input_artifact_hashes.json", hash_manifest)
    _write_json(output / "date_boundary_audit.json", date_audit)
    _write_json(
        output / "join_contract.json",
        _join_contract(mapping_hash, str(evidence["lineage_id"])),
    )
    pd.DataFrame(
        [{"category": key, "count": value} for key, value in sorted(join_accounting.items())]
    ).to_csv(output / "join_accounting.csv", index=False, lineterminator="\n")
    joined.to_parquet(output / "joined_compact_panel.parquet", index=False, compression="zstd")
    _write_json(output / "feature_manifest.json", _feature_manifest())
    _write_json(output / "model_configurations.json", _model_configurations())
    _write_json(output / "model_coefficients.json", coefficients_payload)
    assessment_predictions.to_parquet(
        output / "assessment_predictions.parquet", index=False, compression="zstd"
    )
    closure_metrics.to_csv(output / "closure_metrics.csv", index=False, lineterminator="\n")
    movement_metrics.to_csv(output / "movement_metrics.csv", index=False, lineterminator="\n")
    joint_metrics.to_csv(output / "joint_metrics.csv", index=False, lineterminator="\n")
    breakdown.to_csv(output / "monthly_metrics.csv", index=False, lineterminator="\n")
    calibration_bins.to_csv(output / "calibration_bins.csv", index=False, lineterminator="\n")
    bootstrap.to_csv(output / "bootstrap_metrics.csv", index=False, lineterminator="\n")
    nulls.to_csv(output / "null_metrics.csv", index=False, lineterminator="\n")
    concentration.to_csv(output / "concentration_metrics.csv", index=False, lineterminator="\n")
    _write_json(output / "decision.json", decision)
    report = _report(decision, join_accounting, date_audit)
    (output / "report.md").write_text(report, encoding="utf-8")
    _plot_calibration(
        calibration_bins,
        models=("A2", "A3"),
        title="Immediate pair closure calibration",
        path=output / "closure_calibration.png",
    )
    _plot_calibration(
        calibration_bins,
        models=("B0", "B1"),
        title="Large movement calibration",
        path=output / "movement_calibration.png",
    )
    if output.resolve() == (EXPERIMENT_DIR / "artifacts/primary").resolve():
        shutil.copyfile(output / "report.md", EXPERIMENT_DIR / "reports/report.md")
    return decision


def _compare_parquet(reference: Path, candidate: Path) -> tuple[bool, float]:
    left = pd.read_parquet(reference)
    right = pd.read_parquet(candidate)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False, float("inf")
    maximum = 0.0
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]):
            first = left[column].to_numpy(dtype=float)
            second = right[column].to_numpy(dtype=float)
            difference = np.abs(first - second)
            finite = difference[np.isfinite(difference)]
            maximum = max(maximum, float(finite.max()) if len(finite) else 0.0)
            if not np.allclose(first, second, rtol=0.0, atol=1e-12, equal_nan=True):
                return False, maximum
        elif not left[column].equals(right[column]):
            return False, float("inf")
    return True, maximum


def verify_rerun(reference: Path, candidate: Path) -> dict[str, Any]:
    ignored = {"exact_rerun_manifest.json"}
    reference_files = {
        path.name: path
        for path in reference.iterdir()
        if path.is_file() and path.name not in ignored
    }
    candidate_files = {
        path.name: path
        for path in candidate.iterdir()
        if path.is_file() and path.name not in ignored
    }
    if set(reference_files) != set(candidate_files):
        raise RuntimeError("blocked_reproducibility_or_audit_failure: artifact set differs")
    comparisons: list[dict[str, Any]] = []
    passed = True
    for name in sorted(reference_files):
        left, right = reference_files[name], candidate_files[name]
        left_hash, right_hash = _sha256(left), _sha256(right)
        byte_identical = left_hash == right_hash
        numeric_equal = byte_identical
        maximum_difference = 0.0
        if not byte_identical and left.suffix == ".parquet":
            numeric_equal, maximum_difference = _compare_parquet(left, right)
        item_passed = byte_identical or numeric_equal
        passed &= item_passed
        comparisons.append(
            {
                "artifact": name,
                "reference_sha256": left_hash,
                "candidate_sha256": right_hash,
                "byte_identical": byte_identical,
                "strict_numeric_equal": numeric_equal,
                "maximum_absolute_difference": maximum_difference,
                "passed": item_passed,
            }
        )
    manifest = {
        **SAFETY_FLAGS,
        "passed": passed,
        "comparison_count": len(comparisons),
        "all_scientific_artifacts_reproduced": passed,
        "comparisons": comparisons,
    }
    _write_json(reference / "exact_rerun_manifest.json", manifest)
    _write_json(candidate / "exact_rerun_manifest.json", manifest)
    if not passed:
        raise RuntimeError("blocked_reproducibility_or_audit_failure")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "artifacts/primary",
        help="Artifact output directory.",
    )
    parser.add_argument(
        "--verify-rerun",
        action="store_true",
        help="Compare --output to --reference without rerunning models.",
    )
    parser.add_argument("--reference", type=Path, help="Primary artifacts for exact comparison.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verify_rerun:
        if args.reference is None:
            raise SystemExit("--verify-rerun requires --reference")
        manifest = verify_rerun(args.reference.resolve(), args.output.resolve())
        print(json.dumps({"exact_rerun_passed": manifest["passed"]}, sort_keys=True))
        return
    decision = run(args.output.resolve())
    print(json.dumps({"decision": decision["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
